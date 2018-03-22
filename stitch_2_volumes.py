#!/usr/bin/env python

import numpy as np
from imes4d.PyrLK3D import PyrLK3D
from imes4d.utils import Timer, ransac, blend_volumes, create_dummy_data
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":

    A = np.load('0.npz')
    B = np.load('1.npz')
    A = A[A.files[0]][::2, ::2, ::2].astype(np.float32)
    B = B[B.files[0]][::2, ::2, ::2].astype(np.float32)

    with Timer('harris'):
        prev_pts = PyrLK3D.harris_corner_3d(A)

    lk = PyrLK3D(A, B, prev_pts, win_size=(5, 5, 5), levels=5, eps=1e-3, max_iterations=200)

    with Timer('pyr_lk'):
        flow, err, it = lk.calc_flow()

    print('mean_eps =', np.mean(err))
    print('mean_it =', np.mean(it))
    print('')

    # find best 25 % matches
    mean_err = np.mean(np.sort(err)[:int(len(err) / 4)])
    best_flow = np.array([i for i, e in zip(flow, err) if e < mean_err])
    best_prev = np.array([i for i, e in zip(prev_pts, err) if e < mean_err])

    fig3d = plt.figure('p')
    ax3 = Axes3D(fig3d)
    ax3.quiver(best_prev[:, 0], best_prev[:, 1], best_prev[:, 2],
               best_flow[:, 0], best_flow[:, 1], best_flow[:, 2])
    ax3.set_xlim([0, A.shape[0]])
    ax3.set_ylim([0, A.shape[1]])
    ax3.set_zlim([0, A.shape[2]])
    plt.savefig('vectors.pdf', dpi=300, bbox_inches='tight')

    # stitch volume
    t, inlier_idx = ransac(best_prev, (best_prev+best_flow), 'rigid', inlier_threshold=0.5, ransac_it=100)
    print('inliers:', len(inlier_idx))
    print(t)
    stitched = blend_volumes(A, B, t)

    plt.figure('A')
    plt.imshow(A[40])
    plt.figure('B')
    plt.imshow(B[40])
    plt.figure('stitched')
    plt.imshow(stitched[40+8])

    np.savez_compressed('stitched.npz', stitched)

    plt.show()
