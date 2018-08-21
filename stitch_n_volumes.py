#!/usr/bin/env python

import numpy as np
from imes4d.PyrLK3D import PyrLK3D
from imes4d.utils import Timer, ransac, blend_collection


if __name__ == "__main__":

    N = 7
    scale = 2
    transformations = []
    data_prefix = 'data/sb_'

    a = np.load(data_prefix + '0.npz')
    a = a[a.files[0]][::scale, ::scale, ::scale].astype(np.float32)

    # first, calculate flow and transformation matrices
    for n in range(1, N):
        # load two adjacent volumes
        with Timer('loading ' + str(n)):
            b = np.load(data_prefix + str(n) + '.npz')
            b = b[b.files[0]][::scale, ::scale, ::scale].astype(np.float32)

        with Timer('harris ' + str(n)):
            prev_pts = PyrLK3D.harris_corner_3d(a)

        lk = PyrLK3D(a, b, prev_pts, win_size=(5, 5, 5), levels=5, eps=1e-3, max_iterations=200)

        with Timer('pyr_lk ' + str(n)):
            flow, err, it = lk.calc_flow()

        # find best 50 % matches
        mean_err = np.mean(np.sort(err)[:int(len(err) / 2)])
        best_flow = np.array([i for i, e in zip(flow, err) if e < mean_err])
        best_prev = np.array([i for i, e in zip(prev_pts, err) if e < mean_err])

        print('best_flow.shape =', best_flow.shape)

        # find transformations, iteratively increase inlier threshold
        with Timer('ransac ' + str(n)):
            i_t = 0.5
            inlier_idx = []
            while len(inlier_idx) < 10:
                t, inlier_idx = ransac(best_prev, (best_prev+best_flow), 'rigid', inlier_threshold=i_t, ransac_it=1000)
                i_t = i_t + 0.5

        print('inliers:', len(inlier_idx))
        print(t)
        transformations.append(t)

        a = b

    # stitch volumes
    with Timer('blending'):
        volumes = []
        for n in range(N):
            vol = np.load(data_prefix + str(n)+'.npz')
            vol = vol[vol.files[0]][::scale, ::scale, ::scale].astype(np.float32)
            volumes.append(vol)

        stitched = blend_collection(volumes, transformations)

    np.savez_compressed('stitched_total.npz', stitched)
