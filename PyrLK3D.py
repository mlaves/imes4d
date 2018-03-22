"""PyrLK3D module docstring."""

import numpy as np
from scipy.ndimage.filters import sobel, gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from skimage.feature.peak import peak_local_max
from numba import jit, float32, int32
from functools import partial
from multiprocessing import Pool, cpu_count


class PyrLK3D:
    """A class for performing pyramidal 3D Lucas-Kanade.
    """

    def __init__(self, prev_volume, next_volume, prev_points, init_p=None,
                 win_size=(5, 5, 5), levels=4, eps=1e-3, max_iterations=100):
        self._prevVol = prev_volume
        self._nextVol = next_volume
        self._prevPts = prev_points
        self._init_p = init_p if init_p else np.zeros(prev_points.shape, dtype=np.float32)
        self._winSize = win_size
        self._levels = levels
        self._eps = eps
        self._maxIter = max_iterations

    def calc_flow(self):
        """Start pyramidal calculation of 3D optical flow field."""
        return self._calc_flow_pyr(self._prevVol, self._nextVol, self._prevPts, self._init_p, self._levels)

    def _calc_flow_pyr(self, prev_vol, next_vol, prev_pts, init_p, level):
        if level == 1:
            return self._calc_flow(prev_vol, next_vol, prev_pts, init_p)
        else:
            # go down one level of pyramid
            # TODO: replace gauss and subsample with skimage.transform.pyramid_reduce when PyPI upgrades skimage to 0.14
            prev_vol2 = gaussian_filter(prev_vol, 1.5, mode='constant')[::2, ::2, ::2]
            next_vol2 = gaussian_filter(next_vol, 1.5, mode='constant')[::2, ::2, ::2]
            p, _, _ = self._calc_flow_pyr(prev_vol2,
                                          next_vol2,
                                          prev_pts * np.float32(0.5), init_p * np.float32(0.5), level - 1)

            # upscale flow guess and apply to current level
            p = p * np.float32(2)

            return self._calc_flow(prev_vol, next_vol, prev_pts, p)

    def _calc_flow(self, prev_vol, next_vol, prev_pts, next_pts):
        """
        Estimates voxel displacements between two subsequent volumes.

        :param prev_vol:
        :param next_vol:
        :param prev_pts:
        :param next_pts:
        :return:
        """

        assert len(prev_pts) == len(next_pts)

        # pre-init returns
        flow = np.zeros((len(prev_pts), 3), dtype=np.float32)
        err = [None] * len(prev_pts)
        it = [self._maxIter] * len(prev_pts)

        for i, (pPt, nPt) in enumerate(zip(prev_pts, next_pts)):
            flow[i], err[i], it[i] = self._calc_flow_pt(pPt, nPt, prev_vol, next_vol,
                                                        self._winSize, self._eps, self._maxIter)

        return flow, err, it

    def _do_not_use_this_calc_flow_parallel(self, prev_vol, next_vol, prev_pts, next_pts):
        """
        Estimates voxel displacements between two subsequent volumes.
        Do not use this function. As numpy's functions are already
        intrinsically parallel, this does not speed up anything (but I have to admit
        it's somehow elegant).

        :param prev_vol:
        :param next_vol:
        :param prev_pts:
        :param next_pts:
        :return:
        """

        import warnings

        warnings.warn("""Do not use this function. As numpy's functions are already
        intrinsically parallel, this does not speed up anything (but I have to admit
        it's somehow elegant).""", DeprecationWarning)

        assert len(prev_pts) == len(next_pts)

        # bind non-iterable parameters to _calc_flow_pt
        f = partial(PyrLK3D._calc_flow_pt, prev_vol=prev_vol, next_vol=next_vol,
                    win_size=self._winSize, min_eps=self._eps, max_iter=self._maxIter)

        # use starmap from multiprocessing to fire up all cores
        with Pool(cpu_count()) as p:
            ret = list(zip(*p.starmap(f, list(zip(prev_pts, next_pts)))))

        return np.array(ret[0]), ret[1], ret[2]

    @staticmethod
    def _calc_flow_pt(prev_pt, init_p, prev_vol, next_vol, win_size, min_eps, max_iter):

        # init displacement vector for point
        p = init_p
        warped = None
        n_iter = max_iter

        # cut out input volume and template
        template = PyrLK3D.cut_volume(prev_vol, prev_pt, win_size)

        for i in range(max_iter):
            # warp point
            prev_pt_w = PyrLK3D.warp(prev_pt, p)

            # cut out template at warped point
            warped = PyrLK3D.cut_volume(next_vol, prev_pt_w, win_size)

            # compute image derivatives
            vx, vy, vz, vt = PyrLK3D.compute_derivatives(template, warped)

            # calculate gradient image
            nabla_next = np.array([vx.flatten(), vy.flatten(), vz.flatten()]).T

            # calculate Jacobian of warp (identity for translations)
            jacobi = np.eye(3, dtype=np.float32)

            # calculate steepest descent image
            steepest = np.matmul(nabla_next, jacobi)

            # calculate hessian approximation
            hessian = np.matmul(steepest.T, steepest)

            # calculate steepest descent parameter update
            p_update = np.matmul(vt.flatten().T, steepest)
            d_p = np.matmul(np.linalg.pinv(hessian), p_update.T)

            p = p + d_p

            # calculate epsilon of parameter update
            eps = np.linalg.norm(d_p)

            # early quit loop if convergence is reached
            if eps <= min_eps:
                n_iter = i
                break

        return p, np.sum(np.abs(template-warped)), n_iter

    @staticmethod
    def warp(pt, param):
        return pt + param

    @staticmethod
    def cut_volume(volume, pt, win_size):
        """
        Allows interpolation of sub-volumes at sub-pixel location.

        :param volume: volume from where to cut out a sub-volume (numpy.ndarray)
        :param pt: point location where to cut out (tuple or numpy.ndarray)
        :param win_size: window size of new sub-volume (tuple or numpy.ndarray)
        :return: sub-volume from volume at pt with win_size dimensions
        :raise: AssertionError if any parameter has wrong data type
        """

        assert isinstance(volume, np.ndarray)
        assert isinstance(pt, tuple) or isinstance(pt, np.ndarray)
        assert isinstance(win_size, tuple) or isinstance(win_size, np.ndarray)

        interp = RegularGridInterpolator((np.arange(volume.shape[0]),
                                          np.arange(volume.shape[1]),
                                          np.arange(volume.shape[2])),
                                         volume, bounds_error=False, fill_value=None)  # caution: default fill is nan!

        # interpolate
        i, j, k = np.meshgrid(np.linspace(pt[0] - win_size[0], pt[0] + win_size[0], 2 * win_size[0] + 1),
                              np.linspace(pt[1] - win_size[1], pt[1] + win_size[1], 2 * win_size[0] + 1),
                              np.linspace(pt[2] - win_size[2], pt[2] + win_size[2], 2 * win_size[0] + 1),
                              indexing='ij')

        return interp((i, j, k)).astype(np.float32)

    @staticmethod
    def compute_derivatives(prev_vol, next_vol):
        """Computes 3D derivatives between two 3D volume images."""

        vx = sobel(next_vol, axis=0, mode='constant')
        vy = sobel(next_vol, axis=1, mode='constant')
        vz = sobel(next_vol, axis=2, mode='constant')

        vt = prev_vol - next_vol

        return vx, vy, vz, vt

    @staticmethod
    @jit(float32[:, :, :](float32[:, :, :], int32, float32, float32))
    def harris_corner_3d(volume, min_distance=10, threshold=0.1, eps=1e-6):
        """Finds corners in volume by extending harris corner detection to 3D.
        Special thanks to scikit-image!"""

        # compute harris response
        # derivatives
        v_x = sobel(volume, axis=0, mode='constant')
        v_y = sobel(volume, axis=1, mode='constant')
        v_z = sobel(volume, axis=2, mode='constant')

        w_xx = gaussian_filter(v_x * v_x, 1.5, mode='constant')
        w_xy = gaussian_filter(v_x * v_y, 1.5, mode='constant')
        w_xz = gaussian_filter(v_x * v_z, 1.5, mode='constant')
        w_yy = gaussian_filter(v_y * v_y, 1.5, mode='constant')
        w_yz = gaussian_filter(v_y * v_z, 1.5, mode='constant')
        w_zz = gaussian_filter(v_z * v_z, 1.5, mode='constant')

        # determinant and trace
        w_det = w_xx * w_yy * w_zz + 2 * w_xy * w_yz * w_xz - w_yy * w_xz ** 2 - w_zz * w_xy ** 2 - w_xx * w_yz ** 2
        w_tr = w_xx + w_yy + w_zz

        # Alison Noble, "Descriptions of Image Surfaces", PhD thesis (1989)
        harris_vol = w_det / (w_tr + eps)

        # calculate final corners by local maximum
        coordinates = peak_local_max(harris_vol, min_distance=min_distance, threshold_rel=threshold)

        return coordinates.astype(np.float32)
