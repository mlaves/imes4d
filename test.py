"""Unittests for imes4d"""

import unittest
from imes4d.utils import translation_transformation, rigid_transformation, affine_transformation, ransac
from imes4d.utils import create_dummy_data, blend_volumes, _calc_new_bounds
from imes4d.PyrLK3D import PyrLK3D
import numpy as np


class ImesTest(unittest.TestCase):
    """Unittests for imes4d"""

    def test_pyr_lk_3d(self):
        shift = np.random.rand(3)-0.5 * 20
        trans = np.eye(4)
        trans[:3, 3] = shift
        a, b = create_dummy_data((128, 128, 128), trans=trans)

        prev_pts = PyrLK3D.harris_corner_3d(a)

        lk = PyrLK3D(a, b, prev_pts)
        flow, _, _ = lk.calc_flow()

        self.assertLess(np.linalg.norm(shift - np.mean(flow, axis=0)), 5e-2)

    def test_translation_transformation(self):
        # generate random transformation
        trans = np.eye(4)
        trans[:3, 3] = np.random.rand(3)

        # generate point set and apply transformation
        a = np.concatenate((np.random.rand(100, 3), np.ones((100, 1))), axis=1)
        b = np.matmul(trans, a.T).T

        # calculate transformation
        t = translation_transformation(a, b)

        # check for equality with tolerance
        self.assertTrue(np.allclose(t, trans), "t not trans")

    def test_rigid_transformation(self):
        # generate random transformation
        trans = np.eye(4)
        a = np.random.rand() * np.pi
        trans[0, 0] = np.cos(a)
        trans[0, 1] = -np.sin(a)
        trans[1, 0] = np.sin(a)
        trans[1, 1] = np.cos(a)
        trans[2, 2] = 1
        trans[:3, 3] = np.random.rand(3)

        # generate point set and apply transformation
        a = np.concatenate((np.random.rand(100, 3), np.ones((100, 1))), axis=1)
        b = np.matmul(trans, a.T).T

        # calculate transformation
        t = rigid_transformation(a, b)

        # check for equality with tolerance
        self.assertTrue(np.allclose(t, trans), "t not trans")

    def test_affine_transformation(self):
        # generate random data in homogeneous
        a = np.concatenate((np.random.rand(4, 3), np.ones((4, 1))), axis=1)

        # generate random transformation matrix
        trans = np.vstack((np.random.rand(3, 4), np.array([0, 0, 0, 1])))

        b = np.matmul(trans, a.T).T

        # calculate transformation
        t = affine_transformation(a, b)

        # check for equality with tolerance
        self.assertTrue(np.allclose(t, trans), "t not trans")

    def test_ransac_rigid(self):
        # generate random transformation
        trans = np.eye(4)
        a = np.random.rand() * np.pi
        trans[0, 0] = np.cos(a)
        trans[0, 1] = -np.sin(a)
        trans[1, 0] = np.sin(a)
        trans[1, 1] = np.cos(a)
        trans[2, 2] = 1
        trans[:3, 3] = np.random.rand(3)

        # generate point set and apply transformation
        a = np.concatenate((np.random.rand(100, 3), np.ones((100, 1))), axis=1)
        b = np.matmul(trans, a.T).T

        # add some outlier
        for _ in range(50):
            i = np.random.randint(0, b.shape[0])
            b[i] = b[i] + 10 * np.random.rand() + 10

        t, inlier_idx = ransac(a, b, 'rigid', inlier_threshold=0.25)

        self.assertTrue(np.allclose(t, trans), "t not trans")

    def test_blend_calc_bounds(self):
        shape = (128, 128, 128)

        shift = (np.random.rand(3)-0.5) * 20
        trans = np.eye(4)
        trans[:3, 3] = shift

        bounds = np.zeros((8, 3))
        bounds[0] = np.array([0, 0, 0])
        bounds[1] = np.array([0, 0, 128])
        bounds[2] = np.array([0, 128, 0])
        bounds[3] = np.array([0, 128, 128])
        bounds[4] = np.array([128, 0, 0])
        bounds[5] = np.array([128, 0, 128])
        bounds[6] = np.array([128, 128, 0])
        bounds[7] = np.array([128, 128, 128])

        new_bounds, new_shape = _calc_new_bounds(shape, trans)

        self.assertTrue(np.allclose(np.array(shape) + np.ceil(np.abs(shift)), np.array(new_shape)))

    def test_blend_blend(self):
        shift = np.random.randint(0, 10)
        trans = np.eye(4)
        trans[:3, 3] = np.ones(3) * shift
        a, b = create_dummy_data((128, 128, 128), trans=trans)

        stitched_predict = blend_volumes(a, b, trans)

        stitched_true = np.zeros(tuple([i + shift for i in a.shape]))
        stitched_true[:a.shape[0], :a.shape[1], :a.shape[2]] = a

        self.assertTrue(np.allclose(stitched_true, stitched_predict))


if __name__ == "__main__":
    unittest.main(verbosity=2)
