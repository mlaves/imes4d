import time
import numpy as np
from scipy.ndimage import affine_transform


class Timer(object):

    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.t_start = time.time()

    def __exit__(self, t, value, traceback):
        if self.name:
            print('[%s]' % self.name),
        print('Elapsed: %s s' % (time.time() - self.t_start))


def slices_to_npz(glob_pattern: str, out_file: str, compressed: bool=True):
    """Takes a glob pattern to a set of volume slices and composes an npz file."""

    from skimage.io import ImageCollection, imread

    images_coll = ImageCollection(glob_pattern.replace('"', '').replace("'", ''),
                                  load_func=lambda x: imread(x, as_grey=True))

    if compressed:
        np.savez_compressed(out_file.replace('"', '').replace("'", ''), images_coll)
    else:
        np.savez(out_file.replace('"', '').replace("'", ''), images_coll)


def create_dummy_data(shape=(256, 256, 256), trans=np.eye(4)):
    """
    Creates two 3D arrays with a shifted cube as dummy data. Raises TypeError if shape is not correct.

    :param shape: shape of creating dummy data (tuple of ints)
    :param trans: shift between dummy data (real or tuple)
    :return: (a, b, trans) tuple containing both volumes a and b and the 4x4 transformation matrix as numpy array
    """

    from scipy.ndimage import affine_transform

    assert isinstance(shape, tuple)
    assert len(shape) == 3

    for i in shape:
        assert isinstance(i, int)
    a = np.zeros((128, 128, 128), dtype=np.float32)

    start = np.array(list(map(lambda x: int(np.floor(x / 4)), shape)), dtype=np.int)
    end = np.array(list(map(lambda x: int(np.floor(3*x / 4)), shape)), dtype=np.int)
    a[start[0]:end[0], start[1]:end[1], start[2]:end[2]] = np.ones(end - start)

    b = affine_transform(a, np.linalg.inv(trans))

    return a, b


def ransac(a, b, model: str ='rigid', inlier_threshold: float = 1.0, ransac_it: int = 100):
    """Estimates parameters of given model by applying RANSAC on corresponding point sets A and B
    (preserves handedness).

    :param a: nx4 array of points
    :param b: nx4 array of points
    :param model: Specify the model for RANSAC. Can be 'translation', 'rigid' or 'affine'
    :param inlier_threshold: Specify the inlier threshold in RANSAC process
    :param ransac_it: number of ransac iterations
    :return: corresponding transformation matrix (None if no transformation was found)
    :raise: NotImplementedError for models which are not implemented yet"""

    max_ransac_it = ransac_it
    num_samples = 0
    estimate_transformation = None

    assert a.shape == b.shape

    if a.shape[1] == 3:
        a = np.concatenate((a, np.ones((a.shape[0], 1))), axis=1)
        b = np.concatenate((b, np.ones((a.shape[0], 1))), axis=1)

    if model == 'translation':
        num_samples = 1
        estimate_transformation = translation_transformation
    elif model == 'rigid':
        num_samples = 4
        estimate_transformation = rigid_transformation
    elif model == 'affine':
        num_samples = 4
        estimate_transformation = affine_transformation

    assert a.shape[0] >= num_samples

    best_inlier = 0
    best_inlier_idx = []
    best_t = None

    for _ in range(max_ransac_it):
        # random sample data for generating hypothetical inliers
        hyp_inliers_idx = np.random.choice(a.shape[0], size=num_samples, replace=False)
        hyp_inliers_a = np.array([a[i] for i in hyp_inliers_idx])
        hyp_inliers_b = np.array([b[i] for i in hyp_inliers_idx])

        # calculate transformation based on hypothetical inliers and selected model
        try:
            t = estimate_transformation(hyp_inliers_a, hyp_inliers_b)
        except AssertionError:
            t = np.eye(4)

        # calculate consensus set for this transformation
        b_ = np.matmul(t, a.T).T
        dists = [np.linalg.norm((x - y)[:3]) for x, y in zip(b_, b)]
        inlier_idx = [i for i, x in enumerate(dists) if x < inlier_threshold]

        # save better consensus set
        if len(inlier_idx) > best_inlier:
            best_inlier = len(inlier_idx)
            best_inlier_idx = inlier_idx
            best_t = t

    # recalculate transformation with best consensus set
    if len(best_inlier_idx) > 0:
        consensus_set_a = np.array([a[i] for i in best_inlier_idx])
        consensus_set_b = np.array([b[i] for i in best_inlier_idx])
        try:
            best_t = estimate_transformation(consensus_set_a, consensus_set_b)
        except AssertionError:
            pass

    return best_t, best_inlier_idx


def translation_transformation(a, b):
    """Finds rigid transformation between two corresponding point sets.

    :param a: nx4 array of points in homogeneous coordinates (with n > 0)
    :param b: nx4 array of points (with n > 0)
    :returns: 4x4 homogeneous transformation matrix
    """

    assert a.shape == b.shape
    assert a.shape[0] > 0
    assert a.shape[1] == 4

    centroid_a = np.mean(a, axis=0)
    centroid_b = np.mean(b, axis=0)

    t = np.eye(4)
    t[:3, 3] = (centroid_b-centroid_a)[:3]

    return t


def rigid_transformation(a, b):
    """Finds rigid transformation between two corresponding point sets.

    :param a: nx4 array of points in homogeneous coordinates (with n > 3)
    :param b: nx4 array of points (with n > 3)
    :returns: 4x4 homogeneous transformation matrix
    """

    assert a.shape == b.shape
    assert a.shape[0] > 3
    assert a.shape[1] == 4

    centroid_a = np.mean(a, axis=0)
    centroid_b = np.mean(b, axis=0)

    u, s, vh = np.linalg.svd(np.matmul((a - centroid_a).T, (b - centroid_b)))
    d = (np.linalg.det(u) * np.linalg.det(vh)) < 0.0

    if d:
        s[-1] = -s[-1]
        u[:, -1] = -u[:, -1]

    trans = np.eye(4)
    r = np.matmul(u, vh).T
    trans[:3, :3] = r[:3, :3]
    t = centroid_b - np.matmul(r, centroid_a)
    trans[:3, 3] = t[:3]

    return trans


def affine_transformation(a, b):
    """Finds rigid transformation between two corresponding point sets.

    :param a: nx4 array of points in homogeneous coordinates (with n > 4)
    :param b: nx4 array of points (with n > 4)
    :returns: 4x4 homogeneous transformation matrix
    """

    assert a.shape == b.shape
    assert a.shape[0] > 3
    assert a.shape[1] == 4

    a = a.T
    b = b.T

    return np.matmul(b, np.linalg.pinv(a))


def blend_volumes(first_volume, second_volume, transformation):
    """
    Blending of two adjacent volumes with given transformation.

    :param first_volume: first volume as 3D numpy array
    :param second_volume: second volume as 3D numpy array
    :param transformation: 4x4 transformation matrix from first to second volume as numpy array
    :return: stitched and blended volume containing both first and second volume
    :raise: AssertionError if dimensions of volumes or transformation matrix differ
    """

    # assert correct shape properties
    assert len(first_volume.shape) == 3
    assert len(second_volume.shape) == 3
    assert first_volume.shape == second_volume.shape
    assert transformation.shape == (4, 4)

    # transformation should be from A to B, thus invert it to stitch B to A
    transformation = np.linalg.inv(transformation)

    # generate empty volume for stitched data
    new_bounds, new_shape = _calc_new_bounds(first_volume.shape, transformation)
    stitched_vol = np.zeros(new_shape)
    first_vol_padded = np.zeros(new_shape)

    # copy first volume in bigger array to have matching dimensions
    x_start = np.abs(np.min(new_bounds[:, 0]))
    y_start = np.abs(np.min(new_bounds[:, 1]))
    z_start = np.abs(np.min(new_bounds[:, 2]))

    first_vol_padded[x_start:x_start+first_volume.shape[0],
                     y_start:y_start+first_volume.shape[1],
                     z_start:z_start+first_volume.shape[2]] = first_volume

    # transform second volume
    stitched_vol[:second_volume.shape[0],
                 :second_volume.shape[1],
                 :second_volume.shape[2]] = second_volume
    shift = np.eye(4)
    shift[:3, 3] = np.array([x_start, y_start, z_start])
    stitched_vol = affine_transform(stitched_vol, np.linalg.inv(np.matmul(shift, transformation)))

    # first, get indices of overlapping voxel
    overlap = np.nonzero(np.logical_and(stitched_vol > 1e-3, first_vol_padded > 1e-3))

    # add first and transformed second volume
    stitched_vol += first_vol_padded

    # divide overlapping voxel by 2
    stitched_vol[overlap] /= 2
    stitched_vol = np.clip(stitched_vol, 0.0, 1.0)

    return stitched_vol


def blend_collection(volumes, transformations):
    """
    Blending of multiple volumes with given transformations between each other.

    :param volumes: collection of n volumes
    :param transformations: collection of n-1 transformations between adjacent volumes
    :return: stitched and blended volume containing all volume
    :raise: AssertionError if dimensions of volumes or transformation matrix differ
    """

    # assert correct shape properties
    assert len(volumes) > 1
    assert len(volumes) == len(transformations) + 1
    for vol in volumes:
        assert len(vol.shape) == 3
    for i in range(len(volumes)-1):
        assert volumes[i].shape == volumes[i+1].shape
    for trans in transformations:
        assert trans.shape == (4, 4)

    # calculate total transformation and max bounds
    total_t = np.eye(4)
    cum_t = [total_t]  # cumulative transformations
    new_bounds = []
    new_shapes = []
    for t in transformations:
        total_t = np.matmul(np.linalg.inv(t), total_t)
        cum_t.append(total_t)

        n_bounds, n_shape = _calc_new_bounds(volumes[0].shape, total_t)
        new_bounds.append(n_bounds)
        new_shapes.append(n_shape)

    # find total min and max shape
    new_shape = list(new_shapes[0])
    for ns in new_shapes:
        new_shape[0] = ns[0] if ns[0] > new_shape[0] else new_shape[0]
        new_shape[1] = ns[1] if ns[1] > new_shape[1] else new_shape[1]
        new_shape[2] = ns[2] if ns[2] > new_shape[2] else new_shape[2]

    new_shape = tuple(new_shape)

    # find total new bounds
    new_bounds = np.array(new_bounds)
    x_start = np.abs(np.min(new_bounds[:, :, 0]))
    y_start = np.abs(np.min(new_bounds[:, :, 1]))
    z_start = np.abs(np.min(new_bounds[:, :, 2]))

    # shift everything to prevent leaving bounds
    shift = np.eye(4)
    shift[:3, 3] = np.array([x_start, y_start, z_start])
    cum_t = map(lambda x: np.matmul(shift, x), cum_t)

    # generate empty volume for stitched data
    stitched_vol = np.zeros(new_shape)

    for vol, t in zip(volumes, cum_t):
        current_vol = np.zeros(new_shape)

        current_vol[:vol.shape[0],
                    :vol.shape[1],
                    :vol.shape[2]] = vol

        current_vol = affine_transform(current_vol, np.linalg.inv(t))

        # get indices of overlapping voxel
        overlap = np.nonzero(np.logical_and(stitched_vol > 1e-3, current_vol > 1e-3))

        # add total and transformed current volume
        stitched_vol += current_vol

        # divide overlapping voxel by 2
        stitched_vol[overlap] /= 2
        stitched_vol = np.clip(stitched_vol, 0.0, 1.0)

    return stitched_vol


def _calc_new_bounds(shape, trans):
    # find max boundaries for new volume
    new_bounds = np.zeros((8, 3), dtype=np.float32)
    new_bounds[0] = np.array([0, 0, 0], dtype=np.float32)
    new_bounds[1] = np.array([0, 0, shape[2]], dtype=np.float32)
    new_bounds[2] = np.array([0, shape[1], 0], dtype=np.float32)
    new_bounds[3] = np.array([0, shape[1], shape[2]], dtype=np.float32)
    new_bounds[4] = np.array([shape[0], 0, 0], dtype=np.float32)
    new_bounds[5] = np.array([shape[0], 0, shape[2]], dtype=np.float32)
    new_bounds[6] = np.array([shape[0], shape[1], 0], dtype=np.float32)
    new_bounds[7] = np.array([shape[0], shape[1], shape[2]], dtype=np.float32)
    new_bounds = np.concatenate((new_bounds, np.ones((8, 1))), axis=1)
    new_bounds = np.matmul(trans, new_bounds.T).T

    new_bounds[0, 0] = np.min([0, np.floor(new_bounds[0, 0])])
    new_bounds[0, 1] = np.min([0, np.floor(new_bounds[0, 1])])
    new_bounds[0, 2] = np.min([0, np.floor(new_bounds[0, 2])])

    new_bounds[1, 0] = np.min([0, np.floor(new_bounds[1, 0])])
    new_bounds[1, 1] = np.min([0, np.floor(new_bounds[1, 1])])
    new_bounds[1, 2] = np.max([shape[2], np.ceil(new_bounds[1, 2])])

    new_bounds[2, 0] = np.min([0, np.floor(new_bounds[2, 0])])
    new_bounds[2, 1] = np.max([shape[1], np.ceil(new_bounds[2, 1])])
    new_bounds[2, 2] = np.min([0, np.floor(new_bounds[2, 2])])

    new_bounds[3, 0] = np.min([0, np.floor(new_bounds[2, 0])])
    new_bounds[3, 1] = np.max([shape[1], np.ceil(new_bounds[2, 1])])
    new_bounds[3, 2] = np.max([shape[2], np.ceil(new_bounds[2, 2])])

    new_bounds[4, 0] = np.max([shape[0], np.ceil(new_bounds[4, 0])])
    new_bounds[4, 1] = np.min([0, np.floor(new_bounds[4, 1])])
    new_bounds[4, 2] = np.min([0, np.floor(new_bounds[4, 2])])

    new_bounds[5, 0] = np.max([shape[0], np.ceil(new_bounds[5, 0])])
    new_bounds[5, 1] = np.min([0, np.floor(new_bounds[5, 1])])
    new_bounds[5, 2] = np.max([shape[2], np.ceil(new_bounds[5, 2])])

    new_bounds[6, 0] = np.max([shape[0], np.ceil(new_bounds[6, 0])])
    new_bounds[6, 1] = np.max([shape[1], np.ceil(new_bounds[6, 1])])
    new_bounds[6, 2] = np.min([0, np.floor(new_bounds[6, 2])])

    new_bounds[7, 0] = np.max([shape[0], np.ceil(new_bounds[7, 0])])
    new_bounds[7, 1] = np.max([shape[1], np.ceil(new_bounds[7, 1])])
    new_bounds[7, 2] = np.max([shape[2], np.ceil(new_bounds[7, 2])])

    new_shape = (np.max(new_bounds[:, 0]) - np.min(new_bounds[:, 0]),
                 np.max(new_bounds[:, 1]) - np.min(new_bounds[:, 1]),
                 np.max(new_bounds[:, 2]) - np.min(new_bounds[:, 2]))

    new_shape = tuple([int(i) for i in new_shape])

    return new_bounds[:, :3].astype(np.int), new_shape
