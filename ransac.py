import os
import random
import numpy as np


def fit_plane(data):
    (rows, cols) = data.shape
    G = np.ones((rows, 3))
    G[:, 0] = data[:, 0]  #X
    G[:, 1] = data[:, 1]  #Y
    Z = data[:, 2]
    (a, b, c), resid, rank, s = np.linalg.lstsq(G, Z, rcond=None)
    x = np.array([a, b, -1, c])
    norm = np.linalg.norm(x)
    x /= norm
    return x

def is_inlier(model, point, threshold_inlier):
    point_withone = np.array([*point, 1])
    dist = abs(np.sum(np.multiply(model, point_withone)))
    return dist < threshold_inlier

def get_inplane_point_idx(model, data, threshold_inlier):
    point_idx = []
    for i, point in enumerate(data):
        if is_inlier(model, point, threshold_inlier):
            point_idx.append(i)
    point_idx = np.array(point_idx)
    return point_idx

def ransac(data, sample_size, num_iters, threshold_inlier, num_points):
    """ RANSAC algorithm to find a 3d plane
    :param data: np array, Nx3
    :param sample_size: number of points at fitting
    :param num_iters: number of iterations
    :param threshold_inlier: threshold of point-to-plane distance
    :param num_points: minimum number of points a plane should include
    :return: best model, in the form (a, b, c, d) such that ax + by + cz + d = 0
             best count, number of points of the dominant plane
    """
    best_model = None
    best_count = num_points
    for i in range(num_iters):
        # randomly choose sample points from data
        sample_idx = np.random.permutation(np.arange(data.shape[0]))
        sample = data[sample_idx[:sample_size], :]
        sample_left = data[sample_idx[sample_size:], :]
        # fit a plane
        model = fit_plane(sample)
        
        # get number of points in plane
        point_idx = get_inplane_point_idx(model, sample_left, threshold_inlier)
        count = point_idx.shape[0]
        print('iter {}: model param {}, in plane point count {}'.format(i, model, count))

        # check if count > num_points
        if count > num_points:
            model = fit_plane(np.vstack([sample, sample_left[point_idx, :]]))
            if count > best_count:
                best_count = count
                best_model = model

    return best_model, best_count


if __name__ == '__main__':
    from utils import parse_points3D

    data = parse_points3D('./home/points3D')

    sample_size = 3
    num_iters = 20
    threshold_inlier = 0.01
    num_points = 1000
    best_model, best_count = ransac(data, sample_size, num_iters, threshold_inlier, num_points)
    print('best fit:', best_model, best_count)