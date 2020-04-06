import os
import numpy as np


def translate(data, dist):
    return data + dist

def get_vector_normal(model):
    """ Compute the vector normal of plane: ax + by + cz + d = 0
    :param model: a, b, c, d
    :return: normalized vector normal
    """
    normal = np.zeros(3)
    normal[:] = model[:3]
    norm = np.linalg.norm(normal)
    normal /= norm
    return normal

def get_rotation_matrix(a, b):
    """ Compute the rotation matrix from vector a to b
    Referenced the algorithm in this link: https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    :param a: normalized, (3,)
    :param b: normalized, (3,)
    :return: R, such that Ra = b
    """
    # find axis and angle using cross product and dot product
    v = np.cross(a, b)    # axis
    s = np.linalg.norm(v) # sine of angle
    c = np.dot(a, b)      # cosine of angle
    # compute skew-symmetric cross-product matrix of v
    v1, v2, v3 = v
    vx = np.array([[0, -v3, v2],
                   [v3, 0, -v1],
                   [-v2, v1, 0]])
    # compute rotation matrix
    R = np.identity(3) + vx + np.dot(vx, vx) * (1 / (1 + c))
    return R

def rotate_point(point, R):
    return np.dot(R, point)

def rotate(data, R):
    return np.apply_along_axis(rotate_point, 1, data, R)