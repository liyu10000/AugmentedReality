"""
Simple Radial Camera Model
Referred to code in: https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
Referred to code in: https://github.com/colmap/colmap/blob/master/src/base/camera_models.h
"""

import numpy as np


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def _world2camera(point, R):
    return np.dot(R, point)

def world2camera(UVW, R, T):
    # pad UVW to UVW1
    UVW1 = np.ones((len(UVW), 4))
    UVW1[:, :3] = UVW
    # put R and T in a 4x4 matrix
    RT = np.zeros((4, 4))
    RT[:3, :3] = R
    RT[:3, 3] = T
    RT[3, 3] = 1
    # apply matrix multiplication on each row of UVW
    XYZ1 = np.apply_along_axis(_world2camera, 1, UVW1, RT)
    XYZ = XYZ1[:, :3]
    return XYZ

def distortion(k, x, y):
    x2 = x ** 2
    y2 = y ** 2
    r2 = x2 + y2
    radial = k * r2
    dx = np.multiply(x, radial)
    dy = np.multiply(y, radial)
    return dx, dy

def camera2film(XYZ, f, k):
    X, Y, Z = XYZ[:, 0], XYZ[:, 1], XYZ[:, 2]
    x = X / Z
    y = Y / Z
    dx, dy = distortion(k, x, y)
    x = f * (x + dx)
    y = f * (y + dy)
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    return np.hstack([x, y])

def film2pixel(xy, cx, cy):
    x, y = xy[:, 0], xy[:, 1]
    u = x + cx
    v = y + cy
    u = u.reshape(-1, 1)
    v = v.reshape(-1, 1)
    return np.hstack([u, v])

def calc_depth(point3D, R, T):
    point3D_h = np.array([*point3D, 1])
    RT = np.hstack([R, T])
    proj_z = np.dot(RT[2, :], point3D_h)
    return proj_z * np.linalg.norm(RT[:, 2])

def forward_projection(data, camera_params, image_pose, calc_depth=None):
    f, cx, cy, k = camera_params['PARAMS']
    QW = image_pose['QW']
    QX = image_pose['QX']
    QY = image_pose['QY']
    QZ = image_pose['QZ']
    TX = image_pose['TX']
    TY = image_pose['TY']
    TZ = image_pose['TZ']
    qvec = [QW, QX, QY, QZ]
    R = qvec2rotmat(qvec)
    T = np.array([TX, TY, TZ])
    XYZ = world2camera(data, R, T)
    xy = camera2film(XYZ, f, k)
    uv = film2pixel(xy, cx, cy)
    uv = uv.astype(int)
    depths = []
    if calc_depth is not None:
        T = T.reshape(-1, 1)
        for d in data:
            depth = calc_depth(d, R, T)
            depths.append(depth)
    depths = np.array(depths)
    return uv, depths

def order_by_depth(combinations, depths):
    face_depths = []
    for comb in combinations:
        ds = depths[comb]
        face_depths.append(np.min(ds))
    new_combinations = [comb for _,comb in sorted(zip(face_depths, combinations), key=lambda pair: pair[0], reverse=True)]
    return np.array(new_combinations)
