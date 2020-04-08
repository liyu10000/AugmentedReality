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

def camera2film(XYZ, f):
    X, Y, Z = XYZ[:, 0], XYZ[:, 1], XYZ[:, 2]
    x = f * X / Z
    y = f * Y / Z
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

def forward_projection(data, cameras, images):
    f, cx, cy, k = cameras['PARAMS']
    IMAGE_ID = images['IMAGE_ID']
    NAME = images['NAME']
    QW = images['QW']
    QX = images['QX']
    QY = images['QY']
    QZ = images['QZ']
    TX = images['TX']
    TY = images['TY']
    TZ = images['TZ']
    for name, qw, qx, qy, qz, tx, ty, tz in zip(NAME, QW, QX, QY, QZ, TX, TY, TZ):
        qvec = [qw, qx, qy, qz]
        R = qvec2rotmat(qvec)
        T = np.array([tx, ty, tz])

        XYZ = world2camera(data, R, T)
        xy = camera2film(XYZ, f)
        uv = film2pixel(xy, cx, cy)
        print(np.min(uv, axis=0))
        print(np.mean(uv, axis=0))
        print(np.max(uv, axis=0))
        
        break

def backward_projection():
    pass



# def Distortion(k, u, v):
#     u2 = u * u
#     v2 = v * v
#     r2 = u2 + v2
#     radial = k * r2
#     du = u * radial
#     dv = v * radial
#     return du, dv

# def WorldToImage(params, u, v):
#     f, c1, c2, k = params

#     # Distortion
#     du, dv = Distortion(k, u, v)
#     x = u + du
#     y = v + dv

#     # Transform to image coordinates
#     x = f * x + c1
#     y = f * y + c2
#     return x, y

# def ImageToWorld(params, x, y, u, v):
#   f, c1, c2, k = params

#   # Lift points to normalized plane
#   u = (x - c1) / f
#   v = (y - c2) / f

#   IterativeUndistortion(k, u, v)



if __name__ == '__main__':
    params = [1855.42, 960, 540, 0.00782192]
    X, Y, Z = -1.34451987, 1.0912873, 8.02443223
    u = X / Z
    v = Y / Z
    x, y = WorldToImage(params, u, v)
    print(x, y)