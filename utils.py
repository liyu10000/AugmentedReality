import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def get_frame(video_name, frame_dir):
    os.makedirs(frame_dir, exist_ok=True)
    vidcap = cv2.VideoCapture(video_name)
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite("{}/frame{}.png".format(frame_dir, count), image)     # save frame as PNG file      
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1


def parse_points3D(txt):
    """ Parse points3D.txt from sparse reconstruction.
        Return a numpy array of xyz coordinates, (n,3).
    """
    xyz = []
    with open(txt, 'r') as f:
        for line in f.readlines():
            if line.startswith('#'):
                continue
            tokens = line.strip().split()
            x = float(tokens[1])
            y = float(tokens[2])
            z = float(tokens[3])
            xyz.append([x, y, z])
    return np.array(xyz)

def parse_cameras(txt):
    """ Parse cameras.txt from sparse reconstruction.
        Assume a single camera model:
            SIMPLE_RADIAL params: f, cx, cy, k.
        Return a python dict containing camara params.
    """
    cameras = {}
    with open(txt, 'r') as f:
        for line in f.readlines():
            if line.startswith('#'):
                continue
            tokens = line.strip().split()
            cameras['CAMERA_ID'] = int(tokens[0])
            cameras['MODEL'] = tokens[1]
            cameras['WIDTH'] = int(tokens[2])
            cameras['HEIGHT'] = int(tokens[3])
            cameras['PARAMS'] = [float(tokens[i]) for i in range(4, len(tokens))]
    return cameras

def parse_images(txt):
    """ Parse images.txt from sparse reconstruction.
        Return a python dict containing camera pose for each image.
    """
    keys = ['IMAGE_ID', 'QW', 'QX', 'QY', 'QZ', 'TX', 'TY', 'TZ', 'CAMERA_ID', 'NAME']
    images = {key:[] for key in keys}
    readline = False
    with open(txt, 'r') as f:
        for line in f.readlines():
            if line.startswith('#') or not readline:
                readline = True
            else:
                readline = False
                tokens = line.strip().split()
                images['IMAGE_ID'].append(int(tokens[0]))
                images['QW'].append(float(tokens[1]))
                images['QX'].append(float(tokens[2]))
                images['QY'].append(float(tokens[3]))
                images['QZ'].append(float(tokens[4]))
                images['TX'].append(float(tokens[5]))
                images['TY'].append(float(tokens[6]))
                images['TZ'].append(float(tokens[7]))
                images['CAMERA_ID'].append(int(tokens[8]))
                images['NAME'].append(tokens[9])
    return images


def detect_outliers_1d(data, threshold):
    """ Detect outliers from a vector of data
    :param data: 1d vector
    :param threshold: determine if a point is outlier
    :return: ids of data points
    """
    mean = np.mean(data)
    std = np.std(data)
    ids = []
    for i, d in enumerate(data):
        z_score = (d - mean) / std 
        if abs(z_score) > threshold:
            ids.append(i)
    return np.array(ids)

def detect_outliers_3d(data, threshold):
    """ Detect outliers from 3d points cloud
    :param data: 3d numpy array, (n,3)
    :param threshold: determine if a point is outlier
    :return: ids of data points
    """
    idx = detect_outliers_1d(data[:, 0], threshold)
    idy = detect_outliers_1d(data[:, 1], threshold)
    idz = detect_outliers_1d(data[:, 2], threshold)
    ids = np.array(list(set(idx).union(set(idy), set(idz))))
    return ids

def remove_outliers(data, threshold):
    ids = detect_outliers_3d(data, threshold)
    data_outliers = data[ids, :]
    print('# outlier points:', data_outliers.shape)

    mask = np.ones(len(data), np.bool)
    mask[ids] = 0
    data_inliers = data[mask]
    print('# inlier points:', data_inliers.shape)
    
    # # run this in a cell to check
    # %matplotlib notebook 
    # plot3D(data_inliers, data_outliers, plot_plane=False)
    
    return data_inliers


def get_3D_box(dimx, dimy, dimz):
    """ Generate coordinates of 8 corners of the box
    """
    corners = np.array([[dimx/2, dimy/2, 0],
                        [-dimx/2, dimy/2, 0],
                        [-dimx/2, -dimy/2, 0],
                        [dimx/2, -dimy/2, 0],
                        [dimx/2, dimy/2, dimz],
                        [-dimx/2, dimy/2, dimz],
                        [-dimx/2, -dimy/2, dimz],
                        [dimx/2, -dimy/2, dimz]])
    return corners


def plot3D(inplane_points, outplane_points, plot_plane=False, model=None, plot_box=False, corners=None):
    fig = plt.figure(1, figsize=(8, 8))
    ax = fig.gca(projection='3d')

    # plot outplane points
    xo = outplane_points[:, 0]
    yo = outplane_points[:, 1]
    zo = outplane_points[:, 2]
    ax.scatter(xo, yo, zo, marker='.', color='gray')

    # plot inplane points
    xi = inplane_points[:, 0]
    yi = inplane_points[:, 1]
    zi = inplane_points[:, 2]
    zi=np.expand_dims(zi,axis=1)
    ax.scatter(xi, yi, zi, marker='o', color='red')

    # plot the surface
    if plot_plane:
        a, b, c, d = model
        xmin, xmax = np.min(xi), np.max(xi)
        xlen = xmax - xmin
        xmin -= xlen * 0.2
        xmax += xlen * 0.2
        ymin, ymax = np.min(yi), np.max(yi)
        ylen = ymax - ymin
        ymin -= ylen * 0.2
        ymax += ylen * 0.2
        x = np.linspace(xmin, xmax, 100)
        y = np.linspace(ymin, ymax, 100)
        X, Y = np.meshgrid(x, y)
        Z = -(a*X + b*Y + d) / c
        surf = ax.plot_surface(X, Y, Z)
        
    # plot the box
    if plot_box:
        combinations = [[0,1,2,3],  # bottom
                        [4,5,6,7],  # top
                        [0,1,5,4],  # right
                        [3,2,6,7],  # left
                        [0,4,7,3],  # front
                        [1,5,6,2]]  # back
        combinations = np.array(combinations)
        for comb in combinations:
            vertices = corners[comb, :]
            ax.add_collection3d(Poly3DCollection([vertices]))
        # ax.scatter3D(corners[:, 0], corners[:, 1], corners[:, 2], color='blue')
        
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()



if __name__ == '__main__':
    # video_name = './home/home.MOV'
    # frame_dir = './home/frames'
    # get_frame(video_name, frame_dir)

    # cameras_txt = './home/cameras.txt'
    # cameras = parse_cameras(cameras_txt)
    # print(cameras)

    images_txt = './home/images.txt'
    images = parse_images(images_txt)
    print(images)
    pprint({k:len(v) for k,v in images.items()})
