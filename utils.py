import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
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

def gen_video(frame_dir, video_name):
    names = [name for name in os.listdir(frame_dir) if name.endswith('png')]
    names = sorted(names, key=lambda name:int(name[5:].split('.')[0]))
    frame = cv2.imread(os.path.join(frame_dir, names[0]))
    height, width, _ = frame.shape
    video = cv2.VideoWriter(video_name, fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=30, frameSize=(width, height))  
    for name in names:
        frame = cv2.imread(os.path.join(frame_dir, name))
        video.write(frame)  
    video.release()


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
            CAMERA_ID = int(tokens[0])
            cameras[CAMERA_ID] = {}
            cameras[CAMERA_ID]['MODEL'] = tokens[1]
            cameras[CAMERA_ID]['WIDTH'] = int(tokens[2])
            cameras[CAMERA_ID]['HEIGHT'] = int(tokens[3])
            cameras[CAMERA_ID]['PARAMS'] = [float(tokens[i]) for i in range(4, len(tokens))]
    return cameras

def parse_images(txt):
    """ Parse images.txt from sparse reconstruction.
        Return a python dict containing camera pose for each image.
    """
    keys = ['QW', 'QX', 'QY', 'QZ', 'TX', 'TY', 'TZ', 'CAMERA_ID', 'NAME']
    images = {}
    readline = False
    with open(txt, 'r') as f:
        for line in f.readlines():
            if line.startswith('#') or not readline:
                readline = True
            else:
                readline = False
                tokens = line.strip().split()
                IMAGE_ID = int(tokens[0])
                images[IMAGE_ID] = {key:None for key in keys}
                images[IMAGE_ID]['QW'] = float(tokens[1])
                images[IMAGE_ID]['QX'] = float(tokens[2])
                images[IMAGE_ID]['QY'] = float(tokens[3])
                images[IMAGE_ID]['QZ'] = float(tokens[4])
                images[IMAGE_ID]['TX'] = float(tokens[5])
                images[IMAGE_ID]['TY'] = float(tokens[6])
                images[IMAGE_ID]['TZ'] = float(tokens[7])
                images[IMAGE_ID]['CAMERA_ID'] = int(tokens[8])
                images[IMAGE_ID]['NAME'] = tokens[9]
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


def add_plane(ax, model, x, y):
    a, b, c, d = model
    xmin, xmax = np.min(x), np.max(x)
    xlen = xmax - xmin
    xmin -= xlen * 0.2
    xmax += xlen * 0.2
    ymin, ymax = np.min(y), np.max(y)
    ylen = ymax - ymin
    ymin -= ylen * 0.2
    ymax += ylen * 0.2
    X = np.linspace(xmin, xmax, 10)
    Y = np.linspace(ymin, ymax, 10)
    X, Y = np.meshgrid(X, Y)
    Z = -(a*X + b*Y + d) / c
    ax.plot_surface(X, Y, Z)

def add_object(ax, corners, combinations):
    for comb in combinations:
        vertices = corners[comb, :]
        ax.add_collection3d(Poly3DCollection([vertices]))
    # ax.scatter3D(corners[:, 0], corners[:, 1], corners[:, 2], color='blue')

def plot3D(inplane_points, outplane_points, 
           plot_plane=False, model=None, 
           plot_box=False, plot_table=False, corners=None, combinations=None):
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
        add_plane(ax, model, xi, yi)
        
    # plot the box
    if plot_box:
        add_object(ax, corners, combinations)
        
    # plot the table
    if plot_table:
        add_object(ax, corners, combinations)
        
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()



if __name__ == '__main__':
    # video_name = './home/home.MOV'
    # frame_dir = './home/frames'
    # get_frame(video_name, frame_dir)

    points3D_txt = './home/points3D.txt'
    points3D = parse_points3D(points3D_txt)
    print(points3D.shape)
    print(np.min(points3D, axis=0))
    print(np.max(points3D, axis=0))
    print(np.mean(points3D, axis=0))

    # cameras_txt = './home/cameras.txt'
    # cameras = parse_cameras(cameras_txt)
    # print(cameras)

    # images_txt = './home/images.txt'
    # images = parse_images(images_txt)
    # print(images)
    # print({k:len(v) for k,v in images.items()})
