import numpy as np

def _create_combinations(idx):
    """ Generate combinations of corners for 6 box faces, given 8 indices of corners
    """
    combinations = np.array([[idx[0],idx[1],idx[2],idx[3]],  # bottom
                             [idx[4],idx[5],idx[6],idx[7]],  # top
                             [idx[0],idx[1],idx[5],idx[4]],  # right
                             [idx[3],idx[2],idx[6],idx[7]],  # left
                             [idx[0],idx[4],idx[7],idx[3]],  # front
                             [idx[1],idx[5],idx[6],idx[2]]]) # back
    return combinations

def get_combinations(num, obj):
    idx = np.arange(num)
    if obj == 'box':
        return _create_combinations(idx)
    elif obj == 'table':
        comb_leg1 = _create_combinations(np.concatenate((idx[:4], idx[16:20])))
        comb_leg2 = _create_combinations(np.concatenate((idx[4:8], idx[20:24])))
        comb_leg3 = _create_combinations(np.concatenate((idx[8:12], idx[24:28])))
        comb_leg4 = _create_combinations(np.concatenate((idx[12:16], idx[28:32])))
        comb_top  = _create_combinations(np.array([idx[16], idx[21], idx[26], idx[31], *idx[32:]]))
        return np.vstack([comb_leg1, comb_leg2, comb_leg3, comb_leg4, comb_top])

def build_3D_box(dimx, dimy, dimz):
    """ Generate coordinates of 8 corners of a 3D box
    """
    halfx, halfy = dimx/2, dimy/2
    corners = [[halfx, halfy, 0],
               [-halfx, halfy, 0],
               [-halfx, -halfy, 0],
               [halfx, -halfy, 0],
               [halfx, halfy, dimz],
               [-halfx, halfy, dimz],
               [-halfx, -halfy, dimz],
               [halfx, -halfy, dimz]]
    return np.array(corners)

def build_3D_table(dimx, dimy, legheight, topheight):
    """ Generate coordinates of corners of a 3D table
    """
    halfx, halfy = dimx/2, dimy/2
    legsize = halfx * 0.2
    corners = [[halfx, halfy, 0],
               [halfx-legsize, halfy, 0],
               [halfx-legsize, halfy-legsize, 0],
               [halfx, halfy-legsize, 0],
               [-halfx+legsize, halfy, 0],
               [-halfx, halfy, 0],
               [-halfx, halfy-legsize, 0],
               [-halfx+legsize, halfy-legsize, 0],
               [-halfx+legsize, -halfy+legsize, 0],
               [-halfx, -halfy+legsize, 0],
               [-halfx, -halfy, 0],
               [-halfx+legsize, -halfy, 0],
               [halfx, -halfy+legsize, 0],
               [halfx-legsize, -halfy+legsize, 0],
               [halfx-legsize, -halfy, 0],
               [halfx, -halfy, 0],
               [halfx, halfy, legheight],
               [halfx-legsize, halfy, legheight],
               [halfx-legsize, halfy-legsize, legheight],
               [halfx, halfy-legsize, legheight],
               [-halfx+legsize, halfy, legheight],
               [-halfx, halfy, legheight],
               [-halfx, halfy-legsize, legheight],
               [-halfx+legsize, halfy-legsize, legheight],
               [-halfx+legsize, -halfy+legsize, legheight],
               [-halfx, -halfy+legsize, legheight],
               [-halfx, -halfy, legheight],
               [-halfx+legsize, -halfy, legheight],
               [halfx, -halfy+legsize, legheight],
               [halfx-legsize, -halfy+legsize, legheight],
               [halfx-legsize, -halfy, legheight],
               [halfx, -halfy, legheight],
               [halfx, halfy, legheight+topheight],
               [-halfx, halfy, legheight+topheight],
               [-halfx, -halfy, legheight+topheight],
               [halfx, -halfy, legheight+topheight]]
    return np.array(corners)


if __name__ == '__main__':
    # check combinations of box corners
    corners = build_3D_box(2, 2, 1)
    combinations = get_combinations(len(corners), 'box')
    print(combinations)
    
    # check combinations of table corners
    corners = build_3D_table(2, 2, 1.5, 0.5)
    combinations = get_combinations(len(corners), 'table')
    print(combinations)