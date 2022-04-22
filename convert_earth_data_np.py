# Copyright 2021 Jiayi Xu

import numpy as np
import xarray as xr
import math
from scipy import interpolate
import argparse
from tqdm import tqdm


def interpolate_index(a, b):
    """
    a is the value to interpolate
    b is the bins
    """
    ind = np.digitize(a, b)
    ceil = np.floor(ind) + 1
    floor = np.floor(ind)

    floor = np.where(floor >= len(b)-1, len(b)-2, floor).astype(np.uint16)
    ceil = np.where(ceil >= len(b), len(b)-1, ceil).astype(np.uint16)

    v_floor = b[floor]
    v_ceil = b[ceil]

    alpha = (a - v_floor) / (v_ceil - v_floor)
    alpha = np.where(alpha > 1, 1, alpha)
    alpha = np.where(alpha < 0, 0, alpha)
    return ((1 - alpha) * floor) + (alpha * ceil)

def lerp3D(x, y, z, values):

    xfloor = np.floor(x)
    xceil = xfloor + 1
    yfloor = np.floor(y)
    yceil = yfloor + 1
    zfloor = np.floor(z)
    zceil = zfloor + 1

    xfloor = np.where(xfloor >= values.shape[0]-1, values.shape[0]-2, xfloor).astype(np.uint16)
    xceil = np.where(xceil >= values.shape[0], values.shape[0]-1, xceil).astype(np.uint16)

    yfloor = np.where(yfloor >= values.shape[1]-1, values.shape[1]-2, yfloor).astype(np.uint16)
    yceil = np.where(yceil >= values.shape[1], values.shape[1]-1, yceil).astype(np.uint16)

    zfloor = np.where(zfloor >= values.shape[2]-1, values.shape[2]-2, zfloor).astype(np.uint16)
    zceil = np.where(zceil >= values.shape[2], values.shape[2]-1, zceil).astype(np.uint16)

    xa = (x - xfloor) / (xceil - xfloor)
    ya = (y - yfloor) / (yceil - yfloor)
    za = (z - zfloor) / (zceil - zfloor)

    cell = np.array([[xfloor, yfloor, zfloor], [xceil, yfloor, zfloor], [xfloor, yfloor, zceil], [xceil, yfloor, zceil], [xfloor, yceil, zfloor], [xceil, yceil, zfloor], [xfloor, yceil, zceil], [xceil, yceil, zceil]], dtype=np.uint16)
    interpolation_weights = np.array([xa, ya, za], dtype=np.float64)

    # Each cell is [x, y, z]
    v0 = cell[0, :, :].T
    v1 = cell[1, :, :].T
    v2 = cell[2, :, :].T
    v3 = cell[3, :, :].T
    v4 = cell[4, :, :].T
    v5 = cell[5, :, :].T
    v6 = cell[6, :, :].T
    v7 = cell[7, :, :].T
    
    # Get the values at the indexes
    v0 = values[v0[:, 0], v0[:, 1], v0[:, 2]]
    v1 = values[v1[:, 0], v1[:, 1], v1[:, 2]]
    v2 = values[v2[:, 0], v2[:, 1], v2[:, 2]]
    v3 = values[v3[:, 0], v3[:, 1], v3[:, 2]]
    v4 = values[v4[:, 0], v4[:, 1], v4[:, 2]]
    v5 = values[v5[:, 0], v5[:, 1], v5[:, 2]]
    v6 = values[v6[:, 0], v6[:, 1], v6[:, 2]]
    v7 = values[v7[:, 0], v7[:, 1], v7[:, 2]]
    
    # Get the interpolation weights
    wx = interpolation_weights[0]
    wy = interpolation_weights[1]
    wz = interpolation_weights[2]
    
    # Perform trilinear interpolation
    return ((1 - wx) * (1 - wz) * (1 - wy) * v0) + (wx * (1 - wz) * (1 - wy) * v1) + ((1 - wx) * wz * (1 - wy) * v2) + (wx * wz * (1 - wy) * v3) + ((1 - wx) * (1 - wz) * wy * v4) + (wx * (1 - wz) * wy * v5) + ((1 - wx) * wz * wy * v6) + (wx * wz * wy * v7)

# transform data values form spherical coordinate to Cartesian coordinate
    # step 1: create a uniform Cartesian coordinate, covering the spherical coordinate
    # step 2: interpolate values of for each grid point on the uniform Cartesian coordinate
        # transform the grid point to Cartesian coordinate, and linearly interpolate the value in 3D Cartesian coordinate
def transform_sperical_to_cartesian(size, temp, vx, vy, vz, sperical_coordinate): 
    lat, r, lon = sperical_coordinate
    
    max_r = r[-1] # get the maximal radius
    
    # define new cartesian grid
    nx = size
    ny = size
    nz = size
    res = np.zeros((4, nx, ny, nz), dtype=np.float32)
    # corresponding coordinate values of each grid point
    x = np.linspace(-max_r, max_r, nx) 
    y = np.linspace(-max_r, max_r, ny)
    z = np.linspace(-max_r, max_r, nz)

    pts = np.stack(np.meshgrid(x, y, z, indexing='xy'))
    
    nan_val = 0 #np.nan
    _r = np.sqrt(pts[0,:,:,:]**2 + pts[1,:,:,:]**2 + pts[2,:,:,:]**2)
    _lat = np.rad2deg(np.arcsin(pts[2,:,:,:] / _r))
    _lon = np.rad2deg(np.arctan2(pts[1,:,:,:], pts[0,:,:,:]))
    _lon = np.where(_lon < 0, _lon + 360, _lon)

    _r_id = interpolate_index(_r, r).flatten()
    _lat_id = interpolate_index(_lat, lat).flatten()
    _lon_id = interpolate_index(_lon, lon).flatten()

    _temp = lerp3D(_lat_id, _r_id, _lon_id, temp).reshape(size, size, size)
    _vx = lerp3D(_lat_id, _r_id, _lon_id, vx).reshape(size, size, size)
    _vy = lerp3D(_lat_id, _r_id, _lon_id, vy).reshape(size, size, size)
    _vz = lerp3D(_lat_id, _r_id, _lon_id, vz).reshape(size, size, size)

    _temp = np.where(np.abs(_r) > max_r, 0, _temp)
    _vx = np.where(np.abs(_r) > max_r, 0, _vx)
    _vy = np.where(np.abs(_r) > max_r, 0, _vy)
    _vz = np.where(np.abs(_r) > max_r, 0, _vz)

    res = np.stack((_temp, _vx, _vy, _vz))

    return _temp, res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='converting data')
    parser.add_argument('--input_path', default='./data/earth/data/nc/', type=str, metavar='PATH', help='prefix path to data')
    parser.add_argument('--output_path', default='./data/earth/data/raw/', type=str, metavar='PATH', help='prefix path to data')
    parser.add_argument('--example', default='spherical001', type=str, metavar='EX', help='input example name')
    parser.add_argument('--size', default=256, type=int, metavar='S', help='size^3 of output cartesian data')

    # Put parameters into a dictionary
    args = vars(parser.parse_args())

    print(f'Converting {args["example"]}')

    data = xr.open_dataset(f'{args["input_path"]}{args["example"]}.nc')

    # spherical coordinates, every variable is stored in the dimensionality order (lat, r, lon)
    _lat = data.lat.values # latitudes in degrees from 90 to -90, a 1x180 numpy array
    _r = data.r.values # radial discretization [3485, 6371], a 1x201 numpy array
    _lon = data.lon.values # longitudes in degrees [0, 360], a 1x360 numpy array
    # print(_lat)

    size = args['size']
    
    temp = data.temperature.values # temperature, this is a 180x201x360 numpy array
    vx = data.vx.values # vx, this is a 180x201x360 numpy array
    vy = data.vy.values # vy, this is a 180x201x360 numpy array
    vz = data.vz.values # vz, this is a 180x201x360 numpy array

    temp_cartesian, all_cartesian = transform_sperical_to_cartesian(size, temp, vx, vy, vz, (_lat, _r, _lon))

    temp_cartesian.T.astype("float32").tofile(f'{args["output_path"]}{args["example"]}_temp_{args["size"]}.raw')
    all_cartesian.T.astype("float32").tofile(f'{args["output_path"]}{args["example"]}_{args["size"]}.raw')
    
    print('Done! ')
