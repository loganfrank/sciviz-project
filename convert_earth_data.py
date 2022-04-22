# Copyright 2021 Jiayi Xu

import numpy as np
import xarray as xr
import math
from scipy import interpolate
import argparse


def interpolate_index(a, b):
    """
    a is the value to interpolate
    b is the bins
    """
    ind = np.digitize(a, b)
    ceil = math.floor(ind) + 1
    floor = math.floor(ind)

    if ceil == len(b):
        floor -= 1
        ceil -= 1

    v_floor = b[floor]
    v_ceil = b[ceil]

    alpha = (a - v_floor) / (v_ceil - v_floor)
    return ((1 - alpha) * floor) + (alpha * ceil)

def lerp3D(pt, values):
    x, y, z = pt
    xfloor = math.floor(x)
    xceil = xfloor + 1
    yfloor = math.floor(y)
    yceil = yfloor + 1
    zfloor = math.floor(z)
    zceil = zfloor + 1

    xa = (x - xfloor) / (xceil - xfloor)
    ya = (y - yfloor) / (yceil - yfloor)
    za = (z - zfloor) / (zceil - zfloor)

    cell = np.array([[xfloor, yfloor, zfloor], [xceil, yfloor, zfloor], [xfloor, yfloor, zceil], [xceil, yfloor, zceil], [xfloor, yceil, zfloor], [xceil, yceil, zfloor], [xfloor, yceil, zceil], [xceil, yceil, zceil]], dtype=np.uint16)
    interpolation_weights = np.array([xa, ya, za], dtype=np.float64)

    # Each cell is [x, y, z]
    v0 = cell[0]
    v1 = cell[1]
    v2 = cell[2]
    v3 = cell[3]
    v4 = cell[4]
    v5 = cell[5]
    v6 = cell[6]
    v7 = cell[7]
    
    # Get the values at the indexes
    v0 = values[v0[0], v0[1], v0[2]]
    v1 = values[v1[0], v1[1], v1[2]]
    v2 = values[v2[0], v2[1], v2[2]]
    v3 = values[v3[0], v3[1], v3[2]]
    v4 = values[v4[0], v4[1], v4[2]]
    v5 = values[v5[0], v5[1], v5[2]]
    v6 = values[v6[0], v6[1], v6[2]]
    v7 = values[v7[0], v7[1], v7[2]]
    
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
    
    npts = nx*ny*nz
    cnt = 0
    nan_val = 0 #np.nan
    for i in range(nx):
        for j in range(ny):
            # print('Progress: ', cnt / npts)
            for k in range(nz):
                cnt += 1
                
                _x = x[i]
                _y = y[j]
                _z = z[k]
                
                _r = np.sqrt(_x**2 + _y**2 + _z**2)
                _lat = np.rad2deg(np.arcsin(_z / _r))
                _lon = np.rad2deg(np.arctan2(_y, _x))
                
                if _lon < 0: 
                    _lon += 360
                
                if np.abs(_r) > max_r:
                    _temp = nan_val
                    _vx = nan_val
                    _vy = nan_val
                    _vz = nan_val
                else:
                    _r_id = interpolate_index(_r, r)
                    _lat_id = interpolate_index(_lat, lat)
                    _lon_id = interpolate_index(_lon, lon)
                    
                    if _r_id is None or _lat_id is None or _lon_id is None: 
                        _temp = nan_val
                        _vx = nan_val
                        _vy = nan_val
                        _vz = nan_val
                    elif (_r_id < 0):
                        _temp = nan_val
                        _vx = nan_val
                        _vy = nan_val
                        _vz = nan_val
                    else:
                        # print(_lat_id, _r_id, _lon_id)
                        pt = np.array([_lat_id, _r_id, _lon_id], dtype=np.float32)
                        _temp = lerp3D(pt, temp)
                        _vx = lerp3D(pt, vx)
                        _vy = lerp3D(pt, vy)
                        _vz = lerp3D(pt, vz)
                
                res[0][i][j][k] = _temp
                res[1][i][j][k] = _vx
                res[2][i][j][k] = _vy
                res[3][i][j][k] = _vz

    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='converting data')
    parser.add_argument('--path', default='./data/earth/', type=str, metavar='PATH', help='prefix path to data')
    parser.add_argument('--example', default='spherical001', type=str, metavar='EX', help='input example name')
    parser.add_argument('--size', default=256, type=int, metavar='S', help='size^3 of output cartesian data')

    print(f'Converting {args["example"]}')

    # Put parameters into a dictionary
    args = vars(parser.parse_args())

    data = xr.open_dataset(f'{args["path"]}{args["example"]}.nc')

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

    temp_cartesian = transform_sperical_to_cartesian(size, temp, vx, vy, vz, (_lat, _r, _lon))

    temp_cartesian.T.astype("float32").tofile(f'{args["path"]}{args["example"]}_{args["size"]}.raw')
    # temp_cartesian.T.astype("float32").tofile("res_" + str(size) + ".raw")
    
    print('Done! ')
