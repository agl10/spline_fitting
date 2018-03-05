# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import numpy as np


from scipy.interpolate import splprep, splev



import time
from sklearn.neighbors import NearestNeighbors
import networkx as nx

from mpl_toolkits.mplot3d import Axes3D
import math

from my_functions import*

from pycubicsplines import*
# from cvxpy import *
#import cvxpy

x = np.arange(10)
y = np.sin(x)
cs = CubicSpline(x, y)
xs = np.arange(-0.5, 9.6, 0.1)
plt.figure(figsize=(6.5, 4))
plt.plot(x, y, 'o', label='data')
plt.plot(xs, np.sin(xs), label='true')
plt.plot(xs, cs(xs), label="S")
plt.plot(xs, cs(xs, 1), label="S'")
plt.plot(xs, cs(xs, 2), label="S''")
plt.plot(xs, cs(xs, 3), label="S'''")
plt.xlim(-0.5, 9.5)
plt.legend(loc='lower left', ncol=2)
plt.show()


x = [-2.5, 0.0, 2.5, 5.0, 7.5, 3.0, -1.0]
y = [0.7, -6, 5, 6.5, 0.0, 5.0, -2.0]

sp = Spline2D(x, y)
s = np.arange(0, sp.s[-1], 0.1)

rx, ry, ryaw, rk = [], [], [], []
for i_s in s:
  ix, iy = sp.calc_position(i_s)
  rx.append(ix)
  ry.append(iy)
  ryaw.append(sp.calc_yaw(i_s))
  rk.append(sp.calc_curvature(i_s))

flg, ax = plt.subplots(1)
plt.plot(x, y, "xb", label="input")
plt.plot(rx, ry, "-r", label="spline")
plt.grid(True)
plt.axis("equal")
plt.xlabel("x[m]")
plt.ylabel("y[m]")
plt.legend()

flg, ax = plt.subplots(1)
plt.plot(s, [math.degrees(iyaw) for iyaw in ryaw], "-r", label="yaw")
plt.grid(True)
plt.legend()
plt.xlabel("line length[m]")
plt.ylabel("yaw angle[deg]")

flg, ax = plt.subplots(1)
plt.plot(s, rk, "-r", label="curvature")
plt.grid(True)
plt.legend()
plt.xlabel("line length[m]")
plt.ylabel("curvature [1/m]")

plt.show()


##########################################
##########################################

if __name__ == "__main__":

    print("\nReading coordinates")
    start_time = time.time()
    # file_name = '/home/atlas2/LonsEngRepos/glugun_sensor_python/src/Alpha/findSeamsInClouds/just_seams.xyz'
    file_name = '/home/andy/Documents/spline_fitting/one_seam.xyz'
    xyz_coords = read_xyz(file_name)
    total_run_time = time.time() - start_time
    print("... Time to read coordinates = " + str(total_run_time))
    print("... The shape of xyz_coords is " + str(xyz_coords.shape))


    ordered_seams, thetas, radii, output = full_seam_finder_0(xyz_coords, decimation=700, num_of_neighbs=25, minimum_distance=.010, dtheta=.05,
                                       z_order_asscending=True)

    #print(sorted(thetas))
    radii
    #print(thetas.shape)
    #print(radii.shape)
    #print(xyz_coords.shape)
    #print(output.shape)
    
    #print(sorted(thetas))
    #print(radii)
    thetas = np.expand_dims(thetas, axis=1)
    #print(thetas.shape)
    
    total = np.concatenate((thetas,output), axis=1)
    #print(total)
    total = total[total[:,0].argsort()]
   # print(total)
    
    #print(total[:,1:3])
    x = (total[:,1:2])
    y = (total[:,2:3])
    
    x = np.reshape(x, ( 1,len(x)))
    y = np.reshape(y, ( 1,len(y)))
    
    #print(repr(np.squeeze(x)))
   # print(np.squeeze(x).tolist())
    
    #x = [-2.5, 0.0, 2.5, 5.0, 7.5, 3.0, -1.0]
    #y = [0.7, -6, 5, 6.5, 0.0, 5.0, -2.0]
    
    #x = [-0.01301, -0.0307, -0.050687, -0.072453, -0.09848, -0.124424, -0.146484]
    #y = [0.114112, 0.111434, 0.108723,0.102973, 0.093847, 0.077632, 0.049974]
    
    x = np.squeeze(x).tolist()
    y = np.squeeze(y).tolist()
    
    #print(x)
    
    sp = Spline2D(x, y)
    s = np.arange(0, sp.s[-1], 0.011)

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = sp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(sp.calc_yaw(i_s))
        rk.append(sp.calc_curvature(i_s))

    flg, ax = plt.subplots(1)
    plt.plot(x, y, "xb", label="input")
    plt.plot(rx, ry, "-r", label="spline")
    plt.grid(True)
    plt.axis("equal")
    plt.xlabel("x[m]")
    plt.ylabel("y[m]")
    plt.legend()

    flg, ax = plt.subplots(1)
    plt.plot(s, [math.degrees(iyaw) for iyaw in ryaw], "-r", label="yaw")
    plt.grid(True)
    plt.legend()
    plt.xlabel("line length[m]")
    plt.ylabel("yaw angle[deg]")

    flg, ax = plt.subplots(1)
    plt.plot(s, rk, "-r", label="curvature")
    plt.grid(True)
    plt.legend()
    plt.xlabel("line length[m]")
    plt.ylabel("curvature [1/m]")

    plt.show()
    