#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 16:04:33 2018

@author: andy
"""

#!/usr/bin/env python
'''
Documentation ...
'''

# For the class here below
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from mpl_toolkits.mplot3d import Axes3D
from pycubicsplines import*
from my_functions import*




def points_to_nparray(points):
    """
    Takes a message of points and
    :param points:
    :return:
    """
    print("\n... Taking geo-msgs list of 'points' and putting into array")
    coords = np.zeros((len(points.input_points.points), 3))
    for i, a_coord in enumerate(points.input_points.points):
        coords[i, 0] = a_coord.x
        coords[i, 1] = a_coord.y
        coords[i, 2] = a_coord.z
    return coords

def ellipse_center(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    num = b*b-a*c
    x0=(c*d-b*f)/num
    y0=(a*f-b*d)/num
    return np.array([x0,y0])

def ellipse_angle_of_rotation(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    return 0.5*np.arctan(2*b/(a-c))

def ellipse_axis_length(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    res1=np.sqrt(up/down1)
    res2=np.sqrt(up/down2)
    return np.array([res1, res2])

def fitting_spline(xyz_array, dtheta=.008, decimation = 50):
    """
    DESCRIPTION
    Given we have found a seam, we now need to fit an ellipse to said seam. This may help when the data is not so
    perfect. For this technique what we will do first is fit a plane to the data.
    :return:
    """

    print("... Fitting Spline")

    #
    # FIT PLANE
    #
    
    print("... lets decimate")
    
    if decimation > 1:
        spaces = np.arange(0, xyz_array.shape[0], decimation)
        xyz_intermed = np.zeros((spaces.shape[0], 3))
        for i, a_ind in enumerate(spaces):
            xyz_intermed[i, :] = xyz_coords[a_ind, :]
        xyz_array = xyz_intermed

    print("... ... Fitting plane")

    # Fit a plane to each array first, we use insight from:
    # https://math.stackexchange.com/questions/99299/best-fitting-plane-given-a-set-of-points

    # Make a duplicate for later usage when finding the d constant value. We use duplicates so we do not alter the
    # input arrays.
    xyz_array_dup_0 = np.copy(xyz_array)

    # Find mean
    xyz_norms = np.mean(xyz_array_dup_0, axis=0)
    # print("\nXYZ mean vlaues are ... ")
    # print(xyz_norms)

    # Subtract mean
    xyz_array_dup_0[:, 0] = xyz_array_dup_0[:, 0] - xyz_norms[0]
    xyz_array_dup_0[:, 1] = xyz_array_dup_0[:, 1] - xyz_norms[1]
    xyz_array_dup_0[:, 2] = xyz_array_dup_0[:, 2] - xyz_norms[2]

    # Transpose
    xyz_array_dup_0 = np.transpose(xyz_array_dup_0)

    # Debugging the crap out of the out of the data input
    # print("\nNorm-shape and values ... ")
    # print(xyz_norms.shape)
    # print(xyz_norms)

    # Singular Value Decomposition
    u, s, v = np.linalg.svd(xyz_array_dup_0)

    # Grab the normal that
    normal = u[:, 2]
    # print("\n Looking at normal values ... ")
    # print(normal)

    # The normal vector SHOULD already be normalized.
    # print("\nNormalize the normal values ... ")
    # norm_value = (normal[0]**2. + normal[1]**2. + normal[2]**2.)**.5
    # print(norm_value)

    # Now we have to find the d-value. Which is the plane constant.
    d = xyz_array[:, 0] * normal[0] + xyz_array[:, 1] * normal[1] + xyz_array[:, 2] * normal[2]
    d = np.mean(d) * -1.

    # DEBUGGING (PLOTTING)
    # Plot out plane
    xx, yy = np.meshgrid(np.arange(-.2, .2, .01), np.arange(-.2, .2, .01))
    zz = (-normal[0] * xx - normal[1] * yy - d) * 1./normal[2]
    #plt3d = plt.figure(2).gca(projection='3d')
    #plt3d.plot_surface(xx, yy, zz, alpha=0.2)
    #plt.show()

    #
    # PARAMETERIZE PLANE
    #

    print("... ... parameterize plane")

    # Now that we have a plane, we need to go ahead and parameterize the plane that we are fit too, and then we
    # can solve for the ellipse using constrained optimziation.
    # See, https://mathinsight.org/plane_parametrization_examples

    # We set some values (normal in the x and y) and then solve for the rest (values in the z), followed up by a
    # normalization operation.
    a_n = np.array([1., 1., 0.])

    a_n[2] = (-a_n[0]*normal[0] - a_n[1]*normal[1]) / normal[2]
    a_n_val = (a_n[0]**2. + a_n[1]**2. + a_n[2]**2.)**.5
    a_n = a_n / a_n_val

    # Find b_n using a cross-product you dumb fuck.
    b_n = np.cross(a_n, normal)
    b_n_val = (b_n[0] ** 2. + b_n[1] ** 2. + b_n[2] ** 2.) ** .5
    b_n = b_n / b_n_val

    # DEBUGGING
    # Check if vectors are orthogonal
    # a_to_b = a_n[0]*b_n[0] + a_n[1]*b_n[1] + a_n[2]*b_n[2]
    # n_to_a = a_n[0]*normal[0] + a_n[1]*normal[1] + a_n[2]*normal[2]
    # n_to_b = normal[0] * b_n[0] + normal[1] * b_n[1] + normal[2] * b_n[2]
    # print("\na_n dotted into b_n = " + str(a_to_b))
    # print("a_n dotted into normal = " + str(n_to_a))
    # print("b_n dotted into normal = " + str(n_to_b))

    # The plane is parameterized s.t. we have x = s*a_n + t*b_n + c, where a_n and b_n are the normal vectors, and c
    # is some point on the plane. We can solve for c by picking x=0, y=0, and solving for z so then
    # c = [0., 0., d/normal[2]
    c = np.array([0., 0., -d/normal[2]])

    #
    # PROJECT ORIGINAL POINTS ONTO PLANE
    #

    print("... ... project onto plane")

    # Now project points onto plane, the code below can be sped up by vectorizing the code.
    # http://www.nabla.hr/CG-LinesPlanesIn3DB5.htm

    projected_points = np.zeros((xyz_array.shape[0], 3))  # This will store the projections onto the plane
    for i in range(xyz_array.shape[0]):
        # Find t-values
        x_0 = xyz_array[i, 0]
        y_0 = xyz_array[i, 1]
        z_0 = xyz_array[i, 2]
        t = -1. * (normal[0]*x_0 + normal[1]*y_0 + normal[2]*z_0 + d) / (normal[0]**2. + normal[1]**2. + normal[2]**2.)
        projected_points[i, 0] = x_0 + t * normal[0]
        projected_points[i, 1] = y_0 + t * normal[1]
        projected_points[i, 2] = z_0 + t * normal[2]

    # Now take projected points and shift to s-t coordinates.
    A = np.zeros((3, 3))
    A[0, 0] = a_n[0]
    A[1, 0] = a_n[1]
    A[2, 0] = a_n[2]
    A[0, 1] = b_n[0]
    A[1, 1] = b_n[1]
    A[2, 1] = b_n[2]
    A[0, 2] = c[0]
    A[1, 2] = c[1]
    A[2, 2] = c[2]
    y = np.zeros((3, 1))
    parameterized_points = np.zeros((projected_points.shape[0], 2))
    for i in range(projected_points.shape[0]):
        y[0, 0] = projected_points[i, 0]
        y[1, 0] = projected_points[i, 1]
        y[2, 0] = projected_points[i, 2]
        st1 = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(A), A)), np.transpose(A)), y)
        parameterized_points[i, 0] = st1[0]
        parameterized_points[i, 1] = st1[1]

    #
    # FIT THE FUCKING ELLIPSE
    #

    print("... ... fit the fucking spline")

    # Yeah so I was working on this and then came across this ... so lets just use their code!
    # http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html

    # This is the actual fitting
    x = parameterized_points[:, 0]
    y = parameterized_points[:, 1]
    #print(x.tolist()) tolist works for x and y
    x = x.tolist()
    y = y.tolist()
    
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
    

    #
    # PUT BACK INTO 3D SPACE
    #

    print("... ... put ellipse back into 3d space")

    x3d = np.zeros((len(rx), 3))
    a_vec = np.ones((3, 1))
    for i, (s, t) in enumerate(zip(rx, ry)):
        a_vec[0] = s
        a_vec[1] = t
        new_point = np.matmul(A, a_vec)
        x3d[i, 0] = new_point[0]
        x3d[i, 1] = new_point[1]
        x3d[i, 2] = new_point[2]

    # DEBUGGING (plotting)
    # Plot out ellipse in 3d-space
    fig = plt.figure(5)
    ax = fig.add_subplot(111, projection='3d') # For newer matplotlib
    ax = Axes3D(fig) # For older matplotlib
    plt.plot(x3d[:, 0], x3d[:, 1], x3d[:, 2], c='b', marker='^')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.view_init(10, 0)
    plt.show()

    print("... ... prepare output")





if __name__ == "__main__":
    # Instantiate a fitting ellipse object and spin up a thread for the service
    print("\nReading coordinates")
    start_time = time.time()
    # file_name = '/home/atlas2/LonsEngRepos/glugun_sensor_python/src/Alpha/findSeamsInClouds/just_seams.xyz'
    file_name = '/home/andy/Documents/spline_fitting/one_seam.xyz'
    xyz_coords = read_xyz(file_name)
    total_run_time = time.time() - start_time
    print("... Time to read coordinates = " + str(total_run_time))
    print("... The shape of xyz_coords is " + str(xyz_coords.shape))
    fitting_spline(xyz_coords)


