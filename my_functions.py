#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 12:53:45 2018

@author: andy
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


def read_xyz(filename):
    """
    DESCRIPTION
    Reads a text file with the x-y-z points and turn these into an numpy array. This will work if the file
    has 6 values across each line.
    :param filename:
    :return: numpy array, Nx3 array of x-y-z points.
    """

    # First count the line numbers
    total_count = 0
    xyz = open(filename)
    for l in enumerate(xyz):
        total_count += 1
    xyz.close()

    # Now read each line and grab the individual x, y, z points.
    xyz_coords = np.zeros((total_count, 3))
    xyz = open(filename)
    for i, line in enumerate(xyz):
        #x, y, z, _, _, _ = line.split()
        x, y, z = line.split()
        xyz_coords[i, 0] = x
        xyz_coords[i, 1] = y
        xyz_coords[i, 2] = z

        # For debugging.
        # if i < 10:
        #     print("\nx = " + str(x) + ", y = " + str(y) + ", z = " + str(z))

    return xyz_coords

def full_seam_finder_0(xyz_coords, decimation=10, num_of_neighbs=25, minimum_distance=.010, dtheta=.05,
                       z_order_asscending=True):
    """
    This is the instantiation of a full seam finder.
    :param xyz_coordinates: Mx3 numpy array of 3D points representing seams.
    :param decimation: positive integer, this is if we want to remove points from the given point cloud. If for example
           decimation=10, that means for every 10 points, only 1 will be kept.
    :param num_of_neighbs: positive integer, this tells us how many neighbors to look at during graph construction.
           The fewer necessary the faster everything is, but but including fewer you run the risk of missing connected
           regions. The default of 25 was found to work well.
    :param minimum_distance: In order to characterize two points as being part of the same seam we use a threshold
           distance. That is if the two points are less than the minimum_distance, then will end up being connected.
    :param dtheta: radians, in order to get a ordered set of points for each seam, we find the centroid of discrete
           groups of points all within delta-theta. The smaller this value, the more final points will be generated.
    :return: list of numpy arrays. Each numpy in said list has shape Nx3, where N is dictated essentailly by the input
             dtheta. Each numpy array is itself a separate ordered list of 3D points. Each row of the numpy arrays has
             values of x, y, and z.
    """

    # DECIMATE THE POINTS
    # Perhaps the number of points is too great and not required for good estimation of seam location. Thus we can
    # here decimate the number of points.

    if decimation > 1:
        spaces = np.arange(0, xyz_coords.shape[0], decimation)
        xyz_intermed = np.zeros((spaces.shape[0], 3))
        for i, a_ind in enumerate(spaces):
            xyz_intermed[i, :] = xyz_coords[a_ind, :]
        xyz_coords = xyz_intermed

    # FIND NEAREST NEIGHBORS
    # Given input data is unorganized, we first find all the nearest neighbors, and distances between said neighbors.
    # This data will be used later to segment the points into separate seams.

    nrbs = NearestNeighbors(n_neighbors=num_of_neighbs, algorithm='auto').fit(xyz_coords)
    distances, indices = nrbs.kneighbors(xyz_coords)

    # FIND ALL SEPARATE SEAMS

    # "indices_list" will be a list of lists. Each sub list had integers, which are indices of the points (as
    # described in xyz_coords, ditances, and indices).
    indices_list = seam_separator(distances, indices, minimum_distance=minimum_distance)

    # REDUCE TO ORDERED SETS
    # Here we take all these large set of points, where each sub-set represents a seam, and for each sub-set we find
    # an ordered set of points to use as way-points for the robot.

    # Ordered seams will be a list of numpy arrays, where each numpy array will an Nx3 array.
    ordered_seams, thetas, radii, output = simple_radial_seam_discretizer(indices_list, xyz_coords, dtheta=dtheta)

    # Reorder the seams along
    ordered_seams = reorder_seams_by_z(ordered_seams, asscending=z_order_asscending)

    return ordered_seams, thetas, radii, output

def reorder_seams_by_z(ordered_seams, asscending=True):
    """
    DESCRIPTION
    Orders the seams by their centroids.

    :param ordered_seams:
    :param asscending:
    :return:
    """

    # We need to find the order of the seams along the z-axis.
    z_centroids = np.zeros((len(ordered_seams), 2))
    for i in range(len(ordered_seams)):
        z_centroids[i, 0] = i

    # Find NAN values
    for an_array in ordered_seams:
        for a_val in an_array[:, 2]:
            if math.isnan(a_val):
                print("\nNAN found in array ...")

    # Find the z-centroid
    for i, an_array in enumerate(ordered_seams):
        z_centroids[i, 1] = np.mean(an_array[:, 2])

    # Now sort by the z-centroids
    if asscending:
        z_centroids = z_centroids[(-z_centroids[:, 1]).argsort()]
    else:
        z_centroids = z_centroids[(z_centroids[:, 1]).argsort()]

    # Now make the new outputs.
    new_ordered_seams = []
    for ind in z_centroids[:, 0]:
        new_ordered_seams.append(ordered_seams[int(ind)])

    return new_ordered_seams

def seam_separator(distances, indices, minimum_distance=.010):
    """
    DESCRIPTION
    This uses actual graph theory to make a number of connected graphs, and then finds then uses connected components
    search to find all the connected components. This is working. The limitation is on the users side. For this to work
    successfully we have to find enough of the neighbors.

    HOW DOES THIS WORK?
    This algorithm here below works by firstly constructing a graph where the 3D points within said graph are the nodes
    of the graph. We then connect the nodes with edges (non-directional) if and only if the distance (Euclidean) is
    below the the limiting value (given as function input). After we construct the graph we can use algorithms built
    into the network-x library to find all the "connected-components". This essentially boils down to find all the nodes
    that are connected together directly or through other intermediary nodes.

    :param distances: nXm numpy array, where n is the number of 3D-points/nodes and m is the number of neighboring
           points. The distances[i,j] is the distance between node i and between node Q, where Q can be found by
           searching the indices matrix: Q = indices[i, j]. That is the indices matrix is the same size as the
           distances matrix.
    :param indices: nXm numpy array, where n is the number of 3D-points/nodes and m is the number of neighboring
           points. This indices matrix is like a helper matrix that only makes sense along side the distances
           matrix as described above.
    :param minimum_distance: scalar, this is a cutoff that tells us
    :return: A list of lists, where each sub-list has indices that represent a connected component.
    """

    # Generate a graph. This is an empty construct at this moment.
    G = nx.Graph()

    # Add all the nodes to the graph. A "node" is simply one of the points in the point cloud.
    # start_time = time.time()
    for i in range(distances.shape[0]):
        G.add_node(i)
    # total_run_time = time.time() - start_time
    # print("... Time to add nodes = " + str(total_run_time))

    # Add edges to the graph if the distance between the nodes is such that it is less than the acceptable distance.
    # That is set by the user.
    # start_time = time.time()
    for i in range(distances.shape[0]):
        for a_ind, a_dist in zip(indices[i, :], distances[i, :]):
            if a_dist <= minimum_distance:
                # Add the fucking edge
                G.add_edge(i, a_ind)
    # total_run_time = time.time() - start_time
    # print("... Time to add edges = " + str(total_run_time))
    # print("The number of connected components is " + str(nx.number_connected_components(G)))

    # Call on built in methods to find connected components in our built in graph.
    hyper = []  # A list of our connected component sets.
    for a_conn in nx.connected_components(G):
        # a_conn is a python set, that we have to pull apart and jam into a python list.
        dummy = []
        for set_el in set(a_conn):
            dummy.append(set_el)
        # print(len(dummy))
        # print("... " + str(dummy[0]) + ", " + str(dummy[1]) + ", " + str(dummy[2]) + ", " + str(dummy[3]))
        hyper.append(dummy)

    return hyper


def simple_radial_seam_discretizer(indices_list, xyz_coords, dtheta):
    """
    DESCRIPTION
    A method to take unorganized seams (point clouds) and replace them with a set of ordered points, where each point
    is roughly centered along each seam.

    This method works by ASSSUMING that the seams are roughly circles centered along the z-axis. We then divide up up
    the seams (point-cloud) by delta-theta values. For each group of points that falls into a delta-theta, we replace
    all those points with a single point at the centroid.

    :param indices_list: list of lists, each sub-list is occupied by integers, where each integer is an index for
           a row in xyz_coords. That is these sub-lists indicate which xyz points constitute a single, individual seam.
    :param xyz_coords: Nx3 array, where each row holds a 3D point (x, y, z)
    :param dtheta: scalar (radians) the size of discretization.
    :return: list of numpy arrays, each array is a separate ordered seam.
    """

    # Project all the points onto z=0  plane: shift into polar coordinates.
    theta = np.zeros(xyz_coords.shape[0])  # Just theta values
    radii = np.zeros(xyz_coords.shape[0])
    output = np.zeros((xyz_coords.shape[0], 2))
    for i, (x, y) in enumerate(zip(xyz_coords[:, 0], xyz_coords[:, 1])):

        if x <= 0:
            if y >= 0:
                if math.fabs(y) > .00001:
                    theta[i] = math.atan(-x/y)
                    radii[i] = np.sqrt((x*x + y*y))
                    output[i,0] = x
                    output[i,1] = y
                else:
                    theta[i] = -100.0
                    radii[i] = np.sqrt((x*x + y*y))
                    output[i,0] = x
                    output[i,1] = y
            else:
                if math.fabs(x) > .00001:
                    theta[i] = math.pi/2. + math.atan(-y/-x)
                    radii[i] = np.sqrt((x*x + y*y))
                    output[i,0] = x
                    output[i,1] = y
                else:
                    theta[i] = -100.0
                    radii[i] = np.sqrt((x*x + y*y))
                    output[i,0] = x
                    output[i,1] = y
        else:
            if y >= 0:
                if math.fabs(x) > .0001:
                    theta[i] = 3. * math.pi / 2. + math.atan(y / x)
                    radii[i] = np.sqrt((x*x + y*y))
                    output[i,0] = x
                    output[i,1] = y
                else:
                    theta[i] = -100
                    radii[i] = np.sqrt((x*x + y*y))
                    output[i,0] = x
                    output[i,1] = y
            else:
                if math.fabs(y) > .0001:
                    theta[i] = math.pi + math.atan(x/-y)
                    radii[i] = np.sqrt((x*x + y*y))
                    output[i,0] = x
                    output[i,1] = y
                else:
                    theta[i] = -100
                    radii[i] = np.sqrt((x*x + y*y))
                    output[i,0] = x
                    output[i,1] = y

    # Divide up the theta ranges
    theata_ranges = np.arange(0., 2.*math.pi, dtheta)
    theata_ranges = np.concatenate((theata_ranges, np.array([2*math.pi])))

    # Now we want to make the output, which are the xyz-points for each seam set.
    ordered_seams = []
    for sub_inds_list in indices_list:

        sub_xyz_pts = xyz_coords[sub_inds_list, :]  # Sub-set of indices we care about
        sub_theta = theta[sub_inds_list]  # The sub-set of theta values we care about
        ordered_xyz_points = np.zeros((theata_ranges.shape[0]-1, 3))
        indices_with_no_points = []

        for i in range(theata_ranges.shape[0]-1):

            # 'start' and 'stop' below define theta range over which we want to grab 3d points.
            start = theata_ranges[i]
            stop = theata_ranges[i+1]

            sub_inds = np.where((sub_theta >= start) & (sub_theta < stop))
            sub_sub_xyz_pts = sub_xyz_pts[sub_inds, :]  # Points within the delta-theta range
            sub_sub_xyz_pts = np.squeeze(sub_sub_xyz_pts, axis=0)  # For some reason an extra axis gets tacked on.

            if sub_sub_xyz_pts.shape[0] > 0:
                ordered_xyz_points[i, :] = np.mean(sub_sub_xyz_pts, axis=0)

            else:
                indices_with_no_points.append(i)

            # # Check for NAN values here, this could happen for a number of reasons but most likely there are no
            # # points in the theta bin.
            # if math.isnan(ordered_xyz_points[i, 0]) or math.isnan(ordered_xyz_points[i, 1]) or math.isnan(ordered_xyz_points[i, 2]):
            #     # print("SEAM DISCRIMINATOR: Found some NANs in the centroids: " + str(ordered_xyz_points[i, :]))
            #
            #     print(sub_sub_xyz_pts.shape)

        # Now kill the parts without points and get rid of those NANs
        ordered_xyz_points = np.delete(ordered_xyz_points, indices_with_no_points, axis=0)

        ordered_seams.append(ordered_xyz_points)

    return ordered_seams, theta, radii, output

def plot_3D_seams(indices_list, xyz_coords):
    """
    This plots out segmented seams. That is given all the x-y-z coordinates, and then a list of lists, where each
    sub-list has all the indices of the x-y-z points that constitute a seam, each seam will be plotted out.
    :param indices_list: list of list, each sublist has integer indices
    :param xyz_coords: np.array, nX3
    :return: N/A will plot out stuff
    """

    # Make 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Colors and markers available
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'r', 'g', 'b', 'c', 'm', 'y']
    markers = ['^', '^', '^', '^', '^', '^', 'o', 'o', 'o', 'o', 'o', 'o']
    n_colors = len(colors)

    # pull out x, y, z values individually
    xv = xyz_coords[:, 0]
    yv = xyz_coords[:, 1]
    zv = xyz_coords[:, 2]

    i = 0
    for sub_inds in indices_list:
        if i == n_colors:
            i = 0
        ax.scatter(xv[sub_inds], yv[sub_inds], zv[sub_inds], c=colors[i], marker=markers[i])
        i += 1

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
    
def plot_3D_ordered_seams(xyz_coords_list):
    """
    This will plot out the ....
    :param xyz_coords_list:
    :return:
    """

    # Make 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Colors and markers available
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'r', 'g', 'b', 'c', 'm', 'y']
    markers = ['^', '^', '^', '^', '^', '^', 'o', 'o', 'o', 'o', 'o', 'o']
    n_colors = len(colors)

    i = 0
    for xyz_array in xyz_coords_list:
        if i == n_colors:
            i = 0

        print("\nPlotting method, looking at z-values ... ")
        print(xyz_array[:, 2])
        ax.plot(xyz_array[:, 0], xyz_array[:, 1], xyz_array[:, 2], c=colors[i], marker=markers[i])
        i += 1

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
    
    
if __name__ == "__main__":

    print("\nReading coordinates")
    start_time = time.time()
    # file_name = '/home/atlas2/LonsEngRepos/glugun_sensor_python/src/Alpha/findSeamsInClouds/just_seams.xyz'
    file_name = '/home/andy/Documents/spline_fitting/one_seam.xyz'
    xyz_coords = read_xyz(file_name)
    total_run_time = time.time() - start_time
    print("... Time to read coordinates = " + str(total_run_time))
    print("... The shape of xyz_coords is " + str(xyz_coords.shape))

    # # TOO MANY lets drop some
    # spaces = np.arange(0, xyz_coords.shape[0], 15)
    # xyz_intermed = np.zeros((spaces.shape[0], 3))
    # for i, a_ind in enumerate(spaces):
    #     xyz_intermed[i, :] = xyz_coords[a_ind, :]
    # xyz_coords = xyz_intermed
    #
    # # http://scikit-learn.org/stable/modules/neighbors.html#unsupervised-nearest-neighbors
    # start_time = time.time()
    # nn = 25
    # print("\n\nGenerating n = " + str(nn) + " nearest neighbors")
    # nrbs = NearestNeighbors(n_neighbors=nn, algorithm='auto').fit(xyz_coords)
    # distances, indices = nrbs.kneighbors(xyz_coords)
    # total_run_time = time.time() - start_time
    # print("... Time to read find nearest neighbors = " + str(total_run_time))
    #
    #
    # #print("\n\nFinding seams ")
    # #array_of_connected_components = a_bs_seam_finder(distances, indices, minimum_distance=.005)
    # #print("... The unique seams are " + str(np.unique(array_of_connected_components)))
    #
    # print("\n\nGenerating Graph")
    # start_time = time.time()
    # indices_list = seam_separator(distances, indices, minimum_distance=.010)
    # total_run_time = time.time() - start_time
    # print("... Time to read build graph = " + str(total_run_time))
    #
    # #plot_3D_seams(indices_list, xyz_coords)
    #
    # # Now discretize seams
    # ordered_seams = simple_radial_seam_discretizer(indices_list, xyz_coords, dtheta=.05)
    #
    # #
    # reorder_seams_by_z(ordered_seams)

    #
    ordered_seams, thetas, radii, output = full_seam_finder_0(xyz_coords, decimation=700, num_of_neighbs=25, minimum_distance=.010, dtheta=.05,
                                       z_order_asscending=True)

    #print(sorted(thetas))
    radii
    print(thetas.shape)
    print(radii.shape)
    print(xyz_coords.shape)
    print(output.shape)
    
    #print(sorted(thetas))
    print(radii)
    thetas = np.expand_dims(thetas, axis=1)
    print(thetas.shape)
    
    total = np.concatenate((thetas,output), axis=1)
    #print(total)
    total = total[total[:,0].argsort()]
    print(total)
    
    print(total[:,1:3])
    
    tck, u = splprep(total[:,1:3],s=0)
    
    #cs = CubicSpline(total[:,0], total[:,1:3], bc_type='periodic')
    #print("ds/dx={:.1f} ds/dy={:.1f}".format(cs(0, 1)[0], cs(0, 1)[1]))
    #ds/dx=0.0 ds/dy=1.0
    #xs = 2 * np.pi * np.linspace(0, 1, 100)
    #plt.figure(figsize=(6.5, 4))
    #plt.plot(y[:, 0], y[:, 1], 'o', label='data')
    #plt.plot(np.cos(xs), np.sin(xs), label='true')
    #plt.plot(cs(xs)[:, 0], cs(xs)[:, 1], label='spline')
    #plt.axes().set_aspect('equal')
    #lt.legend(loc='center')
    #plt.show()
    #
    #ellipse_list = fitting_ellipse(ordered_seams)

    #
    plot_3D_ordered_seams(ordered_seams)
    
    
    