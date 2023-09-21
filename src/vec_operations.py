#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 23:07:28 2020

@author: emilia
"""
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial import ConvexHull
from scipy.linalg import orthogonal_procrustes


def make_vec(point):
    """
    Turn a point into a vector, anchored at the origin.

    :param np.ndarray point: point coordinates (n, )
    :return: vec: (2, n)
    """
    dim = point.shape[0]
    vec = np.stack((np.zeros((dim,)), point))
    return vec


def make_point(vec):
    """
    Turn a vector anchored at the origin into a point.
    :param np.ndarray vec: (2, n)
    :return: point (n, )
    """
    point = vec[-1, :]
    return point


def get_vec_from_points(point1, point2):
    """
    Calculate a vector that joins 2 points.

    :param np.array point1: point 1 coordinates (n, )
    :param np.array point2: point 2 coordinates (n, )
    :return: vec (n, )
    """
    assert point2.shape == point2.shape, 'Both points must have the same number of dimensions.'
    vec = point2 - point1
    return vec


def get_projection(point, plane_vecs):
    """
    Project a point onto a plane defined by plane_vecs.

    Parameters
    ----------
    point : np.ndarray
        Point coordinates (n_dims, ).
    plane_vecs : np.ndarray
        Vectors defining a plane (2, n_dims).

    Returns
    -------
    point_proj : np.ndarray
        Coordinates of the point projected onto the plane.

    """
    # get plane normal
    normal = np.cross(plane_vecs[0, :], plane_vecs[1, :])

    # get point coordinates in the new basis, defined by the plane and its normal
    new_basis = np.concatenate((plane_vecs, normal[None, :]))
    point_new_coords = new_basis @ point

    norm_scale = point_new_coords[-1]  # scale the normal by the point coordinate along that direction

    # find the projected coordinates
    point_proj = point - norm_scale * normal

    return point_proj


def sort_by_path_length(verts):
    """
    Sort a set of vertices to form a quadrilateral by the pairwise distances between them.

    :param np.ndarray verts: vertices (4, n)
    :return: sorted_verts, sorting_order
    """
    if verts.shape[0] != 4:
        raise NotImplementedError('Only implemented for a set of 4 vertices.')

    sorted_verts = np.zeros((verts.shape))
    n_verts = verts.shape[0]
    sorting_order = np.zeros((n_verts,), dtype=int)

    # get pairwise euclidean distances
    distances = euclidean_distances(verts)

    # find points with the largest distance between and set them to be opposite corners of a quadrilateral
    sorting_order[0] = np.where(distances == np.max(distances))[0][0]
    sorting_order[2] = np.where(distances == np.max(distances))[1][0]

    # sort the remaining indices by distances
    curr_point = sorting_order[0]
    possible_neighbours = np.setdiff1d(range(4), sorting_order[[0, 2]])
    sorted_neighbours = np.argsort(distances[possible_neighbours, curr_point])
    sorting_order[1] = possible_neighbours[sorted_neighbours[0]]
    sorting_order[3] = possible_neighbours[sorted_neighbours[1]]

    for i in range(n_verts):
        sorted_verts[i, :] = verts[sorting_order[i], :]

    return sorted_verts, sorting_order


def quadrilat_area(coords):
    """
    Calculate the area of a quadrilateral from its (3D) vertex coordinates.

    :param np.ndarray coords: Coordinate array, shape: (n_colour_bins, 3)
    :return: area
    """
    # get diagonals
    ac = coords[2, :] - coords[0, :]
    bd = coords[3, :] - coords[1, :]

    # calculate area
    area = np.linalg.norm(np.cross(ac, bd)) / 2
    return area


def construct_rot_matrix(u, theta):
    """
    Construct a rotation matrix from a rotation axis u and a rotation angle theta.
    :param np.ndarray u: rotation axis in 3D (3, )
    :param float theta: rotation angle
    :return: R: rotation matrix
    """
    u_x = construct_x_prod_matrix(u)
    R = np.cos(theta) * np.eye(3) + np.sin(theta) * u_x + (1 - np.cos(theta)) * np.outer(u, u)
    return R


def construct_x_prod_matrix(u):
    u_x = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return u_x


def rotate_plane_by_angle(points, theta_degrees, axis='z'):
    """
    Rotates a set of points / a plane by a given degree around a given axis in 3D. Default axis is z.

    :param np.ndarray points: 3D coordinates of the datapoints, shape: (n_points, 3)
    :param float theta_degrees: angle of rotation
    :param str axis: Optional. Axis around which to rootate the points. Choose from: 'x', 'y', 'z'. Default is 'z'.
    :return points_rotated: Coordinates of the rotated datapoints (n_points, 3)
    """

    theta_radians = np.radians(theta_degrees)
    R = np.eye(3)
    if axis == 'z':
        # construct rotation matrix - z component stays the same
        R[:, 0] = np.array([np.cos(theta_radians), np.sin(theta_radians), 0])
        R[:, 1] = np.array([-np.sin(theta_radians), np.cos(theta_radians), 0])
    elif axis == 'x':
        # construct rotation matrix - x component stays the same
        R[:, 1] = np.array([0, np.cos(theta_radians), np.sin(theta_radians)])
        R[:, 2] = np.array([0, -np.sin(theta_radians), np.cos(theta_radians)])
    elif axis == 'y':
        # construct rotation matrix - y component stays the same
        R[:, 0] = np.array([np.cos(theta_radians), 0, -np.sin(theta_radians)])
        R[:, 2] = np.array([np.sin(theta_radians), 0, np.cos(theta_radians)])

    # note in the case of a 180-degree rotation, there will be a small numerical error for sin - the value won't be
    # exactly 0 (but that's ok)

    # apply rotation matrix to datapoints
    points_rotated = R @ points.T

    return points_rotated


def procrustes(data1, data2):
    """
    Procrustes analysis, a similarity test for two data sets. Each input matrix is a set of points or vectors (the rows
    of the matrix). The dimension of the space is the number of columns of each matrix. Given two identically sized
    matrices, procrustes standardizes both such that:
    - :math:`tr(AA^{T}) = 1`.
    - Both sets of points are centered around the origin.
    Procrustes ([1]_, [2]_) then applies the optimal transform to the second
    matrix (including scaling/dilation, rotations, and reflections) to minimize
    :math:`M^{2}=\sum(data1-data2)^{2}`, or the sum of the squares of the
    pointwise differences between the two input datasets.
    This function was not designed to handle datasets with different numbers of
    datapoints (rows).  If two data sets have different dimensionality
    (different number of columns), simply add columns of zeros to the smaller
    of the two.

    This code is copied from the scipy.spatial package, with the only change being that it additionally returns the
    rotation matrix R.

    Parameters
    ----------
    data1 : array_like
        Matrix, n rows represent points in k (columns) space `data1` is the
        reference data, after it is standardised, the data from `data2` will be
        transformed to fit the pattern in `data1` (must have >1 unique points).
    data2 : array_like
        n rows of data in k space to be fit to `data1`.  Must be the  same
        shape ``(numrows, numcols)`` as data1 (must have >1 unique points).
    Returns
    -------
    mtx1 : array_like
        A standardized version of `data1`.
    mtx2 : array_like
        The orientation of `data2` that best fits `data1`. Centered, but not
        necessarily :math:`tr(AA^{T}) = 1`.
    disparity : float
        :math:`M^{2}` as defined above.
    R : array_like
        Rotation matrix that maps mtx1 to mtx2
    Raises
    ------
    ValueError
        If the input arrays are not two-dimensional.
        If the shape of the input arrays is different.
        If the input arrays have zero columns or zero rows.
    See Also
    --------
    scipy.linalg.orthogonal_procrustes
    scipy.spatial.distance.directed_hausdorff : Another similarity test
      for two data sets
    Notes
    -----
    - The disparity should not depend on the order of the input matrices, but
      the output matrices will, as only the first output matrix is guaranteed
      to be scaled such that :math:`tr(AA^{T}) = 1`.
    - Duplicate data points are generally ok, duplicating a data point will
      increase its effect on the procrustes fit.
    - The disparity scales as the number of points per input matrix.
    References
    ----------
    .. [1] Krzanowski, W. J. (2000). "Principles of Multivariate analysis".
    .. [2] Gower, J. C. (1975). "Generalized procrustes analysis".
    Examples
    --------
    The matrix ``b`` is a rotated, shifted, scaled and mirrored version of
    ``a`` here:
    #>>> a = np.array([[1, 3], [1, 2], [1, 1], [2, 1]], 'd')
    #>>> b = np.array([[4, -2], [4, -4], [4, -6], [2, -6]], 'd')
    #>>> mtx1, mtx2, disparity = procrustes(a, b)
    #>>> round(disparity)
    0.0
    """
    mtx1 = np.array(data1, dtype=np.double, copy=True)
    mtx2 = np.array(data2, dtype=np.double, copy=True)

    if mtx1.ndim != 2 or mtx2.ndim != 2:
        raise ValueError("Input matrices must be two-dimensional")
    if mtx1.shape != mtx2.shape:
        raise ValueError("Input matrices must be of same shape")
    if mtx1.size == 0:
        raise ValueError("Input matrices must be >0 rows and >0 cols")

    # translate all the data to the origin
    mtx1 -= np.mean(mtx1, 0)
    mtx2 -= np.mean(mtx2, 0)

    norm1 = np.linalg.norm(mtx1)
    norm2 = np.linalg.norm(mtx2)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("Input matrices must contain >1 unique points")

    # change scaling of data (in rows) such that trace(mtx*mtx') = 1
    mtx1 /= norm1
    mtx2 /= norm2

    # transform mtx2 to minimize disparity
    R, s = orthogonal_procrustes(mtx1, mtx2)
    mtx2 = np.dot(mtx2, R.T) * s

    # measure the dissimilarity between the two datasets
    disparity = np.sum(np.square(mtx1 - mtx2))

    return mtx1, mtx2, disparity, R, s
