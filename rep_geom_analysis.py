#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 17:30:57 2021

@author: emilia
"""
import pickle
import os.path
import torch
import hypertools
import numpy as np
import pandas as pd
import statsmodels.api as sm
import vec_operations as vops 
from numpy.linalg import lstsq, inv
from scipy.stats import zscore, mode, pearsonr, spearmanr, shapiro


from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from scipy.spatial.distance import squareform, pdist
from scipy.linalg import orthogonal_procrustes
from scipy.spatial import ConvexHull
from scipy.stats import shapiro, ttest_1samp, wilcoxon


import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull

import helpers
import custom_plot as cplot
import pdb

from generate_data_vonMises import update_time_params

#%% define low-level funcs
def get_3D_coords(data):
    #% run first PCA to get down to 3D space
    
    # demean
    data -= data.mean()
    # Initialise PCA object
    pca = PCA(n_components=3) 
    # get coordinates in the reduced-dim space
    coords_3D = pca.fit_transform(data) 
    
    return pca, coords_3D


def get_best_fit_plane(data):
    """

    Parameters
    ----------
    data : numpy array 
        Array with the coordinates of points to which the plane will be fitted. Each point corresponds to a row.

    Returns
    -------
    fitted_plane : pca object
        Object created by the sklearn.decomposition.PCA method.
        fitted_plane.components_ gives the plane vectors

    """
    # center data
    data_centered = data - np.mean(data)
    
    # find plane of best fit using PCA from sklearn.decomposition
    pca = PCA(n_components=2)
    fitted_plane = pca.fit(data_centered)
    
    return fitted_plane


def phase_alignment_corr(data):
    '''
    Calculate the correlation between population responses to the same colour 
    shown at different locations.

    Parameters
    ----------
    pca_data : TYPE
        DESCRIPTION.

    Returns
    -------
    result : TYPE
        DESCRIPTION.

    '''
    
    n_colours = len(data)//2
    
    result = {'r':np.empty((n_colours,)),
              'p':np.empty((n_colours,)),
              'test':np.empty((n_colours,),dtype='str')}
    
    for n in range(n_colours):        
        l1_centered = data[n,:] - data[:n_colours,:].mean(0)
        l2_centered = data[n+n_colours,:] - data[n_colours:,:].mean(0)
        
        s1,p1 = shapiro(l1_centered)
        s2,p2 = shapiro(l2_centered)
        
        if np.logical_and(p1>= .05, p2 >= .05):
            result['test'][n] = 'pearson'
            result['r'][n],result['p'][n] = pearsonr(l1_centered,l2_centered)
        else:
            result['test'][n] = 'spearman'
            result['r'][n],result['p'][n] = spearmanr(l1_centered,l2_centered)   
    
    return result


def run_pca_pipeline(constants,data,subspace_order):
    
    n_colours = constants.PARAMS['B']
    
    # do PCA 1
    pca, coords_3D = get_3D_coords(data)
    
    # do PCA 2
    plane1 = get_best_fit_plane(coords_3D[:n_colours,:])
    plane2 = get_best_fit_plane(coords_3D[n_colours:,:])
    
    # get angle between planes and phase alignment
    # theta = get_angle_between_planes_corrected(coords_3D,
    #                                            plane1.components_,
    #                                            plane2.components_)
    
    theta, pa = get_angle_and_phase_between_planes_corrected(coords_3D,
                                                plane1.components_,
                                                plane2.components_)
    
    
    
    
    # make data structure
    subspace = {}
    subspace['binned_data'] = data
    subspace['binned_data_subspace_order'] = subspace_order
    subspace['pca'] = pca
    subspace['3Dcoords'] = coords_3D
    subspace['plane1'] = plane1
    subspace['plane2'] = plane2
    subspace['theta'] = theta
    subspace['psi'] = pa
    
    return subspace

# def get_angle_between_planes(plane1_vecs,plane2_vecs):
#     """
#     Calculate the angle between two planes as each defined by two vectors.

#     Parameters
#     ----------
#     plane1_vecs : numpy array with plane1 vectors as rows
        
#     plane2_vecs : numpy array with plane2 vectors as rows

#     Returns
#     -------
#     angle_degrees: angle between planes in degrees

#     """
#     # calculate normals for each plane
#     normal1 = np.cross(plane1_vecs[0,:],plane1_vecs[1,:])
#     normal2 = np.cross(plane2_vecs[0,:],plane2_vecs[1,:])
    
#     # since each normal has a length of 1, their dot product will be equal to the cosine of the angle between them
#     cos_theta = np.dot(normal1,normal2)
    
#     #fix the numerical error if abs(np.dot) > 1
#     if cos_theta > 1:
#         cos_theta = 1
#     elif cos_theta < -1:
#         cos_theta = -1
#     # get the plane angle - in degrees for convenience
#     # arccos will only ever give angles in the [0,180] degree range
#     # in order to be able to use circular statistics that assume a uniform
#     # distribution as the null, need to fix this so that the angle between 
#     # planes is directional (and spans the whole circle)
    
    # # the solution is the following:
    #     # define a plane spanned by normal1 and normal2
    #     # calculate the normal of that plane (normal3)
    #     # determine the triple product : normal1 . (normal2 x normal3)
    #     # if the above is positive - angle will be arccos(normal1 . normal2)
    #     # otherwise, angle will be 360 - arccos(normal1 . normal2)
    # # normal3 = np.cross(normal1,normal2)
    # # triple_product = np.dot(normal1,np.cross(normal2,normal3))
    
    # # if triple_product >= 0:
    # #     angle_degrees = np.degrees(np.arccos(cos_theta))
    # # else:
    # #     angle_degrees = 360 - np.degrees(np.arccos(cos_theta))
    # angle_degrees = np.degrees(np.arccos(cos_theta))
    # return angle_degrees
   
# def get_angle_between_planes(plane1_vecs,plane2_vecs):
#     """
#     Calculate the angle between two planes as each defined by two vectors.

#     Parameters
#     ----------
#     plane1_vecs : numpy array with plane1 vectors as rows
        
#     plane2_vecs : numpy array with plane2 vectors as rows

#     Returns
#     -------
#     angle_degrees: angle between planes in degrees

#     """
#     # calculate normals for each plane
#     normal1 = np.cross(plane1_vecs[0,:],plane1_vecs[1,:])
#     normal2 = np.cross(plane2_vecs[0,:],plane2_vecs[1,:])
    
#     # since each normal has a length of 1, their dot product will be equal to the cosine of the angle between them
#     cos_theta = np.dot(normal1,normal2)
    
#     #fix the numerical error if abs(np.dot) > 1
#     if cos_theta > 1:
#         cos_theta = 1
#     elif cos_theta < -1:
#         cos_theta = -1
#     # get the plane angle - in degrees for convenience
#     # arccos will only ever give angles in the [0,180] degree range
#     # in order to be able to use circular statistics that assume a uniform
#     # distribution as the null, need to fix this so that the angle between 
#     # planes is directional (and spans the whole circle)
    
#     # the solution is the following:
#         # define a plane spanned by normal1 and normal2
#         # calculate the normal of that plane (normal3)
#         # determine the triple product : normal1 . (normal2 x normal3)
#         # if the above is positive - angle will be arccos(normal1 . normal2)
#         # otherwise, angle will be 360 - arccos(normal1 . normal2)
#     # normal3 = np.cross(normal1,normal2)
#     # triple_product = np.dot(normal1,np.cross(normal2,normal3))
    
#     # if triple_product >= 0:
#     #     angle_degrees = np.degrees(np.arccos(cos_theta))
#     # else:
#     #     angle_degrees = 360 - np.degrees(np.arccos(cos_theta))
#     angle_degrees = np.degrees(np.arccos(cos_theta))
    
#     return angle_degrees

def align_plane_vecs(points,ixs, plane_vecs):
    """
    Align the plane-defining vectors obtained from PCA with two sides of the 
    quadrilateral defined by the datapoints. This correction is necessary for 
    comparing plane angles. This is because PCA returns the vectors 
    corresponding to the directions of maximum variance in the data. If these
    directions differ between two planes being considered (i.e. correspond to
    different sides of the rectangle defined by the points), the angle calculated
    for parallel planes might not be 0 degrees, but 180. Putting the plane-
    defining vectors in a common frame of reference circumvents this issue and 
    makes interpretation of the angles more straightforward. For more details 
    see the plane_fitting.py file.

    Parameters
    ----------
    points : (n_points,3) numpy array with the 3D coordinates of the 
            datapoints the plane was fitted to
        
    pca : sklearn object corresponding to the best-fit plane

    Returns
    -------
    plane_vecs_aligned: (2,3) an array with two orthogonal vectors defining
        the best-fitting plane. The first vector corresponds to the side of the
        quadrilateral joining points 1 and 2, and the second vector is at a 
        90 degree angle from the first one.
    """ 
    
    # project datapoints onto the fitted plane
    points_proj = np.zeros(points.shape)   
    com = np.mean(points, axis=0) # centre of mass
    for i in range(points_proj.shape[0]):
        points_proj[i,:] = vops.getProjection(points[i,:]-com,plane_vecs)
    
    
    # get the vectors corresponding to the sides of the parallelogram
    # these will be the new plane-defining bases
    a = vops.getVecFromPoints(points_proj[ixs[0],:],points_proj[ixs[1],:])
    b = vops.getVecFromPoints(points_proj[ixs[2],:],points_proj[ixs[3],:])
    
    # normalise them
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    
    # change basis to the one defined by the plane + its normal
    # this is so that the vector(s) can be rotated around the normal with
    # rotate_plane_by_angle(), which assumes that the z-component of the vector(s) 
    # is 0
    plane_basis = np.concatenate((plane_vecs,
                                  np.cross(plane_vecs[0,:],plane_vecs[1:,])))
    
    a_newBasis = plane_basis @ a
    b_newBasis = plane_basis @ b
    
    # force the plane-defining vectors to be orthogonal
    if np.abs(np.dot(a,b)) > 0.001:
        # if they're not already - rotate vector a by 90 degrees 
        angle = np.arccos(np.dot(a_newBasis,b_newBasis))
        angle_diff = np.degrees(np.pi/2 - angle)
        
        # rotate
        tmp = rotate_plane_by_angle(b_newBasis,-angle_diff)
        if np.abs(np.dot(a_newBasis,tmp)) < 0.001:
            b_newBasis = tmp
        else:
            # rotate the other way
            tmp = rotate_plane_by_angle(b_newBasis,angle_diff)
            #double check that orthogonal now
            
            if np.abs(np.dot(a_newBasis,tmp)) > 0.001:
                raise ValueError('New vectors still not orthogonal')
            else:
                b_newBasis = tmp
        

    # return to original (standard) basis
    b = plane_basis.T @ b_newBasis
    
    
    # double check that the new vectors in the standard basis *are* orthogonal
    if np.abs(np.dot(a,b)) > 0.001:
        print(np.degrees(np.dot(a,b)))
        raise ValueError('New vectors not orthogonal')
        

    plane_vecs_aligned = np.stack((a,b))
        
    return plane_vecs_aligned


def get_angle_between_planes_corrected(points,plane1_vecs,plane2_vecs):
    """
    Calculate the angle between two planes as each defined by two vectors.

    Parameters
    ----------
    plane1_vecs : numpy array with plane1 vectors as rows
        
    plane2_vecs : numpy array with plane2 vectors as rows

    Returns
    -------
    angle_degrees: angle between planes in degrees

    """
    # get the aligned plane vectors
    # indices of neighbouring sides of the data quadrilateral
    n_colours = len(points)//2
    points1 = points[:n_colours,:] # location 1 datapoints
    points2 = points[n_colours:,:] # location 2
    ixs_all = np.array([[0,1,0,-1],[1,0,1,2],[2,1,2,3],[3,2,3,0]]) 
    # indices of neighbouring sides of the parallelogram in the format 
    # [common vertex,x,common vertex,y]
    cos_theta_all = np.empty((len(ixs_all),))
    sign_theta_all = np.empty((len(ixs_all),))
    for combo in range(len(ixs_all)):
        plane_vecs_aligned1 = align_plane_vecs(points1,ixs_all[combo], plane1_vecs)
        plane_vecs_aligned2 = align_plane_vecs(points2,ixs_all[combo], plane2_vecs)

        # calculate normals for each plane
        normal1 = np.cross(plane_vecs_aligned1[0,:],plane_vecs_aligned1[1,:])
        normal2 = np.cross(plane_vecs_aligned2[0,:],plane_vecs_aligned2[1,:])
        
        # define the sign of the angle
        # this is an arbitrary distinction, but necessary to do circular statistics
        # at the group level. Arccos will always map angles to the [0,180] range,
        # whereas we want them to span the full circle. This rectification 
        # will also mean that the mean angle across group will never be 0.
        # Sign is determined based on thenormnal of plane 1 - if the z coordinate 
        # is >= 0, it is positive, otherwise - negative.
        if normal1[-1]>=0:
            sign_theta_all[combo] = 1
        else:
            sign_theta_all[combo] = -1
        # since each normal has a length of 1, their dot product will be equal
        # to the cosine of the angle between them
        cos_theta_all[combo] = np.dot(normal1,normal2)
    
    
    #fix the numerical error if abs(np.dot) > 1
    cos_theta_all[np.where(cos_theta_all > 1)] = 1
    cos_theta_all[np.where(cos_theta_all < -1)] = -1
    
    # get the plane angle - in degrees for convenience
    angle_degrees_all = np.degrees(np.arccos(cos_theta_all))
    angle_degrees_all *= sign_theta_all
    # get the mode of all comparisons to estimate the final angle
    # this is necessary because some distortions of geometry, whereby the 
    # data quadrilateral is concave can give rise to an angle of 180 degrees
    # depending on the side of the quadrilateral that is used as the plane-defining
    # vector, even though the overall geometries are isometric w.r.t. to one 
    # another and the ground truth. Taking the mode will remove this effect.
    angle_degrees = mode(angle_degrees_all)[0][0]    
    
    return angle_degrees


def procrustes(data1, data2):
    """Procrustes analysis, a similarity test for two data sets.
    Each input matrix is a set of points or vectors (the rows of the matrix).
    The dimension of the space is the number of columns of each matrix. Given
    two identically sized matrices, procrustes standardizes both such that:
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
    
    This code is copied from the scipy.spatial package, the only change is that it 
    additionally returns the rotation matrix R
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
    >>> from scipy.spatial import procrustes
    The matrix ``b`` is a rotated, shifted, scaled and mirrored version of
    ``a`` here:
    >>> a = np.array([[1, 3], [1, 2], [1, 1], [2, 1]], 'd')
    >>> b = np.array([[4, -2], [4, -4], [4, -6], [2, -6]], 'd')
    >>> mtx1, mtx2, disparity = procrustes(a, b)
    >>> round(disparity)
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

    return mtx1, mtx2, disparity, R,s


def change_basis(points):
    n_points = len(points)//2
    # find basis for plane 1
    plane1 = get_best_fit_plane(points[:n_points,:])
    normal1 = np.cross(plane1.components_[0,:],plane1.components_[1,:])
    
    new_basis = np.stack((plane1.components_[0,:],
                                plane1.components_[1,:],
                                normal1),axis=0).T
    # express datapoint coordinates in the new basis
    new_points = (new_basis.T @ points.T).T
    
    return new_points


def detect_concave_quadr(points_2D):
    if len(points_2D) != 4:
        raise ValueError('4 vertices needed')
    hull = ConvexHull(points_2D)
    n_vertices = len(hull.vertices)
    missing_vtx = np.setdiff1d(np.arange(4),hull.vertices)
    is_concave = n_vertices < 4
    return is_concave, missing_vtx
    

def detect_bowtie(points):
    # compress to 2D
    points_2D = compress_to_2D(points)
    
    # indices of the diagonal vertices
    diagonal_ixs = np.array([[0,2],[1,3]])
    # find the lines demarcated by the diagonals
    m1 = np.diff(points_2D[diagonal_ixs[0,:],1])/np.diff(points_2D[diagonal_ixs[0,:],0])
    m2 = np.diff(points_2D[diagonal_ixs[1,:],1])/np.diff(points_2D[diagonal_ixs[1,:],0])
    c1 = points_2D[diagonal_ixs[0,0],1] - m1*points_2D[diagonal_ixs[0,0],0] 
    c2 = points_2D[diagonal_ixs[1,0],1] - m2*points_2D[diagonal_ixs[1,0],0]
    
    # find the point where they cross
    x = (c2-c1)/(m1-m2)
    y = m1*x + c1
    
    # check if it lies between the ends of the two diagonals
    x_vs_d1 = np.logical_and(x >= points_2D[diagonal_ixs[0,:],0].min(),
                             x <= points_2D[diagonal_ixs[0,:],0].max())
    y_vs_d1 = np.logical_and(y >= points_2D[diagonal_ixs[0,:],1].min(),
                             y <= points_2D[diagonal_ixs[0,:],1].max())
    x_vs_d2 = np.logical_and(x >= points_2D[diagonal_ixs[1,:],0].min(),
                             x <= points_2D[diagonal_ixs[1,:],0].max())
    y_vs_d2 = np.logical_and(y >= points_2D[diagonal_ixs[1,:],1].min(),
                             y <= points_2D[diagonal_ixs[1,:],1].max())
    is_centre_within = np.all([x_vs_d1,x_vs_d2,y_vs_d1,y_vs_d2])
    
    # check if the datapoints can form a convex quadrilalteral
    is_concave,_= detect_concave_quadr(points_2D)
    
    is_bowtie = np.logical_and(not is_centre_within,not is_concave)
    
    return is_bowtie


def calc_phase_alignment(points_coplanar):
    n_points = len(points_coplanar)//2
    # change of basis to that defined by plane 1:
    # to zero-out the z-coordinate and reduce the problem to 2D
    points_new = change_basis(points_coplanar)
    # fix precision errors and throw out z-axis
    if points_new[:,-1].sum().round(6) != 0:
        raise ValueError('Change of basis did not null the z-coordinates')
    points_new = points_new.round(6)[:,:2]
    # apply the procrustes analysis
    plane1_std, plane2_std, disparity,R,s = procrustes(points_new[:n_points,:], 
                                                    points_new[n_points:,:])
    
    reflection = np.linalg.det(R)<0
    bowtie1 = detect_bowtie(points_coplanar[:n_points,:])
    bowtie2 = detect_bowtie(points_coplanar[n_points:,:])
    if np.any([reflection,bowtie1,bowtie2]):
        # estimating phase alignment doesn't make sense
        pa = np.nan
    else:
        pa = -np.degrees(np.arctan2(R[1,0],R[0,0]))
    
    return pa, reflection


def construct_x_prod_matrix(u):
    u_x = np.array([[0,-u[2],u[1]],[u[2],0,-u[0]],[-u[1],u[0],0]])
    return u_x


def construct_rot_matrix(u,theta):
    u_x = construct_x_prod_matrix(u)
    R = np.cos(theta)*np.eye(3) + np.sin(theta)*u_x + (1-np.cos(theta))*np.outer(u,u)
    return R


def make_coplanar(points_centered,normal1,normal2):
    # get the plane-defining vectors for both planes
    # this will be problematic if geometries distorted and mismatch in terms 
    # of the 'longest' side - need to port the x,y and normal directions from 
    # the angle correction function
    
    n_points = points_centered.shape[0]//2
    
    if np.logical_and(np.all(normal1==None),np.all(normal2==None)):
        plane1 = get_best_fit_plane(points_centered[:n_points,:])
        plane2 = get_best_fit_plane(points_centered[n_points:,:])
            
        #calculate the normals
        normal1 = np.cross(plane1.components_[0,:],plane1.components_[1,:])
        normal2 = np.cross(plane2.components_[0,:],plane2.components_[1,:])
    
    # get the rotation axis and angle
    # if planes already co-planar, leave them as they are - need a workaround 
    # because cross-product is not defined then (i.e. it's 0)
    if np.abs(np.dot(normal1,normal2).round(6))==1:
        rot_angle=0
        rot_axis=[]
        R = np.eye(len(normal1))
    else:
        rot_angle = np.arccos(np.dot(normal1,normal2))
        # print(np.dot(normal1,normal2))
        if rot_angle > np.pi/2:
            rot_angle += np.pi
        
        # still to add here - sign of the angle
        rot_axis = np.cross(normal1,normal2)/np.linalg.norm(np.cross(normal1,normal2))
        
        # rot_vec = rot_angle * rot_axis
        # R = r.from_rotvec(rot_vec)
        # construct the rotation matrix
        R = construct_rot_matrix(rot_axis,rot_angle)
        
    # apply to plane2 points
    plane2_points = R.T @ points_centered[n_points:,:].T
    # plane2_points = R.apply(points_centered[n_points:,:],inverse=True)
    
    # get new coordinates
    points_new = np.concatenate((points_centered[:n_points,:],plane2_points.T),axis=0)
    # points_new = np.concatenate((points_centered[:n_points,:],plane2_points),axis=0)

    return points_new


def get_phase_alignment(points,normal1,normal2):
    n_points = len(points)//2
    
    # center datapoints
    plane1_points = points[:n_points,:] - points[:n_points,:].mean(0)
    plane2_points = points[n_points:,:] - points[n_points:,:].mean(0)
    
    # get plane_defining vecs from pca
    plane1_vecs = get_best_fit_plane(plane1_points).components_
    plane2_vecs = get_best_fit_plane(plane2_points).components_
    
    # project the datapoints onto their corresponding planes
    plane1_points_p = np.stack([vops.getProjection(plane1_points[p,:],plane1_vecs) for p in range(n_points)])
    plane2_points_p = np.stack([vops.getProjection(plane2_points[p,:],plane2_vecs) for p in range(n_points)])
    
    # center so that mean of each plane is at the origin
    plane1_points_c = plane1_points_p - plane1_points_p.mean(0)
    plane2_points_c = plane2_points_p - plane2_points_p.mean(0)
    
    all_points = np.concatenate((plane1_points_c,plane2_points_c),axis=0)
    # makes planes coplanar
    # by rotating plane 2 to be co-planar with plane 1
    points_coplanar = make_coplanar(all_points,normal1,normal2)
    
    # calculate the phase-alignment using procrustes analysis
    pa,reflection = calc_phase_alignment(points_coplanar)
    
    return pa


def compress_to_2D(points):
    n_points = len(points)
    # find plane vectors
    plane_vecs = get_best_fit_plane(points).components_
    # get projection onto a plane
    points_projected = \
        np.stack([vops.getProjection(points[p,:],plane_vecs) for p in range(n_points)])
    #change_basis to that defined by plane vecs and normal
    normal = np.cross(plane_vecs[0,:],plane_vecs[1,:])
    new_basis = np.concatenate((plane_vecs,normal[None,:]),axis=0).T
    points_newBasis = (new_basis.T @ points_projected.T).T
    
    if points_newBasis[:,-1].round(6).sum() != 0:
        # round to 6 digits to avoid numerical precision errors
        raise ValueError('Change of basis failed to get rid of the z-component')
    # get rid of the z-coord
    points_2D = points_newBasis[:,:2]
    return points_2D
    
    
# def get_plane_bases_corrected(points,vtcs,plane1_vecs,plane2_vecs):
#     # get the aligned plane vectors
#     # indices of neighbouring sides of the data quadrilateral
#     n_colours = len(points)//2
#     dim = points.shape[1]
#     points1 = points[:n_colours,:] # location 1 datapoints
#     points2 = points[n_colours:,:] # location 2
#     ixs_all = np.array([[0,-1,0,1],[1,0,1,2],[2,1,2,3],[3,2,3,0]]) 
#     # indices of neighbouring sides of the parallelogram in the format 
#     # [common vertex,x,common vertex,y]
#     plane1_bases = np.empty((dim,dim,len(ixs_all)))
#     plane2_bases = np.empty((dim,dim,len(ixs_all)))
#     for combo in range(len(ixs_all)):
#         plane_vecs_aligned1 = align_plane_vecs(points1,ixs_all[combo], plane1_vecs)
#         plane_vecs_aligned2 = align_plane_vecs(points2,ixs_all[combo], plane2_vecs)

#         # calculate normals for each plane
#         normal1 = np.cross(plane_vecs_aligned1[0,:],plane_vecs_aligned1[1,:])
#         normal2 = np.cross(plane_vecs_aligned2[0,:],plane_vecs_aligned2[1,:])
        
#         plane1_bases[:,:,combo] = np.concatenate((plane_vecs_aligned1,normal1[None,:]),axis=0).T
#         plane2_bases[:,:,combo] = np.concatenate((plane_vecs_aligned2,normal2[None,:]),axis=0).T
        
#     return plane1_bases, plane2_bases


def get_plane_bases_corrected(points,vtcs,plane1_vecs,plane2_vecs):
    # get the aligned plane vectors
    # indices of neighbouring sides of the data quadrilateral
    
    c_ix = np.where(np.diff(vtcs).min()==1)[0][0] # index of the corner
    ixs = [vtcs[c_ix],vtcs[c_ix+1],vtcs[c_ix],vtcs[c_ix-1]]
    # indices of neighbouring sides of the parallelogram in the format 
    # [corner vertex,side x,corner vertex,side y]
    
    n_colours = len(points)//2
    points1 = points[:n_colours,:] # location 1 datapoints
    points2 = points[n_colours:,:] # location 2

    plane_vecs_aligned1 = align_plane_vecs(points1,ixs, plane1_vecs)
    plane_vecs_aligned2 = align_plane_vecs(points2,ixs, plane2_vecs)

   
    # calculate normals for each plane
    normal1 = np.cross(plane_vecs_aligned1[0,:],plane_vecs_aligned1[1,:])
    normal2 = np.cross(plane_vecs_aligned2[0,:],plane_vecs_aligned2[1,:])
        
    plane1_basis = np.concatenate((plane_vecs_aligned1,normal1[None,:]),axis=0).T
    plane2_basis = np.concatenate((plane_vecs_aligned2,normal2[None,:]),axis=0).T
        
    return plane1_basis, plane2_basis


def get_angle_and_phase_between_planes_corrected(points,plane1_vecs,plane2_vecs):
    """
    Calculate the angle between two planes as each defined by two vectors.

    Parameters
    ----------
    plane1_vecs : numpy array
        plane1 vectors (from PCA, pre-aligment) as rows
        
    plane2_vecs : numpy array
        plane2 vectors (from PCA) as rows

    Returns
    -------
    angle_degrees: angle between planes in degrees
    pa_degrees: phase alignment between planes in degrees

    """
    # get the aligned plane vectors
    n_colours = len(points)//2
    # plane1_bases, plane2_bases = get_plane_bases_corrected(points,plane1_vecs,plane2_vecs)
    # n_bases = plane1_bases.shape[-1]
    # cos_theta_all = np.empty((n_bases,))
    # sin_theta_all = np.empty((n_bases,))
    # sign_theta_all = np.empty((n_bases,))
    # pa_all = np.empty((n_bases,))
    # pdb.set_trace()
    
    
    # compress points to 2D:
    points1_2D = compress_to_2D(points[:n_colours,:])
    points2_2D = compress_to_2D(points[n_colours:,:])
    # check if either of the geometries is concave
    is_concave_geom1,vtx1 = detect_concave_quadr(points1_2D)
    is_concave_geom2,vtx2 = detect_concave_quadr(points2_2D)
    
    if np.any([is_concave_geom1,is_concave_geom2]):
        # need to pick the sides of the quadrilaterals that don't form the 
        # concave angle to be the plane-defining vectors
        vtcs = \
            np.setdiff1d(np.arange(n_colours),np.concatenate((vtx1,vtx2)))
        if len(vtcs) < 3:
            #raise ValueError('Both geometries are concave, and at mismatching vertices.')
            # revert to the standard choice for vertices
            vtcs = np.array([0,1,3])
    else:
        vtcs = np.array([0,1,3])
    plane1_basis,plane2_basis = \
        get_plane_bases_corrected(points,vtcs,plane1_vecs,plane2_vecs)
    
    
    #calculate the angle between planes
    normal1 = plane1_basis[:,-1]
    normal2 = plane2_basis[:,-1]
     
    # since each normal has a length of 1, their dot product will be equal
    # to the cosine of the angle between them
    cos_theta = np.dot(normal1,normal2)
    
    # define the sign of the angle
    # this is an arbitrary distinction, but necessary to do circular statistics
    # at the group level. Arccos will always map angles to the [0,180] range,
    # whereas we want them to span the full circle. This rectification 
    # will also mean that the mean angle across group will never be 0.
    # Sign is determined based on thenormnal of plane 1 - if the z coordinate 
    # is >= 0, it is positive, otherwise - negative.
    if normal1[-1]>=0:
        sign_theta = 1
    else:
        sign_theta = -1
    
    # calculate phase alignment
    # if angle between planes is within +-[90,180] range, it means that the 
    # planes are mirror images and calculating phase alignment does not 
    # make sense - set pa to nan
    if cos_theta <= 0:
        pa_degrees = np.nan
    else:
        pa_degrees = get_phase_alignment(points,normal1,normal2)
    
    # get the plane angle - in degrees for convenience
    angle_degrees= np.degrees(np.arccos(cos_theta))
    angle_degrees *= sign_theta
    
    
   #     # since each normal has a length of 1, their dot product will be equal
   #     # to the cosine of the angle between them
   #     cos_theta_all[b] = np.dot(normal1,normal2)
    # for b in range(n_bases):
        
    #     # calculate the angle between planes
    #     normal1 = plane1_bases[:,-1,b]
    #     normal2 = plane2_bases[:,-1,b]
       
    #     # since each normal has a length of 1, their dot product will be equal
    #     # to the cosine of the angle between them
    #     cos_theta_all[b] = np.dot(normal1,normal2)
        
    #     # define the sign of the angle
    #     # this is an arbitrary distinction, but necessary to do circular statistics
    #     # at the group level. Arccos will always map angles to the [0,180] range,
    #     # whereas we want them to span the full circle. This rectification 
    #     # will also mean that the mean angle across group will never be 0.
    #     # Sign is determined based on thenormnal of plane 1 - if the z coordinate 
    #     # is >= 0, it is positive, otherwise - negative.
    #     if normal1[-1]>=0:
    #         sign_theta_all[b] = 1
    #     else:
    #         sign_theta_all[b] = -1
        
        
    #     # fix any numerical precision errors
    #     # cos_theta_all[b] = cos_theta_all[b].round(6)
    #     # sin_theta_all[b] = sin_theta_all[b].round(6)
        
    #     # calculate phase alignment
    #     # if angle between planes is within +-[90,180] range, it means that the 
    #     # planes are mirror images and calculating phase alignment does not 
    #     # make sense - set pa to nan
    #     if cos_theta_all[b] <= 0:
    #         pa_all[b] = np.nan
    #     else:
    #         pa_all[b] = get_phase_alignment(points,normal1,normal2)
            
    
    
    # # get the plane angle - in degrees for convenience
    # # angle_degrees_all = np.degrees(np.arccos(cos_theta_all))
    
    # # angle_degrees_all = np.degrees(np.arctan2(sin_theta_all,cos_theta_all))
    
    # # get the mode of all comparisons to estimate the final angle
    
    # # #fix the numerical error if abs(np.dot) > 1
    # # cos_theta_all[np.where(cos_theta_all > 1)] = 1
    # # cos_theta_all[np.where(cos_theta_all < -1)] = -1
    
    # # # get the plane angle - in degrees for convenience
    # angle_degrees_all = np.degrees(np.arccos(cos_theta_all))
    # angle_degrees_all *= sign_theta_all
    # # get the mode of all comparisons to estimate the final angle and phase
    # # this is necessary because some distortions of geometry, whereby the 
    # # data quadrilateral is concave can give rise to an angle of 180 degrees
    # # depending on the side of the quadrilateral that is used as the plane-defining
    # # vector, even though the overall geometries are isometric w.r.t. to one 
    # # another (and the ground truth). Taking the mode will remove this effect.
    # angle_degrees = mode(angle_degrees_all)[0][0]    
    # pa_degrees = mode(pa_all)[0][0] 
    return angle_degrees, pa_degrees

    
def rotate_plane_by_angle(points,theta_degrees,axis='z'):
    """
    Rotates a set of points / a plane by a given degree around the z-axis.


    Parameters
    ----------
    points : (n_points,3) numpy array with the 3D coordinates of the 
            datapoints
        
    theta_degrees : angle of rotation

    Returns
    -------
    plane_vecs_aligned: (2,3) an array with two orthogonal vectors defining
        the best-fitting plane. The first vector corresponds to the side of the
        quadrilateral joining points 1 and 2, and the second vector is at a 
        90 degree angle from the first one.
    """ 
    
    theta_radians = np.radians(theta_degrees)
    # construct rotation matrix - z component stays the same
    R = np.eye(3)
    if axis=='z':
        R[:,0] = np.array([np.cos(theta_radians),np.sin(theta_radians),0])
        R[:,1] = np.array([-np.sin(theta_radians),np.cos(theta_radians),0])
    elif axis=='x':
        R[:,1] = np.array([0,np.cos(theta_radians),np.sin(theta_radians)])
        R[:,2] = np.array([0,-np.sin(theta_radians),np.cos(theta_radians)])
    elif axis=='y':
        R[:,0] = np.array([np.cos(theta_radians),0,-np.sin(theta_radians)])
        R[:,2] = np.array([np.sin(theta_radians),0,np.cos(theta_radians)])
        
    # note in the case of a 180 degree rotation, there will be a small 
    # numerical error for sin - the value won't be exactly 0
    
    # apply rotation matrix to datapoints
    points_rotated = R @ points.T
    
    return points_rotated


# def align_plane_vecs(points, pca):
#     """
#     Align the plane-defining vectors obtained from PCA with two sides of the 
#     quadrilateral defined by the datapoints. This correction is necessary for 
#     comparing plane angles. This is because PCA returns the vectors 
#     corresponding to the directions of maximum variance in the data. If these
#     directions differ between two planes being considered (i.e. correspond to
#     different sides of the square defined by the points), the angle calculated
#     for parallel planes might not be 0 degrees, but 180. Putting the plane-
#     defining vectors in a common frame of reference circumvents this issue and 
#     makes interpretation of the angles more straightforward. For more details 
#     see plane_fitting.py file.

#     Parameters
#     ----------
#     points : (n_points,3) numpy array with the 3D coordinates of the 
#             datapoints the plane was fitted to
        
#     pca : sklearn object corresponding to the best-fit plane

#     Returns
#     -------
#     plane_vecs_aligned: (2,3) an array with two orthogonal vectors defining
#         the best-fitting plane. The first vector corresponds to the side of the
#         quadrilateral joining points 1 and 2, and the second vector is at a 
#         90 degree angle from the first one.
#     """ 
    
#     # project datapoints onto the fitted plane
#     points_proj = np.zeros(points.shape)   
#     com = np.mean(points, axis=0) # centre of mass
#     for i in range(points_proj.shape[0]):
#         points_proj[i,:] = vops.getProjection(points[i,:]-com,pca.components_)
    
    
#     # get the vectors corresponding to the sides of the parallelogram
#     # these will be the new plane-defining bases
#     a = vops.getVecFromPoints(points_proj[0,:],points_proj[1,:])
#     b = vops.getVecFromPoints(points_proj[0,:],points_proj[-1,:])
    
#     # normalise them
#     a /= np.linalg.norm(a)
#     b /= np.linalg.norm(b)
    
#     # change basis to the one defined by the plane + its normal
#     # this is so that the vector(s) can be rotated around the normal with
#     # rotate_plane_by_angle(), which assumes that the z-component of the vector(s) 
#     # is 0
#     plane_basis = np.concatenate((pca.components_,
#                                   np.cross(pca.components_[0,:],pca.components_[1:,])))
    
#     a_newBasis = plane_basis @ a
#     b_newBasis = plane_basis @ b
    
#     n_newBasis = np.cross(a_newBasis,b_newBasis)
#     # force the plane-defining vectors to be orthogonal
#     if np.abs(np.dot(a,b)) > 0.001:
#         # if they're not already - rotate vector b 
#         # angle = np.arccos(np.dot(a,b))
        
#         # # a bit of a brute force approach here - try a positive angle first
#         # angle_diff = np.pi/2 - angle
#         # tmp = rotate_plane_by_angle(b_newBasis,angle_diff)
        
#         # # check vectors orthogonal
#         # if np.abs(np.dot(a_newBasis,tmp)) > 0.001:
#         #     # if not - try rotating by the same angle but in the opposite direction
#         #     tmp = rotate_plane_by_angle(b_newBasis,-angle_diff)
        
#         # if np.abs(np.dot(a_newBasis,tmp)) > 0.001:
#         #     raise ValueError('New vectors still not orthogonal')
#         # else:
#         #     b_newBasis = tmp
        
#         # if they're not already - rotate vector b
#         # get directional angle between a and b
#         triple_product = np.dot(a_newBasis,np.cross(b_newBasis,n_newBasis))
    
#         if triple_product >= 0:
#             angle = np.arccos(np.dot(a_newBasis,b_newBasis))
#         else:
#             angle = 2*np.pi - np.arccos(np.dot(a_newBasis,b_newBasis))
        
#         if angle < np.pi/2:
#             angle_diff = np.pi/2 - angle
#             tmp = rotate_plane_by_angle(b_newBasis,angle_diff)
#         else:
#             angle_diff = angle - np.pi/2
#             tmp = rotate_plane_by_angle(b_newBasis,-angle_diff)
        
#         if np.abs(np.dot(a_newBasis,tmp)) > 0.001:
#             raise ValueError('New vectors still not orthogonal')
#         else:
#             b_newBasis = tmp
        

#     # return to original (standard) basis
#     b = plane_basis.T @ b_newBasis
    
    
#     # double check that the new vectors in the standard basis *are* orthogonal
#     if np.abs(np.dot(a,b)) > 0.001:
#         print(np.degrees(np.dot(a,b)))
#         raise ValueError('New vectors not orthogonal')
        

#     plane_vecs_aligned = np.stack((a,b))
        
#     return plane_vecs_aligned


# def align_plane_vecs(points, pca):
#     """
#     Align the plane-defining vectors obtained from PCA with two sides of the 
#     quadrilateral defined by the datapoints. This correction is necessary for 
#     comparing plane angles. This is because PCA returns the vectors 
#     corresponding to the directions of maximum variance in the data. If these
#     directions differ between two planes being considered (i.e. correspond to
#     different sides of the square defined by the points), the angle calculated
#     for parallel planes might not be 0 degrees, but 180. Putting the plane-
#     defining vectors in a common frame of reference circumvents this issue and 
#     makes interpretation of the angles more straightforward. For more details 
#     see plane_fitting.py file.

#     Parameters
#     ----------
#     points : (n_points,3) numpy array with the 3D coordinates of the 
#             datapoints the plane was fitted to
        
#     pca : sklearn object corresponding to the best-fit plane

#     Returns
#     -------
#     plane_vecs_aligned: (2,3) an array with two orthogonal vectors defining
#         the best-fitting plane. The first vector corresponds to the side of the
#         quadrilateral joining points 1 and 2, and the second vector is at a 
#         90 degree angle from the first one.
#     """ 
    
#     # project datapoints onto the fitted plane
#     points_proj = np.zeros(points.shape)   
#     com = np.mean(points, axis=0) # centre of mass
#     for i in range(points_proj.shape[0]):
#         points_proj[i,:] = vops.getProjection(points[i,:]-com,pca.components_)
    
    
#     # get the vectors corresponding to the sides of the parallelogram
#     # these will be the new plane-defining bases
#     a = vops.getVecFromPoints(points_proj[0,:],points_proj[1,:])
#     b = vops.getVecFromPoints(points_proj[0,:],points_proj[2,:])
    
#     # normalise them
#     a /= np.linalg.norm(a)
#     b /= np.linalg.norm(b)
    
#     # change basis to the one defined by the plane + its normal
#     # this is so that the vector(s) can be rotated around the normal with
#     # rotate_plane_by_angle(), which assumes that the z-component of the vector(s) 
#     # is 0
#     plane_basis = np.concatenate((pca.components_,
#                                   np.cross(pca.components_[0,:],pca.components_[1:,])))
    
#     a_newBasis = plane_basis @ a
#     b_newBasis = plane_basis @ b
    
#     # force the plane-defining vectors to be orthogonal
#     if np.abs(np.dot(a,b)) > 0.001:
#         # if they're not already - rotate vector a by 90 degrees 
#         angle = np.arccos(np.dot(a_newBasis,b_newBasis))
#         angle_diff = np.degrees(np.pi/2 - angle)
        
#         # rotate
#         tmp = rotate_plane_by_angle(b_newBasis,-angle_diff)
#         if np.abs(np.dot(a_newBasis,tmp)) < 0.001:
#             b_newBasis = tmp
#         else:
#             # rotate the other way
#             tmp = rotate_plane_by_angle(b_newBasis,angle_diff)
#             #double check that orthogonal now
            
#             if np.abs(np.dot(a_newBasis,tmp)) > 0.001:
#                 raise ValueError('New vectors still not orthogonal')
#             else:
#                 b_newBasis = tmp
        

#     # return to original (standard) basis
#     b = plane_basis.T @ b_newBasis
    
    
#     # double check that the new vectors in the standard basis *are* orthogonal
#     if np.abs(np.dot(a,b)) > 0.001:
#         print(np.degrees(np.dot(a,b)))
#         raise ValueError('New vectors not orthogonal')
        

#     plane_vecs_aligned = np.stack((a,b))
        
#     return plane_vecs_aligned


def fit_mds_to_rdm(rdm):
    mds = MDS(n_components=3, 
              metric=True, 
              dissimilarity='precomputed', 
              max_iter=1000,
              random_state=0)
    return mds.fit_transform(rdm)

#%% define low-level plotting functions


def plot_geometry(ax,data,pca,plot_colours,plot_outline=True,legend_on=True,custom_labels=None):
    
    if not custom_labels:
        labels = ['L1','L2']
    else:
        if len(custom_labels) != 2:
            raise ValueError('Custom labels should be a list with 2 entries')
        else:
            labels = custom_labels
    ms = 50
    n_colours = len(plot_colours)
    # plot the parallelogram defined by colours at location 1
    if plot_outline:
        ax.plot(np.append(data[:n_colours,0],data[0,0]),
              np.append(data[:n_colours,1],data[0,1]),
              np.append(data[:n_colours,2],data[0,2]),'k-')
    ax.scatter(data[0,0],data[0,1], data[0,2],marker='^',s = ms,
              c='k',label=labels[0])
    ax.scatter(data[:n_colours,0],data[:n_colours,1],
              data[:n_colours,2],marker='^',s = ms,c=plot_colours)
  
    # repeat for loc 2
    if plot_outline:
        ax.plot(np.append(data[n_colours:,0],data[n_colours,0]),
              np.append(data[n_colours:,1],data[n_colours,1]),
              np.append(data[n_colours:,2],data[n_colours,2]),'k-')
    ax.scatter(data[-1,0],data[-1,1], data[-1,2],marker='s',s = ms,
              c='k',label=labels[1])
    ax.scatter(data[n_colours:,0],data[n_colours:,1],
              data[n_colours:,2],marker='s',s = ms,c=plot_colours)
    
    if pca:
        ax.set_xlabel('PC1 ['+str(np.round(pca.explained_variance_ratio_[0]*100,1))+'%]',
                      labelpad = 12)
        ax.set_ylabel('PC2 ['+str(np.round(pca.explained_variance_ratio_[1]*100,1))+'%]',
                      labelpad = 12)
        ax.set_zlabel('PC3 ['+str(np.round(pca.explained_variance_ratio_[2]*100,1))+'%]',
                      labelpad = 12)
    if legend_on:
        ax.legend(bbox_to_anchor = (1,.9),loc='center left')    
  
    
def plot_plane(ax,verts,fc='k',a=0.2):
    # plot a polygon with given vertices in 3D
    ax.add_collection3d(Poly3DCollection([verts],facecolor=fc,edgecolor=[],alpha=a))
  
    
# def plot_subspace(ax,points,plane_vecs,fc='k',a=0.2):
#     # plot the best-fitting plane as a convex hull (not necessarily a 
#     # quadrilateral) with vertices being the projections of original points 
#     # onto the plane 
    
#     if (points.shape[1]!=3):
#         raise NotImplementedError('Check shape of data matrix - should be [n_points,3]')
    
#     # 
#     plane_basis = np.concatenate((plane_vecs,
#                                   np.cross(plane_vecs[0,:],plane_vecs[1:,])))
#     # find vertices
#     n_points = points.shape[0]
#     verts = np.empty((n_points,3))
#     verts_2d = np.empty((n_points,3))
#     com = np.mean(points, axis=0) # centre of mass
    
#     for i in range(n_points):
#         verts[i,:] = vops.getProjection(points[i,:]-com,plane_vecs) # get projection of demeaned 3d points
#         # change basis to that defined by the plane + its normal to zero out 
#         # the 3rd coordinate
#         # this is so that can fit a convex hull to the data with 
#         # vops.defPlaneShape() - otherwise ConvexHull will complain about 
#         # points being coplanar 
#         verts_2d[i,:] =  plane_basis @ verts[i,:]
#         verts[i,:] += com #add the mean back


#     # only pass 2D coordinates to ConvexHull
#     convex_verts, sorting_order = vops.defPlaneShape(verts_2d[:,:2],[])
    
#     sorted_verts = verts[sorting_order,:] # sorted 3D coordinates
#     # plot the best-fit plane
#     plot_plane(ax,sorted_verts,fc,a)
#     #return verts, sorted_verts


def plot_subspace(ax,points,plane_vecs,fc='k',a=0.2):
    # plot the best-fitting plane as a quadrilateral with vertices being the projections of original points onto the plane 
    
    if (points.shape[1]!=3):
        raise NotImplementedError('Check shape of data matrix - should be [n_points,3]')
    
    # find vertices
    n_points = points.shape[0]
    verts = np.zeros((n_points,3))
    
    com = np.mean(points, axis=0) # centre of mass
    
    for i in range(n_points):
        # get projection of demeaned 3d points
        verts[i,:] = vops.getProjection(points[i,:]-com,plane_vecs) 
        verts[i,:] += com #add the mean back

    # sort vertices according to shortest path - so that plotted plane will be a quadrilateral
    sorted_verts, sorting_order = vops.sortByPathLength(verts)

    
    # plot the best-fit plane
    plot_plane(ax,sorted_verts,fc,a)
    # return verts, sorted_verts
    
    
def plot_geom_and_subspace(constants,data,plot_title,size=[9,6],custom_labels=None):
    n_colours = constants.PARAMS['B']
    plt.figure(figsize=size)

    ax = plt.subplot(111, projection='3d')
    plot_geometry(ax,
                  data['3Dcoords'],
                  data['pca'], 
                  constants.PLOT_PARAMS['4_colours'],
                  custom_labels = custom_labels)
    plot_subspace(ax,data['3Dcoords'][:n_colours,:],data['plane1'].components_,fc='k',a=0.2)
    plot_subspace(ax,data['3Dcoords'][n_colours:,:],data['plane2'].components_,fc='k',a=0.2)
    ax.set_title(plot_title + ', ' + r'$\theta$' + ' = %.1f' %data['theta']+'°')
    helpers.equal_axes(ax)
    ax.tick_params(pad = 4.0)
    plt.tight_layout()
    
    
#%% model RDMs

def get_model_RDMs(constants):
    
    path = constants.PARAMS['FULL_PATH']
    n_colours=constants.PARAMS['B']
    
    # pick 3D coordinates for the predictor pre-cue and post-cue geometries
    # pre-cue
    pre_up = np.array([[-1,-1,1,1],[1,-1,-1,1],[0,0,0,0]])
    pre_down =  np.array([[0,0,0,0],[-1,1,1,-1],[1,1,-1,-1]])
    pre = np.concatenate((pre_up,pre_down),axis=1)
    
    # post-cue
    post_up = np.array([[1,1,-1,-1],[1,-1,-1,1],[0.5,0.5,0.5,0.5]])
    post_down = np.array([[1,1,-1,-1],[1,-1,-1,1],[-0.5,-0.5,-0.5,-0.5]])
    post = np.concatenate((post_up,post_down),axis=1)
        
    # plot_predictor_geometry(constants,pre,post)

    #% get model RDMs
    
    orthoPlanes_RDM = squareform(pdist(pre.T))
    parallelPlanes_RDM = squareform(pdist(post.T))
    # singlePlane_RDM = squareform(pdist(post_singlePlane))
    
    model_RDMs = np.stack((orthoPlanes_RDM,parallelPlanes_RDM),axis=2)

    # model_RDMs = np.stack((orthoPlanes_RDM,parallelPlanes_RDM,singlePlane_RDM),axis=2)
    # model_RDMs /= np.max(model_RDMs) # normalise values to be within 0 and 1
    
        
    #% plot model/predictor RDMs
    # plot_predictor_RDMs(constants,model_RDMs)
    
    
    #% save model RDMs
    
    # get upper diagonal / square form
    model_RDMs_sqform = np.zeros((squareform(model_RDMs[:,:,0]).shape[0],model_RDMs.shape[-1]))
    for model in range(model_RDMs.shape[-1]):
        model_RDMs_sqform[:,model] = squareform(model_RDMs[:,:,model])
    
    
    if not (os.path.exists(path+'RSA')):
                os.mkdir(path+'RSA')
    
    save_path = path + 'RSA/'
    

    pickle.dump(model_RDMs,open(save_path + 'model_RDMs.pckl','wb'))
    pickle.dump(model_RDMs_sqform,open(save_path + 'model_RDMs_sqform.pckl','wb'))
    
    model_RDMs_order = ['orthogonal','parallel']
    pickle.dump(model_RDMs_order,open(save_path + 'model_RDMs_order.pckl','wb'))
    
    
    #% ROTATIONS
    
    # set up up the data folders
    save_path = path + 'RSA/'
    
    helpers.check_path(save_path+'rotated_coords/')
    helpers.check_path(save_path+'rotated_fullRDMs/')
    helpers.check_path(save_path+'rotated_diagRDMs/')
    
    helpers.check_path(save_path+'ortho_rotated_coords/')
    helpers.check_path(save_path+'ortho_rotated_fullRDMs/')
    helpers.check_path(save_path+'ortho_rotated_diagRDMs/')
    
    # create a bunch of rotation matrices, delta_theta = 5 degrees
    theta_range = np.arange(0,365,5)
    
    for i,theta_degrees in enumerate(theta_range):
        
        ## parallel planes
        points = post.T[n_colours:,:]
        points_r = rotate_plane_by_angle(points,theta_degrees)
        
        new_coords = np.concatenate((post[:,:n_colours],points_r),axis=1)
        
        # save the coords
        pickle.dump(new_coords,open(save_path 
                                    + 'rotated_coords/rotatedBy'
                                    + str(theta_degrees)+'.pckl','wb'))
        
        # calculate and save the RDM
        rotated_RDM = squareform(pdist(new_coords.T))
        #full
        pickle.dump(rotated_RDM,open(save_path 
                                    + 'rotated_fullRDMs/'
                                    + str(theta_degrees)+'.pckl','wb'))
        # diagonal
        rotated_RDM_sqform = squareform(rotated_RDM)
        pickle.dump(rotated_RDM_sqform,open(save_path 
                                + 'rotated_diagRDMs/'
                                + str(theta_degrees)+'.pckl','wb'))
        
        ## orthogonal planes
        ortho_points = pre.T[:n_colours,:]
        ortho_points_r = rotate_plane_by_angle(ortho_points,theta_degrees)
        new_ortho_coords = np.concatenate((ortho_points_r,pre[:,n_colours:]),axis=1)
        
        # save the coords
        pickle.dump(new_ortho_coords,open(save_path 
                                    + 'ortho_rotated_coords/rotatedBy'
                                    + str(theta_degrees)+'.pckl','wb'))
        
        # calculate and save the RDM
        ortho_rotated_RDM = squareform(pdist(new_ortho_coords.T))
        #full
        pickle.dump(ortho_rotated_RDM,open(save_path 
                                    + 'ortho_rotated_fullRDMs/'
                                    + str(theta_degrees)+'.pckl','wb'))
        # diagonal
        ortho_rotated_RDM_sqform = squareform(ortho_rotated_RDM)
        pickle.dump(ortho_rotated_RDM_sqform,open(save_path 
                                + 'ortho_rotated_diagRDMs/'
                                + str(theta_degrees)+'.pckl','wb'))
    
    
    # save the theta range used
    pickle.dump(theta_range,open(save_path+'theta_range.pckl','wb'))
      
    
    #%   MIRROR IMAGES
    
    # get 3D coords
    R = np.array([[0,1,0],[1,0,0],[0,0,1]]) # flip matrix
    post_flipped = post.copy()
    post_flipped = post_flipped.T
    post_flipped[n_colours:,:] = post_flipped[n_colours:,:]@R
    
    # save the coords
    pickle.dump(post_flipped,open(save_path 
                                  + 'flipped_coords'+'.pckl','wb'))
    
    

    # calculate and save the RDM
    flipped_RDM = squareform(pdist(post_flipped))
    #full
    pickle.dump(flipped_RDM,open(save_path 
                                + 'flipped_fullRDM.pckl','wb'))
    # diagonal
    flipped_RDM_sqform = squareform(flipped_RDM)
    pickle.dump(flipped_RDM_sqform,open(save_path 
                            + 'flipped_RDM_sqform.pckl','wb'))
    

def plot_predictor_RDMs(constants,model_RDMs):
    fig, axes = plt.subplots(1,2, sharex=True, sharey = True, figsize=(13.65,5))
    
    titles = ['ortho','parallel']

    for ix,ax in enumerate(axes.flat):
        im = ax.imshow(model_RDMs[:,:,ix],
                       vmin=np.min(model_RDMs),
                       vmax=np.max(model_RDMs),
                       # cmap = sns.color_palette("flare_r",as_cmap = True))
                       cmap = sns.cubehelix_palette(start=.5,rot=-.75,as_cmap = True,reverse=True))
        ax.set_title(titles[ix])
        ax.set_xticks([])
        ax.set_yticks([])
    
    fig.colorbar(im,ax=axes.ravel().tolist())
    return


def plot_predictor_geometry(constants,pre_coords,post_coords):
    colours = constants.PLOT_PARAMS['4_colours']

    #% plot ortho and parallel
    plt.figure(figsize=(12,5),num='Predictor geometry')
    
    ax = plt.subplot(121, projection='3d')
    plot_geometry(ax,pre_coords.T,[],colours,legend_on=False)
    ax.set_title('pre-cue')
    ax.scatter(0,0,0,marker='+',c='k')
    
    
    ax2 = plt.subplot(122, projection='3d')
    plot_geometry(ax2,post_coords.T,[],colours)
    ax2.set_title('post-cue')
    plt.legend(bbox_to_anchor=(1, 1),
                bbox_transform=plt.gcf().transFigure)
    ax2.scatter(0,0,0,marker='+',c='k')
    
    
    plt.savefig(constants.PARAMS['FIG_PATH']+'predictor_geometry.png')
    return


def plot_predictor_geometry_variations(constants):
    load_path = constants.PARAMS['FULL_PATH'] + 'RSA/'
    colours = constants.PLOT_PARAMS['4_colours']
    
    
    plt.figure(figsize=(18,5))
    
    # rotated by 90
    theta_degrees = 90
    coords_90 = pickle.load(open(load_path 
                                    + 'rotated_coords/rotatedBy'
                                    + str(theta_degrees)+'.pckl','rb'))
    ax = plt.subplot(131,projection='3d')
    plot_geometry(ax,coords_90.T,[],colours,legend_on=False)
    ax.set_title('$parallel_{90}$')
    ax.set_xticks([-1,0,1])
    ax.set_yticks([-1,0,1])
    ax.set_zticks([-.5,0,.5])
    ax.set_xlabel('dim 1',labelpad=20.0)
    ax.set_ylabel('dim 2',labelpad=20.0)
    ax.set_zlabel('dim 3',labelpad=30.0)
    ax.tick_params(axis='z',which='major', pad=10)
    
    # flipped
    coords_flipped = pickle.load(open(load_path 
                                  + 'flipped_coords'+'.pckl','rb'))
    
    ax2 = plt.subplot(132,projection='3d')
    plot_geometry(ax2,coords_flipped,[],colours,legend_on=False)
    ax2.set_title('mirror')
    ax2.set_xlabel('dim 1',labelpad=25.0)
    ax2.set_ylabel('dim 2',labelpad=25.0)
    ax2.set_zlabel('dim 3',labelpad=30.0)
    ax2.set_xticks([-1,0,1])
    ax2.set_yticks([-1,0,1])
    ax2.set_zticks([-.5,0,.5])
    ax2.tick_params(which='major', pad=5)
    ax2.tick_params(axis='z', pad=10)
    
    
    
    # ortho 45
    ax3 = plt.subplot(133,projection='3d')
    theta_degrees = 45
    ortho_45 = pickle.load(open(load_path 
                                    + 'ortho_rotated_coords/rotatedBy'
                                    + str(theta_degrees)+'.pckl','rb'))
    plot_geometry(ax3,ortho_45.T,[],colours)
    
    
    ax3.set_title('$ortho_{45}$')
    ax3.set_xlabel('dim 1',labelpad=20.0)
    ax3.set_ylabel('dim 2',labelpad=20.0)
    ax3.set_zlabel('dim 3',labelpad=20.0)
    
    
    # plt.legend(bbox_to_anchor=(.8,.8))
    plt.legend(bbox_to_anchor=(1.5,1.2))
    plt.tight_layout()
    

def get_data_RDMs(constants):
    '''
    Construct data RDMs.

    Parameters
    ----------
    constants : TYPE
        DESCRIPTION.

    Returns
    -------
    rdm_precue : TYPE
        DESCRIPTION.
    rdm_postcue : TYPE
        DESCRIPTION.
    pre_data_RDM_averaged : TYPE
        DESCRIPTION.
    post_data_RDM_averaged : TYPE
        DESCRIPTION.

    '''
    
    load_path = constants.PARAMS['FULL_PATH']+'pca_data/valid_trials/'
    n_conditions = constants.PARAMS['B']*constants.PARAMS['B']*2

    # preallocate structures
    rdm_precue = np.zeros((n_conditions,n_conditions,constants.PARAMS['n_models']))
    rdm_postcue = np.zeros((n_conditions,n_conditions,constants.PARAMS['n_models']))
    
    rdm_precue_binned = np.zeros((constants.PARAMS['M'],constants.PARAMS['M'],constants.PARAMS['n_models']))
    rdm_postcue_binned = np.zeros((constants.PARAMS['M'],constants.PARAMS['M'],constants.PARAMS['n_models']))
    rdm_stimuli_binned = np.zeros((constants.PARAMS['M'],constants.PARAMS['M'],constants.PARAMS['n_models']))
    # rdm_postcue_binned_readout = np.zeros((batch_size//4,batch_size//4))
    rdm_precue_uncued_binned = np.zeros((constants.PARAMS['M'],constants.PARAMS['M'],constants.PARAMS['n_models']))
    rdm_postcue_uncued_binned = np.zeros((constants.PARAMS['M'],constants.PARAMS['M'],constants.PARAMS['n_models']))
    
    # cued vs uncued
    rdm_postcue_cu_binned = np.zeros((constants.PARAMS['M'],constants.PARAMS['M'],constants.PARAMS['n_models'],2))
    
    # pick timepoints of interest
    t1 = constants.PARAMS['trial_timepoints']['delay1_end']-1
    t2 = constants.PARAMS['trial_timepoints']['delay2_end']-1
    
    # plt.figure()
    # create pre- and post-cue RDMs for each model
    for model in range(constants.PARAMS['n_models']):
        # load pca data
        f = open(load_path+'rdm_data_model' + str(model) + '.pckl', 'rb')
        rdm_data = pickle.load(f)
        f.close()
        # data format: trial x time x neuron
        rdm_precue[:,:,model] = squareform(pdist(rdm_data[:,t1,:]))
        rdm_postcue[:,:,model] = squareform(pdist(rdm_data[:,t2,:]))
        
        f = open(load_path+'pca_data_model' + str(model) + '.pckl', 'rb')
        pca_data = pickle.load(f)
        f.close()
        rdm_stimuli_binned[:,:,model] = squareform(pdist(pca_data['data'][:,0,:]))
        rdm_precue_binned[:,:,model] = squareform(pdist(pca_data['delay1']))
        rdm_postcue_binned[:,:,model] = squareform(pdist(pca_data['delay2']))
        
        # normalise values to [0,1] range
        rdm_precue_binned[:,:,model] /= rdm_precue_binned[:,:,model].max()
        rdm_postcue_binned[:,:,model] /=rdm_postcue_binned[:,:,model].max()
        
        
        # get the uncued representations
        f = open(load_path+'pca_data_uncued_model' + str(model) + '.pckl', 'rb')
        pca_data_uncued = pickle.load(f)
        f.close()
        rdm_precue_uncued_binned[:,:,model] = squareform(pdist(pca_data_uncued['delay1']))
        rdm_postcue_uncued_binned[:,:,model] = squareform(pdist(pca_data_uncued['delay2']))
        
        rdm_precue_uncued_binned[:,:,model] /= rdm_precue_uncued_binned[:,:,model].max()
        rdm_postcue_uncued_binned[:,:,model] /=rdm_postcue_uncued_binned[:,:,model].max()
        
        # get cued vs uncued
        f = open(load_path+'trial_type_subspaces_model'+str(model)+'.pckl','rb')
        trial_type_subspaces = pickle.load(f)
        f.close()
        
        cued_up = trial_type_subspaces['cued_up']
        cued_down = trial_type_subspaces['cued_down']
        rdm_postcue_cu_binned[:,:,model,0] = squareform(pdist(cued_up['binned_data']))
        rdm_postcue_cu_binned[:,:,model,1] = squareform(pdist(cued_down['binned_data']))
        
        
        rdm_postcue_cu_binned[:,:,model,0] /= rdm_postcue_cu_binned[:,:,model,0].max()
        rdm_postcue_cu_binned[:,:,model,1] /= rdm_postcue_cu_binned[:,:,model,1].max()
        
        # plot
        # plt.subplot(2,10,int(model)+1)
        # plt.imshow(rdm_postcue_binned[:,:,model])
        # plt.colorbar()
        # plt.title('Model '+str(model))

    #% save
    
    save_path = constants.PARAMS['FULL_PATH']+'RSA/'
    helpers.check_path(save_path)
    indiv_model_rdms = [rdm_precue_binned,rdm_postcue_binned]
    pickle.dump(indiv_model_rdms,open(save_path+'indiv_model_rdms.pckl','wb'))
    indiv_model_rdms_uncued = [rdm_precue_uncued_binned,rdm_postcue_uncued_binned]
    pickle.dump(indiv_model_rdms_uncued,open(save_path+'indiv_model_rdms_uncued.pckl','wb'))
    pickle.dump(rdm_postcue_cu_binned,open(save_path+'indiv_model_rdms_cu.pckl','wb'))

    # average across models
    pre_data_RDM_averaged = np.mean(rdm_precue_binned,2)
    post_data_RDM_averaged = np.mean(rdm_postcue_binned,2)
    pre_data_uncued_RDM_averaged = np.mean(rdm_precue_uncued_binned,2)
    post_data_uncued_RDM_averaged = np.mean(rdm_postcue_uncued_binned,2)
    post_data_cu_RDM_averaged = np.mean(rdm_postcue_cu_binned,2)

    # save
    pickle.dump(pre_data_RDM_averaged,open(save_path+'pre_data_RDM_averaged.pckl','wb'))
    pickle.dump(post_data_RDM_averaged,open(save_path+'post_data_RDM_averaged.pckl','wb'))
    pickle.dump(pre_data_uncued_RDM_averaged,open(save_path+'pre_data_uncued_RDM_averaged.pckl','wb'))
    pickle.dump(post_data_uncued_RDM_averaged,open(save_path+'post_data_uncued_RDM_averaged.pckl','wb'))
    pickle.dump(post_data_cu_RDM_averaged,open(save_path+'post_data_cu_RDM_averaged.pckl','wb'))

    return rdm_precue,rdm_postcue,pre_data_RDM_averaged,post_data_RDM_averaged


# def get_PVEs_3D(constants):
#     load_path = constants.PARAMS['FULL_PATH']+'pca_data/valid_trials/'
#     f = open(load_path+'indiv_model_rdms.pckl','rb')
#     [rdm_precue_binned,rdm_postcue_binned] = pickle.load(f)
    
#     total_PVE3_pre = np.empty((constants.PARAMS['n_models'],))
#     total_PVE3_post = np.empty((constants.PARAMS['n_models'],))
    
#     save_path = load_path + '/pca3/'
#     helpers.check_path(save_path)
#     for model in range(constants.PARAMS['n_models']):
#         # fit PCA to datapoints
#         # center data
#         pre_data = rdm_precue_binned[:,:,model] - np.mean(rdm_precue_binned[:,:,model])
#         post_data = rdm_postcue_binned[:,:,model] - np.mean(rdm_postcue_binned[:,:,model])
    
#         # fit a PCA model from sklearn.decomposition
#         pca_pre = PCA(n_components=3)
#         pca_pre.fit(pre_data)
        
#         pca_post = PCA(n_components=3)
#         pca_post.fit(post_data)
        
#         # get total variance explained by components
#         total_PVE3_pre[model] = pca_pre.explained_variance_ratio_.sum()
#         total_PVE3_post[model] = pca_post.explained_variance_ratio_.sum()
        
#         #save
#         f = open(save_path+'pca3_' + str(model)+'.pckl','rb')
        
#     return total_PVE3_pre, total_PVE3_post

        
# def get_PVEs_2D():
        

# def get_plane_angles_all_models(constants):
#     load_path = constants.PARAMS['FULL_PATH']+'RSA/'
#     f = open(load_path+'indiv_model_rdms.pckl','rb')
#     [rdm_precue_binned,rdm_postcue_binned] = pickle.load(f)
#     n_colours = constants.PARAMS['B']
    
#     helpers.check_path(load_path+ 'angles/')

#     for model in range(constants.PARAMS['n_models']):       
#         # fit MDS
#         mds_precue = fit_mds_to_rdm(rdm_precue_binned[:,:,model])
#         mds_postcue = fit_mds_to_rdm(rdm_postcue_binned[:,:,model])
        
#         # get planes
#         delay1_planeUp = get_best_fit_plane(mds_precue[0:n_colours])
#         delay1_planeDown = get_best_fit_plane(mds_precue[n_colours:])
        
#         delay2_planeUp = get_best_fit_plane(mds_postcue[0:n_colours])
#         delay2_planeDown = get_best_fit_plane(mds_postcue[n_colours:])
        
#         # find angles
#         theta_pre = get_angle_between_planes(delay1_planeUp.components_,delay1_planeDown.components_)
#         theta_post = get_angle_between_planes(delay2_planeUp.components_,delay2_planeDown.components_,)
        
#         # save angles
        
#         f = open(load_path+ 'angles/angles_' + str(model)+ '.pckl','wb')
#         pickle.dump([theta_pre,theta_post],f)


def hyperalign_delay_data(constants,trial_type='valid'):
    # might want to rename it to just hyperalign data - since now it also uses the probe timepoint
    
    load_path = constants.PARAMS['FULL_PATH']+'pca_data/'+trial_type+'_trials/'
    
    delay1_all = []
    delay2_all = []
    if constants.PARAMS['experiment_number'] == 3:
        probe_all = []
    elif constants.PARAMS['experiment_number'] == 2:
        # want to test the model on trials with delay_length = 5
        update_time_params(constants.PARAMS,5)
        
    d1_ix = constants.PARAMS['trial_timepoints']['delay1_end']-1
    d2_ix = constants.PARAMS['trial_timepoints']['delay2_end']-1
    
    # concatenate data from all models into a single list
    for model in range(constants.PARAMS['n_models']):
        f = open(load_path+'eval_data_model'+str(model)+'.pckl','rb')
        eval_data = pickle.load(f)
        f.close()
        
        

        
        delay1 = eval_data['data'][:,d1_ix,:]
        delay2 = eval_data['data'][:,d2_ix,:]
        probe = eval_data['data'][:,-1,:]
        
        delay1_all.append(np.array(delay1))
        delay2_all.append(np.array(delay2))
        
        if constants.PARAMS['experiment_number'] == 3:
            probe_all.append(np.array(probe))
        
    
    # hyperalignment
    delay1_all_aligned = np.stack(hypertools.align(delay1_all))
    delay2_all_aligned = np.stack(hypertools.align(delay2_all))
    if constants.PARAMS['experiment_number'] == 3:
        probe_all_aligned = np.stack(hypertools.align(probe_all))
    
    # save hyperaligned data
    delay_data_hyperaligned = {}
    delay_data_hyperaligned['delay1'] = delay1_all_aligned
    delay_data_hyperaligned['delay2'] = delay2_all_aligned
    if constants.PARAMS['experiment_number'] == 3:
        delay_data_hyperaligned['probe'] = probe_all_aligned
    delay_data_hyperaligned['dimensions'] = ['model','trials','neurons']
    delay_data_hyperaligned['labels'] = eval_data['labels']
    pickle.dump(delay_data_hyperaligned,open(load_path+'delay_data_hyperaligned.pckl','wb'))
    
    return delay_data_hyperaligned

    
def get_model_averages(constants,trial_type='valid'):
    '''
    Get population response patterns to cued and uncued stimuli, averaged across
    trained networks. The matrices containing population responses to 
    all experimental stimuli are hyperaligned across models. Then, the responses
    are averaged across models and data is binned into colour bins and averaged
    across uncued or cued stimuli, to obtain the cued and uncued colour 
    representation matrices, respectively.

    Parameters
    ----------
    constants : TYPE
        DESCRIPTION.

    Returns
    -------
    model_av_binned_cued : TYPE
        DESCRIPTION.
    model_av_binned_uncued : TYPE
        DESCRIPTION.

    '''
    #//// CHANGE THIS SO THAT UNCUED IS RESORTED PRIOR TO BINNING ///////
    # still need to update to be able to do for invalid trials as well #
    load_path = constants.PARAMS['FULL_PATH']+'pca_data/'+trial_type+'_trials/' 
    print('Hyperalignment...')
    
    # if helpers.check_file_exists(load_path+'delay_data_hyperaligned.pckl'):
    #     # load data
    #     delay_data_hyperaligned = \
    #         pickle.load(open(load_path+'delay_data_hyperaligned.pckl','rb'))
    # else:
    #     # hyperalign data
    delay_data_hyperaligned = hyperalign_delay_data(constants)
        
    # average across models
    delay1 = {'data':torch.tensor(delay_data_hyperaligned['delay1'].mean(0)),
              'labels':delay_data_hyperaligned['labels']}
    delay2 = {'data':torch.tensor(delay_data_hyperaligned['delay2'].mean(0)),
              'labels':delay_data_hyperaligned['labels']}
    
    # bin cued (i.e. average across n_samples)
    # n_samples = delay1.shape[0]//constants.PARAMS['M']
    # delay1_binned_cued = torch.reshape(delay1.unsqueeze(1).unsqueeze(0),
    #                             (constants.PARAMS['M'],n_samples,
    #                             1,constants.PARAMS['n_rec']))
    # delay2_binned_cued = torch.reshape(delay2.unsqueeze(1).unsqueeze(0),
    #                             (constants.PARAMS['M'],n_samples,
    #                             1,constants.PARAMS['n_rec']))
    
    # delay1_binned_cued = delay1_binned_cued.mean(1).squeeze()
    # delay2_binned_cued = delay2_binned_cued.mean(1).squeeze()
    
    # added 03.02.22
    delay1_binned_cued = helpers.bin_data(delay1['data'],constants.PARAMS)
    delay2_binned_cued = helpers.bin_data(delay2['data'],constants.PARAMS)
    
    if constants.PARAMS['experiment_number'] == 3:
        probe = {'data':torch.tensor(delay_data_hyperaligned['probe'].mean(0)),
              'labels':delay_data_hyperaligned['labels']}
        
        probe_binned_cued = helpers.bin_data(probe['data'],constants.PARAMS)
        
    # bin uncued
    # n_trials = delay1['data'].shape[0]

    # labels_uncued = np.concatenate((delay_data_hyperaligned["labels"]["c2"][:n_trials//2],
    #                           delay_data_hyperaligned["labels"]["c1"][n_trials//2:]))
    
    # labels_uncued_binned = helpers.bin_labels(labels_uncued,constants.PARAMS['B'])

    # delay1_uncued_down = torch.zeros((constants.PARAMS['B'],constants.PARAMS['n_rec'])) # c2 uncued
    # delay1_uncued_up = torch.zeros((constants.PARAMS['B'],constants.PARAMS['n_rec'])) # c1 uncued
    
    # delay2_uncued_down = torch.zeros((constants.PARAMS['B'],constants.PARAMS['n_rec'])) # c2 uncued
    # delay2_uncued_up = torch.zeros((constants.PARAMS['B'],constants.PARAMS['n_rec'])) # c1 uncued
    
    
    # for colour in range(constants.PARAMS['B']):
    #     ix = np.where(labels_uncued_binned==colour)[0]
    #     ix_down = ix[np.where(ix<n_trials//2)[0]]
    #     ix_up = ix[np.where(ix>=n_trials//2)[0]]
    
    #     delay2_uncued_down[colour,:] = delay2['data'][ix_down,:].mean(0)
    #     delay2_uncued_up[colour,:] = delay2['data'][ix_up,:].mean(0)
        
    #     delay1_uncued_down[colour,:] = delay1['data'][ix_down,:].mean(0)
    #     delay1_uncued_up[colour,:] = delay1['data'][ix_up,:].mean(0)
    
    # #delay1_binned_uncued = torch.cat((delay1_uncued_down,delay1_uncued_up)).type(torch.float64)
    # # delay2_binned_uncued = torch.cat((delay2_uncued_down,delay2_uncued_up)) 
    
    delay1_uncued = helpers.sort_by_uncued(delay1,constants.PARAMS)
    delay2_uncued = helpers.sort_by_uncued(delay2,constants.PARAMS)
    
    delay1_binned_uncued = helpers.bin_data(delay1_uncued['data'],constants.PARAMS)
    delay2_binned_uncued = helpers.bin_data(delay2_uncued['data'],constants.PARAMS)
    
    if constants.PARAMS['experiment_number'] == 3:
        probe_uncued = helpers.sort_by_uncued(probe,constants.PARAMS)
        probe_binned_uncued = helpers.bin_data(probe_uncued['data'],constants.PARAMS)
        
    
    print('              ...done')
    # save
    model_av_binned_cued = {}
    model_av_binned_cued['delay1'] = delay1_binned_cued
    model_av_binned_cued['delay2'] = delay2_binned_cued
    
    model_av_binned_uncued = {}
    model_av_binned_uncued['delay1'] = delay1_binned_uncued
    model_av_binned_uncued['delay2'] = delay2_binned_uncued
    
    if constants.PARAMS['experiment_number'] == 3:
        model_av_binned_cued['probe'] = probe_binned_cued
        model_av_binned_uncued['probe'] = probe_binned_uncued
    
    pickle.dump(model_av_binned_cued,
                open(load_path+'model_av_binned_cued.pckl','wb'))
    pickle.dump(model_av_binned_uncued,
                open(load_path+'model_av_binned_uncued.pckl','wb'))

    return model_av_binned_cued, model_av_binned_uncued
    

def get_averaged_cued_subspaces(constants,trial_type='valid'):
    
    # load data
    load_path = constants.PARAMS['FULL_PATH']+'pca_data/'+trial_type+'_trials/'
    model_av_binned_cued = pickle.load(open(load_path+'model_av_binned_cued.pckl','rb'))
    
    # run the pca pipelines
    delay1_cued = model_av_binned_cued['delay1']
    delay2_cued = model_av_binned_cued['delay2']

    delay1_cued_subspace = run_pca_pipeline(constants, delay1_cued, ['up','down'])
    delay2_cued_subspace = run_pca_pipeline(constants, delay2_cued, ['up','down'])
    
    if constants.PARAMS['experiment_number'] == 3:
        probe_cued = model_av_binned_cued['probe']
        probe_cued_subspace = run_pca_pipeline(constants, probe_cued, ['up','down'])

    
    # save
    model_averaged_subspaces_cued = {}
    model_averaged_subspaces_cued['delay1'] = delay1_cued_subspace
    model_averaged_subspaces_cued['delay2'] = delay2_cued_subspace
    if constants.PARAMS['experiment_number'] == 3:
        model_averaged_subspaces_cued['probe'] = probe_cued_subspace

    
    return model_averaged_subspaces_cued


def plot_averaged_cued_geometry(constants,trial_type='valid'):
    n_colours = constants.PARAMS['B']
    
    
    # get data
    model_averaged_subspaces_cued = get_averaged_cued_subspaces(constants,
                                                                trial_type=trial_type)
    delay1_cued_subspace = model_averaged_subspaces_cued['delay1']
    delay2_cued_subspace = model_averaged_subspaces_cued['delay2']

    if constants.PARAMS['experiment_number'] == 3:
        n_subplots=3
        fsize = (18,5)
        probe_cued_subspace = model_averaged_subspaces_cued['probe']
        legend2 = False
    else:
        n_subplots = 2
        fsize = (12,5)
        legend2 = True
    
    # plot
    plt.figure(figsize=fsize,num='Averaged cued geometry')
    ax = plt.subplot(1,n_subplots,1, projection='3d')
    plot_geometry(ax, 
                  delay1_cued_subspace['3Dcoords'],
                  delay1_cued_subspace['pca'],
                  constants.PLOT_PARAMS['4_colours'],
                  legend_on=False)
    plot_subspace(ax,
                  delay1_cued_subspace['3Dcoords'][:n_colours,:],
                  delay1_cued_subspace['plane1'].components_)
    plot_subspace(ax,
                  delay1_cued_subspace['3Dcoords'][n_colours:,:],
                  delay1_cued_subspace['plane2'].components_)
    # ax.set_title('Angle: %.1f °' %theta_pre)
    ax.set_title('pre-cue')
    helpers.equal_axes(ax)
    ax.tick_params(pad = 4.0)
    
    ax2 = plt.subplot(1,n_subplots,2, projection='3d')
    plot_geometry(ax2, 
                  delay2_cued_subspace['3Dcoords'],
                  delay2_cued_subspace['pca'],
                  constants.PLOT_PARAMS['4_colours'],
                  legend_on=legend2)
    plot_subspace(ax2,
                  delay2_cued_subspace['3Dcoords'][:n_colours,:],
                  delay2_cued_subspace['plane1'].components_)
    plot_subspace(ax2,
                  delay2_cued_subspace['3Dcoords'][n_colours:,:],
                  delay2_cued_subspace['plane2'].components_)
    # ax2.set_title('Angle: %.1f °' %theta_post)
    ax2.set_title('post-cue')
    helpers.equal_axes(ax2)
    ax2.tick_params(pad = 4.0)

    
    print('Theta pre: %.1f' %delay1_cued_subspace['theta'])
    print('Theta post: %.1f' %delay2_cued_subspace['theta'])
    
    if constants.PARAMS['experiment_number'] == 3:
        ax3 = plt.subplot(1,n_subplots,3, projection='3d')
        plot_geometry(ax3, 
                      probe_cued_subspace['3Dcoords'],
                      probe_cued_subspace['pca'],
                      constants.PLOT_PARAMS['4_colours'])
        plot_subspace(ax3,
                      probe_cued_subspace['3Dcoords'][:n_colours,:],
                      probe_cued_subspace['plane1'].components_)
        plot_subspace(ax3,
                      probe_cued_subspace['3Dcoords'][n_colours:,:],
                      probe_cued_subspace['plane2'].components_)
        # ax2.set_title('Angle: %.1f °' %theta_post)
        ax3.set_title('post-probe')
        helpers.equal_axes(ax3)
        ax3.tick_params(pad = 4.0)
        print('Theta probe: %.1f' %probe_cued_subspace['theta'])

    plt.tight_layout()
    plt.savefig(constants.PARAMS['FIG_PATH']+'model_averaged_cued_geometry.png')


def get_averaged_uncued_subspaces(constants,trial_type='valid'):
    
    load_path = constants.PARAMS['FULL_PATH']+'pca_data/'+trial_type+'_trials/'
    model_av_binned_uncued = pickle.load(open(load_path+'model_av_binned_uncued.pckl','rb'))
    
    delay1_uncued = model_av_binned_uncued['delay1']
    delay2_uncued = model_av_binned_uncued['delay2']

    
    delay1_uncued_subspace = run_pca_pipeline(constants, delay1_uncued, ['down','up'])
    delay2_uncued_subspace = run_pca_pipeline(constants, delay2_uncued, ['down','up'])

    
    # save data
    uncued_subspaces = {}
    uncued_subspaces['delay1'] = delay1_uncued_subspace
    uncued_subspaces['delay2'] = delay2_uncued_subspace
    
    pickle.dump(uncued_subspaces,open(load_path+'uncued_subspace.pckl','wb'))
    
    return uncued_subspaces


def get_averaged_trial_type_subspaces(constants):
    # only for delay2
    
    load_path = constants.PARAMS['FULL_PATH']+'pca_data/valid_trials/' 
    n_colours = constants.PARAMS['B']
    
    # load cued data, binned into 4 colour categories
    obj = pickle.load(open(load_path+'model_av_binned_cued.pckl','rb'))
    delay2_cued = obj['delay2']
    
    # load uncued data
    obj = pickle.load(open(load_path+'model_av_binned_uncued.pckl','rb'))
    delay2_uncued = obj['delay2']
    
    # 'cued-up'
    # find the subspace that explains the cued and uncued datapoints on cued-up trials
    delay2_up_trials = torch.cat((delay2_cued[:n_colours,:],delay2_uncued[:n_colours,:]))
    # first half rows are cued colour averages, the other - uncued
    # repeat for cued-down trials
    delay2_down_trials = torch.cat((delay2_uncued[n_colours:,:],delay2_cued[n_colours:,:]))
    
    
    delay2_up_trials_subspace = run_pca_pipeline(constants, delay2_up_trials, ['cued_up','uncued_down'])
    delay2_down_trials_subspace = run_pca_pipeline(constants, delay2_down_trials, ['uncued_up','cued_down'])
    
    trial_type_subspaces = {}
    trial_type_subspaces['cued_up'] = delay2_up_trials_subspace
    trial_type_subspaces['cued_down'] = delay2_down_trials_subspace
    
    pickle.dump(trial_type_subspaces,
                open(load_path+'trial_type_subspaces.pckl','wb'))
    
    return trial_type_subspaces


def get_trial_type_subspaces_indivModels(constants,trial_type='valid'):
    
    n_colours = constants.PARAMS['B']
    load_path = constants.PARAMS['FULL_PATH']+'pca_data/'+trial_type+'_trials/' 
    
    theta = np.empty((constants.PARAMS['n_models'],2))
    psi = np.empty((constants.PARAMS['n_models'],2))
    
    for model in range(constants.PARAMS['n_models']):
        # load data
        path_cued = load_path + 'pca_data_model' + str(model) + '.pckl'
        path_uncued = load_path + 'pca_data_uncued_model' + str(model) + '.pckl'

        data_binned_cued = pickle.load(open(path_cued,'rb'))
        data_binned_uncued = pickle.load(open(path_uncued,'rb'))
        
        # 'cued-up'
        delay2_up_trials = torch.cat((data_binned_cued['delay2'][:n_colours,:],
                                      data_binned_uncued['delay2'][:n_colours,:]))
        # 'cued-down'
        delay2_down_trials = torch.cat((data_binned_cued['delay2'][n_colours:,:],
                                        data_binned_uncued['delay2'][n_colours:,:]))
        
        
        delay2_up_trials_subspace = \
            run_pca_pipeline(constants,
                             delay2_up_trials,
                             ['cued_up','uncued_down'])
        delay2_down_trials_subspace = \
            run_pca_pipeline(constants, 
                             delay2_down_trials, 
                             ['cued_down','uncued_up'])
        
        trial_type_subspaces = {}
        trial_type_subspaces['cued_up'] = delay2_up_trials_subspace
        trial_type_subspaces['cued_down'] = delay2_down_trials_subspace
        
        # save angles
        theta[model,0] = delay2_up_trials_subspace['theta']
        theta[model,1] = delay2_down_trials_subspace['theta']
        
        psi[model,0] = delay2_up_trials_subspace['psi']
        psi[model,1] = delay2_down_trials_subspace['psi']

        pickle.dump(trial_type_subspaces,
                    open(load_path+'trial_type_subspaces_model'+str(model)+'.pckl','wb'))
        
    pickle.dump(theta,
                    open(load_path+'cued_vs_uncued_theta.pckl','wb'))
    pickle.dump(psi,
                    open(load_path+'cued_vs_uncued_psi.pckl','wb'))
    
        
def get_uncued_subspaces_indivModels(constants,trial_type='valid'):
    load_path = constants.PARAMS['FULL_PATH']+'pca_data/'+trial_type+'_trials/' 
    
    all_plane_angles = []
    psi = []
    for model in range(constants.PARAMS['n_models']):
        # load data
        model_number = str(model)
        f = open(load_path+'/pca_data_uncued_model' + model_number + '.pckl', 'rb')
        pca_data_uncued = pickle.load(f)    
        delay2 = pca_data_uncued['delay2']
        f.close()
        
        delay2_uncued_subspace = \
            run_pca_pipeline(constants,
                             delay2,
                             ['uncued_down','uncued_up'])
            
        psi.append(delay2_uncued_subspace['psi'])
        
            
        pickle.dump(delay2_uncued_subspace,
                    open(load_path+'delay2_uncued_subspace_model'+str(model)+'.pckl','wb'))
        
        all_plane_angles.append(delay2_uncued_subspace['theta'])
    
    pickle.dump(all_plane_angles,
                open(load_path+'all_theta_uncued_post-cue.pckl','wb'))
    pickle.dump(psi,
                open(load_path+'all_psi_uncued_post-cue.pckl','wb'))


def get_CDI(constants):
    n_colours = constants.PARAMS['B']
    load_path = constants.PARAMS['FULL_PATH']+'pca_data/valid_trials/' 
    CDI = np.empty((constants.PARAMS['n_models'],2,2,2))
    #[model, cued location, pre-post, cued/uncued]
    for model in range(constants.PARAMS['n_models']):
        path_cued = load_path + 'pca_data_model' + str(model) + '.pckl'
        path_uncued = load_path + 'pca_data_uncued_model' + str(model) + '.pckl'

        data_binned_cued = pickle.load(open(path_cued,'rb'))
        data_binned_uncued = pickle.load(open(path_uncued,'rb'))
        
        # cued-up trials
        cued_up = torch.cat((data_binned_cued['delay1'][:n_colours,:],
                             data_binned_uncued['delay1'][:n_colours,:],
                               data_binned_cued['delay2'][:n_colours,:],
                               data_binned_uncued['delay2'][:n_colours,:]))
        #pre-cue:up;pre-cue:down;post-cue:u;post-cue:down
        cued_up_pca, cued_up_3Dcoords = get_3D_coords(cued_up)
        
        #pre-cue
        # CDI[model,0,0,0] = ConvexHull(cued_up_3Dcoords[:n_colours,:]).area
        # CDI[model,0,0,1] = ConvexHull(cued_up_3Dcoords[n_colours:n_colours*2,:]).area
        # #post-cue
        # CDI[model,0,1,0] = ConvexHull(cued_up_3Dcoords[n_colours*2:n_colours*3,:]).area
        # CDI[model,0,1,1] = ConvexHull(cued_up_3Dcoords[n_colours*3:,:]).area
        
        #pre-cue
        CDI[model,0,0,0] = quadrilatArea(cued_up_3Dcoords[:n_colours,:])
        CDI[model,0,0,1] = quadrilatArea(cued_up_3Dcoords[n_colours:n_colours*2,:])
        #post-cue
        CDI[model,0,1,0] = quadrilatArea(cued_up_3Dcoords[n_colours*2:n_colours*3,:])
        CDI[model,0,1,1] = quadrilatArea(cued_up_3Dcoords[n_colours*3:,:])
        
        
        
        
        # cued-down trials
        cued_down = torch.cat((data_binned_cued['delay1'][n_colours:,:],
                               data_binned_uncued['delay1'][n_colours:,:],
                               data_binned_cued['delay2'][n_colours:,:],
                               data_binned_uncued['delay2'][n_colours:,:]))
        # pre-cue:down; pre-cue:up;post-cue:down;post-cue:up
        cued_down_pca, cued_down_3Dcoords = get_3D_coords(cued_down)
        
        # #pre-cue
        # CDI[model,1,0,0] = ConvexHull(cued_down_3Dcoords[:n_colours,:]).area
        # CDI[model,1,0,1] = ConvexHull(cued_down_3Dcoords[n_colours:n_colours*2,:]).area
        # #post-cue
        # CDI[model,1,1,0] = ConvexHull(cued_down_3Dcoords[n_colours*2:n_colours*3,:]).area
        # CDI[model,1,1,1] = ConvexHull(cued_down_3Dcoords[n_colours*3:,:]).area
        
        #pre-cue
        CDI[model,1,0,0] = quadrilatArea(cued_down_3Dcoords[:n_colours,:])
        CDI[model,1,0,1] = quadrilatArea(cued_down_3Dcoords[n_colours:n_colours*2,:])
        #post-cue
        CDI[model,1,1,0] = quadrilatArea(cued_down_3Dcoords[n_colours*2:n_colours*3,:])
        CDI[model,1,1,1] = quadrilatArea(cued_down_3Dcoords[n_colours*3:,:])
        
    # average across the trial types
    CDI_av = CDI.mean(1)
    # for pre-cue, average the cued and uncued
    CDI_av = np.concatenate((np.expand_dims(CDI_av[:,0,:].mean(-1),-1),CDI_av[:,1,:]),1)
    CDI_av_df = pd.DataFrame(CDI_av,columns=['pre-cue','post_cued','post_uncued'])
    
    # # save structure
    CDI_av_df.to_csv(load_path+'/CDI.csv')
    
    return CDI_av


def quadrilatArea(coords):
    # get diagonals
    ac = coords[2,:]-coords[0,:]
    bd = coords[3,:]-coords[1,:]
   
    # calculate area
    area = np.linalg.norm(np.cross(ac,bd)) / 2
    return area


def plot_CDI(constants,CDI,logTransform=True):
    
    if logTransform:
        CDI = np.log(CDI)
    pal = sns.color_palette("dark")
    cols = [pal[9],pal[2]]
    
    plt.figure(figsize=(6.65,5))
    ax = plt.subplot(111)
    
    ms = 16
    
    for model in range(constants.PARAMS['n_models']):
        ax.plot([0,1-.125],CDI[model,:2],'k-',alpha=.2)
        ax.plot([0,1+.125],CDI[model,[0,2]],'k-',alpha=.2)

        ax.plot(0,CDI[model,0],'o',c='k',markersize=ms) # pre-cue
        ax.plot(1-.125,CDI[model,1],'^',c=cols[0],markersize=ms) # cued
        ax.plot(1+.125,CDI[model,2],'X',c=cols[1],markersize=ms) # uncued

    # add means
    means = CDI.mean(0)
    ax.bar(0,means[0],facecolor='k',alpha=.2,width=.25)
    ax.bar(1-.125,means[1],facecolor=cols[0],alpha=.2,width=.25,label='cued')
    ax.bar(1+.125,means[2],facecolor=cols[1],alpha=.2,width=.25,label='uncued')
    
    ax.set_xlim([-0.25,1.375])

    
    
    
    
    # x_vals = np.arange(2)
    # sem_cued = np.std(CDI.mean(1)[:,:,0],0) / np.sqrt(constants.PARAMS['n_models'])
    # sem_uncued = np.std(CDI.mean(1)[:,:,1],0) / np.sqrt(constants.PARAMS['n_models'])

    # ax.errorbar(x_vals,
    #             CDI.mean(1)[:,:,0].mean(0),
    #             fmt='.-',
    #             yerr = sem_cued,
    #             c = cols[0],
    #             label='cued')
    #             # ecolor=cols[0],
    #             # mfc=cols[0],
    #             # mec=cols[0])
    # ax.errorbar(x_vals,
    #             CDI.mean(1)[:,:,1].mean(0),
    #             fmt='.-',
    #             yerr = sem_uncued,
    #             c=cols[1],
    #             label = 'uncued')
                # mfc=cols[1],
                # mec=cols[1])
    ax.set_xticks(range(2))
    ax.set_xticklabels(labels=['pre-cue', 'post-cue'])
    ax.set_xlim([-.25,1.25])
    if logTransform:
        ax.set_ylabel('log(CDI)')
    else:
        ax.set_ylabel('CDI')
    
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(constants.PARAMS['FIG_PATH']+'CDI.png')


def test_CDI_contrasts(CDI):
    # first contrast: cued >> uncued
    # first contrast: cued >> pre-cue
    # second contrast: uncued == pre-cue
    
    contrasts = ['Contrast 1: cued >> pre-cue','Contrast 2: uncued = pre-cue']
    ixs = np.array([[1,2],[1,0],[2,0]])
    alternatives = ['greater','greater','two-sided']
    for c in range(2):
        print(contrasts[c])
        s,p = shapiro(CDI[:,ixs[c,0]]-CDI[:,ixs[c,0]])   
        
        if p <= .05:
            # data normally distributed - do 1-samp t-test
            print('    1-samp t-test')
            stat,pval = ttest_1samp(CDI[:,ixs[c,0]]-CDI[:,ixs[c,2]],
                                    0,
                                    alternative='greater')
        else:
            # see if log-transform makes the distribution normal
            s1,p1 = shapiro(np.log(CDI[:,ixs[c,0]]-CDI[:,ixs[c,0]]))
            if p1 <= .05:
                # do 1-samp t-test
                print('    1-samp t-test,log-transformed data')
                stat,pval = ttest_1samp(np.log(CDI[:,ixs[c,0]]-CDI[:,ixs[c,0]]),
                                        0,
                                        alternative='greater')
            else:
                # wilcoxon test
                print('    wilcoxon test')
                stat,pval = wilcoxon(CDI[:,ixs[c,0]]-CDI[:,ixs[c,0]],alternative='greater')
    return          
                
    
def plot_uncued_subspaces(constants,uncued_subspaces,trial_type_subspaces):
    # uncued post-cue
    plot_geom_and_subspace(constants,uncued_subspaces['delay2'],'uncued post-cue ',
                           custom_labels = ['L1 uncued', 'L2 uncued'])
    plt.savefig(constants.PARAMS['FIG_PATH'] + 'uncued_post-cue_geometry.png')
    
    # cued-up trials
    plot_geom_and_subspace(constants,trial_type_subspaces['cued_up'],
                           'cued-up trials',
                           custom_labels = ['L1 cued', 'L2 uncued'])
    plt.savefig(constants.PARAMS['FIG_PATH'] + 'cued_up_uncued_down_geometry.png')
    
    # cued-down trials
    plot_geom_and_subspace(constants,trial_type_subspaces['cued_down'],
                           'cued-down trials',
                           custom_labels=['L1 uncued', 'L2 cued'])
    
    plt.savefig(constants.PARAMS['FIG_PATH'] + 'cued_down_uncued_up_geometry.png')


def run_uncued_analysis(constants):
    # need to add the hyperalignment part here at some point
    uncued_subspaces = get_averaged_uncued_subspaces(constants)
    trial_type_subspaces = get_averaged_trial_type_subspaces(constants)
    
    plot_uncued_subspaces(constants,uncued_subspaces,trial_type_subspaces)
    
    CDI = get_CDI(constants)
    plot_CDI(constants,CDI)
    
    return uncued_subspaces, trial_type_subspaces



        

# def get_model_averages_uncued(constants,plot=True):
#     load_path = constants.PARAMS['FULL_PATH']+'pca_data/valid_trials/' 
#     print('get_model_averages_uncued : rewrite into smaller funcs')
    
    
#     ##### delete later
#     import os
#     if helpers.check_file_exists(load_path+'pca_data_hyperaligned.pckl'):
#         os.remove(load_path+'pca_data_hyperaligned.pckl',load_path+'delay_data_hyperaligned.pckl')
#     # #####
    
#     if helpers.check_file_exists(load_path+'delay_data_hyperaligned.pckl'):
#         # load data
#         delay_data_hyperaligned = \
#             pickle.load(open(load_path+'delay_data_hyperaligned.pckl','rb'))
#     else:
#         # hyperalign data
#         delay_data_hyperaligned = hyperalign_delay_data(constants)
        
#     # load cued data
#     obj = pickle.load(open(load_path+'model_averaged_pca_cued.pckl','rb'))
#     delay2_cued = obj['post-cue']['pca_data']
    
#     # load uncued data
#     obj = pickle.load(open(load_path+'uncued_subspace.pckl','rb'))
#     delay2_cued = obj['post-cue']['pca_data']
    
    
    
    
    
#     # load cued subspace in post-cue delay
#     # obj = pickle.load(open(load_path+'model_averaged_pca_cued.pckl','rb'))
#     # delay2_pca_cued = obj['post-cue']['pca']
#     # delay2_3Dcoords_cued = obj['post-cue']['3Dcoords']
    
#     # # project uncued data into the subspace
#     # # up
#     # uncued_up_projected_into_cued = delay2_pca_cued.transform(delay2_uncued_up)
#     # uncued_down_projected_into_cued = delay2_pca_cued.transform(delay2_uncued_down)
    
#     # uncued_up_projected_into_cued_plane = get_best_fit_plane(uncued_up_projected_into_cued)
#     # uncued_down_projected_into_cued_plane = get_best_fit_plane(uncued_down_projected_into_cued)
    
#     # cued_up_plane = get_best_fit_plane(delay2_3Dcoords_cued[:n_colours,:])
#     # cued_down_plane = get_best_fit_plane(delay2_3Dcoords_cued[n_colours:,:])
    
#     # load cued data
#     obj = pickle.load(open(load_path+'model_averaged_pca_cued.pckl','rb'))
#     delay2_cued = obj['post-cue']['pca_data']
#     # find the subspace that explains the cued and uncued datapoints on cued-up trials
#     delay2_up_trials = torch.cat((delay2_cued[:n_colours,:],delay2_uncued[:n_colours,:]))
#     # first half rows are cued colour averages, the other - uncued
#     # repeat for cued-down trials
#     delay2_down_trials = torch.cat((delay2_cued[n_colours:,:],delay2_uncued[n_colours:,:]))
    
#     # do PCA 1
#     delay2_up_trials_pca, delay2_up_trials_3Dcoords = get_3D_coords(delay2_up_trials)
#     delay2_down_trials_pca, delay2_down_trials_3Dcoords = get_3D_coords(delay2_down_trials)
    
#     # do PCA 2
#     delay2_up_trials_planeCued = get_best_fit_plane(delay2_up_trials_3Dcoords[:n_colours,:])
#     delay2_up_trials_planeUncued = get_best_fit_plane(delay2_up_trials_3Dcoords[n_colours:,:])

#     delay2_down_trials_planeCued = get_best_fit_plane(delay2_down_trials_3Dcoords[:n_colours,:])
#     delay2_down_trials_planeUncued = get_best_fit_plane(delay2_down_trials_3Dcoords[n_colours:,:])
    


    
#     fig_path = constants.PARAMS['FIG_PATH']
    
#     # plot
#     if plot:
        

#         # uncued L1 in cued subspace (with L2 shown)
#         # points =np.concatenate((delay2_3Dcoords_cued[:n_colours,:],
#         #                               uncued_down_projected_into_cued))
#         # plt.figure(figsize=(9,6))
#         # ax2 = plt.subplot(111, projection='3d')

#         # plot_geometry(ax2, 
#         #               points,
#         #               delay2_pca_cued,
#         #               constants.PLOT_PARAMS['4_colours'],
#         #               custom_labels = ['loc1 cued', 'loc2 uncued'])
#         # plot_subspace(ax2,delay2_3Dcoords_cued[:n_colours,:],
#         #               cued_up_plane.components_,fc='k',a=0.2)
#         # plot_subspace(ax2,uncued_down_projected_into_cued,
#         #               uncued_down_projected_into_cued_plane.components_,
#         #               fc='k',a=0.2)
#         # theta = get_angle_between_planes_corrected(points,
#         #                                            uncued_down_projected_into_cued_plane.components_,
#         #                                            cued_up_plane.components_)
#         # ax2.set_title('uncued-down in cued, '+ r'$\theta$' + ' = %.1f' %theta)
#         # helpers.equal_axes(ax2)
#         # ax2.tick_params(pad = 4.0)
#         # plt.tight_layout()
#         # plt.savefig(fig_path + 'uncued-down_in_cued_geometry.png')


        
#         # # uncued L2 in cued subspace (with L1 shown)
#         # points = np.concatenate((uncued_up_projected_into_cued,
#         #                               delay2_3Dcoords_cued[n_colours:,:],))
#         # plt.figure(figsize=(9,6))
#         # ax3 = plt.subplot(111, projection='3d')
#         # plot_geometry(ax3, 
#         #               points,
#         #               delay2_pca_cued,
#         #               constants.PLOT_PARAMS['4_colours'],
#         #               custom_labels = ['loc1 uncued', 'loc2 cued'])
#         # plot_subspace(ax3,uncued_up_projected_into_cued,
#         #               uncued_up_projected_into_cued_plane.components_,fc='k',a=0.2)
#         # plot_subspace(ax3,delay2_3Dcoords_cued[n_colours:,:],
#         #               cued_down_plane.components_,fc='k',a=0.2)
#         # theta = get_angle_between_planes_corrected(points,
#         #                                            uncued_up_projected_into_cued_plane.components_,
#         #                                            cued_down_plane.components_)
#         # ax3.set_title('uncued-up in cued, ' + r'$\theta$' + ' = %.1f' %theta)
#         # helpers.equal_axes(ax3)
#         # ax3.tick_params(pad = 4.0)
#         # plt.tight_layout()
#         # plt.savefig(fig_path + 'uncued-up_in_cued_geometry.png')
        
#         # cued L1 trials - cued and uncued subspaces
#         plt.figure(figsize=(9,6))
#         ax2 = plt.subplot(111, projection='3d')

#         plot_geometry(ax2, 
#                       delay2_up_trials_3Dcoords,
#                       delay2_up_trials_pca,
#                       constants.PLOT_PARAMS['4_colours'],
#                       custom_labels = ['loc1 cued', 'loc2 uncued'])
#         plot_subspace(ax2,delay2_up_trials_3Dcoords[:n_colours,:],
#                       delay2_up_trials_planeCued.components_,fc='k',a=0.2)
#         plot_subspace(ax2,delay2_up_trials_3Dcoords[n_colours:,:],
#                       delay2_up_trials_planeUncued.components_,fc='k',a=0.2)
        
#         theta = get_angle_between_planes_corrected(delay2_up_trials_3Dcoords,
#                                                     delay2_up_trials_planeCued.components_,
#                                                     delay2_up_trials_planeUncued.components_)
#         ax2.set_title('cued-up and uncued-down, '+ r'$\theta$' + ' = %.1f' %theta)
#         helpers.equal_axes(ax2)
#         ax2.tick_params(pad = 4.0)
#         plt.tight_layout()
#         plt.savefig(fig_path + 'cued_up_uncued_down_geometry.png')
        
#         # cued L2 trials - cued and uncued subspaces
#         plt.figure(figsize=(9,6))
#         ax3 = plt.subplot(111, projection='3d')

#         plot_geometry(ax3, 
#                       delay2_down_trials_3Dcoords,
#                       delay2_down_trials_pca,
#                       constants.PLOT_PARAMS['4_colours'],
#                       custom_labels = ['loc1 uncued', 'loc2 cued'])
#         plot_subspace(ax3,delay2_down_trials_3Dcoords[:n_colours,:],
#                       delay2_down_trials_planeCued.components_,fc='k',a=0.2)
#         plot_subspace(ax3,delay2_down_trials_3Dcoords[n_colours:,:],
#                       delay2_down_trials_planeUncued.components_,fc='k',a=0.2)
        
#         theta = get_angle_between_planes_corrected(delay2_down_trials_3Dcoords,
#                                                     delay2_down_trials_planeCued.components_,
#                                                     delay2_down_trials_planeUncued.components_)
#         ax3.set_title('cued-down and uncued-up, '+ r'$\theta$' + ' = %.1f' %theta)
#         helpers.equal_axes(ax3)
#         ax3.tick_params(pad = 4.0)
#         plt.tight_layout()
#         plt.savefig(fig_path + 'cued_down_uncued_up_geometry.png')

           

    #return delay1_pca, delay1_3Dcoords, delay1_planeUp, delay1_planeDown, delay2_pca, delay2_3Dcoords, delay2_planeUp, delay2_planeDown


def get_cued_subspaces_indivModels(constants):
    '''
    run the pca pipeline for individual models, to get the angles etc

    Parameters
    ----------
    constants : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    load_path = constants.PARAMS['FULL_PATH']+'pca_data/valid_trials/'
    if constants.PARAMS['experiment_number'] == 3:
        n_timepoints = 3
    else:
        n_timepoints = 2
    theta = np.empty((constants.PARAMS['n_models'],n_timepoints))
    PVEs = np.empty((constants.PARAMS['n_models'],n_timepoints,3))
    psi = np.empty((constants.PARAMS['n_models'],n_timepoints))
    
    for model in range(constants.PARAMS['n_models']):
        # load pre- and post-cue delay data
        f = open(load_path+'pca_data_model'+str(model)+'.pckl','rb')
        pca_data = pickle.load(f)
        f.close()
        delay1 = pca_data['delay1']
        delay2 = pca_data['delay2']
        
        delay1_subspace = run_pca_pipeline(constants,
                             delay1,
                             ['cued_up','cued_down'])
        delay2_subspace = run_pca_pipeline(constants,
                             delay2,
                             ['cued_up','cued_down'])

        cued_subspaces = {}
        cued_subspaces['delay1'] = delay1_subspace
        cued_subspaces['delay2'] = delay2_subspace

        theta[model,0] = delay1_subspace['theta']
        theta[model,1] = delay2_subspace['theta']
        
        PVEs[model,0,:] = delay1_subspace['pca'].explained_variance_ratio_
        PVEs[model,1,:] = delay2_subspace['pca'].explained_variance_ratio_
        
        psi[model,0] = delay1_subspace['psi']
        psi[model,1] = delay2_subspace['psi']
        
        
        if constants.PARAMS['experiment_number'] == 3:
                probe = pca_data['data'][:,-1,:]
                probe_subspace = run_pca_pipeline(constants,
                                 probe,
                                 ['cued_up','cued_down'])
                cued_subspaces['probe'] = probe_subspace
                theta[model,2] = probe_subspace['theta']
                PVEs[model,2,:] = probe_subspace['pca'].explained_variance_ratio_
                psi[model,2] = probe_subspace['psi']


        pickle.dump(cued_subspaces,
                        open(load_path+'cued_subspaces_model'+str(model)+'.pckl','wb'))
    pickle.dump(theta,
                    open(load_path+'all_theta.pckl','wb'))
    pickle.dump(PVEs,
                    open(load_path+'all_PVEs_3D.pckl','wb'))
    pickle.dump(psi,
                    open(load_path+'all_psi.pckl','wb'))
    return theta


def plot_cued_subspaces_indivModels(constants):
    load_path = constants.PARAMS['FULL_PATH']+'pca_data/valid_trials/'
    n_colours = constants.PARAMS['B']
    
    if constants.PARAMS['experiment_number'] == 3:
        n_subplots = 3
        fsize = (16,5)
    else:
        n_subplots = 2
        fsize = (12,5)
        
    for model in range(constants.PARAMS['n_models']):
        # load data
        cued_subspaces = pickle.load(open(load_path+'cued_subspaces_model'+str(model)+'.pckl','rb'))
        delay1_subspace = cued_subspaces['delay1']
        delay2_subspace = cued_subspaces['delay2'] 
        
        plt.figure(figsize=fsize,num=('Model '+str(model)))
        ax = plt.subplot(1,n_subplots,1, projection='3d')
        plot_geometry(ax, 
                      delay1_subspace['3Dcoords'],
                      delay1_subspace['pca'],
                      constants.PLOT_PARAMS['4_colours'],
                      legend_on=False)
        plot_subspace(ax,
                      delay1_subspace['3Dcoords'][:n_colours,:],
                      delay1_subspace['plane1'].components_,fc='k',a=0.2)
        plot_subspace(ax,
                      delay1_subspace['3Dcoords'][n_colours:,:],
                      delay1_subspace['plane2'].components_,fc='k',a=0.2)
        ax.set_title('Angle: %.1f' %delay1_subspace['theta'])
        helpers.equal_axes(ax)
        
        ax2 = plt.subplot(1,n_subplots,2, projection='3d')
        plot_geometry(ax2, 
                      delay2_subspace['3Dcoords'], 
                      delay2_subspace['pca'], 
                      constants.PLOT_PARAMS['4_colours'])
        plot_subspace(ax2,
                      delay2_subspace['3Dcoords'][:n_colours,:],
                      delay2_subspace['plane1'].components_,fc='k',a=0.2)
        plot_subspace(ax2,
                      delay2_subspace['3Dcoords'][n_colours:,:],
                      delay2_subspace['plane2'].components_,fc='k',a=0.2)
        ax2.set_title('Angle: %.1f' %delay2_subspace['theta'])
        helpers.equal_axes(ax2)
        
        if constants.PARAMS['experiment_number'] == 3:
            probe_subspace = cued_subspaces['probe']        

            ax3 = plt.subplot(1,n_subplots,3, projection='3d')
            plot_geometry(ax3, 
                          probe_subspace['3Dcoords'], 
                          probe_subspace['pca'], 
                          constants.PLOT_PARAMS['4_colours'])
            plot_subspace(ax3,
                          probe_subspace['3Dcoords'][:n_colours,:],
                          probe_subspace['plane1'].components_,fc='k',a=0.2)
            plot_subspace(ax3,
                          probe_subspace['3Dcoords'][n_colours:,:],
                          probe_subspace['plane2'].components_,fc='k',a=0.2)
            ax3.set_title('Angle: %.1f' %probe_subspace['theta'])
            helpers.equal_axes(ax3)

    
def run_2step_pca(constants,plot=False):
    '''
    run the pca pipeline for individual models, to get the angles etc

    Parameters
    ----------
    constants : TYPE
        DESCRIPTION.
    plot : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    '''
    load_path = constants.PARAMS['FULL_PATH']+'pca_data/valid_trials/'
    n_colours = constants.PARAMS['B']
    
    # check that paths to save data exist
    # helpers.check_path(load_path + '/angles/')
    # helpers.check_path(load_path + '/pca3/')
    # helpers.check_path(load_path + '/pca2/')
    # helpers.check_path(load_path + '/planes/')
    
    if constants.PARAMS['experiment_number'] == 3:
        n_timepoints = 3
    else:
        n_timepoints = 2

    all_plane_angles = np.empty((constants.PARAMS['n_models'],n_timepoints))
    all_PVEs_3D = np.empty((constants.PARAMS['n_models'],n_timepoints,3))
    for model in range(constants.PARAMS['n_models']):
        # load pre- and post-cue delay data
        f = open(load_path+'pca_data_model'+str(model)+'.pckl','rb')
        pca_data = pickle.load(f)
        f.close()
        delay1 = pca_data['delay1']
        delay2 = pca_data['delay2']
        
        ### PCA 1
        
        ### old code - delete
        # # demean data
        # delay1 -= delay1.mean()
        # delay2 -= delay2.mean()
        
        # #% run first PCA to get down to 3D space
        # delay1_pca = PCA(n_components=3) # Initializes PCA
        # delay2_pca = PCA(n_components=3) # Initializes PCA
        
        # delay1_3Dcoords = delay1_pca.fit_transform(delay1) # get coordinates in the reduced-dim space
        # delay2_3Dcoords = delay2_pca.fit_transform(delay2)
        
        ###
        delay1_subspace = run_pca_pipeline(constants,delay1,['cued_up','cued_down'])
        delay2_subspace = run_pca_pipeline(constants,delay2,['cued_up','cued_down'])
        all_plane_angles[model,0] = delay1_subspace['theta']
        all_plane_angles[model,1] = delay2_subspace['theta']
        all_PVEs_3D[model,0,:] = delay1_subspace['pca'].explained_variance_ratio_
        all_PVEs_3D[model,1,:] = delay2_subspace['pca'].explained_variance_ratio_
        
        if constants.PARAMS['experiment_number'] == 3:
            probe = pca_data['data'][:,-1,:]
            print('change this to a probe timepoint')
            probe_subspace = run_pca_pipeline(constants,probe,['cued_up','cued_down'])
            
            all_plane_angles[model,2] = probe_subspace['theta']
            all_PVEs_3D[model,2,:] = probe_subspace['pca'].explained_variance_ratio_

        # if plot:
        #     plt.figure(figsize=(12,5),num=('Model '+str(model)))
        #     ax = plt.subplot(121, projection='3d')
        #     plot_geometry(ax, delay1_3Dcoords, delay1_pca, constants.PLOT_PARAMS['4_colours'],legend_on=False)
        #     plot_subspace(ax,delay1_3Dcoords[:n_colours,:],delay1_planeUp.components_,fc='k',a=0.2)
        #     plot_subspace(ax,delay1_3Dcoords[n_colours:,:],delay1_planeDown.components_,fc='k',a=0.2)
        #     ax.set_title('Angle: %.1f' %theta_pre)
        #     helpers.equal_axes(ax)
           
        #     ax2 = plt.subplot(122, projection='3d')
        #     plot_geometry(ax2, delay2_3Dcoords, delay2_pca, constants.PLOT_PARAMS['4_colours'])
        #     plot_subspace(ax2,delay2_3Dcoords[:n_colours,:],delay2_planeUp.components_,fc='k',a=0.2)
        #     plot_subspace(ax2,delay2_3Dcoords[n_colours:,:],delay2_planeDown.components_,fc='k',a=0.2)
        #     ax2.set_title('Angle: %.1f' %theta_post)
        #     helpers.equal_axes(ax2)
     
        
    pickle.dump(all_plane_angles,open(load_path+'all_plane_angles.pckl','wb'))
    pickle.dump(all_PVEs_3D,open(load_path+'all_PVEs_3D.pckl','wb'))
        

def plot_full_data_RDMs(rdm_precue,rdm_postcue):
    '''
    Plot rdms averaged across models - all (combinatorial) conditions

    Parameters
    ----------
    rdm_precue : TYPE
        DESCRIPTION.
    rdm_postcue : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    plt.figure(figsize=(10,20),num='Full data RDMs')
    plt.subplot(121)
    plt.imshow(np.mean(rdm_precue,2),cmap = sns.cubehelix_palette(start=.5,rot=-.75,as_cmap = True,reverse=False))
    plt.xticks([])
    plt.yticks([])
    
    plt.colorbar()
    plt.title('pre-cue')
    
    plt.subplot(122)
    plt.imshow(np.mean(rdm_postcue,2),cmap = sns.cubehelix_palette(start=.5,rot=-.75,as_cmap = True,reverse=True))
    plt.colorbar()
    plt.title('post-cue')


def plot_binned_data_RDMs(pre_data_RDM_averaged,post_data_RDM_averaged):
    '''
    Plot rdms averaged across models - binned across uncued locations

    Parameters
    ----------
    pre_data_RDM_averaged : TYPE
        DESCRIPTION.
    post_data_RDM_averaged : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''

    fig, ax = plt.subplots(1,2,figsize=(13.65,5),num = 'Averaged data RDMs')
    
    
    im1 = ax[0].imshow(pre_data_RDM_averaged,
                       # cmap = sns.color_palette("flare_r",as_cmap = True))
                       cmap = sns.cubehelix_palette(start=.5,rot=-.75,as_cmap = True,reverse=True))
    ax[0].set_title('pre-cue')
    ax[0].set_yticks([])
    ax[0].set_xticks([])
    
    fig.colorbar(im1,ax=ax[0])
    
    
    
    im2 = ax[1].imshow(post_data_RDM_averaged,
                       # cmap = sns.color_palette("flare_r",as_cmap = True))
                       cmap = sns.cubehelix_palette(start=.5,rot=-.75,as_cmap = True,reverse=True))
    ax[1].set_title('post-cue')
    ax[1].set_yticks([])
    ax[1].set_xticks([])
    
    fig.colorbar(im2,ax=ax[1])


def plot_indiv_model_RDMs(constants):
    file_path = constants.PARAMS['FULL_PATH']+'RSA/indiv_model_rdms.pckl'
    pre_RDMs, post_RDMs = pickle.load(open(file_path,'rb'))
    
    for m in range(constants.PARAMS['n_models']):
        fig, ax = plt.subplots(1,2,figsize=(13.65,5),num = 'Model %d' %m)
        im1 = ax[0].imshow(pre_RDMs[:,:,m],
                       # cmap = sns.color_palette("flare_r",as_cmap = True))
                       cmap = sns.cubehelix_palette(start=.5,rot=-.75,as_cmap = True,reverse=True))
        ax[0].set_title('pre-cue')
        ax[0].set_yticks([])
        ax[0].set_xticks([])
        
        fig.colorbar(im1,ax=ax[0])
        
        
        
        im2 = ax[1].imshow(post_RDMs[:,:,m],
                           # cmap = sns.color_palette("flare_r",as_cmap = True))
                           cmap = sns.cubehelix_palette(start=.5,rot=-.75,as_cmap = True,reverse=True))
        ax[1].set_title('post-cue')
        ax[1].set_yticks([])
        ax[1].set_xticks([])
        
        fig.colorbar(im2,ax=ax[1])
    

def RDM_reg(model_RDM,data_RDM):
    '''
    Run RDM-based regression.

    Parameters
    ----------
    model_RDM : TYPE
        DESCRIPTION.
    data_RDM : TYPE
        DESCRIPTION.


    Returns
    -------
    results : TYPE
        DESCRIPTION.

    '''
    if len(model_RDM.shape)>1:
        if model_RDM.shape[0]==model_RDM.shape[1]:
            # if not in diagonal form
            raise ValueError('Please enter the model RDM in diagonal form')
    # z-score RDMs
    model_RDM = zscore(model_RDM)
    # create predictor matrix - add bias
    X = sm.add_constant(model_RDM)
    
    # extract diagonal from data RDM
    y = squareform(data_RDM)
    #z-score
    y = zscore(y)
    
    # run regression
    results = sm.OLS(y, X).fit()
    
    return results


def run_main_RDM_reg(constants,pre_data_RDM_averaged,post_data_RDM_averaged):
    '''
    Run RDM-based regression with ortho- and parallel plane predictors.

    Parameters
    ----------
    constants : TYPE
        DESCRIPTION.
    pre_data_RDM_averaged : TYPE
        DESCRIPTION.
    post_data_RDM_averaged : TYPE
        DESCRIPTION.

    Returns
    -------
    results_pre : TYPE
        DESCRIPTION.
    results_post : TYPE
        DESCRIPTION.

    '''

    # load model RDMs
    path = constants.PARAMS['FULL_PATH'] + 'RSA/model_RDMs_sqform.pckl'
    f = open(path,'rb')
    model_RDMs = pickle.load(f) # individual predictor arrays stacked along axis 1
    f.close()
    # run regression analysis
    
    results_pre = RDM_reg(model_RDMs,pre_data_RDM_averaged)
    results_post = RDM_reg(model_RDMs,post_data_RDM_averaged)

    # print results
    print('RDM-based regression results: pre-cue')
    print(results_pre.summary())
    
    print('RDM-based regression results: post-cue')
    print(results_post.summary())
    
    return results_pre, results_post

class rotRegResultsTbl:
    def __init__(self,theta_range):
        self.theta_range = theta_range
        self.betas = np.empty((len(theta_range),1))
        self.CIs = np.empty((len(theta_range),2))
        self.p_vals = np.empty((len(theta_range),1))

    def add_vals(self,i,results):
        self.betas[i] = results.params[1]
        self.CIs[i,:] = results.conf_int()[1,:]
        self.p_vals[i] = results.pvalues[1]

    def plot(self):
        x = self.theta_range
        y = [self.betas.squeeze(),(np.diff(self.CIs)/2).squeeze()]
        plt.figure(figsize=(8.15,4.8))
        ax = plt.subplot(111)
        
        self.H1,self.H2 = cplot.shadow_plot(ax,x,y,precalc=True,color='k')
        self.H3 = ax.plot(x,np.zeros(len(x)),'k--')
        
        ax.set_xlabel('Theta [°]')
        ax.set_ylabel('Beta')
        
        plt.tight_layout()


def plot_rot_RDM_reg(constants,pre_ortho,pre_parallel,post_ortho,post_parallel):
    fig_path = constants.PARAMS['FIG_PATH']+'rot_regressions/'
    helpers.check_path(fig_path)
    
    pre_ortho.plot()
    plt.title('pre-cue ortho rotations')
    plt.savefig(fig_path+'pre_ortho.png')
    
    pre_parallel.plot()
    plt.title('pre-cue parallel rotations')
    plt.savefig(fig_path+'pre_parallel.png')

    
    post_ortho.plot()
    plt.title('post-cue ortho rotations')
    plt.savefig(fig_path+'post_ortho.png')

    
    post_parallel.plot()
    plt.title('post-cue parallel rotations')
    plt.savefig(fig_path+'post_parallel.png')
    

def run_rotation_RDM_reg(constants,pre_data_RDM_averaged,post_data_RDM_averaged):
    path = constants.PARAMS['FULL_PATH'] + 'RSA/'
    theta_range = pickle.load(open(path+'theta_range.pckl','rb'))
    
    
    pre_ortho = rotRegResultsTbl(theta_range)
    pre_parallel = rotRegResultsTbl(theta_range)
    post_ortho = rotRegResultsTbl(theta_range)
    post_parallel = rotRegResultsTbl(theta_range)
    
    for i, theta_deg in enumerate(theta_range):
        rdm_parallel = pickle.load(open(path+ 'rotated_diagRDMs/'
                                + str(theta_deg)+'.pckl','rb'))
        rdm_ortho = pickle.load(open(path+ 'ortho_rotated_diagRDMs/'
                                + str(theta_deg)+'.pckl','rb'))
        
        # pre-cue
        results_ortho_pre = RDM_reg(rdm_ortho,pre_data_RDM_averaged)
        results_parallel_pre = RDM_reg(rdm_parallel,pre_data_RDM_averaged)

        pre_ortho.add_vals(i,results_ortho_pre)
        pre_parallel.add_vals(i,results_parallel_pre)
        
        # post-cue
        results_ortho_post = RDM_reg(rdm_ortho,post_data_RDM_averaged)
        results_parallel_post = RDM_reg(rdm_parallel,post_data_RDM_averaged)

        post_ortho.add_vals(i,results_ortho_post)
        post_parallel.add_vals(i,results_parallel_post)
    
    return pre_ortho,pre_parallel,post_ortho,post_parallel


def run_mirror_im_RDM_reg(constants,post_data_RDM_averaged):
    path = constants.PARAMS['FULL_PATH'] + 'RSA/'
    flipped_rdm = pickle.load(open(path+'flipped_RDM_sqform.pckl','rb'))
    results = RDM_reg(flipped_rdm, post_data_RDM_averaged)
    print('RDM-based regression: parallel flipped RDM with post-cue data')
    print(results.summary())
    

def plot_main_RDM_reg_results(constants,results_pre,results_post):
    '''
    Plot the results (betas) from the RDM-based regression.

    Parameters
    ----------
    constants : TYPE
        DESCRIPTION.
    results_pre : TYPE
        DESCRIPTION.
    results_post : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    # load model RDM order
    path = constants.PARAMS['FULL_PATH'] + 'RSA/model_RDMs_order.pckl'
    f = open(path,'rb')
    model_RDMs_order = pickle.load(f) # individual predictor arrays stacked along axis 1
    f.close()
    plt.figure(figsize = (9.15,5))
    # plot pre-cue results
    ax1 = plt.subplot(121)
    
    n_predictors = len(model_RDMs_order)
    
    ix = np.arange(1,n_predictors+1)
    
    betas = results_pre.params[ix]
    CIs = results_pre.conf_int()[ix,1] - betas
    ax1.bar(np.arange(len(betas)),betas,color = [.5,.5,.5],yerr = CIs)
    plt.xticks(np.arange(len(betas)),labels = model_RDMs_order,rotation=20)
    
    
    # plot post-cue results
    ax2 = plt.subplot(122,sharex = ax1,sharey = ax1)
    
    betas = results_post.params[ix]
    CIs = results_post.conf_int()[ix,1] - betas
    ax2.bar(np.arange(len(betas)),betas,color = [.5,.5,.5],yerr = CIs)
    
    
    plt.xticks(np.arange(len(betas)),labels = model_RDMs_order,rotation=20)
    ax1.set_ylabel('Beta coefficient')
    ax1.set_title('pre-cue')
    ax2.set_title('post-cue')
    
    plt.tight_layout()


## add - mirror inage regression and rotation regressions



##%% do MDS on model-averaged RDM

def get_MDS_from_RDMs(pre_data_RDM_averaged,post_data_RDM_averaged):
    # mds_stimuli = fit_mds_to_rdm(np.mean(rdm_stimuli_binned,2))
    mds_precue = fit_mds_to_rdm(pre_data_RDM_averaged)
    mds_postcue = fit_mds_to_rdm(post_data_RDM_averaged)
    return mds_precue,mds_postcue


def plot_data_MDS(mds_precue,mds_postcue,constants):
    
    plot_colours = constants.PLOT_PARAMS['4_colours']
    n_colours = len(plot_colours)

    plt.figure()
    ax1 = plt.subplot(121, projection='3d')
    ax2 = plt.subplot(122, projection='3d')
    plot_geometry(ax1,mds_precue,[],plot_colours,legend_on=False)
    plot_geometry(ax2,mds_postcue,[],plot_colours)    
    
    ax1.set_xlabel('dim 1',labelpad=25.0)
    ax1.set_ylabel('dim 2',labelpad=25.0)
    ax1.set_zlabel('dim 3',labelpad=35.0)
    
    ax2.set_xlabel('dim 1',labelpad=25.0)
    ax2.set_ylabel('dim 2',labelpad=30.0)
    ax2.set_zlabel('dim 3',labelpad=35.0)
    
    # ax1.set_yticks([-.1,0,.1])
    # ax1.set_zticks([-.1,0,.1])
    
    
    # ax2.set_xticks([-.1,0,.1])
    # ax2.set_yticks([-.1,0,.1])
    # ax2.set_zticks([-.1,0,.1])
    
    plt.tight_layout()
    plt.subplots_adjust(wspace = 0.2)
    
    
    # plt.subplots_adjust(wspace = 0.4)
    
    helpers.equal_axes(ax1)
    helpers.equal_axes(ax2)
    
    plt.legend(bbox_to_anchor=(1.5,1))
    plt.tight_layout()
    
    
    # ax1.set_xticks(np.arange(-.01,.02,.01))
    # ax1.set_yticks(np.arange(-.01,.02,.01))
    # ax1.set_zticks(np.arange(-.01,.02,.01))
    
    # lims = ax2.get_xlim()
    # ax2.set_xticks(np.arange(-.04,.05,.04))
    # ax2.set_yticks(np.arange(-.04,.05,.04))
    # ax2.set_zticks(np.arange(-.04,.05,.04))
    
    
    
    ax1.set_title('pre-cue')
    ax2.set_title('post-cue')
    
    ax1.tick_params(labelsize = 26)
    ax1.tick_params(axis='x',pad=3)
    ax1.tick_params(axis='y',pad=5)
    ax1.tick_params(axis='z',pad=20)
    
    
    ax2.tick_params(labelsize = 26)
    ax2.tick_params(axis='x',pad=3)
    ax2.tick_params(axis='y',pad=5)
    ax2.tick_params(axis='z',pad=20)
    
    
    #% get and plot planes of best fit
    # PRE_CUE
    print('Data MDS - plane angles ')
    # get directions of max variance, i.e. vectors defining the plane
    delay1_planeUp = get_best_fit_plane(mds_precue[0:n_colours])
    delay1_planeDown = get_best_fit_plane(mds_precue[n_colours:])
    # calculate angle between planes
    theta_pre = get_angle_between_planes_corrected(mds_precue,delay1_planeUp.components_,delay1_planeDown.components_)
    print('Angle pre-cue: %.2f' %theta_pre)
    
    plot_subspace(ax1,mds_precue[:n_colours,:],delay1_planeUp.components_,fc='k',a=0.2)
    plot_subspace(ax1,mds_precue[n_colours:,:],delay1_planeDown.components_,fc='k',a=0.2)
    
    # POST_CUE
    
    delay2_planeUp = get_best_fit_plane(mds_postcue[0:n_colours])
    delay2_planeDown = get_best_fit_plane(mds_postcue[n_colours:])
    
    theta_post = get_angle_between_planes_corrected(mds_postcue,delay2_planeUp.components_,delay2_planeDown.components_,)
    print('Angle post-cue: %.2f' %theta_post)
    
    plot_subspace(ax2,mds_postcue[:n_colours,:],delay2_planeUp.components_,fc='k',a=0.2)
    plot_subspace(ax2,mds_postcue[n_colours:,:],delay2_planeDown.components_,fc='k',a=0.2)
    
    ax1.set_title('pre-cue')
    ax2.set_title('post-cue')
    
    # ax1.set_xlabel('PC1',labelpad = 20,style = 'italic')
    # ax1.set_ylabel('PC2',labelpad = 20,style = 'italic')
    # ax1.set_zlabel('PC3',labelpad = 15,style = 'italic')
    
    # ax2.set_xlabel('PC1',labelpad = 20,style = 'italic')
    # ax2.set_ylabel('PC2',labelpad = 20,style = 'italic')
    # ax2.set_zlabel('PC3',labelpad = 15,style = 'italic')
    


def run_full_rep_geom_analysis(constants):
    print('......REPRESENTATIONAL GEOMETRY ANALYSIS......')
    # get_model_RDMs(constants)
    # # RDM analysis
    rdm_precue,rdm_postcue,pre_data_RDM_averaged,post_data_RDM_averaged = get_data_RDMs(constants)
    # # plot_full_data_RDMs(rdm_precue,rdm_postcue)
    # plot_binned_data_RDMs(pre_data_RDM_averaged,post_data_RDM_averaged)
    # results_pre, results_post = run_main_RDM_reg(constants,pre_data_RDM_averaged,post_data_RDM_averaged)
    # plot_main_RDM_reg_results(constants,results_pre,results_post)
    # pre_ortho,pre_parallel,post_ortho,post_parallel = \
    #     run_rotation_RDM_reg(constants,pre_data_RDM_averaged,post_data_RDM_averaged)
    # plot_rot_RDM_reg(constants,pre_ortho,pre_parallel,post_ortho,post_parallel)  
    # run_mirror_im_RDM_reg(constants,post_data_RDM_averaged)
        
    mds_precue,mds_postcue =get_MDS_from_RDMs(pre_data_RDM_averaged,post_data_RDM_averaged)
    plot_data_MDS(mds_precue,mds_postcue,constants)
    model_av_binned_cued, model_av_binned_uncued = get_model_averages(constants)
    plot_averaged_cued_geometry(constants)
    # run_2step_pca(constants) # to get angle vals for individual models
    _ = get_cued_subspaces_indivModels(constants)
    #plot_cued_subspaces_indivModels(constants)
    uncued_subspaces, trial_type_subspaces = run_uncued_analysis(constants)
    
    
    # get other angles
    get_trial_type_subspaces_indivModels(constants)
    get_uncued_subspaces_indivModels(constants,trial_type='valid')
    
    
    uncued_pre = pickle.load(open(constants.PARAMS['FULL_PATH']+'RSA/'+'pre_data_uncued_RDM_averaged.pckl','rb'))
    uncued = pickle.load(open(constants.PARAMS['FULL_PATH']+'RSA/'+'post_data_uncued_RDM_averaged.pckl','rb'))
    cued_uncued = pickle.load(open(constants.PARAMS['FULL_PATH']+'RSA/'+'post_data_cu_RDM_averaged.pckl','rb'))
    
    mds_uncued = fit_mds_to_rdm(uncued)
    mds_uncued_pre = fit_mds_to_rdm(uncued_pre)
    mds_cu_L1 = fit_mds_to_rdm(cued_uncued[:,:,0])    
    mds_cu_L2 = fit_mds_to_rdm(cued_uncued[:,:,1])    
    
    ix = np.concatenate((np.arange(4,8),np.arange(4)))
    plot_data_MDS(mds_cu_L1,mds_cu_L2[ix,:],constants)
    plot_data_MDS(mds_uncued_pre,mds_uncued,constants)



#%%

def get_angle(constants):
    load_path = constants.PARAMS['FULL_PATH'] +'pca_data/valid_trials/' #'partial_training/'
    delay_len = constants.PARAMS['trial_timings']['delay1_dur']
    
    
    theta_post = np.empty((constants.PARAMS['n_models'],delay_len))
    
    
    
    d2_start = constants.PARAMS['trial_timepoints']['delay2_start']-1
    # loop over models
    for model in range(constants.PARAMS['n_models']):        
        #% load data
        # fully trained
        pca_data_ft = pickle.load(open(load_path+'pca_data_model'+str(model)+'.pckl','rb'))
       
        # run the PCA pipeline on both delays, separately for each timepoint
        for t in range(delay_len):
            subspace_d2 = run_pca_pipeline(constants,
                                           pca_data_ft['data'][:,d2_start+t,:],
                                           ['up','down'])
            theta_post[model,t] = subspace_d2['theta']
    
        
    
    return theta_post

# check the cued0uncued anglels across the entire post-cue delay (and retrocue)

def get_trial_type_subspaces_indivModels(constants,trial_type='valid'):
    n_colours = constants.PARAMS['B']
    load_path = constants.PARAMS['FULL_PATH']+'pca_data/'+trial_type+'_trials/' 
    
    theta = np.empty((constants.PARAMS['n_models'],2))
    psi = np.empty((constants.PARAMS['n_models'],2))
    
    for model in range(constants.PARAMS['n_models']):
        # load data
        path_cued = load_path + 'pca_data_model' + str(model) + '.pckl'
        path_uncued = load_path + 'pca_data_uncued_model' + str(model) + '.pckl'

        data_binned_cued = pickle.load(open(path_cued,'rb'))
        data_binned_uncued = pickle.load(open(path_uncued,'rb'))
        
        # 'cued-up'
        delay2_up_trials = torch.cat((data_binned_cued['delay2'][:n_colours,:],
                                      data_binned_uncued['delay2'][:n_colours,:]))
        # 'cued-down'
        delay2_down_trials = torch.cat((data_binned_cued['delay2'][n_colours:,:],
                                        data_binned_uncued['delay2'][n_colours:,:]))
        
        
        delay2_up_trials_subspace = \
            run_pca_pipeline(constants,
                             delay2_up_trials,
                             ['cued_up','uncued_down'])
        delay2_down_trials_subspace = \
            run_pca_pipeline(constants, 
                             delay2_down_trials, 
                             ['cued_down','uncued_up'])
        
        trial_type_subspaces = {}
        trial_type_subspaces['cued_up'] = delay2_up_trials_subspace
        trial_type_subspaces['cued_down'] = delay2_down_trials_subspace
        
        # save angles
        theta[model,0] = delay2_up_trials_subspace['theta']
        theta[model,1] = delay2_down_trials_subspace['theta']
        
        psi[model,0] = delay2_up_trials_subspace['psi']
        psi[model,1] = delay2_down_trials_subspace['psi']

        pickle.dump(trial_type_subspaces,
                    open(load_path+'trial_type_subspaces_model'+str(model)+'.pckl','wb'))
        
    pickle.dump(theta,
                    open(load_path+'cued_vs_uncued_theta.pckl','wb'))
    pickle.dump(psi,
                    open(load_path+'cued_vs_uncued_psi.pckl','wb'))