#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 15:07:08 2021

@author: emilia
"""

# representational geometry analysis

import numpy as np
import pickle
import os.path
import vec_operations as vops 
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from scipy.spatial.distance import squareform, pdist
from helpers import check_path
import seaborn as sns
import matplotlib.pyplot as plt


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


def get_angle_between_planes(plane1_vecs,plane2_vecs):
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
    # calculate normals for each plane
    normal1 = np.cross(plane1_vecs[0,:],plane1_vecs[1,:])
    normal2 = np.cross(plane2_vecs[0,:],plane2_vecs[1,:])
    
    # since each normal has a length of 1, their dot product will be equal to the cosine of the angle between them
    cos_theta = np.dot(normal1,normal2)
    
    # fix the numerical error if abs(np.dot) > 1
    if cos_theta > 1:
        cos_theta = 1
    elif cos_theta < -1:
        cos_theta = -1
    
    # get the plane angle - in degrees for convenience
    angle_degrees = np.degrees(np.arccos(cos_theta))
    
    return angle_degrees

    
def rotate_plane_by_angle(points,theta_radians):
    """
    Rotates a set of points / a plane by a given degree around the z-axis.


    Parameters
    ----------
    points : (n_points,3) numpy array with the 3D coordinates of the 
            datapoints
        
    theta_radians : angle of rotation

    Returns
    -------
    plane_vecs_aligned: (2,3) an array with two orthogonal vectors defining
        the best-fitting plane. The first vector corresponds to the side of the
        quadrilateral joining points 1 and 2, and the second vector is at a 
        90 degree angle from the first one.
    """ 
    # construct rotation matrix - z component stays the same
    R = np.eye(3)
    R[:,0] = np.array([np.cos(theta_radians),np.sin(theta_radians),0])
    R[:,1] = np.array([-np.sin(theta_radians),np.cos(theta_radians),0])

    # apply rotation matrix to datapoints
    points_rotated = R @ points.T
    
    return points_rotated

def align_plane_vecs(points, pca):
    """
    Align the plane-defining vectors obtained from PCA with two sides of the 
    quadrilateral defined by the datapoints. This correction is necessary for 
    comparing plane angles. This is because PCA returns the vectors 
    corresponding to the directions of maximum variance in the data. If these
    directions differ between two planes being considered, the angle calculated
    for parallel planes might not be 0 degrees, but 180. Putting the plane-
    defining vectors in a common frame of reference circumvents this issue and 
    makes interpretation of the angles more straightforward. For more details 
    see plane_fitting.py file.

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
        points_proj[i,:] = vops.getProjection(points[i,:]-com,pca.components_)
    
    
    # get the vectors corresponding to the sides of the parallelogram
    # these will be the new plane-defining bases
    a = vops.getVecFromPoints(points_proj[0,:],points_proj[1,:])
    b = vops.getVecFromPoints(points_proj[0,:],points_proj[-1,:])
    
    # normalise them
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    
    #change basis to the one defined by the plane + its normal
    # this is so that the vector(s) can be rotated around the normal easily
    plane_basis = np.concatenate((pca.components_,
                                  np.cross(pca.components_[0,:],pca.components_[1:,])))
    
    a_newBasis = plane_basis @ a
    b_newBasis = plane_basis @ b
    
    
    # force the plane-defining vectors to be orthogonal
    if np.abs(np.dot(a,b)) > 0.001:
        # if they're not already - rotate vector b 
        angle = np.arccos(np.dot(a,b))
        
        # a bit of a brute force approach here - try a positive angle first
        angle_diff = np.pi/2 - angle
        tmp = rotate_plane_by_angle(b_newBasis,angle_diff)
        
        # check vectors orthogonal
        if np.abs(np.dot(a_newBasis,tmp)) > 0.001:
            # if not - try rotating by the same angle but in the opposite direction
            tmp = rotate_plane_by_angle(b_newBasis,-angle_diff)
        
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


def fit_mds_to_rdm(rdm):
    mds = MDS(n_components=3, 
              metric=True, 
              dissimilarity='precomputed', 
              max_iter=1000,
              random_state=0)
    return mds.fit_transform(rdm)
#%% define plotting functions

def plot_geometry(ax,points,plot_colours,plot_outline = True,legend_on=True):
    ms = 75
    n_colours = len(plot_colours)
    # plot the parallelogram defined by colours at location 1
    if plot_outline:
        ax.plot(np.append(points[:n_colours,0],points[0,0]),
              np.append(points[:n_colours,1],points[0,1]),
              np.append(points[:n_colours,2],points[0,2]),'k-')
    ax.scatter(points[0,0],points[0,1], points[0,2],marker='^',s = ms,
              c='k',label='loc1')
    ax.scatter(points[:n_colours,0],points[:n_colours,1],
              points[:n_colours,2],marker='^',s = ms,c=plot_colours)
  
    # repeat for loc 2
    if plot_outline:
        ax.plot(np.append(points[n_colours:,0],points[n_colours,0]),
              np.append(points[n_colours:,1],points[n_colours,1]),
              np.append(points[n_colours:,2],points[n_colours,2]),'k-')
    ax.scatter(points[-1,0],points[-1,1], points[-1,2],marker='s',s = ms,
              c='k',label='loc2')
    ax.scatter(points[n_colours:,0],points[n_colours:,1],
              points[n_colours:,2],marker='s',s = ms,c=plot_colours)
    if legend_on:
        ax.legend()


def plot_plane(ax,verts,fc='k',a=0.2):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # plot a polygon with given vertices in 3D
    ax.add_collection3d(Poly3DCollection([verts],facecolor=fc,edgecolor=[],alpha=a))
    
    
def plot_subspace(ax,points,plane_vecs,fc='k',a=0.2):
    # plot the best-fitting plane as a quadrilateral with vertices being the projections of original points onto the plane 
    
    if (points.shape[1]!=3):
        raise NotImplementedError('Check shape of data matrix - should be [n_points,3]')
    
    # find vertices
    n_points = points.shape[0]
    verts = np.zeros((n_points,3))
    
    com = np.mean(points, axis=0) # centre of mass
    
    for i in range(n_points):
        verts[i,:] = vops.getProjection(points[i,:]-com,plane_vecs) # get projection of demeaned 3d points
        verts[i,:] += com #add the mean back
    
    # sort vertices according to shortest path - so that plotted plane will be a quadrilateral
    sorted_verts, sorting_order = vops.sortByPathLength(verts)
    # plot the best-fit plane
    plot_plane(ax,sorted_verts,fc,a)
    #return verts, sorted_verts

#%% pipeline

def get_model_RDMs(constants):
    path = constants.PARAMS['FULL_PATH']
    
    colours = constants.PLOT_PARAMS['4_colours']
    n_colours=len(colours)

    
    pre_up = np.array([[-1,-1,1,1],[1,-1,-1,1],[0,0,0,0]])
    pre_down =  np.array([[0,0,0,0],[-1,1,1,-1],[1,1,-1,-1]])
    
    pre = np.concatenate((pre_up,pre_down),axis=1)
    
    # post - both rotate
    
    post_up = np.array([[1,1,-1,-1],[1,-1,-1,1],[0.5,0.5,0.5,0.5]])
    post_down = np.array([[1,1,-1,-1],[1,-1,-1,1],[-0.5,-0.5,-0.5,-0.5]])
    
    post = np.concatenate((post_up,post_down),axis=1)
    
    
    # no post-cue plane distance RDM
    post_singlePlane = np.concatenate([post[:,:n_colours].T]*2) # coordinates
    
    #%% plot ortho and parallel
    
    plt.figure(figsize=(12,5))
    
    ax = plt.subplot(121, projection='3d')
    plot_geometry(ax,pre.T,colours,legend_on=False)
    ax.set_title('pre-cue')
    ax.scatter(0,0,0,marker='+',c='k')
    
    
    ax2 = plt.subplot(122, projection='3d')
    plot_geometry(ax2,post.T,colours)
    ax.set_title('pre-cue')
    ax2.set_title('post-cue')
    plt.legend(bbox_to_anchor=(1, 1),
                bbox_transform=plt.gcf().transFigure)
    
    ax2.scatter(0,0,0,marker='+',c='k')
    
    fig_path = path+'RSA'
    check_path(fig_path)
    
    plt.savefig(fig_path+'/geom_predictions_from_paper')

    #%% get model RDMs
    
    orthoPlanes_RDM = squareform(pdist(pre.T,))#metric='correlation'))
    parallelPlanes_RDM = squareform(pdist(post.T))#,metric='correlation'))
    singlePlane_RDM = squareform(pdist(post_singlePlane))#,metric='correlation'))
    
    
    model_RDMs = np.stack((orthoPlanes_RDM,parallelPlanes_RDM,singlePlane_RDM),axis=2)
    # model_RDMs /= np.max(model_RDMs) # normalise values to be within 0 and 1
    
        
    #%% plot model RDMs
    
    fig, axes = plt.subplots(1,3, sharex=True, sharey = True, figsize=(10,50))
    
    
    titles = ['ortho','parallel','single']
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
    
    
    #%% save model RDMs
    
    # get upper diagonal / square form
    model_RDMs_sqform = np.zeros((squareform(model_RDMs[:,:,0]).shape[0],model_RDMs.shape[-1]))
    for model in range(model_RDMs.shape[-1]):
        model_RDMs_sqform[:,model] = squareform(model_RDMs[:,:,model])
    
    
    if not (os.path.exists(path+'RSA')):
                os.mkdir(path+'RSA')
    
    save_path = path + 'RSA/'
    
    
    pickle.dump(model_RDMs,open(save_path + 'model_RDMs.pckl','wb'))
    pickle.dump(model_RDMs_sqform,open(save_path + 'model_RDMs_sqform.pckl','wb'))
    
    model_RDMs_order = ['orthogonal','parallel','single']
    
    pickle.dump(model_RDMs_order,open(save_path + 'model_RDMs_order.pckl','wb'))
    
    
    #%% ROTATIONS
    
    # set up up the data folders
    save_path = path + 'RSA/'
    
    if not (os.path.exists(save_path+'rotated_coords/')):
        os.mkdir(save_path+'rotated_coords/')
    if not (os.path.exists(save_path+'rotated_fullRDMs/')):
        os.mkdir(save_path+'rotated_fullRDMs/')
    if not (os.path.exists(save_path+'rotated_diagRDMs/')):
        os.mkdir(save_path+'rotated_diagRDMs/')
        
    if not (os.path.exists(save_path+'ortho_rotated_coords/')):
        os.mkdir(save_path+'ortho_rotated_coords/')
    if not (os.path.exists(save_path+'ortho_rotated_fullRDMs/')):
        os.mkdir(save_path+'ortho_rotated_fullRDMs/')
    if not (os.path.exists(save_path+'ortho_rotated_diagRDMs/')):
        os.mkdir(save_path+'ortho_rotated_diagRDMs/')
    
    
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
    
    
    
    
    #%% plot parallel variations - rotation by 90 degrees, mirror image and ortho 45
    plt.rcParams.update({'font.size': 25})
    
    plt.figure(figsize=(18,5))
    ax = plt.subplot(131,projection='3d')
    
    theta_degrees = 90
    new_coords = pickle.load(open(save_path 
                                    + 'rotated_coords/rotatedBy'
                                    + str(theta_degrees)+'.pckl','rb'))
    
    
    plot_geometry(ax,new_coords.T,colours,legend_on=False)
    ax.set_title('$parallel_{90}$')
    
    ax.set_xticks([-1,0,1])
    ax.set_yticks([-1,0,1])
    ax.set_zticks([-.5,0,.5])
    
    ax.set_xlabel('dim 1',labelpad=20.0)
    ax.set_ylabel('dim 2',labelpad=20.0)
    ax.set_zlabel('dim 3',labelpad=30.0)
    
    ax.tick_params(axis='z',which='major', pad=10)
    
    
    
    R = np.array([[0,1,0],[1,0,0],[0,0,1]]) # flip matrix
    post_flipped = post.copy()
    post_flipped = post_flipped.T
    
    ax2 = plt.subplot(132,projection='3d')
    plot_geometry(ax2,post_flipped,colours,legend_on=False)
    ax2.set_title('mirror')
    
    ax2.set_xlabel('dim 1',labelpad=25.0)
    ax2.set_ylabel('dim 2',labelpad=25.0)
    ax2.set_zlabel('dim 3',labelpad=30.0)
    
    
    ax2.set_xticks([-1,0,1])
    ax2.set_yticks([-1,0,1])
    ax2.set_zticks([-.5,0,.5])
    
    ax2.tick_params(which='major', pad=5)
    ax2.tick_params(axis='z', pad=10)
    
    
    
    
    ax3 = plt.subplot(133,projection='3d')
    theta_degrees =45
    new_coords = pickle.load(open(save_path 
                                    + 'ortho_rotated_coords/rotatedBy'
                                    + str(theta_degrees)+'.pckl','rb'))
    plot_geometry(ax3,new_coords.T,colours)
    
    
    ax3.set_title('$ortho_{45}$')
    ax3.set_xlabel('dim 1',labelpad=20.0)
    ax3.set_ylabel('dim 2',labelpad=20.0)
    ax3.set_zlabel('dim 3',labelpad=20.0)
    
    
    # plt.legend(bbox_to_anchor=(.8,.8))
    plt.legend(bbox_to_anchor=(1.5,1.2))
    
    plt.tight_layout()
    
    
    
    #%%   MIRROR IMAGES
    
    
    R = np.array([[0,1,0],[1,0,0],[0,0,1]]) # flip matrix
    post_flipped = post.copy()
    post_flipped = post_flipped.T
    post_flipped[n_colours:,:] = post_flipped[n_colours:,:]@R
    
    plt.figure()
    ax = plt.subplot(111, projection='3d')
    plot_geometry(ax,post_flipped,colours,legend_on=False)
    # ax.set_title('mirror')
    
    
    ax.scatter(0,0,0,marker='+',c='k')
    
    plane1 = get_best_fit_plane(post_flipped[0:n_colours])
    plane2 = get_best_fit_plane(post_flipped[n_colours:]) 
    
    angle = get_angle_between_planes(plane1.components_,plane2.components_)
    print('Angle: %.2f' %angle)
    
    
    ax.set_title('flipped ['+str(np.round(angle))+'°]')
    plt.savefig(fig_path+'/geom_flipped')
    
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
    
    
    #%% post-cue phase-aligned
        
    plt.figure()
    ax = plt.subplot(111, projection='3d')
    plot_geometry(ax,post.T,colours,legend_on = False)
    
    ax.scatter(0,0,0,marker='+',c='k')
    
    plane1 = get_best_fit_plane(post[:,0:n_colours].T)
    plane2 = get_best_fit_plane(post[:,n_colours:].T) 
    
    angle = get_angle_between_planes(plane1.components_,plane2.components_)
    
    ax.set_title('phase-aligned ['+str(np.round(angle))+'°]')
    
    plt.savefig(fig_path+'/geom_phaseAligned')
    
    
    print('Angle: %.2f' %angle)


