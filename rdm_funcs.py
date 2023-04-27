#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 12:54:19 2021

@author: emilia
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform, pdist
import pickle
import os.path
import vec_operations as vops 
from rep_geom import *
import constants
from helpers import check_path

#%%
def pick_point_coords():
    colours = ['r','y','g','b']
    n_colours=len(colours)
    
    # pre-cue delay : orthogonal planes
    pre_up = np.array([[-1,-1,1,1],[1,-1,-1,1],[0,0,0,0]])
    pre_down =  np.array([[0,0,0,0],[-1,1,1,-1],[1,1,-1,-1]])
    
    pre = np.concatenate((pre_up,pre_down),axis=1)
    
    # post-cue delay - both planes rotate into a parallel configuration
    
    post_up = np.array([[1,1,-1,-1],[1,-1,-1,1],[0.5,0.5,0.5,0.5]])
    post_down = np.array([[1,1,-1,-1],[1,-1,-1,1],[-0.5,-0.5,-0.5,-0.5]])
    
    post = np.concatenate((post_up,post_down),axis=1)
    
    return colours, n_colours, pre, post

def make_model_rdms(path):
    #%% pick point coordinates
    
    colours, n_colours, pre, post = pick_point_coords()
    
    
    # post-cue delay - single plane RDM
    post_singlePlane = np.concatenate([post[:,:n_colours].T]*2) # coordinates
    
    #%% construct RDMs

    orthoPlanes_RDM = squareform(pdist(pre.T,))#metric='correlation'))
    parallelPlanes_RDM = squareform(pdist(post.T))#,metric='correlation'))
    singlePlane_RDM = squareform(pdist(post_singlePlane))#,metric='correlation'))
    
    
    model_RDMs = np.stack((orthoPlanes_RDM,parallelPlanes_RDM,singlePlane_RDM),axis=2)
    
    #%% plot RDMs

    fig, axes = plt.subplots(1,3, sharex=True, sharey = True, figsize=(10,50))
    
    
    titles = ['ortho','parallel','single']
    for ix,ax in enumerate(axes.flat):
        im = ax.imshow(model_RDMs[:,:,ix],vmin=np.min(model_RDMs[:,:,ix]),vmax=np.max(model_RDMs[:,:,ix]))
        ax.set_title(titles[ix])
        ax.set_xticks([])
        ax.set_yticks([])
    
    fig.colorbar(im,ax=axes.ravel().tolist())
    
    #%% save model RDMs

    # get upper diagonal / square form
    model_RDMs_sqform = np.zeros((squareform(model_RDMs[:,:,0]).shape[0],model_RDMs.shape[-1]))
    for model in range(model_RDMs.shape[-1]):
        model_RDMs_sqform[:,model] = squareform(model_RDMs[:,:,model])
    
    
    save_path = path + 'RSA/'
    check_path(save_path)
    
    
    pickle.dump(model_RDMs,open(save_path + 'model_RDMs.pckl','wb'))
    pickle.dump(model_RDMs_sqform,open(save_path + 'model_RDMs_sqform.pckl','wb'))
    
    model_RDMs_order = ['orthogonal','parallel','single']
    
    pickle.dump(model_RDMs_order,open(save_path + 'model_RDMs_order.pckl','wb'))


def rotatePlaneByAngle(points,theta_degrees):
    theta = np.radians(theta_degrees)

    # want to rotate around the y axis

    # construct rotation matrix
    R = np.eye(3)
    # z component stays the same
    R[:,0] = np.array([np.cos(theta),np.sin(theta),0])
    R[:,1] = np.array([-np.sin(theta),np.cos(theta),0])

    # apply rotation to points

    points_rotated = R @ points.T
    
    return points_rotated   
 
def make_rotated_rdms(path):
    # get base point coordinates which are to be rotated
    colours, n_colours, pre, post = pick_point_coords()
    
    
    save_path = path + 'RSA/'
    check_path(save_path)
    
    # create paths
    check_path(save_path+'rotated_coords/')
    check_path(save_path+'rotated_fullRDMs/')
    check_path(save_path+'rotated_diagRDMs/')
    check_path(save_path+'ortho_rotated_coords/')
    check_path(save_path+'ortho_rotated_fullRDMs/')
    check_path(save_path+'ortho_rotated_diagRDMs/')
    
    theta_range = np.arange(0,365,5)

    for i,theta_degrees in enumerate(theta_range):
        points = post.T[n_colours:,:]
        points_r = rotatePlaneByAngle(points,theta_degrees)
        
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
        
        ortho_points = pre.T[:n_colours,:]
        ortho_points_r = rotatePlaneByAngle(ortho_points,theta_degrees)
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

    
