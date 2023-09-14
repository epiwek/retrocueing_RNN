#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 19:05:48 2020

@author: emilia
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import vec_operations as vops
import pickle
import os.path
from rep_geom import get_best_fit_plane, get_angle_between_planes, align_plane_vecs
import helpers
import constants


#%% define plotting functions

def plot_geometry(ax,Z,pca,plot_colours,plot_outline = True,legend_on = True, **kwargs):
    
    n_colours = len(plot_colours)
    # plot the parallelogram defined by colours at location 1
    if plot_outline:
        ax.plot(np.append(Z[:n_colours,0],Z[0,0]),
              np.append(Z[:n_colours,1],Z[0,1]),
              np.append(Z[:n_colours,2],Z[0,2]),'k-',**kwargs)
    ax.scatter(Z[0,0],Z[0,1], Z[0,2],marker='o',s = 40,
              c='k',label='loc1',**kwargs)
    ax.scatter(Z[:n_colours,0],Z[:n_colours,1],
              Z[:n_colours,2],marker='o',s = 40,c=plot_colours,**kwargs)
  
    # repeat for loc 2
    if Z.shape[0]>n_colours:
        if plot_outline:
            ax.plot(np.append(Z[n_colours:,0],Z[n_colours,0]),
                  np.append(Z[n_colours:,1],Z[n_colours,1]),
                  np.append(Z[n_colours:,2],Z[n_colours,2]),'k-',**kwargs)
        ax.scatter(Z[-1,0],Z[-1,1], Z[-1,2],marker='s',s = 40,
                  c='k',label='loc2',**kwargs)
        ax.scatter(Z[n_colours:,0],Z[n_colours:,1],
                  Z[n_colours:,2],marker='s',s = 40,c=plot_colours,**kwargs)
        
    if pca:
        ax.set_xlabel('PC1 ['+str(np.round(pca.explained_variance_ratio_[0]*100,1))+'%]')
        ax.set_ylabel('PC2 ['+str(np.round(pca.explained_variance_ratio_[1]*100,1))+'%]')
        ax.set_zlabel('PC3 ['+str(np.round(pca.explained_variance_ratio_[2]*100,1))+'%]')
    
    if legend_on:
        ax.legend()

def plot_plane(ax,verts,fc='k',a=0.2):
    # plot a polygon with given vertices in 3D
    ax.add_collection3d(Poly3DCollection([verts],facecolor=fc,edgecolor=[],alpha=a))
    
def plot_subspace(ax,points,plane_vecs,fc='k',a=0.2):
    # plot the best-fitting plane as a quadrilateral with vertices being the projections of original points onto the plane 
    
    if (points.shape[1]!=3):
        raise NotImplementedError('Check the shape of the data matrix - should be (n_points,3)')
    
    # find vertices
    n_points = points.shape[0]
    verts = np.zeros((n_points,3))
    
    com = np.mean(points, axis=0) # centre of mass
    
    for i in range(n_points):
        verts[i,:] = vops.getProjection(points[i,:]-com,plane_vecs) # get projection of demeaned 3d points
        verts[i,:] += com #add the mean back
    
    # sort vertices according to shortest path - so that plotted plane will be a quadrilateral
    sorted_verts, sorting_order = vops.defPlaneShape(verts,plane_vecs)
    #sorted_verts, sorting_order = vops.sortByVecAngle(verts)
    #sorted_verts = verts
    # plot the best-fit plane
    plot_plane(ax,sorted_verts,fc,a)
    #return verts, sorted_verts


def plot_plane_old(ax,Y_l,points,scale=1.0,fc='k',a=0.2):
    # plot the best fitting plane as a quadrilateral, with vertices being some
    #scaled distance from the centre-of-mass of the original point set
    # (plots look better if e.g. one set of points forms a much denser cloud than the other)

    com = np.mean(points,axis=0) # centre of mass
    #sorted_verts, sorting_order = vops.sortByPathLength(points)
    # set the scale to be equal to largest distance between points
    #scale = np.linalg.norm(sorted_verts[0]-sorted_verts[2])
    Y_l.components_ *= scale
    # plot the best-fit plane
    verts = np.array([com-Y_l.components_[0,:],com-Y_l.components_[1,:],
                    com+Y_l.components_[0,:],com+Y_l.components_[1,:]])
    #verts *= scale
    ax.add_collection3d(Poly3DCollection([verts],facecolor=fc,edgecolor=[],alpha=a))
    
#%% load model from file

load_path = constants.PARAMS['FULL_PATH']+'pca_data/valid_trials/'
plot_data = False

for model_number in range(constants.PARAMS['n_models']):
    
    model_number = str(model_number)
    print('Model: '+model_number)
    # load pca data
    f = open(load_path+'/pca_data_model' + model_number + '.pckl', 'rb')
    obj = pickle.load(f)    
    # [trial_data,delay1,delay2] = obj
    # trial_data = trial_data.squeeze().permute([1,2,0])
    pca_data = obj
    f.close()
    
    trial_data = pca_data['data']
    delay1 = pca_data['delay1']
    delay2 = pca_data['delay2']
    
    all_timepoints_pca = {}
    all_timepoints_3dcoords = {}
    for t in range(trial_data.shape[1]):
        all_timepoints_pca[str(t)] =  PCA(n_components=3)
        trial_data[:,:,t]
        all_timepoints_3dcoords[str(t)] = \
            all_timepoints_pca[str(t)].fit_transform(trial_data[:,:,t] - torch.mean(trial_data[:,:,t]))
    
        
    #%% run PCA
    delay1_pca = PCA(n_components=3) # Initializes PCA
    delay2_pca = PCA(n_components=3) # Initializes PCA
    
    SI_pca = PCA(n_components=3)
    SI_data = trial_data[:,0,:]
    
    # demean data
    delay1 -= torch.mean(delay1)
    delay2 -= torch.mean(delay2)
    SI_data -= torch.mean(SI_data)
    
    # run PCA
    delay1_3dcoords = delay1_pca.fit_transform(delay1) # get coordinates in the reduced-dim space
    delay2_3dcoords = delay2_pca.fit_transform(delay2)
    SI_3dcoords = SI_pca.fit_transform(SI_data)
    
    
    # %% plotdata in reduced dim space
    
    plot_colours = ['b','g','y','r']
    n_colours = len(plot_colours)
    
    if plot_data:
        plt.figure(figsize=(12,5))
        
        ax = plt.subplot(121, projection='3d')
        plot_geometry(ax, delay1_3dcoords, delay1_pca, plot_colours)
        # plt.title('pre-cue')
        
        
        ax2 = plt.subplot(122, projection='3d')
        plot_geometry(ax2, delay2_3dcoords, delay2_pca, plot_colours)
        # plt.title('post-cue')
        
        
        equal_axes = True
        
        if equal_axes:
            # equal x, y and z axis scale
            ax_lims = np.array(ax.xy_viewLim)
            ax.set_xlim3d(np.min(ax_lims),np.max(ax_lims))
            ax.set_ylim3d(np.min(ax_lims),np.max(ax_lims))
            ax.set_zlim3d(np.min(ax_lims),np.max(ax_lims))
            
            ax2_lims = np.array(ax2.xy_viewLim)
            ax2.set_xlim3d(np.min(ax2_lims),np.max(ax2_lims))
            ax2.set_ylim3d(np.min(ax2_lims),np.max(ax2_lims))
            ax2.set_zlim3d(np.min(ax2_lims),np.max(ax2_lims))
        
        
        equal_axes = True
        
        if equal_axes:
            # equal x, y and z axis scale
            ax_lims = np.array(ax.xy_viewLim)
            ax.set_xlim3d(np.min(ax_lims),np.max(ax_lims))
            ax.set_ylim3d(np.min(ax_lims),np.max(ax_lims))
            ax.set_zlim3d(np.min(ax_lims),np.max(ax_lims))
            
            ax2_lims = np.array(ax2.xy_viewLim)
            ax2.set_xlim3d(np.min(ax2_lims),np.max(ax2_lims))
            ax2.set_ylim3d(np.min(ax2_lims),np.max(ax2_lims))
            ax2.set_zlim3d(np.min(ax2_lims),np.max(ax2_lims))
    
    #% add planes of best fit
    
    #pre-cue
    # delay1_up_pca = PCA(n_components=2) # do PCA to get best-fit plane for delay 1 up (loc 1)
    # delay1_down_pca = PCA(n_components=2)
    
    # delay1_planeUp = delay1_up_pca.fit(delay1_3dcoords[0:n_colours,:]-np.mean(delay1_3dcoords[0:n_colours,:])) # get directions of max variance, i.e. vectors defining the plane
    # delay1_planeDown = delay1_down_pca.fit(delay1_3dcoords[n_colours:,:] - np.mean(delay1_3dcoords[n_colours:,:]))
    
    
    delay1_planeUp = get_best_fit_plane(delay1_3dcoords[0:n_colours,:])
    delay1_planeDown = get_best_fit_plane(delay1_3dcoords[n_colours:,:])
    
    # delay1_planeUp.components_ += -np.mean(delay1_3dcoords[0:n_colours,:])
    # delay1_planeDown.components_ += - np.mean(delay1_3dcoords[n_colours:,:])
    # delay1_planeUp = get_best_fit_plane(delay1_3dcoords[0:n_colours,:])
    # delay1_planeDown = get_best_fit_plane(delay1_3dcoords[n_colours:,:])
    
    # delay1_planeUp.components_ /= np.linalg.norm(delay1_planeUp.components_)
    # delay1_planeDown.components_ /= np.linalg.norm(delay1_planeDown.components_)
    
    # calculate angle between planes
    # cos_theta_pre = np.dot(np.cross(delay1_planeUp.components_[0,:],delay1_planeUp.components_[1,:]),
    #                     np.cross(delay1_planeDown.components_[0,:],delay1_planeDown.components_[1,:]))
    # print('Angle pre-cue: %.2f' %(np.degrees(np.arccos(cos_theta_pre))))
    # angle_pre = np.degrees(np.arccos(cos_theta_pre))
    
    # align plane vectors with sides of the quadrilateral
    delay1_planeUp_vecs = align_plane_vecs(delay1_3dcoords[0:n_colours,:],delay1_planeUp)
    delay1_planeDown_vecs = align_plane_vecs(delay1_3dcoords[n_colours:,:],delay1_planeDown)
    
    # calculate angle between planes
    theta_pre = get_angle_between_planes(delay1_planeUp_vecs, delay1_planeDown_vecs)
    
    print('Angle pre-cue: %.2f' %(theta_pre))
    
    if plot_data:
        ax.set_title('pre-cue ['+str(np.round(theta_pre))+']')
        
        plot_subspace(ax,delay1_3dcoords[:n_colours,:],delay1_planeUp.components_,fc='k',a=0.2)
        plot_subspace(ax,delay1_3dcoords[n_colours:,:],delay1_planeDown.components_,fc='k',a=0.2)

    
    
    
    #post-cue
    # delay2_up_pca = PCA(n_components=2)
    # delay2_down_pca = PCA(n_components=2)
    
    delay2_planeUp = get_best_fit_plane(delay2_3dcoords[:n_colours,:])
    delay2_planeDown = get_best_fit_plane(delay2_3dcoords[n_colours:,:])
    
    # align planes
    delay2_planeUp_vecs = align_plane_vecs(delay2_3dcoords[:n_colours,:],delay2_planeUp)
    delay2_planeDown_vecs = align_plane_vecs(delay2_3dcoords[n_colours:,:],delay2_planeDown)

    # calculate angle between planes
    theta_post = get_angle_between_planes(delay2_planeUp_vecs, delay2_planeDown_vecs)
    

    print('Angle post-cue: %.2f' %(theta_post))    
    if plot_data:
        ax2.set_title('post-cue ['+str(np.round(theta_post))+']')
        
        
        plot_subspace(ax2,delay2_3dcoords[:n_colours,:],delay2_planeUp.components_,fc='k',a=0.2)
        plot_subspace(ax2,delay2_3dcoords[n_colours:,:],delay2_planeDown.components_,fc='k',a=0.2)
    
    
    
    #%% SI vs delay1
    
    
    # project SI into delay1 and vice versa
    # SI_in_delay1 = delay1_pca.transform(SI_data)
    # delay1_in_SI = SI_pca.transform(delay1)
    
    
    # plt.figure(figsize=(12,10))
    # ax = plt.subplot(221, projection='3d')
    # plot_geometry(ax, delay1_3dcoords, delay1_pca, plot_colours)
    # ax.set_title('delay1 in delay1')
    # # plt.title('pre-cue')
    
    
    # ax2 = plt.subplot(222, projection='3d')
    # plot_geometry(ax2, SI_in_delay1, delay1_pca, plot_colours)
    # # plt.title('post-cue')
    # ax2.set_title('SI in delay1')
    
    # ax3 = plt.subplot(223, projection='3d')
    # plot_geometry(ax3, SI_3dcoords, SI_pca, plot_colours)
    # ax3.set_title('SI in SI')
    
    # ax4 = plt.subplot(224, projection='3d')
    # plot_geometry(ax4, delay1_in_SI, SI_pca, plot_colours)
    # ax4.set_title('delay1 in SI')
    
    # helpers.equal_axes(ax)
    # helpers.equal_axes(ax2)
    # helpers.equal_axes(ax3)
    # helpers.equal_axes(ax4)
    
    
    # plt.figure(figsize=(12,5))
    # ax = plt.subplot(121, projection='3d')
    # plot_geometry(ax, delay1_3dcoords, delay1_pca, plot_colours)
    # plot_geometry(ax, SI_in_delay1, delay1_pca, plot_colours,alpha=.4)
    # ax.set_title('delay1')
    # # plt.title('pre-cue')
    
    
    # ax2 = plt.subplot(122, projection='3d')
    # plot_geometry(ax2, SI_in_delay1, SI_pca, plot_colours)
    # plot_geometry(ax2, delay1_in_SI, SI_pca, plot_colours,alpha=.4)
    # # plt.title('post-cue')
    # ax2.set_title('SI')
    
    # helpers.equal_axes(ax)
    # helpers.equal_axes(ax2)
    # helpers.equal_axes(ax3)
    # helpers.equal_axes(ax4)
    #%% save fig and data
    
    plt.savefig(load_path+'/SI_vs_delay1_fig_'+model_number)
    
    angles = [theta_pre, theta_post]
    
    if not (os.path.exists(load_path+'/angles')):
            os.mkdir(load_path+'/angles')
    if not (os.path.exists(load_path+'/pca3')):
            os.mkdir(load_path+'/pca3')
    if not (os.path.exists(load_path+'/pca2')):
            os.mkdir(load_path+'/pca2')
    if not (os.path.exists(load_path+'/planes')):
            os.mkdir(load_path+'/planes')
    
    path = load_path + '/angles/angles_' + model_number + '.pckl'
    pickle.dump(angles,open(path,'wb'))
    
    pca3 = [delay1_pca, delay2_pca]
    path = load_path + '/pca3/pca3_' + model_number + '.pckl'
    pickle.dump(pca3,open(path,'wb'))
    
    pca2 = [delay1_planeUp,delay1_planeDown,delay2_planeUp,delay2_planeDown]
    path = load_path + '/pca2/pca2_' + model_number + '.pckl'
    pickle.dump(pca2,open(path,'wb'))
    
    planes = [delay1_planeUp.components_,delay1_planeDown.components_,
              delay2_planeUp.components_,delay2_planeDown.components_]
    path = load_path + '/planes/planes_' + model_number + '.pckl'
    pickle.dump(planes,open(path,'wb'))
    
    coords3D = [delay1_3dcoords , delay2_3dcoords]
    path = load_path + '/pca3/coords3D_' + model_number + '.pckl'
    pickle.dump(coords3D,open(path,'wb'))
#%% fit planes to original (n_dimensional) data

# for i,model_number in enumerate(converged):
# # for model_number in range(10):
#     model_number = str(model_number)
    
#     # load pca data
#     f = open(load_path+'/pca_data_model' + model_number + '.pckl', 'rb')
#     obj = pickle.load(f)
#     f.close()
    
#     [trial_data,delay1,delay2] = obj
#     f.close()
    
#     delay1_up_pca = PCA(n_components=2) # Initializes PCA
#     delay1__down_pca = PCA(n_components=2) # Initializes PCA
#     delay2__up_pca = PCA(n_components=2) # Initializes PCA
#     delay2_down_pca = PCA(n_components=2) # Initializes PCA
    
#     # demean data
#     delay1 -= torch.mean(delay1)
#     delay2 -= torch.mean(delay2)
    
#     # run PCA
#     delay1_3dcoords = delay1_pca.fit_transform(delay1) # get coordinates in the reduced-dim space
#     delay2_3dcoords = delay2_pca.fit_transform(delay2)
    
    