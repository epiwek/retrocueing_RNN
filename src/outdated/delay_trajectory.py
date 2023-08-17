#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 12:50:49 2021

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
from rep_geom import *
import helpers


#%% define plotting functions

def plot_geometry(ax,Z,pca,plot_colours,plot_outline = True,legend_on=False, **kwargs):
    
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

#model_number = input('Specify model number: ')
#model_type = input('Specify model type (LSTM or RNN): ')

# model_number = '0'
# models = np.array([ 0,  1,  2,  3,  5,  6,  8, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20,
#        21, 23, 24, 25, 26, 27, 28, 29])
# models = np.array([2, 3, 4, 5, 6, 7, 8, 9])
# models = np.array(range(10))

model_type = 'RNN'
    
if (model_type == 'RNN'):
    # load_path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/data/pca_data_sameTrainingData'
    #load_path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/data/pca_data_sameTrainingInit'
    # load_path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/data_1hot/pca_data'
    # load_path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/data_1hot/forAndrew/Xavier_200rec/lr0.005/pca_data'
    # load_path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/data_gaussian/with_fixation/nrec300/lr0.005/pca_data'
    # load_path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/data_gaussian_hardcoded/nrec300/lr0.005/pca_data'
    # load_path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/data_vonMises/kappa100.0/nrec200/lr0.005/pca_data'
    # load_path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/data_1hot_new/nrec200/lr0.005/pca_data'
    
    base_path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/'+\
                            'data_vonMises/MSELoss/with_fixation_longTrials/kappa1.0/nrec200/lr0.001/'



elif (model_type == 'LSTM'):
    load_path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/LSTM_data/pca_data'
    #load_path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/LSTM_data_1hot/pca_data'
else :
    print('Invalid model type')

load_path = base_path + '/pca_data'
f = open(load_path+'/converged.pckl','rb')
converged = pickle.load(f)
f.close()


load_path = base_path + '/saved_models/'
# load task params    
f = open(load_path+'task_params.pckl', 'rb')
obj = pickle.load(f)
[n_stim,trial_timings] = obj
f.close()

plot_colours = ['r','y','g','b']
n_colours = len(plot_colours)

#%% plotting

def plot_trajectory(ax,precue_trajectory,plot_colours):
    n_colours = len(plot_colours)
    time = precue_trajectory.shape[0]
    
    for c in range(n_colours):
        ax.plot(precue_trajectory[:,c,0],precue_trajectory[:,c,1],
                    np.arange(time),'o-',
                    c = plot_colours[c])
    
    
    t_0 = np.concatenate((precue_trajectory1[0,:,:],
                                    np.zeros((n_colours,1))),1)
    t_end = np.concatenate((precue_trajectory1[-1,:,:],
                                    np.ones((n_colours,1))*
                                    (time-1)),1)
    
    # plot shaded rectangles at t0 and t_end
    plot_geometry(ax,t_0,[],plot_colours,alpha=.6)
    plot_geometry(ax,t_end,[],plot_colours,alpha=.6)
    plot_plane(ax,stimulus_onset)
    plot_plane(ax,end_precue_delay)
    
#%%
plt.rcParams.update({'font.size': 15})

load_path = base_path + '/pca_data'

for i,model_number in enumerate(converged):
# for model_number in np.arange(1):
    model_number = str(model_number)
    print('Model: '+model_number)
    # load pca data
    f = open(load_path+'/pca_data_model' + model_number + '.pckl', 'rb')
    obj = pickle.load(f)    
    [trial_data,delay1,delay2] = obj
    print('change eval to squeeze trial_data')
    trial_data = trial_data.squeeze().permute([1,2,0])
    f.close()
    
    
    precue_end_loc1_pca = PCA(n_components=2)
    precue_end_loc1_pca.fit(delay1[:n_colours,:] - torch.mean(delay1[:n_colours,:]))
    
    precue_end_loc2_pca = PCA(n_components=2)
    precue_end_loc2_pca.fit(delay1[n_colours:,:] - torch.mean(delay1[n_colours:,:]))
    
    time = trial_timings['delay1_dur']+1
    precue_trajectory1 = []
    precue_trajectory2 = []
    for t in range(trial_timings['delay1_dur']+1):
        precue_trajectory1.append(precue_end_loc1_pca.transform(
            trial_data[:n_colours,t,:]-torch.mean(trial_data[:n_colours,t,:])))
        precue_trajectory2.append(precue_end_loc2_pca.transform(
            trial_data[n_colours:,t,:]-torch.mean(trial_data[n_colours:,t,:])))
    
    precue_trajectory1 = np.array(precue_trajectory1)
    precue_trajectory2 = np.array(precue_trajectory2)
    plt.figure()
    ax = plt.subplot(111,projection='3d')
    # plot
    # for c in range(n_colours):
    #     ax.plot(precue_trajectory1[:,c,0],precue_trajectory1[:,c,1],
    #                 np.arange(time),'o-',
    #                 c = plot_colours[c])
    
    # end_precue_delay = np.concatenate((precue_trajectory1[-1,:,:],
    #                                 np.ones((n_colours,1))*
    #                                 (trial_timings['delay1_dur'])),1)
    # stimulus_onset = np.concatenate((precue_trajectory1[0,:,:],
    #                                 np.zeros((n_colours,1))),1)
    # plot_geometry(ax,stimulus_onset,[],plot_colours,alpha=.6)
    # plot_geometry(ax,end_precue_delay,[],plot_colours,alpha=.6)
    
    # plot_plane(ax,stimulus_onset)
    # plot_plane(ax,end_precue_delay)
    
    plot_trajectory(ax,precue_trajectory,plot_colours)
    
    ax.set_zticks(np.arange(-1,6))
    # ax.set_zticklabels(np.arange(-1,6))
    
    
    loc1_PC1_PVE = np.round(precue_end_loc1_pca.explained_variance_ratio_[0]*100,2)
    loc1_PC2_PVE = np.round(precue_end_loc1_pca.explained_variance_ratio_[1]*100,2)
    ax.set_xlabel('PC1 [' +str(loc1_PC1_PVE)+'%]',labelpad=20)
    ax.set_ylabel('PC2 [' +str(loc1_PC2_PVE)+'%]',labelpad=20)
    ax.set_zlabel('Pre-cue delay onset [t]',labelpad=20)
    ax.set_title('Model '+model_number+' : '+'loc1 precue colour subspace')
    
    
    # can also plot the same for the other location
    # plt.figure()
    # ax = plt.subplot(111,projection='3d')
    
    # for c in range(n_colours):
    #     ax.plot(precue_trajectory2[:,c,0],
    #             precue_trajectory2[:,c,1],
    #                np.arange(time),'o-',
    #                c = plot_colours[c])
    
    # end_precue_delay = np.concatenate((precue_trajectory2[-1,:,:],
    #                                 np.ones((n_colours,1))*
    #                                 (trial_timings['delay1_dur'])),1)
    # stimulus_onset = np.concatenate((precue_trajectory2[0,:,:],
    #                                 np.zeros((n_colours,1))),1)
    # plot_geometry(ax,stimulus_onset,[],plot_colours,alpha=.6)
    # plot_geometry(ax,end_precue_delay,[],plot_colours,alpha=.6)
    
    # plot_plane(ax,stimulus_onset)
    # plot_plane(ax,end_precue_delay)
    
    # ax.set_zticks(np.arange(-1,6))
    # # ax.set_zticklabels(np.arange(-1,6))
    
    
    # loc2_PC1_PVE = np.round(precue_end_loc2_pca.explained_variance_ratio_[0]*100,2)
    # loc2_PC2_PVE = np.round(precue_end_loc2_pca.explained_variance_ratio_[1]*100,2)
    # ax.set_xlabel('PC1 [' +str(loc2_PC1_PVE)+'%]',labelpad=20)
    # ax.set_ylabel('PC2 [' +str(loc2_PC2_PVE)+'%]',labelpad=20)
    # ax.set_zlabel('Pre-cue delay onset [t]',labelpad=20)
    # ax.set_title('Model '+model_number+' : '+'loc2 precue colour subspace')
    