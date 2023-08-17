#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 14:52:32 2021

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
import seaborn as sns
from rep_geom import get_best_fit_plane, get_angle_between_planes
import pycircstat


#%%


# def plot_geometry(ax,Z,pca,plot_colours,plot_outline = True):
    
#     n_colours = len(plot_colours)
#     # plot the parallelogram defined by colours at location 1
#     if plot_outline:
#         ax.plot(np.append(Z[:n_colours,0],Z[0,0]),
#               np.append(Z[:n_colours,1],Z[0,1]),
#               np.append(Z[:n_colours,2],Z[0,2]),'k-')
#     ax.scatter(Z[0,0],Z[0,1], Z[0,2],marker='o',s = 40,
#               c='k',label='loc1')
#     ax.scatter(Z[:n_colours,0],Z[:n_colours,1],
#               Z[:n_colours,2],marker='o',s = 40,c=plot_colours)
  
#     # repeat for loc 2
#     if plot_outline:
#         ax.plot(np.append(Z[n_colours:,0],Z[n_colours,0]),
#               np.append(Z[n_colours:,1],Z[n_colours,1]),
#               np.append(Z[n_colours:,2],Z[n_colours,2]),'k-')
#     ax.scatter(Z[-1,0],Z[-1,1], Z[-1,2],marker='s',s = 40,
#               c='k',label='loc2')
#     ax.scatter(Z[n_colours:,0],Z[n_colours:,1],
#               Z[n_colours:,2],marker='s',s = 40,c=plot_colours)

#     ax.set_xlabel('PC1 ['+str(np.round(pca.explained_variance_ratio_[0]*100,1))+'%]')
#     ax.set_ylabel('PC2 ['+str(np.round(pca.explained_variance_ratio_[1]*100,1))+'%]')
#     ax.set_zlabel('PC3 ['+str(np.round(pca.explained_variance_ratio_[2]*100,1))+'%]')

#     ax.legend()

def plot_geometry(ax,points,plot_colours,plot_outline = True,legend_on=True):
    ms = 150
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
#%% run PCA on uncued data
import helpers 
from analysis import make_rdm, fit_mds_to_rdm

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
    
    load_path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/'+\
                            'data_vonMises/MSELoss/with_fixation_longTrials/kappa1.0/nrec200/lr0.001/pca_data'



elif (model_type == 'LSTM'):
    load_path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/LSTM_data/pca_data'
    #load_path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/LSTM_data_1hot/pca_data'
else :
    print('Invalid model type')

f = open(load_path+'/converged.pckl','rb')
converged = pickle.load(f)
f.close()
n_colours=4
n_models = len(converged)


delay1_ix = 5
delay2_ix = 11

angle_pre = np.empty((n_models,))
angle_post = np.empty((n_models,))

delay1_binned_3dcoords = np.empty((n_colours*2,3,n_models))
delay2_binned_3dcoords = np.empty((n_colours*2,3,n_models))


indiv_model_RDMs_pre = np.empty((n_colours*2,n_colours*2,n_models))
indiv_model_RDMs_post = np.empty((n_colours*2,n_colours*2,n_models))

for i,model_number in enumerate(converged):
# for model_number in np.arange(1):
    model_number = str(model_number)
    
    # load pca data
    f = open(load_path+'/eval_data_model' + model_number + '.pckl', 'rb')
    obj = pickle.load(f)    
    data = obj["data"]
    f.close()
    
    n_trials = data.shape[0]
    n_rec = data.shape[-1]
    labels_uncued = np.concatenate((obj["labels"]["c2"][:n_trials//2],
                             obj["labels"]["c1"][n_trials//2:]))
    
    labels_uncued_binned = helpers.bin_labels(labels_uncued,n_colours)
    
    delay1 = data[:,5,:]
    delay2 = data[:,11,:]
    
    
     # bin and average the data
    
    delay1_binned = np.zeros((n_colours*2,n_rec))
    delay2_binned = np.zeros((n_colours*2,n_rec))
    
    for colour in range(n_colours):
        ix = np.where(labels_uncued_binned==colour)[0]
        ix_down = ix[np.where(ix<n_trials//2)[0]]
        ix_up = ix[np.where(ix>=n_trials//2)[0]]
        
        delay1_binned[colour,:] = torch.mean(delay1[ix_down,:],0)
        delay1_binned[colour+n_colours,:] = torch.mean(delay1[ix_up,:],0)
        
        delay2_binned[colour,:] = torch.mean(delay2[ix_down,:],0)
        delay2_binned[colour+n_colours,:] = torch.mean(delay2[ix_up,:],0)
    
   
    #% calculate RDMs
    indiv_model_RDMs_pre[:,:,i] = make_rdm(delay1_binned)
    indiv_model_RDMs_post[:,:,i] = make_rdm(delay2_binned)
    
    
    
    # run PCA
    delay1_pca = PCA(n_components=3) # Initializes PCA
    delay2_pca = PCA(n_components=3) # Initializes PCA
    
    
    delay1 -= torch.mean(delay1)
    delay2 -= torch.mean(delay2)
    
    
    delay1_3dcoords = delay1_pca.fit_transform(delay1) # get coordinates in the reduced-dim space
    delay2_3dcoords = delay2_pca.fit_transform(delay2)
    
    
    delay1_binned -= np.mean(delay1_binned)
    delay2_binned -= np.mean(delay2_binned)
    
    delay1_binned_pca = PCA(n_components=3) # Initializes PCA
    delay2_binned_pca = PCA(n_components=3) # Initializes PCA
    
    delay1_binned_3dcoords[:,:,i] =  delay1_binned_pca.fit_transform(delay1_binned)
    delay2_binned_3dcoords[:,:,i] =  delay2_binned_pca.fit_transform(delay2_binned)
    
    
    # calculate planes
    
    plane1_pre = get_best_fit_plane(delay1_binned_3dcoords[:n_colours,:,i])
    plane2_pre = get_best_fit_plane(delay1_binned_3dcoords[n_colours:,:,i])
    
    plane1_post = get_best_fit_plane(delay2_binned_3dcoords[:n_colours,:,i])
    plane2_post = get_best_fit_plane(delay2_binned_3dcoords[n_colours:,:,i])
    
    angle_pre[i] = get_angle_between_planes(plane1_pre.components_, 
                                         plane2_pre.components_)
    angle_post[i] = get_angle_between_planes(plane1_post.components_, 
                                         plane2_post.components_)
    
    
   
    
    #% plot

    # plot_colours = ['r','y','g','b']

    # plt.figure()
    # ax = plt.subplot(121,projection='3d')
    
    # plot_geometry(ax, delay1_binned_3dcoords[:,:,i], plot_colours,legend_on=False)
    # plot_subspace(ax,delay1_binned_3dcoords[:n_colours,:,i],plane1_pre.components_)
    # plot_subspace(ax,delay1_binned_3dcoords[n_colours:,:,i],plane2_pre.components_)
    
    # ax.set_xlabel('PC1',labelpad=20)
    # ax.set_ylabel('PC2',labelpad=20)
    # ax.set_zlabel('PC3',labelpad=20)
    # ax.set_title('pre-cue')
    
    
    # ax2 = plt.subplot(122,projection='3d')
    # plot_geometry(ax2, delay2_binned_3dcoords[:,:,i], plot_colours)
    # plot_subspace(ax2,delay2_binned_3dcoords[:n_colours,:,i],plane1_post.components_)
    # plot_subspace(ax2,delay2_binned_3dcoords[n_colours:,:,i],plane2_post.components_)
    
    # # plt.legend(['loc1 uncued','loc2 uncued'])
    
    # ax2.set_xlabel('PC1',labelpad=20)
    # ax2.set_ylabel('PC2',labelpad=20)
    # ax2.set_zlabel('PC3',labelpad=20)
    
    # ax2.set_xticks(np.arange(-0.5,.5,.5))
    # ax2.set_yticks(np.arange(-0.5,.5,.5))
    # ax2.set_zticks(np.arange(-0.5,.5,.5))
    
    
    # ax2.set_title('post-cue')
    

#%% plot example model
# plot_colours = ['r','y','g','b']
# plt.figure(figsize=(12,5))

# ax = plt.subplot(121, projection='3d')
# plot_geometry(ax, delay1_binned_3dcoords[:,:,0], plot_colours)
# ax2 = plt.subplot(122, projection='3d')
# plot_geometry(ax2, delay2_binned_3dcoords[:,:,0], plot_colours)



#%% calculate and plot mds


RDM_pre = np.mean(indiv_model_RDMs_pre,-1)          # model-averaged
RDM_post = np.mean(indiv_model_RDMs_post,-1)


from helpers import equal_axes

plot_colours = ['r','y','g','b']

mds_coords_pre = fit_mds_to_rdm(RDM_pre)
mds_coords_post = fit_mds_to_rdm(RDM_post)

plt.figure()
ax = plt.subplot(121,projection='3d')

plot_geometry(ax, mds_coords_pre, plot_colours,legend_on=False)

ax.set_title('pre-cue')


ax2 = plt.subplot(122,projection='3d')
plot_geometry(ax2, mds_coords_post, plot_colours,legend_on=True)
ax2.set_title('post-cue')


equal_axes(ax)
equal_axes(ax2)


#%% make plots nice

ticks1 = np.arange(-.01,.02,.01)
ax.set_xticks(ticks1)
ax.set_yticks(ticks1)
ax.set_zticks(ticks1)

ax.set_xlabel('dim1',labelpad=30,rotation = 350)
ax.set_ylabel('dim2',labelpad=35,rotation=45)
ax.set_zlabel('dim3',labelpad=35,rotation=90)


ax.tick_params(labelsize = 26)
ax.tick_params(axis='x',pad=3)
ax.tick_params(axis='y',pad=5)
ax.tick_params(axis='z',pad=20)


ticks2 = np.arange(-.01,.02,.01)
ax2.set_xticks(ticks2)
ax2.set_yticks(ticks2)
ax2.set_zticks(ticks2)

ax2.tick_params(labelsize = 26)
ax2.tick_params(axis='x',pad=3)
ax2.tick_params(axis='y',pad=5)
ax2.tick_params(axis='z',pad=20)


ax2.set_xlabel('dim1',labelpad=30,rotation=340)
ax2.set_ylabel('dim2',labelpad=30,rotation=45)
ax2.set_zlabel('dim3',labelpad=35,rotation=90)



# plt.legend(bbox_to_anchor=(1,1))
plt.tight_layout()

#%% plot angles
angles_radians = np.radians(np.stack((angle_pre,angle_post)).T)

import seaborn as sns
plt.rcParams.update({'font.size': 30})

pal = sns.color_palette("dark")
inds = [3,0]
cols = [pal[ix] for ix in inds]


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111,polar=True)
ax.grid(False)
r = 1
ms = 16
for i in range(n_models):
    
    ax.plot(angles_radians[i,:],np.ones((2,))*r,'k-',alpha=0.2)
    ax.plot(angles_radians[i,0],r,'o',color = cols[0],alpha=0.2,markersize=ms)
    ax.plot(angles_radians[i,1],r,'o',color = cols[1],alpha=0.2,markersize=ms)
    

ax.plot(pycircstat.descriptive.median(angles_radians[:,0]),r,'o',c = cols[0],markersize=ms,label='pre')
ax.plot(pycircstat.descriptive.median(angles_radians[:,1]),r,'o',c = cols[1],markersize=ms,label='post')




plt.legend(bbox_to_anchor=(1, 1),
           bbox_transform=plt.gcf().transFigure)
 
#%%
ax.set_yticks([])
ax.tick_params(axis='x', which='major', pad=23)

#%% rayleigh test for pre and post-

p_pre, v_pre = pycircstat.tests.rayleigh(angles_radians[:,0])
p_post, v_post = pycircstat.tests.rayleigh(angles_radians[:,1])

np.degrees(pycircstat.descriptive.median(angles_radians[:,1]))
#%%

plt.figure(figsize=(12,5))
    
ax = plt.subplot(121, projection='3d')
ax.plot(np.append(delay1_3dcoords[:n_colours,0],delay1_3dcoords[0,0]),
          np.append(delay1_3dcoords[:n_colours,1],delay1_3dcoords[0,1]),
          np.append(delay1_3dcoords[:n_colours,2],delay1_3dcoords[0,2]),'k-')
ax.scatter(delay1_3dcoords[0,0],delay1_3dcoords[0,1], delay1_3dcoords[0,2],marker='o',s = 40,
          c='k',label='loc1')
ax.scatter(delay1_3dcoords[:n_colours,0],delay1_3dcoords[:n_colours,1],
          delay1_3dcoords[:n_colours,2],marker='o',s = 40,c=plot_colours)


ax2 = plt.subplot(122, projection='3d')

ax2.plot(np.append(delay1_3dcoords[n_colours:,0],delay1_3dcoords[n_colours,0]),
      np.append(delay1_3dcoords[n_colours:,1],delay1_3dcoords[n_colours,1]),
      np.append(delay1_3dcoords[n_colours:,2],delay1_3dcoords[n_colours,2]),'k-')
ax2.scatter(delay1_3dcoords[-1,0],delay1_3dcoords[-1,1], delay1_3dcoords[-1,2],marker='s',s = 40,
      c='k',label='loc2')
ax2.scatter(delay1_3dcoords[n_colours:,0],delay1_3dcoords[n_colours:,1],
      delay1_3dcoords[n_colours:,2],marker='s',s = 40,c=plot_colours)


#%% plot pre-cue

plt.figure()
n_colours = 16
plot_colours = [sns.color_palette("hls", n_colours)]


plt.figure(figsize=(12,5))
ax = plt.subplot(121, projection='3d')

for i in range(16):
    ix = np.arange(i*16,(i+1)*16)
    cols = sns.light_palette(plot_colours[0][i],16,input="hsl")
    ax.scatter(delay1_3dcoords[ix,0],delay1_3dcoords[ix,1],
      delay1_3dcoords[ix,2],marker='o',s = 40,c=cols)
    ax.plot(delay1_3dcoords[ix,0],delay1_3dcoords[ix,1],
      delay1_3dcoords[ix,2],'k--')

ax.set_title('pre-cue loc1')   
ax.set_xlabel('PC1',labelpad=15)
ax.set_ylabel('PC2',labelpad=15)
ax.set_zlabel('PC3',labelpad=15)


ax2 = plt.subplot(122, projection='3d')
for i in range(16):
    ix = np.arange(i*16,(i+1)*16)+16*16
    cols = sns.light_palette(plot_colours[0][i],16,input="hsl")
    ax2.scatter(delay1_3dcoords[ix,0],delay1_3dcoords[ix,1],
      delay1_3dcoords[ix,2],marker='o',s = 40,c=cols)
    ax2.plot(delay1_3dcoords[ix,0],delay1_3dcoords[ix,1],
      delay1_3dcoords[ix,2],'k--')
    
ax2.set_title('pre-cue loc2')   
ax2.set_xlabel('PC1',labelpad=15)
ax2.set_ylabel('PC2',labelpad=15)
ax2.set_zlabel('PC3',labelpad=15)

#%% plot post-cue

plt.figure()
n_colours = 16
plot_colours = [sns.color_palette("hls", n_colours)]


plt.figure(figsize=(12,5))
ax = plt.subplot(121, projection='3d')

for i in range(16):
    # loop through cued colours
    ix = np.arange(i*16,(i+1)*16)
    cols = sns.light_palette(plot_colours[0][i],16,input="hsl")
    ax.scatter(delay2_3dcoords[ix,0],delay2_3dcoords[ix,1],
      delay2_3dcoords[ix,2],marker='o',s = 40,c=cols)
    ax.plot(delay2_3dcoords[ix,0],delay2_3dcoords[ix,1],
      delay2_3dcoords[ix,2],'k--')
    
ax.set_title('post-cue loc1')   
ax.set_xlabel('PC1',labelpad=30)
ax.set_ylabel('PC2',labelpad=30)
ax.set_zlabel('PC3',labelpad=30)


ax2 = plt.subplot(122, projection='3d')
for i in range(16):
    ix = np.arange(i*16,(i+1)*16)+16*16
    cols = sns.light_palette(plot_colours[0][i],16,input="hsl")
    ax2.scatter(delay2_3dcoords[ix,0],delay2_3dcoords[ix,1],
      delay2_3dcoords[ix,2],marker='^',s = 40,c=cols)
    ax2.plot(delay2_3dcoords[ix,0],delay2_3dcoords[ix,1],
      delay2_3dcoords[ix,2],'k--')
    
ax2.set_title('post-cue loc2')   
ax2.set_xlabel('PC1',labelpad=30)
ax2.set_ylabel('PC2',labelpad=30)
ax2.set_zlabel('PC3',labelpad=30)

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
    
#%% plot post-cue -both locagions in one plot

plt.figure()
n_colours = 16
plot_colours = [sns.color_palette("hls", n_colours)]


plt.figure(figsize=(5,5))
ax = plt.subplot(111, projection='3d')

for i in range(16):
    # loop through cued colours
    ix = np.arange(i*16,(i+1)*16)
    cols = sns.light_palette(plot_colours[0][i],16,input="hsl")
    ax.scatter(delay2_3dcoords[ix,0],delay2_3dcoords[ix,1],
      delay2_3dcoords[ix,2],marker='o',s = 40,c=cols)
    ax.plot(delay2_3dcoords[ix,0],delay2_3dcoords[ix,1],
      delay2_3dcoords[ix,2],'k--')
    
ax.set_title('post-cue')   
ax.set_xlabel('PC1',labelpad=30)
ax.set_ylabel('PC2',labelpad=30)
ax.set_zlabel('PC3',labelpad=30)


for i in range(16):
    ix = np.arange(i*16,(i+1)*16)+16*16
    cols = sns.light_palette(plot_colours[0][i],16,input="hsl")
    ax.scatter(delay2_3dcoords[ix,0],delay2_3dcoords[ix,1],
      delay2_3dcoords[ix,2],marker='^',s = 40,c=cols)
    ax.plot(delay2_3dcoords[ix,0],delay2_3dcoords[ix,1],
      delay2_3dcoords[ix,2],'k--')


equal_axes = True
if equal_axes:
    # equal x, y and z axis scale
    ax_lims = np.array(ax.xy_viewLim)
    ax.set_xlim3d(np.min(ax_lims),np.max(ax_lims))
    ax.set_ylim3d(np.min(ax_lims),np.max(ax_lims))
    ax.set_zlim3d(np.min(ax_lims),np.max(ax_lims))
    




#%% plot one colour when it's cued vs uncued - same location /pre-cue

import seaborn as sns

cols1 = sns.color_palette("Blues",16)
cols2 = sns.color_palette("Greens",16)

ms=40

loc1_ix = np.where(labels["loc"][0,:,:]==1)[0] # location1 cued
loc2_ix = np.where(labels["loc"][1,:,:]==1)[0] # location2 cued


colour_vals = np.unique(labels["c1"])
cued_c1_ix = loc1_ix[np.where(labels["c1"][loc1_ix]==colour_vals[0])[0]]
uncued_c1_ix = loc2_ix[np.where(labels["c1"][loc2_ix]==colour_vals[0])[0]]

plt.figure(figsize=(12,5))
ax = plt.subplot(121, projection='3d')
# for i in range(16):
ax.scatter(delay1_3dcoords[cued_c1_ix,0],delay1_3dcoords[cued_c1_ix,1],
           delay1_3dcoords[cued_c1_ix,2],marker='o',c=cols1,edgecolors='k',
           s=ms)


ax2 = plt.subplot(122, projection='3d')

ax2.scatter(delay1_3dcoords[uncued_c1_ix,0],delay1_3dcoords[uncued_c1_ix,1],
           delay1_3dcoords[uncued_c1_ix,2],marker='o',c=cols1, edgecolors='k',
           s=ms)
    
ax.set_title('c1-loc1 cued, pre-cue')
ax2.set_title('c1-loc1 uncued, pre-cue')

ax2.set_xlabel('PC1',labelpad=20)
ax2.set_ylabel('PC2',labelpad=30)
ax2.set_zlabel('PC3',labelpad=30)


#%% the other location

cols1 = sns.color_palette("Blues",16)
cols2 = sns.color_palette("Greens",16)

ms=40

loc1_ix = np.where(labels["loc"][0,:,:]==1)[0] # location1 cued
loc2_ix = np.where(labels["loc"][1,:,:]==1)[0] # location2 cued


colour_vals = np.unique(labels["c1"])
cued_c1_ix = loc2_ix[np.where(labels["c2"][loc2_ix]==colour_vals[0])[0]]
uncued_c1_ix = loc1_ix[np.where(labels["c2"][loc1_ix]==colour_vals[0])[0]]

plt.figure(figsize=(12,5))
ax = plt.subplot(121, projection='3d')
# for i in range(16):
ax.scatter(delay1_3dcoords[cued_c1_ix,0],delay1_3dcoords[cued_c1_ix,1],
           delay1_3dcoords[cued_c1_ix,2],marker='o',c=cols1,edgecolors='k',
           s=ms)


ax2 = plt.subplot(122, projection='3d')

ax2.scatter(delay1_3dcoords[uncued_c1_ix,0],delay1_3dcoords[uncued_c1_ix,1],
           delay1_3dcoords[uncued_c1_ix,2],marker='o',c=cols1, edgecolors='k',
           s=ms)
    
ax.set_title('c1-loc2 cued, pre-cue')
ax2.set_title('c1-loc2 uncued, pre-cue')

ax2.set_xlabel('PC1',labelpad=20)
ax2.set_ylabel('PC2',labelpad=30)
ax2.set_zlabel('PC3',labelpad=30)

#%% plot one colour when it's cued vs uncued - same location /post-cue


ms=40

loc1_ix = np.where(labels["loc"][0,:,:]==1)[0] # location1 cued
loc2_ix = np.where(labels["loc"][1,:,:]==1)[0] # location2 cued


colour_vals = np.unique(labels["c1"])
cued_c1_ix = loc1_ix[np.where(labels["c1"][loc1_ix]==colour_vals[0])[0]]
uncued_c1_ix = loc2_ix[np.where(labels["c1"][loc2_ix]==colour_vals[0])[0]]

plt.figure(figsize=(12,5))
ax = plt.subplot(121, projection='3d')
# for i in range(16):
ax.scatter(delay2_3dcoords[cued_c1_ix,0],delay2_3dcoords[cued_c1_ix,1],
           delay2_3dcoords[cued_c1_ix,2],marker='o',c=cols1,edgecolors='k',
           s=ms)


ax2 = plt.subplot(122, projection='3d')

ax2.scatter(delay2_3dcoords[uncued_c1_ix,0],delay2_3dcoords[uncued_c1_ix,1],
           delay2_3dcoords[uncued_c1_ix,2],marker='o',c=cols1, edgecolors='k',
           s=ms)
    
ax.set_title('c1-loc1 cued, pre-cue')
ax2.set_title('c1-loc1 uncued, pre-cue')

ax2.set_xlabel('PC1',labelpad=20)
ax2.set_ylabel('PC2',labelpad=30)
ax2.set_zlabel('PC3',labelpad=30)

#%% the other location


ms=40

loc1_ix = np.where(labels["loc"][0,:,:]==1)[0] # location1 cued
loc2_ix = np.where(labels["loc"][1,:,:]==1)[0] # location2 cued


colour_vals = np.unique(labels["c1"])
cued_c1_ix = loc2_ix[np.where(labels["c2"][loc2_ix]==colour_vals[0])[0]]
uncued_c1_ix = loc1_ix[np.where(labels["c2"][loc1_ix]==colour_vals[0])[0]]


plt.figure(figsize=(12,5))
ax = plt.subplot(121, projection='3d')
# for i in range(16):
ax.scatter(delay2_3dcoords[cued_c1_ix,0],delay2_3dcoords[cued_c1_ix,1],
           delay2_3dcoords[cued_c1_ix,2],marker='o',c=cols1,edgecolors='k',
           s=ms)


ax2 = plt.subplot(122, projection='3d')

ax2.scatter(delay2_3dcoords[uncued_c1_ix,0],delay2_3dcoords[uncued_c1_ix,1],
           delay2_3dcoords[uncued_c1_ix,2],marker='o',c=cols1, edgecolors='k',
           s=ms)
    
ax.set_title('c1-loc2 cued, pre-cue')
ax2.set_title('c1-loc2 uncued, pre-cue')

ax2.set_xlabel('PC1',labelpad=20)
ax2.set_ylabel('PC2',labelpad=30)
ax2.set_zlabel('PC3',labelpad=30)


#%% plot post-cue, all together


# location 1
cued_c1_ix = loc1_ix[np.where(labels["c1"][loc1_ix]==colour_vals[0])[0]]
uncued_c1_ix = loc2_ix[np.where(labels["c1"][loc2_ix]==colour_vals[0])[0]]


plt.figure(figsize=(5,5))
ax = plt.subplot(111, projection='3d')

# loc 1 cued
ax.scatter(delay2_3dcoords[cued_c1_ix,0],delay2_3dcoords[cued_c1_ix,1],
           delay2_3dcoords[cued_c1_ix,2],marker='o',c=cols1,edgecolors='k',
           s=ms,label='loc1')
ax.plot(delay2_3dcoords[cued_c1_ix,0],delay2_3dcoords[cued_c1_ix,1],
           delay2_3dcoords[cued_c1_ix,2],'k-')


# loc 1 uncued
ax.scatter(delay2_3dcoords[uncued_c1_ix,0],delay2_3dcoords[uncued_c1_ix,1],
           delay2_3dcoords[uncued_c1_ix,2],marker='o',c=cols1, edgecolors='k',
           s=ms)

ax.plot(delay2_3dcoords[uncued_c1_ix,0],delay2_3dcoords[uncued_c1_ix,1],
           delay2_3dcoords[uncued_c1_ix,2],'k--')



cued_c1_ix = loc2_ix[np.where(labels["c2"][loc2_ix]==colour_vals[0])[0]]
uncued_c1_ix = loc1_ix[np.where(labels["c2"][loc1_ix]==colour_vals[0])[0]]


#loc 2 cued
ax.scatter(delay2_3dcoords[cued_c1_ix,0],delay2_3dcoords[cued_c1_ix,1],
           delay2_3dcoords[cued_c1_ix,2],marker='^',c=cols1,edgecolors='k',
           s=ms,label='loc2')
ax.plot(delay2_3dcoords[cued_c1_ix,0],delay2_3dcoords[cued_c1_ix,1],
           delay2_3dcoords[cued_c1_ix,2],'k-',label='cued')



# loc 2 uncued
ax.scatter(delay2_3dcoords[uncued_c1_ix,0],delay2_3dcoords[uncued_c1_ix,1],
           delay2_3dcoords[uncued_c1_ix,2],marker='^',c=cols1, edgecolors='k',
           s=ms)

ax.plot(delay2_3dcoords[uncued_c1_ix,0],delay2_3dcoords[uncued_c1_ix,1],
           delay2_3dcoords[uncued_c1_ix,2],'k--',label='uncued')  

plt.legend(bbox_to_anchor=(1.2,1))


ax.set_xlabel('PC1',labelpad=20)
ax.set_ylabel('PC2',labelpad=30)
ax.set_zlabel('PC3',labelpad=30)

# ax.set_title('post-cue,c1')


equal_axes = True
if equal_axes:
    # equal x, y and z axis scale
    ax_lims = np.array(ax.xy_viewLim)
    ax.set_xlim3d(np.min(ax_lims),np.max(ax_lims))
    ax.set_ylim3d(np.min(ax_lims),np.max(ax_lims))
    ax.set_zlim3d(np.min(ax_lims),np.max(ax_lims))


#%% add another colour

# location 1
cued_c1_ix = loc1_ix[np.where(labels["c1"][loc1_ix]==colour_vals[1])[0]]
uncued_c1_ix = loc2_ix[np.where(labels["c1"][loc2_ix]==colour_vals[1])[0]]


# loc 1 cued
ax.scatter(delay2_3dcoords[cued_c1_ix,0],delay2_3dcoords[cued_c1_ix,1],
           delay2_3dcoords[cued_c1_ix,2],marker='o',c=cols2,edgecolors='k',
           s=ms,label='loc1')
ax.plot(delay2_3dcoords[cued_c1_ix,0],delay2_3dcoords[cued_c1_ix,1],
           delay2_3dcoords[cued_c1_ix,2],'k-')


# loc 1 uncued
ax.scatter(delay2_3dcoords[uncued_c1_ix,0],delay2_3dcoords[uncued_c1_ix,1],
           delay2_3dcoords[uncued_c1_ix,2],marker='o',c=cols2, edgecolors='k',
           s=ms)

ax.plot(delay2_3dcoords[uncued_c1_ix,0],delay2_3dcoords[uncued_c1_ix,1],
           delay2_3dcoords[uncued_c1_ix,2],'k--')



cued_c1_ix = loc2_ix[np.where(labels["c2"][loc2_ix]==colour_vals[1])[0]]
uncued_c1_ix = loc1_ix[np.where(labels["c2"][loc1_ix]==colour_vals[1]])[0]]


#loc 2 cued
ax.scatter(delay2_3dcoords[cued_c1_ix,0],delay2_3dcoords[cued_c1_ix,1],
           delay2_3dcoords[cued_c1_ix,2],marker='^',c=cols2,edgecolors='k',
           s=ms,label='loc2')
ax.plot(delay2_3dcoords[cued_c1_ix,0],delay2_3dcoords[cued_c1_ix,1],
           delay2_3dcoords[cued_c1_ix,2],'k-',label='cued')



# loc 2 uncued
ax.scatter(delay2_3dcoords[uncued_c1_ix,0],delay2_3dcoords[uncued_c1_ix,1],
           delay2_3dcoords[uncued_c1_ix,2],marker='^',c=cols2, edgecolors='k',
           s=ms)

ax.plot(delay2_3dcoords[uncued_c1_ix,0],delay2_3dcoords[uncued_c1_ix,1],
           delay2_3dcoords[uncued_c1_ix,2],'k--',label='uncued')  



#%% plot pre-cue, all together


# location 1
cued_c1_ix = loc1_ix[np.where(labels["c1"][loc1_ix]==colour_vals[0])[0]]
uncued_c1_ix = loc2_ix[np.where(labels["c1"][loc2_ix]==colour_vals[0])[0]]


plt.figure(figsize=(5,5))
ax = plt.subplot(111, projection='3d')

# loc 1 cued
ax.scatter(delay1_3dcoords[cued_c1_ix,0],delay1_3dcoords[cued_c1_ix,1],
           delay1_3dcoords[cued_c1_ix,2],marker='o',c=cols1,edgecolors='k',
           s=ms,label='loc1')
ax.plot(delay1_3dcoords[cued_c1_ix,0],delay1_3dcoords[cued_c1_ix,1],
           delay1_3dcoords[cued_c1_ix,2],'k-')


# loc 1 uncued
ax.scatter(delay1_3dcoords[uncued_c1_ix,0],delay1_3dcoords[uncued_c1_ix,1],
           delay1_3dcoords[uncued_c1_ix,2],marker='o',c=cols1, edgecolors='k',
           s=ms)

ax.plot(delay1_3dcoords[uncued_c1_ix,0],delay1_3dcoords[uncued_c1_ix,1],
           delay1_3dcoords[uncued_c1_ix,2],'k--')



cued_c1_ix = loc2_ix[np.where(labels["c2"][loc2_ix]==colour_vals[0])[0]]
uncued_c1_ix = loc1_ix[np.where(labels["c2"][loc1_ix]==colour_vals[0])[0]]


#loc 2 cued
ax.scatter(delay1_3dcoords[cued_c1_ix,0],delay1_3dcoords[cued_c1_ix,1],
           delay1_3dcoords[cued_c1_ix,2],marker='^',c=cols1,edgecolors='k',
           s=ms,label='loc2')
ax.plot(delay1_3dcoords[cued_c1_ix,0],delay1_3dcoords[cued_c1_ix,1],
           delay1_3dcoords[cued_c1_ix,2],'k-',label='cued')



# loc 2 uncued
ax.scatter(delay1_3dcoords[uncued_c1_ix,0],delay1_3dcoords[uncued_c1_ix,1],
           delay1_3dcoords[uncued_c1_ix,2],marker='^',c=cols1, edgecolors='k',
           s=ms)

ax.plot(delay1_3dcoords[uncued_c1_ix,0],delay1_3dcoords[uncued_c1_ix,1],
           delay1_3dcoords[uncued_c1_ix,2],'k--',label='uncued')  

plt.legend(bbox_to_anchor=(1.2,1))


ax.set_xlabel('PC1',labelpad=20)
ax.set_ylabel('PC2',labelpad=30)
ax.set_zlabel('PC3',labelpad=30)

# ax.set_title('post-cue,c1')


equal_axes = True
if equal_axes:
    # equal x, y and z axis scale
    ax_lims = np.array(ax.xy_viewLim)
    ax.set_xlim3d(np.min(ax_lims),np.max(ax_lims))
    ax.set_ylim3d(np.min(ax_lims),np.max(ax_lims))
    ax.set_zlim3d(np.min(ax_lims),np.max(ax_lims))

#%% add another colour

# location 1
cued_c1_ix = loc1_ix[np.where(labels["c1"][loc1_ix]==colour_vals[1])[0]]
uncued_c1_ix = loc2_ix[np.where(labels["c1"][loc2_ix]==colour_vals[1])[0]]


# loc 1 cued
ax.scatter(delay1_3dcoords[cued_c1_ix,0],delay1_3dcoords[cued_c1_ix,1],
           delay1_3dcoords[cued_c1_ix,2],marker='o',c=cols2,edgecolors='k',
           s=ms,label='loc1')
ax.plot(delay1_3dcoords[cued_c1_ix,0],delay1_3dcoords[cued_c1_ix,1],
           delay1_3dcoords[cued_c1_ix,2],'k-')


# loc 1 uncued
ax.scatter(delay1_3dcoords[uncued_c1_ix,0],delay1_3dcoords[uncued_c1_ix,1],
           delay1_3dcoords[uncued_c1_ix,2],marker='o',c=cols2, edgecolors='k',
           s=ms)

ax.plot(delay1_3dcoords[uncued_c1_ix,0],delay1_3dcoords[uncued_c1_ix,1],
           delay1_3dcoords[uncued_c1_ix,2],'k--')



cued_c1_ix = loc2_ix[np.where(labels["c2"][loc2_ix]==colour_vals[1])[0]]
uncued_c1_ix = loc1_ix[np.where(labels["c2"][loc1_ix]==colour_vals[1])[0]]


#loc 2 cued
ax.scatter(delay1_3dcoords[cued_c1_ix,0],delay1_3dcoords[cued_c1_ix,1],
           delay1_3dcoords[cued_c1_ix,2],marker='^',c=cols2,edgecolors='k',
           s=ms,label='loc2')
ax.plot(delay1_3dcoords[cued_c1_ix,0],delay1_3dcoords[cued_c1_ix,1],
           delay1_3dcoords[cued_c1_ix,2],'k-',label='cued')



# loc 2 uncued
ax.scatter(delay1_3dcoords[uncued_c1_ix,0],delay1_3dcoords[uncued_c1_ix,1],
           delay1_3dcoords[uncued_c1_ix,2],marker='^',c=cols2, edgecolors='k',
           s=ms)

ax.plot(delay1_3dcoords[uncued_c1_ix,0],delay1_3dcoords[uncued_c1_ix,1],
           delay1_3dcoords[uncued_c1_ix,2],'k--',label='uncued')  


#%% plot uncued vs uncued binned,for both pre- and post-cue

plot_colours = ['r','y','g','b']

plt.figure()
ax = plt.subplot(121,projection='3d')

plot_geometry(ax, delay1_binned_3dcoords, plot_colours,legend_on=False)

ax.set_xlabel('PC1',labelpad=20)
ax.set_ylabel('PC2',labelpad=20)
ax.set_zlabel('PC3',labelpad=20)
ax.set_title('pre-cue')


ax2 = plt.subplot(122,projection='3d')
plot_geometry(ax2, delay2_binned_3dcoords, plot_colours)

# plt.legend(['loc1 cued','loc2 uncued'])

ax2.set_xlabel('PC1',labelpad=20)
ax2.set_ylabel('PC2',labelpad=20)
ax2.set_zlabel('PC3',labelpad=20)

ax2.set_xticks(np.arange(-0.5,.5,.5))
ax2.set_yticks(np.arange(-0.5,.5,.5))
ax2.set_zticks(np.arange(-0.5,.5,.5))


ax2.set_title('post-cue')


