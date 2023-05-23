#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:30:34 2021

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
from scipy.spatial import ConvexHull
import custom_plot as cplot

#%%

def plot_geometry(ax,points,plot_colours,plot_outline = True,legend_on=True):
    ms = 150
    n_colours = len(plot_colours)
    # plot the parallelogram defined by colours at location 1
    if plot_outline:
        ax.plot(np.append(points[:n_colours,0],points[0,0]),
              np.append(points[:n_colours,1],points[0,1]),
              np.append(points[:n_colours,2],points[0,2]),'k-')
    ax.scatter(points[0,0],points[0,1], points[0,2],marker='^',s = ms,
              c='k',label='loc1 cued')
    ax.scatter(points[:n_colours,0],points[:n_colours,1],
              points[:n_colours,2],marker='^',s = ms,c=plot_colours)
  
    # repeat for loc 2
    if plot_outline:
        ax.plot(np.append(points[n_colours:,0],points[n_colours,0]),
              np.append(points[n_colours:,1],points[n_colours,1]),
              np.append(points[n_colours:,2],points[n_colours,2]),'k-')
    ax.scatter(points[-1,0],points[-1,1], points[-1,2],marker='s',s = ms,
              c='k',label='loc2 uncued')
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
#%%
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
    
    path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/'+\
                            'data_vonMises/MSELoss/with_fixation_longTrials/kappa1.0/nrec200/lr0.001/'



elif (model_type == 'LSTM'):
    load_path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/LSTM_data/pca_data'
    #load_path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/LSTM_data_1hot/pca_data'
else :
    print('Invalid model type')


load_path = path+'pca_data'
f = open(load_path+'/converged.pckl','rb')
converged = pickle.load(f)
f.close()
n_colours=4

n_models = len(converged)

angle_pre = np.empty((n_models,))
angle_post = np.empty((n_models,))

plot_colours = ['r','y','g','b']
delay1_ix=5
delay2_ix=11

indiv_model_RDMs_pre = np.empty((n_colours*2,n_colours*2,n_models))
indiv_model_RDMs_post = np.empty((n_colours*2,n_colours*2,n_models))

col_discrim = np.empty((n_models,2,2,2)) #[model, location, pre/post,cued/uncued]

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
    labels_cued = np.concatenate((obj["labels"]["c1"][:n_trials//2],
                             obj["labels"]["c2"][n_trials//2:]))
    
    labels_uncued_binned = helpers.bin_labels(labels_uncued,n_colours)
    labels_cued_binned = helpers.bin_labels(labels_cued,n_colours)
    
    delay1 = data[:,delay1_ix,:]
    delay2 = data[:,delay2_ix,:]
     # bin and average the data
    
    delay1_binned = np.zeros((n_colours*2,n_rec))
    delay2_binned = np.zeros((n_colours*2,n_rec))
    
    delay1_binned_loc2 = np.zeros((n_colours*2,n_rec))
    delay2_binned_loc2 = np.zeros((n_colours*2,n_rec))
    
    for colour in range(n_colours):
        ix_cued = np.where(labels_cued_binned==colour)[0]
        ix_uncued = np.where(labels_uncued_binned==colour)[0]
        # ix_down = ix[np.where(ix<n_trials//2)[0]]
        # ix_up = ix[np.where(ix>=n_trials//2)[0]]
        ix_cued_up = ix_cued[np.where(ix_cued<n_trials//2)[0]]
        ix_uncued_down = ix_uncued[np.where(ix_uncued<n_trials//2)[0]]
        
        ix_uncued_up = ix_uncued[np.where(ix_uncued>=n_trials//2)[0]]
        ix_cued_down = ix_cued[np.where(ix_cued>=n_trials//2)[0]]
        
        
        
        delay1_binned[colour,:] = torch.mean(delay1[ix_cued_up,:],0)
        delay1_binned[colour+n_colours,:] = torch.mean(delay1[ix_uncued_down,:],0)
        
        delay2_binned[colour,:] = torch.mean(delay2[ix_cued_up,:],0)
        delay2_binned[colour+n_colours,:] = torch.mean(delay2[ix_uncued_down,:],0)
        
        
        # loc2 cued loc1 uncued
        delay1_binned_loc2[colour,:] = torch.mean(delay1[ix_cued_down,:],0)
        delay1_binned_loc2[colour+n_colours,:] = torch.mean(delay1[ix_uncued_up,:],0)
        
        delay2_binned_loc2[colour,:] = torch.mean(delay2[ix_cued_down,:],0)
        delay2_binned_loc2[colour+n_colours,:] = torch.mean(delay2[ix_uncued_up,:],0)
    
    #% calculate RDMs
    indiv_model_RDMs_pre[:,:,i] = make_rdm(delay1_binned)
    indiv_model_RDMs_post[:,:,i] = make_rdm(delay2_binned)
    
    
    # run pca
    delay1_pca = PCA(n_components=3) # Initializes PCA
    delay2_pca = PCA(n_components=3) # Initializes PCA
    
    
    delay1 -= torch.mean(delay1)
    delay2 -= torch.mean(delay2)
    
    # run PCA
    delay1_3dcoords = delay1_pca.fit_transform(delay1) # get coordinates in the reduced-dim space
    delay2_3dcoords = delay2_pca.fit_transform(delay2)
    
    
    delay1_binned -= np.mean(delay1_binned)
    delay2_binned -= np.mean(delay2_binned)
    
    
    delay1_binned_pca = PCA(n_components=3) # Initializes PCA
    delay2_binned_pca = PCA(n_components=3) # Initializes PCA 
    
    delay1_binned_3dcoords =  delay1_binned_pca.fit_transform(delay1_binned)
    delay2_binned_3dcoords =  delay2_binned_pca.fit_transform(delay2_binned)
    
    # repeat for cued loc2 uncued loc1
    
    delay1_binned_loc2 -= np.mean(delay1_binned_loc2)
    delay2_binned_loc2 -= np.mean(delay2_binned_loc2)
    
    delay1_binned_loc2_pca = PCA(n_components=3) # Initializes PCA
    delay2_binned_loc2_pca = PCA(n_components=3) # Initializes PCA
    
    delay1_binned_loc2_3dcoords =  delay1_binned_loc2_pca.fit_transform(delay1_binned_loc2)
    delay2_binned_loc2_3dcoords =  delay2_binned_loc2_pca.fit_transform(delay2_binned_loc2)
    
    # calculate colour discriminability - area of the polygon formed by the datapoints
    
    #[model, location, pre/post,cued/uncued]
    # up cued
    col_discrim[i,0,0,0] =  ConvexHull(delay1_binned_3dcoords[:4,:]).area
    col_discrim[i,0,1,0] =  ConvexHull(delay2_binned_3dcoords[:4,:]).area
    
    #down uncued
    col_discrim[i,1,0,1] =  ConvexHull(delay1_binned_3dcoords[4:,:]).area
    col_discrim[i,1,1,1] =  ConvexHull(delay2_binned_3dcoords[4:,:]).area
    
    # down cued
    col_discrim[i,1,0,0] =  ConvexHull(delay1_binned_loc2_3dcoords[:4,:]).area
    col_discrim[i,1,1,0] =  ConvexHull(delay2_binned_loc2_3dcoords[:4,:]).area
    
    #up uncued
    col_discrim[i,0,0,1] =  ConvexHull(delay1_binned_loc2_3dcoords[4:,:]).area
    col_discrim[i,0,1,1] =  ConvexHull(delay2_binned_loc2_3dcoords[4:,:]).area
    
    # plane fitting
    
    plane1_pre = get_best_fit_plane(delay1_binned_3dcoords[:n_colours,:])
    plane2_pre = get_best_fit_plane(delay1_binned_3dcoords[n_colours:,:])
    
    plane1_post = get_best_fit_plane(delay2_binned_3dcoords[:n_colours,:])
    plane2_post = get_best_fit_plane(delay2_binned_3dcoords[n_colours:,:])
    
    
    # add correction
    print('Add plane correction')
    angle_pre[i] = get_angle_between_planes(plane1_pre.components_, 
                                         plane2_pre.components_)
    angle_post[i] = get_angle_between_planes(plane1_post.components_, 
                                         plane2_post.components_)
    
    
    # plt.figure()
    # ax = plt.subplot(121,projection='3d')
    
    # plot_geometry(ax, delay1_binned_3dcoords, plot_colours,legend_on=False)
    # plot_subspace(ax,delay1_binned_3dcoords[:n_colours,:],plane1_pre.components_)
    # plot_subspace(ax,delay1_binned_3dcoords[n_colours:,:],plane2_pre.components_)
    
    # helpers.equal_axes(ax)
    # ax.set_xlabel('PC1',labelpad=40)
    # ax.set_ylabel('PC2',labelpad=25)
    # ax.set_zlabel('PC3',labelpad=40)
    # ax.set_title('pre-cue')
    
    
    # ax2 = plt.subplot(122,projection='3d')
    # plot_geometry(ax2, delay2_binned_3dcoords, plot_colours)
    # plot_subspace(ax2,delay2_binned_3dcoords[:n_colours,:],plane1_post.components_)
    # plot_subspace(ax2,delay2_binned_3dcoords[n_colours:,:],plane2_post.components_)
    
    # helpers.equal_axes(ax2)
    # # plt.legend(['loc1 cued','loc2 uncued'])
    
    # ax2.set_xlabel('PC1',labelpad=25)
    # ax2.set_ylabel('PC2',labelpad=40)
    # ax2.set_zlabel('PC3',labelpad=25)
    
    # ax2.set_title('post-cue')
    
    
    
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


ticks2 = np.arange(-.02,.03,.02)
ax2.set_xticks(ticks2)
ax2.set_yticks(ticks2)
ax2.set_zticks(ticks2)

ax2.tick_params(labelsize = 26)
ax2.tick_params(axis='x',pad=3)
ax2.tick_params(axis='y',pad=5)
ax2.tick_params(axis='z',pad=20)


ax2.set_xlabel('dim1',labelpad=30,rotation=355)
ax2.set_ylabel('dim2',labelpad=30,rotation=45)
ax2.set_zlabel('dim3',labelpad=35,rotation=90)



plt.legend(bbox_to_anchor=(.5,1))
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
#%% plot

# plot_colours = [sns.color_palette("hls", n_colours)]
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

ax2.set_title('post-cue')


#%% plot the colour discriminability

# loc1

plt.figure()

colours = ['k','k']
xx = np.zeros((n_models,2))
xx[:,-1] +=1

ax1 = plt.subplot(121)
# # loc 1
cplot.plot_paired_data(xx,col_discrim[:,0,:,0],ax1,['k','dimgrey'],
                        ms=1,alpha=.8,linewidth=1)
cplot.plot_paired_data(xx,col_discrim[:,0,:,1],ax1,['k','lightgrey'],
                        ms=1,alpha=.8,linewidth=1)
ax1.set_title('location 1')


# # loc 2
ax2 = plt.subplot(122,sharex = ax1,sharey=ax1)

cplot.plot_paired_data(xx,col_discrim[:,1,:,0],ax2,['k','dimgrey'],
                        ms=1,alpha=.8,linewidth=1)
cplot.plot_paired_data(xx,col_discrim[:,1,:,1],ax2,['k','lightgrey'],
                        ms=1,alpha=.8,linewidth=1)


ax2.set_title('location 2')

# ax2.plot(1,col_discrim[0,1,1,0],'dimgrey',ms=10,alpha=.9,label='cued')
# ax2.plot(1,col_discrim[0,1,1,1],'lightgrey',ms=10,alpha=.9,label='uncued')


    #[model, location, pre/post,cued/uncued]

ax1.plot(xx[:,0],col_discrim[:,0,0,0],'ko',ms=10)
ax1.plot(xx[:,0],col_discrim[:,0,0,1],'ko',ms=10)
ax1.plot(xx[:,1],col_discrim[:,0,1,0],'o',c = 'dimgrey',ms=10,alpha=.8) #cued
ax1.plot(xx[:,1],col_discrim[:,0,1,1],'o',c = 'lightgrey',ms=10) #uncued

ax2.plot(xx[:,0],col_discrim[:,1,0,0],'ko',ms=10)
ax2.plot(xx[:,0],col_discrim[:,1,0,1],'ko',ms=10)
ax2.plot(xx[:,1],col_discrim[:,1,1,0],'o',c = 'dimgrey',ms=10,alpha=.8,label='cued') #cued
ax2.plot(xx[:,1],col_discrim[:,1,1,1],'o',c = 'lightgrey',ms=10,label='uncued') #uncued



plt.legend()

ax1.set_ylabel('Colour discriminability [a.u.]')

ax1.set_xticks([0,1])
ax1.set_xticklabels(['pre','post'])
ax1.set_xlim(-.2,1.2)

#%% save into a csv file for JASP
import pandas

#[model, location, pre/post,cued/uncued]

# average across locations
data4Jasp = np.mean(col_discrim,1)

# reshape into 2D - delay/cued interactions
tmp = np.reshape(data4Jasp,(data4Jasp.shape[0],data4Jasp.shape[1]*data4Jasp.shape[2]))
# cols are: pre-cued, pre-uncued, post-cued, post-uncued
# average across the first two columns
data4Jasp = np.concatenate((np.expand_dims(np.mean(tmp[:,:2],1),1),tmp[:,2:]),1)



col_labels = ['pre','post-cued','post-uncued']

col_discrim_tbl = pandas.DataFrame(data= data4Jasp,columns = col_labels)
    
col_discrim_tbl.to_csv(path+'/col_discrim_tbl.csv')



