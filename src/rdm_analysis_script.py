#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 10:50:00 2020

@author: emilia
"""
from sklearn.manifold import MDS
from scipy.spatial.distance import squareform, pdist

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import vec_operations as vops
import pickle
import constants
from helpers import check_path


#%%
load_path = constants.PARAMS['FULL_PATH']+'pca_data/valid_trials/'


# f = open(constants.PARAMS['FULL_PATH']+'saved_models/model_params.pckl','rb')
# obj = pickle.load(f)
# [n_inp,n_rec,n_iter,batch_size,learning_rate,fixation] = obj
# n_colCh = (n_inp - 2 - int(fixation))//2

#%% construct data RDMs

n_conditions = constants.PARAMS['B']*constants.PARAMS['B']*2


rdm_precue = np.zeros((n_conditions,n_conditions,constants.PARAMS['n_models']))
rdm_postcue = np.zeros((n_conditions,n_conditions,constants.PARAMS['n_models']))

rdm_precue_binned = np.zeros((constants.PARAMS['M'],constants.PARAMS['M'],constants.PARAMS['n_models']))
rdm_postcue_binned = np.zeros((constants.PARAMS['M'],constants.PARAMS['M'],constants.PARAMS['n_models']))
rdm_stimuli_binned = np.zeros((constants.PARAMS['M'],constants.PARAMS['M'],constants.PARAMS['n_models']))
# rdm_postcue_binned_readout = np.zeros((batch_size//4,batch_size//4))


# plt.figure()
# pick timepoints of interest
t1 = constants.PARAMS['trial_timepoints']['delay1_end']-1
t2 = constants.PARAMS['trial_timepoints']['delay2_end']-1

for model in range(constants.PARAMS['n_models']):

    # load pca data
    f = open(load_path+'/rdm_data_model' + str(model) + '.pckl', 'rb')
    obj = pickle.load(f)
    rdm_data = obj
    f.close()
    # data format: trial x time x neuron
    
    rdm_precue[:,:,model] = squareform(pdist(rdm_data[:,t1,:],'correlation'))
    rdm_postcue[:,:,model] = squareform(pdist(rdm_data[:,t2,:],'correlation'))
    
    f = open(load_path+'/pca_data_model' + str(model) + '.pckl', 'rb')
    pca_data = pickle.load(f)
    f.close()
    rdm_stimuli_binned[:,:,model] = squareform(pdist(pca_data['data'][:,0,:],'correlation'))
    rdm_precue_binned[:,:,model] = squareform(pdist(pca_data['delay1'],'correlation'))
    rdm_postcue_binned[:,:,model] = squareform(pdist(pca_data['delay2'],'correlation'))
    
    # rdm_postcue_binned_readout[:,:] = squareform(pdist(readout,'correlation'))
    
    # plt.subplot(2,10,int(model_number)+1)
    # plt.imshow(rdm_postcue_binned[:,:,i])
    # plt.colorbar()
    # plt.title('Model '+model_number)

#%% save

save_path = constants.PARAMS['FULL_PATH']+'RSA/'
check_path(save_path)

indiv_model_rdms = [rdm_precue_binned,rdm_postcue_binned]
pickle.dump(indiv_model_rdms,open(save_path+'indiv_model_rdms.pckl','wb'))

# average axcross models
pre_data_RDM_averaged = np.mean(rdm_precue_binned,2)
post_data_RDM_averaged = np.mean(rdm_postcue_binned,2)

pickle.dump(pre_data_RDM_averaged,open(save_path+'pre_data_RDM_averaged.pckl','wb'))
pickle.dump(post_data_RDM_averaged,open(save_path+'post_data_RDM_averaged.pckl','wb'))


# # center
# pre_data_RDM_averaged -= np.mean(pre_data_RDM_averaged) 
# post_data_RDM_averaged -= np.mean(post_data_RDM_averaged)


#%% plot rdms averaged across models - all conditions
import seaborn as sns

plt.figure(figsize=(10,20))
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

#%%  plot rdms averaged across models - binned across uncued locations

# plt.figure(figsize=(10,20),num = 'Averaged RDMs')


fig, ax = plt.subplots(1,2,figsize=(10,10),num = 'Averaged RDMs')


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



#%% correlate model and data RDMs


# load model RDMs
path = base_path + 'RSA/model_RDMs_sqform.pckl'
f = open(path,'rb')
model_RDMs = pickle.load(f) # individual predictor arrays stacked along axis 1
f.close()

path = base_path + 'RSA/model_RDMs_order.pckl'
f = open(path,'rb')
model_RDMs_order = pickle.load(f) # individual predictor arrays stacked along axis 1
f.close()


# # correlate pre-cue
# from scipy.stats import spearmanr
# up_ix = np.triu_indices(model_RDMs.shape[0],k=1) # upper triangle index
# (r_pre, p_pre) = spearmanr(model_RDMs[up_ix[0],up_ix[1],0].reshape(-1),pre_data_RDM_averaged[up_ix[0],up_ix[1]].reshape(-1))
# print('Correlation between pre-cue data and model predictions: r = %.3f, p = %.3f' %(r_pre,p_pre))


# # correlate post-cue
# (r_post, p_post) = spearmanr(model_RDMs[up_ix[0],up_ix[1],1].reshape(-1),post_data_RDM_averaged[up_ix[0],up_ix[1]].reshape(-1))
# print('Correlation between post-cue data and model predictions: r = %.3f, p = %.3f' %(r_post,p_post))


# # correlate pre-cue
# up_ix = np.triu_indices(model_RDMs.shape[0],k=1)
# (r_pre, p_pre) = spearmanr(model_RDMs[up_ix[0],up_ix[1],0].reshape(-1),pre_data_RDM_averaged[up_ix[0],up_ix[1]].reshape(-1))
# print('Correlation between pre-cue data and model predictions: r = %.3f, p = %.3f' %(r_pre,p_pre))


# # correlate post-cue
# (r_post, p_post) = spearmanr(model_RDMs[up_ix[0],up_ix[1],1].reshape(-1),post_data_RDM_averaged[up_ix[0],up_ix[1]].reshape(-1))
# print('Correlation between post-cue data and model predictions: r = %.3f, p = %.3f' %(r_post,p_post))

#%%
# from scipy.spatial.distance import pdist, squareform
from scipy.stats import zscore, stats, linregress
from numpy.linalg import lstsq, inv
import statsmodels.api as sm


model_RDMs = zscore(model_RDMs)
X = sm.add_constant(model_RDMs)

y_pre = squareform(pre_data_RDM_averaged)
y_post = squareform(post_data_RDM_averaged)


b_pre, res_pre, rank_pre, s_pre = lstsq(X, y_pre)
b_post, res_post, rank_post, s_post = lstsq(X, y_post)


results_pre = sm.OLS(y_pre, X).fit()
print('Regression results: pre-cue')
print(results_pre.summary())

results_post = sm.OLS(y_post, X).fit()
print('Regression results: post-cue')
print(results_post.summary())

#%% plot results
# beta_names = model_RDMs_order.copy()
# beta_names.insert(0,'const')



plt.figure()
ax1 = plt.subplot(121)

ix = np.arange(1,4)

betas = results_pre.params[ix]
CIs = results_pre.conf_int()[ix,1] - betas
ax1.bar(np.arange(len(betas)),betas,color = [.5,.5,.5],yerr = CIs)
plt.xticks(np.arange(len(betas)),labels = model_RDMs_order,rotation=20)



ax2 = plt.subplot(122,sharex = ax1,sharey = ax1)

betas = results_post.params[ix]
CIs = results_post.conf_int()[ix,1] - betas
ax2.bar(np.arange(len(betas)),betas,color = [.5,.5,.5],yerr = CIs)



plt.xticks(np.arange(len(betas)),labels = model_RDMs_order,rotation=20)
ax1.set_ylabel('Beta coefficient')
ax1.set_title('pre-cue')
ax2.set_title('post-cue')

plt.tight_layout(pad=2)


#%% plot predictor RDMs


path = base_path + 'RSA/model_RDMs.pckl'
f = open(path,'rb')
model_RDMs_full = pickle.load(f) # pre- and post-cue arrays stacked along axis 2
f.close()

ortho180 = pickle.load(open(base_path + 'RSA/ortho_rotated_fullRDMs/180.pckl','rb'))

# plt.figure(figsize=(10,20),num = 'Model RDMs')
# plt.subplot(121)
# plt.imshow(squareform(model_RDMs[:,0]))
# plt.colorbar()
# plt.title('pre-cue')
# plt.yticks([])
# plt.xticks([])


# plt.subplot(122)
# plt.imshow(squareform(model_RDMs[:,1]))
# plt.colorbar()
# plt.title('post-cue')
# plt.yticks([])
# plt.xticks([])

fig, axes = plt.subplots(1,3, sharex=True, sharey = True)


titles = model_RDMs_order
for ix,ax in enumerate(axes.flat):
    if ix == 0:
        im = ax.imshow(ortho180,vmin=np.min(ortho180),vmax=np.max(ortho180))
    else:
        im = ax.imshow(model_RDMs_full[:,:,ix],vmin=np.min(model_RDMs_full[:,:,ix]),vmax=np.max(model_RDMs_full[:,:,ix]))
    ax.set_title(titles[ix])
    ax.set_xticks([])
    ax.set_yticks([])

fig.colorbar(im,ax=axes.ravel().tolist())

# plt.tight_layout(pad=2)


#%% create an x2 + x3 RDM

# betas =  results_post.params[-2:]
# betas /= np.sum(betas)

# x2x3RDM = betas[0] * model_RDMs_full[:,:,1] +  betas[1] * model_RDMs_full[:,:,2]

# plt.figure()
# plt.imshow(x2x3RDM)
# plt.colorbar()


from sklearn.manifold import MDS
from scipy.spatial.distance import squareform, pdist


def fit_mds_to_rdm(rdm):
    mds = MDS(n_components=3, 
              metric=True, 
              dissimilarity='precomputed', 
              max_iter=1000,
              random_state=0)
    return mds.fit_transform(rdm)


# mds_x2x3 = fit_mds_to_rdm(x2x3RDM)


# plt.figure()
# ax = plt.subplot(111,projection='3d')

# colours = ['r','y','g','b']

# plot_geometry(ax,mds_x2x3,colours)


# mds_parallel = fit_mds_to_rdm(model_RDMs_full[:,:,1])
# d_parallel = np.linalg.norm(vops.getVecFromPoints(mds_parallel[0,:],mds_parallel[4,:]))
# # plane distance
# d = []
# for n in range(4):
#     v = vops.getVecFromPoints(mds_x2x3[n,:], mds_x2x3[n+4,:])
#     d.append(np.linalg.norm(v))



# d = np.array(d)
# print('Mean distance between planes: %.2f' %np.mean(d))

#%% do MDS on model-averaged RDM
mds_stimuli = fit_mds_to_rdm(np.mean(rdm_stimuli_binned,2))
mds_precue = fit_mds_to_rdm(pre_data_RDM_averaged)
mds_postcue = fit_mds_to_rdm(post_data_RDM_averaged)


# mds1 = MDS(n_components=3, 
#               metric=False, 
#               dissimilarity='precomputed', 
#               max_iter=1000,
#               random_state=0)
# mds2 = MDS(n_components=3, 
#               metric=False, 
#               dissimilarity='precomputed', 
#               max_iter=1000,
#               random_state=0)


# mds_precue = mds1.fit_transform(pre_data_RDM_averaged)
# mds_postcue = mds2.fit_transform(post_data_RDM_averaged)




def plot_geometry(ax,data,pca,plot_colours,plot_outline = True,legend_on = True):
    ms = 150
    n_colours = len(plot_colours)
    # plot the parallelogram defined by colours at location 1
    if plot_outline:
        ax.plot(np.append(data[:n_colours,0],data[0,0]),
              np.append(data[:n_colours,1],data[0,1]),
              np.append(data[:n_colours,2],data[0,2]),'k-')
    ax.scatter(data[0,0],data[0,1], data[0,2],marker='^',s = ms,
              c='k',label='loc1')
    ax.scatter(data[:n_colours,0],data[:n_colours,1],
              data[:n_colours,2],marker='^',s = ms,c=plot_colours)
  
    # repeat for loc 2
    if plot_outline:
        ax.plot(np.append(data[n_colours:,0],data[n_colours,0]),
              np.append(data[n_colours:,1],data[n_colours,1]),
              np.append(data[n_colours:,2],data[n_colours,2]),'k-')
    ax.scatter(data[-1,0],data[-1,1], data[-1,2],marker='s',s = ms,
              c='k',label='loc2')
    ax.scatter(data[n_colours:,0],data[n_colours:,1],
              data[n_colours:,2],marker='s',s = ms,c=plot_colours)
    
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
    #sorted_verts, sorting_order = vops.sortByVecAngle(verts)
    #sorted_verts = verts
    # plot the best-fit plane
    plot_plane(ax,sorted_verts,fc,a)
    #return verts, sorted_verts

#%%  plot mds

from helpers import equal_axes

colours = ['r','y','g','b']
n_colours = len(colours)

plt.figure()
ax1 = plt.subplot(121, projection='3d')
ax2 = plt.subplot(122, projection='3d')
plot_geometry(ax1,mds_precue,[],colours,legend_on=False)
plot_geometry(ax2,mds_postcue,[],colours)    

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

equal_axes(ax1)
equal_axes(ax2)

plt.legend(bbox_to_anchor=(1.5,1))
plt.tight_layout()


ax1.set_xticks(np.arange(-.01,.02,.01))
ax1.set_yticks(np.arange(-.01,.02,.01))
ax1.set_zticks(np.arange(-.01,.02,.01))

lims = ax2.get_xlim()
ax2.set_xticks(np.arange(-.04,.05,.04))
ax2.set_yticks(np.arange(-.04,.05,.04))
ax2.set_zticks(np.arange(-.04,.05,.04))



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

#%% get and plot planes of best fit
from rep_geom import get_best_fit_plane, get_angle_between_planes
####### PRE_CUE

# get directions of max variance, i.e. vectors defining the plane
delay1_planeUp = get_best_fit_plane(mds_precue[0:n_colours])
delay1_planeDown = get_best_fit_plane(mds_precue[n_colours:])

# calculate angle between planes
theta_pre = get_angle_between_planes(delay1_planeUp.components_,delay1_planeDown.components_)
print('Angle pre-cue: %.2f' %theta_pre)

plot_subspace(ax1,mds_precue[:n_colours,:],delay1_planeUp.components_,fc='k',a=0.2)
plot_subspace(ax1,mds_precue[n_colours:,:],delay1_planeDown.components_,fc='k',a=0.2)

####### POST_CUE

delay2_planeUp = get_best_fit_plane(mds_postcue[0:n_colours])
delay2_planeDown = get_best_fit_plane(mds_postcue[n_colours:])

theta_post = get_angle_between_planes(delay2_planeUp.components_,delay2_planeDown.components_,)
print('Angle post-cue: %.2f' %theta_post)

plot_subspace(ax2,mds_postcue[:n_colours,:],delay2_planeUp.components_,fc='k',a=0.2)
plot_subspace(ax2,mds_postcue[n_colours:,:],delay2_planeDown.components_,fc='k',a=0.2)

ax1.set_title('pre-cue')
ax2.set_title('post-cue')

ax1.set_xlabel('PC1',labelpad = 20,style = 'italic')
ax1.set_ylabel('PC2',labelpad = 20,style = 'italic')
ax1.set_zlabel('PC3',labelpad = 15,style = 'italic')

ax2.set_xlabel('PC1',labelpad = 20,style = 'italic')
ax2.set_ylabel('PC2',labelpad = 20,style = 'italic')
ax2.set_zlabel('PC3',labelpad = 15,style = 'italic')



#%% plot pre-cue RDMs for each model

plt.figure()
for m in range(n_models):
    plt.subplot(2,5,m+1)
    plt.imshow(rdm_precue_binned[:,:,m])
    plt.colorbar()
    
#%% MDS for each model
colours = ['r','y','g','b']

plt.figure()
for m in range(n_models):
    ax = plt.subplot(2,5,m+1, projection='3d')
    mds = fit_mds_to_rdm(rdm_precue_binned[:,:,m])
    plot_geometry(ax,mds,[],colours,legend_on=False)

#%% do the regression separately for each model



#%%


# #%% plot RDMs for individual models

# for model in range(n_models):
#     plt.figure(num=model+1)
#     plt.subplot(111)
# #     plt.imshow(rdm_precue_binned[:,:,converged[model]])
# #     plt.colorbar()
# #     plt.title('pre-cue')
    
# #     plt.subplot(122)
#     plt.imshow(rdm_postcue_binned[:,:,converged[model]])
#     plt.colorbar()
#     plt.title('post-cue')
# #%% do mds on rdms








# mds_precue = fit_mds_to_rdm(pre_data_RDM_averaged)
# mds_postcue = fit_mds_to_rdm(post_data_RDM_averaged)


# print('mds complete')

# #%%define plotting funcs



# #%% look at intrinsic dimensionality
# mds_indiv_models = np.zeros((8,3,2,n_models)) # conditions x coordinates x pre/post-cue x model
# PVEs_indiv_models = np.zeros((2,2,n_models)) # plane location x pre/post cue x model
# angles_indiv_models = np.zeros((2,n_models))
# for model in range(n_models):
    
#     # do mds
#     mds_indiv_models[:,:,0,model] = fit_mds_to_rdm(rdm_precue_binned[:,:,model])
#     mds_indiv_models[:,:,1,model] = fit_mds_to_rdm(rdm_postcue_binned[:,:,model])
    
#     # fit planes

#     pre_Up = get_best_fit_plane(mds_indiv_models[:n_colours,:,0,model])
#     pre_Down = get_best_fit_plane(mds_indiv_models[n_colours:,:,0,model])
#     post_Up = get_best_fit_plane(mds_indiv_models[:n_colours,:,1,model])
#     post_Down = get_best_fit_plane(mds_indiv_models[n_colours:,:,1,model])
    
#     angles_indiv_models[0,model] = get_angle_between_planes(pre_Up.components_,pre_Down.components_,)
#     angles_indiv_models[1,model] = get_angle_between_planes(post_Up.components_,post_Down.components_,)
    
    
#     PVEs_indiv_models[0,0,model] = np.sum(pre_Up.explained_variance_ratio_)
#     PVEs_indiv_models[1,0,model] = np.sum(pre_Down.explained_variance_ratio_)
#     PVEs_indiv_models[0,1,model] = np.sum(post_Up.explained_variance_ratio_)
#     PVEs_indiv_models[1,1,model] = np.sum(post_Down.explained_variance_ratio_)
    
#     # plot mds for indiv. models
#     # plt.figure(num=model)
#     # ax = plt.subplot(121, projection='3d')
#     # ax2 = plt.subplot(122, projection='3d')
#     # plot_geometry(ax,mds_indiv_models[:,:,0,model],[],colours)
#     # plot_geometry(ax2,mds_indiv_models[:,:,1,model],[],colours)   
    
#     # plot_subspace(ax,mds_indiv_models[:n_colours,:,0,model],pre_Up.components_,fc='k',a=0.2)
#     # plot_subspace(ax,mds_indiv_models[n_colours:,:,0,model],pre_Down.components_,fc='k',a=0.2)
    
#     # plot_subspace(ax2,mds_indiv_models[:n_colours,:,1,model],post_Up.components_,fc='k',a=0.2)
#     # plot_subspace(ax2,mds_indiv_models[n_colours:,:,1,model],post_Down.components_,fc='k',a=0.2)
    
    
  
# PVEs_indiv_models *= 100 # convert to %

# #%% plot angles - fit to MDS data
# # plt.figure()
# # for model in range(10):
# #     plt.plot(angles_indiv_models[:,model],'k-o')


# #%% plot MDS for individual models
    

# #%% get the % variance explained for each plane


# pve_preUp = (np.sum(delay1_planeUp.explained_variance_ratio_)*100)
# pve_preDown = (np.sum(delay1_planeDown.explained_variance_ratio_)*100)

# pve_postUp = (np.sum(delay2_planeUp.explained_variance_ratio_)*100)
# pve_postDown = (np.sum(delay2_planeDown.explained_variance_ratio_)*100)

# print('% variance explained for fitted planes:')
# print('        pre-cue, loc1: %.2f' %pve_preUp)
# print('        pre-cue, loc2: %.2f' %pve_preDown)
# print('        post-cue, loc1: %.2f' %pve_postUp)
# print('        post-cue, loc2: %.2f' %pve_postDown)

# fig = plt.figure(figsize=(10,10))
# ax = fig.add_subplot(111)
# pre_sim = np.mean([pve_preUp,pve_preDown])
# post_sim = np.mean([pve_postUp,pve_postDown])
# av_monkey = 97.0

# SEM_pre,SEM_post = np.std(np.mean(PVEs_indiv_models,0),1)
# plt.bar(range(3),[pre_sim,post_sim,av_monkey], 
#         color = [0.8, 0.8, 0.8],
#         width = 0.5, 
#         yerr = [SEM_pre,SEM_post,0])
# plt.xticks(range(3),labels = ['pre-cue', 'post-cue', 'data'])
# plt.ylabel('% variance explained')
# plt.ylim([0 ,105])


# # do stats no the PVEs
# mean_PVE_pre,mean_PVE_post = np.mean(PVEs_indiv_models,0)

# plt.figure()
# plt.subplot(121)
# plt.hist(mean_PVE_pre)

# plt.subplot(122)
# plt.hist(mean_PVE_post)

# from scipy.stats import shapiro, ttest_1samp, wilcoxon

# # check for normality
# sw_pre, p_sw_pre = shapiro(mean_PVE_pre)
# sw_post, p_sw_post = shapiro(mean_PVE_post)

# stats = {}
# print('Pre-cue')
# if p_sw_pre<0.05:
#     print('    wilcoxon test')
#     stat_pre, p_pre = wilcoxon(mean_PVE_pre,np.ones(n_models)*av_monkey)
# else:
#     print('    one-sample t-test')
#     stat_pre, p_pre = ttest_1samp(mean_PVE_pre,av_monkey)
# print('        stat = %.3f, p = %.3f' %(stat_pre,p_pre))

# print('Post-cue')
# if p_sw_post<0.05:
#     print('    wilcoxon test')
#     stat_post, p_post = wilcoxon(mean_PVE_post,np.ones(n_models)*av_monkey)
# else:
#     print('    one-sample t-test')
#     stat_post, p_post = ttest_1samp(mean_PVE_post,av_monkey)
# print('        stat = %.3f, p = %.3f' %(stat_post,p_post))

# # plt.figure()
# # ax1 = plt.subplot(111, projection='3d')
# # plot_geometry(ax1,mds_precue,[],colours)
# # scale = 1.0
# # plot_plane_old(ax1,delay2_planeUp,mds_precue[:n_colours,:],scale,fc='k',a=0.2)
# # plot_plane_old(ax1,delay2_planeDown,mds_precue[n_colours:,:],scale,fc='k',a=0.2)



# # #%%
# # ax2 = plt.subplot(122, projection='3d')
# # plot_geometry(ax2,mds_postcue,[],colours)

# # #%%
# # #def plot_plane_old(ax,Y_l,points,scale=1.0,fc='k',a=0.2):
# # scale = 0.2
# # plot_plane_old(ax1,delay1_planeUp,mds_precue[:n_colours,:],scale,fc='k',a=0.2)
# # plot_plane_old(ax1,delay1_planeDown,mds_precue[n_colours:,:],scale,fc='k',a=0.2)
# # ax1.set_yticks(np.arange(-.2,.2,.1))
# # ax1.set_zticks(np.arange(-.2,.1,.1))
# # ax1.set_title('pre-cue')

# # scale = 0.3
# # plot_plane_old(ax2,delay2_planeUp,mds_postcue[:n_colours,:],scale,fc='k',a=0.2)
# # plot_plane_old(ax2,delay2_planeDown,mds_postcue[n_colours:,:],scale,fc='k',a=0.2)
# # ax2.set_title('post-cue')


# #%% check the distribution of data in the RMDs

# # from scipy.stats import shapiro
# # plt.figure()
# # plt.subplot(121)
# # plt.hist(pre_data_RDM_averaged.reshape(-1))

# # s,p = shapiro(pre_data_RDM_averaged.reshape(-1))

# # plt.subplot(122)
# # plt.hist(model_RDMs[:,:,0].reshape(-1))


# # plt.figure()
# # plt.subplot(121)
# # plt.hist(post_data_RDM_averaged.reshape(-1))

# # plt.subplot(122)
# # plt.hist(model_RDMs[:,:,1].reshape(-1))

