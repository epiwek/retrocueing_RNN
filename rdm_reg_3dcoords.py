#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 14:52:13 2021

@author: emilia
"""

# do the RDM regression on 3D data (after PCA)


import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import vec_operations as vops
import pickle
import os.path
from rep_geom import *


from sklearn.manifold import MDS
from scipy.spatial.distance import squareform, pdist

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import vec_operations as vops
import pickle

#%% 

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
                            'data_vonMises/MSELoss/with_fixation/kappa1.0/nrec200/lr0.005/pca_data'



elif (model_type == 'LSTM'):
    load_path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/LSTM_data/pca_data'
    #load_path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/LSTM_data_1hot/pca_data'
else :
    print('Invalid model type')


base_path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/'+\
                            'data_vonMises/MSELoss/with_fixation/kappa1.0/nrec200/lr0.005/'
f = open(load_path+'/converged.pckl','rb')
converged = pickle.load(f)
f.close()

n_models = len(converged)
#%% create RDMs
n_conditions = 8
rdm_precue = np.zeros((n_conditions,n_conditions,n_models))
rdm_postcue = np.zeros((n_conditions,n_conditions,n_models))

for i,model_number in enumerate(converged):
# for model_number in np.arange(3,4):
    model_number = str(model_number)
    
    # load pca data
    path = load_path 
    f = open(load_path+ '/pca3/coords3D_' + model_type + model_number + '.pckl', 'rb')
    obj = pickle.load(f)    
    [delay1_3dcoords , delay2_3dcoords] = obj
    f.close()
    
    # create RDMs
    
    rdm_precue[:,:,i] = squareform(pdist(delay1_3dcoords))
    rdm_postcue[:,:,i] = squareform(pdist(delay2_3dcoords))
    
rdm_precue_averaged = np.mean(rdm_precue,-1)
rdm_postcue_averaged = np.mean(rdm_postcue,-1)

#%% plot
    
plt.figure(figsize=(10,20),num = 'Averaged RDMs')
plt.subplot(121)
plt.imshow(rdm_precue_averaged,cmap = sns.cubehelix_palette(start=.5,rot=-.75,as_cmap = True,reverse=True))
plt.colorbar()
plt.title('pre-cue')
plt.yticks([])
plt.xticks([])


plt.subplot(122)
plt.imshow(rdm_postcue_averaged,cmap = sns.cubehelix_palette(start=.5,rot=-.75,as_cmap = True,reverse=True))
plt.colorbar()
plt.title('post-cue')
plt.yticks([])
plt.xticks([])

#%% load model RDMs
path = base_path + 'RSA/model_RDMs_sqform.pckl'
f = open(path,'rb')
model_RDMs = pickle.load(f) # pre- and post-cue arrays stacked along axis 2
f.close()

path = base_path + 'RSA/model_RDMs_order.pckl'
f = open(path,'rb')
model_RDMs_order = pickle.load(f) # pre- and post-cue arrays stacked along axis 2
f.close()


path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/data_vonMises/MSELoss/with_fixation/kappa1.0/nrec200/lr0.005/'
save_path = path + 'RSA/'
theta_degrees =180
ortho_180_RDM = pickle.load(open(save_path 
                                + 'ortho_rotated_diagRDMs/'
                                + str(theta_degrees)+'.pckl','rb'))

# swap ortho for ortho_180
model_RDMs[:,0] = ortho_180_RDM.T

#%% regression analysis
from scipy.stats import zscore, stats, linregress
from numpy.linalg import lstsq, inv
import statsmodels.api as sm


model_RDMs = zscore(model_RDMs)
X = sm.add_constant(model_RDMs)

y_pre = squareform(rdm_precue_averaged)
y_post = squareform(rdm_postcue_averaged)


b_pre, res_pre, rank_pre, s_pre = lstsq(X, y_pre)
b_post, res_post, rank_post, s_post = lstsq(X, y_post)


results_pre = sm.OLS(y_pre, X).fit()
print('Regression results: pre-cue')
print(results_pre.summary())

results_post = sm.OLS(y_post, X).fit()
print('Regression results: post-cue')
print(results_post.summary())

#%% MDS visualisation - sanity check

from sklearn.manifold import MDS
from scipy.spatial.distance import squareform, pdist


def fit_mds_to_rdm(rdm):
    mds = MDS(n_components=3, 
              metric=True, 
              dissimilarity='precomputed', 
              max_iter=1000,
              random_state=0)
    return mds.fit_transform(rdm)


mds_pre = fit_mds_to_rdm(rdm_precue_averaged)
mds_post = fit_mds_to_rdm(rdm_postcue_averaged)


def plot_geometry(ax,data,pca,plot_colours,plot_outline = True,legend_on = True):
    
    n_colours = len(plot_colours)
    # plot the parallelogram defined by colours at location 1
    if plot_outline:
        ax.plot(np.append(data[:n_colours,0],data[0,0]),
              np.append(data[:n_colours,1],data[0,1]),
              np.append(data[:n_colours,2],data[0,2]),'k-')
    ax.scatter(data[0,0],data[0,1], data[0,2],marker='^',s = 40,
              c='k',label='loc1')
    ax.scatter(data[:n_colours,0],data[:n_colours,1],
              data[:n_colours,2],marker='^',s = 40,c=plot_colours)
  
    # repeat for loc 2
    if plot_outline:
        ax.plot(np.append(data[n_colours:,0],data[n_colours,0]),
              np.append(data[n_colours:,1],data[n_colours,1]),
              np.append(data[n_colours:,2],data[n_colours,2]),'k-')
    ax.scatter(data[-1,0],data[-1,1], data[-1,2],marker='s',s = 40,
              c='k',label='loc2')
    ax.scatter(data[n_colours:,0],data[n_colours:,1],
              data[n_colours:,2],marker='s',s = 40,c=plot_colours)
    
    if pca:
        ax.set_xlabel('PC1 ['+str(np.round(pca.explained_variance_ratio_[0]*100,1))+'%]')
        ax.set_ylabel('PC2 ['+str(np.round(pca.explained_variance_ratio_[1]*100,1))+'%]')
        ax.set_zlabel('PC3 ['+str(np.round(pca.explained_variance_ratio_[2]*100,1))+'%]')
    if legend_on:
        ax.legend()
#%%


plt.figure()
ax = plt.subplot(121,projection='3d')

colours = ['r','y','g','b']

plot_geometry(ax,mds_pre,[],colours)


ax2 = plt.subplot(122,projection='3d')
plot_geometry(ax2,mds_post,[],colours)


