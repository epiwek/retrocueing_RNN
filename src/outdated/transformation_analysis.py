#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 23:02:23 2021

@author: emilia
"""

# from scipy.spatial.distance import pdist, squareform
from scipy.stats import zscore, stats, linregress
from numpy.linalg import lstsq, inv
import statsmodels.api as sm

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
n_rec=200
#%%


plot_colours = ['r','y','g','b']
delay1_ix=5
delay2_ix=11

A_pre_cued_up = np.empty((n_rec,n_colours,n_models))
A_pre_cued_down = np.empty((n_rec,n_colours,n_models))
A_post_cued_up = np.empty((n_rec,n_colours,n_models))
A_post_cued_down = np.empty((n_rec,n_colours,n_models))

A_pre_uncued_up = np.empty((n_rec,n_colours,n_models))
A_pre_uncued_down = np.empty((n_rec,n_colours,n_models))
A_post_uncued_up = np.empty((n_rec,n_colours,n_models))
A_post_uncued_down = np.empty((n_rec,n_colours,n_models))

X_cued_up = np.empty((n_rec,n_rec,n_models))
X_cued_down = np.empty((n_rec,n_rec,n_models))

X_uncued_up = np.empty((n_rec,n_rec,n_models))
X_uncued_down = np.empty((n_rec,n_rec,n_models))
#%%
for i,model_number in enumerate(converged):
# for model_number in np.arange(1):
    model_number = str(model_number)
    
    # load pca data
    f = open(load_path+'/eval_data_model' + model_number + '.pckl', 'rb')
    obj = pickle.load(f)    
    data = obj["data"]
    f.close()
        
    n_trials = data.shape[0]
    
    labels_uncued = np.concatenate((obj["labels"]["c2"][:n_trials//2],
                             obj["labels"]["c1"][n_trials//2:]))
    labels_cued = np.concatenate((obj["labels"]["c1"][:n_trials//2],
                             obj["labels"]["c2"][n_trials//2:]))
    
    labels_uncued_binned = helpers.bin_labels(labels_uncued,n_colours)
    labels_cued_binned = helpers.bin_labels(labels_cued,n_colours)
    
    delay1 = data[:,delay1_ix,:]
    delay2 = data[:,delay2_ix,:]
     
    # bin and average the data
    delay1_binned_uncued = np.zeros((n_colours*2,n_rec))
    delay2_binned_uncued = np.zeros((n_colours*2,n_rec))
    
    for colour in range(n_colours):
        ix_cued = np.where(labels_cued_binned==colour)[0]
        ix_uncued = np.where(labels_uncued_binned==colour)[0]

        ix_uncued_up = ix_uncued[np.where(ix_uncued>=n_trials//2)[0]]
        ix_uncued_down = ix_uncued[np.where(ix_uncued<n_trials//2)[0]]
        
        A_pre_uncued_up[:,colour,i] = torch.mean(delay1[ix_uncued_up,:],0)
        A_pre_uncued_down[:,colour,i] = torch.mean(delay1[ix_uncued_down,:],0)
        
        A_post_uncued_up[:,colour,i] = torch.mean(delay2[ix_uncued_up,:],0)
        A_post_uncued_down[:,colour,i] = torch.mean(delay2[ix_uncued_down,:],0)
    
    
    #% get the cued-averaged data
    f = open(load_path+'/pca_data_model' + model_number + '.pckl', 'rb')
    obj = pickle.load(f)    
    [trial_data,delay1_binned_cued,delay2_binned_cued] = obj
    f.close()
    
    A_pre_cued_up[:,:,i] = delay1_binned_cued[:n_colours,:].T
    A_pre_cued_down[:,:,i] = delay1_binned_cued[n_colours:,:].T
    A_post_cued_up[:,:,i] = delay2_binned_cued[:n_colours,:].T
    A_post_cued_down[:,:,i] = delay2_binned_cued[n_colours:,:].T
    
    # run regression to estimate Xs
    
    # X = sm.add_constant(model_RDMs)
    
    # A_pcu = sm.add_constant(A_pre_cued_up[:,:,i]).T
    

    X_cued_up[:,:,i], res, rank, s = lstsq(A_pre_cued_up[:,:,i].T,A_post_cued_up[:,:,i].T)
    X_cued_down[:,:,i], res, rank, s = lstsq(A_pre_cued_down[:,:,i].T,A_post_cued_down[:,:,i].T)
    
    X_uncued_up[:,:,i], res, rank, s = lstsq(A_pre_uncued_up[:,:,i].T,A_post_uncued_up[:,:,i].T)
    X_uncued_down[:,:,i], res, rank, s = lstsq(A_pre_uncued_down[:,:,i].T,A_post_uncued_down[:,:,i].T)
    
#%% test cross-generalisation

recon_error = np.empty((4,4,n_models))
for i in range(n_models):
    # same data for validation
    res = A_post_cued_up[:,:,i] - (X_cued_up[:,:,i] @ A_pre_cued_up[:,:,i])
    recon_error[0,0,i] = np.mean([np.linalg.norm(res[:,i]) for i in range(4)])
    
    # same condition but other item
    res = A_post_cued_up[:,:,i] - (X_uncued_down[:,:,i] @ A_pre_cued_up[:,:,i])
    recon_error[1,0,i] = np.mean([np.linalg.norm(res[:,i]) for i in range(4)])
    
    # different condition but same item
    res = A_post_cued_up[:,:,i] - (X_uncued_up[:,:,i] @ A_pre_cued_up[:,:,i])
    recon_error[2,0,i] = np.mean([np.linalg.norm(res[:,i]) for i in range(4)])
    
    # different condition and other item
    res = A_post_cued_up[:,:,i] - (X_uncued_down[:,:,i] @ A_pre_cued_up[:,:,i])
    recon_error[3,0,i] = np.mean([np.linalg.norm(res[:,i]) for i in range(4)])
    
    
    ###
    # same data for validation
    res = A_post_cued_down[:,:,i] - (X_cued_down[:,:,i] @ A_pre_cued_down[:,:,i])
    recon_error[0,1,i] = np.mean([np.linalg.norm(res[:,i]) for i in range(4)])
    
    # same condition but other item
    res = A_post_cued_down[:,:,i] - (X_uncued_up[:,:,i] @ A_pre_cued_down[:,:,i])
    recon_error[1,1,i] = np.mean([np.linalg.norm(res[:,i]) for i in range(4)])
    
    # different condition but same item
    res = A_post_cued_down[:,:,i] - (X_uncued_down[:,:,i] @ A_pre_cued_down[:,:,i])
    recon_error[2,1,i] = np.mean([np.linalg.norm(res[:,i]) for i in range(4)])
    
    # different condition and item
    res = A_post_cued_down[:,:,i] - (X_cued_up[:,:,i] @ A_pre_cued_down[:,:,i])
    recon_error[3,1,i] = np.mean([np.linalg.norm(res[:,i]) for i in range(4)])
    
    
    ###
    # same data for validation
    res = A_post_uncued_up[:,:,i] - (X_uncued_up[:,:,i] @ A_pre_uncued_up[:,:,i])
    recon_error[0,2,i] = np.mean([np.linalg.norm(res[:,i]) for i in range(4)])
    
    # same condition but other item
    res = A_post_uncued_up[:,:,i] - (X_cued_down[:,:,i] @ A_pre_uncued_up[:,:,i])
    recon_error[1,2,i] = np.mean([np.linalg.norm(res[:,i]) for i in range(4)])
    
    # different condition but same item
    res = A_post_uncued_up[:,:,i] - (X_cued_up[:,:,i] @ A_pre_uncued_up[:,:,i])
    recon_error[2,2,i] = np.mean([np.linalg.norm(res[:,i]) for i in range(4)])
    
    # different condition and other item
    res = A_post_uncued_up[:,:,i] - (X_uncued_down[:,:,i] @ A_pre_uncued_up[:,:,i])
    recon_error[3,2,i] = np.mean([np.linalg.norm(res[:,i]) for i in range(4)])
    
    
    ###
    # same data for validation
    res = A_post_uncued_down[:,:,i] - (X_uncued_down[:,:,i] @ A_pre_uncued_down[:,:,i])
    recon_error[0,3,i] = np.mean([np.linalg.norm(res[:,i]) for i in range(4)])
    
    # same condition but other item
    res = A_post_uncued_down[:,:,i] - (X_cued_up[:,:,i] @ A_pre_uncued_down[:,:,i])
    recon_error[1,3,i] = np.mean([np.linalg.norm(res[:,i]) for i in range(4)])
    
    # different condition but same item
    res = A_post_uncued_down[:,:,i] - (X_cued_down[:,:,i] @ A_pre_uncued_down[:,:,i])
    recon_error[2,3,i] = np.mean([np.linalg.norm(res[:,i]) for i in range(4)])
    
    # different condition and other item
    res = A_post_uncued_down[:,:,i] - (X_uncued_up[:,:,i] @ A_pre_uncued_down[:,:,i])
    recon_error[3,3,i] = np.mean([np.linalg.norm(res[:,i]) for i in range(4)])
    
    
#%% plot

plt.figure()

for e in range(4):
    plt.plot(np.ones((n_models,))*e,np.mean(recon_error[e,:,:],0),'o')



sem = np.std(np.mean(recon_error,1),1)/np.sqrt(n_models)
means = np.mean(np.mean(recon_error,1),1)

plt.figure()
plt.plot(range(4),means,'ro-')
plt.plot(range(4),means+sem,'ko-')
plt.plot(range(4),means-sem,'ko-')

#%% make into a table
import pandas
df = pandas.DataFrame(data= np.mean(recon_error,1).T,columns = \
                      ['same_cond_same_item','same_cond_diff_item',
                       'diff_cond_same_item','diff_cond_diff_item'])

    
df.to_csv(load_path+'/recon_errors.csv')

