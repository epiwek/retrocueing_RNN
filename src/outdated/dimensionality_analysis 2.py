#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 16:04:48 2021

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


#%% plotting funcs

def add_jitter(data,jitter):
    # check that data is a vector
    if len(data.shape)>1:
        raise ValueError('Input data should be 1d')
    
    unique = np.unique(data)
    x_jitter = np.zeros(data.shape)
    # check for duplicate vals
    if len(unique) != len(data) :
        # there are duplicates
        for i in range(len(unique)):
            # save indices of duplicates
            dups_ix = np.where(data==unique[i])[0]
            n_dups = len(dups_ix)
            if jitter=='auto':
                x_jitter[dups_ix] = np.linspace(-.2,.2,n_dups)
            else:
                # custom jitter value
                x_jitter[dups_ix] = \
                    np.arange(0,n_dups*jitter,jitter) \
                    - np.mean(np.arange(0,n_dups*jitter,jitter))
    return x_jitter

def plot_paired_data(x_data,y_data,ax,colours,jitter='auto',**kwargs):
    # determine x jitter
    x_jitter = np.zeros(y_data.shape)
    if len(y_data.shape) == 1:
        x_jitter = np.expand_dims(x_jitter,0)
        y_data = np.expand_dims(y_data,0)
        x_data = np.expand_dims(x_data,0)
    
    for i in range(2):
        x_jitter[:,i] = add_jitter(y_data[:,i],jitter=jitter)
    
    #plot
    for i in range(y_data.shape[0]):
        ax.plot(x_data[i,:]+x_jitter[i,:],y_data[i,:],'k-',**kwargs)
        for j in range(y_data.shape[1]):
            ax.plot(x_data[i,j]+x_jitter[i,j],y_data[i,j],'o',
                    color = colours[j],**kwargs)
    
    

#%%#%% load model from file

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
n_models = len(converged)

n_colours = 4
#%% run pca - separately for each location and delay
# to check how much variance the first 2 PCs explain (i.e. if there are location-specific **planes*)

PVEs = np.zeros((n_models,3,2)) # model x [up, down, mean] x [pre,post]
# for i,model_number in enumerate(converged):
for i,model_number in enumerate(converged):
    model_number = str(model_number)
    
    # load pca data
    # f = open(load_path+'/eval_data_model' + model_number + '.pckl', 'rb')
    # obj = pickle.load(f)    
    # # [trial_data,delay1,delay2] = obj
    
    f = open(load_path+'/pca_data_model' + model_number + '.pckl', 'rb')
    obj = pickle.load(f)    
    [trial_data,delay1,delay2] = obj
    
    # data = obj["data"]
    # d1_ix = 5
    # d2_ix = 11
    # delay1 = data[:,d1_ix,:]
    # delay2 = data[:,d2_ix,:]
    
    n_colours = delay1.shape[0]//2
    
    f.close()
    #% run PCA
    
    delay1up = delay1[:n_colours,:]
    delay1down = delay1[n_colours:,:]
    delay2up = delay2[:n_colours,:]
    delay2down = delay2[n_colours:,:]
    
    delay1_up_model = PCA(n_components=2) # Initializes PCA
    delay1_down_model = PCA(n_components=2) # Initializes PCA
    delay2_up_model= PCA(n_components=2) # Initializes PCA
    delay2_down_model = PCA(n_components=2) # Initializes PCA
    
    delay1up -= torch.mean(delay1up)
    delay1down -= torch.mean(delay1down)
    delay2up -= torch.mean(delay2up)
    delay2down -= torch.mean(delay2down)
    
    
    # run PCA
    delay1_up = delay1_up_model.fit(delay1up) # get coordinates in the reduced-dim space
    delay1_down = delay1_down_model.fit(delay1down)
    delay2_up = delay2_up_model.fit(delay2up)
    delay2_down = delay2_down_model.fit(delay2down)
    
    PVEs[i,0,0] = np.sum(delay1_up.explained_variance_ratio_) # up pre
    PVEs[i,1,0] = np.sum(delay1_down.explained_variance_ratio_) # down pre
    PVEs[i,2,0] = np.mean(PVEs[i,:2,0]) # mean pre
    
    PVEs[i,0,1] = np.sum(delay2_up.explained_variance_ratio_) # up post
    PVEs[i,1,1] = np.sum(delay2_down.explained_variance_ratio_) # down post
    PVEs[i,2,1] = np.mean(PVEs[i,:2,1]) # mean post

PVEs = PVEs*100 # convert to %   
#%% plot
import matplotlib
matplotlib.rcParams.update({'font.size': 30})
import seaborn as sns

pal = sns.color_palette("dark")
inds = [3,0]
cols = [pal[ix] for ix in inds]


ms = 16
plt.figure(figsize=(10,10))


for m in range(n_models): 
    plt.plot(['pre','post'],PVEs[m,2,:],'k-',alpha=0.2)
    plt.plot(['pre'],PVEs[m,2,0],'o',color = cols[0],markersize=ms,alpha=.2)
    plt.plot(['post'],PVEs[m,2,1],'o',color = cols[1],markersize=ms,alpha=.2)


CI = 1.98 * np.std(PVEs[:,2,:],0)/np.sqrt(n_models)
plt.errorbar(['pre','post'],np.mean(PVEs[:,2,:],0),yerr = CI,c='k')
plt.plot(['pre'],np.mean(PVEs[:,2,0]),'-o',color = cols[0],markersize = ms)  
plt.plot(['post'],np.mean(PVEs[:,2,1]),'-o',color = cols[1],markersize = ms)

# plt.bar(['pre'],np.mean(PVEs[:,2,0]),color = cols[0],alpha=0.2)  
# plt.bar(['post'],np.mean(PVEs[:,2,1]),color = cols[1],alpha=0.2)

plt.xlim([-.2,1.2])
plt.ylim([95,100])
plt.ylabel('Mean PVE [%]')

plt.xlabel('Phantom',c='w')

plt.tight_layout()


#%% do stats

from scipy.stats import shapiro, ttest_1samp, wilcoxon


sw_pre, p_pre = shapiro(PVEs[:,2,0])
sw_post, p_post = shapiro(PVEs[:,2,1])


if np.logical_or(p_pre<0.05,p_post<0.05):
    print('    wilcoxon test')
    stat, p_val = wilcoxon(PVEs[:,2,1]-PVEs[:,2,0])
else:
    print('    one-sample t-test')
    stat, p_val = ttest_1samp(PVEs[:,2,1]-PVEs[:,2,0],0)
print('        stat = %.3f, p = %.3f' %(stat,p_val))


#%% run pca - separately for each delay to estimate dimensionality of representations
# get number of components that explains 95% of variance in data

PVEs = np.zeros((n_models,3,2)) # model x [up, down, mean] x [pre,post]
n_dims95 = np.zeros((n_models,2)) # model x [pre, post]

for i,model_number in enumerate(converged):
# for model_number in range(1):
    model_number = str(model_number)
    
    # load pca data
    f = open(load_path+'/rdm_data_model' + model_number + '.pckl', 'rb')
    # obj =   pickle.load(f)  
    # data = obj["data"]
    data = pickle.load(f) 
    # [trial_data,delay1,delay2]=  pickle.load(f)  
    
    f.close()
    #% run PCA
    

    delay1 = data[:,1,:]
    delay2 = data[:,3,:]
    delay1_model = PCA(n_components=.95,svd_solver='full') # Initializes PCA
    delay2_model= PCA(n_components=.95,svd_solver='full') # Initializes PCA
    
    # delay1_model = PCA(n_components=3) # Initializes PCA
    # delay2_model= PCA(n_components=3) # Initializes PCA
    
    delay1 -= torch.mean(delay1)
    delay2 -= torch.mean(delay2)    
    
    # run PCA
    delay1_pca = delay1_model.fit_transform(delay1) # get coordinates in the reduced-dim space
    delay2_pca = delay2_model.fit_transform(delay2)
    
    n_dims95[i,0] = delay1_model.components_.shape[0]
    n_dims95[i,1] = delay2_model.components_.shape[0]
    # PVEs[i,0,0] = np.sum(delay1_up.explained_variance_ratio_) # up pre
    # PVEs[i,1,0] = np.sum(delay1_down.explained_variance_ratio_) # down pre
    # PVEs[i,2,0] = np.mean(PVEs[i,:2,0]) # mean pre
    
    # PVEs[i,0,1] = np.sum(delay2_up.explained_variance_ratio_) # up post
    # PVEs[i,1,1] = np.sum(delay2_down.explained_variance_ratio_) # down post
    # PVEs[i,2,1] = np.mean(PVEs[i,:2,1]) # mean post

#%% plot

import matplotlib
matplotlib.rcParams.update({'font.size': 30})
import seaborn as sns

pal = sns.color_palette("dark")
inds = [3,0]
cols = [pal[ix] for ix in inds]


ms = 16
plt.figure(figsize=(10,10))
ax = plt.subplot(111)

xs = np.zeros(n_dims95.shape)
xs[:,1] +=1
plot_paired_data(xs,n_dims95,ax,cols,jitter=0.01,alpha=.2)

#% add medians
medians = np.median(n_dims95,0)
plot_paired_data(range(2),medians,ax,cols,markersize=10)

ax.set_xticks([0,1])
ax.set_xticklabels(['pre','post'])
ax.set_xlim([-.2,1.2])

ax.set_ylabel('N dims')
ax.set_xlabel('N dims',c='w')



#%%
for m in range(n_models): 
    plt.plot(['pre','post'],n_dims95[m,:],'k-',alpha=0.2)
    plt.plot(['pre'],n_dims95[m,0],'o',color = cols[0],markersize=ms,alpha=.2)
    plt.plot(['post'],n_dims95[m,1],'o',color = cols[1],markersize=ms,alpha=.2)


# CI = 1.98 * np.std(PVEs[:,2,:],0)/np.sqrt(n_models)
# plt.errorbar(['pre','post'],np.mean(PVEs[:,2,:],0),yerr = CI,c='k')
# plt.plot(['pre'],np.mean(PVEs[:,2,0]),'-o',color = cols[0],markersize = ms)  
# plt.plot(['post'],np.mean(PVEs[:,2,1]),'-o',color = cols[1],markersize = ms)

# plt.bar(['pre'],np.mean(PVEs[:,2,0]),color = cols[0],alpha=0.2)  
# plt.bar(['post'],np.mean(PVEs[:,2,1]),color = cols[1],alpha=0.2)

# plt.xlim([-.2,1.2])
# plt.ylabel('Mean PVE [%]')

# plt.xlabel('Phantom',c='w')

# plt.tight_layout()


#%% run pca on each delay separately
# to check if the points can be embedded in a 3D space (i.e. how much variance the first PCs explain)

PVEs = np.zeros((n_models,2)) # model x [pre,post]
# for i,model_number in enumerate(converged):
for i,model_number in enumerate(converged):
    model_number = str(model_number)
    
    # load pca data
    f = open(load_path+'/eval_data_model' + model_number + '.pckl', 'rb')
    obj = pickle.load(f)    
    # [trial_data,delay1,delay2] = obj
    
    data = obj["data"]
    
    
    # data = obj
    delay1 = data[:,1,:]
    delay2 = data[:,3,:]
    
    n_colours = delay1.shape[0]//2
    
    f.close()
    #% run PCA
    
    delay1_model = PCA(n_components=3) # Initializes PCA
    delay2_model= PCA(n_components=3) # Initializes PCA
    
    delay1 -= torch.mean(delay1)
    delay2 -= torch.mean(delay2)
    
    
    # run PCA
    delay1_model.fit(delay1) # get coordinates in the reduced-dim space
    delay2_model.fit(delay2)
    
    PVEs[i,0] = np.sum(delay1_model.explained_variance_ratio_) # up pre
    PVEs[i,1] = np.sum(delay2_model.explained_variance_ratio_) # down pre
    

    
#%% plot
import matplotlib
matplotlib.rcParams.update({'font.size': 30})
import seaborn as sns

pal = sns.color_palette("dark")
inds = [3,0]
cols = [pal[ix] for ix in inds]


ms = 10
plt.figure(figsize=(10,10))
ax = plt.subplot(111)

plot_paired_data(xs,PVEs,ax,cols,alpha=.2,markersize=ms)


# for m in range(n_models): 
#     plt.plot(['pre','post'],PVEs[m,2,:],'k-',alpha=0.2)
#     plt.plot(['pre'],PVEs[m,2,0],'o',color = cols[0],markersize=ms,alpha=.2)
#     plt.plot(['post'],PVEs[m,2,1],'o',color = cols[1],markersize=ms,alpha=.2)


CI = 1.98 * np.std(PVEs,0)/np.sqrt(n_models)
plt.errorbar(range(2),np.mean(PVEs,0),yerr = CI,c='k')
plt.plot(0,np.mean(PVEs[:,0]),'-o',color = cols[0],markersize = ms)  
plt.plot(1,np.mean(PVEs[:,1]),'-o',color = cols[1],markersize = ms)

# # plt.bar(['pre'],np.mean(PVEs[:,2,0]),color = cols[0],alpha=0.2)  
# # plt.bar(['post'],np.mean(PVEs[:,2,1]),color = cols[1],alpha=0.2)

plt.ylabel('Total PVE by PC1-3 [%]')

plt.ylim([.7,1])


ax.set_xticks([0,1])
ax.set_xticklabels(['pre','post'])
ax.set_xlim([-.2,1.2])
plt.xlabel('Phantom',c='w')

plt.tight_layout()


#%% DO IT FOR UNCUED SUBSPACES
import helpers
import torch
#% run pca - separately for each location and delay
# to check how much variance the first 2 PCs explain (i.e. if there are location-specific **planes*)

PVEs = np.zeros((n_models,3,2)) # model x [up, down, mean] x [pre,post]
# for i,model_number in enumerate(converged):
for i,model_number in enumerate(converged):
    model_number = str(model_number)
    
    # load pca data
    f = open(load_path+'/eval_data_model' + model_number + '.pckl', 'rb')
    obj = pickle.load(f)    
    # [trial_data,delay1,delay2] = obj
    
    data = obj["data"]
    
    delay1 = data[:,1,:]
    delay2 = data[:,3,:]
    n_trials = data.shape[0]
    
    n_colours = 4
    n_rec = data.shape[-1]
    
    f.close()
    
    # bin and average the data
    labels_uncued = np.concatenate((obj["labels"]["c2"][:n_trials//2],
                             obj["labels"]["c1"][n_trials//2:]))
    
    labels_uncued_binned = helpers.bin_labels(labels_uncued,n_colours)
    
    delay1down = np.zeros((n_colours,n_rec)) # c2 uncued
    delay1up = np.zeros((n_colours,n_rec)) # c1 uncued
    delay2down = np.zeros((n_colours,n_rec))
    delay2up = np.zeros((n_colours,n_rec))
    
    for colour in range(n_colours):
        ix = np.where(labels_uncued_binned==colour)[0]
        ix_down = ix[np.where(ix<n_trials//2)[0]]
        ix_up = ix[np.where(ix>=n_trials//2)[0]]
        
        delay1down[colour,:] = torch.mean(delay1[ix_down,:],0)
        delay1up[colour,:] = torch.mean(delay1[ix_up,:],0)
        
        delay2down[colour,:] = torch.mean(delay2[ix_down,:],0)
        delay2up[colour,:] = torch.mean(delay2[ix_up,:],0)
        
        
    #% run PCA

    
    delay1_up_model = PCA(n_components=2) # Initializes PCA
    delay1_down_model = PCA(n_components=2) # Initializes PCA
    delay2_up_model= PCA(n_components=2) # Initializes PCA
    delay2_down_model = PCA(n_components=2) # Initializes PCA

    delay1up -= np.mean(delay1up)
    delay1down -= np.mean(delay1down)
    delay2up -= np.mean(delay2up)
    delay2down -= np.mean(delay2down)
    
    
    # # run PCA
    delay1_up_model.fit(delay1up) # get coordinates in the reduced-dim space
    delay1_down_model.fit(delay1down)
    delay2_up_model.fit(delay2up)
    delay2_down_model.fit(delay2down)
    
    PVEs[i,0,0] = np.sum(delay1_up_model.explained_variance_ratio_) # up pre
    PVEs[i,1,0] = np.sum(delay1_down_model.explained_variance_ratio_) # down pre
    PVEs[i,2,0] = np.mean(PVEs[i,:2,0]) # mean pre
    
    PVEs[i,0,1] = np.sum(delay2_up_model.explained_variance_ratio_) # up post
    PVEs[i,1,1] = np.sum(delay2_down_model.explained_variance_ratio_) # down post
    PVEs[i,2,1] = np.mean(PVEs[i,:2,1]) # mean post


#%% plot
plt.rcParams.update({'font.size': 30})
import seaborn as sns
import custom_plot as cplt 


pal = sns.color_palette("dark")
inds = [3,0]
cols = [pal[ix] for ix in inds]


ms = 16
plt.figure(figsize=(10,10))
ax = plt.subplot(111)

xs = np.zeros((n_models,2))
xs[:,1] = 1

cplt.plot_paired_data(xs, PVEs[:,-1,:]*100, ax, cols, markersize=ms,alpha=.2)
# cplt.plot_paired_data([0,1], np.mean(PVEs[:,-1,:],0), ax, cols)
CI = 100* 1.98 * np.std(PVEs[:,2,:],0)/np.sqrt(n_models)
plt.errorbar([0,1],np.mean(PVEs[:,-1,:],0)*100,yerr = CI,c='k')

# for m in range(n_models): 
#     plt.plot(['pre','post'],PVEs[m,2,:],'k-',alpha=0.2)
#     plt.plot(['pre'],PVEs[m,2,0],'o',color = cols[0],markersize=ms,alpha=.2)
#     plt.plot(['post'],PVEs[m,2,1],'o',color = cols[1],markersize=ms,alpha=.2)



# plt.plot(['pre'],np.mean(PVEs[:,2,0]),'-o',color = cols[0],markersize = ms)  
# plt.plot(['post'],np.mean(PVEs[:,2,1]),'-o',color = cols[1],markersize = ms)

# plt.bar(['pre'],np.mean(PVEs[:,2,0]),color = cols[0],alpha=0.2)  
# plt.bar(['post'],np.mean(PVEs[:,2,1]),color = cols[1],alpha=0.2)

plt.xlim([-.2,1.2])
plt.ylim([98.5,100])
plt.ylabel('Mean PVE [%]')
plt.xticks([0,1],['pre-cue','post-cue'])
plt.xlabel('Phantom',c='w')

plt.tight_layout()
