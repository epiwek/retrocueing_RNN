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

#%%

# load hidden activations

model_type = 'RNN'
# data_type = '1-hot'
data_type = 'gaussian'

    
if (model_type == 'RNN'):
    # load_path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/data/pca_data_sameTrainingData'
    #load_path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/data/pca_data_sameTrainingInit'
    # load_path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/data_1hot/pca_data'
    # load_path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/data_gaussian/with_fixation/nrec300/lr0.005/pca_data'
    # base_path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/data_gaussian_hardcoded/nrec300/lr0.005/'
    # base_path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/data_vonMises/kappa20.0/nrec200/lr0.005/'
    base_path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/'+\
                            'data_vonMises/MSELoss/with_fixation/kappa1.0/nrec200/lr0.005/'

    #forAndrew/Xavier_200rec/lr0.005/fullGD/pca_data'
elif (model_type == 'LSTM'):
    raise NotImplementedError()
    # load_path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/LSTM_data/pca_data/'
    #load_path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/LSTM_data_1hot/pca_data'
else :
    print('Invalid model type')

load_path = base_path + 'pca_data'
f = open(load_path+'/converged.pckl','rb')
converged = pickle.load(f)
f.close()
# converged = np.arange(10)

n_models = len(converged)

# base_path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/data_1hot/'
# base_path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/data_gaussian/with_fixation/nrec300/lr0.005/'


#forAndrew/Xavier_200rec/lr0.005/fullGD/'
f = open(base_path+'saved_models/model_params.pckl','rb')
obj = pickle.load(f)
if data_type == 'gaussian':
    [n_inp,n_rec,n_iter,batch_size,learning_rate,fixation] = obj
    n_colCh = (n_inp - 2 - int(fixation))//2
else:
    [n_inp,n_rec,n_iter,batch_size,learning_rate] = obj

#%% load model RDMs
f = open(base_path+'RSA/indiv_model_rdms.pckl','rb')
[rdm_precue_binned,rdm_postcue_binned] = pickle.load(f)
f.close()


#%% load model RDMs


# load model RDMs
path = base_path + 'RSA/model_RDMs_sqform.pckl'
f = open(path,'rb')
model_RDMs = pickle.load(f) # individual predictor arrays stacked along axis 1
f.close()

path = base_path + 'RSA/model_RDMs_order.pckl'
f = open(path,'rb')
model_RDMs_order = pickle.load(f) # individual predictor arrays stacked along axis 1
f.close()




path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/data_vonMises/MSELoss/with_fixation/kappa1.0/nrec200/lr0.005/'
save_path = path + 'RSA/'
theta_degrees =180
ortho_180_RDM = pickle.load(open(save_path 
                                + 'ortho_rotated_diagRDMs/'
                                + str(theta_degrees)+'.pckl','rb'))

# swap ortho for ortho_180
model_RDMs[:,0] = ortho_180_RDM.T

#%% regression

from scipy.stats import zscore, stats, linregress
from numpy.linalg import lstsq, inv
import statsmodels.api as sm


model_RDMs = zscore(model_RDMs)
X = sm.add_constant(model_RDMs)

betas = {"pre-cue":np.zeros((4,n_models)),"post-cue":np.zeros((4,n_models))}
r2 = {"pre-cue":np.zeros((n_models,)),"post-cue":np.zeros((n_models,))}
CIs = {"pre-cue":np.zeros((4,2,n_models)),"post-cue":np.zeros((4,2,n_models))} 

CIs_magnitude = {"pre-cue":np.zeros((4,n_models)),"post-cue":np.zeros((4,n_models))} 


for m in range(n_models):
    y_pre = squareform(rdm_precue_binned[:,:,m])
    y_post = squareform(rdm_postcue_binned[:,:,m])

    
    results_pre = sm.OLS(y_pre, X).fit()
    results_post = sm.OLS(y_post, X).fit()
    
    betas["pre-cue"][:,m] = results_pre.params
    betas["post-cue"][:,m] = results_post.params
    
    CIs["pre-cue"][:,:,m] = results_pre.conf_int()
    CIs["post-cue"][:,:,m] = results_post.conf_int()
    r2["pre-cue"][m] = results_pre.rsquared
    r2["post-cue"][m] = results_post.rsquared
    
    CIs_magnitude["pre-cue"][:,m] = np.abs(CIs["pre-cue"][:,1,0]-CIs["pre-cue"][:,0,0])
    CIs_magnitude["post-cue"][:,m] = np.abs(CIs["post-cue"][:,1,0]-CIs["post-cue"][:,0,0])

#%% plot betas
plt.rcParams.update({'font.size': 15})

periods = ['pre-cue','post-cue']

for p in range(len(periods)):
    plt.figure()
    for b in range(len(model_RDMs_order)):
        plt.subplot(3,1,b+1)
        plt.errorbar(range(n_models),betas[periods[p]][b+1,:],yerr=CIs_magnitude[periods[p]][b+1,:])
        xlims = plt.xlim()
        plt.plot(xlims, [0,0], 'k--')
        plt.ylabel('Beta coefficient')
        plt.xlabel('model number')
        
        plt.title(model_RDMs_order[b]+' ' + periods[p])
    plt.tight_layout()

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
    
    
#%% plot R2

import seaborn as sns

pal = sns.color_palette("dark")
inds = [3,0]
cols = [pal[ix] for ix in inds]


ms = 16

xs = np.zeros((n_models,2))
xs[:,1] += 1
ys = np.array([r2["pre-cue"],r2["post-cue"]]).T

plt.figure()
ax = plt.subplot(111)


plot_paired_data(xs,ys,ax,cols,jitter='auto',alpha=.2,markersize=ms)
plot_paired_data([0,1],np.mean(ys,0),ax,cols,markersize=ms)

plt.xlim([-.2,1.2])

plt.xticks([0,1],periods)
plt.ylabel('R${^2}$')





    
