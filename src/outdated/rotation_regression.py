#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 12:48:51 2021

@author: emilia
"""

from sklearn.manifold import MDS
from scipy.spatial.distance import squareform, pdist

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import vec_operations as vops
import pickle

import pycircstat


#%%

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
                            'data_vonMises/MSELoss/with_fixation_longTrials/kappa1.0/nrec200/lr0.001/'

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
    
    
#%% load data RDMs

pre_data_RDM_averaged = pickle.load(open(base_path+'RSA/pre_data_RDM_averaged.pckl','rb'))
post_data_RDM_averaged = pickle.load(open(base_path+'RSA/post_data_RDM_averaged.pckl','rb'))

#%% load and do regression for each rotation
from scipy.stats import zscore, stats, linregress
from numpy.linalg import lstsq, inv
import statsmodels.api as sm


def runRDMreg(model_RDM,data_RDM):
    # z-score RDMs
    model_RDM = zscore(model_RDM)
    # create predictor matrix - add bias
    X = sm.add_constant(model_RDM)
    
    # extract diagonal from data RDM
    y = squareform(data_RDM)
    
    # run regression
    # b, res, rank, s = lstsq(X, y)
    results = sm.OLS(y, X).fit()
    
    return results



theta_range = pickle.load(open(base_path+'RSA/theta_range.pckl','rb'))

betas = np.zeros((len(theta_range),))
beta_CIs = np.zeros((len(theta_range),2))
p_vals = np.zeros((len(theta_range),))
for i, theta_degrees in enumerate(theta_range):
    # load RDM
    
    rot_RDM = pickle.load(open(base_path+'RSA/rotated_diagRDMs/'
                            + str(theta_degrees)+'.pckl','rb'))
    
    results = runRDMreg(rot_RDM,post_data_RDM_averaged)
    betas[i] = results.params[1] # beta for the rotated model
    
    CI = results.conf_int()
    beta_CIs[i,0] = CI[1,0] # lower 95%CI
    beta_CIs[i,1] = CI[1,1] # upper
    
    p_vals[i] = results.pvalues[1]
    

#%% plot
plt.rcParams.update({'font.size': 19})

plt.figure(figsize=(6,5))
plt.plot(theta_range,betas,'k-')
plt.fill_between(theta_range, beta_CIs[:,0],beta_CIs[:,1],alpha = 0.3,facecolor='k')

plt.plot(theta_range,theta_range*0,'k--')

plt.ylabel('Beta coefficient')
plt.xlabel('Rotation angle [°]')

plt.xlim([0,360])
plt.xticks(np.arange(0,361,60))

# plt.title('post-cue')
plt.tight_layout()

ylims = plt.ylim()
yticks = plt.yticks()

#%% repeat for pre-cue

betas = np.zeros((len(theta_range),))
beta_CIs = np.zeros((len(theta_range),2))
p_vals = np.zeros((len(theta_range),))
for i, theta_degrees in enumerate(theta_range):
    # load RDM
    
    rot_RDM = pickle.load(open(base_path+'RSA/rotated_diagRDMs/'
                            + str(theta_degrees)+'.pckl','rb'))
    
    results = runRDMreg(rot_RDM,pre_data_RDM_averaged)
    betas[i] = results.params[1] # beta for the rotated model
    
    CI = results.conf_int()
    beta_CIs[i,0] = CI[1,0] # lower 95%CI
    beta_CIs[i,1] = CI[1,1] # upper
    
    p_vals[i] = results.pvalues[1]


#%% ortho parallel

plt.figure(figsize=(10,10))
plt.plot(theta_range,betas,'k-')
plt.fill_between(theta_range, beta_CIs[:,0],beta_CIs[:,1],alpha = 0.3,facecolor='k')

plt.plot(theta_range,theta_range*0,'k--')

plt.ylabel('Beta coefficient')
plt.xlabel('Rotation angle [°]')

plt.xlim([0,360])
plt.xticks(np.arange(0,361,60))
plt.ylim(ylims)

# plt.title('pre-cue')

plt.tight_layout()
#%% rotated ortho



betas = np.zeros((len(theta_range),))
beta_CIs = np.zeros((len(theta_range),2))
p_vals = np.zeros((len(theta_range),))
for i, theta_degrees in enumerate(theta_range):
    # load RDM
    
    rot_RDM = pickle.load(open(base_path+'RSA/ortho_rotated_diagRDMs/'
                            + str(theta_degrees)+'.pckl','rb'))
    
    results = runRDMreg(rot_RDM,pre_data_RDM_averaged)
    betas[i] = results.params[1] # beta for the rotated model
    
    CI = results.conf_int()
    beta_CIs[i,0] = CI[1,0] # lower 95%CI
    beta_CIs[i,1] = CI[1,1] # upper
    
    p_vals[i] = results.pvalues[1]
    
#%%    
plt.figure(figsize=(6,5))
plt.plot(theta_range,betas,'k-')
plt.fill_between(theta_range, beta_CIs[:,0],beta_CIs[:,1],alpha = 0.3,facecolor='k')

plt.plot(theta_range,theta_range*0,'k--')

plt.ylabel('Beta coefficient')
plt.xlabel('Rotation angle [degrees]')

plt.xlim([0,360])
plt.xticks(np.arange(0,361,60))
plt.ylim(ylims)
# plt.yticks(yticks[0])


plt.tight_layout()

# plt.title('pre-cue - ortho')