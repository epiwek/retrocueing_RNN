#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 18:10:45 2021

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
    
    
#%% load data RDMs

pre_data_RDM_averaged = pickle.load(open(base_path+'RSA/pre_data_RDM_averaged.pckl','rb'))
post_data_RDM_averaged = pickle.load(open(base_path+'RSA/post_data_RDM_averaged.pckl','rb'))


#%% regression
from scipy.stats import zscore, stats, linregress
from numpy.linalg import lstsq, inv
import statsmodels.api as sm


def runRDMreg(model_RDM,data_RDM):
    if len(model_RDM.shape)>1:
        if model_RDM.shape[0]==model_RDM.shape[1]:
            # if not in diagonal form
            raise ValueError('Please enter the model RDM in diagonal form')
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

#%% load mirror flip RDM

model_RDM = pickle.load(open(base_path+'RSA/flipped_RDM_sqform.pckl','rb'))

results = runRDMreg(model_RDM,post_data_RDM_averaged)

print(results.summary())

