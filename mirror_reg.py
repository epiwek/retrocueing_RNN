#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 19:49:11 2021

@author: emilia
"""

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


mirror_RDM = pickle.load(open(base_path+'RSA/flipped_RDM_sqform.pckl','rb'))

results = runRDMreg(mirror_RDM,post_data_RDM_averaged)

betas = results.params[1] # beta for the rotated model
   
CI = results.conf_int()
p_vals = results.pvalues[1]
