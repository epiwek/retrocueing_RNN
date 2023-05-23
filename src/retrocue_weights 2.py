#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 10:49:43 2021

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
from define_model import define_model


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
    
    base_path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/'+\
                            'data_vonMises/MSELoss/with_fixation_longTrials/kappa1.0/nrec200/lr0.001/'



elif (model_type == 'LSTM'):
    load_path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/LSTM_data/pca_data'
    #load_path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/LSTM_data_1hot/pca_data'
else :
    print('Invalid model type')


load_path = base_path + 'pca_data'
f = open(load_path+'/converged.pckl','rb')
converged = pickle.load(f)
f.close()
n_colours=4
n_models = len(converged)


load_path = base_path + 'saved_models/'
f = open(load_path+'model_params.pckl', 'rb')
obj = pickle.load(f)

[n_inp,n_rec,n_iter,batch_size,learning_rate,fixation] = obj
    # n_colCh = (n_inp - 2 - int(fixation))//2
n_colCh = (n_inp - 2 - int(fixation))//2


# load task params    
f = open(load_path+'task_params.pckl', 'rb')
obj = pickle.load(f)
[n_stim,trial_timings] = obj
f.close()
#%%

Win = []
Wrec = []

Win0 = []
Wrec0 = []


for i,model_number in enumerate(converged):
    # for model_number in range(n_models):
    print('Model %d:' %model_number)
    #% load model
    model, RNN = define_model(n_inp,n_rec,n_stim)
    
    Win0.append(model.Wrec.weight_ih_l0.detach().T)
    Wrec0.append(model.Wrec.weight_hh_l0.detach().T)
    
    model = torch.load(load_path+'model'+str(model_number))
    print('.... Loaded')
     
    #% save weights
    Win.append(model.Wrec.weight_ih_l0.detach().T)
    Wrec.append(model.Wrec.weight_hh_l0.detach())

Win = torch.stack(Win)    
Wrec = torch.stack(Wrec)

Win0 = torch.stack(Win0)    
Wrec0 = torch.stack(Wrec0)

#%% plot the weights for an example model

plt.figure()
plt.imshow(Win[0,:2,:],aspect=20,cmap='vlag')
plt.colorbar()

plt.yticks([0,1])
plt.ylabel('retrocue')

plt.xlabel('neuron')

plt.title('Model 0')

#%% calculate the correlation coefficient for each model

from scipy.stats import shapiro, pearsonr, spearmanr, wilcoxon, ttest_1samp

r_coef = np.empty((n_models,))
p_val = np.empty((n_models,))
test = np.empty((n_models,))
for i,model_number in enumerate(converged):    
    
    sw, p1 = shapiro(Win[i,0,:])
    sw, p2 = shapiro(Win[i,1,:])
    
    # correlation coef
    if np.logical_or(p1<=.05,p2<=.05):
        r_coef[i],p_val[i] = spearmanr(Win[i,0,:],Win[i,1,:])
        test[i] = 0 # spearman
    else:
        r_coef[i],p_val[i] = pearsonr(Win[i,0,:],Win[i,1,:])
        test[i] = 1 # pearson

#%% do stats

sw, p = shapiro(r_coef)


if p<0.05:
    print('    wilcoxon test')
    stat, p_val = wilcoxon(r_coef)
else:
    print('    one-sample t-test')
    stat, p_val = ttest_1samp(r_coef)
print('        stat = %.3f, p = %.3f' %(stat,p_val))
#%%
# permutation tests

n_draws = 1000
r_coef_perm = np.empty((n_models,n_draws))
for n in range(n_draws):    
    for m in range(n_models):
        perm_ix = np.random.permutation(range(n_rec))
        sw, p1 = shapiro(Win[i,0,perm_ix])
        sw, p2 = shapiro(Win[i,1,:])
        
        if np.logical_or(p1<=.05,p2<=.05):
            r_coef_perm[m,n],p =  spearmanr(Win[i,0,perm_ix],Win[i,1,:])
        else:
            print('Pearson test, model %d, draw %d' %(m,n))
            r_coef_perm[m,n],p = pearsonr(Win[i,0,perm_ix],Win[i,1,:])
            

#%% plot means

mean_r_coef_perm = np.mean(r_coef_perm,0)

plt.figure()
plt.hist(mean_r_coef_perm)


ylims = plt.ylim()


yy = np.linspace(ylims[0],ylims[1])
xx = [np.mean(r_coef)]*len(yy)
plt.plot(xx,yy,'r--')


#%% plot r coefs

plt.figure()
plt.boxplot(r_coef)

plt.ylabel('r$_s$')
plt.xticks([])

sw, p3 = shapiro(r_coef)
if p3<=0.05:
    print('Wilcoxon test: ')
    stat, p_val = wilcoxon(r_coef)
else:
    print('1STT test: ')
    stat, p_val = ttest_1samp(r_coef)
print('        stat = %.3f, p = %.3f' %(stat,p_val))

#%% do the colour input units change at all? across training

mean_delta_weight = np.empty((n_models,))
std_delta_weight = np.empty((n_models,))

for i in range(n_models):

    plt.figure()
    # plt.subplot(121)
    # plt.imshow(Win0[0,2:,:],aspect=20)
    # plt.colorbar()
    # plt.subplot(122)
    # plt.imshow(Win[0,2:,:],aspect=20)
    # plt.colorbar()
    
    plt.imshow((Win[i,2:,:]-Win0[i,2:,:]).T,cmap = 'vlag',aspect=.05)
    plt.colorbar()
    plt.title('Model '+str(i))
    
    mean_delta_weight[i] = torch.mean(abs(Win[0,2:,:]-Win0[0,2:,:]))
    std_delta_weight[i]  = torch.std(abs(Win[0,2:,:]-Win0[0,2:,:]))
    
    

#%% permutation tests



