#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 21:59:10 2022

@author: emilia
"""
import pickle
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

import get_subspace_alignment_index as ai

import constants
from scipy.stats import boxcox
from sklearn import linear_model as lm




path = constants.PARAMS['FULL_PATH']
#%% get AI
AI_cued = ai.get_AI_cued_within_delay(constants.PARAMS)
AI_cu = ai.get_AI_cued_vs_uncued(constants.PARAMS)
AI_uncued = ai.get_AI_uncued_within_delay(constants.PARAMS)

#%% get training speed metrics 

n_epochs_to_convergence = np.empty((constants.PARAMS['n_models'],))
all_loss = torch.zeros((constants.PARAMS['n_models'],700))
for model in range(constants.PARAMS['n_models']):
    model_number = str(model)
    
    # load training data
    f = open(path+'training_data/'+'training_data_model'+model_number+'.pckl','rb')
    track_training = pickle.load(f)
    f.close()
    
    all_loss[model,:len(track_training['loss_epoch'])] = track_training['loss_epoch']
    # i = np.where(ix_sort==model)[0]
    # plt.plot(track_training['loss_epoch'],'-',c = colours[model,:],
    #           label='model '+model_number)
    
    n_epochs_to_convergence[model] = len(track_training['loss_epoch'])


cum_loss = all_loss.sum(-1)


#%%

X = np.stack((AI_cued[0,1,:],AI_cu[0,1,:],AI_uncued[0,1,:]))
tbl = np.concatenate((n_epochs_to_convergence[:,None],cum_loss[:,None],X.T),axis=1)

df = pd.DataFrame(tbl,columns=['n_epochs_to_convergence','cum_loss','AI_cued',
                               'AI_cued_uncued','AI_uncued'])
df.to_csv(path+'pca_data/valid_trials/geometry_vs_learning_regression_data.csv')

#%% visualise
ys = np.concatenate((n_epochs_to_convergence[:,None],cum_loss[:,None]),axis=1)
log_ys = np.log(ys)
x_labels = ['AI cued','AI cued vs uncued','AI uncued']
y_labels=['n_epochs_to_convergence','cum_loss']
log_y_labels=['log(n epochs)','log(cumulative loss)']

# X_norm = (X.T - X.mean(1))/np.std(X,1)


# beta_tbl = [[5.297,-.125,.17,.909],[.209,-.328,.184,.962]]
lin_reg = []
for y in range(2):
    plt.figure(figsize=(12.5,4.1))
    lin_reg.append(lm.LinearRegression())
    lin_reg[y].fit(X.T,log_ys[:,y])
    for x in range(3):    
        plt.subplot(1,3,x+1)
        plt.plot(X[x,:],log_ys[:,y],'ko')
        plt.xlabel(x_labels[x])
        plt.ylabel(log_y_labels[y])
        plt.xlim([-0.05,1.05])
        
        y_hat = lin_reg[y].intercept_ + X[x,:]*lin_reg[y].coef_[x]
        plt.plot(X[x,:],y_hat,'r-')
        
        plt.tight_layout()
        plt.savefig(constants.PARAMS['FIG_PATH']+'reg_'+log_y_labels[y]+'.png')
        
for y in range(2):
    plt.figure()
    for x in range(3):    
        plt.subplot(1,3,x+1)
        plt.plot((X[x,:]),ys[:,y],'ko')
        plt.xlabel(x_labels[x])
        plt.ylabel(y_labels[y])
        
        
#%% if vaidity paradigm - pool data from all conditions

paths = [constants.PARAMS['BASE_PATH']+'data_vonMises/experiment_3/validity_0.5/5_cycles/sigma0.0/kappa5.0/nrec200/lr0.0001/',
         constants.PARAMS['BASE_PATH']+'data_vonMises/experiment_3/validity_0.75/5_cycles/sigma0.0/kappa5.0/nrec200/lr0.0001/',
         constants.PARAMS['BASE_PATH']+'data_vonMises/experiment_3/validity_1/5_cycles/sigma0.0/kappa5.0/nrec200/lr0.0001/']
      
        
        
n_epochs_to_convergence = np.empty((constants.PARAMS['n_models']*3,))
all_loss = torch.zeros((constants.PARAMS['n_models']*3,600))
for i,p in enumerate(paths):
    print('Condition %d out of %d' %(i+1,len(paths)))
    for model in range(constants.PARAMS['n_models']):
        model_number = str(model)
        
        # load training data
        f = open(p+'training_data/'+'training_data_model'+model_number+'.pckl','rb')
        track_training = pickle.load(f)
        f.close()
        
        model_ix = i*constants.PARAMS['n_models']+model
        
        all_loss[model_ix,:len(track_training['loss_epoch'])] = track_training['loss_epoch']
        # i = np.where(ix_sort==model)[0]
        # plt.plot(track_training['loss_epoch'],'-',c = colours[model,:],
        #           label='model '+model_number)
        
        n_epochs_to_convergence[model_ix] = len(track_training['loss_epoch'])


cum_loss = all_loss.sum(-1)

condition = np.concatenate((['0.5']*constants.PARAMS['n_models'],
                      ['0.75']*constants.PARAMS['n_models'],
                      ['1.0']*constants.PARAMS['n_models']))

AI_cued = np.empty((2,constants.PARAMS['n_models']*3))
AI_cu = np.empty((2,constants.PARAMS['n_models']*3))
AI_uncued = np.empty((2,constants.PARAMS['n_models']*3))

# check that all funcs work with probe
for i,p in enumerate(paths):
    p += 'pca_data/valid_trials'
    model_ix = i*constants.PARAMS['n_models'] + np.arange(constants.PARAMS['n_models'])
    AI_cued[:,model_ix] = ai.get_AI_cued_probe(constants.PARAMS,custom_path=p)
    AI_cu[:,model_ix] = ai.get_AI_cued_vs_uncued_probe(constants.PARAMS,custom_path=p)
    AI_uncued[:,model_ix] = ai.get_AI_uncued_within_delay_probe(constants.PARAMS,custom_path=p)
    

X = np.stack((AI_cued[0,:],AI_cu[0,:],AI_uncued[0,:]))
tbl = np.concatenate((n_epochs_to_convergence[:,None],cum_loss[:,None],X.T,condition[:,None]),axis=1)


common_path = constants.PARAMS['BASE_PATH'] + 'data_vonMises/experiment_3/'
df = pd.DataFrame(tbl,columns=['n_epochs_to_convergence','cum_loss','AI_cued',
                               'AI_cued_uncued','AI_uncued','cue validity'])
df.to_csv(common_path+'geometry_vs_learning_regression_data.csv')

#%%
common_path = constants.PARAMS['BASE_PATH'] + 'data_vonMises/experiment_3/'        

paths = [common_path+'validity_0.5/5_cycles/sigma0.0/kappa5.0/nrec200/lr0.0001/',
         common_path+'validity_0.75/5_cycles/sigma0.0/kappa5.0/nrec200/lr0.0001/',
         common_path+'validity_1/5_cycles/sigma0.0/kappa5.0/nrec200/lr0.0001/']
      
validity = ['0.5','0.75','1.0']
for i,p in enumerate(paths):
    n_epochs_to_convergence = np.empty((30,))
    all_loss = torch.zeros((30,600))
    print('Condition %d out of %d' %(i+1,len(paths)))
    for model in range(30,60):
        model_number = str(model)
        
        # load training data
        f = open(p+'training_data/'+'training_data_model'+model_number+'.pckl','rb')
        track_training = pickle.load(f)
        f.close()
        
        
        all_loss[model-30,:len(track_training['loss_epoch'])] = track_training['loss_epoch']
        # i = np.where(ix_sort==model)[0]
        # plt.plot(track_training['loss_epoch'],'-',c = colours[model,:],
        #           label='model '+model_number)
        
        n_epochs_to_convergence[model-30] = len(track_training['loss_epoch'])
    
    cum_loss = all_loss.sum(-1)
    n_epochs_bc, _ = boxcox(n_epochs_to_convergence)
    cum_loss_bc, _ = boxcox(cum_loss)
    
    p += 'pca_data/valid_trials'
    AI_cued = ai.get_AI_cued_probe(constants.PARAMS,custom_path=p)
    AI_cu = ai.get_AI_cued_vs_uncued_probe(constants.PARAMS,custom_path=p)
    AI_uncued = ai.get_AI_uncued_within_delay_probe(constants.PARAMS,custom_path=p)

    X = np.stack((AI_cued[0,30:],AI_cu[0,30:],AI_uncued[0,30:]))
    tbl = np.concatenate((n_epochs_to_convergence[:,None],cum_loss[:,None],
                          n_epochs_bc[:,None],cum_loss_bc[:,None],
                          X.T,),axis=1)
    df = pd.DataFrame(tbl,columns=['n_epochs_to_convergence','cum_loss',
                                   'n_epochs_to_convergence_bc','cum_loss_bc','AI_cued',
                                   'AI_cued_uncued','AI_uncued'])
    df.to_csv(common_path+'geometry_vs_learning_regression_data_validity_' +validity[i]+ '_batch2.csv')


#%% btch 3

common_path = constants.PARAMS['BASE_PATH'] + 'data_vonMises/experiment_3/'        

paths = [common_path+'validity_0.5/5_cycles/sigma0.0/kappa5.0/nrec200/lr0.0001/',
         common_path+'validity_0.75/5_cycles/sigma0.0/kappa5.0/nrec200/lr0.0001/',
         common_path+'validity_1/5_cycles/sigma0.0/kappa5.0/nrec200/lr0.0001/']
      
validity = ['0.5','0.75','1.0']
for i,p in enumerate(paths):
    n_epochs_to_convergence = np.empty((40,))
    all_loss = torch.zeros((40,700))
    print('Condition %d out of %d' %(i+1,len(paths)))
    for model in range(60,100):
        model_number = str(model)
        
        # load training data
        f = open(p+'training_data/'+'training_data_model'+model_number+'.pckl','rb')
        track_training = pickle.load(f)
        f.close()
        
        
        all_loss[model-60,:len(track_training['loss_epoch'])] = track_training['loss_epoch']
        # i = np.where(ix_sort==model)[0]
        # plt.plot(track_training['loss_epoch'],'-',c = colours[model,:],
        #           label='model '+model_number)
        
        n_epochs_to_convergence[model-60] = len(track_training['loss_epoch'])
    
    cum_loss = all_loss.sum(-1)
    n_epochs_bc, _ = boxcox(n_epochs_to_convergence)
    cum_loss_bc, _ = boxcox(cum_loss)
    
    p += 'pca_data/valid_trials'
    AI_cued = ai.get_AI_cued_probe(constants.PARAMS,custom_path=p)
    AI_cu = ai.get_AI_cued_vs_uncued_probe(constants.PARAMS,custom_path=p)
    AI_uncued = ai.get_AI_uncued_within_delay_probe(constants.PARAMS,custom_path=p)

    X = np.stack((AI_cued[0,60:],AI_cu[0,60:],AI_uncued[0,60:]))
    tbl = np.concatenate((n_epochs_to_convergence[:,None],cum_loss[:,None],
                          n_epochs_bc[:,None],cum_loss_bc[:,None],
                          X.T,),axis=1)
    df = pd.DataFrame(tbl,columns=['n_epochs_to_convergence','cum_loss',
                                   'n_epochs_to_convergence_bc','cum_loss_bc','AI_cued',
                                   'AI_cued_uncued','AI_uncued'])
    df.to_csv(common_path+'geometry_vs_learning_regression_data_validity_' +validity[i]+ '_batch3.csv')


#%%

common_path = constants.PARAMS['BASE_PATH'] + 'data_vonMises/experiment_3/'        

paths = [common_path+'validity_0.5/5_cycles/sigma0.0/kappa5.0/nrec200/lr0.0001/',
         common_path+'validity_0.75/5_cycles/sigma0.0/kappa5.0/nrec200/lr0.0001/',
         common_path+'validity_1/5_cycles/sigma0.0/kappa5.0/nrec200/lr0.0001/']
      
validity = ['0.5','0.75','1.0']
for i,p in enumerate(paths):
    n_epochs_to_convergence = np.empty((30,))
    all_loss = torch.zeros((30,600))
    print('Condition %d out of %d' %(i+1,len(paths)))
    for model in range(30):
        model_number = str(model)
        
        # load training data
        f = open(p+'training_data/'+'training_data_model'+model_number+'.pckl','rb')
        track_training = pickle.load(f)
        f.close()
        
        
        all_loss[model,:len(track_training['loss_epoch'])] = track_training['loss_epoch']
        # i = np.where(ix_sort==model)[0]
        # plt.plot(track_training['loss_epoch'],'-',c = colours[model,:],
        #           label='model '+model_number)
        
        n_epochs_to_convergence[model] = len(track_training['loss_epoch'])
    
    cum_loss = all_loss.sum(-1)
    n_epochs_bc, _ = boxcox(n_epochs_to_convergence)
    cum_loss_bc, _ = boxcox(cum_loss)
    
    p += 'pca_data/valid_trials'
    AI_cued = ai.get_AI_cued_probe(constants.PARAMS,custom_path=p)
    AI_cu = ai.get_AI_cued_vs_uncued_probe(constants.PARAMS,custom_path=p)
    AI_uncued = ai.get_AI_uncued_within_delay_probe(constants.PARAMS,custom_path=p)

    X = np.stack((AI_cued[0,:30],AI_cu[0,:30],AI_uncued[0,:30]))
    tbl = np.concatenate((n_epochs_to_convergence[:,None],cum_loss[:,None],
                          n_epochs_bc[:,None],cum_loss_bc[:,None],
                          X.T,),axis=1)
    df = pd.DataFrame(tbl,columns=['n_epochs_to_convergence','cum_loss',
                                   'n_epochs_to_convergence_bc','cum_loss_bc','AI_cued',
                                   'AI_cued_uncued','AI_uncued'])
    df.to_csv(common_path+'geometry_vs_learning_regression_data_validity_' +validity[i]+ '.csv')



#%%

i=2
p = paths[i]
n_epochs_to_convergence = np.empty((constants.PARAMS['n_models'],))
all_loss = torch.zeros((constants.PARAMS['n_models'],700))
print('Condition %d out of %d' %(i+1,len(paths)))
for model in range(constants.PARAMS['n_models']):
    model_number = str(model)
    
    # load training data
    f = open(p+'training_data/'+'training_data_model'+model_number+'.pckl','rb')
    track_training = pickle.load(f)
    f.close()
    
    
    all_loss[model,:len(track_training['loss_epoch'])] = track_training['loss_epoch']
    # i = np.where(ix_sort==model)[0]
    # plt.plot(track_training['loss_epoch'],'-',c = colours[model,:],
    #           label='model '+model_number)
    
    n_epochs_to_convergence[model] = len(track_training['loss_epoch'])

cum_loss = all_loss.sum(-1)
n_epochs_bc, _ = boxcox(n_epochs_to_convergence)
cum_loss_bc, _ = boxcox(cum_loss)

p += 'pca_data/valid_trials'
AI_cued = ai.get_AI_cued_probe(constants.PARAMS,custom_path=p)
AI_cu = ai.get_AI_cued_vs_uncued_probe(constants.PARAMS,custom_path=p)
AI_uncued = ai.get_AI_uncued_within_delay_probe(constants.PARAMS,custom_path=p)

X = np.stack((AI_cued[0,:],AI_cu[0,:],AI_uncued[0,:]))
tbl = np.concatenate((n_epochs_to_convergence[:,None],cum_loss[:,None],
                      n_epochs_bc[:,None],cum_loss_bc[:,None],
                      X.T,),axis=1)
df = pd.DataFrame(tbl,columns=['n_epochs_to_convergence','cum_loss',
                                'n_epochs_to_convergence_bc','cum_loss_bc','AI_cued',
                                'AI_cued_uncued','AI_uncued'])
df.to_csv(common_path+'geometry_vs_learning_regression_data_validity_' +validity[i]+ '_all.csv')

#%% plots

def plot_pred_vs_dep(validity,transf=1,dep=0,cond='_all'):
    # transf = 1 - bc, otherwise raw
    # dep = 1 - cum loss, otherwise n epochs
    # load dataframe
    path = constants.PARAMS['BASE_PATH'] + 'data_vonMises/experiment_3/' \
        + 'geometry_vs_learning_regression_data_validity_' + str(validity) \
    
    df = pd.read_csv(path + cond + '.csv')
    # preds = ['AI_cued','AI_cued_uncued','AI_uncued']
    # dep = ['n_epochs_to_convergence_bc','cum_loss_bc']
    # dep_s = ['n_epochs','cum_loss']
    # transf = ['raw','_bc']
    # cond = '_batch1'
    
    dep_ix = 1+transf*2 + dep
    reg = lm.LinearRegression()
    reg.fit(df.iloc[:,-3:],df.iloc[:,dep_ix])
    
    fig, ax = plt.subplots(1,3,sharey=True)
    fig.set_size_inches(12.6,3.9)
    # plt.figure(figsize=(12.6,3.9))
    for x in range(3):
        # plt.subplot(1,3,x+1,sharex = True, sharey= True)
        # plot y residual vs predictor
        # res_reg = lm.LinearRegression()
        # res_ix = np.setdiff1d(np.arange(-3,0,1),[-3+x])
        # res_reg.fit(df.iloc[:,res_ix],df.iloc[:,dep_ix])
        # res = res_reg.predict(df.iloc[:,res_ix])
        # y = df.iloc[:,dep_ix] - res
        y = df.iloc[:,dep_ix]
        ax[x].plot(df.iloc[:,5+x],y,'ko')
        ax[x].set_xlabel(df.columns[-3+x].replace('_',' '))
        # add regression line
        y_hat = reg.intercept_ + df.iloc[:,-3+x]*reg.coef_[x]
        ax[x].plot(df.iloc[:,-3+x],y_hat,'r-')
    
    ax[0].set_ylabel(df.columns[dep_ix].replace('_',' '))
    plt.tight_layout()
    
    plt.savefig(constants.PARAMS['BASE_PATH'] + 'data_vonMises/experiment_3/'\
                +'figs/geometry_vs_'+df.columns[dep_ix]+'_validity_'\
                    +str(validity)+cond+'.png')
    
#%%

for t in range(2):
    for d in range(2):
        for v in [0.5,0.75,1.0]:
            for c in ['_batch1','_batch2','_batch3','_all' ]:
                plot_pred_vs_dep(v,transf=t,dep=d,cond=c)
    
    # for i in range(2):
    #     plt.figure(figsize=(12.6,3.9))
    #     for x in range(3):
    #         plt.subplot(1,3,x+1)
    #         plt.plot(tbl[:30,4+x],tbl[:30,2+i],'ko')
    #         plt.xlabel(preds[x])
    #         plt.ylabel(dep[i])
    #     plt.tight_layout()
    #     plt.savefig(common_path+'figs/geometry_vs_'+dep_s[i]+'_validity_'+validity[i]+ cond+'.png')
# from helpers import check_path
# check_path(common_path+'figs/')

# 

#%%
# n epochs to conv

r, p = spearmanr(df.iloc[:,1],df.iloc[:,5:],axis=0)
plt.figure()
for i in range(3):
    plt.subplot(1,3,i+1)
    plt.plot(df.iloc[:,5+i],df.iloc[:,1],'ko')
    plt.text(plt.xlim()[0]+.05,120,'r = %.2f' %r[0,i+1])
    plt.text(plt.xlim()[0]+.05,110,'p = %.3f' %(p[0,i+1]))
    plt.xlabel(df.columns[-3+i])
    plt.ylabel(df.columns[1])
# cum loss

r, p = spearmanr(df.iloc[:,2],df.iloc[:,5:],axis=0)
plt.figure()
for i in range(3):
    plt.subplot(1,3,i+1)
    plt.plot(df.iloc[:,5+i],df.iloc[:,2],'ko')
    plt.text(plt.xlim()[0]+.05,0.35,'r = %.2f' %r[0,i+1])
    plt.text(plt.xlim()[0]+.05,0.33,'p = %.3f' %(p[0,i+1]))
    plt.xlabel(df.columns[-3+i])
    plt.ylabel(df.columns[2])

