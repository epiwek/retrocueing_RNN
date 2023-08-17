#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 11:43:54 2021

@author: emilia
"""
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import pycircstat
# matplotlib.rcParams.update({'font.size': 22})

import generate_data_vonMises as dg
from helpers import check_path, wrap_angle


from scipy.signal import argrelextrema
from scipy.stats import shapiro, pearsonr, wilcoxon, ttest_1samp
from sklearn.decomposition import PCA


import retrocue_model as retnet
from rep_geom_analysis import run_pca_pipeline
import custom_plot as cplot


from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema

import pandas as pd
import itertools

from model_stats import plot_plane_angles_single
from get_subspace_alignment_index import get_AI_tbl

import stats

#%% 
def get_dLoss_dt(loss_vals,smooth_sd,smooth=True):
    '''
    Calculate the derivative of the loss function wrt time. Used for finding 
    learning plateaus.

    Parameters
    ----------
    params: dict
        dictionary containing the Gaussian filter s.d. in
        params['conv_criterion']['smooth_sd']
    loss_vals : torch.tensor
        loss values for every epoch (averaged cross all training examples).
    
   
    Returns
    -------
    dLoss : numpy array
        derivative of the loss wrt to time.
    loss_clean : numpy array
        loss values after smoothing

    '''
    if len(loss_vals.shape)<2:
        ValueError('Loss_vals can''t be a 1-dimensional array')

    # convolve with a Gaussian filter to smooth the loss curve
    if smooth:
        loss_clean = gaussian_filter1d(loss_vals,smooth_sd)
    else:
        loss_clean = loss_vals
    loss_clean = torch.tensor(loss_clean)
    
    # calculate the derivative
    dLoss = torch.zeros(loss_clean.shape[0]-1)
    for i in range(loss_clean.shape[0]-1):
            dLoss[i] = loss_clean[i+1] - loss_clean[i]
    return dLoss, loss_clean


def find_learning_plateaus(params):
    '''
    Loads loss data from training and calculates its derivative to find the 
    plateaus in learning curves.
    
    Parameters
    ----------
       
    params: dict
       task and model parameters

        
    Returns
    -------
    lc_plateaus : numpy array (n_models,) 
        contains the epoch indices corresponding to the learning plateau for 
        each model

    '''
    data_path = params['FULL_PATH']
    fig_path = params['FIG_PATH'] + 'learning_dynamics_plots/'
    check_path(fig_path)
    
    lc_plateaus = np.zeros((params['n_models'],1),dtype=int)
    # loop over models

    for model in range(params['n_models']):
        
        model_number = str(model)
        
        # load data
        f = open(data_path+'training_data/'+'training_data_model'+model_number+'.pckl','rb')
        track_training = pickle.load(f)
        f.close()
        
        # slope_loss = []
        # window = constants.PARAMS['conv_criterion']['window']
        # for i in range(window,len(track_training['loss_epoch'])):
        #     slope_loss.append(get_loss_slope(constants.PARAMS,track_training['loss_epoch'][i-window+1:i+1]))
        # _________________________
        # calculate the derivative of the loss wrt time
        dLoss,loss_clean = get_dLoss_dt(track_training['loss_epoch'],4)
    
        # find local minima
        ix = argrelextrema(np.array(dLoss), np.greater)[0]
        
        # exclude the start plateau - any timepoints were the loss value is 
        # within a 5% margin from the initial value
        ix = np.setdiff1d(ix,ix[np.where(loss_clean[ix]>=loss_clean[0]*.95)[0]])
        
        # ix2 = argrelextrema(np.array(loss_clean), np.less)[0]

        # lc_plateaus[model,:] = ix[:2]-1
        # lc_plateaus = np.setdiff1d(ix,ix2)
        
        # only look at the first plateau
        lc_plateaus[model] = ix[0]
        # _________________________
        
        # plot loss and local minima for each model
        plt.figure()
        plt.subplot(211)
        plt.plot(track_training['loss_epoch'],'k-',label='loss raw')
        plt.plot(loss_clean,'g-',label='loss clean') 
        plt.plot(lc_plateaus[model],loss_clean[lc_plateaus[model]],'.r')
        plt.ylabel('Loss')
        plt.legend(loc='upper right',prop={'size': 10})
        
        plt.subplot(212)
        plt.plot(dLoss) 
        plt.plot(lc_plateaus[model],dLoss[lc_plateaus[model]],'r.')
        plt.xlabel('epoch')
        plt.ylabel('dLoss')
        
        plt.suptitle('Model '+model_number)
        plt.tight_layout()
        
        plt.savefig(fig_path+model_number+'.png')
        
    # save the epoch indices corresponding to plateaus
    pickle.dump(lc_plateaus,open(data_path+'training_data/lc_plateau.pckl','wb'))
    
    return lc_plateaus


def partial_train_and_eval(constants,test_data,device):
    '''
    Train and evaluate the untrained and partially-trained (until loss plateau)
    models.
    
    Parameters
    ----------
    constants : dict
        constants for the experiment.
    test_data : dict
        test dataset.
    device : torch object
        device for model training.

    Returns
    -------
    None.

    '''
    print('PARTIAL TRAINING ANALYSIS')
    eval_path = constants.PARAMS['FULL_PATH'] +'pca_data/valid_trials/partial_training/'
    check_path(eval_path)
    check_path(eval_path+'untrained/')
    check_path(eval_path+'plateau/')
    
    # find learning plateau timepoint for each modell
    lc_plateaus = find_learning_plateaus(constants.PARAMS)
    
    for m in range(constants.PARAMS['n_models']):
        print('Model %d' %m)
        print('Untrained')
        constants.PARAMS['model_number'] = m
        
        # set seed for reproducibility - controls both the initialisation and trial sequences
        torch.manual_seed(constants.PARAMS['model_number'])
        
        
        #% eval untrained
      
        #% initialise model
        torch.manual_seed(constants.PARAMS['model_number'])
        model = retnet.RNN(constants.PARAMS,device)
        #evaluate
        eval_data,pca_data,rdm_data,model_outputs = \
        retnet.eval_model(model,test_data,constants.PARAMS,eval_path+'untrained/')
        
        #% learning plateau
        # train model
        print('Partially trained')
        constants.PARAMS['n_epochs'] = lc_plateaus[m][0]
        model, track_training = retnet.train_model(constants.PARAMS,
                                                   constants.TRAINING_DATA,
                                                   device)
        # evaluate
        eval_data,pca_data,rdm_data,model_outputs = \
        retnet.eval_model(model,test_data,constants.PARAMS,eval_path+'plateau/')

    
# def get_cosine_angle(constants):
#     load_path = constants.PARAMS['FULL_PATH'] +'pca_data/valid_trials/' #'partial_training/'
#     train_stages = ['untrained','plateau','trained']
#     delay_len = constants.PARAMS['trial_timings']['delay1_dur']
    
#     theta_pre = np.empty((constants.PARAMS['n_models'],len(train_stages),delay_len))
#     theta_post = np.empty((constants.PARAMS['n_models'],len(train_stages),delay_len))
    
#     pve_pre = np.empty((constants.PARAMS['n_models'],len(train_stages),delay_len))
#     pve_post = np.empty((constants.PARAMS['n_models'],len(train_stages),delay_len))
    
#     d1_start = constants.PARAMS['trial_timepoints']['delay1_start']
#     d2_start = constants.PARAMS['trial_timepoints']['delay2_start']
#     # loop over models
#     for model in range(constants.PARAMS['n_models']):        
#         #% load data
#         # fully trained
#         pca_data_ft = pickle.load(open(load_path+'pca_data_model'+str(model)+'.pckl','rb'))
#         # untrained
#         pca_data_ut = pickle.load(open(load_path+'partial_training/untrained/'+
#                            'pca_data_model'+str(model)+'.pckl','rb'))
#         # plateau
#         pca_data_p = pickle.load(open(load_path+'partial_training/plateau/'+
#                            'pca_data_model'+str(model)+'.pckl','rb'))
        
#         all_data = [pca_data_ut,pca_data_p,pca_data_ft]
#         # run the PCA pipeline on both delays, separately for each timepoint
#         for stage in range(len(all_data)):
#             for t in range(delay_len):
#                 subspace_d1 = run_pca_pipeline(constants,
#                                                all_data[stage]['data'][:,d1_start+t,:],
#                                                ['up','down'])
#                 subspace_d2 = run_pca_pipeline(constants,
#                                                all_data[stage]['data'][:,d2_start+t,:],
#                                                ['up','down'])
#                 theta_pre[model,stage,t] = subspace_d1['theta']
#                 theta_post[model,stage,t] = subspace_d2['theta']
                
#                 pve_pre[model,stage,t] = subspace_d1['pca'].explained_variance_ratio_.sum()
#                 pve_post[model,stage,t] = subspace_d2['pca'].explained_variance_ratio_.sum()
    
#         cos_theta_pre = np.cos(np.radians(theta_pre))
#         cos_theta_post = np.cos(np.radians(theta_post))
        
#         cos_theta = {}
#         cos_theta['delay1'] = cos_theta_pre
#         cos_theta['delay2'] = cos_theta_post
#         cos_theta['dimensions'] = ['model','train_stage','delay_timepoint']
#         cos_theta['train_stages'] = train_stages
        
#         pve = {}
#         pve['delay1'] = pve_pre
#         pve['delay2'] = pve_post
#         pve['dimensions'] = ['model','train_stage','delay_timepoint']
#         pve['train_stages'] = train_stages
        
    
#     return cos_theta, pve


def get_angle(constants):
    '''
    Calculate angle theta between the Cued planes at the 3 different timepoints 
    during training (untrained,plateau,trained).

    Parameters
    ----------
    constants : dict
        constants for the experiment.

    Returns
    -------
    theta : dict
        Angle between planes. Entries include:
            - 'delay1' - array with theta values for all timepoints in the first
                (pre-cue) delay, size : (n_models,3,n_delay_timepoints)
            - 'delay2' - anaogous array with data from the second (post-cue) 
                delay
            - 'dimensions' - list of labels for the dimensions of the above arrays
            - 'train_stages' - list of train stage labels
    cos_theta : dict
        Analogous data structure containing the cos theta values.
    pve : dict
        Analogous data structure containing the total percentage of variance 
        explained by 3 principal components at each delay timepoint and 
        training stage, for each model.
    '''
    load_path = constants.PARAMS['FULL_PATH'] +'pca_data/valid_trials/'
    train_stages = ['untrained','plateau','trained']
    delay_len = constants.PARAMS['trial_timings']['delay1_dur']
    
    theta_pre = np.empty((constants.PARAMS['n_models'],len(train_stages),delay_len))
    theta_post = np.empty((constants.PARAMS['n_models'],len(train_stages),delay_len))
    
    pve_pre = np.empty((constants.PARAMS['n_models'],len(train_stages),delay_len))
    pve_post = np.empty((constants.PARAMS['n_models'],len(train_stages),delay_len))
    
    d1_start = constants.PARAMS['trial_timepoints']['delay1_start']
    d2_start = constants.PARAMS['trial_timepoints']['delay2_start']
    # loop over models
    for model in range(constants.PARAMS['n_models']):        
        #% load data
        # fully trained
        pca_data_ft = pickle.load(open(load_path+'pca_data_model'+str(model)+'.pckl','rb'))
        # untrained
        pca_data_ut = pickle.load(open(load_path+'partial_training/untrained/'+
                           'pca_data_model'+str(model)+'.pckl','rb'))
        # plateau
        pca_data_p = pickle.load(open(load_path+'partial_training/plateau/'+
                           'pca_data_model'+str(model)+'.pckl','rb'))
        
        all_data = [pca_data_ut,pca_data_p,pca_data_ft]
        # run the PCA pipeline on both delays, separately for each timepoint
        for stage in range(len(all_data)):
            for t in range(delay_len):
                subspace_d1 = run_pca_pipeline(constants,
                                               all_data[stage]['data'][:,d1_start+t,:],
                                               ['up','down'])
                subspace_d2 = run_pca_pipeline(constants,
                                               all_data[stage]['data'][:,d2_start+t,:],
                                               ['up','down'])
                theta_pre[model,stage,t] = subspace_d1['theta']
                theta_post[model,stage,t] = subspace_d2['theta']
    
        theta = {}
        theta['delay1'] = theta_pre
        theta['delay2'] = theta_post
        theta['dimensions'] = ['model','train_stage','delay_timepoint']
        theta['train_stages'] = train_stages
        
        cos_theta = {}
        cos_theta['delay1'] = np.cos(np.radians(theta_pre))
        cos_theta['delay2'] = np.cos(np.radians(theta_post))
        cos_theta['dimensions'] = ['model','train_stage','delay_timepoint']
        cos_theta['train_stages'] = train_stages
        
        pve = {}
        pve['delay1'] = pve_pre
        pve['delay2'] = pve_post
        pve['dimensions'] = ['model','train_stage','delay_timepoint']
        pve['train_stages'] = train_stages
    
    return theta, cos_theta, pve


def get_AI(constants):
    '''
    Calculate the AI between the Cued planes at the 3 different timepoints 
    during training (untrained,plateau,trained).

    Parameters
    ----------
    constants : dict
        constants for the experiment.

    Returns
    -------
    AI : dict
        Angle between planes. Entries include:
            - 'delay1' - array with theta values for all timepoints in the first
                (pre-cue) delay, size : (n_models,3,n_delay_timepoints)
            - 'delay2' - anaogous array with data from the second (post-cue) 
                delay
            - 'dimensions' - list of labels for the dimensions of the above arrays
            - 'train_stages' - list of train stage labels
    pve : dict
        Analogous data structure containing the total percentage of variance 
        explained by 3 principal components at each delay timepoint and 
        training stage, for each model.
    '''
    load_path = constants.PARAMS['FULL_PATH'] +'pca_data/valid_trials/'
    train_stages = ['untrained','plateau','trained']
    delay_len = constants.PARAMS['trial_timings']['delay1_dur']
    
    AI_pre = np.empty((constants.PARAMS['n_models'],len(train_stages),delay_len))
    AI_post = np.empty((constants.PARAMS['n_models'],len(train_stages),delay_len))
    
    pve_pre = np.empty((constants.PARAMS['n_models'],len(train_stages),delay_len))
    pve_post = np.empty((constants.PARAMS['n_models'],len(train_stages),delay_len))
    
    d1_start = constants.PARAMS['trial_timepoints']['delay1_start']
    d2_start = constants.PARAMS['trial_timepoints']['delay2_start']
    # loop over models
    
    paths = [load_path,
             load_path+'partial_training/untrained/',
             load_path+'partial_training/plateau/']
    
    for model in range(constants.PARAMS['n_models']):        
        #% load data
        # fully trained
        pca_data_ft = pickle.load(open(load_path+'pca_data_model'+str(model)+'.pckl','rb'))
        # untrained
        pca_data_ut = pickle.load(open(load_path+'partial_training/untrained/'+
                            'pca_data_model'+str(model)+'.pckl','rb'))
        # plateau
        pca_data_p = pickle.load(open(load_path+'partial_training/plateau/'+
                            'pca_data_model'+str(model)+'.pckl','rb'))
        
        all_data = [pca_data_ut,pca_data_p,pca_data_ft]
        
        half_split = pca_data_ut['data'].shape[0]//2
        # run the PCA pipeline on both delays, separately for each timepoint
        for stage in range(3):
            for t in range(delay_len):
                
                AI_pre[model,stage,t] = get_AI_tbl(constants.PARAMS,
                                           all_data[stage]['data'][:half_split,d1_start+t,:],
                                           all_data[stage]['data'][half_split:,d1_start+t,:],
                                           max_dim=2)
                AI_post[model,stage,t] = get_AI_tbl(constants.PARAMS,
                                           all_data[stage]['data'][:half_split,d2_start+t,:],
                                           all_data[stage]['data'][half_split:,d2_start+t,:],
                                           max_dim=2)
        AI = {}
        AI['delay1'] = AI_pre
        AI['delay2'] = AI_post
        AI['dimensions'] = ['model','train_stage','delay_timepoint']
        AI['train_stages'] = train_stages
        
        # theta = {}
        # theta['delay1'] = theta_pre
        # theta['delay2'] = theta_post
        # theta['dimensions'] = ['model','train_stage','delay_timepoint']
        # theta['train_stages'] = train_stages
        
        # cos_theta = {}
        # cos_theta['delay1'] = np.cos(np.radians(theta_pre))
        # cos_theta['delay2'] = np.cos(np.radians(theta_post))
        # cos_theta['dimensions'] = ['model','train_stage','delay_timepoint']
        # cos_theta['train_stages'] = train_stages
        
        # pve = {}
        # pve['delay1'] = pve_pre
        # pve['delay2'] = pve_post
        # pve['dimensions'] = ['model','train_stage','delay_timepoint']
        # pve['train_stages'] = train_stages
    
    return AI



def get_retro_weights(constants):
    '''
    Extracts the learnt retrocue-recurrent weight vectors for a given model.

    Parameters
    ----------
    constants : dict
        constants for the experiment.

    Returns
    -------
    cue1_weights : array (n_recurrent,)
        Vector of the retrocue-recurrent weights for the retrocue 1 unit.
    cue2_weights : array (n_recurrent,)
        Analogous for retrocue 2 unit..

    '''
    load_path = constants.PARAMS['FULL_PATH']+'saved_models/'
    device = torch.device('cpu')
    
    
    # model, net_type = retnet.define_model(constants.PARAMS,device)
    model = retnet.load_model(load_path,constants.PARAMS,device)
    
    # extract weights
    cue1_weights = model.inp.weight[:,0].detach()
    cue2_weights = model.inp.weight[:,1].detach()
    
    return cue1_weights, cue2_weights


def corr_retro_weights(constants):
    '''
    

    Parameters
    ----------
    constants : TYPE
        DESCRIPTION.

    Returns
    -------
    r : TYPE
        DESCRIPTION.
    p_val : TYPE
        DESCRIPTION.

    '''
    r = np.empty((constants.PARAMS['n_models']))
    p_val = np.empty((constants.PARAMS['n_models']))
    
    for m in range(constants.PARAMS['n_models']):
        # get weights
        constants.PARAMS['model_number'] = m
        cue1_weights, cue2_weights = get_retro_weights(constants)
        
        
        # correlate
        r[m],p_val[m] = pearsonr(cue1_weights,cue2_weights)
    
    return r,p_val
    
    
def test_r_retro_weights(r):
    s,p = shapiro(r)
    print('Retrocue-recurrent weight correlation results')
    if p < .05:
        stat,p_val = wilcoxon(r)
        print('wilcoxon test')
    else:
        stat,p_val = ttest_1samp(r,0)
        d = stats.cohens_d(r, [0])
        print('Cohen''s d = %.2f' %(d))
        print('1STT')
    print('statistic: %.3f, p-val: %.3f' %(stat,p_val))
    return stat,p_val
    
    
#%% plot

def plot_data_shadowplot(constants,data,ylabel):
    '''
    Plot angle or pve data as shadowplot.

    Parameters
    ----------
    constants : TYPE
        DESCRIPTION.
    data : dict
        cos_theta or pve.
    ylabel : str
        measured variable.

    Returns
    -------
    None.

    '''
    labels = data['train_stages']
    
    cols = sns.color_palette("rocket_r",len(labels))
    
    plt.figure(figsize=(10,4))
    ax = plt.subplot(121)
    ax2 = plt.subplot(122,sharey=ax)
    
    for i in range(len(labels)):
        H1,H2 = cplot.shadow_plot(ax,
                          np.arange(data['delay1'].shape[-1])+1,
                          data['delay1'][:,i,:],alpha=.3,
                          color=cols[i])
        
        
        
        H3,H4 = cplot.shadow_plot(ax2,
                          np.arange(data['delay1'].shape[-1])+1,
                          data['delay2'][:,i,:],alpha=.3,
                          color=cols[i])
    
    ax.legend(ax.get_children()[3:6],labels,bbox_to_anchor=(1,.7),loc='center right')   

    ax.set_title('pre-cue delay')
    
    ax2.set_title('post-cue delay')
    ax2.set_xlabel('Time from delay onset')
    
    ax.set_ylabel(ylabel)
    
    
    plt.tight_layout()
    # plt.savefig(constants.PARAMS['FIG_PATH']+'learning_dynamics.png')
    
    
# def plot_angle(constants,theta):
#     labels = theta['train_stages']
    
#     cols = sns.color_palette("rocket_r",len(labels))
    
#     plt.figure(figsize=(10,4))
#     ax = plt.subplot(121)
#     ax2 = plt.subplot(122,sharey=ax)
    
#     for i in range(len(labels)):
#         theta_mean_pre = np.degrees(pycircstat.mean(np.radians(theta['delay1'][:,i,:]),axis=0))
#         theta_sem_pre = np.degrees(pycircstat.std(np.radians(theta['delay1'][:,i,:]),axis=0)/np.sqrt(constants.PARAMS['n_models']))
#         H1,H2 = cplot.shadow_plot(ax,
#                           np.arange(theta['delay1'].shape[-1])+1,
#                           [theta_mean_pre,theta_sem_pre],precalc = True,
#                           alpha=.3,
#                           color=cols[i])
        
        
#         theta_mean_post = np.degrees(pycircstat.mean(np.radians(theta['delay2'][:,i,:]),axis=0))
#         theta_sem_post = np.degrees(pycircstat.std(np.radians(theta['delay2'][:,i,:]),axis=0)/np.sqrt(constants.PARAMS['n_models']))
#         H3,H4 = cplot.shadow_plot(ax2,
#                           np.arange(theta['delay1'].shape[-1])+1,
#                           [theta_mean_post,theta_sem_post],precalc = True,
#                           alpha=.3,
#                           color=cols[i])
    
#     ax.legend(ax.get_children()[3:6],labels,bbox_to_anchor=(1,.6),loc='center right')
   


#     ax.set_title('pre-cue delay')
    
#     ax2.set_title('post-cue delay')
#     ax2.set_xlabel('Time from delay onset')
    
#     ax.set_ylabel('$θ$ [°]')
    
    
#     plt.tight_layout()
#     plt.savefig(constants.PARAMS['FIG_PATH']+'learning_dynamics_shadow_plot.png')


# def plot_angle(constants,theta):
#     labels = theta['train_stages']
    
#     cols = sns.color_palette("rocket_r",len(labels))
    
#     plt.figure(figsize=(10,4))
#     ax = plt.subplot(121)
#     ax2 = plt.subplot(122,sharey=ax)
    
#     for i in range(len(labels)):
#         theta_mean_pre = np.degrees(pycircstat.mean(np.radians(theta['delay1'][:,i,:]),axis=0))
#         theta_sem_pre = np.degrees(pycircstat.std(np.radians(theta['delay1'][:,i,:]),axis=0)/np.sqrt(constants.PARAMS['n_models']))
#         H1,H2 = cplot.shadow_plot(ax,
#                           np.arange(theta['delay1'].shape[-1])+1,
#                           [theta_mean_pre,theta_sem_pre],precalc = True,
#                           alpha=.3,
#                           color=cols[i])
        
        
#         theta_mean_post = np.degrees(pycircstat.mean(np.radians(theta['delay2'][:,i,:]),axis=0))
#         theta_sem_post = np.degrees(pycircstat.std(np.radians(theta['delay2'][:,i,:]),axis=0)/np.sqrt(constants.PARAMS['n_models']))
#         H3,H4 = cplot.shadow_plot(ax2,
#                           np.arange(theta['delay1'].shape[-1])+1,
#                           [theta_mean_post,theta_sem_post],precalc = True,
#                           alpha=.3,
#                           color=cols[i])
    
#     ax.legend(ax.get_children()[3:6],labels,bbox_to_anchor=(1,.6),loc='center right')
   


#     ax.set_title('pre-cue delay')
    
#     ax2.set_title('post-cue delay')
#     ax2.set_xlabel('Time from delay onset')
    
#     ax.set_ylabel('$θ$ [°]')
    
    
#     plt.tight_layout()
#     plt.savefig(constants.PARAMS['FIG_PATH']+'learning_dynamics_shadow_plot.png')
    

def plot_angle(constants,theta):
    '''
    Plot angle theta as errorbars.

    Parameters
    ----------
    constants : TYPE
        DESCRIPTION.
    theta : dict
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    labels = theta['train_stages']
    
    cols = sns.color_palette("rocket_r",len(labels))
    
    plt.figure(figsize=(10,4))
    ax = plt.subplot(121)
    ax2 = plt.subplot(122,sharey=ax)
    for i in range(len(labels)):
        theta_mean_pre = pycircstat.mean(np.radians(theta['delay1'][:,i,:]),axis=0)
        # wrap to -pi,pi
        theta_mean_pre = wrap_angle(torch.from_numpy(theta_mean_pre))
        theta_mean_pre = np.degrees(np.array(theta_mean_pre))
        theta_sem_pre = np.degrees(pycircstat.std(np.radians(theta['delay1'][:,i,:]),axis=0)/np.sqrt(constants.PARAMS['n_models']))
        # H1,H2 = cplot.shadow_plot(ax,
        #                   np.arange(theta['delay1'].shape[-1])+1,
        #                   [theta_mean_pre,theta_sem_pre],precalc = True,
        #                   alpha=.3,
        #                   color=cols[i])
        
        ax.errorbar(x=np.arange(theta['delay1'].shape[-1])+1,y=theta_mean_pre,
                    yerr=theta_sem_pre,color=cols[i],label=labels[i])
        
        
        theta_mean_post = pycircstat.mean(np.radians(theta['delay2'][:,i,:]),axis=0)
        theta_mean_post = wrap_angle(torch.from_numpy(theta_mean_post))
        theta_mean_post = np.degrees(np.array(theta_mean_post))

        theta_sem_post = np.degrees(pycircstat.std(np.radians(theta['delay2'][:,i,:]),axis=0)/np.sqrt(constants.PARAMS['n_models']))
        # H3,H4 = cplot.shadow_plot(ax2,
        #                   np.arange(theta['delay1'].shape[-1])+1,
        #                   [theta_mean_post,theta_sem_post],precalc = True,
        #                   alpha=.3,
        #                   color=cols[i])
        ax2.errorbar(x=np.arange(theta['delay2'].shape[-1])+1,y=theta_mean_post,
                    yerr=theta_sem_post,color=cols[i])
    
    # ax.legend(ax.get_children()[3:6],labels,bbox_to_anchor=(1,.6),loc='center right')
    ax.legend(bbox_to_anchor=(1,.5),loc='center right')


    ax.set_title('pre-cue delay')
    
    ax2.set_title('post-cue delay')
    ax2.set_xlabel('Time from delay onset')
    
    ax.set_ylabel('$θ$ [°]')
    
    
    plt.tight_layout()
    plt.savefig(constants.PARAMS['FIG_PATH']+'learning_dynamics_plot.svg')
    

def plot_AI_across_training(constants,AI):
    '''
    Plot angle theta as errorbars.

    Parameters
    ----------
    constants : TYPE
        DESCRIPTION.
    theta : dict
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    labels = AI['train_stages']
    
    cols = sns.color_palette("rocket_r",len(labels))
    
    plt.figure(figsize=(5,10))
    ax = plt.subplot(211)
    ax2 = plt.subplot(212,sharey=ax,sharex = ax)
    pal = sns.color_palette("dark")
    inds = [3,0]
    cols = [pal[ix] for ix in inds]
    
    AI_mean_pre = AI['delay1'][:,:,-1].mean(0)
    AI_sem_pre = np.std(AI['delay1'][:,:,-1],axis=0)/np.sqrt(constants.PARAMS['n_models'])
    
    AI_mean_post = AI['delay2'][:,:,-1].mean(0)
    AI_sem_post = np.std(AI['delay2'][:,:,-1],axis=0)/np.sqrt(constants.PARAMS['n_models'])
    
    
    x_vals = np.arange(len(labels))

    ax.errorbar(x_vals-.02,y=AI_mean_pre,
                yerr=AI_sem_pre,color=cols[0],lw=7,alpha=.2)
    ax.errorbar(x_vals+.02,y=AI_mean_post,
                yerr=AI_sem_post,color=cols[1],lw=7,alpha=.2)
    ax.scatter(x_vals-.02,y=AI_mean_pre,color=cols[0],s=100,label='pre-cue')
    ax.scatter(x_vals+.02,y=AI_mean_post,color=cols[1],marker='^',s=100,label='post-cue')
    
    plt.xlim([-.5,2.5])
    plt.ylabel('AI')
    plt.xticks(ticks=range(3),labels=labels)
    plt.legend()
    plt.tight_layout()
    
    
    
    ax.bar(labels,AI_mean_pre,
                color=cols,alpha=.2)
    ax2.bar(labels,AI_mean_post,
                color=cols,alpha=.2)
    # ax.bar(x_vals-.25,AI_mean_pre,width=.5,
    #             color=cols[0],alpha=.2)
    # ax.bar(x_vals+.25,AI_mean_post,width=.5,
    #             color=cols[1],alpha=.2)
    
    for i in range(len(labels)):
        # ax.plot([labels[i]]*constants.PARAMS['n_models'],AI['delay1'][:,i,-1],
        #         'o',color=cols[i])
        
        ax2.plot([labels[i]]*constants.PARAMS['n_models'],AI['delay2'][:,i,-1],
                '^',color=cols[i])
        
    for m in range(constants.PARAMS['n_models']):
        ax.plot(labels,AI['delay1'][m,:,-1],
                'k-',alpha=.2)
        
    #     ax.plot(x_vals+.25,AI['delay2'][m,:,-1],
    #             'k-',alpha=.2)
    
    
    # for i in range(len(labels)):
    #     AI_mean_pre = (AI['delay1'][:,i,-1]).mean(0)
    #     AI_sem_pre = np.std(AI['delay1'][:,i,-1],axis=0)/np.sqrt(constants.PARAMS['n_models'])
    #     # H1,H2 = cplot.shadow_plot(ax,
    #     #                   np.arange(theta['delay1'].shape[-1])+1,
    #     #                   [theta_mean_pre,theta_sem_pre],precalc = True,
    #     #                   alpha=.3,
    #     #                   color=cols[i])
        
    #     ax.errorbar(x=np.arange(AI['delay1'].shape[-1])+1,y=AI_mean_pre,
    #                 yerr=AI_sem_pre,color=cols[i],label=labels[i])
        
    #     AI_mean_post = (AI['delay2'][:,i,:]).mean(0)
    #     AI_sem_post =np.std(AI['delay2'][:,i,:],axis=0)/np.sqrt(constants.PARAMS['n_models'])
    #     # H3,H4 = cplot.shadow_plot(ax2,
    #     #                   np.arange(theta['delay1'].shape[-1])+1,
    #     #                   [theta_mean_post,theta_sem_post],precalc = True,
    #     #                   alpha=.3,
    #     #                   color=cols[i])
    #     ax2.errorbar(x=np.arange(AI['delay2'].shape[-1])+1,y=AI_mean_post,
    #                 yerr=AI_sem_post,color=cols[i])
    
    # # ax.legend(ax.get_children()[3:6],labels,bbox_to_anchor=(1,.6),loc='center right')
    # ax.legend(bbox_to_anchor=(1,.5),loc='center right')


    # ax.set_title('pre-cue delay')
    
    # ax2.set_title('post-cue delay')
    # ax2.set_xlabel('Time from delay onset')
    
    # ax.set_ylabel('AI')
    
    
    # plt.tight_layout()
    plt.savefig(constants.PARAMS['FIG_PATH']+'learning_dynamics_plot_AI_barplot.svg')


def plot_AI_across_training_all_delay_timepoints(constants,AI):
    '''
    Plot angle theta as errorbars, using all delay timepoints.

    Parameters
    ----------
    constants : TYPE
        DESCRIPTION.
    theta : dict
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    labels = AI['train_stages']
    
    cols = sns.color_palette("rocket_r",len(labels))
    
    plt.figure(figsize=(10,4))
    ax = plt.subplot(121)
    ax2 = plt.subplot(122,sharey=ax)
    for i in range(len(labels)):
        AI_mean_pre = (AI['delay1'][:,i,:]).mean(0)
        AI_sem_pre =np.std(AI['delay1'][:,i,:],axis=0)/np.sqrt(constants.PARAMS['n_models'])
        # H1,H2 = cplot.shadow_plot(ax,
        #                   np.arange(theta['delay1'].shape[-1])+1,
        #                   [theta_mean_pre,theta_sem_pre],precalc = True,
        #                   alpha=.3,
        #                   color=cols[i])
        
        ax.errorbar(x=np.arange(AI['delay1'].shape[-1])+1,y=AI_mean_pre,
                    yerr=AI_sem_pre,color=cols[i],label=labels[i])
        
        AI_mean_post = (AI['delay2'][:,i,:]).mean(0)
        AI_sem_post =np.std(AI['delay2'][:,i,:],axis=0)/np.sqrt(constants.PARAMS['n_models'])
        # H3,H4 = cplot.shadow_plot(ax2,
        #                   np.arange(theta['delay1'].shape[-1])+1,
        #                   [theta_mean_post,theta_sem_post],precalc = True,
        #                   alpha=.3,
        #                   color=cols[i])
        ax2.errorbar(x=np.arange(AI['delay2'].shape[-1])+1,y=AI_mean_post,
                    yerr=AI_sem_post,color=cols[i])
    
    # ax.legend(ax.get_children()[3:6],labels,bbox_to_anchor=(1,.6),loc='center right')
    ax.legend(bbox_to_anchor=(1,.5),loc='center right')


    ax.set_title('pre-cue delay')
    
    ax2.set_title('post-cue delay')
    ax2.set_xlabel('Time from delay onset')
    
    ax.set_ylabel('AI')
    
    
    plt.tight_layout()
    plt.savefig(constants.PARAMS['FIG_PATH']+'learning_dynamics_plot_AI.svg')


def plot_angle_across_training(constants,theta):
    '''
    Plot angle theta for each timepoint and training stage.

    Parameters
    ----------
    constants : TYPE
        DESCRIPTION.
    theta : dict
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    labels = theta['train_stages']
    
    cols = sns.color_palette("rocket_r",len(labels))
    
    n_timepoints = theta['delay1'].shape[2]
    # plt.figure(figsize=(10,4))
    # ax = plt.subplot(121)
    # ax2 = plt.subplot(122,sharey=ax)
    ring_cols = np.linspace(0,.85,7)
    
    check_path(constants.PARAMS['FIG_PATH']+'theta_pre_lds/')
    check_path(constants.PARAMS['FIG_PATH']+'theta_post_lds/')
    for i in range(len(labels)):
        check_path(constants.PARAMS['FIG_PATH']+'theta_pre_lds/'+labels[i])
        check_path(constants.PARAMS['FIG_PATH']+'theta_post_lds/'+labels[i])
        for t in range(n_timepoints):
            # plot pre-cue
            plot_plane_angles_single(constants,
                                     np.radians(theta['delay1'][:,i,t]),
                                     'pre',r=None)
            ax = plt.gca()
            # change marker colour
            ax.get_children()[0].set_color(cols[i])
            ax.get_children()[1].set_color(cols[i])
            # change ring colour
            ax.get_children()[2].set_color([ring_cols[t]]*3)
            
            if np.logical_and(t>0,t<n_timepoints-1):
                ax.set_xticks([])
            
            figname = constants.PARAMS['FIG_PATH']+'theta_pre_lds/'+labels[i]\
                +'/theta_pre_t'+str(t)+'_'+labels[i]
            plt.savefig(figname+'.png')
            
            
            # plot post-cue
            plot_plane_angles_single(constants,
                                     np.radians(theta['delay2'][:,i,t]),
                                     'post',r=None)
            ax2 = plt.gca()
            # change marker colour
            ax2.get_children()[0].set_color(cols[i])
            ax2.get_children()[1].set_color(cols[i])
            # change ring colour
            ax2.get_children()[2].set_color([ring_cols[t]]*3)
            
            
            if np.logical_and(t>0,t<n_timepoints-1): 
                ax2.set_xticks([])
            figname2 = constants.PARAMS['FIG_PATH']+'theta_post_lds/'+labels[i]\
                +'/theta_post_t'+str(t)+'_'+labels[i]
            plt.savefig(figname2+'.png')
        


def plot_example_loss_plot(constants,AI):
    labels = AI['train_stages']
    
    cols = sns.color_palette("rocket_r",len(labels))
    
    # load training data for an example model
    m = 10
    path = constants.PARAMS['FULL_PATH']+'training_data/'
    track_training = pickle.load(open(path+'training_data_model'+str(m)+'.pckl','rb'))
    lc_plateau = pickle.load(open(path+'lc_plateau.pckl','rb'))
    
    plt.figure(figsize=(6,4))
    plt.plot(track_training['loss_epoch'],'k-')
    
    x = [0,lc_plateau[m][0],len(track_training['loss_epoch'])-1]
    y = track_training['loss_epoch'][x]
    plt.scatter(x,y,c=cols)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    sns.despine(top=True,right=True)
    yl = plt.ylim()
    plt.ylim((0,yl[1]))
    plt.tight_layout()
    plt.savefig(constants.PARAMS['FIG_PATH']+'example_loss_plot.png')
    plt.savefig(constants.PARAMS['FIG_PATH']+'example_loss_plot.svg')

 
    
def plot_r_retro_weights(constants,r):
    plt.figure(figsize=(4.5,7.25))
    sns.stripplot(data=r,orient='v',color='k')
    plt.bar(x=0,height=r.mean(),width = .4,color=[.5,.5,.5])
    plt.plot(np.linspace(-.5,.5,50),np.zeros(50),'k--')
    plt.ylim([-1,.05])
    plt.xlim((-.5,.5))
    plt.xticks([])
    plt.ylabel('Pearson r')
    plt.tight_layout()
    
    plt.savefig(constants.PARAMS['FIG_PATH']+'retro_weights_corr.png')
    

def plot_example_retro_weights(constants):
    m = 0
    constants.PARAMS['model_number'] = m
    cue1_weights, cue2_weights = get_retro_weights(constants)
    r,p_val = pearsonr(cue1_weights,cue2_weights)
    
    plt.figure(figsize=(4.3,4))
    plt.plot(cue1_weights,cue2_weights,'o', mec = 'k', mfc = [0.75,0.75,0.75])
    # sns.regplot(x=cue1_weights.numpy(),y=cue2_weights.numpy(),color='k')
    plt.xlabel('cue 1 weights')
    plt.ylabel('cue 2 weights')   
    
    x = cue1_weights.max()*.1
    y = cue2_weights.max()*.75
    plt.text(x,y,'r = %.3f' %r)
    plt.tight_layout()
    
    #plt.savefig(constants.PARAMS['FIG_PATH']+'example_retro_weights.png')


def plot_all_retro_weights(constants):
    plt.figure()
    for m in range(constants.PARAMS['n_models']):
        # get weights
        constants.PARAMS['model_number'] = m
        cue1_weights, cue2_weights = get_retro_weights(constants)
        
        # plot
        plt.subplot(2,5,m+1)
        plt.plot(cue1_weights,cue2_weights,'o', mec = 'k', mfc = [0.75,0.75,0.75])
    plt.xlabel('cue 1 weights')
    plt.ylabel('cue 2 weights')   
    
    plt.suptitle('Retrocue weights')
    
#%% save into a table
def export_data(constants,cos_theta):
    path = constants.PARAMS['FULL_PATH'] +'pca_data/valid_trials/'
    cos_theta_pre = np.moveaxis(cos_theta['delay1'],-1,1)
    cos_theta_post= np.moveaxis(cos_theta['delay2'],-1,1)
    
    cos_theta_pre = cos_theta_pre.reshape(cos_theta_pre.shape[0],
                                      cos_theta_pre.shape[1]*cos_theta_pre.shape[-1],order='F')
    cos_theta_post = cos_theta_post.reshape(cos_theta_post.shape[0],
                                        cos_theta_post.shape[1]*cos_theta_post.shape[-1],order='F')
    
    # new shape is (model,training_point x time) - so it goes model0: training0time0, 
    # training0time1,training0time3, training0time4, training1time0 etc
    
    training_stages = cos_theta['train_stages']
    timepoints = ['t'+str(t) for t in np.arange(cos_theta['delay1'].shape[-1])]
    
    
    col_labels = list(itertools.product(training_stages,timepoints))
    
    df_pre = pd.DataFrame(data=cos_theta_pre, columns=col_labels)
    df_post = pd.DataFrame(data=cos_theta_post, columns=col_labels)
    
    df_pre.to_csv(path+'/cos_angle_pre_training.csv')
    df_post.to_csv(path+'/cos_angle_post_training.csv')
    
    # split values according to both delay timepoint and training stage factors,
    # for regression analysis

    for t,stage in enumerate(training_stages):
        df_pre = pd.DataFrame(data=cos_theta['delay1'][:,t,:],columns=timepoints)
        df_post = pd.DataFrame(data=cos_theta['delay2'][:,t,:],columns=timepoints)
        
        df_pre.to_csv(path+'/cos_angle_pre_'+stage+'.csv')
        df_post.to_csv(path+'/cos_angle_post_'+stage+'.csv')
        
        
def run_learning_stages_analysis(constants,test_data,device):
    partial_train_and_eval(constants,test_data,device)
    theta, cos_theta, pve = get_angle(constants)
    plot_angle(constants,theta)
    plot_data_shadowplot(constants,pve,'PVE by 3 PCs')
    # plt.savefig(constants.PARAMS['FIG_PATH']+'learning_dynamics_pve_plot.svg')
    plot_example_loss_plot(constants,theta)
    export_data(constants,cos_theta)
    

def run_retro_weight_analysis(constants):
    r,p = corr_retro_weights(constants)
    stat,p_val = test_r_retro_weights(r)
    plot_example_retro_weights(constants)
    
    
    
#%% experiment 4

def check_learning_speed_expt3(params):
    if params['experiment_number'] != 3:
        return
    
    
    # pal = sns.color_palette("dark")
    
    # inds = [3,0]
    # markers = ['o','^']
    # cols = [pal[ix] for ix in inds]
    # ms = 10
    
    n_delay_lengths = 8
    # conditions = ['pre-cue','post-cue']
    delay2_lengths = np.arange(n_delay_lengths)
    
    common_path = params['BASE_PATH']\
        + 'data_vonMises/experiment_4/'\
    
    # n_models = 30
    # all_AI = np.empty((2,len(conditions),n_models,len(delay2_lengths)))
    
    
    end_loss = np.empty((n_delay_lengths,params['n_models']))
    n_epochs_to_conv = np.empty((n_delay_lengths,params['n_models']))
    # plt.figure(figsize=(7,5))
    # ax1 = plt.subplot(111)
    # # ax2 = plt.subplot(122,sharey=ax1,sharex=ax1) 
    # jitter=.125
    for j,dl in enumerate(delay2_lengths):
        # load data
        load_path = common_path + 'delay2_' + str(dl) + 'cycles/'\
                 +'sigma' + str(params['sigma'])\
                    +'/kappa' + str(params['kappa_val'])\
                    +'/nrec' + str(params['n_rec'])\
                        +'/lr' + str(params['learning_rate']) + '/'
        print('Delay 2 length: %d' %dl)
    
    

        for model in range(params['n_models']):
             
            model_number = str(model)
            
            # load data
            f = open(load_path+'training_data/'+'training_data_model'+model_number+'.pckl','rb')
            track_training = pickle.load(f)
            f.close()
        
       
            end_loss[j,model] = track_training['loss_epoch'][-1]
            n_epochs_to_conv[j,model] = track_training['loss_epoch'].shape[0]
    
    
    
    
    labels = np.stack([delay2_lengths]*params['n_models']).T.flatten()
    
    end_loss_df = pd.DataFrame(np.stack((end_loss.flatten(),labels),1),columns=['end loss','delay2_length'])
    end_loss_df.to_csv(common_path+'end_loss_vs_delay2_length.csv')
    
    training_time_df = pd.DataFrame(np.stack((n_epochs_to_conv.flatten(),labels),1),columns=['netc','delay2_length'])
    training_time_df.to_csv(common_path+'training_time_vs_delay2_length.csv')
    
    
    # plot time to convergence vs delay2 length
    sem_netc = n_epochs_to_conv.std(-1) / np.sqrt(params['n_models'])

    pal = sns.color_palette("dark")
    inds = [3,0]
    cols = [pal[ix] for ix in inds]
    ms = 10
    
    plt.figure(figsize=(7,5))
    plt.errorbar(delay2_lengths,n_epochs_to_conv.mean(-1),yerr=sem_netc,color=cols[1])
    plt.plot(delay2_lengths,n_epochs_to_conv.mean(-1),'o',color=cols[1],ms=ms)
    plt.ylabel('time to covergence (n epochs)')
    plt.xticks(delay2_lengths)
    plt.xlabel('Post-cue delay length')
    plt.tight_layout()
    
    plt.savefig(common_path + 'training_speed_vs_delay2_length_sigma_' + str(params['sigma']) + '.png')
    plt.savefig(common_path + 'training_speed_vs_delay2_length_sigma_' + str(params['sigma']) + '.svg')
    
    
    # plot the end loss vs delay2 length
    sem_end_loss = end_loss.std(-1) / np.sqrt(params['n_models'])

    plt.figure(figsize=(7,5))    
    plt.errorbar(delay2_lengths,end_loss.mean(-1),yerr=sem_end_loss,color=cols[1])
    plt.plot(delay2_lengths,end_loss.mean(-1),'o',color=cols[1])
    plt.ylabel('Last epoch loss')
    plt.xticks(delay2_lengths)
    plt.xlabel('Post-cue delay length')
    plt.tight_layout()
    
    plt.savefig(common_path + 'end_loss_vs_delay2_length_sigma_' + str(params['sigma']) + '.png')
    plt.savefig(common_path + 'end_loss_vs_delay2_length_sigma_' + str(params['sigma']) + '.svg')