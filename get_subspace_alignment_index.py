#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 15:12:24 2021


"""

#%% imports
import helpers
import torch
import numpy as np
import scipy.linalg as LA
# import constants
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import shapiro, ttest_1samp, wilcoxon
import pandas as pd



#########
# STILL TO DO - RENAME THE PREPOST ETC IN FIGURE NAMES TO WD/AD
# check the dimensionality of the representations for each condition
##########


#%% Compute alignment indices via PCA

def get_simple_AI(X,Y,max_dim):

    '''
    Computes Alignment index (AI), see Elsayed et al. 2016 
    
    @author: Dante Wasmuht
    
    Parameters
    ----------
    X : 2D array
        Data matrix of the format: (conditions,neurons) or (samples,features)
    Y : 2D array
        Analogous data array for the other experimental condition.
    max_dim : int
        Dimensionality for the calculated subspaces.

    Returns
    -------
    AI : scalar
        Subspace AI value between the max_dim-dimensional subspaces of datasets
        X and Y.
    
    '''
    
   
    # de-mean data matrices
    X_preproc = (X - X.mean(0)[None,:]).T
    Y_preproc = (Y - Y.mean(0)[None,:]).T
   
    # get covariance matrices
    c_mat_x = np.cov(X_preproc)
    c_mat_y = np.cov(Y_preproc)
   
    # perform eigenvalue decomposition on covariance matrices
    eig_vals_x, eig_vecs_x = LA.eigh(c_mat_x, eigvals=(c_mat_x.shape[0] - max_dim, c_mat_x.shape[0]-1))
    eig_vals_y, eig_vecs_y = LA.eigh(c_mat_y, eigvals=(c_mat_y.shape[0] - max_dim, c_mat_y.shape[0]-1))
   
    # sort eigenvectors according to eigenvalues (descending order)
    eig_vecs_x = eig_vecs_x[:, np.argsort(np.real(eig_vals_x))[::-1]]
    eig_vals_x = eig_vals_x[::-1]
   
    eig_vecs_y = eig_vecs_y[:, np.argsort(np.real(eig_vals_y))[::-1]]
    eig_vals_y = eig_vals_y[::-1]
   
    # compute the alignment index: 1 = maximal subspaces overlap; 0 = subspaces are maximally orthogonal
        # In the numerator the data from Y is projected onto the X subspace and the variance of Y in X subspace is calculated
        # the denominator contains the variance of Y in subspace Y (...not the full space! > Although this could be an option too...)
    ai_Y_in_X = np.trace(np.dot(np.dot(eig_vecs_x.T,c_mat_y),eig_vecs_x)) / np.sum(eig_vals_y)
    ai_X_in_Y = np.trace(np.dot(np.dot(eig_vecs_y.T,c_mat_x),eig_vecs_y)) / np.sum(eig_vals_x)
   
    return (ai_Y_in_X + ai_X_in_Y)/2


def get_trial_ixs(params,cv=2):
    '''
    Split the data into cross-validation folds and get the corresponding trial 
    indices.

    Parameters
    ----------
    params : dict
        Experiment parameters.
    cv : int, optional
        Number of cross-validation folds. The default is 2.

    Returns
    -------
    train : array
        training sets indices.
    test : array
        testing sets indices.

    '''
    n_samples = params['n_trial_types']*params['n_trial_instances_test']
    trial_labels = [np.arange(params['n_trial_types'])]*params['n_trial_instances_test']
    trial_labels = np.stack(trial_labels,1).reshape(-1)
    
    
    skf = StratifiedKFold(n_splits=cv,shuffle=True)
    train,test = skf.split(np.zeros(n_samples),trial_labels)
    
    return train,test


def check_intrinsic_dim(params,custom_path=[],n_PCs=4):
    '''
    Check the intrinsic dimensionality of the dataset to determine the required
    dimensionality for the AI analysis. Print the total proportion of variance
    explained (PVE) by 2 PCs at the endpoint of each delay.

    Parameters
    ----------
    params : dict
        Experiment parameters.
    custom_path : str, optional
        Alternative datapath. The default is [], which corresponds to the pca 
        data structure for cued items on valid trials.
    n_PCs: int, optional
        Maximum number of PCs to use in calculating PVE values. Default is 4, 
        which is the maximum dimensionality of the dataset containing activation
        patterns to 4 colour bins shown at a single location.
    Returns
    -------
    csum_PVE : array
        Total PVE by increasing numbers of PCs.

    '''
    def run_pca(data):
        # center data
        data_centered = data - data.mean()
        
        # do PCA
        pca = PCA()
        pca.fit(data_centered)
        
        # get PVEs
        PVE = pca.explained_variance_ratio_
        return PVE
    
    # determine datapath
    if not len(custom_path):
        load_path = params['FULL_PATH'] + 'pca_data/valid_trials'
    else:
        load_path = custom_path
            
    if params['experiment_number'] == 3:
        n_timepoints = 3
    else:
        n_timepoints = 2
    
    
    csum_PVE = np.zeros((n_PCs,n_timepoints,params['n_models']))

    for model in range(params['n_models']):
        # load data
        model_number = str(model)
        f = open(load_path+'/pca_data_model' + model_number + '.pckl', 'rb')
        pca_data = pickle.load(f)    
        delay1 = pca_data['delay1'] #condition x neurons
        delay2 = pca_data['delay2']

        # f = open(load_path+'/trial_type_subspaces_model' + model_number + '.pckl', 'rb')
        # pca_data = pickle.load(f)    
        # delay1 = pca_data['cued_up']['binned_data'] #condition x neurons
        # delay2 = pca_data['cued_down']['binned_data']
        # f.close()
        
        half_split = delay1.shape[0]//2 # get the last index of loc1 datapoints
        
        # get the PVE values for data from each location and timepoint
        pre_PVEs = np.stack((run_pca(delay1[:half_split,:]),
                    run_pca(delay1[half_split:,:]))).mean(0)
        post_PVEs = np.stack((run_pca(delay2[:half_split,:]),
                     run_pca(delay2[half_split:,:]))).mean(0)
        
        csum_PVE[:,0,model] = np.cumsum(pre_PVEs)[:n_PCs] 
        csum_PVE[:,1,model] = np.cumsum(post_PVEs)[:n_PCs]
        
        if params['experiment_number'] == 3:
            probe = pca_data['data'][:,-1,:]
            probe_PVEs = np.stack((run_pca(probe[:half_split,:]),
                     run_pca(probe[half_split:,:]))).mean(0)
            csum_PVE[:,2,model] = np.cumsum(probe_PVEs)[:n_PCs]
            
    # check how much variance is captured by the first 2PCs across all models
    means_2PCs = csum_PVE[1,:,:].mean(-1)
    sems_2PCs = np.std(csum_PVE[1,:,:],-1)/np.sqrt(params['n_models'])
    
    print('Variance explained by 2PCs in the pre-cue delay: mean = %.4f, SEM = %.4f'
          %(means_2PCs[0],sems_2PCs[0]))
    print('Variance explained by 2PCs in the post-cue delay: mean = %.4f, SEM = %.4f'
          %(means_2PCs[1],sems_2PCs[1]))
    
    if params['experiment_number'] == 3:
        print('Variance explained by 2PCs at the probe timepoint: mean = %.4f, SEM = %.4f'
          %(means_2PCs[2],sems_2PCs[2]))
    return csum_PVE


def print_descriptives(AI_tbl,comparison):
    '''
    Print descriptive statistics for an AI table. If the table is 2D, print the
    statistics (M+SEM)for a given timepoint for all dimensionalities explored.
    If the table is 3D, for each dimesionality, print the statistics for each 
    timepoint along with its label given by the 'comparison' argument.

    Parameters
    ----------
    AI_tbl : array
        Table with AI values, format : (n_dims,n_timepoints,n_models)
        or (n_dims,n_models)
    comparison : list
        Labels for different conditions the AI was calculated for, e.g. names of
        the delays.

    Returns
    -------
    None.

    '''
    if len(AI_tbl.shape)<3:
        # if want to report the mean and SEM for some timepoint - pass a 2D table
        for dim in range(AI_tbl.shape[0]):
            sem = np.std(AI_tbl[dim,:])/np.sqrt(AI_tbl.shape[-1])
            s = '          '
            s += f'AI {dim+2}: mean = {AI_tbl[dim,:].mean():.3f}, '+\
               f'sem  = {sem:.3f}'
            print(s)   
    else:
        # if want to compare values across timepoints
        for dim in range(AI_tbl.shape[0]):
            s = '          '
            s += f'AI {dim+2}: '
            for c in len(comparison):
                s += f'mean {comparison[c]} = {AI_tbl[dim,c,:].mean():.3f}, '
            print(s)
        
        # inferential stats
        # print('2D wilcoxon test')
        # stat, p_val = wilcoxon(AI_table_prepost[0,1,:]-AI_table_prepost[0,0,:],
        #                        alternative='greater')
        # print('        stat = %.3f, p = %.3f' %(stat,p_val))
        
        # #%%
        # print('AI 2: mean_pre = %.3f, mean_post = %.3f' %(means_prepost[1,0],means_prepost[1,1]))
        
        
        # print('3D wilcoxon test')
        # stat, p_val = wilcoxon(AI_table_prepost[1,1,:]-AI_table_prepost[1,0,:],
        #                        alternative='greater')
        # print('        stat = %.3f, p = %.3f' %(stat,p_val))

    return


def plot_AI(params,AI_table_prepost,opt,binning):
    '''
    Plot AI. Average across group depicted as a bar alongside individual datapoints.

    Parameters
    ----------
    params : dict
        DESCRIPTION.
    AI_table_prepost : TYPE
        DESCRIPTION.
    opt : TYPE
        DESCRIPTION.
    binning : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    print('add the validity paradigm option here')
    if opt == 'prepost':
        pal = sns.color_palette("dark")
        inds = [3,0]
        cols = [pal[ix] for ix in inds]
        alpha1, alpha2 = .2, .2
        labels = ['pre-cue', 'post-cue']
    elif opt == 'unrotrot':
        cols = ['k','k']
        alpha1, alpha2 = .2, .4
        labels = ['unrotated', 'rotated']
        # labels = ['L1 cued', 'L2 cued']
    ms = 16
    
    if np.logical_and(opt == 'prepost', binning == 'cued'):
        plt.figure(figsize=(6.65,5),num='AI prepost cued')
    else:
        plt.figure(figsize=(6.65,5), num = 'AI '+opt+' '+binning)
    
    n_dims = AI_table_prepost.shape[0]
    max_dim = n_dims+1
    
    jitter = 0.25/2
    means_prepost = np.zeros((n_dims,2))
    
        
    for dim in range(n_dims):
        for model in range(params['n_models']):
            plt.plot([dim+2-jitter,dim+2+jitter],AI_table_prepost[dim,:,model],'k-',alpha=.2)
            plt.plot(dim+2-jitter,AI_table_prepost[dim,0,model],'o',c=cols[0],markersize=ms)
            plt.plot(dim+2+jitter,AI_table_prepost[dim,1,model],'o',c=cols[1],markersize=ms)
    
        # add means
        means_prepost = AI_table_prepost[dim,:,:].mean(-1)
        
        if dim == n_dims-1:
            # plot means as bars add labels
            plt.bar(dim+2-jitter,means_prepost[0],facecolor=cols[0],
                                                    alpha=alpha1,width=.25,
                                                    label = labels[0])
            plt.bar(dim+2+jitter,means_prepost[1],facecolor=cols[1],
                                                    alpha=alpha2,width=.25,
                                                    label=labels[1])
        else:
            plt.bar(dim+2-jitter,means_prepost[0],facecolor=cols[0],alpha=alpha1,width=.25)
            plt.bar(dim+2+jitter,means_prepost[1],facecolor=cols[1],alpha=alpha2,width=.25)
    
    
    plt.xticks(range(2,max_dim+1))
    plt.xlabel('Dimensionality')
    plt.ylabel('AI')
    plt.ylim([0,1.1])
    
    plt.legend(bbox_to_anchor=(1,1))
    
    plt.tight_layout()
    
    return 
#%% calculate the AI for cued subspaces
def get_AI_within_delay(params,custom_path=[],cued=True):
    '''
    Calculate the AI values between the Cued subspaces in the same delay. Can 
    also be used to get the same for the Uncued subspaces.

    Parameters
    ----------
    params : TYPE
        DESCRIPTION.
    custom_path : TYPE, optional
        DESCRIPTION. The default is [].
    cued : bool, optional
        Whether to calculate the AI using data for cued colours (i.e. averaged
        across uncued). The default is True. If False, caculates the AI for 
        Uncued subspaces.

    Returns
    -------
    AI_table_wd : array
        AI values, format: (n_dims,n_timepoints,n_models)

    '''
    if not len(custom_path):
        load_path = params['FULL_PATH'] + 'pca_data/valid_trials'
    else:
        load_path = custom_path
    
    # set the maximum dimensionality for the calculated subspaces
    max_dim = 3
    n_dims = max_dim+1-2
    
    if params['experiment_number'] == 3:
        # include the post-probe delay
        n_timepoints = 3
    else:
        # only look at the pre-/post-retrocue delays
        n_timepoints = 2
    AI_table_wd = np.zeros((n_dims,n_timepoints,params['n_models']))
    

    for model in range(params['n_models']):
        model_number = str(model)
        if cued:
            # load the cued dataset
            f = open(load_path+'/pca_data_model' + model_number + '.pckl', 'rb')
        else:
            # load the uncued dataset
            f = open(load_path+'/pca_data_uncued_model' + model_number + '.pckl', 'rb')
            
        pca_data = pickle.load(f)    
        delay1 = pca_data['delay1'] #condition x neurons
        delay2 = pca_data['delay2']
        if params['experiment_number'] == 3:
            probe = pca_data['data'][:,-1,:]
        f.close()
        
        half_split = delay1.shape[0]//2 # get the ix of the last loc1 datapoint
        
        # calculate the AI for different dimensionalities
        for max_dim in range(2,max_dim+1):
            # AI pre-cue
            AI_table_wd[max_dim-2,0,model]= get_simple_AI(delay1[:half_split,:],
                                                           delay1[half_split:,:],max_dim) 
            # AI post-cue
            AI_table_wd[max_dim-2,1,model]=get_simple_AI(delay2[:half_split,:],
                                                          delay2[half_split:,:],max_dim) 
            if params['experiment_number'] == 3:
                # AI post-probe
                AI_table_wd[max_dim-2,2,model]=get_simple_AI(probe[:half_split,:],
                                                              probe[half_split:,:],max_dim) 
    return AI_table_wd


def get_AI_tbl(params,data1,data2,max_dim=3):
    n_dims = max_dim+1-2
    AI_table = np.zeros((n_dims,))
    
    for max_dim in range(2,max_dim+1):
            AI_table[max_dim-2]= get_simple_AI(data1,data2,max_dim)
    
    return AI_table
# def get_AI_cued_within_delay(params,custom_path=[]):
#     '''
    

#     Parameters
#     ----------
#     params : TYPE
#         DESCRIPTION.
#     custom_path : TYPE, optional
#         DESCRIPTION. The default is [].

#     Returns
#     -------
#     AI_table_wd : TYPE
#         DESCRIPTION.

#     '''
#     if not len(custom_path):
#         load_path = params['FULL_PATH'] + 'pca_data/valid_trials'
#     else:
#         load_path = custom_path
        
#     max_dim = 3
#     n_dims = max_dim+1-2
    
#     if params['experiment_number'] == 3:
#         # include the post-probe delay
#         n_timepoints = 3
#     else:
#         # only look at the pre-/post-retrocue delays
#         n_timepoints = 2
#     AI_table_wd = np.zeros((n_dims,n_timepoints,params['n_models']))
    

#     for model in range(params['n_models']):
#         model_number = str(model)
#         f = open(load_path+'/pca_data_model' + model_number + '.pckl', 'rb')
#         pca_data = pickle.load(f)    
#         delay1 = pca_data['delay1'] #condition x neurons
#         delay2 = pca_data['delay2']
#         if params['experiment_number'] == 3:
#             probe = pca_data['data'][:,-1,:]
#         f.close()
        
#         half_split = delay1.shape[0]//2
    
#         for max_dim in range(2,max_dim+1):
            
#             AI_table_wd[max_dim-2,0,model]= get_simple_AI(delay1[:half_split,:],
#                                                            delay1[half_split:,:],max_dim) # AI pre
#             AI_table_wd[max_dim-2,1,model]=get_simple_AI(delay2[:half_split,:],
#                                                           delay2[half_split:,:],max_dim) # AI post
#             if params['experiment_number'] == 3:
#                 AI_table_wd[max_dim-2,2,model]=get_simple_AI(probe[:half_split,:],
#                                                               probe[half_split:,:],max_dim) # AI post
#     return AI_table_wd


# def get_AI_cued_probe(params,custom_path=[]):
#     if not len(custom_path):
#         load_path = params['FULL_PATH'] + 'pca_data/valid_trials'
#     else:
#         load_path = custom_path
#     max_dim = 3
#     n_dims = max_dim+1-2

#     AI_table_probe = np.zeros((n_dims,params['n_models']))
#     # AI_table_updown = np.zeros((n_dims,2,constants.PARAMS['n_models']))
    
#     # csum_PVE = np.zeros((n_dims,2,constants.PARAMS['n_models']))

#     for model in range(params['n_models']):
#         model_number = str(model)
#         f = open(load_path+'/pca_data_model' + model_number + '.pckl', 'rb')
#         pca_data = pickle.load(f)    
#         probe = pca_data['data'][:,-1,:]
#         f.close()
        
#         half_split = probe.shape[0]//2
    
#         for max_dim in range(2,max_dim+1):
            
#             AI_table_probe[max_dim-2,model]= get_simple_AI(probe[:half_split,:],
#                                                            probe[half_split:,:],max_dim)
#     return AI_table_probe
    
            
#%% do updown in x-val

def get_AI_cued_across_delays(params,cv=2):
    '''
    For each location, get the AI between its subspaces from the 
    pre- and post-cue delay intervals. Do it in 2-fold cross-validation to 
    determine if both or only one of the pre-cue subspaces are are rotated post-cue
    to form a parallel plane geometry.

    Parameters
    ----------
    params : dict
        Experiment parameters.
    cv : int, optional
        Number of cross-validation folds. The default is 2.

    Returns
    -------
    AI_tbl : array
        AI values for the 'unrotated' and 'rotated' planes, averaged across cv 
        folds. Format: (n_dims,(unrotated,rotated),model)
    same_ixs : array
        Indexes of the unrotated plane for each model.
    trial_ixs : dict
        Train and test trial indexes for the cross-validation folds.

    '''
    load_path = params['FULL_PATH'] + 'pca_data/valid_trials'
    max_dim = 3
    n_dims = max_dim-1
    
    AI_table_ad_train = np.zeros((n_dims,2,params['n_models'],cv))
    AI_table_ad_test = np.zeros((n_dims,2,params['n_models'],cv))
    
    same_ixs = np.zeros((n_dims,params['n_models'],2))
    
    d1_ix = params['trial_timepoints']['delay1_end']-1
    d2_ix = params['trial_timepoints']['delay2_end']-1
    
    trial_ixs = {'train':{},'test':{}}
    
    for model in range(params['n_models']):
        # load data
        model_number = str(model)
        f = open(load_path+'/eval_data_model' + model_number + '.pckl', 'rb')
        eval_data = pickle.load(f)    
        f.close()
        
        train,test = get_trial_ixs(params,cv=2)
        trial_ixs['train'][model_number],trial_ixs['test'][model_number] = \
            train, test
        
        
        loc_split = params['B']
        for i in range(cv):
            # bin the train and test datasets
            data_train = helpers.bin_data(eval_data['data'][train[i],:,:],params)
            data_test = helpers.bin_data(eval_data['data'][test[i],:,:],params)
            
            delay1_train = data_train[:,d1_ix,:]
            delay1_test = data_test[:,d1_ix,:]
            
            delay2_train = data_train[:,d2_ix,:]
            delay2_test = data_test[:,d2_ix,:]
            
            # calculate the AI on train data 
            for dim in range(2,max_dim+1):
                AI_table_ad_train[dim-2,0,model,i]=get_simple_AI(delay1_train[:loc_split,:],
                                                             delay2_train[:loc_split,:],max_dim) #loc1
                AI_table_ad_train[dim-2,1,model,i]=get_simple_AI(delay1_train[loc_split:,:],
                                                             delay2_train[loc_split:,:],max_dim) #loc2
            
            # find the plane that stays the same
            same_ixs[:,model,i] = np.argmax(AI_table_ad_train[:,:,model,i],1)
            
            # calculate the AI on test data
            for dim in range(2,max_dim+1):
                if same_ixs[dim-2,model,i] == 0:
                    stay_plane_ix = np.arange(loc_split)
                else:
                    stay_plane_ix = np.arange(loc_split,loc_split*2)
                switch_plane_ix = np.setdiff1d(np.arange(loc_split*2),stay_plane_ix)
                
                AI_table_ad_test[dim-2,0,model,i]=get_simple_AI(delay1_test[stay_plane_ix,:],
                                                             delay2_test[stay_plane_ix,:],dim) #stay plane
                AI_table_ad_test[dim-2,1,model,i]=get_simple_AI(delay1_test[switch_plane_ix,:],
                                                             delay2_test[switch_plane_ix,:],dim) #switch plane
            
    return AI_table_ad_test.mean(-1), same_ixs, trial_ixs












#%% plot up down

# plot_AI(AI_table_updown,'updown','cued')

# plt.savefig(constants.PARAMS['FIG_PATH']+'AI_updown_cued.png')


# pal = sns.color_palette("dark")
# inds = [3,0]
# cols = [pal[ix] for ix in inds]
# ms = 16

# plt.figure(figsize=(10,10))

# jitter = 0.25/2
# means_updown= np.zeros((n_dims,2))

    

# for dim in range(n_dims):
#     for model in range(constants.PARAMS['n_models']):
#         plt.plot([dim+2-jitter,dim+2+jitter],AI_table_updown[dim,:,model],'k-',alpha=.2)
#         plt.plot(dim+2-jitter,AI_table_updown[dim,0,model],'o',c='k',markersize=ms)
#         plt.plot(dim+2+jitter,AI_table_updown[dim,1,model],'o',c='k',markersize=ms)

#     # add means
#     means_updown[dim,:] = np.mean(AI_table_updown[dim,:,:],-1)
    
#     if dim == n_dims-1:
#         # add labels
#         plt.bar(dim+2-jitter,means_updown[dim,0],facecolor='k',
#                                                 alpha=.2,width=.25,
#                                                 label = 'loc1')
#         plt.bar(dim+2+jitter,means_updown[dim,1],facecolor='k',
#                                                 alpha=.4,width=.25,
#                                                 label='loc2')
#     else:
#             plt.bar(dim+2-jitter,means_updown[dim,0],facecolor='k',alpha=.2,width=.25)
#             plt.bar(dim+2+jitter,means_updown[dim,1],facecolor='k',alpha=.4,width=.25)


# plt.xticks(range(2,max_dim+1))
# plt.xlabel('Dimensionality')
# plt.ylabel('AI')
# plt.ylim([-.1,1])

# plt.legend()

# plt.tight_layout()
        
# #%% descriptive stats

# mean_AI2 = np.mean(AI_table_updown[0,:,:])
# CI_AI2 = 1.98 * (np.std(np.mean(AI_table_updown[0,:,:],0))/np.sqrt(constants.PARAMS['n_models']))
# mean_AI3 = np.mean(AI_table_updown[1,:,:])
# CI_AI3 = 1.98 * (np.std(np.mean(AI_table_updown[1,:,:],0))/np.sqrt(constants.PARAMS['n_models']))

#%% do the same analysis for uncued - are the uncued colour subspaces transformed from pre- to post-cue?
#also,  how are the two uncued subspaces arranged with respect to each other pre- and post-cue?
# note precue, there is no 'uncued' subspace, there is only a location one


# def get_AI_uncued_within_delay(params,custom_path=[]):
#     if not len(custom_path):
#         load_path = params['FULL_PATH'] + 'pca_data/valid_trials'
#     else:
#         load_path = custom_path
#     max_dim = 3
#     n_dims = max_dim+1-2
    

#     AI_table_wd = np.zeros((n_dims,2,params['n_models']))

#     for model in range(params['n_models']):
#         model_number = str(model)
#         f = open(load_path+'/pca_data_uncued_model' + model_number + '.pckl', 'rb')
#         pca_data_uncued = pickle.load(f)    
#         delay1 = pca_data_uncued['delay1'] #condition x neurons
#         delay2 = pca_data_uncued['delay2']
#         f.close()
        
#         half_split = delay1.shape[0]//2
#         for max_dim in range(2,max_dim+1):
            
#             AI_table_wd[max_dim-2,0,model]= get_simple_AI(delay1[:half_split,:],
#                                                            delay1[half_split:,:],max_dim) # AI pre
#             AI_table_wd[max_dim-2,1,model]=get_simple_AI(delay2[:half_split,:],
#                                                           delay2[half_split:,:],max_dim) # AI post

#     return AI_table_wd


# def get_AI_uncued_within_delay_probe(params,custom_path=[]):
#     if not len(custom_path):
#         load_path = params['FULL_PATH'] + 'pca_data/valid_trials'
#     else:
#         load_path = custom_path
#     max_dim = 3
#     n_dims = max_dim+1-2
    

#     AI_table_wd = np.zeros((n_dims,params['n_models']))

#     for model in range(params['n_models']):
#         model_number = str(model)
#         f = open(load_path+'/pca_data_uncued_model' + model_number + '.pckl', 'rb')
#         pca_data_uncued = pickle.load(f)    
#         probe = pca_data_uncued['data'][:,-1,:]
#         f.close()
        
#         half_split = probe.shape[0]//2
#         for max_dim in range(2,max_dim+1):
            
#             AI_table_wd[max_dim-2,model]= get_simple_AI(probe[:half_split,:],
#                                                            probe[half_split:,:],max_dim) # AI probe
           
#     return AI_table_wd
    






#%% plot up down

pal = sns.color_palette("dark")
# inds = [3,0]
# cols = [pal[ix] for ix in inds]
# ms = 16

# plt.figure(figsize=(10,10))

# jitter = 0.25/2
# means_updown= np.zeros((n_dims,2))

    

# for dim in range(n_dims):
#     for model in range(constants.PARAMS['n_models']):
#         plt.plot([dim+2-jitter,dim+2+jitter],AI_table_updown[dim,:,model],'k-',alpha=.2)
#         plt.plot(dim+2-jitter,AI_table_updown[dim,0,model],'o',c='k',markersize=ms)
#         plt.plot(dim+2+jitter,AI_table_updown[dim,1,model],'o',c='k',markersize=ms)

#     # add means
#     means_updown[dim,:] = np.mean(AI_table_updown[dim,:,:],-1)
    
#     if dim == n_dims-1:
#         # add labels
#         plt.bar(dim+2-jitter,means_updown[dim,0],facecolor='k',
#                                                 alpha=.2,width=.25,
#                                                 label = 'loc1')
#         plt.bar(dim+2+jitter,means_updown[dim,1],facecolor='k',
#                                                 alpha=.4,width=.25,
#                                                 label='loc2')
#     else:
#             plt.bar(dim+2-jitter,means_updown[dim,0],facecolor='k',alpha=.2,width=.25)
#             plt.bar(dim+2+jitter,means_updown[dim,1],facecolor='k',alpha=.4,width=.25)


# plt.xticks(range(2,max_dim+1))
# plt.xlabel('Dimensionality')
# plt.ylabel('AI')
# plt.ylim([-.1,1])

# plt.legend()

# plt.tight_layout()


# AI_table_cueduncued = np.squeeze(AI_table_cueduncued)
#%% print descriptives and stats
# mean_across_both_locations = np.mean(np.squeeze(AI_table_cueduncued),0)

# from scipy.stats import shapiro, ttest_1samp, wilcoxon
# sw, p = shapiro(mean_across_both_locations)


# grand_mean = np.mean(mean_across_both_locations)
# CI95 = (np.std(mean_across_both_locations)/np.sqrt(constants.PARAMS['n_models']))*1.98
# print('AI cued vs uncued: mean = %.3f, 95 CI: = %.3f' %(grand_mean,CI95))

# if p<0.05:
#     print('    wilcoxon test')
#     stat, p_val = wilcoxon(mean_across_both_locations)
# else:
#     print('    one-sample t-test')
#     stat, p_val = ttest_1samp(mean_across_both_locations,0)
# print('        stat = %.3f, p = %.3f' %(stat,p_val/2))

#%% plot

# pal = sns.color_palette("dark")
# inds = [3,0]
# cols = [pal[ix] for ix in inds]
# ms = 16

# plt.figure(figsize=(10,10))

# jitter = 0.25/2
# means_cueduncued= np.zeros((n_dims,2))

    

# for dim in range(n_dims):
#     for model in range(constants.PARAMS['n_models']):
#         plt.plot([dim+2-jitter,dim+2+jitter],AI_table_cueduncued[dim,:,model],'k-',alpha=.2)
#         plt.plot(dim+2-jitter,AI_table_cueduncued[dim,0,model],'o',c='k',markersize=ms)
#         plt.plot(dim+2+jitter,AI_table_cueduncued[dim,1,model],'o',c='k',markersize=ms)

#     # add means
#     means_cueduncued[dim,:] = np.mean(AI_table_cueduncued[dim,:,:],-1)
    
#     if dim == n_dims-1:
#         # add labels
#         plt.bar(dim+2-jitter,means_cueduncued[dim,0],facecolor='k',
#                                                 alpha=.2,width=.25,
#                                                 label = 'cued loc1')
#         plt.bar(dim+2+jitter,means_cueduncued[dim,1],facecolor='k',
#                                                 alpha=.4,width=.25,
#                                                 label='cued loc2')
#     else:
#             plt.bar(dim+2-jitter,means_cueduncued[dim,0],facecolor='k',alpha=.2,width=.25)
#             plt.bar(dim+2+jitter,means_cueduncued[dim,1],facecolor='k',alpha=.4,width=.25)


# plt.xticks(range(2,max_dim+1))
# plt.xlabel('Dimensionality')
# plt.ylabel('AI')
# plt.ylim([-.1,1])

# plt.legend()

# plt.tight_layout()


#%% do uncued up-down in x-val

def get_AI_uncued_across_delays(params,same_ixs,trial_ixs):
    load_path = params['FULL_PATH'] + 'pca_data/valid_trials'
    max_dim = 3
    n_dims = max_dim+1-2
    
    AI_tbl_ad = np.zeros((n_dims,2,params['n_models'],2))
    
    
    d1_ix = params['trial_timepoints']['delay1_end']-1
    d2_ix = params['trial_timepoints']['delay2_end']-1

    cv = same_ixs.shape[-1]
    
    for model in range(params['n_models']):
        # load data
        model_number = str(model)
        f = open(load_path+'/eval_data_model' + model_number + '.pckl', 'rb')
        eval_data = pickle.load(f)    
        f.close()
        
        # resort by uncued stimulus
        eval_data = helpers.sort_by_uncued(eval_data,params) 
        test = trial_ixs['test'][model_number]
        
        loc_split = params['B']
        
        for i in range(cv):
            # bin the train and test datasets
            data_test = helpers.bin_data(eval_data['data'][test[i],:,:],params)
            
            delay1_test = data_test[:,d1_ix,:]
            
            delay2_test = data_test[:,d2_ix,:]
            
            # calculate the AI on train data 
            # for dim in range(2,max_dim+1):
            #     AI_table_train[dim-2,0,model,i]=get_simple_AI(delay1_train[:loc_split,:],
            #                                                  delay2_train[:loc_split,:],max_dim) #loc1
            #     AI_table_train[dim-2,1,model,i]=get_simple_AI(delay1_train[loc_split:,:],
            #                                                  delay2_train[loc_split:,:],max_dim) #loc2
            
            # find the plane that stays the same
            
            
            # calculate the AI on test data
            for dim in range(2,max_dim+1):
                if same_ixs[dim-2,model,i] == 0:
                    # uncued data sorted into [uncued-loc2;uncued-loc1]
                    # so if loc 1 plane corresponds to the bottom half of the 
                    # data matrix
                    stay_plane_ix = np.arange(loc_split,loc_split*2)
                else:
                    stay_plane_ix = np.arange(loc_split)
                switch_plane_ix = np.setdiff1d(np.arange(loc_split*2),stay_plane_ix)
                
                AI_tbl_ad[dim-2,0,model,i]=get_simple_AI(delay1_test[stay_plane_ix,:],
                                                             delay2_test[stay_plane_ix,:],dim) #stay plane
                AI_tbl_ad[dim-2,1,model,i]=get_simple_AI(delay1_test[switch_plane_ix,:],
                                                             delay2_test[switch_plane_ix,:],dim) #switch plane
            
    return AI_tbl_ad.mean(-1)
    
                                
def get_AI_cued_vs_uncued(params,trial_type='valid',custom_path = []):
    # how are the two uncued subspaces arranged with respect to the cued ones post-cue?
    # are they in the null space?
    if not len(custom_path):
        load_path = params['FULL_PATH'] + 'pca_data/'+trial_type+'_trials'
    else:
        load_path = custom_path
    
    max_dim = 3
    n_dims = max_dim+1-2
    
    AI_tbl_cu = np.zeros((n_dims,2,params['n_models']))
    
    for model in range(params['n_models']):
        model_number = str(model)

        # load data
        # cued
        f = open(load_path+'/pca_data_model' + model_number + '.pckl', 'rb')
        pca_data_cued = pickle.load(f)
        delay2_cued = pca_data_cued['delay2']
        
        # uncued
        f = open(load_path+'/pca_data_uncued_model' + model_number + '.pckl', 'rb')
        pca_data_uncued = pickle.load(f)    
        delay2_uncued = pca_data_uncued['delay2']
        f.close()
        
        if params['experiment_number'] == 3:       
            probe_cued = pca_data_cued['data'][:,-1,:]
            probe_uncued = pca_data_uncued['data'][:,-1,:]

        loc_split = delay2_uncued.shape[0]//2
        
    
        for max_dim in range(2,max_dim+1):
            AI_tbl_cu[max_dim-2,0,model]= get_simple_AI(delay2_cued[:loc_split,:],delay2_uncued[:loc_split,:],max_dim) # AI cued up
            AI_tbl_cu[max_dim-2,1,model]=get_simple_AI(delay2_cued[loc_split:,:],delay2_uncued[loc_split:,:],max_dim) # AI cued down
           
    return AI_tbl_cu


def get_AI_cued_vs_uncued_probe(params,custom_path=[]):
    # how are the two uncued subspaces arranged with respect to the cued ones post-cue?
    # are they in the null space?
    if not len(custom_path):
        load_path = params['FULL_PATH'] + 'pca_data/valid_trials'
    else:
        load_path = custom_path
    max_dim = 3
    n_dims = max_dim+1-2
    
    AI_tbl_cu_probe = np.zeros((n_dims,params['n_models']))
    
    for model in range(params['n_models']):
        model_number = str(model)

        # load data
        # cued
        f = open(load_path+'/pca_data_model' + model_number + '.pckl', 'rb')
        pca_data_cued = pickle.load(f)
        probe_cued = pca_data_cued['data'][:,-1,:]
        
        # uncued
        f = open(load_path+'/pca_data_uncued_model' + model_number + '.pckl', 'rb')
        pca_data_uncued = pickle.load(f)    
        probe_uncued = pca_data_uncued['data'][:,-1,:]
        f.close()


        loc_split = probe_uncued.shape[0]//2
        
    
        for max_dim in range(2,max_dim+1):
            AI_tbl_cu_probe[max_dim-2,model]= get_simple_AI(probe_cued[:loc_split,:],probe_uncued[:loc_split,:],max_dim) # AI cued up
            AI_tbl_cu_probe[max_dim-2,model]=get_simple_AI(probe_cued[loc_split:,:],probe_uncued[loc_split:,:],max_dim) # AI cued down
           
    return AI_tbl_cu_probe



def get_AI_cued_vs_uncued_same_location(params,trial_type='valid'):
    # how are the two cued and uncued subspaces for a given location arranged 
    # wrt to each other?
    load_path = params['FULL_PATH'] + 'pca_data/'+trial_type+'_trials'
    max_dim = 3
    n_dims = max_dim+1-2
    
    AI_tbl_cu = np.zeros((n_dims,2,params['n_models']))
    
    for model in range(params['n_models']):
        model_number = str(model)

        # load data
        # cued
        f = open(load_path+'/pca_data_model' + model_number + '.pckl', 'rb')
        pca_data_cued = pickle.load(f)
        delay2_cued = pca_data_cued['delay2']
        probe_cued = pca_data_cued['data'][:,-1,:]
        
        # uncued
        f = open(load_path+'/pca_data_uncued_model' + model_number + '.pckl', 'rb')
        pca_data_uncued = pickle.load(f)    
        delay2_uncued = pca_data_uncued['delay2']
        f.close()
        probe_uncued = pca_data_uncued['data'][:,-1,:]
        
        loc_split = delay2_uncued.shape[0]//2
        
    
        for max_dim in range(2,max_dim+1):
            # AI_tbl_cu[max_dim-2,0,model]= get_simple_AI(delay2_cued[:loc_split,:],delay2_uncued[:loc_split,:],max_dim) # AI cued up
            # AI_tbl_cu[max_dim-2,1,model]=get_simple_AI(delay2_cued[loc_split:,:],delay2_uncued[loc_split:,:],max_dim) # AI cued down
           
            AI_tbl_cu[max_dim-2,0,model]= get_simple_AI(delay2_cued[:loc_split,:],delay2_uncued[loc_split:,:],max_dim) # AI cued up
            AI_tbl_cu[max_dim-2,1,model]= get_simple_AI(delay2_cued[loc_split:,:],delay2_uncued[:loc_split,:],max_dim)
    return AI_tbl_cu
    
    
    
#% concatenate data structures for export

def export_AI_data(params,AI_tbl_cued_wd,AI_tbl_cued_ad,AI_tbl_uncued_wd,AI_tbl_cu):
    print('          export data to file')
    big_AI_tbl = np.concatenate((AI_tbl_cued_wd,
                                 AI_tbl_cued_ad,
                                 AI_tbl_uncued_wd,
                                 AI_tbl_cu),axis=1)
    big_AI_tbl = big_AI_tbl.transpose([-1,1,0])
    big_AI_tbl = big_AI_tbl.reshape((big_AI_tbl.shape[0],
                                     big_AI_tbl.shape[1]*big_AI_tbl.shape[-1]),
                                    order='F')
    
    cols = [['cued_prevspre','cued_postvspost','cued_rot','cued_unrot',
            'uncued_prevspre','uncued_postvspost',
            'cuedvsuncued+pair1','cuedvsuncued_pair2']]*2
    cols = [i+j for a,b in zip(cols,[['_2D']*10,['_3D']*10]) for i,j in zip(a,b)]
    
    big_AI_tbl = pd.DataFrame(big_AI_tbl,columns = cols)
    
    path = params['FULL_PATH'] + 'pca_data/valid_trials/big_AI_tbl.csv'
    big_AI_tbl.to_csv(path)
    

def run_contrasts(AI_tbl,alt):
    if AI_tbl.shape[1]>2:
        return
    else:
        # only run contrasts in the deterministic Buschman paradigm
        n_dims = AI_tbl.shape[0]
        for dim in range(n_dims):
            s = f'                    {dim+2}D: '
            # check if data normally distributed
            sw, sp = shapiro(np.diff(AI_tbl[dim,:,:],axis=0).squeeze())
                
            if sp >= .05:
                # do parametric stats
                stat, pval = ttest_1samp(np.diff(AI_tbl[dim,:,:],axis=0).squeeze(),
                                         0,alternative=alt)
                s += '1 sample t-test '
            else:
                stat, pval = wilcoxon(np.diff(AI_tbl[dim,:,:],axis=0).squeeze(),
                                      alternative=alt)
                s += 'wilcoxon sign rank test '
            
            s += alt + f' : stat = {stat:.3f}, p-val = {pval:.3f}'
            print(s)   
            
#%% 

def run_cued_within_delay(params):
    '''
    Run the Cued geometry AI pipeline.

    Parameters
    ----------
    params : dict
        Experiment parameters.

    Returns
    -------
    AI_tbl_cued_wd : array
        AI values, format: (n_dims,n_timepoints,n_models)

    '''
    print('          compare cued within delay')
    _ = check_intrinsic_dim(params)
    # AI_tbl_cued_wd = get_AI_cued_within_delay(params)
    AI_tbl_cued_wd = get_AI_within_delay(params)
    # save
    
    plot_AI(params,AI_tbl_cued_wd,'prepost','cued')
    plt.savefig(params['FIG_PATH']+'AI_prepost_cued.png')

    print_descriptives(AI_tbl_cued_wd,['pre','post'])
    # stats - is the AI greater for post-cue delay than pre-cue
    run_contrasts(AI_tbl_cued_wd,'greater')
    
    return AI_tbl_cued_wd


def run_cued_across_delay(params):
    print('          compare cued across delays')
    AI_tbl_cued_ad, same_ixs, trial_ixs = get_AI_cued_across_delays(params,cv=2)
    plot_AI(params,AI_tbl_cued_ad,'unrotrot','cued')
    plt.savefig(params['FIG_PATH']+'AI_unrotrot_cued.png')
    print_descriptives(AI_tbl_cued_ad,['unrotated','rotated'])
    # stats - is the AI for the 'unrotated' plane greater than for rotated?
    run_contrasts(AI_tbl_cued_ad,'two-sided')
    return AI_tbl_cued_ad, same_ixs, trial_ixs


def run_cued_AI_pipeline(params):
    '''
    Run the Cued geometry AI pipeline.

    Parameters
    ----------
    params : dict
        Experiment parameters.

    Returns
    -------
    AI_tbl_cued_wd : array
        AI values, format: (n_dims,n_timepoints,n_models)

    '''
    print('          compare cued within delay')
    _ = check_intrinsic_dim(params)
    # AI_tbl_cued_wd = get_AI_cued_within_delay(params)
    
    # calculate the AI between subspaces from the same delay
    AI_tbl_cued_wd = get_AI_within_delay(params)
    # save
    
    plot_AI(params,AI_tbl_cued_wd,'prepost','cued')
    plt.savefig(params['FIG_PATH']+'AI_prepost_cued.png')

    print_descriptives(AI_tbl_cued_wd,['pre','post'])
    # stats - is the AI greater for post-cue delay than pre-cue
    run_contrasts(AI_tbl_cued_wd,'greater')
    
    # calculate the AI between the same location subspaces from different delays
    # to see if one of both are rotated post-cue
    print('          compare cued across delays')
    AI_tbl_cued_ad, same_ixs, trial_ixs = get_AI_cued_across_delays(params,cv=2)
    plot_AI(params,AI_tbl_cued_ad,'unrotrot','cued')
    plt.savefig(params['FIG_PATH']+'AI_unrotrot_cued.png')
    print_descriptives(AI_tbl_cued_ad,['unrotated','rotated'])
    # stats - is the AI for the 'unrotated' plane greater than for rotated?
    run_contrasts(AI_tbl_cued_ad,'two-sided')
    
    return AI_tbl_cued_wd,AI_tbl_cued_ad, same_ixs, trial_ixs

def run_uncued_within_delay(params):
    '''
    Run the Uncued geometry AI pipeline.

    Parameters
    ----------
    params : dict
        Experiment parameters.

    Returns
    -------
    AI_tbl_uncued_wd : array
        AI values, format: (n_dims,n_timepoints,n_models).

    '''
    print('          compare uncued within delay')
    # AI_tbl_uncued_wd = get_AI_uncued_within_delay(params)
    AI_tbl_uncued_wd = get_AI_within_delay(params,cued=False)
    plot_AI(params,AI_tbl_uncued_wd,'prepost','uncued')
    plt.savefig(params['FIG_PATH']+'AI_prepost_uncued.png')
    # stats - is the AI greater for post-cue delay than pre-cue
    run_contrasts(AI_tbl_uncued_wd,'greater')
    # add print descriptives here
    print_descriptives(AI_tbl_uncued_wd,['pre','post'])
    return AI_tbl_uncued_wd


# def run_uncued_across_delay(params,same_ixs,trial_ixs):
#     print('          compare uncued across delays')
#     AI_tbl_uncued_ad = get_AI_uncued_across_delays(params,same_ixs,trial_ixs)
#     plot_AI(params,AI_tbl_uncued_ad,'unrotrot','uncued')
#     plt.savefig(params['FIG_PATH']+'AI_unrotrot_uncued.png')
#     # stats - the AI for the plane that was 'unrotated' when cued should be small
#     # whilst for the other plane, it should be large, if the transformations
#     # induced by the two retrocues are opposites
#     print_descriptives(AI_tbl_uncued_ad,['unrotated','rotated'])
#     run_contrasts(AI_tbl_uncued_ad,'two-sided')
#     return AI_tbl_uncued_ad


def run_uncued_AI_pipeline(params):
    '''
    Run the Uncued geometry AI pipeline.

    Parameters
    ----------
    params : dict
        Experiment parameters.

    Returns
    -------
    AI_tbl_uncued_wd : array
        AI values, format: (n_dims,n_timepoints,n_models).

    '''
    # Check if uncued subspaces are also rotated into a common frame of 
    # reference post-cuye, like their Cued counterparts.
    print('          compare uncued within delay')
    # AI_tbl_uncued_wd = get_AI_uncued_within_delay(params)
    AI_tbl_uncued_wd = get_AI_within_delay(params,cued=False)
    plot_AI(params,AI_tbl_uncued_wd,'prepost','uncued')
    plt.savefig(params['FIG_PATH']+'AI_prepost_uncued.png')
    # stats - is the AI greater for post-cue delay than pre-cue
    run_contrasts(AI_tbl_uncued_wd,'greater')
    # add print descriptives here
    print_descriptives(AI_tbl_uncued_wd,['pre','post'])
    
    
    return AI_tbl_uncued_wd

def run_uncued_vs_cued(params):
    print('          compare cued vs uncued')
    AI_tbl_cu = get_AI_cued_vs_uncued(params)
    # plot
    # ADD THE PLOTTER HERE
    # save plot
    print_descriptives(AI_tbl_cu.mean(1),[])
    return AI_tbl_cu


def run_cued_vs_uncued_AI_pipeline(params):
    '''
    Run the Cued/Uncued geometry AI pipeline.

    Parameters
    ----------
    params : dict
        Experiment parameters.

    Returns
    -------
    AI_tbl_cu : array
        AI values, format: (n_dims,n_timepoints,n_models)..

    '''
    print('          compare cued vs uncued')
    AI_tbl_cu = get_AI_cued_vs_uncued(params)
    # plot
    # ADD THE PLOTTER HERE
    # save plot
    print_descriptives(AI_tbl_cu.mean(1),[])
    return AI_tbl_cu
    

def run_full_AI_analysis(params):
    print('Running the AI subspace analysis:')
    AI_tbl_cued_wd = run_cued_within_delay(params)
    AI_tbl_cued_ad, same_ixs, trial_ixs = run_cued_across_delay(params)
    AI_tbl_uncued_wd = run_uncued_within_delay(params)
    # AI_tbl_uncued_ad = run_uncued_across_delay(params,same_ixs,trial_ixs)
    AI_tbl_cu = run_uncued_vs_cued(params)    
    
    export_AI_data(params,
                    AI_tbl_cued_wd,
                    AI_tbl_cued_ad,
                    AI_tbl_uncued_wd,
                    AI_tbl_cu)
    print('          done.')
    
    
def run_retrocue_timing_comparison(params):
    # plot the plane angles for the retrocue_timing condition
    
    pal = sns.color_palette("dark")
    
    inds = [3,0]
    markers = ['o','^']
    cols = [pal[ix] for ix in inds]
    ms = 10
    
    n_delay_lengths = 8
    conditions = ['pre-cue','post-cue']
    delay2_lengths = np.arange(n_delay_lengths)
    
    common_path = params['BASE_PATH']\
        + 'data_vonMises/experiment_4/'\
    
    n_models = 30
    all_AI = np.empty((2,len(conditions),n_models,len(delay2_lengths)))
    
    
    plt.figure(figsize=(7,5))
    ax1 = plt.subplot(111)
    # ax2 = plt.subplot(122,sharey=ax1,sharex=ax1) 
    jitter=.125
    for j,dl in enumerate(delay2_lengths):
        # load data
        load_path = common_path + 'delay2_' + str(dl) + 'cycles/'\
                 +'sigma' + str(params['sigma'])\
                    +'/kappa' + str(params['kappa_val'])\
                    +'/nrec' + str(params['n_rec'])\
                        +'/lr' + str(params['learning_rate']) + '/'\
                            +'pca_data/valid_trials'
        print('Delay 2 length: %d' %dl)
        _ = check_intrinsic_dim(params,custom_path=load_path)
        all_AI[:,:,:,j] = get_AI_cued_within_delay(params,custom_path=load_path)
        
        
        ax1.plot(np.ones((n_models,))*dl-jitter,all_AI[0,0,:,j],
                  marker=markers[0],ls='',color=cols[0],alpha=.2,markersize=ms)
        
        ax1.plot(np.ones((n_models,))*dl+jitter,all_AI[0,1,:,j],
                  marker=markers[1],ls='',color=cols[1],alpha=.2,markersize=ms)
    
    # add means
    ax1.plot(delay2_lengths-jitter,all_AI[0,0,:,:].mean(0),
             marker=markers[0],color=cols[0],markersize=ms,label='pre-cue')
    ax1.plot(delay2_lengths+jitter,all_AI[0,1,:,:].mean(0),
             marker=markers[1],color=cols[1],markersize=ms,label='post-cue')
    
    ax1.set_ylim([-0.1,1.1])
    ax1.set_xticks(delay2_lengths)
    
    ax1.set_ylabel('AI between cued subspaces')
    ax1.set_xlabel('Post-cue delay length')
    
    # ax1.set_title('Pre-cue')
    # ax2.set_title('Post-cue')
    
    ax1.legend(bbox_to_anchor=(1,1))
    plt.tight_layout()
    
    plt.savefig(common_path + 'compare_cued_AIs_sigma' + str(params['sigma']) + '.png')
    
    # export data
    data_for_export = all_AI[0,:,:,:].swapaxes(-1,1).reshape(2,-1).swapaxes(1,0)
    group_label = np.stack([[np.ones(n_models,)*i for i in range(n_delay_lengths)]]).reshape(-1)
    group_label = np.expand_dims(group_label,-1)
    data_for_export = np.concatenate((data_for_export,group_label),axis=1)
    df = pd.DataFrame(data=data_for_export,columns=['pre-cue','post-cue','delay2 length'])
    df.to_csv(common_path+'/AI_pre_post.csv')
   
    
def get_probabilistic_table(params):
    common_path = params['BASE_PATH']\
        + 'data_vonMises/MSELoss_custom/validity_paradigm/3delays/'
    
    n_models = params['n_models']
    conditions = ['deterministic','probabilistic','neutral']
    AI2 = np.empty((3,n_models,len(conditions)))
    AI3 = np.empty((3,n_models,len(conditions)))
    group = []
    for i,c in enumerate(conditions):
        path = common_path + c  +'/sigma' + str(params['sigma'])\
           +'/kappa' + str(params['kappa_val'])\
           +'/nrec' + str(params['n_rec'])\
               +'/lr' + str(params['learning_rate']) + '/'\
                   +'pca_data/valid_trials'
        
        AI = get_AI_cued_within_delay(params,path)
        AI2[:,:,i] = AI[0,:,:]
        AI3[:,:,i] = AI[1,:,:]
        group.append([c]*n_models)
    AI2 = AI2.reshape(3,-1,order='F')
    AI3 = AI3.reshape(3,-1,order='F')
    group = np.array(group).reshape(-1)
    
    AItbl2 = pd.DataFrame(data = np.concatenate((AI2,group[None,:]),axis=0).T,
                          columns = ['pre-cue','post-cue','post-probe','condition'])
    AItbl3 = pd.DataFrame(data = np.concatenate((AI3,group[None,:]),axis=0).T,
                          columns = ['pre-cue','post-cue','post-probe','condition'])
    
    AItbl2.to_csv(common_path+'/cued_AItbl2.csv')
    AItbl3.to_csv(common_path+'/cued_AItbl3.csv')

# def export_data(constants,cos_theta):
#     path = constants.PARAMS['FULL_PATH'] +'pca_data/valid_trials/'
#     cos_theta_pre = np.moveaxis(cos_theta['delay1'],-1,1)
#     cos_theta_post= np.moveaxis(cos_theta['delay2'],-1,1)
    
#     cos_theta_pre = cos_theta_pre.reshape(cos_theta_pre.shape[0],
#                                       cos_theta_pre.shape[1]*cos_theta_pre.shape[-1])
#     cos_theta_post = cos_theta_post.reshape(cos_theta_post.shape[0],
#                                         cos_theta_post.shape[1]*cos_theta_post.shape[-1])
    
#     # new shape is (model,training_point x time) - so it goes model0: training0time0, 
#     # training0time1,training0time3, training0time4, training1time0 etc
    
#     training_stages = cos_theta['train_stages']
#     timepoints = ['t'+str(t) for t in np.arange(cos_theta['delay1'].shape[-1])]
    
    
#     col_labels = list(itertools.product(training_stages,timepoints))
    
#     df_pre = pandas.DataFrame(data=cos_theta_pre, columns=col_labels)
#     df_post = pandas.DataFrame(data=cos_theta_post, columns=col_labels)
    
#     df_pre.to_csv(path+'/cos_angle_pre_training.csv')
#     df_post.to_csv(path+'/cos_angle_post_training.csv')



