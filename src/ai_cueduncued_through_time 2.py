#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 16:35:50 2022

@author: emilia
"""

from helpers import sort_by_uncued, bin_data
import constants
import pickle
from get_subspace_alignment_index import get_simple_AI
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro, ttest_1samp, wilcoxon, pearsonr,spearmanr

load_path = constants.PARAMS['FULL_PATH'] + 'pca_data/valid_trials'
n_models = constants.PARAMS['n_models']


n_colours = constants.PARAMS['B']

t_start = constants.PARAMS['trial_timepoints']['delay2_start']-1
t_end = constants.PARAMS['seq_len']
t_range = t_end-t_start

max_dim=3

AI_tbl_cu = np.empty((max_dim-1,2,t_range,constants.PARAMS['n_models']))
AI_tbl_cu_precue = np.empty((max_dim-1,2,t_range,constants.PARAMS['n_models']))


for model in range(n_models):
    
    print('Model %d' %model)
    model_number = str(model)
    
    # load eval data
    f = open(load_path+'/eval_data_model' + model_number + '.pckl', 'rb')
    obj = pickle.load(f)
    data = obj["data"]
    f.close()
    
    n_trials = data.shape[0]
    
    
    # bin data by cued colour
    data_binned_cued = bin_data(data,constants.PARAMS)

    # bin data by uncued colour   
    data_sorted_uncued = sort_by_uncued(obj, constants.PARAMS)
    data_binned_uncued = bin_data(data_sorted_uncued['data'],constants.PARAMS)
    
    loc_split = n_colours
    # split by cued location
    # cued_up_trials = torch.cat((data_binned_cued[:n_colours,:,:],
    #                             data_binned_uncued[:n_colours,:,:]))
    
    # cued_down_trials = torch.cat((data_binned_cued[n_colours:,:,:],
    #                             data_binned_uncued[n_colours:,:,:]))
    
    for t in np.arange(t_start,t_end):
        for max_dim in range(2,max_dim+1):
            AI_tbl_cu[max_dim-2,0,t-t_start,model]= get_simple_AI(data_binned_cued[:loc_split,t,:],
                                                          data_binned_uncued[:loc_split,t,:],max_dim) # AI cued up
            AI_tbl_cu[max_dim-2,1,t-t_start,model]= get_simple_AI(data_binned_cued[loc_split:,t,:],
                                                         data_binned_uncued[loc_split:,t,:],max_dim) # AI cued down
            
            AI_tbl_cu_precue[max_dim-2,0,t-t_start,model]= get_simple_AI(data_binned_cued[:loc_split,t-t_start,:],
                                                          data_binned_uncued[:loc_split,t-t_start,:],max_dim) # AI cued up
            AI_tbl_cu_precue[max_dim-2,1,t-t_start,model]= get_simple_AI(data_binned_cued[loc_split:,t-t_start,:],
                                                         data_binned_uncued[loc_split:,t-t_start,:],max_dim) # AI cued down
            


plt.figure()
colours = np.array(sns.color_palette("husl", constants.PARAMS['n_models']))

for m in range(n_models):
    plt.plot(AI_tbl_cu[0,:,:,m].mean(0),'-o',c=colours[m,:])         



def plot_angle(constants,AI_tbl_cu):
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
    
    
    cols = sns.color_palette("rocket_r",1)
    
    plt.figure(figsize=(5,4))
    ax = plt.subplot(111)
    
    # wrap to -pi,pi
    AI_mean = AI_tbl_cu[0,:,:,:].mean(0).mean(-1)
    AI_SEM = np.std(AI_tbl_cu[0,:,:,:].mean(0),-1)
    
    # H1,H2 = cplot.shadow_plot(ax,
    #                   np.arange(theta['delay1'].shape[-1])+1,
    #                   [theta_mean_pre,theta_sem_pre],precalc = True,
    #                   alpha=.3,
    #                   color=cols[i])
    
    ax.errorbar(x=np.arange(AI_tbl_cu.shape[2]),y=AI_mean,
                yerr=AI_SEM,color=cols[0])
    
    
    
    
    # ax.legend(ax.get_children()[3:6],labels,bbox_to_anchor=(1,.6),loc='center right')
    ax.set_xlabel('time from retrocue')
    ax.set_ylabel('AI cued/uncued')
    plt.tight_layout()
    

#%%/
from learning_dynamics import corr_retro_weights

def corr_retro_weight_r_AI(r_weights,AI_cu):
   
    stat1,p1 = shapiro(r_weights)
    stat2,p2 = shapiro(AI_cu)
    
    # correlate

    if np.logical_or(p1< .05, p2 < .05):
        # spearman
        print('spearman')
        r,p_val = spearmanr(r_weights,AI_cu)
    else:
        print('pearson')
        r,p_val = pearsonr(r_weights,AI_cu)
    return r,p_val
   
r = corr_retro_weights(constants)

t=1     
plt.figure()
plt.plot(r,AI_tbl_cu[0,:,t,:].mean(0),'ko')
plt.xlabel('r retro weights')
plt.ylabel('AI cu delay2 t0')

