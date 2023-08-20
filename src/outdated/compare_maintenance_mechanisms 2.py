#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 19:20:09 2022

@author: emilia

This calculates the mean delay ctg scores and compares those between expts 1 & 2
"""


import pickle
import constants_expt1 as c
import numpy as np
from scipy.stats import zscore
import statsmodels.api as sm
from scipy.stats import shapiro, ttest_ind, ttest_1samp, mannwhitneyu
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt



def run_reg(X,y):
    X = zscore(X)
    # create predictor matrix - add bias
    X = sm.add_constant(X)
    
    #z-score data
    y = zscore(y)
    
    # run regression
    results = sm.OLS(y, X).fit()
    
    return results


# create idealised stable temporal attractor and dynamic code matrices

delay_length = c.PARAMS['trial_timings']['delay1_dur']
trial_length = c.PARAMS['seq_len']

# dynamic_model = np.eye((trial_length))
# dynamic_model[dynamic_model==0]=.5
# # dynamic_model = np.flipud((dynamic_model))

# stable_model = np.eye((trial_length))
# stable_model[stable_model==0]=.5


d1_start = c.PARAMS['trial_timepoints']['delay1_start']
d1_end = c.PARAMS['trial_timepoints']['delay1_end']
d2_start = c.PARAMS['trial_timepoints']['delay2_start']
d2_end = c.PARAMS['trial_timepoints']['delay2_end']



d1_x,d1_y = np.concatenate((np.triu_indices(c.PARAMS['trial_timings']['delay1_dur'],k=1),
                            np.tril_indices(c.PARAMS['trial_timings']['delay1_dur'],k=-1)),1)
d1_x += d1_start
d1_y += d1_start

d2_x,d2_y = np.concatenate((np.triu_indices(c.PARAMS['trial_timings']['delay2_dur'],k=1),
                            np.tril_indices(c.PARAMS['trial_timings']['delay2_dur'],k=-1)),1)
d2_x += d2_start
d2_y += d2_start


d_x = np.concatenate((d1_x,d2_x))
d_y = np.concatenate((d1_y,d2_y))

diag_ix = np.concatenate((np.arange(d1_start,d1_end),np.arange(d2_start,d2_end)))
# stable_model[d1_start:d1_end,d1_start:d1_end] = 1 # stable attractor in pre-cue delay
# stable_model[d2_start:d2_end,d2_start:d2_end] = 1
# # stable_model = np.flipud(stable_model)


# # get condensed forms
# dynamic_model_diag = dynamic_model[np.triu_indices_from(dynamic_model,k=1)]
# stable_model_diag = stable_model[np.triu_indices_from(stable_model,k=1)]

# X = np.stack((dynamic_model_diag,stable_model_diag),1) + np.random.randn(66,2)*.001
# # add noise otherwise the dynamic model is just a constant
# X[X>1]=1
# compare the variable delays and standard Buschman models in terms of their maintenance mechanisms

vardelay_model_path = c.PARAMS['BASE_PATH'] +\
    'data_vonMises/experiment_2/' +\
        'sigma0.0/kappa5.0/nrec200/lr0.0001/'
standard_model_path = c.PARAMS['BASE_PATH'] +\
    'data_vonMises/experiment_1/' +\
        'sigma0.07/kappa5.0/nrec200/lr0.0001/'

# for i,condition in enumerate([vardelay_model_path,standard_model_path]):
#     scores_struct = pickle.load(open(condition+'pca_data/valid_trials/ctg_scores.pckl','rb'))   
#     mean_scores = np.mean(scores_struct['scores'][:,:,2,:],-1) # mean across models
    
#     # average the two triangles
#     ix_up = np.triu_indices_from(mean_scores,k=1)
#     ix_down = tuple([ix_up[1],ix_up[0]])
    
#     mean_scores_symm = np.stack((mean_scores[ix_up],mean_scores[ix_down])).mean(0)
    
#     results = run_reg(X,mean_scores_symm)


mean_scores = np.empty((c.PARAMS['n_models'],2,2)) # model, condition
off_diag_scores = np.empty((c.PARAMS['n_models'],2)) # model, condition
diag_scores = np.empty((c.PARAMS['n_models'],2)) # model, condition
for i,condition in enumerate([vardelay_model_path,standard_model_path]):
    scores_struct = pickle.load(open(condition+'pca_data/valid_trials/ctg_scores.pckl','rb'))   
    
    # get the two delays - mean value for each (only off-diagonal elements)
    mean_scores[:,0,i] = scores_struct['scores'][d1_x,d1_y,-1,:].mean(0)
    mean_scores[:,1,i] = scores_struct['scores'][d2_x,d2_y,-1,:].mean(0)
    
    
    # different implementation - averge ascores across both delays
    
    diag_scores[:,i] = np.diagonal(scores_struct['scores'][:,:,-1,:])[:,diag_ix].mean(-1)
    off_diag_scores[:,i] = scores_struct['scores'][d_x,d_y,-1,:].mean(0)
    


mean_delay_scores = np.concatenate((mean_scores[:,:,0].mean(1),mean_scores[:,:,1].mean(1)))
labels = np.array([['variable']*c.PARAMS['n_models'],['fixed']*c.PARAMS['n_models']]).reshape(-1)
tbl = pd.DataFrame(np.stack((mean_delay_scores,labels),1),columns=['mean delay score','condition'])
tbl['mean delay score'] = tbl['mean delay score'].astype(float)


shapiro(off_diag_scores[:,0])
shapiro(off_diag_scores[:,1])

shapiro(off_diag_scores[:,0]/diag_scores[:,0])
shapiro(off_diag_scores[:,1]/diag_scores[:,1])

# test if variable delays > fixed
ttest_ind(np.log(mean_scores[:,:,0].mean(1)),np.log(mean_scores[:,:,1].mean(1)),alternative='greater')

# test if variable and fixed delays against chance

ttest_1samp(mean_scores[:,:,0].mean(1),0.5,alternative='greater')
ttest_1samp(np.log(mean_scores[:,:,1].mean(1)),np.log(0.5),alternative='greater')


plt.figure(figsize=(5.5,5))
sns.boxplot(data=tbl, x='condition',y='mean delay score',
            palette=[sns.color_palette("Set2")[i] for i in [0,-2]])
            # palette="Set2")
plt.plot(plt.xlim(),[.5,.5],'k--')
plt.ylim([.4,.85])
plt.tight_layout()

# plt.savefig(c.PARAMS['BASE_PATH']+'data_vonMises/experiment_2/compare_fixed_and_var_delays.png')
 


#%%


# if delta_t == constants.PARAMS['seq_len']:
#     # all timepoints
    
#     plt.axhline(constants.PARAMS['trial_timepoints']['delay1_end']-.5, color='k')
#     plt.axvline(constants.PARAMS['trial_timepoints']['delay1_end']-.5, color='k')
    
#     plt.axhline(constants.PARAMS['trial_timepoints']['delay2_start']-.5, color='k')
#     plt.axvline(constants.PARAMS['trial_timepoints']['delay2_start']-.5, color='k')
    
#     plt.xticks(range(delta_t),labels=np.arange(delta_t)+1)
#     plt.yticks(range(delta_t),labels=np.arange(delta_t)+1)