#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 17:33:58 2022

@author: emilia
"""
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import constants
import pycircstat
import custom_plot as cplot



#%% get pre-training angles

# model_path = constants.PARAMS['FULL_PATH']
    
# load_path = model_path + 'saved_models/'

# for m in np.arange(constants.PARAMS['n_models']):
#     print('Model %d' %m)
    
#     constants.PARAMS['model_number'] = m
#     model = retnet.load_model(load_path,constants.PARAMS,device)


    
    
# get and sort angles
load_path = constants.PARAMS['FULL_PATH'] + 'pca_data/valid_trials/'
angles = pickle.load(open(load_path+'all_plane_angles.pckl','rb'))
angles = angles[:,-1]
angles_radians = np.radians(angles)

# median-split the angles
ix_low = np.where(angles_radians <= pycircstat.median(angles_radians))[0]
ix_high = np.where(angles_radians > pycircstat.median(angles_radians))[0]

ix_sort = np.argsort(angles)

# get training data

path = constants.PARAMS['FULL_PATH']

colours = np.array(sns.color_palette("coolwarm", constants.PARAMS['n_models']))

# max_n_trials = 388
max_n_trials = 31

plt.figure(figsize=(6.45,4.95))
all_loss = np.zeros((constants.PARAMS['n_models'],max_n_trials))
for model in ix_sort:
    model_number = str(model)
    
    # load training data
    f = open(path+'training_data/'+'training_data_model'+model_number+'.pckl','rb')
    track_training = pickle.load(f)
    f.close()
    
    all_loss[model,:len(track_training['loss_epoch'])] = track_training['loss_epoch']
    i = np.where(ix_sort==model)[0]
    plt.plot(track_training['loss_epoch'],'-',c = colours[model,:],
              label='model '+model_number)

plt.ylabel('Loss')
plt.xlabel('Epoch')
# plt.title('neutral')

plt.tight_layout()


# plt.figure(figsize=(6.45,4.95))
# for model in ix_sort:
#     model_number = str(model)
    
#     i = np.where(ix_sort==model)[0]
#     plt.plot(all_loss[model,:],'-',c = colours[i,:],alpha=.8,
#               label='model '+model_number)

# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.tight_layout()


#%% plot mean low and high angles
plt.figure(figsize=(6.45,4.95))
ax = plt.subplot(111)

H1,H2 = cplot.shadow_plot(ax, np.arange(max_n_trials), all_loss[ix_low,:max_n_trials], color = colours[0,:])
H3, H4 = cplot.shadow_plot(ax, np.arange(max_n_trials), all_loss[ix_high,:max_n_trials], color = colours[-1,:])

plt.legend([H1,H3],['low angle', 'high angle'])

plt.ylabel('Loss')
plt.xlabel('Epoch')

# plt.title('neutral')
plt.tight_layout()



#%% plot the distribution of the angles

pal = sns.color_palette("dark")
if constants.PARAMS['experiment'] == 'validity_paradigm':
    inds = [3,0,-3]
    markers = ['o','^','s']
    labels = ['pre-cue','post-cue','post-probe']
    i=2
else:
    inds = [3,0]
    markers = ['o','^']
    labels = ['pre','post']
    i=1

cols = [pal[ix] for ix in inds]

fig = plt.figure(figsize=(7.9,5))
ax = fig.add_subplot(111,polar=True)
ax.grid(False)
r = 1
ms = 12

for model in range(constants.PARAMS['n_models']):        
    ax.plot(angles_radians[model],r,marker=markers[i],
            color = cols[i],alpha=0.2,markersize=ms)
    
ax.tick_params(axis='x', which='major', pad=14)
ax.set_ylim([0,1.05])
ax.set_yticks([])

#%%

path = constants.PARAMS['FULL_PATH'] +'pca_data/valid_trials/'
cos_theta_tbl = pd.read_csv(path+'/cos_angle_post_training.csv')

cos_theta = np.stack((cos_theta_tbl["('untrained', 't4')"].to_numpy(),
                      cos_theta_tbl["('trained', 't4')"].to_numpy())).T
rel_angle = np.degrees(np.arccos(cos_theta))


#%%
from rep_geom_analysis import run_pca_pipeline

def get_cosine_angle(constants):
    load_path = constants.PARAMS['FULL_PATH'] +'pca_data/valid_trials/' #'partial_training/'
    train_stages = ['untrained','plateau','trained']
    delay_len = constants.PARAMS['trial_timings']['delay1_dur']
    
    theta_post = np.empty((constants.PARAMS['n_models'],len(train_stages),delay_len))
    
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

                subspace_d2 = run_pca_pipeline(constants,
                                               all_data[stage]['data'][:,d2_start+t,:],
                                               ['up','down'])
                theta_post[model,stage,t] = subspace_d2['theta']
    
    return theta_post

theta_post = get_cosine_angle(constants)

rel_angle = np.abs(theta_post[:,0,-1] - theta_post[:,-1,-1])

rel_angle_radians = np.radians(rel_angle)

# median-split the angles
ix_low = np.where(rel_angle_radians <= pycircstat.median(rel_angle_radians))[0]
ix_high = np.where(rel_angle_radians > pycircstat.median(rel_angle_radians))[0]

ix_sort = np.argsort(rel_angle_radians)



path = constants.PARAMS['FULL_PATH']

colours = np.array(sns.color_palette("coolwarm", constants.PARAMS['n_models']))

# max_n_trials = 388
max_n_trials = 250

plt.figure(figsize=(6.45,4.95))
all_loss = np.zeros((constants.PARAMS['n_models'],max_n_trials))
for model in ix_sort:
    model_number = str(model)
    
    # load training data
    f = open(path+'training_data/'+'training_data_model'+model_number+'.pckl','rb')
    track_training = pickle.load(f)
    f.close()
    
    all_loss[model,:len(track_training['loss_epoch'])] = track_training['loss_epoch']
    i = np.where(ix_sort==model)[0]
    plt.plot(track_training['loss_epoch'],'-',c = colours[model,:],
              label='model '+model_number)

plt.ylabel('Loss')
plt.xlabel('Epoch')
# plt.title('neutral')

plt.tight_layout()


plt.figure(figsize=(6.45,4.95))
ax = plt.subplot(111)

H1,H2 = cplot.shadow_plot(ax, np.arange(max_n_trials), all_loss[ix_low,:max_n_trials], color = colours[0,:])
H3, H4 = cplot.shadow_plot(ax, np.arange(max_n_trials), all_loss[ix_high,:max_n_trials], color = colours[-1,:])

plt.legend([H1,H3],['low angle', 'high angle'])

plt.ylabel('Loss')
plt.xlabel('Epoch')

# plt.title('neutral')
plt.tight_layout()



#%% do the same but with the AI
from get_subspace_alignment_index import *

load_path = constants.PARAMS['FULL_PATH'] +'pca_data/valid_trials/'
check_intrinsic_dim(constants,load_path+'partial_training/untrained/')

ai_tbl_u = get_AI_cued_within_delay(constants,custom_path=load_path+'partial_training/untrained/')
ai_tbl_t = get_AI_cued_within_delay(constants)

# relative AI
diff_AI = ai_tbl_t[0,1,:] - ai_tbl_u[0,1,:]

ix_low = np.where(diff_AI <= np.median(diff_AI))[0]
ix_high = np.where(diff_AI > np.median(diff_AI))[0]

ix_sort = np.argsort(diff_AI)

# post-training AI

# ix_low = np.where(ai_tbl_t[0,1,:] <= np.median(ai_tbl_t[0,1,:]))[0]
# ix_high = np.where(ai_tbl_t[0,1,:] > np.median(ai_tbl_t[0,1,:]))[0]

# ix_sort = np.argsort(ai_tbl_t[0,1,:])

#%%
n_epochs_to_convergence = np.empty((constants.PARAMS['n_models'],))
for model in range(constants.PARAMS['n_models']):
    model_number = str(model)
    
    # load training data
    f = open(path+'training_data/'+'training_data_model'+model_number+'.pckl','rb')
    track_training = pickle.load(f)
    f.close()
    
    # all_loss[model,:len(track_training['loss_epoch'])] = track_training['loss_epoch']
    # i = np.where(ix_sort==model)[0]
    # plt.plot(track_training['loss_epoch'],'-',c = colours[model,:],
    #           label='model '+model_number)
    
    n_epochs_to_convergence[model] = len(track_training['loss_epoch'])

plt.figure()
plt.plot(diff_AI,n_epochs_to_convergence,'ko')
plt.xlabel('diff AI')
plt.ylabel('N epochs to convergence')
plt.tight_layout()
