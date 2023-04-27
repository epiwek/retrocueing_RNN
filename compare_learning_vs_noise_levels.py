#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 17:43:23 2022

@author: emilia
"""
import numpy as np
import constants
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
path = constants.PARAMS['FULL_PATH']
# plt.rcParams.update({'font.size': 10})

colours = np.array(sns.color_palette("husl", 10))

noise_lvls = [0.01,0.03,0.05,0.07] #,0.03,0.05,0.07,0.1]#,0.1,0.5]
# noise_lvls_target = np.array([0.1,0.2,0.3,0.5,0.7])


# noise_lvls = np.array([0,0.1,0.2,0.3])#,0.5,0.7])
# noise_lvls = np.sqrt(noise_lvls_target**2 / len(constants.PARAMS['noise_timesteps']))

# plateau_length_median = np.empty((len(noise_lvls)))
# plateau_length = np.empty((len(noise_lvls),10))

n_models = 10

if n_models != 10:
    raise ValueError('Change number of subplots in the outer loop to correspond to n_models')
plt.figure()  
end_loss = np.empty((n_models,len(noise_lvls)))
for m in np.arange(n_models):
    model_number = str(m)
    # plt.figure(figsize=(9,5))
    plt.subplot(2,5,m+1)
    plt.title('Model ' + model_number)
    for i,n in enumerate(noise_lvls):
        path = constants.PARAMS['COND_PATH'] \
            +'sigma' + str(n)\
                +'/kappa' + str(constants.PARAMS['kappa_val'])\
                +'/nrec' + str(constants.PARAMS['n_rec'])\
                    +'/lr' + str(constants.PARAMS['learning_rate']) + '/'
                    
    
        
        
        # load training data
        f = open(path+'training_data/'+'training_data_model'+model_number+'.pckl','rb')
        track_training = pickle.load(f)
        f.close()
        
        plt.plot(track_training['loss_epoch'],c = colours[i,:],label=str(noise_lvls[i]))
        end_loss[m,i] = track_training['loss_epoch'][-1]
        # if n == 0.01:
        #     end_loss[m] = track_training['loss_epoch'][-1]
        # if n == 0.5:
        #     plt.plot(np.arange(len(track_training['loss_epoch'])),
        #              torch.ones((len(track_training['loss_epoch']),))*end_loss[m],'k--')
        #plateau_length[i,m] = np.logical_and(track_training['loss_epoch']<0.007,track_training['loss_epoch']>0.006).sum()
plt.legend(bbox_to_anchor = (1,1))
    #plateau_length_median[i] = np.median(plateau_length[i,:])
    
    
#%%

def get_loss_slope(window,loss_vals):
    #window = params['conv_criterion']['window']
    p = np.polyfit(np.arange(window), loss_vals[-window:], 1)
    return p[0]

path = constants.PARAMS['COND_PATH'] \
    +'sigma0.0' \
        +'/kappa' + str(constants.PARAMS['kappa_val'])\
        +'/nrec' + str(constants.PARAMS['n_rec'])\
            +'/lr' + str(constants.PARAMS['learning_rate']) + '/'
            
slopes = np.empty((n_models,))
for m in np.arange(n_models):
    model_number = str(m)
    f = open(path+'training_data/'+'training_data_model'+model_number+'.pckl','rb')
    track_training = pickle.load(f)
    f.close()
    
    slopes[m] = get_loss_slope(10,track_training['loss_epoch'])

np.logical_and(slopes >= -2e-05,
                        slopes <= 0)


#%%

n = 0.07

n_epochs = []
for m in np.arange(n_models):
    # plt.subplot(2,5,m+1)
    model_number = str(m)
    
    
    
    path = '/Volumes/EP_Passport/emilia/data_vonMises/experiment_1/7_cycles_new_cc/' \
        +'sigma' + str(n)\
            +'/kappa' + str(constants.PARAMS['kappa_val'])\
            +'/nrec' + str(constants.PARAMS['n_rec'])\
                +'/lr' + str(constants.PARAMS['learning_rate']) + '/'
                
    f = open(path+'training_data/'+'training_data_model'+model_number+'.pckl','rb')
    track_training = pickle.load(f)
    f.close()
    
    # plt.plot(track_training['loss_epoch'],c = 'k',label='thr=0.0005')
    # conv crit                    
    n_epochs.append(len(track_training['loss_epoch']))


plt.figure(figsize=(9,5))
end_loss = np.empty((n_models,))

for m in np.arange(n_models):
    plt.subplot(2,5,m+1)
    model_number = str(m)
    
    
    
    path = '/Volumes/EP_Passport/emilia/data_vonMises/experiment_1/7_cycles_end_loss_0.0005/' \
        +'sigma' + str(n)\
            +'/kappa' + str(constants.PARAMS['kappa_val'])\
            +'/nrec' + str(constants.PARAMS['n_rec'])\
                +'/lr' + str(constants.PARAMS['learning_rate']) + '/'
                
    f = open(path+'training_data/'+'training_data_model'+model_number+'.pckl','rb')
    track_training = pickle.load(f)
    f.close()
    
    plt.plot(track_training['loss_epoch'],c = 'k',label='thr=0.0005')
    # conv crit
    # path = '/Volumes/EP_Passport/emilia/data_vonMises/experiment_1/7_cycles_new_cc/' \
    #     +'sigma' + str(n)\
    #         +'/kappa' + str(constants.PARAMS['kappa_val'])\
    #         +'/nrec' + str(constants.PARAMS['n_rec'])\
    #             +'/lr' + str(constants.PARAMS['learning_rate']) + '/'
                
    # f = open(path+'training_data/'+'training_data_model'+model_number+'.pckl','rb')
    # track_training = pickle.load(f)
    # f.close()
    # plt.plot(track_training['loss_epoch'],c = 'r',label='cc')
    
    # find where slope < new criterion
    search = True
    thr = -2e-05
    i = n_epochs[m]
    while np.logical_and(np.logical_and(i>=n_epochs[m],i<len(track_training['loss_epoch'])),
                         search):
        slope = get_loss_slope(10,track_training['loss_epoch'][:i])
        search = not np.logical_and(slope>=thr,slope<=0)
        i += 1
    
    plt.plot(track_training['loss_epoch'][:i-1],c = 'r',label='cc')
    
    
#%% delete

angles = get_cued_subspaces_indivModels(constants)
run_plane_angles_analysis(constants)

        