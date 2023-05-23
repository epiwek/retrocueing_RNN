#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 11:37:47 2021

@author: emilia
"""
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import constants

matplotlib.rcParams.update({'font.size': 22})
#%% set up variables

path = constants.PARAMS['FULL_PATH']


training_alg = 'SGD'

colours = np.array(sns.color_palette("husl", constants.PARAMS['n_models']))

end_loss = []
n_epochs_to_convergence = []

if constants.PARAMS['condition']=='probabilistic':
    end_loss_valid = []
#%% plot learning curves for all models

plt.figure(figsize=(5,5))

for model_number in np.arange(constants.PARAMS['n_models']):
    model_number = str(model_number)
    
    # load training data
    f = open(path+'training_data/'+'training_data_model'+model_number+'.pckl','rb')
    track_training = pickle.load(f)
    f.close()
    
    if training_alg == 'SGD':
        # if np.logical_or((int(model_number)==5),(int(model_number)==6)):
        # plt.plot(torch.sum(track_loss[:,:500],0),'-k')
        # plt.plot(torch.sum(track_training['loss'],0),'-k')
        plt.plot(torch.sum(track_training['loss'],0),'-',c = colours[int(model_number),:],label='model '+model_number)
        end_loss.append(torch.sum(track_training['loss'][:,-1],0))
        n_epochs_to_convergence.append(track_training['loss'].shape[-1])
        # else:
            # plt.plot(torch.sum(track_loss,0),'-',c = colours[int(model_number),:],label='model '+model_number)
        if constants.PARAMS['condition']=='probabilistic':
            valid_ix = np.setdiff1d(np.arange(track_training['loss'].shape[0]),
                                    track_training['invalid_trials'][-1,:])
            end_loss_valid.append(torch.sum(track_training['loss'][valid_ix,-1],0))
    else:
        plt.plot(track_training['loss'][0,:],'k-',label='model '+model_number)


end_loss = np.array(end_loss)

# plt.legend()
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.title(constants.PARAMS['condition'])

plt.tight_layout()

#%%

plt.savefig(path+'loss_fig.png')