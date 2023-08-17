#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 20:50:02 2021

@author: emilia
"""

import numpy as np
import torch
import pickle
import retrocue_model as retnet

import constants


#%% set up params and create dataset

# if torch.cuda.is_available():
#     print('GPU available')
# else:
#     print('GPU not available')
device = torch.device('cpu')

#training_cond = 'data_vonMises/MSELoss/with_fixation_13cycles_noise_h_g_'+constants.PARAMS['condition']+'_'+constants.PARAMS['noise_period']+'_' + constants.PARAMS['optim']+'/epsilon'+str(constants.PARAMS['epsilon'])+'/kappa'+str(constants.PARAMS['kappa_val'])+'/nrec'+str(constants.PARAMS['n_rec'])+'/lr'+str(constants.PARAMS['learning_rate'])+'/'

path = constants.PARAMS['FULL_PATH']
#save_path = constants.BASE_PATH + 'data_vonMises/MSELoss/with_fixation_13cycles_noise_h_g_'+constants.PARAMS['condition']+'_'+constants.PARAMS['noise_period']+'_'+constants.PARAMS['optim']+'/epsilon'+str(constants.PARAMS['epsilon'])+'/'


for m in np.arange(constants.PARAMS['n_models']):
    constants.PARAMS['model_number'] = m
    
    #load data
    track_training = pickle.load(open(path+'training_data/'+'training_data_model'+str(m)+'.pckl','rb'))
    
    
    for key,value in track_training.items():
        track_training[key] = track_training[key].to(device)
        
    #% save on cpu
    retnet.save_data(track_training, constants.PARAMS, path+'training_data/'+'training_data_model')
    print('Model '+str(m)+' done')


