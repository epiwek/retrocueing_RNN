#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 12:58:08 2021

@author: emilia
"""

import numpy as np
import torch
from torch import nn
from joblib import Parallel, delayed
import pickle
from helpers import check_path, print_progress
import retrocue_model as retnet

import constants


path = constants.PARAMS['FULL_PATH']
#save_path = constants.BASE_PATH + 'data_vonMises/MSELoss/with_fixation_13cycles_noise_h_g_'+constants.PARAMS['condition']+'_'+constants.PARAMS['noise_period']+'_'+constants.PARAMS['optim']+'/epsilon'+str(constants.PARAMS['epsilon'])+'/'


for m in np.arange(constants.PARAMS['n_models']):
    constants.PARAMS['model_number'] = m
    
    #load data
    track_training = pickle.load(open(path+'training_data/'+'training_data_model'+str(m)+'.pckl','rb'))
    
    for key,value in track_training.items():
        print(key)
        print(track_training[key].is_cuda)