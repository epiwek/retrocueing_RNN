#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 18:04:08 2021

@author: emilia
"""
from generate_data_vonMisses import make_stimuli_vonMises
import os
import numpy as np

## task and model parameters

PARAMS = {'n_stim':16,
          'kappa_val': 1.0,
          'add_fixation':True,
          'n_colCh':17,
          'n_rec':200,
          'n_out':17}                   

PARAMS['n_trial_types'] = (PARAMS['n_stim']**2)*2
PARAMS['trial_timings'] = {}
PARAMS['trial_timings']['stim_dur']=1
PARAMS['trial_timings']['delay1_dur'] = 8
PARAMS['trial_timings']['cue_dur']= 1
PARAMS['trial_timings']['delay2_dur'] = 8
PARAMS['trial_timings']['resp_dur'] = 1
PARAMS['trial_timings']['ITI_dur'] = 0
PARAMS['seq_len'] = sum(PARAMS['trial_timings'].values())


# noise params
PARAMS['epsilon'] = 2.0/19 # scaling factor for noise (bound if uniform, s.d. if normal)

PARAMS['multiple_sds'] = False
PARAMS['epsilon_delays'] = .0625

PARAMS['noise_type'] = 'hidden' # hidden or input
PARAMS['noise_distr'] = 'normal' # normal or uniform


PARAMS['noise_period'] = 'all'



if PARAMS['noise_period'] == 'probe':
    # trial timepoints to which the noise is applied
    PARAMS['noise_timesteps'] = [-1]
elif PARAMS['noise_period'] == 'delays':
    PARAMS['noise_timesteps'] = np.concatenate((range(1,9),
                        range(10,sum(PARAMS['trial_timings'].values())-1)))
elif PARAMS['noise_period'] == 'probe_and_delays':
    PARAMS['noise_timesteps'] = np.concatenate((range(1,9),
                        range(10,sum(PARAMS['trial_timings'].values())-1),[int(PARAMS['seq_len']-1)]))
elif PARAMS['noise_period'] == 'all':
    PARAMS['noise_timesteps'] = 'all'
else:
    ValueError('Invalid noise period.')
    
        
# cue validity params
PARAMS['add_probe'] = True
PARAMS['cue_validity'] = .75 # proportion of trials where the retrocued and probed locations match

if PARAMS['cue_validity'] == 1:
    PARAMS['condition'] = 'deterministic'
elif PARAMS['cue_validity'] == .5:
    PARAMS['condition'] = 'neutral'
else:
    PARAMS['condition'] = 'probabilistic'



## training params
PARAMS['n_models'] = 10

PARAMS['n_epochs']=20000
PARAMS['learning_rate']=10**(-5)
PARAMS['init_scale']=1
# 'init_scale' - factor by which the weight init should be scaled - needed in 
# order to be able to train longer sequences

PARAMS['loss_fn'] = 'MSE'
# PARAMS['loss_fn'] = 'CEL'

PARAMS['optim'] = 'RMSprop' #'SGDm'
PARAMS['MSE_criterion'] = 0.1
PARAMS['n_jobs'] = 1

PARAMS['from_scratch'] = True # train from scratch

PARAMS['n_trial_instances'] = 1
PARAMS['batch_size']= PARAMS['n_trial_types'] * PARAMS['n_trial_instances']


# make a base training dataset - no noise, trials ordered by the cued colour
TRAINING_DATA = make_stimuli_vonMises(PARAMS,epoch='test')

## binning params for PCA analysis
PARAMS['n_inp'] = TRAINING_DATA['inputs'].shape[-1]
PARAMS['B'] = 4
PARAMS['L'] = 2
PARAMS['M'] = PARAMS['B'] * PARAMS['L']
    

## paths

PARAMS['BASE_PATH'] = os.path.abspath(os.getcwd())+'/'

# main training condition, incl. noise type and period
PARAMS['COND_PATH'] = PARAMS['BASE_PATH'] + 'data_vonMises/MSELoss/with_fixation_13cycles_noise_h_g_'+\
    PARAMS['condition']+'_'+PARAMS['noise_period']+'_'+PARAMS['optim']+'/'

# full parameterisation
PARAMS['FULL_PATH'] = PARAMS['COND_PATH'] \
        +'epsilon' + str(PARAMS['epsilon'])\
            +'/kappa' + str(PARAMS['kappa_val'])\
            +'/nrec' + str(PARAMS['n_rec'])\
                +'/lr' + str(PARAMS['learning_rate']) + '/'