#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 18:04:08 2021

@author: emilia
"""
from generate_data_vonMisses import make_stimuli_vonMises
import os
import numpy as np
import itertools
import torch

## TASK AND MODEL PARAMETERS ##

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
PARAMS['trial_timings']['resp_dur'] = 0
PARAMS['trial_timings']['ITI_dur'] = 0

# variable delay params
PARAMS['var_delays'] = True

if PARAMS['var_delays']:
    
    PARAMS['delay_lengths'] = [1,4,7,10]
    PARAMS['default_length'] = 5
    PARAMS['which_delay'] = 'both' #first, second or both
    
    if PARAMS['which_delay'] == 'both':
        PARAMS['delay_combos'] = torch.tensor(list(itertools.combinations_with_replacement(PARAMS['delay_lengths'], 2)))
        # set the delay durations to the longest value
        PARAMS['trial_timings']['delay1_dur'] = np.max(PARAMS['delay_lengths'])
        PARAMS['trial_timings']['delay2_dur'] = np.max(PARAMS['delay_lengths'])

    else:
        PARAMS['delay_combos'] = torch.tensor(list(itertools.combinations_with_replacement(PARAMS['delay_lengths'], 1)))
        if PARAMS['which_delay'] == 'first':
            PARAMS['delay_combos'] = \
                torch.cat((PARAMS['delay_combos'],
                           torch.ones((len(PARAMS['delay_combos']),1),dtype=int)\
                               *PARAMS['default_length']),1)
                    
            # set the delay durations to the longest/default value
            PARAMS['trial_timings']['delay1_dur'] = np.max(PARAMS['delay_lengths'])
            PARAMS['trial_timings']['delay2_dur'] = PARAMS['default_length']
        else:
            PARAMS['delay_combos'] = \
                torch.cat((torch.ones((len(PARAMS['delay_combos']),1),dtype=int)\
                           *PARAMS['default_length'],PARAMS['delay_combos']),1)
                    
            # set the delay durations to the longest/default value
            PARAMS['trial_timings']['delay1_dur'] = PARAMS['default_length']
            PARAMS['trial_timings']['delay2_dur'] = np.max(PARAMS['delay_lengths'])
            
    PARAMS['n_delay_combos'] = len(PARAMS['delay_combos'])    


PARAMS['seq_len'] = sum(PARAMS['trial_timings'].values())

# delay start and end points

PARAMS['trial_timepoints'] = {}
PARAMS['trial_timepoints']['delay1_start'] = PARAMS['trial_timings']['stim_dur']
PARAMS['trial_timepoints']['delay1_end'] = PARAMS['trial_timings']['stim_dur']\
    +PARAMS['trial_timings']['delay1_dur']
PARAMS['trial_timepoints']['delay2_start'] = PARAMS['trial_timings']['stim_dur']\
    +PARAMS['trial_timings']['delay1_dur']+PARAMS['trial_timings']['cue_dur']
PARAMS['trial_timepoints']['delay2_end'] = PARAMS['trial_timings']['stim_dur']\
    +PARAMS['trial_timings']['delay1_dur']+PARAMS['trial_timings']['cue_dur']\
        +PARAMS['trial_timings']['delay2_dur']



# noise params
PARAMS['epsilon'] = 0 # scaling factor for noise (boundary if uniform, s.d. if normal)

PARAMS['multiple_sds'] = False
PARAMS['epsilon_delays'] = .0625

PARAMS['noise_type'] = 'none' # hidden or input
PARAMS['noise_distr'] = 'normal' # normal or uniform


PARAMS['noise_period'] = 'none'



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
elif PARAMS['noise_period'] == 'none':
    PARAMS['noise_timesteps'] = 'none'
else:
    ValueError('Invalid noise period.')
    
        
# cue validity params
PARAMS['add_probe'] = False
PARAMS['cue_validity'] = 1 # proportion of trials where the retrocued and probed locations match

if PARAMS['cue_validity'] == 1:
    PARAMS['condition'] = 'deterministic'
elif PARAMS['cue_validity'] == .5:
    PARAMS['condition'] = 'neutral'
else:
    PARAMS['condition'] = 'probabilistic'



## TRAINING PARAMETERS ##

PARAMS['n_models'] = 10

PARAMS['n_epochs']=2000
PARAMS['learning_rate']=10**(-5)
PARAMS['init_scale']=1
# 'init_scale' - factor by which the weight init should be scaled - needed in 
# order to be able to train longer sequences without hidden noise

PARAMS['loss_fn'] = 'MSE' # 'MSE' or 'CEL'

PARAMS['optim'] = 'RMSprop' #'SGDm'
PARAMS['MSE_criterion'] = 0.1
PARAMS['n_jobs'] = 4

PARAMS['from_scratch'] = True  # train from scratch

PARAMS['n_trial_instances'] = 1
PARAMS['batch_size']= PARAMS['n_trial_types'] * PARAMS['n_trial_instances']

if PARAMS['var_delays']:
    # matrix with all trial-wise delay values
    PARAMS['delay_mat'] = torch.cat([PARAMS['delay_combos']]*PARAMS['batch_size'])
    PARAMS['batch_size'] *= PARAMS['n_delay_combos']
    
# make a base training dataset - no noise, trials ordered by the cued colour
TRAINING_DATA = make_stimuli_vonMises(PARAMS,epoch='test')

## binning params for PCA analysis
PARAMS['n_inp'] = TRAINING_DATA['inputs'].shape[-1]
PARAMS['B'] = 4
PARAMS['L'] = 2
PARAMS['M'] = PARAMS['B'] * PARAMS['L']
    

## PATHS ##

PARAMS['BASE_PATH'] = os.path.abspath(os.getcwd())+'/'

# main training condition, incl. noise type and period
PARAMS['COND_PATH'] = PARAMS['BASE_PATH'] + 'data_vonMises/MSELoss/with_fixation_variable_delays_'\
    +PARAMS['condition']+'_'+PARAMS['noise_period']+'_'+PARAMS['optim']+'/'\
        +PARAMS['which_delay'] + '/'
#PARAMS['COND_PATH'] = PARAMS['BASE_PATH'] + 'data_vonMises/MSELoss/with_fixation_13cycles_noise_h_g_'+\
#    PARAMS['condition']+'_'+PARAMS['noise_period']+'_'+PARAMS['optim']+'/'



# full parameterisation
PARAMS['FULL_PATH'] = PARAMS['COND_PATH'] \
        +'epsilon' + str(PARAMS['epsilon'])\
            +'/kappa' + str(PARAMS['kappa_val'])\
            +'/nrec' + str(PARAMS['n_rec'])\
                +'/lr' + str(PARAMS['learning_rate']) + '/'
