#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 18:04:08 2021

@author: emilia
"""
import os
import numpy as np
import itertools
import torch
import seaborn as sns
from scipy.stats import norm
from generate_data_vonMises import make_stimuli_vonMises

## TASK AND MODEL PARAMETERS ##

PARAMS = {'n_stim':16,
          'kappa_val': 5.0,
          'add_fixation':False,
          'n_colCh':17,
          'n_rec':200,
          'n_out':17}                   

PARAMS['experiment'] = 'Buschman_var_delays'
# PARAMS['target_type'] = 'class_label' # or 'Gaussian'
PARAMS['target_type']='angle_val'
PARAMS['phi'] = torch.linspace(-np.pi, np.pi, PARAMS['n_colCh']+1)[:-1]

PARAMS['n_trial_types'] = (PARAMS['n_stim']**2)*2
PARAMS['trial_timings'] = {}
PARAMS['trial_timings']['stim_dur']=1
PARAMS['trial_timings']['delay1_dur'] = 5
PARAMS['trial_timings']['cue_dur']= 1
PARAMS['trial_timings']['delay2_dur'] = 5
PARAMS['trial_timings']['probe_dur'] = 0
PARAMS['trial_timings']['delay3_dur'] = 0


# variable delay params
PARAMS['var_delays'] = True

if PARAMS['var_delays']:
    
    PARAMS['delay_lengths'] = [2,3,5,6]
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
PARAMS['sigma'] = 0.0 # scaling factor for noise (boundary if uniform, s.d. if normal)


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
    PARAMS['noise_timesteps'] = np.arange(PARAMS['seq_len'])
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

PARAMS['n_epochs']=20
PARAMS['learning_rate']=10**(-4)
PARAMS['init_scale']=1
# 'init_scale' - factor by which the weight init should be scaled - needed in 
# order to be able to train longer sequences without hidden noise

PARAMS['loss_fn'] = 'MSE_custom' #'CEL' # or 'MSE'
PARAMS['l2_activity'] = False # add an L2 loss on the hidden activity
PARAMS['Br'] = 5 * 10**(-4) # add an L2 loss on the hidden activity


PARAMS['optim'] = 'RMSprop' #'Adam' #'RMSprop' #'SGDm'
PARAMS['MSE_criterion'] = 0.0005
PARAMS['conv_criterion'] = {}
PARAMS['conv_criterion']['window'] = 5
PARAMS['conv_criterion']['thr_loss'] = 0.004 # last mini-pleateau
PARAMS['conv_criterion']['thr_slope'] = -5e-05

PARAMS['n_jobs'] = 1

PARAMS['from_scratch'] = False  # train from scratch

PARAMS['n_trial_instances'] = 1
PARAMS['n_trial_instances_test'] = 100
PARAMS['stim_set_size']= PARAMS['n_trial_types'] * PARAMS['n_trial_instances']

PARAMS['batch_size']= 1

if PARAMS['var_delays']:
    # matrix with all trial-wise delay values
    PARAMS['delay_mat'] = torch.cat([PARAMS['delay_combos']]*PARAMS['stim_set_size'])
    PARAMS['stim_set_size'] *= PARAMS['n_delay_combos']
    
# make a base training dataset - no noise, trials ordered by the cued colour
TRAINING_DATA = make_stimuli_vonMises(PARAMS,epoch='test')

## binning params for PCA analysis
PARAMS['n_inp'] = TRAINING_DATA['inputs'].shape[-1]
PARAMS['B'] = 4
PARAMS['L'] = 2
PARAMS['M'] = PARAMS['B'] * PARAMS['L']
    


PLOT_PARAMS = {}
PLOT_PARAMS['4_colours'] = sns.color_palette("husl",4)

## PATHS ##

# PARAMS['BASE_PATH'] = os.path.abspath(os.getcwd())+'/'
PARAMS['BASE_PATH'] = '/Volumes/EP_Passport/emilia'+'/'


# main training condition, incl. noise type and period
PARAMS['COND_PATH'] = PARAMS['BASE_PATH'] + 'data_vonMises/MSELoss_custom/Buschman_paradigm_vardelays/'

print(PARAMS['COND_PATH'])


# full parameterisation
PARAMS['FULL_PATH'] = PARAMS['COND_PATH'] \
        +'sigma' + str(PARAMS['sigma'])\
            +'/kappa' + str(PARAMS['kappa_val'])\
            +'/nrec' + str(PARAMS['n_rec'])\
                +'/lr' + str(PARAMS['learning_rate']) + '/'


PARAMS['FIG_PATH'] = PARAMS['FULL_PATH'] + 'figs/'