#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 18:04:08 2021

This is the configuration file for Experiment 2, post-cue delay length = 3 cycles variation. To run the experiment,
pass the path to the folder which is to contain the data from all Experiments on lines 205-206.

@author: emilia
"""
import numpy as np
import itertools
import torch
import seaborn as sns
from src.generate_data_von_mises import make_stimuli_vonMises
from src.helpers import check_path

# %% TASK AND MODEL PARAMETERS ##

PARAMS = {'n_stim': 16,
          'kappa_val': 5.0,
          'add_fixation': False,
          'n_colCh': 17,
          'n_rec': 200,
          'n_out': 17}

PARAMS['experiment_number'] = 2
PARAMS['n_delays'] = 2
PARAMS['expt_key'] = 'expt_2_delay2_3cycles'
PARAMS['ai_vs_learning_speed'] = False  # variant of the experiment for running the AI vs learning speed analysis.
# different from the base experiment configuration in terms of the training stop procedure used (stop training once
# the loss falls below a hard threshold) and an increased number of models trained

PARAMS['n_trial_types'] = (PARAMS['n_stim'] ** 2) * 2
PARAMS['trial_timings'] = {}
PARAMS['trial_timings']['stim_dur'] = 1
PARAMS['trial_timings']['delay1_dur'] = 11
PARAMS['trial_timings']['cue_dur'] = 1
PARAMS['trial_timings']['delay2_dur'] = 14 - PARAMS['trial_timings']['delay1_dur']
PARAMS['trial_timings']['probe_dur'] = 0
PARAMS['trial_timings']['delay3_dur'] = 0

PARAMS['phi'] = torch.linspace(-np.pi, np.pi, PARAMS['n_colCh'] + 1)[:-1]

# variable delay params
PARAMS['var_delays'] = False

if PARAMS['var_delays']:

    PARAMS['delay_lengths'] = [2, 5, 6, 8]
    PARAMS['default_length'] = 5
    PARAMS['which_delay'] = 'both'  # first, second or both

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
                           torch.ones((len(PARAMS['delay_combos']), 1), dtype=int) \
                           * PARAMS['default_length']), 1)

            # set the delay durations to the longest/default value
            PARAMS['trial_timings']['delay1_dur'] = np.max(PARAMS['delay_lengths'])
            PARAMS['trial_timings']['delay2_dur'] = PARAMS['default_length']
        else:
            PARAMS['delay_combos'] = \
                torch.cat((torch.ones((len(PARAMS['delay_combos']), 1), dtype=int) \
                           * PARAMS['default_length'], PARAMS['delay_combos']), 1)

            # set the delay durations to the longest/default value
            PARAMS['trial_timings']['delay1_dur'] = PARAMS['default_length']
            PARAMS['trial_timings']['delay2_dur'] = np.max(PARAMS['delay_lengths'])

    PARAMS['n_delay_combos'] = len(PARAMS['delay_combos'])

PARAMS['seq_len'] = sum(PARAMS['trial_timings'].values())

# delay start and end points
PARAMS['trial_timepoints'] = {}
PARAMS['trial_timepoints']['delay1_start'] = PARAMS['trial_timings']['stim_dur']
PARAMS['trial_timepoints']['delay1_end'] = PARAMS['trial_timepoints']['delay1_start'] \
                                           + PARAMS['trial_timings']['delay1_dur']
PARAMS['trial_timepoints']['delay2_start'] = PARAMS['trial_timepoints']['delay1_end'] \
                                             + PARAMS['trial_timings']['cue_dur']
PARAMS['trial_timepoints']['delay2_end'] = PARAMS['trial_timepoints']['delay2_start'] \
                                           + PARAMS['trial_timings']['delay2_dur']
PARAMS['trial_timepoints']['delay3_start'] = PARAMS['trial_timepoints']['delay2_end'] \
                                             + PARAMS['trial_timings']['probe_dur']
PARAMS['trial_timepoints']['delay3_end'] = PARAMS['trial_timepoints']['delay3_start'] \
                                           + PARAMS['trial_timings']['delay3_dur']

# noise params
PARAMS['sigma'] = 0.0  # scaling factor for noise (boundary if uniform, s.d. if normal)
PARAMS['noise_type'] = 'hidden'  # hidden or input
PARAMS['noise_distr'] = 'normal'  # normal or uniform
PARAMS['noise_period'] = 'all'

if PARAMS['noise_period'] == 'probe':
    # trial timepoints to which the noise is applied
    PARAMS['noise_timesteps'] = [-1]
elif PARAMS['noise_period'] == 'delays':
    PARAMS['noise_timesteps'] = np.concatenate((range(1, 9),
                                                range(10, sum(PARAMS['trial_timings'].values()) - 1)))
elif PARAMS['noise_period'] == 'probe_and_delays':
    PARAMS['noise_timesteps'] = np.concatenate((range(1, 9),
                                                range(10, sum(PARAMS['trial_timings'].values()) - 1),
                                                [int(PARAMS['seq_len'] - 1)]))
elif PARAMS['noise_period'] == 'all':
    PARAMS['noise_timesteps'] = np.arange(PARAMS['seq_len'])
elif PARAMS['noise_period'] == 'none':
    PARAMS['noise_timesteps'] = []
else:
    ValueError('Invalid noise period.')

# PARAMS['sigma'] = np.sqrt(PARAMS['sigma']**2 / len(PARAMS['noise_timesteps']))

# cue validity params

PARAMS['add_probe'] = False

PARAMS['cue_validity'] = 1  # proportion of trials where the retrocued and probed locations match

if PARAMS['cue_validity'] == 1:
    PARAMS['condition'] = 'deterministic'
elif PARAMS['cue_validity'] == .5:
    PARAMS['condition'] = 'neutral'
else:
    PARAMS['condition'] = 'probabilistic'

## TRAINING PARAMETERS ##

PARAMS['n_models'] = 30

PARAMS['n_epochs'] = 1000
PARAMS['learning_rate'] = 10 ** (-4)
PARAMS['init_scale'] = 1
# 'init_scale' - factor by which the weight init should be scaled - needed in 
# order to be able to train longer sequences without hidden noise

PARAMS['criterion_type'] = 'abs_loss'  # 'abs_loss' # or 'loss_der'
PARAMS['MSE_criterion'] = 0.0005
PARAMS['conv_criterion'] = {}  # parameters for the loss_der convergence criterion
PARAMS['conv_criterion']['smooth_sd'] = 3  # standard deviation value for the Gaussian smoother
PARAMS['conv_criterion']['window'] = 15  # smoothing window
PARAMS['conv_criterion']['thr_slope'] = -2e-05  # threshold for the dLoss/dt value
PARAMS['conv_criterion']['thr_loss'] = 0.0036
# threshold for the loss value - set to the level that corresponds to monkey performance

PARAMS['n_jobs'] = 1

PARAMS['n_trial_instances'] = 1
PARAMS['n_trial_instances_test'] = 100
PARAMS['stim_set_size'] = PARAMS['n_trial_types'] * PARAMS['n_trial_instances']

PARAMS['batch_size'] = 1  # for mini-batch training
PARAMS['n_batches'] = PARAMS['stim_set_size'] // PARAMS['batch_size']

if PARAMS['var_delays']:
    # matrix with all trial-wise delay values
    # PARAMS['delay_mat'] = torch.cat([PARAMS['delay_combos']]*PARAMS['batch_size'])
    # PARAMS['batch_size'] *= PARAMS['n_delay_combos']

    PARAMS['delay_mat'] = torch.cat([PARAMS['delay_combos']] * PARAMS['stim_set_size'])
    PARAMS['stim_set_size'] *= PARAMS['n_delay_combos']

# make a base training dataset - no noise, trials ordered by the cued colour
TRAINING_DATA = make_stimuli_vonMises(PARAMS, epoch='test')

## binning params for PCA analysis
PARAMS['n_inp'] = TRAINING_DATA['inputs'].shape[-1]
PARAMS['B'] = 4  # number of colour bins
PARAMS['L'] = 2  # number of cue locations
PARAMS['M'] = PARAMS['B'] * PARAMS['L']  # total number of bins

PLOT_PARAMS = {'4_colours': sns.color_palette("husl", 4), 'save_plots': False}

# %% PATHS ##
# this is what you need to set
PARAMS['BASE_PATH'] = 'your_datafolder/'
PARAMS['MATLAB_PATH'] = 'your_matlab_files_path/'  # path to the matlab files (mixture model parameters)

# path to the datafiles from the current experiment
PARAMS['COND_PATH'] = f"{PARAMS['BASE_PATH']}data_vonMises/experiment_{PARAMS['experiment_number']}/"
if PARAMS['ai_vs_learning_speed']:
    # Variant of the experiment for running the AI vs learning speed analysis.
    # Different from the base experiment configuration in terms of the training stop procedure used (stop training once
    # the loss falls below a hard threshold) and number of models trained (increased to 50)
    PARAMS['COND_PATH'] += 'ai_vs_learning_speed/'

if PARAMS['experiment_number'] == 4:
    PARAMS['COND_PATH'] += f"validity_{PARAMS['cue_validity']}/5_cycles/"
elif PARAMS['experiment_number'] == 2:
    PARAMS['COND_PATH'] += f"delay2_{PARAMS['trial_timings']['delay2_dur']}cycles/"

print(PARAMS['COND_PATH'])  # print the condition path to the console
check_path(PARAMS['COND_PATH'])

# common path for all experiment 2 variants - for saving plots and data structures
PARAMS['EXPT2_PATH'] = f"{PARAMS['BASE_PATH']}data_vonMises/experiment_{PARAMS['experiment_number']}/"

# full parameterisation
PARAMS['FULL_PATH'] = f"{PARAMS['COND_PATH']}sigma{PARAMS['sigma']}/kappa{PARAMS['kappa_val']}/nrec{PARAMS['n_rec']}/" \
                      f"lr{PARAMS['learning_rate']}/"

PARAMS['FIG_PATH'] = f"{PARAMS['FULL_PATH']}figs/"
check_path(PARAMS['FIG_PATH'])  # create the figure folder if it doesn't exist

PARAMS['RAW_DATA_PATH'] = f"{PARAMS['FULL_PATH']}evaluation_data/"
PARAMS['RESULTS_PATH'] = f"{PARAMS['FULL_PATH']}evaluation_data/"
