#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 14:50:57 2021

@author: emilia
"""

import numpy as np
import torch
import time
import shutil
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

from src.helpers import check_path, transfer_to_cpu
import src.retrocue_model as retnet
import src.generate_data_vonMises as dg
import src.behav_analysis as behav
# import src.rep_geom_analysis as rg
# import learning_dynamics as ld
# import compare_maintenance_mechanisms as cmm

# import src.get_subspace_alignment_index as ai

# import pdb

# plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'font.size': 22})
plt.rcParams['svg.fonttype'] = 'none'
import seaborn as sns

sns.set_context("notebook", font_scale=1.5)
sns.set_style("ticks")

# import constants - pick experiment
import constants.constants_expt1 as c

# %% pick experiment, set up flags, paths and device


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')  # train on CPU to ensure deterministic behaviour

save_path = c.PARAMS['FULL_PATH']


train_flag = False
parallelise_flag = False
eval_flag = True
analysis_flag = False


def run_experiment(c, parallelise_flag, train_flag, eval_flag, analysis_flag):
    # %% train models
    if train_flag:
        start_time = time.time()
        # copy the configuration file to data the destination folder
        shutil.copy2(c.PARAMS['BASE_PATH'] + 'constants.py',
                     c.PARAMS['FULL_PATH'] + 'constants.py')

        def train_and_save(m):
            c.PARAMS['model_number'] = m
            model, track_training = retnet.train_model(c.PARAMS, c.TRAINING_DATA, device)

            retnet.save_model(save_path, c.PARAMS, model, track_training)

        # train models in parallel or sequentially
        if parallelise_flag:
            Parallel(n_jobs=c.PARAMS['n_jobs'])(
                delayed(train_and_save)(m) for m in np.arange(c.PARAMS['n_models']))
        else:
            for m in range(c.PARAMS['n_models']):
                train_and_save(m)

        end_time = time.time()

        print('Time elapsed: %.2f minutes' % ((end_time - start_time) / 60))
        print('Params: ')
        print('    Condition: ' + c.PARAMS['condition'])
        print('    N epochs: %4d ' % (c.PARAMS['n_epochs']))
        print('    Noise s.d.: %.3f' % c.PARAMS['sigma'])
        print('    Init scale: %.3f' % c.PARAMS['init_scale'])
        print('    N rec: %3d' % c.PARAMS['n_rec'])
        print('    Lr : %.8f' % c.PARAMS['learning_rate'])

        if device == torch.device('cuda'):
            # transfer training data to the CPU for analysis
            transfer_to_cpu()
    # %% create test dataset

    if eval_flag:
        print('Evaluating models ...')

        # % generate test datataset(s) / conditions
        test_data = dg.generate_test_dataset(c.PARAMS)

        # % set up paths and save dataset(s)
        model_path = c.PARAMS['FULL_PATH']
        load_path = model_path + 'saved_models/'
        eval_path = model_path + 'pca_data/'
        # valid trials
        valid_path = eval_path + 'valid_trials/'
        check_path(valid_path)

        # keep the nested if statements for better legibility
        if c.PARAMS['cue_validity'] < 1:
            # Experiment 4, 75% and 50% validity conditions
            # valid trials
            test_data_valid_trials = dg.subset_of_trials(c.PARAMS,
                                                         test_data,
                                                         test_data['valid_trial_ixs'])
            retnet.save_data(test_data_valid_trials, valid_path + 'test_dataset')

            # invalid trials
            invalid_path = eval_path + 'invalid_trials/'
            check_path(invalid_path)
            test_data_invalid_trials = dg.subset_of_trials(c.PARAMS,
                                                           test_data,
                                                           test_data['invalid_trial_ixs'])
            retnet.save_data(test_data_invalid_trials, invalid_path + 'test_dataset')
        else:
            if c.PARAMS['experiment_number'] < 3:
                # Experiments 1 and 2
                # trained delay
                retnet.save_data(test_data['trained'], valid_path + 'test_dataset')

                # out-of-range temporal generalisation
                out_range_path = valid_path + 'out_range_tempGen/'
                check_path(out_range_path)
                retnet.save_data(test_data['out-of-range'], out_range_path + 'test_dataset')

                # out-of-range shorter
                out_range_path_shorter = valid_path + 'out_range_tempGen_shorter/'
                check_path(out_range_path_shorter)
                retnet.save_data(test_data['out-of-range-shorter'], out_range_path_shorter + 'test_dataset')

                # in-range temporal generalisation
                in_range_path = valid_path + 'in_range_tempGen/'
                check_path(in_range_path)
                retnet.save_data(test_data['in-range'], in_range_path + 'test_dataset')
            else:
                # Expt 4 with cue_validity = 1, Expt 3
                retnet.save_data(test_data, valid_path + 'test_dataset')

        # % evaluate models on test dataset
        for m in np.arange(c.PARAMS['n_models']):
            print('Model %d' % m)

            c.PARAMS['model_number'] = m
            model = retnet.load_model(load_path, c.PARAMS, device)

            if c.PARAMS['cue_validity'] < 1:
                # evaluate separately for valid and invalid trials
                _, _, _ = retnet.eval_model(model,
                                            test_data_valid_trials,
                                            c.PARAMS,
                                            valid_path,
                                            trial_type='valid')

                # invalid
                _, _, _ = retnet.eval_model(model,
                                            test_data_invalid_trials,
                                            c.PARAMS,
                                            invalid_path,
                                            trial_type='invalid')

            else:
                if c.PARAMS['experiment_number'] < 3:
                    # test on the/one of the delay lengths used in training
                    c.PARAMS = dg.update_time_params(c.PARAMS,
                                                     c.PARAMS['test_delay_lengths'][0])
                    _, _, _ = retnet.eval_model(model, test_data['trained'], c.PARAMS, valid_path)

                    # test out-of-range temporal generalisation (longer than maximum trained delay)
                    c.PARAMS = dg.update_time_params(c.PARAMS,
                                                     c.PARAMS['test_delay_lengths'][1])
                    _, _, _ = retnet.eval_model(model,
                                                test_data['out-of-range'],
                                                test_data['out-of-range']['params'],
                                                out_range_path)

                    # test out-of-range temporal generalisation (shorter than minimum trained delay)
                    c.PARAMS = dg.update_time_params(c.PARAMS, 1)
                    _, _, _ = retnet.eval_model(model,
                                                test_data['out-of-range-shorter'],
                                                test_data['out-of-range-shorter']['params'],
                                                out_range_path_shorter)

                    # test in-range temporal generalisation
                    c.PARAMS = \
                        dg.update_time_params(c.PARAMS,
                                              c.PARAMS['test_delay_lengths'][2])
                    _, _, _ = retnet.eval_model(model,
                                                test_data['in-range'],
                                                test_data['in-range']['params'],
                                                in_range_path)
                else:
                    # expt 3 with cue_validity = 1, expt 4
                    # only valid trials with the trained delay length
                    _, _, _ = retnet.eval_model(model, test_data, c.PARAMS, valid_path)

        retnet.export_behav_data_to_matlab(c.PARAMS)

    # %% analysis pipeline
    if analysis_flag:
        # valid trials
        print('To fix - remove / standardise the paths')
        common_path = c.PARAMS['DATA_PATH']

        # get all test conditions and their corresponding data folder names
        test_conditions, folder_names = dg.generate_test_conditions()

        # get full test folder paths
        test_paths = [common_path + f for f in folder_names[c.PARAMS['expt_key']]]

        if c.PARAMS['experiment_number'] == 1:
            # Behaviour
            # test_paths = [valid_path, out_range_path, out_range_path_shorter, in_range_path]
            behav.run_behav_analysis(c, test_conditions[c.PARAMS['expt_key']], test_paths)

            # Cued geometry
                #  geometry
                # AI
            # Uncued geometry
                #  geometry
                # AI
            # Cued/Uncued geometry
                #  geometry
                # AI
            # Connectivity

            # Learning dynamics



            # plot learning curves

        # representational geometry analysis
        # rg.run_full_rep_geom_analysis(c)

        # subspace alignment analysis
        # ai.run_full_AI_analysis(c)

        # learning dynamics analysis
        # ld.run_learning_stages_analysis(c,c.TRAINING_DATA,device)

        # if c.PARAMS['experiment_number']==2:
        #     cmm.run_maintenance_mechanism_analysis()
