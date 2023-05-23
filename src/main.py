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

from helpers import check_path, transfer_to_cpu
import retrocue_model as retnet
import generate_data_vonMises as dg
import rep_geom_analysis as rg
import learning_dynamics as ld
# import compare_maintenance_mechanisms as cmm

import get_subspace_alignment_index as ai
# import pdb

# plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'font.size': 22})
plt.rcParams['svg.fonttype'] = 'none'
import seaborn as sns
sns.set_context("notebook", font_scale=1.5)
sns.set_style("ticks")


# import constants
import constants_expt2 as constants
#%% set up flags, paths and device


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu') # train on CPU to ensure deterministic behaviour

save_path = constants.PARAMS['FULL_PATH']

parallelise_flag = False
train_flag  = False
eval_flag   = False
analysis_flag = False


def run_experiment(constants, parallelise_flag, train_flag, eval_flag, analysis_flag):

    #%% train

    if train_flag:
        start_time = time.time()
        # copy the configuration file to data the destination folder
        shutil.copy2(constants.PARAMS['BASE_PATH']+'constants.py',
                     constants.PARAMS['FULL_PATH']+'constants.py')
        def train_and_save(m):
            constants.PARAMS['model_number'] = m
            model, track_training = retnet.train_model(constants.PARAMS,constants.TRAINING_DATA,device)

            retnet.save_model(save_path,constants.PARAMS,model,track_training)


        if parallelise_flag:
            Parallel(n_jobs = constants.PARAMS['n_jobs'])(delayed(train_and_save)(m) for m in np.arange(constants.PARAMS['n_models']))
        else:
            for m in range(constants.PARAMS['n_models']):
                 train_and_save(m)


        end_time = time.time()

        print('Time elapsed: %.2f minutes' %((end_time - start_time)/60))
        print('Params: ')
        print('    Condition: '+constants.PARAMS['condition'])
        print('    N epochs: %4d '%(constants.PARAMS['n_epochs']))
        print('    Noise s.d.: %.3f' %constants.PARAMS['sigma'])
        print('    Init scale: %.3f' %constants.PARAMS['init_scale'])
        print('    N rec: %3d' %constants.PARAMS['n_rec'])
        print('    Lr : %.8f' %constants.PARAMS['learning_rate'])

        if device==torch.device('cuda'):
            # transfer training data to the CPU for analysis
            transfer_to_cpu()

    # %% create test dataset

    if eval_flag:
        # model, RNN = retnet.define_model(constants.PARAMS, device)

        #% generate test datataset(s)
        test_data = dg.generate_test_dataset(constants.PARAMS)

        #% set up paths and save dataset(s)
        model_path = constants.PARAMS['FULL_PATH']
        load_path = model_path + 'saved_models/'
        eval_path = model_path+'pca_data/'
        # valid trials
        valid_path = eval_path + 'valid_trials/'
        check_path(valid_path)

        if constants.PARAMS['cue_validity'] < 1:
            # valid trials
            test_data_valid_trials = dg.subset_of_trials(constants.PARAMS,
                                                         test_data,
                                                    test_data['valid_trial_ixs'])
            retnet.save_data(test_data_valid_trials,[],valid_path+'test_dataset')

            # invalid trials
            invalid_path = eval_path + 'invalid_trials/'
            check_path(invalid_path)
            test_data_invalid_trials = dg.subset_of_trials(constants.PARAMS,
                                                           test_data,
                                                    test_data['invalid_trial_ixs'])
            retnet.save_data(test_data_invalid_trials,[],invalid_path+'test_dataset')
        else:
            if constants.PARAMS['experiment_number']<3:
                # trained delay
                retnet.save_data(test_data['trained'],
                                  [],
                                  valid_path+'test_dataset')

                # out-of-range temporal generalisation
                out_range_path = valid_path + 'out_range_tempGen/'
                check_path(out_range_path)
                retnet.save_data(test_data['out-of-range'],
                                  [],
                                  out_range_path+'test_dataset')

                # out-of-range shorter
                out_range_path_shorter = valid_path + 'out_range_tempGen_shorter/'
                check_path(out_range_path_shorter)
                retnet.save_data(test_data['out-of-range-shorter'],
                                 [],
                                 out_range_path_shorter+'test_dataset')

                # in-range temporal generalisation
                in_range_path = valid_path + 'in_range_tempGen/'
                check_path(in_range_path)
                retnet.save_data(test_data['in-range'],
                                  [],
                                  in_range_path+'test_dataset')
            else:
                # expt 3 with cue_validity = 1, expt 4
                retnet.save_data(test_data,[],valid_path+'test_dataset')

        #% evaluate models on test dataset
        for m in np.arange(constants.PARAMS['n_models']):
            print('Model %d' %m)

            constants.PARAMS['model_number'] = m
            model = retnet.load_model(load_path,constants.PARAMS,device)

            if constants.PARAMS['cue_validity'] < 1:
                # evaluate separately for valid and invalid trials
                eval_data,pca_data,rdm_data,model_outputs = \
                retnet.eval_model(model,test_data_valid_trials,constants.PARAMS,valid_path,trial_type='valid')

                # invalid
                eval_data,pca_data,rdm_data,model_outputs = \
                retnet.eval_model(model,test_data_invalid_trials,constants.PARAMS,invalid_path,trial_type='invalid')

            else:
                if constants.PARAMS['experiment_number']<3:
                    # test on the/one of the delay lengths used in training
                    constants.PARAMS = \
                        dg.update_time_params(constants.PARAMS,
                                              constants.PARAMS['test_delay_lengths'][0])
                    eval_data,pca_data,rdm_data,model_outputs = \
                    retnet.eval_model(model,test_data['trained'],constants.PARAMS,valid_path)

                    # test out-of-range temporal generalisation
                    constants.PARAMS = \
                        dg.update_time_params(constants.PARAMS,
                                              constants.PARAMS['test_delay_lengths'][1])
                    eval_data,pca_data,rdm_data,model_outputs = \
                    retnet.eval_model(model,
                                      test_data['out-of-range'],
                                      test_data['out-of-range']['params'],
                                      out_range_path)

                    # shorter
                    constants.PARAMS = \
                        dg.update_time_params(constants.PARAMS,
                                              1)
                    eval_data,pca_data,rdm_data,model_outputs = \
                    retnet.eval_model(model,
                                      test_data['out-of-range-shorter'],
                                      test_data['out-of-range-shorter']['params'],
                                      out_range_path_shorter)

                    # test in-range temporal generalisation
                    constants.PARAMS = \
                        dg.update_time_params(constants.PARAMS,
                                              constants.PARAMS['test_delay_lengths'][2])
                    eval_data,pca_data,rdm_data,model_outputs = \
                    retnet.eval_model(model,
                                      test_data['in-range'],
                                      test_data['in-range']['params'],
                                      in_range_path)
                else:
                    # expt 3 with cue_validity = 1, expt 4
                    # only valid trials with the trained delay length
                    eval_data,pca_data,rdm_data,model_outputs = \
                        retnet.eval_model(model,test_data,constants.PARAMS,valid_path)


    #%% analysis pipeline
    if analysis_flag:
        # plot learning curves

        # representational geometry analysis
        rg.run_full_rep_geom_analysis(constants)

        # subspace alignment analysis
        # ai.run_full_AI_analysis(constants)

        # learning dynamics analysis
        # ld.run_learning_stages_analysis(constants,constants.TRAINING_DATA,device)

        # if constants.PARAMS['experiment_number']==2:
        #     cmm.run_maintenance_mechanism_analysis()
