#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 14:50:57 2021

This is the main simulation file. It contains functions that train models, evaluate them on datasets and analyse the
data.

To run an Experiment, make sure to choose the appropriate configuration file saved in the 'constants/' directory. E.g.,
for Experiment 1, you should choose the 'constants_expt1.py' module.

You can run the simulations in two ways. To do it from your favourite IDE, open this file and run it (after modifying
line 395 to import the appropriate module, and setting the appropriate flags on line 397).

To run it from the command line,  call the run_experiment_cli.py file instead, passing the name of the constants module,
as well as the experiment flags (only the ones corresponding to the aspects of the simulation that you wish to run).

The experiments flags include: parallelise_flag, train_flag, eval_flag, and analysis_flag and control different aspects
 of the simulation, namely model training (and whether it should be done in parallel), evaluation, and data analysis.

For example, to run Experiment 1 (only the analysis phase), you would run the following from the command line:
>> python run_experiment_cli.py constants.constants_expt1 --analysis_flag

@author: emilia
"""

import numpy as np
import torch
import time
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns

from src.helpers import check_path, transfer_to_cpu
import src.retrocue_model as retnet
import src.generate_data_von_mises as dg
import src.behav_analysis as behav
import src.preprocess_rep_geom_data as ppc
import src.rep_geom_analysis as rg
import src.decoding_analysis as dec
import src.subspace_alignment_index as ai
import src.learning_dynamics_and_connectivity_analysis as ld
import src.subspace_AI_vs_learning_speed as ai_learning_speed

#%% set parameters for plots

plt.rcParams.update({'font.size': 22})
plt.rcParams['svg.fonttype'] = 'none'

sns.set_context("notebook", font_scale=1.5)
sns.set_style("ticks")

#%% run the entire simulation for a given experiment (train and evaluate the models, and analyse the data)


def train_models(constants, parallelise_flag, device):
    """
    Train models for a given Experiment.

    :param module constants: A Python module containing constants and configuration data for the simulation.
        Configuration files are saved in the 'constants/' directory.
    :param bool parallelise_flag: If True, trains models in parallel.
    :param torch.object device: torch.device({device_name}) on which to train the models. E.g., torch.device('cpu') for
        the CPU.
    :return:
    """
    # train models

    start_time = time.time()
    save_path = constants.PARAMS['FULL_PATH']
    print('Training models')

    def train_and_save(m):
        constants.PARAMS['model_number'] = m
        model, track_training = retnet.train_model(constants.PARAMS, constants.TRAINING_DATA, device)

        retnet.save_model(save_path, constants.PARAMS, model, track_training)

    # train models in parallel or sequentially
    if parallelise_flag:
        Parallel(n_jobs=constants.PARAMS['n_jobs'])(
            delayed(train_and_save)(m) for m in np.arange(constants.PARAMS['n_models']))
    else:
        for m in range(constants.PARAMS['n_models']):
            train_and_save(m)

    end_time = time.time()

    print('Time elapsed: %.2f minutes' % ((end_time - start_time) / 60))
    print('Params: ')
    print('    Condition: ' + constants.PARAMS['condition'])
    print('    N epochs: %4d ' % (constants.PARAMS['n_epochs']))
    print('    Noise s.d.: %.3f' % constants.PARAMS['sigma'])
    print('    Init scale: %.3f' % constants.PARAMS['init_scale'])
    print('    N rec: %3d' % constants.PARAMS['n_rec'])
    print('    Lr : %.8f' % constants.PARAMS['learning_rate'])

    if device == torch.device('cuda'):
        # transfer training data to the CPU for analysis
        transfer_to_cpu(constants)

    return


def evaluate_models(constants):
    """
    Create all test datasets appropriate for a given Experiment and evaluate models on them.
    :param module constants: A Python module containing constants and configuration data for the simulation.
        Configuration files are saved in the 'constants/' directory.
    """
    device = torch.device('cpu')  # models are on the CPU

    print('Evaluating models ...')

    # % generate test dataset(s) / conditions
    all_test_data = dg.generate_test_dataset(constants.PARAMS)

    # % set up paths and save dataset(s)
    model_path = constants.PARAMS['FULL_PATH']
    load_path = model_path + 'saved_models/'
    eval_path = model_path + 'evaluation_data/'
    # valid trials
    valid_path = eval_path + 'valid_trials/'

    check_path(eval_path)
    check_path(valid_path)  # check if paths exist, if not - create

    # keep the nested if statements for better legibility
    if constants.PARAMS['cue_validity'] < 1:
        # Experiment 4, 75% and 50% validity conditions
        # valid trials
        test_data_valid_trials = dg.subset_of_trials(constants.PARAMS,
                                                     all_test_data,
                                                     all_test_data['valid_trial_ixs'])
        retnet.save_data(test_data_valid_trials, valid_path + 'test_dataset')

        # invalid trials
        invalid_path = eval_path + 'invalid_trials/'
        check_path(invalid_path)
        test_data_invalid_trials = dg.subset_of_trials(constants.PARAMS,
                                                       all_test_data,
                                                       all_test_data['invalid_trial_ixs'])
        retnet.save_data(test_data_invalid_trials, invalid_path + 'test_dataset')
    else:
        if constants.PARAMS['experiment_number'] == 1 or constants.PARAMS['experiment_number'] == 3:
            # Experiments 1 and 3
            # trained delay
            retnet.save_data(all_test_data['trained'], valid_path + 'test_dataset')

            # out-of-range temporal generalisation
            out_range_path = valid_path + 'out_range_tempGen/'
            check_path(out_range_path)
            retnet.save_data(all_test_data['out-of-range'], out_range_path + 'test_dataset')

            # out-of-range shorter
            out_range_path_shorter = valid_path + 'out_range_tempGen_shorter/'
            check_path(out_range_path_shorter)
            retnet.save_data(all_test_data['out-of-range-shorter'], out_range_path_shorter + 'test_dataset')

            # in-range temporal generalisation
            in_range_path = valid_path + 'in_range_tempGen/'
            check_path(in_range_path)
            retnet.save_data(all_test_data['in-range'], in_range_path + 'test_dataset')
        else:
            # Expt 4 with cue_validity = 1, Expt 2
            retnet.save_data(all_test_data, valid_path + 'test_dataset')

    # % evaluate models on test dataset
    for m in np.arange(constants.PARAMS['n_models']):
        print('Model %d' % m)

        constants.PARAMS['model_number'] = m
        model = retnet.load_model(load_path, constants.PARAMS, device)

        if constants.PARAMS['cue_validity'] < 1:
            # evaluate separately for valid and invalid trials
            _, _, _ = retnet.eval_model(model,
                                        test_data_valid_trials,
                                        constants.PARAMS,
                                        valid_path,
                                        trial_type='valid')

            # invalid
            _, _, _ = retnet.eval_model(model,
                                        test_data_invalid_trials,
                                        constants.PARAMS,
                                        invalid_path,
                                        trial_type='invalid')

        else:
            if constants.PARAMS['experiment_number'] == 1 or constants.PARAMS['experiment_number'] == 3:
                # test on the/one of the delay lengths used in training
                constants.PARAMS = dg.update_time_params(constants.PARAMS,
                                                         constants.PARAMS['test_delay_lengths'][0])
                _, _, _ = retnet.eval_model(model, all_test_data['trained'], constants.PARAMS, valid_path)

                # test out-of-range temporal generalisation (longer than maximum trained delay)
                constants.PARAMS = dg.update_time_params(constants.PARAMS,
                                                         constants.PARAMS['test_delay_lengths'][1])
                _, _, _ = retnet.eval_model(model,
                                            all_test_data['out-of-range'],
                                            all_test_data['out-of-range']['params'],
                                            out_range_path)

                # test out-of-range temporal generalisation (shorter than minimum trained delay)
                constants.PARAMS = dg.update_time_params(constants.PARAMS, 1)
                _, _, _ = retnet.eval_model(model,
                                            all_test_data['out-of-range-shorter'],
                                            all_test_data['out-of-range-shorter']['params'],
                                            out_range_path_shorter)

                # test in-range temporal generalisation
                constants.PARAMS = \
                    dg.update_time_params(constants.PARAMS,
                                          constants.PARAMS['test_delay_lengths'][2])
                _, _, _ = retnet.eval_model(model,
                                            all_test_data['in-range'],
                                            all_test_data['in-range']['params'],
                                            in_range_path)
            else:
                # expt 4 with cue_validity = 1, expt 2
                # only valid trials with the trained delay length
                _, _, _ = retnet.eval_model(model, all_test_data, constants.PARAMS, valid_path)

    retnet.export_behav_data_to_matlab(constants.PARAMS)
    return


def analyse_data(constants):
    """
    Run the whole analysis pipeline for a given Experiment.

    :param module constants: A Python module containing constants and configuration data for the simulation.
        Configuration files are saved in the 'constants/' directory.
    """
    # get all test conditions and their corresponding data folder names
    test_conditions, folder_names = dg.generate_test_conditions()

    # get full test folder paths - relevant for the behavioural analysis
    test_paths = [constants.PARAMS['RAW_DATA_PATH'] + f for f in folder_names[constants.PARAMS['expt_key']]]

    # Get the data (binned into B colour bins) for the rep geom and AI analyses
    _, all_data = ppc.get_all_binned_data(constants)

    # To control the specific analyses being run, comment/uncomment the relevant lines below.
    if constants.PARAMS['experiment_number'] == 1:
        if constants.PARAMS['ai_vs_learning_speed']:
            # Run the AI vs learning speed analysis, reported in Figure 4D

            # run the 'Cued', 'Uncued' and 'Cued-uncued' AI analyses
            for geometry_name in ['cued', 'uncued', 'cued_uncued']:
                ai.run_AI_analysis(constants, all_data, geometry_name)

                # get the learning speed metrics, and export them alongside the AI measures into a csv file, to be
                # analysed in JASP
                ai_learning_speed.get_all_data(constants)

        else:
            # 1. Behaviour (Figure 1D and supplementary Figure S3A, top row)
            # test_paths = [valid_path, out_range_path, out_range_path_shorter, in_range_path]
            behav.run_behav_analysis(constants, test_conditions[constants.PARAMS['expt_key']], test_paths)
            #
            # 2. Cued geometry (Figure 2)
            # get geometry (angles)
            rg.run_cued_geom_analysis(constants, all_data)
            # get AI cued (also runs the rotated/unrotated cued plane analysis)
            ai.run_AI_analysis(constants, all_data, 'cued')
            # additional analyses reported in the supplementary materials:
            # run the rotated/unrotated cued plane analysis (with angles, Figure S1D)
            # rg.run_unrotated_rotated_geometry(constants)
            # decoding analysis to confirm the single readout hypothesis
            # dec.run_cg_decoding_cued_analysis(constants, trial_type='valid')

            # 3. Uncued geometry (Figure 3, left column)
            # run uncued item decoding
            dec.run_decoding_uncued_analysis(constants)
            # run the uncued geometry
            rg.run_uncued_geom_analysis(constants, all_data)
            # AI
            ai.run_AI_analysis(constants, all_data, 'uncued')

            # 4. Cued/Uncued geometry (Figure 3, right column)
            # run the cued/uncued geometry
            rg.run_cued_uncued_geom_analysis(constants, all_data)
            # AI
            ai.run_AI_analysis(constants, all_data, 'cued_uncued')

            # 5. CDI analysis (Figure 3H)
            rg.run_CDI_analysis(constants)
            # analogous analysis with decoding (reported in the supplementary materials, Figure S1C)
            # dec.run_colour_discrim_analysis(constants)

            # 6. Connectivity analysis (Figure 4B)
            ld.run_retro_weight_analysis(constants)

            # 7. Learning dynamics analysis (Figure 4 A,C)
            ld.run_learning_dynamics_analysis(constants)

            # 8. run the cross-temporal generalisation analysis of the Cued item representations (Figure 6A, left)
            # dec.run_ctg_analysis(constants)
    elif constants.PARAMS['experiment_number'] == 2:
        # AI Cued geometry analysis reported in Fig. 5
        ai.run_AI_analysis_experiment_2(constants)
        # analogous analysis with plane angles theta
        # rg.run_cued_geometry_experiment_2(constants)
    elif constants.PARAMS['experiment_number'] == 3:
        # run the cross-temporal generalisation analysis of cued item representations
        # if we want to see the plot for the other delay_length conditions, pass a different delay_length value
        # e.g. for the in-range temporal generalisation condition, set to 4 cycles; for the out-of-training range
        # delay, set it 10 cycles
        dec.run_ctg_analysis(constants, delay_length=7)  # Figure 4A, right
        # compare the memory maintenance mechanisms between experiments 1 and 3
        dec.run_maintenance_mechanism_analysis()  # Figure 4B

        # run the Cued geometry analysis - Figure 4C-E
        rg.run_cued_geom_analysis(constants, all_data)

        # Other analyses reported in the supplementary materials
        # behavioural analysis Figure S3A, bottom row
        behav.run_behav_analysis(constants, test_conditions[constants.PARAMS['expt_key']], test_paths)
        # geometry analyses with angles
        # rg.run_uncued_geom_analysis(constants, all_data)
        # rg.run_cued_uncued_geom_analysis(constants, all_data)
        # rg.run_unrotated_rotated_geometry(constants)
        # AI analyses
        # ai.run_AI_analysis(constants, all_data, 'cued')

    elif constants.PARAMS['experiment_number'] == 4:
        if constants.PARAMS['ai_vs_learning_speed']:
            # Run the AI vs learning speed analysis - reported in Fig. 8C
            # get and export the data for the AI vs learning speed regression analysis
            for geometry_name in ['cued', 'uncued', 'cued_uncued']:
                # run the 'Cued', 'Uncued' and 'Cued-uncued' AI analyses
                ai.run_AI_analysis(constants, all_data, geometry_name)

                # get the learning speed metrics, and export them alongside the AI measures into a csv file, to be
                # analysed in JASP
                ai_learning_speed.get_all_data(constants)
        else:
            # run the behavioural analysis, comparing performance on valid and invalid trials (for conditions with
            # probabilistic cues; Figure 7B). Plot the mixture model parameters fitted to behavioural data (Figure
            # 7C, note the parameters are fitted in Matlab).
            behav.run_behav_analysis(constants, test_conditions[constants.PARAMS['expt_key']], test_paths)

            # run the Cued geometry analysis - Figure 8A
            rg.run_cued_geom_analysis(constants, all_data)

            # run the CDI analysis - Figure 8B
            rg.run_CDI_analysis(constants)

            # export the CDI data to csv to correlate with the mixture model params in JASP
            rg.export_CDI_and_mixture_model_params_validity(constants)
    else:
        raise ValueError('Wrong experiment number. The manuscript contains Experiments 1-4 only.')

    return


def run_experiment(constants, parallelise_flag, train_flag, eval_flag, analysis_flag):
    """
    Run the whole simulation for a given Experiment. Trains and evaluates the models, and analyses the data.

    :param module constants: A Python module containing constants and configuration data for the simulation.
        Configuration files are saved in the 'constants/' directory.
    :param bool parallelise_flag: If True, will train models in parallel (provided that the train_flag is set to True).
    :param bool train_flag: If True, will train models.
    :param bool eval_flag: If True, will evaluate trained models.
    :param bool analysis_flag: If True, will run the entire analysis pipeline for the given Experiment. Results will be
        printed to the console. If you would like to save the plots to file, change the PLOT_PARAMS['save_plots'] field
         in the experimental constants module to True.
    :return:
    """
    device = torch.device('cpu')  # train on CPU to ensure deterministic behaviour

    # train models on the specified device (in parallel if parallelise_flag is True)
    if train_flag:
        train_models(constants, parallelise_flag, device)

    # create test dataset and evaluate models after freezing weights
    if eval_flag:
        evaluate_models(constants)

    # run analysis pipeline
    if analysis_flag:
        analyse_data(constants)

    return
#%% run the experiment


if __name__ == "__main__":
    # Pick experiment: pick the appropriate constants module (containing the experiment parameters, paths and other
    # configurations) from the 'constants/' folder.
    # E.g. for Experiment 1, use the line below
    import constants.constants_expt1 as c
    # Run the full analysis pipeline
    run_experiment(c, parallelise_flag=False, train_flag=False, eval_flag=False, analysis_flag=True)
