#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 11:43:54 2021
This script containts the functions implementing the learning dynamics and connectivity analyses (Fig. 4 in the
manuscript).

@author: emilia
"""

import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema

import src.retrocue_model as retnet
import src.custom_plot as plotter
import src.stats as stats
import src.generate_data_von_mises as dg
import src.subspace_alignment_index as ai
from src.subspace import Geometry
from src.helpers import check_path

# %% learning dynamics analysis


def get_dLoss_dt(loss_vals, smooth_sd, smooth_loss=True):
    """
    Calculate the derivative of the loss function wrt time. Used for finding learning plateaus.

    :param torch.Tensor loss_vals: loss values for every epoch (averaged cross all training examples).
    :param float smooth_sd: Standard deviation value for the Gaussian filter smoothing the loss
    :param bool smooth_loss: Optional. If True, applies a Gaussian filter to smooth the loss. Default is True.
    :return: dLoss, loss_clean: Tensors containing the derivative of the loss and the smoothed loss values

    """
    assert len(loss_vals.shape) == 1, 'Loss_vals must be a 1-dimensional array'

    # convolve with a Gaussian filter to smooth the loss curve
    if smooth_loss:
        loss_clean = gaussian_filter1d(loss_vals, smooth_sd)
    else:
        loss_clean = loss_vals
    loss_clean = torch.tensor(loss_clean)

    # calculate the derivative
    dLoss = torch.zeros(loss_clean.shape[0] - 1)
    for i in range(loss_clean.shape[0] - 1):
        dLoss[i] = loss_clean[i + 1] - loss_clean[i]
    return dLoss, loss_clean


def load_model_loss(constants, model_number):
    """
    Loads the training data dictionary for a single model.
    :param constants:
    :param int model_number: Number of the model for which to load the data.
    :return: track_training - Training data dictionary.
    :rtype: dict
    """
    data_path = constants.PARAMS['FULL_PATH']
    # load data
    with open(f"{data_path}training_data/training_data_model{model_number}.pckl", 'rb') as f:
        track_training = pickle.load(f)

    return track_training


def find_learning_plateau(dLoss, loss_clean):
    """
    Find the first learning plateau for a single model.

    :param torch.Tensor dLoss: Derivative of the training loss
    :param torch.Tensor loss_clean: Smoothed training loss values
    :return: plateau_ix: epoch index of the first learning plateau
    """
    # find local minima
    ix = argrelextrema(np.array(dLoss), np.greater)[0]

    # exclude the start plateau - any timepoints were the loss value is
    # within a 5% margin from the initial value
    ix = np.setdiff1d(ix, ix[np.where(loss_clean[ix] >= loss_clean[0] * .95)[0]])

    plateau_ix = ix[0]
    return plateau_ix


def partial_train_and_eval(constants, model_number, plateau_ix):
    """
    Train and evaluate the untrained and partially-trained (until loss plateau) models.

    :param module constants: A Python module containing constants and configuration data for the simulation.
    :param int model_number: Number of the model for which to run the analysis.
    :param int plateau_ix: Trial index of the plateau stage.

    """
    assert constants.PARAMS['experiment_number'] is 1, 'Analysis only implemented for Experiment 1'
    print('PARTIAL TRAINING ANALYSIS')

    eval_path = f"{constants.PARAMS['FULL_PATH']}pca_data/valid_trials/partial_training/"
    check_path(eval_path)
    check_path(f"{eval_path}untrained/")  # create paths
    check_path(f"{eval_path}plateau/")  # create paths

    # get the test dataset
    all_test_data = dg.generate_test_dataset(constants.PARAMS)
    test_data = all_test_data['trained']
    device = torch.device('cpu')

    # set seed for reproducibility - controls both the initialisation and trial sequences
    constants.PARAMS['model_number'] = model_number
    torch.manual_seed(constants.PARAMS['model_number'])

    # % initialise model
    model = retnet.RNN(constants.PARAMS, device)
    print('Model %d' % model_number)

    # % eval untrained
    print('Evaluating untrained')
    _, _, _, = retnet.eval_model(model, test_data, constants.PARAMS, eval_path + 'untrained/')

    # % learning plateau
    # train model
    print('Training up to the plateau')
    constants.PARAMS['n_epochs'] = plateau_ix
    model, _ = retnet.train_model(constants.PARAMS, constants.TRAINING_DATA, device)
    # evaluate
    print('Evaluating partially trained')
    _, _, _, = \
        retnet.eval_model(model, test_data, constants.PARAMS, eval_path + 'plateau/')

    return


def get_model_pca_data(constants, model_number):
    """
    Load the evaluation datasets (averaged across uncued items and binned into colour bins) for a given model trained to
    different extents.

    :param module constants: A Python module containing constants and configuration data for the simulation.
    :param int model_number: Number of the model for which to load the data.
    :return: all_data: dictionary containing the training stage name keys ('untrained', 'plateau', 'trained'), each
        containing the evaluation dataset binned into colour bins (i.e., the pca_data structure).
    """
    data_path = f"{constants.PARAMS['FULL_PATH']}pca_data/valid_trials/"
    data_folders = ['partial_training/untrained/', 'partial_training/plateau/', '']
    conditions = ['untrained', 'plateau', 'trained']
    all_data = {}
    for condition, data_folder in zip(conditions, data_folders):
        # load pca data
        with open(f"{data_path}{data_folder}pca_data_model{model_number}.pckl", 'rb') as f:
            all_data[condition] = pickle.load(f)
    return all_data


def get_cued_geom_measures(constants, model_data):
    """
    Get the Cued geometry measures (theta, PVE, AI) for a given dataset.

    :param module constants: A Python module containing constants and configuration data for the simulation.
    :param np.ndarray model_data: data array containing data from a single timepoint, shape: (n_conditions, n_neurons)
    :return:
    """
    # fit the geometry to data from model
    geometry = Geometry(model_data, constants)
    geometry.get_geometry()

    # calculate the AI
    ai_table = ai.get_AI_all_dims(constants, model_data, max_dim=3)

    # extract the angle measures
    return geometry.theta_degrees, geometry.PVEs, ai_table


def run_learning_plateau_pipeline_single_model(constants, model_number):
    """
    Run the learning plateau analysis pipeline for a single model. Steps include loading the training loss, finding the
    index of the plateau timepoint, evaluating the model at different points of training ('untrained' and at 'plateau'),
    and running the Cued geometry angle and AI analyses for each training stage.

    :param module constants: A Python module containing constants and configuration data for the simulation.
    :param int model_number: Number of the model for which to run the pipeline.
    :return: plateau_ix, all_theta, all_PVEs, all_AI: index of the plateau timepoint and data arrays containing the
        geometry measures (theta, PVEs, AI) for all training stages and delay intervals
    """
    # load the data
    track_training = load_model_loss(constants, model_number)

    # calculate the derivative of the loss
    dLoss, loss_clean = get_dLoss_dt(track_training['loss_epoch'], 4)

    # find the plateaus
    plateau_ix = find_learning_plateau(dLoss, loss_clean)

    if model_number == 0:
        # plot the loss and loss derivative, alongside the identified plateau for an example model
        plotter.plot_loss_and_plateau(track_training['loss_epoch'], loss_clean, dLoss, plateau_ix)
        plt.suptitle('Training loss and its derivative for an example model')
        if constants.PLOT_PARAMS['save_plots']:
            plt.savefig(f"{constants.PARAMS['FIG_PATH']}example_training_loss_and_derivative.png")
            plt.savefig(f"{constants.PARAMS['FIG_PATH']}example_training_loss_and_derivative.svg")

    if model_number == 10:
        # plot the training loss with the training stages demarcated for an example model
        plotter.plot_example_loss_plot(plateau_ix, track_training['loss_epoch'])
        plt.savefig(constants.PARAMS['FIG_PATH'] + 'example_loss_plot.png')
        plt.savefig(constants.PARAMS['FIG_PATH'] + 'example_loss_plot.svg')
    # partial train model - uncomment the line below if you want to partial train the model again
    # partial_train_and_eval(constants, model_number, plateau_ix)

    # get all Cued pca data
    all_data = get_model_pca_data(constants, model_number)

    # run the Cued geometry analysis for the model at different training stages
    all_theta, all_PVEs, all_AI = [], [], []

    for i, stage in enumerate(all_data.keys()):
        for measure in [all_theta, all_PVEs, all_AI]:
            measure.append([])

        for delay in range(constants.PARAMS['n_delays']):
            theta, PVEs, ai_table = get_cued_geom_measures(constants, all_data[stage][f"delay{delay+1}"])
            all_theta[i].append(theta)
            all_PVEs[i].append(PVEs)
            all_AI[i].append(ai_table)

    all_theta = np.stack(all_theta)  # (n_stages, n_delays)
    all_PVEs = np.stack(all_PVEs)  # (n_stages, n_delays, n_PCs)
    all_AI = np.stack(all_AI)  # shape: (n_stages, n_delays, n_dims)

    return plateau_ix, all_theta, all_PVEs, all_AI


def run_learning_stages_all_models(constants):
    """
    Run the learning plateau analysis pipeline and collect the data into arrays.

    :param module constants: A Python module containing constants and configuration data for the simulation.
    :return: lc_plateau, all_theta, all_PVEs, all_AI:  indices of the plateau timepoints (n_models, ) and data arrays
        containing the geometry measures (theta, PVEs, AI) for all models, training stages, and delay intervals
    """
    # run the pipeline for each model
    lc_plateau, all_theta, all_PVEs, all_AI = [], [], [], []
    for model_number in range(constants.PARAMS['n_models']):
        plateau_ix, thetas, PVEs, ai_table = run_learning_plateau_pipeline_single_model(constants, model_number)
        lc_plateau.append(plateau_ix)
        all_theta.append(thetas)
        all_PVEs.append(PVEs)
        all_AI.append(ai_table)

    lc_plateau = np.stack(lc_plateau)
    all_theta = np.stack(all_theta)
    all_PVEs = np.stack(all_PVEs)
    all_AI = np.stack(all_AI)

    return lc_plateau, all_theta, all_PVEs, all_AI


def reformat_geometry_data(constants, all_theta, all_PVEs, all_AI):
    """
    Reformat the geometry data arrays intro nested dictionaries with superordinate keys corresponding to the training
    stage names, and subordinate keys corresponding to delay names (in the 'delay{delay_number}' format).

    :param module constants: A Python module containing constants and configuration data for the simulation.
    :param np.ndarray all_theta: Array containing the theta angles from all models, training stages and delay intervals,
        shape: (n_models, n_training_stages, n_delay_intervals)
    :param np.ndarray all_PVEs: Analogous array containing the PVE values, shape: (n_models, n_training_stages,
        n_delay_intervals, n_PCs)
    :param np.ndarray all_AI: Analogous array containing the PVE values, shape: (n_models, n_training_stages,
        n_delay_intervals, n_dims)
    :return: theta_dict, pve_dict, ai_dict
    """
    # reformat into dictionaries
    stages = ['untrained', 'plateau', 'trained']
    delay_names = [f'delay{delay+1}' for delay in range(constants.PARAMS['n_delays'])]
    theta_dict, pve_dict, ai_dict = {key: {} for key in stages}, {key: {} for key in stages}, \
        {key: {} for key in stages}

    for s, stage in enumerate(stages):
        for d, delay in enumerate(delay_names):
            theta_dict[stage][delay] = all_theta[:, s, d]
            pve_dict[stage][delay] = all_PVEs[:, s, d, :]
            ai_dict[stage][delay] = all_AI[:, s, d, :]

    return theta_dict, pve_dict, ai_dict


def run_learning_dynamics_analysis(constants):
    """
    Run the full learning dynamics analysis. Plots the theta angles and AI values for the Cued geometry for each
    training stage ('untrained', 'plateau' and 'trained') and delay interval.

    :param module constants: A Python module containing constants and configuration data for the simulation.
    """
    lc_plateau, all_theta, all_PVEs, all_AI = run_learning_stages_all_models(constants)
    # save plateau ixs
    with open(f"{constants.PARAMS['RESULTS_PATH']}valid_trials/training_data/lc_plateau.pckl", 'wb') as f:
        pickle.dump(lc_plateau, f)

    # reformat the other arrays into dictionaries
    theta_dict, pve_dict, ai_dict = reformat_geometry_data(constants, all_theta, all_PVEs, all_AI)

    # plot the angle and AI for each training stage and delay
    plotter.plot_learning_dynamics_angles(constants, theta_dict)
    plotter.plot_learning_dynamics_AI(constants, ai_dict, dim=2)

#%% connectivity analysis


def get_retro_weights(constants, model_number):
    """
    Extracts the retrocue-recurrent weight vectors for a given model.

    :param module constants: A Python module containing constants and configuration data for the simulation.
    :param int model_number: Number of the model for which to extract the weights.
    :return: cue1_weights, cue2_weights:  Vectors of the retrocue-recurrent weights for the retrocue 1 and 2 units, each
        of shape (n_recurrent, )
    """
    constants.PARAMS['model_number'] = model_number

    load_path = constants.PARAMS['FULL_PATH'] + 'saved_models/'
    device = torch.device('cpu')

    # model, net_type = retnet.define_model(constants.PARAMS,device)
    model = retnet.load_model(load_path, constants.PARAMS, device)

    # extract weights
    cue1_weights = model.inp.weight[:, 0].detach()
    cue2_weights = model.inp.weight[:, 1].detach()

    return cue1_weights, cue2_weights


def corr_retro_weights(constants):
    """
    Calculate the correlations between the two retrocue weight vectors for all models.

    :param module constants: A Python module containing constants and configuration data for the simulation.
    :return: r, p_val : Vectors of correlation coefficients and p-values for all models, each of shape (n_models, )
    """
    r = np.empty((constants.PARAMS['n_models']))
    p_val = np.empty((constants.PARAMS['n_models']))

    for model_number in range(constants.PARAMS['n_models']):
        # get weights
        cue1_weights, cue2_weights = get_retro_weights(constants, model_number)

        # correlate
        r[model_number], p_val[model_number] = pearsonr(cue1_weights, cue2_weights)

    return r, p_val


def run_retro_weight_analysis(constants):
    """
    Run the retrocueing weights correlation analysis. For each model, correlate the weights vectors corresponding to the
    retrocue1-hidden and retrocue2-hidden weights. Test if the correlation coefficients are significantly different from
    0 at the group level. Plot the two retrocueing weights vectors for an example model.

    :param module constants: A Python module containing constants and configuration data for the simulation.
    """
    # correlate the cue1 and cue2 weights for each model
    r, p = corr_retro_weights(constants)
    # test if the correlation coefficients are significantly different from zero at the group level
    stats.run_contrast_single_sample(r, 0, alt='two-sided')

    # plot weights for an example model - shown in Fig. 4B in the manuscript
    example_model = 0
    cue1_weights, cue2_weights = get_retro_weights(constants, example_model)
    plotter.plot_example_retro_weights(cue1_weights, cue2_weights, r[example_model])
    if constants.PLOT_PARAMS['save_plots']:
        plt.savefig(constants.PARAMS['FIG_PATH']+'example_retro_weights.png')
        plt.savefig(constants.PARAMS['FIG_PATH']+'example_retro_weights.svg')
