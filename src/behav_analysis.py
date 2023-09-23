#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 18:51:06 2021

This file contains functions implementing the analysis of model behaviour (i.e., choices).

@author: emilia
"""

import numpy as np
import torch
import warnings
import pickle
import pycircstat
import pandas as pd
from scipy.stats import vonmises
import src.retrocue_model as retnet
from src.stats import get_sem
from src.custom_plot import plot_all_error_data, plot_mixture_model_params_validity
import src.helpers as helpers


# %% define functions


def angle_to_vec(angles):
    """
    Helper function to convert an array of angles into their unit-circle vector representations.
    
    :param torch.Tensor angles: Input array in radians.
    :return angles_vectors : torch.Tensor, first dimension corresponds to the x- and y-coordinates of the angles.
    """
    if type(angles) is not torch.Tensor:
        # convert to a torch tensor
        angles = torch.tensor(angles)

    # check that ang_errors contains values in radians
    if np.abs(angles).max() > 2 * np.pi:
        warnings.warn('Large angular values detected. Check that the input array contains angular errors in radians, '
                      'if so - wrap the values to the [-pi, pi] interval. Converting the values to radians.', Warning)
        angles = np.radians(angles)

    angles_vectors = torch.stack((np.cos(angles), np.sin(angles)))
    return angles_vectors


def wrap_angle(angle):
    """
    Wraps angle(s) to be within [-pi, pi).

    :param np.ndarray or torch.Tensor angle: Input angle(s) in radians
    :return angle_wrapped : np.ndarray, torch.Tensor
    """
    angle_wrapped = (angle + np.pi) % (2 * np.pi) - np.pi
    return angle_wrapped


def get_angular_errors(choices, probed_colours):
    """
    Get the angular error values for each colour stimulus, for a given model.
    Errors are calculated for a given colour, irrespective of the location 
    where it was shown. Error is defined as the angular distance between the model
    response (choice) and the ground truth probed colour value, in radians, wrapped
    to lie within the [-pi, pi] interval.
    
    :param torch.Tensor choices : (n_trials, ) Trial-wise model choices.
    :param torch.Tensor probed_colours : (n_trials, ) Trial-wise probed colours.
    :return ang_errors : torch.Tensor (n output channels, n trials per probed colour)
        Angular error values [radians], sorted into columns according to the probed colour.
        
    """
    assert type(choices) is torch.Tensor, 'choices should be a torch.Tensor'
    assert type(probed_colours) is torch.Tensor, 'probed_colours should be a torch.Tensor'
    assert choices.shape == probed_colours.shape, 'choices and probed_colours must have the same size'
    assert len(choices.shape) == 1, 'choices must be a 1-dimensional torch.Tensor'
    assert len(probed_colours.shape) == 1, 'probed_colours must be a 1-dimensional torch.Tensor'

    n_trials = probed_colours.shape[0]

    # sort the trial indices for each probed colour
    colours = np.unique(probed_colours)
    n_colours = len(colours)
    colour_ix = [None] * len(colours)
    for i, c in enumerate(colours):
        colour_ix[i] = np.where(probed_colours == c)[0]

    # Calculate angular errors (difference between model choices and target responses)
    ang_errors = torch.empty((len(colours), n_trials // len(colours)))
    # error for each colour, irrespective of its location
    for c in range(n_colours):
        ang_errors[c, :] = choices[colour_ix[c]] - probed_colours[colour_ix[c]]

    # Wrap (angular) errors to [-pi, pi)
    ang_errors = wrap_angle(ang_errors)

    return ang_errors


def get_abs_err_mean_sd(ang_errors):
    """
    Calculate the mean and standard deviation of absolute errors, in degrees.
    :param torch.Tensor ang_errors: data array with angular error values in radians. If multidimensional, will be
        flattened.
    :return: (mean, std) of absolute angular errors
    """
    if type(ang_errors) is not torch.Tensor:
        # convert to a torch tensor
        ang_errors = torch.tensor(ang_errors)

    # check that ang_errors contains values in radians
    if np.abs(ang_errors).max() > np.pi:
        warnings.warn('Large angular values detected. Check that the input array contains angular errors in radians, '
                      'if so - wrap the values to the [-pi, pi] interval. Converting the values to radians.', Warning)
        ang_errors = np.radians(ang_errors)

    # convert into absolute errors in degrees
    abs_ang_errors = np.abs(np.degrees(ang_errors.view(-1).numpy()))

    # calculate mean and sd
    return abs_ang_errors.mean(), abs_ang_errors.std()


def get_circ_mean_kappa(ang_errors):
    """
       Calculate the circular mean and kappa of angles, in degrees.

       :param torch.Tensor ang_errors: data array with angular error values in radians. If multidimensional, will be
            flattened.
       :return: (circular mean, kappa) of angular errors in degrees

       """
    if type(ang_errors) is not torch.Tensor:
        # convert to a torch tensor
        ang_errors = torch.tensor(ang_errors)

    # check that ang_errors contains values in radians
    if np.abs(ang_errors).max() > np.pi:
        warnings.warn('Large angular values detected. Check that the input array contains angular errors in radians, '
                      'if so - wrap the values to the [-pi, pi] interval. Converting the values to radians.', Warning)
        ang_errors = np.radians(ang_errors)

    # get circular mean (in [-pi, pi]) and kappa of the errors, in degrees
    circ_mean_err = np.degrees(wrap_angle(pycircstat.mean(ang_errors.view(-1).numpy())))
    kappa_err = np.degrees(pycircstat.distributions.kappa(ang_errors.view(-1).numpy()))
    return circ_mean_err, kappa_err


def fit_von_mises_distr(ang_errors):
    """
    Fit a vonMises distribution to the angular error data.

    :param torch.Tensor ang_errors: data array with angular error values in radians. If multidimensional, will be
        flattened.
    :return: fitted params: dictionary with the fitted 'kappa', 'mu' and 'scale' parameters, values expressed in radians

    .. note::  When fitting the distribution, need to fix the 'scale' parameter to 1.0, otherwise it fits some kind
     of high frequency oscillatory function (can't recover parameters even on simulated data).
    """
    if type(ang_errors) is not torch.Tensor:
        # convert to a torch tensor
        ang_errors = torch.tensor(ang_errors)

    # check that ang_errors contains values in radians
    if np.abs(ang_errors).max() > np.pi:
        warnings.warn('Large angular values detected. Check that the input array contains angular errors in radians, '
                      'if so - wrap the values to the [-pi, pi] interval. Converting the values to radians.', Warning)
        ang_errors = np.radians(ang_errors)

    # Fit the von mises distribution
    # Note - need to fix the 'scale' parameter to 1.0, in order to be able to fit the pdf, otherwise it fits some kind
    # of high frequency oscillatory function (can't recover parameters even on simulated data).
    fitted_params = {}
    fitted_params['kappa'], fitted_params['mu'], fitted_params['scale'] = vonmises.fit(ang_errors.view(-1).numpy(),
                                                                                       fscale=1.0)
    return fitted_params


def bin_errors(ang_errors):
    """ Bin angular error values on the [-180, 180] degree interval into 40-degree wide bins and calculate the error
    density for each bin.

    :param torch.Tensor ang_errors: Input tensor with angular error values in radians. If multidimensional, will be
        flattened.
    :returns:
        binned_errors : (n_bins, ) Error density values for each bin
        bin_centres : (n_bins, ) Bin centres in radians
        b_max : Upper end of the range of angular error values in radians
    """

    # pick bins
    b_max = np.pi  # max value / end of range
    bin_edges = np.linspace(-b_max, b_max, 10)

    bin_centres = []
    for i in range(len(bin_edges) - 1):
        bin_centres.append(np.mean(bin_edges[i:i + 2]))
    bin_centres = np.array(bin_centres)

    # bin errors
    binned_errors, _ = np.histogram((ang_errors.view(-1).numpy()), bins=bin_edges, density=True)

    return binned_errors, bin_centres, b_max


def get_all_errors(constants, load_path):
    """
    Loop through all models to get the trial-wise and binned angular error values.

    :param module constants: A Python module containing constants and configuration data for the simulation.
    :param str load_path: path to the data folder when model choices / responses are saved.
    :return: error_results: dictionary with the following keys:
        'ang_errors' : torch.Tensor (n_models, n_colours, n_trials_per_colour) - angular error values, sorted by model,
            probed colour and trial
        'binned_errors' : torch.Tensor (n_models, n_bins) - density of errors for each angular error bin, sorted by
            models and bins
        'bin_centres' : list, (n_bins, ) - list of angular error bin centres
        'bin_max' : float -  max angular error value used when binning
        'probed_colours' : torch.Tensor (n_trials,) - probed colour value [radians] on each test trial
        'unprobed_colours' :  torch.Tensor (n_trials,) - unprobed colour value [radians] on each test trial

    """
    ang_errors, mean_abs_err, sd_abs_err, binned_errors = [], [], [], []

    for model_number in np.arange(constants.PARAMS['n_models']):
        # load model choice data
        f = open(f"{load_path}model_outputs_model{model_number}.pckl", 'rb')
        model_outputs = pickle.load(f)
        f.close()

        # calculate angular errors
        ang_errors.append([])
        ang_errors[model_number] = get_angular_errors(model_outputs['choices'],
                                                      model_outputs['labels']['probed_colour'])

        # bin errors
        binned_errors.append([])
        binned_errors[model_number], bin_centres, b = bin_errors(ang_errors[model_number])

        # get mean and SEM of absolute errors
        mean_abs_err.append([]), sd_abs_err.append([])
        mean_abs_err[model_number], sd_abs_err[model_number] = get_abs_err_mean_sd(ang_errors[model_number])

    ang_errors = torch.stack(ang_errors)
    binned_errors = np.stack(binned_errors)

    error_results = {'ang_errors': ang_errors,
                     'binned_errors': binned_errors,
                     'bin_centres': bin_centres,
                     'bin_max': b,
                     'probed_colours': model_outputs['labels']['probed_colour'],
                     'unprobed_colours': model_outputs['labels']['unprobed_colour']}

    return error_results


def get_all_error_stats(error_results):
    """ Calculate some statistics on the errors pooled from all models: (1) mean and sd of absolute mean error,
    (2) mean and sem of binned errors, (3) vonmises distribution parameters.

    :param dict error_results: dictionary containing the angular error analysis results. Should contain the following
        keys:
            'ang_errors': torch.Tensor (n_models, n_colours, n_trials_per_colour) - angular error values, sorted by
                model, probed colour and trial
            'binned_errors': torch.Tensor (n_models, n_bins) - density of errors for each angular error bin, sorted by
                models and bins
    :return: error_stats: dictionary containing the statistics calculated on angular error data from all models.
        Contains the following keys:
            'mean_abs_err': scalar mean of absolute errors, pooled from all models
            'sd_abs_err': scalar sd of absolute errors, pooled from all models
            'mean_errs': mean (across all models) error density for each error bin
            'sem_errs': sem (across all models) of error density for each error bin
            'fitted_params': dictionary with the fitted 'kappa', 'mu' and 'scale' parameters,
                                values expressed in radians
    """

    # get the mean and sem of binned errors across models
    error_stats = {}

    # get mean and sd of absolute errors across all models
    error_stats['mean_abs_err'], error_stats['sd_abs_err'] = get_abs_err_mean_sd(error_results['ang_errors'])

    # get mean and sem of binned errors
    error_stats['mean_errs'] = error_results['binned_errors'].mean(0)
    error_stats['sem_errs'] = get_sem(error_results['binned_errors'])

    # fit distribution to data pooled from all models
    error_stats['fitted_params'] = fit_von_mises_distr(error_results['ang_errors'].view(-1))

    return error_stats


# %% io


def save_error_data_to_file(results, test_path):
    """
    Saves the angular error data dictionary to file. Each key is saved as a separate datafile.

    :param dict results: A dictionary containing the results of the angular error analysis.
    :param str test_path: file destination path
    """
    # % save data
    for key in results.keys():
        retnet.save_data(results[key], f"{test_path}{key}")

    return


def load_mixture_model_params(constants):
    """ Loads the fitted mixture model parameters from file and packs them into a dictionary. Dictionary keys correspond
    to the parameters: 'K', 'pT', 'pNT' and 'pU'. Each key contains a pandas dataframe with the fitted param values for
    valid and invalid trials (stacked on top of one another as rows).

    :param module constants: A Python module containing constants and configuration data for the simulation.
        Must contain a 'PARAMS' dictionary, containing 'cue_validity' and 'n_models' keys.
    :return mixture_param_data_dict: dictionary, keys are named after the mixture model parameters ('K', 'pT', 'pNT'
        and 'pU'), each contains a pandas dataframe with 'trial_type' (valid, invalid), 'condition' (cue validity: 0.5,
        0.75) and 'param' (e.g., 'K') columns. Note data from each model is saved in 2 separate rows (one for valid
        trial estimates, and another for invalid trials).
    """

    if constants.PARAMS['cue_validity'] == 1:
        raise NotImplementedError('Mixture models only fitted for data from Experiment 4, cue validity < 1.')

    mixture_params = ['K', 'pT', 'pNT', 'pU']
    trial_type = ['valid', 'invalid']

    n_models = constants.PARAMS['n_models']
    mixture_param_data_dict = {'model_nr': np.tile(np.arange(n_models * 2), 2)}

    for param in mixture_params:
        file_path = f"{constants.PARAMS['MATLAB_PATH']}{param}_table.csv"

        # check data file exists
        if not helpers.check_file_exists(file_path):
            raise FileNotFoundError("The file '{file_path}' does not exist. Make sure that the mixture model "
                                    "parameters have been fit (in MATLAB) and saved to file.")

        # load data
        data = pd.read_csv(file_path)

        # reshape data into a pandas dataframe - for plotting in seaborn
        condition = np.tile(data['condition'].to_numpy(), 2)  # cue validity
        t_type = [[trial_type[t]] * n_models * 2 for t in range(len(trial_type))]
        t_type = np.reshape(t_type, -1)
        # create '{parameter}_valid' and '{parameter}_invalid' labels'
        label1 = f"{param}_{trial_type[0]}"
        label2 = f"{param}_{trial_type[1]}"

        param_vals = np.concatenate((data[label1].to_numpy(), data[label2].to_numpy()))
        data_reshaped = np.stack((t_type, condition, param_vals), 1)
        data_reshaped = pd.DataFrame(data_reshaped, columns=['trial_type', 'condition', param])
        data_reshaped['trial_type'] = data_reshaped['trial_type'].astype(str)
        data_reshaped['condition'] = data_reshaped['condition'].astype(str)
        data_reshaped[param] = data_reshaped[param].astype(float)
        # save to big dictionary
        mixture_param_data_dict[param] = data_reshaped

    return mixture_param_data_dict


def run_behav_analysis(constants, expt_test_conditions, expt_test_paths):
    """
    Runs the entire behavioural analysis.

    The steps are:
    1) For each test condition and model:
        - calculate the angular error values on each trial
        - bin the error values into 40-degree wide bins and extract the error densities
    2) For each test condition : calculate some statistics on the error data from all models, print the mean and sd of
        absolute error
    3) Save data from each condition to file
    4) For each test condition, plot the error distribution (average +-SEM across models) of binned errors alongside the
        fitted von-Mises distribution
    5) For probabilistic conditions (where cue validity < 1) from experiment 4: load and plot the mixture model
        parameters fitted (in MATLAB) to the data from valid and invalid trials.

    :param module constants: A Python module containing constants and configuration data for the simulation.
    :param list expt_test_conditions: Test conditions for the current experiment.
    :param list expt_test_paths: Full paths to the directories containing test data for each of the test condition.
    :return:

    ..note :: expt_test_conditions should be obtained by passing the expt_key into the test_conditions dictionary
        generated by the generate_test_conditions function from src.generate_data_vonMises. expt_test_paths should
        incorporate (1) the path to the project directory and (2) the folder_names generated by the same function
        (i.e., generate_test_conditions), accessed by passing the expt_key key.

    :Example:
        To run the analysis for Experiment 1, we would first run the generate_test_conditions from
        src.generate_data_vonMises to obtain the test_conditions and folder_names dictionaries. Then, to obtain the
        list of expt_test_conditions, we would pass the expt_key 'expt_1' (available under constants.PARAMS['expt_key']
        to test_conditions:
            expt_test_conditions = test_conditions['expt_1']
        To obtain the list of expt_test_paths, we would run the following:
            test_paths = [constants.PARAMS['DATA_PATH'] + f for f in folder_names[constants.PARAMS['expt_key']]]
    """
    all_results = {}

    print('RUNNING THE BEHAVIOURAL ANALYSIS')
    for i, (condition, path) in enumerate(zip(expt_test_conditions, expt_test_paths)):
        all_results[condition] = {}

        # get the raw angular error values for all models, as well as the density values for binned errors
        error_results = get_all_errors(constants, path)

        # save current results (raw and binned angular errors, probed and unprobed colours) to the big dictionary
        for key in error_results.keys():
            all_results[condition][key] = error_results[key]

        # get the angular error statistics: mean and sem of binned errors across models, von mises distribution
        # parameter fits and mean and sd of absolute errors across all models
        error_stats = get_all_error_stats(error_results)

        # append these to the big dictionary
        for key in error_stats.keys():
            all_results[condition][key] = error_stats[key]

        # print the statistics for this condition
        print(f'Test condition: {condition}. Mean absolute angular error across all models: %.2f, SD: %.2f degrees'
              % (all_results[condition]['mean_abs_err'], all_results[condition]['sd_abs_err']))

        # save data to file
        save_error_data_to_file(all_results[condition], path)

    # plot error distributions
    plot_all_error_data(constants, expt_test_conditions, all_results)
    # plt.savefig(constants.PARAMS['FIG_PATH']+'error_distribution.png')

    if constants.PARAMS['cue_validity'] < 1:
        # load mixture model parameters
        mixture_param_data_dict = load_mixture_model_params(constants)

        # plot mixture model parameters
        plot_mixture_model_params_validity(mixture_param_data_dict)

    return
