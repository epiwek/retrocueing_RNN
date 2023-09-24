#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 15:12:24 2021


"""

import pickle
import importlib
import numpy as np
import pandas as pd
import scipy.linalg as la
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import src.stats as stats
import src.preprocess_rep_geom_data as ppc
import src.plotting_funcs as plotter


# %% Compute alignment indices via PCA


def get_intrinsic_dim(constants, model_data, n_PCs=4):
    """
    Calculate the intrinsic dimensionality of the dataset to determine the required dimensionality for the AI analysis.
    Fits a PCA model separately to the first and second half of rows from ``model_data`` and calculates the proportion
    variance explained by each PC, then averages the estimates across the two PCA models.

    :param module constants: A Python module containing constants and configuration data for the simulation.
    :param numpy.ndarray model_data: Data array from a single model, shape: (n_conditions, n_neurons), where
        n_conditions = n_locations * n_colour_bins
    :param int n_PCs: Maximum number of PCs to use in calculating PVE values. Default is 4, which is the maximum
        dimensionality of the dataset containing activation patterns to 4 colour bins shown at a single location.

   :returns: Total PVE by increasing numbers of PCs.
   :rtype: numpy.ndarray

    """
    assert isinstance(n_PCs, int), 'n_PCs should be an integer'
    assert n_PCs > 0, 'n_PCs should be a positive integer'
    assert len(model_data.shape) == 2, 'model_data should be a 2D array of shape (n_conditions, n_neurons)'

    def run_pca(data):
        # center data
        data_centered = data - data.mean()

        # do PCA
        pca = PCA()
        pca.fit(data_centered)

        # get PVEs
        PVE = pca.explained_variance_ratio_
        return PVE

    n_colour_bins = constants.PARAMS['B']
    # fit a full PCA model (with n_PCs = n_datapoints) to the data for each plane / location, then average the
    # PVE by each PC across the two planes
    PVEs = np.stack((run_pca(model_data[:n_colour_bins, :]), (run_pca(model_data[n_colour_bins:, :])))).mean(0)
    csum_PVE = np.cumsum(PVEs)[:n_PCs]  # cumulative sum of PVE by first n_PCs

    return csum_PVE


def get_simple_AI(X, Y, max_dim):
    """
    Computes Alignment index (AI), see Elsayed et al. 2016

    @author: Dante Wasmuht

    Parameters
    ----------
    X : 2D array
        Data matrix of the format: (conditions,neurons) or (samples,features)
    Y : 2D array
        Analogous data array for the other experimental condition.
    max_dim : int
        Dimensionality for the calculated subspaces.

    Returns
    -------
    AI : scalar
        Subspace AI value between the max_dim-dimensional subspaces of datasets
        X and Y.

    """

    # de-mean data matrices
    X_preproc = (X - X.mean(0)[None, :]).T
    Y_preproc = (Y - Y.mean(0)[None, :]).T

    # get covariance matrices
    c_mat_x = np.cov(X_preproc)
    c_mat_y = np.cov(Y_preproc)

    # perform eigenvalue decomposition on covariance matrices
    eig_vals_x, eig_vecs_x = la.eigh(c_mat_x, eigvals=(c_mat_x.shape[0] - max_dim, c_mat_x.shape[0] - 1))
    eig_vals_y, eig_vecs_y = la.eigh(c_mat_y, eigvals=(c_mat_y.shape[0] - max_dim, c_mat_y.shape[0] - 1))

    # sort eigenvectors according to eigenvalues (descending order)
    eig_vecs_x = eig_vecs_x[:, np.argsort(np.real(eig_vals_x))[::-1]]
    eig_vals_x = eig_vals_x[::-1]

    eig_vecs_y = eig_vecs_y[:, np.argsort(np.real(eig_vals_y))[::-1]]
    eig_vals_y = eig_vals_y[::-1]

    # compute the alignment index: 1 = maximal subspaces overlap; 0 = subspaces are maximally orthogonal. In the
    # numerator, the data from Y is projected onto the X subspace and the variance of Y in X subspace is calculated.
    # The denominator contains the variance of Y in subspace Y (...not the full space! > Although this could be an
    # option too...)
    ai_Y_in_X = np.trace(np.dot(np.dot(eig_vecs_x.T, c_mat_y), eig_vecs_x)) / np.sum(eig_vals_y)
    ai_X_in_Y = np.trace(np.dot(np.dot(eig_vecs_y.T, c_mat_x), eig_vecs_y)) / np.sum(eig_vals_x)

    return (ai_Y_in_X + ai_X_in_Y) / 2


def get_AI_all_dims(constants, model_data, max_dim=3):
    """
    Calculate the AI values between a pair of neural subspaces, for a required range of dimensionalities (2, max_dim).

    Single-model level function.

    :param module constants: A Python module containing constants and configuration data for the simulation.
    :param np.ndarray model_data: data array with binned activation patterns for all neurons (n_colour_bins*n_locations,
        n_neurons)
    :param int max_dim: Optional. Maximum dimensionality for the calculated subspaces. Default is 3.
    :return: array of AI values for each dimensionality from the chosen range
    """
    assert isinstance(max_dim, int), 'max_dim must be an integer'

    n_colour_bins = constants.PARAMS['B']

    # calculate the AI for different dimensionalities
    ai = []

    for dim in range(2, max_dim + 1):
        # calculate the AI for the location 1 (first n_colour_bins rows) and location 2 (last n_colour_bins rows)
        ai.append(get_simple_AI(model_data[:n_colour_bins, :], model_data[n_colour_bins:, :], dim))

    ai = np.stack(ai)
    return ai


def get_unrotated_rotated_label(constants, model_preprocessed_data):
    """
    Get the unrotated and rotated plane labels, based on the AI values for each plane, calculated on the train dataset.

    :param module constants: A Python module containing constants and configuration data for the simulation.
    :param dict model_preprocessed_data: evaluation data for a single model saved in a dictionary. Train dataset saved
        under the 'train' key, where it is further split into a 'loc1' and 'loc2' datasets (saved under the namesake
        keys).
    :return:
    """
    # run the AI analysis for the train data
    loc1_ai = get_AI_all_dims(constants, model_preprocessed_data['train']['loc1'])
    loc2_ai = get_AI_all_dims(constants, model_preprocessed_data['train']['loc2'])

    n_dims = len(loc1_ai)
    # determine the labels
    all_ai = np.stack((loc1_ai, loc2_ai))
    unrotated_plane_ix = np.argmax(all_ai, axis=0)

    # create a dictionary mapping the 'rotated' and 'unrotated' labels to each location
    labels = [('loc1', 'loc2') if unrotated_plane_ix[dim] == 0 else ('loc2', 'loc1') for dim in range(n_dims)]
    labels_dict_list = [{'unrotated': labels[dim][0], 'rotated': labels[dim][1]} for dim in range(n_dims)]
    return labels_dict_list


# %% stats
def print_csum_PVE_stats(constants, csum_PVE, delay_name=None):
    """

    :param module constants: A Python module containing constants and configuration data for the simulation.
    :param np.ndarray csum_PVE: Cumulative sums of proportion variance explained by increasing numbers of PCs for the
        PCA model fit to the data. Shape: (n_PCs, n_models).
    :param str delay_name: Optional. Name of the delay interval / timepoint from which the evaluation data came.
    :return:
    """
    # check how much variance is captured by the first 2PCs across all models
    # calculate the mean and SEM of variance explained by the first 2 PCs
    assert len(csum_PVE.shape) == 2, 'Group-level function: csum_PVE should be 2D of a (n_PCs, n_models) shape'

    means_2PCs = csum_PVE[1, :].mean(-1) * 100
    sems_2PCs = np.std(csum_PVE[1, :], -1) / np.sqrt(constants.PARAMS['n_models']) * 100

    cond_str = f"in {delay_name}" if delay_name is not None else ''
    print(
        f'Total %% variance explained by the first 2PCs {cond_str}: mean = %.4f, SEM = %.4f' % (means_2PCs, sems_2PCs))

    return


def print_descriptive_stats(ai_table, comparison):
    """
    Print descriptive statistics for an AI table. If the table is 2D, print the statistics (M+SEM) for a given
    timepoint for all dimensionalities explored. If the table is 3D, for each dimensionality, print the statistics
    for each timepoint along with its label given by the 'comparison' argument.

    :param numpy.ndarray ai_table: Table with AI values, format: (n_dims, n_timepoints, n_models) or (n_dims, n_models).
    :param list comparison: Labels for different conditions the AI was calculated for, e.g., names of the delays.

    """
    if len(ai_table.shape) < 3:
        print(f' {comparison} AI estimates')
        # if you want to report the mean and SEM for some timepoint - pass a 2D table
        for dim in range(ai_table.shape[0]):
            sem = np.std(ai_table[dim, :]) / np.sqrt(ai_table.shape[-1])
            s = f'          AI {dim + 2}: mean = {ai_table[dim, :].mean():.2f}, ' + \
                f'sem  = {sem:.2f}'
            print(s)
    else:
        # if you want to compare values across timepoints
        for dim in range(ai_table.shape[0]):
            s = f'          AI {dim + 2}: '
            for comp in range(len(comparison)):
                s += f'mean {comparison[comp]} = {ai_table[dim, comp, :].mean():.2f}, '
            print(s)

    return


def print_inferential_stats(constants, ai_table, geometry_name):
    """
    Print descriptive statistics for an AI table corresponding to a particular geometry. If geometry is 'cued' or
    'uncued', tests if the post-cue AI values are significantly greater than the pre-cue AI values (by the means of a
    one-tailed paired samples t-test, or a non-parametric equivalent). If the geometry is 'unrot_rot' (corresponding to
    the unrotated/rotated plane analysis for the Cued geometry), tests if the unrotated plane AI is  significantly
    larger than that for the rotated plane. However, note that the statistics reported for this comparison in the
    publication were ran in JASP.

    :param module constants: A Python module containing constants and configuration data for the simulation.
    :param numpy.ndarray ai_table: Table with AI values, format: (n_dims, n_timepoints, n_models).
    :param str geometry_name: name of the geometry for which the comparison is to run. Choose from: 'cued', 'uncued'
        and 'unrot_rot'.
    :return:
    """
    assert len(ai_table.shape) == 3, 'AI_tbl must be a 3D array of shape (n_dimensions, n_timepoints, n_models)'
    if constants.PARAMS['experiment_number'] == 4:
        raise Warning('Analysis only compares the pre- and post-cue AI estimates. You might want to extend it to also '
                      'include the post-probe timepoint for Experiment 4.')
    assert geometry_name in ['cued', 'uncued', 'unrot_rot'], \
        "Incorrect geometry name, choose from : 'cued', 'uncued', 'unrot_rot'"

    for dim in range(ai_table.shape[0]):
        if geometry_name in ['cued', 'uncued']:
            # test if post-cue AI is greater than pre-cue AI
            print(f'Testing post-cue AI > pre-cue AI {dim + 2}')
            stats.run_contrast_paired_samples(ai_table[dim, 1, :], ai_table[dim, 0, :], alt='greater')
        elif geometry_name == 'unrot_rot':
            # test if post-cue AI is greater than pre-cue AI
            print(f'Testing unrotated plane AI > rotated plane AI {dim + 2}')
            stats.run_contrast_paired_samples(ai_table[dim, 1, :], ai_table[dim, 0, :], alt='greater')
            print('Note - statistics reported in the publication were ran in JASP.')

    return


# %% define looper functions that will loop across models and delay intervals
def model_looper(constants, all_data, geometry_name, func, delay_name=None):
    """
    Calculates the output of single model-level function 'func' for a specific geometry and memory delay for all models.
    Possible geometries include: 'cued', 'uncued', 'cued_up_uncued_down' and 'cued_down_uncued_up'. Returns the output
    table in the following format: (n_dimensionalities, n_models)

    :param module constants: A Python module containing constants and configuration data for the simulation.
    :param dict all_data: Data dictionary containing averaged and binned 'pca_data' arrays for all models and possible
        geometries. See the output of the 'get_all_binned_data' function for more details.
    :param str geometry_name: Desired geometry. Choose from: 'cued', 'uncued', 'cued_up_uncued_down' and
        'cued_down_uncued_up'
    :param function func: Single model-level function to be applied to the data from individual models.
    :param str delay_name: Desired delay interval. Choose from: 'delay1', 'delay2' and 'delay3' (only for Experiment 4).
    :return: output table, (n_dimensionalities, n_delays, n_models)

    .. note:: This function mirrors the model_geometry_looper function from rep_geom_analysis. Both could probably be
    rewritten as decorators.
    """
    assert geometry_name in ['cued', 'uncued', 'cued_up_uncued_down', 'cued_down_uncued_up'], \
        "Incorrect geometry name, choose from : 'cued', 'uncued', 'cued_up_uncued_down' and 'cued_down_uncued_up'"

    output_table = []
    for model in range(constants.PARAMS['n_models']):
        # get the model data
        if delay_name is None:
            model_data = all_data[model][geometry_name]
        else:
            model_data = all_data[model][geometry_name][delay_name]

        # calculate the output for the current model
        output = func(constants, model_data)
        output_table.append(output)

    output_table = np.stack(output_table).T  # (n_dimensionalities, n_models)

    return output_table


def delay_looper(constants, all_data, geometry_name, single_model_func, group_func=None):
    """
    Calculates the output of single model-level function 'single_model_func' for a specific geometry for all memory
    delays and models. If required, can also apply the function 'group_func' to the 'func' output aggregated across all
    models, separately for each delay interval. Possible geometries include: 'cued', 'uncued', 'cued_up_uncued_down' and
    'cued_down_uncued_up'. Returns the output table in the following format: (n_dimensionalities, n_delays, n_models)

    :param module constants: A Python module containing constants and configuration data for the simulation.
    :param dict all_data: Data dictionary containing averaged and binned 'pca_data' arrays for all models and possible
        geometries. See the output of the 'get_all_binned_data' function for more details.
    :param str geometry_name: Desired geometry. Choose from: 'cued', 'uncued', 'cued_up_uncued_down' and
        'cued_down_uncued_up'
    :param function single_model_func: Single model-level function to be applied to the data from individual models.
    :param function group_func: Optional. Group-level function to be applied to the 'single_mode_func' output aggregated
        across all models, separately for each delay interval. Default is None.
    :return:  output table, (n_dimensionalities, n_delays, n_models)

    .. note:: This function mirrors the delay_geometry_looper function from rep_geom_analysis. Both could probably be
        rewritten as decorators.

    """
    assert geometry_name in ['cued', 'uncued', 'cued_up_uncued_down', 'cued_down_uncued_up'], \
        "Incorrect geometry name, choose from : 'cued', 'uncued', 'cued_up_uncued_down' and 'cued_down_uncued_up'"

    output_table = []
    for delay in range(constants.PARAMS['n_delays']):
        output = model_looper(constants, all_data, geometry_name, single_model_func, delay_name=f"delay{delay + 1}")
        output_table.append(output)

        if group_func is not None:
            group_func(constants, output, delay_name=f"delay{delay + 1}")
    output_table = np.stack(output_table).transpose([1, 0, 2])  # (n_dimensionalities, n_delays, n_models)
    return output_table


def experiment_2_looper(constants):
    """
    Loop through all versions of Experiment 2 (defined by the length of the post-cue delay interval). Get the data and
    calculate the Cued geometry in a single loop. Returns the theta angle estimates and PC variance explained values for
    the fitted 3D subspaces.

    :param module constants: A Python module containing constants and configuration data for the simulation.
    :return: all_theta (n_models, n_delays, n_delay2_lengths), all_PVEs (n_models, n_delays, n_PCs, n_delay2_lengths)
    """
    assert constants.PARAMS['experiment_number'] == 2, \
        'This function should only be used for Experiment 2 (retrocue timing).'
    delay2_max_length = \
        (constants.PARAMS['trial_timings']['delay1_dur'] + constants.PARAMS['trial_timings']['delay2_dur']) // 2

    # loop through the different experiment variants, load their respective constants modules and collect Cued data
    ai_table = []
    for delay2_length in range(delay2_max_length + 1):
        module_name = f"constants.constants_expt2_delay2_{delay2_length}cycles"
        c = importlib.import_module(module_name)

        # get the data - make sure that all data from all variants of the experiment is saved to file.
        try:
            _, all_data = ppc.get_all_binned_data(c, trial_type='valid')
        except FileNotFoundError:
            print(
                f"Data from post-cue delay length {delay2_length} cycles not found. Make sure models from all variants"
                f" of Experiment 2 have been evaluated and data saved.")
            return

        # get the cued AI
        ai_table.append(delay_looper(c, all_data, 'cued', get_AI_all_dims))

    ai_table = np.stack(ai_table).transpose([3, 1, 2, 0])  # (n_models, n_dimensions, n_timepoints, n_delay2_lengths)

    return delay2_max_length, ai_table


# %% rotated / unrotated geometry - special case, as it is done in cross-validation

def get_AI_unrotated_rotated(constants, max_dim=3):
    """
    Calculate the AI values between a pair of neural subspaces from different timepoints, for a required range of
    dimensionalities (2, max_dim). Done in cross-validation.

    Single-model level function.

    :param module constants: A Python module containing constants and configuration data for the simulation.
    :param int max_dim: Optional. Maximum dimensionality for the calculated subspaces. Default is 3.
    :return: array of AI values for each dimensionality from the chosen range
    """
    assert isinstance(max_dim, int), 'max_dim must be an integer'
    if constants.PARAMS['experiment_number'] not in [1, 3]:
        raise NotImplementedError('Unrotated / Rotated plane analysis only implemented for Experiments 1 and 3')

    # get the data
    preprocessed_data = ppc.get_unrotated_rotated_data(constants, get_test_train_split=False)

    n_cv_folds = preprocessed_data.shape[1]
    plane_label_keys = ['unrotated', 'rotated']
    n_dims = max_dim - 1  # total number of dimensionalities for which AI is calculated
    # calculate the AI for different dimensionalities
    ai_dict = {key: np.empty((n_dims, constants.PARAMS['n_models'], n_cv_folds)) for key in plane_label_keys}

    # loop over cross-validation folds
    for cv in range(n_cv_folds):
        data = preprocessed_data[:, cv]

        test_data = {dim: {} for dim in range(n_dims)}
        # loop over models
        for model in range(constants.PARAMS['n_models']):
            # get the unrotated and rotated location labels
            labels_dict_list = get_unrotated_rotated_label(constants, data[model])

            # loop over AI dimensionalities
            for dim in range(n_dims):
                # construct a new test dictionary with these labels (aka relabel locations)
                test_data[dim][model] = ppc.relabel_test_data(data[model]['test'], labels_dict_list[dim])

                # calculate the rotated and unrotated subspace AI
                for plane_label in plane_label_keys:
                    ai_dict[plane_label][dim, model, cv] = \
                        get_AI_all_dims(constants, test_data[dim][model][plane_label])[dim]

    return ai_dict


def average_unrotated_rotated_AI(ai_dict):
    """
    Averages the unrotated and rotated AI estimates across cross-validation folds.

    :param dict ai_dict: dictionary containing the AI table for the unrotated and rotated plane, saved under their
        namesake keys.
    :return: ai_averaged array of shape (n_dimensions, n_plane_types, n_models), where the second dimension contains the
        unrotated and rotated plane estimates, in this order
    """
    # average tha unrotated/rotated AI values across the CV folds
    ai_averaged = []
    for plane_label in ai_dict.keys():
        data = ai_dict[plane_label]
        ai_averaged.append(data.mean(-1))

    ai_averaged = np.stack(ai_averaged).transpose([1, 0, 2])  # (dim, unrotated/rotated plane, model)
    return ai_averaged


# %% io
def save_unrot_rot_data(constants, ai_dict, ai_averaged, trial_type='valid'):
    """
    Save the unrotated and rotated AI analysis data into file.

    :param module constants: A Python module containing constants and configuration data for the simulation.
    :param dict ai_dict: dictionary containing the AI table for the unrotated and rotated plane, saved under their
        namesake keys.
    :param np.ndarray ai_averaged: array of shape (n_dimensions, n_plane_types, n_models), where the second dimension
        contains the unrotated and rotated plane estimates, in this order
    :param str trial_type: Optional. Relevant for the probabilistic paradigm (experiment 4). Pass 'valid' or 'invalid'.
        The default is 'valid'.
    :return:
    """
    data_path = f"{constants.PARAMS['RESULTS_PATH']}{trial_type}_trials/"

    # save the AI dict and table
    with open(f"{data_path}/AI_dict_unrot_rot_analysis.pckl", 'wb') as f:
        pickle.dump(ai_dict, f)
    with open(f"{data_path}/AI_tbl_unrotrot.pckl", 'wb') as f:
        # AI values averaged across cv folds
        pickle.dump(ai_averaged, f)

    # save to csv
    df = pd.DataFrame(ai_averaged.reshape(-1, 30).T,
                      columns=['unrotated_2D', 'rotated_2D', 'unrotated_3D', 'rotated_3D'])
    df.to_csv(f"{data_path}AI_tbl_unrotrot.csv")

    return


def export_big_AI_tbl(constants, big_AI_list, trial_type='valid'):
    """
    Collect all AI estimates into a pandas dataframe and export to CSV for a JASP analysis.

    :param module constants: A Python module containing constants and configuration data for the simulation.
    :param list big_AI_list:
    :param str trial_type: Optional. Relevant for the probabilistic paradigm (experiment 4). Pass 'valid' or 'invalid'.
        The default is 'valid'.
    :return:
    """
    print('          export data to file')

    # reshape data into columns
    big_AI_tbl = np.concatenate(big_AI_list, axis=1)
    big_AI_tbl = big_AI_tbl.transpose([-1, 1, 0])
    big_AI_tbl = big_AI_tbl.reshape((big_AI_tbl.shape[0],
                                     big_AI_tbl.shape[1] * big_AI_tbl.shape[-1]),
                                    order='F')

    cols = [['cued_prevspre', 'cued_postvspost', 'cued_rot', 'cued_unrot',
             'uncued_prevspre', 'uncued_postvspost',
             'cuedvsuncued+pair1', 'cuedvsuncued_pair2']] * 2
    cols = [i + j for a, b in zip(cols, [['_2D'] * 10, ['_3D'] * 10]) for i, j in zip(a, b)]

    big_AI_tbl = pd.DataFrame(big_AI_tbl, columns=cols)

    path = f"{constants.PARAMS['RESULTS_PATH']}{trial_type}_trials/big_AI_tbl.csv"
    big_AI_tbl.to_csv(path)

# %% runners


def run_AI_analysis(constants, all_data, geometry_name, trial_type='valid'):
    """
    Run a full AI analysis for a given geometry. Steps include:
    1. calculating the intrinsic dimensionality of the data - to check that fitting 2D subspaces captures the majority
        of the variance in the data
    2. calculating the AI metric between a pair of subspaces, for all models and delay intervals (save in an AI table)
    3. printing of descriptive and inferential statistics for the AI table
    3. plotting the data

    If the geometry name is 'cued_up_uncued_down' or vice versa, the function will only evaluate the geometry for the
    given trial type. However, if the passed geometry name is 'cued_uncued', the function will loop through both trial
    types and average the AI estimates across them, prior to steps 3 and 4.

    Additionally, if the geometry name is 'cued', the function will additionally perform the 'unrotated/rotated' plane
    analysis (only for Experiments 1 and 3).

    :param module constants: A Python module containing constants and configuration data for the simulation.
    :param dict all_data: Data dictionary containing averaged and binned 'pca_data' arrays for all models and possible
        geometries. See the output of the 'get_all_binned_data' function for more details.
    :param str geometry_name: Desired geometry. Choose from: 'cued', 'uncued', 'cued_up_uncued_down' and
        'cued_down_uncued_up'
    :param str trial_type: Optional. Relevant for the probabilistic paradigm (experiment 4). Pass 'valid' or 'invalid'.
        The default is 'valid'.
    :return:
    """
    assert geometry_name in ['cued', 'uncued', 'cued_up_uncued_down', 'cued_down_uncued_up', 'cued_uncued'], \
        "Invalid geometry_name, choose from: 'cued', 'uncued', 'cued_up_uncued_down', 'cued_down_uncued_up', and " \
        "'cued_uncued'"

    print(f'RUNNING THE AI ANALYSIS, {geometry_name} GEOMETRY')

    if geometry_name == 'cued_uncued':
        # run the analysis for both trial types ('cued_up_uncued_down' and 'cued_down_uncued_up') and average results
        # across them
        ai_table = []
        for g_name in ['cued_up_uncued_down', 'cued_down_uncued_up']:
            print(f'Running {g_name} trials')
            # calculate the intrinsic dimensionality of the data - to check that fitting 2D subspaces captures the
            # majority of the  variance in the data
            _ = delay_looper(constants, all_data, g_name, get_intrinsic_dim, print_csum_PVE_stats)

            # calculate the AI
            ai_table.append(delay_looper(constants, all_data, g_name, get_AI_all_dims))
        # average across trial types
        ai_table = np.stack(ai_table).mean(0)
        # extract the post-cue (or post-probe, for Experiment 4) timepoint only
        timepoint = -1
        ai_table = ai_table[:, timepoint, :][:, None, :]  # keep 3D for compatibility with the plotter function
        print_descriptive_stats(ai_table.squeeze(), comparison=[geometry_name])

    else:
        # calculate the intrinsic dimensionality of the data - to check that fitting 2D subspaces captures the
        # majority of the variance in the data
        _ = delay_looper(constants, all_data, geometry_name, get_intrinsic_dim, print_csum_PVE_stats)

        # calculate the AI
        ai_table = delay_looper(constants, all_data, geometry_name, get_AI_all_dims)
        # shape: (n_dimensions, n_timepoints, n_models)

        delay_names = ['pre-cue', 'post-cue', 'post_probe'][:constants.PARAMS['n_delays']]
        print_descriptive_stats(ai_table, comparison=delay_names)

    # save the AI table
    data_path = f"{constants.PARAMS['RESULTS_PATH']}{trial_type}_trials/"
    with open(f"{data_path}/AI_tbl_{geometry_name}.pckl", 'wb') as f:
        pickle.dump(ai_table, f)

    # run statistics
    if geometry_name in ['cued', 'uncued'] and constants.PARAMS['experiment_number'] in [1, 3]:
        print_inferential_stats(constants, ai_table, geometry_name)

    # plot data
    if geometry_name in ['cued', 'uncued']:
        # plot all timepoints
        plotter.plot_AI(constants, ai_table, geometry_name)
    elif geometry_name in ['cued_up_uncued_down', 'cued_down_uncued_up']:
        # plot the post-cue (or post-probe) timepoint only
        timepoint = -1
        plotter.plot_AI(constants, ai_table[:, timepoint, :][:, None, :], geometry_name)
    else:
        # plot only post-cue
        plotter.plot_AI(constants, ai_table, geometry_name)

    if constants.PLOT_PARAMS['save_plots']:
        plt.savefig(f"{constants.PARAMS['FIG_PATH']}AI_{geometry_name}.png")
        plt.savefig(f"{constants.PARAMS['FIG_PATH']}AI_{geometry_name}.svg")

    if geometry_name == 'cued' and constants.PARAMS['experiment_number'] in [1, 3]:
        print(f'AI analysis, {geometry_name} geometry, rotated / unrotated plane comparison')
        # also run the rotated-unrotated plane analysis
        # get the unrotated/rotated plane AI estimates in cross-validation
        ai_dict = get_AI_unrotated_rotated(constants, max_dim=3)
        ai_averaged = average_unrotated_rotated_AI(ai_dict)  # average across the cv folds,
        # shape: (dim, unrotated/rotated plane, model)

        # save to file
        save_unrot_rot_data(constants, ai_dict, ai_averaged, trial_type)

        # get statistics
        print_descriptive_stats(ai_averaged, comparison=['unrotated', 'rotated'])

        print('Note the statistics for this analysis were ran in JASP')
        # note the statistics reported in the paper were ran in JASP, as the scipy.stats implementation of the Wilcoxon
        # test returns a W-statistic = 0 but a highly significant p-value
        # print_inferential_stats(constants, ai_averaged, 'unrot_rot')

        # plot
        plotter.plot_AI(constants, ai_averaged, 'unrotated_rotated')

        # save plots
        if constants.PLOT_PARAMS['save_plots']:
            plt.savefig(f"{constants.PARAMS['FIG_PATH']}AI_unrotated_rotated_cued_plane.png")
            plt.savefig(f"{constants.PARAMS['FIG_PATH']}AI_unrotated_rotated_cued_plane.svg")
    return


def run_AI_analysis_experiment_2(constants):
    """
    Run the Cued AI analysis for Experiment 2 (retrocue timing). Plots the pre- and post-cue AI against the post-cue
    delay length.

    :param module constants: A Python module containing constants and configuration data for the simulation.
    :return:
    """
    print('EXPERIMENT 2 - CUED AI ANALYSIS')
    assert constants.PARAMS['experiment_number'] == 2, \
        'This function should only be used for Experiment 2 (retrocue timing)'

    delay2_max_length, ai_table = experiment_2_looper(constants)
    # ai_table shape: (n_models, n_dimensions, n_timepoints, n_delay2_lengths)
    # plot the angle comparison - for 2D subspaces
    dim_ix = 0  # index of the 2D AI estimates
    plotter.plot_geometry_estimates_experiment_2(constants, delay2_max_length, ai_table[:, dim_ix, :, :])
    # save plot
    if constants.PLOT_PARAMS['save_plots']:
        plt.savefig(f"{constants.PARAMS['EXPT2_PATH']}'compare_cued_AI.svg")
        plt.savefig(f"{constants.PARAMS['EXPT2_PATH']}'compare_cued_AI.png")
    return


def run_ai_all_geometries(constants, all_data, trial_type='valid'):
    data_path = f"{constants.PARAMS['RESULTS_PATH']}{trial_type}_trials/"

    big_AI_tbl = []
    for geometry_name in ['cued', 'uncued', 'cued_up_uncued_down', 'cued_down_uncued_up']:
        # run the analysis
        run_AI_analysis(constants, all_data, geometry_name)

        # load data from file
        with open(f"{data_path}AI_tbl_{geometry_name}.pckl", 'rb') as f:
            if geometry_name in ['cued_up_uncued_down', 'cued_down_uncued_up']:
                # add only the post-cue (or post-probe) timepoint
                timepoint = -1
                ai_tbl = pickle.load(f)
                big_AI_tbl.append(ai_tbl[:, timepoint, :][:, None, :])
            else:
                # append the whole table
                big_AI_tbl.append(pickle.load(f))

        if geometry_name == 'cued':
            # also load the unrotated/rotated estimates
            with open(f"{data_path}AI_tbl_unrotrot.pckl", 'rb') as f:
                big_AI_tbl.append(pickle.load(f))

    # reformat into a pandas dataframe and save as csv
    export_big_AI_tbl(constants, big_AI_tbl, trial_type='valid')

    return
