#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 17:58:34 2021

@author: emilia

This file contains all decoding analysis functions, including:
    1) decoding of uncued stimulus colour in the post-cue delay
    2) cross-temporal decoding
    3) comparison of  maintenance mechanisms between models from Experiments 1 & 2
    4) single-readout hypothesis : cross-decoding of cued colours across two parallel planes
    5) analogue to the CDI analysis: compare the discriminability of colour 
        representations pre-cue, after they are cued and uncued

1) This analysis asks if there is still information about the uncued item 
    colour in the post-cue delay.
2) This analysis computes the cross-temporal decoding accuracy scores for the
    cued items across the entire trial duration.
3) This analysis calculates the mean delay cross-temporal generalisation scores and 
    compares those between Experiments 1 & 2 to assess whether the networks trained
    with variable delay lengths (expt 2) form a more temporally stable working 
    memory code than those trained with fixed delays (expt 1).
4) This analysis asks if a common linear readout can be used to extract the cued
    colour information across both the cue-1 and cue-2 trials.
5) This analysis seeks to confirm the conclusion from Fig. 3H whilst taking 
    into account the noise in the representations. Linear decoders are trained
    in cross-validation to discriminate between colours in the pre-cue delay, 
    as well as after they are cued or uncued, and the test scores compared 
    between the 3 conditions.
"""
import pickle
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LinDiscrAn
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from mne.decoding import GeneralizingEstimator

import src.helpers as helpers
import src.generate_data_von_mises as dg
import src.plotting_funcs as plotter
from src.stats import run_contrast_single_sample, run_contrast_unpaired_samples

# %% common low-level functions


def get_class_pairs(y):
    """
    Gets all possible class (i.e., label) pairwise combinations.

    :param numpy.ndarray y: Vector of trial-wise class labels (n_trials, ).
    :return combos: list of all possible binary class combinations.
    """
    classes = np.unique(y)
    combos = list(combinations(classes, 2))
    return combos


def lda(X, y, cv=2):
    """
    Fits binary LDA classifiers to discriminate between labels in cross-validation.

    :param np.ndarray X: Data matrix, shape: (n_samples,n_features)
    :param np.ndarray y: Trial-wise class labels, shape: (n_samples,)
    :param int cv: Optional, number of cross-validation folds. The default is 2.
    :return : scores_test : Array with test decoding accuracy for each LDA classifier, shape: (n_classifiers,)
    """
    # get all possible class pairs    
    class_combos = get_class_pairs(y)
    scores_test = np.zeros((len(class_combos),))
    for i in range(len(class_combos)):
        y1, y2 = class_combos[i]  # class labels

        # find trials of the above-specified classes
        trial_ix = np.where(np.logical_or(y == y1, y == y2))[0]

        # make classifier pipeline
        clf = make_pipeline(StandardScaler(), LinDiscrAn())

        # fit a classifier in cross-val
        results = cross_validate(clf, X[trial_ix, :], y[trial_ix],
                                 cv=cv,
                                 return_estimator=False)

        # average test scores across cv folds
        scores_test[i] = results['test_score'].mean()

    return scores_test


def lda_cg(X1, y1, X2, y2, cv=2):
    """
    Fits binary LDA classifiers to discriminate between labels to one dataset and tests performance on 1) a held-out
    portion of the same dataset and 2) another dataset (cross-generalisation performance).

    :param np.ndarray X1: Data matrix for dataset 1, (n_samples,n_features).
    :param np.ndarray y1: Trial-wise class labels for dataset 1, (n_samples,).
    :param np.ndarray X2: Data matrix for dataset 2, (n_samples,n_features).
    :param np.ndarray y2: Trial-wise class labels for dataset 2, (n_samples,).
    :param int cv: Optional, number of cross-validation folds. The default is 2.

    :return:
        scores_test: Test decoding accuracy for each LDA classifier on the withheld part of the training dataset, shape:
            (n_classifiers, ).
        scores_cg: Test decoding accuracy for each LDA classifier on the generalisation dataset, (n_classifiers, ).

    """
    # get all possible class pair combinations    
    class_combos = get_class_pairs(y1)
    scores_test = np.zeros((len(class_combos), 2))  # (n_classifiers,n_datasets)
    scores_cg = np.zeros((len(class_combos), 2))
    for i in range(len(class_combos)):
        l1, l2 = class_combos[i]  # pair of class labels

        # find trials of the above-specified classes
        trial_ix_y1 = np.where(np.logical_or(y1 == l1, y1 == l2))[0]
        trial_ix_y2 = np.where(np.logical_or(y2 == l1, y2 == l2))[0]

        # make classifier pipelines
        clf1 = make_pipeline(StandardScaler(), LinDiscrAn())
        clf2 = make_pipeline(StandardScaler(), LinDiscrAn())

        # fit classifiers in cross-val
        results1 = cross_validate(clf1,
                                  X1[trial_ix_y1, :],
                                  y1[trial_ix_y1],
                                  cv=cv,
                                  return_estimator=True)
        results2 = cross_validate(clf2,
                                  X2[trial_ix_y2, :],
                                  y2[trial_ix_y2],
                                  cv=cv,
                                  return_estimator=True)

        # average test scores across cv folds
        scores_test[i, 0] = results1['test_score'].mean()
        scores_test[i, 1] = results2['test_score'].mean()

        # calculate cross-generalisation performance
        scores_cg[i, 0] = np.mean(
            [results1['estimator'][i].score(X2[trial_ix_y2, :], y2[trial_ix_y2]) for i in range(cv)])
        scores_cg[i, 1] = np.mean(
            [results2['estimator'][i].score(X1[trial_ix_y1, :], y1[trial_ix_y1]) for i in range(cv)])

    return scores_test.mean(-1), scores_cg.mean(-1)


def lda_cg_time(X, y):
    """
    Test LDA classifiers to discriminate between pairs of classes based on data from a single timepoint and test their
    performance on all the other timepoints.

    :param np.ndarray X: Data matrix, shape: (n_samples,n_features)
    :param np.ndarray y: Trial-wise class labels, shape: (n_samples,)
    :return: test_scores: Cross-temporal generalisation scores array (n_timepoints,n_timepoints)
    :rtype: np.ndarray
    """
    # split data and labels into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)
    time = X.shape[-1]
    # get all possible class1-class2 combinations for binary discriminations
    class_combos = get_class_pairs(y)
    clfs = {}  # classifiers
    scores_Xtest = np.empty((len(class_combos), time, time))
    # loop over class pairs
    for i in range(len(class_combos)):
        y1, y2 = class_combos[i]  # class labels

        # find trials of above-specified classes
        train_ix = np.where(np.logical_or(y_train == y1, y_train == y2))[0]
        test_ix = np.where(np.logical_or(y_test == y1, y_test == y2))[0]

        # make classifier pipeline
        clf = make_pipeline(StandardScaler(), LinDiscrAn())
        time_gen = GeneralizingEstimator(clf)

        # fit the classifier
        clfs[str(i)] = time_gen.fit(X=X_train[train_ix, :, :], y=y_train[train_ix])

        # test performance on with-held data, all timepoints
        scores_Xtest[i, :, :] = time_gen.score(X=X_test[test_ix, :, :], y=y_test[test_ix])

    return scores_Xtest.mean(0)


def load_model_data(constants, model_number, trial_type='valid', binned=False):
    """
    Load the evaluation data file for a given model.

    :param module constants: A Python module containing constants and configuration data for the simulation.
    :param int model_number: Number of the model for which to load the dataset.
    :param str trial_type: Optional. Relevant for the probabilistic paradigm (experiment 4). Pass 'valid' or 'invalid'.
        The default is 'valid'.
    :param bool binned: Optional. If True, load data binned into constants.PARAMS['B'] colour bins (i.e., the 'pca_data'
        structure). Default is False.
    :return:
    """
    # load eval data for a single model
    load_path = f"{constants.PARAMS['FULL_PATH']}pca_data/{trial_type}_trials/"

    if binned:
        # load pca data
        file_path = f"{load_path}/pca_data_model{model_number}.pckl"
    else:
        # load evaluation data
        file_path = f"{load_path}/eval_data_model{model_number}.pckl"

    with open(file_path, 'rb') as f:
        model_data = pickle.load(f)

    return model_data


def get_delay_index(constants, delay_name):
    """
    Get the index of the datapoint corresponding to a given delay name.

    :param module constants: A Python module containing constants and configuration data for the simulation.
    :param str delay_name: Name of the required delay. Choose from: 'precue', 'postcue' and 'postprobe'
    :return: delay_ix: index of the endpoint of the delay
    """
    if delay_name == 'precue':
        delay_ix = constants.PARAMS['trial_timepoints']['delay1_end'] - 1
    elif delay_name == 'postcue':
        # post-cue delay (uncued colours)
        delay_ix = constants.PARAMS['trial_timepoints']['delay2_end'] - 1
    elif delay_name == 'postprobe':
        delay_ix = constants.PARAMS['trial_timepoints']['delay3_end'] - 1
    else:
        raise ValueError("Invalid delay name. Choose from: 'precue', 'postcue' and 'postprobe'")
    return delay_ix


def extract_delay_data(delay_ix, eval_data):
    """
    Extract the data corresponding to the required timepoint(s).

    :param int or list or np.ndarray delay_ix: index (indices) of the required timepoints
    :param dict eval_data: Data dictionary. Must contain a key 'data' with a data array of shape (m, n_timepoints, n)
    :return: delay_data: Data array containing the subset of the data corresponding to the required trial timepoint(s),
        shape: (m, len(delay_ix), n)
    """
    delay_data = eval_data['data'][:, delay_ix, :]
    return delay_data


def get_colour_labels(constants, eval_data, item_status='cued', trial_type='valid'):
    """
    Extract colour labels for a given dataset. Labels are binned into constants.PARAMS['B'] colour bins.

    :param module constants: A Python module containing constants and configuration data for the simulation.
    :param dict eval_data: Data dictionary. Must contain 'data' and 'labels' keys. The data array under 'data' should
        be of the following shape: (n_trials, n_timepoints, n_neurons). The 'labels' key should contain a sub-dictionary
        with 'c1' and 'c2' keys, containing arrays with the colour 1 and colour 2 values for each trial, respectively.
    :param str item_status: Which item to run the decoding analysis for, choose from 'cued', 'uncued', 'probed', and
        'unprobed'.
    :param str trial_type: Optional. Relevant for the probabilistic paradigm (experiment 4). Pass 'valid' or 'invalid'.
        The default is 'valid'.
    :return: labels_binned: array of trial-wise colour labels, binned into constants.PARAMS['B'] colour bins.

    .. note :: Currently only implemented for 'valid' trials, passing trial_type='invalid' will produce an error.
    """
    assert item_status in ['cued', 'uncued', 'probed', 'unprobed'], \
        "Incorrect item status. Choose from 'uncued', 'cued', 'probed', and 'unprobed'."
    assert trial_type == 'valid', 'Incorrect trial type. Analysis only implemented for valid trials.'
    # get the labels
    n_trials = eval_data['data'].shape[0]
    if item_status in ['uncued', 'unprobed']:
        labels = np.concatenate((eval_data["labels"]["c2"][:n_trials // 2],
                                 eval_data["labels"]["c1"][n_trials // 2:]))
    else:
        # 'probed', 'cued'
        labels = np.concatenate((eval_data['labels']['c1'][:n_trials // 2],
                                 eval_data['labels']['c2'][n_trials // 2:]), 0)

    # Note - if you want to extend the functionality to invalid trials, extend the if statement above to take
    # conjunctions of trial_type and item_status to determine the labels.

    # bin the labels into B colour bins
    labels_binned = helpers.bin_labels(labels, constants.PARAMS['B'])

    return labels_binned


def split_into_location_data(constants, labels, delay_data):
    """
    Split the data and labels arrays into subsets, according to the location of the cued item.

    :param module constants: A Python module containing constants and configuration data for the simulation.
    :param labels: Labels array, shape (n_trials, )
    :param np.ndarray delay_data: Data array containing the subset of the data corresponding to the required trial
        timepoint(s), shape: (n_trials, len(delay_ix), n_neurons) or (n_trials, n_neurons)
    :return: labels_split, data_split: arrays with the data split into location subsets, shape:
        (n_locations, n_trials_per_location) and (n_locations, n_trials_per_location, n_neurons)
    """
    # split the labels according to the cued location
    n_trials = labels.shape[0]
    if constants.PARAMS['L'] is not 2:
        raise ValueError('The split_ix below will not work for n_locations other than 2.')

    split_ix = n_trials // constants.PARAMS['L']
    ixs = [np.arange(split_ix), np.arange(split_ix, n_trials)]

    # Note - this could be rewritten by reshaping the original arrays
    labels_split, data_split = [], []
    for ix in ixs:
        labels_split.append(labels[ix])
        data_split.append(delay_data[ix, :])

    labels_split = np.stack(labels_split)
    data_split = np.stack(data_split)
    return labels_split, data_split


def shuffle_trials(model_number, labels, data):
    """
    Shuffle the trials in data and labels arrays.

    :param int model_number: Number of the model, used as seed to the random number generator for reproducibility.
    :param numpy.ndarray labels: Labels array, shape: (n_locations, n_trials_per_location)
    :param numpy.ndarray data: Data array, shape: (n_locations, n_trials_per_location, n_neurons)
    :return: labels_shuffled, data_shuffled: lists of length n_locations with the shuffled data and labels arrays
    """
    n_trials_per_loc = labels.shape[1]
    # shuffle trials for the decoder
    rng = np.random.default_rng(seed=model_number)
    trial_order = [rng.permutation(n_trials_per_loc), rng.permutation(n_trials_per_loc)]
    labels_shuffled, data_shuffled = [], []
    for i, shuffled_ixs in enumerate(trial_order):
        labels_shuffled.append(labels[i, shuffled_ixs])
        data_shuffled.append(data[i, shuffled_ixs, :])

    return labels_shuffled, data_shuffled


def model_looper(constants, delay_name, pipeline_func, item_status, trial_type='valid', **kwargs):
    """
    Run the decoding pipeline for all models and collect the test scores into an array.

    :param module constants: A Python module containing constants and configuration data for the simulation.
    :param str or int or list delay_name: Desired delay interval. Depending on the pipeline function being used, either
        pass a delay name (precue', 'postcue' or 'postprobe'), or a delay timepoint index / list of indices.
    :param func pipeline_func: A function that implements a whole decoding analysis for a single model. The steps are:
        get the data, split into locations, shuffle trials and fit/test the decoders.
    :param str item_status: Which item to run the decoding analysis for, choose from 'cued', 'uncued', 'probed', and
        'unprobed'.
    :param str trial_type: Type of trials for which to run the decoder. Choose from 'valid' and 'invalid'
    :param kwargs: any additional parameters to pass to the pipeline_func
    :return:
    """
    model_scores = []

    for model_number in range(constants.PARAMS['n_models']):
        print(f"Model {model_number}/{constants.PARAMS['n_models']}")
        model_scores.append(pipeline_func(constants, model_number, delay_name, item_status, trial_type, **kwargs))

    model_scores = np.stack(model_scores)
    return model_scores


def get_decoding_within_plane(constants, data_shuffled, labels_shuffled):
    """
    Train and test LDA binary classifiers to discriminate between pairs of  colour labels from a single item location
    and timepoint.

    :param module constants: A Python module containing constants and configuration data for the simulation.
    :param list data_shuffled: List containing the location-specific datasets for fitting and testing the decoders.
    :param list labels_shuffled: List containing the location-specific labels for fitting and testing the decoders.
    :return: model_scores : Array of average test decoding accuracy scores for all models (n_models, )
    """

    n_locs = constants.PARAMS['L']
    scores = []
    for loc in range(n_locs):
        # train and test LDA classifiers
        scores.append(lda(data_shuffled[loc], labels_shuffled[loc]))
    # save LDA model test scores
    model_scores = np.stack(scores).mean()
    # save to file
    return model_scores


def get_decoding_across_planes(constants, data_shuffled, labels_shuffled):
    """
    Train LDA binary classifiers to discriminate between pairs of colour labels from a single item location (and
    timepoint), and test the performance on the corresponding data from the other item location.

    :param module constants: A Python module containing constants and configuration data for the simulation.
    :param list data_shuffled: List containing the location-specific datasets for fitting and testing the decoders.
    :param list labels_shuffled: List containing the location-specific labels for fitting and testing the decoders.
    :return : model_scores - average test and generalisation decoding accuracies
    """

    n_locs = constants.PARAMS['L']
    model_scores = []
    # loop through all possible combinations of the locations (for two locations, there will be just one possibility)
    for (loc1, loc2) in itertools.combinations(range(n_locs), 2):
        # get the data and labels for the first location from the pair
        X1, y1 = data_shuffled[loc1], labels_shuffled[loc1]
        # repeat for the second pair
        X2, y2 = data_shuffled[loc2], labels_shuffled[loc2]

        # train LDA classifiers on one of the locations and test performance on the other, and vice versa
        scores_test, scores_cg = lda_cg(X1, y1, X2, y2, cv=2)  # shape = (n_classifiers, )
        model_scores.append([scores_test.mean(), scores_cg.mean()])  # average across all fitted binary classifiers
    model_scores = np.stack(model_scores).mean(0)  # average across location combinations to produce a single score
    return model_scores


def run_decoding_pipeline_single_model(constants, model_number, delay_name, item_status, trial_type='valid', cg=False):
    """
    Run the full decoding pipeline for a single model. Steps include: loading the data, constructing the colour labels,
    splitting the dataset into location arrays, shuffling trials and fitting and testing binary decoders.

    :param module constants: A Python module containing constants and configuration data for the simulation.
    :param int model_number: Number of the model for which to run the analysis pipeline.
    :param str delay_name: Desired delay interval. Choose from: 'precue', 'postcue' and 'postprobe' (only for Experiment
        4).
    :param str item_status: Which item to run the decoding analysis for, choose from 'cued', 'uncued', 'probed', and
        'unprobed'.
    :param str trial_type: Optional. Type of trials for which to run the analysis. Choose from 'valid' and 'invalid'.
        Default is 'valid'.
    :param bool cg: Optional. If True, runs a cross-generalising decoder (i.e. fits the decoder to the dataset
        corresponding to one condition (location), and tests it on a different condition). Default is False, which fits
        and tests the decoder on a dataset corresponding to a single condition (testing is done on withheld trials).
    :return: model_scores: array with decoder test scores for the given model.
    """
    # get data
    eval_data = load_model_data(constants, model_number, trial_type)

    # create labels
    labels_binned = get_colour_labels(constants, eval_data, item_status)

    # extract delay data
    delay_ix = get_delay_index(constants, delay_name)
    delay_data = extract_delay_data(delay_ix, eval_data)

    # split datasets by the cue location and shuffle trials
    labels, data = split_into_location_data(constants, labels_binned, delay_data)
    labels_shuffled, data_shuffled = shuffle_trials(model_number, labels, data)

    # fit decoders
    if cg:
        # fit a cross-generalising decoder (train on one location and test on the other)
        model_scores = get_decoding_across_planes(constants, data_shuffled, labels_shuffled)
    else:
        # fit a standard decoder - train and test on a single location (test on withheld trials)
        model_scores = get_decoding_within_plane(constants, data_shuffled, labels_shuffled)

    return model_scores
# %% 1) decode uncued colour items in post-cue delay - is the information still there?


def run_decoding_uncued_analysis(constants, trial_type='valid'):
    """
    Runs the entire uncued colour decoding analysis. LDA classifiers trained and tested in the post-cue delay.

    :param module constants: A Python module containing constants and configuration data for the simulation.
    :param str trial_type: Optional. Relevant for the probabilistic paradigm (experiment 4). Pass 'valid' or 'invalid'.
        The default is 'valid'.
    """
    print('RUNNING THE UNCUED COLOUR DECODING ANALYSIS FOR THE POST-CUE DELAY')
    assert constants.PARAMS['experiment_number'] == 1, 'Analysis only tested for Experiment 1.'
    # get decoding test scores
    model_scores = model_looper(constants, 'postcue', run_decoding_pipeline_single_model, 'uncued', trial_type)

    # save to file
    # save_path = f"{constants.PARAMS['FULL_PATH']}pca_data/{trial_type}_trials/"
    # with open(f"{save_path}decoding_acc_uncued_postcue_delay.pckl", 'wb') as f:
    #     pickle.dump(model_scores, f)

    # run contrast - test against chance decoding (50%)
    print('...Run contrast: mean test decoding significantly greater than chance (0.5) ')
    run_contrast_single_sample(model_scores, 0.5)

    print('...Mean decoding accuracy: %.4f' % model_scores.mean())


# %% 2: cross-temporal decoding


def run_ctg_pipeline_single_model(constants, model_number, time_range, item_status='cued', trial_type='valid'):
    """
    Run the cross-temporal generalisation decoding pipeline for a single model. Steps include loading the data,
    constructing the colour labels, splitting the dataset into location arrays, shuffling trials and fitting and
    testing the cross-temporal generalising decoders.

    :param module constants: A Python module containing constants and configuration data for the simulation.
    :param int model_number: Number of the model for which to run the analysis pipeline.
    :param list time_range: Start and end timepoints defining the temporal range for which to run the analysis, in the
        form: [t_min, t_max]
    :param str item_status: Which item to run the decoding analysis for, choose from 'cued', 'uncued', 'probed', and
        'unprobed'.
    :param str trial_type: Type of trials for which to run the decoder. Choose from 'valid' and 'invalid'
    :return: scores: array with decoder test scores, shape: (n_conditions, n_train_timepoints, n_test_timepoints), where
        conditions correspond to the cued item locations and their grand average, in that order
    """
    if item_status is not 'cued':
        raise NotImplementedError('Analysis not implemented for uncued items.')
    # run the cross-temporal decoding pipeline for a single model

    # load data
    model_data = load_model_data(constants, model_number, trial_type)
    # get cued labels, binned
    labels = get_colour_labels(constants, model_data, 'cued')

    # extract delay data
    delay_ix = np.arange(time_range[0], time_range[1])
    delay_data = extract_delay_data(delay_ix, model_data)

    # split datasets by the cue location and shuffle trials
    labels, data = split_into_location_data(constants, labels, delay_data)
    labels_shuffled, data_shuffled = shuffle_trials(model_number, labels, data)

    # reshape data into (n_trials x n_rec x n_timepoints)
    data_shuffled = [data_subset.transpose([0, -1, 1]) for data_subset in data_shuffled]

    # fit decoder
    scores = []
    # loop through the different cue locations
    for loc_data, loc_labels in zip(data_shuffled, labels_shuffled):
        scores.append(lda_cg_time(loc_data, loc_labels))
    scores = np.stack(scores)
    scores = np.concatenate((scores, scores.mean(0)[None]))   # add mean across the locations as the last row

    return scores


def run_ctg_analysis(constants, trial_type='valid', delay_length=7):
    """
    Runs the full cross-temporal decoding analysis pipeline. Binary LDA classifiers are trained to discriminate between
    the cued stimulus labels throughout the entire trial length. Saves the data into file and plots the cross-temporal
    decoding matrix, averaged across all models.

    :param module constants: A Python module containing constants and configuration data for the simulation.
    :param str trial_type: Type of trials for which to run the analysis. Choose from 'valid' and 'invalid'
    :param int delay_length:

    """
    print('RUNNING THE CROSS-TEMPORAL GENERALISATION DECODING ANALYSIS')
    assert constants.PARAMS['experiment_number'] in [1, 3], 'Analysis only tested for Experiments 1 and 3.'

    t_min = 0
    # if experiment 3 (variable delays), update the length of delay intervals saved in constants to the required value
    # For example, to compare findings to those from Experiment 1, pass 7
    # If we want to test the within-training range delay, set delay_length to 4 cycles; for the out-of-training range
    # delay, set it 10 cycles
    if constants.PARAMS['experiment_number'] == 3:
        dg.update_time_params(constants.PARAMS, delay_length)

    t_max = constants.PARAMS['trial_timepoints']['delay2_end']

    # fit the decoders for all models
    scores = model_looper(constants, [t_min, t_max], run_ctg_pipeline_single_model, 'cued', trial_type)
    # shape: (n_models, n_conditions, n_training_timepoints, n_test_timepoints) where n_conditions = 3 and conditions
    # correspond : (location 1 scores, location 2 scores, scores averaged across the locations)

    # plot
    # average scores across all models for plotting
    scores_av = scores[:, -1, :, :].mean(0)  # last column contains the scores averaged across the locations
    plotter.plot_ctg(constants, scores_av, [t_min, t_max])

    if constants.PLOT_PARAMS['save_plots']:
        if delay_length == 7:
            # standard delay length
            cond_name = ''
        elif delay_length == 4:
            cond_name = 'in_range_tempGen'
            plt.title(f'Cross-temporal generalisation decoding of cued colours: {cond_name}')
        elif delay_length == 10:
            cond_name = 'out_range_tempGen'
            plt.title(f'Cross-temporal generalisation decoding of cued colours: {cond_name}')
        else:
            cond_name = f'_delay_length_{delay_length}_cycles'
            plt.title(f'Cross-temporal generalisation decoding of cued colours: {cond_name}')
        plt.savefig(f"{constants.PARAMS['FIG_PATH']}cross_temp_decoding_alltimepoints{cond_name}.png")
        plt.savefig(f"{constants.PARAMS['FIG_PATH']}cross_temp_decoding_alltimepoints{cond_name}.svg")

# %% 3) compare maintenance mechanisms between Experiments 1 & 3


def get_delay_timepoints():
    """
    Get the indices corresponding to the delay 1 and 2 timepoints in the ctg matrix (the diagonal and off-diagonal
    elements).
    :return: d_x, d_y, diag_ix: x- and y-indices for the off-diagonal entries, diagonal indices
    """
    import constants.constants_expt1 as c1
    d1_start = c1.PARAMS['trial_timepoints']['delay1_start']
    d1_end = c1.PARAMS['trial_timepoints']['delay1_end']
    d2_start = c1.PARAMS['trial_timepoints']['delay2_start']
    d2_end = c1.PARAMS['trial_timepoints']['delay2_end']

    # get the indices corresponding to the delay 1 and 2 timepoints in the ctg matrix (only the off-diagonal elements)
    # delay1
    d1_x, d1_y = np.concatenate((np.triu_indices(c1.PARAMS['trial_timings']['delay1_dur'], k=1),
                                 np.tril_indices(c1.PARAMS['trial_timings']['delay1_dur'], k=-1)), 1)
    d1_x += d1_start
    d1_y += d1_start

    # delay2
    d2_x, d2_y = np.concatenate((np.triu_indices(c1.PARAMS['trial_timings']['delay2_dur'], k=1),
                                 np.tril_indices(c1.PARAMS['trial_timings']['delay2_dur'], k=-1)), 1)
    d2_x += d2_start
    d2_y += d2_start

    # concatenate
    d_x = np.concatenate((d1_x, d2_x))
    d_y = np.concatenate((d1_y, d2_y))
    # get the indices of the diagonal elements
    diag_ix = np.concatenate((np.arange(d1_start, d1_end), np.arange(d2_start, d2_end)))

    return d_x, d_y, diag_ix


def get_mean_delay_scores():
    """
    Calculate the mean diagonal and off-diagonal decoding scores for all models.
    :return: diag_scores, off_diag_scores
    """
    import constants.constants_expt1 as c1
    import constants.constants_expt3 as c3
    # get the paths to expt 1 (standard model) and 3 (variable delays model)
    # datafiles
    standard_model_path = c1.PARAMS['RESULTS_PATH']
    vardelay_model_path = c3.PARAMS['RESULTS_PATH']

    # get the indices corresponding to the delay timepoints in the ctg matrix  
    d_x, d_y, diag_ix = get_delay_timepoints()

    # calculate the mean delay scores for diagonal and off-diagonal elements
    off_diag_scores = np.empty((c1.PARAMS['n_models'], 2))  # model, condition
    diag_scores = np.empty((c1.PARAMS['n_models'], 2))  # model, condition

    for i, condition in enumerate([vardelay_model_path, standard_model_path]):
        with open(f"{condition}valid_trials/ctg_scores.pckl", 'rb') as f:
            scores_struct = pickle.load(f)

        # get the mean off- and diagonal scores, averaged across both delays
        diag_scores[:, i] = np.diagonal(scores_struct['scores'][:, :, -1, :])[:, diag_ix].mean(-1)
        off_diag_scores[:, i] = scores_struct['scores'][d_x, d_y, -1, :].mean(0)

    return diag_scores, off_diag_scores


def run_all_contrasts(off_diag_scores, diag_scores):
    """
    Run all 3 contrasts for the maintenance mechanism analysis. Contrasts 1 and 2 test if the mean off-diagonal decoding
    scores are significantly higher than the chance decoding level (50%). Contrast 3 tests whether the mean ratio
    between the off- and diagonal elements is significantly higher in the variable than fixed delay condition. The ratio
    is used as an index of the temporal stability of the code - for a perfectly temporally stable code, it should be ~1.

    :param numpy.ndarray off_diag_scores: Mean off-diagonal (cross-temporal) decoding scores for individual models.
        Values from the variable delay condition in the first, fixed - in the second column.
        Shape: (n_models, n_conditions)
    :param numpy.ndarray diag_scores: Mean diagonal decoding scores for individual models, in the same format as the
        off-diag_scores. Shape: (n_models, n_conditions)
    """
    # contrast 1: test variable delays off-diagonal mean against chance (0.5)
    print('...Contrast 1: Variable delays mean ctg decoding > chance')
    run_contrast_single_sample(off_diag_scores[:, 0], h_mean=.5, alt='greater')
    print('... mean = %.2f' % off_diag_scores[:, 0].mean())
    # contrast 2: test fixed delays off-diagonal mean against chance (0.5)
    print('...Contrast 2: Fixed delays mean ctg decoding > chance')
    run_contrast_single_sample(off_diag_scores[:, 1], h_mean=.5, alt='greater')
    print('... mean = %.2f' % off_diag_scores[:, 1].mean())
    # contrast 3: test if mean off-/diagonal ratio for variable delays > fixed delays
    print('...Contrast 3: Variable delays mean ratio off-/diagonal decoding > fixed delays')
    run_contrast_unpaired_samples(off_diag_scores[:, 0] / diag_scores[:, 0],
                                  off_diag_scores[:, 1] / diag_scores[:, 1],
                                  alt='greater')
    return


def run_maintenance_mechanism_analysis():
    """
    Runs the entire maintenance mechanism analysis. Calculates the mean delay cross-temporal generalisation scores and
    compares those between Experiments 1 & 3 to assess whether the networks trained with variable delay lengths (Expt 3)
    form a more temporally stable working memory code than those trained with fixed delays (Expt 1).
    """

    print('COMPARING THE MEMORY MAINTENANCE MECHANISMS BETWEEN EXPERIMENTS 1 & 3')
    # calculate the off- and on-diagonal decoding scores
    diag_scores, off_diag_scores = get_mean_delay_scores()

    # run the statistical tests
    run_all_contrasts(off_diag_scores, diag_scores)

    # boxplot of the off-diagonal decoding accuracy
    plotter.plot_off_diagonal(off_diag_scores)
    return
# %% 4) single-readout hypothesis


def run_cg_decoding_cued_analysis(constants, trial_type='valid'):
    """
    Runs the entire cued colour cross-generalisation decoding analysis. LDA classifiers trained and tested in the
    post-cue delay.

    :param module constants: A Python module containing constants and configuration data for the simulation.
    :param str trial_type: Optional. Relevant for the probabilistic paradigm (experiment 4). Pass 'valid' or 'invalid'.
        The default is 'valid'.
    """
    print('RUNNING THE CUED COLOUR DECODING AND CROSS-GENERALISATION ANALYSIS - SINGLE READOUT HYPOTHESIS REPORTED IN'
          'SUPPLEMENTARY NOTE 1')
    assert constants.PARAMS['experiment_number'] == 1, 'Analysis only tested for Experiment 1.'

    # get decoding cg scores
    model_scores = model_looper(constants, 'postcue', run_decoding_pipeline_single_model, 'cued', trial_type, cg=True)

    # save into file
    cg_decoding_cued_postcue_delay = {'test_accuracy': model_scores[:, 0], 'cross_gen_accuracy': model_scores[:, 1]}
    # save_path = f"{constants.PARAMS['FULL_PATH']}pca_data/{trial_type}_trials/"
    # with open(f"{save_path}cg_decoding_cued_postcue_delay.pckl", 'wb') as f:
    #     pickle.dump(cg_decoding_cued_postcue_delay, f)

    # Run contrasts
    print('...Run contrast: mean test decoding significantly greater than chance (0.5) ')
    run_contrast_single_sample(cg_decoding_cued_postcue_delay['test_accuracy'], 0.5)
    print('...Mean decoding accuracy: %.4f' % cg_decoding_cued_postcue_delay['test_accuracy'].mean())

    print('...Run contrast: mean cross_generalisation significantly greater than chance (0.5) ')
    run_contrast_single_sample(cg_decoding_cued_postcue_delay['cross_gen_accuracy'], 0.5)
    print('...Mean cross-generalisation accuracy: %.4f' % cg_decoding_cued_postcue_delay['cross_gen_accuracy'].mean())


# %% 5) analogue to the CDI analysis: compare the discriminability of colour representations pre-cue, as well as in the
# post-cue delay (i.e., after they are cued or uncued)


def run_colour_discrim_analysis(constants, trial_type='valid'):
    """
    Runs a decoding analysis complementary to the CDI analysis reported in Fig. 3H (for Expt 1). Train LDA decoders in
    cross-validation to discriminate between colours in the pre-cue delay, as well as after they are cued or uncued, and
    the compare test scores between the 3 conditions to assess how the amount of information about cued and uncued items
    changes across delays. Results from this analysis for Experiment 1 are reported in Supplementary Fig. S1 C.

    :param module constants: A Python module containing constants and configuration data for the simulation.
    :param str trial_type: Optional. Relevant for the probabilistic paradigm (experiment 4). Pass 'valid' or 'invalid'.
        The default is 'valid'.
    """
    assert trial_type == 'valid', "Analysis not implemented for invalid trials. Add this functionality by altering " \
                                  "the 'get_colour_labels' function."
    assert constants.PARAMS['experiment_number'] == 1, 'Analysis only tested for Experiment 1.'

    print('RUNNING THE COLOUR DISCRIMINABILITY ANALYSIS USING DECODING SCORES - REPORTED IN SUPPLEMENTARY FIGURE S1C')
    # get the pre-cue decoding test accuracy
    # this could be rewritten as a
    model_scores_precue = model_looper(constants, 'precue', run_decoding_pipeline_single_model, 'cued',
                                       trial_type)
    model_scores_uncued = model_looper(constants, 'postcue', run_decoding_pipeline_single_model, 'uncued',
                                       trial_type)
    model_scores_cued = model_looper(constants, 'postcue', run_decoding_pipeline_single_model, 'cued',
                                     trial_type)

    if constants.PARAMS['experiment_number'] == 4:
        model_scores_unprobed = model_looper(constants, 'postprobe', run_decoding_pipeline_single_model,
                                             'unprobed', trial_type)
        model_scores_probed = model_looper(constants, 'postprobe', run_decoding_pipeline_single_model,
                                           'probed', trial_type)
        all_scores = np.stack((model_scores_precue, model_scores_cued,
                               model_scores_uncued, model_scores_probed,
                               model_scores_unprobed), axis=1)
        labels = ['pre-cue', 'cued', 'uncued', 'probed', 'unprobed']
        if constants.PARAMS['cue_validity'] < 1:
            raise Warning("Analysis only implemented for valid trials. To recreate the full CDI comparison (including "
                          "invalid trials), extend the functionality in the 'get_colour_labels' function.")
    else:
        all_scores = np.stack((model_scores_precue, model_scores_cued, model_scores_uncued), axis=1)
        labels = ['pre-cue', 'cued', 'uncued']

    # load the post-cue (uncued and cued) test accuracies from file
    data_path = f"{constants.PARAMS['FULL_PATH']}pca_data/{trial_type}_trials/"
    pickle.dump(all_scores, open(data_path + 'cdi_analogous_decoding_scores.pckl', 'wb'))
    # export to csv for JASP
    scores_tbl = pd.DataFrame(all_scores, columns=labels)
    scores_tbl.to_csv(data_path + 'cdi_analogous_decoding_scores.csv')

    # do pairwise contrasts and plot - for Experiment 1
    if constants.PARAMS['experiment_number'] == 1:
        # contrasts - ran in JASP
        # rg.test_CDI_contrasts(constants, scores_tbl)

        # plot (as units of standard normal distribution)
        all_scores_transf = norm.ppf(all_scores)
        all_scores_transf_df = pd.DataFrame(all_scores_transf, columns=labels)

        _ = plotter.plot_CDI(constants, all_scores_transf_df, log_transform=False)
        plt.ylabel('Test decoding accuracy [snd units]')

        if constants.PLOT_PARAMS['save_plots']:
            plt.savefig(constants.PARAMS['FIG_PATH'] + ' cdi_analogue_with_decoding.png')
            plt.savefig(constants.PARAMS['FIG_PATH'] + ' cdi_analogue_with_decoding.svg')

    return
