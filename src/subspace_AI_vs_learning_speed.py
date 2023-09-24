#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 21:59:10 2022

This file contains the functions implementing the AI vs learning speed analysis (reported in Fig. 4D and 8C in the
manuscript).

@author: emilia
"""
import pickle
import torch
import pandas as pd
import numpy as np
import src.learning_dynamics_and_connectivity_analysis as ld


def get_training_speed_metrics(constants):

    n_epochs_to_convergence = []
    cum_loss = []

    for model_number in range(constants.PARAMS['n_models']):

        # get model loss
        track_training = ld.load_model_loss(constants, model_number)

        cum_loss.append(track_training['loss_epoch'].sum())
        n_epochs_to_convergence.append(len(track_training['loss_epoch']))

    cum_loss = torch.stack(cum_loss)
    n_epochs_to_convergence = torch.from_numpy(np.stack(n_epochs_to_convergence))
    return n_epochs_to_convergence, cum_loss


def get_all_data(constants):
    print('Get all data for the geometry vs learning speed analysis.')

    # get training speed
    n_epochs_to_convergence, cum_loss = get_training_speed_metrics(constants)

    # get the AI values
    data_path = f"{constants.PARAMS['RESULTS_PATH']}valid_trials/"
    ai_tbl = []
    for geometry_name in ['cued', 'uncued', 'cued_uncued']:
        try:
            # load data
            with open(f"{data_path}AI_tbl_{geometry_name}.pckl", 'rb') as f:
                ai_data = pickle.load(f)
            # extract the 2D AI values
            dim_2_ix = 0
            if geometry_name in ['cued', 'uncued']:
                # append only the post-cue (or post-probe) timepoint
                timepoint = -1
                ai_tbl.append(ai_data[dim_2_ix, timepoint, :])
            else:
                # cued-uncued data only contains the post-cue timepoint
                ai_tbl.append(ai_data[dim_2_ix, :].squeeze())

        except FileNotFoundError:
            raise FileNotFoundError(f"Make sure that the {geometry_name} AI analysis has been performed and data "
                                    f"saved.")

    ai_tbl = np.stack(ai_tbl)

    # stack all into a pandas dataframe and export to csv to be analysed in JASP
    tbl = np.concatenate((n_epochs_to_convergence[:, None], cum_loss[:, None], ai_tbl.T), axis=1)
    df = pd.DataFrame(tbl, columns=['n_epochs_to_convergence', 'cum_loss', 'AI_cued',
                                    'AI_uncued', 'AI_cued_uncued'])
    df.to_csv(f"{data_path}geometry_vs_learning_regression_data.csv")
    print('Data export to csv. Analyse in JASP.')
    return
