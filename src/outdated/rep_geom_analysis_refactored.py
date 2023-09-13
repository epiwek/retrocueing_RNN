#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 17:30:57 2021

@author: emilia
"""
import pickle
import os.path
import torch
import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.linalg import lstsq, inv
from scipy.stats import zscore, mode, pearsonr, spearmanr, shapiro

from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from scipy.spatial.distance import squareform, pdist
from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes
from scipy.spatial import ConvexHull
# import hypertools
from scipy.stats import shapiro, ttest_1samp, wilcoxon

import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull

import src.helpers as helpers
import src.vec_operations as vops

import src.custom_plot as cplot
import pdb

from src.generate_data_vonMises import update_time_params

import src.subspace as subspace
from src.subspace import Geometry


# %% define low-level plotting functions

def plot_plane(ax, verts, fc='k', a=0.2):
    # plot a polygon with given vertices in 3D
    ax.add_collection3d(Poly3DCollection([verts], facecolor=fc, edgecolor=[], alpha=a))








# def plot_subspace(ax,points,plane_vecs,fc='k',a=0.2):
#     # plot the best-fitting plane as a convex hull (not necessarily a 
#     # quadrilateral) with vertices being the projections of original points 
#     # onto the plane 

#     if (points.shape[1]!=3):
#         raise NotImplementedError('Check shape of data matrix - should be [n_points,3]')

#     # 
#     plane_basis = np.concatenate((plane_vecs,
#                                   np.cross(plane_vecs[0,:],plane_vecs[1:,])))
#     # find vertices
#     n_points = points.shape[0]
#     verts = np.empty((n_points,3))
#     verts_2d = np.empty((n_points,3))
#     com = np.mean(points, axis=0) # centre of mass

#     for i in range(n_points):
#         verts[i,:] = vops.getProjection(points[i,:]-com,plane_vecs) # get projection of demeaned 3d points
#         # change basis to that defined by the plane + its normal to zero out 
#         # the 3rd coordinate
#         # this is so that can fit a convex hull to the data with 
#         # vops.defPlaneShape() - otherwise ConvexHull will complain about 
#         # points being coplanar 
#         verts_2d[i,:] =  plane_basis @ verts[i,:]
#         verts[i,:] += com #add the mean back


#     # only pass 2D coordinates to ConvexHull
#     convex_verts, sorting_order = vops.defPlaneShape(verts_2d[:,:2],[])

#     sorted_verts = verts[sorting_order,:] # sorted 3D coordinates
#     # plot the best-fit plane
#     plot_plane(ax,sorted_verts,fc,a)
#     #return verts, sorted_verts


def plot_subspace(ax, points, plane_vecs, fc='k', a=0.2):
    # plot the best-fitting plane as a quadrilateral with vertices being the projections of original points onto the plane 

    if (points.shape[1] != 3):
        raise NotImplementedError('Check shape of data matrix - should be [n_points,3]')

    # find vertices
    n_points = points.shape[0]
    verts = np.zeros((n_points, 3))

    com = np.mean(points, axis=0)  # centre of mass

    for i in range(n_points):
        # get projection of demeaned 3d points
        verts[i, :] = vops.get_projection(points[i, :] - com, plane_vecs)
        verts[i, :] += com  # add the mean back

    # sort vertices according to shortest path - so that plotted plane will be a quadrilateral
    sorted_verts, sorting_order = vops.sort_by_path_length(verts)

    # plot the best-fit plane
    plot_plane(ax, sorted_verts, fc, a)
    # return verts, sorted_verts


def plot_geom_and_subspace(constants, data, plot_title, size=[9, 6], custom_labels=None):
    n_colours = constants.PARAMS['B']
    plt.figure(figsize=size)
    ax = plt.subplot(111, projection='3d')
    plot_geometry(ax,
                  data['3Dcoords'],
                  data['pca'],
                  constants.PLOT_PARAMS['4_colours'],
                  custom_labels=custom_labels)
    plot_subspace(ax, data['3Dcoords'][:n_colours, :], data['plane1'].components_, fc='k', a=0.2)
    plot_subspace(ax, data['3Dcoords'][n_colours:, :], data['plane2'].components_, fc='k', a=0.2)
    ax.set_title(plot_title + ', ' + r'$\theta$' + ' = %.1f' % data['theta'] + '°')
    helpers.equal_axes(ax)
    ax.tick_params(pad=4.0)
    plt.tight_layout()


#%% geometry-specific plotters




#%% analyses

def get_plane_angle_cued_across_delays(constants, cv=2):
    """
    For each location, get the plane angle theta between the Cued subspaces from the
    pre- and post-cue delay intervals. Do it in 2-fold cross-validation to 
    determine if both or only one of the pre-cue subspaces is rotated post-cue,
    to form a parallel plane geometry.

    Parameters
    ----------
    constants : dict
        Experiment parameters.
    cv : int, optional
        Number of cross-validation folds. The default is 2.

    Returns
    -------
    theta_unrotated_mean : array
        AI values for the 'unrotated' and 'rotated' planes, averaged across cv 
        folds. Format: (n_dims,(unrotated,rotated),model)
    theta_rotated_mean : array
        Indexes of the unrotated plane for each model.
    psi_unrotated_mean : dict
        Train and test trial indexes for the cross-validation folds.
    psi_unrotated_mean
    """
    load_path = constants.PARAMS['FULL_PATH'] + 'pca_data/valid_trials'

    theta_ad_test = np.zeros((2, constants.PARAMS['n_models'], cv))
    psi_ad_test = np.zeros((2, constants.PARAMS['n_models'], cv))

    # if experiment 2, update the length of delay intervals saved in constants 
    # to 7 cycles (same as for expt1)
    if constants.PARAMS['experiment_number'] == 2:
        update_time_params(constants.PARAMS, 7)

    d1_ix = constants.PARAMS['trial_timepoints']['delay1_end'] - 1
    d2_ix = constants.PARAMS['trial_timepoints']['delay2_end'] - 1

    # uncomment below if you would like to draw new cross-validation folds
    # trial_ixs = {'train':{},'test':{}}

    # otherwise - load the indices of trials used for the cross-validated AI analysis
    trial_ixs = pickle.load(open(load_path + '/trial_ixs_for_unrotrot_analysis.pckl', 'rb'))

    same_ixs = np.zeros((constants.PARAMS['n_models'], cv))

    PVE_3D = np.zeros((2, 2, constants.PARAMS['n_models']))  # cv folds, unrotated/rotated, model number

    for model in range(constants.PARAMS['n_models']):
        # load model data
        model_number = str(model)
        print('Model ' + model_number)
        f = open(load_path + '/eval_data_model' + model_number + '.pckl', 'rb')
        eval_data = pickle.load(f)
        f.close()

        # uncomment below if want to draw new cross-validation folds
        # train,test = get_trial_ixs(params,cv=2)

        # otherwise - load those previusly used for the analogous AI analysis
        train, test = trial_ixs['train'][model_number], trial_ixs['test'][model_number]

        loc_split = constants.PARAMS['B'] # end row index of the location 1 datapoints
        for i in range(2):
            # bin the train and test datasets into colour bins
            data_train = helpers.bin_data(eval_data['data'][train[i], :, :], constants.PARAMS)
            data_test = helpers.bin_data(eval_data['data'][test[i], :, :], constants.PARAMS)

            # extract the train and test arrays
            delay1_train = data_train[:, d1_ix, :]
            delay1_test = data_test[:, d1_ix, :]

            delay2_train = data_train[:, d2_ix, :]
            delay2_test = data_test[:, d2_ix, :]

            # get the location-wise subspaces from training data
            loc1_subspace = subspace.Geometry(torch.cat((delay1_train[:loc_split, :],
                                                        delay2_train[:loc_split, :])),
                                              constants,['pre_delay', 'post_delay'])
            loc1_subspace.get_geometry()
            # loc 2 cued
            loc2_subspace = subspace.Geometry(torch.cat((delay1_train[loc_split:, :],
                                                        delay2_train[loc_split:, :])),
                                              constants,
                                              ['pre_delay', 'post_delay'])
            loc2_subspace.get_geometry()

            # find the plane that stays the 'same' - abs cos of theta will be larger
            same_ixs[model,i] = np.argmax([np.abs(loc1_subspace.cos_theta),np.abs(loc2_subspace.cos_theta)])

            if same_ixs[model, i] == 0:
                stay_plane_ix = np.arange(loc_split)
            else:
                stay_plane_ix = np.arange(loc_split, loc_split * 2)
            switch_plane_ix = np.setdiff1d(np.arange(loc_split * 2), stay_plane_ix)

            # get the unrotated and rotated (test) data
            unrotated_data = torch.cat((delay1_test[stay_plane_ix, :],
                                   delay2_test[stay_plane_ix, :]), dim=0)
            rotated_data = torch.cat((delay1_test[switch_plane_ix, :],
                                 delay2_test[switch_plane_ix, :]), dim=0)

            # get the unrotated and rotated subspaces
            unrotated_subspace = subspace.Geometry(unrotated_data, constants, ['pre_delay', 'post_delay'])
            rotated_subspace = subspace.Geometry(rotated_data, constants, ['pre_delay', 'post_delay'])

            unrotated_subspace.get_geometry()
            rotated_subspace.get_geometry()

            # save the plane angles, phase alignment angles and PVEs
            theta_ad_test[0, model, i] = unrotated_subspace.theta_degrees
            theta_ad_test[1, model, i] = rotated_subspace.theta_degrees

            psi_ad_test[0, model, i] = unrotated_subspace.psi_degrees
            psi_ad_test[1, model, i] = rotated_subspace.psi_degrees

            PVE_3D[i, 0, model] = unrotated_subspace.PVEs.sum()
            PVE_3D[i, 1, model] = rotated_subspace.PVEs.sum()

    # get averages across cv splits
    theta_unrotated_mean = np.abs(np.array([theta_ad_test[0, m, :].mean(-1) if np.sign(
        theta_ad_test[0, m, :]).sum() != 0 else np.abs(theta_ad_test[0, m, :]).mean(-1) for m in
                                            range(constants.PARAMS['n_models'])]))
    theta_rotated_mean = np.abs(np.array([theta_ad_test[1, m, :].mean(-1) if np.sign(
        theta_ad_test[1, m, :]).sum() != 0 else np.abs(theta_ad_test[1, m, :]).mean(-1) for m in
                                          range(constants.PARAMS['n_models'])]))
    psi_unrotated_mean = np.array([psi_ad_test[0, m, :].mean(-1) if np.sign(
        psi_ad_test[0, m, :]).sum() != 0 else np.abs(psi_ad_test[0, m, :]).mean(-1) for m in
                                   range(constants.PARAMS['n_models'])])
    psi_rotated_mean = np.array([psi_ad_test[1, m, :].mean(-1) if np.sign(psi_ad_test[1, m, :]).sum() != 0 else np.abs(
        psi_ad_test[1, m, :]).mean(-1) for m in range(constants.PARAMS['n_models'])])

    # save
    # pickle.dump(theta_unrotated_mean, open(load_path + '/theta_unrotated_plane.pckl', 'wb'))
    # pickle.dump(theta_rotated_mean, open(load_path + '/theta_rotated_plane.pckl', 'wb'))
    # pickle.dump(psi_unrotated_mean, open(load_path + '/psi_unrotated_plane.pckl', 'wb'))
    # pickle.dump(psi_rotated_mean, open(load_path + '/psi_rotated_plane.pckl', 'wb'))

    return theta_unrotated_mean, theta_rotated_mean, psi_unrotated_mean, psi_rotated_mean


def get_CDI(constants):
    '''
    Calculate the colour discriminability index (CDI) for different trial
    timepoints:
        - pre-cue
        - post-cue (for cued and uncued colours)
        - post-probe (for experiment 3) : for both valid and invalid trials.
    Data from all trials characterised by the same cued location is projected
    into a 3D subspace with PCA, to get the 3D coordinates for each
    location-specific planes at the specified timepoints. CDI is defined as the
    surface area of the data quadrilateral designated by the location-specific
    colour datapoints.


    Parameters
    ----------
    constants : TYPE
        DESCRIPTION.

    Returns
    -------
    CDI_av : TYPE
        DESCRIPTION.
    CDI : TYPE
        DESCRIPTION.

    '''
    n_colours = constants.PARAMS['B']
    n_timepoints = 3
    n_locations = 2

    n_cued_locations = 2
    if constants.PARAMS['experiment_number'] == 4:
        if constants.PARAMS['cue_validity'] < 1:
            trial_types = ['valid', 'invalid']
        else:
            trial_types = ['valid']
        n_validity_types = len(trial_types)
        CDI = np.empty((constants.PARAMS['n_models'], n_cued_locations, n_timepoints, n_locations, n_validity_types))
    else:
        CDI = np.empty((constants.PARAMS['n_models'], 2, 2, 2, 1))
        # model, trial type (cued location), timepoint, plane1/plane2, trial type: valid/invalid
    PVEs = np.empty((constants.PARAMS['n_models'], 2))  # model, trial type
    # [model, cued location, pre-post, cued/uncued]
    for model in range(constants.PARAMS['n_models']):

        if constants.PARAMS['experiment_number'] == 4:
            # for trials defined by a given *cued* location (up,down),
            # aggregate data from valid and invalid trials

            # for example, for cued 'up' trials, the rows of the aggregated data
            # array will correspond to (in sets of n_colours rows, from top to
            # bottom):
            #  (1) pre_cue_up - valid trials
            #  (2) pre_cue_down - valid trials
            #  (3) cued - valid trials
            #  (4) uncued - valid trials
            #  (5) probed - valid trials
            #  (6) unprobed - valid trials
            #  (7) pre_cue_up - invalid trials
            #  (8) pre_cue_down - invalid trials
            #  (9) cued - invalid trials
            #  (10) uncued - invalid trials
            #  (11) probed - invalid trials
            #  (12) unprobed - invalid trials
            # giving rise to a M=n_colours*12 by N=n_neurons array

            cued_up, cued_down = [], []

            for trials in trial_types:
                load_path = constants.PARAMS['FULL_PATH'] + 'pca_data/' + trials + '_trials/'

                path_cued = load_path + 'pca_data_model' + str(model) + '.pckl'
                path_uncued = load_path + 'pca_data_uncued_model' + str(model) + '.pckl'
                path_probed = load_path + 'pca_data_probed_model' + str(model) + '.pckl'
                path_unprobed = load_path + 'pca_data_unprobed_model' + str(model) + '.pckl'

                data_binned_cued = pickle.load(open(path_cued, 'rb'))
                data_binned_uncued = pickle.load(open(path_uncued, 'rb'))
                data_binned_probed = pickle.load(open(path_probed, 'rb'))
                data_binned_unprobed = pickle.load(open(path_unprobed, 'rb'))

                # find row indices corresponding to the 'cued_up' and 'cued_down'
                # trials in the pca_data arrays
                cued_up_ix = np.where(data_binned_cued['labels']['cued_loc'] == 0)[0]
                cued_down_ix = np.where(data_binned_cued['labels']['cued_loc'] == 1)[0]

                cued_up.append(torch.cat((data_binned_cued['delay1'][cued_up_ix, :],
                                          data_binned_uncued['delay1'][cued_up_ix, :],
                                          data_binned_cued['delay2'][cued_up_ix, :],
                                          data_binned_uncued['delay2'][cued_up_ix, :],
                                          data_binned_probed['delay3'][cued_up_ix, :],
                                          data_binned_unprobed['delay3'][cued_up_ix, :])))

                cued_down.append(torch.cat((data_binned_cued['delay1'][cued_down_ix, :],
                                            data_binned_uncued['delay1'][cued_down_ix, :],
                                            data_binned_cued['delay2'][cued_down_ix, :],
                                            data_binned_uncued['delay2'][cued_down_ix, :],
                                            data_binned_probed['delay3'][cued_down_ix, :],
                                            data_binned_unprobed['delay3'][cued_down_ix, :])))

            # convert the lists into the M=n_colours*12 by N=n_neurons arrays
            cued_up = torch.stack(cued_up).reshape(-1, 200)
            cued_down = torch.stack(cued_down).reshape(-1, 200)

        else:

            # experiment 1
            load_path = constants.PARAMS['FULL_PATH'] + 'pca_data/' + 'valid_trials/'

            path_cued = load_path + 'pca_data_model' + str(model) + '.pckl'
            path_uncued = load_path + 'pca_data_uncued_model' + str(model) + '.pckl'

            data_binned_cued = pickle.load(open(path_cued, 'rb'))
            data_binned_uncued = pickle.load(open(path_uncued, 'rb'))

            # cued-up trials
            cued_up = torch.cat((data_binned_cued['delay1'][:n_colours, :],
                                 data_binned_uncued['delay1'][:n_colours, :],
                                 data_binned_cued['delay2'][:n_colours, :],
                                 data_binned_uncued['delay2'][:n_colours, :]))
            # pre-cue:up;pre-cue:down;post-cue:up;post-cue:down

            # cued-down trials
            cued_down = torch.cat((data_binned_cued['delay1'][n_colours:, :],
                                   data_binned_uncued['delay1'][n_colours:, :],
                                   data_binned_cued['delay2'][n_colours:, :],
                                   data_binned_uncued['delay2'][n_colours:, :]))
            # pre-cue:down; pre-cue:up;post-cue:down;post-cue:up

        # get the common 3D subspace for cued-up trials
        cued_up_subspace = subspace.Geometry(cued_up, constants)
        cued_up_subspace.get_3d_coords()
        PVEs[model, 0] = cued_up_subspace.PVEs.sum()


        # cued_up_pca, cued_up_3Dcoords = get_3D_coords(cued_up)
        # PVEs[model, 0] = cued_up_pca.explained_variance_ratio_.sum()

        # get the common subspace for cued-down trials
        cued_down_subspace = subspace.Geometry(cued_down,constants)
        cued_down_subspace.get_3d_coords()
        PVEs[model, 1] = cued_down_subspace.PVEs.sum()

        # cued_down_pca, cued_down_3Dcoords = get_3D_coords(cued_down)
        # PVEs[model, 1] = cued_down_pca.explained_variance_ratio_.sum()

        # calculate the CDI
        # for cued_loc_ix, cued_loc_3D_coords in enumerate([cued_up_3Dcoords, cued_down_3Dcoords]):
        for cued_loc_ix, cued_loc_3D_coords in enumerate([cued_up_subspace.coords_3d, cued_down_subspace.coords_3d]):
            # dimensions correspond to:
            # model, trial type (cued location), timepoint, plane1/plane2,
            # trial type: valid/invalid

            # pre-cue, valid trials
            CDI[model, cued_loc_ix, 0, 0, 0] = vops.quadrilat_area(cued_loc_3D_coords[:n_colours, :])
            CDI[model, cued_loc_ix, 0, 1, 0] = vops.quadrilat_area(cued_loc_3D_coords[n_colours:n_colours * 2, :])
            # post-cue, valid trials
            CDI[model, cued_loc_ix, 1, 0, 0] = vops.quadrilat_area(cued_loc_3D_coords[n_colours * 2:n_colours * 3, :])
            CDI[model, cued_loc_ix, 1, 1, 0] = vops.quadrilat_area(cued_loc_3D_coords[n_colours * 3:n_colours * 4, :])

            if constants.PARAMS['experiment_number'] == 4:
                # add the probe timepoints and invalid trials

                # post-probe, valid trials
                CDI[model, cued_loc_ix, 2, 0, 0] = vops.quadrilat_area(cued_loc_3D_coords[n_colours * 4:n_colours * 5, :])
                CDI[model, cued_loc_ix, 2, 1, 0] = vops.quadrilat_area(cued_loc_3D_coords[n_colours * 5:n_colours * 6, :])

                if constants.PARAMS['cue_validity'] < 1:
                    # pre-cue, invalid trials
                    CDI[model, cued_loc_ix, 0, 0, 1] = vops.quadrilat_area(cued_loc_3D_coords[n_colours * 6:n_colours * 7, :])
                    CDI[model, cued_loc_ix, 0, 1, 1] = vops.quadrilat_area(cued_loc_3D_coords[n_colours * 7:n_colours * 8, :])
                    # post-cue, invalid trials
                    CDI[model, cued_loc_ix, 1, 0, 1] = vops.quadrilat_area(cued_loc_3D_coords[n_colours * 8:n_colours * 9, :])
                    CDI[model, cued_loc_ix, 1, 1, 1] = vops.quadrilat_area(
                        cued_loc_3D_coords[n_colours * 9:n_colours * 10, :])
                    # post-probe, invalid trials
                    CDI[model, cued_loc_ix, 2, 0, 1] = vops.quadrilat_area(
                        cued_loc_3D_coords[n_colours * 10:n_colours * 11, :])
                    CDI[model, cued_loc_ix, 2, 1, 1] = vops.quadrilat_area(
                        cued_loc_3D_coords[n_colours * 11:n_colours * 12, :])

    # average across the trial types (cued location)
    CDI_av = CDI.mean(1).squeeze()
    if constants.PARAMS['experiment_number'] == 4:
        # probabilistic paradigms:
        # average the pre-cue timepoint across (1) cued/uncued planes and (2)
        # valid and invalid trials, leading to one pre-cue entry
        # average the post-cue timepoint across valid and invalid trials,
        # leading to one cued and one uncued entry
        # keep all the post-probe entries: probed_valid, probed_invalid,
        # unprobed_valid and unprobed_invalid
        if constants.PARAMS['cue_validity'] < 1:
            CDI_av = np.concatenate((np.expand_dims(CDI_av[:, 0, :, :].mean((-1, -2)), -1),
                                     CDI_av[:, 1, :, :].mean(-1),
                                     CDI_av[:, 2, :, :].reshape([30, -1])), 1)

            CDI_for_stats = CDI.mean(1).squeeze()
            # model, timepoint, plane1/plane2, trial type (valid/invalid)
            CDI_for_stats = CDI_for_stats.reshape((CDI_for_stats.shape[0],
                                                   np.product(CDI_for_stats.shape[1:])))

            time_labels = ['precue_', 'postcue_']
            loc_labels = ['cued', 'uncued']
            trial_labels = ['_valid', '_invalid']

            column_labels = [tt + l + t for tt in time_labels for l in loc_labels for t in trial_labels]
            column_labels2 = ['postprobe_' + l + t for l in ['probed', 'unprobed'] for t in trial_labels]
            for i in column_labels2:
                column_labels.append(i)


        else:
            # deterministic paradigm:
            # average the pre-cue timepoint across cued/uncued planes
            CDI_av = np.concatenate((np.expand_dims(CDI_av[:, 0, :].mean(-1), -1),
                                     CDI_av[:, 1, :],
                                     CDI_av[:, 2, :]), 1)
            CDI_for_stats = CDI.mean(1).squeeze()
            # model, timepoint, plane1/plane2 (cued/uncued)
            CDI_for_stats = CDI_for_stats.reshape((CDI_for_stats.shape[0],
                                                   np.product(CDI_for_stats.shape[1:])))
            time_labels = ['precue_', 'postcue_']
            loc_labels = ['cued', 'uncued']
            column_labels = [tt + l for tt in time_labels for l in loc_labels]
            column_labels2 = ['postprobe_' + l for l in ['probed', 'unprobed']]
            for i in column_labels2:
                column_labels.append(i)

        CDI_av_df = pd.DataFrame(CDI_for_stats,
                                 columns=column_labels)
        save_path = constants.PARAMS['FULL_PATH'] + 'pca_data/'
        CDI_av_df.to_csv(save_path + 'CDI.csv')

        # save the CDI_av array to file
        pickle.dump(CDI_av, open(save_path + 'CDI_for_plotting.pckl', 'wb'))
        # save the unaveraged array
        pickle.dump(CDI, open(save_path + 'CDI_unaveraged.pckl', 'wb'))
    else:
        # for pre-cue, average the cued and uncued
        CDI_av = np.concatenate((np.expand_dims(CDI_av[:, 0, :].mean(-1), -1), CDI_av[:, 1, :]), 1)
        CDI_av_df = pd.DataFrame(CDI_av, columns=['pre-cue', 'post_cued', 'post_uncued'])
        # # save structure
        CDI_av_df.to_csv(load_path + '/CDI.csv')
        pickle.dump(CDI_av, open(load_path + 'CDI_for_plotting.pckl', 'wb'))
        pickle.dump(CDI, open(load_path + 'CDI_unaveraged.pckl', 'wb'))

    mean_PVE = PVEs.mean() * 100
    sem_PVE = np.std(PVEs.mean(-1)) * 100 / np.sqrt(constants.PARAMS['n_models'])
    print('%% variance explained by 3D subspaces mean = %.2f, sem = %.3f' % (mean_PVE, sem_PVE))
    return CDI_av





def plot_CDI(constants, CDI, logTransform=True, save_fig=True):
    '''
    Plot the CDI metric. Plot style depends on the experiment number. For
    experiment 3, this is a barplot with bar height corresponding to the mean
    across models and error bars corresponding to SEM. Bars correspond to
    conditions, namely: pre-cue, cued, uncued, probed and unprobed colours.
    For experiment 1,


    Parameters
    ----------
    constants : TYPE
        DESCRIPTION.
    CDI : TYPE
        DESCRIPTION.
    logTransform : TYPE, optional
        DESCRIPTION. The default is True.
    save_fig : TYPE, optional
        DESCRIPTION. The default is True.
    trials : TYPE, optional
        DESCRIPTION. The default is 'valid'.

    Returns
    -------
    None.

    '''
    if logTransform:
        CDI = np.log(CDI)
    pal = sns.color_palette("dark")

    if constants.PARAMS['experiment_number'] == 4:
        pre_cue_ix = 0
        cued_ix, uncued_ix = 1, 2

        if constants.PARAMS['cue_validity'] < 1:
            cols = ['k', pal[9], pal[2], pal[6], pal[6], pal[4], pal[4]]
            x_vals = np.array([0, .5, .75, 1.25, 1.5, 1.75, 2])
            probed_valid_ix = 3
            probed_invalid_ix = 4
            unprobed_valid_ix = 5
            unprobed_invalid_ix = 6
        else:
            cols = ['k', pal[9], pal[2], pal[6], pal[4]]
            x_vals = np.array([0, .5, .75, 1.25, 1.5])
            probed_valid_ix = 3
            unprobed_valid_ix = 4
    else:
        cols = ['k', pal[9], pal[2]]
        x_vals = np.array([0, .875, 1.125])
        pre_cue_ix = 0
        cued_ix, uncued_ix = 1, 2

    plt.figure(figsize=(6.65, 5))
    ax = plt.subplot(111)

    ms = 16
    n_conditions = CDI.shape[1]

    if constants.PARAMS['experiment_number'] != 4:
        # plot individual model datapoints
        for model in range(0, constants.PARAMS['n_models']):
            # plot the lines joining datapoints belonging to a single model
            # pre-cue to cued
            ax.plot(x_vals[[pre_cue_ix, cued_ix]], CDI[model, [pre_cue_ix, cued_ix]], 'k-', alpha=.2)
            # pre-cue to uncued
            ax.plot(x_vals[[pre_cue_ix, uncued_ix]], CDI[model, [pre_cue_ix, uncued_ix]], 'k-', alpha=.2)

            # plot the datapoints
            ax.plot(x_vals[pre_cue_ix], CDI[model, 0], 'o',
                    c=cols[pre_cue_ix], markersize=ms)  # pre-cue
            ax.plot(x_vals[cued_ix], CDI[model, cued_ix], '^',
                    c=cols[cued_ix], markersize=ms)  # cued
            ax.plot(x_vals[uncued_ix], CDI[model, uncued_ix],
                    'X', c=cols[uncued_ix], markersize=ms)  # uncued

            # uncomment code below (and get rid of the if statement above) if
            # want to add datapoints for expt4, but this makes the plot look
            # really messy
            if constants.PARAMS['experiment_number'] == 4:
                # lines
                # cued to probed (valid trials)
                ax.plot(x_vals[[cued_ix, probed_valid_ix]],
                        CDI[model, [cued_ix, probed_valid_ix]], 'g-', alpha=.2)
                # cued to unprobed (invalid trials)
                ax.plot(x_vals[[cued_ix, unprobed_invalid_ix]],
                        CDI[model, [cued_ix, unprobed_invalid_ix]], 'r--', alpha=.2)
                # uncued to unprobed (valid trials)
                ax.plot(x_vals[[uncued_ix, unprobed_valid_ix]],
                        CDI[model, [uncued_ix, unprobed_valid_ix]], 'g-', alpha=.2)
                # uncued to probed (invalid trials)
                ax.plot(x_vals[[uncued_ix, probed_invalid_ix]],
                        CDI[model,], 'r--', alpha=.2)

                # datapoints
                # probed-valid
                ax.plot(x_vals[probed_valid_ix], CDI[model, probed_valid_ix],
                        's', c=cols[probed_valid_ix], markersize=ms)
                # unprobed-valid
                ax.plot(x_vals[unprobed_valid_ix], CDI[model, unprobed_valid_ix],
                        'd', c=cols[unprobed_valid_ix], markersize=ms)
                # probed-invalid
                ax.plot(x_vals[probed_invalid_ix], CDI[model, probed_invalid_ix],
                        's', c=cols[probed_invalid_ix], markersize=ms, fillstyle='none')
                # unprobed-invalid
                ax.plot(x_vals[unprobed_invalid_ix], CDI[model, unprobed_invalid_ix],
                        'd', c=cols[unprobed_invalid_ix], markersize=ms, fillstyle='none')

    # add means (barplot)
    means = CDI.mean(0)

    if constants.PARAMS['experiment_number'] == 4:
        # plot bars with SEM errorbars
        sem = np.std(CDI, axis=0) / np.sqrt(CDI.shape[0])
        if constants.PARAMS['cue_validity'] < 1:
            data_labels = ['precue', 'cued', 'uncued', 'probed valid', 'probed invalid',
                           'unprobed valid', 'unprobed invalid']
            # plot bars
            for i in range(n_conditions):
                if np.logical_or(i == probed_invalid_ix, i == unprobed_invalid_ix):
                    # plot hatched bars
                    ax.bar(x_vals[i], means[i], yerr=sem[i], ec=cols[i], facecolor='w',
                           capsize=2, hatch='///', alpha=.25, width=.25, label=data_labels[i])
                elif i == 0:
                    # no data label
                    ax.bar(x_vals[i], means[i], yerr=sem[i], facecolor=cols[i], capsize=2,
                           alpha=.2, width=.25)
                else:
                    ax.bar(x_vals[i], means[i], yerr=sem[i], facecolor=cols[i], capsize=2,
                           alpha=.2, width=.25, label=data_labels[i])
        else:
            data_labels = ['precue', 'cued', 'uncued', 'probed', 'unprobed']
            # plot bars
            for i in range(n_conditions):
                if i == 0:
                    # no data label
                    ax.bar(x_vals[i], means[i], yerr=sem[i], facecolor=cols[i], capsize=2,
                           alpha=.2, width=.25)
                else:
                    ax.bar(x_vals[i], means[i], yerr=sem[i], facecolor=cols[i], capsize=2,
                           alpha=.2, width=.25, label=data_labels[i])

    else:
        data_labels = ['precue', 'cued', 'uncued']
        for i in range(n_conditions):
            if i == 0:
                # no data label
                ax.bar(x_vals[i], means[i], facecolor=cols[i], alpha=.2, width=.25)
            else:
                ax.bar(x_vals[i], means[i], facecolor=cols[i], alpha=.2, width=.25,
                       label=data_labels[i])
        ax.set_xlim([-0.25, 1.375])

    # set x-ticks and labels
    if constants.PARAMS['experiment_number'] == 4:
        xticks = [x_vals[pre_cue_ix], x_vals[[cued_ix, uncued_ix]].mean(), x_vals[probed_valid_ix:].mean()]
        ax.set_xticks(xticks)
        ax.set_xticklabels(labels=['pre-cue', 'post-cue', 'post-probe'])
    else:
        ax.set_xticks(range(2))
        ax.set_xticklabels(labels=['pre-cue', 'post-cue'])

    if logTransform:
        ax.set_ylabel('log(CDI)')
    else:
        ax.set_ylabel('CDI')

    plt.legend()
    plt.tight_layout()

    if save_fig:
        plt.savefig(constants.PARAMS['FIG_PATH'] + 'CDI.png')
        plt.savefig(constants.PARAMS['FIG_PATH'] + 'CDI.svg')


def test_CDI_contrasts(CDI):
    # first contrast: cued >> uncued
    # second contrast: cued >> pre-cue
    # third contrast (uncued == pre-cue) done in JASP

    contrasts = ['Contrast 1: cued > uncued', 'Contrast 2: cued > pre-cue']
    ixs = np.array([[1, 2], [1, 0]])
    for c in range(2):
        print(contrasts[c])
        s, p = shapiro(CDI[:, ixs[c, 0]] - CDI[:, ixs[c, 1]])

        if p > .05:
            # data normally distributed - do 1-samp t-test
            print('    1-samp t-test')
            stat, pval = ttest_1samp(CDI[:, ixs[c, 0]] - CDI[:, ixs[c, 1]],
                                     0,
                                     alternative='greater')
        else:
            # see if log-transform makes the distribution normal
            s1, p1 = shapiro(np.log(CDI[:, ixs[c, 0]] - CDI[:, ixs[c, 1]]))
            if p1 <= .05:
                # do 1-samp t-test
                print('    1-samp t-test,log-transformed data')
                stat, pval = ttest_1samp(np.log(CDI[:, ixs[c, 0]] - CDI[:, ixs[c, 1]]),
                                         0,
                                         alternative='greater')
            else:
                # wilcoxon test
                print('    wilcoxon test')
                stat, pval = wilcoxon(CDI[:, ixs[c, 0]] - CDI[:, ixs[c, 1]], alternative='greater')
        print('    stat = %.2f, p = %.3f' % (stat, pval))
    return


def plot_uncued_subspaces(constants, uncued_subspaces, trial_type_subspaces):
    # uncued post-cue
    plot_geom_and_subspace(constants, uncued_subspaces['delay2'], 'uncued post-cue ',
                           custom_labels=['L1 uncued', 'L2 uncued'])
    plt.savefig(constants.PARAMS['FIG_PATH'] + 'uncued_post-cue_geometry.png')

    # cued-up trials
    plot_geom_and_subspace(constants, trial_type_subspaces['cued_up'],
                           'cued-up trials',
                           custom_labels=['L1 cued', 'L2 uncued'])
    plt.savefig(constants.PARAMS['FIG_PATH'] + 'cued_up_uncued_down_geometry.png')

    # cued-down trials
    plot_geom_and_subspace(constants, trial_type_subspaces['cued_down'],
                           'cued-down trials',
                           custom_labels=['L1 uncued', 'L2 cued'])

    plt.savefig(constants.PARAMS['FIG_PATH'] + 'cued_down_uncued_up_geometry.png')


# def run_uncued_analysis(constants):
#     # need to add the hyperalignment part here at some point
#     # uncued_subspaces = get_averaged_uncued_subspaces(constants)
#     trial_type_subspaces = get_averaged_trial_type_subspaces(constants)
#
#     plot_uncued_subspaces(constants, uncued_subspaces, trial_type_subspaces)
#
#     CDI = get_CDI(constants, trials='invalid')
#     plot_CDI(constants, CDI, save_fig=False)
#
#     return uncued_subspaces, trial_type_subspaces


# Cued geometry

def get_cued_geometry(constants, trial_type='valid', save_data=False):
    """
    Get the Cued item geometry for each trained model.

    :param constants: dict
    :param trial_type: str, optional. Choose between 'valid' and 'invalid' (only in experiment 4 - probabilistic
        retrocues). Default is 'valid'.
    :return:
    """

    load_path = constants.PARAMS['FULL_PATH'] + 'pca_data/' + trial_type + '_trials/'
    if constants.PARAMS['experiment_number'] == 4:
        n_timepoints = 3
    else:
        n_timepoints = 2

    theta = np.empty((constants.PARAMS['n_models'], n_timepoints))
    PVEs = np.empty((constants.PARAMS['n_models'], n_timepoints, 3))
    psi = np.empty((constants.PARAMS['n_models'], n_timepoints))

    for model in range(constants.PARAMS['n_models']):
        # load pre- and post-cue delay data
        with open(load_path + 'pca_data_model' + str(model) + '.pckl', 'rb') as f:
            pca_data = pickle.load(f)

        delay1 = pca_data['delay1']
        delay2 = pca_data['delay2']

        # get the Cued geometry in each delay
        delay1_geometry = subspace.Geometry(delay1, constants, ['cued_up', 'cued_down'])
        delay2_geometry = subspace.Geometry(delay2, constants, ['cued_up', 'cued_down'])

        delay1_geometry.get_geometry()
        delay2_geometry.get_geometry()

        cued_geometry = {'delay1': delay1_geometry, 'delay2': delay2_geometry}

        theta[model, 0] = delay1_geometry.theta_degrees
        theta[model, 1] = delay2_geometry.theta_degrees

        PVEs[model, 0, :] = delay1_geometry.PVEs
        PVEs[model, 1, :] = delay2_geometry.PVEs

        psi[model, 0] = delay1_geometry.psi_degrees
        psi[model, 1] = delay2_geometry.psi_degrees

        if constants.PARAMS['experiment_number'] == 4:
            probe = pca_data['data'][:, -1, :]
            probe_geometry = subspace.Geometry(probe, constants, ['cued_up', 'cued_down'])
            probe_geometry.get_geometry()

            cued_geometry['probe'] = probe_geometry
            theta[model, 2] = probe_geometry.theta_degrees
            PVEs[model, 2, :] = probe_geometry.PVEs
            psi[model, 2] = probe_geometry.psi_degrees


        if save_data:
            pickle.dump(cued_geometry,
                        open(load_path + 'cued_subspaces_model' + str(model) + '.pckl', 'wb'))
    if save_data:
        pickle.dump(theta,
                    open(load_path + 'all_theta.pckl', 'wb'))
        pickle.dump(PVEs,
                    open(load_path + 'all_PVEs_3D.pckl', 'wb'))
        pickle.dump(psi,
                    open(load_path + 'all_psi.pckl', 'wb'))
    return theta, psi, PVEs

def get_trial_type_subspaces(constants, trial_type='valid'):
    # cued uncued geometry
    n_colours = constants.PARAMS['B']
    load_path = constants.PARAMS['FULL_PATH'] + 'pca_data/' + trial_type + '_trials/'

    theta = np.empty((constants.PARAMS['n_models'], 2))
    psi = np.empty((constants.PARAMS['n_models'], 2))

    for model in range(constants.PARAMS['n_models']):
        # load data
        path_cued = load_path + 'pca_data_model' + str(model) + '.pckl'
        path_uncued = load_path + 'pca_data_uncued_model' + str(model) + '.pckl'

        data_binned_cued = pickle.load(open(path_cued, 'rb'))
        data_binned_uncued = pickle.load(open(path_uncued, 'rb'))

        # 'cued-up'
        delay2_up_trials = torch.cat((data_binned_cued['delay2'][:n_colours, :],
                                      data_binned_uncued['delay2'][:n_colours, :]))
        # 'cued-down'
        delay2_down_trials = torch.cat((data_binned_cued['delay2'][n_colours:, :],
                                        data_binned_uncued['delay2'][n_colours:, :]))

        # calculate the geometry
        delay2_up_trials_subspace = subspace.Geometry(delay2_up_trials, constants, ['cued_up', 'uncued_down'])
        delay2_down_trials_subspace = subspace.Geometry(delay2_down_trials, constants, ['cued_down', 'uncued_up'])

        delay2_up_trials_subspace.get_geometry()
        delay2_down_trials_subspace.get_geometry()

        # delay2_up_trials_subspace = \
        #     run_pca_pipeline(constants,
        #                      delay2_up_trials,
        #                      ['cued_up', 'uncued_down'])
        # delay2_down_trials_subspace = \
        #     run_pca_pipeline(constants,
        #                      delay2_down_trials,
        #                      ['cued_down', 'uncued_up'])

        # save subspaces
        trial_type_subspaces = {'cued_up': delay2_up_trials_subspace, 'cued_down': delay2_down_trials_subspace}

        # save angles
        theta[model, 0] = delay2_up_trials_subspace.theta_degrees
        theta[model, 1] = delay2_down_trials_subspace.theta_degrees

        psi[model, 0] = delay2_up_trials_subspace.psi_degrees
        psi[model, 1] = delay2_down_trials_subspace.psi_degrees

        pickle.dump(trial_type_subspaces,
                    open(load_path + 'trial_type_subspaces_model' + str(model) + '.pckl', 'wb'))

    pickle.dump(theta,
                open(load_path + 'cued_vs_uncued_theta.pckl', 'wb'))
    pickle.dump(psi,
                open(load_path + 'cued_vs_uncued_psi.pckl', 'wb'))


def get_uncued_subspaces(constants, trial_type='valid'):
    load_path = constants.PARAMS['FULL_PATH'] + 'pca_data/' + trial_type + '_trials/'

    all_plane_angles = []
    psi = []
    for model in range(constants.PARAMS['n_models']):
        # load data
        model_number = str(model)
        f = open(load_path + '/pca_data_uncued_model' + model_number + '.pckl', 'rb')
        pca_data_uncued = pickle.load(f)
        delay2 = pca_data_uncued['delay2']
        f.close()

        # calculate the geometry
        delay2_uncued_subspace = subspace.Geometry(delay2,constants,['uncued_down', 'uncued_up'])
        delay2_uncued_subspace.get_geometry()
        # delay2_uncued_subspace = \
        #     run_pca_pipeline(constants,
        #                      delay2,
        #                      ['uncued_down', 'uncued_up'])


        psi.append(delay2_uncued_subspace.psi_degrees)

        pickle.dump(delay2_uncued_subspace,
                    open(load_path + 'delay2_uncued_subspace_model' + str(model) + '.pckl', 'wb'))

        all_plane_angles.append(delay2_uncued_subspace.theta_degrees)

    pickle.dump(all_plane_angles,
                open(load_path + 'all_theta_uncued_post-cue.pckl', 'wb'))
    pickle.dump(psi,
                open(load_path + 'all_psi_uncued_post-cue.pckl', 'wb'))






# def run_2step_pca(constants, plot=False):
#     '''
#     run the pca pipeline for individual models, to get the angles etc
#
#     Parameters
#     ----------
#     constants : TYPE
#         DESCRIPTION.
#     plot : TYPE, optional
#         DESCRIPTION. The default is False.
#
#     Returns
#     -------
#     None.
#
#     '''
#     load_path = constants.PARAMS['FULL_PATH'] + 'pca_data/valid_trials/'
#     n_colours = constants.PARAMS['B']
#
#     # check that paths to save data exist
#     # helpers.check_path(load_path + '/angles/')
#     # helpers.check_path(load_path + '/pca3/')
#     # helpers.check_path(load_path + '/pca2/')
#     # helpers.check_path(load_path + '/planes/')
#
#     if constants.PARAMS['experiment_number'] == 4:
#         n_timepoints = 3
#     else:
#         n_timepoints = 2
#
#     all_plane_angles = np.empty((constants.PARAMS['n_models'], n_timepoints))
#     all_PVEs_3D = np.empty((constants.PARAMS['n_models'], n_timepoints, 3))
#     for model in range(constants.PARAMS['n_models']):
#         # load pre- and post-cue delay data
#         f = open(load_path + 'pca_data_model' + str(model) + '.pckl', 'rb')
#         pca_data = pickle.load(f)
#         f.close()
#         delay1 = pca_data['delay1']
#         delay2 = pca_data['delay2']
#
#         ### PCA 1
#
#         ### old code - delete
#         # # demean data
#         # delay1 -= delay1.mean()
#         # delay2 -= delay2.mean()
#
#         # #% run first PCA to get down to 3D space
#         # delay1_pca = PCA(n_components=3) # Initializes PCA
#         # delay2_pca = PCA(n_components=3) # Initializes PCA
#
#         # delay1_3Dcoords = delay1_pca.fit_transform(delay1) # get coordinates in the reduced-dim space
#         # delay2_3Dcoords = delay2_pca.fit_transform(delay2)
#
#         ###
#         delay1_subspace = run_pca_pipeline(constants, delay1, ['cued_up', 'cued_down'])
#         delay2_subspace = run_pca_pipeline(constants, delay2, ['cued_up', 'cued_down'])
#         all_plane_angles[model, 0] = delay1_subspace['theta']
#         all_plane_angles[model, 1] = delay2_subspace['theta']
#         all_PVEs_3D[model, 0, :] = delay1_subspace['pca'].explained_variance_ratio_
#         all_PVEs_3D[model, 1, :] = delay2_subspace['pca'].explained_variance_ratio_
#
#         if constants.PARAMS['experiment_number'] == 4:
#             probe = pca_data['data'][:, -1, :]
#             print('change this to a probe timepoint')
#             probe_subspace = run_pca_pipeline(constants, probe, ['cued_up', 'cued_down'])
#
#             all_plane_angles[model, 2] = probe_subspace['theta']
#             all_PVEs_3D[model, 2, :] = probe_subspace['pca'].explained_variance_ratio_
#
#         # if plot:
#         #     plt.figure(figsize=(12,5),num=('Model '+str(model)))
#         #     ax = plt.subplot(121, projection='3d')
#         #     plot_geometry(ax, delay1_3Dcoords, delay1_pca, constants.PLOT_PARAMS['4_colours'],legend_on=False)
#         #     plot_subspace(ax,delay1_3Dcoords[:n_colours,:],delay1_planeUp.components_,fc='k',a=0.2)
#         #     plot_subspace(ax,delay1_3Dcoords[n_colours:,:],delay1_planeDown.components_,fc='k',a=0.2)
#         #     ax.set_title('Angle: %.1f' %theta_pre)
#         #     helpers.equal_axes(ax)
#
#         #     ax2 = plt.subplot(122, projection='3d')
#         #     plot_geometry(ax2, delay2_3Dcoords, delay2_pca, constants.PLOT_PARAMS['4_colours'])
#         #     plot_subspace(ax2,delay2_3Dcoords[:n_colours,:],delay2_planeUp.components_,fc='k',a=0.2)
#         #     plot_subspace(ax2,delay2_3Dcoords[n_colours:,:],delay2_planeDown.components_,fc='k',a=0.2)
#         #     ax2.set_title('Angle: %.1f' %theta_post)
#         #     helpers.equal_axes(ax2)
#
#     pickle.dump(all_plane_angles, open(load_path + 'all_plane_angles.pckl', 'wb'))
#     pickle.dump(all_PVEs_3D, open(load_path + 'all_PVEs_3D.pckl', 'wb'))
#
#





    #
    # # get_model_RDMs(constants)
    # # # RDM analysis
    # # rdm_precue, rdm_postcue, pre_data_RDM_averaged, post_data_RDM_averaged = get_data_RDMs(constants)
    # # # plot_full_data_RDMs(rdm_precue,rdm_postcue)
    # # plot_binned_data_RDMs(pre_data_RDM_averaged,post_data_RDM_averaged)
    # # results_pre, results_post = run_main_RDM_reg(constants,pre_data_RDM_averaged,post_data_RDM_averaged)
    # # plot_main_RDM_reg_results(constants,results_pre,results_post)
    # # pre_ortho,pre_parallel,post_ortho,post_parallel = \
    # #     run_rotation_RDM_reg(constants,pre_data_RDM_averaged,post_data_RDM_averaged)
    # # plot_rot_RDM_reg(constants,pre_ortho,pre_parallel,post_ortho,post_parallel)
    # # run_mirror_im_RDM_reg(constants,post_data_RDM_averaged)
    #
    # # mds_precue, mds_postcue = get_MDS_from_RDMs(pre_data_RDM_averaged, post_data_RDM_averaged)
    # # plot_data_MDS(mds_precue, mds_postcue, constants)
    # # model_av_binned_cued, model_av_binned_uncued = get_model_averages(constants)
    # # plot_averaged_cued_geometry(constants)
    # # run_2step_pca(constants) # to get angle vals for individual models
    #
    # _ = get_cued_geometry(constants)
    # # plot_cued_subspaces_indivModels(constants)
    # uncued_subspaces, trial_type_subspaces = run_uncued_analysis(constants)
    #
    # # get other angles
    # get_trial_type_subspaces(constants)
    # get_uncued_subspaces(constants, trial_type='valid')
    # #
    # # uncued_pre = pickle.load(open(constants.PARAMS['FULL_PATH'] + 'RSA/' + 'pre_data_uncued_RDM_averaged.pckl', 'rb'))
    # # uncued = pickle.load(open(constants.PARAMS['FULL_PATH'] + 'RSA/' + 'post_data_uncued_RDM_averaged.pckl', 'rb'))
    # # cued_uncued = pickle.load(open(constants.PARAMS['FULL_PATH'] + 'RSA/' + 'post_data_cu_RDM_averaged.pckl', 'rb'))
    #
    # # mds_uncued = fit_mds_to_rdm(uncued)
    # # mds_uncued_pre = fit_mds_to_rdm(uncued_pre)
    # # mds_cu_L1 = fit_mds_to_rdm(cued_uncued[:, :, 0])
    # # mds_cu_L2 = fit_mds_to_rdm(cued_uncued[:, :, 1])
    # #
    # # ix = np.concatenate((np.arange(4, 8), np.arange(4)))
    # # plot_data_MDS(mds_cu_L1, mds_cu_L2[ix, :], constants)
    # # plot_data_MDS(mds_uncued_pre, mds_uncued, constants)


# %%

# def get_angle(constants):
#     load_path = constants.PARAMS['FULL_PATH'] + 'pca_data/valid_trials/'  # 'partial_training/'
#     delay_len = constants.PARAMS['trial_timings']['delay1_dur']
#
#     theta_post = np.empty((constants.PARAMS['n_models'], delay_len))
#
#     d2_start = constants.PARAMS['trial_timepoints']['delay2_start'] - 1
#     # loop over models
#     for model in range(constants.PARAMS['n_models']):
#         # % load data
#         # fully trained
#         pca_data_ft = pickle.load(open(load_path + 'pca_data_model' + str(model) + '.pckl', 'rb'))
#
#         # run the PCA pipeline on both delays, separately for each timepoint
#         for t in range(delay_len):
#             subspace_d2 = run_pca_pipeline(constants,
#                                            pca_data_ft['data'][:, d2_start + t, :],
#                                            ['up', 'down'])
#             theta_post[model, t] = subspace_d2['theta']
#
#     return theta_post


# check the cued-uncued angles across the entire post-cue delay (and retrocue)

# def get_trial_type_subspaces_indivModels(constants, trial_type='valid'):
#     n_colours = constants.PARAMS['B']
#     load_path = constants.PARAMS['FULL_PATH'] + 'pca_data/' + trial_type + '_trials/'
#
#     theta = np.empty((constants.PARAMS['n_models'], 2))
#     psi = np.empty((constants.PARAMS['n_models'], 2))
#
#     for model in range(constants.PARAMS['n_models']):
#         # load data
#         path_cued = load_path + 'pca_data_model' + str(model) + '.pckl'
#         path_uncued = load_path + 'pca_data_uncued_model' + str(model) + '.pckl'
#
#         data_binned_cued = pickle.load(open(path_cued, 'rb'))
#         data_binned_uncued = pickle.load(open(path_uncued, 'rb'))
#
#         # 'cued-up'
#         delay2_up_trials = torch.cat((data_binned_cued['delay2'][:n_colours, :],
#                                       data_binned_uncued['delay2'][:n_colours, :]))
#         # 'cued-down'
#         delay2_down_trials = torch.cat((data_binned_cued['delay2'][n_colours:, :],
#                                         data_binned_uncued['delay2'][n_colours:, :]))
#
#         delay2_up_trials_subspace = \
#             run_pca_pipeline(constants,
#                              delay2_up_trials,
#                              ['cued_up', 'uncued_down'])
#         delay2_down_trials_subspace = \
#             run_pca_pipeline(constants,
#                              delay2_down_trials,
#                              ['cued_down', 'uncued_up'])
#
#         trial_type_subspaces = {}
#         trial_type_subspaces['cued_up'] = delay2_up_trials_subspace
#         trial_type_subspaces['cued_down'] = delay2_down_trials_subspace
#
#         # save angles
#         theta[model, 0] = delay2_up_trials_subspace['theta']
#         theta[model, 1] = delay2_down_trials_subspace['theta']
#
#         psi[model, 0] = delay2_up_trials_subspace['psi']
#         psi[model, 1] = delay2_down_trials_subspace['psi']
#
#         pickle.dump(trial_type_subspaces,
#                     open(load_path + 'trial_type_subspaces_model' + str(model) + '.pckl', 'wb'))
#
#     pickle.dump(theta,
#                 open(load_path + 'cued_vs_uncued_theta.pckl', 'wb'))
#     pickle.dump(psi,
#                 open(load_path + 'cued_vs_uncued_psi.pckl', 'wb'))




##

