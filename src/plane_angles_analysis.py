#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 10:34:57 2020

@author: emilia

requires the pycircstat toolbox, available at https://github.com/circstat/pycircstat
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import src.vec_operations as vops
import pickle
import pycircstat
# import constants
import seaborn as sns
import pdb
import src.helpers as helpers
import src.custom_plot as cplot


# %% plots


#     if constants.PARAMS['experiment_number'] == 3:
#         inds = [3,0,-3]
#         markers = ['o','^','s']
#         labels = ['pre-cue','post-cue','post-probe']
#         n_subplots = 1
#     else:
#         if cond=='cued_vs_uncued':
#             markers = ['^','s']
#             labels = ['L1 cued','L2 cued']
#             cols = ['k','k']
#             alpha1, alpha2 = .2, .4
#             n_subplots = 1
#         elif cond == 'pre_vs_post':
#             inds = [3,0]
#             markers = ['o','^']
#             labels = ['pre','post']
#             n_subplots = 1
#         elif cond == 'pre':
#             inds = [3]
#             markers = ['o']
#             n_subplots = 1
#         elif cond == 'post':
#             inds = [0]
#             markers = ['^']
#             n_subplots = 1
#         elif cond == 'phase_alignment':
#             inds = [3,0]
#             markers = ['o','^']
#             n_subplots = 2


#     fig = plt.figure(figsize=(7.9,5))
#     for n in range(n_subplots):
#         ax = fig.add_subplot(1,n_subplots,n+1,polar=True)
#         if len(angles_radians.shape) > 1:
#             plot_polar_scatter(ax,angles_radians[:,n],r,**kwargs)


# def plot_plane_angles(constants,angles_radians,r=None,cued_vs_uncued=False,add_legend=True):
#     pal = sns.color_palette("dark")
#     if constants.PARAMS['experiment_number'] == 3:
#         inds = [3,0,-3]
#         markers = ['o','^','s']
#         labels = ['pre-cue','post-cue','post-probe']
#     else:
#         if cued_vs_uncued:
#             markers = ['^','s']
#             labels = ['L1 cued','L2 cued']
#             cols = ['k','k']
#             alpha1, alpha2 = .2, .4
#         else:
#             inds = [3,0]
#             markers = ['o','^']
#             labels = ['pre','post']

#     fig = plt.figure(figsize=(7.9,5))

#     if np.all(r):
#         # if r == None
#         r = np.ones((constants.PARAMS['n_models'],))

#     rr=1 # radius value for the mean
#     if cued_vs_uncued:
#         # fig = plt.figure(figsize=(16,5))
#         # ax1 = fig.add_subplot(121,polar=True)
#         # ax2 = fig.add_subplot(122,polar=True)

#         ax1 = fig.add_subplot(111,polar=True)
#         ax1.grid(False)
#         # ax1.set_thetamin(0)
#         # ax1.set_thetamax(180)
#         # ax1.set_xticks(np.radians(np.linspace(0,180,7)))
#         # ax2.grid(False)
#         ms = 12

#         # ax1.plot(angles_radians[:,0],np.ones((constants.PARAMS['n_models'],))*r,
#         #          marker=markers[0],linestyle='',
#         #         color = cols[0],alpha=alpha1,markersize=ms)
#         # ax2.plot(angles_radians[:,1],np.ones((constants.PARAMS['n_models'],))*r,
#         #          marker=markers[1],linestyle='',
#         #         color = cols[1],alpha=alpha2,markersize=ms)

#         ax1.plot(angles_radians,r,
#                   marker='o',linestyle='',
#                 color = cols[0],alpha=alpha1,markersize=ms)

#         # add means
#         # ax1.plot(pycircstat.descriptive.mean(angles_radians[:,0]),r,
#         #          marker=markers[0],c = cols[0],markersize=ms)
#         # ax2.plot(pycircstat.descriptive.mean(angles_radians[:,1]),r,
#         #          marker=markers[1],c = cols[1],markersize=ms)

#         ax1.plot(pycircstat.descriptive.mean(angles_radians),rr,
#                   marker='o',c = cols[0],markersize=ms)

#         ax1.set_ylim([0,1.05])
#         ax1.set_yticks([])
#         ax1.tick_params(axis='x', which='major', pad=14)
#         plt.tight_layout()


#         # ax2.set_ylim([0,1.05])
#         # ax2.set_yticks([])

#         # ax1.set_title('L1 cued')
#         # ax2.set_title('L2 cued')
#         return

#     cols = [pal[ix] for ix in inds]

#     ax = fig.add_subplot(111,polar=True)
#     ax.grid(False)
#     # ax.set_thetamin(0)
#     # ax.set_thetamax(180)
#     # ax.set_xticks(np.radians(np.linspace(0,180,7)))
#     ms = 12

#     for model in range(len(angles_radians)):        
#         if len(angles_radians.shape)<2:
#             # plot only post-cue
#             ax.plot(angles_radians[model],r[model],'o',color = cols[1],alpha=0.2,markersize=ms)
#         else:
#             ax.plot(angles_radians[model,:],np.ones((len(inds),))*r[model],'k-',alpha=0.2)
#             for i in range(len(inds)):
#                 ax.plot(angles_radians[model,i],r[model],marker=markers[i],
#                         color = cols[i],alpha=0.2,markersize=ms)

#     if len(angles_radians.shape)<2:
#         ax.plot(pycircstat.descriptive.mean(angles_radians),rr,'o',c = cols[1],markersize=ms)
#     else:
#         for i in range(len(inds)):
#             ax.plot(pycircstat.descriptive.mean(angles_radians[:,i]),rr,
#                     marker=markers[i],c = cols[i],markersize=ms,label=labels[i])

#     #helpers.circ_mean(torch.tensor(angles_radians[:,1]))

#     if add_legend:
#         plt.legend(bbox_to_anchor=(.9, .9),
#                 bbox_transform=plt.gcf().transFigure)
#     plt.tight_layout()
#     # plt.legend()

#     #%
#     ax.tick_params(axis='x', which='major', pad=14)
#     ax.set_ylim([0,1.05])
#     ax.set_yticks([])


def plot_angles_retrocue_timing(constants):
    # plot the plane angles for the retrocue_timing condition

    pal = sns.color_palette("dark")

    inds = [3, 0]
    markers = ['o', '^']
    cols = [pal[ix] for ix in inds]
    ms = 10

    conditions = ['pre-cue', 'post-cue']
    delay2_lengths = np.arange(6)

    common_path = constants.PARAMS['BASE_PATH'] \
                  + 'data_vonMises/experiment_4/'

    n_models = 30
    all_angles = np.empty((n_models, len(conditions), len(delay2_lengths)))
    angles_circ_mean = np.empty((len(conditions), len(delay2_lengths)))

    plt.figure(figsize=(7, 5))
    ax1 = plt.subplot(111)
    ax1.set_thetamin(0)
    ax1.set_thetamax(180)
    ax1.set_xticks(np.radians(np.linspace(0, 180, 7)))
    jitter = .125
    # ax2 = plt.subplot(122,sharey=ax1,sharex=ax1) 
    for j, dl in enumerate(delay2_lengths):
        # load data
        load_path = common_path + 'delay2_' + str(dl) + 'cycles/' \
                    + 'sigma' + str(constants.PARAMS['sigma']) \
                    + '/kappa' + str(constants.PARAMS['kappa_val']) \
                    + '/nrec' + str(constants.PARAMS['n_rec']) \
                    + '/lr' + str(constants.PARAMS['learning_rate']) + '/' \
                    + 'pca_data/valid_trials/'
        all_angles[:, :, j] = pickle.load(open(load_path + 'all_plane_angles.pckl', 'rb'))
        angles_circ_mean[:, j] = pycircstat.descriptive.mean(np.radians(all_angles[:, :, j]), axis=0)
        angles_circ_mean[:, j] = np.degrees(angles_circ_mean[:, j])

        ax1.plot(np.ones((n_models,)) * dl - jitter, all_angles[:, 0, j],
                 marker=markers[0], ls='', color=cols[0], alpha=.2, markersize=ms)

        ax1.plot(np.ones((n_models,)) * dl + jitter, all_angles[:, 1, j],
                 marker=markers[1], ls='', color=cols[1], alpha=.2, markersize=ms)

    # add means
    ax1.plot(delay2_lengths - jitter, angles_circ_mean[0, :],
             marker=markers[0], color=cols[0], markersize=ms, label='pre-cue')
    ax1.plot(delay2_lengths + jitter, angles_circ_mean[1, :],
             marker=markers[1], color=cols[1], markersize=ms, label='post-cue')

    ax1.set_ylim((all_angles.min() - ms), all_angles.max() + ms)
    ax1.set_xticks(delay2_lengths)

    ax1.set_ylabel('Cued plane angle [°]')
    ax1.set_xlabel('Post-cue delay length [cycles]')

    ax1.legend(bbox_to_anchor=(.9, .9))
    # ax1.set_title('Pre-cue')
    # ax2.set_title('Post-cue')

    plt.tight_layout()

    plt.savefig(common_path + 'compare_cued_angles_sigma' + str(constants.PARAMS['sigma']) + '.png')


# %% circular stats for theta

def print_mean_and_ci_angle(angles_radians, angle_name, geometry_name):
    """
    Print the circular mean and 95% CI of angles, in degrees.

    :param angles_radians: (n_models, n_delays) Theta angle values in radians
    :type angles_radians: np.ndarray
    :param angle_name: Angle name, choose from 'theta' and 'psi'.
    :type angle_name: str
    :param geometry_name: Name of geometry described by the angles.
    :type geometry_name: str
    """
    n_delays = angles_radians.shape[1]
    delay_names = ['pre-cue', 'post-cue', 'post-probe']
    for delay in range(n_delays):
        # filter out NaNs if there are any
        nonnan_ix = np.where(np.invert(np.isnan(angles_radians[:, delay])))[0]
        pct_nans = (1 - (len(nonnan_ix) / len(angles_radians))) * 100

        mean_angle = pycircstat.descriptive.mean(angles_radians[nonnan_ix, delay])
        mean_angle = helpers.wrap_angle(mean_angle)  # wrap to [-pi,pi]
        mean_angle = np.degrees(mean_angle)

        ci = pycircstat.descriptive.mean_ci_limits(angles_radians[nonnan_ix, delay], ci=.95)
        ci = np.degrees(ci)

        print(f"Angle {angle_name} for {geometry_name} geometry, {delay_names[delay]} delay: "
              f"mean = %.2f, 95CI = %.2f degrees, percent NaN values: %.1f" % (mean_angle, ci, pct_nans))


def get_inf_stats_theta_cued(angles_radians):
    """
    Run and print the results of the inferential statistics on the distribution of angles theta for the Cued geometry.

    Runs 3 tests:
        1) V-test to test whether the pre-cue angles are clustered with a mean=90 degrees
        2) V-test to test whether the post-cue angles are clustered with a mean=0 degrees
        3) Circular one-sample t-test to test whether the angular pre-cue - post-cue difference is ~=0

    Additionally, for experiment 4, it also runs the following contrast:
        4) V-test to test whether the post-probe angles are clustered with a mean=0 degrees

    :param angles_radians: (n_models, n_delays) Theta cued angle values in radians
    :type angles_radians: np.ndarray
    """
    # plane angles are defined on the [-pi, pi] interval

    # pre-cue angles
    # test for non-uniform distribution around 90
    # note: data is bimodal, with peaks at + and -90 degrees
    # v-test assumes a unimodal or uniform distribution over the *entire* circle
    # as the sign of the angle does not matter for this contrast, we transform
    # the data first by taking the absolute value (to force it to be unimodal),
    # and then by multiplying by 2 (to extend the range to [0,360])
    p_pre, v_pre = pycircstat.tests.vtest(np.abs(angles_radians[:, 0]) * 2, np.radians(90) * 2)
    print('Pre-cue angles theta: V-test for uniformity/mean=90 :')
    print('    v-stat = %.3f, p = %.3f' % (v_pre, p_pre))

    # post-cue angles
    # test for non-uniform distribution around 0 - here the distribution of the
    # data is unimodal, therefore we do not transform it 
    p_post, v_post = pycircstat.tests.vtest(angles_radians[:, 1], 0)
    print('Post-cue angles theta: V-test for uniformity/mean=0 :')
    print('    v-stat = %.3f, p = %.3f' % (v_post, p_post))

    # test for a significant difference in angles
    # here again we have a bimodal distribution
    # to make the data appropriate for the mtest, we first take the absolute of
    # the pre-post difference, and then stretch it to the [0,360] range by 
    # multiplying by 2
    angle_diff_signed = angles_radians[:, 0] - angles_radians[:, 1]
    angle_diff = np.abs(angle_diff_signed) * 2
    diff = pycircstat.tests.mtest(angle_diff, 0)
    diff_mean = np.degrees(diff[1] / 2)  # divide by 2 to go back to original range
    diff_result = diff[0]
    diff_CI = ((np.degrees(diff[2][1]) - np.degrees(diff[2][0])) / 2) / 2  # same here
    # diff_SEM = (np.diff(diff_CI)/2)/1.96

    print('Pre- vs post-cue Cued angles theta: circular one-sample t-test for angular difference ~=0 :')
    print('     H = %d, mean = %.3f degrees, CI = %.3f degrees' % (diff_result[0], diff_mean, diff_CI))

    if angles_radians.shape[1] == 3:
        # experiment 4
        nonnan_ix = np.where(np.invert(np.isnan(angles_radians[:, 2])))[0]
        p_probe, v_probe = pycircstat.tests.vtest(angles_radians[nonnan_ix, 2], 0)
        print('Post-probe angles: V-test for uniformity/mean=0 :')
        print('    v-stat = %.3f, p = %.3f' % (v_probe, p_probe))


# def get_inf_stats_theta_expt4(angles_radians):
#     # pre-cue angles
#     # test for non-uniform distribution around 90
#     p_pre, v_pre = pycircstat.tests.vtest(np.abs(angles_radians[:, 0])*2, np.radians(90)*2)
#     print('V-test for uniformity/mean=90 :')
#     print('    v-stat = %.3f, p = %.3f' %(v_pre,p_pre))
#
#     p_post, v_post = pycircstat.tests.vtest(angles_radians[:, 1], 0)
#     print('V-test for uniformity/mean=0 :')
#     print('    v-stat = %.3f, p = %.3f' %(v_post, p_post))
#
#     nonnan_ix = np.where(np.invert(np.isnan(angles_radians[:,2])))[0]
#     p_probe, v_probe = pycircstat.tests.vtest(angles_radians[nonnan_ix,2], 0)
#     print('V-test for uniformity/mean=0 :')
#     print('    v-stat = %.3f, p = %.3f' %(v_probe,p_probe))


def get_inf_stats_theta_uncued(angles_radians):
    """
    Run and print the results of a Rayleigh test on the distribution of angles theta for the Uncued geometry.

    :param angles_radians: (n_models) Theta uncued angle values for the post-cue delay in radians
    :type angles_radians: np.ndarray
    """
    if len(angles_radians.shape) > 1:
        raise NotImplementedError('Analysis not implemented for Experiment 4, assumes angles_radians is one-dimensional'
                                  ' with a single value for each model.')

    p_val, z_stat = pycircstat.tests.rayleigh(angles_radians)
    print('Rayleigh test for uniformity of uncued post-cue angles theta:')
    print('    z-stat = %.3f, p = %.3f' % (z_stat, p_val))


def get_inf_stats_theta_cued_vs_uncued(angles_radians):
    """
    Run and print the results of a v-test on the distribution of theta angles for the Cued-Uncued geometry,
    to test whether the angles are clustered with a mean = 90. If the test comes back non-significant, runs a follow-up
    Rayleigh test to check for significant clustering and prints the circular mean.

    :param angles_radians: (n_models) Theta cued-uncued angle values in radians
    :type angles_radians: np.ndarray
    """
    if len(angles_radians.shape) > 1:
        raise NotImplementedError(
            'Analysis not implemented for Experiment 4, assumes angles_radians is one-dimensional '
            'with a single value for each model.')
    # need to stretch the data to the [0,360] range by multiplying by 2, then test for significant clustering at
    # 90 degrees
    p, v = pycircstat.tests.vtest(angles_radians * 2, np.pi)
    print('Cued-uncued theta: V-test for uniformity/mean=90 :')
    print('    v-stat = %.3f, p = %.3f' % (v, p))

    if p >= .05:
        # follow up with a Rayleigh test + mean
        p_val, z_stat = pycircstat.tests.rayleigh(angles_radians * 2)
        m = np.degrees(pycircstat.mean(angles_radians))
        print('Rayleigh test for uniformity of cued/uncued theta:')
        print('    z-stat = %.3f, p = %.3f, mean = %.3f' % (z_stat, p_val, m))


# %% circular stats - psi

# psi cued
# def get_descriptive_stats_psi_cued(psi_radians):
#     """
#
#     :param psi_radians:
#     :return:
#     """
#
#     n_delays = psi_radians.shape[1]
#     for delay in n_delays()
#
#     nonnan_ix = np.where(np.invert(np.isnan(psi_radians[:, 0])))[0]
#     mean_pre = pycircstat.descriptive.mean(psi_radians[nonnan_ix, 0])
#     mean_pre = helpers.wrap_angle(mean_pre)  # wrap to [-pi,pi]
#     # ci_pre = pycircstat.descriptive.mean_ci_limits(pa_radians[nonnan_ix,0], ci=.95)
#
#     nonnan_ix = np.where(np.invert(np.isnan(psi_radians[:, 1])))[0]
#     mean_post = pycircstat.descriptive.mean(psi_radians[nonnan_ix, 1])
#     mean_post = helpers.wrap_angle(mean_post)
#     ci_post = pycircstat.descriptive.mean_ci_limits(psi_radians[nonnan_ix, 1], ci=.95)
#
#     # print('Pa: pre-cue mean ± 95 CI: %.2f ± %.2f' %(np.degrees(mean_pre),np.degrees(ci_pre)))
#     print('Pa: pre-cue mean: %.2f ' % (np.degrees(mean_pre)))
#     print('Pa : post-cue mean ± 95 CI: %.2f ± %.2f' % (np.degrees(mean_post), np.degrees(ci_post)))


def get_inf_stats_psi_cued(psi_radians):
    """
    Run and print the results of the inferential statistics on the distribution of angles psi for the Cued geometry.

    Runs 2 tests:
        1) Rayleigh test to test whether the pre-cue angles are significantly clustered
        2) V-test to test whether the post-cue angles are clustered with a mean=0 degrees

    Additionally, for experiment 4, it also runs the following contrast:
    3) V-test to test whether the post-probe angles are clustered with a mean=0 degrees

    :param psi_radians: (n_models, n_delays) Psi cued angle values in radians
    :type psi_radians: np.ndarray
    """
    # test pre-cue angles - no specific hypothesis
    # need to remove nans first
    non_nan_ix = np.where(np.invert(np.isnan(psi_radians[:, 0])))[0]
    p_val, z_stat = pycircstat.tests.rayleigh(psi_radians[non_nan_ix, 0])
    print('Cued geometry angles psi in the pre-cue delay: Rayleigh test for uniformity:')
    print('    z-stat = %.3f, p = %.3f, N = %d' % (z_stat, p_val, len(non_nan_ix)))

    # test post-cue angles - H1: clustered with mean of 0
    non_nan_ix = np.where(np.invert(np.isnan(psi_radians[:, 1])))[0]
    p, v = pycircstat.tests.vtest(psi_radians[non_nan_ix, 1], 0)

    print('Cued geometry angles psi in the post-cue delay: V-test for uniformity/mean=0:')
    print('    v-stat = %.3f, p = %.3f, N = %d' % (v, p, len(non_nan_ix)))

    # test post-probe angles, if applicable
    if psi_radians.shape[1] == 3:
        non_nan_ix = np.where(np.invert(np.isnan(psi_radians[:, 2])))[0]
        p, v = pycircstat.tests.vtest(psi_radians[non_nan_ix, 2], 0)

        print('Cued geometry angles psi in the post-probe delay: V-test for uniformity/mean=0:')
        print('    v-stat = %.3f, p = %.3f, N = %d' % (v, p, len(non_nan_ix)))


def get_inf_stats_psi_other(psi_radians, geometry_name):
    """
    Run and print the results of a v-test on the distribution of psi angles for the Uncued or Cued-Uncued geometry,
    to test whether the angles are clustered with a mean = 0.

    :param psi_radians: (n_models) Psi angle values in radians
    :type pa_radians: np.ndarray
    :param geometry_name: name of the geometry. Pick from 'Uncued' and 'Cued-Uncued'
    :type geometry_name: str
=    """
    if geometry_name == 'cued':
        raise ValueError("For the Cued geometry, use the 'get_inf_stats_psi_cued' function.")

    if len(psi_radians.shape) > 1:
        raise NotImplementedError(
            'Analysis not implemented for Experiment 4, assumes psi_radians is one-dimensional '
            'with a single value for each model.')
    # test uncued phase alignment in the post-cue delay
    # need to remove nans first
    non_nan_ix = np.where(np.invert(np.isnan(psi_radians)))[0]
    p_val, v = pycircstat.tests.vtest(psi_radians[non_nan_ix], 0)

    print(f'V test for {geometry_name} post-cue phase alignment:')
    print('    v-stat = %.3f, p = %.3f, N = %d' % (v, p_val, len(non_nan_ix)))


# %% runners
def run_theta_angles_analysis(constants, theta_degrees, geometry_name):
    # convert angles into radians
    theta_radians = np.radians(theta_degrees)

    # plot angles
    if geometry_name == 'cued':
        cplot.plot_plane_angles_multiple(constants, theta_radians)
        # plt.savefig(constants.PARAMS['FIG_PATH'] + 'plane_angles.svg')

        # get descriptive stats
        print_theta_mean(theta_radians, geometry_name)

        # get inferential stats
        get_inf_stats_theta_cued(theta_radians)

    elif geometry_name == 'uncued':
        pass


def run_psi_angles_analysis(constants, psi_degrees):
    psi_radians = np.radians(psi_degrees)

    cplot.plot_plane_angles_multiple(constants, psi_radians, paired=False)
    plt.savefig(constants.PARAMS['FIG_PATH'] + 'phase_align_cued.svg')

    # run stats
    get_descriptive_stats_psi_cued(psi_radians)
    if constants.PARAMS['experiment_number'] == 4:
        get_inf_stats_angles_expt4(psi_radians)
    else:
        get_inf_stats_pa_cued(psi_radians)
    return


def run_plane_angles_analysis(constants):
    # load data
    load_path = constants.PARAMS['FULL_PATH'] + 'pca_data/valid_trials/'
    angles = pickle.load(open(load_path + 'all_theta.pckl', 'rb'))
    angles_radians = np.radians(angles)
    PVEs_3D = pickle.load(open(load_path + 'all_PVEs_3D.pckl', 'rb'))

    # plot PVEs
    plot_PVEs_3D(constants, PVEs_3D)
    plt.savefig(constants.PARAMS['FIG_PATH'] + 'PVEs_3D.png')
    # plot angles
    plot_plane_angles_multiple(constants, angles_radians)
    plt.savefig(constants.PARAMS['FIG_PATH'] + 'plane_angles.svg')

    # get descriptive stats
    get_descriptive_stats_angles_cued(angles_radians)

    # get inferential stats
    get_inf_stats_angles_cued(angles_radians)


def run_plane_angles_analysis_uncued(constants):
    # load data
    load_path = constants.PARAMS['FULL_PATH'] + 'pca_data/valid_trials/'
    angles = pickle.load(open(load_path + 'all_theta_uncued_post-cue.pckl', 'rb'))
    angles_radians = np.radians(angles)

    # plot angles
    plot_plane_angles_single(constants, angles_radians, 'post')
    plt.savefig(constants.PARAMS['FIG_PATH'] + 'theta_uncued_postcue.svg')

    # run stats
    get_inf_stats_angles_uncued(angles_radians)


def run_plane_angles_analysis_cued_vs_uncued(constants):
    # load data
    load_path = constants.PARAMS['FULL_PATH'] + 'pca_data/valid_trials/'
    angles = pickle.load(open(load_path + 'cued_vs_uncued_theta.pckl', 'rb'))
    angles_radians = np.radians(angles)

    # get rectified mean
    angles_radians_rect_mean = pycircstat.mean(np.abs(angles_radians), axis=1)
    plot_plane_angles_single(constants, angles_radians_rect_mean, 'cu')
    plt.savefig(constants.PARAMS['FIG_PATH'] + 'theta_cued_vs_uncued.svg')
    plt.savefig(constants.PARAMS['FIG_PATH'] + 'theta_cued_vs_uncued.png')

    get_inf_stats_angles_cued_vs_uncued(angles_radians_rect_mean)


def run_phase_align_cued(constants):
    load_path = constants.PARAMS['FULL_PATH'] + 'pca_data/valid_trials/'
    pa = pickle.load(open(load_path + 'all_psi.pckl', 'rb'))
    pa_radians = np.radians(pa)

    plot_plane_angles_multiple(constants, pa_radians, paired=False)
    plt.savefig(constants.PARAMS['FIG_PATH'] + 'phase_align_cued.svg')
    # run stats
    get_descriptive_stats_pa_cued(pa_radians)
    if constants.PARAMS['experiment_number'] == 3:
        get_inf_stats_angles_expt3(pa_radians)
    else:
        get_inf_stats_pa_cued(pa_radians)


def run_phase_align_uncued(constants):
    load_path = constants.PARAMS['FULL_PATH'] + 'pca_data/valid_trials/'
    pa = pickle.load(open(load_path + 'all_psi_uncued_post-cue.pckl', 'rb'))
    pa_radians = np.radians(pa)

    plot_plane_angles_single(constants, pa_radians, cond='post')
    plt.savefig(constants.PARAMS['FIG_PATH'] + 'phase_align_uncued.svg')
    # run stats
    get_inf_stats_pa_uncued(pa_radians)


def run_phase_align_cued_vs_uncued(constants):
    load_path = constants.PARAMS['FULL_PATH'] + 'pca_data/valid_trials/'
    pa = pickle.load(open(load_path + 'cued_vs_uncued_psi.pckl', 'rb'))
    pa_radians = np.radians(pa)

    pa_radians_rect_mean = pycircstat.mean(np.abs(pa_radians), axis=1)
    # plot_plane_angles_multiple(constants,pa_radians,paired=True)
    plot_plane_angles_single(constants,
                             pa_radians_rect_mean, cond='cu')
    plt.savefig(constants.PARAMS['FIG_PATH'] + 'phase_align_cued_uncued.svg')
    # run stats
    get_inf_stats_pa_cued_vs_uncued(pycircstat.mean(pa_radians, axis=1))


def run_plane_angles_analysis_expt3(constants):
    # load data
    load_path = constants.PARAMS['FULL_PATH'] + 'pca_data/valid_trials/'
    angles = pickle.load(open(load_path + 'all_theta.pckl', 'rb'))
    angles_radians = np.radians(angles)
    PVEs_3D = pickle.load(open(load_path + 'all_PVEs_3D.pckl', 'rb'))

    # plot PVEs
    plot_PVEs_3D(constants, PVEs_3D)
    plt.savefig(constants.PARAMS['FIG_PATH'] + 'PVEs_3D.png')
    # plot angles
    plot_plane_angles_multiple(constants, angles_radians, paired=False, expt3=True)
    plt.savefig(constants.PARAMS['FIG_PATH'] + 'plane_angles.svg')

    # get inferential stats
    get_inf_stats_angles_expt3(angles_radians)

    ## add PA


def run_plane_angles_analysis_rotated_unrotated(constants):
    load_path = constants.PARAMS['FULL_PATH'] + 'pca_data/valid_trials/'
    theta_unrotated = pickle.load(open(load_path + 'theta_unrotated_plane.pckl', 'rb'))
    theta_rotated = pickle.load(open(load_path + 'theta_rotated_plane.pckl', 'rb'))
    psi_unrotated = pickle.load(open(load_path + 'psi_unrotated_plane.pckl', 'rb'))
    psi_rotated = pickle.load(open(load_path + 'psi_rotated_plane.pckl', 'rb'))

    # plot theta unroated and rotated
    theta_angles_radians = np.radians(np.stack((theta_unrotated, theta_rotated), 1))
    plot_plane_angles_multiple(constants, theta_angles_radians, r=None, paired=False, cu=True, expt3=False,
                               custom_labels=['unrotated', 'rotated'])

    plt.savefig(constants.PARAMS['FIG_PATH'] + 'theta_rotunrot.png')
    plt.savefig(constants.PARAMS['FIG_PATH'] + 'theta_rotunrot.svg')

    # plot the difference rotated-unrotated
    rot_unrot_dist = np.abs(theta_rotated) - np.abs(theta_unrotated)
    plot_plane_angles_single(constants, np.radians(rot_unrot_dist), 'cu', r=None)
    plt.savefig(constants.PARAMS['FIG_PATH'] + 'theta_rotunrot_diff.png')
    plt.savefig(constants.PARAMS['FIG_PATH'] + 'theta_rotunrot_diff.svg')

    # angles_radians = np.radians(np.stack((theta_unrotated_mean,rot_unrot_dist),1))

    # plot pa
    psi_angles_radians = np.radians(np.stack((psi_unrotated, psi_rotated), 1))
    plot_plane_angles_multiple(constants, psi_angles_radians, r=None, paired=False, cu=True, expt3=False,
                               custom_labels=['unrotated', 'rotated'])
    plt.savefig(constants.PARAMS['FIG_PATH'] + 'psi_rotunrot.png')
    plt.savefig(constants.PARAMS['FIG_PATH'] + 'psi_rotunrot.svg')

    # test difference in theta rotated and unrotated
    rot_unrot_diff = np.abs(theta_angles_radians)[:, 1] - np.abs(theta_angles_radians)[:, 0]
    pycircstat.rayleigh(rot_unrot_diff)
    np.degrees(pycircstat.descriptive.mean(rot_unrot_diff))

    # test phase alignemnt for unrotated

    nonnan_ix = np.where(np.invert(np.isnan(psi_angles_radians[:, 0])))[0]
    pycircstat.rayleigh((psi_angles_radians[nonnan_ix, 0]))
    np.degrees(pycircstat.descriptive.mean((psi_angles_radians[nonnan_ix, 0])))

    nonnan_ix = np.where(np.invert(np.isnan(psi_angles_radians[:, 1])))[0]
    pycircstat.rayleigh((psi_angles_radians[nonnan_ix, 1]))
    np.degrees(pycircstat.descriptive.mean((psi_angles_radians[nonnan_ix, 1])))
