#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 10:34:57 2020

@author: emilia

This file contains functions to analyse plane angles.

requires the pycircstat toolbox, available at https://github.com/circstat/pycircstat
"""

import numpy as np
import matplotlib.pyplot as plt
import pycircstat

import src.helpers as helpers
import src.custom_plot as cplot

# %% circular stats for theta


def rectify_and_average(angles_radians):
    """ Rectify (take the absolute value) angles and take their circular mean across the second array dimension. Results
    will be wrapped to the [0, pi] interval.
    :param np.ndarray angles_radians: Array with angles in radians, shape should be (n_models, n_conditions)
    """
    assert len(angles_radians.shape) == 2, 'Angles should be a 2d array'
    angles_rect_mean = pycircstat.mean(np.abs(angles_radians), axis=1)
    return angles_rect_mean


# noinspection PyStringFormat
def print_mean_and_ci_angle(angles_radians, angle_name, geometry_name):
    """
    Print the circular mean and 95% CI of angles, in degrees.

    :param np.ndarray angles_radians: (n_models, n_delays) Theta angle values in radians
    :param str angle_name: Angle name, choose from 'theta' and 'psi'.
    :param str geometry_name: Name of geometry described by the angles.
    """
    n_delays = angles_radians.shape[1]
    delay_names = ['pre-cue', 'post-cue', 'post-probe']
    for delay in range(n_delays):
        # filter out NaNs if there are any
        nonnan_ix = np.where(np.invert(np.isnan(angles_radians[:, delay])))[0]
        pct_nans = (1 - (len(nonnan_ix) / len(angles_radians))) * 100

        if delay_names[delay] == 'pre-cue' and angle_name == 'theta':
            # rectify the angles
            mean_angle = pycircstat.descriptive.mean(np.abs(angles_radians[nonnan_ix, delay]))
            try:
                # the CI function issues an error when the data is not concentrated enough
                ci = pycircstat.descriptive.mean_ci_limits(np.abs(angles_radians[nonnan_ix, delay]), ci=.95)
            except UserWarning:
                print(f'Warning: {delay_names[delay]} {angle_name} '
                      f'angles not concentrated enough to calculate CI, setting to NaN.')
                ci = np.nan
        else:
            mean_angle = pycircstat.descriptive.mean(angles_radians[nonnan_ix, delay])
            try:
                ci = pycircstat.descriptive.mean_ci_limits(angles_radians[nonnan_ix, delay], ci=.95)
            except UserWarning:
                print(f'{delay_names[delay]} {angle_name} '
                      f'angles not concentrated enough to calculate CI, setting to NaN.')
                ci = np.nan

        mean_angle = helpers.wrap_angle(mean_angle)  # wrap to [-pi,pi]
        mean_angle = np.degrees(mean_angle)

        ci = np.degrees(ci)

        print(f"Angle {angle_name} for {geometry_name} geometry, {delay_names[delay]} delay: "
              f"mean = %.2f, 95CI = %.2f degrees, percent NaN values: %.1f" % (mean_angle, ci, pct_nans))


# noinspection PyStringFormat
def get_inf_stats_theta_cued(angles_radians):
    """
    Run and print the results of the inferential statistics on the distribution of angles theta for the Cued geometry.

    Runs 3 tests:
        1) V-test to test whether the pre-cue angles are clustered with a mean=90 degrees
        2) V-test to test whether the post-cue angles are clustered with a mean=0 degrees
        3) Circular one-sample t-test to test whether the angular pre-cue - post-cue difference is ~=0

    Additionally, for experiment 4, it also runs the following contrast:
        4) V-test to test whether the post-probe angles are clustered with a mean=0 degrees

    :param np.ndarray angles_radians: (n_models, n_delays) Theta cued angle values in radians
    """
    # plane angles are defined on the [-pi, pi] interval

    # pre-cue angles
    # test for non-uniform distribution around 90
    # note: data is bimodal, with peaks at + and -90 degrees
    # v-test assumes an unimodal or uniform distribution over the *entire* circle
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
        # experiment 3
        nonnan_ix = np.where(np.invert(np.isnan(angles_radians[:, 2])))[0]
        p_probe, v_probe = pycircstat.tests.vtest(angles_radians[nonnan_ix, 2], 0)
        print('Post-probe angles: V-test for uniformity/mean=0 :')
        print('    v-stat = %.3f, p = %.3f' % (v_probe, p_probe))


def get_inf_stats_theta_uncued(angles_radians):
    """
    Run and print the results of a Rayleigh test on the distribution of angles theta for the Uncued geometry. H1 states
    that the angles are significantly clustered.

    :param np.ndarray angles_radians: (n_models) Theta uncued angle values for the post-cue delay in radians
    """
    if len(angles_radians.shape) > 1:
        raise NotImplementedError('Analysis not implemented for Experiment 4, assumes angles_radians is one-dimensional'
                                  ' with a single value for each model.')

    p_val, z_stat = pycircstat.tests.rayleigh(angles_radians)
    print('Rayleigh test for uniformity of uncued post-cue angles theta:')
    print('    z-stat = %.3f, p = %.3f' % (z_stat, p_val))


# noinspection PyStringFormat
def get_inf_stats_theta_cued_vs_uncued(angles_radians):
    """
    Run and print the results of a v-test on the distribution of theta angles for the Cued-Uncued geometry. H1 states
    that the angles are clustered with a mean = 90. If the test comes back non-significant, runs a follow-up Rayleigh
    test to check for significant clustering and prints the circular mean.

    :param np.ndarray angles_radians: (n_models) Theta cued-uncued angle values in radians
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

def get_inf_stats_psi_cued(psi_radians):
    """
    Run and print the results of the inferential statistics on the distribution of angles psi for the Cued geometry.

    Runs 2 tests:
        1) Rayleigh test to test whether the pre-cue angles are significantly clustered (H1)
        2) V-test to test whether the post-cue angles are clustered with a mean=0 degrees (H1)

    Additionally, for experiment 4, it also runs the following contrast:
    3) V-test to test whether the post-probe angles are clustered with a mean=0 degrees

    :param np.ndarray psi_radians: (n_models, n_delays) Psi cued angle values in radians
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


def get_inf_stats_psi_phase_aligned(psi_radians, geometry_name):
    """
    Run and print the results of a v-test on the distribution of psi angles for the Uncued or Cued-Uncued geometry. H1
    states that the angles are clustered with a mean = 0.

    :param np.ndarray psi_radians: (n_models) Psi angle values in radians
    :param str geometry_name: name of the geometry. Pick from 'Uncued' and 'Cued-Uncued'
=    """
    if geometry_name == 'cued' or geometry_name == 'Cued':
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


# noinspection PyStringFormat
def get_inf_stats_angle_clustered(angle_radians, angle_name, geometry_name, rectified=False):
    """
    Run and print the results of a Rayleigh test on the distribution of angles for a given geometry. H1 states that
    the angles are significantly clustered. Print the circular mean.

    :param np.ndarray angle_radians: (n_models) Angle values in radians
    :param str geometry_name: name of the angle
    :param str angle_name: name of the geometry.
    :param bool rectified: Optional flag for rectified angles. If true, multiplies the angle estimates by 2 (to stretch
        the possible range to [0, 2pi] - this is the assumption of a Rayleigh test). Default is False.
    """

    assert len(angle_radians.shape) == 1, 'angle_radians should be one-dimensional '

    # need to remove nans first
    non_nan_ix = np.where(np.invert(np.isnan(angle_radians)))[0]

    if rectified:
        # multiple angles by 2 to stretch the range to [0, 2pi] as per Rayleigh test assumption
        p_val, z_stat = pycircstat.tests.rayleigh(angle_radians[non_nan_ix]*2)
    else:
        p_val, z_stat = pycircstat.tests.rayleigh(angle_radians[non_nan_ix])

    angle_mean = pycircstat.descriptive.mean(angle_radians[non_nan_ix])
    angle_mean_degrees = np.degrees(angle_mean)

    print(f'Rayleigh test for uniformity in {geometry_name} angles {angle_name}:')
    print('    z-stat = %.3f, p = %.3f, N = %d, mean = %.2f' % (z_stat, p_val, len(non_nan_ix), angle_mean_degrees))

# %% runners


def run_angles_analysis(constants, theta_degrees, psi_degrees, geometry_name):
    """
    Analyse the plane angles for a given geometry ('cued', 'uncued' or 'cued-uncued').

    Plots the theta and psi angles and runs the inferential statistical tests.

    :param module constants: A Python module containing constants and configuration data for the simulation.
    :param np.ndarray theta_degrees: Array with theta angle values in degree of shape (n_models, n_delays).
    :param np.ndarray or None psi_degrees: Array with psi angle values in degree of shape (n_models, n_delays). If not
        calculated for a given geometry, pass None.
    :param str geometry_name: Geometry name, choose from: 'cued', 'uncued' and 'cued-uncued'.

    """
    assert geometry_name in ['cued', 'uncued', 'cued-uncued'], \
        "Invalid geometry name, choose from: 'cued', 'uncued' and 'cued-uncued'"

    # convert angles into radians
    theta_radians = np.radians(theta_degrees)
    psi_radians = np.radians(psi_degrees) if psi_degrees is not None else None

    if geometry_name == 'cued':
        # plot theta
        if constants.PARAMS['experiment_number'] > 2:
            # plot pre- and post-cue on separate subplots
            cplot.plot_plane_angles_multiple(constants, theta_radians, paired=False, fig_name=f"theta_{geometry_name}")
        elif constants.PARAMS['experiment_number'] == 1:
            # plot pre- and post-cue on one plot
            cplot.plot_plane_angles_multiple(constants, theta_radians, fig_name=f"theta_{geometry_name}")

        # save figure
        if constants.PLOT_PARAMS['save_plots']:
            plt.savefig(f"constants.PARAMS['FIG_PATH']theta_plane_angles_{geometry_name}.svg")

        # get descriptive stats for theta
        print_mean_and_ci_angle(theta_radians, 'theta', geometry_name)

        # get inferential stats for theta
        get_inf_stats_theta_cued(theta_radians)

        # plot psi
        cplot.plot_plane_angles_multiple(constants, psi_radians, paired=False, fig_name=f"psi_{geometry_name}")
        # save figure
        if constants.PLOT_PARAMS['save_plots']:
            plt.savefig(f"{constants.PARAMS['FIG_PATH']}psi_plane_angles_{geometry_name}.svg")

        # get inferential stats for psi
        get_inf_stats_psi_cued(psi_radians)

    elif geometry_name == 'uncued':

        # plot theta angles
        cplot.plot_plane_angles_single(constants, theta_radians, 'post', fig_name=f"theta_post_{geometry_name}")
        if constants.PLOT_PARAMS['save_plots']:
            plt.savefig(f"constants.PARAMS['FIG_PATH']theta_plane_angles_{geometry_name}_postcue.svg")

        # run inferential stats on theta
        get_inf_stats_theta_uncued(theta_radians)

        # plot psi
        cplot.plot_plane_angles_single(constants,
                                       psi_radians,
                                       cond='post',
                                       fig_name=f"psi_post_{geometry_name}")
        if constants.PLOT_PARAMS['save_plots']:
            plt.savefig(f"constants.PARAMS['FIG_PATH']psi_plane_angles_{geometry_name}.svg")

        # get inferential stats for psi
        get_inf_stats_psi_phase_aligned(psi_radians, geometry_name)

    else:
        # cued-uncued geometry

        # get rectified mean angles
        angles_radians_rect_mean = pycircstat.mean(np.abs(theta_radians), axis=1)

        # plot theta angles
        cplot.plot_plane_angles_single(constants, angles_radians_rect_mean, 'cu', fig_name=f"theta_{geometry_name}")
        if constants.PLOT_PARAMS['save_plots']:
            plt.savefig(f"constants.PARAMS['FIG_PATH']theta_plane_angles_{geometry_name}.svg")

        # run inferential stats on theta
        get_inf_stats_theta_cued_vs_uncued(angles_radians_rect_mean)


def run_rot_unrot_angles_analysis(constants, psi_degrees, theta_degrees, plane_labels):
    """
    Analyse the plane angles for the 'rotated' and 'unrotated' Cued plane.

    For each angle (theta, psi), rectifies the values to be within [0, pi] and takes the average across cross-validation
    folds, then plots the rotated and unrotated angle averages.

    For theta, checks if the difference between the rotated and unrotated angles is significantly clustered, across all
    models. For psi, checks if the two psi estimates are significantly clustered.

    :param module constants: A Python module containing constants and configuration data for the simulation.
    :param dict psi_degrees: Dictionary with 'rotated' and 'unrotated' keys. Each entry contains the psi angle values in
        degrees, of shape (n_models, n_cv_folds)
    :param dict theta_degrees: Dictionary with 'rotated' and 'unrotated' keys. Each entry contains the theta angle
        values in degrees, of shape (n_models, n_cv_folds)
    :param list plane_labels: list of plane labels ('unrotated' and 'rotated')
    """
    if constants.PARAMS['experiment_number'] == 2 or constants.PARAMS['experiment_number'] == 4:
        raise NotImplementedError('Analysis only implemented for Experiments 1 and 3')

    psi, theta = {}, {}
    plot_markers = ['bt', 'bs']
    # loop over rotated and unrotated planes
    for marker, label in zip(plot_markers, plane_labels):
        # convert angles into radians
        psi_radians = np.radians(psi_degrees[label])
        theta_radians = np.radians(theta_degrees[label])

        # rectify and average angles across cv folds
        psi[label] = rectify_and_average(psi_radians)
        theta[label] = rectify_and_average(theta_radians)

        # plot psi
        cplot.plot_plane_angles_single(constants, psi[label], marker, fig_name=f"psi_{label}")
        # save
        if constants.PLOT_PARAMS['save_plots']:
            plt.savefig(f"constants.PARAMS['FIG_PATH']psi_plane_angles_{label}_cued_plane.svg")

        # plot theta
        cplot.plot_plane_angles_single(constants, theta[label], marker, fig_name=f"theta_{label}")
        if constants.PLOT_PARAMS['save_plots']:
            plt.savefig(f"constants.PARAMS['FIG_PATH']theta_plane_angles_{label}_cued_plane.svg")

    # compare theta with a Rayleigh test
    theta_difference = theta['rotated'] - theta['unrotated']
    get_inf_stats_angle_clustered(theta_difference, 'theta', 'rotated unrotated difference')

    # analyse psi with Rayleigh tests
    for label in plane_labels:
        get_inf_stats_angle_clustered(psi[label], 'psi', label)
