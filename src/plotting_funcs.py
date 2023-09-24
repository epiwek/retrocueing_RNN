#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 17:01:58 2021

@author: emilia
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pycircstat
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import src.vec_operations as vops
from scipy.stats import vonmises
import src.helpers as helpers


def plot_settings():
    """ Set plot parameters: fontsize and seaborn plotting context/style."""
    plt.rcParams.update({'font.size': 22})
    plt.rcParams['svg.fonttype'] = 'none'
    sns.set_context("notebook", font_scale=1.5)
    sns.set_style("ticks")


plot_settings()


# %% experimental set up


def plot_example_stimulus(data, cmap=sns.cubehelix_palette(start=.5, rot=-.75, as_cmap=True, reverse=True), **kwargs):
    """
    Plot the sequence of inputs corresponding to an example trial stimulus as a heatmap, with timepoints on the x-axis
    and input channels on the y-axis. Colour denotes the input activation level.
    :param np.ndarray data: Input array containing the sequence of inputs for an example trial.
    :param cmap: Optional. Colour map values. Default is the cubehelix sea green palette.
    :param kwargs: additional parameters to the plt.imshow function
    :return: fig, ax, cax (colour axis)
    """

    fig, ax = plt.subplots()

    if len(data.shape) < 3:
        # plot a specific trial
        cax = ax.imshow(data, cmap=cmap, **kwargs)
    else:
        # pick a random trial to plot from the full array
        ix = np.random.randint(data.shape[1])
        cax = ax.imshow(data[:, ix, :], cmap=cmap, **kwargs)
    fig.colorbar(cax, ticks=[0, 1])
    return fig, ax, cax


# %% paired data plots


def add_jitter(data, jitter=None):
    """ Add small jitter values to the data to get rid of duplicate values. Jitter value is determined based on the
    number of duplicate datapoints with a given value.
    :param np.ndarray data: data array
    :param float jitter: Optional. Custom maximum jitter value to use as the end of the range. Default is None, which
        sets the range to [-.2, .2].
    :returns: x_jitter  - jitter values for each datapoint
    """
    # check that data is a vector
    if len(data.shape) > 1:
        raise ValueError('Input data should be 1D')

    unique = np.unique(data)
    x_jitter = np.zeros(data.shape)
    # check for duplicate vals - if there are none, keep the jitter values as all 0s
    if len(unique) != len(data):
        # there are duplicates
        for i in range(len(unique)):
            # save indices of duplicates
            dups_ix = np.where(data == unique[i])[0]
            n_dups = len(dups_ix)
            if jitter is None:
                x_jitter[dups_ix] = np.linspace(-.2, .2, n_dups)
            else:
                # custom jitter value
                x_jitter[dups_ix] = \
                    np.arange(0, n_dups * jitter, jitter) - np.mean(np.arange(0, n_dups * jitter, jitter))
    return x_jitter


def plot_paired_data(x_data, y_data, ax, colours, jitter=None, **kwargs):
    """
    Plot paired data. Datapoints from a given model are plotted as dots of given colours and joined by a black line.
    Jitter is added to the x-values so that datapoints with the same x-value do not overlap.

    :param np.ndarray x_data: Array with x-values from all models (n_models, n_conditions). Can be 1-dimensional if you
         want to add jitter to some constant x-value.
    :param np.ndarray y_data: Array with y-values. Can be 1-dimensional if just want to add jitter to some constant
        x-value.
    :param matplotlib.axes._subplots.AxesSubplot ax: Current axes
    :param list colours: list of colour values for the plotter
    :param float jitter: Optional. Custom jitter values for all data points. Default is None, which adds jitter values
        from a [-.2, .2] interval.
    :param kwargs: additional parameters to the plt.plot() function
    """
    # determine x jitter
    x_jitter = np.zeros(y_data.shape)
    if len(y_data.shape) == 1:
        # if data is 1-dimensional, add a fake dimension so that the code doesn't break
        x_jitter = np.expand_dims(x_jitter, 0)
        y_data = np.expand_dims(y_data, 0)
        x_data = np.expand_dims(x_data, 0)

    for i in range(2):
        x_jitter[:, i] = add_jitter(y_data[:, i], jitter=jitter)

    # plot
    # loop over samples (models)
    for model in range(y_data.shape[0]):
        # plot all the data from a given model - black line
        ax.plot(x_data[model, :] + x_jitter[model, :], y_data[model, :], 'k-', **kwargs)
        # loop over the condition pairs - plot the individual datapoints as dots of a given colour
        for condition in range(y_data.shape[1]):
            ax.plot(x_data[model, condition] + x_jitter[model, condition],
                    y_data[model, condition], 'o', color=colours[condition], **kwargs)


# %% 3D data and planes


def plot_geometry(ax, data_3D, plot_colours, PVEs=None, plot_outline=True, legend_on=True, custom_labels=None):
    """
    Plot the 3D datapoints corresponding to the colour representation from the two cue locations. Datapoints from each
    location are joined by a black line to form a parallelogram.

    :param matplotlib.axes._subplots.AxesSubplot ax: matplotlib.pyplot axes object to which to add the plot
    :param np.ndarray data_3D: Array containing the 3D coordinates for the colour representations from the two cue
        locations. Datapoints from the two locations should be stacked as rows (i.e., location 1 datapoints should
        correspond to the top half of the array). Shape: (n_datapoints, n_dims=3)
    :param list plot_colours: list of colours for the datapoints
    :param list PVEs: Optional, array containing the PVE (percent variance explained) by each PC axis. Default is None.
    :param bool plot_outline: Optional. If True, joins the datapoints corresponding to a single location with a black
        line, to form a parallelogram. Default is True.
    :param bool legend_on: Optional. If True, adds legend labels denoting the marker shape corresponding to each
        location. Default is True.
    :param list custom_labels: Optional. List of legend labels. Default is None, which sets the labels as 'L1' and 'L2'.
    """
    if custom_labels is None:
        labels = ['L1', 'L2']
    else:
        assert len(custom_labels) == 2, 'Custom labels should be a list with 2 entries'
        labels = custom_labels

    assert data_3D.shape[0] % 2 == 0, 'Data array should contain the same number of datapoints for each location,' \
                                      ' stacked as rows.'
    assert data_3D.shape[1] == 3, 'Data array should contain 3D coordinates for the datapoints as columns.'

    ms = 50  # marker size
    n_colours = data_3D.shape[0] // 2  # number of colour datapoints for each location
    location_indices = [np.arange(n_colours), np.arange(n_colours, data_3D.shape[0])]
    markers = ['^', 's']

    # loop over the two cue locations
    for loc, loc_label, ixs, marker in zip(range(2), labels, location_indices, markers):
        # plot the parallelogram defined by colours at location loc
        if plot_outline:
            # join points with a black line
            ax.plot(np.append(data_3D[ixs, 0], data_3D[loc * n_colours, 0]),
                    np.append(data_3D[ixs, 1], data_3D[loc * n_colours, 1]),
                    np.append(data_3D[ixs, 2], data_3D[loc * n_colours, 2]), 'k-')

        # plot first datapoint with loc abel
        ax.scatter(data_3D[0, 0], data_3D[0, 1], data_3D[0, 2], marker='^', s=ms,
                   c='k', label=loc_label)
        # plot individual datapoints
        ax.scatter(data_3D[ixs, 0], data_3D[ixs, 1], data_3D[ixs, 2], marker=marker, s=ms, c=plot_colours)

    if PVEs is not None:
        # add PVE (percent variance explained) labels for each axis
        ax.set_xlabel('PC1 [' + str(np.round(PVEs[0] * 100, 1)) + '%]',
                      labelpad=12)
        ax.set_ylabel('PC2 [' + str(np.round(PVEs[1] * 100, 1)) + '%]',
                      labelpad=12)
        ax.set_zlabel('PC3 [' + str(np.round(PVEs[2] * 100, 1)) + '%]',
                      labelpad=5)
    if legend_on:
        ax.legend(bbox_to_anchor=(1, .9), loc='center left')


def plot_plane(ax, vertices, fc='k', alpha=0.2):
    """
    Plot a grey shaded polygon with the given 3D vertices.
    :param matplotlib.axes._subplots.AxesSubplot ax: matplotlib.pyplot axes object to which to add the plot
    :param np.ndarray vertices: Array of the plane vertices (n_vertices, n_dims=3)
    :param str fc: Optional. Colour of the plotted plane. Default is black. Accepts any argument that can be passed as
        'facecolor' to a matplotlib plotting function.
    :param float alpha: Optional. Alpha (transparency) value for the plane (should be within [0, 1], where 0 is fully
        transparent and 1 fully opaque). Default is 0.2.
    """
    assert vertices.shape[1] == 3, 'Vertices should be 3-dimensional.'
    # plot a polygon with given vertices in 3D
    ax.add_collection3d(Poly3DCollection([vertices], facecolor=fc, edgecolor=[], alpha=alpha))


def plot_subspace(ax, points, plane_vecs, fc='k', alpha=0.2):
    """
    Plot the best-fitting subspace as a quadrilateral with vertices being the projections of original points onto the
    plane. Sort the datapoints so that the plotted plane will be a convex quadrilateral.

    :param matplotlib.axes._subplots.AxesSubplot ax: matplotlib.pyplot axes object to which to add the plot
    :param np.ndarray points: Data array, shape: (n_datapoints, 3)
    :param np.ndarray plane_vecs: Array containing the plane_defining vectors (n_vectors, n_dims)
    :param str fc: Optional. Colour of the plotted plane. Default is black. Accepts any argument that can be passed as
        'facecolor' to a matplotlib plotting function.
    :param float alpha: Optional. Alpha (transparency) value for the plane (should be within [0, 1], where 0 is fully
        transparent and 1 fully opaque). Default is 0.2.
    :return:
    """
    # plot the best-fitting plane

    if points.shape[1] != 3:
        raise NotImplementedError('Check the shape of the data matrix - should be (n_points, 3)')

    # find vertices
    n_points = points.shape[0]
    verts = np.zeros((n_points, 3))

    com = np.mean(points, axis=0)  # centre of mass

    for i in range(n_points):
        # get projection of demeaned 3d points
        verts[i, :] = vops.get_projection(points[i, :] - com, plane_vecs)
        verts[i, :] += com  # add the mean back

    # sort vertices according to the shortest path - so that plotted plane will be a quadrilateral
    sorted_verts, sorting_order = vops.sort_by_path_length(verts)

    # plot the best-fit plane
    plot_plane(ax, sorted_verts, fc, alpha)


def plot_full_geometry(constants, model_numbers, subspaces, fig_names=None):
    """
    For each example model, plot the given geometry in each delay interval.
    :param module constants: A Python module containing constants and configuration data for the simulation.
    :param list model_numbers: Numbers of example models.
    :param dict subspaces: Fitted subspaces, saved in a dictionary indexed by delay name keys in the
        'delay{delay_number}' format (e.g. 'delay1'). Each delay name key contains model number keys with the fitted
        Subspace class.
    :param list fig_names: Optional. List of custom figure names. Default is None.
    :return: fig_list: list of Figure objects
    """
    n_subplots = constants.PARAMS['n_delays']
    fig_size = (n_subplots * 5, 5)
    fig_list = []
    for model in model_numbers:

        fig = plt.figure(figsize=fig_size, num=f"Model {model}")

        if fig_names is not None:
            # set figure window name
            fig.canvas.manager.set_window_title(fig_names[model])

        fig_list.append(fig)

        for delay in range(1, constants.PARAMS['n_delays'] + 1):
            ax = plt.subplot(1, n_subplots, delay, projection='3d')
            plot_geometry(ax,
                          subspaces[f"delay{delay}"][model].coords_3d,
                          constants.PLOT_PARAMS['4_colours'],
                          subspaces[f"delay{delay}"][model].PVEs,
                          legend_on=(delay == constants.PARAMS['n_delays']))
            # add the planes of best fit
            plot_subspace(ax,
                          subspaces[f"delay{delay}"][model].plane1.coords_3d,
                          subspaces[f"delay{delay}"][model].plane1.plane_vecs_xy,
                          fc='k', alpha=0.2)
            plot_subspace(ax,
                          subspaces[f"delay{delay}"][model].plane2.coords_3d,
                          subspaces[f"delay{delay}"][model].plane2.plane_vecs_xy,
                          fc='k', alpha=0.2)

            ax.set_title('Angle: %.1f' % subspaces[f"delay{delay}"][model].theta_degrees)
            helpers.equal_axes(ax)
    return fig_list


def plot_PVEs_3D(constants, PVEs_3D, fig_name=None):
    """
    Plot the percent of variance explained (PVE) in the data by the 3 principal components (PC). Values for
    individual models (for each timepoint examined) plotted as dots joined by black lines,  group-wise averages as
    bars.
    :param module constants: A Python module containing constants and configuration data for the simulation.
    :param numpy.ndarray PVEs_3D: Percent of variance explained values for all models and delays in the format
        (n_models, n_delays, n_dims=3)
    :param str fig_name: Optional. Custom figure name. Default is None.
    """

    # set plot colours and markers
    pal = sns.color_palette("dark")
    if constants.PARAMS['experiment_number'] == 4:
        inds = [3, 0, -3]
        markers = ['o', '^', 's']
    else:
        inds = [3, 0]
        markers = ['o', '^']
    cols = [pal[ix] for ix in inds]
    ms = 10

    # plot
    fig = plt.figure(figsize=(6.5, 5))

    if fig_name is not None:
        # set figure window name
        fig.canvas.manager.set_window_title(fig_name)

    ax = plt.subplot(111)

    for model in range(constants.PARAMS['n_models']):
        # add datapoints for individual models
        for i in range(len(inds)):
            ax.plot(i, PVEs_3D[model, i, :].sum(), marker=markers[i],
                    color=cols[i], alpha=0.2, markersize=ms)
        ax.plot(range(len(inds)), PVEs_3D[model, :, :].sum(-1), 'k-', alpha=0.2)  # sum PC1-3

    # add group-wise averages
    ax.plot(range(len(inds)), PVEs_3D.sum(-1).mean(0), 'k-')
    for i in range(len(inds)):
        ax.plot(i, PVEs_3D[:, i, :].sum(-1).mean(), marker=markers[i],
                color=cols[i], markersize=ms)

    # add labels
    plt.ylabel('PVE by first 3 PCs')
    if constants.PARAMS['experiment_number'] == 4:
        plt.xlim((-0.5, 2.5))
        plt.xticks([0, 1, 2], ['pre-cue', 'post-cue', 'probe'])
    else:
        plt.xlim((-0.5, 1.5))
        plt.xticks([0, 1], ['pre-cue', 'post-cue'])

    plt.tight_layout()
    sns.despine()


# %% angle plots

# % polar plot of angles
def plot_polar_scatter(ax, data, r, **kwargs):
    """
    Scatterplot on a polar axis.
    :param matplotlib.axes._subplots.AxesSubplot ax: matplotlib.pyplot axes object to which to add the plot
    :param np.ndarray data: Data to be plotted, shape (n_datapoints, )
    :param np.ndarray r: radius values, shape (n_datapoints, )
    :param kwargs: additional parameters to the plot function
    :return:
    """
    ax.plot(data, r, **kwargs)


def plot_polar_scatter_paired(ax, data, r, **kwargs):
    """
    Plot a paired dataset on a polar axis.
    :param matplotlib.axes._subplots.AxesSubplot ax: matplotlib.pyplot axes object to which to add the plot
    :param np.ndarray data: Data to be plotted, shape (n_datapoints, 2)
    :param r: radius values, shape (n_datapoints, )
    :param kwargs: additional parameters to the plot function
    :return:
    """
    for i in range(len(data)):
        ax.plot(data[i, :], np.tile(r[i], (data.shape[1], 1)), **kwargs)


def format_polar_axes(ax, r=None):
    """
    Format polar axes by removing the grid and y-ticks.

    :param matplotlib.axes._subplots.AxesSubplot ax: matplotlib.pyplot axes object to be formatted
    :param np.ndarray r: radius values, shape (n_datapoints, )
    """
    ax.grid(False)
    ax.tick_params(axis='x', which='major', pad=14)
    if np.all(r is None):
        # custom radius values
        ax.set_ylim([0, r.max() + .05 * r.max()])
    else:
        ax.set_ylim([0, 1.05])
    ax.set_yticks([])


def plot_plane_angles_multiple(constants, angles_radians, r=None, paired=True, cu=False, custom_labels=None,
                               fig_name=None):
    """
    Plot multiple angles, either in a single figure with datapoints from a single row joined by black lines, or in
    separate figures.

    :param module constants: A Python module containing constants and configuration data for the simulation.
    :param np.ndarray angles_radians: Angles to be plotted, shape (n_datapoints, m)
    :param np.ndarray r: Optional. Custom radius values, shape (n_datapoints, ). Default is None, which sets the radius
        value to 1.
    :param bool paired: Optional. If True, datapoints from a single row are joined by black lines. Default is True.
    :param bool cu: Optional. If True, sets the markers and plot colours to the 'cued-uncued' standard (black triangles
        and squares).
    :param list custom_labels: Optional. List of custom legend labels. Default is None, which sets the labels to the
        conditions being compared for the given geometry / experiment.
    :param str fig_name: Optional. Custom figure window name. Default is None.
    :return:
    """
    if r is None:
        # if r == None
        r = np.ones((constants.PARAMS['n_models'],))  # radius values for the datapoints
        radius = 1
    else:
        radius = r.max()  # radius value for the mean

    labels = custom_labels

    if cu:
        # cued vs uncued geometry
        markers = ['^', 's']
        if custom_labels is None:
            labels = ['L1 cued', 'L2 cued']
        cols = ['k', 'k']
        alphas = [.2, .2]
    elif constants.PARAMS['experiment_number'] == 4:
        # experiment 4 cued
        pal = sns.color_palette("dark")
        inds = [3, 0, -3]
        cols = [pal[ix] for ix in inds]
        markers = ['o', '^', 's']
        if custom_labels is None:
            labels = ['pre-cue', 'post-cue', 'post-probe']
        alphas = [.2, .2, .2]
    else:
        # cued geometry
        pal = sns.color_palette("dark")
        inds = [3, 0]
        cols = [pal[ix] for ix in inds]
        markers = ['o', '^']
        if custom_labels is None:
            labels = ['pre', 'post']
        alphas = [.2, .2]
    ms = 12

    n_cats = angles_radians.shape[-1]
    pct_nans = np.zeros((n_cats,))
    if paired:
        fig = plt.figure(figsize=(7.9, 5))
        ax = fig.add_subplot(111, polar=True)

        # plot all datapoints
        for i in range(n_cats):
            plot_polar_scatter(ax, angles_radians[:, i], r,
                               marker=markers[i], color=cols[i],
                               ls='None',
                               alpha=alphas[i], markersize=ms)
            # add grand means (need to remove nans)
            nonnan_ix = np.where(np.invert(np.isnan(angles_radians[:, i])))[0]
            plot_polar_scatter(ax, pycircstat.descriptive.mean(angles_radians[nonnan_ix, i]),
                               radius, marker=markers[i], color=cols[i],
                               markersize=ms, label=labels[i])

            pct_nans[i] = 100 * (len(angles_radians) - len(nonnan_ix)) / len(angles_radians)
        # join the values corresponding to an individual model
        plot_polar_scatter_paired(ax, angles_radians, r, c='k', ls='-', alpha=0.2)
        format_polar_axes(ax, r)
        ax.legend(bbox_to_anchor=(1.4, 1))
    else:
        # separate subplots for each category
        fig = plt.figure(figsize=(7.9 * n_cats, 5))
        for i in range(n_cats):
            ax = fig.add_subplot(1, n_cats, i + 1, polar=True)
            # plot all datapoints
            plot_polar_scatter(ax, angles_radians[:, i],
                               r, marker=markers[i], ls='None', color=cols[i],
                               alpha=alphas[i], markersize=ms)
            # add grand means (need to remove nans)
            nonnan_ix = np.where(np.invert(np.isnan(angles_radians[:, i])))[0]
            plot_polar_scatter(ax, pycircstat.descriptive.mean(angles_radians[nonnan_ix, i]),
                               radius, marker=markers[i], color=cols[i], markersize=ms, label=labels[i])
            format_polar_axes(ax, r)
            ax.legend(bbox_to_anchor=(1.4, 1))
            pct_nans[i] = 100 * (len(angles_radians) - len(nonnan_ix)) / len(angles_radians)

    if fig_name is not None:
        # set figure window name
        fig.canvas.manager.set_window_title(fig_name)

    plt.tight_layout()


def plot_plane_angles_single(constants, angles_radians, cond, r=None, fig_name=None):
    """
    Plot angles for a single condition.

    :param module constants: A Python module containing constants and configuration data for the simulation.
    :param np.ndarray angles_radians: Angles to be plotted, shape (n_datapoints, m)
    :param str cond: Plotted condition, sets the plot style (markers, colours, etc). Choose from 'pre' (red dots),
        'post' (navy triangles), 'cu' (black dots), 'bt' (black triangles) and 'bs' (black squares).
    :param np.ndarray r: Optional. Custom radius values, shape (n_datapoints, ). Default is None, which sets the radius
        value to 1.
    :param str fig_name: Optional. Custom figure window name. Default is None.
    """
    pal = sns.color_palette("dark")
    if r is None:
        r = np.ones((constants.PARAMS['n_models'],))
        rr = 1
    else:
        rr = r.max()  # radius value for the mean

    if cond == 'pre':
        # pre-cue
        ix = 3
        cols = pal[ix]
        markers = 'o'
    elif cond == 'post':
        # post-cue
        ix = 0
        cols = pal[ix]
        markers = '^'
    elif cond == 'cu':
        # cued uncued
        markers = 'o'
        cols = 'k'
    elif cond == 'bt':
        # black triangle
        markers = '^'
        cols = 'k'
    elif cond == 'bs':
        # black square
        markers = 's'
        cols = 'k'
    else:
        raise ValueError("Invalid cond, choose from 'pre', 'post', 'cu', 'bt' and 'bs'")
    ms = 12

    fig = plt.figure(figsize=(7.9, 5))
    ax = fig.add_subplot(111, polar=True)

    # plot all datapoints
    plot_polar_scatter(ax, angles_radians,
                       r, marker=markers, ls='None', color=cols, alpha=0.2, markersize=ms)
    # add grand means (need to remove nans)
    nonnan_ix = np.where(np.invert(np.isnan(angles_radians)))[0]
    plot_polar_scatter(ax, pycircstat.descriptive.mean(angles_radians[nonnan_ix]),
                       rr, marker=markers, color=cols, markersize=ms)
    format_polar_axes(ax, r)

    if fig_name is not None:
        # set figure window name
        fig.canvas.manager.set_window_title(fig_name)

    plt.tight_layout()


def plot_geometry_estimates_experiment_2(constants, delay2_max_length, all_estimates, are_angles=False):
    """
    Plots the result of the Cued geometry comparison analysis between the variants of Experiment 2 (retrocue timing, AI
    plot shown in Fig. 5B).

    Plots the Cued geometry estimates (either in the form of plane angles or AI values) against the length of the
    post-cue delay interval. Pre-cue estimates plotted in red, post-cue in navy. Datapoints for individual models and
    group averages plotted as semi-transparent and opaque markers, respectively.

    :param module constants: A Python module containing constants and configuration data for the simulation.
    :param int delay2_max_length: maximum length of the post-cue delay interval (7 for the data reported in the
        publication).
    :param np.ndarray all_estimates: Cued geometry estimates for all models and delay timepoints, shape:
        (n_models, n_timepoints, n_delay_2_lengths)
    :param bool are_angles: Optional. Flag determining whether the estimates are angles or AI values. Default is False.
    :return:
    """

    if are_angles:
        # angles will be rectified
        all_estimates = np.abs(all_estimates)

    # experiment settings
    labels = ['pre-cue', 'post-cue']
    delay2_lengths = np.arange(delay2_max_length + 1)
    n_models = constants.PARAMS['n_models']
    n_delays = constants.PARAMS['n_delays']

    # plot settings
    pal = sns.color_palette("dark")
    assert n_delays == 2, \
        "Edit the plot_geometry_estimates_experiment_2 function to include more markers, more colours and different " \
        "x-axis jitter values if plotting data from more than 2 timepoints"

    markers = ['o', '^']
    cols = [pal[ix] for ix in [3, 0]]
    jitter = .125  # jitter value separating the datapoints from different delays on the x-axis
    jitter_sign = [-1, 1]
    ms = 10
    plt.figure(figsize=(9, 5), num='Experiment 2: Cued geometry vs retrocue timing comparison')
    ax = plt.subplot(111)

    # loop through the delay intervals
    for delay_no in range(n_delays):
        if are_angles:
            # plot circular means
            means = np.degrees(pycircstat.descriptive.mean(np.radians(all_estimates[:, delay_no, :]), axis=0))
        else:
            means = all_estimates[:, delay_no, :].mean(0)
        ax.plot(delay2_lengths + jitter_sign[delay_no] * jitter, means, marker=markers[delay_no],
                color=cols[delay_no], markersize=ms, ls='--',
                label=labels[delay_no])

        # loop through the experiment variants (delay2 lengths) to plot the individual model datapoints
        for ix, delay_length in enumerate(delay2_lengths):
            ax.plot(np.ones((n_models,)) * delay_length + jitter_sign[delay_no] * jitter,
                    all_estimates[:, delay_no, ix],
                    marker=markers[delay_no], ls='', color=cols[delay_no], alpha=.2, markersize=ms)

    ax.set_xticks(delay2_lengths)
    if are_angles:
        ax.set_ylabel('abs(cued plane angle) [°]')
        ax.set_ylim((all_estimates.min() - ms), all_estimates.max() + ms)
    else:
        ax.set_ylabel('Cued AI')
    ax.set_xlabel('Post-cue delay length [cycles]')
    ax.legend(bbox_to_anchor=(1, 1))
    sns.despine()
    plt.tight_layout()


def plot_err_distr(binned_errors, bin_centres, b, fitted_params, sem=None, ax=None, c='k', dashed=False, fig_name=None):
    """
    Plots the error distribution. Plots the datapoints as dots (with optional errorbars) plus the fitted vonMises
    distribution.

    :param np.ndarray binned_errors: Binned error values.
    :param np.ndarray bin_centres: Centres of the bins
    :param float b: end of the bin range
    :param dict fitted_params: Dictionary containing the fitted von Mises distribution parameters
    :param np.ndarray sem: SEM values for errorbars.
    :param matplotlib.axes._subplots.AxesSubplot ax: matplotlib.pyplot axes object to be formatted
    :param str c: Optional. Colour value for the plotted. Default is 'k' (black).
    :param bool dashed: Optional. If True, uses a dashed line for the fitted distribution plot. Default is False.
    :param str fig_name: Optional. Custom figure window name. Default is None.
    :return:
    """
    pdf_x = np.linspace(-b, b, 100)  # x vals for the fitted von Mises pdf
    pdf_y = vonmises.pdf(pdf_x,
                         fitted_params['kappa'],
                         fitted_params['mu'],
                         fitted_params['scale'])
    if ax is None:
        # create a new figure
        fig = plt.figure()
        if fig_name is not None:
            # set figure window name
            fig.canvas.manager.set_window_title(fig_name)
        # add axes
        ax = plt.subplot(111)

    if sem is not None:
        # add error bars for datapoints
        ax.errorbar(bin_centres, binned_errors, yerr=sem, fmt='none', ecolor=c)
    # plot datapoints
    ax.plot(bin_centres, binned_errors, 'o', mec=c, mfc='w')
    if dashed:
        line_style = '--'
    else:
        line_style = '-'
    # plot the fitted von-mises distribution
    ax.plot(pdf_x, pdf_y, line_style, c=c, lw=2, label='fit')
    x_ticks = [-100, 0, 100]  # to match the monkey figure
    ax.set_xticks(np.radians(x_ticks))
    ax.set_xticklabels(x_ticks)
    ax.set_xlabel('angular error (degrees)')
    ax.set_ylabel('density of trials')
    sns.despine()
    plt.show()


def plot_all_error_data(constants, test_conditions, all_results, fig_name=None):
    """
    High-level plotter for the error distributions. For experiments 1-3, plots the error distributions for each
    test condition in a separate subplot. For experiment 4, plots the error distributions from valid and invalid
    trials on one plot, as green and red, respectively.

    :param module constants: A Python module containing constants and configuration data for the simulation.
    :param list test_conditions: Test conditions to be plotted. Should correspond to a subset of the keys in the
        all_results data dictionary.
    :param dict all_results: Nested dictionary containing the results of the behavioural analysis to be plotted.
        Superordinate keys should correspond to test conditions, subordinate keys include the different measurements and
        parameters: 'mean_errs', 'bin_centres', 'bin_max', 'fitted_params', 'sem_errs'.
    :param str fig_name: Optional. Custom figure window name. Default is None.
    :return:
    """

    if constants.PARAMS['cue_validity'] == 1:
        # experiments 1-3 and experiment 4, cue_validity = 1
        # plot results as different panels
        n_conditions = len(test_conditions)
        fig, axes = plt.subplots(1, n_conditions, sharex='all', sharey='all')
        fig_size = (12.8, 3.4) if n_conditions > 1 else (4.27, 3.4)
        fig.set_size_inches(fig_size)
        if fig_name is not None:
            # set figure window name
            fig.canvas.manager.set_window_title(fig_name)

        for i, condition in enumerate(test_conditions):
            curr_axes = axes[i] if n_conditions > 1 else axes
            plot_err_distr(all_results[condition]['mean_errs'],
                           all_results[condition]['bin_centres'],
                           all_results[condition]['bin_max'],
                           all_results[condition]['fitted_params'],
                           sem=all_results[condition]['sem_errs'],
                           ax=curr_axes)
            curr_axes.set_title(condition)
        plt.tight_layout()
    else:
        # plot valid and invalid trials on one plot
        fig = plt.figure(figsize=(4.27, 3.4))
        if fig_name is not None:
            # set figure window name
            fig.canvas.manager.set_window_title(fig_name)
        ax = plt.subplot(111)
        # plot valid trials
        conditions = ['valid_trials', 'invalid_trials']
        plot_colours = ['g', 'r']
        for condition, colour in zip(conditions, plot_colours):
            dashed = constants.PARAMS['cue_validity'] == .5
            plot_err_distr(all_results[condition]['mean_errs'],
                           all_results[condition]['bin_centres'],
                           all_results[condition]['bin_max'],
                           all_results[condition]['fitted_params'],
                           sem=all_results[condition]['sem_errs'],
                           ax=ax,
                           c=colour,
                           dashed=dashed)
        plt.tight_layout()


def plot_mixture_model_params_validity(mixture_param_data_dict):
    """
    Plot the mixture model parameters fit to choice data from models from Experiment 4. Each model was trained under a
    given cue validity condition (0.5 and 0.75) evaluated on valid and invalid trials. Each mixture model parameter is
    plotted as a separate figure. Data is plotted as parameter mean+-SEM, with trial type (valid, invalid) on x-axis and
    condition (cue validity 0.5, 0.75) denoted by plot colour.

    :param dict mixture_param_data_dict : A dictionary containing mixture parameter data. It should have the following
        keys:
        - 'model_nr' (array): An Array with model number for each observation.
        - parameter-specific keys (e.g. 'K'), each containing a pandas dataframe with 'trial_type', 'condition' and
            parameter-specific columns.

    """
    # extract the names of the mixture parameters from the data dictionary
    mixture_params = [key for key in mixture_param_data_dict.keys() if key != 'model_nr']
    for param in mixture_params:
        sns.catplot(x='trial_type', y=param, hue='condition', data=mixture_param_data_dict[param],
                    kind="point", markers=["^", "o"], units=mixture_param_data_dict['model_nr'], dodge=True, ci=68,
                    palette=sns.color_palette("Set2")[2:0:-1])

        plt.xlabel('trial type')


def plot_CDI(constants, CDI, log_transform=True):
    """
    Plot the CDI metric. Plot style depends on the experiment number.

    For experiment 1, values for individual models are plotted as markers (with colour denoting the condition: pre-cue,
    cued, and uncued), joined by black lines. Group averages are plotted as semi-transparent bars.

    For experiment 3, the function produces a barplot,  with bar height corresponding to the mean across models and
    error bars corresponding to SEM. Bars correspond to conditions, namely: pre-cue, cued, uncued, probed and unprobed
    colours.

    :param module constants: A Python module containing constants and configuration data for the simulation.
    :param pandas.DataFrame CDI: CDI data frame, rows correspond to estimates for individual models, and columns
        include: 'pre-cue', 'post-cue', 'post-probe' (for Experiment 1; for Experiment 3, cue_validity < 1 also:
        'probed_invalid', 'unprobed_invalid', 'probed_valid', 'unprobed_valid')
    :param bool log_transform: Optional. If True, log-transforms the CDI values (for better discriminability in plots).
        Default is True.
    """
    if log_transform:
        CDI = np.log(CDI)
        y_label = 'log(CDI)'
    else:
        y_label = 'CDI'

    pal = sns.color_palette("dark")
    ms = 16

    # set x_vals, x_ticks and plot colours
    if constants.PARAMS['experiment_number'] == 4:
        if constants.PARAMS['cue_validity'] < 1:
            # probabilistic conditions
            cols = ['k', pal[9], pal[2], pal[6], pal[6], pal[4], pal[4]]
            x_vals = [0, .5, .75, 1.25, 1.5, 1.75, 2]
            markers = None
        else:
            # deterministic condition
            cols = ['k', pal[9], pal[2], pal[6], pal[4]]
            x_vals = [0, .5, .75, 1.25, 1.5]
            markers = None

        x_vals = pd.DataFrame(np.tile(x_vals, (constants.PARAMS['n_models'], 1)), columns=CDI.columns)
        x_ticks = [x_vals['pre-cue'].loc[0], x_vals[['cued', 'uncued']].loc[0].mean(), x_vals.iloc[0, 3:].mean()]
        x_tick_labels = ['pre-cue', 'post-cue', 'post-probe']
    else:
        # Experiments 1-3
        cols = ['k', pal[9], pal[2]]
        x_vals = np.tile([0, .875, 1.125], (constants.PARAMS['n_models'], 1))
        x_vals = pd.DataFrame(x_vals, columns=CDI.columns)
        x_ticks = range(2)
        x_tick_labels = ['pre-cue', 'post-cue']
        markers = {key: val for key, val in zip(list(CDI.columns), ['o', '^', 'X'])}

    # convert into a dictionary to use column names
    cols = {key: val for key, val in zip(list(CDI.columns), cols)}

    plt.figure(figsize=(6.65, 5), num='CDI results')
    ax = plt.subplot(111)

    if constants.PARAMS['experiment_number'] < 4:
        if constants.PARAMS['experiment_number'] > 1:
            raise NotImplementedError('Not tested for Experiments 2 and 3')

        # plot the lines joining datapoints belonging to a single model
        # pre-cue to cued
        ax.plot(x_vals[['pre-cue', 'cued']].T.to_numpy(), CDI[['pre-cue', 'cued']].T.to_numpy(), 'k-', alpha=.2)
        # pre-cue to uncued
        ax.plot(x_vals[['pre-cue', 'uncued']].T.to_numpy(), CDI[['pre-cue', 'uncued']].T.to_numpy(), 'k-', alpha=.2)

        # plot the datapoints
        for condition in ['pre-cue', 'cued', 'uncued']:
            ax.plot(x_vals[condition], CDI[condition], markers[condition], c=cols[condition], markersize=ms)

    # add mean +- SEM (barplot + errorbars)
    if constants.PARAMS['experiment_number'] == 4:
        # plot bars with SEM errorbars
        if constants.PARAMS['cue_validity'] < 1:
            # data_labels = ['precue', 'cued', 'uncued', 'probed valid', 'probed invalid',
            #                'unprobed valid', 'unprobed invalid']
            # plot bars
            for cond in list(CDI.columns):
                if cond in ['probed_invalid', 'unprobed_invalid']:
                    # plot hatched bars
                    ax.bar(x_vals[cond].mean(), CDI[cond].mean(), yerr=CDI[cond].sem(), ec=cols[cond], facecolor='w',
                           capsize=2, hatch='///', alpha=.25, width=.25, label=cond)
                else:
                    ax.bar(x_vals[cond].mean(), CDI[cond].mean(), yerr=CDI[cond].sem(), ec=cols[cond],
                           facecolor=cols[cond], capsize=2, alpha=.2, width=.25,
                           label=[] if cond is 'pre-cue' else cond)
        else:
            # plot bars
            for cond in list(CDI.columns):
                ax.bar(x_vals[cond].mean(), CDI[cond].mean(), yerr=CDI[cond].sem(), ec=cols[cond], facecolor=cols[cond],
                       capsize=2, alpha=.2, width=.25, label=cond if cond is not 'pre-cue' else None)
        ax.set_ylim([-1.5, 9])

    else:
        if constants.PARAMS['experiment_number'] > 1:
            raise NotImplementedError('This function was not tested for Experiments 2 and 3')

        for cond in list(CDI.columns):
            # means but no errorbars
            ax.bar(x_vals[cond].mean(), CDI[cond].mean(), facecolor=cols[cond], alpha=.2, width=.25,
                   label=cond if cond is not 'pre-cue' else None)
        ax.set_xlim([-0.25, 1.375])

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels)
    ax.set_ylabel(y_label)

    plt.legend()
    plt.tight_layout()

    sns.despine()


def plot_AI(constants, ai_table, geometry_name):
    """
    Plots the results of the AI analysis. Values for individual models depicted as dots, average across group as bars.

    :param module constants: A Python module containing constants and configuration data for the simulation.
    :param np.ndarray ai_table: AI data array, (n_dims, n_timepoints, n_models)
    :param str geometry_name: Desired geometry. Choose from: 'cued', 'uncued', 'cued_up_uncued_down',
        'cued_down_uncued_up', 'cued_uncued', and 'unrotated_rotated'. Controls the plot aesthetics.
    """

    assert geometry_name in ['cued', 'uncued', 'cued_up_uncued_down', 'cued_down_uncued_up', 'cued_uncued',
                             'unrotated_rotated'], "Invalid geometry_name, choose from: 'cued', 'uncued', " \
                                                   "'cued_up_uncued_down', 'cued_down_uncued_up', 'cued_uncued', " \
                                                   "'unrotated_rotated'"
    assert len(ai_table.shape) == 3, 'ai_table must be 3-dimensional, with shape: (n_dims, n_timepoints, n_models)'
    # plot settings: colours, labels, alpha values etc
    if geometry_name == 'unrotated_rotated':
        cols = ['k', 'k']
        alphas = [.2, .4]
        labels = ['unrotated', 'rotated']
        jitter = np.array([0.25 / 2, 0.25 / 2])
        jitter_sign = np.array([-1, 1])
    elif geometry_name in ['cued_uncued', 'cued_up_uncued_down', 'cued_down_uncued_up']:
        cols = ['k', 'k']
        labels = [geometry_name]
        alphas = [.2, .4]
        jitter = np.array([0.25 / 2])
        jitter_sign = np.array([0])
    else:
        pal = sns.color_palette("dark")
        inds = [3, 0]
        cols = [pal[ix] for ix in inds]
        cols.append('k')
        alphas = [.2, .2, .2]
        labels = ['pre-cue', 'post-cue', 'post-probe']
        jitter = np.array([0.25 / 2, 0.25 / 2, 0.25 * (3/2)])
        jitter_sign = np.array([-1, 1, 1])

    ms = 16
    n_dims = ai_table.shape[0]  # n of dimensions for which AI was calculated
    n_conditions = ai_table.shape[1]  # n timepoints
    x_vals = np.arange(n_dims) + 2  # dimensionality values for which AI was calculated, starting at 2

    assert len(labels) == len(jitter), 'Each condition should have a corresponding jitter value'

    plt.figure(figsize=(6.65, 5), num=f'AI {geometry_name}')
    for dim in range(n_dims):
        for cond in range(n_conditions):
            # plot individual model datapoints (as dots)
            x = np.ones((constants.PARAMS['n_models'],)) * x_vals[dim] + jitter[cond] * jitter_sign[cond]
            plt.plot(x, ai_table[dim, cond, :], 'o', c=cols[cond], markersize=ms)

            if cond > 0:
                # multiple conditions/timepoints - link datapoints corresponding to a single model with a black line
                x_previous_cond = np.ones((constants.PARAMS['n_models'],)) * x_vals[dim] + \
                                  jitter[cond - 1] * jitter_sign[cond - 1]
                plt.plot([x_previous_cond, x], ai_table[dim, [cond - 1, cond], :], 'k-', alpha=.2)

            # plot grand means (as bars)
            plt.bar(x_vals[dim] + jitter[cond] * jitter_sign[cond], ai_table[dim, cond, :].mean(), facecolor=cols[cond],
                    alpha=alphas[cond], width=.25, label=labels[cond] if dim == n_dims - 1 else None)

    plt.xticks(x_vals, labels=x_vals)
    plt.xlabel('Dimensionality')
    plt.ylabel('AI')
    plt.ylim([0, 1.1])
    plt.legend(bbox_to_anchor=(1, 1))
    sns.despine()
    plt.tight_layout()

    return


def plot_off_diagonal(off_diag_scores):
    """
    Creates a boxplot of the mean off-diagonal decoding scores for the variable (Experiment 3) and fixed delay
    (Experiment 1) conditions. Chance level plotted as dashed line. Results reported in Fig. 6B in the manuscript.

    :param np.ndarray off_diag_scores: Mean off-diagonal (cross-temporal) decoding scores for individual models. Values
        from the variable delay condition in the first, fixed - in the second column. Shape: (n_models, n_conditions)

    """
    # reformat data into a pandas dataframe for seaborn
    n_models = off_diag_scores.shape[0]
    labels = np.array([['variable'] * n_models, ['fixed'] * n_models]).reshape(-1)
    tbl = pd.DataFrame(np.stack((off_diag_scores.reshape(-1, order='F'),
                                 labels), 1),
                       columns=['mean delay score', 'condition'])
    tbl['mean delay score'] = tbl['mean delay score'].astype(float)

    plt.figure(figsize=(5.5, 5), num='Off-diagonal cross-temporal decoding scores comparison')
    sns.boxplot(data=tbl, x='condition', y='mean delay score',
                palette=[sns.color_palette("Set2")[i] for i in [0, -2]])
    sns.despine()
    plt.plot(plt.xlim(), [.5, .5], 'k--')
    plt.ylim([.4, .85])
    plt.tight_layout()


def plot_ctg(constants, scores_grand_av, time_range):
    """
    Plot the results of the cross-temporal decoding analysis as a heatmap. If plotting all trial timepoints, trial
    period boundaries will be demarcated by black horizontal and vertical lines.

    :param module constants: A Python module containing constants and configuration data for the simulation.
    :param np.ndarray scores_grand_av: Cross-temporal generalisation decoding test scores, shape:
        (n_training_timepoints, n_test_timepoints)
    :param list time_range: [t_min, t_max] where t_min and t_max correspond to the first and last trial timepoints the
        decoders were trained and tested on.
    """
    delta_t = time_range[1] - time_range[0]

    plt.figure(num='Cross-temporal decoding results')
    plt.imshow(scores_grand_av, origin='lower', cmap='RdBu_r', vmin=0, vmax=1.)
    plt.colorbar()

    plt.axhline(.5, color='k')  # origin
    plt.axvline(.5, color='k')

    if delta_t == constants.PARAMS['seq_len']:
        # all timepoints - demarcate the trial period boundaries by black lines

        plt.axhline(constants.PARAMS['trial_timepoints']['delay1_end'] - .5, color='k')
        plt.axvline(constants.PARAMS['trial_timepoints']['delay1_end'] - .5, color='k')
        plt.axhline(constants.PARAMS['trial_timepoints']['delay2_start'] - .5, color='k')
        plt.axvline(constants.PARAMS['trial_timepoints']['delay2_start'] - .5, color='k')

        plt.xticks(np.arange(0, delta_t, 5), labels=np.arange(0, delta_t, 5) + 1)
        plt.yticks(np.arange(0, delta_t, 5), labels=np.arange(0, delta_t, 5) + 1)

    else:
        plt.xticks(range(delta_t), labels=range(-1, delta_t - 1))
        plt.yticks(range(delta_t), labels=range(-1, delta_t - 1))

    plt.ylabel('Training time')
    plt.xlabel('Testing time')

    plt.tight_layout()
    return


def plot_loss_and_plateau(loss_raw, loss_clean, dLoss, plateau_ix):
    """
    Plot the raw and smoothed loss values, with the plateau timepoint indicated by a red dot.

    :param torch.Tensor loss_raw: training loss values
    :param torch.Tensor loss_clean:  smoothed training loss values
    :param torch.Tensor dLoss: loss derivative values
    :param int plateau_ix: index of the plateau timepoint

    """
    plt.figure()
    plt.subplot(211)
    plt.plot(loss_raw, 'k-', label='loss raw')
    plt.plot(loss_clean, 'g-', label='loss clean')
    plt.plot(plateau_ix, loss_clean[plateau_ix], '.r')
    plt.ylabel('Loss')
    plt.legend(loc='upper right', prop={'size': 10})

    plt.subplot(212)
    plt.plot(dLoss)
    plt.plot(plateau_ix, dLoss[plateau_ix], 'r.')
    plt.xlabel('epoch')
    plt.ylabel('dLoss')


def plot_example_retro_weights(cue1_weights, cue2_weights, r):
    """
    Scatterplot of the retrocue weights. Each dot denotes the cue1 and cue2 input weights to a given recurrent node.

    :param torch.Tensor cue1_weights: input weights from the first retrocue node to the recurrent nodes, (n_neurons, )
    :param torch.Tensor cue2_weights: analogous for the second retrocue input node
    :param float r: correlation coefficient between the two weight sets
    """
    plt.figure(figsize=(4.3, 4))
    plt.plot(cue1_weights, cue2_weights, 'o', mec='k', mfc=[0.75, 0.75, 0.75])
    plt.xlabel('cue 1 weights')
    plt.ylabel('cue 2 weights')

    x = cue1_weights.max() * .1
    y = cue2_weights.max() * .75
    plt.text(x, y, 'r = %.3f' % r)
    plt.tight_layout()
    sns.despine()


def plot_learning_dynamics_angles(constants, theta_dict):
    """
    Plot the theta angles for models evaluated at different times during training. Produces a new figure for each delay
    interval and training stage. Each figure shows the theta values for individual models and the grand averages
    (plotted as semi-transparent and opaque markers, respectively). The results are reported in Fig. 4C left in the
    manuscript.

    :param module constants: A Python module containing constants and configuration data for the simulation.
    :param dict theta_dict: A nested dictionary with the theta values for different training stages and delay intervals.
        Stage keys are superordinate to delay name keys, e.g. ai_dict['untrained']['delay1']. Each entry contains an
        array with the theta values, shape: (n_models, )
    """
    labels = theta_dict.keys()
    cols = sns.color_palette("rocket_r", len(labels))
    delay_names = ['pre', 'post']
    for s, stage in enumerate(theta_dict.keys()):
        # loop through training stages
        for d, delay in enumerate(theta_dict[stage].keys()):
            # loop through delay intervals
            plot_plane_angles_single(constants,
                                     np.radians(theta_dict[stage][delay]),
                                     delay_names[d],
                                     fig_name=f'Theta {delay} {stage}')
            ax = plt.gca()
            # change marker colour
            ax.get_children()[0].set_color(cols[s])
            ax.get_children()[1].set_color(cols[s])

            if constants.PLOT_PARAMS['save_plots']:
                plt.savefig(f"{constants.PARAMS['FIG_PATH']}'learning_dynamics_theta_{delay}_{stage}.svg")
                plt.savefig(f"{constants.PARAMS['FIG_PATH']}'learning_dynamics_theta_{delay}_{stage}.png")


def plot_learning_dynamics_AI(constants, ai_dict, dim=2):
    """
    Plot the AI for models evaluated at different times during training. Produces a new figure for each delay interval.
    Each figure shows the AI values for individual models as well as the grand averages (plotted as semi-transparent
    bars). The results are reported in Fig. 4C right in the manuscript.

    :param module constants: A Python module containing constants and configuration data for the simulation.
    :param dict ai_dict: A nested dictionary with the AI values for different training stages and delay intervals. Stage
        keys are superordinate to delay name keys, e.g. ai_dict['untrained']['delay1']. Each entry contains an array
        with the AI values, shape: (n_models, n_dims)
    :param int dim: Optional. Dimensionality of the AI estimate to be plotted. Default is 2, which plots the AI
        calculated for 2-dimensional subspaces.
    """
    stage_keys = list(ai_dict.keys())
    delay_keys = list(ai_dict[stage_keys[0]].keys())

    cols = sns.color_palette("rocket_r", len(stage_keys))
    markers = ['o', '^']
    # ms = 16

    for d, delay in enumerate(delay_keys):
        # loop through delay intervals
        plt.figure(figsize=(6.65, 5), num=f'AI learning dynamics {delay}')
        for s, stage in enumerate(stage_keys):
            # loop through training stages

            # plot individual model datapoints
            x_vals = np.ones((constants.PARAMS['n_models'],))*s
            plt.plot(x_vals, ai_dict[stage][delay][:, dim-1], markers[d], c=cols[s])

            # plot grand means as bars
            plt.bar(s,  ai_dict[stage][delay][:, dim-1].mean(), facecolor=cols[s], alpha=.2)

        plt.xticks(range(len(stage_keys)), labels=stage_keys, rotation=30)
        plt.ylabel('AI')
        plt.ylim([0, 1.1])
        sns.despine()
        plt.tight_layout()

        if constants.PLOT_PARAMS['save_plots']:
            plt.savefig(f"{constants.PARAMS['FIG_PATH']}'learning_dynamics_AI_{delay}.svg")
            plt.savefig(f"{constants.PARAMS['FIG_PATH']}'learning_dynamics_AI_{delay}.png")


def plot_example_loss_plot(plateau_ix, training_loss):
    """
    Plot training loss against epochs for an example model. Timepoints corresponding to the different training stages
    ('untrained', 'plateau', and 'trained') demarcated by coloured dots. This plot corresponds to Fig. 4A in the
    manuscript.

    :param int plateau_ix: Index of the plateau timepoint
    :param torch.Tensor training_loss: training loss values, shape (n_epochs, )
    :return:
    """
    assert len(training_loss.shape) == 1, 'training_loss should be 1D and contain average loss values for all epochs'

    labels = ['untrained', 'plateau', 'trained']
    cols = sns.color_palette("rocket_r", len(labels))

    # plot all loss
    plt.figure(figsize=(6, 4))
    plt.plot(training_loss, 'k-')

    # add dots to show the different training stages
    x = [0, plateau_ix, len(training_loss) - 1]  # x values for training stages
    y = training_loss[x]
    plt.scatter(x, y, c=cols)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    sns.despine()
    yl = plt.ylim()
    plt.ylim((0, yl[1]))
    plt.tight_layout()
