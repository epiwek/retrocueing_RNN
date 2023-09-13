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
#%% experimental set up


def plot_example_stimulus(data, cmap=sns.cubehelix_palette(start=.5, rot=-.75, as_cmap=True, reverse=True), **kwargs):
    """
    Plot the sequence of inputs corresponding to an example trial stimulus as a heatmap, with timepoints on the x-axis
    and input channels on the y-axis. Colour denotes the input activation level.
    :param data: Input array containing the sequence of inputs for an example trial.
    :type data: np.ndarray
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

#%% paired data plots


def add_jitter(data, jitter='auto'):
    """ Add small jitter values to the data to get rid of duplicate values. Jitter value is determined based on the
    number of duplicate datapoints with a given value.
    :param data: data array
    :type data: np.ndarray
    :param jitter: Optional. Custom maximum jitter value to use as the end of the range. Default is 'auto', which sets
        the range to [-.2, .2].
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
            if jitter == 'auto':
                x_jitter[dups_ix] = np.linspace(-.2, .2, n_dups)
            else:
                # custom jitter value
                x_jitter[dups_ix] = \
                    np.arange(0, n_dups*jitter, jitter) \
                    - np.mean(np.arange(0, n_dups*jitter, jitter))
    return x_jitter


def plot_paired_data(x_data, y_data, ax, colours, jitter='auto', **kwargs):
    """
    Plot paired data. Datapoints from a given model are plotted as dots of given colours and joined by a black line.
    Jitter is added to the x-values so that datapoints with the same x-value do not overlap.

    :param x_data: Array with x-values from all models (n_models, n_conditions). Can be 1-dimensional if just want to
        add jitter to some constant x-value.
    :param y_data: Array with y-values. Can be 1-dimensional if just want to add jitter to some constant x-value.
    :param ax: matplotlib.pyplot axes
    :param colours: list of colour values for the plotter
    :param jitter: Optional. Custom jitter values for all data points. Default is 'auto', which adds jitter values from
        a [-.2, .2] interval.
    :param kwargs: additional parameters to the plt.plot function
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
            
#%% 3D data and planes


def plot_geometry(ax, data_3D, plot_colours, PVEs=None, plot_outline=True, legend_on=True, custom_labels=None):
    """
    Plot the 3D datapoints corresponding to the colour representation from the two cue locations. Datapoints from each
    location are joined by a black line to form a parallelogram.

    :param ax: matplotlib.pyplot axes object to which to add the plot
    :type ax: object
    :param data_3D: Array containing the 3D coordinates for the colour representations from the two cue locations.
        Datapoints from the two locations should be stacked as rows (i.e., location 1 datapoints should correspond to
        the top half of the array). Shape: (n_datapoints, n_dims=3)
    :type data_3D: np.ndarray
    :param plot_colours: list of colours for the datapoints
    :type plot_colours: list
    :param PVEs: Optional, array containing the PVE (percent variance explained) by each PC axis. Default is None.
    :type PVEs: list or None
    :param plot_outline: Optional. If True, joins the datapoints corresponding to a single location with a black line,
        to form a parallelogram. Default is True.
    :type plot_outline: bool
    :param legend_on: Optional. If True, adds legend labels denoting the marker shape corresponding to each location.
        Default is True.
    :type legend_on: bool
    :param custom_labels: Optional. List of legend labels. Default is None, which sets the labels as 'L1' and 'L2'.
    :type custom_labels: list or None
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
            ax.plot(np.append(data_3D[ixs, 0], data_3D[loc*n_colours, 0]),
                    np.append(data_3D[ixs, 1], data_3D[loc*n_colours, 1]),
                    np.append(data_3D[ixs, 2], data_3D[loc*n_colours, 2]), 'k-')

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
    :param ax: matplotlib.pyplot axes object to which to add the plot
    :type ax: object
    :param vertices: Array of the plane vertices (n_vertices, n_dims=3)
    :type vertices: np.ndarray
    :param fc: Optional. Colour of the plotted plane. Default is black.
    :type fc: list or str
    :param alpha: Optional. Alpha (transparency) value for the plane (should be within [0, 1], where 0 is fully
        transparent and 1 fully opaque). Default is 0.2.
    :type alpha: float
    """
    assert vertices.shape[1] == 3, 'Vertices should be 3-dimensional.'
    # plot a polygon with given vertices in 3D
    ax.add_collection3d(Poly3DCollection([vertices], facecolor=fc, edgecolor=[], alpha=alpha))

# def plot_subspace(ax,points,plane_vecs,fc='k',a=0.2):
#     # plot the best-fitting plane as a quadrilateral with vertices being the projections of original points onto the plane
#
#     if (points.shape[1]!=3):
#         raise NotImplementedError('Check the shape of the data matrix - should be (n_points,3)')
#
#     # find vertices
#     n_points = points.shape[0]
#     verts = np.zeros((n_points,3))
#
#     com = np.mean(points, axis=0) # centre of mass
#
#     for i in range(n_points):
#         verts[i,:] = vops.getProjection(points[i,:]-com,plane_vecs) # get projection of demeaned 3d points
#         verts[i,:] += com #add the mean back
#
#     # sort vertices according to shortest path - so that plotted plane will be a quadrilateral
#     sorted_verts, sorting_order = vops.defPlaneShape(verts,plane_vecs)
#     #sorted_verts, sorting_order = vops.sortByVecAngle(verts)
#     #sorted_verts = verts
#     # plot the best-fit plane
#     plot_plane(ax,sorted_verts,fc,a)
#     #return verts, sorted_verts


def plot_subspace(ax, points, plane_vecs, fc='k', a=0.2):
    """
    Plot the best-fitting subspace as a quadrilateral with vertices being the projections of original points onto the
    plane. Sort the datapoints so that the plotted plane will be a convex quadrilateral.

    :param ax:
    :param points:
    :param plane_vecs:
    :param fc:
    :param a:
    :return:
    """
    # plot the best-fitting plane

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


def plot_full_geometry(constants, model_numbers, subspaces, fig_names=None):
    # rename to plot geometry
    n_subplots = constants.PARAMS['n_delays']
    fig_size = (n_subplots * 5, 5)
    fig_list = []
    for model in model_numbers:

        fig = plt.figure(figsize=fig_size, num=f"Model {model}")

        if fig_names is not None:
            # set figure window name
            fig.canvas.manager.set_window_title(fig_names[model])

        fig_list.append(fig)

        for delay in range(1, constants.PARAMS['n_delays']+1):
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
                          fc='k', a=0.2)
            plot_subspace(ax,
                          subspaces[f"delay{delay}"][model].plane2.coords_3d,
                          subspaces[f"delay{delay}"][model].plane2.plane_vecs_xy,
                          fc='k', a=0.2)

            ax.set_title('Angle: %.1f' % subspaces[f"delay{delay}"][model].theta_degrees)
            helpers.equal_axes(ax)
    return fig_list


def plot_PVEs_3D(constants, PVEs_3D, fig_name=None):
    """
    Plot the percent of variance explained (PVE) in the data by the 3 principal components (PC). Values for
    individual models (for each timepoint examined) plotted as dots joined by black lines,  group-wise averages as
    bars.
    :param constants: Experimental constants file.
    :type constants: module
    :param PVEs_3D: Percent of variance explained values for all models and delays in the format (n_models, n_delays, n_dims=3)
    :type PVEs_3D: numpy.ndarray
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

    # plt.plot(np.mean(np.sum(pves,0),0),'_-r')
    # plt.plot(np.mean(np.sum(pves,0),0),marker=1,c='r')
    # plt.plot(np.mean(np.sum(pves,0),0),marker=0,c='r')
    plt.tight_layout()
    sns.despine()


#%% angle plots

# % polar plot of angles
def plot_polar_scatter(ax, data, r, **kwargs):
    ax.plot(data, r, **kwargs)


def plot_polar_scatter_paired(ax, data, r, **kwargs):
    for i in range(len(data)):
        ax.plot(data[i, :], np.tile(r[i], (data.shape[1], 1)), **kwargs)


def format_polar_axes(ax, r=None):
    ax.grid(False)
    ax.tick_params(axis='x', which='major', pad=14)
    if np.all(r is None):
        # custom radius values
        ax.set_ylim([0, r.max() + .05 * r.max()])
    else:
        ax.set_ylim([0, 1.05])
    ax.set_yticks([])


def plot_plane_angles_multiple(constants, angles_radians, r=None, paired=True, cu=False, custom_labels=None, fig_name=None):
    if r is None:
        # if r == None
        r = np.ones((constants.PARAMS['n_models'],)) # radius values for the datapoints
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

    # if np.sum(pct_nans) != 0:
    #     for i in range(n_cats):
    #         txt = "{category} data contains {pct: .1f} % NaNs"
    #         print(txt.format(category=labels[i], pct=pct_nans[i]))


def plot_plane_angles_single(constants, angles_radians, cond, r=None, fig_name=None):
    pal = sns.color_palette("dark")
    if np.all(r == None):
        # if r == None
        r = np.ones((constants.PARAMS['n_models'],))
        rr = 1
    else:
        rr = r.max()  # radius value for the mean

    if cond == 'pre':
        ix = 3
        cols = pal[ix]
        markers = 'o'
    elif cond == 'post':
        ix = 0
        cols = pal[ix]
        markers = '^'
    elif cond == 'cu':
        markers = 'o'
        cols = 'k'
    elif cond == 'bt':
        markers = '^'
        cols = 'k'
    elif cond == 'bs':
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

    # pct_nans = 100 * (len(angles_radians) - len(nonnan_ix)) / len(angles_radians)
    # if pct_nans != 0:
    #     print('Data contains %.1f %% NaNs' % pct_nans)


# %% other plots
def shadow_plot(ax, x, y, precalc=False, alpha=0.3, **kwargs):
    if precalc:
        y_mean = y[0]
        y_sem = y[1]
    else:
        y_mean = np.mean(y, axis=0)
        y_sem = np.std(y, axis=0) / np.sqrt(len(y))
    # pdb.set_trace()
    H1 = ax.fill_between(x, y_mean + y_sem, y_mean - y_sem,
                         alpha=alpha, **kwargs)
    H2 = ax.plot(x, y_mean, **kwargs)
    return H1, H2


def plot_err_distr(binned_errors, bin_centres, b, fitted_params, sem=None, ax=None, c='k', dashed=False, fig_name=None):
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
    """ High-level plotter for the error distributions. For experiments 1-3, plots the error distributions for each
    test condition in a separate subplot. For experiment 4, plots the error distributions from valid and invalid
    trials on one plot, as green and red, respectively."""

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

    :param mixture_param_data_dict : A dictionary containing mixture parameter data.
        It should have the following keys:
        - 'model_nr' (array): An Array with model number for each observation.
        - parameter-specific keys (e.g. 'K'), each containing a pandas dataframe with 'trial_type', 'condition' and
            parameter-specific columns.
    :type mixture_param_data_dict: dict


    :return: None
    """
    # extract the names of the mixture parameters from the data dictionary
    mixture_params = [key for key in mixture_param_data_dict.keys() if key != 'model_nr']
    for param in mixture_params:
        sns.catplot(x='trial_type', y=param, hue='condition', data=mixture_param_data_dict[param],
                    kind="point", markers=["^", "o"], units=mixture_param_data_dict['model_nr'], dodge=True, ci=68,
                    palette=sns.color_palette("Set2")[2:0:-1])

        plt.xlabel('trial type')


#%%

def plot_CDI(constants, CDI, log_transform=True):
    '''
    Plot the CDI metric. Plot style depends on the experiment number. For experiment 3, this is a barplot with bar 
    height corresponding to the mean across models and error bars corresponding to SEM. Bars correspond to conditions, 
    namely: pre-cue, cued, uncued, probed and unprobed colours.
    For experiment 1, 
    

    Parameters
    ----------
    constants : TYPE
        DESCRIPTION.
    CDI : TYPE
        DESCRIPTION.
    log_transform : TYPE, optional
        DESCRIPTION. The default is True.
    save_fig : TYPE, optional
        DESCRIPTION. The default is True.
    trials : TYPE, optional
        DESCRIPTION. The default is 'valid'.

    Returns
    -------
    None.

    '''
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

    fig = plt.figure(figsize=(6.65, 5))
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
                if cond in ['probed_invalid', 'unprobed_invalid']: #np.logical_or(i == probed_invalid_ix, i == unprobed_invalid_ix):
                    # plot hatched bars
                    ax.bar(x_vals[cond].mean(), CDI[cond].mean(), yerr=CDI[cond].sem(), ec=cols[cond], facecolor='w',
                           capsize=2, hatch='///', alpha=.25, width=.25, label=cond)
                else:
                    ax.bar(x_vals[cond].mean(), CDI[cond].mean(), yerr=CDI[cond].sem(), ec=cols[cond],
                           facecolor=cols[cond], capsize=2, alpha=.2, width=.25,
                           label=[] if cond is 'pre-cue' else cond)
                # else:
                #     # data label
                #     ax.bar(x_vals[cond], means[cond], yerr=sem[cond], ec=cols[cond], capsize=2,
                #            alpha=.2, width=.25, label=cond)

        else:
            # plot bars
            # raise NotImplementedError('Not tested')

            for cond in list(CDI.columns):

                # if cond is 'pre-cue':
                #     # no data label
                #     ax.bar(x_vals[cond], means[cond], yerr=sem[cond], ec=cols[cond], capsize=2,
                #            alpha=.2, width=.25)
                # else:
                ax.bar(x_vals[cond].mean(), CDI[cond].mean(), yerr=CDI[cond].sem(), ec=cols[cond], facecolor=cols[cond],
                       capsize=2, alpha=.2, width=.25, label=cond if cond is not 'pre-cue' else None)
        ax.set_ylim([-1.5, 9])

    else:
        if constants.PARAMS['experiment_number'] > 1:
            raise NotImplementedError('Not tested for Experiments 2 and 3')

        for cond in list(CDI.columns):
            # if i == 0:
            #     # no data label
            #     ax.bar(x_vals[i], means[i], facecolor=cols[i], alpha=.2, width=.25)
            # else:
            #     ax.bar(x_vals[i], means[i], facecolor=cols[i], alpha=.2, width=.25,
            #            label=data_labels[i])
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
