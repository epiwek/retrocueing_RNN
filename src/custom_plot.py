#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 17:01:58 2021

@author: emilia
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import src.vec_operations as vops
from scipy.stats import vonmises
# import rep_geom
import src.helpers as helpers

import pdb

#%%

def plot_settings():
    plt.rcParams.update({'font.size': 22})
    plt.rcParams['svg.fonttype'] = 'none'
    sns.set_context("notebook", font_scale=1.5)
    sns.set_style("ticks")

#%% experimental set up


def plot_example_stimulus(data, **kwargs):
    # plt.figure()
    
    # if len(data.shape)<3:
    #     # plot a specific trial
    #     plt.imshow(data,*kwargs)
    # else:
    #     # pick a random trial to plot
    #     ix = np.random.randint(data.shape[1])
    #     plt.imshow(data[:,ix,:],**kwargs)
    # plt.colorbar
    
    fig, ax = plt.subplots()
    
    if len(data.shape) < 3:
        # plot a specific trial
        cax = ax.imshow(data, **kwargs)
    else:
        # pick a random trial to plot from the full array
        ix = np.random.randint(data.shape[1])
        cax = ax.imshow(data[:, ix, :], **kwargs)
    fig.colorbar(cax, ticks=[0,1])
    return fig, ax, cax

#%% paired data


def add_jitter(data, jitter='auto'):
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
    # determine x jitter
    x_jitter = np.zeros(y_data.shape)
    if len(y_data.shape) == 1:
        x_jitter = np.expand_dims(x_jitter,0)
        y_data = np.expand_dims(y_data,0)
        x_data = np.expand_dims(x_data,0)
    
    for i in range(2):
        x_jitter[:,i] = add_jitter(y_data[:, i], jitter=jitter)
    
    #plot
    # loop over samples (models)
    for i in range(y_data.shape[0]):
        ax.plot(x_data[i, :] + x_jitter[i, :], y_data[i, :], 'k-', **kwargs)
        # loop over the pair
        for j in range(y_data.shape[1]):
            ax.plot(x_data[i, j] + x_jitter[i, j], y_data[i, j], 'o', color=colours[j], **kwargs)
            
#%% 3D data and planes

def plot_geometry(ax, Z, pca, plot_colours, plot_outline = True, legend_on = True, **kwargs):
    """
    Plot 3D data for one or two subspaces.
    
    Parameters
    ----------
    
    ax :  

        

    """
    n_colours = len(plot_colours)
    # plot the parallelogram defined by colours at location 1
    if plot_outline:
        ax.plot(np.append(Z[:n_colours,0],Z[0,0]),
              np.append(Z[:n_colours,1],Z[0,1]),
              np.append(Z[:n_colours,2],Z[0,2]),'k-',**kwargs)
    ax.scatter(Z[0,0],Z[0,1], Z[0,2],marker='o',s = 40,
              c='k',label='loc1',**kwargs)
    ax.scatter(Z[:n_colours,0],Z[:n_colours,1],
              Z[:n_colours,2],marker='o',s = 40,c=plot_colours,**kwargs)
  
    # repeat for loc 2 - if data supplied
    if Z.shape[0]>n_colours:
        if plot_outline:
            ax.plot(np.append(Z[n_colours:,0],Z[n_colours,0]),
                  np.append(Z[n_colours:,1],Z[n_colours,1]),
                  np.append(Z[n_colours:,2],Z[n_colours,2]),'k-',**kwargs)
        ax.scatter(Z[-1,0],Z[-1,1], Z[-1,2],marker='s',s = 40,
                  c='k',label='loc2',**kwargs)
        ax.scatter(Z[n_colours:,0],Z[n_colours:,1],
                  Z[n_colours:,2],marker='s',s = 40,c=plot_colours,**kwargs)
        
    # add the PVEs for each PC to the corresponding axis - if data supplied
    if pca:
        ax.set_xlabel('PC1 ['+str(np.round(pca.explained_variance_ratio_[0]*100,1))+'%]')
        ax.set_ylabel('PC2 ['+str(np.round(pca.explained_variance_ratio_[1]*100,1))+'%]')
        ax.set_zlabel('PC3 ['+str(np.round(pca.explained_variance_ratio_[2]*100,1))+'%]')
    
    if legend_on:
        ax.legend()


def plot_plane(ax,verts,fc='k',a=0.2):
    # plot a polygon with given vertices in 3D
    ax.add_collection3d(Poly3DCollection([verts],facecolor=fc,edgecolor=[],alpha=a))
    
    
def plot_subspace(ax,points,plane_vecs,fc='k',a=0.2):
    # plot the best-fitting plane as a quadrilateral with vertices being the projections of original points onto the plane 
    
    if (points.shape[1]!=3):
        raise NotImplementedError('Check the shape of the data matrix - should be (n_points,3)')
    
    # find vertices
    n_points = points.shape[0]
    verts = np.zeros((n_points,3))
    
    com = np.mean(points, axis=0) # centre of mass
    
    for i in range(n_points):
        verts[i,:] = vops.getProjection(points[i,:]-com,plane_vecs) # get projection of demeaned 3d points
        verts[i,:] += com #add the mean back
    
    # sort vertices according to shortest path - so that plotted plane will be a quadrilateral
    sorted_verts, sorting_order = vops.defPlaneShape(verts,plane_vecs)
    #sorted_verts, sorting_order = vops.sortByVecAngle(verts)
    #sorted_verts = verts
    # plot the best-fit plane
    plot_plane(ax,sorted_verts,fc,a)
    #return verts, sorted_verts


def plot_plane_old(ax,Y_l,points,scale=1.0,fc='k',a=0.2):
    # plot the best fitting plane as a quadrilateral, with vertices being some
    #scaled distance from the centre-of-mass of the original point set
    # (plots look better if e.g. one set of points forms a much denser cloud than the other)

    com = np.mean(points,axis=0) # centre of mass
    #sorted_verts, sorting_order = vops.sortByPathLength(points)
    # set the scale to be equal to largest distance between points
    #scale = np.linalg.norm(sorted_verts[0]-sorted_verts[2])
    Y_l.components_ *= scale
    # plot the best-fit plane
    verts = np.array([com-Y_l.components_[0,:],com-Y_l.components_[1,:],
                    com+Y_l.components_[0,:],com+Y_l.components_[1,:]])
    #verts *= scale
    ax.add_collection3d(Poly3DCollection([verts],facecolor=fc,edgecolor=[],alpha=a))
    
#%% other plots
def shadow_plot(ax,x,y, precalc = False, alpha = 0.3, **kwargs):
    
    if precalc:
        y_mean = y[0]
        y_sem = y[1]
    else:
        y_mean = np.mean(y,axis=0)
        y_sem = np.std(y,axis=0)/np.sqrt(len(y))
    # pdb.set_trace()
    H1 = ax.fill_between(x,y_mean+y_sem,y_mean-y_sem,
                    alpha = alpha, **kwargs)
    H2 = ax.plot(x,y_mean,**kwargs)
    return H1, H2


def plot_PVEs_3D(constants, PVEs_3D):
    """
    Plot the percent of variance explained (PVE) in the data by the 3 principal components (PC). Values for
    individual models (for each timepoint examined) plotted as dots joined by black lines,  group-wise averages as
    bars.
    :param constants: :param PVEs_3D: :return:
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
    plt.figure(figsize=(6.5, 5))
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


def plot_err_distr(binned_errors, bin_centres, b, fitted_params, sem=None, ax=None, c='k'):
    pdf_x = np.linspace(-b, b, 100)  # x vals for the fitted von Mises pdf
    pdf_y = vonmises.pdf(pdf_x,
                         fitted_params['kappa'],
                         fitted_params['mu'],
                         fitted_params['scale'])
    if ax is None:
        # create a new figure
        plt.figure()
        ax = plt.subplot(111)

    if sem is not None:
        # add error bars for datapoints
        ax.errorbar(bin_centres, binned_errors, yerr=sem, fmt='none', ecolor=c)
    ax.plot(bin_centres, binned_errors, 'o', mec=c, mfc='w')
    ax.plot(pdf_x, pdf_y, '-', c=c, lw=2, label='fit')
    x_ticks = [-100, 0, 100] # to match the monkey figure
    ax.set_xticks(np.radians(x_ticks))
    ax.set_xticklabels(x_ticks)
    ax.set_xlabel('angular error (degrees)')
    ax.set_ylabel('density of trials')
    sns.despine()
    plt.show()


def plot_all_error_data(constants, test_conditions, all_results):
    """ High-level plotter for the error distributions. For experiments 1-3, plots the error distributions for each
    test condition in a separate subplot. For experiment 4, plots the error distributions from valid and invalid
    trials on one plot, as green and red, respectively."""

    if constants.PARAMS['cue_validity'] == 1:
        # experiments 1-3 and experiment 4, cue_validity = 1
        # plot results as different panels
        n_conditions = len(test_conditions)
        fig, axes = plt.subplots(1, n_conditions, sharex='all', sharey='all')
        fig.set_size_inches((12.8, 3.4))
        for i, condition in enumerate(test_conditions):
            plot_err_distr(all_results[condition]['mean_errs'],
                           all_results[condition]['bin_centres'],
                           all_results[condition]['bin_max'],
                           all_results[condition]['fitted_params'],
                           sem=all_results[condition]['sem_errs'],
                           ax=axes[i])
            plt.title(condition)
        plt.tight_layout()
    else:
        # plot valid and invalid trials on one plot
        plt.figure()
        ax = plt.subplot(111)
        # plot valid trials
        conditions = ['valid_trials', 'invalid_trials']
        plot_colours = ['g', 'r']
        for condition, colour in zip(conditions, plot_colours):
            plot_err_distr(all_results[condition]['mean_errs'],
                           all_results[condition]['bin_centres'],
                           all_results[condition]['bin_max'],
                           all_results[condition]['fitted_params'],
                           sem=all_results[condition]['sem_errs'],
                           ax=ax,
                           c=colour)


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
