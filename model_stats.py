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
import vec_operations as vops
import pickle
import pycircstat
import constants
import seaborn as sns
import pdb
import helpers

#%% plots
def plot_PVEs_3D(PVEs_3D):    
    pal = sns.color_palette("dark")
    if constants.PARAMS['experiment_number'] == 3:
        inds = [3,0,-3]
        markers = ['o','^','s']
    else:
        inds = [3,0]
        markers = ['o','^']
    cols = [pal[ix] for ix in inds]
    ms = 10
    plt.figure(figsize=(6.5,5))
    ax = plt.subplot(111)
    for model in range(constants.PARAMS['n_models']):
                
        ax.plot(range(len(inds)),PVEs_3D[model,:,:].sum(-1),'k-',alpha=0.2) # sum PC1-3 
        for i in range(len(inds)):
            ax.plot(i,PVEs_3D[model,i,:].sum(),marker=markers[i],
                    color = cols[i],alpha=0.2,markersize=ms)
    
    
    plt.plot(range(len(inds)),PVEs_3D.sum(-1).mean(0),'k-')
    for i in range(len(inds)):
        ax.plot(i,PVEs_3D[:,i,:].sum(-1).mean(),marker=markers[i],
                color = cols[i],markersize=ms)
    
    
    # plt.plot(np.mean(np.sum(pves,0),0),'_-r')   
    # plt.plot(np.mean(np.sum(pves,0),0),marker=1,c='r')
    # plt.plot(np.mean(np.sum(pves,0),0),marker=0,c='r')  
    
    # plt.yticks(np.arange(0.8,1.01,0.05))
    plt.ylabel('PVE by first 3 PCs')
    
    if constants.PARAMS['experiment_number'] == 3:
        plt.xlim((-0.5,2.5))
        plt.xticks([0,1,2],['pre-cue','post-cue','probe'])
    else:
        plt.xlim((-0.5,1.5))
        plt.xticks([0,1],['pre-cue','post-cue'])
    
    # plt.plot(np.mean(np.sum(pves,0),0),'_-r')   
    # plt.plot(np.mean(np.sum(pves,0),0),marker=1,c='r')
    # plt.plot(np.mean(np.sum(pves,0),0),marker=0,c='r')    
    plt.tight_layout()
    

#% polar plot of angles
def plot_polar_scatter(ax,data,r,**kwargs):
    ax.plot(data,r,**kwargs)
    

def plot_polar_scatter_paired(ax,data,r,**kwargs):
    for i in range(len(data)):
        ax.plot(data[i,:],np.tile(r[i],(data.shape[1],1)),**kwargs)


def format_polar_axes(ax,r=None):
    ax.grid(False)
    ax.tick_params(axis='x', which='major', pad=14)
    if np.all(r == None):
        # custom radius values  
        ax.set_ylim([0,r.max()+.05*r.max()])
    else:
        ax.set_ylim([0,1.05])
    ax.set_yticks([])
    
    
def plot_plane_angles_multiple(constants,angles_radians,r=None,paired=True,cu=False,expt3=False):
    if np.all(r == None):
        # if r == None
        r = np.ones((constants.PARAMS['n_models'],))
        rr=1
    else:
        rr=r.max() # radius value for the mean
    
    if cu:
        # cued vs uncued geometry
        markers = ['^','s']
        labels = ['L1 cued','L2 cued']
        cols = ['k','k']
        alphas = [.2, .2]
    elif constants.PARAMS['experiment_number']==3:
        # experiment 3 cued
        pal = sns.color_palette("dark")
        inds = [3,0,-3]
        cols = [pal[ix] for ix in inds]
        markers = ['o','^','s']
        labels = ['pre-cue','post-cue','post-probe']
        alphas = [.2, .2, .2]
    else:
        # cued geometry
        pal = sns.color_palette("dark")
        inds = [3,0]
        cols = [pal[ix] for ix in inds]
        markers = ['o','^']
        labels = ['pre','post']
        alphas = [.2, .2]
    ms = 12
    
    n_cats = angles_radians.shape[-1]
    pct_nans = np.zeros((n_cats,))
    if paired:
        fig = plt.figure(figsize=(7.9,5))
        ax = fig.add_subplot(111,polar=True)
        
        # plot all datapoints
        for i in range(n_cats):
            plot_polar_scatter(ax,angles_radians[:,i],r,
                               marker=markers[i],color=cols[i],
                               ls='None',
                               alpha=alphas[i],markersize=ms)
            # add grand means (need to remove nans)
            nonnan_ix = np.where(np.invert(np.isnan(angles_radians[:,i])))[0]
            plot_polar_scatter(ax,pycircstat.descriptive.mean(angles_radians[nonnan_ix,i]),
                               rr,marker=markers[i],color=cols[i],
                               markersize=ms,label=labels[i])
            
            pct_nans[i] = 100*(len(angles_radians) - len(nonnan_ix))/len(angles_radians)
        # join the values corresponding to an individual model 
        plot_polar_scatter_paired(ax,angles_radians,r,c='k',ls='-',alpha=0.2)   
        format_polar_axes(ax,r)
        ax.legend(bbox_to_anchor=(1.4,1))
    else:
        # separate subplots for each category
        fig = plt.figure(figsize=(7.9*n_cats,5))
        for i in range(n_cats):
            ax = fig.add_subplot(1,n_cats,i+1,polar=True)
            # plot all datapoints
            plot_polar_scatter(ax,angles_radians[:,i],
                               r,marker=markers[i],ls='None',color=cols[i],
                               alpha=alphas[i],markersize=ms)
            # add grand means (need to remove nans)
            nonnan_ix = np.where(np.invert(np.isnan(angles_radians[:,i])))[0]
            plot_polar_scatter(ax,pycircstat.descriptive.mean(angles_radians[nonnan_ix,i]),
                               rr,marker=markers[i],color = cols[i],markersize=ms,label=labels[i])
            format_polar_axes(ax,r)
            ax.legend(bbox_to_anchor=(1.4,1))
            pct_nans[i] = 100*(len(angles_radians) - len(nonnan_ix))/len(angles_radians)
        
    plt.tight_layout()
    
    if np.sum(pct_nans) != 0:
        for i in range(n_cats):
            txt = "{category} data contains {pct: .1f} % NaNs"
            print(txt.format(category=labels[i],pct=pct_nans[i]))
    

    
    
def plot_plane_angles_single(constants,angles_radians,cond,r=None):
    pal = sns.color_palette("dark")
    if np.all(r == None):
        # if r == None
        r = np.ones((constants.PARAMS['n_models'],))
        rr = 1
    else:
        rr=r.max() # radius value for the mean
    
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
        

    ms = 12
    
    fig = plt.figure(figsize=(7.9,5))
    ax = fig.add_subplot(111,polar=True)
        
    # plot all datapoints
    plot_polar_scatter(ax,angles_radians,
                       r,marker=markers,ls='None',color=cols,alpha=0.2,markersize=ms)
    # add grand means (need to remove nans)
    nonnan_ix = np.where(np.invert(np.isnan(angles_radians)))[0]
    plot_polar_scatter(ax,pycircstat.descriptive.mean(angles_radians[nonnan_ix]),
                       rr,marker=markers,color=cols,markersize=ms)
    format_polar_axes(ax,r)
    plt.tight_layout()
    
    pct_nans = 100*(len(angles_radians) - len(nonnan_ix))/len(angles_radians)
    if pct_nans != 0:
        print('Data contains %.1f %% NaNs' %pct_nans)
    


        
    
    
    
    
    
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
    
    inds = [3,0]
    markers = ['o','^']
    cols = [pal[ix] for ix in inds]
    ms = 10
    
    
    conditions = ['pre-cue','post-cue']
    delay2_lengths = np.arange(6)
    
    common_path = constants.PARAMS['BASE_PATH']\
        + 'data_vonMises/experiment_4/'
          
    
    n_models = 30
    all_angles = np.empty((n_models,len(conditions),len(delay2_lengths)))
    angles_circ_mean = np.empty((len(conditions),len(delay2_lengths)))
    
    plt.figure(figsize=(7,5))
    ax1 = plt.subplot(111)
    ax1.set_thetamin(0)
    ax1.set_thetamax(180)
    ax1.set_xticks(np.radians(np.linspace(0,180,7)))
    jitter = .125
    # ax2 = plt.subplot(122,sharey=ax1,sharex=ax1) 
    for j,dl in enumerate(delay2_lengths):
        # load data
        load_path = common_path + 'delay2_' + str(dl) + 'cycles/'\
                 +'sigma' + str(constants.PARAMS['sigma'])\
                    +'/kappa' + str(constants.PARAMS['kappa_val'])\
                    +'/nrec' + str(constants.PARAMS['n_rec'])\
                        +'/lr' + str(constants.PARAMS['learning_rate']) + '/'\
                            +'pca_data/valid_trials/'
        all_angles[:,:,j] = pickle.load(open(load_path+'all_plane_angles.pckl','rb'))
        angles_circ_mean[:,j] = pycircstat.descriptive.mean(np.radians(all_angles[:,:,j]),axis=0)
        angles_circ_mean[:,j] = np.degrees(angles_circ_mean[:,j])
        
        ax1.plot(np.ones((n_models,))*dl-jitter,all_angles[:,0,j],
                 marker=markers[0],ls='',color=cols[0],alpha=.2,markersize=ms)
        
        ax1.plot(np.ones((n_models,))*dl+jitter,all_angles[:,1,j],
                 marker=markers[1],ls='',color=cols[1],alpha=.2,markersize=ms)
    
    # add means
    ax1.plot(delay2_lengths-jitter,angles_circ_mean[0,:],
             marker=markers[0],color=cols[0],markersize=ms,label='pre-cue')
    ax1.plot(delay2_lengths+jitter,angles_circ_mean[1,:],
             marker=markers[1],color=cols[1],markersize=ms,label='post-cue')
    
    ax1.set_ylim((all_angles.min()-ms),all_angles.max()+ms)
    ax1.set_xticks(delay2_lengths)
    
    ax1.set_ylabel('Cued plane angle [°]')
    ax1.set_xlabel('Post-cue delay length [cycles]')
    
    ax1.legend(bbox_to_anchor=(.9,.9))
    # ax1.set_title('Pre-cue')
    # ax2.set_title('Post-cue')
    
    
    plt.tight_layout()
    
    plt.savefig(common_path + 'compare_cued_angles_sigma' + str(constants.PARAMS['sigma']) + '.png')


#%% circular stats 

def get_descriptive_stats_angles_cued(angles_radians):
    mean_pre = pycircstat.descriptive.mean(angles_radians[:,0])
    mean_pre = helpers.wrap_angle(mean_pre) # wrap to [-pi,pi]
    # ci_pre = pycircstat.descriptive.mean_ci_limits(angles_radians[:,0], ci=.95)
    
    mean_post = pycircstat.descriptive.mean(angles_radians[:,1])
    mean_post = helpers.wrap_angle(mean_post)
    # ci_post = pycircstat.descriptive.mean_ci_limits(angles_radians[:,1], ci=.95)
    
    # print('pre-cue mean ± 95 CI: %.2f ± %.2f' %(np.degrees(mean_pre),np.degrees(ci_pre)))
    # print('post-cue mean ± 95 CI: %.2f ± %.2f' %(np.degrees(mean_post),np.degrees(ci_post)))
    
    print('pre-cue mean: %.2f ' %(np.degrees(mean_pre)))
    print('post-cue mean: %.2f ' %(np.degrees(mean_post)))
    

def get_descriptive_stats_pa_cued(pa_radians):
    
    nonnan_ix = np.where(np.invert(np.isnan(pa_radians[:,0])))[0]
    mean_pre = pycircstat.descriptive.mean(pa_radians[nonnan_ix,0])
    mean_pre = helpers.wrap_angle(mean_pre) # wrap to [-pi,pi]
    # ci_pre = pycircstat.descriptive.mean_ci_limits(pa_radians[nonnan_ix,0], ci=.95)
    
    nonnan_ix = np.where(np.invert(np.isnan(pa_radians[:,1])))[0]
    mean_post = pycircstat.descriptive.mean(pa_radians[nonnan_ix,1])
    mean_post = helpers.wrap_angle(mean_post)
    ci_post = pycircstat.descriptive.mean_ci_limits(pa_radians[nonnan_ix,1], ci=.95)
    
    # print('Pa: pre-cue mean ± 95 CI: %.2f ± %.2f' %(np.degrees(mean_pre),np.degrees(ci_pre)))
    print('Pa: pre-cue mean: %.2f ' %(np.degrees(mean_pre)))
    print('Pa : post-cue mean ± 95 CI: %.2f ± %.2f' %(np.degrees(mean_post),np.degrees(ci_post)))
    
    
def get_inf_stats_angles_cued(angles_radians):
    # plane angles are defined on the [-180,180] interval
    
    # pre-cue angles
    # test for non-uniform distribution around 90
    # note: data is bimodal, with peaks at + and -90 degrees
    # v-test assumes a unimodal or uniform distribution over the *entire* circle
    # as the sign of the angle does not matter for this contrast, we transform
    # the data first by taking the absolute value (to force it to be unimodal),
    # and then by multiplying by 2 (to extend the range to [0,360])
    p_pre, v_pre = pycircstat.tests.vtest(np.abs(angles_radians[:,0])*2,np.radians(90)*2)
    print('V-test for uniformity/mean=90 :')
    print('    v-stat = %.3f, p = %.3f' %(v_pre,p_pre))

    # post-cue angles
    # test for non-uniform distribution around 0 - here the distribution of the
    # data is unimodal, therefore we do not transform it 
    p_post, v_post = pycircstat.tests.vtest(angles_radians[:,1], 0)
    print('V-test for uniformity/mean=0 :')
    print('    v-stat = %.3f, p = %.3f' %(v_post,p_post))
    
    

    # test for a significant difference in angles
    # here again we have a bimodal distribution
    # to make the data appropriate for the mtest, we first take the absolute of
    # the pre-post difference, and then stretch it to the [0,360] range by 
    # multiplying by 2
    angle_diff_signed = angles_radians[:,0] - angles_radians[:,1]
    angle_diff = np.abs(angle_diff_signed)*2
    diff = pycircstat.tests.mtest(angle_diff,0)
    diff_mean = np.degrees(diff[1]/2) # divide by 2 to go back to original range
    diff_result = diff[0]
    diff_CI = ((np.degrees(diff[2][1])-np.degrees(diff[2][0]))/2)/2 # same here
    # diff_SEM = (np.diff(diff_CI)/2)/1.96
    
    print('circular one-sample t-test for angular difference ~=0 :')
    print('     H = %d, mean = %.3f, CI = %.3f' %(diff_result[0],diff_mean,diff_CI))


def get_inf_stats_angles_expt3(angles_radians):
    # pre-cue angles
    # test for non-uniform distribution around 90
    p_pre, v_pre = pycircstat.tests.vtest(np.abs(angles_radians[:,0])*2,np.radians(90)*2)
    print('V-test for uniformity/mean=90 :')
    print('    v-stat = %.3f, p = %.3f' %(v_pre,p_pre))
    
    p_post, v_post = pycircstat.tests.vtest(angles_radians[:,1], 0)
    print('V-test for uniformity/mean=0 :')
    print('    v-stat = %.3f, p = %.3f' %(v_post,p_post))
        
    nonnan_ix = np.where(np.invert(np.isnan(angles_radians[:,2])))[0]
    p_probe, v_probe = pycircstat.tests.vtest(angles_radians[nonnan_ix,2], 0)
    print('V-test for uniformity/mean=0 :')
    print('    v-stat = %.3f, p = %.3f' %(v_probe,p_probe))


def get_inf_stats_angles_uncued(angles_radians):
    p_val, z_stat = pycircstat.tests.rayleigh(angles_radians)
    print('Rayleigh test for uniformity of uncued post-cue angles :')
    print('    z-stat = %.3f, p = %.3f' %(z_stat,p_val))


def get_inf_stats_angles_cued_vs_uncued(angles_radians):
    p, v = pycircstat.tests.vtest(angles_radians, np.pi)
    print('V-test for uniformity/mean=90 :')
    print('    v-stat = %.3f, p = %.3f' %(v,p))
    
    if p >= .05:
        # follow up with a Rayeigh test + mean
        p_val, z_stat = pycircstat.tests.rayleigh(angles_radians)
        m = pycircstat.mean(angles_radians)
        print('Rayleigh test for uniformity of cued/uncued theta:')
        print('    z-stat = %.3f, p = %.3f, mean = %.3f' %(z_stat,p_val,m))
  
    
def get_inf_stats_pa_cued(pa_radians):
    # test pre-cue - no specific hypothesis
    # need to remove nans first
    nonnan_ix = np.where(np.invert(np.isnan(pa_radians[:,0])))[0]
    p_val, z_stat = pycircstat.tests.rayleigh(pa_radians[nonnan_ix,0])
    print('Rayleigh test for uniformity of cued pre-cue phase alignment:')
    print('    z-stat = %.3f, p = %.3f, N = %d' %(z_stat,p_val,len(nonnan_ix)))
    
    # test post-cue - mean of 0
    nonnan_ix = np.where(np.invert(np.isnan(pa_radians[:,1])))[0]
    p, v = pycircstat.tests.vtest(pa_radians[nonnan_ix,1],0)
    
    print('V-test for uniformity/mean=0 for post-cue phase alignment:')
    print('    v-stat = %.3f, p = %.3f, N = %d' %(v,p,len(nonnan_ix)))
    

def get_inf_stats_pa_uncued(pa_radians):
    # test uncued phase alignment in the post-cue delay
    # need to remove nans first
    nonnan_ix = np.where(np.invert(np.isnan(pa_radians)))[0]
    p_val, v = pycircstat.tests.vtest(pa_radians[nonnan_ix],0)
    # print('Rayleigh test for uniformity of uncued post-cue phase alignment:')
    # print('    z-stat = %.3f, p = %.3f, N = %d' %(z_stat,p_val,len(nonnan_ix)))
    print('V test for cued/uncued post-cue phase alignment:')
    print('    v-stat = %.3f, p = %.3f, N = %d' %(v,p_val,len(nonnan_ix)))

    
    
def get_inf_stats_pa_cued_vs_uncued(pa_radians):
    # test uncued phase alignment in the post-cue delay
    # need to remove nans first
    nonnan_ix = np.where(np.invert(np.isnan(pa_radians)))[0]
    p_val, v = pycircstat.tests.vtest(pa_radians[nonnan_ix],0)
    # print('Rayleigh test for uniformity of cued/uncued post-cue phase alignment:')
    # print('    z-stat = %.3f, p = %.3f, N = %d' %(z_stat,p_val,len(nonnan_ix)))

    print('V test for cued/uncued post-cue phase alignment:')
    print('    v-stat = %.3f, p = %.3f, N = %d' %(v,p_val,len(nonnan_ix)))

    
    
def run_plane_angles_analysis(constants):
    # load data
    load_path = constants.PARAMS['FULL_PATH'] + 'pca_data/valid_trials/'
    angles = pickle.load(open(load_path+'all_theta.pckl','rb'))
    angles_radians = np.radians(angles)
    PVEs_3D = pickle.load(open(load_path+'all_PVEs_3D.pckl','rb'))
    
    # plot PVEs
    plot_PVEs_3D(PVEs_3D)
    plt.savefig(constants.PARAMS['FIG_PATH']+'PVEs_3D.png')
    # plot angles
    plot_plane_angles_multiple(constants,angles_radians)
    plt.savefig(constants.PARAMS['FIG_PATH']+'plane_angles.svg')

    # get descriptive stats
    get_descriptive_stats_angles_cued(angles_radians)
    
    # get inferential stats
    get_inf_stats_angles_cued(angles_radians)
    
    
def run_plane_angles_analysis_uncued(constants):
     # load data
    load_path = constants.PARAMS['FULL_PATH'] + 'pca_data/valid_trials/'
    angles = pickle.load(open(load_path+'all_theta_uncued_post-cue.pckl','rb'))
    angles_radians = np.radians(angles)
    
    # plot angles
    plot_plane_angles_single(constants,angles_radians,'post')
    plt.savefig(constants.PARAMS['FIG_PATH']+'theta_uncued_postcue.svg')
    
    # run stats
    get_inf_stats_angles_uncued(angles_radians)
    

def run_plane_angles_analysis_cued_vs_uncued(constants):
    # load data
    load_path = constants.PARAMS['FULL_PATH'] + 'pca_data/valid_trials/'
    angles = pickle.load(open(load_path+'cued_vs_uncued_theta.pckl','rb'))
    angles_radians = np.radians(angles)
    
    plot_plane_angles_single(constants,
                             pycircstat.mean(angles_radians,axis=1),'cu')
    plt.savefig(constants.PARAMS['FIG_PATH']+'theta_cued_vs_uncued.svg')
    
    get_inf_stats_angles_cued_vs_uncued(pycircstat.mean(angles_radians,axis=1))
    

def run_phase_align_cued(constants):
    load_path = constants.PARAMS['FULL_PATH'] + 'pca_data/valid_trials/'
    pa = pickle.load(open(load_path+'all_psi.pckl','rb'))
    pa_radians = np.radians(pa)
    
    plot_plane_angles_multiple(constants,pa_radians,paired=False)
    plt.savefig(constants.PARAMS['FIG_PATH']+'phase_align_cued.svg')
    # run stats
    get_descriptive_stats_pa_cued(pa_radians)
    if constants.PARAMS['experiment_number'] == 3:
        get_inf_stats_angles_expt3(pa_radians)
    else:
        get_inf_stats_pa_cued(pa_radians)
    
    
def run_phase_align_uncued(constants):
    load_path = constants.PARAMS['FULL_PATH'] + 'pca_data/valid_trials/'
    pa = pickle.load(open(load_path+'all_psi_uncued_post-cue.pckl','rb'))
    pa_radians = np.radians(pa)
    
    plot_plane_angles_single(constants,pa_radians,cond='post')
    plt.savefig(constants.PARAMS['FIG_PATH']+'phase_align_uncued.svg')
    # run stats
    get_inf_stats_pa_uncued(pa_radians)
    

def run_phase_align_cued_vs_uncued(constants):
    load_path = constants.PARAMS['FULL_PATH'] + 'pca_data/valid_trials/'
    pa = pickle.load(open(load_path+'cued_vs_uncued_psi.pckl','rb'))
    pa_radians = np.radians(pa)
    
    # plot_plane_angles_multiple(constants,pa_radians,paired=True)
    plot_plane_angles_single(constants,
                             pycircstat.mean(pa_radians,axis=1),cond='cu')
    plt.savefig(constants.PARAMS['FIG_PATH']+'phase_align_cued_uncued.svg')
    # run stats
    get_inf_stats_pa_cued_vs_uncued(pycircstat.mean(pa_radians,axis=1))
    
    
def run_plane_angles_analysis_expt3(constants):
    # load data
    load_path = constants.PARAMS['FULL_PATH'] + 'pca_data/valid_trials/'
    angles = pickle.load(open(load_path+'all_theta.pckl','rb'))
    angles_radians = np.radians(angles)
    PVEs_3D = pickle.load(open(load_path+'all_PVEs_3D.pckl','rb'))
    
    # plot PVEs
    plot_PVEs_3D(PVEs_3D)
    plt.savefig(constants.PARAMS['FIG_PATH']+'PVEs_3D.png')
    # plot angles
    plot_plane_angles_multiple(constants,angles_radians,paired=False)
    plt.savefig(constants.PARAMS['FIG_PATH']+'plane_angles.svg')
    
    # get inferential stats
    get_inf_stats_angles_expt3(angles_radians)
    
    ## add PA
        