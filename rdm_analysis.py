#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 10:50:00 2020

@author: emilia
"""
from sklearn.manifold import MDS
from scipy.spatial.distance import squareform, pdist

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import vec_operations as vops
import pickle
import constants
from helpers import check_path
import seaborn as sns
from scipy.stats import zscore, stats, linregress
from numpy.linalg import lstsq, inv
import statsmodels.api as sm



def get_data_RDMs(constants):
    '''
    Construct data RDMs.

    Parameters
    ----------
    constants : TYPE
        DESCRIPTION.

    Returns
    -------
    rdm_precue : TYPE
        DESCRIPTION.
    rdm_postcue : TYPE
        DESCRIPTION.
    pre_data_RDM_averaged : TYPE
        DESCRIPTION.
    post_data_RDM_averaged : TYPE
        DESCRIPTION.

    '''
    
    load_path = constants.PARAMS['FULL_PATH']+'pca_data/valid_trials/'
    n_conditions = constants.PARAMS['B']*constants.PARAMS['B']*2

    # preallocate structures
    rdm_precue = np.zeros((n_conditions,n_conditions,constants.PARAMS['n_models']))
    rdm_postcue = np.zeros((n_conditions,n_conditions,constants.PARAMS['n_models']))
    
    rdm_precue_binned = np.zeros((constants.PARAMS['M'],constants.PARAMS['M'],constants.PARAMS['n_models']))
    rdm_postcue_binned = np.zeros((constants.PARAMS['M'],constants.PARAMS['M'],constants.PARAMS['n_models']))
    rdm_stimuli_binned = np.zeros((constants.PARAMS['M'],constants.PARAMS['M'],constants.PARAMS['n_models']))
    # rdm_postcue_binned_readout = np.zeros((batch_size//4,batch_size//4))


    # pick timepoints of interest
    t1 = constants.PARAMS['trial_timepoints']['delay1_end']-1
    t2 = constants.PARAMS['trial_timepoints']['delay2_end']-1
    
    plt.figure()
    # create pre- and post-cue RDMs for each model
    for model in range(constants.PARAMS['n_models']):
    
        # load pca data
        f = open(load_path+'/rdm_data_model' + str(model) + '.pckl', 'rb')
        obj = pickle.load(f)
        rdm_data = obj
        f.close()
        # data format: trial x time x neuron
        
        rdm_precue[:,:,model] = squareform(pdist(rdm_data[:,t1,:],'correlation'))
        rdm_postcue[:,:,model] = squareform(pdist(rdm_data[:,t2,:],'correlation'))
        
        f = open(load_path+'/pca_data_model' + str(model) + '.pckl', 'rb')
        pca_data = pickle.load(f)
        f.close()
        rdm_stimuli_binned[:,:,model] = squareform(pdist(pca_data['data'][:,0,:],'correlation'))
        rdm_precue_binned[:,:,model] = squareform(pdist(pca_data['delay1'],'correlation'))
        rdm_postcue_binned[:,:,model] = squareform(pdist(pca_data['delay2'],'correlation'))
    
        # rdm_postcue_binned_readout[:,:] = squareform(pdist(readout,'correlation'))
        
        # plot
        plt.subplot(2,10,int(model)+1)
        plt.imshow(rdm_postcue_binned[:,:,model])
        plt.colorbar()
        plt.title('Model '+str(model))

    #% save
    
    save_path = constants.PARAMS['FULL_PATH']+'RSA/'
    check_path(save_path)
    
    indiv_model_rdms = [rdm_precue_binned,rdm_postcue_binned]
    pickle.dump(indiv_model_rdms,open(save_path+'indiv_model_rdms.pckl','wb'))
    
    # average across models
    pre_data_RDM_averaged = np.mean(rdm_precue_binned,2)
    post_data_RDM_averaged = np.mean(rdm_postcue_binned,2)
    
    pickle.dump(pre_data_RDM_averaged,open(save_path+'pre_data_RDM_averaged.pckl','wb'))
    pickle.dump(post_data_RDM_averaged,open(save_path+'post_data_RDM_averaged.pckl','wb'))


    # # center
    # pre_data_RDM_averaged -= np.mean(pre_data_RDM_averaged) 
    # post_data_RDM_averaged -= np.mean(post_data_RDM_averaged)

    return rdm_precue,rdm_postcue,pre_data_RDM_averaged,post_data_RDM_averaged

def plot_full_data_RDMs(rdm_precue,rdm_postcue):
    '''
    Plot rdms averaged across models - all (combinatorial) conditions

    Parameters
    ----------
    rdm_precue : TYPE
        DESCRIPTION.
    rdm_postcue : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    plt.figure(figsize=(10,20))
    plt.subplot(121)
    plt.imshow(np.mean(rdm_precue,2),cmap = sns.cubehelix_palette(start=.5,rot=-.75,as_cmap = True,reverse=False))
    plt.xticks([])
    plt.yticks([])
    
    plt.colorbar()
    plt.title('pre-cue')
    
    plt.subplot(122)
    plt.imshow(np.mean(rdm_postcue,2),cmap = sns.cubehelix_palette(start=.5,rot=-.75,as_cmap = True,reverse=True))
    plt.colorbar()
    plt.title('post-cue')


def plot_binned_data_RDMs(pre_data_RDM_averaged,post_data_RDM_averaged):
    '''
    Plot rdms averaged across models - binned across uncued locations

    Parameters
    ----------
    pre_data_RDM_averaged : TYPE
        DESCRIPTION.
    post_data_RDM_averaged : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''

    fig, ax = plt.subplots(1,2,figsize=(10,10),num = 'Averaged RDMs')
    
    
    im1 = ax[0].imshow(pre_data_RDM_averaged,
                       # cmap = sns.color_palette("flare_r",as_cmap = True))
                       cmap = sns.cubehelix_palette(start=.5,rot=-.75,as_cmap = True,reverse=True))
    ax[0].set_title('pre-cue')
    ax[0].set_yticks([])
    ax[0].set_xticks([])
    
    fig.colorbar(im1,ax=ax[0])
    
    
    
    im2 = ax[1].imshow(post_data_RDM_averaged,
                       # cmap = sns.color_palette("flare_r",as_cmap = True))
                       cmap = sns.cubehelix_palette(start=.5,rot=-.75,as_cmap = True,reverse=True))
    ax[1].set_title('post-cue')
    ax[1].set_yticks([])
    ax[1].set_xticks([])
    
    fig.colorbar(im2,ax=ax[1])

def run_RDM_based_reg(constants,pre_data_RDM_averaged,post_data_RDM_averaged):
    '''
    Run RDM-based regression.

    Parameters
    ----------
    constants : TYPE
        DESCRIPTION.
    pre_data_RDM_averaged : TYPE
        DESCRIPTION.
    post_data_RDM_averaged : TYPE
        DESCRIPTION.

    Returns
    -------
    results_pre : TYPE
        DESCRIPTION.
    results_post : TYPE
        DESCRIPTION.

    '''

    # load model RDMs
    path = constants.PARAMS['FULL_PATH'] + 'RSA/model_RDMs_sqform.pckl'
    f = open(path,'rb')
    model_RDMs = pickle.load(f) # individual predictor arrays stacked along axis 1
    f.close()

    # run regression analysis
    model_RDMs = zscore(model_RDMs)
    X = sm.add_constant(model_RDMs) # predictor matrix
    
    y_pre = squareform(pre_data_RDM_averaged) # pre-cue data
    y_post = squareform(post_data_RDM_averaged) # post-cue data
      
    results_pre = sm.OLS(y_pre, X).fit() #fit OLS model
    print('Regression results: pre-cue')
    print(results_pre.summary())
    
    results_post = sm.OLS(y_post, X).fit()
    print('Regression results: post-cue')
    print(results_post.summary())
    
    return results_pre, results_post

def plot_RDM_reg_results(constants,results_pre,results_post):
    '''
    Plot the results (betas) from the RDM-based regression.

    Parameters
    ----------
    constants : TYPE
        DESCRIPTION.
    results_pre : TYPE
        DESCRIPTION.
    results_post : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    # load model RDM order
    path = constants.PARAMS['FULL_PATH'] + 'RSA/model_RDMs_order.pckl'
    f = open(path,'rb')
    model_RDMs_order = pickle.load(f) # individual predictor arrays stacked along axis 1
    f.close()
    
    
    plt.figure()
    # plot pre-cue results
    ax1 = plt.subplot(121)
    
    ix = np.arange(1,4)
    
    betas = results_pre.params[ix]
    CIs = results_pre.conf_int()[ix,1] - betas
    ax1.bar(np.arange(len(betas)),betas,color = [.5,.5,.5],yerr = CIs)
    plt.xticks(np.arange(len(betas)),labels = model_RDMs_order,rotation=20)
    
    
    # plot post-cue results
    ax2 = plt.subplot(122,sharex = ax1,sharey = ax1)
    
    betas = results_post.params[ix]
    CIs = results_post.conf_int()[ix,1] - betas
    ax2.bar(np.arange(len(betas)),betas,color = [.5,.5,.5],yerr = CIs)
    
    
    plt.xticks(np.arange(len(betas)),labels = model_RDMs_order,rotation=20)
    ax1.set_ylabel('Beta coefficient')
    ax1.set_title('pre-cue')
    ax2.set_title('post-cue')
    
    plt.tight_layout(pad=2)


#%% plot predictor RDMs


path = base_path + 'RSA/model_RDMs.pckl'
f = open(path,'rb')
model_RDMs_full = pickle.load(f) # pre- and post-cue arrays stacked along axis 2
f.close()

ortho180 = pickle.load(open(base_path + 'RSA/ortho_rotated_fullRDMs/180.pckl','rb'))

# plt.figure(figsize=(10,20),num = 'Model RDMs')
# plt.subplot(121)
# plt.imshow(squareform(model_RDMs[:,0]))
# plt.colorbar()
# plt.title('pre-cue')
# plt.yticks([])
# plt.xticks([])


# plt.subplot(122)
# plt.imshow(squareform(model_RDMs[:,1]))
# plt.colorbar()
# plt.title('post-cue')
# plt.yticks([])
# plt.xticks([])

fig, axes = plt.subplots(1,3, sharex=True, sharey = True)


titles = model_RDMs_order
for ix,ax in enumerate(axes.flat):
    if ix == 0:
        im = ax.imshow(ortho180,vmin=np.min(ortho180),vmax=np.max(ortho180))
    else:
        im = ax.imshow(model_RDMs_full[:,:,ix],vmin=np.min(model_RDMs_full[:,:,ix]),vmax=np.max(model_RDMs_full[:,:,ix]))
    ax.set_title(titles[ix])
    ax.set_xticks([])
    ax.set_yticks([])

fig.colorbar(im,ax=axes.ravel().tolist())

# plt.tight_layout(pad=2)




#%% do MDS on model-averaged RDM

def get_MDS_from_RDMs(pre_data_RDM_averaged,post_data_RDM_averaged):
    # mds_stimuli = fit_mds_to_rdm(np.mean(rdm_stimuli_binned,2))
    mds_precue = fit_mds_to_rdm(pre_data_RDM_averaged)
    mds_postcue = fit_mds_to_rdm(post_data_RDM_averaged)
return mds_precue,mds_postcue




def plot_geometry(ax,data,pca,plot_colours,plot_outline = True,legend_on = True):
    ms = 150
    n_colours = len(plot_colours)
    # plot the parallelogram defined by colours at location 1
    if plot_outline:
        ax.plot(np.append(data[:n_colours,0],data[0,0]),
              np.append(data[:n_colours,1],data[0,1]),
              np.append(data[:n_colours,2],data[0,2]),'k-')
    ax.scatter(data[0,0],data[0,1], data[0,2],marker='^',s = ms,
              c='k',label='loc1')
    ax.scatter(data[:n_colours,0],data[:n_colours,1],
              data[:n_colours,2],marker='^',s = ms,c=plot_colours)
  
    # repeat for loc 2
    if plot_outline:
        ax.plot(np.append(data[n_colours:,0],data[n_colours,0]),
              np.append(data[n_colours:,1],data[n_colours,1]),
              np.append(data[n_colours:,2],data[n_colours,2]),'k-')
    ax.scatter(data[-1,0],data[-1,1], data[-1,2],marker='s',s = ms,
              c='k',label='loc2')
    ax.scatter(data[n_colours:,0],data[n_colours:,1],
              data[n_colours:,2],marker='s',s = ms,c=plot_colours)
    
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
        raise NotImplementedError('Check shape of data matrix - should be [n_points,3]')
    
    # find vertices
    n_points = points.shape[0]
    verts = np.zeros((n_points,3))
    
    com = np.mean(points, axis=0) # centre of mass
    
    for i in range(n_points):
        verts[i,:] = vops.getProjection(points[i,:]-com,plane_vecs) # get projection of demeaned 3d points
        verts[i,:] += com #add the mean back
    
    # sort vertices according to shortest path - so that plotted plane will be a quadrilateral
    sorted_verts, sorting_order = vops.sortByPathLength(verts)
    #sorted_verts, sorting_order = vops.sortByVecAngle(verts)
    #sorted_verts = verts
    # plot the best-fit plane
    plot_plane(ax,sorted_verts,fc,a)
    #return verts, sorted_verts


def plot_data_MDS(mds_precue,mds_postcue,plot_colours):
    
    
    n_colours = len(plot_colours)

    plt.figure()
    ax1 = plt.subplot(121, projection='3d')
    ax2 = plt.subplot(122, projection='3d')
    plot_geometry(ax1,mds_precue,[],plot_colours,legend_on=False)
    plot_geometry(ax2,mds_postcue,[],plot_colours)    
    
    ax1.set_xlabel('dim 1',labelpad=25.0)
    ax1.set_ylabel('dim 2',labelpad=25.0)
    ax1.set_zlabel('dim 3',labelpad=35.0)
    
    ax2.set_xlabel('dim 1',labelpad=25.0)
    ax2.set_ylabel('dim 2',labelpad=30.0)
    ax2.set_zlabel('dim 3',labelpad=35.0)
    
    # ax1.set_yticks([-.1,0,.1])
    # ax1.set_zticks([-.1,0,.1])
    
    
    # ax2.set_xticks([-.1,0,.1])
    # ax2.set_yticks([-.1,0,.1])
    # ax2.set_zticks([-.1,0,.1])
    
    plt.tight_layout()
    plt.subplots_adjust(wspace = 0.2)
    
    
    # plt.subplots_adjust(wspace = 0.4)
    
    equal_axes(ax1)
    equal_axes(ax2)
    
    plt.legend(bbox_to_anchor=(1.5,1))
    plt.tight_layout()
    
    
    ax1.set_xticks(np.arange(-.01,.02,.01))
    ax1.set_yticks(np.arange(-.01,.02,.01))
    ax1.set_zticks(np.arange(-.01,.02,.01))
    
    lims = ax2.get_xlim()
    ax2.set_xticks(np.arange(-.04,.05,.04))
    ax2.set_yticks(np.arange(-.04,.05,.04))
    ax2.set_zticks(np.arange(-.04,.05,.04))
    
    
    
    ax1.set_title('pre-cue')
    ax2.set_title('post-cue')
    
    ax1.tick_params(labelsize = 26)
    ax1.tick_params(axis='x',pad=3)
    ax1.tick_params(axis='y',pad=5)
    ax1.tick_params(axis='z',pad=20)
    
    
    ax2.tick_params(labelsize = 26)
    ax2.tick_params(axis='x',pad=3)
    ax2.tick_params(axis='y',pad=5)
    ax2.tick_params(axis='z',pad=20)
    
    
    #% get and plot planes of best fit
    # PRE_CUE
    
    # get directions of max variance, i.e. vectors defining the plane
    delay1_planeUp = get_best_fit_plane(mds_precue[0:n_colours])
    delay1_planeDown = get_best_fit_plane(mds_precue[n_colours:])
    
    # calculate angle between planes
    theta_pre = get_angle_between_planes(delay1_planeUp.components_,delay1_planeDown.components_)
    print('Angle pre-cue: %.2f' %theta_pre)
    
    plot_subspace(ax1,mds_precue[:n_colours,:],delay1_planeUp.components_,fc='k',a=0.2)
    plot_subspace(ax1,mds_precue[n_colours:,:],delay1_planeDown.components_,fc='k',a=0.2)
    
    # POST_CUE
    
    delay2_planeUp = get_best_fit_plane(mds_postcue[0:n_colours])
    delay2_planeDown = get_best_fit_plane(mds_postcue[n_colours:])
    
    theta_post = get_angle_between_planes(delay2_planeUp.components_,delay2_planeDown.components_,)
    print('Angle post-cue: %.2f' %theta_post)
    
    plot_subspace(ax2,mds_postcue[:n_colours,:],delay2_planeUp.components_,fc='k',a=0.2)
    plot_subspace(ax2,mds_postcue[n_colours:,:],delay2_planeDown.components_,fc='k',a=0.2)
    
    ax1.set_title('pre-cue')
    ax2.set_title('post-cue')
    
    ax1.set_xlabel('PC1',labelpad = 20,style = 'italic')
    ax1.set_ylabel('PC2',labelpad = 20,style = 'italic')
    ax1.set_zlabel('PC3',labelpad = 15,style = 'italic')
    
    ax2.set_xlabel('PC1',labelpad = 20,style = 'italic')
    ax2.set_ylabel('PC2',labelpad = 20,style = 'italic')
    ax2.set_zlabel('PC3',labelpad = 15,style = 'italic')


def run_full_rep_geom_analysis(constants):
    get_model_RDMs(constants)
    rdm_precue,rdm_postcue,pre_data_RDM_averaged,post_data_RDM_averaged = get_data_RDMs(constants)
    plot_full_data_RDMs(rdm_precue,rdm_postcue)
    plot_binned_data_RDMs(pre_data_RDM_averaged,post_data_RDM_averaged)
    results_pre, results_post = run_RDM_based_reg(constants,pre_data_RDM_averaged,post_data_RDM_averaged)
    plot_RDM_reg_results(constants,results_pre,results_post)
    
    mds_precue,mds_postcue =get_MDS_from_RDMs(pre_data_RDM_averaged,post_data_RDM_averaged)
    plot_data_MDS(mds_precue,mds_postcue,constants.PLOT_PARAMS['4_colours'])
    
#%% plot pre-cue RDMs for each model

# plt.figure()
# for m in range(n_models):
#     plt.subplot(2,5,m+1)
#     plt.imshow(rdm_precue_binned[:,:,m])
#     plt.colorbar()
    
# #%% MDS for each model
# colours = ['r','y','g','b']

# plt.figure()
# for m in range(n_models):
#     ax = plt.subplot(2,5,m+1, projection='3d')
#     mds = fit_mds_to_rdm(rdm_precue_binned[:,:,m])
#     plot_geometry(ax,mds,[],colours,legend_on=False)



