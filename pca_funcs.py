#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 12:50:39 2020

@author: emilia
"""
from sklearn.decomposition import PCA
import numpy as np
import pickle
import torch

def pca3D(delay1,delay2,n_colours):
    
    # define models
    delay1_pca = PCA(n_components=3)
    delay2_pca = PCA(n_components=3)
    
    # get point coordinates in 3d space
    delay1_3dcoords = delay1_pca.fit_transform(delay1)
    delay2_3dcoords = delay2_pca.fit_transform(delay2)
    
    # Find planes of best fit and angle between them
    
    # delay1
    delay1_up_pca = PCA(n_components=2)
    delay1_down_pca = PCA(n_components=2)
    delay1_planeUp = delay1_up_pca.fit(delay1_3dcoords[0:n_colours,:]-np.mean(delay1_3dcoords[0:n_colours,:])) # get directions of max variance, i.e. vectors defining the plane
    delay1_planeDown = delay1_down_pca.fit(delay1_3dcoords[n_colours:,:] - np.mean(delay1_3dcoords[n_colours:,:]))
    
    # calculate angle between planes
    cos_theta_pre = np.dot(np.cross(delay1_planeUp.components_[0,:],delay1_planeUp.components_[1,:]),
                   np.cross(delay1_planeDown.components_[0,:],delay1_planeDown.components_[1,:]))
    theta_pre = np.degrees(np.arccos(cos_theta_pre))
    #delay 2
    delay2_up_pca = PCA(n_components=2)
    delay2_down_pca = PCA(n_components=2)

    delay2_planeUp = delay2_up_pca.fit(delay2_3dcoords[0:n_colours,:] - np.mean(delay2_3dcoords[0:n_colours,:]))
    delay2_planeDown = delay2_down_pca.fit(delay2_3dcoords[n_colours:,:] - np.mean(delay2_3dcoords[n_colours:,:]))
    
    cos_theta_post = np.dot(np.cross(delay2_planeUp.components_[0,:],delay2_planeUp.components_[1,:]),
                   np.cross(delay2_planeDown.components_[0,:],delay2_planeDown.components_[1,:]))
    theta_post = np.degrees(np.arccos(cos_theta_post))
    
    return delay1_3dcoords, delay2_3dcoords, delay2_planeUp, delay2_planeDown, theta_pre, theta_post

def prepare_data_RNN(path_to_params,model):
    from make_stimuli import make_stimuli
    
    # load task params    
    f = open(path_to_params+'task_params.pckl', 'rb')
    obj = pickle.load(f)
    [n_stim,n_colCh,tuning_params,trial_timings] = obj
    f.close()
    
    # load model params
    f = open(path_to_params+'model_params.pckl', 'rb')
    obj = pickle.load(f)
    [n_inp,n_rec,n_iter,batch_size,learning_rate] = obj
    f.close()
    
    # subsampling params
    Time = sum(trial_timings.values()) # sequence length
    #colour_space = torch.linspace(0, 2*np.pi, n_stim+1)[:-1] # all possible stimulus values    
    
    # bin data for PCA 
    n_bins = 4 # colour bins
    n_stimperbin = n_stim//n_bins #n of stim colours/bin
    S = n_bins*2 # colour bin x location combos
    batch = 200
    n_samples = batch//S # n trials per category

    # to do: add a proper error
    if ((batch % S)!=0):
        print('Error: n_samples must be an integer')

    else:
        c_specs = {} # constraint specs
        c_specs['n_subcats'] = n_bins
        c_specs['n_stimperbin'] = n_stimperbin
        c_specs['n_samples'] = n_samples
        # these are used by the make_stimuli function to constrain the number of trials
        # e.g. to be equal for each bin
        
        # evaluate model
        model.eval()
        I, loc, c1, c2, T = make_stimuli(n_stim,n_colCh,batch,trial_timings,tuning_params,
                                         constraints = 'on',c_specs = c_specs)
        with torch.no_grad():
            output, h_n = model.Wrec(I)
        
        loc = loc[:,:,0].squeeze() # get rid of the time dimension
        
    trial_data = torch.reshape(output.permute(1,0,-1).unsqueeze(0),(S,n_samples,Time,n_rec)) 
    # reshape into condition (bin x location) x trial x time x unit
    trial_data = trial_data.permute(1,-1,0,2) # n_samples x N x S x T
    
    return trial_data, c_specs
    