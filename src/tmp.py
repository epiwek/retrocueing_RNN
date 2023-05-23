#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 16:35:59 2021

@author: emilia
"""

import numpy as np
for m in range(10):
    f = open(valid_path+'model_outputs_model'+str(m)+'.pckl','rb')
    model_outputs = pickle.load(f)
    f.close()
    outputs = model_outputs['data']
    
    # plot first 5 posteriors
    
    # plt.figure()
    # plt.plot(outputs[:5,:].T,'k-o')
    # plt.title('Model %d' %m)
    # plt.xlabel('Output channel')
    # plt.ylabel('Activation')
    # plt.tight_layout()
    
    loc1_ix = np.array(test_valid['loc'][0,:,:].squeeze(),dtype=bool)
    loc2_ix = np.array(test_valid['loc'][1,:,:].squeeze(),dtype=bool)
    
    probed_colour = torch.cat((test_valid['c1_label'][loc1_ix],test_valid['c2_label'][loc2_ix]))
    
    peak = np.argsort(outputs,1)[:,-1]
    
    diff = helpers.wrap_angle(peak - probed_colour)
    # plt.figure()
    # plt.hist(diff)
    
    
    bins = np.unique(diff)
    counts = np.empty((len(bins),))
    for b,bin_val in enumerate(bins):
        counts[b] = (diff == bin_val).float().sum()
        
    counts /= len(diff)
    
    plt.figure()
    plt.plot(np.degrees(bins),counts,'-o')
    plt.title('Model %d' %m)
    plt.tight_layout()
    
    
    
    
#%%
import torch
import constants
from retrocue_model import apply_conv_criterion
path = constants.PARAMS['FULL_PATH']
# window = constants.PARAMS['conv_criterion']['trial_window']
window = 5

thr_loss = 0.004
thr_slope = -5e-05
plt.figure()

def get_loss_slope(window,loss_vals):
    p = np.polyfit(np.arange(window), loss_vals, 1)
    return p[0]
# def apply_conv_crit(loss_val,p,thr_loss,thr_slope):
#     slope_cond = np.logical_and(p>=thr_slope,p<= 0)
#     loss_cond = torch.all(loss_val<=thr_loss)
#     converged = np.logical_and(loss_cond,slope_cond)
#     return converged




for m in range(10):
    
    # load training data
    f = open(path+'training_data/'+'training_data_model'+str(m)+'.pckl','rb')
    track_training = pickle.load(f)
    f.close()
    
    loss_vals = track_training['loss_epoch']
    
    n_epochs = len(loss_vals)
    dLoss = torch.empty((n_epochs-window,))
    loss_clean = torch.empty((n_epochs-window,))
    
    # converged = torch.empty((n_epochs-window,))
    # for i in range(window,n_epochs):
    #     d,l = retnet.get_dLoss_dt(constants.PARAMS,loss_vals[i-window:i+1])
    #     dLoss[i-window] = d.mean()
    #     loss_clean[i-window] = l[-1]
        
       
    #     converged[i-window] = retnet.apply_conv_criterion(constants.PARAMS,loss_vals[i-window:i+1])
    
    slopes = torch.empty((n_epochs-window,))
    converged = torch.empty((n_epochs-window,))
    for i in range(window,n_epochs):
        slopes[i-window] = get_loss_slope(window,loss_vals[i-window+1:i+1])
        # converged[i-window] = apply_conv_crit(loss_vals[i],slopes[i-window],thr_loss,thr_slope)
        converged[i-window] = apply_conv_criterion(constants.PARAMS,loss_vals[i-window+1:i+1])

    
    
    
    # slopes2 = torch.empty((n_epochs-window,))
    # for i in range(window,n_epochs):
    #     p = np.polyfit(np.arange(i-window,i+1), loss_vals[i-window:i+1], 1)
    #     slopes2[i-window] = p[0]
    #     x= np.arange(i-window,i+1)
    #     y = np.arange(i-window,i+1)*p[0] + p[1]
    #     plt.plot(x,y)

    
    plt.subplot(2,5,m+1)
    plt.plot(loss_vals)
    plt.plot(torch.where(converged)[0]+window,np.zeros((len(torch.where(converged)[0]))),'r*')
    
    # xlims = plt.xlim()
    # plt.subplot(212)
    # plt.plot(np.arange(window,n_epochs),dLoss)
    # plt.plot(np.arange(window,n_epochs),np.ones((n_epochs-window,))*constants.PARAMS['conv_criterion']['thr_dLoss'],'k--')
    # plt.plot(torch.where(converged)[0]+window,np.ones((len(torch.where(converged)[0])))*0.0001,'r*')
    # plt.title('dLoss')
    # plt.xlim(xlims)
    
    
    plt.suptitle('Model %d' %m)
    # plt.tight_layout()
    
    # plt.figure()
    # plt.plot(loss_clean)
    
  
    

# def conv_crit(constants,loss_vals):
#     p = np.polyfit(np.arange(len(loss_vals)), loss_vals, 1)
#     if p <= thr_dLoss
    
    