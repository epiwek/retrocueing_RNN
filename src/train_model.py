#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 22:07:39 2021

@author: emilia
"""

import numpy as np
import torch
from torch import nn
from torch import optim
import pickle
import os.path

import matplotlib.pyplot as plt
from define_model import define_model
from generate_data_1hot import make_stimuli_1hot
from generate_data_vonMisses import make_stimuli_vonMises,change_cue_validity
from helpers import print_progress

#%%

def train_model(params,loss_fn,data,device):
    """
    Train the RNN model and save it along with the training details.
    
    Parameters
    ----------
    
    params : dictionary 
       
    loss_fn : torch.nn modeule
        
    data : dictionary
        

    Returns
    -------
    model : torch object
        Object created by the sklearn.decomposition.PCA method.
        fitted_plane.components_ gives the plane vectors
    track_training : dictionary
    """
    
    # set seed for reproducibility
    torch.manual_seed(params['model_number'])
    
    #% initialise model
    model, net_type = define_model(params,device)
    
    # transfer model to GPU if avaialable
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'])
    
    track_training = {}
    loss_all = torch.zeros(params['batch_size'],params['n_epochs']).to(device)
    shuffling_order = torch.zeros((params['batch_size'],
                                                     params['n_epochs']),
                                                    dtype=torch.long).to(device)
    net_outputs = torch.zeros((params['batch_size'],
                                             params['n_colCh'],
                                             params['n_epochs'])).to(device)
    
    # mean and std of hidden activations on the first trial of each batch
    hidden_stats  = torch.zeros((params['n_epochs'],
                                                   params['seq_len'],2)).to(device)
    
    inputs_base = data['inputs']
    inputs_base = inputs_base.to(device)
    
    targets = data['targets']
    targets = targets.to(device)


    if params['condition']=='probabilistic':
        invalid_trials = []
    
    # loop over epochs
    for i in range(params['n_epochs']):            
            # shuffle dataset for SGD
            shuffling_order[:,i] = \
                torch.randperm(params['batch_size'],dtype=torch.long).to(device)
            
            # make some cues invalid
            if params['condition']=='probabilistic':
                inputs = torch.tensor(inputs_base) # create a copy of the inputs
                inputs, ixs = change_cue_validity(inputs,params) # change some trials
                # saved the ixs
                invalid_trials.append(ixs)
            # loop over training examples
            for trial in range(params['batch_size']):
                
                outputs, o, hidden = \
                    model(inputs[:,shuffling_order[trial,i],:].unsqueeze(1))
                
                if trial == 0:
                    hidden_stats[i,:,0] = \
                        torch.std_mean(o,-1)[0].squeeze()
                    hidden_stats[i,:,1] = \
                        torch.std_mean(o,-1)[1].squeeze()
                # Compute loss
                loss = loss_fn(outputs.unsqueeze(0), 
                               targets
                               [shuffling_order
                                [trial,i],:].unsqueeze(0))
                # Keep track of outputs and loss
                loss_all[trial,i] = loss.item()
                net_outputs[trial,:,i] = outputs.detach()
                # Compute gradients
                optimizer.zero_grad()
                loss.backward()
                # Update weights
                optimizer.step()
            
            # print progress
            if (i%100 == 0):
                print('Model %2d :    %.2f%% iterations of SGD completed...loss = %.2f' \
                          % (int(params['model_number']),
                             100* (i + 1) / params['n_epochs'],
                             torch.sum(loss_all[:,i])))
                # print_progress(i, params['n_epochs'])
            if (i==params['n_epochs']-1):
                print('Model %2d :    100%% iterations of SGD completed...loss = %.2f' \
                          % (int(params['model_number']),
                             torch.sum((loss_all[:,i]))))
                # print_progress(i, params['n_epochs'])
    track_training = {}
    track_training['loss'] = loss_all
    track_training['shuffling_order'] = shuffling_order
    track_training['outputs'] = net_outputs
    # mean and std of hidden activations on the first trial of each batch
    track_training['hidden_stats'] = hidden_stats
    if params['condition']=='probabilistic':
        track_training['invalid_trials'] = torch.tensor(invalid_trials)
    
    return model, track_training


def partial_training(params,loss_fn,data,trial_sequence):
    """
    Train the RNN model up to some point using a fixed trial order sequence and
    save it.
    
    Parameters
    ----------
    
    params : dictionary 
       
    loss_fn : torch.nn modeule
        
    data : dictionary
    
    trial_sequence: numpy array, contains shuffling order of trials used for SGD
        

    Returns
    -------
    model : torch object
        Object created by the sklearn.decomposition.PCA method.
        fitted_plane.components_ gives the plane vectors
    """
    # set seed for reproducibility
    torch.manual_seed(params['model_number'])
    
    #% initialise model
    model, net_type = define_model(params['n_inp'],
                                   params['n_rec'],
                                   params['n_colCh'])

    optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'])
    
    
    track_training = {}
    track_training['loss'] = torch.zeros(params['batch_size'],params['n_epochs'])
    track_training['shuffling_order'] = torch.zeros((params['batch_size'],
                                                     params['n_epochs']),
                                                    dtype=torch.long)
    track_training['outputs'] = torch.zeros((params['batch_size'],
                                             params['n_colCh'],
                                             params['n_epochs']))
     # loop over epochs
    for i in range(params['n_epochs']):            
            # shuffle dataset for SGD
            
            # loop over training examples
            for trial in range(params['batch_size']):
                
                outputs, o, hidden = \
                    model(data['inputs'][:,trial_sequence[trial,i],:].unsqueeze(1))
                

                # Compute loss
                loss = loss_fn(outputs.unsqueeze(0), 
                               data['targets']
                               [trial_sequence[trial,i],:].unsqueeze(0))
                # Keep track of outputs and loss
                track_training['loss'][trial,i] = loss.item()
                track_training['outputs'][trial,:,i] = outputs.detach()
                # Compute gradients
                optimizer.zero_grad()
                loss.backward()
                # Update weights
                optimizer.step()
            
            # print progress
            if (i%100 == 0):
                print('Model %2d :    %.2f%% iterations of SGD completed...loss = %.2f' \
                          % (int(params['model_number']),
                             100* (i + 1) / params['n_epochs'],
                             torch.sum(track_training['loss'][:,i])))
            if (i==params['n_epochs']-1):
                print('Model %2d :    100%% iterations of SGD completed...loss = %.2f' \
                          % (int(params['model_number']),
                             torch.sum((track_training['loss'][:,i]))))
    return model, track_training
    
def save_model(path,params,model,track_training,partial_training=False):
    """
    Parameters
    ----------
    path :
    
    params : dictionary 
    
    model :
    track_training : torch.nn modeule
        
    partial_training : dictionary
        

    Returns
    -------
    None
    """
    # check if path exists - if not, create one
    if not (os.path.exists(path+'kappa'+str(params['kappa_val']))):
        os.mkdir(path+'kappa'+str(params['kappa_val']))
        path = path + 'kappa'+str(params['kappa_val'])+'/'
        os.mkdir(path+'nrec'+str(params['n_rec']))
        os.mkdir(path+'nrec'+str(params['n_rec'])+'/lr'+
                 str(params['learning_rate']))
    else:
        path = path + 'kappa'+str(params['kappa_val'])+'/'
        
    if not (os.path.exists(path+'nrec'+str(params['n_rec']))):
           os.mkdir(path+'nrec'+str(params['n_rec']))
           os.mkdir(path+'nrec'+str(params['n_rec'])+'/lr'+
                    str(params['learning_rate']))
    else:
        if not (os.path.exists(path+'nrec'+str(params['n_rec'])+'/lr'+
                               str(params['learning_rate']))):
            os.mkdir(path+'nrec'+str(params['n_rec'])+'/lr'+
                     str(params['learning_rate']))
    
    path = path+'nrec'+str(params['n_rec'])+'/lr'+\
        str(params['learning_rate'])+'/'   
    
    # save model
    if not (os.path.exists(path+'saved_models')):
            os.mkdir(path+'saved_models') 
    
    if partial_training:
        if not (os.path.exists(path+'saved_models/partial_training/')):
            os.mkdir(path+'saved_models/partial_training/')
        
        torch.save(model,path+'saved_models/partial_training/model' 
                   + str(params['model_number'])+'_'+str(params['n_epochs'])
                   +'epochs')
        torch.save(model.state_dict(),path+'saved_models/partial_training/model'+
               str(params['model_number'])+'_'+ str(params['n_epochs'])
               +'epochs'+'_statedict')
    else:
        torch.save(model,path+'saved_models/model' + str(params['model_number']))
        torch.save(model.state_dict(),path+'saved_models/model'+
                   str(params['model_number'])+'_statedict')
        # save training data and loss
        if not (os.path.exists(path+'training_data')):
                os.mkdir(path+'training_data')     
        
        f = open(path+'training_data/training_data_model'+
                 str(params['model_number'])+'.pckl','wb')
        pickle.dump(track_training,f)
        f.close()
    
    print('Model saved')
   
def eval_model(model,test_data,params,save_path):
    # # load
    # model, RNN = define_model(n_inp,n_rec,n_stim)
    # model = torch.load(load_path+'model'+str(model_number))
    
    #% evaluate model on pca data
    model.eval()
    with torch.no_grad():
        #output, (hidden,cell) = model.Wrec(I)
        output, hidden = model.Wrec(test_data['inputs'])
        readout = model.out(hidden)
    print('Model evaluated')
    
    # full dataset
    eval_data = {"dimensions":['trial','time','n_rec'],
                 "data":output.permute(1,0,-1),
                 "labels":
                     {"loc":test_data['loc'],
                      "c1":test_data['c1'],
                      "c2":test_data['c2']}}
    
    # save
    save_data(eval_data,params,save_path+'eval_data_model')
    
    # pca data - binned and averaged across uncued 
    # M = B*L - colour-bin*location combinations
    n_samples = params['batch_size']//params['M']
    trial_data = torch.reshape(output.permute(1,0,-1).unsqueeze(0),
                               (params['M'],n_samples,
                                params['seq_len'],params['n_rec']))
    
    # average across n_samples
    trial_data = torch.mean(trial_data,1) #(M,seq_len,n_rec)
    d1_ix = params['trial_timings']['stim_dur'] + \
        params['trial_timings']['delay1_dur'] - 1
    d2_ix =  sum(params['trial_timings'].values())-1
    # delay1 = torch.mean(trial_data[:,d1_ix,:],0)
    # delay2 = torch.mean(trial_data[:,d2_ix,:],0)
    delay1 = trial_data[:,d1_ix,:]
    delay2 = trial_data[:,d2_ix,:]
    
    labels = {}
    labels['loc'] = np.zeros((params['M'],),dtype=int)
    labels['loc'][params['B']:] = 1
    labels['c1'] = np.empty((params['M'],))
    labels['c1'][:params['B']] = np.arange(params['B'])
    labels['c2'] = np.empty((params['M'],))
    labels['c2'][params['B']:] = np.arange(params['B'])
    
    pca_data = {'dimensions':['M','time','n_rec'],
                'data':trial_data,'delay1':delay1,'delay2':delay2,
                "labels":labels}
    
    # save
    save_data(pca_data,params,save_path+'pca_data_model')
   
        
    # rdm data - averaged across uncued    - rewrite this so it makes sense, but works for now
    B = params['batch_size'] // params['B']
    rdm_data = output.permute(1,0,2).unsqueeze(0)
    rdm_data = torch.reshape(rdm_data,(B,params['B'],params['seq_len'],params['n_rec']))
    rdm_data = torch.mean(rdm_data,1) # bin uncued
    
    rdm_data = torch.reshape(rdm_data.unsqueeze(0),(params['n_stim']*2,params['B'],params['seq_len'],params['n_rec']))
    rdm_data = torch.reshape(rdm_data.unsqueeze(0),(params['B']*2,params['B'],params['B'],params['seq_len'],params['n_rec']))
    rdm_data = torch.mean(rdm_data,1) # average across cued
    rdm_data = torch.reshape(rdm_data,(params['B']*2*params['B'],params['seq_len'],params['n_rec'])) # condition, time, nrec
    
    save_data(rdm_data,params,save_path+'rdm_data_model')
    print('.... Data saved')
    # f = open(dpath+'rdm_data_model'+str(model_number)+'.pckl','wb')
    # pickle.dump(rdm_data,f)
    # f.close()
    
    model_outputs = {'data':readout.squeeze(),'labels':{"loc":test_data['loc'],
                      "c1":test_data['c1'],
                      "c2":test_data['c2']}}
    
    save_data(rdm_data,params,save_path+'model_outputs_model')
    
    print('.... Data saved')
    
    return eval_data,pca_data,rdm_data,model_outputs

def save_data(data,params,save_path):
    f = open(save_path+str(params['model_number'])+'.pckl','wb')
    pickle.dump(data,f)
    f.close()