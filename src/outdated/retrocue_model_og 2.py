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
import helpers
import pdb

from generate_data_vonMisses import change_cue_validity

#%%
        
def define_model(params, device,model_class = 'RNN'):
    global model, RNN # outputs need to be global variables, otherwise will be unable to save model with pickle
    
    class RNN(nn.Module):
        def __init__(self, params):
            super(RNN, self).__init__()
            # input
            self.n_rec = params['n_rec'] # number of recurrent neurons
            self.n_inp = params['n_inp']
            self.n_out = params['n_out']
            
            if params['noise_type']=='hidden':
            # _____________________________________________________ # 
            # implementation for simulations with hidden noise:
            # (need to add noise after the Wrec@ht-1 step, but before the nonlinearity)
                self.inp = nn.Linear(self.n_inp,self.n_rec)
                self.inp.weight = nn.Parameter(self.inp.weight*params['init_scale'])
                self.inp.bias = nn.Parameter(self.inp.bias*params['init_scale'])
                
                self.Wrec = nn.Parameter((torch.randn(self.n_rec, self.n_rec) / np.sqrt(self.n_rec))*params['init_scale'])  # recurrent weights
                self.relu = nn.ReLU()
            else:
                # _____________________________________________________ # 
                # implementation for simulations without hidden noise:
                    
                self.Wrec = nn.RNN(self.n_inp,self.n_rec, nonlinearity = 'relu')
                
                self.Wrec.bias_hh_l0 = nn.Parameter(self.Wrec.bias_hh_l0*params['init_scale'])
                self.Wrec.bias_ih_l0 = nn.Parameter(self.Wrec.bias_ih_l0*params['init_scale'])
                self.Wrec.weight_hh_l0 = nn.Parameter(self.Wrec.weight_hh_l0*params['init_scale'])
                self.Wrec.weight_ih_l0 = nn.Parameter(self.Wrec.weight_ih_l0*params['init_scale'])
           
            
            # output layer
            self.out = nn.Linear(self.n_rec, self.n_out) # output layer
            
           
        def step(self, input_ext,hidden,noise):
            if params['noise_type']=='hidden':
                # print('hidden t-1')
                # pdb.set_trace()
                # print(hidden)
                # print('hidden t')
                hidden = self.relu(self.inp(input_ext.unsqueeze(0)) + hidden @ self.Wrec.T \
                    + noise)
                
                # print(hidden)
                # hidden = self.relu(hidden)
                # print('hidden t relu')
                # print(hidden)

                
                
                output = hidden.clone().detach()# not neededm but keep it so that other code doesn't break down
            else:
                output, hidden = self.Wrec(input_ext.unsqueeze(0),hidden)
            
            return output, hidden

        
        def forward(self, inputs):
            """
            Run the RNN with input timecourses
            """
            if params['noise_type']=='input':
                # Add noise to inputs
                inputs = add_noise(inputs,params,device)
                
                # One line implementation - doesn't use self.step, but not possible
                # to add hidden noise to each timestep
                o, h_n = self.Wrec(inputs)
                
            elif params['noise_type']=='hidden':
                # Add noise to hidden units
                seq_len = inputs.shape[0]
                batch = inputs.shape[1]
                # To add hidden noise, need to use the expanded implementation below
                # Initialize network state
                hidden = torch.zeros((1,inputs.size(1), self.n_rec),device = device) # 0s
            
                # Run the input through the network - across time
                o = torch.empty((seq_len,batch,self.n_rec),device = device)
                for i in range(seq_len):
                    print('t = %d' %i)
                    if len(np.where(params['noise_timesteps']==i)[0])>0:
                        # timestep with added noise
                        if params['noise_distr'] == 'uniform':
                            noise = (torch.rand(hidden.size(),device = device)-0.5)*params['epsilon']
                        elif params['noise_distr'] == 'normal':
                            # Add Gaussian noise to appropriate timesteps of the trial
                            noise = (torch.randn(hidden.size(),device = device))*params['epsilon']
                    else:
                        # timestep without noise
                        noise = torch.zeros(hidden.size(),device = device)
                        
                        # hidden.add_(noise)
                        # if np.logical_and(i>0,i<4):
                        #     print('hidden + noise')
                        #     print(hidden)
                        
                        
                    
                    if i == seq_len - 1:
                        # last time step of the sequence
                        o[i,:,:], h_n = self.step(inputs[i, :, :], hidden,noise)
                    else:
                        o[i,:,:], hidden = self.step(inputs[i, :, :], hidden,noise)
                    # if np.logical_and(i>0,i<4):
                    #     print('hidden after update')
                    #     print(hidden)
           
            else:
                # No noise
                o, h_n = self.Wrec(inputs)
            
            # pass the recurrent activation from the last timestep through the decoder layer
            output = self.out(h_n)
            return output.squeeze(), o, h_n   
        
    model = RNN(params)
    
    return model, RNN

def train_model(params,loss_fn,data,device):
    """
    Train the RNN model and save it, along with the training details.
    
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

    #optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    optimizer = optim.SGD(model.parameters(),lr=params['learning_rate'],momentum=.9)
    track_training = {}
    loss_all = torch.zeros(params['stim_set_size'],params['n_epochs']).to(device)
    shuffling_order = torch.zeros((params['stim_set_size'],
                                                     params['n_epochs']),
                                                    dtype=torch.long).to(device)
    net_outputs = torch.zeros((params['stim_set_size'],
                                             params['n_colCh'],
                                             params['n_epochs'])).to(device)
    
    # mean and std of hidden activations on the first trial of each batch
    hidden_stats  = torch.zeros((params['n_epochs'],
                                                   params['seq_len'],2)).to(device)
    
    inputs_base = data['inputs']
    inputs_base = inputs_base.to(device)
    
    targets = data['targets']
    targets = targets.to(device)


    if params['condition']!='determinisitc':
        invalid_trials = []
    
    # loop over epochs
    for i in range(params['n_epochs']):   
            # print('Epoch %d : ' %i)         
            # shuffle dataset for SGD
            shuffling_order[:,i] = \
                torch.randperm(params['batch_size'],dtype=torch.long).to(device)
            
            # make some cues invalid
            if params['condition']!='determinisitc':
                inputs = inputs_base.clone() # create a copy of the inputs
                inputs, ixs = change_cue_validity(inputs,params) # change some trials
                # saved the ixs
                invalid_trials.append(ixs)
            else:
                inputs = inputs_base
            # loop over training examples
            for trial in range(params['stim_set_size']):
                # print('Trial %d : ' %trial)     
                
                outputs, o, hidden = \
                    model(inputs[:,shuffling_order[trial,i],:].unsqueeze(1))
                # print(torch.where(o<0))
                if np.logical_and(trial == 0,i==0):
                    hidden_stats[i,:,0] = \
                        torch.std_mean(o,-1)[0].squeeze()
                    hidden_stats[i,:,1] = \
                        torch.std_mean(o,-1)[1].squeeze()
                    print('First forward pass')
                    print('    Means:')
                    print(hidden_stats[i,:,1])
                    print('    S.d.:')
                    print(hidden_stats[i,:,0])
                
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
            # if (i%500 == 0):
            if ((i*100/params['n_epochs'])%25)==0:
                print('Model %2d :    %.2f%% iterations of SGD completed...loss = %.2f' \
                          % (int(params['model_number']),
                              100* (i + 1) / params['n_epochs'],
                              torch.sum(loss_all[:,i])))
                # print_progress(i, params['n_epochs'])
            if (i==params['n_epochs']-1):
                print('Model %2d :    100%% iterations of SGD completed...loss = %.2f' \
                          % (int(params['model_number']),
                              torch.sum((loss_all[:,i]))))
    # converged = False
    # i=0
    # while np.logical_or(not converged,i>params['n_epochs']*2):
    #     # shuffle dataset for SGD
    #     shuffling_order[:,i] = \
    #         torch.randperm(params['batch_size'],dtype=torch.long).to(device)
        
    #     # make some cues invalid
    #     if params['condition']=='probabilistic':
    #         inputs = torch.tensor(inputs_base) # create a copy of the inputs
    #         inputs, ixs = change_cue_validity(inputs,params) # change some trials
    #         # saved the ixs
    #         invalid_trials.append(ixs)
    #     else:
    #         inputs = inputs_base
    #     # loop over training examples
    #     for trial in range(params['batch_size']):
            
    #         outputs, o, hidden = \
    #             model(inputs[:,shuffling_order[trial,i],:].unsqueeze(1))
            
    #         # if trial == 0:
    #         #     hidden_stats[i,:,0] = \
    #         #         torch.std_mean(o,-1)[0].squeeze()
    #         #     hidden_stats[i,:,1] = \
    #         #         torch.std_mean(o,-1)[1].squeeze()
    #         # Compute loss
    #         loss = loss_fn(outputs.unsqueeze(0), 
    #                         targets
    #                         [shuffling_order
    #                         [trial,i],:].unsqueeze(0))
    #         # Keep track of outputs and loss
    #         loss_all[trial,i] = loss.item()
    #         net_outputs[trial,:,i] = outputs.detach()
    #         # Compute gradients
    #         optimizer.zero_grad()
    #         loss.backward()
    #         # Update weights
    #         optimizer.step()
        
    #     # print progress
    #     if (i%500 == 0):
    #         print('Model %2d :    %5d epochs completed...loss = %.2f' \
    #                   % (int(params['model_number']),
    #                       i+1,
    #                       torch.sum(loss_all[:,i])))
    #     if torch.sum(loss_all[:,i]) < 0.05:
    #         converged = True
    #         print('Model %2d :    Training completed after %5d epochs...loss = %.2f' \
    #                       % (int(params['model_number']),i,
    #                           torch.sum((loss_all[:,i]))))
    #     if i >= params['n_epochs']*2:
    #         print('Model %2d :    Exiting training - too many epochs' \
    #                       % (int(params['model_number'])))
    #     i+=1
    
    track_training = {}
    track_training['loss'] = loss_all
    track_training['shuffling_order'] = shuffling_order
    track_training['outputs'] = net_outputs
    # mean and std of hidden activations on the first trial of each batch
    track_training['hidden_stats'] = hidden_stats
    
    if params['condition']!='determinisitc':
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

def progressive_training(params,loss_fn,data,device):
    """
    Train the RNN model and save it, along with the training details.
    
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
    main_path = path +'epsilon' + str(params['epsilon'])+\
        '/kappa'+ str(params['kappa_val'])+\
            '/nrec'+str(params['n_rec'])+\
                '/lr'+str(params['learning_rate'])+'/'
    
    helpers.check_path(path)
    model_path = main_path + 'saved_models/'
    helpers.check_path(model_path)

    
    if partial_training:
        # if not (os.path.exists(path+'saved_models/partial_training/')):
        #     os.mkdir(path+'saved_models/partial_training/')
        model_path = path + 'partial_training/'
        
        torch.save(model,model_path+'model' + str(params['model_number'])+\
                   '_'+str(params['n_epochs'])+'epochs')
        torch.save(model.state_dict(),model_path+'model'+str(params['model_number'])+
                   '_'+ str(params['n_epochs'])+'epochs'+'_statedict')
    else:
        torch.save(model,model_path+'model' + str(params['model_number']))
        torch.save(model.state_dict(),model_path+'model'+
                   str(params['model_number'])+'_statedict')
        # save training data and loss
        training_path = main_path + 'training_data/'
        helpers.check_path(training_path)
        
        f = open(training_path+'training_data_model'+
                 str(params['model_number'])+'.pckl','wb')
        pickle.dump(track_training,f)
        f.close()
    
    print('Model saved')

def load_model(path,params,device):
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
    # path = path+'nrec'+str(params['n_rec'])+'/lr'+\
    #     str(params['learning_rate'])+'/'
    
    if device.type == 'cuda':
        model = torch.load(path+'model'+str(params['model_number']))
    else:
         model = torch.load(path+'model'+str(params['model_number']),
                            map_location=torch.device('cpu'))
    print('.... Loaded')
    return model
      
def eval_model(model,test_data,params,save_path):
    """
    Parameters
    ----------
    model :
    
    test_data :
        
    params : dictionary 
    
    save_path : str
   

    Returns
    -------
    eval_data :
        
    pca_data:
        
    rdm_data:
        
    model_outputs:
        
    """
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
    # n_samples = params['batch_size']//params['M']
    n_samples = test_data['inputs'].shape[1]//params['M']
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
    # B = params['batch_size'] // params['B']
    rdm_data = output.permute(1,0,2).unsqueeze(0)
    # rdm_data = torch.reshape(rdm_data,(B,params['B'],params['seq_len'],params['n_rec']))
    # rdm_data = torch.mean(rdm_data,1) # bin uncued
    
    # # average across instances
    
    # rdm_data = torch.reshape(rdm_data.unsqueeze(0),(params['n_stim']*2,params['B'],params['seq_len'],params['n_rec']))
    # rdm_data = torch.reshape(rdm_data.unsqueeze(0),(params['B']*2,params['B'],params['B'],params['seq_len'],params['n_rec']))
    # rdm_data = torch.mean(rdm_data,1) # average across cued
    # rdm_data = torch.reshape(rdm_data,(params['B']*2*params['B'],params['seq_len'],params['n_rec'])) # condition, time, nrec
    
    # save_data(rdm_data,params,save_path+'rdm_data_model')
    # print('.... Data saved')
    # f = open(dpath+'rdm_data_model'+str(model_number)+'.pckl','wb')
    # pickle.dump(rdm_data,f)
    # f.close()
    
    model_outputs = {'data':readout.squeeze(),'labels':{"loc":test_data['loc'],
                      "c1":test_data['c1'],
                      "c2":test_data['c2']}}
    
    save_data(model_outputs,params,save_path+'model_outputs_model')
    
    print('.... Data saved')
    print('Warning - rdm data not calculated correctly')
    
    return eval_data,pca_data,rdm_data,model_outputs

def save_data(data,params,save_path):
    """
    Saves specified data structures.
    
    Parameters
    ----------
    data : array-like or dictionary
    
    params : dictionary 
    
    save_path : str
    
    Returns
    -------
    None
    """
    
    if not params:
        # save data without specifying model number
        f = open(save_path+'.pckl','wb')
    else:
        f = open(save_path+str(params['model_number'])+'.pckl','wb')
    pickle.dump(data,f)
    f.close()
    
def add_noise(data,params,device):
    """
    Adds iid noise to the input data fed to the network. Noise is drawn from 
    the specified distribution, 
    either:
    ~ U(-params['epsilon'],params['epsilon']) 
    or:
    ~ N(0,params['epsilon'])
    and added to the base input data.
    Activation values are then constrained to be within the [0,1] range.
    Use this function to add noise for each epoch instead of creating a new 
    (noisy) dataset from scratch to speed up training.
    
    Parameters
    ----------
    data : torch.Tensor (params['seq_len'],params['batch_size'],params['n_inp'])
        Tensor containing the base input data for the network
        
    params : dictionary 
        params['epsilon'] controls bound / s.d. for the noise distribution
        params['noise_distr'] specifies the distribution from which noise will 
            be drawn (standard normal or uniform)
        
    device : torch.device object
        Must match the location of the model

    Returns
    -------
    data_noisy : torch.Tensor 
        Contains a copy of the original input data with added noise.
    """
    # data_noisy = data.clone() + torch.rand(data.shape).to(device)*params['epsilon']
    
    if params['noise_timesteps'] == 'all':
        if params['noise_distr'] == 'normal':
            data_noisy = data.clone() + (torch.randn(data.shape).to(device))*params['epsilon']
        elif params['noise_distr'] == 'uniform':
            data_noisy = data.clone() + (torch.rand(data.shape).to(device)-0.5)*params['epsilon']
    else:
        data_noisy = data.clone()
        for t in range(len(params['noise_timesteps'])):
            if params['noise_distr'] == 'normal':
                data_noisy[params['noise_timesteps'][t],:,:] = \
                    data_noisy[params['noise_timesteps'][t],:,:] + \
                (torch.randn((1,data.shape[1],data.shape[2]))).to(device)*params['epsilon']
            elif params['noise_distr'] == 'uniform':
                data_noisy[params['noise_timesteps'][t],:,:] = \
                    data_noisy[params['noise_timesteps'][t],:,:] + \
                (torch.rand((1,data.shape[1],data.shape[2]))-0.5).to(device)*params['epsilon']
            
                
    # check if adding noise has caused any activations to be outside of the [0,1] range - if so, fix
    above_ix = torch.where(data_noisy>1)
    below_ix = torch.where(data_noisy<0)
    if above_ix:
        data_noisy[above_ix] = 1
    if below_ix:
        data_noisy[below_ix] = 0

    return data_noisy    
