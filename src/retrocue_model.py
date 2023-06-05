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
import math
import os
import helpers
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema
# from pycircstat import mean as circ_mean
# from pycircstat.descriptive import cdiff

# import pdb
# import custom_plot as cplot

from generate_data_vonMises import change_cue_validity
import helpers
import pdb

import random

def seed_torch(seed=1029):
    '''
    Set the seed for all packages to ensure reproducibility.

    Parameters
    ----------
    seed : int, optional
        Seed value. The default is 1029.

    Returns
    -------
    None.

    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


class RNN(nn.Module):
    def __init__(self, params,device):
        super(RNN, self).__init__()
        # PARAMETERS
        self.n_rec = params['n_rec'] # number of recurrent neurons
        self.n_inp = params['n_inp']
        self.n_out = params['n_out']
        self.device = device
        self.noise_sigma = params['sigma']
        self.noise_distr = params['noise_distr']
        self.noise_timesteps = params['noise_timesteps']
        
        # set seed
        torch.manual_seed(params['model_number'])
        
        # LAYERS
        # note: We need to add noise after the Wrec @ ht-1 step, but before the nonlinearity.
        # So, we cannot use the torch RNN class, because it does it all in one step.

        # input layer
        self.inp = nn.Linear(self.n_inp,self.n_rec)
        self.inp.weight = nn.Parameter(self.inp.weight*params['init_scale'])  # Xavier init
        self.inp.bias = nn.Parameter(self.inp.bias*params['init_scale'])  # Xavier init

        # recurrent layer
        self.Wrec = nn.Parameter(torch.nn.init.orthogonal_(torch.empty((self.n_rec, self.n_rec))))  # orthogonal init
        self.relu = nn.ReLU()
        
        # output layer
        self.out = nn.Linear(self.n_rec, self.n_out) # output layer
        self.out.weight = nn.Parameter(self.out.weight*params['init_scale'])  # Xavier init
        self.out.bias = nn.Parameter(self.out.bias*params['init_scale'])  # Xavier init
        self.softmax = nn.Softmax(dim=-1)
       
    def step(self,input_ext,hidden,noise):
        hidden = self.relu(self.inp(input_ext.unsqueeze(0)) + hidden @ self.Wrec.T \
            + noise)
        h = hidden.clone().detach()
        # We need to detach the hidden state to be able to save it into a matrix
        # in the forward method, otherwise it messes with the computational graph.
        
        return h, hidden

    
    def forward(self, inputs):
        """
        Run the RNN with the input time course.
        """

        # Add noise to hidden units
        seq_len = inputs.shape[0]
        batch_size = inputs.shape[1]
        # To add hidden noise, need to use the expanded implementation below:
        
        # Initialize network state
        # hidden states from current time point
        hidden = torch.zeros((1,inputs.size(1), self.n_rec),device = self.device) # 0s
        # hidden states from all timepoints
        h = torch.empty((seq_len, batch_size, self.n_rec), device = self.device)

        # Run the input through the network - across time
        for timepoint in range(seq_len):
            if len(np.where(self.noise_timesteps == timepoint)[0]) > 0:
                # Add Gaussian noise to appropriate timesteps of the trial
                noise = (torch.randn(hidden.size(),device = self.device))*self.noise_sigma
            else:
                # timestep without noise
                noise = torch.zeros(hidden.size(),device = self.device)

            # if i == seq_len - 1:
            #     # last time step of the sequence
            #     o[i,:,:], h_n = self.step(inputs[i, :, :], hidden,noise)
            #     # h_n = self.do(h_n)
            # else:
            h[timepoint,:,:], hidden = self.step(inputs[timepoint, :, :], hidden,noise)
                
        # pass the recurrent activation from the last timestep through the decoder layer
        output = self.out(hidden)
        output = self.softmax(output)
        return output.squeeze(), h, hidden


def train_model(params, data, device):
    '''
    Train the RNN model and save it, along with the training details.

    Parameters
    ----------
    params : dict
        Experiment parameters.
    data : TYPE
        DESCRIPTION.
    device : TYPE
        DESCRIPTION.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    model : torch object
        DESCRIPTION.
    track_training : dict
        DESCRIPTION.

    '''
    
    # set seed for reproducibility
    torch.manual_seed(params['model_number'])
    
    #% initialise model
    model = RNN(params,device)
    
    if not(params['from_scratch']):
        # read in the pre-trained model
        model_path = params['FULL_PATH'] + 'saved_models/'
        model = load_model(model_path,params,device)
    
    # transfer model to the desired device
    model.to(device)
    
    # set the optimiser
    if params['optim'] == 'SGD':
        optimizer = optim.SGD(model.parameters(),lr=params['learning_rate'])#,momentum=.9)
    elif params['optim'] == 'SGDm':
        optimizer = optim.SGD(model.parameters(),lr=params['learning_rate'],momentum=.9)
    elif params['optim'] == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(),lr=params['learning_rate'])
    elif params['optim'] == 'Adam':
        optimizer = optim.Adam(model.parameters(),lr=params['learning_rate'])
    
    # set the loss function
    if params['loss_fn'] == 'MSE':
        loss_fn = torch.nn.MSELoss()
    elif params['loss_fn'] == 'CEL':
        loss_fn = torch.nn.CrossEntropyLoss()

            
    
    
    n_valid_trials = int(params['cue_validity']*params['stim_set_size'])
    n_invalid_trials = params['stim_set_size'] - n_valid_trials
    
    loss_all = torch.empty(params['stim_set_size'],
                           params['n_epochs']).to(device)
    loss_valid = torch.empty(n_valid_trials,
                             params['n_epochs']).to(device)
    loss_epoch = torch.empty((params['n_epochs'],)).to(device)
    shuffling_order = torch.empty((params['stim_set_size'],
                                   params['n_epochs']),
                                  dtype=torch.long).to(device)
    net_outputs = torch.empty((params['stim_set_size'],
                               params['n_colCh'],
                               params['n_epochs'])).to(device)
    
    # mean and std of hidden activations on the first trial of each batch
    hidden_stats = torch.empty((params['n_epochs'],
                                 params['seq_len'],2)).to(device)
    
    if params['condition']!='deterministic':
       invalid_trials = torch.empty((params['n_epochs'],n_invalid_trials)).to(device)
       
    if not params['from_scratch']:
        # load training structures and append to them
        f = open(params['FULL_PATH']+'training_data/'+'training_data_model'+\
                 str(params['model_number'])+'.pckl','rb')
        track_training = pickle.load(f)
        f.close()
        
        loss_all = torch.cat((track_training['loss'].to(device),loss_all),-1)
        loss_valid = torch.cat((track_training['loss_valid'].to(device),loss_valid),-1)
        loss_epoch = torch.cat((track_training['loss_epoch'].to(device),loss_epoch),-1)
        shuffling_order = torch.cat((track_training['shuffling_order'].to(device),shuffling_order),-1)
        net_outputs = torch.cat((track_training['outputs'].to(device),net_outputs),-1)
        if params['condition']!='deterministic':
            invalid_trials = \
                torch.cat((track_training['invalid_trials'].to(device),
                           invalid_trials),0)
        n_previous_epochs = track_training['loss_epoch'].shape[0]
    
    inputs_base = data['inputs']
    inputs_base = inputs_base.to(device)
    
    targets = data['targets']
    targets = targets.to(device)
        

    window = params['conv_criterion']['window']
    # dLoss = torch.empty((params['n_epochs']-1-window,))
    # loss_clean = torch.empty((params['n_epochs']-window,))
   
    if params['from_scratch']:
        epochs = range(params['n_epochs'])
    else:
        epochs = range(n_previous_epochs,n_previous_epochs+params['n_epochs'])
    
    # loop over epochs
    for ix,i in enumerate(epochs):
            # print('Epoch %d : ' %i)         
            # shuffle dataset for SGD
            shuffling_order[:,i] = \
                torch.randperm(params['stim_set_size'],dtype=torch.long).to(device)
            # determine the delay durations on each trial
            if params['var_delays']:                    
                delay_mask = \
                    var_delay_mask(params['delay_mat'][shuffling_order[:,i],:],
                                   params,
                                   device)
                
            # make some cues invalid
            if params['condition']!='deterministic':
                inputs = inputs_base.clone() # create a copy of the inputs
                inputs, ixs = change_cue_validity(inputs,params) # change some trials
                # saved the ixs
                invalid_trials[i,:] = torch.tensor(ixs)
            else:
                inputs = inputs_base
            
            # loop over training examples
            for trial in range(params['stim_set_size']):
                # print('Trial %d : ' %trial)   
                if params['var_delays']:
                    trial_input = inputs[delay_mask[:,trial],
                                         shuffling_order[trial,i],:]                    
                else:
                    trial_input = inputs[:,shuffling_order[trial,i],:]
                
                outputs, o, hidden = \
                    model(trial_input.unsqueeze(1))
                # print(torch.where(o<0))
                if np.logical_and(trial == 0,i==0):
                    # hidden_stats[i,:,0] = \
                    #     torch.std_mean(o,-1)[0].squeeze()
                    # hidden_stats[i,:,1] = \
                    #     torch.std_mean(o,-1)[1].squeeze()
                    print('First forward pass')
                    print('    Means:')
                    print(torch.std_mean(o,-1)[0].squeeze())
                    print('    S.d.:')
                    print(torch.std_mean(o,-1)[1].squeeze())
                
                # Compute loss
                if params['target_type'] == 'Gaussian':
                    loss = loss_fn(outputs.unsqueeze(0), 
                                    targets
                                    [shuffling_order
                                    [trial,i],:].unsqueeze(0))
                elif params['target_type'] == 'class_label':
                    loss = loss_fn(outputs.unsqueeze(0), 
                                    targets[shuffling_order[trial,i]].unsqueeze(0))
                elif params['target_type']=='angle_val':
                    loss = custom_MSE_loss(params,outputs,targets[shuffling_order[trial,i]])
                
                # pdb.set_trace()
                # Keep track of outputs and loss
                # if params['l2_activity']:
                #     #loss += params['Br']*torch.norm(torch.norm(o.squeeze(),dim=1))**2
                #     loss += (1/(model.n_inp*params['seq_len']))*params['Br']*\
                #         torch.sum(torch.norm(o.squeeze(),dim=1)**2)
                if params['target_type']=='angle_val':
                    loss_all[trial,i] = loss.detach()
                else:
                    loss_all[trial,i] = loss.item()
                
                net_outputs[trial,:,i] = outputs.detach()
                # Compute gradients
                optimizer.zero_grad()
                loss.backward()
                # Update weights
                optimizer.step()
            if params['condition']!='deterministic':
                valid_ix = torch.from_numpy(np.setdiff1d(np.arange(params['stim_set_size']),invalid_trials[i,:]))
                
            else:
                valid_ix = torch.arange(params['stim_set_size'])
                
            if len(valid_ix) != n_valid_trials:
                ValueError('loss_valid has a wrong preallocated size!')
                
            loss_valid[:,i] = loss_all[valid_ix,i]
            
            # mean over epoch - only valid trials
            loss_epoch[i] = loss_all[valid_ix,i].mean()
            
            # print progress
            # for non-deterministic conditions, only show loss on valid trials                
            if ((ix*100/params['n_epochs'])%25)==0:
                print('Model %2d :    %.2f%% iterations of SGD completed...loss = %.5f' \
                          % (int(params['model_number']),
                              100* (ix + 1) / params['n_epochs'],
                              loss_epoch[i]))
                # print_progress(i, params['n_epochs'])
            if (ix==params['n_epochs']-1):
                print('Model %2d :    100%% iterations of SGD completed...loss = %.5f' \
                          % (int(params['model_number']),
                              loss_epoch[i]))
            
            
            # if loss reaches the required level, stop training            
            
            # select the loss values to be used for the convergence criterion
            if params['criterion_type'] == 'abs_loss':
                criterion_reached = (loss_epoch[i] <= params['MSE_criterion'])
            elif params['criterion_type'] == 'loss_der':
                if i<window:
                    criterion_reached = False
                else:
                    criterion_reached = \
                        apply_conv_criterion(params,loss_epoch[i-window+1:i+1])
            else:
                raise ValueError('Specify which convergence criterion to use')
                            
            
            if criterion_reached:
                print('Model %2d converged after %d epochs. End loss = %.5f' \
                       % (int(params['model_number']), i+1,
                          loss_epoch[i]))
                # prune preallocated tensors
                net_outputs = net_outputs[:,:,:i+1]
                loss_all = loss_all[:,:i+1]
                shuffling_order = shuffling_order[:,:i+1]
                # dLoss = dLoss[:i+1]
                # loss_clean = loss_clean[:i+1]
                loss_epoch = loss_epoch[:i+1]
                break
    
    
    # save training data
    track_training = {}
    track_training['loss'] = loss_all
    track_training['loss_valid'] = loss_valid
    track_training['loss_epoch'] = loss_epoch
    track_training['shuffling_order'] = shuffling_order
    track_training['outputs'] = net_outputs
    # mean and std of hidden activations on the first trial of each batch
    track_training['hidden_stats'] = hidden_stats
    # track_training['dLoss'] = dLoss
    # track_training['loss_clean'] = loss_clean
    
    if params['condition']!='deterministic':
        track_training['invalid_trials'] = invalid_trials
        
        
    return model, track_training


# def partial_training(params,data,trial_sequence,device):
#     """
#     Train the RNN model up to some point using a fixed trial order sequence and
#     save it.
    
#     Parameters
#     ----------
    
#     params : dictionary 
       
#     loss_fn : torch.nn modeule
        
#     data : dictionary
    
#     trial_sequence: numpy array, contains shuffling order of trials used for SGD
        

#     Returns
#     -------
#     model : torch object
#         Object created by the sklearn.decomposition.PCA method.
#         fitted_plane.components_ gives the plane vectors
#     """
#     # set seed for reproducibility
#     torch.manual_seed(params['model_number'])
    
#     #% initialise model
#     # model, net_type = define_model(params['n_inp'],
#     #                                params['n_rec'],
#     #                                params['n_colCh'])

#     # transfer model to GPU if available
#     model.to(device)
    
#     if params['optim'] == 'SGD':
#         optimizer = optim.SGD(model.parameters(),lr=params['learning_rate'])#,momentum=.9)
#     elif params['optim'] == 'SGDm':
#         optimizer = optim.SGD(model.parameters(),lr=params['learning_rate'],momentum=.9)
#     elif params['optim'] == 'RMSprop':
#         optimizer = optim.RMSprop(model.parameters(),lr=params['learning_rate'])
    
    
#     if params['loss_fn'] == 'MSE':
#         loss_fn = torch.nn.MSELoss()
#     elif params['loss_fn'] == 'CEL':
#         loss_fn = torch.nn.CrossEntropyLoss()
    
    
#     track_training = {}
#     track_training['loss'] = torch.zeros(params['batch_size'],params['n_epochs'])
#     track_training['shuffling_order'] = torch.zeros((params['batch_size'],
#                                                      params['n_epochs']),
#                                                     dtype=torch.long)
#     track_training['outputs'] = torch.zeros((params['batch_size'],
#                                              params['n_colCh'],
#                                              params['n_epochs']))
#      # loop over epochs
#     for i in range(params['n_epochs']):            
#             # shuffle dataset for SGD
            
#             # loop over training examples
#             for trial in range(params['batch_size']):
                
#                 outputs, o, hidden = \
#                     model(data['inputs'][:,trial_sequence[trial,i],:].unsqueeze(1))
                

#                 # Compute loss
#                 loss = loss_fn(outputs.unsqueeze(0), 
#                                data['targets']
#                                [trial_sequence[trial,i],:].unsqueeze(0))
#                 # Keep track of outputs and loss
#                 track_training['loss'][trial,i] = loss.item()
#                 track_training['outputs'][trial,:,i] = outputs.detach()
#                 # Compute gradients
#                 optimizer.zero_grad()
#                 loss.backward()
#                 # Update weights
#                 optimizer.step()
            
#             # print progress
#             if (i%100 == 0):
#                 print('Model %2d :    %.2f%% iterations of SGD completed...loss = %.2f' \
#                           % (int(params['model_number']),
#                              100* (i + 1) / params['n_epochs'],
#                              torch.sum(track_training['loss'][:,i])))
#             if (i==params['n_epochs']-1):
#                 print('Model %2d :    100%% iterations of SGD completed...loss = %.2f' \
#                           % (int(params['model_number']),
#                              torch.sum((track_training['loss'][:,i]))))
#     return model, track_training
  
  
def save_model(path,params,model,track_training):
    '''
    Save the torch model along with the training data.

    Parameters
    ----------
    path : str
        Path to the experiment folder.
    params : dict
        Experiment parameters.
    model : torch object
        Model to save.
    track_training : dict
        Training data.

    Returns
    -------
    None.

    '''
    # check if paths exist - if not, create them
    
    model_path = path + 'saved_models/'
    data_path = path + 'training_data/'
    
    helpers.check_path(model_path)
    helpers.check_path(data_path)

    # save model
    torch.save(model,model_path+'model' + str(params['model_number']))
    torch.save(model.state_dict(),model_path+'model'+
               str(params['model_number'])+'_statedict')
    # save training data, including loss    
    f = open(data_path+'training_data_model'+
             str(params['model_number'])+'.pckl','wb')
    pickle.dump(track_training,f)
    f.close()
    
    print('Model saved')


def load_model(path,params,device):
    '''
    Load model from file.

    Parameters
    ----------
    path : str
        Path to model file.
    params : dict
        Experiment parameters, incuding model number.
    device : torch obj
        Device to put them model on.

    Returns
    -------
    model : torch obj
        RNN model.

    '''
    
    if device.type == 'cuda':
        model = torch.load(path+'model'+str(params['model_number']))
    else:
         model = torch.load(path+'model'+str(params['model_number']),
                            map_location=torch.device('cpu'))
    print('.... Loaded')
    return model
      

def eval_model(model,test_data,params,save_path,trial_type='valid'):
    '''

    Parameters
    ----------
    model : TYPE
        DESCRIPTION.
    test_data : TYPE
        DESCRIPTION.
    params : TYPE
        DESCRIPTION.
    save_path : TYPE
        DESCRIPTION.
    trial_type : str, optional
        Trial type label, relevant for experiment 3. Set to either 'valid' or 
        'invalid'. Default is valid.

    Returns
    -------
    eval_data : TYPE
        DESCRIPTION.
    pca_data : TYPE
        DESCRIPTION.
    rdm_data : TYPE
        DESCRIPTION.
    model_outputs : TYPE
        DESCRIPTION.

    '''

    
    model.to(torch.device('cpu')) # put model on cpu
    
    #% evaluate model on test data after freezing the weights
    model.eval()
    with torch.no_grad():
        if params['noise_type']=='hidden':
            readout, hidden_all_timepoints, hidden_T = \
                    model(test_data['inputs'])
            # hidden_T corresponds to the hidden layer activity on the last 
            # timepoint only
        else:
            hidden_all_timepoints, hidden_T = model.Wrec(test_data['inputs'])
            readout = model.out(hidden_T)
    print('.... evaluated')

    # create the full test dataset structure
    eval_data = {"dimensions":['trial','time','n_rec'],
                 "data":hidden_all_timepoints.permute(1,0,-1),
                 "labels":
                     {"loc":test_data['loc'],
                      "c1":test_data['c1'],
                      "c2":test_data['c2']}}
    if params['experiment_number'] == 3:
        # add cued and probed labels
        eval_data['labels']['cued_loc'] = test_data['cued_loc']
        eval_data['labels']['probed_loc'] = test_data['probed_loc']
    
    
    if params['experiment_number']==3:
        if trial_type == 'valid':
           save_data(eval_data,params,save_path+'eval_data_model')
        else:
            save_data(eval_data,params,save_path+'eval_data_uncued_model')
    else:
        # save
        save_data(eval_data,params,save_path+'eval_data_model')
    
    # pca_data: dataset binned by cued colour and averaged across uncued colours
    # M = B*L - colour-bin*location combinations
    
    # data is sorted by the cued colour (probed if experiment 3) automatically
    # when created, so no need to resort
    n_samples = test_data['inputs'].shape[1]//params['M']
    seq_len = test_data['inputs'].shape[0]
    # remove the code below
    # trial_data = torch.reshape(hidden_all_timepoints.permute(1,0,-1).unsqueeze(0),
    #                            (params['M'],n_samples,
    #                             seq_len,params['n_rec']))
    # # bin (i.e. average across n_samples)
    # trial_data = trial_data.mean(1) #(M,seq_len,n_rec)
    
    # trial_data2 = helpers.bin_data(output.permute(1,0,-1),params)
    trial_data = helpers.bin_data(eval_data['data'],params)
  
    
    # extract the end-points of the two delays
    d1_ix = params['trial_timepoints']['delay1_end']-1
    d2_ix = params['trial_timepoints']['delay2_end']-1
    
    if params['experiment_number']==3:
        d3_ix = params['trial_timepoints']['delay3_end']-1
    
    # create labels
    labels = {}
    labels['loc'] = np.zeros((params['M'],),dtype=int)
    labels['loc'][params['B']:] = 1
    labels['col'] = np.concatenate([np.arange(params['B'])]*2,0)
    # labels['c1'] = np.empty((params['M'],))
    # labels['c1'][:params['B']] = np.arange(params['B'])
    # labels['c2'] = np.empty((params['M'],))
    # labels['c2'][params['B']:] = np.arange(params['B'])
    
    pca_data = {'dimensions':['M','time','n_rec'],
                'data':trial_data,'delay1':trial_data[:,d1_ix,:],
                'delay2':trial_data[:,d2_ix,:],
                "labels":labels}
    
    if params['experiment_number']==3:
        pca_data['delay3'] = trial_data[:,d3_ix,:]
        # save
        save_data(pca_data,params,save_path+'pca_data_probed_model')
        if trial_type == 'valid':
            pca_data['labels']['cued_loc'] = np.concatenate((np.zeros((8//2,),dtype=int),
                                                             np.ones((8//2,),dtype=int)))
            pca_data['labels']['probed_loc'] = np.concatenate((np.zeros((8//2,),dtype=int),
                                                               np.ones((8//2,),dtype=int)))
            save_data(pca_data,params,save_path+'pca_data_model')
        else:
            pca_data['labels']['cued_loc'] = np.concatenate((np.ones((8//2,),dtype=int),
                                                             np.zeros((8//2,),dtype=int)))
            pca_data['labels']['probed_loc'] = np.concatenate((np.zeros((8//2,),dtype=int),
                                                               np.ones((8//2,),dtype=int)))
            save_data(pca_data,params,save_path+'pca_data_uncued_model')
    else:
        save_data(pca_data,params,save_path+'pca_data_model')    
    
    # pca_uncued: like above, but binned by uncued colours and averaged across 
    # cued colours
   
    # create labels
    labels_uncued = {}
    labels_uncued['loc_uncued'] = np.zeros((params['M'],),dtype=int)
    labels_uncued['loc_uncued'][:params['B']] = 1
    labels_uncued['col'] = np.concatenate([np.arange(params['B'])]*2,0)
    
    # sort and bin
    eval_data_uncued,full_sorting_ix = helpers.sort_by_uncued(eval_data,params)
    trial_data_uncued = helpers.bin_data(eval_data_uncued['data'],params)
    
    pca_data_uncued =  {'dimensions':['M','n_rec'],
                        'data':trial_data_uncued,
                        'delay1':trial_data_uncued[:,d1_ix,:],
                        'delay2':trial_data_uncued[:,d2_ix,:],
                        "labels":labels_uncued}
    if params['experiment_number']==3:
        pca_data_uncued['delay3'] = trial_data_uncued[:,d3_ix,:]
        # save
        save_data(pca_data_uncued,params,save_path+'pca_data_unprobed_model')
        if trial_type == 'valid':
            pca_data_uncued['labels']['cued_loc'] = np.concatenate((np.zeros((8//2,),dtype=int),
                                                             np.ones((8//2,),dtype=int)))
            pca_data_uncued['labels']['probed_loc'] = np.concatenate((np.zeros((8//2,),dtype=int),
                                                               np.ones((8//2,),dtype=int)))
            save_data(pca_data_uncued,params,save_path+'pca_data_uncued_model')
            save_data(eval_data_uncued,params,save_path+'eval_data_uncued_model')
        else:
            pca_data_uncued['labels']['cued_loc'] = np.concatenate((np.ones((8//2,),dtype=int),
                                                             np.zeros((8//2,),dtype=int)))
            pca_data_uncued['labels']['probed_loc'] = np.concatenate((np.zeros((8//2,),dtype=int),
                                                               np.ones((8//2,),dtype=int)))
            save_data(pca_data_uncued,params,save_path+'pca_data_model')
            save_data(eval_data_uncued,params,save_path+'eval_data_model')
    else:
        # save
        save_data(pca_data_uncued,params,save_path+'pca_data_uncued_model')
    
    # rdm data - averaged across uncued    - rewrite this so it makes sense, but works for now
    # B = params['batch_size'] // params['B']
    rdm_data = hidden_all_timepoints.permute(1,0,-1).unsqueeze(0) # time x trial x n_rec
    
    n_trial_instances = hidden_all_timepoints.shape[1] // (params['n_stim']**2 * params['L'])
    n_samples = n_trial_instances * params['n_stim'] # n samples for each cued colour
    n_cued_conditions = params['n_stim'] * params['L'] # n of colours x locations
    rdm_data = torch.reshape(rdm_data,(n_cued_conditions,n_samples,seq_len,params['n_rec']))
    
    # rdm_labels1 = torch.reshape(test_data['c1'],(n_cued_conditions,n_samples))
    # rdm_labels2 = torch.reshape(test_data['c2'],(n_cued_conditions,n_samples))
    
    rdm_data = torch.mean(rdm_data,1) # bin uncued
    
    # # average across instances
    
    # rdm_data = torch.reshape(rdm_data.unsqueeze(0),(params['n_stim']*2,params['B'],params['seq_len'],params['n_rec']))
    # rdm_data = torch.reshape(rdm_data.unsqueeze(0),(params['B']*2,params['B'],params['B'],params['seq_len'],params['n_rec']))
    # rdm_data = torch.mean(rdm_data,1) # average across cued
    # rdm_data = torch.reshape(rdm_data,(params['B']*2*params['B'],params['seq_len'],params['n_rec'])) # condition, time, nrec
    
    save_data(rdm_data,params,save_path+'rdm_data_model')
    # f = open(dpath+'rdm_data_model'+str(model_number)+'.pckl','wb')
    # pickle.dump(rdm_data,f)
    # f.close()
    
    model_outputs = {'data':readout.squeeze(),'labels':{"loc":test_data['loc'],
                      "c1":test_data['c1'],
                      "c2":test_data['c2']}}
    
    save_data(model_outputs,params,save_path+'model_outputs_model')
    
    
        
    
    print('.... and data saved')
    # print('Warning - rdm data not calculated correctly')
    
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
    ~ U(-params['sigma'],params['sigma']) 
    or:
    ~ N(0,params['sigma'])
    and added to the base input data.
    Activation values are then constrained to be within the [0,1] range.
    Use this function to add noise for each epoch instead of creating a new 
    (noisy) dataset from scratch to speed up training.
    
    Parameters
    ----------
    data : torch.Tensor (params['seq_len'],params['batch_size'],params['n_inp'])
        Tensor containing the base input data for the network
        
    params : dictionary 
        params['sigma'] controls bound / s.d. for the noise distribution
        params['noise_distr'] specifies the distribution from which noise will 
            be drawn (standard normal or uniform)
        
    device : torch.device object
        Must match the location of the model

    Returns
    -------
    data_noisy : torch.Tensor 
        Contains a copy of the original input data with added noise.
    """
    # data_noisy = data.clone() + torch.rand(data.shape).to(device)*params['sigma']
    
    if params['noise_period'] == 'all':
        if params['noise_distr'] == 'normal':
            data_noisy = data.clone() + (torch.randn(data.shape).to(device))*params['sigma']
        elif params['noise_distr'] == 'uniform':
            data_noisy = data.clone() + (torch.rand(data.shape).to(device)-0.5)*params['sigma']
    else:
        data_noisy = data.clone()
        for t in range(len(params['noise_timesteps'])):
            if params['noise_distr'] == 'normal':
                data_noisy[params['noise_timesteps'][t],:,:] = \
                    data_noisy[params['noise_timesteps'][t],:,:] + \
                (torch.randn((1,data.shape[1],data.shape[2]))).to(device)*params['sigma']
            elif params['noise_distr'] == 'uniform':
                data_noisy[params['noise_timesteps'][t],:,:] = \
                    data_noisy[params['noise_timesteps'][t],:,:] + \
                (torch.rand((1,data.shape[1],data.shape[2]))-0.5).to(device)*params['sigma']
            
                
    # check if adding noise has caused any activations to be outside of the [0,1] range - if so, fix
    above_ix = torch.where(data_noisy>1)
    below_ix = torch.where(data_noisy<0)
    if above_ix:
        data_noisy[above_ix] = 1
    if below_ix:
        data_noisy[below_ix] = 0

    return data_noisy    

# def var_delay(delay_ix,trial_data,params,device):
#     """
#     Adds iid noise to the input data fed to the network. Noise is drawn from 
#     the specified distribution, 
#     either:
#     """
#     if not params['var_delays']:
#         return;

#     delay1_len = params['delay_mat'][delay_ix,0]
#     delay2_len = params['delay_mat'][delay_ix,1]
    
#     delay1 = torch.cat([trial_data[params['trial_timepoints']['delay1_start'],:].unsqueeze(0)]*delay1_len)
#     delay2 = torch.cat([trial_data[params['trial_timepoints']['delay2_start'],:].unsqueeze(0)]*delay2_len)
    
#     input_trial = torch.cat((trial_data[:params['trial_timepoints']['delay1_start'],:],
#                              delay1,
#                              trial_data[params['trial_timepoints']['delay1_end']:params['trial_timepoints']['delay2_start'],:],
#                              delay2,
#                              trial_data[params['trial_timepoints']['delay2_end']:,:]))
#     return input_trial

def var_delay_mask(delay_mat,params):
    '''
    Generate a mask to be used with the input data to modify the delay length.

    Parameters
    ----------
    delay_mat : array (n_trials,2)
        Trial-wise delay length values in cycles, for both delays.
    params : dict
        Experiment parameters.

    Returns
    -------
    delay_mask : array
        Boolean mask for the input data array mnodifying the delay lengths on 
        each trial.

    '''
    if not params['var_delays']:
        return;

    delay_mask = torch.ones((params['seq_len'],params['stim_set_size']),dtype=bool)
    
    for trial in range(params['stim_set_size']):
        delay1_len = delay_mat[trial,0]
        delay2_len = delay_mat[trial,1]
        
        delay_mask[params['trial_timepoints']['delay1_start']+delay1_len:\
                   params['trial_timepoints']['delay1_end'],trial]=False
        delay_mask[params['trial_timepoints']['delay2_start']+delay2_len\
                   :params['trial_timepoints']['delay2_end'],trial]=False
    
    return delay_mask



def get_dLoss_dt(params,loss_vals):
    '''
    Calculate the derivative of the loss function wrt time. Used for finding 
    learning plateaus.

    Parameters
    ----------
    params: dict
        dictionary containing the Gaussian filter s.d. in
        params['conv_criterion']['smooth_sd']
    loss_vals : torch.tensor
        loss values for every epoch (averaged across all training examples).
    
   
    Returns
    -------
    dLoss : array
        Derivative of the loss wrt to time.
    loss_clean : array
        Loss values after smoothing

    '''
    if len(loss_vals.shape)<2:
        ValueError('Loss_vals can''t be a 1-dimensional array')

    # convolve with a Gaussian filter to smooth the loss curve
    loss_clean = gaussian_filter1d(loss_vals,params['conv_criterion']['smooth_sd'])
    loss_clean = torch.tensor(loss_clean)
    
    # calculate the derivative
    dLoss = torch.zeros(loss_clean.shape[0]-1)
    for i in range(loss_clean.shape[0]-1):
            dLoss[i] = loss_clean[i+1] - loss_clean[i]
    return dLoss, loss_clean


def get_loss_slope(params,loss_vals):
    '''
    Get the slope of the loss curve over a window of trials (saved in the 
    experiment parameters dictionary).

    Parameters
    ----------
    params : dict
        experiment parameters.
    loss_vals : array
        Vector of loss values from different trials.

    Returns
    -------
    slope : float
        Fitted slope value.

    '''
    window = params['conv_criterion']['window']
    p = np.polyfit(np.arange(window), loss_vals, 1)
    return p[0]


def apply_conv_criterion(params,loss_vals):
    '''
    Apply the convergence criterion to determine whether training should 
    conclude. Two conditions must be satisfied:
        1) slope of the training loss mus be negative and >= threshold
        2) all of the training loss values must fall below their threshold
    The second condition ensures that training is not stopped prematurely, at 
    the mid-training plateau.

    Parameters
    ----------
    params : dict
        Experiment parameters.
    loss_vals : array
        Training loss values from recent trials.

    Returns
    -------
    criterion_reached : bool
        Flag, returns True if the two convergence conditinos have been met.

    '''
    # get the loss slope
    a =  get_loss_slope(params,loss_vals)
    
    cond1 = np.logical_and(a >= params['conv_criterion']['thr_slope'],
                            a <= 0)
    cond2 = torch.all(loss_vals < params['conv_criterion']['thr_loss'])
      
    criterion_reached = np.logical_and(cond1,cond2)
    return criterion_reached


def custom_MSE_loss(params,output,target_scalar):
    '''
    Loss function for network training. The loss term is given by the mean 
    squared product of the (i) difference between the target and output vectors
    and (ii) the circular distance between the output unit tuning centres and
    the cued colour value (in radians).

    Parameters
    ----------
    params : dict
        Experiment parameters.
    output : array (n_out,)
        Network output on the last timepoint of the trial.
    target_scalar : array (1,)
        Target vector, encoded as a circular value of the cued colour.

    Returns
    -------
    loss : TYPE
        DESCRIPTION.

    '''
    
    # get the 1-hot representation of the target
    target_1hot = make_target_1hot(params,target_scalar)
    # calculate the circular distance between the output tuning centres 
    # and the cued colour value
    circ_dist = helpers.circ_diff(params['phi'],target_scalar)
    # calculate the full loss
    loss = ((circ_dist*(target_1hot - output))**2).mean()
    return loss

def make_target_1hot(params,target_scalar):
    '''
    Convert a scalar target into a 1-hot vector, where the individual rows 
    correspond to the output units (associated with different colour tuning 
    peaks).

    Parameters
    ----------
    params : dict
        Experiment parameters.
    target_scalar : float
        Cued colour value in radians.

    Returns
    -------
    target_1hot : array (n_out,)
        Target vector with a 1 at the row corresponding to the output unit with 
        a tuning curve centered on the cued colour value.

    '''
    target_1hot = torch.zeros((len(params['phi']),))
    target_1hot[torch.where(params['phi']==target_scalar)[0]]=1
    return target_1hot


    
