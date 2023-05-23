#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 22:07:39 2021

@author: emilia
"""

import numpy as np
import torch
from torch import nn, optim, autograd
import pickle
import os.path
import helpers
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema

from torch.utils.data import DataLoader


# import pdb
import custom_plot as cplot

from generate_data_vonMisses import change_cue_validity

import pdb



#%%

# from sklearn.model_selection import StratifiedKFold

# class StratifiedBatchSampler:
#     """Stratified batch sampling
#     Provides equal representation of target classes in each batch
#     """
#     def __init__(self, y, batch_size, shuffle=True):
#         if torch.is_tensor(y):
#             y = y.numpy()
#         assert len(y.shape) == 1, 'label array must be 1D'
#         n_batches = int(len(y) / batch_size)
#         self.skf = StratifiedKFold(n_splits=n_batches, shuffle=shuffle)
#         self.X = torch.randn(len(y),1).numpy()
#         self.y = y
#         self.shuffle = shuffle

#     def __iter__(self):
#         if self.shuffle:
#             self.skf.random_state = torch.randint(0,int(1e8),size=()).item()
#         for train_idx, test_idx in self.skf.split(self.X, self.y):
#             yield test_idx

#     def __len__(self):
#         return len(self.y)



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
                self.inp.weight = nn.Parameter(self.inp.weight*params['init_scale']) # Xavier init
                self.inp.bias = nn.Parameter(self.inp.bias*params['init_scale']) # Xavier init
                
                self.Wrec = nn.Parameter(torch.randn(self.n_rec, self.n_rec)\
                                          * (params['init_scale']/ np.sqrt(self.n_rec)))  # recurrent weights - Gaussian init
                
                self.Wrec = nn.Parameter(torch.nn.init.orthogonal_(torch.empty((self.n_rec, self.n_rec))))
                # Wrec = I + Σoff
                #self.Wrec = torch.diag(torch.ones(self.n_rec)) + \
                #    torch.abs(1-torch.diag(torch.ones(self.n_rec)))*\
                #        (torch.randn(self.n_rec, self.n_rec) / np.sqrt(self.n_rec))
                #self.Wrec = nn.Parameter(self.Wrec)
                self.relu = nn.ReLU()
                self.softmax = nn.Softmax(dim=-1)
                # self.do = nn.Dropout()

            else:
                
                
                # # _____________________________________________________ # 
                # # implementation for simulations without hidden noise:
                    
                self.Wrec = nn.RNN(self.n_inp,self.n_rec, nonlinearity = 'relu')
                
                self.Wrec.bias_hh_l0 = nn.Parameter(self.Wrec.bias_hh_l0*params['init_scale'])
                self.Wrec.bias_ih_l0 = nn.Parameter(self.Wrec.bias_ih_l0*params['init_scale'])
                self.Wrec.weight_hh_l0 = nn.Parameter(self.Wrec.weight_hh_l0*params['init_scale'])
                self.Wrec.weight_ih_l0 = nn.Parameter(self.Wrec.weight_ih_l0*params['init_scale'])
            
            # output layer
            self.out = nn.Linear(self.n_rec, self.n_out) # output layer
            # self.out.weight = nn.Parameter(torch.ones(self.out.weight.shape))
            self.out.weight = nn.Parameter(self.out.weight*params['init_scale']) #Xavier init
            # self.out.weight = nn.Parameter(torch.randn((self.n_out,self.n_rec))\
            #                                *(params['init_scale']/ np.sqrt(self.n_rec))) #Gaussian init
            
            self.out.bias = nn.Parameter(self.out.bias*params['init_scale']) #Xavier init
            # self.out.bias = nn.Parameter(self.out.bias*0) # set bias to 0
            #self.out.bias = nn.Parameter(torch.randn((self.n_out,))\
            #                               *(params['init_scale']/ np.sqrt(self.n_rec))) #Gaussian init
           
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
                # to add hidden noise to each timestep inside the activation func
                o, h_n = self.Wrec(inputs)
                
            elif params['noise_type']=='hidden':
                # Add noise to hidden units
                seq_len = inputs.shape[0]
                batch = inputs.shape[1]
                # To add hidden noise, need to use the expanded implementation below:
                
                # Initialize network state
                hidden = torch.zeros((1,inputs.size(1), self.n_rec),device = device) # 0s
            
                # Run the input through the network - across time
                o = torch.empty((seq_len,batch,self.n_rec),device = device)
                for i in range(seq_len):
                    
                    if len(np.where(params['noise_timesteps']==i)[0])>0:
                        # timestep with added noise
                        if params['noise_distr'] == 'uniform':
                            noise = (torch.rand(hidden.size(),device = device)-0.5)*params['epsilon']
                        elif params['noise_distr'] == 'normal':
                            # Add Gaussian noise to appropriate timesteps of the trial
                            if params['multiple_sds']:
                                # if want to add different levels of noise to different timepoints
                                # i.e. probe and delays
                                if i == seq_len-1:
                                    # probe
                                    noise = (torch.randn(hidden.size(),device = device))*params['epsilon']
                                else:
                                    # delays
                                    noise = (torch.randn(hidden.size(),device = device))*params['epsilon_delays']
                            else:
                                noise = (torch.randn(hidden.size(),device = device))*params['epsilon']
                            
                    else:
                        # timestep without noise
                        noise = torch.zeros(hidden.size(),device = device)
                        
                        
                    
                    if i == seq_len - 1:
                        # last time step of the sequence
                        o[i,:,:], h_n = self.step(inputs[i, :, :], hidden,noise)
                        # h_n = self.do(h_n)
                    else:
                        o[i,:,:], hidden = self.step(inputs[i, :, :], hidden,noise)
                        #pdb.set_trace()
                        # hidden = self.do(hidden)
                    # if np.logical_and(i>0,i<4):
                    #     print('hidden after update')
                    #     print(hidden)
                # apply dropout
                
                
            else:
                # No noise
                o, h_n = self.Wrec(inputs)
            
            # pass the recurrent activation from the last timestep through the decoder layer
            output = self.out(h_n)
            output = self.softmax(output)
            return output.squeeze(), o, h_n   
        
    model = RNN(params)
    
    return model, RNN

def train_model(params,data,device):
    """
    Train the RNN model and save it, along with the training details.
    
    Parameters
    ----------
    
    params : dictionary 
       
    loss_fn : torch.nn module
        
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
    
    if not(params['from_scratch']):
        # read in the pre-trained model
        model_path = params['FULL_PATH'] + 'saved_models/'
        model = load_model(model_path,params,device)
    
    # transfer model to GPU if available
    model.to(device)
    
    if params['optim'] == 'SGD':
        optimizer = optim.SGD(model.parameters(),lr=params['learning_rate'])#,momentum=.9)
    elif params['optim'] == 'SGDm':
        optimizer = optim.SGD(model.parameters(),lr=params['learning_rate'],momentum=.9)
    elif params['optim'] == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(),lr=params['learning_rate'])
    elif params['optim'] == 'Adam':
        optimizer = optim.Adam(model.parameters(),lr=params['learning_rate'])
    
    
    if params['loss_fn'] == 'MSE':
        loss_fn = torch.nn.MSELoss()
    elif params['loss_fn'] == 'CEL':
        loss_fn = torch.nn.CrossEntropyLoss()
            
    loss_all = torch.zeros(params['n_batches'],
                           params['n_epochs']).to(device)
    
    n_valid_trials = int(params['cue_validity']*params['batch_size'])
    loss_valid = torch.zeros(n_valid_trials,
                             params['n_batches'],
                             params['n_epochs']).to(device)
    loss_epoch = torch.empty((params['n_epochs'],)).to(device)

    shuffling_order = torch.zeros((params['batch_size'],
                                                     params['n_epochs']),
                                                    dtype=torch.long).to(device)
    net_outputs = torch.zeros((params['n_batches'],params['batch_size'],
                                             params['n_colCh'],
                                             params['n_epochs'])).to(device)
    
    # mean and std of hidden activations on the first trial of each batch
    hidden_stats  = torch.zeros((params['n_epochs'],
                                                   params['seq_len'],2)).to(device)
        

    inputs_base = data['inputs']
    inputs_base = inputs_base.to(device)
    
    targets = data['targets']
    targets = targets.to(device)
    
    dLoss = torch.empty((params['n_epochs']-1,))
    loss_clean = torch.empty((params['n_epochs'],))

    window = params['conv_criterion']['trial_window']
    
    if params['condition']!='deterministic':
        invalid_trials = []
    # loop over epochs
    for i in range(params['n_epochs']):   
            # print('Epoch %d : ' %i)         
            # shuffle dataset for SGD
            shuffling_order[:,i] = \
                torch.randperm(params['batch_size'],dtype=torch.long).to(device)
            
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
                invalid_trials.append(ixs)
            else:
                inputs = inputs_base
            
            dset = list(zip(inputs.permute(1,0,-1),targets))
            data_loader = DataLoader(dset,
                                    batch_size = params['batch_size'],
                                    shuffle=True)
            # loop over training examples
            for batch, (data_mb,targets_mb) in enumerate(data_loader):
            # for trial in range(params['batch_size']):
                # print('Trial %d : ' %trial)   
                
                if params['var_delays']:
                    trial_input = data_mb[:,delay_mask[:,batch],:].permute(1,0,-1)                 
                else:
                    trial_input = data_mb.permute(1,0,-1)
               
                outputs, o, hidden = \
                    model(trial_input)
                # print(torch.where(o<0))
                if np.logical_and(batch == 0,i==0):
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
                if params['batch_size']==1:
                    # SGD
                    if params['target_type'] == 'Gaussian':
                        loss = loss_fn(outputs.unsqueeze(0), 
                                        targets_mb)
                    elif params['target_type'] == 'class_label':
                        loss = loss_fn(outputs.unsqueeze(0), targets_mb)
                    elif params['target_type']=='angle_val':
                        # pdb.set_trace()
                        loss,loss_all_examples = custom_MSE_loss(params,outputs,targets_mb)
                else:
                    # minibatch
                    if params['target_type'] == 'Gaussian':
                        loss = loss_fn(outputs, 
                                        targets_mb)
                    elif params['target_type'] == 'class_label':
                        loss = loss_fn(outputs, targets_mb)
                    elif params['target_type']=='angle_val':
                        # pdb.set_trace()
                        loss,loss_all_examples = custom_MSE_loss(params,outputs,targets_mb)
                # pdb.set_trace()
                # Keep track of outputs and loss
                if params['target_type']=='angle_val':
                    loss_all[batch,i] = loss.detach()
                else:
                    loss_all[batch,i] = loss.item()

                net_outputs[batch,:,:,i] = outputs.detach()
                # Compute gradients
                optimizer.zero_grad()
                loss.backward()
                # Update weights
                optimizer.step()
            
                if params['condition']!='deterministic':
                    valid_ix = torch.from_numpy(np.setdiff1d(np.arange(params['batch_size']),invalid_trials[i]))
                    
                else:
                    valid_ix = torch.arange(params['batch_size'])
                
            # if len(valid_ix) != n_valid_trials:
            #     ValueError('loss_valid has the wrong preallocated size!')
                loss_valid[:,batch,i] = loss_all_examples[valid_ix].detach()
            
            # print progress
            # for non-deterministic conditions, only show loss on valid trials   
            # pdb.set_trace()
            loss_epoch[i] = helpers.circ_mean(loss_valid[:,:,i]).abs()
            
            
            if ((i*100/params['n_epochs'])%25)==0:
                print('Model %2d :    %.2f%% iterations of SGD completed...loss = %.2f' \
                          % (int(params['model_number']),
                              100* (i + 1) / params['n_epochs'],
                              loss_epoch[i]))

                              # torch.mean(loss_all[valid_ix,i])))
                # print_progress(i, params['n_epochs'])
            if (i==params['n_epochs']-1):
                print('Model %2d :    100%% iterations of SGD completed...loss = %.2f' \
                          % (int(params['model_number']),
                             loss_epoch[i]))
                              # torch.mean((loss_all[valid_ix,i]))))
                              # torch.mean((loss_all[:,i]))))
                              
            
            # if loss reaches the required level, stop training
            # if torch.mean((loss_all[valid_ix,i])) <= params['MSE_criterion']:
            if loss_epoch[i] <= params['MSE_criterion']:
                
            # select the loss values to be used for the convergence criterion
            # criterion_reached = apply_conv_criterion(loss_valid[:,i-1:i+1],params)
            #criterion_reached = False
            #if criterion_reached:
                print('Model %2d converged after %d epochs. End loss = %.2f' \
                        % (int(params['model_number']), i+1,
                          # torch.mean(loss_all[valid_ix,i])))
                          loss_epoch[i]))
                # prune preallocated tensors
                net_outputs = net_outputs[:,:,:i+1]
                loss_all = loss_all[:,:i+1]
                loss_valid = loss_valid[:,:,:i+1]
                shuffling_order = shuffling_order[:,:i+1]
                dLoss = dLoss[:i+1]
                loss_clean = loss_clean[:i+1]
                loss_epoch = loss_epoch[:i+1]
                break
    
    
    # save training data
    if params['from_scratch']:
        track_training = {}
        track_training['loss'] = loss_all
        track_training['loss_valid'] = loss_valid
        track_training['loss_epoch'] = loss_epoch
        track_training['shuffling_order'] = shuffling_order
        track_training['outputs'] = net_outputs
        # mean and std of hidden activations on the first trial of each batch
        track_training['hidden_stats'] = hidden_stats
        track_training['dLoss'] = dLoss
        track_training['loss_clean'] = loss_clean
        if params['condition']!='deterministic':
            track_training['invalid_trials'] = torch.tensor(invalid_trials).to(device)
    else:
        # load training structures and append to them
        f = open(params['FULL_PATH']+'training_data/'+'training_data_model'+str(params['model_number'])+'.pckl','rb')
        track_training = pickle.load(f)
        f.close()
        print('Add loss_epoch etc here')

        track_training['loss'] = torch.cat((track_training['loss'].to(device),loss_all),-1)
        track_training['loss_valid'] = torch.cat((track_training['loss_valid'].to(device),loss_valid),-1)
        track_training['shuffling_order'] = torch.cat((track_training['shuffling_order'].to(device),shuffling_order),-1)
        track_training['outputs'] = torch.cat((track_training['outputs'].to(device),net_outputs),-1)
        if params['condition']!='deterministic':
            track_training['invalid_trials'] = torch.cat((track_training['invalid_trials'].to(device),torch.tensor(invalid_trials).to(device)),0)
    
    return model, track_training


def partial_training(params,data,trial_sequence,device):
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

    # transfer model to GPU if available
    model.to(device)
    
    if params['optim'] == 'SGD':
        optimizer = optim.SGD(model.parameters(),lr=params['learning_rate'])#,momentum=.9)
    elif params['optim'] == 'SGDm':
        optimizer = optim.SGD(model.parameters(),lr=params['learning_rate'],momentum=.9)
    elif params['optim'] == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(),lr=params['learning_rate'])
    
    
    if params['loss_fn'] == 'MSE':
        loss_fn = torch.nn.MSELoss()
    elif params['loss_fn'] == 'CEL':
        loss_fn = torch.nn.CrossEntropyLoss()
    
    
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
    eval_data : dict
        full evaluation/testing dataset, including all timepoints and expt conditions
        
    pca_data: averaged across uncued stimuli and binned into PARAMS['B'] colour bins
        
    rdm_data: only averaged across uncued stimuli, not binned
        
    model_outputs:
        
    """
    # # load
    # model, RNN = define_model(n_inp,n_rec,n_stim)
    # model = torch.load(load_path+'model'+str(model_number))
    
    #% evaluate model on pca data
    
    model.eval()
    with torch.no_grad():
        if params['noise_type']=='hidden':
            readout, output, hidden = \
                    model(test_data['inputs'])
        else:
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
    
    # bin (i.e. average across n_samples)
    trial_data = torch.mean(trial_data,1) #(M,seq_len,n_rec)
    # d1_ix = params['trial_timings']['stim_dur'] + \
    #     params['trial_timings']['delay1_dur'] - 1
    # d2_ix =  sum(params['trial_timings'].values())-1
    
    d1_ix = params['trial_timepoints']['delay1_end']-1
    d2_ix = params['trial_timepoints']['delay2_end']-1
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
    rdm_data = output.permute(1,0,-1).unsqueeze(0) # time x trial x n_rec
    
    n_samples = params['n_trial_instances'] * params['n_stim'] # n samples for each cued colour
    n_cued_conditions = params['n_stim'] * params['L'] # n of colours x locations
    rdm_data = torch.reshape(rdm_data,(n_cued_conditions,n_samples,params['seq_len'],params['n_rec']))
    
    # rdm_labels1 = torch.reshape(test_data['c1'],(n_cued_conditions,n_samples))
    # rdm_labels2 = torch.reshape(test_data['c2'],(n_cued_conditions,n_samples))
    
    rdm_data = torch.mean(rdm_data,1) # bin uncued
    
    # # average across instances
    
    # rdm_data = torch.reshape(rdm_data.unsqueeze(0),(params['n_stim']*2,params['B'],params['seq_len'],params['n_rec']))
    # rdm_data = torch.reshape(rdm_data.unsqueeze(0),(params['B']*2,params['B'],params['B'],params['seq_len'],params['n_rec']))
    # rdm_data = torch.mean(rdm_data,1) # average across cued
    # rdm_data = torch.reshape(rdm_data,(params['B']*2*params['B'],params['seq_len'],params['n_rec'])) # condition, time, nrec
    
    save_data(rdm_data,params,save_path+'rdm_data_model')
    print('.... Data saved')
    # f = open(dpath+'rdm_data_model'+str(model_number)+'.pckl','wb')
    # pickle.dump(rdm_data,f)
    # f.close()
    
    model_outputs = {'data':readout.squeeze(),'labels':{"loc":test_data['loc'],
                      "c1":test_data['c1'],
                      "c2":test_data['c2']}}
    
    save_data(model_outputs,params,save_path+'model_outputs_model')
    
    print('.... Data saved')
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
    
    if params['noise_period'] == 'all':
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

def var_delay_mask(delay_mat,params,device):
    """
    Generates a mask to be used with the input data to modify delay length.
    """
    if not params['var_delays']:
        return;

    delay_mask = torch.ones((params['seq_len'],params['batch_size']),dtype=bool)
    
    for trial in range(params['batch_size']):
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
        loss values at every timepoint - note the function assumes training
        with SGD, so will sum the values along the first (epoch) dimension.
    
   
    Returns
    -------
    dLoss : np.array
        derivative of the loss wrt to time.
    loss_clean : np.array
        loss values averaged over a batch and after optional smoothing

    '''
    if len(loss_vals.shape)<2:
        ValueError('Loss_vals can''t be a 1-dimensional array')
    loss_mean = loss_vals.mean(0) # average the loss across all trials within an epoch

    # convolve with a Gaussian filter to smooth the loss curve
    loss_clean = gaussian_filter1d(loss_mean,params['conv_criterion']['smooth_sd'])
    loss_clean = torch.tensor(loss_clean)
    
    # calculate the derivative
    dLoss = torch.zeros(loss_clean.shape[0]-1)
    for i in range(loss_clean.shape[0]-1):
            dLoss[i] = loss_clean[i+1] - loss_clean[i]
    return dLoss, loss_clean, loss_mean


def apply_conv_criterion(params,loss_vals):
    '''
    Applies the convergence criteria, returns True if satisfied.

    Parameters
    ----------
    loss_vals : torch.tensor
        DESCRIPTION.
    params : dict
        Dictionary containing the convergence criterion parameters:
         - params['conv_criterion']['cond1'] is the min value for dLoss
         - params['conv_criterion']['cond2'] is the max value for loss_vals

    Returns
    -------
    criterion_reached : bool
        True if both cond1 and cond2 satisfied.

    '''
    

    
    # calculate the dLoss/dt
    dLoss,loss_clean,loss_mean = get_dLoss_dt(params,loss_vals)
    
    # find ixs where dLoss satisfies cond1
    # ix = np.where(np.logical_and(dLoss<0,
    #                              dLoss>params['conv_criterion']['cond1']))[0]
    # # ix -= 1 # index scale for loss_vals
    cond1 = (dLoss[-1]>params['conv_criterion']['cond1'])
    
    # find ixs where loss_vals satisfy cond2
    cond2 = (loss_mean[-1]<params['conv_criterion']['cond2'])
    
    # cond1cond2 = np.intersect1d(ix,ix2)
    # criterion_reached = (len(cond1cond2)!=0)
    criterion_reached = np.logical_and(cond1,cond2)
    
    return criterion_reached


def custom_MSE_loss(params,output,target):
    target = torch.stack([target]*len(params['phi']),dim=1)
    diff = helpers.circ_diff(target,params['phi'])
    if len(diff.shape)==2:
        # batch training
        loss_all_examples = torch.empty((params['batch_size'],))
        for i in range(params['batch_size']):
            loss_all_examples[i] = helpers.circ_mean((diff[i,:]*output[i,:])**2)
        loss_all_examples = loss_all_examples.abs()
        # average across trials in a batch
        loss = helpers.circ_mean(loss_all_examples)

    else:
        #SGD
        loss = helpers.circ_mean((diff*output)**2)
        loss = loss.abs()
        loss_all_examples = loss.clone()
    
    # take the absolute value - only interested in an unsigned distance, 
    #and circular mean will be in [-pi,pi]
    # loss_all_examples = loss.abs()
    # pdb.set_trace()
    
    # if len(loss_all_examples)>1:
    #     # average across trials in a batch
    #     loss = helpers.circ_mean(loss)
    #     loss = loss.abs()
    # else:
    #     loss = loss_all_examples.clone()
    
    return loss,loss_all_examples
           
    
    
