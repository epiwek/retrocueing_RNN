#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 15:52:01 2020

@author: emilia
"""

import numpy as np
import torch
from torch import nn
from torch import optim
import pickle
import matplotlib.pyplot as plt
from define_model import define_model
from generate_data_1hot import make_stimuli_1hot
# from generate_data_gaussian import make_stimuli_gaussian
from generate_data_vonMisses_cp import make_stimuli_vonMises

import time
start_time = time.time()

import os.path

#%% define model class - RNN or LSTM

model_class = 'RNN'

#%% load training data
# if model_class == 'RNN':
#     path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/data_1hot/training_data'
# else:
#     path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/LSTM_data_1hot/training_data'

# f = open(path+'/common_training_dataset.pckl','rb')
# obj = pickle.load(f)
# [inputs,targets] = obj
# f.close()

trial_timings = {}
trial_timings['stim_dur']=1
trial_timings['delay1_dur'] = 5
trial_timings['cue_dur']= 1
trial_timings['delay2_dur'] = 5
trial_timings['ITI_dur'] = 0
trial_timings['resp_dur'] = 0


data_type = 'gaussian'
# data_type = '1-hot'

# n_rec = 100 # n hidden units

if data_type == 'gaussian':
    # Gaussian stimuli
    fixation = True
    n_stim = 16#32 # n colour stimuli
    n_colCh = 17
    
else:
    # 1-hot stimuli
    n_stim = 4#32 # n colour stimuli
    n_inp = n_stim*2+2 # stimulus and context units
    n_colCh = n_stim

#%% model and task params
#lrs = np.array([1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3])
# lrs = np.arange(2,11)
lrs = np.array([4])
lr = 3
hidden_sizes = np.array([200])
n_models = 10

if data_type == 'gaussian':
    # generate stimuli - Gaussian
    batch_size = n_stim*n_stim*2
    kappa_val = 1.0
    I, loc, c1, c2, T = make_stimuli_vonMises(n_stim, n_colCh, batch_size, trial_timings,add_fixation=fixation,kappa = kappa_val)
    
    n_inp = I.shape[-1]
    
    # add an option to normalise inputs
    # normalise = input('Normalise inputs? [Y/N] : ')
    normalise = 'N'
    # if normalise == 'Y':
    #     # mean-center
    #     # I -= torch.mean(I)
        
    #     # I = (I-torch.mean(I,-1).repeat(n_inp,1,1).permute(1,2,0))
    #     # or
    #     # z-score input at each timestep
    #     I = (I-torch.mean(I,-1).repeat(n_inp,1,1).permute(1,2,0))/torch.std(I,-1).repeat(n_inp,1,1).permute(1,2,0) 
        
    # elif normalise != 'N':
    #     raise ValueError('Only [Y/N] answers accepted.')
else:
    # generate stimuli - single batch (for full GD)
    batch_size = 32
    I, T = make_stimuli_1hot(batch_size, n_stim, trial_timings, constraints = 'on',fixation = 'off')
    n_inp = I.shape[-1]


#%%
for i in range(len(hidden_sizes)):
# #model_number = input('Specify model number: ')
    # print(i,10**(-int(lr)))
    n_rec = hidden_sizes[i]
    print('nrec = %d' %n_rec)
    for model_number in np.arange(10):
        
        torch.manual_seed(model_number)
        model_number = str(model_number)
        print('Model '+model_number)
        
        # # 1-hot stimuli
        # n_stim = 4#32 # n colour stimuli
        # n_inp = n_stim*2+2 # stimulus and context units
        # n_rec = 20 # n hidden units
        
        # structured stimuli
        # to be implemented
        
        # Set up SGD parameters
        n_iter = 3750 # iterations of SGD
        
        learning_rate = 10**(-int(lr))
        #learning_rate = 0.05
        
        print(learning_rate)
        
        if data_type == 'gaussian':
             # MSE
              loss_fn = nn.MSELoss()
             # loss_fn = nn.CrossEntropyLoss()
        else:
            # cross-entropy
            # loss_fn = nn.CrossEntropyLoss()
            loss_fn = nn.MSELoss()
        
       
        #% initialise model
        
        model, net_type = define_model(n_inp, n_rec,n_colCh, model_class)

        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    
        #% train
        # Placeholders
        track_loss = torch.zeros(batch_size,n_iter)
        track_targets = torch.zeros((batch_size,n_iter))
        track_outputs = torch.zeros((batch_size,n_colCh,n_iter))
        track_shuffling = torch.zeros((batch_size,n_iter),dtype=torch.long)
        
        cache_stats = torch.zeros((n_iter,I.shape[0],2))
        
        # Loop over iterations
        for i in range(n_iter):
            # if (i + 1) % 500 == 0: # print progress every 100 iterations
            # print('%.2f%% iterations of SGD completed...loss = %.2f' % (100* (i + 1) / n_iter, loss))
            
            
            # print('Epoch %d' %i)
            
            # Sample stimulus
            #I, loc, c1, c2, T = make_stimuli(n_stim,n_colCh,batch_size,trial_timings,tuning_params)
            
            
            # Run model
            
            # outputs, hidden = model(I)
            # # Compute loss
            # loss = loss_fn(outputs, T)#.type(torch.long))
            # # Keep track of outputs and loss
            # track_loss[i] = loss.item()
            # track_targets[:,i] = T.detach()
            # track_outputs[:,:,i] = outputs.detach()
            # # Compute gradients
            # optimizer.zero_grad()
            # loss.backward()
            # # Update weights
            # optimizer.step()
            
            # SGD on every trial - shuffle dataset
            track_shuffling[:,i] = torch.randperm(batch_size,dtype=torch.long)
            for trial in range(batch_size):
                
                outputs, o, hidden = model(I[:,track_shuffling[trial,i],:].unsqueeze(1))
                
                if trial == 0:
                    cache_stats[i,:,0] = torch.std_mean(o,-1)[0].squeeze()
                    cache_stats[i,:,1] = torch.std_mean(o,-1)[1].squeeze()
                # Compute loss
                # loss = loss_fn(outputs.unsqueeze(0), T[shuff_ix[trial]].unsqueeze(0))#.type(torch.long))
                loss = loss_fn(outputs.unsqueeze(0), T[track_shuffling[trial,i],:].unsqueeze(0))
                # Keep track of outputs and loss
                track_loss[trial,i] = loss.item()
                # track_targets[trial,i] = T[shuff_ix[trial]].detach()
                track_outputs[trial,:,i] = outputs.detach()
                # Compute gradients
                optimizer.zero_grad()
                loss.backward()
                # Update weights
                optimizer.step()
                # print('        %.2f%% iterations of SGD completed...loss = %.2f' % (100* (trial + 1) / batch_size, loss.item()))
            if (i%100 == 0):
                print('Model %2d :    %.2f%% iterations of SGD completed...loss = %.2f' % (int(model_number),100* (i + 1) / n_iter, torch.sum(track_loss[:,i])))
            if (i==n_iter-1):
                print('Model %2d :    100%% iterations of SGD completed...loss = %.2f' % (int(model_number), torch.sum(track_loss[:,i])))
            
            
        #% Plot the loss
        # fig, ax = plt.subplots()
        # ax.plot(torch.sum(track_loss,0))
        # ax.set_xlabel('iterations of stochastic gradient descent')
        # ax.set_ylabel('loss')
        # #ax.set_ylim(0,1)
        
        # ax.set_title('Learning_rate = ' +str(learning_rate))
        # ax.set_title('Model ' +str(model_number))

        # # fig_path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/'\
        # #     +'shared_repo/retrocue_rnn/emilia/data_1hot/loss_plots/'+'Model'+ model_number
        # # plt.savefig(fig_path)
    
        #% save model
        
        if model_class == 'RNN':
            if data_type == 'gaussian':
                path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/'+\
                            'data_vonMises/MSELoss/with_fixation_longTrials/'
                # raise NotImplementedError()
            else:
                # path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/data_1hot_new'
                raise NotImplementedError()
        else:
            # path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/LSTM_data_1hot/'
              raise NotImplementedError()
        
        
        if data_type == 'gaussian':
            # check if path exists - if not, create one
            if not (os.path.exists(path+'kappa'+str(kappa_val))):
                os.mkdir(path+'kappa'+str(kappa_val))
                path = path + 'kappa'+str(kappa_val)+'/'
                os.mkdir(path+'nrec'+str(n_rec))
                os.mkdir(path+'nrec'+str(n_rec)+'/lr'+str(learning_rate))
            else:
                path = path + 'kappa'+str(kappa_val)+'/'
        
        if not (os.path.exists(path+'nrec'+str(n_rec))):
            os.mkdir(path+'nrec'+str(n_rec))
            os.mkdir(path+'nrec'+str(n_rec)+'/lr'+str(learning_rate))
        else:
            if not (os.path.exists(path+'nrec'+str(n_rec)+'/lr'+str(learning_rate))):
                os.mkdir(path+'nrec'+str(n_rec)+'/lr'+str(learning_rate))
        
        path = path+'nrec'+str(n_rec)+'/lr'+str(learning_rate)+'/'   
        
        #path = 'saved_models/'
        if not (os.path.exists(path+'saved_models')):
                os.mkdir(path+'saved_models') 
        torch.save(model,path+'saved_models/model' + model_number)
        torch.save(model.state_dict(),path+'saved_models/model'+model_number+'_statedict')
        
        
        # save training data and loss
        if not (os.path.exists(path+'training_data')):
                os.mkdir(path+'training_data') 
        training_data = [track_loss, track_targets, track_outputs, track_shuffling]
        
        
        f = open(path+'training_data/training_data_model'+model_number+'.pckl','wb')
        pickle.dump(training_data,f)
        f.close()
        
        print('Model saved')
    
            
    plt.figure()
    converged = []
    for model_number in range(10):
        model_number = str(model_number)
        f = open(path+'training_data/'+'training_data_model'+model_number+'.pckl','rb')
        obj = pickle.load(f)
        [track_loss, track_targets, track_outputs, track_shuffling] = obj
        #print('Model %d, end loss = %.2f' %(int(model_number),torch.sum(track_loss[:,-1])))
        if torch.sum(track_loss[:,-1])<=0.05:
            converged.append(int(model_number))
        plt.plot(torch.sum(track_loss,0), label='model '+model_number)
    
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    # #ax.set_ylim(0,1)
    
    plt.title('Learning_rate = ' +str(learning_rate))
            
    
    
    if model_class == 'RNN':
        if data_type == 'gaussian':
            if not (os.path.exists(path+'loss_plots')):
                os.mkdir(path+'loss_plots') 
            fig_path = path + 'loss_plots/'+'1e-0'+str(lr)
        else:
            # fig_path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/'\
            #     +'shared_repo/retrocue_rnn/emilia/data_gaussian_hardcoded/loss_plots/'+'1e-0'+str(lr)
            if not (os.path.exists(path+'loss_plots')):
                os.mkdir(path+'loss_plots') 
            fig_path = path + 'loss_plots/'+'1e-0'+str(lr)
    else:
        fig_path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/'\
            +'shared_repo/retrocue_rnn/emilia/LSTM_data_1hot/loss_plots/'+'1e-0'+str(lr)
    plt.savefig(fig_path)
        
    #% task and model params
    #path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/data_1hot/saved_models/'
    
    if data_type == 'gaussian':
        model_params = [n_inp,n_rec,n_iter,batch_size,learning_rate,fixation]
    else:
        model_params = [n_inp,n_rec,n_iter,batch_size,learning_rate]
    
    
    
    import pickle
    
    f = open(path+'saved_models/model_params.pckl', 'wb')
    pickle.dump(model_params,f)
    f.close()
    
    
    task_params = [n_stim,trial_timings]
    f = open(path+'saved_models/task_params.pckl', 'wb')
    pickle.dump(task_params,f)
    f.close()
    
    print('Task params saved')
    
    #% save indices of models that converged successful
    if not (os.path.exists(path+'pca_data')):
        os.mkdir(path+'pca_data') 
    
    f = open(path+'pca_data/converged.pckl','wb')
    pickle.dump(converged,f)
    f.close()

#%% 

end_time = time.time()
print('Time elapsed: %.2f minutes' %((end_time - start_time)/60))

        