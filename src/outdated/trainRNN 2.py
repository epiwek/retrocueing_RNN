#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 18:31:08 2020

@author: emilia
"""

import numpy as np
import torch
from torch import nn
from torch import optim
from make_stimuli import make_stimuli
import matplotlib.pyplot as plt
from define_model import define_model

#torch.manual_seed(1)

#%%  Set up model and task
for m in np.arange(2,10):
    
    print('...... MODEL %d .......' %m)
    #model_number = input('Specify model number: ')
    model_number = str(m)
    torch.manual_seed(m)
    # task parameters
    
    n_stim = 4#32 # n colour stimuli
    n_colCh = 4#32 # n color neurons/location
    tuning_params = [0.8, 2.0] #height and width of tuning curve
    trial_timings = {}
    trial_timings['stim_dur']=1
    trial_timings['delay1_dur'] = 1
    trial_timings['cue_dur']= 1
    trial_timings['delay2_dur'] = 1
    trial_timings['ITI_dur'] = 0
    trial_timings['resp_dur'] = 0
    
    # model params and hyperparams
    n_inp = n_stim*2 + 3 #stimuli + cues + fixation
    n_rec = 10 # n hidden units
    model, RNN = define_model(n_inp,n_rec,n_stim)
    
    # Set up SGD parameters
    n_iter = 12000 # iterations of SGD
    batch_size = 50
    learning_rate = .04#1e-5
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # Squared error loss function
    #loss_fn = nn.MSELoss()
    
    # negLL
    #loss_fn = nn.NLLLoss()
    
    # cross-entropy
    loss_fn = nn.CrossEntropyLoss()
    
    #%% show init weights
    
    plt.figure()
    
    plt.subplot(131)
    plt.imshow(model.Wrec.weight_ih_l0.detach())
    plt.colorbar()
    plt.title('ih0')
    
    plt.subplot(132)
    plt.imshow(model.Wrec.weight_hh_l0.detach())
    plt.colorbar()
    plt.title('hh0')
    
    plt.subplot(133)
    plt.imshow(model.out.weight.detach())
    plt.colorbar()
    plt.title('ho0')
    
    #%% load training data
    import pickle
    path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/data/training_data_sameTrainingData'
    
    f = open(path+'/common_training_dataset.pckl','rb')
    obj = pickle.load(f)
    [inputs,targets] = obj
    f.close()
    #%% train
    
    # Placeholders
    track_loss = torch.zeros(n_iter)
    track_targets = torch.zeros((batch_size,n_iter))
    track_outputs = torch.zeros((batch_size,n_stim,n_iter))
    
    # Loop over iterations
    for i in range(n_iter):
        if (i + 1) % 50 == 0: # print progress every 100 iterations
            print('%.2f%% iterations of SGD completed...loss = %.2f' % (100* (i + 1) / n_iter, loss))
        # Sample stimulus
        #I, loc, c1, c2, T = make_stimuli(n_stim,n_colCh,batch_size,trial_timings,tuning_params)
        I = inputs[str(i)]
        T = targets[str(i)]
        
        # Run model
        outputs, hidden = model(I)
        # Compute loss
        loss = loss_fn(outputs, T)#.type(torch.long))
        # Keep track of outputs and loss
        track_loss[i] = loss.item()
        track_targets[:,i] = T.detach()
        track_outputs[:,:,i] = outputs.detach()
        # Compute gradients
        optimizer.zero_grad()
        loss.backward()
        # Update weights
        optimizer.step()
    
    #%% Plot the loss
    fig, ax = plt.subplots()
    ax.plot(track_loss)
    ax.set_xlabel('iterations of stochastic gradient descent')
    ax.set_ylabel('loss')
    #ax.set_ylim(0,1)
    
    ax.set_title('Model '+model_number)
    
    fig_path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/'\
        +'shared_repo/retrocue_rnn/emilia/data/loss_plot_sameTrainingData/'+'Model'+ model_number
    plt.savefig(fig_path)
    
    #%% save model
    
    path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/data/'
    
    #path = 'saved_models/'
    torch.save(model,path+'saved_models_sameTrainingData/model' + model_number)
    
    torch.save(model.state_dict(),path+'saved_models_sameTrainingData/model'+model_number+'_statedict')
    
    
    # save training data and loss
    training_data = [track_loss, track_targets, track_outputs]
    
    import pickle
    f = open(path+'training_data_sameTrainingData/model'+model_number+'_td.pckl', 'wb')
    pickle.dump('training_data',f)
    f.close()
    
    print('Model saved')

#%% task and model params
# path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/saved_models/'


# task_params = [n_stim,n_colCh,tuning_params,trial_timings]
# model_params = [n_inp,n_rec,n_iter,batch_size,learning_rate]


# import pickle
# f = open(path+'task_params.pckl', 'wb')
# pickle.dump(task_params,f)
# f.close()

# f = open(path+'model_params.pckl', 'wb')
# pickle.dump(model_params,f)
# f.close()



#%%

# plt.figure()

# plt.subplot(121)
# plt.imshow(model.Wrec.weight_ih_l0.detach())
# plt.colorbar()
# plt.title('ih')

# plt.subplot(122)
# plt.imshow(model.Wrec.weight_hh_l0.detach())
# plt.colorbar()
# plt.title('hh')

# #%%

# plt.figure()

# plt.subplot(121)
# plt.hist(model.Wrec.weight_ih_l0.detach().reshape(model.Wrec.weight_ih_l0.shape[0]*model.Wrec.weight_ih_l0.shape[1]))
# plt.title('ih')
# plt.xlim([-1,1])

# plt.subplot(122)
# plt.hist(model.Wrec.weight_hh_l0.detach().reshape(model.Wrec.weight_hh_l0.shape[0]**2))
# plt.title('hh')
# plt.xlim([-1,1])
