#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 12:47:56 2020

@author: emilia
"""

import numpy as np
import torch
from torch import nn
from torch import optim
import pickle
import matplotlib.pyplot as plt


#%%  Set up model and task



#torch.manual_seed(0)

model_number = input('Specify model number: ')
# task parameters

# structured inputs
n_stim = 4#32 # n colour stimuli
n_colCh = 4#32 # n color neurons/location


# model params and hyperparams
n_inp = n_stim*2 + 3 #stimuli + cues + fixation


# 1-hot inputs
n_stim = 4#32 # n colour stimuli


n_rec = 10 # n hidden units

# Set up SGD parameters
n_iter = 4000 # iterations of SGD
batch_size = 50
learning_rate = .01#1e-5

# Squared error loss function
#loss_fn = nn.MSELoss()

# negLL
#loss_fn = nn.NLLLoss()

# cross-entropy
loss_fn = nn.CrossEntropyLoss()

#%% define RNN class

class RNN(nn.Module):
        def __init__(self, n_inp, n_rec, n_out, ih0, hh0,ho0):
            super(RNN, self).__init__()
            # input
            self.n_rec = n_rec # number of recurrent neurons
            self.n_inp = n_inp
            self.n_out = n_out
    
            self.Wrec = nn.RNN(self.n_inp,self.n_rec, nonlinearity = 'relu')
            
            # load the initialisation
            self.Wrec.weight_ih_l0 = nn.Parameter(ih0)
            self.Wrec.weight_hh_l0 = nn.Parameter(hh0)
            
            
            
            self.Wrec.weight_ih_l0 = nn.Parameter(torch.randn(self.n_rec,self.n_inp)*((np.sqrt(2)/np.sqrt(self.n_inp))/2))
            self.Wrec.weight_hh_l0 = nn.Parameter(torch.randn(self.n_rec,self.n_rec)*((np.sqrt(2)/np.sqrt(self.n_rec)))/2)
            
            self.Wrec.bias_ih_l0 = nn.Parameter(torch.zeros(self.n_rec))
            self.Wrec.bias_hh_l0 = nn.Parameter(torch.zeros(self.n_rec))
            
            # input and hidden
            self.out = nn.Linear(self.n_rec, self.n_out) # output layer
            
            self.out.weight = nn.Parameter(ho0)
            self.out.bias = nn.Parameter(torch.zeros(self.n_out))
            
            #ignore (scratch code):
            #self.inp = nn.Linear(17, self.n_neurons) # input weights
            #self.Wrec = nn.Parameter(torch.randn(self.n_neurons, self.n_neurons) / np.sqrt(self.n_neurons))  # recurrent weights    
             #self.Wrec.bias_ih_l0 = nn.Parameter(torch.ones(n_rec))
            #self.Wrec.bias_hh_l0 = nn.Parameter(torch.ones(n_rec))
            #self.Wrec.weight_hh_l0 = nn.Parameter(self.Wrec.weight_hh_l0*1e-2)
            #self.Wrec.weight_ih_l0 = nn.Parameter(self.Wrec.weight_ih_l0*1e-2)
            
        # def step(self, input_ext, hidden):
        #     o, hidden = self.Wrec(input_ext.unsqueeze(0),hidden)
        #     output = self.out(hidden)
        #     return output, hidden
        #     # ignore:
        #     #hidden = torch.relu(torch.matmul(self.Wrec, hidden.unsqueeze(-1)).squeeze() + self.inp(input_ext))
            
        def forward(self, inputs):
            """
            Run the RNN with input timecourses
            """
            # Initialize network state
            #hidden = torch.zeros((1,inputs.size(1), self.n_rec)) # 0s
            
            o, h_n = self.Wrec(inputs)
            # Run the input through the network - across time
            # for i in range(inputs.size(2)):
            #     output, hidden = self.step(inputs[:, :, i].T, hidden)
            # return output.squeeze(), hidden
            
            output = self.out(h_n)
            return output.squeeze(), h_n   
# #%% choose an initialisation and save it to file
# ih0 = torch.randn(n_rec,n_inp)*(np.sqrt(2)/np.sqrt(n_inp))
# hh0 = torch.randn(n_rec,n_rec)*(np.sqrt(2)/np.sqrt(n_rec))
# ho0 = torch.randn(n_stim,n_rec)

# init_weights = [ih0, hh0, ho0]
# path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/data/saved_models_sameTrainingInit'
# f = open(path+'/init_weights.pckl','wb')
# pickle.dump(init_weights,f)
# f.close()

#%% load initialisation
path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/data/saved_models_sameTrainingInit'
f = open(path+'/init_weights.pckl','rb')
obj = pickle.load(f)
f.close()

[ih0, hh0, ho0] = obj

plt.figure()
plt.subplot(131)
plt.imshow(hh0)
plt.colorbar()
plt.title('ih0')

plt.subplot(132)
plt.imshow(ih0)
plt.colorbar()
plt.title('hh0')

plt.subplot(133)
plt.imshow(ho0)
plt.colorbar()
plt.title('ho0')


#%% load training data
path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/data/training_data_sameTrainingInit'

f = open(path+'/common_training_dataset.pckl','rb')
obj = pickle.load(f)
[inputs,targets] = obj
f.close()
#%% initialise model


model = RNN(n_inp,n_rec,n_stim,ih0,hh0,ho0)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

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
    +'shared_repo/retrocue_rnn/emilia/data/loss_plots_sameTrainingInit/'+'Model'+ model_number
plt.savefig(fig_path)

#%% save model

path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/data/'

#path = 'saved_models/'
torch.save(model,path+'saved_models_sameTrainingInit/model' + model_number)

torch.save(model.state_dict(),path+'saved_models_sameTrainingInit/model'+model_number+'_statedict')


# save training data and loss
training_data = [track_loss, track_targets, track_outputs]


print('Model saved')


#%% plot final weights

plt.figure()
plt.subplot(131)
plt.imshow(model.Wrec.weight_ih_l0.detach())
plt.colorbar()
plt.title('ih')

plt.subplot(132)
plt.imshow(model.Wrec.weight_hh_l0.detach())
plt.colorbar()
plt.title('hh')

plt.subplot(133)    
plt.imshow(model.out.weight.detach())
plt.colorbar()
plt.title('ho')

