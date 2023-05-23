#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 10:52:14 2020

@author: emilia
"""

import numpy as np
import torch
from torch import nn
from utils import generate_data
import pickle


XaviersComputer = False

# if not(XaviersComputer):
#     cd '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/'
    
model_number = 3

if XaviersComputer:
    BASE_PATH = '/Users/Xavier/Desktop/Work/Summerfield-Lab/retrocue_rnn/'
        
else:
    BASE_PATH = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/' + \
         'shared_repo/retrocue_rnn/'
         
LSTM_PATH = BASE_PATH + 'emilia/lstm_model_weights/Model'+ str(model_number) + \
         '/Model' + str(model_number) + '_LSTM_weights.npy'
DENSE_PATH = BASE_PATH + 'emilia/lstm_model_weights/Model'+ str(model_number) + \
         '/Model' + str(model_number) + '_Dense_weights.npy'
   

# load weights and data
x, y, lab = generate_data(labels=True)

x_data = torch.from_numpy(x)
y_data = torch.from_numpy(y)

x_data = x_data.type(torch.float32)
y_data = y_data.type(torch.float32)

lstm_weights = np.load(LSTM_PATH ,allow_pickle=True)
dense_weights = np.load(DENSE_PATH,allow_pickle=True)

x_data = x_data.permute(1,0,-1)

odds = x_data[:,0:32:2,:]
evens_reds = x_data[:,1:32:8,:]
evens_blues = x_data[:,3:32:8,:]
evens_greens = x_data[:,5:32:8,:]
evens_yellows = x_data[:,7:32:8,:]
x_data = torch.cat((odds, evens_reds, evens_blues, evens_greens, evens_yellows), 1)

#%% define model
torch.manual_seed(0)

class LSTM(nn.Module):
    def __init__(self, n_inp, n_rec, n_out):
        super(LSTM, self).__init__()
        # input
        self.n_rec = n_rec # number of recurrent neurons
        self.n_inp = n_inp
        self.n_out = n_out
        self.lstm = nn.LSTM(n_inp,n_rec)
        
        #self.out = nn.Linear(self.n_rec, self.n_out) # output layer
        
    #def step(self, input_ext, hidden, cell):
    #    o, (hidden, cell) = self.lstm(input_ext.unsqueeze(0),(hidden, cell))
    #    output = self.out(o)
    #    return output, hidden, cell
        
    def forward(self, inputs):
        """
        Run the RNN with input timecourses
        """
        # Initialize network state
        #h_0 = torch.zeros((1,inputs.size(1), self.n_rec)) # n layers x batch x n_rec
        #c_0 = torch.zeros((1,inputs.size(1), self.n_rec))
        # Run the input through the network - across time
        #for i in range(inputs.size(1)):
        #    output, (hidden, cell) = self.step(inputs[:, i, :], hidden, cell)
        #return output.squeeze(), hidden, cell
        
        o, (hidden, cell) = self.lstm(inputs) #,(h_0, c_0))
        #print(o.shape)
        #output = self.out(o[-1,:,:].squeeze())
        
        return o, (hidden, cell)
    
#%% set weights and biases
torch.manual_seed(0)

batch_size = x_data.shape[1]
seq_len = x_data.shape[0]
n_inp = x_data.shape[2]
n_rec = lstm_weights[0].shape[1]//4
n_out = y_data.shape[1]

Ws = lstm_weights[0] # TF order : i, j, f, o
Us = lstm_weights[1]
bs = lstm_weights[2]

# for torch, need them to be in the following order: i, f, j, o
Ws[:,4:12] = Ws[:,np.concatenate((np.arange(8,12),np.arange(4,8)))]
Us[:,4:12] = Us[:,np.concatenate((np.arange(8,12),np.arange(4,8)))]
bs[4:12] = bs[np.concatenate((np.arange(8,12),np.arange(4,8)))]


model = LSTM(n_inp,n_rec,n_out)
model.lstm.weight_ih_l0 = nn.Parameter(torch.from_numpy(Ws.T),requires_grad = False)
model.lstm.weight_hh_l0 = nn.Parameter(torch.from_numpy(Us.T),requires_grad = False)
model.lstm.bias_ih = nn.Parameter(torch.from_numpy(bs),requires_grad = False)
model.lstm.bias_hh = nn.Parameter(torch.zeros((4*n_rec,)),requires_grad = False)

#model.out.weight = nn.Parameter(torch.from_numpy(dense_weights[0]).double())
#model.out.bias = nn.Parameter(torch.from_numpy(dense_weights[1]).double())

#%% run model on data
model.eval()

with torch.no_grad():
    output, (h_n, c_n) = model(x_data)

# visualise hidden layer
# plt.figure()
# for i in range(output.shape[0]):
#     plt.subplot(3,6,i+1)
#     plt.imshow(output[i,:,:].detach().numpy())
#     plt.colorbar()

#%% reshape data for PCA
trial_data = output
# time x batch x unit

# average across all unattended stimuli
trial_reds_up = trial_data[:,0:4,:].mean(axis=1)
trial_blues_up = trial_data[:,4:8,:].mean(axis=1)
trial_greens_up = trial_data[:,8:12,:].mean(axis=1)
trial_yellows_up = trial_data[:,12:16,:].mean(axis=1)

trial_reds_down = trial_data[:,16:20,:].mean(axis=1)
trial_blues_down = trial_data[:,20:24,:].mean(axis=1)
trial_greens_down = trial_data[:,24:28,:].mean(axis=1)
trial_yellows_down = trial_data[:,28:32,:].mean(axis=1)


trial_data = torch.stack((trial_reds_up, trial_blues_up, trial_greens_up, trial_yellows_up,
                        trial_reds_down, trial_blues_down, trial_greens_down, trial_yellows_down), axis=1)


# extract pre- and post-cue activity
delay1 = trial_data[9].detach()
delay2 = trial_data[17].detach()

#%% save data structs

PCA_PATH = BASE_PATH + 'emilia/LSTM_data/pca_data/'
pca_data = [trial_data,delay1,delay2]
f = open(PCA_PATH+'pca_data_model'+str(model_number)+'.pckl','wb')
pickle.dump(pca_data,f)
f.close()

#%% plot weights
from matplotlib import pyplot as plt

plt.figure()
plt.imshow(Ws)
plt.colorbar()
plt.title('Model ' + str(model_number) +': Ws')


plt.figure()
plt.imshow(Us)
plt.colorbar()
plt.title('Model ' + str(model_number) +': Us')


plt.figure()
plt.imshow(torch.from_numpy(bs).unsqueeze(0))
plt.colorbar()
plt.title('Model ' + str(model_number) +': bs')

