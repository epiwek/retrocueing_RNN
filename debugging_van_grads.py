#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 12:16:51 2021

@author: emilia
"""


import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

from joblib import Parallel, delayed

import matplotlib.pyplot as plt

from helpers import check_path, transfer_to_cpu
import retrocue_model as retnet

import constants
import time

import custom_plot as cplot
import rep_geom_analysis as rga

import pickle
# import pdb
#%% set up flags, paths and device


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

save_path = constants.PARAMS['COND_PATH']
check_path(save_path)

parallelise_flag = True
train_flag  = True
eval_flag   = False
analysis_flag = False

torch.manual_seed(0)
    
#%%

def define_model2(params,model1):
    class RNN2(nn.Module):
            def __init__(self, params):
                super(RNN2, self).__init__()
                # input
                self.n_rec = params['n_rec'] # number of recurrent neurons
                self.n_inp = params['n_inp']
                self.n_out = params['n_out']
                
                if params['noise_type']=='hidden':
                # _____________________________________________________ # 
                # implementation for simulations with hidden noise:
                # (need to add noise after the Wrec@ht-1 step, but before the nonlinearity)
                    self.inp = nn.Linear(self.n_inp,self.n_rec)
                    # self.inp.weight = nn.Parameter(self.inp.weight*params['init_scale']) # Xavier init
                    self.inp.weight = nn.Parameter(model1.inp.weight.clone().detach())
                    # self.inp.weight = nn.Parameter(torch.ones(self.inp.weight.shape))

                    # self.inp.weight = nn.Parameter(torch.randn((self.n_rec,self.n_inp)) *\
                    #                                 params['init_scale'] / np.sqrt(self.n_inp)) # Gaussian init
                    
                    # self.inp.bias = nn.Parameter(self.inp.bias*params['init_scale']) # Xavier init
                    self.inp.bias = nn.Parameter(model1.inp.bias.clone().detach())
                    # self.inp.bias = nn.Parameter(torch.randn((self.n_rec,)) *\
                    #                                 params['init_scale'] / np.sqrt(self.n_inp)) # Gaussian init
    
                    # self.Wrec = (torch.rand(self.n_rec, self.n_rec)*2-1)\
                    #                           * (params['init_scale']/ np.sqrt(self.n_rec))  # recurrent weights - Xavier init
                    self.Wrec = model1.Wrec.clone().detach()
                    # self.Wrec = (torch.eye(self.n_rec))
                    self.Wrec_last = nn.Parameter(self.Wrec.clone())
                    # self.Wrec = nn.Parameter((torch.eye(self.n_rec)))
                    # self.Wrec = nn.Parameter(torch.randn(self.n_rec, self.n_rec)\
                    #                           * (params['init_scale']/ np.sqrt(self.n_rec)))  # recurrent weights - Gaussian init
                    
                    
                    # Wrec = I + Σoff
                    #self.Wrec = torch.diag(torch.ones(self.n_rec)) + \
                    #    torch.abs(1-torch.diag(torch.ones(self.n_rec)))*\
                    #        (torch.randn(self.n_rec, self.n_rec) / np.sqrt(self.n_rec))
                    #self.Wrec = nn.Parameter(self.Wrec)
                    self.relu = nn.ReLU()
                    self.softmax = nn.Softmax(dim=1)
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
                # self.out.weight = nn.Parameter(self.out.weight*params['init_scale']) #Xavier init
                self.out.weight = nn.Parameter(model1.out.weight.clone().detach()) #Xavier init
                # self.out.weight = nn.Parameter(torch.ones(self.out.weight.shape))

                # self.out.weight = nn.Parameter(torch.randn((self.n_out,self.n_rec))\
                #                                *(params['init_scale']/ np.sqrt(self.n_rec))) #Gaussian init
                
                # self.out.bias = nn.Parameter(self.out.bias*params['init_scale']) #Xavier init
                self.out.bias = nn.Parameter(model1.out.bias.clone().detach())
                # self.out.bias = nn.Parameter(self.out.bias*0) # set bias to 0
                #self.out.bias = nn.Parameter(torch.randn((self.n_out,))\
                #                               *(params['init_scale']/ np.sqrt(self.n_rec))) #Gaussian init
               
            def step(self, input_ext,hidden,noise):
                
                hidden = self.relu(self.inp(input_ext.unsqueeze(0)) + hidden @ self.Wrec.T \
                    + noise)
                output = hidden.clone().detach()
                
                return output, hidden
            def step_last(self,input_ext,hidden,noise):
                hidden = self.relu(self.inp(input_ext.unsqueeze(0)) + hidden @ self.Wrec_last.T \
                    + noise)
                output = hidden.clone().detach()
                
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
                            o[i,:,:], h_n = self.step_last(inputs[i, :, :], hidden,noise)
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
                return output.squeeze(), o, h_n   
    model = RNN2(params)
    return model

#%%

from torch import optim
m = 0
constants.PARAMS['model_number'] = m

# trial = constants.TRAINING_DATA['inputs'][:,0,:].unsqueeze(1)
# target = constants.TRAINING_DATA['targets'][0].unsqueeze(0)
loss_fn = torch.nn.CrossEntropyLoss()



model1, RNN = retnet.define_model(constants.PARAMS,device)
model2 = define_model2(constants.PARAMS,model1)

optim1 = optim.RMSprop(model1.parameters(),lr=constants.PARAMS['learning_rate'])
optim2 = optim.RMSprop(model2.parameters(),lr=constants.PARAMS['learning_rate'])


# double check that the inits are the same
print('Wrec: ' + str((model1.Wrec == model2.Wrec).float().mean()))
print('Win: ' + str((model1.inp.weight == model2.inp.weight).float().mean()))
print('b_in: ' + str((model1.inp.bias == model2.inp.bias).float().mean()))
print('Wout: ' + str((model1.out.weight == model2.out.weight).float().mean()))
print('b_out: ' + str((model1.out.bias == model2.out.bias).float().mean()))



#%%

# get predictions
# output1, o1, hidden1 = model1(trial)
# output2, o2, hidden2 = model2(trial)


# (output1 == output2).float().mean()
#%%

# # get loss
# loss1 = loss_fn(output1.unsqueeze(0),target)
# loss2 = loss_fn(output2.unsqueeze(0),target)



# # zero gradients
# optim1.zero_grad()
# optim2.zero_grad()

# # backprop
# loss1.backward()
# loss2.backward()

# # step
# optim1.step()
# optim2.step()

# model2.Wrec = model2.Wrec_last.clone().detach()

n_epochs = 200

#%%classic SGD
loss1_all = torch.empty((n_epochs,constants.PARAMS['batch_size']))
loss2_all = torch.empty((n_epochs,constants.PARAMS['batch_size']))
grad_wrec1 = torch.empty((n_epochs,constants.PARAMS['batch_size'],model2.Wrec.shape[0],model2.Wrec.shape[0]))
grad_wrec2 = torch.empty((n_epochs,constants.PARAMS['batch_size'],model2.Wrec.shape[0],model2.Wrec.shape[0]))

#%% 

# model1.train()

# for i in range(n_epochs):
#     shuffling_order = \
#                 torch.randperm(constants.PARAMS['batch_size'],dtype=torch.long).to(device)
#     if i%20 == 0:
#         print('Epoch %d' %i)
#     for t in range(constants.PARAMS['batch_size']):
#         model2 = define_model2(constants.PARAMS,model1)

#         trial = constants.TRAINING_DATA['inputs'][:,shuffling_order[t],:].unsqueeze(1)
#         target = constants.TRAINING_DATA['targets'][shuffling_order[t]].unsqueeze(0)
        
        
#         # get predictions
#         output1, o1, hidden1 = model1(trial)
#         output2, o2, hidden2 = model2(trial)
        
#         # get loss
#         loss1 = loss_fn(output1.unsqueeze(0),target)
#         loss2 = loss_fn(output2.unsqueeze(0),target)
        
#         # zero gradients
#         optim1.zero_grad()
#         optim2.zero_grad()
        
#         # backprop
#         loss1.backward()
#         loss2.backward()
        
#         loss1_all[i,t] = loss1.item()
#         loss2_all[i,t] = loss2.item()
        
        
#         # step
#         optim1.step()
#         optim2.step()
        
#         grad_wrec1[i,t,:,:] = model1.Wrec.grad
#         grad_wrec2[i,t,:,:] = model2.Wrec_last.grad
        
        
#         model2.Wrec = model2.Wrec_last.clone().detach()


#%% mini batch training

loss1_all = torch.empty((n_epochs,constants.PARAMS['n_batches']))
loss2_all = torch.empty((n_epochs,constants.PARAMS['n_batches']))
grad_wrec1 = torch.empty((n_epochs,constants.PARAMS['n_batches'],model2.Wrec.shape[0],model2.Wrec.shape[0]))
grad_wrec2 = torch.empty((n_epochs,constants.PARAMS['n_batches'],model2.Wrec.shape[0],model2.Wrec.shape[0]))



model1.train()
inputs = constants.TRAINING_DATA['inputs']
targets = constants.TRAINING_DATA['targets']
for i in range(n_epochs):
    shuffling_order = \
                torch.randperm(constants.PARAMS['batch_size'],dtype=torch.long).to(device)
    if i%20 == 0:
        print('Epoch %d' %i)
    
    dset = list(zip(inputs.permute(1,0,-1),targets))
    data_loader = DataLoader(dset,
                                batch_size = constants.PARAMS['n_stim'],
                                shuffle=True)
    for batch, (data_mb,targets_mb) in enumerate(data_loader):
        model2 = define_model2(constants.PARAMS,model1)
        
        trial = data_mb.permute(1,0,-1)
        
        # get predictions
        output1, o1, hidden1 = model1(trial)
        output2, o2, hidden2 = model2(trial)
        
        # get loss
        loss1 = loss_fn(output1,targets_mb)
        loss2 = loss_fn(output2,targets_mb)
        
        # zero gradients
        optim1.zero_grad()
        optim2.zero_grad()
        
        # backprop
        loss1.backward()
        loss2.backward()
        
        loss1_all[i,batch] = loss1.item()
        loss2_all[i,batch] = loss2.item()
        
        
        # step
        optim1.step()
        optim2.step()
        
        grad_wrec1[i,batch,:,:] = model1.Wrec.grad
        grad_wrec2[i,batch,:,:] = model2.Wrec_last.grad
        
        
        model2.Wrec = model2.Wrec_last.clone().detach()
        

#%% get some gradient stats

model1_mean_gradients = grad_wrec1.view(n_epochs,512,-1).mean(-1).mean(-1)
model2_mean_gradients = grad_wrec2.view(n_epochs,512,-1).mean(-1).mean(-1)

abs_diff_mean_gradients = (model2_mean_gradients-model1_mean_gradients).abs()

abs_diff_norm_to_m2 = abs_diff_mean_gradients / model2_mean_gradients.abs()
abs_diff_norm_to_m1 = abs_diff_mean_gradients / model1_mean_gradients.abs()


plt.figure()
plt.plot(model1_mean_gradients,label='BPTT gradients')
plt.plot(model2_mean_gradients,label='no BPTT gradients')
plt.plot(abs_diff_mean_gradients,label='abs difference')
plt.xlabel('Epochs')
plt.ylabel('Mean gradient for Wrec')
plt.legend()

#%%
norm_diff = (model1_mean_gradients-model2_mean_gradients)/model2_mean_gradients
plt.figure()
plt.plot(norm_diff)

plt.ylabel('Wrec gradient')
plt.xlabel('Epoch')

plt.title('(BPTT gradient - no BPTT gradient) / no BPTT gradient')

#%% save the variables

save_path = constants.PARAMS['FULL_PATH']+'4cycles_fixationOn/'
check_path(save_path)
f = open(save_path+'grad_wrec1.pckl','wb')
pickle.dump(grad_wrec1,f)
f.close()

f = open(save_path+'grad_wrec2.pckl','wb')
pickle.dump(grad_wrec2,f)
f.close()

f = open(save_path+'model1_mean_gradients.pckl','wb')
pickle.dump(model1_mean_gradients,f)
f.close()

f = open(save_path+'model2_mean_gradients.pckl','wb')
pickle.dump(model2_mean_gradients,f)
f.close()


f = open(save_path+'loss1.pckl','wb')
pickle.dump(loss1,f)
f.close()

f = open(save_path+'loss2.pckl','wb')
pickle.dump(loss2,f)
f.close()




