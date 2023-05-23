#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 19:32:03 2020

@author: emilia
"""
import numpy as np
import torch
from torch import nn
# torch.manual_seed(0)

def define_model(params, device,model_class = 'RNN'):
    global model, RNN # outputs need to be global variables, otherwise will be unable to save model with pickle
    
    #%% define RNN class
    # if model_class == 'RNN' :
    class RNN(nn.Module):
        def __init__(self, params):
            super(RNN, self).__init__()
            # input
            self.n_rec = params['n_rec'] # number of recurrent neurons
            self.n_inp = params['n_inp']
            self.n_out = params['n_out']
    
            self.Wrec = nn.RNN(self.n_inp,self.n_rec, nonlinearity = 'relu')
            
            self.Wrec.bias_hh_l0 = nn.Parameter(self.Wrec.bias_hh_l0*params['init_scale'])
            self.Wrec.bias_ih_l0 = nn.Parameter(self.Wrec.bias_ih_l0*params['init_scale'])
            self.Wrec.weight_hh_l0 = nn.Parameter(self.Wrec.weight_hh_l0*params['init_scale'])
            self.Wrec.weight_ih_l0 = nn.Parameter(self.Wrec.weight_ih_l0*params['init_scale'])
            
            
            # change to Kaiming initialisation
            # self.Wrec.weight_ih_l0 = nn.Parameter(torch.randn(self.n_rec,self.n_inp)*((np.sqrt(2)/np.sqrt(self.n_inp))/10))
            # self.Wrec.weight_hh_l0 = nn.Parameter(torch.randn(self.n_rec,self.n_rec)*((np.sqrt(2)/np.sqrt(self.n_rec))/10))
            
            # self.Wrec.bias_ih_l0 = nn.Parameter(torch.zeros(self.n_rec))
            # self.Wrec.bias_hh_l0 = nn.Parameter(torch.zeros(self.n_rec))
            
            # self.Wrec.weight_ih_l0 = nn.Parameter(-2*np.sqrt(1/200) * torch.rand(self.n_rec,self.n_inp)+ np.sqrt(1/200))
            # self.Wrec.weight_hh_l0 = nn.Parameter(-2*np.sqrt(1/200)*torch.rand(self.n_rec,self.n_rec)+np.sqrt(1/200))
            
            # self.Wrec.bias_ih_l0 = nn.Parameter(-2*np.sqrt(1/200) *torch.rand(self.n_rec) + np.sqrt(1/200))
            # self.Wrec.bias_hh_l0 = nn.Parameter(-2*np.sqrt(1/200) * torch.rand(self.n_rec) + np.sqrt(1/200))
            
            
            # input and hidden
            self.out = nn.Linear(self.n_rec, self.n_out) # output layer
            
            #self.out.weight = nn.Parameter(torch.randn(self.n_out,self.n_rec)*(np.sqrt(2)/np.sqrt(self.n_rec)))
            #self.out.bias = nn.Parameter(torch.zeros(self.n_out))
            
            #ignore (scratch code):
            #self.inp = nn.Linear(17, self.n_neurons) # input weights
            #self.Wrec = nn.Parameter(torch.randn(self.n_neurons, self.n_neurons) / np.sqrt(self.n_neurons))  # recurrent weights    
              #self.Wrec.bias_ih_l0 = nn.Parameter(torch.ones(n_rec))
            #self.Wrec.bias_hh_l0 = nn.Parameter(torch.ones(n_rec))
            #self.Wrec.weight_hh_l0 = nn.Parameter(self.Wrec.weight_hh_l0*1e-2)
            #self.Wrec.weight_ih_l0 = nn.Parameter(self.Wrec.weight_ih_l0*1e-2)
            
        # def step(self, input_ext, hidden):
        #     output, hidden = self.Wrec(input_ext.unsqueeze(0),hidden)
        #     # output = self.out(hidden)
        #     return output, hidden
        # #     # ignore:
        # #     #hidden = torch.relu(torch.matmul(self.Wrec, hidden.unsqueeze(-1)).squeeze() + self.inp(input_ext))
            
        def forward(self, inputs):
            """
            Run the RNN with input timecourses
            """
            # Initialize network state
            # hidden = torch.zeros((1,inputs.size(1), self.n_rec)) # 0s
            inputs_with_noise = add_noise(data,params,device)
            o, h_n = self.Wrec(inputs_with_noise)
            # Run the input through the network - across time
            # for i in range(inputs.size(0)):
            #     # print(inputs[i,0,0].T.shape)
            #     output, hidden = self.step(inputs[i, :, :], hidden)
                
            #     std,mu = torch.std_mean(hidden)
            #     print(std,mu)
            #     # print(hidden)
            # output = self.out(output)
            # return output.squeeze(), hidden
            
            output = self.out(h_n)
            return output.squeeze(), o, h_n   
        
    model = RNN(params)
    
    return model, RNN
    
    # else:
    #     class LSTM(nn.Module):
    #         def __init__(self, n_inp, n_rec, n_out):
    #             super(LSTM, self).__init__()
    #             # input
    #             self.n_rec = n_rec # number of recurrent neurons
    #             self.n_inp = n_inp
    #             self.n_out = n_out
        
    #             self.Wrec = nn.LSTM(self.n_inp,self.n_rec)
                
    #             # change to Kaiming initialisation
    #             # self.Wrec.weight_ih_l0 = nn.Parameter(torch.randn(self.n_rec,self.n_inp)*((np.sqrt(2)/np.sqrt(self.n_inp))))
    #             # self.Wrec.weight_hh_l0 = nn.Parameter(torch.randn(self.n_rec,self.n_rec)*((np.sqrt(2)/np.sqrt(self.n_rec))))
                
    #             # self.Wrec.bias_ih_l0 = nn.Parameter(torch.zeros(self.n_rec))
    #             # self.Wrec.bias_hh_l0 = nn.Parameter(torch.zeros(self.n_rec))
                
    #             # input and hidden
    #             self.out = nn.Linear(self.n_rec, self.n_out) # output layer
                
    #             #self.out.weight = nn.Parameter(torch.randn(self.n_out,self.n_rec)*(np.sqrt(2)/np.sqrt(self.n_rec)))
    #             #self.out.bias = nn.Parameter(torch.zeros(self.n_out))
        
                
    #         # def step(self, input_ext, hidden):
    #         #     o, hidden = self.Wrec(input_ext.unsqueeze(0),hidden)
    #         #     output = self.out(hidden)
    #         #     return output, hidden
    #         #     # ignore:
    #         #     #hidden = torch.relu(torch.matmul(self.Wrec, hidden.unsqueeze(-1)).squeeze() + self.inp(input_ext))
                
    #         def forward(self, inputs):
    #             """
    #             Run the LSTM with input timecourses
    #             """
                
    #             o, (hidden, cell) = self.Wrec(inputs) #,(h_0, c_0))
                
    #             output = self.out(o[-1,:,:].squeeze())
    #             return output.squeeze(), o   
            
    #     model = LSTM(params)
        
        # return model, LSTM

# implementation from scratch
# def define_model(n_inp, n_rec,n_out):
#     global model, RNN # outputs need to be global variables, otherwise will be unable to save model with pickle
    
#     #%% define RNN class
    
#     class RNN(nn.Module):
#         def __init__(self, n_inp, n_rec, n_out):
#             super(RNN, self).__init__()
#             # input
#             self.n_rec = n_rec # number of recurrent neurons
#             self.n_inp = n_inp
#             self.n_out = n_out
#             self.gain_inp = np.sqrt(2)/np.sqrt(self.n_inp)
#             self.gain_rec = (np.sqrt(2)/np.sqrt(self.n_rec))/2
            
#             self.ih = nn.Parameter(self.gain_inp*torch.randn((self.n_rec,self.n_inp)))
#             self.hh = nn.Parameter(self.gain_rec*torch.randn((self.n_rec,self.n_rec)))
#             self.bias_h = nn.Parameter(torch.zeros(self.n_rec))
            
#             # input and hidden
#             self.out = nn.Linear(self.n_rec, self.n_out) # output layer
#             self.out.bias = nn.Parameter(torch.zeros(self.n_out))

            
#         def step(self, input_ext, hidden):
#             batch = input_ext.shape[0]
#             inp = torch.matmul(self.ih,input_ext.T)
#             hidden = torch.matmul(self.hh,hidden) + inp + torch.stack([self.bias_h] * batch,dim=1)
#             hidden = torch.relu(hidden)
            
#             output = self.out(hidden.T)
#             return output, hidden
#         #     # ignore:
#         #    
            
#         def forward(self, inputs):
#             """
#             Run the RNN with input timecourses
#             """
#             # Initialize network state
#             batch = inputs.shape[1]
#             hidden = torch.zeros((self.n_rec,batch)) # 0s
            
#             hidden_n = torch.zeros((inputs.shape[0],self.n_rec,batch))
            
#             #o, h_n = self.Wrec(inputs)
#             # Run the input through the network - across time
#             for i in range(inputs.size(0)):
#                 output, hidden = self.step(inputs[i, :, :], hidden)
#                 hidden_n[i,:,:] = hidden.clone()
#                 # print('Timestep %d' %i)
#                 # mean, std = torch.std_mean(hidden)
#                 # print('    Recurrent layer std = %.4f, mean =  %.4f' %(mean,std))
#                 # mean, std = torch.std_mean(output)
#                 # print('    Output layer std = %.4f, mean =  %.4f' %(mean,std))
                
                
                
#             return output.squeeze(), hidden, hidden_n
            
#             #output = self.out(h_n)
#             #return output.squeeze(), h_n   
        
#     model = RNN(n_inp,n_rec,n_out)
    
#     return model, RNN


# def define_model(n_inp, n_rec,n_out):
