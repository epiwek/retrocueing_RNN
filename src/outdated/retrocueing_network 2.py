#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 17:52:19 2021

@author: emilia
"""
import numpy as np
import torch
from torch import nn
from torch import optim
import pickle
import matplotlib.pyplot as plt
from generate_data_1hot import make_stimuli_1hot
# from generate_data_gaussian import make_stimuli_gaussian
from generate_data_vonMisses import make_stimuli_vonMises

import time
import os.path
import define_dataset as dset




start_time = time.time()


def define_model(n_inp, n_rec,n_out):
    global model, RNN # outputs need to be global variables, otherwise will be unable to save model with pickle
    
    #% define RNN class

    class RNN(nn.Module):
        def __init__(self, n_inp, n_rec, n_out):
            super(RNN, self).__init__()
            # input
            self.n_rec = n_rec # number of recurrent neurons
            self.n_inp = n_inp
            self.n_out = n_out
    
            self.Wrec = nn.RNN(self.n_inp,self.n_rec, nonlinearity = 'relu')
            
            # self.Wrec.bias_hh_l0 = nn.Parameter(self.Wrec.bias_hh_l0/2)
            # self.Wrec.bias_ih_l0 = nn.Parameter(self.Wrec.bias_ih_l0/2)
            # self.Wrec.weight_hh_l0 = nn.Parameter(self.Wrec.weight_hh_l0/2)
            # self.Wrec.weight_ih_l0 = nn.Parameter(self.Wrec.weight_ih_l0/2)
            
            
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
        #     o, hidden = self.Wrec(input_ext.unsqueeze(0),hidden)
        #     output = self.out(hidden)
        #     return output, hidden
        #     # ignore:
        #     #hidden = torch.relu(torch.matmul(self.Wrec, hidden.unsqueeze(-1)).squeeze() + self.inp(input_ext))
            
        def forward(self, inputs):
            """
            Run the RNN with input timecourses
            """
            
            o, h_n = self.Wrec(inputs) # forward pass through the recurrent layer
            output = self.out(h_n) # pass the last recurrent output through the decoding layer
            return output.squeeze(), h_n
        
    model = RNN(n_inp,n_rec,n_out)
    
    return model, RNN

def train_model(args,model, train_data, loss_fn, optimizer):
    
    # load parameters and input data
    dset.load_input_data(fileloc,datasetname)
    
    # initialise model
    model, RNN_class = define_model(n_inp, n_rec,n_colCh)
    
    if loss_type == 'MSE':
        loss_fn = nn.MSELoss()
    elif loss_type == 'CLE':
        loss_fn = nn.CrossEntropyLoss()
    
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    
recurrent_train(args, model, device, train_loader, optimizer, criterion, epoch, printOutput=True):
    