#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 19:33:26 2022

@author: emilia
"""

# estimate loss level that is equivalent to monkey performance
import constants
import numpy as np
from retrocue_model import custom_MSE_loss
from scipy.stats import vonmises
from helpers import circ_diff
from torch.nn import Softmax
import torch

# generate posteriors with arbitrary widths

kappa_vals = np.linspace(0.001,1000,500)
target_angle = constants.PARAMS['phi'][0]
target = torch.zeros((constants.PARAMS['n_out'],))
target[0] = 1
x = circ_diff(constants.PARAMS['phi'],target_angle)

loss = np.empty((len(kappa_vals,)))
abs_mean_err = np.empty((len(kappa_vals,)))
sm = Softmax(dim=-1)

for i,k in enumerate(kappa_vals):
    output = sm(torch.from_numpy(vonmises.pdf(x,kappa=k)))
    abs_mean_err[i] = np.degrees((np.abs(x)*output).sum())
    loss[i] = custom_MSE_loss(constants.PARAMS, output, target_angle)
    
# find the error closest to monkey value and check the corresponding loss value
ix = np.argmin(np.abs(abs_mean_err-51))
threshold = loss[ix]
print('Threshold: %.5f' %threshold)
    
# loss = torch.empty((17,))  
# diff = helpers.circ_diff(constants.PARAMS['phi'],target)
# kappas = (np.linspace(0.0001,17,17))
# sm = nn.Softmax(dim=-1)

# plt.figure()
# alphas=np.linspace(.2,1,17)
# for i in range(17):
#     output = sm(torch.from_numpy(vonmises.pdf(diff,kappas[i])))
#     loss[i]= custom_MSE_loss(constants.PARAMS,output,target)
#     ix = np.argsort(diff)
#     plt.plot(diff[ix],output[ix],'ko-',alpha=alphas[i])
