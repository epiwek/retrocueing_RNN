#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 16:52:18 2021

Plotting script to visualise:
    1) stimulus colours used during training
    2) colour unit tuning curves - applies to both the input and output units of the RNN
    
@author: emilia
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import constants
from scipy.stats import vonmises



phi  = torch.linspace(-np.pi, np.pi, constants.PARAMS['n_colCh']+1)[:-1] # Tuning curve centers 
colour_space = torch.linspace(-np.pi, np.pi, constants.PARAMS['n_stim']+1)[:-1] # Training colours
plot_colours = sns.color_palette("hsv", len(colour_space))

sns.set_style("white")
sns.set_context("paper")

#%% plot all training stimuli

fig = plt.figure(figsize=(10.0,2.9))
ax = fig.add_subplot(121,polar = True)

ax.scatter(colour_space,np.ones(len(colour_space)),c=plot_colours)
ax.set_yticks([])
ax.tick_params(axis='x', which='major', pad=5)

ax.set_title('Training set colours',
             fontdict = {'fontsize': 10,
                         'fontweight': 'bold'},
             y = 1.0,
             pad = 40)

#%% plot tuning curves for each output channel


ax2 = fig.add_subplot(122)
tc_colours = sns.color_palette("hsv", len(phi))

x_vals = np.linspace(-np.pi,np.pi,100)
scale = max(vonmises.pdf(x_vals, constants.PARAMS['kappa_val'],loc=0)) #rescale the pdf to get a peak of height 1

for i in range(len(phi)):
    y_vals = vonmises.pdf(x_vals,constants.PARAMS['kappa_val'],phi[i])/scale
    ax2.plot(x_vals,y_vals,'-',c=tc_colours[i])
    
ax2.set_xticks(np.linspace(-np.pi,np.pi,5))
ax2.set_xticklabels(np.linspace(-360,360,5))
ax2.set_yticks([0,0.5,1])

ax2.set_xlabel('Colour value [°]')
ax2.set_ylabel('Unit activation')


ax2.set_title('Colour unit tuning curves',
             fontdict = {'fontsize': 10,
                         'fontweight': 'bold'},
             y = 1.0,
             pad = 40)

plt.tight_layout()

plt.savefig(constants.PARAMS['FULL_PATH']+'colour_stimuli.png')

#%% show the distributed representation for an example training colour

fig = plt.figure(figsize=(6.0,2.9))

n_colours = 3

for i in np.linspace(0,constants.PARAMS['n_stim'],num=n_colours,endpoint=False,dtype=int):
    plt.plot(constants.TRAINING_DATA['example_processed_colours'][i,:],'o-',c = tc_colours[i])

plt.xlabel('Colour channel')
plt.ylabel('Channel activity')
plt.title('Distributed representations for %d example colours' %n_colours)

plt.tight_layout()
