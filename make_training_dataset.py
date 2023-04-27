#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 16:40:05 2020

@author: emilia
"""

from make_stimuli import make_stimuli

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
batch_size = 50
n_epochs = 12000

#%% construct a training dataset

inputs = {}
targets = {}
for i in range(n_epochs):
    print(i)
    I, loc, c1, c2, T = make_stimuli(n_stim,n_colCh,batch_size,trial_timings,tuning_params)
    inputs[str(i)] = I
    targets[str(i)] = T

#%% save
import pickle

training_data = [inputs,targets]
path = '/Users/emilia/OneDrive - Nexus365/DPHIL/PROJECTS/retrocue_RNN_project/shared_repo/retrocue_rnn/emilia/data/training_data_sameTrainingData/'
f = open(path+'common_training_dataset.pckl','wb')
pickle.dump(training_data,f)
f.close()
