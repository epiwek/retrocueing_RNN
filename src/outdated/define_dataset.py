#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 16:13:55 2021

Functions for creating and visualising the dataset.
@author: emilia
"""

import numpy as np
import torch
import itertools
import constants as const
from scipy.stats import vonmises
from torch.utils.data import Dataset, DataLoader
# from torch.utils.tensorboard import SummaryWriter
# from datetime import datetime



class CreateDataset(Dataset):
    """A class to hold a dataset.
    - input tensor
    - target
    - cue
    - colour1val
    - colour2val
    - label
    """

    def __init__(self, dataset):
        """
        Args:
            datafile (string): name of numpy datafile
        """
        self.index = dataset['index']
        self.label = dataset['label']
        self.input = dataset['input']
        self.target = dataset['target']
        self.cue = dataset['retrocue']
        self.c1 = dataset['colour1value']
        self.c2 = dataset['colour2value']
        
        self.data = {'index':self.index,'input':self.input,'target':self.target, 'retrocue':self.cue,
               'colour1value':self.c1, 'colour2value':self.c2,  'labels':self.labels}


    def __len__(self):
        return len(self.index)

    def __getitem__(self, ix):
        # for retrieving either a single sample of data, or a subset of data
        # lets us retrieve several items at once (HRS: may not be actually used)
        if torch.is_tensor(ix):
            ix = ix.tolist()
        sample =  {'index':self.index[ix],'input':self.input[:,ix,:],'target':self.target[ix], 'retrocue':self.cue[ix],
               'colour1value':self.c1[ix], 'colour2value':self.c2[ix],  'labels':self.labels[ix]}
        return sample


def load_input_data(fileloc,datasetname):
    # load an existing dataset
    print('Loading dataset: ' + datasetname + '.npy')
    data = np.load(fileloc+datasetname+'.npy', allow_pickle=True)
    numpy_dataset = data.item()

    # turn out datasets into pytorch Datasets
    dataset = CreateDataset(numpy_dataset)
    print('Done.')
    
    return dataset, numpy_dataset
        

def generate_input_data(filename,args):
   
    # args:
    # n_stim
    # nColCh
    # trial_timings
    # kappa
    # epoch
    # add_fixation
    # test = True - if want data to be ordered by the cued colour, e.g. for PCA
    
    
    # Task structure

    # STIMULUS  - present 2 stimuli from circular space
    # DELAY1    - optional fixation
    # CUE       - present the location retrocue
    # DELAY2    - optional fixation
    # report the correct stimulus (replicate the input pattern)
    
    print('Generating dataset...')
    
    
    phi  = torch.linspace(-np.pi, np.pi, args.n_colCh+1)[:-1] # Tuning curve centers 
    colour_space = torch.linspace(-np.pi, np.pi, args.n_stim+1)[:-1] # All colour values
    
    # Get all possible colour1-colour2 combinations - look-up table
    colour_combos = torch.tensor(list(itertools.product
                                      (np.arange(args.n_stim),
                                       np.arange(args.n_stim))))
    n_combos  = len(colour_combos)
    trial_dur = sum(args.trial_timings.values())
    
    # each trial will repeat once
    n_trials = (args.n_stim**2)*2
    n_instances = n_trials // (n_combos*2) # n of trials per condition, should be 1
    
    # Create the location retrocue matrix
    loc = torch.zeros((n_trials,)) # location of the retrocue on each trial
    loc[:n_trials//2]=1 # first half trials of location 0
    loc = torch.cat((loc.unsqueeze(-1),1.0-loc.unsqueeze(-1)),-1)
    
    # Create the colour and label vectors
    c1, c2 = torch.zeros((n_trials,)),torch.zeros((n_trials,)) # colour stimuli
    labels = torch.empty((n_trials,)) # tensor containing labels for all trials
    
    if args.loss_type == 'MSE':
        target = torch.zeros((args.n_trials,args.n_colCh))
    elif args.loss_type == 'CE':
        target = torch.zeros((args.n_trials,))
    
    # Loop over all the possible colour combinations
    for c in range(n_combos):
        trial_ix = np.arange(c*n_instances,(c+1)*n_instances) # for loc 0
        
        s_ix = c #stimulus combination index for the look-up table
        
        s1 = colour_combos[s_ix,0] #index of stimulus 1
        s2 = colour_combos[s_ix,1] #index of stimulus 2
        
        # set the colours for location 1 trials
        c1[trial_ix] = colour_space[s1] # colour 1 - position in circular space
        c2[trial_ix] = colour_space[s2] # colour 2
        
        # repeat for trials with location 2
        c1[trial_ix+n_trials//2] = colour_space[s2] # colour 1, use s2 so that trial order is blocked by the cued location
        c2[trial_ix+n_trials//2] = colour_space[s1] # colour 2
        
        # set labels
        labels[trial_ix] = ['c'+colour_combos[s_ix,0]+','
                            +'c'+colour_combos[s_ix,1]+','
                            +'loc0']
        labels[trial_ix+args.n_trials//2] = ['c'+colour_combos[s_ix,1]+','
                                        +'c'+colour_combos[s_ix,0]
                                        +','+'loc1']
        
        # Generate targets - if using Cross Entropy loss only
        if args.loss_type == 'CE':
            target[trial_ix] = s1
            target[trial_ix+args.n_trials//2] = s1
    
    # Process/encode the stimulus colours:
    #   the activation value of each input unit is given by the von Mises pdf
    #   and depends on the units preferred colour (i.e. location of the peak, phi)
    
    scale = max(vonmises.pdf(np.linspace(-np.pi,np.pi,100), args.kappa,loc=0))
    #rescale the pdf to get a peak of height 1
    
    c1p = np.zeros((n_trials,args.n_colCh)) # processed stimuli
    c2p = np.zeros((n_trials,args.n_colCh))
    for c in range(args.n_colCh):
        c1p[:,c] = vonmises.pdf(c1,args.kappa,phi[c])/scale
        c2p[:,c] = vonmises.pdf(c2,args.kappa,phi[c])/scale
    c1p = torch.tensor(c1p) # convert into torch tensors
    c2p = torch.tensor(c2p)
    
    # Generate targets - if using MSE loss only
    if args.loss_type == 'MSE':
        target[:args.n_trials//2,:] = c1p[:args.n_trials//2,:] # loc1 trials
        target[args.n_trials//2:,:] = c2p[args.n_trials//2:,:] # loc2 trials
    
    
    # Dynamic input - create an input matrix : n_channels x time
    args.n_channels = args.n_colCh*2 + 2 + int(args.add_fixation) 
    #number of input channels = n stimuli + cues + (optional) fixation
    
    # stretch inputs in time
    c1p = c1p.repeat((args.trial_timings['stim_dur'],1,1)).permute((-1,1,0))
    c2p = c2p.repeat((args.trial_timings['stim_dur'],1,1)).permute((-1,1,0))
    loc = loc.repeat((args.trial_timings['cue_dur'],1,1)).permute((-1,1,0))

    # Trial structure : stim, delay1, cue, delay2
    # Order of inputs: cue, colour, fixation
    inp = torch.zeros((args.n_channels,args.n_trials,trial_dur)) # input matrix
    
    
    # STIM
    start = args.trial_timings['ITI_dur'] # start time for this trial period
    inp[2:args.n_colCh+2,:,start:start+args.trial_timings['stim_dur']] = c1p
    inp[args.n_colCh+2:args.n_colCh*2+2,:,start:start+args.trial_timings['stim_dur']] = c2p

    # CUE
    start = args.trial_timings['ITI_dur']+args.trial_timings['stim_dur']+args.trial_timings['delay1_dur']
    inp[0:2,:,start:start+args.trial_timings['cue_dur']] = loc # cue1 channel
        
    if args.add_fixation:
        # DELAY 1
        start = args.trial_timings['ITI_dur']+args.trial_timings['stim_dur']
        inp[-1,:,start:start+args.trial_timings['delay1_dur']] = 1; # fixation on during delay1
        
        # DELAY 2
        start = args.trial_timings['ITI_dur']+args.trial_timings['cue_dur']\
            +args.trial_timings['delay1_dur']+args.trial_timings['cue_dur']
        inp[-1,:,start:start+args.trial_timings['delay2_dur']] = 1 # delay2
   
    
    # Permute dimensions to get a (time, batch, n channels) input matrix
    inp = inp.permute(-1,1,0)
    
    data = {'index':np.arange(args.n_trials),'input':inp,'target':target, 'retrocue':loc,
               'colour1value':c1, 'colour2value':c2,  'labels':labels}
    
    np.save(const.DATASET_DIRECTORY+filename+'.npy', data)

    # convert into pytorch Datasets
    dataset = CreateDataset(data)
    
    
    print('Done.')
    return dataset    
   

# def load_input_data(fileloc,datasetname):
#     # load an existing dataset
#     print('Loading dataset: ' + datasetname + '.npy')
#     data = np.load(fileloc+datasetname+'.npy', allow_pickle=True)
#     numpy_trainset = data.item().get("trainset")
#     numpy_testset = data.item().get("testset")
#     numpy_crossvalset = data.item().get("crossval_testset")
    
#     # turn out datasets into pytorch Datasets
#     trainset = CreateDataset(numpy_trainset)
#     testset = CreateDataset(numpy_testset)
#     crossvalset = CreateDataset(numpy_crossvalset)
    
#     return trainset, testset, crossvalset, numpy_trainset, numpy_testset, numpy_crossvalset