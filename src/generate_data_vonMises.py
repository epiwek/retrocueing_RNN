#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 16:39:34 2020

@author: emilia
"""
import numpy as np
import torch
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import custom_plot as cplot
from scipy.stats import vonmises
from helpers import check_path


def make_stimuli_vonMises(params,epoch = 'train'):
    """
    Generate training / testing data for the network.
    
    --------------------
       Task structure

        - present 2 stimuli from circular space (have to be different colours) - in the paper used 64 points
        - pre-cue delay
        - present retrocue
        - post-cue delay
        - (if cond = 'probabilistic') - present the probe
        - report the correct stimulus identity
    
    Parameters
    ----------
    
    params : dict 
         Dictionary containing the experiment parameters (saved in the constants file).   
    epoch: str
        'train' or 'test' - the latter sorts trials by cued stimulus
        
    
    Returns
    -------
    data_dict : dict
        Entries:
        'inputs': input array, size (n_timepoints,n_trials,n_inp_channels)
        'targets': target vector of angular colour values, size : (n_trials)
        'c1','c2': angular colour values for loccation 1 and 2, size: (n_trials)
        'loc': truth array with the cued location values, loc1 and 2 correspond
            to the first and second columns, respectively. Size: (2,n_trials,1)
        'c1_label':integer label corresponding to the colour from location 1,
            size: (n_trials)
        'c2_label': analogous for location 2
        'example_processed_colours':  activity pattern for an exampe colour 
            encoded by the set of colour input units, size: (n_colours,n_colour_units)
        'output_tuning_centres': tuning peaks of the output units, size: (n_colour_units)
    """

    phi  = torch.linspace(-np.pi, np.pi, params['n_colCh']+1)[:-1] # # Tuning curve centers 
    colour_space = torch.linspace(-np.pi, np.pi, params['n_stim']+1)[:-1] # possible colours
    
    colour_combos = torch.tensor(list(itertools.product(np.arange(params['n_stim']),np.arange(params['n_stim'])))) # all possible colour1-colour2 combinations
    n_combos  = len(colour_combos)

    trial_dur = params['seq_len']
  
        
    if (params['stim_set_size'] % params['n_stim']*params['n_stim']*2)!= 0:
        raise ValueError('Full stim set size must be a multiple of the n of stim combinations')
    else:
        n_instances = params['stim_set_size'] // (n_combos*2) # n of trials per condition
    
    loc = torch.zeros((params['stim_set_size'],),dtype=torch.long)
    loc[:params['stim_set_size']//2]=1 # first half trials of location 0
    loc = torch.cat((loc.unsqueeze(-1),1.0-loc.unsqueeze(-1)),-1)
    
    c1, c2 = torch.zeros((params['stim_set_size'],)),torch.zeros((params['stim_set_size'],))
    
    # loop through all possible colour1-colour2 combinations and create n_instances of trials for each one
    for c in range(n_combos):
        trial_ix = np.arange(c*n_instances,(c+1)*n_instances) # for loc 0
        
        # set stimuli        
        s1 = colour_combos[c,0]
        s2 = colour_combos[c,1]
        
        c1[trial_ix] = colour_space[s1] # colour 1 - position in circular space
        c2[trial_ix] = colour_space[s2] # colour 2
        
        # repeat for trials with location 2
        c1[trial_ix+params['stim_set_size']//2] = colour_space[s1] # colour 1 
        c2[trial_ix+params['stim_set_size']//2] = colour_space[s2] # colour 2
        

        # # set target
        
        # T[trial_ix] = s1.type(torch.long)
        # T[trial_ix+params['batch_size']//2] = s2.type(torch.long)
        
    
    if epoch == 'test':
        # sort context 2 so that it's also blocked by tested stimulus
        loc2_ix = np.arange(params['stim_set_size']//2,params['stim_set_size'])
        loc2_ix_sorted = []
        for c in range(len(colour_space)):
            loc2_ix_sorted.append(np.where(c2[loc2_ix]==colour_space[c])[0])
        loc2_ix_sorted = np.array(loc2_ix_sorted)
        loc2_ix_sorted += params['stim_set_size']//2
        loc2_ix_sorted = np.reshape(loc2_ix_sorted,(-1))
        
        full_sorting_ix = np.concatenate((np.arange(params['stim_set_size']//2),loc2_ix_sorted))
        
        c1 = c1[full_sorting_ix]
        c2 = c2[full_sorting_ix]        
        
        # T = T[full_sorting_ix]
         
    
    # processed/encoded colours
    scale = max(vonmises.pdf(np.linspace(-np.pi,np.pi,100), params['kappa_val'],loc=0)) 
    #rescale the pdf to get a peak of height 1
    
    # processed / encoded colour stimuli
    c1p = np.zeros((params['stim_set_size'],params['n_colCh']))
    c2p = np.zeros((params['stim_set_size'],params['n_colCh']))
    
    example_cp = np.zeros((len(colour_space),params['n_colCh'])) 
    # distributed representations for each training colour
    
    # loop through all colour channels and apply their tuning functions to 
    # the current stimulus to determine their activation values
    for c in range(params['n_colCh']):
        c1p[:,c] = vonmises.pdf(c1,params['kappa_val'],phi[c])/scale
        c2p[:,c] = vonmises.pdf(c2,params['kappa_val'],phi[c])/scale
        
        example_cp[:,c] = vonmises.pdf(colour_space,params['kappa_val'],phi[c])/scale
        
    c1p = torch.tensor(c1p)
    c2p = torch.tensor(c2p)

    

    # targets for MSE Loss
    if params['target_type']=='Gaussian':
        T = torch.zeros((params['stim_set_size'],params['n_colCh'])) # for MSE loss
        T[:params['stim_set_size']//2] = c1p[:params['stim_set_size']//2,:] # location 0 trials
        T[params['stim_set_size']//2:] = c2p[params['stim_set_size']//2:,:] # location 1 trials
    elif params['target_type'] == 'class_label':
        T = torch.zeros((params['stim_set_size'],))
        T[:params['stim_set_size']//2] = torch.max(c1p[:params['stim_set_size']//2,:],1)[1]
        T[params['stim_set_size']//2:] = torch.max(c2p[params['stim_set_size']//2:,:],1)[1]
        T = T.type(torch.long)
        
        c1_label = torch.max(c1p,1)[1]
        c2_label =torch.max(c2p,1)[1]
    elif params['target_type']=='angle_val':
        T = torch.empty((params['stim_set_size'],))
        T[:params['stim_set_size']//2] = phi[torch.max(c1p[:params['stim_set_size']//2,:],1)[1]]
        T[params['stim_set_size']//2:] = phi[torch.max(c2p[params['stim_set_size']//2:,:],1)[1]]
        
        c1_label = torch.max(c1p,1)[1]
        c2_label = torch.max(c2p,1)[1]

   # dynamic input
    n_channels = params['n_colCh']*2 + 2 #stimuli + cues
        
    # stretch inputs in time
    c1p = c1p.repeat((params['trial_timings']['stim_dur'],1,1)).permute((-1,1,0))
    c2p = c2p.repeat((params['trial_timings']['stim_dur'],1,1)).permute((-1,1,0))
    loc = loc.repeat((params['trial_timings']['cue_dur'],1,1)).permute((-1,1,0))

    # trial structure
    #stim, delay1, cue, delay2,  optional probe, delay3
    inp = torch.zeros((n_channels,params['stim_set_size'],trial_dur))
    # order of inputs: cue, colours

    
    # STIM
    start = 0
    inp[2:params['n_colCh']+2,:,start:start+params['trial_timings']['stim_dur']] = c1p
    inp[params['n_colCh']+2:params['n_colCh']*2+2,:,start:start+params['trial_timings']['stim_dur']] = c2p

    # CUE
    start = params['trial_timings']['stim_dur']+params['trial_timings']['delay1_dur']
    inp[0:2,:,start:start+params['trial_timings']['cue_dur']] = loc # cue1 channel
    # inp[0:2,:,start:start+params['trial_timings']['stim_dur']] = loc # cue1 channel
    
    # PROBE
    if params['trial_timings']['probe_dur']!=0:
        # add a probe at the response screen
        # cue validity manipulated elsewhere - in retrocue_model.train_model via retrocue_model.add_noise
        
        start = params['trial_timepoints']['delay2_end']
        inp[0:2,:,start:start+params['trial_timings']['probe_dur']:] = \
                loc[:,:,0].unsqueeze(-1).repeat(1,1,params['trial_timings']['probe_dur'])
        
    
    # switch dimensions 
    inp = inp.permute(-1,1,0) # time, batch, n channels
    
    # save into a dictionary
    data_dict = {'inputs':inp,
                 'targets':T,
                 'c1':c1,
                 'c2':c2,
                 'c1_label':c1_label,
                 'c2_label':c2_label,
                 'loc':loc,
                 'example_processed_colours':example_cp,
                 'output_tuning_centres':phi}
    if np.logical_and(params['experiment_number']==3,params['cue_validity']==1):
        data_dict['cued_loc'] = loc
        data_dict['probed_loc'] = loc
        
    return data_dict


def change_cue_validity(inputs,params,opt='random'):
    """
    In the probabilistic cue paradigm, change a proportion of the cues to be 
    different from the probe.

    Parameters
    ----------
    inputs : array (n_timepoints,n_trials,n_inp_channels)
        Inputs to the network, only valid trials.
    params : dict
        Dictionary containing the experiment parameters.
    opt : str, optional
        'constrained' ensures equal frequencies of invalid trials for each 
        colour stimulus. Use for generating a balanced test dataset.The default
        is 'random'.

    Returns
    -------
    inputs : array (n_timepoints,n_trials,n_inp_channels)
        Modified input array containing both valid and invalid trials.
    invalid_trial_ixs : array
        Indexes of the invalid trials. Length depends on the proportion of valid
        trials, controlled by the 'cue_validity' parameter.

    """
    # get the cue timepoints
    cue_start = params['trial_timings']['stim_dur']+\
            params['trial_timings']['delay1_dur']
    cue_end = cue_start + params['trial_timings']['cue_dur']
    
    if opt == 'constrained':
        # equal props of invalid trials for each stimulus - for generating test data
        # n_invalid_trials = int(np.round((1-params['cue_validity'])*
        #                                 (params['stim_set_size']//
        #                                  params['n_trial_types'])))
        # set to 50% valid and invalid trials for balanced analyses
        n_invalid_trials = int(np.round(.5*(params['stim_set_size']//
                                         params['n_trial_types'])))
        invalid_trial_ixs = torch.stack([torch.arange(n_invalid_trials)]*
                                        params['n_trial_types'])
        invalid_trial_ixs = invalid_trial_ixs + \
            torch.arange(params['n_trial_types']).unsqueeze(1)*params['n_trial_instances_test']
        invalid_trial_ixs = invalid_trial_ixs.view(-1)
    else:  
        # draw some trials where the cue will be invalid at random - for training data
        n_invalid_trials = int(np.round((1-params['cue_validity'])*
                                        params['stim_set_size']))
        
        invalid_trial_ixs = np.random.choice(int(params['stim_set_size']),
                                              size = (n_invalid_trials,),
                                              replace = False)
    
    # change the retrocue on invalid trials
    inputs[cue_start:cue_end,invalid_trial_ixs,0:2] = \
        torch.abs(inputs[cue_start:cue_end,invalid_trial_ixs,0:2] - 1.0)
    
    return inputs,invalid_trial_ixs
     
   
def subset_of_trials(params,data_dict,trial_ixs):
    """
    Extract a subset of trials from the data dictionary.
    
    --------------------
    
    Parameters
    ----------
    data_dict  : dict
        Dictionary containing the dataset.
    trial_ixs : array
        Indexes of trials to be pulled out into a new data dictionary.
    Returns
    -------
    data_subset_dict : dict
        Changed entries:
        'inputs','targets','c1','c2','loc'
        Added entries: 
            'cued_loc', 'probed_loc'
    """
    
    data_subset_dict = {}
    data_subset_dict['inputs'] = data_dict['inputs'][:,trial_ixs,:]
    data_subset_dict['targets'] = data_dict['targets'][trial_ixs]
    data_subset_dict['c1'] = data_dict['c1'][trial_ixs]
    data_subset_dict['c2'] = data_dict['c2'][trial_ixs]
    data_subset_dict['loc'] = data_dict['loc'][:,trial_ixs,:]
    if params['experiment_number'] == 3:
        # add cued location dictionary entry
        cue_ix = params['trial_timepoints']['delay1_end']
        probe_ix = params['trial_timepoints']['delay2_end']
        data_subset_dict['cued_loc'] = data_dict['inputs'][cue_ix,trial_ixs,0]
        data_subset_dict['probed_loc'] = data_dict['inputs'][probe_ix,trial_ixs,0]
    
    return data_subset_dict


def generate_test_dataset(params, plot=True):
    '''
    Generatesa test dataset appropriate for the current experiment. Plots an 
    example trial.

    Parameters
    ----------
    params : dict
        Dictionary containing the experiment parameters.
    plot : bool, optional
        Flag for plotting an example trial. The default is True.

    Returns
    -------
    test_data : dict
        Test dataset(s).

    '''
    # increase number of stimuli
    params['stim_set_size']= params['n_trial_types'] * \
        params['n_trial_instances_test']
    test_data = make_stimuli_vonMises(params,epoch='test')
    
        
    if params['condition'] != 'deterministic':
        # testing dataset for probabilistic/neutral condition
        
        # set the indices of invalid trials
        test_data['inputs'],invalid_trial_ixs = \
            change_cue_validity(test_data['inputs'],params,opt='constrained')
        test_data['invalid_trial_ixs'] = invalid_trial_ixs
        valid_trial_ixs = np.setdiff1d(np.arange(params['stim_set_size']),
                                        invalid_trial_ixs)
        test_data['valid_trial_ixs'] = valid_trial_ixs
    
    if params['experiment_number'] < 3:
        # testing datasets for experiment 2
        # trained delay length, same as for expt  1
        params = update_time_params(params,params['test_delay_lengths'][0])
        test_data = make_stimuli_vonMises(params,epoch='test')
        
        # out-of-range temporal generalisation - longer
        params = update_time_params(params,params['test_delay_lengths'][1])
        test_data_outRange = make_stimuli_vonMises(params,epoch='test')
        test_data_outRange['delay_length'] = params['test_delay_lengths'][1]
        test_data_outRange['params'] = params
        
        # out-of-range temporal generalisation - shorter
        params = update_time_params(params,1)
        test_data_outRange_shorter = make_stimuli_vonMises(params,epoch='test')
        test_data_outRange_shorter['delay_length'] = 1
        test_data_outRange_shorter['params'] = params
        
        # in range
        params = update_time_params(params,params['test_delay_lengths'][2])
        test_data_inRange = make_stimuli_vonMises(params,epoch='test')
        test_data_inRange['delay_length'] = params['test_delay_lengths'][2]
        test_data_inRange['params'] = params
        
        test_data = {'trained':test_data,
                     'in-range':test_data_inRange,
                     'out-of-range':test_data_outRange,
                     'out-of-range-shorter':test_data_outRange_shorter}

    # plot an example trial
    sns.set_context("notebook", font_scale=1.5)
    sns.set_style("ticks")
    if  params['experiment_number'] >=3 :
        cplot.plot_example_stimulus(test_data['inputs'].T,
                                    cmap = sns.cubehelix_palette(start=.5,rot=-.75,as_cmap = True,reverse=True))
    else:
        params = update_time_params(params,params['test_delay_lengths'][0])
        cplot.plot_example_stimulus(test_data['trained']['inputs'].T,
                                    cmap = sns.cubehelix_palette(start=.5,rot=-.75,as_cmap = True,reverse=True))
        plt.xticks(np.arange(0,params['seq_len'],5),labels=np.arange(0,params['seq_len'],5)+1)
        plt.xlabel('time')
        plt.yticks(ticks=[-0.5,1.5,18.5,35.5],labels=[])
        fig = plt.gcf()
        fig.set_size_inches(3,5)
        plt.tight_layout()
        fig_path = params['FULL_PATH'] + 'figs/'
        check_path(fig_path)
        plt.savefig(fig_path + 'example_test_stimulus.png')
    
    return test_data


def update_time_params(params,delay_length):
    '''
    Updates the time parameters saves in the constants file. Used to make the 
    delay lengths in experiment 2 equal to those in experiment 1, as well as
    to generate test datasets with different trial lengths.

    Parameters
    ----------
    params : dict
        Dictionary containing the experiment parameters.
    delay_length : int
        Desired length of the delay interval in cycles.

    Returns
    -------
    params : dict
        updated time parameters.

    '''
    
    params['trial_timings']['delay1_dur'] = delay_length
    params['trial_timings']['delay2_dur'] = delay_length
        
    params['seq_len'] = sum(params['trial_timings'].values())

    # delay start and end points
    
    params['trial_timepoints'] = {}
    params['trial_timepoints']['delay1_start'] = params['trial_timings']['stim_dur']
    params['trial_timepoints']['delay1_end'] = params['trial_timings']['stim_dur']\
        +params['trial_timings']['delay1_dur']
    params['trial_timepoints']['delay2_start'] = params['trial_timings']['stim_dur']\
        +params['trial_timings']['delay1_dur']+params['trial_timings']['cue_dur']
    params['trial_timepoints']['delay2_end'] = params['trial_timings']['stim_dur']\
        +params['trial_timings']['delay1_dur']+params['trial_timings']['cue_dur']\
            +params['trial_timings']['delay2_dur']
            
    return params



