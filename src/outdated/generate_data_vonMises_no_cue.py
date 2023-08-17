#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 16:39:34 2020

@author: emilia
"""
import numpy as np
import torch
import itertools
from scipy.stats import vonmises
import pdb
import seaborn as sns
import matplotlib.pyplot as plt
import custom_plot as cplot
from helpers import check_path

#%% to do - crreate a BatchSizeError

# class Error(Exception):
#     """Base class for exceptions in this module."""
#     pass

# class BatchSizetError(Error):
#     """Exception raised for errors in the input.

#     Attributes:
#         expression -- input expression in which the error occurred
#         message -- explanation of the error
#     """

#     def __init__(self, expression, message):
#         self.expression = expression
#         self.message = message
#%%
def make_stimuli_vonMises(params,epoch = 'train'):
    """
    Generate training / testing data for the net.
    
    --------------------
       Task structure

        - present 2 stimuli from circular space (have to be different colours) - in the paper used 64 points
        - delay
        - present retrocue
        - delay
        - (if cond = 'probabilistic') - probe
        - report the correct stimulus identity - recreate the input pattern 
                corresponding to the cued/probed stimulus
    
    Parameters
    ----------
    
    params : dictionary 
            
    epoch: 'train' or 'test' - the latter sorts trials by cued stimulus
        
    
    Returns
    -------
    data_dict : dictionary
        entries:
        'inputs':inp,
        'targets':T,
        'c1','c2
        'loc'
    """
    # Task structure

    # present 2 stimuli from circular space (have to be different colours) - in the paper used 64 points
    # delay
    # present retrocue
    # delay
    # report the correct stimulus identity (class label)
    
    phi  = torch.linspace(-np.pi, np.pi, params['n_colCh']+1)[:-1] # # Tuning curve centers 
    colour_space = torch.linspace(-np.pi, np.pi, params['n_stim']+1)[:-1] # possible colours
    
    colour_combos = torch.tensor(list(itertools.product(np.arange(params['n_stim']),np.arange(params['n_stim'])))) # all possible colour1-colour2 combinations
    n_combos  = len(colour_combos)

    trial_dur = sum(params['trial_timings'].values())
  
    # if constraints == 'off':
    #     # relevant location
    #     loc = torch.randint(2,(params['batch_size'],),dtype = torch.float)
    #     loc = torch.cat((loc.unsqueeze(-1),1.0-loc.unsqueeze(-1)),-1) # 2 binary channels, size = batch x 2
    
    #     # colour stimuli
    #     c1 = colour_space[torch.randint(len(colour_space),(params['batch_size'],))] # colour 1
    #     c2 = colour_space[torch.randint(len(colour_space),(params['batch_size'],))]
        
    #     # use only 4 stimuli
    #     #ix = torch.randint(2,(params['batch_size'],))
    #     #c1 = colour_space[ix] # colour 1
    #     #c2 = colour_space[1-ix]

    #     # constrain so that colours have to be different
    #     while (c2 == c1).any():
    #         rep_ix = np.where(c2==c1)
    #         c2[rep_ix] =  colour_space[torch.randint(len(colour_space),(len(rep_ix),))] 

    # else: # add constraints to have e.g. equal trial numbers for each condition for PCA
        
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
    scale = max(vonmises.pdf(np.linspace(-np.pi,np.pi,100), params['kappa_val'],loc=0)) #rescale the pdf to get a peak of height 1
    
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
        c2_label =torch.max(c2p,1)[1]
        
    
    

    
    # determine target response
    # T = torch.randint(1,(params['batch_size'],)) # target colour ix
    # resp_col = torch.zeros((params['batch_size'],)) # actual colour value
    # resp_col[np.where(loc[:,0])[0]] = c1[np.where(loc[:,0])[0]] # respond to colour 1
    # resp_col[np.where(loc[:,1])[0]] = c2[np.where(loc[:,1])[0]] # respond to colour 2

    # convert colour value (scalar) into a target index vector - for CrossEntropyLoss
    # for i in range(params['batch_size']):
    #     col_ix = np.where(colour_space == resp_col[i])
    #     T[i] = col_ix[0][0]

   # dynamic input
    n_channels = params['n_colCh']*2 + 2 + int(params['add_fixation']) #stimuli + cues + (optional) fixation
        
    # stretch inputs in time
    c1p = c1p.repeat((params['trial_timings']['stim_dur'],1,1)).permute((-1,1,0))
    c2p = c2p.repeat((params['trial_timings']['stim_dur'],1,1)).permute((-1,1,0))
    loc = loc.repeat((params['trial_timings']['cue_dur'],1,1)).permute((-1,1,0))

    # trial structure
    #ITI with fixation, stim, delay with fixation, cue, delay with fixation
    inp = torch.zeros((n_channels,params['stim_set_size'],trial_dur))
    # order of inputs: cue, color, fixation

    # ITI
    #inp[-1,:,0:params['trial_timings']['ITI_dur']] = 1; # fixation on during ITI - turned off for now
    
    # STIM
    start = params['trial_timings']['ITI_dur']
    inp[2:params['n_colCh']+2,:,start:start+params['trial_timings']['stim_dur']] = c1p
    inp[params['n_colCh']+2:params['n_colCh']*2+2,:,start:start+params['trial_timings']['stim_dur']] = c2p

    # CUE
    # start = params['trial_timings']['ITI_dur']+params['trial_timings']['stim_dur']+params['trial_timings']['delay1_dur']
    # inp[0:2,:,start:start+params['trial_timings']['cue_dur']] = loc # cue1 channel
    # inp[0:2,:,start:start+params['trial_timings']['stim_dur']] = loc # cue1 channel
    
    if params['add_probe']:
        # add a probe at the response screen
        # cue validity manipulated elsewhere - in retrocue_model.train_model via retrocue_model.add_noise
        if loc.shape[-1]>1:
            # if cue duration is longer than 1 cycle
            inp[0:2,:,-params['trial_timings']['resp_dur']:] = \
                loc[:,:,0].unsqueeze(-1).repeat(1,1,params['trial_timings']['resp_dur'])
        else:
            inp[0:2,:,-params['trial_timings']['resp_dur']:] = \
                loc.repeat(1,1,params['trial_timings']['resp_dur'])
        
    if params['add_fixation']:
        # DELAY 1
        start = params['trial_timings']['ITI_dur']+params['trial_timings']['stim_dur']
        inp[-1,:,start:start+params['trial_timings']['delay1_dur']] = 1; # fixation on during delay1
        
        # DELAY 2
        start = params['trial_timings']['ITI_dur']+params['trial_timings']['cue_dur']\
            +params['trial_timings']['delay1_dur']+params['trial_timings']['cue_dur']
        inp[-1,:,start:start+params['trial_timings']['delay2_dur']] = 1 # delay2
    
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
    
    return data_dict


def change_cue_validity(inputs,params,opt='random'):
    # get the cue timepoints
    cue_start = params['trial_timings']['ITI_dur']+\
        params['trial_timings']['stim_dur']+\
            params['trial_timings']['delay1_dur']
    cue_end = cue_start + params['trial_timings']['cue_dur']
    
    if opt == 'constrained':
        # equal props of invalid trials for each stimulus - for generating test data
        n_invalid_trials = int(np.round((1-params['cue_validity'])*
                                        (params['stim_set_size']//
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
    # inputs[cue_start:cue_end,invalid_trial_ixs,0:2] = \
    #     torch.abs(inputs[cue_start:cue_end,invalid_trial_ixs,0:2] - 1.0)
    
    return inputs,invalid_trial_ixs
     
   
def subset_of_trials(data_dict,trial_ixs):
    """
    Extract a subset of trials from the data dictionary.
    
    --------------------
    
    Parameters
    ----------
    data_dict  : dictionary
    trial_ixs : array-like
    
    Returns
    -------
    data_subset_dict : dictionary
        changed entries:
        'inputs','targets','c1','c2','loc'
    """
    
    data_subset_dict = {}
    data_subset_dict['inputs'] = data_dict['inputs'][:,trial_ixs,:]
    data_subset_dict['targets'] = data_dict['targets'][trial_ixs]
    data_subset_dict['c1'] = data_dict['c1'][trial_ixs]
    data_subset_dict['c2'] = data_dict['c2'][trial_ixs]
    data_subset_dict['loc'] = data_dict['loc'][:,trial_ixs,:]
    
    return data_subset_dict


def generate_test_dataset(params, plot=True):
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
    
    if params['var_delays']:
        # testing datasets for variable delay condition
        # trained delay length = 5, to compare easily with the Buschman paradigm
        params = update_time_params(params,5)
        test_data = make_stimuli_vonMises(params,epoch='test')
        
        # out-of-range temporal generalisation
        params = update_time_params(params,np.max(params['delay_lengths'])+1)
        test_data_outRange = make_stimuli_vonMises(params,epoch='test')
        test_data_outRange['delay_length'] = np.max(params['delay_lengths'])+1
        test_data_outRange['params'] = params
        
        # in range
        params = update_time_params(params,int(np.mean(params['delay_lengths'])))
        print('Warning - in-range temporal generalisation takes the mean of the training delay lengths as the test delay length.')
        test_data_inRange = make_stimuli_vonMises(params,epoch='test')
        test_data_inRange['delay_length'] = int(np.mean(params['delay_lengths']))
        test_data_inRange['params'] = params
        
        test_data = {'trained':test_data,
                      'in-range':test_data_inRange,
                      'out-of-range':test_data_outRange}

    # plot an example trial
    sns.set_context("notebook", font_scale=1.5)
    sns.set_style("ticks")
    if not params['var_delays']:
        cplot.plot_example_stimulus(test_data['inputs'].T,
                                    cmap = sns.cubehelix_palette(start=.5,rot=-.75,as_cmap = True,reverse=True))
        plt.xticks([0,5,10])
        plt.xlabel('time')
        plt.yticks(ticks=[-0.5,1.5,18.5,35.5],labels=[])
        fig = plt.gcf()
        fig.set_size_inches(3,5)
        fig_path = params['FULL_PATH'] + 'figs/'
        check_path(fig_path)
        plt.savefig(fig_path + 'example_test_stimulus.png')
    
    return test_data


def update_time_params(params,delay_length):
    
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
    
#%%

# plt.figure()

# ax1 = plt.subplot(211)

# ax1.plot(phi_degrees,test_dataset['targets'][colour_ix[10][2],:],'k-o')
# ax1.set_title('Target')
# ax1.set_ylim([0,1.1])

# # posterior mean
# t_probs = test_dataset['targets'][colour_ix[10][2],:] / torch.sum(test_dataset['targets'][colour_ix[10][2],:])

# post_mean = torch.sum(phi_degrees*t_probs)

# post_mean_y = np.linspace(0,1,100)
# post_mean_x = torch.ones(post_mean_y.shape) * post_mean

# ax1.plot(post_mean_x,post_mean_y,'r-','posterior mean')

# # true colour
# true_x = torch.ones(post_mean_y.shape) * np.degrees(colours[10])

# ax1.plot(true_x,post_mean_y,'b-','true colour')

#%%

# ax2 = plt.subplot(212)
# ix = 11

# ax2.plot(phi_degrees,test_dataset['targets'][colour_ix[ix][2],:],'k-o')
# ax2.set_title('Target')
# ax2.set_ylim([0,1.1])

# # posterior mean
# t_probs = test_dataset['targets'][colour_ix[ix][2],:] / torch.sum(test_dataset['targets'][colour_ix[ix][2],:])

# post_mean = torch.sum(phi_degrees*t_probs)

# post_mean_y = np.linspace(0,1,100)
# post_mean_x = torch.ones(post_mean_y.shape) * post_mean

# ax2.plot(post_mean_x,post_mean_y,'r-',label='posterior mean')

# # true colour
# true_x = torch.ones(post_mean_y.shape) * np.degrees(colours[ix])

# ax2.plot(true_x,post_mean_y,'b-',label='true colour')

# ax2.legend()

# #%%

# plt.figure()
# n = 4
# start = 0
# for i,ix in enumerate(np.arange(start,start+n)):
#     plt.subplot(n,1,i+1)
#     plt.plot(phi_degrees,test_dataset['targets'][colour_ix[ix][2],:],'k-o')
    
#     t_probs = test_dataset['targets'][colour_ix[ix][2],:] / torch.sum(test_dataset['targets'][colour_ix[ix][2],:])

#     # post_mean = torch.sum(phi_degrees*t_probs) - phi_degrees[0]*t_probs[0]
#     post_mean = np.degrees(posterior_mean(phi,t_probs))
    
#     post_mean_y = np.linspace(0,1,100)
#     post_mean_x = torch.ones(post_mean_y.shape) * post_mean
    
#     plt.plot(post_mean_x,post_mean_y,'r-',label='posterior mean')
    
#     # true colour
#     true_x = torch.ones(post_mean_y.shape) * np.degrees(colours[ix])
    
#     plt.plot(true_x,post_mean_y,'b--',label='true colour')
    
    
    
# plt.legend()
# #%%

# ax2 = plt.subplot(212)

# ax2.plot(phi_degrees,outputs[colour_ix[10][2],:],'k-o')
# ax2.plot(phi_degrees,outputs[colour_ix[10][3],:])
# ax2.plot(phi_degrees,outputs[colour_ix[10][4],:])
# ax2.plot(phi_degrees,outputs[colour_ix[10][5],:])
# ax2.plot(phi_degrees,outputs[colour_ix[10][6],:])
# ax2.plot(phi_degrees,outputs[colour_ix[10][7],:])

# ax2.set_title('Output')
# ax2.set_ylim([0,1])
# ax2.set_xlabel('Preferred Tuning direction [degrees])')
# ax2.set_ylabel('Activation value')



# # ax3 = plt.subplot(313)
# # ax3.plot(phi_degrees,outputs[colour_ix[10][1],:]/test_dataset['targets'][colour_ix[10][1]],'k-o')
# # ax3.set_title('O-T')
# # # ax3.set_ylim([0,1])
# # ax3.set_xlabel('Preferred Tuning direction [degrees])')
# # ax3.set_ylabel('Activation value')


# plt.tight_layout()



# #%%

# plt.figure()



