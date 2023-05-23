#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 13:35:52 2023

@author: emilia
"""
import torch
import numpy as np
import generate_data_vonMises as dg
import retrocue_model as retnet
import itertools
import matplotlib.pyplot as plt
import helpers

def test_shuffled_dataset(constants,load_path,device,m=0):
    '''
    Sanity check: does the order of trials in the test dataset affect model 
    output (or hidden activity, to be more precise)? It should not matter, as 
    models are evaluated after freezing the weights.

    Parameters
    ----------
    constants : TYPE
        DESCRIPTION.
    load_path : TYPE
        DESCRIPTION.
    device : TYPE
        DESCRIPTION.
    m : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    None.

    '''
    # paths
    model_path = constants.PARAMS['FULL_PATH']    
    load_path = model_path + 'saved_models/'
    eval_path = model_path+'pca_data/'
    # valid trials
    valid_path = eval_path + 'valid_trials/'
    invalid_path = eval_path + 'invalid_trials/'
    
    
    constants.PARAMS['model_number'] = m
    model = retnet.load_model(load_path,constants.PARAMS,device)
    
    
    
    test_data = dg.generate_test_dataset(constants.PARAMS)
    test_data_valid_trials = dg.subset_of_trials(constants.PARAMS,
                                                 test_data,
                                            test_data['valid_trial_ixs'])
    
    # invalid trials
    test_data_invalid_trials = dg.subset_of_trials(constants.PARAMS,
                                                   test_data,
                                            test_data['invalid_trial_ixs'])
    
    
    n_trials = len(test_data['valid_trial_ixs'])
    shuffling_ix = torch.from_numpy(np.random.permutation(np.arange(n_trials)))
    
    print('Warning - comment out the pca_uncued section in retnet.eval_model()\
          , otherwise the evaluation for shuffled trials will break')
    
    for trial_type, ordered_dataset in zip(['valid','invalid'],[test_data_valid_trials,test_data_invalid_trials]):
        print('{} trials'.format(trial_type))
        # create a shuffled dataset
        test_data_shuffled = {}
        for key in ordered_dataset.keys():
            if len(ordered_dataset[key].shape)==1:
                # entry is a vector
                test_data_shuffled[key] = ordered_dataset[key][shuffling_ix]
            elif len(ordered_dataset[key].shape)==3:
                # entry is a 3D array - 2nd dimebsion corresponds to trialls
                test_data_shuffled[key] = ordered_dataset[key][:,shuffling_ix,:]
        
        # check that the two datasets are identical when 'unshuffled'
       
        c1 = torch.all(test_data_shuffled['inputs']==ordered_dataset['inputs'][:,shuffling_ix,:])
        print('Datasets identical when ''unshuffled''' %c1)
        
        # evaluate model on both datasets
        
        eval_data_shuffled,_,_,_= \
        retnet.eval_model(model,test_data_shuffled,constants.PARAMS,valid_path,trial_type='valid')
        
        eval_data,_,_,_= \
        retnet.eval_model(model,ordered_dataset,constants.PARAMS,valid_path,trial_type='valid')
        
        # check that the model activation data is the same for corresponding trials
        c2 = torch.all(eval_data_shuffled['data']==eval_data['data'][shuffling_ix,:,:])
        print('Model activity identical for ordered and shuffled datasets' %c2)
    
        # if true, means that model output is insensitive to the order of trials 
        # in the dataset - and it shouldn't be at test
        

def test_trial_freqs(constants,load_path,device,m=0):
    '''
    Test if the valid and invalid datasets have equal trial frequencies/counts 
    in all respects. In other words, is the only difference between them that
    the cued and probed locations match / do not match?

    Parameters
    ----------
    constants : TYPE
        DESCRIPTION.
    load_path : TYPE
        DESCRIPTION.
    device : TYPE
        DESCRIPTION.
    m : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    None.

    '''
    # paths
    model_path = constants.PARAMS['FULL_PATH']    
    load_path = model_path + 'saved_models/'
    eval_path = model_path+'pca_data/'
    # valid trials
    valid_path = eval_path + 'valid_trials/'
    invalid_path = eval_path + 'invalid_trials/'
    
    test_data = dg.generate_test_dataset(constants.PARAMS)
    test_data_valid_trials = dg.subset_of_trials(constants.PARAMS,
                                                 test_data,
                                            test_data['valid_trial_ixs'])
    # invalid trials
    test_data_invalid_trials = dg.subset_of_trials(constants.PARAMS,
                                                   test_data,
                                            test_data['invalid_trial_ixs'])
    
    
    # step 1 - are there equal numbers of cued-l1 and cued-l2 trials for both datasets?
    cued_loc1_count = []
    cued_loc2_count = []
    for dataset in [test_data_valid_trials,test_data_invalid_trials]:
        cued_l1_ix = torch.where(dataset['cued_loc']==1)[0]
        cued_l2_ix = torch.where(dataset['cued_loc']==0)[0]
        
        cued_loc1_count.append(len(cued_l1_ix))
        cued_loc2_count.append(len(cued_l2_ix))
    test1 = cued_loc1_count == cued_loc2_count
    print('Are there equal numbers of cued-l1 and cued-l2 trials in both datasets? {}'.format(test1))

    
    # step 2 - test if all colours at location 1 (c1) and c2 are present in both datasets
    test2_1 = torch.all(torch.unique(test_data_valid_trials['c1'])==torch.unique(test_data_invalid_trials['c1']))
    test2_2 = torch.all(torch.unique(test_data_valid_trials['c2'])==torch.unique(test_data_invalid_trials['c2']))
    print('Are all c1 and c2 colours represented in both datasets? {}'.format(torch.logical_and(test2_1,test2_2)))
              
    
    # step 3 - test if the frequencies at which c1 and c2 are shown are equal (irrespective of the cued location)
    freqs_c1 = [[(test_data['c1'] == t).float().mean() for t in torch.unique(test_data['c1'])] for test_data in [test_data_valid_trials,test_data_invalid_trials]]
    freqs_c2 = [[(test_data['c2'] == t).float().mean() for t in torch.unique(test_data['c2'])] for test_data in [test_data_valid_trials,test_data_invalid_trials]]
    test3_1 = (freqs_c1[0]==freqs_c1[1])
    test3_2 = (freqs_c2[0]==freqs_c2[1])
    print('Are both c1 and c2 colours presented with equal frequencies, irrespective of the cued location, in both datasets? {}'.format(np.logical_and(test3_1,test3_2)))
    
    
    # step 4 - test if the colours c1 and c2 are presented with equal frequencies
    # for both cued locations
    
    cond_counts_c1 = []
    cond_counts_c2 = []
    # do it in a for loop for better readability
    for dataset in [test_data_valid_trials,test_data_invalid_trials]:
        cued_l1_ix = torch.where(dataset['cued_loc']==1)[0]
        cued_l2_ix = torch.where(dataset['cued_loc']==0)[0]
        
        cond_counts_c1.append([])
        cond_counts_c2.append([])
        
        for colour in torch.unique(test_data['c1']):
            cond_counts_c1[-1].append((dataset['c1'][cued_l1_ix]==colour).float().sum())
            cond_counts_c1[-1].append((dataset['c1'][cued_l2_ix]==colour).float().sum())
            cond_counts_c2[-1].append((dataset['c2'][cued_l1_ix]==colour).float().sum())
            cond_counts_c2[-1].append((dataset['c2'][cued_l2_ix]==colour).float().sum())
        
    test4 = cond_counts_c1==cond_counts_c2
    print('Are the conditional c1 and c2 colour counts (conditioned on cued location) the same for both datasets? {}'.format(test4))
    
    
    # step 5 - test if the pairwise colour combinations are presented with the 
    # same count irrespective of the cued location, for both datasets
    # I think if step 3 above checks out, this is implied, but can also test directly
    
    cond_trial_counts_loc1 = []
    cond_trial_counts_loc2 = []
    all_colour_combos = torch.tensor(list(itertools.product(torch.unique(test_data['c1']),
                                                            torch.unique(test_data['c2'])))) # all possible colour1-colour2 combinations
    
    # do it in a for loop for better readability
    for dataset in [test_data_valid_trials,test_data_invalid_trials]:
        cued_l1_ix = torch.where(dataset['cued_loc']==1)[0]
        cued_l2_ix = torch.where(dataset['cued_loc']==0)[0]
        
        cond_trial_counts_loc1.append([])
        cond_trial_counts_loc2.append([])
        
        
        for c1,c2 in all_colour_combos:
            combo_loc1_ix = np.intersect1d(torch.where(dataset['c1'][cued_l1_ix]==c1)[0],
                                            torch.where(dataset['c2'][cued_l1_ix]==c2)[0])
            combo_loc2_ix = np.intersect1d(torch.where(dataset['c1'][cued_l2_ix]==c1)[0],
                                            torch.where(dataset['c2'][cued_l2_ix]==c2)[0])
            
            cond_trial_counts_loc1[-1].append(len(combo_loc1_ix))
            cond_trial_counts_loc2[-1].append(len(combo_loc2_ix))
    test5 = cond_trial_counts_loc1==cond_trial_counts_loc2
    print('Are all trials, defined by the unique c1-c2 combination, presented \
          with equal counts irrespective of the cued location, \
              for both datasets? {}'.format(test5))
              
    # step 6 - check that for valid trials, all cued locations match probed locations
    # and for invalid, all cued match unprobed
    
    cue_timepoint = constants.PARAMS['trial_timepoints']['delay1_end']
    probe_timepoint = constants.PARAMS['trial_timepoints']['delay2_end']
    
    test6 = []
    for condition,dataset in zip(['valid','invalid'],[test_data_valid_trials,test_data_invalid_trials]):
        cued_l1_ix = torch.where(dataset['inputs'][cue_timepoint,:,0]==1)[0]
        cued_l2_ix = torch.where(dataset['inputs'][cue_timepoint,:,0]==0)[0]
        
        probed_l1_ix = torch.where(dataset['inputs'][probe_timepoint,:,0]==1)[0]
        probed_l2_ix = torch.where(dataset['inputs'][probe_timepoint,:,0]==0)[0]
        
        if condition == 'valid':
            # cues and probes match
            test6.append(torch.all(cued_l1_ix==probed_l1_ix))
            test6.append(torch.all(cued_l2_ix==probed_l2_ix))
        else:
            # cues and probes mismatch
            test6.append(torch.all(cued_l1_ix!=probed_l1_ix))
            test6.append(torch.all(cued_l2_ix!=probed_l2_ix))
    print('Do cues and probes match on all valid trials and mismatch on all invalid trials? {}'.format(np.all(test6)))
    

def test_trial_order(constants,load_path,device,m=0):
    # paths
    model_path = constants.PARAMS['FULL_PATH']    
    load_path = model_path + 'saved_models/'
    eval_path = model_path+'pca_data/'
    # valid trials
    valid_path = eval_path + 'valid_trials/'
    invalid_path = eval_path + 'invalid_trials/'
    
    
    constants.PARAMS['model_number'] = m
    model = retnet.load_model(load_path,constants.PARAMS,device)
    
    
    
    test_data = dg.generate_test_dataset(constants.PARAMS)
    test_data_valid_trials = dg.subset_of_trials(constants.PARAMS,
                                                 test_data,
                                            test_data['valid_trial_ixs'])
    
    # invalid trials
    test_data_invalid_trials = dg.subset_of_trials(constants.PARAMS,
                                                   test_data,
                                            test_data['invalid_trial_ixs'])
    
    cue_timepoint = constants.PARAMS['trial_timepoints']['delay1_end']
    probe_timepoint = constants.PARAMS['trial_timepoints']['delay2_end']
    # check how c1 and c2 trials are ordered in the valid and invalid datasets
    for trial_type, dataset in zip(['valid','invalid'],[test_data_valid_trials,test_data_invalid_trials]):
        plt.figure()
        plt.subplot(121)
        plt.imshow(np.repeat(dataset['c1'][:,None],1000,1))
        plt.title('C1 labels')
        plt.subplot(122)
        plt.imshow(np.repeat(dataset['c2'][:,None],1000,1))
        plt.title('C2 labels')
        
        plt.suptitle('{} trials'.format(trial_type))
    
    # check how cues and probes are ordered in the two datasets
    for trial_type, dataset in zip(['valid','invalid'],[test_data_valid_trials,test_data_invalid_trials]):
        plt.figure()
        plt.subplot(121)
        plt.imshow(np.repeat(dataset['inputs'][cue_timepoint,:,0][:,None],1000,1))
        plt.title('cued labels')
        plt.subplot(122)
        plt.imshow(np.repeat(dataset['inputs'][probe_timepoint,:,0][:,None],1000,1))
        plt.title('probed labels')
        
        plt.suptitle('{} trials'.format(trial_type))
    
    # resort by uncued
    for trial_type, dataset in zip(['valid','invalid'],[test_data_valid_trials,test_data_invalid_trials]):
        # add the required entries to the dataset dict so that it can be passed
        # into the sort_by_uncued func
        dataset['data'] = torch.permute(dataset['inputs'],[1,0,-1])
        dataset['labels'] = {'c1':dataset['c1'],'c2':dataset['c2']}
        data_struct_sorted, full_sorting_ix = helpers.sort_by_uncued(dataset,constants.PARAMS,key='data')
        
        plt.figure()
        plt.subplot(121)
        plt.imshow(np.repeat(dataset['c1'][full_sorting_ix,None],1000,1))
        plt.title('C1 labels')
        plt.subplot(122)
        plt.imshow(np.repeat(dataset['c2'][full_sorting_ix,None],1000,1))
        plt.title('C2 labels')
        
        plt.suptitle('{} trials passed by the sort_by_uncued func'.format(trial_type))
    
    return


def check_hidden_activity(constants,load_path,device,m=0):
    # nb this script requires modification of retnet.eval_model - comment out 
    # everythhing after eval_data and chabge the function output to return only that
    # paths
    model_path = constants.PARAMS['FULL_PATH']    
    load_path = model_path + 'saved_models/'
    eval_path = model_path+'pca_data/'
    # valid trials
    valid_path = eval_path + 'valid_trials/'
    invalid_path = eval_path + 'invalid_trials/'
    
    
    constants.PARAMS['model_number'] = m
    model = retnet.load_model(load_path,constants.PARAMS,device)
    
    
    
    test_data = dg.generate_test_dataset(constants.PARAMS)
    test_data_valid_trials = dg.subset_of_trials(constants.PARAMS,
                                                 test_data,
                                            test_data['valid_trial_ixs'])
    
    # invalid trials
    test_data_invalid_trials = dg.subset_of_trials(constants.PARAMS,
                                                   test_data,
                                            test_data['invalid_trial_ixs'])
    
    
    test_data_single_trial_type = {}
    trial_ix = 51 # index of the trial type to be compared, taken from the valid dataset
    n_trials = len(test_data_valid_trials['targets'])
    offset_ix = torch.where(test_data_invalid_trials['c2'][12800:]==test_data_valid_trials['c2'][trial_ix])[0][0]
    for key in test_data_valid_trials.keys():
        
        if len(test_data_valid_trials[key].shape)==3:
            test_data_single_trial_type[key] = \
                torch.cat((test_data_valid_trials[key][:,trial_ix,:][:,None],
                           test_data_invalid_trials[key][:,n_trials//2 +offset_ix,:][:,None]),dim=1)
        elif len(test_data_valid_trials[key].shape)==1:
            test_data_single_trial_type[key] = \
                torch.stack((test_data_valid_trials[key][trial_ix],
                           test_data_invalid_trials[key][n_trials//2+offset_ix]))
                
    eval_data = \
    retnet.eval_model(model,test_data_single_trial_type,constants.PARAMS,invalid_path,trial_type='invalid')
    
    delay2_end = constants.PARAMS['trial_timepoints']['delay2_end']
    torch.all(eval_data['data'][0,:delay2_end,:] == eval_data['data'][1,:delay2_end,:])
    
    # visualise activity post-probe
    plt.figure()
    plt.subplot(121)
    plt.imshow(eval_data['data'][0,delay2_end-2:,:].T)
    plt.title('valid trial')
    plt.subplot(122)
    plt.imshow(eval_data['data'][1,delay2_end-2:,:].T)
    plt.title('invalid trial')
    
    
    # plot difference in activity on valid - invalid
    plt.figure()
    plt.imshow(eval_data['data'][0,delay2_end-2:,:].T - eval_data['data'][1,delay2_end-2:,:].T,
               cmap='coolwarm')
    plt.colorbar()
    plt.title('Diff in activity valid - invalid trial')
    
    
    # check whether the sd of the population activity differs between valid and
    # invalid trials from after the probe
    
    test_data_same_trial_types = {}
    trial_ixs = np.arange(0,12800,800) # index of the trial type to be compared, taken from the valid dataset
    n_trials = len(test_data_valid_trials['targets'])
    trial_ixs_invalid = \
        [np.intersect1d(torch.where(test_data_invalid_trials['c2'][12800:]==test_data_valid_trials['c2'][i])[0],
                    torch.where(test_data_invalid_trials['c1'][12800:]==test_data_valid_trials['c1'][i])[0])[0]+n_trials//2 for i in trial_ixs]
    
    for key in test_data_valid_trials.keys():
        
        if len(test_data_valid_trials[key].shape)==3:
            test_data_same_trial_types[key] = \
                torch.cat((test_data_valid_trials[key][:,trial_ixs,:],
                           test_data_invalid_trials[key][:,trial_ixs_invalid,:]),dim=1)
        elif len(test_data_valid_trials[key].shape)==1:
            test_data_same_trial_types[key] = \
                torch.stack((test_data_valid_trials[key][trial_ixs],
                           test_data_invalid_trials[key][trial_ixs_invalid]))
                
    eval_data = \
    retnet.eval_model(model,test_data_same_trial_types,constants.PARAMS,invalid_path,trial_type='invalid')
    
    
    sd_post_probe_valid = eval_data['data'][:len(trial_ixs),:,:].std(dim=-1)
    sd_post_probe_invalid = eval_data['data'][len(trial_ixs):,:,:].std(dim=-1)
    
    plt.figure()
    plt.plot(sd_post_probe_valid.mean(0),'go-')
    plt.plot(sd_post_probe_invalid.mean(0),'ro-')
    plt.xticks([0,3,6,9,12,15],['stim','delay1','cue','delay2','probe','delay3'],rotation=45)
    plt.ylabel('mean std across neurons')
    return
    
    
