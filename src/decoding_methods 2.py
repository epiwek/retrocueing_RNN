#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 17:58:34 2021

@author: emilia

This file contains all decoding analysis functions, including:
    1) decoding of uncued stimulus colour in the post-cue delay
    2) cross-temporal decoding
    3) comparison of  maintenance mechanisms between models from expts 1 & 2
    4) single-readout hypothesis : cross-decoding of cued colours across two parallel planes
    5) analogue to the CDI analysis: compare the discriminability of colour 
        representations pre-cue, after they are cued and uncued
    6) analogue to the rotated/unrotated plane AI analysis: compare the 
        cross-temporal generalisation scores (between the precue and postcue 
        delays) between the two cued planes

1) This analysis asks if there is still information about the uncued item 
    colour in the post-cue delay.
2) This analysis computes the cross-temporal decoding accuracy scores for the
    cued items across the entire trial duration.
3) This analysis calculates the mean delay cross-temporal generalisation scores and 
    compares those between expts 1 & 2 to assess whether the networks trained 
    with variable delay lengths (expt 2) form a more temporally stable working 
    memory code than those trained with fixed delays (expt 1).
4) This analysis asks if a common linear readout can be used to extract the cued
    colour information across both the cue-1 and cue-2 trials.
5) This analysis seeks to confirm the conclusion from Fig. 3H whilst taking 
    into account the noise in the representatioons. Linear decoders are trained
    in cross-validation to discriminate between colours in the pre-cue delay, 
    as well as after they are cued or uncued, and the test scores compared 
    between the 3 conditions.
5) This analysis seeks to confirm the conclusion from Fig. 2H, that is only one
    of the pre-cue planes being rotated to form the parallel plane geometry in 
    the post-cue delay. Like for the corresponding AI analysis, this analysis
    is done in 2-fold cross-validation. Cross-temporal generalisation decoders 
    are fitted to haf the data, to finf the putative 'rotated' and 'unrotated' 
    planes. Then, the analysis is repeated on the second half of the data, and 
    scores for the 'rotated' and 'unrotated' planes saved. If all models 
    consistently rotate noy one of the planes, then the 'unrotated' plane scores
    shold be significantly higher 1) than chance (50%) and 2) than the rotated 
    scores.
    

"""
import pickle
import numpy as np
import pandas as pd
import time

import seaborn as sns
import matplotlib.pyplot as plt

import helpers
import constants_expt1 as c
import generate_data_vonMises as dg

from itertools import combinations
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mne.decoding import GeneralizingEstimator

from scipy.stats import norm

from stats import run_contrast_single_sample,run_contrast_unpaired_samples
import rep_geom_analysis as rg
from get_subspace_alignment_index import get_trial_ixs, plot_AI


#%% common low-level functions
def get_class_pairs(y):
    '''
    Gets all possible class pair combinations.

    Parameters
    ----------
    y : array (n_trials,)
        Vector of trial-wise class labels.

    Returns
    -------
    combos : list
        All possible binary class combinations.

    '''
    classes = np.unique(y)
    combos = list(combinations(classes,2))
    return combos

# import pdb
# from sklearn.model_selection import StratifiedKFold, KFold
def lda(X,y, cv=2, n_repeats=1):
    '''
    Fits binary LDA classifiers to discriminate between labels in cross-validation.

    Parameters
    ----------
    X : array (n_samples,n_features)
        Data matrix.
    y : array (n_samples,)
        Trial-wise class labells.
    cv : int, optional
        Number of cross-validation folds. The default is 2.

    Returns
    -------
    scores_test : array (n_classifiers,)
        Test decoding accuracy for each LDA classifier.

    '''
    # get all possible class pairs    
    class_combos= get_class_pairs(y)
    scores_test = np.zeros((len(class_combos),n_repeats))
    # pdb.set_trace()
    for i in range(len(class_combos)):
        y1,y2 = class_combos[i] # class labels
        
        # find trials of the abovespecified classes
        trial_ix = np.where(np.logical_or(y==y1,y==y2))[0]
        
        
        # make classifier pipeline
        clf = make_pipeline(StandardScaler(), LDA())
        
        # fit a classifier in cross-val
        results = cross_validate(clf,X[trial_ix,:],y[trial_ix],
                                 cv=cv,
                                 return_estimator=False)
        
        # skf = StratifiedKFold(n_splits=2,shuffle=False)
        # # skf.get_n_splits(X[trial_ix,:],y[trial_ix])
        # for i, (train_index, test_index) in enumerate(skf.split(X[trial_ix,:],y[trial_ix])):
        #     print(f"Fold {i}:")
        #     print(f"  Train: index={train_index}")
        #     print(f"  Test:  index={test_index}")
            
        #     clf = make_pipeline(StandardScaler(), LDA())
        #     # fit classifier
        #     clf.fit(X[trial_ix,:][train_index,:],y[trial_ix][train_index])
        #     # test classifier
        #     clf.score(X[trial_ix,:][test_index,:],y[trial_ix][test_index])
        
        # test_class_freqs(y,y_unbinned,y_undecoded,train_index,test_index)
        # print(results['test_score'])
        # average test scores across cv folds
        scores_test[i] = results['test_score'].mean()
    
    return scores_test     


def test_class_freqs(y_decoded_binned,y_decoded_unbinned,y_undecoded,train_index,test_index):
    # count class freqs - should be 50/50
    
    c1 = y_decoded_binned[train_index].mean() == np.unique(y_decoded_binned[train_index]).mean()
    c2 = y_decoded_binned[test_index].mean() == np.unique(y_decoded_binned[test_index]).mean()
    print(f'are the class freqs equal for both train and test trials? {c1,c2}')
    
    # subclasses
    counts1,counts2 = [],[]
    for i in np.unique(y_decoded_unbinned):
        counts1.append(len(np.where(y_decoded_unbinned[train_index]==i)[0]))
        counts2.append(len(np.where(y_decoded_unbinned[test_index]==i)[0]))
    print('Are the subclass counts equal for both train and test trials?')
    print(f'    Train={counts1}')
    print(f'    Test={counts2}')
    
    # undecoded colours
    counts1,counts2 = [],[]
    for i in np.unique(y_undecoded):
        counts1.append(len(np.where(y_undecoded[train_index]==i)[0]))
        counts2.append(len(np.where(y_undecoded[test_index]==i)[0]))    
    print('Are the undecoded class counts equal for both train and test trials?')
    print(f'    Train={counts1}')
    print(f'    Test={counts2}')

# subclass1 = np.unique(y_unbinned[trial_ix])[3]
# subclass1_ix = np.where(y_unbinned[trial_ix]==subclass1)[0]

# train_index = np.setdiff1d(train_index,subclass1_ix)


def lda_cg(X1,y1,X2,y2,cv=2):
    '''
    Fits binary LDA classifiers to discriminate between labels to one dataset
    and tests performance on 1) a held-out portion of the same dataset and 
    2) another dataset (cross-generalisation performance).

    Parameters
    ----------
    X1 : array (n_samples,n_features)
        Data matrix for dataset 1.
    y1 : array (n_samples,)
        Trial-wise class labels for dataset 1.
    X2 : array (n_samples,n_features)
        Data matrix for dataset 2.
    y2 : array (n_samples,)
        Trial-wise class labels for dataset 2.
    cv : int, optional
        Number of cross-validation folds. The default is 2.

    Returns
    -------
    scores_test : array (n_classifiers,2)
        Test decoding accuracy for each LDA classifier.

    '''
    # get all possible class pair combinations    
    class_combos = get_class_pairs(y1)
    scores_test = np.zeros((len(class_combos),2)) # (n_classifiers,n_datasets)
    scores_cg = np.zeros((len(class_combos),2))
    for i in range(len(class_combos)):
        l1,l2 = class_combos[i] # pair of class labels
        
        # find trials of the above-specified classes
        trial_ix_y1 = np.where(np.logical_or(y1==l1,y1==l2))[0]
        trial_ix_y2 = np.where(np.logical_or(y2==l1,y2==l2))[0]
        
        # make classifier pipelines
        clf1 = make_pipeline(StandardScaler(), LDA())
        clf2 = make_pipeline(StandardScaler(), LDA())
        
        # fit classifiers in cross-val
        results1 = cross_validate(clf1,
                                 X1[trial_ix_y1,:],
                                 y1[trial_ix_y1],
                                 cv=cv,
                                 return_estimator=True)
        results2 = cross_validate(clf2,
                                 X2[trial_ix_y2,:],
                                 y2[trial_ix_y2],
                                 cv=cv,
                                 return_estimator=True)
        
        # average test scores across cv folds
        scores_test[i,0] = results1['test_score'].mean()
        scores_test[i,1] = results2['test_score'].mean()
        
        # calculate cross-generalisation performance
        scores_cg[i,0] = np.mean([results1['estimator'][i].score(X2[trial_ix_y2,:],y2[trial_ix_y2]) for i in range(cv)])
        scores_cg[i,1] = np.mean([results2['estimator'][i].score(X1[trial_ix_y1,:],y1[trial_ix_y1]) for i in range(cv)])
    
    return scores_test.mean(-1), scores_cg.mean(-1)


def lda_cg_time(X,y):
    '''
    Test LDA classifiers to discriminate between pairs of classes based on data
    from a single timepoint and test their performance on all the other 
    timepoints.

    Parameters
    ----------
    X : array (n_trials,n_neurons,n_timepoints)
        Data matrix.
    y : array (n_trials,)
        Labels vector.

    Returns
    -------
    test_scores: array (n_timepoints,n_timepoints)
        Cross-temporal generalisation scores.
    clfs : dict
        Fitted binary LDA classifiers.

    '''
    # split data and labels into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2,random_state=0)
    time = X.shape[-1]
    # get all possible class1-class2 combinations for binary discriminations
    class_combos= get_class_pairs(y)
    clfs = {} # classifiers
    scores_Xtest = np.empty((len(class_combos),time,time))
    # loop over class pairs
    for i in range(len(class_combos)):
        y1,y2 = class_combos[i] # class labels
        
        # find trials of above-specified classes
        train_ix = np.where(np.logical_or(y_train==y1,y_train==y2))[0]
        test_ix = np.where(np.logical_or(y_test==y1,y_test==y2))[0]
        
        # make classifier pipeline
        clf = make_pipeline(StandardScaler(), LDA())
        time_gen = GeneralizingEstimator(clf)
    
        # fit the classifier
        clfs[str(i)] = time_gen.fit(X=X_train[train_ix,:,:],y=y_train[train_ix])
        
        # test performance on with-held data, all timepoints
        scores_Xtest[i,:,:] = time_gen.score(X=X_test[test_ix,:,:],y=y_test[test_ix])
        
    return np.mean(scores_Xtest,0), clfs
#%% 1) decode uncued in post-cue - is the information still there?

def get_decoding_within_plane(constants,which_delay,trial_type='valid'):
    '''
    Train and test LDA binary classifiers to discriminate between pairs of 
     colour labels in the pre or post-cue delay (at the last time-point). For 
     the post-cue and post-probe delays, decoding is done for the uncued 
     colours only.

    Parameters
    ----------
    constants : dict
        Dictionary containing the constants for the experiment.
    which_delay : str
        Name of the memory delay to extract the data from. Pass 'precue', 
        'postcue' or 'postprobe'.
    trial_type : str, optional
        Optional. Relevant for the probabilistic paradigm (experiment 3). Pass 
        'valid' or 'invalid'. The default is 'valid'.

    Returns
    -------
    model_scores : array
        Average test decoding acuracy scores for all models.

    '''
    load_path = constants.PARAMS['FULL_PATH']+'pca_data/' + trial_type + '_trials/'
    model_scores = np.empty((constants.PARAMS['n_models'],))
    
    if which_delay == 'precue':
        delay_ix = constants.PARAMS['trial_timepoints']['delay1_end']-1
        fname_str = 'precue'
    elif which_delay == 'postcue':
        # post-cue delay (uncued colours)
        delay_ix = constants.PARAMS['trial_timepoints']['delay2_end']-1
        fname_str = 'uncued_postcue'
    elif which_delay == 'postprobe':
        delay_ix = constants.PARAMS['trial_timepoints']['delay3_end']-1
        fname_str = 'unprobed_postprobe'
        
    for model in range(constants.PARAMS['n_models']):
        # load data
        model_number = str(model)
        print('Model '+ model_number +'/' + str(constants.PARAMS['n_models']))
        f = open(load_path+'/eval_data_model' + model_number + '.pckl', 'rb')
        eval_data = pickle.load(f)    
        f.close()
        
        n_trials = eval_data['data'].shape[0]
        
        # get the uncued colour labels
        labels_uncued = np.concatenate((eval_data["labels"]["c2"][:n_trials//2],
                                  eval_data["labels"]["c1"][n_trials//2:]))
        # bin the labels into B colour bins
        labels_uncued_binned = helpers.bin_labels(labels_uncued,constants.PARAMS['B'])
        # split the labels according to the cued location
        loc1_labels = labels_uncued_binned[:n_trials//2]
        loc2_labels = labels_uncued_binned[n_trials//2:]
        
        delayloc1 = eval_data['data'][:n_trials//2,delay_ix,:]
        delayloc2 = eval_data['data'][n_trials//2:,delay_ix,:]
        
        # shuffle trials
        rng = np.random.default_rng(seed=model)
        
        trial_order1 = rng.permutation(n_trials//2)
        trial_order2 = rng.permutation(n_trials//2)
        
        delayloc1 = delayloc1[trial_order1,:]
        delayloc2 = delayloc2[trial_order2,:]
        loc1_labels = loc1_labels[trial_order1]
        loc2_labels = loc2_labels[trial_order2]
        
        # train and test LDA classifiers
        scores_loc1 = lda(delayloc1,loc1_labels)
        scores_loc2 = lda(delayloc2,loc2_labels)
        
        # save LDA model test scores
        model_scores[model] = np.stack((scores_loc1,scores_loc2)).mean()
    # save to file 
    pickle.dump(model_scores,open(load_path+'/decoding_acc_'+fname_str+'_delay.pckl','wb'))
    return model_scores


def run_decoding_uncued_analysis(constants):
    '''
    Runs the entire uncued colour decoding analysis. LDA classifiers trained 
    and tested in the post-cue delay.

    Parameters
    ----------
    constants : dict
        Dictionary containing the constants for the experiment.

    Returns
    -------
    None.

    '''
    print('Run the uncued colour decoding analysis')
    # get decoding test scores
    model_scores = get_decoding_within_plane(constants,which_delay='postcue')
    
    # run constrast - test against chance
    print('...Run contrast: mean test decoding significantly greater than chance (0.5) ')
    run_contrast_single_sample(model_scores, 0.5)
    
    print('...Mean decoding accuracy: %.4f' %model_scores.mean())

#%% 2: cross-temporal decoding

def fit_and_plot_ctg(constants,tmin,tmax,custom_path=[]):
    '''
    Trains binary LDA classifiers to discriminate between cued colours based 
    on the data from one timepoint and tests their performance on all the other
    timepoints. Plots the ctg matrix averaged across all models, with trial
    even boundaries demarcated by black horizontal and vertical lines.

    Parameters
    ----------
    constants : dict
        Dictionary containing the constants for the experiment.
    tmin : int
        Lower bound of the time interval to be used by the ctg classifier.
    tmax : int
        Upper bound of the time interval to be used by the ctg classifier.

    Returns
    -------
    scores : array (n_timepoints,n_timepoints,n_modes)
        Cross-temporal generalisation scores for all timepoints and models.
    scores_grand_av : array (n_timepoints,n_timepoints)
        Cross-temporal generalisation scores averaged across all models.

    '''
    if len(custom_path) != 0:
        load_path = constants.PARAMS['FULL_PATH'] + 'pca_data/valid_trials' + custom_path
    else:
        load_path = constants.PARAMS['FULL_PATH'] + 'pca_data/valid_trials'
    n_models = constants.PARAMS['n_models']

    delta_t = tmax-tmin
    scores = np.empty((delta_t,delta_t,3,n_models)) #(time,time,[loc1,loc2,average],model)
    # pdb.set_trace()
    start_time = time.time()
    
    n_colours = constants.PARAMS['B']
    # FIT
    for model in range(n_models):
        
        print('Model %d' %model)
        model_number = str(model)
        
        # load eval data
        f = open(load_path+'/eval_data_model' + model_number + '.pckl', 'rb')
        obj = pickle.load(f)
        data = obj["data"]
        f.close()
        
        data = data.permute(0,-1,1) # trial x n_rec x time
        n_trials = data.shape[0]
        
        labels_cued = np.concatenate((obj['labels']['c1'][:n_trials//2],
                                 obj['labels']['c2'][n_trials//2:]),0)
        
        labels_binned = helpers.bin_labels(labels_cued,n_colours)
        
        X1 = data[:n_trials//2,:,tmin:tmax]
        y1 = labels_binned[:n_trials//2] # loc 1 cued
        
        X2 = data[n_trials//2:,:,tmin:tmax]
        y2 = labels_binned[n_trials//2:] # loc 2 cued
        
        # shuffle trials
        rng = np.random.default_rng(seed=model)
        
        trial_order1 = rng.permutation(n_trials//2)
        trial_order2 = rng.permutation(n_trials//2)
        
        X1 = X1[trial_order1,:,:]
        y1 = y1[trial_order1]
        X2 = X2[trial_order2,:,:]
        y2 = y2[trial_order2]
        
        scores[:,:,0,model],clf = lda_cg_time(X1,y1)
        scores[:,:,1,model],clf = lda_cg_time(X2,y2)
        scores[:,:,2,model] = np.mean(scores[:,:,:2,model],2)
        
        
    
    # save scores
    scores_struct = {'scores':scores,
                     'dims':['training_time','testing_time','loc1loc2av','model']}
    pickle.dump(scores_struct,open(load_path+'/ctg_scores.pckl','wb'))

    # PLOT grand average

    scores_grand_av = np.mean(scores[:,:,2,:],-1)
    
    
    plt.figure()
    plt.imshow(scores_grand_av,origin='lower',cmap='RdBu_r',vmin=0, vmax=1.)
    plt.colorbar()
    
    plt.axhline(.5, color='k')
    plt.axvline(.5, color='k')
    
    if delta_t == constants.PARAMS['seq_len']:
        # all timepoints
        
        plt.axhline(constants.PARAMS['trial_timepoints']['delay1_end']-.5, color='k')
        plt.axvline(constants.PARAMS['trial_timepoints']['delay1_end']-.5, color='k')
        
        plt.axhline(constants.PARAMS['trial_timepoints']['delay2_start']-.5, color='k')
        plt.axvline(constants.PARAMS['trial_timepoints']['delay2_start']-.5, color='k')
        
        # plt.xticks(range(delta_t),labels=np.arange(delta_t)+1)
        # plt.yticks(range(delta_t),labels=np.arange(delta_t)+1)
        plt.xticks(np.arange(0,delta_t,5),labels=np.arange(0,delta_t,5)+1)
        plt.yticks(np.arange(0,delta_t,5),labels=np.arange(0,delta_t,5)+1)
        
    else:
        plt.xticks(range(delta_t), labels=range(-1,delta_t-1))
        plt.yticks(range(delta_t), labels=range(-1,delta_t-1))
        
        
    plt.ylabel('Training time')
    plt.xlabel('Testing time')
    
    # plt.title('Test accuracy')
    plt.tight_layout()

    end_time = time.time()
    total_time = (end_time - start_time)/60
    
    print('Time elapsed: %.2f mins' %total_time)
    return scores, scores_grand_av


def run_ctg_analysis(constants):
    '''
    Runs the full cross-temporal decoding analysis pipeline. Binary LDA 
    classifiers are trained to discriminate between cued stimulus labels 
    throughout the entire trial length. Saves the data into fie and plots the 
    cross-temporal decoding matrix, averaged across all models.
    
    Parameters
    ----------
    constants : dict
        Dictionary containing the constants for the experiment.
        
    Returns
    -------
    None.

    '''
    tmin=0
    # if experiment 2, update the length of delay intervals saved in constants 
    # to 7 cycles (same as for expt1)
    if constants.PARAMS['experiment_number']==2:
        dg.update_time_params(constants.PARAMS,7)
    tmax=constants.PARAMS['trial_timepoints']['delay2_end']
    fit_and_plot_ctg(constants,tmin,tmax)
    plt.savefig(constants.PARAMS['FIG_PATH']+'cross_temp_decoding_alltimepoints.png')
    plt.savefig(constants.PARAMS['FIG_PATH']+'cross_temp_decoding_alltimepoints.svg')


def run_ctg_analysis_new_delays(constants):
    
    '''
    Runs the full cross-temporal decoding analysis pipeline. Binary LDA 
    classifiers are trained to discriminate between cued stimulus labels 
    throughout the entire trial length. Saves the data into fie and plots the 
    cross-temporal decoding matrix, averaged across all models.
    
    Parameters
    ----------
    constants : dict
        Dictionary containing the constants for the experiment.
        
    Returns
    -------
    None.

    '''
    
    tmin=0
    # if experiment 2, update the length of delay intervals saved in constants 
    # first to the within-training range delay (4 cycles) , then to the 
    # out-of-training range delay (10 cycles)
    for delay_length,custom_path in zip([4,10],['in_range_tempGen','out_range_tempGen']):
        dg.update_time_params(constants.PARAMS,delay_length)
        tmax=constants.PARAMS['trial_timepoints']['delay2_end']
        
        fit_and_plot_ctg(constants,tmin,tmax,custom_path='/'+custom_path)
        plt.title(custom_path)
        plt.savefig(constants.PARAMS['FIG_PATH']+'cross_temp_decoding_alltimepoints_'+custom_path+'.png')
        plt.savefig(constants.PARAMS['FIG_PATH']+'cross_temp_decoding_alltimepoints_'+custom_path+'.svg')

#%% 3) compare maintenance mechanisms between expts 1 & 2

def get_delay_timepoints():
    d1_start = c.PARAMS['trial_timepoints']['delay1_start']
    d1_end = c.PARAMS['trial_timepoints']['delay1_end']
    d2_start = c.PARAMS['trial_timepoints']['delay2_start']
    d2_end = c.PARAMS['trial_timepoints']['delay2_end']


    # get the indices corresponding to the delay 1 and 2 timepoints in the ctg 
    # matrix (only the off-diagonal elements)
    # delay1
    d1_x,d1_y = np.concatenate((np.triu_indices(c.PARAMS['trial_timings']['delay1_dur'],k=1),
                                np.tril_indices(c.PARAMS['trial_timings']['delay1_dur'],k=-1)),1)
    d1_x += d1_start
    d1_y += d1_start
    
    # delay2
    d2_x,d2_y = np.concatenate((np.triu_indices(c.PARAMS['trial_timings']['delay2_dur'],k=1),
                                np.tril_indices(c.PARAMS['trial_timings']['delay2_dur'],k=-1)),1)
    d2_x += d2_start
    d2_y += d2_start

    # concatenate
    d_x = np.concatenate((d1_x,d2_x))
    d_y = np.concatenate((d1_y,d2_y))
    # get the indices of the diagonal elements
    diag_ix = np.concatenate((np.arange(d1_start,d1_end),np.arange(d2_start,d2_end)))
    
    return d_x,d_y,diag_ix


def get_mean_delay_scores():
    # get the paths to expt 1 (standard model) and 2 (variable delays model) 
    # datafiles
    standard_model_path = c.PARAMS['BASE_PATH'] +\
        'data_vonMises/experiment_1/' +\
            'sigma0.07/kappa5.0/nrec200/lr0.0001/'
    vardelay_model_path = c.PARAMS['BASE_PATH'] +\
        'data_vonMises/experiment_2/' +\
            'sigma0.0/kappa5.0/nrec200/lr0.0001/'
    
    # get the indices corresponding to the delay timepoints in the ctg matrix  
    d_x, d_y, diag_ix = get_delay_timepoints()
    
    # calculate the mean delay scores for diagonal and off-diagonal elements
    off_diag_scores = np.empty((c.PARAMS['n_models'],2)) # model, condition
    diag_scores = np.empty((c.PARAMS['n_models'],2)) # model, condition
    for i,condition in enumerate([vardelay_model_path,standard_model_path]):
        scores_struct = pickle.load(open(condition+'pca_data/valid_trials/ctg_scores.pckl','rb'))   
        
        # get the mean off- and diagonal scores, averaged across both delays
        diag_scores[:,i] = np.diagonal(scores_struct['scores'][:,:,-1,:])[:,diag_ix].mean(-1)
        off_diag_scores[:,i] = scores_struct['scores'][d_x,d_y,-1,:].mean(0)
        

    return diag_scores,off_diag_scores


def run_all_contrasts(off_diag_scores,diag_scores):
    '''
    Run all 3 contrasts for the analysis. Contrasts 1 and 2 test if the mean 
    off-diagonal decoding scores are significantly higher than the chance 
    decoding level (50%). Contrast 3 tests whether the mean ratio between the 
    off- and diagonal elements is significantly higher in the variable than 
    fixed delay condition. The ratio is used as an index of the temporal 
    stabilility of the code - for a perfectly temporally stable code, it 
    should be ~1.

    Parameters
    ----------
    off_diag_scores : array (n_models,n_conditions)
        Mean off-diagonal (cross-temporal) decoding scores for individual models.
        Values from the variable delay condition in the first, fixed - in the 
        second column.
    diag_scores : array (n_models,n_conditions)
        Mean diagonal decoding scores for individual models, in the same format
        as the off-diag_scores.

    Returns
    -------
    None.

    '''
    # contrast 1: test variabe delays off-diagonal mean against chance (0.5)
    print('...Contrast 1: Variable delays mean ctg decoding > chance')
    run_contrast_single_sample(off_diag_scores[:,0],h_mean=[.5],alt='greater')
    print('... mean = %.2f' % off_diag_scores[:,0].mean())
    # contrast 2: test fixed delays off-diagonal mean against chance (0.5)
    print('...Contrast 2: Fixed delays mean ctg decoding > chance')
    run_contrast_single_sample(off_diag_scores[:,1],h_mean=[.5],alt='greater')
    print('... mean = %.2f' % off_diag_scores[:,1].mean())
    # contrast 3: test if mean off-/diagonal ratio for variable delays > fixed delays
    print('...Contrast 3: Variable delays mean ratio off-/diagonal decoding > fixed delays')
    run_contrast_unpaired_samples(off_diag_scores[:,0]/diag_scores[:,0],
                                off_diag_scores[:,1]/diag_scores[:,1],
                                alt='greater')
    return


def plot_off_diagonal(off_diag_scores):
    '''
    Creates a boxplot of the mean off-diagonal decoding scores for the variable
    (expt 2) and fixed delay (expt 1) conditions. Chance level plotted as 
    dashed line.

    Parameters
    ----------
    off_diag_scores : array (n_models,n_conditions)
        Mean off-diagonal (cross-temporal) decoding scores for individual models.
        Values from the variable delay condition in the first, fixed - in the 
        second column.

    Returns
    -------
    None.

    '''
    # reformat data into a pandas dataframe for seaborn
    labels = np.array([['variable']*c.PARAMS['n_models'],['fixed']*c.PARAMS['n_models']]).reshape(-1)
    tbl = pd.DataFrame(np.stack((off_diag_scores.reshape(-1,order='F'),
                                 labels),1),
                       columns=['mean delay score','condition'])
    tbl['mean delay score'] = tbl['mean delay score'].astype(float)
    
    plt.figure(figsize=(5.5,5))
    sns.boxplot(data=tbl, x='condition',y='mean delay score',
                palette=[sns.color_palette("Set2")[i] for i in [0,-2]])
                # palette="Set2")
    plt.plot(plt.xlim(),[.5,.5],'k--')
    plt.ylim([.4,.85])
    plt.tight_layout()
    
    
def run_maintenance_mechanism_analysis():
    '''
    Runs the entire maintenance mechanism analysis. Calculates the mean delay 
    cross-temporal generalisation scores and compares those between expts 1 & 2
    to assess whether the networks trained with variable delay lengths (expt 2)
    form a more temporally stable working memory code than those trained with 
    fixed delays (expt 1).

    Returns
    -------
    None.

    '''
    print('Comparing the memory maintenance mechanisms between expts 1 & 2')
    # calculate the off- and on-diagonal decoding scores
    diag_scores,off_diag_scores = get_mean_delay_scores()

    # run the statistical tests
    run_all_contrasts(off_diag_scores,diag_scores)
    
    # boxplot of the off-diagonal decoding accuracy
    plot_off_diagonal(off_diag_scores)
    return


#%% 4) single-readout hypothesis 

def get_cg_decoding_cued(constants,which_delay='postcue',trial_type='valid'):
    '''
    

    Parameters
    ----------
    constants : TYPE
        DESCRIPTION.
    which_delay : str, optional
        Name of the memory delay to extract the data from. Pass 'precue', 
        'postcue' or 'postprobe'. Default is 'postcue'.
    trial_type : TYPE, optional
        DESCRIPTION. The default is 'valid'.

    Returns
    -------
    model_scores_test : TYPE
        DESCRIPTION.
    model_scores_cg : TYPE
        DESCRIPTION.

    '''
    '''
    Train  LDA binary classifiers to discriminate between pairs of 
    cued colour labels on cue-1 trials and test their x-generalisation 
    performance on cue-2 trials, and vice-versa. Data taken from the last 
    time-point of the post-cue delay.

    Parameters
    ----------
    constants : dict
        Dictionary containing the constants for the experiment.

    Returns
    -------
    model_scores : array
        Average test decoding acuracy scores for all models.

    '''
    load_path = constants.PARAMS['FULL_PATH']+'pca_data/'+trial_type+'_trials/'
    model_scores_test = np.empty((constants.PARAMS['n_models'],))
    model_scores_cg = np.empty((constants.PARAMS['n_models'],))
    
    if which_delay == 'precue':
        delay_ix = constants.PARAMS['trial_timepoints']['delay1_end']-1
        # fname_str = 'precue'
    elif which_delay == 'postcue':
        # post-cue delay (uncued colours)
        delay_ix = constants.PARAMS['trial_timepoints']['delay2_end']-1
        # fname_str = 'uncued_postcue'
    elif which_delay == 'postprobe':
        delay_ix = constants.PARAMS['trial_timepoints']['delay3_end']-1
        # fname_str = 'unprobed_postprobe'
    
    for model in range(constants.PARAMS['n_models']):
        # load data
        model_number = str(model)
        print('Model '+ model_number +'/' + str(constants.PARAMS['n_models']))
        f = open(load_path+'/eval_data_model' + model_number + '.pckl', 'rb')
        eval_data = pickle.load(f)    
        f.close()
        
        n_trials = eval_data['data'].shape[0]
        
        # get the cued colour labels
        labels_cued = np.concatenate((eval_data["labels"]["c1"][:n_trials//2],
                                  eval_data["labels"]["c2"][n_trials//2:]))
        # bin the labels into B colour bins
        labels_cued_binned = helpers.bin_labels(labels_cued,constants.PARAMS['B'])
        
        # separate the labels into two arrays, one for each cued location
        loc1_labels = labels_cued_binned[:n_trials//2]
        loc2_labels = labels_cued_binned[n_trials//2:]
        
        # split the hidden activity data into two arrays, each one only 
        # containing trials where one of the location was cued
        delayloc1 = eval_data['data'][:n_trials//2,delay_ix,:]
        delayloc2 = eval_data['data'][n_trials//2:,delay_ix,:]
        
        # shuffle trials
        rng = np.random.default_rng(seed=model)
        
        trial_order1 = rng.permutation(n_trials//2)
        trial_order2 = rng.permutation(n_trials//2)
        
        delayloc1 = delayloc1[trial_order1,:]
        delayloc2 = delayloc2[trial_order2,:]
        loc1_labels = loc1_labels[trial_order1]
        loc2_labels = loc2_labels[trial_order2]
        
        
        # do LDA to get the classification test (same location cue, withheld 
        # trials) and cross-generalisation (other cue trials) scores
        scores_test, scores_cg = lda_cg(delayloc1,loc1_labels,
                                        delayloc2,loc2_labels,
                                        cv=2)
        # average scores across the different binary classifiers and save
        model_scores_test[model] = scores_test.mean()
        model_scores_cg[model] = scores_cg.mean()
    return model_scores_test, model_scores_cg


def run_cg_decoding_cued_analysis(constants):
    '''
    Runs the entire cued colour cross-generalisation decoding analysis. 
    LDA classifiers trained and tested in the post-cue delay.

    Parameters
    ----------
    constants : dict
        Dictionary containing the constants for the experiment.

    Returns
    -------
    None.

    '''
    print('Run the cued colour decoding and cross-generalisation analysis')
    # get decoding test and cg scores
    model_scores_test, model_scores_cg = get_cg_decoding_cued(constants)
    
    # save into file
    cg_decoding_cued_postcue_delay = {}
    cg_decoding_cued_postcue_delay['test_accuracy']=model_scores_test
    cg_decoding_cued_postcue_delay['cross_gen_accuracy']=model_scores_cg
    load_path = constants.PARAMS['FULL_PATH']+'pca_data/valid_trials/'
    pickle.dump(cg_decoding_cued_postcue_delay,open(load_path+'cg_decoding_cued_postcue_delay.pckl','wb'))
    # run constrast - test against chance
    print('...Run contrast: mean test decoding significantly greater than chance (0.5) ')
    run_contrast_single_sample(model_scores_test, 0.5)
    
    print('...Mean decoding accuracy: %.4f' %model_scores_test.mean())
    
    print('...Run contrast: mean cross_generalisation significantly greater than chance (0.5) ')
    run_contrast_single_sample(model_scores_cg, 0.5)
    
    print('...Mean cross-generalisation accuracy: %.4f' %model_scores_cg.mean())


#%% 5) analogue to the CDI analysis: compare the discriminability of colour 
#    representations pre-cue, after they are cued and uncued

def run_colour_discrim_analysis(constants,trial_type='valid'):
    '''
    Runs a decoding analysis complementary to the CDI analysis reported in 
    Fig. 3H. Train LDA decoders in cross-validation to discriminate between 
    colours in the pre-cue delay, as well as after they are cued or uncued, and
    the compare test scores between the 3 conditions to asses how the amount of
    information about cued and uncued items changes across delays.

    Parameters
    ----------
    constants : dict
        Dictionary containing the constants for the experiment.
    trial_type : str, optional
        Optional. Relevant for the probabilistic paradigm (experiment 3). Pass 
        'valid' or 'invalid'. The default is 'valid'.

    Returns
    -------
    None.

    '''
    # get the pre-cue decoding test accuracy
    model_scores_precue = get_decoding_within_plane(constants,'precue',trial_type)
    
    model_scores_uncued = get_decoding_within_plane(constants,'postcue',trial_type)
    model_scores_cued, cg = get_cg_decoding_cued(constants,'postcue',trial_type)
    
    
    if constants.PARAMS['experiment_number']==3:
        model_scores_unprobed = get_decoding_within_plane(constants,'postprobe',trial_type)
        model_scores_probed, _ = get_cg_decoding_cued(constants,'postprobe',trial_type)
    
    # load the post-cue (uncued and cued) test accuracies from file
    data_path = constants.PARAMS['FULL_PATH']+'pca_data/'+trial_type+'_trials/'
    # model_scores_uncued = pickle.load(open(load_path+'decoding_acc_uncued_postcue_delay.pckl','rb'))
    # cued_scores_dict = pickle.load(open(load_path+'cg_decoding_cued_postcue_delay.pckl','rb'))
    # model_scores_cued = cued_scores_dict['test_accuracy']
    
    
    if constants.PARAMS['experiment_number']==3:
        all_scores = np.stack((model_scores_precue,model_scores_cued,
                               model_scores_uncued,model_scores_probed,
                               model_scores_unprobed),axis=1)
        labels = ['pre_cue','cued','uncued','probed','unprobed']
       
                           
    else:
        all_scores = np.stack((model_scores_precue,model_scores_cued,model_scores_uncued),axis=1)
        labels = ['pre_cue','cued','uncued']
    
    # all_labels =  np.stack(([[label]*constants.PARAMS['n_models'] for label in labels]),axis=1)
    pickle.dump(all_scores,open(data_path+'cdi_analogous_decoding_scores.pckl','wb'))
    # export to csv for jasp
    
    # scores_tbl = pd.DataFrame(np.stack((all_scores.flatten(),
    #                              all_labels.flatten()),1),
    #                    columns=['mean delay score','condition'])
    scores_tbl = pd.DataFrame(all_scores,columns=labels)
    scores_tbl.to_csv(data_path+'cdi_analogous_decoding_scores.csv')
    # do pairwise contrasts
    rg.test_CDI_contrasts(all_scores)
    
    # plot
    # standardise results before plotting
    # z_all_scores = (all_scores - all_scores.mean())/np.std(all_scores)
    
    # plot as units of standard normal distr
    all_scores_transf = norm.ppf(all_scores)
    
    ax = rg.plot_CDI(constants,all_scores_transf,logTransform=False,save_fig=False)
    plt.ylabel('Test decoding accuracy [snd units]')
    
    plt.savefig(constants.PARAMS['FIG_PATH']+' cdi_analogue_with_decoding.png')
    plt.savefig(constants.PARAMS['FIG_PATH']+' cdi_analogue_with_decoding.svg')
   
    return all_scores
    

def export_colour_discrim_results(constants):
    if constants.PARAMS['experiment_number']==1:
        all_scores = run_colour_discrim_analysis(constants,trial_type='valid')
    elif constants.PARAMS['experiment_number']==3:
        validity_levels = ['val1','val0_75','val0_5']
        all_scores = [[] for i in range(len(validity_levels))]
        # for v in validity_levels:
             
            
    return
    
#%% 6) analogue to the rotated/unrotated plane AI analysis: compare the 
# cross-temporal generalisation scores (between the precue and postcue 
# delays) between the two cued planes

def get_decoding_unrotrot(constants,cv=2):
    '''
    For each location, using half the data, get the cross-temporal generalisation
    score across the two delays. Relabel the location with a higher score as 
    'unrotated' and repeat the analysis using the other half of data, this time
    saving the ctg scores for the 'rotated' and 'unrotated' planes. If models
    keep one pre-cue subspace unchanged, the unrotated scores should be 
    significantly higher than 1) chance and 2) the rotated scores.
    

    Parameters
    ----------
    params : dict
        Experiment parameters.
    cv : int, optional
        Number of cross-validation folds. The default is 2.

    Returns
    -------
    ctg_decoding_test : array
        AI values for the 'unrotated' and 'rotated' planes, averaged across cv 
        folds. Format: (n_dims,(unrotated,rotated),model)
   
    same_ixs : array (n_cv_folds,n_models)
        Indexes of the unrotated plane for each model.

    '''
    load_path = constants.PARAMS['FULL_PATH'] + 'pca_data/valid_trials'


    d1_ix = constants.PARAMS['trial_timepoints']['delay1_end']-1
    d2_ix = constants.PARAMS['trial_timepoints']['delay2_end']-1
    
    trial_ixs = pickle.load(open(load_path+'/trial_ixs_for_unrotrot_analysis.pckl','rb'))
    
    
    # cross-validated test decoding score of the same location during the same delay
    # i.e., how decodable is the colour information for this condition
    test_decoding_train = np.zeros((2,2,constants.PARAMS['n_models'])) 
    test_decoding_test = np.zeros((2,2,constants.PARAMS['n_models'])) 
    # cross-temporal-generalisation decoding score. Decoder trained on a given 
    # location with data from one delay and tested on the other delay
    # i.e., if the plane is unrotated and phase-aligned, then the score should be
    # very high
    ctg_decoding_train = np.zeros((2,2,constants.PARAMS['n_models'])) 
    ctg_decoding_test = np.zeros((2,2,constants.PARAMS['n_models'])) 
    # cv folds, unrotated/rotated, model
    
    same_ixs = np.zeros((cv,constants.PARAMS['n_models'])) #ixs of the unrotated plane
    
    
    for model in range(constants.PARAMS['n_models']):
        # load data
        model_number = str(model)
        f = open(load_path+'/eval_data_model' + model_number + '.pckl', 'rb')
        eval_data = pickle.load(f)    
        f.close()
                
        train, test = trial_ixs['train'][model_number],trial_ixs['test'][model_number]
        
        loc_split = constants.PARAMS['B']
        for i in range(2):
            # bin the train and test datasets intro colour bins
            # data_train = helpers.bin_data(eval_data['data'][train[i],:,:],params)
            # data_test = helpers.bin_data(eval_data['data'][test[i],:,:],params)
            
            # delay1_train = data_train[:,d1_ix,:]
            # delay1_test = data_test[:,d1_ix,:]
            
            # delay2_train = data_train[:,d2_ix,:]
            # delay2_test = data_test[:,d2_ix,:]
            
            data_train = eval_data['data'][train[i],:,:]
            data_test = eval_data['data'][test[i],:,:]
            
            # run decoding to find the unrotated and rotated subspace
        
            
            labels_loc1_train = eval_data["labels"]["c1"][train[i]]
            labels_loc2_train = eval_data["labels"]["c2"][train[i]]
            
            labels_loc1_test = eval_data["labels"]["c1"][test[i]]
            labels_loc2_test = eval_data["labels"]["c2"][test[i]]
            
            # sort labels
            loc1_train_sorting_ix, labels_loc1_train = helpers.sort_labels(labels_loc1_train)
            loc2_train_sorting_ix, labels_loc2_train = helpers.sort_labels(labels_loc2_train)
            
            loc1_test_sorting_ix, labels_loc1_test = helpers.sort_labels(labels_loc1_test)
            loc2_test_sorting_ix, labels_loc2_test = helpers.sort_labels(labels_loc2_test)
        
            
            # bin the labels into B colour bins
            labels_loc1_train_binned = helpers.bin_labels(labels_loc1_train,constants.PARAMS['B'])
            labels_loc2_train_binned = helpers.bin_labels(labels_loc2_train,constants.PARAMS['B'])
            labels_loc1_test_binned = helpers.bin_labels(labels_loc1_test,constants.PARAMS['B'])
            labels_loc2_test_binned = helpers.bin_labels(labels_loc2_test,constants.PARAMS['B'])
            
            # sort the arrays
            data_train_loc1 = data_train[loc1_train_sorting_ix,:,:]
            data_train_loc2 = data_train[loc2_train_sorting_ix,:,:]
            data_test_loc1 = data_test[loc1_test_sorting_ix,:,:]
            data_test_loc2 = data_test[loc2_test_sorting_ix,:,:]
            

            # shuffle trials
            rng = np.random.default_rng(seed=model)

            trial_order_loc1_train = rng.permutation(len(loc1_train_sorting_ix))
            trial_order_loc2_train = rng.permutation(len(loc2_train_sorting_ix))
            trial_order_loc1_test = rng.permutation(len(loc1_test_sorting_ix))
            trial_order_loc2_test = rng.permutation(len(loc2_test_sorting_ix))
            
            labels_loc1_train_binned = labels_loc1_train_binned[trial_order_loc1_train]
            labels_loc2_train_binned = labels_loc2_train_binned[trial_order_loc2_train]
            labels_loc1_test_binned = labels_loc1_test_binned[trial_order_loc1_test]
            labels_loc2_test_binned = labels_loc2_test_binned[trial_order_loc2_test]
            
            
            data_train_loc1 = data_train_loc1[trial_order_loc1_train,:,:]
            data_train_loc2 = data_train_loc2[trial_order_loc2_train,:,:]
            data_test_loc1 = data_test_loc1[trial_order_loc1_test,:,:]
            data_test_loc2 = data_test_loc2[trial_order_loc2_test]
            

            # do LDA to get the classification at test (same location cue and 
            # delay, withheld trials) and cross-temporal-generalisation (same
            # location cue, other delay) scores
            
            # use the train data to find the location for which cross-temporal 
            # decoding across the two delays has higher accuracy - this is the 
            #'unrotated' location
            # scores format: [train_timepoint, test_timepoint]
            scores_cg_train_loc1, _ = lda_cg_time(np.swapaxes(data_train_loc1[:,[d1_ix,d2_ix],:],-1,1),labels_loc1_train_binned)
            scores_cg_train_loc2, _ = lda_cg_time(np.swapaxes(data_train_loc2[:,[d1_ix,d2_ix],:],-1,1),labels_loc2_train_binned)
            
            # rotated location
            switch_ix = np.argmin((np.array([scores_cg_train_loc1[0,1],scores_cg_train_loc1[1,0]]).mean(),
                                   np.array([scores_cg_train_loc2[0,1],scores_cg_train_loc2[1,0]]).mean()))
            # unrotated location
            stay_ix = np.setdiff1d([0,1],switch_ix)[0]
            
            same_ixs[i,model] = stay_ix
            # repeat the analysis using the test data
            scores_cg_test_loc1, _ = lda_cg_time(np.swapaxes(data_test_loc1[:,[d1_ix,d2_ix],:],-1,1),labels_loc1_test_binned)
            scores_cg_test_loc2, _ = lda_cg_time(np.swapaxes(data_test_loc2[:,[d1_ix,d2_ix],:],-1,1),labels_loc2_test_binned)
            
            
            # save the data in the frame of
            # reference of the 'rotated' and 'unrotated' plane
            if stay_ix == 0:
                # unrotated decoding (loc1)
                ctg_decoding_train[i,0,model]  = np.array([scores_cg_train_loc1[0,1],scores_cg_train_loc1[1,0]]).mean()
                ctg_decoding_test[i,0,model]  = np.array([scores_cg_test_loc1[0,1],scores_cg_test_loc1[1,0]]).mean()
                # rotated decoding
                ctg_decoding_train[i,1,model]  = np.array([scores_cg_train_loc2[0,1],scores_cg_train_loc2[1,0]]).mean()
                ctg_decoding_test[i,1,model]  = np.array([scores_cg_test_loc2[0,1],scores_cg_test_loc2[1,0]]).mean()
                
                # decoder accuracy on withheld data from the same delay 
                test_decoding_train[i,0,model] = np.array([scores_cg_train_loc1[0,0],scores_cg_train_loc1[1,1]]).mean()
                test_decoding_test[i,0,model] = np.array([scores_cg_test_loc1[0,0],scores_cg_test_loc1[1,1]]).mean()
                test_decoding_train[i,1,model] = np.array([scores_cg_train_loc2[0,0],scores_cg_train_loc2[1,1]]).mean()
                test_decoding_test[i,1,model] = np.array([scores_cg_test_loc2[0,0],scores_cg_test_loc2[1,1]]).mean()
            else:
                # unrotated decoding (loc2)
                ctg_decoding_train[i,0,model]  = np.array([scores_cg_train_loc2[0,1],scores_cg_train_loc2[1,0]]).mean()
                ctg_decoding_test[i,0,model]  = np.array([scores_cg_test_loc2[0,1],scores_cg_test_loc2[1,0]]).mean()
                
                # rotated decoding
                ctg_decoding_train[i,1,model]  = np.array([scores_cg_train_loc1[0,1],scores_cg_train_loc1[1,0]]).mean()
                ctg_decoding_test[i,1,model]  = np.array([scores_cg_test_loc1[0,1],scores_cg_test_loc1[1,0]]).mean()
                
                
                # decoder accuracy on withheld data from the same delay 
                test_decoding_train[i,0,model] = np.array([scores_cg_train_loc2[0,0],scores_cg_train_loc2[1,1]]).mean()
                test_decoding_test[i,0,model] = np.array([scores_cg_test_loc2[0,0],scores_cg_test_loc2[1,1]]).mean()
                test_decoding_train[i,1,model] = np.array([scores_cg_train_loc1[0,0],scores_cg_train_loc1[1,1]]).mean()
                test_decoding_test[i,1,model] = np.array([scores_cg_test_loc1[0,0],scores_cg_test_loc1[1,1]]).mean()
                
            
    
    
    # save the ctg scores
    pickle.dump(ctg_decoding_train,open(load_path+'/ctg_decoding_train_rot_unrot.pckl','wb'))
    pickle.dump(ctg_decoding_test,open(load_path+'/ctg_decoding_test_rot_unrot.pckl','wb'))
    
    # save the test decoding scores
    pickle.dump(test_decoding_train,open(load_path+'/test_decoding_train_rot_unrot.pckl','wb'))
    pickle.dump(test_decoding_test,open(load_path+'/test_decoding_test_rot_unrot.pckl','wb'))
    
    # export to csv for JASP
    ctg_scores_jasp = pd.DataFrame(data=ctg_decoding_test.mean(0).T,columns = ['unrotated','rotated'])
    ctg_scores_jasp.to_csv(load_path+'/ctg_decoding_test_jasp.csv')
    
    # save the unrotated plane ixs
    pickle.dump(same_ixs,open(load_path+'/unrot_plane_ixs_ctg_decoding.pckl','wb'))
    
    
    # plot the ctg scores for unrotated and rotated planes
    plot_AI(constants.PARAMS,ctg_decoding_test.mean(0)[None,:,:],'unrotrot','cued')
    plt.xticks([1.875,2.125],labels=['unrotated','rotated'])
    plt.ylabel('ctg decoding accuracy')
    plt.xlabel('Cued colour plane')
    plt.legend([])
    
    plt.savefig(constants.PARAMS['FIG_PATH']+'ctg_decoding_acc_rotunrot.png')
    plt.savefig(constants.PARAMS['FIG_PATH']+'ctg_decoding_acc_rotunrot.svg')
    
    # run stats
    # contrast 1: unrotated > chance (0.5)
    print('Contrast 1: unrotated > chance (0.5)')
    print('Mean = %.2f' %(ctg_decoding_test.mean(0)[0,:].mean()*100))
    run_contrast_single_sample(ctg_decoding_test.mean(0)[0,:],[.5],alt='greater')
    
    # contrast 2: rotated =/= chance
    # note here we should be doing a Bayesian test - done in JASP
    print('Contrast 2: rotated =/= chance (0.5)')
    print('Mean = %.2f' %(ctg_decoding_test.mean(0)[1,:].mean()*100))
    run_contrast_single_sample(ctg_decoding_test.mean(0)[1,:],[.5],alt='two-sided')
    
    # contrast 3: unrotated > rotated 
    
    print('Contrast 3: unrotated > rotated (0.5)')
    run_contrast_single_sample(ctg_decoding_test.mean(0)[0,:]-ctg_decoding_test.mean(0)[1,:],
                               [0],alt='greater')

    return ctg_decoding_test, same_ixs
    