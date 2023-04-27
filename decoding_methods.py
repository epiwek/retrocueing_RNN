#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 17:58:34 2021

@author: emilia

This file contains all decoding analysis functions, including:
    1) decoding of uncued stimulus colour in the post-cue delay
    2) cross-temporal decoding
    3) comparison of  maintenance mechanisms between models from expts 1 & 2

1) This analysis asks if there is still information about the uncued item 
    colour in the post-cue delay.
2) This analysis computes the cross-temporal decoding accuracy scores for the
    cued items across the entire trial duration.
3) This analysis calculates the mean delay cross-temporal generalisation scores and 
    compares those between expts 1 & 2 to assess whether the networks trained 
    with variable delay lengths (expt 2) form a more temporally stable working 
    memory code than those trained with fixed delays (expt 1).

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

from stats import run_contrast_single_sample,run_contrast_unpaired_samples
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


def lda(X,y, cv=2):
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
    scores_test = np.zeros((len(class_combos)))
    for i in range(len(class_combos)):
        y1,y2 = class_combos[i] # class labels
        
        # find trials of the abovespecified classes
        trial_ix = np.where(np.logical_or(y==y1,y==y2))[0]
        
        # make classifier pipeline
        clf = make_pipeline(StandardScaler(), LDA())
        
        # fit a classifier in cross-val
        results = cross_validate(clf,X[trial_ix,:],y[trial_ix],cv=cv,return_estimator=False)
        
        # average test scores across cv folds
        scores_test[i] = results['test_score'].mean()
    
    return scores_test     


def lda_cg_time(X,y):
    '''
    Test LDA classifiers to discriminate between pairs of classes based on data
    from a single timepoint and test their performance on all the other 
    timepoints.

    Parameters
    ----------
    X : array (n_samples,n_features)
        Data matrix.
    y : array (n_samples,)
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

def get_decoding_uncued(constants):
    '''
    Train and test LDA binary classifiers to discriminate between pairs of 
    uncued colour labels in the post-cue delay (at thelast time-point).

    Parameters
    ----------
    constants : dict
        Dictionary containing the constants for the experiment.

    Returns
    -------
    model_scores : array
        Average test decoding acuracy scores for all models.

    '''
    load_path = constants.PARAMS['FULL_PATH']+'pca_data/valid_trials/'
    model_scores = np.empty((constants.PARAMS['n_models'],))
    for model in range(constants.PARAMS['n_models']):
        # load data
        model_number = str(model)
        print('Model '+ model_number +'/' + str(constants.PARAMS['n_models']))
        f = open(load_path+'/eval_data_model' + model_number + '.pckl', 'rb')
        eval_data = pickle.load(f)    
        f.close()
        
        n_trials = eval_data['data'].shape[0]
        delay2_ix = constants.PARAMS['trial_timepoints']['delay2_end']-1
        
        labels_uncued = np.concatenate((eval_data["labels"]["c2"][:n_trials//2],
                                  eval_data["labels"]["c1"][n_trials//2:]))
        labels_uncued_binned = helpers.bin_labels(labels_uncued,constants.PARAMS['B'])
        loc1_labels = labels_uncued_binned[:n_trials//2]
        loc2_labels = labels_uncued_binned[n_trials//2:]
        
        delay2loc1 = eval_data['data'][:n_trials//2,delay2_ix,:]
        delay2loc2 = eval_data['data'][n_trials//2:,delay2_ix,:]
        
        # do LDA
        scores_loc1 = lda(delay2loc1,loc1_labels)
        scores_loc2 = lda(delay2loc2,loc2_labels)
        
        model_scores[model] = np.stack((scores_loc1,scores_loc2)).mean()
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
    model_scores = get_decoding_uncued(constants)
    
    # run constrast - test against chance
    print('...Run contrast: mean test decoding significantly greater than chance (0.5) ')
    run_contrast_single_sample(model_scores, 0.5)
    
    print('...Mean decoding accuracy: %.4f' %model_scores.mean())

#%% 2: cross-temporal decoding

def fit_and_plot_ctg(constants,tmin,tmax):
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
    fit_and_plot_ctg(tmin,tmax)
    plt.savefig(constants.PARAMS['FIG_PATH']+'cross_temp_decoding_alltimepoints.png')


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
    run_contrast_single_sample(off_diag_scores[:,0],h_mean=.5,alt='greater')
    
    # contrast 2: test fixed delays off-diagonal mean against chance (0.5)
    print('...Contrast 2: Fixed delays mean ctg decoding > chance')
    run_contrast_single_sample(off_diag_scores[:,1],h_mean=.5,alt='greater')
    
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
