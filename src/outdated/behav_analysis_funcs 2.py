#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 18:51:06 2021

@author: emilia
"""
import torch
import pickle
import pycircstat

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import vonmises
from scipy.io import savemat

import constants

from helpers import angle_to_vec, wrap_angle


def get_choices(outputs,params,policy = 'softmax'):
    """
    Convert the distributed patterns of activity in the output layer into 
    choices using a specified policy.
    
    Parameters
    ----------
    outputs : array-like (batch_size, output channels)
        output layer activation values from the model of interest
    
    params : dictionary 
           
    policy : str
        'softmax' (default), 'hardmax' or 'posterior_mean' 

    Returns
    -------
    choices: array-like (batch_size,)
        chosen stimulus colour defined as angle [rad] in circular colour space
    
    """
    # get tuning curve centres
    phi  = torch.linspace(-np.pi, np.pi, params['n_colCh']+1)[:-1]
    # convert output activation values into choice probabilities
    if params['loss_fn'] == 'CEL':
        softmax = torch.nn.Softmax(dim=-1)
        choice_probs = softmax(outputs)
    else:
        choice_probs = outputs
        
    n_trials = choice_probs.shape[0]
    # get choices on each trial
    if policy == 'softmax':
        # softmax policy - sample choices proportionally to their probabilities
        choice_ixs = torch.multinomial(choice_probs,1)
        choices = phi[choice_ixs].squeeze()            
    elif policy == 'posterior_mean':
        choices = posterior_mean(phi,choice_probs)
    elif policy == 'hardmax':
        # hardmax policy - choose the output wih the highest choice probabiitty
        ix = np.argsort(choice_probs,1)[:,-1]
        choices = torch.empty((n_trials,))
        for i in range(n_trials):
            choices[i] = phi[ix[i]]
    else:
        raise ValueError('Specify a valid policy')
    return choices


def posterior_mean(angles,probs,**kwargs):
    """
    Convert the distributed patterns of activity in the output layer into 
    choices using the posterior mean policy.
    
    Parameters
    ----------
    outputs : array-like (batch_size, output channels)
        output layer activation values from the model of interest
    
    params : dictionary 
       
    load_path : str
    
    policy : str
        'posterior_mean' (default) or 'softmax'

    Returns
    -------
    choices: array-like (batch_size,)
    """
    n_trials = probs.shape[0]
    
    # convert the angles to vectors to calculate the circular mean
    angles_vectors = angle_to_vec(angles)
    angles_vectors = torch.stack(([angles_vectors]*n_trials),1)
    mean_vectors = torch.sum(torch.tensor(angles_vectors)*probs,-1)
    
    # convert back into angles
    if len(mean_vectors.shape)<2:
        # for vectors
        choices = np.arctan2(mean_vectors[1],mean_vectors[0])
    else:
        # for matrices
        choices = np.arctan2(mean_vectors[1,:],mean_vectors[0,:])
    
    return choices

# TAKE OUT
# def angle_to_vec(angles):
#     """
#     Helper function to convert an array of angles into their unit-circle 
#     vector representations. 
    
#     Parameters
#     ----------
#     angles : array-like ()
    
#     Returns
#     -------
#     angles_vectors : array-like ()
#     """
#     angles_vectors = torch.stack((np.cos(angles),np.sin(angles)))
#     return angles_vectors


def get_angular_error(test_dataset,params,load_path):
    """
    Get the distribution of errors for each colour stimulus for a given model.
    Errors are calculated for a given colour, irrespective of the location 
    where it was shown. Error is defined as the absolute of the difference 
    between output layer and target activation values, averaged across all 
    stimulus colours and trials.
    
    Parameters
    ----------
    test_dataset : dict
        Dictionary containing the test dataset.
    params : dict 
        Parameters for the experiment
    load_path : str
        Path to the file containing model outputs matrix.
    Returns
    -------
    ang_errors : torch.Tensor (n output channels, n trials per cued colour)
        Error values distributed across the output layer channels, referenced 
        to the original stimulus activation pattern
    choices : torch.Tensor (n test trials,)
        reported stimulus colours defined as angle [rad] in circular colour space
        
    """
    # get indices of trials where loc1 and loc2 were probed
    loc1_ix = np.array(test_dataset['loc'][0,:,:].squeeze(),dtype=bool)
    loc2_ix = np.array(test_dataset['loc'][1,:,:].squeeze(),dtype=bool)
    
    # get probed colour for each trial - encoded as the centre of the tuning curve
    probed_colour = torch.cat((test_dataset['c1'][loc1_ix],test_dataset['c2'][loc2_ix]))
    
    n_trials = probed_colour.shape[0]
    # sort the trial indices for each probed colour
    colours = np.unique(probed_colour)
    colour_ix = [None] * len(colours)
    for i,c in enumerate(colours):
        colour_ix[i] = np.where(probed_colour==c)[0]
    
    # load output layer activations
    f = open(load_path+'model_outputs_model'+str(params['model_number'])+'.pckl','rb')
    model_outputs = pickle.load(f)
    f.close()
    outputs = model_outputs['data']
    
    # convert into choices
    choices = get_choices(outputs,params)
    
    
    # if want to assess theoretical performance for softmax:
    #outputs = test_dataset['targets'] 
    #choices = get_choices(outputs,params,load_path,policy='softmax')

    # calculate angular errors
    ang_errors = torch.empty((len(colours),n_trials//len(colours))) 
    # error for each colour, irrespective of its location
    for c in range(params['n_stim']):
        ang_errors[c,:] = choices[colour_ix[c]]-probed_colour[colour_ix[c]]
    
    
    # Wrap angles to [-pi, pi)
    ang_errors = wrap_angle(ang_errors)
    
    return ang_errors, choices
    

def get_probed_colour(test_dataset):
    '''
    Get the probed and unprobed colour matrices.

    Parameters
    ----------
    test_dataset : dict
        Dictionary containing the test dataset.

    Returns
    -------
    probed_colour : torch.Tensor (n_trials,)
        matrix with the trial-wise probed colour values
    unprobed_colour : torch.Tensor (n_trials,)
        matrix with the trial-wise unprobed colour values

    '''
    # get indices of trials where loc1 and loc2 were probed
    loc1_ix = np.array(test_dataset['loc'][0,:,:].squeeze(),dtype=bool)
    loc2_ix = np.array(test_dataset['loc'][1,:,:].squeeze(),dtype=bool)
    # get probed colour for each trial
    probed_colour = torch.cat((test_dataset['c1'][loc1_ix],test_dataset['c2'][loc2_ix]))
    unprobed_colour = torch.cat((test_dataset['c2'][loc1_ix],test_dataset['c1'][loc2_ix]))
    return probed_colour,unprobed_colour

# TAKE OUT
# def wrap_angle(angle):
#     """
#     Wraps angle(s) to be within [-pi, pi).
    
#     Parameters
#     ----------
#     angle : array-like
#         angle in radians

#     Returns
#     -------
#     angle_wrapped : array-like
#     """
#     angle_wrapped = (angle+np.pi) % (2*np.pi) - np.pi
#     return angle_wrapped
 

def get_datasets(constants):
    '''
    Get test datasets for different experimenta conditions.

    Parameters
    ----------
    constants : dict
        dictionary containing the constants for the experiment.

    Returns
    -------
    conditions : list
        Names of the testing conditions.
    full_paths : list
        list containing the paths to the test datasets.
    ds : list
        list containing the test datasets.

    '''
    main_path = constants.PARAMS['FULL_PATH'] + 'pca_data/'
    
    # get conditional paths
    if constants.PARAMS['experiment_number']<3:
        conditions  = ['trained','in-range','out-of-range']
        cond_paths = ['','in_range_tempGen/','out_range_tempGen/']
        full_paths = [main_path+'valid_trials/'+cond_paths[i] for i in range(len(cond_paths))]
    elif constants.PARAMS['experiment_number']==3:
        conditions  = ['valid']
        if constants.PARAMS['cue_validity'] !=1:
            conditions.append('invalid')
        cond_paths = conditions
        full_paths = [main_path+cond_paths[i]+'_trials/' for i in range(len(cond_paths))]
    elif constants.PARAMS['experiment_number']==4:
        conditions = ['valid']
        cond_paths = ['valid_trials/']
        full_paths = [main_path+cond_paths[i] for i in range(len(cond_paths))]
    
    # get datasets
    ds = []
    for i,c in enumerate(conditions):
        if constants.PARAMS['experiment_number']<3:
            ds.append(pickle.load((open(full_paths[i]+'test_dataset.pckl','rb'))))
        else:
            ds.append(pickle.load((open(full_paths[i]+'0.pckl','rb'))))
    return conditions, full_paths, ds


def bin_errors(constants,ang_errors_all_models):
    '''
    Bin model angular errors into 9 equally spaced bins.

    Parameters
    ----------
    constants : dict
        Dictionary containing the constants for the experiment.
    ang_errors_all_models : torch.Tensor 
        shape: (n models, n output channels, n trials per cued colour)
        Error values distributed across the output layer channels, referenced 
        to the original stimulus activation pattern

    Returns
    -------
    bin_edges : array
        Edges of the bins.
    binned_errors : array
        Error density values for each bin.

    '''
    # set the bin range to the absolute max error rounded to the nearest 10th
    # b = np.round(abs_max_err,-1) 
    bin_range = np.pi
    bin_edges = np.linspace(-bin_range,bin_range,10)
    bin_centres = []
    for i in range(len(bin_edges)-1):
        bin_centres.append(np.mean(bin_edges[i:i+2]))

    binned_errors = np.empty((constants.PARAMS['n_models'],len(bin_centres)))
    for m in range(constants.PARAMS['n_models']):
        # TAKE OUT
        # binned_errors[m,:], bins, patches = \
            # np.histogram(ang_errors_all_models[m,:,:].view(-1).numpy(),
            #                     bins=bin_edges)
                                # density=True,
                                # ec='k',
                                # fc='lightgrey',
                                # label='data')
        binned_errors[m,:], bin_edges = \
            np.histogram(ang_errors_all_models[m,:,:].view(-1).numpy(),
                                bins=bin_edges, 
                                range=(-bin_range,bin_range),
                                density=True)
    return bin_edges, binned_errors, bin_centres


def get_descr_stats(constants,ang_errors_all_models):
    '''
    Calculate the descriptive statistics (mean absolute error and standard 
    deviation) for each model. Priont the mean and SEM across all models.

    Parameters
    ----------
    constants : dict
        Dictionary containing the constants for the experiment.
    ang_errors_all_models : torch.Tensor 
        shape: (n modes,n output channels, n trials per cued colour)
        Error values distributed across the output layer channels, referenced 
        to the original stimulus activation pattern

    Returns
    -------
    mean_abs_err : array (n models,)
        Mean absolute angular error values for each model.
    sd_abs_err : array (n models,)
        Standard deviation of absolute angular error values for each model.

    '''
    mean_abs_err = np.empty((constants.PARAMS['n_models'],))
    sd_abs_err = np.empty((constants.PARAMS['n_models'],))
    
    for m in range(constants.PARAMS['n_models']):
        mean_abs_err[m] = \
            np.mean(np.abs(np.degrees(ang_errors_all_models[m,:,:].view(-1).numpy())))
        sd_abs_err[m] = \
            np.std(np.abs(np.degrees(ang_errors_all_models[m,:,:].view(-1).numpy())))
    
    print('Mean absolute angular error across all models: %.2f' %np.mean(mean_abs_err))
    print('SEM: %.2f' %(np.std(mean_abs_err)/np.sqrt(constants.PARAMS['n_models'])))
    
    return mean_abs_err, sd_abs_err


def fit_vonMises_pdf(constants,ang_errors_all_models):
    '''
    Fit a von Mises (circular normal) probability density function to the 
    angular error data. Function is parameterised by:
        - kappa - concentration parameter
        - mu - centre/mean of the distribution
        - scale (fixed to 1.0)
    
    Parameters
    ----------
    constants : dict
        Dictionary containing the constants for the experiment.
    ang_errors_all_models : torch.Tensor 
        shape: (n models, n output channels, n trials per cued colour)
        Error values distributed across the output layer channels, referenced 
        to the original stimulus activation pattern.

    Returns
    -------
    fitted_params_indiv_models : pandas data frame
        Fitted parametters for each model.
    fitted_params_average : array
        Parameters fit to the data concatenated across all modes.

    '''
    fitted_params_indiv_models = np.empty((constants.PARAMS['n_models'],3)) 

    for m in range(constants.PARAMS['n_models']):
        fitted_params_indiv_models[m,0],fitted_params_indiv_models[m,1],\
            fitted_params_indiv_models[m,2] = \
            vonmises.fit(ang_errors_all_models[m,:,:].view(-1).numpy(),fscale=1.0)
    # note - need to fix the scale to 1.0 in order to be able to fit the pdf,
    # otherwise it fits some kind of high frequency oscillatory function (can't
    # recover parameters even on simulated data)
    fitted_params_indiv_models = \
        pd.DataFrame(fitted_params_indiv_models,columns = ['kappa','mu','scale'])
    
    fitted_params_average = vonmises.fit(ang_errors_all_models.view(-1).numpy(),
                                         fscale=1)
    return fitted_params_indiv_models, fitted_params_average
#%% plotters
def plot_angular_error(error,phi_degrees):
    """
    Plots error distribution.
    
    Parameters
    ----------
    error : array-like
        error values
    phi_degrees : array_like 
        x-labels

    Returns
    -------
    None
        
    """
    plt.figure()

    plt.plot(phi_degrees,error,'ko-')
    plt.xticks(phi_degrees)
    
    plt.xlabel('Angular error')
    plt.ylabel('mean output-target activity')
    plt.tight_layout()
    
    
def plot_err_distr_indiv_models(ang_errors_struct):
    '''
    Plot the binned angular errors along with the fitted von Mises pdf. Data 
    for each model plotted in a separate figure.

    Parameters
    ----------
    ang_errors_struct : dict
        Structure containing the binned angular errors alongside the bin edges.

    Returns
    -------
    None.

    '''
    vm_params,_ = fit_vonMises_pdf(constants,ang_errors_struct['ang_errors'])
    pdf_x = np.linspace(-b,b, 100) # x vals for the fitted von Mises pdf
    for m in range(constants.PARAMS['n_models']):
        ang_errors_struct['binned_errors']
        # plot the fitted pdf - would need to fix so that histogram bars sum up to
        # 1, otherwise peak of fitted pdf is on a different scale
        
        pdf_y = vonmises.pdf(pdf_x,
                          vm_params[m,0],
                          vm_params[m,1],
                          vm_params[m,2])
        
        plt.figure()
        plt.title('Model %d' %m)
        print('check whether the histogram works')
        plt.hist(ang_errors_struct['bin_edges'], ang_errors_struct['binned_errors'])
        ax.plot(pdf_x,pdf_y,'k-',label='fit')
        ax.set_ylabel('Density of trials')
        ax.set_xlabel('Angular error [°]')


def plot_average_err_distr(ang_errors_struct):
    '''
    Errorbar plot for binned angular errors. Plots the mean+-SEM across all 
    models for each bin.

    Parameters
    ----------
    ang_errors_struct : dict
        Dictionary containing the binned angular error data.

    Returns
    -------
    None.

    '''
    sns.set_context("notebook", font_scale=2)
    sns.set_style("ticks")
    
    
    fig, ax = plt.subplots()
    
    # # average across models
    # mean_err = ang_errors_struct['binned_errors'].mean(0)
    # sem_err = np.std(ang_errors_struct['binned_errors'],0)/np.sqrt(constants.PARAMS['n_models'])
    ax.errorbar(ang_errors_struct['bin_centres'],
                ang_errors_struct['binned_errors_mean'],
                yerr=ang_errors_struct['binned_errors_sem'],
                ls = '',marker= 'o',mfc='w',mec = 'k',
                ecolor = 'k', label = 'data')
    
    pdf_x = np.linspace(ang_errors_struct['bin_edges'][0],
                        ang_errors_struct['bin_edges'][-1],
                        100) # x vals for the fitted von Mises pdf    
    _, vm_params = fit_vonMises_pdf(constants,ang_errors_struct['ang_errors'])
    pdf_y = vonmises.pdf(pdf_x,
                               vm_params[0],vm_params[1],vm_params[2])
    
    
    ax.plot(pdf_x,pdf_y,'k-',label='fit')
    ax.set_ylabel('Density of trials')
    ax.set_xlabel('Angular error [°]')
    ax.set_xticks([-np.radians(100),0,np.radians(100)])
    ax.set_xticklabels(labels = ['-100','0','100'])
    
    plt.tight_layout()
    
    sns.despine()
    # figpath = constants.PARAMS['FIG_PATH'] + 'error_distribution.png'
    # plt.savefig(figpath)    
    return


def plot_valid_and_invalid_average(constants,valid_struct,invalid_struct):
    '''
    Errorbar plot of binned angular errors on valid and invalid trials. Plots 
    the mean+-SEM across all models for each bin. Only to be used with 
    probabilistic conditions.

    Parameters
    ----------
    constants : dict
        Dictionary containing the constants for the experiment.
    valid_struct : dict
        Dictionary containing the binned angular error data on valid trials.
    invalid_struct : dict
        Dictionary containing the binned angular error data on invalid trials..

    Returns
    -------
    None.

    '''
    if constants.PARAMS['cue_validity'] == 1:
        # only use with probabilstic conditions.
        return
    
    fig, ax = plt.subplots(111)
    
    # plot valid trials - mean and sem
    ax.errorbar(valid_struct['bin_centres'],
                valid_struct['binned_errors_mean'],
                yerr=valid_struct['binned_errors_sem'],
                ls = '',marker= 'o',mfc='w',mec = 'g', 
                ecolor = 'g', label = 'valid')
    
    # add a vonMises fit
    _, vm_params_valid = fit_vonMises_pdf(constants,valid_struct['ang_errors'])
    pdf_x = np.linspace(-np.pi,np.pi,100)
    pdf_y_valid = vonmises.pdf(pdf_x,
                               vm_params_valid[0],
                               vm_params_valid[1],
                               vm_params_valid[2])
    ax.plot(pdf_x,pdf_y_valid,'g-')
    
   
    # repeat all for invalid trials
    ax.errorbar(invalid_struct['bin_centres'],
                invalid_struct['binned_errors_mean'],
                yerr=invalid_struct['binned_errors_sem'],
                ls = '',marker= 'o',mfc='w',mec = 'r',
                ecolor = 'r', label = 'invalid')

    _, vm_params_invalid = fit_vonMises_pdf(constants,invalid_struct['ang_errors'])    
    pdf_y_invalid = vonmises.pdf(pdf_x,
                                 vm_params_invalid[0],
                                 vm_params_invalid[1],
                                 vm_params_invalid[2])

    if constants.PARAMS['cue_validity']== .5:
        ax.plot(pdf_x,pdf_y_invalid,'r--')
    else:
        ax.plot(pdf_x,pdf_y_invalid,'r-')
        plt.legend()
    
    ax.set_ylabel('Density')
    ax.set_xlabel('Angular error [radians]')
    ax.set_xticks([-np.radians(100),0,np.radians(100)])
    ax.set_xticklabels(labels = ['-100','0','100'])
    
    plt.tight_layout()
    sns.despine()
    
    #figpath = constants.PARAMS['FIG_PATH'] + 'validity_effect_paper.png'


def plot_mixture_model_params_validity(constants):
    load_path = '/Users/emilia/OneDrive - Nexus365/MATLAB/rnn_retrocue_data/'
    fig_path = constants.PARAMS['BASE_PATH'] + 'data_vonMises/experiment_3/'
    params = ['K','pT','pNT','pU']
    trial_type = ['valid','invalid']
    
    n_models = constants.PARAMS['n_models']
    
    data_reshaped = []
    for p in range(len(params)):
        # load data
        data = pd.read_csv(load_path + params[p]+'_table.csv')
        # reshape data for seaborn
        condition = np.tile(data['condition'].to_numpy(),2)
        t_type = [[trial_type[t]]*n_models*2 for t in range(len(trial_type))]
        t_type = np.reshape(t_type,-1)
        label1 = params[p]+'_'+trial_type[0]
        label2 = params[p]+'_'+trial_type[1]
        
        param_vals = np.concatenate((data[label1].to_numpy(),data[label2].to_numpy()))
        data_reshaped = np.stack((t_type,condition,param_vals),1)
        data_reshaped = pd.DataFrame(data_reshaped,columns=['trial_type','condition',params[p]])
        model_nr = np.tile(np.arange(n_models*2),2)
        
        g = sns.catplot(x='trial_type',y =params[p],hue='condition', data=data_reshaped,
                    kind="point",markers=["^", "o"],units=model_nr,dodge=True,ci=68,
                    palette=sns.color_palette("Set2")[2:0:-1])
        g.figure.set_size_inches((5.6,5))
        sns.despine(top=False,right=False)
        
        plt.xlabel('trial type')
        
        plt.savefig(fig_path+'mix_model_'+params[p]+'.svg')
   

# def plot_mixture_model_params_vardelays(constants):
#     load_path = '/Users/emilia/OneDrive - Nexus365/MATLAB/rnn_retrocue_data/'
#     params = ['K','pT','pNT','pU']    

#     for p in range(len(params)):
#         # load data
#         data = pd.read_csv(load_path + 'var_delays_'+params[p]+'_table.csv')
        
#         sns.catplot(x='condition',y =params[p], data=data,color='k',
#                     kind="point",ci=68)
        
#         plt.xlabel('delay length')
#         plt.savefig(constants.PARAMS['FIG_PATH']+'mix_model_'+params[p]+'.png')
        
#     return
#%% full pipeline

def run_behav_analysis(constants):
    '''
    Run the full behavioural analysis.

    Parameters
    ----------
    constants : dict
        Dictionary containing the constants for the experiment.

    Returns
    -------
    None.

    '''
    # get paths and datasets
    conditions, full_paths, dataset = get_datasets(constants)
    
    # get the trial-wise angular errors for all conditions
    all_behav_data = {}
    for i,c in enumerate(conditions):
        ang_errors = []
        responses = []
        for m in range(constants.PARAMS['n_models']):
            ang_errors.append([])
            responses.append([])
            
            constants.PARAMS['model_number'] = m
            ang_errors[m],responses[m] = get_angular_error(dataset[i],
                                                           constants.PARAMS,
                                                           full_paths[i])
            
        
        all_behav_data[c] = {}
        all_behav_data[c]['ang_errors'] = torch.stack(ang_errors)
        all_behav_data[c]['responses'] = torch.stack(responses)
        all_behav_data[c]['cued_colour'],all_behav_data[c]['uncued_colour'] = \
            get_probed_colour(dataset[i])
        
        # get binned errors
        all_behav_data[c]['bin_edges'], all_behav_data[c]['binned_errors'], \
            all_behav_data[c]['bin_centres'] \
                = bin_errors(constants,all_behav_data[c]['ang_errors'])
        # average across models
        all_behav_data[c]['binned_errors_mean'] = all_behav_data[c]['binned_errors'].mean(0)
        all_behav_data[c]['binned_errors_sem'] = \
            np.std(all_behav_data[c]['binned_errors'],0)/np.sqrt(constants.PARAMS['n_models'])
        # plot
        # plot_err_distr_indiv_models(all_behav_data[c])
        plot_average_err_distr(all_behav_data[c])
        # figpath = constants.PARAMS['FIG_PATH'] \
        #     + 'av_error_distribution_' + c + '.png'
        # plt.savefig(figpath)    
        
        # print descriptive statistics
        print('Descriptive stats for condition ' +c)
        _,_ = get_descr_stats(constants,all_behav_data[c]['ang_errors'])
    
        
    if np.logical_and(constants.PARAMS['experiment_number']==3,
                      constants.PARAMS['cue_validity'] != 1):
        # for probabilistic conditions from expt 3:
        # compare errors on valid and invalid trials
        plot_valid_and_invalid_average(constants,
                                       all_behav_data['valid'],
                                       all_behav_data['invalid'])
        # plot the mixture model parameters fitted in Matlab - if not done yet,
        # comment out
        plot_mixture_model_params_validity(constants)
        
    # save and export data
    # save_path = constants.PARAMS['FULL_PATH'] + 'pca_data/'
    # pickle.dump(all_behav_data,open(save_path+'all_behav_data.pckl'))
    # export_data(constants,conditions,all_behav_data)
    
    return

#%% set up some paths

load_path = constants.PARAMS['FULL_PATH'] + 'pca_data/'

# valid_path = load_path + 'valid_trials/'


# valid_path = load_path + 'valid_trials/' + 'in_range_tempGen/'
valid_path = load_path + 'valid_trials/' + 'out_range_tempGen/'


# test_valid = pickle.load((open(valid_path+'0.pckl','rb')))
test_valid = pickle.load((open(valid_path+'test_dataset.pckl','rb')))

# if constants.PARAMS['var_delays']:
# test_condition = 'trained'
# test_condition = 'in-range'
# test_condition = 'out-of-range'

# test_valid = test_valid[test_condition]


ang_err_valid = []


if constants.PARAMS['condition']!='deterministic':
    invalid_path = load_path + 'invalid_trials/'
    ang_err_invalid = []
    test_invalid = pickle.load((open(invalid_path+'test_dataset.pckl','rb')))


#%% get trial-wise angular errors for all models
ang_errors = []
responses = []

if constants.PARAMS['condition']!='deterministic':
    ang_errors_invalid = []
    responses_invalid = []


for m in np.arange(constants.PARAMS['n_models']):
    ang_errors.append([])
    responses.append([])

    constants.PARAMS['model_number'] = m
    ang_errors[m],responses[m] = get_angular_error(test_valid,constants.PARAMS,valid_path)
    
    if constants.PARAMS['condition']!='deterministic':
        ang_errors_invalid.append([])
        responses_invalid.append([])
    
        ang_errors_invalid[m],responses_invalid[m] = get_angular_error(test_invalid,constants.PARAMS,invalid_path)
    
ang_errors = torch.stack(ang_errors)
responses = torch.stack(responses)
cued_colour,uncued_colour = get_probed_colour(test_valid)

#% save
# retnet.save_data(responses,[],valid_path+'responses')
# retnet.save_data(ang_errors,[],valid_path+'ang_errors')
# retnet.save_data(cued_colour,[],valid_path+'cued_colour')
# retnet.save_data(uncued_colour,[],valid_path+'uncued_colour')



if constants.PARAMS['condition']!='deterministic':
    ang_errors_invalid = torch.stack(ang_errors_invalid)
    responses_invalid = torch.stack(responses_invalid)
    cued_colour_invalid,uncued_colour_invalid = get_probed_colour(test_invalid)
    
    # save
    # retnet.save_data(responses_invalid,[],invalid_path+'responses')
    # retnet.save_data(ang_errors_invalid,[],invalid_path+'ang_errors')
    # retnet.save_data(cued_colour_invalid,[],invalid_path+'cued_colour')
    # retnet.save_data(uncued_colour_invalid,[],invalid_path+'uncued_colour')



#%% plot the angular errors distribution for each model

mean_abs_err = np.empty((constants.PARAMS['n_models'],))
mean_abs_sd = np.empty((constants.PARAMS['n_models'],))

mean_err = np.empty((constants.PARAMS['n_models'],))
kappa_err = np.empty((constants.PARAMS['n_models'],))

if constants.PARAMS['condition']!='deterministic':
    abs_max_err = np.max([np.degrees(ang_errors.abs().max()),np.degrees(ang_errors_invalid.abs().max())])
else:
    abs_max_err = np.degrees(ang_errors.abs().max())

# set the bin range to the absolute max error rounded to the nearest 10th
# b = np.round(abs_max_err,-1) 
b=np.pi
bin_edges = np.linspace(-b,b,10)
pdf_x = np.linspace(-b,b, 100) # x vals for the fitted von Mises pdf


fitted_params = np.empty((constants.PARAMS['n_models'],3)) 
# model / kappa, loc and scale of the von Mises pdf 


bin_centres = []
for i in range(len(bin_edges)-1):
    bin_centres.append(np.mean(bin_edges[i:i+2]))


binned_err_valid = np.empty((constants.PARAMS['n_models'],len(bin_centres)))

# if constants.PARAMS['n_models'] != 10:
#     raise ValueError('Change the number of subplots')

if constants.PARAMS['n_models'] == 10:
    n_rows = 2
elif constants.PARAMS['n_models'] == 20:
    n_rows = 4
elif constants.PARAMS['n_models'] == 30:
    n_rows = 6
else:
    raise ValueError('behav_analysis : adjust number of subplots to match number of models')
fig, axs = plt.subplots(n_rows,5,sharex=True,sharey=True)
for m,ax in enumerate(axs.flat):
    mean_abs_err[m] = np.mean(np.abs(np.degrees(ang_errors[m,:,:].view(-1).numpy())))
    mean_abs_sd[m] = np.std(np.abs(np.degrees(ang_errors[m,:,:].view(-1).numpy())))
    
    
    binned_err_valid[m,:], bins, patches = ax.hist((ang_errors[m,:,:].view(-1).numpy()),
                                bins=bin_edges,
                                density=True,
                                ec='k',
                                fc='lightgrey',
                                label='data')
    # 
    
    # binned_err_valid[m,:], bins = np.histogram(np.degrees(ang_errors[m,:,:].view(-1).numpy()),
    #                            bins=bin_edges,
    #                            density=False)
    
    # binned_err_valid[m,:] = binned_err_valid[m,:] / binned_err_valid[m,:].sum()
    # ax.bar(bins[:-1],binned_err_valid[m,:],width = bins[1]-bins[0])

    # get circular mean and kappa of the errors
    mean_err[m] = np.degrees(wrap_angle(pycircstat.mean(ang_errors[m,:,:].view(-1).numpy())))
    kappa_err[m] = np.degrees(pycircstat.distributions.kappa(ang_errors[m,:,:].view(-1).numpy()))
    
    # fit a vonMises distribution to the error data
    fitted_params[m,0],fitted_params[m,1],fitted_params[m,2] = \
        vonmises.fit(ang_errors[m,:,:].view(-1).numpy(),fscale=1.0)
    # note - need to fix the scale to 1.0 in order to be able to fit the pdf,
    # otherwise it fits some kind of high frequency oscillatory function (can't
    # recover parameters even on simulated data)

    # plot the fitted pdf - would need to fix so that histogram bars sum up to
    # 1, otherwise peak of fitted pdf is on a different scale
    
    pdf_y = vonmises.pdf(pdf_x,
                          fitted_params[m,0],
                          fitted_params[m,1],
                          fitted_params[m,2])
    
    ax.plot(pdf_x, pdf_y, 'r-', lw=2, label='fit')
    
    # x_min = np.min(np.degrees(ang_errors[m,:,:].view(-1).numpy()))
    # x_max = np.max(np.degrees(ang_errors[m,:,:].view(-1).numpy()))

    # pdf_x = np.linspace(x_min,x_max,100)
    # pdf_y = vonmises.pdf(np.radians(pdf_x), kappa = np.radians(kappa_err[m]), loc=np.radians(mean_err[m]))
    
    
    # rv = vonmises(kappa,loc,scale)
    # plt.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
    
    # ax.plot(pdf_x, pdf_y, 'r--', linewidth=2,label='fit')
    
    ax.set_title('Model ' + str(m))
    

fitted_params = pd.DataFrame(fitted_params,columns = ['kappa','mu','scale'])
    
# add mean and std info onto the plots - in a separate loop so that can place 
# it at the same height for every subplot, defined as 0.8 of the y-scale
txt = r'$\mu=%.2f$'+'\n'+r'$\kappa=%.2f$'
txt_xloc = axs[0,0].get_xlim()[0]+ 10# x-location for the text
for m,ax in enumerate(axs.flat):
    if m==9:
        # put it below the legend
        ax.text(txt_xloc,axs[0,0].get_ylim()[1]*.85, txt %(mean_err[m],kappa_err[m]),fontsize='xx-small')
    else:
        ax.text(txt_xloc,axs[0,0].get_ylim()[1]*.85, txt %(mean_err[m],kappa_err[m]),fontsize='xx-small')
    

ax.set_xlabel('Angular error [degrees]')
ax.set_ylabel('Density')
# ax.legend(bbox_to_anchor=(1, 1),loc='upper right')


print('Mean absolute angular error across all models: %.2f' %np.mean(mean_abs_err))
print('SEM: %.2f' %(np.std(mean_abs_err)/np.sqrt(constants.PARAMS['n_models'])))



# #%% plot the histogram from data pooled across all models
# fig, ax2 = plt.subplots(1)

# binned_valid_all, bins, patches = ax2.hist(np.degrees(ang_errors.view(-1).numpy()),
#                                bins=bin_edges,
#                                density=True,
#                                ec='k',
#                                fc='lightgrey',
#                                label='data') 

# ax2.set_xlabel('Angular error [degrees]')
# ax2.set_ylabel('Density')
#%% plot invalid trials

fitted_params_invalid = np.empty((constants.PARAMS['n_models'],3)) 
# model / kappa, loc and scale of the von Mises pdf 
if constants.PARAMS['condition']!='deterministic':
    binned_err_invalid = np.empty((constants.PARAMS['n_models'],len(bin_centres)))

    mean_err_invalid = np.empty((constants.PARAMS['n_models'],))
    kappa_err_invalid = np.empty((constants.PARAMS['n_models'],))
    fig, axs = plt.subplots(n_rows,5,sharex=True,sharey=True)
    for m,ax in enumerate(axs.flat):
        # plt.subplot(2,5,m+1,sharex = fig,sharey = fig)
        binned_err_invalid[m,:], bins, patches = ax.hist((ang_errors_invalid[m,:,:].view(-1).numpy()),
                                    bins=bin_edges,
                                    density=True,
                                    ec='k',
                                    fc='lightgrey',
                                    label='data')
        
        
        mean_err_invalid[m] = np.degrees(pycircstat.mean(ang_errors_invalid[m,:,:].view(-1).numpy()))
        kappa_err_invalid[m] = np.degrees(pycircstat.distributions.kappa(ang_errors_invalid[m,:,:].view(-1).numpy()))
        
        x_min = np.min(np.degrees(ang_errors_invalid[m,:,:].view(-1).numpy()))
        x_max = np.max(np.degrees(ang_errors_invalid[m,:,:].view(-1).numpy()))
        
        fitted_params_invalid[m,0],fitted_params_invalid[m,1],fitted_params_invalid[m,2] = \
            vonmises.fit(ang_errors_invalid[m,:,:].view(-1).numpy(), fscale=1)
        # pdf_x = np.linspace(x_min,x_max,100)
        pdf_y = vonmises.pdf(pdf_x, 
                             fitted_params_invalid[m,0],
                             fitted_params_invalid[m,1],
                             fitted_params_invalid[m,2])        
    
        ax.plot(pdf_x, pdf_y, 'r-', lw=2, label='fit')
    
        
        ax.set_title('Model ' + str(m))
    
    ax.set_xlabel('Angular error [degrees]')
    ax.set_ylabel('Density')
    ax.legend()

fitted_params_invalid = pd.DataFrame(fitted_params_invalid,columns = ['kappa','mu','scale'])

#%% compare the distributions between valid and invalid trials

if  constants.PARAMS['condition']!='deterministic':
    if constants.PARAMS['n_models'] == 1:
        fig, ax = plt.subplots(1)
        m = 0
        ax.plot(bin_centres,binned_err_valid[m,:],
                'g--o',lw = 1, label='valid')
        ax.plot(bin_centres,binned_err_invalid[m,:],
                'r--o',lw = 1,label='invalid')
        
        ax.set_title('Model '+str(m))
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        ax.set_ylabel('Density')
        ax.set_xlabel('Angular error [degrees]')
        
        plt.tight_layout()
    else:
        fig, axs = plt.subplots(2,5,sharex=True,sharey=True)
        for m,ax in enumerate(axs.flat):
            ax.plot(bin_centres,binned_err_valid[m,:],
                    'g--o',lw = 1, label='valid')
            ax.plot(bin_centres,binned_err_invalid[m,:],
                    'r--o',lw = 1,label='invalid')
            
            ax.set_title('Model '+str(m))
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        ax.set_ylabel('Density')
        ax.set_xlabel('Angular error [degrees]')


#%% compare the distributions between valid and invalid trials - plot data and fits

pdf_x = np.linspace(-np.pi,np.pi,100)

if  constants.PARAMS['condition']!='deterministic':
    fig, axs = plt.subplots(2,5,sharex=True,sharey=True)
    for m,ax in enumerate(axs.flat):
        
        ax.plot(bin_centres,binned_err_valid[m,:],
                'o',mfc = 'w', mec = 'g', label='valid')
        ax.plot(bin_centres,binned_err_invalid[m,:],
                'o',mfc = 'w', mec = 'r',label='invalid')
        
        pdf_y_valid = vonmises.pdf(pdf_x,
                          fitted_params.kappa[m],
                          fitted_params.mu[m],
                          fitted_params.scale[m])
        pdf_y_invalid = vonmises.pdf(pdf_x,
                          fitted_params_invalid.kappa[m],
                          fitted_params_invalid.mu[m],
                          fitted_params_invalid.scale[m])
        
        ax.plot(pdf_x,pdf_y_valid,'g-')
        ax.plot(pdf_x,pdf_y_invalid,'r-')
        
        ax.set_title('Model '+str(m))
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax.set_ylabel('Density')
    ax.set_xlabel('Angular error [degrees]')


plt.savefig(constants.PARAMS['FULL_PATH'] + 'validity_effect_all_models_paper.png')
#%% plot grand averages

# fig,ax = plt.subplots(1)

# valid_mean = np.mean(binned_err_valid,0)
# valid_sem = np.std(binned_err_valid,0)/np.sqrt(constants.PARAMS['n_models'])
# ax.plot(bin_centres,valid_mean,color='g',label='valid mean')
# ax.fill_between(bin_centres,valid_mean+valid_sem,valid_mean-valid_sem,color='g',
#                     alpha=.3,label='valid SEM')


# if constants.PARAMS['condition']!='deterministic':
#     invalid_mean = np.mean(binned_err_invalid,0)
#     invalid_sem = np.std(binned_err_invalid,0)/np.sqrt(constants.PARAMS['n_models'])
#     ax.plot(bin_centres,invalid_mean,color='r',label='invalid mean')
#     ax.fill_between(bin_centres,invalid_mean+invalid_sem,invalid_mean-invalid_sem,color='r',
#                         alpha=.3,label='invalid SEM')

# ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# ax.set_xlabel('Angular error [degrees]')
# ax.set_ylabel('Density')


# fig.set_size_inches((10, 5))

#% stats
# from scipy.stats import ttest_ind,wilcoxon,shapiro,ttest_1samp

# __, p_sh = shapiro(mean_err-mean_err_invalid)

# if p_sh<0.05:
#     stat,pval = wilcoxon(mean_err-mean_err_invalid)
#     print('Wilcoxon test for valid and invalid trials: ')
# else:
#     stat,pval = ttest_1samp(mean_err-mean_err_invalid,0)
#     print('T-test for valid vs invalid trials: ')

# print('stat = %.2f, p-val = %.3f' %(stat,pval))

# plt.title(r'$\sigma$' +'= ' + str(np.round(constants.PARAMS['sigma'],4)))
# plt.tight_layout()

# fig_path = constants.PARAMS['FULL_PATH'] + 'err_distr.png'
# plt.savefig(fig_path)

# #%% save data
# if constants.PARAMS['condition']!='deterministic':
#     mean_sem_err_data = {'bin_centres':bin_centres,
#                           'valid_mean':valid_mean,
#                           'valid_sem':valid_sem,
#                           'invalid_mean':invalid_mean,
#                           'invalid_sem':invalid_sem}

# else:
#     mean_sem_err_data = {'bin_centres':bin_centres,
#                           'valid_mean':valid_mean,
#                           'valid_sem':valid_sem,
#                           'binned_err_valid':binned_err_valid,
#                           'mean_err':mean_err,
#                           'kappa_err':kappa_err}
    

#%% plot fits


if constants.PARAMS['cue_validity'] < 1:


    sns.set_context("notebook", font_scale=1.5)
    sns.set_style("ticks")
    
    
    plt.figure()
    fig, ax = plt.subplots()
    
    
    
    # average across models
    valid_mean = binned_err_valid.mean(0)
    invalid_mean = binned_err_invalid.mean(0)
    
    valid_sem = np.std(binned_err_valid,0)/np.sqrt(constants.PARAMS['n_models'])
    invalid_sem = np.std(binned_err_invalid,0)/np.sqrt(constants.PARAMS['n_models'])
    

    
    ax.errorbar(bin_centres,valid_mean,yerr=valid_sem,ls = '',marker= 'o',mfc='w',mec = 'g', ecolor = 'g', label = 'valid')
    ax.errorbar(bin_centres,invalid_mean,yerr=invalid_sem,ls = '',marker= 'o',mfc='w',mec = 'r', ecolor = 'r', label = 'invalid')
    
    
    pdf_x = np.linspace(-np.pi,np.pi,100)
    params_valid = vonmises.fit(ang_errors.view(-1).numpy(),fscale=1)
    params_invalid = vonmises.fit(ang_errors_invalid.view(-1).numpy(),fscale=1)
    
    pdf_y_valid = vonmises.pdf(pdf_x,
                               params_valid[0],params_valid[1],params_valid[2])
    pdf_y_invalid = vonmises.pdf(pdf_x,
                               params_invalid[0],params_invalid[1],params_invalid[2])
    
    ax.plot(pdf_x,pdf_y_invalid,'r-')
    if constants.PARAMS['cue_validity']==.5:
        ax.plot(pdf_x,pdf_y_valid,'g--')
    else:
        ax.plot(pdf_x,pdf_y_valid,'g-')
 
    
    
    
    ax.set_ylabel('Density of trials')
    ax.set_xlabel('Angular error [degrees]')
    # ax.set_title(constants.PARAMS['noise_period'],fontstyle='italic')
    # ax.set_xticks([-np.pi/2,0,np.pi/2])
    # ax.set_xticklabels(labels = ['-'+ r'$\pi$' + '/2','0',r'$\pi$' + '/2'])
    ax.set_xticks([-np.radians(100),0,np.radians(100)])
    ax.set_xticklabels(labels = ['-100','0','100'])
    
    plt.tight_layout()
    
    sns.despine()
    
    figpath = constants.PARAMS['FIG_PATH'] + 'validity_effect_paper.png'
    # plt.savefig(figpath)



if constants.PARAMS['cue_validity']==1:
    sns.set_context("notebook", font_scale=2)
    sns.set_style("ticks")
    
    
    plt.figure()
    
    fig, ax = plt.subplots()
    
    # average across models
    valid_mean = binned_err_valid.mean(0)
    valid_sem = np.std(binned_err_valid,0)/np.sqrt(constants.PARAMS['n_models'])
    
    ax.errorbar(bin_centres,valid_mean,yerr=valid_sem,
                ls = '',marker= 'o',mfc='w',mec = 'k',
                ecolor = 'k', label = 'data')
    pdf_x = np.linspace(-np.pi,np.pi,100)
    params_valid = vonmises.fit(ang_errors.view(-1).numpy(),fscale=1)
    pdf_y_valid = vonmises.pdf(pdf_x,
                               params_valid[0],params_valid[1],params_valid[2])
    
    ax.plot(pdf_x,pdf_y_valid,'k-',label='fit')
    ax.set_ylabel('Density of trials')
    ax.set_xlabel('Angular error [°]')
    ax.set_xticks([-np.radians(100),0,np.radians(100)])
    # ax.set_xticklabels(labels = ['-'+ r'$\pi$' + '/2','0',r'$\pi$' + '/2'])
    ax.set_xticklabels(labels = ['-100','0','100'])
    
    plt.tight_layout()
    
    sns.despine()
    figpath = constants.PARAMS['FIG_PATH'] + 'error_distribution.png'
    plt.savefig(figpath)

# pickle.dump(mean_sem_err_data,open(constants.PARAMS['FULL_PATH']+'mean_sem_err_data.pckl','wb'))


# #%% plot mean + fitted gaussian - to finish

# if constants.PARAMS['condition']=='deterministic':
    
#     valid_mean = np.mean(binned_err_valid,0)
    
#     fig,ax = plt.subplots(1)
#     ax.plot(bin_centres,valid_mean,'ko',markerfacecolor='w')
    
#     # get fitted distr line
    
#     mean_err = np.mean(np.degrees(torch.mean(ang_errors[:,:,:],0).view(-1).numpy()))
#     std_err = np.std(np.degrees(torch.mean(ang_errors[:,:,:],0).view(-1).numpy()))
    
#     x_min = bin_centres[0]
#     x_max = bin_centres[-1]
    
#     pdf_x = np.linspace(x_min,x_max,100)
#     pdf_y = norm.pdf(pdf_x,loc=mean_err,scale=std_err)
#     ax.plot(pdf_x, pdf_y, 'r--', linewidth=2)
    
#     ax.set_title('Model ' + str(m))
#%% export to matlab


def export_data(constants,conditions,all_behav_data):
    '''
    Export data into a Matlab .mat file for the fitting of a mixture model. 
    Only implemented for experiments 1-3.

    Parameters
    ----------
    constants : dict
        DESCRIPTION.
    conditions : list
        DESCRIPTION.
    all_behav_data : dict
        Dictionary with all behavioural data generated under all test datasets.

    Returns
    -------
    None.

    '''
    if constants.PARAMS['experiment_number'] == '3':
        for i,c in enumerate(conditions):
            data_for_matlab = {}
            data_for_matlab['reported_colour'] = all_behav_data[c]['responses'].numpy()
            data_for_matlab['cued_colour'] = all_behav_data[c]['cued_colour'].numpy()
            data_for_matlab['uncued_colour'] = all_behav_data[c]['uncued_colour'].numpy()
            
            
            matlab_path = '/Users/emilia/OneDrive - Nexus365/MATLAB/rnn_retrocue_data/'+\
                  constants.PARAMS['condition'] +'_' + c + '_mixmodel_data.mat'
        
            savemat(matlab_path,data_for_matlab)   
    elif constants.PARAMMS['experiment_number']<3:
        conditions  = ['trained','in-range','out-of-range']
        for i,c in enumerate(conditions):
            data_for_matlab = {}
            data_for_matlab['reported_colour'] = all_behav_data[c]['responses'].numpy()
            data_for_matlab['cued_colour'] = all_behav_data[c]['cued_colour'].numpy()
            data_for_matlab['uncued_colour'] = all_behav_data[c]['uncued_colour'].numpy()
            
            
            matlab_path = '/Users/emilia/OneDrive - Nexus365/MATLAB/rnn_retrocue_data/'+\
                  'expt_' + str(constants.PARAMS['experiment_number']) \
                      +'_' + c + '_mixmodel_data.mat'

        
        
     # if constants.PARAMS['condition']!='deterministic':
     #        data_for_matlab = {}
        
     #        data_for_matlab['reported_colour'] = responses_invalid.numpy()
     #        data_for_matlab['cued_colour'] = cued_colour_invalid.numpy()
     #        data_for_matlab['uncued_colour'] = uncued_colour_invalid.numpy()
            
     #        matlab_path = '/Users/emilia/OneDrive - Nexus365/MATLAB/rnn_retrocue_data/'+\
     #            constants.PARAMS['condition'] +'_invalid_mixmodel_data.mat'
            
     #        savemat(matlab_path,data_for_matlab)
    
    
    
    
    return
    
if constants.PARAMS['experiment'] == 'Buschman_var_delays':
    pickle.dump(ang_errors.view(constants.PARAMS['n_models'],-1),open(valid_path+'ang.errors.pckl','wb'))

# if constants.PARAMS['experiment'] == 'validity_paradigm':
#     data_for_matlab = {}
    
    
    
#     data_for_matlab['reported_colour'] = responses.numpy()
#     data_for_matlab['cued_colour'] = cued_colour.numpy()
#     data_for_matlab['uncued_colour'] = uncued_colour.numpy()
    
#     # matlab_path = '/Users/emilia/OneDrive - Nexus365/MATLAB/rnn_retrocue_data/'+\
#     #     constants.PARAMS['condition'] +'_mixmodel_data.mat'
    
#     matlab_path = '/Users/emilia/OneDrive - Nexus365/MATLAB/rnn_retrocue_data/'+\
#         'var_delays_' + test_condition +'_mixmodel_data.mat'
    
    
#     savemat(matlab_path,data_for_matlab)
    
if constants.PARAMS['experiment_number'] == '3':
    data_for_matlab = {}
    
    data_for_matlab['reported_colour'] = responses.numpy()
    data_for_matlab['cued_colour'] = cued_colour.numpy()
    data_for_matlab['uncued_colour'] = uncued_colour.numpy()
    
    matlab_path = '/Users/emilia/OneDrive - Nexus365/MATLAB/rnn_retrocue_data/'+\
          constants.PARAMS['condition'] +'_mixmodel_data.mat'
    
    
    
    savemat(matlab_path,data_for_matlab)   
    if constants.PARAMS['condition']!='deterministic':
        data_for_matlab = {}
    
        data_for_matlab['reported_colour'] = responses_invalid.numpy()
        data_for_matlab['cued_colour'] = cued_colour_invalid.numpy()
        data_for_matlab['uncued_colour'] = uncued_colour_invalid.numpy()
        
        matlab_path = '/Users/emilia/OneDrive - Nexus365/MATLAB/rnn_retrocue_data/'+\
            constants.PARAMS['condition'] +'_invalid_mixmodel_data.mat'
        
        savemat(matlab_path,data_for_matlab)


# #%%
# # plt.figure()
# # stats = []
# # for i in range(16):
# #     stats.append(torch.std_mean(ang_errors[0,i,:]))
    
# # stats = torch.tensor(stats)
# # plt.plot(stats[:,0],'r*--',label='mean')
# # plt.plot(stats[:,1],'ko--',label='std')


# # plt.figure()
# # plt.plot(kappa_err,'ro',label='observed')
# # plt.plot(kappa_err_t,'k+',label='theoretical')
# # plt.ylabel('Kappa error [°]')
# # plt.xlabel('Model number')
# # plt.title('Kappa of the error distribution, postmean policy implemetation with dot prod, observed versus theoretical')


# #%%
# from scipy.stats import norm


# x = np.linspace(norm.ppf(0.01),
#                   norm.ppf(0.99), 100)

# y = norm.pdf(x)

# plt.figure()
# plt.plot(x,y)

# y /= y.sum()

# post_mean = y @ x

# x1 = np.linspace(norm.ppf(0.01,loc=1),
#                   norm.ppf(0.99,loc=1), 100)
# y1 = norm.pdf(x1,loc=1)

# # y1 /= y1.sum()
# # post_mean = y1 @ x1

# # distort the distribution
# ixs = np.random.randint(len(y1),size=(10,))
# noise = np.abs(np.random.randn(len(ixs)))

# y1[ixs] = noise

# y1 /= y1.sum()

# plt.figure()
# plt.plot(x1,y1)

# post_mean = y1 @ x1

# choices = np.random.choice(x1, p = y1)

# # ax.plot(x, norm.pdf(x),

# #%%

# plt.figure()
# n_trials = responses.shape[1]
# targets_scalars = torch.cat((test_valid['c1'][:n_trials//2],test_valid['c2'][n_trials//2:]))

# for m in range(constants.PARAMS['n_models']):
#     plt.subplot(2,5,m+1)
#     plt.plot(np.degrees(responses[m,:]),label='choices')
#     plt.plot(np.degrees(targets_scalars),label='targets')
    
    
# #%%

# plt.figure()
# for m in range(constants.PARAMS['n_models']):
#     plt.subplot(2,5,m+1)
#     plt.plot(np.degrees(ang_errors[m,:,:].view(-1)))
    
    
# #%%
# plt.figure()
# for c in range(3):
#     plt.subplot(3,1,c+1)
#     # get the colours from the 'middle' to avoid wrapping issue
#     c_ix = c
#     plt.plot(np.degrees(torch.reshape(responses,(ang_errors.shape))[m,c_ix,:]),
#               label='choices')
#     plt.plot(np.degrees(torch.reshape(targets_scalars,(ang_errors.shape[1],ang_errors.shape[-1]))[c_ix,:]),
#               label='targets')
    


def plot_mixture_model_params_validity(constants):
    
    load_path = '/Users/emilia/OneDrive - Nexus365/MATLAB/rnn_retrocue_data/'
    fig_path = constants.PARAMS['BASE_PATH'] + 'data_vonMises/experiment_3/'
    params = ['K','pT','pNT','pU']
    trial_type = ['valid','invalid']
    
    n_models = constants.PARAMS['n_models']
    
    data_reshaped = []
    for p in range(len(params)):
        # load data
        data = pd.read_csv(load_path + params[p]+'_table.csv')
        # reshape data for seaborn
        condition = np.tile(data['condition'].to_numpy(),2)
        t_type = [[trial_type[t]]*n_models*2 for t in range(len(trial_type))]
        t_type = np.reshape(t_type,-1)
        label1 = params[p]+'_'+trial_type[0]
        label2 = params[p]+'_'+trial_type[1]
        
        param_vals = np.concatenate((data[label1].to_numpy(),data[label2].to_numpy()))
        data_reshaped = np.stack((t_type,condition,param_vals),1)
        data_reshaped = pd.DataFrame(data_reshaped,columns=['trial_type','condition',params[p]])
        model_nr = np.tile(np.arange(n_models*2),2)
        
        g = sns.catplot(x='trial_type',y =params[p],hue='condition', data=data_reshaped,
                    kind="point",markers=["^", "o"],units=model_nr,dodge=True,ci=68,
                    palette=sns.color_palette("Set2")[2:0:-1])
        g.figure.set_size_inches((5.6,5))
        
        plt.xlabel('trial type')
        
        # plt.savefig(fig_path+'mix_model_'+params[p]+'.png')
    

def plot_mixture_model_params_vardelays(constants):
    load_path = '/Users/emilia/OneDrive - Nexus365/MATLAB/rnn_retrocue_data/'
    params = ['K','pT','pNT','pU']    

    for p in range(len(params)):
        # load data
        data = pd.read_csv(load_path + 'var_delays_'+params[p]+'_table.csv')
        
        sns.catplot(x='condition',y =params[p], data=data,color='k',
                    kind="point",ci=68)
        
        plt.xlabel('delay length')
        plt.savefig(constants.PARAMS['FIG_PATH']+'mix_model_'+params[p]+'.png')
        
    return


def compare_temp_gen(constants):
    valid_path = constants.PARAMS['FULL_PATH'] + 'pca_data/' + 'valid_trials/'
    
    