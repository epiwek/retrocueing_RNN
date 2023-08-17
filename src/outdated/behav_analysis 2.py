#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 18:51:06 2021

@author: emilia
"""


import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

import pickle
# import constants
from scipy.stats import vonmises, pearsonr, spearmanr, shapiro, linregress

import seaborn as sns
import pycircstat

import pdb

from scipy.io import savemat

#%% define functions
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
        'posterior_mean' or 'softmax' (default)

    Returns
    -------
    choices: array-like (batch_size,)
        chosen stimulus colour defined as angle [rad] in circular colour space
    
    """
    # get tuning curve centres
    
    phi  = torch.linspace(-np.pi, np.pi, params['n_colCh']+1)[:-1]
    # convert output activation values into choice probabilities
    # choice_probs = outputs / torch.sum(outputs,-1).unsqueeze(-1)
    if params['loss_fn'] == 'CEL':
        softmax = torch.nn.Softmax(dim=-1)
        choice_probs = softmax(outputs)
    else:
        choice_probs = outputs
    n_trials = choice_probs.shape[0]
    # get choices on each trial
    if policy == 'softmax':
        # softmax policy
        # need to do it in a for loop bc np.random.choice() only accepts 1d arrays for p
        choices = torch.empty((n_trials,))
        for i in range(n_trials):
                # normalise the choice probabilities again
                # this is to avoid the numerical precision error numpy throws when
                # 'probabilities do not sum up to 1'
                
                p_vec = choice_probs[i,:]
                p_vec = torch.tensor(p_vec,dtype= torch.float64)
                p_vec /= p_vec.sum()
                choices[i] = torch.tensor(np.random.choice(phi, p = p_vec))
    elif policy == 'posterior_mean':
        # posterior mean policy
        # choices = choice_probs @ phi
        choices = posterior_mean(phi,choice_probs)
    elif policy == 'hardmax':
        ix = np.argsort(choice_probs,1)[:,-1]
        choices = torch.empty((n_trials,))
        for i in range(n_trials):
            choices[i] = phi[ix[i]]
    return choices


def posterior_mean(angles,probs,**kwargs):
    """
    Convert the distributed patterns of activity in the output layer into 
    choices using a specified policy.
    
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


def angle_to_vec(angles):
    """
    Helper function to convert an array of angles into their unit-circle 
    vector representations. 
    
    Parameters
    ----------
    angles : array-like ()
    
    Returns
    -------
    angles_vectors : array-like ()
    """
    angles_vectors = torch.stack((np.cos(angles),np.sin(angles)))
    return angles_vectors


def get_angular_error(test_dataset,params,load_path,constants):
    """
    Get the distribution of errors for each colour stimulus for a given model.
    Errors are calculated for a given colour, irrespective of the location 
    where it was shown. Error is defined as the absolute of the difference 
    between output layer and target activation values, averaged across all 
    stimulus colours and trials.
    
    Parameters
    ----------
    test_dataset : dictionary
    
    params : dictionary 
       
    load_path : str

    Returns
    -------
    mean_ang_error : torch.Tensor (n output channels, n trials per cued colour)
        error values distributed across the output layer channels, referenced 
        to the original stimulus activation pattern
    responses : torch.Tensor
        trial-wise choices
        
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
    
    
def mixture_model(choices,cued_colour,uncued_colour):
    
    return

def get_probed_colour(test_dataset):  
    # get indices of trials where loc1 and loc2 were probed
    loc1_ix = np.array(test_dataset['loc'][0,:,:].squeeze(),dtype=bool)
    loc2_ix = np.array(test_dataset['loc'][1,:,:].squeeze(),dtype=bool)
    # get probed colour for each trial
    probed_colour = torch.cat((test_dataset['c1'][loc1_ix],test_dataset['c2'][loc2_ix]))
    unprobed_colour = torch.cat((test_dataset['c2'][loc1_ix],test_dataset['c1'][loc2_ix]))
    return probed_colour,unprobed_colour

def wrap_angle(angle):
    """
    Wraps angle(s) to be within [-pi, pi).
    
    Parameters
    ----------
    angle : array-like
        angle in radians

    Returns
    -------
    angle_wrapped : array-like
    """
    angle_wrapped = (angle+np.pi) % (2*np.pi) - np.pi
    return angle_wrapped
 


#%% set up some paths

# load_path = constants.PARAMS['FULL_PATH'] + 'pca_data/'

# valid_path = load_path + 'valid_trials/'


# # valid_path = load_path + 'valid_trials/' + 'in_range_tempGen/'
# # valid_path = load_path + 'valid_trials/' + 'out_range_tempGen/'


# test_valid = pickle.load((open(valid_path+'0.pckl','rb')))
# if constants.PARAMS['var_delays']:
#     # test_condition = 'trained'
#     test_condition = 'in-range'
#     # test_condition = 'out-of-range'
#     # test_valid = test_valid['trained']
#     # test_valid = test_valid['in-range']
#     test_valid = test_valid[test_condition]


# # ang_err_valid = []


# if constants.PARAMS['condition']!='deterministic':
#     invalid_path = load_path + 'invalid_trials/'
#     ang_err_invalid = []
#     test_invalid = pickle.load((open(invalid_path+'0.pckl','rb')))


def get_errors(constants,test_valid,valid_path):
    # rewrite so works for both valid and invalid
    #% get trial-wise angular errors for all models
    ang_errors = []
    responses = []
    
    # if constants.PARAMS['condition']!='deterministic':
    #     ang_errors_invalid = []
    #     responses_invalid = []
    
    
    for m in np.arange(constants.PARAMS['n_models']):
        ang_errors.append([])
        responses.append([])
    
        constants.PARAMS['model_number'] = m
        ang_errors[m],responses[m] = get_angular_error(test_valid,constants.PARAMS,valid_path,constants)
        
        # if constants.PARAMS['condition']!='deterministic':
        #     ang_errors_invalid.append([])
        #     responses_invalid.append([])
        
        #     ang_errors_invalid[m],responses_invalid[m] = get_angular_error(test_invalid,constants.PARAMS,invalid_path,constants)
        
    ang_errors = torch.stack(ang_errors)
    responses = torch.stack(responses)
    cued_colour,uncued_colour = get_probed_colour(test_valid)

    #% save
    # retnet.save_data(responses,[],valid_path+'responses')
    # retnet.save_data(ang_errors,[],valid_path+'ang_errors')
    # retnet.save_data(cued_colour,[],valid_path+'cued_colour')
    # retnet.save_data(uncued_colour,[],valid_path+'uncued_colour')



    # if constants.PARAMS['condition']!='deterministic':
    #     ang_errors_invalid = torch.stack(ang_errors_invalid)
    #     responses_invalid = torch.stack(responses_invalid)
    #     cued_colour_invalid,uncued_colour_invalid = get_probed_colour(test_invalid)
    
    # save
    # retnet.save_data(responses_invalid,[],invalid_path+'responses')
    # retnet.save_data(ang_errors_invalid,[],invalid_path+'ang_errors')
    # retnet.save_data(cued_colour_invalid,[],invalid_path+'cued_colour')
    # retnet.save_data(uncued_colour_invalid,[],invalid_path+'uncued_colour')
    
    return ang_errors, responses, cued_colour, uncued_colour



def plot_distrib_indivModels(constants,ang_errors,ang_errors_invalid):
    #% plot the angular errors distribution for each model
    
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
    
    
    fig, axs = plt.subplots(2,5,sharex=True,sharey=True)
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



def plot_hist_pooled_data(constants,ang_errors,bin_edges):
    #% plot the histogram from data pooled across all models
    fig, ax2 = plt.subplots(1)
    
    binned_valid_all, bins, patches = ax2.hist(np.degrees(ang_errors.view(-1).numpy()),
                                   bins=bin_edges,
                                   density=True,
                                   ec='k',
                                   fc='lightgrey',
                                   label='data') 
    
    ax2.set_xlabel('Angular error [degrees]')
    ax2.set_ylabel('Density')
    
    
def plot_invalid_trials(constants,ang_errors_invalid,bin_edges,bin_centres):
    #% plot invalid trials
    
    fitted_params_invalid = np.empty((constants.PARAMS['n_models'],3)) 
    # model / kappa, loc and scale of the von Mises pdf 
    if constants.PARAMS['condition']!='deterministic':
        binned_err_invalid = np.empty((constants.PARAMS['n_models'],len(bin_centres)))
    
        mean_err_invalid = np.empty((constants.PARAMS['n_models'],))
        kappa_err_invalid = np.empty((constants.PARAMS['n_models'],))
        fig, axs = plt.subplots(2,5,sharex=True,sharey=True)
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


def compare_distributions(constants,binned_err_valid,binned_err_invalid,bin_centres):
    #% compare the distributions between valid and invalid trials
    
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


def compare_distributions_data_and_fits(constants,binned_err_valid,binned_err_invalid,bin_centres,fitted_params,fitted_params_invalid):
    #% compare the distributions between valid and invalid trials - plot data and fits
    
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
    

def plot_grand_averages(constants,binned_err_valid,binned_err_invalid,bin_centres):
    #% plot grand averages
    
    fig,ax = plt.subplots(1)
    
    valid_mean = np.mean(binned_err_valid,0)
    valid_sem = np.std(binned_err_valid,0)/np.sqrt(constants.PARAMS['n_models'])
    ax.plot(bin_centres,valid_mean,color='g',label='valid mean')
    ax.fill_between(bin_centres,valid_mean+valid_sem,valid_mean-valid_sem,color='g',
                        alpha=.3,label='valid SEM')
    
    
    if constants.PARAMS['condition']!='deterministic':
        invalid_mean = np.mean(binned_err_invalid,0)
        invalid_sem = np.std(binned_err_invalid,0)/np.sqrt(constants.PARAMS['n_models'])
        ax.plot(bin_centres,invalid_mean,color='r',label='invalid mean')
        ax.fill_between(bin_centres,invalid_mean+invalid_sem,invalid_mean-invalid_sem,color='r',
                            alpha=.3,label='invalid SEM')
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax.set_xlabel('Angular error [degrees]')
    ax.set_ylabel('Density')
    
    
    fig.set_size_inches((10, 5))

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

    # plt.title(r'$\sigma$' +'= ' + str(np.round(constants.PARAMS['epsilon'],4)))
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
    


def plot_grand_averages_fits(constants,binned_err_valid,binned_err_invalid,bin_centres,ang_errors,ang_errors_invalid):
    #% plot fits
    
    
    if constants.PARAMS['condition']!='deterministic':
    
    
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
        if constants.PARAMS['condition']=='probabilistic':
            ax.plot(pdf_x,pdf_y_valid,'g-')
        elif constants.PARAMS['condition']=='neutral':
            ax.plot(pdf_x,pdf_y_valid,'g--')
        
        
        
        ax.set_ylabel('Density')
        ax.set_xlabel('Angular error [radians]')
        ax.set_title(constants.PARAMS['noise_period'],fontstyle='italic')
        ax.set_xticks([-np.pi/2,0,np.pi/2],labels = ['-'+ r'$\pi$' + '/2','0',r'$\pi$' + '/2'])
        
        plt.tight_layout()
        
        sns.despine()
        
        figpath = constants.PARAMS['FULL_PATH'] + 'validity_effect_paper.png'
        plt.savefig(figpath)
    
    
    
    if constants.PARAMS['condition']=='deterministic':
        sns.set_context("notebook", font_scale=1.5)
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
        ax.set_ylabel('Density')
        ax.set_xlabel('Angular error [radians]')
        ax.set_xticks([-np.pi/2,0,np.pi/2],labels = ['-'+ r'$\pi$' + '/2','0',r'$\pi$' + '/2'])
        
        plt.tight_layout()
        
        sns.despine()
        figpath = constants.PARAMS['FULL_PATH'] + 'error_distribution.png'
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

def export_to_matlab(responses,cued_colour,uncued_colour,test_condition):
    #% export to matlab
    data_for_matlab = {}
    
    data_for_matlab['reported_colour'] = responses.numpy()
    data_for_matlab['cued_colour'] = cued_colour.numpy()
    data_for_matlab['uncued_colour'] = uncued_colour.numpy()
    
    # matlab_path = '/Users/emilia/OneDrive - Nexus365/MATLAB/rnn_retrocue_data/'+\
    #     constants.PARAMS['condition'] +'_mixmodel_data.mat'
    
    matlab_path = '/Users/emilia/OneDrive - Nexus365/MATLAB/rnn_retrocue_data/'+\
        'var_delays_' + test_condition +'_mixmodel_data.mat'
    
    
    savemat(matlab_path,data_for_matlab)
    

 
#%% run var delays

def get_errors_variable_delays(constants):
    load_path = constants.PARAMS['FULL_PATH'] + 'pca_data/valid_trials/'
    test_valid = pickle.load((open(load_path+'0.pckl','rb')))
    
    if constants.PARAMS['experiment_number']==2:
        test_conditions = ['trained','in-range','out-of-range', 'out-of-range-shorter']
        folders = ['','in_range_tempGen/','out_range_tempGen/','out_range_tempGen_shorter/']
    else:
        test_conditions = ['trained','in-range','out-of-range']
    
    data_for_matlab = {}
    
    for i,key in enumerate(test_conditions):
        # test_condition = 'trained'
        # test_condition = 'in-range'
        # test_condition = 'out-of-range'
        # test_valid = test_valid['trained']
        # test_valid = test_valid['in-range']
        test_valid = test_valid[key]
        load_path = constants.PARAMS['FULL_PATH'] + 'pca_data/valid_trials/'  + folders[i]
        
        #+ test_conditions[i]+'/'
        
        
        ang_errors, responses, cued_colour, uncued_colour = get_errors(constants,test_valid,load_path)
        






def plot_mixture_model_params_validity(constants):
    
    load_path = '/Users/emilia/OneDrive - Nexus365/MATLAB/rnn_retrocue_data/'
    fig_path = constants.PARAMS['BASE_PATH'] + 'data_vonMises/MSELoss_custom/validity_paradigm/'
    params = ['K','pT','pNT','pU']
    trial_type = ['valid','invalid']
    
    n_models = constants.PARAMS['n_models']
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
        
        sns.catplot(x='trial_type',y =params[p],hue='condition', data=data_reshaped,
                    kind="point",markers=["^", "o"],units=model_nr,dodge=True,ci=68,
                    palette=sns.color_palette("Set2")[2:0:-1])
        
        plt.xlabel('trial type')
        
        plt.savefig(fig_path+'mix_model_'+params[p]+'.png')
    

def plot_mixture_model_params_vardelays(constants):
    load_path = '/Users/emilia/OneDrive - Nexus365/MATLAB/rnn_retrocue_data/'
    fig_path = constants.PARAMS['BASE_PATH'] + 'data_vonMises/MSELoss_custom/validity_paradigm/'
    params = ['K','pT','pNT','pU']
    trial_type = ['trained','in-range','out-of-range']
    
    n_models = constants.PARAMS['n_models']

    # for p in range(len(params)):
        
        
    return


def corr_CDI_and_mixture_model_params_validity(constants,CDI):
    CDI_probed_diff_valid = CDI[:,3]-CDI[:,5]
    CDI_probed_diff_invalid = CDI[:,4]-CDI[:,6]
    
    probed_benefit_diff_norm = (CDI_probed_diff_valid - CDI_probed_diff_invalid) / CDI_probed_diff_valid
    s1,p1 = shapiro(probed_benefit_diff_norm)
    
    load_path = '/Users/emilia/OneDrive - Nexus365/MATLAB/rnn_retrocue_data/'
    params = ['K','pT','pNT','pU']
    trial_type = ['valid','invalid']
    
    all_data = np.zeros((constants.PARAMS['n_models'],1+len(params)))
    all_data_labels = [p+'_diff' for p in params]
    all_data_labels.insert(0,'norm CDI benefit')
    all_data[:,0] = probed_benefit_diff_norm
    
    for pp,param in enumerate(params):
        # load data
        data = pd.read_csv(load_path + param +'_table.csv')
        
        label1 = param +'_'+trial_type[0] # valid
        label2 = param +'_'+trial_type[1] # invalid
        
        # extract the validity = 75% data
        prob_ix = np.where(data['condition']=='probabilistic')[0]
        param_valid = data[label1][prob_ix]
        param_invalid = data[label2][prob_ix]
        
        param_diff = param_valid - param_invalid
        all_data[:,pp+1] = param_diff
        
        
        s2, p2 = shapiro(param_diff)
        # correlate with the difference in CDI between 
        if np.logical_and(p1>.05,p2>.05):
            # pearson corr
            r,p = pearsonr(param_diff,probed_benefit_diff_norm)
            txt = r'$r_{p}$ = '
        else:
            # spearman corr
            r,p = spearmanr(param_diff,probed_benefit_diff_norm)   
            txt = r'$r_{s}$ = '
        
        # get significance 
        if p.round(3) < 0.001:
            p_print = 0.001
            p_op = '<'
        else:
            p_print = p.round(3)
            p_op = '='
            
        txt+='{r:.2f}, \n p {p_op} {p:.3f}'.format(r=r,p_op=p_op,p=p_print)
        
        
        # calculate best linear fit for plotting
        lin_fit =  linregress(param_diff,probed_benefit_diff_norm)
        
        plt.figure()
        plt.plot(param_diff,probed_benefit_diff_norm,'o',mec = 'k', mfc = [0.75,0.75,0.75])
        plt.xlabel(param)
        plt.ylabel('normalised CDI benefit')
        plt.ylim([0,plt.ylim()[1]])
        

        if np.logical_or(param=='K',param=='pT'):
            plt.xlim([-0.001,plt.xlim()[1]])
        
        # add line of best fit
        x_min,x_max = plt.xlim()
        x_vals = np.linspace(x_min+x_max*.02,x_max*.98)
        y_vals = lin_fit.intercept + lin_fit.slope*x_vals
        plt.plot(x_vals,y_vals,'r')
        
        # x_max = plt.xlim()[1]
        y_max = plt.ylim()[1]
        plt.text(x_max*.7,y_max*.6,txt)
        plt.tight_layout()
        
        plt.savefig(constants.PARAMS['FIG_PATH']+'corr_CDI_benefit_'+param+'.png')
        plt.savefig(constants.PARAMS['FIG_PATH']+'corr_CDI_benefit_'+param+'.svg')
    
    #save data to csv
    df = pd.DataFrame(all_data,columns=all_data_labels)
    df.to_csv(constants.PARAMS['FULL_PATH']+'pca_data/CDI_norm_benefit_and_mixture_model_params.csv')
