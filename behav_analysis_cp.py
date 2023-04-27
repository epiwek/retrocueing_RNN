#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 18:51:06 2021

@author: emilia
"""


import numpy as np
import torch
import matplotlib.pyplot as plt

import pickle
import constants
from scipy.stats import vonmises,norm
import pycircstat

#%% define functions

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

def vec_to_angle(vec):
    if len(vec.shape)<2:
        angles = np.arctan2(vec[1],vec[0])
    else:
        angles = np.arctan2(vec[1,:],vec[0,:])
    return angles

def get_choices(outputs,params,load_path):
    """
    Convert the distributed patterns of activity in the output layer into 
    choices (defined as the angle in colour space corresponding to the centre 
    of the tuning curve).
    
    Parameters
    ----------
    outputs : array-like (batch_size, output channels)
        output layer activation values from the model of interest
    
    params : dictionary 
       
    load_path : str

    Returns
    -------
    choices: array-like (batch_size,)
    
    """
    # get tuning curve centres
    phi  = torch.linspace(-np.pi, np.pi, params['n_colCh']+1)[:-1]
    
    # convert output activation values into choice probabilities
    choice_probs = outputs / torch.sum(outputs,-1).unsqueeze(-1)
    
    # get choices on each trial - posterior mean
    choices = posterior_mean(phi,choice_probs)

    return choices
    
def posterior_mean(angles,probs,**kwargs):
    """
    
    """
    n_trials = probs.shape[0]
    
    # convert the angles to vectors to calculate the circular mean
    angles_vectors = angle_to_vec(angles)
    angles_vectors = torch.stack(([angles_vectors]*n_trials),1)
    mean_vectors = torch.sum(torch.tensor(angles_vectors)*probs,-1)
    
    # convert back into angles
    if len(mean_vectors.shape)<2:
        # for vectors
        mean_angles = np.arctan2(mean_vectors[1],mean_vectors[0])
    else:
        # for matrices
        mean_angles = np.arctan2(mean_vectors[1,:],mean_vectors[0,:])
    
    return mean_angles
    
def get_angular_error(test_dataset,params,load_path):
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
    choices = get_choices(outputs,params,load_path)
    
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

#%% set up paths

load_path = constants.PARAMS['FULL_PATH'] + 'pca_data/'

valid_path = load_path + 'valid_trials/'
test_valid = pickle.load((open(valid_path+'0.pckl','rb')))
ang_err_valid = []


if constants.PARAMS['condition']!='deterministic':
    invalid_path = load_path + 'invalid_trials/'
    ang_err_invalid = []
    test_invalid = pickle.load((open(invalid_path+'0.pckl','rb')))


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
fig, axs = plt.subplots(2,5,sharex=True,sharey=True)
# bin_edges = np.linspace(-60,60,60)
b = 180
bin_edges = np.linspace(-b,b,b//10)

bin_centres = []
for i in range(len(bin_edges)-1):
    bin_centres.append(np.mean(bin_edges[i:i+2]))


binned_err_valid = np.empty((constants.PARAMS['n_models'],len(bin_centres)))

for m,ax in enumerate(axs.flat):
    # plt.subplot(2,5,m+1,sharex = fig,sharey = fig)
    mean_abs_err[m] = np.mean(np.abs(np.degrees(ang_errors[m,:,:].view(-1).numpy())))
    mean_abs_sd[m] = np.std(np.abs(np.degrees(ang_errors[m,:,:].view(-1).numpy())))
    
    
    binned_err_valid[m,:], bins, patches = ax.hist(np.degrees(ang_errors[m,:,:].view(-1).numpy()),
                               bins=bin_edges,
                               density=True,
                               ec='k',
                               fc='lightgrey',
                               label='data')
    
    # get circular mean and kappa of the errors
    mean_err[m] = np.degrees(wrap_angle(pycircstat.mean(ang_errors[m,:,:].view(-1).numpy())))
    kappa_err[m] = np.degrees(pycircstat.distributions.kappa(ang_errors[m,:,:].view(-1).numpy()))
    
    
    # x_min = np.min(np.degrees(ang_errors[m,:,:].view(-1).numpy()))
    # x_max = np.max(np.degrees(ang_errors[m,:,:].view(-1).numpy()))

    # pdf_x = np.linspace(x_min,x_max,100)
    # pdf_y = vonmises.pdf(np.radians(pdf_x), kappa = np.radians(kappa_err[m]), loc=np.radians(mean_err[m]))
    
    
    # ax.plot(pdf_x, pdf_y, 'r--', linewidth=2,label='fit')
    
    ax.set_title('Model ' + str(m))
    

    
# add mean and std info onto the plots - in a separate loop so that can place 
# it at the same height for every subplot, defined as 0.8 of the y-scale
txt = r'$\mu=%.2f$'+'\n'+r'$\kappa=%.2f$'
txt_xloc = axs[0,0].get_xlim()[0]+ 0.1# x-location for the text
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



#%% plot invalid trials

if constants.PARAMS['condition']!='deterministic':
    binned_err_invalid = np.empty((constants.PARAMS['n_models'],len(bin_centres)))

    mean_err_invalid = np.empty((constants.PARAMS['n_models'],))
    kappa_err_invalid = np.empty((constants.PARAMS['n_models'],))
    fig, axs = plt.subplots(2,5,sharex=True,sharey=True)
    for m,ax in enumerate(axs.flat):
        # plt.subplot(2,5,m+1,sharex = fig,sharey = fig)
        binned_err_invalid[m,:], bins, patches = ax.hist(np.degrees(ang_errors_invalid[m,:,:].view(-1).numpy()),
                                   bins=bin_edges,
                                   density=True,
                                   ec='k',
                                   fc='lightgrey',
                                   label='data')
        
        
        mean_err_invalid[m] = np.degrees(pycircstat.mean(ang_errors_invalid[m,:,:].view(-1).numpy()))
        kappa_err_invalid[m] = np.degrees(pycircstat.distributions.kappa(ang_errors_invalid[m,:,:].view(-1).numpy()))
        
        x_min = np.min(np.degrees(ang_errors_invalid[m,:,:].view(-1).numpy()))
        x_max = np.max(np.degrees(ang_errors_invalid[m,:,:].view(-1).numpy()))
        
        pdf_x = np.linspace(x_min,x_max,100)
        pdf_y = vonmises.pdf(pdf_x, kappa = kappa_err_invalid[m], loc=mean_err_invalid[m])
        ax.plot(pdf_x, pdf_y, 'r--', linewidth=2,label='fit')
        
        ax.set_title('Model ' + str(m))
    
    ax.set_xlabel('Angular error [degrees]')
    ax.set_ylabel('Density')
    ax.legend()

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


#%% do a 1STT for a single model
# from scipy import stats
# m=0
# # subsample valid trials if probabilistic
# if constants.PARAMS['condition']=='probabilistic':
#     n_valid = len(ang_errors[m,:,:].view(-1).numpy())
#     n_invalid = len(ang_errors_invalid[m,:,:].view(-1).numpy())
#     valid_subsamp_ix = torch.randperm(n_valid)[:n_invalid]
#     # results will be slightly different each time this is run - because of the random subsampling of trials
#     tstat,pval = stats.ttest_1samp(ang_errors[m,:,:].view(-1).numpy()[valid_subsamp_ix]-ang_errors_invalid[m,:,:].view(-1).numpy(),0)
    
#     lstat,lpval = stats.levene(ang_errors[m,:,:].view(-1).numpy()[valid_subsamp_ix],ang_errors_invalid[m,:,:].view(-1).numpy(),center='mean')
# elif constants.PARAMS['condition']=='neutral':
#     tstat,pval = stats.ttest_1samp(ang_errors[m,:,:].view(-1).numpy()-ang_errors_invalid[m,:,:].view(-1).numpy(),0)
#     lstat,lpval = stats.levene(ang_errors[m,:,:].view(-1).numpy(),ang_errors_invalid[m,:,:].view(-1).numpy(),center='mean')
    
#%% plot grand average

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

plt.tight_layout()

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

plt.title('RMSprop, '+ r'$\sigma$' +'= ' + str(np.round(constants.PARAMS['epsilon'],4))+\
          ', '+r'$\alpha$' +'= ' + str(constants.PARAMS['learning_rate']))
# fig_path = constants.PARAMS['FULL_PATH'] + 'err_distr.png'
# plt.savefig(fig_path)

#%% save data
# if constants.PARAMS['condition']!='deterministic':
#     mean_sem_err_data = {'bin_centres':bin_centres,
#                          'valid_mean':valid_mean,
#                          'valid_sem':valid_sem,
#                          'invalid_mean':invalid_mean,
#                          'invalid_sem':invalid_sem}

# else:
#     mean_sem_err_data = {'bin_centres':bin_centres,
#                          'valid_mean':valid_mean,
#                          'valid_sem':valid_sem,
#                          'binned_err_valid':binned_err_valid,
#                          'mean_err':mean_err,
#                          'kappa_err':kappa_err}
    

# pickle.dump(mean_sem_err_data,open(constants.PARAMS['FULL_PATH']+'mean_sem_err_data.pckl','wb'))


#%% plot mean + fitted gaussian - to finish

if constants.PARAMS['condition']=='deterministic':
    
    valid_mean = np.mean(binned_err_valid,0)
    
    fig,ax = plt.subplots(1)
    ax.plot(bin_centres,valid_mean,'ko',markerfacecolor='w')
    
    # get fitted distr line
    
    mean_err_all = np.degrees(pycircstat.mean(ang_errors))
    kappa_err_all = np.degrees(pycircstat.distributions.kappa(ang_errors.view(-1).numpy()))[0]
    
    x_min = bin_centres[0]
    x_max = bin_centres[-1]
    
    pdf_x = np.linspace(x_min,x_max,100)
    
    # pdf_y = vonmises.pdf(np.radians(pdf_x),np.radians(kappa_err_all))#,np.radians(mean_err_all),1)*np.max(valid_mean)
    pdf_y = norm.pdf(pdf_x,mean_err_all,kappa_err_all)
    ax.plot(pdf_x, pdf_y, 'r--', linewidth=2)

#%% export to matlab
# from scipy.io import savemat
# data_for_matlab = {}

# data_for_matlab['reported_colour'] = responses.numpy()
# data_for_matlab['cued_colour'] = cued_colour.numpy()
# data_for_matlab['uncued_colour'] = uncued_colour.numpy()

# matlab_path = '/Users/emilia/OneDrive - Nexus365/MATLAB/rnn_retrocue_data/'+\
#    constants.PARAMS['condition'] +'_mixmodel_data.mat'

# savemat(matlab_path,data_for_matlab)


# if constants.PARAMS['condition']!='deterministic':
#     data_for_matlab = {}

#     data_for_matlab['reported_colour'] = responses_invalid.numpy()
#     data_for_matlab['cued_colour'] = cued_colour_invalid.numpy()
#     data_for_matlab['uncued_colour'] = uncued_colour_invalid.numpy()
    
#     matlab_path = '/Users/emilia/OneDrive - Nexus365/MATLAB/rnn_retrocue_data/'+\
#        constants.PARAMS['condition'] +'_invalid_mixmodel_data.mat'
    
#     savemat(matlab_path,data_for_matlab)



