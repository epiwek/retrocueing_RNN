#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 17:48:18 2021

@author: emilia
"""
import numpy as np
import os.path
from pathlib import Path
import sys
import torch
import pickle
from scipy.stats import chi2, pearsonr, shapiro


def check_path(path):
    '''
    Check if path exists; if not, create it.

    Parameters
    ----------
    path : str
        Path to be created.

    Returns
    -------
    None.

    '''
    # old implementation - does not create paths recursively
    # if not (os.path.exists(path)):
    #     os.mkdir(path)
    #     print('Path created')
    
    # better implementation - allows recursive path creation
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
        
   
def check_file_exists(path_to_file):
    '''
    Check if a file exists.

    Parameters
    ----------
    path_to_file : str
        Path to file.

    Returns
    -------
    exists : TYPE
        DESCRIPTION.

    '''
    exists = os.path.exists(path_to_file)
    return exists



def sort_by_uncued(data_struct,params,key='data'):
    '''
    Sort trials by the uncued stimulus colour.

    Parameters
    ----------
    data_struct : dict
        Data structure, containing the keywords:
            - ['data'] : data matrix in the shape of (n_trials,n_neurons) or 
                            (n_trials,n_timepoints,n_neurons)
            - ['labels'] : including 'c1' (colour 1) and 'c2'
    params : dict
        Experiment parameters.
    key : str, optional
         Keyword for the data_struct dictionary, under which the sorted data 
         will be stored. The default is 'data', which overrides the unsorted 
         dataset entry in the input dictionary.

    Returns
    -------
    data_struct : dict
        Data structure including the sorted dataset.

    '''
    if len(data_struct['data'].shape) == 2:
        # if data only contains a single timepoint, insert an extra dimension
        data_struct[key] = data_struct[key].unsqueeze(1)
        
    n_trials = data_struct['data'].shape[0]
    loc1_ix = np.arange(n_trials//2) # trials where loc1 was cued
    loc2_ix = np.arange(n_trials//2,n_trials)
    
    # labels_uncued = np.concatenate((eval_data["labels"]["c2"][loc1_ix],
    #                           eval_data["labels"]["c1"][loc2_ix]))
    loc1_ix_sorted = []
    loc2_ix_sorted = []
    
    colour_space = torch.linspace(-np.pi, np.pi, params['n_stim']+1)[:-1] # possible colours

    for c in range(len(colour_space)):
        loc1_ix_sorted.append(np.where(data_struct["labels"]["c2"][loc1_ix]==colour_space[c])[0])
        loc2_ix_sorted.append(np.where(data_struct["labels"]["c1"][loc2_ix]==colour_space[c])[0])
    
    loc1_ix_sorted = np.array(loc1_ix_sorted)
    loc2_ix_sorted = np.array(loc2_ix_sorted)
    loc2_ix_sorted += n_trials//2
    loc1_ix_sorted = np.reshape(loc1_ix_sorted,(-1))
    loc2_ix_sorted = np.reshape(loc2_ix_sorted,(-1))
    full_sorting_ix = np.concatenate((loc1_ix_sorted,loc2_ix_sorted))

    # data_struct = data_struct[key][full_sorting_ix,:,:].squeeze()
    data_struct[key] = data_struct[key][full_sorting_ix,:,:].squeeze()
    # update labels
    data_struct['labels']['c1'] = data_struct['labels']['c1'][full_sorting_ix]
    data_struct['labels']['c2'] = data_struct['labels']['c2'][full_sorting_ix]
    data_struct['labels']['loc'] =  data_struct['labels']['loc'][:,full_sorting_ix,:]
    
    return data_struct, full_sorting_ix


def sort_labels(labels):
    '''
    Get the indices that will sort a labels array.
    
    Parameters
    ----------
    labels : array (N,)
        Array with labels to be sorted. N corresponds to number of trials.

    Returns
    -------
    sorted_ix : array (N,)
        Index that will sort the labels array.
    labels_sorted : array (N,)
        Sorted labels array.
    '''
    
    
    # labels_uncued = np.concatenate((eval_data["labels"]["c2"][loc1_ix],
    #                           eval_data["labels"]["c1"][loc2_ix]))
    sorted_ix = []
    labels_sorted = []
    

    for label in np.unique(labels):
        sorted_ix.append(np.where(labels==label)[0])
        labels_sorted.append(labels[np.where(labels==label)[0]].numpy())
    
    sorted_ix = np.array(sorted_ix).flatten()
    labels_sorted = np.array(labels_sorted).flatten()

    return sorted_ix, labels_sorted



def bin_data(data,params,custom_B=np.nan):
    '''
    Bin the trials into a specified number of colour bins. Trials need to be 
    pre-sorted by the labels; i.e., for N repeats of a trial type, the first N
    trials must correspond to label 1, the next N to label 2, etc. As a default,
    number of colour bins is extracted from the experiment parameters (params 
    dictionary). A custom value can be requested by passing a custom_B 
    parameter. The size of the first dimension of the resulting (binned) data 
    array will equal M = L*B, corresponding to B colour bins at L locations,
    where L is extracted from the params dictionary.

    Parameters
    ----------
    data : array (n_trials,n_neurons) or (n_trials,n_timepoints,n_neurons)
        Dataset to be binned.
    params : dict
        Experiment parameters.
    custom_B : int
        Optional. Custom number of colour bins. Default is NaN which reads in B
        from the params dictionary.

    Returns
    -------
    data_binned : array (M,n_neurons)
        Binned dataset.

    '''
    if np.isnan(custom_B):
        M = params['M']
    else:
        M = custom_B * params['L']
    # bin into M colour bins
    n_samples = data.shape[0]//M
    if len(data.shape) == 2:
        # matrix with single timepoint
        data_binned = data.unsqueeze(0).reshape((M,n_samples,params['n_rec']))
    elif len(data.shape) == 3:
        # matrix with multiple timepoints
        n_timepoints = data.shape[1]
        data_binned = data.unsqueeze(0).reshape((M,n_samples,
                                    n_timepoints,params['n_rec']))
        
    data_binned = torch.from_numpy(data_binned.numpy().mean(1))
    
    return data_binned


def bin_labels(labels_list,n_bins):
    '''
    Bin the trial labels into n_bins categories. Trials do not need to be 
    pre-sorted by the labels.

    Parameters
    ----------
    labels_list : array (n_trials,)
        Trial-wise labels.
    n_bins : int
        Number of bins.

    Raises
    ------
    ValueError
        If the number of unique labels is not divisible by n_bins.

    Returns
    -------
    labels_list_binned : array (n_trials,)
        Trial-wise binned labels.

    '''
    labels = np.unique(labels_list)
    n_labels = len(labels)
    
    if (n_labels % n_bins != 0):
        raise ValueError('N labels must be divisible by N bins')
    
    bin_over = n_labels//n_bins # n old labels per bin
    # new label assignments for each old albe
    binned_labels = np.array([np.arange(n_bins)]*bin_over).reshape(-1,order='F')
    
    # recode the old labels into integers to use as ixs for binned_labels
    labels_list_integer = np.array([np.where(labels==val)[0] \
                            for ix,val in enumerate(np.array(labels_list))])
    # get the trial-wise new, binned labels
    labels_list_binned = np.squeeze(np.array([binned_labels[val] \
                            for ix,val in enumerate(np.array(labels_list_integer))]))
        
    return labels_list_binned


def equal_axes(ax):
    """
    Set the x, y and z-axis ranges to the same values.

    Parameters
    ----------
    ax : matplotlib Axes object
        Plot axes object.

    Returns
    -------
    None.

    """
    # move into a plotting script when get a chance
    ax_lims = np.array(ax.xy_viewLim)
    ax.set_xlim3d(np.min(ax_lims),np.max(ax_lims))
    ax.set_ylim3d(np.min(ax_lims),np.max(ax_lims))
    ax.set_zlim3d(np.min(ax_lims),np.max(ax_lims))
    

def transfer_to_cpu(constants):
    """
    Transfer training data to the CPU for analysis.

    Returns
    -------
    None.

    """
    
    # imports allow the function to be easily used outside of the main.py script
    import retrocue_model as retnet

    device = torch.device('cpu')
    path = constants.PARAMS['FULL_PATH']    
    
    for m in np.arange(constants.PARAMS['n_models']):
        constants.PARAMS['model_number'] = m
        
        # load data
        track_training = pickle.load(open(path+'training_data/'+'training_data_model'+str(m)+'.pckl','rb'))
        
        for key,value in track_training.items():
            track_training[key] = track_training[key].to(device)
            
        # save on cpu
        retnet.save_data(track_training, constants.PARAMS, f"{path}training_data/training_data_model")
        print(f"Model {m} done")


#%% circular statistics methods

def circ_diff(angle1,angle2):
    '''
    Calculate the difference between two angles, constrained to lie within [-pi,pi].

    Parameters
    ----------
    angle1 : float
        Value for angle 1 in radians
    angle2 : float
        Value for angle 2 in radians.

    Returns
    -------
    angle_diff
        angle 1 - angle 2.

    '''
    return wrap_angle(angle1-angle2)


def wrap_angle(angle):
    """
    Wrap angles to be within [-pi,pi]

    Parameters
    ----------
    angle : float
        Angle value in radians.

    Returns
    -------
    angle_wrapped : float
        Angle value after wrapping to the [-pi,pi] range.

    """
    return (angle + torch.tensor(np.pi)) % (2*torch.tensor(np.pi)) - torch.tensor(np.pi)

# def wrap_angle_0360(angle):
#     return (angle) % (2*torch.tensor(np.pi))


def circ_mean(angles):
    '''
    Calculate the circular mean of a set of angles.

    Parameters
    ----------
    angles : array
        Angles sample array. If multidimensional, will be flattened before computing the mean.

    Returns
    -------
    circ_mean : float
        Circular mean of the angles.

    '''

    if len(angles.shape) > 1:
        # flatten
        angles = angles.view(-1)
    # convert the angles into x- and y-coordinates on the unit circle, then take their cartesian means
    cart_mean = angle_to_vec(angles).mean(-1)
    # convert back into angles
    circ_mean = vec_to_angle(cart_mean)
    return circ_mean


def angle_to_vec(angles):
    """
    Helper function to convert an array of angles into their unit-circle 
    vector representations. 
    
    Parameters
    ----------
    angles : array
    
    Returns
    -------
    angles_vectors : array
    """
    angles_vectors = torch.stack((torch.cos(angles), torch.sin(angles)))
    return angles_vectors


def vec_to_angle(vec):
    '''
    Convert an angle from a unit-circle vector representation into radians.

    Parameters
    ----------
    vec : array (2,)
        2D unit vector corresponding to the angle.

    Returns
    -------
    angle : float
        Angle in radians, defined on the [-pi,pi] interval.

    '''
    return torch.atan2(vec[1], vec[0])


def corrcl(circ_var,l_var):
    '''
    Calculate the correlation coefficient between a circular and linear variable.
    Removes NaNs from the dataset.

    Parameters
    ----------
    circ_var : array (n_samples,)
        vector containing the circular variable entries in radians.
    l_var : array (n_samples,)
        vector containing the linear variable entries.

    Returns
    -------
    rcl : float
        Correlation coefficient.
    p_val : float
        Probability value.

    '''
    # get rid of nans
    nan_ix = np.where(np.isnan(l_var))[0]
    clean_ix = np.setdiff1d(np.arange(len(circ_var)),nan_ix)
    
    # get the sin and cos of circular samples
    sines = np.sin(circ_var[clean_ix])
    cosines = np.cos(circ_var[clean_ix])
    
    # calculate the partial correlation coefficient
    rcx,p1 = pearsonr(cosines,l_var[clean_ix])
    rsx,p2 = pearsonr(sines,l_var[clean_ix])
    rcs,p3 = pearsonr(sines,cosines)
    
    # calcuate the full correlation coefficient
    rcl = np.sqrt((rcx**2 + rsx**2 - 2*rcx*rsx*rcs)/(1-rcs**2))
    
    # calculate the test statistic and check significance value
    test_stat = len(clean_ix)*rcl**2
    df = 2
    p_val = 1 - chi2.cdf(test_stat,df)
    
    return rcl, p_val

