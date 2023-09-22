#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 22:07:39 2021

@author: emilia
"""
import random
import numpy as np
import torch
from torch import nn, optim
import pickle
import os
from scipy.ndimage import gaussian_filter1d
from scipy.io import savemat
from src.generate_data_von_mises import change_cue_validity
import src.helpers as helpers
from src.generate_data_von_mises import generate_test_conditions


def seed_torch(seed=1029):
    """
    Set the seed for all packages to ensure reproducibility.

    Parameters
    ----------
    seed : int, optional
        Seed value. The default is 1029.

    Returns
    -------
    None.

    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


class RNN(nn.Module):
    def __init__(self, params, device):
        super(RNN, self).__init__()
        # PARAMETERS
        self.n_rec = params['n_rec']  # number of recurrent neurons
        self.n_inp = params['n_inp']
        self.n_out = params['n_out']
        self.device = device
        self.noise_sigma = params['sigma']
        self.noise_distr = params['noise_distr']
        self.noise_timesteps = params['noise_timesteps']

        # set seed
        torch.manual_seed(params['model_number'])

        # LAYERS
        # note: We need to add noise after the Wrec @ ht-1 step, but before the nonlinearity.
        # So, we cannot use the torch RNN class, because it does it all in one step.

        # input layer
        self.inp = nn.Linear(self.n_inp, self.n_rec)
        self.inp.weight = nn.Parameter(self.inp.weight * params['init_scale'])  # Xavier init
        self.inp.bias = nn.Parameter(self.inp.bias * params['init_scale'])  # Xavier init

        # recurrent layer
        self.Wrec = nn.Parameter(torch.nn.init.orthogonal_(torch.empty((self.n_rec, self.n_rec))))  # orthogonal init
        self.relu = nn.ReLU()

        # output layer
        self.out = nn.Linear(self.n_rec, self.n_out)  # output layer
        self.out.weight = nn.Parameter(self.out.weight * params['init_scale'])  # Xavier init
        self.out.bias = nn.Parameter(self.out.bias * params['init_scale'])  # Xavier init
        self.softmax = nn.Softmax(dim=-1)

    def step(self, input_ext, hidden, noise):
        """
        Run the RNN for one timestep.
        """
        hidden = self.relu(self.inp(input_ext.unsqueeze(0)) + hidden @ self.Wrec.T + noise)
        h = hidden.clone().detach()
        # We need to detach the hidden state to be able to save it into a matrix
        # in the forward method, otherwise it messes with the computational graph.

        return h, hidden

    def forward(self, inputs):
        """
        Run the RNN with the input time course.
        """

        # Add noise to hidden units
        seq_len = inputs.shape[0]
        batch_size = inputs.shape[1]
        # To add hidden noise, need to use the expanded implementation below:

        # Initialize network state
        # hidden states from current time point
        hidden = torch.zeros((1, inputs.size(1), self.n_rec), device=self.device)  # 0s
        # hidden states from all timepoints
        h = torch.empty((seq_len, batch_size, self.n_rec), device=self.device)

        # Run the input through the network - across time
        for timepoint in range(seq_len):
            if len(np.where(self.noise_timesteps == timepoint)[0]) > 0:
                # Add Gaussian noise to appropriate timesteps of the trial
                noise = (torch.randn(hidden.size(), device=self.device)) * self.noise_sigma
            else:
                # timestep without noise
                noise = torch.zeros(hidden.size(), device=self.device)

            h[timepoint, :, :], hidden = self.step(inputs[timepoint, :, :], hidden, noise)

        # pass the recurrent activation from the last timestep through the decoder layer and apply softmax
        output = self.out(hidden)
        output = self.softmax(output)
        return output.squeeze(), h, hidden


def sample_choices(outputs, params, policy='softmax'):
    """
    Convert the output layer activity (corresponding to the choice probabilities for all possible colour responses) into
    trial-wise choices using a specified policy.

    Parameters
    ----------
    outputs : array-like (batch_size, n output channels)
        output layer activation values (corresponding to the choice probabilities) from the model of interest

    params : dictionary
        Experimental parameters.

    policy : str
        'hardmax', or 'softmax' (default)

    Returns
    -------
    choices: array-like (batch_size,)
        chosen stimulus colour defined as an angle [rad] in circular colour space

    """
    # get tuning curve centres
    phi = torch.linspace(-np.pi, np.pi, params['n_colCh'] + 1)[:-1]
    n_trials = outputs.shape[0]

    # sample choices for all trials
    if policy == 'softmax':
        # softmax policy - sample the choices proportionally to their respective probabilities
        # need to do it in a for loop because np.random.choice() only accepts 1d arrays for parameter p
        choices = torch.empty((n_trials,))
        for i in range(n_trials):
            # normalise the choice probabilities again
            # this is to avoid the numerical precision error numpy throws when 'probabilities do not sum up to 1'
            p_vec = outputs[i, :]
            p_vec = torch.tensor(p_vec, dtype=torch.float64)
            p_vec /= p_vec.sum()
            # sample choices
            choices[i] = torch.tensor(np.random.choice(phi, p=p_vec))
    elif policy == 'hardmax':
        # hardmax policy - pick the choice associated with the highest probability
        ix = np.argsort(outputs, 1)[:, -1]
        choices = torch.empty((n_trials,))
        for i in range(n_trials):
            choices[i] = phi[ix[i]]
    else:
        raise ValueError('Not a valid policy name. Please pick one of the following: softmax, hardmax')
    return choices


def train_model(params, data, device):
    """
    Train the RNN model and save it, along with the training details.

    Parameters
    ----------
    params : dict
        Experiment parameters.
    data : dict
        Training data dictionary.
    device : torch.device()
        Device on which to train the model.


    Returns
    -------
    model : torch object
        Trained model.
    track_training : dict
        Training data.

    """

    # set seed for reproducibility
    torch.manual_seed(params['model_number'])

    # initialise model
    model = RNN(params, device)

    # transfer model to the desired device
    model.to(device)

    # set the optimiser
    optimizer = optim.RMSprop(model.parameters(), lr=params['learning_rate'])

    # pre-allocate some tensors
    n_valid_trials = int(params['cue_validity'] * params['stim_set_size'])
    n_invalid_trials = params['stim_set_size'] - n_valid_trials

    loss_all = torch.empty(params['stim_set_size'],
                           params['n_epochs']).to(device)
    loss_valid = torch.empty(n_valid_trials,
                             params['n_epochs']).to(device)
    loss_epoch = torch.empty((params['n_epochs'],)).to(device)
    shuffling_order = torch.empty((params['stim_set_size'],
                                   params['n_epochs']),
                                  dtype=torch.long).to(device)
    net_outputs = torch.empty((params['stim_set_size'],
                               params['n_colCh'],
                               params['n_epochs'])).to(device)

    if params['condition'] != 'deterministic':
        invalid_trials = torch.empty((params['n_epochs'], n_invalid_trials)).to(device)
    else:
        invalid_trials = None

    # extract the inputs and targets and put on the device
    inputs_base = data['inputs']
    inputs_base = inputs_base.to(device)
    targets = data['targets']
    targets = targets.to(device)

    # set the convergence criterion parameters
    window = params['conv_criterion']['window']
    epochs = range(params['n_epochs'])

    # loop over epochs
    for ix, i in enumerate(epochs):
        # print('Epoch %d : ' %i)
        # shuffle dataset for SGD
        shuffling_order[:, i] = \
            torch.randperm(params['stim_set_size'], dtype=torch.long).to(device)

        if params['var_delays']:
            # Experiment 3: variable delay condition
            # determine the delay durations on each trial
            delay_mask = \
                var_delay_mask(params['delay_mat'][shuffling_order[:, i], :],
                               params)
            delay_mask.to(device)
        else:
            delay_mask = None

        if params['condition'] != 'deterministic':
            # make some cues invalid
            inputs = inputs_base.clone()  # create a copy of the inputs
            inputs, ixs = change_cue_validity(inputs, params)  # change some trials
            # save the ixs
            invalid_trials[i, :] = torch.tensor(ixs)
        else:
            inputs = inputs_base

        # loop over training examples
        for trial in range(params['stim_set_size']):
            if params['var_delays']:
                trial_input = inputs[delay_mask[:, trial], shuffling_order[trial, i], :]
            else:
                trial_input = inputs[:, shuffling_order[trial, i], :]

            outputs, o, hidden = \
                model(trial_input.unsqueeze(1))

            # print the mean and sd of the hidden activity on the last timestep on the first forward pass
            # if np.logical_and(trial == 0, i == 0):
            #     print('First forward pass')
            #     print('    Means:')
            #     print(torch.std_mean(o, -1)[0].squeeze())
            #     print('    S.d.:')
            #     print(torch.std_mean(o, -1)[1].squeeze())

            # Compute loss
            loss = custom_MSE_loss(params, outputs, targets[shuffling_order[trial, i]])

            # Keep track of outputs and loss
            loss_all[trial, i] = loss.detach()
            net_outputs[trial, :, i] = outputs.detach()
            # Compute gradients
            optimizer.zero_grad()
            loss.backward()
            # Update weights
            optimizer.step()

        # save loss into array
        if params['condition'] != 'deterministic':
            valid_ix = torch.from_numpy(np.setdiff1d(np.arange(params['stim_set_size']), invalid_trials[i, :]))
        else:
            valid_ix = torch.arange(params['stim_set_size'])

        if len(valid_ix) != n_valid_trials:
            raise ValueError('loss_valid has a wrong pre-allocated size!')

        loss_valid[:, i] = loss_all[valid_ix, i]

        # mean over epoch - only valid trials
        loss_epoch[i] = loss_all[valid_ix, i].mean()

        # print progress
        # for non-deterministic conditions, only show loss on valid trials
        if ((ix * 100 / params['n_epochs']) % 25) == 0:
            print('Model %2d :    %.2f%% iterations of SGD completed...loss = %.5f' \
                  % (int(params['model_number']),
                     100 * (ix + 1) / params['n_epochs'],
                     loss_epoch[i]))
            # print_progress(i, params['n_epochs'])
        if (ix == params['n_epochs'] - 1):
            print('Model %2d :    100%% iterations of SGD completed...loss = %.5f' \
                  % (int(params['model_number']),
                     loss_epoch[i]))

        # check if convergence criterion satisfied
        if params['criterion_type'] == 'abs_loss':
            criterion_reached = (loss_epoch[i] <= params['MSE_criterion'])
        elif params['criterion_type'] == 'loss_der':
            if i < window:
                criterion_reached = False
            else:
                criterion_reached = \
                    apply_conv_criterion(params, loss_epoch[i - window + 1:i + 1])
        else:
            raise ValueError('Specify which convergence criterion to use')

        # if loss reaches the required level, stop training
        if criterion_reached:
            print('Model %2d converged after %d epochs. End loss = %.5f' % (int(params['model_number']), i + 1,
                                                                            loss_epoch[i]))
            # prune preallocated tensors
            net_outputs = net_outputs[:, :, :i + 1]
            loss_all = loss_all[:, :i + 1]
            shuffling_order = shuffling_order[:, :i + 1]
            # dLoss = dLoss[:i+1]
            # loss_clean = loss_clean[:i+1]
            loss_epoch = loss_epoch[:i + 1]
            break

    # save training data
    track_training = {'loss': loss_all, 'loss_valid': loss_valid, 'loss_epoch': loss_epoch,
                      'shuffling_order': shuffling_order, 'outputs': net_outputs}
    # track_training['dLoss'] = dLoss
    # track_training['loss_clean'] = loss_clean

    if params['condition'] != 'deterministic':
        track_training['invalid_trials'] = invalid_trials

    return model, track_training


def save_model(path, params, model, track_training):
    """
    Save the torch model along with the training data.

    Parameters
    ----------
    path : str
        Path to the main data folder.
    params : dict
        Experiment parameters.
    model : torch object
        Model to save.
    track_training : dict
        Training data.

    Returns
    -------
    None.

    """
    # check if paths exist - if not, create them

    model_path = path + 'saved_models/'
    data_path = path + 'training_data/'

    helpers.check_path(model_path)
    helpers.check_path(data_path)

    # save model
    torch.save(model, f"{model_path}model{params['model_number']}")
    torch.save(model.state_dict(), f"{model_path}model{str(params['model_number'])}_statedict")
    # save training data, including loss
    with open(f"{data_path}training_data_model{ str(params['model_number'])}.pckl", 'wb') as f:
        pickle.dump(track_training, f)

    print('Model saved')


def load_model(path, params, device):
    """
    Load model from file.

    Parameters
    ----------
    path : str
        Path to model file.
    params : dict
        Experiment parameters, incuding model number.
    device : torch obj
        Device to put them model on.

    Returns
    -------
    model : torch obj
        RNN model.

    """

    if device.type == 'cuda':
        model = RNN(params, device)
        model.load_state_dict(torch.load(f"{path}model{str(params['model_number'])}_statedict"))
    else:
        # for some reason cannot load models straight from file with:
        # model = torch.load(path + 'model' + str(params['model_number']), map_location=torch.device('cpu'))
        # so replaced it with:
        model = RNN(params, device)
        model.load_state_dict(torch.load(f"{path}model{str(params['model_number'])}_statedict"))

    print('.... Loaded')
    return model


def add_delay_keywords(data_dict, params):
    """
    Add additional keywords to the existing data dictionary data_dict. Keys correspond to the names of the delay
    intervals, in the format 'delay{delay_number}' and contain only the data from the end timepoint of the corresponding
    delay interval.

    :param dict data_dict: Data dictionary with 'data' key, containing the data array of shape (m, n_timepoints, n)
    :param dict params: Dictionary with Experiment parameters.
    :return: data_dict
    """
    # Extract the end-points of the delay intervals
    d1_ix = params['trial_timepoints']['delay1_end'] - 1
    d2_ix = params['trial_timepoints']['delay2_end'] - 1
    data_dict['delay1'] = data_dict['data'][:, d1_ix, :]
    data_dict['delay2'] = data_dict['data'][:, d2_ix, :]

    if params['experiment_number'] == 4:
        d3_ix = params['trial_timepoints']['delay3_end'] - 1
        data_dict['delay3'] = data_dict['data'][:, d3_ix, :]

    return data_dict


def get_pca_data_labels(averaged_across, trial_type, params):
    """
    Get the location labels for pca_data structures, in a 'labels' dictionary.

    For experiments 1-3, the dictionary will contain a single 'loc' keu, containing the cued or uncued location labels,
    depending on the condition. For experiment 4, the dictionary will contain 'cued' and 'probed' keys, containing
    the corresponding location labels.

    :param str averaged_across: Name of the item the data was averaged across. E.g., for the data containing binned
        activation responses to the cued items, this will be the 'uncued' item. Choose from: 'uncued', 'cued',
        'single_up', and 'single_down'.
    :param str trial_type: Name of the trial type. Choose from 'valid' and 'invalid'.
    :param dict params: Dictionary of Experiment parameters.
    :return: labels: dictionary of labels
    """
    # colour labels
    labels = {'col': np.concatenate([np.arange(params['B'])] * 2, 0)}
    # location labels
    if averaged_across == 'uncued':
        # location of the cued item
        labels['loc'] = np.zeros((params['M'],), dtype=int)
        labels['loc'][params['B']:] = 1  # second half of trials
    elif averaged_across == 'cued':
        # location of the uncued item
        labels['loc'] = np.zeros((params['M'],), dtype=int)
        labels['loc'][:params['B']] = 1  # first half of trials
    elif averaged_across == 'single_up':
        # location of the cued (and uncued) item
        labels['loc'] = np.zeros((params['M'],), dtype=int)
    elif averaged_across == 'single_down':
        # location of the cued (and uncued) item
        labels['loc'] = np.ones((params['M'],), dtype=int)

    if params['experiment_number'] == 4:
        # create cued and probed colour labels
        if trial_type == 'valid':
            if averaged_across == 'single_up':
                labels['cued_loc'] = np.zeros(params['B']*2, dtype=int)
                labels['probed_loc'] = np.zeros(params['B']*2, dtype=int)
            elif averaged_across == 'single_down':
                labels['cued_loc'] = np.ones(params['B']*2, dtype=int)
                labels['probed_loc'] = np.ones(params['B']*2, dtype=int)
            else:
                labels['cued_loc'] = np.concatenate((np.zeros((params['B'],), dtype=int),
                                                     np.ones((params['B'],), dtype=int)))
                labels['probed_loc'] = np.concatenate((np.zeros((params['B'],), dtype=int),
                                                       np.ones((params['B'],), dtype=int)))
        else:
            if averaged_across == 'single_up':
                labels['cued_loc'] = np.ones(params['B'] * 2, dtype=int)
                labels['probed_loc'] = np.zeros(params['B'] * 2, dtype=int)
            if averaged_across == 'single_up':
                labels['cued_loc'] = np.zeros(params['B'] * 2, dtype=int)
                labels['probed_loc'] = np.ones(params['B'] * 2, dtype=int)
            else:
                labels['cued_loc'] = np.concatenate((np.ones((params['B'],), dtype=int),
                                                     np.zeros((params['B'],), dtype=int)))
                labels['probed_loc'] = np.concatenate((np.zeros((params['B'],), dtype=int),
                                                       np.ones((params['B'],), dtype=int)))

    return labels


def format_pca_data(pca_data, averaged_across, trial_type, params):
    """
    Format the pca_data dictionary. Add a 'labels' key, as well as delay name keys (e.g. 'delay1'), containing the data
    from the endpoint of the appropriate delay interval.

    :param dict pca_data: Data dictionary.
    :param str averaged_across: Name of the item the data was averaged across. E.g., for the data containing binned
        activation responses to the cued items, this will be the 'uncued' item. Choose from: 'uncued', 'cued',
        'single_up', and 'single_down'.
    :param str trial_type: Name of the trial type. Choose from 'valid' and 'invalid'.
    :param dict params: Dictionary of Experiment parameters.
    :return: pca_data: Reformatted dictionary
    """
    # create labels
    pca_data['labels'] = get_pca_data_labels(averaged_across, trial_type, params)
    # add delay-specific fields
    pca_data = add_delay_keywords(pca_data, params)

    return pca_data


def eval_model(model, test_data, params, save_path, trial_type='valid'):
    """
    Evaluate model on the test dataset after freezing weights and save results to files.
    :param torch.object model:  Trained pytorch model.
    :param dict test_data: Test dataset.
    :param dict params: Experimental parameters.
    :param str save_path: Path for saving data.
    :param str trial_type: Optional. Trial type label, relevant for experiment 3. Set to either 'valid' or 'invalid'.
        Default is valid.
    :return: eval_data, pca_data_all, model_outputs: data dictionaries
    """

    assert trial_type in ['valid', 'invalid'], "Trial type must be 'valid' or 'invalid'"

    if trial_type == 'invalid':
        assert params['cue_validity'] < 1, 'Invalid trial evaluation only implemented for probabilistic cue conditions.'

    model.to(torch.device('cpu'))  # put model on cpu

    # 1. Evaluate model on the test data after freezing the weights
    model.eval()
    with torch.no_grad():
        readout, hidden_all_timepoints, hidden_T = \
            model(test_data['inputs'])
        # hidden_T corresponds to the hidden layer activity on the last timepoint only

    # 2.  Create the full (unaveraged) test dataset dictionary
    eval_data = {"dimensions": ['trial', 'time', 'n_rec'],
                 "data": hidden_all_timepoints.permute(1, 0, -1),
                 "labels":
                     {"loc": test_data['loc'],
                      "c1": test_data['c1'],
                      "c2": test_data['c2']}}
    if params['experiment_number'] == 4:
        # add cued and probed labels
        eval_data['labels']['cued_loc'] = test_data['cued_loc']
        eval_data['labels']['probed_loc'] = test_data['probed_loc']

    # # Save the full test dataset dictionary
    if params['experiment_number'] == 4:
        if trial_type == 'valid':
            save_data(eval_data, save_path + 'eval_data_model', params['model_number'])
        else:
            save_data(eval_data, save_path + 'eval_data_uncued_model', params['model_number'])
    else:
        # save
        save_data(eval_data, save_path + 'eval_data_model', params['model_number'])

    # 3. Create pca_data: dataset binned by cued colour and averaged across uncued colours Data is sorted by the cued
    # colour (probed if experiment 3) automatically when created, so no need to resort, only bin.
    trial_data = helpers.bin_data(params, eval_data['data'])
    # M = B*L - colour-bin*location combinations
    pca_data = {'dimensions': ['M', 'time', 'n_rec'],
                'data': trial_data}
    # add the labels keyword (cued location and colour), as well as dictionary keywords: 'delay1', 'delay2' and
    # 'delay3' (if applicable), containing the data from the endpoints of the delay intervals only.
    pca_data = format_pca_data(pca_data, 'uncued', trial_type, params)

    # save
    if params['experiment_number'] == 4:
        save_data(pca_data, save_path + 'pca_data_probed_model', params['model_number'])
        if trial_type == 'valid':
            save_data(pca_data, save_path + 'pca_data_model', params['model_number'])
        else:
            save_data(pca_data, save_path + 'pca_data_uncued_model', params['model_number'])
    else:
        save_data(pca_data, save_path + 'pca_data_model', params['model_number'])

    # 4. Create pca_uncued: like above, but averaged across cued colours and binned across uncued colours
    # sort and bin
    eval_data_uncued, full_sorting_ix = helpers.sort_by_uncued(eval_data, params)
    trial_data_uncued = helpers.bin_data(params, eval_data_uncued['data'])

    pca_data_uncued = {'dimensions': ['M', 'n_rec'],
                       'data': trial_data_uncued}
    # format the dictionary - add labels and delay-specific keywords
    pca_data_uncued = format_pca_data(pca_data_uncued, 'cued', trial_type, params)
    #save
    if params['experiment_number'] == 4:
        save_data(pca_data_uncued, save_path + 'pca_data_unprobed_model', params['model_number'])
        if trial_type == 'valid':
            save_data(pca_data_uncued, save_path + 'pca_data_uncued_model', params['model_number'])
            save_data(eval_data_uncued, save_path + 'eval_data_uncued_model', params['model_number'])
        else:
            save_data(pca_data_uncued, save_path + 'pca_data_model', params['model_number'])
            save_data(eval_data_uncued, save_path + 'eval_data_model', params['model_number'])
    else:
        save_data(pca_data_uncued, save_path + 'pca_data_uncued_model', params['model_number'])

    # 5. Create similar dictionaries for the Cued/Uncued geometry. These will contain the cued and uncued-averaged
    # data from trials where the 'up' and 'down' locations were cued, respectively.
    # For example, for the 'cued up' trials, the first half of rows will contain the binned activity patterns for the
    # cued (up) items, and the other half for the uncued (down) items.
    if trial_type == 'valid':
        pca_data_cued_up_uncued_down = {'data': torch.cat((pca_data['data'][:params['B'], :, :],
                                                           pca_data_uncued['data'][:params['B'], :, :]))}
        # format the dictionary - add labels and delay-specific keywords
        pca_data_cued_up_uncued_down = format_pca_data(pca_data_cued_up_uncued_down, 'single_up', trial_type, params)
        save_data(pca_data_cued_up_uncued_down, save_path + 'pca_data_cued_up_uncued_down_model', params['model_number'])

        pca_data_cued_down_uncued_up = {'data': torch.cat((pca_data['data'][params['B']:, :, :],
                                                           pca_data_uncued['data'][params['B']:, :, :]))}
        # format the dictionary - add labels and delay-specific keywords
        pca_data_cued_down_uncued_up = format_pca_data(pca_data_cued_down_uncued_up, 'single_down', trial_type, params)
        save_data(pca_data_cued_down_uncued_up, save_path + 'pca_data_cued_down_uncued_up_model', params['model_number'])

    else:
        delay2_end_ix = params['trial_timepoints']['delay2_end']

        # need to use different labels - pca_data corresponds to data binned by the uncued item, pca_uncued corresponds
        # to the cued item
        # note this data is binned according to the *cued*/*uncued* item up to the second cue / probe timepoint, and
        # then according to the *probed*/*unprobed* item for the remaining timepoints

        cued_up = torch.cat((pca_data_uncued['data'][params['B']:, :delay2_end_ix, :],
                             pca_data['data'][params['B']:, delay2_end_ix:, :]), dim=1)
        uncued_down = torch.cat((pca_data['data'][params['B']:, :delay2_end_ix, :],
                                 pca_data_uncued['data'][params['B']:, delay2_end_ix:, :]), dim=1)

        pca_data_cued_up_uncued_down = {'data': torch.cat((cued_up, uncued_down), dim=0)}
        # format the dictionary - add labels and delay-specific keywords
        pca_data_cued_up_uncued_down = format_pca_data(pca_data_cued_up_uncued_down, 'single_up', trial_type, params)
        save_data(pca_data_cued_up_uncued_down, save_path + 'pca_data_cued_up_uncued_down_model',
                  params['model_number'])

        cued_down = torch.cat((pca_data_uncued['data'][:params['B'], :delay2_end_ix, :],
                             pca_data['data'][:params['B'], delay2_end_ix:, :]), dim=1)
        uncued_up = torch.cat((pca_data['data'][:params['B'], :delay2_end_ix, :],
                                 pca_data_uncued['data'][:params['B'], delay2_end_ix:, :]), dim=1)

        pca_data_cued_down_uncued_up = {'data': torch.cat((cued_down, uncued_up), dim=0)}
        # format the dictionary - add labels and delay-specific keywords
        pca_data_cued_down_uncued_up = format_pca_data(pca_data_cued_down_uncued_up, 'single_down', trial_type, params)
        save_data(pca_data_cued_down_uncued_up, save_path + 'pca_data_cued_down_uncued_up_model',
                  params['model_number'])

    # collate all pca data into a single dict
    pca_data_all = {'cued': pca_data,
                    'uncued': pca_data_uncued,
                    'cued_up_uncued_down': pca_data_cued_up_uncued_down,
                    'cued_down_uncued_up': pca_data_cued_down_uncued_up}

    # 6. Create and save a dictionary with model outputs - for behavioural analysis.
    choices = sample_choices(readout.squeeze(), params)
    model_outputs = {'output_activations': readout.squeeze(),
                     'choices': choices,
                     'labels': {"loc": test_data['loc'],
                                "c1": test_data['c1'],
                                "c2": test_data['c2'],
                                "probed_colour": test_data['probed_colour'],
                                "unprobed_colour": test_data['unprobed_colour']}}
    if params['experiment_number'] == 4:
        # add cued and probed labels
        model_outputs['labels']['cued_loc'] = test_data['cued_loc']
    save_data(model_outputs, save_path + 'model_outputs_model', params['model_number'])
    save_data(choices, save_path + 'responses_model', params['model_number'])

    print('.... evaluated and data saved')

    return eval_data, pca_data_all, model_outputs


def export_behav_data_to_matlab(params):
    """
    Export the behavioral data to file to use in Matlab.

    :param dict params: Experimental parameters.

    """
    expt_key = params['expt_key']

    # get all test conditions and paths
    common_path = params['RAW_DATA_PATH']

    test_conditions, folder_names = generate_test_conditions()
    # get full test folder paths
    test_paths = [common_path + f for f in folder_names[expt_key]]

    # loop over all test conditions
    for condition, path in zip(test_conditions[expt_key], test_paths):
        # load model choices
        choices = []
        for model_number in np.arange(params['n_models']):
            # load model choice data
            f = open(f"{path}/model_outputs_model{model_number}.pckl", 'rb')
            model_outputs = pickle.load(f)
            f.close()

            choices.append(model_outputs['choices'])
            probed_colour = model_outputs['labels']['probed_colour']  # these values are the same for all models
            unprobed_colour = model_outputs['labels']['unprobed_colour']  # same here

        choices = torch.stack(choices)

        # export to matlab
        data_for_matlab = {'reported_colour': choices.numpy(),
                           'probed_colour': probed_colour.numpy(),
                           'unprobed_colour': unprobed_colour.numpy()}

        matlab_file_path = f"{params['MATLAB_PATH']}{expt_key}_{condition}_mixmodel_data.mat"
        savemat(matlab_file_path, data_for_matlab)


def save_data(data, save_path, model_number=None):
    """
    Saves specified data structures to file.
    
    Parameters
    ----------
    data : array-like or dictionary
    
    save_path : str

    model_number : Optional, integer. Default is None.
    
    Returns
    -------
    None
    """

    if model_number is None:
        # save data without specifying model number
        f = open(f"{save_path}.pckl", 'wb')
    else:
        f = open(f"{save_path}{model_number}.pckl", 'wb')
    pickle.dump(data, f)
    f.close()


def add_noise(data, params, device):
    """
    Adds iid noise to the input data fed to the network. Noise is drawn from 
    the specified distribution, 
    either:
    ~ U(-params['sigma'],params['sigma']) 
    or:
    ~ N(0,params['sigma'])
    and added to the base input data.
    Activation values are then constrained to be within the [0,1] range.
    Use this function to add noise for each epoch instead of creating a new 
    (noisy) dataset from scratch to speed up training.
    
    Parameters
    ----------
    data : torch.Tensor (params['seq_len'],params['batch_size'],params['n_inp'])
        Tensor containing the base input data for the network
        
    params : dictionary 
        params['sigma'] controls bound / s.d. for the noise distribution
        params['noise_distr'] specifies the distribution from which noise will 
            be drawn (standard normal or uniform)
        
    device : torch.device object
        Must match the location of the model

    Returns
    -------
    data_noisy : torch.Tensor 
        Contains a copy of the original input data with added noise.
    """
    data_noisy = data.clone()

    if params['noise_period'] == 'all':
        if params['noise_distr'] == 'normal':
            data_noisy += (torch.randn(data.shape).to(device)) * params['sigma']
        elif params['noise_distr'] == 'uniform':
            data_noisy += (torch.rand(data.shape).to(device) - 0.5) * params['sigma']
    else:
        for t in range(len(params['noise_timesteps'])):
            if params['noise_distr'] == 'normal':
                data_noisy[params['noise_timesteps'][t], :, :] = \
                    data_noisy[params['noise_timesteps'][t], :, :] + \
                    (torch.randn((1, data.shape[1], data.shape[2]))).to(device) * params['sigma']
            elif params['noise_distr'] == 'uniform':
                data_noisy[params['noise_timesteps'][t], :, :] = \
                    data_noisy[params['noise_timesteps'][t], :, :] + \
                    (torch.rand((1, data.shape[1], data.shape[2])) - 0.5).to(device) * params['sigma']

    # check if adding noise has caused any activations to be outside of the [0,1] range - if so, fix
    data_noisy[torch.where(data_noisy > 1)] = 1
    data_noisy[torch.where(data_noisy < 0)] = 1

    return data_noisy


def var_delay_mask(delay_mat, params):
    """
    Generate a mask to be used with the input data, to modify the delay length.

    Parameters
    ----------
    delay_mat : torch.Tensor (n_trials, 2)
        Trial-wise delay length values in cycles, for both delays.
    params : dict
        Experiment parameters.

    Returns
    -------
    delay_mask : torch.Tensor
        Boolean mask for the input data array modifying the delay lengths on each trial.

    """
    if not params['var_delays']:
        return

    delay_mask = torch.ones((params['seq_len'], params['stim_set_size']), dtype=bool)

    for trial in range(params['stim_set_size']):
        delay1_len = delay_mat[trial, 0]
        delay2_len = delay_mat[trial, 1]

        delay_mask[params['trial_timepoints']['delay1_start'] + delay1_len: \
                   params['trial_timepoints']['delay1_end'], trial] = False
        delay_mask[params['trial_timepoints']['delay2_start'] + delay2_len \
                   :params['trial_timepoints']['delay2_end'], trial] = False

    return delay_mask


def get_dLoss_dt(params, loss_vals):
    """
    Calculate the derivative of the loss function wrt time. Used for finding
    learning plateaus.

    Parameters
    ----------
    params: dict
        dictionary containing the Gaussian filter s.d. in
        params['conv_criterion']['smooth_sd']
    loss_vals : torch.Tensor
        loss values for every epoch (averaged across all training examples).


    Returns
    -------
    dLoss : array
        Derivative of the loss wrt to time.
    loss_clean : array
        Loss values after smoothing

    """
    if len(loss_vals.shape) < 2:
        ValueError('Loss_vals can''t be a 1-dimensional array')

    # convolve with a Gaussian filter to smooth the loss curve
    loss_clean = gaussian_filter1d(loss_vals, params['conv_criterion']['smooth_sd'])
    loss_clean = torch.tensor(loss_clean)

    # calculate the derivative
    dLoss = torch.zeros(loss_clean.shape[0] - 1)
    for i in range(loss_clean.shape[0] - 1):
        dLoss[i] = loss_clean[i + 1] - loss_clean[i]
    return dLoss, loss_clean


def get_loss_slope(params, loss_vals):
    """
    Get the slope of the loss curve over a window of trials (saved in the
    experiment parameters dictionary).

    Parameters
    ----------
    params : dict
        experiment parameters.
    loss_vals : array
        Vector of loss values from different trials.

    Returns
    -------
    slope : float
        Fitted slope value.

    """
    window = params['conv_criterion']['window']
    p = np.polyfit(np.arange(window), loss_vals, 1)
    return p[0]


def apply_conv_criterion(params, loss_vals):
    """
    Apply the convergence criterion to determine whether training should
    conclude. Two conditions must be satisfied:
        1) slope of the training loss mus be negative and >= threshold
        2) all the training loss values must fall below their threshold
    The second condition ensures that training is not stopped prematurely, at
    the mid-training plateau.

    Parameters
    ----------
    params : dict
        Experiment parameters.
    loss_vals : array
        Training loss values from recent trials.

    Returns
    -------
    criterion_reached : bool
        Flag, returns True if the two convergence conditions have been met.

    """
    # get the loss slope
    a = get_loss_slope(params, loss_vals)

    cond1 = np.logical_and(a >= params['conv_criterion']['thr_slope'],
                           a <= 0)
    cond2 = torch.all(loss_vals < params['conv_criterion']['thr_loss'])

    criterion_reached = np.logical_and(cond1, cond2)
    return criterion_reached


def custom_MSE_loss(params, output, target_scalar):
    """
    Loss function for network training. The loss term is given by the mean
    squared product of the (i) difference between the target and output vectors
    and (ii) the circular distance between the output unit tuning centres and
    the cued colour value (in radians).

    Parameters
    ----------
    params : dict
        Experiment parameters.
    output : array (n_out,)
        Network output on the last timepoint of the trial.
    target_scalar : array (1,)
        Target vector, encoded as a circular value of the cued colour.

    Returns
    -------
    loss : torch.Tensor
        Loss value.

    """

    # get the 1-hot representation of the target
    target_1hot = make_target_1hot(params, target_scalar)
    # calculate the circular distance between the output tuning centres 
    # and the cued colour value
    circ_dist = helpers.circ_diff(params['phi'], target_scalar)
    # calculate the full loss
    loss = ((circ_dist * (target_1hot - output)) ** 2).mean()
    return loss


def make_target_1hot(params, target_scalar):
    """
    Convert a scalar target into a 1-hot vector, where the individual rows
    correspond to the output units (associated with different colour tuning
    peaks).

    Parameters
    ----------
    params : dict
        Experiment parameters.
    target_scalar : float
        Cued colour value in radians.

    Returns
    -------
    target_1hot : array (n_out,)
        Target vector with a 1 at the row corresponding to the output unit with
        a tuning curve centered on the cued colour value.

    """
    target_1hot = torch.zeros((len(params['phi']),))
    target_1hot[torch.where(params['phi'] == target_scalar)[0]] = 1
    return target_1hot
