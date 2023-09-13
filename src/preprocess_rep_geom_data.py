import torch
import numpy as np
import src.helpers as helpers
import src.generate_data_von_mises as dg


def get_delay_timepoints(constants):
    """
    Extracts the endpoints of the delay intervals.
    :param constants: Experimental constants module
    :type constants: module
    :return: list of indices corresponding to the delay end timepoints
    """
    # if experiment 3, update the length of delay intervals saved in constants
    # to 7 cycles (same as for expt1)
    if constants.PARAMS['experiment_number'] == 3:
        dg.update_time_params(constants.PARAMS, 7)

    # get indices of the delay end timepoints
    d1_ix = constants.PARAMS['trial_timepoints']['delay1_end'] - 1
    d2_ix = constants.PARAMS['trial_timepoints']['delay2_end'] - 1

    delay_timepoints = [d1_ix, d2_ix]
    if constants.PARAMS['experiment_number'] == 4:
        d3_ix = constants.PARAMS['trial_timepoints']['delay3_end'] - 1
        delay_timepoints.append(d3_ix)

    return delay_timepoints


def split_train_test(eval_data, train_ixs, test_ixs, cv_fold):
    """
    Splits the data into train and test subsets using the training and test trial indices from a given cv fold.

    :param eval_data: dictionary containing the to-be-split data of a () shape, under the 'data' key
    :type eval_data: dict
    :param train_ixs: list of training trial indices for each cross-validation fold, (n_cv_folds)
    :type train_ixs: list
    :param test_ixs: analogous list of test trial indices
    :type test_ixs: list
    :param cv_fold: index of the cv fold
    :type cv_fold: int
    :return: data_train, data_test
    """
    data_train = eval_data['data'][train_ixs[cv_fold], :, :]
    data_test = eval_data['data'][test_ixs[cv_fold], :, :]
    return data_train, data_test


def extract_delays(data, delay_ixs):
    """Extracts the subset of data corresponding to the delay timepoints."""
    delay_data = data[:, delay_ixs, :]
    return delay_data


def make_loc_arrays(delay_data, n_bins):
    """Reshapes the data into two arrays. Each contains data from all delay timepoints and neurons for a single cue
    location."""
    n_neurons = delay_data.shape[-1]
    # extract the data from delays
    loc1_array = delay_data[:n_bins, :, :]
    loc2_array = delay_data[n_bins:, :, :]
    # reshape tensors so that the data from each consecutive delay is stacked as rows and columns correspond to the
    # neurons
    loc1_array = loc1_array.transpose(0, 1).reshape((-1, n_neurons))
    loc2_array = loc2_array.transpose(0, 1).reshape((-1, n_neurons))

    return loc1_array, loc2_array


def preprocess_model_data_rot_unrot(constants, eval_data, train_ixs, test_ixs, n_bins):
    """ Runs the full preprocessing pipeline. """
    n_cv_folds = len(train_ixs)
    delay_ixs = get_delay_timepoints(constants)

    preprocessed_data = [None for _ in range(n_cv_folds)]
    for cv in range(n_cv_folds):
        # split into train and test
        data_train, data_test = split_train_test(eval_data, train_ixs, test_ixs, cv)

        # bin into colour bins
        data_train = helpers.bin_data(constants, data_train)
        data_test = helpers.bin_data(constants, data_test)

        # extract the delay timepoints
        delay_data_train = extract_delays(data_train, delay_ixs)
        delay_data_test = extract_delays(data_test, delay_ixs)

        # reformat into location arrays
        loc1_train, loc2_train = make_loc_arrays(delay_data_train, n_bins)
        loc1_test, loc2_test = make_loc_arrays(delay_data_test, n_bins)

        # save into dictionary
        preprocessed_data[cv] = {'train': {'loc1': loc1_train, 'loc2': loc2_train},
                                 'test': {'loc1': loc1_test, 'loc2': loc2_test}}

    return preprocessed_data


def reshape_CDI_coords(constants, cued_up_coords, cued_down_coords):
    """ Reshape the 3D coordinates into a nested list of the following format (n_validity types, n_delays, n_locations,
    n_colours), where locations are defined as 'cued' and 'uncued'.
    """
    # get the row indices corresponding to valid/invalid trials, different delay intervals and planes
    all_ixs, _, dim_numbers = get_CDI_coord_row_indices(constants)

    cued_up_reshaped, cued_down_reshaped = [], []

    for coords_array, reshaped_array in zip([cued_up_coords, cued_down_coords], [cued_up_reshaped, cued_down_reshaped]):
        for validity_type in range(dim_numbers['n_validity_types']):
            reshaped_array.append([])
            for delay in range(dim_numbers['n_timepoints']):
                reshaped_array[validity_type].append([])
                for plane in range(dim_numbers['n_locations']):
                    ixs = all_ixs[validity_type, delay, plane, :]
                    reshaped_array[validity_type][delay].append(coords_array[ixs, :])

    return cued_up_reshaped, cued_down_reshaped, dim_numbers


def get_CDI_coord_row_indices(constants):
    # get the row indices corresponding to the different conditions (defined by the validity type (valid/invalid) x
    # delay x plane status (cued/uncued or probed/unprobed) combination
    n_locations = 2
    n_timepoints = constants.PARAMS['n_delays']
    n_colours = constants.PARAMS['B']

    # valid trials
    all_ixs = []

    for delay in range(n_timepoints):
        all_ixs.append([])
        for plane in range(n_locations):
            ixs = np.arange(n_colours) + n_colours * plane + n_colours * n_locations * delay
            all_ixs[delay].append(ixs)
    all_ixs = np.array(all_ixs)
    dim_names = ['delay_number', 'plane_number', 'ixs']
    dim_numbers = {'n_timepoints': n_timepoints, 'n_locations': n_locations, 'n_colours': n_colours}

    if constants.PARAMS['cue_validity'] < 1:
        # invalid trials
        all_ixs_invalid = all_ixs + np.max(all_ixs) + 1
        all_ixs = np.stack((all_ixs, all_ixs_invalid), axis=0)
        dim_names.insert(0, 'trial_type')
        dim_numbers['n_validity_types'] = 2  # valid, invalid
    else:
        all_ixs = all_ixs[None, :]  # insert an empty dimension for compatibility with reshape_CDI_coords
        dim_numbers['n_validity_types'] = 1  # only valid

    return all_ixs, dim_names, dim_numbers


def preprocess_CDI_data(constants, all_data_valid, all_data_invalid=None):
    # for trials defined by a given *cued* location (e.g. up), aggregate binned data across cued and uncued items
    # (and valid and invalid trials where appropriate) from the delay end timepoints

    # for example, for cued 'up' trials in Experiment 4, cue validity < 1, the rows of the aggregated data array will
    # correspond to (in sets of n_colours rows, from top to bottom):
    #  (1) pre_cue_up - valid trials
    #  (2) pre_cue_down - valid trials
    #  (3) cued - valid trials
    #  (4) uncued - valid trials
    #  (5) probed - valid trials
    #  (6) unprobed - valid trials
    #  (7) pre_cue_up - invalid trials
    #  (8) pre_cue_down - invalid trials
    #  (9) cued - invalid trials
    #  (10) uncued - invalid trials
    #  (11) probed - invalid trials
    #  (12) unprobed - invalid trials
    # giving rise to an M=n_colours*12 by N=n_neurons array

    # Example 2: For experiment 1, the array will only contain 1-4 from above (n.b. all trials are valid trials in that
    # case)

    delay_timepoints = get_delay_timepoints(constants)
    all_data = {}

    for model in range(constants.PARAMS['n_models']):
        # get the data from the delay end timepoints
        probed_up = extract_delays(all_data_valid[model]['cued_up_uncued_down']['data'], delay_timepoints)
        probed_down = extract_delays(all_data_valid[model]['cued_down_uncued_up']['data'], delay_timepoints)

        probed_up = probed_up.swapaxes(1, 0).reshape(-1, 200)
        probed_down = probed_down.swapaxes(1, 0).reshape(-1, 200)

        if constants.PARAMS['cue_validity'] < 1:
            # make analogous arrays for the invalid trials
            probed_up_invalid = extract_delays(all_data_invalid[model]['cued_up_uncued_down']['data'], delay_timepoints)
            probed_down_invalid = extract_delays(all_data_invalid[model]['cued_down_uncued_up']['data'], delay_timepoints)

            probed_up_invalid = probed_up_invalid.swapaxes(1, 0).reshape(-1, 200)
            probed_down_invalid = probed_down_invalid.swapaxes(1, 0).reshape(-1, 200)

            # concatenate valid and invalid trial arrays into one
            probed_up = torch.cat((probed_up, probed_up_invalid), dim=0)
            probed_down = torch.cat((probed_down, probed_down_invalid), dim=0)

        all_data[model] = {'cued_up_uncued_down': probed_up, 'cued_down_uncued_up': probed_down}

    return all_data
