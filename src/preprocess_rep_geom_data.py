import pickle
import torch
import numpy as np
import src.helpers as helpers
import src.generate_data_von_mises as dg
from sklearn.model_selection import StratifiedKFold


def get_delay_timepoints(constants):
    """
    Extracts the endpoints of the delay intervals.
    :param constants: Experimental constants
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


def get_cv_trial_ixs(constants, model_seed, cv=2):
    """
    Split the data into cross-validation folds and get the corresponding trial indices.

    :param constants: Experimental constants module.
    :type constants: module
    :param model_seed : Seed parameter for the stratified K-fold cross-validator object. Pass model number for
        reproducibility.
    :type model_seed: int
    :param cv: Optional. Number of cross-validation folds. The default is 2.
    :type cv: int
    :return: arrays with train and test sets indices.
    """
    n_samples = constants.PARAMS['n_trial_types'] * constants.PARAMS['n_trial_instances_test']
    trial_labels = [np.arange(constants.PARAMS['n_trial_types'])] * constants.PARAMS['n_trial_instances_test']
    trial_labels = np.stack(trial_labels, 1).reshape(-1)
    # label corresponds to a unique c1-c2-retrocue combination

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=model_seed)
    # get train and test trial indices
    train, test = skf.split(np.zeros(n_samples), trial_labels)

    return train, test


def split_train_test(eval_data, train_ixs, test_ixs, cv_fold):
    """
    Splits the data into train and test subsets using the training and test trial indices from a given cv fold.

    :param eval_data: dictionary containing the to-be-split data of a (n_trials, n_timepoints, n_neurons) shape, under
        the 'data' key
    :type eval_data: dict
    :param train_ixs: list of training trial indices for each cross-validation fold, (n_cv_folds)
    :type train_ixs: list
    :param test_ixs: analogous list of test trial indices
    :type test_ixs: list
    :param cv_fold: index of the cv fold
    :type cv_fold: int
    :return: data_train, data_test arrays (n_trials, n, m)
    """
    data_train = eval_data['data'][train_ixs[cv_fold], :, :]
    data_test = eval_data['data'][test_ixs[cv_fold], :, :]
    return data_train, data_test


def extract_delays(data, delay_ixs):
    """
    Extracts the subset of data corresponding to the delay timepoints.

    :param data: data (m, n_timepoints, n)
    :type data: torch.Tensor
    :param delay_ixs: delay timepoint indices
    :type delay_ixs: list
    :return: delay_data array containing only the entries corresponding to the delay timepoints
    """
    delay_data = data[:, delay_ixs, :]
    return delay_data


def make_loc_arrays(delay_data, n_bins):
    """
    Reshapes the data into two arrays. Each contains data from all delay timepoints and neurons for a single cue
    location. Entries corresponding to the consecutive delay intervals are stacked as rows, columns correspond to
    neurons. E.g., the first column of the array will contain the following data for the first neuron:
        [colour_bin1_delay1;
        colour_bin2_delay1;
        colour_bin3_delay1;
        colour_bin4_delay1;
        colour_bin1_delay2;
        colour_bin2_delay2;
        colour_bin3_delay2;
        colour_bin4_delay2;]

    :param delay_data: data array (n_colour_bins*2, m, n)
    :type delay_data: torch.Tensor
    :param n_bins: number of colour bins
    :type: int
    :return: loc1_array, loc2_array
    """
    n_neurons = delay_data.shape[-1]
    # extract the data from delays
    loc1_array = delay_data[:n_bins, :, :]
    loc2_array = delay_data[n_bins:, :, :]
    # reshape tensors so that the data from each consecutive delay is stacked as rows, and columns correspond to the
    # neurons
    loc1_array = loc1_array.transpose(0, 1).reshape((-1, n_neurons))
    loc2_array = loc2_array.transpose(0, 1).reshape((-1, n_neurons))

    return loc1_array, loc2_array


def preprocess_model_data_rot_unrot(constants, eval_data, train_ixs, test_ixs, n_bins):
    """
    Runs the full preprocessing pipeline.

    Splits the model evaluation data into train and test subsets in cross-validation, extracts the delay end timepoints
    and bins the data into n_bins colour bins. Creates a location-specific array (e.g., an array containing 'cued' and
    'uncued' colour representations from the 'cued_up_uncued_down' trials.

    :param constants: Experimental constants
    :type constants: module
    :param eval_data: Dictionary containing the model evaluation data. Keys include: 'dimensions', 'data', and 'labels'.
        Data entry has the following dimensionality: (n_trials, n_timepoints, n_neurons). For more details, refer to the
        eval_model function in retrocue_model.py
    :type eval_data: dict
    :param train_ixs: list of training trial indices for each cross-validation fold, (n_cv_folds)
    :type train_ixs: list
    :param test_ixs: analogous list of test trial indices
    :type test_ixs: list
    :param n_bins: number of colour bins to bin data into
    :type n_bins: int
    :return: preprocessed_data: list of data dictionaries, each item corresponds to a cross-validation fold and contains
        'train' and 'test' key, each in turn containing location-specific data under 'loc1' and 'loc1' 2.
    """

    n_cv_folds = len(train_ixs)
    delay_ixs = get_delay_timepoints(constants)

    preprocessed_data = []
    for cv in range(n_cv_folds):
        # split into train and test
        data_train, data_test = split_train_test(eval_data, train_ixs, test_ixs, cv)

        # bin into colour bins
        data_train = helpers.bin_data(constants.PARAMS, data_train)
        data_test = helpers.bin_data(constants.PARAMS, data_test)

        # extract the delay timepoints
        delay_data_train = extract_delays(data_train, delay_ixs)
        delay_data_test = extract_delays(data_test, delay_ixs)

        # reformat into location arrays
        loc1_train, loc2_train = make_loc_arrays(delay_data_train, n_bins)
        loc1_test, loc2_test = make_loc_arrays(delay_data_test, n_bins)

        # save into dictionary
        preprocessed_data.append({'train': {'loc1': loc1_train, 'loc2': loc2_train},
                                  'test': {'loc1': loc1_test, 'loc2': loc2_test}})

    return preprocessed_data


def reshape_CDI_coords(constants, cued_up_coords, cued_down_coords):
    """
    Reshape the 3D coordinates into a nested list of the following format: (n_validity types, n_delays, n_locations,
    n_colours), where locations are defined as 'cued' and 'uncued'.

    :param constants: Experimental constants
    :type constants: module
    :param cued_up_coords: 3D coordinates fitted to the data from the 'cued_up_uncued_down' trials (n_datapoints, 3)
    :type cued_up_coords: np.ndarray
    :param cued_down_coords: analogous coordinates for the 'cued_down_uncued_up' trials
    :type cued_down_coords: np.ndarray
    :return: cued_up_reshaped, cued_down_reshaped, dim_numbers (dictionary with n_timepoints, n_locations, n_colours and
    n_validity types).
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
    """
    Get the row indices in the CDI data array that correspond to the different conditions. Conditions are defined by the
    validity type (valid/invalid) x delay x plane status (cued/uncued or probed/unprobed) combination.

    Returns 3 outputs: array with all indices for a given condition (numpy array), dimension names (list) and dimension
    numbers (list).

    :param constants: Experimental constants
    :type constants: module
    :return: all_ixs, dim_names, dim_numbers
    """
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
    """
    Run the full data preprocessing pipeline for the CDI analysis.

    For trials defined by a given *cued* location (e.g. up - cued_up_uncued_down), aggregate binned data across cued and
    uncued items (and valid and invalid trials where appropriate) from the delay end timepoints into a single array.
    The two arrays are saved into an 'all_data' dictionary, where keys are model numbers, and each entry contains the
    data arrays under 'cued_up_uncued_down' and 'cued_down_uncued_up' sub-keys. For example, to access the data from
    'cued_up' trials for model 0, we would call: all_data[0]['cued_up_uncued_down']

    :param constants: Experimental constants
    :type constants: module
    :param all_data_valid: dictionary containing the data from valid trials. Keys correspond to model numbers, and each
        model entry contains sub-keys corresponding to the geometry names, with data stored under 'data'. For the
        purpose of this function, each model-level sub-dictionary must contain the 'cued_up_uncued_down' and
        'cued_down_uncued_up' keys. E.g., for model 0, data from the 'cued_up_uncued_down' trials would be stored under:
        all_data_valid[0]['cued_up_uncued_down']['data']. For more information, consult the 'get_all_binned_data'
        function in rep_geom_analysis.py
    :type all_data_valid: dict
    :param all_data_invalid: Optional. Analogous dictionary containing the data from invalid trials (relevant for
        Experiment 3). Default is None (appropriate for all other experiments).
    :type all_data_invalid: dict
    :return: all_data dictionary, indexed by model number keys, with each model entry containing the aggregated data
        under the 'cued_up_uncued_down' and 'cued_down_uncued_up' keys.
    """
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


# %% get data and labels

def get_unrotated_rotated_data(constants, get_test_train_split=False):
    """
    Get the data for the unrotated/rotated Cued plane analysis, for all models. Data from each cued location is split
    into a training and test set and saved into a numpy array of shape (n_models, n_cv_folds). Each entry in the array
    is a dictionary with the 'train' and 'test' keys, each containing the 'loc1' and 'loc2' sub-keys.

    For example, to access the binned data from trials where loc 1 was cued (cued_up_uncued_down trials), for model m
    and the training dataset from cross-validation fold cv, we would call the following:

    all_data[m,cv]['train']['loc1']

    :param constants: Experimental constants module.
    :type constants: module
    :param get_test_train_split: Optional. Flag determining whether to draw the cross validation folds. If False, loads
        from file. Default is True.
    :param get_test_train_split: bool
    :return: all_data array (n_models, n_cv_folds)
    """
    base_path = constants.PARAMS['FULL_PATH']
    load_path = base_path + 'pca_data/valid_trials'
    n_bins = constants.PARAMS['B']

    # get the train/test indices
    if get_test_train_split:
        # draw train/test trial splits for all models
        trial_ixs = {'train': {}, 'test': {}}
        for model in range(constants.PARAMS['n_models']):
            trial_ixs['train'][model], trial_ixs['test'][model] = get_cv_trial_ixs(constants, model, cv=2)
    else:
        # load test/train ixs from file
        with open(f"{load_path}/trial_ixs_for_unrotrot_analysis.pckl", 'rb') as f:
            trial_ixs = pickle.load(f)

    all_data = []
    for model in range(constants.PARAMS['n_models']):
        # load data
        f_name = f"{load_path}/eval_data_model{model}.pckl"
        with open(f_name, 'rb') as f:
            eval_data = pickle.load(f)

        all_data.append(preprocess_model_data_rot_unrot(constants, eval_data, trial_ixs['train'][str(model)],
                                                        trial_ixs['test'][str(model)], n_bins))

    return np.array(all_data)


def relabel_test_data(model_data_loc_labels, labels_dict):
    """
    Relabel the test data dictionary by swapping the 'loc1' and 'loc2' keys to their corresponding status ('unrotated'
    and 'rotated') labels.

    :param model_data_loc_labels: data dictionary with location labels
    :type model_data_loc_labels: dict
    :param labels_dict: dictionary mapping the location labels onto the unrotated/rotated labels
    :type labels_dict: dict
    :return: model_data_rot_unrot_labels - relabelled data dictionary
    """
    model_data_rot_unrot_labels = {}
    for plane_label in labels_dict.keys():
        loc_label = labels_dict[plane_label]
        model_data_rot_unrot_labels[plane_label] = model_data_loc_labels[loc_label]
    return model_data_rot_unrot_labels


def get_all_binned_data(constants, trial_type='valid', probed_unprobed=False):
    """
    Load binned 'pca_data' dictionaries from all models. Loaded data includes the 'cued', 'uncued',
    'cued_up_uncued_down' and 'cued_down_uncued_up' dictionaries. Data is saved into an 'all_data' dictionary,
    with keys corresponding to the number models. Each model sub-dictionary contains keys corresponding to the above
    data structures. Geometry names are additionally saved into the geometry_names list.

    :param constants: Experimental constants module.
    :type constants: module
    :param trial_type: Optional. Pass 'valid' or 'invalid', default is 'valid'.
    :type trial_type: str
    :param probed_unprobed: Optional. Pass True if you want to include the 'probed' and 'unprobed' geometry data
        (for Experiment 3). Default is False.
    :type probed_unprobed: bool
    :return: geometry_names: list, all_data : dictionary with data for each model and geometry

    .. Note:: For example, to extract the data dictionary containing the data averaged across uncued colours and
    binned across cued colours for model 0, we would want to access the following part of the all_data dictionary:
        all_data[0]['cued']
    """
    base_path = constants.PARAMS['FULL_PATH']
    load_path = base_path + 'pca_data/' + trial_type + '_trials/pca_data_'

    all_data = {}
    geometry_names = ['cued', 'uncued', 'cued_up_uncued_down', 'cued_down_uncued_up']
    geometry_f_names = ['', 'uncued_', 'cued_up_uncued_down_', 'cued_down_uncued_up_']

    if probed_unprobed:
        geometry_names.extend(('probed', 'unprobed'))
        geometry_f_names.extend(('probed_', 'unprobed_'))

    for model in range(constants.PARAMS['n_models']):
        all_data[model] = {}
        for geometry, f_name in zip(geometry_names, geometry_f_names):
            with open(f"{load_path}{f_name}model{model}.pckl", 'rb') as f:
                all_data[model][geometry] = pickle.load(f)
    return geometry_names, all_data


def get_CDI_data(constants):
    """
    Load binned 'pca_data' dictionaries from all models, separately for each cued location trials (e.g.
    cued_up_uncued_down). Data is aggregated across cued and uncued items (and valid and invalid trials where
    appropriate) from the delay end timepoints.

    :param constants: Experimental constants module.
    :type constants: module
    :return: cdi_data nested dictionary with model number keys, each containing the 'cued_up_uncued_down' and
    'cued_down_uncued_up' sub-keys containing the data.

    """
    # get the data from valid and invalid trials and run the preprocessing to create two location-specific data arrays
    if constants.PARAMS['cue_validity'] < 1:
        # experiment 4, probabilistic conditions (cue validity < 1)
        _, all_data_valid = get_all_binned_data(constants, trial_type='valid', probed_unprobed=True)
        _, all_data_invalid = get_all_binned_data(constants, trial_type='invalid', probed_unprobed=True)

        cdi_data = preprocess_CDI_data(constants, all_data_valid, all_data_invalid)
    else:
        # experiment 4, deterministic condition (cue validity = 1) and other experiments
        is_expt_4 = constants.PARAMS['experiment_number'] == 4
        _, all_data_valid = get_all_binned_data(constants, trial_type='valid', probed_unprobed=is_expt_4)
        cdi_data = preprocess_CDI_data(constants, all_data_valid)

    return cdi_data
