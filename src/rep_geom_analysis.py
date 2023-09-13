import numpy as np
import pandas as pd
from src.subspace import Geometry
import pickle
import matplotlib.pyplot as plt
import src.custom_plot as cplot
import src.plane_angles_analysis as angles
import src.vec_operations as vops
import src.preprocess_rep_geom_data as ppc


def get_unrotated_rotated_label(constants, preprocessed_data):
    """
    Find the 'unrotated' and 'rotated' Cued plane labels using the training data and create a dictionary mapping them
    to the two cue locations.

    :param constants: Experimental constants module.
    :type constants: module
    :param preprocessed_data: binned location-wise data from all models, split into a training and test set. For more
        information, see the get_rotated_unrotated_data function
    :type preprocessed_data: np.ndarray
    :return: labels_dict
    """
    # fit the subspace to the training data
    loc1_subspace = Geometry(preprocessed_data['train']['loc1'], constants)
    loc2_subspace = Geometry(preprocessed_data['train']['loc2'], constants)

    loc1_subspace.get_geometry()
    loc2_subspace.get_geometry()

    # find the unrotated plane - lesser absolute value of cosine theta between the pre- and post-cue planes
    unrotated_plane_ix = np.argmax([np.abs(loc1_subspace.cos_theta), np.abs(loc2_subspace.cos_theta)])

    # create a dictionary mapping the 'rotated' and 'unrotated' labels to each location
    labels = ('loc1', 'loc2') if unrotated_plane_ix == 0 else ('loc2', 'loc1')
    labels_dict = {'unrotated': labels[0], 'rotated': labels[1]}

    return labels_dict


def get_rotated_unrotated_data(constants):
    """
    Get the data for the rotated/unrotated Cued plane analysis, for all models. Data from each cued location is split
    into a training and test set and saved into a numpy array of shape (n_models, ). Each entry in the array contains a
    list of length n_cv_folds. Each list item is a dictionary with the 'train' and 'test' keys, each containing the
    'loc1' and 'loc2' sub-keys.

    For example, to access the binned data from trials where loc 1 was cued (cued_up_uncued_down trials), for model m
    and the training dataset from cross-validation fold cv, we would call the following:

    all_data[m][cv]['train']['loc1']

    :param constants: Experimental constants module.
    :type constants: module
    :return: all_data
    """
    base_path = constants.PARAMS['FULL_PATH']
    load_path = base_path + 'pca_data/valid_trials'
    n_bins = constants.PARAMS['B']

    # load test/train ixs
    with open(f"{load_path}/trial_ixs_for_unrotrot_analysis.pckl", 'rb') as f:
        trial_ixs = pickle.load(f)

    all_data = []
    for model in range(constants.PARAMS['n_models']):
        # load data
        f_name = f"{load_path}/eval_data_model{model}.pckl"
        with open(f_name, 'rb') as f:
            eval_data = pickle.load(f)

        all_data.append(ppc.preprocess_model_data_rot_unrot(constants, eval_data, trial_ixs['train'][str(model)],
                                                            trial_ixs['test'][str(model)], n_bins))

    return np.array(all_data)


def relabel_test_data(model_data_loc_labels, labels_dict):
    """
    Relabel the test data dictionary by swapping the 'loc1' and 'loc2' keys to their corresponding status ('rotated' and
    'unrotated') labels.

    :param model_data_loc_labels: data dictionary with location labels
    :type model_data_loc_labels: dict
    :param labels_dict: dictionary mapping the location labels onto the rotated/unrotated labels
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
            with open(load_path + f_name + 'model' + str(model) + '.pckl', 'rb') as f:
                all_data[model][geometry] = pickle.load(f)
    return geometry_names, all_data


def model_geometry_looper(constants, all_data, geometry_name, delay_name=None):
    """
    returns all_subspaces, all_psi, all_theta, all_PVEs

    Calculates the specified geometry for a specific memory delay for all models. Possible geometries include: 'cued',
    'uncued', 'cued_up_uncued_down' and 'cued_down_uncued_up'. Outputs 4 arguments:
        - all_subspaces: dictionary of the Geometry classes for all models, keys correspond to model numbers
        - all_psi: list of psi angle values in degrees for all models
        - all_theta: analogous for theta angle values
        - all_PVEs: list of arrays containing the percent variance explained values for the first 3 PCs of the fitted
            subspaces.

    :param constants: Experimental constants module.
    :type constants: module
    :param all_data: Data dictionary containing averaged and binned 'pca_data' arrays for all models and possible
        geometries. See the output of the 'get_all_binned_data' function for more details.
    :type all_data: dict
    :param geometry_name: Desired geometry. Choose from: 'cued', 'uncued', 'cued_up_uncued_down' and
        'cued_down_uncued_up'
    :type geometry_name: str
    :param delay_name: Desired delay interval. Choose from: 'delay1', 'delay2' and 'delay3' (only for Experiment 4).
    :type delay_name: str
    :return: all_subspaces, all_psi, all_theta, all_PVEs
    """
    all_subspaces, all_coords = {}, {}
    all_psi, all_theta, all_PVEs = [], [], []
    for model in range(constants.PARAMS['n_models']):
        if delay_name is None:
            model_data = all_data[model][geometry_name]
        else:
            model_data = all_data[model][geometry_name][delay_name]
        # fit the geometry to data from model
        all_subspaces[model] = Geometry(model_data, constants)
        all_subspaces[model].get_geometry()

        # extract the angle measures
        all_psi.append(all_subspaces[model].psi_degrees)
        all_theta.append(all_subspaces[model].theta_degrees)
        all_PVEs.append(all_subspaces[model].PVEs)

        all_coords[model] = all_subspaces[model].coords_3d

    return all_subspaces, all_psi, all_theta, all_PVEs, all_coords


def delay_looper_geometry(constants, all_data, geometry_name):
    """
    Run a given geometry analysis for all models and collect the measures (psi, theta, PVEs and subspaces) for data
    from each delay interval. Psi, theta and PVEs are saved into numpy arrays of sizes (n_models, n_delays),
    (n_models, n_delays) and (n_models, n_delays, n_PCs), respectively. The subspaces are saved into a dictionary with
    keys corresponding to the delay names (e.g., 'delay1')

    :param constants: Experimental constants module.
    :type constants: module
    :param all_data: Data dictionary containing averaged and binned 'pca_data' arrays for all models and possible
        geometries. See the output of the 'get_all_binned_data' function for more details.
    :type all_data: dict
    :param geometry_name: Desired geometry. Choose from: 'cued', 'uncued', 'cued_up_uncued_down' and
        'cued_down_uncued_up'
    :type geometry_name: str
    :return: cued_subspaces, psi, theta, PVEs
    """

    # get the geometry for all delay intervals
    psi, theta, PVEs = [], [], []
    cued_subspaces = {}

    for delay in range(constants.PARAMS['n_delays']):
        psi.append([])
        theta.append([])
        PVEs.append([])
        cued_subspaces[f"delay{delay + 1}"], psi[delay], theta[delay], PVEs[delay], _ = \
            model_geometry_looper(constants, all_data, geometry_name, f"delay{delay + 1}")

    psi = np.stack(psi).T  # (n_models, n_delays)
    theta = np.stack(theta).T  # (n_models, n_delays)
    PVEs = np.stack(PVEs).swapaxes(1, 0)  # (n_models, n_delays, PC number)

    return cued_subspaces, psi, theta, PVEs


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

        cdi_data = ppc.preprocess_CDI_data(constants, all_data_valid, all_data_invalid)
    else:
        # experiment 4, deterministic condition (cue validity = 1) and other experiments
        is_expt_4 = constants.PARAMS['experiment_number'] == 4
        _, all_data_valid = get_all_binned_data(constants, trial_type='valid', probed_unprobed=is_expt_4)
        cdi_data = ppc.preprocess_CDI_data(constants, all_data_valid)

    return cdi_data


def get_CDI(cued_up_coords_reshaped, cued_down_coords_reshaped, dim_numbers):
    """
    Calculate the CDI (surface area of the quadrilateral that captures the data coordinates from a particular
    condition). Conditions are defined by the combination of cued item location, delay timepoint, plane status
    (cued/uncued or probed/unprobed) and trial validity status. This is a single-model level function.

    :param cued_up_coords_reshaped: nested list of the following format (n_validity types, n_timepoints, n_locations,
    n_colours) containing the 3D coordinates fitted to the data from the cued_up_uncued_down trials
    :type cued_up_coords_reshaped: list
    :param cued_down_coords_reshaped: analogous list containing the data from the cued_down_uncued_up trials
    :type cued_down_coords_reshaped: list
    :param dim_numbers: Number of conditions for each dimension, namely: n_locations, n_timepoints, n_locations,
        n_validity_types
    :type dim_numbers: dictionary
    :return: CDI array of a (n_locations, n_timepoints, n_locations, n_validity_types) shape
    """

    CDI = np.empty((dim_numbers['n_locations'],  # cued location
                    dim_numbers['n_timepoints'],  # delay number
                    dim_numbers['n_locations'],  # plane status (cued/uncued or probed/unprobed)
                    dim_numbers['n_validity_types']))  # trial type: valid/invalid

    for cued_loc, cued_loc_3D_coords in enumerate([cued_up_coords_reshaped, cued_down_coords_reshaped]):
        # dimensions correspond to:
        # model, trial type (cued location), timepoint, plane1/plane2, trial type: valid/invalid
        for delay in range(dim_numbers['n_timepoints']):
            for plane in range(dim_numbers['n_locations']):
                for validity_type in range(dim_numbers['n_validity_types']):
                    CDI[cued_loc, delay, plane, validity_type] = vops.quadrilat_area(
                        cued_loc_3D_coords[validity_type][delay][plane])
    return CDI.squeeze()


def average_CDI(constants, CDI):
    """
    Average the CDI values across the different trial types (defined by the location of the cued / probed item; e.g.
    cued_up_uncued_down and vice-versa). Furthermore, for all experiments, average all the estimates from the pre-cue
    timepoint (cued/uncued, probed/unprobed and valid/invalid, where appropriate) to create a single pre-cue value. This
    is the CDI_for_stats data frame, to be used for subsequent statistical analyses.

    The CDI_for_plots data frame, to be used for plotting, is identical for most experiments except for experiment 4
    with cue validity <1. In this case, we additionally average the post-cue estimates across valid and invalid trials
    (e.g., average cued-valid and cued-invalid estimates to create a single 'cued' estimate - this is because the
    network does not know whether the trial is valid on invalid at this timepoint, so its activity should be the same
    for both conditions).

    This is a single-model level funtion.

    :param constants: Experimental constants module.
    :type constants: module
    :param CDI: Array with all CDI values of (n_cued_locations, n_delays, n_plane_statuses, n_validity_types). First
        dimension should contain data from different trials (defined by the location of the cued/probed and
        uncued/unprobed items, e.g. cued_up_uncued_down). Third dimension should contain data from planes with different
        status (cued / uncued or probed/unprobed).
    :type CDI: np.ndarray
    :return: CDI_for_plots, CDI_for_stats

    """
    # CDI[cued_loc, delay, plane, validity_type]
    # CDI_for_plots - for plotting
    # average across the trial types (cued location)
    CDI_for_plots = CDI.mean(0).squeeze()

    if constants.PARAMS['experiment_number'] == 4:
        if constants.PARAMS['cue_validity'] < 1:
            # CDI averaged - for plotting
            # probabilistic paradigms:
            # 1. average the pre-cue timepoint across: (1) cued/uncued planes and (2) valid and invalid trials, leading
            # to one pre-cue entry
            # 2. average the post-cue timepoint across valid and invalid trials, leading to one cued and one uncued
            # entry
            # 3. keep all the post-probe entries: probed_valid, probed_invalid, unprobed_valid and unprobed_invalid

            CDI_for_plots = np.concatenate((np.expand_dims(CDI_for_plots[0, :, :].mean(), -1),
                                            CDI_for_plots[1, :, :].mean(-1),
                                            CDI_for_plots[2, :, :].reshape(-1)))
            # labels
            cdi_for_plots_labels = ['pre-cue', 'cued', 'uncued', 'probed_valid', 'probed_invalid', 'unprobed_valid',
                                    'unprobed_invalid']

            # CDI for stats - for statistical analysis
            CDI_for_stats = CDI.mean(0).squeeze()  # average across cue locations
            CDI_for_stats = CDI_for_stats.reshape(-1)  # unravel into a single row

            # create column labels
            time_labels = ['precue_', 'postcue_']
            status_labels = ['cued', 'uncued']
            trial_labels = ['_valid', '_invalid']

            cdi_for_stats_labels = [f"{time}{status}{trial}" for time in time_labels for status in status_labels for
                                    trial in trial_labels]
            cdi_for_stats_labels.extend(
                [f"postprobe_{status}{trial}" for status in ['probed', 'unprobed'] for trial in trial_labels])

        else:
            # CDI_for_plots - for plotting
            # deterministic paradigm:
            # average the pre-cue timepoint across cued/uncued planes, leave the rest the same
            CDI_for_plots = np.concatenate((np.expand_dims(CDI_for_plots[0, :].mean(-1), -1),
                                            CDI_for_plots[1, :], CDI_for_plots[2, :]), 0)
            CDI_for_stats = CDI_for_plots  # same arrays

            time_labels = ['postcue_']
            status_labels = ['cued', 'uncued']

            cdi_for_stats_labels = ['pre-cue']  # single pre-cue entry
            # add post-cue labels
            cdi_for_stats_labels.extend([f"{time}{status}" for time in time_labels for status in status_labels])
            # add post-probe labels
            cdi_for_stats_labels.extend(['postprobe_' + status for status in ['probed', 'unprobed']])

            cdi_for_plots_labels = ['pre-cue', 'cued', 'uncued', 'probed', 'unprobed']

    else:
        # CDI for plots and for stats for al other experiments (1, 2, 3)
        # for pre-cue, average the cued and uncued
        CDI_for_plots = np.concatenate((np.expand_dims(CDI_for_plots[0, :].mean(-1), -1), CDI_for_plots[1, :]), 0)
        CDI_for_stats = CDI_for_plots  # same array

        # construct labels
        cdi_for_stats_labels = ['pre-cue', 'post_cued', 'post_uncued']
        cdi_for_plots_labels = ['pre-cue', 'cued', 'uncued']

        # # # save structure
        # CDI_av_df.to_csv(load_path + '/CDI.csv')
        # pickle.dump(CDI_av, open(load_path + 'CDI_for_plotting.pckl', 'wb'))
        # pickle.dump(CDI, open(load_path + 'CDI_unaveraged.pckl', 'wb'))

    # make into a dataframe
    CDI_for_plots = pd.DataFrame(CDI_for_plots[None, :], columns=cdi_for_plots_labels)
    CDI_for_stats = pd.DataFrame(CDI_for_stats[None, :], columns=cdi_for_stats_labels)
    return CDI_for_plots, CDI_for_stats


def model_CDI_looper(constants, cued_up_coords, cued_down_coords):
    """
    Calculate the CDI for all models. The steps are:
    1) reshape the 3D coord data (into conditions: plane status x delay x location) for easier manipulation
    2) calculate the CDI (rectangular area) for each condition
    3) average across the cued / probed locations, and other conditions depending on the experiment
    4) create CDI_for_plots and CDI_for_stats dataframes

    :param constants: Experimental constants module.
    :type constants: module
    :param cued_up_coords: dictionary with 3D coordinates fitted to the data from cued_up_uncued_down trials for each
        model. Keys correspond to the model number, and each contains a (n_conditions, 3) array.
    :type cued_up_coords: dict
    :param cued_down_coords: analogous dictionary with 3D coordinates fitted to the data from cued_down_uncued_up trials
    :type cued_down_coords: dict
    :return: CDI_for_plots, CDI_for_stats
    """

    CDI_for_plots = []
    CDI_for_stats = []
    for model in range(constants.PARAMS['n_models']):
        cued_up_reshaped, cued_down_reshaped, dim_numbers = ppc.reshape_CDI_coords(constants,
                                                                                   cued_up_coords[model],
                                                                                   cued_down_coords[model])

        CDI = get_CDI(cued_up_reshaped, cued_down_reshaped, dim_numbers)

        CDI_av, CDI_df = average_CDI(constants, CDI)
        CDI_for_plots.append(CDI_av)
        CDI_for_stats.append(CDI_df)

    CDI_for_plots = pd.concat(CDI_for_plots, ignore_index=True)
    CDI_for_stats = pd.concat(CDI_for_stats, ignore_index=True)

    return CDI_for_plots, CDI_for_stats


def save_CDI_to_file(constants, CDI_for_plots, CDI_for_stats):
    """
    Save CDI data frames to file.
    :param constants: Experimental constants module.
    :type constants: module
    :param CDI_for_plots: CDI data for plotting
    :type CDI_for_plots: array-like
    :param CDI_for_stats: CDI data for statistical analysis
    :type CDI_for_stats: array-like
    """
    save_path = constants.PARAMS['FULL_PATH'] + 'pca_data/'

    CDI_for_stats.to_csv(save_path + 'CDI.csv')
    pickle.dump(CDI_for_plots, open(save_path + 'CDI_for_plotting.pckl', 'wb'))

    return


def run_CDI_analysis(constants):
    """
    Run the full CDI analysis pipeline. The steps are:
    1) preprocess CDI data (make the location arrays, e.g. cued_up_uncued_down)
    2) fit a single subspace to each location array to get the 3D coords
    3.1) reshape the 3D coord data (into conditions: plane status x delay x location) for easier manipulation
    3.2) calculate the CDI (rectangular area) for each condition
    3.3) average across the cued / probed locations, and other conditions depending on the experiment
    4) plot
    5) save data to file

    :param constants: Experimental constants module.
    :type constants: module
    """

    # get the single-trial location arrays
    cdi_data = get_CDI_data(constants)

    # fit the subspaces to get 3D coords
    cued_up_subspaces, _, _, cued_up_PVEs, cued_up_coords = model_geometry_looper(constants, cdi_data,
                                                                                  'cued_up_uncued_down')
    cued_down_subspaces, _, _, cued_down_PVEs, cued_down_coords = model_geometry_looper(constants, cdi_data,
                                                                                        'cued_down_uncued_up')

    # calculate the CDI for all models (area of the data rectangle)
    CDI_for_plots, CDI_for_stats = model_CDI_looper(constants, cued_up_coords, cued_down_coords)

    # plot
    cplot.plot_CDI(constants, CDI_for_plots, log_transform=True)

    # plt.savefig(constants.PARAMS['FIG_PATH'] + 'CDI.png')
    # plt.savefig(constants.PARAMS['FIG_PATH'] + 'CDI.svg')

    # save data to file
    # save_CDI_to_file(constants, CDI_for_plots, CDI_for_stats)

    return


def run_cued_geom_analysis(constants, all_data):
    """
    Run the full Cued geometry pipeline. The steps are:
    1) Calculate the Cued geometry for each model.
    2) For Experiment 1, plot the pre-cue and post-cue geometry for example models.
    3) Plot the percentage variance explained (PVE) by the 3-dimensional subspaces for all models.
    4) Run the theta and psi angles analysis: plot, print descriptive and inferential statistics

    :param constants: experimental constants module.
    :type constants: module
    :param all_data: Data dictionary containing averaged and binned 'pca_data' arrays for all models and possible
        geometries. See the output of the 'get_all_binned_data' function for more details.
    :type all_data: dict
    """
    # get the Cued item geometry for all trained networks and delays
    cued_subspaces, psi, theta, PVEs = delay_looper_geometry(constants, all_data, 'cued')

    if constants.PARAMS['experiment_number'] == 1:
        # plot 3D geometry for example models
        models = [8, 4]
        fig_list = cplot.plot_full_geometry(constants, models, cued_subspaces)
        if constants.PLOT_PARAMS['save_plots']:
            for fig, model in zip(fig_list, models):
                fig.savefig(f"{constants.PARAMS['FIG_PATH']}cued_geometry_example_model_{model}.svg")
        pass

    # plot PVEs
    cplot.plot_PVEs_3D(constants, PVEs, fig_name='PVEs_3D_cued_geometry')

    if constants.PLOT_PARAMS['save_plots']:
        plt.savefig(f"{constants.PARAMS['FIG_PATH']}'PVEs_3D_cued_geometry.svg")

    # run the theta and psi angles analysis: plot, print descriptive and inferential statistics
    angles.run_angles_analysis(constants, theta, psi, 'cued')

    return


def run_unrotated_rotated_geometry(constants):
    """
    Run the full rotated/unrotated plane analysis for the Cued geometry. The steps are:
    1) split the data into cross-validation folds
    2) for each fold:
     2.1) find the 'rotated' and 'unrotated' cued location using the training data (based on the cos of
        theta value between the pre- and post-cue planes)
     2.2) relabel the cued locations from test data as 'rotated' and 'unrotated'
     2.3) calculate the theta and psi angles
    3) analyse the angles:
     3.1) rectify and average across cross-validation folds
     3.2) run inferential stat tests
     3.3) plot

    :param constants: experimental constants module.
    :type constants: module

    .. note:: This analysis is not implemented for Experiments 2 and 4, and running it for them will produce an error.

    """
    if constants.PARAMS['experiment_number'] == 2 or constants.PARAMS['experiment_number'] == 4:
        raise NotImplementedError('Unrotated / Rotated plane analysis only implemented for Experiments 1 and 3')

    # get the data
    preprocessed_data = get_rotated_unrotated_data(constants)

    n_cv_folds = preprocessed_data.shape[1]
    plane_label_keys = ['unrotated', 'rotated']
    all_psi = {key: [None for _ in range(n_cv_folds)] for key in plane_label_keys}
    all_theta = {key: [None for _ in range(n_cv_folds)] for key in plane_label_keys}

    # loop over cross-validation folds
    for cv in range(n_cv_folds):
        data = preprocessed_data[:, cv]

        test_data = {}
        # loop over models
        for model in range(constants.PARAMS['n_models']):
            # get the unrotated and rotated location labels
            labels_dict = get_unrotated_rotated_label(constants, data[model])
            # construct a new test dictionary with these labels (aka relabel locations)
            test_data[model] = relabel_test_data(data[model]['test'], labels_dict)

        # calculate the rotated and unrotated plane angles
        for plane_label in plane_label_keys:
            _, all_psi[plane_label][cv], \
                all_theta[plane_label][cv], _, _ = model_geometry_looper(constants, test_data, plane_label)

    # convert dictionary entries to numpy arrays of dimensionality = (n_models, n_cv_folds)
    for plane_label in plane_label_keys:
        all_psi[plane_label] = np.stack(all_psi[plane_label]).T
        all_theta[plane_label] = np.stack(all_theta[plane_label]).T

    # analyse the angles: rectify and average across cross-validation folds, run inferential stat tests and plot
    angles.run_rot_unrot_angles_analysis(constants, all_psi, all_theta, plane_label_keys)
    return


def run_uncued_geom_analysis(constants, all_data):
    """
    Run the full Uncued geometry pipeline for Experiments 1 and 3. The steps are:
    1) Calculate the Uncued geometry for each model.
    2) For Experiment 1, plot the pre-cue and post-cue geometry for example models.
    3) Plot the percentage variance explained (PVE) by the 3-dimensional subspaces for all models.
    4) Run the theta and psi angles analysis: plot, print descriptive and inferential statistics

    :param constants: experimental constants module.
    :type constants: module
    :param all_data: Data dictionary containing averaged and binned 'pca_data' arrays for all models and possible
        geometries. See the output of the 'get_all_binned_data' function for more details.
    :type all_data: dict

    .. note:: This analysis is not implemented for Experiments 2 and 4, and running it for them will produce an error.

    """
    if constants.PARAMS['experiment_number'] == 2 or constants.PARAMS['experiment_number'] == 4:
        raise NotImplementedError('Cued/Uncued geometry analysis only implemented for Experiments 1 and 3.')

    # get the Uncued item geometry for all trained networks and delays
    subspaces, psi, theta, PVEs = delay_looper_geometry(constants, all_data, 'uncued')

    if constants.PARAMS['experiment_number'] == 1:
        # plot 3D geometry for example models
        models = [19, 27]
        cplot.plot_full_geometry(constants, models, subspaces)

        # plot PVEs
        # cplot.plot_PVEs_3D(constants, PVEs, fig_name='PVEs_3D_uncued_geometry')

    # run the angles analysis only on post-cue delay data (pre-cue values will be the same as for the Cued geometry)
    post_cue_ix = 1
    angles.run_angles_analysis(constants, theta[:, post_cue_ix], psi[:, post_cue_ix], 'uncued')

    return


def run_cued_uncued_geom_analysis(constants, all_data):
    """
    Run the full Cued/Uncued geometry pipeline. The steps are:
    1) Calculate the Cued/Uncued geometry for each model.
    2) For Experiment 1, plot the pre-cue and post-cue geometry for example models.
    3) Plot the percentage variance explained (PVE) by the 3-dimensional subspaces for all models.
    4) Run the theta and psi angles analysis: plot, print descriptive and inferential statistics

    :param constants: experimental constants module.
    :type constants: module
    :param all_data: Data dictionary containing averaged and binned 'pca_data' arrays for all models and possible
        geometries. See the output of the 'get_all_binned_data' function for more details.
    :type all_data: dict

    .. note:: This analysis is not implemented for Experiments 2 and 4, and running it for them will produce an error.

    """

    if constants.PARAMS['experiment_number'] == 2 or constants.PARAMS['experiment_number'] == 4:
        raise NotImplementedError('Cued/Uncued geometry analysis only implemented for Experiments 1 and 3.')

    subspace_results = []
    theta_results = []
    for trial_type in ['cued_up_uncued_down', 'cued_down_uncued_up']:
        subspaces, _, theta, _ = delay_looper_geometry(constants, all_data, trial_type)
        subspace_results.append(subspaces)
        theta_results.append(theta)

    if constants.PARAMS['experiment_number'] == 1:
        # plot 3D geometry for example models - from cued up/uncued down trials
        models = [1, 7]
        cplot.plot_full_geometry(constants, models, subspace_results[0])

    # run the angles analysis only on post-cue delay data
    post_cue_ix = 1
    # stack the cued-up/uncued-down and cued-down/uncued-up angle estimates into a single array
    theta_post_cue = np.stack((theta_results[0][:, post_cue_ix], theta_results[1][:, post_cue_ix])).T

    angles.run_angles_analysis(constants, theta_post_cue, None, 'cued-uncued')

    return


def run_full_rep_geom_analysis(constants):
    """ Run the full representational geometry analysis. This includes:
    1) the Cued geometry (including the rotated/unrotated plane analysis for Experiments 1 and 3)
    2) the Uncued geometry
    3) the Cued/Uncued geometry
    4) the CDI analysis

    See the individual geometry runners for more details about each analysis.
    """

    print('......REPRESENTATIONAL GEOMETRY ANALYSIS......')

    # get all data
    geometry_names, all_data = get_all_binned_data(constants, trial_type='valid')

    # get the cued geometry
    run_cued_geom_analysis(constants, all_data)

    if constants.PARAMS['experiment_number'] == 1 or constants.PARAMS['experiment_number'] == 3:
        # run the rotated/unrotated cued plane analysis
        run_unrotated_rotated_geometry(constants)

        # run the uncued geometry
        run_uncued_geom_analysis(constants, all_data)

        # run the cued/uncued geometry
        run_cued_uncued_geom_analysis(constants, all_data)

    if constants.PARAMS['experiment_number'] == 1 or constants.PARAMS['experiment_number'] == 4:
        # run the CDI analysis
        run_CDI_analysis(constants)
