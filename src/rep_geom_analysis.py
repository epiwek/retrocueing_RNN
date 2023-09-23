import numpy as np
import pandas as pd
from src.subspace import Geometry
import pickle
import importlib
import matplotlib.pyplot as plt
import src.custom_plot as plotter
import src.plane_angles_analysis as angles
import src.vec_operations as vops
import src.preprocess_rep_geom_data as ppc
import src.stats as stats


# %% define looper functions that will loop across models, delay intervals and experiments

def model_geometry_looper(constants, all_data, geometry_name, delay_name=None):
    """

    Calculates the specified geometry for a specific memory delay for all models. Possible geometries include: 'cued',
    'uncued', 'cued_up_uncued_down' and 'cued_down_uncued_up'. Outputs 4 arguments:
        - all_subspaces: dictionary of the Geometry classes for all models, keys correspond to model numbers
        - all_psi: list of psi angle values in degrees for all models
        - all_theta: analogous for theta angle values
        - all_PVEs: list of arrays containing the percent variance explained values for the first 3 PCs of the fitted
            subspaces.

    :param module constants: A Python module containing constants and configuration data for the simulation.
    :param dict all_data: Data dictionary containing averaged and binned 'pca_data' arrays for all models and possible
        geometries. See the output of the 'get_all_binned_data' function for more details.
    :param str geometry_name: Desired geometry. Choose from: 'cued', 'uncued', 'cued_up_uncued_down' and
        'cued_down_uncued_up'
    :param str delay_name: Desired delay interval. Choose from: 'delay1', 'delay2' and 'delay3' (only for Experiment 4).
    :return: all_subspaces, all_psi, all_theta, all_PVEs

    .. note:: This function mirrors the model_looper function from subspace_alignment_index. Both could probably be
    rewritten as decorators.
    """
    assert geometry_name in ['cued', 'uncued', 'cued_up_uncued_down', 'cued_down_uncued_up'], \
        "Incorrect geometry name, choose from : 'cued', 'uncued', 'cued_up_uncued_down' and 'cued_down_uncued_up'"

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

    :param module constants: A Python module containing constants and configuration data for the simulation.
    :param dict all_data: Data dictionary containing averaged and binned 'pca_data' arrays for all models and possible
        geometries. See the output of the 'get_all_binned_data' function for more details.
    :param str geometry_name: Desired geometry. Choose from: 'cued', 'uncued', 'cued_up_uncued_down' and
        'cued_down_uncued_up'
    :return: cued_subspaces, psi, theta, PVEs

    .. note:: This function mirrors the delay_looper function from subspace_alignment_index. Both could probably be
    rewritten as decorators.
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


def experiment_2_looper(constants):
    """
    Loop through all versions of Experiment 2 (defined by the length of the post-cue delay interval). Get the data and
    calculate the Cued geometry in a single loop. Returns the theta angle estimates and PC variance explained values for
    the fitted 3D subspaces.

    :param module constants: A Python module containing constants and configuration data for the simulation.
    :return: all_theta (n_models, n_delays, n_delay2_lengths), all_PVEs (n_models, n_delays, n_PCs, n_delay2_lengths)
    """
    assert constants.PARAMS['experiment_number'] == 2, \
        'This function should only be used for Experiment 2 (retrocue timing)'
    delay2_max_length = \
        (constants.PARAMS['trial_timings']['delay1_dur'] + constants.PARAMS['trial_timings']['delay2_dur']) // 2

    all_theta = []
    all_PVEs = []
    # loop through the different experiment variants, load their respective constants modules and collect Cued data
    for delay2_length in range(delay2_max_length + 1):
        module_name = f"constants.constants_expt2_delay2_{delay2_length}cycles"
        c = importlib.import_module(module_name)

        # get the data - make sure that all data from all variants of the experiment is saved to file.
        try:
            _, all_data = ppc.get_all_binned_data(c, trial_type='valid')
        except FileNotFoundError:
            print(f"Data from post-cue delay length {delay2_length} cycles not found. Make sure models from all "
                  f"variants of Experiment 2 have been evaluated and data saved.")
            return

        # get the cued geometry
        _, _, theta, PVEs = delay_looper_geometry(c, all_data, 'cued')
        all_theta.append(theta)
        all_PVEs.append(PVEs)

    all_theta = np.stack(all_theta).transpose([1, 2, 0])  # model x delay x delay2 length
    all_PVEs = np.stack(all_PVEs).transpose([1, 2, 3, 0])  # model x delay x PC number x delay2 length

    return delay2_max_length, all_theta, all_PVEs


def model_CDI_looper(constants, cued_up_coords, cued_down_coords):
    """
    Calculate the CDI for all models. The steps are:
    1) reshape the 3D coord data (into conditions: plane status x delay x location) for easier manipulation
    2) calculate the CDI (rectangular area) for each condition
    3) average across the cued / probed locations, and other conditions depending on the experiment
    4) create CDI_for_plots and CDI_for_stats dataframes

    :param module constants: A Python module containing constants and configuration data for the simulation.
    :param dict cued_up_coords: dictionary with 3D coordinates fitted to the data from cued_up_uncued_down trials for
        each model. Keys correspond to the model number, and each contains a (n_conditions, 3) array.
    :param dict cued_down_coords: analogous dictionary with 3D coordinates fitted to the data from cued_down_uncued_up
        trials
    :return: CDI_for_plots, CDI_for_stats: pandas DataFrames
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


#%% CDI analysis functions
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

    :param module constants: A Python module containing constants and configuration data for the simulation.
    :param np.ndarray CDI: Array with all CDI values of (n_cued_locations, n_delays, n_plane_statuses,n_validity_types).
        First dimension should contain data from different trials (defined by the location of the cued/probed and
        uncued/unprobed items, e.g. cued_up_uncued_down). Third dimension should contain data from planes with different
        status (cued / uncued or probed/unprobed).
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


def test_CDI_contrasts(constants, CDI):
    if constants.PARAMS['experiment_number'] is not 1:
        raise ValueError('Contrasts only implemented for Experiment 1, use JASP instead.')
    # first contrast: cued >> uncued
    # second contrast: cued >> pre-cue
    # third contrast (uncued == pre-cue) done in JASP

    contrast_names = ['Contrast 1: cued > uncued', 'Contrast 2: cued > pre-cue']
    contrast_items = [['post_cued', 'post_uncued'], ['post_cued', 'pre-cue']]
    for c, contrast_name in enumerate(contrast_names):
        print(contrast_names[c])
        item1, item2 = contrast_items[c]
        stats.run_contrast_paired_samples(CDI[item1], CDI[item2], alt='greater')

    return


def save_CDI_to_file(constants, CDI_for_plots, CDI_for_stats):
    """
    Save CDI data frames to file.
    :param module constants: A Python module containing constants and configuration data for the simulation.
    :param pd.DataFrame CDI_for_plots: CDI data for plotting
    :param pd.DataFrame CDI_for_stats: CDI data for statistical analysis
    """
    save_path = constants.PARAMS['FULL_PATH'] + 'pca_data/'

    CDI_for_stats.to_csv(save_path + 'CDI.csv')
    pickle.dump(CDI_for_plots, open(save_path + 'CDI_for_plotting.pckl', 'wb'))

    return

#%% get the rotated and unrated labels


def get_unrotated_rotated_label(constants, preprocessed_data):
    """
    Find the 'unrotated' and 'rotated' Cued plane labels using the training data and create a dictionary mapping them
    to the two cue locations.

    :param module constants: A Python module containing constants and configuration data for the simulation.
    :param np.ndarray preprocessed_data: binned location-wise data from all models, split into a training and test set.
        For more information, see the get_unrotated_rotated_data function from preprocess_rep_geom_data.py
    :return: labels_dict
    """
    # fit the subspace to the training data
    loc1_subspace = Geometry(preprocessed_data['train']['loc1'], constants)
    loc2_subspace = Geometry(preprocessed_data['train']['loc2'], constants)

    loc1_subspace.get_geometry()
    loc2_subspace.get_geometry()

    # find the unrotated plane - lesser absolute value of cosine theta between the pre- and post-cue planes
    unrotated_plane_ix = np.argmax([np.abs(loc1_subspace.cos_theta), np.abs(loc2_subspace.cos_theta)])

    # create a dictionary mapping the 'unrotated' and 'rotated' labels to each location
    labels = ('loc1', 'loc2') if unrotated_plane_ix == 0 else ('loc2', 'loc1')
    labels_dict = {'unrotated': labels[0], 'rotated': labels[1]}

    return labels_dict

#%% runner functions


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

    :param module constants: A Python module containing constants and configuration data for the simulation.
    """

    # get the single-trial location arrays
    cdi_data = ppc.get_CDI_data(constants)

    # fit the subspaces to get 3D coords
    cued_up_subspaces, _, _, cued_up_PVEs, cued_up_coords = model_geometry_looper(constants, cdi_data,
                                                                                  'cued_up_uncued_down')
    cued_down_subspaces, _, _, cued_down_PVEs, cued_down_coords = model_geometry_looper(constants, cdi_data,
                                                                                        'cued_down_uncued_up')

    # calculate the CDI for all models (area of the data rectangle)
    CDI_for_plots, CDI_for_stats = model_CDI_looper(constants, cued_up_coords, cued_down_coords)

    # if constants.PARAMS['experiment_number'] == 1:
        # run contrasts - done in JASP
        # test_CDI_contrasts(constants, CDI_for_stats)

    # plot
    plotter.plot_CDI(constants, CDI_for_plots, log_transform=True)

    if constants.PLOT_PARAMS['save_plots']:
        plt.savefig(constants.PARAMS['FIG_PATH'] + 'CDI.png')
        plt.savefig(constants.PARAMS['FIG_PATH'] + 'CDI.svg')

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

    :param module constants: A Python module containing constants and configuration data for the simulation.
    :param dict all_data: Data dictionary containing averaged and binned 'pca_data' arrays for all models and possible
        geometries. See the output of the 'get_all_binned_data' function for more details.
    """
    # get the Cued item geometry for all trained networks and delays
    cued_subspaces, psi, theta, PVEs = delay_looper_geometry(constants, all_data, 'cued')

    if constants.PARAMS['experiment_number'] == 1:
        # plot 3D geometry for example models
        models = [8, 4]
        fig_list = plotter.plot_full_geometry(constants, models, cued_subspaces)
        if constants.PLOT_PARAMS['save_plots']:
            for fig, model in zip(fig_list, models):
                fig.savefig(f"{constants.PARAMS['FIG_PATH']}cued_geometry_example_model_{model}.svg")
        pass

    # plot PVEs
    plotter.plot_PVEs_3D(constants, PVEs, fig_name='PVEs_3D_cued_geometry')

    if constants.PLOT_PARAMS['save_plots']:
        plt.savefig(f"{constants.PARAMS['FIG_PATH']}'PVEs_3D_cued_geometry.svg")

    # run the theta and psi angles analysis: plot, print descriptive and inferential statistics
    angles.run_angles_analysis(constants, theta, psi, 'cued')

    return


def run_cued_geometry_experiment_2(constants):
    """
    Run the Cued geometry analysis for Experiment 2 (retrocue timing). Plots the pre- and post-cue angles against
    the post-cue delay length.

    :param module constants: A Python module containing constants and configuration data for the simulation.
    """
    assert constants.PARAMS['experiment_number'] == 2, \
        'This function should only be used for Experiment 2 (retrocue timing)'

    # calculate the geometry for all variants of the experiment
    delay2_max_length, all_theta, all_PVEs = experiment_2_looper(constants)

    # plot the angle comparison
    plotter.plot_geometry_estimates_experiment_2(constants, delay2_max_length, all_theta)
    # save plot
    print('Change the path below to that shared by all variants of Expt 2')
    if constants.PLOT_PARAMS['save_plots']:
        plt.savefig(f"{constants.PARAMS['EXPT2_PATH']}'compare_cued_angles.svg")
        plt.savefig(f"{constants.PARAMS['EXPT2_PATH']}'compare_cued_angles.png")

    return


def run_unrotated_rotated_geometry(constants):
    """
    Run the full unrotated/rotated plane analysis for the Cued geometry. The steps are:
    1) split the data into cross-validation folds
    2) for each fold:
     2.1) find the 'unrotated' and 'rotated' cued location using the training data (based on the cos of
        theta value between the pre- and post-cue planes)
     2.2) relabel the cued locations from test data as 'unrotated' and 'rotated'
     2.3) calculate the theta and psi angles
    3) analyse the angles:
     3.1) rectify and average across cross-validation folds
     3.2) run inferential stat tests
     3.3) plot

    :param module constants: A Python module containing constants and configuration data for the simulation.

    .. note:: This analysis is not implemented for Experiments 2 and 4, and running it for them will produce an error.

    """
    if constants.PARAMS['experiment_number'] not in [1, 3]:
        raise NotImplementedError('Unrotated / Rotated plane analysis only implemented for Experiments 1 and 3')

    # get the data
    preprocessed_data = ppc.get_unrotated_rotated_data(constants)

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
            test_data[model] = ppc.relabel_test_data(data[model]['test'], labels_dict)

        # calculate the unrotated and rotated plane angles
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

    :param module constants: A Python module containing constants and configuration data for the simulation.
    :param dict all_data: Data dictionary containing averaged and binned 'pca_data' arrays for all models and possible
        geometries. See the output of the 'get_all_binned_data' function for more details.

    .. note:: This analysis is not implemented for Experiments 2 and 4, and running it for them will produce an error.

    """
    if constants.PARAMS['experiment_number'] not in [1, 3]:
        raise NotImplementedError('Cued/Uncued geometry analysis only implemented for Experiments 1 and 3.')

    # get the Uncued item geometry for all trained networks and delays
    subspaces, psi, theta, PVEs = delay_looper_geometry(constants, all_data, 'uncued')

    if constants.PARAMS['experiment_number'] == 1:
        # plot 3D geometry for example models
        models = [19, 27]
        plotter.plot_full_geometry(constants, models, subspaces)

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

    :param module constants: A Python module containing constants and configuration data for the simulation.
    :param dict all_data: Data dictionary containing averaged and binned 'pca_data' arrays for all models and possible
        geometries. See the output of the 'get_all_binned_data' function for more details.

    .. note:: This analysis is not implemented for Experiments 2 and 4, and running it for them will produce an error.

    """

    if constants.PARAMS['experiment_number'] not in [1, 3]:
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
        plotter.plot_full_geometry(constants, models, subspace_results[0])

    # run the angles analysis only on post-cue delay data
    post_cue_ix = 1
    # stack the cued-up/uncued-down and cued-down/uncued-up angle estimates into a single array
    theta_post_cue = np.stack((theta_results[0][:, post_cue_ix], theta_results[1][:, post_cue_ix])).T

    angles.run_angles_analysis(constants, theta_post_cue, None, 'cued-uncued')

    return


def run_full_rep_geom_analysis(constants):
    """
    Run the full representational geometry analysis. This includes:
    1) the Cued geometry (including the unrotated/rotated plane analysis for Experiments 1 and 3)
    2) the Uncued geometry (for Experiments 1, 3 and 4)
    3) the Cued/Uncued geometry (for Experiments 1, 3 and 4)
    4) the CDI analysis (for Experiments 1, 3 and 4)

    See the individual geometry runners for more details about each analysis.

    :param module constants: A Python module containing constants and configuration data for the simulation.
    """

    print('......REPRESENTATIONAL GEOMETRY ANALYSIS......')

    if constants.PARAMS['experiment_number'] == 2:
        # Experiment 2 is a special case - we need to loop through all its variants (defined by the length of the
        # post-cue delay interval) to generate the Cued geometry comparison plot.
        run_cued_geometry_experiment_2(constants)
        return

    # get all data
    geometry_names, all_data = ppc.get_all_binned_data(constants, trial_type='valid')

    # get the cued geometry
    run_cued_geom_analysis(constants, all_data)

    if constants.PARAMS['experiment_number'] in [1, 3]:
        # run the unrotated/rotated cued plane analysis
        run_unrotated_rotated_geometry(constants)

        # run the uncued geometry
        run_uncued_geom_analysis(constants, all_data)

        # run the cued/uncued geometry
        run_cued_uncued_geom_analysis(constants, all_data)

    if constants.PARAMS['experiment_number'] in [1, 4]:
        # run the CDI analysis
        run_CDI_analysis(constants)
