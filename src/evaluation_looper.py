"""
Use this code to evaluate all models from the different variants of experiment 2 (defined by the length of the pre-cue
delay interval) or experiment 4 (defined by the cue validity). Uncomment the relevant code section to run.

@author: emilia
"""

import importlib
from main import run_experiment


def run_all_variants(experiment_number, parallelise_flag, train_flag, eval_flag, analysis_flag):
    """
    Run the experiment for all variants of Experiment 2 (defined by the length of the post-cue delay interval, delay2)
    or Experiment 4 (defined by the cue validity value).

    :param int experiment_number: Number of the experiment. Choose from 2 or 4.
    :param bool parallelise_flag: If True, will train models in parallel (provided that the train_flag is set to True).
    :param bool train_flag: If True, will train models.
    :param bool eval_flag: If True, will evaluate trained models.

    """
    assert experiment_number in [2, 4], 'This function should only be used to loop through the different variants of ' \
                                        'Experiments 2 or 4.'

    if experiment_number == 2:
        conditions = range(0, 8)  # delay2 lengths
    else:
        # experiment 4
        conditions = ['0_5', '0_75', '1']  # cue validity values

    # loop through conditions and evaluate associated models
    for cond in conditions:
        # import the relevant experimental constants module
        if experiment_number == 2:
            module_name = f"constants.constants_expt2_delay2_{cond}cycles"
        else:
            module_name = f"constants.constants_expt4_val{cond}"
        c = importlib.import_module(module_name)
        # run the experiment for all models
        run_experiment(c, parallelise_flag, train_flag, eval_flag, analysis_flag)


