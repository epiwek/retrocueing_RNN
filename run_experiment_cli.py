"""
This module allows us to run the simulation from the command line interface. To do so, you need to specify the
experimental constants file, as well as the part(s) of the simulation that you would like to run (controlled by flags:
parallelise_flag, train_flag, eval_flag, and analysis_flag). The flags control different aspects of the simulation,
namely model training (and whether it should be done in parallel), evaluation, and data analysis.


The experimental configuration constants modules are located in the 'constants' directory. Therefore, you need to pass
the module name as 'constants.{module_name}'

For example, to run Experiment 1 (only the analysis phase), you would run the following from the command line:
>> python run_experiment_cli.py constants.constants_expt1 --analysis_flag

"""


import importlib
import argparse
from main import run_experiment


def main():

    parser = argparse.ArgumentParser(description="Run the experiment with specified arguments.")

    # pass the experiment configuration (constants module)
    parser.add_argument('--constants', type=str, help="Python constants module name (without .py extension), located in "
                                                    "the'constants/' folder. E.g., constants.constants_expt1")

    # flags
    parser.add_argument('--parallelise_flag', action='store_true',
                        help="Set the flag to True to train models in parallel.")
    parser.add_argument('--train_flag', action='store_true',
                        help="Set the flag to True to train models.")
    parser.add_argument('--eval_flag', action='store_true',
                        help="Set the flag to True to evaluate models.")
    parser.add_argument('--analysis_flag', action='store_true',
                        help="Set the flag to True to run the analysis pipeline.")

    args = parser.parse_args()

    # Import the specified constants module
    try:
        constants_module = importlib.import_module(args.constants)
    except ImportError:
        print(f"Error: Could not import module '{args.constants}'")
        return

    # Call the run_experiment function with the provided arguments
    run_experiment(constants=constants_module,
                   parallelise_flag=args.parallelise_flag,
                   train_flag=args.train_flag,
                   eval_flag=args.eval_flag,
                   analysis_flag=args.analysis_flag)


if __name__ == "__main__":
    main()
