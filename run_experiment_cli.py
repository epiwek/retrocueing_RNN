import importlib
import argparse
from main import run_experiment


def main():

    parser = argparse.ArgumentParser(description="Run the experiment with specified arguments.")

    # pass the experiment configuration (constants module)
    parser.add_argument('constants', type=str, help="Python constants module name (without .py extension), located in "
                                                    "the'constants/' folder")

    # flags
    parser.add_argument('parallelise_flag', type=int, help="Boolean. If True, will train models in parallel.")
    parser.add_argument('train_flag', type=str, help="Boolean. If True, will train models.")
    parser.add_argument('eval_flag', type=float, help="Boolean. If True, will evaluate models.")
    parser.add_argument('analysis_flag', type=float, help="Boolean. If True, will run the analysis pipeline.")

    args = parser.parse_args()

    # Import the specified constants module
    try:
        constants_module = importlib.import_module(args.constants)
    except ImportError:
        print(f"Error: Could not import module '{args.constants}'")
        return

    # Call the run_experiment function with the provided arguments
    run_experiment(constants_module, args.parallelise_flag, args.train_flag, args.eval_flag, args.analysis_flag)


if __name__ == "__main__":
    main()
