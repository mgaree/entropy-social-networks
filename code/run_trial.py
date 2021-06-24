# -*- coding: utf-8 -*-
"""Given a trial number, run model replications for that trial.

This module is written to be command-line runnable, so that an external batch file can more easily
put each trial in a separate process / core. For that, the command should be

    python run_trial.py <trial_num> [starting_replication_num]

where <trial_num> is an integer for the ID number for the desired trial in the experimental
design matrix and [starting_replication_num] is an optional parameter to help with resuming
a partial trial run; both trial numbers and replication numbers start at 1.

"""
import time
import os
import numpy as np

import constants
from experimental_design import load_design_file, get_trial_settings, get_scenario_trial_settings
from model import Model, scenario_model_map
from randomness import REPLICATION_ISEED_STREAM_ARRAY


#results_directory = os.environ['RCAC_SCRATCH'] + '/nonhomogeneous_experiment'
results_directory = r'C:\Users\Mike\Desktop\tmp\run_trial_data for supporting revisions'


def run_trial(trial_num, starting_replication_num=1, design_filename='design.csv', scenario_num=None):
    """Run each replication for the specified trial and save raw replication output to files.

    Args:
        trial_num (int): trial id in design matrix to run
        starting_replication_num (int, optional): Defaults to 1, but if you provide a value here,
            you can resume progress on a trial at a specified replication.
        design_filename (str, optional): CSV file containing experimental design.
        scenario_num (int, optional): If provided, use Model class for corresponding scenario number.

    Returns:
        None. Model raw data is written to ./results/replication_raw_data_{trial_num}_{rep}.npy with
            shape (N, t_max+1, number of measures, 2), where the 2 is for (mean, st. dev.) for
            each measure.

    """
    num_replications = constants.NUM_REPLICATIONS

    design = load_design_file(design_filename)
    if scenario_num is None:
        trial_settings = get_trial_settings(trial_num, design)
    else:
        trial_settings = get_scenario_trial_settings(trial_num, design, scenario_num)
        model_cls = scenario_model_map[scenario_num]

    design_name_root = design_filename.split('.')[0]

    # array of replication times, just for minor perfomance analytics
    rep_times_array = np.full(num_replications, np.nan, dtype=np.float16)

    # run replications of trial
    # I want replication numbers to start at 1, like trial numbers
    for rep in range(starting_replication_num, num_replications + 1):
        start_time = time.time()

        # test output file write access
        replication_raw_data_filename = f"{results_directory}/replication_raw_data_{trial_num}_{rep}.{design_name_root}.npy"
        f = open(replication_raw_data_filename, 'wb')
        f.close()

        if scenario_num is None:
            model = Model(
                iseed=REPLICATION_ISEED_STREAM_ARRAY[rep],
                max_steps=constants.MAX_TIME_STEPS,
                network_parameters=trial_settings['network_parameters'],
                agent_class=trial_settings['agent_class'],
                scheduler_class=trial_settings['activation_regime'],
                error_distribution=trial_settings['influence_error_distribution'],
            )
        else:
            model = model_cls(
                iseed=REPLICATION_ISEED_STREAM_ARRAY[rep],
                max_steps=constants.MAX_TIME_STEPS,
                network_parameters=trial_settings['network_parameters'],
                agent_class=trial_settings['agent_class'],
                scheduler_class=trial_settings['activation_regime'],
                error_distribution=trial_settings['influence_error_distribution'],
                **trial_settings['kwargs']
            )
        model.run_model()

        raw_data = model.raw_data

        # save replication-level raw data to file
        np.save(replication_raw_data_filename, raw_data)

        rep_times_array[rep-1] = time.time() - start_time

    rep_times_array = rep_times_array[~np.isnan(rep_times_array)]  # filter out empty values in case of a partial run

    print(f"Trial {trial_num} complete after {rep_times_array.size} replications.\n"
          f"Time elapsed: {rep_times_array.sum()} sec; per replication: {rep_times_array.mean()} +/- {rep_times_array.std()}.\n\n")

    return


def parse_args(argv, filename="run_trial.py"):
    """Parse command line args to get trial number, design filename, and replication number (optional)."""
    if len(argv) < 3:
        print(f"Usage: python {filename} <trial number> <design filename> <scenario number> [optional: starting replication number]")
        exit()

    trial_num = argv[1]
    try:
        trial_num = int(trial_num)
    except Exception:
        print("<trial number> must be an integer")
        exit()

    if trial_num <= 0:
        print("trial numbers start at 1; we regret the confusion")
        exit()

    design_filename = argv[2]
    scenario_num = int(argv[3])  # making this a required argument for now

    rep_num = 1  # default is to start at the beginning
    try:
        rep_num = argv[4]
    except IndexError:
        pass
    else:
        try:
            rep_num = int(rep_num)
        except ValueError:
            print("[starting replication number] must be a positive integer")
            exit()
        else:
            if rep_num <= 0:
                print("[starting replication number] must be a positive integer")
                exit()

    return trial_num, rep_num, design_filename, scenario_num


if __name__ == "__main__":
    from sys import argv

    trial_num, rep_num, design_filename, scenario_num = parse_args(argv, "run_trial.py")

    run_trial(trial_num, rep_num, design_filename, scenario_num)
