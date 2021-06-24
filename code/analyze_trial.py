# -*- coding: utf-8 -*-
"""
Given a trial number, perform analysis on the trial's raw data to save to file the trial-level results.

This module is written to be command-line runnable, so that an external batch file can more easily
put each trial in a separate process / core. For that, the command should be

    python analyze_trial.py <trial_num> [starting_replication_num]

where <trial_num> is an integer for the ID number for the desired trial in the experimental
design matrix and [starting_replication_num] is an optional parameter to help with resuming
a partial trial run; both trial numbers and replication numbers start at 1.

"""
import time
import numpy as np
from randomgen import RandomGenerator, PCG64

import constants
from experimental_design import load_design_file, get_trial_settings
from randomness import REPLICATION_ISEED_STREAM_ARRAY
from randomness import RNGStream, RandomGenRandomInterface
import data_processors
import networks
from run_trial import results_directory  # to change, edit the constant in run_trial.py



def analyze_trial(trial_num, starting_replication_num=1, design_filename='design.csv'):
    """Analyze trial data from files, processing raw replication data into trial-level responses.

    Args:
        trial_num (int): trial id in design matrix to run
        starting_replication_num (int, optional): Defaults to 1, but if you provide a value here,
            you can resume progress on a trial at a specified replication.
        design_filename (str, optional): CSV file containing experimental design.

    Returns:
        None. Trial results are written to file. Data will have shape (t_max+1, number of measures, 2),
            where the 2 is for (mean, st. dev.) for each measure

    """
    design_name_root = design_filename.split('.')[0]

    # check for filename accessibility
    # ... if file already exists, it will be overwritten, so be careful
    trial_level_results_filename = f"{results_directory}/trial_results_{trial_num}.{design_name_root}.npy"
    # if permission denied or folder doesn't exist, will raise error; let it
    f = open(trial_level_results_filename, 'wb')
    f.close()

    num_replications = constants.NUM_REPLICATIONS

    design = load_design_file(design_filename)  # need these to build edge list
    trial_settings = get_trial_settings(trial_num, design)

    # init replication-level results container
    # t_max + 1 to account for initial state
    replication_level_results = np.zeros(
        (constants.NUM_REPLICATIONS, constants.MAX_TIME_STEPS + 1, data_processors.number_of_measures, 2))

    # array of replication times, just for minor perfomance analytics
    rep_times_array = np.full(num_replications, np.nan, dtype=np.float16)

    # want replication numbers to start at 1, like trial numbers
    for rep in range(starting_replication_num, num_replications + 1):
        start_time = time.time()

        # test output file access
        # will be reading this one
        replication_raw_data_filename = f"{results_directory}/replication_raw_data_{trial_num}_{rep}.{design_name_root}.npy"

        # to resume analysis later, need to save these
        replication_level_results_filename = f"{results_directory}/replication_results_{trial_num}_{rep}.{design_name_root}.npy"
        f = open(replication_level_results_filename, 'wb')
        f.close()

        # here, we're rebuilding the network graph by the same approach as the model init
        network_parameters = trial_settings['network_parameters']
        rg = RandomGenerator(PCG64(REPLICATION_ISEED_STREAM_ARRAY[rep], RNGStream.GRAPH_GENERATOR + 1))
        network_rg = RandomGenRandomInterface(rg)
        G = getattr(networks, network_parameters[0])(*network_parameters[1], network_rg)
        G = networks.prepare_graph_for_trial(G, network_rg)  # make directed, connected, & free of self-loops

        # load raw data file
        raw_data = np.load(replication_raw_data_filename)

        # minor data validation
        if raw_data.min() < constants.MIN_OPINION:
            print(f"Warning: raw_data for replication {rep} contains values below MIN_OPINION!")
        if raw_data.max() > constants.MAX_OPINION:
            print(f"Warning: raw_data for replication {rep} contains values below MIN_OPINION!")

        # we compute agent-level entropy values on the raw data
        # init agent-level container(s), with shape(N, t, num_rv)
        agent_level_entropy = data_processors.process_raw_data_into_entropy_timeseries(raw_data, G)
        
        # now with a N x t x (# entropy array), we average each time step per entropy across all N agents
        replication_level_entropy = np.stack(
            (agent_level_entropy.mean(axis=0), agent_level_entropy.std(axis=0)), axis=2)  # shape(t, num_rv, 2)

        # now we have the replication-level results -- 1 time series per entropy measure (plus st.dev)
        # we will save these to file if replication_level_output_filename is set (it is 'can store' data)
        np.save(replication_level_results_filename, replication_level_entropy)

        # but in either case, we save these in an array
        # -1 to account for my choice to start reps at 1
        replication_level_results[rep - 1] = replication_level_entropy[:]  # shape(num_rep, t, num_rv, 2)

        rep_times_array[rep - 1] = time.time() - start_time

    # next rep

    if starting_replication_num > 1:
        # not all replications are present in replication_level_entropy. the missing data ought to have
        # been saved to file on previous (partial) runs of this script, so recover them
        for rep in range(1, starting_replication_num):
            replication_level_results[rep - 1] = np.load(f"{results_directory}/replication_results_{trial_num}_{rep}.{design_name_root}.npy")

    # finally, average (and standardly deviate) across all replications to
    # produce trial-level response variables (1 time series per entropy measure (6)) plus stdev
    #
    # , 0] to only take mean of means, not mean of stdevs
    mean_replication_level_results = replication_level_results[:, :, :, 0].mean(axis=0)
    stdev_replication_level_results = replication_level_results[:, :, :, 0].std(axis=0)
    trial_level_response_variables = np.stack(
        (mean_replication_level_results, stdev_replication_level_results), axis=2)  # shape(t, num_rv, 2)

    # and preserve our hard work!
    np.save(trial_level_results_filename, trial_level_response_variables)

    rep_times_array = rep_times_array[~np.isnan(rep_times_array)]  # filter out empty values in case of a partial run

    print(f"Analysis of Trial {trial_num} complete ({rep_times_array.size} replications).\n"
          f"Time elapsed: {rep_times_array.sum()} sec; per replication: {rep_times_array.mean()} +/- {rep_times_array.std()}.\n\n")

    return


if __name__ == "__main__":
    from sys import argv
    from run_trial import parse_args

    trial_num, rep_num, design_filename = parse_args(argv, "analyze_trial.py")

    analyze_trial(trial_num, rep_num, design_filename)
