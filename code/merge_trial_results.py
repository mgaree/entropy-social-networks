# -*- coding: utf-8 -*-
"""Merge trial_results_<trial num>.npy for all trials into single file."""

import numpy as np

from constants import MAX_TIME_STEPS
from inventory_output_files import MIN_TRIAL, MAX_TRIAL
from data_processors import number_of_measures
from run_trial import results_directory


def merge_trial_results(design_name_root, min_trial_num=MIN_TRIAL, max_trial_num=MAX_TRIAL):
    master_output_database = np.empty((max_trial_num+min_trial_num, MAX_TIME_STEPS+1, number_of_measures, 2))  # the 2 is for mean, stdev

    if design_name_root == 'design':  # if the base experiment
        design_name_root = ''
    else:
        design_name_root = '.' + design_name_root  # adjust for cleaner code below

    for trial_num in range(min_trial_num, max_trial_num+1):
        try:
            trial_results = np.load(f"{results_directory}/trial_results_{trial_num}{design_name_root}.npy")
        except:
            print(f'no joy loading trial_results_{trial_num}{design_name_root}.npy')
            master_output_database[trial_num] = 0
        else:
            master_output_database[trial_num] = trial_results[:]

    np.save(f"{results_directory}/master_trial_results{design_name_root}.npy", master_output_database)
    print('done.')


nonhomogeneous_settings = {
    1: ('design_informed_uninformed', 1, 840),
    2: ('design_concord_pa', 1, 378),
    3: ('design_bots_humans', 1, 84),
    4: ('design_stubborn_normal', 1, 672)
}


def merge_nonhomogeneous_scenario_results(scenario_num):
    merge_trial_results(*nonhomogeneous_settings[scenario_num])
