# -*- coding: utf-8 -*-
"""Helper methods for preparing experimental design matrix and using it to build models."""

import itertools
import pandas as pd
import numpy as np

import agents
import activation_regimes


#
# Experimental design factors and levels
#

factors = dict(
    population_size=[
        100,
        1000,
        10000,
    ],
    network_structure_model=[
        'erdos_renyi_random(N)',
        'small_world(N, 0.0, 3)',
        'small_world(N, 0.0, 10)',
        'small_world(N, 0.33, 3)',
        'small_world(N, 0.33, 10)',
        'small_world(N, 0.66, 3)',
        'small_world(N, 0.66, 10)',
        'scale_free(N, 1)',
        'scale_free(N, 3)',
        'scale_free(N, 5)',
    ],
    influence_model=[
        'standard_model',
        'similarity_bias',
        'attractive_repulsive',
        'random_adoption',
        'nonlinear',
    ],
    influence_error_distribution=[
        'none',
        'N(0, 0.05)',
        'N(0, 0.1)',
        'N(0, 0.2)',
    ],
    agent_activation_regime=[
        'synchronous',
        'uniform',
        'random',
    ],
)

#
# Helper methods for turning factor-levels into code elements
#

influence_model_to_agent_class_map = {
    'standard_model': agents.StandardModelAgent,
    'similarity_bias': agents.SimilarityBiasAgent,
    'attractive_repulsive': agents.AttractiveRepulsiveAgent,
    'random_adoption': agents.RandomAdoptionAgent,
    'nonlinear': agents.NonlinearAgent,
}

activation_regime_to_class_map = {
    'synchronous': activation_regimes.Synchronous,
    'uniform': activation_regimes.Uniform,
    'random': activation_regimes.Random,
}

error_distribution_string_to_distro_map = {
    'none': None,
    'N(0, 0.05)': ('normal', 0, 0.05),
    'N(0, 0.1)': ('normal', 0, 0.1),
    'N(0, 0.2)': ('normal', 0, 0.1),
}


def get_network_params_by_structure_design_string(N, design_str):
    """Use factor-level for network structure model to generate network parameters.

    The Model class uses its RNG seed and these params to create the networkx.DiGraph for the run.

    Args:
        design_str (str): of form 'small_world(N, 0.33, 3)', from experimental design

    Returns:
        tuple: of form (generator name, (*args))

    """
    generator_name, args = design_str[:-1].split('(')  # -1 to remove trailing )
    args = eval(args)  # eval will turn a string with commas into a tuple and replace 'N' with actual number...
    if not isinstance(args, tuple):  # unless there is no comma, as in erdos_renyi
        args = (args,)

    return (generator_name, args)


def get_agent_class_by_influence_model_design_string(design_str):
    """Use factor-level for influence model to select agent class.

    Args:
        design_str (str): identifier string for agent's influence model

    Returns:
        agents.<class>

    """
    cls = influence_model_to_agent_class_map[design_str]

    return cls


def get_influence_error_distribution_by_design_string(design_str):
    """Use factor-level for influence error term distribution to select generator function.

    Args:
        design_str (str): identifier string for influence error distribution

    Returns:
        tuple or None: tuple is of form (str: function name, *args)

    """
    cls = error_distribution_string_to_distro_map[design_str]

    return cls


def get_activation_regime_by_design_string(design_str):
    """Use factor-level for activation regime to select scheduler class.

    Args:
        design_str (str): identifier string for activation regime

    Returns:
        activation_regime.<class>

    """
    cls = activation_regime_to_class_map[design_str]

    return cls


#
# Helper methods for processing the design matrix
#


def make_full_factorial_design_matrix(output_filename=None):
    """Convert the factors & levels into a design matrix (data frame).

    Args:
        output_filename (str, optional): If specified, the design matrix is saved to a CSV file
            using the specified filename.

    Returns:
        pandas.DataFrame: the experimental design matrix

    """
    factor_names = list(sorted(factors.keys()))  # Sorting simply to have deterministic order
    data = list(itertools.product(*[factors[factor] for factor in factor_names]))

    design = pd.DataFrame(data, columns=factor_names)
    design.index = np.arange(1, len(design) + 1)  # I want trial numbers to start at 1

    if output_filename is not None:
        design.to_csv(output_filename)

    return design


def get_trial_settings(trial_num, design):
    """Get settings for one trial from the design matrix.

    Args:
        trial_num (int): starts at 1, specifies row index of trial in design dataframe
        design (pandas.DataFrame): experimental design matrix

    Returns:
        dict: trial factor-level elements (numbers, names, or classes)

    """
    row = design.loc[trial_num]

    N = row['population_size']

    # only factor with dependency on another one
    network_params = get_network_params_by_structure_design_string(N, row['network_structure_model'])

    agent_cls = get_agent_class_by_influence_model_design_string(row['influence_model'])
    influence_error_distro = get_influence_error_distribution_by_design_string(row['influence_error_distribution'])
    activation_regime = get_activation_regime_by_design_string(row['agent_activation_regime'])

    return {
        'trial_num': trial_num,
        'N': N,
        'network_parameters': network_params,
        'agent_class': agent_cls,
        'influence_error_distribution': influence_error_distro,
        'activation_regime': activation_regime
    }


def load_design_file(filename='design.csv'):
    """Load experimental design CSV into data frame and return it.

    Assumes CSV was created by this module.

    Args:
        filename (str): path to design CSV file

    Returns:
        pandas.DataFrame: experimental design matrix

    """
    design = pd.read_csv(filename, index_col=0)
    return design


#
# Nonhomogeneous design
#


nonhomogeneous_factors = dict(  # reduced version of original factors
    population_size=[
        1000,       # easier to include this here than to modify all trial-runner code to hardcode N=1000
    ],
    network_structure_model=[
        'erdos_renyi_random(N)',
        'small_world(N, 0.0, 3)',
        'small_world(N, 0.0, 10)',
        'small_world(N, 0.66, 3)',
        'small_world(N, 0.66, 10)',
        'scale_free(N, 1)',
        'scale_free(N, 5)',
    ],
    influence_model=[
        'standard_model',
        'similarity_bias',
        'attractive_repulsive',
        'random_adoption',
    ],
    influence_error_distribution=[
        'none',
        'N(0, 0.05)',
        'N(0, 0.1)',
    ],
    agent_activation_regime=[
        'synchronous',
        'random',
    ],
)


def get_scenario_trial_settings(trial_num, design, scenario_num):
    """Get settings for one trial from the design matrix for a nonhomogeneous scenario.

    Args:
        trial_num (int): starts at 1, specifies row index of trial in design dataframe
        design (pandas.DataFrame): experimental design matrix
        scenario_num (int): scenario number to map to nonhomogeneous model class.

    Returns:
        dict: trial factor-level elements (numbers, names, or classes)

    """
    row = design.loc[trial_num]

    settings = get_trial_settings(trial_num, design)
    settings['kwargs'] = {}  # for scenario-specific items

    # I'm hard-coding this for expediency
    if scenario_num == 1:
        settings['kwargs']['fraction_uninformed'] = row['fraction_uninformed']
    elif scenario_num == 2:
        settings['kwargs']['fraction_concord'] = row['fraction_concord']
        settings['kwargs']['fraction_left'] = row['fraction_left']
    elif scenario_num == 4:
        settings['kwargs']['fraction_stubborn'] = row['fraction_stubborn']

    return settings


def make_nonhomogeneous_design(factors_to_add, factors_to_delete, output_filename):
    """Can only run this once per runtime, else the trial counts do not come out right!"""
    global factors
    factors = nonhomogeneous_factors

    # modify factors to be scenario-specific
    for k, v in factors_to_add.items():
        factors[k] = v

    if factors_to_delete is not None:
        # if we delete a factor entirely, get_trial_settings() fails,
        # so instead just cut it to one level (which will be ignored in the Model)
        for k in factors_to_delete:
            factors[k] = [factors[k][0], ]

    design = make_full_factorial_design_matrix(output_filename=output_filename)
    print(output_filename, ': ', len(design.index))
    return design


def make_informed_uninformed_design():
    make_nonhomogeneous_design(
        {'fraction_uninformed': [0.25, 0.33, 0.50, 0.66, 0.75]},
        None,
        'design_informed_uninformed.csv'
        )
def make_concord_pa_design():
    make_nonhomogeneous_design(
        {'fraction_concord': [0.25, 0.50, 0.75], 'fraction_left': [0.00, 0.25, 0.50]},
        ['influence_model'],
        'design_concord_pa.csv'
        )
def make_bots_humans_design():
    make_nonhomogeneous_design(
        {},
        ['agent_activation_regime'],
        'design_bots_humans.csv'
        )
def make_stubborn_normal_design():
    make_nonhomogeneous_design(
        {'fraction_stubborn': [0.05, 0.15, 0.33, 0.50]},
        None,
        'design_stubborn_normal.csv'
        )


if __name__ == '__main__':
    design = load_design_file()
    # print(design.sample(5))

    print(get_trial_settings(270, design))
