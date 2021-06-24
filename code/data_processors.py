# -*- coding: utf-8 -*-
"""Data processors.

(All speed times mentioned in comments are based on a 100 x 500 output data array from a replication.)

"""

import networkx as nx

import numpy as np
from fast_histogram import histogram1d, histogram2d
from fast_histogram._histogram_core import (_histogram1d, _histogram2d)

import entropy

# Change this to True to use the much faster C extension, or False to use the Python functions in
# this file.
use_cdataproc = False
if use_cdataproc:
    from mycfun.cdataproc import relative_entropy_for_replication as re_c
    from mycfun.cdataproc import mutual_information_for_replication as mi_c
    from mycfun.cdataproc import transfer_entropy_for_replication as te_c


EPS = 2e-10  # range edge fix for fast_histogram

number_of_measures = 6  # okay to hardcode this I think

# array indexes
BINNING_RELATIVE  = 0
BINNING_MUTUAL    = 1
BINNING_TRANSFER  = 2
SYMBOLIC_RELATIVE = 3
SYMBOLIC_MUTUAL   = 4
SYMBOLIC_TRANSFER = 5


##
time_window_size = 2


def get_marginal_pmf_array(input_data, N, t_max, num_bins, data_range, time_window_size=None):
    """Compute probability mass function for each agent at each time step using equal bins between -1 and 1.

    "At each time step" means data is included from 0 to t for each time, so the sample size
    increases over time. `time_window_size` alters this, however.

    Args:
        input_data (np.array): an N x t_max array of raw (or symbolized) data.
        N (int): number of agents.
        t_max (int): total number of time steps.
        num_bins (int): number of bins for pmf.
        data_range (tuple): the (min, max) range for the histogram data
        time_window_size (int or None): If None, then use all data from 0 to t for each time;
          otherwise, use at most time_window_size steps

    Returns:
        np.array: array with shape (N, t_max, num_bins) containing marginal pmf at each N, t.

    """
    range_min = data_range[0]
    range_max = data_range[1] + EPS

    # 354ms vs 2630ms for naive way + numpy.histogram
    p_arr = np.empty((N, t_max, num_bins))
    if time_window_size is None:
        for i in range(N):  # for each agent
            for t in range(t_max):  # for each time step
                # fast_histogram incorrectly treats the final bin as half-open, so add EPS. this
                # can change the resulting pmfs every so slightly, but it does include all the data.
                # (otherwise data exactly on the upper range is omitted)
                #
                # note: range= is required, and bins= is # of bins, not their defs
                # p_arr[i, t, :] = histogram1d(input_data[i, 0:t+1], bins=num_bins, range=(-1, 1+EPS)) / data.size
                #
                # _histogram1d is a much faster version of fast_histogram than without the leading underscore,
                # but lacks input validation, so use the slower version during development
                p_arr[i, t, :] = _histogram1d(input_data[i, 0:t+1], num_bins, range_min, range_max) / (t + 1)


    else:  # doing as an if-else at the top level to avoid checking a local constant N*t_max
        print('using rolling window', time_window_size)
        for i in range(N):
            for t in range(t_max):
                start_t = max(0, t+1-time_window_size)
                p_arr[i, t, :] = _histogram1d(input_data[i, start_t:t+1], num_bins, range_min, range_max) / (t + 1 - start_t)
                
    return p_arr


def relative_entropy_for_replication(input_data, bins, output_data, response_var_num, data_range=(-1, 1)):
    """Compute relative entropy for each agent at each time step, relative to uniform distribution.

    "At each time step" means data is included from 0 to t for each time, so the sample size
    increases over time.

    Args:
        input_data (np.array): an N x t_max array of raw (or symbolized) data.
        bins (sequence of tuples): each tuple is the (lower edge, upper edge) of a bin
        output_data (np.array): container for output data with shape (N, t_max, # of response variables)
        response_var_num (int): array index for storing results in output_data
        data_range (tuple): the (min, max) range for the histogram data

    Returns:
        None. Output written directly to output_data array.

    """
    N, t_max = input_data.shape  # t_max may be less than 'normal' for symbolic method
    num_bins = len(bins) - 1

    # create q(i) as uniform distribution across the possible state-space
    q = 1 / num_bins

    ### by using t_max as the length, i'm omitting the final data point... TODO why did I do that?
    p_arr = get_marginal_pmf_array(input_data, N, t_max, num_bins, data_range, time_window_size)

    # vectorizing the entropy calculation gave 30% speedup doing each value in the loop
    log_p_q = np.zeros_like(p_arr)
    np.log2(p_arr / q, where=(p_arr!=0), out=log_p_q)
    output_data[:, :t_max, response_var_num] = np.sum(p_arr * log_p_q, axis=p_arr.ndim-1)


def mutual_information_for_replication(input_data, bins, output_data, response_var_num, G, data_range=(-1, 1)):
    """Compute average mutual information for each agent with its neighbors at each time step.

    Args:
        input_data (np.array): an N x t_max array of raw (or symbolized) data.
        bins (sequence of 2-ples): each tuple is the (lower edge, upper edge) of a bin
        output_data (np.array): container for output data with shape (N, t_max, # of response variables)
        response_var_num (int): array index for storing results in output_data
        G (networkx.DiGraph): graph structure whose .edges list gives each (i, j) neighbor pair
        data_range (tuple): the (min, max) range for the histogram data

    Returns:
        None. Output written directly to output_data array.

    """
    # profiling notes -- given N ~= 100:
    # naive way w/ np.histogram2d = 50s
    # naive way w/ fast_histogram = 20s
    # precompute marginal entropies, use non-double-sum MI equation = 10.3s
    # replacing histogram() with _histogram() = 7.33s
    # new way with broadcasting and such = 5.5s
    # ...but this leads to MemoryError on large networks...
    # modified new way to avoid NxN matrix = 4.6s
    # using "manual histogram" on the joint term = 3.5s

    N, t_max = input_data.shape
    num_bins = len(bins) - 1
    
    # find all the "marginal" entropy, np.sum(p_i * log_p_i), first
    p_marg = get_marginal_pmf_array(input_data, N, t_max, num_bins, data_range, time_window_size)

    log_p_marg = np.zeros_like(p_marg)
    np.log2(p_marg, where=(p_marg!=0), out=log_p_marg)
    marginal_entropy = np.sum(p_marg * log_p_marg, axis=p_marg.ndim-1)  # has shape (N, t_max)

    #
    # then find all p_ij "joint entropy" terms and compute mutual information
    #
    bins_adj = bins[:]
    bins_adj[-1] += EPS

    counts = np.empty((num_bins, num_bins), dtype=np.int32)

    binned_data = np.digitize(input_data, bins_adj)
    binned_data -= 1  # digitize leaves bin 0 for lower outliers, but we assume there are none due to truncation

    mutual_information_ij_acc = np.zeros((N, t_max))

    for i, j in G.edges:
        data_i = binned_data[i]
        data_j = binned_data[j]

        counts.fill(0)  # zeroize counts array

        for t in range(t_max):
            # data_i[t], data_j[t] is the (i, j) bin id pair at time t
            counts[data_i[t], data_j[t]] += 1  # increment that bin

            # the matrix relationship of the pmf is irrelevant, so instead of use a where= clause
            # in the log2(), just filter out all empty states
            start_t = max(0, t+1-time_window_size)  # have to account for rolling window
            p_ijt = counts[counts > 0] / (t + 1 - start_t) ## -- still not working right for time window...something is off somewhere, maybe below?
            # p_ijt = counts[counts > 0] / (t + 1)
            log_p_ijt = np.log2(p_ijt)  # compute the log of just the current pmf

            # at the end, we care about the average MI; accumulate for now, then divide by outdegree later
            mutual_information_ij_acc[i, t] += \
                np.sum(p_ijt * log_p_ijt) - marginal_entropy[i, t] - marginal_entropy[j, t]

    # note: the manual histogram approach *is* faster here than _histogram2d while it is not faster in
    # transfer_entropy_for_replication(). I suspect this is because here, _histogram2d was being called
    # O(|E| x t_max) times, while there was O(N x t_max), and |E| > N.

    adjacency_mtx = nx.to_numpy_array(G, dtype=np.int)
    outdegrees = adjacency_mtx.sum(axis=1).reshape(-1, 1)

    # and then average per agent based on their outdegree
    # some agents follow no one, so their outdegree is zero; catch for that
    output_data[:, :t_max, response_var_num] = np.divide(mutual_information_ij_acc, outdegrees, where=(outdegrees!=0))


def transfer_entropy_for_replication(input_data, bins, output_data, response_var_num, G, data_range=(-1, 1)):
    """Compute average transfer entropy for each agent with its neighbors at each time step.

    Args:
        input_data (np.array): an N x t_max array of raw (or symbolized) data.
        bins (sequence of 2-ples): each tuple is the (lower edge, upper edge) of a bin
        output_data (np.array): container for output data with shape (N, t_max, # of response variables)
        response_var_num (int): array index for storing results in output_data
        G (networkx.DiGraph): graph structure whose .edges list gives each (i, j) neighbor pair
        data_range (tuple): the (min, max) range for the histogram data; not currently used here
            because histogramdd autocomputes as needed from the list of bins

    Returns:
        None. Output written directly to output_data array.

    """
    # profiling notes -- given N ~= 100:
    # loops for p_*, vectorized logs, final TE in loops = 2:30min
    # same, final TE in one-line array ops = 2:50min
    # "naive mode" with loops and minor caching = 50s (!?) -- may be related to sparsness of adj
    # ...but like MI, this causes MemoryError for large N
    # after editing to avoid NxN matrix = 43s
    # after replacing histogramdd with a "manual histogram" method = 9s

    N, t_max = input_data.shape
    num_bins = len(bins) - 1

    range_min = data_range[0]
    range_max = data_range[1] + EPS

    # find all the "marginal" entropy, np.sum(p_i * log_p_i), first
    p_marg = get_marginal_pmf_array(input_data, N, t_max, num_bins, data_range)

    log_p_marg = np.zeros_like(p_marg)
    np.log2(p_marg, where=(p_marg!=0), out=log_p_marg)
    marginal_entropy = np.sum(p_marg * log_p_marg, axis=p_marg.ndim-1)  # has shape (N, t_max)

    # find all the entropy terms for (i_{t+1}, i_t) -- remarkably, this is about 5% faster than caching as we go
    i1it_entropy_terms = np.zeros((N, t_max - 1))

    for i in range(N):
        for t in range(t_max - 1):  # t_max - 1 is correct here, because of the lag b/t i1 and i
            data_i1 = input_data[i, 1:t+2]
            data_i = input_data[i, 0:t+1]

            h = np.ravel(_histogram2d(
                data_i1, data_i, num_bins, range_min, range_max, num_bins, range_min, range_max))
            p_i1it = h[h > 0] / (t + 1)
            log_p_i1it = np.log2(p_i1it)  # compute the log of just the current pmf

            # at the end, we care about the average MI; accumulate for now, then divide by outdegree later
            i1it_entropy_terms[i, t] = np.sum(p_i1it * log_p_i1it)

    # note: the following method for manually building the counts array over time, when applied to
    # building the i1it_entropy_terms in the previous block, is not measurably faster than using
    # _histogram2d, despite being more "efficient".

    #
    # "joint" entropy terms
    #
    bins_adj = bins[:]
    bins_adj[-1] += EPS

    counts = np.empty((num_bins, num_bins, num_bins), dtype=np.int32)

    binned_data = np.digitize(input_data, bins_adj)
    binned_data -= 1  # digitize leaves bin 0 for lower outliers, but we assume there are none due to truncation

    transfer_entropy_ij_acc = np.zeros((N, t_max))

    for i, j in G.edges:
        data_i1 = binned_data[i, 1:]
        data_i = binned_data[i, 0:-1]
        data_j = binned_data[j, 0:-1]

        data = np.stack((data_i1, data_i, data_j), axis=0)

        counts.fill(0)  # zeroize counts array

        for t in range(t_max - 1):
            cur_step = data[:, t]  # [i1 bin id, i bin id, j bin id] for time t
            counts[tuple(cur_step)] += 1  # cur_step occupies a single bin in 3d, so increment that bin
            ### TODO replace += 1...later /(t+1) with += 1/(t+1) ; it's about 9% faster??

            # axes 0, 1, 2 = i_t1, i_t, j_t
            p_i1ijt = counts / (t + 1)  # to get probabilities vs. counts
            p_ijt = p_i1ijt.sum(axis=0)

            # the matrix relationship of the pmf is irrelevant, so instead of use a where= clause
            # in the log2(), just filter out all empty states
            p_i1ijt = p_i1ijt[p_i1ijt > 0]
            p_ijt = p_ijt[p_ijt > 0]

            # at the end, we care about the average MI; accumulate for now, then divide by outdegree later
            transfer_entropy_ij_acc[i, t] += \
                np.sum(p_i1ijt * np.log2(p_i1ijt)) \
                - i1it_entropy_terms[i, t] \
                - np.sum(p_ijt * np.log2(p_ijt)) \
                + marginal_entropy[i, t]

    adjacency_mtx = nx.to_numpy_array(G, dtype=np.int)
    outdegrees = adjacency_mtx.sum(axis=1).reshape(-1, 1)

    # and then average per agent based on their outdegree
    # some agents follow no one, so their outdegree is zero; catch for that
    output_data[:, :t_max, response_var_num] = np.divide(transfer_entropy_ij_acc, outdegrees, where=(outdegrees!=0))


def process_raw_data_into_entropy_timeseries(raw_data, G):
    """Return time series of each entropy response variable for each agent.

    For each agent at each time step, the following entropy measures are computed:
        - binning + relative entropy
        - binning + mutual information
        - binning + transfer entropy
        - symbolic + relative entropy
        - symbolic + mutual information
        - symbolic + transfer entropy

    The output of this method has been validated against entropy.py:get_entropy, which has been
    validated against manual results in entropy.py:testing.

    The C methods have been validated against the Python methods in this module. As a baseline for
    speed, one iteration of process_raw_data_into_entropy_timeseries(*) on a 10K agent by 500 time
    step instance took 16.5 minutes for Python and 1.5 minutes for C--a 91% speedup.

    Args:
        raw_data (np.array): an N x t_max array of raw (or symbolized) data.
        G (networkx.DiGraph): graph structure whose .edges list gives each (i, j) neighbor pair

    Returns:
        np.array: output data with shape (N, t_max, number of measures); each [N, :, number] is a
            time series of entropy values for that entropy measure for a given agent.

    """
    N, t_max = raw_data.shape

    output_data = np.zeros((*raw_data.shape, number_of_measures))  # each (i, t) will be an array

    if use_cdataproc:
        # put graph information in a more convenient way for using C; of course C can do this,
        # but I didn't want to spend the time figuring out how (writing C is hard)
        edges_as_nparray = np.array(G.edges, dtype=np.int32)  # convenience for passing edge list to C
        outdegrees_array = np.zeros(N, dtype=np.int32)
        for i, v in G.out_degree:
            outdegrees_array[i] = v

        # =========================
        # binning approach
        # =========================

        bins = entropy.bins_map['relative entropy']
        num_bins = len(bins - 1)
        digitized_raw_data = np.digitize(raw_data, bins).astype(np.int32) - 1  # -1 to make bins base 0
        re_c(digitized_raw_data, num_bins, output_data, BINNING_RELATIVE)

        bins = entropy.bins_map['mutual information']
        num_bins = len(bins - 1)
        digitized_raw_data = np.digitize(raw_data, bins).astype(np.int32) - 1
        mi_c(digitized_raw_data, num_bins, output_data, BINNING_MUTUAL, edges_as_nparray, outdegrees_array)

        bins = entropy.bins_map['transfer entropy']
        num_bins = len(bins - 1)
        digitized_raw_data = np.digitize(raw_data, bins).astype(np.int32) - 1
        te_c(digitized_raw_data, num_bins, output_data, BINNING_TRANSFER, edges_as_nparray, outdegrees_array)

        # # =========================
        # # symbolic approach
        # # =========================

        symbolized_raw_data = entropy.symbolize_time_series(raw_data).astype(np.int32) - 1  # note that this has shape (N, t-m+1)
        bins = entropy.bins_map['symbolic']
        num_bins = len(bins - 1)

        # the C methods expect input and output to have the same length t_max, but that's not the case for
        # symbolic method, so we slice output_data, adjusting for the pattern length
        re_c(symbolized_raw_data, num_bins, output_data[:, :t_max-entropy.m+1, :], SYMBOLIC_RELATIVE)
        mi_c(symbolized_raw_data, num_bins, output_data[:, :t_max-entropy.m+1, :], SYMBOLIC_MUTUAL, edges_as_nparray, outdegrees_array)
        te_c(symbolized_raw_data, num_bins, output_data[:, :t_max-entropy.m+1, :], SYMBOLIC_TRANSFER, edges_as_nparray, outdegrees_array)

    else:
        # =========================
        # binning approach
        # =========================

        bins = entropy.bins_map['relative entropy']
        relative_entropy_for_replication(raw_data, bins, output_data, BINNING_RELATIVE)

        bins = entropy.bins_map['mutual information']
        mutual_information_for_replication(raw_data, bins, output_data, BINNING_MUTUAL, G)

        #bins = entropy.bins_map['transfer entropy']
        #transfer_entropy_for_replication(raw_data, bins, output_data, BINNING_TRANSFER, G)

        # =========================
        # symbolic approach
        # =========================

        symbolized_raw_data = entropy.symbolize_time_series(raw_data)  # note that this has shape (N, t-m+1)
        symbols_range = (1, entropy.symbol_amap.shape[0])
        bins = entropy.bins_map['symbolic']

        # data_range arg needed here because otherwise the histogram functions expect data in [-1, 1]
        relative_entropy_for_replication(symbolized_raw_data, bins, output_data, SYMBOLIC_RELATIVE, data_range=symbols_range)
        mutual_information_for_replication(symbolized_raw_data, bins, output_data, SYMBOLIC_MUTUAL, G, data_range=symbols_range)
        #transfer_entropy_for_replication(symbolized_raw_data, bins, output_data, SYMBOLIC_TRANSFER, G, data_range=symbols_range)

    return output_data


# FUTURE TODO: something to build all trial-level output files into single database file
