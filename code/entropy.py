# -*- coding: utf-8 -*-
"""Methods for calculating entropy for the simulation responses on the raw replication data.

Aggregation into trial-level response variables is done elsewhere.

Note: at this point, most of this module is here for posterity/reference, except for the code for
the symbolic method. Entropy calculations for actual trial runs are done in data_processors.py. The
forms there are mainly optimized and somewhat harder to interpret, so these original forms are
available for reference.

These functions have also been validated through manual test cases (`testing()`), so they are good
tools for validating the results created by data_processors.

"""

import numpy as np


#
# Support for symbolic method
#

m = 3  # subchain length per symbol

# same map as Borge-Holthoefer2016, manually entered
# indexes decreased by 1 to be base-0, and letters swapped for integers to support np.histogram
symbol_amap = np.array([
    [0, 1, 2, 1],
    [0, 2, 1, 2],
    [2, 0, 1, 3],
    [1, 0, 2, 4],
    [1, 2, 0, 5],
    [2, 1, 0, 6],
])

# using bitmasking and a 1-d lookup table yields order of magnitude speedup in symbolize_time_series
symbol_amap_leftcols = symbol_amap[:, :-1]  # remove last column, the symbol number
symbol_mask = np.array([2**i for i in range(m)])  # [1, 2, 4, ...]

masked_leftcols = np.dot(symbol_amap_leftcols, symbol_mask)  # bitmasks each pattern

symbol_lut = np.zeros(max(masked_leftcols) + 1, dtype=np.int8)
symbol_lut[masked_leftcols] = symbol_amap[:, -1]  # results in [0 0 0 0 6 5 3 0 2 4 1]

# -- no longer using symbol_array, retaining for posterity
# # if I store symbol_amap as a tensor, it's a lookup by 3d index instead of search...may be much faster?
# symbol_array = np.zeros((3,3,3), dtype=np.int8)
# for i, j, k, v in symbol_amap:
#     symbol_array[i, j, k] = v


def rolling_window(a, window):
    """Build array of rolling windows.

    Method from https://rigtorp.se/2011/01/01/rolling-statistics-numpy.html

    """
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def symbolize_time_series(data, m=m, mask=symbol_mask, lut=symbol_lut):
    """Convert raw opinion time series data into time series of symbol/pattern ids.

    Args:
        data (np.array): ... works on 2d array
        m (int): pattern length
        mask (np.array): sequence of ints for bitmasking, e.g. [1, 2, 4, 8]
        lut (np.array): lookup table where lut[bitmask value] = pattern id

    Returns:
        np.array: symbolized data

    """
    patterns = np.argsort(rolling_window(data, m), kind='quicksort')
    n = np.empty((patterns.shape[0], patterns.shape[1]), dtype=np.int8)

    # -- keeping in comments for posterity, numpy learning, examples of optimization

    # 955ms mean function time using np.ndindex to iterate
    # for i, j in np.ndindex(n.shape):
    #     pat = patterns[i, j, :]
    #     n[i, j] = symbol_array[tuple(pat)]

    # 750ms mean function time using double loop and range() for i, j to iterate
    # for i in range(n.shape[0]):
    #     for j in range(n.shape[1]):
    #         pat = patterns[i, j, :]
    #         n[i, j] = symbol_array[tuple(pat)]

    # 671ms for ravel-vector-unravel
    # rows, cols = n.shape
    # n = np.ravel(n)
    # patterns = patterns.reshape(1, -1, m)[0, :]  # slicing away the 1st dimension saves ~20ms
    #
    # for i, pat in enumerate(patterns):
    #     n[i] = symbol_array[(*pat,)]  # (*pat,) vs tuple(pat) saves ~12ms
    # n = n.reshape(rows, cols)

    # 25ms (!) mean function time; requires building different symbol lookup table, but that's free
    rows, cols = n.shape  # store shape for unraveling
    n = np.ravel(n)
    patterns = patterns.reshape(1, -1, m)[0, :]  # make into a 2d matrix

    patterns = np.dot(patterns, mask)  # bitmask each pattern in the array, yielding a 1d matrix of mask values
    n = lut[patterns]  # for each mask value in patterns, set that position in n equal to lookup table value (symbol id)

    n = n.reshape(rows, cols)

    return n


# Bins management

# selected with awareness of a planned maximum 500 observations
# MI and TE use much smaller values because these are bins per dimension. MI gets squared and
# TE gets cubed. I wanted to avoid having cases of at most one observation per bin.
suggested_num_bins = {'relative entropy': 50, 'mutual information': 7, 'transfer entropy': 6}
symbolic_num_bins = symbol_amap.shape[0]
bins_map = {
    'relative entropy': np.linspace(-1, 1, suggested_num_bins['relative entropy'] + 1),
    'mutual information': np.linspace(-1, 1, suggested_num_bins['mutual information'] + 1),
    'transfer entropy': np.linspace(-1, 1, suggested_num_bins['transfer entropy'] + 1),
    'symbolic': np.linspace(0.5, symbolic_num_bins + 0.5, symbolic_num_bins + 1),  # center the bins on the data
}


#
# Entropy measures
#

def relative_entropy(p, q):
    """Compute relative entropy D(p || q).

    Relative entropy (or Kullback-Leibler distance) is most basically defined as

        sum_i p(i) log [ p(i) / q(i) ]

    where i is the set of states in the probability mass functions of p and q.

    Args:
        p (array-like): vector form of probability mass function TODO clarify docs for providing 3d vector
        q (array-like or float): vector form of probability mass function, or constant if q ~ uniform

    Returns:
        float: relative entropy

    """
    # the 'where' clause protects against log(0) = -inf; instead log(0) = 0
    # using on a zeroized array to protect against 'nan's in the locations that the where clause is not met
    log_p_q = np.zeros_like(p)
    np.log2(p / q, where=(p!=0), out=log_p_q)

    return np.sum(p * log_p_q, axis=p.ndim-1)  # specifying the last axis lets us provide multiple i's at once


def mutual_information(p_i, p_j, p_ij):
    """Compute mutual information M_IJ.

    Mutual information is most basically defined as the double sum

        sum_{i,j} p(i,j) log [ p(i,j) / p(i)p(j) ].

    TODO: Writeup documentation for the alternate form of the equation below (does not use division in the logs)

    Args:
        p_i (array-like): vector form of probability mass function for process I
        p_j (array-like): vector form of probability mass function for process J
        p_ij (array-like): matrix form of joint pmf for I and J

    Returns:
        float: mutual information of I and J

    """
    # I initialize zero matrixes and put log() output to them because otherwise, I would rarely and randomly get 'nan'
    # returned instead of zero. I think it had to do with the where= clause and the state of the internal,
    # uninitialized array used for log() output. see the comment at the end for the compact form.
    log_p_ij = np.zeros_like(p_ij)
    log_p_i = np.zeros_like(p_i)
    log_p_j = np.zeros_like(p_j)

    np.log2(p_ij, where=(p_ij!=0), out=log_p_ij)
    np.log2(p_i, where=(p_i!=0), out=log_p_i)
    np.log2(p_j, where=(p_j!=0), out=log_p_j)

    return np.sum(p_ij * log_p_ij) - np.sum(p_i * log_p_i) - np.sum(p_j * log_p_j)


def transfer_entropy(p_i1ij, p_i1i, p_ij, p_i):
    """Compute transfer entropy TE_J->I.

    Transfer entropy is most basically defined as the triple sum

        sum_{i_{t+1}, i_t, j_t} p(i_{t+1}, i_t, j_t) log [ p(i_{t+1} | i_t, j_t) / p(i_{t+1} | i_t)]

    where here, we look at only a single time step for i_t and j_t; the complete definition lets those
    be variable length vectors. I_{t+1} is effectively the lag-1 time series of I.

    However, below we use a different form of the equation that uses a linear combination of the
    entropies of the individual input arguments. This avoids the mess of computing conditional
    probability mass functions or properly broadcasting matrices during multiplication,
    which might actually be easy but I failed to find that in the available time, in part due to
    working with 3d data. The equation used below is

        ( sum_{i1,i,j} p_i1ij * log p_i1ij )   -   ( sum_{i1,i} p_i1i * log p_i1i )
             - ( sum_{i,j} p_ij * log p_ij )   +   ( sum_{i} p_i * log p_i )

    and is based on "Transfer Entropy" by Gençağa, Deniz (2018). The triple sum version may be faster
    due to needing fewer logarithm evaluations, but as written below, it's about 40us on a 100x500
    input data set.

    Args:
        p_i1ij (array-like): matrix form of joint pmf for I_{t+1}, I_t, and J_t
        p_i1i (array-like): matrix form of joint pmf for I_{t+1} and I_t
        p_ij (array-like): matrix form of joint pmf for I_t and J_t
        p_i (array-like): vector form of probability mass function for process I_t

    Returns:
        float: transfer entropy of J onto I

    """
    # see relative_entropy() comments on why we zeroize output matrixes
    log_p_i1ij = np.zeros_like(p_i1ij)
    log_p_i1i = np.zeros_like(p_i1i)
    log_p_ij = np.zeros_like(p_ij)
    log_p_i = np.zeros_like(p_i)

    np.log2(p_i1ij, where=(p_i1ij!=0), out=log_p_i1ij)
    np.log2(p_i1i, where=(p_i1i!=0), out=log_p_i1i)
    np.log2(p_ij, where=(p_ij!=0), out=log_p_ij)
    np.log2(p_i, where=(p_i!=0), out=log_p_i)

    return np.sum(p_i1ij * log_p_i1ij) - np.sum(p_i1i * log_p_i1i) - np.sum(p_ij * log_p_ij) + np.sum(p_i * log_p_i)


#
# Dispatch function
#


def get_entropy(data_mode, entropy_measure, raw_input, i, j=None, num_bins=None, t_stop=None):
    """Get entropy value using specified data handling mode and entropy measure for single agent/agent pair.

    data_mode (str): one of {binning, symbolic}
    entropy_measure (str): one of {relative entropy, mutual information, transfer entropy}
    raw_input (np.array): raw data array with shape (agents, time); each row is one agent time series
       FUTURE: change so this can be either raw input or symbolic; see note below in "symbolize data"
    i (int): row index (agent ID) of raw_input for entropy calculation
    j (int): row index for 2nd agent in mutual information/transfer entropy calculation;
        required for mutual information and transfer entropy, and ignored for relative entropy
    num_bins (int, optional): number of bins per dimension; joint pmf will have ^2 or ^3 bins;
        if omitted and using binning data mode, default value is selected from suggested_num_bins;
        if using symbolic data mode, num_bins is based on symbol map and user input is ignored
    t_stop (int, optional): FUTURE

    """
    # extract data sample
    if entropy_measure == 'relative entropy':
        data = raw_input[i, :].reshape(1, -1)
    elif entropy_measure == 'mutual information':
        try:
            data = raw_input[(i, j), :]
        except Exception:
            if j is None:
                raise ValueError('j must be specified for the selected entropy_measure')
            else:
                raise
    elif entropy_measure == 'transfer entropy':
        try:
            data_it1 = raw_input[i, 1:]
            data_i = raw_input[i, :-1]
            data_j = raw_input[j, :-1]
        except Exception:
            if j is None:
                raise ValueError('j must be specified for the selected entropy_measure')
            else:
                raise

        data = np.stack((data_it1, data_i, data_j), axis=0)
    else:
        raise ValueError('Invalid entropy_measure')

    # prepare bins and symbolize data if needed

    if data_mode == 'binning':
        # create constant set of bins
        if num_bins is None:
            bins = bins_map[entropy_measure]  # this way, create bins once and reuse
        else:
            bins = np.linspace(-1, 1, num_bins + 1)  # num_bins+1 edges = num_bins bins
    elif data_mode == 'symbolic':
        bins = bins_map['symbolic']

        # symbolize data
        # FUTURE: it's going to be much faster to symbolize the entire replication output at once
        # rather than each i, j or so subset
        # so later, refactor such that the caller is responsible for symbolizing, then get_entropy will assume
        # you've passed in symbolic data
        data = symbolize_time_series(data)
    else:
        raise ValueError('Invalid data_mode')

    # calculate entropy

    res = None

    if entropy_measure == 'relative entropy':
        # bin the data and make pmf, filtering out empty bins
        # (since those don't represent actual states held by the process)
        counts = np.histogram(data, bins)[0]
        p = counts / data.size  # only true for 1d data

        # create q(i) as uniform distribution across the possible state-space
        q = 1 / (len(bins) - 1)

        res = relative_entropy(p, q)

    elif entropy_measure == 'mutual information':
        # create pmfs
        counts = np.histogram2d(data[0], data[1], bins)[0].T  # transposing because hist2d orders the axis "backwards"
        p_ij = counts / data.shape[1]  # data.shape[1] = number of time steps

        # marginals found by the row/column sums
        p_i = np.sum(p_ij, axis=1)
        p_j = np.sum(p_ij, axis=0)

        res = mutual_information(p_i, p_j, p_ij)

    elif entropy_measure == 'transfer entropy':
        # create pmfs
        counts = np.histogramdd(data.T, (bins, bins, bins))[0]  # .T because histogramdd needs shape (obs, dim)

        # axes 0, 1, 2 = i_t1, i, j
        p_i1ij = counts / data.shape[1]  # to get probabilities vs. counts
        p_i1i = p_i1ij.sum(axis=2)
        p_ij = p_i1ij.sum(axis=0)
        p_i = p_i1ij.sum(axis=(0, 2))

        res = transfer_entropy(p_i1ij, p_i1i, p_ij, p_i)

    else:
        raise ValueError('Invalid entropy_measure')

    return res


#
# Testing code for module
#

def testing():
    """Perform some test calculations against manually verified results."""
    from math import isclose

    raw_input = np.array([
        [-1, -.8, -.6, -.4, -.2, 0, .2, .4, .6, .8],
        [0, .1, .2, .3, .4, .5, .4, .3, .2, .1],
        [.2, .2, .2, .2, .2, .2, .2, .2, .2, .2],
        [-1, 1, -1, 1, -1, 1, -1, 1, -1, 1]
    ])

    # test cases
    # for relative entropy: (i, result)
    re_b = [(0, 0.07807), (1, 1.63904), (2, 3), (3, 2)]
    re_s = [(0, 2.58496), (1, 1.17932), (2, 2.58496), (3, 1.58496)]

    # for mutual information: (i, j, result)
    mi_b = [(0, 1, 0.88129), (0, 2, 0), (1, 2, 0), (2, 1, 0), (3, 0, 0.04902)]
    mi_s = [(0, 1, 0), (0, 2, 0), (1, 2, 0), (2, 1, 0), (3, 0, 0)]

    # for transfer entropy: (i, j, result)
    te_b = [(0, 1, 0), (1, 3, 0.21113), (0, 3, 0.22222)]
    te_s = [(0, 1, 0), (1, 3, 0.17787), (0, 3, 0)]

    num_failures = 0

    print("-testing relative entropy-")

    print("--by binning--")
    for i, v in re_b:
        res = get_entropy('binning', 'relative entropy', raw_input, i, j=None, num_bins=8)
        if not isclose(res, v, abs_tol=1e-5):
            print("failure for {}, {} - got {}".format(i, v, res))
            num_failures += 1

    print("--by symbolics--")
    for i, v in re_s:
        res = get_entropy('symbolic', 'relative entropy', raw_input, i, j=None, num_bins=8)
        if not isclose(res, v, abs_tol=1e-5):
            print("failure for {}, {} - got {}".format(i, v, res))
            num_failures += 1

    print("-testing mutual information-")

    print("--by binning--")
    for i, j, v in mi_b:
        res = get_entropy('binning', 'mutual information', raw_input, i, j, num_bins=3)
        if not isclose(res, v, abs_tol=1e-5):
            print("failure for {}, {}, {} - got {}".format(i, j, v, res))
            num_failures += 1

    print("--by symbolics--")
    for i, j, v in mi_s:
        res = get_entropy('symbolic', 'mutual information', raw_input, i, j, num_bins=3)
        if not isclose(res, v, abs_tol=1e-5):
            print("failure for {}, {}, {} - got {}".format(i, j, v, res))
            num_failures += 1

    print("-testing transfer entropy-")
    print("--by binning--")
    for i, j, v in te_b:
        res = get_entropy('binning', 'transfer entropy', raw_input, i, j, num_bins=3)
        if not isclose(res, v, abs_tol=1e-5):
            print("failure for {}, {}, {} - got {}".format(i, j, v, res))
            num_failures += 1

    print("--by symbolics--")
    for i, j, v in te_s:
        res = get_entropy('symbolic', 'transfer entropy', raw_input, i, j, num_bins=3)
        if not isclose(res, v, abs_tol=1e-5):
            print("failure for {}, {}, {} - got {}".format(i, j, v, res))
            num_failures += 1

    print()
    print()
    print("failures reported: {}".format(num_failures))
    if num_failures > 0:
        print("-----TESTING FAILED-----")
    else:
        print("Testing passed!")

    return


if __name__ == "__main__":
    testing()
