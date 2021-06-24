# -*- coding: utf-8 -*-
"""Some configuration constants for the simulation."""

MIN_OPINION = -1
MAX_OPINION = 1

MAX_TIME_STEPS = 500
NUM_REPLICATIONS = 1

# Based on Deffuant2000 Fig. 2, rescaled to our opinion range
SIMILARITY_BIAS_MU = 0.5  # not rescaled, because we only want to move half the difference
SIMILARITY_BIAS_EPSILON = 0.2 * (MAX_OPINION - MIN_OPINION)

ATTRACTIVE_REPULSIVE_MU = SIMILARITY_BIAS_MU
