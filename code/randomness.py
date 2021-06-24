# -*- coding: utf-8 -*-
"""Helpers for managing random number generation."""

from enum import IntEnum, auto
from networkx.utils import PythonRandomInterface


class RNGStream(IntEnum):
    """Convenience enum for RNG stream management."""

    GRAPH_GENERATOR = auto()  # distribution for making graphs and converting to directed in networks.py
    NEIGHBOR_WEIGHTS = auto()
    INITIAL_OPINIONS = auto()
    INTERNAL_BIAS = auto()
    SUSCEPTIBILITY_INFLUENCE = auto()
    ERROR_DISTRIBUTION = auto()
    AGENT_ACTIVATION = auto()  # used for shuffling/selecting agents in Uniform/Random activation regime
    SINGLE_NEIGHBOR_SELECTION = auto()  # for selecting a neighbor in the RandomAdoption, SimilarityBias, and AttractiveRepulsive agents


"""
REPLICATION_ISEED_STREAM_ARRAY is a constant array of distinct seed numbers used for model iseed values for
each replication. This array was created as follows:

    from randomgen import RandomGenerator, PCG64
    rg = RandomGenerator(PCG64(8675309))
    REPLICATION_ISEED_STREAM_ARRAY = rg.randint(10_000_000, size=120)

We chose to have a constant iseed array to be better able to stop a trial after some replication and
then resume the trial on the next replication. Previously, we let the model call rg.randint() on its
own to create the seed. However, we would need to save the generator's state between runs, which is
more than just the seed, and if a replication was terminated in an uncontrolled way, the state could
be lost. Now, the model will select its starting iseed simply as REPLICATION_ISEED_STREAM_ARRAY[rep_num].
"""
REPLICATION_ISEED_STREAM_ARRAY = [5560691, 2822670, 3466006, 8852913, 9199637, 3515365, 4224451,
         60959, 1112940, 7639305, 9784311, 3977416, 9112512, 5494304,
       5641057, 9154956,  635151, 2016388, 5161333, 7544378, 4226982,
       8817008, 5442735, 3103087, 5984773, 3050395, 2072985, 7413792,
       7737136, 8345976, 9201081, 8665562, 8291228, 2335328, 9899581,
       1984050, 5950936, 8544523, 3955923, 6374185, 8489146, 1845941,
       2138654, 3812282, 8135659, 7132472, 7490599, 1992484, 1707479,
       4776839, 9122799, 1266329, 3255737, 6624620, 6880797, 8878783,
       3237246,  649312,  957355, 6774482, 4569284, 7332985, 6272114,
       2949643, 4427136, 5112127, 8160224, 7998346, 3735207, 3574051,
       1733449, 5276564, 3648935, 2085761,  535558, 6913948, 5045915,
       2590638, 1519729, 5466587, 1567966, 8795149, 3084642, 3867632,
       4556126, 1918148, 4729020, 2656041, 3838280, 7451316, 8565418,
       3507822, 2793870,  662888, 6957750, 9154661,  645528,   96937,
       3066628, 5719520,  712654, 1709766, 3249045, 6633805, 1836936,
       6798357, 1354256, 7326255, 2413177, 5015533, 3672363, 1155323,
       5246148, 3644300, 4449186, 5079619, 6244908, 7277004,  515009,
        490333]  # this array contains 120 values


class RandomGenRandomInterface(PythonRandomInterface):
    """Wrapper/interface class for random number generator use with NetworkX.

     NetworkX depends on random number generators to be either Python's random or numpy.random.
    It wraps those in its own PythonRandomInterface for use in random graph generators.

    I want to use randomgen generators, so I'm subclassing PythonRandomInterface to wrap randomgen's
    interface, then in model.py, I create a randomgen generator, wrap it with the new interface, and
    pass that to the random graph generators as their `seed` arg (seed can be int, certain packages,
    a RandomState instance, or a PythonRandomInterface).

    This will give me the same control mechanism over networkx's use of randomness as I have over every
    other source.

    """

    def __init__(self, rng=None):
        self._rng = rng

    def random(self, n=None):
        return self._rng.random_sample(n)

    def uniform(self, a, b):
        return self._rng.random_sample(a, b)

    def randrange(self, a, b=None):
        return self._rng.randint(a, b)

    def choice(self, seq):
        # return self._rng.choice(seq)  # using _rng.choice raises a 0.4s process up to 176s -- 440x increase!
        return seq[self._rng.randint(0, len(seq))]

    def gauss(self, mu, sigma):
        return self._rng.normal(mu, sigma)

    def shuffle(self, seq):
        return self._rng.shuffle(seq)

    def sample(self, seq, k):
        return self._rng.choice(list(seq), size=(k,), replace=False)

    def randint(self, a, b):
        return self._rng.randint(a, b + 1)

    def expovariate(self, scale):
        return self._rng.exponential(1 / scale)

    def paretovariate(self, shape):
        return self._rng.pareto(shape)
