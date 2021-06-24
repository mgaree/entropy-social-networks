# -*- coding: utf-8 -*-
"""Agent update/activation regimes (a.k.a. update schedulers)."""

import numpy as np


class MyBaseScheduler:
    """Base class for schedulers / activation regimes.

    This is based on an older version of mesa.time.BaseScheduler, extended for better RNG control.
    The current (07jun19) Mesa version stores agents in an OrderedDict and adds some accessors. That
    seems nice, but I don't benefit from the added overhead in this project.

    rg may be None or of type randomgen.RandomGenerator

    """

    def __init__(self, model, rg=None):
        self.model = model
        self.steps = 0
        self.time = 0
        self.agents = []

        if rg:
            self.rg = rg
        else:
            self.rg = np.random


class Synchronous(MyBaseScheduler):
    """Synchronous activation calls upon all agents to take their actions.

    ``simultaneously in one discrete time step'' cite{Flache2017} by using
    the system state from the end of the previous time step.

    This requires all agents to record the opinions of their neighbors first,
    then update their own opinions using the stored information. Otherwise,
    neighbors could update their opinions before their followers have the chance
    to use the previous opinions.

    """

    def step(self):
        """Take one synchronous step through the schedule."""
        for agent in self.agents:
            agent.step()  # Agent is reponsible for recording neighbor opinions

        for agent in self.agents:
            agent.advance()  # Agent uses recorded data to update own opinion

        self.steps += 1
        self.time += 1


class Uniform(MyBaseScheduler):
    """Activate agents one at a time in a random order each pass.

    Uniform activation activates agents one at a time, and the sequence
    is shuffled each time through the full population, akin to sampling
    without replacement.

    """

    def step(self):
        """Take one step through schedule, shuffling the sequence first."""
        # sample = self.rg.choice(self.agents, size=len(self.agents), replace=False)
        # shuffle is 50x faster than choice without replacement
        self.rg.shuffle(self.agents)  # shuffles in-place, but nothing seems to depend on original order

        # for agent in sample:
        for agent in self.agents:
            agent.step()
            agent.advance()

        self.steps += 1
        self.time += 1


class Random(MyBaseScheduler):
    """Randomly sample the population one at a time for activation.

    Random activation samples N agents from the population with replacement,
    so agents may see an unequal number of activations per turn/call to step().

    """

    def step(self):
        """Take one step through the schedule, sampling with replacement N times."""
        # sample = self.rg.choice(self.agents, size=len(self.agents), replace=True)
        # building a random list of indexes for iteration is 4x faster than sampling w/ replacement
        indexes = self.rg.randint(0, len(self.agents), size=len(self.agents))

        # for agent in sample:
        for ind in indexes:
            agent = self.agents[ind]
            agent.step()
            agent.advance()

        self.steps += 1
        self.time += 1


class RandomWeighted(MyBaseScheduler):
    """Randomly weighted-sample the population one at a time for activation.

    Random activation samples N agents from the population with replacement,
    so agents may see an unequal number of activations per turn/call to step().

    """

    def step(self):
        """Take one step through the schedule, sampling with replacement N times."""
        # TODO profile this before worrying about optimizing the performance;
        # but if needed, see https://stackoverflow.com/a/1761646/5437547
        sample = self.rg.choice(self.agents, size=len(self.agents), replace=True, p=self.weights)

        for agent in sample:
            agent.step()
            agent.advance()

        self.steps += 1
        self.time += 1
