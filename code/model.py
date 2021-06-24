# -*- coding: utf-8 -*-
"""Model implementation for simulation."""

import numpy as np
import networkx as nx

from randomgen import RandomGenerator, PCG64

from constants import MIN_OPINION, MAX_OPINION
import networks
from randomness import RNGStream, RandomGenRandomInterface


class Model:
    """Model class. Borrowed heavily from my 2018 WSC project."""

    def __init__(self, iseed, max_steps, network_parameters,
                 agent_class, scheduler_class, error_distribution=None, network_digraph=None):
        """Create a new model, configured based on design point inputs.

        Args:
            iseed (int): seed for the random number generator. We use multiple
              streams, and each stream will be initialized with this seed, as in
              PCG64(seed=iseed, stream=<stream num>).
            max_steps (int): maximum number of time steps to run the model.
            network_parameters (tuple): Information for creating network graph using networks.py,
              of form (generator function name, (*args)).
            agent_class (obj): Agent class for model from agents.py.
            scheduler_class (obj): Scheduler for activation regime, from
              activation_regimes.py.
            error_distribution (tuple or None): Information for the distribution of error
              terms used by agents in their influence model. If None, then all error terms
              are zero. Otherwise, (distribution function name, *args) for distribution function.
              For example, error_distribution = ('normal', 0, 1).
            network_digraph (networkx.DiGraph, optional): If given, use this for the network and
              ignore network_parameters. Used mainly during testing.

        Attributes:
            schedule (obj): Activation regime / update scheduler for agents.
            running (boolean): Indicates if the model should continue running.
            datacollector (DataCollector): Object for step-wise data
              collection.
            agents (list of Agents): Model agents.
            rg (list of RandomGenerators): Streams for RNG. See RNGStream enum
              for use cases.
            G (NetworkX.DiGraph): Graph structure of network.
            N (int): Number of nodes/agents in G.
            error_distribution (tuple or None): If not None, then a tuple of the form
              (distribution function name, (*args, population size)).

        """
        # Prepare random number streams
        self.iseed = iseed
        # +1 because auto.enum starts at 1
        self.rg = [RandomGenerator(PCG64(self.iseed, i + 1)) for i in range(len(RNGStream) + 1)]

        self.max_steps = max_steps  # some agent classes need this, so save it first

        # Create network; see randomness for details on this RNG approach
        if network_digraph is not None and isinstance(network_digraph, nx.DiGraph):
            self.G = network_digraph
        else:
            network_rg = RandomGenRandomInterface(self.rg[RNGStream.GRAPH_GENERATOR])
            self.G = getattr(networks, network_parameters[0])(*network_parameters[1], network_rg)
            self.G = networks.prepare_graph_for_trial(self.G, network_rg)  # make directed, connected, & free of self-loops

        # this could be less than the design parameter due to connected component < full graph
        self.N = self.G.number_of_nodes()

        # Distribution and array for error terms. It is faster to have the model create a vector
        # of random variates for the full population each time step than to have each agent generate
        # them one at a time.
        if error_distribution is None:
            # No error terms for this model, so create a constant array of zeros
            self.error_distribution = None
            self.error_terms = np.zeros(self.N, dtype=np.int8)
        else:
            # Modify self.error_distribution to include population size self.N, for convenience
            self.error_distribution = (error_distribution[0], (*error_distribution[1:], self.N))

            # Prepare error terms if needed for agent initialization
            self.refresh_error_terms()

        #
        # Prepare opinion data / data stores
        #
        # Agents will read and write their opinion values to the model during their finalize()
        # process. This should allow a more matrix-numpy approach to things than iterating
        # over lists of neighbor objects. This will be the same data matrix for the final output.
        #
        # Time savings on 10K run: ~12sec.
        #
        # raw_data is filled by model after each time step in collect_opinion_data()
        # latest_opinions is filled by each agent during their finalize()
        # agents will read neighbor opinions from latest_opinions rather than accessing the
        # neighbor objects directly.
        self.raw_data = np.zeros((self.N, self.max_steps + 1), dtype=np.float64)  # +1 due to initial opinions
        self.create_initial_opinions()  # agent.__init__ may expect this
        self.latest_opinions = np.copy(self.agent_initial_opinions)

        # Create agents - unique_ids must be [0..N] because they are used for indexing in raw_data
        self.agents = [agent_class(i, self) for i in range(self.N)]

        # Create scheduler
        # schedule can shuffle its list of agents, so send it a new copy to preserve model's list
        self.schedule = scheduler_class(self, self.rg[RNGStream.AGENT_ACTIVATION])
        self.schedule.agents = list(self.agents)

        self.running = True

        # end init()

    def create_initial_opinions(self):
        """Create array of initial opinion values for agents to refer to.

        Agents are free to generate their own values if the generation method here is not suitable.
        Creating an array of random numbers once is slightly faster than creating each number
        singly (~1s over 10K agents).

        """
        self.agent_initial_opinions = np.array(self.rg[RNGStream.INITIAL_OPINIONS].uniform(
            MIN_OPINION, MAX_OPINION, self.N))

        self.raw_data[:, 0] = self.agent_initial_opinions

    def collect_opinion_data(self, t):
        """Copy latest opinions to data store."""
        self.raw_data[:, t] = self.latest_opinions

    def refresh_error_terms(self):
        """Regenerate array of error terms for agent influence models."""
        if self.error_distribution is not None:
            self.error_terms = self.rg[RNGStream.ERROR_DISTRIBUTION].__getattribute__(
                self.error_distribution[0])(*self.error_distribution[1])
        else:
            pass  # assumes "zero distribution" created during __init__

    def step(self):
        """Advance a single step of the simulation."""
        self.refresh_error_terms()

        self.schedule.step()

        self.collect_opinion_data(self.schedule.time)

    def run_model(self):
        """Run all steps for simulation run."""
        while self.running:
            self.step()

            if self.schedule.time >= self.max_steps:
                self.running = False

    def get_weights_matrix(self):
        """Return matrix of agent weights based on the agent.w_j vectors.

        Not currently needed for project, but may be convenient to have.

        """
        weights = np.zeros((self.N, self.N))
        for agent in self.agents:
            weights[agent.unique_id, agent.neighbor_ids] = agent.w_j
        return weights


class NonhomogeneousBaseModel(Model):
    """Base model class for nonhomogeneous scenarios. Slightly refactors Model class."""

    def __init__(self, iseed, max_steps, network_parameters,
                 agent_class, scheduler_class, error_distribution=None, network_digraph=None):
        """See Model class for documentation.

        This refactors __init__ from Model to allow extension by scenario subclasses with less code
        duplication, but it does mean this functionally duplicates the Model class.

        """
        # Prepare random number streams
        self.iseed = iseed
        # +1 because auto.enum starts at 1
        self.rg = [RandomGenerator(PCG64(self.iseed, i + 1)) for i in range(len(RNGStream) + 1)]

        self.max_steps = max_steps  # some agent classes need this, so save it first

        # Create network; see randomness for details on this RNG approach
        if network_digraph is not None and isinstance(network_digraph, nx.DiGraph):
            self.G = network_digraph
        else:
            network_rg = RandomGenRandomInterface(self.rg[RNGStream.GRAPH_GENERATOR])
            self.G = getattr(networks, network_parameters[0])(*network_parameters[1], network_rg)
            self.G = networks.prepare_graph_for_trial(self.G, network_rg)  # make directed, connected, & free of self-loops

        self.N = self.G.number_of_nodes()

        # Distribution and array for error terms. It is faster to have the model create a vector
        # of random variates for the full population each time step than to have each agent generate
        # them one at a time.
        if error_distribution is None:
            # No error terms for this model, so create a constant array of zeros
            self.error_distribution = None
            self.error_terms = np.zeros(self.N, dtype=np.int8)
        else:
            # Modify self.error_distribution to include population size self.N, for convenience
            self.error_distribution = (error_distribution[0], (*error_distribution[1:], self.N))

            # Prepare error terms if needed for agent initialization
            self.refresh_error_terms()

        # Prepare opinion data / data stores
        self.raw_data = np.zeros((self.N, self.max_steps + 1), dtype=np.float64)  # +1 due to initial opinions
        self.create_initial_opinions()  # agent.__init__ may expect this
        self.latest_opinions = np.copy(self.agent_initial_opinions)

        # Create agents
        self.create_agents(agent_class)

        # Create scheduler
        self.setup_scheduler(scheduler_class)

        self.running = True

        # end init()

    def create_agents(self, agent_class):
        self.agents = [agent_class(i, self) for i in range(self.N)]

    def setup_scheduler(self, scheduler_class):
        self.schedule = scheduler_class(self, self.rg[RNGStream.AGENT_ACTIVATION])
        self.schedule.agents = list(self.agents)


class InformedUninformedModel(NonhomogeneousBaseModel):
    """Model class for informed/uninformed agents scenario."""

    def __init__(self, iseed, max_steps, network_parameters,
                 agent_class, scheduler_class, error_distribution=None, fraction_uninformed=0.0):
        """Create a new model, configured based on design point inputs.

        Args:
            <see Model definition for other args>
            fraction_uninformed (float): Fraction of population assigned to the Uninformed class; all
                other agents default to the Informed class.

        """
        self.fraction_uninformed = fraction_uninformed

        super().__init__(iseed, max_steps, network_parameters, agent_class, scheduler_class, error_distribution)

    def create_agents(self, agent_class):
        from agents import InformedUninformedAgentFactory
        self.agents = [InformedUninformedAgentFactory(i, self, agent_class) for i in range(self.N)]

        # Create temporary RNG stream
        rg = RandomGenerator(PCG64(self.iseed, 50))  # 50 is arbitrary

        # Designate agent ids to use for Concord-type agents
        arr = np.arange(self.N)
        rg.shuffle(arr)
        num_uninformed = int(self.fraction_uninformed * self.N)
        uninformed_indexes = arr[0:num_uninformed]

        # Randomly set a fraction of agents to start as Uninformed
        for i in uninformed_indexes:
            self.agents[i].make_uninformed()


class ConcordPartialAntagonismModel(NonhomogeneousBaseModel):
    """Model class for Concord-type/Partial Antagonism-type scenario."""

    def __init__(self, iseed, max_steps, network_parameters,
                 agent_class, scheduler_class, error_distribution=None, fraction_concord=0.5, fraction_left=0.5):
        """Create a new model, configured based on design point inputs.

        Concord-type agents == similarity bias
        Partial Antagonism-type agents == attractive-repulsive

        Args:
            <see Model definition for other args>
            fraction_concord (float): Fraction of population assigned to the Concord class; all
                other agents default to the Partial Antagonism class.
            fraction_left (float): Fraction of population with left-oriented [-1, 0] bias; all other
                agents draw their initial opinion from right-oriented [0, 1].

        """
        # agent_class is ignored for this model
        agent_class = None

        self.fraction_concord = fraction_concord
        self.fraction_left = fraction_left

        # overrides for create_agents and create_initial_opinions apply the fraction_concord and
        # fraction_left, respectively, when called from inside super().__init__
        super().__init__(iseed, max_steps, network_parameters, agent_class, scheduler_class, error_distribution)

    def create_agents(self, agent_class):
        from agents import SimilarityBiasAgent, AttractiveRepulsiveAgent

        # Create temporary RNG stream
        rg = RandomGenerator(PCG64(self.iseed, 50))  # 50 is arbitrary

        # Designate agent ids to use for Concord-type agents
        arr = np.arange(self.N)
        rg.shuffle(arr)
        num_concord = int(self.fraction_concord * self.N)
        concord_indexes = arr[0:num_concord]

        # Create agents using designated type
        self.agents = [SimilarityBiasAgent(i, self) if i in concord_indexes else AttractiveRepulsiveAgent(i, self) for i in range(self.N)]

    def create_initial_opinions(self):
        """Create array of initial opinion values for agents to refer to.

        This applies the left-right bias for the scenario.

        """
        num_left = int(self.fraction_left * self.N)
        num_right = self.N - num_left
        self.agent_initial_opinions = np.concatenate((
            np.array(self.rg[RNGStream.INITIAL_OPINIONS].uniform(MIN_OPINION, 0, num_left)),
            np.array(self.rg[RNGStream.INITIAL_OPINIONS].uniform(0, MAX_OPINION, num_right))
            ))

        # randomize the sequence of these biased initial opinions
        rg = RandomGenerator(PCG64(self.iseed, 60))  # 60 is arbitrary
        rg.shuffle(self.agent_initial_opinions)

        self.raw_data[:, 0] = self.agent_initial_opinions


class BotsHumansModel(NonhomogeneousBaseModel):
    """Model class for bots/humans agents scenario."""

    # Constants based on Gilani2017
    fraction_bots = 0.4314
    bot_update_rate = 0.5185
    human_update_rate = 1 - bot_update_rate
    bot_bot_bias = 3
    human_human_bias = 2

    def __init__(self, iseed, max_steps, network_parameters,
                 agent_class, scheduler_class, error_distribution=None):
        """Create a new model, configured based on design point inputs.

        Only one base case of this scenario is used, so the factors are set as class constants
        instead of design factor arguments.

        """
        from activation_regimes import RandomWeighted
        scheduler_class = RandomWeighted  # override whatever was input, as we only use RandomWeighted

        super().__init__(iseed, max_steps, network_parameters, agent_class, scheduler_class, error_distribution)

        # for update frequency
        num_bots = int(self.fraction_bots * self.N)
        prob_bot = self.bot_update_rate / num_bots  # single-bot probabilty * num bots = prob. of any bot
        prob_human = self.human_update_rate / (self.N - num_bots)

        weights = [prob_bot if a.scenario_class == 'bot' else prob_human for a in self.agents]
        self.schedule.weights = weights

    def create_agents(self, agent_class):
        self.agents = [agent_class(i, self) for i in range(self.N)]

        # Create temporary RNG stream
        rg = RandomGenerator(PCG64(self.iseed, 50))  # 50 is arbitrary

        # designate agents to be bots
        arr = np.arange(self.N)
        rg.shuffle(arr)
        num_bots = int(self.fraction_bots * self.N)
        bot_indexes = arr[0:num_bots]

        # designate bot/human
        for agent in self.agents:
            if agent.unique_id in bot_indexes:
                agent.scenario_class = 'bot'
            else:
                agent.scenario_class = 'human'

        for agent in self.agents:
            # bias edge weights for same-class neighbors
            # multiply weights of same-class neighbors by the correct _bias constant, then re-normalize
            for i, j in enumerate(agent.neighbor_ids):
                if self.agents[j].scenario_class == agent.scenario_class:
                    agent.w_j[i] *= self.bot_bot_bias if agent.scenario_class == 'bot' else self.human_human_bias

            agent.w_j = agent.w_j / agent.w_j.sum()  # renormalize


class StubbornNormalModel(NonhomogeneousBaseModel):
    """Model class for stubborn/normal agents scenario."""

    def __init__(self, iseed, max_steps, network_parameters,
                 agent_class, scheduler_class, error_distribution=None, fraction_stubborn=0.0):
        """Create a new model, configured based on design point inputs.

        Args:
            <see Model definition for other args>
            fraction_stubborn (float): Fraction of population assigned to the Stubborn class; all
                other agents default to the Normal class.

        """
        # init agent using the normal class from the experimental design (all influence models are in play)
        super().__init__(iseed, max_steps, network_parameters, agent_class, scheduler_class, error_distribution)

        # Create temporary RNG stream
        rg = RandomGenerator(PCG64(self.iseed, 50))  # 50 is arbitrary

        # Randomly set a fraction of agents to Stubborn
        arr = np.arange(self.N)
        rg.shuffle(arr)
        num_stubborn = int(fraction_stubborn * self.N)
        stubborn_indexes = arr[0:num_stubborn]

        for i in stubborn_indexes:
            # convert a Normal agent into a Stubborn agent
            self.agents[i].step = self.agents[i].noop
            self.agents[i].advance = self.agents[i].noop


# Models for nonhomogeneous scenarios
scenario_model_map = {
    1: InformedUninformedModel,
    2: ConcordPartialAntagonismModel,
    3: BotsHumansModel,
    4: StubbornNormalModel
    }


def build_test_model():
    """Create a test model instance for development work."""
    # m = Model(iseed, max_steps, network_parameters, agent_class, scheduler_class, error_distribution=None)
    import agents
    import activation_regimes
    m = Model(
        1234056, 500, ('scale_free', (100, 5)), agents.StandardModelAgent,
        activation_regimes.Uniform, ('normal', 0, 0.05))
    return m


if __name__ == '__main__':
    m = build_test_model()
    m.run_model()
    # print(m.raw_data, m.raw_data.shape)
    # np.save('test_output123.data.npy', m.raw_data)
    # print(m.raw_data[:,-1])
    print('--complete--')
