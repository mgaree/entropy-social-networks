# -*- coding: utf-8 -*-
"""Agent classes used by Mesa model for entropy experiment."""

import numpy as np
from numpy import dot, abs, array, exp, arange  # binding to a local variable as a micro-optimization

from constants import MIN_OPINION, MAX_OPINION, SIMILARITY_BIAS_MU, SIMILARITY_BIAS_EPSILON, ATTRACTIVE_REPULSIVE_MU
from randomness import RNGStream


class BaseAgent:
    """Base Agent class for subclassing.

    Agent's opinion value (o_i) is some function of stochastic noise terms and
    the neighbors' influence values.

    The step-advance approach to updating opinion values is to support the
    Synchronous activation regime: step() records the current state and advance()
    updates the agent's opinion using the recorded state.

    Args:
        unique_id (int): Unique id number associated with instance.
        model (obj:Model): Reference to parent Model.

    Attributes:
        unique_id (int): Unique id number associated with this agent.
        model (obj:Model): Reference to parent Model.
        error (function): Member of parent Model used to generate error terms.
        o_i (float): Current opinion value.
        o_i1 (float): Initial opinion value.
        w_j (np.array of floats): Edge/neighbor weights.
        neighbor_ids (list of ints): Neighbor ids of self, based on self.model.G.
        o_j (np.array of floats): Most recently cached values of neighbors
          opinions.

    """

    def __init__(self, unique_id, model):
        """Create a new agent."""
        self.unique_id = unique_id
        self.model = model

        # Sorting is not neccessary for the model, but it helps ensure consistency between runs.
        # A Consistently Ordered graph is available in NetworkX but it has caveats. However, the fact
        # it exists suggests the default graphs may be inconsistently ordered, so sorting here is wise.
        #
        # By casting it as an np.array here, we speed up step() by 13s (10K run);
        # otherwise, same performance as old-style, querying neighbor objects each step.
        self.neighbor_ids = array(sorted(self.model.G.neighbors(self.unique_id)), dtype=np.int64)

        # Set initial opinion
        self.o_i1 = self.model.agent_initial_opinions[self.unique_id]
        self.o_i = self.o_i1

        self._o_i_next = 0  # temp variable to support Synchronous activation regime

        # Create edge/neighbor weights using Model's RNG stream, then
        # normalize so they sum to 1
        self.w_j = self.model.rg[RNGStream.NEIGHBOR_WEIGHTS].uniform(
            0, 1, self.model.G.out_degree(self.unique_id))
        self.w_j = self.w_j / self.w_j.sum()

        # If an agent has no neighbors, it won't activate, so set its standard activation
        # routines to a 'noop'; see below.
        if self.neighbor_ids.size == 0:
            self.step = self.noop
            self.advance = self.noop
            self.finalize = self.noop  # ok to skip this, since model.latest_opinions is already full

    def noop(self):
        """Empty function used for step/advance/finalize for agents with no neighbors.

        During __init__(), if an agent has no neighbors, self.neighbor_ids.size == 0, then it will
        never need to update. Instead of subjecting the agent to try-except or if-else during the
        normal update steps, we overwrite the update steps to be this noop function. This cleans
        up the code for the regular functions at the cost of a single if check during __init__().

        """
        pass

    def step(self):
        """Update opinion value privately, gathering neighbor opinions as needed.

        This stores the value the new opinion privately before any agent publicly updates their
        opinions. This is essential for Synchronous-style activation regimes (but is not so useful
        for sequential ones).

        Each subclass is responsible for implementing this.

        """
        raise NotImplementedError

    def advance(self):
        """Make the agent's new opinion public, ensure it is in-bounds, and publish to the model."""
        self.o_i = self._o_i_next  # make private new opinion public

        # this approach is faster than numpy.clip, ternary if-else, and max(.., min(..))
        if self.o_i > MAX_OPINION:
            self.o_i = MAX_OPINION
        elif self.o_i < MIN_OPINION:
            self.o_i = MIN_OPINION

        self.model.latest_opinions[self.unique_id] = self.o_i


class DeGrootAgent(BaseAgent):
    """Used for validation.

    With the DeGroot influence model, agents will converge to stable opinions if the strongly
    connected component they occupy is aperiodic, and they will reach a consensus if there is a
    single such giant component. For example:

        G = nx.to_directed(nx.barabasi_albert_graph(100, 1))
        nx.is_aperiodic(G)  # ==> False, so will not converge

        G = nx.to_directed(nx.barabasi_albert_graph(100, 3))
        nx.is_aperiodic(G)  # ==> True, so will converge
        nx.number_strongly_connected_components(G)  # ==> 1, so will reach a consensus

    will, even with any of the three activation regimes (Synchronous, Uniform, or Random).
    Create each of these graphs and run a model created with `network_digraph=G`, then plot the
    `model.raw_data` to observe convergence or lack thereof.

    This helps to validate the code for the Model and activation regimes, and it gives a common
    baseline for comparing other agent influence models.

    """

    def step(self):
        """Compute new opinion using DeGroot Model."""
        # Get neighbors' current opinions.
        #
        # This code is duplicated across several agent classes. Originally, agent activation involved
        # a step (get neighbor data), advance (update own opinion), and finalize (clip to bounds
        # and publish to model). That approach let us declare step() in the base class and customize
        # advance() for each agent. While making for better code, it added an extra function call
        # per agent per time step; changing to the current approach reduced trial runtime ~7%. Due
        # to the size of the planned experiment, this is good benefit for the cost of slighly
        # messier code.
        self.o_j = self.model.latest_opinions[self.neighbor_ids]

        # Privately update opinion using influence model on neighbors' current opinions.
        self._o_i_next = dot(self.w_j, self.o_j) + self.model.error_terms[self.unique_id]


class StandardModelAgent(BaseAgent):
    """Interacts with agents via Friedkin and Johnsen's standard model.

    o_i(t+1) = a_{ii} sum_j w_{ij} o_j(t) + (1-a_{ii}) o_i(1)

    Attributes (excluding inherited items):
        a_ii (float): agent's susceptibility to external influence, a_ii in [0, 1]

    """

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

        self.a_ii = self.model.rg[RNGStream.SUSCEPTIBILITY_INFLUENCE].uniform(0, 1)
        self.biased_initial_opinion = (1 - self.a_ii) * self.o_i1  # this is constant and used often

    def step(self):
        """Compute new opinion using Standard Model."""
        self.o_j = self.model.latest_opinions[self.neighbor_ids]

        # using w_j instead of w_ij since the agent only has its own weight vector
        self._o_i_next = self.a_ii * dot(self.w_j, self.o_j) + self.biased_initial_opinion \
            + self.model.error_terms[self.unique_id]


class NonlinearAgent(BaseAgent):
    """Interacts with agents via non-linear equation.

    Influence model suggested by Chan (2017, private correspondence):

    o_i = frac{b_{i0}}{1 + exp{({w_{i1} o_{j1} + w_{i2} o_{j2} + ...})}} + varepsilon_i

    Attributes (excluding inherited items):
        b_i0 (float): agent's internal bias

    """

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

        self.b_i0 = self.model.rg[RNGStream.INTERNAL_BIAS].uniform(0, 1)

    def step(self):
        """Compute new opinion using nonlinear model."""
        self.o_j = self.model.latest_opinions[self.neighbor_ids]

        self._o_i_next = self.b_i0 / (1 + exp(dot(self.w_j, self.o_j))) + self.model.error_terms[self.unique_id]


class SingleNeighborInteractionBaseAgent(BaseAgent):
    """Interacts with one neighbor per update, selected at weighted random. Base class for others."""

    def __init__(self, unique_id, model):
        """Override base class method to pre-cache some numbers.

        Since the neighbors are fixed during a run, we can pre-compute the order of neighbors
        that we randomly interact with each time step. The selection probabilities are weighted
        by self.w_j.

        Pre-computing the visit order is ~60s faster per run than calling choice() every advance() (!!)

        """
        super().__init__(unique_id, model)

        if self.neighbor_ids.size == 0:
            return  # noop assignment happened in BaseAgent.__init__ already, so step/advance won't occur

        indices = arange(len(self.neighbor_ids))
        self.neighbor_indexes_over_time_for_adoption = self.model.rg[RNGStream.SINGLE_NEIGHBOR_SELECTION].choice(
            indices, self.model.max_steps + 1, p=self.w_j, replace=True)


class SimilarityBiasAgent(SingleNeighborInteractionBaseAgent):
    """Interacts with one neighboor, selected at random, via similarity bias model in Flache2017.

    o_i(t+1) = o_i(t) + f_w(o_i(t), o_j(t)) (o_j(t) - o_i(t)),

    f_w(o_i, o_j) = mu, if |o_i - o_j| leq epsilon and 0 otherwise

    Constant values of mu and epsilon are used, based on Deffuant2000, Fig. 2, rescaled for our
    larger opinion interval.

    """

    def step(self):
        """Compute new opinion using Similarity Bias model."""
        # This approach (prebuilding a selection sequence and selecting from that neighbor now)
        # saves 12s per run vs getting all neighbor opinions and selecting one.
        # Due to 'noop' assignment, we do not need to handle no-neighbor exception states.
        neighbor_index_to_select = self.neighbor_indexes_over_time_for_adoption[self.model.schedule.time]
        self.selected_neighbor_opinion = self.model.latest_opinions[self.neighbor_ids[neighbor_index_to_select]]

        # Due to 'noop' assignment, we do not need to handle no-neighbor exception states
        o_diff = self.selected_neighbor_opinion - self.o_i

        if abs(o_diff) < SIMILARITY_BIAS_EPSILON:
            self._o_i_next = self.o_i + SIMILARITY_BIAS_MU * (o_diff) + self.model.error_terms[self.unique_id]
        else:
            # the act of assessing the neighbor's opinion can induce noise
            # -- noise is not in the cited models, but I've added it to all models in this project
            self._o_i_next = self.o_i + self.model.error_terms[self.unique_id]


class AttractiveRepulsiveAgent(SingleNeighborInteractionBaseAgent):
    """Interacts with neighbors via attractive-repulsive model in Flache2017.

    There, the model is

        o_i(t+1) = o_i(t) + f_w(o_i(t), o_j(t)) (o_j(t) - o_i(t))
    and
        f_w(o_i, o_j) = mu (1 - 2 |o_j(t) - o_i(t)|),

    but this assumes o_i in [0, 1]. Since we are considering o_i in [-1, 1], we adjust the
    weight function to be

        f_w(o_i, o_j) = mu (1 - 1 |o_j(t) - o_i(t)|).

    """

    def step(self):
        """Compute new opinion using Attractive-Repulsive model."""
        neighbor_index_to_select = self.neighbor_indexes_over_time_for_adoption[self.model.schedule.time]
        self.selected_neighbor_opinion = self.model.latest_opinions[self.neighbor_ids[neighbor_index_to_select]]

        o_diff = self.selected_neighbor_opinion - self.o_i

        self._o_i_next = self.o_i + ATTRACTIVE_REPULSIVE_MU * (1 - abs(o_diff)) * o_diff \
            + self.model.error_terms[self.unique_id]


class RandomAdoptionAgent(SingleNeighborInteractionBaseAgent):
    """Interacts by adopting the opinion of one neighbor selected at random."""

    def step(self):
        """Compute new opinion using Random Adoption model."""
        neighbor_index_to_select = self.neighbor_indexes_over_time_for_adoption[self.model.schedule.time]
        self.selected_neighbor_opinion = self.model.latest_opinions[self.neighbor_ids[neighbor_index_to_select]]

        self._o_i_next = self.selected_neighbor_opinion + self.model.error_terms[self.unique_id]


# Agents for nonhomogeneous experiment

def InformedUninformedAgentFactory(unique_id, model, agent_class):
    class InformedUninformedAgent(agent_class):
        """If Informed, interacts as a normal member of any other agent class. If Uninformed, does nothing until becoming Informed.

        Notes:
        - Informed agent can ask an Uninformed agent for their opinion, in which case nothing happens (re: SingleNeighborInteractionBaseAgent)
        - Uninformed agents have zero opinion, but they will affect classical averaging models by still contributing to the denominator of the mean

        """

        def __init__(self, unique_id, model):
            """Initialize agent as a descended of the given agent_class."""
            super().__init__(unique_id, model)

            self.is_informed = True

        def make_uninformed(self):
            """Convert informed agent to uninformed status."""
            self.is_informed = False
            self.o_i1 = self.o_i = self._o_i_next = 0

            # since Informed->Uninformed only happens at t=0:
            self.model.agent_initial_opinions[self.unique_id] = self.model.raw_data[self.unique_id, 0] = self.model.latest_opinions[self.unique_id] = 0
            # note: it is not ideal to let the agent directly edit the model, but I'm going to allow it here for my own convenience

        def make_informed(self):
            """Convert uninformed agent to informed status."""
            self.is_informed = True

            # adopt the opinion of an arbitrary Informed neighbor
            for j in self.neighbor_ids:
                if self.model.agents[j].is_informed:
                    self.o_i = self.model.latest_opinions[j]
                    break

            # set any attributes normally init'ed that are functions of having an opinion (a_ii, etc.)
            try:
                self.biased_initial_opinion = (1 - self.a_ii) * self.o_i
            except AttributeError:  # self.a_ii doesn't exist, so don't need biased_initial_opinion
                pass

        def has_informed_neighbors(self):
            """Return True if agent has at least one Informed neighbor."""
            for j in self.neighbor_ids:
                if self.model.agents[j].is_informed:
                    return True

            return False

        def step(self):
            if self.is_informed:
                # if SingleNeighborInteractionBaseAgent subclass, check if selected neighbor is Uninformed
                try:
                    neighbor_index_to_select = self.neighbor_indexes_over_time_for_adoption[self.model.schedule.time]
                except AttributeError:
                    # no neighbor_indexes_over_time_for_adoption attribute, so not a SingleNeighborInteractionBaseAgent descendent;
                    # unconditionally make the step() since classical models still work
                    super().step()
                else:
                    # only make the single-neighbor interaction step if it will be with an Informed neighbor
                    if self.model.agents[neighbor_index_to_select].is_informed:
                        super().step()

            else:
                # check if should become Informed
                if self.has_informed_neighbors():
                    self.make_informed()
                    self.step()  # re-run step() as a newly-Informed agent
    #end class def

    return InformedUninformedAgent(unique_id, model)
