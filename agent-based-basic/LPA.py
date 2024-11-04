"""
Module for the LPAgent class that can be subclassed by agents.
"""

import numpy as np
from SimPy import Simulation as Sim
from conf import LABELS, GRAPH_TYPE, STATE_CHANGING_METHOD
from main_LPA import INDEX_DNA_COLUMN_NAME


# An agent has its vector label with raw values and a state which depends on the vector label
# if L0 > L1; then the state is adapter
# if L0 <= L1; then the state is non adapter
class LPAgent(Sim.Process):
    # Variables shared between all instances of this class
    TIMESTEP_DEFAULT = 1.0

    def __init__(self, initializer, name='network_process'):
        Sim.Process.__init__(self, name)
        self.initialize(*initializer)

    def initialize(self, id, sim, LPNet):
        self.id = id
        self.sim = sim
        self.LPNet = LPNet
        self.VL = {label: LPNet.nodes[id][label] for label in LABELS}  # [0.0, 1.0] for example
        self.state = 1 if self.VL[LABELS[0]] == 0. and self.VL[LABELS[1]] == 1 else -1

    """
    Start the agent execution
    it executes a state change then wait for the next step
    """

    def Run(self):
        while True:
            self.update_step()
            yield Sim.hold, self, LPAgent.TIMESTEP_DEFAULT / 2
            self.state_changing()
            yield Sim.hold, self, LPAgent.TIMESTEP_DEFAULT / 2

    """
    Updates all the belonging coefficients
    """

    def state_changing(self):
        if GRAPH_TYPE == "U":
            neighbours = list(self.LPNet.neighbors(self.id))
        else:
            neighbours = list(self.LPNet.predecessors(self.id))

        # aggregation function with same weights for neighbours and personal opinion
        # NO BIAS
        # W_i = W_j = 1 / (k+1) with k = number of neighbours
        if STATE_CHANGING_METHOD == 0:
            for label in LABELS:
                # the weight of the current agent opinion is the opinion perseverance, and it is
                # computed using a beta distribution with alpha = 2 and beta = 2
                vl = self.LPNet.nodes[self.id][label]
                # agent's perseverance is an attribute of each node, since it is constant in time
                # it has been initialised at network initialisation
                opinion_perseverance = self.LPNet.nodes[self.id]["perseverance"]
                self_avg = vl * opinion_perseverance

                # the total weight of the neighbours needs to obey the condition : w_i + \sum w_j = 1 -> \sum w_j = 1 - w_i
                total_opinion_plasticity = 1 - opinion_perseverance
                neighbours_avg = 0

                for i in list(neighbours):
                    # the weight of each neighbour is its similarity value
                    weight = self.LPNet.get_edge_data(self.id, i)["weight"]
                    # the opinion plasiticity is the normalised version of the weight computed with the bias
                    # normalised to the maximum value which is 1 - perseverance
                    opinion_plasticity = weight * total_opinion_plasticity
                    neighbours_avg += float(self.LPNet.nodes[i][label]) * opinion_plasticity

                # the aggregation function is
                # opinion perseverance * agent'sopinion + sum of opinion plasticity * neighbour's opinion
                self.VL[label] = self_avg + neighbours_avg

        if STATE_CHANGING_METHOD == 1:
            privileged = 0.9
            discriminated = 0.1
            self.aggregation_function(
                neighbours,
                privileged=privileged,
                discriminated=discriminated,
                bias_attribute_label="Gender",
                bias_attributes=[self.LPNet.nodes[self.id]["Gender"]])
            bias_factor = privileged if self.LPNet.nodes[self.id]["Gender"] == "Male" else discriminated
        # once the vector label is changed, given the neighbours opinion, the agent's state changes
        self.state = determine_state(self.VL,
                                     get_sharing_index(self.LPNet.nodes[self.id]),
                                     LABELS,
                                     original_value=self.state,
                                     use_sharing_index=False)

    def aggregation_function(self, neighbours, privileged, discriminated, bias_attribute_label, bias_attributes):
        for label in LABELS:
            # the weight of the current agent opinion is the opinion perseverance, and it is
            # computed using a beta distribution with alpha = 2 and beta = 2
            vl = self.LPNet.nodes[self.id][label]
            opinion_perseverance = self.LPNet.nodes[self.id]["perseverance"]
            self_avg = vl * opinion_perseverance

            # the total weight of the neighbours needs to obey the condition : w_i + \sum w_j = 1 -> \sum w_j = 1 - w_i
            total_opinion_plasticity = 1 - opinion_perseverance
            neighbours_avg = 0
            for i in list(neighbours):
                bias_condition = [bias_attribute == self.LPNet.nodes[i][bias_attribute_label] for bias_attribute in
                                  bias_attributes]
                # if at least one bias condition is satisfied then the weight is the privileged
                # ex: if Male == the current gender, I have a privilege; otherwise (Female) is discriminated.
                weight = privileged if (True in bias_condition) else discriminated
                # the opinion plasiticity is the normalised version of the weight computed with the bias
                # normalised to the maximum value which is 1 - perseverance
                opinion_plasticity = normalise(weight, to=total_opinion_plasticity)
                neighbours_avg += float(self.LPNet.nodes[i][label]) * opinion_plasticity

            # the aggregation function is
            # opinion perseverance * agent'sopinion + sum of opinion plasticity * neighbour's opinion
            self.VL[label] = self_avg + neighbours_avg

    """
    Update the VL and the state used by other agents to update themselves
    """

    def update_step(self):
        for label in LABELS:
            self.LPNet.nodes[self.id][label] = self.VL[label]
        self.LPNet.nodes[self.id]["state"] = self.state

    def compute_average_index_DNA(self, neighbours):
        neighbours_size = len(list(neighbours))
        sum_for_average = 0
        for neighbour in list(neighbours):
            sum_for_average += float(self.LPNet.nodes[neighbour]["attribute-0"])
        return float(sum_for_average / neighbours_size)


# the normalisation returns the weight value between 0 and "to".
# zero is the min value so the normalisation is just the division of x over to.
# otherwise, it would have been (x - min) / (to - min)
def normalise(x, to):
    return float(x * to)


def determine_state(vl, index, labels, original_value, use_sharing_index = True):
    # 0; 1    # adapter     1; 0    # non adapter
    non_adapter_label = vl[labels[0]]
    adapter_label = vl[labels[1]]
    # if non adapter
    if non_adapter_label > adapter_label:
        if use_sharing_index:
            # if the agent has 0.8 as index -> 80% of times becomes adapter
            if index < np.random.rand():
                return + 1
        else:
            return -1
    if non_adapter_label < adapter_label:
        # if the adapter's value is greater than non-adapter's value -> it becomes adapter
        return 1
    else:
        # if they are equal ([0.5, 0.5]) -> then it stays the same
        return original_value



def get_sharing_index(node):
    return node[INDEX_DNA_COLUMN_NAME]
