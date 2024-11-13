"""
Module for the LPAgent class that can be subclassed by agents.
"""

import numpy as np
from conf import LABELS, GRAPH_TYPE, STATE_CHANGING_METHOD
from main_LPA import INDEX_DNA_COLUMN_NAME, USE_SHARING_INDEX


# An agent has its vector label with raw values and a state which depends on the vector label
# if L0 > L1; then the state is adapter
# if L0 <= L1; then the state is non adapter
class LPAgent:
    # Variables shared between all instances of this class
    TIMESTEP_DEFAULT = 1.0

    def __init__(self, initializer, name='network_process'):
        self.initialize(*initializer)

    def initialize(self, env, id, sim, LPNet):
        self.env = env
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
            yield self.env.timeout(LPAgent.TIMESTEP_DEFAULT / 2)
            self.state_changing()
            yield self.env.timeout(LPAgent.TIMESTEP_DEFAULT / 2)

    """
    Updates all the belonging coefficients
    """

    def state_changing(self):
        if GRAPH_TYPE == "U":
            neighbours = list(self.LPNet.neighbors(self.id))
        else:
            neighbours = list(self.LPNet.predecessors(self.id))

        # used for A0
        if STATE_CHANGING_METHOD == 0:
            # both perseverance and total plasticity have the same value -> 1 / (k+1)
            weight = float(1 / (len(neighbours) + 1))
            for label in LABELS:
                self_avg = self.LPNet.nodes[self.id][label] * weight

                neighbours_acc = 0
                for i in list(neighbours):
                    neighbours_acc += float(self.LPNet.nodes[i][label])

                self.VL[label] = self_avg + neighbours_acc * weight

        # used for A1 (NO bias, but different weights)
        if STATE_CHANGING_METHOD == 1:
            for label in LABELS:
                # the weight of the current agent opinion is the opinion perseverance, and it is
                # computed using a beta distribution with alpha = 2 and beta = 2
                vl = self.LPNet.nodes[self.id][label]
                # agent's perseverance is an attribute of each node, since it is constant in time
                # it has been initialised at network initialisation
                opinion_perseverance = 0.8 #self.LPNet.nodes[self.id]["perseverance"]
                self_avg = vl * opinion_perseverance

                # the total weight of the neighbours needs to obey the condition : w_i + \sum w_j = 1 -> \sum w_j = 1 - w_i
                total_opinion_plasticity = 1 - opinion_perseverance
                neighbours_acc = []
                neighbours_weights = []

                for i in list(neighbours):
                    # the weight of each neighbour is its similarity value
                    neighbours_weights.append(self.LPNet.get_edge_data(self.id, i)["weight"])
                    neighbours_acc.append(float(self.LPNet.nodes[i][label]))

                # the opinion plasticity is the normalised version of the weight computed with the bias
                # normalised to the maximum value which is 1 - perseverance
                neighbours_acc = np.array(neighbours_acc)
                normalised_weights = reweight(neighbours_weights, total_opinion_plasticity)

                # the aggregation function is
                # opinion perseverance * agent'sopinion + sum of opinion plasticity * neighbour's opinion
                self.VL[label] = self_avg + np.sum(neighbours_acc * 0.2) #normalised_weights)

        if STATE_CHANGING_METHOD == 2:
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
                                     use_sharing_index=USE_SHARING_INDEX)

    def aggregation_function(self, neighbours, privileged, discriminated, bias_attribute_label, bias_attributes):
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
            neighbours_acc = []
            neighbours_weights = []

            for i in list(neighbours):
                # the weight of each neighbour is its similarity value
                bias_condition = [bias_attribute == self.LPNet.nodes[i][bias_attribute_label] for bias_attribute in
                                  bias_attributes]
                # if at least one bias condition is satisfied then the weight is the privileged
                # ex: if Male == the current gender, I have a privilege; otherwise (Female) is discriminated.
                weight = privileged if (True in bias_condition) else discriminated
                neighbours_weights.append(weight)
                neighbours_acc.append(float(self.LPNet.nodes[i][label]))

            # the opinion plasticity is the normalised version of the weight computed with the bias
            # normalised to the maximum value which is 1 - perseverance
            neighbours_acc = np.array(neighbours_acc)
            normalised_weights = reweight(neighbours_weights, total_opinion_plasticity)

            # the aggregation function is
            # opinion perseverance * agent'sopinion + sum of opinion plasticity * neighbour's opinion
            self.VL[label] = self_avg + np.sum(neighbours_acc * normalised_weights)

    """
    Update the VL and the state used by other agents to update themselves
    """

    def update_step(self):
        for label in LABELS:
            self.LPNet.nodes[self.id][label] = self.VL[label]
        self.LPNet.nodes[self.id]["state"] = self.state


# the normalisation returns the weight value between 0 and "to".
# zero is the min value so the normalisation is just the division of x over to.
# otherwise, it would have been (x - min) / (to - min)
def reweight(x, to):
    x = np.array(x)
    return (x * to) / np.sum(x)


def determine_state(vl, index, labels, original_value, use_sharing_index):
    # 0; 1    # adapter     1; 0    # non adapter
    non_adapter_label = vl[labels[0]]
    adapter_label = vl[labels[1]]
    is_currently_adapter = original_value == +1
    # if non adapter
    if not is_currently_adapter:
        if non_adapter_label < adapter_label:
            if use_sharing_index:
                # if the agent has 0.8 as index -> 80% of times becomes adapter
                rand = np.random.rand()
                if index > rand:
                    return +1
    else:
        # if they are equal ([0.5, 0.5]) -> then it stays the same
        return original_value


def get_sharing_index(node):
    return node[INDEX_DNA_COLUMN_NAME]
