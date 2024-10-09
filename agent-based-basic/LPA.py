"""
Module for the LPAgent class that can be subclassed by agents.
"""
import random

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
        if STATE_CHANGING_METHOD == 0:
            neighbours_size = len(list(neighbours))
            for label in LABELS:
                neighbours_avg = 0
                for i in list(neighbours):
                    neighbours_avg += float(self.LPNet.nodes[i][label]) / float(neighbours_size + 1)
                    # ssum += float((self.LPNet.nodes[i][j])*(list(list(self.LPNet.edges(data=True))[(self.id+1)*(j-1)][2].values())[0])) / float(neighboursSize + 1)

                self_avg = self.LPNet.nodes[self.id][label] / float(neighbours_size + 1)
                self.VL[
                    label] = self_avg + neighbours_avg  # aggregation function with same weight for both self' and neighbours' opinion average

        # aggregation function with same weights, BUT only using as neighbours agent
        # 1. with same gender as the current agent
        # 2. in the same age range as the current agent
        if STATE_CHANGING_METHOD == 1:
            GENDER_CONDITION = lambda node: self.LPNet.nodes[self.id]["Gender"] == self.LPNet.nodes[node]["Gender"]
            AGE_CONDITION = lambda node: self.LPNet.nodes[self.id]["Age"] == self.LPNet.nodes[node]["Age"]
            filtered_neighbours = list(filter(AGE_CONDITION, neighbours))
            neighbours_size = len(list(filtered_neighbours))
            for label in LABELS:
                neighbours_avg = 0
                for i in list(filtered_neighbours):
                    neighbours_avg += float(self.LPNet.nodes[i][label]) / float(neighbours_size + 1)

                self_avg = self.LPNet.nodes[self.id][label] / float(neighbours_size + 1)
                # aggregation function with same weight for both self' and neighbours' opinion average -> 1 / (neighbours +1)
                self.VL[label] = self_avg + neighbours_avg

        if STATE_CHANGING_METHOD == 2:
            SAME_GROUP_WEIGHT = 0.5  # percentage of how many to keep and trust with their opinions
            DIFFERENT_GROUP_WEIGHT = 1 - SAME_GROUP_WEIGHT
            BIAS_ATTRIBUTE = "Age"
            bias_attribute = self.LPNet.nodes[self.id][BIAS_ATTRIBUTE]
            self.aggregation_function(
                neighbours,
                same_weight=SAME_GROUP_WEIGHT,
                different_weight=DIFFERENT_GROUP_WEIGHT,
                bias_attribute_label="Age",
                bias_attribute=bias_attribute)
        if STATE_CHANGING_METHOD == 3:
            FEMALE_WEIGHT = 0.1  # female neghbours' opinion weight
            MALE_WEIGHT = 0.9  # male neighbours' opinion weight

            self.aggregation_function(
                neighbours,
                same_weight=FEMALE_WEIGHT,
                different_weight=MALE_WEIGHT,
                bias_attribute_label="Age",
                bias_attribute="ATTENZIONEEEE")
        # once the vector label is changed, given the neighbours opinion, the agent's state changes
        self.state = determine_state(self.VL, get_index(self.LPNet.nodes[self.id]), LABELS, original_value=self.state)

    def aggregation_function(self, neighbours, same_weight, different_weight, bias_attribute_label, bias_attribute):
        neighbours_size = len(neighbours)
        for label in LABELS:
            neighbours_avg = 0
            for i in list(neighbours):
                weight = same_weight if "2" == self.LPNet.nodes[i][bias_attribute_label] or "4" == self.LPNet.nodes[i][bias_attribute_label] else different_weight
                neighbours_avg += (float(self.LPNet.nodes[i][label]) * weight) / float(neighbours_size + 1)

            self_avg = self.LPNet.nodes[self.id][label] / float(neighbours_size + 1)
            # aggregation function with same weight for both self' and neighbours' opinion average -> 1 / (# neighbours +1)
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


def determine_state(vl, index, labels, original_value):
    # 0; 1    # adapter     1; 0    # non adapter
    non_adapter_label = vl[labels[0]]
    adapter_label = vl[labels[1]]
    # if non adapter
    if non_adapter_label > adapter_label:
        # if the agent has 0.8 as index -> 80% of times becomes adapter
        if index < np.random.rand():
            return + 1
    if non_adapter_label < adapter_label:
        # if the adapter's value is greater than non-adapter's value -> it becomes adapter
        return 1
    else:
        # if they are equal ([0.5, 0.5]) -> then it stays the same
        return original_value


def get_index(node):
    return node[INDEX_DNA_COLUMN_NAME]
