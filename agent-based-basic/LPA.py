"""
Module for the LPAgent class that can be subclassed by agents.
"""
import random

import numpy as np
from SimPy import Simulation as Sim
from conf import LABELS, GRAPH_TYPE, STATE_CHANGING_METHOD
from Main_LPA import INDEX_DNA_COLUMN_NAME


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

        # aggregation function with same weights, BUT with GENDER bias (TRUST_PERC% of times I don't trust the opposite gender)
        if STATE_CHANGING_METHOD == 1:
            TRUST_PERCENTAGE = 0.1  # percentage of how many to keep and trust with their opinions
            gender = self.LPNet.nodes[self.id]["Gender"]
            # extracts the neighbours with different gender
            different_gender_neighbours = list(
                filter(lambda node: gender != self.LPNet.nodes[node]["Gender"], neighbours))
            # computes how many of them needs to be removed based on the TRUST_PERCENTAGE
            to_be_removed_count = int(len(different_gender_neighbours) * (1 - TRUST_PERCENTAGE))
            nodes_to_be_removed = np.random.choice(different_gender_neighbours, to_be_removed_count, replace=False)
            print(f"Keeping only the {TRUST_PERCENTAGE * 100}% of opposite-gender neighbours ({len(nodes_to_be_removed)} has been removed from {len(neighbours)} ({len(different_gender_neighbours)}))")
            print(nodes_to_be_removed)
            # remove the ones extracted with random choice
            for i in nodes_to_be_removed:
                neighbours.remove(i)
            neighbours_size = len(neighbours)
            for label in LABELS:
                neighbours_avg = 0
                for i in list(neighbours):
                    neighbours_avg += float(self.LPNet.nodes[i][label]) / float(neighbours_size + 1)
                    # ssum += float((self.LPNet.nodes[i][j])*(list(list(self.LPNet.edges(data=True))[(self.id+1)*(j-1)][2].values())[0])) / float(neighboursSize + 1)

                self_avg = self.LPNet.nodes[self.id][label] / float(neighbours_size + 1)
                # aggregation function with same weight for both self' and neighbours' opinion average -> 1 / (neighbours +1)
                self.VL[label] = self_avg + neighbours_avg

        # once the vector label is changed, given the neighbours opinion, the agent's state changes
        self.state = determine_state(self.VL, get_index(self.LPNet.nodes[self.id]), LABELS, original_value=self.state)

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
