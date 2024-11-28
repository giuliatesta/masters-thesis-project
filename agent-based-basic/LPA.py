"""
Module for the LPAgent class that can be subclassed by agents.
"""

import numpy as np
from conf import LABELS, GRAPH_TYPE
from initial_network_plots import format_double


# An agent has its vector label with raw values and a state which depends on the vector label
# if L0 > L1; then the state is adapter
# if L0 <= L1; then the state is non adapter
class LPAgent:
    # Variables shared between all instances of this class
    TIMESTEP_DEFAULT = 1.0

    def __init__(self, env, node_id, sim, LPNet, social_bias):
        self.env = env
        self.id = node_id
        self.sim = sim
        self.LPNet = LPNet
        self.social_bias = social_bias
        self.VL = {label: LPNet.nodes[node_id][label] for label in LABELS}  # [0.0, 1.0] for example
        self.state = 1 if self.VL[LABELS[0]] == 0. and self.VL[LABELS[1]] == 1 else -1

    # update rule of the agents
    # re-evaluation of the vector labels based on the vls of neighbours
    def Run(self):
        while True:
            self.evaluate_vector_labels()
            yield self.env.timeout(LPAgent.TIMESTEP_DEFAULT / 2)
            self.update_step()
            yield self.env.timeout(LPAgent.TIMESTEP_DEFAULT / 2)

    def evaluate_vector_labels(self):
        from main_LPA import VL_UPDATE_METHOD
        neighbours = get_neighbours(self.LPNet, self.id, self.social_bias)
        rule = VL_UPDATE_METHOD

        non_adapter_label = "L0"
        adapter_label = "L1"
        if rule == "simple-contagion":
            for i in list(neighbours):
                if self.LPNet.nodes[i][non_adapter_label] == 0.0 and self.LPNet.nodes[i][adapter_label] == 1.0:
                    self.VL[non_adapter_label] = 0.0
                    self.VL[adapter_label] = 1.0
                    return 0
        elif rule == "majority":
            adapters_count = 0
            for i in list(neighbours):
                if self.LPNet.nodes[i][non_adapter_label] == 0.0 and self.LPNet.nodes[i][adapter_label] == 1.0:
                    adapters_count += 1
            if adapters_count >= len(neighbours) * 0.5:
                self.VL[non_adapter_label] = 0.0
                self.VL[adapter_label] = 1.0
                return 0
        else:
            OP = 0
            if rule == "over-confidence":
                OP = 0.8
            if rule == "over-influenced":
                OP = 0.2
            if rule == "extreme-influenced":
                OP = 0.02
            if rule == "extreme-confidence":
                OP = 0.98
            if rule == "beta-dist":
                OP = self.LPNet.nodes[self.id]["perseverance"]
            OL = 1 - OP
            self_avg = self.LPNet.nodes[self.id][adapter_label] * OP
            neighbours_acc = []
            neighbours_weights = []
            adapter_count = 0
            for i in list(neighbours):
                weight = 0
                # if is an adapter, then its opinion is considered with weight 1
                if self.LPNet.nodes[i][adapter_label] == 1.0:
                    weight = 1
                    adapter_count += 1
                neighbours_weights.append(weight)
                neighbours_acc.append(float(self.LPNet.nodes[i][adapter_label]))
            # need to reweight to satisfy the balance between plasticity and perseverance
            reweighed = np.array(neighbours_weights) * OL / adapter_count
            neighbours_avg = sum([neighbours_acc[i] * reweighed[i] for i in range(len(neighbours))])
            nc = [format_double(i) for i in neighbours_acc]
            nw = [format_double(i) for i in reweighed]
            #print(f"node {self.id}) OP: {OP: .2f}, OL: {OL: .2f}, self avg: {self_avg}, neigh avg: {neighbours_avg}, "
            #     f"current state: {self.LPNet.nodes()[self.id]['state']}\ncurrent VLS: {self.VL}, adapters: {adapter_count}")
            # print(f"nc: {nc}\nnw:{nw}")
            self.VL[adapter_label] = self_avg + neighbours_avg
            self.VL[non_adapter_label] = 1 - self.VL[adapter_label]
            #print(f"node {self.id} new VLS: {self.VL}\n")

    """
    Update the VL and the state used by other agents to update themselves
    """

    def update_step(self):
        for label in LABELS:
            self.LPNet.nodes[self.id][label] = self.VL[label]


def get_neighbours(graph, node, social_bias):
    neighbours = list(graph.neighbors(node)) if GRAPH_TYPE == "U" else list(graph.predecessors(node))

    agent_gender = graph.nodes()[node]["Gender"]
    agent_age = graph.nodes()[node]["Age"]
    agent_education_level = graph.nodes()[node]["Education"]

    if social_bias == "no-bias":
        return neighbours
    if social_bias == "against-opposite-gender":
        # filters outs all agents that have the opposite gender to the current LPAgent
        return [n for n in neighbours if graph.nodes()[n]["Gender"] != agent_gender]
    if social_bias == "against-women":
        # for males, it filters out all females il the neighbourhood
        if agent_gender == "Male":
            return [n for n in neighbours if graph.nodes()[n]["Gender"] == "Male"]
    if social_bias == "against-young":
        # for all agents with more than 30 years old, it filters outs all younger agents in the neighbourhood
        if not has_less_than_30_years_old(agent_age):
            # keeps only older agents
            return [n for n in neighbours if not has_less_than_30_years_old(graph.nodes()[n]["Age"])]
    if social_bias == "against-old":
        # for all younger agents, it filters out the older agents in the neighbourhood
        if has_less_than_30_years_old(agent_age):
            # keeps only younger agents
            return [n for n in neighbours if has_less_than_30_years_old(graph.nodes()[n]["Age"])]
    if social_bias == "against-low-educated":
        # 2: Lower secondary (upper elementary school or similar),
        # 4: Primary (elementary school or similar)
        # for all agents with higher education, it filters outs all agent with lower education in the neighbourhood
        if agent_education_level != 2 or agent_education_level != 4:
            # keeps only higher-educated agents
            # 1: tertiary and higher (uni, phds); 3: upper secondary (high school)
            return [n for n in neighbours if graph.nodes()[n]["Education"] == 1 or graph.nodes()[n]["Education"] == 3]
    return neighbours


# the normalisation returns the weight value between 0 and upper_limit.
# scaled all the weights in x to the max value (upper_limit)
# the sum of all weights in x must be equal to upper_limit
def reweight(x, upper_limit):
    x = np.array(x)
    return (x * upper_limit) / np.sum(x)


def has_less_than_30_years_old(age):
    return age == 2 or age == 4
