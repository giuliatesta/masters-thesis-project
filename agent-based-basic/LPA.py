"""
Module for the LPAgent class that can be subclassed by agents.
"""

import numpy as np
from conf import LABELS, GRAPH_TYPE
from main_LPA import STATE_CHANGING_METHOD
from initial_network_plots import format_double


# An agent has its vector label with raw values and a state which depends on the vector label
# if L0 > L1; then the state is adapter
# if L0 <= L1; then the state is non adapter
class LPAgent:
    # Variables shared between all instances of this class
    TIMESTEP_DEFAULT = 1.0

    def __init__(self, env, node_id, sim, LPNet):
        self.env = env
        self.id = node_id
        self.sim = sim
        self.LPNet = LPNet
        self.VL = {label: LPNet.nodes[node_id][label] for label in LABELS}  # [0.0, 1.0] for example
        self.state = 1 if self.VL[LABELS[0]] == 0. and self.VL[LABELS[1]] == 1 else -1

    # update rule of the agents
    # re-evaluation of the vector labels based on the vls of neighbours
    def Run(self):
        while True:
            self.state_changing()
            yield self.env.timeout(LPAgent.TIMESTEP_DEFAULT/2)
            self.update_step()
            yield self.env.timeout(LPAgent.TIMESTEP_DEFAULT/2)

    def state_changing(self):
        neighbours = self.get_neighbours()
        rule = STATE_CHANGING_METHOD

        for label in LABELS:
            current_vl = self.LPNet.nodes[self.id][label]
            self_avg = 0
            neighbours_avg = 0
            if rule == "same-weights":
                OP = 1 / (len(neighbours) + 1)
                OL = OP
                self_avg = current_vl * OP
                neighbours_avg = 0
                for i in list(neighbours):
                    neighbours_avg += float(self.LPNet.nodes[i][label]) * OL

            if rule == "beta-dist":
                OP = self.LPNet.nodes[self.id]["perseverance"]
                OL = 1 - OP
                self_avg = current_vl * OP
                neighbours_acc = []
                neighbours_weights = []
                for i in list(neighbours):
                    # the weight of each neighbour is its similarity value
                    neighbours_weights.append(self.LPNet.get_edge_data(self.id, i)["weight"])
                    neighbours_acc.append(float(self.LPNet.nodes[i][label]))

                # the opinion plasticity is the normalised version of the weight computed with the bias
                # normalised to the maximum value which is 1 - perseverance
                reweighed = reweight(neighbours_weights, OL)
                neighbours_avg = sum([neighbours_acc[i] * reweighed[i] for i in range(len(neighbours))])
                print(f"node {self.id}) OP: {OP: .2f}, self avg: {self_avg: .2f}, neigh avg: {neighbours_avg: .2f}")
                nc = [format_double(i) for i in neighbours_acc]
                nw = [format_double(i) for i in reweighed]
                print(f"{nc}\n{nw} -> {sum(reweighed): .2f}")
            if rule == "over-confidence" or rule == "over-influenced":
                OP = 0.8 if rule == "over-confidence" else 0.2
                OL = 1 - OP

                self_avg = current_vl * OP
                neighbours_acc = []
                neighbours_weights = []
                for i in list(neighbours):
                    neighbours_weights.append(self.LPNet.get_edge_data(self.id, i)["weight"])
                    neighbours_acc.append(float(self.LPNet.nodes[i][label]))
                neighbours_count = len(neighbours)
                neighbours_avg = sum([neighbours_acc[i] * (OL / neighbours_count) for i in range(neighbours_count)])

            # TODO divide into different biased case scenarios
            # now, it is just to keep what it has been done
            if rule == "social-bias":
                OP = self.LPNet.nodes[self.id]["perseverance"]
                OL = 1 - OP
                self_avg = current_vl * OP
                neighbours_acc = []
                neighbours_weights = []
                for i in list(neighbours):
                    # the weight depends on some social bias: trusting more Males than Females
                    neighbours_weights.append(0.8 if self.LPNet.nodes[self.id]["Gender"] == "Male" else 0.2)
                    neighbours_acc.append(float(self.LPNet.nodes[i][label]))

                reweighed = reweight(neighbours_weights, OL)
                neighbours_avg = sum([neighbours_acc[i] * reweighed[i] for i in range(len(neighbours))])

            print(f"node {self.id} for {label}: {self.VL[label]: .2f} -> {self_avg + neighbours_avg: .2f}")
            self.VL[label] = self_avg + neighbours_avg

    """
    Update the VL and the state used by other agents to update themselves
    """

    def update_step(self):
        for label in LABELS:
            self.LPNet.nodes[self.id][label] = self.VL[label]

    def get_neighbours(self):
        return list(self.LPNet.neighbors(self.id)) if GRAPH_TYPE == "U" else list(self.LPNet.predecessors(self.id))


# the normalisation returns the weight value between 0 and upper_limit.
# scaled all the weights in x to the max value (upper_limit)
# the sum of all weights in x must be equal to upper_limit
def reweight(x, upper_limit):
    x = np.array(x)
    return (x * upper_limit) / np.sum(x)
