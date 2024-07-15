"""
Module for the LPAgent class that can be subclassed by agents.
"""
from SimPy import Simulation as Sim
from conf import LABELS, GRAPH_TYPE, STATE_CHANGING_METHOD


class LPAgent(Sim.Process):
    # Variables shared between all instances of this class
    TIMESTEP_DEFAULT = 1.0

    def __init__(self, initializer, name='network_process'):
        Sim.Process.__init__(self, name)
        self.initialize(*initializer)

    def initialize(self, id, sim, LPNet):
        print(id)
        self.id = id
        self.sim = sim
        self.LPNet = LPNet
        self.VL = {label: LPNet.nodes[id][label] for label in LABELS}
        self.raw = {label: 0.0 for label in LABELS}
        print(self.VL)

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

        DNA_THRESHOLD = 0.5
        for label in LABELS:

            if STATE_CHANGING_METHOD == 0:
                neighbours_size = len(list(neighbours))
                for j in LABELS:
                    ssum = 0
                    # Calculate the second term of the update rule
                    for i in list(neighbours):
                        ssum += float(self.LPNet.nodes[i][j]) / float(neighbours_size + 1)
                        # ssum += float((self.LPNet.nodes[i][j])*(list(list(self.LPNet.edges(data=True))[(self.id+1)*(j-1)][2].values())[0])) / float(neighboursSize + 1)

                    avg = self.LPNet.nodes[self.id][j] / float(neighbours_size + 1) + ssum
                    self.VL[label] = avg
                    self.raw[label] = avg

            # if STATE_CHANGING_METHOD == 1:
            # if the index_DNA is bigger than a threshold then the agent becomes adapter
            # otherwise it becomes non adapter

            if STATE_CHANGING_METHOD == 2:
                # if the average of the index_DNA of the neighbours is higher than a threshold
                # then the agent becomes adapter; otherwise it's non adapter
                avg_DNA = self.compute_average_index_DNA(neighbours)
                self.VL[label] = 1 if avg_DNA >= DNA_THRESHOLD else 0
                self.raw[label] = avg_DNA

            if STATE_CHANGING_METHOD == 3:
                # finds the neighbour with the highest weight and check if its adapter or not
                highest_weight_neighbours = []
                max_weight = 0

                for neighbour in list(neighbours):
                    data = self.LPNet.get_edge_data(self.id, neighbour)
                    weight = data['weight']
                    if weight > max_weight:
                        max_weight = weight
                        highest_weight_neighbours = [neighbour]
                    elif weight == max_weight:
                        highest_weight_neighbours.append(neighbour)

                avg_DNA = self.compute_average_index_DNA(neighbours)
                self.VL[label] = 1 if avg_DNA >= DNA_THRESHOLD else 0
                self.raw[label] = avg_DNA

            if STATE_CHANGING_METHOD == 4:
                # compute the average using the weight and whether they are adapter (+1) or non adapter(-1).
                # if avg > 0 and the agent is non adopter changes its state to adopter;
                # if avg < 0 and the agent is adopter changes its state to non adopter;
                # and then if avg =0 the agent keeps its opinion.
                # then compute the aggregation function:  w1 * my opinion + w2 * neighbours’ opinion.
                sum_for_avg = 0
                neighbours_size = len(list(neighbours))
                for neighbour in list(neighbours):
                    data = self.LPNet.get_edge_data(self.id, neighbour)
                    weight = data.get('weight', 0)  # default value = 0 if it doesn't exist
                    sum_for_avg += weight * (1 if self.LPNet.nodes[neighbour][label] == 1 else -1)
                avg = float(sum_for_avg / neighbours_size)

                # AGGREGATION FUNCTION:
                # agent_weight = float(1 / (neighbours_size + 1))
                # neighbours_weight = agent_weight
                # self.raw[label] = agent_weight * self.VL[label] + neighbours_weight * avg
                self.raw[label] = avg
                if (avg > 0 and self.VL[label] == 0) or (avg < 0 and self.VL[label] == 0):
                    self.VL[label] = not self.VL[label]
            if STATE_CHANGING_METHOD == 5:
                # sort the neighbours by weight, consider only n of them
                # if 50% adapter 50% non adapter -> the agent keeps its opinion
                # otherwise, the agent changes its opinion (MAJORITY RULE)
                weights = []
                for i, neighbour in enumerate(neighbours):
                    data = self.LPNet.get_edge_data(self.id, neighbour)
                    weights.append(data.get('weight', 0))  # default value = 0 if it doesn't exist

                sorted_neighbours = [value for value, _ in
                                     sorted(zip(neighbours, weights), key=lambda x: x[1], reverse=True)]
                neighbours_vls = [self.LPNet.nodes[neighbour][label] for neighbour in sorted_neighbours]
                self.VL[label] = apply_majority(neighbours_vls, self.VL[label])
            print(f"state changing: {self.VL}, {self.raw}")

    """
    Update the VL used by other agents to update themselves
    """

    def update_step(self):
        for label in LABELS:
            # print(f"LPNET.nodes: {self.LPNet.nodes[self.id]}, VL: {self.VL[label]}")
            self.LPNet.nodes[self.id][label] = self.VL[label]

    def compute_average_index_DNA(self, neighbours):
        neighbours_size = len(list(neighbours))
        sum_for_average = 0
        for neighbour in list(neighbours):
            sum_for_average += float(self.LPNet.nodes[neighbour]["attribute-0"])
        return float(sum_for_average / neighbours_size)


def apply_majority(values, previous_value):
    total = len(values)
    adapters = values.count(1) / total
    non_adapters = values.count(0) / total
    return 1 if adapters > non_adapters else \
        (0 if non_adapters > adapters else previous_value)
