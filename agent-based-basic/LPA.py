"""
Module for the LPAgent class that can be subclassed by agents.
"""
from SimPy import Simulation as Sim
from conf import LABELS, GRAPH_TYPE, STATE_CHANGING_METHOD


class LPAgent(Sim.Process):
    # Variables shared between all instances of this class
    TIMESTEP_DEFAULT = 1.0

    def __init__(self, initialiser, name='network_process'):
        Sim.Process.__init__(self, name)
        self.initialize(*initialiser)

    # id = node's number
    # VL = {"attribute-name":"attribute-value", ...}
    # LPNet = Graph with 698 nodes and 2396 edges
    # sim
    def initialize(self, id, sim, LPNet):
        print(id)
        self.id = id
        self.sim = sim
        self.LPNet = LPNet
        self.VL = {l: LPNet.nodes[id][l] for l in LABELS}
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
        neighbours_size = len(list(neighbours))

        for label in LABELS:
            DNA_THRESHOLD = 0.5
            # if STATE_CHANGING_METHOD == 1:
            # if the index_DNA is bigger than a threshold then the agent becomes adapter
            # otherwise it becomes non adapter
            if STATE_CHANGING_METHOD == 2:
                # if the average of the index_DNA of the neighbours is higher than a threshold
                # then the agent becomes adapter; otherwise it's non adapter
                sum = 0
                for neighbour in list(neighbours):
                    sum += float(self.LPNet.nodes[neighbour]["attribute-0"])
                avg_DNA = float(sum / neighbours_size)
                self.VL[label] = 1 if avg_DNA >= DNA_THRESHOLD else 0
                self.raw[label] = avg_DNA
                print(f"state changing: {self.VL}, {self.raw}")

    """
    Update the VL used by other agents to update themselves
    """

    def update_step(self):
        for label in LABELS:
            #print(f"LPNET.nodes: {self.LPNet.nodes[self.id]}, VL: {self.VL[label]}")
            self.LPNet.nodes[self.id][label] = self.VL[label]
