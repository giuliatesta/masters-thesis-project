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
        self.VL["raw"] = 0
        print(self.VL)

    """
    Start the agent execution
    it executes a state change then wait for the next step
    """

    def Run(self):
        while True:
            self.updateStep()
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
                if avg_DNA >= DNA_THRESHOLD:
                    self.VL[label] = 1
                else:
                    self.VL[label] = 0
                self.VL["raw"] = avg_DNA
                print(f"state changing: {self.VL}")

            # ssum = 0
            # # Calculate the second term of the update rule
            # for neighbour in list(neighbours):
            #     ssum += float(self.LPNet.nodes[neighbour][label]) / float(neighbours_size + 1)
            #     # ssum += float((self.LPNet.nodes[i][j])*(list(list(self.LPNet.edges(data=True))[(self.id+1)*(j-1)][2].values())[0])) / float(neighboursSize + 1)
            #
            # # Update str(j)'s belonging coefficient
            # self.VL[label] = self.LPNet.nodes[self.id][label] / float(neighbours_size + 1) + ssum

    """
    Update the VL used by other agents to update themselves
    """

    def updateStep(self):
        for label in LABELS:
            #print(f"LPNET.nodes: {self.LPNet.nodes[self.id]}, VL: {self.VL[label]}")
            self.LPNet.nodes[self.id][label] = self.VL[label]
