"""
Module for the LPAgent class that can be subclassed by agents.
"""
from conf import LABELS, GRAPH_TYPE


class LPAgent:
    # Variables shared between all instances of this class
    TIMESTEP_DEFAULT = 1.0

    def __init__(self, env, initializer, name='network_process'):
        self.env = env
        # TODO
        # self.process = sim.Process(env, name)
        self.initialize(*initializer)

    def initialize(self, id, LPNet):
        self.id = id
        self.LPNet = LPNet
        self.VL = {l: LPNet.nodes[id][l] for l in LABELS}

    """
    Start the agent execution
    it executes a state change then wait for the next step
    """

    def Run(self):
        while True:
            self.updateStep()
            yield self.env.timeout(), self, LPAgent.TIMESTEP_DEFAULT / 2
            self.state_changing()
            yield self.env.timeout(), self, LPAgent.TIMESTEP_DEFAULT / 2

    """
    Updates all the belonging coefficients
    """

    def state_changing(self):
        if GRAPH_TYPE == "U":
            neighbours = list(self.LPNet.neighbors(self.id))
        else:
            neighbours = list(self.LPNet.predecessors(self.id))
        neighboursSize = len(list(neighbours))

        for j in LABELS:
            ssum = 0
            # Calculate the second term of the update rule
            for i in list(neighbours):
                ssum += float(self.LPNet.nodes[i][j]) / float(neighboursSize + 1)
                # ssum += float((self.LPNet.nodes[i][j])*(list(list(self.LPNet.edges(data=True))[(self.id+1)*(j-1)][2].values())[0])) / float(neighboursSize + 1)

            # Update str(j)'s belonging coefficient
            self.VL[j] = self.LPNet.nodes[self.id][j] / float(neighboursSize + 1) + ssum

    """
    Update the VL used by other agents to update themselves
    """

    def updateStep(self):
        for j in LABELS:
            self.LPNet.nodes[self.id][j] = self.VL[j]
