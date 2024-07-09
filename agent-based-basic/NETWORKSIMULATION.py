from SimPy import Simulation as Sim
import NETWORKLOGGER
from conf import TRIALS


class NetworkSimulation(Sim.Simulation):

    """
    Simulation support for agents in a complex network.
    Can run multiple fresh trials with the same input parameters. Writes system
    state evolution to file (states & network topologies)
    """

    def __init__(self, LPNet, LPAgent, max_time):
        Sim.Simulation.__init__(self)
        self.LPNet = LPNet
        self.LPAgent = LPAgent
        self.until = max_time

    """
    Run a simulation for TRIALS trials
    """

    def run_simulation(self):
        print("Starting simulation...")
        for i in range(TRIALS):
            print("--- Trial %i ---" % i)
            self.run_trial(i)
        print("Simulation completed.")

    """
    Run a single trial
    """

    def run_trial(self, id):
        self.initialize()

        print("Set up LP agents...")

        # Initialize agents
        for i in self.LPNet.nodes():
            agent = self.LPAgent.LPAgent((i, self, self.LPNet))
            self.LPNet.nodes[i]['agent'] = agent
            self.activate(agent, agent.Run())

        print("Set up logging...")

        # Set up logging
        logging_interval = 1
        logger = NETWORKLOGGER.NetworkLogger(self, logging_interval)
        self.activate(logger, logger.Run(), prior=True)

        # Run simulation
        self.simulate(self.until)

        # Write log files
        # saves in *.pickled the resulting nodes at each run until maxTime
        # only for specific iteration --> modify tt variable to add and/or remove trials if uncesseray or useless.
        logger.log_trial_to_files(id)
