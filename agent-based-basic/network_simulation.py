import simpy as sim
from network_logging import NetworkLogger
from conf import TRIALS


class NetworkSimulation:

    """
    Simulation support for agents in a complex network.
    Can run multiple fresh trials with the same input parameters. Writes system
    state evolution to file (states & network topologies)
    """

    def __init__(self, LPNet, LPAgent, max_time):
        self.env = sim.Environment()
        self.LPNet = LPNet
        self.LPAgent = LPAgent
        self.until = max_time

    """
    Run a simulation for TRIALS trials
    """

    def run_simulation(self, run_index):
        print("Starting simulation...")
        for i in range(TRIALS):
            print("--- Trial %i ---" % i)
            self.run_trial(i, run_index)
        print("Simulation completed.")

    """
    Run a single trial
    """

    def run_trial(self, id, run_index):
        # the initilization is done simply by initializing the Environment
        # self.initialize()

        print("Set up LP agents...")

        # Initialize agents
        for i in self.LPNet.nodes():
            agent = self.LPAgent.LPAgent((self.env, i, self, self.LPNet))
            self.LPNet.nodes[i]['agent'] = agent
            self.env.process(agent.Run())


        print("Set up logging...")

        # Set up logging
        logging_interval = 1
        logger = NetworkLogger(self, logging_interval)
        self.env.process(logger.Run())

        # Run simulation
        self.env.run(self.until)

        # Write log files
        # saves in *.pickled the resulting nodes at each run until maxTime
        # only for specific iteration --> modify tt variable to add and/or remove trials if uncesseray or useless.
        logger.log_trial_to_files(id, run_index)
