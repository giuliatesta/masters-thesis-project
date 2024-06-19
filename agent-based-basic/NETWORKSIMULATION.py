import simpy as sim
import NETWORKLOGGER
from conf import TRIALS


class NetworkSimulation:
    """
    Simulation support for agents in a complex network.
    Can run multiple fresh trials with the same input parameters. Writes system
    state evolution to file (states & network topologies)
    """
    def __init__(self, LPNet, LPAgent, maxTime):
        self.env = sim.Environment()
        self.LPNet = LPNet
        self.LPAgent = LPAgent
        self.until = maxTime

    """
    Run a simulation for TRIALS trials
    """
    def runSimulation(self):
        print("Starting simulation...")
        for i in range(TRIALS):
            print("--- Trial %i ---" % i)
            self.runTrial(i)
        print("Simulation completed.")

    """
    Run a single trial
    """

    def runTrial(self, id):
        # the initilization is done simply by initializing the Environment
        # self.initialize()

        print("Set up LP agents...")

        # Initialize agents
        for i in self.LPNet.nodes():
            agent = self.LPAgent.LPAgent((i, self.env, self.LPNet))
            self.LPNet.nodes[i]['agent'] = agent
            self.env.process(agent.Run())

        print("Set up logging...")

        # Set up logging
        logging_interval = 1
        logger = NETWORKLOGGER.NetworkLogger(self, logging_interval)
        # TODO prior=True ???
        self.env.process(logger.Run())

        # Run simulation
        self.env.run(self.until)

        # Write log files
        logger.logTrialToFiles(id)
