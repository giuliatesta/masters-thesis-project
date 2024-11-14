import numpy as np
import simpy as sim
from network_logging import NetworkLogger
from conf import TRIALS, LABELS


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

        print("Let's start the agents")

        # Initialize agents
        for i in self.LPNet.nodes():
            agent = self.LPAgent.LPAgent((self.env, i, self, self.LPNet))
            self.LPNet.nodes[i]['agent'] = agent
            self.env.process(agent.Run())
        print("All agents run")

        print("Updating the states...")
        self.env.process(self.update_states())

        print("Let's start logging")

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

    def update_states(self):
        from main_LPA import USE_SHARING_INDEX
        for i in self.LPNet.nodes():
            node = self.LPNet.nodes[i]
            if i in [0, 12, 220]: print(f"old state: {node['state']}, {USE_SHARING_INDEX}")
            node["state"] = determine_state(i, node, use_sharing_index=USE_SHARING_INDEX)
            assert node["state"] == 1 or node["state"] == -1
            if i in [0, 12, 220]: print(f"new state: {node['state']}")
        yield self.env.timeout(1)


def determine_state(id, node, use_sharing_index):
    vl = get_vector_label(node)
    current_state = get_state(node)
    index = get_sharing_index(node)

    # 0; 1    # adapter     1; 0    # non adapter
    non_adapter_label = vl[0]
    adapter_label = vl[1]
    is_currently_adapter = (current_state == +1)
    # if non adapter
    if not is_currently_adapter:
        if non_adapter_label < adapter_label:
            if use_sharing_index:
                rand = np.random.rand()
                print(f"HERE: {index} > {rand} ? ")
                # if the agent has 0.8 as index -> 80% of times becomes adapter
                if index > rand:
                    print(f"YES for node {id}")
                    return +1

    # if they are equal ([0.5, 0.5]) -> then it stays the same
    return current_state


def get_sharing_index(node):
    return node["sha_ind_norm"]


def get_state(node):
    return node["state"]


def get_vector_label(node):
    vl = []
    for label in LABELS:
        vl.append(node[label])
    return vl
