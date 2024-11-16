import numpy as np
import simpy as sim
from network_logging import NetworkLogger
from conf import TRIALS, LABELS


# it can run multiple fresh trials with the same input parameters.
# writes system state evolution to file (states & network topologies)
class NetworkSimulation:

    def __init__(self, LPNet, LPAgent, max_time):
        self.env = sim.Environment()
        self.LPNet = LPNet
        self.LPAgent = LPAgent
        self.until = max_time

    # runs a simulation for TRIALS trials
    def run_simulation(self, run_index):
        print("Starting simulation...")
        for i in range(TRIALS):
            print("--- Trial %i ---" % i)
            self.run_trial(i, run_index)
        print("Simulation completed.")

    # runs a single trial
    def run_trial(self, trial_id, run_index):
        # process that counts the number of adapters at the beginning of an iteration
        self.env.process(self.count_adapters())

        print("Let's start the agents")
        # a process for each node is initialised and activated
        for i in self.LPNet.nodes():
            agent = self.LPAgent.LPAgent(self.env, i, self, self.LPNet)
            self.LPNet.nodes[i]['agent'] = agent
            self.env.process(agent.Run())

        # the node's states are updated at the end of the interaction
        # (after all agents have interacted with one another)
        self.env.process(self.update_states())

        logging_interval = 1
        logger = NetworkLogger(self, logging_interval)
        self.env.process(logger.Run())

        # Run simulation
        self.env.run(self.until)

        # Write log files
        # saves in *.pickled the resulting nodes at each run until maxTime
        # only for specific iteration --> modify tt variable to add and/or remove trials if unnecessary or useless.
        logger.log_trial_to_files(trial_id, run_index)

    def update_states(self):
        from main_LPA import USE_SHARING_INDEX
        # for each node its state is updated based on the freshly re-calculated vector labels
        # an assertion guarantees that each node has a valid state
        while True:
            print("Updating the states...")
            for i in self.LPNet.nodes():
                node = self.LPNet.nodes[i]
                # if i in [0, 12, 220]: print(f"old state: {node['state']}, {USE_SHARING_INDEX}")
                node["state"] = determine_state(node, use_sharing_index=USE_SHARING_INDEX)
                assert node["state"] == 1 or node["state"] == -1
                print(f"node {i}) new state: {self.LPNet.nodes[i]['state']}")
            yield self.env.timeout(1.5)

    # counts the number of adapters at each iteration
    def count_adapters(self):
        while True:
            adapters = 0
            non_adapters = 0
            for i in self.LPNet.nodes():
                node = self.LPNet.nodes[i]
                if node["state"] == 1:
                    adapters += 1
                else:
                    non_adapters += 1
            print(f"At the beginning of step {self.env.now}: # Adapters: {adapters}, # Non-Adapters: {non_adapters}")
            yield self.env.timeout(1)


# determine the new state of a node based on its vector label
# the state can change, if the node is not already an adopter
# and for the confirmation bias scenarios (B* simulations) the sharing index is greater than a random number

# TODO: what happens when use_sharing_index is False, they cannot update their state.
#  Check it with Prof
def determine_state(node, use_sharing_index):
    vl = get_vector_label(node)
    current_state = get_state(node)
    index = get_sharing_index(node)

    # 0; 1    # adapter     1; 0    # non adapter
    non_adapter_label = vl[0]
    adapter_label = vl[1]
    is_currently_non_adapter = (current_state == -1)
    # if non adapter
    if is_currently_non_adapter:
        # and adapter label is bigger than non adapter label
        if adapter_label > non_adapter_label:
            print(f"{node} could become adopter")
            if use_sharing_index:
                rand = np.random.rand()
                print("is currectly non adopter, L0 < L1, uses shar.ind and index > rand?")
                # if the agent has 0.8 as index -> 80% of times becomes adapter
                if index > rand:
                    print("yes")
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
