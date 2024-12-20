import numpy as np
import simpy as sim

from network_logging import NetworkLogger
from conf import TRIALS, LABELS, GRAPH_TYPE, RESULTS_DIR
from network_plotter import NetworkPlotter
import utils

# it can run multiple fresh trials with the same input parameters.
# writes system state evolution to file (states & network topologies)
class NetworkSimulation:

    def __init__(self, LPNet, LPAgent, max_time, results_dir):
        self.env = sim.Environment()
        self.LPNet = LPNet
        self.LPAgent = LPAgent
        self.until = max_time
        self.results_dir = results_dir

    # runs a simulation for TRIALS trials
    def run_simulation(self, run_index, cognitive_bias="no-bias", social_bias="no-bias"):
        print("Starting simulation...")
        for i in range(TRIALS):
            print("--- Trial %i ---" % i)
            self.run_trial(i, run_index, cognitive_bias, social_bias)
        print("Simulation completed.")

    # runs a single trial
    def run_trial(self, trial_id, run_index, cognitive_bias, social_bias):
        # process that counts the number of adapters at the beginning of an iteration
        self.env.process(self.count_adapters())
        print("Let's start the agents")
        # a process for each node is initialised and activated
        for i in self.LPNet.nodes():
            agent = self.LPAgent.LPAgent(self.env, i, self, self.LPNet, social_bias)
            self.LPNet.nodes[i]['agent'] = agent
            self.env.process(agent.Run())

        # the node's states are updated at the end of the interaction
        # (after all agents have interacted with one another)
        self.env.process(self.update_states(cognitive_bias))

        #plotter = NetworkPlotter(self.env, self.LPNet, RESULTS_DIR)
        #self.env.process(plotter.Run())

        logging_interval = 1
        logger = NetworkLogger(self, logging_interval, self.results_dir)
        self.env.process(logger.Run())

        # Run simulation
        self.env.run(self.until)

        adapters_indices = sorted([node for node in self.LPNet.nodes() if
                                   self.LPNet.nodes()[node]["state"] == 1])

        print(f"Final adapters:\n{adapters_indices}")
        # Write log files
        # saves in *.pickled the resulting nodes at each run until maxTime
        # only for specific iteration --> modify tt variable to add and/or remove trials if unnecessary or useless.
        logger.log_trial_to_files(trial_id, run_index)

        #plotter.create_animation()

    def update_states(self, cognitive_bias):
        # for each node its state is updated based on the freshly re-calculated vector labels
        # an assertion guarantees that each node has a valid state
        while True:
            print("Updating the states...")
            for i in self.LPNet.nodes():
                node = self.LPNet.nodes[i]
                # if i in [0, 12, 220]: print(f"old state: {node['state']}, {USE_SHARING_INDEX}")
                node["state"] = determine_state(node, self.LPNet, cognitive_bias)
                assert node["state"] == 1 or node["state"] == -1
                print(f"node {i}) new state: {self.LPNet.nodes[i]['state']}")
            yield self.env.timeout(1.5)

    # counts the number of adapters at each iteration
    def count_adapters(self):
        while True:
            adapters, non_adapters = utils.get_adapters_count(self.LPNet)
            print(f"At the beginning of step {self.env.now}: # Adapters: {adapters}, # Non-Adapters: {non_adapters}")
            yield self.env.timeout(1)


# determine the new state of a node based on its vector label
# the state can change, if the node is not already an adopter
# and for the confirmation bias scenarios (B* simulations) the sharing index is greater than a random number
def determine_state(node, graph, cognitive_bias):
    vl = get_vector_label(node)
    current_state = get_state(node)
    index = get_sharing_index(node)

    node_id = node["agent"].id
    neighbours = list(graph.neighbors(node_id)) if GRAPH_TYPE == "U" else list(graph.predecessors(node_id))
    adapters = 0
    for i in neighbours:
        if graph.nodes()[i]["L1"] == 1:
            adapters += 1
    are_adapters_majority = (adapters >= len(neighbours) * 0.5)

    # 0; 1    # adapter     1; 0    # non adapter
    non_adapter_label = vl[0]
    adapter_label = vl[1]
    is_currently_non_adapter = (current_state == -1)

    def try_become_adopter(sharing_index, current_state):
        if sharing_index > np.random.rand():
            print(f"{node_id}: Success! It's a Girl")
            return +1
        else: return current_state

    # if non adapter
    if is_currently_non_adapter:
        print("currently non adopter")
        # and adapter label is bigger than non adapter label
        if adapter_label > non_adapter_label:
            print("could try to become adopter")
            if cognitive_bias == "confirmation-bias":
                return try_become_adopter(index, current_state)
            if cognitive_bias == "availability-bias":
                print(f"{node_id}: can apply AB ({are_adapters_majority})")
                if are_adapters_majority:
                    return +1
            if cognitive_bias == "confirmation-availability-bias":
                if are_adapters_majority:
                    print("HERE")
                    return +1
                else:
                    return try_become_adopter(index, current_state)
            elif cognitive_bias == "no-bias":
                #print(f"{node_id}: else")
                return +1
            else:
                print(f"{node_id}: else")
               # print(f"{node_id}: else ")
                return current_state
   # print(f"{node_id}: Keep its state")
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
