# Step 2: Define SimPy processes
import simpy as sp
from networkx import Graph

from network_building import plot_network


def entity_process(env, graph):
    while True:
        yield env.timeout(1)


def simulate(graph: Graph):
    env = sp.Environment()
    env.process(entity_process(env, graph))
    env.run(until=10)

    plot_network(graph)
