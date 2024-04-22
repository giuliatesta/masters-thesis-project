import simpy as sp
from networkx import Graph

from network_building import plot_network


# TODO
def entity_process(env, graph):
    yield env.timeout(1)
    return ''


def simulate(graph: Graph):
    env = sp.Environment()
    process = env.process(entity_process(env, graph))
    env.run(until=process)

    plot_network(graph)
