import numpy as np
import pandas as pd
from networkx.algorithms.components import connected_components, is_weakly_connected, weakly_connected_components, \
        is_strongly_connected, strongly_connected_components, is_connected
from numpy.f2py.auxfuncs import replace

from create_input import create_input_files
from preprocessing import load_dataset_csv
from create_network import plot_network
from initial_network_plots import nodes_by_degree_distribution
import utils
from after_simluation_plots import states_changing_heat_map

# vector_labels = utils.read_pickled_file(
#     "/Users/giuliatesta/PycharmProjects/masters-thesis-project/agent-based-basic/work/results/simulation_03_BASE_LINE/trial_0_LPStates_L0_L1_0_RUN_0.pickled")
# states = utils.read_pickled_file("/Users/giuliatesta/PycharmProjects/masters-thesis-project/agent-based-basic/work/results/simulation_03_BASE_LINE/trial_0_LPStates_0_RUN_0_STATES.pickled")
# states_changing_heat_map(
#     states=states,
#     vector_labels=vector_labels,
#     step= 0,
#     title="State changing Heat Map (SIM 3 - BASE LINE)",
#     path= "/Users/giuliatesta/PycharmProjects/masters-thesis-project/agent-based-basic/work/results/simulation_03_BASE_LINE")


# for threshold in np.arange(0.2, 0.8, 0.05):
#     data = load_dataset_csv("../dataset/df_DNA_sharingEU.csv", index=False)
#     graph = create_input_files(data, [
#         "sha_ind_norm",
#         "Gender",
#         "Education",
#         "Income_level",
#         "Age",
#         "Would_subscribe_car_sharing_if_available"],
#         similarity_threshold=threshold)
#     string_threshold = f"{threshold:.2f}".replace(".", "")
#     nodes_by_degree_distribution(graph, title=f"Degree Distribution of Nodes (similarity threshold = {threshold})", file_name=f"nodes_by_degree_{string_threshold}")

data = load_dataset_csv("../dataset/df_DNA_sharingEU.csv", index=False)
graph = create_input_files(data, [
        "sha_ind_norm",
        "Gender",
        "Education",
        "Income_level",
        "Age",
        "Would_subscribe_car_sharing_if_available"],
        similarity_threshold=0.60)

f = open("/Users/giuliatesta/PycharmProjects/masters-thesis-project/dataset/components.txt", "w")
for c in connected_components(graph):
        f.write(f"{len(c)}\t{c}\n")
f.close()
print("-------------------")
print(f"Is the network connected? {is_connected(graph)}")

plot_network(graph, "/Users/giuliatesta/PycharmProjects/masters-thesis-project/agent-based-basic/network_analysis/network/network_060.png")