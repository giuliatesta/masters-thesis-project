import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from utils import read_pickled_file

# results coming from simulation_03_BASE_LINE which has 20% of initial adapters and no biases.
BASELINE = {1: 100, 2: 513, 3: 719, 4: 828, 5: 887, 6: 924, 7: 948, 8: 963, 9: 973, 10: 980, 11: 984, 13: 991, 14: 993,
            15: 994, 16: 995, 17: 996, 18: 997, 19: 997, 20: 998, 21: 998, 22: 998, 23: 998, 24: 998, 25: 999, 26: 999,
            27: 999, 28: 999, 29: 999, 30: 999, 31: 999}


# plots the evolution of the number of adapters as the time steps proceeds in the simulation
# shows the differences with the BASELINE and indicates the slope of the curve
def draw_adapter_by_time_plot(adapters, results_file_path, title):
    plt.figure(figsize=(10, 6))
    plt.plot(list(adapters.keys()), list(adapters.values()), marker='x', label="Current simulation")
    last_value = list(adapters.values())[-1]
    first_value = list(adapters.values())[0]
    for key, value in adapters.items():
        if value == last_value:
            # plt.plot(key, value, marker='o')
            plt.axvline(x=key, linestyle='--', color=('cyan', 0.5), label='Convergence threshold')
            break
    slope = (last_value - first_value) / len(adapters)
    plt.text(28, 280, f"slope: {slope:.2f}", fontsize=10,
             bbox=dict(facecolor='none', alpha=0.2))
    # for key, value in adapters.items():
    #     print(f"({key}, {value}")
    #     plt.text(key, value, f"({key:.2f}, {value})", fontsize=6, ha="center")
    plt.plot(list(BASELINE.keys()), list(BASELINE.values()), 'g--', linewidth=1, label='Base Line')
    plt.xlabel('Time steps')
    plt.ylabel('Adapters')
    plt.title('Number of adapters by time' if title == "" else title)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(f"{results_file_path}/avg_adapters_by_time_plot.png", dpi=1000)


# plots multiple simulations on the same plot
# simulations indicates the evolution of the number of adapters in the networks
def plot_multiple_adapters_by_time():
    adapters90 = {1: 100, 2: 516, 3: 720, 4: 828, 5: 889, 6: 928, 7: 950, 8: 964, 9: 974, 10: 981, 11: 985, 13: 991,
                  14: 993, 15: 994, 16: 995, 17: 996, 18: 997, 19: 997, 20: 998, 21: 998, 22: 998, 23: 999, 24: 999,
                  25: 999, 26: 999, 27: 999, 28: 999, 29: 999, 30: 999, 31: 1000}
    adapters70 = {1: 100, 2: 514, 3: 717, 4: 828, 5: 891, 6: 927, 7: 951, 8: 964, 9: 974, 10: 981, 11: 985, 13: 992,
                  14: 993, 15: 995, 16: 995, 17: 996, 18: 997, 19: 997, 20: 998, 21: 998, 22: 998, 23: 998, 24: 999,
                  25: 999, 26: 999, 27: 999, 28: 999, 29: 999, 30: 999, 31: 999}
    adapters50 = {1: 100, 2: 515, 3: 720, 4: 829, 5: 891, 6: 928, 7: 951, 8: 965, 9: 975, 10: 981, 11: 985, 13: 991,
                  14: 993, 15: 994, 16: 995, 17: 996, 18: 997, 19: 997, 20: 998, 21: 998, 22: 998, 23: 999, 24: 999,
                  25: 999, 26: 999, 27: 999, 28: 999, 29: 999, 30: 999, 31: 999}
    plt.figure(figsize=(10, 6))
    plt.plot(list(adapters90.keys()), list(adapters90.values()), marker='x', label="Pro same gender by 90%")
    plt.plot(list(adapters70.keys()), list(adapters70.values()), marker='o', label="Pro same gender by 70%")
    plt.plot(list(adapters50.keys()), list(adapters50.values()), marker='.', label="Pro same gender by 50%")
    last_value = list(adapters90.values())[-1]
    first_value = list(adapters90.values())[0]
    slope90 = (last_value - first_value) / len(adapters90)
    last_value = list(adapters70.values())[-1]
    first_value = list(adapters70.values())[0]
    slope70 = (last_value - first_value) / len(adapters70)
    last_value = list(adapters50.values())[-1]
    first_value = list(adapters50.values())[0]
    slope50 = (last_value - first_value) / len(adapters50)
    plt.text(25, 300, f"slope 90%: {slope90:.2f}\nslope 70%: {slope70:.2f}\nslope 50%: {slope50:.2f}", fontsize=10,
             bbox=dict(facecolor='none', alpha=0.2))

    # for key, value in adapters.items():
    #     print(f"({key}, {value}")
    #     plt.text(key, value, f"({key:.2f}, {value})", fontsize=6, ha="center")
    plt.plot(list(BASELINE.keys()), list(BASELINE.values()), 'm--', linewidth=1, label='Base Line')
    plt.xlabel('Time steps')
    plt.ylabel('Adapters')
    plt.title('Gender bias simulations: pro same gender (90/70/50)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(
        f"/Users/giuliatesta/PycharmProjects/masters-thesis-project/agent-based-basic/work/results/gender_bias_adapters_by_time_plot.png",
        dpi=1000)


# plots the heat map representing the vector labels.txt changing in a specific time step during a simulation
# TODO complete (wrong)
import matplotlib.colors as mcolors

def states_changing_heat_map(states, vector_labels, step, title, path):
    # Extract the vector labels.txt and states for the given step
    vector_data = vector_labels[step][1]

    # Example Data
    # Assuming you have a list of VL0, VL1, and the resulting states
    VL0 = np.array([vl[0] for vl in vector_data])
    VL1 = np.array([vl[1] for vl in vector_data])

    # Determine the final state: +1 if VL1 > VL0, -1 otherwise
    final_state = states[step][1]
    print(final_state)

    # Calculate the number of rows and columns for a square-like grid
    n_nodes = len(VL0)
    n_rows = int(np.sqrt(n_nodes))
    n_cols = int(np.ceil(n_nodes / n_rows))

    # Reshape the data into a 2D grid, padding with NaN if necessary
    grid = np.full((n_rows * n_cols), np.nan)
    grid[:n_nodes] = final_state
    grid = grid.reshape(n_rows, n_cols)


    # Create the plot
    plt.figure(figsize=(12, 8))
    im = plt.imshow(grid, cmap="plasma", interpolation='nearest', vmin=-1, vmax=1)
    cbar = plt.colorbar(im, label='Adoption State')
    cbar.set_ticks([-1, 1])
    cbar.set_ticklabels(['Non-adopter (-1)', 'Adopter (+1)'])
    plt.title(f'Opinion Formation Simulation Heatmap ({n_nodes} nodes)')
    plt.xlabel('Column')
    plt.ylabel('Row')

    # Add text annotations
    adopters = np.sum(final_state == 1)
    non_adopters = np.sum(final_state == -1)
    plt.text(0.02, 0.98, f'Adopters: {adopters}', transform=plt.gca().transAxes, verticalalignment='top')
    plt.text(0.02, 0.94, f'Non-adopters: {non_adopters}', transform=plt.gca().transAxes, verticalalignment='top')

    plt.show()
    plt.savefig(path)
