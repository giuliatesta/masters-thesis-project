import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from utils import read_pickled_file

# results coming from simulation_03_BASE_LINE which has 20% of initial adapters and no biases.
BASELINE = {1: 100, 2: 513, 3: 719, 4: 828, 5: 887, 6: 924, 7: 948, 8: 963, 9: 973, 10: 980, 11: 984, 13: 991, 14: 993, 15: 994, 16: 995, 17: 996, 18: 997, 19: 997, 20: 998, 21: 998, 22: 998, 23: 998, 24: 998, 25: 999, 26: 999, 27: 999, 28: 999, 29: 999, 30: 999, 31: 999}


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
    slope = (last_value-first_value) / len(adapters)
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
    adapters90= {1: 100, 2: 516, 3: 720, 4: 828, 5: 889, 6: 928, 7: 950, 8: 964, 9: 974, 10: 981, 11: 985, 13: 991, 14: 993, 15: 994, 16: 995, 17: 996, 18: 997, 19: 997, 20: 998, 21: 998, 22: 998, 23: 999, 24: 999, 25: 999, 26: 999, 27: 999, 28: 999, 29: 999, 30: 999, 31: 1000}
    adapters70= {1: 100, 2: 514, 3: 717, 4: 828, 5: 891, 6: 927, 7: 951, 8: 964, 9: 974, 10: 981, 11: 985, 13: 992, 14: 993, 15: 995, 16: 995, 17: 996, 18: 997, 19: 997, 20: 998, 21: 998, 22: 998, 23: 998, 24: 999, 25: 999, 26: 999, 27: 999, 28: 999, 29: 999, 30: 999, 31: 999}
    adapters50= {1: 100, 2: 515, 3: 720, 4: 829, 5: 891, 6: 928, 7: 951, 8: 965, 9: 975, 10: 981, 11: 985, 13: 991, 14: 993, 15: 994, 16: 995, 17: 996, 18: 997, 19: 997, 20: 998, 21: 998, 22: 998, 23: 999, 24: 999, 25: 999, 26: 999, 27: 999, 28: 999, 29: 999, 30: 999, 31: 999}
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
    plt.savefig(f"/Users/giuliatesta/PycharmProjects/masters-thesis-project/agent-based-basic/work/results/gender_bias_adapters_by_time_plot.png", dpi=1000)


# plots the heat map representing the vector labels changing in a specific time step during a simulation
# TODO complete (wrong)
def states_changing_heat_map(states, vector_labels, step, title, path):
    # Extract the vector labels and states for the given step
    vector_data = vector_labels[step][1]
    VL0 = [vl[0] for vl in vector_data]
    VL1 = [vl[1] for vl in vector_data]
    states_data = states[step][1]

    # Check the range of the data
    print(f"VL0 range: {min(VL0)} to {max(VL0)}")
    print(f"VL1 range: {min(VL1)} to {max(VL1)}")
    print(f"States data: {set(states_data)}")  # Should be {-1, 1}

    # Create an empty 1000x1000 heatmap
    heatmap_data = np.zeros((1000, 1000))

    # Scale VL0 and VL1 to fit the grid (from [0,1] to [0,999])
    for i in range(len(VL0)):
        x_idx = int(VL0[i] * 999)  # scale VL0 to grid index
        y_idx = int(VL1[i] * 999)  # scale VL1 to grid index
        heatmap_data[x_idx, y_idx] = states_data[i]  # fill with state (+1/-1)

        # Mask zeros (which are unfilled cells)
    masked_heatmap = np.ma.masked_where(heatmap_data == 0, heatmap_data)

    # Plot the heatmap with masking for zero values
    plt.figure(figsize=(10, 8))
    sns.heatmap(masked_heatmap.T, cmap="coolwarm", cbar=True, center=0,
                xticklabels=False, yticklabels=False, mask=(heatmap_data == 0))

    plt.title(title)
    plt.xlabel("VL0")
    plt.ylabel("VL1")

    # Save the heatmap to the provided path
    plt.savefig(path)
    plt.show()