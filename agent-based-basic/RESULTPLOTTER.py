import sys

import numpy as np
from matplotlib import pyplot as plt

from UTILS import read_pickled_file


class ResultPlotter(object):

    def __init__(self, file_paths):
        self.raw_data = []
        self.file_paths = file_paths
        for file_path in file_paths:
            self.raw_data.append(read_pickled_file(file_path))

    def draw_adapter_by_time_plot(self, file_to_use_index=0):
        plot_file_name = self.file_paths[file_to_use_index].split('/')[-1] + "_PLOT.png"
        adapters = prepare_adapters_by_time(self.raw_data[file_to_use_index])
        print(adapters)
        # Plot the data
        plt.figure(figsize=(10, 6))
        plt.plot(list(adapters.keys()), list(adapters.values()), marker='x')

        plt.xlabel('Time Step')
        plt.ylabel('Number of adapters')
        plt.title('Number of adapters by time')
        plt.grid(True)
        plt.savefig(f"./plots/adapters_by_time/{plot_file_name}")

    def draw_adapter_by_time_different_thresholds_plot(self, thresholds):
        if len(thresholds) != len(self.raw_data):
            print("Inconsistencies in the data to be plotted. Length of thresholds is different from length of raw "
                  "data.")
            sys.exit(1)

        plt.figure(figsize=(10, 6))
        for raw in self.raw_data:
            adapters = prepare_adapters_by_time(raw)
            plt.plot(list(adapters.keys()), list(adapters.values()), marker='x')

        plt.xlabel('Time Step')
        plt.ylabel('Number of adapters')
        plt.title('Number of adapters by time')
        plt.grid(True)
        plt.show()

    def heatmap(self):
        iterations = len(self.raw_data)
        state_vectors = np.vstack([np.array(step[1]) for step in self.raw_data])
        heatmap_data = np.vstack(state_vectors)
        plt.figure()
        plt.imshow(heatmap_data, aspect='auto', cmap='hot', interpolation='nearest')
        plt.colorbar(label='State Value')
        plt.xlabel('State Index')
        plt.ylabel('Iteration')
        plt.title('Heatmap of Simulation States Over Time')
        plt.show()


def prepare_adapters_by_time(raw_data):
    # [time_step, [values for each node]]
    # [0, [-1, -1, 1, -1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1 ]]
    return {raw[0] : raw[1].count(1) for raw in raw_data}
