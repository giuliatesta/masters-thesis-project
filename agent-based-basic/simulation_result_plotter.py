import sys

import numpy as np
from matplotlib import pyplot as plt

from utils import read_pickled_file

# results coming from simulation_03 which has 20% of initial adapters and no biases.
BASE_LINE = {1: 99, 2: 540, 3: 740, 4: 838, 5: 886, 6: 919, 7: 948, 8: 966, 9: 974, 10: 980, 11: 983, 13: 986, 14: 987, 15: 988, 16: 989, 17: 990, 18: 990, 19: 990, 20: 991, 21: 991, 22: 991, 23: 992, 24: 992, 25: 992, 26: 992, 27: 992, 28: 992, 29: 992, 30: 992, 31: 992}


class ResultPlotter(object):
    def __init__(self, file_paths):
        self.raw_data = []
        self.file_paths = file_paths
        for file_path in file_paths:
            self.raw_data.append(read_pickled_file(file_path))

    def draw_adapter_by_time_plot(self, file_to_use_index=0):
        adapters = prepare_adapters_by_time(self.raw_data[file_to_use_index])
        print(adapters)
        # Plot the data
        plt.figure(figsize=(10, 6))
        plt.plot(list(adapters.keys()), list(adapters.values()), marker='x', label="Current simulation")
        # for key, value in adapters.items():
        #     print(f"({key}, {value}")
        #     plt.text(key, value, f"({key:.2f}, {value})", fontsize=6, ha="center")
        plt.plot(list(BASE_LINE.keys()), list(BASE_LINE.values()), 'g--', linewidth=1, label='Base Line')
        plt.xlabel('Time steps')
        plt.ylabel('Adapters')
        plt.title('Number of adapters by time')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.savefig(self.file_paths[file_to_use_index].rsplit('/', 1)[0] + "/adapters_by_time_plot.png", dpi=1200)

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
        state_vectors = [np.array(step[1]) for step in self.raw_data]
        heatmap_data = np.vstack(state_vectors)
        plt.figure()
        plt.imshow(heatmap_data, aspect='auto', cmap='hot', interpolation='nearest')
        plt.colorbar(label='State Value')
        plt.xlabel('State Index')
        plt.ylabel('Iteration')
        plt.title('Heatmap of Simulation States Over Time')
        plt.show()


def prepare_adapters_by_time(raw_data):
    # [time_step, [values for each node]] = [0, [-1, -1, 1, -1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1]]
    adapters = {}
    for raw in raw_data:
        acc = 0
        for val in raw[1]:
            if val == 1:
                acc += 1
        adapters[raw[0]] = acc
    return adapters
