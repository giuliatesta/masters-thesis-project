import sys

from matplotlib import pyplot as plt

from UTILS import read_pickled_file


class ResultPlotter(object):

    def __init__(self, file_paths):
        self.raw_data = []
        for file_path in file_paths:
            self.raw_data.append(read_pickled_file(file_path))

    def draw_adapter_by_time_plot(self, file_to_use_index=0):
        adapters = prepare_adapters_by_time(self.raw_data[file_to_use_index])
        # Plot the data
        plt.figure(figsize=(10, 6))
        plt.plot(list(adapters.keys()), list(adapters.values()),  marker='x')

        plt.xlabel('Time Step')
        plt.ylabel('Number of adapters')
        plt.title('Number of adapters by time')
        plt.grid(True)
        plt.show()

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


def prepare_adapters_by_time(raw_data):
    # [time_step, [values for each node]]

    # [0, [[1.0], [1.0], [0.0], [0.0], [0.0], [1.0], [0.0], [1.0], [1.0], [1.0], [0.0], [1.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [0.0], [0.0]]]
    # [1, [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [0.0], [1.0], [1.0], [1.0]]]
    time_steps = len(raw_data)
    adapters = {i: 0 for i in range(len(raw_data))}
    for i in range(time_steps):
        results = raw_data[i][1]
        adapter_count_for_time_step = (int(sum(arr[0] == 1.0 for arr in results)))
        adapters[i] = adapter_count_for_time_step
    return adapters
