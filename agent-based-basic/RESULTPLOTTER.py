from matplotlib import pyplot as plt

from UTILS import read_pickled_file


class ResultPlotter(object):

    def __init__(self, file_path):
        self.raw_data = read_pickled_file(file_path)

    def draw_adapter_by_time_plot(self):
        # [0, [[1.0], [1.0], [0.0], [0.0], [0.0], [1.0], [0.0], [1.0], [1.0], [1.0], [0.0], [1.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [0.0], [0.0]]]
        # [1, [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [0.0], [1.0], [1.0], [1.0]]]
        time_steps = len(self.raw_data[0][1])       # Assuming all indices have the same number of time steps
        adapters = {i: 0 for i in range(len(self.raw_data))}
        print(adapters)
        # Count the number of 1's at each time step for each index
        for i, results in self.raw_data:
            for t in range(time_steps):

                adapters[i][t] = int(sum(arr[0] == 1.0 for arr in results[t]))

        # Plot the data
        plt.figure(figsize=(10, 6))
        for i, count_list in adapters.items():
            plt.plot(count_list, label=f'Index {i}')

        plt.xlabel('Time Step')
        plt.ylabel('Count of adapters')
        plt.title('Count of adapters at each time step for each iteration')
        plt.legend()
        plt.grid(True)
        plt.show()