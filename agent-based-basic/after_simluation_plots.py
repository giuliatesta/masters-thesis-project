import numpy as np
from matplotlib import pyplot as plt

# results coming from simulation_03_BASE_LINE which has 20% of initial adapters and no biases.
BASELINE = {1: 100, 2: 513, 3: 719, 4: 828, 5: 887, 6: 924, 7: 948, 8: 963, 9: 973, 10: 980, 11: 984, 13: 991, 14: 993,
            15: 994, 16: 995, 17: 996, 18: 997, 19: 997, 20: 998, 21: 998, 22: 998, 23: 998, 24: 998, 25: 999, 26: 999,
            27: 999, 28: 999, 29: 999, 30: 999, 31: 999}


# plots the evolution of the number of adapters as the time steps proceeds in the simulation
# shows the differences with the BASELINE and indicates the slope of the curve
def draw_adapter_by_time_plot(adapters, results_file_path, title, additional_text=""):
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
   #  plt.text(28, 280, f"slope: {slope:.2f}", fontsize=10, bbox=dict(facecolor='none', alpha=0.2))
    # for key, value in adapters.items():
    #     print(f"({key}, {value}")
    #     plt.text(key, value, f"({key:.2f}, {value})", fontsize=6, ha="center")
    # BASELINE = {1: 133, 2: 526, 3: 722, 4: 828, 5: 892, 6: 927, 7: 949, 8: 964, 9: 975, 10: 981, 11: 985, 13: 991, 14: 993, 15: 995, 16: 995, 17: 996, 18: 997, 19: 997, 20: 998, 21: 998, 22: 998, 23: 998, 24: 998, 25: 999, 26: 999, 27: 999, 28: 999, 29: 999, 30: 999, 31: 999}
    # plt.plot(list(BASELINE.keys()), list(BASELINE.values()), 'g--', linewidth=1, label='Base Line')
    plt.xlabel('Time steps')
    plt.ylabel('Adapters')
    plt.title('Number of adapters by time' if title == "" else title)
    plt.legend(loc='lower right')
    additional_text += f"\nSlope: : {slope:.2f}"
    print(additional_text)
    plt.text(15, 400, additional_text, fontsize=10, bbox=dict(facecolor='none', alpha=0.2))
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
    plt.text(25, 100, f"slope 90%: {slope90:.2f}\nslope 70%: {slope70:.2f}\nslope 50%: {slope50:.2f}", fontsize=10,
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
def states_changing_heat_map(states, vector_labels, step, title, path):
    vector_data = vector_labels[step][1]
    states_data = np.array(states[step][1])

    adapters_count = np.sum(states_data == 1)
    non_adapters_count = len(states_data) - adapters_count

    print(f"Adapters: {adapters_count}")
    print(f"Non Adapters: {non_adapters_count}")

    VL0 = np.array([vl[0] for vl in vector_data])
    VL1 = np.array([vl[1] for vl in vector_data])

    # Calculate the number of rows and columns for a square-like grid
    n_nodes = len(VL0)
    n_rows = int(np.sqrt(n_nodes))
    n_cols = int(np.ceil(n_nodes / n_rows))

    # Reshape the data into a 2D grid, padding with NaN if necessary
    grid = np.full((n_rows * n_cols), np.nan)
    grid[:n_nodes] = VL0
    grid = grid.reshape(n_rows, n_cols)


    # Create the plot
    # plt.figure()
    im = plt.imshow(grid, cmap="RdPu",  interpolation='nearest', vmin=0, vmax=1)
    plt.colorbar(im)
    # plt.title(title)
    plt.title(f'Step {step}) Adapters: {adapters_count}, Non-adapters: {non_adapters_count}')
    plt.xlabel('VL0')
    plt.ylabel('VL1')
   #  plt.show()
    #plt.savefig(path)
