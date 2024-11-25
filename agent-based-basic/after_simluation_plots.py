import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from conf import RESULTS_DIR
import utils
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# results coming from case scenarios A*
BASELINE = {}


# plots the evolution of the number of adapters as the time steps proceeds in the simulation
# shows the differences with the BASELINE and indicates the slope of the curve
def draw_adapter_by_time_plot(adapters, results_file_path, title, additional_text="", confidence=0.95):
    plt.figure(figsize=(10, 6))
    x = np.array(list(adapters.keys()))
    y = np.array(list(adapters.values()))
    for key, value in adapters.items():
        if value == y[-1]:
            plt.axvline(x=key, linestyle='--', color=('cyan', 0.5), label='Convergence threshold')
            break

    slope = 90 - (y[-1] - y[0]) / len(adapters)

    ci = confidence * np.std(y) / np.sqrt(len(y))
    fig, ax = plt.subplots()
    ax.plot(x, y, marker='x', label="Current simulation")
    ax.fill_between(x, (y - ci), (y + ci), color='b', alpha=.1)

    # plt.plot(list(BASELINE.keys()), list(BASELINE.values()), 'g--', linewidth=1, label='Base Line')
    plt.xlabel('Time steps')
    plt.ylabel('Adapters')
    plt.title('Number of adapters by time' if title == "" else title)
    plt.legend(loc='lower right')
    if additional_text != "":
        additional_text += f"\n• Slope: : {slope:.2f}"
        additional_text += f"\n• Max value: {max(y)}"
        print(additional_text)
        plt.text(13, max(adapters.values()) / 2, additional_text, fontsize=10, bbox=dict(facecolor='none', alpha=0.2))
    plt.grid(True)
    plt.savefig(f"{results_file_path}/avg_adapters_by_time_plot{'_annotated' if additional_text != '' else ''}.png",
                dpi=1000)


# plots multiple simulations on the same plot
# simulations indicates the evolution of the number of adapters in the networks
def plot_multiple_adapters_by_time():
    adapters1 = {1: 25, 2: 25, 3: 747, 4: 1000, 5: 1000, 6: 1000, 7: 1000, 8: 1000, 9: 1000, 10: 1000, 11: 1000, 12: 1000, 13: 1000, 14: 1000, 15: 1000, 16: 1000, 17: 1000, 18: 1000, 19: 1000, 20: 1000, 21: 1000, 22: 1000, 23: 1000, 24: 1000, 25: 1000, 26: 1000, 27: 1000, 28: 1000, 29: 1000, 30: 1000}
    adapters2 = {1: 25, 2: 25, 3: 684, 4: 684, 5: 684, 6: 684, 7: 684, 8: 684, 9: 684, 10: 684, 11: 684, 12: 684, 13: 684, 14: 684, 15: 684, 16: 684, 17: 684, 18: 684, 19: 684, 20: 684, 21: 684, 22: 684, 23: 684, 24: 684, 25: 684, 26: 684, 27: 684, 28: 684, 29: 684, 30: 684}
    adapters3 = {1: 25, 2: 25, 3: 675, 4: 675, 5: 675, 6: 675, 7: 675, 8: 675, 9: 675, 10: 675, 11: 675, 12: 675, 13: 675, 14: 675, 15: 675, 16: 675, 17: 675, 18: 675, 19: 675, 20: 675, 21: 675, 22: 675, 23: 675, 24: 675, 25: 675, 26: 675, 27: 675, 28: 675, 29: 675, 30: 675}
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(list(adapters1.keys()), list(adapters1.values()), marker='o', label="Baseline")
    ax.plot(list(adapters2.keys()), list(adapters2.values()), marker='o', label="Confirmation Bias")
    ax.plot(list(adapters3.keys()), list(adapters3.values()), marker='.', label="Availability Bias")
    last_value = list(adapters1.values())[-1]
    first_value = list(adapters1.values())[0]
    slope1 = 90 - (last_value - first_value) / len(adapters1)
    last_value = list(adapters2.values())[-1]
    first_value = list(adapters2.values())[0]
    slope2 = 90 - (last_value - first_value) / len(adapters2)
    last_value = list(adapters3.values())[-1]
    first_value = list(adapters3.values())[0]
    slope3 = 90 - (last_value - first_value) / len(adapters3)
    plt.text(8, 50, f"slope 5%: {slope1:.2f}\nslope 20%: {slope2:.2f}\nslope 40%: {slope3:.2f}", fontsize=10,
             bbox=dict(facecolor='none', alpha=0.2))

    # plt.plot(list(BASELINE.keys()), list(BASELINE.values()), 'm--', linewidth=1, label='Base Line')
    plt.xlabel('Time steps')
    plt.ylabel('Adapters')
    plt.title('Simple contagion with Initialization with Sharing Index with 5% and with Cognitive Biases')
    plt.legend(loc='center right')
    plt.grid(True)

    inset_ax = inset_axes(ax, width="30%", height="30%", loc="lower right")  # Adjust size and location
    inset_ax.plot(list(adapters1.keys()), list(adapters1.values()))
    inset_ax.plot(list(adapters2.keys()), list(adapters2.values()))
    inset_ax.plot(list(adapters3.keys()), list(adapters3.values()))
    inset_ax.set_xlim(2, 5)
    inset_ax.set_ylim(600, 1050)
    inset_ax.set_title("Close-up", fontsize=10)
    inset_ax.grid(True)

    plt.savefig(
        f"./work/case-scenarios/SIMPLE_CONTAGION/adapters-with-SI-with-cognitive-bias.png",
        dpi=1000)


# plots the heat map representing the vector labels.txt changing in a specific time step during a simulation
def states_changing_heat_map(states, vector_labels, step):
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
    im = plt.imshow(grid, cmap="RdPu", interpolation='nearest', vmin=0, vmax=1)
    plt.colorbar(im)
    # plt.title(title)
    plt.title(f'Step {step}) Adapters: {adapters_count}, Non-adapters: {non_adapters_count}')
    plt.xlabel('VL0')
    plt.ylabel('VL1')
    #  plt.show()
    # plt.savefig(path)


def description_text_for_plots(rule, simulation_id, sim_threshold, vl_update, initialisation, adapters_perc,
                               cognitive_bias, alpha, beta):
    text = (
                f"• Initialisation of VLs: {initialisation} {f'({adapters_perc}%)' if initialisation != 'would-subscribe-attribute' else ''}\n"
                + f"• VLs update method: {vl_update}\n"
                + f"• State update: {cognitive_bias}\n"
                + f"• Similarity threshold: {sim_threshold}\n")
    title = f"Number of adapters by time\n({rule} - SIM {simulation_id})"
    if rule == "same-weights":
        text += "• OP: 1 / (k+1), OL: 1 / (k+1)"
    if rule == "beta-dist":
        text += f"• OP: scaled similarity weights\n• OL: beta(alpha = {alpha}, beta = {beta})\n"
        if alpha == 2 and beta == 2:
            text += f"(quasi-normal distribution for beta - SIM {simulation_id})"
        if alpha == 2 and beta == 5:
            text += f"(society with rigid agents - SIM {simulation_id})"
        if alpha == 5 and beta == 2:
            text += f"(society with open-to-change agents - SIM {simulation_id})"
    if rule == "over-confidence":
        text += "• OP: 0.8, OL: 0.2\n"
    if rule == "over-influenced":
        text += "• OP: 0.2, OL: 0.8\n"
    if rule == "extreme-influenced":
        text += "• OP: 0.02, OL: 0.98\n"
    if rule == "simple-contagion":
        text += "• At the first interaction with an adapter, it becomes adapter"
    if rule == "majority":
        text += "• OP: 0.0, OL: 1 (w_ij = 1 / # adapters if adapter; otherwise 0)\n"
    return title, text



plot_multiple_adapters_by_time()