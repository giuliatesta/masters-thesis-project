import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from plotly.graph_objs import Line

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
def plot_multiple_adapters_by_time(confidence=0.95):
    plots = {
    "Baseline":{1: 284, 2: 284, 3: 976, 4: 1000, 5: 1000, 6: 1000, 7: 1000, 8: 1000, 9: 1000, 10: 1000, 11: 1000, 12: 1000, 13: 1000, 14: 1000, 15: 1000, 16: 1000, 17: 1000, 18: 1000, 19: 1000, 20: 1000, 21: 1000, 22: 1000, 23: 1000, 24: 1000, 25: 1000, 26: 1000, 27: 1000, 28: 1000, 29: 1000, 30: 1000},
    "Against opposite gender": {1: 284, 2: 284, 3: 889, 4: 995, 5: 995, 6: 995, 7: 995, 8: 995, 9: 995, 10: 995, 11: 995, 12: 995, 13: 995, 14: 995, 15: 995, 16: 995, 17: 995, 18: 995, 19: 995, 20: 995, 21: 995, 22: 995, 23: 995, 24: 995, 25: 995, 26: 995, 27: 995, 28: 995, 29: 995, 30: 995},
    "Against women": {1: 284, 2: 284, 3: 968, 4: 999, 5: 999, 6: 999, 7: 999, 8: 999, 9: 999, 10: 999, 11: 999, 12: 999, 13: 999, 14: 999, 15: 999, 16: 999, 17: 999, 18: 999, 19: 999, 20: 999, 21: 999, 22: 999, 23: 999, 24: 999, 25: 999, 26: 999, 27: 999, 28: 999, 29: 999, 30: 999},
    "Against young": {1: 284, 2: 284, 3: 968, 4: 1000, 5: 1000, 6: 1000, 7: 1000, 8: 1000, 9: 1000, 10: 1000, 11: 1000, 12: 1000, 13: 1000, 14: 1000, 15: 1000, 16: 1000, 17: 1000, 18: 1000, 19: 1000, 20: 1000, 21: 1000, 22: 1000, 23: 1000, 24: 1000, 25: 1000, 26: 1000, 27: 1000, 28: 1000, 29: 1000, 30: 1000},
    "Against old":  {1: 284, 2: 284, 3: 974, 4: 1000, 5: 1000, 6: 1000, 7: 1000, 8: 1000, 9: 1000, 10: 1000, 11: 1000, 12: 1000, 13: 1000, 14: 1000, 15: 1000, 16: 1000, 17: 1000, 18: 1000, 19: 1000, 20: 1000, 21: 1000, 22: 1000, 23: 1000, 24: 1000, 25: 1000, 26: 1000, 27: 1000, 28: 1000, 29: 1000, 30: 1000},
    "Against low-educated": {1: 284, 2: 284, 3: 967, 4: 999, 5: 999, 6: 999, 7: 999, 8: 999, 9: 999, 10: 999, 11: 999, 12: 999, 13: 999, 14: 999, 15: 999, 16: 999, 17: 999, 18: 999, 19: 999, 20: 999, 21: 999, 22: 999, 23: 999, 24: 999, 25: 999, 26: 999, 27: 999, 28: 999, 29: 999, 30: 999}

    }
    text = ""
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.xlabel('Time steps')
    plt.ylabel('Adapters')
    plt.title('Simple contagion with Initialization with Attribute with Social Biases')
    plt.grid(True)
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'pink']
    i=0
    inset_ax = inset_axes(ax, width="40%", height="40%", loc="center right")  # Adjust size and location
    for label, plot in plots.items():
        x = list(plot.keys())
        y = np.array(list(plot.values()))
        ax.plot(x,y, color=colors[i], label=label)
        i += 1
        ci = confidence * np.std(y) / np.sqrt(len(y))
        ax.fill_between(x, (y - ci), (y + ci), alpha=.1)
        last_value = y[-1]
        first_value = y[0]
        slope = 90 - (last_value - first_value) / len(plot)
        text += f"slope for {label}: {slope:.2f}"
        inset_ax.plot(x, y)
    plt.text(8, 200, text, fontsize=10,bbox=dict(facecolor='none', alpha=0.2))

    legend_elements = [Line2D([0], [0], color=colors[i], label=list(plots.keys())[i]) for i in range(0, 6)]

    # Add legend to the plot
    fig.legend(handles=legend_elements, loc="lower center")
    print("here")
    inset_ax.set_xlim(2.5, 4)
    inset_ax.set_ylim(800, 1050)
    inset_ax.set_title("Close-up", fontsize=10)
    inset_ax.grid(True)

    plt.savefig(
        f"./work/case-scenarios/SIMPLE_CONTAGION/would-subscribe-attribute-with-social-bias.png",
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