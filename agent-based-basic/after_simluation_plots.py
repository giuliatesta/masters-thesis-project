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

labels = [
    ["5% Initial Random Adapters", "20% Initial Random Adapters", "40% Initial Random Adapters"],
    ["5% Initial Adapters with SI", "20% Initial Adapters with SI", "40% Initial Adapters with SI"],
    ["Baseline", "Confirmation Bias", "Availability Bias"],
    ["Baseline", "Against Opposite Gender", "Against Women", "Against Young", "Against Old", " Against Low-Educated"]
]
titles = [
    'Complex contagion with Random Initialization ',
    'Complex contagion with Initialization with SI',
    'Complex contagion with Initialization with attribute'
]

file_names = [
    "random-adapters-comparison.png",   #0
    "random-adapters-with-cognitive-bias.png",#1
    "random-adapters-with-social-bias.png",#2
    "adapters-with-SI-comparison.png",#3
    "adapters-with-SI-with-cognitive-bias.png",#4
    "adapters-with-SI-with-social-bias.png",#5
    "would-subscribe-attribute-with-cognitive-bias.png",#6
    "would-subscribe-attribute-with-social-bias.png",#7
]
# plots multiple simulations on the same plot
# simulations indicates the evolution of the number of adapters in the networks
def plot_multiple_adapters_by_time(confidence=0.95, close_up=True):
    plots = {
    labels[0][0]: {1: 50, 2: 50, 3: 810, 4: 810, 5: 810, 6: 810, 7: 810, 8: 810, 9: 810, 10: 810, 11: 810, 12: 810, 13: 810, 14: 810, 15: 810, 16: 810, 17: 810, 18: 810, 19: 810, 20: 810, 21: 810, 22: 810, 23: 810, 24: 810, 25: 810, 26: 810, 27: 810, 28: 810, 29: 810, 30: 810},
    labels[0][1]: {1: 200, 2: 200, 3: 906, 4: 913, 5: 913, 6: 913, 7: 913, 8: 913, 9: 913, 10: 913, 11: 913, 12: 913, 13: 913, 14: 913, 15: 913, 16: 913, 17: 913, 18: 913, 19: 913, 20: 913, 21: 913, 22: 913, 23: 913, 24: 913, 25: 913, 26: 913, 27: 913, 28: 913, 29: 913, 30: 913},
    labels[0][2]: {1: 400, 2: 400, 3: 928, 4: 954, 5: 954, 6: 954, 7: 954, 8: 954, 9: 954, 10: 954, 11: 954, 12: 954, 13: 954, 14: 954, 15: 954, 16: 954, 17: 954, 18: 954, 19: 954, 20: 954, 21: 954, 22: 954, 23: 954, 24: 954, 25: 954, 26: 954, 27: 954, 28: 954, 29: 954, 30: 954}
    #labels[3][3]: {1: 284, 2: 284, 3: 363, 4: 364, 5: 364, 6: 364, 7: 364, 8: 364, 9: 364, 10: 364, 11: 364, 12: 364, 13: 364, 14: 364, 15: 364, 16: 364, 17: 364, 18: 364, 19: 364, 20: 364, 21: 364, 22: 364, 23: 364, 24: 364, 25: 364, 26: 364, 27: 364, 28: 364, 29: 364, 30: 364},
    #labels[3][4]: {1: 284, 2: 284, 3: 357, 4: 357, 5: 357, 6: 357, 7: 357, 8: 357, 9: 357, 10: 357, 11: 357, 12: 357, 13: 357, 14: 357, 15: 357, 16: 357, 17: 357, 18: 357, 19: 357, 20: 357, 21: 357, 22: 357, 23: 357, 24: 357, 25: 357, 26: 357, 27: 357, 28: 357, 29: 357, 30: 357},
    #labels[3][5]: {1: 284, 2: 284, 3: 355, 4: 359, 5: 359, 6: 359, 7: 359, 8: 359, 9: 359, 10: 359, 11: 359, 12: 359, 13: 359, 14: 359, 15: 359, 16: 359, 17: 359, 18: 359, 19: 359, 20: 359, 21: 359, 22: 359, 23: 359, 24: 359, 25: 359, 26: 359, 27: 359, 28: 359, 29: 359, 30: 359}
    }
    slopes=[]
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.xlabel('Time steps')
    plt.ylabel('Adapters')

    plt.title(titles[0])
    plt.grid(True)
    colors = ['purple', 'pink', 'blue', 'red', 'green', 'orange']
    i=0
    if close_up:
        inset_ax = inset_axes(ax, width="40%", height="40%", loc="center right")  # Adjust size and location
    for label, plot in plots.items():
        x = list(plot.keys())
        y = np.array(list(plot.values()))
        ax.plot(x,y, color=colors[i], label=label)
        ci = confidence * np.std(y) / np.sqrt(len(y))
        ax.fill_between(x, (y - ci), (y + ci), color=colors[i], alpha=.1)
        last_value = y[-1]
        first_value = y[0]
        slopes.append(90 - (last_value - first_value) / len(plot))
        if close_up:
            inset_ax.plot(x, y, color=colors[i])
        i += 1
    legend_elements = [Line2D([0], [0], color=colors[i], label=f"{list(plots.keys())[i]} ({slopes[i]: .2f})") for i in range(0, len(plots))]

    # Add legend to the plot
    fig.legend(handles=legend_elements, loc="lower center", ncol=1)
    plt.subplots_adjust(bottom=0.25)
    print("here")
    if close_up:
        inset_ax.set_xlim(2.5, 4.5)
        inset_ax.set_ylim(750, 1000)
        inset_ax.set_title("Close-up", fontsize=10)
        inset_ax.grid(True)

    plt.savefig(
        f"./work/case-scenarios/COMPLEX_CONTAGION/BETA-DISTRIBUTION/OPEN-SOCIETY/"
        +file_names[0])


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
                               bias, alpha, beta):
    text = (
                f"• Initialisation of VLs: {initialisation} {f'({adapters_perc}%)' if initialisation != 'would-subscribe-attribute' else ''}\n"
                + f"• VLs update method: {vl_update}\n"
                + f"• Applied Bias: {bias}\n"
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