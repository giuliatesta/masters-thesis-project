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
    ["5% Initial Random Adapters", "20% Initial Random Adapters", "40% Initial Random Adapters"], #0
    ["5% Initial Adapters with SI", "20% Initial Adapters with SI", "40% Initial Adapters with SI"],#1
    ["5% RI", "20% RI", "40% RI", "5% SII", "20% SII", "40% SII", "IWS"],#2
    ["5% RI", "20% RI", "40% RI", "5% RI with bias", "20% RI with bias", "40% RI with bias"],#3
    ["5% SII", "20% SII", "40% SII", "5% SII with bias", "20% SII with bias", "40% SII with bias"], #4
    ["WSI", "WSI with bias"],#5
    ["Baseline", "Against Opposite Gender", "Against Women", "Against Young", "Against Old", " Against Low-Educated"]#6
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
    labels[5][0]: {1: 284, 2: 284, 3: 897, 4: 897, 5: 897, 6: 897, 7: 897, 8: 897, 9: 897, 10: 897, 11: 897, 12: 897, 13: 897, 14: 897, 15: 897,},
    labels[5][1]: {1: 284, 2: 284, 3: 893, 4: 893, 5: 893, 6: 893, 7: 893, 8: 893, 9: 893, 10: 893, 11: 893, 12: 893, 13: 893, 14: 893, 15: 893,},
    # labels[4][2]: {1: 200, 2: 200, 3: 895, 4: 895, 5: 895, 6: 895, 7: 895, 8: 895, 9: 895, 10: 895, 11: 895, 12: 895, 13: 895, 14: 895, 15: 895,},
    # labels[4][3]: {1: 25, 2: 25, 3: 660, 4: 660, 5: 660, 6: 660, 7: 660, 8: 660, 9: 660, 10: 660, 11: 660, 12: 660, 13: 660, 14: 660, 15: 660,},
    # labels[4][4]: {1: 100, 2: 100, 3: 850, 4: 850, 5: 850, 6: 850, 7: 850, 8: 850, 9: 850, 10: 850, 11: 850, 12: 850, 13: 850, 14: 850, 15: 850,},
    # labels[4][5]: {1: 200, 2: 200, 3: 887, 4: 887, 5: 887, 6: 887, 7: 887, 8: 887, 9: 887, 10: 887, 11: 887, 12: 887, 13: 887, 14: 887, 15: 887,},
    #labels[3][6]: {1: 284, 2: 284, 3: 897, 4: 897, 5: 897, 6: 897, 7: 897, 8: 897, 9: 897, 10: 897, 11: 897, 12: 897, 13: 897, 14: 897, 15: 897,}
    }
    slopes=[]
    legend_elements=[]
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.xlabel('Time steps')
    plt.ylabel('Adapters')

    plt.title("Complex Contagion with Beta Distribution (Society with more easily influenced agents)\n"
              "with Against Females Social Bias for WSI Initialisation")
    plt.grid(True)
    #colors = ['deeppink', 'orchid', 'mediumvioletred', 'deepskyblue', 'steelblue', 'blue', 'forestgreen', 'darkgreen',]
    colors = [ 'forestgreen', 'darkgreen',]
    linestyle = ['solid', 'dashed', 'dashdot', (5, (10, 3)) , (0, (3, 1, 1, 1, 1, 1)), (0, (3, 5, 1, 5, 1, 5)), "dashdot"]
    markers = ["o", "*", "P", "o", "*", "P"]    #["","","","","","","",""]
    i=0
    if close_up:
        inset_ax = inset_axes(ax, width="40%", height="40%", loc="lower right")  # Adjust size and location
    for label, plot in plots.items():
        x = list(plot.keys())
        y = np.array(list(plot.values()))
        ax.plot(x,y, color=colors[i], label=label, linestyle=linestyle[i], marker=markers[i])
        ci = confidence * np.std(y) / np.sqrt(len(y))
        ax.fill_between(x, (y - ci), (y + ci), color=colors[i], alpha=.1)
        last_value = y[-1]
        first_value = y[0]
        slopes.append(90 - (last_value - first_value) / len(plot))
        if close_up:
            inset_ax.plot(x, y, color=colors[i], linestyle=linestyle[i], marker=markers[i])
        legend_elements.append(Line2D([0], [0], color=colors[i], label=f"{list(plots.keys())[i]}", linestyle=linestyle[i], marker=markers[i]))
        i += 1
    # Add legend to the plot
    fig.legend(handles=legend_elements, loc="lower center", ncol=3)
    plt.subplots_adjust(bottom=0.2)
    print("here")
    if close_up:
        inset_ax.set_xlim(2.5, 4.5)
        inset_ax.set_ylim(600, 950)
        inset_ax.set_title("Close-up", fontsize=10)
        inset_ax.grid(True)

    plt.savefig(
        f"./work/case-scenarios/COMPLEX_CONTAGION/BETA-DISTRIBUTION/OPEN-SOCIETY/"
        +"CC_open_society_WSI_against_females_bias.png")


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