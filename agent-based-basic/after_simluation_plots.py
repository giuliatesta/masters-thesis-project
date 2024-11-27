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
def plot_multiple_adapters_by_time(confidence=0.95, close_up=False):
    plots = {
    labels[3][0]: {1: 50, 2: 50, 3: 145, 4: 145, 5: 145, 6: 145, 7: 145, 8: 145, 9: 145, 10: 145, 11: 145, 12: 145, 13: 145, 14: 145, 15: 145,},
    labels[3][1]: {1: 200, 2: 200, 3: 283, 4: 322, 5: 322, 6: 322, 7: 322, 8: 322, 9: 322, 10: 322, 11: 322, 12: 322, 13: 322, 14: 322, 15: 322,},
    labels[3][2]: {1: 400, 2: 400, 3: 467, 4: 656, 5: 656, 6: 656, 7: 656, 8: 656, 9: 656, 10: 656, 11: 656, 12: 656, 13: 656, 14: 656, 15: 656,},
    labels[3][3]: {1: 50, 2: 50, 3: 139, 4: 139, 5: 139, 6: 139, 7: 139, 8: 139, 9: 139, 10: 139, 11: 139, 12: 139, 13: 139, 14: 139, 15: 139,},
    labels[3][4]: {1: 200, 2: 200, 3: 290, 4: 322, 5: 322, 6: 322, 7: 322, 8: 322, 9: 322, 10: 322, 11: 322, 12: 322, 13: 322, 14: 322, 15: 322,},
    labels[3][5]: {1: 400, 2: 400, 3: 460, 4: 578, 5: 578, 6: 578, 7: 578, 8: 578, 9: 578, 10: 578, 11: 578, 12: 578, 13: 578, 14: 578, 15: 578,},
    #labels[2][6]: {1: 284, 2: 284, 3: 364, 4: 364, 5: 364, 6: 364, 7: 364, 8: 364, 9: 364, 10: 364, 11: 364, 12: 364, 13: 364, 14: 364, 15: 364,}
    }
    slopes=[]
    legend_elements=[]
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.xlabel('Time steps')
    plt.ylabel('Adapters')

    plt.title("Complex Contagion with Beta Distribution (Society with rigid agents)\n"
              "with Against Young Social Bias for RI Initialisation")
    plt.grid(True)
    #colors = ['deeppink', 'orchid', 'mediumvioletred', 'deepskyblue', 'steelblue', 'blue', 'forestgreen', 'darkgreen',]
    colors = ['deeppink', 'orchid', 'mediumvioletred','deeppink', 'orchid', 'mediumvioletred',]
    linestyle = ['solid', 'dashed', 'dashdot', (5, (10, 3)) , (0, (3, 1, 1, 1, 1, 1)), (0, (3, 5, 1, 5, 1, 5)), "dashdot"]
    markers = ["o", "*", "P", "o", "*", "P"]
    #markers=["","","","","","","",""]
    i=0
    if close_up:
        inset_ax = inset_axes(ax, width="40%", height="40%", loc="center right")  # Adjust size and location
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
        inset_ax.set_xlim(2.5, 4)
        inset_ax.set_ylim(350, 370)
        inset_ax.set_title("Close-up", fontsize=10)
        inset_ax.grid(True)

    plt.savefig(
        f"./work/case-scenarios/COMPLEX_CONTAGION/BETA-DISTRIBUTION/RIGID-SOCIETY/"
        +"CC_rigid_society_RI_against_young_bias.png")

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