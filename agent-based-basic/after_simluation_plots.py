import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import scipy.stats as st
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
    ["Baseline WSI", "Against Low-Educated", "Against Old", "Against Opposite Gender", "Against Females", "Against Young",]#6
]
titles = [
    'Complex contagion with Random Initialization ',
    'Complex contagion with Initialization with SI',
    'Complex contagion with Initialization with attribute'
]

# plots multiple simulations on the same plot
# simulations indicates the evolution of the number of adapters in the networks
def plot_multiple_adapters_by_time(confidence=0.95, close_up=False):
    plots = {
        labels[5][0]: {1: 284, 2: 284, 3: 621, 4: 891, 5: 891, 6: 924, 7: 948, 8: 948, 9: 955, 10: 961, 11: 961, 12: 965, 13: 968, 14: 968, 15: 970},
        labels[5][1]: {1: 284, 2: 284, 3: 329, 4: 543, 5: 543, 6: 667, 7: 768, 8: 768, 9: 829, 10: 873, 11: 873, 12: 898, 13: 918, 14: 918, 15: 929},
        # labels[6][2]: {1: 284, 2: 284, 3: 284, 4: 284, 5: 284, 6: 969, 7: 972, 8: 972, 9: 972, 10: 972, 11: 972, 12: 972, 13: 972, 14: 972, 15: 972},
        # labels[6][3]: {1: 284, 2: 284, 3: 284, 4: 284, 5: 284, 6: 888, 7: 888, 8: 888, 9: 888, 10: 888, 11: 888, 12: 888, 13: 888, 14: 888, 15: 888},
        # labels[6][4]: {1: 284, 2: 284, 3: 284, 4: 284, 5: 284, 6: 959, 7: 962, 8: 962, 9: 962, 10: 964, 11: 964, 12: 965, 13: 965, 14: 965, 15: 965},
        # labels[6][5]: {1: 284, 2: 284, 3: 284, 4: 284, 5: 284, 6: 961, 7: 961, 8: 961, 9: 961, 10: 961, 11: 961, 12: 962, 13: 962, 14: 962, 15: 962},
        #labels[2][6]: {1: 400, 2: 400, 3: 400, 4: 400, 5: 400, 6: 720, 7: 855, 8: 855, 9: 917, 10: 944, 11: 944, 12: 960, 13: 969, 14: 969, 15: 974}
    }
    slopes = []
    legend_elements = []
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.xlabel('Time steps')
    plt.ylabel('Adapters')

# Beta Distribution (α = 2, β = 5)
    # with Extreme influence from neighbours (OP = 0.02, OL = 0.98)
    plt.title("Complex Contagion with Beta Distribution (α = 5, β = 2)\n"
             #"with Extreme Influence by Neighbours (OP = 0.02, OL = 0.98)\n"
             "with Availability and Confirmation Biases for WSI Initialisation"
              )
    # Availability and
    plt.grid(True)

    #colors = ['deeppink', 'orchid', 'mediumvioletred', 'deepskyblue', 'steelblue', 'blue', 'forestgreen', 'darkgreen',]
    #colors = ['deeppink', 'orchid', 'mediumvioletred','deeppink', 'orchid', 'mediumvioletred',]
    #colors = ['deepskyblue', 'steelblue', 'blue', 'deepskyblue', 'steelblue', 'blue',]
    colors = ['forestgreen', 'darkgreen', 'olivedrab', 'lime','seagreen','lawngreen']
    linestyle = ['solid', 'dashed', 'dashdot', (5, (10, 3)) , (0, (3, 1, 1, 1, 1, 1)), (0, (3, 5, 1, 5, 1, 5)), "dashdot"]
    markers = ["o", "*", "P", "o", "*", "P"]
    #markers=["","","","","","","",""]
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
        inset_ax.set_xlim(3, 7)
        inset_ax.set_ylim(650, 950)
        inset_ax.set_title("Close-up", fontsize=10)
        inset_ax.grid(True)

    plt.savefig(
       f"./work/case-scenarios/COMPLEX_CONTAGION/BETA-DISTRIBUTION/RIGID-SOCIETY/"
        +"CC_rigid_society_WSI_both_bias.png")

def states_changing_heat_map(prefix):
    vector_labels = utils.read_pickled_file(f"{prefix}/trial_0_LPStates_L0_L1__RUN_0.pickled")
    states = utils.read_pickled_file(f"{prefix}/trial_0_LPStates__RUN_0_STATES.pickled")

    plt.figure(figsize=(16, 12))
    subplot_index = 1

    for time_step in range(0, 6):
        ax = plt.subplot(4, 2, subplot_index)
        subplot_index += 1

        vector_data = vector_labels[time_step][1]
        states_data = np.array(states[time_step][1])
        adapters_count = np.sum(states_data == 1)
        non_adapters_count = len(states_data) - adapters_count
        VL0 = np.array([vl[0] for vl in vector_data])
        VL1 = np.array([vl[1] for vl in vector_data])

        n_bins = 10
        VL0_bins = np.linspace(0, 1, n_bins + 1)
        VL1_bins = np.linspace(0, 1, n_bins + 1)
        proportion_grid = np.zeros((n_bins, n_bins))

        # Calculate proportions for each bin

        for x in range(n_bins):
            binx = VL0_bins[x]
            for y in range(n_bins):
                biny = VL1_bins[y]
                adapters = 0
                non_adapters = 0
                for i in range(len(vector_data)):
                    vl = vector_data[i]
                    if (binx <= vl[0] < binx + 1) and (biny <= vl[1] < biny + 1):
                        state = states_data[i]
                        if state == 1:
                            adapters += 1
                        else:
                            non_adapters += 1
                proportion_grid[y, x] = float(adapters / non_adapters) if non_adapters > 0 else 0

                ax.text(
                    x, y, f"{proportion_grid[y, x]:.0f}",
                    ha="center", va="center", color="black", fontsize=5
                )
        # Plot the heatmap
        im = ax.imshow(
            proportion_grid, cmap="RdPu", interpolation="nearest",
            vmin=0, vmax=np.nanmax(proportion_grid), origin="lower"
        )

        # Add a colorbar to the subplot
        cbar = plt.colorbar(im, ax=ax)

        # Add title and labels
        ax.set_title(f"Step {time_step} | Adapters: {adapters_count}, Non-adapters: {non_adapters_count}")
        ax.set_xlabel("VL0")
        ax.set_ylabel("VL1")

    # Global adjustments for the figure
    plt.suptitle("Heat maps for the first 6 steps of the simulations", fontsize=20, x=0.5, y=0.98)
    plt.subplots_adjust(wspace=0.4, hspace=0.5)  # Adjust spacing between subplots
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save and display the figure
    plt.savefig(f"{prefix}/heat_map.png")
    plt.show()



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
