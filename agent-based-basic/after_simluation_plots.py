import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import scipy.stats as st
import utils
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# results coming from case scenarios A*
BASELINE = {}

def compute_plot_data(data):
    # we need the slope, the stationaty value, the confidence Interval
    df = pd.DataFrame(columns=["Sim", "Slope", "Std-Err", "Max-Val", "Stationary-Val", "Confidence-Int"])
    i = 0
    for label, points in data.items():
        x = list(points.keys())
        y = list(points.values())
        res = st.linregress(x, y)
        ci1, ci2 = st.t.interval(0.95, len(y) - 1, loc=np.mean(y), scale=st.sem(y))
        df.loc[i] = {
            "Sim": label,
            "Slope": f"{90 - res.slope: .2f}",
            "Std-Err": f"{res.stderr: .2f}",
            "Max-Val": max(y),
            "Stationary-Val": y[-1],
            "Confidence-Int": f"[{ci1: .0f}, {ci2: .0f}]"
        }
        i += 1
    return df



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
    labels[3][0]: {1: 50, 2: 50, 3: 825, 4: 911, 5: 911, 6: 913, 7: 913, 8: 913, 9: 914, 10: 916, 11: 916, 12: 920, 13: 936, 14: 936, 15: 949},
    labels[3][1]: {1: 200, 2: 200, 3: 904, 4: 989, 5: 989, 6: 991, 7: 991, 8: 991, 9: 992, 10: 992, 11: 992, 12: 992, 13: 992, 14: 992, 15: 993},
    labels[3][2]: {1: 400, 2: 400, 3: 930, 4: 997, 5: 997, 6: 998, 7: 998, 8: 998, 9: 998, 10: 998, 11: 998, 12: 998, 13: 998, 14: 998, 15: 999},
    labels[3][3]: {1: 25, 2: 25, 3: 387, 4: 652, 5: 652, 6: 682, 7: 708, 8: 708, 9: 713, 10: 719, 11: 719, 12: 722, 13: 753, 14: 753, 15: 766},
    labels[3][4]: {1: 100, 2: 100, 3: 513, 4: 843, 5: 843, 6: 878, 7: 909, 8: 909, 9: 918, 10: 927, 11: 927, 12: 930, 13: 935, 14: 935, 15: 940},
    labels[3][5]: {1: 200, 2: 200, 3: 589, 4: 876, 5: 876, 6: 912, 7: 943, 8: 943, 9: 952, 10: 961, 11: 961, 12: 964, 13: 967, 14: 967, 15: 969},
    #labels[2][6]: {1: 284, 2: 284, 3: 621, 4: 891, 5: 891, 6: 924, 7: 948, 8: 948, 9: 955, 10: 961, 11: 961, 12: 965, 13: 968, 14: 968, 15: 970}
    }

    slopes = []
    legend_elements = []
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.xlabel('Time steps')
    plt.ylabel('Adapters')

# Beta Distribution (α = 2, β = 5)
    # with Extreme influence from neighbours (OP = 0.02, OL = 0.98)
    plt.title("Complex Contagion with Majority"
             #"with Extreme Influence by Neighbours (OP = 0.02, OL = 0.98)\n"
              "with Confirmation Bias for RI Initialisation"
              )
    # Availability and
    plt.grid(True)

    #colors = ['deeppink', 'orchid', 'mediumvioletred', 'deepskyblue', 'steelblue', 'blue', 'forestgreen', 'darkgreen',]
    colors = ['deeppink', 'orchid', 'mediumvioletred','deeppink', 'orchid', 'mediumvioletred',]
    #colors = ['deepskyblue', 'steelblue', 'blue', 'deepskyblue', 'steelblue', 'blue',]
    #colors = ['forestgreen', 'darkgreen', 'olivedrab', 'lime','seagreen','lawngreen']
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
       #f"./work/case-scenarios/COMPLEX_CONTAGION/EXTREME-INFLUENCED/"
    "./work/case-scenarios/COMPLEX_CONTAGION/BETA-DISTRIBUTION/RIGID-SOCIETY/"
        +"CC_majority_RI_confirmation_bias.png")

# plots the heat map representing the vector labels.txt changing in a specific time step during a simulation
# plots the heat map representing the vector labels.txt changing in a specific time step during a simulation
def states_changing_heat_map(prefix):
    vector_labels = utils.read_pickled_file(f"{prefix}/trial_0_LPStates_L0_L1__RUN_0.pickled")
    states = utils.read_pickled_file(f"{prefix}/trial_0_LPStates__RUN_0_STATES.pickled")

    plt.figure(figsize=(16, 12))
    i = 1
    for time_step in range(0, 6, 1):
        plt.subplot(4, 2, i)
        i += 1
        print(f"Time step: {time_step}, subplot: {i}")
        vector_data = vector_labels[time_step][1]
        states_data = np.array(states[time_step][1])

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
        plt.title(f'Step {time_step}) Adapters: {adapters_count}, Non-adapters: {non_adapters_count}')
        plt.xlabel('VL0')
        plt.ylabel('VL1')

        plt.suptitle("Heat maps for the first 6 steps of the simulations\n", x=0.57, fontsize=16)
    plt.tight_layout()
    plt.show()
    plt.savefig(f"{prefix}/heat_map.png")


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


simulations = {
    "RI-5": {
        "baseline":{1: 50, 2: 50, 3: 919, 4: 1000, 5: 1000, 6: 1000, 7: 1000, 8: 1000, 9: 1000, 10: 1000, 11: 1000, 12: 1000, 13: 1000, 14: 1000, 15: 1000,},
        "CB": {1: 50, 2: 50, 3: 536, 4: 775, 5: 775, 6: 885, 7: 938, 8: 938, 9: 963, 10: 978, 11: 978, 12: 987, 13: 991, 14: 991, 15: 994},
        "CBAB": {1: 50, 2: 50, 3: 912, 4: 1000, 5: 1000, 6: 1000, 7: 1000, 8: 1000, 9: 1000, 10: 1000, 11: 1000, 12: 1000, 13: 1000, 14: 1000, 15: 1000},
        "LOW-EDU":{1: 50, 2: 50, 3: 900, 4: 998, 5: 998, 6: 998, 7: 998, 8: 998, 9: 998, 10: 998, 11: 998, 12: 998, 13: 998, 14: 998, 15: 998},
        "OLD" : {1: 50, 2: 50, 3: 896, 4: 1000, 5: 1000, 6: 1000, 7: 1000, 8: 1000, 9: 1000, 10: 1000, 11: 1000, 12: 1000, 13: 1000, 14: 1000, 15: 1000},
        "OPPOSITE":{1: 50, 2: 50, 3: 682, 4: 992, 5: 992, 6: 993, 7: 993, 8: 993, 9: 993, 10: 993, 11: 993, 12: 993, 13: 993, 14: 993, 15: 993},
        "FEMALES": {1: 50, 2: 50, 3: 886, 4: 999, 5: 999, 6: 999, 7: 999, 8: 999, 9: 999, 10: 999, 11: 999, 12: 999, 13: 999, 14: 999, 15: 999},
        "YOUNG": {1: 50, 2: 50, 3: 903, 4: 1000, 5: 1000, 6: 1000, 7: 1000, 8: 1000, 9: 1000, 10: 1000, 11: 1000, 12: 1000, 13: 1000, 14: 1000, 15: 1000},
    },
    "RI-20": {
        "baseline": {1: 200, 2: 200, 3: 991, 4: 1000, 5: 1000, 6: 1000, 7: 1000, 8: 1000, 9: 1000, 10: 1000, 11: 1000, 12: 1000, 13: 1000, 14: 1000, 15: 1000,},
        "CB": {1: 200, 2: 200, 3: 638, 4: 811, 5: 811, 6: 900, 7: 948, 8: 948, 9: 971, 10: 981, 11: 981, 12: 987, 13: 993, 14: 993, 15: 995},
        "CBAB": {1: 200, 2: 200, 3: 992, 4: 1000, 5: 1000, 6: 1000, 7: 1000, 8: 1000, 9: 1000, 10: 1000, 11: 1000, 12: 1000, 13: 1000, 14: 1000, 15: 1000},
        "LOW-EDU": {1: 200, 2: 200, 3: 988, 4: 999, 5: 999, 6: 999, 7: 999, 8: 999, 9: 999, 10: 999, 11: 999, 12: 999, 13: 999, 14: 999, 15: 999},
        "OLD":{1: 200, 2: 200, 3: 989, 4: 1000, 5: 1000, 6: 1000, 7: 1000, 8: 1000, 9: 1000, 10: 1000, 11: 1000, 12: 1000, 13: 1000, 14: 1000, 15: 1000},
        "OPPOSITE":{1: 200, 2: 200, 3: 938, 4: 994, 5: 994, 6: 994, 7: 994, 8: 994, 9: 994, 10: 994, 11: 994, 12: 994, 13: 994, 14: 994, 15: 994},
        "FEMALES":{1: 200, 2: 200, 3: 986, 4: 1000, 5: 1000, 6: 1000, 7: 1000, 8: 1000, 9: 1000, 10: 1000, 11: 1000, 12: 1000, 13: 1000, 14: 1000, 15: 1000},
        "YOUNG":{1: 200, 2: 200, 3: 988, 4: 1000, 5: 1000, 6: 1000, 7: 1000, 8: 1000, 9: 1000, 10: 1000, 11: 1000, 12: 1000, 13: 1000, 14: 1000, 15: 1000}
    },
    "RI-40": {
        "baseline": {1: 400, 2: 400, 3: 999, 4: 1000, 5: 1000, 6: 1000, 7: 1000, 8: 1000, 9: 1000, 10: 1000, 11: 1000, 12: 1000, 13: 1000, 14: 1000, 15: 1000, },
        "CB": {1: 400, 2: 400, 3: 732, 4: 869, 5: 869, 6: 928, 7: 961, 8: 961, 9: 977, 10: 986, 11: 986, 12: 991, 13: 995, 14: 995, 15: 996},
        "CBAB":{1: 400, 2: 400, 3: 998, 4: 1000, 5: 1000, 6: 1000, 7: 1000, 8: 1000, 9: 1000, 10: 1000, 11: 1000, 12: 1000, 13: 1000, 14: 1000, 15: 1000},
        "LOW-EDU": {1: 400, 2: 400, 3: 998, 4: 999, 5: 999, 6: 999, 7: 999, 8: 999, 9: 999, 10: 999, 11: 999, 12: 999, 13: 999, 14: 999, 15: 999},
        "OLD": {1: 400, 2: 400, 3: 998, 4: 1000, 5: 1000, 6: 1000, 7: 1000, 8: 1000, 9: 1000, 10: 1000, 11: 1000, 12: 1000, 13: 1000, 14: 1000, 15: 1000},
        "OPPOSITE": {1: 400, 2: 400, 3: 980, 4: 996, 5: 996, 6: 996, 7: 996, 8: 996, 9: 996, 10: 996, 11: 996, 12: 996, 13: 996, 14: 996, 15: 996},
        "FEMALES": {1: 400, 2: 400, 3: 997, 4: 999, 5: 999, 6: 999, 7: 999, 8: 999, 9: 999, 10: 999, 11: 999, 12: 999, 13: 999, 14: 999, 15: 999},
        "YOUNG":{1: 400, 2: 400, 3: 998, 4: 1000, 5: 1000, 6: 1000, 7: 1000, 8: 1000, 9: 1000, 10: 1000, 11: 1000, 12: 1000, 13: 1000, 14: 1000, 15: 1000}
    }
}

# plot_multiple_adapters_by_time()
# for vals in simulations.values():
#     df = compute_plot_data(vals)
#     print(df)