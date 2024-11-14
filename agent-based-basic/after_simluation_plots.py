import numpy as np
from matplotlib import pyplot as plt

# results coming from case scenarios A*
BASELINE = {}


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
    adapters90 = {}
    adapters70 = {}
    adapters50 = {}
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

    plt.plot(list(BASELINE.keys()), list(BASELINE.values()), 'm--', linewidth=1, label='Base Line')
    plt.xlabel('Time steps')
    plt.ylabel('Adapters')
    plt.title('Gender bias simulations: pro same gender (90/70/50)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(
        f"./work/results/gender_bias_adapters_by_time_plot.png",
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


def description_text_for_plots(rule, simulation_id, ):
    from main_LPA import ALPHA, BETA, SIMILARITY_THRESHOLD, USE_SHARING_INDEX
    text = ("Adapters: WOULD_SUBSCRIBE_CAR_SHARING (133)\n"
            + f"Similarity threshold: {SIMILARITY_THRESHOLD}\n"
            + f"Vector label changing: NO BIAS"
            + f"State determination: WITH{'' if USE_SHARING_INDEX else 'OUT'} INDEX")
    title = "Number of adapters by time\n"
    if rule == "same-weights":
        text += "OP: 1 / (k+1), OL: 1 / (k+1)"
        title += f"(BASELINE with same weights - SIM {simulation_id})"
    if rule == "beta-dist":
        text += f"OP: scaled similarity weights\nOL: beta(alpha = {ALPHA}, beta = {BETA})"
        if ALPHA == 2 and BETA == 2:
            title += f"(quasi-normal distribution for beta - SIM {simulation_id})"
        if ALPHA == 2 and BETA == 5:
            title += f"(society with rigid agents - SIM {simulation_id})"
        if ALPHA == 5 and BETA == 2:
            title += f"(society with open-to-change agents - SIM {simulation_id})"
    if rule == "over-confidence":
        text += "OP: 0.8, OL: 0.2\n"
        title += f"(overconfidence bias - SIM {simulation_id})"
    if rule == "over-influenced":
        text += "OP: 0.2, OL: 0.8\n"
        title += f"(over-influenced bias - SIM {simulation_id})"
    return title, text
