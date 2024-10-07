import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from networkx import number_connected_components
from pandas import DataFrame

def distribution_over_nodes_count(matrix, title="", x_label="", bins=20):
    print(f"Plotting the distribution {title}")
    plt.figure()
    n, bins, patches = plt.hist(matrix, bins=bins)
    cmap = plt.get_cmap('plasma')
    bin_heights = (n - np.min(n)) / (np.max(n) - np.min(n))
    for patch, color_value in zip(patches, bin_heights):
        patch.set_facecolor(cmap(color_value))

    plt.xlabel(x_label)
    plt.ylabel("Number of nodes")
    new_title = f"Distribution {title}"
    plt.title(new_title)
    plt.grid(True)
    plt.savefig(f"./plots/statistics/{new_title}-{bins}bins.png", dpi=1200)
    plt.close()
    print("Done.")


def format_double(d):
    return "{:.2f}".format(d)


def components_over_threshold(df: DataFrame, title=""):
    from network_building import create_network, plot_network
    print("Plotting number of components over the threshold values...")
    components = []
    thresholds = np.arange(0, 1.0, 0.05)
    for threshold in thresholds:
        graph = create_network(df, similarity_threshold=threshold, no_logs=True,
                               name=f"{title}-{format_double(threshold)}")
        n = number_connected_components(graph)
        # plot_network(graph)
        components.append(n)

    plt.figure()
    plt.plot(thresholds, components, marker="o", color="mediumorchid")
    for i, _ in enumerate(components):
        print(f"({thresholds[i]}, {components[i]}")
        plt.text(thresholds[i], components[i], f"({thresholds[i]:.2f}, {components[i]})", fontsize=5, ha="center")
    plt.xlabel("Threshold")
    plt.ylabel("Number of components")
    plt.title("Number of connected components vs. Threshold")
    plt.grid(True)
    plt.savefig(f"./plots/statistics/{title}.png", dpi=1200)
    print("Done.")

def states_changing_heat_map(data):
    # Step 2: Extract the second component of the vector labels (adapter component)
    num_rounds = len(data)
    num_nodes = len(data[0][1])

    # Create an array to store the adapter component (second element of the vector label)
    adapter_components = np.zeros((num_rounds, num_nodes))

    # Populate the array with the second component of the vector labels
    for i, (run, vectors) in enumerate(data):
        adapter_components[i] = [vector[1] for vector in vectors]

    # Step 3: Define bins for the vector label (e.g., [0, 0.1, ..., 1.0])
    bins = np.linspace(0, 1, 11)  # 10 bins for vector labels between 0 and 1
    bin_labels = (bins[:-1] + bins[1:]) / 2  # Midpoints of the bins

    # Step 4: Create a 2D array to hold counts for the heatmap
    heatmap_data = np.zeros((num_rounds, len(bins) - 1))

    # Step 5: Populate heatmap data by binning the adapter components
    for i in range(num_rounds):
        heatmap_data[i], _ = np.histogram(adapter_components[i], bins=bins)

    # Step 6: Plot the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, xticklabels=np.round(bin_labels, 2),
                yticklabels=np.arange(1, num_rounds + 1), annot=True)
    plt.xlabel('Adapter Component (Binned)')
    plt.ylabel('Simulation Round')
    plt.title('Heatmap of Adapter Components Over Time')
    plt.show()
