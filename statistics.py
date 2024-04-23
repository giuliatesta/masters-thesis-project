from matplotlib import pyplot as plt


def gower_matrix_distribution(matrix):
    print("Plotting the gower's matrix distribution...")
    plt.figure()
    plt.hist(matrix, bins=20)
    plt.xlabel("Gower's matrix values")
    plt.ylabel('Number of nodes')
    plt.title("Distribution of Gower's Matrix Values")
    plt.grid(True)
    plt.savefig("./plots/statistics/gower_matrix_distribution.png")
    print("Done.")