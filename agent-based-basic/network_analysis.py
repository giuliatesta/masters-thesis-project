import networkx as nx
from matplotlib import pyplot as plt
import seaborn as sns

#Network analysis for LPA Network
# Number ofn nodes: 1000
# Number of edges: 243727
# Density of the network: 0.48794194194194196
# Average clustering: 0.615123305197661
# Diameter: 3
# Average shortest path: 1.5120940940940941
class NetworkAnalysis:

    def __init__(self, graph):
        self.graph = graph

    def analyse(self):
        print(f"Network analysis for {self.graph.name}")
        print(f"Number of nodes: {self.graph.number_of_nodes()}")
        print(f"Number of edges: {self.graph.number_of_edges()}")
        print(f"Density of the network: {nx.density(self.graph)}")
        print("------------------------------------------------")
        print(f"Diameter: {nx.diameter(self.graph)}")
        print(f"Average shortest path: {nx.average_shortest_path_length(self.graph)}")
        print("------------------------------------------------")
        print("Degree distribution plot")
        degrees = [degree for node, degree in self.graph.degree()]
        plt.figure()
        plt.hist(degrees, bins=range(min(degrees), max(degrees) + 2), density=True, alpha=0.5, color='mediumorchid',
                 edgecolor='mediumorchid')
        plt.title("Degree Distribution")
        plt.xlabel("Degree")
        plt.ylabel("Density")
        plt.grid(True)
        path = "/Users/giuliatesta/PycharmProjects/masters-thesis-project/agent-based-basic/network_analysis/degrees/degree_distribution.png"
        plt.savefig(path, dpi=1200)
        print(f"saved in {path}")
        print("------------------------------------------------")
        # print(f"Degree centrality: {nx.degree_centrality(self.graph)}")
        plt.figure()
        plt.subplot(2, 2, 1)
        sns.kdeplot(nx.degree_centrality(self.graph), fill=True, legend=True, color="mediumorchid")
        plt.grid()
        plt.title(f'KDE Plot of degree centrality')
        path = "/Users/giuliatesta/PycharmProjects/masters-thesis-project/agent-based-basic/network_analysis/centralities/network_degree_centrality.png"
        #plt.savefig(path, dpi=1200)
        print(f"saved in {path}")

        # print(f"Betweenness centrality: {nx.betweenness_centrality(self.graph)}")
        plt.subplot(2, 2, 2)
        sns.kdeplot(nx.betweenness_centrality(self.graph), fill=True, legend=True, color="mediumorchid")
        plt.grid()
        plt.title(f'KDE Plot of betweenness centrality')
        path = "/Users/giuliatesta/PycharmProjects/masters-thesis-project/agent-based-basic/network_analysis/centralities/network_betweenness_centrality.png"
        # plt.savefig(path, dpi=1200)
        print(f"saved in {path}")

        # print(f"Closeness centrality: {nx.closeness_centrality(self.graph)}")
        plt.subplot(2, 2, 3)
        sns.kdeplot(nx.closeness_centrality(self.graph), fill=True, legend=True, color="mediumorchid")
        plt.grid()
        plt.title(f'KDE Plot of closeness centrality')
        path = "/Users/giuliatesta/PycharmProjects/masters-thesis-project/agent-based-basic/network_analysis/centralities/network_closeness_centrality.png"
        #plt.savefig(path, dpi=1200)
        print(f"saved in {path}")

        # print(f"Eigenvector centrality: {nx.eigenvector_centrality(self.graph)}")
        plt.subplot(2,2, 4)
        sns.kdeplot(nx.eigenvector_centrality(self.graph), fill=True, legend=True, color="mediumorchid")
        plt.grid()
        plt.title(f'KDE Plot of eigenvector centrality')
        plt.tight_layout()
        path = "/Users/giuliatesta/PycharmProjects/masters-thesis-project/agent-based-basic/network_analysis/centralities/network_eigenvector_centrality.png"
        plt.savefig("/Users/giuliatesta/PycharmProjects/masters-thesis-project/agent-based-basic/network_analysis/centralities/centralities.png", dpi=1200)
        print(f"saved in {path}")

        plt.figure()
        degree_distribution = [degree for _, degree in self.graph.degree()]
        sns.kdeplot(degree_distribution, fill=True, legend=True, color="mediumorchid")
        plt.grid()
        plt.title(f'KDE Plot of Network degree distribution')
        path = "/Users/giuliatesta/PycharmProjects/masters-thesis-project/plots/statistics/network_degree_distribution.png"
        plt.savefig(path, dpi=1200)
        print(f"saved in {path}")

# Network analysis for LPA Network
# Number of nodes: 1000
# Number of edges: 243727
# Density of the network: 0.48794194194194196
# ------------------------------------------------
# Diameter: 3
# Average shortest path: 1.5120940940940941