import networkx as nx
from matplotlib import pyplot as plt
import seaborn as sns

class NetworkAnalysis:

    def __init__(self, graph):
        self.graph = graph

    def analyse(self):
        print(f"Network analysis for {self.graph.name}")
        print(f"Number of nodes: {self.graph.number_of_nodes()}")
        print(f"Number of edges: {self.graph.number_of_edges()}")
        print(f"Density of the network: {nx.density(self.graph)}")
        print("------------------------------------------------")
        print(f"Degree centrality: {nx.degree_centrality(self.graph)}")
        print(f"Degree centrality")

        plt.figure()
        plt.subplot(2, 2, 1)
        sns.kdeplot(nx.degree_centrality(self.graph), fill=True, legend=True, color="mediumorchid")
        plt.grid()
        plt.title(f'KDE Plot of degree centrality')
        path = "/Users/giuliatesta/PycharmProjects/masters-thesis-project/plots/statistics/network_degree_centrality.png"
        #plt.savefig(path, dpi=1200)
        print(f"saved in {path}")
        # print(f"Betweenness centrality: {nx.betweenness_centrality(self.graph)}")
        plt.subplot(2, 2, 2)
        sns.kdeplot(nx.betweenness_centrality(self.graph), fill=True, legend=True, color="mediumorchid")
        plt.grid()
        plt.title(f'KDE Plot of betweenness centrality')
        path = "/Users/giuliatesta/PycharmProjects/masters-thesis-project/plots/statistics/network_betweenness_centrality.png"
        # plt.savefig(path, dpi=1200)
        print(f"saved in {path}")
        # print(f"Closeness centrality: {nx.closeness_centrality(self.graph)}")
        plt.subplot(2, 2, 3)
        sns.kdeplot(nx.closeness_centrality(self.graph), fill=True, legend=True, color="mediumorchid")
        plt.grid()
        plt.title(f'KDE Plot of closeness centrality')
        path = "/Users/giuliatesta/PycharmProjects/masters-thesis-project/plots/statistics/network_closeness_centrality.png"
        #plt.savefig(path, dpi=1200)
        print(f"saved in {path}")
        # print(f"Eigenvector centrality: {nx.eigenvector_centrality(self.graph)}")
        plt.subplot(2,2, 4)
        sns.kdeplot(nx.eigenvector_centrality(self.graph), fill=True, legend=True, color="mediumorchid")
        plt.grid()
        plt.title(f'KDE Plot of eigenvector centrality')
        plt.tight_layout()
        path = "/Users/giuliatesta/PycharmProjects/masters-thesis-project/plots/statistics/network_eigenvector_centrality.png"
        plt.savefig("/Users/giuliatesta/PycharmProjects/masters-thesis-project/plots/statistics/centralities.png", dpi=1200)
        print(f"saved in {path}")
        # print("------------------------------------------------")
        print(f"Average clustering: {nx.average_clustering(self.graph)}")
        # print(f"Connected components: {nx.connected_components(self.graph)}")
        print("------------------------------------------------")
        print(f"Diameter: {nx.diameter(self.graph)}")
        print(f"Average shortest path: {nx.average_shortest_path_length(self.graph)}")
        print(f"Shortest path: {nx.shortest_path_length(self.graph)}")

        print(f"Degree distribution")
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