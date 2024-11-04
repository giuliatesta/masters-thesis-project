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
        # plt.subplot(2, 2, 1)
        sns.kdeplot(nx.degree_centrality(self.graph), fill=True, legend=True, color="mediumorchid")
        plt.grid()
        plt.title(f'KDE Plot of degree centrality')
        path = "/Users/giuliatesta/PycharmProjects/masters-thesis-project/agent-based-basic/network_analysis/centralities/network_degree_centrality.png"
        plt.savefig(path, dpi=1200)
        print(f"saved in {path}")

        # print(f"Betweenness centrality: {nx.betweenness_centrality(self.graph)}")
        # plt.subplot(2, 2, 2)
        sns.kdeplot(nx.betweenness_centrality(self.graph), fill=True, legend=True, color="mediumorchid")
        plt.grid()
        plt.title(f'KDE Plot of betweenness centrality')
        path = "/Users/giuliatesta/PycharmProjects/masters-thesis-project/agent-based-basic/network_analysis/centralities/network_betweenness_centrality.png"
        plt.savefig(path, dpi=1200)
        print(f"saved in {path}")

        # print(f"Closeness centrality: {nx.closeness_centrality(self.graph)}")
        #plt.subplot(2, 2, 3)
        sns.kdeplot(nx.closeness_centrality(self.graph), fill=True, legend=True, color="mediumorchid")
        plt.grid()
        plt.title(f'KDE Plot of closeness centrality')
        path = "/Users/giuliatesta/PycharmProjects/masters-thesis-project/agent-based-basic/network_analysis/centralities/network_closeness_centrality.png"
        plt.savefig(path, dpi=1200)
        print(f"saved in {path}")

        # print(f"Eigenvector centrality: {nx.eigenvector_centrality(self.graph)}")
       # plt.subplot(2,2, 4)
        sns.kdeplot(nx.eigenvector_centrality(self.graph), fill=True, legend=True, color="mediumorchid")
        plt.grid()
        plt.title(f'KDE Plot of eigenvector centrality')
        plt.tight_layout()
        path = "/Users/giuliatesta/PycharmProjects/masters-thesis-project/agent-based-basic/network_analysis/centralities/network_eigenvector_centrality.png"
        plt.savefig(path, dpi=1200)
        #plt.savefig("/Users/giuliatesta/PycharmProjects/masters-thesis-project/agent-based-basic/network_analysis/centralities/centralities.png", dpi=1200)
        print(f"saved in {path}")