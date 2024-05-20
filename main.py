from networkx import number_connected_components

from network_building import create_network, plot_network, create_networks_using_one_column
from preprocessing import load_dataset_csv, filter_by
from statistics import components_over_threshold

# a threshold is required to evaluate whether create the edge
# based on the gower's distance of two nodes
SIMILARITY_THRESHOLD = 0.7

# loads the dataset and removes empty values
data = load_dataset_csv("./dataset/EU_travel_survey_demand_innovative_transport_systems.csv")
# filters by the column Country = Italy
data = filter_by(data, "Country", "Italy",  number_of_rows=1000)
# saved the filtered data in the filtered.csv file
data.to_csv(path_or_buf="./dataset/filtered.csv")

# for creating the plot regarding the number of connected components with respect to the threshold
components_over_threshold(data, title="gower_components_over_threshold")

#graph = create_network(data, similarity_threshold=SIMILARITY_THRESHOLD, name="Travel Survey gower similarity network")
#plot_network(graph)
