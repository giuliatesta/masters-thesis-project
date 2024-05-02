from network_building import create_network, plot_network
from network_simulation import simulate
from preprocessing import load_dataset_csv, filter_by, remove_column, remove_all_except_for
from statistics import components_over_threshold

# a threshold is required to evaluate whether create the edge
# based on the gower's distance of two nodes
SIMILARITY_THRESHOLD = 0.75

# loads the dataset and removes empty values
data = load_dataset_csv("./dataset/EU_travel_survey_demand_innovative_transport_systems.csv")
# filters by the column Country = Italy
data = filter_by(data, "Country", "Italy", number_of_rows=1000)
# saved the filtered data in the filtered.csv file
data.to_csv(path_or_buf="./dataset/filtered.csv")
# complete_data = data
# for column in data.columns:
#     removed_data = remove_all_except_for(data,[column])
#     graph = create_network(removed_data, similarity_threshold=SIMILARITY_THRESHOLD, name=f"Network with only {column}")
#     plot_network(graph)
#     data = complete_data

# components_over_threshold(data)
graph = create_network(data, similarity_threshold=SIMILARITY_THRESHOLD, name="Travel Survey Gower's similarity network")
plot_network(graph)
#simulate(graph)
