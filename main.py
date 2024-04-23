from network_building import create_network, plot_network
from network_simulation import simulate
from preprocessing import load_dataset_csv, filter_by

# loads the dataset and removes empty values
data = load_dataset_csv("./dataset/EU_travel_survey_demand_innovative_transport_systems.csv")
# filters by the column Country = Italy
data = filter_by(data, "Country", "Italy", number_of_rows=1000)
# saved the filtered data in the filtered.csv file
data.to_csv(path_or_buf="./dataset/filtered.csv")
graph = create_network(data, "Travel Survey Gower's similarity network")
plot_network(graph)
#simulate(graph)
