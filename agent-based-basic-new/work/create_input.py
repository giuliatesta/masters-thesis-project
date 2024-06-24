import pandas as pd

from network_building import create_network
from preprocessing import load_dataset_csv, filter_by

# to create the INITIAL_VLS file
data = load_dataset_csv("../../dataset/EU_travel_survey_demand_innovative_transport_systems.csv")
data = filter_by(data, "Country", "Italy",  number_of_rows=1000)
labels = pd.DataFrame(["Age"])
labels.to_csv(path_or_buf="./LABELS", index=False, header=False, sep=";")
data["Age"].to_csv(path_or_buf="./INITIAL_VLS", index=False, header=False, sep=";")

graph = create_network(data, similarity_threshold=0.7, name="Travel Survey gower similarity network")
edges = pd.DataFrame(graph.edges)
edges.to_csv(path_or_buf="./EDGES", index=False, header=False, sep=" ")