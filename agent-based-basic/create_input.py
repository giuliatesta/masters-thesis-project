import pandas as pd

from network_building import create_network
from preprocessing import load_dataset_csv, filter_by


def create_input(data, LABELS):

    data.reset_index(drop=True, inplace=True)
    # data = filter_by(data, "Country", "Italy",  number_of_rows=100)
    data = data.head(1000)
    labels = pd.DataFrame(LABELS)
    labels.to_csv(path_or_buf="work/LABELS", index=False, header=False, sep=";")
    print(labels)
    initial_vls = pd.DataFrame(data[LABELS])
    initial_vls.to_csv(path_or_buf="work/INITIAL_VLS", index=False, header=False, sep=";")

    graph = create_network(data, similarity_threshold=0.7, name="Travel Survey gower similarity network")
    edges = pd.DataFrame(graph.edges)
    edges.to_csv(path_or_buf="work/EDGES", index=False, header=False, sep=" ")
