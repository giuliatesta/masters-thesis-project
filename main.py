from preprocessing import load_dataset_csv, filter_by

data = load_dataset_csv("./dataset/EU_travel_survey_demand_innovative_transport_systems.csv")
data = filter_by(data, "Country", "Italy")
data.to_csv(path_or_buf="./dataset/filtered.csv")