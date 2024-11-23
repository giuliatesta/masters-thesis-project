from create_input import create_input_files
from preprocessing import load_dataset_csv
from main_LPA import run_simulations


def can_predict_would_subscribe_attribute_nodes():
    data = load_dataset_csv("../dataset/df_DNA_sharingEU.csv", index=False)
    create_input_files(data, [
        "sha_ind_norm",
        "Gender",
        "Education",
        "Income_level",
        "Age",
        "Would_subscribe_car_sharing_if_available_new"],
                       similarity_threshold=0.6,
                       initialisation="would-subscribe-attribute",
                       perc_of_adapters=5
                       )
    run_simulations(0, "no-bias", "./work/case-scenarios/experiments")


def draw_network_animation():
    data = load_dataset_csv("../dataset/df_DNA_sharingEU.csv", index=False)
    create_input_files(data, [
        "sha_ind_norm",
        "Gender",
        "Education",
        "Income_level",
        "Age",
        "Would_subscribe_car_sharing_if_available_new"],
                       similarity_threshold=0.6,
                       initialisation="would-subscribe-attribute",
                       perc_of_adapters=5
                       )

can_predict_would_subscribe_attribute_nodes()
