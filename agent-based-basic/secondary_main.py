from matplotlib import pyplot as plt

from create_input import create_input_files
from preprocessing import load_dataset_csv
from main_LPA import run_simulations
from conf import RESULTS_DIR
from after_simluation_plots import states_changing_heat_map
import utils


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
    run_simulations(0, "no-bias", RESULTS_DIR)


def draw_heat_map():
    prefix = "./work/case-scenarios/SIMPLE_CONTAGION/NO-BIAS/SC0-1"
    vector_labels = utils.read_pickled_file(f"{prefix}/trial_0_LPStates_L0_L1__RUN_0.pickled")
    states = utils.read_pickled_file(f"{prefix}/trial_0_LPStates__RUN_0_STATES.pickled")

    plt.figure(figsize=(16, 12))
    i = 1
    for time_step in range(0, 6, 1):
        plt.subplot(4, 2, i)
        i += 1
        print(f"Time step: {time_step}, subplot: {i}")
        states_changing_heat_map(
            states=states,
            vector_labels=vector_labels,
            step=time_step)

        plt.suptitle("Heat maps for the first 6 steps of the simulations\n", x=0.57, fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{prefix}/heat_map.png")

