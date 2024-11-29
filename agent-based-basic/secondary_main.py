import json
import os

import pandas as pd
from matplotlib import pyplot as plt

from create_input import create_input_files
from preprocessing import load_dataset_csv
from main_LPA import run_simulations
from conf import RESULTS_DIR
from after_simluation_plots import states_changing_heat_map, compute_plot_data
import utils

from average_results import state_averaging

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
    prefix = "./work/case-scenarios/COMPLEX_CONTAGION/BETA-DISTRIBUTION/OPEN-SOCIETY/COGNITIVE-BIAS/CONFIRMATION-BIAS/CC8-2"
    states_changing_heat_map(prefix,)

def massive_plot_data_computation(folder):
    subfolders= [f.path for f in os.scandir(folder) if f.is_dir()]
    input = {}
    input_keys = ["RI-5%", "RI-20%", "RI-40%", "SII-5%", "SII-20%", "SII-40%", "WSI"]

    for bias_subfolder in sorted(list(subfolders)):
        if not bias_subfolder.endswith(".old"):
            bias_subfolder_name = bias_subfolder.split("/")[-1].upper()
            sim_subfolders = [f.path for f in os.scandir(bias_subfolder) if f.is_dir()]
            for sim in sorted(list(sim_subfolders)):
                states = state_averaging(sim, run_count=5)
                sim_id = sim.split("-")[-1]
                input_key = input_keys[int(sim_id)-1]
                if not input.get(input_key):
                    input[input_key] = {}
                if not input[input_key].get(bias_subfolder_name):
                    input[input_key][bias_subfolder_name] = states
                else:
                    return KeyError

    print(input)
    df = pd.DataFrame()

    for vals in input.values():
        df = pd.concat([df, compute_plot_data(vals)], ignore_index=True)
    df.to_csv(folder+"/input.csv")

massive_plot_data_computation("./work/case-scenarios/COMPLEX_CONTAGION/BETA-DISTRIBUTION/OPEN-SOCIETY")