import json
import os

import pandas as pd
from matplotlib import pyplot as plt

from create_input import create_input_files
from preprocessing import load_dataset_csv
from main_LPA import run_simulations
from conf import RESULTS_DIR
from after_simluation_plots import states_changing_heat_map, compute_plot_data, plot_multiple_adapters_by_time
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
    prefix = "./work/case-scenarios/COMPLEX_CONTAGION/BETA-DISTRIBUTION/OPEN-SOCIETY/CONFIRMATION-BIAS/CC8-2/"
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

    for key, vals in input.items():
        df = pd.concat([df, compute_plot_data(vals, index=key)], ignore_index=True)
    print(f"Saving in {folder}")
    df.to_csv(folder+"/input.csv")

no_bias_labels = ["5% RI", "20% RI", "40% RI", "5% SII", "20% SII", "40% SII", "IWS"]
# def massive_plots_creation(folder, sim_name):
#     subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]
#     for bias_subfolder in sorted(list(subfolders)):
#         if not bias_subfolder.endswith(".old"):
#             bias_subfolder_name = bias_subfolder.split("/")[-1].upper()
#             sim_subfolders = [f.path for f in os.scandir(bias_subfolder) if f.is_dir() ]
#             if bias_subfolder_name == "NO-BIAS":
#                 plots =_create_input({}, sim_subfolders, bias_subfolder_name)
#                 title= sim_name + "without Bias"
#                 plot_multiple_adapters_by_time(plots, folder, title, use_markers=False)
#             else:
<<<<<<< HEAD
=======
#                 plots={"RI":{}, "SII":{}}
#                 no_bias_sims = _create_input({}, [f.path for f in os.scandir(folder+"/NO-BIAS") if f.is_dir()], bias_subfolder_name)
#
>>>>>>> origin/master


def _create_input(input, subfolder, sim_name):
    for sim in sorted(list(subfolder)):
        states = state_averaging(sim, run_count=5)
        sim_id = sim.split("-")[-1]
        input_key = no_bias_labels[int(sim_id) - 1]
        if not input.get(input_key):
            input[input_key] = {}
        if not input[input_key].get(sim_name):
            input[input_key][sim_name] = states
    return input

<<<<<<< HEAD
massive_plot_data_computation("./work/case-scenarios/COMPLEX_CONTAGION/BETA-DISTRIBUTION/OPEN-SOCIETY")
=======
#massive_plot_data_computation("./work/case-scenarios/COMPLEX_CONTAGION/BETA-DISTRIBUTION/OPEN-SOCIETY")
draw_heat_map()
>>>>>>> origin/master
