import json
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from create_input import create_input_files
from preprocessing import load_dataset_csv
from main_LPA import run_simulations
from conf import RESULTS_DIR
#from after_simluation_plots import states_changing_heat_map, plot_multiple_adapters_by_time
import utils
import scipy.stats as st

from average_results import state_averaging
#
# def can_predict_would_subscribe_attribute_nodes():
#     data = load_dataset_csv("../dataset/df_DNA_sharingEU.csv", index=False)
#     create_input_files(data, [
#         "sha_ind_norm",
#         "Gender",
#         "Education",
#         "Income_level",
#         "Age",
#         "Would_subscribe_car_sharing_if_available_new"],
#                        similarity_threshold=0.6,
#                        initialisation="would-subscribe-attribute",
#                        perc_of_adapters=5
#                        )
#     run_simulations(0, "no-bias", "./work/case-scenarios/experiments")
#
#
# def draw_network_animation():
#     data = load_dataset_csv("../dataset/df_DNA_sharingEU.csv", index=False)
#     create_input_files(data, [
#         "sha_ind_norm",
#         "Gender",
#         "Education",
#         "Income_level",
#         "Age",
#         "Would_subscribe_car_sharing_if_available_new"],
#                        similarity_threshold=0.6,
#                        initialisation="would-subscribe-attribute",
#                        perc_of_adapters=5
#                        )
#     run_simulations(0, "no-bias", RESULTS_DIR)
#
#
# def draw_heat_map():
#     prefix = "./work/case-scenarios/COMPLEX_CONTAGION/BETA-DISTRIBUTION/OPEN-SOCIETY/CONFIRMATION-BIAS/CC8-2/"
#     states_changing_heat_map(prefix,)

def compute_plot_data(data, index):
    # we need the slope, the stationaty value, the confidence Interval
    df = pd.DataFrame(columns=["Index", "Sim", "Slope", "Std-Err", "Max-Val", "Stationary-Val", "Confidence-Int"])
    i = 0
    for label, points in data.items():
        x = list(points.keys())
        y = list(points.values())
        res = st.linregress(x, y)
        ci1, ci2 = st.t.interval(0.95, len(y) - 1, loc=np.mean(y), scale=st.sem(y))
        df.loc[i] = {
            "Index": index,
            "Sim": label,
            "Slope": f"{90 - res.slope: .2f}",
            "Std-Err": f"{res.stderr: .2f}",
            "Max-Val": max(y),
            "Stationary-Val": y[-1],
            "Confidence-Int": f"[{ci1: .0f}, {ci2: .0f}]"
        }
        i += 1
    return df

def massive_plot_data_computation(folder):
    subfolders= [f.path for f in os.scandir(folder) if f.is_dir()]
    input = {}
    input_keys = ["RI-5%", "RI-20%", "RI-40%", "SII-5%", "SII-20%", "SII-40%", "WSI"]
    print(f"ciao: {subfolders}")
    for bias_subfolder in sorted(list(subfolders)):
        if not bias_subfolder.endswith(".old"):
            print(f"HERE: {bias_subfolder}")
            bias_subfolder_name = bias_subfolder.split("/")[-1].upper()
            sim_subfolders = [f.path for f in os.scandir(bias_subfolder) if f.is_dir()]
            print(f"{sim_subfolders}")
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
                print("!!")

    print(input)
    df = pd.DataFrame()

    for key, vals in input.items():
        df = pd.concat([df, compute_plot_data(vals, index=key)], ignore_index=True)
    print(f"Saving in {folder}")
    df.to_csv(folder+"/input.csv")


massive_plot_data_computation("./work/case-scenarios/COMPLEX_CONTAGION/BETA-DISTRIBUTION/RIGID-SOCIETY")