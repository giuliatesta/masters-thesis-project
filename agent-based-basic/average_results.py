import os
import numpy as np

import utils


def average_vector_labels(results_directory, results_file_name, ):
    to_be_averaged = []
    for filename in sorted(os.listdir(results_directory)):
        if filename.startswith("trial_") and not filename.endswith("STATES.pickled"):
            file_name = os.path.join(results_directory, filename)
            data = UTILS.read_pickled_file(file_name)
            to_be_averaged.append(data)

    # accumulates the vectors for each time step
    accumulated = {}
    for run in to_be_averaged:
        for iteration in run:
            time_step = iteration[0]
            vector_labels = iteration[1]
            if time_step not in accumulated:
                accumulated[time_step] = []
            if len(accumulated[time_step]) == 0:
                accumulated[time_step] = np.array(vector_labels)
            else:
                accumulated[time_step] = np.add(accumulated[time_step], vector_labels)

    # averages the accumulated vectors and adds the result to the final avg results array
    averages = []
    for time_step, acc in accumulated.items():
        averaged_vls = acc / len(to_be_averaged)
        averages.append([time_step, averaged_vls.tolist()])

    # stores the averaged results in a pickled file called avg_results
    path = results_directory + "/" + results_file_name
    UTILS.store_to_file(list(averages), path)
    print(f"---- AVERAGED RESULTS: {results_file_name}----")
    UTILS.print_pickled_file(path)
    return averages


def average_state_results(results_directory, results_file_name, ):
    to_be_averaged = []
    # reads the results file in the directory and stored them in the results array
    for filename in sorted(os.listdir(results_directory)):
        if filename.endswith("STATES.pickled"):
            file_name = os.path.join(results_directory, filename)
            data = UTILS.read_pickled_file(file_name)
            to_be_averaged.append(data)

    # accumulates the vectors for each time step
    accumulated = {}
    for run in to_be_averaged:
        for iteration in run:
            time_step = iteration[0]
            states = iteration[1]
            if time_step not in accumulated:
                accumulated[time_step] = []
            if len(accumulated[time_step]) == 0:
                accumulated[time_step] = np.array(states)
            else:
                accumulated[time_step] = np.add(accumulated[time_step], states)

    # checks the sign of the accumalated value: if positive, then ADAPTER (+1); if negative, NON ADAPTER (-1)
    averages = []
    for time_step, acc in accumulated.items():
        averaged_states = acc / len(to_be_averaged)
        averages.append([time_step, (np.where(averaged_states >= 0, 1, -1)).tolist()])

    path = results_directory + "/" + results_file_name
    UTILS.store_to_file(list(averages), path)
    print(f"---- AVERAGED STATES: {results_file_name}----")
    UTILS.print_pickled_file(path)
    return averages

