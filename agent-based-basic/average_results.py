import os
import numpy as np

import UTILS


def average_vls_results(results_directory, results_file_name):
    results = []
    avg_results = []
    # reads the results file in the directory and stored them in the results array
    for filename in sorted(os.listdir(results_directory)):
        if filename.startswith("trial_") and not filename.endswith("STATES.pickled"):
            file_name = os.path.join(results_directory, filename)
            data = UTILS.read_pickled_file(file_name)
            results.append(data)

    # accumulates the vectors for each time step
    accumulated = {}
    for runs in results:
        for run in runs:
            time_step = run[0]
            vectors = run[1]
            if time_step not in accumulated:
                accumulated[time_step] = []
            if len(accumulated[time_step]) == 0:
                accumulated[time_step] = np.array(vectors)
            else:
                accumulated[time_step] = np.add(accumulated[time_step], vectors)

    # averages the accumulated vectors and adds the result to the final avg results array
    # print(f"accumulated: {accumulated}")
    for time_step in sorted(accumulated.keys()):
        average_vectors = accumulated[time_step] / len(results)
        # print(f"average vector {time_step}: {accumulated[time_step]}/ { len(results)} = {average_vectors}")
        avg_results.append([time_step, average_vectors.tolist()])

    # stores the averaged results in a pickled file called avg_results
    avg_file_path = results_directory + "/" + results_file_name
    # UTILS.store_to_file(list(avg_results), avg_file_path)
    # print(f"---- AVERAGED RESULTS: {results_file_name}----")
    # UTILS.print_pickled_file(avg_file_path, 2)
    return avg_results


def calculate_states_from_averaged_vls(vls, results_directory, results_file_name):
    def determine_state(vl):
        return -1 if vl[0] > vl[1] else 1

    states = [[run[0], [determine_state(vectors) for vectors in run[1]]] for run in vls]
    avg_file_path = results_directory + "/" + results_file_name
    UTILS.store_to_file(list(states), avg_file_path)
    print(f"---- AVERAGED STATES: {results_file_name}----")
    UTILS.print_pickled_file(avg_file_path)
    return states
