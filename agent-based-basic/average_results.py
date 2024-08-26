import os
import numpy as np

import UTILS


def average_vls_results(results_directory, results_file_name, ):
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
    for run in results:
        for time_step, vectors in run:
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
        # print(f"average vector {time_step}: {average_vectors}")
        avg_results.append([time_step, average_vectors.tolist()])

    # stores the averaged results in a pickled file called avg_results
    avg_file_path = results_directory + "/" + results_file_name
    UTILS.store_to_file(list(avg_results), avg_file_path)
    # print(f"---- AVERAGED RESULTS: {results_file_name}----")
    # UTILS.print_pickled_file(avg_file_path)
    return avg_results


def average_state_results(results_directory, results_file_name, ):
    results = []
    avg_results = []
    # reads the results file in the directory and stored them in the results array
    for filename in sorted(os.listdir(results_directory)):
        if filename.endswith("STATES.pickled"):
            file_name = os.path.join(results_directory, filename)
            data = UTILS.read_pickled_file(file_name)
            results.append(data)

    # accumulates the vectors for each time step
    accumulated = {}
    for run in results:
        for time_step, vectors in run:
            if time_step not in accumulated:
                accumulated[time_step] = []
            if len(accumulated[time_step]) == 0:
                accumulated[time_step] = np.array(vectors)
            else:
                accumulated[time_step] = np.add(accumulated[time_step], vectors)

    # checks the sign of the accumalated value: if positive, then ADAPTER (+1); if negative, NON ADAPTER (-1)
    for time_step in sorted(accumulated.keys()):
        summed_vectors = accumulated[time_step]
        # signs = np.sign(summed_vectors)
        avgs = summed_vectors / len(results)

        # determine final states: +1 if the average is >= 0, else -1
        signs = np.where(avgs >= 0, 1, -1)
        print(f"{summed_vectors} ->\n {avgs},\n{signs}")
        avg_results.append([time_step, signs.tolist()])

    # stores the averaged results in a pickled file called avg_results
    avg_file_path = results_directory + "/" + results_file_name
    UTILS.store_to_file(list(avg_results), avg_file_path)
    print(f"---- AVERAGED STATES: {results_file_name}----")
    UTILS.print_pickled_file(avg_file_path)
    return avg_results
