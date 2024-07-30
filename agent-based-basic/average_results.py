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
    print(f"---- AVERAGED RESULTS: {results_file_name}----")
    UTILS.print_pickled_file(avg_file_path)
    return avg_results


def get_final_state_from_vls(averaged_vls, results_file_name):
    def determine_state(vl):
        first = vl[0]
        second = vl[1]
        if first >= second:
            # if the non adapter is greater than adapter -> it becomes non adapter
            return 1
        elif first < second:
            print(f"first:{first}, second: {second}")
            # if the adapter is greater than non adapter -> it becomes adapter
            return -1

    states = [[i[0], [1 for _ in range(len(i[1]))]] for i in averaged_vls]
    print(f"states: {states}")
    for i, run in enumerate(averaged_vls):
        time_step = run[0]
        vectors = run[1]
        for vector in vectors:
            # states[time_step] = []
            states[i][1].append(determine_state(vector))      # TODO append state given the vectors
        print(f"avg_results[time_step]: {states[i]}")

    #
    # # averages the accumulated vectors and adds the result to the final avg results array
    # # print(f"accumulated: {accumulated}")
    # for time_step in sorted(accumulated.keys()):
    #     average_vectors = accumulated[time_step] / len(results)
    #     # print(f"average vector {time_step}: {average_vectors}")
    #     avg_results.append([time_step, average_vectors.tolist()])
    #
    # # stores the averaged results in a pickled file called avg_results
    # avg_file_path = results_directory + "/" + results_file_name
    UTILS.store_to_file(list(states), results_file_name)
    print(f"---- AVERAGED RESULTS: {results_file_name}----")
    print(states)
    # # print_pickled_file(avg_file_path)
    # return avg_results
