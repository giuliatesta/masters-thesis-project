import math
import os

import utils


# counts the number of adapters in the state results for each simulation index
def count_adapters(states):
    # [time_step, [values for each node]] = [0, [-1, -1, 1, -1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1]]
    adapters = {}
    for state in states:
        acc = 0
        for val in state[1]:
            if val == 1:
                acc += 1
        adapters[state[0]] = acc
    return adapters


# averages the final state results given the multiple RUNS over the same simulation
# multiple runs are necessary to reduce the impact of the random selection of initial adapters
def state_averaging(state_files_path):
    state_files_content = []
    for filename in sorted(os.listdir(state_files_path)):
        if filename.endswith("STATES.pickled"):
            data = utils.read_pickled_file(os.path.join(state_files_path, filename))
            state_files_content.append(data)

    adapters = [count_adapters(states) for states in state_files_content]
    print(adapters)
    print(f"LEN: {len(adapters)}")
    averages = {i: 0 for i in adapters[0].keys()}
    run_count = len(adapters)
    for a in adapters:
        for run, count in a.items():
            averages[run] += count

    averages = {run: math.ceil(acc / run_count) for run, acc in averages.items()}
    print(f"Averaging states in {state_files_path}")
    print(averages)
    return averages

