import pickle
import os

import numpy as np

from conf import LABELS, STATE_CHANGING_METHOD

STATE = "_LPStates"
# / + "log_trial_"
BASE = os.sep + "trial_"

"""
Creates a directory if not exists
"""


def create_if_not_exist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


"""
Create a file named dir/_log_trial/{id}_LPStates.pickled
"""


def make_file_name_vector_labels(dir, id, run):
    return dir + BASE + str(id) + STATE + labels_chain() + "_" + str(STATE_CHANGING_METHOD) + "_RUN_" + str(
        run) + ".pickled"


def make_file_name_states(dir, id, run):
    return dir + BASE + str(id) + STATE + "_" + str(STATE_CHANGING_METHOD) + "_RUN_" + str(run) + "_STATES.pickled"


def labels_chain():
    chain = ""
    for label in LABELS:
        chain += "_" + label
    return chain


"""
Store LPStateTuples in a file identified by trial_id
"""


def store_all_to_file(LPStatesTuples, LPRaws, directory, trial_id, run_index):
    trial_id = str(trial_id)

    file_path = make_file_name_vector_labels(directory, trial_id, run_index)
    store_to_file(LPStatesTuples, file_path)
    print_pickled_file(file_path)

    raw_file_path = make_file_name_states(directory, trial_id, run_index)
    store_to_file(LPRaws, raw_file_path)
    print_pickled_file(raw_file_path)


"""
Store one item (e.g. state list or networkx graph) in a specific pickle file.
"""


def store_to_file(content, filename, verbose=True):
    filename = os.path.normcase(filename)
    directory = os.path.dirname(filename)
    create_if_not_exist(directory)

    f = open(filename, 'wb')
    pickle.dump(content, f)
    f.close()

    if verbose:
        total = len(content)
        print("Written %i items to pickled binary file: %s" % (total, filename))
    return filename


def print_pickled_file(file_path):
    data = read_pickled_file(file_path)
    # Now you can use the data
    for d in data:
        print(d)


def read_pickled_file(file_path):
    # Open the file in binary read mode
    with open(file_path, 'rb') as file:
        # Load the data from the file
        data = pickle.load(file)
    return data

