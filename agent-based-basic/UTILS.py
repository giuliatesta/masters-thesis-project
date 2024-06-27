import pickle
import os
from conf import LABELS
from preprocessing import load_dataset_csv

STATE = "_LPStates"
# / + "log_trial_"
BASE = os.sep + "trial_"

"""
Creates a directory if not exists
"""


def createIfNotExist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


"""
Create a file named dir/_log_trial/{id}_LPStates.pickled
"""


def makeFileNameLPState(dir, id):
    return dir + BASE + str(id) + STATE + labels_chain() + ".pickled"


def labels_chain():
    labels = load_dataset_csv("./work/LABELS", index=False)
    chain = ""
    for label in labels:
        chain += "_" + label
    return chain

"""
Store LPStateTuples in a file identified by trial_id
"""


def storeAllToFile(LPStatesTuples, directory, trial_id):
    print(LPStatesTuples)
    file_path = makeFileNameLPState(directory, str(trial_id))
    storeToFile(LPStatesTuples, file_path)
    read_pickled_file(file_path)


"""
Store one item (e.g. state list or networkx graph) in a specific pickle file.
"""


def storeToFile(stuff, filename, verbose=True):
    filename = os.path.normcase(filename)
    directory = os.path.dirname(filename)
    createIfNotExist(directory)

    f = open(filename, 'wb')
    pickle.dump(stuff, f)
    f.close()

    if verbose:
        total = len(stuff)
        print("Written %i items to pickled binary file: %s" % (total, filename))
    return filename


def read_pickled_file(file_path):
    # Open the file in binary read mode
    with open(file_path, 'rb') as file:
        # Load the data from the file
        data = pickle.load(file)

    # Now you can use the data
    for d in data:
        print(d)
