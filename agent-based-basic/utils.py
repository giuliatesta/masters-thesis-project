import pickle
import os

from conf import LABELS, RESULTS_DIR

STATE = "_LPStates"
# / + "log_trial_"
BASE = os.sep + "trial_"


# creates a directory if not exists
def create_if_not_exist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def make_file_name_vector_labels(path, run_number, run):
    return path + BASE + str(run_number) + STATE + labels_chain() + "_" + "_RUN_" + str(
        run) + ".pickled"


def make_file_name_states(path, run_number, run):
    return path + BASE + str(run_number) + STATE + "_" + "_RUN_" + str(run) + "_STATES.pickled"


def labels_chain():
    chain = ""
    for label in LABELS:
        chain += "_" + label
    return chain


def store_all_to_file(LPVLs, LPStates, directory, run_id, run_index):
    run_id = str(run_id)

    file_path = make_file_name_vector_labels(directory, run_id, run_index)
    store_to_file(LPVLs, file_path)
    # print_pickled_file(file_path)

    raw_file_path = make_file_name_states(directory, run_id, run_index)
    store_to_file(LPStates, raw_file_path)
    # print_pickled_file(raw_file_path)


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


def print_pickled_file(file_path, index=-1):
    data = read_pickled_file(file_path)
    for i, d in enumerate(data):
        print(d)
        if i == index:
            break


def read_pickled_file(file_path):
    # open the file in binary read mode
    with open(file_path, 'rb') as file:
        # load the data from the file
        data = pickle.load(file)
    return data


def write_simulation_readme_file(dir, vl_update, initialisation, adapters_perc, similarity_threshold, cognitive_bias,
                                 states, alpha, beta):
    splits = dir.split("/")
    simulation_type = splits[4]

    with open(dir + "/readme.txt", "w") as file:
        file.write(f"{simulation_type}\n\n")
        file.write(f"VL_UPDATE_METHOD: \"{vl_update}\"\n")
        if vl_update == "beta-dist":
            file.write(f"OP: scaled similarity weights\nOL: beta(alpha = {alpha}, beta = {beta})\n")
            if alpha == 2 and beta == 2:
                file.write("(quasi-normal distribution for beta)")
            if alpha == 2 and beta == 5:
                file.write("(society with rigid agents)")
            if alpha == 5 and beta == 2:
                file.write("(society with open-to-change agents)")
        file.write(f"INITIALISATION: \"{initialisation}\"\n")
        file.write(f"INITIAL_ADAPTERS_PERC: {adapters_perc}%\n")
        file.write(f"APPLY_COGNITIVE_BIAS: \"{cognitive_bias}\"\n")
        file.write(f"SIMILARITY THRESHOLD = {similarity_threshold}\n")
        file.write("RUNS = 5\nITERATION_NUM = 30\nN. EDGES: 42761\nN. NODES: 1000\n\nNO GENDER, AGE, EDUCATION biases\n")
        file.write(f"{states}\n")

        file.close()
