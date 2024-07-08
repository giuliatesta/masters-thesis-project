import csv
import sys

### Useful global variables ###

# Simulation parameters
# Max simulation time(number of iteration of the algorithm)
ITERATION_NUM = 5
# The number of simulations
TRIALS = 1

try:
    # Set files path and graph type
    LABELS_NAMES_FILE = sys.argv[1]
    LABELS_INIT_VALUES_FILE = sys.argv[2]
    EDGES_FILE = sys.argv[3]
    ATTRIBUTES_FILE = sys.argv[4]
    GRAPH_TYPE = sys.argv[5]
    RESULTS_DIR = sys.argv[6]
    STATE_CHANGING_METHOD = int(sys.argv[7])
except IndexError:
    print("Command line arguments not provided correctly")
    sys.exit(1)

# Creates the list of unique labels
with open(LABELS_NAMES_FILE, 'r') as f:
    csv_reader = csv.reader(f, delimiter=';')
    LABELS = [x.lstrip() for x in list(csv_reader)[0]]

