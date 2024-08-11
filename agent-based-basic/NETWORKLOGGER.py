import networkx as nx
from SimPy import Simulation as Sim
import UTILS
from conf import RESULTS_DIR, LABELS
import time

# The iterations whose VLs have to be stored
tt = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]


class NetworkLogger(Sim.Process):

    def __init__(self, sim, logging_interval):
        Sim.Process.__init__(self, sim=sim)
        self.sim = sim
        self.interval = logging_interval
        self.LPNet = nx.Graph()
        self.LPVLTuples = []
        self.LPStates = []

    def Run(self):
        i = 0
        start_time = time.time()
        while True:
            self.log_current_state()
            print("--- %i iterations completed in %fs ---" % (i, (time.time() - start_time)))
            i += 1
            yield Sim.hold, self, self.interval

    def log_current_state(self):
        LPNodes = sorted(self.sim.LPNet.nodes(data=True), key=lambda x: x[0])

        # Actual VL belonging coefficients
        VLs = [[float(node[1]["agent"].VL[str(i)]) for i in LABELS] for node in LPNodes]
        states = [node[1]["agent"].state for node in LPNodes]

        # Add actual VL value to logs
        if self.sim.now() in tt:
            self.LPVLTuples.append([self.sim.now(), VLs])
            self.LPStates.append([self.sim.now(), states])

    def log_trial_to_files(self, id, run_index):
        UTILS.store_all_to_file(self.LPVLTuples, self.LPStates, RESULTS_DIR, id, run_index)
