import utils
from simulation_result_plotter import states_changing_heat_map

vector_labels = utils.read_pickled_file(
    "/Users/giuliatesta/PycharmProjects/masters-thesis-project/agent-based-basic/work/results/simulation_03_BASE_LINE/trial_0_LPStates_L0_L1_0_RUN_0.pickled")
states = utils.read_pickled_file("/Users/giuliatesta/PycharmProjects/masters-thesis-project/agent-based-basic/work/results/simulation_03_BASE_LINE/trial_0_LPStates_0_RUN_0_STATES.pickled")
states_changing_heat_map(
    states=states,
    vector_labels=vector_labels,
    step= 0,
    title="State changing Heat Map (SIM 3 - BASE LINE)",
    path= "/Users/giuliatesta/PycharmProjects/masters-thesis-project/agent-based-basic/work/results/simulation_03_BASE_LINE")
