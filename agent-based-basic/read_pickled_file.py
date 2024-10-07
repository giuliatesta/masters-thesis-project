import plots
import utils

# utils.print_pickled_file(f"./work/results/trial_simulations/trial_0_LPStates_L0_L1_0_RUN_1.pickled", 10)

data = utils.read_pickled_file(
    "/Users/giuliatesta/PycharmProjects/masters-thesis-project/agent-based-basic/work/results/simulation_03_BASE_LINE/trial_0_LPStates_L0_L1_0_RUN_0.pickled")
plots.states_changing_heat_map(data)
