RUNS = 5 #30
SIMILARITY_THRESHOLD = 0.60
ALPHA = 2
BETA = 2
USE_SHARING_INDEX = False
INITIALISATION = "adapters-with-SI"
all_choices = {
    0: "same-weights",  # trivial cases (SIM-*0)
    1: "beta-dist",  # baseline
    2: "over-confidence",  # OL and OP are fixed: OP = 0.8 (confidence in my opinion) (EX0-1)
    3: "over-influenced",  # OL and OP are fixed: OL = 0.8 (too easily influenced by others) (EX0-2)
    4: "extreme-influenced",  # OL and OP are fixed: OP=0.02 and OL=0.98 (EX0-3),
    5: "simple-contagion",   # becomes adopter if at least one neighbours is
    6: "majority",   # becomes adopters if the majority of the neighbours is
    7: "reweigh-only-on-adapters"
}
STATE_CHANGING_METHOD = all_choices[7]
INITIAL_ADAPTERS_PERC = 20
