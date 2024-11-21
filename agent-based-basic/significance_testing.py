import numpy as np
from scipy.stats import ttest_ind

ALPHA = 0.05


def welch_testing(first_group, second_group):
    stat, p_value = ttest_ind(first_group, second_group, equal_var=False)
    print(stat)
    print("---")
    print(p_value)
    if p_value < ALPHA:
        print("Reject the null hypothesis: the means are significantly different.")
    else:
        print("Fail to reject the null hypothesis: the means are not significantly different.")


# SIMPLE CONTAGION without BIAS
sim1 = {1: 25, 2: 25, 3: 764, 4: 1000, 5: 1000, 6: 1000, 7: 1000, 8: 1000, 9: 1000, 10: 1000, 11: 1000, 12: 1000,
        13: 1000, 14: 1000, 15: 1000, 16: 1000, 17: 1000, 18: 1000, 19: 1000, 20: 1000, 21: 1000, 22: 1000, 23: 1000,
        24: 1000, 25: 1000, 26: 1000, 27: 1000, 28: 1000, 29: 1000, 30: 1000}
# SIMPLE CONTAGION with CONFIRMATION BIAS
sim2 = {1: 50, 2: 50, 3: 825, 4: 825, 5: 825, 6: 825, 7: 825, 8: 825, 9: 825, 10: 825, 11: 825, 12: 825, 13: 825,
        14: 825, 15: 825, 16: 825, 17: 825, 18: 825, 19: 825, 20: 825, 21: 825, 22: 825, 23: 825, 24: 825, 25: 825,
        26: 825, 27: 825, 28: 825, 29: 825, 30: 825}
print(list(sim1.values()))

welch_testing(list(sim1.values()), list(sim2.values()))