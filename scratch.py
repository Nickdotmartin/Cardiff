import random
from operator import itemgetter
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
#
max_droped_fr_trials = 10

trial_nums = [25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400, 600, 800, 1000]

for total_n_trials in trial_nums:

    # expected trials plus repeats
    max_trials = total_n_trials + max_droped_fr_trials

    # limit on number of trials without a break
    max_without_break = 120

    n_breaks = max_trials // max_without_break

    if n_breaks > 0:
        take_break = int(max_trials / (n_breaks + 1))
    else:
        take_break = max_without_break

    print(f"take_break: {take_break}, n_breaks: {n_breaks}")
    print(f"max_trials: {max_trials}, take_break: {take_break}, n_breaks: {n_breaks}")

# take_break = 6
# for actual_trials_inc_rpt in list(range(20)):
#     print(actual_trials_inc_rpt)
#     if (actual_trials_inc_rpt % take_break == 1) & (actual_trials_inc_rpt > 1):
#         print('break\n')
