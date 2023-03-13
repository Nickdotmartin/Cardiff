# import random
# from operator import itemgetter
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# import os
# from exp1a_psignifit_analysis import make_diff_from_conc_df

neg_sep_list = [-20.0, -18.0, -6.0, -3.0, -2.0, -1.0, -0.01, 0.0, 1.0, 2.0, 3.0, 6.0, 18.0, 20.0]
# neg_sep_list = [20.0, 0.0, 1.0, 2.0, 3.0, 6.0, 18.0, -20.0, -0.01, -1.0, -2.0, -3.0, -6.0, -18.0]

srtd_below_zero = sorted([i for i in neg_sep_list if i < 0])
srtd_above_zero = sorted([i for i in neg_sep_list if i > 0])
print(f"srtd_below_zero: {srtd_below_zero}")
print(f"srtd_above_zero: {srtd_above_zero}")

if 20 in srtd_above_zero:
    srtd_above_zero.remove(20)
    srtd_above_zero = [20] + srtd_above_zero
print(f"srtd_above_zero: {srtd_above_zero}")


if -20 in srtd_below_zero:
    srtd_below_zero.remove(-20)
    srtd_below_zero = srtd_below_zero + [-20]
print(f"srtd_below_zero: {srtd_below_zero}")

neg_sep_list = srtd_below_zero + srtd_above_zero
print(f"neg_sep_list: {neg_sep_list}")



