# import random
# from operator import itemgetter
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# import os
# from exp1a_psignifit_analysis import make_diff_from_conc_df

separations = [2]  # , 4, 6, 8]

sep_dir = 'sep'
for sep in separations:
    sep_dir = sep_dir + f'_{sep}'

print(sep_dir)

# sep_conds = sep_dir.split('_')[1:]

sep_conds = [int(i) for i in sep_dir.split('_')[1:]]

print(sep_conds)

for i in sep_conds:
    print(i)