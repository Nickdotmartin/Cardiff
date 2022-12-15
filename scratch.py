import random

import pandas as pd
import numpy as np
#
# # sep_values = '0, 6'
# # sep_split = sep_values.split(',')
# # print(f'sep_split: {sep_split}')
#
# # sep_list = [int(i) for i in sep_split]
# # sep_list = [int(i) for i in sep_values.split(',')]
# # print(f"sep_list: {type(sep_list)}: {sep_list}")
# # sep_values = [int(i) for i in expInfo['sep_values'].split(',')]
# # print(f"sep_list: {type(sep_list)}: {sep_list}")
# sep_values = [0, 2, 4, 6]  # select from [0, 1, 2, 3, 6, 18, 99]
# # sep_values = [0]  # select from [0, 1, 2, 3, 6, 18, 99]
# print(f'sep_values: {sep_values}')
# ISI_values = [0, 2, 4, 6]  # select from [-1, 0, 2, 4, 6, 9, 12, 24]
# # ISI_values = [-1, 0, 1, 2]  # select from [-1, 0, 2, 4, 6, 9, 12, 24]
# print(f'ISI_values: {ISI_values}')
# # repeat separation values for each ISI e.g., [0, 0, 6, 6]
# sep_vals_list = list(np.repeat(sep_values, len(ISI_values)))
# print(f'sep_vals_list: {sep_vals_list}')
# # ISI_vals_list cycles through ISIs e.g., [-1, 6, -1, 6]
# ISI_vals_list = list(np.tile(ISI_values, len(sep_values)))
# print(f'ISI_vals_list: {ISI_vals_list}')
# # stair_names_list joins sep_vals_list and ISI_vals_list
# # e.g., ['sep0_ISI-1', 'sep0_ISI6', 'sep6_ISI-1', 'sep6_ISI6']
# stair_names_list = [f'sep{s}_ISI{c}' for s, c in zip(sep_vals_list, ISI_vals_list)]
# print(f'stair_names_list: {stair_names_list}')
# n_stairs = len(sep_vals_list)
# print(f'n_stairs: {n_stairs}')
# stair_idx = list(range(n_stairs))
# print(f'stair_idx: {stair_idx}')
# startVal = [63.6] * n_stairs
# print(f'startVal: {startVal}')
#
# df = pd.DataFrame({'stair_idx': stair_idx, 'label': stair_names_list,
#                    'separation': sep_vals_list, 'ISI': ISI_vals_list,
#                    'startVal': startVal})
# print(df)
# df.to_csv('conds_list')

