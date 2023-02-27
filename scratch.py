import random
from operator import itemgetter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from exp1a_psignifit_analysis import trim_n_high_n_low

trim_n = 2
root_path = '/Users/nickmartin/Library/CloudStorage/OneDrive-CardiffUniversity/PycharmProjects/Cardiff/EXP1_sep4_5/Nick'
get_means_name = 'MASTER_TM2_thresholds.csv'
get_means_path = os.path.join(root_path, get_means_name)
get_means_df = pd.read_csv(get_means_path)
error_type = 'se'
verbose = True


"""Part 2: trim highest and lowest values is required and get average vals and errors"""
# # # trim highest and lowest values
# if trim_n is not None:
#     trimmed_df = trim_n_high_n_low(all_data_psignifit_df, trim_from_ends=trim_n,
#                                    reference_col='separation',
#                                    stack_col_id='stack',
#                                    verbose=verbose)
#     trimmed_df.to_csv(f'{root_path}{os.sep}MASTER_TM{trim_n}_thresholds.csv', index=False)
#
#     get_means_df = trimmed_df
# else:
#     get_means_df = all_data_psignifit_df

# # get means and errors
groupby_sep_df = get_means_df.drop('stack', axis=1)
ave_psignifit_thr_df = groupby_sep_df.groupby('separation', sort=True).mean()
if verbose:
    print(f'\nave_psignifit_thr_df:\n{ave_psignifit_thr_df}')

if error_type in [False, None]:
    error_bars_df = None
elif error_type.lower() in ['se', 'error', 'std-error', 'standard error', 'standard_error']:
    error_bars_df = groupby_sep_df.groupby('separation', sort=True).sem()
elif error_type.lower() in ['sd', 'stdev', 'std_dev', 'std.dev', 'deviation', 'standard_deviation']:
    error_bars_df = groupby_sep_df.groupby('separation', sort=True).std()
else:
    raise ValueError(f"error_type should be in:\nfor none: [False, None]\n"
                     f"for standard error: ['se', 'error', 'std-error', 'standard error', 'standrad_error']\n"
                     f"for standard deviation: ['sd', 'stdev', 'std_dev', 'std.dev', "
                     f"'deviation', 'standard_deviation']")
if verbose:
    print(f'\nerror_bars_df: ({error_type})\n{error_bars_df}')

# save csv with average values
if trim_n is not None:
    ave_psignifit_thr_df.to_csv(f'{root_path}{os.sep}MASTER_ave_TM{trim_n}_thresh.csv')
    error_bars_df.to_csv(f'{root_path}{os.sep}MASTER_ave_TM{trim_n}_thr_error_{error_type}.csv')
else:
    ave_psignifit_thr_df.to_csv(f'{root_path}{os.sep}MASTER_ave_thresh.csv')
    error_bars_df.to_csv(f'{root_path}{os.sep}MASTER_ave_thr_error_{error_type}.csv')

