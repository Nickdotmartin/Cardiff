import os
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from exp1a_psignifit_analysis import split_df_alternate_rows
import os
from statsmodels.stats.weightstats import DescrStatsW

'''
To get weighted means
Load all threshold df
load confidence intervals df.

for each condition - get the 
1. 12 thr
2. 12 CIs

array = np.array([[1, 0, 2],
                  [1, 1, 1]])
weights = np.array([[2, 1, 1],
                    [1, 1, 2]])
print(np.average(array, weights=weights))

can I do it all in 1 step with np.arrays for thr and CIs? (rather than by condition?)
yes - as long as arrays are in same format

'''

# # # # # # # #
# # get a master CI_width (or eta) df with all values and same structure as MASTER_psignifit_thresholds
# conf_type = 'CI_width'
conf_type = 'eta'

verbose = True
old_exp_path = r'C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\Kim_split_runs_weighted_mean'
exp_path = os.path.normpath(old_exp_path)
participant_name = 'Kim'
root_path = os.path.join(exp_path, participant_name)
run_folder_names = ['Kim_1', 'Kim_2', 'Kim_3', 'Kim_4', 'Kim_5', 'Kim_6']

isi_name_list = ['Concurrent', 'ISI0', 'ISI2', 'ISI4',
                     'ISI6', 'ISI9', 'ISI12', 'ISI24']

isi_vals_list = [-1, 0, 2, 4, 6, 9, 12, 24]

sep_list = [0, 1, 2, 3, 6, 18, 20]

if conf_type == 'CI_width':
    conf_csv_name = 'psignifit_CI_width'
elif conf_type == 'eta':
    conf_csv_name = 'psignifit_eta'

print(f"conf_type: {conf_type}, {conf_csv_name}")

# # # # # #
all_conf_list = []
for run_idx, run_name in enumerate(run_folder_names):


    # this_conf_df = pd.read_csv(f'{root_path}{os.sep}{run_name}{os.sep}psignifit_CI_width.csv')
    this_conf_df = pd.read_csv(os.path.join(root_path, run_name, conf_csv_name))
    this_conf_df = pd.read_csv(os.path.join(root_path, run_name, f'{conf_csv_name}.csv'))

    if verbose:
        print(f'{run_idx}. {run_name} - this_conf_df:\n{this_conf_df}')

    if 'Unnamed: 0' in list(this_conf_df):
        this_conf_df.drop('Unnamed: 0', axis=1, inplace=True)

    this_conf_df.drop(columns='stair', inplace=True)

    # split df into group1 and group2
    psig_g1_df = this_conf_df[this_conf_df['group'] == 1]
    psig_g1_df.drop(columns='group', inplace=True)
    rows, cols = psig_g1_df.shape
    psig_g1_df.insert(0, 'stack', [run_idx*2] * rows)

    psig_g2_df = this_conf_df[this_conf_df['group'] == 2]
    psig_g2_df.drop(columns='group', inplace=True)
    psig_g2_df.insert(0, 'stack', [run_idx*2+1] * rows)

    columns_list = ['stack', 'separation'] + isi_name_list
    psig_g1_df.columns = columns_list
    psig_g2_df.columns = columns_list

    if verbose:
        print(f'\npsig_g1_df:\n{psig_g1_df}')
        print(f'\npsig_g2_df:\n{psig_g2_df}')

    all_conf_list.append(psig_g1_df)
    all_conf_list.append(psig_g2_df)

# join all stacks (run/group) data and save as master csv
all_conf_df = pd.concat(all_conf_list, ignore_index=True)
all_conf_df.to_csv(os.path.join(root_path, f'MASTER_{conf_csv_name}.csv'), index=False)
if verbose:
    print(f'\nall_conf_df:\n{all_conf_df}')


# # # # # # # #

# # # part 2 # loop through conditions and trim means

thr_df = pd.read_csv(os.path.join(root_path, f'MASTER_psignifit_thresholds.csv'))
conf_df = pd.read_csv(os.path.join(root_path, f'MASTER_{conf_csv_name}.csv'))
print(f'\nthr_df:\n{thr_df}')
print(f'\nconf_df:\n{conf_df}')



weighted_mean_array = np.zeros(shape=[len(sep_list), len(isi_name_list)])
weighted_std_array = np.zeros(shape=[len(sep_list), len(isi_name_list)])
weighted_se_array = np.zeros(shape=[len(sep_list), len(isi_name_list)])

for isi_idx, isi_name in enumerate(isi_name_list):

    isi_thr_df = thr_df.loc[:, ['separation', isi_name]]
    isi_conf_df = conf_df.loc[:, ['separation', isi_name]]

    print(f'isi_thr_df:\n{isi_thr_df}')
    print(f'isi_conf_df:\n{isi_conf_df}')

    for sep_idx, sep in enumerate(sep_list):
        # sep_thr_df = isi_thr_df[isi_thr_df['separation'] == sep]
        # sep_conf_df = isi_conf_df[isi_conf_df['separation'] == sep]

        sep_thr_df = isi_thr_df.loc[isi_thr_df['separation'] == sep]
        sep_conf_df = isi_conf_df.loc[isi_conf_df['separation'] == sep]

        print(f'\nsep_thr_df (sep={sep}, isi_name={isi_name}):\n{sep_thr_df}')
        print(f'sep_conf_df:\n{sep_conf_df}')

        thr_array = np.array(sep_thr_df[isi_name])
        conf_array = np.array(sep_conf_df[isi_name])

        print(f'thr_array:\n{thr_array}')
        print(f'conf_array:\n{conf_array}')

        # # original version using numpy
        # weighted_mean = np.average(thr_array, weights=conf_array)
        # weighted_mean_array[sep_idx, isi_idx] = weighted_mean
        #
        # weighted_std = np.sqrt(np.cov(thr_array, aweights=conf_array))
        # weighted_std_array[sep_idx, isi_idx] = weighted_std
        #
        # weighted_se = weighted_std / np.sqrt(sum(conf_array) - 1)
        # weighted_se_array[sep_idx, isi_idx] = weighted_se

        # # new version using statsmodels
        '''
        from: https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy.  also, see:
        https://www.statsmodels.org/dev/generated/statsmodels.stats.weightstats.DescrStatsW.html?highlight=descrstatsw#statsmodels.stats.weightstats.DescrStatsW
        '''
        weighted_stats = DescrStatsW(thr_array, weights=conf_array, ddof=0)
        weighted_mean_array[sep_idx, isi_idx] = weighted_stats.mean
        weighted_std_array[sep_idx, isi_idx] = weighted_stats.std
        weighted_se_array[sep_idx, isi_idx] = weighted_stats.std_mean  # is this really SE or weighted std?



        # todo: looks like I can use 2d arrays and specify axis. Check with 1d first,  but then change if poss.

# make dataframe from array
weighted_mean_df = pd.DataFrame(weighted_mean_array, columns=isi_name_list)
weighted_mean_df.insert(0, 'separation', sep_list)

weighted_std_df = pd.DataFrame(weighted_std_array, columns=isi_name_list)
weighted_std_df.insert(0, 'separation', sep_list)

weighted_se_df = pd.DataFrame(weighted_se_array, columns=isi_name_list)
weighted_se_df.insert(0, 'separation', sep_list)

if conf_type == 'CI_width':
    MASTER_df_name = 'MASTER_WM_CI_thr.csv'
    MASTER_std_df_name = 'MASTER_WM_CI_std.csv'
    MASTER_se_df_name = 'MASTER_WM_CI_se.csv'

elif conf_type == 'eta':
    MASTER_df_name = 'MASTER_WM_eta_thr.csv'
    MASTER_std_df_name = 'MASTER_WM_eta_std.csv'
    MASTER_se_df_name = 'MASTER_WM_eta_se.csv'

weighted_mean_df.to_csv(os.path.join(root_path, MASTER_df_name), index=False)
weighted_std_df.to_csv(os.path.join(root_path, MASTER_std_df_name), index=False)
weighted_se_df.to_csv(os.path.join(root_path, MASTER_se_df_name), index=False)

if verbose:
    print(f'\nweighted_mean_df:\n{weighted_mean_df}')
    print(f'\nweighted_std_df:\n{weighted_std_df}')
    print(f'\nweighted_se_df:\n{weighted_se_df}')


