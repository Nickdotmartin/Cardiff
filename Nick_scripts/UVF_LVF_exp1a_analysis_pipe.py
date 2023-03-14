import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from exp1a_psignifit_analysis import a_data_extraction, b3_plot_staircase, c_plots, \
    d_average_participant, e_average_exp_data, make_average_plots, make_long_df, \
    plot_w_errors_no_1probe
from rad_flow_psignifit_analysis import plot_runs_ave_w_errors
from psignifit_tools import get_psignifit_threshold_df, get_psig_thr_w_hue
from python_tools import which_path, running_on_laptop, switch_path

'''
This script if for checking for any differences between thr upper visual field and lower visual field.
It will use the already analysied RUNDATA_sorted.xlsx to do this.
Loop through the participant run folders and append each RUNDATA-sorted.xlsx, with addition 'run' column.
Save to P_all_runs_master_output.csv.

Then run psignifit on this
'''

# # loop through run folders with first 4 scripts (a, get_psignifit_threshold_df, b3, c)
# # then run script d to get master lists and averages
# exp_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/exp1a_data'
# participant_list = ['aa', 'bb', 'cc', 'dd', 'ee']
# exp_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/EXP1_sep4_5'
# exp_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\EXP1_sep4_5"
# exp_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\Exp3_Ricco_NM_v4"
exp_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\exp1a_data"
convert_path1 = os.path.normpath(exp_path)
if running_on_laptop():
    convert_path1 = switch_path(convert_path1, 'mac_oneDrive')
exp_path = convert_path1

participant_list = ['aa', 'bb', 'cc', 'dd', 'ee', 'Nick']  # ['Kim', 'Kris', 'Simon', 'Nick']
# participant_list = ['Nick']  # ['Kim', 'Kris', 'Simon', 'Nick']
p_idx_plus = 1

n_runs = 12
analyse_from_run = 1
trim_list = []

'''Part 1, get threshold for each participant and make master list'''

exp_thr = []
exp_CI_width = []

# for p_idx, participant_name in enumerate(participant_list):
#
#     root_path = os.path.join(exp_path, participant_name)
#
#     # search to automatically get run_folder_names
#     dir_list = os.listdir(root_path)
#     run_folder_names = []
#     for i in range(n_runs):  # numbers 0 to 11
#         check_dir = f'{participant_name}_{i + analyse_from_run}'  # numbers 1 to 12
#         if check_dir in dir_list:
#             run_folder_names.append(check_dir)
#
#     if len(run_folder_names) > 0:
#         print("running analysis for:")
#         for i in run_folder_names:
#             print(i)
#     else:
#         print("no run folders found")
#
#
#     # add RUNDATA-sorted to all_data
#     all_data = []
#
#     for run_idx, run_dir in enumerate(run_folder_names):
#
#         print(f'\ncompiling analysis for {participant_name}, {run_dir}, {participant_name}_{run_idx+1}\n')
#         save_path = f'{root_path}{os.sep}{run_dir}'
#
#         # don't delete this (participant_name = participant_name),
#         # needed to ensure names go name1, name2, name3 not name1, name12, name123
#         p_name = participant_name
#
#         # '''a'''
#         p_name = f'{participant_name}_{run_idx+1}_output.csv'
#         # p_name = f'{participant_name}{run_idx+1}'
#         # isi_list = [-1, 0, 2, 4, 6, 9, 12, 24]
#
#         if os.path.isfile(os.path.join(save_path, 'RUNDATA-sorted.xlsx')):
#             run_data_path = os.path.join(save_path, 'RUNDATA-sorted.xlsx')
#         elif os.path.isfile(os.path.join(save_path, p_name)):
#             run_data_path = os.path.join(save_path, p_name)
#         elif os.path.isfile(os.path.join(save_path, f'{run_dir}_output.csv')):
#             run_data_path = os.path.join(save_path, f'{run_dir}_output.csv')
#         elif os.path.isfile(os.path.join(save_path, f'{participant_name}_output.csv')):
#             run_data_path = os.path.join(save_path, f'{participant_name}_output.csv')
#         else:
#             raise FileNotFoundError(f'{participant_name}, run_dir {run_dir}')
#
#         # run_data_path = f'{save_path}{os.sep}RUNDATA-sorted.xlsx'
#
#         # run_data_path = os.path.join(save_path, )
#
#         if run_data_path[-4:] == 'xlsx':
#             run_data_df = pd.read_excel(run_data_path, engine='openpyxl',
#                                         # usecols=['ISI',
#                                         #          'stair',
#                                         #          'separation',
#                                         #          # 'group',
#                                         #          'probeLum', 'trial_response', 'corner']
#                                         )
#         else:
#             run_data_df = pd.read_csv(run_data_path)
#         print(f"run_data_df:\n{run_data_df}")
#
#         # add isi column for multi-indexing
#         if 'run' not in list(run_data_df.columns):
#             run_data_df.insert(0, 'run', int(run_idx+1))
#         # if verbose:
#         print(f'run_data_df:\n{run_data_df.head()}')
#
#         # get column names to use on all_data_df
#         column_names = list(run_data_df)
#
#         # add to all_data
#         all_data.append(run_data_df)
#
#     # create all_data_df - reshape to 2d
#     all_data_shape = np.shape(all_data)
#     print(f'all_data_shape:\n{all_data_shape}')
#
#     if len(np.shape(all_data)) == 2:
#         sheets, rows, columns = np.shape(all_data)
#         all_data = np.reshape(all_data, newshape=(sheets * rows, columns))
#         # if verbose:
#         print(f'all_data reshaped from {all_data_shape} to {np.shape(all_data)}')
#         all_data_df = pd.DataFrame(all_data, columns=column_names)
#     else:
#         all_data_df = pd.concat(all_data, ignore_index=True)
#
#     visual_field_list = ['UVF' if i < 200 else 'LVF' for i in all_data_df['corner'].to_list()]
#     all_data_df['vis_field'] = visual_field_list
#     # if verbose:
#     print(f"all_data_df:\n{all_data_df}")
#
#     sep_list = sorted(list(all_data_df['separation'].unique()))
#     print(f"sep_list: {sep_list}")
#
#
#     # # if save_all_data:
#     save_name = 'P_all_runs_master_output.csv'
#     save_csv_path = os.path.join(root_path, save_name)
#     # # if verbose:
#     print(f"\nsaving all_data_df to save_csv_path: {save_csv_path}")
#     all_data_df.to_csv(save_csv_path, index=False)
#
#
#
#     all_data_df = pd.read_csv(os.path.join(root_path, 'P_all_runs_master_output.csv'))
#
#     vis_field_names = ['UVF', 'LVF']
#
#
#     both_vfs_thr = []
#     both_vfs_CI_width = []
#
#     for idx, vis_field_name in enumerate(vis_field_names):
#
#
#         print(f'Running psignifit for {vis_field_name}')
#
#         vis_field_df = all_data_df[all_data_df['vis_field'] == vis_field_name]
#         print(vis_field_df)
#
#         isi_list = sorted(list(vis_field_df['ISI'].unique()))
#         print(f"isi_list: {isi_list}")
#
#         sep_list = sorted(list(vis_field_df['separation'].unique()))
#         print(f"sep_list: {sep_list}")
#
#
#
#
#         '''get psignifit thresholds df - use stairs as sep levels rather than using groups'''
#
#         thr_df = get_psignifit_threshold_df(root_path=exp_path,
#                                             p_run_name=participant_name,
#                                             csv_name=vis_field_df,
#                                             n_bins=9, q_bins=True,
#                                             thr_col='probeLum',
#                                             sep_col='separation', sep_list=sep_list,
#                                             isi_col='ISI', isi_list=isi_list,
#                                             conf_int=True, thr_type='Bayes',
#                                             plot_both_curves=False,
#                                             # cols_to_add_dict=None, save_name=f'psignifit_{vis_field_name}_ISI{ISI}_sep{separation}',
#                                             cols_to_add_dict=None, save_name=f'psignifit_{vis_field_name}',
#                                             show_plots=False, save_plots=False,
#                                             verbose=True)
#
#         # thr_df['vis_field'] = vis_field_name
#         thr_df.insert(1, 'vis_field', vis_field_name)
#
#         cond_list = thr_df['separation'].to_list()
#         if vis_field_name == 'LVF':
#             cond_list = [-.01 if i == 0 else -i for i in cond_list]
#         # thr_df['cond'] = cond_list
#         thr_df.insert(2, 'cond', cond_list)
#
#         print(f'psignifit_{vis_field_name}:\n{thr_df}')
#         column_names = list(thr_df)
#
#         # add this VFs thr and CI width to list to concat with other VF
#         both_vfs_thr.append(thr_df)
#
#         CI_width_filename = f'psignifit_{vis_field_name}_CI_width.csv'
#
#         VF_CI_width_df = pd.read_csv(os.path.join(root_path, CI_width_filename))
#         VF_CI_width_df.insert(1, 'vis_field', vis_field_name)
#         VF_CI_width_df.insert(2, 'cond', cond_list)
#         both_vfs_CI_width.append(VF_CI_width_df)
#
#         # progress_df = pd.concat(both_vfs_thr)
#         # save_name = 'psignifit_progress.csv'
#         # save_csv_path = os.path.join(root_path, save_name)
#         # print(f"\nsaving progress_df to save_csv_path:\n{save_csv_path}")
#         # progress_df.to_csv(save_csv_path, index=False)
#
#
#     # create both_vfs_df - reshape to 2d
#     both_vfs_shape = np.shape(both_vfs_thr)
#     sheets, rows, columns = np.shape(both_vfs_thr)
#     both_vfs_thr = np.reshape(both_vfs_thr, newshape=(sheets * rows, columns))
#     print(f'both_vfs_thr reshaped from {both_vfs_shape} to {np.shape(both_vfs_thr)}')
#     both_vfs_df = pd.DataFrame(both_vfs_thr, columns=column_names)
#     print(f"both_vfs_df:\n{both_vfs_df}")
#
#     save_name = 'psignifit_both_vfs.csv'
#     save_csv_path = os.path.join(root_path, save_name)
#     print(f"\nsaving all_data_df to save_csv_path:\n{save_csv_path}")
#     both_vfs_df.to_csv(save_csv_path, index=False)
#
#     # create both_vfs_CI_width_df - reshape to 2d
#     both_vfs_CI_width_shape = np.shape(both_vfs_CI_width)
#     sheets, rows, columns = np.shape(both_vfs_CI_width)
#     both_vfs_CI_width = np.reshape(both_vfs_CI_width, newshape=(sheets * rows, columns))
#     print(f'both_vfs_thr reshaped from {both_vfs_CI_width_shape} to {np.shape(both_vfs_CI_width)}')
#     both_vfs_CI_width_df = pd.DataFrame(both_vfs_CI_width, columns=column_names)
#     print(f"both_vfs_CI_width_df:\n{both_vfs_CI_width_df}")
#
#     save_name = 'both_vfs_CI_width.csv'
#     save_csv_path = os.path.join(root_path, save_name)
#     print(f"\nsaving both_vfs_CI_width to save_csv_path:\n{save_csv_path}")
#     both_vfs_CI_width_df.to_csv(save_csv_path, index=False)
#
#
#     '''Load psignifit_both_vfs and check columns'''
#     # make plot to show UVF and LVF on one axis
#     psig_both_vf_df = pd.read_csv(os.path.join(root_path, 'psignifit_both_vfs.csv'))
#     print(f"\npsig_both_vf_df:\n{psig_both_vf_df}")
#
#     '''Load both_vfs_CI_width_df and check columns'''
#     # make plot to show UVF and LVF on one axis
#     both_vfs_CI_width_df = pd.read_csv(os.path.join(root_path, 'both_vfs_CI_width.csv'))
#     print(f"\nboth_vfs_CI_width_df:\n{both_vfs_CI_width_df}")
#
#     # change 1probe from 99 to 20
#     both_vf_columns = list(psig_both_vf_df.columns)
#     sep_list = psig_both_vf_df['separation'].to_list()
#     sep_list = [20 if i == 99 else i for i in sep_list]
#     psig_both_vf_df['separation'] = sep_list
#     both_vfs_CI_width_df['separation'] = sep_list
#
#     if 'cond' not in both_vf_columns:
#         print("\nMaking cond column")
#         # add condition list which is equal to sep for uVF or negative sep for LVF (with -.01 instead of -0)
#         sep_list = psig_both_vf_df['separation'].to_list()
#         vf_list = psig_both_vf_df['vis_field'].to_list()
#         cond_list = []
#         for vf, sep in zip(vf_list, sep_list):
#             if vf == 'LVF':
#                 if sep == 0:
#                     this_cond = -.01
#                 else:
#                     this_cond = -sep
#             else:
#                 this_cond = sep
#             print(f"vf: {vf}, sep: {sep}, this_cond: {this_cond}")
#             cond_list.append(this_cond)
#         print(f"cond_list: {cond_list}")
#         psig_both_vf_df.insert(2, 'cond', cond_list)
#         both_vfs_CI_width_df.insert(2, 'cond', cond_list)
#
#
#     # change 1probe from 99 to 20
#     cond_list = psig_both_vf_df['cond'].to_list()
#     cond_list = [20 if i == 99 else i for i in cond_list]
#     cond_list = [-20 if i == -99 else i for i in cond_list]
#     psig_both_vf_df['cond'] = cond_list
#     both_vfs_CI_width_df['cond'] = cond_list
#
#
#     save_name = 'psignifit_both_vfs.csv'
#     save_csv_path = os.path.join(root_path, save_name)
#     print(f"\nsaving all_data_df to save_csv_path:\n{save_csv_path}")
#     psig_both_vf_df.to_csv(save_csv_path, index=False)
#
#     save_name = 'both_vfs_CI_width.csv'
#     save_csv_path = os.path.join(root_path, save_name)
#     print(f"\nsaving both_vfs_CI_width to save_csv_path:\n{save_csv_path}")
#     both_vfs_CI_width_df.to_csv(save_csv_path, index=False)
#
#     # add participant name
#     if 'p_name' not in both_vf_columns:
#         psig_both_vf_df.insert(0, 'p_name', participant_name)
#         both_vfs_CI_width_df.insert(0, 'p_name', participant_name)
#
#     print(f"psig_both_vf_df:\n{psig_both_vf_df}")
#     exp_thr.append(psig_both_vf_df)
#     exp_CI_width.append(both_vfs_CI_width_df)
#
#
#
# # save master dfs
# exp_thr_df = pd.concat(exp_thr)
# save_csv_path = os.path.join(exp_path, 'MASTER_exp_VF_thr.csv')
# exp_thr_df.to_csv(save_csv_path, index=False)
# print(f"exp_thr_df:\n{exp_thr_df}")
#
# # save master dfs
# exp_CI_width_df = pd.concat(exp_CI_width)
# save_csv_path = os.path.join(exp_path, 'MASTER_exp_VF_CI.csv')
# exp_CI_width_df.to_csv(save_csv_path, index=False)
# print(f"exp_CI_width_df:\n{exp_thr_df}")
#
#
#
#
#
#
#
# # # make long form df
# exp_VF_thr_df = pd.read_csv(os.path.join(exp_path, 'MASTER_exp_VF_thr.csv'))
# print(f"\nexp_VF_thr_df:\n{exp_VF_thr_df}")
#
# exp_VF_thr_df.rename({'ISI_-1': 'ISI_99'}, axis=1, inplace=True)
# exp_VF_thr_long_df = pd.wide_to_long(exp_VF_thr_df, stubnames='ISI_',
#                               i=['vis_field', 'separation', 'p_name', 'cond'],
#                               j='ISI',
#                               sep='')
# # exp_VF_thr_long_df.rename({'ISI val': 'ISI', 'ISI_': 'probeLum'}, axis=1, inplace=True)
# exp_VF_thr_long_df.rename({'ISI_': 'probeLum'}, axis=1, inplace=True)
# exp_VF_thr_long_df.reset_index(inplace=True)
# print(f"\nexp_VF_thr_long_df:\n{exp_VF_thr_long_df}")
#
# # make long form CIs
# exp_VF_CI_df = pd.read_csv(os.path.join(exp_path, 'MASTER_exp_VF_CI.csv'))
# print(f"\nexp_VF_CI_df:\n{exp_VF_CI_df}")
#
#
# exp_VF_CI_df.rename({'ISI_-1': 'ISI_99'}, axis=1, inplace=True)
# exp_VF_CI_long_df = pd.wide_to_long(exp_VF_CI_df, stubnames='ISI_',
#                               i=['vis_field', 'separation', 'p_name', 'cond'],
#                               j='ISI',
#                               sep='')
# # exp_VF_CI_long_df.rename({'ISI val': 'ISI', 'ISI_': 'probeLum'}, axis=1, inplace=True)
# exp_VF_CI_long_df.rename({'ISI_': 'CI_width'}, axis=1, inplace=True)
# exp_VF_CI_long_df.reset_index(inplace=True)
# print(f"\nexp_VF_CI_long_df:\n{exp_VF_CI_long_df}")
#
#
# # add cond number column
# cond_vals = sorted(exp_VF_thr_long_df['cond'].unique())
# neg_sep_num_dict = dict(zip(cond_vals, list(range(len(cond_vals)))))
# print(f"\nneg_sep_num_dict: {neg_sep_num_dict}")
#
# exp_VF_thr_long_df.insert(4, 'cond_num', exp_VF_thr_long_df["cond"].map(neg_sep_num_dict))
# exp_VF_CI_long_df.insert(4, 'cond_num', exp_VF_CI_long_df["cond"].map(neg_sep_num_dict))
# print(f"\nexp_VF_thr_long_df:\n{exp_VF_thr_long_df}")
# save_csv_path = os.path.join(exp_path, 'MASTER_exp_VF_thr_long.csv')
# exp_VF_thr_long_df.to_csv(save_csv_path, index=False)
#
# save_csv_path = os.path.join(exp_path, 'MASTER_exp_VF_CI_long.csv')
# exp_VF_CI_long_df.to_csv(save_csv_path, index=False)
#
# print('\nPart 1, get threshold for each participant and make master list: finished\n')





'''Part 2: make plots
for each participant make plots for all data
thresholds (with negative sep)
Difference

Make conditions plots
Just concurrent (ISI_-1)
Just sep (0, 2, 3, 6)

All participants
difference, 
Just concurrent (ISI_-1)
Just sep (0, 2, 3, 6)
'''
exp_VF_thr_long_df = pd.read_csv(os.path.join(exp_path, 'MASTER_exp_VF_thr_long.csv'))
print(f"\nexp_VF_thr_long_df:\n{exp_VF_thr_long_df}")

# get means per condition
groupby_sep_thr_df = exp_VF_thr_long_df.drop('p_name', axis=1)
exp_mean_thr_long_df = groupby_sep_thr_df.groupby(['cond_num', 'ISI'], sort=True).mean()
exp_mean_thr_long_df.reset_index(inplace=True)
print(f"\nexp_mean_thr_long_df:\n{exp_mean_thr_long_df}")


exp_VF_CI_long_df = pd.read_csv(os.path.join(exp_path, 'MASTER_exp_VF_CI_long.csv'))
print(f"\nexp_VF_CI_long_df:\n{exp_VF_CI_long_df}")
groupby_sep_CI_df = exp_VF_CI_long_df.drop('p_name', axis=1)
exp_mean_CI_long_df = groupby_sep_CI_df.groupby(['cond_num', 'ISI'], sort=True).mean()
exp_mean_CI_long_df.reset_index(inplace=True)
# exp_mean_CI_long_df = exp_mean_CI_long_df.CI_width.div(2, fill_value=0),
exp_mean_CI_long_df['halved_CI'] = exp_mean_CI_long_df.CI_width.div(2, fill_value=0)
print(f"\nexp_mean_CI_long_df:\n{exp_mean_CI_long_df}")


exp_mean_thr_long_df['ISI'] = [str(i) for i in exp_mean_thr_long_df['ISI'].to_list()]
exp_mean_CI_long_df['ISI'] = [str(i) for i in exp_mean_CI_long_df['ISI'].to_list()]
print(f"\nexp_mean_thr_long_df:\n{exp_mean_thr_long_df}")



'''Fig 1 - all data'''
print('\nFig 1 - all data')

fig_1_thr_df = exp_mean_thr_long_df.copy()
fig_1_err_df = exp_mean_CI_long_df.copy()

# use wide means df
print(f"exp_mean_thr_long_df:\n{exp_mean_thr_long_df}")
fig_1_thr_df = exp_mean_thr_long_df.pivot(index=['cond_num', 'separation', 'cond'], columns='ISI', values='probeLum')
fig_1_thr_df.reset_index(inplace=True, drop=False)
print(f"fig_1_thr_df:\n{fig_1_thr_df}")

# fig_1_thr_df.index.name = None
isi_col_dict = {'99': 'conc', '0': 'ISI_0', '2': 'ISI_2',
                         '4': 'ISI_4', '6': 'ISI_6', '9': 'ISI_9',
                         '12': 'ISI_12', '24': 'ISI_24'}
fig_1_thr_df.rename(columns=isi_col_dict, inplace=True)
print(f"fig_1_thr_df:\n{fig_1_thr_df}")
fig_1_thr_df = fig_1_thr_df[['cond_num', 'separation', 'cond',
                                     'conc', 'ISI_0', 'ISI_2', 'ISI_4',
                                     'ISI_6', 'ISI_9', 'ISI_12', 'ISI_24']]
fig_1_thr_df.index.name = None
print(f"fig_1_thr_df:\n{fig_1_thr_df}")

print(f"\nexp_mean_CI_long_df:\n{exp_mean_CI_long_df}")
wide_mean_CI_df = exp_mean_CI_long_df.pivot(index=['cond_num', 'separation', 'cond'], columns='ISI', values='halved_CI')
wide_mean_CI_df.reset_index(inplace=True, drop=False)
wide_mean_CI_df.index.name = None
wide_mean_CI_df.rename(columns=isi_col_dict, inplace=True)

wide_mean_CI_df = wide_mean_CI_df[['cond_num', 'separation', 'cond',
                                   'conc', 'ISI_0', 'ISI_2', 'ISI_4',
                                   'ISI_6', 'ISI_9', 'ISI_12', 'ISI_24']]
wide_mean_CI_df.index.name = None
print(f"wide_mean_CI_df:\n{wide_mean_CI_df}")


# add cond number column
if 'cond_num' not in list(fig_1_thr_df.columns):
    cond_vals = fig_1_thr_df['cond'].unique()
    neg_sep_num_dict = dict(zip(cond_vals, list(range(len(cond_vals)))))
    print(f"\nneg_sep_num_dict: {neg_sep_num_dict}")

    fig_1_thr_df.insert(4, 'cond_num', exp_mean_thr_long_df["cond"].map(neg_sep_num_dict))
    wide_mean_CI_df.insert(4, 'cond_num', exp_mean_CI_long_df["cond"].map(neg_sep_num_dict))

wide_mean_thr_w_cond_idx_df = fig_1_thr_df.set_index('cond_num')
wide_mean_thr_w_cond_idx_df.sort_index(inplace=True)
if 'p_name' in list(wide_mean_thr_w_cond_idx_df.columns):
    wide_mean_thr_w_cond_idx_df.drop('p_name', axis=1, inplace=True)
if 'vis_field' in list(wide_mean_thr_w_cond_idx_df.columns):
    wide_mean_thr_w_cond_idx_df.drop('vis_field', axis=1, inplace=True)
if 'cond' in list(wide_mean_thr_w_cond_idx_df.columns):
    wide_mean_thr_w_cond_idx_df.drop('cond', axis=1, inplace=True)
if 'separation' in list(wide_mean_thr_w_cond_idx_df.columns):
    wide_mean_thr_w_cond_idx_df.drop('separation', axis=1, inplace=True)
print(f"exp_thr_w_cond_idx_df:\n{wide_mean_thr_w_cond_idx_df}")

wide_mean_CI_w_cond_idx_df = wide_mean_CI_df.set_index('cond_num')
wide_mean_CI_w_cond_idx_df.sort_index(inplace=True)

x_tick_vals = wide_mean_thr_w_cond_idx_df.index.tolist()
x_tick_labels = sorted(list(exp_mean_thr_long_df['cond'].unique()))
x_tick_labels = ['1pr' if i in [20.0, -20.0] else str(i) for i in x_tick_labels]
x_tick_labels = ['-0' if i == '-0.01' else str(i) for i in x_tick_labels]
x_tick_labels = [i[:-2] if i not in ['1pr', '-0'] else i for i in x_tick_labels]

print(f"x_tick_vals: {x_tick_vals}")
print(f"x_tick_labels: {x_tick_labels}")

if 'p_name' in list(wide_mean_CI_w_cond_idx_df.columns):
    wide_mean_CI_w_cond_idx_df.drop('p_name', axis=1, inplace=True)
if 'vis_field' in list(wide_mean_CI_w_cond_idx_df.columns):
    wide_mean_CI_w_cond_idx_df.drop('vis_field', axis=1, inplace=True)
if 'cond' in list(wide_mean_CI_w_cond_idx_df.columns):
    wide_mean_CI_w_cond_idx_df.drop('cond', axis=1, inplace=True)
if 'separation' in list(wide_mean_CI_w_cond_idx_df.columns):
    wide_mean_CI_w_cond_idx_df.drop('separation', axis=1, inplace=True)
print(f"exp_CI_w_cond_idx_df:\n{wide_mean_CI_w_cond_idx_df}")


isi_name_list = [i for i in list(wide_mean_thr_w_cond_idx_df.columns) if 'ISI_' in i]

fig_1a_title = 'all data: compare UVF & LVF\n(Errors are mean of participant CIs, per ISI)'

plot_runs_ave_w_errors(fig_df=wide_mean_thr_w_cond_idx_df, error_df=wide_mean_CI_w_cond_idx_df,
                       jitter=.1, error_caps=True, alt_colours=False,
                       legend_names=isi_name_list,
                       x_tick_vals=x_tick_vals,
                       x_tick_labels=x_tick_labels,
                       x_axis_label='Sep in diag pixels. Neg=LVF, Pos=UVF',
                       even_spaced_x=True, fixed_y_range=False,
                       fig_title=fig_1a_title, save_name='all_data_VFs.png',
                       save_path=exp_path, verbose=True)
ax = plt.gca() # to get the axis
ax.axvline(x=(x_tick_vals[-1]/2), linestyle="-.", color='lightgrey')  # add dotted line at zero

plt.show()
plt.close()



'''Fig 2, difference between UVF and LVF'''
'''Plot shoing difference in VF for each ISI'''
print(f"\nplot diff between VFs for each ISI")
# for each separation value, subtract LFV from UVF for difference score.

get_diff_df = exp_VF_thr_long_df.copy()
print(f"get_diff_df ({get_diff_df.shape}):\n{get_diff_df}")

LVF_df = get_diff_df.loc[get_diff_df['cond_num'] < 7]
cond_num_list = LVF_df['cond_num'].tolist()
ISI_val_list = LVF_df.pop('ISI').tolist()
p_name_list = LVF_df.pop('p_name').tolist()
LVF_df = LVF_df.drop(['cond', 'vis_field'], axis=1)
LVF_df.set_index('separation', inplace=True)


UVF_df = get_diff_df.loc[get_diff_df['cond_num'] >= 7]
UVF_df = UVF_df.drop(['cond', 'ISI', 'vis_field', 'p_name'], axis=1)
UVF_df.set_index('separation', inplace=True)
print(f"LVF_df ({LVF_df.shape}):\n{LVF_df}")
print(f"UVF_df ({UVF_df.shape}):\n{UVF_df}")

# plot difference.
diff_df = UVF_df.subtract(LVF_df, fill_value=0)
print(f"diff_df ({diff_df.shape}):\n{diff_df}")

diff_df['cond_num'] = cond_num_list
if 'ISI' not in list(diff_df.columns):
    diff_df.insert(1, 'ISI', ISI_val_list)
diff_df = diff_df.rename(columns={'probeLum': 'thr_diff'})

pos_sep_vals = sorted(diff_df.index.unique())
diff_df.reset_index(inplace=True)
print(f"diff_df ({diff_df.shape}):\n{diff_df}")

# convert ISI column to string, to make it work as Hue
diff_df['ISI'] = ['conc' if i == 99 else str(i) for i in diff_df['ISI'].to_list()]

fig, ax = plt.subplots(figsize=(10, 6))
sns.pointplot(data=diff_df, x='cond_num', y='thr_diff',
              hue='ISI',
              estimator=np.mean, errorbar='se', dodge=True, markers='.',
              errwidth=1, capsize=.2,
              )

x_tick_vals = sorted(diff_df['cond_num'].unique())
ax.set_xticks(x_tick_vals)
cond_sep_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '6', 5: '18', 6: '1pr'}
x_tick_labels = [cond_sep_dict[k] for k in x_tick_vals]
ax.set_xticklabels(x_tick_labels)
print(f"pos_sep_vals: {pos_sep_vals}")
print(f"x_tick_labels: {x_tick_labels}")

fig_title = f'exp1a all data: diff UVF - LVF\n(Errors are SEs of means collapsed across participants)'
plt.title(fig_title)
x_axis = 'Sep in diag pixels'
ax.set_xlabel(x_axis)
y_axis = 'Threshold different (UVF - LVF)'
ax.set_ylabel(y_axis)
ax.axhline(y=0, linestyle="-.", color='lightgrey')  # add dotted line at zero

save_as = os.path.join(exp_path, 'diff_vfs.png')
plt.savefig(save_as)
plt.show()


'''Make diff plot per participant'''
if 'p_name' not in list(diff_df.columns):
    diff_df.insert(1, 'p_name', p_name_list)
print(f"diff_df ({diff_df.shape}):\n{diff_df}")

p_name_list = set(p_name_list)
print(f"p_name_list: {p_name_list}")

for p_name in p_name_list:
    p_name_diff_df = diff_df[diff_df['p_name'] == p_name]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.pointplot(data=p_name_diff_df, x='cond_num', y='thr_diff',
                  hue='ISI',
                  estimator=np.mean, errorbar='se',
                  dodge=True, markers='.',
                  errwidth=1, capsize=.2,
                  )

    x_tick_vals = sorted(diff_df['cond_num'].unique())
    ax.set_xticks(x_tick_vals)
    cond_sep_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '6', 5: '18', 6: '1pr'}
    x_tick_labels = [cond_sep_dict[k] for k in x_tick_vals]
    ax.set_xticklabels(x_tick_labels)
    print(f"pos_sep_vals: {pos_sep_vals}")
    print(f"x_tick_labels: {x_tick_labels}")

    fig_title = f'exp1a {p_name}: diff UVF - LVF' \
                # f'\n(Errors are SEs of means collapsed across participants)'
    plt.title(fig_title)
    x_axis = 'Sep in diag pixels'
    ax.set_xlabel(x_axis)
    y_axis = 'Threshold different (UVF - LVF)'
    ax.set_ylabel(y_axis)
    ax.axhline(y=0, linestyle="-.", color='lightgrey')  # add dotted line at zero

    save_as = os.path.join(exp_path, f'{p_name}_diff_vfs.png')
    plt.savefig(save_as)
    plt.show()


'''One plot, all participants, collapsed across ISIs'''
fig, ax = plt.subplots(figsize=(10, 6))
sns.pointplot(data=diff_df, x='cond_num', y='thr_diff',
              hue='p_name',
              estimator=np.mean, errorbar='se',
              dodge=True, markers='.',
              errwidth=1, capsize=.2,
              )

x_tick_vals = sorted(diff_df['cond_num'].unique())
ax.set_xticks(x_tick_vals)
cond_sep_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '6', 5: '18', 6: '1pr'}
x_tick_labels = [cond_sep_dict[k] for k in x_tick_vals]
ax.set_xticklabels(x_tick_labels)
print(f"pos_sep_vals: {pos_sep_vals}")
print(f"x_tick_labels: {x_tick_labels}")

fig_title = f'exp1a participants: diff UVF - LVF' \
            f'\n(Errors are SEs of participant means collapsed across ISIs)'
plt.title(fig_title)
x_axis = 'Sep in diag pixels'
ax.set_xlabel(x_axis)
y_axis = 'Threshold different (UVF - LVF)'
ax.set_ylabel(y_axis)
ax.axhline(y=0, linestyle="-.", color='lightgrey')  # add dotted line at zero

save_as = os.path.join(exp_path, f'participant_diff_vfs.png')
plt.savefig(save_as)
plt.show()


'''just concurrent differences - mean and per participant'''
conc_diff_df = diff_df[diff_df['ISI'] == 'conc']
print(f"conc_diff_df ({conc_diff_df.shape}):\n{conc_diff_df}")

x_tick_vals = conc_diff_df['cond_num'].unique()
# sort variables by sorted(neg_sep_num_list) order
neg_sep_num_array = np.array(x_tick_vals)
print(f"\nneg_sep_num_array: {neg_sep_num_array}")
sort_index = np.argsort(x_tick_vals)
print(f"sort_index: {sort_index}")

x_tick_vals = [x_tick_vals[i] for i in sort_index]
print(f"x_tick_vals: {x_tick_vals}")

x_tick_labels = conc_diff_df['cond_num'].unique()
x_tick_labels = [x_tick_labels[i] for i in sort_index]
x_tick_labels = ['1pr' if i == 20 else i for i in x_tick_labels]
print(f"x_tick_labels: {x_tick_labels}")

# plot participant and mean differences.
fig, ax = plt.subplots(figsize=(10, 6))

sns.lineplot(data=conc_diff_df, x='cond_num', y='thr_diff', hue='p_name',
             alpha=.7)

sns.pointplot(data=conc_diff_df, x='cond_num', y='thr_diff',
              estimator=np.mean, errorbar='se',
              markers='.',
              errwidth=1, capsize=.2, color='black')


fig_title = f'Concurrent: diff UVF - LVF\n' \
            f'(Errors are SEs of means collapsed across participants)'
plt.title(fig_title)
x_axis = 'Probe separation (diag pixels)'
ax.set_xlabel(x_axis)
y_axis = 'Threshold different (UVF - LVF)'
ax.set_ylabel(y_axis)
ax.set_xticks(x_tick_vals)
ax.set_xticklabels(x_tick_labels)
ax.axhline(y=0, linestyle="-.", color='lightgrey')  # add dotted line at zero

save_as = os.path.join(exp_path, 'conc_diff_vfs.png')
plt.savefig(save_as)
plt.show()




'''fig 3: make ISI plots'''
print("\nMaking ISI plots")

ISI_plots_df = exp_VF_thr_long_df.copy()
ISI_er_plot_df = exp_VF_CI_long_df.copy()
print(f"ISI_plots_df ({ISI_plots_df.shape}):\n{ISI_plots_df}")
print(f"ISI_er_plot_df ({ISI_er_plot_df.shape}):\n{ISI_er_plot_df}")

sep_vals = ISI_plots_df['separation'].unique()
sep_num_dict = dict(zip(sep_vals, list(range(len(sep_vals)))))
print(f"\nsep_num_dict: {sep_num_dict}")
ISI_plots_df.insert(4, 'sep_num', ISI_plots_df["separation"].map(sep_num_dict))
print(f"ISI_plots_df ({ISI_plots_df.shape}):\n{ISI_plots_df}")

ISI_er_plot_df.insert(4, 'sep_num', ISI_er_plot_df["separation"].map(sep_num_dict))
ISI_er_plot_df['halved_CI'] = ISI_er_plot_df.CI_width.div(2, fill_value=0)
print(f"ISI_er_plot_df ({ISI_er_plot_df.shape}):\n{exp_VF_CI_long_df}")


ISIs_to_plot = [99, 4, 12]

for this_ISI in ISIs_to_plot:

    ISI_name = this_ISI
    if this_ISI == 99:
        ISI_name = 'Concurrent'
    print(f"ISI: {ISI_name} ({this_ISI})")

    ISI_df = ISI_plots_df.loc[ISI_plots_df['ISI'] == this_ISI]
    print(f"ISI_df ({ISI_df.shape}):\n{ISI_df}")

    ISI_error_df = ISI_er_plot_df[ISI_er_plot_df['ISI'] == this_ISI]
    print(f" ISI_error_df ({ISI_error_df.shape}):\n{ISI_error_df}")

    x_tick_vals = sorted(ISI_df['sep_num'].unique())
    print(f"\nx_tick_vals: {x_tick_vals}")

    x_tick_labels = ['1pr' if i == 20 else str(i) for i in sorted(ISI_df['separation'].unique())]
    print(f"\nx_tick_labels: {x_tick_labels}")

    fig, ax = plt.subplots(figsize=(10, 6))

    # thick lines showing means with errors.
    sns.pointplot(data=ISI_df, x='sep_num', y='probeLum',
                  hue='vis_field',
                  errorbar='se', capsize=.05,
                  scale=1.25,
                  )

    # background faint lines showing actual thr
    sns.lineplot(data=ISI_df, x='sep_num', y='probeLum',
                 hue='vis_field',
                 style='p_name',
                 dashes=False,
                 alpha=.3,
                 legend=False
                 )


    ax.set_xticks(x_tick_vals)
    ax.set_xticklabels(x_tick_labels)
    plt.title(f'ISI {ISI_name}: compare UVF & LVF\n(Error bars: SE of Participant thresholds)')
    ax.set_xlabel('Sep in diag pixels')
    ax.set_ylabel('Threshold')
    plt.savefig(os.path.join(exp_path, f'exp1a_ISI_{ISI_name}_VFs'))
    plt.show()



'''figs 4, 5, 6, 6: make sep plots for sep 0, 2, 3, 6'''
print("\nMaking separation plots")
ISI_vals = exp_VF_thr_long_df['ISI'].unique()
ISI_num_dict = dict(zip(ISI_vals, list(range(len(ISI_vals)))))
print(f"\nISI_num_dict: {ISI_num_dict}")
exp_VF_thr_long_df.insert(4, 'ISI_num', exp_VF_thr_long_df["ISI"].map(ISI_num_dict))
sep_to_plot = [0, 2, 3, 6]

for this_sep in sep_to_plot:
    sep_df = exp_VF_thr_long_df[exp_VF_thr_long_df['separation'] == this_sep]
    print(f"sep_df:\n{sep_df}")

    x_tick_vals = sep_df['ISI_num'].unique()
    x_tick_labels = ['conc' if i == 99 else str(i) for i in sep_df['ISI'].unique()]
    print(f"\nx_tick_vals: {x_tick_vals}")
    print(f"x_tick_labels: {x_tick_labels}")


    # thick lines showing means with errors.
    sns.pointplot(data=sep_df, x='ISI_num', y='probeLum',
                  hue='vis_field',
                  style='p_name',
                  scale=1.25,
                  errorbar='se', capsize=.05,
                  )

    # background faint lines showing actual thr
    sns.lineplot(data=sep_df, x='ISI_num', y='probeLum',
                 hue='vis_field',
                 style='p_name',
                 dashes=False,
                 alpha=.3,
                 legend=False
                 )

    ax = plt.gca()  # to get the axis
    ax.set_xticks(x_tick_vals)
    ax.set_xticklabels(x_tick_labels)
    plt.title(f'Sep {this_sep}: compare UVF & LVF\n(Error bars: SE of Participant thresholds)')
    ax.set_xlabel('ISI')
    ax.set_ylabel('Threshold')
    plt.savefig(os.path.join(exp_path, f'exp1a_sep{this_sep}_VFs'))
    plt.show()



'''make plots for each participant showing there actual thresholds for concurrent'''
print("\nMaking participant plots for concurrent probes")

# add sep num column to thr and err dfs.
sep_vals = exp_VF_thr_long_df['separation'].unique()
sep_num_dict = dict(zip(sep_vals, list(range(len(sep_vals)))))
print(f"\nsep_num_dict: {sep_num_dict}")

if 'sep_num' not in list(exp_VF_thr_long_df.columns):
    exp_VF_thr_long_df.insert(4, 'sep_num', exp_VF_thr_long_df["separation"].map(sep_num_dict))
print(f"exp_VF_thr_long_df ({exp_VF_thr_long_df.shape}):\n{exp_VF_thr_long_df}")

if 'sep_num' not in list(exp_VF_CI_long_df.columns):
    exp_VF_CI_long_df.insert(4, 'sep_num', exp_VF_CI_long_df["separation"].map(sep_num_dict))

# add calved CI column to err_df
if 'halved_CI' not in list(exp_VF_CI_long_df.columns):
    exp_VF_CI_long_df['halved_CI'] = exp_VF_CI_long_df.CI_width.div(2, fill_value=0)
print(f"exp_VF_CI_long_df ({exp_VF_CI_long_df.shape}):\n{exp_VF_CI_long_df}")

# Just concurrent data
conc_thr_df = exp_VF_thr_long_df[exp_VF_thr_long_df['ISI'] == 99]
print(f"conc_thr_df ({conc_thr_df.shape}):\n{conc_thr_df}")
conc_CI_df = exp_VF_CI_long_df[exp_VF_CI_long_df['ISI'] == 99]
print(f"conc_CI_df ({conc_CI_df.shape}):\n{conc_CI_df}")

# get participant and vis_field conditions to loop through
p_name_list = conc_thr_df['p_name'].unique()
print(f"p_name_list: {p_name_list}")
vf_list = conc_thr_df['vis_field'].unique()
print(f"vf_list: {vf_list}")

# make plots
cap_size = 7
for p_name in p_name_list:
    p_name_thr_df = conc_thr_df[conc_thr_df['p_name'] == p_name]
    print(f"p_name_thr_df:\n{p_name_thr_df}")

    p_name_err_df = conc_CI_df[conc_CI_df['p_name'] == p_name]
    print(f"p_name_err_df:\n{p_name_err_df}")

    x_tick_vals = sorted(p_name_thr_df['sep_num'].unique())
    x_tick_labels = ['1pr' if i == 20 else str(i) for i in sorted(p_name_thr_df['separation'].unique())]

    fig, ax = plt.subplots(figsize=(10, 6))
    for vis_field in vf_list:

        # dfs just for this vis field
        VF_thr_df = p_name_thr_df[p_name_thr_df['vis_field'] == vis_field]
        VF_err_df = p_name_err_df[p_name_err_df['vis_field'] == vis_field]

        ax.errorbar(data=VF_thr_df, x='sep_num', y='probeLum',
                    yerr=VF_err_df['halved_CI'],
                    marker='o', lw=3, elinewidth=2,
                    capsize=cap_size)

    ax = plt.gca()  # to get the axis
    ax.set_xticks(x_tick_vals)
    ax.set_xticklabels(x_tick_labels)
    plt.title(f'{p_name}: compare UVF & LVF\n(Error bars: Participant 95% CIs)')
    ax.set_xlabel('ISI')
    ax.set_ylabel('Threshold')
    plt.savefig(os.path.join(exp_path, f'exp1a_{p_name}_VFs.png'))
    plt.show()


print('\nexp1a_analysis_pipe_UVF_LVF finished\n')
