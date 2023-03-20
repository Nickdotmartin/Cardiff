import os
import numpy as np
import pandas as pd
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import seaborn as sns
from exp1a_psignifit_analysis import a_data_extraction, b3_plot_staircase, c_plots, \
    d_average_participant, e_average_exp_data, make_average_plots, make_long_df, \
    plot_w_errors_no_1probe
from rad_flow_psignifit_analysis import plot_runs_ave_w_errors, fig_colours, plot_ave_w_errors_markers
from psignifit_tools import get_psignifit_threshold_df, get_psig_thr_w_hue
from python_tools import which_path, running_on_laptop, switch_path, print_nested_round_floats
from PsychoPy_tools import get_pixel_mm_deg_values


'''
This script if for checking for any differences between thr upper visual field and lower visual field.
It will use the already analysied RUNDATA_sorted.xlsx to do this.
Loop through the participant run folders and append each RUNDATA-sorted.xlsx, with addition 'run' column.
Save to P_all_runs_master_output.csv.

Then run psignifit on this
'''

# # loop through run folders with first 4 scripts (a, get_psignifit_threshold_df, b3, c)
# # then run script d to get master lists and averages
exp_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff"
# exp_name = 'Exp3_Ricco_NM_v4'
# exp_name = 'Exp3_Ricco_NM_v5'
# exp_name = 'Exp3_Ricco_v6'
exp_name = 'Exp3_Ricco_all'
ricco_version = 'all'
exp_path = os.path.join(exp_path, exp_name)
convert_path1 = os.path.normpath(exp_path)
if running_on_laptop():
    convert_path1 = switch_path(convert_path1, 'mac_oneDrive')
exp_path = convert_path1

# participant_list = ['Kris', 'Simon', 'Nick', 'Kim']  # ['Kim', 'Kris', 'Simon', 'Nick']
participant_list = ['Kristian', 'Nick', 'Simon', 'Kim']  # , 'bb', 'cc', 'dd', 'ee']

p_idx_plus = 1

n_runs = 12
analyse_from_run = 1
trim_list = []

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
#
#         if 'ISI' not in list(run_data_df.columns):
#             run_data_df.insert(6, 'ISI', 0)
#         elif run_data_df.ISI.isnull().sum() > 0:  # check for missing values
#             print('df has missing ISI value')
#             run_data_df['ISI'] = 0
#         isi_list = run_data_df['ISI'].unique()
#         print(f'isi_list: {isi_list}')
#
#         # if column is missing or if there are missing values, fill these out.
#         all_sep_vals = run_data_df['separation'].tolist()
#         if 'n_pix' not in list(run_data_df.columns):
#             all_n_pix_vals = [5 if i in [99, 20] else i * 5 + 10 for i in all_sep_vals]
#             run_data_df['n_pix'] = all_n_pix_vals
#         elif run_data_df.n_pix.isnull().sum() > 0:  # check for missing values
#             print('df has missing n_pix value')
#             all_n_pix_vals = [5 if i in [99, 20] else i * 5 + 10 for i in all_sep_vals]
#             run_data_df['n_pix'] = all_n_pix_vals
#         n_pixels_list = run_data_df['n_pix'].unique()
#         print(f'n_pixels_list: {n_pixels_list}')
#
#         if 'len_pix' not in list(run_data_df.columns):
#             all_len_pix_vals = [1.5 if i in [99, 20] else i + 2.5 for i in all_sep_vals]
#             run_data_df['len_pix'] = all_len_pix_vals
#         elif run_data_df.len_pix.isnull().sum() > 0:  # check for missing values
#             print('df has missing len_pix value')
#             all_len_pix_vals = [1.5 if i in [99, 20] else i + 2.5 for i in all_sep_vals]
#             run_data_df['len_pix'] = all_len_pix_vals
#         len_pixels_list = run_data_df['len_pix'].unique()
#         print(f'len_pixels_list: {len_pixels_list}')
#
#         if 'diag_deg' not in list(run_data_df.columns):
#             pixel_mm_deg_dict = get_pixel_mm_deg_values(monitor_name='asus_cal')
#             all_len_pix_vals = run_data_df['len_pix'].tolist()
#             all_diag_deg_vals = [i * pixel_mm_deg_dict['diag_deg'] for i in all_len_pix_vals]
#             run_data_df['diag_deg'] = all_diag_deg_vals
#         elif run_data_df.diag_deg.isnull().sum() > 0:  # check for missing values
#             print('df has missing diag_deg value')
#             pixel_mm_deg_dict = get_pixel_mm_deg_values(monitor_name='asus_cal')
#             all_len_pix_vals = run_data_df['len_pix'].tolist()
#             all_diag_deg_vals = [i * pixel_mm_deg_dict['diag_deg'] for i in all_len_pix_vals]
#             run_data_df['diag_deg'] = all_diag_deg_vals
#         len_degrees_list = run_data_df['diag_deg'].unique()
#         print(f'len_degrees_list: {len_degrees_list}')
#
#
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
#         if 'ISI' not in list(vis_field_df.columns):
#             vis_field_df['ISI'] = 0
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
#         # change 1pr size to be 20 not -1
#         thr_df['separation'] = [20 if i == -1 else i for i in thr_df['separation'].tolist()]
#
#         neg_sep_list = thr_df['separation'].to_list()
#         if vis_field_name == 'LVF':
#             neg_sep_list = [-.01 if i == 0 else -i for i in neg_sep_list]
#             # neg_sep_list = [-i for i in neg_sep_list]
#         # thr_df['neg_sep'] = neg_sep_list
#         thr_df.insert(2, 'neg_sep', neg_sep_list)
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
#         VF_CI_width_df.insert(2, 'neg_sep', neg_sep_list)
#         both_vfs_CI_width.append(VF_CI_width_df)
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
#     # sep_list = [20 if i == 99 else i for i in sep_list]
#     psig_both_vf_df['separation'] = sep_list
#     both_vfs_CI_width_df['separation'] = sep_list
#
#     if 'neg_sep' not in both_vf_columns:
#         print("\nMaking neg_sep column")
#         # add condition list which is equal to sep for uVF or negative sep for LVF (with -.01 instead of -0)
#         sep_list = psig_both_vf_df['separation'].to_list()
#         vf_list = psig_both_vf_df['vis_field'].to_list()
#         neg_sep_list = []
#         for vf, sep in zip(vf_list, sep_list):
#             if vf == 'LVF':
#                 # if sep == 0:
#                 #     this_cond = -.01
#                 # else:
#                 #     this_cond = -sep
#                 this_cond = -sep
#             else:
#                 this_cond = sep
#             print(f"vf: {vf}, sep: {sep}, this_cond: {this_cond}")
#             neg_sep_list.append(this_cond)
#         print(f"neg_sep_list: {neg_sep_list}")
#         psig_both_vf_df.insert(2, 'neg_sep', neg_sep_list)
#         both_vfs_CI_width_df.insert(2, 'neg_sep', neg_sep_list)
#
#
#     # change 1probe from 99 to 20
#     neg_sep_list = psig_both_vf_df['neg_sep'].to_list()
#     neg_sep_list = [20 if i == 99 else i for i in neg_sep_list]
#     neg_sep_list = [-20 if i == -99 else i for i in neg_sep_list]
#     psig_both_vf_df['neg_sep'] = neg_sep_list
#     both_vfs_CI_width_df['neg_sep'] = neg_sep_list
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
# # # make long form df
# exp_VF_thr_long_df = pd.read_csv(os.path.join(exp_path, 'MASTER_exp_VF_thr.csv'))
# exp_VF_thr_long_df.rename({'ISI_0': 'probeLum'}, axis=1, inplace=True)
# print(f"\nexp_VF_thr_long_df ({exp_VF_thr_long_df.shape}):\n{exp_VF_thr_long_df}")
#
#
# # get dva column
# sep_vals_list = exp_VF_thr_long_df['separation'].unique()
# print(f"sep_vals_list: {sep_vals_list}")
#
#
# pixel_mm_deg_dict = get_pixel_mm_deg_values(monitor_name='asus_cal')
# print_nested_round_floats(pixel_mm_deg_dict, 'pixel_mm_deg_dict')
# dva_dict = {}
# for sep_cond in sep_vals_list:
#     sep_cond = int(sep_cond)
#
#     if sep_cond in [20, 99, '1pr']:
#         len_pix = 1.5
#         n_pix = 5
#     else:
#         len_pix = 2.5 + sep_cond
#         n_pix = sep_cond * 5 + 10
#
#     probe_name = f"sep_{sep_cond}"
#     dva_dict[sep_cond] = len_pix * pixel_mm_deg_dict['diag_deg']
#
#
# print_nested_round_floats(dva_dict, 'dva_dict')
#
#
# # make long form CIs
# exp_VF_CI_long_df = pd.read_csv(os.path.join(exp_path, 'MASTER_exp_VF_CI.csv'))
# exp_VF_CI_long_df.rename({'ISI_0': 'CI_width'}, axis=1, inplace=True)
# print(f"\nexp_VF_CI_long_df:\n{exp_VF_CI_long_df}")
#
# # add sep num column with 1pr (20) first
# sep_vals_list = sorted(exp_VF_thr_long_df['separation'].unique())
# if sep_vals_list[-1] in [20, 99, '1pr']:
#     sep_vals_list = sep_vals_list[-1:] + sep_vals_list[:-1]
# print(f"sep_vals_list: {sep_vals_list}")
# sep_num_dict = dict(zip(sep_vals_list, list(range(len(sep_vals_list)))))
# print(f"\nsep_num_dict: {sep_num_dict}")
# exp_VF_thr_long_df.insert(2, 'sep_num', exp_VF_thr_long_df["separation"].map(sep_num_dict))
# exp_VF_CI_long_df.insert(2, 'sep_num', exp_VF_CI_long_df["separation"].map(sep_num_dict))
#
# # add neg_sep_num column
# neg_sep_list = exp_VF_thr_long_df['neg_sep'].unique()
#
# srtd_below_zero = sorted([i for i in neg_sep_list if i < 0])
# srtd_above_zero = sorted([i for i in neg_sep_list if i >= 0])
# print(f"srtd_below_zero: {srtd_below_zero}")
# print(f"srtd_above_zero: {srtd_above_zero}")
#
# if 20 in srtd_above_zero:
#     srtd_above_zero.remove(20)
#     srtd_above_zero = [20] + srtd_above_zero
# print(f"srtd_above_zero: {srtd_above_zero}")
#
#
# if -20 in srtd_below_zero:
#     srtd_below_zero.remove(-20)
#     srtd_below_zero = srtd_below_zero + [-20]
# print(f"srtd_below_zero: {srtd_below_zero}")
#
# neg_sep_list = srtd_below_zero + srtd_above_zero
# print(f"neg_sep_list: {neg_sep_list}")
#
#
# neg_sep_num_dict = dict(zip(neg_sep_list, list(range(len(neg_sep_list)))))
# print(f"\nneg_sep_num_dict: {neg_sep_num_dict}")
# exp_VF_thr_long_df.insert(4, 'neg_sep_num', exp_VF_thr_long_df["neg_sep"].map(neg_sep_num_dict))
# exp_VF_CI_long_df.insert(4, 'neg_sep_num', exp_VF_CI_long_df["neg_sep"].map(neg_sep_num_dict))
# # print(f"\nexp_VF_thr_long_df:\n{exp_VF_thr_long_df}")
#
# # add dva column
# exp_VF_thr_long_df.insert(5, 'dva', exp_VF_thr_long_df["separation"].map(dva_dict))
# exp_VF_CI_long_df.insert(5, 'dva', exp_VF_CI_long_df["separation"].map(dva_dict))
# print(f"\nexp_VF_thr_long_df ({list(exp_VF_thr_long_df.columns)}):\n{exp_VF_thr_long_df}")
# print(f"\nexp_VF_thr_long_df['dva'].tolist(): {exp_VF_thr_long_df['dva'].tolist()}")
#
# # add dva_num column
# dva_vals = sorted(exp_VF_thr_long_df['dva'].unique())
# dva_num_dict = dict(zip(dva_vals, list(range(len(dva_vals)))))
# print(f"\ndva_num_dict: {dva_num_dict}")
# exp_VF_thr_long_df.insert(6, 'dva_num', exp_VF_thr_long_df["dva"].map(dva_num_dict))
# exp_VF_CI_long_df.insert(6, 'dva_num', exp_VF_CI_long_df["dva"].map(dva_num_dict))
# print(f"\nexp_VF_thr_long_df:\n{exp_VF_thr_long_df}")
# print(f"\nexp_VF_CI_long_df:\n{exp_VF_CI_long_df}")
#
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
print('\nmaking plots')

exp_VF_thr_long_df = pd.read_csv(os.path.join(exp_path, 'MASTER_exp_VF_thr_long.csv'))
print(f"\nexp_VF_thr_long_df:\n{exp_VF_thr_long_df}")


# get means per condition
groupby_sep_thr_df = exp_VF_thr_long_df.copy()
groupby_sep_thr_df = groupby_sep_thr_df.drop('p_name', axis=1)
exp_mean_thr_long_df = groupby_sep_thr_df.groupby('neg_sep_num', sort=True).mean()
vis_field_list = ['LVF' if i < 0 else 'UVF' for i in exp_mean_thr_long_df['neg_sep'].tolist()]
exp_mean_thr_long_df.insert(3, 'vis_field', vis_field_list)

exp_mean_thr_long_df.reset_index(inplace=True)
print(f"\nexp_mean_thr_long_df:\n{exp_mean_thr_long_df}")


# get mean errors per condition
exp_VF_CI_long_df = pd.read_csv(os.path.join(exp_path, 'MASTER_exp_VF_CI_long.csv'))
print(f"\nexp_VF_CI_long_df:\n{exp_VF_CI_long_df}")

# add in halved CI (rather than total width)
exp_VF_CI_long_df['halved_CI'] = exp_VF_CI_long_df.CI_width.div(2, fill_value=0)
exp_VF_CI_long_df = exp_VF_CI_long_df.drop('CI_width', axis=1)

groupby_sep_CI_df = exp_VF_CI_long_df.copy()
groupby_sep_CI_df = groupby_sep_CI_df.drop('p_name', axis=1)
exp_mean_CI_long_df = groupby_sep_CI_df.groupby('neg_sep_num', sort=True).mean()
exp_mean_CI_long_df.insert(3, 'vis_field', vis_field_list)
exp_mean_CI_long_df.reset_index(inplace=True)

print(f"\nexp_mean_CI_long_df:\n{exp_mean_CI_long_df}")

print(f"\nexp_mean_thr_long_df:\n{exp_mean_thr_long_df}")

# '''Fig 1 - all data'''
#
# print('\nFig 1 - all data')

# fig_1_thr_df = exp_VF_thr_long_df.copy(deep=True)
# print(f"fig_1_thr_df:\n{fig_1_thr_df}")
# fig_1_err_df = exp_VF_CI_long_df.copy(deep=True)
# print(f"fig_1_err_df:\n{fig_1_err_df}")
#
# p_name_list = fig_1_thr_df['p_name'].unique().tolist()
# print(f"p_name_list: {p_name_list}")
#
# # use wide means df
# fig_1_thr_df = fig_1_thr_df.pivot(index='p_name',
#                                   columns=['neg_sep_num', 'neg_sep', 'separation', 'sep_num', 'vis_field', 'dva', 'dva_num'],
#                                   values='probeLum')
#
#
#
# print(f"\nfig_1_thr_df (with multi-indexed columns):\n{fig_1_thr_df}")
#
# # get variables for labels, ticks etc
# multi_idx_cols = fig_1_thr_df.columns.to_list()
# neg_sep_num_list = [x[0] for x in multi_idx_cols]
# neg_sep_list = [x[1] for x in multi_idx_cols]
# sep_list = [x[2] for x in multi_idx_cols]
# sep_num_list = [x[3] for x in multi_idx_cols]
# vis_field_list = [x[4] for x in multi_idx_cols]
# dva_list = [x[5] for x in multi_idx_cols]
# dva_num_list = [x[6] for x in multi_idx_cols]
# print('\nUn-sorted variable lists:')
# print(f"neg_sep_num_list: {neg_sep_num_list}")
# print(f"neg_sep_list: {neg_sep_list}")
# print(f"sep_list: {sep_list}")
# print(f"sep_num_list: {sep_num_list}")
# print(f"vis_field_list: {vis_field_list}")
# print(f"dva_list: {dva_list}")
# print(f"dva_num_list: {dva_num_list}")
#
#
# # sort variables by sorted(neg_sep_num_list) order
# neg_sep_num_array = np.array(neg_sep_num_list)
# print(f"\nneg_sep_num_array: {neg_sep_num_array}")
# sort_index = np.argsort(neg_sep_num_list)
# print(f"sort_index: {sort_index}")
#
#
# neg_sep_num_list = [neg_sep_num_list[i] for i in sort_index]
# neg_sep_list = [neg_sep_list[i] for i in sort_index]
# half_list_len = int(len(sep_list)/2)
# sep_list = [sep_list[i] for i in sort_index][half_list_len:]
# sep_num_list = [sep_num_list[i] for i in sort_index][half_list_len:]
# vis_field_list = [vis_field_list[i] for i in sort_index]
# dva_list = [dva_list[i] for i in sort_index][half_list_len:]
# dva_num_list = [dva_num_list[i] for i in sort_index][half_list_len:]
# print('\nSorted variable lists:')
# print(f"neg_sep_num_list: {neg_sep_num_list}")
# print(f"neg_sep_list: {neg_sep_list}")
# print(f"sep_list: {sep_list}")
# print(f"sep_num_list: {sep_num_list}")
# print(f"vis_field_list: {vis_field_list}")
# print(f"dva_list: {dva_list}")
# print(f"dva_num_list: {dva_num_list}")
#
#
# # sort column order then reduce index to just be the top row.
# fig_1_thr_df.reset_index(inplace=True)
# fig_1_thr_df = fig_1_thr_df.set_index('p_name')
# column_order = sorted(neg_sep_num_list)
# print(f"column_order: {column_order}")
# # re-order
# fig_1_thr_df = fig_1_thr_df[column_order]
# # drop multi-index
# fig_1_thr_df.columns = column_order
# print(f"fig_1_thr_df:\n{fig_1_thr_df}")
#
#
# fig_1_err_df = fig_1_err_df.pivot(index='p_name',
#                                   columns='neg_sep_num', values='halved_CI')
# fig_1_err_df.reset_index(inplace=True, drop=False)
# fig_1_err_df = fig_1_err_df.set_index('p_name')
#
# fig_1_err_df = fig_1_err_df[column_order]
# print(f"fig_1_err_df:\n{fig_1_err_df}")
#
# # get values for x tick locations and labels
# x_tick_vals = neg_sep_num_list
# x_tick_labels = neg_sep_list
# x_tick_labels = ['1pr' if i in [20.0, -20.0] else str(i) for i in x_tick_labels]
# x_tick_labels = ['-0' if i == '-0.01' else str(i) for i in x_tick_labels]
# x_tick_labels = [i[:-2] if i not in ['1pr', '-0'] else i for i in x_tick_labels]
#
# print(f"x_tick_vals: {x_tick_vals}")
# print(f"x_tick_labels: {x_tick_labels}")
# fig_1a_title = f'{exp_name}: compare UVF & LVF\n(Errors are mean of participant CIs, per ISI)'
#
# plot_runs_ave_w_errors(fig_df=fig_1_thr_df.T, error_df=fig_1_err_df.T,
#                        jitter=.1, error_caps=True, alt_colours=False,
#                        x_tick_vals=x_tick_vals,
#                        x_tick_labels=x_tick_labels,
#                        x_axis_label='Sep in diag pixels. Neg=LVF, Pos=UVF',
#                        even_spaced_x=True, fixed_y_range=False,
#                        fig_title=fig_1a_title, save_name='all_data_VFs.png',
#                        save_path=exp_path, verbose=True)
# ax = plt.gca() # to get the axis
# ax.axvline(x=(x_tick_vals[-1]/2), linestyle="-.", color='lightgrey')  # add dotted line at zero
#
# plt.show()
# plt.close()
#
#
# # Fig 2, participant plots
# print('\nFig 2, participant plots:')
#
# # get dfs
# print(f"exp_VF_thr_long_df:\n{exp_VF_thr_long_df}")
# print(f"exp_VF_CI_long_df:\n{exp_VF_CI_long_df}")
#
#
# # add halved CI column to err_df
# if 'halved_CI' not in list(exp_VF_CI_long_df.columns):
#     exp_VF_CI_long_df['halved_CI'] = exp_VF_CI_long_df.CI_width.div(2, fill_value=0)
# print(f"exp_VF_CI_long_df ({exp_VF_CI_long_df.shape}):\n{exp_VF_CI_long_df}")
#
# # get participant and vis_field conditions to loop through
# p_name_list = exp_VF_thr_long_df['p_name'].unique()
# print(f"p_name_list: {p_name_list}")
# vf_list = exp_VF_thr_long_df['vis_field'].unique()
# print(f"vf_list: {vf_list}")
#
# cap_size = 7
# for p_name in p_name_list:
#
#     # get participant data
#     p_long_thr_df = exp_VF_thr_long_df[exp_VF_thr_long_df['p_name'] == p_name]
#     print(f"p_long_thr_df ({p_name}):\n{p_long_thr_df}")
#     p_long_err_df = exp_VF_CI_long_df[exp_VF_CI_long_df['p_name'] == p_name]
#     print(f"p_long_err_df ({p_name}):\n{p_long_err_df}")
#
#     # get x tick values and labels
#     x_tick_vals = p_long_thr_df['sep_num'].unique()
#     print(f"\nx_tick_vals: {x_tick_vals}")
#     x_tick_labels = p_long_thr_df['separation'].unique()
#     x_tick_labels = ['1pr' if i in [20.0, -20.0] else str(i) for i in x_tick_labels]
#     print(f"x_tick_labels: {x_tick_labels}")
#
#     fig, ax = plt.subplots(figsize=(10, 6))
#     my_colours = fig_colours(len(vf_list))
#     legend_handles_list = []
#
#     for idx, vis_field in enumerate(vf_list):
#         # dfs just for this vis field
#         VF_thr_df = p_long_thr_df[p_long_thr_df['vis_field'] == vis_field]
#         VF_err_df = p_long_err_df[p_long_err_df['vis_field'] == vis_field]
#
#         ax.errorbar(data=VF_thr_df, x='sep_num', y='probeLum',
#                     yerr=VF_err_df['halved_CI'],
#                     marker='o', lw=3, elinewidth=2,
#                     capsize=cap_size,
#                     color=my_colours[idx],
#                     )
#
#         # add legend for visual field colours
#         leg_handle = mlines.Line2D([], [], color=my_colours[idx], label=vis_field,
#                                    marker='.', linewidth=.5, markersize=4)
#         legend_handles_list.append(leg_handle)
#
#     ax.legend(handles=legend_handles_list, fontsize=6, title='Vis field', framealpha=.5)
#
#     ax = plt.gca()  # to get the axis
#     ax.set_xticks(x_tick_vals)
#     ax.set_xticklabels(x_tick_labels)
#     plt.title(f'{exp_name}. {p_name}: compare UVF & LVF')
#     x_axis = 'Probe size (equivallent to separation from Exp 1)'
#     ax.set_xlabel(x_axis)
#     ax.set_ylabel('Threshold')
#     plt.savefig(os.path.join(exp_path, f'{exp_name}_{p_name}_VFs'))
#     plt.show()
#
#
#
'''Fig 2, difference between UVF and LVF, per participant and mean'''
# todo: can use these dfs for participant thr plots.  Might be more efficient
print(f"\nplot diff between VFs for each ISI")
# for each separation value, subtract LFV from UVF for difference score.
get_diff_df = exp_VF_thr_long_df.copy()
print(f"get_diff_df ({get_diff_df.shape}):\n{list(get_diff_df.columns)}\n{get_diff_df}")

# Lower VF df
LVF_df = get_diff_df.loc[get_diff_df['neg_sep'] < 0]
LVF_df.sort_values(by=['sep_num', 'p_name'])
LVF_df = LVF_df.drop(['vis_field', 'neg_sep_num', 'neg_sep'], axis=1)
LVF_df.rename({'probeLum': 'LVF_thr'}, axis=1, inplace=True)
LVF_df.reset_index(inplace=True, drop=True)
print(f"LVF_df ({LVF_df.shape}):\n{LVF_df}")

# upper visual field df
UVF_df = get_diff_df.loc[get_diff_df['neg_sep'] >= 0]
UVF_df.sort_values(by=['sep_num', 'p_name'])
UVF_df = UVF_df.drop(['vis_field', 'neg_sep_num', 'neg_sep'], axis=1)
UVF_df.rename({'probeLum': 'UVF_thr'}, axis=1, inplace=True)
UVF_df.reset_index(inplace=True, drop=True)
print(f"UVF_df ({UVF_df.shape}):\n{UVF_df}")

# diff_df has difference and raw scores for upper and lower.
diff_df = pd.merge(LVF_df, UVF_df, on=['p_name', 'sep_num', 'separation', 'dva_num', 'dva'])
diff_df['thr_diff'] = diff_df['UVF_thr'] - diff_df['LVF_thr']
diff_df.to_csv(os.path.join(exp_path, 'diff_df_test.csv'))
print(f"diff_df ({diff_df.shape}):\n{diff_df}")

# plot participant and mean differences.
fig, ax = plt.subplots(figsize=(10, 6))

sns.pointplot(data=diff_df, x='sep_num', y='thr_diff',
              estimator=np.mean, errorbar='se',
              markers='.',
              errwidth=.5, capsize=.1, color='dimgrey')

sns.lineplot(data=diff_df, x='sep_num', y='thr_diff', hue='p_name',
             alpha=.7)



fig_title = f'{exp_name}: diff UVF - LVF\n' \
            f'(Errors are SEs of means collapsed across participants)'
plt.title(fig_title)
x_axis = 'Probe size (equivallent to separation from Exp 1)'
ax.set_xlabel(x_axis)
y_axis = 'Threshold different (UVF - LVF)'
ax.set_ylabel(y_axis)
ax.set_xticks(list(diff_df['sep_num']))
ax.set_xticklabels(['1pr' if i == 20 else i for i in list(diff_df['separation'])])
ax.axhline(y=0, linestyle="-.", color='lightgrey')  # add dotted line at zero

save_as = os.path.join(exp_path, 'diff_vfs.png')
plt.savefig(save_as)
plt.show()

# # make ricco plots
# # exp_VF_thr_long_df = pd.read_csv(os.path.join(exp_path, 'MASTER_exp_VF_thr_long.csv'))
# print(f"\nexp_VF_thr_long_df:\n{exp_VF_thr_long_df}")
# print(f"\nexp_VF_CI_long_df:\n{exp_VF_CI_long_df}")
#
# ricco_thr_df = exp_VF_thr_long_df.copy(deep=True)
# ricco_CI_df = exp_VF_CI_long_df.copy(deep=True)
#
# ricco_thr_df.rename({'probeLum': 'thr',
#                      'dva': 'diag_deg'}, axis=1, inplace=True)
# ricco_CI_df.rename({'halved_CI': 'thr',
#                      'dva': 'diag_deg'}, axis=1, inplace=True)
#
# if 'delta_I' not in ricco_thr_df.columns.tolist():
#     thr_col = ricco_thr_df['thr'].to_list()
#     bgLum = 21.2
#     delta_I_col = [i - bgLum for i in thr_col]
#     ricco_thr_df.insert(8, 'delta_I', delta_I_col)
#
# # just add in CIs to dela I column
# if 'delta_I' not in ricco_CI_df.columns.tolist():
#     thr_col = ricco_CI_df['thr'].to_list()
#     # bgLum = 21.2
#     # delta_I_col = [i - bgLum for i in thr_col]
#     # ricco_CI_df.insert(8, 'delta_I', delta_I_col)
#     ricco_CI_df.insert(8, 'delta_I', thr_col)
#
# p_name_list = ricco_thr_df['p_name'].unique().tolist()
# print(f"p_name_list: {p_name_list}")
#
# vf_list = ricco_thr_df['vis_field'].unique().tolist()
# print(f"vf_list: {vf_list}")
#
#
# for p_name in p_name_list:
#     print(f"p_name: {p_name}")
#     p_name_thr_df = ricco_thr_df[ricco_thr_df['p_name'] == p_name]
#     p_name_CI_df = ricco_CI_df[ricco_CI_df['p_name'] == p_name]
#     print(f"\np_name_thr_df:\n{p_name_thr_df}")
#
#     for vis_field in vf_list:
#         print(f"vis_field: {vis_field}")
#         vis_field_thr_df = p_name_thr_df[p_name_thr_df['vis_field'] == vis_field]
#         vis_field_CI_df = p_name_CI_df[p_name_CI_df['vis_field'] == vis_field]
#         print(f"\nvis_field_thr_df:\n{vis_field_thr_df}")
#
#         exp_ave = False
#
#         # load data and change order to put 1pr last
#         print('*** making participant average plots ***')
#         print(f'exp_path: {exp_path}')
#
#         # fig_df = pd.read_csv(p_ave_path)
#         fig_df = vis_field_thr_df
#         print(f'fig_df:\n{fig_df}')
#
#         # if 'separation' not in fig_df.columns:
#         #     if 'stair_names' in fig_df.columns:
#         #         sep_list = [int(i[:-6]) for i in fig_df['stair_names'].to_list()]
#         #         fig_df.insert(1, 'separation', sep_list)
#
#         # get sep cond values for legend
#         sep_list = fig_df['separation'].to_list()
#         sep_vals_list = [i for i in sep_list]
#         sep_name_list = ['1pr' if i == 20 else f'sep{i}' for i in sep_list]
#         print(f'sep_vals_list: {sep_vals_list}')
#         print(f'sep_name_list: {sep_name_list}\n')
#
#         if 'diag_deg' not in fig_df.columns:
#             if 'len_pix' in fig_df.columns:
#                 pixel_mm_deg_dict = get_pixel_mm_deg_values(monitor_name='asus_cal')
#                 len_degrees_list = [i * pixel_mm_deg_dict['diag_deg'] for i in fig_df['len_pix'].to_list()]
#                 fig_df.insert(2, 'diag_deg', len_degrees_list)
#         print(f'fig_df:\n{fig_df}')
#
#         # error_df = pd.read_csv(err_path)
#         error_df = vis_field_CI_df
#         print(f'error_df:\n{error_df}')
#
#         len_degrees_list = fig_df['diag_deg'].to_list()
#         print(f'len_degrees_list: {len_degrees_list}')
#         if 'diag_deg' not in error_df.columns:
#             error_df.insert(4, 'diag_deg', len_degrees_list)
#         else:
#             error_df['diag_deg'] = len_degrees_list
#
#         if 'cond' not in error_df.columns:
#             cond_list = ['lines'] * len(len_degrees_list)
#             error_df.insert(1, 'cond', cond_list)
#             fig_df.insert(1, 'cond', cond_list)
#         print(f'error_df:\n{error_df}')
#
#         print(f'fig_df:\n{fig_df}')
#
#         # # fig 1 - len degrees v thr
#         wide_df = fig_df.pivot(index=['diag_deg'], columns='cond', values='thr')
#         print(f'wide_df:\n{wide_df}')
#         wide_err_df = error_df.pivot(index=['diag_deg'], columns='cond', values='thr')
#
#         len_degrees_list = fig_df['diag_deg'].to_list()
#         print(f'len_degrees_list: {len_degrees_list}')
#
#         fig_title = f'{p_name} {vis_field} average thresholds\nRicco_v{ricco_version} (bars are 95% CIs)'
#         save_name = f'{p_name}_{vis_field}_ricco_v{ricco_version}_len_deg_v_thr.png'
#         plot_ave_w_errors_markers(fig_df=wide_df, error_df=wide_err_df,
#                                   jitter=False, error_caps=True, alt_colours=False,
#                                   legend_names=sep_name_list,
#                                   even_spaced_x=False,
#                                   fixed_y_range=False,
#                                   x_tick_vals=len_degrees_list,
#                                   x_tick_labels=[round(i, 2) for i in len_degrees_list],
#                                   x_axis_label='Length (degrees)',
#                                   y_axis_label='Threshold',
#                                   log_log_axes=False,
#                                   neg1_slope=False,
#                                   fig_title=fig_title, save_name=save_name,
#                                   save_path=os.path.join(exp_path, p_name), verbose=True)
#         plt.show()
#
#         # fig 2 - log(len len_deg), log(contrast)
#         wide_df = fig_df.pivot(index=['diag_deg'], columns='cond', values='delta_I')
#         print(f'wide_df:\n{wide_df}')
#
#         wide_err_df = error_df.pivot(index=['diag_deg'], columns='cond', values='delta_I')
#         print(f'wide_err_df:\n{wide_err_df}')
#
#
#         fig_title = f'{p_name} {vis_field} average log(degrees), log(∆I) thresholds\nRicco_v{ricco_version} (bars are 95% CIs)'
#         save_name = f'{p_name}_{vis_field}_ricco_v{ricco_version}_log_deg_log_contrast.png'
#         plot_ave_w_errors_markers(fig_df=wide_df, error_df=wide_err_df,
#                                   jitter=False, error_caps=True, alt_colours=False,
#                                   legend_names=sep_name_list,
#                                   even_spaced_x=False,
#                                   fixed_y_range=False,
#                                   x_tick_vals=None,
#                                   x_tick_labels=None,
#                                   x_axis_label='log(length, degrees)',
#                                   y_axis_label='Contrast: log(∆I)',
#                                   log_log_axes=True,
#                                   neg1_slope=True,
#                                   slope_ycol_name='lines',
#                                   slope_xcol_idx_depth=1,
#                                   fig_title=fig_title, save_name=save_name,
#                                   save_path=os.path.join(exp_path, p_name), verbose=True)
#         plt.show()

print('\nRicco_Bloch_analysis_pipe_UVF_LVF finished\n')
