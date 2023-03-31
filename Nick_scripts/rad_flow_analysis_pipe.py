import os
import pandas as pd
import numpy as np
from psignifit_tools import get_psignifit_threshold_df
from python_tools import switch_path
from rad_flow_psignifit_analysis import a_data_extraction, b3_plot_staircase
from rad_flow_psignifit_analysis import c_plots, d_average_participant, make_average_plots, e_average_exp_data
from exp1a_psignifit_analysis import plt_heatmap_row_col

# # loop through run folders with first 5 scripts (a, b1, b2, b3, c)
# # then run script d to get master lists and averages
# exp_path = r'C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\radial_flow_exp'
# participant_list = ['Kim', 'Nick', 'Simon']  # , 'Nick_half_speed']

# exp_path = r'C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\rad_flow_half'
# participant_list = ['Nick_half', 'Simon_half']
# participant_list = ['Nick_half']  # , 'Nick']  # , 'Nick_half_speed']

# # todo: why does a_extract data work for my data but not Simon's???
# exp_path = r'C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\rad_flow_2'
# participant_list = ['Nick', 'Simon']  # , 'Nick_half_speed']

# exp_path = r'C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\rad_flow_2_half'
# participant_list = ['Nick_half_speed', 'Simon_half']  # , 'Nick_half_speed']

exp_path = r'C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\rad_flow_2_350'
participant_list = ['Simon']

exp_path = os.path.normpath(exp_path)
convert_path1 = os.path.normpath(exp_path)
convert_path1 = switch_path(convert_path1, 'windows_oneDrive')
exp_path = convert_path1

# stair_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# isi_vals_list = [1, 4, 6, 9]
# isi_names_list = [f'ISI_{i}' for i in isi_vals_list]


verbose = True
show_plots = True

n_runs = 12
# if the first folder to analyse is 1, p_idx_plus = 1.  If the forst folder is 5, use 5 etc.
p_idx_plus = 1

for p_idx, participant_name in enumerate(participant_list):
    # if participant_name is 'Nick':
    #     p_idx_plus = 5

    root_path = os.path.join(exp_path, participant_name)

    # # manually get run_folder_names with n_runs
    # run_folder_names = [f'{participant_name}_{i+p_idx_plus}' for i in list(range(n_runs))]

    # # search to automatically get run_folder_names
    dir_list = os.listdir(root_path)
    run_folder_names = []
    for i in range(12):  # numbers 0 to 11
        check_dir = f'{participant_name}_{i+p_idx_plus}'   # numbers 1 to 12
        if check_dir in dir_list:
            run_folder_names.append(check_dir)

    print(f'run_folder_names: {run_folder_names}')

    for run_idx, run_dir in enumerate(run_folder_names):

        # add run number , e.g., add five to access Nick_5 on the zeroth iteration
        r_idx_plus = run_idx + p_idx_plus

        print(f'\nrun_idx {run_idx+1}: running analysis for '
              f'{participant_name}, {run_dir}, {participant_name}_{r_idx_plus}')
        save_path = f'{root_path}{os.sep}{run_dir}'

        # # search to automatically get updated isi_list
        dir_list = os.listdir(save_path)
        run_isi_list = []
        # for isi in isi_vals_list:
        for isi in list(range(-2, 18, 1)):
            check_dir = f'ISI_{isi}_probeDur2'
            if check_dir in dir_list:
                run_isi_list.append(isi)
        run_isi_names_list = [f'ISI_{i}' for i in run_isi_list]

        print(f'run_isi_list: {run_isi_list}')

        # don't delete this (p_name = participant_name),
        # needed to ensure names go name1, name2, name3 not name1, name12, name123
        p_name = participant_name
#
#         '''a'''
#         p_name = f'{participant_name}_{r_idx_plus}'
#
#         a_data_extraction(p_name=p_name, run_dir=save_path, isi_list=run_isi_list, verbose=verbose)
#
#         run_data_path = f'{save_path}{os.sep}ALL_ISIs_sorted.xlsx'
#         run_data_df = pd.read_excel(run_data_path, engine='openpyxl',
#                                     # usecols=["ISI", "stair", "stair_name",
#                                     #          "step", "separation", "congruent",
#                                     #          "flow_dir", "probe_jump", "corner",
#                                     #          "newLum", "trial_response"]
#                                     )
#         print(f"run_data_df: {run_data_df.columns.to_list()}\n{run_data_df}")
#
#         print(f"\nsort order of stair_names, separations(double) and cong_list (tiled)")
#
#         stair_names_list = sorted(run_data_df['stair_name'].unique(), reverse=True)
#         # print(f'stair_names_list: {stair_names_list}')
#
#         # sort stairnames in same order as sorted(abs(stairnames))
#         get_order_array = [i + .5 if i > 0 else i for i in stair_names_list]
#         get_order_array = [abs(i) for i in get_order_array]
#         get_order_array = np.array(get_order_array)
#         print(f"\nget_order_array: {get_order_array}")
#         sort_index = np.argsort(get_order_array)
#         reversed_sort_index = sort_index[::-1]
#         print(f"reversed_sort_index: {reversed_sort_index}")
#
#         stair_names_list = [stair_names_list[i] for i in reversed_sort_index]
#         print(f'\nstair_names_list: {stair_names_list}')
#
#         stair_list = list(range(len(stair_names_list)))
#
#
#         # pos_sep_list = sorted(run_data_df['separation'].unique(), reverse=True)
#         pos_sep_list = run_data_df['separation'].unique()
#         # print(f'pos_sep_list: {pos_sep_list}')
#         # second_sep_list = pos_sep_list
#         # print(f'second_sep_list: {second_sep_list}')
#         # sort sep_list in same order as sorted(abs(stairnames))
#         double_sep_list = sorted([item for sublist in zip(pos_sep_list, pos_sep_list) for item in sublist],
#                                  reverse=True)
#         print(f'double_sep_list: {double_sep_list}')
#
#         cong_list = sorted(run_data_df['congruent'].unique(), reverse=True)
#         # print(f'cong_list: {cong_list}')
#         cong_list = list(np.tile(cong_list, len(pos_sep_list)))
#         print(f'cong_list: {cong_list}')
#
#
#
#         '''get psignifit thresholds df'''
#         # cols_to_add_dict = {'stair_names': [18, -18, 6, -6, 3, -3, 2, -2, 1, -1, 0, -0.1],
#         #                     'congruent':  [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
#         #                     'separation': [18, 18, 6, 6, 3, 3, 2, 2, 1, 1, 0, 0]}
#         cols_to_add_dict = {'stair_names': stair_names_list,
#                             'congruent':  cong_list,
#                             'separation': double_sep_list}
#         thr_df = get_psignifit_threshold_df(root_path=root_path,
#                                             p_run_name=run_dir,
#                                             csv_name=run_data_df,
#                                             n_bins=9, q_bins=True,
#                                             sep_col='stair',
#                                             thr_col='probeLum',
#                                             isi_list=run_isi_list,
#                                             sep_list=stair_list,
#                                             conf_int=True,
#                                             thr_type='Bayes',
#                                             plot_both_curves=False,
#                                             cols_to_add_dict=cols_to_add_dict,
#                                             show_plots=False,
#                                             verbose=verbose)
#         print(f'thr_df:\n{thr_df}')
#
#
#         '''b3'''
#         b3_plot_staircase(run_data_path, thr_col='probeLum', show_plots=show_plots, verbose=verbose)
#
#         '''c I don't actually need any of these, instead sort get psignifit thr ands make plots from those.'''
#         c_plots(save_path=save_path, thr_col='probeLum', isi_name_list=run_isi_names_list, show_plots=show_plots, verbose=verbose)


    '''d participant averages'''
    trim_n = None
    if len(run_folder_names) == 12:
        trim_n = 2
    print(f'\ntrim_n: {trim_n}')

    d_average_participant(root_path=root_path, run_dir_names_list=run_folder_names,
                          trim_n=trim_n, error_type='SE', verbose=verbose)
    # groupby_col = None, cols_to_drop = None, cols_to_replace = None,




    # making average plot
    all_df_path = os.path.join(root_path, f'MASTER_TM{trim_n}_thresholds.csv')
    p_ave_path = os.path.join(root_path, f'MASTER_ave_TM{trim_n}_thresh.csv')
    err_path = os.path.join(root_path, f'MASTER_ave_TM{trim_n}_thr_error_SE.csv')
    if trim_n is None:
        all_df_path = os.path.join(root_path, f'MASTER_psignifit_thresholds.csv')
        p_ave_path = os.path.join(root_path, 'MASTER_ave_thresh.csv')
        err_path = os.path.join(root_path, 'MASTER_ave_thr_error_SE.csv')
    exp_ave = False

    # todo: change plots to p_name at top
    make_average_plots(all_df_path=all_df_path,
                       ave_df_path=p_ave_path,
                       error_bars_path=err_path,
                       thr_col='probeLum',
                       n_trimmed=trim_n,
                       ave_over_n=len(run_folder_names),
                       # exp_ave=False,
                       exp_ave=participant_name,
                       show_plots=True, verbose=True)



# print(f'exp_path: {exp_path}')
# # participant_list = ['Nick_half', 'Simon_half']
# print('\nget exp_average_data')
# trim_n = 2
# # todo: sort script to automatically use trim=2 if its there, and not just use untrimmed#
# # todo: make sure ISI cols are in the correct order
# e_average_exp_data(exp_path=exp_path, p_names_list=participant_list,
#                    error_type='SE', n_trimmed=trim_n, verbose=True)
#
#
# all_df_path = os.path.join(exp_path, "MASTER_exp_thr.csv")
# exp_ave_path = os.path.join(exp_path, "MASTER_exp_ave_thr.csv")
# err_path = os.path.join(exp_path, "MASTER_ave_thr_error_SE.csv")
#
# all_df = pd.read_csv(all_df_path)
# if any("Unnamed" in i for i in list(all_df.columns)):
#     unnamed_col = [i for i in list(all_df.columns) if "Unnamed" in i][0]
#     all_df.drop(unnamed_col, axis=1, inplace=True)
# print(f"all_df:\n{all_df}")
#
# all_df_basic_cols = ['participant', 'stair_names', 'congruent', 'separation']
#
# isi_names_list = list(all_df.columns[len(all_df_basic_cols):])
# isi_vals_list = [int(i[4:]) for i in isi_names_list]
#
# # sort isi_names_list by sorted(isi_vals_list) order
# isi_vals_array = np.array(isi_vals_list)
# print(f"\nisi_vals_array: {isi_vals_array}")
# sort_index = np.argsort(isi_vals_array)
# print(f"sort_index: {sort_index}")
#
# isi_vals_list = [isi_vals_list[i] for i in sort_index]
# print(f"isi_vals_list: {isi_vals_list}")
#
# isi_names_list = [isi_names_list[i] for i in sort_index]
# print(f"isi_names_list: {isi_names_list}")
#
# all_col_names = all_df_basic_cols + isi_names_list
# print(f"all_col_names: {all_col_names}")
#
# all_df = all_df[all_col_names]
# print(f"all_df:\n{all_df}")
# all_df.to_csv(all_df_path, index=False)
#
#
#
# ave_df = pd.read_csv(exp_ave_path)
# if any("Unnamed" in i for i in list(ave_df.columns)):
#     unnamed_col = [i for i in list(ave_df.columns) if "Unnamed" in i][0]
#     ave_df.drop(unnamed_col, axis=1, inplace=True)
# print(f"ave_df:\n{ave_df}")
# ave_df_basic_cols = ['stair_names']
# ave_col_names = ave_df_basic_cols + isi_names_list
# print(f"ave_col_names: {ave_col_names}")
# ave_df = ave_df[ave_col_names]
# print(f"ave_df:\n{ave_df}")
# ave_df.to_csv(exp_ave_path, index=False)
#
# err_df = pd.read_csv(err_path)
# if any("Unnamed" in i for i in list(err_df.columns)):
#     unnamed_col = [i for i in list(err_df.columns) if "Unnamed" in i][0]
#     err_df.drop(unnamed_col, axis=1, inplace=True)
# print(f"err_df:\n{err_df}")
# err_df = err_df[ave_col_names]
# print(f"err_df:\n{err_df}")
# err_df.to_csv(err_path, index=False)
#
#
# make_average_plots(all_df_path=all_df_path,
#                    ave_df_path=exp_ave_path,
#                    error_bars_path=err_path,
#                    thr_col='probeLum',
#                    stair_names_col='stair_names',
#                    cond_type_col='congruent',
#                    cond_type_order=[1, -1],
#                    n_trimmed=trim_n,
#                    ave_over_n=len(participant_list),
#                    exp_ave=True,
#                    isi_name_list=isi_names_list,
#                    isi_vals_list=isi_vals_list,
#                    show_plots=True, verbose=True)

print('\nrad_flow_analysis_pipe finished')
