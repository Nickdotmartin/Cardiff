import os
import pandas as pd
import datetime
from exp1a_psignifit_analysis import a_data_extraction, b3_plot_staircase, c_plots, \
    d_average_participant, e_average_exp_data, make_average_plots, make_diff_from_conc_df
from psignifit_tools import get_psignifit_threshold_df
from python_tools import which_path, running_on_laptop, switch_path

project_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff"

# # loop through run folders with first 4 scripts (a, get_psignifit_threshold_df, b3, c)
# # then run script d to get master lists and averages
# eyetracking, jitter_rgb, EXP1_split_probes, Exp1_double_dist, EXP1_sep4_5
# Exp4b_missing_probe\rotation, incoherent, radial, rotation, translation,
# this_exp = r"EXP1_Jan23_rept_dropped"  EXP1_split_probes_Jan23
# this_exp = r"Exp4b_missing_probe_23/Exp4b_missing_probe_23/mixed_dir/incoherent/"
this_exp = r"exp1a_data"
exp_path = os.path.join(project_path, this_exp)
convert_path1 = os.path.normpath(exp_path)
if running_on_laptop():
    convert_path1 = switch_path(convert_path1, 'mac_oneDrive')
exp_path = convert_path1

print(f"exp_path: {exp_path}")
participant_list = ['Nick']  # 'Nick_sep0123', 'Nick_sep45', 'Nick_sep67', 'Nick_sep89', 'Nick_sep18_20']  #  'Simon', 'Nick'
# participant_list = ['p1', 'p2']

split_1probe = False
n_runs = 12

analyse_from_run = 1
trim_list = []

for p_idx, participant_name in enumerate(participant_list):
    root_path = os.path.join(exp_path, participant_name)

    # search to automatically get run_folder_names
    dir_list = os.listdir(root_path)
    run_folder_names = []
    for i in range(n_runs):  # numbers 0 to 11
        check_dir = f'{participant_name}_{i+analyse_from_run}'   # numbers 1 to 12
        if check_dir in dir_list:
            run_folder_names.append(check_dir)

    if len(run_folder_names) > 0:
        print("running analysis for:")
        for i in run_folder_names:
            print(i)
    else:
        print("no run folders found")

    trim_n = None
    if len(run_folder_names) == 12:
        trim_n = 2
    elif len(run_folder_names) > 12:
        # trim_n = 2
        if len(run_folder_names) % 2 == 0:  # if even
            trim_n = int((len(run_folder_names)-12)/2)
        else:
            raise ValueError(f"for this exp you have {len(run_folder_names)} runs, set rules for trimming.")
    trim_list.append(trim_n)

    for run_idx, run_dir in enumerate(run_folder_names):

        r_idx_plus = run_idx + analyse_from_run

        print(f'\nrunning analysis for {participant_name}, {run_dir}, {participant_name}{r_idx_plus}\n')
        # print(f'\nrunning analysis for {participant_name}\n')

        save_path = os.path.join(root_path, run_dir)

        # don't delete this (participant_name = participant_name),
        # needed to ensure names go name1, name2, name3 not name1, name12, name123
        p_name = participant_name
#
#         # # '''a'''
#         p_name = f'{participant_name}_output'  # use this one
#
#         # # I don't need data extraction as all ISIs are in same df.
#         try:
#             run_data_df = pd.read_csv(os.path.join(save_path, f'{p_name}.csv'))
#         except:
#             p_name = f'{participant_name}_{r_idx_plus}_output'  # use this one
#             run_data_df = pd.read_csv(os.path.join(save_path, f'{p_name}.csv'))
#
#         try:
#             run_data_df = run_data_df.sort_values(by=['stair', 'total_nTrials'])
#         except KeyError:
#             run_data_df = run_data_df.sort_values(by=['stair', 'trial_number'])
#
#         if run_data_df['probeColor255'].dtypes == 'int64':
#             lum_col = 'probeLum'
#             print(f"probeColor255 is {run_data_df['probeColor255'].dtypes}, lum_col is {lum_col}")
#         else:
#             '''add newLum column
#                     in old version, the experiment script varies probeLum and converts to float(RGB255) values for screen.
#                     However, monitor can only use int(RGB255).
#                     This function will will round RGB255 values to int(RGB255), then convert to NEW_probeLum
#                     LumColor255Factor = 2.395387069
#                     1. get probeColor255 column.
#                     2. convert to int(RGB255) and convert to new_Lum with int(RGB255)/LumColor255Factor
#                     3. add to run_data_df'''
#             lum_col = 'newLum'
#             print(f"probeColor255 is {run_data_df['probeColor255'].dtypes}, lum_col is {lum_col}")
#             if 'newLum' not in run_data_df.columns.to_list():
#                 LumColor255Factor = 2.395387069
#                 rgb255_col = run_data_df['probeColor255'].to_list()
#                 newLum = [int(i) / LumColor255Factor for i in rgb255_col]
#                 run_data_df.insert(9, 'newLum', newLum)
#                 print(f"added newLum column\n"
#                       f"run_data_df: {run_data_df.columns.to_list()}")
#
#         # remove unnamed columns
#         substring = 'Unnamed: '
#         unnamed_cols = [i for i in run_data_df.columns.to_list() if substring in i]
#         print(f"unnamed_cols: {unnamed_cols}")
#
#         for col_name in unnamed_cols:
#             run_data_df.drop(col_name, axis=1, inplace=True)
#
#         print(f"run_data_df:\n{run_data_df}")
#         sep_list = sorted(list(run_data_df['separation'].unique()))
#         print(f"sep_list: {sep_list}")
#
#         isi_list = sorted(list(run_data_df['ISI'].unique()))
#         # make sure concurrent (-1) is the first value, not zero
#         if -1 in isi_list:
#             if isi_list[0] != -1:
#                 isi_list.remove(-1)
#                 isi_list = [-1] + isi_list
#         print(f"isi_list: {isi_list}")
#
#
#         """There was an error in all experimental scripts that mean that
#          concurrent probes (ISI=-1) were presented for three frames not two.
#          This error was resolved on all scripts on 3rd Nov.
#          If experiment was run after 3rd Nov, keep data as is, otherwise,
#          remove any ISI=-1 conditions from output file (I have a backup)
#          and then analyse the shortened output."""
#
#         if -1 in isi_list:
#
#             # I'm treating 4th Nov as first date with trustworthy data
#             conc_fixed = datetime.date(2022, 11, 4)
#
#             if 'date' in run_data_df.columns.to_list():
#                 date_col = 'date'
#             elif 'Date' in run_data_df.columns.to_list():
#                 date_col = 'Date'
#             else:
#                 print(run_data_df.columns.to_list())
#                 raise ValueError('date column not found')
#             date_str = run_data_df[date_col].iloc[0]
#             print(f"date_str: {date_str}, {type(date_str)}")
#             day, month, year = str.split(date_str, sep='/')
#             date_fmt = datetime.date(int(year), int(month), int(day))
#             print(f"date_fmt: {date_fmt}, {type(date_fmt)}")
#
#             # check if experiment was done before concurrent conditions were fixed
#             if date_fmt >= conc_fixed:
#                 print('phew, this data is all good.')
#             else:
#                 print('\nDropping Concurrent trials\n')
#
#                 run_data_df = run_data_df[run_data_df['ISI'] != -1]
#
#
#         sep_list = sorted(list(run_data_df['separation'].unique()))
#         isi_list = sorted(list(run_data_df['ISI'].unique()))
#         print(f"sep_list: {sep_list}")
#         print(f"isi_list: {isi_list}")
#
#         run_data_path = os.path.join(save_path, 'RUNDATA-sorted.xlsx')
#         run_data_df.to_excel(run_data_path, index=False)
#         print(f"run_data_df:\n{run_data_df}")
#
#         # stair_list = list(run_data_df['stair'].unique())
#         # stair_names_list = list(run_data_df['stair_name'].unique())
#         # cols_to_add_dict = {'stair': stair_list, 'stair_name': stair_names_list}
#         # print(f"cols_to_add_dict:\n{cols_to_add_dict}")
#
#         # does this exp use 1probe data?
#         if (99 in sep_list) or (20 in sep_list):
#             split_1probe = True
#
#         '''get psignifit thresholds df - use stairs as sep levels rather than using groups'''
#         thr_df = get_psignifit_threshold_df(root_path=root_path,
#                                             p_run_name=run_dir,
#                                             csv_name=run_data_df,
#                                             n_bins=9, q_bins=True,
#                                             sep_col='separation',
#                                             sep_list=sep_list,
#                                             thr_col=lum_col,
#                                             isi_list=isi_list,
#                                             conf_int=True,
#                                             thr_type='Bayes',
#                                             plot_both_curves=False,
#                                             save_plots=False,
#                                             cols_to_add_dict=None,
#                                             verbose=True)
#         print(f'thr_df:\n{thr_df}')
#
#
#
    # print(f"\n\ntrim_list: {trim_list}, trim_n: {trim_n}\n\n")
    '''d'''
    # d_average_participant(root_path=root_path, run_dir_names_list=run_folder_names,
    #                       trim_n=trim_n, error_type='SE')

    all_df_path = os.path.join(root_path, f'MASTER_TM{trim_n}_thresholds.csv')
    p_ave_path = os.path.join(root_path, f'MASTER_ave_TM{trim_n}_thresh.csv')
    err_path = os.path.join(root_path, f'MASTER_ave_TM{trim_n}_thr_error_SE.csv')
    if trim_n is None:
        all_df_path = os.path.join(root_path, 'MASTER_psignifit_thresholds.csv')
        p_ave_path = os.path.join(root_path, 'MASTER_ave_thresh.csv')
        err_path = os.path.join(root_path, 'MASTER_ave_thr_error_SE.csv')


    ave_DfC_df, error_DfC_df = make_diff_from_conc_df(all_df_path, root_path, n_trimmed=trim_n)
    print(f"ave_DfC_df:\n{ave_DfC_df}")
    print(f"error_DfC_df:\n{error_DfC_df}")

    p_ave_df = pd.read_csv(p_ave_path)
    print(f"p_ave_df:\n{p_ave_df}")

    isi_vals_list = [int(i[4:]) for i in list(p_ave_df.columns)[1:]]
    if -1 in isi_vals_list:
        if isi_vals_list[0] != -1:
            isi_vals_list.remove(-1)
            isi_vals_list = [-1] + isi_vals_list
    print(isi_vals_list)

    isi_name_list = [f"conc" if i == -1 else f"ISI_{i}" for i in isi_vals_list]

    sep_vals_list = list(p_ave_df['separation'])
    sep_name_list = [f"1probe" if i == 20 else i for i in sep_vals_list]
    if '1probe' in sep_name_list:
        split_1probe = True
    print(f"isi_name_list:\n{isi_name_list}")
    print(f"isi_vals_list:\n{isi_vals_list}")
    print(f"sep_vals_list:\n{sep_vals_list}")
    print(f"sep_name_list:\n{sep_name_list}")

    make_average_plots(all_df_path=all_df_path,
                       ave_df_path=p_ave_path,
                       error_bars_path=err_path,
                       thr_col='newLum',
                       error_type='SE',
                       n_trimmed=trim_n,
                       ave_over_n=len(run_folder_names),
                       split_1probe=split_1probe,
                       isi_name_list=isi_name_list,
                       sep_vals_list=sep_vals_list,
                       sep_name_list=sep_name_list,
                       exp_ave=participant_name,  # participant ave, not exp ave
                       heatmap_annot_fmt='.0f',  # use '.3g' for 3 significant figures, '.2f' for 2dp, '.0f' for int.
                       show_plots=True, verbose=True)

#
# print(f'exp_path: {exp_path}')
# print('\nget exp_average_data')
# # print(f'trim_list: {trim_list}')
#
# # participant_list = ['aa', 'bb', 'cc', 'dd', 'ee', 'Nick']
# # trim_list = [2, 2, 2, 2, 2, 2]
# participant_list = ['Kim', 'Kristian']
# trim_list = [2, 2]
# # lum_col = 'probeLum'
#
# e_average_exp_data(exp_path=exp_path, p_names_list=participant_list,
#                    error_type='SE', n_trimmed=trim_list, verbose=True)
#
# all_df_path = os.path.join(exp_path, 'MASTER_exp_all_thr.csv')
# exp_ave_path = os.path.join(exp_path, 'MASTER_exp_ave_thr.csv')
# err_path = os.path.join(exp_path, 'MASTER_ave_thr_error_SE.csv')
#
# ave_DfC_df, error_DfC_df = make_diff_from_conc_df(all_df_path, exp_path, n_trimmed=2, exp_ave=True)
# print(f"ave_DfC_df:\n{ave_DfC_df}")
# print(f"error_DfC_df:\n{error_DfC_df}")
# # get ISI and sep values for whole experiment (not just last participant)
# exp_ave_df = pd.read_csv(exp_ave_path)
# print(f"exp_ave_df:\n{exp_ave_df}")
#
# isi_vals_list = sorted([int(i[4:]) for i in list(exp_ave_df.columns)[1:]])
# # isi_vals_list = [-1, 0, 2, 4, 6, 9, 12, 24]
# isi_name_list = [f"conc" if i == -1 else f"ISI_{i}" for i in isi_vals_list]
#
# sep_vals_list = list(exp_ave_df['separation'])
# sep_name_list = [f"1probe" if i == 20 else i for i in sep_vals_list]
# print(f"isi_name_list:\n{isi_name_list}")
# print(f"isi_vals_list:\n{isi_vals_list}")
# print(f"sep_vals_list:\n{sep_vals_list}")
# print(f"sep_name_list:\n{sep_name_list}")
#
# make_average_plots(all_df_path=all_df_path,
#                    ave_df_path=exp_ave_path,
#                    error_bars_path=err_path,
#                    error_type='SE',
#                    ave_over_n=len(participant_list),
#                    split_1probe=split_1probe,
#                    isi_name_list=isi_name_list,
#                    sep_vals_list=sep_vals_list,
#                    sep_name_list=sep_name_list,
#                    exp_ave=True,  # participant ave, not exp ave
#                    heatmap_annot_fmt='.0f',  # use '.3g' for 3 significant figures, '.2f' for 2dp, '.0f' for int.
#                    show_plots=True, verbose=True)

print('\nexp1a_analysis_pipe finished\n')
