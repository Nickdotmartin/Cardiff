import os
import pandas as pd
from exp1a_psignifit_analysis import a_data_extraction, b3_plot_staircase, c_plots, \
    d_average_participant, e_average_exp_data, make_average_plots
from psignifit_tools import get_psignifit_threshold_df
from check_home_dir import switch_path

# # loop through run folders with first 4 scripts (a, get_psignifit_threshold_df, b3, c)
# # then run script d to get master lists and averages
# old_exp_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/exp1a_data'
# old_exp_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/Kim_split_runs_weighted_mean'
old_exp_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\exp1a_1probe_test\Bayesian"  # CI95
exp_path = switch_path(old_exp_path, 'wind_oneDrive')
print(f"exp_path: {exp_path}")
participant_list = ['aa', 'bb', 'cc', 'dd', 'ee']
# participant_list = ['Kim']
isi_list = [-1, 0, 2, 4, 6, 9, 12, 24]

n_runs = 6

p_idx_plus = 1

for p_idx, participant_name in enumerate(participant_list):
    root_path = os.path.join(exp_path, participant_name)

    run_folder_names = [f'{participant_name}_{i+1}' for i in list(range(n_runs))]
    print(f'run_folder_names: {run_folder_names}')

    group_list = [1, 2]

    # check whether scrips a, b3 and c have been completed for the last run (e.g., all runs) for this participant
    check_last_c_plots_fig = os.path.join(root_path, run_folder_names[-1], 'g2_dataDivOneProbe.png')

    # # todo: comment this out again?
    # if not os.path.isfile(check_last_c_plots_fig):
    #     print(f'\nNOT completed analysis yet: {check_last_c_plots_fig}')

    for run_idx, run_dir in enumerate(run_folder_names):

        # check whether scripts a, b3 and c have been done for this run for this participant
        # check_last_c_plots_fig = f'{root_path}/{run_dir}/g2_dataDivOneProbe.png'
        check_last_c_plots_fig = os.path.join(root_path, run_dir, 'g2_dataDivOneProbe.png')

        # if os.path.isfile(check_last_c_plots_fig):
        #     print(f'\nalready completed: {check_last_c_plots_fig}')
        #     continue

        print(f'\nrunning analysis for {participant_name}, {run_dir}, {participant_name}{run_idx+1}\n')
        save_path = f'{root_path}{os.sep}{run_dir}'

        # don't delete this (participant_name = participant_name),
        # needed to ensure names go name1, name2, name3 not name1, name12, name123
        p_name = participant_name

        # # '''a'''
        # # p_name = f'{participant_name}_{run_idx+1}_output'  # use this one
        p_name = f'{participant_name}_output'  # use this one for exp1a
        # p_name = f'{participant_name}{run_idx+1}'  # use this one for copy of Kim's data
        isi_list = [-1, 0, 2, 4, 6, 9, 12, 24]
        #
        # # # # for first run, some files are saved just as name not name1
        # # # check_file = os.path.join(save_path, 'ISI_-1_probeDur2', f'{participant_name}_output.csv')
        # # #
        # # # if not os.path.isfile(check_file):
        # # #     raise FileNotFoundError(check_file)
        #
        # run_data_df = a_data_extraction(p_name=p_name, run_dir=save_path, isi_list=isi_list, verbose=True)
        #
        # # todo: can get rid of this once all RUN-data has newLum column
        # run_data_path = os.path.join(save_path, 'RUNDATA-sorted.xlsx')
        # run_data_df = pd.read_excel(run_data_path, engine='openpyxl')
        #
        # '''add newLum column
        # in old version, the experiment script varies probeLum and converts to float(RGB255) values for screen.
        # However, monitor can only use int(RGB255).
        # This function will will round RGB255 values to int(RGB255), then convert to NEW_probeLum
        # LumColor255Factor = 2.395387069
        # 1. get probeColor255 column.
        # 2. convert to int(RGB255) and convert to new_Lum with int(RGB255)/LumColor255Factor
        # 3. add to run_data_df'''
        # if 'newLum' not in run_data_df.columns.to_list():
        #     LumColor255Factor = 2.395387069
        #     rgb255_col = run_data_df['probeColor255'].to_list()
        #     newLum = [int(i) / LumColor255Factor for i in rgb255_col]
        #     run_data_df.insert(9, 'newLum', newLum)
        #     run_data_df.to_excel(os.path.join(save_path, 'RUNDATA-sorted.xlsx'), index=False)
        #     print(f"added newLum column\n"
        #           f"run_data_df: {run_data_df.columns.to_list()}")
        #
        #
        # # run_data_path = f'{save_path}{os.sep}RUNDATA-sorted.xlsx'
        # run_data_path = os.path.join(save_path, 'RUNDATA-sorted.xlsx')
        #
        # run_data_df = pd.read_excel(run_data_path, engine='openpyxl',
        #                             usecols=['ISI',
        #                                      'stair',
        #                                      'separation', 'group',
        #                                      'newLum', 'trial_response'])
        # print(f"run_data_df:\n{run_data_df}")
        #
        #
        # # stair_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        # # cols_to_add_dict = {'group': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        # #                     'separation': [18, 18, 6, 6, 3, 3, 2, 2, 1, 1, 0, 0, 20, 20]}
        # stair_list = [13, 14]
        # cols_to_add_dict = {'group': [1, 2],
        #                     'separation': [20, 20]}
        #
        # '''get psignifit thresholds df - use stairs as sep levels rather than using groups'''
        # thr_df = get_psignifit_threshold_df(root_path=root_path,
        #                                     p_run_name=run_dir,
        #                                     csv_name=run_data_df,
        #                                     n_bins=9, q_bins=True,
        #                                     sep_col='stair',
        #                                     thr_col='newLum',
        #                                     thr_type='CI95',
        #                                     plot_both_curves=True,
        #                                     isi_list=isi_list,
        #                                     sep_list=stair_list,
        #                                     conf_int=True,
        #                                     save_plots=True,
        #                                     cols_to_add_dict=cols_to_add_dict,
        #                                     verbose=True)
        # print(f'thr_df:\n{thr_df}')
        #
        #
        # '''b3'''
        # # run_data_path = f'{save_path}{os.sep}RUNDATA-sorted.xlsx'
        # run_data_path = os.path.join(save_path, 'RUNDATA-sorted.xlsx')
        # thr_path = os.path.join(save_path, 'psignifit_thresholds.csv')
        #
        # b3_plot_staircase(run_data_path, thr_col='newLum', show_plots=False)
        #
        # '''c'''
        # c_plots(save_path=save_path, thr_col='newLum', show_plots=True)




    # trim_n = None  # 1
    # use_trimmed = False
    # if len(run_folder_names) == 6:
    #     trim_n = 1
    #     use_trimmed = True
    # n_trimmed = trim_n
    #
    # print(f"\n\ntrim_n: {trim_n}, "
    #       f"use_trimmed: {use_trimmed}\n\n")

    for n_trimmed in [1, 2, 3, 4, 5]:

        '''d'''
        d_average_participant(root_path=root_path, run_dir_names_list=run_folder_names,
                              trim_n=n_trimmed, error_type='SE')

    # all_df_path = os.path.join(root_path, f'MASTER_TM{n_trimmed}_thresholds.csv')
    # p_ave_path = os.path.join(root_path, f'MASTER_ave_TM{n_trimmed}_thresh.csv')
    # err_path = os.path.join(root_path, f'MASTER_ave_TM{n_trimmed}_thr_error_SE.csv')
    # n_trimmed = trim_n
    # if n_trimmed is None:
    #     all_df_path = os.path.join(root_path, 'MASTER_psignifit_thresholds.csv')
    #     p_ave_path = os.path.join(root_path, 'MASTER_ave_thresh.csv')
    #     err_path = os.path.join(root_path, 'MASTER_ave_thr_error_SE.csv')
    #
    # exp_ave = False
    #
    # make_average_plots(all_df_path=all_df_path,
    #                    ave_df_path=p_ave_path,
    #                    error_bars_path=err_path,
    #                    thr_col='newLum',
    #                    error_type='SE',
    #                    n_trimmed=n_trimmed,
    #                    exp_ave=False,
    #                    show_plots=True, verbose=True)


# print(f'exp_path: {exp_path}')
# print('\nget exp_average_data')
#
# e_average_exp_data(exp_path=exp_path, p_names_list=participant_list,
#                    error_type='SE', use_trimmed=use_trimmed, verbose=True)
#
#
#
# all_df_path = os.path.join(exp_path, 'MASTER_exp_thr.csv')
# exp_ave_path = os.path.join(exp_path, 'MASTER_exp_ave_thr.csv')
# err_path = os.path.join(exp_path, 'MASTER_ave_thr_error_SE.csv')
#
# n_trimmed = None
# exp_ave = True
#
# make_average_plots(all_df_path=all_df_path,
#                    ave_df_path=exp_ave_path,
#                    error_bars_path=err_path,
#                    error_type='SE',
#                    n_trimmed=n_trimmed,
#                    exp_ave=exp_ave,
#                    show_plots=True, verbose=True)

print('\nexp1a_analysis_pipe finished\n')
