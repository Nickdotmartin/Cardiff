import os

import pandas as pd

from exp1a_psignifit_analysis import a_data_extraction, b3_plot_staircase, c_plots, \
    d_average_participant, e_average_exp_data, make_average_plots
from psignifit_tools import get_psignifit_threshold_df

# # loop through run folders with first 4 scripts (a, get_psignifit_threshold_df, b3, c)
# # then run script d to get master lists and averages
exp_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/exp1a_data'
participant_list = ['aa', 'bb', 'cc', 'dd', 'ee']
n_runs = 3

p_idx_plus = 1

for p_idx, participant_name in enumerate(participant_list):
    root_path = f'{exp_path}/{participant_name}'
    # run_folder_names = [f'P{p_idx + p_idx_plus}a-{participant_name}', f'P{p_idx + p_idx_plus}b-{participant_name}',
    #                     f'P{p_idx + p_idx_plus}c-{participant_name}', f'P{p_idx + p_idx_plus}d-{participant_name}',
    #                     f'P{p_idx + p_idx_plus}e-{participant_name}', f'P{p_idx + p_idx_plus}f-{participant_name}']
    # run_folder_names = [f'{participant_name}_1', f'{participant_name}_2',
    #                     f'{participant_name}_3', f'{participant_name}_4',
    #                     f'{participant_name}_5', f'{participant_name}_6']
    run_folder_names = [f'{participant_name}_{i+1}' for i in list(range(n_runs))]
    print(f'run_folder_names: {run_folder_names}')

    group_list = [1, 2]

    # check whether scrips a, b3 and c have been completed for the last run (e.g., all runs) for this participant
    check_last_c_plots_fig = f'{root_path}/{run_folder_names[-1]}/g2_dataDivOneProbe.png'

    if not os.path.isfile(check_last_c_plots_fig):
        print(f'\nNOT completed analysis yet: {check_last_c_plots_fig}')

        for run_idx, run_dir in enumerate(run_folder_names):

            # check whether scripts a, b3 and c have been done for the this run for this participant
            check_last_c_plots_fig = f'{root_path}/{run_dir}/g2_dataDivOneProbe.png'
            if os.path.isfile(check_last_c_plots_fig):
                print(f'\nalready completed: {check_last_c_plots_fig}')
                continue

            print(f'\nrunning analysis for {participant_name}, {run_dir}, {participant_name}{run_idx+1}\n')
            save_path = f'{root_path}{os.sep}{run_dir}'

            # don't delete this (participant_name = participant_name),
            # needed to ensure names go name1, name2, name3 not name1, name12, name123
            p_name = participant_name

            # '''a'''
            p_name = f'{participant_name}_{run_idx+1}_output'
            # p_name = f'{participant_name}{run_idx+1}'
            isi_list = [-1, 0, 2, 4, 6, 9, 12, 24]

            # for first run, some files are saved just as name not name1
            check_file = f'{save_path}{os.sep}ISI_-1_probeDur2/{p_name}.csv'
            if not os.path.isfile(check_file):
                raise FileNotFoundError(check_file)

            run_data_df = a_data_extraction(p_name=p_name, run_dir=save_path, isi_list=isi_list, verbose=True)

            run_data_path = f'{save_path}{os.sep}RUNDATA-sorted.xlsx'

            run_data_df = pd.read_excel(run_data_path, engine='openpyxl',
                                        usecols=['ISI',
                                                 'stair',
                                                 'separation', 'group',
                                                 'probeLum', 'trial_response'])
            print(f"run_data_df:\n{run_data_df}")

            stair_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
            cols_to_add_dict = {'group': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                                'separation': [18, 18, 6, 6, 3, 3, 2, 2, 1, 1, 0, 0, 20, 20]}

            '''get psignifit thresholds df - use stairs as sep levels rather than using groups'''
            thr_df = get_psignifit_threshold_df(root_path=root_path,
                                                p_run_name=run_dir,
                                                csv_name=run_data_df,
                                                n_bins=10, q_bins=True,
                                                sep_col='stair',
                                                isi_list=isi_list,
                                                sep_list=stair_list,
                                                cols_to_add_dict=cols_to_add_dict,
                                                verbose=True)
            print(f'thr_df:\n{thr_df}')


            '''b3'''
            run_data_path = f'{save_path}{os.sep}RUNDATA-sorted.xlsx'
            b3_plot_staircase(run_data_path, show_plots=True)

            '''c'''
            c_plots(save_path=save_path, show_plots=True)

    '''d'''
    d_average_participant(root_path=root_path, run_dir_names_list=run_folder_names,
                          trim_n=1, error_type='SE')

    all_df_path = f'{root_path}/MASTER_TM1_thresholds.csv'
    p_ave_path = f'{root_path}/MASTER_ave_TM_thresh.csv'
    err_path = f'{root_path}/MASTER_ave_TM_thr_error_SE.csv'
    n_trimmed = 1
    exp_ave = False

    make_average_plots(all_df_path=all_df_path,
                       ave_df_path=p_ave_path,
                       error_bars_path=err_path,
                       error_type='SE',
                       n_trimmed=n_trimmed,
                       exp_ave=False,
                       show_plots=True, verbose=True)


print(f'exp_path: {exp_path}')
print('\nget exp_average_data')

e_average_exp_data(exp_path=exp_path, p_names_list=participant_list,
                   error_type='SE', use_trimmed=True, verbose=True)


all_df_path = f'{exp_path}/MASTER_exp_thr.csv'
exp_ave_path = f'{exp_path}/MASTER_exp_ave_thr.csv'
err_path = f'{exp_path}/MASTER_ave_thr_error_SE.csv'
n_trimmed = None
exp_ave=True

make_average_plots(all_df_path=all_df_path,
                   ave_df_path=exp_ave_path,
                   error_bars_path=err_path,
                   error_type='SE',
                   n_trimmed=n_trimmed,
                   exp_ave=exp_ave,
                   show_plots=True, verbose=True)

print('\nexp1a_analysis_pipe finished\n')
