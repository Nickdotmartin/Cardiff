import os

import pandas as pd

from exp1a_psignifit_analysis import a_data_extraction, b3_plot_staircase
from exp1a_psignifit_analysis import c_plots, d_average_participant
from psignifit_tools import get_psignifit_threshold_df

# # loop through run folders with first 4 scripts (a, get_psignifit_threshold_df, b3, c)
# # then run script d to get master lists and averages

# root_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/Kim_split_runs'
root_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/Kim_split_runs/Nick_practice'
participant_name = 'Kim'
# run_folder_names = ['P6a-Kim', 'P6b-Kim', 'P6c-Kim', 'P6d-Kim', 'P6e-Kim', 'P6f-Kim']
run_folder_names = ['P6a-Kim']
# participant_name = 'Kim'

group_list = [1, 2]

for run_idx, run_dir in enumerate(run_folder_names):

    print(f'\nrunning analysis for {participant_name}, {run_dir}, {participant_name}{run_idx+1}\n')
    save_path = f'{root_path}{os.sep}{run_dir}'

    # don't delete this (participant_name = participant_name),
    # needed to ensure names go name1, name2, name3 not name1, name12, name123
    p_name = participant_name

    # '''a'''
    p_name = f'{participant_name}{run_idx+1}'
    isi_list = [-1, 0, 2, 4, 6, 9, 12, 24]
    a_data_extraction(p_name=p_name, run_dir=save_path, isi_list=isi_list, verbose=True)

    run_data_path = f'{save_path}{os.sep}RUNDATA-sorted.xlsx'

    run_data_df = pd.read_excel(run_data_path, engine='openpyxl',
                                usecols=['ISI', 'Separation', 'group',
                                         'probeLum', 'trial_response'])
    print(f"run_data_df:\n{run_data_df}")
    # todo: change get_psignifit_threshold_df to take csv as input so I can use RUNDATA rather than raw data.
    for group in group_list:

        group_df = run_data_df[run_data_df['group'] == group]
        print(f"group_df:\n{list(group_df.columns)}\n{group_df}")

        '''get psignifit thresholds df'''
        thr_df = get_psignifit_threshold_df(root_path=root_path,
                                            p_run_name=run_dir,
                                            # csv_name=p_name,
                                            csv_name=group_df,
                                            n_bins=10, q_bins=True,
                                            isi_list=isi_list,
                                            sep_list=[18, 6, 3, 2, 1, 0, 99],
                                            # sep_list=[18, 18, 6, 6, 3, 3, 2, 2, 1, 1, 0, 0, 99, 99],
                                            group=group,
                                            verbose=True)


        '''b3'''
        # run_data_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/Kim_split_runs/' \
        #                 'Nick_practice/P6a-Kim/RUNDATA-sorted.xlsx'
        b3_plot_staircase(run_data_path, show_plots=False)

        '''c'''
        c_plots(save_path=save_path, show_plots=False)

'''d'''
root_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/Kim_psignifit'  # master folder containing all runs
run_folder_names = ['P6a-Kim', 'P6b-Kim', 'P6c-Kim', 'P6d-Kim', 'P6e-Kim', 'P6f-Kim']
d_average_participant(root_path=root_path, run_dir_names_list=run_folder_names,
                      show_plots=True, trimmed_mean=False, error_bars='SE')
print('exp1a_analysis_pipe finished')
