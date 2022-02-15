import os

import pandas as pd

from psignifit_tools import get_psignifit_threshold_df
from rad_flow_psignifit_analysis import a_data_extraction, b3_plot_staircase
from rad_flow_psignifit_analysis import c_plots, d_average_participant, make_average_plots, e_average_exp_data

# # loop through run folders with first 5 scripts (a, b1, b2, b3, c)
# # then run script d to get master lists and averages
# root_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/radial_flow_exp'
# run_folder_names = ['Nick_5', 'Nick_6', 'Nick_7', 'Nick_8', 'Nick_9', 'Nick_10']
# participant_name = 'Nick'
# isi_list = [1, 4, 6, 8, 9, 10, 12]
# isi_names_list = ['ISI_1', 'ISI_4', 'ISI_6', 'ISI_8', 'ISI_9', 'ISI_10', 'ISI_12']
# stair_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# participant_list = ['Nick']
# exp_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/radial_flow_exp'
#
# exp_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/radial_flow_exp/test_e_ave'
# participant_list = ['Test', 'Test2']
# isi_list = [1, 4, 6]
# isi_names_list = ['ISI_1', 'ISI_4', 'ISI_6']
# stair_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
#
root_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/radial_flow_exp'
exp_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/radial_flow_exp'
participant_list = ['Simon']
run_folder_names = ['Simon_7', 'Simon_8']  # ['Simon_1', 'Simon_2', 'Simon_3', 'Simon_4', 'Simon_5', 'Simon_6', 'Simon_7', 'Simon_8']
isi_list = [6, 9]
isi_names_list = ['ISI_6', 'ISI_9']
stair_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

verbose = True
show_plots = True
p_idx_plus = 1

for p_idx, participant_name in enumerate(participant_list):
    if participant_name is 'Nick':
        p_idx_plus = 5

    root_path = f'{exp_path}/{participant_name}'
    run_folder_names = [f'{participant_name}_{p_idx_plus}', f'{participant_name}_{p_idx_plus + 1}',
                        f'{participant_name}_{p_idx_plus + 2}', f'{participant_name}_{p_idx_plus + 3}',
                        f'{participant_name}_{p_idx_plus + 4}', f'{participant_name}_{p_idx_plus + 5}']
    print(f'run_folder_names: {run_folder_names}')

    for run_idx, run_dir in enumerate(run_folder_names):

        # add run number , e.g., add five to access Nick_5 on the zeroth iteration
        r_idx_plus = run_idx + 1
        if participant_name is 'Nick':
            r_idx_plus = run_idx + 5

        print(f'\nrun_idx{run_idx}: running analysis for {participant_name}, {run_dir}, {participant_name}_{r_idx_plus}\n')
        save_path = f'{root_path}{os.sep}{run_dir}'

        # don't delete this (p_name = participant_name),
        # needed to ensure names go name1, name2, name3 not name1, name12, name123
        p_name = participant_name
        run_data_path = f'{save_path}{os.sep}ALL_ISIs_sorted.xlsx'

        '''a'''
        p_name = f'{participant_name}_{r_idx_plus}'
        a_data_extraction(p_name=p_name, run_dir=save_path, isi_list=isi_list, verbose=verbose)

        run_data_df = pd.read_excel(run_data_path, engine='openpyxl',
                                    usecols=["ISI", "stair", "stair_name",
                                             "step", "separation", "congruent",
                                             "flow_dir", "probe_jump", "corner",
                                             "probeLum", "trial_response"])
        # print(f"run_data_df:\n{run_data_df}")

        '''get psignifit thresholds df'''
        cols_to_add_dict = {'stair_names': [18, -18, 6, -6, 3, -3, 2, -2, 1, -1, 0, -0.1],
                            'congruent':  [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
                            'separation': [18, 18, 6, 6, 3, 3, 2, 2, 1, 1, 0, 0]}
        thr_df = get_psignifit_threshold_df(root_path=root_path,
                                            p_run_name=run_dir,
                                            csv_name=run_data_df,
                                            n_bins=10, q_bins=True,
                                            sep_col='stair',
                                            isi_list=isi_list,
                                            sep_list=stair_list,
                                            cols_to_add_dict=cols_to_add_dict,
                                            verbose=verbose)
        # print(f'thr_df:\n{thr_df}')

        '''b3'''
        b3_plot_staircase(run_data_path, show_plots=show_plots, verbose=verbose)

        '''c I don't actually need any of these, instead sort get psignifit thr ands make plots from those.'''
        c_plots(save_path=save_path, isi_name_list=isi_names_list, show_plots=show_plots, verbose=verbose)

    '''d'''
    run_folder_names = ['Simon_1', 'Simon_2', 'Simon_3', 'Simon_4', 'Simon_5', 'Simon_6', 'Simon_7', 'Simon_8']

    d_average_participant(root_path=root_path, run_dir_names_list=run_folder_names,
                          trim_n=None, error_type='SE', verbose=verbose)

    all_df_path = f'{root_path}/MASTER_psignifit_thresholds.csv'
    p_ave_path = f'{root_path}/MASTER_ave_thresh.csv'
    err_path = f'{root_path}/MASTER_ave_thr_error_SE.csv'
    n_trimmed = None
    exp_ave = False

    make_average_plots(all_df_path=all_df_path,
                       ave_df_path=p_ave_path,
                       error_bars_path=err_path,
                       n_trimmed=n_trimmed,
                       exp_ave=False,
                       show_plots=True, verbose=True)

participant_list = ['Simon', 'Nick']

# get averages_over_participants
e_average_exp_data(exp_path=exp_path, p_names_list=participant_list,
                   error_type='SE', use_trimmed=False, verbose=True)


# testing make_average_plots for experiment averages
all_df_path = f'{exp_path}/MASTER_exp_thr.csv'
exp_ave_path = f'{exp_path}/MASTER_exp_ave_thr.csv'
err_path = f'{exp_path}/MASTER_ave_thr_error_SE.csv'
n_trimmed = None
exp_ave=True

make_average_plots(all_df_path=all_df_path,
                   ave_df_path=exp_ave_path,
                   error_bars_path=err_path,
                   n_trimmed=n_trimmed,
                   exp_ave=exp_ave,
                   show_plots=True, verbose=True)

print('\nrad_flow_analysis_pipe finished')
