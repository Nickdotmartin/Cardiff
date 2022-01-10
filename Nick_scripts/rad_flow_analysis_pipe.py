import os

import pandas as pd

from psignifit_tools import get_psignifit_threshold_df
from rad_flow_psignifit_analysis import a_data_extraction, b3_plot_staircase
from rad_flow_psignifit_analysis import c_plots, d_average_participant

# # loop through run folders with first 5 scripts (a, b1, b2, b3, c)
# # then run script d to get master lists and averages
root_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/radial_flow_exp'
run_folder_names = ['Nick_fake_4', 'Nick_5', 'Nick_fake_6', 'Nick_fake_7']  # , 'Nick_6', 'Nick_7', 'Nick_8']
participant_name = 'Nick'
isi_list = [1, 4, 6]
isi_names_list = ['ISI_1', 'ISI_4', 'ISI_6']
stair_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

for run_idx, run_dir in enumerate(run_folder_names):

    add_to_run_idx = run_idx + 4

    print(f'\nrunning analysis for {participant_name}, {run_dir}, {participant_name}_{add_to_run_idx}\n')
    save_path = f'{root_path}{os.sep}{run_dir}'

    # don't delete this (p_name = participant_name),
    # needed to ensure names go name1, name2, name3 not name1, name12, name123
    p_name = participant_name
    run_data_path = f'{save_path}{os.sep}ALL_ISIs_sorted.xlsx'


    # '''a'''
    p_name = f'{participant_name}_{add_to_run_idx}'
    a_data_extraction(p_name=p_name, run_dir=save_path, isi_list=isi_list, verbose=True)


    run_data_df = pd.read_excel(run_data_path, engine='openpyxl',
                                usecols=["ISI", "stair", "stair_name",
                                         "step", "separation", "congruent",
                                         "flow_dir", "probe_jump", "corner",
                                         "probeLum", "trial_response"])
    print(f"run_data_df:\n{run_data_df}")

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
                                        verbose=True)
    print(f'thr_df:\n{thr_df}')

    '''b3'''
    b3_plot_staircase(run_data_path)

    '''c I don't actually need any of these, instead sort get psignifit thr ands make plots from those.'''
    c_plots(save_path=save_path, isi_name_list=isi_names_list, show_plots=True)

'''d'''
d_average_participant(root_path=root_path, run_dir_names_list=run_folder_names,
                      show_plots=True, trim_n=1, error_bars='SE')

print('\nrad_flow_analysis_pipe finished')
