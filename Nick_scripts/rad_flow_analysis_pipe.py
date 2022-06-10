import os
import pandas as pd
from psignifit_tools import get_psignifit_threshold_df
from rad_flow_psignifit_analysis import a_data_extraction, b3_plot_staircase
from rad_flow_psignifit_analysis import c_plots, d_average_participant, make_average_plots, e_average_exp_data

# # loop through run folders with first 5 scripts (a, b1, b2, b3, c)
# # then run script d to get master lists and averages
# exp_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/radial_flow_exp'
# exp_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/rad_flow_2'
# exp_path = r'C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\rad_flow_2'
exp_path = r'C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\rad_flow_2_check'
exp_path = os.path.normpath(exp_path)

stair_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
participant_list = ['Simon']  # , 'Nick_half_speed']
isi_list = [1, 4, 6, 9]
# isi_list = [1]
isi_names_list = [f'ISI_{i}' for i in isi_list]

exp_path = os.path.normpath(exp_path)

verbose = True
show_plots = True

n_runs = 3
# if the first folder to analyse is 1, p_idx_plus = 1.  If the forst folder is 5, use 5 etc.
p_idx_plus = 4


for p_idx, participant_name in enumerate(participant_list):
    # if participant_name is 'Nick':
    #     p_idx_plus = 5

    root_path = f'{exp_path}/{participant_name}'

    run_folder_names = [f'{participant_name}_{i+p_idx_plus}' for i in list(range(n_runs))]
    print(f'run_folder_names: {run_folder_names}')

    for run_idx, run_dir in enumerate(run_folder_names):

        # add run number , e.g., add five to access Nick_5 on the zeroth iteration
        r_idx_plus = run_idx + p_idx_plus
        # if participant_name is 'Nick':
        #     r_idx_plus = run_idx + 5

        print(f'\nrun_idx {run_idx}: running analysis for '
              f'{participant_name}, {run_dir}, {participant_name}_{r_idx_plus}\n')
        save_path = f'{root_path}{os.sep}{run_dir}'

        # don't delete this (p_name = participant_name),
        # needed to ensure names go name1, name2, name3 not name1, name12, name123
        p_name = participant_name

        '''a'''
        p_name = f'{participant_name}_{r_idx_plus}'

        # a_data_extraction(p_name=p_name, run_dir=save_path, isi_list=isi_list, verbose=verbose)
        #
        run_data_path = f'{save_path}{os.sep}ALL_ISIs_sorted.xlsx'
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
                                            n_bins=9, q_bins=True,
                                            sep_col='stair',
                                            isi_list=isi_list,
                                            sep_list=stair_list,
                                            conf_int=True,
                                            cols_to_add_dict=cols_to_add_dict,
                                            show_plots=False,
                                            verbose=verbose)
        print(f'thr_df:\n{thr_df}')

        #
        # '''b3'''
        # b3_plot_staircase(run_data_path, show_plots=show_plots, verbose=verbose)
        #
        # '''c I don't actually need any of these, instead sort get psignifit thr ands make plots from those.'''
        # c_plots(save_path=save_path, isi_name_list=isi_names_list, show_plots=show_plots, verbose=verbose)

    # '''d participant averages'''
    # trim_n = None
    # if len(run_folder_names) == 12:
    #     trim_n = 1
    # d_average_participant(root_path=root_path, run_dir_names_list=run_folder_names,
    #                       trim_n=trim_n, error_type='SE', verbose=verbose)
    #
    # all_df_path = f'{root_path}/MASTER_TM1_thresholds.csv'
    # p_ave_path = f'{root_path}/MASTER_ave_TM_thresh.csv'
    # err_path = f'{root_path}/MASTER_ave_TM_thr_error_SE.csv'
    # n_trimmed = trim_n
    # if n_trimmed is None:
    #     all_df_path = f'{root_path}/MASTER_psignifit_thresholds.csv'
    #     p_ave_path = f'{root_path}/MASTER_ave_thresh.csv'
    #     err_path = f'{root_path}/MASTER_ave_thr_error_SE.csv'
    # exp_ave = False
    #
    # make_average_plots(all_df_path=all_df_path,
    #                    ave_df_path=p_ave_path,
    #                    error_bars_path=err_path,
    #                    n_trimmed=n_trimmed,
    #                    exp_ave=False,
    #                    show_plots=True, verbose=True)




# print(f'exp_path: {exp_path}')
# print('\nget exp_average_data')
# participant_list = ['Simon', 'Nick_half_speed']
# n_trimmed = None
# use_trimmed = False
# e_average_exp_data(exp_path=exp_path, p_names_list=participant_list,
#                    error_type='SE', use_trimmed=use_trimmed, verbose=True)
#
#
# all_df_path = f'{exp_path}/MASTER_exp_thr.csv'
# exp_ave_path = f'{exp_path}/MASTER_exp_ave_thr.csv'
# err_path = f'{exp_path}/MASTER_ave_thr_error_SE.csv'
# n_trimmed = None
# exp_ave=True
#
# make_average_plots(all_df_path=all_df_path,
#                    ave_df_path=exp_ave_path,
#                    error_bars_path=err_path,
#                    # error_type='SE',
#                    n_trimmed=n_trimmed,
#                    exp_ave=exp_ave,
#                    show_plots=True, verbose=True)



print('\nrad_flow_analysis_pipe finished')
