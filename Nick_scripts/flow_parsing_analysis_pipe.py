import os
import pandas as pd
from flow_parsing_psignifit_tools import get_psignifit_threshold_df, run_psignifit, results_to_psignifit, results_csv_to_np_for_psignifit
from flow_parsing_psignifit_analysis import a_data_extraction, b3_plot_staircase
from flow_parsing_psignifit_analysis import d_average_participant, make_average_plots, e_average_exp_data

# # loop through run folders with first 5 scripts (a, b1, b2, b3, c)
# # then run script d to get master lists and averages
# exp_path = '/Users/nickmartin/Library/CloudStorage/OneDrive-CardiffUniversity/PycharmProjects/Cardiff/flow_parsing'
exp_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\flow_parsing"
print(f"exp_path: {exp_path}")
convert_path1 = os.path.normpath(exp_path)
print(f"convert_path1: {convert_path1}")
exp_path = convert_path1
stair_list = [0, 1]
stair_names_list = ['0_fl_in_pr_out', '1_fl_out_pr_in']
participant_list = ['Simon', 'Nick']
# probe_dur_list = [4, 6, 9]
all_probe_dur_list = [1, 4, 6, 9, 18]


verbose = True
show_plots = True

# n_runs = 3
n_runs = 12
# if the first folder to analyse is 1, p_idx_plus = 1.  If the first folder is 5, use 5 etc.
# p_idx_plus = 4
p_idx_plus = 1


for p_idx, participant_name in enumerate(participant_list):

    root_path = os.path.join(exp_path, participant_name)

    # run_folder_names = [f'{participant_name}_{i+p_idx_plus}' for i in list(range(n_runs))]
    # print(f'run_folder_names: {run_folder_names}')

    # # search to automatically get run_folder_names
    dir_list = os.listdir(root_path)
    run_folder_names = []
    for i in range(12):  # numbers 0 to 11
        check_dir = f'{participant_name}_{i+p_idx_plus}'   # numbers 1 to 12
        if check_dir in dir_list:
            run_folder_names.append(check_dir)

    print(f'run_folder_names: {run_folder_names}')

    for run_idx, run_dir in enumerate(run_folder_names):

        r_idx_plus = run_idx + p_idx_plus

        print(f'\nrun_idx {run_idx}: running analysis for '
              f'{participant_name}, {run_dir}, {participant_name}_{r_idx_plus}\n')

        # for probe_dur in probe_dur_list:
        #     probe_dur_dir = f'probeDur{probe_dur}'
        save_path = f'{root_path}{os.sep}{run_dir}'

        # # search to automatically get updated isi_list
        dir_list = os.listdir(save_path)
        probe_dur_list = []
        for dur in all_probe_dur_list:
            check_dir = f'probeDur{dur}'
            if check_dir in dir_list:
                probe_dur_list.append(dur)
        print(f'probe_dur_list: {probe_dur_list}')
        # run_isi_names_list = [f'ISI_{i}' for i in probe_dur_list]

        # don't delete this (p_name = participant_name),
        # needed to ensure names go name1, name2, name3 not name1, name12, name123
        p_name = participant_name

        '''a'''
        p_name = f'{participant_name}_{r_idx_plus}'

        run_data_df = a_data_extraction(p_name=p_name, run_dir=save_path, dur_list=probe_dur_list, verbose=verbose)
        print(f"run_data_df: {run_data_df.columns.to_list()}\n{run_data_df}\n")

        run_data_path = os.path.join(save_path, 'ALL_durations_sorted.csv')
        print(f"run_data_path: {run_data_path}\n")
        run_data_df = pd.read_csv(run_data_path)
        print(f"run_data_df: {run_data_df.columns.to_list()}\n{run_data_df}\n")


        '''get psignifit thresholds df'''
        cols_to_add_dict = {'stair_names': stair_names_list}
        # todo: check target threshold should be .5, but it doesn't work, so using .50001?
        thr_df = get_psignifit_threshold_df(root_path=root_path,
                                            p_run_name=run_dir,
                                            csv_name=run_data_df,
                                            n_bins=9, q_bins=True,
                                            stair_col='stair',
                                            dur_list=probe_dur_list,
                                            stair_list=stair_list,
                                            target_threshold=.5000001,
                                            cols_to_add_dict=cols_to_add_dict,
                                            verbose=verbose)
        print(f'thr_df:\n{thr_df}')

        '''b3'''
        run_data_path = os.path.join(save_path, 'ALL_durations_sorted.csv')
        b3_plot_staircase(run_data_path, thr_col='probeSpeed', resp_col='rel_answer',  # 'answer'
                          show_plots=show_plots, verbose=verbose)

    '''d participant averages'''
    trim_n = None
    if len(run_folder_names) == 12:
        trim_n = 1
    d_average_participant(root_path=root_path, run_dir_names_list=run_folder_names,
                          trim_n=trim_n, error_type='SE', verbose=verbose)

    all_df_path = f'{root_path}/MASTER_TM1_thresholds.csv'
    p_ave_path = f'{root_path}/MASTER_ave_TM_thresh.csv'
    err_path = f'{root_path}/MASTER_ave_TM_thr_error_SE.csv'
    n_trimmed = trim_n
    if n_trimmed is None:
        all_df_path = f'{root_path}/MASTER_psignifit_thresholds.csv'
        p_ave_path = f'{root_path}/MASTER_ave_thresh.csv'
        err_path = f'{root_path}/MASTER_ave_thr_error_SE.csv'
    exp_ave = False

    make_average_plots(all_df_path=all_df_path,
                       ave_df_path=p_ave_path,
                       error_bars_path=err_path,
                       n_trimmed=n_trimmed,
                       exp_ave=False,
                       show_plots=True, verbose=True)

print(f'exp_path: {exp_path}')
print('\nget exp_average_data')
participant_list = ['Simon', 'Nick']
n_trimmed = None
use_trimmed = False
e_average_exp_data(exp_path=exp_path, p_names_list=participant_list,
                   error_type='SE', use_trimmed=use_trimmed, verbose=True)


all_df_path = f'{exp_path}/MASTER_exp_thr.csv'
exp_ave_path = f'{exp_path}/MASTER_exp_ave_thr.csv'
err_path = f'{exp_path}/MASTER_ave_thr_error_SE.csv'
n_trimmed = None
exp_ave = True

make_average_plots(all_df_path=all_df_path,
                   ave_df_path=exp_ave_path,
                   error_bars_path=err_path,
                   n_trimmed=n_trimmed,
                   exp_ave=exp_ave,
                   show_plots=True, verbose=True)



print('\nflow_parsing_analysis_pipe finished')
