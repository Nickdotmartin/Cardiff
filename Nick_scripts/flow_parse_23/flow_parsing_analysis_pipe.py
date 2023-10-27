import os
import pandas as pd
from flow_parsing_psignifit_analysis import a_data_extraction, b3_plot_staircase
from flow_parsing_psignifit_analysis import d_average_participant, make_flow_parse_plots
from flow_parsing_psignifit_tools import psignifit_thr_df_Oct23
from flow_parsing_psignifit_analysis import e_average_exp_data
import seaborn as sns
import matplotlib.pyplot as plt

# # loop through run folders with first 5 scripts (a, b1, b2, b3, c)
# # then run script d to get master lists and averages
# exp_path = r"C:\Users\sapnm4\PycharmProjects\Cardiff\Nick_scripts\flow_parse_23\flow_parsing_NM_v6"
# exp_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\flow_parsing_NM_v6"
# exp_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\flow_parsing_NM_v7"
# exp_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\flow_parsing_NM_v8"
exp_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\flow_parsing_NM_v8_interleave_dur"
print(f"exp_path: {exp_path}")
convert_path1 = os.path.normpath(exp_path)
print(f"convert_path1: {convert_path1}")
exp_path = convert_path1

monitor = 'OLED'  # 'asus_cal' OLED, 'Nick_work_laptop'
exp_path = os.path.join(exp_path, monitor)


participant_list = ['Nick']  # ' Nicktest_06102023' Nick_extra_prelims

thr_col_name = 'probe_cm_p_sec'
stair_names_col_name = 'stair_name'
bg_dur_name = 'prelim_ms'  # prelim_ms
flow_dir_col_name = 'flow_dir'
flow_name_col_name = 'flow_name'
probe_dur_col_name = 'probe_dur_ms'  # durations
probe_dir_col_name = 'probe_dir'  # directions
resp_col_name = 'response'  # NOT resp_corr
var_cols_list = [stair_names_col_name, flow_dir_col_name, flow_name_col_name, probe_dur_col_name, bg_dur_name]
if 'interleave_dur' in exp_path:
    var_cols_list = [flow_dir_col_name, flow_name_col_name, probe_dur_col_name, bg_dur_name]  # don't include stair_names_col_name

verbose = True
show_plots = True


# if the first folder to analyse is 1, p_idx_plus = 1.  If the first folder is 5, use 5 etc.
p_idx_plus = 1
trim_list = []

# append each run's data to these lists for mean staircases
MASTER_p_trial_data_list = []


for p_idx, participant_name in enumerate(participant_list):

    print(f"\n\n{p_idx}. participant_name: {participant_name}")


    p_name_path = os.path.join(exp_path, participant_name)

    # run_folder_names = [f'{participant_name}_{i+p_idx_plus}' for i in list(range(n_runs))]
    # print(f'run_folder_names: {run_folder_names}')

    # # search to automatically get run_folder_names
    dir_list = os.listdir(p_name_path)
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


        run_path = f'{p_name_path}{os.sep}{run_dir}'

        # # search automatically for probeDur folders to add to probe_dur_int_list
        probe_durs_dir_list = []
        # get a list of all folders in run_path
        dir_list = [f for f in os.listdir(run_path) if os.path.isdir(os.path.join(run_path, f))]
        for folder_name in dir_list:
            if 'probeDur' in folder_name:
                probe_durs_dir_list.append(folder_name)

        print(f'probe_durs_dir_list: {probe_durs_dir_list}')

        # don't delete this (p_name = participant_name),
        # needed to ensure names go name1, name2, name3 not name1, name12, name123
        p_name = participant_name

        '''a'''
        p_name = f'{participant_name}_{r_idx_plus}'

        run_data_df = a_data_extraction(p_name=p_name, run_dir=run_path, dur_list=probe_durs_dir_list, verbose=verbose)
        print(f"run_data_df: {run_data_df.columns.to_list()}\n{run_data_df}\n")

        run_data_path = os.path.join(run_path, 'ALL_durations_sorted.csv')
        print(f"run_data_path: {run_data_path}\n")

        run_data_df = pd.read_csv(run_data_path)



        # '''b3'''
        # # todo: add 'OUT' and 'IN' to y axis labels
        # # todo: why don't line plots match up with final value lines?
        # # todo: x axis tick labels are missing
        # b3_plot_staircase(run_data_path, thr_col='probe_cm_p_sec', resp_col='response',  # 'answer'
        #                   save_name=f"{participant_name}_{r_idx_plus}_staircase.png",
        #                   show_plots=show_plots, verbose=verbose)

        # todo: append run_data_df to MASTER_p_trial_data_list

        # check participant name and run name are present, if not, add them

        # just select columns I need for master df

        # append to MASTER_p_trial_data_list



        thr_df = psignifit_thr_df_Oct23(save_path=run_path, p_run_name=p_name,
                                        run_df=run_data_df, cond_cols_list=var_cols_list,
                                        thr_col=thr_col_name,
                                        resp_col=resp_col_name,
                                        wide_df_cols=probe_dur_col_name,
                                        n_bins=9, q_bins=True,
                                        conf_int=True, thr_type='Bayes',
                                        plot_both_curves=False,
                                        save_name=None,
                                        show_plots=False, save_plots=True,
                                        verbose=True)

        print(f'thr_df:\n{thr_df}')




    # todo: make mean staircase plot for each participant - see order effects script


    '''d participant averages'''
    trim_n = None
    # # todo: for now, don't trim any runs
    # if len(run_folder_names) == 12:
    #     trim_n = 2
    #
    d_average_participant(root_path=p_name_path, run_dir_names_list=run_folder_names,
                          trim_n=trim_n, error_type='SE', verbose=verbose)

    # making average plot
    all_df_path = os.path.join(p_name_path, f'MASTER_TM{trim_n}_thresholds.csv')
    p_ave_path = os.path.join(p_name_path, f'MASTER_ave_TM{trim_n}_thresh.csv')
    err_path = os.path.join(p_name_path, f'MASTER_ave_TM{trim_n}_thr_error_SE.csv')
    if trim_n is None:
        all_df_path = os.path.join(p_name_path, f'MASTER_psignifit_thresholds.csv')
        p_ave_path = os.path.join(p_name_path, 'MASTER_ave_thresh.csv')
        err_path = os.path.join(p_name_path, 'MASTER_ave_thr_error_SE.csv')
    exp_ave = False



    # make plots
    make_flow_parse_plots(all_df_path, p_name_path, participant_name, n_trimmed=trim_n)






#
#
# print(f'exp_path: {exp_path}')
# print('\nget exp_average_data')
# participant_list = ['Nicktest_13102023', 'Nick_extra_prelims']
# n_trimmed = None
# use_trimmed = False
# # e_average_exp_data(exp_path=exp_path, p_names_list=participant_list,
# #                    error_type='SE', use_trimmed=use_trimmed, verbose=True)
#
#
# all_df_path = f'{exp_path}/MASTER_exp_all_thr.csv'
# exp_ave_path = f'{exp_path}/MASTER_exp_ave_thr.csv'
# err_path = f'{exp_path}/MASTER_ave_thr_error_SE.csv'
# n_trimmed = None
# exp_ave = True
#
#
# all_df_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\flow_parsing_NM_v7\OLED\MASTER_exp_all_thr.csv"
# make_flow_parse_plots(all_df_path, exp_path, 'exp_ave', n_trimmed=None, exp_ave=True)


# make_average_plots(all_df_path=all_df_path,
#                    ave_df_path=exp_ave_path,
#                    error_bars_path=err_path,
#                    n_trimmed=n_trimmed,
#                    ave_over_n=len(participant_list),
#                    exp_ave=exp_ave,
#                    show_plots=True, verbose=True)



print('\nflow_parsing_analysis_pipe finished')
