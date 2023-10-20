import os
import pandas as pd
from flow_parsing_psignifit_tools import get_psignifit_threshold_df, run_psignifit, results_to_psignifit, results_csv_to_np_for_psignifit
from flow_parsing_psignifit_analysis import a_data_extraction, b3_plot_staircase
from flow_parsing_psignifit_analysis import d_average_participant, make_flow_parse_plots
from flow_parsing_psignifit_analysis import e_average_exp_data
import seaborn as sns
import matplotlib.pyplot as plt

# # loop through run folders with first 5 scripts (a, b1, b2, b3, c)
# # then run script d to get master lists and averages
# exp_path = r"C:\Users\sapnm4\PycharmProjects\Cardiff\Nick_scripts\flow_parse_23\flow_parsing_NM_v6"
# exp_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\flow_parsing_NM_v6"
# exp_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\flow_parsing_NM_v7"
exp_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\flow_parsing_NM_v8"
print(f"exp_path: {exp_path}")
convert_path1 = os.path.normpath(exp_path)
print(f"convert_path1: {convert_path1}")
exp_path = convert_path1

monitor = 'OLED'  # 'asus_cal' OLED, 'Nick_work_laptop'
exp_path = os.path.join(exp_path, monitor)


participant_list = ['Nick']  # ' Nicktest_06102023' Nick_extra_prelims


verbose = True
show_plots = True


# if the first folder to analyse is 1, p_idx_plus = 1.  If the first folder is 5, use 5 etc.
# p_idx_plus = 4
p_idx_plus = 1

# append each run's data to these lists
MASTER_p_trial_data_list = []


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


        save_path = f'{root_path}{os.sep}{run_dir}'

        # # search automatically for probeDur folders to add to probe_dur_int_list
        probe_dur_int_list = []
        dir_list = os.listdir(save_path)
        for folder in dir_list:
            if 'probeDur' in folder:
                probe_dur_int_list.append(int(folder[8:]))
        print(f'probe_dur_int_list: {probe_dur_int_list}')

        # run_isi_names_list = [f'ISI_{i}' for i in probe_dur_int_list]

        # don't delete this (p_name = participant_name),
        # needed to ensure names go name1, name2, name3 not name1, name12, name123
        p_name = participant_name

        '''a'''
        p_name = f'{participant_name}_{r_idx_plus}'

        run_data_df = a_data_extraction(p_name=p_name, run_dir=save_path, dur_list=probe_dur_int_list, verbose=verbose)
        print(f"run_data_df: {run_data_df.columns.to_list()}\n{run_data_df}\n")

        run_data_path = os.path.join(save_path, 'ALL_durations_sorted.csv')
        print(f"run_data_path: {run_data_path}\n")

        run_data_df = pd.read_csv(run_data_path)

        # '''sort col names'''
        # if 'probe_cm_p_sec' in run_data_df.columns.tolist():
        #     probe_speed_col = 'probe_cm_p_sec'
        # elif 'probeSpeed_cm_per_s' in run_data_df.columns.tolist():
        #     probe_speed_col = 'probeSpeed_cm_per_s'
        #
        # if 'prelim_ms' in run_data_df.columns.tolist():
        #     prelim_col = 'prelim_ms'
        # elif 'prelim_bg_flow_ms' in run_data_df.columns.tolist():
        #     prelim_col = 'prelim_bg_flow_ms'


        '''not sure why these columns are appearing as objects, not floats???'''
        # if any of those columns are 'object' dtype, change to float
        if run_data_df['probe_dur_ms'].dtype == 'object':
            run_data_df['probe_dur_ms'] = run_data_df['probe_dur_ms'].astype(float)
        # if run_data_df['probeSpeed_cm_per_s'].dtype == 'object':
        #     run_data_df['probeSpeed_cm_per_s'] = run_data_df['probeSpeed_cm_per_s'].astype(float)
        if run_data_df['probe_cm_p_sec'].dtype == 'object':
            run_data_df['probe_cm_p_sec'] = run_data_df['probe_cm_p_sec'].astype(float)


        '''b3'''
        # todo: add 'OUT' and 'IN' to y axis labels
        b3_plot_staircase(run_data_path, thr_col='probe_cm_p_sec', resp_col='response',  # 'answer'
                          save_name=f"{participant_name}_{r_idx_plus}_staircase.png",
                          show_plots=show_plots, verbose=verbose)

        # todo: append run_data_df to MASTER_p_trial_data_list

        # check participant name and run name are present, if not, add them

        # just select columns I need for master df

        # append to MASTER_p_trial_data_list


        # get list of all probe durs for this run
        probe_dur_list = run_data_df['probe_dur_ms'].unique().tolist()


        # get stairlist from run_data_df
        stair_list = run_data_df['stair'].unique().tolist()

        # loop through stair_list, getting stair_name, prelim, flow_dir in correct order
        stair_names_list = []
        prelim_list = []
        flow_dir_list = []
        flow_name_list = []
        for stair in stair_list:
            stair_df = run_data_df[run_data_df['stair'] == stair]

            stair_name = stair_df['stair_name'].unique().tolist()
            prelim = stair_df['prelim_ms'].unique().tolist()
            flow_dir = stair_df['flow_dir'].unique().tolist()
            flow_name = stair_df['flow_name'].unique().tolist()


            if len(stair_name) > 1:
                raise ValueError(f"More than one unique stair_name: {stair_name}")
            if len(prelim) > 1:
                raise ValueError(f"More than one unique prelim: {prelim}")
            if len(flow_dir) > 1:
                raise ValueError(f"More than one unique flow_dir: {flow_dir}")
            if len(flow_name) > 1:
                raise ValueError(f"More than one unique flow_name: {flow_name}")
            stair_names_list.append(stair_name[0])
            prelim_list.append(prelim[0])
            flow_dir_list.append(flow_dir[0])
            flow_name_list.append(flow_name[0])

        print(f'stair_list: {stair_list}')
        print(f'stair_names_list: {stair_names_list}')
        print(f'prelim_list: {prelim_list}')
        print(f'flow_dir_list: {flow_dir_list}')
        print(f'flow_name_list: {flow_name_list}')



        # for psignifit just use a reduced df with only the columns needed
        psig_df = run_data_df[['probe_dur_ms', 'stair', 'response', 'probe_cm_p_sec']].copy()
        # todo: changed variable name to probe_speed_cm_sec


        '''get psignifit thresholds df'''
        cols_to_add_dict = {'stair_name': stair_names_list,
                            'prelim': prelim_list,
                            'flow_dir': flow_dir_list,
                            'flow_name': flow_name_list
                            }
        # todo: on curve plot, change y axis to proportion responsed 'out' (not 'correct').  Add prelim and flow_dir to plot title
        thr_df = get_psignifit_threshold_df(root_path=root_path,
                                            p_run_name=run_dir,
                                            csv_name=run_data_df,
                                            n_bins=9, q_bins=True,
                                            thr_col='probe_cm_p_sec',
                                            resp_col='response',
                                            stair_col='stair',
                                            dur_list=probe_dur_list,
                                            stair_list=stair_list,
                                            target_threshold=0.5,  # for a while it didn't work so used .5000001,
                                            cols_to_add_dict=cols_to_add_dict,
                                            verbose=verbose)

        print(f'thr_df:\n{thr_df}')




    # todo: make mean staircase plot for each participant - see order effects script
    # 1. make


    '''d participant averages'''
    trim_n = None
    # # todo: for now, don't trim any runs
    # if len(run_folder_names) == 12:
    #     trim_n = 2
    #
    d_average_participant(root_path=root_path, run_dir_names_list=run_folder_names,
                          trim_n=trim_n, error_type='SE', verbose=verbose)

    # making average plot
    all_df_path = os.path.join(root_path, f'MASTER_TM{trim_n}_thresholds.csv')
    p_ave_path = os.path.join(root_path, f'MASTER_ave_TM{trim_n}_thresh.csv')
    err_path = os.path.join(root_path, f'MASTER_ave_TM{trim_n}_thr_error_SE.csv')
    if trim_n is None:
        all_df_path = os.path.join(root_path, f'MASTER_psignifit_thresholds.csv')
        p_ave_path = os.path.join(root_path, 'MASTER_ave_thresh.csv')
        err_path = os.path.join(root_path, 'MASTER_ave_thr_error_SE.csv')
    exp_ave = False



    # make plots
    make_flow_parse_plots(all_df_path, root_path, participant_name, n_trimmed=trim_n)






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
