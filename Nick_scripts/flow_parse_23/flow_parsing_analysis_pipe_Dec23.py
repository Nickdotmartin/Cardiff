import os
import pandas as pd
from flow_parsing_psignifit_analysis import (a_data_extraction, d_average_participant,
                                             e_average_exp_data)
from flow_parsing_psignifit_tools import psignifit_thr_df_Oct23
from rad_flow_psignifit_analysis import mean_staircase_plots, joined_plot, make_plots_Dec23


'''This script is for analysis and plots.
It loops through each run for each participant and gets the threshold for each condition.
It then gets the means for each participant.
At this point it plots the mean staircase and mean thresholds for that participant.
Finally it gets the experiment level means and plots them.

You should only need to change the 'analyse_what' variable as new data is added.
Other variables should be fine to stay as they are.
'''

# path to dir containing experiment data
exp_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\Direction_detection_Dec23"
exp_path = os.path.normpath(exp_path)
print(f"exp_path: {exp_path}")

# experiment dir contains a folder for each monitor used
monitor = 'OLED'  # 'asus_cal' OLED, 'Nick_work_laptop'
exp_path = os.path.join(exp_path, monitor)
if not os.path.isdir(exp_path):
    raise FileNotFoundError(f'exp_path: {exp_path} not found')

# monitor dir contains a folder for each participant
participant_list = ['test', 'pt1', 'pt3', 'pt4'] # , 'pt2', ]  # ' Nicktest_06102023' Nick_extra_prelims
# participant_list = ['pt1']  # ' Nicktest_06102023' Nick_extra_prelims

# p_idx_plus will analyse all runs starting from this number.
# leave it at one to include all runs in the analysis (or to just analyse new data, see 'analyse_what' below)
p_idx_plus = 1


# these shouldn't need to change
thr_col_name = 'probe_deg_p_sec'  # probe_cm_p_sec
stair_names_col_name = 'stair_name'
bg_dur_name = 'bg_motion_ms'  # prelim_ms
flow_dir_col_name = 'flow_dir'
flow_name_col_name = 'flow_name'
hue_labels = ['Expanding flow', 'Contracting flow']
probe_dur_col_name = 'probe_dur_ms'  # durations
probe_dir_col_name = 'probe_dir'  # directions
resp_col_name = 'response'  # NOT resp_corr

# psignifit will loop through these variables (columns) to get thresholds for each condition
var_cols_list = [flow_dir_col_name, flow_name_col_name, probe_dur_col_name, bg_dur_name]

verbose = True  # if True, prints los of data and progress to console
show_plots = True  # if True, shows plots as they are made

'''Update participant data to analyse or analyse_what variablesd'''
'''select data to analyse: 
    'all' analyses all data, 
    'update_plots' only updates plots, 
    'just_new_data' only analyses new runs that haven't been analysed yet.
        It will update any downstream means and plots if new data is added.'''
analyse_what = 'just_new_data'  # 'update_plots', 'just_new_data', 'all'

new_exp_data = False  # if True, will update exp level results and plots

# list of whether participants' data has been trimmed
trim_list = []

for p_idx, participant_name in enumerate(participant_list):

    print(f"\n\n{p_idx}. participant_name: {participant_name}")

    # path to this participant's data
    p_name_path = os.path.join(exp_path, participant_name)
    print(f"p_name_path: {p_name_path}")

    # if this participant doesn't exist, skip to next participant
    if not os.path.isdir(p_name_path):
        print(f"\n\n\np_name_path: {p_name_path} not found\n\n\n")
        continue


    # append each run's data to these lists for mean staircases
    MASTER_p_trial_data_list = []

    # # search to automatically get run_folder_names
    dir_list = os.listdir(p_name_path)
    run_folder_names = []
    for i in range(12):  # numbers 0 to 11
        check_dir = f'{participant_name}_{i+p_idx_plus}'  # numbers 1 to 12
        if check_dir in dir_list:
            run_folder_names.append(check_dir)
    print(f'run_folder_names: {run_folder_names}')

    new_p_data = False  # if True, will update this participant level results and plots

    '''loop through each run for this participant'''
    for run_idx, run_dir in enumerate(run_folder_names):

        r_idx_plus = run_idx + p_idx_plus

        print(f'\nrun_idx {run_idx}: running analysis for '
              f'{participant_name}, {run_dir}, {participant_name}_{r_idx_plus}\n')
        run_path = os.path.join(p_name_path, run_dir)
        print(f"run_path: {run_path}")


        # # search automatically for probeDur folders to add to probe_dur_int_list
        probe_durs_dir_list = []
        # get a list of all folders in run_path
        dir_list = [f for f in os.listdir(run_path) if os.path.isdir(os.path.join(run_path, f))]
        for folder_name in dir_list:
            if 'probeDur' in folder_name:
                probe_durs_dir_list.append(folder_name)

        print(f'probe_durs_dir_list: {probe_durs_dir_list}')

        # don't delete this (p_run_name = participant_name),
        # needed to ensure names go name1, name2, name3 not name1, name12, name123
        p_run_name = participant_name

        '''a - data extraction'''
        p_run_name = f'{participant_name}_{r_idx_plus}'
        run_data_path = os.path.join(run_path, 'RUNDATA_sorted.csv')
        print(f"run_data_path: {run_data_path}\n")

        # should I analyse this run?
        analyse_this_run = True
        if analyse_what == 'update_plots':
            analyse_this_run = False
        elif analyse_what == 'just_new_data':
            if os.path.isfile(run_data_path):
                analyse_this_run = False
        print(f"\nanalyse_this_run: {analyse_this_run}\n")


        if analyse_this_run:
            new_p_data = True  # signal to update participant ave data and plots
            new_exp_data = True  # signal to update exp ave data and plots

            # do data extraction for this run
            run_data_df = a_data_extraction(p_name=p_run_name, run_dir=run_path,
                                            dur_list=probe_durs_dir_list, verbose=verbose)


        run_data_df = pd.read_csv(run_data_path)
        print(f"run_data_df: {run_data_df.columns.to_list()}\n{run_data_df}")


        # get column showing run number (has changed since start of exp)
        # search for 'run_number' substring in column names
        run_num_col = [col for col in run_data_df.columns if 'run_number' in col]
        if len(run_num_col) == 1:
            run_col_name = run_num_col[0]
        elif len(run_num_col) == 0:
            run_col_name = 'run_number'
            # add 'run' to run_data_df if not already there
            if 'run_number' not in run_data_df.columns.tolist():
                run_data_df.insert(0, 'run_number', run_idx + 1)
        print(f"run_col_name: {run_col_name}")
        MASTER_p_trial_data_list.append(run_data_df)


        '''Get thresholds for each condition'''
        if analyse_this_run:
            thr_df = psignifit_thr_df_Oct23(save_path=run_path,
                                            p_run_name=p_run_name,
                                            run_df=run_data_df,
                                            cond_cols_list=var_cols_list,
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


    '''make mean staircase plot for each participant'''
    if analyse_what == 'update_plots' or new_p_data:

        print(f"\n***making master per-trial df***")
        # join all output data from each run and save as master per-trial csv
        MASTER_p_trial_data_df = pd.concat(MASTER_p_trial_data_list, ignore_index=True)
        # just select columns I need for master df
        MASTER_p_trial_data_df = MASTER_p_trial_data_df[[run_col_name, 'stair', stair_names_col_name, 'step',
                                                         flow_dir_col_name, flow_name_col_name,
                                                         probe_dur_col_name, bg_dur_name,
                                                         probe_dir_col_name, thr_col_name, resp_col_name]]

        MASTER_p_trial_data_name = f'MASTER_p_trial_data.csv'
        MASTER_p_trial_data_df.to_csv(os.path.join(p_name_path, MASTER_p_trial_data_name), index=False)
        if verbose:
            print(f'\nMASTER_p_trial_data_df:\n{MASTER_p_trial_data_df}')

        '''If there are interleaved conds (e.g., bg_motion_ms), do separate staircases for each of them'''
        # if not interleaved_col_list:  # do one staircase plt for all interleaved conditions
        mean_staircase_plots(per_trial_df=MASTER_p_trial_data_df, save_path=p_name_path,
                             participant_name=participant_name, run_col_name=run_col_name,
                             thr_col_name=thr_col_name,
                             isi_col_name=probe_dur_col_name, sep_col_name=None,
                             hue_col_name=flow_dir_col_name, hue_names=hue_labels,
                             ave_type='mean',
                             show_plots=True, save_plots=True, verbose=True)

    '''d participant averages'''
    trim_n = None
    if len(run_folder_names) == 12:
        trim_n = 2
    print(f'\ntrim_n: {trim_n}')

    if new_p_data:  # e.g., it was True for last run/latest data
        d_average_participant(root_path=p_name_path, run_dir_names_list=run_folder_names,
                              trim_n=trim_n, error_type='SE', verbose=verbose)



    # get paths to average data for plots
    all_df_path = os.path.join(p_name_path, f'MASTER_TM{trim_n}_thresholds.csv')
    p_ave_path = os.path.join(p_name_path, f'MASTER_ave_TM{trim_n}_thresh.csv')
    err_path = os.path.join(p_name_path, f'MASTER_ave_TM{trim_n}_thr_error_SE.csv')
    if trim_n is None:
        all_df_path = os.path.join(p_name_path, f'MASTER_psignifit_thresholds.csv')
        p_ave_path = os.path.join(p_name_path, 'MASTER_ave_thresh.csv')
        err_path = os.path.join(p_name_path, 'MASTER_ave_thr_error_SE.csv')
    exp_ave = False

    if analyse_what == 'update_plots' or new_p_data:  # e.g., it was True for last run/latest data
        # ONLY use untrimmed data for this plot.
        all_untrimmed_df = pd.read_csv(os.path.join(p_name_path, f'MASTER_psignifit_thresholds.csv'))
        print(f"\nall_untrimmed_df:\n{all_untrimmed_df}")


        if len(all_untrimmed_df[bg_dur_name].unique()) == 1:
            bg_motion_dur = all_untrimmed_df[bg_dur_name].unique()[0]
            print(f"bg_motion_dur: {bg_motion_dur}")
        else:
            raise ValueError(f"more than one bg_motion_dur value: {all_untrimmed_df[bg_motion_dur].unique()}")

        joined_plot(untrimmed_df=all_untrimmed_df, x_cols_str=probe_dur_col_name,
                    hue_col_name=flow_dir_col_name, hue_labels=hue_labels,
                    participant_name=participant_name,
                    x_label='Probe Duration (ms)', y_label='Probe Velocity (deg per sec)',
                    extra_text=f'All data (untrimmed)',
                    save_path=p_name_path, save_name=None,
                    verbose=True)


        '''run make average plots (with trimmed data if available)'''
        make_plots_Dec23(all_df_path, root_path=p_name_path,
                         participant_name=participant_name, n_trimmed=trim_n,
                         thr_col_name=thr_col_name,
                         x_col_name=probe_dur_col_name,
                         hue_col_name=flow_name_col_name, hue_val_order=['exp', 'cont'],
                         hue_labels=hue_labels,
                         motion_col=bg_dur_name,
                         x_label='Probe duration (ms)', y_label='Probe velocity (deg/sec)',
                         extra_text=None,
                         exp_ave=False)




# remove 'test' data from any participant name so it is not in experiment level analysis and means
# if any name containg 'test', remove it
participant_list = [p_name for p_name in participant_list if 'test' not in p_name]


print(f'exp_path: {exp_path}')
print('\nget exp_average_data')
if new_exp_data:  # e.g., it was True for last run/latest data
    e_average_exp_data(exp_path=exp_path, p_names_list=participant_list,
                       error_type='SE',
                       verbose=True)

if analyse_what == 'update_plots' or new_exp_data:  # e.g., it was True for last run/latest data
    all_df_path = f'{exp_path}/MASTER_exp_all_thr.csv'
    all_df_path = os.path.join(exp_path, 'MASTER_exp_all_thr.csv')

    exp_ave = True
    n_trimmed = trim_list
    # if all values in trim_list are the same, just use that value
    if len(set(n_trimmed)) == 1:
        n_trimmed = n_trimmed[0]

    '''make plots for experiment data'''
    make_plots_Dec23(all_df_path, root_path=exp_path,
                     participant_name='exp_ave', n_trimmed=n_trimmed,
                     thr_col_name=thr_col_name,
                     x_col_name=probe_dur_col_name,
                     hue_col_name=flow_name_col_name, hue_val_order=['exp', 'cont'],
                     hue_labels=hue_labels,
                     motion_col=bg_dur_name,
                     x_label='Probe duration (ms)', y_label='Probe velocity (deg/sec)',
                     extra_text=None,
                     exp_ave=True)

    '''do joined plot for experiment data linking participant means'''
    joined_plot(untrimmed_df=all_df_path, x_cols_str=probe_dur_col_name,
                hue_col_name=flow_dir_col_name, hue_labels=hue_labels,
                participant_name='exp_ave',
                x_label='Probe Duration (ms)', y_label='Probe Velocity (deg per sec)',
                extra_text=None,
                save_path=exp_path, save_name=None,
                verbose=True)


print('\nflow_parsing_analysis_pipe finished')
