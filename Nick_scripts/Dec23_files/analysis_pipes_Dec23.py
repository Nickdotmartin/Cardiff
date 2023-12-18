import os
import pandas as pd
import numpy as np
from psignifit_tools import psignifit_thr_df_Oct23
from rad_flow_psignifit_analysis import (a_data_extraction_Oct23, d_average_participant,
                                         mean_staircase_plots, joined_plot, get_OLED_luminance,
                                         make_plots_Dec23, e_average_exp_data_Dec23)

from flow_parsing_psignifit_tools import psignifit_thr_df_Oct23_flow
from flow_parsing_psignifit_analysis import (a_data_extraction, d_average_participant_flow,
                                             e_average_exp_data)




'''This script is for analysis and plots.
It loops through each run for each participant and gets the threshold for each condition.
It then gets the means for each participant.
At this point it plots the mean staircase and mean thresholds for that participant.
Finally it gets the experiment level means and plots them.

You should only need to change the 'analyse_what' variable as new data is added.
Other variables should be fine to stay as they are.
'''

def target_detect_analysis_pipe_Dec23(exp_data_path,
                                    analyse_what='all',
                                    participant_list=['pt1', 'pt2', 'pt3', 'pt4', 'pt5'],
                                    from_run_number=1,
                                    verbose=True, show_plots=True,
                                    thr_col_name='OLED_lum', stair_names_col_name='stair_name',
                                    cong_col_name='congruent',
                                    cong_labels=['Incongruent', 'Congruent'],
                                    isi_col_name='isi_ms', sep_col_name='separation',
                                    neg_sep_col_name='neg_sep', bg_dur_name='bg_motion_ms',
                                    resp_col_name='resp_corr', run_col_name='run_number',
                                    monitor='OLED'):

    """ Script for analysing data from target detection or missing target detection experiments.
    For each participant's experimental session: it extracts results from output files and
    calculates thresholds at 75% accuracy.
    For each participant: It gets means (or trimmed means if they have done all 12 runs).
        It makes a mean staircase plot, mean threshold plots, and
        difference between congruent and incongruent plot.
    For whole experiment: It gets experiment means and makes threshold plots, and
        difference between congruent and incongruent plot.

    :param exp_data_path: Path to dir with data for this experiment (e.g., contains monitor name folder)
        e.g., end of path should be either Target_detection or Missing_target_detection_Dec23"
    :param analyse_what: 'all', 'update_plots', 'just_new_data'.
        Note: this works fine if there is only one separation value, but might need adapting if there are multiple values.
    :param participant_list: List of participant IDs (folder names)
    :param from_run_number: Default it 1, but can select a different number to start from.
    :param verbose: Print progress to screen.
    :param show_plots: Show plots once made.
    :param thr_col_name: Name of column containing threshold values. Default is 'OLED_lum' for OLED monitor.
    :param stair_names_col_name: Name of column with conditions names. Default is 'stair_name'.
    :param cong_col_name: Name of column with congruency values. Default is 'congruent'.
    :param cong_labels: Labels to associate with congruency values. Default is ['Incongruent', 'Congruent'].
    :param isi_col_name: Name of column with ISI values (ms). Default is 'isi_ms'.
    :param sep_col_name: Name of column with separation values (pixels). Default is 'separation'.
    :param neg_sep_col_name: Name of column with negative separation values (pixels). Default is 'neg_sep'.
        This isn't needed anymore, but useful for putting separation and congruence on the same axis
        (e.g., incongruent have negative values).
    :param bg_dur_name: Name of column showing duration of background motion (ms). Default is 'bg_motion_ms'.
    :param resp_col_name: Name of column showing whether participants responded correctly. Default is 'resp_corr'.
    :param run_col_name: Name of column showing run number. Default is 'run_number'.
    :param monitor: Name of monitor being used.  Should match PsychoPy Monitor centre and should be
        included in monitor_pixel_size_dict in PsychoPy_tools.py.  Default is 'OLED'.
    """


    # path to dir containing experiment data
    exp_path = os.path.normpath(exp_data_path)
    print(f"exp_path: {exp_path}")


    # experiment dir contains a folder for each monitor used
    exp_path = os.path.join(exp_path, monitor)
    if not os.path.isdir(exp_path):
        raise FileNotFoundError(f'exp_path: {exp_path} not found')

    # check this is the right experiment
    if 'target_detection' not in exp_path.lower():
        raise ValueError(f"exp_path: {exp_path} doesn't contain 'target_detection'")

    '''Don't change this code, update parameters when calling function'''
    # psignifit will loop through these variables (columns) to get thresholds for each condition
    var_cols_list = [isi_col_name, sep_col_name, neg_sep_col_name, cong_col_name, bg_dur_name]

    # todo: just_new_data will need tweaking when I have more than one separation.

    # Changing from_run_number will analyse all runs starting from this number.
    p_idx_plus = from_run_number

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
            # remove from participant list
            participant_list.remove(participant_name)
            continue


        # append each run's data to these lists for mean staircases
        MASTER_p_trial_data_list = []

        # # search to automatically get run_folder_names
        dir_list = os.listdir(p_name_path)
        run_folder_names = []
        for i in range(12):  # numbers 0 to 11
            check_dir = f'{participant_name}_{i + p_idx_plus}'  # numbers 1 to 12
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
                # todo: this won't know if new sep6 data has been added
                if os.path.isfile(run_data_path):
                    analyse_this_run = False
            print(f"\nanalyse_this_run: {analyse_this_run}\n")


            if analyse_this_run:
                new_p_data = True  # signal to update participant ave data and plots
                new_exp_data = True  # signal to update exp ave data and plots

                # do data extraction for this run
                run_data_df = a_data_extraction_Oct23(p_name=p_run_name,
                                                      run_dir=run_path,
                                                      verbose=verbose)
            else:
                run_data_df = pd.read_csv(run_data_path)
            print(f"run_data_df: {run_data_df.columns.to_list()}\n{run_data_df}")


            '''check for OLED_lum column and add after probeLum if not there'''
            if monitor == 'OLED':
                thr_col_name = 'OLED_lum'
                if 'OLED_lum' not in run_data_df.columns.tolist():
                    print(f"\nadding OLED_lum column to {run_data_path}")
                    # get probeLum column index
                    probeLum_idx = run_data_df.columns.get_loc('probeColor1')
                    # insert OLED_lum column after probeLum
                    run_data_df.insert(probeLum_idx + 1, 'OLED_lum',
                                       get_OLED_luminance(np.array(run_data_df['probeColor1'].tolist())))

                    # save run_data_df with OLED_lum column
                    run_data_df.to_csv(run_data_path, index=False)
                    print(f"run_data_df:\n{run_data_df.head()}")


            # get column showing run number (has changed since start of exp)
            # search for 'run_number' substring in column names
            if 'run_number' not in run_data_df.columns.tolist():
                run_data_df.insert(0, 'run_number', run_idx + 1)
            # check if run number column is empty
            elif run_data_df['run_number'].isnull().values.any():
                run_data_df['run_number'] = run_idx + 1
            print(f"run_col_name: {run_col_name}")
            MASTER_p_trial_data_list.append(run_data_df)


            '''Get thresholds for each condition'''
            if analyse_this_run:
                # run psignifit on run_data_df using var_cols_list to loop through the variables
                thr_df = psignifit_thr_df_Oct23(save_path=run_path,
                                                p_run_name=p_run_name,
                                                run_df=run_data_df,
                                                cond_cols_list=var_cols_list,
                                                thr_col=thr_col_name,
                                                resp_col=resp_col_name,
                                                wide_df_cols=isi_col_name,
                                                n_bins=9, q_bins=True,
                                                conf_int=True, thr_type='Bayes',
                                                plot_both_curves=False,
                                                save_name=None,
                                                show_plots=False, save_plots=True,
                                                verbose=True)
                print(f'thr_df:\n{thr_df}')



        '''make mean staircase plot for each participant'''
        if new_p_data or analyse_what == 'update_plots':

            print(f"\n***making master per-trial df ***")
            # join all output data from each run and save as master per-trial csv
            MASTER_p_trial_data_df = pd.concat(MASTER_p_trial_data_list, ignore_index=True)
            # just select columns I need for master df
            MASTER_p_trial_data_df = MASTER_p_trial_data_df[[run_col_name,
                                                             'stair', stair_names_col_name, 'step',
                                                             isi_col_name, sep_col_name, neg_sep_col_name,
                                                             cong_col_name, bg_dur_name, bg_dur_name,
                                                             thr_col_name, resp_col_name]]
            MASTER_p_trial_data_name = f'MASTER_p_trial_data.csv'
            MASTER_p_trial_data_df.to_csv(os.path.join(p_name_path, MASTER_p_trial_data_name), index=False)
            if verbose:
                print(f'\nMASTER_p_trial_data_df:\n{MASTER_p_trial_data_df}')

            mean_staircase_plots(per_trial_df=MASTER_p_trial_data_df, save_path=p_name_path,
                                 participant_name=participant_name, run_col_name=run_col_name,
                                 thr_col_name=thr_col_name,
                                 isi_col_name=isi_col_name, sep_col_name=sep_col_name,
                                 hue_col_name=cong_col_name, hue_names=cong_labels,
                                 ave_type='mean',
                                 show_plots=show_plots, save_plots=True, verbose=True)

            # if there are multiple separations, do mean staircase plots for each separation
            if len(MASTER_p_trial_data_df[sep_col_name].unique()) > 1:
                for sep in MASTER_p_trial_data_df[sep_col_name].unique():
                    sep_df = MASTER_p_trial_data_df[MASTER_p_trial_data_df[sep_col_name] == sep]
                    print(f"\nsep_df:\n{sep_df}")
                    mean_staircase_plots(per_trial_df=sep_df, save_path=p_name_path,
                                         participant_name=participant_name, run_col_name=run_col_name,
                                         thr_col_name=thr_col_name,
                                         isi_col_name=isi_col_name, sep_col_name=sep_col_name,
                                         hue_col_name=cong_col_name, hue_names=cong_labels,
                                         ave_type='mean',
                                         show_plots=show_plots, save_plots=True, verbose=True)



        '''d participant averages'''
        trim_n = None
        if len(run_folder_names) == 12:
            trim_n = 2
        print(f'\ntrim_n: {trim_n}')
        trim_list.append(trim_n)


        cols_to_drop = ['stack', stair_names_col_name]
        cols_to_replace = [cong_col_name, sep_col_name, bg_dur_name]
        groupby_cols = ['neg_sep']

        if new_p_data:  # e.g., it was True for last run/latest data
            d_average_participant(root_path=p_name_path, run_dir_names_list=run_folder_names,
                                  trim_n=trim_n,
                                  groupby_col=groupby_cols,
                                  cols_to_drop=cols_to_drop,
                                  cols_to_replace=cols_to_replace,
                                  error_type='SE', verbose=verbose)



        # making average plot
        all_df_path = os.path.join(p_name_path, f'MASTER_TM{trim_n}_thresholds.csv')
        if trim_n is None:
            all_df_path = os.path.join(p_name_path, f'MASTER_psignifit_thresholds.csv')

        '''make joined plot with untrimmed data'''
        if new_p_data or analyse_what == 'update_plots':
            # ONLY use untrimmed data for this plot.
            all_untrimmed_df = pd.read_csv(os.path.join(p_name_path, f'MASTER_psignifit_thresholds.csv'))
            print(f"\nall_untrimmed_df:\n{all_untrimmed_df}")

            for separation in all_untrimmed_df[sep_col_name].unique():
                print(f"\nseparation: {separation}")
                sep_df = all_untrimmed_df[all_untrimmed_df[sep_col_name] == separation]
                print(f"sep_df:\n{sep_df}")
                joined_plot(untrimmed_df=sep_df, x_cols_str=isi_col_name,
                            hue_col_name=cong_col_name, hue_labels=cong_labels,
                            participant_name=participant_name,
                            x_label='ISI (ms)', y_label='Probe Luminance (cd/m\u00b2)',
                            extra_text=f'sep_{int(separation)}',
                            save_path=p_name_path, save_name=None,
                            verbose=True)


                '''run make average plots (with trimmed data if available)'''
                all_df = pd.read_csv(all_df_path)

                for sep in all_df[sep_col_name].unique():
                    print(f"\nsep: {sep}")
                    sep_df = all_df[all_df[sep_col_name] == sep]
                    print(f"sep_df:\n{sep_df}")

                    make_plots_Dec23(sep_df, root_path=p_name_path,
                                     participant_name=participant_name, n_trimmed=trim_n,
                                     thr_col_name=thr_col_name,
                                     x_col_name=isi_col_name,
                                     hue_col_name=cong_col_name, hue_val_order=[-1, 1],
                                     hue_labels=cong_labels,
                                     motion_col=bg_dur_name,
                                     x_label='ISI (ms)', y_label='Probe Luminance (cd/m\u00b2)',
                                     extra_text=f'sep_{int(separation)}',
                                     exp_ave=False)




    # remove 'test' data from any participant name so it is not in experiment level analysis and means
    # if any name containg 'test', remove it
    participant_list = [p_name for p_name in participant_list if 'test' not in p_name]


    print(f'\nget exp_average_data for: {participant_list}')
    # todo: trim list needs to relate to separation values too


    if new_exp_data or analyse_what == 'all':
        e_average_exp_data_Dec23(exp_path=exp_path, p_names_list=participant_list,
                                 error_type='SE',
                                 verbose=True)

    '''run make average plots'''
    if new_exp_data or analyse_what == 'update_plots':  # e.g., it was True for last run/latest data

        all_df_path = os.path.join(exp_path, "MASTER_exp_all_thr.csv")
        all_df = pd.read_csv(all_df_path)

        n_trimmed = trim_list
        # if all values in trim_list are the same, just use that value
        if len(set(n_trimmed)) == 1:
            n_trimmed = n_trimmed[0]

        for sep in all_df[sep_col_name].unique():
            print(f"\nsep: {sep}")
            sep_df = all_df[all_df[sep_col_name] == sep]
            print(f"sep_df:\n{sep_df}")

            '''make plots for experiment data'''
            make_plots_Dec23(sep_df, root_path=exp_path,
                             participant_name='exp_ave', n_trimmed=n_trimmed,
                             thr_col_name=thr_col_name,
                             x_col_name=isi_col_name,
                             hue_col_name=cong_col_name, hue_val_order=[-1, 1],
                             hue_labels=cong_labels,
                             motion_col=bg_dur_name,
                             x_label='ISI (ms)', y_label='Probe Luminance (cd/m\u00b2)',
                             extra_text=f'sep_{int(separation)}',
                             exp_ave=True)

            '''do joined plot for experiment data linking participant means'''
            joined_plot(untrimmed_df=sep_df, x_cols_str=isi_col_name,
                        hue_col_name=cong_col_name, hue_labels=cong_labels,
                        participant_name='exp_ave', x_label='ISI (ms)', y_label='Probe Luminance (cd/m\u00b2)',
                        extra_text=f'sep_{int(separation)}',
                        save_path=exp_path, save_name=None,
                        verbose=True)


    print('\ntarget_detect_analysis_pipe_Dec23 finished')



def direction_detect_analysis_pipe_Dec23(exp_data_path,
                                         analyse_what='all',
                                         participant_list=['pt1', 'pt2', 'pt3', 'pt4', 'pt5'],
                                         from_run_number=1,
                                         verbose=True, show_plots=True,
                                         thr_col_name='probe_deg_p_sec',  # probe_cm_p_sec
                                         stair_names_col_name='stair_name',
                                         bg_dur_name='bg_motion_ms',   # prelim_ms
                                         flow_dir_col_name='flow_dir',
                                         flow_name_col_name='flow_name',
                                         hue_labels=['Expanding flow', 'Contracting flow'],
                                         probe_dur_col_name='probe_dur_ms',   # durations
                                         probe_dir_col_name='probe_dir',   # directions
                                         resp_col_name='response',   # NOT resp_corr
                                         run_col_name='run_number',
                                         monitor='OLED'):

    """
    Script for analysing data from direction detection experiments.

    For each participant's experimental session: it extracts results from output files and
    calculates thresholds at point-of-subjective-equality (e.g., 50% responses in each direction).

    For each participant: It gets means (or trimmed means if they have done all 12 runs).
        It makes a mean staircase plot, mean threshold plots, and
        difference between inward and outward background motion plot.

    For whole experiment: It gets experiment means and makes threshold plots, and
        difference between inward and outward background motion plot.


    :param exp_data_path: Path to dir with data for this experiment (e.g., contains monitor name folder)
        e.g., end of path should be Direction_detection.py"
    :param analyse_what: 'all', 'update_plots', 'just_new_data'.
        Note: this works fine if there is only one separation value, but might need adapting if there are multiple values.
    :param participant_list: List of participant IDs (folder names)
    :param from_run_number: Default it 1, but can select a different number to start from.
    :param verbose: Print progress to screen.
    :param show_plots: Show plots once made.
    :param thr_col_name: Name of column containing threshold values. Default is 'probe_deg_p_sec'.
    :param stair_names_col_name: Name of column with conditions names. Default is 'stair_name'.
    :param bg_dur_name: Name of column showing duration of background motion (ms). Default is 'bg_motion_ms'.
    :param flow_dir_col_name: Name of column with flow direction values. Default is 'flow_dir'.
    :param flow_name_col_name: Name of column with flow name values. Default is 'flow_name'.
    :param hue_labels: Labels to associate with flow name values. Default is ['Expanding flow', 'Contracting flow'].
    :param probe_dur_col_name: Name of column showing duration of probe (ms). Default is 'probe_dur_ms'.
    :param probe_dir_col_name: Name of column showing direction of probe. Default is 'probe_dir'.
    :param resp_col_name: Name of column showing whether participants responded 'inward' or 'outward'. Default is 'response'.
    :param run_col_name: Name of column showing run number. Default is 'run_number'.
    :param monitor: Name of monitor being used.  Should match PsychoPy Monitor centre and should be
        included in monitor_pixel_size_dict in PsychoPy_tools.py.  Default is 'OLED'.
    """


    # path to dir containing experiment data
    exp_path = os.path.normpath(exp_data_path)
    print(f"exp_path: {exp_path}")

    # experiment dir contains a folder for each monitor used
    exp_path = os.path.join(exp_path, monitor)
    if not os.path.isdir(exp_path):
        raise FileNotFoundError(f'exp_path: {exp_path} not found')

    # check this is the right experiment
    if 'Direction_detection_Dec23' in exp_path.lower():
        raise ValueError(f"exp_path: {exp_path} doesn't contain 'Direction_detection_Dec23'")


    # these shouldn't need to change
    # psignifit will loop through these variables (columns) to get thresholds for each condition
    var_cols_list = [flow_dir_col_name, flow_name_col_name, probe_dur_col_name, bg_dur_name]


    # Changing from_run_number will analyse all runs starting from this number.
    p_idx_plus = from_run_number

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
            # remove from participant list
            participant_list.remove(participant_name)
            continue


        # append each run's data to these lists for mean staircases
        MASTER_p_trial_data_list = []

        # # search to automatically get run_folder_names
        dir_list = os.listdir(p_name_path)
        run_folder_names = []
        for i in range(12):  # numbers 0 to 11
            check_dir = f'{participant_name}_{i + p_idx_plus}'  # numbers 1 to 12
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
                                                dur_list=probe_durs_dir_list,
                                                verbose=verbose)


            run_data_df = pd.read_csv(run_data_path)
            print(f"run_data_df: {run_data_df.columns.to_list()}\n{run_data_df}")


            # get column showing run_number (has changed since start of exp)
            if 'run_number' not in run_data_df.columns.tolist():
                run_data_df.insert(0, 'run_number', run_idx + 1)
            # check if run number column is empty
            elif run_data_df['run_number'].isnull().values.any():
                run_data_df['run_number'] = run_idx + 1
            print(f"run_col_name: {run_col_name}")

            # add trial data to list for master staircase plot
            MASTER_p_trial_data_list.append(run_data_df)


            '''Get thresholds for each condition'''
            if analyse_this_run:
                thr_df = psignifit_thr_df_Oct23_flow(save_path=run_path,
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
        if new_p_data or analyse_what == 'update_plots':

            print(f"\n***making master per-trial df***")
            # join all output data from each run and save as master per-trial csv
            MASTER_p_trial_data_df = pd.concat(MASTER_p_trial_data_list, ignore_index=True)
            # just select columns I need for master df
            MASTER_p_trial_data_df = MASTER_p_trial_data_df[[run_col_name,
                                                             'stair', stair_names_col_name, 'step',
                                                             flow_dir_col_name, flow_name_col_name,
                                                             probe_dur_col_name, bg_dur_name,
                                                             probe_dir_col_name,
                                                             thr_col_name, resp_col_name]]
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
                                 show_plots=show_plots, save_plots=True, verbose=True)

        '''d participant averages'''
        trim_n = None
        if len(run_folder_names) == 12:
            trim_n = 2
        print(f'\ntrim_n: {trim_n}')
        trim_list.append(trim_n)

        if new_p_data:  # e.g., it was True for last run/latest data
            d_average_participant_flow(root_path=p_name_path, run_dir_names_list=run_folder_names,
                                       trim_n=trim_n, error_type='SE', verbose=verbose)



        # get paths to average data for plots
        all_df_path = os.path.join(p_name_path, f'MASTER_TM{trim_n}_thresholds.csv')
        if trim_n is None:
            all_df_path = os.path.join(p_name_path, f'MASTER_psignifit_thresholds.csv')

        if new_p_data or analyse_what == 'update_plots':
            # ONLY use untrimmed data for this plot.
            all_untrimmed_df = pd.read_csv(os.path.join(p_name_path, f'MASTER_psignifit_thresholds.csv'))
            print(f"\nall_untrimmed_df:\n{all_untrimmed_df}")

            '''make joined plot with untrimmed data'''
            if len(all_untrimmed_df[bg_dur_name].unique()) == 1:
                bg_motion_dur = all_untrimmed_df[bg_dur_name].unique()[0]
                print(f"bg_motion_dur: {bg_motion_dur}")
            else:
                raise ValueError(f"more than one bg_motion_dur value: {all_untrimmed_df[bg_motion_dur].unique()}")

            joined_plot(untrimmed_df=all_untrimmed_df, x_cols_str=probe_dur_col_name,
                        hue_col_name=flow_dir_col_name, hue_labels=hue_labels,
                        participant_name=participant_name,
                        x_label='Probe Duration (ms)', y_label='Probe Speed (deg / sec)',
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
                             x_label='Probe duration (ms)', y_label='Probe Speed (deg/sec)',
                             extra_text=None,
                             exp_ave=False)




    # remove 'test' data from any participant name so it is not in experiment level analysis and means
    # if any name containg 'test', remove it
    participant_list = [p_name for p_name in participant_list if 'test' not in p_name]
    print(f'\nget exp_average_data for: {participant_list}')


    if new_exp_data or analyse_what == 'all':
        e_average_exp_data(exp_path=exp_path, p_names_list=participant_list,
                           error_type='SE',
                           verbose=True)

    if new_exp_data or analyse_what == 'update_plots':  # e.g., it was True for last run/latest data
        all_df_path = f'{exp_path}/MASTER_exp_all_thr.csv'
        all_df_path = os.path.join(exp_path, 'MASTER_exp_all_thr.csv')

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
                         x_label='Probe duration (ms)', y_label='Probe Speed (deg/sec)',
                         extra_text=None,
                         exp_ave=True)

        '''do joined plot for experiment data linking participant means'''
        joined_plot(untrimmed_df=all_df_path, x_cols_str=probe_dur_col_name,
                    hue_col_name=flow_dir_col_name, hue_labels=hue_labels,
                    participant_name='exp_ave',
                    x_label='Probe Duration (ms)', y_label='Probe Speed (deg / sec)',
                    extra_text=None,
                    save_path=exp_path, save_name=None,
                    verbose=True)


    print('\ndirection_analysis_pipe finished')