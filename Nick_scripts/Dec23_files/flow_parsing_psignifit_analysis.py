import os
import numpy as np
import pandas as pd
from rad_flow_psignifit_analysis import trim_n_high_n_low


"""
This page contains python functions to analyse the radial_flow.py experiment.
The workflow is based on Martin's MATLAB scripts used in OLD_MATLAB_analysis.py.

1. a_data_extraction: put data from one run, multiple durs into one array. 
2. b1_extract_last_values: get the threshold used on the last values in each 
    staircase for each dur.  this is used for b3_plot_staircases.
# not used. 3. b2_last_reversal/m: Computes the threshold for each staircase as an average of 
#     the last N reversals (incorrect responses).  Plot these as mean per sep, 
#     and also multi-plot for each dur with different values for -18&+18 etc
4. b3_plot_staircase:
5. c_plots: get psignifit thr values, 
    plot:   data_psignifit_values (pos_sep_no_one_probe)
            runs_psignifit_values (multi_batman_plots)
6. d_average_participant:
    master_psignifit_thr (all runs, one csv)
    master_average_psignifit_thr (mean of all runs)
    plot:
    master_average_psignifit_thr (pos_sep_no_one_probe)


I've also added functions for repeated bit of code:
data: 

split_df_alternate_rows(): 
    split a dataframe into two dataframes: 
        one with positive sep values, one with negative


plots: 
plot_data_unsym_batman: single ax with pos and neg separation (not symmetrical), dotted line at zero.
Batman plots: 


"""

pd.options.display.float_format = "{:,.2f}".format




def a_data_extraction(p_name, run_dir, dur_list, save_all_data=True, verbose=True):
    """
    This script is a python version of Martin's first MATLAB analysis scripts, described below.

    a_data_extraction.m: Once a participant has completed a run of all durations,
        this script gets all their data into one file, and sorts each duration by stair.

    :param p_name: participant's name as used to save csv files.  e.g., if the
            file is .../nick1.csv, participant name is 'nick1'.
    :param run_dir: directory where durations folders are stored.
    :param dur_list: List of probe_duration values, may differ between experiments.
    :param save_all_data: If True, will save all_data_df as a xlsx.
    :param verbose: If True, will print progress to screen.

    :return: ALL_durations_sorted.xlsx: A pandas DataFrame with n xlsx file of all
        data for one run of all durations.
    """
    print("\n***running a_data_extraction()***\n")

    # get run name/number
    path, run = os.path.split(run_dir)
    if verbose:
        print(f"run: {run}")

    if dur_list is None:
        raise ValueError('Please pass a list of probe_duration values to identify directories containing data.')
    else:
        print(f'dur_list: {dur_list}')

    all_data = []

    # if dur list passed: loop through durations in each run.
    if len(dur_list) > 0:
        for duration in dur_list:
            filepath = os.path.join(run_dir, duration, f'{p_name}_output.csv')
            if verbose:
                print(f"filepath: {filepath}")

            if not os.path.isfile(filepath):
                raise FileNotFoundError(filepath)

            # load data
            this_dur_df = pd.read_csv(filepath)
            if verbose:
                print(f"loaded csv:\n{this_dur_df.head()}")

            # remove any Unnamed columns
            if any("Unnamed" in i for i in list(this_dur_df.columns)):
                unnamed_col = [i for i in list(this_dur_df.columns) if "Unnamed" in i][0]
                this_dur_df.drop(unnamed_col, axis=1, inplace=True)

            # OLED windows machine sometimes adds extra columns, remove: ['thisRow.t', 'notes']
            if 'thisRow.t' in list(this_dur_df.columns):
                this_dur_df.drop('thisRow.t', axis=1, inplace=True)
            if 'notes' in list(this_dur_df.columns):
                this_dur_df.drop('notes', axis=1, inplace=True)

            # sort by staircase
            trial_numbers = list(this_dur_df['trial_number'])
            this_dur_df = this_dur_df.sort_values(by=['stair', 'trial_number'])

            # add duration column for multi-indexing
            this_dur_df.insert(1, 'srtd_trial_idx', trial_numbers)
            if verbose:
                print(f'df sorted by stair: {type(this_dur_df)}\n{this_dur_df}')

            # get column names to use on all_data_df
            column_names = list(this_dur_df)
            if verbose:
                print(f'column_names: {len(column_names)}\n{column_names}')

            # add to all_data
            all_data.append(this_dur_df)

        # create all_data_df - reshape to 2d
        if verbose:
            print(f'all_data: {type(all_data)}\n{all_data}')
        all_data_shape = np.shape(all_data)
        sheets, rows, columns = np.shape(all_data)
        all_data = np.reshape(all_data, newshape=(sheets * rows, columns))
        if verbose:
            print(f'all_data reshaped from {all_data_shape} to {np.shape(all_data)}')
        all_data_df = pd.DataFrame(all_data, columns=column_names)

    else:  # if no dur list passed
        filepath = os.path.join(run_dir, f'{p_name}_output.csv')
        if verbose:
            print(f"filepath: {filepath}")

        if not os.path.isfile(filepath):
            raise FileNotFoundError(filepath)


        # load data
        this_dur_df = pd.read_csv(filepath)
        if verbose:
            print(f"loaded csv:\n{this_dur_df.head()}")

        # remove any Unnamed columns
        if any("Unnamed" in i for i in list(this_dur_df.columns)):
            unnamed_col = [i for i in list(this_dur_df.columns) if "Unnamed" in i][0]
            this_dur_df.drop(unnamed_col, axis=1, inplace=True)

        # OLED Windows machine sometimes adds extra columns, remove: ['thisRow.t', 'notes']
        if 'thisRow.t' in list(this_dur_df.columns):
            this_dur_df.drop('thisRow.t', axis=1, inplace=True)
        if 'notes' in list(this_dur_df.columns):
            this_dur_df.drop('notes', axis=1, inplace=True)

        # sort by staircase
        trial_numbers = list(this_dur_df['trial_number'])
        this_dur_df = this_dur_df.sort_values(by=['stair', 'trial_number'])

        # add duration column for multi-indexing
        # this_dur_df.insert(0, 'probe_dur_ms', duration)
        this_dur_df.insert(1, 'srtd_trial_idx', trial_numbers)
        if verbose:
            print(f'df sorted by stair: {type(this_dur_df)}\n{this_dur_df}')

        all_data_df = this_dur_df

    if verbose:
        print(f"all_data_df:\n{all_data_df}")

    if save_all_data:
        save_name = 'RUNDATA_sorted.csv'
        save_csv_path = os.path.join(run_dir, save_name)
        if verbose:
            # print(f"\nsaving all_data_df to save_excel_path:\n{save_excel_path}")
            print(f"\nsaving all_data_df to save_csv_path:\n{save_csv_path}")
        convert_path1 = os.path.normpath(save_csv_path)
        print(f"convert_path1: {convert_path1}")

        all_data_df.to_csv(convert_path1, index=False)

    print("\n***finished a_data_extraction()***\n")

    return all_data_df


def d_average_participant_flow(root_path, run_dir_names_list,
                               thr_df_name='psignifit_thresholds',
                               error_type=None,
                               trim_n=None,
                               verbose=True):
    """
    d_average_participant: take psignifit_thresholds.csv
    in each participant run folder and make master lists
    MASTER_psignifit_thresholds.csv

    Get mean threshold across 6 run conditions saved as
    MASTER_ave_thresh.csv

    Save master lists to folder containing the six runs (root_path).

    :param root_path: dir containing run folders
    :param run_dir_names_list: names of run folders
    :param thr_df_name: Name of threshold dataframe.  If no name is given it will use 'psignifit_thresholds'.
    :param error_type: Default: None. Can pass sd or se for standard deviation or error.
    :param trim_n: default None.  If int is passed, will call function trim_n_high_n_low(),
            which trims the n highest and lowest values.
    :param verbose: Default true, print progress to screen

    :returns: ave_psignifit_thr_df: (trimmed?) mean threshold for each separation and dur.
    """

    print("\n***running d_average_participant_flow()***")

    all_psignifit_list = []
    for run_idx, run_name in enumerate(run_dir_names_list):

        this_psignifit_df = pd.read_csv(f'{root_path}{os.sep}{run_name}{os.sep}{thr_df_name}.csv')
        print(f'\n{run_idx}. {run_name} - this_psignifit_df:\n{this_psignifit_df}')

        if 'Unnamed: 0' in list(this_psignifit_df):
            this_psignifit_df.drop('Unnamed: 0', axis=1, inplace=True)

        if 'stair_name' not in list(this_psignifit_df.columns):
            this_psignifit_df.insert(0, 'stair_name', [f'flow_{flow_dir}_{flow_name}_bg_motion_ms{prelim_ms}'
                                                  for flow_dir, flow_name, prelim_ms in
                                                  zip(this_psignifit_df['flow_dir'], this_psignifit_df['flow_name'],
                                                      this_psignifit_df['bg_motion_ms'])])
            print(f'\nget_means_df:\n{this_psignifit_df}')
        if 'stair' not in list(this_psignifit_df.columns):
            # generate stair numbers from unique stair_names
            this_psignifit_df.insert(0, 'stair', [i for i in range(len(this_psignifit_df['stair_name'].unique()))])

        rows, cols = this_psignifit_df.shape
        this_psignifit_df.insert(0, 'stack', [run_idx] * rows)

        if verbose:
            print(f'\nthis_psignifit_df:\n{this_psignifit_df}')

        all_psignifit_list.append(this_psignifit_df)

    # join all stacks (runs/groups) data and save as master csv
    all_data_psignifit_df = pd.concat(all_psignifit_list, ignore_index=True)

    all_data_psignifit_df.to_csv(os.path.join(root_path, f"MASTER_{thr_df_name}.csv"), index=False)
    if verbose:
        print(f'\nall_data_psignifit_df:\n{all_data_psignifit_df}')

    """Part 2: trim highest and lowest values is required and get average vals and errors"""
    # # trim highest and lowest values
    if trim_n is not None:
        trimmed_df = trim_n_high_n_low(all_data_psignifit_df, trim_from_ends=trim_n,
                                       reference_col='stair',
                                       stack_col_id='stack',
                                       verbose=verbose)
        trimmed_df.to_csv(f'{root_path}{os.sep}MASTER_TM{trim_n}_thresholds.csv', index=False)

        get_means_df = trimmed_df
    else:
        get_means_df = all_data_psignifit_df

    print(f'\nget_means_df:\n{get_means_df}')

    # # get means and errors
    get_means_df = get_means_df.drop('stack', axis=1)

    # loop through stair_list and add corresponding stair_name and flow_name to list
    stair_list = get_means_df['stair'].unique().tolist()
    stair_names_list = []
    bg_motion_ms_list = []
    flow_dir_list = []
    flow_names_list = []
    for stair in stair_list:
        stair_names_list.append(get_means_df.loc[get_means_df['stair'] == stair, 'stair_name'].unique().tolist()[0])
        bg_motion_ms_list.append(get_means_df.loc[get_means_df['stair'] == stair, 'bg_motion_ms'].unique().tolist()[0])
        flow_dir_list.append(get_means_df.loc[get_means_df['stair'] == stair, 'flow_dir'].unique().tolist()[0])
        flow_names_list.append(get_means_df.loc[get_means_df['stair'] == stair, 'flow_name'].unique().tolist()[0])

    # get_means_df = get_means_df.drop('prelim_ms', axis=1)
    get_means_df = get_means_df.drop('bg_motion_ms', axis=1)
    get_means_df = get_means_df.drop('flow_dir', axis=1)
    get_means_df = get_means_df.drop('stair_name', axis=1)
    get_means_df = get_means_df.drop('flow_name', axis=1)



    # get average values (from numeric columns)
    ave_psignifit_thr_df = get_means_df.groupby(['stair'], sort=False).mean()
    # add stair_names and flow_name back in
    ave_psignifit_thr_df.insert(0, 'stair_name', stair_names_list)
    ave_psignifit_thr_df.insert(1, 'bg_motion_ms', bg_motion_ms_list)
    ave_psignifit_thr_df.insert(2, 'flow_dir', flow_dir_list)
    ave_psignifit_thr_df.insert(3, 'flow_name', flow_names_list)
    if verbose:
        print(f'\nget_means_df:\n{get_means_df}')
        print(f'\nave_psignifit_thr_df:\n{ave_psignifit_thr_df}')

    if error_type in [False, None]:
        error_bars_df = None
    elif error_type.lower() in ['se', 'error', 'std-error', 'standard error', 'standard_error']:
        error_bars_df = get_means_df.groupby('stair', sort=False).sem()
    elif error_type.lower() in ['sd', 'stdev', 'std_dev', 'std.dev', 'deviation', 'standard_deviation']:
        error_bars_df = get_means_df.groupby('stair', sort=False).std()
    else:
        raise ValueError(f"error_type should be in:\nfor none: [False, None]\n"
                         f"for standard error: ['se', 'error', 'std-error', 'standard error', 'standard_error']\n"
                         f"for standard deviation: ['sd', 'stdev', 'std_dev', 'std.dev', "
                         f"'deviation', 'standard_deviation']")

    # replace NaNs with 0
    error_bars_df.fillna(0, inplace=True)

    # add stair_names and flow_name back in
    error_bars_df.insert(0, 'stair_name', stair_names_list)
    error_bars_df.insert(1, 'bg_motion_ms', bg_motion_ms_list)
    error_bars_df.insert(2, 'flow_dir', flow_dir_list)
    error_bars_df.insert(3, 'flow_name', flow_names_list)
    print(f'\nerror_bars_df:\n{error_bars_df}')

    # save csv with average values
    if trim_n is not None:
        ave_psignifit_thr_df.to_csv(f'{root_path}{os.sep}MASTER_ave_TM{trim_n}_thresh.csv')
        error_bars_df.to_csv(f'{root_path}{os.sep}MASTER_ave_TM{trim_n}_thr_error_{error_type}.csv')
    else:
        ave_psignifit_thr_df.to_csv(f'{root_path}{os.sep}MASTER_ave_thresh.csv')
        error_bars_df.to_csv(f'{root_path}{os.sep}MASTER_ave_thr_error_{error_type}.csv')

    print("\n*** finished d_average_participant_flow()***\n")

    return ave_psignifit_thr_df, error_bars_df


def e_average_exp_data(exp_path, p_names_list,
                       error_type='SE',
                       # use_trimmed=True,
                       verbose=True):
    """
    e_average_over_participants: take MASTER_ave_TM_thresh.csv (or MASTER_ave_thresh.csv)
    in each participant folder and make master list
    MASTER_exp_all_thr.csv

    Get mean thresholds averaged across all participants saved as
    MASTER_exp_ave_thr.csv

    Save master lists to exp_path.

    Plots:
    MASTER_exp_ave_thr saved as exp_ave_thr_all_runs.png
    MASTER_exp_ave_thr two-probe/one-probe saved as exp_ave_thr_div_1probe.png
    these both have two versions:
    a. x-axis is separation, dur as different lines
    b. x-axis is dur, separation as different lines
    Heatmap: with average probe lum for dur and separation.

    :param exp_path: dir containing participant folders
    :param p_names_list: names of participant's folders
    :param error_type: Default: None. Can pass sd or se for standard deviation or error.
    :param verbose: Default True, print progress to screen

    :returns: exp_ave_thr_df: experiment mean threshold for each separation and dur.
    """
    print("\n***running e_average_over_participants()***\n")

    """ part1. Munge data, save master lists and get means etc
     - loop through participants and get each MASTER_ave_TM_thresh.csv
    Make master sheets: MASTER_exp_thr and MASTER_exp_ave_thr."""

    all_p_ave_list = []
    for p_idx, p_name in enumerate(p_names_list):

        # look for trimmed mean df, if it doesn't exist, use untrimmed.
        ave_df_name = 'MASTER_ave_TM2_thresh'
        if not os.path.isfile(os.path.join(exp_path, p_name, f'{ave_df_name}.csv')):
            ave_df_name = 'MASTER_ave_TM1_thresh'
            if not os.path.isfile(os.path.join(exp_path, p_name, f'{ave_df_name}.csv')):
                ave_df_name = 'MASTER_ave_thresh'
                if not os.path.isfile(os.path.join(exp_path, p_name, f'{ave_df_name}.csv')):
                    raise FileNotFoundError(f"Can't find averages csv for {p_name}")

        this_p_ave_df = pd.read_csv(f'{exp_path}{os.sep}{p_name}{os.sep}{ave_df_name}.csv')

        if verbose:
            print(f'{p_idx}. {p_name} - {ave_df_name}:\n{this_p_ave_df}')

        if 'Unnamed: 0' in list(this_p_ave_df):
            this_p_ave_df.drop('Unnamed: 0', axis=1, inplace=True)

        rows, cols = this_p_ave_df.shape
        this_p_ave_df.insert(0, 'participant', [p_name] * rows)

        all_p_ave_list.append(this_p_ave_df)

    # join all participants' data and save as master csv
    all_exp_thr_df = pd.concat(all_p_ave_list, ignore_index=True)
    print(f'\nall_exp_thr_df:{list(all_exp_thr_df.columns)}\n{all_exp_thr_df}')

    if verbose:
        print(f'\nall_exp_thr_df:{list(all_exp_thr_df.columns)}\n{all_exp_thr_df}')
    all_exp_thr_df.to_csv(f'{exp_path}{os.sep}MASTER_exp_all_thr.csv', index=False)

    # # get means and errors
    get_means_df = all_exp_thr_df.drop('participant', axis=1)
    get_means_df = get_means_df.drop('stair_name', axis=1)
    get_means_df = get_means_df.drop('flow_name', axis=1)

    # todo: should I change sort to False for groupby?  Cause probelems in
    #  d_average_participants for error_df if there was only a single run of a
    #  condition so error was NaN and somehow order changed.
    exp_ave_thr_df = get_means_df.groupby('stair', sort=True).mean()
    if verbose:
        print(f'\nexp_ave_thr_df:\n{exp_ave_thr_df}')

    if error_type in [False, None]:
        error_bars_df = None
    elif error_type.lower() in ['se', 'error', 'std-error', 'standard error', 'standard_error']:
        error_bars_df = get_means_df.groupby('stair', sort=True).sem()
    elif error_type.lower() in ['sd', 'stdev', 'std_dev', 'std.dev', 'deviation', 'standard_deviation']:
        error_bars_df = get_means_df.groupby('stair', sort=True).std()
    else:
        raise ValueError(f"error_type should be in:\nfor none: [False, None]\n"
                         f"for standard error: ['se', 'error', 'std-error', 'standard error', 'standrad_error']\n"
                         f"for standard deviation: ['sd', 'stdev', 'std_dev', 'std.dev', "
                         f"'deviation', 'standard_deviation']")
    if verbose:
        print(f'\nerror_bars_df: ({error_type})\n{error_bars_df}')

    # save csv with average values
    exp_ave_thr_df.to_csv(f'{exp_path}{os.sep}MASTER_exp_ave_thr.csv')
    error_bars_df.to_csv(f'{exp_path}{os.sep}MASTER_ave_thr_error_{error_type}.csv')

    print("\n*** finished e_average_over_participants()***\n")

    return exp_ave_thr_df, error_bars_df


