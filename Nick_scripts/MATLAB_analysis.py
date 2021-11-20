import pandas as pd
import numpy as np
import os

def a_data_extraction(p_name, run_dir, ISI_list, save_all_data=True, verbose=True):

    """
    This script is a python version of Martin's six MATLAB analysis scripts, described below.

    a_data_extraction.m: Once a participant has completed a run of all ISIs,
        this script gets all their data into one file, and sorts each ISI by stair.

    :param p_name: participant's name as used to save csv files.  e.g., if the
            file is .../nick1.csv, participant name is 'nick1'.
    :param run_dir: directory where ISI folders are stored.
    :param ISI_list: List of ISI values, may differ between experiments.
    :param save_all_data: If True, will save all_data_df as an xlsx.
    :param verbose: If True, will print progress to screen.

    :return: A pandas DataFrame with n xlsx file of all data for one run of all ISIs.
    """

    # get run name/number
    path, run = os.path.split(run_dir)
    if verbose:
        print(f"run: {run}")

    # empty array to append info into
    all_data = []

    # loop through ISIs in each run.
    for ISI in ISI_list:
        filepath = f'{run_dir}{os.path.sep}ISI_{ISI}_probeDur2{os.path.sep}' \
                   f'{p_name}.csv'
        if verbose:
            print(f"filepath: {filepath}")

        # load data
        this_ISI_df = pd.read_csv(filepath)
        if verbose:
            print(f"loaded csv:\n{this_ISI_df.head()}")

        # sort by staircase
        trial_numbers = list(this_ISI_df['total_nTrials'])
        this_ISI_df = this_ISI_df.sort_values(by='stair')

        # add ISI column for multi-indexing
        this_ISI_df.insert(0, 'ISI', ISI)
        this_ISI_df.insert(1, 'srtd_trial_idx', trial_numbers)
        if verbose:
            print(f'df sorted by stair:\n{this_ISI_df.head()}')

        # get column names to use on all_data_df
        column_names = list(this_ISI_df)

        # add to all_data
        all_data.append(this_ISI_df)


    # create all_data_df - reshape to 2d
    all_data_shape = np.shape(all_data)
    sheets, rows, columns = np.shape(all_data)
    all_data = np.reshape(all_data, newshape=(sheets*rows, columns))
    if verbose:
        print(f'all_data reshaped from {all_data_shape} to {np.shape(all_data)}')
    all_data_df = pd.DataFrame(all_data, columns=column_names)

    # # make multi-index
    all_data_df.set_index(['ISI', 'srtd_trial_idx'], inplace=True)
    if verbose:
        print(f"all_data_df:\n{all_data_df}")

    if save_all_data:
        # Save xlsx in run folder if just one run, or participant folder if multiple runs.
        save_name = f'{run}_ALLDATA-sorted.xlsx'
        # save_name = 'ALLDATA-sorted.xlsx'

        save_excel_path = os.path.join(run_dir, save_name)
        if verbose:
            print(f"\nsaving all_data_df to save_excel_path:\n{save_excel_path}")
        all_data_df.to_excel(save_excel_path)

    return all_data_df

# # # # # #
participant_name = 'Kim1'
ISI_list = [0, 2, 4, 6, 9, 12, 24, -1]
# ISI_list = [0, 2]  # , 4, 6, 9, 12, 24, -1]
run_dir = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/Kim/Nick_practice/P6a-Kim'

a_data_extraction(p_name=participant_name, run_dir=run_dir, ISI_list=ISI_list, verbose=True)

def a_data_extraction_all_runs(p_name, p_dir, run_list, ISI_list, save_all_data=True, verbose=True):
    """
    This script is a python version of Martin's six MATLAB analysis scripts, described below.

    a_data_extraction.m: Once a participant has completed a run of all ISIs,
        this script gets all their data into one file, and sorts each ISI by stair.

    NEW here: This script can also take multiple directories to collate
        ALL participant info, for multiple runs, all ISIs.

    :param p_name: participant's name as used to save files.  This should not include the run number.
    e.g., if the file is .../nick1.csv, participant name is just 'nick'.
    :param p_dir: directory where the participant's data is stored (e.g. multiple runs).
    :param run_list: a list of all run directory names.
    :param ISI_list: List of ISI values, may differ between experiments.
    :param save_all_data: If True, will save all_data_df as an xlsx.
    :param verbose: If True, will print progress to screen.

    :return: A pandas DataFrame with n xlsx file of all data for one run of all ISIs.
        If multiple run_list are passes, then the output contains all data for one participant.
    """

    # strip run number from participant name if present
    if p_name[-1].isdigit():
        if verbose:
            print(f'last digit stripped from: {p_name}')
        p_name = p_name[:-1]
        if verbose:
            print(f'renamed as: {p_name}')

    # empty array to append info into
    all_data = []

    # loop through runs if multiple runs are passed.
    for run_idx, run in enumerate(run_list):
        run_num = run_idx + 1
        if verbose:
            print(f"\nrun_num: {run_num}; run: {run}")

        # loop through ISIs in each run.
        for ISI in ISI_list:
            filepath = f'{p_dir}{os.path.sep}{run}{os.path.sep}ISI_{ISI}_probeDur2{os.path.sep}' \
                       f'{p_name}{run_num}.csv'
            this_ISI_df = pd.read_csv(filepath)
            if verbose:
                print(f'\nISI: {ISI}')
                print(f"filepath: {filepath}")
                print(f"loaded csv:\n{this_ISI_df.head()}")

            # sort by staircase
            trial_numbers = list(this_ISI_df['total_nTrials'])
            this_ISI_df = this_ISI_df.sort_values(by='stair')

            # add ISI column for multi-indexing
            this_ISI_df.insert(0, 'run', run)
            this_ISI_df.insert(1, 'ISI', ISI)
            this_ISI_df.insert(2, 'srtd_trial_idx', trial_numbers)
            if verbose:
                print(f'df sorted by stair:\n{this_ISI_df.head()}')

            # get column names to use on all_data_df
            column_names = list(this_ISI_df)

            # add to all_data
            all_data.append(this_ISI_df)


    # create all_data_df - reshape to 2d
    all_data_shape = np.shape(all_data)
    if len(all_data_shape) == 3:
        sheets, rows, columns = np.shape(all_data)
        all_data = np.reshape(all_data, newshape=(sheets * rows, columns))
        if verbose:
            print(f'\nall_data reshaped from {all_data_shape} to {np.shape(all_data)}')

    all_data_df = pd.DataFrame(all_data, columns=column_names)

    # # make multi-index
    all_data_df.set_index(['run', 'ISI', 'srtd_trial_idx'], inplace=True)
    if verbose:
        print(f"all_data_df:\n{all_data_df}")

    if save_all_data:
        # Save xlsx in participant folder.
        save_name = f'{len(run_list)}runs_ALLDATA-sorted.xlsx'
        save_excel_path = os.path.join(p_dir, save_name)
        if verbose:
            print(f"\nsaving all_data_df to save_excel_path:\n{save_excel_path}")
        all_data_df.to_excel(save_excel_path)

    return all_data_df


# # # # # # #
# participant_name = 'Kim1'
# ISI_list = [0, 2, 4, 6, 9, 12, 24, -1]
# p_dir = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/Kim'
# run_list = ['P6a-Kim', 'P6b-Kim', 'P6c-Kim', 'P6d-Kim', 'P6e-Kim', 'P6f-Kim']
# a_data_extraction_all_runs(p_name=participant_name, p_dir=p_dir, run_list=run_list,
#                            ISI_list=ISI_list, verbose=True)


"""
2. b1_extract_last_values.m: For each ISI there are 14 staircase values,
    this script gets the last response for each of these values (last_response).
    It also get the last threshold for each staircase and the
    mean of the last 4 or 7 thresholds
    (thresholds-sorted-1last, thresholds-sorted-4last, thresholds-sorted-7last.
3. b2_lastReversal/m: Computes the threshold for each staircase as an aver§gwe of the last N reversals.
4. b3_plot_staircase.m:
5. c_plots.m:
6. d_averageParticipant.m:

"""




