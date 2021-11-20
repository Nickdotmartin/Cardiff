import pandas as pd
import numpy as np
import os

"""
This page contains python functions to mirror Martin's six MATLAB analysis scripts.
1. a_data_extraction: put data from one run, multiple ISIs into one array. 
2. b1_extract_last_values: get the threshold ased on the last values in each staircase for each ISI.
3. b2_lastReversal/m: Computes the threshold for each staircase as an aver§gwe of the last N reversals.
4. b3_plot_staircase.m:
5. c_plots.m:
6. d_averageParticipant.m:

So far I have just done the first two scripts.  I have also added:
a_data_extraction_all_runs: a version of the first script but can get all participant data across multiple runs.  
"""




def a_data_extraction(p_name, run_dir, ISI_list, save_all_data=True, verbose=True):

    """
    This script is a python version of Martin's first MATLAB analysis scripts, described below.

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

    print("\n***running a_data_extraction()***\n")

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
        this_ISI_df = this_ISI_df.sort_values(by=['stair', 'total_nTrials'])

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

    if verbose:
        print(f"all_data_df:\n{all_data_df}")

    if save_all_data:
        # Save xlsx in run folder if just one run, or participant folder if multiple runs.
        save_name = f'{run}_ALLDATA-sorted.xlsx'
        # save_name = 'ALLDATA-sorted.xlsx'

        save_excel_path = os.path.join(run_dir, save_name)
        if verbose:
            print(f"\nsaving all_data_df to save_excel_path:\n{save_excel_path}")
        all_data_df.to_excel(save_excel_path, index=False)

    print("\n***finished a_data_extraction()***\n")


    return all_data_df

# # # # # # #
# participant_name = 'Kim1'
# ISI_list = [-1, 0, 2, 4, 6, 9, 12, 24]
# # ISI_list = [0, 2]  # , 4, 6, 9, 12, 24, -1]
# run_dir = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/Kim/Nick_practice/P6a-Kim'
#
# a_data_extraction(p_name=participant_name, run_dir=run_dir, ISI_list=ISI_list, verbose=True)



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

    print("\n***running a_data_extraction_all_runs()***\n")

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
            this_ISI_df = this_ISI_df.sort_values(by=['stair', 'total_nTrials'])

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

    if verbose:
        print(f"all_data_df:\n{all_data_df}")

    if save_all_data:
        # Save xlsx in participant folder.
        save_name = f'{len(run_list)}runs_ALLDATA-sorted.xlsx'
        save_excel_path = os.path.join(p_dir, save_name)
        if verbose:
            print(f"\nsaving all_data_df to save_excel_path:\n{save_excel_path}")
        all_data_df.to_excel(save_excel_path, index=False)

    print("\n***finished a_data_extraction_all_runs()***\n")


    return all_data_df


# # # # # # #
# participant_name = 'Kim1'
# ISI_list = [0, 2, 4, 6, 9, 12, 24, -1]
# p_dir = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/Kim'
# run_list = ['P6a-Kim', 'P6b-Kim', 'P6c-Kim', 'P6d-Kim', 'P6e-Kim', 'P6f-Kim']
# a_data_extraction_all_runs(p_name=participant_name, p_dir=p_dir, run_list=run_list,
#                            ISI_list=ISI_list, verbose=True)


def b1_extract_last_values(all_data_path, thr_col='probeLum', resp_col='trial_response',
                           last_vals_list=None, verbose=True):

    """
    This script is a python version of Martin's second MATLAB analysis scripts, described below.

    b1_extract_last_values.m: For each ISI there are 14 staircase values,
    this script gets the last response for each of these values (last_response).
    It also get the last threshold for each staircase and the
    mean of the last 4 or 7 thresholds
    (thresholds-sorted-1last, thresholds-sorted-4last, thresholds-sorted-7last.

    :param all_data_path: path to the all_data.xlsx file
    :param thr_col: (default probeLum) name of the column showing the threshold vaired by the staircase.
    :param resp_col: (default: 'trial_response') name of the column showing accuracy per trial.
    :param last_vals_list: get the mean threshold of the last n values.
        It will use [1, 4, 7], unless another list is passed.
    :param verbose: If True, will print progress to screen.

    :return: nothing, but saves the files as:
            'last_response.csv' and f'threshold_sorted_{last_value}last.csv'
    """

    print("\n***running b1_extract_last_values()***")

    # extract path to save files to
    save_path, xlsx_name = os.path.split(csv_path)

    if last_vals_list == None:
        last_vals_list = [1, 4, 7]
    elif type(last_vals_list) == int:
        last_vals_list = [last_vals_list]
    elif type(last_vals_list) == list:
        if not all(type(x) is int for x in last_vals_list):
            raise TypeError(f'last_vals list should be list of ints, not {last_vals_list}.')
    else:
        raise TypeError(f'last_vals list should be list of ints, not {last_vals_list} {type(last_vals_list)}.')

    # open all_data file.  use engine='openpyxl' for xlsx files.
    # For other experiments it might be easier not to do use cols as they might be different.
    all_data_df = pd.read_excel(csv_path, engine='openpyxl',
                                usecols=['ISI', 'stair', 'total_nTrials',
                                         'probeLum', 'trial_response', 'resp.rt'])

    # get list of ISI and stair values to loop through
    ISI_list = all_data_df['ISI'].unique()
    stair_list = all_data_df['stair'].unique()

    # check last_vals_list are shorted than trials per stair.
    trials, columns = np.shape(all_data_df)
    trials_per_stair = int(trials/len(ISI_list)/len(stair_list))
    if max(last_vals_list) > trials_per_stair:
        raise ValueError(f'max(last_vals_list) ({max(last_vals_list)}) must be lower than trials_per_stair ({trials_per_stair}).')

    # get ISI string for column names
    ISI_name_list = [f'ISI{i}' for i in ISI_list]

    if verbose:
        print(f"last_vals_list: {last_vals_list}")
        print(f"{len(ISI_list)} ISI values and {len(stair_list)} stair values")
        print(f"all_data_df:\n{all_data_df}")


    # loop through last values (e.g., [1, 4, 7])
    for last_n_values in last_vals_list:
        if verbose:
            print(f"\nlast_n_values: {last_n_values}")

        # make empty arrays to add results into (rows=stairs, cols=ISIs)
        thr_array = np.zeros(shape=[len(stair_list), len(ISI_list)])
        resp_array = np.zeros(shape=[len(stair_list), len(ISI_list)])


        # loop through ISI values
        for ISI_idx, ISI in enumerate(ISI_list):
            if verbose:
                print(f"\n{ISI_idx}: ISI: {ISI}")

            # get df for this ISI only
            ISI_df = all_data_df[all_data_df['ISI'] == ISI]


            # loop through stairs for this ISI
            for stair_idx, stair in enumerate(stair_list):

                # get df just for one stair at this ISI
                stair_df = ISI_df[ISI_df['stair'] == stair]
                if verbose:
                    print(f'\nstair_df (stair={stair}, ISI={ISI}, last_n_values={last_n_values}):\n{stair_df}')

                # get the mean threshold of the last n values (last_n_values)
                mean_thr = np.mean(list(stair_df[thr_col])[-last_n_values:])
                if verbose:
                    if last_n_values > 1:
                        print(f'last {last_n_values} values: {list(stair_df[thr_col])[-last_n_values:]}')
                    print(f'mean_thr: {mean_thr}')

                # copy value into threshold array
                thr_array[stair_idx, ISI_idx] = mean_thr


                if last_n_values == 1:
                    last_response = list(stair_df[resp_col])[-last_n_values]
                    if verbose:
                        print(f'last_response: {last_response}')

                    # copy value into response array
                    resp_array[stair_idx, ISI_idx] = last_response


        # make dataframe from array
        thr_df = pd.DataFrame(thr_array, columns=ISI_name_list)
        thr_df.insert(0, 'stair', stair_list)
        if verbose:
            print(f"thr_df:\n{thr_df}")

        # save response and threshold arrays
        thr_filename = f'threshold_sorted_{last_n_values}last.csv'
        thr_filepath = os.path.join(save_path, thr_filename)
        thr_df.to_csv(thr_filepath, index=False)

        if last_n_values == 1:
            # make dataframe from array
            resp_df = pd.DataFrame(resp_array, columns=ISI_name_list)
            resp_df.insert(0, 'stair', stair_list)
            if verbose:
                print(f"resp_df:\n{resp_df}")

            # save response and threshold arrays
            resp_filename = 'last_response.csv'
            resp_filepath = os.path.join(save_path, resp_filename)
            resp_df.to_csv(resp_filepath, index=False)

    print("\n***finished b1_extract_last_values()***")



# # # # # # # # #
# csv_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/Kim/Nick_practice/P6a-Kim/P6a-Kim_ALLDATA-sorted.xlsx'
# b1_extract_last_values(all_data_path=csv_path)


"""
3. b2_lastReversal/m: Computes the threshold for each staircase as an aver§gwe of the last N reversals.
4. b3_plot_staircase.m:"""

"""
5. c_plots.m:
6. d_averageParticipant.m:

"""




