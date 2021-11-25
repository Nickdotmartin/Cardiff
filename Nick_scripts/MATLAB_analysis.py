import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

"""
This page contains python functions to mirror Martin's six MATLAB analysis scripts.
1. a_data_extraction: put data from one run, multiple ISIs into one array. 
2. b1_extract_last_values: get the threshold used on the last values in each 
    staircase for each ISI.
3. b2_lastReversal/m: Computes the threshold for each staircase as an average of 
    the last N reversals (incorrect responses).  Plot these as mean per sep, 
    and also multiplot for each ISI with different values for -18&+18 etc
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
    save_path, xlsx_name = os.path.split(all_data_path)

    if last_vals_list is None:
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
    all_data_df = pd.read_excel(all_data_path, engine='openpyxl',
                                usecols=['ISI', 'stair', 'total_nTrials',
                                         'probeLum', 'trial_response', 'resp.rt'])

    # get list of ISI and stair values to loop through
    ISI_list = all_data_df['ISI'].unique()
    stair_list = all_data_df['stair'].unique()

    # check last_vals_list are shorted than trials per stair.
    trials, columns = np.shape(all_data_df)
    trials_per_stair = int(trials/len(ISI_list)/len(stair_list))
    if max(last_vals_list) > trials_per_stair:
        raise ValueError(f'max(last_vals_list) ({max(last_vals_list)}) must be '
                         f'lower than trials_per_stair ({trials_per_stair}).')

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
# test_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/Kim/Nick_practice/P6a-Kim/P6a-Kim_ALLDATA-sorted.xlsx'
# b1_extract_last_values(all_data_path=test_path)


def b2_lastReversal(all_data_path, reversals_list=[2, 3, 4],
                   thr_col='probeLum', resp_col='trial_response',
                   show_plots=True, save_plots=True, verbose=True):
    """
    b2_lastReversal/m: Computes the threshold for each staircase as an average of
    the last N reversals (incorrect responses).  Plot these as mean per sep,
    and also multiplot for each ISI with different values for -18&+18 etc

    However: although each sep values is usd twice (e.g., stairs 1 & 2 are both sep = 18),
    these do not correspond to any meaningful difference (e.g., target_jump 1 or -1)
    as target_jump is decided with random.choice, and this variable is not accessed
    in this analysis.  For plots that show +18 and -18, these are duplicated values.
    I intend to design the sacript to allow for meaningful differences here -
    but I also need to copy this script first to check I have the same procedure
    in case I have missed something.

    :param all_data_path: path to the all_data xlsx file.
    :param reversals_list: list of values to calculate scores from.
        e.g., if reversals_list=[2, 3, 4], will get the mean threshold from last
        n reversals where n=2, 3 or 4.
    :param thr_col: (default probeLum) name of the column showing the threshold
        (e.g., varied by the staircase).
    :param resp_col: (default: 'trial_response') name of the column showing
        (accuracy per trial).
    :param show_plots: whether to display plots on-screen.
    :param save_plots: whether to save the plots.
    :param verbose: If True, will print progress to screen.

    :return: arrays with details of last n reversals.
            Plot of mean for last reversal - all ISIs shows as different lines.
            Batplots with pos and neg sep, separate grid for each ISI
    """
    print("\n***running b2_lastReversal()***\n")

    save_path, xlsx_name = os.path.split(all_data_path)

    # open all_data file.  use engine='openpyxl' for xlsx files.
    # For other experiments it might be easier not to do use cols as they might be different.
    all_data_df = pd.read_excel(all_data_path, engine='openpyxl',
                                usecols=['ISI', 'stair', 'total_nTrials',
                                         'probeLum', 'trial_response', 'resp.rt'])

    # get list of ISI and stair values to loop through
    ISI_list = all_data_df['ISI'].unique()
    stair_list = all_data_df['stair'].unique()
    if verbose:
        print(f"ISI_list: {ISI_list}\nstair_list: {stair_list}")

    # get ISI string for column names
    ISI_name_list = ['concurrent' if i == -1 else f'ISI{i}' for i in ISI_list]
    if verbose:
        print(f"ISI_name_list: {ISI_name_list}")
        print(f"{len(ISI_list)} ISI values and {len(stair_list)} stair values")
        print(f"all_data_df:\n{all_data_df}")

    # get results for n reversals
    for reversals in reversals_list:

        # make empty arrays to add results into (rows=stairs, cols=ISIs)
        mean_rev_lum = np.zeros(shape=[len(stair_list), len(ISI_list)])

        # loop through ISI values
        for ISI_idx, ISI in enumerate(ISI_list):

            # get df for this ISI only
            ISI_df = all_data_df[all_data_df['ISI'] == ISI]

            # loop through stairs for this ISI
            for stair_idx, stair in enumerate(stair_list):

                # get df just for one stair at this ISI
                stair_df = ISI_df[ISI_df['stair'] == stair]
                if verbose:
                    print(f'\nstair_df (stair={stair}, ISI={ISI}, reversals={reversals}):\n{stair_df}')

                    # get indices of last n incorrect responses
                    incorrect_list = stair_df.index[stair_df[resp_col] == 0]

                    # get probeLum for corresponding trials
                    reversal_probeLum_list = stair_df[thr_col].loc[incorrect_list]

                    # just select last n reversals - or whole list if list is shorter than n
                    reversal_probeLum_list = reversal_probeLum_list[-reversals:]

                    # get mean of these probeLums
                    mean_lum = np.mean(list(reversal_probeLum_list))

                    # append to mean_rev_lum array
                    mean_rev_lum[stair_idx, ISI_idx] = mean_lum

        if verbose:
            print(f"mean_rev_lum:\n{mean_rev_lum}")

        # MAKE SYMMETRICAL
        # this has got some dodgy stuff going on here - it truely is symmetyrical as
        # data is copied (e.g, -18 = +18), but presumably there should be different
        # data for positive and negative separations.

        # MATLAB version uses reversalThresh1sym (&2) then takes the mean of these with reversal_threshMean
        # reversalThresh1sym is the variable used in the MATLAB scripts - works with 14 stairs
        reversalThresh1sym = np.array([mean_rev_lum[0, :],  mean_rev_lum[2, :],
                                       mean_rev_lum[4, :], mean_rev_lum[6, :],
                                       mean_rev_lum[8, :], mean_rev_lum[10, :],
                                       mean_rev_lum[8, :], mean_rev_lum[6, :],
                                       mean_rev_lum[4, :], mean_rev_lum[2, :],
                                       mean_rev_lum[0, :], mean_rev_lum[12, :]])

        reversalThresh2sym = np.array([mean_rev_lum[1, :], mean_rev_lum[3, :],
                                       mean_rev_lum[5, :], mean_rev_lum[7, :],
                                       mean_rev_lum[9, :], mean_rev_lum[11, :],
                                       mean_rev_lum[9, :], mean_rev_lum[7, :],
                                       mean_rev_lum[5, :], mean_rev_lum[3, :],
                                       mean_rev_lum[1, :], mean_rev_lum[13, :]])

        reversalThreshMean = np.mean(np.array([reversalThresh1sym, reversalThresh2sym]), axis=0)

        # dataframes for figures
        rev_thr_mean_df = pd.DataFrame(data=reversalThreshMean, columns=ISI_name_list)
        sym_sep_list = [-18, -6, -3, -2, -1, 0, 1, 2, 3, 6, 18, 20]
        rev_thr_mean_df.insert(loc=0, column='Separation', value=sym_sep_list)
        if verbose:
            print(f"\nrev_thr_mean_df:\n{rev_thr_mean_df}")

        rev_thr1_sym_df = pd.DataFrame(data=reversalThresh1sym, columns=ISI_name_list)
        rev_thr1_sym_df.insert(loc=0, column='Separation', value=sym_sep_list)

        rev_thr2_sym_df = pd.DataFrame(data=reversalThresh2sym, columns=ISI_name_list)
        rev_thr2_sym_df.insert(loc=0, column='Separation', value=sym_sep_list)

        # New version should take stairs in this order I think (assuming pos first then neg)
        # sep: -18, -6, -3, -2, -1, 0 & 0, 1, 2, 3, 6, 18, 99&99
        # stair: 1,  3,  5,  7,  9, 10&11, 8, 6, 4, 2, 0, 12&13  if 0 indexed
        # stair: 2,  4,  6,  8, 10, 11&12, 9, 7, 5, 3, 1, 13&14 if 1 indexed

        # get mean difference between pairs of sep values for evaluating analysis,
        # method with lowest mean difference is least noisy method. (for fig2)
        # for each pair of sep values (e.g., stair1&2, stair3&4) subtract one from other.
        # get abs of all values them sum the columns (ISIs)
        diffNext = np.sum(abs(reversalThresh1sym-reversalThresh2sym), axis=0)
        # take the mean of these across all ISIs to get single value
        meandiffNext = np.mean(diffNext)



        # PLOT FIGURES
        # choose colours
        use_colours = 'tab10'
        n_conditions = int(len(ISI_name_list))
        if 10 < n_conditions < 21:
            use_colours = 'tab20'
        elif n_conditions > 20:
            print("\tERROR - more classes than colours!?!?!?")
        sns.set_palette(palette=use_colours, n_colors=n_conditions)
        my_colours = sns.color_palette()


        # FIGURE 1 - shows one axis (x=separation (0-18), y=probeLum) with all ISIs added.
        # it also seems that for ISI=99 there are simple dots added at -1 on the x axis.

        # only take positive separation values
        fig1_df = rev_thr_mean_df[rev_thr_mean_df['Separation'] >= 0]
        fig1_df = fig1_df.set_index('Separation')

        # the values listed as separation=20 are actually for the single probe cond.
        # Chop last row off and add values later.
        fig1_df, one_probe_df = fig1_df.drop(fig1_df.tail(1).index), fig1_df.tail(1)

        # change value from 20 to -1 so its on the left of the plot
        one_probe_lum_list = one_probe_df.values.tolist()[0]
        one_probe_dict = {'ISIs': ISI_name_list,
                          'ProbeLum': one_probe_lum_list,
                          'x_vals': [-1 for i in ISI_name_list]}
        one_probe_df = pd.DataFrame.from_dict(one_probe_dict)

        # make fig1
        fig, ax = plt.subplots(figsize=(10, 6))

        # line plot for main ISIs
        sns.lineplot(data=fig1_df, markers=True, dashes=False)

        # scatter plot for single probe conditions
        sns.scatterplot(data=one_probe_df, x="x_vals", y="ProbeLum",
                        hue="ISIs", style='ISIs')

        # decorate plot
        ax.legend(labels=ISI_name_list, title='ISI',
                  shadow=True,
                  # place lower left corner of legend at specified location.
                  loc='lower left', bbox_to_anchor=(0.96, 0.5))

        ax.set_xticks([-2, -1, 0, 1, 2, 3, 6, 18])
        ax.set_xticklabels(['', 'one\nprobe', 0, 1, 2, 3, 6, 18])

        ax.set_xlabel('Probe separation in diagonal pixels')
        ax.set_ylabel('Probe Luminance')

        fig1_title = f'Mean {thr_col} from last {reversals} reversals'
        plt.title(fig1_title)

        # save, show and close plots
        if save_plots:
            fig1_savename = f'data_last{reversals}_reversals.png'
            plt.savefig(f'{save_path}{os.sep}{fig1_savename}')

        if show_plots:
            plt.show()
        plt.close()

        # FIGURE 2
        # this is a figure with one axis per ISI, showing neg and pos sep
        # (e.g., -18:18)
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12, 6))
        fig.suptitle(f'Last {reversals} reversals per ISI. '
                     f'(mean diff: {round(meandiffNext, 2)})')

        fig2_x_tick_lab = [-18, -6, -3, -2, -1, 0, 1, 2, 3, 6, 18, '1\nprobe']

        ax_counter = 0
        for row_idx, row in enumerate(axes):
            for col_idx, ax in enumerate(row):
                print(row_idx, col_idx, ax)

                # mean threshold from CW and CCW prode jump direction
                sns.lineplot(ax=axes[row_idx, col_idx], data=rev_thr_mean_df,
                             x='Separation', y=ISI_name_list[ax_counter],
                             color=my_colours[ax_counter],
                             linewidth=3, markers=True)

                # stair1: CW probe jumps only
                sns.lineplot(ax=axes[row_idx, col_idx], data=rev_thr1_sym_df,
                             x='Separation', y=ISI_name_list[ax_counter],
                             color=my_colours[ax_counter],
                             linewidth=.5, marker="v")

                # stair2: CCW probe jumps only
                sns.lineplot(ax=axes[row_idx, col_idx], data=rev_thr2_sym_df,
                             x='Separation', y=ISI_name_list[ax_counter],
                             color=my_colours[ax_counter],
                             linewidth=.5, marker="o")

                ax.set_title(ISI_name_list[ax_counter])
                ax.set_xticks(sym_sep_list)
                ax.set_xticklabels(fig2_x_tick_lab)
                ax.xaxis.set_tick_params(labelsize=6)
                ax.set_ylim([40, 90])

                if row_idx == 1:
                    ax.set_xlabel('Probe separation (pixels)')
                else:
                    ax.xaxis.label.set_visible(False)

                if col_idx == 0:
                    ax.set_ylabel('Probe Luminance')
                else:
                    ax.yaxis.label.set_visible(False)

                ax.text(x=0.4, y=0.8, s=round(diffNext[ax_counter], 2),
                        # needs transform to appear with rest of plot.
                        transform=ax.transAxes, fontsize=12)

                # artist for legend
                st1 = mlines.Line2D([], [], color=my_colours[ax_counter],
                                    marker='v', linewidth=.5,
                                    markersize=4, label='Stair1')
                st2 = mlines.Line2D([], [], color=my_colours[ax_counter],
                                    marker='o', linewidth=.5,
                                    markersize=4, label='Stair2')
                mean_line = mlines.Line2D([], [], color=my_colours[ax_counter],
                                    marker=None, linewidth=3, label='mean')
                ax.legend(handles=[st1, st2, mean_line], fontsize=6)

                ax_counter += 1

        plt.tight_layout()

        # save, show and close plots
        if save_plots:
            fig2_savename = f'runs_last{reversals}_reversals.png'
            plt.savefig(f'{save_path}{os.sep}{fig2_savename}')

        if show_plots:
            plt.show()
        plt.close()

        print("\n***finished b2_lastReversal()***\n")


# ################
# all_data_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/Kim/' \
#                 'Nick_practice/P6a-Kim/P6a-Kim_ALLDATA-sorted.xlsx'
# b2_lastReversal(all_data_path=all_data_path, reversals_list=[2, 3, 4],
#                thr_col='probeLum', resp_col='trial_response',
#                show_plots=True, save_plots=True,
#                verbose=True)


"""
4. b3_plot_staircase.m: staircases-ISIxxx.png: xxx corresponds to ISI conditions. 
    Eight figure (8 ISI conditions) with seven panels on each (7 probes separation 
    conditions) showing the Luminance value of two staircases in function of 
    trial number. 
"""


all_data_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/Kim/' \
                'Nick_practice/P6a-Kim/P6a-Kim_ALLDATA-sorted.xlsx'

thr_col='probeLum'
show_plots=True
save_plots=True
verbose=True

save_path, xlsx_name = os.path.split(all_data_path)

# open all_data file.  use engine='openpyxl' for xlsx files.
# For other experiments it might be easier not to do use cols as they might be different.
all_data_df = pd.read_excel(all_data_path, engine='openpyxl',
                            usecols=['ISI', 'stair', 'total_nTrials',
                                     'probeLum', 'trial_response', 'resp.rt'])

# get list of ISI and stair values to loop through
ISI_list = all_data_df['ISI'].unique()
stair_list = all_data_df['stair'].unique()

# get ISI string for column names
ISI_name_list = ['concurrent' if i == -1 else f'ISI{i}' for i in ISI_list]

trials, columns = np.shape(all_data_df)
trials_per_stair = int(trials/len(ISI_list)/len(stair_list))

separation_title = ['18sep', '06sep', '03sep', '02sep', '01sep', '00sep', 'onePb']


if verbose:
    print(f"all_data_df:\n{all_data_df}")
    print(f"{len(ISI_list)} ISI values and {len(stair_list)} stair values")
    print(f"ISI_list: {ISI_list}")
    print(f"ISI_name_list: {ISI_name_list}")
    print(f"stair_list: {stair_list}")
    print(f"trials_per_stair: {trials_per_stair}")
    print(f"separation_title: {separation_title}")



x_tick_lab = list(range(trials_per_stair))

ISI_list = [-1]

# loop through ISI values
for ISI_idx, ISI in enumerate(ISI_list):

    # get df for this ISI only
    ISI_df = all_data_df[all_data_df['ISI'] == ISI]

    # initialise 8 plot figure
    # # this is a figure showing reversals per staircase condition.
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12, 6))
    ax_counter = 0

    for row_idx, row in enumerate(axes):
        for col_idx, ax in enumerate(row):
            print(f'\nrow: {row_idx}, col: {col_idx}: {ax}')

            if ax_counter < 7:
                # print(row_idx, col_idx, ax)

                # # get pairs of stairs (e.g., [[18, -18], [6, -6], ...etc)
                # for stair_idx, stair in enumerate(stair_list):
                # stair = stair_list[ax_counter]

                stair_odd = (ax_counter*2)+1  # 1, 3, 5, 7, 9, 11, 13
                stair_odd_df = ISI_df[ISI_df['stair'] == stair_odd]
                stair_odd_df.insert(0, 'step', list(range(trials_per_stair)))
                final_lum_odd = stair_odd_df.loc[stair_odd_df['step'] == trials_per_stair-1, 'probeLum'].item()
                n_reversals_odd = trials_per_stair - stair_odd_df['trial_response'].sum()

                stair_even = (ax_counter+1)*2  # 2, 4, 6, 8, 10, 12, 14
                stair_even_df = ISI_df[ISI_df['stair'] == stair_even]
                stair_even_df.insert(0, 'step', list(range(trials_per_stair)))
                final_lum_even = stair_even_df.loc[stair_even_df['step'] == trials_per_stair-1, 'probeLum'].item()
                n_reversals_even = trials_per_stair - stair_even_df['trial_response'].sum()

                if verbose:
                    print(f'\nstair_odd_df (stair={stair_odd}, ISI={ISI}:\n{stair_odd_df.head()}')
                    print(f"final_lum_odd: {final_lum_odd}")
                    print(f"n_reversals_odd: {n_reversals_odd}")

                    print(f'\nstair_even_df (stair={stair_even}, ISI={ISI}:\n{stair_even_df.tail()}')
                    print(f"final_lum_even: {final_lum_even}")
                    print(f"n_reversals_even: {n_reversals_even}")

                '''
                use multiplot method from figure 2 above.
                There is also a horizontal line from the last value (step25)
                There is text showing the number of reversals (incorrect responses)
                y-axis can be 0:106 (maxLum), x is 1:25.
                
                later the 8th panel is added - not sure what this is yet...
                '''

                fig.suptitle(f'Staircases and reversals for ISI {ISI}')

                # plot thr per step for odd numbered stair
                sns.lineplot(ax=axes[row_idx, col_idx], data=stair_odd_df,
                             x='step', y=thr_col,
                             color='red',
                             marker="v", markersize=5)

                # line for final probeLum
                ax.axhline(y=final_lum_odd, color='red')

                # text for n_reversals
                ax.text(x=0.4, y=0.9, s=f'{n_reversals_odd} reversals',
                        color='red',
                        # needs transform to appear with rest of plot.
                        transform=ax.transAxes, fontsize=12)

                # plot thr per step for even numbered stair
                sns.lineplot(ax=axes[row_idx, col_idx], data=stair_even_df,
                             x='step', y=thr_col,
                             color='blue',
                             # linewidth=3,
                             marker="o", markersize=4)
                ax.axhline(y=final_lum_even, color='blue')
                ax.text(x=0.4, y=0.7, s=f'{n_reversals_even} reversals',
                        color='blue',
                        # needs transform to appear with rest of plot.
                        transform=ax.transAxes, fontsize=12)

                ax.set_title(f'{ISI_name_list[ISI_idx]} {separation_title[ax_counter]}')
                ax.set_xticks(np.arange(0, trials_per_stair, 5))
                ax.set_ylim([0, 110])

                # if row_idx == 1:
                #     ax.set_xlabel('Probe separation (pixels)')
                # else:
                #     ax.xaxis.label.set_visible(False)
                #
                # if col_idx == 0:
                #     ax.set_ylabel('Probe Luminance')
                # else:
                #     ax.yaxis.label.set_visible(False)
                #
                # ax.text(x=0.4, y=0.8, s=round(diffNext[ax_counter], 2),
                #         # needs transform to appear with rest of plot.
                #         transform=ax.transAxes, fontsize=12)
                #
                # # artist for legend
                # st1 = mlines.Line2D([], [], color=my_colours[ax_counter],
                #                     marker='v', linewidth=.5,
                #                     markersize=4, label='Stair1')
                # st2 = mlines.Line2D([], [], color=my_colours[ax_counter],
                #                     marker='o', linewidth=.5,
                #                     markersize=4, label='Stair2')
                # mean_line = mlines.Line2D([], [], color=my_colours[ax_counter],
                #                           marker=None, linewidth=3, label='mean')
                # ax.legend(handles=[st1, st2, mean_line], fontsize=6)

            else:
                print("Eighth plot")

            ax_counter += 1

    plt.tight_layout()

    # save, show and close plots
    # if save_plots:
    #     fig2_savename = f'runs_last{reversals}_reversals.png'
    #     plt.savefig(f'{save_path}{os.sep}{fig2_savename}')

    if show_plots:
        plt.show()
    plt.close()

    # for the 8th plot, use threshold_sorted_1last.csv

print("\n***finished b3_plot_staircases()***\n")

"""
5. c_plots.m:
6. d_averageParticipant.m:

"""

print('\nend of script')
