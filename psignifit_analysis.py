import os

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

"""
This page contains python functions to analyse the radial_flow.py experiment.
The workflow is based on Martin's MATLAB scripts used in MATLAB_analysis.py.

1. a_data_extraction: put data from one run, multiple ISIs into one array. 
2. b1_extract_last_values: get the threshold used on the last values in each 
    staircase for each isi.  this is used for b3_plot_staircases.
# not used. 3. b2_last_reversal/m: Computes the threshold for each staircase as an average of 
#     the last N reversals (incorrect responses).  Plot these as mean per sep, 
#     and also multi-plot for each isi with different values for -18&+18 etc
4. b3_plot_staircase:
5. c_plots: get psignifit thr values, 
    plot:   data_psignifit_values (pos_sep_no_one_probe)
            runs_psignifit_values (eight_batman_plots)
6. d_averageParticipant:
    master_psignifit_thr (all runs, one csv)
    master_average_psignifit_thr (mean of all runs)
    plot:
    master_average_psignifit_thr (pos_sep_no_one_probe)


I've also added functions for repeated bit of code:
data: 

split_df_alternate_rows(): 
    split a dataframe into two dataframes: 
        one with positive sep values, one with negative

merge_pos_and_neg_sep_dfs():
    merge two dataframes (one with pos sep values, one with negative) back into 
    one combined df in original order

split_df_into_pos_sep_df_and_one_probe_df():
    code to turn array into symmetrical array (e.g., from sep=[0, 1, 2, 3, 6, 18] 
    into sep=[-18, -6, -3, -2, -1, 0, 1, 2, 3, 6, 18])

split df into pos_sep_df and one_probe_df


plots: 
plot_pos_sep_and_one_probe: single ax with pos separation, with/without single probe scatter at -1
Batman plots: 


"""

# todo: none of my studies have the one-probe ISI condition, so get rid of this (isi=-1 or sep=99)
# todo: none of my studies have the concurrent ISI condition, so get rid of that
# todo: sep values + and - relate to inward and outward target jump.  This is only of small interest, not a main factors
# todo: target_flow_same refers to whether the target and flow are both in same
#  direction (inwards, outwards), this is of more interest than target jump and should be treated as such.
#  e.g., analysis scripts should dig into this factor


def split_df_alternate_rows(df):
    """
    Split a dataframe into alternate rows.  Dataframes are organized by
    stair conditions relating to separations
    (e.g., in order 18, -18, 6, -6, 3, -3, 2, -2, 1, -1, 0, 0, 99, -99).
    For some plots this needs to be split into two dfs, e.g., :
    pos_sep_df: [18, 6, 3, 2, 1, 0, 99]
    neg_sep_df: [-18, -6, -3, -2, -1, 0, -99]

    :param df: Dataframe to be split in two
    :return: two dataframes: pos_sep_df, neg_sep_df
    """
    print("\n*** running split_df_alternate_rows() ***")

    n_rows, n_cols = df.shape
    pos_nums = list(range(0, n_rows, 2))
    neg_nums = list(range(1, n_rows, 2))
    pos_sep_df = df.iloc[pos_nums, :]
    neg_sep_df = df.iloc[neg_nums, :]
    pos_sep_df.reset_index(drop=True, inplace=True)
    neg_sep_df.reset_index(drop=True, inplace=True)

    return pos_sep_df, neg_sep_df


#######
# save_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/Kim/Nick_practice/P6a-Kim'
# last_response_df = pd.read_csv(f'{save_path}{os.sep}last_response.csv', dtype=int)
# print(f'last_response_df:\n{last_response_df}')
# df1, df2 = split_df_alternate_rows(last_response_df)
# print(df1)
# print(df2)

def merge_pos_and_neg_sep_dfs(pos_sep_df, neg_sep_df):
    """
    Takes two dataframes relating to positive and negative separation values:
    (e.g., pos_sep_df: [18, 6, 3, 2, 1, 0, 99], neg_sep_df: [-18, -6, -3, -2, -1, 0, -99])
    and merges them into single df with original order
    (e.g., merged_df order [18, -18, 6, -6, 3, -3, 2, -2, 1, -1, 0, 0, 99, -99]).
    Merge is performed based on the index numbers of arrays.

    :param pos_sep_df: dataframe containing data for positive separations
                (e.g., [18, 6, 3, 2, 1, 0, 99])
    :param neg_sep_df: dataframe containing data for positive separations
                (e.g., [-18, -6, -3, -2, -1, 0, -99])
    :return: Combined_df
    """
    print("\n*** running merge_pos_and_neg_sep_dfs() ***")

    # check dfs have same shape
    if pos_sep_df.shape != neg_sep_df.shape:
        raise ValueError(f'Dataframes to be merged must have same shape.\n'
                         f'pos_sep_df: {pos_sep_df.shape}, neg_sep_df: {neg_sep_df.shape}')
    # else:
    #     print('shapes match :)')

    # reset indices for sorting
    pos_sep_df.reset_index(drop=True, inplace=True)
    neg_sep_df.reset_index(drop=True, inplace=True)

    merged_df = pd.concat([pos_sep_df, neg_sep_df]).sort_index()
    merged_df.reset_index(drop=True, inplace=True)

    return merged_df


######
# combo_df = merge_pos_and_neg_sep_dfs(last_response_df, df2)
# print(f'combo:\n{combo_df}')

# todo:do I need this script still?  delete it?
def split_df_into_pos_sep_df_and_one_probe_df(pos_sep_and_one_probe_df,
                                              isi_name_list=None,
                                              verbose=True):
    """
    For plots where positive separations are shown as line plots and 
    one probe results are shown as scatter plot, this function splits the dataframe into two.
    
    :param pos_sep_and_one_probe_df: Dataframe of positive separations with
        one_probe conds at bottom of df (e.g., shown as 20 or 99).  df must be indexed with the separation column.
    :param isi_name_list: List of isi names.  If None, will use default values.
    :param verbose: whether to print progress info to screen

    :return: Pos_sel_df: same as input df but with last row removed
            One_probe_df: constructed from last row of input df 
    """

    # todo: my exp doesn't have one probe condition so I can get rid of one_probe stuff.

    data_df = pos_sep_and_one_probe_df

    if verbose:
        print("\n*** running split_df_into_pos_sep_df_and_one_probe_df() ***")

    if isi_name_list is None:
        # todo: change isi name list - lose concurrent.
        isi_name_list = ['Concurrent', 'ISI0', 'ISI2', 'ISI4',
                         'ISI6', 'ISI9', 'ISI12', 'ISI24']

    # check that the df only contains positive separation values
    if 'sep' in list(data_df.columns):
        # todo: column in data is called 'separation' not 'sep'
        data_df = data_df.rename(columns={'sep': 'Separation'})

    # check if index column is set as 'Separation'
    if data_df.index.name is None:
        data_df = data_df.set_index('Separation')

    # data_df = data_df.loc[data_df['Separation'] >= 0]
    data_df = data_df.loc[data_df.index >= 0]

    if verbose:
        print(f'data_df:\n{data_df}')

    # Chop last row off to use for one_probe condition
    pos_sep_df, one_probe_df = data_df.drop(data_df.tail(1).index), data_df.tail(1)
    if verbose:
        print(f'pos_sep_df:\n{pos_sep_df}')
    # change separation value from -1 so its on the left of the plot
    one_probe_lum_list = one_probe_df.values.tolist()[0]
    one_probe_dict = {'ISIs': isi_name_list,
                      'probeLum': one_probe_lum_list,
                      'x_vals': [-1 for i in isi_name_list]}
    one_probe_df = pd.DataFrame.from_dict(one_probe_dict)
    if verbose:
        print(f'one_probe_df:\n{one_probe_df}')

    return pos_sep_df, one_probe_df


# # choose colour palette
def fig_colours(n_conditions):
    """
    Use this to always get the same colours in the same order with no fuss.
    :param n_conditions: number of different colours
    :return: a colour pallet
    """
    use_colours = 'tab10'

    if 10 < n_conditions < 21:
        use_colours = 'tab20'
    elif n_conditions > 20:
        raise ValueError(f"\tERROR - more classes ({n_conditions}) than colours!?!?!?")
        print("\tERROR - more classes than colours!?!?!?")
    sns.set_palette(palette=use_colours, n_colors=n_conditions)
    my_colours = sns.color_palette()

    return my_colours


# # # all ISIs on one axis - pos sep only, plus single probe
# FIGURE 1 - shows one axis (x=separation (0-18), y=probeLum) with all ISIs added.
# it also seems that for isi=99 there are simple dots added at -1 on the x axis.

def plot_pos_sep_and_one_probe(pos_sep_and_one_probe_df,
                               thr_col='probeLum',
                               fig_title=None,
                               one_probe=False,
                               save_path=None, 
                               save_name=None,
                               isi_name_list=None,
                               pos_set_ticks=None,
                               pos_tick_labels=None,
                               verbose=True):
    """
    This plots a figure with one axis, x has separation values [-2, -1, 0, 1, 2, 3, 6, 18],
    where -2 is not uses, -1 is for the single probe condition - shows as a scatter plot.
    Values sep (0:18) are shown as lineplots.
    Will plot all ISIs on the same axis.

    :param pos_sep_and_one_probe_df: Full dataframe to use for values
    :param thr_col: column in df to use for y_vals
    :param fig_title: default=None.  Pass a string to add as a title.
    :param one_probe: default=True.  Add data for one_probe as scatter.
    :param save_path: default=None.  Path to dir to save fig
    :param save_name: default=None.  name to save fig
    :param isi_name_list: default=NONE: will use defaults, or pass list of names for legend.
    :param pos_set_ticks: default=NONE: will use defaults, or pass list of names for x-axis positions.
    :param pos_tick_labels: default=NONE: will use defaults, or pass list of names for x_axis labels.
    :param verbose: default: True. Won't print anything to screen if set to false.

    :return: plot
    """

    # todo: I don';t have a one_probe condition in my exp, so get rid of this

    if verbose:
        print("\n*** running plot_pos_sep_and_one_probe() ***")
        # print(f'pos_sep_and_one_probe_df:\n{pos_sep_and_one_probe_df}')

    # todo: get rid of concurrent, tick labels can start at zero, not -2.
    if isi_name_list is None:
        isi_name_list = ['Concurrent', 'ISI0', 'ISI2', 'ISI4',
                         'ISI6', 'ISI9', 'ISI12', 'ISI24']
        if verbose:
            print(f'isi_name_list: {isi_name_list}')

    if pos_set_ticks is None:
        pos_set_ticks = [-2, -1, 0, 1, 2, 3, 6, 18]
    if pos_tick_labels is None:
        pos_tick_labels = ['', 'one\nprobe', 0, 1, 2, 3, 6, 18]


    # call function to split df into pos_sep_df and one_probe_df
    if one_probe:
        pos_sep_df, one_probe_df = split_df_into_pos_sep_df_and_one_probe_df(
            pos_sep_and_one_probe_df=pos_sep_and_one_probe_df, isi_name_list=isi_name_list)
        if verbose:
            print(f'pos_sep_df:\n{pos_sep_df}\none_probe_df:\n{one_probe_df}')
    else:
        pos_sep_df = pos_sep_and_one_probe_df

    # make fig1
    fig, ax = plt.subplots(figsize=(10, 6))

    # line plot for main ISIs
    sns.lineplot(data=pos_sep_df, markers=True, dashes=False, ax=ax)

    # scatter plot for single probe conditions
    if one_probe:
        sns.scatterplot(data=one_probe_df, x="x_vals", y=thr_col,
                        hue="ISIs", style='ISIs', ax=ax)

    # decorate plot
    ax.legend(labels=isi_name_list, title='ISI',
              shadow=True,
              # place lower left corner of legend at specified location.
              loc='lower left', bbox_to_anchor=(0.96, 0.5))

    if one_probe:
        ax.set_xticks(pos_set_ticks)
        ax.set_xticklabels(pos_tick_labels)
    else:
        ax.set_xticks(pos_set_ticks[2:])
        ax.set_xticklabels(pos_tick_labels[2:])

    # ax.set_ylim([40, 90])
    ax.set_xlabel('Probe separation in diagonal pixels')
    ax.set_ylabel('Probe Luminance')

    if fig_title is not None:
        plt.title(fig_title)
        
    if save_path is not None:
        if save_name is not None:
            plt.savefig(f'{save_path}{os.sep}{save_name}')

    return fig


# # # all ISIs on one axis - pos sep only, NO single probe

# # # 8 batman plots

# # FIGURE 2
# this is a figure with one axis per isi, showing neg and pos sep
# (e.g., -18:18)

def eight_batman_plots(mean_df, thr1_df, thr2_df,
                       fig_title=None, isi_name_list=None,
                       x_tick_vals=None, x_tick_labels=None,
                       sym_sep_diff_list=None,
                       save_path=None, save_name=None,
                       verbose=True
                       ):
    """
    From array make separate batman plots for
    :param mean_df: df of values for mean thr
    :param thr1_df: df of values from stair 1 (e.g., probe_jump inwards)
    :param thr2_df: df of values for stair 2 (e.g., probe_jump outwards)
    :param fig_title: title for figure or None
    :param isi_name_list: If None, will use default setting, or pass list of
        names for legend
    :param x_tick_vals: If None, will use default setting, or pass list of
        values for x_axis
    :param x_tick_labels: If None, will use default setting, or pass list of
        labels for x_axis
    :param sym_sep_diff_list: list of differences between thr1&thr2 to be added
        as text to figure
    :param save_path: default=None.  Path to dir to save fig
    :param save_name: default=None.  name to save fig
    :param verbose: If True, print info to screen.

    :return: Batman Plot
    """

    if verbose:
        print("\n*** running eight_batman_plots() ***")

    if isi_name_list is None:
        isi_name_list = ['Concurrent', 'ISI0', 'ISI2', 'ISI4',
                         'ISI6', 'ISI9', 'ISI12', 'ISI24']

    if x_tick_vals is None:
        x_tick_vals = [-18, -6, -3, -2, -1, 0, 1, 2, 3, 6, 18, 20]

    if x_tick_labels is None:
        x_tick_labels = [-18, -6, -3, -2, -1, 0, 1, 2, 3, 6, 18, '1\nprobe']

    # check column name for x_values
    if 'sep' in list(mean_df.columns):
        mean_df = mean_df.rename(columns={'sep': 'Separation'})
        thr1_df = thr1_df.rename(columns={'sep': 'Separation'})
        thr2_df = thr2_df.rename(columns={'sep': 'Separation'})


    # set colours
    my_colours = fig_colours(len(isi_name_list))

    # make plots
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12, 6))

    if fig_title is not None:
        fig.suptitle(fig_title)

    ax_counter = 0
    # loop through the eight axes
    for row_idx, row in enumerate(axes):
        for col_idx, ax in enumerate(row):

            # mean threshold from CW and CCW probe jump direction
            sns.lineplot(ax=axes[row_idx, col_idx], data=mean_df,
                         x='Separation', y=isi_name_list[ax_counter],
                         color=my_colours[ax_counter],
                         linewidth=3, markers=True)

            # stair1: CW probe jumps only
            sns.lineplot(ax=axes[row_idx, col_idx], data=thr1_df,
                         x='Separation', y=isi_name_list[ax_counter],
                         color=my_colours[ax_counter],
                         linewidth=.5, marker="v")

            # stair2: CCW probe jumps only
            sns.lineplot(ax=axes[row_idx, col_idx], data=thr2_df,
                         x='Separation', y=isi_name_list[ax_counter],
                         color=my_colours[ax_counter],
                         linewidth=.5, marker="o")

            ax.set_title(isi_name_list[ax_counter])
            ax.set_xticks(x_tick_vals)
            ax.set_xticklabels(x_tick_labels)
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

            if sym_sep_diff_list is not None:
                ax.text(x=0.4, y=0.8, s=round(sym_sep_diff_list[ax_counter], 2),
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
    
    if save_path is not None:
        if save_name is not None:
            plt.savefig(f'{save_path}{os.sep}{save_name}')

    return fig

# # # split array into pos and neg set (sym) and get thr1, thr2 and mean thr


def a_data_extraction(p_name, run_dir, isi_list, save_all_data=True, verbose=True):

    """
    This script is a python version of Martin's first MATLAB analysis scripts, described below.

    a_data_extraction.m: Once a participant has completed a run of all ISIs,
        this script gets all their data into one file, and sorts each isi by stair.

    :param p_name: participant's name as used to save csv files.  e.g., if the
            file is .../nick1.csv, participant name is 'nick1'.
    :param run_dir: directory where isi folders are stored.
    :param isi_list: List of isi values, may differ between experiments.
    :param save_all_data: If True, will save all_data_df as an xlsx.
    :param verbose: If True, will print progress to screen.

    :return: ALL_ISIs_sorted.xlsx: A pandas DataFrame with n xlsx file of all
        data for one run of all ISIs.
    """

    print("\n***running a_data_extraction()***\n")

    # get run name/number
    path, run = os.path.split(run_dir)
    if verbose:
        print(f"run: {run}")

    if isi_list is None:
        isi_list = [0, 1, 4, 6, 12, 24]

    # empty array to append info into
    all_data = []

    # loop through ISIs in each run.
    for isi in isi_list:
        filepath = f'{run_dir}{os.path.sep}ISI_{isi}_probeDur2{os.path.sep}' \
                   f'{p_name}.csv'
        if verbose:
            print(f"filepath: {filepath}")

        # load data
        this_isi_df = pd.read_csv(filepath)
        if verbose:
            print(f"loaded csv:\n{this_isi_df.head()}")

        # sort by staircase
        trial_numbers = list(this_isi_df['trial_number'])
        this_isi_df = this_isi_df.sort_values(by=['stair', 'trial_number'])

        # add isi column for multi-indexing
        this_isi_df.insert(0, 'ISI', isi)
        this_isi_df.insert(1, 'srtd_trial_idx', trial_numbers)
        if verbose:
            print(f'df sorted by stair:\n{this_isi_df.head()}')

        # get column names to use on all_data_df
        column_names = list(this_isi_df)

        # add to all_data
        all_data.append(this_isi_df)

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
        save_name = 'ALL_ISIs_sorted.xlsx'

        save_excel_path = os.path.join(run_dir, save_name)
        if verbose:
            print(f"\nsaving all_data_df to save_excel_path:\n{save_excel_path}")
        all_data_df.to_excel(save_excel_path, index=False)

    print("\n***finished a_data_extraction()***\n")


    return all_data_df

# # # # # # #
# participant_name = 'Kim1'
# isi_list = [-1, 0, 2, 4, 6, 9, 12, 24]
# # isi_list = [0, 2]  # , 4, 6, 9, 12, 24, -1]
# run_dir = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/Kim/Nick_practice/P6a-Kim'
#
# a_data_extraction(p_name=participant_name, run_dir=run_dir, isi_list=isi_list, verbose=True)




def b1_extract_last_values(all_data_path, thr_col='probeLum', resp_col='trial_response',
                           last_vals_list=None, verbose=True):

    """
    This script is a python version of Martin's second MATLAB analysis scripts, described below.

    b1_extract_last_values.m: For each isi there are 14 staircase values,
    this script gets the last threshold and last response (correct/incorrect) for each.

    You can also get the mean of the last n thresholds
    (e.g., 4, 7 etc as thresholds-sorted-4last, thresholds-sorted-7last.

    :param all_data_path: path to the all_data.xlsx file
    :param thr_col: (default probeLum) name of the column showing the threshold varied by the staircase.
    :param resp_col: (default: 'trial_response') name of the column showing accuracy per trial.
    :param last_vals_list: get the mean threshold of the last n values.
        It will use [1] (e.g., last threshold only), unless another list is passed.
    :param verbose: If True, will print progress to screen.

    :return: nothing, but saves the files as:
            'last_response.csv' showing response to last trial and 'last_threshold'.csv.
            If n is greater than 1, files will be
            f'mean_of_last_{last_n_values}_thr.csv' showing mean of last n thresholds
    """

    print("\n***running b1_extract_last_values()***")

    # extract path to save files to
    save_path, xlsx_name = os.path.split(all_data_path)

    # todo: do I actually need the mean of the last 4 or 7 values, or should I just get last 1?

    if last_vals_list is None:
        last_vals_list = [1]
    elif type(last_vals_list) == int:
        last_vals_list = [last_vals_list]
    elif type(last_vals_list) == list:
        if not all(type(x) is int for x in last_vals_list):
            raise TypeError(f'last_vals list should be list of ints, not {last_vals_list}.')
    else:
        raise TypeError(f'last_vals list should be list of ints, not {last_vals_list} {type(last_vals_list)}.')

    # open all_data file.  use engine='openpyxl' for xlsx files.
    # For other experiments it might be easier not to do use cols as they might be different.
    # todo: do I need response.rt?  No but might be good to have on all data for summary stats.
    all_data_df = pd.read_excel(all_data_path, engine='openpyxl',
                                usecols=['ISI', 'stair', 'trial_number',
                                         'probeLum', 'trial_response', 'resp.rt'])

    # get list of isi and stair values to loop through
    isi_list = all_data_df['ISI'].unique()
    stair_list = all_data_df['stair'].unique()

    # check last_vals_list are shorted than trials per stair.
    trials, columns = np.shape(all_data_df)
    trials_per_stair = int(trials/len(isi_list)/len(stair_list))
    if max(last_vals_list) > trials_per_stair:
        raise ValueError(f'max(last_vals_list) ({max(last_vals_list)}) must be '
                         f'lower than trials_per_stair ({trials_per_stair}).')

    # get isi string for column names
    isi_name_list = [f'isi{i}' for i in isi_list]

    if verbose:
        print(f"last_vals_list: {last_vals_list}")
        print(f"{len(isi_list)} isi values and {len(stair_list)} stair values")
        print(f"all_data_df:\n{all_data_df}")


    # loop through last values (e.g., [1, 4, 7])
    for last_n_values in last_vals_list:
        if verbose:
            print(f"\nlast_n_values: {last_n_values}")

        # make empty arrays to add results into (rows=stairs, cols=ISIs)
        thr_array = np.zeros(shape=[len(stair_list), len(isi_list)])
        resp_array = np.zeros(shape=[len(stair_list), len(isi_list)])


        # loop through isi values
        for isi_idx, isi in enumerate(isi_list):
            if verbose:
                print(f"\n{isi_idx}: isi: {isi}")

            # get df for this isi only
            isi_df = all_data_df[all_data_df['ISI'] == isi]

            # loop through stairs for this isi
            for stair_idx, stair in enumerate(stair_list):

                # get df just for one stair at this isi
                stair_df = isi_df[isi_df['stair'] == stair]
                if verbose:
                    print(f'\nstair_df (stair={stair}, isi={isi}, last_n_values={last_n_values}):\n{stair_df}')

                # get the mean threshold of the last n values (last_n_values)
                mean_thr = np.mean(list(stair_df[thr_col])[-last_n_values:])
                if verbose:
                    if last_n_values > 1:
                        print(f'last {last_n_values} values: {list(stair_df[thr_col])[-last_n_values:]}')
                    print(f'mean_thr: {mean_thr}')

                # copy value into threshold array
                thr_array[stair_idx, isi_idx] = mean_thr


                if last_n_values == 1:
                    last_response = list(stair_df[resp_col])[-last_n_values]
                    if verbose:
                        print(f'last_response: {last_response}')

                    # copy value into response array
                    resp_array[stair_idx, isi_idx] = last_response


        # make dataframe from array
        thr_df = pd.DataFrame(thr_array, columns=isi_name_list)
        thr_df.insert(0, 'stair', stair_list)
        if verbose:
            print(f"thr_df:\n{thr_df}")

        # save response and threshold arrays
        # save name if last_n_values > 1
        thr_filename = f'mean_of_last_{last_n_values}_thr.csv'

        if last_n_values == 1:
            # make dataframe from array for last response (correct/incorrect)
            resp_df = pd.DataFrame(resp_array, columns=isi_name_list)
            resp_df.insert(0, 'stair', stair_list)
            if verbose:
                print(f"resp_df:\n{resp_df}")

            # save last response
            resp_filename = 'last_response.csv'
            resp_filepath = os.path.join(save_path, resp_filename)
            resp_df.to_csv(resp_filepath, index=False)

            # save name if last_n_values == 1
            thr_filename = 'last_threshold.csv'

        # save threshold arrays
        thr_filepath = os.path.join(save_path, thr_filename)
        thr_df.to_csv(thr_filepath, index=False)


    print("\n***finished b1_extract_last_values()***")


# # # # # # # # #
# test_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/Kim/' \
#             'Nick_practice/P6a-Kim/ALL_ISIs_sorted.xlsx'
# b1_extract_last_values(all_data_path=test_path)


# def b2_last_reversal(all_data_path,
#                      thr_col='probeLum', resp_col='trial_response',
#                      reversals_list=None,
#                      show_plots=True, verbose=True):
#     """
#     b2_last_reversal/m: Computes the threshold for each staircase as an average of
#     the last n_reversals (incorrect responses).  Plot these as mean per sep,
#     and also multiplot for each isi with different values for -18&+18 etc
#
#     However: although each sep values is usd twice (e.g., stairs 1 & 2 are both sep = 18),
#     these do not correspond to any meaningful difference (e.g., target_jump 1 or -1)
#     as target_jump is decided with random.choice, and this variable is not accessed
#     in this analysis.  For plots that show +18 and -18, these are duplicated values.
#     I intend to design the script to allow for meaningful differences here -
#     but I also need to copy this script first to check I have the same procedure
#     in case I have missed something.
#
#     :param all_data_path: path to the all_data xlsx file.
#     :param thr_col: (default probeLum) name of the column showing the threshold
#         (e.g., varied by the staircase).
#     :param resp_col: (default: 'trial_response') name of the column showing
#         (accuracy per trial).
#     :param reversals_list: Default=None.  If none will use values [2, 4].
#         Else input a list of values to calculate scores from.
#         e.g., if reversals_list=[2, 4], will get the mean threshold from last
#         n_reversals where n=2 or 4.  DO not use an odd number of reversals!
#     :param show_plots: whether to display plots on-screen.
#     :param verbose: If True, will print progress to screen.
#
#     :return: P-reversal{n}-thresholds arrays with details of last n_reversals.
#             Plot of mean for last reversal - all ISIs shows as different lines.
#             Batplots with pos and neg sep, separate grid for each isi
#     """
#     print("\n***running b2_last_reversal()***\n")
#     # todo: delete this script - I don't think i need it.
#     save_path, xlsx_name = os.path.split(all_data_path)
#
#     # open all_data file.  use engine='openpyxl' for xlsx files.
#     # For other experiments it might be easier not to do use cols as they might be different.
#     all_data_df = pd.read_excel(all_data_path, engine='openpyxl',
#                                 usecols=['ISI', 'stair', 'trial_number',
#                                          'probeLum', 'trial_response', 'resp.rt'])
#
#     # get list of isi and stair values to loop through
#     isi_list = all_data_df['ISI'].unique()
#     stair_list = all_data_df['stair'].unique()
#     if verbose:
#         print(f"isi_list: {isi_list}\nstair_list: {stair_list}")
#
#     # get isi string for column names
#     isi_name_list = ['Concurrent' if i == -1 else f'isi{i}' for i in isi_list]
#     if verbose:
#         print(f"isi_name_list: {isi_name_list}")
#         print(f"{len(isi_list)} isi values and {len(stair_list)} stair values")
#         print(f"all_data_df:\n{all_data_df}")
#
#     # get reversals list
#     if reversals_list is None:
#         reversals_list = [2, 4]
#     if verbose:
#         print(f'reversals_list: {reversals_list}')
#
#     # for figures
#     sym_sep_list = [-18, -6, -3, -2, -1, 0, 1, 2, 3, 6, 18, 20]
#     fig2_x_tick_lab = [-18, -6, -3, -2, -1, 0, 1, 2, 3, 6, 18, '1\nprobe']
#
#     # get results for n_reversals
#     for n_reversals in reversals_list:
#
#         # make empty arrays to add results into (rows=stairs, cols=ISIs)
#         mean_rev_lum = np.zeros(shape=[len(stair_list), len(isi_list)])
#
#         # loop through isi values
#         for isi_idx, isi in enumerate(isi_list):
#
#             # get df for this isi only
#             isi_df = all_data_df[all_data_df['ISI'] == isi]
#
#             # loop through stairs for this isi
#             for stair_idx, stair in enumerate(stair_list):
#
#                 # get df just for one stair at this isi
#                 stair_df = isi_df[isi_df['stair'] == stair]
#                 if verbose:
#                     print(f'\nstair_df (stair={stair}, isi={isi}, n_reversals={n_reversals}):\n{stair_df}')
#
#                 # get indices of last n incorrect responses
#                 incorrect_list = stair_df.index[stair_df[resp_col] == 0]
#
#                 # if there are an odd number of reversals, remove the first one
#                 if len(incorrect_list) % 2 == 1:
#                     incorrect_list = incorrect_list[1:]
#
#                     # get probeLum for corresponding trials
#                 reversal_probe_lum_list = stair_df[thr_col].loc[incorrect_list]
#
#                 # just select last n_reversals - or whole list if list is shorter than n
#                 # whole list will always be an even number
#                 reversal_probe_lum_list = reversal_probe_lum_list[-n_reversals:]
#
#                 # get mean of these probeLums
#                 mean_lum = np.mean(list(reversal_probe_lum_list))
#
#                 # append to mean_rev_lum array
#                 mean_rev_lum[stair_idx, isi_idx] = mean_lum
#
#         if verbose:
#             print(f"mean_rev_lum:\n{mean_rev_lum}")
#
#         # MAKE SYMMETRICAL
#         # this has got some dodgy stuff going on here - it truely is symmetrical as
#         # data is copied (e.g, -18 = +18), but presumably there should be different
#         # data for positive and negative separations.
#
#         # MATLAB version uses reversalThresh1sym (&2) then takes the mean of these with reversal_threshMean
#         # reversalThresh1sym is the variable used in the MATLAB scripts - works with 14 stairs
#
#         # dataframe of the mean probeLum for last n incorrect trials
#         mean_rev_lum_df = pd.DataFrame(mean_rev_lum, columns=isi_name_list)
#         print(f"mean_rev_lum_df:\n{mean_rev_lum_df}")
#
#         # make df with just data from positive separation trials with symmetrical structure.
#         # pos_sym_indices corresponds to sep: [18, 6, 3, 2, 1, 0, 1, 2, 3, 6, 18, 99]
#         # todo: use split_df_alternate_rows function
#         pos_sym_indices = [0, 2, 4, 6, 8, 10, 8, 6, 4, 2, 0, 12]
#         rev_thr1_sym_df = mean_rev_lum_df.iloc[pos_sym_indices]
#         rev_thr1_sym_df.reset_index(drop=True, inplace=True)
#
#         # neg_sym_indices corresponds to sep:   [-18, -6, -3, -2, -1, 0, -1, -2, -3, -6, -18, 99]
#         neg_sym_indices = [1, 3, 5, 7, 9, 11, 9, 7, 5, 3, 1, 13]
#         rev_thr2_sym_df = mean_rev_lum_df.iloc[neg_sym_indices]
#         rev_thr2_sym_df.reset_index(drop=True, inplace=True)
#
#         rev_thr_mean_df = pd.concat([rev_thr1_sym_df, rev_thr2_sym_df]).groupby(level=0).mean()
#         rev_thr_mean_df.insert(loc=0, column='Separation', value=sym_sep_list)
#         rev_thr1_sym_df.insert(loc=0, column='Separation', value=sym_sep_list)
#         rev_thr2_sym_df.insert(loc=0, column='Separation', value=sym_sep_list)
#         if verbose:
#             print(f"\nrev_thr_mean_df:\n{rev_thr_mean_df}")
#             print(f'\nrev_thr1_sym_df:\n{rev_thr1_sym_df}')
#             print(f'\nrev_thr2_sym_df:\n{rev_thr2_sym_df}')
#
#         # save rev_thr_mean_df
#         save_name = f'P-reversal{n_reversals}-thresholds.csv'
#         rev_thr_mean_df.to_csv(f'{save_path}{os.sep}{save_name}', index=False)
#
#         # New version should take stairs in this order I think (assuming pos first then neg)
#         # sep: -18, -6, -3, -2, -1, 0 & 0, 1, 2, 3, 6, 18, 99&99
#         # stair: 1,  3,  5,  7,  9, 10&11, 8, 6, 4, 2, 0, 12&13  if 0 indexed
#         # stair: 2,  4,  6,  8, 10, 11&12, 9, 7, 5, 3, 1, 13&14 if 1 indexed
#
#         # get mean difference between pairs of sep values for evaluating analysis,
#         # method with lowest mean difference is least noisy method. (for fig2)
#         # for each pair of sep values (e.g., stair1&2, stair3&4) subtract one from other.
#         # get abs of all values them sum the columns (ISIs)
#         diff_next = np.sum(abs(rev_thr1_sym_df - rev_thr2_sym_df), axis=0)
#         # take the mean of these across all ISIs to get single value
#         mean_diff_next = float(np.mean(diff_next))
#
#
#         # PLOT FIGURES
#         # todo: I don't need this - mean of reversals.
#         # # FIGURE 1 - shows one axis (x=separation (0-18), y=probeLum) with all ISIs added.
#         # # it also seems that for isi=99 there are simple dots added at -1 on the x axis.
#         fig1_title = f'Mean {thr_col} from last {n_reversals} reversals'
#         fig1_savename = f'data_last{n_reversals}_reversals.png'
#         plot_pos_sep_and_one_probe(pos_sep_and_one_probe_df=rev_thr_mean_df,
#                                    fig_title=fig1_title,
#                                    save_path=save_path,
#                                    save_name=fig1_savename)
#         # show and close plots
#         if show_plots:
#             plt.show()
#         plt.close()
#
#
#         # # FIGURE 2 - eight batman plots
#         # # this is a figure with one axis per isi, showing neg and pos sep
#         # # (e.g., -18:18)
#         fig_title = f'Last {n_reversals} reversals per isi. ' \
#                     f'(mean diff: {round(mean_diff_next, 2)})'
#         fig2_savename = f'runs_last{n_reversals}_reversals.png'
#         eight_batman_plots(mean_df=rev_thr_mean_df,
#                            thr1_df=rev_thr1_sym_df,
#                            thr2_df=rev_thr2_sym_df,
#                            fig_title=fig_title,
#                            isi_name_list=isi_name_list,
#                            x_tick_vals=sym_sep_list,
#                            x_tick_labels=fig2_x_tick_lab,
#                            sym_sep_diff_list=diff_next,
#                            save_path=save_path,
#                            save_name=fig2_savename,
#                            verbose=True)
#         # show and close plots
#         if show_plots:
#             plt.show()
#         plt.close()
#
#         print("\n***finished b2_last_reversal()***\n")
#
#
# ################
# # all_data_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/Kim/' \
# #                 'Nick_practice/P6a-Kim/ALL_ISIs_sorted.xlsx'
# # b2_last_reversal(all_data_path=all_data_path,
# #                 reversals_list=[2, 4],
# #                 # reversals_list=[2],
# #                 thr_col='probeLum', resp_col='trial_response',
# #                 show_plots=True,
# #                 verbose=True)


def b3_plot_staircase(all_data_path, thr_col='probeLum', resp_col='trial_response',
                      show_plots=True, save_plots=True, verbose=True):

    """
    b3_plot_staircase: staircases-ISIxxx.png: xxx corresponds to isi conditions.
    One plot for each isi condition.  Each figure has six panels (6 probes separation
    conditions) showing the Luminance value of two staircases as function of
    trial number. Last panel shows last thr per sep condition.

    :param all_data_path: path to the all_data xlsx file.
    :param thr_col: (default probeLum) name of the column showing the threshold
        (e.g., varied by the staircase).
    :param resp_col: (default: 'trial_response') name of the column showing
        (accuracy per trial).
    :param show_plots: whether to display plots on-screen.
    :param save_plots: whether to save the plots.
    :param verbose: If True, will print progress to screen.

    :return:
    one figure per isi value - saved as Staircases_{isi_name}
    n_reversals.csv - number of reversals per stair - used in c_plots
    """

    print("\n*** running b3_plot_staircase() ***\n")

    save_path, xlsx_name = os.path.split(all_data_path)

    # open all_data file.  use engine='openpyxl' for xlsx files.
    # For other experiments it might be easier not to do use cols as they might be different.
    all_data_df = pd.read_excel(all_data_path, engine='openpyxl',
                                usecols=['ISI', 'stair',
                                         # 'trial_number',
                                         'stair_name', 'step', 'separation', 'flow_dir',
                                         'probe_jump', 'trgt_flow_same', 'corner',
                                         'probeLum', 'trial_response', 'resp.rt'])

    # todo: for motion_same contrast use sep*trgt_flow_same
    motion_same_cond = np.where(((all_data_df['separation'] == 0) & (all_data_df['trgt_flow_same'] == -1)), -.1, all_data_df['separation'] * all_data_df['trgt_flow_same'])
    print(f"\nmotion_same_cond:{motion_same_cond[-25:]}")
    all_data_df['motion_same_cond'] = motion_same_cond
    check_df = all_data_df[['separation', 'flow_dir', 'probe_jump', 'trgt_flow_same', 'motion_same_cond']]
    print(f"check_df:\n{check_df.tail(25)}")

    # all_data_df.insert(9, 'motion_same_cond', all_data_df.loc['separation'] * all_data_df.loc['trgt_flow_same'])
    # all_data_df['motion_same_cond'] = all_data_df.loc['separation'] * all_data_df.loc['trgt_flow_same']

    # todo: for flow_dir contrast use sep*flow_dir
    sep_values = [18, 6, 3, 2, 1, 0]
    flow_dir = [1, -1]
    sep_conds = []
    for sep in sep_values:
        for flow in flow_dir:
            sep_value = sep * flow
            if (sep == 0) & (flow == -1):
                sep_value = -.5
            sep_conds.append(sep_value)
    if verbose:
        print(f"\nsep_conds:{sep_conds}")



    # get list of isi and stair values to loop through
    stair_list = all_data_df['stair'].unique()
    isi_list = all_data_df['ISI'].unique()
    # get isi string for column names
    isi_name_list = [f'isi{i}' for i in isi_list]

    trials, columns = np.shape(all_data_df)
    trials_per_stair = int(trials/len(isi_list)/len(stair_list))

    separation_title = ['sep18', 'sep6', 'sep3', 'sep2', 'sep1', 'sep0']

    if verbose:
        print(f"all_data_df:\n{all_data_df}")
        print(f"{len(isi_list)} isi values and {len(stair_list)} stair values")
        print(f"isi_list: {isi_list}")
        print(f"isi_name_list: {isi_name_list}")
        print(f"stair_list: {stair_list}")
        print(f"trials_per_stair: {trials_per_stair}")
        print(f"separation_title: {separation_title}")

    '''the eighth plot is the last thr for each sep (+sep, -sep and mean).  
    get data from threshold_sorterd_1last.csv and reshape here'''
    # load last_threshold.csv for final plot (was created in script b1).
    last_thr_df = pd.read_csv(f'{save_path}{os.sep}last_threshold.csv')
    last_thr_df = last_thr_df.drop(['stair'], axis=1)
    last_thr_df.columns = isi_name_list
    last_thr_df.insert(0, 'sep_cond', sep_conds)
    if verbose:
        print(f'\nlast_thr_df:\n{last_thr_df}')

    # # Split last_thr_df info pos and neg sep (e.g., inward and outward flow)
    last_thr_pos_sep_df, last_thr_neg_sep_df = split_df_alternate_rows(last_thr_df)
    # last_thr_pos_sep_df.reset_index(drop=True, inplace=True)
    # last_thr_neg_sep_df.reset_index(drop=True, inplace=True)
    if verbose:
        print(f'\nlast_thr_pos_sep_df:\n{last_thr_pos_sep_df}')
        print(f'\nlast_thr_neg_sep_df:\n{last_thr_neg_sep_df}')

    # mean of pos and neg sep values [0, 1, 2, 3, 6, 18]
    last_thr_mean_df = pd.concat([last_thr_pos_sep_df, last_thr_neg_sep_df]).groupby(level=0).mean()

    # add sep column in
    sep_list = [0, 1, 2, 3, 6, 18]
    last_thr_mean_df = last_thr_mean_df.drop(['sep_cond'], axis=1)
    last_thr_mean_df.insert(0, 'sep_cond', sep_list)
    # last_thr_pos_sep_df.insert(0, 'sep', sep_list)
    # last_thr_neg_sep_df.insert(0, 'sep', sep_list)
    if verbose:
        print(f'\nlast_thr_mean_df:\n{last_thr_mean_df}')

    # the values listed as separation=20 are actually for the single probe cond.
    # Chop last row off and add values later.
    # todo: don't need one probe stuff, so get rid of this.
    # last_thr_mean_df, mean_one_probe_df = \
    #     last_thr_mean_df.drop(last_thr_mean_df.tail(1).index), last_thr_mean_df.tail(1)
    # last_thr_pos_sep_df, thr1_one_probe_df = \
    #     last_thr_pos_sep_df.drop(last_thr_pos_sep_df.tail(1).index), last_thr_pos_sep_df.tail(1)
    # last_thr_neg_sep_df, thr2_one_probe_df = \
    #     last_thr_neg_sep_df.drop(last_thr_neg_sep_df.tail(1).index), last_thr_neg_sep_df.tail(1)
    # if verbose:
    #     print(f'last_thr_mean_df:\n{last_thr_mean_df}')

    # pu the one_probe values into a df to use later
    # todo: don't need these one_probe lines
    # one_probe_df = pd.concat([thr1_one_probe_df, thr2_one_probe_df, mean_one_probe_df],
    #                          ignore_index=True)
    # one_probe_df.insert(0, 'dset', ['thr1', 'thr2', 'mean'])
    # one_probe_df.insert(0, 'x_val', [-1, -1, -1])
    # if verbose:
    #     print(f'one_probe_df:\n{one_probe_df}')

    # make empty arrays to save reversal n_reversals
    n_reversals_np = np.zeros(shape=[len(stair_list), len(isi_list)])

    # loop through isi values
    for isi_idx, isi in enumerate(isi_list):

        # get df for this isi only
        isi_df = all_data_df[all_data_df['ISI'] == isi]
        isi_name = isi_name_list[isi_idx]

        # initialise 8 plot figure
        # # this is a figure showing n_reversals per staircase condition.
        # todo: get rid of last thr plot and just have 2x3?
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12, 6))
        ax_counter = 0

        for row_idx, row in enumerate(axes):
            for col_idx, ax in enumerate(row):
                print(f'\nrow: {row_idx}, col: {col_idx}: {ax}')

                # for the first seven plots...
                # todo: might need to change this to < 6:
                if ax_counter < 7:

                    # # get pairs of stairs (e.g., [[18, -18], [6, -6], ...etc)
                    stair_odd = (ax_counter*2)+1  # 1, 3, 5, 7, 9, 11
                    stair_odd_df = isi_df[isi_df['stair'] == stair_odd]
                    stair_odd_df.insert(0, 'step', list(range(trials_per_stair)))
                    final_lum_odd = stair_odd_df.loc[stair_odd_df['step'] == trials_per_stair-1, 'probeLum'].item()
                    n_reversals_odd = trials_per_stair - stair_odd_df[resp_col].sum()

                    stair_even = (ax_counter+1)*2  # 2, 4, 6, 8, 10, 12
                    stair_even_df = isi_df[isi_df['stair'] == stair_even]
                    stair_even_df.insert(0, 'step', list(range(trials_per_stair)))
                    final_lum_even = stair_even_df.loc[stair_even_df['step'] == trials_per_stair-1, 'probeLum'].item()
                    n_reversals_even = trials_per_stair - stair_even_df[resp_col].sum()

                    # append n_reversals to n_reversals_np to save later.
                    n_reversals_np[stair_odd-1, isi_idx] = n_reversals_odd
                    n_reversals_np[stair_even-1, isi_idx] = n_reversals_even

                    if verbose:
                        print(f'\nstair_odd_df (stair={stair_odd}, isi_name={isi_name}:\n{stair_odd_df.head()}')
                        print(f"final_lum_odd: {final_lum_odd}")
                        print(f"n_reversals_odd: {n_reversals_odd}")
                        print(f'\nstair_even_df (stair={stair_even}, isi_name={isi_name}:\n{stair_even_df.tail()}')
                        print(f"final_lum_even: {final_lum_even}")
                        print(f"n_reversals_even: {n_reversals_even}")

                    '''
                    use multiplot method from figure 2 above.
                    There is also a horizontal line from the last value (step25)
                    There is text showing the number of reversals (incorrect responses)
                    y-axis can be 0:106 (maxLum), x is 1:25.
    
                    later the 8th panel is added - not sure what this is yet...
                    '''

                    fig.suptitle(f'Staircases and reversals for isi {isi_name}')

                    # plot thr per step for odd numbered stair
                    sns.lineplot(ax=axes[row_idx, col_idx], data=stair_odd_df,
                                 x='step', y=thr_col,
                                 color='tab:red',
                                 marker="v", markersize=5)
                    # line for final probeLum
                    ax.axhline(y=final_lum_odd, linestyle="--", color='tab:red')
                    # text for n_reversals
                    ax.text(x=0.25, y=0.9, s=f'{n_reversals_odd} reversals',
                            color='tab:red',
                            # needs transform to appear with rest of plot.
                            transform=ax.transAxes, fontsize=12)

                    # plot thr per step for even numbered stair
                    sns.lineplot(ax=axes[row_idx, col_idx], data=stair_even_df,
                                 x='step', y=thr_col,
                                 color='tab:blue',
                                 # linewidth=3,
                                 marker="o", markersize=4)
                    ax.axhline(y=final_lum_even, linestyle="-.", color='tab:blue')
                    ax.text(x=0.25, y=0.8, s=f'{n_reversals_even} reversals',
                            color='tab:blue',
                            # needs transform to appear with rest of plot.
                            transform=ax.transAxes, fontsize=12)

                    ax.set_title(f'{isi_name} {separation_title[ax_counter]}')
                    ax.set_xticks(np.arange(0, trials_per_stair, 5))
                    ax.set_ylim([0, 110])

                    # artist for legend
                    if ax_counter == 0:
                        st1 = mlines.Line2D([], [], color='tab:red',
                                            marker='v',
                                            markersize=5, label='Stair1')
                        st1_last_val = mlines.Line2D([], [], color='tab:red',
                                                     linestyle="--", marker=None,
                                                     label='st1_last_val')
                        st2 = mlines.Line2D([], [], color='tab:blue',
                                            marker='o',
                                            markersize=5, label='Stair2')
                        st2_last_val = mlines.Line2D([], [], color='tab:blue',
                                                     linestyle="-.", marker=None,
                                                     label='st2_last_val')
                        ax.legend(handles=[st1, st1_last_val, st2, st2_last_val],
                                  fontsize=5)

                else:
                    """use the last values from each stair pair (e.g., 18, -18) to
                    get the mean threshold for each sep condition.
                    """
                    if verbose:
                        print("Eighth plot")
                        print(f'last_thr_mean_df:\n{last_thr_mean_df}')
                        # print(f'one_probe_df:\n{one_probe_df}')

                    isi_thr_mean_df = pd.concat([last_thr_mean_df["sep"], last_thr_mean_df[isi_name]],
                                                axis=1, keys=['sep', isi_name])
                    if verbose:
                        print(f'isi_thr_mean_df:\n{isi_thr_mean_df}')

                    # line plot for thr1, th2 and mean thr
                    # todo: check colours - they're aren't right on some in Kim folder.
                    sns.lineplot(ax=axes[row_idx, col_idx], data=last_thr_pos_sep_df,
                                 x='sep', y=isi_name, color='tab:red',
                                 linewidth=.5)
                    sns.lineplot(ax=axes[row_idx, col_idx], data=last_thr_neg_sep_df,
                                 x='sep', y=isi_name, color='tab:blue',
                                 linewidth=.5)
                    sns.lineplot(ax=axes[row_idx, col_idx], data=isi_thr_mean_df,
                                 x='sep', y=isi_name, color='tab:green')

                    # # scatter plot for single probe conditions
                    # sns.scatterplot(data=one_probe_df, x="x_val", y=isi_name,
                    #                 hue='dset',
                    #                 palette=['tab:red', 'tab:blue', 'tab:green'])
                    ax.legend(fontsize=8, markerscale=.5)

                    # decorate plot
                    ax.set_title(f'{isi_name} mean last threshold')
                    ax.set_xticks([0, 1, 2, 3, 6, 18])
                    ax.set_xticklabels([0, 1, 2, 3, 6, 18])
                    ax.set_xlabel('Probe separation')
                    ax.set_ylim([0, 110])
                    ax.set_ylabel('Probe Luminance')

                ax_counter += 1

        plt.tight_layout()

        # show and close plots
        if save_plots:
            savename = f'staircases_{isi_name}.png'
            plt.savefig(f'{save_path}{os.sep}{savename}')

        if show_plots:
            plt.show()
        plt.close()

    # save n_reversals to csv for use in script_c figure 5
    n_reversals_df = pd.DataFrame(n_reversals_np, columns=isi_name_list)
    n_reversals_df.insert(0, 'stair', stair_list)
    n_reversals_df.set_index('stair', inplace=True)
    if verbose:
        print(f'n_reversals_df:\n{n_reversals_df}')
    n_reversals_df.to_csv(f'{save_path}{os.sep}n_reversals.csv')

    print("\n***finished b3_plot_staircases()***\n")

####################
# all_data_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/Kim/' \
#                     'Nick_practice/P6a-Kim/ALL_ISIs_sorted.xlsx'
# b3_plot_staircase(all_data_path, thr_col='probeLum', resp_col='trial_response',
#                   show_plots=True, save_plots=True, verbose=True)
# todo: I'm only doing plots 1 & 3.  Actaully I dpnt' need any of these.
#  I could just do plots 1 and 3 for last thr value (not mean of last n)
#  but to be honest, its not necessary.
def c_plots(save_path, thr_col='probeLum', last_vals_list=None,
            show_plots=True, verbose=True):

    """
    5. c_plots.m: uses thresholds-sorted-Nlast and output 1 spreadsheet (P-next-thresholds) and six plots.

    use: thresholds-sorted-Nlast.mat: with N = the number of last values of
                                     each staircase used to compute the
                                     threshold.

    figures:
            data-NlastValues.png: threshold luminance in function of probe
                                  separation.
                                  with N = the number of last values of
                                  each staircase used to compute the
                                  threshold.
            runs-lastNlastValues.png: threshold luminance in function of
                                      probe separation, one panel one isi
                                      condition.
                                      with N = the number of last values of
                                      each staircase used to compute the
                                      threshold.


    :param save_path: path to f'mean_of last_{last_n_values}_thr.csv', where plots will be save.
    :param thr_col: column for threshold (e.g., probeLum)
    :param last_vals_list: Default: None, which reverts to [1, 4, 7].  
            Compute results based on mean of last n trials (e.g., [1, 4, 7])
    :param show_plots: Default True
    :param verbose: Default True.

    :return:
    """
    print("\n*** running c_plots() ***\n")


    isi_list = [-1, 0, 2, 4, 6, 9, 12, 24]
    isi_name_list = ['Concurrent', 'ISI0', 'ISI2', 'ISI4', 'ISI6', 'ISI9', 'ISI12', 'ISI24']
    sym_sep_list = [-18, -6, -3, -2, -1, 0, 1, 2, 3, 6, 18, 20]
    sym_sep_tick_labels = [-18, -6, -3, -2, -1, 0, 1, 2, 3, 6, 18, '1\nprobe']
    pos_sep_list = [0, 1, 2, 3, 6, 18, 20]

    if last_vals_list is None:
        last_vals_list = [1, 4, 7]
    if verbose:
        print(f'last_vals_list: {last_vals_list}')

    for last_n_values in last_vals_list:
        if verbose:
            print(f'\nlast {last_n_values} values')

        # load df mean of last n probeLum values (14 stairs x 8 isi).
        last_n_df = pd.read_csv(f'{save_path}{os.sep}mean_of_last_{last_n_values}_thr.csv')
        if verbose:
            print(f'last_n_df:\n{last_n_df}')

        last_n_df = last_n_df.drop(['stair'], axis=1)
        last_n_df.columns = isi_name_list

        # lastN_pos_sym_np has values for 1-indexed stairs [1, 3, 5, 7, 9, 11, 9, 7, 5, 3, 1, 13]
        # these correspond to separation values:     [18, 6, 3, 2, 1, 0, 1, 2, 3, 6, 18, 99]
        # todo: use split_df_alternate_rows function
        pos_sym_indices = [0, 2, 4, 6, 8, 10, 8, 6, 4, 2, 0, 12]
        last_n_pos_sym_df = last_n_df.iloc[pos_sym_indices]
        last_n_pos_sym_df.reset_index(drop=True, inplace=True)

        # lastN_neg_sym_np has values for 1-indexed stairs [2, 4, 6, 8, 10, 12, 10, 8, 6, 4, 2, 14]
        # these correspond to sep values:   [-18, -6, -3, -2, -1, 0, -1, -2, -3, -6, -18, 99]
        neg_sym_indices = [1, 3, 5, 7, 9, 11, 9, 7, 5, 3, 1, 13]
        last_n_neg_sym_df = last_n_df.iloc[neg_sym_indices]
        last_n_neg_sym_df.reset_index(drop=True, inplace=True)

        # mean of pos and neg sep values [18, 6, 3, 2, 1, 0, 1, 2, 3, 6, 18, 99]
        last_n_sym_mean_df = pd.concat([last_n_pos_sym_df, last_n_neg_sym_df]).groupby(level=0).mean()

        # subtract the dfs from each other, then for each column get the sum of abs values
        diff_val = np.sum(abs(last_n_pos_sym_df - last_n_neg_sym_df), axis=0)
        # take the mean of these across all ISIs to get single value
        mean_diff_val = float(np.mean(diff_val))

        # add sep column into dfs
        last_n_sym_mean_df.insert(0, 'sep', sym_sep_list)
        last_n_pos_sym_df.insert(0, 'sep', sym_sep_list)
        last_n_neg_sym_df.insert(0, 'sep', sym_sep_list)

        if verbose:
            print(f'\nlast_n_pos_sym_df:\n{last_n_pos_sym_df}')
            print(f'\nlast_n_neg_sym_df:\n{last_n_neg_sym_df}')
            print(f'\nlast_n_sym_mean_df:\n{last_n_sym_mean_df}')
            print(f'\ndiff_val:\n{diff_val}')
            print(f'\nmean_diff_val: {mean_diff_val}')


        # # Figure1 - runs-{n}lastValues
        # this is a figure with one axis per isi, showing neg and pos sep
        # (e.g., -18:18) - eight batman plots
        fig_title = f'Last {last_n_values} values per isi. ' \
                    f'(mean diff: {round(mean_diff_val, 2)})'
        fig1_savename = f'runs-{last_n_values}lastValues.png'
        eight_batman_plots(mean_df=last_n_sym_mean_df,
                           thr1_df=last_n_pos_sym_df,
                           thr2_df=last_n_neg_sym_df,
                           fig_title=fig_title,
                           isi_name_list=isi_name_list,
                           x_tick_vals=sym_sep_list,
                           x_tick_labels=sym_sep_tick_labels,
                           sym_sep_diff_list=diff_val,
                           save_path=save_path,
                           save_name=fig1_savename,
                           verbose=True)
        if show_plots:
            plt.show()
        plt.close()

        # # FIGURE2 - doesn't exist in script - but I'll keep their numbers

        # # FIGURE3 - 'data-{n}lastValues.png' - all ISIs on same axis, pos sep only, plus single
        # # use plot_pos_sep_and_one_probe()
        fig3_save_name = f'data-{last_n_values}lastValues.png'
        plot_pos_sep_and_one_probe(pos_sep_and_one_probe_df=last_n_sym_mean_df,
                                   thr_col=thr_col,
                                   fig_title=fig3_save_name[:-4],
                                   one_probe=False,
                                   save_path=save_path,
                                   save_name=fig3_save_name,
                                   verbose=True)
        if show_plots:
            plt.show()
        plt.close()

        # todo: don't need this plot
        # # # # FIGURE4 - 'dataDivOneProbe-{n}lastValues.png' - all ISIs on same axis, pos sep only.
        # #         # does not include single probe
        # # # # use plot_pos_sep_and_one_probe(one_probe=False)
        # # # each sep row in pos_sep_df is divided by one_probe_df.
        # fig4_save_name = f'dataDivOneProbe-{last_n_values}lastValues.png'
        # pos_sep_df, one_probe_df = split_df_into_pos_sep_df_and_one_probe_df(last_n_sym_mean_df)
        # pos_sep_arr = pos_sep_df.to_numpy()
        # one_probe_arr = one_probe_df['probeLum'].to_numpy()
        # div_by_one_probe_arr = (pos_sep_arr.T / one_probe_arr[:, None]).T
        # div_by_one_probe_df = pd.DataFrame(div_by_one_probe_arr, columns=isi_name_list)
        # div_by_one_probe_df.insert(0, 'Separation', pos_sep_list[:-1])
        # div_by_one_probe_df.set_index('Separation', inplace=True)
        # print(f'div_by_one_probe_df:\n{div_by_one_probe_df}')
        #
        # plot_pos_sep_and_one_probe(div_by_one_probe_df,
        #                            thr_col='probeLum',
        #                            fig_title=fig4_save_name[:-4],
        #                            one_probe=False,
        #                            save_path=save_path,
        #                            save_name=fig4_save_name,
        #                            verbose=True)
        # if show_plots:
        #     plt.show()
        # plt.close()

        # these plots (5, 6, 7) only for single last value.
        # if last_n_values == 1:

            # todo: don't need figure 5
            # # # FIGURE5 - 'data-nextValues.png' - all ISIs on same axis, pos sep only, plus single.
            # # # use plot_pos_sep_and_one_probe()
            # # load last-response.csv and reversal-number.csv
            # last_response_df = pd.read_csv(f'{save_path}{os.sep}last_response.csv', dtype=int)
            # n_reversals_df = pd.read_csv(f'{save_path}{os.sep}n_reversals.csv', dtype=int)
            # if verbose:
            #     print(f'\nlast_response_df.head():\n{last_response_df.head()}')
            #     print(f'n_reversals_df.head():\n{n_reversals_df.head()}')
            #
            # # get thresh1_df (& thresh2_df) with same shape as the MATLAB script thresh1
            # # from last_n_pos_sym_df & last_n_neg_sym_df, take rows relating to [-18, -6, -3, -2, -1, 0, 20]
            # rows_to_keep = [0, 1, 2, 3, 4, 5, 11]
            # thresh1_df = last_n_pos_sym_df.iloc[rows_to_keep, :]
            # thresh2_df = last_n_neg_sym_df.iloc[rows_to_keep, :]
            # if verbose:
            #     print(f'\nthresh1_df:\n{thresh1_df}')
            #     print(f'thresh2_df:\n{thresh2_df}')
            #
            # # get response_dfs with same shape as MATLAB resp1 and resp2
            # last_response_df.columns = ['stair'] + isi_name_list
            # resp1_df, resp2_df = split_df_alternate_rows(last_response_df)
            # if verbose:
            #     print(f'\nresp1_df:\n{resp1_df}')
            #     print(f'resp2_df:\n{resp2_df}')
            #
            # # get reversal_num df with same shape as MATLAB resp1 and resp2
            # n_reversals_df.columns = ['stair'] + isi_name_list
            # reversal1_df, reversal2_df = split_df_alternate_rows(n_reversals_df)
            # if verbose:
            #     print(f'\nreversal1_df:\n{reversal1_df}')
            #     print(f'reversal2_df:\n{reversal2_df}')
            #
            # # empty arrays to store results in
            # next_thresh1 = np.zeros(shape=[len(pos_sep_list), len(isi_list)])
            # next_thresh2 = np.zeros(shape=[len(pos_sep_list), len(isi_list)])
            #
            # c = 106 * 0.6  # 106 maxLum - could be another value
            # target_thresh = 0.75
            #
            # # loop through to get predicted next thr value
            # """# formula = (c * (resp - target_thresh) / (1 - NumReversal) and then
            # # previous value - formula """
            #
            # # loop through isi values
            # for isi_idx, isi_name in enumerate(isi_name_list):
            #
            #     # loop through stairs for this isi
            #     for sep_cond in list(range(len(pos_sep_list))):
            #         if verbose:
            #             print(f"\nsep_cond: {sep_cond}, isi_idx: {isi_idx}, isi: {isi_name}")
            #
            #         # access values from relevant cell in dataframes
            #         thr1 = thresh1_df.loc[thresh1_df.index[[sep_cond]], isi_name].item()
            #         resp1 = resp1_df.loc[resp1_df.index[[sep_cond]], isi_name].item()
            #         rev1 = reversal1_df.loc[reversal1_df.index[[sep_cond]], isi_name].item()
            #
            #         thr2 = thresh2_df.loc[thresh2_df.index[[sep_cond]], isi_name].item()
            #         resp2 = resp2_df.loc[resp2_df.index[[sep_cond]], isi_name].item()
            #         rev2 = reversal2_df.loc[reversal2_df.index[[sep_cond]], isi_name].item()
            #
            #         # next_thresh1&2 are only used in next_thresh1_sym and next_thresh2_sym
            #         next_thresh1[sep_cond, isi_idx] = \
            #             thr1 - (c * (resp1 - target_thresh) / (rev1+1))
            #         next_thresh2[sep_cond, isi_idx] = \
            #             thr2 - (c * (resp2 - target_thresh) / (rev2+1))
            #
            #
            # # convert arrays to have symmetrical pattern
            # # next_thresh1_sym and next_thresh2_sym are used in nextThresh_mean
            # # next_thresh1_sym and next_thresh2_sym and nextThresh_mean are also used in fig6
            # next_thresh1_sym = np.array([next_thresh1[0, :], next_thresh1[1, :],
            #                              next_thresh1[2, :], next_thresh1[3, :],
            #                              next_thresh1[4, :], next_thresh1[5, :],
            #                              next_thresh1[4, :], next_thresh1[3, :],
            #                              next_thresh1[2, :], next_thresh1[1, :],
            #                              next_thresh1[0, :], next_thresh1[6, :]], dtype=object)
            # next_thresh2_sym = np.array([next_thresh2[0, :], next_thresh2[1, :],
            #                              next_thresh2[2, :], next_thresh2[3, :],
            #                              next_thresh2[4, :], next_thresh2[5, :],
            #                              next_thresh2[4, :], next_thresh2[3, :],
            #                              next_thresh2[2, :], next_thresh2[1, :],
            #                              next_thresh2[0, :], next_thresh2[6, :]], dtype=object)
            #
            # next_thresh_mean_arr = np.mean(np.array([next_thresh1_sym, next_thresh2_sym]), axis=0)
            #
            # # convert arrays in df
            # next_thresh_mean_df = pd.DataFrame(data=next_thresh_mean_arr, columns=isi_name_list)
            # next_thresh1_sym_df = pd.DataFrame(data=next_thresh1_sym, columns=isi_name_list)
            # next_thresh2_sym_df = pd.DataFrame(data=next_thresh2_sym, columns=isi_name_list)
            #
            # next_thresh_mean_df.insert(loc=0, column='Separation', value=sym_sep_list)
            # next_thresh1_sym_df.insert(loc=0, column='Separation', value=sym_sep_list)
            # next_thresh2_sym_df.insert(loc=0, column='Separation', value=sym_sep_list)
            # if verbose:
            #     print(f'next_thresh_mean_df:\n{next_thresh_mean_df}')
            #     print(f'next_thresh1_sym_df:\n{next_thresh1_sym_df}')
            #     print(f'next_thresh2_sym_df:\n{next_thresh2_sym_df}')
            #
            # # save nextThresh_mean, 1sym and 2sym to excel
            # with pd.ExcelWriter(f'{save_path}{os.sep}P-next-thresholds.xlsx',
            #                     # engine='openpyxl'
            #                     ) as writer:
            #     next_thresh_mean_df.to_excel(writer, sheet_name='next_thresh_mean_df', index=False)
            #     next_thresh1_sym_df.to_excel(writer, sheet_name='next_thresh1_sym_df', index=False)
            #     next_thresh2_sym_df.to_excel(writer, sheet_name='next_thresh2_sym_df', index=False)
            #
            # # # # plot figure 5
            # fig5_save_name = 'data-nextValues.png'
            # plot_pos_sep_and_one_probe(next_thresh_mean_df,
            #                            thr_col='probeLum',
            #                            fig_title=fig5_save_name[:-4],
            #                            one_probe=True,
            #                            save_path=save_path,
            #                            save_name=fig5_save_name,
            #                            verbose=True)
            # if show_plots:
            #     plt.show()
            # plt.close()
            # print('\nplot 5 finished')


            # # # FIGURE6 - 'runs-nextValues.png' - 8 batman plots
            # # todo: don't need figure 6
            # # get mean diff between next_thresh1_sym & next_thresh2_sym per isi
            # # subtract the dfs from each other, then for each column get the sum of abs values
            # diff_next = np.sum(abs(next_thresh1_sym - next_thresh2_sym), axis=0)
            # # get mean difference across all ISIs
            # mean_diff_next = float(np.mean(diff_next))
            # if verbose:
            #     print(f'\ndiff_next:\n{diff_next}\nmean_diff_next: {mean_diff_next}')
            #
            # fig6_savename = 'runs-nextValues.png'
            # fig6_title = f'runs-nextValues per isi (mean diff: {round(mean_diff_next, 2)})'
            # eight_batman_plots(mean_df=next_thresh_mean_df,
            #                    thr1_df=next_thresh1_sym_df,
            #                    thr2_df=next_thresh2_sym_df,
            #                    fig_title=fig6_title,
            #                    isi_name_list=isi_name_list,
            #                    x_tick_vals=sym_sep_list,
            #                    x_tick_labels=sym_sep_tick_labels,
            #                    sym_sep_diff_list=diff_next,
            #                    save_path=save_path,
            #                    save_name=fig6_savename,
            #                    verbose=True)
            # if show_plots:
            #     plt.show()
            # plt.close()
            # if verbose:
            #     print('\nplot 6 finished')


            # # # FIGURE7 - 'dataDivOneProbe-nextValues.png'- all ISIs on same axis, pos sep only.
            # # does not include single probe
            # # todo: don't need plot 7
            # # divide each mean sep value (0, 1, 2, 3, 6, 18) by single probe cond (99/20)
            # divided_df = next_thresh_mean_df.iloc[5:11] / next_thresh_mean_df.iloc[11]
            # divided_df['Separation'] = [0, 1, 2, 3, 6, 18]
            # divided_df = divided_df.set_index('Separation')
            # if verbose:
            #     print(f'divided:\n{divided_df}')
            #
            # # # use plot_pos_sep_and_one_probe(one_probe=False)
            # fig7_title = 'mean of both probes / single probe'
            # plot_pos_sep_and_one_probe(divided_df,
            #                            thr_col='probeLum',
            #                            fig_title=fig7_title,
            #                            one_probe=False,
            #                            save_path=save_path,
            #                            save_name='dataDivOneProbe-nextValues.png',
            #                            verbose=True)
            # if show_plots:
            #     plt.show()
            # plt.close()

    print("\n***finished c_plots()***\n")


# #########
# c_plots(save_path='/Users/nickmartin/Documents/PycharmProjects/Cardiff/Kim/Nick_practice/P6a-Kim',
#         thr_col='probeLum', last_vals_list=[1, 4, 7],
#         show_plots=True, verbose=True)


def d_averageParticipant(root_path, run_dir_names_list,
                         thr_col='probeLum', show_plots=True, verbose=True):
    """
    d_averageParticipant: take P-next-thresholds.xlsx and P-reversal4-thresholds.csv
    in each participant run folder and make master lists  
    MASTER_next_thresh
    MASTER_reversal_4_thresh
    
    Get mean of next threshold across 6 run conditions saved as 
    MASTER_ave_next_thresh.csv
    
    Save master lists to folder containing the six runs (root_path).
    
    Plots:
    mean_next_thresh saved as P_ave_thr_all_runs.png
    mean_next_thresh two-probe/one-probe saved as P_ave_thr_divided_one_probe
    
    :param root_path: dir containing run folders
    :param run_dir_names_list: names of run folders
    :param thr_col: Default: 'probeLum'. column with variable controlled by staircase
    :param show_plots: default True - dispplay plots
    :param verbose: Defaut true, print progress to screen
    """

    print("\n***running d_averageParticipant()***\n")

    # # part1. Munge data, save master lists and get means etc
    # #  - loop through runs and get each P-next-thresholds and P-reversal4-thresholds
    # # Make master sheets: MASTER_next_thresh & MASTER_reversal_4_thresh
    # # Incidentally the MATLAB script didn't specify which reversals data to use,
    # # although the figures imply Martin used last3 reversals.


    isi_name_list = ['Concurrent', 'ISI0', 'ISI2', 'ISI4',
                     'ISI6', 'ISI9', 'ISI12', 'ISI24']
    pos_sep_list = [0, 1, 2, 3, 6, 18, 20]
    all_data_next_list = []
    all_data_4_reversals_list = []

    for run_idx, run_name in enumerate(run_dir_names_list):
        this_next_df = pd.read_excel(f'{root_path}{os.sep}{run_name}{os.sep}P-next-thresholds.xlsx',
                                     sheet_name='next_thresh_mean_df', engine='openpyxl')

        if 'Unnamed: 0' in list(this_next_df):
            this_next_df.drop('Unnamed: 0', axis=1, inplace=True)
            print('foundunnamed')
        this_next_df.insert(0, 'Run', run_idx)
        all_data_next_list.append(this_next_df)
        # print(f'\nthis_next_df: ({run_idx}) {run_name}\n{this_next_df}')

        this_reversal_df = pd.read_csv(f'{root_path}{os.sep}{run_name}{os.sep}P-reversal4-thresholds.csv')
        if 'Unnamed: 0' in list(this_reversal_df):
            this_reversal_df.drop('Unnamed: 0', axis=1, inplace=True)
            print('foundunnamed')

        this_reversal_df.insert(0, 'Run', run_idx)
        all_data_4_reversals_list.append(this_reversal_df)

    all_data_next_df = pd.concat(all_data_next_list, ignore_index=True)
    all_data_next_df.to_csv(f'{root_path}{os.sep}MASTER_next_thresh.csv', index=False)

    all_data_4_reversals_df = pd.concat(all_data_4_reversals_list, ignore_index=True)
    all_data_4_reversals_df.to_csv(f'{root_path}{os.sep}MASTER_reversal_4_thresh.csv', index=False)

    if verbose:
        print(f'all_data_next_df:\n{all_data_next_df}')
        print(f'all_data_4_reversals_df:\n{all_data_4_reversals_df}')

    # get mean of all runs (each sep and ISI)
    # try just grouping the master sheet first, rather than using concat.
    ave_next_thr_df = all_data_next_df.drop('Run', axis=1)
    ave_next_thr_df = ave_next_thr_df.groupby('Separation').mean()
    if verbose:
        print(f'ave_next_thr_df:\n{ave_next_thr_df}')
    ave_next_thr_df.to_csv(f'{root_path}{os.sep}MASTER_ave_next_thresh.csv')
    # std_next_df = ave_next_thr_df.groupby('Separation').std()

    # ave_reversal4_thr_df = all_data_4_reversals_df.drop('Run', axis=1)
    # ave_reversal4_thr_df = ave_reversal4_thr_df.groupby('Separation').mean()
    # if verbose:
    #     print(f'ave_reversal4_thr_df:\n{ave_reversal4_thr_df}')
    # std_reversal_df = = ave_reversal4_thr_df.groupby('Separation').std()



    # part 2. main Figures (these are the ones saved in the matlab script)
    # Fig1: single ax, pos_sep_and_one_probe.  Uses ave_next_thr_df: for thr values (e.g., mean of all runs).
    # Fig2: single ax, pos_sep_and_one_probe (but no one probe).
    #     threshold values are mean of two probes / single probe

    # # Fig1
    fig1_title = f'Participant average threshold across all runs'
    fig1_savename = f'P_ave_thr_all_runs.png'
    plot_pos_sep_and_one_probe(pos_sep_and_one_probe_df=ave_next_thr_df,
                               fig_title=fig1_title,
                               one_probe=False,
                               save_path=root_path,
                               save_name=fig1_savename)
    if show_plots:
        plt.show()
    plt.close()

    # # Fig 2
    # # todo: don't need this plot
    # fig2_save_name = 'P_ave_thr_divided_one_probe'
    # fig2_title = 'Participant average threshold divided by single probe'
    # pos_sep_df, one_probe_df = split_df_into_pos_sep_df_and_one_probe_df(ave_next_thr_df)
    # pos_sep_arr = pos_sep_df.to_numpy()
    # one_probe_arr = one_probe_df['probeLum'].to_numpy()
    # div_by_one_probe_arr = (pos_sep_arr.T / one_probe_arr[:, None]).T
    # div_by_one_probe_df = pd.DataFrame(div_by_one_probe_arr, columns=isi_name_list)
    # div_by_one_probe_df.insert(0, 'Separation', pos_sep_list[:-1])
    # div_by_one_probe_df.set_index('Separation', inplace=True)
    # if verbose:
    #     print(f'div_by_one_probe_df:\n{div_by_one_probe_df}')
    #
    # plot_pos_sep_and_one_probe(div_by_one_probe_df,
    #                            thr_col=thr_col,
    #                            fig_title=fig2_title,
    #                            one_probe=False,
    #                            save_path=root_path,
    #                            save_name=fig2_save_name,
    #                            verbose=True)
    # if show_plots:
    #     plt.show()
    # plt.close()

    '''
    # this code is in the MATLAB script but not used anywhere
    # make data_stats df from all_data_next_df
    # same data, but reshaped to 6 rows x 56 columns.
    # 6 rows are the values for each of the six runs.
    # 56 columns correspond to combinations of 7 pos sep values (0, 1, 2, 3, 6, 18, 20)
    # and 8 ISI values ('Concurrent', 'ISI0', 'ISI2', 'ISI4', 'ISI6', 'ISI9', 'ISI12', 'ISI24')
    # 0_concurrent, 1_concurrent, 2_concurrent etc
    
    combo_name_list = []
    data_stats_list = np.zeros(shape=[6, 56])
    
    combo_counter = 0  # should be 56 combos
    for isi_name in isi_name_list:
        for sep in pos_sep_list:
            combo_name = f'{sep}_{isi_name}'
            combo_name_list.append(combo_name)
    
            for run_idx, run_name in enumerate(run_dir_names_list):
    
                # get threshold at isi_name for each Separation & Run value
                this_val = \
                    all_data_next_df.loc[(all_data_next_df["Separation"] == sep) &
                                         (all_data_next_df["Run"] == run_idx)][isi_name].values[0]
    
                # print(f'{combo_name}, ({run_idx}, {combo_counter}): {this_val}')
    
                data_stats_list[run_idx, combo_counter] = this_val
    
                # data_stats_list[run_idx, combo_counter] = \
                #     all_data_next_df.loc[(all_data_next_df["Separation"] == sep) &
                #                          (all_data_next_df["Run"] == run_idx)]["Concurrent"].values[0]
            combo_counter += 1
            # print('')
    
    data_stats_df = pd.DataFrame(data_stats_list, columns=combo_name_list)
    if verbose:
        print(f'data_stats_df:\n{data_stats_df}')
    '''

    '''
    extra figures - these are made but not saved in the matlab script.
    '''
    print("\n*** finished d_averageParticipant()***\n")




#######
# root_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/Kim'  # master folder containing all runs
# run_dir_names_list = ['P6a-Kim', 'P6b-Kim', 'P6c-Kim', 'P6d-Kim', 'P6e-Kim', 'P6f-Kim']
# d_averageParticipant(root_path=root_path, run_dir_names_list=run_dir_names_list)
