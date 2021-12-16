import os

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

"""
This page contains functions based on Martin's MATLAB analysis scripts.
However, these have been updated to use psignifit for thresholds (rather than 
using the last values, predicted next values, last reversals etc).
1. a_data_extraction: put data from one run, multiple ISIs into one array. 
2. get psignifit thresholds
3. b3_plot_staircase:
4. c_plots:
5. d_average_participant:

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


"""

# todo: some of these functions are duplicated in the OLD_psignifit_analysis.py script.
#  I don't need to two copies, so perhaps make a 'data_tools' script and put them there?

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

    data_df = pos_sep_and_one_probe_df

    if verbose:
        print("\n*** running split_df_into_pos_sep_df_and_one_probe_df() ***")

    if isi_name_list is None:
        isi_name_list = ['Concurrent', 'ISI0', 'ISI2', 'ISI4',
                         'ISI6', 'ISI9', 'ISI12', 'ISI24']

    # check that the df only contains positive separation values
    if 'sep' in list(data_df.columns):
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


def get_trimmed_mean_df(data_df, col_to_average_across='Run',
                        first_col_to_keep='Separation',
                        n_cut_from_each_tail=1,
                        verbose=True):
    """
    Function to calculate the trimmed mean of values in a dataframe.
    Trimmed mean involves removing the highest and lowest value before
    calculating the mean.

    The function assumes you want to find the average threshold for each
    ISI & separation combination across several runs.

    :param data_df: Dataframe containing thresholds for ISI & separation
        combinations across several runs, where runs is one of the columns.
    :param col_to_average_across: Default = 'Runs'.
    :param first_col_to_keep: Default = 'Separation'
    :param n_cut_from_each_tail: Default = 1.  Number to cut from each tail.
        e.g., if 1 is passed it will cut the 1 highest AND 1 lowest value.
    :param verbose: print progress to screen

    :return: trimmed_mean_df Dataframe showing the trimmed means.
    """

    # get number of datapoints to average across (used re-shaping array)
    n_runs = len(data_df[col_to_average_across].unique())

    # sort columns and re-shape data
    data_df = data_df.drop(col_to_average_across, axis=1)
    sep_col = data_df.pop(first_col_to_keep)
    headers = list(data_df.columns)
    n_rows, n_cols = data_df.shape
    data_array = data_df.to_numpy().reshape(n_runs, int(n_rows / n_runs), n_cols)

    # get trimmed mean
    prop_to_cut = n_cut_from_each_tail/n_runs
    trimmed_mean_np = stats.trim_mean(data_array, proportiontocut=prop_to_cut, axis=0)
    trimmed_mean_df = pd.DataFrame(trimmed_mean_np, columns=headers)
    trimmed_mean_df.insert(0, first_col_to_keep, sep_col)
    trimmed_mean_df = trimmed_mean_df.set_index([first_col_to_keep])
    if verbose:
        print(f'\ntrimmed_mean_df:\n{trimmed_mean_df}')

    # get sd
    trimmed_stdev_np = stats.mstats.trimmed_std(data_array,
                                                limits=(prop_to_cut*100, prop_to_cut*100), relative=True)


    return trimmed_mean_df



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
    sns.set_palette(palette=use_colours, n_colors=n_conditions)
    my_colours = sns.color_palette()

    return my_colours


# # # all ISIs on one axis - pos sep only, plus single probe
# FIGURE 1 - shows one axis (x=separation (0-18), y=probeLum) with all ISIs added.
# it also seems that for isi=99 there are simple dots added at -1 on the x axis.

def plot_pos_sep_and_one_probe(pos_sep_and_one_probe_df,
                               thr_col='probeLum',
                               fig_title=None,
                               one_probe=True,
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
    if verbose:
        print("\n*** running plot_pos_sep_and_one_probe() ***")
        # print(f'pos_sep_and_one_probe_df:\n{pos_sep_and_one_probe_df}')

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
    :param thr1_df: df of values from cond 1 (e.g., probe_jump inwards)
    :param thr2_df: df of values for cond 2 (e.g., probe_jump outwards)
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

    :return: ALLDATA-sorted.xlsx: A pandas DataFrame with n xlsx file of all data for one run of all ISIs.
    """

    print("\n***running a_data_extraction()***\n")

    # get run name/number
    path, run = os.path.split(run_dir)
    if verbose:
        print(f"run: {run}")

    if isi_list is None:
        isi_list = [-1, 0, 2, 4, 6, 9, 12, 24]

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
        trial_numbers = list(this_isi_df['total_nTrials'])
        this_isi_df = this_isi_df.sort_values(by=['stair', 'total_nTrials'])

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
        # save_name = f'{run}_ALLDATA-sorted.xlsx'
        save_name = 'ALLDATA-sorted.xlsx'

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


"""
4. b3_plot_staircase.m: staircases-ISIxxx.png: xxx corresponds to isi conditions. 
    Eight figure (8 isi conditions) with seven panels on each (7 probes separation 
    conditions) showing the Luminance value of two staircases as function of 
    trial number. 
"""
def b3_plot_staircase(all_data_path, thr_col='probeLum', resp_col='trial_response',
                      show_plots=True, save_plots=True, verbose=True):

    """
    b3_plot_staircase.m: staircases-ISIxxx.png: xxx corresponds to isi conditions.
    Eight figure (8 isi conditions) with seven panels on each (7 probes separation
    conditions) showing the Luminance value of two staircases as function of
    trial number. Eighth panel shows psignifit thr per sep condition.

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
                                usecols=['ISI', 'stair', 'total_nTrials',
                                         'probeLum', 'trial_response', 'resp.rt'])

    # get list of isi and stair values to loop through
    stair_list = all_data_df['stair'].unique()
    isi_list = all_data_df['ISI'].unique()
    # get isi string for column names
    isi_name_list = ['Concurrent' if i == -1 else f'isi{i}' for i in isi_list]

    trials, columns = np.shape(all_data_df)
    trials_per_stair = int(trials/len(isi_list)/len(stair_list))

    separation_title = ['18sep', '06sep', '03sep', '02sep', '01sep', '00sep', 'onePb']

    if verbose:
        print(f"all_data_df:\n{all_data_df}")
        print(f"{len(isi_list)} isi values and {len(stair_list)} stair values")
        print(f"isi_list: {isi_list}")
        print(f"isi_name_list: {isi_name_list}")
        print(f"stair_list: {stair_list}")
        print(f"separation_title: {separation_title}")
        print(f"trials_per_stair: {trials_per_stair}")

    # # the eighth plot is the psignifit thr for each sep (+sep, -sep and mean).
    # # get data from psignifit_thresholds.csv and reshape here
    psignifit_thr_df = pd.read_csv(f'{save_path}{os.sep}psignifit_thresholds.csv')
    if verbose:
        print(f'\npsignifit_thr_df:\n{psignifit_thr_df}')
    psignifit_thr_df = psignifit_thr_df.drop(['stair'], axis=1)
    psignifit_thr_df.columns = isi_name_list

    # split into pos_sep, neg_sep and mean of pos and neg.
    psig_pos_sep_df, psig_neg_sep_df = split_df_alternate_rows(psignifit_thr_df)
    psig_thr_mean_df = pd.concat([psig_pos_sep_df, psig_neg_sep_df]).groupby(level=0).mean()

    # add sep column in
    sep_list = [0, 1, 2, 3, 6, 18, 20]
    psig_thr_mean_df.insert(0, 'sep', sep_list)
    psig_pos_sep_df.insert(0, 'sep', sep_list)
    psig_neg_sep_df.insert(0, 'sep', sep_list)
    if verbose:
        print(f'\npsig_pos_sep_df:\n{psig_pos_sep_df}')
        print(f'\npsig_neg_sep_df:\n{psig_neg_sep_df}')
        print(f'\npsig_thr_mean_df:\n{psig_thr_mean_df}')

    # the values listed as separation=20 are actually for the single probe cond.
    # Chop last row off and add values later.
    psig_thr_mean_df, mean_one_probe_df = \
        psig_thr_mean_df.drop(psig_thr_mean_df.tail(1).index), psig_thr_mean_df.tail(1)
    psig_pos_sep_df, thr1_one_probe_df = \
        psig_pos_sep_df.drop(psig_pos_sep_df.tail(1).index), psig_pos_sep_df.tail(1)
    psig_neg_sep_df, thr2_one_probe_df = \
        psig_neg_sep_df.drop(psig_neg_sep_df.tail(1).index), psig_neg_sep_df.tail(1)
    if verbose:
        print(f'\npsig_thr_mean_df (chopped off one_probe):\n{psig_thr_mean_df}')

    # put the one_probe values into a df to use later
    one_probe_df = pd.concat([thr1_one_probe_df, thr2_one_probe_df, mean_one_probe_df],
                             ignore_index=True)
    one_probe_df.insert(0, 'dset', ['thr1', 'thr2', 'mean'])
    one_probe_df.insert(0, 'x_val', [-1, -1, -1])
    if verbose:
        print(f'one_probe_df:\n{one_probe_df}')

    # make empty arrays to save reversal n_reversals
    n_reversals_np = np.zeros(shape=[len(stair_list), len(isi_list)])


    # loop through isi values
    for isi_idx, isi in enumerate(isi_list):

        # get df for this isi only
        isi_df = all_data_df[all_data_df['ISI'] == isi]
        isi_name = isi_name_list[isi_idx]

        # initialise 8 plot figure
        # # this is a figure showing n_reversals per staircase condition.
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12, 6))
        ax_counter = 0

        for row_idx, row in enumerate(axes):
            for col_idx, ax in enumerate(row):
                print(f'\nrow: {row_idx}, col: {col_idx}')

                # for the first seven plots...
                if ax_counter < 7:
                    # # get pairs of stairs (e.g., [[18, -18], [6, -6], ...etc)
                    stair_odd = (ax_counter*2)+1  # 1, 3, 5, 7, 9, 11, 13
                    stair_odd_df = isi_df[isi_df['stair'] == stair_odd]
                    stair_odd_df.insert(0, 'step', list(range(trials_per_stair)))
                    final_lum_odd = stair_odd_df.loc[stair_odd_df['step'] == trials_per_stair-1, 'probeLum'].item()
                    n_reversals_odd = trials_per_stair - stair_odd_df[resp_col].sum()

                    stair_even = (ax_counter+1)*2  # 2, 4, 6, 8, 10, 12, 14
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
                    """use the psignifit values from each stair pair (e.g., 18, -18) to
                    get the mean threshold for each sep condition.
                    """
                    if verbose:
                        print("\nEighth plot")
                        print(f'psig_thr_mean_df:\n{psig_thr_mean_df}')
                        print(f'\none_probe_df:\n{one_probe_df}')

                    isi_thr_mean_df = pd.concat([psig_thr_mean_df["sep"], psig_thr_mean_df[isi_name]],
                                                axis=1, keys=['sep', isi_name])
                    if verbose:
                        print(f'isi_thr_mean_df:\n{isi_thr_mean_df}')

                    # line plot for thr1, th2 and mean thr
                    sns.lineplot(ax=axes[row_idx, col_idx], data=psig_pos_sep_df,
                                 x='sep', y=isi_name, color='tab:red',
                                 linewidth=.5)
                    sns.lineplot(ax=axes[row_idx, col_idx], data=psig_neg_sep_df,
                                 x='sep', y=isi_name, color='tab:blue',
                                 linewidth=.5)
                    sns.lineplot(ax=axes[row_idx, col_idx], data=isi_thr_mean_df,
                                 x='sep', y=isi_name, color='tab:green')

                    # scatter plot for single probe conditions
                    sns.scatterplot(data=one_probe_df, x="x_val", y=isi_name,
                                    hue='dset',
                                    palette=['tab:red', 'tab:blue', 'tab:green'])
                    ax.legend(fontsize=8, markerscale=.5)

                    # decorate plot
                    ax.set_title(f'{isi_name} end threshold')
                    ax.set_xticks([-2, -1, 0, 1, 2, 3, 6, 18])
                    ax.set_xticklabels(['', '\n1probe', 0, 1, 2, 3, 6, 18])
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
    n_reversals_df = n_reversals_df.astype(int)
    if verbose:
        print(f'\nn_reversals_df:\n{n_reversals_df}')
    n_reversals_df.to_csv(f'{save_path}{os.sep}n_reversals.csv')

    print("\n***finished b3_plot_staircases()***\n")

####################
# all_data_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/Kim/' \
#                     'Nick_practice/P6a-Kim/ALLDATA-sorted.xlsx'
# b3_plot_staircase(all_data_path, thr_col='probeLum', resp_col='trial_response',
#                   show_plots=True, save_plots=True, verbose=True)

def c_plots(save_path, thr_col='probeLum', show_plots=True, verbose=True):

    """
    5. c_plots.m: uses psignifit_thresholds.csv and output  plots.

    figures:
            data.png: threshold luminance as function of probe separation.
                      Positive separation values only, all ISIs on same axis.
                      Use plot_pos_sep_and_one_probe()
                      
            dataDivOneProbe: threshold luminance as function of probe separation.
                      Positive separation values only, all ISIs on same axis.
                      Use plot_pos_sep_and_one_probe(one_probe=False)

                      
            runs.png: threshold luminance as function of probe separation, 
                      Positive and negative separation values (batman plots), 
                      one panel one isi condition.
                      use eight_batman_plots()
                      
    :param save_path: path to run dir containing psignifit_thresholds.csv, where plots will be save.
    :param thr_col: column for threshold (e.g., probeLum)
    :param show_plots: Default True
    :param verbose: Default True.

    :return:
    """


    print("\n*** running c_plots() ***\n")

    isi_name_list = ['Concurrent', 'ISI0', 'ISI2', 'ISI4', 'ISI6', 'ISI9', 'ISI12', 'ISI24']
    sym_sep_list = [-18, -6, -3, -2, -1, 0, 1, 2, 3, 6, 18, 20]
    sym_sep_tick_labels = [-18, -6, -3, -2, -1, 0, 1, 2, 3, 6, 18, '1\nprobe']
    pos_sep_list = [0, 1, 2, 3, 6, 18, 20]

    # load df mean of last n probeLum values (14 stairs x 8 isi).
    psig_thr_df = pd.read_csv(f'{save_path}{os.sep}psignifit_thresholds.csv')
    if verbose:
        print(f'psig_thr_df:\n{psig_thr_df}')

    psig_thr_df = psig_thr_df.drop(['stair'], axis=1)
    psig_thr_df.columns = isi_name_list

    # # split into pos_sep, neg_sep and mean of pos and neg.
    # psig_pos_sep_df, psig_neg_sep_df = split_df_alternate_rows(psig_thr_df)
    # psig_thr_mean_df = pd.concat([psig_pos_sep_df, psig_neg_sep_df]).groupby(level=0).mean()
    #
    # # add sep column in
    # # sep_list = [0, 1, 2, 3, 6, 18, 20]
    # # psig_thr_mean_df.insert(0, 'sep', sep_list)
    # # psig_pos_sep_df.insert(0, 'sep', sep_list)
    # # psig_neg_sep_df.insert(0, 'sep', sep_list)
    # if verbose:
    #     print(f'\npsig_pos_sep_df:\n{psig_pos_sep_df}')
    #     print(f'\npsig_neg_sep_df:\n{psig_neg_sep_df}')
    #     print(f'\npsig_thr_mean_df:\n{psig_thr_mean_df}')

    # # the values listed as separation=20 are actually for the single probe cond.
    # # Chop last row off and add values later.
    # psig_thr_mean_df, mean_one_probe_df = \
    #     psig_thr_mean_df.drop(psig_thr_mean_df.tail(1).index), psig_thr_mean_df.tail(1)
    # psig_pos_sep_df, thr1_one_probe_df = \
    #     psig_pos_sep_df.drop(psig_pos_sep_df.tail(1).index), psig_pos_sep_df.tail(1)
    # psig_neg_sep_df, thr2_one_probe_df = \
    #     psig_neg_sep_df.drop(psig_neg_sep_df.tail(1).index), psig_neg_sep_df.tail(1)
    # if verbose:
    #     print(f'\npsig_thr_mean_df (chopped off one_probe):\n{psig_thr_mean_df}')

    # # put the one_probe values into a df to use later
    # one_probe_df = pd.concat([thr1_one_probe_df, thr2_one_probe_df, mean_one_probe_df],
    #                          ignore_index=True)
    # one_probe_df.insert(0, 'dset', ['thr1', 'thr2', 'mean'])
    # one_probe_df.insert(0, 'x_val', [-1, -1, -1])
    # if verbose:
    #     print(f'one_probe_df:\n{one_probe_df}')


    # lastN_pos_sym_np has values for 1-indexed stairs [1, 3, 5, 7, 9, 11, 9, 7, 5, 3, 1, 13]
    # these correspond to separation values:     [18, 6, 3, 2, 1, 0, 1, 2, 3, 6, 18, 99]
    pos_sym_indices = [0, 2, 4, 6, 8, 10, 8, 6, 4, 2, 0, 12]
    psig_pos_sep_df = psig_thr_df.iloc[pos_sym_indices]
    psig_pos_sep_df.reset_index(drop=True, inplace=True)

    # lastN_neg_sym_np has values for 1-indexed stairs [2, 4, 6, 8, 10, 12, 10, 8, 6, 4, 2, 14]
    # these correspond to sep values:   [-18, -6, -3, -2, -1, 0, -1, -2, -3, -6, -18, 99]
    neg_sym_indices = [1, 3, 5, 7, 9, 11, 9, 7, 5, 3, 1, 13]
    psig_neg_sep_df = psig_thr_df.iloc[neg_sym_indices]
    psig_neg_sep_df.reset_index(drop=True, inplace=True)

    # mean of pos and neg sep values [18, 6, 3, 2, 1, 0, 1, 2, 3, 6, 18, 99]
    psig_thr_mean_df = pd.concat([psig_pos_sep_df, psig_neg_sep_df]).groupby(level=0).mean()

    # subtract the dfs from each other, then for each column get the sum of abs values
    diff_val = np.sum(abs(psig_pos_sep_df - psig_neg_sep_df), axis=0)
    # take the mean of these across all ISIs to get single value
    mean_diff_val = float(np.mean(diff_val))

    # add sep column into dfs
    psig_thr_mean_df.insert(0, 'sep', sym_sep_list)
    psig_pos_sep_df.insert(0, 'sep', sym_sep_list)
    psig_neg_sep_df.insert(0, 'sep', sym_sep_list)

    if verbose:
        print(f'\npsig_pos_sep_df:\n{psig_pos_sep_df}')
        print(f'\npsig_neg_sep_df:\n{psig_neg_sep_df}')
        print(f'\npsig_thr_mean_df:\n{psig_thr_mean_df}')
        print(f'\ndiff_val:\n{diff_val}')
        print(f'\nmean_diff_val: {mean_diff_val}')


    # # Figure1 - runs-{n}lastValues
    # this is a figure with one axis per isi, showing neg and pos sep
    # (e.g., -18:18) - eight batman plots
    fig_title = f'Psignifit thresholds per ISI. ' \
                f'(mean diff: {round(mean_diff_val, 2)})'
    fig1_savename = f'runs.png'
    eight_batman_plots(mean_df=psig_thr_mean_df,
                       thr1_df=psig_pos_sep_df,
                       thr2_df=psig_neg_sep_df,
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
    fig3_save_name = f'data.png'
    fig_3_title = 'All ISIs and separations'
    plot_pos_sep_and_one_probe(pos_sep_and_one_probe_df=psig_thr_mean_df,
                               thr_col=thr_col,
                               fig_title=fig_3_title,
                               one_probe=True,
                               save_path=save_path,
                               save_name=fig3_save_name,
                               verbose=True)
    if show_plots:
        plt.show()
    plt.close()

    # # # FIGURE4 - 'dataDivOneProbe-{n}lastValues.png' - all ISIs on same axis, pos sep only.
    #         # does not include single probe
    # # # use plot_pos_sep_and_one_probe(one_probe=False)
    # # each sep row in pos_sep_df is divided by one_probe_df.
    fig4_save_name = f'dataDivOneProbe.png'
    pos_sep_df, one_probe_df = split_df_into_pos_sep_df_and_one_probe_df(psig_thr_mean_df)
    pos_sep_arr = pos_sep_df.to_numpy()
    one_probe_arr = one_probe_df['probeLum'].to_numpy()
    div_by_one_probe_arr = (pos_sep_arr.T / one_probe_arr[:, None]).T
    div_by_one_probe_df = pd.DataFrame(div_by_one_probe_arr, columns=isi_name_list)
    div_by_one_probe_df.insert(0, 'Separation', pos_sep_list[:-1])
    div_by_one_probe_df.set_index('Separation', inplace=True)
    print(f'div_by_one_probe_df:\n{div_by_one_probe_df}')

    plot_pos_sep_and_one_probe(div_by_one_probe_df,
                               thr_col='probeLum',
                               fig_title=fig4_save_name[:-4],
                               one_probe=False,
                               save_path=save_path,
                               save_name=fig4_save_name,
                               verbose=True)
    if show_plots:
        plt.show()
    plt.close()


    print("\n***finished c_plots()***\n")


# #########
# c_plots(save_path='/Users/nickmartin/Documents/PycharmProjects/Cardiff/Kim/Nick_practice/P6a-Kim',
#         thr_col='probeLum', last_vals_list=[1, 4, 7],
#         show_plots=True, verbose=True)


def d_average_participant(root_path, run_dir_names_list,
                          thr_col='probeLum',
                          trimmed_mean=False,
                          show_plots=True, verbose=True):
    """
    d_average_participant: take psignifit_thresholds.csv
    in each participant run folder and make master lists  
    MASTER_psignifit_thresholds.csv

    Get mean threshold across 6 run conditions saved as
    MASTER_ave_thresh.csv
    
    Save master lists to folder containing the six runs (root_path).
    
    Plots:
    MASTER_ave_thresh saved as ave_thr_all_runs.png
    MASTER_ave_thresh two-probe/one-probesaved as ave_thr_div_one_probe.png

    :param root_path: dir containing run folders
    :param run_dir_names_list: names of run folders
    :param thr_col: Default: 'probeLum'. column with variable controlled by staircase
    :param trimmed_mean: default False - if True, will call function
        get_trimmed_mean_df(), which drops the highest and lowest value before
        calculating the mean.
    :param show_plots: default True - display plots
    :param verbose: Defaut true, print progress to screen
    """

    print("\n***running d_average_participant()***\n")

    # # part1. Munge data, save master lists and get means etc
    # #  - loop through runs and get each P-next-thresholds and P-reversal4-thresholds
    # # Make master sheets: MASTER_next_thresh & MASTER_reversal_4_thresh
    # # Incidentally the MATLAB script didn't specify which reversals data to use,
    # # although the figures imply Martin used last3 reversals.
    isi_name_list = ['Concurrent', 'ISI0', 'ISI2', 'ISI4',
                     'ISI6', 'ISI9', 'ISI12', 'ISI24']
    pos_sep_list = [0, 1, 2, 3, 6, 18, 20]
    sep_list = [18, -18, 6, -6, 3, -3, 2, -2, 1, -1, 0, -.1, 20, -20]
    all_psignifit_list = []

    for run_idx, run_name in enumerate(run_dir_names_list):

        this_psignifit_df = pd.read_csv(f'{root_path}{os.sep}{run_name}{os.sep}psignifit_thresholds.csv')
        print(f'{run_idx}. {run_name} - this_psignifit_df:\n{this_psignifit_df}')

        if 'Unnamed: 0' in list(this_psignifit_df):
            this_psignifit_df.drop('Unnamed: 0', axis=1, inplace=True)
            print('foundunnamed')
        this_psignifit_df.drop(columns='stair', inplace=True)
        this_psignifit_df.insert(0, 'Separation', sep_list)
        this_psignifit_df.insert(0, 'Run', run_idx)
        all_psignifit_list.append(this_psignifit_df)


    all_data_psignifit_df = pd.concat(all_psignifit_list, ignore_index=True)
    all_data_psignifit_df.to_csv(f'{root_path}{os.sep}MASTER_psignifit_thresholds.csv', index=False)
    if verbose:
        print(f'\nall_data_psignifit_df:\n{all_data_psignifit_df}')

    # todo: split pos and neg values so that there are actually 12 data points to average over,
    #  Could do this in the loop and label then run1_pos, 'run1_neg' etc

    # get mean of all runs (each sep and ISI)
    # try just grouping the master sheet first, rather than using concat.
    if trimmed_mean:
        print('calling robust mean function')
        # todo: for 12 data point, drop the 2 highest and lowst
        ave_TM_psignifit_thr_df = get_trimmed_mean_df(all_data_psignifit_df)
        ave_TM_psignifit_thr_df.to_csv(f'{root_path}{os.sep}MASTER_ave_TM_thresh.csv')
        if verbose:
            print(f'ave_TM_psignifit_thr_df:\n{ave_TM_psignifit_thr_df}')
    else:
        ave_psignifit_thr_df = all_data_psignifit_df.drop('Run', axis=1)
        ave_psignifit_thr_df = ave_psignifit_thr_df.groupby('Separation', sort=False).mean()
        ave_psignifit_thr_df.to_csv(f'{root_path}{os.sep}MASTER_ave_thresh.csv')
        if verbose:
            print(f'ave_psignifit_thr_df:\n{ave_psignifit_thr_df}')


    # part 2. main Figures (these are the ones saved in the matlab script)
    # Fig1: single ax, pos_sep_and_one_probe.  Uses ave_next_thr_df: for thr values (e.g., mean of all runs).
    # Fig2: single ax, pos_sep_and_one_probe (but no one probe).
    #     threshold values are mean of two probes / single probe

    # # Fig1a
    # todo: add error bars

    if trimmed_mean:
        fig_df = ave_TM_psignifit_thr_df
        fig1_title = f'Participant trimmed mean of thresholds across all runs'
        fig1_savename = f'ave_TM_thr_all_runs.png'
    else:
        fig_df = ave_psignifit_thr_df
        fig1_title = f'Participant average threshold across all runs'
        fig1_savename = f'ave_thr_all_runs.png'

    # todo: change this so it includes neg sep values as part of the plot.
    plot_pos_sep_and_one_probe(pos_sep_and_one_probe_df=fig_df,
                               fig_title=fig1_title,
                               save_path=root_path,
                               save_name=fig1_savename)
    if show_plots:
        plt.show()
    plt.close()

    if verbose:
        print('finished fig1a')

    # todo: fig1b with ISI on x-axis and different lines for each sep.

    # # Fig 2
    # todo: add error bars
    if trimmed_mean:
        fig2_save_name = 'ave_TM_thr_div_one_probe.png'
        fig2_title = 'Participant trimmed mean of thresholds divided by single probe'
    else:
        fig2_save_name = 'ave_thr_div_one_probe.png'
        fig2_title = 'Participant average threshold divided by single probe'

    pos_sep_df, one_probe_df = split_df_into_pos_sep_df_and_one_probe_df(fig_df)
    pos_sep_arr = pos_sep_df.to_numpy()
    one_probe_arr = one_probe_df['probeLum'].to_numpy()
    div_by_one_probe_arr = (pos_sep_arr.T / one_probe_arr[:, None]).T
    div_by_one_probe_df = pd.DataFrame(div_by_one_probe_arr, columns=isi_name_list)
    div_by_one_probe_df.insert(0, 'Separation', pos_sep_list[:-1])
    div_by_one_probe_df.set_index('Separation', inplace=True)
    if verbose:
        print(f'div_by_one_probe_df:\n{div_by_one_probe_df}')

    plot_pos_sep_and_one_probe(div_by_one_probe_df,
                               thr_col=thr_col,
                               fig_title=fig2_title,
                               one_probe=False,
                               save_path=root_path,
                               save_name=fig2_save_name,
                               verbose=True)
    if show_plots:
        plt.show()
    plt.close()

    # todo: fig2b with ISI on x-axis and different lines for each sep.



    print("\n*** finished d_average_participant()***\n")




#######
# root_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/Kim'  # master folder containing all runs
# run_dir_names_list = ['P6a-Kim', 'P6b-Kim', 'P6c-Kim', 'P6d-Kim', 'P6e-Kim', 'P6f-Kim']
# d_average_participant(root_path=root_path, run_dir_names_list=run_dir_names_list)
