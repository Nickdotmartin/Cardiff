import os

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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
"""

pd.options.display.float_format = "{:,.2f}".format


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
                                              one_probe_pos=None,
                                              verbose=True):
    """
    For plots where positive separations are shown as line plots and 
    one probe results are shown as scatter plot, this function splits the dataframe into two.
    
    :param pos_sep_and_one_probe_df: Dataframe of positive separations with
        one_probe conds at bottom of df (e.g., shown as 20 or 99).  df must be indexed with the separation column.
    :param isi_name_list: List of isi names.  If None, will use default values.
    :param one_probe_pos: Default=None, use value from df.  Where to set location of 1probes on x-axis.
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
    if one_probe_pos is None:
        print('\n\nidiot check')
        one_probe_pos = one_probe_df.index.tolist()[-1]
        print(f'one_probe_pos: {one_probe_pos}')

    one_probe_lum_list = one_probe_df.values.tolist()[0]
    one_probe_dict = {'ISIs': isi_name_list,
                      'probeLum': one_probe_lum_list,
                      'x_vals': [one_probe_pos for _ in isi_name_list]}
    one_probe_df = pd.DataFrame.from_dict(one_probe_dict)
    if verbose:
        print(f'one_probe_df:\n{one_probe_df}')

    return pos_sep_df, one_probe_df


# # choose colour palette
def fig_colours(n_conditions, alternative_colours=False):
    """
    Use this to always get the same colours in the same order with no fuss.
    :param n_conditions: number of different colours - use 256 for colourmap
        (e.g., for heatmaps or something where colours are used in continuous manner)
    :param alternative_colours: a second pallet of alternative colours.
    :return: a colour pallet
    """

    use_colours = 'colorblind'
    if alternative_colours:
        use_colours = 'husl'

    if 10 < n_conditions < 21:
        use_colours = 'tab20'
    elif n_conditions > 20:
        use_colour = 'spectral'

    use_cmap = False

    my_colours = sns.color_palette(palette=use_colours, n_colors=n_conditions, as_cmap=use_cmap)
    sns.set_palette(palette=use_colours, n_colors=n_conditions)

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
                               error_bars_df=None,
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
    :param error_bars_df: default: None. can pass a dataframe containing x, y and yerr values for error bars.
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
        pos_set_ticks = [0, 1, 2, 3, 6, 18, 20]
    if pos_tick_labels is None:
        pos_tick_labels = [0, 1, 2, 3, 6, 18, 'one\nprobe']

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

    if error_bars_df is not None:
        lvls = error_bars_df.group.unique()
        for i in lvls:
            ax.errorbar(x=error_bars_df[error_bars_df['group'] == i]["x"],
                        y=error_bars_df[error_bars_df['group'] == i]["y"],
                        yerr=error_bars_df[error_bars_df['group'] == i]["stderr"], label=i)

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
        ax.set_xticks(pos_set_ticks[:-1])
        ax.set_xticklabels(pos_tick_labels[:-1])

    # ax.set_ylim([0, 110])
    ax.set_xlabel('Probe separation in diagonal pixels')
    ax.set_ylabel('Probe Luminance')

    if fig_title is not None:
        plt.title(fig_title)
        
    if save_path is not None:
        if save_name is not None:
            plt.savefig(f'{save_path}{os.sep}{save_name}')

    return fig

####################

def plot_1probe_w_errors(fig_df, error_df, split_1probe=True,
                         jitter=True, error_caps=False, alt_colours=False,
                         legend_names=None,
                         x_tick_vals=None,
                         x_tick_labels=None,
                         fixed_y_range=False,
                         fig_title=None, save_name=None, save_path=None,
                         verbose=True):
    """
    Calculate and plot the mean and error estimates (y-axis) at each separation values (x-axis) including 1probe.
    Separate line for each ISI.  Error bar values taken from separate error_df.

    :param fig_df: dataframe to build plot from.  Expects fig_df in the form:
        Separation as index, 1probe as bottom row, ISIs as columns.
    :param error_df: dataframe of same shape as fig_df, but contains error values
    :param split_1probe: Default=True - whether to treat 1probe data separately,
        e.g., not joined with line to 2probe data.
    :param jitter: Jitter x_axis values so points don't overlap.
    :param error_caps: caps on error bars for more easy reading
    :param alt_colours: Use different set of colours to normal (e.g., if ISI on
        x-axis and lines for each Separation).
    :param legend_names: Names of different lines (e.g., ISI names)
    :param x_tick_vals: Positions on x-axis.
    :param x_tick_labels: labels for x-axis.
    :param fixed_y_range: default=False. If True will use full range of y values
        (e.g., 0:110) or can pass a tuple to set y_limits.
    :param fig_title: Title for figure
    :param save_name: filename of plot
    :param save_path: path to folder where plots will be saved
    :param verbose: print progress to screen

    :return: figure
    """
    print('\n*** running plot_1probe_w_errors() ***\n')

    if verbose:
        print(f'fig_df:\n{fig_df}')

    # split 1probe from bottom of fig_df and error_df
    if split_1probe:
        two_probe_df, one_probe_df = split_df_into_pos_sep_df_and_one_probe_df(fig_df)
        two_probe_er_df, one_probe_er_df = split_df_into_pos_sep_df_and_one_probe_df(error_df)
        if verbose:
            print(f'one_probe_df:\n{one_probe_df}')
            print(f'one_probe_er_df:\n{one_probe_er_df}')
    else:
        two_probe_df = fig_df
        two_probe_er_df = error_df
    if verbose:
        print(f'two_probe_df:\n{two_probe_df}')
        print(f'two_probe_er_df:\n{two_probe_er_df}')

    # get names for legend (e.g., different lines)
    column_names = two_probe_df.columns.to_list()
    if legend_names is None:
        legend_names = column_names
    if verbose:
        print(f'Column and Legend names:')
        for a, b in zip(column_names, legend_names):
            print(f"{a}\t=>\t{b}\tmatch: {bool(a==b)}")

    # get number of locations for jitter list
    n_pos_sep = len(two_probe_df.index.to_list())

    jit_max = 0
    if jitter:
        jit_max = .2
        if type(jitter) in [float, np.float]:
            jit_max = jitter

    cap_size = 0
    if error_caps:
        cap_size = 5

    # set colour palette
    my_colours = fig_colours(len(column_names), alternative_colours=alt_colours)

    fig, ax = plt.subplots()

    legend_handles_list = []

    for idx, name in enumerate(column_names):

        # get rand float to add to x-axis for jitter
        jitter_list = np.random.uniform(size=n_pos_sep, low=-jit_max, high=jit_max)

        if split_1probe:
            one_probe = ax.errorbar(x=one_probe_df['x_vals'][idx] + np.random.uniform(low=-jit_max, high=jit_max),
                                    y=one_probe_df['probeLum'][idx],
                                    yerr=one_probe_er_df['probeLum'][idx],
                                    marker='.', lw=2, elinewidth=.7,
                                    capsize=cap_size,
                                    color=my_colours[idx])

        two_probes = ax.errorbar(x=two_probe_df.index + jitter_list,
                                 y=two_probe_df[name], yerr=two_probe_er_df[name],
                                 marker='.', lw=2, elinewidth=.7,
                                 capsize=cap_size,
                                 color=my_colours[idx])

        leg_handle = mlines.Line2D([], [], color=my_colours[idx], label=name,
                                   marker='.', linewidth=.5, markersize=4)
        legend_handles_list.append(leg_handle)

    ax.legend(handles=legend_handles_list, fontsize=6, title='ISI', framealpha=.5)

    if x_tick_vals is not None:
        ax.set_xticks(x_tick_vals)
    if x_tick_labels is not None:
        ax.set_xticklabels(x_tick_labels)

    ax.set_xlabel('Probe separation in diagonal pixels')
    ax.set_ylabel('Probe Luminance')

    if fixed_y_range:
        ax.set_ylim([0, 110])
        if type(fixed_y_range) in [tuple, list]:
            ax.set_ylim([fixed_y_range[0], fixed_y_range[1]])

    if fig_title is not None:
        plt.title(fig_title)

    if save_path is not None:
        if save_name is not None:
            plt.savefig(f'{save_path}{os.sep}{save_name}')

    return fig


###################

def plot_w_errors_no_1probe(wide_df, x_var, y_var, lines_var,
                            legend_names=None,
                            x_tick_labels=None,
                            alt_colours=True,
                            fixed_y_range=False,
                            jitter=True,
                            error_caps=True,
                            fig1b_title=None,
                            fig1b_savename=None,
                            save_path=None,
                            verbose=True):
    """
    Function to plot pointplot with error bars.  Use this for plots unless
    there is a need for the separate 1probe condition.  Note: x-axis is categorical
    so its not easy to move ticks.  If I want to do this, use plot_1probe_w_errors().

    :param wide_df: wide form dataframe with data from multiple runs
    :param x_var: Name of variable to go on x-axis (should be consistent with wide_df)
    :param y_var: Name of variable to go on y-axis (should be consistent with wide_df)
    :param lines_var: Name of variable for the lines (should be consistent with wide_df)
    # :param isi_name_list:
    :param legend_names: Default: None, which will access frequently used names.
        Else pass list of names to appear on legend, use verbose to compare order with matplotlib assumptions.
    :param x_tick_labels: Default: None, which will access frequently used labels.
        Else pass list of labels to appearn on x-axis.  Note: for pointplot x-axis is catagorical,
        not numerical; so all x-ticks are evently spaced.  For variable x-axis use plot_1probe_w_errors().
    :param alt_colours: Default=True.  Use alternative colours to differentiate
        from other plots e.g., colours associated with Sep not ISI.
    :param fixed_y_range: If True it will fix y-axis to 0:110.  Otherwise uses adaptive range.
    :param jitter: Points on x-axis.
    :param error_caps: Whether to have caps on error bars.
    :param fig1b_title: Title for figure
    :param fig1b_savename: Name for figure.
    :param save_path: Path to folder to save plot.
    :param verbose: Print progress to screen.

    :return: fig
    """

    print('\n*** Running plot_w_errors_no_1probe() ***')

    # get default values.
    if legend_names is None:
        legend_names = ['0', '1', '2', '3', '6', '18', '1probe']
    if x_tick_labels is None:
        x_tick_labels = ['conc', 0, 2, 4, 6, 9, 12, 24]

    # get names for legend (e.g., different lines_var)
    n_colours = len(wide_df[lines_var].unique())

    # do error bars have caps?
    cap_size = 0
    if error_caps:
        cap_size = .1

    # convert wide_df to long for getting means and standard error.
    long_fig_df = make_long_df(wide_df, idx_col='stack')
    if verbose:
        print(f'long_fig_df:\n{long_fig_df}')

    my_colours = fig_colours(n_colours, alternative_colours=alt_colours)
    print(f"my_colours - {np.shape(my_colours)}\n{my_colours}")

    fig, ax = plt.subplots()
    sns.pointplot(data=long_fig_df, x=x_var, y=y_var, hue=lines_var,
                  estimator=np.mean, ci=68, dodge=jitter, markers='.',
                  errwidth=1, capsize=cap_size, palette=my_colours, ax=ax)

    # sort legend
    handles, orig_labels = ax.get_legend_handles_labels()
    if legend_names is None:
        legend_names = orig_labels
    if verbose:
        print(f'orig_labels and Legend names:')
        for a, b in zip(orig_labels, legend_names):
            print(f"{a}\t=>\t{b}\tmatch: {bool(a == b)}")
    ax.legend(handles, legend_names, fontsize=6, title=lines_var, framealpha=.5)

    # decorate plot
    if x_tick_labels is not None:
        ax.set_xticklabels(x_tick_labels)
    ax.set_xlabel(x_var)

    if y_var is 'probeLum':
        ax.set_ylabel('Probe Luminance')
    else:
        ax.set_ylabel(y_var)

    if fixed_y_range:
        ax.set_ylim([0, 110])
        if type(fixed_y_range) in [tuple, list]:
            ax.set_ylim([fixed_y_range[0], fixed_y_range[1]])

    if fig1b_title is not None:
        plt.title(fig1b_title)

    if save_path is not None:
        if fig1b_savename is not None:
            plt.savefig(f'{save_path}{os.sep}{fig1b_savename}')

    return fig


###########################


def plot_thr_heatmap(heatmap_df,
                     x_tick_labels=None,
                     y_tick_labels=None,
                     fig_title=None,
                     save_name=None,
                     save_path=None,
                     verbose=True):
    """
    Function for making a heatmap
    :param heatmap_df: Expects dataframe with Separation as index and ISIs as columns.
    :param x_tick_labels: Labels for columns
    :param y_tick_labels: Labels for rows
    :param fig_title:
    :param save_name:
    :param save_path:
    :param verbose:
    :return: Heatmap
    """

    print('\n*** running plot_thr_heatmap() ***\n')

    if verbose:
        print(f'heatmap_df:\n{heatmap_df}')

    if x_tick_labels is None:
        x_tick_labels = ['conc', 'isi 0', 'isi 2', 'isi 4', 'isi 6', 'isi 9', 'isi12', 'isi 24']
    if y_tick_labels is None:
        y_tick_labels = [0, 1, 2, 3, 6, 18, '1probe']

    # get mean of each column, then mean of those
    mean_thr = float(np.mean(heatmap_df.mean()))
    if verbose:
        print(f'mean_val: {round(mean_thr, 2)}')

    heatmap = sns.heatmap(data=heatmap_df,
                          annot=True, center=mean_thr,
                          cmap=sns.color_palette("Spectral", as_cmap=True),
                          xticklabels=x_tick_labels, yticklabels=y_tick_labels)

    heatmap.set_xlabel('ISI')

    if fig_title is not None:
        plt.title(fig_title)

    if save_path is not None:
        if save_name is not None:
            plt.savefig(f'{save_path}{os.sep}{save_name}')

    return heatmap

##########################

def trim_n_high_n_low(all_data_df, trim_from_ends=None, reference_col='Separation',
                      stack_col_id='stack', verbose=True):
    """
    Function for trimming the n highest and lowest values from each condition of a set with multiple runs.

    :param all_data_df: Dataset to be trimmed.
    :param trim_from_ends: number of values to trim from each end of the distribution.
    :param reference_col: Idx column containing repeated conditions (e.g., Separation has same label for each stack).
    :param stack_col_id: idx column showing different runs/groups etc (e.g., stack)
    :param verbose: in true will print progress to screen.

    :return: trimmed df
    """
    print('\n*** running trim_high_n_low() ***')

    '''Part 1, convert 2d df into 3d numpy array'''
    # prepare to extract numpy
    if verbose:
        print(f'all_data_df {all_data_df.shape}:\n{all_data_df.head(25)}')

    # get unique values to loop over
    stack_list = list(all_data_df[stack_col_id].unique())
    datapoints_per_cond = len(stack_list)

    if verbose:
        print(f'stack_list: {stack_list}\n'
              f'datapoints_per_cond: {datapoints_per_cond}')

    # loop through df to get 3d numpy
    my_list = []
    for stack in stack_list:
        stack_df = all_data_df[all_data_df[stack_col_id] == stack]
        stack_df = stack_df.drop(stack_col_id, axis=1)
        sep_list = list(stack_df.pop(reference_col))
        isi_name_list = list(stack_df.columns)
        # print(f'stack{stack}_df ({stack_df.shape}):\n{stack_df}')
        my_list.append(stack_df.to_numpy())

    # 3d numpy array are indexed with [depth, row, col]
    # use variables depth_3d, rows_3d, cols_all later to reshaped_2d_array trimmed array
    my_3d_array = np.array(my_list)
    depth_3d, rows_3d, cols_all = np.shape(my_3d_array)

    if trim_from_ends is not None:
        target_3d_depth = depth_3d - 2*trim_from_ends
    else:
        target_3d_depth = depth_3d
    target_2d_rows = target_3d_depth * rows_3d
    if verbose:
        print(f'\nUse these values for defining array shapes.\n'
              f'target_3d_depth (depth-trim): {target_3d_depth}, '
              f'3d shape after trim (target_3d_depth, rows_3d, cols_all) = '
              f'({target_3d_depth}, {rows_3d}, {cols_all})\n'
              f'2d array shape (after trim, but before Separation, stack or headers are added): '
              f'(target_2d_rows, cols_all) = ({target_2d_rows}, {cols_all})')

    '''Part 2, trim highest and lowest n values from each depth slice to get trimmed_3d_list'''
    if verbose:
        print('\ngetting depth slices to trim...')
    trimmed_3d_list = []
    counter = 0
    for col in list(range(cols_all)):
        row_list = []
        for row in list(range(rows_3d)):
            depth_slice = my_3d_array[:, row, col]
            depth_slice = np.sort(depth_slice)
            if trim_from_ends is not None:
                trimmed = depth_slice[trim_from_ends: -trim_from_ends]
            else:
                trimmed = depth_slice[:]
            # if verbose:
                # print(f'{counter}: {trimmed}')
            row_list.append(trimmed)
            counter += 1
        trimmed_3d_list.append(row_list)

    """
    Part 3, turn 3d numpy back into 2d df.
    trimmed_3d_list is a list of arrays (e.g., 3d).  Each array relates to a
    depth-stack of my_3d_array which has now be trimmed (e.g., fewer rows).
    However, trimmed_3d_list has the same depth and number of columns as my_3d_array.
    trimmed_array re-shapes trimmed_3d_list so all values are in their original
    row and column positions (e.g., Separation and ISI).
    However, the 3rd dimension (depth) is not in original order, but in ascending order."""

    trimmed_3d_array = np.array(trimmed_3d_list)
    print(f'\n\nReshaping trimmed data\ntrimmed_3d_array: {np.shape(trimmed_3d_array)}')
    if verbose:
        print(trimmed_3d_array)

    ravel_array_f = np.ravel(trimmed_3d_array, order='F')
    print(f'\n1. ravel_array_f: {np.shape(ravel_array_f)}')
    if verbose:
        print(ravel_array_f)

    reshaped_3d_array = ravel_array_f.reshape(target_3d_depth, rows_3d, cols_all)
    print(f'\n2. reshaped_3d_array: {np.shape(reshaped_3d_array)}')
    if verbose:
        print(reshaped_3d_array)

    reshaped_2d_array = reshaped_3d_array.reshape(target_2d_rows, -1)
    print(f'\n3. reshaped_2d_array {np.shape(reshaped_2d_array)}')
    if verbose:
        print(reshaped_2d_array)

    # make dataframe and insert column for separation and stack (trimmed run/group)
    trimmed_df = pd.DataFrame(reshaped_2d_array, columns=isi_name_list)
    stack_col_vals = np.repeat(np.arange(target_3d_depth), rows_3d)
    sep_col_vals = sep_list*target_3d_depth
    trimmed_df.insert(0, 'stack', stack_col_vals)
    trimmed_df.insert(1, reference_col, sep_col_vals)
    print(f'\ntrimmed_df {trimmed_df.shape}:\n{trimmed_df}')

    print(f'trimmed {trim_from_ends} highest and lowest values ({2*trim_from_ends} in total) from each of the '
          f'{datapoints_per_cond} datapoints so there are now '
          f'{target_3d_depth} datapoints for each of the '
          f'{rows_3d} x {cols_all} conditions.')

    print('\n*** finished trim_high_n_low() ***')

    return trimmed_df


##################

def make_long_df(wide_df, wide_stubnames='ISI',
                 col_to_keep='Separation', idx_col='Run', verbose=True):

    """
    Function to convert a wide form df containing multiple measurements at each value
    (e.g., data dfs concatenated from several runs), into long-form dataframe
    :param wide_df: Expects a dataframe made by concatenating data from several runs.
        Dataframe expected to contain columns for: Run, Separation and ISI levels.
    :param wide_stubnames: repeated prefix for several columns (e.g., ISI0, ISI1, ISI2, ISI4 etc).
    :param col_to_keep: Existing columns of useful data (e.g., Separation)
    :param idx_col: Existing column of irrelavent data to use as index (e.g., Run)
    :param verbose: print progress to screen.

    :return: Long form dataframe with single index
    """
    print('\n*** running make_long_df() ***\n')

    if verbose:
        print(f'wide_df:\n{wide_df}')

    # add spaces to ISI names and change concurrent to 999.
    orig_col_names = list(wide_df.columns)
    new_col_names = [f"ISI {i.strip('ISI')}" if 'ISI' in i else i for i in orig_col_names]
    # change 'concurrent' to 999 not -1 as wide_to_long won't take negative numbers
    new_col_names = [f"ISI 999" if i == 'Concurrent' else i for i in new_col_names]
    wide_df.columns = new_col_names

    # use pandas wide_to_long for transform df
    long_df = pd.wide_to_long(wide_df, stubnames=wide_stubnames, i=[idx_col, col_to_keep], j='data',
                              sep=' ')
    if verbose:
        print(f'long_df:\n{long_df}')

    # # replace column values and labels
    long_df = long_df.rename({wide_stubnames: 'probeLum'}, axis='columns')
    # change concurrent ISI column from 999 to -1
    long_df.index = pd.MultiIndex.from_tuples([(x[0], x[1], 'ISI -1') if x[2] == 999 else
                                               (x[0], x[1], f'ISI {x[2]}') for x in long_df.index])

    # rename index columns
    long_df.index.names = [idx_col, col_to_keep, wide_stubnames]

    # go from multi-indexed df to single-indexed df.
    long_sngl_idx_df = long_df.reset_index([col_to_keep, wide_stubnames])
    if verbose:
        print(f'long_sngl_idx_df:\n{long_sngl_idx_df}')

    return long_sngl_idx_df


###########################


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
                         linewidth=2, linestyle="dotted", markers=True)

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
            ax.set_ylim([0, 110])

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
                                markersize=4, label='group1')
            st2 = mlines.Line2D([], [], color=my_colours[ax_counter],
                                marker='o', linewidth=.5,
                                markersize=4, label='group2')
            mean_line = mlines.Line2D([], [], color=my_colours[ax_counter],
                                      marker=None, linewidth=2, linestyle="dotted",
                                      label='mean')
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

    :return: RUNDATA-sorted.xlsx: A pandas DataFrame with n xlsx file of all data for one run of all ISIs.
    """

    print("\n***running a_data_extraction()***")

    # get run name/number
    path, run = os.path.split(run_dir)
    if verbose:
        print(f"run: {run}")

    if isi_list is None:
        isi_list = [-1, 0, 2, 4, 6, 9, 12, 24]

    # dicts for mapping stair number to separation and group.
    stair_sep_dict = {1: 18, 2: 18, 3: 6, 4: 6, 5: 3, 6: 3, 7: 2,
                      8: 2, 9: 1, 10: 1, 11: 0, 12: 0, 13: 99, 14: 99}
    stair_group_dict = {1: 1, 2: 2, 3: 1, 4: 2, 5: 1, 6: 2, 7: 1,
                        8: 2, 9: 1, 10: 2, 11: 1, 12: 2, 13: 1, 14: 2}

    # raw results csv doesn't have separation or group columns, so assume I'll add them and re-save raw data.
    resave_results = True

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

        # remove any Unnamed columns
        if any("Unnamed" in i for i in list(this_isi_df.columns)):
            unnamed_col = [i for i in list(this_isi_df.columns) if "Unnamed" in i][0]
            this_isi_df.drop(unnamed_col, axis=1, inplace=True)

        if 'Separation' in list(this_isi_df.columns):
            resave_results = False

        # add isi column for multi-indexing
        this_isi_df.insert(0, 'ISI', isi)

        # add in group column to differentiate identical stairs (e.g., 1&2, 3&4 etc)
        stack_list = this_isi_df['stair'].map(stair_group_dict)
        this_isi_df.insert(1, 'group', stack_list)

        # add in Separation column mapped from stair_sep_dict
        sep_list = this_isi_df['stair'].map(stair_sep_dict)
        this_isi_df.insert(2, 'Separation', sep_list)

        # sort by group, stair, original trial number
        trial_numbers = list(this_isi_df['total_nTrials'])
        this_isi_df = this_isi_df.sort_values(by=['group', 'stair', 'total_nTrials'])
        this_isi_df.insert(0, 'srtd_trial_idx', trial_numbers)

        if verbose:
            print(f'df sorted by stair:\n{this_isi_df.head()}')

        # get column names to use on all_data_df
        column_names = list(this_isi_df)

        # add to all_data
        all_data.append(this_isi_df)

        if resave_results:
            this_isi_df.to_csv(f'{run_dir}{os.path.sep}ISI_{isi}_probeDur2{os.path.sep}'
                               f'{p_name}_w_sep.csv', index=False)

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
        save_name = 'RUNDATA-sorted.xlsx'
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
        print(f"all_data_df:\n{all_data_df.head()}")
        print(f"{len(isi_list)} isi values and {len(stair_list)} stair values")
        print(f"isi_list: {isi_list}")
        print(f"isi_name_list: {isi_name_list}")
        print(f"stair_list: {stair_list}")
        print(f"separation_title: {separation_title}")
        print(f"trials_per_stair: {trials_per_stair}")

    '''the eighth plot is the psignifit thr for each sep (+sep, -sep and mean).
    get data from psignifit_thresholds.csv and reshape here'''
    thr_csv_name = f'{save_path}{os.sep}psignifit_thresholds.csv'
    psignifit_thr_df = pd.read_csv(thr_csv_name)
    if verbose:
        print(f'\npsignifit_thr_df:\n{psignifit_thr_df}')

    # remove extra columns
    if 'stair' in list(psignifit_thr_df.columns):
        psignifit_thr_df = psignifit_thr_df.drop(['stair'], axis=1)

    if 'group' in list(psignifit_thr_df.columns):
        psignifit_thr_df = psignifit_thr_df.drop(['group'], axis=1)

    if 'Separation' in list(psignifit_thr_df.columns):
        # test_sep_list = psignifit_thr_df.pop(['Separation'], axis=1)
        sep_list = psignifit_thr_df.pop('Separation').tolist()
        sep_list = [20 if i == 99 else i for i in sep_list]
    else:
        sep_list = [18, 6, 3, 2, 1, 0, 20]
    print(f'sep_list: {sep_list}')

    psignifit_thr_df.columns = isi_name_list

    # split into pos_sep, neg_sep and mean of pos and neg.
    psig_pos_sep_df, psig_neg_sep_df = split_df_alternate_rows(psignifit_thr_df)
    psig_thr_mean_df = pd.concat([psig_pos_sep_df, psig_neg_sep_df]).groupby(level=0).mean()

    # add sep column in
    rows, cols = psig_thr_mean_df.shape
    if len(sep_list) == rows*2:
        # takes every other item
        sep_list = sep_list[::2]
    print(f'sep_list: {sep_list}')

    psig_thr_mean_df.insert(0, 'Separation', sep_list)
    psig_pos_sep_df.insert(0, 'Separation', sep_list)
    psig_neg_sep_df.insert(0, 'Separation', sep_list)
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
    one_probe_df = pd.concat([mean_one_probe_df, thr1_one_probe_df, thr2_one_probe_df],
                             ignore_index=True)
    one_probe_df.insert(0, 'dset', ['mean', 'group1', 'group2'])
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
                                            markersize=5, label='group1')
                        st1_last_val = mlines.Line2D([], [], color='tab:red',
                                                     linestyle="--", marker=None,
                                                     label='g1_last_val')
                        st2 = mlines.Line2D([], [], color='tab:blue',
                                            marker='o',
                                            markersize=5, label='group2')
                        st2_last_val = mlines.Line2D([], [], color='tab:blue',
                                                     linestyle="-.", marker=None,
                                                     label='g2_last_val')
                        ax.legend(handles=[st1, st1_last_val, st2, st2_last_val],
                                  fontsize=5, loc='lower right')

                else:
                    """use the psignifit values from each stair pair (e.g., 18, -18) to
                    get the mean threshold for each sep condition.
                    """
                    if verbose:
                        print("\nEighth plot")
                        print(f'psig_thr_mean_df:\n{psig_thr_mean_df}')
                        print(f'\none_probe_df:\n{one_probe_df}')

                    isi_thr_mean_df = pd.concat([psig_thr_mean_df['Separation'], psig_thr_mean_df[isi_name]],
                                                axis=1, keys=['Separation', isi_name])
                    if verbose:
                        print(f'isi_thr_mean_df:\n{isi_thr_mean_df}')

                    # line plot for thr1, th2 and mean thr
                    sns.lineplot(ax=axes[row_idx, col_idx], data=isi_thr_mean_df,
                                 x='Separation', y=isi_name, color='lightgreen',
                                 linewidth=3)

                    sns.lineplot(ax=axes[row_idx, col_idx], data=psig_pos_sep_df,
                                 x='Separation', y=isi_name, color='red',
                                 linestyle="--")
                    sns.lineplot(ax=axes[row_idx, col_idx], data=psig_neg_sep_df,
                                 x='Separation', y=isi_name, color='blue',
                                 linestyle="dotted")


                    # scatter plot for single probe conditions
                    sns.scatterplot(data=one_probe_df, x="x_val", y=isi_name,
                                    hue='dset',
                                    palette=['lightgreen', 'red', 'blue'])

                    # artist for legend
                    group1 = mlines.Line2D([], [], color='red',
                                             linestyle="--", marker=None,
                                             label='Congruent thr')

                    group2 = mlines.Line2D([], [], color='blue',
                                               linestyle="dotted", marker=None,
                                               label='Incongruent thr')
                    mean_thr = mlines.Line2D([], [], color='lightgreen',
                                                 linestyle="solid", marker=None,
                                                 label='mean thr')
                    ax.legend(handles=[group1, group2, mean_thr], fontsize=6,
                              loc='lower right')

                    # decorate plot
                    ax.set_title(f'{isi_name} psignifit thresholds')
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

    n_reversal_save_name = f'{save_path}{os.sep}n_reversals.csv'
    n_reversals_df.to_csv(n_reversal_save_name, index=False)

    print("\n***finished b3_plot_staircases()***\n")

####################
# all_data_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/Kim/' \
#                     'Nick_practice/P6a-Kim/RUNDATA-sorted.xlsx'
# b3_plot_staircase(all_data_path, thr_col='probeLum', resp_col='trial_response',
#                   show_plots=True, save_plots=True, verbose=True)

def c_plots(save_path, thr_col='probeLum', show_plots=True, verbose=True):

    """
    5. c_plots.m: uses psignifit_thresholds.csv and outputs plots.

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
    """

    print("\n*** running c_plots() ***\n")

    isi_name_list = ['Concurrent', 'ISI0', 'ISI2', 'ISI4', 'ISI6', 'ISI9', 'ISI12', 'ISI24']
    sym_sep_list = [-18, -6, -3, -2, -1, 0, 1, 2, 3, 6, 18, 20]
    sym_sep_tick_labels = [-18, -6, -3, -2, -1, 0, 1, 2, 3, 6, 18, '1\nprobe']
    pos_sep_list = [0, 1, 2, 3, 6, 18, 20]

    # load df mean of last n probeLum values (14 stairs x 8 isi).
    thr_csv_name = f'{save_path}{os.sep}psignifit_thresholds.csv'
    psig_thr_df = pd.read_csv(thr_csv_name)
    if verbose:
        print(f'psig_thr_df:\n{psig_thr_df}')

    psig_thr_df = psig_thr_df.drop(['stair'], axis=1)
    if 'group' in list(psig_thr_df.columns):
        group_col_s = psig_thr_df.pop('group')
    if 'Separation' in list(psig_thr_df.columns):
        sep_col_s = psig_thr_df.pop('Separation')
    psig_thr_df.columns = isi_name_list

    # lastN_pos_sym_np has values for 1-indexed stairs [1, 3, 5, 7, 9, 11, 9, 7, 5, 3, 1, 13]
    # but the rows I select are zero indexed, use rows [0, 2, 4, 6, 8, 10, 8, 6, 4, 2, 0, 12]
    # these correspond to separation values:          [18, 6, 3, 2, 1, 0, 1, 2, 3, 6, 18, 99]
    if verbose:
        print('\npreparing data for batman plots')
    pos_sym_indices = [0, 2, 4, 6, 8, 10, 8, 6, 4, 2, 0, 12]
    psig_pos_sep_df = psig_thr_df.iloc[pos_sym_indices]
    psig_pos_sep_df.reset_index(drop=True, inplace=True)

    # lastN_neg_sym_np has values for 1-indexed stairs   [2,  4,  6,  8, 10, 12, 10, 8,  6,  4,  2,  14]
    # but the rows I select are zero indexed, use rows   [1,  3,  5,  7, 9,  11, 9,  7,  5,  3,  1,  13]
    # these correspond to sep values:                  [-18, -6, -3, -2, -1, 0, -1, -2, -3, -6, -18, 99]
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
    psig_thr_mean_df.insert(0, 'Separation', sym_sep_list)
    psig_pos_sep_df.insert(0, 'Separation', sym_sep_list)
    psig_neg_sep_df.insert(0, 'Separation', sym_sep_list)

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

    # #  (figure2 doesn't exist in Martin's script - but I'll keep their numbers)

    # add columns back in to split thr_df by group for data and divby1probe plots.
    if 'group' not in list(psig_thr_df.columns):
        psig_thr_df.insert(0, 'group', group_col_s)
    if 'Separation' not in list(psig_thr_df.columns):
        sep_list = [19 if i == 20 else i for i in sep_col_s.tolist()]
        psig_thr_df.insert(1, 'Separation', sep_list)
    print(f'\nmaking plots from group1 and group2 data with psig_thr_df:\n{psig_thr_df}')

    for group in [1, 2]:

        group_plot_df = psig_thr_df[psig_thr_df['group'] == group]
        group_plot_df = group_plot_df.drop(['group'], axis=1)
        print(f'\nrunning group{group} with group_plot_df:\n{group_plot_df}')

        # # FIGURE3 - 'data-{n}lastValues.png' - all ISIs on same axis, pos sep only, plus single
        # # use plot_pos_sep_and_one_probe()
        fig3_save_name = f'data.png'
        fig_3_title = 'All ISIs and separations'

        if group is not None:
            fig3_save_name = f'g{group}_data.png'
            fig_3_title = f'g{group} All ISIs and separations'

        plot_pos_sep_and_one_probe(pos_sep_and_one_probe_df=group_plot_df,
                                   thr_col=thr_col,
                                   fig_title=fig_3_title,
                                   one_probe=True,
                                   save_path=save_path,
                                   save_name=fig3_save_name,
                                   pos_set_ticks=[0, 1, 2, 3, 6, 18, 19],
                                   verbose=True)
        if show_plots:
            plt.show()
        plt.close()

        # # # FIGURE4 - 'dataDivOneProbe-{n}lastValues.png' - all ISIs on same axis, pos sep only.
        #         # does not include single probe
        # # # use plot_pos_sep_and_one_probe(one_probe=False)
        # # each sep row in pos_sep_df is divided by one_probe_df.
        fig4_save_name = f'dataDivOneProbe.png'
        fig4_title = f'two-probe conditions divided by one-probe conditions'

        if group is not None:
            fig4_save_name = f'g{group}_dataDivOneProbe.png'
            fig4_title = f'g{group} two-probe conditions divided by one-probe conditions'

        pos_sep_df, one_probe_df = split_df_into_pos_sep_df_and_one_probe_df(group_plot_df)
        pos_sep_arr = pos_sep_df.to_numpy()
        one_probe_arr = one_probe_df['probeLum'].to_numpy()
        div_by_one_probe_arr = (pos_sep_arr.T / one_probe_arr[:, None]).T
        div_by_one_probe_df = pd.DataFrame(div_by_one_probe_arr, columns=isi_name_list)
        div_by_one_probe_df.insert(0, 'Separation', pos_sep_list[:-1])
        div_by_one_probe_df.set_index('Separation', inplace=True)
        print(f'div_by_one_probe_df:\n{div_by_one_probe_df}')

        plot_pos_sep_and_one_probe(div_by_one_probe_df,
                                   thr_col='probeLum',
                                   fig_title=fig4_title,
                                   one_probe=False,
                                   save_path=save_path,
                                   save_name=fig4_save_name,
                                   pos_set_ticks=[0, 1, 2, 3, 6, 18, 19],
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
                          error_bars=None,
                          trim_n=None,
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
    these both have two versions:
    a. x-axis is separation, ISI as different lines
    b. x-axis is ISI, Separation as different lines
    Heatmap: with average probe lum for ISI and Separation.

    :param root_path: dir containing run folders
    :param run_dir_names_list: names of run folders
    :param error_bars: Default: None. Can pass sd or se for standard deviation or error.
    :param trim_n: default None.  If int is passed, will call function trim_n_high_n_low(),
            which trims the n highest and lowest values.
    :param show_plots: default True - display plots
    :param verbose: Defaut true, print progress to screen

    :returns: ave_psignifit_thr_df: (trimmed?) mean threshold for each separation and ISI.
    """

    print("\n***running d_average_participant()***\n")

    """ part1. Munge data, save master lists and get means etc
     - loop through runs and get each P-next-thresholds and P-reversal4-thresholds
    Make master sheets: MASTER_next_thresh & MASTER_reversal_4_thresh
    Incidentally the MATLAB script didn't specify which reversals data to use,
    although the figures imply Martin used last3 reversals."""

    isi_name_list = ['Concurrent', 'ISI0', 'ISI2', 'ISI4',
                     'ISI6', 'ISI9', 'ISI12', 'ISI24']
    pos_sep_list = [0, 1, 2, 3, 6, 18, 20]

    all_psignifit_list = []
    for run_idx, run_name in enumerate(run_dir_names_list):

        this_psignifit_df = pd.read_csv(f'{root_path}{os.sep}{run_name}{os.sep}psignifit_thresholds.csv')
        print(f'{run_idx}. {run_name} - this_psignifit_df:\n{this_psignifit_df}')

        if 'Unnamed: 0' in list(this_psignifit_df):
            this_psignifit_df.drop('Unnamed: 0', axis=1, inplace=True)

        this_psignifit_df.drop(columns='stair', inplace=True)

        # split df into group1 and group2
        psig_g1_df = this_psignifit_df[this_psignifit_df['group'] == 1]
        psig_g1_df.drop(columns='group', inplace=True)
        rows, cols = psig_g1_df.shape
        psig_g1_df.insert(0, 'stack', [run_idx*2] * rows)

        psig_g2_df = this_psignifit_df[this_psignifit_df['group'] == 2]
        psig_g2_df.drop(columns='group', inplace=True)
        psig_g2_df.insert(0, 'stack', [run_idx*2+1] * rows)

        columns_list = ['stack', 'Separation'] + isi_name_list
        psig_g1_df.columns = columns_list
        psig_g2_df.columns = columns_list

        if verbose:
            print(f'\npsig_g1_df:\n{psig_g1_df}')
            print(f'\npsig_g2_df:\n{psig_g2_df}')

        all_psignifit_list.append(psig_g1_df)
        all_psignifit_list.append(psig_g2_df)

    # join all stacks (run/group) data and save as master csv
    all_data_psignifit_df = pd.concat(all_psignifit_list, ignore_index=True)
    all_data_psignifit_df.to_csv(f'{root_path}{os.sep}MASTER_psignifit_thresholds.csv', index=False)
    if verbose:
        print(f'\nall_data_psignifit_df:\n{all_data_psignifit_df}')

    """Part 2: trim highest and lowest values is required and get average vals and errors"""
    # # trim highest and lowest values
    if trim_n is not None:
        trimmed_df = trim_n_high_n_low(all_data_psignifit_df, trim_from_ends=trim_n,
                                       reference_col='Separation',
                                       stack_col_id='stack',
                                       verbose=verbose)
        get_means_df = trimmed_df
    else:
        get_means_df = all_data_psignifit_df

    # # get means and errors
    groupby_sep_df = get_means_df.drop('stack', axis=1)
    ave_psignifit_thr_df = groupby_sep_df.groupby('Separation', sort=True).mean()
    if verbose:
        print(f'\nave_psignifit_thr_df:\n{ave_psignifit_thr_df}')

    if error_bars in [False, None]:
        error_bars_df = None
    elif error_bars.lower() in ['se', 'error', 'std-error', 'standard error', 'standard_error']:
        error_bars_df = groupby_sep_df.groupby('Separation', sort=True).sem()
    elif error_bars.lower() in ['sd', 'stdev', 'std_dev', 'std.dev', 'deviation', 'standard_deviation']:
        error_bars_df = groupby_sep_df.groupby('Separation', sort=True).std()
    else:
        raise ValueError(f"error_bars should be in:\nfor none: [False, None]\n"
                         f"for standard error: ['se', 'error', 'std-error', 'standard error', 'standrad_error']\n"
                         f"for standard deviation: ['sd', 'stdev', 'std_dev', 'std.dev', "
                         f"'deviation', 'standard_deviation']")
    print(f'\nerror_bars_df: ({error_bars})\n{error_bars_df}')

    # save csv with average values
    if trim_n is not None:
        ave_psignifit_thr_df.to_csv(f'{root_path}{os.sep}MASTER_ave_TM_thresh.csv', index=False)
    else:
        ave_psignifit_thr_df.to_csv(f'{root_path}{os.sep}MASTER_ave_thresh.csv', index=False)


    """part 3. main Figures (these are the ones saved in the matlab script)
    Fig1: plot average threshold for each ISI and sep.
    Fig2: divide all 2probe conds (pos_sep) by one_probe condition for each stack.
    For both figures there are 2 versions:
        a) Sep on x-axis, different line for each ISI
        b) ISI on x-axis, different line for each Sep"""

    print(f"\nfig_1a\n")
    if trim_n is not None:
        fig1_title = f'Average thresholds across all runs (trim={trim_n}).'
        fig1_savename = f'ave_TM_thr_all_runs.png'
    else:
        fig1_title = f'Average threshold across all runs'
        fig1_savename = f'ave_thr_all_runs.png'

    fig1a = plot_1probe_w_errors(fig_df=ave_psignifit_thr_df, error_df=error_bars_df,
                                 split_1probe=True, jitter=True,
                                 error_caps=True, alt_colours=False,
                                 legend_names=['Concurrent', 'ISI 0', 'ISI 2', 'ISI 4',
                                               'ISI 6', 'ISI 9', 'ISI 12', 'ISI 24'],
                                 x_tick_vals=[0, 1, 2, 3, 6, 18, 20],
                                 x_tick_labels=[0, 1, 2, 3, 6, 18, '1probe'],
                                 fixed_y_range=False,
                                 fig_title=fig1_title, save_name=fig1_savename,
                                 save_path=root_path, verbose=True)
    if show_plots:
        plt.show()
    plt.close()

    if verbose:
        print('finished fig1a')

    print(f"\nfig_1b\n")
    # fig 1b, ISI on x-axis, different line for each sep
    if trim_n is not None:
        fig1b_title = f'Probe luminance at each ISI value per Separation (trim={trim_n}).'
        fig1b_savename = f'ave_TM_thr_all_runs_transpose.png'
    else:
        fig1b_title = f'Probe luminance at each ISI value per Separation'
        fig1b_savename = f'ave_thr_all_runs_transpose.png'

    fig1b = plot_w_errors_no_1probe(wide_df=get_means_df,
                                    x_var='ISI', y_var='probeLum',
                                    lines_var='Separation',
                                    legend_names=['0', '1', '2', '3', '6', '18', '1probe'],
                                    x_tick_labels=['conc', 0, 2, 4, 6, 9, 12, 24],
                                    alt_colours=True,
                                    fixed_y_range=False,
                                    jitter=True,
                                    error_caps=True,
                                    fig1b_title=fig1b_title,
                                    fig1b_savename=fig1b_savename,
                                    save_path=root_path,
                                    verbose=True)
    if show_plots:
        plt.show()
    plt.close()

    if verbose:
        print('finished fig1b')

    print(f"\nfig_2a\n")
    # # Fig 2  - divide all 2probe conditions (pos_sep) by one_probe for each stack
    # use get_means_df with all (or trimmed) data.
    # first split each stack into 1probe and pos_sep (2probe), divide and make back into long df
    stack_list = list(get_means_df['stack'].unique())
    print(f'stack_list: {stack_list}')

    divided_list = []
    # loop through stacks
    for stack in stack_list:
        # for each stack_df, split into pos_sep_df and one_probe_df
        stack_df = get_means_df[get_means_df['stack'] == stack]
        stack_df = stack_df.drop('stack', axis=1)
        pos_sep_df, one_probe_df = split_df_into_pos_sep_df_and_one_probe_df(stack_df)

        # divide pos_sep by one_probe and append to list
        pos_sep_arr = pos_sep_df.to_numpy()
        one_probe_arr = one_probe_df['probeLum'].to_numpy()
        div_by_one_probe_arr = (pos_sep_arr.T / one_probe_arr[:, None]).T
        div_by_one_probe_df = pd.DataFrame(div_by_one_probe_arr, columns=isi_name_list)
        div_by_one_probe_df.insert(0, 'stack', [stack] * len(div_by_one_probe_df))
        div_by_one_probe_df.insert(1, 'Separation', pos_sep_list[:-1])
        divided_list.append(div_by_one_probe_df)

    # put back into long form df with stack Sep cols
    divided_df = pd.concat(divided_list)
    print(f'divided_df:\n{divided_df}')

    # # get means and errors
    div_groupby_sep_df = divided_df.drop('stack', axis=1)
    div_ave_psignifit_thr_df = div_groupby_sep_df.groupby('Separation', sort=True).mean()
    if verbose:
        print(f'\ndiv_ave_psignifit_thr_df:\n{div_ave_psignifit_thr_df}')

    if error_bars in [False, None]:
        div_error_bars_df = None
    elif error_bars.lower() in ['se', 'error', 'std-error', 'standard error', 'standard_error']:
        div_error_bars_df = div_groupby_sep_df.groupby('Separation', sort=True).sem()
    elif error_bars.lower() in ['sd', 'stdev', 'std_dev', 'std.dev', 'deviation', 'standard_deviation']:
        div_error_bars_df = div_groupby_sep_df.groupby('Separation', sort=True).std()
    print(f'\ndiv_error_bars_df: ({error_bars})\n{div_error_bars_df}')

    if trim_n is not None:
        fig2a_save_name = 'ave_TM_thr_div_one_probe.png'
        fig2a_title = f'Average thresholds divided by single probe (trim={trim_n}).'
    else:
        fig2a_save_name = 'ave_thr_div_one_probe.png'
        fig2a_title = 'Average threshold divided by single probe'

    fig2a = plot_1probe_w_errors(fig_df=div_ave_psignifit_thr_df, error_df=div_error_bars_df,
                                 split_1probe=False, jitter=True,
                                 error_caps=True, alt_colours=False,
                                 legend_names=['Concurrent', 'ISI 0', 'ISI 2', 'ISI 4',
                                               'ISI 6', 'ISI 9', 'ISI 12', 'ISI 24'],
                                 x_tick_vals=[0, 1, 2, 3, 6, 18],
                                 x_tick_labels=[0, 1, 2, 3, 6, 18],
                                 fixed_y_range=False,
                                 fig_title=fig2a_title, save_name=fig2a_save_name,
                                 save_path=root_path, verbose=True)
    if show_plots:
        plt.show()
    plt.close()

    if verbose:
        print('finished fig2a')

    print(f"\nfig_2b\n")
    # fig 2b, ISI on x-axis, different line for each sep
    if trim_n is not None:
        fig2b_save_name = 'ave_TM_thr_div_one_probe_transpose.png'
        fig2b_title = f'Average thresholds divided by single probe at each ISI (trim={trim_n}).'
    else:
        fig2b_save_name = 'ave_thr_div_one_probe_transpose.png'
        fig2b_title = 'Average thresholds divided by single probe at each ISI'

    fig2b = plot_w_errors_no_1probe(wide_df=divided_df,
                                    x_var='ISI', y_var='probeLum',
                                    lines_var='Separation',
                                    legend_names=['0', '1', '2', '3', '6', '18'],
                                    x_tick_labels=['conc', 0, 2, 4, 6, 9, 12, 24],
                                    alt_colours=True,
                                    fixed_y_range=False,
                                    jitter=True,
                                    error_caps=True,
                                    fig1b_title=fig2b_title,
                                    fig1b_savename=fig2b_save_name,
                                    save_path=root_path,
                                    verbose=True)
    if show_plots:
        plt.show()
    plt.close()

    if verbose:
        print('finished fig2b')

    print(f"\nHeatmap\n")
    # get mean of each col, then mean of that
    x_tick_labels = ['conc', 'isi 0', 'isi 2', 'isi 4', 'isi 6', 'isi 9', 'isi12', 'isi 24']
    y_tick_labels = [0, 1, 2, 3, 6, 18, '1probe']
    if trim_n is not None:
        heatmap_title = f'Mean Threshold for each ISI and separation (trim={trim_n}).'
        heatmap_savename = 'mean_TM_thr_heatmap'
    else:
        heatmap_title = 'Mean Threshold for each ISI and separation'
        heatmap_savename = 'mean_thr_heatmap'

    heatmap = plot_thr_heatmap(heatmap_df=ave_psignifit_thr_df,
                               x_tick_labels=x_tick_labels,
                               y_tick_labels=y_tick_labels,
                               fig_title=heatmap_title,
                               save_name=heatmap_savename,
                               save_path=root_path,
                               verbose=True)
    plt.show()

    print("\n*** finished d_average_participant()***\n")

    return ave_psignifit_thr_df

#######
# root_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/Kim'  # master folder containing all runs
# run_dir_names_list = ['P6a-Kim', 'P6b-Kim', 'P6c-Kim', 'P6d-Kim', 'P6e-Kim', 'P6f-Kim']
# d_average_participant(root_path=root_path, run_dir_names_list=run_dir_names_list)
