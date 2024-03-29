import os
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm

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

split_df_into_pos_sep_df_and_1probe_df():
    code to turn array into symmetrical array (e.g., from sep=[0, 1, 2, 3, 6, 18] 
    into sep=[-18, -6, -3, -2, -1, 0, 1, 2, 3, 6, 18])
"""

pd.options.display.float_format = "{:,.2f}".format


def lookup_p_name(p_name):
    """lookup participant name in old_p_list and return new_p_list name or vice versa"""
    old_p_list = ['Tony', 'Simon', 'Maria', 'Kristian', 'Kim', 'Nick']
    new_p_list = ['aa', 'bb', 'cc', 'dd', 'ee', 'ff']

    if p_name in old_p_list:
        p_idx = old_p_list.index(p_name)
        return_name = new_p_list[p_idx]
    elif p_name in new_p_list:
        p_idx = new_p_list.index(p_name)
        return_name = old_p_list[p_idx]
    else:
        raise ValueError(f'{p_name} not found in either old_p_list or new_p_list'
                         f'\nold_p_list: {old_p_list}'
                         f'\nnew_p_list: {new_p_list}')

    return return_name


def conc_to_first_isi_col(df, col_to_change='ISI_-1'):
    """
    Function to sort dataframe where concurrent is given as 'ISI_-1'.
    This can appear as the last ISI column instead of first.

    This simple function won't work if ISI columns aren't at the end,
    and will require a bit more sophistication
    (e.g., extract ISI columns first, then check if ISI_-1 is last)

    :param df: dataframe to be tested and sorted if necessary.
    :return: dataframe - which has been sorted if needed"""

    if df.columns.tolist()[-1] == col_to_change:
        col_list = df.columns.tolist()
        other_cols = [i for i in col_list if 'ISI' not in i]
        isi_cols = [i for i in col_list if 'ISI' in i]
        new_cols_list = other_cols + isi_cols[-1:] + isi_cols[:-1]
        out_df = df.reindex(columns=new_cols_list)
        print(f"Concurrent column moved to start\n{out_df}")
    else:
        out_df = df

    return out_df


# def split_df_alternate_rows(df):
#     """
#     my original version of this function
#     Split a dataframe into alternate rows.  Dataframes are organized by
#     stair conditions relating to separations
#     (e.g., in order 18, -18, 6, -6, 3, -3, 2, -2, 1, -1, 0, 0, 99, -99).
#     For some plots this needs to be split into two dfs, e.g., :
#     pos_sep_df: [18, 6, 3, 2, 1, 0, 99]
#     neg_sep_df: [-18, -6, -3, -2, -1, 0, -99]
#
#     :param df: Dataframe to be split in two
#     :return: two dataframes: pos_sep_df, neg_sep_df
#     """
#     print("\n*** running split_df_alternate_rows() ***")
#
#     n_rows, n_cols = df.shape
#     pos_nums = list(range(0, n_rows, 2))
#     neg_nums = list(range(1, n_rows, 2))
#     pos_sep_df = df.iloc[pos_nums, :]
#     neg_sep_df = df.iloc[neg_nums, :]
#     pos_sep_df.reset_index(drop=True, inplace=True)
#     neg_sep_df.reset_index(drop=True, inplace=True)
#
#     return pos_sep_df, neg_sep_df
# refactor this code to be shorter and more efficient
def split_df_alternate_rows(df):
    """
    Co-pilot version of this function
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
    pos_sep_df = pd.DataFrame(columns=df.columns)
    neg_sep_df = pd.DataFrame(columns=df.columns)
    for i in range(n_rows):
        if i % 2 == 0:
            pos_sep_df = pos_sep_df.append(df.iloc[i, :])
        else:
            neg_sep_df = neg_sep_df.append(df.iloc[i, :])
    pos_sep_df.reset_index(drop=True, inplace=True)
    neg_sep_df.reset_index(drop=True, inplace=True)

    return pos_sep_df, neg_sep_df

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


def split_df_into_pos_sep_df_and_1probe_df(pos_sep_and_1probe_df,
                                           thr_col='newLum',
                                           isi_name_list=None,
                                           one_probe_pos=None,
                                           even_spaced_x=False,
                                           verbose=True):
    """
    For plots where positive separations are shown as line plots and 
    one probe results are shown as scatter plot, this function splits the dataframe into two.
    
    :param pos_sep_and_1probe_df: Dataframe of positive separations with
        one_probe conds at bottom of df (e.g., shown as 20 or 99).  df must be indexed with the separation column.
    :param thr_col: Column to extract threshold values from - expects either probeLum or newLum.
    :param isi_name_list: List of isi names.  If None, will use default values.
    :param one_probe_pos: Default=None, use value from df.  Where to set location of 1probes on x-axis.
    :param even_spaced_x: Default=False, use value from df.  if True use numbered values for x.
    :param verbose: whether to print progress info to screen

    :return: Pos_sel_df: same as input df but with last row removed
            One_probe_df: constructed from last row of input df 
    """

    data_df = pos_sep_and_1probe_df

    if verbose:
        print("\n*** running split_df_into_pos_sep_df_and_1probe_df() ***")

    if isi_name_list is None:
        # todo: change to use isi_name_list from df
        isi_name_list = ['ISI_-1', 'ISI_0', 'ISI_2', 'ISI_4',
                         'ISI_6', 'ISI_9', 'ISI_12', 'ISI_24']

    # check that the df only contains positive separation values
    if 'sep' in list(data_df.columns):
        data_df = data_df.rename(columns={'sep': 'separation'})

    # check if index column is set as 'separation'
    if data_df.index.name is None:
        data_df = data_df.set_index('separation')

    # replace 1pr with 20 if string is given
    index_values = data_df.index.tolist()
    print(f'index_values: {index_values}')
    # for idx in index_values:
    #     print(type(idx))

    if '1Probe' in index_values:
        data_df.rename(index={'1Probe': 20}, inplace=True)
    elif '1probe' in index_values:
        data_df.rename(index={'1probe': 20}, inplace=True)
    elif '1pr' in index_values:
        data_df.rename(index={'1pr': 20}, inplace=True)
    data_df.index = data_df.index.astype(int)

    # index_values = data_df.index.tolist()
    # print(f'index_values:\n{index_values}')
    # for idx in index_values:
    #     print(type(idx))
    # if verbose:
    #     print(f'data_df:\n{data_df}')

    # data_df = data_df.loc[data_df['separation'] >= 0]
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
                      f'{thr_col}': one_probe_lum_list,
                      'x_vals': [one_probe_pos for _ in isi_name_list]}
    one_probe_df = pd.DataFrame.from_dict(one_probe_dict)
    if verbose:
        print(f'one_probe_df:\n{one_probe_df}')

    if even_spaced_x:
        pos_sep_df.reset_index(inplace=True)
        # print(f'list(range(len(index_values)))[:-1]: {list(range(len(index_values)))[:-1]}')
        pos_sep_df['separation'] = list(range(len(index_values)))[:-1]
        pos_sep_df = pos_sep_df.set_index('separation')

        # pos_sep_df.index = list(range(len(index_values)))[:-1]
        one_probe_df['x_vals'] = list(range(len(index_values)))[-1]
        print(f'pos_sep_df:\n{pos_sep_df}')
        print(f'one_probe_df:\n{one_probe_df}')

    print("\n*** finished split_df_into_pos_sep_df_and_1probe_df() ***")

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

    if n_conditions > 20:
        use_colour = 'spectral'
    elif 10 < n_conditions < 21:
        use_colours = 'tab20'


    use_cmap = False

    my_colours = sns.color_palette(palette=use_colours, n_colors=n_conditions, as_cmap=use_cmap)
    sns.set_palette(palette=use_colours, n_colors=n_conditions)

    return my_colours


def get_n_rows_n_cols(n_plots):
    """
    Function to get the optimal configuration of subplot (upto n=25).
    Ideally plots will be in a square arrangement, or in a rectangle.
    Start by adding columns to each row upto 4 columns, then add rows.

    It doesn't like to loop through rows and axes separately if there is only one row.
    So plan to put the master plot on 2nd row if there are less than 3 sep values.

    :param n_plots: number of plots
    :return: n_rows, n_cols
    """

    if n_plots > 25:
        raise ValueError(f"\t\tToo many plots for this function: {n_plots}\n\n")

    # ideally have no more than 4 rows, unless more than 16 plots
    if n_plots <= 4:
        row_whole_divide = n_plots // 2  # how many times this number of plots goes into 4.
        row_remainder = n_plots % 2  # remainder after the above calculation.
        if row_remainder == 0:
            n_rows = row_whole_divide
        else:
            n_rows = row_whole_divide + 1
    elif n_plots <= 16:
        row_whole_divide = n_plots // 4  # how many times this number of plots goes into 4.
        row_remainder = n_plots % 4  # remainder after the above calculation.
        if row_remainder == 0:
            n_rows = row_whole_divide
        else:
            n_rows = row_whole_divide + 1
    else:
        n_rows = 5

    # ideally have no more than 4 cols, unless more than 20 plots
    col_whole_divide = n_plots // n_rows  # how many times this number of plots goes into n_rows.
    col_remainder = n_plots % n_rows  # remainder after the above calculation.
    if col_remainder == 0:
        n_cols = col_whole_divide
    else:
        n_cols = col_whole_divide + 1

    return n_rows, n_cols


def trim_n_high_n_low(all_data_df, trim_from_ends=None, reference_col='separation',
                      stack_col_id='stack', verbose=True):
    """
    Function for trimming the n highest and lowest values from each condition of a set with multiple runs.

    :param all_data_df: Dataset to be trimmed.
    :param trim_from_ends: number of values to trim from each end of the distribution.
    :param reference_col: Idx column containing repeated conditions (e.g., separation has same label for each stack).
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

    if trim_from_ends == 0:
        trim_from_ends = None
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
              f'2d array shape (after trim, but before separation, stack or headers are added): '
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
            row_list.append(trimmed)
            counter += 1
        trimmed_3d_list.append(row_list)

    """
    Part 3, turn 3d numpy back into 2d df.
    trimmed_3d_list is a list of arrays (e.g., 3d).  Each array relates to a
    depth-stack of my_3d_array which has now be trimmed (e.g., fewer rows).
    However, trimmed_3d_list has the same depth and number of columns as my_3d_array.
    trimmed_array re-shapes trimmed_3d_list so all values are in their original
    row and column positions (e.g., separation and ISI).
    However, the 3rd dimension (depth) is not in original order, but in ascending order."""

    trimmed_3d_array = np.array(trimmed_3d_list)
    if verbose:
        print(f'\n\nReshaping trimmed data\ntrimmed_3d_array: {np.shape(trimmed_3d_array)}')
        print(trimmed_3d_array)

    ravel_array_f = np.ravel(trimmed_3d_array, order='F')
    if verbose:
        print(f'\n1. ravel_array_f: {np.shape(ravel_array_f)}')
        print(ravel_array_f)

    reshaped_3d_array = ravel_array_f.reshape(target_3d_depth, rows_3d, cols_all)
    if verbose:
        print(f'\n2. reshaped_3d_array: {np.shape(reshaped_3d_array)}')
        print(reshaped_3d_array)

    reshaped_2d_array = reshaped_3d_array.reshape(target_2d_rows, -1)
    if verbose:
        print(f'\n3. reshaped_2d_array {np.shape(reshaped_2d_array)}')
        print(reshaped_2d_array)

    # make dataframe and insert column for separation and stack (trimmed run/group)
    trimmed_df = pd.DataFrame(reshaped_2d_array, columns=isi_name_list)
    stack_col_vals = np.repeat(np.arange(target_3d_depth), rows_3d)
    sep_col_vals = sep_list*target_3d_depth
    trimmed_df.insert(0, 'stack', stack_col_vals)
    trimmed_df.insert(1, reference_col, sep_col_vals)

    total_trimmed = 0
    if trim_from_ends is not None:
        total_trimmed = 2*trim_from_ends
    if verbose:
        print(f'\ntrimmed_df {trimmed_df.shape}:\n{trimmed_df}')
        print(f'trimmed {trim_from_ends} highest and lowest values ({total_trimmed} in total) from each of the '
              f'{datapoints_per_cond} datapoints so there are now '
              f'{target_3d_depth} datapoints for each of the '
              f'{rows_3d} x {cols_all} conditions.')

    print('\n*** finished trim_high_n_low() ***')

    return trimmed_df


##################

def axis_break(axis, xpos, subplot=False):
    """
    Add a break // to an axis (e.g., between separation values and 1Probe).

    To use it the code is:
        axis_break(ax, xpos=[0.1, 0.12])

    :param axis: the fig ax to add break to (e.g., the "ax" in: fig, ax = plt.subplots())
    :param xpos: the position of the break (between 0 and 1) as a list.
    :param subplot: True if the axis is a subplot (multiplot grid) and behaves slightly diferently
    """

    slant = 1.5  # the slant of the break markers (proportion of vertical to horizontal extent)
    anchor = (xpos[0], -1)
    w = xpos[1] - xpos[0]

    print(f"axis: {axis}, xpos: {xpos}, anchor: {anchor}, w: {w}")

    if subplot:

        h = 10  # default height was 1, but changing this stops multiplots from changing shape (height).
        zorder_1 = 20  # changing the zorder seems to help with the break line being hidden by the axis for mulitplots
    else:
        h = 1
        zorder_1 = 3

    kwargs = dict(marker=[(-1, -slant), (1, slant)], markersize=12, zorder=zorder_1,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    axis.add_patch(Rectangle(anchor, w, h, fill=True, color="white",
                             transform=axis.transAxes, clip_on=False,
                             zorder=3))
    axis.plot(xpos, [0, 0], transform=axis.transAxes, **kwargs)



def break_axis_before_label(ax, label='1Probe', label_list=None, subplot=False):
    """
    Get the axis values for the break before the given label and call axis_break()

    :param ax: The axis to break (e.g., the "ax" in: fig, ax = plt.subplots())
    :param label: The label to break before (e.g., '1Probe')
    :param label_list: List of labels to use to get the axis positions
    :param subplot: True if the axis is a subplot (multiplot grid) and behaves slightly diferently
    """

    # get index of label_list that matches the given label
    if label in label_list:
        label_idx = label_list.index(label)
        print(f'label_idx: {label_idx}')
    else:
        raise ValueError(f'Label {label} not in label_list {label_list}')

    # get the tick locations (between 0 and 1)
    x_min, x_max = ax.get_xlim()
    tick_pos_list = [(tick - x_min) / (x_max - x_min) for tick in ax.get_xticks()]
    print(f'tick_pos_list: {tick_pos_list}')
    
    # distance between key ticks
    tick_dist = tick_pos_list[label_idx] - tick_pos_list[label_idx-1]
    print(f'tick_dist: {tick_dist}')

    # slash_midpoint position
    slash_midpoint = tick_pos_list[label_idx] - (tick_dist / 2)
    print(f'slash_midpoint: {slash_midpoint}')
    
    # distance between the slashes
    slash_dist = tick_dist / 20
    print(f'slash_dist: {slash_dist}')

    # the axis values to use when calling axis_break
    break_ax_vals = [slash_midpoint - slash_dist, slash_midpoint + slash_dist]
    print(f"break_ax_vals: {break_ax_vals}")

    # call axis_break to make the change
    axis_break(ax, xpos=break_ax_vals, subplot=subplot)




###################

def make_long_df(wide_df, wide_stubnames='ISI', thr_col='newLum',
                 col_to_keep='separation', idx_col='Run', verbose=True):

    """
    Function to convert a wide form df containing multiple measurements at each value
    (e.g., data dfs concatenated from several runs), into long-form dataframe
    :param wide_df: Expects a dataframe made by concatenating data from several runs.
        Dataframe expected to contain columns for: Run, separation and ISI levels.
    :param wide_stubnames: repeated prefix for several columns (e.g., ISI0, ISI1, ISI2, ISI4 etc).
    :param thr_col: Column to extract threshold values from - expects either probeLum or newLum.
    :param col_to_keep: Existing columns of useful data (e.g., separation)
    :param idx_col: Existing column of irrelevant data to use as index (e.g., Run)
    :param verbose: print progress to screen.

    :return: Long form dataframe with single index
    """
    print('\n*** running make_long_df() ***\n')

    if verbose:
        print(f'wide_df:\n{wide_df}')

    # add spaces to ISI names and change concurrent to 999.
    orig_col_names = list(wide_df.columns)

    # sometimes col_names have spaces, sometimes underscores - these lines should catch either
    if 'ISI_' in orig_col_names[-1]:
        new_col_names = [f"ISI {i.strip('ISI_')}" if 'ISI_' in i else i for i in orig_col_names]
    elif 'ISI ' in orig_col_names[-1]:
            new_col_names = [f"ISI {i.strip('ISI ')}" if 'ISI ' in i else i for i in orig_col_names]
    else:
        new_col_names = [f"ISI {i.strip('ISI')}" if 'ISI' in i else i for i in orig_col_names]

    # change 'concurrent' to 999 not -1 as wide_to_long won't take negative numbers
    new_col_names = [f"ISI 999" if i in ['Concurrent', 'Conc', 'ISI_-1', 'ISI -1'] else i for i in new_col_names]

    print(f'orig_col_names:\n{orig_col_names}')
    print(f'new_col_names:\n{new_col_names}')

    wide_df.columns = new_col_names
    print(f'wide_df:\n{wide_df}')

    # todo: just trying restting index to see if that helps: KeyError: "None of [Index(['stack', 'separation'], dtype='object')] are in the [columns]"
    wide_df = wide_df.reset_index()
    print(f'wide_df (reset index):\n{wide_df}')

    print("\n\nidiot check")
    print(f"wide_stubnames: {wide_stubnames}")
    print(f"idx_col: {idx_col}")
    print(f"col_to_keep: {col_to_keep}")


    # use pandas wide_to_long for transform df
    long_df = pd.wide_to_long(wide_df, stubnames=wide_stubnames, i=[idx_col, col_to_keep], j='data',
                              sep=' ')
    if verbose:
        print(f'long_df:\n{long_df}')

    # # replace column values and labels
    long_df = long_df.rename({wide_stubnames: f'{thr_col}'}, axis='columns')
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


def plot_pos_sep_and_1probe(pos_sep_and_1probe_df,
                            thr_col='newLum',
                            fig_title=None,
                            one_probe=True,
                            save_path=None,
                            save_name=None,
                            isi_name_list=None,
                            pos_set_ticks=None,
                            pos_tick_labels=None,
                            error_bars_df=None,
                            ra_cd_v_line=None,
                            y_axis_label=None,
                            verbose=True):
    """
    This plots a figure with one axis, x has separation values [-2, -1, 0, 1, 2, 3, 6, 18],
    where -2 is not uses, -1 is for the single probe condition - shows as a scatter plot.
    Values sep (0:18) are shown as lineplots.
    Will plot all ISIs on the same axis.

    :param pos_sep_and_1probe_df: Full dataframe to use for values
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
        print("\n*** running plot_pos_sep_and_1probe() ***")
        # print(f'pos_sep_and_1probe_df:\n{pos_sep_and_1probe_df}')

    if isi_name_list is None:
        isi_name_list = ['Conc', 'ISI0', 'ISI2', 'ISI4',
                         'ISI6', 'ISI9', 'ISI12', 'ISI24']
        if verbose:
            print(f'isi_name_list: {isi_name_list}')

    if pos_set_ticks is None:
        pos_set_ticks = [0, 1, 2, 3, 6, 18, 20]
    if pos_tick_labels is None:
        pos_tick_labels = [0, 1, 2, 3, 6, 18, 'one\nprobe']

    # call function to split df into pos_sep_df and one_probe_df
    if one_probe:
        pos_sep_df, one_probe_df = split_df_into_pos_sep_df_and_1probe_df(
            pos_sep_and_1probe_df=pos_sep_and_1probe_df, isi_name_list=isi_name_list)
        if verbose:
            print(f'pos_sep_df:\n{pos_sep_df}\none_probe_df:\n{one_probe_df}')
    else:
        pos_sep_df = pos_sep_and_1probe_df

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


    # add vertical grey dashed line for Ricco's area or Bloch critical duration.
    if ra_cd_v_line:
        plt.axvline(x=ra_cd_v_line, linestyle='--', color='lightgrey')

    # decorate plot
    # todo: fix this so all handles are either lines or markers (currently every other one is a rectangle)

    ax.legend(labels=isi_name_list,
              title='ISI',
              shadow=True,
              # place lower left corner of legend at specified location.
              loc='lower left', bbox_to_anchor=(0.96, 0.5))

    print("idiot check")
    print(f"pos_set_ticks: {pos_set_ticks}")
    print(f"pos_tick_labels: {pos_tick_labels}")
    if one_probe:
        ax.set_xticks(pos_set_ticks)
        ax.set_xticklabels(pos_tick_labels)
    else:
        ax.set_xticks(pos_set_ticks[:-1])
        ax.set_xticklabels(pos_tick_labels[:-1])

    # ax.set_ylim([0, 110])
    ax.set_xlabel('Probe separation in diagonal pixels')

    if not y_axis_label:
        y_axis_label = 'Probe Luminance'
    ax.set_ylabel(y_axis_label)

    if fig_title is not None:
        plt.title(fig_title)
        
    if save_path is not None:
        if save_name is not None:
            plt.savefig(os.path.join(save_path, save_name))

    return fig

####################

def plot_1probe_w_errors(fig_df, error_df,
                         thr_col='newLum', split_1probe=True,
                         jitter=True, error_caps=False, alt_colours=False,
                         legend_names=None,
                         x_tick_vals=None,
                         x_tick_labels=None,
                         fixed_y_range=False,
                         even_spaced_x=False,
                         ra_cd_v_line=None,
                         fig_title=None, save_name=None, save_path=None,
                         verbose=True):
    """
    Calculate and plot the mean and error estimates (y-axis) at each separation values (x-axis) including 1Probe.
    Separate line for each ISI.  Error bar values taken from separate error_df.

    :param fig_df: dataframe to build plot from.  Expects fig_df in the form:
        separation as index, 1Probe as bottom row, ISIs as columns.
    :param error_df: dataframe of same shape as fig_df, but contains error values
    :param thr_col: Column to extract threshold values from - expects either probeLum or newLum.
    :param split_1probe: Default=True - whether to treat 1Probe data separately,
        e.g., not joined with line to 2probe data.
    :param jitter: Jitter x_axis values so points don't overlap.
    :param error_caps: caps on error bars for more easy reading
    :param alt_colours: Use different set of colours to normal (e.g., if ISI on
        x-axis and lines for each separation).
    :param legend_names: Names of different lines (e.g., ISI names)
    :param x_tick_vals: Positions on x-axis.
    :param x_tick_labels: labels for x-axis.
    :param fixed_y_range: default=False. If True, it will use full range of y values
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
        print(f'error_df:\n{error_df}')

    # split 1Probe from bottom of fig_df and error_df
    if split_1probe:
        two_probe_df, one_probe_df = split_df_into_pos_sep_df_and_1probe_df(fig_df, even_spaced_x=even_spaced_x)
        two_probe_er_df, one_probe_er_df = split_df_into_pos_sep_df_and_1probe_df(error_df, even_spaced_x=even_spaced_x)
        if verbose:
            print(f'two_probe_df:\n{two_probe_df}')
            print(f'two_probe_er_df:\n{two_probe_er_df}')
            print(f'one_probe_df:\n{one_probe_df}')
            print(f'one_probe_er_df:\n{one_probe_er_df}')
    else:
        two_probe_df = fig_df
        # two_probe_df.drop('separation', axis=1, inplace=True)
        if 'separation' in list(two_probe_df.columns):
            two_probe_df.set_index('separation', drop=True, inplace=True)
        two_probe_er_df = error_df
        # two_probe_er_df.drop('separation', axis=1, inplace=True)
        if 'separation' in list(two_probe_er_df.columns):
            two_probe_er_df.set_index('separation', drop=True, inplace=True)

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
    print(f"\nn_pos_sep: {n_pos_sep}")

    jit_max = 0
    if jitter:
        jit_max = .1
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
        print(f'idx: {idx}, name: {name}')

        # get rand float to add to x-axis for jitter
        jitter_list = np.random.uniform(size=n_pos_sep, low=-jit_max, high=jit_max)

        if split_1probe:
            if x_tick_vals is not None:
                print(f'x_tick_vals: {x_tick_vals}')
                print(f'one_probe_df[thr_col][idx]: {one_probe_df[thr_col][idx]}')
                print(f'one_probe_er_df[thr_col][idx]: {one_probe_er_df[thr_col][idx]}')

                # ax.errorbar(x=x_tick_vals + np.random.uniform(size=len(x_tick_vals), low=-jit_max, high=jit_max),
                ax.errorbar(x=x_tick_vals[-1] + np.random.uniform(size=1, low=-jit_max, high=jit_max),
                            y=one_probe_df[thr_col][idx],
                            yerr=one_probe_er_df[thr_col][idx],
                            marker='.', lw=2, elinewidth=.7,
                            capsize=cap_size,
                            color=my_colours[idx])
            else:
                ax.errorbar(x=one_probe_df['x_vals'][idx] + np.random.uniform(low=-jit_max, high=jit_max),
                            y=one_probe_df[thr_col][idx],
                            yerr=one_probe_er_df[thr_col][idx],
                            marker='.', lw=2, elinewidth=.7,
                            capsize=cap_size,
                            color=my_colours[idx])

        # force it to use sep index
        if x_tick_vals is not None:
            # # I've updated this as the x tick vals list was longer than jitter list.
            print(f"x_tick_vals[:-1]: {x_tick_vals[:-1]} + jitter_list: {jitter_list} == {x_tick_vals[:-1] + jitter_list}")
            ax.errorbar(x=x_tick_vals[:-1] + jitter_list,
            # ax.errorbar(x=x_tick_vals[-1] + jitter_list[-1],
                        y=two_probe_df[name], yerr=two_probe_er_df[name],
                        marker='.', lw=2, elinewidth=.7,
                        capsize=cap_size,
                        color=my_colours[idx])
        else:
            ax.errorbar(x=two_probe_df.index + jitter_list,
                        y=two_probe_df[name], yerr=two_probe_er_df[name],
                        marker='.', lw=2, elinewidth=.7,
                        capsize=cap_size,
                        color=my_colours[idx])

        if legend_names is not None:
            leg_label = legend_names[idx]
        else:
            leg_label = name
        leg_handle = mlines.Line2D([], [], color=my_colours[idx], label=leg_label,
                                       marker='.', linewidth=.5, markersize=4)
        legend_handles_list.append(leg_handle)

    # x values and labels
    if x_tick_vals is not None:
        ax.set_xticks(x_tick_vals)
    if x_tick_labels is not None:
        ax.set_xticklabels(x_tick_labels)
    ax.set_xlabel('Probe separation in diagonal pixels')
    ax.set_ylabel('Probe Luminance')

    # break axis if necessary
    if split_1probe:
        # break axis before 1Probe
        break_axis_before_label(ax, '1Probe', x_tick_labels, subplot=False)

    if fixed_y_range:
        ax.set_ylim([0, 110])
        if type(fixed_y_range) in [tuple, list]:
            ax.set_ylim([fixed_y_range[0], fixed_y_range[1]])

    # add vertical grey dashed line for Ricco's area or Bloch critical duration.
    if ra_cd_v_line:
        plt.axvline(x=ra_cd_v_line, linestyle='--', color='lightgrey')
        leg_handle = mlines.Line2D([], [], label=f"Ricco's Area",
                                   color='lightgrey', linestyle='--')
        legend_handles_list.append(leg_handle)

    # add legend
    ax.legend(handles=legend_handles_list, title='ISI', fontsize=6, framealpha=.5)

    if fig_title is not None:
        plt.title(fig_title)

    if save_path is not None:
        if save_name is not None:
            plt.savefig(os.path.join(save_path, save_name))

    return fig


###################

def plot_w_errors_no_1probe(wide_df, x_var, y_var, lines_var,
                            long_df_idx_col='stack',
                            legend_names=None,
                            x_tick_labels=None,
                            alt_colours=True,
                            fixed_y_range=False,
                            jitter=True,
                            error_caps=True,
                            ra_cd_v_line=None,
                            fig1b_title=None,
                            fig1b_savename=None,
                            save_path=None,
                            verbose=True):
    """
    Function to plot pointplot with error bars.  Use this for plots unless
    there is a need for the separate 1Probe condition.  Note: x-axis is categorical,
    so it's not easy to move ticks.  If I want to do this, use plot_1probe_w_errors().

    :param wide_df: wide form dataframe with data from multiple runs
    :param x_var: Name of variable to go on x-axis (should be consistent with wide_df)
    :param y_var: Name of variable to go on y-axis (should be consistent with wide_df)
    :param lines_var: Name of variable for the lines (should be consistent with wide_df)
    :param long_df_idx_col: Name of column to use as index when going wide to long form df.
    :param legend_names: Default: None, which will access frequently used names.
        Else pass list of names to appear on legend, use verbose to compare order with matplotlib assumptions.
    :param x_tick_labels: Default: None, which will access frequently used labels.
        Else pass list of labels to appear on x-axis.  Note: for pointplot x-axis is categorical,
        not numerical; so all x-ticks are evenly spaced.  For variable x-axis use plot_1probe_w_errors().
    :param alt_colours: Default=True.  Use alternative colours to differentiate
        from other plots e.g., colours associated with Sep not ISI.
    :param fixed_y_range: If True it will fix y-axis to 0:110.  Otherwise, uses adaptive range.
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
        legend_names = ['0', '1', '2', '3', '6', '18', '1Probe']
    if x_tick_labels is None:
        x_tick_labels = ['Conc', 0, 2, 4, 6, 9, 12, 24]

    # get names for legend (e.g., different lines_var)
    n_colours = len(wide_df[lines_var].unique())

    # do error bars have caps?
    cap_size = 0
    if error_caps:
        cap_size = .1

    # convert wide_df to long for getting means and standard error.
    long_fig_df = make_long_df(wide_df, thr_col=y_var, idx_col=long_df_idx_col)
    if verbose:
        print(f'long_fig_df:\n{long_fig_df}')

    my_colours = fig_colours(n_colours, alternative_colours=alt_colours)
    print(f"my_colours - {np.shape(my_colours)}\n{my_colours}")

    fig, ax = plt.subplots()
    sns.pointplot(data=long_fig_df, x=x_var, y=y_var, hue=lines_var,
                  estimator=np.mean, errorbar='se', dodge=jitter, markers='.',
                  errwidth=1, capsize=cap_size, palette=my_colours, ax=ax)

    # decorate plot
    if x_tick_labels is not None:
        x_ticks = list(range(len(x_tick_labels)))
        print(f"x_ticks: {x_ticks}, x_tick_labels: {x_tick_labels}")
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_tick_labels)
    ax.set_xlabel(x_var)

    # if ISI 24 in x_tick_labsls, break axis before it.
    if 24 in x_tick_labels:
        break_axis_before_label(ax, 24, x_tick_labels, subplot=False)


    if y_var in ['newLum', 'probeLum']:
        ax.set_ylabel('Probe Luminance')
    else:
        ax.set_ylabel(y_var)

    if fixed_y_range:
        ax.set_ylim([0, 110])
        if type(fixed_y_range) in [tuple, list]:
            ax.set_ylim([fixed_y_range[0], fixed_y_range[1]])

    # sort legend handles and labels
    handles, orig_labels = ax.get_legend_handles_labels()
    if legend_names is None:
        legend_names = orig_labels
    if verbose:
        print(f'orig_labels and Legend names:')
        for a, b in zip(orig_labels, legend_names):
            print(f"{a}\t=>\t{b}\tmatch: {bool(a == b)}")

    # add vertical grey dashed line for Ricco's area or Bloch critical duration.
    if ra_cd_v_line:
        plt.axvline(x=ra_cd_v_line, linestyle='--', color='lightgrey')

        # add vertical grey dashed line for Ricco's area or Bloch critical duration.
        if ra_cd_v_line:
            plt.axvline(x=ra_cd_v_line, linestyle='--', color='lightgrey')
            leg_handle = mlines.Line2D([], [], label=f"Bloch's CD",
                                       color='lightgrey', linestyle='--')
            handles.append(leg_handle)
            legend_names.append(f"Bloch's CD")

    # add legend
    ax.legend(handles, legend_names, fontsize=6, title=lines_var, framealpha=.5)

    if fig1b_title is not None:
        plt.title(fig1b_title)

    if save_path is not None:
        if fig1b_savename is not None:
            plt.savefig(os.path.join(save_path, fig1b_savename))

    return fig


###########################


def plot_thr_heatmap(heatmap_df,
                     x_tick_labels=None,
                     y_tick_labels=None,
                     midpoint=None,
                     ra_cd_v_line=None,
                     ra_cd_h_line=None,
                     fig_title=None,
                     save_name=None,
                     save_path=None,
                     annot_fmt='.0f',
                     my_colourmap='RdYlGn_r',
                     verbose=True):
    """
    Function for making a heatmap
    :param heatmap_df: Expects dataframe with separation as index and ISIs as columns.
    :param x_tick_labels: Labels for columns
    :param y_tick_labels: Labels for rows
    :param midpoint: Value to use as midpoint of colour range (e.g., mean or zero etc).
    :param fig_title: title for figure
    :param save_name: filename for plot
    :param save_path: path to save plot to.
    :param my_colourmap: Colours to use.  
    :param verbose: Whether to print progress to screen.
    :return: Heatmap
    """

    print('\n*** running plot_thr_heatmap() ***\n')

    # rename columns and index so plots have right axis labels
    heatmap_df.rename(columns={'ISI_-1': 'Conc', 'Concurrent': 'Conc'},
                      index={20: '1pr', '20': '1pr', '1Probe': '1pr'}, inplace=True)

    # if 'ISI_' in any of the column names, rename them to just the ISI value.
    for col in heatmap_df.columns:
        if type(col) == str and 'ISI_' in col:
            new_col = col.split('_')[1]
            heatmap_df.rename(columns={col: new_col}, inplace=True)
        # if 'ISI_' in col:
        #     new_col = col.split('_')[1]
        #     heatmap_df.rename(columns={col: new_col}, inplace=True)

    if verbose:
        print(f'heatmap_df:\n{heatmap_df}')

    if verbose:
        print(f'heatmap_df:\n{heatmap_df}')

    # if x_tick_labels is None:
    #     x_tick_labels = list(heatmap_df.columns)
    #     x_tick_labels = ['Conc' if i in ['ISI_-1', 'Concurrent'] else i for i in x_tick_labels]
    # if y_tick_labels is None:
    #     y_tick_labels = list(heatmap_df.index)
    #     y_tick_labels = ['1pr' if i in [20, '20', '1Probe'] else i for i in y_tick_labels]
    #     y_tick_labels = ['1pr' if i == 20 else i for i in y_tick_labels]

    # get mean of each column, then mean of those
    if midpoint is None:
        mean_thr = float(np.mean(heatmap_df.mean()))
        midpoint = mean_thr
        if verbose:
            print(f'midpoint (mean_val_: {round(mean_thr, 2)}')


    heatmap = sns.heatmap(data=heatmap_df,
                          annot=True,
                          # fmt=annot_fmt,
                          center=midpoint,
                          cmap=my_colourmap,
                          # xticklabels=x_tick_labels, yticklabels=y_tick_labels,
                          square=False)

    # set axis labels, assume columns are ISI and separation is index.
    heatmap.set_xlabel('ISI')
    heatmap.set_ylabel('Separation')

    # This sets the yticks "upright" with 0, as opposed to sideways with 90.
    plt.yticks(rotation=0)
    
    # add vertical grey dashed line for Ricco's area or Bloch critical duration.
    if ra_cd_v_line:
        plt.axvline(x=int(str(ra_cd_v_line).split('.')[0]),
                    linestyle='--', color='dimgrey')
        
    # add horizontal grey dashed line for Ricco's area or Bloch critical duration.
    if ra_cd_h_line:
        plt.axhline(y=int(str(ra_cd_h_line).split('.')[0]),
                    linestyle='--', color='dimgrey')

    if fig_title is not None:
        plt.title(fig_title)

    if save_path is not None:
        if save_name is not None:
            plt.savefig(os.path.join(save_path, save_name))

    return heatmap

##########################


def plt_heatmap_row_col(heatmap_df,
                        colour_by='row',
                        x_axis_label=None,
                        y_axis_label=None,
                        ra_cd_v_line=None,
                        ra_cd_h_line=None,
                        fig_title=None,
                        midpoint=None,
                        fontsize=12,
                        annot_fmt='.0f',
                        save_name=None,
                        save_path=None,
                        my_colourmap='RdYlGn_r',
                        verbose=True):
    """
    Function for making a heatmap
    :param heatmap_df: Expects dataframe with separation as index and ISIs as columns.
    :param colour_by: colour plots by 'rows' or 'cols.  That is, each row or column is coloured individually.
    :param x_axis_label: Name for x_axis
    :param y_axis_label: name for y_axis
    :param fig_title: Title for figure.
    :param midpoint: Value to use a midpoint for colours (e.g., mean or zero).
    :param fontsize: Textsize for heatmap values.
    :param save_name: Name for figure.
    :param save_path: Path to save figure to.
    :param verbose: Whether to print progress to screen.
    :param my_colourmap: Colourmap to use
    :return: Heatmap
    """

    print(f'\n*** running plt_heatmap_row_col(colour_by={colour_by}) ***\n')


    # rename columns by stripping "ISI_" from column names
    heatmap_df.rename(columns=lambda x: x.replace('ISI_', ''), inplace=True)

    # rename columns and index so plots have right axis labels
    heatmap_df.rename(columns={'ISI_-1': 'Conc', 'Concurrent': 'Conc', '-1': 'Conc'},
                      index={20: '1pr', '20': '1pr'}, inplace=True)
    if verbose:
        print(f'heatmap_df:\n{heatmap_df}')

    # get labels to loop over rows or columns
    col_names = list(heatmap_df.columns)
    row_names = list(heatmap_df.index)
    print(f"col_names: {col_names}")
    print(f"row_names: {row_names}")


    if str.lower(colour_by) in ['col', 'columns', 'horizontal']:
        fig, axs = plt.subplots(ncols=len(col_names), sharey=True)
        loop_over = col_names
    else:
        fig, axs = plt.subplots(nrows=len(row_names), sharex=True)
        loop_over = row_names

    # loop over rows or columns making heatmaps
    print(f"loop over {colour_by}s: ({loop_over})")
    for idx, (ax, tick_label) in enumerate(zip(axs, loop_over)):
        print(f"idx: {idx}, ax: {ax}, tick_label: {tick_label}")
        if str.lower(colour_by) in ['col', 'columns', 'horizontal']:
            use_this_data = heatmap_df[[tick_label]]
        else:
            use_this_data = heatmap_df.loc[[tick_label]]

        sns.heatmap(data=use_this_data,
                    ax=ax,
                    linewidths=.05,
                    cmap=my_colourmap,
                    center=midpoint,
                    annot=True,
                    annot_kws={'fontsize': fontsize},
                    fmt=annot_fmt,
                    cbar=False,
                    square=True)

        # if heatmap is by column
        if str.lower(colour_by) in ['col', 'columns']:
            # # arrange labels and ticks per ax
            ax.set_xlabel(None)
            if ax == axs[0]:
                ax.set_ylabel(y_axis_label, fontsize=fontsize)
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            else:
                ax.tick_params(left=False)
                ax.set_ylabel(None)
            plt.subplots_adjust(wspace=-0.5, hspace=-.5)

            # add horizontal grey dashed line to each vertical ax (for RA or CD)
            if ra_cd_h_line:
                ax.axhline(y=int(str(ra_cd_h_line).split('.')[0]),
                           linestyle='--', color='black', linewidth=2)

            # add vertical grey dashed line BETWEEN vertical axes (for RA or CD).
            if ra_cd_v_line:
                if idx == int(str(ra_cd_v_line).split('.')[0]):
                    # e.g., if threshold would be drawn at 3.6 on lineplot, put divider is after ax idx 3
                    ax.axvline(x=1, linestyle='--', color='black', linewidth=5)

        # if heatmap is by row
        else:  # str.lower(colour_by) in ['row', 'rows']:
            # # arrange labels and ticks per ax
            ax.set_ylabel(None)
            if ax == axs[-1]:
                ax.set_xlabel(x_axis_label, fontsize=fontsize)
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            else:
                ax.tick_params(bottom=False)
            plt.subplots_adjust(hspace=0.1)

            # add horizontal grey dashed line BETWEEN horizontal axes (for RA or CD).
            if ra_cd_h_line:
                if idx == int(str(ra_cd_h_line).split('.')[0]):
                    # e.g., if threshold would be drawn at 3.6 on lineplot, put divider is after ax idx 3
                    ax.axhline(y=1.0, linestyle='--', color='black', linewidth=5)

            # add vertical grey dashed line to each horizontal ax (for RA or CD)
            if ra_cd_v_line:
                ax.axvline(x=int(str(ra_cd_v_line).split('.')[0]),
                           linestyle='--', color='black', linewidth=2)


    # # make colourbar: numbers are (1) the horizontal and (2) vertical position
    # # of the bottom left corner, then (3) width and (4) height of colourbar.
    cb_ax = fig.add_axes([0.85, 0.11, 0.02, 0.77])
    cbar = plt.colorbar(cm.ScalarMappable(cmap=my_colourmap), cax=cb_ax)
    cbar.set_ticks(np.arange(0, 1.1, 0.5))
    cbar.set_ticklabels(['Lowest', 'Med', 'Highest'])
    cbar.set_label(f'Luminance threshold per {colour_by}')

    if fig_title is not None:
        plt.suptitle(fig_title, fontsize=fontsize+2)

    if save_path is not None:
        if save_name is not None:
            plt.savefig(os.path.join(save_path, save_name))

    print(f'\n*** finished plt_heatmap_row_col(colour_by={colour_by}) ***\n')

    return fig


def make_diff_from_conc_df(MASTER_TM2_thr_df, root_path, error_type='SE',
                           n_trimmed=2, mean_and_err_df=True, exp_ave=False, verbose=True):
    """
    Load in the MASTER_TM2_thresholds.csv with trimmed thresholds for a participant.

    For each row (stack and separation combination), get the difference between Concurrent
    and all other ISIs by subtracting concurrent from that ISI.

    Then calculate the mean difference from concurrent across these stacks and also the error (SE).

    Returns the average diff from conc and the error on the average
    :param MASTER_TM2_thr_df: Either an actual DataFrame or a path to dataframe.
        thr_df is a ISI (columns) x separation (rows) dataframe.
    :param error_type: Default: None. Can pass sd or se for standard deviation or error.
    :param root_path: Path to save new csvs to.
    :param n_trimmed: Number of datapoints trimmed from each end.  Used for naming file.
    :param mean_and_err_df: If true, will return the mean diff from conc from all participant runs (and error).
                            If False, will return the diff from conc per run (not the mean of them)
    :param exp_ave: If True, will add 'Exp' to fig titles so its clear these are experiment level results.
        If False will add 'P' for participant level; or use participant_name to identify whose results it is.

    :param verbose: Default True, print progress to screen

    :return: MASTER_DfC_df, MASTER_DfC_error_SE_df
    """
    print('\n*** running make_diff_from_conc_df() ***')

    # todo: chop off 1Probe data?

    if type(exp_ave) == str:  # e.g. participant's name
        ave_over = exp_ave
        idx_col = 'stack'
    elif exp_ave == True:
        ave_over = 'Exp'
        idx_col = 'p_stack_sep'
    else:
        ave_over = 'P'
        idx_col = 'stack'
    print(f'ave_over: {ave_over}')
    print(f'exp_ave: {exp_ave}')
    print(f"idx_col: {idx_col}")

    if isinstance(MASTER_TM2_thr_df, str):
        thr_df = pd.read_csv(MASTER_TM2_thr_df)
    elif isinstance(MASTER_TM2_thr_df, pd.DataFrame):
        thr_df = MASTER_TM2_thr_df
    else:
        raise TypeError(f'MASTER_TM2_thr_df should be str (path to dataframe) or '
                        f'DataFrame, not {type(MASTER_TM2_thr_df)}')

    if 'separation' in list(thr_df.columns):
        thr_df.set_index('separation', inplace=True, drop=True)
    thr_df.rename(index={20: '1Probe'}, inplace=True)

    if ave_over == 'Exp':
        # thr_df.drop('p_stack_sep', axis=1, inplace=True)
        # thr_df.drop('participant', axis=1, inplace=True)
        p_stack_sep_S = thr_df.pop('p_stack_sep').to_list()
        participant_S = thr_df.pop('participant').to_list()
        print(f'p_stack_sep_S: {p_stack_sep_S}')
        print(f'participant_S: {participant_S}')

    # check column order
    if 'Concurrent' in list(thr_df.columns):
        conc_col = 'Concurrent'
    elif 'Conc' in list(thr_df.columns):
        conc_col = 'Conc'
    else:
        conc_col = 'ISI_-1'

    thr_df = conc_to_first_isi_col(thr_df, col_to_change=conc_col)

    # multi-index so stack values are preserved for all_dfc_df
    thr_df.reset_index(inplace=True, drop=False)
    # thr_df.set_index(['separation', 'stack'], inplace=True, drop=True)
    multi_index_cols = ['separation', 'stack']
    if 'cond_type' in list(thr_df.columns):
        multi_index_cols.append('cond_type')
    if 'neg_sep' in list(thr_df.columns):
        multi_index_cols.append('neg_sep')
    thr_df.set_index(multi_index_cols, inplace=True, drop=True)

    if verbose:
        print(f'thr_df ({list(thr_df.columns)}):\n{thr_df}')


    '''all_dfc_df is an ISI x Sep df where the concurrent column is subtracted from all columns.
    therefore, the first column has zero for all values,
    and all other columns show the difference between an ISI and concurrent.'''
    if 'Concurrent' in list(thr_df.columns):
        all_dfc_df = thr_df.iloc[:, :].sub(thr_df.Concurrent, axis=0)
    elif 'Conc' in list(thr_df.columns):
        all_dfc_df = thr_df.iloc[:, :].sub(thr_df.Conc, axis=0)
    elif 'ISI_-1' in list(thr_df.columns):
        all_dfc_df = thr_df.iloc[:, :].sub(thr_df['ISI_-1'], axis=0)
    elif 'ISI 999' in list(thr_df.columns):
        # remane ISI 999 to ISI -1
        # thr_df.rename(columns={'ISI 999': 'ISI -1'}, inplace=True)
        # all_dfc_df = thr_df.iloc[:, :].sub(thr_df['ISI -1'], axis=0)
        all_dfc_df = thr_df.iloc[:, :].sub(thr_df['ISI 999'], axis=0)

    all_dfc_df.reset_index(inplace=True, drop=False)
    if verbose:
        print(f'all_dfc_df:\n{all_dfc_df}')

    if not mean_and_err_df:  # just return all dfc values, not mean and error

        if ave_over == 'Exp':
            all_dfc_df.insert(0, 'p_stack_sep', p_stack_sep_S)
            all_dfc_df.insert(1, 'participant', participant_S)
        if verbose:
            print(f'returning all_dfc_df:\n{all_dfc_df}')

        return all_dfc_df

    else:  # if mean_and_err_df, make and retrun mean and error of dfc

        # get means and errors
        get_means_df = all_dfc_df
        if verbose:
            print(f'get_means_df:\n{get_means_df}')

        # # get means and errors
        # reset index so we can groupby separation and remove stack
        # get_means_df.reset_index(inplace=True, drop=False)
        get_means_df.set_index('separation', inplace=True, drop=True)

        # use neg sep to average over if available
        ave_over_col = 'separation'
        multi_index_cols = ['separation']
        if 'cond_type' in list(get_means_df.columns):
            multi_index_cols.append('cond_type')
        if 'neg_sep' in list(get_means_df.columns):
            multi_index_cols.append('neg_sep')
            ave_over_col = multi_index_cols
        if len(multi_index_cols) > 1:
            get_means_df.reset_index(inplace=True, drop=False)
            get_means_df.set_index(multi_index_cols, inplace=True, drop=True)
        print(f'get_means_df:\n{get_means_df}')
        

        groupby_sep_df = get_means_df.drop('stack', axis=1)
        ave_DfC_df = groupby_sep_df.groupby(ave_over_col, sort=True).mean()
        ave_DfC_df.reset_index(inplace=True, drop=False)

        if verbose:
            print(f'\nave_DfC_df:\n{ave_DfC_df}')

        if error_type in [False, None]:
            error_DfC_df = None
        elif error_type.lower() in ['se', 'error', 'std-error', 'standard error', 'standard_error']:
            error_DfC_df = groupby_sep_df.groupby(ave_over_col, sort=True).sem()
        elif error_type.lower() in ['sd', 'stdev', 'std_dev', 'std.dev', 'deviation', 'standard_deviation']:
            error_DfC_df = groupby_sep_df.groupby(ave_over_col, sort=True).std()
        else:
            raise ValueError(f"error_type should be in:\nfor none: [False, None]\n"
                             f"for standard error: ['se', 'error', 'std-error', 'standard error', 'standard_error']\n"
                             f"for standard deviation: ['sd', 'stdev', 'std_dev', 'std.dev', "
                             f"'deviation', 'standard_deviation']")
        error_DfC_df.reset_index(inplace=True, drop=False)
        if verbose:
            print(f'\nerror_DfC_df: ({error_type})\n{error_DfC_df}')

        # save csv with average values
        if n_trimmed is not None:
            ave_DfC_df.to_csv(os.path.join(root_path, f'MASTER_ave_TM{n_trimmed}_DfC.csv'), index=False)
            error_DfC_df.to_csv(os.path.join(root_path, f'MASTER_ave_TM{n_trimmed}_DfC_error_{error_type}.csv'), index=False)
        else:
            ave_DfC_df.to_csv(os.path.join(root_path, 'MASTER_ave_DfC.csv'), index=False)
            error_DfC_df.to_csv(os.path.join(root_path, f'MASTER_ave_DfC_error_{error_type}.csv'), index=False)

        print('\n*** finished make_diff_from_conc_df() ***')

        return ave_DfC_df, error_DfC_df



def plot_diff_from_conc_lineplot(ave_DfC_df, err_DfC_df, fig_title=None,
                                 ra_cd_v_line=None,
                                 save_name=None, save_path=None):
    """
    Function to plot the difference in threshold from concurrent for each ISI.

    :param ave_DfC_df: Either an actual DataFrame or a path to dataframe.
        ave_DfC_df is a ISI (columns) x separation (rows) dataframe.
    :param err_DfC_df: Either an actual DataFrame or a path to dataframe showing errors on threshold averages.
        err_DfC_df is a ISI (columns) x separation (rows) dataframe.
    :param fig_title: Title for figure.
    :param save_name: File name to save figure.
    :param save_path: Path to save file if a ave_DfC_df is a dataframe.

    :return: Fig
    """
    print('\n*** running plot_diff_from_conc_lineplot() ***')

    # get data
    if type(ave_DfC_df) is str:
        if os.path.isfile(ave_DfC_df):
            ave_DfC_df = pd.read_csv(ave_DfC_df)

    if type(err_DfC_df) is str:
        if os.path.isfile(err_DfC_df):
            err_DfC_df = pd.read_csv(err_DfC_df)

    # set separarion to be the index column
    if ave_DfC_df.index.name != 'separation':
        drop_idx = False
        if ave_DfC_df.index.name == None:
            drop_idx = True
        ave_DfC_df.reset_index(inplace=True, drop=drop_idx)
        err_DfC_df.reset_index(inplace=True, drop=drop_idx)
        ave_DfC_df.set_index('separation', drop=True, inplace=True)
        err_DfC_df.set_index('separation', drop=True, inplace=True)

    # if neg_sep in dfs, drop it
    if 'neg_sep' in list(ave_DfC_df.columns):
        ave_DfC_df.drop(columns=['neg_sep'], inplace=True)
        err_DfC_df.drop(columns=['neg_sep'], inplace=True)
    #if cond_type in dfs, pop it to use for hue
    with_hue = False
    if 'cond_type' in list(ave_DfC_df.columns):
        cond_type_list = ave_DfC_df.pop('cond_type').tolist()
        print(f"\ncond_type_list:\n{cond_type_list}")
        with_hue = True
        err_DfC_df.drop(columns=['cond_type'], inplace=True)
    print(f"\nave_DfC_df:\n{ave_DfC_df}")
    print(f"\nerr_DfC_df:\n{err_DfC_df}")

    column_names = ave_DfC_df.columns.to_list()
    x_tick_labels = ['Conc' if i in ['ISI_-1', 'Concurrent'] else i for i in column_names]
    # strip 'ISI_' from x_tick_labels if present
    x_tick_labels = [int(i.strip('ISI_')) if 'ISI_' in i else i for i in x_tick_labels]
    print(f"\nx_tick_labels:\n{x_tick_labels}")

    index_names = ave_DfC_df.index.tolist()
    print(f"\nindex_names:\n{index_names}")
    # # check if any of the index_names include the substring 'missing'
    # missing_sep = [i for i in index_names if 'missing' in str(i)]
    # exp1_sep = [i for i in index_names if 'exp1' in str(i)]
    # if (len(missing_sep) > 0) and (len(exp1_sep) > 0):
    #     with_hue = True
    #     print(f"\nwith_hue: {with_hue}")

    my_colours = fig_colours(len(set(index_names)))
    colour_dict = dict(zip(set(index_names), my_colours))

    fig, ax = plt.subplots()
    legend_handles_list = []
    cond_handles_list = []
    check_handle_duplicates = []

    # for idx, name in enumerate(index_names):
    for idx, sep_row in enumerate(ave_DfC_df.index):
        print(f"idx: {idx}, sep_row: {sep_row}")

        linestyle = '-'

        if with_hue:
            # use dashed line if with_hue is True
            if 'missing' in cond_type_list[idx]:
                linestyle = '--'

        ax.errorbar(x=column_names,
                    y=ave_DfC_df.iloc[idx],
                    yerr=err_DfC_df.iloc[idx],
                    marker='.', lw=2, elinewidth=.7,
                    capsize=5,
                    linestyle=linestyle,
                    color=colour_dict[sep_row])

        # legend handle for separation
        if sep_row not in check_handle_duplicates:
            leg_handle = mlines.Line2D([], [], color=colour_dict[sep_row], label=sep_row,
                                       marker='P', markersize=4, linewidth=0)
            legend_handles_list.append(leg_handle)
            check_handle_duplicates.append(sep_row)

        #legend handle for cond type
        if with_hue:
            if cond_type_list[idx] not in check_handle_duplicates:
                leg_handle = mlines.Line2D([], [], color='black', label=cond_type_list[idx],
                                           linestyle=linestyle, linewidth=2, markersize=0)
                cond_handles_list.append(leg_handle)
                check_handle_duplicates.append(cond_type_list[idx])

    legend_handles_list = legend_handles_list + cond_handles_list

    ax.set_xticks(list(range(len(x_tick_labels))))
    ax.set_xticklabels(x_tick_labels)

    # break axes if (isi) 24 is present
    if 24 in x_tick_labels:
        break_axis_before_label(ax, 24, x_tick_labels)


    plt.axhline(y=0, color='lightgrey', linestyle='dashed')
    plt.ylabel('Luminance difference from concurrent')
    plt.xlabel('ISI')
    plt.title(fig_title)

    # add vertical grey dashed line for Ricco's area or Bloch critical duration.
    if ra_cd_v_line:
        plt.axvline(x=ra_cd_v_line, linestyle='--', color='grey', label='Critical Duration')
        leg_handle = mlines.Line2D([], [], color='grey', label='Critical Duration',
                                   linestyle='--')
        legend_handles_list.append(leg_handle)

    ax.legend(handles=legend_handles_list, fontsize=8, title='separation', framealpha=.5)

    if save_path is not None:
        if save_name is not None:
            plt.savefig(os.path.join(save_path, save_name))

    print('\n*** finished plot_diff_from_conc_lineplot() ***')

    return fig




def plot_thr_3dsurface(plot_df,
                       drop_1probe=False,
                       drop_conc=False,
                       ra_cd_size_dict=None,
                       even_spaced=False,
                       transpose_df=False,
                       show_min_per_sep=True,
                       my_elev=None, my_rotation=None,
                       fig_title=None,
                       save_path=None, save_name=None,
                       verbose=True):
    """
    Function to plot a 3d surface plot of average thresholds.
    Can add markers to show minimum per ISI or Separation.
    
    I currently like to use transpose_df=True, with my_rotation=315.

    :param plot_df: dataframe (sep index, ISI columns)
    :param even_spaced: Evenly spaced axis are better if I am transforming data (not sure why)
    :param transpose_df: If not using my_rotation, I can manually transpose df.
    :param show_min_per_sep: If True, Show minimum value per separation with markers.
    :param my_elev: Change viewing angle elevation.  90 is top down, 0 is side on, default is 30.
    :param my_rotation: Change viewing angle azimuth.
                    Default is -60: This gives 0-24 ISI on the left and 0-18 sep on the right.
                    45: 0-18 sep on left, 24-0 ISI on right.
                    135: 24-0 ISI on left, 18-0 sep on right.
                    225: 18-0 sep on left, 0-24 ISI on right.
                    315: 0-24 ISI on the left and 0-18 sep on the right.
    :param fig_title: Title if I want to override defaults.
    :param save_path: Path to save to.
    :param save_name: File name to save
    :param verbose: print progress to screen

    :return: figure
    """
    print('\n*** running plot_thr_3dsurface() ***')

    if verbose:
        print(f'\ninput plot_df:\n{plot_df}')

    # change index and column labels to int
    sep_labels = list(plot_df.index)
    isi_labels = list(plot_df.columns)

    # if col labels contain "ISI " or "ISI_", strip it from values and convert to int
    if 'ISI ' in isi_labels[1]:
        if 'Conc' in isi_labels[0]:
            isi_labels[0] = 'ISI -1'
        isi_labels = [int(i.split(' ')[1]) for i in isi_labels]
    elif 'ISI_' in isi_labels[1]:
        if 'Conc' in isi_labels[0]:
            isi_labels[0] = 'ISI_-1'
        isi_labels = [int(i.split('_')[1]) for i in isi_labels]
    plot_df.columns = isi_labels

    # if plot_df.index is 'separation', just convert to int
    if plot_df.index.name == 'separation':
        sep_labels = [20 if i == '1Probe' else int(i) for i in sep_labels]
        plot_df.index = sep_labels

    if drop_1probe:
        if 20 in plot_df.index:
            plot_df = plot_df.drop(20)
            print(f'20 (1Probe) dropped from index: {plot_df.index}')
            sep_labels = list(plot_df.index)
        else:
            print(f'20 (1Probe) not in index: {plot_df.index}')
    if drop_conc:
        if -1 in list(plot_df.columns):
            plot_df = plot_df.drop(-1, axis=1)
            print(f'-1 (concurrent) dropped from columns: {plot_df.columns}')
            isi_labels = list(plot_df.columns)
        else:
            print(f'-1 (concurrent) not in columns: {plot_df.columns}')
    if verbose:
        print(f'\ninput plot_df:\n{plot_df}')
        print(f'sep_labels: {sep_labels}')
        print(f'isi_labels: {isi_labels}')

    x_label = 'ISI'
    y_label = 'Separation'
    figure_title = 'Average threshold for each ISI and separation condition.'
    x_isi_y_sep = True  # do I need this for getting correct labels & locations if I transpose df?

    if transpose_df:  # swap index and columns
        plot_df = plot_df.T
        x_label = 'Separation'
        y_label = 'ISI'
        x_isi_y_sep = False

        # swap row and column labels
        sep_labels = list(plot_df.index)
        isi_labels = list(plot_df.columns)


    # # arrays to use for the meshgrid (scatter points for lower ISI per sep)
    sep_array = np.array(sep_labels)
    isi_array = np.array(isi_labels)

    if even_spaced:  # # replace cols and row names with ascending values,

        # evenly spaced axes for meshgrid
        sep_array = np.array(list(range(len(sep_array))))
        isi_array = np.array(list(range(len(isi_array))))

        # give df basic rows and cols
        plot_df.reset_index(inplace=True, drop=True)
        plot_df.columns = isi_array

    if verbose:
        print(f'transformed plot_df:\n{plot_df}')

    # data for surface
    x_array, y_array = np.meshgrid(isi_array, sep_array)
    z_array = plot_df.to_numpy()
    if verbose:
        print(f'\nvalues for surface:')
        print(f'sep_array: {sep_array}')
        print(f'isi_array: {isi_array}')
        print(f'x_array:\n{x_array}')
        print(f'y_array:\n{y_array}')
        print(f'z_array:\n{z_array}')

    # make figure
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')


    # get the min and max values from the data for the colourbar.
    # Use Matplotlib's TwoSlopeNorm() to centre the colormap at zero.
    z_thr_min = np.min(z_array)
    z_thr_max = np.max(z_array)
    print(f"z_thr_min: {z_thr_min}, z_thr_max: {z_thr_max}")
    two_slope_norm = colors.TwoSlopeNorm(vmin=z_thr_min, vcenter=0, vmax=z_thr_max)

    # use two_slope_norm as my colourmap
    my_cmap = cm.bwr

    surf = ax.plot_surface(X=x_array, Y=y_array, Z=z_array,
                           cmap=my_cmap,
                           norm=two_slope_norm,
                           edgecolor='grey',
                           alpha=.5)
    fig.colorbar(surf, ax=ax, shrink=0.75, label='thr diff from conc')

    if show_min_per_sep:
        fig_title = f"{fig_title}\nMarkers show ISI cond with lowest thr per separation"

        if even_spaced:
            plot_df.reset_index(drop=True, inplace=True)
        scat_x = []
        scat_isi_idx = []
        scat_y = []
        scat_z = []

        # figure out if separation is in rows or columns.  If rows, then min is per column.  If columns, then min is per row
        if x_label == 'Separation':
            print(f"x_label is separation, so each sep val has its own column")
            for index, row in plot_df.T.iterrows():
                scat_x.append(index)
                scat_isi_idx.append(row.argmin())
                scat_z.append(row.min()+1)
            scat_y = [sep_array[i] for i in scat_isi_idx]

        else:  # min per column
            print(f"y_label is separation, so each sep val has its own row")

            for index, row in plot_df.iterrows():
                scat_isi_idx.append(row.argmin())
                scat_y.append(index)
                scat_z.append(row.min())
                print(f"x: {row.argmin()}, y: {index}, z: {row.min()}")
            scat_x = [isi_array[i] for i in scat_isi_idx]

        if verbose:
            print(f'\nValues for 3d scatter\n'
                  f'scat_x: {scat_x}\n'
                  f'scat_y: {scat_y}\n'
                  f'scat_z: {scat_z}\n')

        # Creating scatter plot - marking lowest ISI cond per separation with a diamond
        ax.scatter3D(scat_x, scat_y, scat_z,
                     color="black",
                     marker='D')

    # if given, mark the RA and CD with a line or plane
    if ra_cd_size_dict:
        print(f"getting RA and CD size from ra_cd_size_dict: {ra_cd_size_dict}")
        if even_spaced:
            # if even spaced, use ra_sep_idx and cd_isi_idx
            ra_val = ra_cd_size_dict['ra_sep_idx']
            cd_val = ra_cd_size_dict['cd_isi_idx']
        else:
            # if not evenly spaced, use ra_size_sep and cd_size_isi
            ra_val = ra_cd_size_dict['ra_size_sep']
            cd_val = ra_cd_size_dict['cd_size_isi']

        if verbose:
            print(f'\n(even_spaced: {even_spaced}).  ra_val: {ra_val} cd_val: {cd_val}')

        if not even_spaced:
            # todo: I've not configured ra and cd for even_spaced yet
            # if not even_spaced, use actual ra_val and cd_val (not idx)

            # RA: I need to interpolate between each point to get the z value for the ra_val
            # find sep val below ra_size_sep
            ra_minus1 = [i for i in sep_array if i < ra_val][-1]
            print(f"ra_minus1: {ra_minus1}")

            # find idx of sep val above ra_size_sep
            ra_plus1 = [i for i in sep_array if i > ra_val][0]
            print(f"ra_plus1: {ra_plus1}")

            # get size between values either size of ra_size_sep
            sep_vals_gap = sep_array[ra_minus1 + 1] - sep_array[ra_minus1]
            print(f"sep_vals_gap: {sep_vals_gap}")

            # take decimal from end of ra_size_sep
            decimal_to_add = ra_val - sep_array[ra_minus1]
            print(f"decimal_to_add: {decimal_to_add}")

            # scale decimal to size of gap, e.g., if ra_size_sep is 3.5, and it is between values of 3 and 6,
            # then on plots the line should be 1/6th of the way between 3 and 6.
            scaled_dec_to_add = decimal_to_add / sep_vals_gap
            print(f"scaled_dec_to_add: {scaled_dec_to_add}")

            # get the theshold (z) values for the slice below and above ra for interpolation
            ra_minus1_z = list(plot_df.loc[ra_minus1, :])
            ra_plus1_z = list(plot_df.loc[ra_plus1, :])
            print(f"ra_minus1_z: {ra_minus1_z}")
            print(f"ra_plus1_z: {ra_plus1_z}")

            # z values for ra = ra_minus1 + (ra_plus1 - ra_minus1) * scaled_dec_to_add
            ra_z = [ra_minus1_z[i] + (ra_plus1_z[i] - ra_minus1_z[i]) * scaled_dec_to_add for i in range(len(ra_minus1_z))]
            print(f"ra_z: {ra_z}")
            print(f"plot_df:\n{plot_df}")

            print(f"isi_array: {isi_array}")


            # # add a line to mark the ra_val on the separation axis which follows the contours of the surface
            # for each isi_array value, the line passes through ra_val with the z value from ra_z (isi 0 to 24)
            ax.plot(xs=isi_array, ys=[ra_val]*len(isi_array), zs=ra_z, color='green', linewidth=2, linestyle="--")

            # add a line from z=0 up to the last ra_z point on the separation axis (isi = 24, 24)
            ax.plot(xs=[isi_array[-1], isi_array[-1]], ys=[ra_val, ra_val], zs=[z_thr_min, ra_z[-1]], color='green', linewidth=2, linestyle="--")

            # add a line from last isi_array point to first isi_array point on the separation axis at z=0 (isi 24 to 0)
            ax.plot(xs=[isi_array[-1], isi_array[0]], ys=[ra_val, ra_val], zs=[z_thr_min, z_thr_min], color='green', linewidth=2, linestyle="--")

            # add a line from z=0 up to the first ra_z point on the separation axis (isi = 0, 0)
            ax.plot(xs=[isi_array[0], isi_array[0]], ys=[ra_val, ra_val], zs=[z_thr_min, ra_z[0]], color='green', linewidth=2, linestyle="--")


            ############
            print(f"\n\ncd_val: {cd_val}")

            # CD: I need to interpolate between each point to get the z value for the cd_val
            # find isi val below cd_size_isi
            print(f"isi_array: {isi_array}")
            cd_minus1 = [i for i in isi_array if i < cd_val][-1]
            print(f"cd_minus1: {cd_minus1}")

            # find idx of sep val above ra_size_sep
            cd_plus1 = [i for i in isi_array if i > cd_val][0]
            print(f"cd_plus1: {cd_plus1}")

            # get size between values either size of ra_size_sep
            cd_vals_gap = cd_plus1 - cd_minus1
            print(f"cd_vals_gap: {cd_vals_gap}")

            # take decimal from end of ra_size_sep
            decimal_to_add = cd_val - cd_minus1
            print(f"decimal_to_add: {decimal_to_add}")

            # scale decimal to size of gap, e.g., if cd_size_isi is 3.5, and it is between values of 4 and 6,
            # then on plots the line should be 1/4th of the way between 4 and 6 = .5 / 2 = .25
            scaled_dec_to_add = decimal_to_add / cd_vals_gap
            print(f"scaled_dec_to_add: {scaled_dec_to_add}")

            # get the theshold (z) values for the slice below and above cd for interpolation
            cd_minus1_z = list(plot_df.loc[:, cd_minus1])
            cd_plus1_z = list(plot_df.loc[:, cd_plus1])
            print(f"cd_minus1_z: {cd_minus1_z}")
            print(f"cd_plus1_z: {cd_plus1_z}")

            # z values for ra = cd_minus1 + (cd_plus1 - cd_minus1) * scaled_dec_to_add
            cd_z = [cd_minus1_z[i] + (cd_plus1_z[i] - cd_minus1_z[i]) * scaled_dec_to_add for i in
                    range(len(cd_minus1_z))]
            print(f"cd_z: {cd_z}")
            print(f"plot_df:\n{plot_df}")

            # # add a line to mark the cd_val on the separation axis which follows the contours of the surface
            # for each isi_array value, the line passes through cd_val with the z value from cd_z (sep 0-18)
            ax.plot(xs=[cd_val] * len(sep_array), ys=sep_array, zs=cd_z,
                    color='orange', linewidth=2, linestyle="--")

            # add a line from z=0 up to the last cd_z point on the separation axis (sep 18)
            ax.plot(xs=[cd_val, cd_val], ys=[sep_array[-1], sep_array[-1]], zs=[z_thr_min, cd_z[-1]],
                    color='orange', linewidth=2, linestyle="--")

            # add a line from last sep_array point to first sep_array point on the isi axis at z=0 (sep 18 to 0)
            ax.plot(xs=[cd_val, cd_val], ys=[sep_array[-1], sep_array[0]], zs=[z_thr_min, z_thr_min],
                    color='orange', linewidth=2, linestyle="--")

            # add a line from z=0 up to the first cd_z point on the separation axis (sep 0)
            ax.plot(xs=[cd_val, cd_val], ys=[sep_array[0], sep_array[0]], zs=[z_thr_min, cd_z[0]],
                    color='orange', linewidth=2, linestyle="--")


            # add ra and cd lines to legend
            ra_line_leg = mlines.Line2D([], [], color='green', linewidth=2, linestyle="--",
                                label="Ricco's Area")
            cd_line_leg = mlines.Line2D([], [], color='orange', linewidth=2, linestyle="--",
                                label="Critical Duration")
            ax.legend(handles=[ra_line_leg, cd_line_leg], fontsize=8)



    # change labels for 1Probe and concurrent so it is clear
    if -1 in sep_labels:
        sep_labels = ['Conc' if i == -1 else i for i in sep_labels]
    if -1 in isi_labels:
        isi_labels = ['Conc' if i == -1 else i for i in isi_labels]
    if 20 in sep_labels:
        sep_labels = ['1pr' if i == 20 else i for i in sep_labels]
    if 20 in isi_labels:
        isi_labels = ['1pr' if i == 20 else i for i in isi_labels]

    # set labels and ticks
    # It seems to know when to use sep_labels and when to use isi_labels
    # if x_isi_y_sep:
    ax.set_xticks(isi_array)
    ax.set_xticklabels(isi_labels)
    ax.set_xlabel(x_label)

    ax.set_yticks(sep_array)
    ax.set_yticklabels(sep_labels)
    ax.set_ylabel(y_label)

    ax.set_zlabel('threshold')

    if fig_title is not None:
        figure_title = fig_title
    plt.suptitle(figure_title)

    # change viewing angle:  default elev=30, azim=300 (counterclockwise on z axis)
    if my_elev is not None or my_rotation is not None:
        ax.view_init(elev=my_elev, azim=my_rotation)
        print(f"elev: {my_elev}, azim: {my_rotation}")
    else:
        print('default ax.azim {}'.format(ax.azim))
        print('default ax.elev {}'.format(ax.elev))

    if save_path is not None:
        if save_name is not None:
            plt.savefig(f'{save_path}/{save_name}.png')

    print('*** finished plot_thr_3dsurface() ***')

    return fig



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
        isi_name_list = ['Conc', 'ISI0', 'ISI2', 'ISI4',
                         'ISI6', 'ISI9', 'ISI12', 'ISI24']

    if x_tick_vals is None:
        x_tick_vals = [-18, -6, -3, -2, -1, 0, 1, 2, 3, 6, 18, 20]

    if x_tick_labels is None:
        x_tick_labels = [-18, -6, -3, -2, -1, 0, 1, 2, 3, 6, 18, '1\nprobe']

    # check column name for x_values
    if 'sep' in list(mean_df.columns):
        mean_df = mean_df.rename(columns={'sep': 'separation'})
        thr1_df = thr1_df.rename(columns={'sep': 'separation'})
        thr2_df = thr2_df.rename(columns={'sep': 'separation'})

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
                         x='separation', y=isi_name_list[ax_counter],
                         color=my_colours[ax_counter],
                         linewidth=2, linestyle="dotted", markers=True)

            # stair1: CW probe jumps only
            sns.lineplot(ax=axes[row_idx, col_idx], data=thr1_df,
                         x='separation', y=isi_name_list[ax_counter],
                         color=my_colours[ax_counter],
                         linewidth=.5, marker="v")

            # stair2: CCW probe jumps only
            sns.lineplot(ax=axes[row_idx, col_idx], data=thr2_df,
                         x='separation', y=isi_name_list[ax_counter],
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
            plt.savefig(os.path.join(save_path, save_name))

    return fig



def plot_n_sep_thr_w_scatter(all_thr_df, thr_col='probeLum', exp_ave=False,
                             ra_cd_v_line=None,
                             fig_title=None,
                             save_name=None, save_path=None, verbose=True):
    """
    Function to make a page with seven axes showing the threshold for each separation,
    and an eighth plot showing all separations.

    :param all_thr_df: Dataframe with thresholds from multiple runs or participants
                        (e.g., MASTER_thresholds.csv or MASTER_exp_all_thr.csv)
    :param thr_col: Column to extract threshold values from, expects 'probeLum' or 'newLum'.
    :param exp_ave: Use True for experiment thr from multiple participants or
                    False for thr from multiple runs from a single participant.
    :param fig_title: Title for figure.
    :param save_name: File name to save plot.
    :param save_path: Directory to save plot into.
    :param verbose: print progress to screen,

    :return: Figure
    """
    print('\n*** running plot_n_sep_thr_w_scatter()***')
    if type(all_thr_df) is str:
        if os.path.isfile(all_thr_df):
            all_thr_df = pd.read_csv(all_thr_df)
    # Average over experiment or participant (with or without participant name)
    if exp_ave is True:
        ave_over = 'Exp'
        long_df_idx_col = 'p_stack_sep'
    elif type(exp_ave) == str:
        ave_over = exp_ave
        long_df_idx_col = 'stack'
    else:
        ave_over = 'P'
        long_df_idx_col = 'stack'

    if not fig_title:
        fig_title = f'{ave_over} average thresholds per separation'

    # if there are any instances of "1Probe" in the separation column, change to 20
    if 'separation' in list(all_thr_df.columns):
        if '1Probe' in list(all_thr_df['separation']):
            all_thr_df['separation'] = all_thr_df['separation'].replace('1Probe', 20)
            all_thr_df['separation'] = all_thr_df['separation'].astype(int)
    elif all_thr_df.index.name == 'separation':
        if '1Probe' in list(all_thr_df.index):
            all_thr_df.index = all_thr_df.index.replace('1Probe', 20)
            all_thr_df.index = all_thr_df.index.astype(int)
    elif 'separation' in list(all_thr_df.index.names):
        if '1Probe' in list(all_thr_df.index.get_level_values('separation')):
            # get index level for separation
            sep_idx_level = all_thr_df.index.names.index('separation')
            print(f"sep_idx_level: {sep_idx_level}")

            # replace instances of 1Probe in list of separation values then replace all values
            sep_idx_level_vals = list(all_thr_df.index.levels[sep_idx_level])
            sep_idx_level_vals = [int(20) if x == '1Probe' else int(x) for x in sep_idx_level_vals]
            all_thr_df.index.set_levels(sep_idx_level_vals, level=sep_idx_level, inplace=True)
    if verbose:
        print(f'input: all_thr_df: \n{all_thr_df}')

    # just_thr_df = all_thr_df.loc[:, 'ISI 999':]
    all_thr_cols_list = list(all_thr_df.columns)
    if 'Concurrent' in all_thr_cols_list:
        just_thr_df = all_thr_df.loc[:, 'Concurrent':]
    elif 'Conc' in all_thr_cols_list:
        just_thr_df = all_thr_df.loc[:, 'Conc':]
    elif 'ISI_-1' in all_thr_cols_list:
        cond_idx = all_thr_cols_list.index('ISI_-1')
        just_thr_df = all_thr_df.iloc[:, cond_idx:]
    elif 'ISI -1' in all_thr_cols_list:
        cond_idx = all_thr_cols_list.index('ISI -1')
        just_thr_df = all_thr_df.iloc[:, cond_idx:]
    else:
        just_thr_df = all_thr_df.iloc[:, 2:]


    min_thr = just_thr_df.min().min()
    max_thr = just_thr_df.max().max()
    if verbose:
        print(f'just_thr_df:\n{just_thr_df}')
        print(f'min_thr: {min_thr}; max_thr: {max_thr}')

    # convert wide_df to long for getting means and standard error.
    if verbose:
        print("making long_fig_df")
        print(f"all_thr_df:\n{all_thr_df}")
        print(f"thr_col: {thr_col}")
        print(f"long_df_idx_col: {long_df_idx_col}")
    if long_df_idx_col == 'p_stack_sep' and 'p_stack_sep' not in list(all_thr_df.columns):
        if 'stack' in list(all_thr_df.columns):
            long_df_idx_col = ['index', 'stack']
        else:
            raise ValueError(f"long_df_idx_col: {long_df_idx_col} not in all_thr_df.columns: {list(all_thr_df.columns)}")
        if verbose:
            print(f"long_df_idx_col: {long_df_idx_col}")

    long_fig_df = make_long_df(all_thr_df, thr_col=thr_col, idx_col=long_df_idx_col)
    if verbose:
        print(f'long_fig_df:\n{long_fig_df}')

    # can't sort sep list if it contains strings (e.g. 1Probe)
    sep_list = sorted(list(long_fig_df['separation'].unique()))
    print(f'sep_list:\n{sep_list}')

    isi_names_list = sorted(list(long_fig_df['ISI'].unique()))
    print(f'isi_labels:\n{isi_names_list}')
    if type(isi_names_list[0]) == str:
        if 'ISI ' in isi_names_list[0]:
            isi_vals_list = [int(i.strip('ISI ')) for i in isi_names_list]
        elif 'ISI_' in isi_names_list[0]:
            isi_vals_list = [int(i.strip('ISI_')) for i in isi_names_list]
    isi_vals_list = sorted(isi_vals_list)
    isi_vals_list = ['Conc' if i == -1 else i for i in isi_vals_list]

    if len(sep_list) > 1:

        my_colours = fig_colours(len(sep_list))

        # get configuration of subplots
        n_plots = len(sep_list) + 1
        n_rows, n_cols = get_n_rows_n_cols(n_plots)
        print(f"n_plots: {n_plots}, n_rows: {n_rows}, n_cols: {n_cols}, empty: {(n_rows * n_cols) - n_plots}")

        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(3 * n_cols, 3 * n_rows))
        ax_counter = 0

        for row_idx, row in enumerate(axes):
            for col_idx, ax in enumerate(row):

                # for the first n-1 plots...
                if ax_counter < len(sep_list):

                    fig.suptitle(fig_title)
                    this_sep = sep_list[ax_counter]
                    sep_df = long_fig_df[long_fig_df['separation'] == this_sep]
                    print(f"sep_df (this_sep: {this_sep}):\n{sep_df}")

                    # show individual data points
                    sns.stripplot(data=sep_df, x='ISI', y=thr_col,
                                  ax=axes[row_idx, col_idx],
                                  color=my_colours[ax_counter],
                                  size=2, jitter=True,
                                  alpha=.5)

                    # show mean and error bars
                    sns.pointplot(ax=axes[row_idx, col_idx],
                                  data=sep_df, x='ISI', y=thr_col,
                                  estimator=np.mean, errorbar='se',
                                  markers='.', errwidth=2, capsize=.1,
                                  color=my_colours[ax_counter])

                    if this_sep == 20:
                        this_sep = '1Probe'

                    ax.set_title(f'Separation = {this_sep}')
                    ax.set_xticklabels(isi_vals_list)
                    ax.set_ylim([min_thr - 2, max_thr + 2])

                    # break axis before ISI 24.
                    # todo: this currently chops the break for some plots
                    if 24 in isi_vals_list:
                        break_axis_before_label(ax=axes[row_idx, col_idx],
                                                label=24, label_list=isi_vals_list,
                                                subplot=True)

                    # add vertical grey dashed line for Ricco's area or Bloch critical duration.
                    if ra_cd_v_line:
                        ax.axvline(ra_cd_v_line, linestyle='--', color='lightgrey')

                elif ax_counter == len(sep_list):

                    # show individual data points
                    sns.stripplot(data=long_fig_df, x='ISI', y=thr_col,
                                  hue='separation',
                                  ax=axes[row_idx, col_idx],
                                  dodge=True, jitter=True, size=2,
                                  palette=my_colours,
                                  alpha=.5)

                    # show means and error bars
                    sns.pointplot(ax=axes[row_idx, col_idx],
                                  data=long_fig_df, x='ISI', y=thr_col,
                                  hue='separation',
                                  estimator=np.mean, errorbar='se',
                                  markers='.', dodge=.3,
                                  errwidth=2, capsize=.1,
                                  palette=my_colours)

                    ax.set_ylim([min_thr - 2, max_thr + 2])
                    ax.set_xticklabels(isi_vals_list)
                    ax.set_title(f'All Separation conditions')

                    # break axis before ISI 24.
                    # todo: this currently chops the break for some plots
                    if 24 in isi_vals_list:
                        break_axis_before_label(ax=axes[row_idx, col_idx],
                                                label=24, label_list=isi_vals_list,
                                                subplot=True)

                    # add vertical grey dashed line for Ricco's area or Bloch critical duration.
                    if ra_cd_v_line:
                        ax.axvline(ra_cd_v_line, linestyle='--', color='lightgrey')

                    # no legend (I struggled to find this command for a while)
                    ax.legend([], [], frameon=False)

                else:  # if there are more subplots than data, remove the axes
                    ax.set_axis_off()

                ax_counter += 1

        if n_rows < 3:
            plt.tight_layout()
        else:
            plt.subplots_adjust(hspace=.5)

    else:  # if just 1 sep value, just a single plot (don't need totals plot)

        fig, ax = plt.subplots()

        fig.suptitle(fig_title)
        this_sep = sep_list[0]
        sep_df = long_fig_df[long_fig_df['separation'] == this_sep]

        print(f"sep_df:\n{sep_df}")

        # show individual data points
        sns.stripplot(data=sep_df, x='ISI', y=thr_col,
                      size=5, jitter=True,
                      alpha=.5)

        # show mean and error bars
        sns.pointplot(data=sep_df, x='ISI', y=thr_col,
                      estimator=np.mean, errorbar='se',
                      markers='.', errwidth=2, capsize=.1,
                      )

        if this_sep == 20:
            this_sep = '1Probe'

        ax.set_title(f'Separation = {this_sep}')
        ax.set_xticklabels(isi_vals_list)
        ax.set_ylim([min_thr - 2, max_thr + 2])
        print(f"y_lim: {min_thr - 2} : {max_thr + 2}")

        # add vertical grey dashed line for Ricco's area or Bloch critical duration.
        if ra_cd_v_line:
            ax.axvline(ra_cd_v_line, linestyle='--', color='lightgrey')
 
    if save_path:
        if save_name:
            print(f"saving fig to: {os.path.join(save_path, save_name)}")
            plt.savefig(os.path.join(save_path, save_name))

    print('\n*** finished plot_n_sep_thr_w_scatter()***')

    return fig


def a_data_extraction(p_name, run_dir, isi_list, save_all_data=True, verbose=True):

    """
    This script is a python version of Martin's first MATLAB analysis scripts, described below.

    a_data_extraction.m: Once a participant has completed a run of all ISIs,
        this script gets all their data into one file, and sorts each isi by stair.

    :param p_name: participant's name as used to save csv files.  e.g., if the
            file is .../aa_output.csv, participant name is 'aa'.
    :param run_dir: directory where isi folders are stored.
    :param isi_list: List of isi values, may differ between experiments.
    :param save_all_data: If True, will save all_data_df as a xlsx.
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
                        8: 2, 9: 1, 10: 2, 11: 1, 12: 2, 13: 1, 14: 2, 0: 2}

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
            print(f"loaded csv ({list(this_isi_df.columns)}):\n{this_isi_df.head()}")

        # remove any Unnamed columns
        if any("Unnamed" in i for i in list(this_isi_df.columns)):
            unnamed_col = [i for i in list(this_isi_df.columns) if "Unnamed" in i][0]
            this_isi_df.drop(unnamed_col, axis=1, inplace=True)

        # add isi column for multi-indexing
        this_isi_df.insert(0, 'ISI', isi)

        # add in group column to differentiate identical stairs (e.g., 1&2, 3&4 etc)
        stack_list = this_isi_df['stair'].map(stair_group_dict)
        this_isi_df.insert(1, 'group', stack_list)

        # add in separation column mapped from stair_sep_dict
        sep_list = this_isi_df['stair'].map(stair_sep_dict)
        if 'separation' not in list(this_isi_df.columns):
            this_isi_df.insert(2, 'separation', sep_list)

        # sort by group, stair, original trial number
        trial_num_header = 'total_nTrials'
        if 'trial_number' in list(this_isi_df.columns):
            trial_num_header = 'trial_number'
        trial_numbers = list(this_isi_df[trial_num_header])
        this_isi_df = this_isi_df.sort_values(by=['group', 'stair', trial_num_header])
        this_isi_df.insert(0, 'srtd_trial_idx', trial_numbers)

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
        save_name = 'RUNDATA-sorted.xlsx'
        save_excel_path = os.path.join(run_dir, save_name)
        if verbose:
            print(f"\nsaving all_data_df to save_excel_path:\n{save_excel_path}")
        all_data_df.to_excel(save_excel_path, index=False)

    print("\n***finished a_data_extraction()***\n")

    return all_data_df



def a_data_extraction_sep(participant_name, run_dir, sep_dirs, save_all_data=True, verbose=True):

    """
    This script is a python version of Martin's first MATLAB analysis scripts, described below.

    a_data_extraction.m: Once a participant has completed a run of all ISIs,
        this script gets all their data into one file, and sorts each isi by stair.

    :param participant_name: participant's name as used to save csv files.  e.g., if the
            file is .../aa_output.csv, participant name is 'aa'.
    :param run_dir: path to directory containing sep_dirs.
    :param sep_dirs: list of directory names for different separations, where output csvs are stored.
    :param save_all_data: If True, will save all_data_df as a xlsx.
    :param verbose: If True, will print progress to screen.

    :return: RUNDATA-sorted.xlsx: A pandas DataFrame with n xlsx file of all data for one run of all separations.
    """

    print("\n***running a_data_extraction_sep()***")

    # empty array to append info into
    all_sep_data = []

    # loop through list of separation folders (containg output files).
    for sep_dir in sep_dirs:
        filepath = os.path.join(run_dir, sep_dir)
        print(f"sep_dir: {sep_dir}")

        # there are a couple of possible output file naming conventions, so try both.
        try:
            p_name = f'{participant_name}_output'  # use this one
            output_df = pd.read_csv(os.path.join(filepath, f'{p_name}.csv'))
            print("\tfound p_name_output.csv")
        except:
            # try with run number. (last character(s) of run_dir, after '_'.
            run_number = run_dir.split('_')[-1]
            p_name = f'{participant_name}_{run_number}_output'  # use this one
            output_df = pd.read_csv(os.path.join(filepath, f'{p_name}.csv'))
            print("\tfound p_name_run_number_output.csv")


        # remove any Unnamed columns
        if any("Unnamed" in i for i in list(output_df.columns)):
            unnamed_col = [i for i in list(output_df.columns) if "Unnamed" in i][0]
            output_df.drop(unnamed_col, axis=1, inplace=True)

        all_sep_data.append(output_df)

    run_data_df = pd.concat(all_sep_data)
    run_data_df = run_data_df.sort_values(by=['step', 'trial_number', 'ISI', 'separation'])
    # print(f"run_data_df ({run_data_df.shape}):\n{run_data_df}")

    if save_all_data:
        save_name = 'RUNDATA-sorted.xlsx'
        save_excel_path = os.path.join(run_dir, save_name)
        if verbose:
            print(f"\nsaving all_data_df to save_excel_path:\n{save_excel_path}")
        run_data_df.to_excel(save_excel_path, index=False)


    print("\n***finished a_data_extraction_sep()***\n")

    return run_data_df


def b3_plot_staircase(all_data_path, thr_col='newLum', resp_col='trial_response',
                      show_plots=True, save_plots=True, verbose=True):
    # todo: add variable to load thresholds csv and mark psignifit threshold on staircase plot.

    """
    b3_plot_staircase.m: staircases-ISIxxx.png: xxx corresponds to isi conditions.
    Eight figure (8 isi conditions) with seven panels on each (7 probes separation
    conditions) showing the Luminance value of two staircases as function of
    trial number. Eighth panel shows psignifit thr per sep condition.

    :param all_data_path: path to the all_data xlsx file.
    :param thr_col: (default newLum) name of the column showing the threshold
        (e.g., varied by the staircase).  Original was probeLum.
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
    if xlsx_name[-3:] == 'csv':
        all_data_df = pd.read_csv(all_data_path)
    else:
        all_data_df = pd.read_excel(all_data_path, engine='openpyxl')

    # get list of isi and stair values to loop through
    stair_list = all_data_df['stair'].unique()
    isi_list = all_data_df['ISI'].unique()
    sep_list = all_data_df['separation'].unique()
    # get isi string for column names
    isi_name_list = ['Conc' if i == -1 else f'isi{i}' for i in isi_list]
    sep_name_list = ['1pr' if i == 99 else f'sep{i}' for i in sep_list]

    trials, columns = np.shape(all_data_df)
    trials_per_stair = int(trials / len(isi_list) / len(stair_list))

    if verbose:
        print(f"all_data_df ({list(all_data_df.columns)}):\n{all_data_df.head()}")
        print(f"{len(isi_list)} isi values and {len(stair_list)} stair values")
        print(f"isi_list: {isi_list}")
        print(f"isi_name_list: {isi_name_list}")
        print(f"stair_list: {stair_list}")
        print(f"sep_name_list: {sep_name_list}")
        print(f"trials_per_stair: {trials_per_stair}")

    '''the eighth plot is the psignifit thr for each sep (+sep, -sep and mean).
    get data from psignifit_thresholds.csv and reshape here'''
    thr_csv_name = os.path.join(save_path, 'psignifit_thresholds.csv')

    psignifit_thr_df = pd.read_csv(thr_csv_name)
    if verbose:
        print(f'\npsignifit_thr_df:\n{psignifit_thr_df}')

    # remove extra columns
    if 'stair' in list(psignifit_thr_df.columns):
        psignifit_thr_df = psignifit_thr_df.drop(['stair'], axis=1)

    if 'group' in list(psignifit_thr_df.columns):
        psignifit_thr_df = psignifit_thr_df.drop(['group'], axis=1)

    if 'separation' in list(psignifit_thr_df.columns):
        # test_sep_list = psignifit_thr_df.pop(['separation'], axis=1)
        sep_list = psignifit_thr_df.pop('separation').tolist()
        sep_list = [20 if i == 99 else i for i in sep_list]
    else:
        sep_list = [18, 6, 3, 2, 1, 0, 20]
    print(f'sep_list: {sep_list}')

    psignifit_thr_df.columns = isi_name_list

    # split into pos_sep, neg_sep and mean of pos and neg.
    psig_odd_stair_df, psig_even_stair_df = split_df_alternate_rows(psignifit_thr_df)
    psig_thr_mean_df = pd.concat([psig_odd_stair_df, psig_even_stair_df]).groupby(level=0).mean()

    # add sep column in
    rows, cols = psig_thr_mean_df.shape
    if len(sep_list) == rows * 2:
        # takes every other item
        sep_list = sep_list[::2]
    print(f'sep_list: {sep_list}')

    psig_thr_mean_df.insert(0, 'separation', sep_list)
    psig_odd_stair_df.insert(0, 'separation', sep_list)
    psig_even_stair_df.insert(0, 'separation', sep_list)
    if verbose:
        print(f'\npsig_odd_stair_df:\n{psig_odd_stair_df}')
        print(f'\npsig_even_stair_df:\n{psig_even_stair_df}')
        print(f'\npsig_thr_mean_df:\n{psig_thr_mean_df}')


    # make empty arrays to save reversal n_reversals
    n_reversals_np = np.zeros(shape=[len(stair_list), len(isi_list)])

    # loop through isi values
    for isi_idx, isi in enumerate(isi_list):

        # get df for this isi only
        isi_df = all_data_df[all_data_df['ISI'] == isi]
        print(f"isi_df:\n{isi_df}")


        # psignifit series for this isi only
        isi_name = isi_name_list[isi_idx]
        print(f'\nisi_name: {isi_name}')
        psig_odd_isi_S = psig_odd_stair_df.loc[:, ['separation', isi_name]]
        psig_even_isi_S = psig_even_stair_df.loc[:, ['separation', isi_name]]
        print(f"psig_odd_isi_S:\n{psig_odd_isi_S}")
        print(f"psig_even_isi_S:\n{psig_even_isi_S}")

        # initialize 8 plot figure
        # # this is a figure showing n_reversals per staircase condition.
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12, 6))
        ax_counter = 0

        for row_idx, row in enumerate(axes):
            for col_idx, ax in enumerate(row):
                print(f'\nrow: {row_idx}, col: {col_idx}')

                # for the first seven plots...
                if ax_counter < 7:
                    # # get pairs of stairs (e.g., [[18, -18], [6, -6], ...etc)
                    stair_odd = (ax_counter * 2) + 1  # 1, 3, 5, 7, 9, 11, 13
                    stair_odd_df = isi_df[isi_df['stair'] == stair_odd]
                    stair_odd_df.insert(0, 'step', list(range(trials_per_stair)))
                    final_lum_odd = stair_odd_df.loc[stair_odd_df['step'] == trials_per_stair - 1, thr_col].item()
                    n_reversals_odd = trials_per_stair - stair_odd_df[resp_col].sum()

                    stair_even = (ax_counter + 1) * 2  # 2, 4, 6, 8, 10, 12, 14
                    stair_even_df = isi_df[isi_df['stair'] == stair_even]
                    stair_even_df.insert(0, 'step', list(range(trials_per_stair)))
                    final_lum_even = stair_even_df.loc[stair_even_df['step'] == trials_per_stair - 1, thr_col].item()
                    n_reversals_even = trials_per_stair - stair_even_df[resp_col].sum()

                    # append n_reversals to n_reversals_np to save later.
                    n_reversals_np[stair_odd - 1, isi_idx] = n_reversals_odd
                    n_reversals_np[stair_even - 1, isi_idx] = n_reversals_even

                    # # psignifit threshold
                    psig_odd_thr = psig_odd_isi_S[isi_name][ax_counter]
                    psig_even_thr = psig_even_isi_S[isi_name][ax_counter]

                    if verbose:
                        print(f'\nstair_odd_df (stair={stair_odd}, isi_name={isi_name}:\n{stair_odd_df.head()}')
                        print(f"final_lum_odd: {final_lum_odd}")
                        print(f"n_reversals_odd: {n_reversals_odd}")
                        print(f'\nstair_even_df (stair={stair_even}, isi_name={isi_name}:\n{stair_even_df.tail()}')
                        print(f"final_lum_even: {final_lum_even}")
                        print(f"n_reversals_even: {n_reversals_even}")

                        print(f"psig_odd_thr: {psig_odd_thr}")
                        print(f"psig_even_thr: {psig_even_thr}")


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
                    # line for final Luminance
                    ax.axhline(y=final_lum_odd, linestyle="dotted", color='tab:red')
                    ax.axhline(y=psig_odd_thr, linestyle="dashed", color='tab:brown')
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
                    ax.axhline(y=final_lum_even, linestyle="dotted", color='tab:blue')
                    ax.axhline(y=psig_even_thr, linestyle="dashed", color='royalblue')

                    ax.text(x=0.25, y=0.8, s=f'{n_reversals_even} reversals',
                            color='tab:blue',
                            # needs transform to appear with rest of plot.
                            transform=ax.transAxes, fontsize=12)

                    ax.set_title(f'{isi_name} {sep_name_list[ax_counter]}')
                    ax.set_xticks(np.arange(0, trials_per_stair, 5))
                    ax.set_ylim([0, 110])

                else:
                    st1 = mlines.Line2D([], [], color='tab:red',
                                        marker='v',
                                        markersize=5, label='group1')
                    st1_last_val = mlines.Line2D([], [], color='tab:red',
                                                 linestyle="dotted", marker=None,
                                                 label='g1_last_val')
                    st1_psig = mlines.Line2D([], [], color='tab:brown',
                                             linestyle="dashed", marker=None,
                                             label='g1_psig_thr')
                    st2 = mlines.Line2D([], [], color='tab:blue',
                                        marker='o',
                                        markersize=5, label='group2')
                    st2_last_val = mlines.Line2D([], [], color='tab:blue',
                                                 linestyle="dotted", marker=None,
                                                 label='g2_last_val')
                    st2_psig = mlines.Line2D([], [], color='royalblue',
                                             linestyle="dashed", marker=None,
                                             label='g2_psig_thr')
                    ax.legend(handles=[st1, st1_last_val, st1_psig, st2, st2_last_val, st2_psig],
                              fontsize=8, loc='center')
                    print('empty plot')

                ax_counter += 1

        plt.tight_layout()

        # show and close plots
        if save_plots:
            save_fig_path = os.path.join(save_path, f'staircases_{isi_name}.png')
            print(f'saving fig to: {save_fig_path}')
            plt.savefig(save_fig_path)

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

    n_reversal_save_name = os.path.join(save_path, 'n_reversals.csv')
    n_reversals_df.to_csv(n_reversal_save_name, index=False)

    print("\n***finished b3_plot_staircases()***\n")



def c_plots(save_path, thr_col='newLum', show_plots=True, verbose=True):

    """
    5. c_plots.m: uses psignifit_thresholds.csv and outputs plots.

    figures:
            data.png: threshold luminance as function of probe separation.
                      Positive separation values only, all ISIs on same axis.
                      Use plot_pos_sep_and_1probe()
                      
            dataDivOneProbe: threshold luminance as function of probe separation.
                      Positive separation values only, all ISIs on same axis.
                      Use plot_pos_sep_and_1probe(one_probe=False)

                      
            runs.png: threshold luminance as function of probe separation, 
                      Positive and negative separation values (batman plots), 
                      one panel one isi condition.
                      use eight_batman_plots()
                      
    :param save_path: path to run dir containing psignifit_thresholds.csv, where plots will be saved.
    :param thr_col: column for threshold (e.g., 'newLum', 'probeLum')
    :param show_plots: Default True
    :param verbose: Default True.
    """

    print("\n*** running c_plots() ***\n")

    isi_name_list = ['Conc', 'ISI0', 'ISI2', 'ISI4', 'ISI6', 'ISI9', 'ISI12', 'ISI24']
    sym_sep_list = [-18, -6, -3, -2, -1, 0, 1, 2, 3, 6, 18, 20]
    sym_sep_tick_labels = [-18, -6, -3, -2, -1, 0, 1, 2, 3, 6, 18, '1\nprobe']
    pos_sep_list = [0, 1, 2, 3, 6, 18, 20]

    # load df mean of last n luminance values (14 stairs x 8 isi).
    thr_csv_name = os.path.join(save_path, 'psignifit_thresholds.csv')
    psig_thr_df = pd.read_csv(thr_csv_name)
    if verbose:
        print(f'psig_thr_df:\n{psig_thr_df}')

    psig_thr_df = psig_thr_df.drop(['stair'], axis=1)
    if 'group' in list(psig_thr_df.columns):
        group_col_s = psig_thr_df.pop('group')
    if 'separation' in list(psig_thr_df.columns):
        sep_col_s = psig_thr_df.pop('separation')
    psig_thr_df.columns = isi_name_list

    # lastN_pos_sym_np has values for 1-indexed stairs [1, 3, 5, 7, 9, 11, 9, 7, 5, 3, 1, 13]
    # but the rows I select are zero indexed, use rows [0, 2, 4, 6, 8, 10, 8, 6, 4, 2, 0, 12]
    # these correspond to separation values:          [18, 6, 3, 2, 1, 0, 1, 2, 3, 6, 18, 99]
    if verbose:
        print('\npreparing data for batman plots')
    pos_sym_indices = [0, 2, 4, 6, 8, 10, 8, 6, 4, 2, 0, 12]
    psig_pos_sym_sep_df = psig_thr_df.iloc[pos_sym_indices]
    psig_pos_sym_sep_df.reset_index(drop=True, inplace=True)

    # lastN_neg_sym_np has values for 1-indexed stairs   [2,  4,  6,  8, 10, 12, 10, 8,  6,  4,  2,  14]
    # but the rows I select are zero indexed, use rows   [1,  3,  5,  7, 9,  11, 9,  7,  5,  3,  1,  13]
    # these correspond to sep values:                  [-18, -6, -3, -2, -1, 0, -1, -2, -3, -6, -18, 99]
    neg_sym_indices = [1, 3, 5, 7, 9, 11, 9, 7, 5, 3, 1, 13]
    psig_neg_sym_sep_df = psig_thr_df.iloc[neg_sym_indices]
    psig_neg_sym_sep_df.reset_index(drop=True, inplace=True)

    # mean of pos and neg sep values [18, 6, 3, 2, 1, 0, 1, 2, 3, 6, 18, 99]
    psig_sym_thr_mean_df = pd.concat([psig_pos_sym_sep_df, psig_neg_sym_sep_df]).groupby(level=0).mean()

    # subtract the dfs from each other, then for each column get the sum of abs values
    diff_val = np.sum(abs(psig_pos_sym_sep_df - psig_neg_sym_sep_df), axis=0)
    # take the mean of these across all ISIs to get single value
    mean_diff_val = float(np.mean(diff_val))

    # add sep column into dfs
    psig_sym_thr_mean_df.insert(0, 'separation', sym_sep_list)
    psig_pos_sym_sep_df.insert(0, 'separation', sym_sep_list)
    psig_neg_sym_sep_df.insert(0, 'separation', sym_sep_list)

    if verbose:
        print(f'\npsig_pos_sym_sep_df:\n{psig_pos_sym_sep_df}')
        print(f'\npsig_neg_sym_sep_df:\n{psig_neg_sym_sep_df}')
        print(f'\npsig_sym_thr_mean_df:\n{psig_sym_thr_mean_df}')
        print(f'\ndiff_val:\n{diff_val}')
        print(f'\nmean_diff_val: {mean_diff_val}')

    # # Figure1 - runs-{n}lastValues
    # this is a figure with one axis per isi, showing neg and pos sep
    # (e.g., -18:18) - eight batman plots
    fig_title = f'MIRRORED Psignifit thresholds per ISI. ' \
                f'(mean diff: {round(mean_diff_val, 2)})'
    fig1_savename = f'MIRRORED_runs.png'

    eight_batman_plots(mean_df=psig_sym_thr_mean_df,
                       thr1_df=psig_pos_sym_sep_df,
                       thr2_df=psig_neg_sym_sep_df,
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
    if 'separation' not in list(psig_thr_df.columns):
        sep_list = [19 if i == 20 else i for i in sep_col_s.tolist()]
        psig_thr_df.insert(1, 'separation', sep_list)
    print(f'\nmaking plots from group1 and group2 data with psig_thr_df:\n{psig_thr_df}')

    for group in [1, 2]:

        group_plot_df = psig_thr_df[psig_thr_df['group'] == group]
        group_plot_df = group_plot_df.drop(['group'], axis=1)
        print(f'\nrunning group{group} with group_plot_df:\n{group_plot_df}')

        # # FIGURE3 - 'data-{n}lastValues.png' - all ISIs on same axis, pos sep only, plus single
        # # use plot_pos_sep_and_1probe()
        fig3_save_name = f'data.png'
        fig_3_title = 'All ISIs and separations'

        if group is not None:
            fig3_save_name = f'g{group}_data.png'
            fig_3_title = f'g{group} All ISIs and separations'

        plot_pos_sep_and_1probe(pos_sep_and_1probe_df=group_plot_df,
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
        # # # use plot_pos_sep_and_1probe(one_probe=False)
        # # each sep row in pos_sep_df is divided by one_probe_df.
        fig4_save_name = f'dataDivOneProbe.png'
        fig4_title = f'two-probe conditions divided by one-probe conditions'

        if group is not None:
            fig4_save_name = f'g{group}_dataDivOneProbe.png'
            fig4_title = f'g{group} two-probe conditions divided by one-probe conditions'

        pos_sep_df, one_probe_df = split_df_into_pos_sep_df_and_1probe_df(group_plot_df)
        pos_sep_arr = pos_sep_df.to_numpy()
        one_probe_arr = one_probe_df[thr_col].to_numpy()
        div_by_1probe_arr = (pos_sep_arr.T / one_probe_arr[:, None]).T
        div_by_1probe_df = pd.DataFrame(div_by_1probe_arr, columns=isi_name_list)
        div_by_1probe_df.insert(0, 'separation', pos_sep_list[:-1])
        div_by_1probe_df.set_index('separation', inplace=True)
        print(f'div_by_1probe_df:\n{div_by_1probe_df}')

        plot_pos_sep_and_1probe(div_by_1probe_df,
                                thr_col=thr_col,
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




def d_average_participant(root_path, run_dir_names_list,
                          error_type=None,
                          trim_n=None,
                          isi_names_list=None,
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
    :param error_type: Default: None. Can pass sd or se for standard deviation or error.
    :param trim_n: default None.  If int is passed, will call function trim_n_high_n_low(),
            which trims the n highest and lowest values.
    :param isi_names_list: List of ISI column names.
    :param ave_type: default = 'thresholds', which will use experiment thresholds.
        The other option if diff_from_conc, which will averager over difference from conconcurrent scores.
    :param verbose: Default True, print progress to screen

    :returns: ave_psignifit_thr_df: (trimmed?) mean threshold for each separation and ISI.
    """

    print("\n***running d_average_participant()***\n")

    """ part1. Munge data, save master lists and get means etc
     - loop through runs and get each P-next-thresholds and P-reversal4-thresholds
    Make master sheets: MASTER_next_thresh & MASTER_reversal_4_thresh
    Incidentally the MATLAB script didn't specify which reversals data to use,
    although the figures imply Martin used last3 reversals."""

    if isi_names_list is None:
        isi_names_list = ['Conc', 'ISI0', 'ISI2', 'ISI4',
                          'ISI6', 'ISI9', 'ISI12', 'ISI24']

    all_psignifit_list = []
    for run_idx, run_name in enumerate(run_dir_names_list):

        this_psignifit_df = pd.read_csv(os.path.join(root_path, run_name, 'psignifit_thresholds.csv'))

        if verbose:
            print(f'\n{run_idx}. {run_name} - this_psignifit_df:\n{this_psignifit_df}')

        if 'Unnamed: 0' in list(this_psignifit_df):
            this_psignifit_df.drop('Unnamed: 0', axis=1, inplace=True)

        if 'stair' in list(this_psignifit_df):
            this_psignifit_df.drop(columns='stair', inplace=True)

        # split df into group1 and group2
        if 'group' in list(this_psignifit_df):
            psig_g1_df = this_psignifit_df[this_psignifit_df['group'] == 1]
            psig_g1_df.drop(columns='group', inplace=True)
            rows, cols = psig_g1_df.shape
            psig_g1_df.insert(0, 'stack', [run_idx*2] * rows)

            psig_g2_df = this_psignifit_df[this_psignifit_df['group'] == 2]
            psig_g2_df.drop(columns='group', inplace=True)
            psig_g2_df.insert(0, 'stack', [run_idx*2+1] * rows)

            columns_list = ['stack', 'separation'] + isi_names_list
            psig_g1_df.columns = columns_list
            psig_g2_df.columns = columns_list

            if verbose:
                print(f'\npsig_g1_df:\n{psig_g1_df}')
                print(f'\npsig_g2_df:\n{psig_g2_df}')

            all_psignifit_list.append(psig_g1_df)
            all_psignifit_list.append(psig_g2_df)
        else:
            rows, cols = this_psignifit_df.shape
            this_psignifit_df.insert(0, 'stack', [run_idx] * rows)
            if verbose:
                print(f'this_psignifit_df:\n{this_psignifit_df}')

            all_psignifit_list.append(this_psignifit_df)

    # join all stacks (run/group) data and save as master csv
    all_data_psignifit_df = pd.concat(all_psignifit_list, ignore_index=True)

    # check that ISI_-1 is at last position
    all_data_psignifit_df = conc_to_first_isi_col(all_data_psignifit_df)
    all_data_psignifit_df.to_csv(os.path.join(root_path, 'MASTER_psignifit_thresholds.csv'), index=False)
    if verbose:
        print(f'\nall_data_psignifit_df:\n{all_data_psignifit_df}')

    """Part 2: trim highest and lowest values is required and get average vals and errors"""
    # # trim highest and lowest values
    if trim_n is not None:
        trimmed_df = trim_n_high_n_low(all_data_psignifit_df, trim_from_ends=trim_n,
                                       reference_col='separation',
                                       stack_col_id='stack',
                                       verbose=verbose)
        trimmed_df.to_csv(os.path.join(root_path, f'MASTER_TM{trim_n}_thresholds.csv'), index=False)

        get_means_df = trimmed_df
    else:
        get_means_df = all_data_psignifit_df

    # # get means and errors
    groupby_sep_df = get_means_df.drop('stack', axis=1)
    ave_psignifit_thr_df = groupby_sep_df.groupby('separation', sort=True).mean()
    if verbose:
        print(f'\nave_psignifit_thr_df:\n{ave_psignifit_thr_df}')

    if error_type in [False, None]:
        error_bars_df = None
    elif error_type.lower() in ['se', 'error', 'std-error', 'standard error', 'standard_error']:
        error_bars_df = groupby_sep_df.groupby('separation', sort=True).sem()
    elif error_type.lower() in ['sd', 'stdev', 'std_dev', 'std.dev', 'deviation', 'standard_deviation']:
        error_bars_df = groupby_sep_df.groupby('separation', sort=True).std()
    else:
        raise ValueError(f"error_type should be in:\nfor none: [False, None]\n"
                         f"for standard error: ['se', 'error', 'std-error', 'standard error', 'standard_error']\n"
                         f"for standard deviation: ['sd', 'stdev', 'std_dev', 'std.dev', "
                         f"'deviation', 'standard_deviation']")
    if verbose:
        print(f'\nerror_bars_df: ({error_type})\n{error_bars_df}')

    # save csv with average values
    ave_psignifit_thr_df.reset_index(inplace=True, drop=False)
    error_bars_df.reset_index(inplace=True, drop=False)

    if trim_n is not None:
        ave_psignifit_thr_df.to_csv(os.path.join(root_path, f'MASTER_ave_TM{trim_n}_thresh.csv'), index=False)
        error_bars_df.to_csv(os.path.join(root_path, f'MASTER_ave_TM{trim_n}_thr_error_{error_type}.csv'), index=False)
    else:
        ave_psignifit_thr_df.to_csv(os.path.join(root_path, 'MASTER_ave_thresh.csv'), index=False)
        error_bars_df.to_csv(os.path.join(root_path, f'MASTER_ave_thr_error_{error_type}.csv'), index=False)

    # # get difference from concurrent plot.
    # if any(item in ['ISI_-1', 'ISI -1', 'conc', 'Conc', 'Concurrent', 'concurrent']
    #        for item in list(ave_psignifit_thr_df.columns)):
    #     print("making difference from concurrent df")
    #     ave_DfC_df = make_diff_from_conc_df(ave_psignifit_thr_df)
    #     ave_DfC_df.to_csv(os.path.join(root_path, 'MASTER_diff_from_conc.csv'), index=False)

    # else:
    #     print("concurrent not found, so not making a difference-from-concurrent plot.")


    print("\n*** finished d_average_participant()***\n")

    return ave_psignifit_thr_df, error_bars_df



def e_average_exp_data(exp_path, p_names_list,
                       error_type='SE',
                       n_trimmed=None,
                       verbose=True):
    """
    OLD VERSION took the mean of each participant's mean.
    NEW version will take the mean of each participant's trimmed results.
    That is all remaining datapoints after trimming.
    e_average_over_participants: take MASTER_TM2_thresholds.csv in each
    participant folder and make master list: MASTER_exp_all_thr.csv

    Get mean thresholds averaged across all participants saved as
    MASTER_exp_ave_thr.csv

    Save master lists to exp_path.

    :param exp_path: dir containing participant folders
    :param p_names_list: names of participant's folders
    :param error_type: Default: None. Can pass sd or se for standard deviation or error.
    :param n_trimmed: default 2.  If not equal to 2, raise error, as calculating means will be biased
    if not all participants have the same number of datapoints.
    :param verbose: Default true, print progress to screen

    :returns: exp_ave_thr_df: experiment mean threshold for each separation and ISI.
    """
    print("\n***running e_average_exp_data()***\n")

    """ part1. Munge data, save master lists and get means etc
     - loop through participants and get each MASTER_TM2_thresholds.csv
    Make master sheets: MASTER_exp_all_thr and MASTER_exp_ave_thr."""

    all_p_ave_list = []
    for p_idx, p_name in enumerate(p_names_list):

        # ave_df_name = 'MASTER_ave_thresh'
        # if type(n_trimmed) == int:
        #     ave_df_name = f'MASTER_ave_TM{n_trimmed}_thresh'
        # elif type(n_trimmed) == list:
        #     if n_trimmed[p_idx] is not None:
        #         ave_df_name = f'MASTER_ave_TM{n_trimmed[p_idx]}_thresh'
        # this_p_ave_df = pd.read_csv(os.path.join(exp_path, p_name, f'{ave_df_name}.csv'))

        # p_all_df_name = 'MASTER_TM2_thresholds.csv'

        p_all_df_name = 'MASTER_psignifit_thresholds.csv'
        # if type(n_trimmed) == int:
        #     if n_trimmed > 0:
        if type(n_trimmed) == int and n_trimmed > 0:
            p_all_df_name = f'MASTER_TM{n_trimmed}_thresholds.csv'
        elif type(n_trimmed) == list:
            # if n_trimmed[p_idx] is not None:
            if type(n_trimmed[p_idx]) == int and n_trimmed[p_idx] > 0:
                p_all_df_name = f'MASTER_TM{n_trimmed[p_idx]}_thresholds.csv'
        # this_p_ave_df = pd.read_csv(os.path.join(exp_path, p_name, f'{ave_df_name}.csv'))
        this_p_all_df = pd.read_csv(os.path.join(exp_path, p_name, p_all_df_name))

        if verbose:
            print(f'{p_idx}. {p_name} - {p_all_df_name}:\n{this_p_all_df}')

        if 'Unnamed: 0' in list(this_p_all_df):
            this_p_all_df.drop('Unnamed: 0', axis=1, inplace=True)

        this_p_all_df = this_p_all_df.rename(columns={'Conc': 'ISI_-1',
                                                      'ISI0': 'ISI_0',
                                                      'ISI2': 'ISI_2',
                                                      'ISI4': 'ISI_4',
                                                      'ISI6': 'ISI_6',
                                                      'ISI9': 'ISI_9',
                                                      'ISI12': 'ISI_12',
                                                      'ISI24': 'ISI_24',
                                                      'ISI-1': 'ISI_-1',
                                                      })


        rows, cols = this_p_all_df.shape
        this_p_all_df.insert(0, 'participant', [p_name] * rows)

        p_stack_sep_list = [f"{p_name}_{a}_sep{b}" for a, b in
                            zip(this_p_all_df['stack'].tolist(), this_p_all_df['separation'].tolist())]
        this_p_all_df.insert(0, 'p_stack_sep', p_stack_sep_list)


        all_p_ave_list.append(this_p_all_df)

    # join all participants' data and save as master csv
    # all_exp_thr_df = pd.concat(all_p_ave_list, ignore_index=True)
    # all_exp_thr_df.to_csv(os.path.join(exp_path, 'MASTER_exp_all_thr.csv'), index=False)
    exp_all_thr_df = pd.concat(all_p_ave_list, ignore_index=True)
    exp_all_thr_df.to_csv(os.path.join(exp_path, 'MASTER_exp_all_thr.csv'), index=False)

    if verbose:
        print(f'\nexp_all_thr_df:\n{exp_all_thr_df}')

    # # get means and errors
    groupby_sep_df = exp_all_thr_df.drop(['p_stack_sep', 'participant', 'stack'], axis=1)
    exp_ave_thr_df = groupby_sep_df.groupby('separation', sort=True).mean()
    if verbose:
        print(f'\nexp_ave_thr_df:\n{exp_ave_thr_df}')

    if error_type in [False, None]:
        error_bars_df = None
    elif error_type.lower() in ['se', 'error', 'std-error', 'standard error', 'standard_error']:
        error_bars_df = groupby_sep_df.groupby('separation', sort=True).sem()
    elif error_type.lower() in ['sd', 'stdev', 'std_dev', 'std.dev', 'deviation', 'standard_deviation']:
        error_bars_df = groupby_sep_df.groupby('separation', sort=True).std()
    else:
        raise ValueError(f"error_type should be in:\nfor none: [False, None]\n"
                         f"for standard error: ['se', 'error', 'std-error', 'standard error', 'standard_error']\n"
                         f"for standard deviation: ['sd', 'stdev', 'std_dev', 'std.dev', "
                         f"'deviation', 'standard_deviation']")
    if verbose:
        print(f'\nerror_bars_df: ({error_type})\n{error_bars_df}')

    # save csv with average values
    # todo: should this be with index=False?
    exp_ave_thr_df.to_csv(os.path.join(exp_path, 'MASTER_exp_ave_thr.csv'))
    error_bars_df.to_csv(os.path.join(exp_path, f'MASTER_ave_thr_error_{error_type}.csv'))

    print("\n*** finished e_average_exp_data()***\n")

    return exp_ave_thr_df, error_bars_df


def make_average_plots(all_df_path, ave_df_path, error_bars_path,
                       thr_col='probeLum',
                       ave_over_n=None,
                       n_trimmed=None,
                       error_type='SE',
                       exp_ave=False,
                       split_1probe=True,
                       ra_cd_size_dict=None,
                       heatmap_annot_fmt='{:.2f}',
                       plt_suffix='',
                       show_plots=True, verbose=True):
    """
    Plots:
    MASTER_ave_thresh saved as ave_thr_all_runs.png
    MASTER_ave_thresh two-probe/one-probe saved as ave_thr_div_1probe.png
    these both have two versions:
    a. x-axis is separation, ISI as different lines
    b. x-axis is ISI, separation as different lines
    Heatmap: with average probe lum for ISI and separation.
    
    :param all_df_path: Path to df with all participant/stack data.
    :param ave_df_path: Path to df with average across all stacks/participants
    :param error_bars_path: Path to df for errors bars with SE/SD associated with averages.
    :param thr_col: Column to extract threshold values from, expects 'probeLum' or 'newLum'.
    :param ave_over_n: Number of values (runs or participants) to average over and get errors.
    :param n_trimmed: Whether averages data has been trimmed.
    :param error_type: 'SE', 'sd' or some other type of error (CI?)
    :param exp_ave: If True, will add 'Exp' to fig titles so its clear these are experiment level results.
        If False will add 'P' for participant level; or use participant_name to identify whose results it is.
    :param split_1probe: If there is 1Probe data, this should be separated
        (e.g., line plots not joined) from other values. Also required for making div_by_1probe plots.
    :param isi_name_list: List of ISI column names.
    :param sep_vals_list: List of separation values (including 99 for 1Probe)
    :param sep_name_list: List of separation names (e.g., '1Probe' if 99 in sep_vals_list).
    :param ra_cd_size_dict: dictionary containing participant Ricco area and Bloch critical duration info.
    :param heatmap_annot_fmt: Number of digits to display in heatmap values.
    :param plt_suffix: Suffix to add to end of plot file names.
    :param show_plots: Whether to display figures once they are made.
    :param verbose: Whether to print progress to screen.
    """
    print("\n*** running make_average_plots()***\n")

    save_path, df_name = os.path.split(ave_df_path)

    # Average over experiment or participant (with or without participant name)
    if type(exp_ave) == str:  # e.g. participant's name
        ave_over = exp_ave
        idx_col = 'stack'
    elif exp_ave is True:
        ave_over = 'Exp'
        idx_col = 'p_stack_sep'
    else:
        ave_over = 'P'
        idx_col = 'stack'

    # if type(all_df_path) is 'pandas.core.frame.DataFrame':
    if isinstance(all_df_path, pd.DataFrame):
        all_df = all_df_path
    else:
        all_df = pd.read_csv(all_df_path)
    all_df = conc_to_first_isi_col(all_df)
    print(f'\nall_df:\n{all_df}')

    # if type(ave_df_path) is 'pandas.core.frame.DataFrame':
    if isinstance(ave_df_path, pd.DataFrame):
        ave_df = ave_df_path
    else:
        ave_df = pd.read_csv(ave_df_path)
    ave_df = conc_to_first_isi_col(ave_df)
    print(f'\nave_df:\n{ave_df}')

    # if type(error_bars_path) is 'pandas.core.frame.DataFrame':
    if isinstance(error_bars_path, pd.DataFrame):
        error_bars_df = error_bars_path
    else:
        error_bars_df = pd.read_csv(error_bars_path)
    error_bars_df = conc_to_first_isi_col(error_bars_df)
    print(f'\nerror_bars_df:\n{error_bars_df}')

    # remove any Unnamed columns
    if any("Unnamed" in i for i in list(all_df.columns)):
        unnamed_col = [i for i in list(all_df.columns) if "Unnamed" in i][0]
        all_df.drop(unnamed_col, axis=1, inplace=True)

    all_df_headers = list(all_df.columns)
    print(f'all_df_headers: {all_df_headers}')

    if 'p_stack_sep' in all_df_headers:
        isi_cols_list = all_df_headers[4:]
    else:
        isi_cols_list = all_df_headers[2:]
    print(f'isi_cols_list: {isi_cols_list}')

    if 'ISI_' in isi_cols_list[-1]:
        print(f"found 'ISI_X' (e.g., '{isi_cols_list[-1][:4]}' joined with '{isi_cols_list[-1][4:]}')")
        isi_vals_list = [-1 if 'conc' in str.lower(i) else int(i[4:]) for i in isi_cols_list]
    elif 'ISI ' in isi_cols_list[-1]:
        print(f"found 'ISI X' (e.g., '{isi_cols_list[-1][:4]}' joined with '{isi_cols_list[-1][4:]}')")
        isi_vals_list = [-1 if 'conc' in str.lower(i) else int(i[4:]) for i in isi_cols_list]
    else:
        print(f"using 'ISIX' (e.g., '{isi_cols_list[-1][:3]}' joined with '{isi_cols_list[-1][3:]}')")
        isi_vals_list = [-1 if 'conc' in str.lower(i) else int(i[3:]) for i in isi_cols_list]

    isi_name_list = ['Conc' if i in ['ISI_-1', 'Concurrent', 'ISI -1', 'ISI-1', -1] else
                     i for i in isi_vals_list]

    sep_vals_list = ave_df['separation'].tolist()
    sep_name_list = ['1Probe' if i == 20 else i for i in sep_vals_list]
    sep_idx_list = list(range(len(sep_vals_list)))

    # if n_trimmed is a list of identical values, replace with int (e.g., [2, 2, 2] becomes 2)
    if isinstance(n_trimmed, list) and len(set(n_trimmed)) == 1:
        n_trimmed = n_trimmed[0]

    if verbose:
        print(f'\nall_df_headers: {all_df_headers}')
        print(f'isi_cols_list: {isi_cols_list}')
        print(f'isi_vals_list: {isi_vals_list}')
        print(f'isi_name_list: {isi_name_list}')
        print(f'sep_vals_list: {sep_vals_list}')
        print(f'sep_name_list: {sep_name_list}')
        print(f'sep_idx_list: {sep_idx_list}')
        print(f'n_trimmed: {n_trimmed}')

    # todo: if '1Probe' in sep_name_list, calculate mean_1Probe_val from all_df
    # get the mean of all thresholds where 'separation' == 20

    # print(f"mean_1Probe_val: {mean_1Probe_val}")



    # todo: if 18 in sep_name_list, extract proba_sum_val

    # print(f"proba_sum_val: {proba_sum_val}")


    # add option to add horizontal line for mean_1Probe_val and proba_sum_val.


    if ra_cd_size_dict:
        # associate RA_size_sep with sep_idx_list number to draw line at correct point of axis.
        ra_size_sep = ra_cd_size_dict['ra_size_sep']
        print(f"\nra_size_sep: {ra_size_sep}")

        if ra_size_sep in sep_vals_list:
            ra_sep_idx = sep_vals_list.index(ra_size_sep)
        else:
            # find idx of sep val below ra_size_sep
            ra_sep_idx_minus1 = len([i for i in sep_vals_list if i < ra_size_sep]) - 1
            print(f"ra_sep_idx_minus1: {ra_sep_idx_minus1}")

            # get size between values either size of ra_size_sep
            sep_vals_gap = sep_vals_list[ra_sep_idx_minus1 + 1] - sep_vals_list[ra_sep_idx_minus1]
            print(f"sep_vals_gap: {sep_vals_gap}")

            # take decimal from end of ra_size_sep
            # decimal_to_add = ra_size_sep % 1
            decimal_to_add = ra_size_sep - sep_vals_list[ra_sep_idx_minus1]
            print(f"decimal_to_add: {decimal_to_add}")

            # scale decimal to size of gap, e.g., if ra_size_sep is 3.5, and it is between values of 3 and 6,
            # then on plots the line should be 1/6th of the way between 3 and 6.
            scaled_dec_to_add = decimal_to_add / sep_vals_gap
            print(f"scaled_dec_to_add: {scaled_dec_to_add}")

            ra_sep_idx = ra_sep_idx_minus1 + scaled_dec_to_add

        print(f"ra_sep_idx: {ra_sep_idx}\n")

        ra_cd_size_dict['ra_sep_idx'] = ra_sep_idx

        if 'ra_CI95_sep' in ra_cd_size_dict.keys():
            ra_CI95_sep = ra_cd_size_dict['ra_CI95_sep']
            print(f"\nra_CI95_sep: {ra_CI95_sep}")

            # convert into appropriate sep_idx units for plotting
            print(f"sep_vals_gap: {sep_vals_gap}")
            ra_sep_idx_CI95 = ra_CI95_sep / sep_vals_gap
            print(f"ra_sep_idx_CI95: {ra_sep_idx_CI95}")

            ra_cd_size_dict['ra_sep_idx_CI95'] = ra_sep_idx_CI95

        cd_size_isi = ra_cd_size_dict['cd_size_isi']
        print(f"\ncd_size_isi: {cd_size_isi}")

        if cd_size_isi in isi_vals_list:
            cd_isi_idx = isi_vals_list.index(cd_size_isi)
        else:
            # find idx of sep val below cd_size_isi
            cd_isi_idx_minus1 = len([i for i in isi_vals_list if i < cd_size_isi]) - 1
            print(f"cd_isi_idx_minus1: {cd_isi_idx_minus1}")

            # get size between values either size of cd_size_isi
            isi_vals_gap = isi_vals_list[cd_isi_idx_minus1 + 1] - isi_vals_list[cd_isi_idx_minus1]
            print(f"isi_vals_gap: {isi_vals_gap}")

            # take decimal from end of cd_size_isi
            # decimal_to_add = cd_size_isi % 1
            decimal_to_add = cd_size_isi - isi_vals_list[cd_isi_idx_minus1]

            print(f"decimal_to_add: {decimal_to_add}")

            # scale decimal to size of gap, e.g., if cd_size_isi is 3.5, and it is between values of 3 and 6,
            # then on plots the line should be 1/6th of the way between 3 and 6.
            scaled_dec_to_add = decimal_to_add / isi_vals_gap
            print(f"scaled_dec_to_add: {scaled_dec_to_add}")

            cd_isi_idx = cd_isi_idx_minus1 + scaled_dec_to_add

        print(f"cd_isi_idx: {cd_isi_idx}\n")

        ra_cd_size_dict['cd_isi_idx'] = cd_isi_idx

        if 'cd_CI95_isi' in ra_cd_size_dict.keys():
            cd_CI95_isi = ra_cd_size_dict['cd_CI95_isi']
            print(f"\ncd_CI95_isi: {cd_CI95_isi}")

            # convert into appropriate isi_idx units for plotting
            print(f"isi_vals_gap: {isi_vals_gap}")
            cd_isi_idx_CI95 = ra_CI95_sep / isi_vals_gap
            print(f"cd_isi_idx_CI95: {cd_isi_idx_CI95}")

            ra_cd_size_dict['cd_isi_idx_CI95'] = cd_isi_idx_CI95

    else:
        ra_cd_size_dict = {'ra_size_sep': None, 'ra_size_deg': None, 'ra_sep_idx': None,
                           'cd_size_isi': None, 'cd_size_ms': None, 'cd_isi_idx': None}

    for k, v in ra_cd_size_dict.items():
        print(f"{k}: {v}")


    """part 3. main Figures (these are the ones saved in the matlab script)
    Fig1: plot average threshold for each ISI and sep.
    Fig2: divide all 2probe conds (pos_sep) by one_probe condition for each participant.
    For both figures there are 2 versions:
        a) Sep on x-axis, different line for each ISI
        b) ISI on x-axis, different line for each Sep"""

    if len(sep_vals_list) == 1:
        print("skipping fig_1a as there is only 1 sep value")
    else:
        print(f"\nfig_1a\n")
        if n_trimmed is not None:
            fig1_title = f'{ave_over} average thresholds across all runs\n(n={ave_over_n}, trim={n_trimmed}, err={error_type}).'
            fig1_savename = f'ave_TM{n_trimmed}_thr_all_runs.png'
        else:
            fig1_title = f'{ave_over} average threshold across all runs\n(n={ave_over_n}, err={error_type})'
            fig1_savename = f'ave_thr_all_runs{plt_suffix}.png'

        plot_1probe_w_errors(fig_df=ave_df, error_df=error_bars_df,
                             split_1probe=split_1probe, jitter=True,
                             error_caps=True, alt_colours=False,
                             legend_names=isi_name_list,
                             x_tick_vals=sep_idx_list,
                             x_tick_labels=sep_name_list,
                             fixed_y_range=False,
                             even_spaced_x=True,
                             ra_cd_v_line=ra_cd_size_dict['ra_sep_idx'],
                             fig_title=fig1_title, save_name=fig1_savename,
                             save_path=save_path, verbose=True)
        if show_plots:
            plt.show()
        plt.close()

        if verbose:
            print('finished fig1a')

    if len(isi_name_list) == 1:
        print("skipping fig_1b as there is only 1 ISI value")
    else:

        print(f"\n\nfig_1b")
        # fig 1b, ISI on x-axis, different line for each sep
        if n_trimmed is not None:
            fig1b_title = f'{ave_over} probe luminance at each ISI value per separation\n' \
                          f'(n={ave_over_n}, trim={n_trimmed}, err={error_type}).'
            fig1b_savename = f'ave_TM{n_trimmed}_thr_all_runs_transpose{plt_suffix}.png'
        else:
            fig1b_title = f'{ave_over} probe luminance at each ISI value per separation\n' \
                          f'(n={ave_over_n}, err={error_type})'
            fig1b_savename = f'ave_thr_all_runs_transpose{plt_suffix}.png'

        plot_w_errors_no_1probe(wide_df=all_df, x_var='ISI', y_var=thr_col,
                                lines_var='separation', long_df_idx_col=idx_col,
                                legend_names=sep_name_list,
                                x_tick_labels=isi_name_list,
                                alt_colours=True, fixed_y_range=False, jitter=True,
                                error_caps=True,
                                ra_cd_v_line=ra_cd_size_dict['cd_isi_idx'],
                                fig1b_title=fig1b_title,
                                fig1b_savename=fig1b_savename, save_path=save_path,
                                verbose=True)
        if show_plots:
            plt.show()
        plt.close()

        if verbose:
            print('finished fig1b')

    print(f"\nfig_1c\n")
    all_df.columns = all_df_headers

    # fig 1c, eight plots - seven showing a particular separation, eighth showing all separations.
    if n_trimmed is not None:
        f'{ave_over} average thresholds per separation'
        fig1c_title = f'{ave_over} average thresholds per separation.  (n={ave_over_n}, trim={n_trimmed}, err={error_type}).'
        fig1c_savename = f'ave_TM{n_trimmed}_thr_per_sep{plt_suffix}.png'
    else:
        fig1c_title = f'{ave_over} average thresholds per separation.  (n={ave_over_n}, err={error_type})'
        fig1c_savename = f'ave_thr_per_sep{plt_suffix}.png'

    plot_n_sep_thr_w_scatter(all_thr_df=all_df, exp_ave=exp_ave,
                             ra_cd_v_line=ra_cd_size_dict['cd_isi_idx'],
                             fig_title=fig1c_title,
                             save_name=fig1c_savename, save_path=save_path, verbose=True)
    if show_plots:
        plt.show()
    plt.close()

    if verbose:
        print('finished fig_1c')


    # if split_1probe:
    #     print(f"\nfig_2a\n")
    #     # # Fig 2  - divide all 2probe conditions (pos_sep) by one_probe for each data_set
    #     # use ave_df with all (or trimmed) data.
    #     # first split each data_set into 1Probe and pos_sep (2probe), divide and make back into long df
    #     dset_list = list(all_df[idx_col].unique())
    #     print(f'dset_list: {dset_list}')
    #
    #     divided_list = []
    #     # loop through data_sets
    #     for data_set in dset_list:
    #         # for each data_set_df, split into pos_sep_df and one_probe_df
    #         data_set_df = all_df[all_df[idx_col] == data_set]
    #         data_set_df = data_set_df.drop(idx_col, axis=1)
    #         pos_sep_df, one_probe_df = split_df_into_pos_sep_df_and_1probe_df(data_set_df)
    #
    #         # divide pos_sep by one_probe and append to list
    #         pos_sep_arr = pos_sep_df.to_numpy()
    #         one_probe_arr = one_probe_df[thr_col].to_numpy()
    #         div_by_1probe_arr = (pos_sep_arr.T / one_probe_arr[:, None]).T
    #         div_by_1probe_df = pd.DataFrame(div_by_1probe_arr, columns=isi_name_list)
    #         div_by_1probe_df.insert(0, idx_col, [data_set] * len(div_by_1probe_df))
    #         div_by_1probe_df.insert(1, 'separation', pos_sep_list[:-1])
    #         divided_list.append(div_by_1probe_df)
    #
    #     # put back into long form df with data_set Sep cols
    #     divided_df = pd.concat(divided_list)
    #     print(f'divided_df:\n{divided_df}')
    #
    #     # # get means and errors
    #     div_groupby_sep_df = divided_df.drop(idx_col, axis=1)
    #     div_ave_psignifit_thr_df = div_groupby_sep_df.groupby('separation', sort=True).mean()
    #     if verbose:
    #         print(f'\ndiv_ave_psignifit_thr_df:\n{div_ave_psignifit_thr_df}')
    #
    #     if error_type in [False, None]:
    #         div_error_bars_df = None
    #     elif error_type.lower() in ['se', 'error', 'std-error', 'standard error', 'standard_error']:
    #         div_error_bars_df = div_groupby_sep_df.groupby('separation', sort=True).sem()
    #     elif error_type.lower() in ['sd', 'stdev', 'std_dev', 'std.dev', 'deviation', 'standard_deviation']:
    #         div_error_bars_df = div_groupby_sep_df.groupby('separation', sort=True).std()
    #     print(f'\ndiv_error_bars_df: ({error_type})\n{div_error_bars_df}')
    #
    #     if n_trimmed is not None:
    #         fig2a_save_name = f'ave_TM{n_trimmed}_thr_div_1probe.png'
    #         fig2a_title = f'{ave_over} average thresholds divided by single probe\n' \
    #                       f'(n={ave_over_n}, trim={n_trimmed}, err={error_type}).'
    #     else:
    #         fig2a_save_name = 'ave_thr_div_1probe.png'
    #         fig2a_title = f'{ave_over} average threshold divided by single probe\n(n={ave_over_n}, err={error_type})'
    #
    #     plot_1probe_w_errors(fig_df=div_ave_psignifit_thr_df, error_df=div_error_bars_df,
    #                          split_1probe=False, jitter=True,
    #                          error_caps=True, alt_colours=False,
    #                          legend_names=['Conc', 'ISI 0', 'ISI 2', 'ISI 4',
    #                                        'ISI 6', 'ISI 9', 'ISI 12', 'ISI 24'],
    #                          x_tick_vals=[0, 1, 2, 3, 6, 18],
    #                          x_tick_labels=[0, 1, 2, 3, 6, 18],
    #                          fixed_y_range=False,
    #                          fig_title=fig2a_title, save_name=fig2a_save_name,
    #                          save_path=save_path, verbose=True)
    #     if show_plots:
    #         plt.show()
    #     plt.close()
    #
    #     if verbose:
    #         print('finished fig2a')




    print(f"\nHeatmap 1\n")
    if 'separation' in list(ave_df.columns):
        ave_df.set_index('separation', drop=True, inplace=True)

    if n_trimmed is not None:
        heatmap_title = f'{ave_over} mean Threshold\nfor each ISI and separation (n={ave_over_n}, trim={n_trimmed}).'
        heatmap_savename = f'mean_TM{n_trimmed}_thr_heatmap{plt_suffix}'
    else:
        heatmap_title = f'{ave_over} mean Threshold\nfor each ISI and separation (n={ave_over_n})'
        heatmap_savename = f'mean_thr_heatmap{plt_suffix}'

    if ra_cd_size_dict['cd_isi_idx']:
        adjusted_v_line = ra_cd_size_dict['cd_isi_idx'] + 1
    else:
        adjusted_v_line = None

    if ra_cd_size_dict['ra_sep_idx']:
        adjusted_h_line = ra_cd_size_dict['ra_sep_idx'] + 1
    else:
        adjusted_h_line = None

    # regular (not transpose)
    plot_thr_heatmap(heatmap_df=ave_df.copy(), x_tick_labels=isi_name_list,
                     y_tick_labels=sep_name_list,
                     ra_cd_v_line=adjusted_v_line,
                     ra_cd_h_line=adjusted_h_line,
                     fig_title=heatmap_title,
                     save_name=heatmap_savename, save_path=save_path,
                     annot_fmt=heatmap_annot_fmt,
                     verbose=True)

    if show_plots:
        plt.show()
    plt.close()

    print(f"\nHeatmap 1b - errors\n")
    if 'separation' in list(error_bars_df.columns):
        error_bars_df.set_index('separation', drop=True, inplace=True)

    if n_trimmed is not None:
        heatmap_title = f'{ave_over} mean Error ({error_type})\nfor each ISI and separation (n={ave_over_n}, trim={n_trimmed}).'
        heatmap_savename = f'mean_TM{n_trimmed}_error_heatmap{plt_suffix}'
    else:
        heatmap_title = f'{ave_over} mean Error ({error_type})\nfor each ISI and separation (n={ave_over_n})'
        heatmap_savename = f'mean_error_heatmap{plt_suffix}'

    if ra_cd_size_dict['cd_isi_idx']:
        adjusted_v_line = ra_cd_size_dict['cd_isi_idx'] + 1
    else:
        adjusted_v_line = None

    if ra_cd_size_dict['ra_sep_idx']:
        adjusted_h_line = ra_cd_size_dict['ra_sep_idx'] + 1
    else:
        adjusted_h_line = None

    # regular (not transpose)
    plot_thr_heatmap(heatmap_df=error_bars_df.copy(), x_tick_labels=isi_name_list,
                     y_tick_labels=sep_name_list,
                     ra_cd_v_line=adjusted_v_line,
                     ra_cd_h_line=adjusted_h_line,
                     fig_title=heatmap_title,
                     save_name=heatmap_savename, save_path=save_path,
                     annot_fmt=heatmap_annot_fmt,
                     verbose=True)

    if show_plots:
        plt.show()
    plt.close()

    # if split_1probe:
    #     print(f"\nHeatmap 2. div 1Probe\n")
    #     print(f'div_ave_psignifit_thr_df:\n{div_ave_psignifit_thr_df}')
    #     if 'separation' in list(div_ave_psignifit_thr_df.columns):
    #         div_ave_psignifit_thr_df.set_index('separation', drop=True, inplace=True)
    #
    #     # get mean of each col, then mean of that
    #
    #     if n_trimmed is not None:
    #         heatmap_title = f'{ave_over} mean Threshold/1Probe for each ISI and separation\n' \
    #                         f'(n={ave_over_n}, trim={n_trimmed}).'
    #         heatmap_savename = f'mean_TM{n_trimmed}_thr_div_1probe_heatmap'
    #     else:
    #         heatmap_title = f'{ave_over} mean Threshold/1Probe for each ISI and separation\n' \
    #                         f'(n={ave_over_n})'
    #         heatmap_savename = 'mean_thr_div_1probe_heatmap'
    #
    #     div_1pr_sep_names_list = sep_name_list
    #     if '1Probe' in div_1pr_sep_names_list:
    #         div_1pr_sep_names_list = div_1pr_sep_names_list.remove("1Probe")
    #
    #     plot_thr_heatmap(heatmap_df=div_ave_psignifit_thr_df.T,
    #                      x_tick_labels=div_1pr_sep_names_list, y_tick_labels=isi_name_list,
    #                      annot_fmt=heatmap_annot_fmt,
    #                      fig_title=heatmap_title, save_name=heatmap_savename,
    #                      save_path=save_path, verbose=True)
    #     if show_plots:
    #         plt.show()
    #     plt.close()
    #
    # if show_plots:
    #     plt.show()
    # plt.close()


    # only do per-row and per-col heatmaps if there is 2d data
    if len(isi_name_list) > 1 and len(sep_name_list) > 1:
        print('making heatmaps per-row and per-column')

        print(f"\nHeatmap per row\n")
        if 'separation' in list(ave_df.columns):
            ave_df.set_index('separation', drop=True, inplace=True)

        # get mean of each col, then mean of that
        if n_trimmed is not None:
            heatmap_pr_title = f'{ave_over} Heatmap per row (n={ave_over_n}, trim={n_trimmed}).'
            heatmap_pr_savename = f'mean_TM{n_trimmed}_heatmap_per_row{plt_suffix}'
        else:
            heatmap_pr_title = f'{ave_over} Heatmap per row (n={ave_over_n})'
            heatmap_pr_savename = f'mean_heatmap_per_row{plt_suffix}'

        if ra_cd_size_dict['cd_isi_idx']:
            adjusted_v_line = ra_cd_size_dict['cd_isi_idx'] + 1
        else:
            adjusted_v_line = None

        plt_heatmap_row_col(heatmap_df=ave_df,
                            colour_by='row',
                            x_axis_label='ISI',
                            y_axis_label='Separation',
                            ra_cd_v_line=adjusted_v_line,
                            ra_cd_h_line=ra_cd_size_dict['ra_sep_idx'],
                            fig_title=heatmap_pr_title,
                            annot_fmt=heatmap_annot_fmt,
                            save_name=heatmap_pr_savename,
                            save_path=save_path,
                            verbose=True)
        if show_plots:
            plt.show()
        plt.close()

        print(f"\nHeatmap per col\n")
        if 'separation' in list(ave_df.columns):
            ave_df.set_index('separation', drop=True, inplace=True)

        # get mean of each col, then mean of that
        if n_trimmed is not None:
            heatmap_pc_title = f'{ave_over} Heatmap per col (n={ave_over_n}, trim={n_trimmed}).'
            heatmap_pc_savename = f'mean_TM{n_trimmed}_heatmap_per_col{plt_suffix}'
        else:
            heatmap_pc_title = f'{ave_over} Heatmap per col (n={ave_over_n})'
            heatmap_pc_savename = f'mean_heatmap_per_col{plt_suffix}'

        if ra_cd_size_dict['ra_sep_idx']:
            adjusted_h_line = ra_cd_size_dict['ra_sep_idx'] + 1
        else:
            adjusted_h_line = None

        plt_heatmap_row_col(heatmap_df=ave_df,
                            colour_by='col',
                            x_axis_label='ISI',
                            y_axis_label='Separation',
                            ra_cd_v_line=ra_cd_size_dict['cd_isi_idx'],
                            ra_cd_h_line=adjusted_h_line,
                            annot_fmt=heatmap_annot_fmt,
                            fig_title=heatmap_pc_title,
                            save_name=heatmap_pc_savename,
                            save_path=save_path,
                            verbose=True)
        if show_plots:
            plt.show()
        plt.close()

    print(f"\nfig_3a - difference from concurrent\n")
    run_this_plot = False
    # check if this dataset contains the 'concurrent' (ISI -1) condition
    if any(item in ['ISI_-1', 'ISI -1', 'conc', 'Conc', 'Concurrent', 'concurrent']
           for item in list(ave_df.columns)):
        run_this_plot = True

        ave_DfC_name = 'MASTER_ave_DfC.csv'
        error_DfC_name = f'MASTER_ave_DfC_error_{error_type}.csv'
        # if n_trimmed > 0:
        if n_trimmed is not None:
            ave_DfC_name = f'MASTER_ave_TM{n_trimmed}_DfC.csv'
            error_DfC_name = f'MASTER_ave_TM{n_trimmed}_DfC_error_{error_type}.csv'

        # check if the difference from concurrent files have arelady been made.
        if os.path.isfile(os.path.join(save_path, ave_DfC_name)):
            print("found DcF files")
            ave_DfC_df = pd.read_csv(os.path.join(save_path, ave_DfC_name))
            error_DfC_df = pd.read_csv(os.path.join(save_path, error_DfC_name))
        else:
            print("making DfC files")
            ave_DfC_df, error_DfC_df = make_diff_from_conc_df(all_df_path, save_path,
                                                              n_trimmed=n_trimmed, exp_ave=exp_ave)

        # error_DfC_df.rename(columns={'Concurrent': 'Conc', 'ISI_-1': 'Conc'},
        #                              inplace=True)
        print(f"ave_DfC_df:\n{ave_DfC_df}")
        print(f"error_DfC_df:\n{error_DfC_df}")

    if run_this_plot:

        if n_trimmed is not None:
            fig3a_save_name = f'diff_from_conc_TM{n_trimmed}.png'
            fig3a_title = f'{ave_over} ISI difference in threshold from concurrent\n(n={ave_over_n}, trim={n_trimmed}).'
        else:
            fig3a_save_name = 'diff_from_conc.png'
            fig3a_title = f'{ave_over} ISI difference in threshold from concurrent\n(n={ave_over_n})'

        plot_diff_from_conc_lineplot(ave_DfC_df, err_DfC_df=error_DfC_df,
                                     ra_cd_v_line=ra_cd_size_dict['cd_isi_idx'],
                                     fig_title=fig3a_title,
                                     save_name=fig3a_save_name, save_path=save_path)
        if show_plots:
            plt.show()
        plt.close()

        if verbose:
            print('finished fig3a')

        print(f"\nplot_diff_from_conc_heatmap\n")
        if n_trimmed is not None:
            heatmap_dfc_title = f'{ave_over} plot_diff_from_conc_heatmap (n={ave_over_n}, trim={n_trimmed}).'
            heatmap_dfc_savename = f'mean_TM{n_trimmed}_plot_diff_from_conc_heatmap{plt_suffix}'
        else:
            heatmap_dfc_title = f'{ave_over} plot_diff_from_conc_heatmap (n={ave_over_n})'
            heatmap_dfc_savename = f'mean_plot_diff_from_conc_heatmap{plt_suffix}'

        if ra_cd_size_dict['cd_isi_idx']:
            adjusted_v_line = ra_cd_size_dict['cd_isi_idx'] + 1
        else:
            adjusted_v_line = None

        if ra_cd_size_dict['ra_sep_idx']:
            adjusted_h_line = ra_cd_size_dict['ra_sep_idx'] + 1
        else:
            adjusted_h_line = None

        plot_thr_heatmap(heatmap_df=ave_DfC_df.copy(),
                         midpoint=0,
                         ra_cd_v_line=adjusted_v_line,
                         ra_cd_h_line=adjusted_h_line,
                         annot_fmt=heatmap_annot_fmt,
                         fig_title=heatmap_dfc_title, save_name=heatmap_dfc_savename,
                         save_path=save_path, verbose=True)

        if show_plots:
            plt.show()
        plt.close()


        print(f"\nplot diff_from_conc/SE heatmap\n")
        if n_trimmed is not None:
            heatmap_dfc_title = f'{ave_over} diff_from_conc/{error_type} heatmap (n={ave_over_n}, trim={n_trimmed}).'
            heatmap_dfc_savename = f'mean_TM{n_trimmed}_diff_from_conc_div_{error_type}_heatmap{plt_suffix}'
        else:
            heatmap_dfc_title = f'{ave_over} diff_from_conc/{error_type}  (n={ave_over_n})'
            heatmap_dfc_savename = f'mean_diff_from_conc_div_{error_type}_heatmap{plt_suffix}'

        print(f"ave_DfC_df:\n{ave_DfC_df}")
        print(f"error_DfC_df:\n{error_DfC_df}")

        # if 'ISI_' in error_DfC_df.columns, change to be ints:
        error_DfC_col_names = error_DfC_df.columns
        error_DfC_col_names = ['Conc' if 'ISI_-1' in col_name else col_name for col_name in error_DfC_col_names]
        error_DfC_col_names = [int(col_name.replace('ISI_', '')) if 'ISI_' in col_name else col_name for col_name in error_DfC_col_names]
        error_DfC_df.columns = error_DfC_col_names

        # if '1Probe' in error_DfC_df index, change to '1pr'
        if '1Probe' in error_DfC_df.index:
            error_DfC_df.index = [idx.replace('1Probe', '1pr') for idx in error_DfC_df.index]
            print(f"error_DfC_df:\n{error_DfC_df}")

        # copy ave_DfC_df and edit column names
        my_ave_DfC_df = ave_DfC_df.copy()
        my_ave_DfC_col_names = my_ave_DfC_df.columns
        my_ave_DfC_col_names = ['Conc' if 'ISI_-1' in col_name else col_name for col_name in my_ave_DfC_col_names]
        my_ave_DfC_col_names = [int(col_name.replace('ISI_', '')) if 'ISI_' in col_name else col_name for col_name in my_ave_DfC_col_names]
        my_ave_DfC_df.columns = my_ave_DfC_col_names

        # if '1Probe' in my_ave_DfC_df index, change to '1pr'
        if '1Probe' in my_ave_DfC_df.index:
            my_ave_DfC_df.index = [idx.replace('1Probe', '1pr') for idx in my_ave_DfC_df.index]
        print(f"my_ave_DfC_df:\n{my_ave_DfC_df}")


        dfc_div_error_df = my_ave_DfC_df.div(error_DfC_df).fillna(0)
        print(f"dfc_div_error_df:\n{dfc_div_error_df}")

        if ra_cd_size_dict['cd_isi_idx']:
            adjusted_v_line = ra_cd_size_dict['cd_isi_idx'] + 1
        else:
            adjusted_v_line = None

        if ra_cd_size_dict['ra_sep_idx']:
            adjusted_h_line = ra_cd_size_dict['ra_sep_idx'] + 1
        else:
            adjusted_h_line = None

        plot_thr_heatmap(heatmap_df=dfc_div_error_df.copy(),
                         midpoint=0,
                         ra_cd_v_line=adjusted_v_line,
                         ra_cd_h_line=adjusted_h_line,
                         annot_fmt=heatmap_annot_fmt,
                         fig_title=heatmap_dfc_title, save_name=heatmap_dfc_savename,
                         save_path=save_path, verbose=True)



        print(f"\nt-scores plot\n")
        if n_trimmed is not None:
            fig1_title = f'{ave_over} average DfC / {error_type} across all runs\n(n={ave_over_n}, trim={n_trimmed}).'
            fig1_savename = f'ave_TM{n_trimmed}_DfC_div_{error_type}{plt_suffix}.png'
        else:
            fig1_title = f'{ave_over} average average DfC / {error_type}  across all runs\n(n={ave_over_n})'
            fig1_savename = f'ave_DfC_div_{error_type}{plt_suffix}.png'

        # reset the index and rename the current index column 'separation'
        dfc_div_error_df = dfc_div_error_df.reset_index().rename(columns={'index': 'separation'})
        print(f"dfc_div_error_df:\n{dfc_div_error_df}")
        if "Bloch's CD" in sep_name_list:
            # remove "Bloch's CD" from sep_name_list
            print(f"sep_name_list: {sep_name_list}")
            sep_name_list.remove("Bloch's CD")
            print(f"sep_name_list: {sep_name_list}")

        plot_pos_sep_and_1probe(dfc_div_error_df,
                                thr_col='newLum',
                                fig_title=fig1_title,
                                one_probe=True,
                                save_path=save_path,
                                save_name=fig1_savename,
                                isi_name_list=isi_name_list,
                                pos_set_ticks=sep_vals_list,
                                pos_tick_labels=sep_name_list,
                                ra_cd_v_line=ra_cd_size_dict['ra_size_sep'],
                                y_axis_label=f't-value (diff from conc / {error_type})',
                                error_bars_df=None,
                                verbose=True)
        if show_plots:
            plt.show()
        plt.close()

        if verbose:
            print('finished t-scores plot')

        if len(isi_name_list) > 1 and len(sep_name_list) > 1:
            print('making dfc heatmaps per-row and per-column')

            print(f"\nplot_diff_from_conc_per_row\n")
            if n_trimmed is not None:
                heatmap_dfc_pr_title = f'{ave_over} plot_diff_from_conc_per_row (n={ave_over_n}, trim={n_trimmed}).'
                heatmap_dfc_pr_savename = f'mean_TM{n_trimmed}_plot_diff_from_conc_per_row{plt_suffix}'
            else:
                heatmap_dfc_pr_title = f'{ave_over} plot_diff_from_conc_per_row (n={ave_over_n})'
                heatmap_dfc_pr_savename = f'mean_plot_diff_from_conc_per_row{plt_suffix}'

            if 'ISI_-1' in list(ave_DfC_df.columns):
                DfC_no_conc_df = ave_DfC_df.drop(['ISI_-1'], axis=1)
            elif 'Conc' in list(ave_DfC_df.columns):
                DfC_no_conc_df = ave_DfC_df.drop(['Conc'], axis=1)
            elif 'Concurrent' in list(ave_DfC_df.columns):
                DfC_no_conc_df = ave_DfC_df.drop(['Concurrent'], axis=1)
            print(f"ave_DfC_df:\n{ave_DfC_df}")
            print(f"DfC_no_conc_df:\n{DfC_no_conc_df}")

            plt_heatmap_row_col(heatmap_df=DfC_no_conc_df,
                                colour_by='row',
                                midpoint=0,
                                x_axis_label='ISI',
                                y_axis_label='Separation',
                                ra_cd_v_line=ra_cd_size_dict['cd_isi_idx'],
                                ra_cd_h_line=ra_cd_size_dict['ra_sep_idx'],
                                fig_title=heatmap_dfc_pr_title,
                                fontsize=10,
                                annot_fmt=heatmap_annot_fmt,
                                save_name=heatmap_dfc_pr_savename,
                                save_path=save_path,
                                verbose=True)
            if show_plots:
                plt.show()
            plt.close()


            '''Make diff from conc surface plot with RA/CD lines'''

            for k, v in ra_cd_size_dict.items():
                print(f"{k}: {v}")

            print(f"\nplot_diff_from_conc_surface\n")

            if n_trimmed is not None:
                dfc_surface_title = f'{ave_over} difference from concurrent surface (n={ave_over_n}, trim={n_trimmed}).'
                dfc_surface_savename = f'mean_TM{n_trimmed}_dfc_surface_{plt_suffix}'
            else:
                dfc_surface_title = f'{ave_over} difference from concurrent surface (n={ave_over_n})'
                dfc_surface_savename = f'mean__dfc_surface_{plt_suffix}'

            plot_thr_3dsurface(plot_df=ave_DfC_df,
                               drop_1probe=True,
                               drop_conc=False,
                               ra_cd_size_dict=ra_cd_size_dict,
                               even_spaced=False,
                               transpose_df=True,
                               show_min_per_sep=True,
                               my_rotation=330,
                               fig_title=dfc_surface_title,
                               save_path=save_path, save_name=dfc_surface_savename,
                               verbose=True)
            if show_plots:
                plt.show()
            plt.close()


            # fig 1c, eight diff from conc plots - seven showing a particular separation, eighth showing all separations.
            if n_trimmed is not None:
                dfc_per_sep_title = f'{ave_over} average diff from conc per separation.  (n={ave_over_n}, trim={n_trimmed}, err={error_type}).'
                dfc_per_sep_savename = f'ave_TM{n_trimmed}_dfc_per_sep{plt_suffix}.png'
            else:
                dfc_per_sep_title = f'{ave_over} average diff from conc per separation.  (n={ave_over_n}, err={error_type})'
                dfc_per_sep_savename = f'ave_dfc_per_sep{plt_suffix}.png'

            print("\n\nidiot check")
            print(f"all_df:\n{all_df}")

            # make all_dfc_df (not ave)
            all_dfc_df = make_diff_from_conc_df(all_df, root_path=save_path, mean_and_err_df=False, exp_ave=exp_ave)

            print("\n\nidiot check")
            print(f"all_dfc_df:\n{all_dfc_df}")

            plot_n_sep_thr_w_scatter(all_thr_df=all_dfc_df, exp_ave=exp_ave,
                                     ra_cd_v_line=ra_cd_size_dict['cd_isi_idx'],
                                     fig_title=dfc_per_sep_title,
                                     save_name=dfc_per_sep_savename, save_path=save_path, verbose=True)
            if show_plots:
                plt.show()
            plt.close()

            if verbose:
                print('finished diff from conc per sep')



    print("\n*** finished make_average_plots()***\n")
