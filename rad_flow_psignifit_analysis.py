import os

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

"""
This page contains python functions to analyse the radial_flow.py experiment.
The workflow is based on Martin's MATLAB scripts used in OLD_MATLAB_analysis.py.

1. a_data_extraction: put data from one run, multiple ISIs into one array. 
2. b1_extract_last_values: get the threshold used on the last values in each 
    staircase for each isi.  this is used for b3_plot_staircases.
# not used. 3. b2_last_reversal/m: Computes the threshold for each staircase as an average of 
#     the last N reversals (incorrect responses).  Plot these as mean per sep, 
#     and also multi-plot for each isi with different values for -18&+18 etc
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


def split_df_alternate_rows(df):
    """
    Split a dataframe into alternate rows.  Dataframes are organized by
    stair conditions relating to separations
    (e.g., in order [18, -18, 6, -6, 3, -3, 2, -2, 1, -1, 0, 0]).
    For some plots this needs to be split into two dfs, e.g., :
    pos_sep_df: [18, 6, 3, 2, 1, 0]
    neg_sep_df: [-18, -6, -3, -2, -1, 0]

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
    row and column positions (e.g., separation and ISI).
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


def make_long_df(wide_df,
                 cols_to_keep=['congruent', 'separation'],
                 cols_to_change=['isi1', 'isi4', 'isi6'],
                 cols_to_change_show='probeLum',
                 new_col_name='isi', strip_from_cols='isi', verbose=True):
    """
    Function to convert wide-form_df to long-form_df.  e.g., if there are several
    columns showing ISIs (cols_to_change), this puts them all into one column (new_col_name).

    :param wide_df: dataframe to be changed
    :param cols_to_keep: Columns to use for indexing (e.g., ['congruent', 'separation'...etc]
    :param cols_to_change: List of columns showing data at different levels e.g., [isi1, isi4, isi6...etc].
    :param cols_to_change_show: What is being measured in repeated cols, e.g., probeLum.
    :param new_col_name: name for new col describing levels e.g. isi
    :param strip_from_cols: string to strip from col names when for new cols.
        e.g., if strip_from_cols='isi_', then [isi_1, isi_4] becomes [1, 4].
    :param verbose: if true, prints progress to screen.

    :return: long_df
    """
    print("\n*** running make_long_df() ***\n")

    new_col_names = cols_to_keep + [new_col_name] + [cols_to_change_show]

    # make longform data
    if verbose:
        print(f"\n preparing to loop through: {cols_to_change}")
    long_list = []
    for this_col in cols_to_change:
        this_df_cols = cols_to_keep + [this_col]
        this_df = wide_df[this_df_cols]

        # strip text from col names, try lower/upper case if strip_from_cols not found.
        if strip_from_cols not in [False, None]:
            if strip_from_cols in this_col:
                this_col = this_col.strip(strip_from_cols)
            elif strip_from_cols.lower() in this_col:
                this_col = this_col.strip(strip_from_cols.lower())
            elif strip_from_cols.upper() in this_col:
                this_col = this_col.strip(strip_from_cols.upper())
            else:
                raise ValueError(f"can't strip {strip_from_cols} from {this_col}")

        this_df.insert(len(cols_to_keep), new_col_name, [this_col] * len(this_df.index))
        this_df.columns = new_col_names
        long_list.append(this_df)

    long_df = pd.concat(long_list)
    long_df.reset_index(drop=True, inplace=True)
    if verbose:
        print(f'long_df:\n{long_df}')

    print("\n*** finished make_long_df() ***\n")

    return long_df


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


def multi_plot_shape(n_figs, min_rows=1):
    """
    Function to make multi-plot figure with right number of rows and cols, 
    to fit n_figs, but with smallest shape for landscape.
    :param n_figs: Number of plots I need to make.
    :param min_rows: Minimum number of rows (sometimes won't work with just 1)

    :return: n_rows, n_cols
    """
    n_rows = 1
    if n_figs > 3:
        n_rows = 2
    if n_figs > 8:
        n_rows = 3
    if n_figs > 12:
        n_rows = 4

    if n_rows < min_rows:
        n_rows = min_rows

    td = n_figs // n_rows
    mod = n_figs % n_rows
    n_cols = td + mod

    # there are some weird results, this helps catch 11, 14, 15 etc from going mad.
    if n_rows * (n_cols - 1) > n_figs:
        n_cols = n_cols - 1
        if n_rows * (n_cols - 1) > n_figs:
            n_cols = n_cols - 1
    # plots = n_rows * n_cols
    # print(f"{n_figs}: n_rows: {n_rows}, td: {td}, mod: {mod}, n_cols: {n_cols}, plots: {plots}, ")

    if n_figs > 20:
        raise ValueError('too many plots for one page!')

    return n_rows, n_cols


# # # all ISIs on one axis -
# FIGURE 1 - shows one axis (x=separation (-18:18), y=probeLum) with multiple ISI lines.
# dotted line at zero to make batman more apparent.
def plot_data_unsym_batman(pos_and_neg_sep_df,
                           fig_title=None,
                           save_path=None,
                           save_name=None,
                           isi_name_list=None,
                           x_tick_values=None,
                           x_tick_labels=None,
                           verbose=True):
    """
    This plots a figure with one axis, x has separation values [-18, -6, -3, -2, -1, 0, 1, 2, 3, 6, 18],
    Will plot all ISIs on the same axis as lineplots.

    :param pos_and_neg_sep_df: Full dataframe to use for values
    :param fig_title: default=None.  Pass a string to add as a title.
    :param save_path: default=None.  Path to dir to save fig
    :param save_name: default=None.  name to save fig
    :param isi_name_list: default=NONE: will use defaults, or pass list of names for legend.
    :param x_tick_values: default=NONE: will use defaults, or pass list of names for x-axis positions.
    :param x_tick_labels: default=NONE: will use defaults, or pass list of names for x_axis labels.
    :param verbose: default: True. Won't print anything to screen if set to false.

    :return: plot
    """
    if verbose:
        print("\n*** running plot_data_unsym_batman() ***")

    # get plot details
    if isi_name_list is None:
        raise ValueError('please pass an isi_name_list')
    if verbose:
        print(f'isi_name_list: {isi_name_list}')

    if x_tick_values is None:
        x_tick_values = [-18, -6, -3, -2, -1, -.1, 0, 1, 2, 3, 6, 18]
    if x_tick_labels is None:
        x_tick_labels = [-18, -6, -3, -2, -1, '-0', 0, 1, 2, 3, 6, 18]

    # make fig1
    fig, ax = plt.subplots(figsize=(10, 6))

    # line plot for main ISIs
    sns.lineplot(data=pos_and_neg_sep_df, markers=True, dashes=False, ax=ax)
    ax.axvline(x=5.5, linestyle="-.", color='lightgrey')

    # decorate plot
    if x_tick_values is not None:
        ax.set_xticks(x_tick_values)
    if x_tick_labels is not None:
        ax.set_xticklabels(x_tick_labels)
    ax.set_xlabel('Probe separation in diagonal pixels')
    ax.set_ylabel('Probe Luminance')

    ax.legend(labels=isi_name_list, title='ISI',
              shadow=True,
              # place lower left corner of legend at specified location.
              loc='lower left', bbox_to_anchor=(0.96, 0.5))

    if fig_title is not None:
        plt.title(fig_title)

    # save plot
    if save_path is not None:
        if save_name is not None:
            plt.savefig(f'{save_path}{os.sep}{save_name}')

    return fig


def plot_runs_ave_w_errors(fig_df, error_df,
                           jitter=True, error_caps=False, alt_colours=False,
                           legend_names=None,
                           x_tick_vals=None,
                           x_tick_labels=None,
                           even_spaced_x=False,
                           fixed_y_range=False,
                           fig_title=None, save_name=None, save_path=None,
                           verbose=True):
    """
    Calculate and plot the mean and error estimates (y-axis) at each separation values (x-axis).
    Separate line for each ISI.  Error bar values taken from separate error_df.

    :param fig_df: dataframe to build plot from.  Expects fig_df in the form:
        separation as index, ISIs as columns.
    :param error_df: dataframe of same shape as fig_df, but contains error values
    :param jitter: Jitter x_axis values so points don't overlap.
    :param error_caps: caps on error bars for more easy reading
    :param alt_colours: Use different set of colours to normal (e.g., if ISI on
        x-axis and lines for each separation).
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
    print('\n*** running plot_runs_ave_w_errors() ***\n')

    if verbose:
        print(f'fig_df:\n{fig_df}')
        print(f'\nerror_df:\n{error_df}')

    # get names for legend (e.g., different lines)
    column_names = fig_df.columns.to_list()

    if legend_names is None:
        legend_names = column_names
    if verbose:
        print(f'\nColumn and Legend names:')
        for a, b in zip(column_names, legend_names):
            print(f"{a}\t=>\t{b}\tmatch: {bool(a==b)}")

    if x_tick_vals is None:
        x_tick_vals = fig_df.index

    # for evenly spaced items on x_axis
    if even_spaced_x:
        x_tick_vals = list(range(len(x_tick_vals)))

    # adding jitter works well if df.index are all int
    # need to set it up to use x_tick_vals if df.index is not all int or float
    check_idx_num = all(isinstance(x, (int, float)) for x in fig_df.index)
    print(f'check_idx_num: {check_idx_num}')

    check_x_val_num = all(isinstance(x, (int, float)) for x in x_tick_vals)
    print(f'check_x_val_num: {check_x_val_num}')

    if jitter:
        if not all(isinstance(x, (int, float)) for x in x_tick_vals):
            x_tick_vals = list(range(len(x_tick_vals)))

    # get number of locations for jitter list
    n_pos_sep = len(fig_df.index.to_list())

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
        x_values = x_tick_vals + jitter_list

        ax.errorbar(x=x_values, y=fig_df[name],
                    yerr=error_df[name],
                    marker='.', lw=2, elinewidth=.7,
                    capsize=cap_size, color=my_colours[idx])

        leg_handle = mlines.Line2D([], [], color=my_colours[idx], label=name,
                                   marker='.', linewidth=.5, markersize=4)
        legend_handles_list.append(leg_handle)

    # decorate plot
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


def plot_isi_x_axis_w_errors(wide_df, cols_to_keep=['congruent', 'separation'],
                             cols_to_change=['isi1', 'isi4', 'isi6'],
                             cols_to_change_show='probeLum', new_col_name='ISI',
                             strip_from_cols='isi',
                             x_axis='separation', y_axis='probeLum',
                             hue_var='ISI', style_var='congruent', style_order=[1, -1],
                             error_bars=False,
                             jitter=False,
                             even_spaced_x=False,
                             x_tick_vals=None,
                             fig_title=None,
                             fig_savename=None,
                             verbose=True):
    """
    Funtion to plot line_plot with multiple lines.  This function works with a single dataset,
    or with a df containing several datasets, in which case it plots the mean with error bars at .68 confidence interval.
    It will work with either separation on x-axis and different lines for each ISI, or vice versa.
    The first part of the function converts a wide dataframe to a long dataframe.

    :param wide_df: Data to be plotted
    :param cols_to_keep: Variables that will be included in long dataframe.
    :param cols_to_change: Columns containing different measurements of some
        variable (e.g., isi1, isi4, isi6...etc) that will be converted into
        longform (e.g., isi: [1, 4, 6]).
    :param cols_to_change_show: What is being measured in cols to change (e.g., probeLum; dependent variable)
    :param new_col_name: What the cols to change describe (e.g., isi; independent variable)
    :param strip_from_cols: string to remove if independent variables are to be
        turned into numberic values (e.g., for isi1, isi4 isi6, strip 'isi' to get 1, 4,6).
    :param x_axis: Variable to be shown along x-axis (e.g., separation or isi)
    :param y_axis: Variable to be shown along y-axis (e.g., probeLum)
    :param hue_var: Variable to be shown with different lines (e.g., isi or separation)
    :param style_var: Addition variable to show with solid or dashed lines (e.g., congruent or incongruent)
    :param style_order: Order of style var as displayed in df (e.g., [1, -1])
    :param error_bars: True or false, whether to display error bars
    :param jitter: Whether to jitter items on x-axis to make easier to read.
        Can be True, False or float for amount of jitter in relation to x-axis values.
    :param even_spaced_x: Whether to evenly spave ticks on x-axis.
        For example to make the left side of log-scale x-values easier to read.
    :param x_tick_vals: Values/labels for x-axis.  Can be string, int or float.
    :param fig_title: Title for figure
    :param fig_savename: Save path/name for figure
    :param verbose: Whether to print progress to screen.

    :return: figure
    """
    print('\n*** running plot_isi_x_axis_w_errors() ***\n')

    # convert wide df to long
    long_df = make_long_df(wide_df=wide_df, cols_to_keep=cols_to_keep,
                           cols_to_change=cols_to_change, cols_to_change_show=cols_to_change_show,
                           new_col_name=new_col_name, strip_from_cols=strip_from_cols, verbose=True)

    # data_for_x to use for x_axis data, whilst keeping original values list (x_axis)
    data_for_x = x_axis

    # for evenly spaced items on x_axis
    if even_spaced_x:
        orig_x_vals = sorted(x_tick_vals)
        new_x_vals = list(range(len(orig_x_vals)))

        # check if x_tick_vals refer to values in df, if not, use df values.
        if list(long_df[x_axis])[0] in orig_x_vals:
            x_space_dict = dict(zip(orig_x_vals, new_x_vals))
        else:
            x_space_dict = dict(zip(set(list(long_df[x_axis])), new_x_vals))

        # add column with new evenly spaced x-values, relating to original x_values
        spaced_x = [x_space_dict[i] for i in list(long_df[x_axis])]
        long_df.insert(0, 'spaced_x', spaced_x)
        data_for_x = 'spaced_x'

    # for jittering values on x-axis
    if jitter:
        jitter_keys = list(long_df[hue_var])
        n_jitter = len(jitter_keys)
        jit_max = .2
        if type(jitter) in [float, np.float]:
            jit_max = jitter

        # get rand float to add to x-axis values for jitter
        jitter_vals = np.random.uniform(size=n_jitter, low=-jit_max, high=jit_max)
        jitter_dict = dict(zip(jitter_keys, jitter_vals))
        jitter_x = [a+jitter_dict[b] for a, b in zip(list(long_df[data_for_x]), list(long_df[hue_var]))]
        long_df.insert(0, 'jitter_x', jitter_x)
        data_for_x = 'jitter_x'

    conf_interval = None
    if error_bars:
        conf_interval = 68

    if verbose:
        print(f'long_df:\n{long_df}')
        print(f'error_bars: {error_bars}')
        print(f'conf_interval: {conf_interval}')

    # initialize plot
    my_colours = fig_colours(n_conditions=len(set(list(long_df[hue_var]))))
    fig, ax = plt.subplots(figsize=(10, 6))

    # with error bards for d_averages example
    sns.lineplot(data=long_df, x=data_for_x, y=y_axis, hue=hue_var,
                 style=style_var, style_order=style_order,
                 ci=conf_interval, err_style='bars', err_kws={'elinewidth': .7, 'capsize': 5},
                 palette=my_colours, ax=ax)

    if even_spaced_x:
        ax.set_xticks(new_x_vals)
        ax.set_xticklabels(orig_x_vals)
    else:
        ax.set_xticks(x_tick_vals)

    plt.xlabel(x_axis)
    plt.title(fig_title)

    # Change legend labels for congruent and incongruent data
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels[:-2] + ['True', 'False'])

    plt.savefig(fig_savename)

    print('\n*** finished plot_isi_x_axis_w_errors() ***\n')

    return fig


# # # 8 batman plots
# this is a figure with one axis per isi, showing neg and pos sep (e.g., -18:18)
def multi_batman_plots(mean_df, thr1_df, thr2_df,
                       fig_title=None, isi_name_list=None,
                       x_tick_vals=None, x_tick_labels=None,
                       sym_sep_diff_list=None,
                       save_path=None, save_name=None,
                       verbose=True):
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
        print("\n*** running multi_batman_plots() ***")

    # get plot info
    if isi_name_list is None:
        isi_name_list = list(mean_df.columns[2:])
    if x_tick_vals is None:
        x_tick_vals = list(mean_df['separation'])
    if x_tick_labels is None:
        x_tick_labels = list(mean_df['separation'])
    if verbose:
        print(f'isi_name_list: {isi_name_list}')
        print(f'x_tick_vals: {x_tick_vals}')
        print(f'x_tick_labels: {x_tick_labels}')

    # make plots
    my_colours = fig_colours(len(isi_name_list))
    n_rows, n_cols = multi_plot_shape(len(isi_name_list), min_rows=2)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12, 6))

    if fig_title is not None:
        fig.suptitle(fig_title)

    ax_counter = 0
    # loop through the eight axes
    for row_idx, row in enumerate(axes):
        for col_idx, ax in enumerate(row):
            if ax_counter < len(isi_name_list):

                # mean threshold from CW and CCW probe jump direction
                sns.lineplot(ax=axes[row_idx, col_idx], data=mean_df,
                             x='separation', y=isi_name_list[ax_counter],
                             color=my_colours[ax_counter],
                             linewidth=2, linestyle="dotted", markers=True)

                sns.lineplot(ax=axes[row_idx, col_idx], data=thr1_df,
                             x='separation', y=isi_name_list[ax_counter],
                             color=my_colours[ax_counter],
                             linewidth=.5, marker="v")

                sns.lineplot(ax=axes[row_idx, col_idx], data=thr2_df,
                             x='separation', y=isi_name_list[ax_counter],
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
                                    markersize=4, label='Congruent')
                st2 = mlines.Line2D([], [], color=my_colours[ax_counter],
                                    marker='o', linewidth=.5,
                                    markersize=4, label='Incongruent')
                mean_line = mlines.Line2D([], [], color=my_colours[ax_counter],
                                          marker=None, linewidth=2, linestyle="dotted",
                                          label='mean')
                ax.legend(handles=[st1, st2, mean_line], fontsize=6)

                ax_counter += 1
            else:
                fig.delaxes(ax=axes[row_idx, col_idx])

    plt.tight_layout()
    
    if save_path is not None:
        if save_name is not None:
            plt.savefig(f'{save_path}{os.sep}{save_name}')

    return fig



def plot_thr_heatmap(heatmap_df,
                     x_tick_labels=None,
                     y_tick_labels=None,
                     fig_title=None,
                     save_name=None,
                     save_path=None,
                     verbose=True):
    """
    Function for making a heatmap
    :param heatmap_df: Expects dataframe with separation as index and ISIs as columns.
    :param x_tick_labels: Labels for columns
    :param y_tick_labels: Labels for rows
    :param fig_title: Title for figue
    :param save_name: name to save fig
    :param save_path: location to save fig
    :param verbose: if True, will prng progress to screen,

    :return: Heatmap
    """
    print('\n*** running plot_thr_heatmap() ***\n')

    if verbose:
        print(f'heatmap_df:\n{heatmap_df}')

    if x_tick_labels is None:
        x_tick_labels = list(heatmap_df.columns)
    if y_tick_labels is None:
        y_tick_labels = list(heatmap_df.index)

    # get mean of each column, then mean of those
    mean_thr = float(np.mean(heatmap_df.mean()))
    if verbose:
        print(f'mean_val: {round(mean_thr, 2)}')

    heatmap = sns.heatmap(data=heatmap_df,
                          annot=True, center=mean_thr,
                          cmap=sns.color_palette("Spectral_r", as_cmap=True),
                          xticklabels=x_tick_labels, yticklabels=y_tick_labels)

    heatmap.set_xlabel('ISI')

    if fig_title is not None:
        plt.title(fig_title)

    if save_path is not None:
        if save_name is not None:
            plt.savefig(f'{save_path}{os.sep}{save_name}')

    return heatmap



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
        raise ValueError('Please pass a list of isi values to identify directories containing data.')

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
        save_name = 'ALL_ISIs_sorted.xlsx'

        save_excel_path = os.path.join(run_dir, save_name)
        if verbose:
            print(f"\nsaving all_data_df to save_excel_path:\n{save_excel_path}")
        all_data_df.to_excel(save_excel_path, index=False)

    print("\n***finished a_data_extraction()***\n")


    return all_data_df


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
                                usecols=["ISI", "stair", "step", "separation", 
                                         "flow_dir", "probe_jump", "corner",
                                         "probeLum", "trial_response"])

    # get list of isi and stair values to loop through
    stair_list = all_data_df['stair'].unique()
    isi_list = all_data_df['ISI'].unique()
    # get isi string for column names
    isi_name_list = [f'isi{i}' for i in isi_list]

    trials, columns = np.shape(all_data_df)
    trials_per_stair = int(trials/len(isi_list)/len(stair_list))

    if verbose:
        print(f"all_data_df:\n{all_data_df}")
        print(f"{len(isi_list)} isi values and {len(stair_list)} stair values")
        print(f"isi_list: {isi_list}")
        print(f"isi_name_list: {isi_name_list}")
        print(f"stair_list: {stair_list}")
        print(f"trials_per_stair: {trials_per_stair}")

    '''the eighth plot is the last thr for each sep (+sep, -sep and mean).  
    get data from psignifit_thresholds.csv and reshape here'''
    thr_csv_name = f'{save_path}{os.sep}psignifit_thresholds.csv'
    psignifit_thr_df = pd.read_csv(thr_csv_name)
    if verbose:
        print(f'\npsignifit_thr_df:\n{psignifit_thr_df}')

    # remove extra columns
    if 'stair' in list(psignifit_thr_df.columns):
        psignifit_thr_df = psignifit_thr_df.drop(['stair'], axis=1)

    if 'stair_names' in list(psignifit_thr_df.columns):
        psignifit_thr_df = psignifit_thr_df.drop(['stair_names'], axis=1)

    if 'congruent' in list(psignifit_thr_df.columns):
        psignifit_thr_df = psignifit_thr_df.drop(['congruent'], axis=1)

    if 'separation' in list(psignifit_thr_df.columns):
        sep_list = psignifit_thr_df.pop('separation').tolist()
    print(f'sep_list: {sep_list}')

    psignifit_thr_df.columns = isi_name_list

    # split into pos_sep, neg_sep and mean of pos and neg.
    psig_cong_sep_df, psig_incong_sep_df = split_df_alternate_rows(psignifit_thr_df)
    psig_thr_mean_df = pd.concat([psig_cong_sep_df, psig_incong_sep_df]).groupby(level=0).mean()

    # add sep column in
    rows, cols = psig_thr_mean_df.shape
    if len(sep_list) == rows*2:
        # takes every other item
        sep_list = sep_list[::2]
    else:
        raise ValueError(f"I dunno why the number of rows ({rows}) isn't double "
                         f"the sep_list ({sep_list})")
    separation_title = [f'sep{i}' for i in sep_list]
    if verbose:
        print(f'sep_list: {sep_list}')
        print(f"separation_title: {separation_title}")

    psig_thr_mean_df.insert(0, 'separation', sep_list)
    psig_cong_sep_df.insert(0, 'separation', sep_list)
    psig_incong_sep_df.insert(0, 'separation', sep_list)
    if verbose:
        print(f'\npsig_cong_sep_df:\n{psig_cong_sep_df}')
        print(f'\npsig_incong_sep_df:\n{psig_incong_sep_df}')
        print(f'\npsig_thr_mean_df:\n{psig_thr_mean_df}')

    # make empty arrays to save reversal n_reversals
    n_reversals_np = np.zeros(shape=[len(stair_list), len(isi_list)])

    # loop through isi values
    for isi_idx, isi in enumerate(isi_list):

        # get df for this isi only
        isi_df = all_data_df[all_data_df['ISI'] == isi]
        isi_name = isi_name_list[isi_idx]

        # initialise 8 plot figure - last plot will be blank
        # # this is a figure showing n_reversals per staircase condition.
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12, 6))
        ax_counter = 0

        for row_idx, row in enumerate(axes):
            for col_idx, ax in enumerate(row):
                print(f'\nrow: {row_idx}, col: {col_idx}: {ax}')

                # for the first six plots...
                if ax_counter < 6:

                    # # get pairs of stairs (e.g., [[18, -18], [6, -6], ...etc)
                    stair_even_cong = ax_counter*2  # 0, 2, 4, 6, 8, 10
                    stair_even_cong_df = isi_df[isi_df['stair'] == stair_even_cong]
                    final_lum_even_cong = \
                        stair_even_cong_df.loc[stair_even_cong_df['step'] == trials_per_stair-1, 'probeLum'].item()
                    n_reversals_even_cong = trials_per_stair - stair_even_cong_df[resp_col].sum()

                    stair_odd_incong = (ax_counter*2)+1  # 1, 3, 5, 7, 9, 11
                    stair_odd_incong_df = isi_df[isi_df['stair'] == stair_odd_incong]
                    final_lum_odd_incong = \
                        stair_odd_incong_df.loc[stair_odd_incong_df['step'] == trials_per_stair-1, 'probeLum'].item()
                    n_reversals_odd_incong = trials_per_stair - stair_odd_incong_df[resp_col].sum()

                    # append n_reversals to n_reversals_np to save later.
                    n_reversals_np[stair_even_cong-1, isi_idx] = n_reversals_even_cong
                    n_reversals_np[stair_odd_incong-1, isi_idx] = n_reversals_odd_incong

                    if verbose:
                        print(f'\nstair_even_cong_df (stair={stair_even_cong}, '
                              f'isi_name={isi_name}:\n{stair_even_cong_df}')
                        print(f"final_lum_even_cong: {final_lum_even_cong}")
                        print(f"n_reversals_even_cong: {n_reversals_even_cong}")
                        print(f'\nstair_odd_incong_df (stair={stair_odd_incong}, '
                              f'isi_name={isi_name}:\n{stair_odd_incong_df}')
                        print(f"final_lum_odd_incong: {final_lum_odd_incong}")
                        print(f"n_reversals_odd_incong: {n_reversals_odd_incong}")

                    fig.suptitle(f'Staircases and reversals for isi {isi_name}')

                    # plot thr per step for even_cong numbered stair
                    sns.lineplot(ax=axes[row_idx, col_idx], data=stair_even_cong_df,
                                 x='step', y=thr_col, color='tab:blue',
                                 marker="o", markersize=4)
                    # line for final probeLum
                    ax.axhline(y=final_lum_even_cong, linestyle="-.", color='tab:blue')
                    # text for n_reversals
                    ax.text(x=0.25, y=0.8, s=f'{n_reversals_even_cong} reversals',
                            color='tab:blue',
                            # needs transform to appear with rest of plot.
                            transform=ax.transAxes, fontsize=12)

                    # plot thr per step for odd_incong numbered stair
                    sns.lineplot(ax=axes[row_idx, col_idx], data=stair_odd_incong_df,
                                 x='step', y=thr_col, color='tab:red',
                                 marker="v", markersize=5)
                    ax.axhline(y=final_lum_odd_incong, linestyle="--", color='tab:red')
                    ax.text(x=0.25, y=0.9, s=f'{n_reversals_odd_incong} reversals',
                            color='tab:red',
                            # needs transform to appear with rest of plot.
                            transform=ax.transAxes, fontsize=12)

                    ax.set_title(f'{isi_name} {separation_title[ax_counter]}')
                    ax.set_xticks(np.arange(0, trials_per_stair, 5))
                    ax.set_ylim([0, 110])

                    # artist for legend
                    if ax_counter == 0:
                        st1 = mlines.Line2D([], [], color='tab:red',
                                            marker='v',
                                            markersize=5, label='Congruent')
                        st1_last_val = mlines.Line2D([], [], color='tab:red',
                                                     linestyle="--", marker=None,
                                                     label='Cong: last val')
                        st2 = mlines.Line2D([], [], color='tab:blue',
                                            marker='o',
                                            markersize=5, label='Incongruent')
                        st2_last_val = mlines.Line2D([], [], color='tab:blue',
                                                     linestyle="-.", marker=None,
                                                     label='Incong: last val')
                        ax.legend(handles=[st1, st1_last_val, st2, st2_last_val],
                                  fontsize=6, loc='lower right')

                elif ax_counter == 6:
                    """use the psignifit from each stair pair (e.g., 18, -18) to
                    get the mean threshold for each sep condition.
                    """
                    if verbose:
                        print("Seventh plot")
                        print(f'psig_thr_mean_df:\n{psig_thr_mean_df}')

                    isi_thr_mean_df = pd.concat([psig_thr_mean_df['separation'], psig_thr_mean_df[isi_name]],
                                                axis=1, keys=['separation', isi_name])
                    if verbose:
                        print(f'isi_thr_mean_df:\n{isi_thr_mean_df}')

                    # line plot for thr1, th2 and mean thr
                    sns.lineplot(ax=axes[row_idx, col_idx], data=isi_thr_mean_df,
                                 x='separation', y=isi_name, color='lightgreen',
                                 linewidth=3)
                    sns.lineplot(ax=axes[row_idx, col_idx], data=psig_cong_sep_df,
                                 x='separation', y=isi_name, color='red',
                                 linestyle="--")
                    sns.lineplot(ax=axes[row_idx, col_idx], data=psig_incong_sep_df,
                                 x='separation', y=isi_name, color='blue',
                                 linestyle="dotted")

                    # artist for legend
                    cong_thr = mlines.Line2D([], [], color='red', linestyle="--",
                                             marker=None, label='Congruent thr')

                    incong_thr = mlines.Line2D([], [], color='blue', linestyle="dotted",
                                               marker=None, label='Incongruent thr')
                    mean_thr = mlines.Line2D([], [], color='lightgreen', linestyle="solid",
                                             marker=None, label='mean thr')
                    ax.legend(handles=[cong_thr, incong_thr, mean_thr],
                              fontsize=6, loc='lower right')

                    # decorate plot
                    ax.set_title(f'{isi_name} psignifit thresholds')
                    ax.set_xticks([0, 1, 2, 3, 6, 18])
                    ax.set_xticklabels([0, 1, 2, 3, 6, 18])
                    ax.set_xlabel('Probe separation')
                    ax.set_ylim([0, 110])
                    ax.set_ylabel('Probe Luminance')

                else:
                    fig.delaxes(ax=axes[row_idx, col_idx])

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


def c_plots(save_path, isi_name_list=None, show_plots=True, verbose=True):
    """
    5. c_plots.m: uses psignifit_thresholds.csv and outputs plots.

    figures:
            MIRRORED_runs.png: threshold luminance as function of probe separation,
                  Positive and negative separation values (batman plots),
                  one panel for each isi condition.
                  use multi_batman_plots()

            data.png: threshold luminance as function of probe separation.
                Positive and negative separation values (batman plot),
                all ISIs on same axis.
                Use plot_data_unsym_batman()

            compare_data.png: threshold luminance as function of probe separation.
                Positive and negative separation values (batman plot),
                all ISIs on same axis.
                doesn't use a function, built in c_plots()


    :param save_path: path to run dir containing psignifit_thresholds.csv, where plots will be save.
    :param isi_name_list: Default None: can input a list of names of ISIs,
            useful if I only have data for a few ISI values.
    :param show_plots: Default True
    :param verbose: Default True.
    """
    print("\n*** running c_plots() ***\n")

    # load df mean of last n probeLum values (14 stairs x 8 isi).
    thr_csv_name = f'{save_path}{os.sep}psignifit_thresholds.csv'
    psig_thr_df = pd.read_csv(thr_csv_name)
    if verbose:
        print(f'psig_thr_df:\n{psig_thr_df}')

    if isi_name_list is None:
        isi_name_list = list(psig_thr_df.columns[4:])
    if verbose:
        print(f'isi_name_list: {isi_name_list}')

    psig_thr_df = psig_thr_df.drop(['stair'], axis=1)
    if 'separation' in list(psig_thr_df.columns):
        sep_col_s = psig_thr_df.pop('separation')
    if 'stair_names' in list(psig_thr_df.columns):
        stair_names_list = list(psig_thr_df['stair_names'])
        print(f'stair_names_list: {stair_names_list}')

    if 'congruent' in list(psig_thr_df.columns):
        cong_col_s = psig_thr_df.pop('congruent')

    psig_thr_df.columns = ['stair_names']+isi_name_list
    if verbose:
        print(f'\npsig_thr_df:\n{psig_thr_df}')

    if verbose:
        print('\npreparing data for batman plots')
    # note: for sym_sep_list just a single value of zero, no -.10
    symm_sep_indices = [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]
    sym_sep_list = [-18, -6, -3, -2, -1, 0, 1, 2, 3, 6, 18]

    psig_cong_sep_df = psig_thr_df[psig_thr_df['stair_names'] >= 0]
    psig_cong_sym_df = psig_cong_sep_df.iloc[symm_sep_indices]
    psig_cong_sym_df.reset_index(drop=True, inplace=True)

    psig_incong_sep_df = psig_thr_df[psig_thr_df['stair_names'] < 0]
    psig_incong_sym_df = psig_incong_sep_df.iloc[symm_sep_indices]
    psig_incong_sym_df.reset_index(drop=True, inplace=True)

    # mean of pos and neg stair_name values [18, 6, 3, 2, 1, 0, 1, 2, 3, 6, 18]
    psig_sym_thr_mean_df = pd.concat([psig_cong_sym_df, psig_incong_sym_df]).groupby(level=0).mean()

    # subtract the dfs from each other, then for each column get the sum of abs values
    diff_val = np.sum(abs(psig_cong_sym_df - psig_incong_sym_df), axis=0)
    # take the mean of these across all ISIs to get single value
    mean_diff_val = float(np.mean(diff_val))

    # add sep column into dfs
    psig_sym_thr_mean_df.insert(0, 'separation', sym_sep_list)
    psig_cong_sym_df.insert(0, 'separation', sym_sep_list)
    psig_incong_sym_df.insert(0, 'separation', sym_sep_list)

    if verbose:
        print(f'\npsig_cong_sym_df:\n{psig_cong_sym_df}')
        print(f'\npsig_incong_sym_df:\n{psig_incong_sym_df}')
        print(f'\npsig_sym_thr_mean_df:\n{psig_sym_thr_mean_df}')
        print(f'\ndiff_val:\n{diff_val}')
        print(f'\nmean_diff_val: {mean_diff_val}')

    # # Figure1 - runs-{n}lastValues
    # this is a figure with one axis per isi, showing neg and pos sep
    # (e.g., -18:18) - eight batman plots
    fig_title = f'MIRRORED Psignifit thresholds per ISI. ' \
                f'(mean diff: {round(mean_diff_val, 2)})'
    fig1_savename = f'MIRRORED_runs.png'

    multi_batman_plots(mean_df=psig_sym_thr_mean_df,
                       thr1_df=psig_cong_sym_df,
                       thr2_df=psig_incong_sym_df,
                       fig_title=fig_title,
                       isi_name_list=isi_name_list,
                       x_tick_vals=sym_sep_list,
                       x_tick_labels=sym_sep_list,
                       sym_sep_diff_list=diff_val,
                       save_path=save_path,
                       save_name=fig1_savename,
                       verbose=True)
    if show_plots:
        plt.show()
    plt.close()


    #  (figure2 doesn't exist in Martin's script - but I'll keep their numbers)

    # # FIGURE3 - 'data.png' - all ISIs on same axis, pos and neg sep, looks like batman.
    # # use plot_data_unsym_batman()
    fig3_save_name = f'data_pos_and_neg.png'
    fig_3_title = 'All ISIs and separations\n' \
                  '(positive values for congruent probe/flow motion, ' \
                  'negative for incongruent).'

    psig_thr_df = psig_thr_df.sort_values(by=['stair_names'])
    psig_thr_df.drop('stair_names', axis=1, inplace=True)
    psig_thr_df.reset_index(drop=True, inplace=True)
    psig_thr_idx_list = list(psig_thr_df.index)
    stair_names_list = sorted(stair_names_list)
    stair_names_list = ['-0' if i == -.10 else int(i) for i in stair_names_list]

    if verbose:
        print(f'\npsig_thr_df:\n{psig_thr_df}')
        print(f'\npsig_thr_idx_list: {psig_thr_idx_list}')
        print(f'\nstair_names_list: {stair_names_list}')

    plot_data_unsym_batman(pos_and_neg_sep_df=psig_thr_df,
                           fig_title=fig_3_title,
                           save_path=save_path,
                           save_name=fig3_save_name,
                           isi_name_list=isi_name_list,
                           x_tick_values=psig_thr_idx_list,
                           x_tick_labels=stair_names_list,
                           verbose=True)
    if show_plots:
        plt.show()
    plt.close()

    #########
    # fig to compare congruent and incongruent thr for each ISI

    if 'congruent' not in list(psig_thr_df.columns):
        psig_thr_df.insert(0, 'congruent', cong_col_s)
    if 'separation' not in list(psig_thr_df.columns):
        psig_thr_df.insert(1, 'separation', sep_col_s)
    if 'stair_names' in list(psig_thr_df.columns):
        psig_thr_df.drop('stair_names', axis=1, inplace=True)

    isi_cols_list = list(psig_thr_df.columns)[-len(isi_name_list):]
    if verbose:
        print(f'psig_thr_df:\n{psig_thr_df}')
        print(f'isi_name_list: {isi_name_list}')
        print('\nfig4 single run data')
    fig4_title = 'Congruent and incongruent probe/flow motion for each ISI'
    fig4_savename = 'compare_data_isi'

    plot_isi_x_axis_w_errors(wide_df=psig_thr_df, cols_to_keep=['congruent', 'separation'],
                             cols_to_change=isi_name_list,
                             cols_to_change_show='probeLum', new_col_name='ISI',
                             strip_from_cols='isi',
                             x_axis='separation', y_axis='probeLum', x_tick_vals=[0, 1, 2, 3, 6, 18],
                             hue_var='ISI', style_var='congruent', style_order=[1, -1],
                             error_bars=True, even_spaced_x=True, jitter=False,
                             fig_title=fig4_title, fig_savename=fig4_savename, verbose=True)
    if show_plots:
        plt.show()
    plt.close()

    if verbose:
        print('\nfig5 single run data')
    fig5_title = 'Congruent and incongruent probe/flow motion for each separation'
    fig5_savename = 'compare_data_sep'
    plot_isi_x_axis_w_errors(wide_df=psig_thr_df, cols_to_keep=['congruent', 'separation'],
                             cols_to_change=isi_name_list,
                             cols_to_change_show='probeLum', new_col_name='ISI',
                             strip_from_cols='isi',
                             x_axis='ISI', y_axis='probeLum',
                             hue_var='separation', style_var='congruent', style_order=[1, -1],
                             error_bars=True,
                             even_spaced_x=True,
                             jitter=False,
                             fig_title=fig5_title,
                             fig_savename=fig5_savename, x_tick_vals=[1, 4, 6],
                             verbose=True)
    if show_plots:
        plt.show()
    plt.close()

    print("\n***finished c_plots()***\n")


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
    this has two versions:
    a. x-axis is separation, ISI as different lines
    b. x-axis is ISI, separation as different lines
    Heatmap: with average probe lum for ISI and separation.

    :param root_path: dir containing run folders
    :param run_dir_names_list: names of run folders
    :param error_bars: Default: None. Can pass sd or se for standard deviation or error.
    :param trim_n: default None.  If int is passed, will call function trim_n_high_n_low(),
            which trims the n highest and lowest values.
    :param show_plots: default True - display plots
    :param verbose: Default true, print progress to screen

    :returns: ave_psignifit_thr_df: (trimmed?) mean threshold for each separation and ISI.
    """

    print("\n***running d_average_participant()***\n")

    """ part1. Munge data, save master lists and get means etc
     - loop through runs and get each P-next-thresholds and P-reversal4-thresholds
    Make master sheets: MASTER_next_thresh & MASTER_reversal_4_thresh
    Incidentally the MATLAB script didn't specify which reversals data to use,
    although the figures imply Martin used last3 reversals."""

    all_psignifit_list = []
    for run_idx, run_name in enumerate(run_dir_names_list):

        this_psignifit_df = pd.read_csv(f'{root_path}{os.sep}{run_name}{os.sep}psignifit_thresholds.csv')
        print(f'{run_idx}. {run_name} - this_psignifit_df:\n{this_psignifit_df}')

        if 'Unnamed: 0' in list(this_psignifit_df):
            this_psignifit_df.drop('Unnamed: 0', axis=1, inplace=True)

        this_psignifit_df.drop(columns='stair', inplace=True)

        isi_name_list = list(this_psignifit_df.columns)[3:]

        rows, cols = this_psignifit_df.shape
        this_psignifit_df.insert(0, 'stack', [run_idx] * rows)

        if verbose:
            print(f'\nthis_psignifit_df:\n{this_psignifit_df}')

        all_psignifit_list.append(this_psignifit_df)


    # join all stacks (runs/groups) data and save as master csv
    all_data_psignifit_df = pd.concat(all_psignifit_list, ignore_index=True)
    all_data_psignifit_df.to_csv(f'{root_path}{os.sep}MASTER_psignifit_thresholds.csv', index=False)
    if verbose:
        print(f'\nall_data_psignifit_df:\n{all_data_psignifit_df}')

    """Part 2: trim highest and lowest values is required and get average vals and errors"""
    # # trim highest and lowest values
    if trim_n is not None:
        trimmed_df = trim_n_high_n_low(all_data_psignifit_df, trim_from_ends=trim_n,
                                       reference_col='stair_names',
                                       stack_col_id='stack',
                                       verbose=verbose)
        get_means_df = trimmed_df
    else:
        get_means_df = all_data_psignifit_df

    # # get means and errors
    groupby_sep_df = get_means_df.drop('stack', axis=1)
    groupby_sep_df = groupby_sep_df.drop('congruent', axis=1)
    groupby_sep_df = groupby_sep_df.drop('separation', axis=1)

    ave_psignifit_thr_df = groupby_sep_df.groupby('stair_names', sort=True).mean()
    if verbose:
        print(f'\nave_psignifit_thr_df:\n{ave_psignifit_thr_df}')

    if error_bars in [False, None]:
        error_bars_df = None
    elif error_bars.lower() in ['se', 'error', 'std-error', 'standard error', 'standard_error']:
        error_bars_df = groupby_sep_df.groupby('stair_names', sort=True).sem()
    elif error_bars.lower() in ['sd', 'stdev', 'std_dev', 'std.dev', 'deviation', 'standard_deviation']:
        error_bars_df = groupby_sep_df.groupby('stair_names', sort=True).std()
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
        a) Sep on x-axis, different line for each ISI
        b) ISI on x-axis, different line for each Sep"""

    stair_names_list = sorted(list(all_data_psignifit_df['stair_names'].unique()))
    stair_names_list = [-.1 if i == -.10 else int(i) for i in stair_names_list]
    stair_names_labels = ['-0' if i == -.10 else int(i) for i in stair_names_list]
    print(f"\nstair_names_list: {stair_names_list}")
    print(f"stair_names_labels: {stair_names_labels}")

    print(f"\nfig_1a\n")
    if trim_n is not None:
        fig_1a_title = f'Average thresholds per ISI across all runs (trim={trim_n}).\n' \
                       f'(positive values for congruent probe/flow motion, negative for incongruent).'
        fig_1a_savename = f'ave_TM_thr_pos_and_neg.png'
    else:
        fig_1a_title = f'Average threshold per ISI across all runs.\n' \
                       f'(positive values for congruent probe/flow motion, negative for incongruent).'
        fig_1a_savename = f'ave_thr_pos_and_neg.png'


    plot_runs_ave_w_errors(fig_df=ave_psignifit_thr_df, error_df=error_bars_df,
                           jitter=True, error_caps=True, alt_colours=False,
                           legend_names=isi_name_list,
                           x_tick_vals=stair_names_list,
                           x_tick_labels=stair_names_labels,
                           even_spaced_x=False,
                           fixed_y_range=False,
                           fig_title=fig_1a_title, save_name=fig_1a_savename,
                           save_path=root_path, verbose=True)
    if show_plots:
        plt.show()
    plt.close()
    #
    # if verbose:
    #     print('finished fig_1a')
    #
    # # todo: transpose version e.gh., isi on x-axis
    #
    # ####################
    # # can't use plot_runs_ave_w_errors for comparrisons between cong/incong
    # print('\nnew fig')
    # ave_thr_df_T = ave_psignifit_thr_df.T
    # error_bars_df_T = error_bars_df.T
    # print(f'ave_thr_df_T:\n{ave_thr_df_T}')
    # print(f'error_bars_df_T:\n{error_bars_df_T}')
    #
    # ave_thr_df_T.columns = [str(i) for i in list(ave_thr_df_T.columns)]
    # error_bars_df_T.columns = [str(i) for i in list(error_bars_df_T.columns)]
    # col_names = list(ave_thr_df_T.columns)
    # print(f'col_names: {col_names}')
    # print(f'col_names: {type(col_names[0])}')

    # print(f"\nfig_1b")
    # if trim_n is not None:
    #     fig_1b_title = f'Average thresholds per separation across all runs (trim={trim_n}).'
    #     fig_1b_savename = f'ave_TM_thr_all_runs_sep.png'
    # else:
    #     fig_1b_title = f'Average threshold per separation across all runs'
    #     fig_1b_savename = f'ave_thr_all_runs_sep.png'
    #
    # plot_runs_ave_w_errors(fig_df=ave_psignifit_thr_df.T, error_df=error_bars_df.T,
    #                        jitter=True, error_caps=True, alt_colours=False,
    #                        legend_names=ave_thr_df_T.columns,
    #                        x_tick_vals=isi_name_list,
    #                        x_tick_labels=isi_name_list,
    #                        even_spaced_x=False,
    #                        fixed_y_range=False,
    #                        fig_title=fig_1b_title, save_name=fig_1b_savename,
    #                        save_path=root_path, verbose=True)
    # if show_plots:
    #     plt.show()
    # plt.close()
    #
    # if verbose:
    #     print('finished fig_1b')
    ####################


    # print(f"\nfig_1c\n")
    # # fig 1c, ISI on x-axis, different line for each sep
    # print('fig2a participant average with errors')
    # save_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/radial_flow_exp/'
    # thr_csv_name = f'{save_path}{os.sep}MASTER_psignifit_thresholds.csv'
    # ave_thr_df = pd.read_csv(thr_csv_name)

    # wide_df = ave_thr_df
    # print(f'wide_df:\n{wide_df}')

    print(f"\nfig_1a\n")
    if trim_n is not None:
        fig_1a_title = f'Average thresholds per ISI across all runs (trim={trim_n}).'
        fig_1a_savename = f'ave_TM_thr_all_runs_isi.png'
    else:
        fig_1a_title = f'Average threshold per ISI across all runs'
        fig_1a_savename = f'ave_thr_all_runs_isi.png'

    plot_isi_x_axis_w_errors(wide_df=get_means_df, cols_to_keep=['congruent', 'separation'],
                             cols_to_change=['isi1', 'isi4', 'isi6'],
                             cols_to_change_show='probeLum', new_col_name='ISI',
                             strip_from_cols='isi',
                             x_axis='separation', y_axis='probeLum',
                             hue_var='ISI', style_var='congruent', style_order=[1, -1],
                             error_bars=True,
                             even_spaced_x=True,
                             jitter=True,
                             fig_title=fig_1a_title,
                             fig_savename=fig_1a_savename, x_tick_vals=[0, 1, 2, 3, 6, 18],
                             verbose=True)
    if show_plots:
        plt.show()
    plt.close()

    print(f"\nfig_1b")
    if trim_n is not None:
        fig_1b_title = f'Average thresholds per separation across all runs (trim={trim_n}).'
        fig_1b_savename = f'ave_TM_thr_all_runs_sep.png'
    else:
        fig_1b_title = f'Average threshold per separation across all runs'
        fig_1b_savename = f'ave_thr_all_runs_sep.png'

    plot_isi_x_axis_w_errors(wide_df=get_means_df, cols_to_keep=['congruent', 'separation'],
                             cols_to_change=['isi1', 'isi4', 'isi6'],
                             cols_to_change_show='probeLum', new_col_name='ISI',
                             strip_from_cols='isi',
                             x_axis='ISI', y_axis='probeLum',
                             hue_var='separation', style_var='congruent', style_order=[1, -1],
                             error_bars=True,
                             even_spaced_x=True,
                             jitter=.01,
                             fig_title=fig_1b_title,
                             fig_savename=fig_1b_savename, x_tick_vals=['ISI 1', 'ISI 4', 'ISI 6'],
                             verbose=True)
    if show_plots:
        plt.show()
    plt.close()

    #
    # if trim_n is not None:
    #     # fig_1c_title = f'Probe luminance at each ISI value per separation (trim={trim_n}).'
    #     # fig_1c_savename = f'ave_TM_thr_all_runs_transpose.png'
    #     fig_1c_title = f'Congruent and incongruent probe/flow motion for each ISI (trim={trim_n})'
    #     fig_1c_savename = 'compare_mean_data_TM'
    # else:
    #     # fig_1c_title = f'Probe luminance at each ISI value per separation'
    #     # fig_1c_savename = f'ave_thr_all_runs_transpose.png'
    #     fig_1c_title = 'Congruent and incongruent probe/flow motion for each ISI'
    #     fig_1c_savename = 'compare_mean_data'
    #
    # print(f"\nget_means_df\n{get_means_df}")
    #
    # isi_cols_list = list(get_means_df.columns)[-len(isi_name_list):]
    # if verbose:
    #     print(f'get_means_df:\n{get_means_df}')
    #     print(f'isi_name_list: {isi_name_list}')
    #
    # # todo: use function in scratch
    # # todo: add jitted x_axis values such that each isi is separate (but congruent/incongruent are aligned)
    # # todo: add error bars
    #
    # # convert wide df to long
    # long_df = make_long_df(wide_df=get_means_df, cols_to_keep=['congruent', 'separation'],
    #                        cols_to_change=isi_cols_list, cols_to_change_show='probeLum',
    #                        new_col_name='ISI', strip_from_cols='isi', verbose=True)
    #
    # # make plot with pos sep only
    # # same colour with dashed or dotted for congruent/incongruent
    # fig, ax = plt.subplots(figsize=(10, 6))
    # sns.lineplot(data=long_df,
    #              x='separation', y='probeLum', hue='ISI', style='congruent',
    #              style_order=[1, -1],
    #              markers=True, dashes=True,
    #              estimator=np.mean, ci='sd', err_style='bars',
    #              ax=ax)
    # ax.set_xticks(sorted(list(long_df['separation'].unique())))
    # plt.title(fig_1c_title)
    #
    # # Change legend labels for congruent and incongruent data
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles=handles, labels=labels[:-2] + ['True', 'False'])
    #
    # plt.savefig(f'{root_path}{os.sep}{fig_1c_savename}')
    #
    # if show_plots:
    #     plt.show()
    # plt.close()
    #
    # if verbose:
    #     print('finished fig_1c')
    #
    #
    print(f"\nHeatmap\n")
    # get mean of each col, then mean of that

    if trim_n is not None:
        heatmap_title = f'Mean Threshold for each ISI and separation (trim={trim_n}).'
        heatmap_savename = 'mean_TM_thr_heatmap'
    else:
        heatmap_title = 'Mean Threshold for each ISI and separation'
        heatmap_savename = 'mean_thr_heatmap'

    plot_thr_heatmap(heatmap_df=ave_psignifit_thr_df,
                     x_tick_labels=isi_name_list,
                     y_tick_labels=stair_names_list,
                     fig_title=heatmap_title,
                     save_name=heatmap_savename,
                     save_path=root_path,
                     verbose=True)
    if show_plots:
        plt.show()
    plt.close()

    if trim_n is not None:
        heatmap_title = f'Mean Threshold for each ISI and separation 2(trim={trim_n}).'
        heatmap_savename = 'mean_TM_thr_heatmap2'
    else:
        heatmap_title = 'Mean Threshold for each ISI and separation2'
        heatmap_savename = 'mean_thr_heatmap2'

    plot_thr_heatmap(heatmap_df=ave_psignifit_thr_df.T,
                     x_tick_labels=stair_names_list,
                     y_tick_labels=isi_name_list,
                     fig_title=heatmap_title,
                     save_name=heatmap_savename,
                     save_path=root_path,
                     verbose=True)
    if show_plots:
        plt.show()
    plt.close()

    print("\n*** finished d_average_participant()***\n")

    return ave_psignifit_thr_df
