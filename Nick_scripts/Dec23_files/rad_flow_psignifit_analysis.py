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

'''
These first two functions are to be used to take the rgb255 values from radial flow experiments that were run on the uncalibrated monitor (asus_2_13_240Hz), 
and use the luminance profiles measured with the spyder to find the corresponding rgb255 values for the calibrated monitor (asus_cal).
'''




def trim_n_high_n_low(all_data_df, trim_from_ends=None, reference_col='separation',
                      stack_col_id='stack', verbose=True):
    """
    Function for trimming the n highest and lowest values from each condition of
    a set with multiple runs.

    :param all_data_df: Dataset to be trimmed.
    :param trim_from_ends: number of values to trim from each end of the distribution.
    :param reference_col: Idx column containing repeated conditions (e.g., separation has the same label for each stack).
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
        target_3d_depth = depth_3d - 2 * trim_from_ends
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
            #     print(f'{counter}: {trimmed}')
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
    sep_col_vals = sep_list * target_3d_depth
    trimmed_df.insert(0, 'stack', stack_col_vals)
    trimmed_df.insert(1, reference_col, sep_col_vals)
    print(f'\ntrimmed_df {trimmed_df.shape}\n{trimmed_df.dtypes}:\n{trimmed_df}')

    print(f'trimmed {trim_from_ends} highest and lowest values ({2 * trim_from_ends} in total) from each of the '
          f'{datapoints_per_cond} datapoints so there are now '
          f'{target_3d_depth} datapoints for each of the '
          f'{rows_3d} x {cols_all} conditions.')

    print('\n*** finished trim_high_n_low() ***')

    return trimmed_df


def make_long_df(wide_df,
                 cols_to_keep=['congruent', 'separation'],
                 cols_to_change=['ISI_1', 'ISI_4', 'ISI_6'],
                 cols_to_change_show='newLum',
                 new_col_name='ISI', strip_from_cols='ISI_', verbose=True):
    """
    Function to convert wide-form_df to long-form_df.  e.g., if there are several
    columns showing ISIs (cols_to_change), this puts them all into one column (new_col_name).

    :param wide_df: dataframe to be changed.
    :param cols_to_keep: Columns to use for indexing (e.g., ['congruent', 'separation'...etc]).
    :param cols_to_change: List of columns showing data at different levels e.g., [ISI_1, ISI_4, ISI_6...etc].
    :param cols_to_change_show: What is being measured in repeated cols, e.g., newLum, probeLum.
    :param new_col_name: name for new col describing levels e.g. isi
    :param strip_from_cols: string to strip from col names when for new cols.
        e.g., if strip_from_cols='ISI_', then [ISI_1, ISI_4, ISI_6] becomes [1, 4, 6].
    :param verbose: if true, prints progress to screen.

    :return: long_df
    """
    print("\n*** running make_long_df() ***\n")

    new_col_names = cols_to_keep + [new_col_name] + [cols_to_change_show]

    # make longform data
    if verbose:
        print(f"preparing to loop through: {cols_to_change}")
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

            # Dataframe should have one dtype so using float
            try:
                this_col = float(this_col)
            except ValueError:
                pass

        this_df.insert(len(cols_to_keep), new_col_name, [this_col] * len(this_df.index))
        this_df.columns = new_col_names
        long_list.append(this_df)

    long_df = pd.concat(long_list)
    long_df.reset_index(drop=True, inplace=True)


    if verbose:
        print(f'long_df:\n{long_df}')

    print("\n*** finished make_long_df() ***\n")

    return long_df

#

def get_OLED_luminance(rgb1):
    """
    This function takes a list of rgb1 values and returns a list of luminance values.
    New luminance values are calculated using a polynomial fit to the measured values.
    Measurements were taken on 17/11/2023 by Nick with spyderX pro running DisplayCal on MacBook.
    :param rgb1: value to be converted to luminance
    :return: corresponding luminance value
    """

    '''data to use for fitting'''
    # measurements were taken at 18 evenly spaced rgb1 points between 0 and 1
    rbg1_values = [0, 0.058823529, 0.117647059, 0.176470588, 0.235294118, 0.294117647, 0.352941176,
                   0.411764706, 0.470588235, 0.529411765, 0.588235294, 0.647058824, 0.705882353,
                   0.764705882, 0.823529412, 0.882352941, 0.941176471, 1]

    # measured luminance values for each of the rgb1 values
    measured_lum = [0.01, 0.17, 0.48, 0.91, 1.55, 2.45, 3.58,
                    4.91, 6.49, 8.4, 10.37, 12.77, 13.03,
                    16.3, 19.61, 23.26, 24.78, 24.8]

    # because of the kink in the data (after rgb1=.64) I'm just going to take the first 12 values
    measured_lum = measured_lum[:12]
    rbg1_values = rbg1_values[:12]

    '''fitting the curve'''
    # calculate polynomial to fit the leasured values.
    z = np.polyfit(rbg1_values, measured_lum, 3)
    f = np.poly1d(z)

    '''getting correct luminance values'''
    if type(rgb1) == list:
        rgb1 = np.array(rgb1)
    new_lum = f(rgb1)

    return new_lum


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

    :param n_plots: number of plots
    :return: n_rows, n_cols
    """

    if n_plots > 25:
        raise ValueError(f"\t\tToo many plots for this function: {n_plots}\n\n")

    # ideally have no more than 4 rows, unless more than 16 plots
    if n_plots == 4:
        n_rows = n_cols = 2
    elif n_plots == 9:
        n_rows = n_cols = 3
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


def get_ax_idx(plot_idx, n_rows, n_cols):
    """
    Give a set of subplots with shape (n_rows, n_cols), and a plot number (zero indexed),
    return the index of the subplot in the figure.
    :param plot_idx: the number of the plot, starting at 0.
    :param n_rows: number of rows in the figure
    :param n_cols: number of columns in the figure
    :return: tuple with the index of the subplot in the figure (row, col)
    """
    row_idx = plot_idx // n_cols
    col_idx = plot_idx % n_cols
    return row_idx, col_idx




def plot_diff(ave_thr_df, stair_names_col='stair_names', fig_title=None,
              save_path=None, save_name=None,
              x_axis_isi=False, verbose=True):
    """
    Function to plot the difference between congruent and incongruent conditions.
    :param ave_thr_df: Dataframe to use to get differences
    :param stair_names_col: Column name containing separation and congruence values.
    :param fig_title: Title for fig
    :param save_path: path to save fig to
    :param save_name: name to save fig
    :param x_axis_isi: If False, with have Separation on x-axis with different lines for each ISI.
                        If True, will have ISI on x-axis and diffferent lines for each Separation.
    :param verbose: Prints progress to screen
    :return:
    """
    print('*** running plot_diff() ***')

    # if stair_names_col is set as index, move it to regular column and add standard index
    if ave_thr_df.index.name == stair_names_col:
        ave_thr_df.reset_index(drop=False, inplace=True)

    # sort df so it is in ascending order - participant and exp dfs are in diff order to begin so this avoids that complication.
    srtd_ave_thr_df = ave_thr_df.sort_values(by=stair_names_col, ascending=True)
    srtd_ave_thr_df.reset_index(drop=True, inplace=True)

    if verbose:
        print(f'srtd_ave_thr_df:\n{srtd_ave_thr_df}')

    # get rows to slice for each df
    # the should be in opposite order; e.g., cong desc 18, 6, 3...; incong asc -18, -6, -3...
    cong_rows = sorted(srtd_ave_thr_df.index[srtd_ave_thr_df[stair_names_col] >= 0].tolist(), reverse=False)
    incong_rows = sorted(srtd_ave_thr_df.index[srtd_ave_thr_df[stair_names_col] < 0].tolist(), reverse=True)
    if verbose:
        print(f'\ncong_rows: {cong_rows}')
        print(f'incong_rows: {incong_rows}')

    # slice rows for cong and incong df
    cong_df = srtd_ave_thr_df.iloc[cong_rows, :]
    incong_df = srtd_ave_thr_df.iloc[incong_rows, :]

    pos_sep_list = [int(i) for i in list(cong_df[stair_names_col].tolist())]
    cong_df.reset_index(drop=True, inplace=True)
    incong_df.reset_index(drop=True, inplace=True)
    if verbose:
        print(f'\ncong_df: {cong_df.shape}\n{cong_df}')
        print(f'\nincong_df: {incong_df.shape}\n{incong_df}')
        print(f'\npos_sep_list: {pos_sep_list}')


    # check to make sure incong_df is in correct order - e.g., if cong is asc, incong should descend.
    # if last cong is 18, last incong should be -18
    # if last cong is 0, last incong should be -.1
    check_incong_list = [int(i) for i in list(incong_df[stair_names_col].tolist())]
    print(f'\ncheck_incong_list: {check_incong_list}')
    swap_order = False
    use_ascending = False
    if pos_sep_list[0] == -check_incong_list[0]:
        print('yeah, correct order (1)')
    elif pos_sep_list[-1] == -check_incong_list[-1]:
        print('yeah, correct order (2)')
    elif pos_sep_list[-1] == -check_incong_list[0]:  # -18 is first for incong, use ascenind=False
        print('wrong, swap order (1)')
        swap_order = True
        use_ascending = True
    elif pos_sep_list[0] == -check_incong_list[-1]:  # -18 is last for incong, use ascenind=True
        print('wrong, swap order (2)')
        swap_order = True
    else:
        print("I dunno what's doing on!?")
        raise ValueError("cant get correct order for diff df")
    if swap_order:
        incong_df = incong_df.sort_values(by=stair_names_col, ascending=use_ascending)
        incong_df.reset_index(drop=True, inplace=True)
        print(f'\nincong_df: {incong_df.shape}\n{incong_df}')


    # # drop any string or object columns, then put back in later
    df_dtypes = cong_df.dtypes
    print(f"df_dtypes:\n{df_dtypes}")
    cong_df = cong_df.select_dtypes(exclude=['object'])
    print(f"cong_df:\n{cong_df}")
    incong_df = incong_df.select_dtypes(exclude=['object'])
    print(f"incong_df:\n{incong_df}")

    # subtract one from the other
    diff_df = cong_df - incong_df
    diff_df.drop(stair_names_col, inplace=True, axis=1)

    if x_axis_isi:
        diff_df = diff_df.T
        diff_df.columns = pos_sep_list
        isi_names_list = list(diff_df.index)
        x_tick_labels = isi_names_list
        x_axis_label = 'ISI'
        legend_title = 'Separation'
    else:
        x_tick_labels = pos_sep_list
        x_axis_label = 'Separation'
        legend_title = 'ISI'

    if verbose:
        print(f'\ndiff_df:\n{diff_df}')
        print(f'\nx_axis_label: {x_axis_label}')
        print(f'\nlegend_title: {legend_title}')

    # make plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=diff_df, ax=ax)

    # decorate plot
    if fig_title:
        plt.title(fig_title)
    plt.axhline(y=0, color='lightgrey', linestyle='dashed')
    ax.set_xticks(list(range(len(x_tick_labels))))
    ax.set_xticklabels(x_tick_labels)
    plt.xlabel(x_axis_label)
    plt.ylabel("Difference in luminance (cong-incong)")
    ax.legend(title=legend_title)

    if save_name:
        plt.savefig(os.path.join(save_path, save_name))
        print(f'plt saved to: {os.path.join(save_path, save_name)}')

    print('*** finished plot_diff() ***')

    return fig


def joined_plot(untrimmed_df, x_cols_str='isi_ms',
                hue_col_name='congruent', hue_labels=['congruent', 'incongruent'],
                participant_name='',
                x_label='ISI (ms)', y_label='Probe Luminance',
                extra_text=None,
                save_path=None, save_name=None,
                verbose=True):
    """
    y-axis is a continuous variable (e.g., probeLum, probeSpeed)
    x_axis is a series of categorical variables (e.g., ISI or probe duration)
    hue is a comparrison between two conditions (e.g., congruent vs incongruent, or inward vs outward)
    This plot does a couple of things.
    1. It has a line plot showing means for the hue variables at each x_axis value, with error bars.
        The means are slightly offset so they error bars don't overlap.
        Hue[0] values on left and hue[1] on right of x-axis labels.
    2. There are scatter plots for each individual measurement,
        with lines joining datapoints for hue variables from the the same experiental session.
        These are offset to the left and right of the x-axis labels.

    As such, the plot will have 5 positions for each x value (including spaces):
    [space, hue[0] scatter, means, hue[1] scatter, space]

    :param untrimmed_df: Dataframe containing values from All (e.g., not trimmed) runs (or participants).
        There should be a column for a hue variable,
        and separate columns for each condition, beginning with the same string (e.g., isi_ms_0, isi_ms_16 etc)
    :param x_cols_str: string which identifies the columns relating to the x_axis (e.g., 'isi_ms' or 'probe_dur_ms')
    :param hue_col_name: Name of column showing comparison variable (e.g., 'Congruent' or 'flow_dir')
    :param hue_labals: Labels to attach to values in hue column
        (e.g., ['Incongruent', 'Congruent'] or ['Inward', 'Outward'])
    :param participant_name: Name of participant (or experiment if experiment averages)
    :param extra_text: Extra text to add to title and save name (e.g., separation, bg motion duration etc)
    :param save_path: Path to save figs to
    :param save_name: name to save fig as
    :param verbose: How much to print to screen
    """
    print("\n\n*** running joined_plot()***\n")

    # if all_df is a string, this is the filepath to the df, open the file
    if isinstance(untrimmed_df, pd.DataFrame):
        pass
    else:
        untrimmed_df = pd.read_csv(untrimmed_df)

    # copy dataframe so that changes to it don't impact original dataframe
    all_df = untrimmed_df.copy()

    # get list of all columns containing x_cols_str (e.g., 'isi_ms' or 'probe_dur_ms')
    not_x_col_list = [col for col in all_df.columns if x_cols_str not in col]
    x_col_list = [col for col in all_df.columns if x_cols_str in col]


    '''Part 1.
    get dataframe for scatterplots, with hue[0] and hue[1] (e.g., 'cong_' and 'incong_') columns 
    for each x_cols_str (e.g., isi) column.
    Instead of separate rows for each hue value with x_cols_str values in separate columns,
    (e.g.,          x[0], x[1], x[2]
            Hue[0]  val1, val3, val5
            Hue[1]  val2, val4, val6)

    There is now twice the number of columns and half the number of rows.
    e.g.,  x[0]_hue[0], x[0]_hue[1], x[1]_hue[0], x[1]_hue[1], x[2]_hue[0], x[2]_hue[1]
            val1      , val2       , val3       , val4       , val5       , val6'''

    # drop hue_col_name from not_x_col_list
    not_x_col_list.remove(hue_col_name)
    print(f"\nx_col_list: {x_col_list}")

    # get shorter hue labels for renaming columns
    if len(hue_labels[0]) > 4:
        short_hue = [hue_labels[0][:4], hue_labels[1][:4]]
        # if short labels are the same, use the full version of the second label
        if short_hue[0] == short_hue[1]:
            short_hue[1] = hue_labels[1]


    # get dataframe just with hue_labels[0] values (e.g., incongruent or outward)
    x_hue0_df = all_df.copy()
    x_hue0_df = x_hue0_df[x_hue0_df[hue_col_name] == -1]
    x_hue0_df = x_hue0_df.drop(columns=[hue_col_name])

    # get number of runs (stack)
    if 'participant' in x_hue0_df.columns:
        n_runs = len(x_hue0_df['participant'].unique())
    else:
        n_runs = len(x_hue0_df['stack'].unique())

    # add hue[0] to each column name in x_col_list, then rename columns in x_hue0_df
    x_hue0_col_list = [f"{short_hue[0]}_{col}" for col in x_col_list]
    # rename columns in x_col_list as x_hue0_df
    for idx, col_name in enumerate(x_col_list):
        x_hue0_df.rename(columns={col_name: x_hue0_col_list[idx]}, inplace=True)
    print(f"\nx_hue0_df: ({list(x_hue0_df.columns)})\n{x_hue0_df}")

    # get dataframe just with hue_labels[1] values
    x_hue1_df = all_df.copy()
    x_hue1_df = x_hue1_df[x_hue1_df[hue_col_name] == 1]
    x_hue1_df = x_hue1_df.drop(columns=[hue_col_name])
    # add hue_labsls[1] to each column name in x_col_list
    x_hue1_col_list = [f"{short_hue[1]}_{col}" for col in x_col_list]
    # rename columns in x_col_list as x_hue1_col_list
    for idx, col_name in enumerate(x_col_list):
        x_hue1_df.rename(columns={col_name: x_hue1_col_list[idx]}, inplace=True)
    print(f"\nx_hue1_df: ({list(x_hue1_df.columns)})\n{x_hue1_df}")


    # alternate column names from x_and_hue_0_col_list and x_and_hue_1_col_list
    all_x_col_list = []
    for idx in range(len(x_col_list)):
        all_x_col_list.append(x_hue0_col_list[idx])
        all_x_col_list.append(x_hue1_col_list[idx])
    print(f"\nall_x_col_list: {all_x_col_list}")

    # prepare x_hue1_df and x_hue0_df to be joined
    x_hue0_df.drop(columns=not_x_col_list, inplace=True)
    x_hue1_df.drop(columns=not_x_col_list, inplace=True)
    x_hue0_df.reset_index(drop=True, inplace=True)
    x_hue1_df.reset_index(drop=True, inplace=True)

    # just_hue_df is x_hue0_df and x_hue1_df joined, with columns for each x and hue combination.
    just_hue_df = pd.concat([x_hue0_df, x_hue1_df], axis=1, join='inner')
    print(f"\njust_hue_df: ({list(just_hue_df.columns)})\n{just_hue_df}")

    # change order of columns in just_hue_df to match all_col_list
    just_hue_df = just_hue_df[all_x_col_list]
    print(f"\njust_hue_df: ({list(just_hue_df.columns)})\n{just_hue_df}")


    '''Part 2
    Make long_df for lineplot, with x_ms in one column, and hue_col_name in another column'''
    # make long_df, moving x_cols_str_ to single x_cols_str column (e.g. all 'isi_ms_' columns moved to single column 'isi_ms')
    cols_to_change = [col for col in all_df.columns if f'{x_cols_str}_' in col]
    long_df = make_long_df(wide_df=all_df,
                           cols_to_keep=[hue_col_name],
                           cols_to_change=cols_to_change,
                           cols_to_change_show='y_values',
                           new_col_name=x_cols_str, strip_from_cols=f'{x_cols_str}_', verbose=verbose)

    # get a list of isi values in long_df
    x_vals_list = long_df[x_cols_str].unique().tolist()

    # for each value in x_vals_list, make a new list which is i*5+3
    x_mean_pos_list = [i * 5 + 2 for i in range(len(x_vals_list))]
    print(f"\nx_mean_pos_list: {x_mean_pos_list}")
    long_df['x_pos'] = long_df[x_cols_str].map(dict(zip(x_vals_list, x_mean_pos_list)))

    # create a new column, 'x_dodge_pos', with values from x_pos column,
    # but if hue_col_name is -1, subtract .1, if hue_col_name is 1, add .1
    long_df['x_dodge_pos'] = long_df['x_pos']
    long_df.loc[long_df[hue_col_name] == -1, 'x_dodge_pos'] = long_df['x_dodge_pos'] - .1
    long_df.loc[long_df[hue_col_name] == 1, 'x_dodge_pos'] = long_df['x_dodge_pos'] + .1

    print(f"\nlong_df:\n{long_df}")

    '''part 3.
    plot lineplot with error bars'''
    fig, ax = plt.subplots()

    # plot means with error bars
    sns.lineplot(data=long_df,
                 x='x_dodge_pos', y='y_values', hue=hue_col_name,
                 palette=sns.color_palette("tab10", n_colors=2),
                 linewidth=3,
                 errorbar='se', err_style='bars',
                 err_kws={'capsize': 5, 'elinewidth': 2, 'capthick': 2},
                 ax=ax
                 )

    '''part 4.
    plot scatter plot, with pairs of datapoints joined for hue variables from the same experimental session'''

    print(f"\njust_hue_df:\n{just_hue_df}")

    # create jitter for x positions
    jitter = 0.0  # use 0.05 for jitter
    df_x_jitter = pd.DataFrame(np.random.normal(loc=0, scale=jitter, size=just_hue_df.values.shape), columns=just_hue_df.columns)

    # we are going to add to the jitter values to put them either side of the mean x pos
    x_scatter_x_pos_list = []
    for val in x_mean_pos_list:
        x_scatter_x_pos_list.append(val - 1)
        x_scatter_x_pos_list.append(val + 1)
    df_x_jitter += x_scatter_x_pos_list

    print(f"\nx_mean_pos_list: {x_mean_pos_list}")
    print(f"\nnp.array(x_scatter_x_pos_list):\n{np.array(x_scatter_x_pos_list)}")
    print(f"\ndf_x_jitter:\n{df_x_jitter}")

    # plot scatter plot
    palette_tab10 = sns.color_palette("tab10", 10)
    for idx, col_name in enumerate(list(just_hue_df.columns)):
        if idx % 2 == 0:
            this_colour = palette_tab10[0]
        else:
            this_colour = palette_tab10[1]
        ax.plot(df_x_jitter[col_name], just_hue_df[col_name], 'o', alpha=.40, zorder=1, ms=4, mew=1, color=this_colour)

    # join scatter plot with lines
    for idx in range(0, len(just_hue_df.columns), 2):
        ax.plot(df_x_jitter.loc[:, just_hue_df.columns[idx:idx + 2]].T,
                just_hue_df.loc[:, just_hue_df.columns[idx:idx + 2]].T,
                color='grey', linewidth=0.5, linestyle='--', zorder=-1)

    ax.set_xlim(0, max(x_mean_pos_list) + 2)
    # set x tick labels to x_vals_str_list, which should start at 2, and go up in 5s
    labels_go_here = list(range(2, max(x_mean_pos_list) + 2, 5))
    ax.set_xticks(labels_go_here,
                  labels=x_vals_list)

    # decorate plot, x axis label, legend, suptitle and title
    suptitle_text = f"{participant_name} thresholds, means and SE of {n_runs} runs"
    if 'exp' in participant_name:
        suptitle_text = f"{participant_name} thresholds, means and SE of {n_runs} participants"

    # if y_axis crosses zero, add grey dashed line at zero
    # (e.g., min value is less than zero and max is more than zero)
    if long_df['y_values'].min() < 0 and long_df['y_values'].max() > 0:
        ax.axhline(y=0, linestyle="--", color='grey')
        suptitle_text = suptitle_text + '\n-ive = outwards, +ive = inwards'

    plt.suptitle(suptitle_text)  # big title
    ax.set_title(extra_text)  # smaller title underneath

    # get x tick labels, if '-1.0' or '-1' is first label, change to 'Conc'
    if 'ISI' in x_label.upper():
        # get x tick labels, if '-1.0' or '-1' is first label, change to 'Conc'
        x_tick_labels = ax.get_xticklabels()
        if x_tick_labels[0].get_text() in ['-1', '-1.0', -1, -1.0]:
            x_tick_labels[0].set_text('Concurrent')
        else:
            print(f"-1 not in x_tick_labels: {x_tick_labels}")
        ax.set_xticklabels(x_tick_labels)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(labels=hue_labels, loc='best', framealpha=.3)


    if save_name is None:
        if extra_text is not None:
            # remove any commas and replace any spaces in extra_text with underscores
            extra_text = extra_text.replace(',', '')
            extra_text = extra_text.replace(' ', '_')
            fig_name = f"{participant_name}_{extra_text}_joinedplot.png"
        else:
            fig_name = f"{participant_name}_joinedplot.png"
    else:
        fig_name = f"{save_name}_joinedplot.png"
    if verbose:
        print(f"\n***saving joinedplot to {os.path.join(save_path, fig_name)}***")
    plt.savefig(os.path.join(save_path, fig_name))

    plt.show()



def a_data_extraction_Oct23(p_name, run_dir, save_all_data=True, verbose=True):
    """
    This script is a python version of Martin's first MATLAB analysis scripts, described below.

    a_data_extraction.m: Once a participant has completed a run of all ISIs,
        this script gets all their data into one file, and sorts each isi by stair.

    :param p_name: participant's name as used to save csv files.  e.g., if the
            file is .../nick1.csv, participant name is 'nick1'.
    :param run_dir: directory where isi folders are stored.
    :param save_all_data: If True, will save all_data_df as a xlsx.
    :param verbose: If True, will print progress to screen.

    :return: ALL_ISIs_sorted.xlsx: A pandas DataFrame with n xlsx file of all
        data for one run of all ISIs.
    """
    print("\n***running a_data_extraction()***\n")

    # get run name/number
    path, run = os.path.split(run_dir)
    if verbose:
        print(f"run: {run}")

    all_data_list = []  # list to append each output file to

    '''search for output files and get sub_dir names'''
    # get the names of all subdirectories in run_dir that contain a file with '_output.csv' in the name
    for root, dirs, files in os.walk(run_dir):
        if len(files) > 0:  # if there are files in the directory, check for '_output.csv' in the filename
            for filename in files:
                if '_output.csv' in filename:

                    # check that this is a proper run (e.g., not 'debug' or 'incomplete')
                    if 'debug' not in filename and 'incomplete' not in filename:
                        filepath = os.path.join(root, filename)

                        # load data
                        this_isi_df = pd.read_csv(filepath)
                        if verbose:
                            print(f"loaded csv:\n{this_isi_df.head()}")

                        # remove any Unnamed columns
                        if any("Unnamed" in i for i in list(this_isi_df.columns)):
                            unnamed_col = [i for i in list(this_isi_df.columns) if "Unnamed" in i][0]
                            this_isi_df.drop(unnamed_col, axis=1, inplace=True)

                        # OLED adds extra cols that I don't need
                        if any("thisRow.t" in i for i in list(this_isi_df.columns)):
                            this_row_col = [i for i in list(this_isi_df.columns) if "thisRow.t" in i][0]
                            this_isi_df.drop(this_row_col, axis=1, inplace=True)
                        if any("notes" in i for i in list(this_isi_df.columns)):
                            notes_col = [i for i in list(this_isi_df.columns) if "notes" in i][0]
                            this_isi_df.drop(notes_col, axis=1, inplace=True)

                        # sort by staircase
                        trial_numbers = list(this_isi_df['trial_number'])
                        this_isi_df = this_isi_df.sort_values(by=['stair', 'trial_number'])

                        this_isi_df.insert(1, 'srtd_trial_idx', trial_numbers)
                        if verbose:
                            print(f'df sorted by stair: {type(this_isi_df)}\n{this_isi_df}')

                        # get column names to use on all_data_df
                        column_names = list(this_isi_df)
                        if verbose:
                            print(f'column_names: {len(column_names)}\n{column_names}')

                        # add to all_data_list
                        all_data_list.append(this_isi_df)

    all_data_df = pd.concat(all_data_list)

    if verbose:
        print(f"all_data_df:\n{all_data_df}")

    if save_all_data:
        save_name = 'RUNDATA_sorted.csv'

        save_excel_path = os.path.join(run_dir, save_name)
        if verbose:
            print(f"\nsaving all_data_df to save_excel_path:\n{save_excel_path}")
        all_data_df.to_csv(save_excel_path, index=False)

    print("\n***finished a_data_extraction()***\n")

    return all_data_df


def mean_staircase_plots(per_trial_df, save_path, participant_name, run_col_name='run',
                         thr_col_name='probeLum',
                         isi_col_name='isi_ms', sep_col_name='separation',
                         hue_col_name='congruent', hue_names=['Incongruent', 'Congruent'],
                         ave_type='mean',
                         show_plots=True, save_plots=True, verbose=False):
    """
    Take a MASTER_p_trial_data.csv file and plot the staircases and mean for each ISI and separation.

    :param per_trial_df: Dataframe, name or path to MASTER_p_trial_data.csv file.
        Should have raw data for each trial, not averages or thresholds.
    :param save_path: Path to save figure to.
    :param participant_name: Name of participant (or other identifier) to use in figure title.
    :param isi_col_name: Column name to use for ISI values (or other variable of interest).
    :param sep_col_name: Columns name to use for separation values (or other variable of interest).
    :param run_col_name: Column denoting separate runs of each conditions (e.g., 0, 1, 2, 3 etc)
    :param thr_col_name: Column name to use for threshold values (thing being measured for y_axis).
    :param hue_col_name: Column name to use for hue (e.g., congruent/incongruent).
    :param hue_names: Names to use for hue_col_name (e.g., ['Incongruent', 'Congruent']).
        Should be in same order as numerical values in hue column.
    :param ave_type: 'mean' or 'median' - how to average across runs.
    :param show_plots: Whether to show plots on screen.
    :param save_plots: Whether to save plots to file.
    :param verbose: Whether to print progress to screen.
    """

    print("\n***running mean_staircase()***")

    # # mean staircase for this bg type will loop through ISI and separation,
    # then plot staircases and mean (step v probeLum) for each run and congruenct/incongruent
    '''
    1. Calculate ave thr for each step for congruent and incongruent data.
    2. loop through per_trial_df for each ISI/sep combination.
    3. make plot showing all 12 congruent and incongruent staircases on the same plot (different colours)
    and ave cong and incong ontop.
    '''

    # # set up colours
    run_colour_list = ['pink', 'lightblue']
    ave_colour_list = ['red', 'blue']

    # if file name or path is passed instead of dataframe, read in dataframe
    if isinstance(per_trial_df, str):
        if os.path.isfile(per_trial_df):
            per_trial_df = pd.read_csv(per_trial_df)
        elif os.path.isfile(os.path.join(save_path, per_trial_df)):
            per_trial_df = pd.read_csv(os.path.join(save_path, per_trial_df))
    elif isinstance(per_trial_df, pd.DataFrame):
        pass
    print(f'\nper_trial_df ({per_trial_df.shape}):\n{per_trial_df}')

    isi_list = per_trial_df[isi_col_name].unique()
    if verbose:
        print(f"isi_list: {isi_list}")

    if sep_col_name == None:
        sep_list = [None]
    else:
        sep_list = per_trial_df[sep_col_name].unique()
    if verbose:
        print(f"sep_list: {sep_list}")

    n_runs = len(per_trial_df[run_col_name].unique())

    plot_idx = 'df_cols'
    # check if there is only one condition (e.g., only one isi)
    if len(sep_list) > 1 and len(isi_list) > 1:
        n_rows = len(sep_list)
        n_cols = len(isi_list)
    elif len(isi_list) > 1:
        n_rows, n_cols = get_n_rows_n_cols(len(isi_list))
        plot_idx = 'len(isi_list)'
        if verbose:
            print(f"plot indices from get_n_rows_n_cols(len(isi_list)) for {n_rows} rows and {n_cols} cols")
    elif len(sep_list) > 1:
        n_rows, n_cols = get_n_rows_n_cols(len(sep_list))
        plot_idx = 'len(sep_list)'
        if verbose:
            print(f"plot indices from get_n_rows_n_cols(len(sep_list) for {n_rows} rows and {n_cols} cols")
    else:
        n_rows = len(sep_list)
        n_cols = len(isi_list)

    # initialise plot
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols,
                             figsize=(n_cols * 6, n_rows * 6),
                             sharex=True)
    if verbose:
        print(f'\nplotting {n_cols} rows and {n_rows} cols (axes: {axes})')

    # if there is only a single condition, put axes into a list to allow indexing, consistent with
    # analyses where there are multiple conditons (e.g., axes[0])
    if n_cols == 1 and n_rows == 1:
        axes = [axes]

    # loop through isi values
    for isi_idx, isi in enumerate(isi_list):
        # get df for this isi only
        isi_df = per_trial_df[per_trial_df[isi_col_name] == isi]
        if verbose:
            print(f"\n{isi_idx}. ISI: {isi} ({isi_df.shape})"
                  # f"\n{isi_df}"
                  )

        # loop through sep values
        for sep_idx, sep in enumerate(sep_list):

            if sep == None:
                sep_df = isi_df
            else:
                # get df for this sep only
                sep_df = isi_df[isi_df[sep_col_name] == sep]
            if verbose:
                print(f"\n{sep_idx}. sep {sep} ({sep_df.shape}):"
                      f"\n{sep_df.columns}"
                      f"\n{sep_df}"
                      )

            # # check shape for accessing axes
            if plot_idx == 'df_cols':
                if len(sep_list) > 1 and len(isi_list) > 1:
                    this_ax = sep_idx, isi_idx
                else:
                    this_ax = 0
            elif plot_idx == 'len(isi_list)':  # only one sep
                this_ax = get_ax_idx(isi_idx, n_rows, n_cols)
            elif plot_idx == 'len(sep_list)':  # only one isi
                this_ax = get_ax_idx(sep_idx, n_rows, n_cols)
            print(f"this_ax: {this_ax}")

            # # get ave values for each step
            this_sep_df = sep_df[[run_col_name, 'step', hue_col_name, thr_col_name]]
            if verbose:
                print(f"this_sep_df ({this_sep_df.shape}):\n{this_sep_df}")
            # if get_median:
            if ave_type == 'median':
                ave_step_thr_df = this_sep_df.groupby([hue_col_name, 'step'], sort=False).median()
            else:
                ave_step_thr_df = this_sep_df.groupby([hue_col_name, 'step'], sort=False).mean()
            ave_step_thr_df.reset_index(drop=False, inplace=True)
            ave_step_thr_df.drop(run_col_name, axis=1, inplace=True)
            if verbose:
                print(f"ave_step_thr_df ({ave_step_thr_df.shape}):\n{ave_step_thr_df}")
            wide_ave_step_thr_df = ave_step_thr_df.pivot(index='step', columns=hue_col_name, values=thr_col_name)
            wide_ave_step_thr_df.columns = hue_names
            if verbose:
                print(f"wide_ave_step_thr_df ({wide_ave_step_thr_df.shape}):\n{wide_ave_step_thr_df}")

            stack_list = sep_df[run_col_name].unique()
            if verbose:
                print(f"stack_list: {sep_list}")

            # loop through stack values
            for stack_idx, stack in enumerate(stack_list):
                # get df for this stack only
                stack_df = sep_df[sep_df[run_col_name] == stack]
                if verbose:
                    print(f"\n{stack_idx}. stack {stack} ({stack_df.shape}):"
                          # f"\n{stack_df}"
                          )

                this_stack_df = stack_df[['step', hue_col_name, thr_col_name]]
                if verbose:
                    print(f"this_stack_df ({this_stack_df.shape}):\n{this_stack_df}")

                # I now have the data I need - reshape it so cong and incong are different columns
                wide_df = this_stack_df.pivot(index='step', columns=hue_col_name, values=thr_col_name)
                wide_df.columns = hue_names
                if verbose:
                    print(f"wide_df ({wide_df.shape}):\n{wide_df}")

                for idx, name in enumerate(hue_names):
                    axes[this_ax].errorbar(x=list(range(25)), y=wide_df[name],
                                           color=run_colour_list[idx])

            # add mean line
            for idx, name in enumerate(hue_names):
                axes[this_ax].errorbar(x=list(range(25)), y=wide_ave_step_thr_df[name],
                                       color=ave_colour_list[idx])
                # add scatter with small black dots
                axes[this_ax].scatter(x=list(range(25)), y=wide_ave_step_thr_df[name],
                                      color='black', s=10)

            # decorate each subplot
            if sep == None:
                axes[this_ax].set_title(f"{isi_col_name}{isi}")
            else:
                axes[this_ax].set_title(f"{isi_col_name}{isi}, {sep_col_name}{sep}")
            axes[this_ax].set_xlabel('step (25 per condition, per run)')
            axes[this_ax].set_ylabel(thr_col_name)

            # if y_axis crosses zero, add grey dashed line at zero (e.g., min value is less than zero and max is more than zero)
            if wide_ave_step_thr_df.min().min() < 0 and wide_ave_step_thr_df.max().max() > 0:
                axes[this_ax].axhline(y=0, linestyle="--", color='grey')

    # delete any unused axes
    if plot_idx == 'df_cols':
        if len(sep_list) > 1 and len(isi_list) > 1:
            for ax in axes.flat[len(isi_list) * len(sep_list):]:
                ax.remove()
        elif len(isi_list) > 1:
            for ax in axes.flat[len(isi_list):]:
                ax.remove()
        elif len(sep_list) > 1:
            for ax in axes.flat[len(sep_list):]:
                ax.remove()

    # artist for legend
    st0 = mlines.Line2D([], [], color=run_colour_list[0],
                        markersize=4, label=hue_names[0])
    st1 = mlines.Line2D([], [], color=run_colour_list[1],
                        markersize=4, label=hue_names[1])
    ave_line_0 = mlines.Line2D([], [], color=ave_colour_list[0],
                               marker=None, linewidth=.5,
                               label=f'{hue_names[0]} ({ave_type})')
    ave_line_1 = mlines.Line2D([], [], color=ave_colour_list[1],
                               marker=None, linewidth=.5,
                               label=f'{hue_names[1]} ({ave_type})')

    handles_list = [st0, ave_line_0, st1, ave_line_1]
    fig.legend(handles=handles_list, fontsize=16, loc='lower right')

    fig.suptitle(f"{participant_name}: {n_runs} staircases with {ave_type}", fontsize=20)

    # if get_median:
    if sep_col_name == None:
        if ave_type == 'median':
            save_name = f"all_run_stairs_{participant_name}_{isi_col_name}{isi}_median.png"
        else:
            save_name = f"all_run_stairs_{participant_name}_{isi_col_name}{isi}_mean.png"
    else:  # if there is a sep col
        # if there are multiple separation values
        if len(sep_list) > 1:
            sep = 'all'
        if ave_type == 'median':
            save_name = f"all_run_stairs_{participant_name}_{isi_col_name}{isi}_{sep_col_name[:3]}{sep}_median.png"
        else:
            save_name = f"all_run_stairs_{participant_name}_{isi_col_name}{isi}_{sep_col_name[:3]}{sep}_mean.png"

    if save_plots:
        print(f"save plot to: {os.path.join(save_path, save_name)}")
        # if save_path doesn't exist, make it
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, save_name))

    if show_plots:
        plt.show()
    plt.close()

    print(f"\n*** finished mean_staircase() ***\n")



def d_average_participant(root_path, run_dir_names_list,
                          thr_df_name='psignifit_thresholds',
                          groupby_col=None, cols_to_drop=None, cols_to_replace=None,
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
    :param groupby: Name of column(s) to average over.  Can be a string or list of strings.
    :param cols_to_drop: Name of column(s) to drop.  Can be a string or list of strings.
    :param cols_to_replace: Name of column(s) to drop for average, then add back in.  Can be a string or list of strings.
    :param error_type: Default: None. Can pass sd or se for standard deviation or error.
    :param trim_n: default None.  If int is passed, will call function trim_n_high_n_low(),
            which trims the n highest and lowest values.
    :param verbose: Default true, print progress to screen

    :returns: ave_psignifit_thr_df: (trimmed?) mean threshold for each separation and ISI.
    """

    print("\n***running d_average_participant()***")

    all_psignifit_list = []
    for run_idx, run_name in enumerate(run_dir_names_list):

        this_psignifit_df = pd.read_csv(os.path.join(root_path, run_name, f'{thr_df_name}.csv'))
        print(f'\n{run_idx}. {run_name} - this_psignifit_df:\n{this_psignifit_df}')

        if 'Unnamed: 0' in list(this_psignifit_df):
            this_psignifit_df.drop('Unnamed: 0', axis=1, inplace=True)

        if 'stair' in list(this_psignifit_df):
            this_psignifit_df.drop(columns='stair', inplace=True)

        rows, cols = this_psignifit_df.shape
        this_psignifit_df.insert(0, 'stack', [run_idx] * rows)

        if verbose:
            print(f'\nthis_psignifit_df:\n{this_psignifit_df}')

        all_psignifit_list.append(this_psignifit_df)

    # join all stacks (runs/groups) data and save as master csv
    all_data_psignifit_df = pd.concat(all_psignifit_list, ignore_index=True)
    # todo: since I added extra ISI conditions, ISI conds are not in ascending order.
    #  Perhaps re-order columns before saving?

    all_data_psignifit_df.to_csv(os.path.join(root_path, f'MASTER_{thr_df_name}.csv'), index=False)
    if verbose:
        print(f'\nall_data_psignifit_df:\n{all_data_psignifit_df.dtypes}\n{all_data_psignifit_df}')

    """Part 2: trim highest and lowest values is required and get average vals and errors"""
    # # trim highest and lowest values
    if trim_n is not None:
        if 'stair_names' in list(all_data_psignifit_df.columns):
            ref_col = 'stair_names'
        elif 'cond_type' in list(all_data_psignifit_df.columns):
            ref_col = 'cond_type'
        elif 'neg_sep' in list(all_data_psignifit_df.columns):
            ref_col = 'neg_sep'
        trimmed_df = trim_n_high_n_low(all_data_psignifit_df, trim_from_ends=trim_n,
                                       reference_col=ref_col,
                                       stack_col_id='stack',
                                       verbose=verbose)
        trimmed_df.to_csv(os.path.join(root_path, f'MASTER_TM{trim_n}_thresholds.csv'), index=False)

        get_means_df = trimmed_df
    else:
        get_means_df = all_data_psignifit_df

    print(f'\nget_means_df(head): {get_means_df.columns.to_list()}\n'
          f'{get_means_df.dtypes}\n{get_means_df}')

    # # get means and errors
    # # If I have cols to groupby and drop then use those, if not use all that long code below.
    # if cols_to_drop is not None & groupby_col is not None:
    print("\ngetting means and errors")
    print(f'cols_to_drop: {cols_to_drop}')
    print(f'groupby_col: {groupby_col}')





    if all(v is not None for v in [cols_to_drop, groupby_col]):
        print('yes running with groupby_col and cols_to_drop')


        # check all col_names in cols_to_drop are actually in the df
        for col_name in cols_to_drop:
            if col_name not in list(get_means_df.columns):
                cols_to_drop.remove(col_name)

        # check all col_names in groupby_col are actually in the df
        for col_name in groupby_col:
            if col_name not in list(get_means_df.columns):
                groupby_col.remove(col_name)

        groupby_sep_df = get_means_df.drop(cols_to_drop, axis=1)
        print(f"groupby_sep_df:\n{groupby_sep_df}")

        if cols_to_replace is not None:

            # check all col_names in cols_to_replace are actually in the df
            for col_name in cols_to_replace:
                if col_name not in list(get_means_df.columns):
                    cols_to_replace.remove(col_name)

            # first create a dictionary where the keys are the unique values from the neg_sep column and the
            # keys are the corresponding values in the 'cond_type' and 'separation' columns
            cols_to_replace_dict = {}

            if 'neg_sep' in groupby_col:
                for neg_sep_val in list(groupby_sep_df['neg_sep'].unique()):
                    # cols_to_replace_dict[neg_sep_val] = {'cond_type': groupby_sep_df[groupby_sep_df['neg_sep'] == neg_sep_val]['cond_type'].unique()[0],
                    #                                      'separation': groupby_sep_df[groupby_sep_df['neg_sep'] == neg_sep_val]['separation'].unique()[0]}
                    cols_to_replace_dict[neg_sep_val] = {}
                    for col_name in cols_to_replace:
                        cols_to_replace_dict[neg_sep_val][col_name] = groupby_sep_df[groupby_sep_df['neg_sep'] == neg_sep_val][col_name].unique()[0]
            print(f"cols_to_replace_dict: {cols_to_replace_dict}")



            # replace_cols = groupby_sep_df[[cols_to_replace]]
            # replace_cols = groupby_sep_df.pop(cols_to_replace)
            replace_cols = pd.concat([groupby_sep_df.pop(x) for x in cols_to_replace], axis=1)
            print(f"replace_cols: {cols_to_replace}\n{replace_cols}")
            # groupby_sep_df = groupby_sep_df.drop(cols_to_replace, axis=1)
            print(f"groupby_sep_df:\n{groupby_sep_df}")

        # for SImon's ricco data the dtypes were all object apart from stack.
        # Will try to convert them to numeric
        print(f'groupby_sep_df.dtypes:\n{groupby_sep_df.dtypes}\n{groupby_sep_df}')
        cols_to_ave = groupby_sep_df.columns.to_list()
        groupby_sep_df[cols_to_ave] = groupby_sep_df[cols_to_ave].apply(pd.to_numeric)
        print(f'groupby_sep_df.dtypes:\n{groupby_sep_df.dtypes}\n{groupby_sep_df}')

        ave_psignifit_thr_df = groupby_sep_df.groupby(groupby_col, sort=False,
                                                      # as_index=False
                                                      ).mean()
        ave_psignifit_thr_df.reset_index(inplace=True)

        if verbose:
            print(f'\ngroupby_sep_df:\n{groupby_sep_df}')
            print(f'\nave_psignifit_thr_df:\n{ave_psignifit_thr_df}')


        if cols_to_replace is not None:

            if 'neg_sep' in groupby_col:
                for col_name in cols_to_replace:
                    ave_psignifit_thr_df.insert(1, col_name, [cols_to_replace_dict[x][col_name] for x in ave_psignifit_thr_df['neg_sep'].to_list()])



        if verbose:
            print(f'\ngroupby_sep_df:\n{groupby_sep_df}')
            print(f'\nave_psignifit_thr_df:\n{ave_psignifit_thr_df}')

        if error_type in [False, None]:
            error_bars_df = None
        elif error_type.lower() in ['se', 'error', 'std-error', 'standard error', 'standard_error']:
            error_bars_df = groupby_sep_df.groupby(groupby_col, sort=False).sem()
        elif error_type.lower() in ['sd', 'stdev', 'std_dev', 'std.dev', 'deviation', 'standard_deviation']:
            error_bars_df = groupby_sep_df.groupby(groupby_col, sort=False).std()
        else:
            raise ValueError(f"error_type should be in:\nfor none: [False, None]\n"
                             f"for standard error: ['se', 'error', 'std-error', 'standard error', 'standard_error']\n"
                             f"for standard deviation: ['sd', 'stdev', 'std_dev', 'std.dev', "
                             f"'deviation', 'standard_deviation']")

        error_bars_df.reset_index(inplace=True)
        if verbose:
            print(f'\nerror_bars_df:\n{error_bars_df}')

        if cols_to_replace is not None:

            if 'neg_sep' in groupby_col:
                for col_name in cols_to_replace:
                    error_bars_df.insert(1, col_name, [cols_to_replace_dict[x][col_name] for x in error_bars_df['neg_sep'].to_list()])

        if verbose:
            print(f'\nerror_bars_df:\n{error_bars_df}')


    else:  # if there are no group_col by or cols_to_drop
        print('No groupby_col or cols_to_drop')

        if 'stair_names' in get_means_df.columns:
            groupby_sep_df = get_means_df.drop('stack', axis=1)
            if 'congruent' in groupby_sep_df.columns:
                groupby_sep_df = groupby_sep_df.drop('congruent', axis=1)


            else:
                groupby_sep_df = groupby_sep_df.drop('separation', axis=1)
                if 'cond' in groupby_sep_df.columns:
                    groupby_sep_df = groupby_sep_df.drop('cond', axis=1)
            ave_psignifit_thr_df = groupby_sep_df.groupby('stair_names', sort=False).mean()

            if verbose:
                print(f'\nave_psignifit_thr_df:\n{ave_psignifit_thr_df}')
                print(f'\ngroupby_sep_df:\n{groupby_sep_df}')



            # groupby_cols is a list of column names which do not include columns containing the substring  'ISI_'
            groupby_cols = [x for x in groupby_sep_df.columns if 'ISI_' not in x]


            if error_type in [False, None]:
                error_bars_df = None
            elif error_type.lower() in ['se', 'error', 'std-error', 'standard error', 'standard_error']:
                # error_bars_df = groupby_sep_df.groupby('stair_names', sort=False).sem()
                error_bars_df = groupby_sep_df.groupby(groupby_cols, sort=False).sem()

            elif error_type.lower() in ['sd', 'stdev', 'std_dev', 'std.dev', 'deviation', 'standard_deviation']:
                error_bars_df = groupby_sep_df.groupby(groupby_cols, sort=False).std()
            else:
                raise ValueError(f"error_type should be in:\nfor none: [False, None]\n"
                                 f"for standard error: ['se', 'error', 'std-error', 'standard error', 'standard_error']\n"
                                 f"for standard deviation: ['sd', 'stdev', 'std_dev', 'std.dev', "
                                 f"'deviation', 'standard_deviation']")
            print(f'\nerror_bars_df:\n{error_bars_df}')

        else:
            # todo: do I still need this?
            # for Exp2_Bloch_NM_v2
            if thr_df_name == 'long_thr_df':
                groupby_sep_df = get_means_df.drop('stack', axis=1)
                # ave_psignifit_thr_df = groupby_sep_df.groupby(['cond_type', 'dur_ms', 'ISI'], sort=False).mean()
                ave_psignifit_thr_df = groupby_sep_df.groupby(['cond_type', 'stair_name', 'isi_fr', 'dur_ms'], sort=False).mean()

                if verbose:
                    print(f'\nave_psignifit_thr_df:\n{ave_psignifit_thr_df}')
                if error_type in [False, None]:
                    error_bars_df = None
                elif error_type.lower() in ['se', 'error', 'std-error', 'standard error', 'standard_error']:
                    # error_bars_df = groupby_sep_df.groupby(['cond_type', 'dur_ms', 'ISI'], sort=False).sem()
                    error_bars_df = groupby_sep_df.groupby(['cond_type', 'stair_name', 'isi_fr', 'dur_ms'], sort=False).sem()
                elif error_type.lower() in ['sd', 'stdev', 'std_dev', 'std.dev', 'deviation', 'standard_deviation']:
                    # error_bars_df = groupby_sep_df.groupby(['cond_type', 'dur_ms', 'ISI'], sort=False).std()
                    error_bars_df = groupby_sep_df.groupby(['cond_type', 'stair_name', 'isi_fr', 'dur_ms'], sort=False).std()
                else:
                    raise ValueError(f"error_type should be in:\nfor none: [False, None]\n"
                                     f"for standard error: ['se', 'error', 'std-error', 'standard error', 'standard_error']\n"
                                     f"for standard deviation: ['sd', 'stdev', 'std_dev', 'std.dev', "
                                     f"'deviation', 'standard_deviation']")
                print(f'\nerror_bars_df: ({error_type})\n{error_bars_df}')

            elif 'probeSpeed' in get_means_df.columns.to_list():
                groupby_sep_df = get_means_df.drop('stack', axis=1)
                ave_psignifit_thr_df = groupby_sep_df.groupby(['probeSpeed'], sort=True).mean()
                if verbose:
                    print(f'\nave_psignifit_thr_df:\n{ave_psignifit_thr_df}')

                if error_type in [False, None]:
                    error_bars_df = None
                elif error_type.lower() in ['se', 'error', 'std-error', 'standard error', 'standard_error']:
                    error_bars_df = groupby_sep_df.groupby(['probeSpeed'], sort=True).sem()
                elif error_type.lower() in ['sd', 'stdev', 'std_dev', 'std.dev', 'deviation', 'standard_deviation']:
                    error_bars_df = groupby_sep_df.groupby(['probeSpeed'], sort=True).std()
                else:
                    raise ValueError(f"error_type should be in:\nfor none: [False, None]\n"
                                     f"for standard error: ['se', 'error', 'std-error', 'standard error', 'standard_error']\n"
                                     f"for standard deviation: ['sd', 'stdev', 'std_dev', 'std.dev', "
                                     f"'deviation', 'standard_deviation']")
                print(f'\nerror_bars_df: ({error_type})\n{error_bars_df}')

    ave_psignifit_thr_df.reset_index(inplace=True)
    error_bars_df.reset_index(inplace=True)

    # save csv with average values
    # todo: since I added extra ISI conditions, ISI conds are not in ascending order.
    #  Perhaps re-order columns before saving?
    if trim_n is not None:
        ave_psignifit_thr_df.to_csv(os.path.join(root_path, f'MASTER_ave_TM{trim_n}_thresh.csv'), index=False)
        error_bars_df.to_csv(os.path.join(root_path, f'MASTER_ave_TM{trim_n}_thr_error_{error_type}.csv'), index=False)
    else:
        ave_psignifit_thr_df.to_csv(os.path.join(root_path, 'MASTER_ave_thresh.csv'), index=False)
        error_bars_df.to_csv(os.path.join(root_path, f'MASTER_ave_thr_error_{error_type}.csv'), index=False)
    print("\n*** finished d_average_participant()***\n")

    return ave_psignifit_thr_df, error_bars_df





def e_average_exp_data_Dec23(exp_path, p_names_list,
                             error_type='SE',
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
    print("\n***running e_average_exp_data_Dec23()***\n")

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
    # all_exp_thr_df.to_csv(f'{exp_path}{os.sep}MASTER_exp_all_thr.csv', index=False)
    all_exp_thr_df.to_csv(os.path.join(exp_path, 'MASTER_exp_all_thr.csv'), index=False)

    # # get means and errors
    get_means_df = all_exp_thr_df.drop('participant', axis=1)

    groupby_col = 'neg_sep'
    # exp_ave_thr_df = get_means_df.groupby('stair', sort=True).mean()
    exp_ave_thr_df = get_means_df.groupby(groupby_col, sort=True).mean()
    if verbose:
        print(f'\nexp_ave_thr_df:\n{exp_ave_thr_df}')

    if error_type in [False, None]:
        error_bars_df = None
    elif error_type.lower() in ['se', 'error', 'std-error', 'standard error', 'standard_error']:
        error_bars_df = get_means_df.groupby(groupby_col, sort=True).sem()
    elif error_type.lower() in ['sd', 'stdev', 'std_dev', 'std.dev', 'deviation', 'standard_deviation']:
        error_bars_df = get_means_df.groupby(groupby_col, sort=True).std()
    else:
        raise ValueError(f"error_type should be in:\nfor none: [False, None]\n"
                         f"for standard error: ['se', 'error', 'std-error', 'standard error', 'standrad_error']\n"
                         f"for standard deviation: ['sd', 'stdev', 'std_dev', 'std.dev', "
                         f"'deviation', 'standard_deviation']")
    # repace any NaNs with zero in error_bars_df
    error_bars_df.fillna(0, inplace=True)
    if verbose:
        print(f'\nerror_bars_df: ({error_type})\n{error_bars_df}')

    # save csv with average values
    exp_ave_thr_df.to_csv(f'{exp_path}{os.sep}MASTER_exp_ave_thr.csv')
    error_bars_df.to_csv(f'{exp_path}{os.sep}MASTER_ave_thr_error_{error_type}.csv')

    print("\n*** finished e_average_exp_data_Dec23()***\n")

    return exp_ave_thr_df, error_bars_df


def make_plots_Dec23(all_df_path, root_path, participant_name, n_trimmed,
                     thr_col_name='probeLum',
                     x_col_name='isi_ms',
                     hue_col_name='congruent', hue_val_order=[-1, 1],
                     hue_labels=['Incongruent', 'Congruent'],
                     motion_col='bg_motion_ms',
                     x_label='ISI (ms)', y_label='Probe Luminance',
                     extra_text=None,
                     exp_ave=False):
    """
    Make plots showing ISI or cong difference (incong-cong) for all ISI.

    :param all_df_path: dataframe or path to dataframe with datum column,
        e.g. if analysing participant data, it should have multiple runs, not means;
        if analysing experiment data, it should have multiple participants, not just means.
    :param root_path: path to save to
    :param participant_name: name of participant (or 'exp_means)
    :param n_trimmed: none, or number of vals trimmed from each end
    :return:   plots
    """

    print("\n\n*** running make_flow_parse_plots() ***")

    if isinstance(all_df_path, pd.DataFrame):
        all_df = all_df_path
    else:
        all_df = pd.read_csv(all_df_path)

    if exp_ave:
        datum_col = 'participant'
        n_to_ave_over = len(all_df[datum_col].unique().tolist())
    else:
        datum_col = 'stack'
        n_to_ave_over = len(all_df[datum_col].unique().tolist())

    # rename columns with long float names - if col name contains '.', only have two characters after it
    all_df.columns = [i[:i.find('.') + 3] if '.' in i else i for i in all_df.columns.tolist()]
    print(f"all_df columns: {list(all_df.columns)}")

    # # # make long df # # #
    # make long df with all probe_durs in one column (threshold) with probe_dur_ms as another column
    print(f"all_df: {all_df.columns.tolist()}\n: {all_df}\n")

    strip_this = f'{x_col_name}_'

    n_cols_to_change = 0
    for col in all_df.columns.tolist():
        if strip_this in col:
            n_cols_to_change += 1

    cols_to_keep = all_df.columns.tolist()[:-n_cols_to_change]
    cols_to_change = all_df.columns.tolist()[-n_cols_to_change:]


    print(f"strip _this: {strip_this}")
    print(f"cols_to_keep: {cols_to_keep}")
    print(f"cols_to_change: {cols_to_change}")
    all_long_df = make_long_df(wide_df=all_df,
                               cols_to_keep=cols_to_keep,
                               cols_to_change=cols_to_change,
                               cols_to_change_show=thr_col_name,
                               new_col_name=x_col_name,
                               strip_from_cols=strip_this, verbose=True)

    print(f"all_long_df: {all_long_df.columns.tolist()}\n: {all_long_df}\n")

    # simple_all_long_df is the same as all_long_df but without the columns 'stair' and 'flow_dir'
    # make a copy as I'll use all_long_df later
    simple_all_long_df = all_long_df.copy()
    # simple_all_long_df = simple_all_long_df.drop(columns=['stair', 'flow_dir'])
    if 'stair' in list(simple_all_long_df.columns):
        simple_all_long_df.drop(columns=['stair'], inplace=True)
    if 'flow_dir' in list(simple_all_long_df.columns):
        simple_all_long_df.drop(columns=['flow_dir'], inplace=True)

    # sort simple_all_long_df by probe_dur_ms and prelim
    simple_all_long_df.sort_values(by=[x_col_name, motion_col], inplace=True)


    print(f"simple_all_long_df: {simple_all_long_df.columns.tolist()}:\n{simple_all_long_df}\n")

    # # MAKE PLOTS # # #

    # lineplot with err bars and scatter, same as joined plot, but not joined
    fig, ax = plt.subplots()

    # lineplot with errorbars
    sns.pointplot(data=simple_all_long_df,
                  x=x_col_name, y=thr_col_name,
                  hue=hue_col_name, hue_order=hue_val_order,
                  palette=sns.color_palette("tab10", n_colors=2),
                  dodge=.1,  # float allowed for dodge here
                  errorbar='se', capsize=.1, errwidth=2,
                  ax=ax)

    # # add scatter showing each data point
    sns.stripplot(data=simple_all_long_df,
                  x=x_col_name, y=thr_col_name,
                  hue=hue_col_name, hue_order=hue_val_order,
                  palette=sns.color_palette("tab10", n_colors=2),
                  size=3, dodge=True, ax=ax)

    # if y-axis crosses zero, add horizonal line at zero
    if min(simple_all_long_df[thr_col_name]) < 0 < max(simple_all_long_df[thr_col_name]):
        ax.axhline(y=0.0, color='grey', linestyle='--', linewidth=.7)
        # title should describe what positive and negative means
        if extra_text is not None:
            plt.title(f"{extra_text}\n-ive = outward, +ive = inward")
        else:
            plt.title("-ive = outward, +ive = inward")
    else:
        if extra_text is not None:
            plt.title(extra_text)

    # decorate plots: main title, axis labels and legend
    plt.suptitle(f"{participant_name} mean thresholds with SE, n={n_to_ave_over}, TM={n_trimmed}")


    # update axis labels
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    print(f"idiot check, x_label: {x_label}")
    if 'ISI' in x_label.upper():
        # get x tick labels, if '-1.0' or '-1' is first label, change to 'Conc'
        x_tick_labels = ax.get_xticklabels()
        if x_tick_labels[0].get_text() in ['-1', '-1.0']:
            x_tick_labels[0].set_text('Concurrent')
        ax.set_xticklabels(x_tick_labels)

    # plot legend but only with two handles/labels - e.g., Expanding (for exp) and Contracting (cont)
    handles, labels = ax.get_legend_handles_labels()
    labels = [hue_labels[0] if i == str(hue_val_order[0]) else hue_labels[1] for i in labels]
    ax.legend(handles[0:2], labels[0:2], loc='best', title='flow direction')

    # save figure
    if extra_text is not None:
        if n_trimmed:
            plt.savefig(os.path.join(root_path, f"{participant_name}_TM{n_trimmed}_{extra_text}_not_joined.png"))
        else:
            plt.savefig(os.path.join(root_path, f"{participant_name}_{extra_text}_not_joined.png"))
    else:
        if n_trimmed:
            plt.savefig(os.path.join(root_path, f"{participant_name}_TM{n_trimmed}_not_joined.png"))
        else:
            plt.savefig(os.path.join(root_path, f"{participant_name}_not_joined.png"))

    plt.show()

    # make diff_df: for each datum, bg_motion and probe_dur_ms, get the difference between exp and cont
    diff_list = []
    for x_val in all_long_df[x_col_name].unique().tolist():
        x_val_df = all_long_df[all_long_df[x_col_name] == x_val]
        for bg_motion in x_val_df[motion_col].unique().tolist():
            bg_motion_df = x_val_df[x_val_df[motion_col] == bg_motion]
            for datum in bg_motion_df[datum_col].unique().tolist():
                datum_df = bg_motion_df[bg_motion_df[datum_col] == datum]

                print(f"\ndatum: {datum}, x_val: {x_val}, bg_motion: {bg_motion}\n"
                      f"datum_df: {datum_df.columns.tolist()}:\n"
                      # f"{datum_df}\n"
                      f"")
                hue0_val = datum_df[datum_df[hue_col_name] == hue_val_order[0]][thr_col_name].values[0]
                hue1_val = datum_df[datum_df[hue_col_name] == hue_val_order[1]][thr_col_name].values[0]

                '''hue0_val and hue1_val can be positive or negative, so always subtracting could be wrong.
                If either are negative, I will add a constant to both, such that neither are negative.'''
                if min(hue0_val, hue1_val) < 0:
                    min_value = min(hue0_val, hue1_val)
                    hue0_val += abs(min_value)
                    hue1_val += abs(min_value)
                diff = hue1_val - hue0_val
                print(f"hue0_val: {hue0_val}\nhue1_val: {hue1_val}\ndiff: {diff}")

                # diff_list.append([probe_dur_ms, bg_motion, datum, exp_speed, cont_speed, diff])
                diff_list.append([x_val, bg_motion, datum, hue0_val, hue1_val, diff])

    print(f"diff_list: {diff_list}\n")

    # make diff_df from diff_list
    diff_df = pd.DataFrame(diff_list,
                           columns=[x_col_name, motion_col, datum_col, hue0_val, hue1_val, 'diff'])
    print(f"diff_df: {diff_df.columns.tolist()}\n{diff_df}\n")

    # drop exp_speed and cont_speed columns
    diff_df = diff_df.drop(columns=[hue0_val, hue1_val])

    # if there are multiple probe_dur_ms, make a lineplot with probe_dur_ms on x-axis, diff on y-axis, bg_motion as hue
    if len(diff_df[x_col_name].unique().tolist()) > 1:
        fig, ax = plt.subplots()

        # individual datapoints
        sns.stripplot(data=diff_df, x=x_col_name, y='diff',
                      dodge=True, color='grey', alpha=.7,
                      ax=ax)

        # lineplot with errorbars
        sns.pointplot(data=diff_df, x=x_col_name, y='diff',
                      errorbar='se', capsize=.1, errwidth=2,
                      ax=ax)


        plt.axhline(y=0.0, color='grey', linestyle='--')

        # plt.suptitle(f"probe_speed difference (flow exp - cont)")
        plt.suptitle(f"{y_label} difference ({hue_labels[1]} - {hue_labels[0]})")

        # update axis labels
        plt.ylabel(f"Difference ({hue_labels[1]} - {hue_labels[0]})")
        plt.xlabel(x_label)
        if 'ISI' in x_label.upper():
            # get x tick labels, if '-1.0' or '-1' is first label, change to 'Conc'
            x_tick_labels = ax.get_xticklabels()
            if x_tick_labels[0].get_text() in ['-1', '-1.0', -1, -1.0]:
                x_tick_labels[0].set_text('Concurrent')
            else:
                print(f"-1 not in x_tick_labels: {x_tick_labels}")
            ax.set_xticklabels(x_tick_labels)

        else:
            print(f"'ISI not in x-axis label: {x_label}")

        if extra_text is not None:
            plt.title(f"{participant_name}, {extra_text}, n={n_to_ave_over},  TM={n_trimmed}\n"
                      f"-ive = {hue_labels[0]} greater, +ive = {hue_labels[1]} greater")
        else:
            plt.title(f"{participant_name}, n={n_to_ave_over},  TM={n_trimmed}\n"
                      f"-ive = {hue_labels[0]} greater, +ive = {hue_labels[1]} greater")
        plt.subplots_adjust(top=0.85)  # add space below suptitle

        if extra_text is not None:
            if n_trimmed:
                plt.savefig(os.path.join(root_path, f"{participant_name}_TM{n_trimmed}_{extra_text}_diff_line.png"))
            else:
                plt.savefig(os.path.join(root_path, f"{participant_name}_{extra_text}_diff_line.png"))
        else:
            if n_trimmed:
                plt.savefig(os.path.join(root_path, f"{participant_name}_TM{n_trimmed}_diff_line.png"))
            else:
                plt.savefig(os.path.join(root_path, f"{participant_name}_diff_line.png"))
        plt.show()

    print("\n*** finished make_flow_parse_plots()***\n")