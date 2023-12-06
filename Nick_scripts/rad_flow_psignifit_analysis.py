import os
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import math

from exp1a_psignifit_analysis import plt_heatmap_row_col

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


def return_target_or_above_below(array_of_values, target_value):
    """
    Returns the target value if it is in the array, otherwise,
    returns the array values above and below the target (to use for interpolation).

    array_of_values: An array of values to search.
    target_value: The target value.
        """

    if type(array_of_values) is list:
        array_of_values = np.array(array_of_values)

    '''
    ValueError: zero-size array to reduction operation minimum which has no identity
    This error occurs when the array is empty, 
    e.g., the target value is less than the minimum value, or more than the maximum value in the array.
    Because this is technically possible for RGB255 values of 0 or 255 (which differ between uncalibraed and asus_cal, 
    the script will correct this if it occurs and return the minimum or maximum value in the array.
    '''

    # Check if the target value is in the array.
    if target_value in array_of_values:
        # return target_value
        # rather than returning the target, I return the array value that matches the target,
        # this might deal with any issues arrising from int/float comparisons.
        matching_array_val = array_of_values[array_of_values == target_value]
        if len(set(matching_array_val)) == 1:
            matching_array_val = matching_array_val[0]
        else:
            raise ValueError(
                f"More than one matching value found in array_of_values: {array_of_values} for target_value: {target_value}")

        return matching_array_val

    else:

        # it is technically possible for the value to be outside the range of the array,
        # as the min and max lum for uncalibrated are .13 and 151.05, and for calibrated are .14 and 150.63.
        # in these cases, return the minimum or maximum value in the array.
        if (.13 <= target_value < .14) and np.amin(array_of_values) == .14:
            return .14
        elif (150.63 < target_value <= 151.05) and (np.amax(array_of_values) == 150.63):
            return 150.63
        else:

            # if target not in array, return the value below and above.
            item_below = np.amax(array_of_values[array_of_values < target_value])
            item_above = np.amin(array_of_values[array_of_values > target_value])

        # Return a tuple of the array values above and below the target.
        return item_below, item_above


def rad_flow_mon_conversion(uncalibrated_rgb255_val, verbose=False):
    """
    this is a function that uses a nested-dictionary containing two sets of luminance values
    (one for each of two monitor settings) at each of 18 evenly spaced rgb255 values.
    The function will take an rgb255 value from the uncalibrated monitor and find the corresponding luminance value.
    It will then use that luminance value to find the corresponding rgb255 value for the calibrated monitor (asus_cal).

    :param uncalibrated_rgb255_val: The rgb255 value from the uncalibrated monitor.

    :return: The equivalent rgb255 value for the calibrated monitor (asus_cal).
    """

    if verbose:
        print("\n*** running rad_flow_mon_conversion() ***")
        print(f'uncalibrated_rgb255_val: {uncalibrated_rgb255_val}')

    spyder_values_dicts = {'uncalibrated': {0: 0.13, 15: 0.46, 30: 1.52, 45: 3.59, 60: 6.69, 75: 10.91, 90: 16.18,
                                            105: 22.66, 120: 30.29, 135: 39.22, 150: 49.27, 165: 60.44, 180: 72.73,
                                            195: 86.43, 210: 101.26, 225: 117.09, 240: 134.09, 255: 151.05},
                           'asus_cal': {0: 0.14, 15: 8.05, 30: 16.66, 45: 25.58, 60: 34.6, 75: 43.44, 90: 52.51,
                                        105: 61.69, 120: 70.94, 135: 79.81, 150: 89.37, 165: 98.38, 180: 107.48,
                                        195: 116.36, 210: 125.33, 225: 134.43, 240: 142.94, 255: 150.63}
                           }

    array_of_rgb255_vals = np.array(list(spyder_values_dicts['uncalibrated'].keys()))
    if verbose:
        print(f'array_of_rgb255_vals: {array_of_rgb255_vals}')

    '''
    Part one, get the uncalibrated luminance from the uncalibrated RGB255 value
    '''

    # check if the target value is in the array, otherwise, return the array values above and below the target.
    vals_for_uncali_interp = return_target_or_above_below(array_of_rgb255_vals, uncalibrated_rgb255_val)
    if verbose:
        print(f"vals_for_uncali_interp: {vals_for_uncali_interp}")

    # if (isinstance(vals_for_uncali_interp, int)) or (isinstance(vals_for_uncali_interp, float)):
    #     if verbose:
    #         print(f"vals_for_uncali_interp is an int, so use {vals_for_uncali_interp} for conversion")
    #     # if the uncalibrated_rgb255_val is in the array, use that for conversion
    #     uncalibrated_lum_val = spyder_values_dicts['uncalibrated'][vals_for_uncali_interp]
    #
    # # elif len(vals_for_uncali_interp) == 1:
    # else:
    #     # interpolate between the array values below and above the uncalibrated_rgb255_val to find the
    #     # luminance value for the uncalibrated monitor that corresponds to the uncalibrated rgb255 value
    #
    #     if verbose:
    #         print(f"use {uncalibrated_rgb255_val} to interpolate between {vals_for_uncali_interp} to get corresponding "
    #               f"interpolation between {[spyder_values_dicts['uncalibrated'][vals_for_uncali_interp[0]], spyder_values_dicts['uncalibrated'][vals_for_uncali_interp[1]]]}")
    #     uncalibrated_lum_val = np.interp(uncalibrated_rgb255_val, vals_for_uncali_interp,
    #                                      [spyder_values_dicts['uncalibrated'][vals_for_uncali_interp[0]],
    #                                       spyder_values_dicts['uncalibrated'][vals_for_uncali_interp[1]]])

    if isinstance(vals_for_uncali_interp, tuple):
        # interpolate between the array values below and above the uncalibrated_rgb255_val to find the
        # luminance value for the uncalibrated monitor that corresponds to the uncalibrated rgb255 value
        if verbose:
            print(f"use {uncalibrated_rgb255_val} to interpolate between {vals_for_uncali_interp} to get corresponding "
                  f"interpolation between {[spyder_values_dicts['uncalibrated'][vals_for_uncali_interp[0]], spyder_values_dicts['uncalibrated'][vals_for_uncali_interp[1]]]}")
        uncalibrated_lum_val = np.interp(uncalibrated_rgb255_val, vals_for_uncali_interp,
                                         [spyder_values_dicts['uncalibrated'][vals_for_uncali_interp[0]],
                                          spyder_values_dicts['uncalibrated'][vals_for_uncali_interp[1]]])

    else:
        if verbose:
            print(f"vals_for_uncali_interp is an {type(vals_for_uncali_interp)}, so use {vals_for_uncali_interp} for conversion")
        # if the uncalibrated_rgb255_val is in the array, use that for conversion
        uncalibrated_lum_val = spyder_values_dicts['uncalibrated'][vals_for_uncali_interp]



    if verbose:
        print(f'uncalibrated_lum_val: {uncalibrated_lum_val}')

    '''
    Part 2: get the calibrated rgb255 value from the uncalibrated luminance value'''

    # now switch, such that the array is the calibrated luminances and we want to find the corresponding rgb255 value
    array_of_cal_lum_vals = np.array(list(spyder_values_dicts['asus_cal'].values()))
    if verbose:
        print(f'\narray_of_cal_lum_vals: {array_of_cal_lum_vals}')

    vals_for_asus_cal_interp = return_target_or_above_below(array_of_cal_lum_vals, uncalibrated_lum_val)
    if verbose:
        print(f"vals_for_asus_cal_interp: {vals_for_asus_cal_interp}")

    # if isinstance(vals_for_asus_cal_interp, int):
    #     # if the uncalibrated_rgb255_val is in the array, use that for conversion
    #     calibrated_rgb255_val = list(spyder_values_dicts['asus_cal'].keys())[
    #         list(spyder_values_dicts['asus_cal'].values()).index(vals_for_asus_cal_interp)][0]
    # else:
    #     # interpolate between the array values below and above the uncalibrated_rgb255_val to find the
    #     # luminance value for the uncalibrated monitor that corresponds to the uncalibrated rgb255 value
    #
    #     asus_cal_key_below = list(spyder_values_dicts['asus_cal'].keys())[
    #         list(spyder_values_dicts['asus_cal'].values()).index(vals_for_asus_cal_interp[0])]
    #     asus_cal_key_above = list(spyder_values_dicts['asus_cal'].keys())[
    #         list(spyder_values_dicts['asus_cal'].values()).index(vals_for_asus_cal_interp[1])]
    #
    #     if verbose:
    #         print(f"use {uncalibrated_lum_val} to interpolate between {vals_for_asus_cal_interp} to get corresponding "
    #               f"interpolation between {asus_cal_key_below, asus_cal_key_above}")
    #
    #     calibrated_rgb255_val = int(
    #         np.interp(uncalibrated_lum_val, vals_for_asus_cal_interp, [asus_cal_key_below, asus_cal_key_above]))
    #     if verbose:
    #         print(f'calibrated_rgb255_val: {calibrated_rgb255_val}\n')

    if isinstance(vals_for_asus_cal_interp, tuple):
        # interpolate between the array values below and above the uncalibrated_rgb255_val to find the
        # luminance value for the uncalibrated monitor that corresponds to the uncalibrated rgb255 value
        asus_cal_key_below = list(spyder_values_dicts['asus_cal'].keys())[
            list(spyder_values_dicts['asus_cal'].values()).index(vals_for_asus_cal_interp[0])]
        asus_cal_key_above = list(spyder_values_dicts['asus_cal'].keys())[
            list(spyder_values_dicts['asus_cal'].values()).index(vals_for_asus_cal_interp[1])]
        if verbose:
            print(f"use {uncalibrated_lum_val} to interpolate between {vals_for_asus_cal_interp} to get corresponding "
                  f"interpolation between {asus_cal_key_below, asus_cal_key_above}")
        calibrated_rgb255_val = int(
            np.interp(uncalibrated_lum_val, vals_for_asus_cal_interp, [asus_cal_key_below, asus_cal_key_above]))
        if verbose:
            print(f'calibrated_rgb255_val: {calibrated_rgb255_val}\n')

    else:
        # if the uncalibrated_rgb255_val is in the array, use that for conversion
        calibrated_rgb255_val = list(spyder_values_dicts['asus_cal'].keys())[
            list(spyder_values_dicts['asus_cal'].values()).index(vals_for_asus_cal_interp)]
        if isinstance(calibrated_rgb255_val, tuple):
            if len(set(calibrated_rgb255_val)) == 1:
                calibrated_rgb255_val = calibrated_rgb255_val[0]
            else:
                raise ValueError(f"calibrated_rgb255_val: {calibrated_rgb255_val} is not a single value")

    if verbose:
        print(f"calibrated_rgb255_val: {calibrated_rgb255_val}")
        print("*** finished get_calibrated_rgb255_val() ***\n")

    return calibrated_rgb255_val



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


def get_sorted_neg_sep_indices(neg_sep_to_sort):
    """
    Function to take neg_sep_vals_list in order of stairs and return indices to
    sort neg_sep_vals_list in order [18, -18, 6, -6, 3, -3, 2, -2, 1, -1, 0, -.01].
    This is sorted(abs(list_to_sort), reverse=True) APART from the last two values.

    :param neg_sep_to_sort: list of negative separations to sort
    :return: indices used to sort into correct order
    """

    # first sort list into order sorted(abs(neg_sep_to_sort), reverse=True)
    neg_sep_to_sort_sorted = sorted(neg_sep_to_sort, key=abs, reverse=True)

    # then swap last two elements so that 0.0 is before -0.1
    neg_sep_to_sort_sorted[-1], neg_sep_to_sort_sorted[-2] = neg_sep_to_sort_sorted[-2], neg_sep_to_sort_sorted[-1]
    print(f"neg_sep_to_sort_sorted: {neg_sep_to_sort_sorted}")

    # get indices to sort neg_sep_to_sort into neg_sep_to_sort_sorted
    sorted_neg_sep_indices = [neg_sep_to_sort.index(x) for x in neg_sep_to_sort_sorted]
    print(f"sorted_neg_sep_indices: {sorted_neg_sep_indices}")

    return sorted_neg_sep_indices


def sort_with_neg_sep_indices(list_to_sort, sorted_neg_sep_indices):
    """
    Function to sort list_to_sort using sorted_neg_sep_indices.

    :param list_to_sort: list that needs sorting
    :param sorted_neg_sep_indices: indices used to sort into correct order
    :return: sorted_list
    """

    # check list_to_sort and sorted_neg_sep_indices are same length, if not, raise error
    if len(list_to_sort) != len(sorted_neg_sep_indices):
        raise ValueError("list_to_sort and sorted_neg_sep_indices are not the same length")

    # sort list_to_sort into order of sorted_neg_sep_indices
    sorted_list = [list_to_sort[i] for i in sorted_neg_sep_indices]
    print(f"sorted_list: {sorted_list}")

    return sorted_list




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

    # if new_col_name == 'ISI':
    #      long_df["ISI"] = long_df["ISI"].astype(int)

    if verbose:
        print(f'long_df:\n{long_df}')

    print("\n*** finished make_long_df() ***\n")

    return long_df


def neg_sep_to_sep_w_cond_type(orig_df, pos_neg_labels=['Congruent', 'Incongruent'],
                               neg_sep_col='neg_sep',
                               cols_to_drop=['stair_names', 'neg_sep']):
    """
    Function to convert a dataframe with a column of negative separations to a dataframe with a column cond_type.
    :param orig_df: dataframe with no cond_type col but with neg_sep_col or stair_names col showing neg_sep.
    :param pos_neg_labels: labels for positive and negative conditions (in that order)
    :param neg_sep_col: column to evaluate for negative separations
    :param cols_to_drop: columns to drop (will attempt to drop them even if missing without failing)

    :return: new_df with cond_type column
    """

    print('\n*** running neg_sep_to_sep_w_cond_type() ***')

    # make a copy of the original df so the original is unchanged
    edit_this_df = orig_df.copy()

    # see if 'cond_type' already exists, if not add it using pos_neg_labels
    if 'cond_type' not in edit_this_df.columns:
        if pos_neg_labels is not None:
            cond_type_vals = [pos_neg_labels[1] if i < 0 else pos_neg_labels[0] for i in
                              edit_this_df[neg_sep_col]]
            edit_this_df.insert(loc=0, column='cond_type', value=cond_type_vals)
        else:
            raise ValueError(f'cond_type not in edit_this_df.columns and pos_neg_labels is None')

    if 'separation' not in edit_this_df.columns:
        sep_vals = [0 if i == .01 else abs(i) for i in edit_this_df[neg_sep_col]]
        edit_this_df.insert(loc=1, column='separation', value=sep_vals)

    # drop any columns in cols_to_drop
    for i in cols_to_drop:
        if i in edit_this_df.columns:
            edit_this_df.drop(columns=i, inplace=True)

    # sort by separation so that the new columns are in the right order
    new_df = edit_this_df.sort_values(by='separation')

    # Convert the values in the separation column to ints to give shorted column names
    new_df['separation'] = new_df['separation'].astype(int)

    print('\n*** finished neg_sep_to_sep_w_cond_type() ***')

    return new_df


def transpose_df_w_cond_type(orig_df,
                             cols_to_rows=['ISI_4', 'ISI_6', 'ISI_9'],
                             add_pos_neg_labels=['Congruent', 'Incongruent'],
                             cols_to_drop=['stair_names', 'neg_sep'],
                             cond_type_col='cond_type',
                             rows_to_cols='separation',
                             verbose=True
                             ):
    """
    Function to transpose a dataframe with a column that is to be kept as a column,
    and the rest of the columns are to be transposed into rows.
    e.g.,  make 'separation' values into columns ['sep_0', 'sep_2'...], and 'ISI' columns from cols_to_rows into rows

    :param orig_df: dataframe to transpose
    :param cols_to_rows: column names to transpose into rows (e.g., ['ISI_-1', 'ISI_0'...])
    :param add_pos_neg_labels: labels to add to cond_type_col if it doesn't exist
    :param cols_to_drop: columns to drop from orig_df (e.g., ['stair_names', 'neg_sep'])
    :param cond_type_col: column names to keep as a columns, will make this column if it doesn't exist.
    :param rows_to_cols: row names to transpose into columns (e.g., 'separation').  Not currently using this

    :return: transposed dataframe
    """
    print("\n*** running transpose_df_w_cond_type() ***")

    # make a copy of the original df so the original is unchanged
    edit_this_df = orig_df.copy()

    if verbose:
        print(f'edit_this_df: \n{edit_this_df}')

    # see if cond_type_col already exists, if not add it using add_pos_neg_labels
    if cond_type_col not in edit_this_df.columns:
        if add_pos_neg_labels is not None:
            if 'neg_sep' in edit_this_df.columns:
                cond_type_vals = [add_pos_neg_labels[1] if i < 0 else add_pos_neg_labels[0] for i in
                                  edit_this_df['neg_sep']]
            else:
                raise ValueError(f'neg_sep not in edit_this_df.columns and add_pos_neg_labels is not None')
            edit_this_df.insert(loc=0, column=cond_type_col, value=cond_type_vals)
        else:
            raise ValueError('There are no add_pos_neg_labels to use to construct cond_type column')


    # drop any columns in cols_to_drop
    for i in cols_to_drop:
        if i in edit_this_df.columns:
            edit_this_df.drop(columns=i, inplace=True)

    # sort by separation so that the new columns are in the right order
    edit_this_df.sort_values(by='separation', inplace=True)

    # Convert the values in the separation column to ints to give shorted column names
    edit_this_df['separation'] = edit_this_df['separation'].astype(int)

    # Melt the DataFrame into long form with columns ['cond_type', 'separation', 'ISI', 'value']
    long_df = edit_this_df.melt(id_vars=[cond_type_col, 'separation'], value_vars=cols_to_rows, var_name='ISI',
                           value_name='value')

    # make a list of the unique values in the sep column prefixed with 'sep_'
    sep_names_list = ['sep_' + str(i) for i in long_df['separation'].unique()]

    # use pivot_table to reshape the df so that the ['sep', and 'value'] columns are reshaped into individual 'sep_' columns containing the corresponding values
    transposed_df = long_df.pivot_table(index=[cond_type_col, 'ISI'], columns='separation', values='value')

    # rename the columns which match values in long_df['sep'].unique() with the values in sep_names_list
    transposed_df.columns = sep_names_list

    # reset the index so that the cond_type and ISI columns are no longer the index
    transposed_df.reset_index(inplace=True, drop=False)

    # strip 'ISI_' from the ISI column values
    transposed_df['ISI'] = transposed_df['ISI'].str.strip('ISI_')

    # sort values by cond_type and ISI
    transposed_df.sort_values(by=[cond_type_col, 'ISI'], inplace=True)

    if verbose:
        print(f'transposed_df: \n{transposed_df}')

    return transposed_df


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



def simple_line_plot(indexed_df, fig_title=None, legend_title=None,
                     x_tick_vals=None, x_tick_labels=None,
                     x_axis=None, y_axis=None,
                     log_x=False, log_y=False,
                     save_as=None):
    """
    Function to plot a simple line plot.  No error bars.
    :param indexed_df: DF where index col is 'separation' or 'stair_names' etc.
    :param fig_title: Title for figure
    :param legend_title: Title for legend
    :param x_tick_vals: Values for x-ticks
    :param x_tick_labels: Labels for x ticks
    :param x_axis: Label for x-axis
    :param y_axis: Label for y-axis
    :param log_x: Make x-axis log scale
    :param log_y: Make y-axis log scale
    :param save_as: Full path (including name) to save to
    :return: Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=indexed_df, markers=True, dashes=False, ax=ax)
    if fig_title is not None:
        plt.title(fig_title)
    if legend_title is not None:
        plt.legend(title=legend_title)
    if x_tick_vals is not None:
        ax.set_xticks(x_tick_vals)
    if x_tick_labels is not None:
        ax.set_xticklabels(x_tick_labels)
        if -18 in x_tick_labels:
            # add dotted line at zero
            ax.axvline(x=5.5, linestyle="-.", color='lightgrey')
    if log_x:
        ax.set(xscale="log")
        x_axis = f'log {x_axis}'
    if log_y:
        ax.set(yscale="log")
        y_axis = f'log {y_axis}'
    if x_axis is not None:
        ax.set_xlabel(x_axis)
    if y_axis is not None:
        ax.set_ylabel(y_axis)
    if save_as is not None:
        plt.savefig(save_as)
    return fig


def run_thr_plot(thr_df, x_col='separation', y_col='ISI_0', hue_col='cond',
                 x_ticks_vals=None, x_tick_names=None,
                 x_axis_label='Probe cond (separation)',
                 y_axis_label='Probe Luminance',
                 fig_title='Ricco_v2: probe cond vs thr', save_as=None):
    """
    Function to make a simple plot from one run showing lineplots for circles, lines and 2probe data.
    Single threshold values so no error bars.

    :param thr_df: dataframe from one run
    :param x_col: column to use for x vals
    :param y_col: column to use for y vals
    :param hue_col: column to use for hue (different coloured lines on plot)
    :param x_ticks_vals: values to place on x-axis ticks
    :param x_tick_names: labels for x-tick values
    :param x_axis_label: x-axis label
    :param y_axis_label: y-axis label
    :param fig_title: figure title
    :param save_as: path and filename to save to
    :return: figure
    """
    print('*** running run_thr_plot (x=ordinal, y=thr) ***')
    fig, ax = plt.subplots(figsize=(10, 6))
    print(f'thr_df:\n{thr_df}')
    sns.lineplot(data=thr_df, x=x_col, y=y_col, hue=hue_col, marker='o')
    if x_ticks_vals is not None:
        ax.set_xticks(x_ticks_vals)
    if x_tick_names is not None:
        ax.set_xticklabels(x_tick_names)
    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(y_axis_label)
    plt.title(fig_title)
    if save_as:
        plt.savefig(save_as)
    print('*** finished run_thr_plot ***\n')
    return fig


def run_thr_plot_w_markers(thr_df, x_col='separation', y_col='ISI_0', hue_col='cond',
                           x_ticks_vals=None, x_tick_names=None,
                           x_axis_label='Probe cond (separation)',
                           y_axis_label='Probe Luminance',
                           legend_names=None,
                           fig_title='Ricco_v2: probe cond vs thr', save_as=None):
    """
    Function to make a simple plot from one run showing lineplots for circles, lines and 2probe data.
    Single threshold values so no error bars.

    Legend names can relate datapoints back to Exp1 sep conds, different to x-axis labels.

    :param thr_df: dataframe from one run
    :param x_col: column to use for x vals
    :param y_col: column to use for y vals
    :param hue_col: column to use for hue (different coloured lines on plot)
    :param x_ticks_vals: values to place on x-axis ticks
    :param x_tick_names: labels for x-tick values
    :param x_axis_label: x-axis label
    :param y_axis_label: y-axis label
    :param legend_names: labels for legend
    :param fig_title: figure title
    :param save_as: path and filename to save to
    :return: figure
    """
    print('\n*** running run_thr_plot_w_markers (x=ordinal, y=thr) ***')
    fig, ax = plt.subplots(figsize=(6, 6))
    print(f'thr_df:\n{thr_df}')
    sns.lineplot(data=thr_df, x=x_col, y=y_col, hue=hue_col, marker=None, ax=ax,
                 legend=True)

    sns.scatterplot(data=thr_df, x=x_col, y=y_col, hue=x_col, ax=ax,
                    legend=True, s=100,
                    palette=fig_colours(n_conditions=len(legend_names)),)

    if x_ticks_vals is not None:
        ax.set_xticks(x_ticks_vals)
    if x_tick_names is not None:
        ax.set_xticklabels(x_tick_names)
    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(y_axis_label)

    handles, labels = ax.get_legend_handles_labels()
    # for i, j in zip(handles, labels):
    #     print(i, j)
    handles = handles[1:]  # removes first handle (blue line)
    labels = legend_names  # + [labels[-1]]
    ax.legend(labels=labels, handles=handles)

    plt.title(fig_title)
    if save_as:
        plt.savefig(save_as)
    print('*** finished run_thr_plot_w_markers ***\n')
    return fig

def simple_log_log_plot(thr_df, x_col='area_deg', y_col='weber_thr', hue_col='cond',
                        x_ticks_vals=None, x_tick_names=None,
                        x_axis_label='log(area_deg)',
                        y_axis_label='log(∆ threshold)',
                        fig_title='Ricco_v2: log(area_deg) v log(thr)',
                        show_neg1slope=True,
                        save_as=None):
    """
    Function to make a simple plot from one run showing lineplots for circles, lines and 2probe data.
    Data is plotted on log-log axis (log(∆thr) and log(area_deg)).
    Single threshold values so no error bars.

    :param thr_df: dataframe from one run
    :param x_col: column to use for x vals
    :param y_col: column to use for y vals
    :param hue_col: column to use for hue (different coloured lines on plot)
    :param x_ticks_vals: values to place on x-axis ticks
    :param x_tick_names: labels for x-tick values
    :param x_axis_label: x-axis label
    :param y_axis_label: y-axis label
    :param fig_title: figure title
    :param show_neg1slope: If True, plots a line with slope=-1 starting from
            first datapoint of circles.
    :param save_as: path and filename to save to
    :return: figure
    """
    print(f'\n*** running simple_log_log_plot (x=log({x_col}), y=log(∆{y_col})) ***')
    print(f'thr_df:\n{thr_df}')
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.lineplot(data=thr_df, x=x_col, y=y_col, hue=hue_col, marker='o', ax=ax)
    if x_ticks_vals:
        ax.set_xticks(x_ticks_vals)
    if x_tick_names:
        ax.set_xticklabels(x_tick_names)
    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(y_axis_label)

    # set scale for axes (same on each)
    x_min = thr_df[x_col].min() * .9
    x_max = thr_df[x_col].max() * 1.1
    x_ratio = x_max / x_min
    y_min = thr_df[y_col].min() * .9
    y_max = thr_df[y_col].max() * 1.1
    y_ratio = y_max / y_min
    largest_diff = max([x_ratio, y_ratio])
    axis_range = 10 ** math.ceil(math.log10(largest_diff))

    ax.set(xlim=(x_min, x_min * axis_range), ylim=(y_min, y_min * axis_range))
    ax.set(xscale="log", yscale="log")

    # add guideline with slope of -1 which crosses through the circles 1probe weber_thr value.
    if show_neg1slope:
        # if x_col == 'area_deg':
        #     if '-1_circles' in thr_df['stair_names'].unique():
        #         start_point = '-1_circles'
        #     elif '-1_lines' in thr_df['stair_names'].unique():
        #         start_point = '-1_lines'
        #     slope_start_x = thr_df.loc[thr_df['stair_names'] == start_point, x_col].item()
        #     slope_start_y = thr_df.loc[thr_df['stair_names'] == start_point, y_col].item()
        # elif x_col == 'dur_ms':
        #     slope_start_x = thr_df.iloc[0]['dur_ms']
        #     slope_start_y = thr_df.iloc[0][y_col]
        # elif x_col == 'length':
        #     slope_start_x = thr_df.iloc[0]['length']
        #     slope_start_y = thr_df.iloc[0][y_col]
        slope_start_x = thr_df.iloc[0][x_col]
        slope_start_y = thr_df.iloc[0][y_col]
        print(f'slope_start_x: {slope_start_x}')
        print(f'slope_start_y: {slope_start_y}')
        ax.plot([slope_start_x, slope_start_x * 100], [slope_start_y, slope_start_y / 100], c='r',
                label='-1 slope', linestyle='dashed')
    ax.legend()
    plt.title(fig_title)
    if save_as:
        plt.savefig(save_as)
    print('*** finished simple_log_log_plot ***')
    return fig


def log_log_w_markers_plot(thr_df, x_col='len_deg', y_col='delta_I',
                           hue_col=None,
                           legend_names=None,
                           x_axis_label='log(len_deg)',
                           y_axis_label='log(∆ threshold)',
                           fig_title='Ricco_v6: log(len) v log(thr)',
                           show_neg1slope=True,
                           save_as=None):
    """
    Function to make a simple plot log-log plot with legend showing x-vals.
    Data is plotted on log-log axis (log(∆thr) and log(area_deg)).
    Single threshold values so no error bars.

    :param thr_df: dataframe from one run
    :param x_col: column to use for x vals
    :param y_col: column to use for y vals
    # :param hue_col: column to use for hue (different coloured lines on plot)
    # :param x_ticks_vals: values to place on x-axis ticks
    :param legend_names: labels for legend
    :param x_axis_label: x-axis label
    :param y_axis_label: y-axis label
    :param fig_title: figure title
    :param show_neg1slope: If True, plots a line with slope=-1 starting from
            first datapoint of circles.
    :param save_as: path and filename to save to
    :return: figure
    """
    print(f'\n*** running log_log_w_markers_plot (x=log({x_col}), y=log(∆{y_col})) ***')
    print(f'thr_df:\n{thr_df}')
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.lineplot(data=thr_df, x=x_col, y=y_col, hue=hue_col, marker=None, ax=ax,
                 legend=True)
    sns.scatterplot(data=thr_df, x=x_col, y=y_col, hue=x_col, ax=ax,
                    legend=True, s=100,
                    palette=fig_colours(n_conditions=len(legend_names)),)

    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(y_axis_label)

    # set scale for axes (same on each)
    x_min = thr_df[x_col].min() * .9
    x_max = thr_df[x_col].max() * 1.1
    x_ratio = x_max / x_min
    y_min = thr_df[y_col].min() * .9
    y_max = thr_df[y_col].max() * 1.1
    y_ratio = y_max / y_min
    largest_diff = max([x_ratio, y_ratio])
    axis_range = 10 ** math.ceil(math.log10(largest_diff)) / 5
    # axis_range = 10 ** math.ceil(math.log10(largest_diff))

    ax.set(xlim=(x_min, x_min * axis_range), ylim=(y_min, y_min * axis_range))
    ax.set(xscale="log", yscale="log")

    # add guideline with slope of -1 which crosses through the circles 1probe weber_thr value.
    if show_neg1slope:
        slope_start_x = thr_df.iloc[0][x_col]
        slope_start_y = thr_df.iloc[0][y_col]
        print(f'slope_start_x: {slope_start_x}')
        print(f'slope_start_y: {slope_start_y}')
        ax.plot([slope_start_x, slope_start_x * 100], [slope_start_y, slope_start_y / 100], c='r',
                label='-1 slope', linestyle='dashed')

    handles, labels = ax.get_legend_handles_labels()
    # for i, j in zip(handles, labels):
    #     print(i, j)
    # remove first handle (blue line), but keep last one (-1 slope)
    handles = handles[1:]
    # labels = legend_names + [labels[-1]]
    labels = ['-1 slope'] + legend_names
    ax.legend(labels=labels, handles=handles)

    plt.title(fig_title)
    if save_as:
        plt.savefig(save_as)
    print('*** finished log_log_w_markers_plot ***')
    return fig



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
        print(f"\npos_and_neg_sep_df:\n{pos_and_neg_sep_df}")
        print(f"\nx_tick_values: {x_tick_values}")
        print(f"x_tick_labels: {x_tick_labels}")

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
        ax.set_xticks(x_tick_values)
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
            plt.savefig(os.path.join(save_path, save_name))

    if verbose:
        print("\n*** finished plot_data_unsym_batman() ***\n")
    return fig


def plot_runs_ave_w_errors(fig_df, error_df,
                           jitter=True, error_caps=False, alt_colours=False,
                           legend_names=None,
                           x_tick_vals=None,
                           x_tick_labels=None,
                           even_spaced_x=False,
                           fixed_y_range=False,
                           x_axis_label=None,
                           y_axis_label=None,
                           log_log_axes=False,
                           neg1_slope=False,
                           slope_ycol_name=None,
                           slope_xcol_idx_depth=1,
                           fig_title=None, save_name=None, save_path=None,
                           verbose=True):
    """
    Calculate and plot the mean and error estimates (y-axis) at each separation values (x-axis).
    Separate line for each ISI.  Error bar values taken from separate error_df.

    :param fig_df: dataframe to build plot from.  Expects fig_df in the form:
        separation as index, ISIs as columns.
    :param error_df: dataframe of same shape as fig_df, but contains error values.
    :param jitter: Jitter x_axis values so points don't overlap.
    :param error_caps: caps on error bars for more easy reading.
    :param alt_colours: Use different set of colours to normal (e.g., if ISI on
        x-axis and lines for each separation).
    :param legend_names: Names of different lines (e.g., ISI names).
    :param x_tick_vals: Positions on x-axis.
    :param x_tick_labels: labels for x-axis.
    :param even_spaced_x: If True, x-ticks are evenly spaced,
        if False they will be spaced according to numeric value (e.g., 0, 1, 2, 3, 6, 18).
    :param fixed_y_range: default=False. If True, it uses full range of y values
        (e.g., 0:110) or can pass a tuple to set y_limits.
    :param x_axis_label: Label for x-axis.  If None passed, will use 'Probe separation in diagonal pixels'.
    :param y_axis_label: Label for y-axis.  If None passed, will use 'Probe Luminance'.
    :param log_log_axes: If True, both axes are in log scale, else in normal scale.
    :param neg1_slope: If True, adds a reference line with slope=-1.
    :param slope_ycol_name: Name of column to take the start of slope from
    :param slope_xcol_idx_depth: Some dfs have 2 index cols, so input 2.
    :param fig_title: Title for figure.
    :param save_name: filename of plot.
    :param save_path: path to folder where plots will be saved.
    :param verbose: print progress to screen.

    :return: figure
    """
    print('\n*** running plot_runs_ave_w_errors() ***\n')

    if verbose:
        print(f'fig_df:\n{fig_df}')
        print(f'\nerror_df:\n{error_df}')

    # get names for legend (e.g., different lines)
    column_names = fig_df.columns.to_list()

    # # remove 'prelim', 'flow_dir', 'flow_name' from column_names, if present
    # if 'prelim' in column_names:
    #     column_names.remove('prelim')
    # if 'flow_dir' in column_names:
    #     column_names.remove('flow_dir')
    # if 'flow_name' in column_names:
    #     column_names.remove('flow_name')
    # print(f"column_names: {column_names}")


    if legend_names is None:
        legend_names = column_names
    if verbose:
        print(f'\nColumn and Legend names:')
        for a, b in zip(column_names, legend_names):
            print(f"{a}\t=>\t{b}\tmatch: {bool(a == b)}")

    if x_tick_vals is None:
        x_tick_vals = fig_df.index

    # for evenly spaced items on x_axis
    if even_spaced_x:
        x_tick_vals = list(range(len(x_tick_vals)))


    if jitter:
        # adding jitter works well if df.index are all int
        # need to set it up to use x_tick_vals if df.index is not all int or float
        check_idx_num = all(isinstance(x, (int, float)) for x in fig_df.index)
        print(f'check_idx_num: {check_idx_num}')

        check_x_val_num = all(isinstance(x, (int, float)) for x in x_tick_vals)
        print(f'check_x_val_num: {check_x_val_num}')

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

        if verbose:
            print(f"{idx}. name: {name}")

        # get rand float to add to x-axis for jitter
        # jitter_list = np.random.uniform(size=n_pos_sep, low=-jit_max, high=jit_max)
        jitter_list = np.random.uniform(size=len(x_tick_vals), low=-jit_max, high=jit_max)
        x_values = x_tick_vals + jitter_list

        # print(f"idiot check\n"
        #       f"x_values: {x_values}\n"
        #       f"fig_df[name]: {fig_df[name]}\n"
        #       f"error_df[name]: {error_df[name]}")

        ax.errorbar(x=x_values, y=fig_df[name],
                    yerr=error_df[name],
                    marker='.', lw=2, elinewidth=.7,
                    capsize=cap_size, color=my_colours[idx])

        leg_handle = mlines.Line2D([], [], color=my_colours[idx], label=name,
                                   marker='.', linewidth=.5, markersize=4)
        legend_handles_list.append(leg_handle)

    # decorate plot
    if x_tick_vals is not None:
        ax.set_xticks(x_tick_vals)
    if x_tick_labels is not None:
        ax.set_xticks(x_tick_vals)
        ax.set_xticklabels(x_tick_labels)

    if x_axis_label is None:
        ax.set_xlabel('Probe separation in diagonal pixels')
    else:
        ax.set_xlabel(x_axis_label)

    if y_axis_label is None:
        ax.set_ylabel('Probe Luminance')
    else:
        ax.set_ylabel(y_axis_label)

    if fixed_y_range:
        ax.set_ylim([0, 110])
        if type(fixed_y_range) in [tuple, list]:
            ax.set_ylim([fixed_y_range[0], fixed_y_range[1]])

    if log_log_axes:
        ax.set(xscale="log", yscale="log")

    if neg1_slope:
        # add guideline with slope of -1 which crosses through the circles 1probe weber_thr value.
        slope_start_x = fig_df.index[0]
        if slope_xcol_idx_depth == 2:
            slope_start_x = fig_df.index[0][0]
        slope_start_y = fig_df.iloc[0][slope_ycol_name]

        # if 'circles' in column_names:
        #     slope_start_x = fig_df.index[0][0]
        #     slope_start_y = fig_df.iloc[0]['circles']
        # elif '1probe' in column_names:
        #     slope_start_x = fig_df.index[0]
        #     slope_start_y = fig_df.iloc[0]['1probe']
        # elif 'lines' in column_names:
        #     slope_start_x = fig_df.index[0][0]
        #     slope_start_y = fig_df.iloc[0]['lines']
        # elif 'weber_thr' in column_names:  # todo: be careful, this name is in a few dfs - not very discriminative
        #     slope_start_x = fig_df.index[0]
        #     slope_start_y = fig_df.iloc[0]['weber_thr']
        print(f'slope_start_x: {slope_start_x}')
        print(f'slope_start_y: {slope_start_y}')
        ax.plot([slope_start_x, slope_start_x * 100], [slope_start_y, slope_start_y / 100], c='r',
                label='-1 slope', linestyle='dashed')
        leg_handle = mlines.Line2D([], [], color='r', label='-1 slope', linestyle='dashed',
                                   # marker='.', linewidth=.5, markersize=4
                                   )
        legend_handles_list.append(leg_handle)

    ax.legend(handles=legend_handles_list, fontsize=6,
              framealpha=.5)

    if fig_title is not None:
        plt.title(fig_title)

    if save_path is not None:
        if save_name is not None:
            plt.savefig(os.path.join(save_path, save_name))
            print(f'plt saved to: {os.path.join(save_path, save_name)}')

    return fig


def plot_ave_w_errors_markers(fig_df, error_df,
                              jitter=True, error_caps=False, alt_colours=False,
                              legend_names=None,
                              x_tick_vals=None,
                              x_tick_labels=None,
                              even_spaced_x=False,
                              fixed_y_range=False,
                              x_axis_label=None,
                              y_axis_label=None,
                              log_log_axes=False,
                              neg1_slope=False,
                              slope_ycol_name=None,
                              slope_xcol_idx_depth=1,
                              fig_title=None, save_name=None, save_path=None,
                              verbose=True):
    """
    Calculate and plot the mean and error estimates (y-axis) at each separation values (x-axis).
    Separate line for each ISI.  Error bar values taken from separate error_df.

    :param fig_df: dataframe to build plot from.  Expects fig_df in the form:
        separation as index, ISIs as columns.
    :param error_df: dataframe of same shape as fig_df, but contains error values.
    :param jitter: Jitter x_axis values so points don't overlap.
    :param error_caps: caps on error bars for more easy reading.
    :param alt_colours: Use different set of colours to normal (e.g., if ISI on
        x-axis and lines for each separation).
    :param legend_names: Names of markers (different to x-axis_labels, e.g., sep_cond)
    :param x_tick_vals: Positions on x-axis.
    :param x_tick_labels: labels for x-axis.
    :param even_spaced_x: If True, x-ticks are evenly spaced,
        if False they will be spaced according to numeric value (e.g., 0, 1, 2, 3, 6, 18).
    :param fixed_y_range: default=False. If True, it uses full range of y values
        (e.g., 0:110) or can pass a tuple to set y_limits.
    :param x_axis_label: Label for x-axis.  If None passed, will use 'Probe separation in diagonal pixels'.
    :param y_axis_label: Label for y-axis.  If None passed, will use 'Probe Luminance'.
    :param log_log_axes: If True, both axes are in log scale, else in normal scale.
    :param neg1_slope: If True, adds a reference line with slope=-1.
    :param slope_ycol_name: Name of column to take the start of slope from
    :param slope_xcol_idx_depth: Some dfs have 2 index cols, so input 2.
    :param fig_title: Title for figure.
    :param save_name: filename of plot.
    :param save_path: path to folder where plots will be saved.
    :param verbose: print progress to screen.

    :return: figure
    """
    print('\n*** running plot_ave_w_errors_markers() ***\n')

    if verbose:
        print(f'fig_df:\n{fig_df}')
        print(f'\nerror_df:\n{error_df}')

    # get names for legend (e.g., different lines)
    column_names = fig_df.columns.to_list()

    if legend_names is None:
        legend_names = column_names

    if x_tick_vals is None:
        x_tick_vals = fig_df.index

    # for evenly spaced items on x_axis
    if even_spaced_x:
        x_tick_vals = list(range(len(x_tick_vals)))


    if jitter:
        # adding jitter works well if df.index are all int
        # need to set it up to use x_tick_vals if df.index is not all int or float
        check_idx_num = all(isinstance(x, (int, float)) for x in fig_df.index)
        print(f'check_idx_num: {check_idx_num}')

        check_x_val_num = all(isinstance(x, (int, float)) for x in x_tick_vals)
        print(f'check_x_val_num: {check_x_val_num}')

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
        cap_size = 10

    # set colour palette
    my_colours = fig_colours(len(legend_names), alternative_colours=alt_colours)
    for i in my_colours:
        print(i)
    fig, ax = plt.subplots()

    legend_handles_list = []

    for idx, name in enumerate(column_names):
        print(idx, name)
        # get rand float to add to x-axis for jitter
        jitter_list = np.random.uniform(size=n_pos_sep, low=-jit_max, high=jit_max)
        x_values = x_tick_vals + jitter_list

        y_values = fig_df[name].to_list()
        y_err_values = error_df[name].to_list()

        # blue line plot
        sns.lineplot(data=fig_df, x=x_values, y=name, marker=None, ax=ax, zorder=1)

        # error bars with different colours
        for x_val, y_val, y_err_val, color, name in \
                zip(x_values, y_values, y_err_values, my_colours, legend_names):
            ax.errorbar(x=x_val, y=y_val, yerr=y_err_val,
                        elinewidth=3, capsize=cap_size, ecolor=color, zorder=2)

            leg_handle = mlines.Line2D([], [], color=color, label=name,
                                       marker='P', linewidth=0)
            legend_handles_list.append(leg_handle)

    # decorate plot
    if x_tick_vals is not None:
        ax.set_xticks(x_tick_vals)
    if x_tick_labels is not None:
        ax.set_xticks(x_tick_vals)
        ax.set_xticklabels(x_tick_labels)

    if x_axis_label is None:
        ax.set_xlabel('Probe separation in diagonal pixels')
    else:
        ax.set_xlabel(x_axis_label)

    if y_axis_label is None:
        ax.set_ylabel('Probe Luminance')
    else:
        ax.set_ylabel(y_axis_label)

    if fixed_y_range:
        ax.set_ylim([0, 110])
        if type(fixed_y_range) in [tuple, list]:
            ax.set_ylim([fixed_y_range[0], fixed_y_range[1]])

    if log_log_axes:
        ax.set(xscale="log", yscale="log")

    if neg1_slope:
        # add guideline with slope of -1 which crosses through the circles 1probe weber_thr value.
        slope_start_x = fig_df.index[0]
        if slope_xcol_idx_depth == 2:
            slope_start_x = fig_df.index[0][0]
        slope_start_y = fig_df.iloc[0][slope_ycol_name]
        print(f'slope_start_x: {slope_start_x}')
        print(f'slope_start_y: {slope_start_y}')
        # ax.plot([slope_start_x, slope_start_x * 100], [slope_start_y, slope_start_y / 100], c='r',
        ax.plot([slope_start_x, slope_start_x * 11], [slope_start_y, slope_start_y / 11], c='r',
                label='-1 slope', linestyle='dashed', zorder=0)
        leg_handle = mlines.Line2D([], [], color='r', label='-1 slope', linestyle='dashed',
                                   # marker='.', linewidth=.5, markersize=4
                                   )
        legend_handles_list.append(leg_handle)

    ax.legend(handles=legend_handles_list, framealpha=.5
              # fontsize=6,
              )

    if fig_title is not None:
        plt.title(fig_title)

    if save_path is not None:
        if save_name is not None:
            plt.savefig(os.path.join(save_path, save_name))
            print(f'plt saved to: {os.path.join(save_path, save_name)}')

    print('\n*** finished plot_ave_w_errors_markers() ***\n')

    return fig


def plot_w_errors_either_x_axis(wide_df, cols_to_keep=['congruent', 'separation'],
                                cols_to_change=['ISI_1', 'ISI_4', 'ISI_6'],
                                cols_to_change_show='newLum', new_col_name='ISI',
                                strip_from_cols='ISI_',
                                x_axis='separation', y_axis='newLum',
                                hue_var='ISI', style_var='congruent', style_order=[1, -1],
                                error_bars=False,
                                jitter=False,
                                log_scale=False,
                                even_spaced_x=False,
                                x_tick_vals=None,
                                legend_names=None,
                                fig_title=None,
                                fig_savename=None,
                                save_path=None,
                                verbose=True):
    """
    Function to plot line_plot with multiple lines.  This function works with a single dataset,
    or with a df containing several datasets, in which case it plots the mean
    with error bars at .68 confidence interval.
    It will work with either separation on x-axis and different lines for each ISI, or vice versa.
    The first part of the function converts a wide dataframe to a long dataframe.

    :param wide_df: Data to be plotted.
    :param cols_to_keep: Variables that will be included in long dataframe.
    :param cols_to_change: Columns containing different measurements of some
        variable (e.g., ISI_1, ISI_4, ISI_6...etc) that will be converted into
        longform (e.g., ISI: [1, 4, 6]).
    :param cols_to_change_show: What is being measured in cols to change (e.g., newLum; dependent variable).
    :param new_col_name: What the cols to change describe (e.g., isi; independent variable).
    :param strip_from_cols: string to remove if independent variables are to be
        turned into numeric values (e.g., for ISI_1, ISI_4, ISI_6, strip 'ISI_' to get 1, 4,6).
    :param x_axis: Variable to be shown along x-axis (e.g., separation or isi).
    :param y_axis: Variable to be shown along y-axis (e.g., newLum).
    :param hue_var: Variable to be shown with different lines (e.g., isi or separation).
    :param style_var: Addition variable to show with solid or dashed lines (e.g., congruent or incongruent).
    :param style_order: Order of style var as displayed in df (e.g., [1, -1]).
    :param error_bars: True or false, whether to display error bars (SE).
    :param jitter: Whether to jitter items on x-axis to make easier to read.
        Can be True, False or float for amount of jitter in relation to x-axis values.
    :param log_scale: Put axes onto log scale.
    :param even_spaced_x: Whether to evenly space ticks on x-axis.
        For example to make the left side of log-scale-like x-values easier to read.
    :param x_tick_vals: Values/labels for x-axis.  Can be string, int or float.
    :param fig_title: Title for figure.
    :param fig_savename: Save name for figure.
    :param save_path: Save path for figure.
    :param verbose: Whether to print progress to screen.

    :return: figure
    """
    print('\n*** running plot_w_errors_either_x_axis() ***\n')

    # convert wide df to long
    long_df = make_long_df(wide_df=wide_df, cols_to_keep=cols_to_keep,
                           cols_to_change=cols_to_change, cols_to_change_show=cols_to_change_show,
                           new_col_name=new_col_name, strip_from_cols=strip_from_cols, verbose=verbose)

    # data_for_x to use for x_axis data, whilst keeping original values list (x_axis)
    data_for_x = x_axis

    if log_scale:
        even_spaced_x = False

    # for evenly spaced items on x_axis
    if even_spaced_x:
        orig_x_vals = x_tick_vals  # sorted(x_tick_vals)
        new_x_vals = list(range(len(orig_x_vals)))

        # check if x_tick_vals refer to values in df, if not, use df values.
        if list(long_df[x_axis])[0] in orig_x_vals:
            x_space_dict = dict(zip(orig_x_vals, new_x_vals))
        else:
            print("warning: x_tick_vals don't appear in long_df")
            found_x_vals = sorted(set(list(long_df[x_axis])))
            print(f'found_x_vals (from df): {found_x_vals}')
            x_space_dict = dict(zip(found_x_vals, new_x_vals))

        # add column with new evenly spaced x-values, relating to original x_values

        # sort long df by x_axis so tick labels are in ascending order
        long_df.sort_values(by=x_axis, inplace=True)

        spaced_x = [x_space_dict[i] for i in list(long_df[x_axis])]
        long_df.insert(0, 'spaced_x', spaced_x)
        data_for_x = 'spaced_x'
        if verbose:
            print(f'orig_x_vals: {orig_x_vals}')
            print(f'new_x_vals: {new_x_vals}')
            print(f'x_space_dict: {x_space_dict}')

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
        jitter_x = [a + jitter_dict[b] for a, b in zip(list(long_df[data_for_x]), list(long_df[hue_var]))]
        long_df.insert(0, f'jitter_{hue_var}_x', jitter_x)
        data_for_x = f'jitter_{hue_var}_x'

    conf_interval = None
    if error_bars:
        conf_interval = 68

    if verbose:
        print(f'long_df:\n{long_df}')
        print(f'error_bars: {error_bars}')
        print(f'conf_interval: {conf_interval}')
        print(f'x_tick_vals: {x_tick_vals}')
        print(f'data_for_x: {data_for_x}')
        print(f'y_axis: {y_axis}')
        print(f'hue_var: {hue_var}')
        print(f'style_var: {style_var}')
        print(f'style_order: {style_order}')

    # initialize plot
    if hue_var is None:
        my_colours = fig_colours(n_conditions=5)
    else:
        my_colours = fig_colours(n_conditions=len(set(list(long_df[hue_var]))))

    fig, ax = plt.subplots(figsize=(10, 6))

    # with error bars for d_averages example
    sns.lineplot(data=long_df, x=data_for_x, y=y_axis, hue=hue_var,
                 style=style_var, style_order=style_order,
                 estimator='mean',
                 errorbar='se', err_style='bars', err_kws={'elinewidth': 1, 'capsize': 5},
                 palette=my_colours, ax=ax)

    if log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')
    elif even_spaced_x:
        ax.set_xticks(new_x_vals)
        ax.set_xticklabels(orig_x_vals)
    else:
        ax.set_xticklabels(x_tick_vals)

    plt.xlabel(x_axis)
    plt.title(fig_title)

    # Change legend labels for congruent and incongruent data
    if hue_var != 'stair_names':
        handles, labels = ax.get_legend_handles_labels()
        # ax.legend(handles=handles, labels=labels[:-2] + ['True', 'False'])
        if legend_names is not None:
            if len(legend_names) == len(labels):
                labels=legend_names
        ax.legend(handles=handles, labels=labels)

    plt.savefig(os.path.join(save_path, fig_savename))
    print(f'plt saved to: {os.path.join(save_path, fig_savename)}')

    print('\n*** finished plot_w_errors_either_x_axis() ***\n')

    return fig


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
    # n_rows, n_cols = multi_plot_shape(len(isi_name_list), min_rows=2)
    n_rows, n_cols = get_n_rows_n_cols((len(isi_name_list)))
    print(f'\nplotting {n_rows} rows and {n_cols} cols')

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12, 6))
    print(f'\nplotting {n_rows} rows and {n_cols} cols (axes: {axes})')

    if fig_title is not None:
        fig.suptitle(fig_title)

    ax_counter = 0
    # loop through the eight axes
    for row_idx, row in enumerate(axes):
        print(f'\nrow_idx: {row_idx}, row: {row} type(row): {type(row)}')

        # if there are multiple ISIs
        if isinstance(row, np.ndarray):
            for col_idx, ax in enumerate(row):
                print(f'col_idx: {col_idx}, ax: {ax}')

                if ax_counter < len(isi_name_list):

                    print(f'\t{ax_counter}. isi_name_list[ax_counter]: {isi_name_list[ax_counter]}')

                    # mean threshold from CW and CCW probe jump direction
                    sns.lineplot(ax=axes[row_idx, col_idx], data=mean_df,
                                 x='separation', y=isi_name_list[ax_counter],
                                 color=my_colours[ax_counter],
                                 linewidth=.5,
                                 markers=True)

                    sns.lineplot(ax=axes[row_idx, col_idx], data=thr1_df,
                                 x='separation', y=isi_name_list[ax_counter],
                                 color=my_colours[ax_counter],
                                 linestyle="dashed",
                                 marker="v")

                    sns.lineplot(ax=axes[row_idx, col_idx], data=thr2_df,
                                 x='separation', y=isi_name_list[ax_counter],
                                 color=my_colours[ax_counter],
                                 linestyle="dotted",
                                 marker="o")

                    ax.set_title(isi_name_list[ax_counter])
                    ax.set_xticks(x_tick_vals)
                    ax.set_xticklabels(x_tick_labels)
                    ax.xaxis.set_tick_params(labelsize=6)

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
                                        marker='v',
                                        linestyle="dashed",
                                        markersize=4, label='Congruent')
                    st2 = mlines.Line2D([], [], color=my_colours[ax_counter],
                                        marker='o',
                                        linestyle="dotted",
                                        markersize=4, label='Incongruent')
                    mean_line = mlines.Line2D([], [], color=my_colours[ax_counter],
                                              marker=None,
                                              linewidth=.5,
                                              label='mean')
                    ax.legend(handles=[st1, st2, mean_line], fontsize=6)

                    ax_counter += 1
                else:
                    fig.delaxes(ax=axes[row_idx, col_idx])

        # if there is only 1 ISI in this row
        else:
            print(f'last plot\n'
                  f'{row_idx}. isi_name_list[row_idx]: {isi_name_list[row_idx]}')

            ax = row
            # mean threshold from CW and CCW probe jump direction
            sns.lineplot(ax=axes[row_idx], data=mean_df,
                         x='separation', y=isi_name_list[row_idx],
                         color=my_colours[row_idx],
                         linewidth=.5,
                         markers=True)

            sns.lineplot(ax=axes[row_idx], data=thr1_df,
                         x='separation', y=isi_name_list[row_idx],
                         color=my_colours[row_idx], linestyle="dashed",
                         marker="v")

            sns.lineplot(ax=axes[row_idx], data=thr2_df,
                         x='separation', y=isi_name_list[row_idx],
                         color=my_colours[row_idx], linestyle="dotted",
                         marker="o")

            ax.set_title(isi_name_list[row_idx])
            ax.set_xticks(x_tick_vals)
            ax.set_xticklabels(x_tick_labels)
            ax.xaxis.set_tick_params(labelsize=6)

            ax.set_xlabel('Probe separation (pixels)')
            ax.set_ylabel('Probe Luminance')

            if sym_sep_diff_list is not None:
                ax.text(x=0.4, y=0.8, s=round(sym_sep_diff_list[row_idx], 2),
                        # needs transform to appear with rest of plot.
                        transform=ax.transAxes, fontsize=12)

            # artist for legend
            st1 = mlines.Line2D([], [], color=my_colours[row_idx],
                                marker='v',
                                linestyle="dashed",
                                markersize=4, label='Congruent')
            st2 = mlines.Line2D([], [], color=my_colours[row_idx],
                                marker='o',
                                linestyle="dotted",
                                markersize=4, label='Incongruent')
            mean_line = mlines.Line2D([], [], color=my_colours[row_idx],
                                      marker=None,
                                      linewidth=.5,
                                      label='mean')
            ax.legend(handles=[st1, st2, mean_line], fontsize=6)

            print(f'ax_counter: {ax_counter}, len(isi_name_list): {len(isi_name_list)}')
            if ax_counter + 1 == len(isi_name_list):
                print(f'idiot check, no more plots to make here')
                # continue
                break

    plt.tight_layout()

    if save_path is not None:
        if save_name is not None:
            plt.savefig(os.path.join(save_path, save_name))
            print(f'plt saved to: {os.path.join(save_path, save_name)}')

    print("\n*** finished multi_batman_plots() ***")

    return fig


def multi_pos_sep_per_isi(ave_thr_df, error_df,
                          stair_names_col='stair_names',
                          pos_neg_labels=['Congrent', 'Incongruent'],
                          even_spaced_x=True, error_caps=True,
                          fig_title=None,
                          save_path=None, save_name=None,
                          verbose=True):
    """
    Function to plot multi-plot for comparing cong and incong for each isi.

    :param ave_thr_df: dataframe to analyse containing mean thresholds
    :param error_df: dataframe containing error values
    :param stair_names_col: name of column containing separation and congruent info
    :param pos_neg_labels: names of positive (1st) and negative (2nd) values e.g., ['Congruent', 'Incongruent']
    :param even_spaced_x: If true will evenly space ticks on x-axis.
        If false will use values given which might not be evenly spaces (e.g., 1, 2, 3, 6, 18)
    :param error_caps: Whether to add caps to error bars
    :param fig_title: Title for page of figures
    :param save_path: directory to save into
    :param save_name: name of saved file
    :param verbose: if Ture, will print progress to screen

    :return: figure
    """
    print("\n*** running multi_pos_sep_per_isi() ***")

    # if stair_names col is being used as the index, change stair_names to regular column and add index.
    if ave_thr_df.index.name == stair_names_col:
        # get the dataframe max and min values for y_axis limits.
        max_thr = ave_thr_df.max().max() + 5
        min_thr = ave_thr_df.min().min() - 5
        ave_thr_df.reset_index(drop=False, inplace=True)
        error_df.reset_index(drop=False, inplace=True)
    else:
        get_min_max = ave_thr_df.iloc[:, 1:]
        max_thr = get_min_max.max().max() + 5
        min_thr = get_min_max.min().min() - 5

    if verbose:
        print(f'ave_thr_df:\n{ave_thr_df}\n'
              f'error_df:\n{error_df}')
        print(f'max_thr: {max_thr}\nmin_thr: {min_thr}')

    stair_names_list = ave_thr_df[stair_names_col].to_list()


    if abs(stair_names_list[0]) == abs(stair_names_list[1]):
        print('these are alternating pairs of cong incong')
        cong_rows = sorted(ave_thr_df.index[ave_thr_df[stair_names_col] >= 0].tolist(), reverse=True)
        incong_rows = sorted(ave_thr_df.index[ave_thr_df[stair_names_col] < 0].tolist(), reverse=True)
    elif abs(stair_names_list[0]) == abs(stair_names_list[-1]):
        print('these are mirror image of all cong then all incong in rev order\n'
              "don't reverse cong rows")
        cong_rows = sorted(ave_thr_df.index[ave_thr_df[stair_names_col] >= 0].tolist(), reverse=False)
        incong_rows = sorted(ave_thr_df.index[ave_thr_df[stair_names_col] < 0].tolist(), reverse=True)
    elif abs(sorted(stair_names_list)[0]) == abs(sorted(stair_names_list)[-1]):
        print("these values might be repeated (e.g., [-1, -2, -3, 1, 2, 3], so don't sort them")
        cong_rows = ave_thr_df.index[ave_thr_df[stair_names_col] >= 0].tolist()
        incong_rows = ave_thr_df.index[ave_thr_df[stair_names_col] < 0].tolist()
    else:
        raise ValueError(f"Not sure which rows are paired in this dataframe.\n{ave_thr_df}")


    if verbose:
        print(f'\ncong_rows: {cong_rows}')
        print(f'incong_rows: {incong_rows}')

    # # slice rows for cong and incong df
    cong_df = ave_thr_df.iloc[cong_rows, :]
    cong_err_df = error_df.iloc[cong_rows, :]
    incong_df = ave_thr_df.iloc[incong_rows, :]
    incong_err_df = error_df.iloc[incong_rows, :]


    pos_sep_list = [int(i) for i in list(sorted(cong_df[stair_names_col].tolist()))]
    isi_names_list = list(cong_df.columns)[1:]

    cong_df.reset_index(drop=True, inplace=True)
    cong_err_df.reset_index(drop=True, inplace=True)
    incong_df.reset_index(drop=True, inplace=True)
    incong_err_df.reset_index(drop=True, inplace=True)
    if verbose:
        print(f'\npos_sep_list: {pos_sep_list}')
        print(f'isi_names_list: {isi_names_list}')
        print(f'\ncong_df: {cong_df.shape}\n{cong_df}')
        print(f'\ncong_err_df: {cong_err_df.shape}\n{cong_err_df}')
        print(f'\nincong_df: {incong_df.shape}\n{incong_df}')
        print(f'\nincong_err_df: {incong_err_df.shape}\n{incong_err_df}\n')

    if type(pos_neg_labels[0]) != str:
        pos_neg_labels = ['Congruent', 'Incongruent']

    cap_size = 0
    if error_caps:
        cap_size = 5

    if even_spaced_x:
        x_values = list(range(len(pos_sep_list)))
    else:
        x_values = pos_sep_list

    # todo: update make colours to allow me to use tab20 to give me 10 pairs of colours.
    #  Check that they are correctly arranged so that inward and outward motion colours go together for each isi.

    # make plots
    my_colours = fig_colours(len(isi_names_list))

    # # n_rows, n_cols = multi_plot_shape(len(isi_names_list), min_rows=2)
    # n_rows, n_cols = get_n_rows_n_cols((len(isi_names_list)))
    # fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_rows * 3, n_cols * 3))

    # get configuration of subplots
    n_plots = len(isi_names_list)  #  + 1
    n_rows, n_cols = get_n_rows_n_cols(n_plots)
    print(f"n_plots: {n_plots}, n_rows: {n_rows}, n_cols: {n_cols}, empty: {(n_rows * n_cols) - n_plots}")

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(3 * n_cols, 3 * n_rows))

    print(f'\nplotting {n_rows} rows and {n_cols} cols for {len(axes)} plots')
    print(f'type(axes): {type(axes)}; axes: {axes}')

    if fig_title is not None:
        fig.suptitle(fig_title)


    ax_counter = 0
    # loop through the different axes
    for row_idx, row in enumerate(axes):

        print(f'row_idx: {row_idx}, type(row): {type(row)}, row: {row}')
        # if there are multiple ISIs
        if isinstance(row, np.ndarray):
            print(f'type (AxesSubplot): {type(row)}')
            for col_idx, ax in enumerate(row):
                print(f'col_idx: {col_idx}; ax: {ax}')

                if ax_counter < len(isi_names_list):

                    this_isi = isi_names_list[ax_counter]
                    print(f'this_isi: {this_isi}')

                    # plot each datapoint for congruent

                    # plot each datapoint for incongruent


                    # plots means and errors for congruent
                    ax.errorbar(x=x_values, y=cong_df[this_isi],
                                yerr=cong_err_df[this_isi],
                                marker=None, lw=2, elinewidth=.7,
                                capsize=cap_size,
                                color=my_colours[ax_counter])

                    # plots means and errors for incongruent
                    ax.errorbar(x=x_values, y=incong_df[this_isi],
                                yerr=incong_err_df[this_isi],
                                linestyle='dashed',
                                marker=None, lw=2, elinewidth=.7,
                                capsize=cap_size,
                                color=my_colours[ax_counter])

                    ax.set_title(isi_names_list[ax_counter])
                    if even_spaced_x:
                        ax.set_xticks(list(range(len(pos_sep_list))))
                    else:
                        ax.set_xticks(pos_sep_list)
                    ax.set_xticklabels(pos_sep_list)

                    # todo: tidy this function - get rid of min and max calls and commented out stuff.
                    # ax.set_ylim([min_thr, max_thr])

                    if row_idx == 1:
                        ax.set_xlabel('Probe separation (pixels)')
                    else:
                        ax.xaxis.label.set_visible(False)

                    if col_idx == 0:
                        ax.set_ylabel('Probe Luminance')
                    else:
                        ax.yaxis.label.set_visible(False)

                    # artist for legend
                    
                    st1 = mlines.Line2D([], [], color=my_colours[ax_counter],
                                        # marker='v',
                                        linewidth=.5,
                                        # markersize=4, label='Congruent')
                                        markersize=4, label=pos_neg_labels[0])

                    st2 = mlines.Line2D([], [], color=my_colours[ax_counter],
                                        # marker='o',
                                        marker=None, linewidth=.5, linestyle="dotted",
                                        # markersize=4, label='Incongruent')
                                        markersize=4, label=pos_neg_labels[1])

                    ax.legend(handles=[st1, st2], fontsize=6)

                    ax_counter += 1
                else:
                    fig.delaxes(ax=axes[row_idx, col_idx])

        # if there is only one isi in this row
        else:
            print(f'type (NOT AxesSubplot):{type(row)}')

            ax = row
            this_isi = isi_names_list[row_idx]
            print(f'\nax: {ax}; this_isi: {this_isi}')
            print(f'x_values: {x_values}')
            print(f'cong_df[this_isi].top_list(): {cong_df[this_isi].to_list()}')
            print(f'cong_err_df[this_isi].top_list(): {cong_err_df[this_isi].to_list()}')
            check_nan = cong_err_df[this_isi].isnull().values.any()
            print(f'check_nan: {check_nan}')
            if check_nan:

                ax.errorbar(x=x_values, y=cong_df[this_isi],
                            # yerr=cong_err_df[this_isi],
                            linestyle='dashed',
                            marker=None, lw=2,
                            # elinewidth=.7, capsize=cap_size,
                            color=my_colours[row_idx])

                ax.errorbar(x=x_values, y=incong_df[this_isi],
                            # yerr=incong_err_df[this_isi],
                            marker=None, lw=2,
                            # elinewidth=.7, capsize=cap_size,
                            color=my_colours[row_idx])

            else:
                # if NOT nan

                ax.errorbar(x=x_values, y=cong_df[this_isi],
                            yerr=cong_err_df[this_isi],
                            linestyle='dashed',
                            marker=None, lw=2,
                            elinewidth=.7, capsize=cap_size,
                            color=my_colours[row_idx])

                ax.errorbar(x=x_values, y=incong_df[this_isi],
                            yerr=incong_err_df[this_isi],
                            marker=None, lw=2,
                            elinewidth=.7, capsize=cap_size,
                            color=my_colours[row_idx])

            ax.set_title(isi_names_list[row_idx])
            if even_spaced_x:
                ax.set_xticks(list(range(len(pos_sep_list))))
            else:
                ax.set_xticks(pos_sep_list)
            ax.set_xticklabels(pos_sep_list)

            # ax.set_ylim([min_thr, max_thr])

            # if row_idx == 1:
            ax.set_xlabel('Probe separation (pixels)')
            # else:
            #     ax.xaxis.label.set_visible(False)

            # if col_idx == 0:
            ax.set_ylabel('Probe Luminance')
            # else:
            #     ax.yaxis.label.set_visible(False)

            # artist for legend
            st1 = mlines.Line2D([], [], color=my_colours[row_idx],
                                # marker='v',
                                linewidth=.5,
                                # markersize=4, label='Congruent')
                                markersize=4, label=pos_neg_labels[0])

            st2 = mlines.Line2D([], [], color=my_colours[row_idx],
                                # marker='o',
                                marker=None, linewidth=.5, linestyle="dotted",
                                # markersize=4, label='Incongruent')
                                markersize=4, label=pos_neg_labels[1])

            ax.legend(handles=[st1, st2], fontsize=6)

            print(f'ax_counter: {ax_counter}, len(isi_name_list): {len(isi_names_list)}')
            if ax_counter + 1 == len(isi_names_list):
                print(f'idiot check, no more plots to make here')
                break

    plt.tight_layout()

    if save_path is not None:
        if save_name is not None:
            plt.savefig(os.path.join(save_path, save_name))
            print(f'plt saved to: {os.path.join(save_path, save_name)}')


    print("\n*** finished multi_pos_sep_per_isi() ***")

    return fig


def multi_plt_per_col_w_hue(ave_thr_df, error_df,
                            cond_type_col='cond_type',
                            pos_neg_labels=['Congrent', 'Incongruent'],
                            x_label_col='separation',
                            even_spaced_x=True, error_caps=True,
                            fig_title=None,
                            save_path=None, save_name=None,
                            verbose=True):
    """
    Function to plot multi-plot for comparing cong and incong for each isi.

    :param ave_thr_df: dataframe to analyse containing mean thresholds
    :param error_df: dataframe containing error values
    :param cond_type_col: name of column containing condition type info e.g., 'cond_type'
    :param pos_neg_labels: names of positive (1st) and negative (2nd) values e.g., ['Congruent', 'Incongruent']
    :param x_label_col: name of column containing x-axis labels e.g., 'separation'
    :param even_spaced_x: If true will evenly space ticks on x-axis.
        If false will use values given which might not be evenly spaces (e.g., 1, 2, 3, 6, 18)
    :param error_caps: Whether to add caps to error bars
    :param fig_title: Title for page of figures
    :param save_path: directory to save into
    :param save_name: name of saved file
    :param verbose: if Ture, will print progress to screen

    :return: figure
    """
    print("\n*** running multi_plt_per_col_w_hue() ***")

    # get names of columns that are not cond_type_col or x_label_col
    each_plot_col = [i for i in ave_thr_df.columns if i not in [cond_type_col, x_label_col]]

    # get a list of x_tick_labels from sorted set of ave_thr_df[x_label_col]
    x_tick_labels = sorted(set(ave_thr_df[x_label_col]))
    x_tick_labels = [int(i) for i in x_tick_labels]

    if verbose:
        print(f'ave_thr_df:\n{ave_thr_df}\n'
              f'error_df:\n{error_df}')
        print(f"each_plot_col: {each_plot_col}")
        print(f"x_tick_labels: {x_tick_labels}")

    # make two dataframes, cond_0_df and cond_1_df based on pos_neg_labels
    if not all(i in ave_thr_df[cond_type_col].unique() for i in pos_neg_labels):
        raise ValueError(
            f"one or more of the pos_neg_labels ({pos_neg_labels}) not in cond_type_col ({ave_thr_df[cond_type_col].unique()})")
    cond_0_df = ave_thr_df[ave_thr_df[cond_type_col] == pos_neg_labels[0]].copy()
    cond_1_df = ave_thr_df[ave_thr_df[cond_type_col] == pos_neg_labels[1]].copy()
    cond_0_err_df = error_df[error_df[cond_type_col] == pos_neg_labels[0]].copy()
    cond_1_err_df = error_df[error_df[cond_type_col] == pos_neg_labels[1]].copy()

    # drop cond_type_col from all four dataframes
    cond_0_df.drop(columns=cond_type_col, inplace=True)
    cond_1_df.drop(columns=cond_type_col, inplace=True)
    cond_0_err_df.drop(columns=cond_type_col, inplace=True)
    cond_1_err_df.drop(columns=cond_type_col, inplace=True)

    if verbose:
        print(f'\ncond_0_df: {cond_0_df.shape}\n{cond_0_df}')
        print(f'cond_0_err_df: {cond_0_err_df.shape}\n{cond_0_err_df}')
        print(f'cond_1_df: {cond_1_df.shape}\n{cond_1_df}')
        print(f'cond_1_err_df: {cond_1_err_df.shape}\n{cond_1_err_df}\n')

    cap_size = 0
    if error_caps:
        cap_size = 5

    if even_spaced_x:
        x_values = list(range(len(x_tick_labels)))
    else:
        x_values = x_tick_labels

    # make plots
    my_colours = fig_colours(len(each_plot_col))

    # get configuration of subplots
    n_plots = len(each_plot_col)  # + 1
    n_rows, n_cols = get_n_rows_n_cols(n_plots)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(3 * n_cols, 3 * n_rows))
    print(
        f'\nplotting {n_rows} rows and {n_cols} cols for {len(axes)} plots (with {(n_rows * n_cols) - n_plots} empty)')

    if fig_title is not None:
        fig.suptitle(fig_title)

    ax_counter = 0
    # loop through the different axes
    for row_idx, row in enumerate(axes):

        print(f'\nrow_idx: {row_idx}, type(row): {type(row)}, row: {row}')
        # if there are multiple ISIs
        if isinstance(row, np.ndarray):
            print(f'type (AxesSubplot): {type(row)}')
            for col_idx, ax in enumerate(row):
                print(f'col_idx: {col_idx}; ax: {ax}')

                if ax_counter < len(each_plot_col):

                    this_col_label = each_plot_col[ax_counter]
                    print(f'this_col_label: {this_col_label}')

                    # plots means and errors for congruent
                    ax.errorbar(x=x_values, y=cond_0_df[this_col_label],
                                yerr=cond_0_err_df[this_col_label],
                                marker=None, lw=2, elinewidth=.7,
                                capsize=cap_size,
                                color=my_colours[ax_counter])

                    # plots means and errors for incongruent
                    ax.errorbar(x=x_values, y=cond_1_df[this_col_label],
                                yerr=cond_1_err_df[this_col_label],
                                linestyle='dashed',
                                marker=None, lw=2, elinewidth=.7,
                                capsize=cap_size,
                                color=my_colours[ax_counter])

                    ax.set_title(each_plot_col[ax_counter])
                    if even_spaced_x:
                        ax.set_xticks(list(range(len(x_tick_labels))))
                    else:
                        ax.set_xticks(x_tick_labels)
                    ax.set_xticklabels(x_tick_labels)

                    if row_idx == 1:
                        ax.set_xlabel(x_label_col)
                    else:
                        ax.xaxis.label.set_visible(False)

                    if col_idx == 0:
                        ax.set_ylabel('Probe Luminance')
                    else:
                        ax.yaxis.label.set_visible(False)

                    # artist for legend
                    st1 = mlines.Line2D([], [], color=my_colours[ax_counter],
                                        linewidth=.5,
                                        markersize=4, label=pos_neg_labels[0])
                    st2 = mlines.Line2D([], [], color=my_colours[ax_counter],
                                        marker=None, linewidth=.5, linestyle="dotted",
                                        markersize=4, label=pos_neg_labels[1])
                    ax.legend(handles=[st1, st2], fontsize=6)

                    ax_counter += 1
                else:
                    fig.delaxes(ax=axes[row_idx, col_idx])

        else:  # if there is only one isi in this row
            print(f'type (NOT AxesSubplot):{type(row)}')

            ax = row
            this_col_label = each_plot_col[row_idx]
            print(f'ax: {ax}; this_col_label: {this_col_label}')
            print(f'x_values: {x_values}')
            check_nan = cond_0_err_df[this_col_label].isnull().values.any()
            print(f'check_nan: {check_nan}')
            if check_nan:
                ax.errorbar(x=x_values, y=cond_0_df[this_col_label],
                            linestyle='dashed',
                            marker=None, lw=2,
                            color=my_colours[row_idx])

                ax.errorbar(x=x_values, y=cond_1_df[this_col_label],
                            marker=None, lw=2,
                            color=my_colours[row_idx])
            else:
                # if NOT nan
                ax.errorbar(x=x_values, y=cond_0_df[this_col_label],
                            yerr=cond_0_err_df[this_col_label],
                            linestyle='dashed',
                            marker=None, lw=2,
                            elinewidth=.7, capsize=cap_size,
                            color=my_colours[row_idx])

                ax.errorbar(x=x_values, y=cond_1_df[this_col_label],
                            yerr=cond_1_err_df[this_col_label],
                            marker=None, lw=2,
                            elinewidth=.7, capsize=cap_size,
                            color=my_colours[row_idx])

            ax.set_title(each_plot_col[row_idx])
            if even_spaced_x:
                ax.set_xticks(list(range(len(x_tick_labels))))
            else:
                ax.set_xticks(x_tick_labels)
            ax.set_xticklabels(x_tick_labels)

            ax.set_xlabel(x_label_col)
            ax.set_ylabel('Probe Luminance')

            # artist for legend
            st1 = mlines.Line2D([], [], color=my_colours[row_idx],
                                linewidth=.5,
                                markersize=4, label=pos_neg_labels[0])
            st2 = mlines.Line2D([], [], color=my_colours[row_idx],
                                marker=None, linewidth=.5, linestyle="dotted",
                                markersize=4, label=pos_neg_labels[1])
            ax.legend(handles=[st1, st2], fontsize=6)

            print(f'ax_counter: {ax_counter}, len(isi_name_list): {len(each_plot_col)}')
            if ax_counter + 1 == len(each_plot_col):
                print(f'idiot check, no more plots to make here')
                break

    plt.tight_layout()

    if save_path is not None:
        if save_name is not None:
            plt.savefig(os.path.join(save_path, save_name))
            print(f'plt saved to: {os.path.join(save_path, save_name)}')

    print("\n*** finished multi_plt_per_col_w_hue() ***")

    return fig


def rad_flow_line_plot(all_df, participant_name,
                       isi_col_name='isi_ms', cong_col_name='congruent',
                       trim_n=None,
                       extra_text=None, save_path=None, show_plots=True,
                       # todo: add title and save_name args
                       title=None, save_name=None,
                       verbose=True):
    """
    Function for doing lineplot of radial flow data showing congruent vs incongruent.
    :param all_df: Datafrrame containing all runs (or runs after trimming)
    :param participant_name: Participant name (or Exp if exp data)
    :param isi_col_name: name of column with ISI conditions.
    :param cong_col_name: name of column with congruent/incongruent conditions.
    :param trim_n: Number of datapoints trimmed if any.
    :param extra_text: Extra text to add to title and save name.
    :param save_path: Path to save plot to.
    :param show_plots: Whether to show plots
    :param title: Title to use (or none for default)
    :param save_name: Save name to use (or none for default)
    :param verbose: Whether to print progress to screen.
    :return:
    """

    # make a lineplot showing congruent and incongruent thresholds for each ISI
    # if all_df is string, check it exists then open it as all_df
    if isinstance(all_df, str):
        if os.path.isfile(all_df):
            all_df = pd.read_csv(all_df)

    # make long_df, moving 'isi_' columns to single column 'isi_ms'
    cols_to_change = [col for col in all_df.columns if 'isi_ms_' in col]
    long_df = make_long_df(wide_df=all_df,
                           cols_to_keep=[cong_col_name],
                           cols_to_change=cols_to_change,
                           cols_to_change_show='probeLum',
                           new_col_name=isi_col_name, strip_from_cols='isi_ms_', verbose=verbose)
    print(f"long_df: {long_df.shape}\ncolumns: {list(long_df.columns)}\n{long_df}\n")

    # make line plot with error bars for congruent and incongruent with isi on x axis
    # use the basic palette
    sns.pointplot(data=long_df, x=isi_col_name, y='probeLum', hue=cong_col_name,
                 errorbar='se',
                 palette=sns.color_palette("tab10", n_colors=2),
                  dodge=.1)  # float al;owed for dodge here
    # # add small scatterplot points with same colours, with .5 alpha/opacity
    # todo: manually edit position on x axis to match point plot?    OR better still..
    # todo: split hue values and have faint lines joining them
    # add small scatterplot points with same colours, with .5 alpha/opacity
    sns.stripplot(data=long_df, x=isi_col_name, y='probeLum', hue=cong_col_name,
                  palette=sns.color_palette("tab10", n_colors=2),
                  legend=False,
                  alpha=.5,
                  dodge=True,  # boolean only
                  jitter=False
                  )

    # change legend labels such that they are 1=congruent and -1=incongruent
    handles, labels = plt.gca().get_legend_handles_labels()
    new_labels = []
    # todo: add code for updating title or save_name

    for label in labels:
        if label == '1':
            new_labels.append('Congruent')
        elif label == '-1':
            new_labels.append('Incongruent')
        else:
            new_labels.append(label)
    # make legend box 50% opaque
    plt.legend(title='Probe & background', handles=handles, labels=new_labels,
               framealpha=0.5)

    # for x-axis labels, if the isi is -1, change to 'Concurrent'
    x_tick_values = sorted(long_df[isi_col_name].unique())
    x_tick_labels = []
    for label in x_tick_values:
        if label in [-1, '-1']:
            x_tick_labels.append('Concurrent')
        else:
            x_tick_labels.append(label)

    # decorate plot
    plt.xlabel('ISI (ms)')
    plt.ylabel('Probe luminance')
    plt.xticks(ticks=x_tick_values, labels=x_tick_labels)

    suptitle_text = f"{participant_name} thresholds for each ISI. {extra_text}"
    if trim_n is not None:
        suptitle_text = f"{participant_name} thresholds for each ISI, trimmed {trim_n}. {extra_text}"
    plt.suptitle(suptitle_text)
    plt.title("separation = 4; motion window = 200ms")
    # if not save_path:
    # save_path, df_name = os.path.split(all_df_path)
    # plt.savefig(os.path.join(save_path, f"{df_name[:-4]}_lineplot.png"))
    fig_name = f"{participant_name}_lineplot.png"
    if extra_text is not None:
        fig_name = f"{participant_name}_{extra_text}_lineplot.png"
    if verbose:
        print(f"\n***saving lineplot to {os.path.join(save_path, fig_name)}***")
    plt.savefig(os.path.join(save_path, fig_name))
    if show_plots:
        plt.show()


def plot_thr_heatmap(heatmap_df,
                     x_tick_labels=None,
                     x_axis_label=None,
                     y_tick_labels=None,
                     y_axis_label=None,
                     heatmap_midpoint=None,
                     fig_title=None,
                     save_name=None,
                     save_path=None,
                     verbose=True):
    """
    Function for making a heatmap
    :param heatmap_df: Expects dataframe with separation as index and ISIs as columns.
    :param x_tick_labels: Labels for columns
    :param y_tick_labels: Labels for rows
    :param fig_title: Title for figure
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


    if heatmap_midpoint is None:
        heatmap_midpoint = mean_thr
        colourmap = sns.color_palette("Spectral", as_cmap=True)
    else:
        colourmap = sns.color_palette("bwr", as_cmap=True)
    print(f"heatmap_midpoint: {heatmap_midpoint}")

    heatmap = sns.heatmap(data=heatmap_df,
                          annot=True,
                          fmt='.3g', # format to 3 significant figures
                          center=heatmap_midpoint,
                          cmap=colourmap,
                          xticklabels=x_tick_labels, yticklabels=y_tick_labels)

    # keep y ticks upright rather than rotates (90)
    plt.yticks(rotation=0)

    # add central mirror symmetry line
    # plt.axvline(x=6, color='grey', linestyle='dashed')

    if x_axis_label is None:
        if 'ISI' in str(x_tick_labels[0]).upper():
            heatmap.set_xlabel('ISI')
            heatmap.set_ylabel('Separation')
        else:
            heatmap.set_xlabel('Separation')
            heatmap.set_ylabel('ISI')
    else:
        heatmap.set_xlabel(x_axis_label)
        heatmap.set_ylabel(y_axis_label)

    if fig_title is not None:
        plt.title(fig_title)

    if save_path is not None:
        if save_name is not None:
            plt.savefig(os.path.join(save_path, save_name))
            print(f'plt saved to: {os.path.join(save_path, save_name)}')
    print('\n*** finished plot_thr_heatmap() ***\n')


    return heatmap


def compare_prelim_plots(p_name, exp_path):
    """
    This function will allow me to plot the different background and prelim conditions for comparssion.
    It assumes that there is a {p_name}_ALLbg_ave_thresh.csv in a 'compare_prelims' dir within the p_name dir.
    It will save plots into the compare_prelims dir.

    :param p_name: Participant name
    :param exp_path: Path to the experiment dir (participant dir is within this)

    """

    '''1. find the data'''
    p_name_dir = os.path.join(exp_path, p_name)
    if not os.path.exists(p_name_dir):
        raise ValueError(f"\n\n\tp_name_dir does not exist: {p_name_dir}")

    print(f"\n\np_name: {p_name}\np_name_dir:{p_name_dir}")

    # look for compare_prelims dir
    if 'compare_prelims' in os.listdir(p_name_dir):
        save_path = os.path.join(p_name_dir, 'compare_prelims')
        print(f"save_path: {save_path}")
    else:
        raise ValueError(f"\n\n\tcompare_prelims dir does not exist.\np_name_dir: {p_name_dir}\ncontents: {os.listdir(p_name_dir)}")

    # # load _ALLbg_thresholds.csv only - look for trimmed data first, then untrimmed.
    p_master_all_name = f'{p_name}_TM2_ALLbg_thresholds.csv'
    p_master_all_path = os.path.join(save_path, p_master_all_name)
    if not os.path.exists(p_master_all_path):
        p_master_all_name = f'{p_name}_TM1_ALLbg_thresholds.csv'
        p_master_all_path = os.path.join(save_path, p_master_all_name)
        if not os.path.exists(p_master_all_path):
            p_master_all_name = f'{p_name}_ALLbg_thresholds.csv'
            p_master_all_path = os.path.join(save_path, p_master_all_name)
            if not os.path.exists(p_master_all_path):
                raise ValueError(f"\n\n\tp_master_all_path does not exist: {p_master_all_path}")
    p_master_all_df = pd.read_csv(os.path.join(save_path, p_master_all_name))
    print(f"\n\np_master_all_df:\n{p_master_all_df}")

    '''I only need this if I need the average and error dfs'''
    # # look for ALLbg dfs, start by assuming they have been trimmed, then try other options
    # trim_n = 2
    # p_master_all_name = f'{p_name}_TM{trim_n}_ALLbg_ave_thresh.csv'
    # p_master_ave_path = os.path.join(save_path, p_master_all_name)
    # if not os.path.exists(p_master_ave_path):
    #     print('trying trim_n = 1')
    #     trim_n = 1
    #     p_master_all_name = f'{p_name}_TM{trim_n}_ALLbg_thresholds.csv'
    #     p_master_ave_path = os.path.join(save_path, p_master_all_name)
    #     if not os.path.exists(p_master_ave_path):
    #         trim_n = None
    #         print('trying trim_n = None')
    #         p_master_all_name = f'{p_name}_ALLbg_thresholds.csv'
    #         p_master_ave_path = os.path.join(save_path, p_master_all_name)
    #         if not os.path.exists(p_master_ave_path):
    #             raise ValueError(f"\n\n\tp_master_ave_path does not exist: {p_master_ave_path}")
    #
    # # # load the ave, error and all threshold dfs
    # if trim_n is not None:
    #     p_master_ave_df = pd.read_csv(os.path.join(save_path, f'{p_name}_TM{trim_n}_ALLbg_ave_thresh.csv'))
    #     p_master_all_df = pd.read_csv(os.path.join(save_path, f'{p_name}_TM{trim_n}_ALLbg_thresholds.csv'))
    #     p_master_err_df = pd.read_csv(os.path.join(save_path, f'{p_name}_TM{trim_n}_ALLbg_thr_error_SE.csv'))
    # else:
    #     p_master_ave_df = pd.read_csv(os.path.join(save_path, f'{p_name}_ALLbg_ave_thresh.csv'))
    #     p_master_all_df = pd.read_csv(os.path.join(save_path, f'{p_name}_ALLbg_thresholds.csv'))
    #     p_master_err_df = pd.read_csv(os.path.join(save_path, f'{p_name}_ALLbg_thr_error_SE.csv'))
    #
    # print(f"trim_n: {trim_n}")
    # # print(f"\n\np_master_ave_df:\n{p_master_ave_df}")
    # print(f"\n\np_master_all_df:\n{p_master_all_df}")
    # # print(f"\n\np_master_err_df:\n{p_master_err_df}")


    # set the df to be 'all' so that I can have error bars
    use_this_df = p_master_all_df.copy()

    '''2. prepare the data'''
    # drop stair_names column as it is redundant
    use_this_df.drop(columns=['stair_names'], inplace=True)

    # insert cond_type column with 'Congruent' if neg_sep >= 0 and 'Incongruent' if neg_sep < 0
    cond_type_list = ['Incongruent' if neg_sep < 0 else 'Congruent' for neg_sep in use_this_df['neg_sep']]
    use_this_df.insert(1, 'cond_type', cond_type_list)
    print(f"\nuse_this_df:\n{use_this_df}")

    # change separation column to int
    use_this_df['separation'] = use_this_df['separation'].astype(int)

    # make a list of all ISIs conds (all_isi_list), strip str, sort, then add string back
    all_isi_list = [col for col in use_this_df.columns if 'ISI_' in col]
    stripped_list = sorted([int(x.strip('ISI_')) for x in all_isi_list])
    isi_list = ['ISI_' + str(x) for x in stripped_list]
    print(f"\nall_isi_list:\n{all_isi_list}")


    # make my_colur_dict so that prelims are always in same colour even if some conditions are missing
    # use different colours for plots by prelim or by cond_type
    my_prelim_colours = fig_colours(n_conditions=len(use_this_df['prelim_ms'].unique().tolist()))
    my_prelim_col_dict = {prelim: my_prelim_colours[i] for i, prelim in enumerate(use_this_df['prelim_ms'].unique().tolist())}
    my_cond_colours = fig_colours(n_conditions=len(use_this_df['cond_type'].unique().tolist()), alternative_colours=True)
    my_cond_col_dict = {cond: my_cond_colours[i] for i, cond in enumerate(use_this_df['cond_type'].unique().tolist())}
    my_colour_dict = {**my_prelim_col_dict, **my_cond_col_dict}
    print(f"\nmy_colour_dict:\n{my_colour_dict}")

    # get list of all background types
    background_list = use_this_df['background'].unique().tolist()

    '''3. loop through conditions (background and ISIs) and plots'''

    for this_background in background_list:

        background_df = use_this_df[use_this_df['background'] == this_background].copy()

        # drop background column
        background_df.drop(columns=['background'], inplace=True)

        for this_isi in all_isi_list:

            this_isi_df = background_df[['prelim_ms', 'cond_type', 'neg_sep', 'separation', this_isi]].copy()

            # drop rows where this_isi is NaN
            this_isi_df.dropna(subset=[this_isi], inplace=True)
            print(f"\n\nthis_isi_df: ({this_isi}):\n{this_isi_df}")

            # sort prelim_list (strip 'ms', convert to ints, sort then put them back to str with ms)
            prelim_list = sorted(list(set(this_isi_df['prelim_ms'].tolist())))

            if 'bg' in prelim_list[0]:
                # strip 'bg' for each element, convert to int, sort, then add 'bg' back
                prelim_list = sorted([int(prelim.strip('bg')) for prelim in prelim_list])
                prelim_list = ['bg' + str(prelim) for prelim in prelim_list]

            print(f"\nprelim_list (n={len(prelim_list)}): {prelim_list}")


            '''4. Batman plots - all prelim durations on same panel'''
            # 1. all patman plots on same panel
            # get the x_tick_labels and x_tick_values (neg_sep, -18 to +18)
            x_tick_labels = this_isi_df['neg_sep'].unique().tolist()
            x_tick_labels = sorted([i for i in x_tick_labels])
            x_tick_labels = [str(i) for i in x_tick_labels]
            x_tick_values = list(range(len(x_tick_labels)))
            print(f"\nx_tick_labels:\n{x_tick_labels}")
            print(f"\nx_tick_values:\n{x_tick_values}")

            # make copy of this_isi_df and add new column for x_tick_values, mapped onto neg_sep values
            batman_plot_df = this_isi_df.copy()
            tick_dict = dict(zip(x_tick_labels, x_tick_values))
            print(f"\ntick_dict:\n{tick_dict}")
            tick_vals_col_list = [tick_dict[str(neg_sep)] for neg_sep in batman_plot_df['neg_sep']]
            print(f"\ntick_vals_col_list:\n{tick_vals_col_list}")
            batman_plot_df.insert(1, 'tick_vals', tick_vals_col_list)
            print(f"\nbatman_plot_df:\n{batman_plot_df}")

            # sort final values for x_tick_labels
            x_tick_labels = [str('-0') if i == .01 else int(float(i)) for i in x_tick_labels]

            fig, ax = plt.subplots()
            sns.lineplot(x='tick_vals', y=this_isi, hue='prelim_ms', data=batman_plot_df,
                         palette=my_prelim_col_dict,
                         err_style='bars', errorbar='se', err_kws={'capsize': 5})
            ax.set_xticks(x_tick_values)
            ax.set_xticklabels(x_tick_labels)
            plt.axvline(x=max(x_tick_values)/2, color='lightgrey', linestyle='dashed')
            plt.title(f"{p_name}\n{this_background} {this_isi} prelim motion with neg_sep")
            plt.savefig(os.path.join(save_path, f"{this_background}_{p_name}_{this_isi}_neg_sep.png"))
            plt.show()
            plt.close()


            '''5. batman plots in separate panels if there is more than 1 prelim dur'''
            if len(prelim_list) > 1:

                fig, axes = plt.subplots(nrows=1, ncols=len(prelim_list),
                                         figsize=(len(prelim_list)*5, 5))
                ax_counter = 0

                for row_idx, ax in enumerate(axes):
                    sns.lineplot(x='tick_vals', y=this_isi, hue='cond_type',
                                 data=batman_plot_df[batman_plot_df['prelim_ms'] == prelim_list[row_idx]],
                                 ax=axes[row_idx], palette=my_cond_col_dict,
                                 err_style='bars', errorbar='se', err_kws={'capsize': 5})
                    ax.set_xticks(x_tick_values)
                    ax.set_xticklabels(x_tick_labels)
                    ax.axvline(x=max(x_tick_values)/2, color='lightgrey', linestyle='dashed')
                    ax.set_title(f"{prelim_list[row_idx]}")
                    ax.set_xlabel('Separation')
                    ax.set_ylabel('Threshold')

                    # suppress legend for individal panels, but put one in at the end
                    if ax_counter == 0:
                        ax.legend(loc='upper left')
                    else:
                        try:
                            ax.get_legend().remove()
                        except AttributeError:
                            print(f"\n\n\tax.get_legend().remove() failed")
                    ax_counter += 1

                plt.suptitle(f"{p_name}\n{this_background} {this_isi} prelim motion with neg_sep panels")
                plt.savefig(os.path.join(save_path, f"{this_background}_{p_name}_{this_isi}_neg_sep_panel.png"))
                plt.show()
                plt.close()


            '''6. different between conditions plots'''
            # plots showing the difference between congruent and incongruent conditions for each prelim, ISI and neg_sep
            # split into two dataframes, one for congruent and one for incongruent
            print(f"\nthis_isi_df:\n{this_isi_df}")

            this_isi_df = this_isi_df[['prelim_ms', 'cond_type', 'separation', this_isi]].copy()
            print(f"\nthis_isi_df:\n{this_isi_df}")
            congruent_df = this_isi_df[this_isi_df['cond_type'] == 'Congruent'].copy()
            incongruent_df = this_isi_df[this_isi_df['cond_type'] == 'Incongruent'].copy()

            # rename this_isi column to f'{this_isi}_cong' and f'{this_isi}_incong' for congruent_df and incongruent_df
            congruent_df.rename(columns={this_isi: f'{this_isi}_cong'}, inplace=True)
            incongruent_df.rename(columns={this_isi: f'{this_isi}_incong'}, inplace=True)

            # drop cond_type column from both dataframes
            congruent_df.drop(columns=['cond_type'], inplace=True)
            incongruent_df.drop(columns=['cond_type'], inplace=True)

            # set index to prelim and separation
            congruent_df.set_index(['prelim_ms', 'separation'], inplace=True)
            incongruent_df.set_index(['prelim_ms', 'separation'], inplace=True)

            print(f"\ncongruent_df:\n{congruent_df}")
            print(f"\nincongruent_df:\n{incongruent_df}")

            # join congruent_df and incongruent_df
            diff_df = congruent_df.join(incongruent_df)
            print(f"\ndiff_df:\n{diff_df}")

            # add column for difference between congruent and incongruent
            diff_df[f'{this_isi}_diff'] = diff_df[f'{this_isi}_cong'] - diff_df[f'{this_isi}_incong']

            # reset index
            diff_df.reset_index(inplace=True)

            # add x tick vals col
            # change separation values to ints
            diff_df['separation'] = diff_df['separation'].astype(int)
            x_tick_labels = sorted(diff_df['separation'].unique())
            x_tick_values = list(range(len(x_tick_labels)))
            tick_dict = dict(zip(x_tick_labels, x_tick_values))
            tick_vals_col_list = [tick_dict[sep] for sep in diff_df['separation']]
            diff_df.insert(1, 'tick_vals', tick_vals_col_list)
            print(f"\ndiff_df:\n{diff_df}")


            # plot difference between congruent and incongruent conditions
            sns.lineplot(x='tick_vals', y=f'{this_isi}_diff', hue='prelim_ms', data=diff_df,
                         palette=my_prelim_col_dict,
                         err_style='bars', errorbar='se', err_kws={'capsize': 5})

            # add horizontal line at 0
            plt.axhline(y=0, color='lightgrey', linestyle='dashed')
            plt.xticks(x_tick_values, x_tick_labels)
            plt.title(f"{p_name}\n{this_background} {this_isi} congruent - incongruent difference")
            plt.xlabel('Separation')
            plt.savefig(os.path.join(save_path, f"{this_background}_{p_name}_{this_isi}_diff.png"))
            plt.show()
            plt.close()

    print(f"\n\n\t***compare_prelim_plots() completed for {p_name}***")


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



def a_data_extraction(p_name, run_dir, isi_list, save_all_data=True, verbose=True):
    """
    This script is a python version of Martin's first MATLAB analysis scripts, described below.

    a_data_extraction.m: Once a participant has completed a run of all ISIs,
        this script gets all their data into one file, and sorts each isi by stair.

    :param p_name: participant's name as used to save csv files.  e.g., if the
            file is .../nick1.csv, participant name is 'nick1'.
    :param run_dir: directory where isi folders are stored.
    :param isi_list: List of isi values, may differ between experiments.
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

    if isi_list is None:
        raise ValueError('Please pass a list of isi values to identify directories containing data.')
    else:
        print(f'isi_list: {isi_list}')

    all_data_list = []

    # loop through ISIs in each run.
    for isi in isi_list:
        filepath = f'{run_dir}{os.path.sep}ISI_{isi}_probeDur2{os.path.sep}' \
                   f'{p_name}.csv'
        if verbose:
            print(f"filepath: {filepath}")

        if not os.path.isfile(filepath):
            filepath = f'{run_dir}{os.path.sep}ISI_{isi}{os.path.sep}' \
                       f'{p_name}.csv'

            if not os.path.isfile(filepath):
                filepath = f'{run_dir}{os.path.sep}ISI_{isi}{os.path.sep}' \
                           f'{p_name}_output.csv'

                if not os.path.isfile(filepath):  # try searching through sep folders
                    filepath = f'{run_dir}{os.path.sep}sep_{isi}{os.path.sep}' \
                               f'{p_name}_output.csv'

                    if not os.path.isfile(filepath):
                        # raise FileNotFoundError(filepath)
                        print(f"File not found: {filepath}")
                        continue

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
        if 'ISI' not in list(this_isi_df):
            this_isi_df.insert(0, 'ISI', isi)
        this_isi_df.insert(1, 'srtd_trial_idx', trial_numbers)
        if verbose:
            print(f'df sorted by stair: {type(this_isi_df)}\n{this_isi_df}')

        # get column names to use on all_data_df
        column_names = list(this_isi_df)
        if verbose:
            print(f'column_names: {len(column_names)}\n{column_names}')

        # I've changed column names lately, so there are some extra ones.  In which case, just use old cols.
        if 'actual_bg_color' in column_names:
            print("getting rid of extra columns (e.g., 'actual_bg_color', "
                  "'bgcolor_to_rgb1', 'bgLumP', 'bgLum', 'bgColor255')")
            cols_to_use = ['ISI', 'srtd_trial_idx', 'trial_number', 'stair',
                           'stair_name', 'step', 'separation', 'congruent',
                           'flow_dir', 'probe_jump', 'corner', 'probeLum',
                           'trial_response', 'resp.rt', 'probeColor1', 'probeColor255',
                           'probe_ecc', 'BGspeed', 'orientation', 'ISI_actual_ms',
                           '1_Participant', '2_Probe_dur_in_frames_at_240hz',
                           '3_fps', '4_ISI_dur_in_ms', '5_Probe_orientation',
                           '6_Probe_size', '7_Trials_counter', '8_Background',
                           'date', 'time', 'stair_list', 'n_trials_per_stair']
            this_isi_df = this_isi_df[cols_to_use]
            column_names = cols_to_use

        # add to all_data_list
        all_data_list.append(this_isi_df)

    # create all_data_df - reshape to 2d
    # all_data = np.ndarray(all_data_list, dtype=object)
    # all_data_df = pd.concat(all_data_list)
    # print(f'all_data_df: {type(all_data_df)}, {all_data_df.shape}\n{all_data_df}')

    # all_data = np.ndarray(all_data_df, dtype=object)

    all_data_df = pd.concat(all_data_list)
    print(f"all_data_df:\n{all_data_df}")

    # all_data = all_data_list
    # if verbose:
    #     print(f'all_data: {type(all_data)}\n{all_data}')
    # all_data_shape = np.shape(all_data)
    # print(f'all_data_shape:  {all_data_shape}')
    #
    # sheets, rows, columns = np.shape(all_data)
    #
    # all_data = np.reshape(all_data, newshape=(sheets * rows, columns))
    # if verbose:
    #     print(f'all_data reshaped from {all_data_shape} to {np.shape(all_data)}')
    # all_data_df = pd.DataFrame(all_data, columns=column_names)

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


def b3_plot_staircase(all_data_path, thr_col='newLum', resp_col='trial_response',
                      show_plots=True, save_plots=True, verbose=True):
    """
    b3_plot_staircase: staircases-ISIxxx.png: xxx corresponds to isi conditions.
    One plot for each isi condition.  Each figure has six panels (6 probes separation
    conditions) showing the Luminance value of two staircases as function of
    trial number. Last panel shows last thr per sep condition.

    :param all_data_path: path to the all_data xlsx file.
    :param thr_col: (default newLum) name of the column showing the threshold
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
    if xlsx_name[-3:] == 'csv':
        all_data_df = pd.read_csv(all_data_path)
    else:
        all_data_df = pd.read_excel(all_data_path, engine='openpyxl',
                                    # usecols=["ISI", "stair", "step", "separation",
                                    #          "flow_dir", "probe_jump", "corner",
                                    #          "newLum", "probeLum", "trial_response"]
                                    # note: don't use usecols as some exps have 'NewLum' but radflow doesn't
                                    )

    # get list of isi and stair values to loop through
    stair_list = all_data_df['stair'].unique()
    isi_list = all_data_df['ISI'].unique()
    # get isi string for column names
    isi_name_list = [f'ISI_{i}' for i in isi_list]

    trials, columns = np.shape(all_data_df)
    trials_per_stair = int(trials / len(isi_list) / len(stair_list))

    if verbose:
        print(f"all_data_df:\n{all_data_df}")
        print(f"{len(isi_list)} isi values and {len(stair_list)} stair values")
        print(f"isi_list: {isi_list}")
        print(f"isi_name_list: {isi_name_list}")
        print(f"stair_list: {stair_list}")
        print(f"trials_per_stair: {trials_per_stair}")

    '''the eighth plot is the last thr for each sep (+sep, -sep and mean).  
    get data from psignifit_thresholds.csv and reshape here'''
    thr_csv_name = os.path.join(save_path, 'psignifit_thresholds.csv')
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
    if len(sep_list) == rows * 2:
        # takes every other item
        sep_list = sep_list[::2]
    else:
        # todo: for exp2_bloch I don't have double the number of rows, so I should do something here.
        raise ValueError(f"I dunno why the number of rows ({rows}) isn't double "
                         f"the sep_list ({sep_list})")
    separation_title = [f'sep_{i}' for i in sep_list]
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
        print(f"\n{isi_idx}. staircase for ISI: {isi}, {isi_name}")

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
                    stair_even_cong = ax_counter * 2  # 0, 2, 4, 6, 8, 10
                    stair_even_cong_df = isi_df[isi_df['stair'] == stair_even_cong]
                    final_lum_even_cong = \
                        stair_even_cong_df.loc[stair_even_cong_df['step'] == trials_per_stair - 1, thr_col].item()
                    n_reversals_even_cong = trials_per_stair - stair_even_cong_df[resp_col].sum()

                    stair_odd_incong = (ax_counter * 2) + 1  # 1, 3, 5, 7, 9, 11
                    stair_odd_incong_df = isi_df[isi_df['stair'] == stair_odd_incong]
                    final_lum_odd_incong = \
                        stair_odd_incong_df.loc[stair_odd_incong_df['step'] == trials_per_stair - 1, thr_col].item()
                    n_reversals_odd_incong = trials_per_stair - stair_odd_incong_df[resp_col].sum()

                    # append n_reversals to n_reversals_np to save later.
                    n_reversals_np[stair_even_cong - 1, isi_idx] = n_reversals_even_cong
                    n_reversals_np[stair_odd_incong - 1, isi_idx] = n_reversals_odd_incong

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
                    # line for final newLum
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
                        st1 = mlines.Line2D([], [], color='tab:blue',
                                            marker='v',
                                            markersize=5, label='Congruent')
                        st1_last_val = mlines.Line2D([], [], color='tab:blue',
                                                     linestyle="--", marker=None,
                                                     label='Cong: last val')
                        st2 = mlines.Line2D([], [], color='tab:red',
                                            marker='o',
                                            markersize=5, label='Incongruent')
                        st2_last_val = mlines.Line2D([], [], color='tab:red',
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
                                 x='separation', y=isi_name, color='blue',
                                 linestyle="--")
                    sns.lineplot(ax=axes[row_idx, col_idx], data=psig_incong_sep_df,
                                 x='separation', y=isi_name, color='red',
                                 linestyle="dotted")

                    # artist for legend
                    cong_thr = mlines.Line2D([], [], color='blue', linestyle="--",
                                             marker=None, label='Congruent thr')

                    incong_thr = mlines.Line2D([], [], color='red', linestyle="dotted",
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
            save_name = f'staircases_{isi_name}.png'
            plt.savefig(os.path.join(save_path, save_name))
            
        if show_plots:
            plt.show()
        plt.close()

    # save n_reversals to csv for use in script_c figure 5
    n_reversals_df = pd.DataFrame(n_reversals_np, columns=isi_name_list)
    n_reversals_df.insert(0, 'stair', stair_list)
    n_reversals_df.set_index('stair', inplace=True)
    if verbose:
        print(f'n_reversals_df:\n{n_reversals_df}')
    n_reversals_df.to_csv(os.path.join(save_path, 'n_reversals.csv'))

    print("\n***finished b3_plot_staircases()***\n")


def b3_plot_stair_sep0(all_data_path, thr_col='newLum', resp_col='trial_response',
                       show_plots=True, save_plots=True, verbose=True):
    """
    b3_plot_staircase: staircases-ISIxxx.png: xxx corresponds to isi conditions.
    One plot for each isi condition.  Each figure has six panels (6 probes separation
    conditions) showing the Luminance value of two staircases as function of
    trial number. Last panel shows last thr per sep condition.

    :param all_data_path: path to the all_data xlsx file.
    :param thr_col: (default newLum) name of the column showing the threshold
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
    print("\n*** running b3_plot_stair_sep0() ***\n")

    save_path, xlsx_name = os.path.split(all_data_path)

    # open all_data file.  use engine='openpyxl' for xlsx files.
    # For other experiments it might be easier not to do use cols as they might be different.
    if xlsx_name[-3:] == 'csv':
        all_data_df = pd.read_csv(all_data_path)
    else:
        all_data_df = pd.read_excel(all_data_path, engine='openpyxl',
                                    usecols=["ISI", "stair", "step", "separation",
                                             "flow_dir", "probe_jump", "corner",
                                             "newLum", "probeLum", "trial_response"])

    # get list of isi and stair values to loop through
    stair_list = all_data_df['stair'].unique()
    isi_list = all_data_df['ISI'].unique()
    # get isi string for column names
    isi_name_list = [f'ISI_{i}' for i in isi_list]

    trials, columns = np.shape(all_data_df)
    # trials_per_stair = int(trials/len(isi_list)/len(stair_list))
    trials_per_stair = int(trials / len(stair_list))

    if verbose:
        print(f"all_data_df:\n{all_data_df}")
        print(f"{len(isi_list)} isi values and {len(stair_list)} stair values")
        print(f"isi_list: {isi_list}")
        print(f"isi_name_list: {isi_name_list}")
        print(f"stair_list: {stair_list}")
        print(f"trials_per_stair: {trials_per_stair}")

    '''the eighth plot is the last thr for each sep (+sep, -sep and mean).  
    get data from psignifit_thresholds.csv and reshape here'''
    thr_csv_name = os.path.join(save_path, 'psignifit_thresholds.csv')
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

    psignifit_thr_df.columns = isi_name_list

    separation_title = [f'sep_{i}' for i in sep_list]
    if verbose:
        print(f'sep_list: {sep_list}')
        print(f"separation_title: {separation_title}")

    psignifit_thr_df.insert(0, 'separation', sep_list)
    if verbose:
        print(f'\npsignifit_thr_df:\n{psignifit_thr_df}\n')

    # make empty arrays to save reversal n_reversals
    # n_reversals_np = np.zeros(shape=[len(stair_list), len(isi_list)])
    n_reversals_np = np.zeros(shape=[len(stair_list)])

    # initialise 8 plot figure - last plot will be blank
    # # this is a figure showing n_reversals per staircase condition.
    # n_rows, n_cols = multi_plot_shape(len(isi_name_list), min_rows=2)
    n_rows, n_cols = get_n_rows_n_cols((len(isi_name_list)))

    print(f"making {len(isi_name_list)} plots, rows={n_rows}, cols={n_cols}")

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12, 6))
    ax_counter = 0

    for row_idx, row in enumerate(axes):
        for col_idx, ax in enumerate(row):
            print(f'\nrow: {row_idx}, col: {col_idx}: {ax}')

            # for the first six plots...
            if ax_counter < len(isi_name_list):

                isi_name = isi_name_list[ax_counter]

                # # get pairs of stairs (e.g., [[18, -18], [6, -6], ...etc)
                this_stair_df = all_data_df[all_data_df['stair'] == ax_counter]
                print(f"this_stair_df:\n{this_stair_df}")

                final_lum_val = \
                    this_stair_df.loc[this_stair_df['step'] == trials_per_stair - 1, thr_col].item()
                n_reversals = trials_per_stair - this_stair_df[resp_col].sum()

                # append n_reversals to n_reversals_np to save later.
                n_reversals_np[ax_counter - 1] = n_reversals

                if verbose:
                    print(f'\nthis_stair_df (stair={ax_counter}, '
                          f'isi_name={isi_name}:\n{this_stair_df}')
                    print(f"final_lum_val: {final_lum_val}")
                    print(f"n_reversals: {n_reversals}")

                fig.suptitle(f'Staircases and reversals')

                # plot thr per step for even_cong numbered stair
                sns.lineplot(ax=axes[row_idx, col_idx], data=this_stair_df,
                             x='step', y=thr_col, color='tab:blue',
                             marker="o", markersize=4)
                # line for final newLum
                ax.axhline(y=final_lum_val, linestyle="-.", color='tab:blue')
                # text for n_reversals
                ax.text(x=0.25, y=0.8, s=f'{n_reversals} reversals',
                        color='tab:blue',
                        # needs transform to appear with rest of plot.
                        transform=ax.transAxes, fontsize=12)

                ax.set_title(isi_name)
                ax.set_xticks(np.arange(0, trials_per_stair, 5))
                ax.set_ylim([0, 110])

                # artist for legend
                if ax_counter == 0:
                    st1 = mlines.Line2D([], [], color='tab:blue',
                                        marker='v',
                                        markersize=5, label='Data')
                    st1_last_val = mlines.Line2D([], [], color='tab:blue',
                                                 linestyle="--", marker=None,
                                                 label='last val')
                    ax.legend(handles=[st1, st1_last_val],
                              fontsize=6, loc='lower right')

            else:
                fig.delaxes(ax=axes[row_idx, col_idx])

            ax_counter += 1

    plt.tight_layout()

    # show and close plots
    if save_plots:
        # save_name = f'staircases_{isi_name}.png'
        save_name = 'staircases.png'
        plt.savefig(os.path.join(save_path, save_name))

    if show_plots:
        plt.show()
    plt.close()

    # save n_reversals to csv for use in script_c figure 5
    print(f'n_reversals_np:\n{n_reversals_np}')

    n_reversals_df = pd.DataFrame(n_reversals_np)
    n_reversals_df.insert(0, 'del', isi_name_list)
    # n_reversals_df.insert(0, 'stair', stair_list)
    # n_reversals_df.set_index('stair', inplace=True)
    n_reversals_df = n_reversals_df.T
    n_reversals_df.columns = isi_name_list
    n_reversals_df.drop('del', inplace=True)
    if verbose:
        print(f'n_reversals_df:\n{n_reversals_df}')
    n_reversals_df.to_csv(os.path.join(save_path, 'n_reversals.csv'))

    print("\n***finished b3_plot_stair_sep0()***\n")


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




def c_plots(save_path, thr_col='newLum', isi_name_list=None, show_plots=True, verbose=True):
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
    :param thr_col: Name of column containing threshold values.
    :param isi_name_list: Default None: can input a list of names of ISIs,
            useful if I only have data for a few ISI values.
    :param show_plots: Default True
    :param verbose: Default True.
    """
    print("\n*** running c_plots() ***\n")

    # load df mean of last n newLum values (14 stairs x 8 isi).
    thr_csv_name = os.path.join(save_path, 'psignifit_thresholds.csv')

    # this psig_thr_df is in stair order (e.g., stairs 0, 1, 2, 3 == sep: 18, -18, 6, -6 etc)
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

    psig_thr_df.columns = ['stair_names'] + isi_name_list
    if verbose:
        # psig_thr_df still in stair order  (e.g., sep: 18, -18, 6, -6 etc)
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
    diff_val.drop(index='stair_names', inplace=True)

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

    make_multi_batman_plot = False
    if len(sep_col_s) > 1:
        if len(isi_name_list) > 1:
            make_multi_batman_plot = True

    if make_multi_batman_plot:
        print("running multi_batman_plots ")

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
                           verbose=verbose)
        if show_plots:
            plt.show()
        plt.close()
    else:
        print("NOT running multi_batman_plots - not enough sep or ISI values")

    #  (figure2 doesn't exist in Martin's script - but I'll keep their numbers)

    # # FIGURE3 - 'data.png' - all ISIs on same axis, pos and neg sep, looks like batman.
    # # use plot_data_unsym_batman()
    fig3_save_name = f'data_pos_and_neg.png'
    fig_3_title = 'All ISIs and separations\n' \
                  '(positive values for congruent probe/flow motion, ' \
                  'negative for incongruent).'

    psig_sorted_df = psig_thr_df.sort_values(by=['stair_names'])
    psig_sorted_df.drop('stair_names', axis=1, inplace=True)
    psig_sorted_df.reset_index(drop=True, inplace=True)
    psig_thr_idx_list = list(psig_sorted_df.index)
    stair_names_list = sorted(stair_names_list)
    stair_names_list = ['-0' if i == -.10 else int(i) for i in stair_names_list]

    if verbose:
        # psig_sorted_df sorted by stair_NAMES (e.g., sep: -18, -6, -3 etc)
        print(f'\npsig_sorted_df:\n{psig_sorted_df}')
        print(f'\npsig_thr_idx_list: {psig_thr_idx_list}')
        print(f'\nstair_names_list: {stair_names_list}')

    plot_data_unsym_batman(pos_and_neg_sep_df=psig_sorted_df,
                           fig_title=fig_3_title,
                           save_path=save_path,
                           save_name=fig3_save_name,
                           isi_name_list=isi_name_list,
                           x_tick_values=psig_thr_idx_list,
                           x_tick_labels=stair_names_list,
                           verbose=verbose)
    if show_plots:
        plt.show()
    plt.close()

    #########
    # fig to compare congruent and incongruent thr for each ISI
    # The remaining plots go back to using psig_thr_df, in stair order (not stair_names)
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

    plot_w_errors_either_x_axis(wide_df=psig_thr_df, cols_to_keep=['congruent', 'separation'],
                                cols_to_change=isi_name_list,
                                cols_to_change_show=thr_col, new_col_name='ISI',
                                strip_from_cols='ISI_',
                                x_axis='separation', y_axis=thr_col, x_tick_vals=[0, 1, 2, 3, 6, 18],
                                hue_var='ISI', style_var='congruent', style_order=[1, -1],
                                error_bars=True, even_spaced_x=True, jitter=False,
                                fig_title=fig4_title, fig_savename=fig4_savename,
                                save_path=save_path, verbose=verbose)
    if show_plots:
        plt.show()
    plt.close()

    if verbose:
        print('\nfig5 single run data')

    x_tick_vals = [int(i[4:]) for i in isi_cols_list]
    print(f'x_tick_vals: {x_tick_vals}')
    fig5_title = 'Congruent and incongruent probe/flow motion for each separation'
    fig5_savename = 'compare_data_sep'

    plot_w_errors_either_x_axis(wide_df=psig_thr_df, cols_to_keep=['congruent', 'separation'],
                                cols_to_change=isi_name_list,
                                cols_to_change_show=thr_col, new_col_name='ISI',
                                strip_from_cols='ISI_',
                                x_axis='ISI', y_axis=thr_col,
                                hue_var='separation', style_var='congruent', style_order=[1, -1],
                                error_bars=True, even_spaced_x=True, jitter=False,
                                fig_title=fig5_title, fig_savename=fig5_savename,
                                save_path=save_path, x_tick_vals=x_tick_vals,
                                verbose=verbose)
    if show_plots:
        plt.show()
    plt.close()

    print("\n***finished c_plots()***\n")


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

        # if cols_to_replace is not None:
        #     replace_cols = ave_psignifit_thr_df[[cols_to_replace]]
        #     print(f"replace_cols: {cols_to_replace}\n{replace_cols}")
        if cols_to_replace is not None:
            # ave_psignifit_thr_df.insert(1, cols_to_replace, replace_cols)

            # make a cond_type list from the neg_sep column
            # cond_type_list = [cols_to_replace_dict[x]['cond_type'] for x in ave_psignifit_thr_df['neg_sep'].to_list()]
            # cond_type_list = []
            # for neg_sep_val in list(ave_psignifit_thr_df['neg_sep']):
            #     cond_type_list.append(cols_to_replace_dict[neg_sep_val]['cond_type'])
            # print(f"cond_type_list: {cond_type_list}")

            # insert cond_type and separation columns to ave_psignifit_thr_df from cols_to_replace_dict
            # ave_psignifit_thr_df.insert(1, 'cond_type', [cols_to_replace_dict[x]['cond_type'] for x in ave_psignifit_thr_df['neg_sep'].to_list()])
            # ave_psignifit_thr_df.insert(2, 'separation', [cols_to_replace_dict[x]['separation'] for x in ave_psignifit_thr_df['neg_sep'].to_list()])

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
            # error_bars_df = error_bars_df.drop(cols_to_replace, axis=1)
            # error_bars_df.insert(0, cols_to_replace, replace_cols)
            # error_bars_df.fillna(0)

            # insert cond_type and separation columns to error_bars_df from cols_to_replace_dict
            # error_bars_df.insert(0, 'cond_type', [cols_to_replace_dict[x]['cond_type'] for x in error_bars_df['neg_sep'].to_list()])
            # error_bars_df.insert(1, 'separation', [cols_to_replace_dict[x]['separation'] for x in error_bars_df['neg_sep'].to_list()])

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

            # if 'area_deg' in groupby_sep_df.columns:
            #     # for ricco_v2 experiment
            #     # print(f"\nwhatabouthis:\n{groupby_sep_df.groupby(['cond', 'separation'], sort=True).sem()}")
            #
            #     ave_psignifit_thr_df = groupby_sep_df.groupby(['cond', 'separation'], sort=False).mean()
            #     # print(f'\njust made ave_psignifit_thr_df:\n{ave_psignifit_thr_df}')
            #     stair_names = groupby_sep_df['stair_names'].unique()
            #     ave_psignifit_thr_df.insert(0, 'stair_names', stair_names)
            #     cond_values = ave_psignifit_thr_df.index.get_level_values('cond').to_list()
            #     sep_values = ave_psignifit_thr_df.index.get_level_values('separation').to_list()
            #     area_deg_values = ave_psignifit_thr_df['area_deg'].to_list()
            #     area_pix_vals = ave_psignifit_thr_df['n_pixels'].to_list()
            #     len_values = ave_psignifit_thr_df['length'].to_list()
            #
            #     print(f'\ncond_values:\n{cond_values}')
            #     print(f'sep_values:\n{sep_values}')
            #     # print('just made ave_psignifit_thr_df')
            else:
                groupby_sep_df = groupby_sep_df.drop('separation', axis=1)
                if 'cond' in groupby_sep_df.columns:
                    groupby_sep_df = groupby_sep_df.drop('cond', axis=1)
                # ave_psignifit_thr_df = groupby_sep_df.groupby('stair_names', sort=False).mean()
            ave_psignifit_thr_df = groupby_sep_df.groupby('stair_names', sort=False).mean()

            if verbose:
                print(f'\nave_psignifit_thr_df:\n{ave_psignifit_thr_df}')
                print(f'\ngroupby_sep_df:\n{groupby_sep_df}')

            # groupby_sep_df = groupby_sep_df.drop(['separation', 'cond', 'area_deg'], axis=1)
            # print(f'\ngroupby_sep_df:\n{groupby_sep_df}')

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
            # print('just made error_bars_df')

            # if 'area_deg' in error_bars_df.columns.to_list():
            #     # todo:do I still need this - area_deg not in ricco df
            #     print(f'\nerror_bars_df:\n{error_bars_df}')
            #
            #     # getting sep and col vals from here, not above, as order changes if conds have NaNs due to only 1 run.
            #     stair_names_list = error_bars_df.index.get_level_values('stair_names').to_list()
            #     print(f'\nstair_names_list:\n{stair_names_list}')
            #     sep_vals = []
            #     cond_vals = []
            #     for name in stair_names_list:
            #         x = name.split("_")
            #         sep_vals.append(int(x[0]))
            #         cond_vals.append(x[1])
            #     print(f'\nsep_vals:\n{sep_vals}')
            #     print(f'\ncond_vals:\n{cond_vals}')
            #
            #     # for ricco_v2 exp - change order to match ave_psignifit_thr_df
            #     error_bars_df.insert(0, 'cond', cond_vals)
            #     error_bars_df['separation'] = sep_vals
            #     error_bars_df['area_deg'] = area_deg_values
            #     error_bars_df['n_pixels'] = area_pix_vals
            #     error_bars_df['length'] = len_values
            #     error_bars_df.reset_index()
            #     print(f'check columns: {error_bars_df.columns.to_list()}')
            #     # col_order = ['cond', 'separation', 'stair_names', 'area', 'weber_thr', 'ISI_0']
            #     col_order = ['cond', 'separation', 'area_deg', 'n_pixels', 'length', 'delta_I', 'weber_thr', 'thr']
            #
            #     error_bars_df.reset_index(inplace=True)
            #     error_bars_df = error_bars_df[col_order]
            #     error_bars_df.set_index(['cond', 'separation'], inplace=True)
            #
            #     print(f'\nerror_bars_df: ({error_type})\n{error_bars_df}')
            # print(f'\nerror_bars_df: ({error_type})\n{error_bars_df}')

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




def e_average_exp_data(exp_path, p_names_list,
                       # stair_names_col,
                       # groupby_col=None, cols_to_drop=None, cols_to_replace=None,
                       exp_type='rad_flow',
                       error_type='SE',
                       n_trimmed=None,
                       verbose=True):


    """
    e_average_over_participants: take MASTER_ave_TM_thresh.csv (or MASTER_ave_thresh.csv)
    in each participant folder and make master list
    MASTER_exp_thr.csv

    Get mean thresholds averaged across all participants saved as
    MASTER_exp_ave_thr.csv

    Save master lists to exp_path.

    Plots:
    MASTER_exp_ave_thr saved as exp_ave_thr_all_runs.png
    MASTER_exp_ave_thr two-probe/one-probe saved as exp_ave_thr_div_1probe.png
    these both have two versions:
    a. x-axis is separation, ISI as different lines
    b. x-axis is ISI, separation as different lines
    Heatmap: with average probe lum for ISI and separation.

    :param exp_path: dir containing participant folders
    :param p_names_list: names of participant's folders
    :param exp_type: type of experiment.  This is because different columns etc
        are required for radial_flow or Ricco etc.
    :param error_type: Default: None. Can pass sd or se for standard deviation or error.
    :param n_trimmed: default None.  If None, use MASTER_ave_thresh.
            Else use: trimmed_mean ave (MASTER_ave_TM{n_trimmed}_thresh),

    :param verbose: Default True, print progress to screen

    :returns: exp_ave_thr_df: experiment mean threshold for each separation and ISI.
    """
    print("\n***running e_average_over_participants()***\n")

    """ part1. Munge data, save master lists and get means etc
     - loop through participants and get each MASTER_ave_TM_thresh.csv
    Make master sheets: MASTER_exp_thr and MASTER_exp_ave_thr."""

    print(f"p_names_list: {p_names_list}")

    all_p_ave_list = []
    for p_idx, p_name in enumerate(p_names_list):

        # ave_df_name = 'MASTER_ave_thresh'
        # if n_trimmed is not None:
        #     ave_df_name = f'MASTER_ave_TM{n_trimmed}_thresh'

        # check for trimmed means
        # ave_df_name = f'MASTER_ave_TM{n_trimmed}_thresh'
        ave_df_name = f'MASTER_ave_TM2_thresh'
        this_ave_df_path = os.path.join(exp_path, p_name, f'{ave_df_name}.csv')

        # # if trimmed mean doesn't exists (e.g., because participant hasn't done 12 runs)
        if not os.path.isfile(this_ave_df_path):
            ave_df_name = f'MASTER_ave_TM1_thresh'
            this_ave_df_path = os.path.join(exp_path, p_name, f'{ave_df_name}.csv')
            if not os.path.isfile(this_ave_df_path):
                print(f"Couldn't find trimmed mean data for {p_name}\nUsing untrimmed instead.")
                this_ave_df_path = os.path.join(exp_path, p_name, 'MASTER_ave_thresh.csv')

        this_p_ave_df = pd.read_csv(this_ave_df_path)

        if verbose:
            print(f'\n{p_idx}. {p_name} - this_p_ave_df:\n{this_p_ave_df}')

        if 'Unnamed: 0' in list(this_p_ave_df):
            this_p_ave_df.drop('Unnamed: 0', axis=1, inplace=True)

        rows, cols = this_p_ave_df.shape
        this_p_ave_df.insert(0, 'participant', [p_name] * rows)

        if exp_type == 'Dec23':
            pass
        elif exp_type in ['Ricco', 'Bloch', 'speed_detection']:
            this_p_ave_df.rename(columns={'ISI_0': 'thr'}, inplace=True)
        elif exp_type in ['Bloch_v5']:
            if 'stair_name' in this_p_ave_df.columns.tolist():
                this_p_ave_df.rename(columns={'stair_name': 'stair_names'}, inplace=True)
        elif exp_type == 'missing_probe':
            print('Do I need to do anything here for missing probe exp?')
            # # todo: I can probably delete these
            # cont_type_col = this_p_ave_df['cond_type'].tolist()
            # neg_sep_col = this_p_ave_df['neg_sep'].tolist()
            # sep_col = this_p_ave_df['separation'].tolist()
        elif exp_type == 'missing_mixed':
            print('Do I need to do anything here for missing probe exp?')
        else:
            if 'stair_names' in this_p_ave_df.columns.tolist():
                stair_names_list = this_p_ave_df['stair_names'].tolist()
            elif 'separation' in this_p_ave_df.columns.tolist():
                stair_names_list = this_p_ave_df['separation'].tolist()
            if verbose:
                print(f'stair_names_list: {stair_names_list}')
            sep_list = [0 if x == -.10 else abs(int(x)) for x in stair_names_list]
            cong_list = [-1 if x < 0 else 1 for x in stair_names_list]
            this_p_ave_df.insert(2, 'congruent', cong_list)
            if 'separation' not in this_p_ave_df.columns.tolist():
                this_p_ave_df.insert(3, 'separation', sep_list)


        all_p_ave_list.append(this_p_ave_df)

    # join all participants' data and save as master csv
    all_exp_thr_df = pd.concat(all_p_ave_list, ignore_index=True)
    cols_list = list(all_exp_thr_df.columns)
    if cols_list == ['participant', 'stair_names', 'congruent', 'separation',
                     'ISI_6', 'ISI_9', 'ISI_1', 'ISI_4', 'ISI_8', 'ISI_10', 'ISI_12']:
        new_cols_list = ['participant', 'stair_names', 'congruent', 'separation',
                         'ISI_1', 'ISI_4', 'ISI_6', 'ISI_8', 'ISI_9', 'ISI_10', 'ISI_12']
        all_exp_thr_df = all_exp_thr_df[new_cols_list]
    if verbose:
        print(f'\nall_exp_thr_df:{list(all_exp_thr_df.columns)}\n{all_exp_thr_df}')
    all_exp_thr_df.to_csv(os.path.join(exp_path, 'MASTER_exp_thr.csv'), index=False)

    # # get means and errors
    print(f"exp_type: {exp_type}")
    groupby_sep_df = all_exp_thr_df.drop('participant', axis=1)
    if exp_type == 'Ricco':
        groupby_col = 'stair_names'
        sort_rows = False
    elif exp_type == 'Ricco_v5':
        groupby_col = 'separation'
        sort_rows = False
    elif exp_type == 'Ricco_v6':
        groupby_col = 'separation'
        sort_rows = False
    elif exp_type in ['Ricco_vall', 'Ricco_all']:
        groupby_col = 'separation'
        sort_rows = False
    elif exp_type == 'Bloch':
        groupby_sep_df['stair_names'] = groupby_sep_df['cond_type'] + "_" + groupby_sep_df["ISI"].map(str)
        groupby_sep_df = groupby_sep_df.drop('cond_type', axis=1)
        groupby_col = 'stair_names'
        sort_rows = False
    elif exp_type == 'Bloch_v5':
        # groupby_sep_df['stair_names'] = groupby_sep_df['cond_type'] + "_" + groupby_sep_df["isi_fr"].map(str)
        if 'cond_type' in list(groupby_sep_df.columns):
            groupby_sep_df = groupby_sep_df.drop('cond_type', axis=1)
        # groupby_col = 'stair_names'
        # groupby_col = ['stair_names', 'isi_fr', 'dur_ms']
        groupby_col = ['isi_fr', 'dur_ms']
        sort_rows = False
    elif exp_type == 'radial':
        groupby_col = 'participant'
        sort_rows = False
    elif exp_type == 'speed_detection':
        groupby_col = 'participant'
        sort_rows = False
    elif exp_type == 'missing_probe':
        groupby_col = ['cond_type', 'neg_sep']
        groupby_sep_df = groupby_sep_df.drop('separation', axis=1)
        sort_rows = 'neg_sep'
    elif exp_type == 'missing_mixed':
        groupby_col = ['neg_sep']
        # groupby_sep_df = groupby_sep_df.drop('separation', axis=1)
        sort_rows = 'neg_sep'
    # groupby_col=['cond_type', 'neg_sep'], cols_to_drop='stack', cols_to_replace='separation',

    else:
        groupby_sep_df = groupby_sep_df.drop('separation', axis=1)
        groupby_sep_df = groupby_sep_df.drop('congruent', axis=1)
        # todo: should I change sort to False for groupby?  Causes problems in
        #  d_average_participants for error_df if there was only a single run of a
        #  condition so error was NaN and somehow order changed.
        groupby_col = 'stair_names'
        sort_rows = True

    exp_ave_thr_df = groupby_sep_df.groupby(groupby_col, sort=sort_rows).mean()
    exp_ave_thr_df = exp_ave_thr_df.sort_values(by=groupby_col)

    if verbose:
        print(f'\nexp_ave_thr_df:\n{exp_ave_thr_df}')

    if error_type in [False, None]:
        error_bars_df = None
    elif error_type.lower() in ['se', 'error', 'std-error', 'standard error', 'standard_error']:
        error_bars_df = groupby_sep_df.groupby(groupby_col, sort=sort_rows).sem()
    elif error_type.lower() in ['sd', 'stdev', 'std_dev', 'std.dev', 'deviation', 'standard_deviation']:
        error_bars_df = groupby_sep_df.groupby(groupby_col, sort=sort_rows).std()
    else:
        raise ValueError(f"error_type should be in:\nfor none: [False, None]\n"
                         f"for standard error: ['se', 'error', 'std-error', 'standard error', 'standrad_error']\n"
                         f"for standard deviation: ['sd', 'stdev', 'std_dev', 'std.dev', "
                         f"'deviation', 'standard_deviation']")

    # replace NaNs with zero in error_bars_df
    if error_bars_df is not None:
        error_bars_df = error_bars_df.fillna(0)

    if verbose:
        print(f'\nerror_bars_df: ({error_type})\n{error_bars_df}')

    # save csv with average values
    exp_ave_thr_df.to_csv(os.path.join(exp_path, 'MASTER_exp_ave_thr.csv'))
    error_bars_df.to_csv(os.path.join(exp_path, f'MASTER_ave_thr_error_{error_type}.csv'))
    print("\n*** finished e_average_over_participants()***\n")

    return exp_ave_thr_df, error_bars_df


# def d_average_participant_Dec23(root_path, run_dir_names_list,
#                                 thr_df_name='psignifit_thresholds',
#                                 error_type=None,
#                                 trim_n=None,
#                                 verbose=True):
#     """
#     d_average_participant: take psignifit_thresholds.csv
#     in each participant run folder and make master lists
#     MASTER_psignifit_thresholds.csv
#
#     Get mean threshold across 6 run conditions saved as
#     MASTER_ave_thresh.csv
#
#     Save master lists to folder containing the six runs (root_path).
#
#     :param root_path: dir containing run folders
#     :param run_dir_names_list: names of run folders
#     :param thr_df_name: Name of threshold dataframe.  If no name is given it will use 'psignifit_thresholds'.
#     :param error_type: Default: None. Can pass sd or se for standard deviation or error.
#     :param trim_n: default None.  If int is passed, will call function trim_n_high_n_low(),
#             which trims the n highest and lowest values.
#     :param verbose: Default true, print progress to screen
#
#     :returns: ave_psignifit_thr_df: (trimmed?) mean threshold for each separation and dur.
#     """
#
#     print("\n***running d_average_participant_Dec23()***")
#
#     all_psignifit_list = []
#     for run_idx, run_name in enumerate(run_dir_names_list):
#
#         this_psignifit_df = pd.read_csv(f'{root_path}{os.sep}{run_name}{os.sep}{thr_df_name}.csv')
#         print(f'\n{run_idx}. {run_name} - this_psignifit_df:\n{this_psignifit_df}')
#
#         if 'Unnamed: 0' in list(this_psignifit_df):
#             this_psignifit_df.drop('Unnamed: 0', axis=1, inplace=True)
#
#         # if 'stair' in list(this_psignifit_df):
#         #     stair_list = this_psignifit_df['stair'].to_list
#         #     # this_psignifit_df.drop(columns='stair', inplace=True)
#
#         if 'stair_name' not in list(this_psignifit_df.columns):
#             # generate a stair_name from the columns preceeding probe_dur_ms columns (e.g., 'flow_{flow_dir}_{flow_name}_prelim_{prelim_ms}")
#             # this_psignifit_df.insert(0, 'stair_name', [f'flow_{flow_dir}_{flow_name}_prelim_{prelim_ms}'
#             #                                       for flow_dir, flow_name, prelim_ms in
#             #                                       zip(this_psignifit_df['flow_dir'], this_psignifit_df['flow_name'],
#             #                                           this_psignifit_df['prelim_ms'])])
#             this_psignifit_df.insert(0, 'stair_name', [f'flow_{flow_dir}_{flow_name}_bg_motion_ms{prelim_ms}'
#                                                   for flow_dir, flow_name, prelim_ms in
#                                                   zip(this_psignifit_df['flow_dir'], this_psignifit_df['flow_name'],
#                                                       this_psignifit_df['bg_motion_ms'])])
#             print(f'\nget_means_df:\n{this_psignifit_df}')
#         if 'stair' not in list(this_psignifit_df.columns):
#             # generate stair numbers from unique stair_names
#             this_psignifit_df.insert(0, 'stair', [i for i in range(len(this_psignifit_df['stair_name'].unique()))])
#
#         rows, cols = this_psignifit_df.shape
#         this_psignifit_df.insert(0, 'stack', [run_idx] * rows)
#
#         if verbose:
#             print(f'\nthis_psignifit_df:\n{this_psignifit_df}')
#
#         all_psignifit_list.append(this_psignifit_df)
#
#     # join all stacks (runs/groups) data and save as master csv
#     all_data_psignifit_df = pd.concat(all_psignifit_list, ignore_index=True)
#     # todo: since I added extra dur conditions, dur conds are not in ascending order.
#     #  Perhaps re-order columns before saving?
#
#     all_data_psignifit_df.to_csv(f'{root_path}{os.sep}MASTER_{thr_df_name}.csv', index=False)
#     if verbose:
#         print(f'\nall_data_psignifit_df:\n{all_data_psignifit_df}')
#
#     """Part 2: trim highest and lowest values is required and get average vals and errors"""
#     # # trim highest and lowest values
#     if trim_n is not None:
#         trimmed_df = trim_n_high_n_low(all_data_psignifit_df, trim_from_ends=trim_n,
#                                        reference_col='stair',
#                                        stack_col_id='stack',
#                                        verbose=verbose)
#         trimmed_df.to_csv(f'{root_path}{os.sep}MASTER_TM{trim_n}_thresholds.csv', index=False)
#
#         get_means_df = trimmed_df
#     else:
#         get_means_df = all_data_psignifit_df
#
#     print(f'\nget_means_df:\n{get_means_df}')
#
#     # # get means and errors
#     get_means_df = get_means_df.drop('stack', axis=1)
#
#     # loop through stair_list and add corresponding stair_name and flow_name to list
#     stair_list = get_means_df['stair'].unique().tolist()
#     stair_names_list = []
#     # prelim_list = []
#     bg_motion_ms_list = []
#     flow_dir_list = []
#     flow_names_list = []
#     for stair in stair_list:
#         stair_names_list.append(get_means_df.loc[get_means_df['stair'] == stair, 'stair_name'].unique().tolist()[0])
#         # prelim_list.append(get_means_df.loc[get_means_df['stair'] == stair, 'prelim_ms'].unique().tolist()[0])
#         bg_motion_ms_list.append(get_means_df.loc[get_means_df['stair'] == stair, 'bg_motion_ms'].unique().tolist()[0])
#         flow_dir_list.append(get_means_df.loc[get_means_df['stair'] == stair, 'flow_dir'].unique().tolist()[0])
#         flow_names_list.append(get_means_df.loc[get_means_df['stair'] == stair, 'flow_name'].unique().tolist()[0])
#
#     # get_means_df = get_means_df.drop('prelim_ms', axis=1)
#     get_means_df = get_means_df.drop('bg_motion_ms', axis=1)
#
#     get_means_df = get_means_df.drop('flow_dir', axis=1)
#
#     get_means_df = get_means_df.drop('stair_name', axis=1)
#     get_means_df = get_means_df.drop('flow_name', axis=1)
#
#
#
#     # get average values (from numeric columns)
#     ave_psignifit_thr_df = get_means_df.groupby(['stair'], sort=False).mean()
#     # add stair_names and flow_name back in
#     ave_psignifit_thr_df.insert(0, 'stair_name', stair_names_list)
#     # ave_psignifit_thr_df.insert(1, 'prelim_ms', prelim_list)
#     ave_psignifit_thr_df.insert(1, 'bg_motion_ms', bg_motion_ms_list)
#     ave_psignifit_thr_df.insert(2, 'flow_dir', flow_dir_list)
#     ave_psignifit_thr_df.insert(3, 'flow_name', flow_names_list)
#     if verbose:
#         print(f'\nget_means_df:\n{get_means_df}')
#         print(f'\nave_psignifit_thr_df:\n{ave_psignifit_thr_df}')
#
#     if error_type in [False, None]:
#         error_bars_df = None
#     elif error_type.lower() in ['se', 'error', 'std-error', 'standard error', 'standard_error']:
#         error_bars_df = get_means_df.groupby('stair', sort=False).sem()
#     elif error_type.lower() in ['sd', 'stdev', 'std_dev', 'std.dev', 'deviation', 'standard_deviation']:
#         error_bars_df = get_means_df.groupby('stair', sort=False).std()
#     else:
#         raise ValueError(f"error_type should be in:\nfor none: [False, None]\n"
#                          f"for standard error: ['se', 'error', 'std-error', 'standard error', 'standard_error']\n"
#                          f"for standard deviation: ['sd', 'stdev', 'std_dev', 'std.dev', "
#                          f"'deviation', 'standard_deviation']")
#
#     # replace NaNs with 0
#     error_bars_df.fillna(0, inplace=True)
#
#     # add stair_names and flow_name back in
#     error_bars_df.insert(0, 'stair_name', stair_names_list)
#     # error_bars_df.insert(1, 'prelim_ms', prelim_list)
#     error_bars_df.insert(1, 'bg_motion_ms', bg_motion_ms_list)
#     error_bars_df.insert(2, 'flow_dir', flow_dir_list)
#     error_bars_df.insert(3, 'flow_name', flow_names_list)
#     print(f'\nerror_bars_df:\n{error_bars_df}')
#
#     # save csv with average values
#     # todo: since I added extra dur conditions, dur conds are not in ascending order.
#     #  Perhaps re-order columns before saving?
#     if trim_n is not None:
#         ave_psignifit_thr_df.to_csv(f'{root_path}{os.sep}MASTER_ave_TM{trim_n}_thresh.csv')
#         error_bars_df.to_csv(f'{root_path}{os.sep}MASTER_ave_TM{trim_n}_thr_error_{error_type}.csv')
#     else:
#         ave_psignifit_thr_df.to_csv(f'{root_path}{os.sep}MASTER_ave_thresh.csv')
#         error_bars_df.to_csv(f'{root_path}{os.sep}MASTER_ave_thr_error_{error_type}.csv')
#
#     print("\n*** finished d_average_participant_Dec23()***\n")
#
#     return ave_psignifit_thr_df, error_bars_df


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



def make_average_plots(all_df_path, ave_df_path, error_bars_path,
                       thr_col='newLum',
                       stair_names_col='stair_names',
                       cond_type_col='congruent',
                       cond_type_order=[1, -1],
                       pos_neg_labels=['Congruent', 'Incongruent'],
                       n_trimmed=False,
                       ave_over_n=None,
                       exp_ave=False,
                       isi_name_list=None,
                       isi_vals_list=None,
                       show_plots=True,
                       save_path=None,
                       verbose=True):
    """Plots:
    MASTER_ave_thresh saved as ave_thr_all_runs.png
    MASTER_ave_thresh two-probe/one-probe saved as ave_thr_div_1probe.png
    these both have two versions:
    a. x-axis is separation, ISI as different lines
    b. x-axis is ISI, separation as different lines
    Heatmap: with average probe lum for ISI and separation.

    :param all_df_path: Path to df with all participant/stack data.
    :param ave_df_path: Path to df with average across all stacks/participants
    :param error_bars_path: Path to df for errors bars with SE/SD associated with averages.
    :param thr_col: Name of column containing threshold values.
    :param stair_names_col: Name of column containing condition names (e.g., neg_sep)
    :param cond_type_col: Name of column containing condition type (e.g., congruent)
    :param cond_type_order: Order of condition types (e.g., [1, -1]).
                            Does not have to match the order they appear in the dfs,
                            but the order you want them presented.
                            The first item is a solid line, the second is a dashed line.
    :param pos_neg_labels: Labels for condition types (e.g., ['Congruent', 'Incongruent'])
    :param n_trimmed: Whether averages data has been trimmed.
    :param ave_over_n: Number of runs or participants it is averaging over.
    :param exp_ave: If False, this script is for participant averages over runs.
                    If True, the script if for experiment averages over participants.
    :param isi_name_list: List of ISI names (e.g., ['ISI_0', 'ISI_1', 'ISI_2'])
    :param isi_vals_list: List of ISI values (e.g., [0, 1, 2])
    :param show_plots:
    :param verbose:
    :return: """

    print("\n*** running make_average_plots()***\n")

    # todo: check why plots have changed order - since I added extra ISI conditions.

    if save_path is None:
        # if ave_df_path is str (e.g. path to csv)
        if isinstance(ave_df_path, str):
            save_path, df_name = os.path.split(ave_df_path)
            print(f'save_path (ave_df_path): {save_path}')
            print(f'df_name (ave_df_path): {df_name}')
        else:
            raise ValueError(f"save_path must be specified if ave_df_path is not a path to a csv.")
    else:
        print(f'save_path: {save_path}')

    # if exp_ave:
    #     ave_over = 'Exp'
    # else:
    #     ave_over = 'P'
    # Average over experiment or participant (with or without participant name)
    if type(exp_ave) == str:  # e.g. participant's name
        ave_over = exp_ave
        # idx_col = 'stack'
    elif exp_ave is True:
        ave_over = 'Exp'
        # idx_col = 'p_stack_sep'
    else:
        ave_over = 'P'
        # idx_col = 'stack'

    # if type(all_df_path) is 'pandas.core.frame.DataFrame':
    if isinstance(all_df_path, pd.DataFrame):
        all_df = all_df_path
    else:
        all_df = pd.read_csv(all_df_path)

    # if type(ave_df_path) is 'pandas.core.frame.DataFrame':
    if isinstance(ave_df_path, pd.DataFrame):
        ave_df = ave_df_path
    else:
        ave_df = pd.read_csv(ave_df_path)

    # if type(error_bars_path) is 'pandas.core.frame.DataFrame':
    if isinstance(error_bars_path, pd.DataFrame):
        error_bars_df = error_bars_path
    else:
        error_bars_df = pd.read_csv(error_bars_path)

    # remove 'index' column if it exists
    if 'index' in list(all_df.columns):
        all_df.drop('index', axis=1, inplace=True)
    if 'index' in list(ave_df.columns):
        ave_df.drop('index', axis=1, inplace=True)
    if 'index' in list(error_bars_df.columns):
        error_bars_df.drop('index', axis=1, inplace=True)

    if verbose:
        print(f'\nall_df:\n{all_df}')
        print(f'\nave_df:\n{ave_df}')
        print(f'\nerror_bars_df:\n{error_bars_df}')

    if isi_name_list is None:
        # isi_name_list = list(all_df.columns[4:])
        # get a list of all column names containing 'isi' or 'ISI'
        isi_name_list = [i for i in list(all_df.columns) if 'isi' in i.lower()]
    if isi_vals_list is None:
        # isi_vals_list = [int(i[4:]) for i in isi_name_list]
        isi_vals_list = []
        for i in isi_name_list:
            if '.' in i:  # remove any characters after the '.'
                i = i[:i.index('.')]
            i = ''.join([j for j in i if j.isdigit()])  # remove any non-numeric characters
            isi_vals_list.append(int(i))

    if verbose:
        print(f'\nisi_name_list: {isi_name_list}')
        print(f'isi_vals_list: {isi_vals_list}')

    # get pos sep values and pos and neg values
    stair_names_list = sorted(list(all_df[stair_names_col].unique()))
    stair_names_list = [-.1 if i == -.10 else int(i) for i in stair_names_list]
    stair_names_labels = ['-0' if i == -.10 else int(i) for i in stair_names_list]
    if verbose:
        print(f"stair_names_list: {stair_names_list}")
        print(f"stair_names_labels: {stair_names_labels}")

    # get positive separation values, in ascending order
    if 'separation' in list(ave_df.columns):
        pos_sep_vals_list = ave_df['separation'].unique().tolist()
    else:
        pos_sep_vals_list = ave_df[stair_names_col].unique().tolist()
    pos_sep_vals_list = list(set([int(0) if i == -.1 else abs(int(i)) for i in pos_sep_vals_list]))
    if pos_sep_vals_list[0] > pos_sep_vals_list[-1]:
        pos_sep_vals_list.reverse()
    if verbose:
        print(f'pos_sep_vals_list: {pos_sep_vals_list}')

    # if n_trimmed is a list of identical values, replace with int (e.g., [2, 2, 2] becomes 2)
    if isinstance(n_trimmed, list) and len(set(n_trimmed)) == 1:
        n_trimmed = n_trimmed[0]
    print(f'n_trimmed: {n_trimmed}')


    """part 3. main Figures (these are the ones saved in the matlab script)
    Fig1: plot average threshold for each ISI and sep.
    Fig2: divide all 2probe conds (pos_sep) by one_probe condition for each participant.
    For both figures there are 2 versions:
        a) Sep on x-axis, different line for each ISI
        b) ISI on x-axis, different line for each Sep"""

    print(f"\nfig_1a")
    if n_trimmed is not None:
        fig_1a_title = f'{ave_over} ave thresholds per ISI. (n={ave_over_n}, trim={n_trimmed}).\n' \
                       f'positive=congruent probe/flow motion, negative=incongruent. Bars=SE.'
        if type(cond_type_order[0]) is str:
            fig_1a_title = f'{ave_over} ave thresholds per ISI. (n={ave_over_n}, trim={n_trimmed}).\n' \
                           f'positive={cond_type_order[0]}, negative={cond_type_order[1]}. Bars=SE.'
        fig_1a_savename = f'ave_TM{n_trimmed}_thr_pos_and_neg.png'
    else:
        fig_1a_title = f'{ave_over} ave threshold per ISI. (n={ave_over_n})\n' \
                       f'positive=congruent probe/flow motion, negative=incongruent. Bars=SE.'
        if type(cond_type_order[0]) is str:
            fig_1a_title = f'{ave_over} ave threshold per ISI. (n={ave_over_n})\n' \
                           f'positive={cond_type_order[0]}, negative={cond_type_order[1]}. Bars=SE.'
        fig_1a_savename = f'ave_thr_pos_and_neg.png'

    # use ave_w_sep_idx_df for fig 1a and heatmap
    ave_w_sep_idx_df = ave_df.set_index(stair_names_col)
    err_w_sep_idx_df = error_bars_df.set_index(stair_names_col)
    ave_w_sep_idx_df.sort_index(inplace=True)
    err_w_sep_idx_df.sort_index(inplace=True)
    if 'cond_type' in list(ave_w_sep_idx_df.columns):
        ave_w_sep_idx_df.drop('cond_type', axis=1, inplace=True)
        err_w_sep_idx_df.drop('cond_type', axis=1, inplace=True)
    if 'separation' in list(ave_w_sep_idx_df.columns):
        ave_w_sep_idx_df.drop('separation', axis=1, inplace=True)
        err_w_sep_idx_df.drop('separation', axis=1, inplace=True)
    if 'neg_sep' in list(ave_w_sep_idx_df.columns):
        ave_w_sep_idx_df.drop('neg_sep', axis=1, inplace=True)
        err_w_sep_idx_df.drop('neg_sep', axis=1, inplace=True)
    if 'stair_names' in list(ave_w_sep_idx_df.columns):
        ave_w_sep_idx_df.drop('stair_names', axis=1, inplace=True)
        err_w_sep_idx_df.drop('stair_names', axis=1, inplace=True)
    print(f"ave_w_sep_idx_df:\n{ave_w_sep_idx_df}")
    print(f"err_w_sep_idx_df:\n{err_w_sep_idx_df}")

    # if I delete this messy plot, I can also delete the function that made it.
    plot_runs_ave_w_errors(fig_df=ave_w_sep_idx_df, error_df=err_w_sep_idx_df,
                           jitter=.1, error_caps=True, alt_colours=False,
                           legend_names=isi_name_list,
                           x_tick_vals=stair_names_list,
                           x_tick_labels=stair_names_labels,
                           even_spaced_x=True, fixed_y_range=False,
                           fig_title=fig_1a_title, save_name=fig_1a_savename,
                           save_path=save_path, verbose=verbose)
    if show_plots:
        plt.show()
    plt.close()

    if len(isi_vals_list) == 1:
        print("skipping fig_1b as there is only 1 ISI value")
    else:
        print(f"\nfig_1b")
        if n_trimmed is not None:
            fig_1b_title = f'{ave_over} average thresholds per separation (n={ave_over_n}, trim={n_trimmed}).\n' \
                           f'Bars=.68 CI'
            fig_1b_savename = f'ave_TM{n_trimmed}_thr_all_runs_sep.png'
        else:
            fig_1b_title = f'{ave_over} average threshold per separation\n' \
                           f'Bars=.68 CI, n={ave_over_n}'
            fig_1b_savename = f'ave_thr_all_runs_sep.png'

            # tod: It also seems that using evenly_spaced_x is messing it up.  Not sure why.


        plot_w_errors_either_x_axis(wide_df=all_df, cols_to_keep=[cond_type_col, 'separation'],
                                    cols_to_change=isi_name_list,
                                    cols_to_change_show=thr_col, new_col_name='ISI',
                                    strip_from_cols='ISI_', x_axis='ISI', y_axis=thr_col,
                                    hue_var='separation', style_var=cond_type_col, style_order=cond_type_order,
                                    legend_names=isi_name_list,
                                    error_bars=True, even_spaced_x=True, jitter=.05,
                                    fig_title=fig_1b_title, fig_savename=fig_1b_savename,
                                    save_path=save_path, x_tick_vals=isi_vals_list, verbose=verbose)
        if show_plots:
            plt.show()
        plt.close()

    print(f"\nfig_1c")
    if n_trimmed is not None:
        fig_1c_title = f'{ave_over} average thresholds per ISI across all runs (trim={n_trimmed}).\n' \
                       f'Bars=.68 CI, n={ave_over_n}'
        fig_1c_savename = f'ave_TM{n_trimmed}_thr_all_runs_isi.png'
    else:
        fig_1c_title = f'{ave_over} average threshold per ISI across all runs\n' \
                       f'Bars=.68 CI, n={ave_over_n}'
        fig_1c_savename = f'ave_thr_all_runs_isi.png'

    print(f'pos_sep_vals_list: {pos_sep_vals_list}')

    plot_w_errors_either_x_axis(wide_df=all_df, cols_to_keep=[cond_type_col, 'separation'],
                                cols_to_change=isi_name_list,
                                cols_to_change_show=thr_col, new_col_name='ISI',
                                strip_from_cols='ISI_',
                                x_axis='separation', y_axis=thr_col,
                                hue_var='ISI', style_var=cond_type_col, style_order=cond_type_order,
                                error_bars=True,
                                log_scale=False,
                                even_spaced_x=True, jitter=.05,
                                fig_title=fig_1c_title, fig_savename=fig_1c_savename,
                                x_tick_vals=pos_sep_vals_list, save_path=save_path, verbose=verbose)
    if show_plots:
        plt.show()
    plt.close()

    #################
    if len(isi_vals_list) == 1:
        print("skipping fig_1d and fig_1e (multiplots) as there is only 1 ISI value")
    else:
        '''Plots per ISI'''
        # check if 'cond_type' column is in ave_df, if not add it with neg_sep_to_sep_w_cond_type()
        if cond_type_col not in ave_df.columns:
            print(f"adding {cond_type_col} column to ave_df")
            new_ave_df = neg_sep_to_sep_w_cond_type(orig_df=ave_df, pos_neg_labels=pos_neg_labels,
                                                    neg_sep_col='neg_sep',
                                                    cols_to_drop=['stair_names', 'neg_sep'])
            new_err_df = neg_sep_to_sep_w_cond_type(orig_df=error_bars_df, pos_neg_labels=pos_neg_labels,
                                                    neg_sep_col='neg_sep',
                                                    cols_to_drop=['stair_names', 'neg_sep'])
        else:
            print(f"cond_type_col: {cond_type_col} already in ave_df")
            # drop 'neg_sep' column from ave_df and error_bars_df
            new_ave_df = ave_df.drop(columns=['neg_sep'])
            new_err_df = error_bars_df.drop(columns=['neg_sep'])

        print(f"new_ave_df: \n{new_ave_df}")
        print(f"new_err_df: \n{new_err_df}")

        # plot per ISI
        # figure 1d multiple plots with single line.
        print("\n\nfig_1d: one ax per ISI, pos_sep, compare congruent and incongruent.")
        if n_trimmed is not None:
            fig_1d_title = f'{ave_over} Congruent and Incongruent thresholds for each ISI (trim={n_trimmed}).\n' \
                           f'Bars=SE, n={ave_over_n}'
            if type(cond_type_order[0]) is str:
                fig_1d_title = f'{ave_over} {cond_type_order[0]} and {cond_type_order[1]} thresholds for each ISI (trim={n_trimmed}).\n' \
                               f'Bars=SE, n={ave_over_n}'
            fig_1d_savename = f'ave_TM{n_trimmed}_per_isi.png'
        else:
            fig_1d_title = f'{ave_over} Congruent and Incongruent thresholds for each ISI\n' \
                           f'Bars=SE, n={ave_over_n}'
            if type(cond_type_order[0]) is str:
                fig_1d_title = f'{ave_over} {cond_type_order[0]} and {cond_type_order[1]} thresholds for each ISI\n' \
                               f'Bars=SE, n={ave_over_n}'
            fig_1d_savename = f'ave_thr_per_isi.png'

        multi_plt_per_col_w_hue(ave_thr_df=new_ave_df, error_df=new_err_df,
                                cond_type_col='cond_type',
                                pos_neg_labels=pos_neg_labels,
                                x_label_col='separation',
                                even_spaced_x=True, error_caps=True,
                                fig_title=fig_1d_title,
                                save_path=save_path, save_name=fig_1d_savename,
                                verbose=verbose)
        if show_plots:
            plt.show()
        plt.close()


        '''Plots per separation'''
        # transpose plots to do plot per separation
        transposed_df = transpose_df_w_cond_type(orig_df=ave_df,
                                                 cols_to_rows=isi_name_list,
                                                 add_pos_neg_labels=pos_neg_labels,
                                                 cond_type_col='cond_type',
                                                 verbose=True)
        transposed_err_df = transpose_df_w_cond_type(orig_df=error_bars_df,
                                                     cols_to_rows=isi_name_list,
                                                     add_pos_neg_labels=pos_neg_labels,
                                                     cond_type_col='cond_type',
                                                     verbose=True)
        print(f'transposed_df: \n{transposed_df}')
        print(f'transposed_err_df: \n{transposed_err_df}')

        # figure 1 multiple plots with single line.  Per separation
        print("\n\nfig_1e: one ax per sep")
        if n_trimmed is not None:
            fig_1e_title = f'{ave_over} Congruent and Incongruent thresholds for each separation (trim={n_trimmed}).\n' \
                           f'Bars=SE, n={ave_over_n}'
            if type(cond_type_order[0]) is str:
                fig_1e_title = f'{ave_over} {cond_type_order[0]} and {cond_type_order[1]} thresholds for each separation (trim={n_trimmed}).\n' \
                               f'Bars=SE, n={ave_over_n}'
            fig_1e_savename = f'ave_TM{n_trimmed}_per_sep.png'
        else:
            fig_1e_title = f'{ave_over} Congruent and Incongruent thresholds for each separation\n' \
                           f'Bars=SE, n={ave_over_n}'
            if type(cond_type_order[0]) is str:
                fig_1e_title = f'{ave_over} {cond_type_order[0]} and {cond_type_order[1]} thresholds for each separation\n' \
                               f'Bars=SE, n={ave_over_n}'
            fig_1e_savename = f'ave_thr_per_sep.png'

        multi_plt_per_col_w_hue(ave_thr_df=transposed_df, error_df=transposed_err_df,
                                cond_type_col='cond_type',
                                pos_neg_labels=pos_neg_labels,
                                x_label_col='ISI',
                                even_spaced_x=True, error_caps=True,
                                fig_title=fig_1e_title,
                                save_path=save_path, save_name=fig_1e_savename,
                                verbose=verbose)
        if show_plots:
            plt.show()
        plt.close()



    if len(isi_vals_list) == 1:
        print("skipping fig2a and fig2b as there is only 1 ISI value")
    else:
        print('\nfig2a: Mean participant difference between congruent and incongruent conditions (x-axis=Sep)')
        if n_trimmed is not None:
            fig_2a_title = f'{ave_over} Mean Difference (Congruent - Incongruent Conditions).\n' \
                           f'(x-axis=Sep)  trim={n_trimmed}, n={ave_over_n}.'
            fig_2a_savename = f'ave_TM{n_trimmed}_diff_x_sep.png'
        else:
            fig_2a_title = f'{ave_over} Mean Difference (Congruent - Incongruent Conditions).\n' \
                           f'(x-axis=Sep), n={ave_over_n}'
            fig_2a_savename = f'ave_diff_x_sep.png'

        use_these_cols = [stair_names_col] + isi_name_list
        print(f"ave_df: \n{ave_df}")
        print(f"use_these_cols: {use_these_cols}")
        print(f"ave_df[use_these_cols]: \n{ave_df[use_these_cols]}")

        plot_diff(ave_df[use_these_cols], stair_names_col=stair_names_col,
                  fig_title=fig_2a_title, save_path=save_path, save_name=fig_2a_savename,
                  x_axis_isi=False, verbose=verbose)
        if show_plots:
            plt.show()
        plt.close()

        print('\nfig2b: Mean participant difference between congruent and incongruent conditions (x-axis=ISI)')

        if n_trimmed is not None:
            fig_2b_title = f'{ave_over} Mean Difference Between Congruent and Incongruent Conditions (x-axis=ISI).\n' \
                           f'(Positive=congruent has higher threshold, n={ave_over_n}, trim={n_trimmed}).'
            fig_2b_savename = f'ave_TM{n_trimmed}_diff_x_isi.png'
        else:
            fig_2b_title = f'{ave_over} Mean Difference Between Congruent and Incongruent Conditions (x-axis=ISI).\n' \
                           f'(Positive=congruent has higher threshold, n={ave_over_n}).'
            fig_2b_savename = f'ave_diff_x_isi.png'


        plot_diff(ave_df[use_these_cols], stair_names_col=stair_names_col,
                  fig_title=fig_2b_title, save_path=save_path, save_name=fig_2b_savename,
                  x_axis_isi=True, verbose=verbose)
        if show_plots:
            plt.show()
        plt.close()


    print(f"\nHeatmap")
    if n_trimmed is not None:
        heatmap_title = f'{ave_over}\nmean Threshold for each ISI and separation (n={ave_over_n}, trim={n_trimmed}).'
        heatmap_savename = f'mean_TM{n_trimmed}_thr_heatmap'
    else:
        heatmap_title = f'{ave_over}\nmean Threshold for each ISI and separation (n={ave_over_n})'
        heatmap_savename = 'mean_thr_heatmap'

    if type(cond_type_order[0]) is str:
        heatmap_title = heatmap_title + f'\npos = {cond_type_order[0]}, neg = {cond_type_order[1]}'

    plot_thr_heatmap(heatmap_df=ave_w_sep_idx_df,
                     x_tick_labels=isi_name_list,
                     y_tick_labels=stair_names_labels,
                     fig_title=heatmap_title,
                     save_name=heatmap_savename,
                     save_path=save_path,
                     verbose=verbose)
    if show_plots:
        plt.show()
    plt.close()

    if len(isi_vals_list) == 1:
        print("skipping Heatmap per row and per column as there is only 1 ISI value")
    else:
        print(f"\nHeatmap per row\n")
        if 'separation' in list(ave_df.columns):
            ave_df.set_index('separation', drop=True, inplace=True)

        # get mean of each col, then mean of that
        if n_trimmed is not None:
            heatmap_pr_title = f'{ave_over} Heatmap per row (n={ave_over_n}, trim={n_trimmed}).'
            heatmap_pr_savename = f'mean_TM{n_trimmed}_heatmap_per_row'
        else:
            heatmap_pr_title = f'{ave_over} Heatmap per row (n={ave_over_n})'
            heatmap_pr_savename = 'mean_heatmap_per_row'

        if type(cond_type_order[0]) is str:
            heatmap_pr_title = heatmap_pr_title + f'\npos = {cond_type_order[0]}, neg = {cond_type_order[1]}'

        plt_heatmap_row_col(heatmap_df=ave_w_sep_idx_df,
                            colour_by='row',
                            x_axis_label='ISI',
                            fontsize=12,
                            annot_fmt='.3g',
                            y_axis_label='Separation',
                            fig_title=heatmap_pr_title,
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
            heatmap_pr_title = f'{ave_over} Heatmap per col (n={ave_over_n}, trim={n_trimmed}).'
            heatmap_pr_savename = f'AAA_mean_TM{n_trimmed}_heatmap_per_col.png'
        else:
            heatmap_pr_title = f'{ave_over} Heatmap per col (n={ave_over_n})'
            heatmap_pr_savename = 'AAA_mean_heatmap_per_col.png'

        if type(cond_type_order[0]) is str:
            heatmap_pr_title = heatmap_pr_title + f'\npos = {cond_type_order[0]}, neg = {cond_type_order[1]}'


        plt_heatmap_row_col(heatmap_df=ave_w_sep_idx_df,
                            colour_by='col',
                            x_axis_label='ISI',
                            fontsize=12,
                            annot_fmt='.3g',
                            y_axis_label='Separation',
                            fig_title=heatmap_pr_title,
                            save_name=heatmap_pr_savename,
                            save_path=save_path,
                            verbose=True)
        if show_plots:
            plt.show()
        plt.close()

    print("\n*** finished make_average_plots()***\n")


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