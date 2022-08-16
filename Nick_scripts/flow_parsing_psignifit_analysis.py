import os
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import math
from rad_flow_psignifit_analysis import split_df_alternate_rows, trim_n_high_n_low, \
    plot_runs_ave_w_errors, plot_w_errors_either_x_axis, multi_pos_sep_per_isi, plot_thr_heatmap

"""
This page contains python functions to analyse the radial_flow.py experiment.
The workflow is based on Martin's MATLAB scripts used in OLD_MATLAB_analysis.py.

1. a_data_extraction: put data from one run, multiple durs into one array. 
2. b1_extract_last_values: get the threshold used on the last values in each 
    staircase for each dur.  this is used for b3_plot_staircases.
# not used. 3. b2_last_reversal/m: Computes the threshold for each staircase as an average of 
#     the last N reversals (incorrect responses).  Plot these as mean per sep, 
#     and also multi-plot for each dur with different values for -18&+18 etc
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

#
# def split_df_alternate_rows(df):
#     """
#     Split a dataframe into alternate rows.  Dataframes are organized by
#     stair conditions relating to separations
#     (e.g., in order [18, -18, 6, -6, 3, -3, 2, -2, 1, -1, 0, 0]).
#     For some plots this needs to be split into two dfs, e.g., :
#     pos_sep_df: [18, 6, 3, 2, 1, 0]
#     neg_sep_df: [-18, -6, -3, -2, -1, 0]
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
#
#
# def trim_n_high_n_low(all_data_df, trim_from_ends=None, reference_col='separation',
#                       stack_col_id='stack', verbose=True):
#     """
#     Function for trimming the n highest and lowest values from each condition of a set with multiple runs.
#
#     :param all_data_df: Dataset to be trimmed.
#     :param trim_from_ends: number of values to trim from each end of the distribution.
#     :param reference_col: Idx column containing repeated conditions (e.g., separation has the same label for each stack).
#     :param stack_col_id: idx column showing different runs/groups etc (e.g., stack)
#     :param verbose: in true will print progress to screen.
#
#     :return: trimmed df
#     """
#     print('\n*** running trim_high_n_low() ***')
#
#     '''Part 1, convert 2d df into 3d numpy array'''
#     # prepare to extract numpy
#     if verbose:
#         print(f'all_data_df {all_data_df.shape}:\n{all_data_df.head(25)}')
#
#     # get unique values to loop over
#     stack_list = list(all_data_df[stack_col_id].unique())
#     datapoints_per_cond = len(stack_list)
#
#     if verbose:
#         print(f'stack_list: {stack_list}\n'
#               f'datapoints_per_cond: {datapoints_per_cond}')
#
#     # loop through df to get 3d numpy
#     my_list = []
#     for stack in stack_list:
#         stack_df = all_data_df[all_data_df[stack_col_id] == stack]
#         stack_df = stack_df.drop(stack_col_id, axis=1)
#         sep_list = list(stack_df.pop(reference_col))
#         dur_name_list = list(stack_df.columns)
#         # print(f'stack{stack}_df ({stack_df.shape}):\n{stack_df}')
#         my_list.append(stack_df.to_numpy())
#
#     # 3d numpy array are indexed with [depth, row, col]
#     # use variables depth_3d, rows_3d, cols_all later to reshaped_2d_array trimmed array
#     my_3d_array = np.array(my_list)
#     depth_3d, rows_3d, cols_all = np.shape(my_3d_array)
#
#     if trim_from_ends is not None:
#         target_3d_depth = depth_3d - 2 * trim_from_ends
#     else:
#         target_3d_depth = depth_3d
#     target_2d_rows = target_3d_depth * rows_3d
#     if verbose:
#         print(f'\nUse these values for defining array shapes.\n'
#               f'target_3d_depth (depth-trim): {target_3d_depth}, '
#               f'3d shape after trim (target_3d_depth, rows_3d, cols_all) = '
#               f'({target_3d_depth}, {rows_3d}, {cols_all})\n'
#               f'2d array shape (after trim, but before separation, stack or headers are added): '
#               f'(target_2d_rows, cols_all) = ({target_2d_rows}, {cols_all})')
#
#     '''Part 2, trim highest and lowest n values from each depth slice to get trimmed_3d_list'''
#     if verbose:
#         print('\ngetting depth slices to trim...')
#     trimmed_3d_list = []
#     counter = 0
#     for col in list(range(cols_all)):
#         row_list = []
#         for row in list(range(rows_3d)):
#             depth_slice = my_3d_array[:, row, col]
#             depth_slice = np.sort(depth_slice)
#             if trim_from_ends is not None:
#                 trimmed = depth_slice[trim_from_ends: -trim_from_ends]
#             else:
#                 trimmed = depth_slice[:]
#             # if verbose:
#             #     print(f'{counter}: {trimmed}')
#             row_list.append(trimmed)
#             counter += 1
#         trimmed_3d_list.append(row_list)
#
#     """
#     Part 3, turn 3d numpy back into 2d df.
#     trimmed_3d_list is a list of arrays (e.g., 3d).  Each array relates to a
#     depth-stack of my_3d_array which has now be trimmed (e.g., fewer rows).
#     However, trimmed_3d_list has the same depth and number of columns as my_3d_array.
#     trimmed_array re-shapes trimmed_3d_list so all values are in their original
#     row and column positions (e.g., separation and dur).
#     However, the 3rd dimension (depth) is not in original order, but in ascending order."""
#
#     trimmed_3d_array = np.array(trimmed_3d_list)
#     print(f'\n\nReshaping trimmed data\ntrimmed_3d_array: {np.shape(trimmed_3d_array)}')
#     if verbose:
#         print(trimmed_3d_array)
#
#     ravel_array_f = np.ravel(trimmed_3d_array, order='F')
#     print(f'\n1. ravel_array_f: {np.shape(ravel_array_f)}')
#     if verbose:
#         print(ravel_array_f)
#
#     reshaped_3d_array = ravel_array_f.reshape(target_3d_depth, rows_3d, cols_all)
#     print(f'\n2. reshaped_3d_array: {np.shape(reshaped_3d_array)}')
#     if verbose:
#         print(reshaped_3d_array)
#
#     reshaped_2d_array = reshaped_3d_array.reshape(target_2d_rows, -1)
#     print(f'\n3. reshaped_2d_array {np.shape(reshaped_2d_array)}')
#     if verbose:
#         print(reshaped_2d_array)
#
#     # make dataframe and insert column for separation and stack (trimmed run/group)
#     trimmed_df = pd.DataFrame(reshaped_2d_array, columns=dur_name_list)
#     stack_col_vals = np.repeat(np.arange(target_3d_depth), rows_3d)
#     sep_col_vals = sep_list * target_3d_depth
#     trimmed_df.insert(0, 'stack', stack_col_vals)
#     trimmed_df.insert(1, reference_col, sep_col_vals)
#     print(f'\ntrimmed_df {trimmed_df.shape}:\n{trimmed_df}')
#
#     print(f'trimmed {trim_from_ends} highest and lowest values ({2 * trim_from_ends} in total) from each of the '
#           f'{datapoints_per_cond} datapoints so there are now '
#           f'{target_3d_depth} datapoints for each of the '
#           f'{rows_3d} x {cols_all} conditions.')
#
#     print('\n*** finished trim_high_n_low() ***')
#
#     return trimmed_df
#
#
# def make_long_df(wide_df,
#                  cols_to_keep=['congruent', 'separation'],
#                  cols_to_change=['dur_1', 'dur_4', 'dur_6'],
#                  cols_to_change_show='probeLum',
#                  new_col_name='probe_dur', strip_from_cols='dur_', verbose=True):
#     """
#     Function to convert wide-form_df to long-form_df.  e.g., if there are several
#     columns showing durs (cols_to_change), this puts them all into one column (new_col_name).
#
#     :param wide_df: dataframe to be changed.
#     :param cols_to_keep: Columns to use for indexing (e.g., ['congruent', 'separation'...etc]).
#     :param cols_to_change: List of columns showing data at different levels e.g., [dur_1, dur_4, dur_6...etc].
#     :param cols_to_change_show: What is being measured in repeated cols, e.g., probeLum.
#     :param new_col_name: name for new col describing levels e.g. dur
#     :param strip_from_cols: string to strip from col names when for new cols.
#         e.g., if strip_from_cols='dur_', then [dur_1, dur_4, dur_6] becomes [1, 4, 6].
#     :param verbose: if true, prints progress to screen.
#
#     :return: long_df
#     """
#     print("\n*** running make_long_df() ***\n")
#
#     new_col_names = cols_to_keep + [new_col_name] + [cols_to_change_show]
#
#     # make longform data
#     if verbose:
#         print(f"preparing to loop through: {cols_to_change}")
#     long_list = []
#     for this_col in cols_to_change:
#         this_df_cols = cols_to_keep + [this_col]
#         this_df = wide_df[this_df_cols]
#
#         # strip text from col names, try lower/upper case if strip_from_cols not found.
#         if strip_from_cols not in [False, None]:
#             if strip_from_cols in this_col:
#                 this_col = this_col.strip(strip_from_cols)
#             elif strip_from_cols.lower() in this_col:
#                 this_col = this_col.strip(strip_from_cols.lower())
#             elif strip_from_cols.upper() in this_col:
#                 this_col = this_col.strip(strip_from_cols.upper())
#             else:
#                 raise ValueError(f"can't strip {strip_from_cols} from {this_col}")
#
#         this_df.insert(len(cols_to_keep), new_col_name, [this_col] * len(this_df.index))
#         this_df.columns = new_col_names
#         long_list.append(this_df)
#
#     long_df = pd.concat(long_list)
#     long_df.reset_index(drop=True, inplace=True)
#     if verbose:
#         print(f'long_df:\n{long_df}')
#
#     print("\n*** finished make_long_df() ***\n")
#
#     return long_df
#
#
# def fig_colours(n_conditions, alternative_colours=False):
#     """
#     Use this to always get the same colours in the same order with no fuss.
#     :param n_conditions: number of different colours - use 256 for colourmap
#         (e.g., for heatmaps or something where colours are used in continuous manner)
#     :param alternative_colours: a second pallet of alternative colours.
#     :return: a colour pallet
#     """
#
#     use_colours = 'colorblind'
#     if alternative_colours:
#         use_colours = 'husl'
#
#     if n_conditions > 20:
#         use_colour = 'spectral'
#     elif 10 < n_conditions < 21:
#         use_colours = 'tab20'
#
#     use_cmap = False
#
#     my_colours = sns.color_palette(palette=use_colours, n_colors=n_conditions, as_cmap=use_cmap)
#     sns.set_palette(palette=use_colours, n_colors=n_conditions)
#
#     return my_colours
#
#
# def multi_plot_shape(n_figs, min_rows=1):
#     """
#     Function to make multi-plot figure with right number of rows and cols,
#     to fit n_figs, but with the smallest shape for landscape.
#     :param n_figs: Number of plots I need to make.
#     :param min_rows: Minimum number of rows (sometimes won't work with just 1)
#
#     :return: n_rows, n_cols
#     """
#     n_rows = 1
#     if n_figs > 3:
#         n_rows = 2
#     if n_figs > 8:
#         n_rows = 3
#     if n_figs > 12:
#         n_rows = 4
#
#     if n_rows < min_rows:
#         n_rows = min_rows
#
#     td = n_figs // n_rows
#     mod = n_figs % n_rows
#     n_cols = td + mod
#
#     # there are some weird results, this helps catch 11, 14, 15 etc from going mad.
#     if n_rows * (n_cols - 1) > n_figs:
#         n_cols = n_cols - 1
#         if n_rows * (n_cols - 1) > n_figs:
#             n_cols = n_cols - 1
#     # plots = n_rows * n_cols
#     # print(f"{n_figs}: n_rows: {n_rows}, td: {td}, mod: {mod}, n_cols: {n_cols}, plots: {plots}, ")
#
#     if n_figs > 20:
#         raise ValueError('too many plots for one page!')
#
#     return n_rows, n_cols
#
#
# def simple_line_plot(indexed_df, fig_title=None, legend_title=None,
#                      x_tick_vals=None, x_tick_labels=None,
#                      x_axis=None, y_axis=None,
#                      log_x=False, log_y=False,
#                      save_as=None):
#     """
#     Function to plot a simple line plot.  No error bars.
#     :param indexed_df: DF where index col is 'separation' or 'stair_names' etc.
#     :param fig_title: Title for figure
#     :param legend_title: Title for legend
#     :param x_tick_vals: Values for x-ticks
#     :param x_tick_labels: Labels for x ticks
#     :param x_axis: Label for x-axis
#     :param y_axis: Label for y-axis
#     :param log_x: Make x-axis log scale
#     :param log_y: Make y-axis log scale
#     :param save_as: Full path (including name) to save to
#     :return: Figure
#     """
#     fig, ax = plt.subplots(figsize=(10, 6))
#     sns.lineplot(data=indexed_df, markers=True, dashes=False, ax=ax)
#     if fig_title is not None:
#         plt.title(fig_title)
#     if legend_title is not None:
#         plt.legend(title=legend_title)
#     if x_tick_vals is not None:
#         ax.set_xticks(x_tick_vals)
#     if x_tick_labels is not None:
#         ax.set_xticklabels(x_tick_labels)
#         if -18 in x_tick_labels:
#             # add dotted line at zero
#             ax.axvline(x=5.5, linestyle="-.", color='lightgrey')
#     if log_x:
#         ax.set(xscale="log")
#         x_axis = f'log {x_axis}'
#     if log_y:
#         ax.set(yscale="log")
#         y_axis = f'log {y_axis}'
#     if x_axis is not None:
#         ax.set_xlabel(x_axis)
#     if y_axis is not None:
#         ax.set_ylabel(y_axis)
#     if save_as is not None:
#         plt.savefig(save_as)
#     return fig
#
#
# def run_thr_plot(thr_df, x_col='separation', y_col='dur_0', hue_col='cond',
#                  x_ticks_vals=None, x_tick_names=None,
#                  x_axis_label='Probe cond (separation)',
#                  y_axis_label='Probe Luminance',
#                  fig_title='Ricco_v2: probe cond vs thr', save_as=None):
#     """
#     Function to make a simple plot from one run showing lineplots for circles, lines and 2probe data.
#     Single threshold values so no error bars.
#
#     :param thr_df: dataframe from one run
#     :param x_col: column to use for x vals
#     :param y_col: column to use for y vals
#     :param hue_col: column to use for hue (different coloured lines on plot)
#     :param x_ticks_vals: values to place on x-axis ticks
#     :param x_tick_names: labels for x-tick values
#     :param x_axis_label: x-axis label
#     :param y_axis_label: y-axis label
#     :param fig_title: figure title
#     :param save_as: path and filename to save to
#     :return: figure
#     """
#     print('*** running run_thr_plot (x=ordinal, y=thr) ***')
#     fig, ax = plt.subplots(figsize=(10, 6))
#     print(f'thr_df:\n{thr_df}')
#     sns.lineplot(data=thr_df, x=x_col, y=y_col, hue=hue_col, marker='o')
#     if x_ticks_vals is not None:
#         ax.set_xticks(x_ticks_vals)
#     if x_tick_names is not None:
#         ax.set_xticklabels(x_tick_names)
#     ax.set_xlabel(x_axis_label)
#     ax.set_ylabel(y_axis_label)
#     plt.title(fig_title)
#     if save_as:
#         plt.savefig(save_as)
#     print('*** finished run_thr_plot ***\n')
#     return fig
#
#
# def simple_log_log_plot(thr_df, x_col='area_deg', y_col='weber_thr', hue_col='cond',
#                         x_ticks_vals=None, x_tick_names=None,
#                         x_axis_label='log(area_deg)',
#                         y_axis_label='log(∆ threshold)',
#                         fig_title='Ricco_v2: log(area_deg) v log(thr)',
#                         show_neg1slope=True,
#                         save_as=None):
#     """
#     Function to make a simple plot from one run showing lineplots for circles, lines and 2probe data.
#     Data is plotted on log-log axis (log(∆thr) and log(area_deg)).
#     Single threshold values so no error bars.
#
#     :param thr_df: dataframe from one run
#     :param x_col: column to use for x vals
#     :param y_col: column to use for y vals
#     :param hue_col: column to use for hue (different coloured lines on plot)
#     :param x_ticks_vals: values to place on x-axis ticks
#     :param x_tick_names: labels for x-tick values
#     :param x_axis_label: x-axis label
#     :param y_axis_label: y-axis label
#     :param fig_title: figure title
#     :param show_neg1slope: If True, plots a line with slope=-1 starting from
#             first datapoint of circles.
#     :param save_as: path and filename to save to
#     :return: figure
#     """
#     print('\n*** running simple_log_log_plot (x=log(area_deg), y=log(∆thr)) ***')
#     print(f'thr_df:\n{thr_df}')
#     fig, ax = plt.subplots(figsize=(6, 6))
#     sns.lineplot(data=thr_df, x=x_col, y=y_col, hue=hue_col, marker='o', ax=ax)
#     if x_ticks_vals:
#         ax.set_xticks(x_ticks_vals)
#     if x_tick_names:
#         ax.set_xticklabels(x_tick_names)
#     ax.set_xlabel(x_axis_label)
#     ax.set_ylabel(y_axis_label)
#
#     # set scale for axes (same on each)
#     x_min = thr_df[x_col].min() * .9
#     x_max = thr_df[x_col].max() * 1.1
#     x_ratio = x_max / x_min
#     y_min = thr_df[y_col].min() * .9
#     y_max = thr_df[y_col].max() * 1.1
#     y_ratio = y_max / y_min
#     largest_diff = max([x_ratio, y_ratio])
#     axis_range = 10 ** math.ceil(math.log10(largest_diff))
#
#     ax.set(xlim=(x_min, x_min * axis_range), ylim=(y_min, y_min * axis_range))
#     ax.set(xscale="log", yscale="log")
#
#     # add guideline with slope of -1 which crosses through the circles 1probe weber_thr value.
#     if show_neg1slope:
#         if x_col == 'area_deg':
#             if '-1_circles' in thr_df['stair_names'].unique():
#                 start_point = '-1_circles'
#             elif '-1_lines' in thr_df['stair_names'].unique():
#                 start_point = '-1_lines'
#             slope_start_x = thr_df.loc[thr_df['stair_names'] == start_point, x_col].item()
#             slope_start_y = thr_df.loc[thr_df['stair_names'] == start_point, y_col].item()
#         elif x_col == 'dur_ms':
#             slope_start_x = thr_df.iloc[0]['dur_ms']
#             slope_start_y = thr_df.iloc[0]['weber_thr']
#         print(f'slope_start_x: {slope_start_x}')
#         print(f'slope_start_y: {slope_start_y}')
#         ax.plot([slope_start_x, slope_start_x * 100], [slope_start_y, slope_start_y / 100], c='r',
#                 label='-1 slope', linestyle='dashed')
#     ax.legend()
#     plt.title(fig_title)
#     if save_as:
#         plt.savefig(save_as)
#     print('*** finished simple_log_log_plot ***')
#     return fig
#
#
# # # # all durs on one axis -
# # FIGURE 1 - shows one axis (x=separation (-18:18), y=probeLum) with multiple dur lines.
# # dotted line at zero to make batman more apparent.
# def plot_data_unsym_batman(pos_and_neg_sep_df,
#                            fig_title=None,
#                            save_path=None,
#                            save_name=None,
#                            dur_name_list=None,
#                            x_tick_values=None,
#                            x_tick_labels=None,
#                            verbose=True):
#     """
#     This plots a figure with one axis, x has separation values [-18, -6, -3, -2, -1, 0, 1, 2, 3, 6, 18],
#     Will plot all durs on the same axis as lineplots.
#
#     :param pos_and_neg_sep_df: Full dataframe to use for values
#     :param fig_title: default=None.  Pass a string to add as a title.
#     :param save_path: default=None.  Path to dir to save fig
#     :param save_name: default=None.  name to save fig
#     :param dur_name_list: default=NONE: will use defaults, or pass list of names for legend.
#     :param x_tick_values: default=NONE: will use defaults, or pass list of names for x-axis positions.
#     :param x_tick_labels: default=NONE: will use defaults, or pass list of names for x_axis labels.
#     :param verbose: default: True. Won't print anything to screen if set to false.
#
#     :return: plot
#     """
#     if verbose:
#         print("\n*** running plot_data_unsym_batman() ***")
#         print(f"\npos_and_neg_sep_df:\n{pos_and_neg_sep_df}")
#         print(f"\nx_tick_values: {x_tick_values}")
#         print(f"x_tick_labels: {x_tick_labels}")
#
#     # get plot details
#     if dur_name_list is None:
#         raise ValueError('please pass an dur_name_list')
#     if verbose:
#         print(f'dur_name_list: {dur_name_list}')
#
#     if x_tick_values is None:
#         x_tick_values = [-18, -6, -3, -2, -1, -.1, 0, 1, 2, 3, 6, 18]
#     if x_tick_labels is None:
#         x_tick_labels = [-18, -6, -3, -2, -1, '-0', 0, 1, 2, 3, 6, 18]
#
#     # make fig1
#     fig, ax = plt.subplots(figsize=(10, 6))
#
#     # line plot for main durs
#     sns.lineplot(data=pos_and_neg_sep_df, markers=True, dashes=False, ax=ax)
#     ax.axvline(x=5.5, linestyle="-.", color='lightgrey')
#
#     # decorate plot
#     if x_tick_values is not None:
#         ax.set_xticks(x_tick_values)
#     if x_tick_labels is not None:
#         ax.set_xticks(x_tick_values)
#         ax.set_xticklabels(x_tick_labels)
#     ax.set_xlabel('Probe separation in diagonal pixels')
#     ax.set_ylabel('Probe Luminance')
#
#     ax.legend(labels=dur_name_list, title='probe_dur',
#               shadow=True,
#               # place lower left corner of legend at specified location.
#               loc='lower left', bbox_to_anchor=(0.96, 0.5))
#
#     if fig_title is not None:
#         plt.title(fig_title)
#
#     # save plot
#     if save_path is not None:
#         if save_name is not None:
#             plt.savefig(f'{save_path}{os.sep}{save_name}')
#
#     if verbose:
#         print("\n*** finished plot_data_unsym_batman() ***\n")
#     return fig
#
#
# def plot_runs_ave_w_errors(fig_df, error_df,
#                            jitter=True, error_caps=False, alt_colours=False,
#                            legend_names=None,
#                            x_tick_vals=None,
#                            x_tick_labels=None,
#                            even_spaced_x=False,
#                            fixed_y_range=False,
#                            x_axis_label=None,
#                            y_axis_label=None,
#                            log_log_axes=False,
#                            neg1_slope=False,
#                            fig_title=None, save_name=None, save_path=None,
#                            verbose=True):
#     """
#     Calculate and plot the mean and error estimates (y-axis) at each separation values (x-axis).
#     Separate line for each dur.  Error bar values taken from separate error_df.
#
#     :param fig_df: dataframe to build plot from.  Expects fig_df in the form:
#         separation as index, durs as columns.
#     :param error_df: dataframe of same shape as fig_df, but contains error values.
#     :param jitter: Jitter x_axis values so points don't overlap.
#     :param error_caps: caps on error bars for more easy reading.
#     :param alt_colours: Use different set of colours to normal (e.g., if dur on
#         x-axis and lines for each separation).
#     :param legend_names: Names of different lines (e.g., dur names).
#     :param x_tick_vals: Positions on x-axis.
#     :param x_tick_labels: labels for x-axis.
#     :param even_spaced_x: If True, x-ticks are evenly spaced,
#         if False they will be spaced according to numeric value (e.g., 0, 1, 2, 3, 6, 18).
#     :param fixed_y_range: default=False. If True, it uses full range of y values
#         (e.g., 0:110) or can pass a tuple to set y_limits.
#     :param x_axis_label: Label for x-axis.  If None passed, will use 'Probe separation in diagonal pixels'.
#     :param y_axis_label: Label for y-axis.  If None passed, will use 'Probe Luminance'.
#     :param log_log_axes: If True, both axes are in log scale, else in normal scale.
#     :param neg_1_slope: If True, adds a reference line with slope=-1.
#     :param fig_title: Title for figure.
#     :param save_name: filename of plot.
#     :param save_path: path to folder where plots will be saved.
#     :param verbose: print progress to screen.
#
#     :return: figure
#     """
#     print('\n*** running plot_runs_ave_w_errors() ***\n')
#
#     if verbose:
#         print(f'fig_df:\n{fig_df}')
#         print(f'\nerror_df:\n{error_df}')
#
#     # get names for legend (e.g., different lines)
#     column_names = fig_df.columns.to_list()
#
#     if legend_names is None:
#         legend_names = column_names
#     if verbose:
#         print(f'\nColumn and Legend names:')
#         for a, b in zip(column_names, legend_names):
#             print(f"{a}\t=>\t{b}\tmatch: {bool(a == b)}")
#
#     if x_tick_vals is None:
#         x_tick_vals = fig_df.index
#
#     # for evenly spaced items on x_axis
#     if even_spaced_x:
#         x_tick_vals = list(range(len(x_tick_vals)))
#
#     # adding jitter works well if df.index are all int
#     # need to set it up to use x_tick_vals if df.index is not all int or float
#     check_idx_num = all(isinstance(x, (int, float)) for x in fig_df.index)
#     print(f'check_idx_num: {check_idx_num}')
#
#     check_x_val_num = all(isinstance(x, (int, float)) for x in x_tick_vals)
#     print(f'check_x_val_num: {check_x_val_num}')
#
#     if jitter:
#         if not all(isinstance(x, (int, float)) for x in x_tick_vals):
#             x_tick_vals = list(range(len(x_tick_vals)))
#
#     # get number of locations for jitter list
#     n_pos_sep = len(fig_df.index.to_list())
#
#     jit_max = 0
#     if jitter:
#         jit_max = .2
#         if type(jitter) in [float, np.float]:
#             jit_max = jitter
#
#     cap_size = 0
#     if error_caps:
#         cap_size = 5
#
#     # set colour palette
#     my_colours = fig_colours(len(column_names), alternative_colours=alt_colours)
#
#     fig, ax = plt.subplots()
#
#     legend_handles_list = []
#
#     for idx, name in enumerate(column_names):
#         # get rand float to add to x-axis for jitter
#         jitter_list = np.random.uniform(size=n_pos_sep, low=-jit_max, high=jit_max)
#         x_values = x_tick_vals + jitter_list
#
#         ax.errorbar(x=x_values, y=fig_df[name],
#                     yerr=error_df[name],
#                     marker='.', lw=2, elinewidth=.7,
#                     capsize=cap_size, color=my_colours[idx])
#
#         leg_handle = mlines.Line2D([], [], color=my_colours[idx], label=name,
#                                    marker='.', linewidth=.5, markersize=4)
#         legend_handles_list.append(leg_handle)
#
#     # decorate plot
#     if x_tick_vals is not None:
#         ax.set_xticks(x_tick_vals)
#     if x_tick_labels is not None:
#         ax.set_xticks(x_tick_vals)
#         ax.set_xticklabels(x_tick_labels)
#
#     if x_axis_label is None:
#         ax.set_xlabel('Probe separation in diagonal pixels')
#     else:
#         ax.set_xlabel(x_axis_label)
#
#     if y_axis_label is None:
#         ax.set_ylabel('Probe Luminance')
#     else:
#         ax.set_ylabel(y_axis_label)
#
#     if fixed_y_range:
#         ax.set_ylim([0, 110])
#         if type(fixed_y_range) in [tuple, list]:
#             ax.set_ylim([fixed_y_range[0], fixed_y_range[1]])
#
#     if log_log_axes:
#         ax.set(xscale="log", yscale="log")
#
#     if neg1_slope:
#         # add guideline with slope of -1 which crosses through the circles 1probe weber_thr value.
#         if 'circles' in column_names:
#             slope_start_x = fig_df.index[0][0]
#             slope_start_y = fig_df.iloc[0]['circles']
#         elif '1probe' in column_names:
#             slope_start_x = fig_df.index[0]
#             slope_start_y = fig_df.iloc[0]['1probe']
#         print(f'slope_start_x: {slope_start_x}')
#         print(f'slope_start_y: {slope_start_y}')
#         ax.plot([slope_start_x, slope_start_x * 100], [slope_start_y, slope_start_y / 100], c='r',
#                 label='-1 slope', linestyle='dashed')
#         leg_handle = mlines.Line2D([], [], color='r', label='-1 slope', linestyle='dashed',
#                                    # marker='.', linewidth=.5, markersize=4
#                                    )
#         legend_handles_list.append(leg_handle)
#
#     ax.legend(handles=legend_handles_list, fontsize=6,
#               framealpha=.5)
#
#     if fig_title is not None:
#         plt.title(fig_title)
#
#     if save_path is not None:
#         if save_name is not None:
#             plt.savefig(f'{save_path}{os.sep}{save_name}')
#
#     return fig
#
#
# def plot_w_errors_either_x_axis(wide_df, cols_to_keep=['congruent', 'separation'],
#                                 cols_to_change=['dur_1', 'dur_4', 'dur_6'],
#                                 cols_to_change_show='probeLum', new_col_name='probe_dur',
#                                 strip_from_cols='dur_',
#                                 x_axis='separation', y_axis='probeLum',
#                                 hue_var='probe_dur', style_var='congruent', style_order=[1, -1],
#                                 error_bars=False,
#                                 jitter=False,
#                                 log_scale=False,
#                                 even_spaced_x=False,
#                                 x_tick_vals=None,
#                                 fig_title=None,
#                                 fig_savename=None,
#                                 save_path=None,
#                                 verbose=True):
#     """
#     Function to plot line_plot with multiple lines.  This function works with a single dataset,
#     or with a df containing several datasets, in which case it plots the mean
#     with error bars at .68 confidence interval.
#     It will work with either separation on x-axis and different lines for each dur, or vice versa.
#     The first part of the function converts a wide dataframe to a long dataframe.
#
#     :param wide_df: Data to be plotted.
#     :param cols_to_keep: Variables that will be included in long dataframe.
#     :param cols_to_change: Columns containing different measurements of some
#         variable (e.g., dur_1, dur_4, dur_6...etc) that will be converted into
#         longform (e.g., dur: [1, 4, 6]).
#     :param cols_to_change_show: What is being measured in cols to change (e.g., probeLum; dependent variable).
#     :param new_col_name: What the cols to change describe (e.g., dur; independent variable).
#     :param strip_from_cols: string to remove if independent variables are to be
#         turned into numeric values (e.g., for dur_1, dur_4, dur_6, strip 'dur_' to get 1, 4,6).
#     :param x_axis: Variable to be shown along x-axis (e.g., separation or dur).
#     :param y_axis: Variable to be shown along y-axis (e.g., probeLum).
#     :param hue_var: Variable to be shown with different lines (e.g., dur or separation).
#     :param style_var: Addition variable to show with solid or dashed lines (e.g., congruent or incongruent).
#     :param style_order: Order of style var as displayed in df (e.g., [1, -1]).
#     :param error_bars: True or false, whether to display error bars.
#     :param jitter: Whether to jitter items on x-axis to make easier to read.
#         Can be True, False or float for amount of jitter in relation to x-axis values.
#     :param log_scale: Put axes onto log scale.
#     :param even_spaced_x: Whether to evenly space ticks on x-axis.
#         For example to make the left side of log-scale-like x-values easier to read.
#     :param x_tick_vals: Values/labels for x-axis.  Can be string, int or float.
#     :param fig_title: Title for figure.
#     :param fig_savename: Save name for figure.
#     :param save_path: Save path for figure.
#     :param verbose: Whether to print progress to screen.
#
#     :return: figure
#     """
#     print('\n*** running plot_w_errors_either_x_axis() ***\n')
#
#     # convert wide df to long
#     long_df = make_long_df(wide_df=wide_df, cols_to_keep=cols_to_keep,
#                            cols_to_change=cols_to_change, cols_to_change_show=cols_to_change_show,
#                            new_col_name=new_col_name, strip_from_cols=strip_from_cols, verbose=verbose)
#
#     # data_for_x to use for x_axis data, whilst keeping original values list (x_axis)
#     data_for_x = x_axis
#
#     if log_scale:
#         even_spaced_x = False
#
#     # for evenly spaced items on x_axis
#     if even_spaced_x:
#         orig_x_vals = x_tick_vals  # sorted(x_tick_vals)
#         new_x_vals = list(range(len(orig_x_vals)))
#
#         # check if x_tick_vals refer to values in df, if not, use df values.
#         if list(long_df[x_axis])[0] in orig_x_vals:
#             x_space_dict = dict(zip(orig_x_vals, new_x_vals))
#         else:
#             print("warning: x_tick_vals don't appear in long_df")
#             found_x_vals = sorted(set(list(long_df[x_axis])))
#             print(f'found_x_vals (from df): {found_x_vals}')
#             x_space_dict = dict(zip(found_x_vals, new_x_vals))
#
#         # add column with new evenly spaced x-values, relating to original x_values
#         spaced_x = [x_space_dict[i] for i in list(long_df[x_axis])]
#         long_df.insert(0, 'spaced_x', spaced_x)
#         data_for_x = 'spaced_x'
#         if verbose:
#             print(f'orig_x_vals: {orig_x_vals}')
#             print(f'new_x_vals: {new_x_vals}')
#             print(f'x_space_dict: {x_space_dict}')
#
#     # for jittering values on x-axis
#     if jitter:
#         jitter_keys = list(long_df[hue_var])
#         n_jitter = len(jitter_keys)
#         jit_max = .2
#         if type(jitter) in [float, np.float]:
#             jit_max = jitter
#
#         # get rand float to add to x-axis values for jitter
#         jitter_vals = np.random.uniform(size=n_jitter, low=-jit_max, high=jit_max)
#         jitter_dict = dict(zip(jitter_keys, jitter_vals))
#         jitter_x = [a + jitter_dict[b] for a, b in zip(list(long_df[data_for_x]), list(long_df[hue_var]))]
#         long_df.insert(0, 'jitter_x', jitter_x)
#         data_for_x = 'jitter_x'
#
#     conf_interval = None
#     if error_bars:
#         conf_interval = 68
#
#     if verbose:
#         print(f'long_df:\n{long_df}')
#         print(f'error_bars: {error_bars}')
#         print(f'conf_interval: {conf_interval}')
#         print(f'x_tick_vals: {x_tick_vals}')
#
#     # initialize plot
#     my_colours = fig_colours(n_conditions=len(set(list(long_df[hue_var]))))
#     fig, ax = plt.subplots(figsize=(10, 6))
#
#     # with error bards for d_averages example
#     sns.lineplot(data=long_df, x=data_for_x, y=y_axis, hue=hue_var,
#                  style=style_var, style_order=style_order,
#                  estimator='mean',
#                  ci=conf_interval, err_style='bars', err_kws={'elinewidth': .7, 'capsize': 5},
#                  palette=my_colours, ax=ax)
#
#     if log_scale:
#         ax.set_xscale('log')
#         ax.set_yscale('log')
#     elif even_spaced_x:
#         ax.set_xticks(new_x_vals)
#         ax.set_xticklabels(orig_x_vals)
#     else:
#         ax.set_xticklabels(x_tick_vals)
#
#     plt.xlabel(x_axis)
#     plt.title(fig_title)
#
#     # Change legend labels for congruent and incongruent data
#     handles, labels = ax.get_legend_handles_labels()
#     ax.legend(handles=handles, labels=labels[:-2] + ['True', 'False'])
#
#     print(f'test save path:\n{save_path}\n{fig_savename}')
#     plt.savefig(os.path.join(save_path, fig_savename))
#
#     print('\n*** finished plot_w_errors_either_x_axis() ***\n')
#
#     return fig
#
#
# def plot_diff(ave_thr_df, stair_names_col='stair_names', fig_title=None, save_path=None, save_name=None,
#               x_axis_dur=False, verbose=True):
#     """
#     Function to plot the difference between congruent and incongruent conditions.
#     :param ave_thr_df: Dataframe to use to get differences
#     :param stair_names_col: Column name containing separation and congruence values.
#     :param fig_title: Title for fig
#     :param save_path: path to save fig to
#     :param save_name: name to save fig
#     :param x_axis_dur: If False, with have Separation on x-axis with different lines for each dur.
#                         If True, will have dur on x-axis and diffferent lines for each Separation.
#     :param verbose: Prints progress to screen
#     :return:
#     """
#     print('*** running plot_diff() ***')
#
#     # if stair_names_col is set as index, move it to regular column and add standard index
#     if ave_thr_df.index.name == stair_names_col:
#         ave_thr_df.reset_index(drop=False, inplace=True)
#     if verbose:
#         print(f'ave_thr_df:\n{ave_thr_df}')
#
#     # get rows to slice for each df to be in ascending order
#     # if stair_names_col in list(ave_thr_df.columns):
#     stair0_rows = sorted(ave_thr_df.index[ave_thr_df['stair_names'] >= 0].tolist(), reverse=True)
#     stair1_rows = sorted(ave_thr_df.index[ave_thr_df['stair_names'] < 0].tolist(), reverse=True)
#     # else:
#     #     stair0_rows = sorted(ave_thr_df.index[ave_thr_df.index >= 0].tolist())
#     #     stair1_rows = sorted(ave_thr_df.index[ave_thr_df.index < 0].tolist(), reverse=True)
#     if verbose:
#         print(f'\nstair0_rows: {stair0_rows}')
#         print(f'stair1_rows: {stair1_rows}')
#
#     # slice rows for cong and incong df
#     stair0_df = ave_thr_df.iloc[stair0_rows, :]
#     stair1_df = ave_thr_df.iloc[stair1_rows, :]
#
#     pos_sep_list = [int(i) for i in list(sorted(stair0_df['stair_names'].tolist()))]
#     stair0_df.reset_index(drop=True, inplace=True)
#     stair1_df.reset_index(drop=True, inplace=True)
#     if verbose:
#         print(f'\nstair0_df: {stair0_df.shape}\n{stair0_df}')
#         print(f'\nstair1_df: {stair1_df.shape}\n{stair1_df}')
#         print(f'\npos_sep_list: {pos_sep_list}')
#
#     # subtract one from the other
#     diff_df = stair0_df - stair1_df
#     diff_df.drop('stair_names', inplace=True, axis=1)
#
#     if x_axis_dur:
#         diff_df = diff_df.T
#         diff_df.columns = pos_sep_list
#         dur_names_list = list(diff_df.index)
#         x_tick_labels = dur_names_list
#         x_axis_label = 'probe_dur'
#         legend_title = 'Separation'
#     else:
#         x_tick_labels = pos_sep_list
#         x_axis_label = 'Separation'
#         legend_title = 'probe_dur'
#
#     if verbose:
#         print(f'\ndiff_df:\n{diff_df}')
#         print(f'\nx_axis_label: {x_axis_label}')
#
#     # make plot
#     fig, ax = plt.subplots(figsize=(10, 6))
#     sns.lineplot(data=diff_df, ax=ax)
#
#     # decorate plot
#     if fig_title:
#         plt.title(fig_title)
#     plt.axhline(y=0, color='lightgrey', linestyle='dashed')
#     ax.set_xticks(list(range(len(x_tick_labels))))
#     ax.set_xticklabels(x_tick_labels)
#     plt.xlabel(x_axis_label)
#     plt.ylabel("Difference in probeLum (cong-incong)")
#     ax.legend(title=legend_title)
#
#     if save_name:
#         plt.savefig(os.path.join(save_path, save_name))
#
#     print('*** finished plot_diff() ***')
#
#     return fig
#
#
# # # # 8 batman plots
# # this is a figure with one axis per dur, showing neg and pos sep (e.g., -18:18)
# def multi_batman_plots(mean_df, thr1_df, thr2_df,
#                        fig_title=None, dur_name_list=None,
#                        x_tick_vals=None, x_tick_labels=None,
#                        sym_sep_diff_list=None,
#                        save_path=None, save_name=None,
#                        verbose=True):
#     """
#     From array make separate batman plots for
#     :param mean_df: df of values for mean thr
#     :param thr1_df: df of values from stair 1 (e.g., probe_jump inwards)
#     :param thr2_df: df of values for stair 2 (e.g., probe_jump outwards)
#     :param fig_title: title for figure or None
#     :param dur_name_list: If None, will use default setting, or pass list of
#         names for legend
#     :param x_tick_vals: If None, will use default setting, or pass list of
#         values for x_axis
#     :param x_tick_labels: If None, will use default setting, or pass list of
#         labels for x_axis
#     :param sym_sep_diff_list: list of differences between thr1&thr2 to be added
#         as text to figure
#     :param save_path: default=None.  Path to dir to save fig
#     :param save_name: default=None.  name to save fig
#     :param verbose: If True, print info to screen.
#
#     :return: Batman Plot
#     """
#     print("\n*** running multi_batman_plots() ***")
#
#     # get plot info
#     if dur_name_list is None:
#         dur_name_list = list(mean_df.columns[2:])
#     if x_tick_vals is None:
#         x_tick_vals = list(mean_df['separation'])
#     if x_tick_labels is None:
#         x_tick_labels = list(mean_df['separation'])
#     if verbose:
#         print(f'dur_name_list: {dur_name_list}')
#         print(f'x_tick_vals: {x_tick_vals}')
#         print(f'x_tick_labels: {x_tick_labels}')
#
#     # make plots
#     my_colours = fig_colours(len(dur_name_list))
#     n_rows, n_cols = multi_plot_shape(len(dur_name_list), min_rows=2)
#     fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12, 6))
#     print(f'\nplotting {n_rows} rows and {n_cols} cols for {len(axes)} plots')
#
#     if fig_title is not None:
#         fig.suptitle(fig_title)
#
#     ax_counter = 0
#     # loop through the eight axes
#     for row_idx, row in enumerate(axes):
#         print(f'\nrow_idx: {row_idx}, row: {row} type(row): {type(row)}')
#
#         # if there are multiple durs
#         if isinstance(row, np.ndarray):
#             for col_idx, ax in enumerate(row):
#                 print(f'col_idx: {col_idx}, ax: {ax}')
#
#                 if ax_counter < len(dur_name_list):
#
#                     print(f'\t{ax_counter}. dur_name_list[ax_counter]: {dur_name_list[ax_counter]}')
#
#                     # mean threshold from CW and CCW probe jump direction
#                     sns.lineplot(ax=axes[row_idx, col_idx], data=mean_df,
#                                  x='separation', y=dur_name_list[ax_counter],
#                                  color=my_colours[ax_counter],
#                                  linewidth=.5,
#                                  markers=True)
#
#                     sns.lineplot(ax=axes[row_idx, col_idx], data=thr1_df,
#                                  x='separation', y=dur_name_list[ax_counter],
#                                  color=my_colours[ax_counter],
#                                  linestyle="dashed",
#                                  marker="v")
#
#                     sns.lineplot(ax=axes[row_idx, col_idx], data=thr2_df,
#                                  x='separation', y=dur_name_list[ax_counter],
#                                  color=my_colours[ax_counter],
#                                  linestyle="dotted",
#                                  marker="o")
#
#                     ax.set_title(dur_name_list[ax_counter])
#                     ax.set_xticks(x_tick_vals)
#                     ax.set_xticklabels(x_tick_labels)
#                     ax.xaxis.set_tick_params(labelsize=6)
#
#                     if row_idx == 1:
#                         ax.set_xlabel('Probe separation (pixels)')
#                     else:
#                         ax.xaxis.label.set_vdurble(False)
#
#                     if col_idx == 0:
#                         ax.set_ylabel('Probe Luminance')
#                     else:
#                         ax.yaxis.label.set_vdurble(False)
#
#                     if sym_sep_diff_list is not None:
#                         ax.text(x=0.4, y=0.8, s=round(sym_sep_diff_list[ax_counter], 2),
#                                 # needs transform to appear with rest of plot.
#                                 transform=ax.transAxes, fontsize=12)
#
#                     # artist for legend
#                     st1 = mlines.Line2D([], [], color=my_colours[ax_counter],
#                                         marker='v',
#                                         linestyle="dashed",
#                                         markersize=4, label='Congruent')
#                     st2 = mlines.Line2D([], [], color=my_colours[ax_counter],
#                                         marker='o',
#                                         linestyle="dotted",
#                                         markersize=4, label='Incongruent')
#                     mean_line = mlines.Line2D([], [], color=my_colours[ax_counter],
#                                               marker=None,
#                                               linewidth=.5,
#                                               label='mean')
#                     ax.legend(handles=[st1, st2, mean_line], fontsize=6)
#
#                     ax_counter += 1
#                 else:
#                     fig.delaxes(ax=axes[row_idx, col_idx])
#
#         # if there is only 1 dur in this row
#         else:
#             print(f'last plot\n'
#                   f'{row_idx}. dur_name_list[row_idx]: {dur_name_list[row_idx]}')
#
#             ax = row
#             # mean threshold from CW and CCW probe jump direction
#             sns.lineplot(ax=axes[row_idx], data=mean_df,
#                          x='separation', y=dur_name_list[row_idx],
#                          color=my_colours[row_idx],
#                          linewidth=.5,
#                          markers=True)
#
#             sns.lineplot(ax=axes[row_idx], data=thr1_df,
#                          x='separation', y=dur_name_list[row_idx],
#                          color=my_colours[row_idx], linestyle="dashed",
#                          marker="v")
#
#             sns.lineplot(ax=axes[row_idx], data=thr2_df,
#                          x='separation', y=dur_name_list[row_idx],
#                          color=my_colours[row_idx], linestyle="dotted",
#                          marker="o")
#
#             ax.set_title(dur_name_list[row_idx])
#             ax.set_xticks(x_tick_vals)
#             ax.set_xticklabels(x_tick_labels)
#             ax.xaxis.set_tick_params(labelsize=6)
#
#             ax.set_xlabel('Probe separation (pixels)')
#             ax.set_ylabel('Probe Luminance')
#
#             if sym_sep_diff_list is not None:
#                 ax.text(x=0.4, y=0.8, s=round(sym_sep_diff_list[row_idx], 2),
#                         # needs transform to appear with rest of plot.
#                         transform=ax.transAxes, fontsize=12)
#
#             # artist for legend
#             st1 = mlines.Line2D([], [], color=my_colours[row_idx],
#                                 marker='v',
#                                 linestyle="dashed",
#                                 markersize=4, label='Congruent')
#             st2 = mlines.Line2D([], [], color=my_colours[row_idx],
#                                 marker='o',
#                                 linestyle="dotted",
#                                 markersize=4, label='Incongruent')
#             mean_line = mlines.Line2D([], [], color=my_colours[row_idx],
#                                       marker=None,
#                                       linewidth=.5,
#                                       label='mean')
#             ax.legend(handles=[st1, st2, mean_line], fontsize=6)
#
#             print(f'ax_counter: {ax_counter}, len(dur_name_list): {len(dur_name_list)}')
#             if ax_counter + 1 == len(dur_name_list):
#                 print(f'idiot check, no more plots to make here')
#                 # continue
#                 break
#
#     plt.tight_layout()
#
#     if save_path is not None:
#         if save_name is not None:
#             plt.savefig(f'{save_path}{os.sep}{save_name}')
#
#     print("\n*** finished multi_batman_plots() ***")
#
#     return fig
#
#
# def multi_pos_sep_per_dur(ave_thr_df, error_df,
#                           stair_names_col='stair_names',
#                           even_spaced_x=True, error_caps=True,
#                           fig_title=None,
#                           save_path=None, save_name=None,
#                           verbose=True):
#     """
#     Function to plot multi-plot for comparing cong and incong for each dur.
#
#     :param ave_thr_df: dataframe to analyse containing mean thresholds
#     :param error_df: dataframe containing error values
#     :param stair_names_col: name of column containing separation and congruent info
#     :param even_spaced_x: If true will evenly space ticks on x-axis.
#         If false will use values given which might not be evenly spaces (e.g., 1, 2, 3, 6, 18)
#     :param error_caps: Whether to add caps to error bars
#     :param fig_title: Title for page of figures
#     :param save_path: directory to save into
#     :param save_name: name of saved file
#     :param verbose: if Ture, will print progress to screen
#
#     :return: figure
#     """
#     print("\n*** running multi_pos_sep_per_dur() ***")
#
#     # if stair_names col is being used as the index, change stair_names to regular column and add index.
#     if ave_thr_df.index.name == stair_names_col:
#         # get the dataframe max and min values for y_axis limits.
#         max_thr = ave_thr_df.max().max() + 5
#         min_thr = ave_thr_df.min().min() - 5
#         ave_thr_df.reset_index(drop=False, inplace=True)
#         error_df.reset_index(drop=False, inplace=True)
#     else:
#         get_min_max = ave_thr_df.iloc[:, 1:]
#         max_thr = get_min_max.max().max() + 5
#         min_thr = get_min_max.min().min() - 5
#
#     if verbose:
#         print(f'ave_thr_df:\n{ave_thr_df}\n'
#               f'error_df:\n{error_df}')
#         print(f'max_thr: {max_thr}\nmin_thr: {min_thr}')
#
#     stair0_rows = sorted(ave_thr_df.index[ave_thr_df[stair_names_col] >= 0].tolist(), reverse=True)
#     stair1_rows = sorted(ave_thr_df.index[ave_thr_df[stair_names_col] < 0].tolist(), reverse=True)
#     if verbose:
#         print(f'\nstair0_rows: {stair0_rows}')
#         print(f'stair1_rows: {stair1_rows}')
#
#     # slice rows for cong and incong df
#     stair0_df = ave_thr_df.iloc[stair0_rows, :]
#     stair0_err_df = error_df.iloc[stair0_rows, :]
#     stair1_df = ave_thr_df.iloc[stair1_rows, :]
#     stair1_err_df = error_df.iloc[stair1_rows, :]
#
#     pos_sep_list = [int(i) for i in list(sorted(stair0_df[stair_names_col].tolist()))]
#     dur_names_list = list(stair0_df.columns)[1:]
#
#     stair0_df.reset_index(drop=True, inplace=True)
#     stair0_err_df.reset_index(drop=True, inplace=True)
#     stair1_df.reset_index(drop=True, inplace=True)
#     stair1_err_df.reset_index(drop=True, inplace=True)
#     if verbose:
#         print(f'\npos_sep_list: {pos_sep_list}')
#         print(f'dur_names_list: {dur_names_list}')
#         print(f'\nstair0_df: {stair0_df.shape}\n{stair0_df}')
#         print(f'\nstair0_err_df: {stair0_err_df.shape}\n{stair0_err_df}')
#         print(f'\nstair1_df: {stair1_df.shape}\n{stair1_df}')
#         print(f'\nstair1_err_df: {stair1_err_df.shape}\n{stair1_err_df}\n')
#
#     cap_size = 0
#     if error_caps:
#         cap_size = 5
#
#     if even_spaced_x:
#         x_values = list(range(len(pos_sep_list)))
#     else:
#         x_values = pos_sep_list
#
#     # make plots
#     my_colours = fig_colours(len(dur_names_list))
#     n_rows, n_cols = multi_plot_shape(len(dur_names_list), min_rows=2)
#     fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12, 6))
#     print(f'\nplotting {n_rows} rows and {n_cols} cols for {len(axes)} plots')
#
#     if fig_title is not None:
#         fig.suptitle(fig_title)
#
#     ax_counter = 0
#     # loop through the different axes
#     for row_idx, row in enumerate(axes):
#         print(f'row_idx: {row_idx}, type(row): {type(row)}, row: {row}')
#         # if there are multiple durs
#         if isinstance(row, np.ndarray):
#             print(f'type is {type(row)}')
#             for col_idx, ax in enumerate(row):
#                 if ax_counter < len(dur_names_list):
#
#                     this_dur = dur_names_list[ax_counter]
#
#                     ax.errorbar(x=x_values, y=stair0_df[this_dur],
#                                 yerr=stair0_err_df[this_dur],
#                                 marker=None, lw=2, elinewidth=.7,
#                                 capsize=cap_size,
#                                 color=my_colours[ax_counter])
#
#                     ax.errorbar(x=x_values, y=stair1_df[this_dur],
#                                 yerr=stair1_err_df[this_dur],
#                                 linestyle='dashed',
#                                 marker=None, lw=2, elinewidth=.7,
#                                 capsize=cap_size,
#                                 color=my_colours[ax_counter])
#
#                     ax.set_title(dur_names_list[ax_counter])
#                     if even_spaced_x:
#                         ax.set_xticks(list(range(len(pos_sep_list))))
#                     else:
#                         ax.set_xticks(pos_sep_list)
#                     ax.set_xticklabels(pos_sep_list)
#                     ax.set_ylim([min_thr, max_thr])
#
#                     if row_idx == 1:
#                         ax.set_xlabel('Probe separation (pixels)')
#                     else:
#                         ax.xaxis.label.set_vdurble(False)
#
#                     if col_idx == 0:
#                         ax.set_ylabel('Probe Luminance')
#                     else:
#                         ax.yaxis.label.set_vdurble(False)
#
#                     # artist for legend
#                     st1 = mlines.Line2D([], [], color=my_colours[ax_counter],
#                                         # marker='v',
#                                         linewidth=.5,
#                                         markersize=4, label='Congruent')
#                     st2 = mlines.Line2D([], [], color=my_colours[ax_counter],
#                                         # marker='o',
#                                         marker=None, linewidth=.5, linestyle="dotted",
#                                         markersize=4, label='Incongruent')
#
#                     ax.legend(handles=[st1, st2], fontsize=6)
#
#                     ax_counter += 1
#                 else:
#                     fig.delaxes(ax=axes[row_idx, col_idx])
#
#         # if there is only one dur in this row
#         else:
#             ax = row
#             this_dur = dur_names_list[row_idx]
#
#             ax.errorbar(x=x_values, y=stair0_df[this_dur],
#                         yerr=stair0_err_df[this_dur],
#                         marker=None, lw=2, elinewidth=.7,
#                         capsize=cap_size,
#                         color=my_colours[row_idx])
#
#             ax.errorbar(x=x_values, y=stair1_df[this_dur],
#                         yerr=stair1_err_df[this_dur],
#                         linestyle='dashed',
#                         marker=None, lw=2, elinewidth=.7,
#                         capsize=cap_size,
#                         color=my_colours[row_idx])
#
#             ax.set_title(dur_names_list[row_idx])
#             if even_spaced_x:
#                 ax.set_xticks(list(range(len(pos_sep_list))))
#             else:
#                 ax.set_xticks(pos_sep_list)
#             ax.set_xticklabels(pos_sep_list)
#             ax.set_ylim([min_thr, max_thr])
#
#             # if row_idx == 1:
#             ax.set_xlabel('Probe separation (pixels)')
#             # else:
#             #     ax.xaxis.label.set_vdurble(False)
#
#             # if col_idx == 0:
#             ax.set_ylabel('Probe Luminance')
#             # else:
#             #     ax.yaxis.label.set_vdurble(False)
#
#             # artist for legend
#             st1 = mlines.Line2D([], [], color=my_colours[row_idx],
#                                 # marker='v',
#                                 linewidth=.5,
#                                 markersize=4, label='Congruent')
#             st2 = mlines.Line2D([], [], color=my_colours[row_idx],
#                                 # marker='o',
#                                 marker=None, linewidth=.5, linestyle="dotted",
#                                 markersize=4, label='Incongruent')
#
#             ax.legend(handles=[st1, st2], fontsize=6)
#
#             print(f'ax_counter: {ax_counter}, len(dur_name_list): {len(dur_names_list)}')
#             if ax_counter + 1 == len(dur_names_list):
#                 print(f'idiot check, no more plots to make here')
#                 break
#
#     plt.tight_layout()
#
#     if save_path is not None:
#         if save_name is not None:
#             plt.savefig(f'{save_path}{os.sep}{save_name}')
#
#     print("\n*** finished multi_pos_sep_per_dur() ***")
#
#     return fig
#
#
# def plot_thr_heatmap(heatmap_df,
#                      x_tick_labels=None,
#                      y_tick_labels=None,
#                      fig_title=None,
#                      save_name=None,
#                      save_path=None,
#                      verbose=True):
#     """
#     Function for making a heatmap
#     :param heatmap_df: Expects dataframe with separation as index and durs as columns.
#     :param x_tick_labels: Labels for columns
#     :param y_tick_labels: Labels for rows
#     :param fig_title: Title for figure
#     :param save_name: name to save fig
#     :param save_path: location to save fig
#     :param verbose: if True, will prng progress to screen,
#
#     :return: Heatmap
#     """
#     print('\n*** running plot_thr_heatmap() ***\n')
#
#     if verbose:
#         print(f'heatmap_df:\n{heatmap_df}')
#
#     if x_tick_labels is None:
#         x_tick_labels = list(heatmap_df.columns)
#     if y_tick_labels is None:
#         y_tick_labels = list(heatmap_df.index)
#
#     # get mean of each column, then mean of those
#     mean_thr = float(np.mean(heatmap_df.mean()))
#     if verbose:
#         print(f'mean_val: {round(mean_thr, 2)}')
#
#     heatmap = sns.heatmap(data=heatmap_df,
#                           annot=True, center=mean_thr,
#                           cmap=sns.color_palette("Spectral", as_cmap=True),
#                           xticklabels=x_tick_labels, yticklabels=y_tick_labels)
#
#     # keep y ticks upright rather than rotates (90)
#     plt.yticks(rotation=0)
#
#     # add central mirror symmetry line
#     plt.axvline(x=6, color='grey', linestyle='dashed')
#
#     if 'probe_dur' in str(x_tick_labels[0]).upper():
#         heatmap.set_xlabel('probe_dur')
#         heatmap.set_ylabel('Separation')
#     else:
#         heatmap.set_xlabel('Separation')
#         heatmap.set_ylabel('probe_dur')
#
#     if fig_title is not None:
#         plt.title(fig_title)
#
#     if save_path is not None:
#         if save_name is not None:
#             plt.savefig(f'{save_path}{os.sep}{save_name}')
#
#     return heatmap


def a_data_extraction(p_name, run_dir, dur_list, save_all_data=True, verbose=True):
    """
    This script is a python version of Martin's first MATLAB analysis scripts, described below.

    a_data_extraction.m: Once a participant has completed a run of all durations,
        this script gets all their data into one file, and sorts each duration by stair.

    :param p_name: participant's name as used to save csv files.  e.g., if the
            file is .../nick1.csv, participant name is 'nick1'.
    :param run_dir: directory where durations folders are stored.
    :param dur_list: List of probe_duration values, may differ between experiments.
    :param save_all_data: If True, will save all_data_df as a xlsx.
    :param verbose: If True, will print progress to screen.

    :return: ALL_durations_sorted.xlsx: A pandas DataFrame with n xlsx file of all
        data for one run of all durations.
    """
    print("\n***running a_data_extraction()***\n")

    # get run name/number
    path, run = os.path.split(run_dir)
    if verbose:
        print(f"run: {run}")

    if dur_list is None:
        raise ValueError('Please pass a list of probe_duration values to identify directories containing data.')
    else:
        print(f'dur_list: {dur_list}')

    all_data = []

    # loop through durations in each run.
    for duration in dur_list:
        filepath = f'{run_dir}{os.path.sep}probeDur{duration}{os.path.sep}' \
                   f'{p_name}_output.csv'
        if verbose:
            print(f"filepath: {filepath}")

        if not os.path.isfile(filepath):
            raise FileNotFoundError(filepath)
            # print(f'\n\nFileNotFound: {filepath}.\n'
            #       f'Continue looping through other files.\n')

        # load data
        this_dur_df = pd.read_csv(filepath)
        if verbose:
            print(f"loaded csv:\n{this_dur_df.head()}")

        # remove any Unnamed columns
        if any("Unnamed" in i for i in list(this_dur_df.columns)):
            unnamed_col = [i for i in list(this_dur_df.columns) if "Unnamed" in i][0]
            this_dur_df.drop(unnamed_col, axis=1, inplace=True)

        # sort by staircase
        trial_numbers = list(this_dur_df['trial_number'])
        this_dur_df = this_dur_df.sort_values(by=['stair', 'trial_number'])

        # add duration column for multi-indexing
        # this_dur_df.insert(0, 'probe_dur', duration)
        this_dur_df.insert(1, 'srtd_trial_idx', trial_numbers)
        if verbose:
            print(f'df sorted by stair: {type(this_dur_df)}\n{this_dur_df}')

        # get column names to use on all_data_df
        column_names = list(this_dur_df)
        if verbose:
            print(f'column_names: {len(column_names)}\n{column_names}')

        # I've changed column names lately, so there are some extra ones.  In which case, just use old cols.
        # if 'actual_bg_color' in column_names:
        #     print("getting rid of extra columns (e.g., 'actual_bg_color', "
        #           "'bgcolor_to_rgb1', 'bgLumP', 'bgLum', 'bgColor255')")
        #     cols_to_use = ['duration', 'srtd_trial_idx', 'trial_number', 'stair',
        #                    'stair_name', 'step', 'separation', 'congruent',
        #                    'flow_dir', 'probe_jump', 'corner', 'probeLum',
        #                    'trial_response', 'resp.rt', 'probeColor1', 'probeColor255',
        #                    'probe_ecc', 'BGspeed', 'orientation', 'dur_actual_ms',
        #                    'dur_frames', '1_Participant', '2_Probe_dur_in_frames_at_240hz',
        #                    '3_fps', '4_dur_dur_in_ms', '5_Probe_orientation',
        #                    '6_Probe_size', '7_Trials_counter', '8_Background',
        #                    'date', 'time', 'stair_list', 'n_trials_per_stair']
        #     this_dur_df = this_dur_df[cols_to_use]
        #     column_names = cols_to_use

        # add to all_data
        all_data.append(this_dur_df)

    # create all_data_df - reshape to 2d
    if verbose:
        print(f'all_data: {type(all_data)}\n{all_data}')
    all_data_shape = np.shape(all_data)
    sheets, rows, columns = np.shape(all_data)
    all_data = np.reshape(all_data, newshape=(sheets * rows, columns))
    if verbose:
        print(f'all_data reshaped from {all_data_shape} to {np.shape(all_data)}')
    all_data_df = pd.DataFrame(all_data, columns=column_names)

    if verbose:
        print(f"all_data_df:\n{all_data_df}")

    if save_all_data:
        # save_name = 'ALL_durations_sorted.xlsx'
        # save_excel_path = os.path.join(run_dir, save_name)
        save_name = 'ALL_durations_sorted.csv'
        save_csv_path = os.path.join(run_dir, save_name)
        if verbose:
            # print(f"\nsaving all_data_df to save_excel_path:\n{save_excel_path}")
            print(f"\nsaving all_data_df to save_csv_path:\n{save_csv_path}")
        # all_data_df.to_excel(save_excel_path, index=False)
        convert_path1 = os.path.normpath(save_csv_path)
        print(f"convert_path1: {convert_path1}")

        all_data_df.to_csv(convert_path1, index=False)

    print("\n***finished a_data_extraction()***\n")

    return all_data_df


def b3_plot_staircase(all_data_path, thr_col='probeSpeed', resp_col='answer',
                      show_plots=True, save_plots=True, verbose=True):
    """
    b3_plot_staircase: staircases-durxxx.png: xxx corresponds to dur conditions.
    One plot for each dur condition.

    # todo: don't need different plots for each sep cond - so just make one plot.

    Each figure has six panels (6 probes separation
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
    one figure per dur value - saved as Staircases_{dur_name}
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
                                    usecols=[
                                        "trial_number", "srtd_trial_idx",
                                        "stair", "stair_name", "step",
                                        "probe_dur", "flow_dir", "probeDir",
                                        "answer", "rel_answer",
                                        "probeSpeed", "abs_probeSpeed"])

    # get list of dur and stair values to loop through
    stair_list = all_data_df['stair'].unique()
    dur_list = all_data_df['probe_dur'].unique()
    # get dur string for column names
    dur_name_list = [f'dur_{i}' for i in dur_list]

    trials, columns = np.shape(all_data_df)
    trials_per_stair = int(trials / len(dur_list) / len(stair_list))

    if verbose:
        print(f"all_data_df:\n{all_data_df}")
        print(f"{len(dur_list)} dur values and {len(stair_list)} stair values")
        print(f"dur_list: {dur_list}")
        print(f"dur_name_list: {dur_name_list}")
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
        stair_names_list = psignifit_thr_df.pop('stair_names').tolist()
        print(f'stair_names_list: {stair_names_list}')


    if 'congruent' in list(psignifit_thr_df.columns):
        psignifit_thr_df = psignifit_thr_df.drop(['congruent'], axis=1)

    if 'separation' in list(psignifit_thr_df.columns):
        sep_list = psignifit_thr_df.pop('separation').tolist()
        print(f'sep_list: {sep_list}')

    psignifit_thr_df.columns = dur_name_list

    # split into pos_sep, neg_sep and mean of pos and neg.
    psig_stair0_sep_df, psig_stair1_sep_df = split_df_alternate_rows(psignifit_thr_df)
    psig_thr_mean_df = pd.concat([psig_stair0_sep_df, psig_stair1_sep_df]).groupby(level=0).mean()

    # add stair_names column in
    rows, cols = psig_thr_mean_df.shape

    psig_thr_mean_df.insert(0, 'stair_name', 'mean')
    psig_stair0_sep_df.insert(0, 'stair_name', stair_names_list[0])
    psig_stair1_sep_df.insert(0, 'stair_name', stair_names_list[1])
    if verbose:
        print(f'\npsig_stair0_sep_df:\n{psig_stair0_sep_df}')
        print(f'\npsig_stair1_sep_df:\n{psig_stair1_sep_df}')
        print(f'\npsig_thr_mean_df:\n{psig_thr_mean_df}')

    # make empty arrays to save reversal n_reversals
    n_reversals_np = np.zeros(shape=[len(stair_list), len(dur_list)])

    # loop through dur values
    for dur_idx, dur in enumerate(dur_list):

        # get df for this dur only
        dur_df = all_data_df[all_data_df['probe_dur'] == dur]
        dur_name = dur_name_list[dur_idx]
        print(f"\n{dur_idx}. staircase for dur: {dur}, {dur_name}")

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))

        ax_counter = 0

        # todo: don't need different plots for each sep cond - so just make one plot.

                # # get pairs of stairs (e.g., [[18, -18], [6, -6], ...etc)
        stair_even_cong = 0  # 0, 2, 4, 6, 8, 10
        stair_even_stair0_df = dur_df[dur_df['stair'] == stair_even_cong]
        final_lum_even_cong = \
            stair_even_stair0_df.loc[stair_even_stair0_df['step'] == trials_per_stair - 1, thr_col].item()
        n_reversals_even_cong = trials_per_stair - stair_even_stair0_df[resp_col].sum()

        stair_odd_incong = 1  # 1, 3, 5, 7, 9, 11
        stair_odd_stair1_df = dur_df[dur_df['stair'] == stair_odd_incong]
        final_lum_odd_incong = \
            stair_odd_stair1_df.loc[stair_odd_stair1_df['step'] == trials_per_stair - 1, thr_col].item()
        n_reversals_odd_incong = trials_per_stair - stair_odd_stair1_df[resp_col].sum()

        # append n_reversals to n_reversals_np to save later.
        n_reversals_np[stair_even_cong - 1, dur_idx] = n_reversals_even_cong
        n_reversals_np[stair_odd_incong - 1, dur_idx] = n_reversals_odd_incong

        if verbose:
            print(f'\nstair_even_stair0_df (stair={stair_even_cong}, '
                  f'dur_name={dur_name}:\n{stair_even_stair0_df}')
            print(f"final_lum_even_cong: {final_lum_even_cong}")
            print(f"n_reversals_even_cong: {n_reversals_even_cong}")
            print(f'\nstair_odd_stair1_df (stair={stair_odd_incong}, '
                  f'dur_name={dur_name}:\n{stair_odd_stair1_df}')
            print(f"final_lum_odd_incong: {final_lum_odd_incong}")
            print(f"n_reversals_odd_incong: {n_reversals_odd_incong}")

        fig.suptitle(f'Staircases and reversals for {dur_name}')

        # plot thr per step for even_cong numbered stair
        # sns.lineplot(ax=axes[row_idx, col_idx], data=stair_even_stair0_df,
        sns.lineplot(ax=axes, data=stair_even_stair0_df,
                     x='step', y=thr_col, color='tab:blue',
                     marker="o", markersize=4)
        # line for final newLum
        axes.axhline(y=final_lum_even_cong, linestyle="-.", color='tab:blue')
        # text for n_reversals
        axes.text(x=0.25, y=0.8, s=f'{n_reversals_even_cong} reversals',
                  color='tab:blue',
                  # needs transform to appear with rest of plot.
                  transform=axes.transAxes, fontsize=12)

        # plot thr per step for odd_incong numbered stair
        # sns.lineplot(ax=axes[row_idx, col_idx], data=stair_odd_stair1_df,
        sns.lineplot(ax=axes, data=stair_odd_stair1_df,
                     x='step', y=thr_col, color='tab:red',
                     marker="v", markersize=5)
        axes.axhline(y=final_lum_odd_incong, linestyle="--", color='tab:red')
        axes.text(x=0.25, y=0.9, s=f'{n_reversals_odd_incong} reversals',
                  color='tab:red',
                  # needs transform to appear with rest of plot.
                  transform=axes.transAxes, fontsize=12)

        axes.set_title(f'{dur_name}')
        axes.set_xticks(np.arange(0, trials_per_stair, 5))
        # axes.set_ylim([0, 110])

        # artist for legend
        st1 = mlines.Line2D([], [], color='tab:blue',
                            marker='v',
                            markersize=5, label='0_fl_in_pr_out')
        st1_last_val = mlines.Line2D([], [], color='tab:blue',
                                     linestyle="--", marker=None,
                                     label='0_fl_in_pr_out: last val')
        st2 = mlines.Line2D([], [], color='tab:red',
                            marker='o',
                            markersize=5, label='1_fl_out_pr_in')
        st2_last_val = mlines.Line2D([], [], color='tab:red',
                                     linestyle="-.", marker=None,
                                     label='1_fl_out_pr_in: last val')
        axes.legend(handles=[st1, st1_last_val, st2, st2_last_val],
                  fontsize=6, loc='lower right')

        plt.tight_layout()

        # show and close plots
        if save_plots:
            save_name = f'staircases_{dur_name}.png'
            plt.savefig(os.path.join(save_path, save_name))

        if show_plots:
            plt.show()
        plt.close()

    # save n_reversals to csv for use in script_c figure 5
    n_reversals_df = pd.DataFrame(n_reversals_np, columns=dur_name_list)
    n_reversals_df.insert(0, 'stair', stair_list)
    n_reversals_df.set_index('stair', inplace=True)
    if verbose:
        print(f'n_reversals_df:\n{n_reversals_df}')
    n_reversals_df.to_csv(os.path.join(save_path, 'n_reversals.csv'))

    print("\n***finished b3_plot_staircases()***\n")


# def c_plots(save_path, dur_name_list=None, show_plots=True, verbose=True):
#     """
#     5. c_plots.m: uses psignifit_thresholds.csv and outputs plots.
#
#     figures:
#             MIRRORED_runs.png: threshold luminance as function of probe separation,
#                   Positive and negative separation values (batman plots),
#                   one panel for each dur condition.
#                   use multi_batman_plots()
#
#             data.png: threshold luminance as function of probe separation.
#                 Positive and negative separation values (batman plot),
#                 all durs on same axis.
#                 Use plot_data_unsym_batman()
#
#             compare_data.png: threshold luminance as function of probe separation.
#                 Positive and negative separation values (batman plot),
#                 all durs on same axis.
#                 doesn't use a function, built in c_plots()
#
#
#     :param save_path: path to run dir containing psignifit_thresholds.csv, where plots will be save.
#     :param dur_name_list: Default None: can input a list of names of durs,
#             useful if I only have data for a few dur values.
#     :param show_plots: Default True
#     :param verbose: Default True.
#     """
#     print("\n*** running c_plots() ***\n")
#
#     # load df mean of last n probeLum values (14 stairs x 8 dur).
#     thr_csv_name = f'{save_path}{os.sep}psignifit_thresholds.csv'
#
#     # this psig_thr_df is in stair order (e.g., stairs 0, 1, 2, 3 == sep: 18, -18, 6, -6 etc)
#     psig_thr_df = pd.read_csv(thr_csv_name)
#     if verbose:
#         print(f'psig_thr_df:\n{psig_thr_df}')
#
#     if dur_name_list is None:
#         dur_name_list = list(psig_thr_df.columns[4:])
#     if verbose:
#         print(f'dur_name_list: {dur_name_list}')
#
#     psig_thr_df = psig_thr_df.drop(['stair'], axis=1)
#     if 'separation' in list(psig_thr_df.columns):
#         sep_col_s = psig_thr_df.pop('separation')
#     if 'stair_names' in list(psig_thr_df.columns):
#         stair_names_list = list(psig_thr_df['stair_names'])
#         print(f'stair_names_list: {stair_names_list}')
#
#     if 'congruent' in list(psig_thr_df.columns):
#         stair0_col_s = psig_thr_df.pop('congruent')
#
#     psig_thr_df.columns = ['stair_names'] + dur_name_list
#     if verbose:
#         # psig_thr_df still in stair order  (e.g., sep: 18, -18, 6, -6 etc)
#         print(f'\npsig_thr_df:\n{psig_thr_df}')
#
#     if verbose:
#         print('\npreparing data for batman plots')
#
#     # note: for sym_sep_list just a single value of zero, no -.10
#     symm_sep_indices = [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]
#     sym_sep_list = [-18, -6, -3, -2, -1, 0, 1, 2, 3, 6, 18]
#
#     psig_stair0_sep_df = psig_thr_df[psig_thr_df['stair_names'] >= 0]
#     psig_stair0_sym_df = psig_stair0_sep_df.iloc[symm_sep_indices]
#     psig_stair0_sym_df.reset_index(drop=True, inplace=True)
#
#     psig_stair1_sep_df = psig_thr_df[psig_thr_df['stair_names'] < 0]
#     psig_stair1_sym_df = psig_stair1_sep_df.iloc[symm_sep_indices]
#     psig_stair1_sym_df.reset_index(drop=True, inplace=True)
#
#     # mean of pos and neg stair_name values [18, 6, 3, 2, 1, 0, 1, 2, 3, 6, 18]
#     psig_sym_thr_mean_df = pd.concat([psig_stair0_sym_df, psig_stair1_sym_df]).groupby(level=0).mean()
#
#     # subtract the dfs from each other, then for each column get the sum of abs values
#     diff_val = np.sum(abs(psig_stair0_sym_df - psig_stair1_sym_df), axis=0)
#     diff_val.drop(index='stair_names', inplace=True)
#
#     # take the mean of these across all durs to get single value
#     mean_diff_val = float(np.mean(diff_val))
#
#     # add sep column into dfs
#     psig_sym_thr_mean_df.insert(0, 'separation', sym_sep_list)
#     psig_stair0_sym_df.insert(0, 'separation', sym_sep_list)
#     psig_stair1_sym_df.insert(0, 'separation', sym_sep_list)
#
#     if verbose:
#         print(f'\npsig_stair0_sym_df:\n{psig_stair0_sym_df}')
#         print(f'\npsig_stair1_sym_df:\n{psig_stair1_sym_df}')
#         print(f'\npsig_sym_thr_mean_df:\n{psig_sym_thr_mean_df}')
#         print(f'\ndiff_val:\n{diff_val}')
#         print(f'\nmean_diff_val: {mean_diff_val}')
#
#     # # Figure1 - runs-{n}lastValues
#     # this is a figure with one axis per dur, showing neg and pos sep
#     # (e.g., -18:18) - eight batman plots
#     fig_title = f'MIRRORED Psignifit thresholds per dur. ' \
#                 f'(mean diff: {round(mean_diff_val, 2)})'
#     fig1_savename = f'MIRRORED_runs.png'
#
#     multi_batman_plots(mean_df=psig_sym_thr_mean_df,
#                        thr1_df=psig_stair0_sym_df,
#                        thr2_df=psig_stair1_sym_df,
#                        fig_title=fig_title,
#                        dur_name_list=dur_name_list,
#                        x_tick_vals=sym_sep_list,
#                        x_tick_labels=sym_sep_list,
#                        sym_sep_diff_list=diff_val,
#                        save_path=save_path,
#                        save_name=fig1_savename,
#                        verbose=verbose)
#     if show_plots:
#         plt.show()
#     plt.close()
#
#     #  (figure2 doesn't exist in Martin's script - but I'll keep their numbers)
#
#     # # FIGURE3 - 'data.png' - all durs on same axis, pos and neg sep, looks like batman.
#     # # use plot_data_unsym_batman()
#     fig3_save_name = f'data_pos_and_neg.png'
#     fig_3_title = 'All durs and separations\n' \
#                   '(positive values for congruent probe/flow motion, ' \
#                   'negative for incongruent).'
#
#     psig_sorted_df = psig_thr_df.sort_values(by=['stair_names'])
#     psig_sorted_df.drop('stair_names', axis=1, inplace=True)
#     psig_sorted_df.reset_index(drop=True, inplace=True)
#     psig_thr_idx_list = list(psig_sorted_df.index)
#     stair_names_list = sorted(stair_names_list)
#     stair_names_list = ['-0' if i == -.10 else int(i) for i in stair_names_list]
#
#     if verbose:
#         # psig_sorted_df sorted by stair_NAMES (e.g., sep: -18, -6, -3 etc)
#         print(f'\npsig_sorted_df:\n{psig_sorted_df}')
#         print(f'\npsig_thr_idx_list: {psig_thr_idx_list}')
#         print(f'\nstair_names_list: {stair_names_list}')
#
#     plot_data_unsym_batman(pos_and_neg_sep_df=psig_sorted_df,
#                            fig_title=fig_3_title,
#                            save_path=save_path,
#                            save_name=fig3_save_name,
#                            dur_name_list=dur_name_list,
#                            x_tick_values=psig_thr_idx_list,
#                            x_tick_labels=stair_names_list,
#                            verbose=verbose)
#     if show_plots:
#         plt.show()
#     plt.close()
#
#     #########
#     # fig to compare congruent and incongruent thr for each dur
#     # The remaining plots go back to using psig_thr_df, in stair order (not stair_names)
#     if 'congruent' not in list(psig_thr_df.columns):
#         psig_thr_df.insert(0, 'congruent', stair0_col_s)
#     if 'separation' not in list(psig_thr_df.columns):
#         psig_thr_df.insert(1, 'separation', sep_col_s)
#     if 'stair_names' in list(psig_thr_df.columns):
#         psig_thr_df.drop('stair_names', axis=1, inplace=True)
#
#     dur_cols_list = list(psig_thr_df.columns)[-len(dur_name_list):]
#     if verbose:
#         print(f'psig_thr_df:\n{psig_thr_df}')
#         print(f'dur_name_list: {dur_name_list}')
#         print('\nfig4 single run data')
#     fig4_title = 'Congruent and incongruent probe/flow motion for each dur'
#     fig4_savename = 'compare_data_dur'
#
#     plot_w_errors_either_x_axis(wide_df=psig_thr_df, cols_to_keep=['congruent', 'separation'],
#                                 cols_to_change=dur_name_list,
#                                 cols_to_change_show='probeLum', new_col_name='probe_dur',
#                                 strip_from_cols='dur_',
#                                 x_axis='separation', y_axis='probeLum', x_tick_vals=[0, 1, 2, 3, 6, 18],
#                                 hue_var='probe_dur', style_var='congruent', style_order=[1, -1],
#                                 error_bars=True, even_spaced_x=True, jitter=False,
#                                 fig_title=fig4_title, fig_savename=fig4_savename,
#                                 save_path=save_path, verbose=verbose)
#     if show_plots:
#         plt.show()
#     plt.close()
#
#     if verbose:
#         print('\nfig5 single run data')
#
#     x_tick_vals = [int(i[4:]) for i in dur_cols_list]
#     print(f'x_tick_vals: {x_tick_vals}')
#     fig5_title = 'Congruent and incongruent probe/flow motion for each separation'
#     fig5_savename = 'compare_data_sep'
#
#     plot_w_errors_either_x_axis(wide_df=psig_thr_df, cols_to_keep=['congruent', 'separation'],
#                                 cols_to_change=dur_name_list,
#                                 cols_to_change_show='probeLum', new_col_name='probe_dur',
#                                 strip_from_cols='dur_',
#                                 x_axis='probe_dur', y_axis='probeLum',
#                                 hue_var='separation', style_var='congruent', style_order=[1, -1],
#                                 error_bars=True, even_spaced_x=True, jitter=False,
#                                 fig_title=fig5_title, fig_savename=fig5_savename,
#                                 save_path=save_path, x_tick_vals=x_tick_vals,
#                                 verbose=verbose)
#     if show_plots:
#         plt.show()
#     plt.close()
#
#     print("\n***finished c_plots()***\n")

def d_average_participant(root_path, run_dir_names_list,
                          thr_df_name='psignifit_thresholds',
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
    :param error_type: Default: None. Can pass sd or se for standard deviation or error.
    :param trim_n: default None.  If int is passed, will call function trim_n_high_n_low(),
            which trims the n highest and lowest values.
    :param verbose: Default true, print progress to screen

    :returns: ave_psignifit_thr_df: (trimmed?) mean threshold for each separation and dur.
    """

    print("\n***running d_average_participant()***")

    all_psignifit_list = []
    for run_idx, run_name in enumerate(run_dir_names_list):

        this_psignifit_df = pd.read_csv(f'{root_path}{os.sep}{run_name}{os.sep}{thr_df_name}.csv')
        print(f'\n{run_idx}. {run_name} - this_psignifit_df:\n{this_psignifit_df}')

        if 'Unnamed: 0' in list(this_psignifit_df):
            this_psignifit_df.drop('Unnamed: 0', axis=1, inplace=True)

        if 'stair' in list(this_psignifit_df):
            stair_list = this_psignifit_df['stair'].to_list
            # this_psignifit_df.drop(columns='stair', inplace=True)

        rows, cols = this_psignifit_df.shape
        this_psignifit_df.insert(0, 'stack', [run_idx] * rows)

        if verbose:
            print(f'\nthis_psignifit_df:\n{this_psignifit_df}')

        all_psignifit_list.append(this_psignifit_df)

    # join all stacks (runs/groups) data and save as master csv
    all_data_psignifit_df = pd.concat(all_psignifit_list, ignore_index=True)
    # todo: since I added extra dur conditions, dur conds are not in ascending order.
    #  Perhaps re-order columns before saving?

    all_data_psignifit_df.to_csv(f'{root_path}{os.sep}MASTER_{thr_df_name}.csv', index=False)
    if verbose:
        print(f'\nall_data_psignifit_df:\n{all_data_psignifit_df}')

    """Part 2: trim highest and lowest values is required and get average vals and errors"""
    # # trim highest and lowest values
    if trim_n is not None:
        trimmed_df = trim_n_high_n_low(all_data_psignifit_df, trim_from_ends=trim_n,
                                       reference_col='stair',
                                       stack_col_id='stack',
                                       verbose=verbose)
        trimmed_df.to_csv(f'{root_path}{os.sep}MASTER_TM{trim_n}_thresholds.csv', index=False)

        get_means_df = trimmed_df
    else:
        get_means_df = all_data_psignifit_df

    print(f'\nget_means_df:\n{get_means_df}')

    # # get means and errors
    groupby_sep_df = get_means_df.drop('stack', axis=1)
    ave_psignifit_thr_df = groupby_sep_df.groupby(['stair'], sort=False).mean()
    if verbose:
        print(f'\ngroupby_sep_df:\n{groupby_sep_df}')
        print(f'\nave_psignifit_thr_df:\n{ave_psignifit_thr_df}')

    if error_type in [False, None]:
        error_bars_df = None
    elif error_type.lower() in ['se', 'error', 'std-error', 'standard error', 'standard_error']:
        error_bars_df = groupby_sep_df.groupby('stair', sort=False).sem()
    elif error_type.lower() in ['sd', 'stdev', 'std_dev', 'std.dev', 'deviation', 'standard_deviation']:
        error_bars_df = groupby_sep_df.groupby('stair', sort=False).std()
    else:
        raise ValueError(f"error_type should be in:\nfor none: [False, None]\n"
                         f"for standard error: ['se', 'error', 'std-error', 'standard error', 'standard_error']\n"
                         f"for standard deviation: ['sd', 'stdev', 'std_dev', 'std.dev', "
                         f"'deviation', 'standard_deviation']")
    print(f'\nerror_bars_df:\n{error_bars_df}')

    # save csv with average values
    # todo: since I added extra dur conditions, dur conds are not in ascending order.
    #  Perhaps re-order columns before saving?
    if trim_n is not None:
        ave_psignifit_thr_df.to_csv(f'{root_path}{os.sep}MASTER_ave_TM_thresh.csv')
        error_bars_df.to_csv(f'{root_path}{os.sep}MASTER_ave_TM_thr_error_{error_type}.csv')
    else:
        ave_psignifit_thr_df.to_csv(f'{root_path}{os.sep}MASTER_ave_thresh.csv')
        error_bars_df.to_csv(f'{root_path}{os.sep}MASTER_ave_thr_error_{error_type}.csv')

    print("\n*** finished d_average_participant()***\n")

    return ave_psignifit_thr_df, error_bars_df


def e_average_exp_data(exp_path, p_names_list,
                       error_type='SE',
                       use_trimmed=True,
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
    a. x-axis is separation, dur as different lines
    b. x-axis is dur, separation as different lines
    Heatmap: with average probe lum for dur and separation.

    :param exp_path: dir containing participant folders
    :param p_names_list: names of participant's folders
    :param error_type: Default: None. Can pass sd or se for standard deviation or error.
    :param use_trimmed: default True.  If True, use trimmed_mean ave (MASTER_ave_TM_thresh),
         if False, use MASTER_ave_thresh.
    :param verbose: Default True, print progress to screen

    :returns: exp_ave_thr_df: experiment mean threshold for each separation and dur.
    """
    print("\n***running e_average_over_participants()***\n")

    """ part1. Munge data, save master lists and get means etc
     - loop through participants and get each MASTER_ave_TM_thresh.csv
    Make master sheets: MASTER_exp_thr and MASTER_exp_ave_thr."""

    all_p_ave_list = []
    for p_idx, p_name in enumerate(p_names_list):

        ave_df_name = 'MASTER_ave_thresh'
        if use_trimmed:
            ave_df_name = 'MASTER_ave_TM_thresh'

        this_p_ave_df = pd.read_csv(f'{exp_path}{os.sep}{p_name}{os.sep}{ave_df_name}.csv')

        ave_df_name = 'MASTER_ave_thresh'
        if n_trimmed is not None:
            ave_df_name = f'MASTER_ave_TM{n_trimmed}_thresh'

        this_ave_df_path = os.path.join(exp_path, p_name, f'{ave_df_name}.csv')
        # # if trimmed mean doesn't exists (e.g., because participant hasn't done 12 runs)
        if not os.path.isfile(this_ave_df_path):
            print(f"Couldn't find trimmed mean data for {p_name}\nUsing untrimmed instead.")
            this_ave_df_path = os.path.join(exp_path, p_name, 'MASTER_ave_thresh.csv')

        if verbose:
            print(f'{p_idx}. {p_name} - this_p_ave_df:\n{this_p_ave_df}')

        if 'Unnamed: 0' in list(this_p_ave_df):
            this_p_ave_df.drop('Unnamed: 0', axis=1, inplace=True)

        stair_list = this_p_ave_df['stair'].tolist()
        stair_names_list = ['0_fl_in_pr_out' if x == 0 else '1_fl_out_pr_in' for x in stair_list]
        # sep_list = [0 if x == -.10 else abs(int(x)) for x in stair_names_list]

        rows, cols = this_p_ave_df.shape
        this_p_ave_df.insert(0, 'participant', [p_name] * rows)
        this_p_ave_df.insert(2, 'stair_names', stair_names_list)
        # this_p_ave_df.insert(3, 'separation', sep_list)

        all_p_ave_list.append(this_p_ave_df)

    # join all participants' data and save as master csv
    all_exp_thr_df = pd.concat(all_p_ave_list, ignore_index=True)
    print(f'\nall_exp_thr_df:{list(all_exp_thr_df.columns)}\n{all_exp_thr_df}')

    cols_list = list(all_exp_thr_df.columns)
    conds_list = sorted(int(i) for i in cols_list[3:])
    srtd_cond_list = [str(i) for i in conds_list]
    new_cols_list = cols_list[:3] + srtd_cond_list
    all_exp_thr_df = all_exp_thr_df[new_cols_list]

    if verbose:
        print(f'\nall_exp_thr_df:{list(all_exp_thr_df.columns)}\n{all_exp_thr_df}')
    all_exp_thr_df.to_csv(f'{exp_path}{os.sep}MASTER_exp_thr.csv', index=False)

    # # get means and errors
    groupby_sep_df = all_exp_thr_df.drop('participant', axis=1)
    # groupby_sep_df = groupby_sep_df.drop('separation', axis=1)
    # groupby_sep_df = groupby_sep_df.drop('congruent', axis=1)

    # todo: should I change sort to False for groupby?  Cause probelems in
    #  d_average_participants for error_df if there was only a single run of a
    #  condition so error was NaN and somehow order changed.
    exp_ave_thr_df = groupby_sep_df.groupby('stair', sort=True).mean()
    if verbose:
        print(f'\nexp_ave_thr_df:\n{exp_ave_thr_df}')

    if error_type in [False, None]:
        error_bars_df = None
    elif error_type.lower() in ['se', 'error', 'std-error', 'standard error', 'standard_error']:
        error_bars_df = groupby_sep_df.groupby('stair', sort=True).sem()
    elif error_type.lower() in ['sd', 'stdev', 'std_dev', 'std.dev', 'deviation', 'standard_deviation']:
        error_bars_df = groupby_sep_df.groupby('stair', sort=True).std()
    else:
        raise ValueError(f"error_type should be in:\nfor none: [False, None]\n"
                         f"for standard error: ['se', 'error', 'std-error', 'standard error', 'standrad_error']\n"
                         f"for standard deviation: ['sd', 'stdev', 'std_dev', 'std.dev', "
                         f"'deviation', 'standard_deviation']")
    if verbose:
        print(f'\nerror_bars_df: ({error_type})\n{error_bars_df}')

    # save csv with average values
    exp_ave_thr_df.to_csv(f'{exp_path}{os.sep}MASTER_exp_ave_thr.csv')
    error_bars_df.to_csv(f'{exp_path}{os.sep}MASTER_ave_thr_error_{error_type}.csv')

    print("\n*** finished e_average_over_participants()***\n")

    return exp_ave_thr_df, error_bars_df

def plot_diff(ave_thr_df, stair_names_col='stair', fig_title=None,
              save_path=None, save_name=None,
              x_axis_label=None,
              difference_description=None,
              verbose=True):
    """
    Function to plot the difference between congruent and incongruent conditions.
    :param ave_thr_df: Dataframe to use to get differences
    :param stair_names_col: Column name containing separation and congruence values.
    :param fig_title: Title for fig
    :param save_path: path to save fig to
    :param save_name: name to save fig
    :param x_axis_label: Can manually add the x-axis label, or if None will infer it is either 'Separation' or 'ISI'.
    :param difference_description: Can manually add the difference_description, or if None will infer it is 'cong - incong'.
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

    # get list of values for each row
    headers_list = list(srtd_ave_thr_df)
    dur_vals_list = headers_list[1:]
    stair0_vals = srtd_ave_thr_df.loc[0, dur_vals_list]
    stair1_vals = srtd_ave_thr_df.loc[1, dur_vals_list]
    print(f"stair0_vals:\n{stair0_vals}\n\nstair1_vals:\n{stair1_vals}")


    # subtract one from the other
    diff_df = stair0_vals - stair1_vals
    print(f'\ndiff_df:\n{diff_df}')

    diff_df = diff_df.T
    x_tick_labels = dur_vals_list

    if verbose:
        print(f'\ndiff_df:\n{diff_df}')
        print(f'\nx_axis_label: {x_axis_label}')
        print(f'\ndifference_description: {difference_description}')

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
    plt.ylabel(difference_description)
    if save_name:
        plt.savefig(os.path.join(save_path, save_name))

    print('*** finished plot_diff() ***')

    return fig


def make_average_plots(all_df_path, ave_df_path, error_bars_path,
                       n_trimmed=False,
                       exp_ave=False,
                       dur_values_list=None,
                       show_plots=True, verbose=True):

    # todo: if necessary, have a separate function to transform data before feeding it into here.

    """Plots:
    MASTER_ave_thresh saved as ave_thr_all_runs.png
    MASTER_ave_thresh two-probe/one-probe saved as ave_thr_div_1probe.png
    these both have two versions:
    a. x-axis is separation, dur as different lines
    b. x-axis is dur, separation as different lines
    Heatmap: with average probe lum for dur and separation.

    :param all_df_path: Path to df with all participant/stack data.
    :param ave_df_path: Path to df with average across all stacks/participants
    :param error_bars_path: Path to df for errors bars with SE/SD associated with averages.
    :param n_trimmed: Whether averages data has been trimmed.
    :param exp_ave: If False, this script is for participant averages over runs.
                    If True, the script if for experiment averages over participants.
    :param show_plots:
    :param verbose:
    :return: """

    print("\n*** running make_average_plots()***\n")

    # todo: check why plots have changed order - since I added extra dur conditions.

    save_path, df_name = os.path.split(ave_df_path)

    if exp_ave:
        ave_over = 'Exp'
    else:
        ave_over = 'P'

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

    if dur_values_list is None:
        dur_values_list = list(all_df.columns[3:])
    dur_name_list = [f'dur_{i}' for i in dur_values_list]

    if verbose:
        print(f'\nall_df:\n{all_df}')
        print(f'\nave_df:\n{ave_df}')
        print(f'\nerror_bars_df:\n{error_bars_df}')
        print(f'\ndur_name_list; {dur_name_list}')
        print(f'\ndur_values_list; {dur_values_list}')

    stair_names_list = sorted(list(all_df['stair_names'].unique()))
    if verbose:
        print(f"\nstair_names_list: {stair_names_list}")


    """part 3. main Figures (these are the ones saved in the matlab script)
    Fig1: plot average threshold for each dur and sep.
    Fig2: divide all 2probe conds (pos_sep) by one_probe condition for each participant.
    For both figures there are 2 versions:
        a) Sep on x-axis, different line for each dur
        b) dur on x-axis, different line for each Sep"""

    print(f"\nfig_1a")
    if n_trimmed is not None:
        fig_1a_title = f'{ave_over} average thresholds per dur across all runs (trim={n_trimmed}).\n' \
                       f'Bars=SE.'
        fig_1a_savename = f'ave_TM_thr_pos_and_neg.png'
    else:
        fig_1a_title = f'{ave_over} average threshold per dur across all runs.\n' \
                       f'Bars=SE.'
        fig_1a_savename = f'ave_thr_pos_and_neg.png'

    # use ave_w_stairname_idx_df for fig 1a and heatmap
    ave_w_stairname_idx_df = ave_df.copy()
    ave_w_stairname_idx_df.insert(0, 'stair_names', stair_names_list)
    ave_w_stairname_idx_df.drop('stair', inplace=True, axis=1)
    ave_w_stairname_idx_df = ave_w_stairname_idx_df.set_index('stair_names')

    print(f"ave_w_stairname_idx_df:\n{ave_w_stairname_idx_df}")

    # if I delete this messy plot, I can also delete the function that made it.
    plot_runs_ave_w_errors(fig_df=ave_w_stairname_idx_df, error_df=error_bars_df,
                           jitter=False, error_caps=True, alt_colours=False,
                           legend_names=dur_values_list,
                           x_tick_vals=stair_names_list,
                           x_tick_labels=stair_names_list,
                           even_spaced_x=True, fixed_y_range=False,
                           x_axis_label='Condition ("fl"=background motion, "pr"=probe)',
                           y_axis_label='Probe Speed (pixels per frame)',
                           fig_title=fig_1a_title, save_name=fig_1a_savename,
                           save_path=save_path, verbose=verbose)
    if show_plots:
        plt.show()
    plt.close()

    print(f"\nfig_1b")
    if n_trimmed is not None:
        fig_1b_title = f'{ave_over} average thresholds per stair across all runs (trim={n_trimmed}).\n' \
                       f'Bars=.68 CI'
        fig_1b_savename = f'ave_TM_thr_all_runs.png'
    else:
        fig_1b_title = f'{ave_over} average threshold per stair across all runs\n' \
                       f'Bars=.68 CI'
        fig_1b_savename = f'ave_thr_all_runs.png'

    plot_w_errors_either_x_axis(wide_df=all_df, cols_to_keep=['stair_names', 'stair'],
                                cols_to_change=dur_values_list,
                                cols_to_change_show='probeSpeed', new_col_name='probe_dur',
                                strip_from_cols=None, x_axis='probe_dur', y_axis='probeSpeed',
                                hue_var='stair_names', style_var=None, style_order=None,
                                error_bars=True, even_spaced_x=True,
                                x_tick_vals=dur_values_list,
                                jitter=False,   # .01,
                                fig_title=fig_1b_title, fig_savename=fig_1b_savename,
                                save_path=save_path, verbose=verbose)
    if show_plots:
        plt.show()
    plt.close()



    print('\nfig2a: Mean participant difference between congruent and incongruent conditions (x-axis=Sep)')
    diff_description = "Difference: 0_fl_in_pr_out - 1_fl_out_pr_in"

    if n_trimmed is not None:
        fig_2a_title = f'{ave_over} Mean {diff_description}.\n' \
                       f'trim={n_trimmed}.'
        fig_2a_savename = f'ave_TM_diff.png'
    else:
        fig_2a_title = f'{ave_over} Mean {diff_description}.'
        fig_2a_savename = f'ave_diff.png'

    plot_diff(ave_df, stair_names_col='stair',
              fig_title=fig_2a_title, save_path=save_path, save_name=fig_2a_savename,
              x_axis_label='Probe Durations',
              difference_description=diff_description,
              verbose=verbose)
    if show_plots:
        plt.show()
    plt.close()

    print('\nfig2b: Mean participant difference between congruent and incongruent conditions (x-axis=dur)')


    print(f"\nHeatmap")
    if n_trimmed is not None:
        heatmap_title = f'{ave_over} mean Threshold for each dur and separation (trim={n_trimmed}).'
        heatmap_savename = 'mean_TM_thr_heatmap'
    else:
        heatmap_title = f'{ave_over} mean Threshold for each dur and separation'
        heatmap_savename = 'mean_thr_heatmap'

    plot_thr_heatmap(heatmap_df=ave_w_stairname_idx_df.T,
                     heatmap_midpoint=0.0,
                     fig_title=heatmap_title,
                     save_name=heatmap_savename,
                     save_path=save_path,
                     verbose=verbose)
    if show_plots:
        plt.show()
    plt.close()

    print("\n*** finished make_average_plots()***\n")
