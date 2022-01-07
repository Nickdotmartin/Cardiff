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


# todo: some of these functions are duplicated in the exp1a_psitgnifit_analysis.py script.  
#  I don't need to two copies, so perhaps make a 'data_tools' script and put them there?


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
        data_df = data_df.rename(columns={'sep': 'separation'})

    # check if index column is set as 'separation'
    if data_df.index.name is None:
        data_df = data_df.set_index('separation')

    # data_df = data_df.loc[data_df['separation'] >= 0]
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


# # # all ISIs on one axis - pos sep only, plus single probe
# FIGURE 1 - shows one axis (x=separation (0-18), y=probeLum) with all ISIs added.
# it also seems that for isi=99 there are simple dots added at -1 on the x axis.
# todo: rename - plot_unsymm_sep
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
    ax.axvline(x=5.5, linestyle="-.", color='lightgrey')

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
        ax.set_xticks(pos_set_ticks)
        ax.set_xticklabels(pos_tick_labels)
        # ax.set_xticks(pos_set_ticks[2:])
        # ax.set_xticklabels(pos_tick_labels[2:])

    # ax.set_ylim([40, 90])
    ax.set_xlabel('Probe separation in diagonal pixels')
    ax.set_ylabel('Probe Luminance')

    if fig_title is not None:
        plt.title(fig_title)
        
    if save_path is not None:
        if save_name is not None:
            plt.savefig(f'{save_path}{os.sep}{save_name}')

    return fig


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
        separation as index, 1probe as bottom row, ISIs as columns.
    :param error_df: dataframe of same shape as fig_df, but contains error values
    :param split_1probe: Default=True - whether to treat 1probe data separately,
        e.g., not joined with line to 2probe data.
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
                            hue_var=None,
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

    # todo: add hue_var to doc_string

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
    print(f'lines_var: {lines_var}, n_colours: {n_colours}')

    # do error bars have caps?
    cap_size = 0
    if error_caps:
        cap_size = .1

    # convert wide_df to long for getting means and standard error.
    # long_fig_df = make_long_df(wide_df, idx_col='stack')
    long_fig_df = make_long_df(wide_df,
                 cols_to_keep=['congruent', 'stair_names'],
                 cols_to_change=['isi1', 'isi4', 'isi6'],
                 cols_to_change_show='probeLum',
                 new_col_name='isi', strip_from_cols='isi', verbose=True)
    if verbose:
        print(f'long_fig_df:\n{long_fig_df}')

    my_colours = fig_colours(n_colours, alternative_colours=alt_colours)
    print(f"my_colours - {np.shape(my_colours)}\n{my_colours}")


    fig, ax = plt.subplots()

    if hue_var is not None:
        hue_list = sorted(list(long_fig_df[hue_var].unique()))
        line_styles = ['dotted', 'dashed', 'dashdot', 'solid', 'loosely dotted',
                       'densely dotted', 'loosely dashed', 'densely dashed',
                       'densely dashed', 'dashdotted', 'densely dashdotted',
                       'dashdotdotted', 'loosely dashdotdotted', 'densely dashdotdotted']
        for hue_idx, hue in enumerate(hue_list):
            hue_df = long_fig_df[long_fig_df[hue_var] == hue]
            print(f'\nhue_df ({hue}):\n{hue_df}')
            sns.pointplot(data=hue_df, x=x_var, y=y_var, hue=lines_var,
                          estimator=np.mean, ci=68, dodge=jitter, markers='.',
                          linestyle='dashdotted',  #  line_styles[hue_idx],
                          errwidth=1, capsize=cap_size, palette=my_colours, ax=ax)
    else:
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
    :param heatmap_df: Expects dataframe with separation as index and ISIs as columns.
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


# # # all ISIs on one axis - pos sep only, NO single probe

# # # 8 batman plots

# # FIGURE 2
# this is a figure with one axis per isi, showing neg and pos sep
# (e.g., -18:18)

def multi_batman_plots(mean_df, thr1_df, thr2_df,
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
        print("\n*** running multi_batman_plots() ***")

    if isi_name_list is None:
        isi_name_list = ['Concurrent', 'ISI0', 'ISI2', 'ISI4',
                         'ISI6', 'ISI9', 'ISI12', 'ISI24']
    if verbose:
        print(f'isi_name_list: {isi_name_list}')

    if x_tick_vals is None:
        x_tick_vals = [-18, -6, -3, -2, -1, 0, 1, 2, 3, 6, 18]

    if x_tick_labels is None:
        x_tick_labels = [-18, -6, -3, -2, -1, 0, 1, 2, 3, 6, 18]

    # check column name for x_values
    if 'sep' in list(mean_df.columns):
        mean_df = mean_df.rename(columns={'sep': 'separation'})
        thr1_df = thr1_df.rename(columns={'sep': 'separation'})
        thr2_df = thr2_df.rename(columns={'sep': 'separation'})


    # set colours
    my_colours = fig_colours(len(isi_name_list))

    n_rows, n_cols = multi_plot_shape(len(isi_name_list), min_rows=2)
    print(f'n_rows: {n_rows}, n_cols {n_cols}')
    # make plots
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
                # write 'empty' so its clear this is empty on purpose
                ax.text(x=0.45, y=0.5, s='empty',
                        # needs transform to appear with rest of plot.
                        transform=ax.transAxes, fontsize=8)

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



# todo: I don't think I need this - delete
# def b1_extract_last_values(all_data_path, thr_col='probeLum', resp_col='trial_response',
#                            last_vals_list=None, verbose=True):
#
#     """
#     This script is a python version of Martin's second MATLAB analysis scripts, described below.
#
#     b1_extract_last_values.m: For each isi there are 14 staircase values,
#     this script gets the last threshold and last response (correct/incorrect) for each.
#
#     You can also get the mean of the last n thresholds
#     (e.g., 4, 7 etc as thresholds-sorted-4last, thresholds-sorted-7last.
#
#     :param all_data_path: path to the all_data.xlsx file
#     :param thr_col: (default probeLum) name of the column showing the threshold varied by the staircase.
#     :param resp_col: (default: 'trial_response') name of the column showing accuracy per trial.
#     :param last_vals_list: get the mean threshold of the last n values.
#         It will use [1] (e.g., last threshold only), unless another list is passed.
#     :param verbose: If True, will print progress to screen.
#
#     :return: nothing, but saves the files as:
#             'last_response.csv' showing response to last trial and 'last_threshold'.csv.
#             If n is greater than 1, files will be
#             f'mean_of_last_{last_n_values}_thr.csv' showing mean of last n thresholds
#     """
#
#     print("\n***running b1_extract_last_values()***")
#
#     # extract path to save files to
#     save_path, xlsx_name = os.path.split(all_data_path)
#
#     # todo: do I actually need the mean of the last 4 or 7 values, or should I just get last 1?
#
#     if last_vals_list is None:
#         last_vals_list = [1]
#     elif type(last_vals_list) == int:
#         last_vals_list = [last_vals_list]
#     elif type(last_vals_list) == list:
#         if not all(type(x) is int for x in last_vals_list):
#             raise TypeError(f'last_vals list should be list of ints, not {last_vals_list}.')
#     else:
#         raise TypeError(f'last_vals list should be list of ints, not {last_vals_list} {type(last_vals_list)}.')
#
#     # open all_data file.  use engine='openpyxl' for xlsx files.
#     # For other experiments it might be easier not to do use cols as they might be different.
#     # todo: do I need response.rt?  No but might be good to have on all data for summary stats.
#     all_data_df = pd.read_excel(all_data_path, engine='openpyxl',
#                                 usecols=['ISI', 'stair', 'trial_number',
#                                          'probeLum', 'trial_response', 'resp.rt'])
#
#     # get list of isi and stair values to loop through
#     isi_list = all_data_df['ISI'].unique()
#     stair_list = all_data_df['stair'].unique()
#
#     # check last_vals_list are shorted than trials per stair.
#     trials, columns = np.shape(all_data_df)
#     trials_per_stair = int(trials/len(isi_list)/len(stair_list))
#     if max(last_vals_list) > trials_per_stair:
#         raise ValueError(f'max(last_vals_list) ({max(last_vals_list)}) must be '
#                          f'lower than trials_per_stair ({trials_per_stair}).')
#
#     # get isi string for column names
#     isi_name_list = [f'isi{i}' for i in isi_list]
#
#     if verbose:
#         print(f"last_vals_list: {last_vals_list}")
#         print(f"{len(isi_list)} isi values and {len(stair_list)} stair values")
#         print(f"all_data_df:\n{all_data_df}")
#
#
#     # loop through last values (e.g., [1, 4, 7])
#     for last_n_values in last_vals_list:
#         if verbose:
#             print(f"\nlast_n_values: {last_n_values}")
#
#         # make empty arrays to add results into (rows=stairs, cols=ISIs)
#         thr_array = np.zeros(shape=[len(stair_list), len(isi_list)])
#         resp_array = np.zeros(shape=[len(stair_list), len(isi_list)])
#
#
#         # loop through isi values
#         for isi_idx, isi in enumerate(isi_list):
#             if verbose:
#                 print(f"\n{isi_idx}: isi: {isi}")
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
#                     print(f'\nstair_df (stair={stair}, isi={isi}, last_n_values={last_n_values}):\n{stair_df}')
#
#                 # get the mean threshold of the last n values (last_n_values)
#                 mean_thr = np.mean(list(stair_df[thr_col])[-last_n_values:])
#                 if verbose:
#                     if last_n_values > 1:
#                         print(f'last {last_n_values} values: {list(stair_df[thr_col])[-last_n_values:]}')
#                     print(f'mean_thr: {mean_thr}')
#
#                 # copy value into threshold array
#                 thr_array[stair_idx, isi_idx] = mean_thr
#
#
#                 if last_n_values == 1:
#                     last_response = list(stair_df[resp_col])[-last_n_values]
#                     if verbose:
#                         print(f'last_response: {last_response}')
#
#                     # copy value into response array
#                     resp_array[stair_idx, isi_idx] = last_response
#
#
#         # make dataframe from array
#         thr_df = pd.DataFrame(thr_array, columns=isi_name_list)
#         thr_df.insert(0, 'stair', stair_list)
#         if verbose:
#             print(f"thr_df:\n{thr_df}")
#
#         # save response and threshold arrays
#         # save name if last_n_values > 1
#         thr_filename = f'mean_of_last_{last_n_values}_thr.csv'
#
#         if last_n_values == 1:
#             # make dataframe from array for last response (correct/incorrect)
#             resp_df = pd.DataFrame(resp_array, columns=isi_name_list)
#             resp_df.insert(0, 'stair', stair_list)
#             if verbose:
#                 print(f"resp_df:\n{resp_df}")
#
#             # save last response
#             resp_filename = 'last_response.csv'
#             resp_filepath = os.path.join(save_path, resp_filename)
#             resp_df.to_csv(resp_filepath, index=False)
#
#             # save name if last_n_values == 1
#             thr_filename = 'last_threshold.csv'
#
#         # save threshold arrays
#         thr_filepath = os.path.join(save_path, thr_filename)
#         thr_df.to_csv(thr_filepath, index=False)
#
#
#     print("\n***finished b1_extract_last_values()***")


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
#         rev_thr_mean_df.insert(loc=0, column='separation', value=sym_sep_list)
#         rev_thr1_sym_df.insert(loc=0, column='separation', value=sym_sep_list)
#         rev_thr2_sym_df.insert(loc=0, column='separation', value=sym_sep_list)
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
#         multi_batman_plots(mean_df=rev_thr_mean_df,
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
                    final_lum_even_cong = stair_even_cong_df.loc[stair_even_cong_df['step'] == trials_per_stair-1, 'probeLum'].item()
                    n_reversals_even_cong = trials_per_stair - stair_even_cong_df[resp_col].sum()

                    stair_odd_incong = (ax_counter*2)+1  # 1, 3, 5, 7, 9, 11
                    stair_odd_incong_df = isi_df[isi_df['stair'] == stair_odd_incong]
                    final_lum_odd_incong = stair_odd_incong_df.loc[stair_odd_incong_df['step'] == trials_per_stair-1, 'probeLum'].item()
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
                    # write 'empty' so its clear this is empty on purpose
                    ax.text(x=0.4, y=0.5, s='empty',
                            # needs transform to appear with rest of plot.
                            transform=ax.transAxes, fontsize=12)

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


def c_plots(save_path, thr_col='probeLum', isi_name_list=None,
            show_plots=True, verbose=True):

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
                Use plot_pos_sep_and_one_probe()

            compare_data.png: threshold luminance as function of probe separation.
                Positive and negative separation values (batman plot),
                all ISIs on same axis.
                doesn't use a function, built in c_plots()


    :param save_path: path to run dir containing psignifit_thresholds.csv, where plots will be save.
    :param thr_col: column for threshold (e.g., probeLum)
    :param isi_name_list: Default None: can input a list of names of ISIs,
            useful if I only have data for a few ISI values.
    :param show_plots: Default True
    :param verbose: Default True.
    """
    print("\n*** running c_plots() ***\n")

    if isi_name_list is None:
        isi_name_list = ['Concurrent', 'ISI0', 'ISI2', 'ISI4',
                         'ISI6', 'ISI9', 'ISI12', 'ISI24']
    if verbose:
        print(f'isi_name_list: {isi_name_list}')
    sym_sep_list = [-18, -6, -3, -2, -1, 0, 1, 2, 3, 6, 18]
    pos_sep_list = [0, 1, 2, 3, 6, 18]

    # load df mean of last n probeLum values (14 stairs x 8 isi).
    thr_csv_name = f'{save_path}{os.sep}psignifit_thresholds.csv'
    psig_thr_df = pd.read_csv(thr_csv_name)
    if verbose:
        print(f'psig_thr_df:\n{psig_thr_df}')

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
    symm_sep_indices = [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]

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
    #
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

    # # FIGURE3 - 'data-{n}lastValues.png' - all ISIs on same axis, pos sep only
    # # use plot_pos_sep_and_one_probe()
    fig3_save_name = f'data.png'
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


    plot_pos_sep_and_one_probe(pos_sep_and_one_probe_df=psig_thr_df,
                               fig_title=fig_3_title,
                               one_probe=False,
                               save_path=save_path,
                               save_name=fig3_save_name,
                               isi_name_list=isi_name_list,
                               pos_set_ticks=psig_thr_idx_list,
                               pos_tick_labels=stair_names_list,
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

    # convert wide df to long
    long_df = make_long_df(wide_df=psig_thr_df, cols_to_keep=['congruent', 'separation'],
                           cols_to_change=isi_cols_list, cols_to_change_show='probeLum',
                           new_col_name='ISI', strip_from_cols='ISI_', verbose=True)

    # make plot with pos sep only
    # same colour with dashed or dotted for congruent/incongruent
    fig_title = 'Congruent and incongruent probe/flow motion for each ISI'
    fig_savename = 'compare_data'
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=long_df,
                 x='separation', y='probeLum', hue='ISI', style='congruent',
                 style_order=[1, -1],
                 markers=True, dashes=True, ax=ax)
    ax.set_xticks(pos_sep_list)
    plt.title(fig_title)

    # Change legend labels for congruent and incongruent data
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels[:-2] + ['True', 'False'])

    plt.savefig(f'{save_path}{os.sep}{fig_savename}')

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
    :param verbose: Defaut true, print progress to screen

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
    Fig2: divide all 2probe conds (pos_sep) by one_probe condition for each stack.
    For both figures there are 2 versions:
        a) Sep on x-axis, different line for each ISI
        b) ISI on x-axis, different line for each Sep"""

    stair_names_list = sorted(list(all_data_psignifit_df['stair_names'].unique()))
    stair_names_list = [-.1 if i == -.10 else int(i) for i in stair_names_list]
    stair_names_labels = ['-0' if i == -.10 else int(i) for i in stair_names_list]
    print(f"\nstair_names_list: {stair_names_list}")
    print(f"stair_names_labels: {stair_names_labels}")

    print(f"\nfig_1a\n")
    if trim_n is not None:
        fig1_title = f'Average thresholds across all runs (trim={trim_n}).'
        fig1_savename = f'ave_TM_thr_all_runs.png'
    else:
        fig1_title = f'Average threshold across all runs'
        fig1_savename = f'ave_thr_all_runs.png'

    fig1a = plot_1probe_w_errors(fig_df=ave_psignifit_thr_df, error_df=error_bars_df,
                                 split_1probe=False, jitter=True,
                                 error_caps=True, alt_colours=False,
                                 legend_names=isi_name_list,
                                 x_tick_vals=stair_names_list,
                                 x_tick_labels=stair_names_labels,
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
        fig1b_title = f'Probe luminance at each ISI value per separation (trim={trim_n}).'
        fig1b_savename = f'ave_TM_thr_all_runs_transpose.png'
    else:
        fig1b_title = f'Probe luminance at each ISI value per separation'
        fig1b_savename = f'ave_thr_all_runs_transpose.png'

    print(f"\nget_means_df\n{get_means_df}")

    # # currently not working
    # fig1b = plot_w_errors_no_1probe(wide_df=get_means_df,
    #                                 x_var='isi', y_var='probeLum',
    #                                 lines_var='stair_names',
    #                                 hue_var='congruent',
    #                                 legend_names=stair_names_list,
    #                                 x_tick_labels=isi_name_list,
    #                                 alt_colours=True,
    #                                 fixed_y_range=False,
    #                                 jitter=True,
    #                                 error_caps=True,
    #                                 fig1b_title=fig1b_title,
    #                                 fig1b_savename=fig1b_savename,
    #                                 save_path=root_path,
    #                                 verbose=True)
    # if show_plots:
    #     plt.show()
    # plt.close()

    isi_cols_list = list(get_means_df.columns)[-len(isi_name_list):]
    if verbose:
        print(f'get_means_df:\n{get_means_df}')
        print(f'isi_name_list: {isi_name_list}')

    # convert wide df to long
    long_df = make_long_df(wide_df=get_means_df, cols_to_keep=['congruent', 'separation'],
                           cols_to_change=isi_cols_list, cols_to_change_show='probeLum',
                           new_col_name='ISI', strip_from_cols='isi', verbose=True)

    # make plot with pos sep only
    # same colour with dashed or dotted for congruent/incongruent
    fig_title = 'Congruent and incongruent probe/flow motion for each ISI'
    fig_savename = 'compare_mean_data'
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=long_df,
                 x='separation', y='probeLum', hue='ISI', style='congruent',
                 style_order=[1, -1],
                 markers=True, dashes=True,
                 estimator=np.mean, ci='sd', err_style='bars',
                 ax=ax)
    ax.set_xticks(sorted(list(long_df['separation'].unique())))
    plt.title(fig_title)

    # Change legend labels for congruent and incongruent data
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels[:-2] + ['True', 'False'])

    plt.savefig(f'{root_path}{os.sep}{fig_savename}')

    if show_plots:
        plt.show()
    plt.close()

    if verbose:
        print('finished fig1b')


    print(f"\nHeatmap\n")
    # get mean of each col, then mean of that

    if trim_n is not None:
        heatmap_title = f'Mean Threshold for each ISI and separation (trim={trim_n}).'
        heatmap_savename = 'mean_TM_thr_heatmap'
    else:
        heatmap_title = 'Mean Threshold for each ISI and separation'
        heatmap_savename = 'mean_thr_heatmap'

    heatmap = plot_thr_heatmap(heatmap_df=ave_psignifit_thr_df,
                               x_tick_labels=isi_name_list,
                               y_tick_labels=stair_names_list,
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
