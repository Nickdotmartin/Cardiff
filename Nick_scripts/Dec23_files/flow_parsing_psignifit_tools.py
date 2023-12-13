import os
import pickle
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import numpy as np
import pandas as pd
import psignifit as ps
from psignifit_tools import csv_to_np_for_psignifit_Oct23
"""
see psignifit instruction here: https://github.com/wichmann-lab/python-psignifit/wiki
demos at: https://github.com/wichmann-lab/psignifit/wiki/Experiment-Types 
or:/Users/nickmartin/opt/anaconda3/envs/Cardiff3.6/lib/python3.6/site-packages/psignifit/demos
This script contains the analysis pipeline for individual participants.
1. load in CSV file.
2. convert data into format for analysis: 3 cols [stimulus level | nCorrect | ntotal]
3. run psignifit for fit, conf intervals and threshold etc
4. Plot psychometric function
"""
# todo: should this be using abs_probeSpeed or probeSpeed?
# def results_csv_to_np_for_psignifit(csv_path, duration,
#                                     # sep,
#                                     p_run_name, stair_col='stair',
#                                     stair_levels=None,
#                                     thr_col='probeSpeed', resp_col='response',
#                                     quartile_bins=True, n_bins=10, save_np_path=None,
#                                     verbose=True):
#
#     """
#     Converts a results csv to a numpy array for running psignifit.  Numpy array
#     has 3 cols [stimulus level | nCorrect | ntotal].
#
#     :param csv_path: path to results csv, or df.
#     :param duration: which duration are results for?
#     # :param sep: which separation are results for?
#     :param p_run_name: participant and run name e.g., Kim1, Nick4 etc
#     :param thr_col: name of column containing thresholds
#     :param resp_col: name of column containing thresholds
#     :param stair_col: name of column containing separations: use 'stair' if there
#         is no separation column.
#     :param stair_levels: default=None - in which case will access sep from stair_col.
#         If there is no separation column in df, then enter
#         a list of stair level(s) to analyse (e.g., for sep=18, use stair_levels=[0, 1]).
#     :param quartile_bins: If True, will use pd.qcut for bins containing an equal
#         number of items based on the distribution of thresholds.  i.e., bins will
#         not be of equal size intervals but will have equal value count.  If False,
#         will use pd.cut for bins of equal size based on values, but value count may vary.
#     :param n_bins: default 10.
#     :param save_np_path: default None.  will save to path if one is given.
#     :param verbose: default True.  Will print progress to screen.
#
#     :return:
#         numpy array: for analysis in psignifit,
#         psignifit_dict: contains details for psignifit e.g., for title, save plot etc.
#     """
#
#     print('\n*** running results_csv_to_np_for_psignifit() ***')
#
#     if type(csv_path) == pd.core.frame.DataFrame:
#         raw_data_df = csv_path
#     else:
#         raw_data_df = pd.read_csv(csv_path, usecols=[stair_col, thr_col, resp_col])
#     if verbose:
#         print(f"raw_data ({list(raw_data_df.columns)}):\n{raw_data_df}")
#
#     # access separation either through separation column or stair column
#     if stair_levels is None:
#         if stair_col == 'stair':
#             raise ValueError(f'stair_col is {stair_col} but no stair_levels given.  '
#                              f'Enter a list of stair levels corresponding to the '
#                              f'separation value or enter the name of the column '
#                              f'showing separation values. ')
#         # stair_levels = [sep]
#
#     raw_data_df = raw_data_df[raw_data_df[stair_col].isin(stair_levels)]
#     if verbose:
#         print(f"raw_data, stair_levels:{stair_levels}:\n{raw_data_df}")
#
#     # get useful info
#     n_rows, n_cols = raw_data_df.shape
#     thr_min = raw_data_df[thr_col].min()
#     thr_max = raw_data_df[thr_col].max()
#     if verbose:
#         print(f"\nn_rows: {n_rows}, n_cols: {n_cols}")
#         print(f"{thr_col} min: {thr_min}, max: {thr_max}")
#
#     # check n_resp_in in raw_df
#     n_resp_in = sum(list(raw_data_df['response']))
#     n_resp_out = (raw_data_df['response'] == 0).sum()
#     if verbose:
#         print(f'n_resp_in: {n_resp_in}')
#         print(f'n_resp_out: {n_resp_out}')
#
#     # check stair_names
#     stair_name = list(raw_data_df['stair_name'].unique())
#     if len(stair_name) != 1:
#         print(f"stair_name: {stair_name}")
#         raise ValueError(f'Number of stair_names ({len(stair_name)}) is not 1. ')
#     else:
#         stair_name = stair_name[0]
#         print(f"stair_name: {stair_name}")
#
#
#     # get prelim name
#     if 'prelim_ms' in list(raw_data_df):
#         prelim = sorted(list(raw_data_df['prelim_ms'].unique()))
#         if len(prelim) != 1:
#             raise ValueError(f'Number of prelim ({len(prelim)}) is not 1. ')
#
#     # get flow_name
#     if 'flow_name' in list(raw_data_df):
#         flow_name = sorted(list(raw_data_df['flow_name'].unique()))
#         if len(flow_name) != 1:
#             raise ValueError(f'Number of flow_name ({len(flow_name)}) is not 1. ')
#
#     # dataset_name = f'{p_run_name}_dur{round(duration, 2)}_stair{stair_levels[0]}'
#     dataset_name = f'{p_run_name}_dur{round(duration, 2)}_stair{stair_name}'
#
#
#     # idiot check - why isn't it working?
#     print(f"raw_data_df: {raw_data_df.shape}, {raw_data_df.columns}\n{raw_data_df.head()}")
#     print(f"thr_col: {thr_col}")
#     thr_s = raw_data_df[thr_col]
#     # todo: if it doesn't work (below), use thr_s instead of raw_data_df[thr_col]
#     print(f"thr_s: {thr_s.shape}, {thr_s.dtypes}\n{thr_s}")
#     # print(f"raw_data_df['thr_col']: {raw_data_df['thr_col'].shape}\n{raw_data_df['thr_col']}")
#
#     # put responses into bins (e.g., 10)
#     # # use pd.qcut for bins containing an equal number of items based on distribution.
#     # # i.e., bins will not be of equal size but will have equal value count
#     if quartile_bins:
#         # bin_col, bin_labels = pd.qcut(x=raw_data_df[thr_col], q=n_bins,
#         bin_col, bin_labels = pd.qcut(x=raw_data_df[thr_col], q=n_bins,
#                                       precision=3, retbins=True, duplicates='drop')
#     # # use pd.cut for bins of equal size based on values, but value count may vary.
#     else:
#         bin_col, bin_labels = pd.cut(x=raw_data_df[thr_col], bins=n_bins,
#                                      precision=3, retbins=True, ordered=True)
#
#     raw_data_df['bin_col'] = bin_col
#
#     # get n_trials for each bin
#     bin_count = pd.value_counts(raw_data_df['bin_col'])
#     if verbose:
#         print(f"\nbin_count (trials per bin):\n{bin_count}")
#
#     # get bin intervals as list of type(pandas.Interval)
#     bins = sorted([i for i in list(bin_count.index)])
#
#     # loop through bins and get number resp_in_per_bin
#     data_arr = []
#     found_bins_left = []
#     for idx, bin_interval in enumerate(bins):
#         # print(idx, bin_interval)
#         this_bin_vals = [bin_interval.left, bin_interval.right]
#         this_bin_df = raw_data_df.loc[raw_data_df['bin_col'] == bin_interval]
#         if this_bin_df.empty:
#             data_arr.append([bin_interval.left, this_bin_vals, 0, 0])
#         else:
#             print(f'\nthis_bin_df: {this_bin_df.shape}\n{this_bin_df}')
#             resp_in_per_bin = this_bin_df['response'].sum()
#             print(f'\tresp_in_per_bin: {resp_in_per_bin}/{this_bin_df.shape[0]}\n')
#             data_arr.append([bin_interval.left, this_bin_vals, resp_in_per_bin, bin_count[bin_interval]])
#             found_bins_left.append(round(bin_interval.left, 3))
#
#
#     data_df = pd.DataFrame(data_arr, columns=['bin_left', 'stim_level', 'resp_in', 'n_total'])
#     data_df = data_df.sort_values(by='bin_left', ignore_index=True)
#     data_df['prop_corr'] = round(np.divide(data_df['resp_in'], data_df['n_total']).fillna(0), 2)
#     if verbose:
#         print(f"\ndata_df (with extra cols):\n{data_df}")
#     data_df = data_df.drop(columns=['stim_level', 'prop_corr'])
#
#     # # # # 2. convert data into format for analysis: 3 cols [stimulus level | nCorrect | ntotal]
#     data_np = data_df.to_numpy()
#     # print(f"data:\n{data}")
#
#     bin_data_dict = {'csv_path': csv_path, 'dset_name': dataset_name,
#                      'duration': duration,
#                      'prelim': prelim, 'flow_name': flow_name,
#                      # 'sep': sep,
#                      'p_run_name': p_run_name,
#                      'stair_levels': stair_levels,
#                      'stair_name': stair_name,
#                      'quartile_bins': quartile_bins, 'n_bins': n_bins,
#                      'save_np_path': save_np_path}
#
#     if save_np_path is not None:
#         np.savetxt(f"{save_np_path}{os.sep}{dataset_name}.csv", data_np, delimiter=",")
#         with open(f"{save_np_path}{os.sep}{dataset_name}.pickle", 'wb') as handle:
#             pickle.dump(bin_data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
#     print('\n*** finished results_csv_to_np_for_psignifit() ***\n')
#
#     return data_np, bin_data_dict

# # # # # # # #

def run_psignifit(data_np, bin_data_dict, save_path, target_threshold=.5,
                  sig_name='norm', est_type='MAP', n_blocks=None,
                  save_plot=True, show_plot=False, verbose=True):

    """
    Will run psignifit on data_np to fit curve and output dict.

    :param data_np: np.array with three cols (no headers) [stimulus level | nCorrect | ntotal]
    :param bin_data_dict: dictionary of setting of experiment and analysis
        (e.g., duration, sep, stair) and for converting raw output into data_np, (e.g., n_bins, qcut).
    :param save_path: path to save plot and dict
    :param target_threshold: threshold if this percentage correct
    :param sig_name: default: 'norm', can also choose 'logistic'.
    :param est_type: default: 'MAP' (maximum a posteriori), can also choose 'mean' (posterior mean).
    :param n_blocks: default: None. Pass a value to set the number of unique
        probeSpeed values in the array or number of bins if greater than 25.
        e.g., if you want to have 30 bins enter 30.
    :param save_plot: default: True.
    :param show_plot: default: False.  Display plot on sceen. Useful if doing a
        single pot or not saving, but don't use for multiple plots as it slows
        things down and eats memory.
    :param verbose:

    :return: figure of fitted curve and dict of details
    """

    print('\n*** running run_psignifit() ***')

    # # To start psignifit you need to pass a dictionary, which specifies, what kind
    #      of experiment you did and any other parameters of the fit you might want
    options = dict()  # initialize as an empty dict

    options['sigmoidName'] = sig_name  # 'norm'  # 'logistic'

    '''Exp1 and rad flow were 4afc exerpeiments, this is not.
    Yes/No experiments: Intended for simple detection experiments asking subjects whether they perceived a single presented stimulus or not, 
    or any other experiment which has two possible answers of which one is reached for "high" stimulus levels and the other for "low" stimulus levels. 
    This sets both asymptotes free to vary and applies a prior to them favouring small values, e.g. asymptotes near 0 and 1 respectively.
    Equal Asymptote Experiments: This setting is essentially a special case of Yes/No experiments. 
    Here the asymptotes are "yoked", i. e. they are assumed to be equally far from 0 or 1. 
    This corresponds to the assumption that stimulus independent errors are equally likely for clear "Yes" answers as for clear "No" answers.
    
    '''
    # todo: perhaps I should try yes/no
    options['expType'] = 'equalAsymptote'  #
    options['expN'] = 2
    options['estimateType'] = est_type  # 'mean'  # 'MAP'  'mean'

    # number of bins/unique probeSpeed values
    if type(n_blocks) is int:
        if n_blocks > 25:
            options['nBlocks'] = n_blocks

    # set percent correct corresponding to the threshold
    options['threshPC'] = target_threshold

    if verbose:
        print(f'data_np ([stimulus level | nCorrect | ntotal]):\n{data_np}')
        print(f'options (dict): {options}')

    # results
    res = ps.psignifit(data_np, options)

    if verbose:
        print("res['options']")
        for k, v in res['options'].items():
            if k in ['nblocks', 'stimulusRange']:
                print(f"{k}: {v}")

    # get threshold
    options['sigmoidName'] = 'norm'
    # todo: this might need to be reversed now I've got +ive for out and -ive for in
    print(f'options (dict): {options}')
    res = ps.psignifit(data_np, options)
    threshold = ps.getThreshold(res, target_threshold)
    # if threshold percent correct not reached, try different sigmoid
    # threshold = ps.getThreshold(res, target_threshold)
    # try:
    #     threshold = ps.getThreshold(res, target_threshold)
    # except AssertionError:  # 'The threshold percent correct is not reached by the sigmoid!'
    #     print('\nChanging sigmoid to overcome assertion error')
    #     if sig_name == 'norm':
    #         sig_name = 'neg_gauss'
    #     else:
    #         sig_name = 'norm'
    #     options['sigmoidName'] = sig_name
    #     print(f'options (dict): {options}')
    #     res = ps.psignifit(data_np, options)
    #     threshold = ps.getThreshold(res, target_threshold)

    if options['estimateType'] == 'mean':
        threshold = round(threshold[0][0], 2)
    else:
        threshold = round(threshold[0], 2)
    if verbose:
        print(f'\nthreshold: {threshold}')

    slope_at_target = ps.getSlopePC(res, target_threshold)
    if verbose:
        print(f'slope_at_target: {slope_at_target}')

    # # # 4. Plot psychometric function
    dset_name = bin_data_dict['dset_name']

    if (show_plot is False) & (save_plot is False):
        print('not making plots')
        fit_curve_plot = None
    else:


        plt.figure()
        plt.title(f"{dset_name}\n"
                  f"threshPC: {target_threshold}, threshold: {threshold}, "
                  f"sig: {sig_name}, "
                  f"est: {est_type}")
        fit_curve_plot = ps.psigniplot.plotPsych(res, showImediate=False)

        if save_plot:
            print(f'saving plot to: {save_path}{os.sep}{dset_name}_psig.png')
            plt.savefig(f'{save_path}{os.sep}{dset_name}_psig.png')

        if show_plot:
            plt.show()
        plt.close()

    psignifit_dict = {'data': data_np, 'csv_path': bin_data_dict['csv_path'],
                      'dset_name': dset_name,
                      'save_path': save_path, 'save_plot': save_plot,
                      'sig_name': sig_name, 'est_type': est_type,
                      'exp_type': options['expType'], 'expN': options['expN'],
                      'target_threshold': target_threshold,
                      'Threshold': threshold, 'slope_at_target': slope_at_target}

    print('\n*** finished run_psignifit() ***\n')

    return fit_curve_plot, psignifit_dict


# # # # # # # # #


def results_to_psignifit(csv_path, save_path, duration, 
                         # sep,
                         p_run_name,
                         stair_col='stair', stair_levels=None,
                         thr_col='probeSpeed', resp_col='response',
                         quartile_bins=True, n_bins=10, save_np=False,
                         target_threshold=.5,
                         sig_name='norm', est_type='MAP',
                         save_plot=True, show_plot=False,
                         verbose=True):
    """
    Function to fit curve with psignifit to raw_data.csv in one go.  It calls
    results_csv_to_np_for_psignifit() and run_psignifit().

    :param csv_path: path to results csv
    :param duration: which duration are results for?
    # :param sep: which separation are results for?
    :param p_run_name: participant and run name e.g., Kim1, Nick4 etc
    :param thr_col: name of column containing thresholds
    :param resp_col: name of column containing thresholds
    :param stair_col: name of column containing separations: use 'stair' if there
        is no separation column.
    :param stair_levels: default=None - in which case will access sep from stair_col.
        If there is no separation column in df, then enter
        a list of stair level(s) to analyse (e.g., for sep=18, use stair_levels=[0, 1]).
    :param quartile_bins: If True, will use pd.qcut for bins containing an equal
        number of items based on the distribution of thresholds.  i.e., bins will
        not be of equal size intervals but will have equal value count.  If False,
        will use pd.cut for bins of equal size based on values, but value count may vary.
    :param n_bins: default 10.
    :param save_np: default: False.  If True, save the numpy array to save_path
    :param save_path: path to save plot and dict
    :param target_threshold: threshold if this percentage correct
    :param sig_name: default: 'norm', can also choose 'logistic'.
    :param est_type: default: 'MAP' (maximum a postieriori), can also choose 'mean' (posterior mean).
    :param save_plot: default: True.
    :param show_plot: default: False.  Display plot on sceen. Useful if doing a
        single pot or not saving, but don't use for multiple plots as it slows
        things down and eats memory.
    :param verbose: if True will print progress to screen

    :return: figure of fitted curve and dict of details
    """

    print('\n*** running results_to_psignifit() ***')

    if save_np is False:
        save_np_path = None
    else:
        save_np_path = save_path
        if save_path is None:
            tail, head = os.path.split(csv_path)
            save_path, head = os.path.split(tail)
            print(f'\nNo save_path given, saving to:\n{save_path}')

    results_np, bin_data_dict = results_csv_to_np_for_psignifit(csv_path=csv_path,
                                                                duration=duration, 
                                                                # sep=sep,
                                                                p_run_name=p_run_name,
                                                                stair_col=stair_col,
                                                                stair_levels=stair_levels,
                                                                thr_col=thr_col,
                                                                resp_col=resp_col,
                                                                quartile_bins=quartile_bins,
                                                                n_bins=n_bins,
                                                                save_np_path=save_np_path,
                                                                verbose=True
                                                                )

    if verbose:
        print(f'\nresults_np:|bin min|resp_in|n_trials\n{results_np}')
        print(f'\nbin_data_dict: ')
        for k, v in bin_data_dict.items():
            print(f'{k}: {v}')

    fit_curve_plot, psignifit_dict = run_psignifit(data_np=results_np,
                                                   bin_data_dict=bin_data_dict,
                                                   save_path=save_path,
                                                   target_threshold=target_threshold,
                                                   sig_name=sig_name,
                                                   est_type=est_type,
                                                   save_plot=save_plot,
                                                   show_plot=show_plot,
                                                   verbose=True)

    print('\n*** finished results_to_psignifit() ***\n')

    return fit_curve_plot, psignifit_dict


# # # # # #


# def get_psignifit_threshold_df(root_path, p_run_name, csv_name, n_bins=9, q_bins=True,
#                                thr_col='probeSpeed', resp_col='response',
#                                stair_col='stair',
#                                dur_list=None,
#                                stair_list=None,
#                                # group=None,
#                                target_threshold=.5,
#                                cols_to_add_dict=None, save_name=None,
#                                verbose=True):
#     """
#     Function to make a dataframe (stair x duration) of psignifit threshold values for an entire run.
#
#     :param root_path: path to folder containing duration folders
#     :param p_run_name: Name of this run directory where csv is stored (e.g., P6a-Kim or P6b-Kim etc)
#     :param csv_name: Dataframe to analyse or Name of results csv to load (e.g., Kim1, Kim2 etc)
#     :param n_bins: Default=10. Number of bins to use.
#     :param q_bins: Default=True. If True, uses quartile bins, if false will use equally space bins.
#     :param stair_col: name of column containing separations: use 'stair' if there
#         is no separation column.
#     :param dur_list: Default=None. list of duration values.  If None passed will use default values.
#     :param stair_list: Default=None.  List of stair values.  If None passed will use defaults.
#     :param target_threshold: Accuracy to get threshold for.
#     :param cols_to_add_dict: add dictionary of columns to insert to finished df (header=key, column=value)
#     :param save_name: Pass a name to save output or if None will save as 'psignifit_thresholds'.
#     :param verbose: Print progress to screen
#
#     :return: Dataframe of thresholds from psignifit for each duration and stair.
#     """
#
#     print('\n*** running get_psignifit_threshold_df() ***')
#
#     if dur_list is None:
#         dur_list = [1, 4, 6, 9]
#     dur_name_list = dur_list
#
#     if stair_list is None:
#         stair_list = [0, 1, 2, 3]
#
#     thr_array = np.zeros(shape=[len(stair_list), len(dur_list)])
#
#     # identify whether csv_name is actaully a csv_name or infact a dataframe ready to use.
#     load_csv = True
#     if type(csv_name) is str:
#         if csv_name[-4:] == '.csv':
#             csv_name = csv_name[:-4]
#     elif type(csv_name) is pd.core.frame.DataFrame:
#         load_csv = False
#     else:
#         raise TypeError(f'csv_name should be a string or df, not {type(csv_name)}')
#
#     # loop through duration values
#     for dur_idx, duration in enumerate(dur_list):
#         if verbose:
#             print(f"\n{dur_idx}: duration: {duration}")
#
#         # get df for this duration only
#         if load_csv:
#             dur_df = pd.read_csv(f'{root_path}{os.sep}{p_run_name}'
#                                  f'{os.sep}probeDur{duration}/{csv_name}.csv')
#             if 'Unnamed: 0' in list(dur_df):
#                 dur_df.drop('Unnamed: 0', axis=1, inplace=True)
#         else:
#             dur_df = csv_name[csv_name['probe_dur_ms'] == duration]
#
#         if verbose:
#             print(f'\nrunning analysis for {p_run_name}\n')
#             print(f"dur_df:\n{dur_df}")
#
#         # stair_list = sorted(list(dur_df['stair'].unique()))
#         # print(f"stair_list: {stair_list}")
#         #
#         # if len(stair_list) != len(stair_list):
#         #     raise ValueError(f'Number of stairs ({len(stair_list)}) does not '
#         #                      f'match number of separations ({len(stair_list)}).\n'
#         #                      f'Please enter stair_list when calling get_psignifit_threshold_df()')
#
#         # loop through stairs for this duration
#         for stair_idx, stair in enumerate(stair_list):
#
#             # get df just for one stair at this duration
#             stair_df = dur_df[dur_df[stair_col] == stair]
#             if verbose:
#                 print(f'\nstair_df ({stair_col}={stair}, duration={duration}):\n{stair_df}')
#
#                 print(f'response "in" = {stair_df["response"].sum()}')
#
#
#
#             # # # test with csv to numpy
#             # yes script now works directly with df, don't need to load csv.
#             # now move on to doing full thing
#
#             # stair = stair_list[stair_idx]
#             # stair_levels = [stair]
#             print(f'\nstair_col: {stair_col}, stair_levels: {[stair]}')
#
#             # # for all in one function
#             # # # # #
#             print(f'root_path: {root_path}')
#             save_path = os.path.join(root_path, p_run_name)
#             print(f'save_path: {save_path}')
#
#             if stair in [0, 2]:
#                 sig_name = 'neg_gauss'
#             else:
#                 sig_name = 'norm'
#             # sig_name = 'norm'
#
#             fit_curve_plot, psignifit_dict = results_to_psignifit(csv_path=stair_df,
#                                                                   save_path=save_path,
#                                                                   duration=duration,
#                                                                   # stair=stair,
#                                                                   p_run_name=p_run_name,
#                                                                   stair_col=stair_col, stair_levels=[stair],
#                                                                   thr_col=thr_col, resp_col=resp_col,
#                                                                   quartile_bins=q_bins, n_bins=n_bins,
#                                                                   save_np=False, target_threshold=target_threshold,
#                                                                   sig_name=sig_name, est_type='MAP',
#                                                                   save_plot=True, show_plot=True,
#                                                                   verbose=verbose
#                                                                   )
#
#             # append result to zeros_df
#             threshold = psignifit_dict['Threshold']
#             thr_array[stair_idx, dur_idx] = threshold
#
#     # save zeros df - run and q_bin in name.
#     print(f'thr_array:\n{thr_array}')
#
#     # make dataframe from array
#     thr_df = pd.DataFrame(thr_array, columns=dur_name_list)
#     thr_df.insert(0, stair_col, stair_list)
#
#     if cols_to_add_dict is not None:
#         for idx, (header, col_vals) in enumerate(cols_to_add_dict.items()):
#             thr_df.insert(idx+1, header, col_vals)
#
#     if verbose:
#         print(f"thr_df:\n{thr_df}")
#
#     # save threshold array
#     if save_name is None:
#         thr_filename = f'psignifit_thresholds'
#     else:
#         thr_filename = save_name
#     # if group is not None:
#     #     thr_filename = f'g{group}_{thr_filename}'
#     thr_filename = f'{thr_filename}.csv'
#
#
#     thr_filepath = os.path.join(root_path, p_run_name, thr_filename)
#     print(f'saving psignifit_thresholds.csv to {thr_filepath}')
#     thr_df.to_csv(thr_filepath, index=False)
#
#     print('\n*** finished get_psignifit_threshold_df() ***\n')
#
#     return thr_df

# # # ##############
# root_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/radial_flow_exp'
# p_run_name = 'Nick_3'
# thr_df = get_psignifit_threshold_df(root_path=root_path, p_run_name=p_run_name,
#                                     csv_name=p_run_name,
#                                     n_bins=10, q_bins=True,
#                                     dur_list=None, stair_list=None, verbose=True)

def psignifit_thr_df_Oct23_flow(save_path, p_run_name, run_df, cond_cols_list,
                                thr_col='probe_cm_p_sec',
                                resp_col='response',
                                wide_df_cols=None,
                                n_bins=9, q_bins=True,
                                conf_int=True, thr_type='Bayes',
                                plot_both_curves=False,
                                save_name=None,
                                show_plots=False, save_plots=True,
                                verbose=True):
    """
    Function to make a dataframe (stair x isi) of psignifit threshold values for an entire run.

    :param save_path: directory to save thresholds csv to.
    :param p_run_name: Name of this run directory where csv is stored (e.g., P6a-Kim or P6b-Kim etc)
    :param run_df: Dataframe to analyse or Name of results csv to load (e.g., Kim1, Kim2 etc)
    :param cond_cols_list: List of column names to loop through to get unique conditions.
    :param n_bins: Default=10. Number of bins to use.
    :param q_bins: Default=True. If True, uses quartile bins, if false will use equally space bins.
    :param thr_col: name of column containing thresholds.
    :param resp_col: name of column containing responses that update the staircase (1, 0, not key pressed).
    :param sep_col: name of column containing separations: use 'stair' if there
        is no separation column.
    :param thr_col: name of column containing DV: e.g., 'probeLum' or 'NEW_probeLum'.

    :param conf_int: default: True.  Save and plot confidence/credible intervals
    :param thr_type: default: 'Bayes'.  This gets threshold estimate from 'Fit' in results dict.
            Can also pass 'CI95' to get threshold estimate from getThreshold.
            Analysis of 1probe results suggests 'Bayes' is most reliable (lowest SD).
    :param plot_both_curves: default: False.  If true will plot both 'Bayes' and 'CI95' on plots,
            but currently needs thr_type set to CI95.
    :param save_name: Pass a name to save output or if None will save as 'psignifit_thresholds'.
    :param show_plots: If True, will show plots immediately.
    :param save_plots: If True, will save plots.
    :param verbose: Print progress to screen

    :return: Dataframe of thresholds from psignifit for each ISI and stair.
    """

    print('\n*** running psignifit_thr_df_Oct23_flow() ***')
    if verbose:
        print(f'\nrunning analysis for {p_run_name}\n')

    if thr_type not in ['Bayes', 'CI95']:
        raise ValueError

    '''for each variable in cond_cols_list, get the number of values in the dataframe'''
    var_dict = {}
    n_vals_tup_list = []
    for variable in cond_cols_list:
        values = list(run_df[variable].unique())
        n_values = len(values)
        print(f"{variable}: {n_values}")
        var_dict[variable] = sorted(values)
        n_vals_tup_list.append((variable, n_values))
    print(f"var_dict:\n{var_dict}")
    print(f"n_vals_tup_list:\n{n_vals_tup_list}")

    # sort n_vals_tup_list in ascending order of number of values
    n_vals_tup_list.sort(key=lambda x: x[1])
    print(f"n_vals_tup_list:\n{n_vals_tup_list}")

    # remove 'probe_dur_ms' and 'flow_dir' from n_vals_tup_list
    short_n_vals_tup_list = [x for x in n_vals_tup_list if x[0] not in ['probe_dur_ms', 'flow_dir']]

    # # make sorted_var_dict, with 'probe_dur_ms' as the first variable, 'flow_dir' as the last,
    # # and all others in ascending order of number of values
    sorted_var_dict = {}
    # add variables to sorted_var_dict in same order as in short_n_vals_tup_list
    for variable, n_values in short_n_vals_tup_list:
        sorted_var_dict[variable] = var_dict[variable]
    if 'flow_dir' in var_dict:
        sorted_var_dict['flow_dir'] = var_dict['flow_dir']
    sorted_var_dict['probe_dur_ms'] = var_dict['probe_dur_ms']
    print(f"sorted_var_dict:\n{sorted_var_dict}")

    #  to get all unique combinations of values for all variables in sorted_var_dict
    keys, values = zip(*sorted_var_dict.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print(f'permutations_dicts: (len = {len(permutations_dicts)}')

    # df shape will have thresholds for ISI in separate columns, and rows for all other permulations
    n_probe_dur_ms = len(run_df['probe_dur_ms'].unique())
    n_cols = n_probe_dur_ms
    n_rows = int(len(permutations_dicts) / n_probe_dur_ms)
    print(f"n_rows, n_cols: {n_rows}, {n_cols}")

    long_thr_list = []

    # loop through run_df, running psignifit for each unique combination of variables
    # select rows in run_df where the values in each row match the values in each dict in permutations_dicts
    # for each dict in permutations_dicts, get the rows in run_df where the values in each row match the values in the dict
    for this_cond_dict in permutations_dicts:
        print(f"\nthis_cond_dict: {this_cond_dict}")

        this_cond_df = run_df.copy()
        for key, value in this_cond_dict.items():
            this_cond_df = this_cond_df[this_cond_df[key] == value]

        print(f"this_cond_df: ({this_cond_df.shape})\n{this_cond_df.head()}\n({list(this_cond_df.columns)})")

        # raise value error if the number of rows != 25
        if this_cond_df.shape[0] != 25:
            # raise ValueError(f"this_cond_df has {this_cond_df.shape[0]} trials, not 25")
            print(f"this_cond_df has {this_cond_df.shape[0]} trials, not 25")
            pass
        else:

            # turn this dict into a string called cond_info. This will be used for fig titles and saving figs
            # for each value, if it is a float close to an integer, convert it to an integer if less than +/- 0.01
            # otherwise, round it to 2 decimal places.
            cond_info = f'{p_run_name}'
            for k, v in this_cond_dict.items():
                if isinstance(v, float):
                    if abs(v - round(v)) < 0.01:
                        v = int(round(v))
                    else:
                        v = round(v, 2)
                cond_info += f'_{k}_{v}'
            print(f"cond_info: {cond_info}")

            # run psignifit
            # get results in correct numpy format
            results_np, bin_data_dict = csv_to_np_for_psignifit_Oct23(csv_path=this_cond_df,
                                                                      p_run_name=p_run_name,
                                                                      dataset_name=cond_info,
                                                                      thr_col=thr_col, resp_col=resp_col,
                                                                      quartile_bins=q_bins, n_bins=n_bins,
                                                                      save_np_path=None,
                                                                      verbose=True)

            if verbose:
                print(f'\nresults_np:|bin middle|n_correct|n_trials\n{results_np}')
                print(f'\nbin_data_dict: ')
                for k, v in bin_data_dict.items():
                    print(k, v)

            if save_plots:
                # if 'curve_plots' dir does not exist, make it
                if not os.path.exists(os.path.join(save_path, 'curve_plots')):
                    os.makedirs(os.path.join(save_path, 'curve_plots'))
                save_plots_path = os.path.join(save_path, 'curve_plots')

            fit_curve_plot, psignifit_dict = run_psignifit(data_np=results_np,
                                                           bin_data_dict=bin_data_dict,
                                                           target_threshold=.5,
                                                           sig_name='norm', est_type='MAP',
                                                           # conf_int=conf_int,
                                                           # thr_type=thr_type,
                                                           # plot_both_curves=plot_both_curves,
                                                           save_path=save_plots_path,
                                                           # save_plots=save_plots,
                                                           # show_plots=show_plots,
                                                           verbose=True)

            print(f"\npsignifit_dict:\n")
            for k, v in psignifit_dict.items():
                print(k, v)

            # add threshold and conditions info to long_thr_list
            this_cond_dict['thr'] = psignifit_dict['Threshold']
            long_thr_list.append(this_cond_dict)

    # convert long_thr_list to df
    long_thr_df = pd.DataFrame(long_thr_list)
    print(f"\nlong_thr_df:\n{long_thr_df}")

    if wide_df_cols:
        wide_df_idx_cols = [i[0] for i in n_vals_tup_list if i[0] != wide_df_cols]

        # convert long_thr_df to wide_thr_df, with ISI_ms as columns showing thr, in format f'ISI_{int(probe_dur_ms)}'
        wide_thr_df = long_thr_df.pivot_table(index=wide_df_idx_cols, columns='probe_dur_ms', values='thr')

        # get list of column names
        wide_thr_df_cols = list(wide_thr_df.columns)
        print(f"\nwide_thr_df_cols:\n{wide_thr_df_cols}")

        # if any col name is a float with .0, convert to int, else round 2 decimal places
        wide_thr_df_cols = [int(i) if i.is_integer() else round(i, 2) for i in wide_thr_df_cols]

        # append 'ISI_' to each col name
        wide_thr_df_cols = [f'{wide_df_cols}_{i}' for i in wide_thr_df_cols]

        # apply new column names
        wide_thr_df.columns = wide_thr_df_cols

        print(f"\nwide_thr_df:\n{wide_thr_df}")

        # reset index
        wide_thr_df.reset_index(inplace=True)
        print(f"\nwide_thr_df:\n{wide_thr_df}")

        thr_df = wide_thr_df
    else:
        thr_df = long_thr_df

    # save threshold array
    if save_name is None:
        thr_filename = f'psignifit_thresholds'
        other_name_prefix = 'psignifit'
    else:
        thr_filename = save_name
        other_name_prefix = save_name

    thr_filename = f'{thr_filename}.csv'

    thr_filepath = os.path.join(save_path, thr_filename)
    print(f'saving psignifit_thresholds.csv to {thr_filepath}')
    thr_df.to_csv(thr_filepath, index=False)

    print('\n*** finished psignifit_thr_df_Oct23_flow() ***\n')

    return thr_df