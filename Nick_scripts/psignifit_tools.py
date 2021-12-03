import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psignifit as ps

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

def results_csv_to_np_for_psignifit(csv_path, isi, sep, p_run_name, sep_col='stair',
                                    stair_levels=None, 
                                    thr_col='probeLum', resp_col='trial_response',
                                    quartile_bins=False, n_bins=10, save_np_path=None,
                                    verbose=True):

    """
    Converts a results csv to a numpy array for running psignifit.  Numpy array
    has 3 cols [stimulus level | nCorrect | ntotal].


    :param csv_path: path to results csv
    :param isi: which ISI are results for?
    :param sep: which separation are results for?
    :param p_run_name: participant and run name e.g., Kim1, Nick4 etc
    :param thr_col: name of column containing thresholds
    :param resp_col: name of column containing thresholds
    :param sep_col: name of column containing separations: use 'stair' if there
        is no separation column.       
    :param stair_levels: default=None - in which case will access sep from sep_col.
        If there is no separation column in df, then enter 
        a list of stair level(s) to analyse (e.g., for sep=18, use stair_levels=[0, 1]).
    :param quartile_bins: If True, will use pd.qcut for bins containing an equal
        number of items based on the distribution of thresholds.  i.e., bins will
        not be of equal size intervals but will have equal value count.  If False,
        will use pd.cut for bins of equal size based on values, but value count may vary.
    :param n_bins: default 10.
    :param save_np_path: default None.  will save to path if one is given.
    :param verbose: default True.  Will print progress to screen.

    :return:
        numpy array: for analysis in psignifit,
        psignifit_dict: contains details for psignifit e.g., for title, save plot etc.
    """

    print('*** running results_csv_to_np_for_psignifit ***')

    raw_data_df = pd.read_csv(csv_path, usecols=[sep_col, thr_col, resp_col])
    if verbose:
        print(f"raw_data:\n{raw_data_df.head()}")

    # access separation either through separation column or stair column
    if stair_levels is None:
        if sep_col == 'stair':
            raise ValueError(f'sep_col is {sep_col} but no stair_levels given.  '
                             f'Enter a list of stair levels corresponding to the '
                             f'separation value or enter the name of the column '
                             f'showing separation values. ')
        stair_levels = sep
    
    raw_data_df = raw_data_df[raw_data_df[sep_col].isin(stair_levels)]
    if verbose:
        print(f"raw_data, stair_levels:{stair_levels}:\n{raw_data_df.head()}")

    dataset_name = f'{p_run_name}_ISI{isi}_sep{sep}'

    # get useful info
    n_rows, n_cols = raw_data_df.shape
    thr_min = raw_data_df[thr_col].min()
    thr_max = raw_data_df[thr_col].max()
    if verbose:
        print(f"\nn_rows: {n_rows}, n_cols: {n_cols}")
        print(f"{thr_col} min, max: {thr_min}, {thr_max}")

    # check total_n_correct in raw_df
    total_n_correct = sum(list(raw_data_df['trial_response']))
    total_errors = (raw_data_df['trial_response'] == 0).sum()
    if verbose:
        print(f'total_n_correct: {total_n_correct}')
        print(f'total_errors: {total_errors}\n\n')

    # put responses into bins (e.g., 10)
    # # use pd.qcut for bins containing an equal number of items based on distribution.
    # # i.e., bins will not be of equal size but will have equal value count
    if quartile_bins:
        bin_col, bin_labels = pd.qcut(x=raw_data_df[thr_col], q=n_bins,
                                      precision=3, retbins=True)
    # # use pd.cut for bins of equal size based on values, but value count may vary.
    else:
        bin_col, bin_labels = pd.cut(x=raw_data_df[thr_col], bins=n_bins,
                                     precision=3, retbins=True, ordered=True)

    raw_data_df['bin_col'] = bin_col

    # get n_trials for each bin
    bin_count = pd.value_counts(raw_data_df['bin_col'])
    if verbose:
        print(f"\nbin_count (trials per bin):\n{bin_count}")

        # get bin intervals as list of type(pandas.Interval)
    bins = sorted([i for i in list(bin_count.index)])

    # loop through bins and get correct per bin
    data_arr = []
    found_bins_left = []
    for idx, bin_interval in enumerate(bins):
        # print(idx, bin_interval)
        this_bin_vals = [bin_interval.left, bin_interval.right]
        this_bin_df = raw_data_df.loc[raw_data_df['bin_col'] == bin_interval]
        if this_bin_df.empty:
            data_arr.append([bin_interval.left, this_bin_vals, 0, 0])
        else:
            # print(f'this_bin_df: {this_bin_df.shape}\n{this_bin_df}')
            correct_per_bin = this_bin_df['trial_response'].sum()
            # print(f'\tcorrect_per_bin: {correct_per_bin}/{this_bin_df.shape[0]}\n')
            data_arr.append([bin_interval.left, this_bin_vals, correct_per_bin, bin_count[bin_interval]])
            found_bins_left.append(round(bin_interval.left, 3))


    data_df = pd.DataFrame(data_arr, columns=['bin_left', 'stim_level', 'n_correct', 'n_total'])
    data_df = data_df.sort_values(by='bin_left', ignore_index=True)
    data_df['prop_corr'] = round(np.divide(data_df['n_correct'], data_df['n_total']).fillna(0), 2)
    if verbose:
        print(f"data_df (with extra cols):\n{data_df}")
    data_df = data_df.drop(columns=['stim_level', 'prop_corr'])

    # # # # 2. convert data into format for analysis: 3 cols [stimulus level | nCorrect | ntotal]
    data_np = data_df.to_numpy()
    # print(f"data:\n{data}")

    bin_data_dict = {'csv_path': csv_path, 'dset_name': dataset_name,
                     'isi': isi, 'sep': sep, 'p_run_name': p_run_name,
                     'stair_levels': stair_levels,
                     'quartile_bins': quartile_bins, 'n_bins': n_bins,
                     'save_np_path': save_np_path}

    if save_np_path is not None:
        np.savetxt(f"{save_np_path}{os.sep}{dataset_name}.csv", data_np, delimiter=",")
        with open(f"{save_np_path}{os.sep}{dataset_name}.pickle", 'wb') as handle:
            pickle.dump(bin_data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('*** finished results_csv_to_np_for_psignifit ***')


    return data_np, bin_data_dict

# # # # # # # #
# # # 1. load in CSV file.
# exp_csv_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/Kim/' \
#                'Nick_practice/P6a-Kim/ISI_24_probeDur2/Kim1.csv'
# isi=24
# participant_run_name = 'Kim1'
# sep=18
# stair_levels = [1, 2]
# q_bins=False
# n_bins=43
# results_np, bin_data_dict = results_csv_to_np_for_psignifit(csv_path=exp_csv_path,
#                                                              isi=isi, sep=18,
#                                                              p_run_name=participant_run_name,
#                                                              stair_levels=stair_levels,
#                                                              quartile_bins=q_bins, n_bins=n_bins,
#                                                              save_np_path=
#                                                             '/Users/nickmartin/Documents/'
#                                                             'PycharmProjects/Cardiff/Kim/' \
#                                                             f'Nick_practice/P6a-Kim')
#
# print(f'results_np: {results_np}')
#
#
# data = results_np
# target_threshold = .75
# sig_name = 'norm'  # 'logistic' 'norm'
# est_type = 'mean'  # 'MAP'
# dset_name = bin_data_dict['dset_name']
# csv_path = bin_data_dict['csv_path']
# save_path = bin_data_dict['save_np_path']
# if save_path is None:
#     tail, head = os.path.split(csv_path)
#     save_path, head = os.path.split(tail)
# save_plot=True


def run_psignifit(data_np, bin_data_dict, save_path, target_threshold=.75,
                  sig_name='norm', est_type='MAP', save_plot=True, verbose=True):

    """
    Will run psignifit on data_np to fit curve and output dict.

    :param data_np: np.array with three cols (no headers) [stimulus level | nCorrect | ntotal]
    :param bin_data_dict: dictionary of setting of experiment and analysis
        (e.g., isi, sep, stair) and for converting raw output into data_np, (e.g., n_bins, qcut).
    :param save_path: path to save plot and dict
    :param target_threshold: threshold if this percentage correct
    :param sig_name: default: 'norm', can also choose 'logistic'.
    :param est_type: default: 'MAP' (maximum a posteriori), can also choose 'mean' (posterior mean).
    :param save_plot: default: True.
    :param verbose:

    :return: figure of fitted curve and dict of details
    """

    print('*** running run_psignifit ***')

    # # To start psignifit you need to pass a dictionary, which specifies, what kind
    #      of experiment you did and any other parameters of the fit you might want
    options = dict()  # initialize as an empty dict

    options['sigmoidName'] = sig_name  # 'norm'  # 'logistic'
    options['expType'] = 'nAFC'
    options['expN'] = 4

    options['estimateType'] = est_type  # 'mean'  # 'MAP'  'mean'

    # set percent correct corresponding to the threshold
    options['threshPC'] = target_threshold

    if verbose:
        print(f'\n\noptions (dict): {options}')

    # results
    res = ps.psignifit(data_np, options)

    if verbose:
        print("res['options']")
        for k, v in res['options'].items():
            print(f"{k}: {v}")

    # get threshold
    threshold = ps.getThreshold(res, target_threshold)
    if options['estimateType'] == 'mean':
        threshold = round(threshold[0][0], 2)
    else:
        threshold = round(threshold[0], 2)
    if verbose:
        print(f'threshold: {threshold}')

    # # # 4. Plot psychometric function
    dset_name = bin_data_dict['dset_name']

    plt.figure()
    plt.title(f"{dset_name}: stair: {bin_data_dict['stair_levels']}\n"
              f"threshPC: {target_threshold}, threshold: {threshold}, "
              f"sig: {sig_name}, "
              f"est: {est_type}")
    fit_curve_plot = ps.psigniplot.plotPsych(res)

    if save_plot:
        print(f'saving plot to: {save_path}{os.sep}{dset_name}_psig.png')
        plt.savefig(f'{save_path}{os.sep}{dset_name}_psig.png')

    # ps.psigniplot.plotMarginal(res)
    # ps.psigniplot.plot2D(res, 0,1)
    # ps.psigniplot.plotPrior(res)

    slope_at_target = ps.getSlopePC(res, target_threshold)
    print(f'slope_at_target: {slope_at_target}')

    psignifit_dict = {'data': data_np, 'csv_path': bin_data_dict['csv_path'],
                      'dset_name': dset_name,
                      'save_path': save_path, 'save_plot': save_plot,
                      'sig_name': sig_name, 'est_type': est_type,
                      'exp_type': options['expType'], 'expN': options['expN'],
                      'target_threshold': target_threshold,
                      'Threshold': threshold, 'slope_at_target': slope_at_target}

    print('*** finished run_psignifit ***')

    return fit_curve_plot, psignifit_dict



# # # # # # # # #
# # # 1. load in CSV file.
# exp_csv_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/Kim/' \
#                'Nick_practice/P6a-Kim/ISI_4_probeDur2/Kim1.csv'
# isi=24
# participant_run_name = 'Kim1'
# sep=18
# stair_levels = [1, 2]
# q_bins=False
# n_bins=10
# results_np, bin_data_dict = results_csv_to_np_for_psignifit(csv_path=exp_csv_path,
#                                                              isi=isi, sep=18,
#                                                              p_run_name=participant_run_name,
#                                                              stair_levels=stair_levels,
#                                                              quartile_bins=q_bins, n_bins=n_bins,
#                                                              save_np_path=None)
#
# print(f'results_np:\n{results_np}')
# print('\nbin_data_dict')
# for k, v in bin_data_dict.items():
#     print(k, v)
# #
# # data = results_np
# target_threshold = .75
# sig_name = 'norm'  # 'logistic' 'norm'
# est_type = 'MAP'  # 'MAP'
# # dset_name = bin_data_dict['dset_name']
# csv_path = bin_data_dict['csv_path']
# save_path = bin_data_dict['save_np_path']
# print(f'check save_path: {save_path}')
#
# if save_path is None:
#     tail, head = os.path.split(csv_path)
#     save_path, head = os.path.split(tail)
#
# print(f'check save_path: {save_path}')
# save_plot=True
#
# fit_curve_plot, psignifit_dict = run_psignifit(data_np=results_np, bin_data_dict=bin_data_dict,
#                                                save_path=save_path, target_threshold=target_threshold,
#                                                sig_name=sig_name, est_type=est_type, save_plot=save_plot,
#                                                verbose=True)
#



def results_to_psignifit(csv_path, save_path, isi, sep, p_run_name,
                         sep_col='stair', stair_levels=None,
                         thr_col='probeLum', resp_col='trial_response',
                         quartile_bins=False, n_bins=10, save_np=False,
                         target_threshold=.75,
                         sig_name='norm', est_type='MAP', save_plot=True, verbose=True
                         ):

    """
    Function to fit curve with psignifit to raw_data.csv in one go.  It calls
    results_csv_to_np_for_psignifit() and run_psignifit().

    :param csv_path: path to results csv
    :param isi: which ISI are results for?
    :param sep: which separation are results for?
    :param p_run_name: participant and run name e.g., Kim1, Nick4 etc
    :param thr_col: name of column containing thresholds
    :param resp_col: name of column containing thresholds
    :param sep_col: name of column containing separations: use 'stair' if there
        is no separation column.
    :param stair_levels: default=None - in which case will access sep from sep_col.
        If there is no separation column in df, then enter
        a list of stair level(s) to analyse (e.g., for sep=18, use stair_levels=[0, 1]).
    :param quartile_bins: If True, will use pd.qcut for bins containing an equal
        number of items based on the distribution of thresholds.  i.e., bins will
        not be of equal size intervals but will have equal value count.  If False,
        will use pd.cut for bins of equal size based on values, but value count may vary.
    :param n_bins: default 10.
    :param save_np: default: False.  If True will save the numpy array to save_path
    :param save_path: path to save plot and dict
    :param target_threshold: threshold if this percentage correct
    :param sig_name: default: 'norm', can also choose 'logistic'.
    :param est_type: default: 'MAP' (maximum a postieriori), can also choose 'mean' (posterior mean).
    :param save_plot: default: True.
    :param verbose: if True will print progress to screen

    :return: figure of fitted curve and dict of details
    """

    print('*** running results_to_psignifit ***')

    if save_np is False:
        save_np_path = None
    else:
        save_np_path = save_path
        if save_path is None:
            tail, head = os.path.split(csv_path)
            save_path, head = os.path.split(tail)
            print(f'\nNo save_path given, saving to:\n{save_path}')

    results_np, bin_data_dict = results_csv_to_np_for_psignifit(csv_path=csv_path,
                                                                isi=isi, sep=sep,
                                                                p_run_name=p_run_name,
                                                                sep_col=sep_col,
                                                                stair_levels=stair_levels,
                                                                thr_col=thr_col,
                                                                resp_col=resp_col,
                                                                quartile_bins=quartile_bins,
                                                                n_bins=n_bins,
                                                                save_np_path=save_np_path,
                                                                verbose=True
                                                                )

    if verbose:
        print(f'results_np:\n{results_np}')
        print('\nbin_data_dict')
        for k, v in bin_data_dict.items():
            print(k, v)

    fit_curve_plot, psignifit_dict = run_psignifit(data_np=results_np,
                                                   bin_data_dict=bin_data_dict,
                                                   save_path=save_path,
                                                   target_threshold=target_threshold,
                                                   sig_name=sig_name,
                                                   est_type=est_type,
                                                   save_plot=save_plot,
                                                   verbose=True)

    print('*** finished results_to_psignifit ***')

    return fit_curve_plot, psignifit_dict



# # # # # #
# exp_csv_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/Kim/' \
#                'Nick_practice/P6a-Kim/ISI_24_probeDur2/Kim1.csv'
# save_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/Kim/' \
#                'Nick_practice/P6a-Kim'
# isi = 24
# participant_run_name = 'Kim1'
# sep = 18
# stair_levels = [1, 2]
# q_bins = False
# n_bins = 10
#
# fit_curve_plot, psignifit_dict = results_to_psignifit(csv_path=exp_csv_path, save_path=save_path,
#                                                       isi=isi, sep=sep, p_run_name=participant_run_name,
#                                                       sep_col='stair', stair_levels=stair_levels,
#                                                       thr_col='probeLum', resp_col='trial_response',
#                                                       quartile_bins=q_bins, n_bins=n_bins, save_np=False,
#                                                       target_threshold=.75,
#                                                       sig_name='norm', est_type='MAP', save_plot=True, verbose=True
#                                                       )
