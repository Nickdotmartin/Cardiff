import os
import pickle

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
                                    quartile_bins=True, n_bins=10, save_np_path=None,
                                    verbose=True):

    """
    Converts a results csv to a numpy array for running psignifit.  Numpy array
    has 3 cols [stimulus level | nCorrect | ntotal].

    :param csv_path: path to results csv, or df.
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

    print('\n*** running results_csv_to_np_for_psignifit() ***')

    if type(csv_path) == pd.core.frame.DataFrame:
        raw_data_df = csv_path
    else:
        raw_data_df = pd.read_csv(csv_path, usecols=[sep_col, thr_col, resp_col])
    if verbose:
        print(f"raw_data:\n{raw_data_df}")

    # access separation either through separation column or stair column
    if stair_levels is None:
        if sep_col == 'stair':
            raise ValueError(f'sep_col is {sep_col} but no stair_levels given.  '
                             f'Enter a list of stair levels corresponding to the '
                             f'separation value or enter the name of the column '
                             f'showing separation values. ')
        stair_levels = [sep]
    
    raw_data_df = raw_data_df[raw_data_df[sep_col].isin(stair_levels)]
    if verbose:
        print(f"raw_data, stair_levels:{stair_levels}:\n{raw_data_df}")

    # dataset_name = f'{p_run_name}_ISI{isi}_sep{sep}_stair{stair_levels[0]}'
    dataset_name = f'{p_run_name}_ISI{isi}_{sep_col}{stair_levels[0]}'

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
        print(f'total_errors: {total_errors}')

    # put responses into bins (e.g., 10)
    # # use pd.qcut for bins containing an equal number of items based on distribution.
    # # i.e., bins will not be of equal size but will have equal value count
    if quartile_bins:
        bin_col, bin_labels = pd.qcut(x=raw_data_df[thr_col], q=n_bins,
                                      precision=3, retbins=True, duplicates='drop')
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

    # loop through bins and get number correct_per_bin
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
            # found_bins_left.append(round(bin_interval.left, 3))
            found_bins_left.append(bin_interval.left)


    data_df = pd.DataFrame(data_arr, columns=['bin_left', 'stim_level', 'n_correct', 'n_total'])
    data_df = data_df.sort_values(by='bin_left', ignore_index=True)
    # data_df['prop_corr'] = round(np.divide(data_df['n_correct'], data_df['n_total']).fillna(0), 2)
    data_df['prop_corr'] = np.divide(data_df['n_correct'], data_df['n_total']).fillna(0)

    if verbose:
        print(f"\ndata_df (with extra cols):\n{data_df}")
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

    print('\n*** finished results_csv_to_np_for_psignifit() ***\n')

    return data_np, bin_data_dict

# # # # # # # #

def run_psignifit(data_np, bin_data_dict, save_path, target_threshold=.75,
                  sig_name='norm', est_type='MAP', n_blocks=None,
                  conf_int=True,
                  save_plots=True, show_plots=False, verbose=True):

    """
    Will run psignifit on data_np to fit curve and output dict.

    :param data_np: np.array with three cols (no headers) [stimulus level | nCorrect | ntotal]
    :param bin_data_dict: dictionary of setting of experiment and analysis
        (e.g., isi, sep, stair) and for converting raw output into data_np, (e.g., n_bins, qcut).
    :param save_path: path to save plot and dict
    :param target_threshold: threshold if this percentage correct
    :param sig_name: default: 'norm', can also choose 'logistic'.
    :param est_type: default: 'MAP' (maximum a posteriori), can also choose 'mean' (posterior mean).
    :param n_blocks: default: None. Pass a value to set the number of unique
        probeLum values in the array or number of bins if greater than 25.
        e.g., if you want to have 30 bins enter 30.
    :param save_plots: default: True.
    :param show_plots: default: False.  Display plot on sceen. Useful if doing a
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
    options['expType'] = 'nAFC'
    options['expN'] = 4
    options['estimateType'] = est_type  # 'mean'  # 'MAP'  'mean'

    # number of bins/unique probeLum values
    if type(n_blocks) is int:
        if n_blocks > 25:
            options['nBlocks'] = n_blocks

    # set percent correct corresponding to the threshold
    options['threshPC'] = target_threshold

    '''psignifit recommend adjusting to a realistic (generous) range if an 
    adaptive proccedure is used.
    https://github.com/wichmann-lab/psignifit/wiki/Priors
    '''
    # options['stimulusRange'] = [21.2, 106]  # gives everything massive CIs


    if conf_int:
        options['confP'] = [.95]

    if verbose:
        print(f'data_np:\n{data_np}')
        print(f'options (dict): {options}')

    # results
    res = ps.psignifit(data_np, options)

    if verbose:
        print("res['options']")
        for k, v in res['options'].items():
            # if k in ['nblocks', 'stimulusRange']:
            print(f"{k}: {v}")

    # get threshold
    '''
    https://github.com/wichmann-lab/python-psignifit/wiki/How-to-Get-Thresholds-and-Slopes
    The only change you may observe is that the getThreshold function always returns a tuple 
    with the point estimate and confidence intervals. This could not be avoided as python functions 
    are not told which of their outputs are used by the calling program. 
    The warnings about these confidence intervals being a rough worst case approximation still 
    applies as spelled out in detail for the MATLAB version.

    To get only the threshold call:
    ps.getThreshold(res,percentCorrect)[0]
    
    so I probably can add conf_int = threshold[1]
    
    READ THEIR psignifit 4 PAPER FIRST
    '''

    res_thresh, width, res_lambda, res_gamma, eta = res['Fit']
    print(f'\nall_results:\n'
          f'res_thresh: {res_thresh}\n'  # 
          f'width: {width}\n'  # (difference between the 95 and the 5 percent point of the unscaled sigmoid)
          f'res_lambda: {res_lambda}\n'  # upper asymptote/lapse rate
          f'res_gamma: {res_gamma}\n'  # lower asymptote/guess rate
          f'eta: {eta}\n'  # scaling the extra variance introduced (a value near zero indicates your data to be basically binomially distributed, whereas values near one indicate severely overdispersed data)
          f'')

    if conf_int:
        # todo: comment this back in once I have bayesian estimates
        [threshold, CI_limits] = ps.getThreshold(res, target_threshold)
        print(f'ps.getThreshold(res, target_threshold): {ps.getThreshold(res, target_threshold)}')
        if options['estimateType'] == 'mean':
            # threshold = round(threshold[0], 2)
            threshold = threshold[0]
        else:
            # threshold = round(threshold, 2)
            threshold = threshold


        CI_limits = list(CI_limits[0])

        print(f'\nCI_limits:\n{CI_limits}\nfor options.confP: {options["confP"]}')

        # # # todo: get rid of this once I have bayesian estimates
        # threshold = res['Fit'][0]
        # CI_limits = list(res['conf_Intervals'][0])



    else:
        threshold = ps.getThreshold(res, target_threshold)
        if options['estimateType'] == 'mean':
            # threshold = round(threshold[0][0], 2)
            threshold = threshold[0][0]
        else:
            # threshold = round(threshold[0], 2)
            threshold = threshold[0]

    if verbose:
        print(f'\nthreshold: {threshold}')

    print(f'\n\nidiot check:\n'
          # f'all results; ps.psignifit(data_np, options)\n'
          # f'{ps.psignifit(data_np, options)}\n\n'
          f'res["Fit"]:\n{res["Fit"]}\n'
          f'res["conf_Intervals"]\n{res["conf_Intervals"]}\n')

    # slope_at_target = round(ps.getSlopePC(res, target_threshold), 2)
    slope_at_target = ps.getSlopePC(res, target_threshold)

    if verbose:
        print(f'slope_at_target: {slope_at_target}')

    # # # 4. Plot psychometric function
    dset_name = bin_data_dict['dset_name']

    if (show_plots is False) & (save_plots is False):
        print('not making plots')
        fit_curve_plot = None
    else:
        print(f'making plots (save_plots: {save_plots})')
        plt.figure()
        plt.title(f"{dset_name}: sig: {sig_name}, est: {est_type}\n"
                  f"threshPC: {target_threshold}, threshold: {round(threshold, 2)}, slope: {round(slope_at_target, 2)}")
        if conf_int:
            plotOptions = dict()
            # plotOptions['CIthresh'] = True
            # plotOptions['axisHandle'] = 'Fit'

            fit_curve_plot = ps.psigniplot.plotPsych(res, lineColor=[1, 0, 0], plotAsymptote=False,
                                                     CIthresh=True, showImediate=False)


            fit_patch = mpatches.Patch(color=[1, 0, 0], label='Bayesian Fit')
            CI_patch = mpatches.Patch(color='black', label='CI=95% Fit')
            fit_curve_plot.legend(handles=[fit_patch, CI_patch], loc='lower right')

            plotOptions = dict()
            CI_res = res.copy()
            CI_res['Fit'][0] = threshold
            CI_res['conf_Intervals'][0][0] = CI_limits[0]
            CI_res['conf_Intervals'][0][1] = CI_limits[1]

            print(f'plotOptions:\n{plotOptions}')

            fit_curve_plot2 = ps.psigniplot.plotPsych(CI_res, plotData=False, CIthresh=True,
                                                      showImediate=False)

        else:
            fit_curve_plot = ps.psigniplot.plotPsych(res, showImediate=False)

        if save_plots:
            plot_path = os.path.join(save_path, f'{dset_name}_psig.png')

            print(f'saving plot to: {plot_path}')
            plt.savefig(plot_path)

        if show_plots:
            plt.show()
        plt.close()

    # ps.psigniplot.plotMarginal(res)
    # ps.psigniplot.plot2D(res, 0,1)
    # ps.psigniplot.plotPrior(res)

    psignifit_dict = {'data': data_np, 'csv_path': bin_data_dict['csv_path'],
                      'dset_name': dset_name,
                      'save_path': save_path, 'save_plots': save_plots,
                      'sig_name': sig_name, 'est_type': est_type,
                      'exp_type': options['expType'], 'expN': options['expN'],
                      'target_threshold': target_threshold,
                      'stimulus_range': list(res['options']['stimulusRange']),
                      'Threshold': threshold, 'slope_at_target': slope_at_target,
                      'CI_limits': CI_limits, 'width': width, 'eta': eta}

    print('\n*** finished run_psignifit() ***\n')

    return fit_curve_plot, psignifit_dict


# # # # # # # # #


def results_to_psignifit(csv_path, save_path, isi, sep, p_run_name,
                         sep_col='stair', stair_levels=None,
                         thr_col='probeLum', resp_col='trial_response',
                         quartile_bins=False, n_bins=10, save_np=False,
                         target_threshold=.75,
                         sig_name='norm', est_type='MAP',
                         conf_int=True,
                         save_plots=True, show_plots=False,
                         verbose=True):
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
    :param save_np: default: False.  If True, save the numpy array to save_path
    :param save_path: path to save plot and dict
    :param target_threshold: threshold if this percentage correct
    :param sig_name: default: 'norm', can also choose 'logistic'.
    :param est_type: default: 'MAP' (maximum a postieriori), can also choose 'mean' (posterior mean).
    :param save_plots: default: True.
    :param show_plots: default: False.  Display plot on sceen. Useful if doing a
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
        print(f'\nresults_np:|bin min|n_correct|n_trials\n{results_np}')
        print(f'\nbin_data_dict: ')
        for k, v in bin_data_dict.items():
            print(k, v)

    fit_curve_plot, psignifit_dict = run_psignifit(data_np=results_np,
                                                   bin_data_dict=bin_data_dict,
                                                   save_path=save_path,
                                                   target_threshold=target_threshold,
                                                   sig_name=sig_name,
                                                   est_type=est_type,
                                                   conf_int=conf_int,
                                                   save_plots=save_plots,
                                                   show_plots=show_plots,
                                                   verbose=True)

    print('\n*** finished results_to_psignifit() ***\n')

    return fit_curve_plot, psignifit_dict


# # # # # #


def get_psignifit_threshold_df(root_path, p_run_name, csv_name, n_bins=10, q_bins=True,
                               sep_col='separation', thr_col='probeLum',
                               isi_list=None, sep_list=None,
                               group=None,
                               conf_int=True,
                               cols_to_add_dict=None, save_name=None,
                               show_plots=False, save_plots=True,
                               verbose=True):
    """
    Function to make a dataframe (stair x isi) of psignifit threshold values for an entire run.

    :param root_path: path to folder containing ISI folders
    :param p_run_name: Name of this run directory where csv is stored (e.g., P6a-Kim or P6b-Kim etc)
    :param csv_name: Dataframe to analyse or Name of results csv to load (e.g., Kim1, Kim2 etc)
    :param n_bins: Default=10. Number of bins to use.
    :param q_bins: Default=True. If True, uses quartile bins, if false will use equally space bins.
    :param sep_col: name of column containing separations: use 'stair' if there
        is no separation column.
    :param thr_col: name of column containing DV: e.g., 'probeLum' or 'NEW_probeLum'.
    :param isi_list: Default=None. list of ISI values.  If None passed will use default values.
    :param sep_list: Default=None.  List of separation values.  If None passed will use defualts.
    :param group: Default=None.  Pass a group id for exp1a to differentiate between
                    identical stairs  (e.g., 1&2, 3&4 etc).
    :param cols_to_add_dict: add dictionary of columns to insert to finished df (header=key, column=value)
    :param save_name: Pass a name to save output or if None will save as 'psignifit_thresholds'.
    :param save_plots: If True, will save plots.
    :param verbose: Print progress to screen

    :return: Dataframe of thresholds from psignifit for each ISI and stair.
    """

    print('\n*** running get_psignifit_threshold_df() ***')

    if isi_list is None:
        isi_list = [0, 1, 4, 6, 12, 24]
    isi_name_list = [f'ISI_{i}' for i in isi_list]

    if sep_list is None:
        sep_list = [18, 6, 3, 2, 1, 0]
        # sep_list = [18, 18, 6, 6, 3, 3, 2, 2, 1, 1, 0, 0]

    thr_array = np.zeros(shape=[len(sep_list), len(isi_list)])
    CI_limits_array = np.zeros(shape=[len(sep_list), len(isi_list)*2])
    CI_width_array = np.zeros(shape=[len(sep_list), len(isi_list)])
    eta_array = np.zeros(shape=[len(sep_list), len(isi_list)])

    # identify whether csv_name is actaully a csv_name or infact a dataframe ready to use.
    load_csv = True
    if type(csv_name) is str:
        if csv_name[-4:] == '.csv':
            csv_name = csv_name[:-4]
    elif type(csv_name) is pd.core.frame.DataFrame:
        load_csv = False
    else:
        raise TypeError(f'csv_name should be a string or df, not {type(csv_name)}')

    # loop through isi values
    for isi_idx, isi in enumerate(isi_list):
        if verbose:
            print(f"\n{isi_idx}: isi: {isi}")

        # get df for this isi only
        if load_csv:
            isi_df = pd.read_csv(f'{root_path}{os.sep}{p_run_name}'
                                 f'{os.sep}ISI_{isi}_probeDur2/{csv_name}.csv')
            if 'Unnamed: 0' in list(isi_df):
                isi_df.drop('Unnamed: 0', axis=1, inplace=True)
        else:
            isi_df = csv_name[csv_name['ISI'] == isi]

        if verbose:
            print(f'\nrunning analysis for {p_run_name}\n')
            print(f"isi_df:\n{isi_df}")

        # stair_list = sorted(list(isi_df['stair'].unique()))
        # print(f"stair_list: {stair_list}")
        # 
        # if len(stair_list) != len(sep_list):
        #     raise ValueError(f'Number of stairs ({len(stair_list)}) does not '
        #                      f'match number of separations ({len(sep_list)}).\n'
        #                      f'Please enter sep_list when calling get_psignifit_threshold_df()')

        # loop through stairs for this isi
        for sep_idx, sep in enumerate(sep_list):

            # get df just for one stair at this isi
            sep_df = isi_df[isi_df[sep_col] == sep]
            if verbose:
                print(f'\nsep_df ({sep_col}={sep}, isi={isi}:\n{sep_df}')

                print(f'n correct = {sep_df["trial_response"].sum()}')



            # # # test with csv to numpy
            # yes script now works directly with df, don't need to load csv.
            # now move on to doing full thing

            # sep = sep_list[sep_idx]
            # stair_levels = [stair]
            print(f'\nsep: {sep_col}, stair_levels: {[sep]}')

            # # for all in one function
            # # # # #
            print(f'root_path: {root_path}')
            save_path = os.path.join(root_path, p_run_name)
            # save_path = f'{root_path}{os.sep}{p_run_name}'
            print(f'save_path: {save_path}')

            fit_curve_plot, psignifit_dict = results_to_psignifit(csv_path=sep_df,
                                                                  save_path=save_path,
                                                                  isi=isi, sep=sep, p_run_name=p_run_name,
                                                                  sep_col=sep_col, stair_levels=[sep],
                                                                  thr_col=thr_col, resp_col='trial_response',
                                                                  quartile_bins=q_bins, n_bins=n_bins,
                                                                  save_np=False, target_threshold=.75,
                                                                  sig_name='norm', est_type='MAP',
                                                                  conf_int=conf_int,
                                                                  save_plots=save_plots, show_plots=show_plots,
                                                                  verbose=verbose
                                                                  )

            # append result to zeros_df
            threshold = psignifit_dict['Threshold']
            thr_array[sep_idx, isi_idx] = threshold

            if conf_int:
                CI_limits = psignifit_dict['CI_limits']
                # CI_limits_array[sep_idx, isi_idx*2] = round(CI_limits[0], 2)
                # CI_limits_array[sep_idx, (isi_idx*2)+1] = round(CI_limits[1], 2)
                CI_limits_array[sep_idx, isi_idx*2] = CI_limits[0]
                CI_limits_array[sep_idx, (isi_idx*2)+1] = CI_limits[1]


                # # CI_width = max limint minus min limit
                # print(f'type(CI_limits): {type(CI_limits)}')
                # print(f'CI_limits[0]: {CI_limits[0]}')
                # print(f'CI_limits[1]: {CI_limits[1]}')
                # print(f'type(CI_limits[0]): {type(CI_limits[0])}')
                # print(f'type(CI_limits[1]): {type(CI_limits[1])}')
                # print(f'psignifit_dict:')
                # for k, v in psignifit_dict.items():
                #     print(f"{k}: {v}")

                if not np.isnan(CI_limits).any():
                    CI_width_array[sep_idx, isi_idx] = CI_limits[1] - CI_limits[0]
                else:
                    print(f'found a NAN in CI_limits: {CI_limits}')
                    print(f"using psignifit_dict['stimulus_range']: {psignifit_dict['stimulus_range']}")
                    if CI_limits[0] != np.nan:
                        CI_width = psignifit_dict['stimulus_range'][1] - CI_limits[0]
                    elif CI_limits[1] != np.nan:
                        CI_width = CI_limits[1] - psignifit_dict['stimulus_range'][0]
                    else:
                        CI_width = psignifit_dict['stimulus_range'][1] - psignifit_dict['stimulus_range'][0]
                    print(f'CI_width: {CI_width}')
                    CI_width_array[sep_idx, isi_idx] = CI_width

                # scaling the extra variance introduced
                # (a value near zero indicates your data to be basically binomially distributed,
                # whereas values near one indicate severely overdispersed data)
                eta = psignifit_dict['eta']
                eta_array[sep_idx, isi_idx] = eta

    # save zeros df - run and q_bin in name.
    print(f'thr_array:\n{thr_array}')
    if conf_int:
        print(f'CI_limits_array:\n{CI_limits_array}')
        print(f'CI_width_array:\n{CI_width_array}')

    # make dataframe from array
    thr_df = pd.DataFrame(thr_array, columns=isi_name_list)
    thr_df.insert(0, sep_col, sep_list)

    if conf_int:
        CI_limits_headers = [[f'{i}_lo@95', f'{i}_hi@95'] for i in isi_name_list]
        CI_limits_headers = [j for sub in CI_limits_headers for j in sub]
        print(CI_limits_headers)
        CI_limits_df = pd.DataFrame(CI_limits_array, columns=CI_limits_headers)
        CI_limits_df.insert(0, sep_col, sep_list)

        CI_width_df = pd.DataFrame(CI_width_array, columns=isi_name_list)
        CI_width_df.insert(0, sep_col, sep_list)

        eta_df = pd.DataFrame(eta_array, columns=isi_name_list)
        eta_df.insert(0, sep_col, sep_list)

    if cols_to_add_dict is not None:
        for idx, (header, col_vals) in enumerate(cols_to_add_dict.items()):
            thr_df.insert(idx+1, header, col_vals)
            if conf_int:
                CI_limits_df.insert(idx+1, header, col_vals)
                CI_width_df.insert(idx+1, header, col_vals)
                eta_df.insert(idx+1, header, col_vals)

    if verbose:
        print(f"thr_df:\n{thr_df}")
        if conf_int:
            print(f"CI_limits_df:\n{CI_limits_df}")
            print(f"CI_width_df:\n{CI_width_df}")

    # save threshold array
    if save_name is None:
        thr_filename = f'psignifit_thresholds'
    else:
        thr_filename = save_name
    if group is not None:
        thr_filename = f'g{group}_{thr_filename}'
    thr_filename = f'{thr_filename}.csv'


    thr_filepath = os.path.join(root_path, p_run_name, thr_filename)
    print(f'saving psignifit_thresholds.csv to {thr_filepath}')
    thr_df.to_csv(thr_filepath, index=False)

    if conf_int:
        CI_limits_filename = f'psignifit_CI_limits.csv'
        CI_limits_filepath = os.path.join(root_path, p_run_name, CI_limits_filename)
        print(f'saving psignifit_CI.csv to {CI_limits_filepath}')
        CI_limits_df.to_csv(CI_limits_filepath, index=False)

        CI_width_filename = f'psignifit_CI_width.csv'
        CI_width_filepath = os.path.join(root_path, p_run_name, CI_width_filename)
        print(f'saving psignifit_CI_width.csv to {CI_width_filepath}')
        CI_width_df.to_csv(CI_width_filepath, index=False)

        eta_filename = f'psignifit_eta.csv'
        eta_filepath = os.path.join(root_path, p_run_name, eta_filename)
        print(f'saving psignifit_eta.csv to {eta_filepath}')
        eta_df.to_csv(eta_filepath, index=False)

    print('\n*** finished get_psignifit_threshold_df() ***\n')

    return thr_df

# # # ##############
# root_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/radial_flow_exp'
# p_run_name = 'Nick_3'
# thr_df = get_psignifit_threshold_df(root_path=root_path, p_run_name=p_run_name,
#                                     csv_name=p_run_name,
#                                     n_bins=10, q_bins=True,
#                                     isi_list=None, sep_list=None, verbose=True)
