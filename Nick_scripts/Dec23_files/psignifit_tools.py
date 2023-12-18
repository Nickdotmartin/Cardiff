import os
import pickle
import itertools
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



def run_psignifit(data_np, bin_data_dict, save_path, target_threshold=.75,
                  sig_name='norm', est_type='MAP', n_blocks=None,
                  conf_int=True,  thr_type='Bayes', plot_both_curves=False,
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
    :param conf_int: default: True.  Save and plot confidence/credible intervals
    :param thr_type: default: 'Bayes'.  This gets threshold estimate from 'Fit' in results dict.
            Can also pass 'CI95' to get threshold estimate from getThreshold.
            Analysis of 1probe results suggests 'Bayes' is most reliable (lowest SD).
    :param plot_both_curves: default: False.  If true will plot both 'Bayes' and 'CI95' on plots,
            but currently needs thr_type set to CI95.
    :param show_plots: default: False.  Display plot on sceen. Useful if doing a
        single pot or not saving, but don't use for multiple plots as it slows
        things down and eats memory.
    :param save_plots:  default=True.  Will save plots.
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

    if conf_int:
        options['confP'] = [.95]

    if verbose:
        print(f'data_np {np.shape(data_np)}:\n{data_np}')
        print(f'options (dict): {options}')

    # results
    res = ps.psignifit(data_np, options)

    if verbose:
        print("res['options']")
        for k, v in res['options'].items():
            print(f"{k}: {v}")

    '''
    get thresholds
    
    I have two methods for getting the thresholds.  
    1. The first (I've called 'Bayes' is described in the basic usage wiki 
    https://github.com/wichmann-lab/python-psignifit/wiki/Basic-Usage
    This calls the 'Fit' parameter of the results dictionary and associated CIs.
    
    2. The second (I've called 'CI95') is described in the get Thresholds Wiki 
    https://github.com/wichmann-lab/python-psignifit/wiki/How-to-Get-Thresholds-and-Slopes
    This calls the getThreshold function which takes the results dict and percentCorrect as variables.  
    
    I can do either, although, as it stands the Bayes method with trimmed means performs the best, 
    that is, smallest SD across 8 sets of 1probe estimates per participant.
    '''

    # # 1. Bayes threshold and CI (from results dict 'Fit')

    Bayes_thr, width, res_lambda, res_gamma, eta = res['Fit']

    Bayes_CI_limits = list(res['conf_Intervals'][0])

    print(f'\nall_results:\n'
          f'Bayes_thr: {Bayes_thr}\n'  # 
          f'width: {width}\n'  # (difference between the 95 and the 5 percent point of the unscaled sigmoid)
          f'res_lambda: {res_lambda}\n'  # upper asymptote/lapse rate
          f'res_gamma: {res_gamma}\n'  # lower asymptote/guess rate
          f'eta: {eta}\n'  # scaling the extra variance introduced (a value near zero indicates your data to be 
          # basically binomially distributed, whereas values near one indicate severely overdispersed data)
          f'Bayes_CI_limits: {Bayes_CI_limits}')

    # # 2. CI95 threshold from getThreshold function
    [CI95_thr, CI95_limits] = ps.getThreshold(res, target_threshold)
    print(f'ps.getThreshold(res, target_threshold): {ps.getThreshold(res, target_threshold)}')
    if options['estimateType'] == 'mean':
        CI95_thr = CI95_thr[0]
    else:
        CI95_thr = CI95_thr
    CI95_limits = list(CI95_limits[0])
    print(f'\nCI95_limits:\n{CI95_limits}\nfor options.confP: {options["confP"]}')

    # select thr_type
    if thr_type == 'Bayes':
        threshold = Bayes_thr
        CI_limits = Bayes_CI_limits
    elif thr_type == 'CI95':
        threshold = CI95_thr
        CI_limits = CI95_limits
    else:
        raise ValueError("thr_type must be in ['Bayes', 'CI95']")

    if verbose:
        print(f'\nthreshold: {threshold}\n'
              f'CI_limits: {CI_limits}')

    '''I'm not sure whether the slope relates to Bayes or CI95 slope???'''
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

        if conf_int:

            if not plot_both_curves:

                print(f'plotting {thr_type}')

                plt.title(f"{dset_name}: sig: {sig_name}, est: {est_type}\n"
                          f"threshPC: {target_threshold}, {thr_type} thr: {round(threshold, 2)}, "
                          f"slope: {round(slope_at_target, 2)}")

                if thr_type == 'CI95':

                    CI_res = res.copy()
                    CI_res['Fit'][0] = threshold
                    CI_res['conf_Intervals'][0][0] = CI_limits[0]
                    CI_res['conf_Intervals'][0][1] = CI_limits[1]

                    fit_curve_plot = ps.psigniplot.plotPsych(CI_res, plotData=False, CIthresh=True,
                                                             showImediate=False, fontName='sans-serif')

                elif thr_type == 'Bayes':

                    fit_curve_plot = ps.psigniplot.plotPsych(res, plotAsymptote=True, fontName='sans-serif',
                                                             CIthresh=True, showImediate=False)

            else:

                if thr_type == 'Bayes':
                    raise ValueError('plot_both_curves currently only set up to run with CI95 as thr type.')

                print(f'plotting both threshold types')
                plt.title(f"{dset_name}: sig: {sig_name}, est: {est_type}\n"
                          f"threshPC: {target_threshold}, Bayes thr: {round(res['Fit'][0], 2)}, "
                          f"CI95 thr: {round(threshold, 2)}")
                fit_curve_plot = ps.psigniplot.plotPsych(res, lineColor=[1, 0, 0], plotAsymptote=False,
                                                         CIthresh=True, showImediate=False, fontName='sans-serif')

                fit_patch = mpatches.Patch(color=[1, 0, 0], label='Bayesian Fit')
                CI_patch = mpatches.Patch(color='black', label='CI=95% Fit')
                fit_curve_plot.legend(handles=[fit_patch, CI_patch], loc='lower right')

                CI_res = res.copy()
                CI_res['Fit'][0] = threshold
                CI_res['conf_Intervals'][0][0] = CI_limits[0]
                CI_res['conf_Intervals'][0][1] = CI_limits[1]


        else:
            fit_curve_plot = ps.psigniplot.plotPsych(res, showImediate=False, fontName='sans-serif')

        if save_plots:
            plot_path = os.path.join(save_path, f'{dset_name}_psig.png')

            print(f'saving plot to: {plot_path}')
            plt.savefig(plot_path)

        if show_plots:
            plt.show()
        plt.close()


    psignifit_dict = {'data': data_np, 'csv_path': bin_data_dict['csv_path'],
                      'dset_name': dset_name,
                      'save_path': save_path, 'save_plots': save_plots,
                      'sig_name': sig_name, 'est_type': est_type,
                      'exp_type': options['expType'], 'expN': options['expN'],
                      'thr_type': thr_type,
                      'target_threshold': target_threshold,
                      'stimulus_range': list(res['options']['stimulusRange']),
                      'Threshold': threshold, 'slope_at_target': slope_at_target,
                      'CI_limits': CI_limits, 'width': width, 'eta': eta}

    print('\n*** finished run_psignifit() ***\n')

    return fit_curve_plot, psignifit_dict



#############################################################
# todo: change csv_path to cond_df
def csv_to_np_for_psignifit_Oct23(csv_path,
                                  # isi, sep,
                                  p_run_name,
                                  # sep_col='stair',
                                  # stair_levels=None,
                                  dataset_name=None,
                                  # hue_name=None, hue_level=None,
                                  thr_col='probeLum', resp_col='trial_response',
                                  quartile_bins=True, n_bins=9, save_np_path=None,
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

    # todo: sort out stair_levels variable - this is confusing and from a time when
    #  I couldn't decide between separation and stairs.  I should rename or refactor this.

    if type(csv_path) == pd.core.frame.DataFrame:
        raw_data_df = csv_path
    else:
        # raw_data_df = pd.read_csv(csv_path, usecols=[sep_col, thr_col, resp_col])
        raw_data_df = pd.read_csv(csv_path, usecols=[thr_col, resp_col])
    if verbose:
        print(f"raw_data:\n{raw_data_df}")


    # get useful info
    n_rows, n_cols = raw_data_df.shape
    thr_min = raw_data_df[thr_col].min()
    thr_max = raw_data_df[thr_col].max()
    if verbose:
        print(f"\nn_rows: {n_rows}, n_cols: {n_cols}")
        print(f"{thr_col} min, max: {thr_min}, {thr_max}")

    # check total_n_correct in raw_df
    total_n_correct = sum(list(raw_data_df[resp_col]))
    total_errors = (raw_data_df[resp_col] == 0).sum()
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
    for idx, bin_interval in enumerate(bins):
        this_bin_vals = [bin_interval.left, bin_interval.right]
        this_bin_midpoint = bin_interval.left + ((bin_interval.right - bin_interval.left) / 2)
        this_bin_df = raw_data_df.loc[raw_data_df['bin_col'] == bin_interval]
        if this_bin_df.empty:
            data_arr.append([this_bin_midpoint, this_bin_vals, 0, 0])
        else:
            correct_per_bin = this_bin_df[resp_col].sum()
            data_arr.append([this_bin_midpoint, this_bin_vals, correct_per_bin, bin_count[bin_interval]])


    # make data df
    data_df = pd.DataFrame(data_arr, columns=['bin_middle', 'stim_level', 'n_correct', 'n_total'])
    data_df = data_df.sort_values(by='bin_middle', ignore_index=True)
    data_df['prop_corr'] = np.divide(data_df['n_correct'], data_df['n_total']).fillna(0)
    if verbose:
        print(f"\ndata_df (with extra cols):\n{data_df}")
    data_df = data_df.drop(columns=['stim_level', 'prop_corr'])

    # # # # 2. convert data into format for analysis: 3 cols [stimulus level | nCorrect | ntotal]
    data_np = data_df.to_numpy()

    bin_data_dict = {'csv_path': csv_path,
                     'dset_name': dataset_name,
                     'quartile_bins': quartile_bins, 'n_bins': n_bins,
                     'save_np_path': save_np_path}

    if save_np_path is not None:
        np.savetxt(os.path.join(save_np_path, f"{dataset_name}.csv"), data_np, delimiter=",")

        with open(os.path.join(save_np_path, f"{dataset_name}.pickle", 'wb')) as handle:
            pickle.dump(bin_data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('\n*** finished results_csv_to_np_for_psignifit() ***\n')

    return data_np, bin_data_dict



def psignifit_thr_df_Oct23(save_path, p_run_name, run_df, cond_cols_list,
                           thr_col='probeLum',
                           resp_col='trial_response',
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

    print('\n*** running psignifit_thr_df_Oct23() ***')
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

    # remove 'isi_ms' and 'congruent' from n_vals_tup_list
    short_n_vals_tup_list = [x for x in n_vals_tup_list if x[0] not in ['isi_ms', 'congruent']]

    # # make sorted_var_dict, with 'isi_ms' as the first variable, 'congruent' as the last,
    # # and all others in ascending order of number of values
    sorted_var_dict = {}
    # add variables to sorted_var_dict in same order as in short_n_vals_tup_list
    for variable, n_values in short_n_vals_tup_list:
        sorted_var_dict[variable] = var_dict[variable]
    sorted_var_dict['congruent'] = var_dict['congruent']
    sorted_var_dict['isi_ms'] = var_dict['isi_ms']
    print(f"sorted_var_dict:\n{sorted_var_dict}")

    #  to get all unique combinations of values for all variables in sorted_var_dict
    keys, values = zip(*sorted_var_dict.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print(f'permutations_dicts: (len = {len(permutations_dicts)}')


    # df shape will have thresholds for ISI in separate columns, and rows for all other permulations
    n_isi_ms = len(run_df['isi_ms'].unique())
    n_cols = n_isi_ms
    n_rows = int(len(permutations_dicts) / n_isi_ms)
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
                if 'isi' in k.lower() and v == -1.0:
                    v = 'Conc'
                elif isinstance(v, float):
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
                                  quartile_bins=q_bins, n_bins=n_bins, save_np_path=None,
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
                                                           target_threshold=.75,
                                                           sig_name='norm', est_type='MAP',
                                                           conf_int=conf_int,
                                                           thr_type=thr_type,
                                                           plot_both_curves=plot_both_curves,
                                                           save_path=save_plots_path,
                                                           save_plots=save_plots,
                                                           show_plots=show_plots,
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

        # convert long_thr_df to wide_thr_df, with ISI_ms as columns showing thr, in format f'ISI_{int(isi_ms)}'
        wide_thr_df = long_thr_df.pivot_table(index=wide_df_idx_cols, columns='isi_ms', values='thr')

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


    print('\n*** finished psignifit_thr_df_Oct23() ***\n')

    return thr_df