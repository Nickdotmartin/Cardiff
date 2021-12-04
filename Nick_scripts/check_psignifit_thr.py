import os

import numpy as np
import pandas as pd

from psignifit_tools import results_to_psignifit

root_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/Kim'
run_folder_names = ['P6e-Kim', 'P6f-Kim']
# run_folder_names = ['P6a-Kim', 'P6b-Kim', 'P6c-Kim', 'P6d-Kim', 'P6e-Kim', 'P6f-Kim']
# root_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/Kim/Nick_practice'
# run_folder_names = ['P6a-Kim']

participant_name = 'Kim'
isi_list = [-1, 0, 2, 4, 6, 9, 12, 24]
isi_name_list = [f'isi{i}' for i in isi_list]
stair_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
sep_list = [18, 18, 6, 6, 3, 3, 2, 2, 1, 1, 0, 0, 99, 99]
verbose = False

q_bins = True
n_bins = 10

# loop through runs
for run_idx, run_dir in enumerate(run_folder_names):

    print(f'run {run_idx}: {run_dir}')

    thr_array = np.zeros(shape=[len(stair_list), len(isi_list)])


    # loop through ISI
    # loop through isi values
    for isi_idx, isi in enumerate(isi_list):
        if verbose:
            print(f"\n{isi_idx}: isi: {isi}")

        isi_name = isi_name_list[isi_idx]
        p_name = participant_name

        p_name = f'{participant_name}{run_idx+5}'
        print(f'p_name: {p_name}')

        # get df for this isi only
        isi_df = pd.read_csv(f'{root_path}{os.sep}{run_dir}'
                             f'{os.sep}ISI_{isi}_probeDur2/{p_name}.csv')

        print(f'\nrunning analysis for {p_name}, {run_dir}\n')

        # loop through stairs for this isi
        for stair_idx, stair in enumerate(stair_list):

            # get df just for one stair at this isi
            stair_df = isi_df[isi_df['stair'] == stair]
            if verbose:
                print(f'\nstair_df (stair={stair}, isi={isi}:\n'
                      f'{type(stair_df)}'
                      )

            # # # test with csv to numpy
            # yes script now wirks directly with df, don't need to load csv.
            # now move on to doing full thing

            sep=sep_list[stair_idx]
            stair_levels = [stair]

            # # for all in one function
            # # # # #

            fit_curve_plot, psignifit_dict = results_to_psignifit(csv_path=stair_df, save_path=root_path,
                                                                  isi=isi, sep=sep, p_run_name=p_name,
                                                                  sep_col='stair', stair_levels=stair_levels,
                                                                  thr_col='probeLum', resp_col='trial_response',
                                                                  quartile_bins=q_bins, n_bins=n_bins,
                                                                  save_np=False, target_threshold=.75,
                                                                  sig_name='norm', est_type='MAP',
                                                                  save_plot=False, verbose=verbose
                                                                  )

            # append result to zeros_df
            threshold = psignifit_dict['Threshold']
            thr_array[stair_idx, isi_idx] = threshold

    # save zeros df - run and q_bin in name.
    print(f'thr_array:\n{thr_array}')

    # make dataframe from array
    thr_df = pd.DataFrame(thr_array, columns=isi_name_list)
    thr_df.insert(0, 'stair', stair_list)
    if verbose:
        print(f"thr_df:\n{thr_df}")

    # save response and threshold arrays
    if q_bins:
        bin_type = 'qbin'
    else:
        bin_type = 'cbin'
    thr_filename = f'{bin_type}_{run_dir}.csv'
    thr_filepath = os.path.join(root_path, thr_filename)
    thr_df.to_csv(thr_filepath, index=False)

print('\n\nfinished')
