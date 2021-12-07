import os

import matplotlib.pyplot as plt
import pandas as pd

from psignifit_tools import run_psignifit, results_to_psignifit

root_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/Kim'
run_folder_names = ['P6a-Kim', 'P6b-Kim', 'P6c-Kim', 'P6d-Kim', 'P6e-Kim', 'P6f-Kim']
# run_folder_names = ['P6a-Kim', 'P6b-Kim', 'P6c-Kim', 'P6d-Kim', 'P6e-Kim', 'P6f-Kim']

print('test_all kim data with psignifit.')
# # Loop through to build dfs of mean diff (e.g., stair1, stair2..)
# run_folder_names = ['P6a-Kim', 'P6b-Kim', 'P6c-Kim', 'P6d-Kim', 'P6e-Kim', 'P6f-Kim']
# root_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/Kim/Nick_practice'
# run_folder_names = ['P6a-Kim']

# participant_name = 'Kim'
# isi_list = [-1, 0, 2, 4, 6, 9, 12, 24]
# isi_name_list = [f'isi{i}' for i in isi_list]
# stair_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
# sep_list = [18, 18, 6, 6, 3, 3, 2, 2, 1, 1, 0, 0, 99, 99]
# verbose = False
#
# q_bins = True
# n_bins = 10
#
# # loop through runs
# for run_idx, run_dir in enumerate(run_folder_names):
#
#     print(f'run {run_idx}: {run_dir}')
#
#     thr_array = np.zeros(shape=[len(stair_list), len(isi_list)])
#
#
#     # loop through ISI
#     # loop through isi values
#     for isi_idx, isi in enumerate(isi_list):
#         if verbose:
#             print(f"\n{isi_idx}: isi: {isi}")
#
#         isi_name = isi_name_list[isi_idx]
#         p_name = participant_name
#
#         p_name = f'{participant_name}{run_idx+1}'
#         print(f'p_name: {p_name}')
#
#         # get df for this isi only
#         isi_df = pd.read_csv(f'{root_path}{os.sep}{run_dir}'
#                              f'{os.sep}ISI_{isi}_probeDur2/{p_name}.csv')
#
#         print(f'\nrunning analysis for {p_name}, {run_dir}\n')
#
#         # loop through stairs for this isi
#         for stair_idx, stair in enumerate(stair_list):
#
#             # get df just for one stair at this isi
#             stair_df = isi_df[isi_df['stair'] == stair]
#             if verbose:
#                 print(f'\nstair_df (stair={stair}, isi={isi}:\n'
#                       f'{type(stair_df)}'
#                       )
#
#             # # # test with csv to numpy
#             # yes script now works directly with df, don't need to load csv.
#             # now move on to doing full thing
#
#             sep = sep_list[stair_idx]
#             stair_levels = [stair]
#
#             # # for all in one function
#             # # # # #
#
#             fit_curve_plot, psignifit_dict = results_to_psignifit(csv_path=stair_df, save_path=root_path,
#                                                                   isi=isi, sep=sep, p_run_name=p_name,
#                                                                   sep_col='stair', stair_levels=stair_levels,
#                                                                   thr_col='probeLum', resp_col='trial_response',
#                                                                   quartile_bins=q_bins, n_bins=n_bins,
#                                                                   save_np=False, target_threshold=.75,
#                                                                   sig_name='norm', est_type='MAP',
#                                                                   save_plot=False, verbose=verbose
#                                                                   )
#
#             # append result to zeros_df
#             threshold = psignifit_dict['Threshold']
#             thr_array[stair_idx, isi_idx] = threshold
#
#     # save zeros df - run and q_bin in name.
#     print(f'thr_array:\n{thr_array}')
#
#     # make dataframe from array
#     thr_df = pd.DataFrame(thr_array, columns=isi_name_list)
#     thr_df.insert(0, 'stair', stair_list)
#     if verbose:
#         print(f"thr_df:\n{thr_df}")
#
#     # save response and threshold arrays
#     if q_bins:
#         bin_type = 'qbin'
#     else:
#         bin_type = 'cbin'
#     thr_filename = f'{bin_type}_{run_dir}.csv'
#     thr_filepath = os.path.join(root_path, thr_filename)
#     thr_df.to_csv(thr_filepath, index=False)

print('make batman plots')
# row_indices = [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 6]
# sym_sep_list = [-18, -6, -3, -2, -1, 0, 1, 2, 3, 6, 18, 20]
# fig2_x_tick_lab = [-18, -6, -3, -2, -1, 0, 1, 2, 3, 6, 18, '1\nprobe']
# isi_list = [-1, 0, 2, 4, 6, 9, 12, 24]
# isi_name_list = ['Concurrent' if i == -1 else f'isi{i}' for i in isi_list]
# bin_type_list = ['cbin', 'qbin']
#
# cbin_mean_diffs = []
# qbin_mean_diffs = []
#
# # # Loop through to make batman plots from diff dfs.
# for bin_type in bin_type_list:
#     if bin_type is 'cbin':
#         diff_list = cbin_mean_diffs
#     else:
#         diff_list = qbin_mean_diffs
#
#     for run_name in run_folder_names:
#         print(f'\n{bin_type}, {run_name}')
#         this_df_path = f'{root_path}{os.sep}{bin_type}_{run_name}.csv'
#         this_df = pd.read_csv(this_df_path)
#         # print(f'this_df:\n{this_df}')
#
#         # batman plots need three dfs, pos, neg and mean.
#         pos_df, neg_df = split_df_alternate_rows(this_df)
#         print(f'pos_df:\n{pos_df}')
#         # print(f'neg_df:\n{neg_df}')
#
#         mean_thr_df = pd.concat([pos_df, neg_df]).groupby(level=0).mean()
#         # print(f'mean_thr_df:\n{mean_thr_df}')
#
#         # expand dfs to have pos and neg sep values
#         mean_thr_df.columns = ['stair'] + isi_name_list
#         pos_df.columns = ['stair'] + isi_name_list
#         neg_df.columns = ['stair'] + isi_name_list
#         pos_sym_df = pos_df.iloc[row_indices]
#         pos_sym_df.reset_index(drop=True, inplace=True)
#         pos_sym_df.insert(loc=0, column='Separation', value=sym_sep_list)
#         print(f'pos_sym_df:\n{pos_sym_df}')
#         neg_sym_df = neg_df.iloc[row_indices]
#         neg_sym_df.reset_index(drop=True, inplace=True)
#         neg_sym_df.insert(loc=0, column='Separation', value=sym_sep_list)
#         mean_sym_df = mean_thr_df.iloc[row_indices]
#         mean_sym_df.reset_index(drop=True, inplace=True)
#         mean_sym_df.insert(loc=0, column='Separation', value=sym_sep_list)
#
#         # # diff values are calculated here, not in batman plots.
#         # get mean difference between pairs of sep values for evaluating analysis,
#         # method with lowest mean difference is least noisy method. (for fig2)
#         # for each pair of sep values (e.g., stair1&2, stair3&4) subtract one from other.
#         # get abs of all values them sum the columns (ISIs)
#         diff_next = np.sum(abs(pos_sym_df - neg_sym_df), axis=0)
#         # take the mean of these across all ISIs to get single value
#         mean_diff_next = float(np.mean(diff_next))
#
#         diff_list.append(mean_diff_next)
#
#         fig_title = f'runs_psignifit thresholds per isi. ' \
#                     f'(mean diff: {round(mean_diff_next, 2)})'
#         fig2_savename = f'runs_psignifit_thresholds.png'
#         save_path = f'{root_path}{os.sep}{run_name}'
#         eight_batman_plots(mean_df=mean_sym_df, thr1_df=pos_sym_df, thr2_df=neg_sym_df,
#                            fig_title=fig_title, isi_name_list=isi_name_list,
#                            x_tick_vals=sym_sep_list, x_tick_labels=fig2_x_tick_lab,
#                            sym_sep_diff_list=diff_next,
#                            save_path=save_path, save_name=fig2_savename,
#                            verbose=True
#                            )
#         plt.show()
#         plt.close()
#
# print(f'\nFinnished getting differences\n'
#       f'cbin_mean_diffs: {cbin_mean_diffs}\n'
#       f'qbin_mean_diffs: {qbin_mean_diffs}')

print('\n\ncompare 10 bins, and no bins')
save_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/Kim/compare_unbinned'
root_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/Kim'
run_folder_names = ['P6a-Kim', 'P6b-Kim', 'P6c-Kim', 'P6d-Kim', 'P6e-Kim', 'P6f-Kim']
participant_name = 'Kim'
isi_list = [-1, 0, 2, 4, 6, 9, 12, 24]
isi_name_list = [f'isi{i}' for i in isi_list]
stair_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
sep_list = [18, 18, 6, 6, 3, 3, 2, 2, 1, 1, 0, 0, 99, 99]
target_threshold = .75
verbose = True
q_bins = True
n_bins = 10

results_array = []

# loop through runs
for run_idx, run_dir in enumerate(run_folder_names):

    print(f'\nrun {run_idx}: {run_dir}')

    # thr_array = np.zeros(shape=[len(stair_list), len(isi_list)])

    # loop through isi values
    for isi_idx, isi in enumerate(isi_list):
        if verbose:
            print(f"\n{isi_idx}: isi: {isi}")

        isi_name = isi_name_list[isi_idx]
        p_name = participant_name

        p_name = f'{participant_name}{run_idx+1}'
        print(f'p_name: {p_name}')

        # get df for this isi only
        csv_path = f'{root_path}{os.sep}{run_dir}{os.sep}ISI_{isi}_probeDur2/{p_name}.csv'
        isi_df = pd.read_csv(csv_path)

        print(f'\nrunning analysis for {p_name}, {run_dir}\n')

        # loop through stairs for this isi
        for stair_idx, stair in enumerate(stair_list):

            # get df just for one stair at this isi
            stair_df = isi_df[isi_df['stair'] == stair]
            sep = sep_list[stair_idx]
            stair_levels = [stair]
            if verbose:
                print(f'\nstair_df (stair={stair}, isi={isi}:\n'
                      f'{stair_df.head()}')
            # raw data - unbinned
            # drop columns other than needed
            unbinned_df = stair_df[['probeLum', 'trial_response']]
            unbinned_df['n_total'] = 1
            n_unique_values = len(unbinned_df['probeLum'].unique())
            print(f'n_unique_values: {n_unique_values}')
            unbinned_np = unbinned_df.to_numpy()
            print(f'unbinned_np:\n{unbinned_np}')

            dest_name = f'{participant_name}_{run_dir}_{isi_name}_sep{sep}_stair{stair}_UB'
            bin_data_dict = {'dset_name': dest_name,
                             'stair_levels': stair_levels,
                             'csv_path': csv_path}

            unbinned_plot, unbinned_dict = run_psignifit(data_np=unbinned_np,
                                                         bin_data_dict=bin_data_dict,
                                                         save_path=save_path,
                                                         target_threshold=target_threshold,
                                                         # sig_name=sig_name, est_type=est_type,
                                                         n_blocks=n_unique_values,
                                                         save_plot=True, show_plot=False,
                                                         verbose=True)
            plt.close()
            unbinned_thr = unbinned_dict['Threshold']
            print(f'\nunbinned_thr: {unbinned_thr}')

            # # for all in one function
            binned_plot, binned_dict = results_to_psignifit(csv_path=stair_df, save_path=save_path,
                                                            isi=isi, sep=sep, p_run_name=p_name,
                                                            sep_col='stair', stair_levels=stair_levels,
                                                            thr_col='probeLum', resp_col='trial_response',
                                                            quartile_bins=q_bins, n_bins=n_bins,
                                                            save_np=False, target_threshold=.75,
                                                            sig_name='norm', est_type='MAP',
                                                            save_plot=True, verbose=verbose
                                                            )
            plt.close()

            binned_thr = binned_dict['Threshold']
            print(f'\nbinned_thr: {binned_thr}')

            results_array.append([dest_name, n_unique_values, unbinned_thr, binned_thr])


results_df = pd.DataFrame(results_array, columns=['dset_name', 'n_unique', 'unbinned_thr', 'binned_thr'])
print(f'results_df:\n{results_df}')
results_df.to_csv(f'{save_path}{os.sep}compare_unbinned.csv', index=False)

# for each run, ISI, sep

# feed in raw df to psignifit to get thr with unbinned data
# feed in binned df

# compare results.


print('\n\nfinished')
