import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# from exp1a_psignifit_analysis import a_data_extraction, b3_plot_staircase, c_plots, \
#     d_average_participant, e_average_exp_data, make_average_plots
from rad_flow_psignifit_analysis import b3_plot_stair_sep0, c_plots, d_average_participant
from rad_flow_psignifit_analysis import make_average_plots, e_average_exp_data, \
    plot_runs_ave_w_errors, run_thr_plot, simple_log_log_plot, make_long_df, \
    log_log_w_markers_plot, run_thr_plot_w_markers, plot_ave_w_errors_markers
from psignifit_tools import get_psignifit_threshold_df
from python_tools import which_path, running_on_laptop, switch_path

# # loop through run folders with first 4 scripts (a, get_psignifit_threshold_df, b3, c)
# # then run script d to get master lists and averages
exp_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\Exp2_Bloch_NM_v5"
# exp_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\Exp2_Bloch_NM_v5\test_stuff"
# exp_path = r"C:\Users\sapnm4\PycharmProjects\Cardiff\Nick_scripts\Exp2_Bloch_NM_DEMO"


convert_path1 = os.path.normpath(exp_path)
if running_on_laptop():
    convert_path1 = switch_path(convert_path1, 'mac_oneDrive')
exp_path = convert_path1
print(f"exp_path: {exp_path}")

participant_list = ['Kim', 'Nick', 'Tony', 'Simon', 'Kris']  # 'Nick', 'bb', 'cc', 'dd', 'ee']
# participant_list = ['Nick']  # 'Nick', 'bb', 'cc', 'dd', 'ee']

bloch_version = 5
n_runs = 12

analyse_from_run = 1
trim_list = []

for p_idx, participant_name in enumerate(participant_list):
    root_path = os.path.join(exp_path, participant_name)

    # search to automatically get run_folder_names
    dir_list = os.listdir(root_path)
    run_folder_names = []
    for i in range(n_runs):  # numbers 0 to 11
        check_dir = f'{participant_name}_{i+analyse_from_run}'   # numbers 1 to 12
        if check_dir in dir_list:
            run_folder_names.append(check_dir)
    print(f'run_folder_names: {run_folder_names}')

    trim_n = None
    if len(run_folder_names) == 12:
        trim_n = 2
    elif len(run_folder_names) > 12:
        # trim_n = 2
        if len(run_folder_names) % 2 == 0:  # if even
            trim_n = int((len(run_folder_names) - 12) / 2)
        else:
            raise ValueError(f"for this exp you have {len(run_folder_names)} runs, set rules for trimming.")
    trim_list.append(trim_n)

    for run_idx, run_dir in enumerate(run_folder_names):

        save_path = f'{root_path}{os.sep}{run_dir}'

        # don't delete this (participant_name = participant_name),
        # needed to ensure names go name1, name2, name3 not name1, name12, name123
        p_name = participant_name

        # '''a'''
        p_name = f'{participant_name}_{run_idx+analyse_from_run}'
        print(f'\nrunning analysis for {participant_name}, {run_dir}, {p_name}\n')


    #     # for first run, some files are saved just as name not name1
    #     run_data_path = f'{save_path}{os.sep}{p_name}_output.csv'
    #     if not os.path.isfile(run_data_path):
    #         raise FileNotFoundError(run_data_path)
    #     print(f'run_data_path: {run_data_path}')
    #
    #     run_data_df = pd.read_csv(run_data_path)
    #     # remove any Unnamed columns
    #     if any("Unnamed" in i for i in list(run_data_df.columns)):
    #         unnamed_col = [i for i in list(run_data_df.columns) if "Unnamed" in i][0]
    #         run_data_df.drop(unnamed_col, axis=1, inplace=True)
    #     run_data_df.sort_values(by=['stair', 'step'], inplace=True, ignore_index=True)
    #
    #     '''add newLum column
    #             in old version, the experiment script varies probeLum and converts to float(RGB255) values for screen.
    #             However, monitor can only use int(RGB255).
    #             This function will will round RGB255 values to int(RGB255), then convert to NEW_probeLum
    #             LumColor255Factor = 2.395387069
    #             1. get probeColor255 column.
    #             2. convert to int(RGB255) and convert to new_Lum with int(RGB255)/LumColor255Factor
    #             3. add to run_data_df'''
    #     if 'newLum' not in run_data_df.columns.to_list():
    #         LumColor255Factor = 2.395387069
    #         rgb255_col = run_data_df['probeColor255'].to_list()
    #         newLum = [int(i) / LumColor255Factor for i in rgb255_col]
    #         run_data_df.insert(9, 'newLum', newLum)
    #         print(f"added newLum column\n"
    #               f"run_data_df: {run_data_df.columns.to_list()}")
    #
    #     # save sorted csv
    #     run_data_df.to_csv(run_data_path, index=False)
    #     print(f"run_data_df: {run_data_df.columns}\n{run_data_df}")
    #
    #     # extract values from dataframe
    #
    #
    #     # use this isi list for psignifit
    #     dur_ms_list = run_data_df['dur_ms'].unique()
    #     print(f'dur_ms_list: {dur_ms_list}')
    #     cond_types = run_data_df['cond_type'].unique().tolist()
    #     print(f'cond_types: {cond_types}')
    #
    #
    #     '''get psignifit thresholds df - use stairs as sep levels rather than using groups'''
    #     thr_df = get_psignifit_threshold_df(root_path=root_path,
    #                                         p_run_name=run_dir,
    #                                         csv_name=run_data_df,
    #                                         n_bins=9, q_bins=True,
    #                                         isi_col='dur_ms', isi_list=dur_ms_list,
    #                                         sep_col='cond_type',
    #                                         thr_col='newLum',
    #                                         sep_list=cond_types,
    #                                         conf_int=True,
    #                                         thr_type='Bayes',
    #                                         plot_both_curves=False,
    #                                         cols_to_add_dict=None,
    #                                         verbose=True)
    #     print(f'thr_df: {type(thr_df)}\n{thr_df}')
    #
    #
    #     '''convert wide thr_df to long_thr_df with additional columns'''
    #     run_data_path = f'{save_path}{os.sep}{p_name}_output.csv'
    #     run_data_df = pd.read_csv(run_data_path)
    #     print(f'run_data_df:\n{run_data_df}')
    #     print(f'run_data_df columns: {run_data_df.columns.to_list()}')
    #
    #     '''get values from run data to add to long_df'''
    #     # infer isi_fr fron dur fr
    #     isi_fr_list = [i-4 for i in run_data_df['dur_fr'].unique().tolist()]
    #     isi_fr_name_list = ['conc' if i == -2 else f'ISI_{i}' for i in isi_fr_list]
    #     print(f'isi_fr_name_list: {isi_fr_name_list}')
    #
    #     stair_names_list = run_data_df['stair_name'].unique().tolist()
    #     print(f'stair_names_list: {stair_names_list}')
    #
    #     # get dur_ms vals to use when converting to long df.
    #     dur_ms_list = run_data_df['dur_ms'].unique().tolist()
    #     dur_ms_name_list = [f'dur_ms_{i}' for i in dur_ms_list]
    #     print(f'dur_ms_name_list: {dur_ms_name_list}')
    #
    #     # load thr_df to make long
    #     thr_df_path = f'{save_path}{os.sep}psignifit_thresholds.csv'
    #     thr_df = pd.read_csv(thr_df_path)
    #     print(f'thr_df:\n{thr_df}')
    #
    #     long_thr_df = make_long_df(wide_df=thr_df,
    #                                cols_to_keep=['cond_type'],
    #                                cols_to_change=dur_ms_name_list,
    #                                cols_to_change_show='newLum',
    #                                new_col_name='dur_ms', strip_from_cols='dur_ms_', verbose=True)
    #
    #     # add in additional columns
    #     dur_ms_list = [round(i, 2) for i in dur_ms_list]
    #     long_thr_df['dur_ms'] = dur_ms_list
    #
    #     long_thr_df.insert(1, 'stair_name', stair_names_list)
    #     long_thr_df.insert(2, 'isi_fr', isi_fr_name_list)
    #
    #     # # make extra columns for -1 slope=complete plots - delta_I
    #     thr_col = long_thr_df['newLum'].to_list()
    #     bgLum = 21.2
    #     delta_I_col = [i-bgLum for i in thr_col]
    #     long_thr_df.insert(5, 'delta_I', delta_I_col)
    #
    #     print(f'long_thr_df:\n{long_thr_df}')
    #     long_thr_df_path = f'{save_path}{os.sep}long_thr_df.csv'
    #     long_thr_df.to_csv(long_thr_df_path, index=False)
    #
    #
    #
    #     '''run plots from here'''
    #     long_thr_df_path = f'{save_path}{os.sep}long_thr_df.csv'
    #     long_thr_df = pd.read_csv(long_thr_df_path)
    #     print(f'\nlong_thr_df:\n{long_thr_df}')
    #
    #     isi_name_list = list(long_thr_df['isi_fr'].unique())
    #     print(f'isi_name_list: {isi_name_list}')
    #     dur_ms_list = list(long_thr_df['dur_ms'].unique())
    #     print(f'dur_ms_list: {dur_ms_list}')
    #
    #
    #     # fig 1. basic plot with exp1 axes - dur_ms vs thr
    #     run_thr_plot_w_markers(long_thr_df, x_col='dur_ms', y_col='newLum', hue_col='cond_type',
    #                            x_ticks_vals=[int(i) for i in dur_ms_list], x_tick_names=[int(i) for i in dur_ms_list],
    #                            x_axis_label='duration (ms)',
    #                            y_axis_label='Probe Luminance',
    #                            legend_names=isi_name_list,
    #                            fig_title=f'P{p_idx+1} run{run_idx+analyse_from_run}. Bloch_v{bloch_version}: duration vs thresholds',
    #                            save_as=f'{save_path}{os.sep}bloch_v{bloch_version}_dur_v_thr.png')
    #     plt.show()
    #
    #     # fig 2. plot with log-log axes - dur_ms v delta_I
    #     log_log_w_markers_plot(long_thr_df, x_col='dur_ms', y_col='delta_I', hue_col='cond_type',
    #                            legend_names=isi_name_list,
    #                            x_axis_label='log(duration, ms)',
    #                            y_axis_label='Contrast: log(∆I)',
    #                            fig_title=f'P{p_idx+1} run{run_idx+analyse_from_run}. Bloch_v{bloch_version}: log(duration) v log(∆I)',
    #                            save_as=f'{save_path}{os.sep}bloch_v{bloch_version}_log_dur_log_contrast.png')
    #     plt.show()
    #
    #
    #
    # '''d'''
    # thr_df_name = 'long_thr_df'
    #
    # print(f"\ntrim_n: {trim_n}")
    # d_average_participant(root_path=root_path, run_dir_names_list=run_folder_names,
    #                       thr_df_name=thr_df_name, trim_n=trim_n, error_type='SE')
    #
    # # making average plot
    # all_df_path = os.path.join(root_path, f'MASTER_TM{trim_n}_thresholds.csv')
    # p_ave_path = os.path.join(root_path, f'MASTER_ave_TM{trim_n}_thresh.csv')
    # err_path = os.path.join(root_path, f'MASTER_ave_TM{trim_n}_thr_error_SE.csv')
    # if trim_n is None:
    #     all_df_path = os.path.join(root_path, f'MASTER_{thr_df_name}.csv')
    #     p_ave_path = os.path.join(root_path, 'MASTER_ave_thresh.csv')
    #     err_path = os.path.join(root_path, 'MASTER_ave_thr_error_SE.csv')
    #
    #
    #
    # # load data and change order to put 1pr last
    # print('*** making average plot ***')
    # # reshape dfs so that the different conds are in separate columns.
    # ave_df = pd.read_csv(p_ave_path)
    # print(f'ave_df:\n{ave_df}')
    #
    # # # fig 1, standard axes (not log)
    # wide_df = ave_df.pivot(index=['dur_ms'], columns='cond_type', values='newLum')
    # print(f'wide_df:\n{wide_df}')
    #
    # dur_list = ave_df['dur_ms'].to_list()
    # print(f'dur_list: {dur_list}')
    # dur_labels = [format(i, '.3g') for i in dur_list]
    # print(f'dur_labels: {dur_labels}')
    #
    # isi_name_list = list(ave_df['isi_fr'].unique())
    # print(f'isi_name_list: {isi_name_list}')
    #
    # error_df = pd.read_csv(err_path)
    # wide_err_df = error_df.pivot(index=['dur_ms'], columns='cond_type', values='newLum')
    # print(f'wide_err_df:\n{wide_err_df}')
    # fig_title = f'P{p_idx} average thresholds - Bloch_v{bloch_version} (n={len(run_folder_names)})'
    # save_name = f'bloch_v{bloch_version}_dur_v_thr.png'
    #
    # plot_ave_w_errors_markers(fig_df=wide_df, error_df=wide_err_df,
    #                           jitter=False, error_caps=True, alt_colours=False,
    #                           legend_names=isi_name_list,
    #                           even_spaced_x=False,
    #                           fixed_y_range=False,
    #                           x_tick_vals=dur_list,
    #                           x_tick_labels=dur_labels,
    #                           x_axis_label='Duration (ms)',
    #                           y_axis_label='Threshold',
    #                           log_log_axes=False,
    #                           neg1_slope=False,
    #                           fig_title=fig_title, save_name=save_name,
    #                           save_path=root_path, verbose=True)
    # plt.show()
    #
    # # # fig 2 - log dur, log(∆I)
    # wide_df = ave_df.pivot(index=['dur_ms'], columns='cond_type', values='delta_I')
    # print(f'wide_df:\n{wide_df}')
    #
    # error_df = pd.read_csv(err_path)
    # wide_err_df = error_df.pivot(index=['dur_ms'], columns='cond_type', values='delta_I')
    # print(f'wide_err_df:\n{wide_err_df}')
    #
    # fig_title = f'P{p_idx} average log(∆I) thresholds - Bloch_v{bloch_version} (n={len(run_folder_names)})'
    # save_name = f'bloch_v{bloch_version}_log_dur_log_contrast.png'
    # plot_ave_w_errors_markers(fig_df=wide_df, error_df=wide_err_df,
    #                           jitter=False, error_caps=True, alt_colours=False,
    #                           legend_names=isi_name_list,
    #                           even_spaced_x=False,
    #                           fixed_y_range=False,
    #                           x_tick_vals=None,
    #                           x_tick_labels=None,
    #                           x_axis_label='log(duration ms)',
    #                           y_axis_label='Contrast: log(∆I)',
    #                           log_log_axes=True,
    #                           neg1_slope=True,
    #                           slope_ycol_name='1probe',
    #                           slope_xcol_idx_depth=1,
    #                           fig_title=fig_title, save_name=save_name,
    #                           save_path=root_path, verbose=True)
    # plt.show()


print(f'exp_path: {exp_path}')
print('\nget exp_average_data')
# trim_n = 2
# participant_list = ['Kim', 'Nick', 'Tony', 'Simon', 'Kris']

# e_average_exp_data(exp_path=exp_path, p_names_list=participant_list, exp_type=f'Bloch_v{bloch_version}',
#                    error_type='SE', n_trimmed=trim_n, verbose=True)


all_df_path = os.path.join(exp_path, 'MASTER_exp_thr.csv')
exp_ave_path = os.path.join(exp_path, 'MASTER_exp_ave_thr.csv')
err_path = os.path.join(exp_path, 'MASTER_ave_thr_error_SE.csv')

# print('*** making average plot ***')
# print('*** making average plot ***')
fig_df = pd.read_csv(exp_ave_path)
print(f'fig_df: {fig_df.columns.to_list()}\n{fig_df}')

# isi_vals_list = fig_df['ISI'].to_list()
# isi_names_list = ['1pr' if i == -2 else int(i) for i in isi_vals_list]
# print(f'isi_names_list: {isi_names_list}')

# dur_list = fig_df['dur_ms'].to_list()
# print(f'dur_list: {dur_list}')
# dur_labels = [format(i, '.3g') for i in dur_list]
# print(f'dur_labels: {dur_labels}')
#
# isi_name_list = list(fig_df['isi_fr'].unique())
# print(f'isi_name_list: {isi_name_list}')

error_df = pd.read_csv(err_path)
print(f'error_df:\n{error_df}')

ave_thr_by_dur_df = fig_df[['dur_ms', 'newLum']]
ave_thr_by_dur_df.set_index('dur_ms', inplace=True)
err_thr_by_dur_df = error_df[['dur_ms', 'newLum']]
err_thr_by_dur_df.set_index('dur_ms', inplace=True)
print(f'ave_thr_by_dur_df:\n{ave_thr_by_dur_df}')



dur_list = fig_df['dur_ms'].to_list()
print(f'dur_list: {dur_list}')
dur_labels = [format(i, '.3g') for i in dur_list]
print(f'dur_labels: {dur_labels}')

isi_name_list = list(fig_df['isi_fr'].unique())
print(f'isi_name_list: {isi_name_list}')

# error_df = pd.read_csv(err_path)
# wide_err_df = error_df.pivot(index=['dur_ms'], columns='cond_type', values='newLum')
# print(f'wide_err_df:\n{wide_err_df}')
fig_title = f'Experiment average thresholds - Bloch_v{bloch_version} (n={len(run_folder_names)})'
save_name = f'bloch_v{bloch_version}_dur_v_thr.png'

plot_ave_w_errors_markers(fig_df=ave_thr_by_dur_df, error_df=err_thr_by_dur_df,
                          jitter=False, error_caps=True, alt_colours=False,
                          legend_names=isi_name_list,
                          even_spaced_x=False,
                          fixed_y_range=False,
                          x_tick_vals=dur_list,
                          x_tick_labels=dur_labels,
                          x_axis_label='Duration (ms)',
                          y_axis_label='Threshold',
                          log_log_axes=False,
                          neg1_slope=False,
                          fig_title=fig_title, save_name=save_name,
                          save_path=root_path, verbose=True)
plt.show()

# # fig 2 - log dur, log(∆I)
# wide_df = ave_df.pivot(index=['dur_ms'], columns='cond_type', values='delta_I')
# print(f'wide_df:\n{wide_df}')
#
# error_df = pd.read_csv(err_path)
# wide_err_df = error_df.pivot(index=['dur_ms'], columns='cond_type', values='delta_I')
# print(f'wide_err_df:\n{wide_err_df}')

# # plot 2: log-log axes - log(∆i)
log_dur_log_delta_I_df = fig_df[['dur_ms', 'delta_I']]
log_dur_log_delta_I_df.set_index('dur_ms', inplace=True)
err_log_dur_log_delta_I_df = error_df[['dur_ms', 'delta_I']]
err_log_dur_log_delta_I_df.set_index('dur_ms', inplace=True)
print(f'log_dur_log_delta_I_df:\n{log_dur_log_delta_I_df}')
fig_title = f'Experiment average log(∆I) thresholds - Bloch_v5 (n={len(participant_list)})'
save_name = 'bloch_v5_log_dur_log_contrast.png'

fig_title = f'Experiment average log(∆I) thresholds - Bloch_v{bloch_version} (n={len(participant_list)})'
save_name = f'bloch_v{bloch_version}_log_dur_log_contrast.png'
plot_ave_w_errors_markers(fig_df=log_dur_log_delta_I_df, error_df=err_log_dur_log_delta_I_df,
                          jitter=False, error_caps=True, alt_colours=False,
                          legend_names=isi_name_list,
                          even_spaced_x=False,
                          fixed_y_range=False,
                          x_tick_vals=None,
                          x_tick_labels=None,
                          x_axis_label='log(duration ms)',
                          y_axis_label='Contrast: log(∆I)',
                          log_log_axes=True,
                          neg1_slope=True,
                          slope_ycol_name='delta_I',
                          slope_xcol_idx_depth=1,
                          fig_title=fig_title, save_name=save_name,
                          save_path=root_path, verbose=True)
plt.show()

#
# # fig 1 - ave thr by sep
# ave_thr_by_dur_df = fig_df[['dur_ms', 'newLum']]
# ave_thr_by_dur_df.set_index('dur_ms', inplace=True)
# err_thr_by_dur_df = error_df[['dur_ms', 'newLum']]
# err_thr_by_dur_df.set_index('dur_ms', inplace=True)
# print(f'ave_thr_by_dur_df:\n{ave_thr_by_dur_df}')
#
# fig_title = f'Experiment average thresholds - Bloch_v5  (n={len(participant_list)})'
# save_name = 'bloch_v5_dur_v_thr.png'
# plot_runs_ave_w_errors(fig_df=ave_thr_by_dur_df, error_df=err_thr_by_dur_df,
#                        jitter=False, error_caps=True, alt_colours=False,
#                        legend_names=None,
#                        even_spaced_x=False,
#                        fixed_y_range=False,
#                        x_tick_vals=dur_list,
#                        x_tick_labels=dur_labels,
#                        x_axis_label='duration (ms)',
#                        y_axis_label='Luminance threshold',
#                        log_log_axes=False,
#                        neg1_slope=False,
#                        fig_title=fig_title, save_name=save_name,
#                        save_path=exp_path, verbose=True)
# plt.show()
#
# # # plot 2: log-log axes - log(∆i)
# log_dur_log_delta_I_df = fig_df[['dur_ms', 'delta_I']]
# log_dur_log_delta_I_df.set_index('dur_ms', inplace=True)
# err_log_dur_log_delta_I_df = error_df[['dur_ms', 'delta_I']]
# err_log_dur_log_delta_I_df.set_index('dur_ms', inplace=True)
# print(f'log_dur_log_delta_I_df:\n{log_dur_log_delta_I_df}')
# fig_title = f'Experiment average log(∆I) thresholds - Bloch_v5 (n={len(participant_list)})'
# save_name = 'bloch_v5_log_dur_log_contrast.png'
# plot_runs_ave_w_errors(fig_df=log_dur_log_delta_I_df, error_df=err_log_dur_log_delta_I_df,
#                        jitter=False, error_caps=True, alt_colours=False,
#                        legend_names=None,
#                        even_spaced_x=False,
#                        fixed_y_range=False,
#                        x_tick_vals=None,
#                        x_tick_labels=None,
#                        x_axis_label='log(duration ms)',
#                        y_axis_label='Contrast: log(∆I)',
#                        log_log_axes=True,
#                        neg1_slope=True,
#                        slope_ycol_name='delta_I',
#                        slope_xcol_idx_depth=1,
#                        fig_title=fig_title, save_name=save_name,
#                        save_path=exp_path, verbose=True)
# plt.show()

# # # plot 3: log-log axes - log(∆I/I)
#
# log_dur_log_weber_df = fig_df[['dur_ms', 'weber_thr']]
# log_dur_log_weber_df.set_index('dur_ms', inplace=True)
# err_log_dur_log_weber_df = error_df[['dur_ms', 'weber_thr']]
# err_log_dur_log_weber_df.set_index('dur_ms', inplace=True)
# print(f'log_area_log_weber_df:\n{log_dur_log_weber_df}')
#
# fig_title = 'Experiment average log(∆I/I) thresholds - Bloch_v5'
# save_name = 'bloch_v5_log_dur_log_weber.png'
# plot_runs_ave_w_errors(fig_df=log_dur_log_weber_df, error_df=err_log_dur_log_weber_df,
#                        jitter=False, error_caps=True, alt_colours=False,
#                        legend_names=None,
#                        even_spaced_x=False,
#                        fixed_y_range=False,
#                        x_tick_vals=None,
#                        x_tick_labels=None,
#                        x_axis_label='log(duration ms) - 1probe condition',
#                        y_axis_label='Weber threshold: log(∆I/I)',
#                        log_log_axes=True,
#                        neg1_slope=True,
#                        slope_ycol_name='weber_thr',
#                        slope_xcol_idx_depth=1,
#                        fig_title=fig_title, save_name=save_name,
#                        save_path=exp_path, verbose=True)
# plt.show()


print('*** finished exp average plot ***')
#
# # # todo: wrap these plot functions for participant and experiment averages into a function


print('\nExp2_Bloch_analysis_pipe finished\n')
