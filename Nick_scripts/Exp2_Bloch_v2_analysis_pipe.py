import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# from exp1a_psignifit_analysis import a_data_extraction, b3_plot_staircase, c_plots, \
#     d_average_participant, e_average_exp_data, make_average_plots
from rad_flow_psignifit_analysis import b3_plot_stair_sep0, c_plots, d_average_participant
from rad_flow_psignifit_analysis import make_average_plots, e_average_exp_data, \
    plot_runs_ave_w_errors, run_thr_plot, simple_log_log_plot, make_long_df
from psignifit_tools import get_psignifit_threshold_df

# # loop through run folders with first 4 scripts (a, get_psignifit_threshold_df, b3, c)
# # then run script d to get master lists and averages
exp_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/Exp2_Bloch_NM_v2'
participant_list = ['Nick']  # , 'bb', 'cc', 'dd', 'ee']
n_runs = 3

p_idx_plus = 1

for p_idx, participant_name in enumerate(participant_list):
    root_path = f'{exp_path}/{participant_name}'

    run_folder_names = [f'{participant_name}_{i+1}' for i in list(range(n_runs))]
    print(f'run_folder_names: {run_folder_names}')

    for run_idx, run_dir in enumerate(run_folder_names):

        print(f'\nrunning analysis for {participant_name}, {run_dir}, {participant_name}{run_idx+1}\n')
        save_path = f'{root_path}{os.sep}{run_dir}'

        # don't delete this (participant_name = participant_name),
        # needed to ensure names go name1, name2, name3 not name1, name12, name123
        p_name = participant_name

        # '''a'''
        p_name = f'{participant_name}_{run_idx+1}'

        # # for first run, some files are saved just as name not name1
        # run_data_path = f'{save_path}{os.sep}{p_name}_output.csv'
        # if not os.path.isfile(run_data_path):
        #     raise FileNotFoundError(run_data_path)
        # print(f'run_data_path: {run_data_path}')
        #
        # run_data_df = pd.read_csv(run_data_path)
        # # remove any Unnamed columns
        # if any("Unnamed" in i for i in list(run_data_df.columns)):
        #     unnamed_col = [i for i in list(run_data_df.columns) if "Unnamed" in i][0]
        #     run_data_df.drop(unnamed_col, axis=1, inplace=True)
        # run_data_df.sort_values(by=['stair', 'step'], inplace=True, ignore_index=True)
        #
        # # save sorted csv
        # run_data_df.to_csv(run_data_path, index=False)
        # print(f"run_data_df: {run_data_df.columns}\n{run_data_df}")
        #
        # # extract values from dataframe
        # isi_list = run_data_df['ISI'].unique()
        # print(f'isi_list: {isi_list}')
        # cond_types = run_data_df['cond_type'].unique()
        # print(f'cond_types: {cond_types}')
        #
        #
        # '''get psignifit thresholds df - use stairs as sep levels rather than using groups'''
        # thr_df = get_psignifit_threshold_df(root_path=root_path,
        #                                     p_run_name=run_dir,
        #                                     csv_name=run_data_df,
        #                                     n_bins=10, q_bins=True,
        #                                     isi_list=isi_list,
        #                                     sep_col='cond_type',
        #                                     sep_list=cond_types,
        #                                     cols_to_add_dict=None,
        #                                     verbose=True)
        # print(f'thr_df: {type(thr_df)}\n{thr_df}')

        # todo: get delta_thr here so I can calculate averages with it.

        '''# Bloch doesn't currently work with b3_plot_staircase or c_plots
        b3_plot_staircase(run_data_path, show_plots=True)
        c_plots(save_path=save_path, isi_name_list=isi_name_list, show_plots=True)'''

        run_data_path = f'{save_path}{os.sep}{p_name}_output.csv'
        run_data_df = pd.read_csv(run_data_path,
                                  # usecols=
                                  # ['trial_number', 'stair', 'stair_name', 'step',
                                  #  'separation', 'cond_type', 'ISI', 'corner',
                                  #  'probeLum', 'delta_lum', 'trial_response', '3_fps']
                                  )
        print(f'run_data_df:\n{run_data_df}')
        isi_list = list(run_data_df['ISI'].unique())
        isi_name_list = [f'ISI_{i}' for i in isi_list]
        isi_labels_list = ['conc' if i == -2 else i for i in isi_list]
        print(f'isi_list: {isi_list}')
        print(f'isi_name_list: {isi_name_list}')
        print(f'isi_labels_list: {isi_labels_list}')

        thr_df_path = f'{save_path}{os.sep}psignifit_thresholds.csv'
        thr_df = pd.read_csv(thr_df_path)
        print(f'thr_df:\n{thr_df}')

        long_thr_df = make_long_df(wide_df=thr_df,
                                   cols_to_keep=['cond_type'],
                                   cols_to_change=isi_name_list,
                                   cols_to_change_show='probeLum',
                                   new_col_name='ISI', strip_from_cols='ISI_', verbose=True)
        print(f'long_thr_df:\n{long_thr_df}')

        # basic plot with regular axes
        run_thr_plot(long_thr_df, x_col='ISI', y_col='probeLum', hue_col='cond_type',
                     # x_ticks_vals=isi_list,
                     x_tick_names=isi_labels_list,
                     x_axis_label='ISI (2probe cond)',
                     y_axis_label='Probe Luminance',
                     fig_title='Bloch_v2: probe cond vs thr',
                     save_as=f'{save_path}{os.sep}bloch_v2_cond_v_thr.png')
        plt.show()

        # not sure I actually need to do a log-log plot for duration.
        print(f'long_thr_df:\n{long_thr_df}')

        # check for 'area' and 'delta_thr' col
        col_names = long_thr_df.columns.to_list()

        if 'dur_ms' not in col_names:
            # convert separation into area (units are pixels)
            dur_dict = {-2.0: {'frames': 2, 'duration': 8.333333333},
                        0.0: {'frames': 4, 'duration': 16.66666667},
                        8.33: {'frames': 6, 'duration': 25},
                        16.67: {'frames': 8, 'duration': 33.33333333},
                        25.0: {'frames': 10, 'duration': 41.66666667},
                        37.5: {'frames': 13, 'duration': 54.16666667},
                        50.0: {'frames': 16, 'duration': 66.66666667},
                        100.0: {'frames': 28, 'duration': 116.6666667},
                        200.0: {'frames': 52, 'duration': 216.6666667},
                        }
            ISI_col = long_thr_df['ISI'].to_list()
            ISI_col = [float(i) for i in ISI_col]
            print(f'ISI_col: {ISI_col}')
            dur_col = [dur_dict[i]['duration'] for i in ISI_col]
            print(f'dur_col: {dur_col}')
            long_thr_df.insert(1, 'dur_ms', dur_col)

            thr_col = long_thr_df['probeLum'].to_list()
            bgLum = 21.2
            delta_thr_col = [(i-bgLum)/bgLum for i in thr_col]
            long_thr_df.insert(4, 'delta_thr', delta_thr_col)

            if 'stair_name' in col_names:
                long_thr_df.drop('stair_name', axis=1, inplace=True)

            long_thr_df_path = f'{save_path}{os.sep}long_thr_df.csv'
            long_thr_df.to_csv(long_thr_df_path, index=False)
            print(f'long_thr_df:\n{long_thr_df}')


        # plot with log-log axes
        simple_log_log_plot(long_thr_df, x_col='dur_ms', y_col='delta_thr', hue_col='cond_type',
                            x_ticks_vals=None, x_tick_names=None,
                            x_axis_label='log(duration ms) - 1probe condition',
                            y_axis_label='log(∆ threshold)',
                            fig_title='Bloch_v2: log(area) v log(thr)',
                            show_neg1slope=True,
                            save_as=f'{save_path}{os.sep}bloch_v2_log_dur_log_delta.png')
        plt.show()


    '''d'''
    trim_n = None
    if len(run_folder_names) == 12:
        trim_n = 1
    thr_df_name = 'long_thr_df'
    # d_average_participant(root_path=root_path, run_dir_names_list=run_folder_names,
    #                       thr_df_name=thr_df_name, trim_n=trim_n, error_type='SE')


    # making average plot
    all_df_path = f'{root_path}/MASTER_TM1_thresholds.csv'
    p_ave_path = f'{root_path}/MASTER_ave_TM_thresh.csv'
    err_path = f'{root_path}/MASTER_ave_TM_thr_error_SE.csv'
    n_trimmed = trim_n
    if n_trimmed == None:
        all_df_path = f'{root_path}/MASTER_{thr_df_name}.csv'
        p_ave_path = f'{root_path}/MASTER_ave_thresh.csv'
        err_path = f'{root_path}/MASTER_ave_thr_error_SE.csv'

    exp_ave = False

    # load data and change order to put 1pr last
    print('*** making average plot ***')
    # reshape dfs so that the different conds are in separate columns.
    ave_df = pd.read_csv(p_ave_path)
    print(f'ave_df:\n{ave_df}')

    wide_df = ave_df.pivot(index=['ISI'], columns='cond_type', values='probeLum')
    print(f'wide_df:\n{wide_df}')

    x_values = wide_df.index.get_level_values('ISI').to_list()
    x_values = [int(i) if i.is_integer() else i for i in x_values]
    print(f'x_values: {x_values}')
    x_labels = ['conc' if i == -2.0 else i for i in x_values]
    print(f'x_labels: {x_labels}')

    error_df = pd.read_csv(err_path)
    wide_err_df = error_df.pivot(index=['ISI'], columns='cond_type', values='probeLum')
    print(f'wide_err_df:\n{wide_err_df}')

    fig_title = 'Participant average thresholds - Bloch_v2'
    save_name = 'bloch_v2_sep_v_thr.png'
    plot_runs_ave_w_errors(fig_df=wide_df, error_df=wide_err_df,
                           jitter=False, error_caps=True, alt_colours=False,
                           legend_names=None,
                           even_spaced_x=True,
                           fixed_y_range=False,
                           x_tick_vals=x_values,
                           x_tick_labels=x_labels,
                           x_axis_label='ISI (2probe condition)',
                           y_axis_label='Threshold',
                           log_log_axes=False,
                           neg1_slope=False,
                           fig_title=fig_title, save_name=save_name,
                           save_path=root_path, verbose=True)
    plt.show()

    wide_df = ave_df.pivot(index=['dur_ms'], columns='cond_type', values='delta_thr')
    print(f'wide_df:\n{wide_df}')

    error_df = pd.read_csv(err_path)
    wide_err_df = error_df.pivot(index=['dur_ms'], columns='cond_type', values='delta_thr')
    print(f'wide_err_df:\n{wide_err_df}')

    fig_title = 'Participant average ∆thresholds - Bloch_v2'
    save_name = 'bloch_v2_log_dur_log_delta.png'
    plot_runs_ave_w_errors(fig_df=wide_df, error_df=wide_err_df,
                           jitter=False, error_caps=True, alt_colours=False,
                           legend_names=None,
                           even_spaced_x=False,
                           fixed_y_range=False,
                           x_tick_vals=None,
                           x_tick_labels=None,
                           x_axis_label='log(duration ms) - 1probe condition',
                           y_axis_label='log(∆ Threshold)',
                           log_log_axes=True,
                           neg1_slope=True,
                           fig_title=fig_title, save_name=save_name,
                           save_path=root_path, verbose=True)
    plt.show()
    print('*** finished average plot ***')


#
#     make_average_plots(all_df_path=all_df_path,
#                        ave_df_path=p_ave_path,
#                        error_bars_path=err_path,
#                        n_trimmed=n_trimmed,
#                        exp_ave=False,
#                        show_plots=True, verbose=True)
#
#
# print(f'exp_path: {exp_path}')
# print('\nget exp_average_data')
#
# participant_list = ['Nick', 'Simon']
#
# e_average_exp_data(exp_path=exp_path, p_names_list=participant_list,
#                    error_type='SE', use_trimmed=True, verbose=True)
#
#
# all_df_path = f'{exp_path}/MASTER_exp_thr.csv'
# exp_ave_path = f'{exp_path}/MASTER_exp_ave_thr.csv'
# err_path = f'{exp_path}/MASTER_ave_thr_error_SE.csv'
# n_trimmed = None
# exp_ave = True
#
# make_average_plots(all_df_path=all_df_path,
#                    ave_df_path=exp_ave_path,
#                    error_bars_path=err_path,
#                    error_type='SE',
#                    n_trimmed=n_trimmed,
#                    exp_ave=exp_ave,
#                    show_plots=True, verbose=True)

print('\nExp2_Bloch_analaysis_pipe finished\n')
