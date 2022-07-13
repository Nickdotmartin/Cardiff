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
exp_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\Exp2_Bloch_NM_v5"
# exp_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\Exp2_Bloch_NM_v3"

convert_path1 = os.path.normpath(exp_path)
print(f"convert_path1: {convert_path1}")
exp_path = convert_path1

participant_list = ['Nick', 'Tony']  # 'Nick', 'bb', 'cc', 'dd', 'ee']
n_runs = 6

p_idx_plus = 1

for p_idx, participant_name in enumerate(participant_list):
    root_path = os.path.join(exp_path, participant_name)

    # # manually get run_folder_names with n_runs
    # run_folder_names = [f'{participant_name}_{i+p_idx_plus}' for i in list(range(n_runs))]

    # search to automatically get run_folder_names
    dir_list = os.listdir(root_path)
    run_folder_names = []
    for i in range(12):  # numbers 0 to 11
        check_dir = f'{participant_name}_{i+p_idx_plus}'   # numbers 1 to 12
        if check_dir in dir_list:
            run_folder_names.append(check_dir)

    print(f'run_folder_names: {run_folder_names}')

    for run_idx, run_dir in enumerate(run_folder_names):

        print(f'\nrunning analysis for {participant_name}, {run_dir}, {participant_name}{run_idx+p_idx_plus}\n')
        save_path = f'{root_path}{os.sep}{run_dir}'

        # don't delete this (participant_name = participant_name),
        # needed to ensure names go name1, name2, name3 not name1, name12, name123
        p_name = participant_name

        # '''a'''
        p_name = f'{participant_name}_{run_idx+p_idx_plus}'

        # for first run, some files are saved just as name not name1
        run_data_path = f'{save_path}{os.sep}{p_name}_output.csv'
        if not os.path.isfile(run_data_path):
            raise FileNotFoundError(run_data_path)
        print(f'run_data_path: {run_data_path}')

        run_data_df = pd.read_csv(run_data_path)
        # remove any Unnamed columns
        if any("Unnamed" in i for i in list(run_data_df.columns)):
            unnamed_col = [i for i in list(run_data_df.columns) if "Unnamed" in i][0]
            run_data_df.drop(unnamed_col, axis=1, inplace=True)
        run_data_df.sort_values(by=['stair', 'step'], inplace=True, ignore_index=True)

        '''add newLum column
                in old version, the experiment script varies probeLum and converts to float(RGB255) values for screen.
                However, monitor can only use int(RGB255).
                This function will will round RGB255 values to int(RGB255), then convert to NEW_probeLum
                LumColor255Factor = 2.395387069
                1. get probeColor255 column.
                2. convert to int(RGB255) and convert to new_Lum with int(RGB255)/LumColor255Factor
                3. add to run_data_df'''
        if 'newLum' not in run_data_df.columns.to_list():
            LumColor255Factor = 2.395387069
            rgb255_col = run_data_df['probeColor255'].to_list()
            newLum = [int(i) / LumColor255Factor for i in rgb255_col]
            run_data_df.insert(9, 'newLum', newLum)
            # run_data_df.to_excel(os.path.join(save_path, 'RUNDATA-sorted.xlsx'), index=False)
            print(f"added newLum column\n"
                  f"run_data_df: {run_data_df.columns.to_list()}")

        # run_data_path = os.path.join(save_path, 'RUNDATA-sorted.xlsx')

        # save sorted csv
        run_data_df.to_csv(run_data_path, index=False)
        print(f"run_data_df: {run_data_df.columns}\n{run_data_df}")

        # extract values from dataframe
        isi_list = run_data_df['ISI'].unique()
        print(f'isi_list: {isi_list}')
        cond_types = run_data_df['cond_type'].unique()
        print(f'cond_types: {cond_types}')


        '''get psignifit thresholds df - use stairs as sep levels rather than using groups'''
        thr_df = get_psignifit_threshold_df(root_path=root_path,
                                            p_run_name=run_dir,
                                            csv_name=run_data_df,
                                            n_bins=9, q_bins=True,
                                            isi_list=isi_list,
                                            sep_col='cond_type',
                                            thr_col='newLum',
                                            sep_list=cond_types,
                                            conf_int=True,
                                            thr_type='Bayes',
                                            plot_both_curves=False,
                                            cols_to_add_dict=None,
                                            verbose=True)
        print(f'thr_df: {type(thr_df)}\n{thr_df}')
        #
        # '''# Bloch doesn't currently work with b3_plot_staircase or c_plots
        # b3_plot_staircase(run_data_path, show_plots=True)
        # c_plots(save_path=save_path, isi_name_list=isi_name_list, show_plots=True)'''
        #

        run_data_path = f'{save_path}{os.sep}{p_name}_output.csv'
        run_data_df = pd.read_csv(run_data_path)
        print(f'run_data_df:\n{run_data_df}')
        print(f'run_data_df columns: {run_data_df.columns.to_list()}')
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
                                   cols_to_change_show='newLum',
                                   new_col_name='ISI', strip_from_cols='ISI_', verbose=True)
        print(f'long_thr_df:\n{long_thr_df}')
        long_thr_df_path = f'{save_path}{os.sep}long_thr_df.csv'
        long_thr_df.to_csv(long_thr_df_path, index=False)

        '''run plots from here'''
        long_thr_df_path = f'{save_path}{os.sep}long_thr_df.csv'
        long_thr_df = pd.read_csv(long_thr_df_path)
        print(f'\nlong_thr_df:\n{long_thr_df}')
        isi_list = list(long_thr_df['ISI'].unique())
        isi_name_list = [f'ISI_{i}' for i in isi_list]
        isi_labels_list = ['conc' if i == -2 else i for i in isi_list]
        print(f'isi_list: {isi_list}')
        print(f'isi_name_list: {isi_name_list}')
        print(f'isi_labels_list: {isi_labels_list}')

        # basic plot with regular axes
        run_thr_plot(long_thr_df, x_col='ISI', y_col='newLum', hue_col='cond_type',
                     x_ticks_vals=isi_list,
                     x_tick_names=None,
                     x_axis_label='ISI (as if two-probes from exp1)',
                     y_axis_label='Probe Luminance',
                     fig_title='Bloch_v5: probe cond vs thr',
                     save_as=f'{save_path}{os.sep}bloch_v5_cond_v_thr.png')
        plt.show()

        print(f'long_thr_df:\n{long_thr_df}')

        # # check for 'dur_ms' and 'weber_thr' col
        col_names = long_thr_df.columns.to_list()

        if 'dur_ms' not in col_names:
            # convert separation into area (units are pixels)
            dur_dict = {-2.0: {'frames': 2, 'duration': 8.333333333},
                        0.0: {'frames': 4, 'duration': 16.66666667},
                        8.3333334: {'frames': 6, 'duration': 25},
                        16.6666667: {'frames': 8, 'duration': 33.33333333},
                        25.0: {'frames': 10, 'duration': 41.66666667},
                        37.5: {'frames': 13, 'duration': 54.16666667},
                        50.0: {'frames': 16, 'duration': 66.66666667},
                        100.0: {'frames': 28, 'duration': 116.6666667}}
            ISI_col = long_thr_df['ISI'].to_list()
            ISI_col = [float(i) for i in ISI_col]
            print(f'ISI_col: {ISI_col}')
            dur_col = [dur_dict[i]['duration'] for i in ISI_col]
            print(f'dur_col: {dur_col}')
            long_thr_df.insert(1, 'dur_ms', dur_col)

            # # make extra columns for -1 slope=complete plots - delta_I and weber_frac
            thr_col = long_thr_df['newLum'].to_list()
            bgLum = 21.2
            delta_I_col = [i-bgLum for i in thr_col]
            weber_thr_col = [(i-bgLum)/i for i in thr_col]
            long_thr_df.insert(4, 'delta_I', delta_I_col)
            long_thr_df.insert(5, 'weber_thr', weber_thr_col)

            # # make extra columns for horizontal=complete plots - I*dur and delta_I*dur
            I_times_t_col = [a*b for a, b in zip(thr_col, dur_col)]
            delta_I_times_t_col = [a*b for a, b in zip(delta_I_col, dur_col)]
            long_thr_df.insert(6, 'I_times_t', I_times_t_col)
            long_thr_df.insert(7, 'delta_I_times_t', delta_I_times_t_col)

            print(f'long_thr_df:\n{long_thr_df}')



            if 'stair_name' in col_names:
                long_thr_df.drop('stair_name', axis=1, inplace=True)

            long_thr_df_path = f'{save_path}{os.sep}long_thr_df.csv'
            long_thr_df.to_csv(long_thr_df_path, index=False)
            print(f'long_thr_df:\n{long_thr_df}')


        # plot with log-log axes - Weber
        simple_log_log_plot(long_thr_df, x_col='dur_ms', y_col='weber_thr', hue_col='cond_type',
                            x_ticks_vals=None, x_tick_names=None,
                            x_axis_label='log(duration ms) - 1probe condition',
                            y_axis_label='log(∆I/I)',
                            fig_title='Bloch_v5: log(duration) v log(∆I/I)',
                            show_neg1slope=True,
                            save_as=f'{save_path}{os.sep}bloch_v5_log_dur_log_weber.png')
        plt.show()

        # plot with log-log axes - log(∆i)
        simple_log_log_plot(long_thr_df, x_col='dur_ms', y_col='delta_I', hue_col='cond_type',
                            x_ticks_vals=None, x_tick_names=None,
                            x_axis_label='log(duration ms) - 1probe condition',
                            y_axis_label='log(∆I)',
                            fig_title='Bloch_v5: log(duration) v log(∆I)',
                            show_neg1slope=True,
                            save_as=f'{save_path}{os.sep}bloch_v5_log_dur_log_contrast.png')
        plt.show()


    '''d'''
    trim_n = None
    if len(run_folder_names) == 12:
        trim_n = 2
    thr_df_name = 'long_thr_df'

    d_average_participant(root_path=root_path, run_dir_names_list=run_folder_names,
                          thr_df_name=thr_df_name, trim_n=trim_n, error_type='SE')


    # making average plot
    all_df_path = os.path.join(root_path, f'MASTER_TM{trim_n}_thresholds.csv')
    p_ave_path = os.path.join(root_path, f'MASTER_ave_TM{trim_n}_thresh.csv')
    err_path = os.path.join(root_path, f'MASTER_ave_TM{trim_n}_thr_error_SE.csv')
    if trim_n is None:
        all_df_path = os.path.join(root_path, f'MASTER_{thr_df_name}.csv')
        p_ave_path = os.path.join(root_path, 'MASTER_ave_thresh.csv')
        err_path = os.path.join(root_path, 'MASTER_ave_thr_error_SE.csv')

    # load data and change order to put 1pr last
    print('*** making average plot ***')
    # reshape dfs so that the different conds are in separate columns.
    ave_df = pd.read_csv(p_ave_path)
    print(f'ave_df:\n{ave_df}')



    # # fig 1, standard axes (not log)
    wide_df = ave_df.pivot(index=['dur_ms'], columns='cond_type', values='newLum')
    print(f'wide_df:\n{wide_df}')

    dur_list = ave_df['dur_ms'].to_list()
    print(f'dur_list: {dur_list}')

    error_df = pd.read_csv(err_path)
    wide_err_df = error_df.pivot(index=['dur_ms'], columns='cond_type', values='newLum')
    print(f'wide_err_df:\n{wide_err_df}')
    fig_title = 'Participant average thresholds - Bloch_v5'
    save_name = 'bloch_v5_dur_v_thr.png'
    plot_runs_ave_w_errors(fig_df=wide_df, error_df=wide_err_df,
                           jitter=False, error_caps=True, alt_colours=False,
                           legend_names=None,
                           even_spaced_x=True,
                           fixed_y_range=False,
                           x_tick_vals=dur_list,
                           x_tick_labels=dur_list,
                           x_axis_label='Duration (ms)',
                           y_axis_label='Threshold',
                           log_log_axes=False,
                           neg1_slope=False,
                           fig_title=fig_title, save_name=save_name,
                           save_path=root_path, verbose=True)
    plt.show()

    # # fig 2 - log dur, log(∆I)
    wide_df = ave_df.pivot(index=['dur_ms'], columns='cond_type', values='delta_I')
    print(f'wide_df:\n{wide_df}')

    error_df = pd.read_csv(err_path)
    wide_err_df = error_df.pivot(index=['dur_ms'], columns='cond_type', values='delta_I')
    print(f'wide_err_df:\n{wide_err_df}')

    fig_title = 'Participant average log(∆I) thresholds - Bloch_v5'
    save_name = 'bloch_v5_log_dur_log_contrast.png'
    plot_runs_ave_w_errors(fig_df=wide_df, error_df=wide_err_df,
                           jitter=False, error_caps=True, alt_colours=False,
                           legend_names=None,
                           even_spaced_x=False,
                           fixed_y_range=False,
                           x_tick_vals=None,
                           x_tick_labels=None,
                           x_axis_label='log(duration ms) - 1probe condition',
                           y_axis_label='Contrast: log(∆I)',
                           log_log_axes=True,
                           neg1_slope=True,
                           slope_ycol_name='1probe',
                           slope_xcol_idx_depth=1,
                           fig_title=fig_title, save_name=save_name,
                           save_path=root_path, verbose=True)
    plt.show()

    # # fig 3 - log dur, log(weber_thr)
    wide_df = ave_df.pivot(index=['dur_ms'], columns='cond_type', values='weber_thr')
    print(f'wide_df:\n{wide_df}')

    error_df = pd.read_csv(err_path)
    wide_err_df = error_df.pivot(index=['dur_ms'], columns='cond_type', values='weber_thr')
    print(f'wide_err_df:\n{wide_err_df}')

    fig_title = 'Participant average log(∆I/I) thresholds - Bloch_v5'
    save_name = 'bloch_v5_log_dur_log_weber.png'
    plot_runs_ave_w_errors(fig_df=wide_df, error_df=wide_err_df,
                           jitter=False, error_caps=True, alt_colours=False,
                           legend_names=None,
                           even_spaced_x=False,
                           fixed_y_range=False,
                           x_tick_vals=None,
                           x_tick_labels=None,
                           x_axis_label='log(duration ms) - 1probe condition',
                           y_axis_label='Weber threshold: log(∆I/I)',
                           log_log_axes=True,
                           neg1_slope=True,
                           slope_ycol_name='1probe',
                           slope_xcol_idx_depth=1,
                           fig_title=fig_title, save_name=save_name,
                           save_path=root_path, verbose=True)
    plt.show()
    print('*** finished participant average plot ***')



print(f'exp_path: {exp_path}')
print('\nget exp_average_data')

# participant_list = ['Nick', 'Tony']

e_average_exp_data(exp_path=exp_path, p_names_list=participant_list, exp_type='Bloch',
                   error_type='SE', n_trimmed=trim_n, verbose=True)


all_df_path = os.path.join(exp_path, 'MASTER_exp_thr.csv')
exp_ave_path = os.path.join(exp_path, 'MASTER_exp_ave_thr.csv')
err_path = os.path.join(exp_path, 'MASTER_ave_thr_error_SE.csv')

# print('*** making average plot ***')
print('*** making average plot ***')
fig_df = pd.read_csv(exp_ave_path)
print(f'fig_df:\n{fig_df}')

isi_vals_list = fig_df['ISI'].to_list()
isi_names_list = ['1pr' if i == -2 else int(i) for i in isi_vals_list]
print(f'isi_names_list: {isi_names_list}')

dur_list = fig_df['dur_ms'].to_list()
print(f'dur_list: {dur_list}')


error_df = pd.read_csv(err_path)
print(f'error_df:\n{error_df}')


# fig 1 - ave thr by sep
ave_thr_by_dur_df = fig_df[['dur_ms', 'thr']]
ave_thr_by_dur_df.set_index('dur_ms', inplace=True)
err_thr_by_dur_df = error_df[['dur_ms', 'thr']]
err_thr_by_dur_df.set_index('dur_ms', inplace=True)
print(f'ave_thr_by_dur_df:\n{ave_thr_by_dur_df}')

fig_title = 'Experiment average thresholds - Bloch_v5'
save_name = 'bloch_v5_isi_v_thr.png'
plot_runs_ave_w_errors(fig_df=ave_thr_by_dur_df, error_df=err_thr_by_dur_df,
                       jitter=False, error_caps=True, alt_colours=False,
                       legend_names=None,
                       even_spaced_x=True,
                       fixed_y_range=False,
                       x_tick_vals=None,
                       x_tick_labels=None,
                       x_axis_label='duration (ms)',
                       y_axis_label='Luminance threshold',
                       log_log_axes=False,
                       neg1_slope=False,
                       fig_title=fig_title, save_name=save_name,
                       save_path=exp_path, verbose=True)
plt.show()

# # plot 2: log-log axes - log(∆i)
log_dur_log_delta_I_df = fig_df[['dur_ms', 'delta_I']]
log_dur_log_delta_I_df.set_index('dur_ms', inplace=True)
err_log_dur_log_delta_I_df = error_df[['dur_ms', 'delta_I']]
err_log_dur_log_delta_I_df.set_index('dur_ms', inplace=True)
print(f'log_dur_log_delta_I_df:\n{log_dur_log_delta_I_df}')
fig_title = 'Experiment average log(∆I) thresholds - Bloch_v5'
save_name = 'bloch_v5_log_dur_log_contrast.png'
plot_runs_ave_w_errors(fig_df=log_dur_log_delta_I_df, error_df=err_log_dur_log_delta_I_df,
                       jitter=False, error_caps=True, alt_colours=False,
                       legend_names=None,
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
                       save_path=exp_path, verbose=True)
plt.show()

# # plot 3: log-log axes - log(∆I/I)

log_dur_log_weber_df = fig_df[['dur_ms', 'weber_thr']]
log_dur_log_weber_df.set_index('dur_ms', inplace=True)
err_log_dur_log_weber_df = error_df[['dur_ms', 'weber_thr']]
err_log_dur_log_weber_df.set_index('dur_ms', inplace=True)
print(f'log_area_log_weber_df:\n{log_dur_log_weber_df}')

fig_title = 'Experiment average log(∆I/I) thresholds - Bloch_v5'
save_name = 'bloch_v5_log_dur_log_weber.png'
plot_runs_ave_w_errors(fig_df=log_dur_log_weber_df, error_df=err_log_dur_log_weber_df,
                       jitter=False, error_caps=True, alt_colours=False,
                       legend_names=None,
                       even_spaced_x=False,
                       fixed_y_range=False,
                       x_tick_vals=None,
                       x_tick_labels=None,
                       x_axis_label='log(duration ms) - 1probe condition',
                       y_axis_label='Weber threshold: log(∆I/I)',
                       log_log_axes=True,
                       neg1_slope=True,
                       slope_ycol_name='weber_thr',
                       slope_xcol_idx_depth=1,
                       fig_title=fig_title, save_name=save_name,
                       save_path=exp_path, verbose=True)
plt.show()


print('*** finished exp average plot ***')

# todo: wrap these plot functions for participant and experiment averages into a function


print('\nExp2_Bloch_analaysis_pipe finished\n')
