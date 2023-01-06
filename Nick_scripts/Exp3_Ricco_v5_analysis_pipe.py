import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from rad_flow_psignifit_analysis import b3_plot_staircase, b3_plot_stair_sep0, c_plots, d_average_participant
from rad_flow_psignifit_analysis import make_average_plots, e_average_exp_data, \
    plot_runs_ave_w_errors, plot_w_errors_either_x_axis, run_thr_plot, \
    simple_log_log_plot, log_log_w_markers_plot
from psignifit_tools import get_psignifit_threshold_df
from check_home_dir import which_path, running_on_laptop, switch_path

# # loop through run folders with first 4 scripts (a, get_psignifit_threshold_df, b3, c)
# # then run script d to get master lists and averages
exp_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\Exp3_Ricco_NM_v5"
# exp_path = r"C:\Users\sapnm4\PycharmProjects\Cardiff\Nick_scripts\Exp3_Ricco_NM_DEMO"
convert_path1 = os.path.normpath(exp_path)
if running_on_laptop():
    convert_path1 = switch_path(convert_path1, 'mac_oneDrive')
exp_path = convert_path1
print(f"exp_path: {exp_path}")

# participant_list = ['Kim']  # , 'bb', 'cc', 'dd', 'ee']
# participant_list = ['Kim', 'Nick', 'Simon']  # , 'bb', 'cc', 'dd', 'ee']
participant_list = ['Nick']


n_runs = 1
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

        # # # for first run, some files are saved just as name not name1
        # run_data_path = f'{save_path}{os.sep}{p_name}_output.csv'
        # if not os.path.isfile(run_data_path):
        #     raise FileNotFoundError(run_data_path)
        # print(f'run_data_path: {run_data_path}')
        # run_data_df = pd.read_csv(run_data_path)
        #
        # # remove any Unnamed columns
        # if any("Unnamed" in i for i in list(run_data_df.columns)):
        #     unnamed_col = [i for i in list(run_data_df.columns) if "Unnamed" in i][0]
        #     run_data_df.drop(unnamed_col, axis=1, inplace=True)
        # run_data_df.sort_values(by=['stair', 'step'], inplace=True, ignore_index=True)
        #
        # '''add newLum column
        # in old version, the experiment script varies probeLum and converts to float(RGB255) values for screen.
        # However, monitor can only use int(RGB255).
        # This function will will round RGB255 values to int(RGB255), then convert to NEW_probeLum
        # LumColor255Factor = 2.395387069
        # 1. get probeColor255 column.
        # 2. convert to int(RGB255) and convert to new_Lum with int(RGB255)/LumColor255Factor
        # 3. add to run_data_df'''
        # if 'newLum' not in run_data_df.columns.to_list():
        #     LumColor255Factor = 2.395387069
        #     rgb255_col = run_data_df['probeColor255'].to_list()
        #     newLum = [int(i) / LumColor255Factor for i in rgb255_col]
        #     run_data_df.insert(9, 'newLum', newLum)
        #     print(f"added newLum column\n"
        #           f"run_data_df: {run_data_df.columns.to_list()}")
        #
        # # # save sorted csv
        # run_data_df.to_csv(run_data_path, index=False)
        #
        # run_data_df = pd.read_csv(run_data_path,
        #                           usecols=['trial_number', 'stair', 'stair_name',
        #                                    'step', 'separation', 'cond_type',
        #                                    'ISI', 'corner', 'newLum',
        #                                    'trial_response', '4. fps',
        #                                    'n_pix', 'len_pix', 'diag_deg'])
        # print(f"run_data_df: {run_data_df.columns.to_list()}\n{run_data_df}\n")
        #
        # # extract values from dataframe
        # separation_values = run_data_df['separation'].unique()
        #
        # stair_list = run_data_df['stair'].unique()
        # isi_list = run_data_df['ISI'].unique()
        # print(f'separation_values: {separation_values}')
        # print(f'isi_list: {isi_list}')
        # print(f'stair_list: {stair_list}')
        #
        # n_pixels_list = run_data_df['n_pix'].unique()
        # len_pixels_list = run_data_df['len_pix'].unique()
        # len_degrees_list = run_data_df['diag_deg'].unique()
        # print(f'n_pixels_list: {n_pixels_list}')
        # print(f'len_pixels_list: {len_pixels_list}')
        # print(f'len_degrees_list: {len_degrees_list}')
        #
        # cond_types = run_data_df['cond_type'].unique()
        # sep_vals_list = list(np.repeat(separation_values, len(cond_types)))
        # cond_type_list = list(np.tile(cond_types, len(separation_values)))
        # print(f'cond_types: {cond_types}')
        # print(f'sep_vals_list: {sep_vals_list}')
        # print(f'cond_type_list: {cond_type_list}')
        #
        # stair_names_list = run_data_df['stair_name'].unique()
        # print(f'stair_names_list: {stair_names_list}')
        # cols_to_add_dict = {'stair_names': stair_names_list,
        #                     'separation': sep_vals_list,
        #                     'cond': cond_type_list,
        #                     'n_pix': n_pixels_list,
        #                     'len_pix': len_pixels_list,
        #                     'diag_deg': len_degrees_list,
        #                     }
        #
        # thr_save_name = 'psignifit_thresholds'
        # thr_df = get_psignifit_threshold_df(root_path=root_path,
        #                                     p_run_name=run_dir,
        #                                     csv_name=run_data_df,
        #                                     n_bins=9, q_bins=True,
        #                                     sep_col='stair_name',
        #                                     thr_col='newLum',
        #                                     isi_list=isi_list,
        #                                     sep_list=stair_names_list,
        #                                     conf_int=True,
        #                                     thr_type='Bayes',
        #                                     plot_both_curves=False,
        #                                     save_plots=False,
        #                                     cols_to_add_dict=cols_to_add_dict,
        #                                     save_name=thr_save_name,
        #                                     verbose=True)
        # print(f'thr_df: {type(thr_df)}\n{thr_df}')
        #
        # run_data_path = f'{save_path}{os.sep}{p_name}_output.csv'
        # run_data_df = pd.read_csv(run_data_path)
        # print(f'run_data_df:\n{run_data_df}')


        '''Run figs from here'''
        thr_df_path = f'{save_path}{os.sep}psignifit_thresholds.csv'
        # thr_df_path = f'{save_path}{os.sep}{thr_save_name}.csv'
        thr_df = pd.read_csv(thr_df_path)
        print(f'thr_df:\n{thr_df}\n')

        print(f'thr_df ({thr_df.columns.to_list()}):\n{thr_df}')

        # check for 'area' and 'weber_thr' col
        sep_list = thr_df['separation'].to_list()
        sep_vals_list = [i for i in sep_list]
        sep_name_list = ['1pr' if i == -1 else f'sep{i}' for i in sep_list]
        print(f'sep_vals_list: {sep_vals_list}')
        print(f'sep_name_list: {sep_name_list}\n')

        n_pix_list = thr_df['n_pix'].unique().tolist()
        len_pix_list = thr_df['len_pix'].unique().tolist()
        len_deg_list = thr_df['diag_deg'].unique().tolist()
        print(f'n_pix_list: {n_pix_list}')
        print(f'len_pix_list: {len_pix_list}')
        print(f'len_deg_list: {len_deg_list}')

        # add area in pixels and length in arc-min
        col_names = thr_df.columns.to_list()
        print(f'col_names:\n{col_names}\n')

        if 'ISI_0' in col_names:
            thr_df.rename(columns={'ISI_0': 'thr'}, inplace=True)

        if 'delta_I' not in col_names:
            thr_col = thr_df['thr'].to_list()
            bgLum = 21.2
            delta_I_col = [i - bgLum for i in thr_col]
            thr_df.insert(5, 'delta_I', delta_I_col)


        if 'stair_name' in col_names:
            thr_df.drop('stair_name', axis=1, inplace=True)

        print(f'thr_df:\n{thr_df}')
        thr_df.to_csv(thr_df_path, index=False)


        # # fig 1. basic plot with exp1 axes - n_pix vs thr
        # run_thr_plot(thr_df, x_col='n_pix', y_col='thr', hue_col='cond',
        #              x_ticks_vals=n_pix_list,
        #              x_axis_label='Number of pixels',
        #              y_axis_label='Probe Luminance',
        #              fig_title='P. Ricco_v5: n_pix vs thresholds',
        #              save_as=f'{save_path}{os.sep}ricco_v5_n_pix_v_thr.png')
        # plt.show()
        #
        #
        # # fig 2. plot with log-log axes -  n_pix v delta_I
        # simple_log_log_plot(thr_df, x_col='n_pix', y_col='delta_I', hue_col='cond',
        #                     x_ticks_vals=None, x_tick_names=None,
        #                     x_axis_label='log(Number of pixels)',
        #                     y_axis_label='Contrast: log(∆I)',
        #                     fig_title='P. Ricco_v5: log(n_pix) v log(∆I)',
        #                     save_as=f'{save_path}{os.sep}ricco_v5_log_n_pix_log_contrast.png')
        # plt.show()
        #
        # # not using len pix, but sep cond instead
        # # # fig 3. basic plot with exp1 axes - len_pix vs thr
        # # run_thr_plot(thr_df, x_col='len_pix', y_col='thr', hue_col='cond',
        # #              x_ticks_vals=len_pixels_list, x_tick_names=len_pixels_list,
        # #              x_axis_label='length (diagonal pixels)',
        # #              y_axis_label='Probe Luminance',
        # #              fig_title='P. Ricco_v5: length vs thresholds',
        # #              save_as=f'{save_path}{os.sep}ricco_v5_len_v_thr.png')
        # # plt.show()
        # #
        # # # fig 4. plot with log-log axes - length len_pix v delta_I
        # # simple_log_log_plot(thr_df, x_col='len_pix', y_col='delta_I', hue_col='cond',
        # #                     x_ticks_vals=None, x_tick_names=None,
        # #                     x_axis_label='log(length, diagonal pixels)',
        # #                     y_axis_label='Contrast: log(∆I)',
        # #                     fig_title='P. Ricco_v5: log(length) v log(∆I)',
        # #                     save_as=f'{save_path}{os.sep}ricco_v5_log_length_log_contrast.png')
        # # plt.show()
        #
        # # fig 3. basic plot with exp1 axes - sep cond vs thr
        # run_thr_plot(thr_df, x_col='separation', y_col='thr', hue_col='cond',
        #              x_ticks_vals=sep_vals_list, x_tick_names=sep_name_list,
        #              x_axis_label='length (separation cond)',
        #              y_axis_label='Probe Luminance',
        #              fig_title='P. Ricco_v5: separation condition vs thresholds',
        #              save_as=f'{save_path}{os.sep}ricco_v5_sep_cond_v_thr.png')
        # plt.show()

        # fig 4. plot with log-log axes - length len_pix v delta_I
        log_log_w_markers_plot(thr_df, x_col='len_pix', y_col='delta_I',
                               # hue_col='separation',
                               # x_ticks_vals=None,
                               x_tick_names=sep_name_list,
                               x_axis_label='log(length, diagonal pixels)',
                               y_axis_label='Contrast: log(∆I)',
                               fig_title='P. Ricco_v5: log(length) v log(∆I)',
                               save_as=f'{save_path}{os.sep}ricco_v5_log_length_log_contrast.png')
        plt.show()

        # # fig 5. basic plot with exp1 axes - len_degrees vs thr
        # run_thr_plot(thr_df, x_col='diag_deg', y_col='thr', hue_col='cond',
        #              x_ticks_vals=len_deg_list, x_tick_names=[round(i, 2) for i in len_deg_list],
        #              x_axis_label='length (degrees)',
        #              y_axis_label='Probe Luminance',
        #              fig_title='P. Ricco_v5: length vs thresholds',
        #              save_as=f'{save_path}{os.sep}ricco_v5_len_deg_v_thr.png')
        # plt.show()
        #
        # # fig 6. plot with log-log axes - length len_degrees v delta_I
        # simple_log_log_plot(thr_df, x_col='diag_deg', y_col='delta_I', hue_col='cond',
        #                     x_ticks_vals=None, x_tick_names=None,
        #                     x_axis_label='log(length, degrees)',
        #                     y_axis_label='Contrast: log(∆I)',
        #                     fig_title='P. Ricco_v5: log(degrees) v log(∆I)',
        #                     save_as=f'{save_path}{os.sep}ricco_v5_log_degrees_log_contrast.png')
        # plt.show()


#     '''d'''
#     print(f"\ntrim_n: {trim_n}")
#     d_average_participant(root_path=root_path, run_dir_names_list=run_folder_names,
#                           trim_n=trim_n, error_type='SE',
#                           groupby_col=['separation'],
#                           cols_to_drop=['stair_names', 'stack', 'cond'],
#                           )
#
#     # making average plot
#     all_df_path = os.path.join(root_path, f'MASTER_TM{trim_n}_thresholds.csv')
#     p_ave_path = os.path.join(root_path, f'MASTER_ave_TM{trim_n}_thresh.csv')
#     err_path = os.path.join(root_path, f'MASTER_ave_TM{trim_n}_thr_error_SE.csv')
#     if trim_n is None:
#         all_df_path = os.path.join(root_path, 'MASTER_psignifit_thresholds.csv')
#         p_ave_path = os.path.join(root_path, 'MASTER_ave_thresh.csv')
#         err_path = os.path.join(root_path, 'MASTER_ave_thr_error_SE.csv')
#
#     exp_ave = False
#
#     # todo: wrap these plot functions for participant and experiment averages into a function
#     # load data and change order to put 1pr last
#     print('*** making average plot ***')
#     print(f'root_path: {root_path}')
#
#
#     fig_df = pd.read_csv(p_ave_path)
#     print(f'fig_df:\n{fig_df}')
#
#     error_df = pd.read_csv(err_path)
#
#
#
#
#     len_pixels_list = fig_df['len_pix'].to_list()
#     print(f'len_pixels_list: {len_pixels_list}')
#     if 'len_pix' not in error_df.columns:
#         error_df.insert(4, 'len_pix', len_pixels_list)
#     else:
#         error_df['len_pix'] = len_pixels_list
#
#     len_degrees_list = fig_df['diag_deg'].to_list()
#     print(f'len_degrees_list: {len_degrees_list}')
#     if 'diag_deg' not in error_df.columns:
#         error_df.insert(4, 'diag_deg', len_degrees_list)
#     else:
#         error_df['diag_deg'] = len_degrees_list
#
#
#     if 'cond' not in error_df.columns:
#         cond_list = ['lines'] * len(len_pixels_list)
#         error_df.insert(1, 'cond', cond_list)
#         fig_df.insert(1, 'cond', cond_list)
#     print(f'error_df:\n{error_df}')
#
#     print(f'fig_df:\n{fig_df}')
#
#
#
#     # # fig 3 - len pixels v thr
#     wide_df = fig_df.pivot(index=['len_pix'], columns='cond', values='thr')
#     print(f'wide_df:\n{wide_df}')
#     wide_err_df = error_df.pivot(index=['len_pix'], columns='cond', values='thr')
#
#     len_pixels_list = fig_df['len_pix'].to_list()
#     print(f'len_pixels_list: {len_pixels_list}')
#
#     fig_title = f'Participant average thresholds - Ricco_v5 (n={len(run_folder_names)})'
#     save_name = 'ricco_v5_len_v_thr.png'
#     plot_runs_ave_w_errors(fig_df=wide_df, error_df=wide_err_df,
#                            jitter=False, error_caps=True, alt_colours=False,
#                            legend_names=None,
#                            even_spaced_x=False,
#                            fixed_y_range=False,
#                            x_tick_vals=len_pixels_list,
#                            x_tick_labels=len_pixels_list,
#                            x_axis_label='Length (diagonal pixels)',
#                            y_axis_label='Threshold',
#                            log_log_axes=False,
#                            neg1_slope=False,
#                            fig_title=fig_title, save_name=save_name,
#                            save_path=root_path, verbose=True)
#     plt.show()
#
#     # fig 4 - log(len len_pix), log(contrast)
#     wide_df = fig_df.pivot(index=['len_pix'], columns='cond', values='delta_I')
#     print(f'wide_df:\n{wide_df}')
#
#     wide_err_df = error_df.pivot(index=['len_pix'], columns='cond', values='delta_I')
#     print(f'wide_err_df:\n{wide_err_df}')
#
#     fig_title = f'Participant average log(len), log(∆I) thresholds - Ricco_v5 (n={len(run_folder_names)})'
#     save_name = 'ricco_v5_log_len_log_contrast.png'
#     plot_runs_ave_w_errors(fig_df=wide_df, error_df=wide_err_df,
#                            jitter=False, error_caps=True, alt_colours=False,
#                            legend_names=None,
#                            even_spaced_x=False,
#                            fixed_y_range=False,
#                            x_tick_vals=None,
#                            x_tick_labels=None,
#                            x_axis_label='log(length, diagonal pixels)',
#                            y_axis_label='Contrast: log(∆I)',
#                            log_log_axes=True,
#                            neg1_slope=True,
#                            slope_ycol_name='lines',
#                            slope_xcol_idx_depth=1,
#                            fig_title=fig_title, save_name=save_name,
#                            save_path=root_path, verbose=True)
#     plt.show()
#
#
#
#     # # fig 5 - len degrees v thr
#     wide_df = fig_df.pivot(index=['diag_deg'], columns='cond', values='thr')
#     print(f'wide_df:\n{wide_df}')
#     wide_err_df = error_df.pivot(index=['diag_deg'], columns='cond', values='thr')
#
#     len_degrees_list = fig_df['diag_deg'].to_list()
#     print(f'len_degrees_list: {len_degrees_list}')
#
#     fig_title = f'Participant average thresholds - Ricco_v5 (n={len(run_folder_names)})'
#     save_name = 'ricco_v5_len_deg_v_thr.png'
#     plot_runs_ave_w_errors(fig_df=wide_df, error_df=wide_err_df,
#                            jitter=False, error_caps=True, alt_colours=False,
#                            legend_names=None,
#                            even_spaced_x=False,
#                            fixed_y_range=False,
#                            x_tick_vals=len_degrees_list,
#                            x_tick_labels=[round(i, 2) for i in len_degrees_list],
#                            x_axis_label='Length (degrees)',
#                            y_axis_label='Threshold',
#                            log_log_axes=False,
#                            neg1_slope=False,
#                            fig_title=fig_title, save_name=save_name,
#                            save_path=root_path, verbose=True)
#     plt.show()
#
#     # fig 6 - log(len len_pix), log(contrast)
#     wide_df = fig_df.pivot(index=['diag_deg'], columns='cond', values='delta_I')
#     print(f'wide_df:\n{wide_df}')
#
#     wide_err_df = error_df.pivot(index=['diag_deg'], columns='cond', values='delta_I')
#     print(f'wide_err_df:\n{wide_err_df}')
#
#     fig_title = f'Participant average log(degrees), log(∆I) thresholds - Ricco_v5 (n={len(run_folder_names)})'
#     save_name = 'ricco_v5_log_deg_log_contrast.png'
#     plot_runs_ave_w_errors(fig_df=wide_df, error_df=wide_err_df,
#                            jitter=False, error_caps=True, alt_colours=False,
#                            legend_names=None,
#                            even_spaced_x=False,
#                            fixed_y_range=False,
#                            x_tick_vals=None,
#                            x_tick_labels=None,
#                            x_axis_label='log(length, degrees)',
#                            y_axis_label='Contrast: log(∆I)',
#                            log_log_axes=True,
#                            neg1_slope=True,
#                            slope_ycol_name='lines',
#                            slope_xcol_idx_depth=1,
#                            fig_title=fig_title, save_name=save_name,
#                            save_path=root_path, verbose=True)
#     plt.show()
#
#
#     n_pixels_list = fig_df['n_pix'].to_list()
#     print(f'n_pixels_list: {n_pixels_list}')
#     if 'n_pix' not in error_df.columns:
#         error_df.insert(4, 'n_pix', n_pixels_list)
#     else:
#         error_df['n_pix'] = n_pixels_list
#
#     # # fig 1 - n pixels v thr
#     wide_df = fig_df.pivot(index=['n_pix'], columns='cond', values='thr')
#     print(f'wide_df:\n{wide_df}')
#     wide_err_df = error_df.pivot(index=['n_pix'], columns='cond', values='thr')
#
#     n_pixels_list = fig_df['n_pix'].to_list()
#     print(f'n_pixels_list: {n_pixels_list}')
#
#     fig_title = f'Participant average thresholds - Ricco_v5 (n={len(run_folder_names)})'
#     save_name = 'ricco_v5_n_pix_v_thr.png'
#     plot_runs_ave_w_errors(fig_df=wide_df, error_df=wide_err_df,
#                            jitter=False, error_caps=True, alt_colours=False,
#                            legend_names=None,
#                            even_spaced_x=False,
#                            fixed_y_range=False,
#                            x_tick_vals=n_pixels_list,
#                            x_tick_labels=n_pixels_list,
#                            x_axis_label='Number of pixels',
#                            y_axis_label='Threshold',
#                            log_log_axes=False,
#                            neg1_slope=False,
#                            fig_title=fig_title, save_name=save_name,
#                            save_path=root_path, verbose=True)
#     plt.show()
#
#     # fig 2 - log(len len_pix), log(contrast)
#     wide_df = fig_df.pivot(index=['n_pix'], columns='cond', values='delta_I')
#     print(f'wide_df:\n{wide_df}')
#
#     wide_err_df = error_df.pivot(index=['n_pix'], columns='cond', values='delta_I')
#     print(f'wide_err_df:\n{wide_err_df}')
#
#     fig_title = f'Participant average log(len), log(∆I) thresholds - Ricco_v5 (n={len(run_folder_names)})'
#     save_name = 'ricco_v5_log_n_pix_log_contrast.png'
#     plot_runs_ave_w_errors(fig_df=wide_df, error_df=wide_err_df,
#                            jitter=False, error_caps=True, alt_colours=False,
#                            legend_names=None,
#                            even_spaced_x=False,
#                            fixed_y_range=False,
#                            x_tick_vals=None,
#                            x_tick_labels=None,
#                            x_axis_label='log(number of pixels)',
#                            y_axis_label='Contrast: log(∆I)',
#                            log_log_axes=True,
#                            neg1_slope=True,
#                            slope_ycol_name='lines',
#                            slope_xcol_idx_depth=1,
#                            fig_title=fig_title, save_name=save_name,
#                            save_path=root_path, verbose=True)
#     plt.show()
#
#     print('*** finished participant average plots ***')
#
#
# participant_list = ['Nick']  #, 'Kim', 'Kris', 'Simon']
# print(f'exp_path: {exp_path}')
# print('\nget exp_average_data')
# e_average_exp_data(exp_path=exp_path, p_names_list=participant_list,
#                    # exp_type='Ricco',
#                    exp_type='Ricco_v5',
#                    error_type='SE', n_trimmed=trim_n, verbose=True)
#
#
# all_df_path = os.path.join(exp_path, 'MASTER_exp_thr.csv')
# exp_ave_path = os.path.join(exp_path, 'MASTER_exp_ave_thr.csv')
# err_path = os.path.join(exp_path, 'MASTER_ave_thr_error_SE.csv')
# exp_ave = True
#
# # # making Experiment average plot
# print('*** making average plot (number of pixels)***')
# fig_df = pd.read_csv(exp_ave_path)
# print(f'fig_df:\n{fig_df}')
#
# len_pix_list = fig_df['len_pix'].to_list()
# print(f'len_pix_list: {len_pix_list}')
#
# error_df = pd.read_csv(err_path)
# print(f'error_df:\n{error_df}')
#
#
# # # fig 1 - ave thr by sep
# ave_thr_by_n_pix_df = fig_df[['n_pix', 'thr']]
# ave_thr_by_n_pix_df.set_index('n_pix', inplace=True)
# err_thr_by_n_pix_df = error_df[['n_pix', 'thr']]
# err_thr_by_n_pix_df.set_index('n_pix', inplace=True)
# print(f'ave_thr_by_n_pix_df:\n{ave_thr_by_n_pix_df}')
#
# fig_title = f'Experiment average thresholds - Ricco_v5 (n={len(participant_list)})'
# save_name = 'ricco_v5_n_pix_v_thr.png'
# plot_runs_ave_w_errors(fig_df=ave_thr_by_n_pix_df, error_df=err_thr_by_n_pix_df,
#                        jitter=False, error_caps=True, alt_colours=False,
#                        legend_names=None,
#                        even_spaced_x=False,
#                        fixed_y_range=False,
#                        x_tick_vals=len_pix_list,
#                        x_tick_labels=len_pix_list,
#                        x_axis_label='Number of pixels',
#                        y_axis_label='Threshold',
#                        log_log_axes=False,
#                        neg1_slope=False,
#                        fig_title=fig_title, save_name=save_name,
#                        save_path=exp_path, verbose=True)
# plt.show()
#
#
# # # fig 2: log n_pix, log thr
# log_n_pix_log_contrast_df = fig_df[['n_pix', 'delta_I']]
# log_n_pix_log_contrast_df.set_index('n_pix', inplace=True)
# err_log_n_pix_log_contrast_df = error_df[['n_pix', 'delta_I']]
# err_log_n_pix_log_contrast_df.set_index('n_pix', inplace=True)
# fig_title = f'Experiment average log(n_pix), log(∆I) thresholds - Ricco_v5 (n={len(participant_list)})'
# save_name = 'ricco_v5_log_n_pix_log_contrast.png'
# plot_runs_ave_w_errors(fig_df=log_n_pix_log_contrast_df, error_df=err_log_n_pix_log_contrast_df,
#                        jitter=False, error_caps=True, alt_colours=False,
#                        legend_names=None,
#                        even_spaced_x=False,
#                        fixed_y_range=False,
#                        x_tick_vals=None,
#                        x_tick_labels=None,
#                        x_axis_label='log(number of pixels)',
#                        y_axis_label='Contrast: log(∆I)',
#                        log_log_axes=True,
#                        neg1_slope=True,
#                        slope_ycol_name='delta_I',
#                        slope_xcol_idx_depth=1,
#                        fig_title=fig_title, save_name=save_name,
#                        save_path=exp_path, verbose=True)
# plt.show()
#
#
# print('*** making average plot (length, mm) ***')
# fig_df = pd.read_csv(exp_ave_path)
# print(f'fig_df:\n{fig_df}')
#
# len_pix_list = fig_df['len_pix'].to_list()
# print(f'len_pix_list: {len_pix_list}')
#
# error_df = pd.read_csv(err_path)
# print(f'error_df:\n{error_df}')
#
# # # fig 1 - ave thr by sep
# ave_thr_by_len_df = fig_df[['len_pix', 'thr']]
# ave_thr_by_len_df.set_index('len_pix', inplace=True)
# err_thr_by_len_df = error_df[['len_pix', 'thr']]
# err_thr_by_len_df.set_index('len_pix', inplace=True)
# print(f'ave_thr_by_len_df:\n{ave_thr_by_len_df}')
#
# fig_title = f'Experiment average thresholds - Ricco_v5 (n={len(participant_list)})'
# save_name = 'ricco_v5_len_v_thr.png'
# plot_runs_ave_w_errors(fig_df=ave_thr_by_len_df, error_df=err_thr_by_len_df,
#                        jitter=False, error_caps=True, alt_colours=False,
#                        legend_names=None,
#                        even_spaced_x=False,
#                        fixed_y_range=False,
#                        x_tick_vals=len_pix_list,
#                        x_tick_labels=len_pix_list,
#                        x_axis_label='Length: (diagonal pixels)',
#                        y_axis_label='Threshold',
#                        log_log_axes=False,
#                        neg1_slope=False,
#                        fig_title=fig_title, save_name=save_name,
#                        save_path=exp_path, verbose=True)
# plt.show()
#
#
# # # fig 2: log len, log thr
# log_len_log_contrast_df = fig_df[['len_pix', 'delta_I']]
# log_len_log_contrast_df.set_index('len_pix', inplace=True)
# err_log_len_log_contrast_df = error_df[['len_pix', 'delta_I']]
# err_log_len_log_contrast_df.set_index('len_pix', inplace=True)
# fig_title = f'Experiment average log(len), log(∆I) thresholds - Ricco_v5 (n={len(participant_list)})'
# save_name = 'ricco_v5_log_len_log_contrast.png'
# plot_runs_ave_w_errors(fig_df=log_len_log_contrast_df, error_df=err_log_len_log_contrast_df,
#                        jitter=False, error_caps=True, alt_colours=False,
#                        legend_names=None,
#                        even_spaced_x=False,
#                        fixed_y_range=False,
#                        x_tick_vals=None,
#                        x_tick_labels=None,
#                        x_axis_label='log(length, diagonal pixels)',
#                        y_axis_label='Contrast: log(∆I)',
#                        log_log_axes=True,
#                        neg1_slope=True,
#                        slope_ycol_name='delta_I',
#                        slope_xcol_idx_depth=1,
#                        fig_title=fig_title, save_name=save_name,
#                        save_path=exp_path, verbose=True)
# plt.show()
#
#
# print('*** making average plot (length degrees)***')
# fig_df = pd.read_csv(exp_ave_path)
# print(f'fig_df:\n{fig_df}')
#
# len_deg_list = fig_df['diag_deg'].to_list()
# print(f'len_deg_list: {len_deg_list}')
#
# error_df = pd.read_csv(err_path)
# print(f'error_df:\n{error_df}')
#
# # # fig 1 - ave thr by sep
# ave_thr_by_deg_df = fig_df[['diag_deg', 'thr']]
# ave_thr_by_deg_df.set_index('diag_deg', inplace=True)
# err_thr_by_deg_df = error_df[['diag_deg', 'thr']]
# err_thr_by_deg_df.set_index('diag_deg', inplace=True)
# print(f'ave_thr_by_deg_df:\n{ave_thr_by_deg_df}')
#
# fig_title = f'Experiment average thresholds - Ricco_v5 (n={len(participant_list)})'
# save_name = 'ricco_v5_deg_v_thr.png'
# plot_runs_ave_w_errors(fig_df=ave_thr_by_deg_df, error_df=err_thr_by_deg_df,
#                        jitter=False, error_caps=True, alt_colours=False,
#                        legend_names=None,
#                        even_spaced_x=False,
#                        fixed_y_range=False,
#                        x_tick_vals=len_deg_list,
#                        x_tick_labels=[round(i, 2) for i in len_deg_list],
#                        x_axis_label='Length: (degrees)',
#                        y_axis_label='Threshold',
#                        log_log_axes=False,
#                        neg1_slope=False,
#                        fig_title=fig_title, save_name=save_name,
#                        save_path=exp_path, verbose=True)
# plt.show()
#
#
# # # fig 2: log len, log thr
# log_deg_log_contrast_df = fig_df[['diag_deg', 'delta_I']]
# log_deg_log_contrast_df.set_index('diag_deg', inplace=True)
# err_log_deg_log_contrast_df = error_df[['diag_deg', 'delta_I']]
# err_log_deg_log_contrast_df.set_index('diag_deg', inplace=True)
# fig_title = f'Experiment average log(degrees), log(∆I) thresholds - Ricco_v5 (n={len(participant_list)})'
# save_name = 'ricco_v5_log_deg_log_contrast.png'
# plot_runs_ave_w_errors(fig_df=log_deg_log_contrast_df, error_df=err_log_deg_log_contrast_df,
#                        jitter=False, error_caps=True, alt_colours=False,
#                        legend_names=None,
#                        even_spaced_x=False,
#                        fixed_y_range=False,
#                        x_tick_vals=None,
#                        x_tick_labels=None,
#                        x_axis_label='log(degrees)',
#                        y_axis_label='Contrast: log(∆I)',
#                        log_log_axes=True,
#                        neg1_slope=True,
#                        slope_ycol_name='delta_I',
#                        slope_xcol_idx_depth=1,
#                        fig_title=fig_title, save_name=save_name,
#                        save_path=exp_path, verbose=True)
# plt.show()
#
# print('*** finished experiment average plots ***')


print('\nExp3_Ricco_v5_analysis_pipe finished\n')
