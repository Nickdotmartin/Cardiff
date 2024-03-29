import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# from exp1a_psignifit_analysis import a_data_extraction, b3_plot_staircase, c_plots, \
#     d_average_participant, e_average_exp_data, make_average_plots
from rad_flow_psignifit_analysis import b3_plot_staircase, b3_plot_stair_sep0, c_plots, d_average_participant
from rad_flow_psignifit_analysis import make_average_plots, e_average_exp_data, \
    plot_runs_ave_w_errors, plot_w_errors_either_x_axis, run_thr_plot, simple_log_log_plot
from psignifit_tools import get_psignifit_threshold_df

# # loop through run folders with first 4 scripts (a, get_psignifit_threshold_df, b3, c)
# # then run script d to get master lists and averages
# exp_path = '/Users/nickmartin/Library/CloudStorage/OneDrive-CardiffUniversity/PycharmProjects/Cardiff/Exp3_Ricco_NM_v3'
exp_path = r'C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects/Cardiff/Exp3_Ricco_NM_v3'
convert_path1 = os.path.normpath(exp_path)
print(f"convert_path1: {convert_path1}")
exp_path = convert_path1
participant_list = ['Nick_60hz_2ms_dur_test']  # 'Nick''Nick_120ms_dur', 'bb', 'cc', 'dd', 'ee']
n_runs = 1

p_idx_plus = 1

for p_idx, participant_name in enumerate(participant_list):
    # root_path = f'{exp_path}/{participant_name}'
    root_path = os.path.join(exp_path, participant_name)

    run_folder_names = [f'{participant_name}_{i+1}' for i in list(range(n_runs))]
    print(f'run_folder_names: {run_folder_names}')

    for run_idx, run_dir in enumerate(run_folder_names):

        save_path = f'{root_path}{os.sep}{run_dir}'

        # don't delete this (participant_name = participant_name),
        # needed to ensure names go name1, name2, name3 not name1, name12, name123
        p_name = participant_name

        # '''a'''
        p_name = f'{participant_name}_{run_idx+1}'
        print(f'\nrunning analysis for {participant_name}, {run_dir}, {p_name}\n')

        # # for first run, some files are saved just as name not name1
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

        # # save sorted csv
        run_data_df.to_csv(run_data_path, index=False)

        run_data_df = pd.read_csv(run_data_path, usecols=
                                  ['trial_number', 'stair', 'stair_name', 'step',
                                   'separation', 'cond_type', 'ISI', 'corner',
                                   'probeLum',
                                   # 'weber_lum',
                                   'trial_response', '3_fps'])
        print(f"run_data_df: {run_data_df.columns.to_list()}\n{run_data_df}\n")

        # extract values from dataframe
        separation_values = run_data_df['separation'].unique()
        stair_list = run_data_df['stair'].unique()
        isi_list = run_data_df['ISI'].unique()
        print(f'separation_values: {separation_values}')
        print(f'isi_list: {isi_list}')
        print(f'stair_list: {stair_list}')

        cond_types = run_data_df['cond_type'].unique()
        sep_vals_list = list(np.repeat(separation_values, len(cond_types)))
        cond_type_list = list(np.tile(cond_types, len(separation_values)))
        print(f'cond_types: {cond_types}')
        print(f'sep_vals_list: {sep_vals_list}')
        print(f'cond_type_list: {cond_type_list}')

        stair_names_list = run_data_df['stair_name'].unique()
        print(f'stair_names_list: {stair_names_list}')
        cols_to_add_dict = {'stair_names': stair_names_list,
                            'separation': sep_vals_list,
                            'cond': cond_type_list}

        thr_save_name = 'psignifit_thresholds'
        thr_df = get_psignifit_threshold_df(root_path=root_path,
                                            p_run_name=run_dir,
                                            csv_name=run_data_df,
                                            n_bins=10, q_bins=True,
                                            sep_col='stair_name',
                                            isi_list=isi_list,
                                            sep_list=stair_names_list,
                                            cols_to_add_dict=cols_to_add_dict,
                                            save_name=thr_save_name,
                                            verbose=True)
        print(f'thr_df: {type(thr_df)}\n{thr_df}')


        '''b3'''
        run_data_path = f'{save_path}{os.sep}{p_name}_output.csv'
        run_data_df = pd.read_csv(run_data_path,
                                  # usecols=
                                  # ['trial_number', 'stair', 'stair_name', 'step',
                                  #  'separation', 'cond_type', 'ISI', 'corner',
                                  #  'probeLum', 'weber_lum', 'trial_response', '3_fps']
                                  )
        print(f'run_data_df:\n{run_data_df}')

        # Ricco doesn't currently work with b3_plot_staircase or c_plots
        # b3_plot_staircase(run_data_path, show_plots=True)
        # # c_plots(save_path=save_path, isi_name_list=isi_name_list, show_plots=True)

        thr_df_path = f'{save_path}{os.sep}psignifit_thresholds.csv'
        # thr_df_path = f'{save_path}{os.sep}{thr_save_name}.csv'
        thr_df = pd.read_csv(thr_df_path)

        if 'delta_thr' in list(thr_df.columns):
            thr_df.drop('delta_thr', axis=1, inplace=True)

        print(f'thr_df:\n{thr_df}\n')

        sep_list = thr_df['separation'].unique()
        sep_vals_list = [i for i in sep_list]
        sep_name_list = ['1pr' if i == -1 else f'sep{i}' for i in sep_list]
        print(f'sep_vals_list: {sep_vals_list}')
        print(f'sep_name_list: {sep_name_list}\n')

        # basic plot with regular axes
        run_thr_plot(thr_df, x_col='separation', y_col='ISI_0', hue_col='cond',
                     x_ticks_vals=sep_vals_list, x_tick_names=sep_name_list,
                     x_axis_label='Probe cond (separation)',
                     y_axis_label='Probe Luminance',
                     fig_title='Ricco_v3: probe cond vs thr',
                     save_as=f'{save_path}{os.sep}ricco_v3_cond_v_thr.png')
        plt.show()

        print(f'thr_df:\n{thr_df}')

        # convert separation into area (units are pixels, mm and degrees are diagonal pixels)
        area_dict = {-1: {'pixels': {'radius': 2.15, 'area': 14.522012},
                          'mm': {'radius': 0.898865301409574, 'area': 2.5382775},
                          'degrees': {'radius': 0.0936293639654774, 'area': 0.0275406}},
                     0: {'pixels': {'radius': 2.5, 'area': 19.634954},
                         'mm': {'radius': 1.04519221, 'area': 3.4319599},
                         'degrees': {'radius': 0.10887135, 'area': 0.0372372}},
                     1: {'pixels': {'radius': 2.8, 'area': 24.630086},
                         'mm': {'radius': 1.17061528, 'area': 4.3050505},
                         'degrees': {'radius': 0.12193592, 'area': 0.0467104}},
                     2: {'pixels': {'radius': 3.4, 'area': 36.316811},
                         'mm': {'radius': 1.42146141, 'area': 6.3477530},
                         'degrees': {'radius': 0.14806504, 'area': 0.0688739}},
                     3: {'pixels': {'radius': 4.1, 'area': 52.810173},
                         'mm': {'radius': 1.71411523, 'area': 9.2305993},
                         'degrees': {'radius': 0.17854902, 'area': 0.1001532}},
                     6: {'pixels': {'radius': 6.1, 'area': 116.898663},
                         'mm': {'radius': 2.55026899, 'area': 20.4325163},
                         'degrees': {'radius': 0.26564610, 'area': 0.2216954}},
                     18: {'pixels': {'radius': 14.6, 'area': 669.661890},
                          'mm': {'radius': 6.10392251, 'area': 117.0490508},
                          'degrees': {'radius': 0.63580870, 'area': 1.2699973}},
                     36: {'pixels': {'radius': 27.3, 'area': 2341.397589},
                          'mm': {'radius': 11.41349894, 'area': 409.2488603},
                          'degrees': {'radius': 1.18887518, 'area': 4.4404031}}}

        # check for 'area' and 'weber_thr' col
        # old area was given in pixels
        col_names = thr_df.columns.to_list()
        if 'area_deg' not in col_names:
            sep_col = thr_df['separation'].to_list()
            area_col = [area_dict[i]['degrees']['area'] for i in sep_col]
            thr_df.insert(3, 'area_deg', area_col)

        if 'weber_thr' not in col_names:
            thr_col = thr_df['ISI_0'].to_list()
            bgLum = 21.2
            # delta_thr_col = [(i-bgLum)/bgLum for i in thr_col]
            weber_thr_col = [(i-bgLum)/i for i in thr_col]
            thr_df.insert(4, 'weber_thr', weber_thr_col)

        if 'stair_name' in col_names:
            thr_df.drop('stair_name', axis=1, inplace=True)

        print(f'thr_df:\n{thr_df}')
        thr_df.to_csv(thr_df_path, index=False)


        # plot with log-log axes
        simple_log_log_plot(thr_df, x_col='area_deg', y_col='weber_thr', hue_col='cond',
                         x_ticks_vals=None, x_tick_names=None,
                         x_axis_label='log(area degrees$^2$) - circles condition',
                         y_axis_label='log(∆I/I)',
                         fig_title='Ricco_v3: log(area) v log(∆I/I)',
                         save_as=f'{save_path}{os.sep}ricco_v3_log_area_log_weber.png')
        plt.show()

    #
    # '''d'''
    # trim_n = None
    # if len(run_folder_names) == 12:
    #     trim_n = 1
    #
    # # run_folder_names = ['Nick_1', 'Nick_2', 'Nick_3']
    # d_average_participant(root_path=root_path, run_dir_names_list=run_folder_names,
    #                       trim_n=trim_n, error_type='SE')

    # # making average plot
    # all_df_path = os.path.join(root_path, 'MASTER_TM1_thresholds.csv')
    # p_ave_path = os.path.join(root_path, 'MASTER_ave_TM_thresh.csv')
    # err_path = os.path.join(root_path, 'MASTER_ave_TM_thr_error_SE.csv')
    # n_trimmed = trim_n
    # if n_trimmed is None:
    #     all_df_path = os.path.join(root_path, 'MASTER_psignifit_thresholds.csv')
    #     p_ave_path = os.path.join(root_path, 'MASTER_ave_thresh.csv')
    #     err_path = os.path.join(root_path, 'MASTER_ave_thr_error_SE.csv')

    # exp_ave = False
    #
    #
    # # load data and change order to put 1pr last
    # print('*** making average plot ***')
    # print(f'root_path: {root_path}')
    # # reshape dfs so that the different conds are in separate columns.
    # fig_df = pd.read_csv(p_ave_path)
    # print(f'fig_df:\n{fig_df}')
    #
    # wide_df = fig_df.pivot(index=['separation'], columns='cond', values='ISI_0')
    # print(f'wide_df:\n{wide_df}')
    #
    # x_values = wide_df.index.get_level_values('separation').to_list()
    # x_labels = ['1pr' if i == -1 else i for i in x_values]
    # print(f'x_values: {x_values}')
    #
    # error_df = pd.read_csv(err_path)
    # print(f'error_df:\n{error_df}')
    # wide_err_df = error_df.pivot(index=['separation'], columns='cond', values='ISI_0')
    #
    # fig_title = 'Participant average thresholds - Ricco_v3'
    # save_name = 'ricco_v3_sep_v_thr.png'
    # plot_runs_ave_w_errors(fig_df=wide_df, error_df=wide_err_df,
    #                        jitter=False, error_caps=True, alt_colours=False,
    #                        legend_names=None,
    #                        even_spaced_x=True,
    #                        fixed_y_range=False,
    #                        x_tick_vals=x_values,
    #                        x_tick_labels=x_labels,
    #                        x_axis_label='Separation (2probe cond)',
    #                        y_axis_label='Threshold',
    #                        log_log_axes=False,
    #                        neg1_slope=False,
    #                        fig_title=fig_title, save_name=save_name,
    #                        save_path=root_path, verbose=True)
    # plt.show()
    #
    # wide_df = fig_df.pivot(index=['area_deg', 'separation'], columns='cond', values='weber_thr')
    # print(f'wide_df:\n{wide_df}')
    #
    # area_values = wide_df.index.get_level_values('area_deg').to_list()
    # print(f'area_values: {area_values}')
    #
    # error_df = pd.read_csv(err_path)
    # # replace NaN with zero
    # error_df['weber_thr'] = error_df['weber_thr'].fillna(0)
    # print(f'error_df:\n{error_df}')
    # wide_err_df = error_df.pivot(index=['area', 'separation'], columns='cond', values='weber_thr')
    # print(f'wide_err_df:\n{wide_err_df}')
    #
    # fig_title = 'Participant average ∆I/I thresholds - Ricco_v3'
    # save_name = 'ricco_v3_log_area_log_weber.png'
    # plot_runs_ave_w_errors(fig_df=wide_df, error_df=wide_err_df,
    #                        jitter=False, error_caps=True, alt_colours=False,
    #                        legend_names=None,
    #                        even_spaced_x=False,
    #                        fixed_y_range=False,
    #                        x_tick_vals=area_values,
    #                        x_tick_labels=None,
    #                        x_axis_label='log(area degrees$^2$) - circles condition',
    #                        y_axis_label='log(∆I/I)',
    #                        log_log_axes=True,
    #                        neg1_slope=True,
    #                        fig_title=fig_title, save_name=save_name,
    #                        save_path=root_path, verbose=True)
    # plt.show()
    # print('*** finished average plot ***')

    # # make_average_plots() doesn't really work here.
    # make_average_plots(all_df_path=all_df_path,
    #                    ave_df_path=p_ave_path,
    #                    error_bars_path=err_path,
    #                    n_trimmed=n_trimmed,
    #                    exp_ave=False,
    #                    show_plots=True, verbose=True)
#
# participant_list = ['Nick', 'Nick_120ms_dur']
# print(f'exp_path: {exp_path}')
# print('\nget exp_average_data')
#
# e_average_exp_data(exp_path=exp_path, p_names_list=participant_list,
#                    error_type='SE', use_trimmed=True, verbose=True)
#
#
# all_df_path = os.path.join(exp_path, 'MASTER_exp_thr.csv')
# exp_ave_path = os.path.join(exp_path, 'MASTER_exp_ave_thr.csv')
# err_path = os.path.join(exp_path, 'MASTER_ave_thr_error_SE.csv')
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

print('\nExp3_Ricco_v3_analysis_pipe finished\n')
