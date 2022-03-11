import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# from exp1a_psignifit_analysis import a_data_extraction, b3_plot_staircase, c_plots, \
#     d_average_participant, e_average_exp_data, make_average_plots
from rad_flow_psignifit_analysis import b3_plot_staircase, b3_plot_stair_sep0, c_plots, d_average_participant
from rad_flow_psignifit_analysis import make_average_plots, e_average_exp_data, plot_runs_ave_w_errors

from psignifit_tools import get_psignifit_threshold_df

# # loop through run folders with first 4 scripts (a, get_psignifit_threshold_df, b3, c)
# # then run script d to get master lists and averages
exp_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/Exp3_Ricco_NM_v2'
# participant_list = ['Nick_test']  # , 'bb', 'cc', 'dd', 'ee']
participant_list = ['Nick']  # , 'bb', 'cc', 'dd', 'ee']

p_idx_plus = 1

for p_idx, participant_name in enumerate(participant_list):
    root_path = f'{exp_path}/{participant_name}'
    # run_folder_names = [f'{participant_name}_1']
    run_folder_names = [f'{participant_name}_1', f'{participant_name}_2',
                        f'{participant_name}_3']  # , f'{participant_name}_4',
                        # f'{participant_name}_5', f'{participant_name}_6']


    for run_idx, run_dir in enumerate(run_folder_names):

        print(f'\nrunning analysis for {participant_name}, {run_dir}, {participant_name}{run_idx+1}\n')
        save_path = f'{root_path}{os.sep}{run_dir}'

        # don't delete this (participant_name = participant_name),
        # needed to ensure names go name1, name2, name3 not name1, name12, name123
        p_name = participant_name

        # '''a'''
        p_name = f'{participant_name}_{run_idx+1}'

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

        # # # save sorted csv
        # run_data_df.to_csv(run_data_path, index=False)
        #
        # run_data_df = pd.read_csv(run_data_path, usecols=
        #                           ['trial_number', 'stair', 'stair_name', 'step',
        #                            'separation', 'cond_type', 'ISI', 'corner',
        #                            'probeLum', 'delta_lum', 'trial_response', '3_fps'])
        # print(f"run_data_df: {run_data_df.columns.to_list()}\n{run_data_df}\n")
        #
        # # extract values from dataframe
        # separation_values = run_data_df['separation'].unique()
        # stair_list = run_data_df['stair'].unique()
        # isi_list = run_data_df['ISI'].unique()
        # print(f'separation_values: {separation_values}')
        # print(f'isi_list: {isi_list}')
        # print(f'stair_list: {stair_list}')
        #
        # cond_types = run_data_df['cond_type'].unique()
        # sep_vals_list = list(np.repeat(separation_values, 3))
        # cond_type_list = list(np.tile(cond_types, len(separation_values)))
        # print(f'cond_types: {cond_types}')
        # print(f'sep_vals_list: {sep_vals_list}')
        # print(f'cond_type_list: {cond_type_list}')
        #
        # stair_names_list = run_data_df['stair_name'].unique()
        # print(f'stair_names_list: {stair_names_list}')
        # cols_to_add_dict = {'separation': sep_vals_list,
        #                     'cond': cond_type_list}
        #
        # thr_save_name = 'psignifit_thresholds'
        # thr_df = get_psignifit_threshold_df(root_path=root_path,
        #                                     p_run_name=run_dir,
        #                                     csv_name=run_data_df,
        #                                     n_bins=10, q_bins=True,
        #                                     sep_col='stair_name',
        #                                     isi_list=isi_list,
        #                                     sep_list=stair_names_list,
        #                                     cols_to_add_dict=cols_to_add_dict,
        #                                     save_name=thr_save_name,
        #                                     verbose=True)
        # print(f'thr_df: {type(thr_df)}\n{thr_df}')
        #
        #
        # '''b3'''
        # run_data_path = f'{save_path}{os.sep}{p_name}_output.csv'
        # run_data_df = pd.read_csv(run_data_path, usecols=
        #                           ['trial_number', 'stair', 'stair_name', 'step',
        #                            'separation', 'cond_type', 'ISI', 'corner',
        #                            'probeLum', 'delta_lum', 'trial_response', '3_fps'])
        # print(f'run_data_df:\n{run_data_df}')
        #
        # # Ricco doesn't currently work with b3_plot_staircase or c_plots
        # # b3_plot_staircase(run_data_path, show_plots=True)
        # # # c_plots(save_path=save_path, isi_name_list=isi_name_list, show_plots=True)
        #
        # print('*** making plot with x=ordinal, y=thr ***')
        # thr_df_path = f'{save_path}{os.sep}test1.csv'
        thr_df_path = f'{save_path}{os.sep}psignifit_thresholds.csv'
        # thr_df_path = f'{save_path}{os.sep}{thr_save_name}.csv'
        thr_df = pd.read_csv(thr_df_path)
        print(f'thr_df:\n{thr_df}')
        #
        # sep_list = thr_df['separation'].unique()
        # sep_vals_list = [i for i in sep_list]
        # sep_name_list = ['1pr' if i == -1 else f'sep{i}' for i in sep_list]
        # print(f'sep_vals_list: {sep_vals_list}')
        # print(f'sep_name_list: {sep_name_list}')
        #
        # fig, ax = plt.subplots(figsize=(10, 6))
        # sns.lineplot(data=thr_df, x='separation', y='ISI_0', hue='cond', marker='o')
        # ax.set_xticks(sep_vals_list)
        # ax.set_xticklabels(sep_name_list)
        # ax.set_xlabel('Probe cond (separation)')
        # ax.set_ylabel('Probe Luminance')
        # plt.title(f'Ricco_v2: probe cond vs thr')
        # # plt.savefig(f'{save_path}{os.sep}ricco_v2_cond_v_thr.png')
        # plt.show()
        # print('*** finished plot with x=ordinal, y=thr ***\n')

        print('*** making plot with x=log(area), y=log(∆thr) ***')
        print(f'thr_df:\n{thr_df}')

        # convert separation into area
        area_dict = {-1: {'radius': 2.15, 'area': 14.522012041218817},
                     0: {'radius': 2.5, 'area': 19.634954084936208},
                     1: {'radius': 2.8, 'area': 24.630086404143974},
                     2: {'radius': 3.4, 'area': 36.316811075498},
                     3: {'radius': 4.1, 'area': 52.81017250684442},
                     6: {'radius': 6.1, 'area': 116.89866264007618},
                     18: {'radius': 14.6, 'area': 669.6618900392003}}
        sep_col = thr_df['separation'].to_list()
        area_col = [area_dict[i]['area'] for i in sep_col]
        thr_df.insert(3, 'area', area_col)

        thr_col = thr_df['ISI_0'].to_list()
        bgLum = 21.2
        delta_thr_col = [(i-bgLum)/bgLum for i in thr_col]
        thr_df.insert(4, 'delta_thr', delta_thr_col)
        print(f'thr_df:\n{thr_df}')


        fig, ax = plt.subplots(figsize=(6, 6))
        sns.lineplot(data=thr_df, x='area', y='delta_thr', hue='cond', marker='o', ax=ax)

        # ax.set_xticks(sep_vals_list)
        # ax.set_xticklabels(sep_name_list)
        ax.set_xlabel('log(area)')
        ax.set_ylabel('log(∆ threshold)')
        ax.set(xlim=(10, 10000), ylim=(.01, 10))

        ax.set(xscale="log", yscale="log")

        # add guideline with slope of -1 which crosses through the circles 1probe delta_thr value.
        circle_1pr_delta = thr_df.loc[thr_df['stair_name'] == '-1_circles', 'delta_thr'].item()
        circle_1pr_area = 14.52
        # ax.axline((circle_1pr_area, circle_1pr_delta), slope=-1, color='red', linestyle='dashed', label='-1 slope')
        ax.plot([circle_1pr_area, circle_1pr_area*100], [circle_1pr_delta, circle_1pr_delta/100], c='r',
                label='-1 slope', linestyle='dashed')
        ax.legend()
        plt.title(f'Ricco_v2: log(area) v log(thr)')
        plt.savefig(f'{save_path}{os.sep}ricco_v2_logArea_v_logThr.png')
        plt.show()
        print('*** finished plot with with x=log(area), y=log(∆thr) ***')


        # # just circle data, try a regression plot or something
        # circle_df = thr_df[thr_df['cond']=='circles']
        #
        # print(f'circle_df:\n{circle_df}')
        # fig, ax = plt.subplots(figsize=(6, 6))
        # ax.set(xscale="log", yscale="log")
        # # sns.scatterplot(data=circle_df, x='area', y='delta_thr', ax=ax)
        # sns.regplot(data=circle_df, x='area', y='delta_thr', ax=ax,
        #             logx=True, fit_reg=True, truncate=True)
        # # ax.set_xlim(auto=True)
        # # ax.set(xscale="log", yscale="log")
        # ax.set_xlim(10, )
        # ax.set_ylim(.01, )
        # # ax.set(xlim=(1, ), ylim=(.01, ))
        #
        # plt.show()



#     '''d'''
#     trim_n = None
#     if len(run_folder_names) == 12:
#         trim_n = 1
#     d_average_participant(root_path=root_path, run_dir_names_list=run_folder_names,
#                           trim_n=trim_n, error_type='SE')
#
#
#     # making average plot
#     all_df_path = f'{root_path}/MASTER_TM1_thresholds.csv'
#     p_ave_path = f'{root_path}/MASTER_ave_TM_thresh.csv'
#     err_path = f'{root_path}/MASTER_ave_TM_thr_error_SE.csv'
#     n_trimmed = trim_n
#     if n_trimmed is None:
#         all_df_path = f'{root_path}/MASTER_psignifit_thresholds.csv'
#         p_ave_path = f'{root_path}/MASTER_ave_thresh.csv'
#         err_path = f'{root_path}/MASTER_ave_thr_error_SE.csv'
#
#     exp_ave = False
#
#     # load data and change order to put 1pr last
#     print('*** making average plot ***')
#     fig_df = pd.read_csv(p_ave_path)
#     fig_df.columns = ['cond', 'thr']
#     sep_vals_list = fig_df.cond.tolist()
#
#     fig_df = fig_df.set_index('cond')
#
#
#     error_df = pd.read_csv(err_path)
#     error_df.columns = ['cond', 'thr']
#     print(f'fig_df:\n{fig_df}')
#     print(f'error_df:\n{error_df}')
#
#     sep_vals_list = [-1 if i == -99 else i for i in sep_vals_list]
#     sep_name_list = ['1pr' if i == -1 else i for i in sep_vals_list]
#     print(f'sep_vals_list: {sep_vals_list}')
#     print(f'sep_name_list: {sep_name_list}')
#
#     fig_title = 'Participant average thresholds - Ricco'
#     save_name = 'ave_thr_all_runs.png'
#     plot_runs_ave_w_errors(fig_df=fig_df, error_df=error_df,
#                            jitter=True, error_caps=True, alt_colours=False,
#                            legend_names=None,
#                            x_tick_vals=sep_vals_list,
#                            x_tick_labels=sep_name_list,
#                            even_spaced_x=False, fixed_y_range=False,
#                            x_axis_label='Probe length',
#                            fig_title=fig_title, save_name=save_name,
#                            save_path=root_path, verbose=True)
#     plt.show()
#     print('*** finished average plot ***')
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

print('\nExp2_Bloch_analysis_pipe finished\n')
