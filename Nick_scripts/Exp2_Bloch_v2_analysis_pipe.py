import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# from exp1a_psignifit_analysis import a_data_extraction, b3_plot_staircase, c_plots, \
#     d_average_participant, e_average_exp_data, make_average_plots
from rad_flow_psignifit_analysis import b3_plot_stair_sep0, c_plots, d_average_participant
from rad_flow_psignifit_analysis import make_average_plots, e_average_exp_data, plot_runs_ave_w_errors

from psignifit_tools import get_psignifit_threshold_df

# # loop through run folders with first 4 scripts (a, get_psignifit_threshold_df, b3, c)
# # then run script d to get master lists and averages
exp_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/Exp2_Bloch_NM_v2'
participant_list = ['Nick_60hz']  # , 'bb', 'cc', 'dd', 'ee']
n_runs = 1

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

        # for first run, some files are saved just as name not name1
        # run_data_path = f'{save_path}{os.sep}ISI_-1_probeDur2/{p_name}.csv'
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

        # save sorted csv
        # run_data_df.to_csv(run_data_path, index=False)
        print(f"run_data_df: {run_data_df.columns}\n{run_data_df}")

        # extract values from dataframe
        isi_list = run_data_df['ISI'].unique()
        stair_list = run_data_df['stair'].unique()
        sep_list = [0]*len(stair_list)
        print(f'isi_list: {isi_list}')
        print(f'stair_list: {stair_list}')
        print(f'sep_list: {sep_list}')
        # cosl to add...
        stair_names_list = run_data_df['stair_name'].unique()
        cond_types = run_data_df['cond_type'].unique()
        cond_type_list = list(np.tile(cond_types, len(isi_list)))
        # dur_ms_list = run_data_df['dur_ms'].unique()
        print(f'stair_names_list: {stair_names_list}')
        print(f'cond_type_list: {cond_type_list}')
        cols_to_add_dict = {'separation': sep_list,
                            'stair_names': stair_names_list,
                            'cond': cond_type_list}

        # todo: work out how to get it to loop through and get thr for all conds.

        '''get psignifit thresholds df - use stairs as sep levels rather than using groups'''
        thr_df = get_psignifit_threshold_df(root_path=root_path,
                                            p_run_name=run_dir,
                                            csv_name=run_data_df,
                                            n_bins=10, q_bins=True,
                                            sep_col='stair_name',
                                            isi_list=isi_list,
                                            sep_list=stair_names_list,
                                            cols_to_add_dict=cols_to_add_dict,
                                            verbose=True)

        # move ISI-99 to end of list, then reorder dataframe columns
        thr_df_cols = thr_df.columns.tolist()
        # if 'ISI_-99' in thr_df_cols:
        #     thr_df_cols.append(thr_df_cols.pop(thr_df_cols.index('ISI_-99')))
        # elif 'ISI_-99.0' in thr_df_cols:
        #     thr_df_cols.append(thr_df_cols.pop(thr_df_cols.index('ISI_-99.0')))
        # else:
        #     raise ValueError(f"can't find ISI -99 condition in {thr_df_cols}")
        # thr_df = thr_df[thr_df_cols]
        print(f'thr_df: {type(thr_df)}\n{thr_df}')
#
#
#         '''b3'''
#         run_data_path = f'{save_path}{os.sep}{p_name}_output.csv'
#         b3_plot_stair_sep0(run_data_path, show_plots=True)
#
#         '''c'''
#         print('*** making threshold plot ***')
#
#         # c_plots(save_path=save_path, isi_name_list=isi_name_list, show_plots=True)
#         fps = run_data_df['3_fps'].iloc[0]
#         one_frame = 1000/fps
#         probe_dur = round(2*one_frame, 3)
#         print(f'probe_dur: {probe_dur} at {fps} fps')
#
#         thr_list = thr_df.iloc[0][1:].tolist()
#         print(f'thr_df:\n{thr_df}')
#         print(f'thr_list: {thr_list}')
#
#         isi_vals_list = [float(i[4:]) for i in thr_df_cols[1:]]
#         isi_vals_list = [110 if i == -99.0 else i for i in isi_vals_list]
#         isi_name_list = ['1pr' if i == 110 else f'ISI{i}' for i in isi_vals_list]
#         print(f'isi_vals_list: {isi_vals_list}')
#         print(f'isi_name_list: {isi_name_list}')
#
#         fig, ax = plt.subplots(figsize=(10, 6))
#         sns.lineplot(x=isi_vals_list, y=thr_list, marker='o')
#         ax.set_xticks(isi_vals_list)
#         ax.set_xticklabels(isi_name_list)
#         ax.set_xlabel('Inter-stimulus Interval')
#         ax.set_ylabel('Probe Luminance')
#         plt.title(f'Bloch: {probe_dur}ms probes with varying ISI')
#         plt.savefig(f'{save_path}{os.sep}bloch_thr.png')
#         plt.show()
#         print('*** finished threshold plot ***')
#
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
#     if n_trimmed == None:
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
#     fig_df_idx = fig_df.index.tolist()
#     fig_df_idx.append(fig_df_idx.pop(0))
#     print(f'fig_fd_idx: {fig_df_idx}')
#     fig_df = fig_df.reindex(fig_df_idx)
#     fig_df = fig_df.set_index('cond')
#
#
#     error_df = pd.read_csv(err_path)
#     error_df.columns = ['cond', 'thr']
#     error_df = error_df.reindex(fig_df_idx)
#     print(f'fig_df:\n{fig_df}')
#     print(f'error_df:\n{error_df}')
#
#     isi_vals_list = fig_df.index.tolist()
#     isi_vals_list = [float(i[4:]) for i in isi_vals_list]
#     isi_vals_list = [110 if i == -99.0 else int(i) for i in isi_vals_list]
#     isi_name_list = ['1pr' if i == 110 else i for i in isi_vals_list]
#     print(f'isi_vals_list: {isi_vals_list}')
#     print(f'isi_name_list: {isi_name_list}')
#
#     fig_title = 'Participant average thresholds - Bloch'
#     save_name = 'ave_thr_all_runs.png'
#     plot_runs_ave_w_errors(fig_df=fig_df, error_df=error_df,
#                            jitter=True, error_caps=True, alt_colours=False,
#                            legend_names=None,
#                            x_tick_vals=isi_vals_list,
#                            x_tick_labels=isi_name_list,
#                            even_spaced_x=False, fixed_y_range=False,
#                            x_axis_label='ISI',
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
