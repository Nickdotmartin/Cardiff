import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from exp1a_psignifit_analysis import a_data_extraction, b3_plot_staircase, c_plots, \
    d_average_participant, e_average_exp_data, make_average_plots
from psignifit_tools import get_psignifit_threshold_df

'''
This script if for checking for any differences between thr upper visual field and lower visual field.
It will use the already analysied RUNDATA_sorted.xlsx to do this.
Loop through the participant run folders and append each RUNDATA-sorted.xlsx, with addition 'run' column.
Save to P_all_runs_master_output.csv.

Then run psignifit on this
'''

# # loop through run folders with first 4 scripts (a, get_psignifit_threshold_df, b3, c)
# # then run script d to get master lists and averages
exp_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/exp1a_data'
participant_list = ['aa', 'bb', 'cc', 'dd', 'ee']

p_idx_plus = 1

for p_idx, participant_name in enumerate(participant_list):
    root_path = f'{exp_path}/{participant_name}'

    run_folder_names = [f'{participant_name}_1', f'{participant_name}_2',
                        f'{participant_name}_3', f'{participant_name}_4',
                        f'{participant_name}_5', f'{participant_name}_6']

    group_list = [1, 2]

    # add RUNDATA-sorted to all_data
    all_data = []

    # for run_idx, run_dir in enumerate(run_folder_names):
    #
    #     print(f'\ncompiling analysis for {participant_name}, {run_dir}, {participant_name}_{run_idx+1}\n')
    #     save_path = f'{root_path}{os.sep}{run_dir}'
    #
    #     # don't delete this (participant_name = participant_name),
    #     # needed to ensure names go name1, name2, name3 not name1, name12, name123
    #     p_name = participant_name
    #
    #     # '''a'''
    #     p_name = f'{participant_name}_{run_idx+1}_output'
    #     # p_name = f'{participant_name}{run_idx+1}'
    #     isi_list = [-1, 0, 2, 4, 6, 9, 12, 24]
    #
    #
    #     run_data_path = f'{save_path}{os.sep}RUNDATA-sorted.xlsx'
    #
    #     run_data_df = pd.read_excel(run_data_path, engine='openpyxl',
    #                                 usecols=['ISI',
    #                                          'stair',
    #                                          'separation', 'group',
    #                                          'probeLum', 'trial_response', 'corner'])
    #     print(f"run_data_df:\n{run_data_df}")
    #
    #     # add isi column for multi-indexing
    #     run_data_df.insert(0, 'run', int(run_idx+1))
    #     # if verbose:
    #     print(f'run_data_df:\n{run_data_df.head()}')
    #
    #     # get column names to use on all_data_df
    #     column_names = list(run_data_df)
    #
    #     # add to all_data
    #     all_data.append(run_data_df)
    #
    # # create all_data_df - reshape to 2d
    # all_data_shape = np.shape(all_data)
    # sheets, rows, columns = np.shape(all_data)
    # all_data = np.reshape(all_data, newshape=(sheets * rows, columns))
    # # if verbose:
    # print(f'all_data reshaped from {all_data_shape} to {np.shape(all_data)}')
    # all_data_df = pd.DataFrame(all_data, columns=column_names)
    #
    # # if verbose:
    # print(f"all_data_df:\n{all_data_df}")
    #
    # # if save_all_data:
    # save_name = 'P_all_runs_master_output.csv'
    # save_csv_path = os.path.join(root_path, save_name)
    # # if verbose:
    # print(f"\nsaving all_data_df to save_csv_path: {save_csv_path}")
    # all_data_df.to_csv(save_csv_path, index=False)
    #
    # all_data_df = pd.read_csv(os.path.join(root_path, 'P_all_runs_master_output.csv' ))
    #
    # UVF_df = all_data_df[all_data_df['corner'] < 200]
    # LVF_df = all_data_df[all_data_df['corner'] > 200]
    #
    # vis_field_list = [UVF_df, LVF_df]
    # vis_field_names = ['UVF', 'LVF']
    #
    # both_vfs = []
    #
    # for idx, vis_field_df in enumerate(vis_field_list):
    #
    #     vis_field_name = vis_field_names[idx]
    #
    #     print(f'Running psignifit for {vis_field_name}\n{vis_field_df}')
    #
    #
    #     # stair_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    #     sep_list = [0, 1, 2, 3, 6, 18, 99]
    #     isi_list = [-1, 0, 2, 4, 6, 9, 12, 24]
    #     cols_to_add_dict = {
    #         # 'group': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
    #         #                 'separation': [18, 18, 6, 6, 3, 3, 2, 2, 1, 1, 0, 0, 20, 20],
    #                         'vis_field': [vis_field_name, vis_field_name, vis_field_name, vis_field_name,
    #                                       vis_field_name, vis_field_name, vis_field_name]}
    #
    #     '''get psignifit thresholds df - use stairs as sep levels rather than using groups'''
    #     thr_df = get_psignifit_threshold_df(root_path=exp_path,
    #                                         p_run_name=participant_name ,
    #                                         csv_name=vis_field_df,
    #                                         n_bins=10, q_bins=True,
    #                                         sep_col='separation',
    #                                         isi_list=isi_list,
    #                                         sep_list=sep_list,
    #                                         cols_to_add_dict=cols_to_add_dict,
    #                                         save_name=f'psignifit_{vis_field_name}',
    #                                         verbose=True)
    #     print(f'psignifit_{vis_field_name}:\n{thr_df}')
    #     column_names = list(thr_df)
    #
    #     both_vfs.append(thr_df)
    #
    # # create all_data_df - reshape to 2d
    # both_vfs_shape = np.shape(both_vfs)
    # sheets, rows, columns = np.shape(both_vfs)
    # both_vfs = np.reshape(both_vfs, newshape=(sheets * rows, columns))
    # print(f'both_vfs reshaped from {both_vfs_shape} to {np.shape(both_vfs)}')
    # both_vfs_df = pd.DataFrame(both_vfs, columns=column_names)
    # print(f"both_vfs_df:\n{both_vfs_df}")
    #
    # save_name = 'psignifit_both_vfs.csv'
    # save_csv_path = os.path.join(root_path, save_name)
    # print(f"\nsaving all_data_df to save_csv_path:\n{save_csv_path}")
    # both_vfs_df.to_csv(save_csv_path, index=False)

    psig_both_vf_df = pd.read_csv(os.path.join(root_path, 'psignifit_both_vfs.csv'))
    # sep_values = psig_both_vf_df['separation'].to_list()
    # sep_values = [20 if i == 99 else i for i in sep_values]
    # sep_labels = ['1pr' if i == 20 else i for i in sep_values]
    # psig_both_vf_df['separation'] = sep_values
    cond_vals = [0, 1, 2, 3, 6, 18, 20,
                 -.1, -1, -2, -3, -6, -18, -20]
    cond_labels = [0, 1, 2, 3, 6, 18, '1pr',
                   -.1, -1, -2, -3, -6, -18, '-1pr']
    psig_both_vf_df.insert(0, 'cond', cond_vals)
    psig_both_vf_df.sort_values(by='cond', inplace=True)
    psig_both_vf_df.set_index('cond', drop=True, inplace=True)
    print(f"psig_both_vf_df:\n{psig_both_vf_df}")
    psig_both_vf_df = psig_both_vf_df.drop(['separation', 'vis_field'], axis=1)
    print(f"psig_both_vf_df:\n{psig_both_vf_df}")

    # make plot to compare UVF and LVF
    fig_title = f'{participant_name} compare UVF & LVF'
    legend_title = 'ISI'
    x_tick_vals = cond_vals
    x_tick_labels = cond_labels
    x_axis = 'Sep in diag pixels. Neg=LVF, Pos=UVF'
    y_axis = 'Threshold'
    log_x = False
    log_y = False
    save_as = os.path.join(root_path, 'compare_vfs.png')

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=psig_both_vf_df,
                 markers=True, dashes=False, ax=ax)
    if fig_title is not None:
        plt.title(fig_title)
    if legend_title is not None:
        plt.legend(title=legend_title)
    if x_tick_vals is not None:
        ax.set_xticks(x_tick_vals)
    if x_tick_labels is not None:
        ax.set_xticklabels(x_tick_labels)
        if -18 in x_tick_labels:
            # add dotted line at zero
            ax.axvline(x=0, linestyle="-.", color='lightgrey')
    if log_x:
        ax.set(xscale="log")
        x_axis = f'log {x_axis}'
    if log_y:
        ax.set(yscale="log")
        y_axis = f'log {y_axis}'
    if x_axis is not None:
        ax.set_xlabel(x_axis)
    if y_axis is not None:
        ax.set_ylabel(y_axis)
    if save_as is not None:
        plt.savefig(save_as)
    plt.show()



#
#
#         '''b3'''
#         run_data_path = f'{save_path}{os.sep}RUNDATA-sorted.xlsx'
#         b3_plot_staircase(run_data_path, show_plots=True)
#
#         '''c'''
#         c_plots(save_path=save_path, show_plots=True)
#
#     '''d'''
#     d_average_participant(root_path=root_path, run_dir_names_list=run_folder_names,
#                           trim_n=1, error_type='SE')
#
#     all_df_path = f'{root_path}/MASTER_TM1_thresholds.csv'
#     p_ave_path = f'{root_path}/MASTER_ave_TM_thresh.csv'
#     err_path = f'{root_path}/MASTER_ave_TM_thr_error_SE.csv'
#     n_trimmed = 1
#     exp_ave = False
#
#     make_average_plots(all_df_path=all_df_path,
#                        ave_df_path=p_ave_path,
#                        error_bars_path=err_path,
#                        error_type='SE',
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

print('\nexp1a_analysis_pipe_UVF_LVF finished\n')
