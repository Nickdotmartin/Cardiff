import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from psignifit_tools import get_psignifit_threshold_df, get_psig_thr_w_hue
from rad_flow_psignifit_analysis import a_data_extraction, b3_plot_staircase
from rad_flow_psignifit_analysis import c_plots, d_average_participant
from rad_flow_psignifit_analysis import make_average_plots, e_average_exp_data
from check_home_dir import switch_path
# from exp1a_psignifit_analysis import a_data_extraction, b3_plot_staircase, c_plots, \
#     d_average_participant, e_average_exp_data, make_average_plots
from exp1a_psignifit_analysis import plt_heatmap_row_col


exp_path = r'C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\EXP4_missing_probe'
exp_path = os.path.normpath(exp_path)

coherence_type = 'Radial'
exp_path = os.path.join(exp_path, coherence_type)

participant_list = ['Simon']  # 'Nick'
analyse_from_run = 1

verbose = True
show_plots = True

n_runs = 12  # 12
# if the first folder to analyse is 1, analyse_from_run = 1.  If the first folder is 5, use 5 etc.

for p_idx, participant_name in enumerate(participant_list):

    root_path = os.path.join(exp_path, participant_name)

    # # search to automatically get run_folder_names
    dir_list = os.listdir(root_path)
    run_folder_names = []
    for i in range(n_runs):  # numbers 0 to 11
        check_dir = f'{participant_name}_{i+analyse_from_run}'   # numbers 1 to 12
        if check_dir in dir_list:
            run_folder_names.append(check_dir)

    print(f'run_folder_names: {run_folder_names}')

    '''for trimming'''
    trim_n = None
    if len(run_folder_names) == 12:
        trim_n = 2
    print(f'\ntrim_n: {trim_n}')

    # add RUNDATA-sorted to all_data
    all_data = []

    for run_idx, run_dir in enumerate(run_folder_names):

        # add run number , e.g., add five to access Nick_5 on the zeroth iteration
        r_idx_plus = run_idx + analyse_from_run

        print(f'\nrun_idx {run_idx+1}: running analysis for '
              f'{participant_name}, {run_dir}, {participant_name}_{r_idx_plus}')
        save_path = os.path.join(root_path, run_dir)

        # don't delete this (p_name = participant_name),
        # needed to ensure names go name1, name2, name3 not name1, name12, name123
        p_name = participant_name

        '''a'''
        # # I don't need data extraction as all ISIs are in same df.
        p_name = f'{participant_name}_output'  # use this one
        try:
            run_data_df = pd.read_csv(os.path.join(save_path, f'{p_name}.csv'))
        except:
            p_name = f'{participant_name}_{r_idx_plus}_output'
            run_data_df = pd.read_csv(os.path.join(save_path, f'{p_name}.csv'))

        run_data_df = pd.read_csv(os.path.join(save_path, f'{p_name}.csv'))
        run_data_df = run_data_df.sort_values(by=['stair', 'trial_number'])
        print(f"run_data_df: {run_data_df.columns.to_list()}\n{run_data_df}")

        # remove unnamed columns
        unnamed_cols = [i for i in run_data_df.columns.to_list() if 'Unnamed: ' in i]
        print(f"unnamed_cols: {unnamed_cols}")

        for col_name in unnamed_cols:
            run_data_df.drop(col_name, axis=1, inplace=True)

        # add isi column for multi-indexing
        run_data_df.insert(0, 'run', int(run_idx+1))
        # if verbose:
        print(f'run_data_df:\n{run_data_df.head()}')

        # get column names to use on all_data_df
        column_names = list(run_data_df)

        # add to all_data
        all_data.append(run_data_df)

    # create all_data_df - reshape to 2d
    all_data_shape = np.shape(all_data)
    sheets, rows, columns = np.shape(all_data)
    all_data = np.reshape(all_data, newshape=(sheets * rows, columns))
    # if verbose:
    print(f'all_data reshaped from {all_data_shape} to {np.shape(all_data)}')
    all_data_df = pd.DataFrame(all_data, columns=column_names)

    # if verbose:
    print(f"all_data_df:\n{all_data_df}")

    # if save_all_data:
    save_name = 'P_all_runs_master_output.csv'
    save_csv_path = os.path.join(root_path, save_name)
    # if verbose:
    print(f"\nsaving all_data_df to save_csv_path: {save_csv_path}")
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
    #
    # psig_both_vf_df = pd.read_csv(os.path.join(root_path, 'psignifit_both_vfs.csv'))
    # # sep_values = psig_both_vf_df['separation'].to_list()
    # # sep_values = [20 if i == 99 else i for i in sep_values]
    # # sep_labels = ['1pr' if i == 20 else i for i in sep_values]
    # # psig_both_vf_df['separation'] = sep_values
    # cond_vals = [0, 1, 2, 3, 6, 18, 20,
    #              -.1, -1, -2, -3, -6, -18, -20]
    # cond_labels = [0, 1, 2, 3, 6, 18, '1pr',
    #                -.1, -1, -2, -3, -6, -18, '-1pr']
    # psig_both_vf_df.insert(0, 'cond', cond_vals)
    # psig_both_vf_df.sort_values(by='cond', inplace=True)
    # psig_both_vf_df.set_index('cond', drop=True, inplace=True)
    # print(f"psig_both_vf_df:\n{psig_both_vf_df}")
    # psig_both_vf_df = psig_both_vf_df.drop(['separation', 'vis_field'], axis=1)
    # print(f"psig_both_vf_df:\n{psig_both_vf_df}")
    #
    # # make plot to compare UVF and LVF
    # fig_title = f'{participant_name} compare UVF & LVF'
    # legend_title = 'ISI'
    # x_tick_vals = cond_vals
    # x_tick_labels = cond_labels
    # x_axis = 'Sep in diag pixels. Neg=LVF, Pos=UVF'
    # y_axis = 'Threshold'
    # log_x = False
    # log_y = False
    # save_as = os.path.join(root_path, 'compare_vfs.png')
    #
    # fig, ax = plt.subplots(figsize=(10, 6))
    # sns.lineplot(data=psig_both_vf_df,
    #              markers=True, dashes=False, ax=ax)
    # if fig_title is not None:
    #     plt.title(fig_title)
    # if legend_title is not None:
    #     plt.legend(title=legend_title)
    # if x_tick_vals is not None:
    #     ax.set_xticks(x_tick_vals)
    # if x_tick_labels is not None:
    #     ax.set_xticklabels(x_tick_labels)
    #     if -18 in x_tick_labels:
    #         # add dotted line at zero
    #         ax.axvline(x=0, linestyle="-.", color='lightgrey')
    # if log_x:
    #     ax.set(xscale="log")
    #     x_axis = f'log {x_axis}'
    # if log_y:
    #     ax.set(yscale="log")
    #     y_axis = f'log {y_axis}'
    # if x_axis is not None:
    #     ax.set_xlabel(x_axis)
    # if y_axis is not None:
    #     ax.set_ylabel(y_axis)
    # if save_as is not None:
    #     plt.savefig(save_as)
    # plt.show()
    #
    # # # # end in v out





        # # # check for old column names and add new columns
        # col_names = run_data_df.columns.to_list()
        # # change probes_type to cond_type
        # if 'probes_type' in col_names:
        #     run_data_df.rename(columns={'probes_type': 'cond_type'}, inplace=True)
        #     print('\nrenamed probes_type to cond_type')
        #
        # # add neg sep column to make batman plots
        # if 'neg_sep' not in col_names:
        #     def make_neg_sep(df):
        #         if (df.cond_type == 'incoherent') and (df.separation == 0.0):
        #             return -.1
        #         elif df.cond_type == 'incoherent':
        #             return 0 - df.separation
        #         else:
        #             return df.separation
        #     run_data_df.insert(7, 'neg_sep', run_data_df.apply(make_neg_sep, axis=1))
        #     print('\nadded neg_sep col')
        #     print(run_data_df['neg_sep'].to_list())
        #
        # # todo: not sure I actually need this column at all - I don't use it
        # # fix stairnames
        # if len(run_data_df['stair_name'].unique().tolist()) == 1:
        #     run_data_df['stair_name'] = run_data_df['cond_type'].str[:3] + '_sep' + run_data_df['separation'].astype(str) + '_ISI' + run_data_df['ISI'].astype(str)
        #     print('\nupdated stair_name')
        #     print(run_data_df['stair_name'].to_list())
        #
        # if 'Unnamed: 0' in col_names:
        #     run_data_df.drop('Unnamed: 0', axis=1, inplace=True)
        #
        # print(f"run_data_df: {run_data_df.columns.to_list()}\n{run_data_df}")
        # run_data_df.to_csv(os.path.join(save_path, f'{p_name}.csv'), index=False)
        #
        # # # # get cond details for this exp
        # separation_list = run_data_df['separation'].unique().tolist()
        # neg_sep_list = run_data_df['neg_sep'].unique().tolist()
        # isi_list = run_data_df['ISI'].unique().tolist()
        # cond_type_list = run_data_df['cond_type'].unique().tolist()
        # print(f'separation_list: {separation_list}')
        # print(f'neg_sep_list: {neg_sep_list}')
        # print(f'isi_list: {isi_list}')
        # print(f'cond_type_list: {cond_type_list}')
        #
        # cols_to_add_dict = {'neg_sep': neg_sep_list}
        #
        # thr_df = get_psig_thr_w_hue(root_path=root_path,
        #                             p_run_name=run_dir,
        #                             output_df=run_data_df,
        #                             n_bins=9, q_bins=True,
        #                             thr_col='probeLum',
        #                             sep_col='separation', sep_list=separation_list,
        #                             isi_col='ISI', isi_list=isi_list,
        #                             hue_col='cond_type', hue_list=cond_type_list,
        #                             trial_correct_col='trial_response',
        #                             conf_int=True,
        #                             thr_type='Bayes',
        #                             plot_both_curves=False,
        #                             cols_to_add_dict=cols_to_add_dict,
        #                             show_plots=False,
        #                             verbose=verbose)
        # print(f'thr_df:\n{thr_df}')

#     # run_folder_names = ['Nick_1', 'Nick_5', 'Nick_6']
#     '''d participant averages'''
#     d_average_participant(root_path=root_path, run_dir_names_list=run_folder_names,
#                           groupby_col=['cond_type', 'neg_sep'], cols_to_drop='stack', cols_to_replace='separation',
#                           trim_n=trim_n, error_type='SE', verbose=verbose)
#
#
#     # making average plot
#     all_df_path = os.path.join(root_path, f'MASTER_TM{trim_n}_thresholds.csv')
#     p_ave_path = os.path.join(root_path, f'MASTER_ave_TM{trim_n}_thresh.csv')
#     err_path = os.path.join(root_path, f'MASTER_ave_TM{trim_n}_thr_error_SE.csv')
#     if trim_n is None:
#         all_df_path = os.path.join(root_path, f'MASTER_psignifit_thresholds.csv')
#         p_ave_path = os.path.join(root_path, 'MASTER_ave_thresh.csv')
#         err_path = os.path.join(root_path, 'MASTER_ave_thr_error_SE.csv')
#     exp_ave = False
#
#
#     make_average_plots(all_df_path=all_df_path,
#                        ave_df_path=p_ave_path,
#                        error_bars_path=err_path,
#                        thr_col='probeLum',
#                        stair_names_col='neg_sep',
#                        cond_type_col='cond_type',
#                        cond_type_order=['incoherent', coherence_type.lower()],
#                        n_trimmed=trim_n,
#                        ave_over_n=len(run_folder_names),
#                        exp_ave=False,
#                        show_plots=True, verbose=True)
#
#
#
# print(f'exp_path: {exp_path}')
# participant_list = ['Nick', 'Simon']
# print('\nget exp_average_data')
# trim_n = None
# e_average_exp_data(exp_path=exp_path, p_names_list=participant_list,
#                    exp_type='missing_probe',
#                    error_type='SE', n_trimmed=trim_n, verbose=True)
#
#
# all_df_path = os.path.join(exp_path, "MASTER_exp_thr.csv")
# exp_ave_path = os.path.join(exp_path, "MASTER_exp_ave_thr.csv")
# err_path = os.path.join(exp_path, "MASTER_ave_thr_error_SE.csv")
#
# make_average_plots(all_df_path=all_df_path,
#                    ave_df_path=exp_ave_path,
#                    error_bars_path=err_path,
#                    thr_col='probeLum',
#                    stair_names_col='neg_sep',
#                    cond_type_col='cond_type',
#                    cond_type_order=['incoherent', coherence_type.lower()],
#                    n_trimmed=trim_n,
#                    ave_over_n=len(participant_list),
#                    exp_ave=True,
#                    show_plots=True, verbose=True)

print('\nExp_4_missing_probe_analysis_pipe finished')
