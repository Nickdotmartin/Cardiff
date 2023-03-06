import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from exp1a_psignifit_analysis import a_data_extraction, b3_plot_staircase, c_plots, \
    d_average_participant, e_average_exp_data, make_average_plots
from psignifit_tools import get_psignifit_threshold_df, get_psig_thr_w_hue
from python_tools import which_path, running_on_laptop, switch_path

'''
This script if for checking for any differences between thr upper visual field and lower visual field.
It will use the already analysied RUNDATA_sorted.xlsx to do this.
Loop through the participant run folders and append each RUNDATA-sorted.xlsx, with addition 'run' column.
Save to P_all_runs_master_output.csv.

Then run psignifit on this
'''

# # loop through run folders with first 4 scripts (a, get_psignifit_threshold_df, b3, c)
# # then run script d to get master lists and averages
# exp_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/exp1a_data'
# participant_list = ['aa', 'bb', 'cc', 'dd', 'ee']
# exp_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/EXP1_sep4_5'
# exp_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\EXP1_sep4_5"
# exp_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\Exp3_Ricco_NM_v4"
exp_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\Exp2_Bloch_NM_v5"
convert_path1 = os.path.normpath(exp_path)
if running_on_laptop():
    convert_path1 = switch_path(convert_path1, 'mac_oneDrive')
exp_path = convert_path1

participant_list = ['Tony', 'Kris', 'Simon', 'Nick', 'Kim']  # ['Kim', 'Kris', 'Simon', 'Nick']
# participant_list = ['Nick']  # ['Kim', 'Kris', 'Simon', 'Nick']
p_idx_plus = 1

n_runs = 12
analyse_from_run = 1
trim_list = []

all_participants = []

for p_idx, participant_name in enumerate(participant_list):

    root_path = os.path.join(exp_path, participant_name)

    # search to automatically get run_folder_names
    dir_list = os.listdir(root_path)
    run_folder_names = []
    for i in range(n_runs):  # numbers 0 to 11
        check_dir = f'{participant_name}_{i + analyse_from_run}'  # numbers 1 to 12
        if check_dir in dir_list:
            run_folder_names.append(check_dir)

    if len(run_folder_names) > 0:
        print("running analysis for:")
        for i in run_folder_names:
            print(i)
    else:
        print("no run folders found")


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
    #     p_name = f'{participant_name}_{run_idx+1}_output.csv'
    #     # p_name = f'{participant_name}{run_idx+1}'
    #     # isi_list = [-1, 0, 2, 4, 6, 9, 12, 24]
    #
    #     if os.path.isfile(os.path.join(save_path, 'RUNDATA-sorted.xlsx')):
    #         run_data_path = os.path.join(save_path, 'RUNDATA-sorted.xlsx')
    #     elif os.path.isfile(os.path.join(save_path, p_name)):
    #         run_data_path = os.path.join(save_path, p_name)
    #     elif os.path.isfile(os.path.join(save_path, f'{run_dir}_output.csv')):
    #         run_data_path = os.path.join(save_path, f'{run_dir}_output.csv')
    #     elif os.path.isfile(os.path.join(save_path, f'{participant_name}_output.csv')):
    #         run_data_path = os.path.join(save_path, f'{participant_name}_output.csv')
    #     else:
    #         raise FileNotFoundError(f'{participant_name}, run_dir {run_dir}')
    #
    #     # run_data_path = f'{save_path}{os.sep}RUNDATA-sorted.xlsx'
    #
    #     # run_data_path = os.path.join(save_path, )
    #
    #     if run_data_path[-4:] == 'xlsx':
    #         run_data_df = pd.read_excel(run_data_path, engine='openpyxl',
    #                                     # usecols=['ISI',
    #                                     #          'stair',
    #                                     #          'separation',
    #                                     #          # 'group',
    #                                     #          'probeLum', 'trial_response', 'corner']
    #                                     )
    #     else:
    #         run_data_df = pd.read_csv(run_data_path)
    #     print(f"run_data_df:\n{run_data_df}")
    #
    #     # add isi column for multi-indexing
    #     if 'run' not in list(run_data_df.columns):
    #         run_data_df.insert(0, 'run', int(run_idx+1))
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
    # print(f'all_data_shape:\n{all_data_shape}')
    #
    # if len(np.shape(all_data)) == 2:
    #     sheets, rows, columns = np.shape(all_data)
    #     all_data = np.reshape(all_data, newshape=(sheets * rows, columns))
    #     # if verbose:
    #     print(f'all_data reshaped from {all_data_shape} to {np.shape(all_data)}')
    #     all_data_df = pd.DataFrame(all_data, columns=column_names)
    # else:
    #     all_data_df = pd.concat(all_data, ignore_index=True)
    #
    # visual_field_list = ['UVF' if i < 200 else 'LVF' for i in all_data_df['corner'].to_list()]
    # all_data_df['vis_field'] = visual_field_list
    # # if verbose:
    # print(f"all_data_df:\n{all_data_df}")
    #
    # sep_list = sorted(list(all_data_df['separation'].unique()))
    # print(f"sep_list: {sep_list}")
    #
    #
    # # # if save_all_data:
    # save_name = 'P_all_runs_master_output.csv'
    # save_csv_path = os.path.join(root_path, save_name)
    # # # if verbose:
    # print(f"\nsaving all_data_df to save_csv_path: {save_csv_path}")
    # all_data_df.to_csv(save_csv_path, index=False)
    #
    #
    #
    # all_data_df = pd.read_csv(os.path.join(root_path, 'P_all_runs_master_output.csv'))
    #
    # vis_field_names = ['UVF', 'LVF']
    #
    #
    # both_vfs = []
    #
    # for idx, vis_field_name in enumerate(vis_field_names):
    #
    #
    #     print(f'Running psignifit for {vis_field_name}')
    #
    #     vis_field_df = all_data_df[all_data_df['vis_field'] == vis_field_name]
    #     print(vis_field_df)
    #
    #     isi_list = sorted(list(vis_field_df['ISI'].unique()))
    #     print(f"isi_list: {isi_list}")
    #
    #     sep_list = sorted(list(vis_field_df['separation'].unique()))
    #     print(f"sep_list: {sep_list}")
    #
    #
    #
    #
    #     '''get psignifit thresholds df - use stairs as sep levels rather than using groups'''
    #
    #     thr_df = get_psignifit_threshold_df(root_path=exp_path,
    #                                         p_run_name=participant_name,
    #                                         csv_name=vis_field_df,
    #                                         n_bins=9, q_bins=True,
    #                                         thr_col='probeLum',
    #                                         sep_col='separation', sep_list=sep_list,
    #                                         isi_col='ISI', isi_list=isi_list,
    #                                         conf_int=True, thr_type='Bayes',
    #                                         plot_both_curves=False,
    #                                         # cols_to_add_dict=None, save_name=f'psignifit_{vis_field_name}_ISI{ISI}_sep{separation}',
    #                                         cols_to_add_dict=None, save_name=f'psignifit_{vis_field_name}',
    #                                         show_plots=False, save_plots=False,
    #                                         verbose=True)
    #
    #     thr_df['vis_field'] = vis_field_name
    #
    #     cond_list = thr_df['separation'].to_list()
    #     if vis_field_name == 'LVF':
    #         cond_list = [-.01 if i == 0 else -i for i in cond_list]
    #     thr_df['cond'] = cond_list
    #
    #     print(f'psignifit_{vis_field_name}:\n{thr_df}')
    #     column_names = list(thr_df)
    #
    #     both_vfs.append(thr_df)
    #
    #     progress_df = pd.concat(both_vfs)
    #     save_name = 'psignifit_progress.csv'
    #     save_csv_path = os.path.join(root_path, save_name)
    #     print(f"\nsaving progress_df to save_csv_path:\n{save_csv_path}")
    #     progress_df.to_csv(save_csv_path, index=False)
    #
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


    # make master df for experiment level values
    print(f"root_path: {root_path}")

    psig_both_vf_df = pd.read_csv(os.path.join(root_path, 'psignifit_both_vfs.csv'))
    print(f"psig_both_vf_df:\n{psig_both_vf_df}")



    ISI_labels_list = list(psig_both_vf_df.columns)
    ISI_labels_list = [i[4:] for i in ISI_labels_list if 'ISI_' in str(i)]
    ISI_labels_list = [i[:4] for i in ISI_labels_list if len(i) > 3]
    print(f"ISI_labels_list: {ISI_labels_list}")
    ISI_vals_list = [i.split('.')[0] for i in ISI_labels_list]
    ISI_vals_list = [int(i) for i in ISI_vals_list]
    print(f"ISI_vals_list: {ISI_vals_list}")

    # move vis_field to first column, remove other columns then transpose.
    if psig_both_vf_df.columns.tolist()[0] != 'vis_field':
        col_list = psig_both_vf_df.columns.tolist()
        # other_cols = [i for i in col_list if 'vis_field' not in i]
        this_col = [i for i in col_list if 'vis_field' in i]
        isi_cols = [i for i in col_list if 'ISI' in i]
        new_cols_list = this_col + isi_cols
        print(f"new_cols_list: {new_cols_list}")
        psig_both_vf_df = psig_both_vf_df.reindex(columns=new_cols_list)
        print(f"Concurrent column moved to start\n{psig_both_vf_df}")
    print(f"psig_both_vf_df:\n{psig_both_vf_df}")

    # psig_both_vf_df = psig_both_vf_df.drop(['separation', 'cond'], axis=1)
    # print(f"psig_both_vf_df:\n{psig_both_vf_df}")

    psig_both_vf_df.set_index('vis_field', inplace=True)
    print(f"psig_both_vf_df:\n{psig_both_vf_df}")

    psig_both_vf_transpose_df = psig_both_vf_df.T
    print(f"psig_both_vf_transpose_df:\n{psig_both_vf_transpose_df}")


    # all conditions
    # make plot to show UVF and LVF on one axis
    fig_title = f'{participant_name} compare UVF & LVF'
    legend_title = 'VF'

    x_tick_vals = ISI_vals_list
    x_tick_labels = ISI_labels_list
    x_axis = 'ISI (probe duration = 16.67ms + ISI). Neg=LVF, Pos=UVF'
    y_axis = 'Threshold'
    log_x = False
    log_y = False
    save_as = os.path.join(root_path, 'compare_vfs.png')

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=psig_both_vf_transpose_df,
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
    if x_axis is not None:
        ax.set_xlabel(x_axis)
    if y_axis is not None:
        ax.set_ylabel(y_axis)
    if save_as is not None:
        plt.savefig(save_as)
    plt.show()

    # make difference df
    # for each separation value, subtract LFV from UVF for difference score.

    get_diff_df = psig_both_vf_df.drop(['cond'], axis=1)
    print(f"get_diff_df:\n{get_diff_df}")

    LVF_df = get_diff_df.loc[get_diff_df['vis_field'] == 'LVF']
    LVF_df = LVF_df.drop(['vis_field'], axis=1)
    LVF_df.set_index('separation', inplace=True)


    UVF_df = get_diff_df.loc[get_diff_df['vis_field'] == 'UVF']
    UVF_df = UVF_df.drop(['vis_field'], axis=1)
    UVF_df.set_index('separation', inplace=True)
    print(f"LVF_df:\n{LVF_df}")
    print(f"UVF_df:\n{UVF_df}")

    # plot difference.
    diff_df = UVF_df.subtract(LVF_df, fill_value=0)

    print(f"diff_df:\n{diff_df}")

    pos_sep_vals = diff_df.index.to_list()


    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=diff_df, markers=True, dashes=False, ax=ax)
    fig_title = f'{participant_name} diff UVF - LVF'
    plt.title(fig_title)
    x_axis = 'ISI (probe duration = 16.67ms + ISI). Neg=LVF, Pos=UVF'
    ax.set_xlabel(x_axis)
    y_axis = 'Threshold different (UVF - LVF)'
    ax.set_ylabel(y_axis)
    save_as = os.path.join(root_path, 'diff_vfs.png')
    plt.savefig(save_as)
    plt.show()


    # # make master df for experiment level values
    # print(f"root_path: {root_path}")
    #
    # psig_both_vf_df = pd.read_csv(os.path.join(root_path, 'psignifit_both_vfs.csv'))
    # print(f"psig_both_vf_df:\n{psig_both_vf_df}")

    if 'cond' not in list(psig_both_vf_df.columns):
        print("\nMaking cond column")
        # add condition list which is equal to sep for uVF or negative sep for LVF (with -.01 instead of -0)
        sep_list = psig_both_vf_df['separation'].to_list()
        vf_list = psig_both_vf_df['vis_field'].to_list()
        cond_list = []
        for vf, sep in zip(vf_list, sep_list):
            if vf == 'LVF':
                if sep == 0:
                    this_cond = -.01
                else:
                    this_cond = -sep
            else:
                this_cond = sep
            print(f"vf: {vf}, sep: {sep}, this_cond: {this_cond}")
            cond_list.append(this_cond)
        print(f"cond_list: {cond_list}")
        psig_both_vf_df.insert(2, 'cond', cond_list)
        # save_name = 'psignifit_both_vfs.csv'
        # save_csv_path = os.path.join(root_path, save_name)
        # print(f"\nsaving all_data_df to save_csv_path:\n{save_csv_path}")
        # psig_both_vf_df.to_csv(save_csv_path, index=False)

    # add participant name
    psig_both_vf_df.insert(0, 'p_name', participant_name)

    # just concurrent
    # just_conc_df = psig_both_vf_df[['p_name', 'vis_field', 'cond', 'ISI_-']]

    print(f"psig_both_vf_df:\n{psig_both_vf_df}")
    all_participants.append(psig_both_vf_df)

all_p_df = pd.concat(all_participants)
save_csv_path = os.path.join(exp_path, 'exp_VFs.csv')
all_p_df.to_csv(save_csv_path, index=False)

print(f"all_p_df:\n{all_p_df}")


print('\nexp1a_analysis_pipe_UVF_LVF finished\n')
