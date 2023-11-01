import os
import pandas as pd
import numpy as np
from operator import itemgetter
from psignifit_tools import psignifit_thr_df_Oct23, get_psignifit_threshold_df, get_psig_thr_w_hue
from python_tools import switch_path
from rad_flow_psignifit_analysis import a_data_extraction_Oct23, get_sorted_neg_sep_indices, sort_with_neg_sep_indices
from rad_flow_psignifit_analysis import b3_plot_staircase, c_plots
from rad_flow_psignifit_analysis import d_average_participant, make_average_plots, e_average_exp_data
from rad_flow_psignifit_analysis import compare_prelim_plots, make_long_df, mean_staircase_plots
from exp1a_psignifit_analysis import plt_heatmap_row_col
import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib.lines as mlines
from rad_flow_psignifit_analysis import get_n_rows_n_cols, get_ax_idx


exp_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\rad_flow_v2_motion_window"
# exp_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\rad_flow_v2_missing_probe"
participant_list = ['Nick']  # ' Nicktest_06102023' Nick_extra_prelims
monitor = 'OLED'  # 'asus_cal' OLED, 'Nick_work_laptop'
background_type = 'flow_dots'  # 'flow_dots', 'no_bg'

exp_path = os.path.normpath(exp_path)
convert_path1 = os.path.normpath(exp_path)
# convert_path1 = switch_path(convert_path1, 'windows_oneDrive')
exp_path = convert_path1
exp_path = os.path.join(exp_path, monitor)

print(f"exp_path: {exp_path}")

verbose = True
show_plots = True

thr_col_name = 'probeLum'
stair_names_col_name = 'stair_name'
cong_col_name = 'congruent'
isi_col_name = 'isi_ms'
sep_col_name = 'separation'
neg_sep_col_name = 'neg_sep'
bg_dur_name = 'motion_dur_ms'  # prelim_ms
resp_col_name = 'resp_corr'
var_cols_list = [stair_names_col_name, isi_col_name, sep_col_name, neg_sep_col_name, cong_col_name, bg_dur_name]



# n_runs = 12
# if the first folder to analyse is 1, p_idx_plus = 1.  If the forst folder is 5, use 5 etc.
p_idx_plus = 1
trim_list = []

for p_idx, participant_name in enumerate(participant_list):

    print(f"\n\n{p_idx}. participant_name: {participant_name}")

    p_name_path = os.path.join(exp_path, participant_name)
    print(f"p_name_path: {p_name_path}")

    p_master_all_dfs_list = []
    p_master_ave_dfs_list = []
    p_master_err_dfs_list = []


    '''search to automatically get bg_cond_dirs containing run_folder_names (e.g., participant_name_1, participant_name_2, etc.).
    There could be any number of bakground folders (flow_dots, no_bg), (motion_window_dur)(stair_per_dir, bg_congruence) etc'''
    background_conds_to_analyse = []
    for root, dirs, files in os.walk(p_name_path):
        if f"{participant_name}_1" in dirs:
            print(f"\nfound {participant_name}_1 in {root}\n")
            # extract bg_cond_dirs from root (e.g., remove p_name_path from root) and leading slash
            bg_cond_dirs = root.replace(p_name_path, '')[1:]
            print(f"bg_cond_dirs: {bg_cond_dirs}")
            background_conds_to_analyse.append(bg_cond_dirs)

            # # split bg_cond_dirs into list of bg_cond_dirs
            # bg_cond_dirs_list = bg_cond_dirs.split(os.sep)
            # print(f"bg_cond_dirs_list: {bg_cond_dirs_list}")
            # break

    print(f"background_conds_to_analyse: {background_conds_to_analyse}")
    # '''check for background_type in folder name, if 'flow_dots', 'flow_rings', 'no_bg', loop through those, else continue'''
    # # # search to automatically get run_folder_names
    # dir_list = os.listdir(p_name_path)
    # bg_dir_list = []
    # for bg_type in ['flow_dots', 'flow_rings', 'no_bg']:
    #     if bg_type in dir_list:
    #         bg_dir_list.append(bg_type)
    # if len(bg_dir_list) == 0:
    #     bg_dir_list.append('No_background_type_found')
    # print(f'bg_dir_list: {bg_dir_list}')

    # for bg_type in bg_dir_list:
    #     print(f"\nbg_type: {bg_type}")
    #     if bg_type != 'No_background_type_found':

    # append each run's data to these lists for mean staircases
    MASTER_p_trial_data_list = []

    for bg_type in background_conds_to_analyse:
        bg_cond_path = os.path.join(p_name_path, bg_type)
        print(f"bg_cond_path: {bg_cond_path}")

        # # search to automatically get run_folder_names
        dir_list = os.listdir(bg_cond_path)
        run_folder_names = []
        for i in range(12):  # numbers 0 to 11
            check_dir = f'{participant_name}_{i + p_idx_plus}'  # numbers 1 to 12
            # print(check_dir)
            if check_dir in dir_list:
                run_folder_names.append(check_dir)

        print(f'run_folder_names: {run_folder_names}')

        trim_n = None
        if len(run_folder_names) == 12:
            trim_n = 2
        elif len(run_folder_names) > 12:
            # trim_n = 2
            if len(run_folder_names) % 2 == 0:  # if even
                trim_n = int((len(run_folder_names)-12)/2)
            else:
                raise ValueError(f"for this exp you have {len(run_folder_names)} runs, set rules for trimming.")
        trim_list.append(trim_n)

        for run_idx, run_dir in enumerate(run_folder_names):

            # add run number , e.g., add five to access Nick_5 on the zeroth iteration
            r_idx_plus = run_idx + p_idx_plus

            print(f'\nrun_idx {run_idx + 1}: running analysis for '
                  f'{participant_name}, {run_dir}, {participant_name}_{r_idx_plus}')
            # run_path = f'{bg_cond_path}{os.sep}{run_dir}'
            run_path = os.path.join(bg_cond_path, run_dir)
            print(f"run_path: {run_path}")


            # don't delete this (p_run_name = participant_name),
            # needed to ensure names go name1, name2, name3 not name1, name12, name123
            p_run_name = participant_name

            '''a'''
            p_run_name = f'{participant_name}_{r_idx_plus}'

            a_data_extraction_Oct23(p_name=p_run_name, run_dir=run_path,
                                    verbose=verbose)

            # run_data_path = f'{run_path}{os.sep}RUNDATA-sorted.xlsx'
            run_data_path = os.path.join(run_path, 'RUNDATA-sorted.xlsx')
            run_data_df = pd.read_excel(run_data_path, engine='openpyxl')
            print(f"run_data_df: {run_data_df.columns.to_list()}\n{run_data_df}")
            # append to master list for mean staircase

            # search for 'Run_number' substring in column names
            run_num_col = [col for col in run_data_df.columns if 'Run_number' in col]
            if len(run_num_col) == 1:
                run_col_name = run_num_col[0]
            elif len(run_num_col) == 0:
                run_col_name = 'run'
                # add 'run' to run_data_df if not already there
                if 'run' not in run_data_df.columns.tolist():
                    run_data_df.insert(0, 'run', run_idx + 1)
            MASTER_p_trial_data_list.append(run_data_df)


            # run psignifit on run_data_df using var_cols_list to loop through the variables
            thr_df = psignifit_thr_df_Oct23(save_path=run_path,
                                            p_run_name=p_run_name,
                                            run_df=run_data_df,
                                            cond_cols_list=var_cols_list,
                                            thr_col='probeLum',
                                            resp_col='resp_corr',
                                            wide_df_cols='isi_ms',
                                            n_bins=9, q_bins=True,
                                            conf_int=True, thr_type='Bayes',
                                            plot_both_curves=False,
                                            save_name=None,
                                            show_plots=True, save_plots=True,
                                            verbose=True)



# #
# #
#             '''b3 - stair case per run'''
#             b3_plot_staircase(run_data_path, thr_col='probeLum', show_plots=show_plots, verbose=verbose)
# # #
# # #         '''c I don't actually need any of these, instead sort get psignifit thr ands make plots from those.'''
# # #         c_plots(save_path=run_path, thr_col='probeLum', isi_name_list=run_isi_names_list, show_plots=show_plots, verbose=verbose)
#

        '''mean staircase for each bg type'''
        print("\n***running mean_staircase()***")
        # join all stacks (runs/groups) data and save as master csv
        MASTER_p_trial_data_df = pd.concat(MASTER_p_trial_data_list, ignore_index=True)

        MASTER_p_trial_data_name = f'MASTER_{bg_type}_p_trial_data_df.csv'

        MASTER_p_trial_data_df.to_csv(os.path.join(bg_cond_path, MASTER_p_trial_data_name), index=False)
        if verbose:
            print(f'\nMASTER_p_trial_data_df:\n{MASTER_p_trial_data_df}')

        mean_staircase_plots(per_trial_df=MASTER_p_trial_data_df, save_path=bg_cond_path, participant_name=participant_name,
                             isi_col_name=isi_col_name, sep_col_name=sep_col_name, run_col_name=run_col_name,
                             thr_col_name=thr_col_name,
                             hue_col_name=cong_col_name, hue_names=['Incongruent', 'Congruent'],
                             ave_type='mean',
                             show_plots=True, save_plots=True, verbose=False)

        # # # mean staircase for this bg type will loop through ISI and separation,
        # # then plot staircases and mean (step v probeLum) for each run and congruenct/incongruent
        # '''
        # 1. Calculate ave thr for each step for congruent and incongruent data.
        # 2. loop through MASTER_p_trial_data_df for each ISI/sep combination.
        # 3. make plot showing all 12 congruent and incongruent staircases on the same plot (different colours)
        # and ave cong and incong ontop.
        # '''
        #
        # get_median = False
        # ave_type = 'mean'
        # if get_median:
        #     ave_type = 'median'
        #
        # hue_names = ['Incongruent', 'Congruent']
        #
        #
        # run_colour_list = ['pink', 'lightblue']
        # ave_colour_list = ['red', 'blue']
        #
        # MASTER_p_trial_data_df = pd.read_csv(os.path.join(bg_cond_path, MASTER_p_trial_data_name))
        # print(f'\nMASTER_p_trial_data_df ({MASTER_p_trial_data_df.shape}):\n{MASTER_p_trial_data_df}')
        #
        #
        # isi_list = MASTER_p_trial_data_df[isi_col_name].unique()
        # if verbose:
        #     print(f"isi_list: {isi_list}")
        #
        # sep_list = MASTER_p_trial_data_df[sep_col_name].unique()
        # if verbose:
        #     print(f"sep_list: {sep_list}")
        #
        # n_runs = len(MASTER_p_trial_data_df[run_col_name].unique())
        #
        # plot_idx = 'df_cols'
        # # check if there is only one condition (e.g., only one isi)
        # if len(sep_list) > 1 and len(isi_list) > 1:
        #     n_rows = len(sep_list)
        #     n_cols = len(isi_list)
        # elif len(isi_list) > 1:
        #     n_rows, n_cols = get_n_rows_n_cols(len(isi_list))
        #     plot_idx = 'len(isi_list)'
        #     if verbose:
        #         print(f"plot indices from get_n_rows_n_cols(len(isi_list) for {n_rows} rows and {n_cols} cols")
        # elif len(sep_list) > 1:
        #     n_rows, n_cols = get_n_rows_n_cols(len(sep_list))
        #     plot_idx = 'len(sep_list)'
        #     if verbose:
        #         print(f"plot indices from get_n_rows_n_cols(len(sep_list) for {n_rows} rows and {n_cols} cols")
        # else:
        #     n_rows = len(sep_list)
        #     n_cols = len(isi_list)
        #
        # # initialise plot
        # fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols,
        #                          figsize=(n_cols * 6, n_rows * 6),
        #                          sharex=True)
        # if verbose:
        #     print(f'\nplotting {n_cols} rows and {n_rows} cols (axes: {axes})')
        #
        # # if there is only a single condition, put axes into a list to allow indexing, consistent with
        # # analyses where there are multiple conditons (e.g., axes[0])
        # if n_cols == 1 and n_rows == 1:
        #     axes = [axes]
        #
        # # loop through isi values
        # for isi_idx, isi in enumerate(isi_list):
        #     # get df for this isi only
        #     isi_df = MASTER_p_trial_data_df[MASTER_p_trial_data_df[isi_col_name] == isi]
        #     if verbose:
        #         print(f"\n{isi_idx}. ISI: {isi} ({isi_df.shape})"
        #               f"\n{isi_df}"
        #               )
        #
        #     # loop through sep values
        #     for sep_idx, sep in enumerate(sep_list):
        #         # get df for this sep only
        #         sep_df = isi_df[isi_df[sep_col_name] == sep]
        #         if verbose:
        #             print(f"\n{sep_idx}. sep {sep} ({sep_df.shape}):"
        #                   # f"\n{sep_df}"
        #                   )
        #
        #         # # check shape for accessing axes
        #         if plot_idx == 'df_cols':
        #             if len(sep_list) > 1 and len(isi_list) > 1:
        #                 this_ax = sep_idx, isi_idx
        #             else:
        #                 this_ax = 0
        #         elif plot_idx == 'len(isi_list)':  # only one sep
        #             this_ax = get_ax_idx(isi_idx, n_rows, n_cols)
        #         elif plot_idx == 'len(sep_list)':  # only one isi
        #             this_ax = get_ax_idx(sep_idx, n_rows, n_cols)
        #
        #
        #         # # get ave values for each step
        #         this_sep_df = sep_df[[run_col_name, 'step', cong_col_name, thr_col_name]]
        #         if verbose:
        #             print(f"this_sep_df ({this_sep_df.shape}):\n{this_sep_df}")
        #         if get_median:
        #             ave_step_thr_df = this_sep_df.groupby([cong_col_name, 'step'], sort=False).median()
        #         else:
        #             ave_step_thr_df = this_sep_df.groupby([cong_col_name, 'step'], sort=False).mean()
        #         ave_step_thr_df.reset_index(drop=False, inplace=True)
        #         ave_step_thr_df.drop(run_col_name, axis=1, inplace=True)
        #         if verbose:
        #             print(f"ave_step_thr_df ({ave_step_thr_df.shape}):\n{ave_step_thr_df}")
        #         wide_ave_step_thr_df = ave_step_thr_df.pivot(index='step', columns=cong_col_name, values=thr_col_name)
        #         wide_ave_step_thr_df.columns = hue_names
        #         if verbose:
        #             print(f"wide_ave_step_thr_df ({wide_ave_step_thr_df.shape}):\n{wide_ave_step_thr_df}")
        #
        #         stack_list = sep_df[run_col_name].unique()
        #         if verbose:
        #                 print(f"stack_list: {sep_list}")
        #
        #         # loop through stack values
        #         for stack_idx, stack in enumerate(stack_list):
        #             # get df for this stack only
        #             stack_df = sep_df[sep_df[run_col_name] == stack]
        #             if verbose:
        #                 print(f"\n{stack_idx}. stack {stack} ({stack_df.shape}):"
        #                       # f"\n{stack_df}"
        #                       )
        #
        #             this_stack_df = stack_df[['step', cong_col_name, thr_col_name]]
        #             if verbose:
        #                 print(f"this_stack_df ({this_stack_df.shape}):\n{this_stack_df}")
        #
        #             # I now have the data I need - reshape it so cong and incong are different columns
        #             wide_df = this_stack_df.pivot(index='step', columns=cong_col_name, values=thr_col_name)
        #             wide_df.columns = hue_names
        #             if verbose:
        #                 print(f"wide_df ({wide_df.shape}):\n{wide_df}")
        #
        #
        #             for idx, name in enumerate(hue_names):
        #                 axes[this_ax].errorbar(x=list(range(25)), y=wide_df[name],
        #                                        color=run_colour_list[idx])
        #
        #
        #         # add mean line
        #         for idx, name in enumerate(hue_names):
        #             axes[this_ax].errorbar(x=list(range(25)), y=wide_ave_step_thr_df[name],
        #                                    color=ave_colour_list[idx])
        #
        #         # decorate each subplot
        #         axes[this_ax].set_title(f"{isi_col_name}{isi}, {sep_col_name}{sep}")
        #         axes[this_ax].set_xlabel('step (25 per condition, per run)')
        #         axes[this_ax].set_ylabel(thr_col_name)
        #
        # # delete any unused axes
        # if plot_idx == 'df_cols':
        #     if len(sep_list) > 1 and len(isi_list) > 1:
        #         for ax in axes.flat[len(isi_list)*len(sep_list):]:
        #             ax.remove()
        #     elif len(isi_list) > 1:
        #         for ax in axes.flat[len(isi_list):]:
        #             ax.remove()
        #     elif len(sep_list) > 1:
        #         for ax in axes.flat[len(sep_list):]:
        #             ax.remove()
        #
        #
        # # artist for legend
        # st0 = mlines.Line2D([], [], color=run_colour_list[0],
        #                     markersize=4, label=hue_names[0])
        # st1 = mlines.Line2D([], [], color=run_colour_list[1],
        #                     markersize=4, label=hue_names[1])
        # ave_line_0 = mlines.Line2D([], [], color=ave_colour_list[0],
        #                            marker=None, linewidth=.5,
        #                            label=f'{hue_names[0]} ({ave_type})')
        # ave_line_1 = mlines.Line2D([], [], color=ave_colour_list[1],
        #                            marker=None, linewidth=.5,
        #                            label=f'{hue_names[1]} ({ave_type})')
        #
        # handles_list = [st0, ave_line_0, st1, ave_line_1]
        # fig.legend(handles=handles_list, fontsize=16, loc='lower right')
        #
        # fig.suptitle(f"{participant_name}: {n_runs} staircases with {ave_type}")
        #
        # if get_median:
        #     save_name = f"all_run_stairs_{participant_name}_{isi_col_name}{isi}_{sep_col_name[:3]}{sep}_median.png"
        # else:
        #     save_name = f"all_run_stairs_{participant_name}_{isi_col_name}{isi}_{sep_col_name[:3]}{sep}_mean.png"
        #
        # print(f"save plot to: {os.path.join(bg_cond_path, save_name)}")
        # plt.savefig(os.path.join(bg_cond_path, save_name))
        #
        # plt.show()

        '''d participant averages'''
        trim_n = None
        if len(run_folder_names) == 12:
            trim_n = 2
        print(f'\ntrim_n: {trim_n}')

        cols_to_drop = ['stack', 'stair_name']
        cols_to_replace = [cong_col_name, sep_col_name, 'motion_dur_ms']
        groupby_cols = ['neg_sep']

        d_average_participant(root_path=bg_cond_path, run_dir_names_list=run_folder_names,
                              trim_n=trim_n,
                              groupby_col=groupby_cols,
                              cols_to_drop=cols_to_drop,
                              cols_to_replace=cols_to_replace,
                              error_type='SE', verbose=verbose)


        # making average plot
        all_df_path = os.path.join(bg_cond_path, f'MASTER_TM{trim_n}_thresholds.csv')
        p_ave_path = os.path.join(bg_cond_path, f'MASTER_ave_TM{trim_n}_thresh.csv')
        err_path = os.path.join(bg_cond_path, f'MASTER_ave_TM{trim_n}_thr_error_SE.csv')
        if trim_n is None:
            all_df_path = os.path.join(bg_cond_path, f'MASTER_psignifit_thresholds.csv')
            p_ave_path = os.path.join(bg_cond_path, 'MASTER_ave_thresh.csv')
            err_path = os.path.join(bg_cond_path, 'MASTER_ave_thr_error_SE.csv')
        exp_ave = False

        # todo: turn into function?
        # make a lineplot showing congruent and incongruent thresholds for each ISI
        all_df = pd.read_csv(all_df_path)
        # todo: automate this - any col_name containing 'isi_ms_' in df columns.
        cols_to_change = ['isi_ms_-1', 'isi_ms_0', 'isi_ms_16.67', 'isi_ms_25', 'isi_ms_33.33', 'isi_ms_50',
                          'isi_ms_100']

        long_df = make_long_df(wide_df=all_df,
                               cols_to_keep=[cong_col_name],
                               cols_to_change=cols_to_change,
                               cols_to_change_show='probeLum',
                               new_col_name=isi_col_name, strip_from_cols='isi_ms_', verbose=True)
        print(f"long_df: {long_df.shape}\ncolumns: {list(long_df.columns)}\n{long_df}\n")

        # make line plot with error bars for congruent and incongruent with isi on x axis
        # use the basic palette
        sns.lineplot(data=long_df, x=isi_col_name, y='probeLum', hue=cong_col_name,
                     errorbar='se', err_style='bars', err_kws={'capsize': 5},
                     palette=sns.color_palette("tab10", n_colors=2))

        # change legend labels such that they are 1=congruent and -1=incongruent
        handles, labels = plt.gca().get_legend_handles_labels()
        new_labels = []
        for label in labels:
            if label == '1':
                new_labels.append('Congruent')
            elif label == '-1':
                new_labels.append('Incongruent')
            else:
                new_labels.append(label)
        # make legend box 50% opaque
        plt.legend(title='Probe & background', handles=handles, labels=new_labels,
                   framealpha=0.5)

        # for x-axis labels, if the isi is -1, change to 'Concurrent'
        x_tick_values = sorted(long_df[isi_col_name].unique())
        x_tick_labels = []
        for label in x_tick_values:
            if label in [-1, '-1']:
                x_tick_labels.append('Concurrent')
            else:
                x_tick_labels.append(label)

        # decorate plot
        plt.xlabel('ISI (ms)')
        plt.ylabel('Probe luminance')
        plt.xticks(ticks=x_tick_values, labels=x_tick_labels)

        suptitle_text = f"{participant_name} thresholds for each ISI"
        if trim_n is not None:
            suptitle_text = f"{participant_name} thresholds for each ISI, trimmed {trim_n}"
        plt.suptitle(suptitle_text)
        plt.title("separation = 4; motion window = 200ms")
        run_path, df_name = os.path.split(all_df_path)
        plt.savefig(os.path.join(run_path, f"{df_name[:-4]}_lineplot.png"))
        plt.show()






            # # decorate plot
            # if x_tick_values is not None:
            #     ax.set_xticks(x_tick_values)
            # if x_tick_labels is not None:
            #     ax.set_xticks(x_tick_values)
            #     ax.set_xticklabels(x_tick_labels)
            # ax.set_xlabel('Probe separation in diagonal pixels')
            # ax.set_ylabel('Probe Luminance')
            #
            # ax.legend(labels=isi_name_list, title=isi_col_name,
            #           shadow=True,
            #           # place lower left corner of legend at specified location.
            #           loc='lower left', bbox_to_anchor=(0.96, 0.5))
            #
            # if fig_title is not None:
            #     plt.title(fig_title)
            #
            # # save plot
            # if save_path is not None:
            #     if save_name is not None:
            #         plt.savefig(os.path.join(save_path, save_name))
            #
            # if verbose:
            #     print("\n*** finished plot_data_unsym_batman() ***\n")


    #         make_average_plots(all_df_path=all_df_path,
    #                            ave_df_path=p_ave_path,
    #                            error_bars_path=err_path,
    #                            thr_col='probeLum',
    #                            stair_names_col='neg_sep',
    #                            cond_type_order=[1, -1],
    #                            pos_neg_labels=['Congruent', 'Incongruent'],
    #                            n_trimmed=trim_n,
    #                            ave_over_n=len(run_folder_names),
    #                            exp_ave=participant_name,
    #                            show_plots=True, verbose=True)
    #
    #         # add columns (background, prelim_ms) to all_df (and ave_df and err_df if needed)
    #         all_df = pd.read_csv(all_df_path)
    #         if 'background' not in all_df.columns.tolist():
    #             all_df.insert(0, 'background', bg_type)
    #         # if 'prelim_ms' not in all_df.columns.tolist():
    #         #     all_df.insert(1, 'prelim_ms', prelim_flow_dur)
    #         p_master_all_dfs_list.append(all_df)
    #
    #         ave_df = pd.read_csv(p_ave_path)
    #         if 'background' not in ave_df.columns.tolist():
    #             ave_df.insert(0, 'background', bg_type)
    #         # if 'prelim_ms' not in ave_df.columns.tolist():
    #         #     ave_df.insert(1, 'prelim_ms', prelim_flow_dur)
    #         p_master_ave_dfs_list.append(ave_df)
    #
    #         err_df = pd.read_csv(err_path)
    #         if 'background' not in err_df.columns.tolist():
    #             err_df.insert(0, 'background', bg_type)
    #         # if 'prelim_ms' not in err_df.columns.tolist():
    #         #     err_df.insert(1
    #         #                   , 'prelim_ms', prelim_flow_dur)
    #         p_master_err_dfs_list.append(err_df)
    #
    #
    # # make master list for each participant with their average threshold for each background type and prelim flow dur
    # p_compare_prelim_dir = os.path.join(exp_path, participant_name, 'compare_prelims')
    # if not os.path.exists(p_compare_prelim_dir):
    #     os.mkdir(p_compare_prelim_dir)
    #
    # p_master_all_df = pd.concat(p_master_all_dfs_list)
    # p_master_all_name = os.path.join(p_compare_prelim_dir, f'{participant_name}_ALLbg_thresholds.csv')
    # if trim_n is not None:
    #     p_master_all_name = os.path.join(p_compare_prelim_dir, f'{participant_name}_TM{trim_n}_ALLbg_thresholds.csv')
    # p_master_all_df.to_csv(p_master_all_name, index=False)
    #
    # # p_root_path = os.path.join(exp_path, participant_name)
    # p_master_ave_df = pd.concat(p_master_ave_dfs_list)
    # p_master_ave_name = os.path.join(p_compare_prelim_dir, f'{participant_name}_ALLbg_ave_thresh.csv')
    # if trim_n is not None:
    #     p_master_ave_name = os.path.join(p_compare_prelim_dir, f'{participant_name}_TM{trim_n}_ALLbg_ave_thresh.csv')
    # p_master_ave_df.to_csv(p_master_ave_name, index=False)
    #
    # p_master_err_df = pd.concat(p_master_err_dfs_list)
    # p_master_err_name = os.path.join(p_compare_prelim_dir, f'{participant_name}_ALLbg_thr_error_SE.csv')
    # if trim_n is not None:
    #     p_master_err_name = os.path.join(p_compare_prelim_dir, f'{participant_name}_TM{trim_n}_ALLbg_thr_error_SE.csv')
    # p_master_err_df.to_csv(p_master_err_name, index=False)
    #
    # # make prelim plots for this participant
    # compare_prelim_plots(participant_name, exp_path)




'''
Don't use this bottom bit as it isn't really set up for comparing prelims

'''
# print(f'exp_path: {exp_path}')
# # participant_list = ['Nick_350', 'Simon']
# participant_list = ['OLED_circles_rings_quartSpd', 'OLED_circles_rings_quartSpd_v2']
# print('\nget exp_average_data')
# # todo: check trim_n is correct
# trim_n = 12
# # todo: sort script to automatically use trim=2 if its there, and not just use untrimmed#
# # todo: make sure ISI cols are in the correct order
# e_average_exp_data(exp_path=exp_path, p_names_list=participant_list,
#                    error_type='SE',
#                    # n_trimmed=trim_n,
#                    verbose=True)
#
#
# all_df_path = os.path.join(exp_path, "MASTER_exp_thr.csv")
# exp_ave_path = os.path.join(exp_path, "MASTER_exp_ave_thr.csv")
# err_path = os.path.join(exp_path, "MASTER_ave_thr_error_SE.csv")
#
# all_df = pd.read_csv(all_df_path)
# if any("Unnamed" in i for i in list(all_df.columns)):
#     unnamed_col = [i for i in list(all_df.columns) if "Unnamed" in i][0]
#     all_df.drop(unnamed_col, axis=1, inplace=True)
# print(f"all_df:\n{all_df}")
#
# all_df_basic_cols = ['participant', 'stair_names', 'congruent', sep_col_name]
#
# # isi_names_list = list(all_df.columns[len(all_df_basic_cols):])
# isi_names_list = [i for i in list(all_df.columns) if 'isi' in i.lower()]
#
# isi_vals_list = [int(i[4:]) for i in isi_names_list]
#
# # sort isi_names_list by sorted(isi_vals_list) order
# isi_vals_array = np.array(isi_vals_list)
# print(f"\nisi_vals_array: {isi_vals_array}")
# sort_index = np.argsort(isi_vals_array)
# print(f"sort_index: {sort_index}")
#
# isi_vals_list = [isi_vals_list[i] for i in sort_index]
# print(f"isi_vals_list: {isi_vals_list}")
#
# isi_names_list = [isi_names_list[i] for i in sort_index]
# print(f"isi_names_list: {isi_names_list}")
#
# all_col_names = all_df_basic_cols + isi_names_list
# print(f"all_col_names: {all_col_names}")
#
# all_df = all_df[all_col_names]
# print(f"all_df:\n{all_df}")
# all_df.to_csv(all_df_path, index=False)
#
#
#
# ave_df = pd.read_csv(exp_ave_path)
# if any("Unnamed" in i for i in list(ave_df.columns)):
#     unnamed_col = [i for i in list(ave_df.columns) if "Unnamed" in i][0]
#     ave_df.drop(unnamed_col, axis=1, inplace=True)
# print(f"ave_df:\n{ave_df}")
# ave_df_basic_cols = ['stair_names']
# ave_col_names = ave_df_basic_cols + isi_names_list
# print(f"ave_col_names: {ave_col_names}")
# ave_df = ave_df[ave_col_names]
# print(f"ave_df:\n{ave_df}")
# ave_df.to_csv(exp_ave_path, index=False)
#
# err_df = pd.read_csv(err_path)
# if any("Unnamed" in i for i in list(err_df.columns)):
#     unnamed_col = [i for i in list(err_df.columns) if "Unnamed" in i][0]
#     err_df.drop(unnamed_col, axis=1, inplace=True)
# print(f"err_df:\n{err_df}")
#
# # replace any NaNs with 0s
# err_df.fillna(0, inplace=True)
#
# err_df = err_df[ave_col_names]
# print(f"err_df:\n{err_df}")
# err_df.to_csv(err_path, index=False)
#
#
# make_average_plots(all_df_path=all_df_path,
#                    ave_df_path=exp_ave_path,
#                    error_bars_path=err_path,
#                    thr_col='probeLum',
#                    stair_names_col='stair_names',
#                    cond_type_col='congruent',
#                    cond_type_order=[1, -1],
#                    n_trimmed=trim_n,
#                    ave_over_n=len(participant_list),
#                    exp_ave=True,
#                    isi_name_list=isi_names_list,
#                    isi_vals_list=isi_vals_list,
#                    show_plots=True, verbose=True)

print('\nrad_flow_analysis_pipe finished')
