import os
import pandas as pd
import numpy as np
from operator import itemgetter
from psignifit_tools import psignifit_thr_df_Oct23, get_psignifit_threshold_df, get_psig_thr_w_hue
from python_tools import switch_path
from rad_flow_psignifit_analysis import a_data_extraction_Oct23, get_sorted_neg_sep_indices, sort_with_neg_sep_indices
from rad_flow_psignifit_analysis import b3_plot_staircase, c_plots, rad_flow_line_plot
from rad_flow_psignifit_analysis import d_average_participant, make_average_plots, e_average_exp_data
from rad_flow_psignifit_analysis import compare_prelim_plots, make_long_df, mean_staircase_plots
from exp1a_psignifit_analysis import plt_heatmap_row_col
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

import matplotlib.lines as mlines
from rad_flow_psignifit_analysis import get_n_rows_n_cols, get_ax_idx


exp_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\rad_flow_v3_motion_window"
# exp_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\rad_flow_v2_missing_probe"
# exp_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\rad_flow_v4_multi_window"
participant_list = ['Nick_SiSettings']  # ' Nicktest_06102023' Nick_extra_prelims
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
bg_dur_name = 'bg_motion_ms'  # bg_motion_ms, prelim_ms, motion_dur_ms
resp_col_name = 'resp_corr'
var_cols_list = [stair_names_col_name, isi_col_name, sep_col_name, neg_sep_col_name, cong_col_name, bg_dur_name]
# don't use stairnames for multi-window
# var_cols_list = [isi_col_name, sep_col_name, neg_sep_col_name, cong_col_name, bg_dur_name]

# todo: if extra interleaved condition
interleaved_col_list = None  # [bg_dur_name]  # use False or None if no interleaved conditions


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


    print(f"background_conds_to_analyse: {background_conds_to_analyse}")


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
            print(f"run_col_name: {run_col_name}")
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
                                            show_plots=False, save_plots=True,
                                            verbose=True)



        '''mean staircase for each bg type'''
        print(f"\n***making master per-trial df (for this bg_type: {bg_type})***")
        # join all output data from each run and save as master per-trial csv
        MASTER_p_trial_data_df = pd.concat(MASTER_p_trial_data_list, ignore_index=True)
        MASTER_p_trial_data_name = f'MASTER_{bg_type}_p_trial_data.csv'
        MASTER_p_trial_data_df.to_csv(os.path.join(bg_cond_path, MASTER_p_trial_data_name), index=False)
        if verbose:
            print(f'\nMASTER_p_trial_data_df:\n{MASTER_p_trial_data_df}')

        '''If there are interleaved conds (e.g., bg_motion_ms), do separate staircases for each of them'''
        if not interleaved_col_list:  # do one staircase plt for all interleaved conditions
            mean_staircase_plots(per_trial_df=MASTER_p_trial_data_df, save_path=bg_cond_path,
                                 participant_name=participant_name, run_col_name=run_col_name,
                                 thr_col_name=thr_col_name,
                                 isi_col_name=isi_col_name, sep_col_name=sep_col_name,
                                 hue_col_name=cong_col_name, hue_names=['Incongruent', 'Congruent'],
                                 ave_type='mean',
                                 show_plots=True, save_plots=True, verbose=True)
        elif len(interleaved_col_list) == 1:  # do one staircase plt for each interleaved condition
            interleaves_vals = MASTER_p_trial_data_df[interleaved_col_list[0]].unique().tolist()
            print(f"\n\n***separate mean_stair for each condition in {interleaved_col_list}***")
            for this_val in interleaves_vals:
                print(f"\n\n{this_val}")
                this_save_path = os.path.join(bg_cond_path, f"{interleaved_col_list[0]}_{this_val}")
                this_df = MASTER_p_trial_data_df[MASTER_p_trial_data_df[interleaved_col_list[0]] == this_val]
                print(f"this_df ({this_df.shape}):\n{this_df}")
                mean_staircase_plots(per_trial_df=this_df, save_path=this_save_path,
                                     participant_name=participant_name, run_col_name=run_col_name,
                                     thr_col_name=thr_col_name,
                                     isi_col_name=isi_col_name, sep_col_name=sep_col_name,
                                     hue_col_name=cong_col_name, hue_names=['Incongruent', 'Congruent'],
                                     ave_type='mean',
                                     show_plots=True, save_plots=True, verbose=True)

        # else:  # if multiple interleaved conditions, separate staircases plot for each combination
        #     # make dict of all interleaved conditions
        #     interleaved_dict = {}
        #     for interleaved_col in interleaved_col_list:
        #         print(f"\n\n***separate mean_stair for each condition in {interleaved_col_list}***")
        #         interleaved_dict[interleaved_col] = MASTER_p_trial_data_df[interleaved_col].unique().tolist()
        #     print(f"interleaved_dict[{interleaved_col}]: {interleaved_dict[interleaved_col]}")
        #     # iterate through all permutations of interleaved conditions in interleaved_dict
        #     # iterate over all possible combinations of interleaved_dict values,
        #     # appending a list of tuples of keys and values
        #     combo_list = []
        #     dict_keys = list(interleaved_dict.keys())
        #     for combo in itertools.product(*interleaved_dict.values()):
        #         # print(f"{' '.join([f'{k}: {v}' for k, v in zip(dict_keys, combo)])}")
        #         combo_list.append(list(zip(dict_keys, combo)))
        #     print(f"combo_list: {combo_list}")



        '''d participant averages'''
        trim_n = None
        if len(run_folder_names) == 12:
            trim_n = 2
        print(f'\ntrim_n: {trim_n}')

        cols_to_drop = ['stack', 'stair_name']
        # cols_to_replace = [cong_col_name, sep_col_name, bg_dur_name]
        # groupby_cols = ['neg_sep']
        if interleaved_col_list is not None:
            cols_to_replace = [cong_col_name, sep_col_name]
            groupby_cols = ['neg_sep', bg_dur_name]
        else:  # if no interleaved conditions
            cols_to_replace = [cong_col_name, sep_col_name, bg_dur_name]
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

        # ONLY use untrimmed data for this plot.
        all_untrimmed_df = pd.read_csv(os.path.join(bg_cond_path, f'MASTER_psignifit_thresholds.csv'))
        # print(f"\nall_df:\n{all_df}")


        extra_text = None
        save_path = bg_cond_path

        all_df = all_untrimmed_df.copy()

        '''
        ax will have 5 positions for each ISI column, going:
        [space, incong scatter, means, cong scatter, space]
        '''


        # get list of all columns containing 'isi'
        not_isi_col_list = [col for col in all_df.columns if 'isi' not in col]
        isi_col_list = [col for col in all_df.columns if 'isi' in col]


        '''get data frame for scatterplots, with 'cong_' and 'incong_' columns for each isi column.'''

        # drop cong_col_name from not_isi_col_list
        not_isi_col_list.remove(cong_col_name)

        print(f"\nisi_col_list: {isi_col_list}")
        isi_cong_df = all_df.copy()
        isi_cong_df = isi_cong_df[isi_cong_df[cong_col_name] == 1]
        isi_cong_df = isi_cong_df.drop(columns=[cong_col_name])
        # add 'cong' to each column name in isi_col_list
        isi_cong_col_list = [f"cong_{col}" for col in isi_col_list]
        # rename columns in isi_col_list as isi_cong_col_list
        for idx, col_name in enumerate(isi_col_list):
            isi_cong_df.rename(columns={col_name: isi_cong_col_list[idx]}, inplace=True)
        print(f"\nisi_cong_df: ({list(isi_cong_df.columns)})\n{isi_cong_df}")

        isi_incong_df = all_df.copy()
        isi_incong_df = isi_incong_df[isi_incong_df[cong_col_name] == -1]
        isi_incong_df = isi_incong_df.drop(columns=[cong_col_name])
        # add 'incong' to each column name in isi_col_list, then rename columns in isi_incong_df
        isi_incong_col_list = [f"incong_{col}" for col in isi_col_list]
        # rename columns in isi_col_list as isi_incong_df
        for idx, col_name in enumerate(isi_col_list):
            isi_incong_df.rename(columns={col_name: isi_incong_col_list[idx]}, inplace=True)
        print(f"\nisi_incong_df: ({list(isi_incong_df.columns)})\n{isi_incong_df}")

        # alternate items from isi_cong_col_list and isi_incong_col_list
        all_isi_col_list = []
        for idx in range(len(isi_col_list)):
            all_isi_col_list.append(isi_incong_col_list[idx])
            all_isi_col_list.append(isi_cong_col_list[idx])
        print(f"\nall_isi_col_list: {all_isi_col_list}")

        # merge isi_cong_df and isi_incong_df,
        isi_cong_df.drop(columns=not_isi_col_list, inplace=True)
        isi_incong_df.drop(columns=not_isi_col_list, inplace=True)

        isi_cong_df.reset_index(drop=True, inplace=True)
        isi_incong_df.reset_index(drop=True, inplace=True)
        all_isi_df = pd.concat([isi_incong_df, isi_cong_df], axis=1, join='inner')
        print(f"\nall_isi_df: ({list(all_isi_df.columns)})\n{all_isi_df}")

        # change order of columns in isi_df to match all_col_list
        all_isi_df = all_isi_df[all_isi_col_list]
        print(f"\nall_isi_df: ({list(all_isi_df.columns)})\n{all_isi_df}")


        '''Make long df for lineplot, with isi_ms in one column, and cong_col_name in another column'''
        # make long_df, moving 'isi_' columns to single column 'isi_ms'
        cols_to_change = [col for col in all_df.columns if 'isi_ms_' in col]
        long_df = make_long_df(wide_df=all_df,
                               cols_to_keep=[cong_col_name],
                               cols_to_change=cols_to_change,
                               cols_to_change_show='probeLum',
                               new_col_name=isi_col_name, strip_from_cols='isi_ms_', verbose=verbose)

        # get a list of isi values in long_df
        isi_vals_list = long_df[isi_col_name].unique().tolist()
        isi_vals_str_list = [str(val) for val in isi_vals_list]

        # for each value in isi_vals_list, make a new list which is i*5+3
        isi_mean_x_pos_list = [i * 5 + 2 for i in range(len(isi_vals_list))]
        print(f"\nisi_mean_x_pos_list: {isi_mean_x_pos_list}")
        long_df['isi_x_pos'] = long_df[isi_col_name].map(dict(zip(isi_vals_list, isi_mean_x_pos_list)))

        # create a new column, 'isi_x_dodge_pos', with values from isi_x_pos column,
        # but if cong_col_name is -1, subtract .1, if cong_col_name is 1, add .1
        long_df['isi_x_dodge_pos'] = long_df['isi_x_pos']
        long_df.loc[long_df[cong_col_name] == -1, 'isi_x_dodge_pos'] = long_df['isi_x_dodge_pos'] - .1
        long_df.loc[long_df[cong_col_name] == 1, 'isi_x_dodge_pos'] = long_df['isi_x_dodge_pos'] + .1

        print(f"\nlong_df:\n{long_df}")


        '''plot lineplot with error bars'''
        fig, ax = plt.subplots()

        # plot means with error bars
        sns.lineplot(data=long_df,
                     x='isi_x_dodge_pos',
                     y='probeLum', hue=cong_col_name,
                     palette=sns.color_palette("tab10", n_colors=2),
                     linewidth=3,
                     errorbar='se',
                     err_style='bars',
                     err_kws={'capsize': 5, 'elinewidth': 2, 'capthick': 2},
                     ax=ax
                     )

        '''plot scatter plot, with pairs of datapionts joined (cong/incong)'''

        df = all_isi_df
        print(f"\ndf:\n{df}")

        # create jitter for x positions
        jitter = 0.0  # use 0.05 for jitter
        df_x_jitter = pd.DataFrame(np.random.normal(loc=0, scale=jitter, size=df.values.shape), columns=df.columns)

        # we are going to add to the jitter values to put them either side of the mean x pos
        isi_scatter_x_pos_list = []
        for val in isi_mean_x_pos_list:
            isi_scatter_x_pos_list.append(val - 1)
            isi_scatter_x_pos_list.append(val + 1)
        df_x_jitter += isi_scatter_x_pos_list

        print(f"\nisi_mean_x_pos_list: {isi_mean_x_pos_list}")
        print(f"\nnp.array(isi_scatter_x_pos_list):\n{np.array(isi_scatter_x_pos_list)}")
        print(f"\ndf_x_jitter:\n{df_x_jitter}")


        # plot scatter plot
        palette_tab10 = sns.color_palette("tab10", 10)
        for idx, col_name in enumerate(list(df.columns)):
            if idx % 2 == 0:
                this_colour = palette_tab10[0]
            else:
                this_colour = palette_tab10[1]
            ax.plot(df_x_jitter[col_name], df[col_name], 'o', alpha=.40, zorder=1, ms=4, mew=1, color=this_colour)


        # join scatter plot with lines
        for idx in range(0, len(df.columns), 2):
            ax.plot(df_x_jitter.loc[:, df.columns[idx:idx + 2]].T,
                    df.loc[:, df.columns[idx:idx + 2]].T,
                    color='grey', linewidth=0.5, linestyle='--', zorder=-1)

        ax.set_xlim(0, max(isi_mean_x_pos_list) + 2)
        # set x tick labels to isi_vals_str_list, which should start at 2, and go up in 5s
        labels_go_here = list(range(2, max(isi_mean_x_pos_list) + 2, 5))
        ax.set_xticks(labels_go_here,
                      labels=isi_vals_list)

        # decorate plot, x axis label, title and legend
        ax.set_xlabel('ISI (ms)')
        ax.set_ylabel('Probe Luminance')
        # todo: add separation and motion ms to title
        title_text = f"{bg_type}"
        if len(all_df[bg_dur_name].unique()) == 1:
            title_text += f", motion_dur:{all_df[bg_dur_name].unique()[0]}ms"
        else:
            raise ValueError(f"more than one motion_dur_ms value: {all_df[bg_dur_name].unique()}")
        if len(all_df['separation'].unique()) == 1:
            title_text += f", sep:{all_df['separation'].unique()[0]}px"
        else:
            raise ValueError(f"more than one separation value: {all_df['separation'].unique()}")
        ax.set_title(title_text)
        ax.legend(labels=['Incongruent', 'Congruent'])

        suptitle_text = f"{participant_name} thresholds & means of each run. {extra_text}"
        plt.suptitle(suptitle_text)

        fig_name = f"{participant_name}_joinedplot.png"
        if extra_text is not None:
            fig_name = f"{participant_name}_{extra_text}_joinedplot.png"
        if verbose:
            print(f"\n***saving joinedplot to {os.path.join(save_path, fig_name)}***")
        plt.savefig(os.path.join(save_path, fig_name))

        plt.show()



        # '''If there are interleaved conds (e.g., bg_motion_ms), do separate staircases for each of them'''
        # if not interleaved_col_list:  # do one staircase plt for all interleaved conditions
        #     rad_flow_line_plot(all_df=all_df_path, participant_name=participant_name,
        #                        trim_n=trim_n,
        #                        extra_text=None, save_path=bg_cond_path, show_plots=True, verbose=True)
        # elif len(interleaved_col_list) == 1:  # do one staircase plt for each interleaved condition
        #     all_df = pd.read_csv(all_df_path)
        #     interleaves_vals = all_df[interleaved_col_list[0]].unique().tolist()
        #     print(f"\n\n***separate mean_stair for each condition in {interleaved_col_list}***")
        #     for this_val in interleaves_vals:
        #         print(f"\n\n{this_val}")
        #         this_save_path = os.path.join(bg_cond_path, f"{interleaved_col_list[0]}_{this_val}")
        #         this_df = all_df[all_df[interleaved_col_list[0]] == this_val]
        #         print(f"this_df ({this_df.shape}):\n{this_df}")
        #         rad_flow_line_plot(all_df=this_df, participant_name=participant_name,
        #                            trim_n=trim_n,
        #                            extra_text=f"{interleaved_col_list[0]}_{this_val}",
        #                            save_path=this_save_path, show_plots=True, verbose=True)




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
