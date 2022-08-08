import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from rad_flow_psignifit_analysis import split_df_alternate_rows


'''
To understand why there is a difference in rad_flow_2 data at sep2 when analysed
using the last luminance value (e.g., step25), yet there is no difference when
using all 25 datapoints with psignifit; this script will look for order effects.
That is, do participants show adaptation to repeated radial flow motion.
This method was suggest by Simon (From Teams > Simon, 24/03/2022).

For each ISI, sep=2, plot all the congruent and incongruent staircases onto one plot,
including a mean staircase.
If there is an order effect then the staircases will diverge over time.

1. loop through all conditions.
2. load data for staircases.
3. add data to master list for calculating mean

4. # two options here
a) add data to plot in loop, one line at a time.
b) create plot from master list with multiple lines simultaneously

5. add mean lines


I could base process on b3_plot_staircase.  But I need to adapt it.
Don't show final thr hline
Don't have multiple panels
(although I could do all stairs and mean on different plots if its too busy)
'''


def plot_order_effects(all_data_path, thr_col='newLum', resp_col='trial_response',
                       show_plots=True, save_plots=True, verbose=True):
    """
    plot_order_effects: for each/a particular condition (ISI/sep combination);
    plot all the congruent and incongruent staircases onto one plot, and add a mean staircase(?).
    If there is an order effect then the staircases will diverge over time.

    :param all_data_path: path to the all_data xlsx file.
    :param thr_col: (default newLum) name of the column showing the threshold
        (e.g., varied by the staircase).
    :param resp_col: (default: 'trial_response') name of the column showing
        (accuracy per trial).
    :param show_plots: whether to display plots on-screen.
    :param save_plots: whether to save the plots.
    :param verbose: If True, will print progress to screen.

    :return:
    one figure per isi value - saved as Staircases_{isi_name}
    n_reversals.csv - number of reversals per stair - used in c_plots
    """
    print("\n*** running plot_order_effects() ***\n")

    save_path, xlsx_name = os.path.split(all_data_path)

    # open all_data file.  use engine='openpyxl' for xlsx files.
    # For other experiments it might be easier not to do use cols as they might be different.
    if xlsx_name[-3:] == 'csv':
        all_data_df = pd.read_csv(all_data_path)
    else:
        all_data_df = pd.read_excel(all_data_path, engine='openpyxl',
                                    # usecols=["ISI", "stair", "step", "separation",
                                    #          "flow_dir", "probe_jump", "corner",
                                    #          "newLum", "probeLum", "trial_response"]
                                    # note: don't use usecols as some exps have 'NewLum' but radflow doesn't
                                    )

    # get list of isi and stair values to loop through
    stair_list = all_data_df['stair'].unique()
    isi_list = all_data_df['ISI'].unique()
    # get isi string for column names
    isi_name_list = [f'ISI_{i}' for i in isi_list]

    trials, columns = np.shape(all_data_df)
    trials_per_stair = int(trials / len(isi_list) / len(stair_list))

    if verbose:
        print(f"all_data_df:\n{all_data_df}")
        print(f"{len(isi_list)} isi values and {len(stair_list)} stair values")
        print(f"isi_list: {isi_list}")
        print(f"isi_name_list: {isi_name_list}")
        print(f"stair_list: {stair_list}")
        print(f"trials_per_stair: {trials_per_stair}")

    '''the eighth plot is the last thr for each sep (+sep, -sep and mean).  
    get data from psignifit_thresholds.csv and reshape here'''
    thr_csv_name = os.path.join(save_path, 'psignifit_thresholds.csv')
    psignifit_thr_df = pd.read_csv(thr_csv_name)
    if verbose:
        print(f'\npsignifit_thr_df:\n{psignifit_thr_df}')

    # remove extra columns
    if 'stair' in list(psignifit_thr_df.columns):
        psignifit_thr_df = psignifit_thr_df.drop(['stair'], axis=1)

    if 'stair_names' in list(psignifit_thr_df.columns):
        psignifit_thr_df = psignifit_thr_df.drop(['stair_names'], axis=1)

    if 'congruent' in list(psignifit_thr_df.columns):
        psignifit_thr_df = psignifit_thr_df.drop(['congruent'], axis=1)

    if 'separation' in list(psignifit_thr_df.columns):
        sep_list = psignifit_thr_df.pop('separation').tolist()
    print(f'sep_list: {sep_list}')

    psignifit_thr_df.columns = isi_name_list

    # split into pos_sep, neg_sep and mean of pos and neg.
    psig_cong_sep_df, psig_incong_sep_df = split_df_alternate_rows(psignifit_thr_df)
    psig_thr_mean_df = pd.concat([psig_cong_sep_df, psig_incong_sep_df]).groupby(level=0).mean()

    # add sep column in
    rows, cols = psig_thr_mean_df.shape
    if len(sep_list) == rows * 2:
        # takes every other item
        sep_list = sep_list[::2]
    else:
        # todo: for exp2_bloch I don't have double the number of rows, so I should do something here.
        raise ValueError(f"I dunno why the number of rows ({rows}) isn't double "
                         f"the sep_list ({sep_list})")
    separation_title = [f'sep_{i}' for i in sep_list]
    if verbose:
        print(f'sep_list: {sep_list}')
        print(f"separation_title: {separation_title}")

    psig_thr_mean_df.insert(0, 'separation', sep_list)
    psig_cong_sep_df.insert(0, 'separation', sep_list)
    psig_incong_sep_df.insert(0, 'separation', sep_list)
    if verbose:
        print(f'\npsig_cong_sep_df:\n{psig_cong_sep_df}')
        print(f'\npsig_incong_sep_df:\n{psig_incong_sep_df}')
        print(f'\npsig_thr_mean_df:\n{psig_thr_mean_df}')

    # make empty arrays to save reversal n_reversals
    n_reversals_np = np.zeros(shape=[len(stair_list), len(isi_list)])

    # loop through isi values
    for isi_idx, isi in enumerate(isi_list):

        # get df for this isi only
        isi_df = all_data_df[all_data_df['ISI'] == isi]
        isi_name = isi_name_list[isi_idx]
        print(f"\n{isi_idx}. staircase for ISI: {isi}, {isi_name}")

        # initialise 8 plot figure - last plot will be blank
        # # this is a figure showing n_reversals per staircase condition.
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12, 6))
        ax_counter = 0

        for row_idx, row in enumerate(axes):
            for col_idx, ax in enumerate(row):
                print(f'\nrow: {row_idx}, col: {col_idx}: {ax}')

                # for the first six plots...
                if ax_counter < 6:

                    # # get pairs of stairs (e.g., [[18, -18], [6, -6], ...etc)
                    stair_even_cong = ax_counter * 2  # 0, 2, 4, 6, 8, 10
                    stair_even_cong_df = isi_df[isi_df['stair'] == stair_even_cong]
                    final_lum_even_cong = \
                        stair_even_cong_df.loc[stair_even_cong_df['step'] == trials_per_stair - 1, thr_col].item()
                    n_reversals_even_cong = trials_per_stair - stair_even_cong_df[resp_col].sum()

                    stair_odd_incong = (ax_counter * 2) + 1  # 1, 3, 5, 7, 9, 11
                    stair_odd_incong_df = isi_df[isi_df['stair'] == stair_odd_incong]
                    final_lum_odd_incong = \
                        stair_odd_incong_df.loc[stair_odd_incong_df['step'] == trials_per_stair - 1, thr_col].item()
                    n_reversals_odd_incong = trials_per_stair - stair_odd_incong_df[resp_col].sum()

                    # append n_reversals to n_reversals_np to save later.
                    n_reversals_np[stair_even_cong - 1, isi_idx] = n_reversals_even_cong
                    n_reversals_np[stair_odd_incong - 1, isi_idx] = n_reversals_odd_incong

                    if verbose:
                        print(f'\nstair_even_cong_df (stair={stair_even_cong}, '
                              f'isi_name={isi_name}:\n{stair_even_cong_df}')
                        print(f"final_lum_even_cong: {final_lum_even_cong}")
                        print(f"n_reversals_even_cong: {n_reversals_even_cong}")
                        print(f'\nstair_odd_incong_df (stair={stair_odd_incong}, '
                              f'isi_name={isi_name}:\n{stair_odd_incong_df}')
                        print(f"final_lum_odd_incong: {final_lum_odd_incong}")
                        print(f"n_reversals_odd_incong: {n_reversals_odd_incong}")

                    fig.suptitle(f'Staircases and reversals for isi {isi_name}')

                    # plot thr per step for even_cong numbered stair
                    sns.lineplot(ax=axes[row_idx, col_idx], data=stair_even_cong_df,
                                 x='step', y=thr_col, color='tab:blue',
                                 marker="o", markersize=4)
                    # line for final newLum
                    ax.axhline(y=final_lum_even_cong, linestyle="-.", color='tab:blue')
                    # text for n_reversals
                    ax.text(x=0.25, y=0.8, s=f'{n_reversals_even_cong} reversals',
                            color='tab:blue',
                            # needs transform to appear with rest of plot.
                            transform=ax.transAxes, fontsize=12)

                    # plot thr per step for odd_incong numbered stair
                    sns.lineplot(ax=axes[row_idx, col_idx], data=stair_odd_incong_df,
                                 x='step', y=thr_col, color='tab:red',
                                 marker="v", markersize=5)
                    ax.axhline(y=final_lum_odd_incong, linestyle="--", color='tab:red')
                    ax.text(x=0.25, y=0.9, s=f'{n_reversals_odd_incong} reversals',
                            color='tab:red',
                            # needs transform to appear with rest of plot.
                            transform=ax.transAxes, fontsize=12)

                    ax.set_title(f'{isi_name} {separation_title[ax_counter]}')
                    ax.set_xticks(np.arange(0, trials_per_stair, 5))
                    ax.set_ylim([0, 110])

                    # artist for legend
                    if ax_counter == 0:
                        st1 = mlines.Line2D([], [], color='tab:blue',
                                            marker='v',
                                            markersize=5, label='Congruent')
                        st1_last_val = mlines.Line2D([], [], color='tab:blue',
                                                     linestyle="--", marker=None,
                                                     label='Cong: last val')
                        st2 = mlines.Line2D([], [], color='tab:red',
                                            marker='o',
                                            markersize=5, label='Incongruent')
                        st2_last_val = mlines.Line2D([], [], color='tab:red',
                                                     linestyle="-.", marker=None,
                                                     label='Incong: last val')
                        ax.legend(handles=[st1, st1_last_val, st2, st2_last_val],
                                  fontsize=6, loc='lower right')

                elif ax_counter == 6:
                    """use the psignifit from each stair pair (e.g., 18, -18) to
                    get the mean threshold for each sep condition.
                    """
                    if verbose:
                        print("Seventh plot")
                        print(f'psig_thr_mean_df:\n{psig_thr_mean_df}')

                    isi_thr_mean_df = pd.concat([psig_thr_mean_df['separation'], psig_thr_mean_df[isi_name]],
                                                axis=1, keys=['separation', isi_name])
                    if verbose:
                        print(f'isi_thr_mean_df:\n{isi_thr_mean_df}')

                    # line plot for thr1, th2 and mean thr
                    sns.lineplot(ax=axes[row_idx, col_idx], data=isi_thr_mean_df,
                                 x='separation', y=isi_name, color='lightgreen',
                                 linewidth=3)
                    sns.lineplot(ax=axes[row_idx, col_idx], data=psig_cong_sep_df,
                                 x='separation', y=isi_name, color='blue',
                                 linestyle="--")
                    sns.lineplot(ax=axes[row_idx, col_idx], data=psig_incong_sep_df,
                                 x='separation', y=isi_name, color='red',
                                 linestyle="dotted")

                    # artist for legend
                    cong_thr = mlines.Line2D([], [], color='blue', linestyle="--",
                                             marker=None, label='Congruent thr')

                    incong_thr = mlines.Line2D([], [], color='red', linestyle="dotted",
                                               marker=None, label='Incongruent thr')
                    mean_thr = mlines.Line2D([], [], color='lightgreen', linestyle="solid",
                                             marker=None, label='mean thr')
                    ax.legend(handles=[cong_thr, incong_thr, mean_thr],
                              fontsize=6, loc='lower right')

                    # decorate plot
                    ax.set_title(f'{isi_name} psignifit thresholds')
                    ax.set_xticks([0, 1, 2, 3, 6, 18])
                    ax.set_xticklabels([0, 1, 2, 3, 6, 18])
                    ax.set_xlabel('Probe separation')
                    ax.set_ylim([0, 110])
                    ax.set_ylabel('Probe Luminance')

                else:
                    fig.delaxes(ax=axes[row_idx, col_idx])

                ax_counter += 1

        plt.tight_layout()

        # show and close plots
        if save_plots:
            save_name = f'staircases_{isi_name}.png'
            plt.savefig(os.path.join(save_path, save_name))

        if show_plots:
            plt.show()
        plt.close()

    # save n_reversals to csv for use in script_c figure 5
    n_reversals_df = pd.DataFrame(n_reversals_np, columns=isi_name_list)
    n_reversals_df.insert(0, 'stair', stair_list)
    n_reversals_df.set_index('stair', inplace=True)
    if verbose:
        print(f'n_reversals_df:\n{n_reversals_df}')
    n_reversals_df.to_csv(os.path.join(save_path, 'n_reversals.csv'))

    print("\n***finished plot_order_effectss()***\n")


exp_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\rad_flow_2"
exp_path = os.path.normpath(exp_path)
participant_list = ['Simon']  # , 'Nick']

stair_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
all_isi_list = [1, 4, 6, 9]
all_isi_names_list = [f'ISI_{i}' for i in all_isi_list]

verbose = True
show_plots = False

n_runs = 12
# if the first folder to analyse is 1, p_idx_plus = 1.  If the first folder is 5, use 5 etc.
p_idx_plus = 1

for p_idx, participant_name in enumerate(participant_list):

    root_path = os.path.join(exp_path, participant_name)

    # # search to automatically get run_folder_names
    dir_list = os.listdir(root_path)
    run_folder_names = []
    for i in range(12):  # numbers 0 to 11
        check_dir = f'{participant_name}_{i+p_idx_plus}'   # numbers 1 to 12
        if check_dir in dir_list:
            run_folder_names.append(check_dir)
    #
    # print(f'run_folder_names: {run_folder_names}')
    #
    # for run_idx, run_dir in enumerate(run_folder_names):
    #
    #     # add run number , e.g., add five to access Nick_5 on the zeroth iteration
    #     r_idx_plus = run_idx + p_idx_plus
    #
    #     print(f'\nrun_idx {run_idx+1}: running analysis for '
    #           f'{participant_name}, {run_dir}, {participant_name}_{r_idx_plus}')
    #     save_path = f'{root_path}{os.sep}{run_dir}'
    #
    #     # # search to automatically get updated isi_list
    #     dir_list = os.listdir(save_path)
    #     run_isi_list = []
    #     for isi in all_isi_list:
    #         check_dir = f'ISI_{isi}_probeDur2'
    #         if check_dir in dir_list:
    #             run_isi_list.append(isi)
    #     run_isi_names_list = [f'ISI_{i}' for i in run_isi_list]
    #
    #     print(f'run_isi_list: {run_isi_list}')
    #
    #     # don't delete this (p_name = participant_name),
    #     # needed to ensure names go name1, name2, name3 not name1, name12, name123
    #     p_name = participant_name
    #     p_name = f'{participant_name}_{r_idx_plus}'
    #
    #     run_data_path = os.path.join(save_path, 'ALL_ISIs_sorted.xlsx')
    #     run_data_df = pd.read_excel(run_data_path, engine='openpyxl',
    #                                 usecols=["ISI", "stair", "stair_name",
    #                                          "step", "separation", "congruent",
    #                                          "flow_dir", "probe_jump", "corner",
    #                                          "probeLum", "trial_response"]
    #                                 )
    #     print(f"run_data_df: {run_data_df.columns.to_list()}\n{run_data_df}")
    #
    #
    #     plot_order_effects(run_data_path, thr_col='probeLum', show_plots=show_plots, verbose=verbose)

    # not sure I actually want to use trimmed means?
    trim_n = None
    if len(run_folder_names) == 12:
        trim_n = 2
    print(f'\ntrim_n: {trim_n}')

    all_df_path = os.path.join(root_path, f'MASTER_TM{trim_n}_thresholds.csv')
    p_ave_path = os.path.join(root_path, f'MASTER_ave_TM{trim_n}_thresh.csv')
    err_path = os.path.join(root_path, f'MASTER_ave_TM{trim_n}_thr_error_SE.csv')
    if trim_n is None:
        all_df_path = os.path.join(root_path, f'MASTER_psignifit_thresholds.csv')
        p_ave_path = os.path.join(root_path, 'MASTER_ave_thresh.csv')
        err_path = os.path.join(root_path, 'MASTER_ave_thr_error_SE.csv')

    all_df = pd.read_csv(all_df_path)
    print(f'all_df: {all_df}')

    '''
    I guess this is the stuff that I can actually wrap into a function.
    goes after d_average_participants.
    But it needs to use all of the data that goes into getting psignifit threshold, 
    not the thresholds themselves'''

    all_data_df = all_df

    df_headers = list(all_data_df)
    print(f'df_headers: {df_headers}')

    # get list of isi and stair values to loop through
    stair_list = all_data_df['stair_names'].unique()
    isi_list = all_data_df['ISI'].unique()
    # get isi string for column names
    isi_name_list = [f'ISI_{i}' for i in isi_list]

    trials, columns = np.shape(all_data_df)
    trials_per_stair = int(trials / len(isi_list) / len(stair_list))

    if verbose:
        print(f"all_data_df:\n{all_data_df}")
        print(f"{len(isi_list)} isi values and {len(stair_list)} stair values")
        print(f"isi_list: {isi_list}")
        print(f"isi_name_list: {isi_name_list}")
        print(f"stair_list: {stair_list}")
        print(f"trials_per_stair: {trials_per_stair}")







print('\n***finished order effects script page***')