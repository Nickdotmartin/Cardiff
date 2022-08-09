import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from rad_flow_psignifit_analysis import split_df_alternate_rows, plot_runs_ave_w_errors


'''
To understand why there is a difference in rad_flow_2 data at sep2 when analysed
using the last luminance value (e.g., step25), yet there is no difference when
using all 25 datapoints with psignifit; this script will look for order effects.
That is, do participants show adaptation to repeated radial flow motion.
This method was suggest by Simon (From Teams > Simon, 24/03/2022).

For each ISI, sep=2, plot all the congruent and incongruent staircases onto one plot,
including a ave staircase.
If there is an order effect then the staircases will diverge over time.

1. loop through all conditions.
2. load data for staircases.
3. add data to master list for calculating ave

4. # two options here
a) add data to plot in loop, one line at a time.
b) create plot from master list with multiple lines simultaneously

5. add ave lines


I could base process on b3_plot_staircase.  But I need to adapt it.
Don't show final thr hline
Don't have multiple panels
(although I could do all stairs and ave on different plots if its too busy)
'''


# exp_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\rad_flow_2"
exp_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\exp1a_data"
exp_path = os.path.normpath(exp_path)
# participant_list = ['Nick', 'Simon']
participant_list = ['aa']

# stair_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
stair_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

# all_isi_list = [1, 4, 6, 9]
all_isi_list = [-1, 0, 1, 2, 4, 6, 9, 12, 24]
all_isi_names_list = [f'ISI_{i}' for i in all_isi_list]

get_median = True
ave_type = 'mean'
if get_median:
    ave_type = 'median'

verbose = True
show_plots = False

n_runs = 12
# if the first folder to analyse is 1, p_idx_plus = 1.  If the first folder is 5, use 5 etc.
p_idx_plus = 1

'''
Part 1 - loop through all runs to make MASTER_p_trial_data_df
'''

for p_idx, participant_name in enumerate(participant_list):

    root_path = os.path.join(exp_path, participant_name)

    # # search to automatically get run_folder_names
    dir_list = os.listdir(root_path)
    run_folder_names = []
    for i in range(12):  # numbers 0 to 11
        check_dir = f'{participant_name}_{i+p_idx_plus}'   # numbers 1 to 12
        if check_dir in dir_list:
            run_folder_names.append(check_dir)

    print(f'run_folder_names: {run_folder_names}')

    MASTER_p_trial_data_list = []

    for run_idx, run_dir in enumerate(run_folder_names):

        # add run number , e.g., add five to access Nick_5 on the zeroth iteration
        r_idx_plus = run_idx + p_idx_plus

        print(f'\nrun_idx {run_idx+1}: running analysis for '
              f'{participant_name}, {run_dir}, {participant_name}_{r_idx_plus}')
        save_path = f'{root_path}{os.sep}{run_dir}'

        # # search to automatically get updated isi_list
        dir_list = os.listdir(save_path)
        run_isi_list = []
        for isi in all_isi_list:
            check_dir = f'ISI_{isi}_probeDur2'
            if check_dir in dir_list:
                run_isi_list.append(isi)
        run_isi_names_list = [f'ISI_{i}' for i in run_isi_list]

        print(f'run_isi_list: {run_isi_list}')

        # don't delete this (p_name = participant_name),
        # needed to ensure names go name1, name2, name3 not name1, name12, name123
        p_name = participant_name
        p_name = f'{participant_name}_{r_idx_plus}'

        # run_data_path = os.path.join(save_path, 'ALL_ISIs_sorted.xlsx')
        run_data_path = os.path.join(save_path, 'RUNDATA-SORTED.xlsx')

        run_data_df = pd.read_excel(run_data_path, engine='openpyxl',
                                    usecols=["stair", "step",
                                             "ISI", "separation",
                                             "congruent", "probeLum"]
                                    )
        print(f"run_data_df: {run_data_df.columns.to_list()}\n{run_data_df}")

        rows, cols = run_data_df.shape
        run_data_df.insert(0, 'stack', [run_idx+1] * rows)

        MASTER_p_trial_data_list.append(run_data_df)

        # join all stacks (runs/groups) data and save as master csv
        MASTER_p_trial_data_df = pd.concat(MASTER_p_trial_data_list, ignore_index=True)

        MASTER_p_trial_data_df.to_csv(os.path.join(root_path, f'MASTER_p_trial_data_df.csv'), index=False)
        if verbose:
            print(f'\nMASTER_p_trial_data_df:\n{MASTER_p_trial_data_df}')

    '''
    Part 2 - 
    1. Calculate ave thr for each step for congruent and incongruent data.
    2. loop through MASTER_p_trial_data_df for each ISI/sep combination.
    3. make plot showing all 12 congruent and incongruent staircases on the same plot (different colours)
    and ave cong and incong ontop.
    '''

    MASTER_p_trial_data_df = pd.read_csv(os.path.join(root_path, f'MASTER_p_trial_data_df.csv'))
    print(f'\nMASTER_p_trial_data_df ({MASTER_p_trial_data_df.shape}):\n{MASTER_p_trial_data_df}')

    isi_list = MASTER_p_trial_data_df['ISI'].unique()
    print(f"isi_list: {isi_list}")


    # loop through isi values
    for isi_idx, isi in enumerate(isi_list):
        # get df for this isi only
        isi_df = MASTER_p_trial_data_df[MASTER_p_trial_data_df['ISI'] == isi]
        print(f"\n{isi_idx}. ISI: {isi} ({isi_df.shape})"
              # f"\n{isi_df}"
              )

        sep_list = isi_df['separation'].unique()
        print(f"sep_list: {sep_list}")

        # loop through sep values
        for sep_idx, sep in enumerate(sep_list):
            # get df for this sep only
            sep_df = isi_df[isi_df['separation'] == sep]
            print(f"\n{sep_idx}. sep {sep} ({sep_df.shape}):"
                  # f"\n{sep_df}"
                  )

            # initialise plot
            fig, ax = plt.subplots()

            # # get ave values for each step
            this_sep_df = sep_df[['stack', 'step', 'congruent', 'probeLum']]
            print(f"this_sep_df ({this_sep_df.shape}):\n{this_sep_df}")
            if get_median:
                ave_step_thr_df = this_sep_df.groupby(['congruent', 'step'], sort=False).median()
            else:
                ave_step_thr_df = this_sep_df.groupby(['congruent', 'step'], sort=False).mean()
            ave_step_thr_df.reset_index(drop=False, inplace=True)
            ave_step_thr_df.drop('stack', axis=1, inplace=True)
            print(f"ave_step_thr_df ({ave_step_thr_df.shape}):\n{ave_step_thr_df}")
            wide_ave_step_thr_df = ave_step_thr_df.pivot(index='step', columns='congruent', values='probeLum')
            column_names = ['Incongruent', 'Congruent']
            wide_ave_step_thr_df.columns = column_names
            print(f"wide_ave_step_thr_df ({wide_ave_step_thr_df.shape}):\n{wide_ave_step_thr_df}")


            stack_list = sep_df['stack'].unique()
            print(f"stack_list: {sep_list}")

            # loop through stack values
            for stack_idx, stack in enumerate(stack_list):
                # get df for this stack only
                stack_df = sep_df[sep_df['stack'] == stack]
                print(f"\n{stack_idx}. stack {stack} ({stack_df.shape}):"
                      # f"\n{stack_df}"
                      )

                this_stack_df = stack_df[['step', 'congruent', 'probeLum']]
                print(f"this_stack_df ({this_stack_df.shape}):\n{this_stack_df}")

                # I now have the data I need - reshape it so cong and incong are different columns
                wide_df = this_stack_df.pivot(index='step', columns='congruent', values='probeLum')
                column_names = ['Incongruent', 'Congruent']
                wide_df.columns = column_names
                print(f"wide_df ({wide_df.shape}):\n{wide_df}")

                total_runs = len(stack_list)

                for idx, name in enumerate(column_names):

                    my_colour = 'pink'
                    ave_colour = 'red'
                    if name == 'Congruent':
                        my_colour = 'lightblue'
                        ave_colour = 'blue'

                    # # make plot
                    ax.errorbar(x=list(range(25)), y=wide_df[name],
                                # marker='.', lw=2, elinewidth=.7,
                                # capsize=cap_size,
                                color=my_colour
                                )
                    fig.suptitle(f"All run stairs\n{participant_name} with {ave_type}: ISI{isi}, sep{sep}")

                    ax.set_xlabel('step (25 per condition, per run)')
                    ax.set_ylabel('ProbeLum')

            for idx, name in enumerate(column_names):

                my_colour = 'pink'
                ave_colour = 'red'
                if name == 'Congruent':
                    my_colour = 'lightblue'
                    ave_colour = 'blue'

                # # make plot
                ax.errorbar(x=list(range(25)), y=wide_ave_step_thr_df[name],
                            # marker='.', lw=2, elinewidth=.7,
                            # capsize=cap_size,
                            color=ave_colour
                            )


                # artist for legend
                st1 = mlines.Line2D([], [], color='lightblue',
                                    # marker='v',
                                    # linestyle="dashed",
                                    markersize=4, label='Congruent')
                st2 = mlines.Line2D([], [], color='pink',
                                    # marker='o',
                                    # linestyle="dotted",
                                    markersize=4, label='Incongruent')
                ave_cong_line = mlines.Line2D([], [], color='red',
                                              marker=None,
                                              linewidth=.5,
                                              label=f'Congruent ({ave_type})')
                ave_incong_line = mlines.Line2D([], [], color='blue',
                                                marker=None,
                                                linewidth=.5,
                                                label=f'Incongruent ({ave_type})')
                ax.legend(handles=[st1, ave_cong_line, st2, ave_incong_line], fontsize=6)

            if get_median:
                save_name = f"all_run_stairs_{participant_name}_isi{isi}_sep{sep}_median.png"
            else:
                save_name = f"all_run_stairs_{participant_name}_isi{isi}_sep{sep}_mean.png"

            print(save_name)
            plt.savefig(os.path.join(root_path, save_name))

            plt.show()








print('\n***finished order effects script page***')