import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import scipy.stats as stats

'''
This first part of the script is to make:
    MASTER_all_Prelim.csv,
    MASTER_ave_all_Prelim.csv &
    MASTER_err_all_Prelim.csv files for each participant.

For each participant, it will loop through all preliminary folders (e.g. bg70, bg350, etc.),
open the MASTER_TM2_thresholds.csv, add a 'prelim' column, and append it to the MASTER_all_Prelim.csv file.
Then from this calculate the mean and standared error for each prelim, ['stair_names', 'neg_sep', 'separation'] and 'congruent' combination.

See below for code for various analyses such as comparing mean staircases, comparing mean thr with t-tests and plotting group mean staircases.

'''



# # set paths
# root_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\rad_flow_Sept23"
#
# exp_list = ['rad_flow_6_rings', 'rad_flow_6_rings_OLED', 'rad_flow_7_spokes']
#
# # loop though each experiment
# for exp in exp_list:
#     print(f"\nexp: {exp}")
#     exp_path = os.path.join(root_path, exp)
#
#     # get list of participants
#     participant_list = os.listdir(exp_path)
#     print(f"\nparticipant_list: {participant_list}")
#
#     # loop through each participant
#     for participant_name in participant_list:
#
#         p_to_skip_list = ['flow_dots', 'Nick_match_rf2_16082023']
#
#         # skip any participants in the p_to_skip_list
#         if participant_name in p_to_skip_list:
#             print(f"\nSkipping {participant_name}")
#             continue
#
#
#         print(f"\nparticipant_name: {participant_name}")
#
#         participant_path = os.path.join(exp_path, participant_name)
#
#         # empty list to append dataframes to
#         master_df_list = []
#
#         # check for flow_type
#         flow_type_list = ['flow_rings', 'flow_dots']
#
#         '''There might be times where both exist but there only should be one flow type'''
#         for flow_type in flow_type_list:
#             if flow_type in os.listdir(participant_path):
#                 print(f"\nflow_type: {flow_type}")
#                 flow_type_path = os.path.join(participant_path, flow_type)
#
#
#                 # loop though preliminary folders
#                 prelim_list = ['bg70', 'bg200', 'bg350']
#
#                 for prelim in prelim_list:
#
#                     if prelim in os.listdir(flow_type_path):
#                         print(f"\nprelim: {prelim}")
#                         prelim_path = os.path.join(flow_type_path, prelim)
#
#                         # open MASTER_TM2_thresholds.csv or MASTER_psignifit_thresholds
#
#                         if os.path.isfile(os.path.join(prelim_path, 'MASTER_TM2_thresholds.csv')):
#                             prelim_master_path = os.path.join(prelim_path, 'MASTER_TM2_thresholds.csv')
#
#
#                         elif os.path.isfile(os.path.join(prelim_path, 'MASTER_psignifit_thresholds.csv')):
#                             prelim_master_path = os.path.join(prelim_path, 'MASTER_psignifit_thresholds.csv')
#
#                         else:
#                             print(f"\nNo MASTER file found in {prelim_path}")
#                             raise FileNotFoundError
#
#                         prelim_master_df = pd.read_csv(prelim_master_path)
#                         print(f"\nprelim_master_df:\n{prelim_master_df.head()}")
#
#                         # add prelim column
#                         prelim_int = int(prelim[2:])
#                         prelim_master_df.insert(0, 'prelim', prelim_int)
#                         print(f"\nprelim_master_df:\n{prelim_master_df.head()}")
#
#                         # append to master_df_list
#                         master_df_list.append(prelim_master_df)
#
#         # concat master_df_list
#         master_df = pd.concat(master_df_list)
#         print(f"\nmaster_df:\n{master_df.head()}")
#
#         # save to MASTER_all_Prelim.csv
#         master_all_prelim_path = os.path.join(participant_path, 'MASTER_all_Prelim.csv')
#         master_df.to_csv(master_all_prelim_path, index=False)
#
#         # make a MASTER_ave_all_prelim.csv and MASTER_err_all_prelim.csv
#         '''
#         mean_df should give me the mean of the threshold_col across all 'stack', when grouped by 'prelim', ['stair_names', 'neg_sep', 'separation'] and 'congruent'.
#         error_df should give me the standard error of the threshold_col across all 'stack', when grouped by 'prelim', ['stair_names', 'neg_sep', 'separation'] and 'congruent'.
#         '''
#
#         # get threshold_col
#         '''Find the name of column containing the substring 'ISI_
#         If there is more than one, then raise an error.'''
#         threshold_col_list = [col for col in master_df.columns if 'ISI_' in col]
#
#         if len(threshold_col_list) > 1:
#             raise ValueError(f"\nMore than one column containing 'ISI_' in {master_df.columns}")
#         else:
#             threshold_col = threshold_col_list[0]
#             print(f"\nthreshold_col: {threshold_col}")
#
#
#         mean_df = master_df.groupby(['prelim', 'stair_names', 'neg_sep', 'separation', 'congruent'])[threshold_col].mean().reset_index()
#         print(f"\nmean_df:\n{mean_df.head()}")
#
#         error_df = master_df.groupby(['prelim', 'stair_names', 'neg_sep', 'separation', 'congruent'])[threshold_col].sem().reset_index()
#         print(f"\nerror_df:\n{error_df.head()}")
#
#         # save mean and error dfs
#         master_ave_all_prelim_path = os.path.join(participant_path, 'MASTER_ave_all_Prelim.csv')
#         mean_df.to_csv(master_ave_all_prelim_path, index=False)
#
#         master_err_all_prelim_path = os.path.join(participant_path, 'MASTER_err_all_Prelim.csv')
#         error_df.to_csv(master_err_all_prelim_path, index=False)


# '''
# This bit of the script will produce mean staircase plots for two participants, overlaid on the same plot.
# This is done with MASTER_p_trial_data_df.csv files, from the prelim dirs.
# For participant 1, use red and orange for two prelims, with dashes and dots for cong and incong.
# For participant 2, use blue and purple for two prelims, with dashes and dots for cong and incong.
#
# This version is too busy, for each sep it has: 2 (participants) x 2 (prelims) x 2 (cong) mean lines, plus
# 12 (stacks) x 2 (participants) x 2 (prelims) x 2 (cong) smaller individual run lines.
# see below for a version which just had 2 (participants) on separate plots for sep, prelim and cong.
#
# '''
# get_median = False
# verbose = True
# show_plots = True
#
# ave_type = 'mean'
# if get_median:
#     ave_type = 'median'
#
# save_dir = f'{ave_type}_stairs'
#
#
# exp_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\rad_flow_Sept23\rad_flow_7_spokes"
# p_name_1 = 'OLED_circles_rings_quartSpd'
# p_name_2 = 'OLED_circles_rings_quartSpd_v2'
#
# p1_colours = ['blue', 'green']
# p2_colours = ['purple', 'orange']
#
#
# save_dir_path = os.path.join(exp_path, 'group_plots', f'{p_name_1}_v_{p_name_2}', 'ave_stairs')
# # make save_dir if it doesn't exist
# if not os.path.isdir(os.path.join(exp_path, 'group_plots')):
#     os.mkdir(os.path.join(exp_path, 'group_plots'))
# if not os.path.isdir(os.path.join(exp_path, 'group_plots', f'{p_name_1}_v_{p_name_2}')):
#     os.mkdir(os.path.join(exp_path, 'group_plots', f'{p_name_1}_v_{p_name_2}'))
# if not os.path.isdir(save_dir_path):
#     os.mkdir(save_dir_path)
# combined_df_list = []
#
# # loop through participants
# for p_name in [p_name_1, p_name_2]:
#     print(f"\np_name: {p_name}")
#
#     for prelim in ['bg70', 'bg350']:
#         print(f"prelim: {prelim}")
#         df_path = os.path.join(exp_path, p_name, 'flow_rings', prelim, 'MASTER_p_trial_data_df.csv')
#         p_trial_data_df = pd.read_csv(df_path)
#
#         # insert p_name and prelim columns
#         p_trial_data_df.insert(0, 'p_name', p_name)
#         prelim_int = int(prelim[2:])
#         p_trial_data_df.insert(1, 'prelim', prelim_int)
#
#         print(f"p_trial_data_df:\n{p_trial_data_df.head()}")
#         combined_df_list.append(p_trial_data_df)
#
# # concat combined_df_list
# combined_df = pd.concat(combined_df_list)
# print(f"\n\ncombined_df: {combined_df.shape}\n{combined_df}")
#
# # make mean staircase plot
# isi_list = combined_df['ISI'].unique()
# print(f"\nisi_list: {isi_list}")
#
# sep_list = combined_df['separation'].unique()
# print(f"sep_list: {sep_list}")
#
# # stack list should be 1 to 12, one for each run.
# stack_list = combined_df['stack'].unique()
# print(f"stack_list: {stack_list}")
#
# cond_type_list = ['Incongruent', 'Congruent']
# print(f"cond_type_list: {cond_type_list}")
#
# # loop through isi values
# for isi_idx, isi in enumerate(isi_list):
#     # get df for this isi only
#     isi_df = combined_df[combined_df['ISI'] == isi]
#     print(f"\n{isi_idx}. ISI: {isi} ({isi_df.shape})"
#           # f"\n{isi_df}"
#           )
#
#
#     # loop through sep values
#     for sep_idx, sep in enumerate(sep_list):
#         # get df for this sep only
#         sep_df = isi_df[isi_df['separation'] == sep]
#         print(f"ISI: {isi}, {sep_idx}. sep {sep} ({sep_df.shape}):"
#               # f"\n{sep_df}"
#               )
#
#         # initialise plot
#         fig, ax = plt.subplots()
#
#         # # get ave values for each step
#         this_sep_df = sep_df[['p_name', 'prelim', 'stack', 'step', 'congruent', 'probeLum']]
#
#         # loop through p_names in p_name column and make a df
#         for p_name in this_sep_df['p_name'].unique():
#             p_name_df = this_sep_df[this_sep_df['p_name'] == p_name]
#
#             # drop p_name column
#             # p_name_df.drop('p_name', axis=1, inplace=True)
#             p_name_df = p_name_df[['prelim', 'stack', 'step', 'congruent', 'probeLum']]
#
#             if p_name == p_name_1:
#                 these_colours = p1_colours
#             elif p_name == p_name_2:
#                 these_colours = p2_colours
#             else:
#                 raise ValueError(f"p_name {p_name} not recognised")
#
#             print(f"\nisi: {isi}, sep: {sep}, p_name: {p_name}, p_name_df ({p_name_df.shape}):"
#                   # f"\n{p_name_df}"
#                   )
#
#
#             # loop through prelim column and make a df
#             for prelim in p_name_df['prelim'].unique():
#
#                 this_linestyle = 'dashed'
#                 if prelim == 350:
#                     this_linestyle = 'dotted'
#
#
#                 prelim_df = p_name_df[p_name_df['prelim'] == prelim]
#
#                 # drop prelim column
#                 # prelim_df.drop('prelim', axis=1, inplace=True)
#                 prelim_df = prelim_df[['stack', 'step', 'congruent', 'probeLum']]
#
#                 print(f"\nisi: {isi}, sep: {sep}, p_name: {p_name}, prelim: {prelim}, prelim_df ({prelim_df.shape}):"
#                       # f"\n{prelim_df}"
#                       )
#
#                 if get_median:
#                     ave_step_thr_df = prelim_df.groupby(['congruent', 'step'], sort=False).median()
#                 else:
#                     ave_step_thr_df = prelim_df.groupby(['congruent', 'step'], sort=False).mean()
#
#
#                 ave_step_thr_df.reset_index(drop=False, inplace=True)
#                 ave_step_thr_df.drop('stack', axis=1, inplace=True)
#                 print(f"ave_step_thr_df ({ave_step_thr_df.shape}):"
#                       # f"\n{ave_step_thr_df}"
#                       )
#                 wide_ave_step_thr_df = ave_step_thr_df.pivot(index='step', columns='congruent', values='probeLum')
#                 wide_ave_step_thr_df.columns = cond_type_list
#                 print(f"wide_ave_step_thr_df ({wide_ave_step_thr_df.shape}):"
#                       # f"\n{wide_ave_step_thr_df}"
#                       )
#
#
#                 # loop through stack values
#                 for stack_idx, stack in enumerate(stack_list):
#                     # get df for this stack only
#                     stack_df = prelim_df[prelim_df['stack'] == stack]
#
#                     this_stack_df = stack_df[['step', 'congruent', 'probeLum']]
#
#                     print(f"\n{stack_idx}. stack {stack} ({stack_df.shape}):"
#                           # f"\n{stack_df}"
#                           )
#
#
#                     # I now have the data I need - reshape it so cong and incong are different columns
#                     wide_df = this_stack_df.pivot(index='step', columns='congruent', values='probeLum')
#                     wide_df.columns = cond_type_list
#                     print(f"wide_df ({wide_df.shape}):"
#                           # f"\n{wide_df}"
#                           )
#
#                     total_runs = len(stack_list)
#
#                     for idx, cong_type in enumerate(cond_type_list):
#
#                         this_colour = these_colours[0]
#                         if cong_type == 'Congruent':
#                             this_colour = these_colours[1]
#
#                         # # make plot
#                         ax.errorbar(x=list(range(25)), y=wide_df[cong_type],
#                                     # lw=2,
#                                     color=this_colour, alpha=.2,
#                                     linestyle=this_linestyle,
#                                     )
#
#                 for idx, cong_type in enumerate(cond_type_list):
#
#                     this_colour = these_colours[0]
#                     if cong_type == 'Congruent':
#                         this_colour = these_colours[1]
#
#                     # # make plot
#                     ax.errorbar(x=list(range(25)), y=wide_ave_step_thr_df[cong_type],
#                                 lw=3,
#                                 color=this_colour,
#                                 linestyle=this_linestyle,
#                                 )
#         # decorate plots
#         fig.suptitle(f"{p_name_1} vs {p_name_2}\nAll run stairs: {ave_type}: ISI{isi}, sep{sep}")
#
#         ax.set_xlabel('step (25 per condition, per run)')
#         ax.set_ylabel('ProbeLum')
#
#         # artist for legend
#         p1_cong = mlines.Line2D([], [], color=p1_colours[0],
#                             markersize=4, label='P1 Congruent')
#         p1_incong = mlines.Line2D([], [], color=p1_colours[1],
#                             markersize=4, label='P1 Incongruent')
#         p2_cong = mlines.Line2D([], [], color=p2_colours[0],
#                             markersize=4, label='P2 Congruent')
#         p2_incong = mlines.Line2D([], [], color=p2_colours[1],
#                             markersize=4, label='P2 Incongruent')
#         bg70 = mlines.Line2D([], [], color='black', linestyle='dashed',
#                              label='bg70')
#         bg350 = mlines.Line2D([], [], color='black', linestyle='dotted',
#                               label='bg350')
#         ax.legend(handles=[p1_cong, p1_incong, p2_cong, p2_incong, bg70, bg350], fontsize=6)
#
#         if get_median:
#             save_name = f"all_run_stairs_isi{isi}_sep{sep}_median.png"
#         else:
#             save_name = f"all_run_stairs_isi{isi}_sep{sep}_mean.png"
#
#
#         print(f"\nSaving to {os.path.join(save_dir_path, save_name)}")
#         plt.savefig(os.path.join(save_dir_path, save_name))
#
#         plt.show()

'''
This bit of the script will produce mean staircase plots for two participants, overlaid on the same plot.
This is done with MASTER_p_trial_data_df.csv files, from the prelim dirs.
For participant 1, use red and orange for two prelims, with dashes and dots for cong and incong.
For participant 2, use blue and purple for two prelims, with dashes and dots for cong and incong.

Produce different plots for each sep, prelim and cond_type.
A single plots will just have two means (participants) and 12 runs
'''
get_median = False
verbose = True
show_plots = True
just_last_15_steps = True
restrict_y = True  # Simon just wants 0 to 15

ave_type = 'mean'
if get_median:
    ave_type = 'median'

save_dir = f'{ave_type}_stairs'


exp_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\rad_flow_Sept23\rad_flow_7_spokes"
p_name_1 = 'OLED_circles_rings_quartSpd'
p_name_2 = 'OLED_circles_rings_quartSpd_v2'

# p1_colours = ['blue', 'green']
# p2_colours = ['purple', 'orange']
cong_colours = ['blue', 'orange']
incong_colours = ['red', 'green']

save_dir_path = os.path.join(exp_path, 'group_plots', f'{p_name_1}_v_{p_name_2}', 'ave_stairs')
# make save_dir if it doesn't exist
if not os.path.isdir(os.path.join(exp_path, 'group_plots')):
    os.mkdir(os.path.join(exp_path, 'group_plots'))
if not os.path.isdir(os.path.join(exp_path, 'group_plots', f'{p_name_1}_v_{p_name_2}')):
    os.mkdir(os.path.join(exp_path, 'group_plots', f'{p_name_1}_v_{p_name_2}'))
if not os.path.isdir(save_dir_path):
    os.mkdir(save_dir_path)
combined_df_list = []

# loop through participants
for p_name in [p_name_1, p_name_2]:
    print(f"\np_name: {p_name}")

    for prelim in ['bg70', 'bg350']:
        print(f"prelim: {prelim}")
        df_path = os.path.join(exp_path, p_name, 'flow_rings', prelim, 'MASTER_p_trial_data_df.csv')
        p_trial_data_df = pd.read_csv(df_path)

        # insert p_name and prelim columns
        p_trial_data_df.insert(0, 'p_name', p_name)
        prelim_int = int(prelim[2:])
        p_trial_data_df.insert(1, 'prelim', prelim_int)

        # print(f"p_trial_data_df:\n{p_trial_data_df.head()}")
        combined_df_list.append(p_trial_data_df)

# concat combined_df_list
combined_df = pd.concat(combined_df_list)

# drop 'stair' and re-order columns, into order with which I will select them
combined_df = combined_df[['ISI', 'separation', 'prelim', 'congruent', 'p_name', 'stack', 'step', 'probeLum']]

if just_last_15_steps:
    combined_df = combined_df[combined_df['step'] > 9]

print(f"\n\ncombined_df: {combined_df.shape}\n{combined_df}")

all_column_names = list(combined_df.columns)
print(f"\nall_column_names: {all_column_names}")

# make mean staircase plot
isi_list = combined_df['ISI'].unique()
print(f"\nisi_list: {isi_list}")

sep_list = combined_df['separation'].unique()
print(f"sep_list: {sep_list}")

# stack list should be 1 to 12, one for each run.
stack_list = combined_df['stack'].unique()
print(f"stack_list: {stack_list}")


# reshape data so probeLum for each participant is in a different column
wide_combined_df = combined_df.pivot_table(index=['ISI', 'separation', 'prelim', 'congruent', 'stack', 'step'],
                                             columns='p_name', values='probeLum')
wide_combined_df.reset_index(drop=True, inplace=True)
print(f"\nwide_combined_df: {wide_combined_df.shape}\n{wide_combined_df}")




# loop through isi values
for isi in combined_df['ISI'].unique():
    # get df for this isi only
    isi_df = combined_df[combined_df['ISI'] == isi]
    isi_df = isi_df[['separation', 'prelim', 'congruent', 'p_name', 'stack', 'step', 'probeLum']]
    print(f"\nISI: {isi} ({isi_df.shape})"
          # f"\n{isi_df}"
          )

    # loop through sep values
    for sep in isi_df['separation'].unique():
        # get df for this sep only and drop sep col
        sep_df = isi_df[isi_df['separation'] == sep]
        sep_df = sep_df[['prelim', 'congruent', 'p_name', 'stack', 'step', 'probeLum']]
        print(f"ISI: {isi}, sep {sep} ({sep_df.shape}):"
              # f"\n{sep_df}"
              )

        # loop through prelim column and make a df
        for prelim in sep_df['prelim'].unique():

            # get df for this prelim only and drop prelim column
            prelim_df = sep_df[sep_df['prelim'] == prelim]
            prelim_df = prelim_df[['congruent', 'p_name', 'stack', 'step', 'probeLum']]
            print(f"isi: {isi}, sep: {sep}, prelim: {prelim}, prelim_df ({prelim_df.shape}):"
                  # f"\n{prelim_df}"
                  )

            for cond_type in prelim_df['congruent'].unique():

                # get label for this cond_type
                this_cond_type = 'congruent'
                these_colours = cong_colours
                if cond_type == -1:
                    this_cond_type = 'incongruent'
                    these_colours = incong_colours

                # get df for this cond_type only and drop cond type column
                cond_type_df = prelim_df[prelim_df['congruent'] == cond_type]
                cond_type_df = cond_type_df[['p_name', 'stack', 'step', 'probeLum']]
                print(f"isi: {isi}, sep: {sep}, prelim: {prelim}, this_cond_type: {this_cond_type}, cond_type_df ({cond_type_df.shape}):"
                      f"\n{cond_type_df}"
                      )

                '''Make plots with thin lines for each stack (run) and thick lines for averages'''
                # initialise plot
                fig, ax = plt.subplots()

                # make plot of individual runs
                for stack in cond_type_df['stack'].unique():
                    stack_df = cond_type_df[cond_type_df['stack'] == stack]
                    sns.lineplot(x='step', y='probeLum', hue='p_name', data=stack_df, ax=ax,
                                 linewidth=1, alpha=.3, legend=False,
                                 palette=these_colours)


                # # get ave values for each step
                if get_median:
                    ave_step_thr_df = cond_type_df.groupby(['p_name', 'step'], sort=False).median()
                else:
                    ave_step_thr_df = cond_type_df.groupby(['p_name', 'step'], sort=False).mean()

                ave_step_thr_df.reset_index(drop=False, inplace=True)
                ave_step_thr_df.drop('stack', axis=1, inplace=True)
                print(f"ave_step_thr_df ({ave_step_thr_df.shape}):"
                      # f"\n{ave_step_thr_df}"
                      )

                # reshape df so that the probeLum for each participant is a different column, indexed by step.
                wide_ave_step_thr_df = ave_step_thr_df.pivot(index='step', columns='p_name', values='probeLum')
                print(f"wide_ave_step_thr_df ({wide_ave_step_thr_df.shape}):"
                      # f"\n{wide_ave_step_thr_df}"
                      )

                # make plot of mean values
                sns.lineplot(data=wide_ave_step_thr_df, ax=ax,
                             dashes=False, linewidth=3,
                             legend=True, palette=these_colours)




                # decorate plots
                title_text = f"{ave_type} stairs. ISI{isi}, sep{sep}, prelim: {prelim}, {this_cond_type}"
                if just_last_15_steps:
                    title_text = f"{ave_type} stairs. ISI{isi}, sep{sep}, prelim: {prelim}, {this_cond_type}, last 15 steps"
                fig.suptitle(title_text)
                ax.set_xlabel('step (25 per condition, per run)')
                ax.set_ylabel('ProbeLum')

                if restrict_y:
                    ax.set_ylim([0, 15])

                # save plots
                save_name = f"{ave_type}_stairs_isi{isi}_sep{sep}_prelim{prelim}_{this_cond_type}.png"
                if just_last_15_steps:
                    save_name = f"{ave_type}_stairs_isi{isi}_sep{sep}_prelim{prelim}_{this_cond_type}_last15.png"
                print(f"\nSaving to {os.path.join(save_dir_path, save_name)}")
                plt.savefig(os.path.join(save_dir_path, save_name))

                plt.show()




'''
This part of the script will compare two participants using their MASTER_all_Prelim.csv files.
For each condition (neg_sep x cong), compare all thresholds with t-tests.
'''



'''This part of the script will produce group mean plots for groups of participants, saved into a group folder.
As well as the group mean lines shown with bold lines, add scatter points for each participant.
'''








