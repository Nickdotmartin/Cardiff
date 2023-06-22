import os
import pandas as pd
import datetime
from exp1a_psignifit_analysis import a_data_extraction, b3_plot_staircase, c_plots, \
    d_average_participant, e_average_exp_data, make_average_plots, conc_to_first_isi_col
from psignifit_tools import get_psignifit_threshold_df
from python_tools import which_path, running_on_laptop, switch_path

# """
# Tis script is to integrate sep 45 and Exp1 scores for me, Simon, Kim, Tony (and Kristian if he does it)
#
# get exp1a MASTER_psignifit_thr from sep45 and exp1.
#
# Combine them
#
# Run resp of analysis script.
#
# save results to
# C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\exp1a_data\combined_exp1_sep45
# """

project_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff"
combined_dir_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\combined_exp1_sep45"

convert_dict = {'Simon': {'exp1_p_name': 'bb', 'sep45_p_name': 'Simon', 'sep45_exp': 'EXP1_sep4_5'},
                'Nick': {'exp1_p_name': 'Nick', 'sep45_p_name': 'Nick', 'sep45_exp': 'EXP1_sep4_5'},
                'Kim': {'exp1_p_name': 'ee', 'sep45_p_name': 'Kim', 'sep45_exp': 'Exp1_Jan23_sep45'},
                # 'Tony': {'exp1_p_name': 'aa', 'sep45_p_name': 'Tony', 'sep45_exp': 'Exp1_Jan23_sep45'},
                # 'Kristian': {'exp1_p_name': 'dd', 'sep45_p_name': 'Kristian', 'sep45_exp': 'Exp1_Jan23_sep45'}
                }

all_df_name = 'MASTER_TM2_thresholds.csv'
p_ave_name = 'MASTER_ave_TM2_thresh.csv'
err_name = 'MASTER_ave_TM2_thr_error_SE.csv'

df_list = [all_df_name, p_ave_name, err_name]

# loop through dictionary
for p_name in list(convert_dict.keys()):
    
    print(f"\n\n\np_name: {p_name}")

    # for each participant, loop through df_list, loading the relevant df from exp1 and sep45 and combining them
    for this_df_name in df_list:

        print(f"\nthis_df_name: {this_df_name}")

        # get exp1 this_df for this participant
        exp1_p_name = convert_dict[p_name]['exp1_p_name']
        exp1_path = os.path.join(project_path, 'exp1a_data', exp1_p_name)
        exp1_this_df_path = os.path.join(exp1_path, this_df_name)
        exp1_this_df = pd.read_csv(exp1_this_df_path)
        if 'Unnamed: 0' in list(exp1_this_df):
            exp1_this_df.drop('Unnamed: 0', axis=1, inplace=True)

        if 'Concurrent' in list(exp1_this_df):
            exp1_this_df = exp1_this_df.rename(columns={'Concurrent': 'ISI_-1',
                                                        'ISI0': 'ISI_0',
                                                        'ISI2': 'ISI_2',
                                                        'ISI4': 'ISI_4',
                                                        'ISI6': 'ISI_6',
                                                        'ISI9': 'ISI_9',
                                                        'ISI12': 'ISI_12',
                                                        'ISI24': 'ISI_24'})
        exp1_this_df = conc_to_first_isi_col(exp1_this_df)
        print(f"exp1_this_df: {exp1_this_df}")

        # get sep45 this_df for this participant
        sep45_p_name = convert_dict[p_name]['sep45_p_name']
        sep45_exp = convert_dict[p_name]['sep45_exp']
        sep45_path = os.path.join(project_path, sep45_exp, sep45_p_name)
        sep45_this_df_path = os.path.join(sep45_path, this_df_name)
        sep45_this_df = pd.read_csv(sep45_this_df_path)
        if 'Unnamed: 0' in list(sep45_this_df):
            sep45_this_df.drop('Unnamed: 0', axis=1, inplace=True)

        if 'Concurrent' in list(sep45_this_df):
            sep45_this_df = sep45_this_df.rename(columns={'Concurrent': 'ISI_-1',
                                                          'ISI0': 'ISI_0',
                                                          'ISI2': 'ISI_2',
                                                          'ISI4': 'ISI_4',
                                                          'ISI6': 'ISI_6',
                                                          'ISI9': 'ISI_9',
                                                          'ISI12': 'ISI_12',
                                                          'ISI24': 'ISI_24'})
        sep45_this_df = conc_to_first_isi_col(sep45_this_df)
        print(f"sep45_this_df: {sep45_this_df}")

        # remove separation conditions from sep45 if they are already in exp1
        sep45_this_df = sep45_this_df[~sep45_this_df['separation'].isin(exp1_this_df['separation'])]
        print(f"sep45_this_df: {sep45_this_df}")

        # combine exp1 and sep45 this_df and sort by separation
        combined_this_df = pd.concat([exp1_this_df, sep45_this_df], axis=0)
        combined_this_df.sort_values(by=['separation'], inplace=True)
        print(f"combined_this_df: {combined_this_df}")

        # save combined_this_df to participant dir in combined_dir_path
        # make participant dir if it doesn't exist
        combined_p_path = os.path.join(combined_dir_path, p_name)
        if not os.path.exists(combined_p_path):
            os.makedirs(combined_p_path)
        combined_this_df_path = os.path.join(combined_p_path, this_df_name)
        combined_this_df.to_csv(combined_this_df_path, index=False)

'''
For analysis I removed my extra conditions (sep7-11) after e_averages.
I also had to comment out fig 1a from make_average plots (participants) because it was throwing an error.'''







#################################################################
    # '''get all exp1 data'''
    # # find and load exp1a MASTER_psignifit_thr
    # exp1_p_name = convert_dict[p_name]['exp1_p_name']
    # exp1_path = os.path.join(project_path, 'exp1a_data', exp1_p_name)
    # exp1_master_thr_path = os.path.join(exp1_path, 'MASTER_psignifit_thr.csv')
    # exp1_master_thr_df = pd.read_csv(exp1_master_thr_path)
    # if 'Unnamed: 0' in list(exp1_master_thr_df):
    #     exp1_master_thr_df.drop('Unnamed: 0', axis=1, inplace=True)
    #
    # if 'Concurrent' in list(exp1_master_thr_df):
    #     exp1_master_thr_df = exp1_master_thr_df.rename(columns={'Concurrent': 'ISI_-1',
    #                                                             'ISI0': 'ISI_0',
    #                                                             'ISI2': 'ISI_2',
    #                                                             'ISI4': 'ISI_4',
    #                                                             'ISI6': 'ISI_6',
    #                                                             'ISI9': 'ISI_9',
    #                                                             'ISI12': 'ISI_12',
    #                                                             'ISI24': 'ISI_24'})
    # exp1_master_thr_df = conc_to_first_isi_col(exp1_master_thr_df)
    # print(f"exp1_master_thr_df: {exp1_master_thr_df}")
    #
    # # find and load sep45 MASTER_psignifit_thr
    # sep45_p_name = convert_dict[p_name]['sep45_p_name']
    # sep45_exp = convert_dict[p_name]['sep45_exp']
    # sep45_path = os.path.join(project_path, sep45_exp, sep45_p_name)
    # sep45_master_thr_path = os.path.join(sep45_path, 'MASTER_psignifit_thr.csv')
    # sep45_master_thr_df = pd.read_csv(sep45_master_thr_path)
    # if 'Unnamed: 0' in list(sep45_master_thr_df):
    #     sep45_master_thr_df.drop('Unnamed: 0', axis=1, inplace=True)
    #
    # if 'Concurrent' in list(sep45_master_thr_df):
    #     sep45_master_thr_df = sep45_master_thr_df.rename(columns={'Concurrent': 'ISI_-1',
    #                                                               'ISI0': 'ISI_0',
    #                                                               'ISI2': 'ISI_2',
    #                                                               'ISI4': 'ISI_4',
    #                                                               'ISI6': 'ISI_6',
    #                                                               'ISI9': 'ISI_9',
    #                                                               'ISI12': 'ISI_12',
    #                                                               'ISI24': 'ISI_24'})
    # sep45_master_thr_df = conc_to_first_isi_col(sep45_master_thr_df)
    # print(f"sep45_master_thr_df: {sep45_master_thr_df}")
    #
    # # remove separation conditions from sep45 if they are already in exp1
    # sep45_master_thr_df = sep45_master_thr_df[~sep45_master_thr_df['separation'].isin(exp1_master_thr_df['separation'])]
    # print(f"sep45_master_thr_df: {sep45_master_thr_df}")
    #
    # # combine exp1 and sep45
    # combined_master_thr_df = pd.concat([exp1_master_thr_df, sep45_master_thr_df])
    #
    # # sort by separation
    # combined_master_thr_df = combined_master_thr_df.sort_values(by=['separation'])
    #
    # # save combined_master_thr_df in new participant dir in combined_exp1_sep45
    # # make new participant dir unless it already exists
    # combined_exp1_sep45_path = os.path.join(project_path, 'combined_exp1_sep45')
    # new_participant_path = os.path.join(combined_exp1_sep45_path, p_name)
    # if not os.path.exists(new_participant_path):
    #     os.mkdir(new_participant_path)
    # combined_master_thr_path = os.path.join(new_participant_path, 'MASTER_psignifit_thr.csv')
    # combined_master_thr_df.to_csv(combined_master_thr_path, index=False)

    # run rest of analysis script





    

#
#
#
# # # loop through run folders with first 4 scripts (a, get_psignifit_threshold_df, b3, c)
# # # then run script d to get master lists and averages
# # eyetracking, jitter_rgb, EXP1_split_probes, Exp1_double_dist, EXP1_sep4_5
# # Exp4b_missing_probe\rotation, incoherent, radial, rotation, translation,
# this_exp = r"EXP1_sep4_5\OLD_NICK_ORIG"
# exp_path = os.path.join(project_path, this_exp)
#
# convert_path1 = os.path.normpath(exp_path)
# if running_on_laptop():
#     convert_path1 = switch_path(convert_path1, 'mac_oneDrive')
#
# exp_path = convert_path1
#
# print(f"exp_path: {exp_path}")
# participant_name = ['Nick_sep0123', 'Nick_sep45', 'Nick_sep67', 'Nick_sep89', 'Nicksep_18_20']  # 'Simon', 'Nick'
# exp_list = ['radial_v4', 'sep_45']
# for idx, exp_name in enumerate(exp_list):
#     if idx == 0:
#         new_save_name = participant_name
#     new_save_name = new_save_name + '_' + exp_name
# print(f"new_save_name: {new_save_name}")
#
# # make this before running script
# new_save_dir = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\combined_exp1_sep45"
#
#
#
# p_master_lists_dirs = [r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\EXP1_sep4_5\Nick_sep45",
#                        r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\Exp1_Jan23_radial_v4\Nick"]
#
# results_df_list = ['MASTER_TM2_thresholds.csv',
#                    'MASTER_ave_TM2_thresh.csv',
#                    'MASTER_ave_TM2_thr_error_SE.csv',
#                    ]
#
# for results_df_name in results_df_list:
#     print(f"\n\n\nresults_df_name: {results_df_name}")
#
#     combine_results = []
#     for exp_dir in p_master_lists_dirs:
#         print(f"\n\nexp_dir: {exp_dir}")
#
#         results_df_path = os.path.join(exp_dir, results_df_name)
#
#         if os.path.isfile(results_df_path):
#             print(f'found: {results_df_path}')
#         else:
#             print(f'\tmissing: {results_df_path}')
#             continue
#
#
#         results_df = pd.read_csv(results_df_path)
#
#         if 'Unnamed: 0' in list(results_df):
#             results_df.drop('Unnamed: 0', axis=1, inplace=True)
#
#         if 'Concurrent' in list(results_df):
#             # results_df = results_df.rename(columns={'Concurrent': 'ISI_-1'})
#             results_df = results_df.rename(columns={'Concurrent': 'ISI_-1',
#                                                     'ISI0': 'ISI_0',
#                                                     'ISI2': 'ISI_2',
#                                                     'ISI4': 'ISI_4',
#                                                     'ISI6': 'ISI_6',
#                                                     'ISI9': 'ISI_9',
#                                                     'ISI12': 'ISI_12',
#                                                     'ISI24': 'ISI_24'})
#
#         # just take sep 5 data
#         if 'separation' in list(results_df):
#             results_df = results_df[results_df['separation'].isin([-5, 5])]
#
#         # change all participant names to Nick
#         rows, cols = results_df.shape
#         # if 'participant' in list(results_df):
#         # results_df['participant'] = ['Nick'] * rows
#
#
#         # # add probe orientation info
#         # if 'EXP1_sep4_5' in exp_dir:
#         #     results_df['separation'] = [0] * rows
#         #     # results_df['ori'] = ['tangent'] * rows
#         # # else:
#         #     # results_df['ori'] = ['exp' if i < 0 else 'cont' for i in results_df['separation'].tolist()]
#
#
#         # print(f"\nlist(results_df):\n{list(results_df)}")
#         #
#         print(f"\neditted results_df:\n{results_df}")
#
#         combine_results.append(results_df)
#
#     # join all stacks (run/group) data and save as master csv
#     combined_data_df = pd.concat(combine_results, ignore_index=True)
#
#     # sort df by separation
#     combined_data_df = combined_data_df.sort_values(by=['separation'])
#
#     # check that ISI_-1 is at last position
#     combined_data_df = conc_to_first_isi_col(combined_data_df)
#     combined_data_df.to_csv(os.path.join(new_save_dir, results_df_name), index=False)
#     # if verbose:
#     print(f'\ncombined {results_df_name}:\n{combined_data_df}')
#
#
# all_df_path = os.path.join(new_save_dir, results_df_list[0])
# p_ave_path = os.path.join(new_save_dir, results_df_list[1])
# err_path = os.path.join(new_save_dir, results_df_list[2])
# make_average_plots(all_df_path=all_df_path,
#                    ave_df_path=p_ave_path,
#                    error_bars_path=err_path,
#                    # thr_col='newLum',
#                    error_type='SE',
#                    # ave_over_n=12,
#                    sep_vals_list=[-5, 0, 5], # 0, 1, 2, 3, 6, 18, 20],
#                    sep_name_list=['exp', 'tang', 'cont'],
#                    n_trimmed=2,
#                    split_1probe=False,
#                    exp_ave=False,  # participant ave, not exp ave
#                    show_plots=True, verbose=True)


print('\ncombine_Exp1_sep45_analysis finished\n')
