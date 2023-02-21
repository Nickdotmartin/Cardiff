import os
import pandas as pd
import datetime
from exp1a_psignifit_analysis import a_data_extraction, b3_plot_staircase, c_plots, \
    d_average_participant, e_average_exp_data, make_average_plots, conc_to_first_isi_col
from psignifit_tools import get_psignifit_threshold_df
from python_tools import which_path, running_on_laptop, switch_path

# """
# Tis script is to integrate Simon's sep 45 and Exp1 scores.
#
# get exp1a MASTER_psignifit_thr from sep45 and exp1.
#
# Combine them
#
# Run resp of analysis script.
#
# save results to
# C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\exp1a_data\Simon_w_sep45
# """
# todo: add Nick's set45 stuff to Exp1 folder.
# add: Add my relevant data from sep45 to Exp1a.



project_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff"

# # loop through run folders with first 4 scripts (a, get_psignifit_threshold_df, b3, c)
# # then run script d to get master lists and averages
# eyetracking, jitter_rgb, EXP1_split_probes, Exp1_double_dist, EXP1_sep4_5
# Exp4b_missing_probe\rotation, incoherent, radial, rotation, translation,
this_exp = r"EXP1_sep4_5"
exp_path = os.path.join(project_path, this_exp)

convert_path1 = os.path.normpath(exp_path)
if running_on_laptop():
    convert_path1 = switch_path(convert_path1, 'mac_oneDrive')

exp_path = convert_path1

print(f"exp_path: {exp_path}")
participant_name = 'Nick'  # 'Nick_sep0123', 'Nick_sep45', 'Nick_sep67', 'Nick_sep89' 'Simon', 'Nick'
exp_list = ['radial_v4', 'sep_45']
for idx, exp_name in enumerate(exp_list):
    if idx == 0:
        new_save_name = participant_name
    new_save_name = new_save_name + '_' + exp_name
print(f"new_save_name: {new_save_name}")

# make this before running script
new_save_dir = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\Nick_sep5_all"



p_master_lists_dirs = [r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\EXP1_sep4_5\Nick_sep45",
                       r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\Exp1_Jan23_radial_v4\Nick"]

results_df_list = ['MASTER_TM2_thresholds.csv',
                   'MASTER_ave_TM2_thresh.csv',
                   'MASTER_ave_TM2_thr_error_SE.csv',
                   ]

for results_df_name in results_df_list:
    print(f"\n\n\nresults_df_name: {results_df_name}")

    combine_results = []
    for exp_dir in p_master_lists_dirs:
        print(f"\n\nexp_dir: {exp_dir}")

        results_df_path = os.path.join(exp_dir, results_df_name)

        if os.path.isfile(results_df_path):
            print(f'found: {results_df_path}')
        else:
            print(f'\tmissing: {results_df_path}')
            continue


        results_df = pd.read_csv(results_df_path)

        if 'Unnamed: 0' in list(results_df):
            results_df.drop('Unnamed: 0', axis=1, inplace=True)

        if 'Concurrent' in list(results_df):
            # results_df = results_df.rename(columns={'Concurrent': 'ISI_-1'})
            results_df = results_df.rename(columns={'Concurrent': 'ISI_-1',
                                                    'ISI0': 'ISI_0',
                                                    'ISI2': 'ISI_2',
                                                    'ISI4': 'ISI_4',
                                                    'ISI6': 'ISI_6',
                                                    'ISI9': 'ISI_9',
                                                    'ISI12': 'ISI_12',
                                                    'ISI24': 'ISI_24'})

        # just take sep 5 data
        if 'separation' in list(results_df):
            results_df = results_df[results_df['separation'].isin([-5, 5])]

        # change all participant names to Nick
        rows, cols = results_df.shape
        # if 'participant' in list(results_df):
        # results_df['participant'] = ['Nick'] * rows


        # add probe orientation info
        if 'EXP1_sep4_5' in exp_dir:
            results_df['separation'] = [0] * rows
            # results_df['ori'] = ['tangent'] * rows
        # else:
            # results_df['ori'] = ['exp' if i < 0 else 'cont' for i in results_df['separation'].tolist()]


        # print(f"\nlist(results_df):\n{list(results_df)}")
        #
        print(f"\neditted results_df:\n{results_df}")

        combine_results.append(results_df)

    # join all stacks (run/group) data and save as master csv
    combined_data_df = pd.concat(combine_results, ignore_index=True)

    # sort df by separation
    combined_data_df = combined_data_df.sort_values(by=['separation'])

    # check that ISI_-1 is at last position
    combined_data_df = conc_to_first_isi_col(combined_data_df)
    combined_data_df.to_csv(os.path.join(new_save_dir, results_df_name), index=False)
    # if verbose:
    print(f'\ncombined {results_df_name}:\n{combined_data_df}')


all_df_path = os.path.join(new_save_dir, results_df_list[0])
p_ave_path = os.path.join(new_save_dir, results_df_list[1])
err_path = os.path.join(new_save_dir, results_df_list[2])
make_average_plots(all_df_path=all_df_path,
                   ave_df_path=p_ave_path,
                   error_bars_path=err_path,
                   # thr_col='newLum',
                   error_type='SE',
                   # ave_over_n=12,
                   sep_vals_list=[-5, 0, 5], # 0, 1, 2, 3, 6, 18, 20],
                   sep_name_list=['exp', 'tang', 'cont'],
                   n_trimmed=2,
                   split_1probe=False,
                   exp_ave=False,  # participant ave, not exp ave
                   show_plots=True, verbose=True)

# make_average_plots(all_df_path, ave_df_path, error_bars_path,
#                        thr_col='probeLum',
#                        ave_over_n=None,
#                        n_trimmed=None,
#                        error_type='SE',
#                        exp_ave=False,
#                        split_1probe=True,
#                        isi_name_list=['Concurrent', 'ISI 0', 'ISI 2', 'ISI 4',
#                                        'ISI 6', 'ISI 9', 'ISI 12', 'ISI 24'],
#                        sep_vals_list=[5], # 0, 1, 2, 3, 6, 18, 20],
#                        sep_name_list=[0, 1, 2, 3, 6, 18, '1probe'],
#                        heatmap_annot_fmt='{:.2f}',
#                        show_plots=True, verbose=True





# # participant_list = ['p1', 'p2']
# split_1probe = False
#
# n_runs = 12
#
# analyse_from_run = 1
#
# trim_list = []
#
# for p_idx, participant_name in enumerate(participant_list):
#     root_path = os.path.join(exp_path, participant_name)
#
#     # search to automatically get run_folder_names
#     dir_list = os.listdir(root_path)
#     run_folder_names = []
#     for i in range(n_runs):  # numbers 0 to 11
#         check_dir = f'{participant_name}_{i+analyse_from_run}'   # numbers 1 to 12
#         if check_dir in dir_list:
#             run_folder_names.append(check_dir)
#
#     if len(run_folder_names) > 0:
#         print("running analysis for:")
#         for i in run_folder_names:
#             print(i)
#     else:
#         print("no run folders found")
#
#     trim_n = None
#     if len(run_folder_names) == 12:
#         trim_n = 2
#     elif len(run_folder_names) > 12:
#         # trim_n = 2
#         if len(run_folder_names) % 2 == 0:  # if even
#             trim_n = int((len(run_folder_names)-12)/2)
#         else:
#             raise ValueError(f"for this exp you have {len(run_folder_names)} runs, set rules for trimming.")
#     trim_list.append(trim_n)
#
#
#     # trim_n = None
#     # lum_col = 'probeLum'
#     print(f"\n\ntrim_list: {trim_list}, trim_n: {trim_n}\n\n")
#
#     '''d'''
#     d_average_participant(root_path=root_path, run_dir_names_list=run_folder_names,
#                           trim_n=trim_n, error_type='SE')
#
#     all_df_path = os.path.join(root_path, f'MASTER_TM{trim_n}_thresholds.csv')
#     p_ave_path = os.path.join(root_path, f'MASTER_ave_TM{trim_n}_thresh.csv')
#     err_path = os.path.join(root_path, f'MASTER_ave_TM{trim_n}_thr_error_SE.csv')
#     if trim_n is None:
#         all_df_path = os.path.join(root_path, 'MASTER_psignifit_thresholds.csv')
#         p_ave_path = os.path.join(root_path, 'MASTER_ave_thresh.csv')
#         err_path = os.path.join(root_path, 'MASTER_ave_thr_error_SE.csv')
#
#     p_ave_df = pd.read_csv(p_ave_path)
#     print(f"p_ave_df:\n{p_ave_df}")
#
#     isi_vals_list = [int(i[4:]) for i in list(p_ave_df.columns)[1:]]
#     isi_name_list = [f"conc" if i == -1 else f"ISI_{i}" for i in isi_vals_list]
#
#     sep_vals_list = list(p_ave_df['separation'])
#     sep_name_list = [f"1probe" if i == 20 else i for i in sep_vals_list]
#     print(f"isi_name_list:\n{isi_name_list}")
#     print(f"isi_vals_list:\n{isi_vals_list}")
#     print(f"sep_vals_list:\n{sep_vals_list}")
#     print(f"sep_name_list:\n{sep_name_list}")
#
#     make_average_plots(all_df_path=all_df_path,
#                        ave_df_path=p_ave_path,
#                        error_bars_path=err_path,
#                        error_type='SE',
#                        n_trimmed=trim_n,
#                        ave_over_n=len(run_folder_names),
#                        split_1probe=split_1probe,
#                        isi_name_list=isi_name_list,
#                        sep_vals_list=sep_vals_list,
#                        sep_name_list=sep_name_list,
#                        exp_ave=participant_name,  # participant ave, not exp ave
#                        heatmap_annot_fmt='.0f',  # use '.3g' for 3 significant figures, '.2f' for 2dp, '.0f' for int.
#                        show_plots=True, verbose=True)


print('\nexp1a_analysis_pipe finished\n')
