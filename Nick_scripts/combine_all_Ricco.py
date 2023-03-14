import os
import pandas as pd
import datetime
from exp1a_psignifit_analysis import a_data_extraction, b3_plot_staircase, c_plots, \
    d_average_participant, e_average_exp_data, make_average_plots, conc_to_first_isi_col
from psignifit_tools import get_psignifit_threshold_df
from python_tools import which_path, running_on_laptop, switch_path

"""
This script is to combine Ricco scores from versions 4, 5 and 6.

For each participant name, loop through each folder and combine individual output run files.

New data to be saved in Exp3_Ricco_all

Should end up with 12 runs per participant, but for me, simon and Kris, these new 12 might contain data from more than 1 version.

I can then use data in this folder for getting RA measurements or for Vis field stuff.  
"""
project_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff"

v4_exp = 'Exp3_Ricco_NM_v4'
v4_participants = ['Kim', 'Kris', 'Nick', 'Simon']

v5_exp = 'Exp3_Ricco_NM_v5'
v5_participants = ['Nick']

v6_exp = 'Exp3_Ricco_v6'
v6_participants = ['Kristian', 'Simon']

same_name_dict = {'Kristian': 'Kris'}

participant_names = list(set(v4_participants + v5_participants + v6_participants))
# remove names I am going to change
participant_names = [i for i in participant_names if i not in same_name_dict.values()]
print(f"participant_names: {participant_names}")

exp_names = [v4_exp, v5_exp, v6_exp]
print(f"exp_names: {exp_names}")

n_runs = 12

# make this before running script
Ricco_all_dir = 'Exp3_Ricco_all'

files_report_dict = {}

print(list(range(1, n_runs+1)))

# loop through runs (e.g., do run 1s for all participants, then runs 2s etc)
for this_run in list(range(1, n_runs+1)):

    print(f"\nlooking for run: {this_run}")


    for p_name in participant_names:
        print(f"\n\np_name: {p_name}")

        # check incase I want to combine files with different p_name
        if p_name in same_name_dict.keys():
            alias = same_name_dict[p_name]
            names_to_check = [p_name, alias]
        else:
            names_to_check = [p_name]

        run_outputs = []

        for this_name in names_to_check:
            print(f"\np_name: {p_name} ({this_name})")

            for this_exp in exp_names:
                # print(f"this_exp: {this_exp}")

                p_name_dir = os.path.join(project_path, this_exp, this_name)

                if os.path.isdir(p_name_dir):
                    print(f"\nfound: p_name: {p_name} ({this_name}), this_exp: {this_exp}")

                    # there are a couple of possible output file naming conventions, so try both.
                    try:
                        output_name = f'{this_name}_output'  # use this one
                        output_df = pd.read_csv(os.path.join(project_path, this_exp, this_name, f"{this_name}_{this_run}", f'{output_name}.csv'))
                        print(f"\t\tfound {this_name}_output.csv")
                    except:
                        # try with run number.
                        output_name = f'{this_name}_{this_run}_output'  # use this one
                        output_df = pd.read_csv(os.path.join(project_path, this_exp, this_name, f"{this_name}_{this_run}", f'{output_name}.csv'))
                        print(f"\t\tappending {this_name}_{this_run}_output.csv")

                        # remove any Unnamed columns
                        if any("Unnamed" in i for i in list(output_df.columns)):
                            unnamed_col = [i for i in list(output_df.columns) if "Unnamed" in i][0]
                            output_df.drop(unnamed_col, axis=1, inplace=True)

                        run_outputs.append(output_df)

                else:
                    print(f"\n\tNOT found: p_name: {p_name} ({this_name}), this_exp: {this_exp}")


        # combine datasets from different versions for this run.
        run_data_df = pd.concat(run_outputs)
        run_data_df = run_data_df.sort_values(by=['step', 'trial_number', 'ISI', 'separation'])
        print(f"run_data_df ({run_data_df.shape}):\n{run_data_df}")

        # make dir if it doesn't exist - use p_name NOT this_name
        new_p_run_path = os.path.join(project_path, Ricco_all_dir, p_name, f"{p_name}_{this_run}")
        if not os.path.isdir(new_p_run_path):
            try:
                os.makedirs(new_p_run_path, exist_ok=True)
                print("Directory '%s' created successfully" % new_p_run_path)
            except OSError as error:
                print("Directory '%s' can not be created" % new_p_run_path)

        save_name = f'{p_name}_{this_run}_output.csv'
        save_excel_path = os.path.join(new_p_run_path, save_name)
        print(f"\nsaving all_data_df to save_excel_path:\n{save_excel_path}")
        # run_data_df.to_excel(save_excel_path, index=False)
        run_data_df.to_csv(save_excel_path, index=False)


# for p_name in participant_names:
#     # print(f"\np_name: {p_name}")
#
#     if p_name in same_name_dict.keys():
#         alias = same_name_dict[p_name]
#         names_to_check = [p_name, alias]
#     else:
#         names_to_check = [p_name]
#
#     for this_name in names_to_check:
#         print(f"\np_name: {p_name} ({this_name})")
#
#         for this_exp in exp_names:
#
#             print(f"this_exp: {this_exp}")
#
#             p_name_dir = os.path.join(project_path, this_exp, p_name)
#
#             if os.path.isdir(p_name_dir):
#                 print(f"found: {p_name_dir}")
#
#                 for this_run in list(len(n_runs)):
#
#                     # there are a couple of possible output file naming conventions, so try both.
#                     try:
#                         p_name = f'{participant_name}_output'  # use this one
#                         output_df = pd.read_csv(os.path.join(filepath, f'{p_name}.csv'))
#                         print("\tfound p_name_output.csv")
#                     except:
#                         # try with run number. (last character(s) of run_dir, after '_'.
#                         run_number = run_dir.split('_')[-1]
#                         p_name = f'{participant_name}_{run_number}_output'  # use this one
#                         output_df = pd.read_csv(os.path.join(filepath, f'{p_name}.csv'))
#                         print("\tfound p_name_run_number_output.csv")
#
#
#             else:
#                 print(f"\tnot found: {p_name_dir}")

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
#         # # just take sep 5 data
#         # if 'separation' in list(results_df):
#         #     results_df = results_df[results_df['separation'].isin([-5, 5])]
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



print('\ncombine_all_Ricco finished\n')
