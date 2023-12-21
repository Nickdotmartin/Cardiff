import os
import pandas as pd



anonimise_dict = {'Tony': 'aa', 'Simon': 'bb', 'Maria': 'cc',
                  'Kristian': 'dd',
                  # 'Kris': 'dd',  # comment this out and run separately, as it was changing names to ddtian!
                  'Kim': 'ee',
                  'Nick': 'ff'
                  }

# exp_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\exp1a_data_v2"
# exp_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\sep45_data_to_zip"
# exp_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\Ricco_&_Bloch_to_zip"
exp_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\missing_probe_to_zip\Exp4b_missing_probe_23\mixed_dir\incoherent"

# walk through all folders and subfolders and rename files ONLY
for root, dirs, files in os.walk(exp_path):
    print(f'\nroot: {root}')
    print(f'dirs: {dirs}')
    print(f'files: {files}')

    # check any master csvs for participant names and rename
    for file in files:
        check_p_name = False
        # if file is csv
        if file[-4:] == '.csv':
            master_df = pd.read_csv(os.path.join(root, file))
            check_p_name = True
            save_as = 'csv'
        elif file[-5:] == '.xlsx':
            master_df = pd.read_excel(os.path.join(root, file))
            check_p_name = True
            save_as = 'xlsx'
        if check_p_name:
            for col_name in ['p_stack_sep', 'p_stack', 'stack', 'participant', 'pname', 'p_name', '1. Participant']:
                if col_name in master_df.columns:
                    for key in anonimise_dict.keys():
                        if key in master_df[col_name].values:
                            print(f'found participant name {key} in {col_name} column of {file}')
                            print(f'master_df:\n{master_df.head()}')
                            master_df[col_name].replace({key: anonimise_dict[key]}, inplace=True)
                            print(f'master_df:\n{master_df}')
                            if save_as == 'csv':
                                master_df.to_csv(os.path.join(root, file))
                            elif save_as == 'xlsx':
                                master_df.to_excel(os.path.join(root, file))
                            print(f'renamed {file}')

    # if any anonimise_dict keys are in the file name, rename the file
    for file in files:
        if any(key in file for key in anonimise_dict.keys()):
            print(f'found file: {file}')
            for key in anonimise_dict.keys():
                if key in file:
                    new_file = file.replace(key, anonimise_dict[key])
                    print(f'renaming {file} ==> {new_file}')
                    os.rename(os.path.join(root, file), os.path.join(root, new_file))


# walk through all folders and subfolders and rename folders
for root, dirs, files in os.walk(exp_path):
    print(f'\nroot: {root}')
    print(f'dirs: {dirs}')
    print(f'files: {files}')
    # if any anonimise_dict keys are in the folder name, rename the folder
    for dir in dirs:
        if any(key in dir for key in anonimise_dict.keys()):
            print(f'found folder: {dir}')
            for key in anonimise_dict.keys():
                if key in dir:
                    new_dir = dir.replace(key, anonimise_dict[key])
                    print(f'renaming {dir} ==> {new_dir}')
                    os.rename(os.path.join(root, dir), os.path.join(root, new_dir))







# rename files and folders to anonimise.
# exp_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/exp1a_data'
# exp_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\test_venv\Target_detection_Dec23"
# isi_list = [-1, 0, 2, 4, 6, 9, 12, 24]
# old_p_list = ['Tony', 'Simon', 'Maria', 'Kristian', 'Kim', 'Nick']
# participant_list = ['aa', 'bb', 'cc', 'dd', 'ee', 'ff']
# new_p_list = ['aa', 'bb', 'cc', 'dd', 'ee', 'ff']
#
# # remane misnames file
# # exp_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\rad_flow_Sept23\rad_flow_7_spokes"
# # isi_list = [6]
# # old_p_list = ['asus_cal_circles_rings_quartSpd']
# # participant_list = ['asus_cal_circles_rings_quartSpd']
# # new_p_list = ['asus_cal_circles_rings_HalfSpd']
#
#
# ready_to_rename = True
#
# p_idx_plus = 1
#
# for p_idx, participant_name in enumerate(participant_list):
#
#     print(f'\nparticipant_name: {participant_name}')
#
#     # root_path = f'{exp_path}/{participant_name}'
#     root_path = os.path.join(exp_path, participant_name)
#
#     # run_folder_names = [f'P{p_idx + p_idx_plus}a-{participant_name}', f'P{p_idx + p_idx_plus}b-{participant_name}',
#     #                     f'P{p_idx + p_idx_plus}c-{participant_name}', f'P{p_idx + p_idx_plus}d-{participant_name}',
#     #                     f'P{p_idx + p_idx_plus}e-{participant_name}', f'P{p_idx + p_idx_plus}f-{participant_name}']
#
#     run_folder_names = [f'{participant_name}_1', f'{participant_name}_2',
#                         f'{participant_name}_3', f'{participant_name}_4',
#                         f'{participant_name}_5', f'{participant_name}_6']
#
#     # '''this bit is just for fixing rad_flow_7_spokes names, although it only worked on bg70 and
#     # failed on bg350 as dir names had already changed.'''
#     # root_path = os.path.join(root_path, 'flow_rings')
#     #
#     # for prelim in ['bg350', 'bg70']:
#     #     root_path = os.path.join(root_path, prelim)
#     #
#     #     run_folder_names = [f'{participant_name}_1', f'{participant_name}_2',
#     #                         f'{participant_name}_3', f'{participant_name}_4',
#     #                         f'{participant_name}_5', f'{participant_name}_6',
#     #                         f'{participant_name}_7', f'{participant_name}_8',
#     #                         f'{participant_name}_9', f'{participant_name}_10',
#     #                         f'{participant_name}_11', f'{participant_name}_12']
#
#     for run_idx, run_dir in enumerate(run_folder_names):
#         # print(f'run_dir: {run_dir}')
#
#         for isi in isi_list:
#             # print(isi)
#
#             data_dir_name = f'ISI_{isi}_probeDur2'
#             # data_dir_name = f'ISI_{isi}'
#
#             run_dir_path = os.path.join(root_path, run_dir)
#             data_file_path = os.path.join(run_dir_path, data_dir_name)
#             # print(f'data_file_path: {data_file_path}')
#
#             os.chdir(data_file_path)
#             print(f'cwd: {os.getcwd()}')
#
#             file_name = f'{participant_name}_output.csv'
#             if os.path.isfile(os.path.join(data_file_path, file_name)):
#                 file_name = f'{participant_name}_output.csv'
#                 # print(f'found file: {file_name}')
#             elif os.path.isfile(os.path.join(data_file_path, f'{participant_name}{run_idx+1}.csv')):
#                 file_name = f'{participant_name}{run_idx + 1}.csv'
#                 # print(f'found file: {file_name}')
#             elif os.path.isfile(os.path.join(data_file_path, f'{participant_name}.csv')):
#                 file_name = f'{participant_name}.csv'
#                 # print(f'found file (isi={isi}): {file_name}')
#             elif os.path.isfile(os.path.join(data_file_path, f'{participant_name}_{run_idx+1}_output.csv')):
#                 file_name = f'{participant_name}_{run_idx+1}_output.csv'
#                 # print(f'found file: {file_name}')
#             else:
#                 raise FileNotFoundError(f'\n***FILE NOT FOUND: {file_name}')
#                 # print(f'\n***FILE NOT FOUND: {file_name}')
#
#             # change participant column in df
#             output_df = pd.read_csv(file_name)
#             # remove any Unnamed columns
#             if any("Unnamed" in i for i in list(output_df.columns)):
#                 unnamed_col = [i for i in list(output_df.columns) if "Unnamed" in i][0]
#                 output_df.drop(unnamed_col, axis=1, inplace=True)
#                 print('unnamed columns removed')
#
#             # get participant name as stored in csv
#             if '01. Participant' in output_df.columns:
#                 p_label = output_df["01. Participant"][0]
#                 # print(f'p_label: {p_label}')
#                 output_df["01. Participant"].replace({p_label: participant_name}, inplace=True)
#                 print(f'output_df["01. Participant"]:\n{output_df["01. Participant"]}')
#             elif '1. Participant' in output_df.columns:
#                 p_label = output_df["1. Participant"][0]
#                 # print(f'p_label: {p_label}')
#                 output_df["1. Participant"].replace({p_label: participant_name}, inplace=True)
#                 print(f'output_df["1. Participant"]:\n{output_df["1. Participant"]}')
#
#
#             if ready_to_rename:
#                 output_df.to_csv(file_name)
#
#             # change raw data file name.  e.g., aa_output, bb_output
#             new_file_name = f'{new_p_list[p_idx]}_output.csv'
#             print(f'renaming {file_name} ==> {new_file_name}')
#             if ready_to_rename:
#                 os.rename(file_name, new_file_name)
#
#         # change eac h run folder names (aa_1, aa_2, aa_2; bb_1, bb_2, bb_3 etc)
#         os.chdir(root_path)
#         print(f'\ncwd: {os.getcwd()}')
#         new_run_dir_name = f'{new_p_list[p_idx]}_{run_idx+1}'
#         print(f'found all files and ready to rename: {run_dir} ==> {new_run_dir_name}\n')
#         if ready_to_rename:
#             os.rename(run_dir, new_run_dir_name)
#
#     # # change each participant name [aa, bb, cc, dd, ee]
#     os.chdir(exp_path)
#     print(f'cwd: {os.getcwd()}')
#     new_p_dir_name = new_p_list[p_idx]
#     print(f'found all files and ready to rename: {participant_name} ==> {new_p_dir_name}\n')
#     if ready_to_rename:
#         os.rename(participant_name, new_p_dir_name)

print('\nfinished rename_anonimise_project')
