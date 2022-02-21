import os
import pandas as pd

# rename files and folders to anonimise.
exp_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/exp1a_data'
isi_list = [-1, 0, 2, 4, 6, 9, 12, 24]
old_p_list = ['Tony', 'Simon', 'Maria', 'Kristian', 'Kim']
participant_list = ['aa', 'bb', 'cc', 'dd', 'ee']
new_p_list = ['aa', 'bb', 'cc', 'dd', 'ee']

ready_to_rename = False

p_idx_plus = 1

for p_idx, participant_name in enumerate(participant_list):

    print(f'\nparticipant_name: {participant_name}')

    # root_path = f'{exp_path}/{participant_name}'
    root_path = os.path.join(exp_path, participant_name)

    # run_folder_names = [f'P{p_idx + p_idx_plus}a-{participant_name}', f'P{p_idx + p_idx_plus}b-{participant_name}',
    #                     f'P{p_idx + p_idx_plus}c-{participant_name}', f'P{p_idx + p_idx_plus}d-{participant_name}',
    #                     f'P{p_idx + p_idx_plus}e-{participant_name}', f'P{p_idx + p_idx_plus}f-{participant_name}']

    run_folder_names = [f'{participant_name}_1', f'{participant_name}_2',
                        f'{participant_name}_3', f'{participant_name}_4',
                        f'{participant_name}_5', f'{participant_name}_6']

    for run_idx, run_dir in enumerate(run_folder_names):
        # print(f'run_dir: {run_dir}')

        for isi in isi_list:
            # print(isi)

            data_dir_name = f'ISI_{isi}_probeDur2'

            run_dir_path = os.path.join(root_path, run_dir)
            data_file_path = os.path.join(run_dir_path, data_dir_name)
            # print(f'data_file_path: {data_file_path}')

            os.chdir(data_file_path)
            print(f'cwd: {os.getcwd()}')

            file_name = f'{participant_name}_output.csv'
            if os.path.isfile(os.path.join(data_file_path, file_name)):
                file_name = f'{participant_name}_output.csv'
                # print(f'found file: {file_name}')
            elif os.path.isfile(os.path.join(data_file_path, f'{participant_name}{run_idx+1}.csv')):
                file_name = f'{participant_name}{run_idx + 1}.csv'
                # print(f'found file: {file_name}')
            elif os.path.isfile(os.path.join(data_file_path, f'{participant_name}.csv')):
                file_name = f'{participant_name}.csv'
                # print(f'found file (isi={isi}): {file_name}')
            else:
                raise FileNotFoundError(f'\n***FILE NOT FOUND: {file_name}')
                # print(f'\n***FILE NOT FOUND: {file_name}')

            # change participant column in df
            output_df = pd.read_csv(file_name)
            # remove any Unnamed columns
            if any("Unnamed" in i for i in list(output_df.columns)):
                unnamed_col = [i for i in list(output_df.columns) if "Unnamed" in i][0]
                output_df.drop(unnamed_col, axis=1, inplace=True)
                print('unnamed columns removed')

            # get participant name as stored in csv
            p_label = output_df["1. Participant"][0]
            # print(f'p_label: {p_label}')
            output_df["1. Participant"].replace({p_label: participant_name}, inplace=True)
            print(f'output_df["1. Participant"]:\n{output_df["1. Participant"]}')
            if ready_to_rename:
                output_df.to_csv(file_name)

            # change raw data file name.  e.g., aa_output, bb_output
            new_file_name = f'{new_p_list[p_idx]}_output.csv'
            print(f'renaming {file_name} ==> {new_file_name}')
            if ready_to_rename:
                os.rename(file_name, new_file_name)

        # change eac h run folder names (aa_1, aa_2, aa_2; bb_1, bb_2, bb_3 etc)
        os.chdir(root_path)
        print(f'\ncwd: {os.getcwd()}')
        new_run_dir_name = f'{new_p_list[p_idx]}_{run_idx+1}'
        print(f'found all files and ready to rename: {run_dir} ==> {new_run_dir_name}\n')
        if ready_to_rename:
            os.rename(run_dir, new_run_dir_name)

    # # change each participant name [aa, bb, cc, dd, ee]
    os.chdir(exp_path)
    print(f'cwd: {os.getcwd()}')
    new_p_dir_name = new_p_list[p_idx]
    print(f'found all files and ready to rename: {participant_name} ==> {new_p_dir_name}\n')
    if ready_to_rename:
        os.rename(participant_name, new_p_dir_name)

print('\nfinished rename_anonimise_project')
