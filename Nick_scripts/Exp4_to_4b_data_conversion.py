import pandas as pd
import os
from check_home_dir import which_path, running_on_laptop, switch_path

'''
Original version of missing_probe exp had 3 conditions (radial, rotaion, translation), 
and for each of these trials were interleaved with incoherent trials.
This gave 450 trials in a 3x3.
We now have a new version which has four conditions (radial, rotaion, translation & incoherent),
There are no interleaved trials so each cond is 225 trials in a 3x3.

This script will take the data me and simon have already collected and will split off
the incoherent trials from the coherent trials and put them into the correct respective folders.

We will have more than 12 incoherent trials, so they will be numbered from 1, in the order
    radial, rotation, translation
N:  1-7,    8-13        14-20
S:  1       2-4         5-16
'''

cardiff_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff"

exp4a_path = os.path.join(cardiff_path, "EXP4_missing_probe")

inc_paths = []
coh_paths = []

for p_name in ['Nick', 'Simon']:
    n_inc_df = 0

    for cond in ['radial', 'rotation', 'translation']:

        for run in list(range(1, 13)):
# for cond in ['radial']:  # 'rotation', 'translation']:
#     for p_name in ['Nick']:  #, 'Simon']:
#         n_inc_df = 0
#         for run in [1]:
            run_dir = f"{p_name}_{run}"

            output_path = os.path.join(exp4a_path, cond, p_name, run_dir)

            if not os.path.isdir(output_path):
                print(f"\tno: this conditons hasn't been done yet: {output_path}")
            else:
                print(f"yes: {output_path}")
                '''
                load the output file, 
                split it by cond type
                    a) coherent
                    b) incoherent.
                save these as two new output files and put them into new 4b folders.
                '''

                output_file_path = os.path.join(output_path, f'{p_name}_{run}_output.csv')
                output_df = pd.read_csv(output_file_path)
                print(f"output_df: {list(output_df.columns)}\n{output_df.head()}")

                inc_df = output_df[output_df['cond_type'] == 'incoherent']
                print(f"inc_df.shape: {inc_df.shape}")
                n_inc_df += 1

                inc_p_run_dir = os.path.join(cardiff_path, "EXP4b_missing_probe", "incoherent", p_name, f"{p_name}_{n_inc_df}")
                # inc_df_path = os.path.join(inc_p_run_dir, f'{p_name}_{n_inc_df}_output.csv')
                # inc_paths.append(inc_df_path)
                print(f"inc_p_run_dir: {inc_p_run_dir}")

                if not os.path.isdir(inc_p_run_dir):
                    os.makedirs(inc_p_run_dir)
                inc_df.to_csv(os.path.join(inc_p_run_dir, f'{p_name}_{n_inc_df}_output.csv'), index=False)

                coh_df = output_df[output_df['cond_type'] != 'incoherent']
                print(f"coh_df.shape: {coh_df.shape}")

                coh_p_run_dir = os.path.join(cardiff_path, "EXP4b_missing_probe", cond, p_name, f"{p_name}_{run}")
                print(f"coh_p_run_dir: {coh_p_run_dir}")

                if not os.path.isdir(coh_p_run_dir):
                    os.makedirs(coh_p_run_dir)
                coh_df.to_csv(os.path.join(coh_p_run_dir, f'{p_name}_{run}_output.csv'), index=False)

                # coh_paths.to_csv(coh_df_path)

# print(f"\n\ncoh_paths:\n")
# for i in coh_paths:
#     print(i)
# print("\n\ninc_paths:\n")
# for i in inc_paths:
#     print(i)





