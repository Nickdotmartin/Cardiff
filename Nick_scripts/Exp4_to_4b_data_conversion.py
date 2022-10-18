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
'''
exp4a_path = "/Users/nickmartin/Library/CloudStorage/OneDrive-CardiffUniversity/PycharmProjects/Cardiff/EXP4_missing_probe"

for cond in ['radial', 'rotation', 'translation']:
    for p_name in ['Nick', 'Simon']:
        for run in list(range(1, 12)):

            run_dir = f"{p_name}_{run}"

            output_path = os.path.join(exp4a_path, cond, p_name, run_dir)

            if not os.path.isdir(output_path):
                print(f"\tno: this conditons hasn't been done yet\n\t{output_path}")
                print(f"yes: {output_path}")
            else:
                print(f"yes: {output_path}")
                '''
                load the output file, 
                split it by cond type
                    a) coherent
                    b) incoherent.
                save these as two new output files and put them into new 4b folders.
                
                '''
