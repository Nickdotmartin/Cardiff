import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from check_home_dir import running_on_laptop, which_path, switch_path

"""
1. Get path to dir containing intervals
2. get list of files and parse out trial numbers and ISIs
3. simple version: just append all values to a list.
4. plot x-axis, trial number, y axis frame dur.
"""

for run in ['run1', 'run2', 'run3_coreRush', 'run4_coreRush']:  #, 'run2', 'run3', 'run4']:


    # path_to_data = f"/Users/nickmartin/Library/CloudStorage/OneDrive-CardiffUniversity/" \
    #                f"PycharmProjects/Cardiff/memory_and_timings/frameIntervals_20221121/{run}"
    path_to_data = f"/Users/nickmartin/Library/CloudStorage/OneDrive-CardiffUniversity/" \
                   f"PycharmProjects/Cardiff/memory_and_timings/LinuxFrameIntervals2_20221122/{run}"
    if not running_on_laptop():
        path_to_data = switch_path(path_to_data, 'windows_oneDrive')

    print(f"run: {run}")


    file_list = [f for f in os.listdir(path_to_data) if os.path.isfile(os.path.join(path_to_data, f))]
    # print(file_list)

    # n_trials = 100
    n_trials = 57
    if run == 'practice':
        n_trials = 25

    all_data_list = []
    # for trial_num in list(range(1, n_trials+1)):
    for trial_num in list(range(1, n_trials)):

        this_substring = f"FrameIntervals_{trial_num}_ISI"
        this_filename = [s for s in file_list if this_substring in s][0]
        print(this_filename)
        isi_val = int(this_filename.split("_ISI")[1])

        with open(os.path.join(path_to_data, this_filename)) as f:
            lines = f.readlines()
            val_list = lines[0].split(", ")
            [all_data_list.append(float(i)) for i in val_list]


    print(all_data_list)
    plt.plot(all_data_list)
    plt.xlabel('trial')
    plt.ylabel('Frame Interval')

    plt.title(f'{run} frame intervals: flip to flip')

    save_name = f'{run}_frameIntervals.png'
    plt.savefig(os.path.join(path_to_data, save_name))
    plt.show()


