import os
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression, LinearRegression
from scipy.special import expit

from check_home_dir import which_path, running_on_laptop, switch_path

'''
Part A
If I re-do the experiment I might write a script to automate the stuff I did by hand in excel:

'''
def make_eyetrack_csv(p_name, run, eye_track_dir=None):
    """
    Function to convert old csv to new csv for analysis.  Input should contain columns called ['time', 'X', 'Y'].
    This old csv also has various messages, which appead in the time and X columns.
    These messages interupt the dataframe, which has three problems.
        1. the message appears in the row above the thing it refers to.
        2. the timestamp col contains info about message type
        3. the x col contains the message
    I want to move these messages to new columns, where they are alongside the frame they refer to.
    There are two types of message:
        a) ones about eyes (fixation, saccade, blink).
        b) ones about the trial segment (Fixation, probe1, ISI, probe2, response.
    (Unfortunately probe1 and some probe2 messages are missing so I will have infer their timings).
    I will use the start of each trial (start_fixation) marker to infer trial numbers.

    Trial numbers will be used to access sep, isi and corner info from experiment output csv

    Output csv will have the following columns:
        time_stamp
        trial_num: (1 indexed)
        trial_frames: frames since the start of this trial.
        segment: e.g., fixation, probe1, isi, probe2, response.
        x_pos:
        y_pos:
        eye_message: e.g., saccade, fixation, blink etc.
        sep: from output.csv
        ISI: from output.csv
        probe_jump: from output.csv
        corner: from output.csv
        probeLum: from output.csv
        resp: from output.csv
        crnr_check: I might not use this.
        other_message: any other info shown
        other_msg_details: I might not use these but useful to see what else comes up

    :param p_name: name of participant, used for accessing dirs and files.
    :param run: run number, used for accessing dirs and files.
    :param eye_track_dir: path to folder with participant folders containing run folders and plink csvs.
    :return: new csv
    """

    print(f"\n*** running make_eyetrack_csv(p_name={p_name}, run={run}, eye_track_dir={eye_track_dir}) ***\n")

    if eye_track_dir is None:
        eye_track_dir = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\eyetracking"

    # get file_path and csv_name
    save_path = os.path.join(eye_track_dir, p_name)
    old_df_name = f"{p_name.upper()}_{run}_pylink.csv"
    old_csv_path = os.path.join(save_path, old_df_name)
    print(f'eye_track_dir:\n{eye_track_dir}')
    print(f'save_path: {save_path}')
    print(f'old_df_name: {old_df_name}')
    eye_df = pd.read_csv(old_csv_path, usecols=['time', 'X', 'Y'])
    print(f"\neye_df: {eye_df.shape}\n{eye_df.head()}\n")

    # get details from the ouput csv
    exp_output_path = os.path.join(save_path, f"{p_name}_{run}", f"{p_name}_output.csv")
    exp_df = pd.read_csv(exp_output_path, index_col='trial_number')
    print(f"exp_df: {exp_df.shape}\n{exp_df.head()}\n")


    new_data_list = []
    trial_num = 0
    trial_frames = 0

    # # variables to persist across many rows
    segment = None
    eye_message = None
    crnr_check = None
    # I might not need these variables
    other_message = None
    other_msg_details = None

    # # variables from output.csv
    sep = None
    ISI = None
    probe_jump = None
    corner = None
    probeLum = None
    resp = None

    for idx, row in eye_df.iterrows():

        # # put these here so they don't copy to each row
        # other_message = None
        # other_msg_details = None
        end_of_old_df = False
        if end_of_old_df:
            break
        else:
            # # check to see row is a (str) message or (int) regular row by checking 'time' column.
            try:
                time_stamp = int(row['time'])
            except:
                time_stamp = None

            if time_stamp:
                # # regular timestamped rows#
                trial_frames += 1
                # convert values to floats unless missing data
                if row['X'] == '   .':
                    x_pos = np.NAN
                else:
                    x_pos = float(row['X'])
                if row['Y'] == '   .':
                    y_pos = np.NAN
                else:
                    y_pos = float(row['Y'])

                # # regular rows will be appended to new_df
                new_row = [time_stamp, trial_num, trial_frames, segment, x_pos, y_pos, eye_message,
                           sep, ISI, probe_jump, corner, probeLum, resp, crnr_check, other_message, other_msg_details]
                print(time_stamp)
                if trial_num > 0:
                    # don't include rows before exp starts
                    new_data_list.append(new_row)
            else:
                # # message rows
                # get str details for segment or eye message
                if 'MSG' in row['time']:
                    # this is a segment message
                    if 'Start fixation' in row['X']:
                        # indicates a new trial
                        trial_num += 1
                        trial_frames = 0
                        segment = 'fix_dot'

                        # access info from df
                        this_exp_row = exp_df.loc[[trial_num]]
                        sep = int(this_exp_row['separation'])
                        ISI = int(this_exp_row['ISI'])
                        probe_jump = int(this_exp_row['probe_jump'])
                        corner = int(this_exp_row['corner'])
                        probeLum = float(this_exp_row['probeLum'])
                        resp = int(this_exp_row['trial_response'])

                    # todo: I need to add in probe1 presentation times somehow
                    elif 'Start probe1' in row['X']:
                        segment = 'probe1'
                    elif 'Start ISI' in row['X']:
                        segment = 'ISI'
                    elif 'Corner' in row['X']:
                        crnr_check = row['X'].lstrip(" 1234567890")[7:]
                    elif 'Start probe2' in row['X']:
                        # todo: I need to add in probe2 presentation times somehow
                        segment = 'probe2'
                    elif 'End stimulus' in row['X']:
                        # todo: I need to add in probe2 presentation times somehow
                        segment = 'Response'
                    elif 'MODE RECORD' in row['X']:
                        # recording has started, don't add anything
                        segment = None
                    else:
                        other_msg_details = row['X']
                        print(f"\t\tother_msg_details: {other_msg_details}")

                # eye movement messages
                elif 'SFIX' in row['time']:
                    eye_message = 'fixation'
                elif 'EFIX' in row['time']:
                    eye_message =None
                elif 'SSACC' in row['time']:
                    eye_message = 'saccade'
                elif 'ESACC' in row['time']:
                    eye_message =None
                elif 'SBLINK' in row['time']:
                    eye_message = 'blink'
                elif 'EBLINK' in row['time']:
                    eye_message =None
                elif row['time'] == 'END':
                    if row['Y'] == 'SAMPLES':
                        # stop iterating through df incase there is stuff at the bottom.
                        # I might not need this
                        end_of_old_df = True

                else:
                    other_message = row['time']
                    print(f"\t\tother_message: {other_message}")
                    print(f"\t\tdetails: {row['time']} % {row['X']} % {row['Y']}")
                    # I think I've caught all the possible messages
                    raise ValueError()

    # make new_df and save to csv
    new_df = pd.DataFrame(new_data_list, columns=['time_stamp', 'trial_num', 'trial_frames', 'segment',
                                                  'x_pos', 'y_pos', 'eye_message',
                                                  'sep', 'ISI', 'probe_jump', 'corner', 'probeLum', 'resp',
                                                  'crnr_check', 'other_message', 'other_msg_details'])
    new_df_name = f"{p_name}_{run}_eyetracking.csv"
    print(f'new_df_name: {new_df_name}')

    print(f'\nnew_df:\n{new_df}')
    new_df.to_csv(os.path.join(save_path, new_df_name), index=False)

    return new_df

# # do one df
# old_csv_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\eyetracking\p1\P1_1_pylink.csv"
# p_name='p1'
# run=1
# make_eyetrack_csv(p_name=p_name, run=run)

# # do all dfs
# participants = ['p1', 'p2']
# run_list = [1, 2, 3]
# for p_name in participants:
#     for run in run_list:
#         make_eyetrack_csv(p_name=p_name, run=run)


'''
There is important stuff missing from the orignial plink data.
I missed out the message for the start of probe1 (!whoops!).  
And some messages for the start of probe2 are missing (I have isi4 but not isi0).

First I will try to work out how many of the 500Hz frames the probe presentations correspond to.  
Then I will go in and change the segment labels accordingly.

Because the fixation time varies, I can't use that to infer when p1 is presented, 
so I will have to use the start of the ISI and go so many rows before that.

For probe2, I will either use the ISI or go so many frames before the response time.  

This script just collates the number of frames (at 500Hz) from eyetracker for each experimental segment.

I can use this to work out what changes to make (e.g., based on means per cond)

'''

def count_missing_segments(p_list, run_list, eye_track_dir=None):
    """

    :param p_name: name of participant, used for accessing dirs and files.
    :param run: run number, used for accessing dirs and files.
    :param eye_track_dir: path to folder with participant folders containing run folders and plink csvs.
    :return:
    """
    if eye_track_dir is None:
        eye_track_dir = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\eyetracking"

    new_list = []

    for p_name in p_list:
        for run in run_list:

            # get file_path and csv_name
            save_path = os.path.join(eye_track_dir, p_name)
            csv_name = f"{p_name}_{run}_eyetracking.csv"
            csv_path = os.path.join(save_path, csv_name)
            print(f'eye_track_dir:\n{eye_track_dir}')
            print(f'save_path: {save_path}')
            print(f'csv_name: {csv_name}')
            print(f'csv_path: {csv_path}')
            eye_df = pd.read_csv(csv_path)
            print(f"\neye_df: {eye_df.shape}\n{eye_df.head()}\n")

            trial_numbers = eye_df['trial_num'].unique().tolist()

            # loop through stairs for this isi
            for idx, trial_number in enumerate(trial_numbers):

                # get df just for one stair at this isi
                trial_df = eye_df[eye_df['trial_num'] == trial_number]
                print(f'\ntrial_df ({trial_number}) {trial_df.shape}:\n{trial_df}')
                # print(f'\ntrial_number: {trial_number}')
                # print(f"\ntrial_df['ISI']: {trial_df['ISI']}")

                isi_val = int(trial_df['ISI'].iloc[0])
                sep_val = int(trial_df['sep'].iloc[0])
                print(f'\ntrial_number: {trial_number}, ISI: {isi_val}, sep: {sep_val}')

                segment_info = trial_df['segment'].value_counts()
                # print(f'segment_info:\n{segment_info}')
                seg_info_idx = list(segment_info.index)
                # print(f'seg_info_idx:\n{seg_info_idx}')

                if 'fix_dot' in seg_info_idx:
                    fix_dot_rows = int(segment_info['fix_dot'])
                else:
                    fix_dot_rows = 0

                if 'probe1' in seg_info_idx:
                    probe1_rows = int(segment_info['probe1'])
                else:
                    probe1_rows = 0

                if 'ISI' in seg_info_idx:
                    isi_rows = int(segment_info['ISI'])
                else:
                    isi_rows = 0

                if 'probe2' in seg_info_idx:
                    probe2_rows = int(segment_info['probe2'])
                else:
                    probe2_rows = 0

                if 'Response' in seg_info_idx:
                    resp_rows = int(segment_info['Response'])
                else:
                    resp_rows = 0

                new_row = [p_name, run, trial_number, isi_val, sep_val, fix_dot_rows, probe1_rows, isi_rows, probe2_rows, resp_rows]
                new_list.append(new_row)

            new_df = pd.DataFrame(new_list, columns=['p_name', 'run', 'trial_number', 'isi_val', 'sep_val',
                                                     'fix_dot_rows', 'probe1_rows', 'isi_rows', 'probe2_rows', 'resp_rows'])
            print(f'\nnew_df:\n{new_df}')

            # new_df_name = f"{p_name}_{run}_seg_count.csv"
            # print(f'new_df_name: {new_df_name}')

            new_df.to_csv(os.path.join(eye_track_dir, 'seg_count.csv'), index=False)



# do all dfs
# participants = ['p1', 'p2']
# run_list = [1, 2, 3]
# count_missing_segments(p_list=participants, run_list=run_list)


def fix_missing_segments(p_name, run, eye_track_dir=None):
    """
    This script will go through the new dataframes and produce a new dataframe with the correct segment labels.

    for all trials, I will change the last 7 fixation frames to be probe1 frames.
    For all ISI0 I will change all 7 ISI frames to be probe2 frames.

    loop through a dataframe loading one trial at a time
    Fix the segment details for that trial
    Append new segment details to new_list
    concatenate all the segmant details for the whole df, and replace segment column

    save details.

    Concatenate list back into one df, with correct segments and trial numbers

    :param p_name: name of participant, used for accessing dirs and files.
    :param run: run number, used for accessing dirs and files.
    :param eye_track_dir: path to folder with participant folders containing run folders and plink csvs.
    :return: new csv
    """

    print(f"\n*** running fix_missing_segments(p_name={p_name}, run={run}, eye_track_dir={eye_track_dir}) ***\n")

    if eye_track_dir is None:
        eye_track_dir = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\eyetracking"

    # get file_path and csv_name
    save_path = os.path.join(eye_track_dir, p_name)
    old_df_name = f"{p_name.upper()}_{run}_eyetracking.csv"
    old_csv_path = os.path.join(save_path, old_df_name)
    print(f'eye_track_dir:\n{eye_track_dir}')
    print(f'save_path: {save_path}')
    print(f'old_df_name: {old_df_name}')
    eye_df = pd.read_csv(old_csv_path, usecols=['time_stamp', 'trial_num', 'trial_frames', 'segment',
                                                'x_pos', 'y_pos', 'eye_message',
                                                'sep', 'ISI', 'probe_jump', 'corner', 'probeLum', 'resp'])
    print(f"\neye_df: {eye_df.shape}\n{eye_df.head()}\n")


    new_seg_list = []
    row_check = []

    # # loop through frames for this trial
    trial_numbers = eye_df['trial_num'].unique().tolist()
    for idx, trial_number in enumerate(trial_numbers):
        # get df just for one stair at this isi
        trial_df = eye_df[eye_df['trial_num'] == trial_number]
        print(f'\ntrial_number: {trial_number}')

        segment_info = trial_df['segment'].value_counts()
        print(f'segment_info:\n{segment_info}')

        orig_seg_col = list(trial_df['segment'])
        print(f'orig_seg_col:\n{orig_seg_col}')

        # get isi details which show if probe2 needs adjusting
        isi_val = int(trial_df['ISI'].iloc[0])
        print(f'\nisi_val: {isi_val}')

        print(f"\ntrial_df: {trial_df.shape}\n{trial_df.head()}\n")
        rows, cols = trial_df.shape

        row_check.append(rows)

        seg_info_idx = list(segment_info.index)
        if 'fix_dot' in seg_info_idx:
            fix_dot_rows = int(segment_info['fix_dot'])
        else:
            fix_dot_rows = 0

        if 'probe1' in seg_info_idx:
            probe1_rows = int(segment_info['probe1'])
        else:
            probe1_rows = 0

        if 'ISI' in seg_info_idx:
            isi_rows = int(segment_info['ISI'])
        else:
            isi_rows = 0

        if 'probe2' in seg_info_idx:
            probe2_rows = int(segment_info['probe2'])
        else:
            probe2_rows = 0

        if 'Response' in seg_info_idx:
            resp_rows = int(segment_info['Response'])
        else:
            resp_rows = 0

        seg_vals = [fix_dot_rows, probe1_rows, isi_rows, probe2_rows, resp_rows]
        print(f"seg_vals (sum: {sum(seg_vals)}): {seg_vals}")
        new_fix_rows = fix_dot_rows-7
        new_probe1_rows = probe1_rows+7

        new_isi_rows = isi_rows
        new_probe2_rows = probe2_rows
        # # note, isi rows can occasionally equal 6.
        if isi_val == 0:
            new_isi_rows = isi_rows-isi_rows
            new_probe2_rows = probe2_rows+isi_rows


        new_seg_vals = [new_fix_rows, new_probe1_rows, new_isi_rows, new_probe2_rows, resp_rows]
        print(f"new_seg_vals (sum: {sum(new_seg_vals)}): {new_seg_vals}")

        if sum(seg_vals) != sum(new_seg_vals):
            raise ValueError("number of new segments doesn't match")

        new_list = [['fix_dot'] * new_fix_rows, ['probe1'] * new_probe1_rows, ['ISI'] * new_isi_rows, ['probe2'] * new_probe2_rows, ['response'] * resp_rows]
        flat_list = [item for sublist in new_list for item in sublist]
        print(f'\nflat_list: {np.shape(flat_list)}')
        # print(flat_list)

        if len(flat_list) != rows:
            raise ValueError('rows and new segments dont match')

        # new_seg_list.append(flat_list)
        new_seg_list += flat_list

    print(f'\nnew_seg_list: {np.shape(new_seg_list)}')

    # print(sum(new_seg_list))
    # print(new_seg_list)

    eye_df['segment'] = new_seg_list
    new_df_name = f"{p_name.upper()}_{run}_eyetrack_fixed.csv"

    eye_df.to_csv(os.path.join(save_path, new_df_name), index=False)



def short_trial_data(p_name, run, eye_track_dir=None):
    """
    This script will go through the new dataframes and produce a new dataframe.

    For each trial, just take: the last 100ms (50frames) of fixation before probe1.
    All probe1, ISI and probe2 frames.
    The first 200ms (100 frames) of the response.


    loop through a dataframe loading one trial at a time
    chop off unneeded fixation and response frames.
    concatenate all the segmant details for the whole df

    save details.

    Concatenate list back into one df, with correct segments and trial numbers

    :param p_name: name of participant, used for accessing dirs and files.
    :param run: run number, used for accessing dirs and files.
    :param eye_track_dir: path to folder with participant folders containing run folders and plink csvs.
    :return: new csv
    """

    print(f"\n*** running short_trial_data(p_name={p_name}, run={run}, eye_track_dir={eye_track_dir}) ***\n")

    if eye_track_dir is None:
        eye_track_dir = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\eyetracking"
    if running_on_laptop():
        eye_track_dir = switch_path(eye_track_dir, 'mac_oneDrive')
    print(f'eye_track_dir:\n{eye_track_dir}')

    # get file_path and csv_name
    save_path = os.path.join(eye_track_dir, p_name)
    old_df_name = f"{p_name.upper()}_{run}_eyetrack_fixed.csv"
    old_csv_path = os.path.join(save_path, old_df_name)
    print(f'save_path: {save_path}')
    print(f'old_df_name: {old_df_name}')
    print(f'old_csv_path:\n{old_csv_path}')
    eye_df = pd.read_csv(old_csv_path, usecols=['trial_num', 'segment',
                                                'x_pos', 'y_pos', 'eye_message',
                                                'sep', 'ISI', 'probe_jump', 'corner', 'probeLum', 'resp'])
    print(f"\neye_df: {eye_df.shape}\n{eye_df.head()}\n")


    all_trials_list = []

    # # loop through frames for this trial
    trial_numbers = eye_df['trial_num'].unique().tolist()
    for idx, trial_number in enumerate(trial_numbers):

        trial_df = eye_df[eye_df['trial_num'] == trial_number]
        print(f'\ntrial_number {trial_df.shape}: {trial_number}')

        this_trial_list = []

        segment_info = trial_df['segment'].value_counts()
        print(f'segment_info:\n{segment_info}')

        seg_names = trial_df['segment'].unique().tolist()
        print(f'seg_names: {seg_names}')

        # # flag for if data is missing
        trial_missing_data = False

        for segment in seg_names:

            seg_df = trial_df[trial_df['segment'] == segment]

            if segment == 'fix_dot':
                # just last 50 rows (100ms)
                seg_df = seg_df.iloc[-50:,:]
            elif segment == 'response':
                # just first 100 rows (200ms)
                seg_df = seg_df.iloc[:100,:]

            # # check that there is data for key segments
            else:
                if seg_df['x_pos'].isnull().values.any():
                    trial_missing_data = True
                if seg_df['y_pos'].isnull().values.any():
                    trial_missing_data = True


            print(f'\nsegment ({segment}): {seg_df.shape}')
            this_trial_list.append(seg_df)


        if trial_missing_data:
            print(f"missing key eye tracking data for trial {trial_number}\n")
        else:
            # add new index col to df
            this_trial_df = pd.concat(this_trial_list)
            print(f"\n{trial_number}. this_trial_df: {this_trial_df.shape}\n{this_trial_df.head()}\n")

            rows, cols = this_trial_df.shape
            new_idx = list(range(-50, rows-50))
            this_trial_df.insert(1, 'trial_idx', new_idx)

            all_trials_list.append(this_trial_df)

    short_trial_df = pd.concat(all_trials_list)
    print(f"\nshort_trial_df: {short_trial_df.shape}\n{short_trial_df.head()}\n")

    new_df_name = f"{p_name.upper()}_{run}_eyetrack_short.csv"
    short_trial_df.to_csv(os.path.join(save_path, new_df_name), index=False)


# # do one df
# old_csv_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\eyetracking\p1\P1_1_pylink.csv"
# p_name = 'p1'
# run = 1
# short_trial_data(p_name=p_name, run=run)

# do all dfs
# participants = ['p1', 'p2']
# run_list = [1, 2, 3]
# for p_name in participants:
#     for run in run_list:
#         short_trial_data(p_name=p_name, run=run)


def add_crner_0cntr_dist_motion(p_list, run_list, eye_track_dir=None, adjust_x=0, adjust_y=0):
    """
    add 'crnr_name' column with corner names
    corner_label_dict = {'45': 'top-right', '135': 'top-left', '225': 'bottom-left', '315': 'bottom-right'}

    I also will add columns where x&y position is zero centered, with (-1, -1) in bottom-left.

    :param p_list: list of participants, used for accessing dirs and files.
    :param run_list: list of run numbers, used for accessing dirs and files.
    :param eye_track_dir: path to folder with participant folders containing run folders and plink csvs.
    :param adjust_x: amount to offset x position if not using 960, 540 as centre
    :param adjust_y: amount to offset y position if not using 960, 540 as centre
    :return:
    """

    print(f"\n*** running add_crner_0cntr_dist_motion()***:\n"
          f"p_list={p_list}, run_list={run_list}, eye_track_dir={eye_track_dir}"
          f"adjust_x: {adjust_x}, adjust_y: {adjust_y}\n")

    if eye_track_dir is None:
        eye_track_dir = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\eyetracking"
    if running_on_laptop():
        eye_track_dir = switch_path(eye_track_dir, 'mac_oneDrive')

    # dicts for mapping new columns
    corner_label_dict = {45: 'top-right', 135: 'top-left', 225: 'bottom-left', 315: 'bottom-right'}
    trgt_0x_dict = {45: 175, 135: -175, 225: -175, 315: 175}
    trgt_0y_dict = {45: 175, 135: 175, 225: -175, 315: -175}


    for p_name in p_list:
        for run in run_list:

            # get file_path and csv_name
            save_path = os.path.join(eye_track_dir, p_name)
            csv_name = f"{p_name}_{run}_eyetrack_short.csv"
            csv_path = os.path.join(save_path, csv_name)
            print(f'eye_track_dir:\n{eye_track_dir}')
            print(f'save_path: {save_path}')
            print(f'csv_name: {csv_name}')
            print(f'csv_path: {csv_path}')
            eye_df = pd.read_csv(csv_path)
            orig_cols = list(eye_df.columns)
            print(f"\neye_df: {eye_df.shape}\n{orig_cols}\n{eye_df.head()}\n")
            print(eye_df.dtypes)

            # add target corner details.
            eye_df['crnr_name'] = eye_df['corner'].map(corner_label_dict)
            eye_df['trgt_X'] = eye_df['corner'].map(trgt_0x_dict)
            eye_df['trgt_Y'] = eye_df['corner'].map(trgt_0y_dict)

            # # zero-centre x and y pos
            fix_x = 960 + adjust_x
            fix_y = 540 + adjust_y
            eye_df['x0_pos'] = eye_df['x_pos'] - fix_x
            eye_df['y0_pos'] = eye_df['y_pos'] - fix_y
            print(f"\neye_df: {eye_df.shape}\n{list(eye_df.columns)}\n{eye_df.head()}\n")


            # # get distance from probe (x_dist, y_dist, sq_dist)
            eye_df['x_dist'] = eye_df['x0_pos'] - eye_df['trgt_X']
            eye_df['y_dist'] = eye_df['y0_pos'] - eye_df['trgt_Y']
            eye_df['sq_dist'] = np.sqrt(np.square(eye_df['x_dist']) + np.square(eye_df['y_dist']))

            # # get distance travelled (x_motion, y_motion, sq_motion)
            eye_df['x_motion'] = eye_df['x_pos'].diff().fillna(0)
            eye_df['y_motion'] = eye_df['y_pos'].diff().fillna(0)
            eye_df['sq_motion'] = np.sqrt(np.square(eye_df['x_motion']) + np.square(eye_df['y_motion']))

            eye_df['cond_name'] = "sep" + eye_df['sep'].astype(str) + "_ISI" + eye_df['ISI'].astype(str)



            all_cols = list(eye_df.columns)
            print(f"\neye_df: {eye_df.shape}\n{all_cols}\n{eye_df.head()}\n")
            new_csv_name = f"{p_name}_{run}_eyetrack_dist_motion.csv"
            new_csv_path = os.path.join(save_path, new_csv_name)
            eye_df.to_csv(new_csv_path, index=False)


# # do one dfs
# participants = ['p1']
# run_list = [1]
# add_crner_0cntr_dist_motion(p_list=participants, run_list=run_list)

# # do all dfs
# participants = ['p1', 'p2']
# run_list = [1, 2, 3]
# add_crner_0cntr_dist_motion(p_list=participants, run_list=run_list)


def saccade_count(p_list, run_list, eye_track_dir=None):
    """
    How many sacaddes are there?
    What segments do they occur in?
    What is there duration (frames/ms)
    What is there magnitude (dist travelled)

    Iterated through all trials.

    Associate response (correct/incorrect)

    1. do saccades predict effect?
        per run - how many sacaddes, blinks and fixations are there (e.g., n_sacaddes, not number of frames containing sac
    2. do saccades predict response?
        Per run -


    :param p_list: list of participants, used for accessing dirs and files.
    :param run_list: list of run numbers, used for accessing dirs and files.
    :param eye_track_dir: path to folder with participant folders containing run folders and plink csvs.
    :return:
    """

    if eye_track_dir is None:
        eye_track_dir = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\eyetracking"
    print(f"\n*** running saccade_count(p_list={p_list}, run_list={run_list}, eye_track_dir={eye_track_dir}) ***\n")

    new_list = []

    for p_name in p_list:
        for run in run_list:

            # get file_path and csv_name
            save_path = os.path.join(eye_track_dir, p_name)
            csv_name = f"{p_name}_{run}_eyetrack_dist_motion.csv"
            csv_path = os.path.join(save_path, csv_name)

            print(f'eye_track_dir: {eye_track_dir}')
            print(f'save_path: {save_path}')
            print(f'csv_name: {csv_name}')
            print(f'csv_path: {csv_path}')
            eye_df = pd.read_csv(csv_path,
                                 # usecols=['trial_num', 'trial_idx', 'segment', 'x_pos', 'y_pos', 'eye_message', 'sep', 'ISI', 'probe_jump', 'corner', 'probeLum', 'resp', 'crnr_name', 'trgt_X', 'trgt_Y', 'x0_pos', 'y0_pos', 'x_dist', 'y_dist', 'sq_dist', 'x_motion', 'y_motion', 'sq_motion']
)
            print(f"\neye_df: {eye_df.shape}\n{list(eye_df.columns)}\n{eye_df.head()}\n")



# # do all dfs
# participants = ['p1', 'p2']
# run_list = [1, 2, 3]
# saccade_count(p_list=participants, run_list=run_list)


def count_errors(p_list, run_list, eye_track_dir=None):
    """
    Make a master df of all trials for all participants.  Get mean number of errors.
    Do errors predict effect (only present for p1, runs 2 and 3).


    :param p_list: list of participants, used for accessing dirs and files.
    :param run_list: list of run numbers, used for accessing dirs and files.
    :param eye_track_dir: path to folder with participant folders containing run folders and plink csvs.
    :return:
    """

    if eye_track_dir is None:
        eye_track_dir = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\eyetracking"
    print(f"\n*** running count_errors(p_list={p_list}, run_list={run_list}, eye_track_dir={eye_track_dir}) ***\n")

    new_list = []

    for p_name in p_list:
        for run in run_list:

            # get file_path and csv_name
            save_path = os.path.join(eye_track_dir, p_name)
            csv_name = f"{p_name}_{run}_eyetrack_dist_motion.csv"
            csv_path = os.path.join(save_path, csv_name)

            print(f'eye_track_dir: {eye_track_dir}')
            print(f'save_path: {save_path}')
            print(f'csv_name: {csv_name}')
            print(f'csv_path: {csv_path}')
            eye_df = pd.read_csv(csv_path,
                                 usecols=['trial_num', 'trial_idx',
                                          'sep', 'ISI', 'probe_jump', 'corner',
                                          'probeLum', 'resp', 'crnr_name']
                                 )
            print(f"\neye_df: {eye_df.shape}\n{list(eye_df.columns)}\n{eye_df.head()}\n")

            trial_numbers = eye_df['trial_num'].unique().tolist()

            # loop through stairs for this isi
            for idx, trial_number in enumerate(trial_numbers):

                # get df just for one trial
                trial_df = eye_df[eye_df['trial_num'] == trial_number]
                print(f'\ntrial_df ({trial_number}) {trial_df.shape}:\n{trial_df}')
                print(f'\ntrial_number: {trial_number}')
                # print(f"\ntrial_df['ISI']: {trial_df['ISI']}")

                isi_val = int(trial_df['ISI'].iloc[0])
                sep_val = int(trial_df['sep'].iloc[0])
                probe_jump = int(trial_df['probe_jump'].iloc[0])
                crnr_name = str(trial_df['crnr_name'].iloc[0])
                probeLum = int(trial_df['probeLum'].iloc[0])
                resp = int(trial_df['resp'].iloc[0])

                # # apend to new dataframe
                new_row = [p_name, run, trial_number, sep_val, isi_val, probe_jump,
                           crnr_name, probeLum, resp]
                new_list.append(new_row)

    # make new output df
    error_count_df = pd.DataFrame(new_list,
                                  columns=['p_name', 'run', 'trial',
                                           'sep_val', 'isi_val', 'probe_jump',
                                           'crnr_name', 'probeLum', 'resp'])
    print(f'\nerror_count_df:\n{error_count_df}')
    print(f"\nerror_count_df: {error_count_df.shape}\n{list(error_count_df.columns)}\n"
          f"{error_count_df.head()}\n")
    error_count_df.to_csv(os.path.join(eye_track_dir, f'error_count.csv'), index=False)


# # do one dfs
# participants = ['p1']
# run_list = [1]
# count_errors(p_list=participants, run_list=run_list)

# # do all dfs
participants = ['p1', 'p2']
run_list = [1, 2, 3]
count_errors(p_list=participants, run_list=run_list)


def trial_dist_from_trgt(p_list, run_list, eye_track_dir=None):
    """
    Does eye position during probes predict response?
    1. get eye positions during probe1 and probe2 (can ignore ISI) (7 or 14 frames max)
    2. calculate mean position distance from probe
    3. calculate closest distance across those frames.
    4. append all participant/run results to new data frame.
        [p_name, run, trial, sep, ISI, probe_jump, crnr_name, resp, mean_sq_dist, min_sq_dist]


    :param p_name: name of participant, used for accessing dirs and files.
    :param run: run number, used for accessing dirs and files.
    :param eye_track_dir: path to folder with participant folders containing run folders and plink csvs.
    :return:
    """

    print(f"\n*** running eye_pos_predict_resp()***:\n"
          f"p_list={p_list}, run_list={run_list}, eye_track_dir={eye_track_dir}")

    if eye_track_dir is None:
        eye_track_dir = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\eyetracking"
    # if using my laptop, adjust dir
    if running_on_laptop():
        eye_track_dir = switch_path(eye_track_dir, 'mac_oneDrive')

    new_list = []

    for p_name in p_list:
        for run in run_list:

            # get file_path and csv_name
            save_path = os.path.join(eye_track_dir, p_name)
            csv_name = f"{p_name}_{run}_eyetrack_dist_motion.csv"
            csv_path = os.path.join(save_path, csv_name)

            print(f'p_name: {p_name}, run: {run}')
            print(f'eye_track_dir: {eye_track_dir}')
            print(f'save_path: {save_path}')
            print(f'csv_name: {csv_name}')
            print(f'csv_path: {csv_path}')
            eye_df = pd.read_csv(csv_path,
                                 usecols=['trial_num', 'segment', 'sep', 'ISI',
                                          'probe_jump', 'crnr_name', 'probeLum',
                                          'resp', 'sq_dist', 'cond_name']
                                 )
            print(f"\neye_df: {eye_df.shape}\n{eye_df.head()}\n")


            # # loop through trials
            trial_numbers = eye_df['trial_num'].unique().tolist()
            for trial in trial_numbers:
                probes_df = eye_df[eye_df['trial_num'] == trial]

                # # just select frames where probes are present
                probe_segments = ['probe1', 'probe2']
                probes_df = probes_df[probes_df['segment'].isin(probe_segments)]
                print(f"\n{trial}. probes_df: {probes_df.shape}\n{probes_df}\n")

                # # get mean distance from targets
                mean_dist = probes_df['sq_dist'].mean()

                # # get closest distance from targets
                min_dist = probes_df['sq_dist'].min()

                # # get other values
                sep_val = int(probes_df['sep'].iloc[0])
                isi_val = int(probes_df['ISI'].iloc[0])
                probe_jump = int(probes_df['probe_jump'].iloc[0])
                cond_name = str(probes_df['cond_name'].iloc[0])
                crnr_name = str(probes_df['crnr_name'].iloc[0])
                probeLum = int(probes_df['probeLum'].iloc[0])
                resp = int(probes_df['resp'].iloc[0])

                # # apend to new dataframe
                new_row = [p_name, run, trial, sep_val, isi_val,
                           cond_name, probe_jump, crnr_name,
                           probeLum, resp, mean_dist, min_dist]
                new_list.append(new_row)


    # make new output df
    dist_from_trgt_df = pd.DataFrame(new_list,
                          columns=['p_name', 'run', 'trial', 'sep_val', 'isi_val',
                                   'cond_name', 'probe_jump','crnr_name',
                                   'probeLum', 'resp', 'mean_dist', 'min_dist'])
    print(f'\ndist_from_trgt_df:\n{dist_from_trgt_df}')
    print(f"\ndist_from_trgt_df: {dist_from_trgt_df.shape}\n{list(dist_from_trgt_df.columns)}\n"
          f"{dist_from_trgt_df.head()}\n")
    dist_from_trgt_df.to_csv(os.path.join(eye_track_dir, f'dist_from_trgt.csv'), index=False)

# # do one dfs
# participants = ['p2']
# run_list = [1]
# trial_dist_from_trgt(p_list=participants, run_list=run_list)
# # do all dfs
# participants = ['p1', 'p2']
# run_list = [1, 2, 3]
# trial_dist_from_trgt(p_list=participants, run_list=run_list)

def dist_log_reg(p_list, run_list, eye_track_dir=None,
                 df_name="dist_from_trgt.csv", predictor='mean_dist'):
    """

    :param p_list:
    :param run_list:
    :param eye_track_dir:
    :param predictor: whether to use mean_dist, min_dist or some other predictor.
    :return:
    """
    print(f"\n*** running dist_log_reg(p_list={p_list}, run_list={run_list}, predictor={predictor}) ***\n")

    if eye_track_dir is None:
        eye_track_dir = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\eyetracking"

    # if using my laptop, adjust dir
    if running_on_laptop():
        eye_track_dir = switch_path(eye_track_dir, 'mac_oneDrive')

    # get file_path and csv_name
    # df_name = "dist_from_trgt.csv"
    csv_path = os.path.join(eye_track_dir, df_name)
    print(f'eye_track_dir: {eye_track_dir}')
    print(f'df_name: {df_name}')
    print(f'csv_path: {csv_path}')
    dist_from_trgt_df = pd.read_csv(csv_path)
    print(f"\ndist_from_trgt_df: {dist_from_trgt_df.shape}\n{list(dist_from_trgt_df.columns)}\n"
          f"{dist_from_trgt_df.head()}\n")

    new_list = []

    for p_name in p_list:

        for run in run_list:

            # # just data for thsi participant/run
            p_run_df = dist_from_trgt_df[dist_from_trgt_df['p_name'] == p_name]
            p_run_df = p_run_df[p_run_df['run'] == run]
            print(f"\np_run_df: {p_run_df.shape}\n{p_run_df.head()}\n")

            # # # # # # # # # # # #
            # logistic regression #
            # # # # # # # # # # # #
            txt_output = []

            # # make directory to save results
            p_run_dir = os.path.join(eye_track_dir, p_name, f"{p_name}_{run}")
            print(f'p_run_dir: {p_run_dir}')
            if not os.path.isdir(os.path.join(p_run_dir, 'regression')):
                os.makedirs(os.path.join(p_run_dir, 'regression'))
            p_run_reg_dir = os.path.join(p_run_dir, 'regression')
            print(f'p_run_reg_dir: {p_run_reg_dir}')

            txt_output.append(f"{p_name}_{run}\n")

            # # get accuracy for weighting model
            num_instances, cols_ = p_run_df.shape
            trial_score = p_run_df['resp'].sum()
            corr_weight = trial_score/num_instances
            incorr_weight = 1-corr_weight
            trial_acc_dict = {0: incorr_weight, 1: corr_weight}
            print(f"trial_acc: {trial_score}/{num_instances}")
            print(f"trial_acc_dict: {trial_acc_dict}")
            txt_output.append(f"instance: {num_instances}, correct_resp: {trial_score}\n"
                              f"weighting:\n{trial_acc_dict}\n")

            # # FIT LOGISTIC REGRESSION MODEL
            X = np.array(p_run_df[predictor].to_list()).reshape(-1, 1)
            y = np.array(p_run_df['resp'].to_list())

            # split the dataset into training (80%) and testing (20%) sets
            test_set_prop = .2
            X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                test_size=test_set_prop,
                                                                random_state=0)

            # instantiate the model
            log_reg = LogisticRegression(
                class_weight='balanced',
                # solver='liblinear',
                # C=10000,
            )
            params_dict = log_reg.get_params()
            print(f"params_dict:\n{params_dict}")

            # fit the model using the training data
            log_reg.fit(X_train, y_train)

            # # get the intercept and coeficient
            intercept = log_reg.intercept_
            slope = log_reg.coef_
            print(f"intercept: {intercept}, slope: {slope}")
            txt_output.append(f"intercept: {intercept}, slope: {slope}\n")

            # use model to make predictions on test data
            y_pred = log_reg.predict(X_test)

            # # get the model score
            score = log_reg.score(X_test, y_test)
            print(f"score: {score}")
            txt_output.append(f"model score: {score}\n")

            # MODEL DIAGNOSTICS
            cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
            print(f"cnf_matrix\n{cnf_matrix}")
            txt_conf_matrix = f"act0\t{cnf_matrix[0][0]}\t{cnf_matrix[0][1]}\n" \
                              f"act1\t{cnf_matrix[1][0]}\t{cnf_matrix[1][1]}\n" \
                              f"    \tpred0\tpred1\n"
            print(f"txt_conf_matrix\n{txt_conf_matrix}")
            txt_output.append(txt_conf_matrix)

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(cnf_matrix)
            ax.grid(False)
            ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
            ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
            ax.set_ylim(1.5, -0.5)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, cnf_matrix[i, j], ha='center', va='center')
            plt.savefig(os.path.join(p_run_reg_dir, f'{predictor}_conf_matrix_png'))
            plt.show()

            report = metrics.classification_report(y_test, y_pred)
            print(f"report:\n{report}")
            txt_output.append(report)

            accuracy = metrics.accuracy_score(y_test, y_pred)
            print(f"accuracy:\n{accuracy}")
            txt_output.append(f"model_accuracy: {accuracy}")

            # # plot ROC curve
            y_pred_proba = log_reg.predict_proba(X_test)[::, 1]
            fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
            auc = metrics.roc_auc_score(y_test, y_pred_proba)
            plt.plot(fpr, tpr, label="AUC=" + str(auc))
            plt.legend(loc=4)
            plt.savefig(os.path.join(p_run_reg_dir, f'{predictor}_roc_auc.png'))
            plt.show()
            txt_output.append(f"auc: {auc}")
            plt.close()

            # get scores for a dummy model that predicts correct for each item
            dummy_score = sum(y_test)/len(y_test)
            model_improvement = score - dummy_score
            print(f"dummy_score: {dummy_score}, model_improvement: {model_improvement}")

            # # # plot logistic function
            plt.clf()  # clear current figure
            # plot true response
            plt.scatter(X_test.ravel(), y_test, color="orange", s=100)
            # plot log_reg predicted response
            plt.scatter(X_test.ravel(), y_pred, color="black")
            # plot logistic function
            loss = expit(X_test * log_reg.coef_ + log_reg.intercept_).ravel()
            plt.plot(X_test, loss, color="red", linewidth=3)

            # add plot details
            plt.ylabel("Response")
            plt.xlabel("Distance from target")
            plt.yticks([0, 1], ['incorrect', 'correct'])
            plt.suptitle(f"{p_name} ({run}): score: {round(score, 2)}, "
                         f"slope: {round(slope[0][0], 3)}, "
                         f"intercept: {round(intercept[0], 5)}\n"
                         f"dummy_score: {round(dummy_score, 2)}, ({round(model_improvement, 2)})",
                         fontsize=20)
            plt.legend(("Log Reg", "True response", "Pred response"), loc="center right")
            plt.savefig(os.path.join(p_run_reg_dir, f'{predictor}_reg_plot.png'))
            plt.show()


            # # save output document
            txt_doc_name = f"{p_name}_{run}_{predictor}_log_reg.txt"
            txt_doc_path = os.path.join(p_run_reg_dir, txt_doc_name)
            with open(txt_doc_path, 'w') as output:
                for row in txt_output:
                    output.write(str(row) + '\n')

            # add regression details to new list
            new_list.append([p_name, run, predictor,
                             params_dict['C'], params_dict['class_weight'],
                             params_dict['solver'], params_dict['penalty'],
                             params_dict['l1_ratio'],
                             test_set_prop, len(y_test),
                             corr_weight, dummy_score,
                             score, model_improvement,
                             intercept, slope,
                             cnf_matrix[0][0], cnf_matrix[0][1],
                             cnf_matrix[1][0], cnf_matrix[1][1]])

    # get means for this model
    new_array = np.array(new_list)
    mean_run_score = new_array[:, 10].mean()
    mean_test_set_score = new_array[:, 11].mean()
    mean_model_score = new_array[:, 12].mean()
    mean_improvement = new_array[:, 13].mean()
    mean_intercept = new_array[:, 14].mean()
    mean_slope = new_array[:, 15].mean()
    mean_TN = new_array[:, 16].mean()
    mean_FP = new_array[:, 17].mean()
    mean_FN = new_array[:, 18].mean()
    mean_TP = new_array[:, 19].mean()

    print(f"mean_model_score: {mean_model_score}, mean_improvement: {mean_improvement}")

    new_list.append(['model_means', "all", predictor,
                     params_dict['C'], params_dict['class_weight'],
                     params_dict['solver'], params_dict['penalty'],
                     params_dict['l1_ratio'],
                     test_set_prop, len(y_test),
                     mean_run_score, mean_test_set_score,
                     mean_model_score, mean_improvement,
                     mean_intercept, mean_slope,
                     mean_TN, mean_FP, mean_FN, mean_TP])

    new_df = pd.DataFrame(new_list,
                          columns=['p_name', 'run', 'predictor',
                                   'neg_reg', 'class_weight',
                                   'solver', 'penalty', 'l1_ratio',
                                   'test_set_prop', 'test_size',
                                   'run_score', 'test_set_score',
                                   'model_score', 'model_improvement',
                                   'intercept', 'slope',
                                   "TN", "FP", 'FN', 'TP'])
    print(f'\nnew_df:\n{new_df}')

    df_path = os.path.join(eye_track_dir, 'log_reg_details.csv')
    if os.path.isfile(df_path):
        orig_df = pd.read_csv(df_path)
        df_list = [orig_df, new_df]
        combined_df = pd.concat(df_list)
        combined_df.to_csv(os.path.join(eye_track_dir, 'log_reg_details.csv'), index=False)
    else:
        new_df.to_csv(os.path.join(eye_track_dir, 'log_reg_details.csv'), index=False)




# # # # do all dfs
# participants = ['p1', 'p2']
# run_list = [1, 2, 3]
# dist_log_reg(p_list=participants, run_list=run_list,
#              df_name="dist_from_trgt.csv", predictor='min_dist')



'''
Does eye movement predict response?
4. get start and end position.
5. calculate direction of travel (could reduce this to 16 directions (N, NNW, NW, WNW etc).
6. could simplify to closer vs further away from probe
7. get probe jump: do eyes move perpendicular?
8. calculate mean speed (sum squares of x and y differences across frames?)
9. how many pixels per second does this correspond to?
10. could this relate to sep or apparent motion?


'''



"""
Part B


At the end I want to know
1. Does eye location predict chance of seeing probe?
2. does direction of movement (e.g., parallel to probes) predict chance of seeing probes?
2b. if so, do eyes move at speed similar probe presentation (e.g., 1 pixel per frame for sep4, ISI4)

"""

def saccade_info(p_list, run_list, eye_track_dir=None):
    """
    for each trial, how many saccades and how many frames did they last.
    # Also, what distance did they travel (straighline, e.g., start x_pos to end x_pos).
    # Was the direction toward target?
    # Were the saccades occuring during probes?

    Get blink info too


    :param p_name: name of participant, used for accessing dirs and files.
    :param run: run number, used for accessing dirs and files.
    :param eye_track_dir: path to folder with participant folders containing run folders and plink csvs.
    :return:
    """
    if eye_track_dir is None:
        eye_track_dir = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\eyetracking"
    # if using my laptop, adjust dir
    if running_on_laptop():
        eye_track_dir = switch_path(eye_track_dir, 'mac_oneDrive')

    new_list = []

    for p_name in p_list:
        for run in run_list:

            # get file_path and csv_name
            save_path = os.path.join(eye_track_dir, p_name)
            csv_name = f"{p_name}_{run}_eyetrack_dist_motion.csv"
            csv_path = os.path.join(save_path, csv_name)

            print(f'eye_track_dir: {eye_track_dir}')
            print(f'save_path: {save_path}')
            print(f'csv_name: {csv_name}')
            print(f'csv_path: {csv_path}')
            eye_df = pd.read_csv(csv_path,
                                 usecols=['trial_num', 'trial_idx', 'cond_name',
                                          'segment',
                                          'x_pos', 'y_pos', 'eye_message',
                                          'sep', 'ISI', 'probe_jump', 'corner',
                                          'probeLum', 'resp', 'crnr_name',
                                          # 'trgt_X', 'trgt_Y', 'x0_pos', 'y0_pos',
                                          # 'x_dist', 'y_dist', 'sq_dist',
                                          # 'x_motion', 'y_motion', 'sq_motion'
                                          ])
            print(f"\neye_df: {eye_df.shape}\n{list(eye_df.columns)}\n{eye_df.head()}\n")


            # # loop through frames for this trial
            trial_list = eye_df['trial_num'].unique().tolist()

            msg_list = eye_df['eye_message'].unique().tolist()
            print(f"msg_list: {msg_list}")


            for trial_num in trial_list:

                # if trial_num > 20:
                #     break
                trial_df = eye_df[eye_df['trial_num'] == trial_num]

                sep_val = int(trial_df['sep'].iloc[0])
                isi_val = int(trial_df['ISI'].iloc[0])
                probe_jump = int(trial_df['probe_jump'].iloc[0])
                cond_name = str(trial_df['cond_name'].iloc[0])
                crnr_name = str(trial_df['crnr_name'].iloc[0])
                probeLum = int(trial_df['probeLum'].iloc[0])
                resp = int(trial_df['resp'].iloc[0])

                segment_list = trial_df['segment'].unique().tolist()

                for segment in segment_list:

                    seg_df = trial_df[trial_df['segment'] == segment]
                    # print(seg_df.head())
                    first_msg = str(seg_df.iloc[0]['eye_message'])

                    fix_count = 0
                    sacc_count = 0
                    blink_count = 0
                    nan_count = 0

                    if first_msg == 'fixation':
                        fix_count += 1
                    elif first_msg == 'saccade':
                        sacc_count += 1
                    elif first_msg == 'blink':
                        blink_count += 1
                    else:
                        nan_count += 1

                    prev_msg = first_msg

                    for idx, row in seg_df.iterrows():

                        this_msg = row['eye_message']

                        if this_msg != prev_msg:

                            if this_msg == 'fixation':
                                fix_count += 1
                            elif this_msg == 'saccade':
                                sacc_count += 1
                            elif this_msg == 'blink':
                                blink_count += 1
                            else:
                                nan_count += 1
                            prev_msg = this_msg

                    new_list.append([p_name, run, trial_num, cond_name, segment,
                                    sep_val, isi_val, probe_jump, crnr_name, probeLum, resp,
                                    fix_count, sacc_count, blink_count, nan_count])
    new_df = pd.DataFrame(new_list,
                          columns=['p_name', 'run', 'trial_num', 'cond_name', 'segment',
                                   'sep_val', 'isi_val', 'probe_jump', 'crnr_name', 'probeLum', 'resp',
                                   'fix_count', 'sacc_count', 'blink_count', 'nan_count'])
    print(f'\nnew_df:\n{new_df}')

    new_df.to_csv(os.path.join(eye_track_dir, 'eye_msg_count.csv'), index=False)



# # do all dfs
# participants = ['p1', 'p2']
# run_list = [1, 2, 3]
# # participants = ['p1']
# # run_list = [1]
# saccade_info(p_list=participants, run_list=run_list)

def get_mean_eye_pos(p_list, run_list, eye_track_dir=None):
    """

    :param p_list: list of participants, used for accessing dirs and files.
    :param run_list: list of run numbers, used for accessing dirs and files.
    :param eye_track_dir: path to folder with participant folders containing run folders and plink csvs.
    :return:
    """
    if eye_track_dir is None:
        eye_track_dir = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\eyetracking"

    new_list = []

    for p_name in p_list:
        for run in run_list:

            # get file_path and csv_name
            save_path = os.path.join(eye_track_dir, p_name)
            csv_name = f"{p_name}_{run}_eyetrack_fixed.csv"
            csv_path = os.path.join(save_path, csv_name)

            print(f'eye_track_dir: {eye_track_dir}')
            print(f'save_path: {save_path}')
            print(f'csv_name: {csv_name}')
            print(f'csv_path: {csv_path}')
            eye_df = pd.read_csv(csv_path, usecols=['time_stamp', 'trial_num', 'trial_frames', 'segment',
                                                    'x_pos', 'y_pos', 'eye_message',
                                                    'sep', 'ISI', 'probe_jump', 'corner', 'probeLum', 'resp'])


            # # loop through frames for this trial
            segments = eye_df['segment'].unique().tolist()
            corners = eye_df['corner'].unique().tolist()
            for idx, segment in enumerate(segments):
                seg_df = eye_df[eye_df['segment'] == segment]

                for idx, corner in enumerate(corners):
                    corner_df = seg_df[seg_df['corner'] == corner]
                    mean_x = corner_df['x_pos'].mean()
                    mean_y = corner_df['y_pos'].mean()

                    print(f"{segment}, {corner}: mean pos: {mean_x}, {mean_y}")

                    new_list.append([p_name, run, segment, corner, mean_x, mean_y])

    new_df = pd.DataFrame(new_list, columns=['p_name', 'run', 'segment', 'corner',
                                             'mean_x', 'mean_y'])
    print(f'\nnew_df:\n{new_df}')

    new_df.to_csv(os.path.join(eye_track_dir, 'mean_eye_pos.csv'), index=False)

# # do all dfs
# participants = ['p1', 'p2']
# run_list = [1, 2, 3]
# get_mean_eye_pos(p_list=participants, run_list=run_list)

def plot_eye_pos(p_name, run, eye_track_dir=None):
    """
    This is just exploratory at this point.

    I am not sure about exactly how to move forward.

    This function plots separate jointplot density functions for x and y positions for
    1. fixation
    2. during probes
    3. during response.

    Plots are coded by the corner the probe appears in.

    Useful for showing systematic bias in the data (e.g., mean fixation is not at zero)


    :param p_name: name of participant, used for accessing dirs and files.
    :param run: run number, used for accessing dirs and files.
    :param eye_track_dir: path to folder with participant folders containing run folders and plink csvs.
    :return: new csv
    """

    print(f"\n*** running plot_eye_movements(p_name={p_name}, run={run}, eye_track_dir={eye_track_dir}) ***\n")

    if eye_track_dir is None:
        eye_track_dir = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\eyetracking"

    # get file_path and csv_name
    save_path = os.path.join(eye_track_dir, p_name)
    df_name = f"{p_name.upper()}_{run}_eyetrack_dist_motion.csv"
    csv_path = os.path.join(save_path, df_name)
    p_run_dir = os.path.join(save_path, f"{p_name}_{run}")
    print(f'eye_track_dir: {eye_track_dir}')
    print(f'save_path: {save_path}')
    print(f'p_run_dir: {p_run_dir}')
    print(f'df_name: {df_name}')
    print(f'csv_path: {csv_path}')
    eye_df = pd.read_csv(csv_path, usecols=['trial_num', 'trial_idx', 'cond_name', 'segment',
                                            'x_pos', 'y_pos', 'eye_message',
                                            'corner', 'crnr_name',
                                            'sep', 'ISI', 'probe_jump', 'probeLum', 'resp']
                         )
    print(f"\neye_df: {eye_df.shape}\n{list(eye_df.columns)}\n{eye_df.head()}\n")

    # sort corners to use as hue
    crnr_palette = {'top-right': 'tab:blue', 'top-left': 'tab:green', 'bottom-left': 'tab:orange',
                    'bottom-right': 'tab:red'}

    if 'crnr_name' not in list(eye_df.columns):
    # change corner to str, so it is categorical not numeric
        eye_df = eye_df.astype({'corner': 'str'})
        corner_label_dict = {'45': 'top-right', '135': 'top-left', '225': 'bottom-left', '315': 'bottom-right'}
        eye_df['crnr_name'] = eye_df['corner'].map(corner_label_dict)

    # # use expected locations for fixation and probes (e.g., fix_x = 1920/2; fix_y = 1080/2)
    fix_x = 960
    fix_y = 540
    dist_from_fix = 175
    min_x = fix_x - dist_from_fix
    max_x = fix_x + dist_from_fix
    min_y = fix_y - dist_from_fix
    max_y = fix_y + dist_from_fix
    zoomed_in_bounds = 50
    print(f"fix pos: ({fix_x}, {fix_y})\nplot bounds (+/- {dist_from_fix}): ({min_x}, {max_x}), ({min_y}, {max_y})")
    print(f"fzoomed in bounds (+/- {zoomed_in_bounds}): ({fix_x - zoomed_in_bounds}, {fix_x + zoomed_in_bounds}), "
          f"({fix_y - zoomed_in_bounds}, {fix_y + zoomed_in_bounds})")

    # # make three plots: fixation, probes(&ISI) and response

    # # # this run mean pos during fixation
    print(f"\nplotting fixation positions: {p_name}_{run}")
    fix_df = eye_df[eye_df['segment'] == 'fix_dot']
    # mean_fix_x = fix_df['x_pos'].mean()
    # mean_fix_y = fix_df['y_pos'].mean()

    g = sns.jointplot(data=fix_df, x="x_pos", y="y_pos",
                      hue='crnr_name', palette=crnr_palette,
                      xlim=(min_x, max_x), ylim=(min_y, max_y),
                      kind="kde",
                      joint_kws=dict(fill=True, thresh=0.05, alpha=.25, palette=crnr_palette),
                      marginal_kws=dict(fill=True, alpha=.25, palette=crnr_palette))

    # draw lines on the joint plot and on margin plots
    for ax in (g.ax_joint, g.ax_marg_x):
        ax.axvline(fix_x, color='grey', ls='--', lw=1)
    for ax in (g.ax_joint, g.ax_marg_y):
        ax.axhline(fix_y, color='grey', ls='--', lw=1)

    # # (0, 0) in top-left corner (not bottom left, so flip y axis)
    plt.gca().invert_yaxis()

    # add title and update legend
    plt.suptitle(f"{p_name}_{run}: fixation")

    # save and show plots
    fig_name = f"eye_pos_in_fix_{p_name}_{run}.png"
    plt.savefig(os.path.join(p_run_dir, 'eye_pos', fig_name))
    plt.show()

    # # # this run mean pos during response time
    print(f"\nplotting response positions: {p_name}_{run}")
    resp_df = eye_df[eye_df['segment'] == 'response']
    g = sns.jointplot(data=resp_df, x="x_pos", y="y_pos",
                      xlim=(min_x, max_x),
                      ylim=(min_y, max_y),
                      hue='crnr_name', palette=crnr_palette,
                      kind="kde", joint_kws=dict(fill=True, thresh=0.05,
                                                 alpha=.25, palette=crnr_palette),
                      marginal_kws=dict(fill=True, alpha=.25, palette=crnr_palette))

    # draw lines on the joint plot and on margin plots
    for ax in (g.ax_joint, g.ax_marg_x):
        ax.axvline(fix_x, color='grey', ls='--', lw=1)
    for ax in (g.ax_joint, g.ax_marg_y):
        ax.axhline(fix_y, color='grey', ls='--', lw=1)

    # # (0, 0) in top-left corner (not bottom left, so flip y axis)
    plt.gca().invert_yaxis()
    plt.suptitle(f"{p_name}_{run}: response")

    fig_name = f"eye_pos_in_resp_{p_name}_{run}.png"
    plt.savefig(os.path.join(p_run_dir, 'eye_pos', fig_name))

    plt.show()

    # # # this run mean pos probes
    print(f"\nplotting position during probes positions: {p_name}_{run}")
    probes_df = eye_df.loc[eye_df['segment'].isin(['probe1', 'ISI', 'probe2'])]
    print(f"\nprobes_df: {probes_df.shape}\n{list(probes_df.columns)}\n"
          f"{probes_df.head()}\n{probes_df.tail()}\n")

    g = sns.jointplot(data=probes_df, x="x_pos", y="y_pos",
                      xlim=(min_x, max_x),
                      ylim=(min_y, max_y),
                      hue='crnr_name', palette=crnr_palette,
                      kind="kde", joint_kws=dict(fill=True, thresh=0.05,
                                                 alpha=.25, palette=crnr_palette),
                      marginal_kws=dict(fill=True, alpha=.25, palette=crnr_palette))

    # draw lines on the joint plot and on margin plots
    for ax in (g.ax_joint, g.ax_marg_x):
        ax.axvline(fix_x, color='grey', ls='--', lw=1)
    for ax in (g.ax_joint, g.ax_marg_y):
        ax.axhline(fix_y, color='grey', ls='--', lw=1)

    # # (0, 0) in top-left corner (not bottom left, so flip y axis)
    plt.gca().invert_yaxis()
    plt.suptitle(f"{p_name}_{run}: during probes")

    fig_name = f"eye_pos_probes_{p_name}_{run}.png"
    plt.savefig(os.path.join(p_run_dir, 'eye_pos', fig_name))

    plt.show()


# do one df
# old_csv_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\eyetracking\p1\P1_1_pylink.csv"
# p_name = 'p1'
# run = 1
# plot_eye_pos(p_name=p_name, run=run)

# # do all dfs
# participants = ['p1', 'p2']
# run_list = [1, 2, 3]
# for p_name in participants:
#     for run in run_list:
#         plot_eye_pos(p_name=p_name, run=run)

def plot_eye_movements(p_name, run, eye_track_dir=None):
    """

    This function plots the eye-movements per trial.

    
    I've plotted:
        eye-movement for each trial (fix, p1, isi, p2) excluding movements during response.
        zoomed-in eye-movement for each trial (fix, p1, isi, p2) excluding movements during response.


    :param p_name: name of participant, used for accessing dirs and files.
    :param run: run number, used for accessing dirs and files.
    :param eye_track_dir: path to folder with participant folders containing run folders and plink csvs.
    :return: new csv
    """

    print(f"\n*** running plot_eye_movements(p_name={p_name}, run={run}, eye_track_dir={eye_track_dir}) ***\n")

    if eye_track_dir is None:
        eye_track_dir = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\eyetracking"

    # get file_path and csv_name
    save_path = os.path.join(eye_track_dir, p_name)
    df_name = f"{p_name.upper()}_{run}_eyetrack_dist_motion.csv"
    csv_path = os.path.join(save_path, df_name)
    p_run_dir = os.path.join(save_path, f"{p_name}_{run}")
    print(f'eye_track_dir: {eye_track_dir}')
    print(f'save_path: {save_path}')
    print(f'p_run_dir: {p_run_dir}')
    print(f'df_name: {df_name}')
    print(f'csv_path: {csv_path}')
    eye_df = pd.read_csv(csv_path,
                         usecols=['trial_num', 'trial_idx', 'segment',
                                  'x_pos', 'y_pos',
                                  'eye_message', 'sep', 'ISI', 'resp',
                                  'corner', 'crnr_name',
                                  'x0_pos', 'y0_pos', ]

                         # usecols=['time_stamp', 'trial_num', 'trial_frames', 'segment',
                         #                    'x_pos', 'y_pos', 'eye_message',
                         #                    'sep', 'ISI', 'probe_jump', 'corner', 'probeLum', 'resp']
                         )
    print(f"\neye_df: {eye_df.shape}\n{list(eye_df.columns)}\n{eye_df.head()}\n")

    # # sort corners to use as hue
    # # change corner to str, so it is categorical not numeric
    # eye_df = eye_df.astype({'corner': 'str'})
    # corner_label_dict = {'45': 'top-right', '135': 'top-left', '225': 'bottom-left', '315': 'bottom-right'}
    # # corner_palette = {'45': 'tab:blue', '135': 'tab:green', '225': 'tab:orange', '315': 'tab:red'}
    # crnr_palette = {'top-right': 'tab:blue', 'top-left': 'tab:green', 'bottom-left': 'tab:orange', 'bottom-right': 'tab:red'}
    # eye_df['crnr_name'] = eye_df['corner'].map(corner_label_dict)
    #
    # # use expected locations for fixation and probes (e.g., fix_x = 1920/2; fix_y = 1080/2)
    fix_x = 0  # 960
    fix_y = 0  # 540
    dist_from_fix = 175
    min_x = fix_x - dist_from_fix
    max_x = fix_x + dist_from_fix
    min_y = fix_y - dist_from_fix
    max_y = fix_y + dist_from_fix
    zoomed_in_bounds = 50
    print(f"fix pos: ({fix_x}, {fix_y})\nplot bounds (+/- {dist_from_fix}): ({min_x}, {max_x}), ({min_y}, {max_y})")
    print(f"fzoomed in bounds (+/- {zoomed_in_bounds}): ({fix_x-zoomed_in_bounds}, {fix_x+zoomed_in_bounds}), "
          f"({fix_y-zoomed_in_bounds}, {fix_y+zoomed_in_bounds})")

    # make plots for each trial

    # # loop through trials for this run
    trial_numbers = eye_df['trial_num'].unique().tolist()
    for idx, trial_number in enumerate(trial_numbers):

        if trial_number in list(range(0, 101)):
            print(f'\ntrial_number: {trial_number}')

            # get df just for one stair at this isi
            trial_df = eye_df[eye_df['trial_num'] == trial_number]

            # # remove 'response' frames after probes dissapear
            # trial_df = trial_df[trial_df['segment'] != 'response']

            # get run details
            sep = int(trial_df['sep'].iloc[0])
            ISI = int(trial_df['ISI'].iloc[0])
            # probe_jump = int(trial_df['probe_jump'].iloc[0])
            # probe_dir = 'CW'
            # if probe_jump == -1:
            #     probe_dir = 'ACW'
            corner = int(trial_df['corner'].iloc[0])
            crnr_name = str(trial_df['crnr_name'].iloc[0])
            # probeLum = float(trial_df['probeLum'].iloc[0])
            resp = int(trial_df['resp'].iloc[0])
            response = 'correct'
            if resp == 0:
                response = 'incorrect'

            colour_dict = {'fix_dot': 'lightblue', 'probe1': 'red', 'ISI': 'blue', 'probe2': 'green', 'response': 'yellow'}


            # # plot1 - zoomed out to include probe locations.
            fig, ax = plt.subplots(figsize=(6, 6))
            sns.lineplot(data=trial_df, x="x0_pos", y="y0_pos", sort=False, hue='segment',
                         palette=colour_dict,
                         style='eye_message',
                         hue_order=['fix_dot', 'response', 'probe2', 'ISI', 'probe1'],
                         style_order=['fixation', 'saccade', 'blink']
                         )

            # plot extends to assumed probe locations
            # plt.plot((fix_x-dist_from_fix, fix_x+dist_from_fix), (fix_y-dist_from_fix, fix_y+dist_from_fix),
            #          linestyle='dashed', color='grey')
            # plt.plot((fix_x-dist_from_fix, fix_x+dist_from_fix), (fix_y+dist_from_fix, fix_y-dist_from_fix),
            #          linestyle='dashed', color='grey')
            plt.plot((min_x, max_x), (min_y, max_y), linestyle='dashed', color='grey')
            plt.plot((min_x, max_x), (max_y, min_y), linestyle='dashed', color='grey')

            # # add indication of where probe is
            if corner == 45:
                dot_x, dot_y = fix_x + dist_from_fix, fix_y - dist_from_fix
            elif corner == 135:
                dot_x, dot_y = fix_x - dist_from_fix, fix_y - dist_from_fix
            elif corner == 225:
                dot_x, dot_y = fix_x - dist_from_fix, fix_y + dist_from_fix
            elif corner == 315:
                dot_x, dot_y = fix_x + dist_from_fix, fix_y + dist_from_fix
            else:
                raise ValueError('which corner?')
            plt.plot(dot_x, dot_y, marker="o", markersize=20)

            # # (0, 0) in top-left corner (not bottom left, so flip y axis)
            plt.gca().invert_yaxis()
            # plt.title(f"Probes: {p_name}_{run}: {trial_number} sep{sep} ISI{ISI}.  {corner_label_dict[corner]}: {response}")
            plt.title(f"Probes: {p_name}_{run}: {trial_number} sep{sep} ISI{ISI}.  {crnr_name}: {response}")
            fig_name = f"probes_{p_name}_{run}_{trial_number}_ISI{ISI}_sep{sep}_{response[:3]}.png"
            plt.savefig(os.path.join(p_run_dir, 'eye_movement', fig_name))
            # plt.show()

            # # plot2 - zoomed in to focus on fixatation
            fig, ax = plt.subplots(figsize=(6, 6))
            # sns.lineplot(data=trial_df, x="x_pos", y="y_pos", sort=False, hue='segment', palette=colour_dict,
            sns.lineplot(data=trial_df, x="x0_pos", y="y0_pos", sort=False, hue='segment', palette=colour_dict,
                         style='eye_message',
                         hue_order=['fix_dot', 'response', 'probe2', 'ISI', 'probe1'],
                         style_order=['fixation', 'saccade', 'blink']

                         )

            # plot extends to assumed probe locations
            plt.plot((fix_x - zoomed_in_bounds, fix_x + zoomed_in_bounds), (fix_y - zoomed_in_bounds, fix_y + zoomed_in_bounds),
                     linestyle='dashed', color='grey')
            plt.plot((fix_x - zoomed_in_bounds, fix_x + zoomed_in_bounds), (fix_y + zoomed_in_bounds, fix_y - zoomed_in_bounds),
                     linestyle='dashed', color='grey')

            # # add indication of where probe is
            if corner == 45:
                dot_x, dot_y = fix_x + zoomed_in_bounds, fix_y - zoomed_in_bounds
            elif corner == 135:
                dot_x, dot_y = fix_x - zoomed_in_bounds, fix_y - zoomed_in_bounds
            elif corner == 225:
                dot_x, dot_y = fix_x - zoomed_in_bounds, fix_y + zoomed_in_bounds
            elif corner == 315:
                dot_x, dot_y = fix_x + zoomed_in_bounds, fix_y + zoomed_in_bounds
            else:
                raise ValueError('which corner?')
            plt.plot(dot_x, dot_y, marker="X", markersize=20, markeredgecolor="yellow", markerfacecolor="orange")

            # # (0, 0) in top-left corner (not bottom left, so flip y axis)
            plt.gca().invert_yaxis()
            plt.title(f"ZoomIn: {p_name}_{run}: {trial_number} sep{sep} ISI{ISI}.  {crnr_name}: {response}")
            fig_name = f"zoomed_{p_name}_{run}_{trial_number}_ISI{ISI}_sep{sep}_{response[:3]}.png"
            plt.savefig(os.path.join(p_run_dir, 'eye_movement', fig_name))
            # plt.show()


# # do one df
# old_csv_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\eyetracking\p1\P1_1_pylink.csv"
# p_name = 'p1'
# run = 1
# plot_eye_movements(p_name=p_name, run=run)

# # # do all dfs
# participants = ['p1', 'p2']
# run_list = [1, 2, 3]
# for p_name in participants:
#     for run in run_list:
#         plot_eye_movements(p_name=p_name, run=run)




print('analysis script ended')