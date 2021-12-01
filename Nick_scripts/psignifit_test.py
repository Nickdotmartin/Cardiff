import numpy as np
import psignifit as ps
import pandas as pd


"""
see psignifit instruction here: https://github.com/wichmann-lab/python-psignifit/wiki
demos at: https://github.com/wichmann-lab/psignifit/wiki/Experiment-Types 
or:/Users/nickmartin/opt/anaconda3/envs/Cardiff3.6/lib/python3.6/site-packages/psignifit/demos
This script contains the analysis pipeline for individual participants.
1. load in CSV file.
2. convert data into format for analysis: 3 cols [stimulus level | nCorrect | ntotal]
3. run psignifit for fit, conf intervals and threshold etc
4. Plot psychometric function
"""

# # 1. load in CSV file.
raw_data_df = pd.read_csv('/Users/nickmartin/Documents/PycharmProjects/Cardiff/'
                       'Nick_scripts/testnm_50/ISI_12_probeDur2/testnm_50.csv',
                          usecols=['probeLum', 'trial_response']
)
print(f"raw_data:\n{raw_data_df.head()}")

# identify col of interest (e.g., probeLum)
focus_col = 'probeLum'
resp_correct = 'trial_response'

# get min and max values for variable of interest to work out bin size
n_bins = 10

focus_min = raw_data_df[focus_col].min()
focus_max = raw_data_df[focus_col].max()
print(f"\nfocus min, max: {focus_min}, {focus_max}")
bin_size = (focus_max - focus_min)/10
print(f"bin_size: {bin_size}")


# put responses into bins (e.g., 10)
raw_data_df['binned'] = pd.cut(x=raw_data_df[focus_col], bins=n_bins)

# get bin_values which are type(pandas.Interval)
bins = raw_data_df.binned.unique()

# for each bin (e.g., probeLum value) get n_trials and n_correct

# get n_correct for each bin
bin_count = pd.value_counts(raw_data_df['binned'])
print(f"\nbin_count:\n{bin_count}")

# loop through bins and use bin_count to get data for psignifit.
# todo: change this to look through bin_count which also contains empty bins
hist_df = []
for idx, bin_interval in enumerate(bins):

    this_bin_df = raw_data_df.loc[raw_data_df['binned'] == bin_interval]
    correct_per_bin = this_bin_df['trial_response'].sum()
    this_bin_vals = [bin_interval.left, bin_interval.right]
    print(idx, this_bin_vals, bin_count[idx], correct_per_bin)
    hist_df.append([bin_interval.left, this_bin_vals, correct_per_bin, bin_count[idx]])

# convert into new np.array
new_df = pd.DataFrame(hist_df, columns=['bin_left', 'stim_level', 'n_correct', 'n_total'])
new_df = new_df.sort_values(by='bin_left')
new_df = new_df.drop(columns='bin_left')
print(f"new_df:\n{new_df}")

# # 2. convert data into format for analysis: 3 cols [stimulus level | nCorrect | ntotal]
data = new_df.to_numpy()
print(f"data:\n{data}")

# # 3. run psignifit for fit, conf intervals and threshold etc


# # 4. Plot psychometric function


