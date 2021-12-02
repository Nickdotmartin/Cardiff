import numpy as np
import psignifit as ps
import pandas as pd
import matplotlib.pyplot as plt

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
exp_csv_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/Kim/' \
               'Nick_practice/P6a-Kim/ISI_24_probeDur2/Kim1.csv'
# todo: use stair column for each sep.  Use all data for more datapoints per sep.?
raw_data_df = pd.read_csv(exp_csv_path,
                          usecols=[
                              # 'stair',
                              'probeLum', 'trial_response']
                          )
print(f"raw_data:\n{raw_data_df.head()}")

# identify col of interest (e.g., probeLum)
quartile_bins = False
# cond_col = 'stair'
focus_col = 'probeLum'
resp_correct = 'trial_response'


# get useful info
n_rows, n_cols = raw_data_df.shape
print(f"\nn_rows: {n_rows}, n_cols: {n_cols}")
focus_min = raw_data_df[focus_col].min()
focus_max = raw_data_df[focus_col].max()
print(f"focus min, max: {focus_min}, {focus_max}")


# put responses into bins (e.g., 10)
n_bins = 10
# # use pd.qcut for bins containing an equal number of itemsbased on distribution.
# # i.e., bins will not be of equal size but will have equal value count
if quartile_bins:
    bin_col, bin_labels = pd.qcut(x=raw_data_df[focus_col], q=n_bins,
                                  precision=3, retbins=True)
# # use pd.cut for bins of equal size based on values, but value count may vary.
else:
    bin_col, bin_labels = pd.cut(x=raw_data_df[focus_col], bins=n_bins,
                                 precision=3, retbins=True, ordered=True)
# bin_labels gives the left(min) value for the bin interval
bin_labels = bin_labels[:n_bins]
# print(f"\nbin_labels ({len(bin_labels)}-{type(bin_labels)}):\n{bin_labels}")
raw_data_df['bin_col'] = bin_col
# # get bin_values which are type(pandas.Interval)
bins = bin_col.unique()

# get n_trials for each bin
bin_count = pd.value_counts(raw_data_df['bin_col'])
print(f"\nbin_count:\n{bin_count}")

# get bin_values for empty bins
empty_bin_idxs = bin_count[bin_count == 0].index

# loop through bins and get correct per bin
data_arr = []
found_bins_left = []
for idx, bin_interval in enumerate(bins):
    this_bin_df = raw_data_df.loc[raw_data_df['bin_col'] == bin_interval]
    correct_per_bin = this_bin_df['trial_response'].sum()
    this_bin_vals = [bin_interval.left, bin_interval.right]
    data_arr.append([bin_interval.left, this_bin_vals, correct_per_bin, bin_count[idx]])
    found_bins_left.append(round(bin_interval.left, 3))

# check for empty bins and add if necessary
if len(found_bins_left) < n_bins:
    print(f'\nOnly found {len(found_bins_left)} bins, adding empty bins')

    for empty_idx, empty in enumerate(empty_bin_idxs):
        data_arr.append([empty.left, [empty.left, empty.right], 0, 0])

data_df = pd.DataFrame(data_arr, columns=['bin_left', 'stim_level', 'n_correct', 'n_total'])
data_df = data_df.sort_values(by='bin_left', ignore_index=True)
data_df = data_df.drop(columns='stim_level')
# data_df = data_df.astype({'n_correct': 'int32', 'n_total': 'int32'})
print(f"data_df:\n{data_df}")
# print(f"data_df:\n{data_df.dtypes}")


# # # # 2. convert data into format for analysis: 3 cols [stimulus level | nCorrect | ntotal]
# dtypes=[("bin_left", 'float32'),
#         ("n_correct", 'int32'),
#         ("n_total", 'int32')]
data = data_df.to_numpy()
# data
print(f"data:\n{data}")

#
# # # 3. run psignifit for fit, conf intervals and threshold etc
# """
#  remark: This format differs slightly from the format used in older
#  psignifit versions. You can convert your data by using the reformat
#  comand. If you are a user of the older psignifits.
#
#
#  --- CONSTRUCT THE OPTIONS DICTONARY ---
#
#  To start psignifit you need to pass a dictionary, which specifies, what kind
#  of experiment you did and any other parameters of the fit you might want
#
#
#  You can create an empty dictionary by simply calling <name> = dict()
# """
#
# options = dict()  # initialize as an empty dict
#
# """
#  Now you can set the different options with lines of the form
#  <name>['<key>'] as in the following lines:
# """
#
# options['sigmoidName'] = 'norm'  # choose a cumulative Gauss as the sigmoid
# options['expType'] = '2AFC'  # choose 2-AFC as the paradigm of the experiment
# # this sets the guessing rate to .5 and
# # fits the rest of the parameters '''
#
# """
#  There are 3 other types of experiments supported out of the box:
#
#  n alternative forces choice: The guessing rate is known.
#        options.expType = "nAFC"
#        options.expN    = <number of alternatives>
#  Yes/No experiments: A free guessing and lapse rate is estimated
#        options.expType = "YesNo"
#  equal asymptote: As Yes/No, but enforces that guessing and lapsing occure
#  equally often
#        options.expType = "equalAsymptote"
#
#  Out of the box psignifit supports the following sigmoid functions,
#  choosen by:
#  options.sigmoidName = ...
#
#  'norm'        a cummulative gauss distribution
#  'logistic'    a logistic function
#  'gumbel'      a cummulative gumbel distribution
#  'rgumbel'     a reversed gumbel distribution
#  'tdist'       a t-distribution with df=1 as a heavytail distribution
#
#  for positive stimulus levels which make sence on a log-scale:
#  'logn'        a cumulative lognormal distribution
#  'Weibull'     a Weibull function
#
#  There are many other options you can set in the options-file. You find
#  them in demo_002
#
#
#  --- NOW RUN PSIGNIFIT ---
#
#  Now we are ready to run the main function, which fits the function to the
#  data. You obtain a struct, which contains all the information about the
#  fitted function and can be passed to the many other functions in this
#  toolbox, to further process the results.
# """
#
# res = ps.psignifit(data, options)
#
# """
#  --- VISUALIZE THE RESULTS ---
#
#  For example you can use the result dict res to plot your psychometric
#  function with the data:
# """
#
# ps.psigniplot.plotPsych(res)
#
# plt.show()
#
# # # 4. Plot psychometric function


