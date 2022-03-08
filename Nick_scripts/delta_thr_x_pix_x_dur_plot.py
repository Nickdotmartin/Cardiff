import matplotlib.pyplot as plt
import numpy as np
import psignifit as ps
import pandas as pd
import seaborn as sns
import os
from rad_flow_psignifit_analysis import plot_w_errors_either_x_axis, simple_line_plot
'''
This script is to convert our data such that the y axis of the plots reflects 
something similar to number of photons per conditions.  For this we need to do a few things.
1. convert thresholds into delta thr by subtracting BGLum (21.2) then dividing by BGLum.
2. All 1probe conditions are multiplied by 10 (5 pixels x 2 frames)
3. All other conditions multiplied by 40 ((2x5 pixels) x (2x2 frames)

The most suitable plot function might be plot_w_errors_either_x_axis.
However, this requires a longform all_data_df.
For now I'll do it without error bars to keep it simple.
'''

# def simple_line_plot(indexed_df, fig_title=None, legend_title=None,
#                      x_tick_vals=None, x_tick_labels=None,
#                      x_axis=None, y_axis=None,
#                      log_x=False, log_y=False,
#                      save_as=None):
#     """
#     Function to plot a simple line plot.  No error bars.
#     :param indexed_df: DF where index col is 'separation' or 'stair_names' etc.
#     :param fig_title: Title for figure
#     :param legend_title: Title for legend
#     :param x_tick_vals: Values for x-ticks
#     :param x_tick_labels: Labels for x ticks
#     :param x_axis: Label for x-axis
#     :param y_axis: Label for y-axis
#     :param log_x: Make x-axis log scale
#     :param log_y: Make y-axis log scale
#     :param save_as: Full path (including name) to save to
#     :return: Figure
#     """
#     fig, ax = plt.subplots(figsize=(10, 6))
#     sns.lineplot(data=indexed_df, markers=True, dashes=False, ax=ax)
#     if fig_title is not None:
#         plt.title(fig_title)
#     if legend_title is not None:
#         plt.legend(title=legend_title)
#     if x_tick_vals is not None:
#         ax.set_xticks(x_tick_vals)
#     if x_tick_labels is not None:
#         ax.set_xticklabels(x_tick_labels)
#         if -18 in x_tick_labels:
#             # add dotted line at zero
#             ax.axvline(x=5.5, linestyle="-.", color='lightgrey')
#     if log_x:
#         ax.set(xscale="log")
#         x_axis = f'log {x_axis}'
#     if log_y:
#         ax.set(yscale="log")
#         y_axis = f'log {y_axis}'
#     if x_axis is not None:
#         ax.set_xlabel(x_axis)
#     if y_axis is not None:
#         ax.set_ylabel(y_axis)
#     if save_as is not None:
#         plt.savefig(save_as)
#     return fig


root_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/'

exp_list = ['exp1a_data', 'radial_flow_exp']
exp_dir = 'radial_flow_exp'
ave_thr_name = 'MASTER_exp_ave_thr.csv'
#
# exp_list = ['rad_flow_2/Nick', 'Exp2_Bloch_NM/Nick', 'Exp3_Ricco_NM/Nick']
# exp_dir = 'Exp3_Ricco_NM/Nick'
# ave_thr_name = 'MASTER_ave_thresh.csv'


# load average thr df
ave_thr_df_path = os.path.join(root_path, exp_dir, ave_thr_name)
ave_thr_df = pd.read_csv(ave_thr_df_path)
print(f'n cols = {len(ave_thr_df.columns.tolist())}')
if 'separation' in ave_thr_df.columns.tolist():
    # exp1a
    index_col_name = 'separation'
    ave_thr_df = ave_thr_df.set_index(index_col_name)

    # RICCO
    if len(ave_thr_df.columns.tolist()) == 1:
        ave_thr_df.columns = ['thr']

elif 'stair_names' in ave_thr_df.columns.tolist():
    # rad_flow/rad_flow_2
    index_col_name = 'stair_names'
    ave_thr_df = ave_thr_df.set_index(index_col_name)
elif len(ave_thr_df.columns.tolist()) == 2:
    print('found 2 columns')
    # bloch
    ave_thr_df.columns = ['cond', 'thr']
    index_col_name = 'cond'
    ave_thr_df = ave_thr_df.set_index(index_col_name)

# reorder columns
if 'ISI_8' in ave_thr_df.columns.to_list():
    print('found isi_8')
    # new_cols_order = ['ISI_1', 'ISI_4', 'ISI_6', 'ISI_8', 'ISI_9', 'ISI_10', 'ISI_12']
    new_cols_order = ['ISI_1', 'ISI_4', 'ISI_6', 'ISI_9', 'ISI_12']
    ave_thr_df = ave_thr_df[new_cols_order]



# 1probe indices to change
one_probe_vals = [-99, -20, 20, 110, 'ISI_-99.0']
index_col = ave_thr_df.index.to_list()
if -18 not in index_col:
    # experiments with -18 do not include 1probe
    sep_vals = [-1 if i in one_probe_vals else i for i in index_col]
    sep_labels = ['1pr' if i == -1 else i for i in sep_vals]
    one_probe_matches = list(set(index_col).intersection(one_probe_vals))
    if len(one_probe_matches) == 1:
        one_probe_name = one_probe_matches[0]
        print(f'\nFOUND\none_probe_name: {one_probe_name}\n')
    else:
        raise ValueError(f'\nFOUND\nseveral one probe matches: {one_probe_matches}')
else:
    sep_vals = index_col
    sep_labels = sep_vals
sep_range = list(range(len(index_col)))
print(f'index_col:{index_col}\n'
      f'sep_range: {sep_range}\n'
      f'sep_vals: {sep_vals}\n'
      f'sep_labels: {sep_labels}')

log_y_axis = True

# todo: convert to cd/m2?
'''At each trial, a white probe of _ cd/m2 luminance appeared at one of four 
meridians (45, 135, 225, or 315 degrees) and at 4 dva eccentricity. 
The probe consisted of 5 pixels (0.49 mm2) arranged in two diagonal lines, 
one of 3 pixels and the other of 2 pixels subtending a maximum visual angle of 0.13 dva.

307.65 cd/m2 and the minimum was 0.1 cd/m2
'''
mon_cdm2_min, mon_cdm2_max = 0.1, 307.65
rgb_255_min, rgb_255_max = 0, 255
bg_cdm2 = 80

# convert to delta thr
bgLum = 21.2
delta_thr_df = (ave_thr_df - bgLum)/bgLum

# fake_data = {0: [1, 1, 1, 1, 1, 1, 1, 1],
#              1: [1, 1, 1, 1, 1, 1, 1, 1],
#              2: [1, 1, 1, 1, 1, 1, 1, 1],
#              3: [1, 1, 1, 1, 1, 1, 1, 1],
#              6: [1, 1, 1, 1, 1, 1, 1, 1],
#              18: [1, 1, 1, 1, 1, 1, 1, 1],
#              20: [1, 1, 1, 1, 1, 1, 1, 1]}
# delta_thr_df = pd.DataFrame.from_dict(data=fake_data, orient='index',
#                                     columns=['Concurrent', 'ISI0', 'ISI2', 'ISI4', 'ISI6', 'ISI9', 'ISI12', 'ISI24'])
# delta_thr_df.index.names = ['separation']

print(f'\ndelta_thr_df:\n{delta_thr_df}\n')

# line plot for delta ISIs
delta_isi_df = delta_thr_df.copy()
delta_isi_df.index = sep_range
print(f'delta_isi_df:\n{delta_isi_df}\n')

fig1_title = f'{exp_dir}: ∆thr for ISIs'
fig1_savename = 'delta_thr_isi.png'
save_as = os.path.join(root_path, exp_dir, fig1_savename)
simple_line_plot(indexed_df=delta_isi_df, fig_title=fig1_title,
                 legend_title='ISI',
                 x_tick_vals=sep_range, x_tick_labels=sep_labels,
                 x_axis='Separation (diagonal pixels)',
                 y_axis='∆ Threshold',
                 log_y=log_y_axis,
                 save_as=save_as
                 )
plt.show()

# only make these plots if there are more than 2 columns
if 'thr' in ave_thr_df.columns.to_list():
    print('Not making transposed fig')
else:
    # line plot for delta separations
    delta_sep_df = delta_thr_df.copy().T
    delta_sep_df.columns = sep_labels
    print(f'delta_sep_df:\n{delta_sep_df}\n')
    fig2_title = f'{exp_dir}: ∆thr for Separations'
    fig2_savename = 'delta_thr_sep.png'
    save_as = os.path.join(root_path, exp_dir, fig2_savename)
    simple_line_plot(indexed_df=delta_sep_df, fig_title=fig2_title,
                     legend_title='Separation',
                     x_axis='ISI (inter-stimulus interval)',
                     y_axis='∆ Threshold',
                     log_y=log_y_axis,
                     save_as=save_as
                     )
plt.show()



# weight by pixels and duration
'''for most conds, thr is multiplied by 40 (2x5 active pixels for 2x2 frames).
for concurrent, conds are multipies by 20 (2x5 active pixels for 2 frames)
for 1probe, conds are multiples by 10 (5 active pixels for 2 frames)
for
'''
weighted_df = delta_thr_df*40
print(f'weighted_df (all * 40):\n{weighted_df}\n')

# change concurrent
# print(f'weighted_df:\n{weighted_df}\n')
if 'Concurrent' in weighted_df.columns.tolist():
    conc = weighted_df['Concurrent']/2
    # print(f'conc:\n{conc}\n')
    weighted_df['Concurrent'] = conc
    print(f'weighted_df (Conc sorted):\n{weighted_df}\n')

    # # convert oneProbe/Concurrent back
    weighted_df["Concurrent"][one_probe_name] = weighted_df["Concurrent"][one_probe_name]*2

# if there is a 1probe cond
if -18 not in index_col:
    # divide 1probe by 4
    oneProbe = weighted_df.loc[one_probe_name]
    oneProbe = oneProbe/4
    weighted_df.loc[one_probe_name] = oneProbe
    print(f'weighted_df (conc & 1pr sorted):\n{weighted_df}\n')

# line plot for weighted delta ISIs
weighted_isi_df = weighted_df.copy()
weighted_isi_df.index = sep_range
print(f'weighted_isi_df:\n{weighted_isi_df}\n')
fig3_title = f'{exp_dir}: weighted ∆thr for ISIs'
fig3_savename = 'weighted_delta_isi.png'
save_as = os.path.join(root_path, exp_dir, fig3_savename)
simple_line_plot(indexed_df=weighted_isi_df, fig_title=fig3_title,
                 legend_title='ISI',
                 x_tick_vals=sep_range, x_tick_labels=sep_labels,
                 x_axis='Separation (diagonal pixels)',
                 y_axis='∆ Threshold *  n_pixels * n_frames',
                 log_y=log_y_axis,
                 save_as=save_as
                 )
plt.show()


# only make these plots if there are more than 2 columns
if 'thr' in ave_thr_df.columns.to_list():
    print('Not making transposed fig')
else:
    # line plot for weighted delta separations
    weighted_sep_df = weighted_df.copy().T
    weighted_sep_df.columns = sep_labels
    print(f'weighted_sep_df:\n{weighted_sep_df}\n')
    fig4_title = f'{exp_dir}: weighted ∆thr for Separations'
    fig4_savename = 'weighted_delta_sep.png'
    save_as = os.path.join(root_path, exp_dir, fig4_savename)

    simple_line_plot(indexed_df=weighted_sep_df, fig_title=fig4_title,
                     legend_title='separation',
                     x_axis='ISI (inter-stimulus interval)',
                     y_axis='∆ Threshold *  n_pixels * n_frames',
                     log_y=log_y_axis,
                     save_as=save_as
                     )
    plt.show()
