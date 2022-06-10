import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as img

'''
script to analyse the CIs in exp1
BASIC
1. get width of CI from CI values,
    get a list of all widths
    
2. Plot distribution and get hi, med and low examples (plots, copied to dirs)

DETAILED
1. loop though all
    participants in participant_list:
    runs in run list:
    read CI_df
    loop_through ISIs subtracting min from max to get CI_width
    append CI_width to CI_width_list
    make CI_width_df
    save as only file in CI_info dir
    
2. from MASTER_LONG_CI_df
    plot distribution
    plot distribution vs slope at thr?
    get 6 lowest, highest, 6 medium (based on distribution choose mode, median, mean

'''
#
#
# # # 1. loop though all participants in participant_list and runs in run list:
#
# exp_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\exp1a_data"
# exp_path = os.path.normpath(exp_path)
#
# # participant_list = os.listdir(exp_path)
# participant_list = ['aa', 'bb', 'cc', 'dd', 'ee']
# print(f'participant_list: {participant_list}')
#
# CI_width_list = []
#
# for participant in participant_list:
#     print(f'\nparticipant: {participant}')
#
#     participant_path = os.path.join(exp_path, participant)
#     run_list = os.listdir(participant_path)
#
#     print(f'run_list: {run_list}')
#
#     # sort run directories
#     # print(run_list[11])
#     # print(run_list[11][-1])
#     # print(int(run_list[0][0]))
#     # print(run_list[11][-1].isdigit())
#     # run_list = [i if i[-1].isdigit() for i in run_list]
#     run_list = [i for i in run_list if i[-1].isdigit()]
#     print(f'run_list: {run_list}')
#
#     for run_name in run_list:
#         print(f'\nrun_name: {run_name}')
#
#         run_path = os.path.join(participant_path, run_name)
#
#
#         CI_csv_path = os.path.join(run_path, 'psignifit_CI.csv')
#
#         #     read CI_df
#         CI_df = pd.read_csv(CI_csv_path)
#         print(f'CI_df:\n{CI_df}')
#
#         CI_headers = list(CI_df.columns)
#         print(f'CI_headers: {CI_headers}')
#
#         CI_cols = CI_headers[3:]
#         print(f'CI_cols: {CI_cols}')
#
#         #     loop_through ISIs subtracting min from max to get CI_width
#         for index, row in CI_df.iterrows():
#             print(f'{index}. row: {row}')
#
#             for lo_CI_col, hi_CI_col in zip(CI_cols[::2], CI_cols[1::2]):
#
#                 # do some sanity checks
#                 # print(lo_CI_col, hi_CI_col)
#                 lo_ISI_cond, lo_CI_pc = lo_CI_col[:-6], lo_CI_col[-2:]
#                 print(lo_CI_col, lo_ISI_cond, lo_CI_pc)
#                 hi_ISI_cond, hi_CI_pc = hi_CI_col[:-6], hi_CI_col[-2:]
#                 print(hi_CI_col, hi_ISI_cond, hi_CI_pc)
#
#                 if lo_ISI_cond != hi_ISI_cond:
#                     raise ValueError
#                 if lo_CI_pc != hi_CI_pc:
#                     raise ValueError
#
#                 # print(row[lo_CI_col], row[hi_CI_col])
#
#                 print(f"row['separation']: {row['separation']}, row['{lo_CI_col}']: {row[lo_CI_col]}, row['{hi_CI_col}']: {row[hi_CI_col]}, \n")
#
#                 # get lo_CI_val, hi_CI_val (and sep), check that CI_vals are not NaNs
#
#                 lo_CI_val, hi_CI_val = row[lo_CI_col], row[hi_CI_col]
#
#                 print(f'np.isnan([lo_CI_val, hi_CI_val]).any(): {np.isnan([lo_CI_val, hi_CI_val]).any()}')
#                 if not np.isnan([lo_CI_val, hi_CI_val]).any():
#
#                     # compute width
#                     CI_width = hi_CI_val - lo_CI_val
#
#                     # get separation
#                     sep = row['separation']
#                     stair = row['stair']
#
#                     #     append CI_width to CI_width_list
#
#                     CI_width_list.append([participant, lo_CI_pc, run_name, stair, sep, lo_ISI_cond, lo_CI_val, hi_CI_val, CI_width])
#
#
# #     make CI_width_df
# CI_width_df = pd.DataFrame(CI_width_list, columns=['participant', 'CI%', 'run_name', 'stair', 'separation', 'ISI', 'lo', 'hi', 'width'])
# print(f'CI_width_df:\n{CI_width_df}')
#
# CI_info_path = os.path.join(exp_path, 'CI_info')
# if not os.path.isdir(CI_info_path):
#     os.makedirs(CI_info_path)
#
# CI_width_df.to_csv(os.path.join(CI_info_path, 'long_CI_width.csv'), index=False)
#
# print(f'finished making {CI_info_path}/long_CI_width.csv')




# 2. from MASTER_LONG_CI_df
exp_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\exp1a_data"
exp_path = os.path.normpath(exp_path)
CI_width_df = pd.read_csv(os.path.join(exp_path, 'CI_info', 'long_CI_width.csv'))
print(f'CI_width_df:\n{CI_width_df}')

# #     plot distribution
# sns.histplot(CI_width_df, x="width")
# plt.title('Distribution of 95% CI widths')
# plt.savefig(os.path.join(exp_path, 'CI_info', 'CI_distribution.png'))
# plt.show()

#     get 6 lowest, highest, 6 medium (based on distribution choose mode, median, mean
sorted_df = CI_width_df.sort_values(by='width')
print(f'sorted_df:\n{sorted_df}')

rows, cols = sorted_df.shape

n_examples = 6

index_list = list(sorted_df.index.values)
print(index_list)

mid_point = int(rows/2)
print(f'mid_point: {mid_point}')

mid_val = mid_point-(int(n_examples/2))
print(f'mid_val: {mid_val}')

indices_for_examples = index_list[:n_examples] + index_list[mid_val: mid_val+n_examples] + index_list[-n_examples:]
print(f'indices_for_examples: {indices_for_examples}')

# examples_df = CI_width_df.iloc[indices_for_examples]
# print(f'examples_df:\n{examples_df}')

# make dirs for lo, median and high or rename file as low, med, hi example?

for counter, index in enumerate(indices_for_examples):

    if counter < n_examples:
        example_type = 'lo'
    elif n_examples < counter < n_examples*2:
        example_type = 'median'
    else:
        example_type = 'hi'

    row = CI_width_df.iloc[index]
    print(f'\n\ncounter: {counter}, index: {index}, type: {example_type}: width: {row["width"]}\nrow: {row}')

    isi_val = row['ISI'][4:]
    print(f'isi_val: {isi_val}')

    stair_val = int(row['stair'])

    image_name = f"{row['run_name']}_ISI{isi_val}_sep{stair_val}_stair{stair_val}_psig.png"
    image_path = os.path.join(exp_path, row['participant'], row['run_name'], image_name)
    image = img.imread(image_path)

    img_plot = plt.imshow(image)
    plt.axis('off')

    save_name = f'{example_type}_{image_name}'
    save_path = os.path.join(exp_path, 'CI_info', save_name)
    plt.savefig(save_path)

    # plt.show()




