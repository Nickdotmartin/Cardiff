import os
import pandas as pd

from psychology_stats import compare_groups


'''
This script is used to compare the results of two different days of the same experiment.

It will run the correct test (Student, Welsh or Mann_whitney U) on groups of data.

I will need to correct for multiple comparisons.

'''

day1_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\rad_flow_Sept23\rad_flow_7_spokes\OLED_circles_rings_quartSpd\MASTER_all_Prelim.csv"
day2_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\rad_flow_Sept23\rad_flow_7_spokes\OLED_circles_rings_quartSpd_v2\MASTER_all_Prelim.csv"

day1_df = pd.read_csv(day1_path)
day2_df = pd.read_csv(day2_path)
print(f"Session 1: {day1_df.shape}\n{day1_df.head()}")
print(f"Session 2: {day2_df.shape}\n{day2_df.head()}")

counter = 1

for sep_cond in day1_df['neg_sep'].unique():
    cond_type_label = 'congruent'
    if sep_cond < 0:
        cond_type_label = 'incongruent'

    sep = abs(sep_cond)

    # print(f"Separation: {sep}, Condition type: {cond_type_label}")

    day1_sep_cond_df = day1_df[day1_df['neg_sep'] == sep_cond]
    day2_sep_cond_df = day2_df[day2_df['neg_sep'] == sep_cond]

    for prelim in day1_sep_cond_df['prelim'].unique():
        print(f"\n\nSeparation: {sep}, Condition type: {cond_type_label}, prelim: {prelim}")

        day1_sep_cond_prelim_df = day1_sep_cond_df[day1_sep_cond_df['prelim'] == prelim]
        day2_sep_cond_prelim_df = day2_sep_cond_df[day2_sep_cond_df['prelim'] == prelim]

        # print(f"day1_sep_cond_prelim_df: {day1_sep_cond_prelim_df.shape}\n{day1_sep_cond_prelim_df.head()}")

        day1_sep_cond_prelim_thr_array = day1_sep_cond_prelim_df['ISI_3'].values
        day2_sep_cond_prelim_thr_array = day2_sep_cond_prelim_df['ISI_3'].values

        # print(f"day1_sep_cond_prelim_thr_array: {day1_sep_cond_prelim_thr_array.shape}\n{day1_sep_cond_prelim_thr_array}")
        # print(f"day2_sep_cond_prelim_thr_array: {day2_sep_cond_prelim_thr_array.shape}\n{day2_sep_cond_prelim_thr_array}")


        # compare groups
        results_dict = compare_groups(day1_sep_cond_prelim_thr_array, day2_sep_cond_prelim_thr_array)
        print(f"{counter}. results_dict:\n{results_dict}")

        counter += 1

        corrected_sig = 0.05 / 12
        print(f"corrected_sig: {corrected_sig}")

        if results_dict['p'] < corrected_sig:
            print(f"%%%% Significant difference {results_dict['p']} < {corrected_sig} %%%%")



