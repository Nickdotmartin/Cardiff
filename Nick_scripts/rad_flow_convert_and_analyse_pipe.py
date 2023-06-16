import os
import pandas as pd
import numpy as np
from operator import itemgetter
from psignifit_tools import get_psignifit_threshold_df
from python_tools import switch_path
from rad_flow_psignifit_analysis import a_data_extraction, get_sorted_neg_sep_indices, sort_with_neg_sep_indices
from rad_flow_psignifit_analysis import b3_plot_staircase, c_plots
from rad_flow_psignifit_analysis import d_average_participant, make_average_plots, e_average_exp_data, rad_flow_mon_conversion
from exp1a_psignifit_analysis import plt_heatmap_row_col

# # loop through run folders with first 5 scripts (a, b1, b2, b3, c)
# # then run script d to get master lists and averages
# exp_path = r'C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\radial_flow_exp'
# participant_list = ['Kim', 'Nick', 'Simon']  # , 'Nick_half_speed']

# exp_path = r'C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\rad_flow_half'
# participant_list = ['Nick_half', 'Simon_half']

# # todo: why does psignifit data work for my data but not Simon's???  I've editted the psignifit (around line 390)code to work for Simon's data
# exp_path = r'C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\rad_flow_2'
# participant_list = ['Simon', 'Nick']  # , 'Nick_half_speed']

# exp_path = r'C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\rad_flow_2_350'
# participant_list = ['Simon', 'Nick_350']

# exp_path = r'C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\rad_flow_2_half'
# participant_list = ['Nick_half_speed', 'Simon_half']  # , 'Nick_half_speed']

# exp_path = r'C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\rad_flow_23'
# participant_list = ['Nick_240Hz_07062023_bg70', 'Nick_OLED_02062023_bg350', 'Nick_240Hz_02062023_bg350']

# exp_path = r'C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\rad_flow_23_OLED'
# participant_list = ['Nick_70_OLED_2.13A', 'Nick_350_OLED_2.13A', 'Simon_OLED_2.13A_black']


# list of exps that used the uncalibrated monitor and need changing
used_uncalibrated_mon = ['radial_flow_exp', 'rad_flow_half', 'rad_flow_2', 'rad_flow_2_350', 'rad_flow_2_half', 'rad_flow_23']


exp_path = os.path.normpath(exp_path)
convert_path1 = os.path.normpath(exp_path)
convert_path1 = switch_path(convert_path1, 'windows_oneDrive')
exp_path = convert_path1

verbose = True
show_plots = True

n_runs = 12
# if the first folder to analyse is 1, p_idx_plus = 1.  If the forst folder is 5, use 5 etc.
p_idx_plus = 1
trim_list = []

for p_idx, participant_name in enumerate(participant_list):

    root_path = os.path.join(exp_path, participant_name)

    # # manually get run_folder_names with n_runs
    # run_folder_names = [f'{participant_name}_{i+p_idx_plus}' for i in list(range(n_runs))]

    # # search to automatically get run_folder_names
    dir_list = os.listdir(root_path)
    run_folder_names = []
    for i in range(n_runs):  # numbers 0 to 11
        check_dir = f'{participant_name}_{i + p_idx_plus}'  # numbers 1 to 12
        print(check_dir)
        if check_dir in dir_list:
            run_folder_names.append(check_dir)

    print(f'run_folder_names: {run_folder_names}')

    trim_n = None
    if len(run_folder_names) == 12:
        trim_n = 2
    elif len(run_folder_names) > 12:
        # trim_n = 2
        if len(run_folder_names) % 2 == 0:  # if even
            trim_n = int((len(run_folder_names)-12)/2)
        else:
            raise ValueError(f"for this exp you have {len(run_folder_names)} runs, set rules for trimming.")
    trim_list.append(trim_n)

    for run_idx, run_dir in enumerate(run_folder_names):

        # add run number , e.g., add five to access Nick_5 on the zeroth iteration
        r_idx_plus = run_idx + p_idx_plus

        print(f'\nrun_idx {run_idx + 1}: running analysis for '
              f'{participant_name}, {run_dir}, {participant_name}_{r_idx_plus}')
        save_path = f'{root_path}{os.sep}{run_dir}'

        # # search to automatically get updated isi_list
        dir_list = os.listdir(save_path)
        run_isi_list = []
        # for isi in isi_vals_list:
        for isi in list(range(-2, 18, 1)):
            check_dir = f'ISI_{isi}_probeDur2'
            # check_dir = f'ISI_{isi}'
            if check_dir in dir_list:
                run_isi_list.append(isi)
            check_dir = f'ISI_{isi}'
            if check_dir in dir_list:
                run_isi_list.append(isi)
        run_isi_names_list = [f'ISI_{i}' for i in run_isi_list]

        print(f'run_isi_list: {run_isi_list}')

        # don't delete this (p_name = participant_name),
        # needed to ensure names go name1, name2, name3 not name1, name12, name123
        p_name = participant_name

        '''a'''
        p_name = f'{participant_name}_{r_idx_plus}'

        # todo: put a_data_extraction back in
        # a_data_extraction(p_name=p_name, run_dir=save_path, isi_list=run_isi_list, verbose=verbose)

        run_data_path = f'{save_path}{os.sep}ALL_ISIs_sorted.xlsx'
        run_data_df = pd.read_excel(run_data_path, engine='openpyxl')
        print(f"run_data_df: {run_data_df.columns.to_list()}\n{run_data_df}")

        # code to make sure luminance is int, not float.
        if run_data_df['probeColor255'].dtypes == 'int64':
            lum_col = 'probeLum'
            print(f"probeColor255 is {run_data_df['probeColor255'].dtypes}, lum_col is {lum_col}")
        else:
            '''add newLum column
                    in old version, the experiment script varies probeLum and converts to float(RGB255) values for screen.
                    However, monitor can only use int(RGB255).
                    This function will will round RGB255 values to int(RGB255), then convert to NEW_probeLum
                    LumColor255Factor = 2.395387069
                    1. get probeColor255 column.
                    2. convert to int(RGB255) and convert to new_Lum with int(RGB255)/LumColor255Factor
                    3. add to run_data_df'''
            lum_col = 'newLum'
            print(f"probeColor255 is {run_data_df['probeColor255'].dtypes}, lum_col is {lum_col}")
            if 'newLum' not in run_data_df.columns.to_list():
                LumColor255Factor = 2.395387069
                rgb255_col = run_data_df['probeColor255'].to_list()
                newLum = [int(i) / LumColor255Factor for i in rgb255_col]
                run_data_df.insert(11, 'newLum', newLum)
                print(f"added newLum column\n"
                      f"run_data_df: {run_data_df.columns.to_list()}")



        # add converted_lum as a new column using the function rad_flow_mon_conversion
        _, exp_name = os.path.split(exp_path)
        print(f'exp_name: {exp_name}')
        print(f'used_uncalibrated_mon: {used_uncalibrated_mon}')
        if exp_name in used_uncalibrated_mon:
            print('This experiment used the uncalibrated monitor - converting luminance values to equivalent from asus_cal as lum_col: converted_lum')
            if 'converted_lum' not in list(run_data_df.columns):
                '''Thi8s converts rgb255 values from the uncalibrated monitor to their equivalents on the
                calibrated monitor (ASUS_CAL) using the luminance lookup tables measured with spyder.
                
                The max lum with these measurements reflects the max lum, as measured on on 12/06/2023, (e.g., around 150), 
                rather than the max lum given in the script (which was set by Martin as 106)'''
                # run_data_df['converted_lum'] = run_data_df.apply(rad_flow_mon_conversion, axis=1)

                probeLum_list = run_data_df[lum_col].to_list()
                print(f'\nprobeLum_list: {probeLum_list}')

                conv_lum_list = []
                for probeLum in probeLum_list:
                    conv_lum_list.append(rad_flow_mon_conversion(probeLum, verbose=True))
                print(f'\nconv_lum_list: {conv_lum_list}')
                run_data_df.insert(11, 'converted_lum', conv_lum_list)
                lum_col = 'converted_lum'

                print('\nadded converted_lum col')
                print(f"run_data_df: {run_data_df.columns.to_list()}\n{run_data_df}")



        # add neg sep column to make batman plots
        if 'neg_sep' not in list(run_data_df.columns):
            def make_neg_sep(df):
                if (df.congruent == -1) and (df.separation == 0.0):
                    return -.1
                elif df.congruent == -1:
                    return 0 - df.separation
                else:
                    return df.separation

            run_data_df.insert(7, 'neg_sep', run_data_df.apply(make_neg_sep, axis=1))
            print('\nadded neg_sep col')
            print(run_data_df['neg_sep'].to_list())

        # remove unnamed columns
        substring = 'Unnamed: '
        unnamed_cols = [i for i in run_data_df.columns.to_list() if substring in i]
        print(f"unnamed_cols: {unnamed_cols}")
        for col_name in unnamed_cols:
            run_data_df.drop(col_name, axis=1, inplace=True)

        # save run_data_df
        # run_data_path = os.path.join(save_path, 'RUNDATA-sorted.xlsx')  # delete this, using ALL_ISIs_sorted.xlsx
        run_data_df.to_excel(run_data_path, index=False)
        print(f"run_data_df:\n{run_data_df}")


        '''
        Data should be analysed in a particular order to ensure correct order on plots etc.
        The order will be (neg_sep) [18, -18, 6, -6, 3, -3, 2, -2, 1, -1, 0, -.1]
        first get values in stair name order, then get indices to put neg_sep in the correct order.
        Use these indices to sort the lists to feed into psignifit.
        '''

        # get a list of all unique values in the 'stair' column, in the order they appear
        stair_list = run_data_df['stair'].unique().tolist()

        # check that there is just one unique value associated with each stair for separation, neg_sep, ISI and congruent.
        # append the unique values to sep_vals_list, neg_sep_vals_list, and cong_vals_list.
        sep_vals_list = []
        neg_sep_vals_list = []
        cong_vals_list = []

        for stair in stair_list:
            stair_df = run_data_df[run_data_df['stair'] == stair]
            separation = stair_df['separation'].unique().tolist()
            neg_sep = stair_df['neg_sep'].unique().tolist()
            congruent = stair_df['congruent'].unique().tolist()
            if len(separation) > 1:
                raise ValueError(f"More than one unique separation: {separation}")
            if len(neg_sep) > 1:
                raise ValueError(f"More than one unique neg_sep: {neg_sep}")
            if len(congruent) > 1:
                raise ValueError(f"More than one unique congruent: {congruent}")

            sep_vals_list.append(separation[0])
            neg_sep_vals_list.append(neg_sep[0])
            cong_vals_list.append(congruent[0])

        print(f"\nsep_vals_list: {sep_vals_list}")
        print(f"cong_vals_list: {cong_vals_list}")

        # sort lists so that neg_sep_vals is in order [18, -18, 6, -6,...1, -1, 0, -.1]
        print(f"\nneg_sep_vals_list: {neg_sep_vals_list}")
        sorted_neg_sep_indices = get_sorted_neg_sep_indices(neg_sep_vals_list)

        # sort stair_list, sep_vals_list, neg_sep_vals_list and cong_vals_list using sorted_neg_sep_indices
        stair_list_sorted = sort_with_neg_sep_indices(stair_list, sorted_neg_sep_indices)
        print(f"stair_list_sorted: {stair_list_sorted}")

        sep_vals_list_sorted = sort_with_neg_sep_indices(sep_vals_list, sorted_neg_sep_indices)
        print(f"sep_vals_list_sorted: {sep_vals_list_sorted}")

        neg_sep_vals_list_sorted = sort_with_neg_sep_indices(neg_sep_vals_list, sorted_neg_sep_indices)
        print(f"neg_sep_vals_list_sorted: {neg_sep_vals_list_sorted}")

        cong_vals_list_sorted = sort_with_neg_sep_indices(cong_vals_list, sorted_neg_sep_indices)
        print(f"cong_vals_list_sorted: {cong_vals_list_sorted}")

        '''get psignifit thresholds df'''
        cols_to_add_dict = {'stair_names': neg_sep_vals_list_sorted,
                            'congruent': cong_vals_list_sorted,
                            'separation': sep_vals_list_sorted}
        print('\ncols_to_add_dict:')
        for k, v in cols_to_add_dict.items():
            print(f'{k}: {v}')

        thr_df = get_psignifit_threshold_df(root_path=root_path,
                                            p_run_name=run_dir,
                                            csv_name=run_data_df,
                                            n_bins=9, q_bins=True,
                                            sep_col='neg_sep',
                                            thr_col=lum_col,
                                            isi_list=run_isi_list,
                                            sep_list=neg_sep_vals_list_sorted,
                                            conf_int=True,
                                            thr_type='Bayes',
                                            plot_both_curves=False,
                                            cols_to_add_dict=cols_to_add_dict,
                                            show_plots=False,
                                            verbose=verbose)
        print(f'thr_df:\n{thr_df}')
#
#
# #         '''b3'''
# #         b3_plot_staircase(run_data_path, thr_col=lum_col, show_plots=show_plots, verbose=verbose)
# #
# #         '''c I don't actually need any of these, instead sort get psignifit thr ands make plots from those.'''
# #         c_plots(save_path=save_path, thr_col=lum_col, isi_name_list=run_isi_names_list, show_plots=show_plots, verbose=verbose)


    '''d participant averages'''
    trim_n = None
    if len(run_folder_names) == 12:
        trim_n = 2
    print(f'\ntrim_n: {trim_n}')

    d_average_participant(root_path=root_path, run_dir_names_list=run_folder_names,
                          trim_n=trim_n, error_type='SE', verbose=verbose)


    # making average plot
    all_df_path = os.path.join(root_path, f'MASTER_TM{trim_n}_thresholds.csv')
    p_ave_path = os.path.join(root_path, f'MASTER_ave_TM{trim_n}_thresh.csv')
    err_path = os.path.join(root_path, f'MASTER_ave_TM{trim_n}_thr_error_SE.csv')
    if trim_n is None:
        all_df_path = os.path.join(root_path, f'MASTER_psignifit_thresholds.csv')
        p_ave_path = os.path.join(root_path, 'MASTER_ave_thresh.csv')
        err_path = os.path.join(root_path, 'MASTER_ave_thr_error_SE.csv')
    exp_ave = False


    make_average_plots(all_df_path=all_df_path,
                       ave_df_path=p_ave_path,
                       error_bars_path=err_path,
                       thr_col='probeLum',
                       stair_names_col='stair_names',
                       n_trimmed=trim_n,
                       ave_over_n=len(run_folder_names),
                       exp_ave=participant_name,
                       show_plots=True, verbose=True)



# print(f'exp_path: {exp_path}')
# # participant_list = ['Nick_350', 'Simon']
# # print('\nget exp_average_data')
# todo: check trim_n is correct
# trim_n = 2
# # todo: sort script to automatically use trim=2 if its there, and not just use untrimmed#
# # todo: make sure ISI cols are in the correct order
e_average_exp_data(exp_path=exp_path, p_names_list=participant_list,
                   error_type='SE',
                   # n_trimmed=trim_n,
                   verbose=True)


all_df_path = os.path.join(exp_path, "MASTER_exp_thr.csv")
exp_ave_path = os.path.join(exp_path, "MASTER_exp_ave_thr.csv")
err_path = os.path.join(exp_path, "MASTER_ave_thr_error_SE.csv")

all_df = pd.read_csv(all_df_path)
if any("Unnamed" in i for i in list(all_df.columns)):
    unnamed_col = [i for i in list(all_df.columns) if "Unnamed" in i][0]
    all_df.drop(unnamed_col, axis=1, inplace=True)
print(f"all_df:\n{all_df}")

all_df_basic_cols = ['participant', 'stair_names', 'congruent', 'separation']

# isi_names_list = list(all_df.columns[len(all_df_basic_cols):])
isi_names_list = [i for i in list(all_df.columns) if 'isi' in i.lower()]

isi_vals_list = [int(i[4:]) for i in isi_names_list]

# sort isi_names_list by sorted(isi_vals_list) order
isi_vals_array = np.array(isi_vals_list)
print(f"\nisi_vals_array: {isi_vals_array}")
sort_index = np.argsort(isi_vals_array)
print(f"sort_index: {sort_index}")

isi_vals_list = [isi_vals_list[i] for i in sort_index]
print(f"isi_vals_list: {isi_vals_list}")

isi_names_list = [isi_names_list[i] for i in sort_index]
print(f"isi_names_list: {isi_names_list}")

all_col_names = all_df_basic_cols + isi_names_list
print(f"all_col_names: {all_col_names}")

all_df = all_df[all_col_names]
print(f"all_df:\n{all_df}")
all_df.to_csv(all_df_path, index=False)



ave_df = pd.read_csv(exp_ave_path)
if any("Unnamed" in i for i in list(ave_df.columns)):
    unnamed_col = [i for i in list(ave_df.columns) if "Unnamed" in i][0]
    ave_df.drop(unnamed_col, axis=1, inplace=True)
print(f"ave_df:\n{ave_df}")
ave_df_basic_cols = ['stair_names']
ave_col_names = ave_df_basic_cols + isi_names_list
print(f"ave_col_names: {ave_col_names}")
ave_df = ave_df[ave_col_names]
print(f"ave_df:\n{ave_df}")
ave_df.to_csv(exp_ave_path, index=False)

err_df = pd.read_csv(err_path)
if any("Unnamed" in i for i in list(err_df.columns)):
    unnamed_col = [i for i in list(err_df.columns) if "Unnamed" in i][0]
    err_df.drop(unnamed_col, axis=1, inplace=True)
print(f"err_df:\n{err_df}")

# replace any NaNs with 0s
err_df.fillna(0, inplace=True)

err_df = err_df[ave_col_names]
print(f"err_df:\n{err_df}")
err_df.to_csv(err_path, index=False)


make_average_plots(all_df_path=all_df_path,
                   ave_df_path=exp_ave_path,
                   error_bars_path=err_path,
                   thr_col='probeLum',
                   stair_names_col='stair_names',
                   cond_type_col='congruent',
                   cond_type_order=[1, -1],
                   n_trimmed=trim_n,
                   ave_over_n=len(participant_list),
                   exp_ave=True,
                   isi_name_list=isi_names_list,
                   isi_vals_list=isi_vals_list,
                   show_plots=True, verbose=True)

print('\nrad_flow_analysis_pipe finished')
