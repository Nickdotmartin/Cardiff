import os
import pandas as pd
import numpy as np
from operator import itemgetter
from psignifit_tools import get_psignifit_threshold_df
from python_tools import switch_path
from rad_flow_psignifit_analysis import a_data_extraction, get_sorted_neg_sep_indices, sort_with_neg_sep_indices
from rad_flow_psignifit_analysis import b3_plot_staircase, c_plots
from rad_flow_psignifit_analysis import d_average_participant, make_average_plots, e_average_exp_data
from rad_flow_psignifit_analysis import compare_prelim_plots
from exp1a_psignifit_analysis import plt_heatmap_row_col

# # loop through run folders with first 5 scripts (a, b1, b2, b3, c)
# # then run script d to get master lists and averages
# exp_path = r'C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\rad_flow_UNCALIBRATED_MON\radial_flow_exp'
# participant_list = ['Kim', 'Nick', 'Simon']  # , 'Nick_half_speed']

# exp_path = r'C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\rad_flow_UNCALIBRATED_MON\rad_flow_half'
# participant_list = ['Nick_half', 'Simon_half']

# # # todo: why does a_extract data work for my data but not Simon's???
# exp_path = r'C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\rad_flow_UNCALIBRATED_MON\rad_flow_2'
# # participant_list = ['Simon', 'Nick']  # , 'Simon']  # , 'Nick_half_speed']
# participant_list = ['Nick']  # , 'Simon']  # , 'Nick_half_speed']


# exp_path = r'C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\rad_flow_UNCALIBRATED_MON\rad_flow_2_350'
# participant_list = ['Simon', 'Nick_350']

# exp_path = r'C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\rad_flow_UNCALIBRATED_MON\rad_flow_2_half'
# participant_list = ['Nick_half_speed', 'Simon_half']  # , 'Nick_half_speed']

# exp_path = r'C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\rad_flow_UNCALIBRATED_MON\rad_flow_23_OLED'
# participant_list = ['Nick_70_OLED_2.13A', 'Nick_350_OLED_2.13A', 'Simon_OLED_2.13A_black']

# exp_path = r'C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\rad_flow_UNCALIBRATED_MON\rad_flow_23'
# participant_list = ['Nick_240Hz_uncal_bg200'] #  'Nick_240Hz_07062023_bg70', 'Nick_OLED_02062023_bg350', 'Nick_240Hz_02062023_bg350', Nick_240Hz_end_june23_bg350, Nick_240_uncal_bg200]


exp_path = r'C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\rad_flow_UNCALIBRATED_MON\rad_flow_martin\rad_flow_6_rings'
participant_list = [
    # 'Nick_match_rf2_16082023',
#                    'Nick_third_ring_spd_16082023',
    #                 'Nick_half_ring_spd_16082023',
    #                 'Nick_quarter_ring_spd_16082023',
    #                 'Nick_orig_dots_17082023',
    # 'Nick_deep_sized_dots_17082023',
    # 'Nick_actual_new_dots_17082023',
    #                 'Nick_act_new_dots_thrd_spd_17082023'
]
exp_path = r'C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\rad_flow_UNCALIBRATED_MON\rad_flow_martin\rad_flow_6_rings_OLED'
participant_list = ['Nick_OLED_dots_normSpd_22082023']

# todo: is this the right script or should I use rad_flow_convert_and_analyse?  For now, just use this for files in UNCALIBRATED dir.

# todo: add code to get separation (plus probes, e.g., sep + 2)
'''
OLD CODE JUST FOR SEP, not sep and probes
pixel_mm_deg_dict = get_pixel_mm_deg_values(monitor_name=monitor_name)
if verbose:
    print(f"diagonal pixel size: {pixel_mm_deg_dict['diag_mm']} mm, or {pixel_mm_deg_dict['diag_deg']} dva")
sep_deg = sep * pixel_mm_deg_dict['diag_deg']
I might want to change this to manually enter view_dist...
'''

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

    print(f"\n\n{p_idx}. participant_name: {participant_name}")

    root_path = os.path.join(exp_path, participant_name)

    p_master_all_dfs_list = []
    p_master_ave_dfs_list = []
    p_master_err_dfs_list = []


    '''check for background_type in folder name, if 'flow_dots', 'flow_rings', 'no_bg', loop through those, else continue'''
    # # search to automatically get run_folder_names
    dir_list = os.listdir(root_path)
    bg_dir_list = []
    for bg_type in ['flow_dots', 'flow_rings', 'no_bg']:
        if bg_type in dir_list:
            bg_dir_list.append(bg_type)
    if len(bg_dir_list) == 0:
        bg_dir_list.append('No_background_type_found')
    # print(f'bg_dir_list: {bg_dir_list}')

    for bg_type in bg_dir_list:
        print(f"\nbg_type: {bg_type}")
        if bg_type != 'No_background_type_found':
            root_path = os.path.join(exp_path, participant_name, bg_type)

            '''Check for prelim flow dur folders, if they exist, loop through them, else continue'''
            dir_list = os.listdir(root_path)
            prelim_flow_dur_list = []

            # look for dirs beginning with 'bg' followed by ints
            for dir_name in dir_list:
                if dir_name[:2] == 'bg':
                    try:
                        int(dir_name[2:])
                        prelim_flow_dur_list.append(dir_name)
                    except ValueError:
                        pass
            if len(prelim_flow_dur_list) == 0:
                prelim_flow_dur_list.append('No_prelim_flow_dur_found')
            # print(f'prelim_flow_dur_list: {prelim_flow_dur_list}')

            for prelim_flow_dur in prelim_flow_dur_list:
                print(f"\nprelim_flow_dur: {prelim_flow_dur}")
                if prelim_flow_dur != 'No_prelim_flow_dur_found':
                    root_path = os.path.join(exp_path, participant_name, bg_type, prelim_flow_dur)


                    # # search to automatically get run_folder_names
                    dir_list = os.listdir(root_path)
                    run_folder_names = []
                    for i in range(n_runs):  # numbers 0 to 11
                        check_dir = f'{participant_name}_{i + p_idx_plus}'  # numbers 1 to 12
                        # print(check_dir)
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

                        a_data_extraction(p_name=p_name, run_dir=save_path, isi_list=run_isi_list, verbose=verbose)

                        run_data_path = f'{save_path}{os.sep}RUNDATA-sorted.xlsx'
                        run_data_df = pd.read_excel(run_data_path, engine='openpyxl')
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

                        # if prelim_ms isn't in the df, add it
                        if 'prelim_ms' not in list(run_data_df.columns):
                            prelim_int = int(prelim_flow_dur[2:])
                            run_data_df.insert(8, 'prelim_ms', prelim_int)
                            print('\nadded prelim_ms col')
                            print(run_data_df['prelim_ms'].to_list())

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
                            # stair_name = stair_df['stair_name'].unique().tolist()
                            separation = stair_df['separation'].unique().tolist()
                            neg_sep = stair_df['neg_sep'].unique().tolist()
                            congruent = stair_df['congruent'].unique().tolist()
                            # if len(stair_name) > 1:
                            #     raise ValueError(f"More than one unique stair name: {stair_name}")
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
                        print(f"neg_sep_vals_list: {neg_sep_vals_list}")
                        # print(f"ISI_vals_list: {ISI_vals_list}")
                        print(f"cong_vals_list: {cong_vals_list}")

                        # sort lists so that neg_sep_vals is in order [18, -18, 6, -6,...1, -1, 0, -.1]
                        print(f"\nneg_sep_vals_list: {neg_sep_vals_list}")
                        sorted_neg_sep_indices = get_sorted_neg_sep_indices(neg_sep_vals_list)

                        # sort stair_list, sep_vals_list, neg_sep_vals_list and cong_vals_list using sorted_neg_sep_indices
                        stair_list_sorted = sort_with_neg_sep_indices(stair_list, sorted_neg_sep_indices)
                        print(f"stair_list_sorted: {stair_list_sorted}")

                        sep_vals_list_sorted = sort_with_neg_sep_indices(sep_vals_list, sorted_neg_sep_indices)
                        print(f"stair_list_sorted: {stair_list_sorted}")

                        neg_sep_vals_list_sorted = sort_with_neg_sep_indices(neg_sep_vals_list, sorted_neg_sep_indices)
                        print(f"stair_list_sorted: {stair_list_sorted}")

                        cong_vals_list_sorted = sort_with_neg_sep_indices(cong_vals_list, sorted_neg_sep_indices)
                        print(f"stair_list_sorted: {stair_list_sorted}")

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
                                                            thr_col='probeLum',
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
# #         b3_plot_staircase(run_data_path, thr_col='probeLum', show_plots=show_plots, verbose=verbose)
# #
# #         '''c I don't actually need any of these, instead sort get psignifit thr ands make plots from those.'''
# #         c_plots(save_path=save_path, thr_col='probeLum', isi_name_list=run_isi_names_list, show_plots=show_plots, verbose=verbose)


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
                                   stair_names_col='neg_sep',
                                   cond_type_order=[1, -1],
                                   pos_neg_labels=['Congruent', 'Incongruent'],
                                   n_trimmed=trim_n,
                                   ave_over_n=len(run_folder_names),
                                   exp_ave=participant_name,
                                   show_plots=True, verbose=True)

                # add columns (background, prelim_ms) to all_df (and ave_df and err_df if needed)
                all_df = pd.read_csv(all_df_path)
                if 'background' not in all_df.columns.tolist():
                    all_df.insert(0, 'background', bg_type)
                if 'prelim_ms' not in all_df.columns.tolist():
                    all_df.insert(1, 'prelim_ms', prelim_flow_dur)
                p_master_all_dfs_list.append(all_df)

                ave_df = pd.read_csv(p_ave_path)
                if 'background' not in ave_df.columns.tolist():
                    ave_df.insert(0, 'background', bg_type)
                if 'prelim_ms' not in ave_df.columns.tolist():
                    ave_df.insert(1, 'prelim_ms', prelim_flow_dur)
                p_master_ave_dfs_list.append(ave_df)

                err_df = pd.read_csv(err_path)
                if 'background' not in err_df.columns.tolist():
                    err_df.insert(0, 'background', bg_type)
                if 'prelim_ms' not in err_df.columns.tolist():
                    err_df.insert(1
                                  , 'prelim_ms', prelim_flow_dur)
                p_master_err_dfs_list.append(err_df)


    # make master list for each participant with their average threshold for each background type and prelim flow dur
    p_compare_prelim_dir = os.path.join(exp_path, participant_name, 'compare_prelims')
    if not os.path.exists(p_compare_prelim_dir):
        os.mkdir(p_compare_prelim_dir)

    p_master_all_df = pd.concat(p_master_all_dfs_list)
    p_master_all_name = os.path.join(p_compare_prelim_dir, f'{participant_name}_ALLbg_thresholds.csv')
    if trim_n is not None:
        p_master_all_name = os.path.join(p_compare_prelim_dir, f'{participant_name}_TM{trim_n}_ALLbg_thresholds.csv')
    p_master_all_df.to_csv(p_master_all_name, index=False)

    # p_root_path = os.path.join(exp_path, participant_name)
    p_master_ave_df = pd.concat(p_master_ave_dfs_list)
    p_master_ave_name = os.path.join(p_compare_prelim_dir, f'{participant_name}_ALLbg_ave_thresh.csv')
    if trim_n is not None:
        p_master_ave_name = os.path.join(p_compare_prelim_dir, f'{participant_name}_TM{trim_n}_ALLbg_ave_thresh.csv')
    p_master_ave_df.to_csv(p_master_ave_name, index=False)

    p_master_err_df = pd.concat(p_master_err_dfs_list)
    p_master_err_name = os.path.join(p_compare_prelim_dir, f'{participant_name}_ALLbg_thr_error_SE.csv')
    if trim_n is not None:
        p_master_err_name = os.path.join(p_compare_prelim_dir, f'{participant_name}_TM{trim_n}_ALLbg_thr_error_SE.csv')
    p_master_err_df.to_csv(p_master_err_name, index=False)

    # make prelim plots for this participant
    compare_prelim_plots(participant_name, exp_path)




#
# print(f'exp_path: {exp_path}')
# # participant_list = ['Nick_350', 'Simon']
# print('\nget exp_average_data')
# # todo: check trim_n is correct
# # trim_n = 2
# # todo: sort script to automatically use trim=2 if its there, and not just use untrimmed#
# # todo: make sure ISI cols are in the correct order
# e_average_exp_data(exp_path=exp_path, p_names_list=participant_list,
#                    error_type='SE',
#                    # n_trimmed=trim_n,
#                    verbose=True)
#
#
# all_df_path = os.path.join(exp_path, "MASTER_exp_thr.csv")
# exp_ave_path = os.path.join(exp_path, "MASTER_exp_ave_thr.csv")
# err_path = os.path.join(exp_path, "MASTER_ave_thr_error_SE.csv")
#
# all_df = pd.read_csv(all_df_path)
# if any("Unnamed" in i for i in list(all_df.columns)):
#     unnamed_col = [i for i in list(all_df.columns) if "Unnamed" in i][0]
#     all_df.drop(unnamed_col, axis=1, inplace=True)
# print(f"all_df:\n{all_df}")
#
# all_df_basic_cols = ['participant', 'stair_names', 'congruent', 'separation']
#
# # isi_names_list = list(all_df.columns[len(all_df_basic_cols):])
# isi_names_list = [i for i in list(all_df.columns) if 'isi' in i.lower()]
#
# isi_vals_list = [int(i[4:]) for i in isi_names_list]
#
# # sort isi_names_list by sorted(isi_vals_list) order
# isi_vals_array = np.array(isi_vals_list)
# print(f"\nisi_vals_array: {isi_vals_array}")
# sort_index = np.argsort(isi_vals_array)
# print(f"sort_index: {sort_index}")
#
# isi_vals_list = [isi_vals_list[i] for i in sort_index]
# print(f"isi_vals_list: {isi_vals_list}")
#
# isi_names_list = [isi_names_list[i] for i in sort_index]
# print(f"isi_names_list: {isi_names_list}")
#
# all_col_names = all_df_basic_cols + isi_names_list
# print(f"all_col_names: {all_col_names}")
#
# all_df = all_df[all_col_names]
# print(f"all_df:\n{all_df}")
# all_df.to_csv(all_df_path, index=False)
#
#
#
# ave_df = pd.read_csv(exp_ave_path)
# if any("Unnamed" in i for i in list(ave_df.columns)):
#     unnamed_col = [i for i in list(ave_df.columns) if "Unnamed" in i][0]
#     ave_df.drop(unnamed_col, axis=1, inplace=True)
# print(f"ave_df:\n{ave_df}")
# ave_df_basic_cols = ['stair_names']
# ave_col_names = ave_df_basic_cols + isi_names_list
# print(f"ave_col_names: {ave_col_names}")
# ave_df = ave_df[ave_col_names]
# print(f"ave_df:\n{ave_df}")
# ave_df.to_csv(exp_ave_path, index=False)
#
# err_df = pd.read_csv(err_path)
# if any("Unnamed" in i for i in list(err_df.columns)):
#     unnamed_col = [i for i in list(err_df.columns) if "Unnamed" in i][0]
#     err_df.drop(unnamed_col, axis=1, inplace=True)
# print(f"err_df:\n{err_df}")
#
# # replace any NaNs with 0s
# err_df.fillna(0, inplace=True)
#
# err_df = err_df[ave_col_names]
# print(f"err_df:\n{err_df}")
# err_df.to_csv(err_path, index=False)
#
#
# make_average_plots(all_df_path=all_df_path,
#                    ave_df_path=exp_ave_path,
#                    error_bars_path=err_path,
#                    thr_col='probeLum',
#                    stair_names_col='stair_names',
#                    cond_type_col='congruent',
#                    cond_type_order=[1, -1],
#                    n_trimmed=trim_n,
#                    ave_over_n=len(participant_list),
#                    exp_ave=True,
#                    isi_name_list=isi_names_list,
#                    isi_vals_list=isi_vals_list,
#                    show_plots=True, verbose=True)

print('\nrad_flow_analysis_pipe finished')
