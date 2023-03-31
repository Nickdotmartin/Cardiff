import os
import pandas as pd
from exp1a_psignifit_analysis import a_data_extraction, b3_plot_staircase, c_plots, \
    d_average_participant, e_average_exp_data, make_average_plots, make_diff_from_conc_df, lookup_p_name
from psignifit_tools import get_psignifit_threshold_df
from python_tools import which_path, running_on_laptop, switch_path
import matplotlib.pyplot as plt

# # loop through run folders with first 4 scripts (a, get_psignifit_threshold_df, b3, c)
# # then run script d to get master lists and averages
# old_exp_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/exp1a_data'
old_exp_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\exp1a_data"

# old_exp_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/Kim_split_runs'
# old_exp_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/exp1a_data_first_three_sessions'
# old_exp_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/exp1a_data_second_three_sessions'
convert_path1 = os.path.normpath(old_exp_path)
if running_on_laptop():
    convert_path1 = switch_path(convert_path1, 'mac_oneDrive')
exp_path = convert_path1


# exp_path = switch_path(old_exp_path, 'wind_oneDrive')
print(f"exp_path: {exp_path}")
participant_list = ['dd', 'aa', 'bb', 'ee', 'Nick', 'cc']
# participant_list = ['bb', 'cc', 'dd', 'ee']
# participant_list = ['cc']

isi_list = [-1, 0, 2, 4, 6, 9, 12, 24]

n_runs = 6

p_idx_plus = 1

# for p_idx, participant_name in enumerate(participant_list):
#     root_path = os.path.join(exp_path, participant_name)
#
#     run_folder_names = [f'{participant_name}_{i+1}' for i in list(range(n_runs))]
#     print(f'run_folder_names: {run_folder_names}')
#
#     group_list = [1, 2]
#
#     for run_idx, run_dir in enumerate(run_folder_names):
#
#         print(f'\nrunning analysis for {participant_name}, {run_dir}, {participant_name}{run_idx+1}\n')
#         save_path = f'{root_path}{os.sep}{run_dir}'
#
#         # don't delete this (participant_name = participant_name),
#         # needed to ensure names go name1, name2, name3 not name1, name12, name123
#         p_name = participant_name

        # # # '''a'''
        # p_name = f'{participant_name}_output'  # use this one
        #
        # run_data_df = a_data_extraction(p_name=p_name, run_dir=save_path, isi_list=isi_list, verbose=True)
        #
        # '''add newLum column
        # in old version, the experiment script varies probeLum and converts to float(RGB255) values for screen.
        # However, monitor can only use int(RGB255).
        # This function will will round RGB255 values to int(RGB255), then convert to NEW_probeLum
        # LumColor255Factor = 2.395387069
        # 1. get probeColor255 column.
        # 2. convert to int(RGB255) and convert to new_Lum with int(RGB255)/LumColor255Factor
        # 3. add to run_data_df'''
        # if 'newLum' not in run_data_df.columns.to_list():
        #     LumColor255Factor = 2.395387069
        #     rgb255_col = run_data_df['probeColor255'].to_list()
        #     newLum = [int(i) / LumColor255Factor for i in rgb255_col]
        #     run_data_df.insert(9, 'newLum', newLum)
        #     run_data_df.to_excel(os.path.join(save_path, 'RUNDATA-sorted.xlsx'), index=False)
        #     print(f"added newLum column\n"
        #           f"run_data_df: {run_data_df.columns.to_list()}")
        #
        #
        # run_data_path = os.path.join(save_path, 'RUNDATA-sorted.xlsx')
        #
        # run_data_df = pd.read_excel(run_data_path, engine='openpyxl',
        #                             usecols=['ISI',
        #                                      'stair',
        #                                      'separation', 'group',
        #                                      # 'probeLum',
        #                                      'newLum', 'trial_response'])
        # print(f"run_data_df:\n{run_data_df}")
        #
        #
        # stair_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        # cols_to_add_dict = {'group': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        #                     'separation': [18, 18, 6, 6, 3, 3, 2, 2, 1, 1, 0, 0, 20, 20]}
        #
        # '''get psignifit thresholds df - use stairs as sep levels rather than using groups'''
        # thr_df = get_psignifit_threshold_df(root_path=root_path,
        #                                     p_run_name=run_dir,
        #                                     csv_name=run_data_df,
        #                                     n_bins=9, q_bins=True,
        #                                     sep_col='stair',
        #                                     thr_col='newLum',
        #                                     isi_list=isi_list,
        #                                     sep_list=stair_list,
        #                                     conf_int=True,
        #                                     thr_type='Bayes',
        #                                     plot_both_curves=False,
        #                                     save_plots=True,
        #                                     cols_to_add_dict=cols_to_add_dict,
        #                                     verbose=True)
        # print(f'thr_df:\n{thr_df}')
        #
        # '''b3'''
        # run_data_path = os.path.join(save_path, 'RUNDATA-sorted.xlsx')
        # thr_path = os.path.join(save_path, 'psignifit_thresholds.csv')
        #
        # b3_plot_staircase(run_data_path, thr_col='newLum', show_plots=False)
        #
        # '''c'''
        # c_plots(save_path=save_path, thr_col='newLum', show_plots=True)


    # trim_n = None
    # if len(run_folder_names) == 6:
    #     trim_n = 2
    #
    # print(f"\n\ntrim_n: {trim_n}, \n\n")
    # #
    # # # if 'first_three_sessions' in old_exp_path:
    # # #     run_folder_names = run_folder_names[:3]
    # # # elif 'second_three_sessions' in old_exp_path:
    # # #     run_folder_names = run_folder_names[3:]
    # # # else:
    # # #     raise ValueError('what experiment am I analysing?')
    # #
    # #
    # '''d'''
    # # d_average_participant(root_path=root_path, run_dir_names_list=run_folder_names,
    # #                       trim_n=trim_n, error_type='SE')
    #
    # all_df_path = os.path.join(root_path, f'MASTER_TM{trim_n}_thresholds.csv')
    # p_ave_path = os.path.join(root_path, f'MASTER_ave_TM{trim_n}_thresh.csv')
    # err_path = os.path.join(root_path, f'MASTER_ave_TM{trim_n}_thr_error_SE.csv')
    # if trim_n is None:
    #     all_df_path = os.path.join(root_path, 'MASTER_psignifit_thresholds.csv')
    #     p_ave_path = os.path.join(root_path, 'MASTER_ave_thresh.csv')
    #     err_path = os.path.join(root_path, 'MASTER_ave_thr_error_SE.csv')
    #
    # ave_DfC_df, error_DfC_df = make_diff_from_conc_df(all_df_path, root_path, n_trimmed=trim_n)
    # # print(f"\nave_DfC_df:\n{ave_DfC_df}")
    # # print(f"error_DfC_df:\n{error_DfC_df}")
    #
    #
    # '''look for RA and CD size'''
    # ra_size_sep = ra_size_deg = cd_size_isi = cd_size_ms = None
    # # get participant name to check
    # if participant_name in ['aa', 'bb', 'cc', 'dd', 'ee']:
    #     check_name = lookup_p_name(participant_name)
    # else:
    #     check_name = participant_name
    # if check_name == 'Kris':
    #     check_name = 'Kristian'
    # print(f"\nparticipant_name: {participant_name}, check_name (): {check_name}")
    #
    # # get RA size
    # ra_size_df = pd.read_csv(r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\Exp3_Ricco_all\RA_size_df.csv")
    # print(f"ra_size_df:\n{ra_size_df}")
    # ra_names_list = ra_size_df['participant'].to_list()
    # # print(f"ra_names_list:\n{ra_names_list}")
    # if check_name in ra_names_list:
    #     ra_size_sep = ra_size_df.loc[ra_size_df['participant'] == check_name, 'separation'].values[0]
    #     ra_size_deg = ra_size_df.loc[ra_size_df['participant'] == check_name, 'degrees'].values[0]
    #
    # # get cd size
    # cd_size_df = pd.read_csv(r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\Exp2_Bloch_NM_v5\CD_size_df.csv")
    # print(f"cd_size_df:\n{cd_size_df}")
    # cd_names_list = cd_size_df['participant'].to_list()
    # # if check_name == 'Kristian':
    # #     check_name = 'Kris'
    #
    # if check_name in cd_names_list:
    #     cd_size_isi = cd_size_df.loc[cd_size_df['participant'] == check_name, 'isi'].values[0]
    #     cd_size_ms = cd_size_df.loc[cd_size_df['participant'] == check_name, 'ms'].values[0]
    #
    # ra_cd_size_dict = {'ra_size_sep': ra_size_sep, 'ra_size_deg': ra_size_deg,
    #                    'cd_size_isi': cd_size_isi, 'cd_size_ms': cd_size_ms}
    # print("\nra_cd_size_dict:")
    # for k, v in ra_cd_size_dict.items():
    #     print(f"{k}: {v}")
    #
    # if participant_name == 'cc':
    #     ra_cd_size_dict = None
    #
    # make_average_plots(all_df_path=all_df_path,
    #                    ave_df_path=p_ave_path,
    #                    error_bars_path=err_path,
    #                    thr_col='newLum',
    #                    error_type='SE',
    #                    ave_over_n=len(run_folder_names),
    #                    n_trimmed=trim_n,
    #                    exp_ave=participant_name,  # participant ave, not exp ave
    #                    ra_cd_size_dict=ra_cd_size_dict,
    #                    show_plots=True, verbose=True)


print(f'exp_path: {exp_path}')
print('\nget exp_average_data')
participant_list = ['aa', 'bb', 'cc', 'dd', 'ee', 'Nick']

trim_list = [2, 2, 2, 2, 2, 2]

# # e_average_exp_data(exp_path=exp_path, p_names_list=participant_list,
# #                    error_type='SE', n_trimmed=trim_list, verbose=True)

'''look for RA and CD size, with error bar data'''
ra_size_sep = ra_size_deg = cd_size_isi = cd_size_ms = None


# get RA size
ra_size_df = pd.read_csv(r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\Exp3_Ricco_all\RA_size_df.csv")
print(f"ra_size_df:\n{ra_size_df}")
ra_names_list = ra_size_df['participant'].to_list()
# print(f"ra_names_list:\n{ra_names_list}")

# get mean and SE/2 (half SE or error bars)
if 'exp_ave' in ra_names_list:
    ra_size_sep = ra_size_df.loc[ra_size_df['participant'] == 'exp_ave', 'separation'].values[0]
    ra_CI95_sep = ra_size_df.loc[ra_size_df['participant'] == 'CI95', 'separation'].values[0]
    ra_size_deg = ra_size_df.loc[ra_size_df['participant'] == 'exp_ave', 'degrees'].values[0]
    ra_CI95_deg = ra_size_df.loc[ra_size_df['participant'] == 'CI95', 'degrees'].values[0]

print(f"ra_size_sep: {ra_size_sep}, ra_CI95_sep: {ra_CI95_sep}\n"
      f"error bars from: {ra_size_sep - ra_CI95_sep} to {ra_size_sep + ra_CI95_sep}")

# get cd size
cd_size_df = pd.read_csv(r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\Exp2_Bloch_NM_v5\CD_size_df.csv")
print(f"cd_size_df:\n{cd_size_df}")
cd_names_list = cd_size_df['participant'].to_list()

if 'exp_ave' in cd_names_list:
    cd_size_isi = cd_size_df.loc[cd_size_df['participant'] == 'exp_ave', 'isi'].values[0]
    cd_CI95_isi = cd_size_df.loc[cd_size_df['participant'] == 'CI95', 'isi'].values[0]
    cd_ave_ms = cd_size_df.loc[cd_size_df['participant'] == 'exp_ave', 'ms'].values[0]
    cd_size_ms = cd_size_df.loc[cd_size_df['participant'] == 'CI95', 'ms'].values[0]

print(f"cd_size_isi: {cd_size_isi}, cd_CI95_isi: {cd_CI95_isi}\n"
      f"error bars from: {cd_size_isi - cd_CI95_isi} to {cd_size_isi + cd_CI95_isi}")

ra_cd_size_dict = {'ra_size_sep': ra_size_sep, 'ra_CI95_sep': ra_CI95_sep,
                   'ra_size_deg': ra_size_deg, 'ra_CI95_deg': ra_CI95_deg,
                   'cd_size_isi': cd_size_isi, 'cd_CI95_isi': cd_CI95_isi,
                   'cd_size_ms': cd_size_ms, 'cd_size_ms': cd_size_ms}
print("\nra_cd_size_dict:")
for k, v in ra_cd_size_dict.items():
    print(f"{k}: {v}")



all_df_path = os.path.join(exp_path, 'MASTER_exp_all_thr.csv')
exp_ave_path = os.path.join(exp_path, 'MASTER_exp_ave_thr.csv')
err_path = os.path.join(exp_path, 'MASTER_ave_thr_error_SE.csv')

ave_DfC_df, error_DfC_df = make_diff_from_conc_df(all_df_path, exp_path, n_trimmed=2, exp_ave=True)
print(f"ave_DfC_df:\n{ave_DfC_df}")
print(f"error_DfC_df:\n{error_DfC_df}")

exp_all_df = pd.read_csv(all_df_path)
print(f"exp_all_df:\n{exp_all_df}")



# make experiment average plots -
make_average_plots(all_df_path=all_df_path,
                   ave_df_path=exp_ave_path,
                   error_bars_path=err_path,
                   thr_col='newLum',
                   ave_over_n=len(participant_list),
                   n_trimmed=2,
                   error_type='SE',
                   exp_ave=True,
                   ra_cd_size_dict=ra_cd_size_dict,
                   show_plots=True, verbose=True)


print('\nexp1a_analysis_pipe finished\n')
