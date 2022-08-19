import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from rad_flow_psignifit_analysis import a_data_extraction, b3_plot_staircase, c_plots, \
    d_average_participant, e_average_exp_data, make_average_plots, plot_w_errors_either_x_axis
# from flow_parsing_psignifit_analysis import make_average_plots, e_average_exp_data

from psignifit_tools import get_psignifit_threshold_df
from check_home_dir import switch_path

# # loop through run folders with first 4 scripts (a, get_psignifit_threshold_df, b3, c)
# # then run script d to get master lists and averages
# exp_path = r"C:\Users\sapnm4\PycharmProjects\Cardiff\Nick_scripts\EXP1_speed_detection"
exp_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\Exp1_speed_detection"

convert_path1 = os.path.normpath(exp_path)
exp_path = convert_path1

print(f"exp_path: {exp_path}")
participant_list = ['Nick']

probe_speed_list = [0, .25, .5, .75, 1.0, 1.5, 2.0]

# make data_extraction for joining different probe durs together
probeDur = 4

n_runs = 12

p_idx_plus = 1

for p_idx, participant_name in enumerate(participant_list):
    root_path = os.path.join(exp_path, participant_name)

    # run_folder_names = [f'{participant_name}_{i+1}' for i in list(range(n_runs))]
    # print(f'run_folder_names: {run_folder_names}')
    # # search to automatically get run_folder_names
    dir_list = os.listdir(root_path)
    run_folder_names = []
    for i in range(12):  # numbers 0 to 11
        check_dir = f'{participant_name}_{i+p_idx_plus}'   # numbers 1 to 12
        if check_dir in dir_list:
            run_folder_names.append(check_dir)
    print(f"run_folder_names: {run_folder_names}")

    for run_idx, run_dir in enumerate(run_folder_names):

        print(f'\nrunning analysis for {participant_name}, {run_dir}, {participant_name}_{run_idx+1}\n')
        # print(f'\nrunning analysis for {participant_name}\n')

        save_path = os.path.join(root_path, run_dir)
        dir_list = os.listdir(save_path)
        probeDur_list = []
        for i in range(25):  # numbers 0 to 11
            check_dir = f'probeDur{i}'  # numbers 1 to 12
            if check_dir in dir_list:
                probeDur_list.append(check_dir)
        print(f"probeDur_list: {probeDur_list}")

        run_data = []

        for probeDur in probeDur_list:
            # don't delete this (participant_name = participant_name),
            # needed to ensure names go name1, name2, name3 not name1, name12, name123
            p_name = participant_name

            # # '''a'''
            p_name = f'{participant_name}_{run_idx+1}_output'  # use this one

            # todo: make data extraxction script to join diffewrent probe duratons together.
            # run_data_df = a_data_extraction(p_name=p_name, run_dir=save_path, isi_list=isi_list, verbose=True)

            dur_data_df = pd.read_csv(os.path.join(save_path, probeDur, f'{p_name}.csv'))
            column_names = list(dur_data_df)
            print(f"dur_data_df:\n{dur_data_df}")

            run_data.append(dur_data_df)

        run_data_shape = np.shape(run_data)
        sheets, rows, columns = np.shape(run_data)
        run_data = np.reshape(run_data, newshape=(sheets * rows, columns))
        print(f'run_data reshaped from {run_data_shape} to {np.shape(run_data)}')
        run_data_df = pd.DataFrame(run_data, columns=column_names)
        if 'Unnamed: 0' in list(run_data_df):
            run_data_df.drop('Unnamed: 0', axis=1, inplace=True)
        print(f"run_data_df:\n{run_data_df}")
        save_name = 'RUNDATA-sorted.xlsx'
        save_excel_path = os.path.join(save_path, save_name)
        print(f"\nsaving run_data_df to save_excel_path:\n{save_excel_path}")
        run_data_df.to_excel(save_excel_path, index=False)

        run_data_df = pd.read_excel(save_excel_path, engine='openpyxl')

        run_data_df = run_data_df.sort_values(by=['stair', 'trial_number'])

        # todo: change psignifit top NewLum once I have new data



        run_data_path = os.path.join(save_path, 'RUNDATA-sorted.xlsx')

        run_data_df = pd.read_excel(run_data_path, engine='openpyxl')
        print(f"run_data_df:\n{run_data_df}")


        probe_speed_list = list(run_data_df['probeSpeed'].unique())
        sep_list = list(run_data_df['separation'].unique())
        isi_list = list(run_data_df['ISI'].unique())
        dur_list = list(run_data_df['probe_dur'].unique())
        stair_list = list(run_data_df['stair'].unique())
        stair_names_list = list(run_data_df['stair_name'].unique())
        cols_to_add_dict = {'stair': stair_list, 'stair_name': stair_names_list,
                            # 'probeSpeed': probe_speed_list,
                            'separation': sep_list, 'isi': isi_list, 'dur': dur_list}

        '''get psignifit thresholds df - use stairs as sep levels rather than using groups'''
        thr_df = get_psignifit_threshold_df(root_path=root_path,
                                            p_run_name=run_dir,
                                            csv_name=run_data_df,
                                            n_bins=9, q_bins=True,
                                            sep_col='probeSpeed',
                                            sep_list=probe_speed_list,
                                            thr_col='probeLum',
                                            isi_col='probe_dur',
                                            isi_list=dur_list,
                                            conf_int=True,
                                            thr_type='Bayes',
                                            plot_both_curves=False,
                                            save_plots=True,
                                            cols_to_add_dict=None,
                                            verbose=True)
        print(f'thr_df:\n{thr_df}')
        #
        # '''b3'''
        # run_data_path = os.path.join(save_path, 'RUNDATA-sorted.xlsx')
        # thr_path = os.path.join(save_path, 'psignifit_thresholds.csv')
        #
        # # b3 needs 'total_Ntrials' as column header, which I have changed to trial_number.
        # # b3_plot_staircase(run_data_path, thr_col='newLum', show_plots=False)
        #
        # '''c'''
        # # c_plots(save_path=save_path, thr_col='newLum', show_plots=True)


    trim_n = None
    if len(run_folder_names) == 12:
        trim_n = 2

    print(f"\n\ntrim_n: {trim_n}, \n\n")

    '''d'''
    d_average_participant(root_path=root_path, run_dir_names_list=run_folder_names,
                          trim_n=trim_n, error_type='SE')

    all_df_path = os.path.join(root_path, f'MASTER_TM{trim_n}_thresholds.csv')
    p_ave_path = os.path.join(root_path, f'MASTER_ave_TM{trim_n}_thresh.csv')
    err_path = os.path.join(root_path, f'MASTER_ave_TM{trim_n}_thr_error_SE.csv')
    if trim_n is None:
        all_df_path = os.path.join(root_path, 'MASTER_psignifit_thresholds.csv')
        p_ave_path = os.path.join(root_path, 'MASTER_ave_thresh.csv')
        err_path = os.path.join(root_path, 'MASTER_ave_thr_error_SE.csv')

    p_all_df = pd.read_csv(all_df_path)
    print(f'p_all_df\n{p_all_df}')
    p_ave_df = pd.read_csv(p_ave_path)
    probe_speed_list = list(p_ave_df['probeSpeed'].unique())
    print(f'p_ave_df\n{p_ave_df}')
    print(f'probe_speed_list\n{probe_speed_list}')

    # make_average_plots(all_df_path=all_df_path,
    #                    ave_df_path=p_ave_path,
    #                    error_bars_path=err_path,
    #                    # thr_col='newLum',
    #                    n_trimmed=trim_n,
    #                    exp_ave=False,  # participant ave, not exp ave
    #                    dur_values_list=probe_speed_list,
    #                    show_plots=True, verbose=True)

    save_path, df_name = os.path.split(p_ave_path)

    if trim_n is not None:
        fig_1b_title = f'P average thresholds per stair across all runs (trim={trim_n}).\n' \
                       f'Bars=.68 CI'
        fig_1b_savename = f'ave_TM_thr_all_runs.png'
    else:
        fig_1b_title = f'P average threshold per stair across all runs\n' \
                       f'Bars=.68 CI'
        fig_1b_savename = f'ave_thr_all_runs.png'

    headers = list(p_all_df.columns)

    substring = 'probe_dur_'

    # ✅ find all strings in list that contain substring
    probe_dur_cols = [item for item in headers if substring in item]
    # print(all_matches)
    plot_w_errors_either_x_axis(wide_df=p_all_df, cols_to_keep=['probeSpeed'],
                                cols_to_change=probe_dur_cols,
                                cols_to_change_show='probeLum', new_col_name='probe_dur',
                                strip_from_cols=substring, x_axis='probeSpeed', y_axis='probeLum',
                                hue_var='probe_dur', style_var=None, style_order=None,
                                error_bars=True, even_spaced_x=True,
                                x_tick_vals=probe_speed_list,
                                jitter=False,  # .01,
                                fig_title=fig_1b_title, fig_savename=fig_1b_savename,
                                save_path=save_path, verbose=True)
    plt.show()
    plt.close()



# print(f'exp_path: {exp_path}')
# print('\nget exp_average_data')
# # participant_list = ['aa', 'bb', 'cc', 'dd', 'ee']
#
# e_average_exp_data(exp_path=exp_path, p_names_list=participant_list,
#                    error_type='SE',
#                    n_trimmed=trim_n,
#                    exp_type='speed_detection',
#                    verbose=True)
#
#
# all_df_path = os.path.join(exp_path, 'MASTER_exp_thr.csv')
# exp_ave_path = os.path.join(exp_path, 'MASTER_exp_ave_thr.csv')
# err_path = os.path.join(exp_path, 'MASTER_ave_thr_error_SE.csv')
#
# save_path, df_name = os.path.split(p_ave_path)
#
# if trim_n is not None:
#     fig_1b_title = f'P average thresholds per stair across all runs (trim={trim_n}).\n' \
#                    f'Bars=.68 CI'
#     fig_1b_savename = f'ave_TM_thr_all_runs.png'
# else:
#     fig_1b_title = f'P average threshold per stair across all runs\n' \
#                    f'Bars=.68 CI'
#     fig_1b_savename = f'ave_thr_all_runs.png'
# plot_w_errors_either_x_axis(wide_df=p_all_df, cols_to_keep=['probeSpeed'],
#                             cols_to_change=['ISI_0'],
#                             cols_to_change_show='probeLum', new_col_name='ISI',
#                             strip_from_cols=None, x_axis='probeSpeed', y_axis='probeLum',
#                             hue_var=None, style_var=None, style_order=None,
#                             error_bars=True, even_spaced_x=True,
#                             x_tick_vals=probe_speed_list,
#                             jitter=False,  # .01,
#                             fig_title=fig_1b_title, fig_savename=fig_1b_savename,
#                             save_path=save_path, verbose=True)
# plt.show()
# plt.close()

#
# # make experiment average plots -
# make_average_plots(all_df_path=all_df_path,
#                    ave_df_path=exp_ave_path,
#                    error_bars_path=err_path,
#                    thr_col='newLum',
#                    error_type='SE',
#                    exp_ave=True,
#                    show_plots=True, verbose=True)

print('\nexp1a_analysis_pipe finished\n')
