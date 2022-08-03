import os
import pandas as pd
from psignifit_tools import get_psignifit_threshold_df
from rad_flow_psignifit_analysis import a_data_extraction, b3_plot_staircase
from rad_flow_psignifit_analysis import c_plots, d_average_participant, make_average_plots, e_average_exp_data

# # loop through run folders with first 5 scripts (a, b1, b2, b3, c)
# # then run script d to get master lists and averages
# exp_path = r'C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\radial_flow_exp'
exp_path = r'C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\rad_flow_half'
# participant_list = ['Kim', 'Nick', 'Simon']  # , 'Nick_half_speed']
participant_list = ['Nick_half']

# exp_path = r'C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\rad_flow_half'
# participant_list = ['Simon_half']  # , 'Nick_half_speed']

# todo: why does a_extract data work for my data but not Simon's???
# exp_path = r'C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\rad_flow_2'
# participant_list = ['Simon', 'Nick']  # , 'Nick_half_speed']

# exp_path = r'C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\rad_flow_2_half'
# participant_list = ['Nick_half_speed', 'Simon_half']  # , 'Nick_half_speed']

exp_path = os.path.normpath(exp_path)

stair_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
all_isi_list = [1, 4, 6, 9]
all_isi_names_list = [f'ISI_{i}' for i in all_isi_list]

exp_path = os.path.normpath(exp_path)

verbose = True
show_plots = False

n_runs = 12
# if the first folder to analyse is 1, p_idx_plus = 1.  If the forst folder is 5, use 5 etc.
p_idx_plus = 1

for p_idx, participant_name in enumerate(participant_list):
    # if participant_name is 'Nick':
    #     p_idx_plus = 5

    root_path = os.path.join(exp_path, participant_name)

    # # manually get run_folder_names with n_runs
    # run_folder_names = [f'{participant_name}_{i+p_idx_plus}' for i in list(range(n_runs))]

    # # search to automatically get run_folder_names
    dir_list = os.listdir(root_path)
    run_folder_names = []
    for i in range(12):  # numbers 0 to 11
        check_dir = f'{participant_name}_{i+p_idx_plus}'   # numbers 1 to 12
        if check_dir in dir_list:
            run_folder_names.append(check_dir)

    print(f'run_folder_names: {run_folder_names}')

    for run_idx, run_dir in enumerate(run_folder_names):

        # add run number , e.g., add five to access Nick_5 on the zeroth iteration
        r_idx_plus = run_idx + p_idx_plus

        print(f'\nrun_idx {run_idx+1}: running analysis for '
              f'{participant_name}, {run_dir}, {participant_name}_{r_idx_plus}\n')
        save_path = f'{root_path}{os.sep}{run_dir}'

        # # search to automatically get updated isi_list
        dir_list = os.listdir(save_path)
        run_isi_list = []
        for isi in all_isi_list:
            check_dir = f'ISI_{isi}_probeDur2'
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

        run_data_path = f'{save_path}{os.sep}ALL_ISIs_sorted.xlsx'
        run_data_df = pd.read_excel(run_data_path, engine='openpyxl',
                                    # usecols=["ISI", "stair", "stair_name",
                                    #          "step", "separation", "congruent",
                                    #          "flow_dir", "probe_jump", "corner",
                                    #          "newLum", "trial_response"]
                                    )
        print(f"run_data_df: {run_data_df.columns.to_list()}\n{run_data_df}")

        # todo: I don't need newLum column!  This was to turn float(RGB255) back into int(rgb255), but these exps use float(rgb).
        # '''add newLum column
        #         in old version, the experiment script varies probeLum and converts to float(RGB255) values for screen.
        #         However, monitor can only use int(RGB255).
        #         This function will will round RGB255 values to int(RGB255), then convert to NEW_probeLum
        #         LumColor255Factor = 2.395387069
        #         1. get probeColor255 column.
        #         2. convert to int(RGB255) and convert to new_Lum with int(RGB255)/LumColor255Factor
        #         3. add to run_data_df'''
        # if 'newLum' not in run_data_df.columns.to_list():
        #     LumColor255Factor = 2.395387069
        #     rgb255_col = run_data_df['probeColor255'].to_list()
        #     newLum = [int(i) / LumColor255Factor for i in rgb255_col]
        #     run_data_df.insert(12, 'newLum', newLum)
        #     run_data_df.to_excel(os.path.join(save_path, 'ALL_ISIs_sorted.xlsx'), index=False)
        #     print(f"added newLum column\n"
        #           f"run_data_df: {run_data_df.columns.to_list()}")


        '''get psignifit thresholds df'''
        cols_to_add_dict = {'stair_names': [18, -18, 6, -6, 3, -3, 2, -2, 1, -1, 0, -0.1],
                            'congruent':  [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
                            'separation': [18, 18, 6, 6, 3, 3, 2, 2, 1, 1, 0, 0]}
        thr_df = get_psignifit_threshold_df(root_path=root_path,
                                            p_run_name=run_dir,
                                            csv_name=run_data_df,
                                            n_bins=9, q_bins=True,
                                            sep_col='stair',
                                            thr_col='probeLum',
                                            isi_list=run_isi_list,
                                            sep_list=stair_list,
                                            conf_int=True,
                                            thr_type='Bayes',
                                            plot_both_curves=False,
                                            cols_to_add_dict=cols_to_add_dict,
                                            show_plots=False,
                                            verbose=verbose)
        print(f'thr_df:\n{thr_df}')


        '''b3'''
        b3_plot_staircase(run_data_path, thr_col='probeLum', show_plots=show_plots, verbose=verbose)

        '''c I don't actually need any of these, instead sort get psignifit thr ands make plots from those.'''
        c_plots(save_path=save_path, thr_col='probeLum', isi_name_list=run_isi_names_list, show_plots=show_plots, verbose=verbose)



    '''d participant averages'''
    trim_n = None
    if len(run_folder_names) == 12:
        trim_n = 2
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
                       n_trimmed=trim_n,
                       exp_ave=False,
                       show_plots=True, verbose=True)


print(f'exp_path: {exp_path}')
participant_list = ['Nick_half', 'Simon_half']
print('\nget exp_average_data')
e_average_exp_data(exp_path=exp_path, p_names_list=participant_list,
                   error_type='SE', n_trimmed=trim_n, verbose=True)


all_df_path = os.path.join(exp_path, "MASTER_exp_thr.csv")
exp_ave_path = os.path.join(exp_path, "MASTER_exp_ave_thr.csv")
err_path = os.path.join(exp_path, "MASTER_ave_thr_error_SE.csv")

exp_ave = True

make_average_plots(all_df_path=all_df_path,
                   ave_df_path=exp_ave_path,
                   error_bars_path=err_path,
                   thr_col='probeLum',
                   n_trimmed=trim_n,
                   exp_ave=exp_ave,
                   show_plots=True, verbose=True)



print('\nrad_flow_analysis_pipe finished')
