import os
import pandas as pd
from exp1a_psignifit_analysis import a_data_extraction, b3_plot_staircase, c_plots, \
    d_average_participant, e_average_exp_data, make_average_plots
from psignifit_tools import get_psignifit_threshold_df
from check_home_dir import switch_path

# # loop through run folders with first 4 scripts (a, get_psignifit_threshold_df, b3, c)
# # then run script d to get master lists and averages
old_exp_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/exp1a_data'
exp_path = switch_path(old_exp_path, 'wind_oneDrive')
print(f"exp_path: {exp_path}")
participant_list = ['aa', 'bb', 'cc', 'dd', 'ee']
# participant_list = ['bb', 'cc', 'dd', 'ee']
# participant_list = ['aa']

isi_list = [-1, 0, 2, 4, 6, 9, 12, 24]

n_runs = 6

p_idx_plus = 1

for p_idx, participant_name in enumerate(participant_list):
    root_path = os.path.join(exp_path, participant_name)

    run_folder_names = [f'{participant_name}_{i+1}' for i in list(range(n_runs))]
    print(f'run_folder_names: {run_folder_names}')

    group_list = [1, 2]

    for run_idx, run_dir in enumerate(run_folder_names):

        print(f'\nrunning analysis for {participant_name}, {run_dir}, {participant_name}{run_idx+1}\n')
        save_path = f'{root_path}{os.sep}{run_dir}'

        # don't delete this (participant_name = participant_name),
        # needed to ensure names go name1, name2, name3 not name1, name12, name123
        p_name = participant_name

        # # '''a'''
        p_name = f'{participant_name}_output'  # use this one

        run_data_df = a_data_extraction(p_name=p_name, run_dir=save_path, isi_list=isi_list, verbose=True)

        '''add newLum column
        in old version, the experiment script varies probeLum and converts to float(RGB255) values for screen.
        However, monitor can only use int(RGB255).
        This function will will round RGB255 values to int(RGB255), then convert to NEW_probeLum
        LumColor255Factor = 2.395387069
        1. get probeColor255 column.
        2. convert to int(RGB255) and convert to new_Lum with int(RGB255)/LumColor255Factor
        3. add to run_data_df'''
        if 'newLum' not in run_data_df.columns.to_list():
            LumColor255Factor = 2.395387069
            rgb255_col = run_data_df['probeColor255'].to_list()
            newLum = [int(i) / LumColor255Factor for i in rgb255_col]
            run_data_df.insert(9, 'newLum', newLum)
            run_data_df.to_excel(os.path.join(save_path, 'RUNDATA-sorted.xlsx'), index=False)
            print(f"added newLum column\n"
                  f"run_data_df: {run_data_df.columns.to_list()}")


        run_data_path = os.path.join(save_path, 'RUNDATA-sorted.xlsx')

        run_data_df = pd.read_excel(run_data_path, engine='openpyxl',
                                    usecols=['ISI',
                                             'stair',
                                             'separation', 'group',
                                             # 'probeLum',
                                             'newLum', 'trial_response'])
        print(f"run_data_df:\n{run_data_df}")


        stair_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        cols_to_add_dict = {'group': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                            'separation': [18, 18, 6, 6, 3, 3, 2, 2, 1, 1, 0, 0, 20, 20]}

        '''get psignifit thresholds df - use stairs as sep levels rather than using groups'''
        thr_df = get_psignifit_threshold_df(root_path=root_path,
                                            p_run_name=run_dir,
                                            csv_name=run_data_df,
                                            n_bins=9, q_bins=True,
                                            sep_col='stair',
                                            thr_col='newLum',
                                            isi_list=isi_list,
                                            sep_list=stair_list,
                                            conf_int=True,
                                            thr_type='Bayes',
                                            plot_both_curves=False,
                                            save_plots=True,
                                            cols_to_add_dict=cols_to_add_dict,
                                            verbose=True)
        print(f'thr_df:\n{thr_df}')

        '''b3'''
        run_data_path = os.path.join(save_path, 'RUNDATA-sorted.xlsx')
        thr_path = os.path.join(save_path, 'psignifit_thresholds.csv')

        b3_plot_staircase(run_data_path, thr_col='newLum', show_plots=False)

        '''c'''
        c_plots(save_path=save_path, thr_col='newLum', show_plots=True)


    trim_n = None
    use_trimmed = False
    if len(run_folder_names) == 6:
        trim_n = 1
        use_trimmed = True
    n_trimmed = trim_n

    print(f"\n\ntrim_n: {trim_n}, use_trimmed: {use_trimmed}\n\n")

    '''d'''
    d_average_participant(root_path=root_path, run_dir_names_list=run_folder_names,
                          trim_n=n_trimmed, error_type='SE')

    all_df_path = os.path.join(root_path, 'MASTER_TM1_thresholds.csv')
    p_ave_path = os.path.join(root_path, 'MASTER_ave_TM_thresh.csv')
    err_path = os.path.join(root_path, 'MASTER_ave_TM_thr_error_SE.csv')
    n_trimmed = trim_n
    if n_trimmed is None:
        all_df_path = os.path.join(root_path, 'MASTER_psignifit_thresholds.csv')
        p_ave_path = os.path.join(root_path, 'MASTER_ave_thresh.csv')
        err_path = os.path.join(root_path, 'MASTER_ave_thr_error_SE.csv')

    exp_ave = False

    make_average_plots(all_df_path=all_df_path,
                       ave_df_path=p_ave_path,
                       error_bars_path=err_path,
                       thr_col='newLum',
                       error_type='SE',
                       n_trimmed=n_trimmed,
                       exp_ave=False,
                       show_plots=True, verbose=True)


print(f'exp_path: {exp_path}')
print('\nget exp_average_data')
participant_list = ['aa', 'bb', 'cc', 'dd', 'ee']

e_average_exp_data(exp_path=exp_path, p_names_list=participant_list,
                   error_type='SE', use_trimmed=use_trimmed, verbose=True)


all_df_path = os.path.join(exp_path, 'MASTER_exp_thr.csv')
exp_ave_path = os.path.join(exp_path, 'MASTER_exp_ave_thr.csv')
err_path = os.path.join(exp_path, 'MASTER_ave_thr_error_SE.csv')

n_trimmed = None
exp_ave = True

make_average_plots(all_df_path=all_df_path,
                   ave_df_path=exp_ave_path,
                   error_bars_path=err_path,
                   thr_col='newLum',
                   error_type='SE',
                   n_trimmed=n_trimmed,
                   exp_ave=exp_ave,
                   show_plots=True, verbose=True)

print('\nexp1a_analysis_pipe finished\n')
