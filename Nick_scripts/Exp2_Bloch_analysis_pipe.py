import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# from exp1a_psignifit_analysis import a_data_extraction, b3_plot_staircase, c_plots, \
#     d_average_participant, e_average_exp_data, make_average_plots
from rad_flow_psignifit_analysis import b3_plot_stair_sep0, c_plots, d_average_participant
from rad_flow_psignifit_analysis import make_average_plots, e_average_exp_data

from psignifit_tools import get_psignifit_threshold_df

# # loop through run folders with first 4 scripts (a, get_psignifit_threshold_df, b3, c)
# # then run script d to get master lists and averages
exp_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/Exp2_Bloch_NM'
participant_list = ['Nick_test']  # , 'bb', 'cc', 'dd', 'ee']

p_idx_plus = 1

for p_idx, participant_name in enumerate(participant_list):
    root_path = f'{exp_path}/{participant_name}'
    # run_folder_names = [f'P{p_idx + p_idx_plus}a-{participant_name}', f'P{p_idx + p_idx_plus}b-{participant_name}',
    #                     f'P{p_idx + p_idx_plus}c-{participant_name}', f'P{p_idx + p_idx_plus}d-{participant_name}',
    #                     f'P{p_idx + p_idx_plus}e-{participant_name}', f'P{p_idx + p_idx_plus}f-{participant_name}']
    run_folder_names = [f'{participant_name}_1']  # , f'{participant_name}_2',
                        # f'{participant_name}_3', f'{participant_name}_4',
                        # f'{participant_name}_5', f'{participant_name}_6']


    for run_idx, run_dir in enumerate(run_folder_names):

        print(f'\nrunning analysis for {participant_name}, {run_dir}, {participant_name}{run_idx+1}\n')
        save_path = f'{root_path}{os.sep}{run_dir}'

        # don't delete this (participant_name = participant_name),
        # needed to ensure names go name1, name2, name3 not name1, name12, name123
        p_name = participant_name

        # '''a'''
        p_name = f'{participant_name}_{run_idx+1}'
        # p_name = f'{participant_name}{run_idx+1}'
        # ISI_ms_list = [0, 8.33, 16.67, 25, 37.5, 50, 100]
        # ISI_ms_list = [0, 16.67, 33.33, 50, 100]
        # isi_list = [0, 2, 4, 6, 9, 12, 24]

        # for 240Hz
        # isi_list = [0, 8.33, 16.67, 25, 37.5, 50, 100]

        # for 60hz
        isi_list = [0, 16.67, 33.33, 50, 100]

        isi_name_list = [f'ISI{i}' for i in isi_list]


        # for first run, some files are saved just as name not name1
        # run_data_path = f'{save_path}{os.sep}ISI_-1_probeDur2/{p_name}.csv'
        run_data_path = f'{save_path}{os.sep}{p_name}_output.csv'
        if not os.path.isfile(run_data_path):
            raise FileNotFoundError(run_data_path)

        run_data_df = pd.read_csv(run_data_path,
                                  usecols=['stair', 'stair_name', 'step',
                                           'ISI', 'ISI_frames', 'separation',
                                           'probeLum', 'trial_response'])
        run_data_df.sort_values(by=['stair', 'step'], inplace=True, ignore_index=True)
        print(f"run_data_df:\n{run_data_df}")

        stair_list = [1, 2, 3, 4]  # , 5, 6]
        sep_list = [0]*len(stair_list)
        cols_to_add_dict = {'separation': sep_list}

        '''get psignifit thresholds df - use stairs as sep levels rather than using groups'''
        thr_df = get_psignifit_threshold_df(root_path=root_path,
                                            p_run_name=run_dir,
                                            csv_name=run_data_df,
                                            n_bins=10, q_bins=True,
                                            sep_col='separation',
                                            isi_list=isi_list,
                                            sep_list=[0],
                                            cols_to_add_dict=None,
                                            verbose=True)
        print(f'thr_df:\n{thr_df}')


        '''b3'''
        # run_data_path = f'{save_path}{os.sep}RUNDATA-sorted.xlsx'
        run_data_path = f'{save_path}{os.sep}{p_name}_output.csv'
        # b3_plot_stair_sep0(run_data_path, show_plots=True)

        '''c'''
        # c_plots(save_path=save_path, isi_name_list=isi_name_list, show_plots=True)

        thr_df = pd.read_csv(f'{save_path}{os.sep}psignifit_thresholds.csv')
        thr_list = thr_df.iloc[0][1:].tolist()
        print(f'thr_df:\n{thr_df}')
        print(f'isi_list:\n{isi_list}')
        print(f'thr_list:\n{thr_list}')

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(x=isi_list, y=thr_list)
        ax.set_xticks(isi_list)
        ax.set_xticklabels(isi_name_list)
        ax.set_xlabel('Inter-stimulus Interval')
        ax.set_ylabel('Probe Luminance')
        plt.title('Bloch: 4ms probes with varying ISI')
        plt.savefig(f'{save_path}{os.sep}bloch_thr.png')
        plt.show()

    '''d'''
    d_average_participant(root_path=root_path, run_dir_names_list=run_folder_names,
                          trim_n=1, error_type='SE')

    all_df_path = f'{root_path}/MASTER_TM1_thresholds.csv'
    p_ave_path = f'{root_path}/MASTER_ave_TM_thresh.csv'
    err_path = f'{root_path}/MASTER_ave_TM_thr_error_SE.csv'
    n_trimmed = 1
    exp_ave = False

    make_average_plots(all_df_path=all_df_path,
                       ave_df_path=p_ave_path,
                       error_bars_path=err_path,
                       error_type='SE',
                       n_trimmed=n_trimmed,
                       exp_ave=False,
                       show_plots=True, verbose=True)


print(f'exp_path: {exp_path}')
print('\nget exp_average_data')

e_average_exp_data(exp_path=exp_path, p_names_list=participant_list,
                   error_type='SE', use_trimmed=True, verbose=True)


all_df_path = f'{exp_path}/MASTER_exp_thr.csv'
exp_ave_path = f'{exp_path}/MASTER_exp_ave_thr.csv'
err_path = f'{exp_path}/MASTER_ave_thr_error_SE.csv'
n_trimmed = None
exp_ave = True

make_average_plots(all_df_path=all_df_path,
                   ave_df_path=exp_ave_path,
                   error_bars_path=err_path,
                   error_type='SE',
                   n_trimmed=n_trimmed,
                   exp_ave=exp_ave,
                   show_plots=True, verbose=True)

print('\nExp2_Bloch_analaysis_pipe finished\n')
