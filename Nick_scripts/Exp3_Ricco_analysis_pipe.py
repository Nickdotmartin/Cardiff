import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# from exp1a_psignifit_analysis import a_data_extraction, b3_plot_staircase, c_plots, \
#     d_average_participant, e_average_exp_data, make_average_plots
from rad_flow_psignifit_analysis import b3_plot_staircase, b3_plot_stair_sep0, c_plots, d_average_participant
from rad_flow_psignifit_analysis import make_average_plots, e_average_exp_data, plot_runs_ave_w_errors

from psignifit_tools import get_psignifit_threshold_df

# # loop through run folders with first 4 scripts (a, get_psignifit_threshold_df, b3, c)
# # then run script d to get master lists and averages
exp_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/Exp3_Ricco_NM'
participant_list = ['Nick']  # , 'bb', 'cc', 'dd', 'ee']
n_runs = 3

p_idx_plus = 1

for p_idx, participant_name in enumerate(participant_list):
    root_path = f'{exp_path}/{participant_name}'
    # run_folder_names = [f'{participant_name}_1', f'{participant_name}_2',
    #                     f'{participant_name}_3']  # , f'{participant_name}_4',
    #                     f'{participant_name}_5', f'{participant_name}_6']
    run_folder_names = [f'{participant_name}_{i+1}' for i in list(range(n_runs))]
    print(f'run_folder_names: {run_folder_names}')

    for run_idx, run_dir in enumerate(run_folder_names):

        print(f'\nrunning analysis for {participant_name}, {run_dir}, {participant_name}{run_idx+1}\n')
        save_path = f'{root_path}{os.sep}{run_dir}'

        # don't delete this (participant_name = participant_name),
        # needed to ensure names go name1, name2, name3 not name1, name12, name123
        p_name = participant_name

        # '''a'''
        p_name = f'{participant_name}_{run_idx+1}'

        # # for first run, some files are saved just as name not name1
        run_data_path = f'{save_path}{os.sep}{p_name}_output.csv'
        if not os.path.isfile(run_data_path):
            raise FileNotFoundError(run_data_path)

        print(f'run_data_path: {run_data_path}')

        run_data_df = pd.read_csv(run_data_path)
        run_data_df.sort_values(by=['stair', 'step'], inplace=True, ignore_index=True)
        #
        # # save sorted csv
        # run_data_df.to_csv(run_data_path)
        # print(f"run_data_df:\n{run_data_df}")
        #
        # extract values from dataframe
        sep_list = run_data_df['separation'].unique()
        stair_list = list(range(len(sep_list)))
        isi_list = [0]
        # cols_to_add_dict = {}
        print(f'isi_list: {isi_list}')
        print(f'stair_list: {stair_list}')
        print(f'sep_list: {sep_list}')

        '''get psignifit thresholds df - use stairs as sep levels rather than using groups'''
        thr_df = get_psignifit_threshold_df(root_path=root_path,
                                            p_run_name=run_dir,
                                            csv_name=run_data_df,
                                            n_bins=10, q_bins=True,
                                            sep_col='separation',
                                            isi_list=isi_list,
                                            sep_list=sep_list,
                                            cols_to_add_dict=None,
                                            verbose=True)


        print(f'thr_df: {type(thr_df)}\n{thr_df}')


        '''b3'''
        run_data_path = f'{save_path}{os.sep}{p_name}_output.csv'
        # todo: Ricco doesn't currently work with either staricase script - make a new one?
        # b3_plot_staircase(run_data_path, show_plots=True)

        '''c'''
        print('*** making threshold plot ***')
        # c_plots(save_path=save_path, isi_name_list=isi_name_list, show_plots=True)
        fps = run_data_df['3_fps'].iloc[0]
        one_frame = 1000/fps
        probe_dur = round(2*one_frame, 3)
        print(f'probe_dur: {probe_dur} at {fps} fps')

        thr_df_path = f'{save_path}{os.sep}psignifit_thresholds.csv'
        thr_df = pd.read_csv(thr_df_path)
        thr_df["separation"][0] = -1
        print(f'thr_df["separation"][0]: {thr_df["separation"][0]}')
        print(f'thr_df:\n{thr_df}')

        sep_list = thr_df['separation'].unique()


        thr_df_cols = thr_df.columns.tolist()

        sep_vals_list = [i for i in sep_list]
        sep_name_list = ['1pr' if i == -1 else f'sep{i}' for i in sep_list]
        print(f'sep_vals_list: {sep_vals_list}')
        print(f'sep_name_list: {sep_name_list}')

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=thr_df, x='separation', y='ISI_0', marker='o')
        ax.set_xticks(sep_vals_list)
        ax.set_xticklabels(sep_name_list)
        ax.set_xlabel('Probe length')
        ax.set_ylabel('Probe Luminance')
        plt.title(f'Ricco: {probe_dur}ms probes with length')
        plt.savefig(f'{save_path}{os.sep}ricco_thr.png')
        plt.show()
        print('*** finished threshold plot ***')

    '''d'''
    trim_n = None
    if len(run_folder_names) == 12:
        trim_n = 1
    d_average_participant(root_path=root_path, run_dir_names_list=run_folder_names,
                          trim_n=trim_n, error_type='SE')


    # making average plot
    all_df_path = f'{root_path}/MASTER_TM1_thresholds.csv'
    p_ave_path = f'{root_path}/MASTER_ave_TM_thresh.csv'
    err_path = f'{root_path}/MASTER_ave_TM_thr_error_SE.csv'
    n_trimmed = trim_n
    if n_trimmed is None:
        all_df_path = f'{root_path}/MASTER_psignifit_thresholds.csv'
        p_ave_path = f'{root_path}/MASTER_ave_thresh.csv'
        err_path = f'{root_path}/MASTER_ave_thr_error_SE.csv'

    exp_ave = False

    # load data and change order to put 1pr last
    print('*** making average plot ***')
    fig_df = pd.read_csv(p_ave_path)
    fig_df.columns = ['cond', 'thr']
    sep_vals_list = fig_df.cond.tolist()

    fig_df = fig_df.set_index('cond')


    error_df = pd.read_csv(err_path)
    error_df.columns = ['cond', 'thr']
    print(f'fig_df:\n{fig_df}')
    print(f'error_df:\n{error_df}')

    sep_vals_list = [-1 if i == -99 else i for i in sep_vals_list]
    sep_name_list = ['1pr' if i == -1 else i for i in sep_vals_list]
    print(f'sep_vals_list: {sep_vals_list}')
    print(f'sep_name_list: {sep_name_list}')

    fig_title = 'Participant average thresholds - Ricco'
    save_name = 'ave_thr_all_runs.png'
    plot_runs_ave_w_errors(fig_df=fig_df, error_df=error_df,
                           jitter=True, error_caps=True, alt_colours=False,
                           legend_names=None,
                           x_tick_vals=sep_vals_list,
                           x_tick_labels=sep_name_list,
                           even_spaced_x=False, fixed_y_range=False,
                           x_axis_label='Probe length',
                           fig_title=fig_title, save_name=save_name,
                           save_path=root_path, verbose=True)
    plt.show()
    print('*** finished average plot ***')

    make_average_plots(all_df_path=all_df_path,
                       ave_df_path=p_ave_path,
                       error_bars_path=err_path,
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
                   # error_type='SE',
                   n_trimmed=n_trimmed,
                   exp_ave=exp_ave,
                   show_plots=True, verbose=True)

print('\nExp2_Bloch_analysis_pipe finished\n')
