import os

import pandas as pd

from exp1a_psignifit_analysis import a_data_extraction, b3_plot_staircase, c_plots, \
    d_average_participant, e_average_exp_data, make_average_plots

# # loop through run folders with first 4 scripts (a, get_psignifit_threshold_df, b3, c)
# # then run script d to get master lists and averages
exp_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/exp1a_data'
participant_list = ['Simon', 'Maria', 'Kristian', 'Kim']  # incomplete 'Tony' 'Martin' 'Nick'
# participant_list = ['Martin']

# use 1 for Martin, 2 for Tony, 3 for 'Simon', 4='Maria', 5='Kristian', 6='Kim'
p_idx_plus = 3

trim_list = [0, 1, 2, 3, 4, 5]

for p_idx, participant_name in enumerate(participant_list):
    root_path = f'{exp_path}/{participant_name}'
    run_folder_names = [f'P{p_idx + p_idx_plus}a-{participant_name}', f'P{p_idx + p_idx_plus}b-{participant_name}',
                        f'P{p_idx + p_idx_plus}c-{participant_name}', f'P{p_idx + p_idx_plus}d-{participant_name}',
                        f'P{p_idx + p_idx_plus}e-{participant_name}', f'P{p_idx + p_idx_plus}f-{participant_name}']

    # p_group_means_list = []
    # p_all_list = []
    # for trim_this in trim_list:
    #     print(f'\n\ntrim_this: {trim_this}')

    group_list = [1, 2]

    # check whether scrips a, b3 and c have been completed for the last run (e.g., all runs) for this participant
    check_last_c_plots_fig = f'{root_path}/{run_folder_names[-1]}/g2_dataDivOneProbe.png'
    if not os.path.isfile(check_last_c_plots_fig):
        print(f'\nNOT completed analysis yet: {check_last_c_plots_fig}')

        for run_idx, run_dir in enumerate(run_folder_names):

            # check whether scripts a, b3 and c have been done for the this run for this participant
            check_last_c_plots_fig = f'{root_path}/{run_dir}/g2_dataDivOneProbe.png'
            if os.path.isfile(check_last_c_plots_fig):
                print(f'\nalready completed: {check_last_c_plots_fig}')
                continue

                print(f'\nrunning analysis for {participant_name}, {run_dir}, {participant_name}{run_idx+1}\n')
                save_path = f'{root_path}{os.sep}{run_dir}'

                # don't delete this (participant_name = participant_name),
                # needed to ensure names go name1, name2, name3 not name1, name12, name123
                p_name = participant_name

                # '''a'''
                p_name = f'{participant_name}{run_idx+1}'
                isi_list = [-1, 0, 2, 4, 6, 9, 12, 24]

                # for first run, some files are saved just as name not name1
                check_file = f'{save_path}{os.sep}ISI_-1_probeDur2/{p_name}.csv'
                if not os.path.isfile(check_file):
                    print(f'file not found: {check_file}')
                    p_name = participant_name
                    check_file = f'{save_path}{os.sep}{p_name}'

            run_data_df = a_data_extraction(p_name=p_name, run_dir=save_path, isi_list=isi_list, verbose=True)

            run_data_path = f'{save_path}{os.sep}RUNDATA-sorted.xlsx'

            run_data_df = pd.read_excel(run_data_path, engine='openpyxl',
                                        usecols=['ISI',
                                                 'stair',
                                                 'separation', 'group',
                                                 'probeLum', 'trial_response'])
            print(f"run_data_df:\n{run_data_df}")

            stair_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
            cols_to_add_dict = {'group': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                                'separation': [18, 18, 6, 6, 3, 3, 2, 2, 1, 1, 0, 0, 20, 20]}

            '''get psignifit thresholds df - use stairs as sep levels rather than using groups'''
            thr_df = get_psignifit_threshold_df(root_path=root_path,
                                                p_run_name=run_dir,
                                                csv_name=run_data_df,
                                                n_bins=10, q_bins=True,
                                                sep_col='stair',
                                                isi_list=isi_list,
                                                sep_list=stair_list,
                                                cols_to_add_dict=cols_to_add_dict,
                                                verbose=True)
            print(f'thr_df:\n{thr_df}')


            '''b3'''
            run_data_path = f'{save_path}{os.sep}RUNDATA-sorted.xlsx'
            b3_plot_staircase(run_data_path, show_plots=True)

            '''c'''
            c_plots(save_path=save_path, show_plots=True)

    '''d'''
    d_average_participant(root_path=root_path, run_dir_names_list=run_folder_names,
                          trim_n=1, error_type='SE')

    all_df_path = f'{root_path}/MASTER_TM1_thresholds.csv'
    p_ave_path = f'{root_path}/MASTER_ave_TM_thresh.csv'
    err_path = f'{root_path}/MASTER_ave_TM_thr_error_SE.csv'
    n_trimmed = 1
    exp_ave = False

    # todo: these plots should say that they are trimmed.

    make_average_plots(all_df_path=all_df_path,
                       ave_df_path=p_ave_path,
                       error_bars_path=err_path,
                       error_type='SE',
                       n_trimmed=n_trimmed,
                       exp_ave=False,
                       show_plots=True, verbose=True)

    #     # todo: delete from here down to the bottom
    #     single_probe_means, single_probe_all_df = d_average_participant(root_path=root_path,
    #                                                 run_dir_names_list=run_folder_names,
    #                                                 error_type='SE',
    #                                                 trim_n=trim_this,
    #                                                 # show_plots=False,
    #                                                 verbose=False
    #                                                 )
    #
    #     print('\n\ngetting values for plots')
    #     print(f'single_probe_all_df:\n{single_probe_all_df}')
    #     single_probe_all_df.drop(['stack', 'separation'], axis=1, inplace=True)
    #     single_probe_all_thr_list = single_probe_all_df.stack().tolist()
    #     print(f'single_probe_all_thr_list: {len(single_probe_all_thr_list)}\n{single_probe_all_thr_list}')
    #     trim_col_vals = [f'trim_{trim_this}' for i in single_probe_all_thr_list]
    #     single_probe_all_thr_df = pd.DataFrame([trim_col_vals, single_probe_all_thr_list]).T
    #     single_probe_all_thr_df.columns = ['trim', 'thr']
    #     print(f'single_probe_all_thr_df:\n{single_probe_all_thr_df}')
    #
    #     p_all_list.append(single_probe_all_thr_df)
    #
    #     single_probe_means = single_probe_means.rename(f'trim_{trim_this}')
    #     print(f'single_probe_means:\n{single_probe_means}')
    #     p_group_means_list.append(single_probe_means)
    #
    #
    #
    # print(f'\n\nhere are all the means and err for {participant_name}')
    # # print(f'p_group_means_list:\n{p_group_means_list}')
    # p_group_means_df = pd.concat(p_group_means_list, axis=1)
    # print(f'\np_group_means_df:\n{p_group_means_df}')
    #
    # p_group_means_for_swarm_df = p_group_means_df.reset_index()
    #
    # p_group_means_for_swarm_df = pd.melt(p_group_means_for_swarm_df,
    #                                              id_vars='index',
    #                                              # value_vars='index',
    #                                              var_name='trim', )
    # print(f'\np_group_means_for_swarm_df:\n{p_group_means_for_swarm_df}')
    #
    # # print(f'\np_all_list: {np.shape(p_all_list)}\n{p_all_list}')
    # p_all_df = pd.concat(p_all_list, axis=0, ignore_index=True)
    # p_all_df['trim_2'] = [int(i[-1]) for i in list(p_all_df['trim'])]
    # p_all_df['thr'] = p_all_df['thr'].astype(float)
    # print(f'\np_all_df: {p_all_df.shape} {p_all_df.dtypes}\n{p_all_df}')
    #
    # p_mean_per_trim = p_group_means_df.mean()
    # p_mean_per_trim_df = pd.DataFrame({'trim': p_mean_per_trim.index, 'thr': p_mean_per_trim.values})
    # print(f'\np_mean_per_trim_df:\n{p_mean_per_trim_df}')
    # p_std_per_trim = p_group_means_df.std(ddof=0)
    # p_std_per_trim_df = pd.DataFrame({'trim': p_std_per_trim.index, 'std': p_std_per_trim.values})
    # print(f'\np_std_per_trim_df:\n{p_std_per_trim_df}')
    #
    # x_axis_labels = ['trim_0', 'trim_1', 'trim_2', 'trim_3', 'trim_4', 'trim_5']
    #
    # fig, ax1 = plt.subplots()  # initializes figure and plots
    # ax2 = ax1.twinx()  # applies twinx to ax2, which is the second y axis.
    #
    # sns.lineplot(data=p_mean_per_trim_df, x='trim', y='thr', marker='o', color='red', ax=ax1)
    # # # sns.scatterplot(data=p_all_df, x='trim', y='thr', ax=ax1)
    # # # sns.swarmplot(data=p_all_df, x='trim', y='thr', ax=ax1)  # all datapoints
    # sns.violinplot(data=p_all_df, x='trim_2', y='thr', ax=ax1, scale='count', saturation=.1)  # all datapoints
    # sns.swarmplot(data=p_group_means_for_swarm_df, x='trim', y='value', ax=ax1, edgecolor='black', linewidth=.9)  # all isi means
    # sns.lineplot(data=p_std_per_trim_df, x='trim', y='std', ax=ax2, color='g')
    # # these lines add the annotations for the plot.
    # ax1.set_xlabel('trim')
    # ax1.set_ylabel('threshold', color='r')
    # ax2.set_ylabel('std dev', color='g')
    # ax1.set_ylim(30, 110)
    # ax2.set_ylim(.8, 2.2)
    # plt.title(f'{participant_name} change in standard deviation as more data is trimmed')
    # plt.savefig(f'/Users/nickmartin/Documents/PycharmProjects/Cardiff/exp1a_data/{participant_name}_trim1probe_sd.png')
    # plt.show()  # shows the plot.

print(f'exp_path: {exp_path}')
print('\nget exp_average_data')

e_average_exp_data(exp_path=exp_path, p_names_list=participant_list,
                   error_type='SE', use_trimmed=True, verbose=True)

# todo: these plots don't need to mention hat they use trimmed data.

# todo: check if heatmaps with int numbers use floats for colours

all_df_path = f'{exp_path}/MASTER_exp_thr.csv'
exp_ave_path = f'{exp_path}/MASTER_exp_ave_thr.csv'
err_path = f'{exp_path}/MASTER_ave_thr_error_SE.csv'
n_trimmed = None
exp_ave=True

make_average_plots(all_df_path=all_df_path,
                   ave_df_path=exp_ave_path,
                   error_bars_path=err_path,
                   error_type='SE',
                   n_trimmed=n_trimmed,
                   exp_ave=exp_ave,
                   show_plots=True, verbose=True)

print('\nexp1a_analysis_pipe finished\n')
