import os
import pandas as pd
from exp1a_psignifit_analysis import plot_thr_heatmap, plt_heatmap_row_col, \
    a_data_extraction, b3_plot_staircase, c_plots, \
    d_average_participant, e_average_exp_data, make_average_plots
from psignifit_tools import get_psignifit_threshold_df
from python_tools import switch_path

# # loop through run folders with first 4 scripts (a, get_psignifit_threshold_df, b3, c)
# # then run script d to get master lists and averages
# old_exp_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/exp1a_data'
# exp_path = switch_path(old_exp_path, 'wind_oneDrive')
exp_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\EXP1_direction_detection_2AFC"
convert_path1 = os.path.normpath(exp_path)
exp_path = convert_path1

print(f"exp_path: {exp_path}")
participant_list = ['Nick']

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

    master_acc_list = []

    for run_idx, run_dir in enumerate(run_folder_names):

        print(f'\nrunning analysis for {participant_name}, {run_dir}, {participant_name}{run_idx+1}\n')
        # print(f'\nrunning analysis for {participant_name}\n')

        save_path = f'{root_path}{os.sep}{run_dir}'
        # save_path = f'{root_path}'

        # don't delete this (participant_name = participant_name),
        # needed to ensure names go name1, name2, name3 not name1, name12, name123
        p_name = participant_name

        # # '''a'''
        p_name = f'{participant_name}_output'  # use this one

    #     # # I don't need data extraction as all ISIs are in same df.
    #     run_data_df = pd.read_csv(os.path.join(save_path, f'{p_name}.csv'))
    #     print(f"run_data_df ({list(run_data_df.columns)}):\n{run_data_df}")
    #
    #     simple_df = run_data_df[['stair', 'separation', 'ISI', 'trial_response']]
    #     print(f"simple_df ({list(simple_df.columns)}):\n{simple_df}")
    #
    #     mean_df = simple_df.groupby(['separation', 'ISI'], sort=True).mean()
    #     # print(f"mean_df ({list(mean_df.columns)}):\n{mean_df}")
    #     mean_df = mean_df.rename(columns={'trial_response': 'accuracy'})
    #
    #     mean_df.reset_index(drop=False, inplace=True)
    #     rows, cols = mean_df.shape
    #     mean_df.insert(0, 'run', [run_idx+1] * rows)
    #     print(f"mean_df ({list(mean_df.columns)}):\n{mean_df}")
    #
    #     master_acc_list.append(mean_df)
    #
    # all_data_df = pd.concat(master_acc_list, ignore_index=True)
    # print(f"all_data_df ({list(all_data_df.columns)}):\n{all_data_df}")

    # all_mean_df = all_data_df.groupby(['separation', 'ISI'], sort=True).mean()
    # all_mean_df.reset_index(drop=False, inplace=True)
    # all_mean_df.drop(['run', 'stair'], axis=1, inplace=True)
    # print(f"all_mean_df ({list(all_mean_df.columns)}):\n{all_mean_df}")


    save_name = 'master_detection_accuracy.csv'
    # all_mean_df.to_csv(os.path.join(root_path, save_name))
    all_mean_df = pd.read_csv(os.path.join(root_path, save_name))

    ave_over_n = len(run_folder_names)

    heat_df = all_mean_df.pivot(index='separation', columns='ISI', values='accuracy')
    print(f"heat_df ({list(heat_df.columns)}):\n{heat_df}")

    plot_thr_heatmap(heatmap_df=heat_df,
                     x_tick_labels=['ISI 0', 'ISI 2', 'ISI 4', 'ISI 6'],
                     y_tick_labels=[0, 2, 4, 6],
                     fig_title=f'Direction detection Accuracy\n(chance=50%, n={ave_over_n})',
                     save_name='Accuracy_heatmap.png',
                     save_path=root_path,
                     verbose=True)

    heatmap_pr_title = f'P Heatmap per row (n={ave_over_n})'
    heatmap_pr_savename = 'mean_heatmap_per_row'

    plt_heatmap_row_col(heatmap_df=heat_df,
                        colour_by='row',
                        x_tick_labels=None,
                        x_axis_label='ISI',
                        y_tick_labels=None,
                        y_axis_label='Separation',
                        fig_title=heatmap_pr_title,
                        save_name=heatmap_pr_savename,
                        save_path=root_path,
                        verbose=True)

    heatmap_pc_title = f'P Heatmap per col (n={ave_over_n})'
    heatmap_pc_savename = 'mean_heatmap_per_col'
    plt_heatmap_row_col(heatmap_df=heat_df,
                        colour_by='col',
                        x_tick_labels=None,
                        x_axis_label='ISI',
                        y_tick_labels=None,
                        y_axis_label='Separation',
                        fig_title=heatmap_pc_title,
                        save_name=heatmap_pc_savename,
                        save_path=root_path,
                        verbose=True)

print('\nexp1a_analysis_pipe finished\n')
