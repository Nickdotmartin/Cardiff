import os
import pandas as pd
import numpy as np
from psignifit_tools import get_psignifit_threshold_df, get_psig_thr_w_hue
from rad_flow_psignifit_analysis import a_data_extraction, b3_plot_staircase
from rad_flow_psignifit_analysis import c_plots, d_average_participant
from rad_flow_psignifit_analysis import make_average_plots, e_average_exp_data
from python_tools import switch_path

from exp1a_psignifit_analysis import plt_heatmap_row_col, make_average_plots


exp_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\Exp1_Jan23_radial_v4"
exp_path = os.path.normpath(exp_path)
print(f'exp_path: {exp_path}')

participant_list = ['Nick']  # 'Nick'
analyse_from_run = 1

verbose = True
show_plots = True

n_runs = 12  # 12
# if the first folder to analyse is 1, analyse_from_run = 1.  If the first folder is 5, use 5 etc.

for p_idx, participant_name in enumerate(participant_list):

    root_path = os.path.join(exp_path, participant_name)

    # # search to automatically get run_folder_names
    dir_list = os.listdir(root_path)
    run_folder_names = []
    for i in range(n_runs):  # numbers 0 to 11
        check_dir = f'{participant_name}_{i+analyse_from_run}'   # numbers 1 to 12
        if check_dir in dir_list:
            run_folder_names.append(check_dir)

    print(f'run_folder_names: {run_folder_names}')

    '''for trimming'''
    trim_n = None
    if len(run_folder_names) == 12:
        trim_n = 2
    print(f'\ntrim_n: {trim_n}')

    for run_idx, run_dir in enumerate(run_folder_names):

        # add run number , e.g., add five to access Nick_5 on the zeroth iteration
        r_idx_plus = run_idx + analyse_from_run

        print(f'\nrun_idx {run_idx+1}: running analysis for '
              f'{participant_name}, {run_dir}, {participant_name}_{r_idx_plus}')
        save_path = os.path.join(root_path, run_dir, 'sep_5')

        # don't delete this (p_name = participant_name),
        # needed to ensure names go name1, name2, name3 not name1, name12, name123
        p_name = participant_name

        '''a'''
        # # I don't need data extraction as all ISIs are in same df.
        p_name = f'{participant_name}_output'  # use this one
        try:
            run_data_df = pd.read_csv(os.path.join(save_path, f'{p_name}.csv'))
        except:
            p_name = f'{participant_name}_{r_idx_plus}_output'
            run_data_df = pd.read_csv(os.path.join(save_path, f'{p_name}.csv'))

        run_data_df = pd.read_csv(os.path.join(save_path, f'{p_name}.csv'))
        run_data_df = run_data_df.sort_values(by=['stair', 'trial_number'])
        print(f"run_data_df: {run_data_df.columns.to_list()}\n{run_data_df}")

        # # check for old column names and add new columns
        col_names = run_data_df.columns.to_list()

        if 'cond_type' not in col_names:
            if 'jump_dir' in col_names:
                if 'outward' in run_data_df['jump_dir'].to_list():
                    cond_type_list = [-1 if x == 'outward' else 1 for x in run_data_df['jump_dir'].to_list()]
                elif 'cont' in run_data_df['jump_dir'].to_list():
                    cond_type_list = [-1 if x == 'exp' else 1 for x in run_data_df['jump_dir'].to_list()]
                run_data_df['cond_type'] = cond_type_list
                # run_data_df.rename(columns={'jump_dir': 'cond_type'}, inplace=True)

        # # todo: should be able to delete this now I've added neg_sep to exp scripts.
        # add neg sep column to make batman plots
        if 'neg_sep' not in col_names:
            def make_neg_sep(df):
                if (df.cond_type in [-1]) and (df.separation == 0.0):
                    return -.1
                elif df.cond_type in [-1]:
                    return 0 - df.separation
                else:
                    return df.separation
            run_data_df.insert(7, 'neg_sep', run_data_df.apply(make_neg_sep, axis=1))
            print('\nadded neg_sep col')
            print(run_data_df['neg_sep'].to_list())


        if 'Unnamed: 0' in col_names:
            run_data_df.drop('Unnamed: 0', axis=1, inplace=True)

        print(f"run_data_df: {run_data_df.columns.to_list()}\n{run_data_df}")

        # todo: save csv?
        # run_data_df.to_csv(os.path.join(save_path, f'{p_name}.csv'), index=False)

        # # # get cond details for this exp
        isi_list = run_data_df['ISI'].unique().tolist()
        separation_list = run_data_df['separation'].unique().tolist()
        neg_sep_list = run_data_df['neg_sep'].unique().tolist()
        jump_dir_list = run_data_df['jump_dir'].unique().tolist()
        cond_type_list = run_data_df['cond_type'].unique().tolist()
        plot_names_list = ['sep5_exp' if i == -1 else 'sep5_cont' for i in cond_type_list]
        print(f'isi_list: {isi_list}')
        print(f'separation_list: {separation_list}')
        print(f'neg_sep_list: {neg_sep_list}')
        print(f'jump_dir_list: {jump_dir_list}')
        print(f'cond_type_list: {cond_type_list}')
        print(f'plot_names_list: {plot_names_list}')

        cols_to_add_dict = {'neg_sep': neg_sep_list,
                            'jump_dir': jump_dir_list,
                            'plot_names': plot_names_list}

        print(f'cols_to_add_dict:')
        for k, v in cols_to_add_dict.items():
            print(k, v)
        thr_df = get_psig_thr_w_hue(root_path=root_path,
                                    p_run_name=run_dir,
                                    output_df=run_data_df,
                                    n_bins=9, q_bins=True,
                                    thr_col='probeLum',
                                    sep_col='separation', sep_list=separation_list,
                                    isi_col='ISI', isi_list=isi_list,
                                    hue_col='cond_type', hue_list=cond_type_list,
                                    trial_correct_col='trial_response',
                                    conf_int=True,
                                    thr_type='Bayes',
                                    plot_both_curves=False,
                                    cols_to_add_dict=cols_to_add_dict,
                                    show_plots=False,
                                    save_plots=False,
                                    verbose=verbose)
        print(f'thr_df:\n{thr_df}')

    # run_folder_names = ['Nick_1', 'Nick_5', 'Nick_6']
    '''d participant averages'''
    d_average_participant(root_path=root_path, run_dir_names_list=run_folder_names,
                          groupby_col=['cond_type', 'neg_sep'],
                          # groupby_col='cond_type',
                          cols_to_drop=['stack', 'jump_dir', 'plot_names'], cols_to_replace='separation',
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

    all_df = pd.read_csv(all_df_path)
    separation_list = all_df['separation'].unique().tolist()

    if len(separation_list) > 1:

        col_names = all_df.columns.to_list()
        print(f"col_names: {col_names}")

        isi_cols_list = [i for i in col_names if 'ISI_' in i]
        isi_vals_list = [int(i[4:]) for i in isi_cols_list]
        isi_names_list = ['conc' if i == 'ISI_-1' else i for i in isi_cols_list]
        neg_sep_list = all_df['neg_sep'].unique().tolist()

        make_average_plots(all_df_path=all_df_path,
                           ave_df_path=p_ave_path,
                           error_bars_path=err_path,
                           thr_col='probeLum',
                           n_trimmed=trim_n,
                           ave_over_n=len(run_folder_names),
                           exp_ave=participant_name,
                           isi_name_list=isi_cols_list,
                           show_plots=True, verbose=True)

    else:

        print('\njust one sep')

        all_df = pd.read_csv(all_df_path)
        p_ave_df = pd.read_csv(p_ave_path)
        err_df = pd.read_csv(err_path)

        df_list = [all_df, p_ave_df, err_df]
        df_path_list = [all_df_path, p_ave_path, err_path]
        for idx, df in enumerate(df_list):
            col_names = df.columns.to_list()

            if 'plot_sep_name' not in col_names:
                plot_sep_name_list = ['sep -5\nexp' if i == -1 else 'sep +5\ncont' for i in df['cond_type'].to_list()]

            if 'separation' in col_names:
                df.drop('separation', axis=1, inplace=True)

            if 'neg_sep' in col_names:
                neg_sep_list = df['neg_sep'].unique().tolist()
                df.rename(columns={'neg_sep': 'separation'}, inplace=True)

            if 'cond_type' in col_names:
                df.drop('cond_type', axis=1, inplace=True)

            if 'plot_names' in col_names:
                df.drop('plot_names', axis=1, inplace=True)

            if 'plot_sep_name' in col_names:
                df.drop('plot_sep_name', axis=1, inplace=True)

            if 'jump_dir' in col_names:
                df.drop('jump_dir', axis=1, inplace=True)

            if 'Unnamed: 0' in col_names:
                df.drop('Unnamed: 0', axis=1, inplace=True)

            df.to_csv(df_path_list[idx], index=False)

            col_names = df.columns.to_list()
            isi_cols_list = [i for i in col_names if 'ISI_' in i]
            print(f"plot_sep_name_list: {plot_sep_name_list}")
            print(f"neg_sep_list: {neg_sep_list}")
            print(f"col_names: {col_names}")
            print(f"isi_cols_list: {isi_cols_list}")


        make_average_plots(all_df_path=all_df_path,
                           ave_df_path=p_ave_path,
                           error_bars_path=err_path,
                           thr_col='probeLum',
                           ave_over_n=len(run_folder_names),
                           n_trimmed=trim_n,
                           error_type='SE',
                           exp_ave=participant_name,
                           split_1probe=False,
                           isi_name_list=isi_cols_list,
                           sep_vals_list=neg_sep_list,
                           sep_name_list=plot_sep_name_list,
                           # heatmap_annot_fmt='{:.2f}',
                           heatmap_annot_fmt='{:.0f}',
                           show_plots=True, verbose=True)


# # todo: not tested this code yet.
# print(f'\nexp_path: {exp_path}')
# participant_list = ['Nick_test', 'Nick_120Hz']
# print('\nget exp_average_data')
# trim_n = [None, 2]
# e_average_exp_data(exp_path=exp_path, p_names_list=participant_list,
#                    exp_type='radial',
#                    error_type='SE', n_trimmed=trim_n, verbose=True)
#
#
# all_df_path = os.path.join(exp_path, "MASTER_exp_all_thr.csv")
# exp_ave_path = os.path.join(exp_path, "MASTER_exp_ave_thr.csv")
# err_path = os.path.join(exp_path, "MASTER_ave_thr_error_SE.csv")
#
# all_df = pd.read_csv(all_df_path)
# cond_type_list = all_df['cond_type'].unique().tolist()
#
#
# make_average_plots(all_df_path=all_df_path,
#                    ave_df_path=exp_ave_path,
#                    error_bars_path=err_path,
#                    thr_col='probeLum',
#                    stair_names_col='neg_sep',
#                    cond_type_col='cond_type',
#                    cond_type_order=cond_type_list,
#                    n_trimmed=trim_n,
#                    ave_over_n=len(participant_list),
#                    exp_ave=True,
#                    show_plots=True, verbose=True)

print('\nExp1_radial_analysis finished')
