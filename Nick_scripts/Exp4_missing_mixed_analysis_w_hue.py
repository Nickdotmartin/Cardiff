import os
import pandas as pd
import numpy as np
from psignifit_tools import get_psignifit_threshold_df, get_psig_thr_w_hue
from rad_flow_psignifit_analysis import a_data_extraction, b3_plot_staircase
from rad_flow_psignifit_analysis import c_plots, d_average_participant
from rad_flow_psignifit_analysis import make_average_plots, e_average_exp_data
from python_tools import switch_path
from exp1a_psignifit_analysis import a_data_extraction_sep
from exp1a_psignifit_analysis import plt_heatmap_row_col


# exp_path = r"C:\Users\sapnm4\PycharmProjects\Cardiff\Nick_scripts\EXP1b_split_probes\split_v_orig"
# exp_path = r"C:\Users\sapnm4\PycharmProjects\Cardiff\Nick_scripts\EXP1_radial\exp_v_cont"
# exp_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\Exp1_Jan23_radial"
exp_path = r"C:\Users\sapnm4\PycharmProjects\Cardiff\Nick_scripts\Exp4_missing_probe_23_mixed"
exp_path = os.path.normpath(exp_path)


participant_list = ['Nick_test']  # 'Nick'
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
        save_path = os.path.join(root_path, run_dir)

        # don't delete this (p_name = participant_name),
        # needed to ensure names go name1, name2, name3 not name1, name12, name123
        p_name = participant_name

        '''a'''
        # todo: copy look for sep_dirs from other analysis script.

        # # # I don't need data extraction as all ISIs are in same df.
        # p_name = f'{participant_name}_output'  # use this one
        # '''check for unique sep folders.  If found: collate those; else look for output file.'''
        # sep_dirs = [d for d in os.listdir(save_path) if os.path.isdir(os.path.join(save_path, d))]
        # sep_dirs = [d for d in sep_dirs if 'sep_' in d]
        #
        # if len(sep_dirs) > 0:
        #     print(f"sep_dirs: {sep_dirs}")
        #     run_data_df = a_data_extraction_sep(participant_name=participant_name,
        #                                         run_dir=save_path, sep_dirs=sep_dirs,
        #                                         save_all_data=True, verbose=True)
        # else:
        #     print("No sep dirs found, looking for output file")
        #
        #     # # I don't need data extraction as all ISIs are in same df.
        #     try:
        #         run_data_df = pd.read_csv(os.path.join(save_path, f'{p_name}.csv'))
        #     except:
        #         p_name = f'{participant_name}_{r_idx_plus}_output'  # use this one
        #         run_data_df = pd.read_csv(os.path.join(save_path, f'{p_name}.csv'))
        #
        # # run_data_df = pd.read_csv(os.path.join(save_path, f'{p_name}.csv'))
        # run_data_df = run_data_df.sort_values(by=['stair', 'trial_number'])
        # print(f"run_data_df: {run_data_df.columns.to_list()}\n{run_data_df}")
        #
        # # remove unnamed columns
        # substring = 'Unnamed: '
        # unnamed_cols = [i for i in run_data_df.columns.to_list() if substring in i]
        # print(f"unnamed_cols: {unnamed_cols}")
        # for col_name in unnamed_cols:
        #     run_data_df.drop(col_name, axis=1, inplace=True)
        #
        # print(f"run_data_df: {run_data_df.columns.to_list()}\n{run_data_df}")
        # run_data_df.to_excel(os.path.join(save_path, 'RUNDATA-sorted.xlsx'), index=False)
        #
        # # # # get cond details for this exp
        # sep_list = sorted(list(run_data_df['separation'].unique()))
        # print(f'sep_list: {sep_list}')
        #
        # isi_list = sorted(list(run_data_df['ISI'].unique()))
        # # make sure concurrent (-1) is the first value, not zero
        # if -1 in isi_list:
        #     if isi_list[0] != -1:
        #         isi_list.remove(-1)
        #         isi_list = [-1] + isi_list
        # print(f"isi_list: {isi_list}")
        #
        # # # neg sep and cond type need to align so -sep==missing and sep==exp 1, so sort accordingly
        # neg_sep_list = sorted(run_data_df['neg_sep'].unique().tolist())
        # cond_type_list = sorted(run_data_df['cond_type'].unique().tolist(), reverse=True)
        #
        # # if there are multiple sep conds, then cond type list needs to be repeated
        # cond_type_list = list(np.repeat(cond_type_list, len(neg_sep_list) / 2))
        #
        # print(f'neg_sep_list: {neg_sep_list}')
        # print(f'cond_type_list: {cond_type_list}')
        #
        # cols_to_add_dict = {'neg_sep': neg_sep_list}
        #
        # # todo: does neg_sep align with sep and cond_type?
        #
        # thr_df = get_psig_thr_w_hue(root_path=root_path,
        #                             p_run_name=run_dir,
        #                             output_df=run_data_df,
        #                             n_bins=9, q_bins=True,
        #                             thr_col='probeLum',
        #                             sep_col='separation', sep_list=sep_list,
        #                             isi_col='ISI', isi_list=isi_list,
        #                             hue_col='cond_type', hue_list=cond_type_list,
        #                             trial_correct_col='trial_response',
        #                             conf_int=True,
        #                             thr_type='Bayes',
        #                             plot_both_curves=False,
        #                             cols_to_add_dict=cols_to_add_dict,
        #                             show_plots=False,
        #                             verbose=verbose)
        # print(f'thr_df:\n{thr_df}')

    # run_folder_names = ['Nick_1', 'Nick_5', 'Nick_6']
    '''d participant averages'''
    # todo: work out how to put separation and cond type back in.
    d_average_participant(root_path=root_path, run_dir_names_list=run_folder_names,
                          # groupby_col=['cond_type', 'neg_sep'], cols_to_drop='stack', cols_to_replace='separation',
                          # groupby_col=['neg_sep'], cols_to_drop='stack', cols_to_replace=['separation', 'cond_type'],
                          groupby_col=['neg_sep'], cols_to_drop=['stack', 'separation', 'cond_type'], cols_to_replace=None,
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

    cond_type_list = ['missing', 'exp1']

    make_average_plots(all_df_path=all_df_path,
                       ave_df_path=p_ave_path,
                       error_bars_path=err_path,
                       thr_col='probeLum',
                       stair_names_col='neg_sep',
                       cond_type_col='cond_type',
                       cond_type_order=cond_type_list,
                       n_trimmed=trim_n,
                       ave_over_n=len(run_folder_names),
                       exp_ave=participant_name,
                       show_plots=True, verbose=True)



print(f'exp_path: {exp_path}')
# participant_list = ['Nick', 'Simon']
print('\nget exp_average_data')
# trim_n = None
e_average_exp_data(exp_path=exp_path, p_names_list=participant_list,
                   exp_type='missing_mixed',
                   error_type='SE', n_trimmed=trim_n, verbose=True)


all_df_path = os.path.join(exp_path, "MASTER_exp_thr.csv")
exp_ave_path = os.path.join(exp_path, "MASTER_exp_ave_thr.csv")
err_path = os.path.join(exp_path, "MASTER_ave_thr_error_SE.csv")
cond_type_list = ['missing', 'exp1']

make_average_plots(all_df_path=all_df_path,
                   ave_df_path=exp_ave_path,
                   error_bars_path=err_path,
                   thr_col='probeLum',
                   stair_names_col='neg_sep',
                   cond_type_col='cond_type',
                   cond_type_order=cond_type_list,
                   n_trimmed=trim_n,
                   ave_over_n=len(participant_list),
                   exp_ave=True,
                   show_plots=True, verbose=True)

print('\nExp_4_missing_mixed_analysis_w_hue finished')
