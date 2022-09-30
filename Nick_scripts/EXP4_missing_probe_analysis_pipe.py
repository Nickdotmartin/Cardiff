import os
import pandas as pd
import numpy as np
from psignifit_tools import get_psignifit_threshold_df, get_psig_thr_w_hue
from rad_flow_psignifit_analysis import a_data_extraction, b3_plot_staircase
from rad_flow_psignifit_analysis import c_plots, d_average_participant
from rad_flow_psignifit_analysis import make_average_plots, e_average_exp_data

# from exp1a_psignifit_analysis import a_data_extraction, b3_plot_staircase, c_plots, \
#     d_average_participant, e_average_exp_data, make_average_plots
from exp1a_psignifit_analysis import plt_heatmap_row_col


exp_path = r'C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\EXP4_missing_probe'
exp_path = os.path.normpath(exp_path)

coherence_type = 'Rotation'  # 'Radial', Translation
exp_path = os.path.join(exp_path, coherence_type)

participant_list = ['Simon']  # , 'Nick']  # , 'Nick_half_speed']

verbose = True
show_plots = True

n_runs = 1  # 12
# if the first folder to analyse is 1, p_idx_plus = 1.  If the forst folder is 5, use 5 etc.
p_idx_plus = 1

for p_idx, participant_name in enumerate(participant_list):

    root_path = os.path.join(exp_path, participant_name)

    # # search to automatically get run_folder_names
    dir_list = os.listdir(root_path)
    run_folder_names = []
    for i in range(12):  # numbers 0 to 11
        check_dir = f'{participant_name}_{i+p_idx_plus}'   # numbers 1 to 12
        if check_dir in dir_list:
            run_folder_names.append(check_dir)

    print(f'run_folder_names: {run_folder_names}')

    # for run_idx, run_dir in enumerate(run_folder_names):
    #
    #     # add run number , e.g., add five to access Nick_5 on the zeroth iteration
    #     r_idx_plus = run_idx + p_idx_plus
    #
    #     print(f'\nrun_idx {run_idx+1}: running analysis for '
    #           f'{participant_name}, {run_dir}, {participant_name}_{r_idx_plus}')
    #     save_path = os.path.join(root_path, run_dir)
    #
    #     # don't delete this (p_name = participant_name),
    #     # needed to ensure names go name1, name2, name3 not name1, name12, name123
    #     p_name = participant_name
    #
    #     '''a'''
    #     # # I don't need data extraction as all ISIs are in same df.
    #     p_name = f'{participant_name}_output'  # use this one
    #     try:
    #         run_data_df = pd.read_csv(os.path.join(save_path, f'{p_name}.csv'))
    #     except:
    #         p_name = f'{participant_name}_{r_idx_plus}_output'
    #         run_data_df = pd.read_csv(os.path.join(save_path, f'{p_name}.csv'))
    #
    #     run_data_df = pd.read_csv(os.path.join(save_path, f'{p_name}.csv'))
    #     run_data_df = run_data_df.sort_values(by=['stair', 'trial_number'])
    #     print(f"run_data_df: {run_data_df.columns.to_list()}\n{run_data_df}")
    #
    #     # # # get cond details for this exp
    #     separation_list = run_data_df['separation'].unique().tolist()
    #     isi_list = run_data_df['ISI'].unique().tolist()
    #     coherence_list = run_data_df['probes_type'].unique().tolist()
    #     print(f'separation_list: {separation_list}')
    #     print(f'isi_list: {isi_list}')
    #     print(f'coherence_list: {coherence_list}')
    #
    #     thr_df = get_psig_thr_w_hue(root_path=root_path,
    #                                 p_run_name=run_dir,
    #                                 output_df=run_data_df,
    #                                 n_bins=9, q_bins=True,
    #                                 thr_col='probeLum',
    #                                 sep_col='separation', sep_list=separation_list,
    #                                 isi_col='ISI', isi_list=isi_list,
    #                                 hue_col='probes_type', hue_list=coherence_list,
    #                                 # stair_col="stair",
    #                                 trial_correct_col='trial_response',
    #                                 conf_int=True,
    #                                 thr_type='Bayes',
    #                                 plot_both_curves=False,
    #                                 # cols_to_add_dict=cols_to_add_dict,
    #                                 show_plots=False,
    #                                 verbose=verbose)
    #     print(f'thr_df:\n{thr_df}')
    #
    #     # '''b3'''
    #     # b3_plot_staircase(run_data_path, thr_col='probeLum', show_plots=show_plots, verbose=verbose)
    #     #
    #     # '''c I don't actually need any of these, instead sort get psignifit thr ands make plots from those.'''
    #     # c_plots(save_path=save_path, thr_col='probeLum', isi_name_list=run_isi_names_list, show_plots=show_plots, verbose=verbose)


    '''d participant averages'''
    trim_n = None
    if len(run_folder_names) == 12:
        trim_n = 2
    print(f'\ntrim_n: {trim_n}')

    d_average_participant(root_path=root_path, run_dir_names_list=run_folder_names,
                          groupby_col=['cond_type', 'probes_type'], cols_to_drop='stack', cols_to_replace='separation',
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

    # make separation values negative
    all_df = pd.read_csv(all_df_path)
    if 'stair_names' not in list(all_df.columns):
        ave_df = pd.read_csv(p_ave_path)
        err_df = pd.read_csv(err_path)
        print(f'\nall_df:\n{all_df}')
        print(f'\nave_df:\n{ave_df}')
        print(f'\nerr_df:\n{err_df}')

        # sort neg sep for all_df
        all_df_neg_sep_options = [
            (all_df['probes_type'] == 'rotation'),
            (all_df['probes_type'] == 'radial'),
            (all_df['probes_type'] == 'translation'),
            (all_df['probes_type'] == 'incoherent'),
            (all_df['probes_type'] == 'incoherent') & (all_df['separation'] == 0)]
        values = [all_df['separation'], all_df['separation'], all_df['separation'], -all_df['separation'], -.1]
        all_df.insert(3, 'stair_names', np.select(all_df_neg_sep_options, values))
        print(f'\nall_df:\n{all_df}')
        # all_df.to_csv(all_df_path)

        # sort neg sep for ave_df
        ave_df_neg_sep_options = [
            (ave_df['probes_type'] == 'rotation'),
            (ave_df['probes_type'] == 'radial'),
            (ave_df['probes_type'] == 'translation'),
            (ave_df['probes_type'] == 'incoherent'),
            (ave_df['probes_type'] == 'incoherent') & (ave_df['separation'] == 0)]
        values = [ave_df['separation'], ave_df['separation'], ave_df['separation'], -ave_df['separation'], -.1]
        ave_df.insert(3, 'stair_names', np.select(ave_df_neg_sep_options, values))
        print(f'\nave_df:\n{ave_df}')
        # ave_df.to_csv(p_ave_path)


        # sort neg sep for err_df
        # err_df_neg_sep_options = [
        #     (err_df['probes_type'] == 'rotation'),
        #     (err_df['probes_type'] == 'radial'),
        #     (err_df['probes_type'] == 'translation'),
        #     (err_df['probes_type'] == 'incoherent'),
        #     (err_df['probes_type'] == 'incoherent') & (err_df['separation'] == 3.00)]
        # values = [err_df['separation'], err_df['separation'], err_df['separation'], -err_df['separation'], -.1999]
        # err_df.insert(3, 'stair_names', np.select(err_df_neg_sep_options, values))
        # print(f'\nerr_df:\n{err_df}')
        # err_df.to_csv(err_path)
        def make_neg_sep(df):
            if (df.probes_type == 'incoherent') and (df.separation == 0.0):
                return -.1
            elif df.probes_type == 'incoherent':
                return 0-df.separation
            else:
                return df.separation


        # err_df['stair_names'] = err_df.apply(make_neg_sep, axis=1)
        err_df.insert(3, 'stair_names', err_df.apply(make_neg_sep, axis=1))
        print(f'\nerr_df:\n{err_df}')



    # make_average_plots(all_df_path=all_df_path,
    #                    ave_df_path=p_ave_path,
    #                    error_bars_path=err_path,
    #                    thr_col='probeLum',
    #                    n_trimmed=trim_n,
    #                    ave_over_n=len(run_folder_names),
    #                    exp_ave=False,
    #                    show_plots=True, verbose=True)


#
# print(f'exp_path: {exp_path}')
# # participant_list = ['Nick_half', 'Simon_half']
# print('\nget exp_average_data')
# trim_n = 2
# # todo: sort script to automatically use trim=2 if its there, and not just use untrimmed
# e_average_exp_data(exp_path=exp_path, p_names_list=participant_list,
#                    error_type='SE', n_trimmed=trim_n, verbose=True)
#
#
# all_df_path = os.path.join(exp_path, "MASTER_exp_thr.csv")
# exp_ave_path = os.path.join(exp_path, "MASTER_exp_ave_thr.csv")
# err_path = os.path.join(exp_path, "MASTER_ave_thr_error_SE.csv")
#
# make_average_plots(all_df_path=all_df_path,
#                    ave_df_path=exp_ave_path,
#                    error_bars_path=err_path,
#                    thr_col='probeLum',
#                    n_trimmed=trim_n,
#                    ave_over_n=len(participant_list),
#                    exp_ave=True,
#                    show_plots=True, verbose=True)

print('\nExp_4_missing_probe_analysis_pipe finished')
