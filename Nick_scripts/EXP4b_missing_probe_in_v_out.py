import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from exp1a_psignifit_analysis import make_long_df
from psignifit_tools import get_psig_thr_w_hue
from python_tools import running_on_laptop, switch_path


# get exp path
project_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff"
this_exp = r"Exp4b_missing_probe\radial"
exp_path = os.path.join(project_path, this_exp)
convert_path1 = os.path.normpath(exp_path)
if running_on_laptop():
    convert_path1 = switch_path(convert_path1, 'mac_oneDrive')
exp_path = convert_path1
print(f"exp_path: {exp_path}")

# other exp details
participant_list = ['Nick_sep4_strip']  #, 'Nick', 'Simon']  # , 'Simon']  # 'Simon', 'Nick'
split_1probe = False
n_runs = 12
analyse_from_run = 1

exp_psig_list = []

for p_idx, participant_name in enumerate(participant_list):
    root_path = os.path.join(exp_path, participant_name)

    # # search to automatically get run_folder_names
    dir_list = os.listdir(root_path)
    run_folder_names = []
    for i in range(n_runs):  # numbers 0 to 11
        check_dir = f'{participant_name}_{i+analyse_from_run}'
        if check_dir in dir_list:
            run_folder_names.append(check_dir)

    p_output_files = []

    for run_idx, run_dir in enumerate(run_folder_names):

        r_idx_plus = run_idx + analyse_from_run
        print(f'\nrunning analysis for {participant_name}, {run_dir}, {participant_name}{r_idx_plus}\n')


        run_save_path = os.path.join(root_path, run_dir)

        # don't delete this (participant_name = participant_name),
        # needed to ensure names go name1, name2, name3 not name1, name12, name123
        p_name = participant_name
        p_name = f'{participant_name}_output'  # use this one

        # I don't need data extraction as all ISIs are in same df.
        try:
            run_data_df = pd.read_csv(os.path.join(run_save_path, f'{p_name}.csv'))
        except:
            p_name = f'{participant_name}_{r_idx_plus}_output'  # use this one
            run_data_df = pd.read_csv(os.path.join(run_save_path, f'{p_name}.csv'))

        try:
            run_data_df = run_data_df.sort_values(by=['stair', 'total_nTrials'])
        except KeyError:
            run_data_df = run_data_df.sort_values(by=['stair', 'trial_number'])


        # remove unnamed columns
        substring = 'Unnamed: '
        unnamed_cols = [i for i in run_data_df.columns.to_list() if substring in i]
        print(f"dropping unnamed_cols: {unnamed_cols}")
        for col_name in unnamed_cols:
            run_data_df.drop(col_name, axis=1, inplace=True)

        if 'example_name ' in run_data_df.columns.to_list():
            run_data_df.rename(columns={'example_name ': 'radial_dir'}, inplace=True)

        # append this run details to list
        p_output_files.append(run_data_df)


    # join all stacks (run/group) data and save as master csv
    p_outputs_df = pd.concat(p_output_files, ignore_index=True)
    # all_psig_rad_dirs_df.to_csv(f'{root_path}{os.sep}MASTER_psig_rad_dirs.csv', index=False)
    print(f'\np_outputs_df ({p_outputs_df.shape}:\n{p_outputs_df}')

    sep_list = list(p_outputs_df['separation'].unique())
    isi_list = list(p_outputs_df['ISI'].unique())
    rad_dir_list = list(p_outputs_df['radial_dir'].unique())
    print(f"sep_list: {sep_list}")
    print(f"isi_list: {isi_list}")
    print(f"rad_dir_list: {rad_dir_list}")

    psig_rad_dirs_name = 'psig_thr_rad_dirs'

    '''get psignifit thresholds df with example_name (rad dir) as hue'''
    thr_df = get_psig_thr_w_hue(root_path=exp_path, p_run_name=participant_name,
                                output_df=p_outputs_df,
                                n_bins=9, q_bins=True,
                                thr_col='probeLum',
                                sep_col='separation', sep_list=sep_list,
                                isi_col='ISI', isi_list=isi_list,
                                hue_col='radial_dir', hue_list=rad_dir_list,
                                trial_correct_col='trial_response',
                                conf_int=True, thr_type='Bayes',
                                plot_both_curves=False,
                                cols_to_add_dict=None, save_name=psig_rad_dirs_name,
                                show_plots=False, save_plots=False,
                                verbose=True)
    print(f'thr_df:\n{thr_df}')


    # make plots for this participant
    psig_rad_dirs_df = thr_df
    print(f'psig_rad_dirs_df:\n{psig_rad_dirs_df}')

    # make participant plot to show radial in and out conditions
    long_psig_rad_dirs_df = make_long_df(psig_rad_dirs_df, wide_stubnames='ISI',
                                         thr_col='probeLum',
                                         col_to_keep='radial_dir', idx_col='separation',
                                         verbose=True)
    long_psig_rad_dirs_df.reset_index(inplace=True)
    print(f'long_psig_rad_dirs_df:\n{long_psig_rad_dirs_df}')

    # make plot to show radial in and out conditions
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=long_psig_rad_dirs_df,
                 x='separation', y="probeLum",
                 hue='ISI',
                 style='radial_dir',
                 markers=True, dashes=True, ax=ax,)
    fig_title = f'{participant_name} radial In and Out'
    plt.title(fig_title)
    save_as = os.path.join(root_path, 'radial_In_v_Out.png')
    plt.savefig(save_as)
    plt.show()

    # make plot to show radial in and out conditions
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=long_psig_rad_dirs_df,
                 x='ISI', y="probeLum",
                 hue='separation',
                 style='radial_dir',
                 markers=True, dashes=True, ax=ax,)
    fig_title = f'{participant_name} radial In and Out sepx'
    plt.title(fig_title)
    save_as = os.path.join(root_path, 'radial_In_v_Out_transpose.png')
    plt.savefig(save_as)
    plt.show()

    # # make participant plot to compare radial in and out conditions
    psig_rad_dirs_df = psig_rad_dirs_df.sort_values(by='radial_dir')
    sep_list = list(psig_rad_dirs_df['separation'].unique())

    print(f'psig_rad_dirs_df:\n{psig_rad_dirs_df}')
    drop_hue_df = psig_rad_dirs_df.drop('radial_dir', axis=1)
    print(f'drop_hue_df:\n{drop_hue_df}')
    rad_dirs_diff_df = drop_hue_df.diff(periods=len(sep_list))
    rad_dirs_diff_df = rad_dirs_diff_df.iloc[len(sep_list):, :]
    rad_dirs_diff_df['separation'] = sep_list
    rad_dirs_diff_df.set_index('separation', inplace=True)
    if 'ISI 999' in list(rad_dirs_diff_df.columns):
        print('fixing 999 inputs')
        rad_dirs_diff_df.rename(columns={'ISI 999': 'ISI -1', 'ISI 3': 'ISI_3', 'ISI 6': 'ISI_6'}, inplace=True)

    print(f'rad_dirs_diff_df:\n{rad_dirs_diff_df}')

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=rad_dirs_diff_df, ax=ax, markers=True)
    min_diff = rad_dirs_diff_df.min().min()

    max_diff = rad_dirs_diff_df.max().max()
    if min_diff < 0:
        if max_diff > 0:
            # specifying horizontal line type
            plt.axhline(y=0, color='grey', linestyle='-')

    fig_title = f'{participant_name} radial In and Out difference'
    plt.suptitle(fig_title)
    fig_description = '+ive = out higher thr; -ive = in higher thr'
    plt.title(fig_description)
    save_as = os.path.join(root_path, 'radial_In_v_Out_diff.png')
    plt.savefig(save_as)
    plt.show()


    # append participant means to exp_psig_list
    rows, cols = thr_df.shape
    thr_df.insert(0, 'participant', [participant_name] * rows)
    exp_psig_list.append(thr_df)


# join all stacks (run/group) data and save as master csv
exp_outputs_df = pd.concat(exp_psig_list, ignore_index=True)
exp_outputs_df.to_csv(f'{exp_path}{os.sep}MASTER_psig_rad_dirs.csv', index=False)
if 'ISI 999' in list(exp_outputs_df.columns):
    print('fixing 999 inputs')
    exp_outputs_df = exp_outputs_df.rename(columns={'ISI 999': 'ISI -1', 'ISI 3': 'ISI_3', 'ISI 6': 'ISI_6'})
print(f'\nexp_outputs_df ({exp_outputs_df.shape}:\n{exp_outputs_df}')

groupby_sep_df = exp_outputs_df.drop('participant', axis=1)
exp_ave_thr_df = groupby_sep_df.groupby(['separation', 'radial_dir'], sort=True).mean()
exp_ave_thr_df.reset_index(inplace=True)
print(f'\nexp_ave_thr_df:\n{exp_ave_thr_df}')

if list(exp_ave_thr_df.columns)[0] == 'separation':
    col_names = list(exp_ave_thr_df.columns)
    psig_rad_dirs_df = exp_ave_thr_df[['separation', 'radial_dir'] + col_names[2:]]
else:
    psig_rad_dirs_df = exp_ave_thr_df
# psig_rad_dirs_df.to_csv(f'{exp_path}{os.sep}Exp_ave_psig_rad_dirs.csv', index=False)
print(f'psig_rad_dirs_df:\n{psig_rad_dirs_df}')

# make experiment plot to show radial in and out conditions
long_psig_rad_dirs_df = make_long_df(psig_rad_dirs_df, wide_stubnames='ISI',
                                     thr_col='probeLum',
                                     col_to_keep='radial_dir', idx_col='separation',
                                     verbose=True)
long_psig_rad_dirs_df.reset_index(inplace=True)
if 'ISI 999' in list(long_psig_rad_dirs_df.columns):
    print('fixing 999 inputs')
    long_psig_rad_dirs_df = long_psig_rad_dirs_df.rename(columns={'ISI 999': 'ISI -1', 'ISI 3': 'ISI_3', 'ISI 6': 'ISI_6'})
print(f'long_psig_rad_dirs_df:\n{long_psig_rad_dirs_df}')

# make plot to show radial in and out conditions

fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(data=long_psig_rad_dirs_df,
             x='separation', y="probeLum",
             hue='ISI',
             style='radial_dir',
             markers=True, dashes=True, ax=ax,
             )
fig_title = f'Radial In and Out'
plt.title(fig_title)
save_as = os.path.join(exp_path, 'radial_In_v_Out.png')
plt.savefig(save_as)
plt.show()

# # make experiment plot to compare radial in and out conditions
psig_rad_dirs_df = psig_rad_dirs_df.sort_values(by='radial_dir')
sep_list = list(psig_rad_dirs_df['separation'].unique())

print(f'psig_rad_dirs_df:\n{psig_rad_dirs_df}')
drop_hue_df = psig_rad_dirs_df.drop('radial_dir', axis=1)
print(f'drop_hue_df:\n{drop_hue_df}')
rad_dirs_diff_df = drop_hue_df.diff(periods=len(sep_list))
rad_dirs_diff_df = rad_dirs_diff_df.iloc[len(sep_list):, :]
rad_dirs_diff_df['separation'] = sep_list
rad_dirs_diff_df.set_index('separation', inplace=True)
print(f'rad_dirs_diff_df:\n{rad_dirs_diff_df}')

fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(data=rad_dirs_diff_df, ax=ax, markers=True)
min_diff = rad_dirs_diff_df.min().min()
max_diff = rad_dirs_diff_df.max().max()
if min_diff < 0:
    if max_diff > 0:
        plt.axhline(y=0, color='grey', linestyle='-')
fig_title = f'Radial In and Out difference'
plt.suptitle(fig_title)
fig_description = '+ive = out higher thr; -ive = in higher thr'
plt.title(fig_description)
save_as = os.path.join(exp_path, 'radial_In_v_Out_diff.png')
plt.savefig(save_as)
plt.show()

print('\nExp4b_missing_probe_in_v_out finished\n')
