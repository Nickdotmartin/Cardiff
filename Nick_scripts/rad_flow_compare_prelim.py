import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from rad_flow_psignifit_analysis import fig_colours, get_n_rows_n_cols


# This function will allow me to choose conditions relating to bg_motion of 70, 200 and 350ms conditions and plot them side_by_side.

# first get the paths to the respective dirs

p_name = 'Nick'
dir_70_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\rad_flow_UNCALIBRATED_MON\rad_flow_2\Nick"
dir_200_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\rad_flow_UNCALIBRATED_MON\rad_flow_23\Nick_240Hz_uncal_bg200"
dir_350_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\rad_flow_UNCALIBRATED_MON\rad_flow_2_350\Nick_350"

save_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\rad_flow_UNCALIBRATED_MON\compare_prelim"
save_path = os.path.join(save_path, p_name)
if not os.path.exists(save_path):
    os.makedirs(save_path)

ave_df_name = 'MASTER_ave_TM2_thresh.csv'

# load the ave_df for each condition and add prelim column
ave_70_df = pd.read_csv(os.path.join(dir_70_path, ave_df_name))
ave_70_df.insert(0, 'prelim', '70ms')
ave_200_df = pd.read_csv(os.path.join(dir_200_path, ave_df_name))
ave_200_df.insert(0, 'prelim', '200ms')
ave_350_df = pd.read_csv(os.path.join(dir_350_path, ave_df_name))
ave_350_df.insert(0, 'prelim', '350ms')


# get the ISI values for each condition (list of columns that contain the substring 'ISI_')
ave_70_isi_list = [col for col in ave_70_df.columns if 'ISI_' in col]
ave_200_isi_list = [col for col in ave_200_df.columns if 'ISI_' in col]
ave_350_isi_list = [col for col in ave_350_df.columns if 'ISI_' in col]

# make a master df by concatenating the lists
master_ave_df = pd.concat([ave_70_df, ave_200_df, ave_350_df], ignore_index=True)

# drop stair_names column as it is redundant
master_ave_df.drop(columns=['stair_names'], inplace=True)

# insert cond_type column with 'Congruent' if neg_sep >= 0 and 'Incongruent' if neg_sep < 0
cond_type_list = ['Incongruent' if neg_sep < 0 else 'Congruent' for neg_sep in master_ave_df['neg_sep']]
master_ave_df.insert(1, 'cond_type', cond_type_list)
print(f"\nmaster_ave_df:\n{master_ave_df}")

# change separation column to int
master_ave_df['separation'] = master_ave_df['separation'].astype(int)

# save master_ave_df
master_ave_df.to_csv(os.path.join(save_path, f'master_prelim_ave_{p_name}_df.csv'), index=False)


# make a list of all ISIs conds (all_isi_list), sorted with the most frequent first
all_isi_list = ave_70_isi_list + ave_200_isi_list + ave_350_isi_list
isi_count_dict = {isi: all_isi_list.count(isi) for isi in all_isi_list}
sorted_isi_count_dict = {k: v for k, v in sorted(isi_count_dict.items(), key=lambda item: item[1], reverse=True)}
all_isi_list = list(sorted_isi_count_dict.keys())
print(f"\nall_isi_list:\n{all_isi_list}")


# make my_colur_dict so that prelims are always in same colour even if some conditions are missing
# use different colours for plots by prelim or by cond_type
my_prelim_colours = fig_colours(n_conditions=len(master_ave_df['prelim'].unique().tolist()))
my_prelim_col_dict = {prelim: my_prelim_colours[i] for i, prelim in enumerate(master_ave_df['prelim'].unique().tolist())}
my_cond_colours = fig_colours(n_conditions=len(master_ave_df['cond_type'].unique().tolist()), alternative_colours=True)
my_cond_col_dict = {cond: my_cond_colours[i] for i, cond in enumerate(master_ave_df['cond_type'].unique().tolist())}
my_colour_dict = {**my_prelim_col_dict, **my_cond_col_dict}
print(f"\nmy_colour_dict:\n{my_colour_dict}")


for this_isi in all_isi_list:


    this_isi_df = master_ave_df[['prelim', 'cond_type', 'neg_sep', 'separation', this_isi]].copy()

    # drop rows where this_isi is NaN
    this_isi_df.dropna(subset=[this_isi], inplace=True)

    print(f"\n\nthis_isi_df: ({this_isi}):\n{this_isi_df}")

    # sort prelim_list (strip 'ms', convert to ints, sort then put them back to str with ms)
    prelim_list = list(set(this_isi_df['prelim'].tolist()))
    prelim_list = sorted([int(prelim.strip('ms')) for prelim in prelim_list])
    prelim_list = [str(prelim) + 'ms' for prelim in prelim_list]
    print(f"\nprelim_list (n={len(prelim_list)}): {prelim_list}")


    '''Batman plots - on same panel, then one with different panels per prelim'''
    # 1. all patman plots on same panel
    # get the x_tick_labels and x_tick_values (neg_sep, -18 to +18)
    x_tick_labels = this_isi_df['neg_sep'].unique().tolist()
    x_tick_labels = sorted([float(i) for i in x_tick_labels])
    x_tick_labels = [str(i) for i in x_tick_labels]
    x_tick_values = list(range(len(x_tick_labels)))
    print(f"\nx_tick_labels:\n{x_tick_labels}")
    print(f"\nx_tick_values:\n{x_tick_values}")

    # make copy of this_isi_df and add new column for x_tick_values, mapped onto neg_sep values
    batman_plot_df = this_isi_df.copy()
    tick_dict = dict(zip(x_tick_labels, x_tick_values))
    print(f"\ntick_dict:\n{tick_dict}")
    tick_vals_col_list = [tick_dict[str(neg_sep)] for neg_sep in batman_plot_df['neg_sep']]
    print(f"\ntick_vals_col_list:\n{tick_vals_col_list}")
    batman_plot_df.insert(1, 'tick_vals', tick_vals_col_list)
    print(f"\nbatman_plot_df:\n{batman_plot_df}")

    # sort final values for x_tick_labels
    x_tick_labels = [str('-0') if i == .01 else int(float(i)) for i in x_tick_labels]

    fig, ax = plt.subplots()
    sns.lineplot(x='tick_vals', y=this_isi, hue='prelim', data=batman_plot_df, palette=my_prelim_col_dict)
    ax.set_xticks(x_tick_values)
    ax.set_xticklabels(x_tick_labels)
    plt.axvline(x=5.5, color='lightgrey', linestyle='dashed')
    plt.title(f"{p_name}.  {this_isi} prelim motion with neg_sep")
    plt.savefig(os.path.join(save_path, f"{p_name}_{this_isi}_neg_sep.png"))
    plt.show()
    plt.close()


    # 2. batman plots in separate panels if there is more than 1 prelim
    if len(prelim_list) > 1:

        fig, axes = plt.subplots(nrows=1, ncols=len(prelim_list),
                                 figsize=(len(prelim_list)*5, 5))
        ax_counter = 0

        for row_idx, ax in enumerate(axes):
            sns.lineplot(x='tick_vals', y=this_isi, hue='cond_type',
                         data=batman_plot_df[batman_plot_df['prelim'] == prelim_list[row_idx]],
                         ax=axes[row_idx], palette=my_cond_col_dict)
            ax.set_xticks(x_tick_values)
            ax.set_xticklabels(x_tick_labels)
            ax.axvline(x=5.5, color='lightgrey', linestyle='dashed')
            ax.set_title(f"{prelim_list[row_idx]}")
            ax.set_xlabel('Separation')
            ax.set_ylabel('Threshold')

            # suppress legend for individal panels, but put one in at the end
            if ax_counter == 0:
                ax.legend(loc='upper left')
            else:
                ax.get_legend().remove()
            ax_counter += 1

        plt.suptitle(f"{p_name}.  {this_isi} prelim motion with neg_sep panels")
        plt.savefig(os.path.join(save_path, f"{p_name}_{this_isi}_neg_sep_panel.png"))
        plt.show()
        plt.close()




    '''different between conditions plots'''
    # plots showing the difference between congruent and incongruent conditions for each prelim, ISI and neg_sep
    # split into two dataframes, one for congruent and one for incongruent
    print(f"\nthis_isi_df:\n{this_isi_df}")

    this_isi_df = this_isi_df[['prelim', 'cond_type', 'separation', this_isi]].copy()
    print(f"\nthis_isi_df:\n{this_isi_df}")
    congruent_df = this_isi_df[this_isi_df['cond_type'] == 'Congruent'].copy()
    incongruent_df = this_isi_df[this_isi_df['cond_type'] == 'Incongruent'].copy()

    # rename this_isi column to f'{this_isi}_cong' and f'{this_isi}_incong' for congruent_df and incongruent_df
    congruent_df.rename(columns={this_isi: f'{this_isi}_cong'}, inplace=True)
    incongruent_df.rename(columns={this_isi: f'{this_isi}_incong'}, inplace=True)

    # drop cond_type column from both dataframes
    congruent_df.drop(columns=['cond_type'], inplace=True)
    incongruent_df.drop(columns=['cond_type'], inplace=True)

    # set index to prelim and separation
    congruent_df.set_index(['prelim', 'separation'], inplace=True)
    incongruent_df.set_index(['prelim', 'separation'], inplace=True)

    print(f"\ncongruent_df:\n{congruent_df}")
    print(f"\nincongruent_df:\n{incongruent_df}")

    # join congruent_df and incongruent_df
    diff_df = congruent_df.join(incongruent_df)
    print(f"\ndiff_df:\n{diff_df}")

    # add column for difference between congruent and incongruent
    diff_df[f'{this_isi}_diff'] = diff_df[f'{this_isi}_cong'] - diff_df[f'{this_isi}_incong']

    # reset index
    diff_df.reset_index(inplace=True)

    # add x tick vals col
    # change separation values to ints
    diff_df['separation'] = diff_df['separation'].astype(int)
    x_tick_labels = sorted(diff_df['separation'].unique())
    x_tick_values = list(range(len(x_tick_labels)))
    tick_dict = dict(zip(x_tick_labels, x_tick_values))
    tick_vals_col_list = [tick_dict[sep] for sep in diff_df['separation']]
    diff_df.insert(1, 'tick_vals', tick_vals_col_list)
    print(f"\ndiff_df:\n{diff_df}")


    # plot difference between congruent and incongruent conditions
    sns.lineplot(x='tick_vals', y=f'{this_isi}_diff', hue='prelim', data=diff_df,
                 palette=my_prelim_col_dict)

    # add horizontal line at 0
    plt.axhline(y=0, color='lightgrey', linestyle='dashed')
    plt.xticks(x_tick_values, x_tick_labels)
    plt.title(f"{p_name}.  {this_isi} congruent - incongruent difference")
    plt.xlabel('Separation')
    plt.savefig(os.path.join(save_path, f"{p_name}_{this_isi}_diff.png"))
    plt.show()
    plt.close()
