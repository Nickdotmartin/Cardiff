import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

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

ave_70_df = pd.read_csv(os.path.join(dir_70_path, ave_df_name))
ave_70_df.insert(0, 'prelim_mot', '70ms')
ave_200_df = pd.read_csv(os.path.join(dir_200_path, ave_df_name))
ave_200_df.insert(0, 'prelim_mot', '200ms')
ave_350_df = pd.read_csv(os.path.join(dir_350_path, ave_df_name))
ave_350_df.insert(0, 'prelim_mot', '350ms')
# print(f"\nave_70_df:\n{ave_70_df}")
# print(f"\nave_200_df:\n{ave_200_df}")
# print(f"\nave_350_df:\n{ave_350_df}")


# get the ISI values for each condition
# get a list of columns that contain the substring 'ISI_'
ave_70_isi_list = [col for col in ave_70_df.columns if 'ISI_' in col]
ave_200_isi_list = [col for col in ave_200_df.columns if 'ISI_' in col]
ave_350_isi_list = [col for col in ave_350_df.columns if 'ISI_' in col]
print(f"\nave_70_isi_list:\n{ave_70_isi_list}")
print(f"\nave_200_isi_list:\n{ave_200_isi_list}")
print(f"\nave_350_isi_list:\n{ave_350_isi_list}")

# make a master df by concatenating the lists
master_ave_df = pd.concat([ave_70_df, ave_200_df, ave_350_df], ignore_index=True)

# drop stair_names column as it is redundant
master_ave_df.drop(columns=['stair_names'], inplace=True)

# insert cond_type column with 'Congruent' if neg_sep >= 0 and 'Incongruent' if neg_sep < 0
cond_type_list = ['Incongruent' if neg_sep < 0 else 'Congruent' for neg_sep in master_ave_df['neg_sep']]
master_ave_df.insert(1, 'cond_type', cond_type_list)
print(f"\nmaster_ave_df:\n{master_ave_df}")

all_isi_list = list(set(ave_70_isi_list + ave_200_isi_list + ave_350_isi_list))
print(f"\nall_isi_list:\n{all_isi_list}")
for this_isi in set(all_isi_list):

    # first do batman plots to see if the curve has shifted
    sns.lineplot(x='neg_sep', y=this_isi, hue='prelim_mot', data=master_ave_df)
    plt.title(f"{p_name}.  {this_isi} prelim motion with neg_sep")
    plt.savefig(os.path.join(save_path, f"{p_name}_{this_isi}_neg_sep.png"))
    plt.show()

    # next do plot to compare congruent and incongruent conditions
    sns.lineplot(x='separation', y=this_isi, hue='prelim_mot', style='cond_type', data=master_ave_df)
    plt.title(f"{p_name}.  {this_isi} prelim motion with pos_sep and cond_type")
    plt.savefig(os.path.join(save_path, f"{p_name}_{this_isi}_pos_sep_cond_type.png"))
    plt.show()
