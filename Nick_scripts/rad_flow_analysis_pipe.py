import os

from OLD_psignifit_analysis import a_data_extraction, b3_plot_staircase

# # loop through run folders with first 5 scripts (a, b1, b2, b3, c)
# # then run script d to get master lists and averages
root_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/radial_flow_exp'
run_folder_names = ['Nick_3', 'Nick_4', 'Nick_5']  # , 'Nick_6', 'Nick_7', 'Nick_8']
participant_name = 'Nick'
isi_list = [1, 4, 6]
# root_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/radial_flow_exp'
# run_folder_names = ['Nick_3']
# participant_name = 'Nick'


for run_idx, run_dir in enumerate(run_folder_names):

    print(f'\nrunning analysis for {participant_name}, {run_dir}, {participant_name}{run_idx+1}\n')
    save_path = f'{root_path}{os.sep}{run_dir}'

    # don't delete this (participant_name = participant_name),
    # needed to ensure names go name1, name2, name3 not name1, name12, name123
    participant_name = participant_name

    # '''a'''
    participant_name = f'{participant_name}_{run_idx+3}'
    # isi_list = [0, 1, 4, 6, 12, 24]
    a_data_extraction(p_name=participant_name, run_dir=save_path, isi_list=isi_list, verbose=True)
    #
    # all_data_path = f'{save_path}{os.sep}ALL_ISIs_sorted.xlsx'

    # '''b1'''
    # all_data_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/' \
    #                 'radial_flow_exp/Nick_3/ALL_ISIs_sorted.xlsx'
    # b1_extract_last_values(all_data_path=all_data_path)

    '''b3'''
    all_data_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/' \
                    'radial_flow_exp/Nick_3/ALL_ISIs_sorted.xlsx'
    b3_plot_staircase(all_data_path)

    # '''c I don't actually need any of these, instead sort get psignifit thr ands make plots from those.'''
    # c_plots(save_path=save_path)

'''d'''
# d_average_participant(root_path=root_path, run_dir_names_list=run_dir_names_list)

print('rad_flow_analysis_pipe finished')
