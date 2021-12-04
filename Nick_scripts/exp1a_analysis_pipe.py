import os

from MATLAB_analysis import b1_extract_last_values

# # loop through run folders with first 5 scripts (a, b1, b2, b3, c)
# # then run script d to get master lists and averages
# root_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/Kim'
# run_folder_names = ['P6a-Kim', 'P6b-Kim', 'P6c-Kim', 'P6d-Kim', 'P6e-Kim', 'P6f-Kim']
# participant_name = 'Kim'
root_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/Kim/Nick_practice'
run_folder_names = ['P6a-Kim']
participant_name = 'Kim'


for run_idx, run_dir in enumerate(run_folder_names):

    print(f'\nrunning analysis for {participant_name}, {run_dir}, {participant_name}{run_idx+1}\n')
    save_path = f'{root_path}{os.sep}{run_dir}'

    # don't delete this (participant_name = participant_name),
    # needed to ensure names go name1, name2, name3 not name1, name12, name123
    participant_name = participant_name

    # '''a'''
    # participant_name = f'{participant_name}{run_idx+1}'
    # isi_list = [-1, 0, 2, 4, 6, 9, 12, 24]
    # a_data_extraction(p_name=participant_name, run_dir=save_path, isi_list=isi_list, verbose=True)
    #
    # all_data_path = f'{save_path}{os.sep}ALLDATA-sorted.xlsx'

    '''b1'''
    all_data_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/Kim/' \
                    'Nick_practice/P6a-Kim/ALLDATA-sorted.xlsx'
    b1_extract_last_values(all_data_path=all_data_path)

    # '''b2'''
    # # all_data_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/Kim/' \
    # #                 'Nick_practice/P6a-Kim/ALLDATA-sorted.xlsx'
    # b2_last_reversal(all_data_path=all_data_path)
    #
    # '''b3'''
    # # all_data_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/Kim/' \
    # #                     'Nick_practice/P6a-Kim/ALLDATA-sorted.xlsx'
    # b3_plot_staircase(all_data_path)
    #
    # '''c'''
    # c_plots(save_path=save_path)

'''d'''
# # root_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/Kim'  # master folder containing all runs
# run_dir_names_list = ['P6a-Kim', 'P6b-Kim', 'P6c-Kim', 'P6d-Kim', 'P6e-Kim', 'P6f-Kim']
# d_averageParticipant(root_path=root_path, run_dir_names_list=run_dir_names_list)

print('exp1a_analysis_pipe finished')
