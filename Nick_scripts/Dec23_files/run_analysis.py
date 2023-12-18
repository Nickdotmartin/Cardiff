from analysis_pipes_Dec23 import target_detect_analysis_pipe_Dec23, direction_detect_analysis_pipe_Dec23

'''
This script is for running analysis on the experiment data.
Enter the path to the folder containing the data you want to analyse.
Analyse what accepts the following inputs: 'all'. 'just_new_data', 'update_lots'.
See the function definitions for more details.


'''

# analyse direction detection
direction_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\test_venv\Direction_detection_Dec23"
direction_detect_analysis_pipe_Dec23(direction_path, analyse_what='all',)



# analyse target detection
target_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\test_venv\Target_detection_Dec23"
target_detect_analysis_pipe_Dec23(target_path, analyse_what='all')


# analyse missing target detection
missing_target_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\test_venv\Missing_target_detection_Dec23"
target_detect_analysis_pipe_Dec23(missing_target_path, analyse_what='all')





