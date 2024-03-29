import cv2
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from rad_flow_psignifit_analysis import make_long_df


'''
This script could just have a single bounding box including both probes.  
(This is fine since there are no concurrent probes.)

Infact, to save time, I've just made the two boxes big enough to include both probes,
and they are identical in terms of location.
In otherwords, two identical boxes (rather than just one). 


Using this script to check Rad_flow_23 on the OLED monitor.  

This script is based on Martin's MATLAB script called 'read_frames_ok',
which he used to analyse the original monitor calibration.

Before running this script, rename videos in format: vid_name = f'cong_ISI_{isi}_sep_{sep}_v{ver}.mp4'

This script will:
1. read in a video file, convert each frame to an image and save.  
However, the first and last 30 frames are NOT at 960fps, so I only need frames 30 to 405.
This gives 375 frames @ 960pfs = .390625 seconds in realtime = 93.75 of my stimuli frames at 240fps
1 frame = 1.0416667ms in realtime
 
2. Allow me to inspect (e.g., visualise frames) to select/trim to 50 frames that include both probes for all ISIs.
Note the longest ISI is 24 frames/100ms, which will equate to 96 frames here at 960Hz.
The id of the first frame to be stored for each video.
I will actually skip back 5 frames prior to this to makes sure I get the full rise.

3. Allow me to inspect and identify the co-ordinates for bounding boxes for each probe.
Martin used 11x11 pixels, although our boxes might be a different size depending on how the camera has captured this.
for now my box is 5x5 pixels.

a) start frame
b) bounding box for probe 1
c) bounding box for probe 2

4. The actual analysis involves recoding the mean intensity of the pixels in each box, across 50 frames.
Note, Martin used the max intensity rather than the mean.  I should do both.
He then reports the mean intensity as the mean of the two boxes. 
That can't be right - as when only one probe is present, the mean will be lowered??? 

5. Seaborn plots of the data.  Frames on x-axis, mean intensity on y axis.


approx durations for the action in frames
@240    @ 240   @240    @960        @120    @120    @120    @960
ISI_ms  ISI_fr  p1+ISI  @960        ISI_ms  ISI_fr  p1+ISI  @960
0       -1      0       0                   
0       0       2       8       
8.3     2       4       16          16.6    2       3       24
16      4       6       24          33.33   4       5       40
25      6       8       32          50      6       7       56  
37.5    9       11      44          75      9       10      80
50      12      14      56
100     24      26      104
'''
#
videos_dir = r"C:\Users\sapnm4\Videos\mon_cali_vids_May23\OLED_70"
images_dir = r"C:\Users\sapnm4\OneDrive - Cardiff University\Pictures\mon_cali_images_May23\OLED_70"
session_name = 'OLED_70'
#
# videos_dir = r"C:\Users\sapnm4\Videos\mon_cali_vids_May23\OLED_350"
# images_dir = r"C:\Users\sapnm4\OneDrive - Cardiff University\Pictures\mon_cali_images_May23\OLED_350"
# session_name = 'OLED_350'


videos_dir = os.path.normpath(videos_dir)
images_dir = os.path.normpath(images_dir)

congruence_vals = ['cong', 'incong']
ISI_vals = [2, 4, 6, 9]  #, 3, 4, 6, 9, 12, 24]
sep_vals = [2, 3, 6]
versions = [1, 2, 3, 4]

fps = 120



# ### Part 1 - extracting frames
# for congruence in congruence_vals:
#     for isi in ISI_vals:
#         for sep in sep_vals:
#             # vid_name = f'ISI_{isi}_sep_{sep}.mp4'  # for may videos
#             for ver in versions:
#                 vid_name = f'{congruence}_ISI_{isi}_sep_{sep}_v{ver}.mp4'
#                 vid_path = os.path.join(videos_dir, vid_name)
#                 # print(vid_path)
#                 if not os.path.isfile(vid_path):
#                     print(f'\n\t\t\tmissing: {vid_name}')
#                 else:
#                     print(f'\n\tfound: {vid_name}')
#
#                     # if vid_name in ['ISI_-1_sep_0_v1.mp4', 'ISI_-1_sep_0_v2.mp4']:
#                     #     print(f"******just do this one: {vid_name}")
#
#                     vidcap = cv2.VideoCapture(vid_path)
#                     totalframecount = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
#                     print("The total number of frames in this video is ", totalframecount)
#
#                     # # # extract frames
#                     success, image = vidcap.read()
#                     count = 1
#                     while success:
#                         image_name = f"{vid_name[:-4]}_fr{count}.jpg"
#
#                         this_vid_image_dir = os.path.join(images_dir, vid_name[:-4])
#                         if not os.path.isdir(this_vid_image_dir):
#                             os.makedirs(this_vid_image_dir)
#
#                         image_save_path = os.path.join(os.path.join(this_vid_image_dir, image_name))
#
#                         #  rotate image if necessary
#                         # if vid_name in ['ISI_-1_sep_5_exp_v2', 'ISI_-1_sep_5_exp_v3', 'ISI_2_sep_5_exp_v3']:
#                         #     print('not rotate?')
#                         # else:
#                         #     # rotate image
#                         #     image = cv2.rotate(image, cv2.ROTATE_180)
#
#                         # don't extract frames < 30 or > 405 as they aren't at 960Hz
#                         if 30 < count < 405:
#                             cv2.imwrite(image_save_path, image)  # save frame as JPG file
#                             print(f'Saved image: {image_save_path}')
#                         else:
#                             print(f'\tNot saving this image: {image_save_path}')
#
#                         success, image = vidcap.read()
#                         count += 1

'''part 1b
loop through video files to make an Excel doc to populate with frame numbers for when probes appear

It starts with lists of sep and isi vals, its a bit convoluted, but it reflects the order of priority for analysis.
the main conditions I need are these in lists 1, 5 and 4. If there is time, then do 2 and 3.
'''

#
# # # loop through theses, and through all versions.
# details_list = []
# # for idx, (isi, sep) in enumerate(zip(ISI_list, sep_list)):
# for congruence in congruence_vals:
#     for isi in ISI_vals:
#         for sep in sep_vals:
#             for ver in versions:
#                 vid_name = f'{congruence}_ISI_{isi}_sep_{sep}_v{ver}.mp4'
#                 vid_path = os.path.join(videos_dir, vid_name)
#                 # print(vid_path)
#                 if not os.path.isfile(vid_path):
#                     print(f'\n\t\t\tmissing: {vid_name}')
#                 else:
#                     print(f'\n\tfound: {vid_name}')
#                     details_list.append([congruence, isi, sep, ver, vid_name, '', '', '', '', '', '', '', '', '', ''])
#
# excel_headers = ['congruence', 'isi', 'sep', 'version', 'filename', 'comments', 'first_pr1_fr', 'analyse', 'first_pr2_fr', 'fix0', 'fix1', 'pr1_down', 'pr1_right', 'pr2_down', 'pr2_right']
# excel_df = pd.DataFrame(details_list, columns=excel_headers)
# print(f"excel_df\n{excel_df}")
# excel_name = 'frame_and_bbox_details_one_box.xlsx'
# save_excel_path = os.path.join(images_dir, excel_name)
# excel_df.to_excel(save_excel_path, engine='openpyxl', index=False)

'''
I need to manually look through the frames to get the frame number where the 
first probe appears and put it into 'first_pr1_fr' on the Excel doc. 
If there is a frame counter this should be on frame 61.  
I first looked by eye to see when the probe appeared. 
Once I had made plots I tweaked the first_pr1_fr number so that:
the value was rising at first_pr1_fr, but not rising at first_pr1_fr-1. 

I then add this code to the column marked 'first_pr2_fr':
for 240Hz monitor
=IF(B2=-1, G2, IF(B2=0, G2+8, IF(B2=2, G2+16, IF(B2=3, G2+20, IF(B2=4, G2+24, IF(B2=6, G2+32, IF(B2=9, G2+44, IF(B2=12, G2+56, IF(B2=24, G2+104, 0)))))))))

for OLED:
=IF(B2=2, G2+24, IF(B2=4, G2+40, IF(B2=6, G2+56, IF(B2=9, G2+80, 0))))


This is so that when I add the first frame where the first probe appears in 'first_pr1_fr', 
'first_pr2_fr' should predict the frame where the second probe appears.  
e.g., if ISI is 2, then the second probe should appear 16 frames after the first.
'''

# part 2
'''
part 2
I am manually using this bit of code to get the co-ordinates for the bounding boxes.

I enter the details to load the image 8 frames after pr1 first appears.
It will display the image, a cropped version focussed on the area where the probes should appear and 
bounding boxed around the probe1 (with co-ordinates for top left corner).

It will also load an image n frames later (depending on isi) and display the same cropped area and bounding box for probe 2.

I might need to try this a few times using different co-ordinates until the two probes are centered in the bounding boxes.     

Best to sort Excel by sep (as these should be fairly consistent).

I manually add coordinates into columns fix0 and fix1.  Or copy previous values down.

I add functions to the Excel document in columns called pr1_down, pr1_right, pr2_down, pr2_right:
pr1_down: fix0+sep, =J2+C2
pr1_right: fix1-sep, =K2-C2  (note, this is for radial)
Note, I'm now setting p2 in relation to fixation, not p1.
pr2_down: fix0-sep-1, =J2-C2-1
pr2_right: fix1+sep, =K2-C2 (or sometimes fix1+sep-2)

The copy & paste values once I have confirmed the correct positions (so formulas don't update them

NOTE: For some reason, y is first (vertical, down), then x (horiontal, right)
'''

congruence = 'cong'
isi = 4
sep = 2
ver = 1
first_frame = 285
show_frame = first_frame + 4
# show_frame = first_frame

'''ROI_size was 5 for May but now using 10 so whole probe in with grey border'''
ROI_size = 50  # 10  # 5
enlarge_scale = 10  # 10  # 10

# # [rows/vertical, cols/horizontal]  0, 0 is top left corner of screen
fixation = 260, 450  # top-left corner of cropped image
# fixation = 275, 493  # top-left corner of cropped image
print(f'fixation: ({fixation[0]}, {fixation[1]})')

# if sep >= 99:
#     p1_tl = fixation[0], fixation[1]
# else:
    # p1_tl = fixation[0]-sep, fixation[1]-sep  # works for tangent
    # if 'exp' in sep_name:
    #     p1_tl = fixation[0]-sep, fixation[1]+sep - 2  # use this for radial
    # else:
p1_tl = fixation[0] + sep, fixation[1] - sep  # use this for radial

# cond_dir = rf"ISI_{isi}_sep_{sep}_v{ver}\all_full_frames"
cond_dir = rf"{congruence}_ISI_{isi}_sep_{sep}_v{ver}"

image_name = f"{congruence}_ISI_{isi}_sep_{sep}_v{ver}_fr{show_frame}.jpg"

image_path = os.path.join(images_dir, cond_dir, image_name)
print(image_path)
print(os.path.isfile(image_path))

# load image used for bbox1
# todo: turn off greyscale?
# img = cv2.imread(image_path)
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
cv2.imshow('original', img)

print(f'size: {img.shape}type: {type(img)}')

# [rows, columns]
crop_size = 190
crop = img[p1_tl[0]:p1_tl[0]+crop_size, p1_tl[1]:p1_tl[1]+crop_size]
cv2.imshow('cropped', crop)

# extract bounding box for first probe from image
p1_box = img[p1_tl[0]: p1_tl[0]+ROI_size, p1_tl[1]: p1_tl[1]+ROI_size]
print(f'p1_box: ({p1_tl[0]}, {p1_tl[1]})')


# display an enlarged image to check bounding box
w = int(p1_box.shape[1] * enlarge_scale)
h = int(p1_box.shape[0] * enlarge_scale)
big_p1_box = cv2.resize(p1_box, (w, h))
cv2.imshow('big_p1_box', big_p1_box)


# I am going to open p2_box from an image from a different frame to p1_box depending on the isi.
# with p2_show_frame = isi*4+8 if 240Hz or isi*8+8 if 120Hz
if isi == -1:
    p2_show_frame = show_frame
elif sep > 50:
    p2_show_frame = show_frame
else:
    p2_show_frame = show_frame + (isi*8+8)
    # p2_show_frame = show_frame

print(f"\nisi: {isi}, show_frame: {show_frame}, p2_show_frame: {p2_show_frame}")
image2_name = f"{congruence}_ISI_{isi}_sep_{sep}_v{ver}_fr{p2_show_frame}.jpg"
image2_path = os.path.join(images_dir, cond_dir, image2_name)

# todo: turn off greyscale?
# img2 = cv2.imread(image2_path)
img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

if sep >= 99:
    p2_adjustment = int(ROI_size / 2)
else:
    # p2_adjustment = int(ROI_size/2) + sep  # works for tangent
    p2_adjustment = int(ROI_size/2)  # works for radial
print(f'p2_adjustment: {p2_adjustment}')

# if 'exp' in sep_name:
#     p2_tl = fixation[0] + sep, fixation[1] - sep  # use this for radial
# else:
# # default use this
p2_tl = fixation[0] - sep - 1, fixation[1] + sep  # use this for radial
# # or this one for sep 6
if sep == 6:
    p2_tl = fixation[0] - sep + 1, fixation[1] + sep - 2   # use this for radial
# # or edit this one
# p2_tl = fixation[0] - sep + 1, fixation[1] + sep - 5   # use this for radial



# display a crop of the 2nd probe image
# crop2 = img2[p2_tl[0]:p2_tl[0]+crop_size, p2_tl[1]:p2_tl[1]+crop_size]
# cv2.imshow('img2', img2)
# cv2.imshow('crop2', crop2)

# todo: note, I've made box2 a copy of box 1,
# p2_box = img2[p2_tl[0]: p2_tl[0]+ROI_size, p2_tl[1]: p2_tl[1]+ROI_size]
p2_box = p1_box
print(f'p2_box: ({p2_tl[0]}, {p2_tl[1]})')



# display enlarged version of 2nd probe bounding box
big_p2_box = cv2.resize(p2_box, (w, h))
cv2.imshow('big_p2_box', big_p2_box)

cv2.waitKey(0)
cv2.destroyAllWindows()

'''
Part 3, loop through Excel,
load images (and convert to grey for simplicity)
Get pixel values from the bounding boxes
save the mean and max values to new csv
'''

# # load Excel sheet with frame numbers where probes appear and x, y co-ordinates for cropping probes.
# excel_path = os.path.join(images_dir, 'frame_and_bbox_details_one_box.xlsx')
# excel_path = os.path.normpath(excel_path)
# print(f'excel_path:\n{excel_path}')
#
# bb_box_df = pd.read_excel(excel_path, sheet_name='Sheet1', engine='openpyxl',
#                           usecols=['congruence', 'isi', 'sep', 'version', 'filename', 'first_pr1_fr', 'analyse', 'first_pr2_fr',
#                                     'fix0', 'fix1', 'idx', 'pr1_down', 'pr1_right', 'pr2_down', 'pr2_right'])
# bb_box_df = bb_box_df[bb_box_df['analyse'] == 1]
# print(f'bb_box_df:\n{bb_box_df}')
#
# int_col_headers = ['isi', 'sep', 'version', 'first_pr1_fr', 'analyse', 'first_pr2_fr',
#                    'fix0', 'fix1', 'idx', 'pr1_down', 'pr1_right', 'pr2_down', 'pr2_right']
# bb_box_df[int_col_headers] = bb_box_df[int_col_headers].fillna(0).astype(int)
# print(f'bb_box_df:\n{bb_box_df}')
#
# rows, cols = bb_box_df.shape
# print(f'rows: {rows}, cols: {cols}')
# print(f'headers: {bb_box_df.columns.to_list()}')
#
# empty_list = []
#
# # loop through each row for Excel, e.g., frames from each video/condition
# for index, row in bb_box_df.iterrows():
#     analyse_this = row['analyse']
#     if analyse_this == 1:
#         congruence = row['congruence']
#         isi = row['isi']
#         sep_name = row['sep']
#         sep = int(sep_name)
#         # probes_dir = sep_name[2:]
#         ver = row['version']
#         filename = row['filename']
#         print(f'\n{filename}, {congruence}, ISI_{isi}, sep_{sep_name}, ver: {ver}')
#
#         pr1_frame = int(row['first_pr1_fr'])
#         pr2_frame = int(row['first_pr2_fr'])
#
#         # set to process 100 frames (or 140 for isi24)
#         from_fr = pr1_frame-8
#         to_fr = from_fr+100
#         if isi == 24:
#             to_fr = from_fr+140
#         if sep == 2400:
#             to_fr = from_fr+140
#         if to_fr > 404:
#             to_fr = 404
#         print(f'from_fr: {from_fr} : to_fr: {to_fr}')
#
#         # probe bounding box is 5x5 pixels
#         # todo: note this box is big
#         ROI_size = 50  # 5 for may22
#
#         pr1_down = int(row['pr1_down'])
#         pr1_right = int(row['pr1_right'])
#
#         if sep < 99:
#             pr2_down = int(row['pr2_down'])
#             pr2_right = int(row['pr2_right'])
#         else:
#             pr2_down = np.nan
#             pr2_right = np.nan
#         print(f'pr1: ({pr1_down}, {pr1_right}), pr2: ({pr2_down}, {pr2_right})')
#
#         # loop through each of the frames
#         for idx, frame in enumerate(list(range(from_fr, to_fr))):
#
#             # load image of this frame
#             # cond_dir = rf"ISI_{isi}_sep_{sep_name}_v{ver}\all_full_frames"
#             cond_dir = rf"{congruence}_ISI_{isi}_sep_{sep_name}_v{ver}"
#             image_name = f"{congruence}_ISI_{isi}_sep_{sep_name}_v{ver}_fr{frame}.jpg"
#             image_path = os.path.join(images_dir, cond_dir, image_name)
#             gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#
#             p1_box = gray_img[pr1_down: pr1_down+ROI_size, pr1_right: pr1_right+ROI_size]
#             p1_mean = np.mean(p1_box)
#             p1_max = np.max(p1_box)
#
#
#             if sep < 99:
#                 p2_box = gray_img[pr2_down: pr2_down+ROI_size, pr2_right: pr2_right+ROI_size]
#                 p2_mean = np.mean(p2_box)
#                 p2_max = np.max(p2_box)
#
#                 joint_mean = np.mean([p1_mean, p2_mean])
#
#                 # joint_max = np.max([p1_max, p2_max])
#                 if p1_max > p2_max:
#                     joint_max = p1_max
#                 else:
#                     joint_max = p2_max
#                 '''strange, the max scores are not saving to the csv as shown here.
#                 I'm converting to str to try to avoid this.'''
#
#             else:
#                 p2_box = np.nan
#                 p2_mean = np.nan
#                 p2_max = np.nan
#
#                 joint_mean = p1_mean
#                 joint_max = p1_max
#
#
#             # save details to empty list
#             save_row = [filename, congruence, isi, sep, sep_name,
#                         # probes_dir,
#                         ver, idx, frame,
#                         pr1_frame, pr1_down, pr1_right, p1_mean, str(p1_max),
#                         pr2_frame, pr2_down, pr2_right, p2_mean, str(p2_max),
#                         joint_mean, str(joint_max)]
#             empty_list.append(save_row)
#             # print(f'{frame}: {save_row}')
#
# print(f'empty_list shape: {np.shape(empty_list)}')
# print(empty_list)
# results_df = pd.DataFrame(data=empty_list,
#                           columns=['filename', 'congruence', 'isi', 'sep', 'sep_name',
#                                    # 'probes_dir',
#                                    'version', 'idx', 'frame',
#                                    'pr1_frame', 'pr1_down', 'pr1_right',
#                                    'p1_mean', 'p1_max',
#                                    'pr2_frame', 'pr2_down', 'pr2_right',
#                                    'p2_mean', 'p2_max',
#                                    'joint_mean', 'joint_max'])
#
# results_path = os.path.join(images_dir, f'monitor_one_box_results_{session_name}.csv')
# results_path = os.path.normpath(results_path)
# results_df.to_csv(results_path, index=False)
#
# print('\nall finished making results csv')

'''
Part 4: Load results csv,
loop through conditions/filenames
make plots:
# per version (e.g., separate plots for version 1, 2, 3)
just make one plot per cond (e.g., ver 1 if available, if not use ver 2 etc)
1. p1, p2, joint mean on same plot
2. p1, p2, joint max on same plot

per condition (e.g., one plot showing versions 1, 2 3)
1. joint means
2. joint max

get cond_means from joint means per version and cond max as mean of joint max per version
per isi
3. by isi - cond means
4. by isi - cond max
'''
#
# # load results csv
# # results_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\project_stuff\monitor_calibration\ASUS\monitor_refresh_results_Oct22.csv"
# results_path = os.path.join(images_dir, f'monitor_one_box_results_{session_name}.csv')
# results_path = os.path.normpath(results_path)
# results_df = pd.read_csv(results_path)
# print(f'results_df:\n{results_df.head()}')
#
# # print(f'headers:\n{list(results_df.columns)}')
#
# '''
#
# approx durations for the action in frames
# @240    @ 240   @240    @960        @120    @120    @120    @960
# ISI_ms  ISI_fr  p1+ISI  @960        ISI_ms  ISI_fr  p1+ISI  @960
# 0       -1      0       0
# 0       0       2       8
# 8.3     2       4       16          16.6    2       3       24
# 16      4       6       24          33.33   4       5       40
# 25      6       8       32          50      6       7       56
# 37.5    9       11      44          75      9       10      80
# 50      12      14      56
# 100     24      26      104
# '''
#
# images_dir = os.path.join(images_dir, 'aaa_one_box_figs')
#
# # # 240 Hz
# # probe2_dict = {-1: 0, 0: 8, 2: 16, 3: 20, 4: 24, 6: 32, 9: 44, 12: 56, 24: 140,
# #                # 'sep': {400: 16, 800: 32, 2400: 96}
# #                }
# # 120 Hz
# probe2_dict = {2: 24, 4: 40, 6: 56, 9: 80}
#
# # loop through ALL conditions
# cond_ver_list = list(results_df['filename'].unique())
# print(f'cond_ver_list (n={len(cond_ver_list)}):\n{cond_ver_list}')
# # cond_ver_list = cond_ver_list[:1]
# # print(f'cond_ver_list (n={len(cond_ver_list)}):\n{cond_ver_list}')
# # cond_list = [i[:-5] for i in cond_ver_list]
# # cond_list = set(cond_list)
# # print(f'cond_list (n={len(cond_list)}):\n{cond_list}')
#
# # x tick labels
# x_tick_label_start = -2
# x_tick_locs = np.arange(0, 100, 8)
# print(f'\nx_tick_locs: {x_tick_locs}')
# x_tick_labels = np.arange(x_tick_label_start, 26 + x_tick_label_start, 2)
# print(f'x_tick_labels: {x_tick_labels}')
#
# # for cond_ver_name in conds_to_use:
# for cond_ver_name in cond_ver_list:
#
#     print(f'\ncond_ver_name: {cond_ver_name}')
#
#     split_txt = cond_ver_name.split("_")
#     print(f'split_txt: {split_txt}')
#
#     congruence=split_txt[0]
#     isi_val = int(split_txt[2])
#     isi_name = f"ISI{isi_val}"
#     sep_val = int(split_txt[4])
#     # probes_dir = split_txt[4]
#     sep_name = f"sep{sep_val}"
#     # ver_name = split_txt[4][:2]
#     ver_name = split_txt[5][:2]
#     print(f'split_txt: {split_txt}')
#     print(f'congruence: {congruence}')
#     print(f'isi_val: {isi_val}')
#     print(f'isi_name: {isi_name}')
#     print(f'sep_val: {sep_val}')
#     print(f'sep_name: {sep_name}')
#     # print(f'probes_dir: {probes_dir}')
#     print(f'ver_name: {ver_name}')
#
#
#     '''insert vertical lines marking start and finish of each probe.
#     First probe should appear at frame 8 (@960)'''
#     vline_pr1_on = 8
#     vline_pr1_off = vline_pr1_on + 8
#
#     if sep_val == 400:
#         vline_pr1_off = vline_pr1_on + 16
#     if sep_val == 800:
#         vline_pr1_off = vline_pr1_on + 32
#     if sep_val == 2400:
#         vline_pr1_off = vline_pr1_on + 96
#
#     vline_pr2_on = 8 + probe2_dict[isi_val]
#     vline_pr2_off = vline_pr2_on + 8
#
#     cond_df = results_df[results_df['filename'] == cond_ver_name]
#     print(f'cond_df:\n{cond_df}')
#
#     # 1. p1, p2, joint mean on same plot
#     long_df = make_long_df(wide_df=cond_df,
#                            cols_to_keep=['filename', 'idx'],
#                            cols_to_change=['p1_mean', 'p2_mean', 'joint_mean'],
#                            cols_to_change_show='mean_lum',
#                            new_col_name='loc', strip_from_cols='_mean', verbose=True)
#     print(f'long_df:\n{long_df}')
#
#     fig, ax = plt.subplots(figsize=(6, 6))
#     sns.lineplot(data=long_df, x='idx', y='mean_lum', hue='loc', ax=ax)
#     ax.set_xlabel(f'frames @ {fps}Hz')
#     ax.set_ylabel('pixel values (0:255')
#
#     # probe on/off markers
#     plt.axvline(x=vline_pr1_on, color='grey', linestyle='--')
#     plt.axvline(x=vline_pr1_off, color='grey', linestyle='--')
#     if sep_val not in [99, 400, 800, 2400]:
#         plt.axvline(x=vline_pr2_on, color='grey', linestyle='-.')
#         plt.axvline(x=vline_pr2_off, color='grey', linestyle='-.')
#
#     ax.set_xticks(x_tick_locs)
#     ax.set_xticklabels(x_tick_labels)
#     # add shaded background for frames at 240
#     x_bg = np.arange(0, 100, 4)
#     for x0, x1 in zip(x_bg[::2], x_bg[1::2]):
#         plt.axvspan(x0, x1, color='black', alpha=0.05, zorder=0, linewidth=None)
#
#     plt.title(f'{cond_ver_name}: mean luminance')
#     # fig_dir = rf"C:\Users\sapnm4\OneDrive - Cardiff University\Pictures\mon_cali_images_Oct22\cond_figs\mean_lum"
#     fig_dir = os.path.join(images_dir, 'aaa_cond_figs', 'mean_lum')
#     if not os.path.isdir(fig_dir):
#         os.makedirs(fig_dir)
#     print(f'{os.path.isdir(fig_dir)}: fig_dir: {fig_dir}')
#     fig_savename = f'{cond_ver_name}_mean_lum.png'
#     fig_path = os.path.join(fig_dir, fig_savename)
#     plt.savefig(fig_path)
#     print(f'{os.path.isfile(fig_path)}: fig_path: {fig_path}')
#     plt.show()
#     plt.close()
#
#     # 2. p1, p2, joint max on same plot
#     long_df = make_long_df(wide_df=cond_df,
#                            cols_to_keep=['filename', 'idx'],
#                            cols_to_change=['p1_max', 'p2_max', 'joint_max'],
#                            cols_to_change_show='max_lum',
#                            new_col_name='loc', strip_from_cols='_max', verbose=True)
#     print(f'long_df:\n{long_df}')
#
#     fig, ax = plt.subplots(figsize=(6, 6))
#     sns.lineplot(data=long_df, x='idx', y='max_lum', hue='loc', ax=ax)
#     ax.set_xlabel(f'frames @ {fps}Hz')
#     ax.set_ylabel('pixel values (0:255')
#
#     # probe on/off markers
#     plt.axvline(x=vline_pr1_on, color='grey', linestyle='--')
#     plt.axvline(x=vline_pr1_off, color='grey', linestyle='--')
#     if sep_val not in [99, 400, 800, 2400]:
#         plt.axvline(x=vline_pr2_on, color='grey', linestyle='-.')
#         plt.axvline(x=vline_pr2_off, color='grey', linestyle='-.')
#
#     ax.set_xticks(x_tick_locs)
#     ax.set_xticklabels(x_tick_labels)
#     # add shaded background for frames at 240
#     x_bg = np.arange(0, 100, 4)
#     for x0, x1 in zip(x_bg[::2], x_bg[1::2]):
#         plt.axvspan(x0, x1, color='black', alpha=0.05, zorder=0, linewidth=None)
#
#     plt.title(f'{cond_ver_name}: max luminance')
#     # fig_dir = rf"C:\Users\sapnm4\OneDrive - Cardiff University\Pictures\mon_cali_images_Oct22\cond_figs\max_lum"
#     fig_dir = os.path.join(images_dir, 'aaa_cond_figs', 'max_lum')
#     if not os.path.isdir(fig_dir):
#         os.makedirs(fig_dir)
#     fig_savename = f'{cond_ver_name}_max_lum.png'
#     fig_path = os.path.join(fig_dir, fig_savename)
#     plt.savefig(fig_path)
#     plt.show()
#     plt.close()
#
#
#
# print('congruence figs')
# '''
# I will make a plot for each ISI comparing all exp plots (blue) with cont plots (orange).
# 1. loop through ISIs, and make df for each ISI.
# 2. loop through congruence (incong vs cong).
# 3. plot all exp conditions onto plots in one colour.
# 4. plot all cont cont conditions onto plots in another colour.'''
#
# # congruence_palette = {'exp': 'tab:red', 'cont': 'tab:green'}
# congruence_palette = {'incong': 'tab:red', 'cong': 'tab:green'}
#
#
# for isi in ISI_vals:
#
#     isi_df = results_df[results_df['isi'] == isi]
#     print(f"\n\nisi: {isi}\n{isi_df}\n")
#
#     '''insert vertical lines marking start and finish of each probe.
#     First probe should appear at frame 8 (@960)'''
#     vline_pr1_on = 8
#     vline_pr1_off = vline_pr1_on + 8
#
#     vline_pr2_on = 8 + probe2_dict[isi]
#     vline_pr2_off = vline_pr2_on + 8
#
#     '''mean lum plots'''
#     # 1. plots by isi - joint means, compare exp and cont - all lines
#     fig, ax = plt.subplots(figsize=(6, 6))
#
#     sns.lineplot(data=isi_df, x='idx', y='joint_mean', hue='congruence',
#                  units='filename', estimator=None, palette=congruence_palette,
#                  alpha=.5, ax=ax)
#     ax.set_xlabel(f'frames @ {fps}Hz')
#     ax.set_ylabel('pixel values (0:255')
#
#
#     # probe on/off markers
#     plt.axvline(x=vline_pr1_on, color='grey', linestyle='--')
#     plt.axvline(x=vline_pr1_off, color='grey', linestyle='--')
#     plt.axvline(x=vline_pr2_on, color='grey', linestyle='-.')
#     plt.axvline(x=vline_pr2_off, color='grey', linestyle='-.')
#
#     ax.set_xticks(x_tick_locs)
#     ax.set_xticklabels(x_tick_labels)
#
#     # add shaded background for frames at 240
#     x_bg = np.arange(0, 108, 4)
#     for x0, x1 in zip(x_bg[::2], x_bg[1::2]):
#         plt.axvspan(x0, x1, color='black', alpha=0.05, zorder=0, linewidth=None)
#
#     plt.title(f'ISI_{isi}: mean luminance, all files')
#     # fig_dir = rf"C:\Users\sapnm4\OneDrive - Cardiff University\Pictures\mon_cali_images_Oct22\isi_figs\mean_lum"
#     fig_dir = os.path.join(images_dir, 'aaa_congruence_figs_all', 'mean_lum')
#     if not os.path.isdir(fig_dir):
#         os.makedirs(fig_dir)
#     fig_savename = f'ISI_{isi}_compare_all_mean_lum.png'
#     fig_path = os.path.join(fig_dir, fig_savename)
#     plt.savefig(fig_path)
#     plt.show()
#     plt.close()
#
#
#
#     # 2. plots by isi - joint_mean, compare exp and cont - mean and errors
#     fig, ax = plt.subplots(figsize=(6, 6))
#
#     sns.lineplot(data=isi_df, x='idx', y='joint_mean', hue='congruence',
#                  palette=congruence_palette, alpha=.5,
#                  # units='filename', estimator=None,
#                  ax=ax)
#     ax.set_xlabel(f'frames @ {fps}Hz')
#     ax.set_ylabel('pixel values (0:255')
#
#     # probe on/off markers
#     plt.axvline(x=vline_pr1_on, color='grey', linestyle='--')
#     plt.axvline(x=vline_pr1_off, color='grey', linestyle='--')
#     plt.axvline(x=vline_pr2_on, color='grey', linestyle='-.')
#     plt.axvline(x=vline_pr2_off, color='grey', linestyle='-.')
#
#     ax.set_xticks(x_tick_locs)
#     ax.set_xticklabels(x_tick_labels)
#
#     # add shaded background for frames at 240
#     x_bg = np.arange(0, 108, 4)
#     for x0, x1 in zip(x_bg[::2], x_bg[1::2]):
#         plt.axvspan(x0, x1, color='black', alpha=0.05, zorder=0, linewidth=None)
#
#     plt.title(f'ISI_{isi}: mean luminance, mean and error of conditions')
#     # fig_dir = rf"C:\Users\sapnm4\OneDrive - Cardiff University\Pictures\mon_cali_images_Oct22\isi_figs\mean_lum"
#     fig_dir = os.path.join(images_dir, 'aaa_congruence_figs_ave', 'mean_lum')
#     if not os.path.isdir(fig_dir):
#         os.makedirs(fig_dir)
#     fig_savename = f'ISI_{isi}_compare_ave_mean_lum.png'
#     fig_path = os.path.join(fig_dir, fig_savename)
#     plt.savefig(fig_path)
#     plt.show()
#     plt.close()
#
#
#
#     '''max lum'''
#     # 3. plots by isi - joint means, compare exp and cont - all lines
#     fig, ax = plt.subplots(figsize=(6, 6))
#
#     sns.lineplot(data=isi_df, x='idx', y='joint_max', hue='congruence',
#                  palette=congruence_palette, alpha=.5,
#                  units='filename', estimator=None,
#                  ax=ax)
#     ax.set_xlabel(f'frames @ {fps}Hz')
#     ax.set_ylabel('pixel values (0:255')
#
#     # probe on/off markers
#     plt.axvline(x=vline_pr1_on, color='grey', linestyle='--')
#     plt.axvline(x=vline_pr1_off, color='grey', linestyle='--')
#     plt.axvline(x=vline_pr2_on, color='grey', linestyle='-.')
#     plt.axvline(x=vline_pr2_off, color='grey', linestyle='-.')
#
#     ax.set_xticks(x_tick_locs)
#     ax.set_xticklabels(x_tick_labels)
#
#     # add shaded background for frames at 240
#     x_bg = np.arange(0, 108, 4)
#     for x0, x1 in zip(x_bg[::2], x_bg[1::2]):
#         plt.axvspan(x0, x1, color='black', alpha=0.05, zorder=0, linewidth=None)
#
#     plt.title(f'ISI_{isi}: max luminance, all files')
#     # fig_dir = rf"C:\Users\sapnm4\OneDrive - Cardiff University\Pictures\mon_cali_images_Oct22\isi_figs\mean_lum"
#     fig_dir = os.path.join(images_dir, 'aaa_congruence_figs_all', 'max_lum')
#     if not os.path.isdir(fig_dir):
#         os.makedirs(fig_dir)
#     fig_savename = f'ISI_{isi}_compare_all_max_lum.png'
#     fig_path = os.path.join(fig_dir, fig_savename)
#     plt.savefig(fig_path)
#     plt.show()
#     plt.close()
#
#
#     # 4. plots by isi - joint_mean, compare exp and cont - mean and errors
#     fig, ax = plt.subplots(figsize=(6, 6))
#
#     sns.lineplot(data=isi_df, x='idx', y='joint_max', hue='congruence',
#                  palette=congruence_palette, alpha=.5,
#                  # units='filename', estimator=None,
#                  ax=ax)
#     ax.set_xlabel(f'frames @ {fps}Hz')
#     ax.set_ylabel('pixel values (0:255')
#
#     # probe on/off markers
#     plt.axvline(x=vline_pr1_on, color='grey', linestyle='--')
#     plt.axvline(x=vline_pr1_off, color='grey', linestyle='--')
#     plt.axvline(x=vline_pr2_on, color='grey', linestyle='-.')
#     plt.axvline(x=vline_pr2_off, color='grey', linestyle='-.')
#
#     ax.set_xticks(x_tick_locs)
#     ax.set_xticklabels(x_tick_labels)
#
#     # add shaded background for frames at 240
#     x_bg = np.arange(0, 108, 4)
#     for x0, x1 in zip(x_bg[::2], x_bg[1::2]):
#         plt.axvspan(x0, x1, color='black', alpha=0.05, zorder=0, linewidth=None)
#
#     plt.title(f'ISI_{isi}: max luminance, mean and error of conditions')
#     # fig_dir = rf"C:\Users\sapnm4\OneDrive - Cardiff University\Pictures\mon_cali_images_Oct22\isi_figs\mean_lum"
#     fig_dir = os.path.join(images_dir, 'aaa_congruence_figs_ave', 'max_lum')
#     if not os.path.isdir(fig_dir):
#         os.makedirs(fig_dir)
#     fig_savename = f'ISI_{isi}_compare_ave_max_lum.png'
#     fig_path = os.path.join(fig_dir, fig_savename)
#     plt.savefig(fig_path)
#     plt.show()
#     plt.close()
#
#
# print('separation figs')
# '''
# I will make a plot for each ISI comparing all separations
# 1. loop through ISIs, and make df for each ISI.
# 2. loop through separations and plot with different colours.'''
#
# # congruence_palette = {'exp': 'tab:red', 'cont': 'tab:green'}
# separation_palette = {2: 'tab:red', 3: 'tab:green', 6: 'tab:blue'}
#
#
# for isi in ISI_vals:
#
#     isi_df = results_df[results_df['isi'] == isi]
#     print(f"\n\nisi: {isi}\n{isi_df}\n")
#
#     '''insert vertical lines marking start and finish of each probe.
#     First probe should appear at frame 8 (@960)'''
#     vline_pr1_on = 8
#     vline_pr1_off = vline_pr1_on + 8
#
#     vline_pr2_on = 8 + probe2_dict[isi]
#     vline_pr2_off = vline_pr2_on + 8
#
#     '''mean lum plots'''
#     # 1. plots by isi - joint means, compare exp and cont - all lines
#     fig, ax = plt.subplots(figsize=(6, 6))
#
#     sns.lineplot(data=isi_df, x='idx', y='joint_mean', hue='sep',
#                  units='filename', estimator=None, palette=separation_palette,
#                  alpha=.5, ax=ax)
#     ax.set_xlabel(f'frames @ {fps}Hz')
#     ax.set_ylabel('pixel values (0:255')
#
#
#     # probe on/off markers
#     plt.axvline(x=vline_pr1_on, color='grey', linestyle='--')
#     plt.axvline(x=vline_pr1_off, color='grey', linestyle='--')
#     plt.axvline(x=vline_pr2_on, color='grey', linestyle='-.')
#     plt.axvline(x=vline_pr2_off, color='grey', linestyle='-.')
#
#     ax.set_xticks(x_tick_locs)
#     ax.set_xticklabels(x_tick_labels)
#
#     # add shaded background for frames at 240
#     x_bg = np.arange(0, 108, 4)
#     for x0, x1 in zip(x_bg[::2], x_bg[1::2]):
#         plt.axvspan(x0, x1, color='black', alpha=0.05, zorder=0, linewidth=None)
#
#     plt.title(f'ISI_{isi}: mean luminance, all files')
#     # fig_dir = rf"C:\Users\sapnm4\OneDrive - Cardiff University\Pictures\mon_cali_images_Oct22\isi_figs\mean_lum"
#     fig_dir = os.path.join(images_dir, 'aaa_sep_figs_all', 'mean_lum')
#     if not os.path.isdir(fig_dir):
#         os.makedirs(fig_dir)
#     fig_savename = f'ISI_{isi}_compare_all_mean_lum.png'
#     fig_path = os.path.join(fig_dir, fig_savename)
#     plt.savefig(fig_path)
#     plt.show()
#     plt.close()
#
#
#
#     # 2. plots by isi - joint_mean, compare exp and cont - mean and errors
#     fig, ax = plt.subplots(figsize=(6, 6))
#
#     sns.lineplot(data=isi_df, x='idx', y='joint_mean', hue='sep',
#                  palette=separation_palette, alpha=.5,
#                  # units='filename', estimator=None,
#                  ax=ax)
#     ax.set_xlabel(f'frames @ {fps}Hz')
#     ax.set_ylabel('pixel values (0:255')
#
#     # probe on/off markers
#     plt.axvline(x=vline_pr1_on, color='grey', linestyle='--')
#     plt.axvline(x=vline_pr1_off, color='grey', linestyle='--')
#     plt.axvline(x=vline_pr2_on, color='grey', linestyle='-.')
#     plt.axvline(x=vline_pr2_off, color='grey', linestyle='-.')
#
#     ax.set_xticks(x_tick_locs)
#     ax.set_xticklabels(x_tick_labels)
#
#     # add shaded background for frames at 240
#     x_bg = np.arange(0, 108, 4)
#     for x0, x1 in zip(x_bg[::2], x_bg[1::2]):
#         plt.axvspan(x0, x1, color='black', alpha=0.05, zorder=0, linewidth=None)
#
#     plt.title(f'ISI_{isi}: mean luminance, mean and error of conditions')
#     # fig_dir = rf"C:\Users\sapnm4\OneDrive - Cardiff University\Pictures\mon_cali_images_Oct22\isi_figs\mean_lum"
#     fig_dir = os.path.join(images_dir, 'aaa_sep_figs_ave', 'mean_lum')
#     if not os.path.isdir(fig_dir):
#         os.makedirs(fig_dir)
#     fig_savename = f'ISI_{isi}_compare_ave_mean_lum.png'
#     fig_path = os.path.join(fig_dir, fig_savename)
#     plt.savefig(fig_path)
#     plt.show()
#     plt.close()
#
#
#
#     '''max lum'''
#     # 3. plots by isi - joint means, compare exp and cont - all lines
#     fig, ax = plt.subplots(figsize=(6, 6))
#
#     sns.lineplot(data=isi_df, x='idx', y='joint_max', hue='sep',
#                  palette=separation_palette, alpha=.5,
#                  units='filename', estimator=None,
#                  ax=ax)
#     ax.set_xlabel(f'frames @ {fps}Hz')
#     ax.set_ylabel('pixel values (0:255')
#
#     # probe on/off markers
#     plt.axvline(x=vline_pr1_on, color='grey', linestyle='--')
#     plt.axvline(x=vline_pr1_off, color='grey', linestyle='--')
#     plt.axvline(x=vline_pr2_on, color='grey', linestyle='-.')
#     plt.axvline(x=vline_pr2_off, color='grey', linestyle='-.')
#
#     ax.set_xticks(x_tick_locs)
#     ax.set_xticklabels(x_tick_labels)
#
#     # add shaded background for frames at 240
#     x_bg = np.arange(0, 108, 4)
#     for x0, x1 in zip(x_bg[::2], x_bg[1::2]):
#         plt.axvspan(x0, x1, color='black', alpha=0.05, zorder=0, linewidth=None)
#
#     plt.title(f'ISI_{isi}: max luminance, all files')
    # # fig_dir = rf"C:\Users\sapnm4\OneDrive - Cardiff University\Pictures\mon_cali_images_Oct22\isi_figs\mean_lum"
    # fig_dir = os.path.join(images_dir, 'aaa_sep_figs_all', 'max_lum')
    # if not os.path.isdir(fig_dir):
    #     os.makedirs(fig_dir)
    # fig_savename = f'ISI_{isi}_compare_all_max_lum.png'
    # fig_path = os.path.join(fig_dir, fig_savename)
    # plt.savefig(fig_path)
    # plt.show()
    # plt.close()
    #
    #
    # # 4. plots by isi - joint_mean, compare exp and cont - mean and errors
    # fig, ax = plt.subplots(figsize=(6, 6))
    #
    # sns.lineplot(data=isi_df, x='idx', y='joint_max', hue='sep',
    #              palette=separation_palette, alpha=.5,
    #              # units='filename', estimator=None,
    #              ax=ax)
    # ax.set_xlabel(f'frames @ {fps}Hz')
    # ax.set_ylabel('pixel values (0:255')
    #
    # # probe on/off markers
    # plt.axvline(x=vline_pr1_on, color='grey', linestyle='--')
    # plt.axvline(x=vline_pr1_off, color='grey', linestyle='--')
    # plt.axvline(x=vline_pr2_on, color='grey', linestyle='-.')
    # plt.axvline(x=vline_pr2_off, color='grey', linestyle='-.')
    #
    # ax.set_xticks(x_tick_locs)
    # ax.set_xticklabels(x_tick_labels)
    #
    # # add shaded background for frames at 240
    # x_bg = np.arange(0, 108, 4)
    # for x0, x1 in zip(x_bg[::2], x_bg[1::2]):
    #     plt.axvspan(x0, x1, color='black', alpha=0.05, zorder=0, linewidth=None)
    #
    # plt.title(f'ISI_{isi}: max luminance, mean and error of conditions')
    # # fig_dir = rf"C:\Users\sapnm4\OneDrive - Cardiff University\Pictures\mon_cali_images_Oct22\isi_figs\mean_lum"
    # fig_dir = os.path.join(images_dir, 'aaa_sep_figs_ave', 'max_lum')
    # if not os.path.isdir(fig_dir):
    #     os.makedirs(fig_dir)
    # fig_savename = f'ISI_{isi}_compare_ave_max_lum.png'
    # fig_path = os.path.join(fig_dir, fig_savename)
    # plt.savefig(fig_path)
    # plt.show()
    # plt.close()



print('finished plotting monitor calibration')
