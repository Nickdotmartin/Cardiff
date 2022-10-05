import cv2
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from rad_flow_psignifit_analysis import make_long_df


'''
This script is based on Martin's MATLAB script called 'read_frames_ok',
which he used to analyse the original monitor calibration.

Before running this script, rename videos in format: vid_name = f'ISI{isi}_sep{sep}_v{ver}.mp4'

This script will:
1. read in a video file, convert each frame to an image and save.  
However, the first and last 30 frames are NOT at 960fps, so I only need frames 30 to 405.
This gives 375 frames @ 960pfs = .390625 seconds in realtime = 93.75 of my stimuli frames at 240fps
1 frame = 1.0416667ms in realtime
 
2. Allow me to inspect (e.g., visualise frames) to select/trim to 100 frames that include both probes for all ISIs.
Note the longest ISI is 24 frames/100ms, which will equate to 96 frames here at 960Hz.
The id of the first frame to be stored for each video.
I will actually skip back 5 frames prior to this to makes sure I get the full rise.

3. Allow me to inspect and identify the co-ordinates for bounding boxes for each probe.
Martin used 11x11 pixels, although our boxes might be a different size depending on how the camera has captured this.
for now my box is 5x5 pixels.

a) start frame
b) bounding box for probe 1
c) bounding box for probe 2

4. The actual analysis involves recoding the mean intensity of the pixels in each box, across 100 frames.
Note, Martin used the max intensity rather than the mean.  I should do both.
He then reports the mean intensity as the mean of the two boxes. 
That can't be right - as when only one probe is present, the mean will be lowered??? 

5. Seaborn plots of the data.  Frames on x-axis, mean intensity on y axis.


approx durations for the action in frames
@ 240   @240    @960
ISI_fr  p1+ISI  @960
-1      0       0
0       2       8
2       4       16
4       6       24
6       8       32          
9       11      44
12      14      56
24      26      104
'''

# videos_dir = r"C:\Users\sapnm4\Videos\monitor_calibration_videos_June22"
videos_dir = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\Kim_scripts\Kim_vid_example"
videos_dir = os.path.normpath(videos_dir)

# images_dir = r"C:\Users\sapnm4\OneDrive - Cardiff University\Pictures\mon_cali_images_June22"
images_dir = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\Kim_scripts\Kim_vid_frames"
images_dir = os.path.normpath(images_dir)

# sep_vals = [6, 3, 0, 1, 18, 2, 99, 400, 800, 2400]
# ISI_vals = [-2, -1, 0, 9, 2, 4, 6, 12, 24]
# versions = [1, 2, 3, 4, 5]
sep_vals = [0]
ISI_vals = [0]
versions = [0]

# todo: NOTE isi'-2' is an error, should have been '2'.  Lucky these are not key conditions!

# ### Part 1 - extracting frames
#
# for sep in sep_vals:
#     for isi in ISI_vals:
#         # vid_name = f'ISI{isi}_sep{sep}.mp4'  # for may videos
#         for ver in versions:
#             # vid_name = f'ISI{isi}_sep{sep}_v{ver}.mp4'  # for June videos
#             vid_name = '20210930_183432.mp4'
#             vid_path = os.path.join(videos_dir, vid_name)
#             # print(vid_path)
#             if not os.path.isfile(vid_path):
#                 print(f'\n\t\t\tmissing: {vid_name}')
#             else:
#                 print(f'\n\tfound: {vid_name}')
#                 vidcap = cv2.VideoCapture(vid_path)
#                 totalframecount = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
#                 print("The total number of frames in this video is ", totalframecount)
#
#                 # # # extract frames
#                 success, image = vidcap.read()
#                 count = 1
#                 while success:
#                     image_name = f"{vid_name[:-4]}_fr{count}.jpg"
#
#                     this_vid_image_dir = os.path.join(images_dir, vid_name[:-4])
#                     if not os.path.isdir(this_vid_image_dir):
#                         os.makedirs(this_vid_image_dir)
#                     all_full_frames_dir = os.path.join(this_vid_image_dir, 'all_full_frames')
#                     if not os.path.isdir(all_full_frames_dir):
#                         os.makedirs(all_full_frames_dir)
#
#                     image_save_path = os.path.join(os.path.join(all_full_frames_dir, image_name))
#                     cv2.imwrite(image_save_path, image)  # save frame as JPG file
#                     success, image = vidcap.read()
#                     print(f'Saved image: {image_save_path}')
#                     count += 1

'''part 1b
loop through video files to make an excel doc to populate with frame numbers for when probes appear

It starts with lists of sep and isi vals, its a bit convoluted, but it reflects the order of priority for analysis.
the main conditions I need are these in lists 1, 5 and 4. If there is time, then do 2 and 3.
'''
# sep_vals1 = [6, 3, 1, 0]  # , 18, 2, 99]
# ISI_vals1 = [-1, 0, 4, 9]  # , 2, 6, 12, 24]
#
# sep_list1 = list(np.repeat(sep_vals1, len(ISI_vals1)))
# ISI_list1 = list(np.tile(ISI_vals1, len(sep_vals1)))
#
# sep_vals2 = [6, 3, 1, 0]  # , 18, 2, 99]
# ISI_vals2 = [-2, 6, 12, 24]
# sep_list2 = list(np.repeat(sep_vals2, len(ISI_vals2)))
# ISI_list2 = list(np.tile(ISI_vals2, len(sep_vals2)))
#
# sep_vals3 = [18, 2]
# ISI_vals3 = [-1, 0, 4, 9, 2, 6, 12, 24]
# sep_list3 = list(np.repeat(sep_vals3, len(ISI_vals3)))
# ISI_list3 = list(np.tile(ISI_vals3, len(sep_vals3)))
#
# sep_list4 = [99]
# ISI_list4 = [0]
#
# sep_list5 = [400, 800, 2400]  # bloch
# ISI_list5 = [0, 0, 0]  # bloch
#
# sep_list = sep_list5 + sep_list1 + sep_list4 + sep_list2 + sep_list3
# ISI_list = ISI_list5 + ISI_list1 + ISI_list4 + ISI_list2 + ISI_list3
# print(f"sep_list ({len(sep_list)}): {sep_list}")
# print(f"ISI_list: {ISI_list}")
#
# stairs = list(range(len(sep_list)))
#
# # for idx, (this_stair, isi, sep) in enumerate(zip(stairs, ISI_list, sep_list)):
# #     print(idx, this_stair, isi, sep)
#
# # cond_dict = {i: {'sep': sep, 'isi': isi} for (i, sep, isi) in zip(stairs, sep_list, ISI_list)}
# # print(cond_dict)
#
# # # loop through theses, and through all versions.
# details_list = []
# version = [1, 2, 3, 4, 5]
# for idx, (isi, sep) in enumerate(zip(ISI_list, sep_list)):
#     for ver in versions:
#         vid_name = f'ISI{isi}_sep{sep}_v{ver}.mp4'  # for June videos
#         vid_path = os.path.join(videos_dir, vid_name)
#         # print(vid_path)
#         if not os.path.isfile(vid_path):
#             print(f'\n\t\t\tmissing: {vid_name}')
#         else:
#             print(f'\n\tfound: {vid_name}')
#             details_list.append([isi, sep, ver, vid_name])
#
# excel_headers = ['isi', 'sep', 'version', 'filename']
# excel_df = pd.DataFrame(details_list, columns=excel_headers)
# print(f"excel_df\n{excel_df}")
# excel_name = 'frame_and_bbox_details.xlsx'
# save_excel_path = os.path.join(images_dir, excel_name)
# excel_df.to_excel(save_excel_path, engine='openpyxl', index=False)

'''
I need to manually look through the frames to get the frame number where the 
first probe appears and put it into 'first_pr1_fr' on the excel doc. 
If there is a frame counter this should be on frame 25.  
I first looked by eye to see when the probe appeared. 
Once I had made plots I tweaked the first_pr1_fr number so that:
the value was rising at first_pr1_fr, but not rising at first_pr1_fr-1. 

I then add this code to the column marked 'first_pr2_fr':
=IF(B2=-1, F2, IF(B2=0, F2+8, IF(B2=2, F2+16, IF(B2=4, F2+24, IF(B2=6, F2+32, IF(B2=9, F2+44, IF(B2=12, F2+56, IF(B2=24, F2+104, 0))))))))
This is so that when I add the first frame where the first probe appears in 'first_pr1_fr', 
'first_pr2_fr' should predict the frame where the second probe appears.  e.g., if ISI is 2, then the second probe should appear 16 frames after the first.
'''

# # part 2
'''
part 2
I am manually using this bit of code to get the co-ordinates for the bounding boxes.

I enter the details to load the image 8 frames after pr1 first appears.
It will display the image, a cropped version focussed on the area where the probes should appear and 
bounding boxed around the probe1 (with co-ordinates for top left corner).

It will also load an image n frames later (depending on isi) and display the same cropped area and bounding box for probe 2.

I might need to try this a few times using different co-ordinates until the two probes are centered in the bounding boxes.     

I manually add the co-ordinates to the excel document in columns called pr1_x, pr1_y, pr2_x, pr2_y.
'''

# isi = 0
# sep = 0
# ver = 0
# first_frame = 299
# show_frame = first_frame + 4
#
# '''ROI_size was 5 for May but now using 10 so whole probe in with grey border'''
# ROI_size = 10  # 5
# enlarge_scale = 10  # 10
#
# # # [rows/vertical, cols/horizontal]
# # p1_tl = 109, 552  # default value for june22
# p1_tl = 0, 250
#
# p2_tl = 104, 559
#
# # cond_dir = rf"ISI{isi}_sep{sep}_v{ver}\all_full_frames"
# # image_name = f"ISI{isi}_sep{sep}_v{ver}_fr{show_frame}.jpg"
# cond_dir = rf"20210930_183432\all_full_frames"
# image_name = f"20210930_183432_fr369.jpg"
# image_path = os.path.join(images_dir, cond_dir, image_name)
# print(image_path)
# print(os.path.isfile(image_path))
#
# # load image used for bbox1
# img = cv2.imread(image_path)
# print(f'size: {img.shape}type: {type(img)}')
#
# # [rows, columns]
# # fixation = 20, 475  # corner of cropped image
# fixation = 0, 250  # corner of cropped image
# crop_size = 190
# crop = img[fixation[0]:fixation[0]+crop_size, fixation[1]:fixation[1]+crop_size]
# cv2.imshow('original', img)
# cv2.imshow('cropped', crop)
#
# # extract bounding box for first probe from image
# p1_box = img[p1_tl[0]: p1_tl[0]+ROI_size, p1_tl[1]: p1_tl[1]+ROI_size]
# print(f'p1_box: ({p1_tl[0]}: {p1_tl[0]+ROI_size}, {p1_tl[1]}: {p1_tl[1]+ROI_size})')
#
# # display an enlarged image to check bounding box
# w = int(p1_box.shape[1] * enlarge_scale)
# h = int(p1_box.shape[0] * enlarge_scale)
# big_p1_box = cv2.resize(p1_box, (w, h))
# cv2.imshow('big_p1_box', big_p1_box)

#
# # I am going to open p2_box from an image from a different frame to p1_box depending on the isi.
# # with p2_show_frame = isi*4+8
# if isi == -1:
#     p2_show_frame = show_frame
# elif sep > 100:
#     p2_show_frame = show_frame
# else:
#     p2_show_frame = show_frame + (isi*4+8)
# print(f"isi: {isi}, show_frame: {show_frame}, p2_show_frame: {p2_show_frame}")
# image2_name = f"ISI{isi}_sep{sep}_v{ver}_fr{p2_show_frame}.jpg"
# image2_path = os.path.join(images_dir, cond_dir, image2_name)
# img2 = cv2.imread(image2_path)
#
# # display a crop of the 2nd probe image
# crop2 = img2[fixation[0]:fixation[0]+crop_size, fixation[1]:fixation[1]+crop_size]
# # cv2.imshow('img2', img2)
# cv2.imshow('crop2', crop2)
# p2_box = img2[p2_tl[0]: p2_tl[0]+ROI_size, p2_tl[1]: p2_tl[1]+ROI_size]
#
# # p2_box = img[p2_tl[0]: p2_tl[0]+ROI_size, p2_tl[1]: p2_tl[1]+ROI_size]
# print(f'p2_box: ({p2_tl[0]}: {p2_tl[0]+ROI_size}, {p2_tl[1]}: {p2_tl[1]+ROI_size})')
#
# # display enlarged version of 2nd probe bounding box
# big_p2_box = cv2.resize(p2_box, (w, h))
# cv2.imshow('big_p2_box', big_p2_box)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

'''
Part 3, loop through excel, 
load images (and convert to grey for simplicity)
Get pixel values from the bounding boxes
save the mean and max values to new csv
'''
#
# load excel sheet with frame numbers where probes appear and x, y co-ordinates for cropping probes.
excel_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\Pictures\mon_cali_images_June22\frame_and_bbox_details_June22.xlsx"
excel_path = os.path.normpath(excel_path)

bb_box_df = pd.read_excel(excel_path)
bb_box_df = bb_box_df.iloc[0:46]  # just key conds 0:46
print(f'bb_box_df:\n{bb_box_df}')

rows, cols = bb_box_df.shape
print(f'rows: {rows}, cols: {cols}')
print(f'headers: {bb_box_df.columns.to_list()}')

empty_list = []

# loop through each row for excel, e.g., frames from each video/condition
for index, row in bb_box_df.iterrows():
    if index < 100:
        isi = row['isi']
        sep = row['sep']
        ver = row['version']
        filename = row['filename']
        print(f'\n{filename}, isi{isi}, sep{sep}, ver: {ver}')

        pr1_frame = int(row['first_pr1_fr'])
        pr2_frame = int(row['first_pr2_fr'])

        # set to process 100 frames (or 140 for isi24)
        # todo: Check I don't take frames <30 or >405 as these are not at 960
        from_fr = pr1_frame-8
        to_fr = from_fr+100
        if isi == 24:
            to_fr = from_fr+140
        if sep == 2400:
            to_fr = from_fr+140
        if to_fr > 435:
            to_fr = 435
        print(f'from_fr: {from_fr} : to_fr: {to_fr}')

        # probe bounding box is 5x5 pixels
        ROI_size = 10  # 5 for may22

        pr1_x = int(row['pr1_x'])
        pr1_y = int(row['pr1_y'])

        if sep < 99:
            pr2_x = int(row['pr2_x'])
            pr2_y = int(row['pr2_y'])
        else:
            pr2_x = np.nan
            pr2_y = np.nan
        print(f'pr1: ({pr1_x}, {pr1_y}), pr2: ({pr2_x}, {pr2_y})')

        # loop through each of the frames
        for idx, frame in enumerate(list(range(from_fr, to_fr))):
            # print(frame)
            # frame = pr1_frame

            # load image of this frame
            # root_dir = r"C:\Users\sapnm4\OneDrive - Cardiff University\Pictures\mon_cali_images_June22"
            cond_dir = rf"ISI{isi}_sep{sep}_v{ver}\all_full_frames"
            image_name = f"ISI{isi}_sep{sep}_v{ver}_fr{frame}.jpg"
            image_path = os.path.join(images_dir, cond_dir, image_name)
            # print(f"{idx}. {image_name} - {os.path.isfile(image_path)}")

            gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


            p1_box = gray_img[pr1_x: pr1_x+ROI_size, pr1_y: pr1_y+ROI_size]
            print(f'p1_box: {type(p1_box)}')
            p1_mean = np.mean(p1_box)
            p1_max = np.max(p1_box)
            # p1_max = str(p1_max)
            print(f'p1_box array: \n{p1_box}')
            print(f'p1_mean: {p1_mean}')
            print(f'p1_max: {p1_max}')
            print(f'p1_max: {type(p1_max)}')

            if sep < 99:
                p2_box = gray_img[pr2_x: pr2_x+ROI_size, pr2_y: pr2_y+ROI_size]
                p2_mean = np.mean(p2_box)
                p2_max = np.max(p2_box)

                joint_mean = np.mean([p1_mean, p2_mean])

                # joint_max = np.max([p1_max, p2_max])
                if p1_max > p2_max:
                    joint_max = p1_max
                else:
                    joint_max = p2_max
                '''strange, the max scores are not saving to the csv as shown here.
                I'm converting to str to try to avoid this.'''

            else:
                p2_box = np.nan
                p2_mean = np.nan
                p2_max = np.nan

                joint_mean = p1_mean
                joint_max = p1_max


            # save details to empty list
            save_row = [filename, isi, sep, idx, frame,
                        pr1_frame, pr1_x, pr1_y, p1_mean, str(p1_max),
                        pr2_frame, pr2_x, pr2_y, p2_mean, str(p2_max),
                        joint_mean, str(joint_max)]
            empty_list.append(save_row)
            print(f'{frame}: {save_row}')

print(f'empty_list shape: {np.shape(empty_list)}')
print(empty_list)
results_df = pd.DataFrame(data=empty_list,
                          columns=['filename', 'isi', 'sep', 'idx', 'frame',
                                   'pr1_frame', 'pr1_x', 'pr1_y',
                                   'p1_mean', 'p1_max',
                                   'pr2_frame', 'pr2_x', 'pr2_y',
                                   'p2_mean', 'p2_max',
                                   'joint_mean', 'joint_max'])

results_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\project_stuff\monitor_calibration\ASUS\monitor_refresh_results_June22.csv"
results_path = os.path.normpath(results_path)
results_df.to_csv(results_path, index=False)

print('\nall finished making results csv')

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

# # load results csv
# results_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\project_stuff\monitor_calibration\ASUS\monitor_refresh_results_June22.csv"
# results_path = os.path.normpath(results_path)
#
# results_df = pd.read_csv(results_path)
# print(f'results_df:\n{results_df}')
#
# # print(f'headers:\n{list(results_df.columns)}')
#
# # loop through ALL conditions
# cond_ver_list = list(results_df['filename'].unique())
# print(f'cond_ver_list (n={len(cond_ver_list)}):\n{cond_ver_list}')
#
# '''
# approx durations for the action in frames
# @ 240   @240    @960
# ISI_fr  p1+ISI  @960
# -1      0       0
# 0       2       8
# 2       4       16
# 4       6       24
# 6       8       32
# 9       11      44
# 12      14      56
# 24      26      104
# note isi-2 should be isi2!
# '''
#
# probe2_dict = {-1: 0, 0: 8, 2: 16, -2: 16, 4: 24, 6: 32, 9: 44, 12: 56, 24: 140}
#
# cond_ver_list = list(results_df['filename'].unique())
# print(f'cond_ver_list (n={len(cond_ver_list)}):\n{cond_ver_list}')
# cond_list = [i[:-5] for i in cond_ver_list]
# cond_list = set(cond_list)
# print(f'cond_list (n={len(cond_list)}):\n{cond_list}')
#
# # x tick labels
# x_tick_label_start = -2
# x_tick_locs = np.arange(0, 100, 8)
# x_tick_labels = np.arange(x_tick_label_start, 25 + x_tick_label_start, 2)


# # for cond_ver_name in conds_to_use:
# for cond_ver_name in cond_ver_list:
#
#     print(f'cond_ver_name:\n{cond_ver_name}')
#
#     split_txt = cond_ver_name.split("_")
#     print(f'split_txt:\n{split_txt}')
#
#     isi_name = split_txt[0]
#     print(f'isi_name:\n{isi_name}')
#     isi_val = int(isi_name[3:])
#     print(f'isi_val:\n{isi_val}')
#
#     sep_name = split_txt[1]
#     print(f'sep_name:\n{sep_name}')
#     sep_val = int(sep_name[3:])
#     print(f'sep_val:\n{sep_val}')
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
#
#
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
#     ax.set_xlabel('frames @ 240Hz')
#     ax.set_ylabel('pixel values (0:255')
#
#     # probe on/off markers
#     plt.axvline(x=vline_pr1_on, color='grey', linestyle='--')
#     plt.axvline(x=vline_pr1_off, color='grey', linestyle='--')
#     if sep_val not in [99, 400, 800, 2400]:
#         plt.axvline(x=vline_pr2_on, color='grey', linestyle='-.')
#         plt.axvline(x=vline_pr2_off, color='grey', linestyle='-.')
#
#     ax.set_xticks(ticks=x_tick_locs, labels=x_tick_labels)
#     # add shaded background for frames at 240
#     x_bg = np.arange(0, 100, 4)
#     for x0, x1 in zip(x_bg[::2], x_bg[1::2]):
#         plt.axvspan(x0, x1, color='black', alpha=0.05, zorder=0, linewidth=None)
#
#     plt.title(f'{cond_ver_name}: mean luminance')
#     fig_dir = rf"C:\Users\sapnm4\OneDrive - Cardiff University\Pictures\mon_cali_images_June22\cond_figs\mean_lum"
#     if not os.path.isdir(fig_dir):
#         os.makedirs(fig_dir)
#     fig_savename = f'{cond_ver_name}_mean_lum.png'
#     fig_path = os.path.join(fig_dir, fig_savename)
#     plt.savefig(fig_path)
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
#     ax.set_xlabel('frames @ 240Hz')
#     ax.set_ylabel('pixel values (0:255')
#
#     # probe on/off markers
#     plt.axvline(x=vline_pr1_on, color='grey', linestyle='--')
#     plt.axvline(x=vline_pr1_off, color='grey', linestyle='--')
#     if sep_val not in [99, 400, 800, 2400]:
#         plt.axvline(x=vline_pr2_on, color='grey', linestyle='-.')
#         plt.axvline(x=vline_pr2_off, color='grey', linestyle='-.')
#
#     ax.set_xticks(ticks=x_tick_locs, labels=x_tick_labels)
#     # add shaded background for frames at 240
#     x_bg = np.arange(0, 100, 4)
#     for x0, x1 in zip(x_bg[::2], x_bg[1::2]):
#         plt.axvspan(x0, x1, color='black', alpha=0.05, zorder=0, linewidth=None)
#
#     plt.title(f'{cond_ver_name}: max luminance')
#     fig_dir = rf"C:\Users\sapnm4\OneDrive - Cardiff University\Pictures\mon_cali_images_June22\cond_figs\max_lum"
#     if not os.path.isdir(fig_dir):
#         os.makedirs(fig_dir)
#     fig_savename = f'{cond_ver_name}_max_lum.png'
#     fig_path = os.path.join(fig_dir, fig_savename)
#     plt.savefig(fig_path)
#     plt.show()
#     plt.close()


#
# print('ISI figs')
# # just take one versio of each cond for now.
# conds_to_use = []
# for cond_name in cond_list:
#     cond_ver_name = cond_name + '1.mp4'
#     if cond_ver_name in cond_ver_list:
#         conds_to_use.append(cond_ver_name)
#     elif cond_name + '2.mp4' in cond_ver_list:
#         conds_to_use.append(cond_name + '2.mp4')
#     elif cond_name + '3.mp4' in cond_ver_list:
#         conds_to_use.append(cond_name + '3.mp4')
#     elif cond_name + '4.mp4' in cond_ver_list:
#         conds_to_use.append(cond_name + '4.mp4')
#     elif cond_name + '5.mp4' in cond_ver_list:
#         conds_to_use.append(cond_name + '5.mp4')
#     else:
#         print(f"\tnot found: {cond_ver_name}")
# print(f'conds_to_use (n={len(conds_to_use)}):\n{conds_to_use}')
# # conds_to_use = conds_to_use[:1]
# # print(f'conds_to_use (n={len(conds_to_use)}):\n{conds_to_use}')
#
# key_results_df = results_df[results_df['filename'].isin(conds_to_use)]
# print(f'key_results_df:\n{key_results_df}')
#
# # loop through isis
# isi_list = list(results_df['isi'].unique())
#
# isi_list = isi_list + ['1pr']
#
# print(f'isi_list:\n{isi_list}')
#
#
# for isi in isi_list:
#
#     print(f"isi: {isi}")
#     # isi_df = results_df[results_df['isi'] == isi]
#     isi_df = key_results_df[key_results_df['isi'] == isi]
#
#     vline_pr1_on = 8
#     vline_pr1_off = vline_pr1_on + 8
#     if isi not in [-1, '1pr']:
#         vline_pr2_on = 8 + probe2_dict[isi]
#         vline_pr2_off = vline_pr2_on + 8
#
#     if isi == 0:
#         # long probes are labelled sep 400, 800 etc shouldnt be here, so remove
#         isi_df = key_results_df.loc[((key_results_df['isi'] == isi) & (key_results_df['sep'] < 25))]
#
#     if isi == '1pr':
#         isi_df = key_results_df.loc[key_results_df['sep'] > 25]
#         x_tick_locs = np.arange(0, 148, 8)
#         x_tick_labels = np.arange(x_tick_label_start, 37 + x_tick_label_start, 2)
#
#
#     print(f'isi_df:\n{isi_df}')
#
#     # 3. by isi - joint means
#     fig, ax = plt.subplots(figsize=(6, 6))
#     sns.lineplot(data=isi_df, x='idx', y='joint_mean', hue='filename', ax=ax)
#     ax.set_xlabel('frames @ 240Hz')
#     ax.set_ylabel('pixel values (0:255')
#
#
#     # probe on/off markers
#     plt.axvline(x=vline_pr1_on, color='grey', linestyle='--')
#     plt.axvline(x=vline_pr1_off, color='grey', linestyle='--')
#     if isi == '1pr':
#         plt.axvline(x=vline_pr1_on + 16, color='grey', linestyle='-.')
#         plt.axvline(x=vline_pr1_on + 32, color='grey', linestyle='-.')
#         plt.axvline(x=vline_pr1_on + 96, color='grey', linestyle='-.')
#     else:
#         plt.axvline(x=vline_pr2_on, color='grey', linestyle='-.')
#         plt.axvline(x=vline_pr2_off, color='grey', linestyle='-.')
#
#     ax.set_xticks(ticks=x_tick_locs, labels=x_tick_labels)
#
#     # add shaded background for frames at 240
#     x_bg = np.arange(0, 108, 4)
#     if isi == '1pr':
#         x_bg = np.arange(0, 148, 4)
#     for x0, x1 in zip(x_bg[::2], x_bg[1::2]):
#         plt.axvspan(x0, x1, color='black', alpha=0.05, zorder=0, linewidth=None)
#
#     plt.title(f'{isi}: mean luminance')
#     fig_dir = rf"C:\Users\sapnm4\OneDrive - Cardiff University\Pictures\mon_cali_images_June22\isi_figs\mean_lum"
#     if not os.path.isdir(fig_dir):
#         os.makedirs(fig_dir)
#     fig_savename = f'isi{isi}_mean_lum.png'
#     fig_path = os.path.join(fig_dir, fig_savename)
#     plt.savefig(fig_path)
#     plt.show()
#     plt.close()
#
#     # 4. by isi - joint max
#     fig, ax = plt.subplots(figsize=(6, 6))
#     sns.lineplot(data=isi_df, x='idx', y='joint_max', hue='filename', ax=ax)
#     ax.set_xlabel('frames @ 240Hz')
#     ax.set_ylabel('pixel values (0:255')
#     plt.title(f'{isi}: max luminance')
#
#     # probe on/off markers
#     plt.axvline(x=vline_pr1_on, color='grey', linestyle='--')
#     plt.axvline(x=vline_pr1_off, color='grey', linestyle='--')
#     if isi == '1pr':
#         plt.axvline(x=vline_pr1_on + 16, color='grey', linestyle='-.')
#         plt.axvline(x=vline_pr1_on + 32, color='grey', linestyle='-.')
#         plt.axvline(x=vline_pr1_on + 96, color='grey', linestyle='-.')
#     else:
#         plt.axvline(x=vline_pr2_on, color='grey', linestyle='-.')
#         plt.axvline(x=vline_pr2_off, color='grey', linestyle='-.')
#
#     ax.set_xticks(ticks=x_tick_locs, labels=x_tick_labels)
#     # add shaded background for frames at 240
#     x_bg = np.arange(0, 100, 4)
#     if isi == '1pr':
#         x_bg = np.arange(0, 148, 4)
#     for x0, x1 in zip(x_bg[::2], x_bg[1::2]):
#         plt.axvspan(x0, x1, color='black', alpha=0.05, zorder=0, linewidth=None)
#
#     fig_dir = rf"C:\Users\sapnm4\OneDrive - Cardiff University\Pictures\mon_cali_images_June22\isi_figs\max_lum"
#     if not os.path.isdir(fig_dir):
#         os.makedirs(fig_dir)
#     fig_savename = f'isi{isi}_max_lum.png'
#     fig_path = os.path.join(fig_dir, fig_savename)
#     plt.savefig(fig_path)
#     plt.show()
#     plt.close()

print('finished plotting monitor calibration')
