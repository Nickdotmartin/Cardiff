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

This script will:
1. read in a video file, convert each frame to an image and save (.45 seconds @ 960frames = 436 frames).

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

videos_dir = r"C:\Users\sapnm4\Videos\monitor_calibration_videos"
videos_dir = os.path.normpath(videos_dir)

save_dir = r"C:\Users\sapnm4\OneDrive - Cardiff University\Pictures\mon_cali_images"
save_dir = os.path.normpath(save_dir)

sep_vals = [6, 3, 0, 1, 18, 2, 99]
ISI_vals = [-1, 0, 9, 2, 4, 6, 12, 24]

# ### Part 1 - extracting frames

# for sep in sep_vals:
#     for isi in ISI_vals:
#         vid_name = f'ISI{isi}_sep{sep}.mp4'
#         vid_path = os.path.join(videos_dir, vid_name)
#         # print(vid_path)
#         if os.path.isfile(vid_path):
#             print(f'\n\tfound: {vid_name}')
#             vidcap = cv2.VideoCapture(vid_path)
#             totalframecount = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
#             print("The total number of frames in this video is ", totalframecount)
#
#             # # # extract frames
#             success, image = vidcap.read()
#             count = 1
#             while success:
#                 image_name = f"{vid_name[:-4]}_fr{count}.jpg"
#
#                 this_vid_image_dir = os.path.join(save_dir, vid_name[:-4])
#                 if not os.path.isdir(this_vid_image_dir):
#                     os.makedirs(this_vid_image_dir)
#                 all_full_frames_dir = os.path.join(this_vid_image_dir, 'all_full_frames')
#                 if not os.path.isdir(all_full_frames_dir):
#                     os.makedirs(all_full_frames_dir)
#
#                 image_save_path = os.path.join(os.path.join(all_full_frames_dir, image_name))
#                 cv2.imwrite(image_save_path, image)  # save frame as JPG file
#                 success, image = vidcap.read()
#                 print(f'Saved image: {image_save_path}')
#                 count += 1
#
#
#
#         else:
#             print(f'\t\tmissing: {vid_name}')

# # part 2
'''
part 2

I am manually using this bit of code to get the co-ordinates for the bounding boxes.
'''

# isi = 9
# sep = 99
# frame = 359
#
# ROI_size = 15  # 5
# enlarge_scale = 10
#
# p1_tl = 172, 435
# # p1_tl = 172, 435
#
# p2_tl = 144, 465
#
#
# root_dir = r"C:\Users\sapnm4\OneDrive - Cardiff University\Pictures\mon_cali_images"
# cond_dir = rf"ISI{isi}_sep{sep}\all_full_frames"
# image_name = f"ISI{isi}_sep{sep}_fr{frame}.jpg"
#
# image_path = os.path.join(root_dir, cond_dir, image_name)
# print(image_path)
# print(os.path.isfile(image_path))
#
# img = cv2.imread(image_path)
#
# # Check the type of read image
# print(f'size: {img.shape}type: {type(img)}')
#
# # Display the image
# # cv2.imshow('image', img)
#
# # [rows, columns]
# fixation = 20, 290
# crop_size = 190
# crop = img[fixation[0]:fixation[0]+crop_size, fixation[1]:fixation[1]+crop_size]
# print(f'cropped: ({fixation[0]}: {fixation[0]+crop_size}, {fixation[1]}: {fixation[1]+crop_size})')
# cv2.imshow('original', img)
# cv2.imshow('cropped', crop)
#
#
# p1_box = img[p1_tl[0]: p1_tl[0]+ROI_size, p1_tl[1]: p1_tl[1]+ROI_size]
# print(f'p1_box: ({p1_tl[0]}: {p1_tl[0]+ROI_size}, {p1_tl[1]}: {p1_tl[1]+ROI_size})')
# # cv2.imshow('p1_box', p1_box)
#
# w = int(p1_box.shape[1] * enlarge_scale)
# h = int(p1_box.shape[0] * enlarge_scale)
# big_p1_box = cv2.resize(p1_box, (w, h))
# cv2.imshow('big_p1_box', big_p1_box)
#
# p2_box = img[p2_tl[0]: p2_tl[0]+ROI_size, p2_tl[1]: p2_tl[1]+ROI_size]
# print(f'p2_box: ({p2_tl[0]}: {p2_tl[0]+ROI_size}, {p2_tl[1]}: {p2_tl[1]+ROI_size})')
#
# print(f'p2_box array: \n{p2_box}')
#
#
# # cv2.imshow('p2_box', p2_box)
# w = int(p2_box.shape[1] * enlarge_scale)
# h = int(p2_box.shape[0] * enlarge_scale)
# big_p2_box = cv2.resize(p2_box, (w, h))
# cv2.imshow('big_p2_box', big_p2_box)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

'''
Once I have made an xlsx document storing the first useful frame of each video I will load it here.

I then also need to visualise frames and be able to try cropping ROIs and displaying them on the screen.

Hopefully I can simply store a single tuple for the ROI (e.g., top-left corner)

add something to convert images to greyscale - perhaps when saving bounding boxes.
Or infact
1. convert bounding box to grey - save image
2. at same time calculate mean value of grey image and save that to csv
'''

# # load excel sheet with frame numbers where probes appear and x, y co-ordinates for cropping probes.
# excel_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\Pictures\mon_cali_images\frame_and_bbox_details.xlsx"
# excel_path = os.path.normpath(excel_path)
#
# bb_box_df = pd.read_excel(excel_path)
# bb_box_df = bb_box_df.iloc[0:31]
# print(f'bb_box_df:\n{bb_box_df}')
#
# rows, cols = bb_box_df.shape
# print(f'rows: {rows}, cols: {cols}')
# print(f'headers: {bb_box_df.columns.to_list()}')
#
# empty_list = []
#
# # loop through each row for excel, e.g., frames from each video/condition
# for index, row in bb_box_df.iterrows():
#     if index < 100:
#         isi = row['isi']
#         sep = row['sep']
#         filename = row['filename']
#         print(f'\n{filename}, isi{isi}, sep{sep}')
#
#         pr1_frame = row['first_pr1_fr']
#         pr2_frame = row['first_pr2_fr']
#
#         # set to process 100 frames (or 140 for isi24)
#         from_fr = pr1_frame-10
#         to_fr = from_fr+100
#         if isi == 24:
#             to_fr = from_fr+140
#         if to_fr > 435:
#             to_fr = 435
#         print(f'from_fr: {from_fr} : to_fr: {to_fr}')
#
#         # probe bounding box is 5x5 pixels
#         ROI_size = 5  # 5
#
#         pr1_x = int(row['pr1_x'])
#         pr1_y = int(row['pr1_y'])
#
#         if sep != 99:
#             pr2_x = int(row['pr2_x'])
#             pr2_y = int(row['pr2_y'])
#         else:
#             pr2_x = np.nan
#             pr2_y = np.nan
#         print(f'pr1: ({pr1_x}, {pr1_y}), pr2: ({pr2_x}, {pr2_y})')
#
#         # loop through each of the frames
#         for idx, frame in enumerate(list(range(from_fr, to_fr))):
#             # print(frame)
#             # frame = pr1_frame
#
#             # load image of this frame
#             root_dir = r"C:\Users\sapnm4\OneDrive - Cardiff University\Pictures\mon_cali_images"
#             cond_dir = rf"ISI{isi}_sep{sep}\all_full_frames"
#             image_name = f"ISI{isi}_sep{sep}_fr{frame}.jpg"
#             image_path = os.path.join(root_dir, cond_dir, image_name)
#
#             gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#
#
#             p1_box = gray_img[pr1_x: pr1_x+ROI_size, pr1_y: pr1_y+ROI_size]
#             print(f'p1_box: {type(p1_box)}')
#             p1_mean = np.mean(p1_box)
#             p1_max = np.max(p1_box)
#             # p1_max = str(p1_max)
#             print(f'p1_box array: \n{p1_box}')
#             print(f'p1_mean: {p1_mean}')
#             print(f'p1_max: {p1_max}')
#             print(f'p1_max: {type(p1_max)}')
#
#             if sep != 99:
#                 p2_box = gray_img[pr2_x: pr2_x+ROI_size, pr2_y: pr2_y+ROI_size]
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
#             save_row = [filename, isi, sep, idx, frame,
#                         pr1_frame, pr1_x, pr1_y, p1_mean, str(p1_max),
#                         pr2_frame, pr2_x, pr2_y, p2_mean, str(p2_max),
#                         joint_mean, str(joint_max)]
#             empty_list.append(save_row)
#             print(f'{frame}: {save_row}')
#
# print(f'empty_list shape: {np.shape(empty_list)}')
# print(empty_list)
# results_df = pd.DataFrame(data=empty_list,
#                           columns=['filename', 'isi', 'sep', 'idx', 'frame',
#                                    'pr1_frame', 'pr1_x', 'pr1_y',
#                                    'p1_mean', 'p1_max',
#                                    'pr2_frame', 'pr2_x', 'pr2_y',
#                                    'p2_mean', 'p2_max',
#                                    'joint_mean', 'joint_max']
#                           )
#
# results_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\project_stuff\monitor_calibration\ASUS\monitor_refresh_results.csv"
# results_path = os.path.normpath(results_path)
# results_df.to_csv(results_path)
#
# print('\nall finished making results csv')

'''
Load results csv, 
loop through conditions/filenames
make plots: 
1. p1, p2, joint mean on same plot
2. p1, p2, joint max on same plot

3. by isi - joint means
4. by isi - joint max
'''

# load results csv
results_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\project_stuff\monitor_calibration\ASUS\monitor_refresh_results.csv"
results_path = os.path.normpath(results_path)

results_df = pd.read_csv(results_path)
print(f'results_df:\n{results_df}')

print(f'headers:\n{results_df.columns}')


# loop through conditions
cond_list = list(results_df['filename'].unique())
print(f'cond_list:\n{cond_list}')

# cond_name = cond_list[0]
for cond_name in cond_list:

    cond_df = results_df[results_df['filename'] == cond_name]
    print(f'cond_df:\n{cond_df}')

    # 1. p1, p2, joint mean on same plot

    long_df = make_long_df(wide_df=cond_df,
                           cols_to_keep=['filename', 'idx'],
                           cols_to_change=['p1_mean', 'p2_mean', 'joint_mean'],
                           cols_to_change_show='mean_lum',
                           new_col_name='loc', strip_from_cols='_mean', verbose=True)
    print(f'long_df:\n{long_df}')

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.lineplot(data=long_df, x='idx', y='mean_lum', hue='loc', ax=ax)
    ax.set_xlabel('frames @ 960Hz')
    ax.set_ylabel('luminance')
    plt.title(f'{cond_name}: mean luminance')
    fig_dir = rf"C:\Users\sapnm4\OneDrive - Cardiff University\Pictures\mon_cali_images\cond_figs"
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)
    fig_savename = f'{cond_name}_mean_lum.png'
    fig_path = os.path.join(fig_dir, fig_savename)
    plt.savefig(fig_path)
    plt.show()
    plt.close()

    # 2. p1, p2, joint max on same plot
    long_df = make_long_df(wide_df=cond_df,
                           cols_to_keep=['filename', 'idx'],
                           cols_to_change=['p1_max', 'p2_max', 'joint_max'],
                           cols_to_change_show='max_lum',
                           new_col_name='loc', strip_from_cols='_max', verbose=True)
    print(f'long_df:\n{long_df}')

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.lineplot(data=long_df, x='idx', y='max_lum', hue='loc', ax=ax)
    ax.set_xlabel('frames @ 960Hz')
    ax.set_ylabel('luminance')
    plt.title(f'{cond_name}: max luminance')
    fig_dir = rf"C:\Users\sapnm4\OneDrive - Cardiff University\Pictures\mon_cali_images\cond_figs"
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)
    fig_savename = f'{cond_name}_max_lum.png'
    fig_path = os.path.join(fig_dir, fig_savename)
    plt.savefig(fig_path)
    plt.show()
    plt.close()

    # loop through isis
    isi_list = list(results_df['isi'].unique())
    print(f'isi_list:\n{isi_list}')


for isi in isi_list:

    isi_df = results_df[results_df['isi'] == isi]
    print(f'isi_df:\n{isi_df}')

    # 3. by isi - joint means

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.lineplot(data=isi_df, x='idx', y='joint_mean', hue='filename', ax=ax)
    ax.set_xlabel('frames @ 960Hz')
    ax.set_ylabel('luminance')
    plt.title(f'{isi}: mean luminance')
    fig_dir = rf"C:\Users\sapnm4\OneDrive - Cardiff University\Pictures\mon_cali_images\isi_figs"
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)
    fig_savename = f'isi{isi}_mean_lum.png'
    fig_path = os.path.join(fig_dir, fig_savename)
    plt.savefig(fig_path)
    plt.show()
    plt.close()

    # 4. by isi - joint max
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.lineplot(data=isi_df, x='idx', y='joint_max', hue='filename', ax=ax)
    ax.set_xlabel('frames @ 960Hz')
    ax.set_ylabel('luminance')
    plt.title(f'{isi}: max luminance')
    fig_dir = rf"C:\Users\sapnm4\OneDrive - Cardiff University\Pictures\mon_cali_images\isi_figs"
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)
    fig_savename = f'isi{isi}_max_lum.png'
    fig_path = os.path.join(fig_dir, fig_savename)
    plt.savefig(fig_path)
    plt.show()
    plt.close()

print('finished plotting monitor calibration')
