import cv2
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from rad_flow_psignifit_analysis import make_long_df


'''
Using this script to check Exp1_Jan23_radial_v2 (and V3), and I get different results for expansion and contraction for 
concurrent probes (which should be identical).

This script is based on Martin's MATLAB script called 'read_frames_ok',
which he used to analyse the original monitor calibration.

Before running this script, rename videos in format: vid_name = f'ISI_{isi}_sep_{sep}_v{ver}.mp4'

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

# images_dir = r"C:\Users\sapnm4\OneDrive - Cardiff University\Pictures\Check_stim_Jan23\check_stim_images_Jan23\v2"
# videos_dir = r"C:\Users\sapnm4\OneDrive - Cardiff University\Pictures\Check_stim_Jan23\check_stim_vids_Jan23\v2"
videos_dir = r"C:\Users\sapnm4\OneDrive - Cardiff University\Pictures\Check_stim_Jan23\check_stim_vids_Jan23\v3"
images_dir = r"C:\Users\sapnm4\OneDrive - Cardiff University\Pictures\Check_stim_Jan23\check_stim_images_Jan23\v3"
videos_dir = os.path.normpath(videos_dir)
images_dir = os.path.normpath(images_dir)

# sep_vals = [0, 1, 2, 3, 6, 18, 99, 400, 800, 2400]
sep_vals = ['5_exp', '5_cont']
ISI_vals = [-1, 0, 2]  #, 3, 4, 6, 9, 12, 24]
versions = [1, 2, 3, 4, 5]  #, 6, 7, 8]

# # there are 225 videos recorded.  How many are good?

# todo: in future don't extract frames < 30 or > 405

# ### Part 1 - extracting frames
# for sep in sep_vals:
#     for isi in ISI_vals:
#         # vid_name = f'ISI_{isi}_sep_{sep}.mp4'  # for may videos
#         for ver in versions:
#             vid_name = f'ISI_{isi}_sep_{sep}_v{ver}.mp4'
#             vid_path = os.path.join(videos_dir, vid_name)
#             # print(vid_path)
#             if not os.path.isfile(vid_path):
#                 print(f'\n\t\t\tmissing: {vid_name}')
#             else:
#                 print(f'\n\tfound: {vid_name}')
#
#                 # if vid_name in ['ISI_-1_sep_0_v1.mp4', 'ISI_-1_sep_0_v2.mp4']:
#                 #     print(f"******just do this one: {vid_name}")
#
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
#                   # # todo: in future don't bother with all_full_frames dir.  Just save under filename.
#                   #   all_full_frames_dir = os.path.join(this_vid_image_dir, 'all_full_frames')
#                   #   if not os.path.isdir(all_full_frames_dir):
#                   #       os.makedirs(all_full_frames_dir)
#
#                     image_save_path = os.path.join(os.path.join(this_vid_image_dir, image_name))
#
#                     # rotate image
#                     image = cv2.rotate(image, cv2.ROTATE_180)
#
#                     # don't extract frames < 30 or > 405 as they aren't at 960Hz
#                     if 30 < count < 405:
#                         cv2.imwrite(image_save_path, image)  # save frame as JPG file
#                         print(f'Saved image: {image_save_path}')
#                     else:
#                         print(f'\tNot saving this image: {image_save_path}')
#
#                     success, image = vidcap.read()
#                     count += 1

'''part 1b
loop through video files to make an Excel doc to populate with frame numbers for when probes appear

It starts with lists of sep and isi vals, its a bit convoluted, but it reflects the order of priority for analysis.
the main conditions I need are these in lists 1, 5 and 4. If there is time, then do 2 and 3.
'''

# sep_vals = [0, 1, 2, 3, 6, 18, 99, 400, 800, 2400]
# ISI_vals = [-1, 0, 2, 3, 4, 6, 9, 12, 24]
# versions = [1, 2, 3, 4, 5, 6, 7, 8]
# # version = [1, 2, 3, 4, 5, 6, 7, 8]
#
# # # loop through theses, and through all versions.
# details_list = []
# # for idx, (isi, sep) in enumerate(zip(ISI_list, sep_list)):
# for sep in sep_vals:
#     for isi in ISI_vals:
#         for ver in versions:
#             vid_name = f'ISI_{isi}_sep_{sep}_v{ver}.mp4'
#             vid_path = os.path.join(videos_dir, vid_name)
#             # print(vid_path)
#             if not os.path.isfile(vid_path):
#                 print(f'\n\t\t\tmissing: {vid_name}')
#             else:
#                 print(f'\n\tfound: {vid_name}')
#                 details_list.append([isi, sep, ver, vid_name])
#
# excel_headers = ['isi', 'sep', 'version', 'filename', 'comments', 'first_pr1_fr', 'analyse', 'first_pr2_fr', 'fix0', 'fix1', 'pr1_down', 'pr1_right', 'pr2_down', 'pr2_right']
# excel_df = pd.DataFrame(details_list, columns=excel_headers)
# print(f"excel_df\n{excel_df}")
# excel_name = 'frame_and_bbox_details.xlsx'
# save_excel_path = os.path.join(images_dir, excel_name)
# excel_df.to_excel(save_excel_path, engine='openpyxl', index=False)

'''
I need to manually look through the frames to get the frame number where the 
first probe appears and put it into 'first_pr1_fr' on the Excel doc. 
If there is a frame counter this should be on frame 25.  
I first looked by eye to see when the probe appeared. 
Once I had made plots I tweaked the first_pr1_fr number so that:
the value was rising at first_pr1_fr, but not rising at first_pr1_fr-1. 

I then add this code to the column marked 'first_pr2_fr':
=IF(A2=-1, F2, IF(A2=0, F2+8, IF(A2=2, F2+16, IF(A2=3, F2+20, IF(A2=4, F2+24, IF(A2=6, F2+32, IF(A2=9, F2+44, IF(A2=12, F2+56, IF(A2=24, F2+104, 0)))))))))
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
pr1_down: fix0-sep
pr1_right: fix1+sep-2  (note this is different here for radial compared to tangent)
I'm now setting p2 in relation to fixation, not p1.
pr2_down: fix0+sep  (note this is different here for radial compared to tangent)
pr2_right: fix1-sep (note this is different here for radial compared to tangent)

The copy & paste values once I have confirmed the correct positions (so formulas don't update them

NOTE: For some reason, y is first (vertical, down), then x (horiontal, right)
'''

# isi = 2
# sep_name = '5_cont'
# sep = 5
# ver = 1
# first_frame = 201
# # todo: add back in first frame + 4
# show_frame = first_frame + 4
# # show_frame = first_frame
#
# '''ROI_size was 5 for May but now using 10 so whole probe in with grey border'''
# ROI_size = 10  # 10  # 5
# enlarge_scale = 10  # 10  # 10
#
# # # [rows/vertical, cols/horizontal]  0, 0 is top left corner of screen
# fixation = 228, 909  # top-left corner of cropped image
# print(f'fixation: ({fixation[0]}, {fixation[1]})')
#
# if sep >= 99:
#     p1_tl = fixation[0], fixation[1]
# else:
#     # p1_tl = fixation[0]-sep, fixation[1]-sep  # works for tangent
#     if 'exp' in sep_name:
#         p1_tl = fixation[0]-sep, fixation[1]+sep - 2  # use this for radial
#     else:
#         p1_tl = fixation[0] + sep, fixation[1] - sep  # use this for radial
#
# # cond_dir = rf"ISI_{isi}_sep_{sep}_v{ver}\all_full_frames"
# cond_dir = rf"ISI_{isi}_sep_{sep_name}_v{ver}"
#
# image_name = f"ISI_{isi}_sep_{sep_name}_v{ver}_fr{show_frame}.jpg"
#
# image_path = os.path.join(images_dir, cond_dir, image_name)
# print(image_path)
# print(os.path.isfile(image_path))
#
# # load image used for bbox1
# # todo: turn off greyscale?
# # img = cv2.imread(image_path)
# img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#
# print(f'size: {img.shape}type: {type(img)}')
#
# # [rows, columns]
# crop_size = 190
# crop = img[p1_tl[0]:p1_tl[0]+crop_size, p1_tl[1]:p1_tl[1]+crop_size]
# cv2.imshow('original', img)
# cv2.imshow('cropped', crop)
#
# # extract bounding box for first probe from image
# p1_box = img[p1_tl[0]: p1_tl[0]+ROI_size, p1_tl[1]: p1_tl[1]+ROI_size]
# print(f'p1_box: ({p1_tl[0]}, {p1_tl[1]})')
#
# '''delete'''
# p1_mean = np.mean(p1_box)
# p1_max = np.max(p1_box)
# print(f'p1_box array: \n{p1_box}')
# print(f'p1_mean: {p1_mean}')
# print(f'p1_max: {p1_max}')
# '''delete'''
#
# # display an enlarged image to check bounding box
# w = int(p1_box.shape[1] * enlarge_scale)
# h = int(p1_box.shape[0] * enlarge_scale)
# big_p1_box = cv2.resize(p1_box, (w, h))
# cv2.imshow('big_p1_box', big_p1_box)
#
#
# # I am going to open p2_box from an image from a different frame to p1_box depending on the isi.
# # with p2_show_frame = isi*4+8
# if isi == -1:
#     p2_show_frame = show_frame
# elif sep > 100:
#     p2_show_frame = show_frame
# else:
#     # todo: add back in isi*4+8
#     p2_show_frame = show_frame + (isi*4+8)
#     # p2_show_frame = show_frame
#
# print(f"isi: {isi}, show_frame: {show_frame}, p2_show_frame: {p2_show_frame}")
# image2_name = f"ISI_{isi}_sep_{sep_name}_v{ver}_fr{p2_show_frame}.jpg"
# image2_path = os.path.join(images_dir, cond_dir, image2_name)
#
# # todo: turn off greyscale?
# # img2 = cv2.imread(image2_path)
# img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
#
# if sep >= 99:
#     p2_adjustment = int(ROI_size / 2)
# else:
#     # p2_adjustment = int(ROI_size/2) + sep  # works for tangent
#     p2_adjustment = int(ROI_size/2)  # works for radial
# print(f'p2_adjustment: {p2_adjustment}')
#
# if 'exp' in sep_name:
#     p2_tl = fixation[0] + sep, fixation[1] - sep  # use this for radial
# else:
#     p2_tl = fixation[0] - sep, fixation[1] + sep - 2  # use this for radial
#
# # display a crop of the 2nd probe image
# crop2 = img2[p2_tl[0]:p2_tl[0]+crop_size, p2_tl[1]:p2_tl[1]+crop_size]
# # cv2.imshow('img2', img2)
# cv2.imshow('crop2', crop2)
# p2_box = img2[p2_tl[0]: p2_tl[0]+ROI_size, p2_tl[1]: p2_tl[1]+ROI_size]
#
# # p2_box = img[p2_tl[0]: p2_tl[0]+ROI_size, p2_tl[1]: p2_tl[1]+ROI_size]
# print(f'p2_box: ({p2_tl[0]}, {p2_tl[1]})')
#
# '''delete'''
# p2_mean = np.mean(p2_box)
# p2_max = np.max(p2_box)
# print(f'p2_box array: \n{p2_box}')
# print(f'p2_mean: {p2_mean}')
# print(f'p2_max: {p2_max}')
# '''delete'''
#
# # display enlarged version of 2nd probe bounding box
# big_p2_box = cv2.resize(p2_box, (w, h))
# cv2.imshow('big_p2_box', big_p2_box)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

'''
Part 3, loop through Excel,
load images (and convert to grey for simplicity)
Get pixel values from the bounding boxes
save the mean and max values to new csv
'''

# '''delete'''
# # load Excel sheet with frame numbers where probes appear and x, y co-ordinates for cropping probes.
# excel_path = os.path.join(images_dir, 'frame_and_bbox_details.xlsx')
# excel_path = os.path.normpath(excel_path)
# print(f'excel_path:\n{excel_path}')
#
# bb_box_df = pd.read_excel(excel_path, sheet_name='Sheet1', engine='openpyxl')
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
#         isi = row['isi']
#         sep_name = row['sep']
#         sep = int(sep_name[0])
#         probes_dir = sep_name[2:]
#         ver = row['version']
#         filename = row['filename']
#
#         if filename == 'ISI_2_sep_5_cont_v1.mp4':
#             print(f'\n{filename}, ISI_{isi}, sep_{sep_name}, ver: {ver}')
#
#             pr1_frame = int(row['first_pr1_fr'])
#             pr2_frame = int(row['first_pr2_fr'])
#
#             # set to process 100 frames (or 140 for isi24)
#             # todo: Check I don't take frames <30 or >405 as these are not at 960
#             from_fr = pr1_frame-8
#             to_fr = from_fr+100
#             if isi == 24:
#                 to_fr = from_fr+140
#             if sep == 2400:
#                 to_fr = from_fr+140
#             if to_fr > 435:
#                 to_fr = 435
#             print(f'from_fr: {from_fr} : to_fr: {to_fr}')
#
#             # probe bounding box is 5x5 pixels
#             ROI_size = 10  # 5 for may22
#
#             pr1_down = int(row['pr1_down'])
#             pr1_right = int(row['pr1_right'])
#
#             if sep < 99:
#                 pr2_down = int(row['pr2_down'])
#                 pr2_right = int(row['pr2_right'])
#             else:
#                 pr2_down = np.nan
#                 pr2_right = np.nan
#             print(f'pr1: ({pr1_down}, {pr1_right}), pr2: ({pr2_down}, {pr2_right})')
#
#             # loop through each of the frames
#             for idx, frame in enumerate(list(range(from_fr, to_fr))):
#                 print(f'\n\nframe: {frame}')
#
#                 # load image of this frame
#                 # cond_dir = rf"ISI_{isi}_sep_{sep_name}_v{ver}\all_full_frames"
#                 cond_dir = rf"ISI_{isi}_sep_{sep_name}_v{ver}"
#                 image_name = f"ISI_{isi}_sep_{sep_name}_v{ver}_fr{frame}.jpg"
#                 image_path = os.path.join(images_dir, cond_dir, image_name)
#
#                 gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#
#                 p1_box = gray_img[pr1_down: pr1_down+ROI_size, pr1_right: pr1_right+ROI_size]
#                 # print(f'p1_box: {type(p1_box)}')
#                 p1_mean = np.mean(p1_box)
#                 p1_max = np.max(p1_box)
#                 print(f'p1_box array: \n{p1_box}')
#                 print(f'p1_mean: {p1_mean}')
#                 print(f'p1_max: {p1_max}')
#
#                 if sep < 99:
#                     p2_box = gray_img[pr2_down: pr2_down+ROI_size, pr2_right: pr2_right+ROI_size]
#                     p2_mean = np.mean(p2_box)
#                     p2_max = np.max(p2_box)
#
#                     print(f'p2_box array: \n{p2_box}')
#                     print(f'p2_mean: {p2_mean}')
#                     print(f'p2_max: {p2_max}')
#
#                     joint_mean = np.mean([p1_mean, p2_mean])
#
#                     # joint_max = np.max([p1_max, p2_max])
#                     if p1_max > p2_max:
#                         joint_max = p1_max
#                     else:
#                         joint_max = p2_max
#                     '''strange, the max scores are not saving to the csv as shown here.
#                     I'm converting to str to try to avoid this.'''
#
#                 else:
#                     p2_box = np.nan
#                     p2_mean = np.nan
#                     p2_max = np.nan
#
#                     joint_mean = p1_mean
#                     joint_max = p1_max
#
#                 print(f'joint_mean: {joint_mean}')
#                 print(f'joint_max: {joint_max}')
#
# '''delete'''

# load Excel sheet with frame numbers where probes appear and x, y co-ordinates for cropping probes.
excel_path = os.path.join(images_dir, 'frame_and_bbox_details.xlsx')
excel_path = os.path.normpath(excel_path)
print(f'excel_path:\n{excel_path}')

bb_box_df = pd.read_excel(excel_path, sheet_name='Sheet1', engine='openpyxl')
print(f'bb_box_df:\n{bb_box_df}')

rows, cols = bb_box_df.shape
print(f'rows: {rows}, cols: {cols}')
print(f'headers: {bb_box_df.columns.to_list()}')

empty_list = []

# loop through each row for Excel, e.g., frames from each video/condition
for index, row in bb_box_df.iterrows():
    analyse_this = row['analyse']
    if analyse_this == 1:
        isi = row['isi']
        sep_name = row['sep']
        sep = int(sep_name[0])
        probes_dir = sep_name[2:]
        ver = row['version']
        filename = row['filename']
        print(f'\n{filename}, ISI_{isi}, sep_{sep_name}, ver: {ver}')

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

        pr1_down = int(row['pr1_down'])
        pr1_right = int(row['pr1_right'])

        if sep < 99:
            pr2_down = int(row['pr2_down'])
            pr2_right = int(row['pr2_right'])
        else:
            pr2_down = np.nan
            pr2_right = np.nan
        print(f'pr1: ({pr1_down}, {pr1_right}), pr2: ({pr2_down}, {pr2_right})')

        # loop through each of the frames
        for idx, frame in enumerate(list(range(from_fr, to_fr))):

            # load image of this frame
            # cond_dir = rf"ISI_{isi}_sep_{sep_name}_v{ver}\all_full_frames"
            cond_dir = rf"ISI_{isi}_sep_{sep_name}_v{ver}"
            image_name = f"ISI_{isi}_sep_{sep_name}_v{ver}_fr{frame}.jpg"
            image_path = os.path.join(images_dir, cond_dir, image_name)

            gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            p1_box = gray_img[pr1_down: pr1_down+ROI_size, pr1_right: pr1_right+ROI_size]
            # print(f'p1_box: {type(p1_box)}')
            p1_mean = np.mean(p1_box)
            p1_max = np.max(p1_box)
            # print(f'p1_box array: \n{p1_box}')
            # print(f'p1_mean: {p1_mean}')
            # print(f'p1_max: {p1_max}')
            # print(f'p1_max: {type(p1_max)}')

            if sep < 99:
                p2_box = gray_img[pr2_down: pr2_down+ROI_size, pr2_right: pr2_right+ROI_size]
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
            save_row = [filename, isi, sep, sep_name, probes_dir, ver, idx, frame,
                        pr1_frame, pr1_down, pr1_right, p1_mean, str(p1_max),
                        pr2_frame, pr2_down, pr2_right, p2_mean, str(p2_max),
                        joint_mean, str(joint_max)]
            empty_list.append(save_row)
            # print(f'{frame}: {save_row}')

print(f'empty_list shape: {np.shape(empty_list)}')
print(empty_list)
results_df = pd.DataFrame(data=empty_list,
                          columns=['filename', 'isi', 'sep', 'sep_name', 'probes_dir', 'version', 'idx', 'frame',
                                   'pr1_frame', 'pr1_down', 'pr1_right',
                                   'p1_mean', 'p1_max',
                                   'pr2_frame', 'pr2_down', 'pr2_right',
                                   'p2_mean', 'p2_max',
                                   'joint_mean', 'joint_max'])

results_path = os.path.join(images_dir, 'monitor_refresh_results_Jan23.csv')
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
#
# load results csv
# results_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\project_stuff\monitor_calibration\ASUS\monitor_refresh_results_Oct22.csv"
results_path = os.path.join(images_dir, 'monitor_refresh_results_Jan23.csv')
results_path = os.path.normpath(results_path)

results_df = pd.read_csv(results_path)
print(f'results_df:\n{results_df}')

# print(f'headers:\n{list(results_df.columns)}')

'''
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

probe2_dict = {-1: 0, 0: 8, 2: 16, 3: 20, 4: 24, 6: 32, 9: 44, 12: 56, 24: 140,
               # 'sep': {400: 16, 800: 32, 2400: 96}
               }


# loop through ALL conditions
cond_ver_list = list(results_df['filename'].unique())
print(f'cond_ver_list (n={len(cond_ver_list)}):\n{cond_ver_list}')
# cond_ver_list = cond_ver_list[:1]
# print(f'cond_ver_list (n={len(cond_ver_list)}):\n{cond_ver_list}')
cond_list = [i[:-5] for i in cond_ver_list]
cond_list = set(cond_list)
print(f'cond_list (n={len(cond_list)}):\n{cond_list}')

# x tick labels
x_tick_label_start = -2
x_tick_locs = np.arange(0, 100, 8)
x_tick_labels = np.arange(x_tick_label_start, 25 + x_tick_label_start, 2)
print(f'\nx_tick_locs: {x_tick_locs}')
print(f'x_tick_labels: {x_tick_labels}')

# for cond_ver_name in conds_to_use:
for cond_ver_name in cond_ver_list:

    print(f'\ncond_ver_name: {cond_ver_name}')

    split_txt = cond_ver_name.split("_")
    isi_val = int(split_txt[1])
    isi_name = f"ISI{isi_val}"
    sep_val = int(split_txt[3])
    probes_dir = split_txt[4]
    sep_name = f"sep{sep_val}"
    # ver_name = split_txt[4][:2]
    ver_name = split_txt[5][:2]
    print(f'split_txt: {split_txt}')
    print(f'isi_val: {isi_val}')
    print(f'isi_name: {isi_name}')
    print(f'sep_val: {sep_val}')
    print(f'sep_name: {sep_name}')
    print(f'probes_dir: {probes_dir}')
    print(f'ver_name: {ver_name}')


    '''insert vertical lines marking start and finish of each probe.
    First probe should appear at frame 8 (@960)'''
    vline_pr1_on = 8
    vline_pr1_off = vline_pr1_on + 8

    if sep_val == 400:
        vline_pr1_off = vline_pr1_on + 16
    if sep_val == 800:
        vline_pr1_off = vline_pr1_on + 32
    if sep_val == 2400:
        vline_pr1_off = vline_pr1_on + 96

    vline_pr2_on = 8 + probe2_dict[isi_val]
    vline_pr2_off = vline_pr2_on + 8

    cond_df = results_df[results_df['filename'] == cond_ver_name]
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
    ax.set_xlabel('frames @ 240Hz')
    ax.set_ylabel('pixel values (0:255')

    # probe on/off markers
    plt.axvline(x=vline_pr1_on, color='grey', linestyle='--')
    plt.axvline(x=vline_pr1_off, color='grey', linestyle='--')
    if sep_val not in [99, 400, 800, 2400]:
        plt.axvline(x=vline_pr2_on, color='grey', linestyle='-.')
        plt.axvline(x=vline_pr2_off, color='grey', linestyle='-.')

    ax.set_xticks(x_tick_locs)
    ax.set_xticklabels(x_tick_labels)
    # add shaded background for frames at 240
    x_bg = np.arange(0, 100, 4)
    for x0, x1 in zip(x_bg[::2], x_bg[1::2]):
        plt.axvspan(x0, x1, color='black', alpha=0.05, zorder=0, linewidth=None)

    plt.title(f'{cond_ver_name}: mean luminance')
    # fig_dir = rf"C:\Users\sapnm4\OneDrive - Cardiff University\Pictures\mon_cali_images_Oct22\cond_figs\mean_lum"
    fig_dir = os.path.join(images_dir, 'cond_figs', 'mean_lum')
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)
    fig_savename = f'{cond_ver_name}_mean_lum.png'
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
    ax.set_xlabel('frames @ 240Hz')
    ax.set_ylabel('pixel values (0:255')

    # probe on/off markers
    plt.axvline(x=vline_pr1_on, color='grey', linestyle='--')
    plt.axvline(x=vline_pr1_off, color='grey', linestyle='--')
    if sep_val not in [99, 400, 800, 2400]:
        plt.axvline(x=vline_pr2_on, color='grey', linestyle='-.')
        plt.axvline(x=vline_pr2_off, color='grey', linestyle='-.')

    ax.set_xticks(x_tick_locs)
    ax.set_xticklabels(x_tick_labels)
    # add shaded background for frames at 240
    x_bg = np.arange(0, 100, 4)
    for x0, x1 in zip(x_bg[::2], x_bg[1::2]):
        plt.axvspan(x0, x1, color='black', alpha=0.05, zorder=0, linewidth=None)

    plt.title(f'{cond_ver_name}: max luminance')
    # fig_dir = rf"C:\Users\sapnm4\OneDrive - Cardiff University\Pictures\mon_cali_images_Oct22\cond_figs\max_lum"
    fig_dir = os.path.join(images_dir, 'cond_figs', 'max_lum')
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)
    fig_savename = f'{cond_ver_name}_max_lum.png'
    fig_path = os.path.join(fig_dir, fig_savename)
    plt.savefig(fig_path)
    plt.show()
    plt.close()



print('probes_dir figs')
'''
I will make a plot for each ISI comparing all exp plots (blue) with cont plots (orange).
1. loop through ISIs, and make df for each ISI.
2. loop through probes_dir (exp v cont).
3. plot all exp conditions onto plots in one colour.
4. plot all cont cont conditions onto plots in another colour.'''

probe_dir_palette = {'exp': 'tab:blue', 'cont': 'tab:orange'}


for isi in ISI_vals:

    isi_df = results_df[results_df['isi'] == isi]
    print(f"\n\nisi: {isi}\n{isi_df}\n")

    '''insert vertical lines marking start and finish of each probe.
    First probe should appear at frame 8 (@960)'''
    vline_pr1_on = 8
    vline_pr1_off = vline_pr1_on + 8

    vline_pr2_on = 8 + probe2_dict[isi]
    vline_pr2_off = vline_pr2_on + 8

    '''mean lum plots'''
    # 1. plots by isi - joint means, compare exp and cont - all lines
    fig, ax = plt.subplots(figsize=(6, 6))

    sns.lineplot(data=isi_df, x='idx', y='joint_mean', hue='probes_dir',
                 units='filename', estimator=None, palette=probe_dir_palette,
                 ax=ax)
    ax.set_xlabel('frames @ 240Hz')
    ax.set_ylabel('pixel values (0:255')


    # probe on/off markers
    plt.axvline(x=vline_pr1_on, color='grey', linestyle='--')
    plt.axvline(x=vline_pr1_off, color='grey', linestyle='--')
    plt.axvline(x=vline_pr2_on, color='grey', linestyle='-.')
    plt.axvline(x=vline_pr2_off, color='grey', linestyle='-.')

    ax.set_xticks(x_tick_locs)
    ax.set_xticklabels(x_tick_labels)

    # add shaded background for frames at 240
    x_bg = np.arange(0, 108, 4)
    for x0, x1 in zip(x_bg[::2], x_bg[1::2]):
        plt.axvspan(x0, x1, color='black', alpha=0.05, zorder=0, linewidth=None)

    plt.title(f'ISI_{isi}: mean luminance, all files')
    # fig_dir = rf"C:\Users\sapnm4\OneDrive - Cardiff University\Pictures\mon_cali_images_Oct22\isi_figs\mean_lum"
    fig_dir = os.path.join(images_dir, 'probes_dir_figs_all', 'mean_lum')
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)
    fig_savename = f'ISI_{isi}_compare_all_mean_lum.png'
    fig_path = os.path.join(fig_dir, fig_savename)
    plt.savefig(fig_path)
    plt.show()
    plt.close()



    # 2. plots by isi - joint_mean, compare exp and cont - mean and errors
    fig, ax = plt.subplots(figsize=(6, 6))

    sns.lineplot(data=isi_df, x='idx', y='joint_mean', hue='probes_dir',
                 palette=probe_dir_palette,
                 # units='filename', estimator=None,
                 ax=ax)
    ax.set_xlabel('frames @ 240Hz')
    ax.set_ylabel('pixel values (0:255')

    # probe on/off markers
    plt.axvline(x=vline_pr1_on, color='grey', linestyle='--')
    plt.axvline(x=vline_pr1_off, color='grey', linestyle='--')
    plt.axvline(x=vline_pr2_on, color='grey', linestyle='-.')
    plt.axvline(x=vline_pr2_off, color='grey', linestyle='-.')

    ax.set_xticks(x_tick_locs)
    ax.set_xticklabels(x_tick_labels)

    # add shaded background for frames at 240
    x_bg = np.arange(0, 108, 4)
    for x0, x1 in zip(x_bg[::2], x_bg[1::2]):
        plt.axvspan(x0, x1, color='black', alpha=0.05, zorder=0, linewidth=None)

    plt.title(f'ISI_{isi}: mean luminance, mean and error of conditions')
    # fig_dir = rf"C:\Users\sapnm4\OneDrive - Cardiff University\Pictures\mon_cali_images_Oct22\isi_figs\mean_lum"
    fig_dir = os.path.join(images_dir, 'probes_dir_figs_ave', 'mean_lum')
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)
    fig_savename = f'ISI_{isi}_compare_ave_mean_lum.png'
    fig_path = os.path.join(fig_dir, fig_savename)
    plt.savefig(fig_path)
    plt.show()
    plt.close()



    '''max lum'''
    # 3. plots by isi - joint means, compare exp and cont - all lines
    fig, ax = plt.subplots(figsize=(6, 6))

    sns.lineplot(data=isi_df, x='idx', y='joint_max', hue='probes_dir',
                 palette=probe_dir_palette,
                 units='filename', estimator=None,
                 ax=ax)
    ax.set_xlabel('frames @ 240Hz')
    ax.set_ylabel('pixel values (0:255')

    # probe on/off markers
    plt.axvline(x=vline_pr1_on, color='grey', linestyle='--')
    plt.axvline(x=vline_pr1_off, color='grey', linestyle='--')
    plt.axvline(x=vline_pr2_on, color='grey', linestyle='-.')
    plt.axvline(x=vline_pr2_off, color='grey', linestyle='-.')

    ax.set_xticks(x_tick_locs)
    ax.set_xticklabels(x_tick_labels)

    # add shaded background for frames at 240
    x_bg = np.arange(0, 108, 4)
    for x0, x1 in zip(x_bg[::2], x_bg[1::2]):
        plt.axvspan(x0, x1, color='black', alpha=0.05, zorder=0, linewidth=None)

    plt.title(f'ISI_{isi}: max luminance, all files')
    # fig_dir = rf"C:\Users\sapnm4\OneDrive - Cardiff University\Pictures\mon_cali_images_Oct22\isi_figs\mean_lum"
    fig_dir = os.path.join(images_dir, 'probes_dir_figs_all', 'max_lum')
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)
    fig_savename = f'ISI_{isi}_compare_all_max_lum.png'
    fig_path = os.path.join(fig_dir, fig_savename)
    plt.savefig(fig_path)
    plt.show()
    plt.close()


    # 4. plots by isi - joint_mean, compare exp and cont - mean and errors
    fig, ax = plt.subplots(figsize=(6, 6))

    sns.lineplot(data=isi_df, x='idx', y='joint_max', hue='probes_dir',
                 palette=probe_dir_palette,
                 # units='filename', estimator=None,
                 ax=ax)
    ax.set_xlabel('frames @ 240Hz')
    ax.set_ylabel('pixel values (0:255')

    # probe on/off markers
    plt.axvline(x=vline_pr1_on, color='grey', linestyle='--')
    plt.axvline(x=vline_pr1_off, color='grey', linestyle='--')
    plt.axvline(x=vline_pr2_on, color='grey', linestyle='-.')
    plt.axvline(x=vline_pr2_off, color='grey', linestyle='-.')

    ax.set_xticks(x_tick_locs)
    ax.set_xticklabels(x_tick_labels)

    # add shaded background for frames at 240
    x_bg = np.arange(0, 108, 4)
    for x0, x1 in zip(x_bg[::2], x_bg[1::2]):
        plt.axvspan(x0, x1, color='black', alpha=0.05, zorder=0, linewidth=None)

    plt.title(f'ISI_{isi}: max luminance, mean and error of conditions')
    # fig_dir = rf"C:\Users\sapnm4\OneDrive - Cardiff University\Pictures\mon_cali_images_Oct22\isi_figs\mean_lum"
    fig_dir = os.path.join(images_dir, 'probes_dir_figs_ave', 'max_lum')
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)
    fig_savename = f'ISI_{isi}_compare_ave_max_lum.png'
    fig_path = os.path.join(fig_dir, fig_savename)
    plt.savefig(fig_path)
    plt.show()
    plt.close()



'''
these plots just do one per ISI
'''
# print('ISI figs')
# # just take one version of each cond for now.
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
# probes_dir_list = list(results_df['probes_dir'].unique())
#
# isi_list = list(results_df['isi'].unique())
#
# # isi_list = isi_list + ['1pr']
#
# print(f'isi_list:\n{isi_list}')
#
# for probes_dir in probes_dir_list:
#
#     probes_dir_df = key_results_df[key_results_df['probes_dir'] == probes_dir]
#
#     for isi in isi_list:
#
#         print(f"isi: {isi}")
#         # isi_df = results_df[results_df['isi'] == isi]
#         # isi_df = key_results_df[key_results_df['isi'] == isi]
#         isi_df = probes_dir_df[probes_dir_df['isi'] == isi]
#
#         vline_pr1_on = 8
#         vline_pr1_off = vline_pr1_on + 8
#         if isi not in [-1, '1pr']:
#             vline_pr2_on = 8 + probe2_dict[isi]
#             vline_pr2_off = vline_pr2_on + 8
#
#         if isi == 0:
#             # long probes are labelled sep 400, 800 etc shouldnt be here, so remove
#             # isi_df = key_results_df.loc[((key_results_df['isi'] == isi) & (key_results_df['sep'] < 25))]
#             isi_df = probes_dir_df.loc[((probes_dir_df['isi'] == isi) & (probes_dir_df['sep'] < 25))]
#
#         if isi == '1pr':
#             isi_df = key_results_df.loc[key_results_df['sep'] > 25]
#             x_tick_locs = np.arange(0, 148, 8)
#             x_tick_labels = np.arange(x_tick_label_start, 37 + x_tick_label_start, 2)
#
#
#         print(f'isi_df:\n{isi_df}')
#
#         # 3. by isi - joint means
#         fig, ax = plt.subplots(figsize=(6, 6))
#         sns.lineplot(data=isi_df, x='idx', y='joint_mean', hue='filename', ax=ax)
#         ax.set_xlabel('frames @ 240Hz')
#         ax.set_ylabel('pixel values (0:255')
#
#
#         # probe on/off markers
#         plt.axvline(x=vline_pr1_on, color='grey', linestyle='--')
#         plt.axvline(x=vline_pr1_off, color='grey', linestyle='--')
#         if isi == '1pr':
#             plt.axvline(x=vline_pr1_on + 16, color='grey', linestyle='-.')
#             plt.axvline(x=vline_pr1_on + 32, color='grey', linestyle='-.')
#             plt.axvline(x=vline_pr1_on + 96, color='grey', linestyle='-.')
#         else:
#             plt.axvline(x=vline_pr2_on, color='grey', linestyle='-.')
#             plt.axvline(x=vline_pr2_off, color='grey', linestyle='-.')
#
#         ax.set_xticks(x_tick_locs)
#         ax.set_xticklabels(x_tick_labels)
#
#         # add shaded background for frames at 240
#         x_bg = np.arange(0, 108, 4)
#         if isi == '1pr':
#             x_bg = np.arange(0, 148, 4)
#         for x0, x1 in zip(x_bg[::2], x_bg[1::2]):
#             plt.axvspan(x0, x1, color='black', alpha=0.05, zorder=0, linewidth=None)
#
#         plt.title(f'{probes_dir}, {isi}: mean luminance')
#         # fig_dir = rf"C:\Users\sapnm4\OneDrive - Cardiff University\Pictures\mon_cali_images_Oct22\isi_figs\mean_lum"
#         fig_dir = os.path.join(images_dir, 'isi_figs', 'mean_lum')
#
#         if not os.path.isdir(fig_dir):
#             os.makedirs(fig_dir)
#         fig_savename = f'{probes_dir}_ISI_{isi}_mean_lum.png'
#         fig_path = os.path.join(fig_dir, fig_savename)
#         plt.savefig(fig_path)
#         plt.show()
#         plt.close()
#
#         # 4. by isi - joint max
#         fig, ax = plt.subplots(figsize=(6, 6))
#         sns.lineplot(data=isi_df, x='idx', y='joint_max', hue='filename', ax=ax)
#         ax.set_xlabel('frames @ 240Hz')
#         ax.set_ylabel('pixel values (0:255')
#         plt.title(f'{probes_dir}, {isi}: max luminance')
#
#         # probe on/off markers
#         plt.axvline(x=vline_pr1_on, color='grey', linestyle='--')
#         plt.axvline(x=vline_pr1_off, color='grey', linestyle='--')
#         if isi == '1pr':
#             plt.axvline(x=vline_pr1_on + 16, color='grey', linestyle='-.')
#             plt.axvline(x=vline_pr1_on + 32, color='grey', linestyle='-.')
#             plt.axvline(x=vline_pr1_on + 96, color='grey', linestyle='-.')
#         else:
#             plt.axvline(x=vline_pr2_on, color='grey', linestyle='-.')
#             plt.axvline(x=vline_pr2_off, color='grey', linestyle='-.')
#
#         ax.set_xticks(x_tick_locs)
#         ax.set_xticklabels(x_tick_labels)
#         # add shaded background for frames at 240
#         x_bg = np.arange(0, 100, 4)
#         if isi == '1pr':
#             x_bg = np.arange(0, 148, 4)
#         for x0, x1 in zip(x_bg[::2], x_bg[1::2]):
#             plt.axvspan(x0, x1, color='black', alpha=0.05, zorder=0, linewidth=None)
#
#         # fig_dir = rf"C:\Users\sapnm4\OneDrive - Cardiff University\Pictures\mon_cali_images_Oct22\isi_figs\max_lum"
#         fig_dir = os.path.join(images_dir, 'isi_figs', 'max_lum')
#         if not os.path.isdir(fig_dir):
#             os.makedirs(fig_dir)
#         fig_savename = f'{probes_dir}_ISI_{isi}_max_lum.png'
#         fig_path = os.path.join(fig_dir, fig_savename)
#         plt.savefig(fig_path)
#         plt.show()
#         plt.close()

print('finished plotting monitor calibration')
