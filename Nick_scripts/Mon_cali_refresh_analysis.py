import cv2
import os


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

isi = 2
sep = 2
frame = 159
ROI_size = 5  # 5
enlarge_scale = 10

p1_tl = 171, 435
# p1_tl = 172, 435

p2_tl = 168, 440


root_dir = r"C:\Users\sapnm4\OneDrive - Cardiff University\Pictures\mon_cali_images"
cond_dir = rf"ISI{isi}_sep{sep}\all_full_frames"
image_name = f"ISI{isi}_sep{sep}_fr{frame}.jpg"

image_path = os.path.join(root_dir, cond_dir, image_name)
print(image_path)
print(os.path.isfile(image_path))

img = cv2.imread(image_path)

# Check the type of read image
print(f'size: {img.shape}type: {type(img)}')

# Display the image
# cv2.imshow('image', img)

# [rows, columns]
fixation = 20, 290
crop_size = 190
crop = img[fixation[0]:fixation[0]+crop_size, fixation[1]:fixation[1]+crop_size]
print(f'cropped: ({fixation[0]}: {fixation[0]+crop_size}, {fixation[1]}: {fixation[1]+crop_size})')
cv2.imshow('original', img)
cv2.imshow('cropped', crop)


p1_box = img[p1_tl[0]: p1_tl[0]+ROI_size, p1_tl[1]: p1_tl[1]+ROI_size]
print(f'p1_box: ({p1_tl[0]}: {p1_tl[0]+ROI_size}, {p1_tl[1]}: {p1_tl[1]+ROI_size})')
# cv2.imshow('p1_box', p1_box)

w = int(p1_box.shape[1] * enlarge_scale)
h = int(p1_box.shape[0] * enlarge_scale)
big_p1_box = cv2.resize(p1_box, (w, h))
cv2.imshow('big_p1_box', big_p1_box)

p2_box = img[p2_tl[0]: p2_tl[0]+ROI_size, p2_tl[1]: p2_tl[1]+ROI_size]
print(f'p2_box: ({p2_tl[0]}: {p2_tl[0]+ROI_size}, {p2_tl[1]}: {p2_tl[1]+ROI_size})')

print(f'p2_box array: \n{p2_box}')


# cv2.imshow('p2_box', p2_box)
w = int(p2_box.shape[1] * enlarge_scale)
h = int(p2_box.shape[0] * enlarge_scale)
big_p2_box = cv2.resize(p2_box, (w, h))
cv2.imshow('big_p2_box', big_p2_box)

cv2.waitKey(0)
cv2.destroyAllWindows()

'''
Once I have made an xlsx document storing the first useful frame of each video I will load it here.

I then also need to visualise frames and be able to try cropping ROIs and displaying them on the screen.

Hopefully I can simply store a single tuple for the ROI (e.g., top-left corner)

add something to convert images to greyscale - perhaps when saving bounding boxes.
Or infact
1. convert bounding box to grey - save image
2. at same time calculate mean value of grey image and save that to csv
'''

