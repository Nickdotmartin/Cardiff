from __future__ import division

import psychopy

'''
The lab machine uses 2021.2.3, but this doesn't work on my laptop.
ImportError: cannot import name '_vmTesting' from 'psychopy.tests' (unknown location)
 However, I can test on an older version (e.g., 2021.2.2) which does work.
psychopy.useVersion('2021.2.3')'''
psychopy.useVersion('2021.2.2')  # works

from psychopy import gui, visual, core, data, event, monitors

import logging
import os
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from datetime import datetime
from math import tan, sqrt
from PsychoPy_tools import get_pixel_mm_deg_values
from kestenSTmaxVal import Staircase
from exp1a_psignifit_analysis import fig_colours

from psychopy import __version__ as psychopy_version
print(f"PsychoPy_version: {psychopy_version}")



'''
Updated version of radial flow experiment.
Has variable fixation time, records frame rate, 
Can have radial or tangential probes.
Can vary the preliminary motion duration.
Has same colour space and background colour as exp1.
Updated wrap_depth_vals (WrapPoints) function.  
'''


def wrap_depth_vals(depth_arr, min_depth, max_depth):
    """
    function to take an array (depth_arr) and adjust any values below min_depth
    or above max_depth with +/- (max_depth-min_depth)
    :param depth_arr: np.random.rand(nDots) array giving depth values for radial_flow dots.
    :param min_depth: value to set as minimum depth.
    :param max_depth: value to set as maximum depth.
    :return: updated depth array.
    """
    depth_adj = max_depth - min_depth
    # adjust depth_arr values less than min_depth by adding depth_adj
    lessthanmin = (depth_arr < min_depth)
    depth_arr[lessthanmin] = depth_arr[lessthanmin] + depth_adj
    # adjust depth_arr values more than max_depth by subtracting depth_adj
    morethanmax = (depth_arr > max_depth)
    depth_arr[morethanmax] = depth_arr[morethanmax] - depth_adj
    return depth_arr


# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)


# Monitor config from monitor centre
# monitor_name = 'Nick_work_laptop'  # 'asus_cal', 'Nick_work_laptop',
# 'Asus_VG24', 'HP_24uh', 'NickMac', 'Iiyama_2_18', 'OLED'

# Store info about the experiment session (numbers keep the order)
expName = 'rad_flow_23'  # from the Builder filename that created this script
expInfo = {'1. Participant': 'Nick_test_23052023',
           '2. Run_number': '1',
           '3. Probe duration in frames': [2, 1, 50, 100],
           '4. fps': [60, 240, 120, 60],
           '5. ISI_dur_in_ms': [100, 50, 41.67, 37.5, 33.34, 25, 16.67, 8.33, 0],
           '6. Probe_orientation': ['radial', 'tangent'],
           '7. Vary_fixation': [True, False],
           '8. Record_frame_durs': [True, False],
           '9. Background': ['flow_rad', 'None'],
           '10. bg_speed_cond': ['Normal', 'Half-speed'],
           '11. prelim_bg_flow_ms': [350, 70],
           '12. monitor_name': ['Nick_work_laptop', 'OLED', 'asus_cal', 'Samsung',
                                'Asus_VG24', 'HP_24uh', 'NickMac', 'Iiyama_2_18'],
           '13. mask_type': ['4_circles', '2_spokes']
           }

# dialogue box
dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
if not dlg.OK:
    core.quit()  # user pressed escape

# dialogue box settings
participant_name = expInfo['1. Participant']
run_number = int(expInfo['2. Run_number'])
probe_duration = int(expInfo['3. Probe duration in frames'])
fps = int(expInfo['4. fps'])
ISI_selected_ms = float(expInfo['5. ISI_dur_in_ms'])
orientation = expInfo['6. Probe_orientation']
vary_fixation = eval(expInfo['7. Vary_fixation'])
record_fr_durs = eval(expInfo['8. Record_frame_durs'])
background = expInfo['9. Background']
bg_speed_cond = expInfo['10. bg_speed_cond']
prelim_bg_flow_ms = int(expInfo['11. prelim_bg_flow_ms'])
monitor_name = expInfo['12. monitor_name']
mask_type = expInfo['13. mask_type']

n_trials_per_stair = 25
probe_ecc = 4
expInfo['date'] = datetime.now().strftime("%d/%m/%Y")
expInfo['time'] = datetime.now().strftime("%H:%M:%S")

# ISI timing in ms and frames
'''ISI can be given (roughly) in ms, for any monitor it will try to match that value in frames.
The script uses on frames so ISI will be used in that format.
The actual ms timing is given for record.
This means that the experiment should have similar ms timings on monitors with different fps
milliseconds: [100, 50, 41.66, 37.5, 33.34, 25, 16.67, 8.33, 0]
frames@240hz: [24,  12,  10,    9,    8,     6,  4,    2,    0]
'''
ISI_frames = int(ISI_selected_ms * fps / 1000)
ISI_actual_ms = (1/fps) * ISI_frames * 1000
ISI = ISI_frames
print(f"\nSelected {ISI_selected_ms}ms ISI.\n"
      f"At {fps}Hz this is {ISI_frames} frames which each take {ISI_actual_ms}ms.\n")
ISI_list = [ISI_frames]
print(f'ISI_list: {ISI_list}')


# Distances between probes & flow direction
separations = [0, 1, 2, 3, 6, 18]
print(f'separations: {separations}')

'''each separation value appears in 2 stairs, e.g.,
stair1 will be sep=18, flow_dir=inwards; stair2 will be sep=18, flow_dir=outwards etc.
e.g., sep_vals_list = [18, 18, 6, 6, 3, 3, 2, 2, 1, 1, 0, 0]
this study does not include the two single probe conditions (labeled 99 in previous exp)
'''

# # main contrast is whether the background and target motion is in same or opposite direction.
# congruence_vals: 1=congruent/same, -1=incongruent/different
congruence_vals = [1, -1]
print(f'congruence_vals: {congruence_vals}')
congruence_names = ['cong', 'incong']
print(f'congruence_names: {congruence_names}')

ISI_vals_list = list(np.repeat(ISI_list, len(separations))) * len(congruence_vals)
print(f'ISI_vals_list: {ISI_vals_list}')
sep_vals_list = list(np.tile(separations, len(ISI_list) * len(congruence_vals)))
print(f'sep_vals_list: {sep_vals_list}')
cong_vals_list = list(np.repeat(congruence_vals, len(sep_vals_list) / len(congruence_vals)))
print(f'cong_vals_list: {cong_vals_list}')
cong_names_list = list(np.repeat(congruence_names, len(sep_vals_list) / len(congruence_vals)))
print(f'cong_names_list: {cong_names_list}')

# stair_names_list joins cong_names_list, sep_vals_list and ISI_vals_list
# e.g., ['cong_sep18_ISI6', 'cong_sep6_ISI6', 'incong_sep18_ISI6', 'incong_sep6_ISI6', ]
stair_names_list = [f'{p}_sep{s}_ISI{i}' for p, s, i in zip(cong_names_list, sep_vals_list, ISI_vals_list)]
print(f'stair_names_list: {stair_names_list}')
n_stairs = len(sep_vals_list)
print(f'n_stairs: {n_stairs}')
total_n_trials = int(n_trials_per_stair * n_stairs)
print(f'total_n_trials: {total_n_trials}')


# background motion converted to frames (e.g., 70ms is 17frames at 240Hz).
prelim_bg_flow_fr = int(prelim_bg_flow_ms * fps / 1000)
actual_prelim_bg_flow_ms = prelim_bg_flow_fr * 1000 / fps
print(f'\nprelim_bg_flow_ms: {prelim_bg_flow_ms}')
print(f'prelim_bg_flow_fr: {prelim_bg_flow_fr}')
print(f'actual_prelim_bg_flow_ms: {actual_prelim_bg_flow_ms}')


# FILENAME
# join participant_name with prelim_bg_flow_ms so that different prelim values don't get overwritten or confused.
participant_name = participant_name + f'_bg{prelim_bg_flow_ms}'

isi_dir = f'ISI_{ISI}'
save_dir = os.path.join(_thisDir, expName, participant_name,
                        f'{participant_name}_{run_number}', isi_dir)

# files are labelled as '_incomplete' unless entire script runs.
incomplete_output_filename = f'{participant_name}_{run_number}_incomplete'
save_output_as = os.path.join(save_dir, incomplete_output_filename)


# Experiment Handler
thisExp = data.ExperimentHandler(name=expName, version=psychopy_version,
                                 extraInfo=expInfo, runtimeInfo=None,
                                 savePickle=True, saveWideText=True,
                                 dataFileName=save_output_as)

# Monitor details: colour, luminance, pixel size and frame rate
print(f"\nmonitor_name: {monitor_name}")
thisMon = monitors.Monitor(monitor_name)


# COLORS AND LUMINANCE
'''
rad_flow_NM_v2 used a lighter screen that exp1.  (bg as 45% not 20%)
flow_bgcolor = [-0.1, -0.1, -0.1]  # dark grey converts to:
rgb: -0.1 = rgb1: .45 = rgb255: 114.75 = lum: 47.8.
for future ref, to match exp1 it should be flow_bgcolor = [-0.6, -0.6, -0.6]  # dark grey
'''
# # Lum to Color255 (maxLum = 253)
LumColor255Factor = 2.39538706913372
maxLum = 106  # 255 RGB
bgLumProp = .2  # .45  # .2
if monitor_name == 'OLED':
    bgLumProp = .0
bgLum = maxLum * bgLumProp
bgColor255 = int(bgLum * LumColor255Factor)
bgColor_rgb1 = bgLum / maxLum
bg_color_rgb = (bgColor_rgb1 * 2) - 1
print(f'bgLum: {bgLum}, bgColor255: {bgColor255}, bgColor_rgb1: {bgColor_rgb1}, bg_color_rgb: {bg_color_rgb}')

# colour space
this_colourSpace = 'rgb255'  # 'rgb255', 'rgb1'
this_bgColour = [bgColor255, bgColor255, bgColor255]
adj_dots_col = int(255 * .15)
if monitor_name == 'OLED':
    this_colourSpace = 'rgb1'  # values between 0 and 1
    this_bgColour = [bgColor_rgb1, bgColor_rgb1, bgColor_rgb1]
    adj_dots_col = .15
print(f"this_colourSpace: {this_colourSpace}, this_bgColour: {this_bgColour}")
print(f"adj_dots_col colours: {adj_dots_col}")

# don't use full screen on external monitor
display_number = 1  # 0 indexed, 1 for external display, 0 for internal
if monitor_name in ['asus_cal', 'Nick_work_laptop', 'NickMac', 'OLED', 'ASUS_2_13_240Hz']:
    display_number = 0
use_full_screen = True
if display_number > 0:
    use_full_screen = False
print(f"display_number: {display_number}, use_full_screen: {use_full_screen}")

widthPix = int(thisMon.getSizePix()[0])
heightPix = int(thisMon.getSizePix()[1])
monitorwidth = thisMon.getWidth()  # monitor width in cm
viewdist = thisMon.getDistance()  # viewing distance in cm
viewdistPix = widthPix / monitorwidth*viewdist
mon = monitors.Monitor(monitor_name, width=monitorwidth, distance=viewdist)
mon.setSizePix((widthPix, heightPix))
print(f"widthPix: {widthPix}, heightPix: {heightPix}, monitorwidth: {monitorwidth}, "
      f"viewdist: {viewdist}, viewdistPix: {viewdistPix}")

# WINDOW
'''if running on pycharm/mac it might need pyglet'''
# todo: note change of winType 23/05/2023
win = visual.Window(monitor=mon, size=(widthPix, heightPix),
                    colorSpace=this_colourSpace, color=this_bgColour,
                    # winType='GLFW',  # I've added this to make it work on pycharm/mac
                    pos=[1, -1],  # pos gives position of top-left of screen
                    units='pix',
                    screen=display_number,
                    allowGUI=False,
                    fullscr=use_full_screen)
print(f'winType: {win.winType}')

# pixel size
pixel_mm_deg_dict = get_pixel_mm_deg_values(monitor_name=monitor_name)
print(f"diagonal pixel size: {pixel_mm_deg_dict['diag_mm']} mm, or {pixel_mm_deg_dict['diag_deg']} dva")


# expected frame duration
expected_fr_sec = 1/fps
expected_fr_ms = expected_fr_sec * 1000
print(f"\nexpected frame duration: {expected_fr_ms} ms (or {round(expected_fr_sec, 5)} seconds).")
actualFrameRate = int(win.getActualFrameRate())
print(f"actual fps: {win.getActualFrameRate()}")
if abs(fps-actualFrameRate) > 5:
    raise ValueError(f"\nfps ({fps}) does not match actualFrameRate ({actualFrameRate}).")

'''set the max and min frame duration to accept, trials with critial frames beyond these bound will be repeated.'''
# frame error tolerance - default is approx 20% but seems to vary between runs(!), so set it manually.
frame_tolerance_prop = .2
max_fr_dur_sec = expected_fr_sec + (expected_fr_sec * frame_tolerance_prop)
max_fr_dur_ms = max_fr_dur_sec * 1000
win.refreshThreshold = max_fr_dur_sec
frame_tolerance_sec = max_fr_dur_sec - expected_fr_sec
frame_tolerance_ms = frame_tolerance_sec * 1000
frame_tolerance_prop = frame_tolerance_sec / expected_fr_sec
min_fr_dur_sec = expected_fr_sec - (expected_fr_sec * frame_tolerance_prop)
print(f"\nframe_tolerance_sec: {frame_tolerance_sec} ({frame_tolerance_prop}% of {expected_fr_sec} sec)")
print(f"max_fr_dur_sec ({100 + (100 * frame_tolerance_prop)}%): {max_fr_dur_sec} (or {max_fr_dur_ms}ms)")
print(f"min_fr_dur_sec ({100 - (100 * frame_tolerance_prop)}%): {min_fr_dur_sec} (or {min_fr_dur_sec * 1000}ms)")

# quit experiment if there are more than 10 trials with dropped frames
max_droped_fr_trials = 10


# ELEMENTS
# fixation bull eye
if background == 'flow_rad':
    fixation = visual.Circle(win, radius=2, units='pix',
                             lineColor='black', fillColor='grey', colorSpace=this_colourSpace)
else:
    fixation = visual.Circle(win, radius=2, units='pix',
                             lineColor='white', fillColor='black', colorSpace=this_colourSpace)

# PROBEs
# default is to use 5 pixel probes,but can use 7 on OLED if needed
probe_n_pixels = 5  # 7

probeVert = [(0, 0), (1, 0), (1, 1), (2, 1), (2, -1), (1, -1),
             (1, -2), (-1, -2), (-1, -1), (0, -1)]
if probe_n_pixels == 7:
    # this one looks back-to-front as the extra bits have turned the 'm's into 'w's,
    # so probes are rotated 180 degrees compared to regular probes.
    probeVert = [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (0, 2), (0, 1),
                 (-1, 1), (-1, 0), (-2, 0), (-2, -2), (-1, -2), (-1, -1), (0, -1)]

probe_size = 1
probe1 = visual.ShapeStim(win, vertices=probeVert, fillColor='white', colorSpace=this_colourSpace,
                          lineWidth=0, opacity=1, size=probe_size, interpolate=False)
probe2 = visual.ShapeStim(win, vertices=probeVert, fillColor='white', colorSpace=this_colourSpace,
                          lineWidth=0, opacity=1, size=probe_size, interpolate=False)


# dist_from_fix is a constant to get 4dva distance from fixation,
dist_from_fix = round((tan(np.deg2rad(probe_ecc)) * viewdistPix) / sqrt(2))


# MASK BEHIND PROBES
'''This is either circles in the four locations where probes can appear, or
two diagoal lines that cross at the fixation point.'''
mask_size = 150

if mask_type == '4_circles':
    raisedCosTexture1 = visual.filters.makeMask(256, shape='raisedCosine',
                                                fringeWidth=0.3, radius=[1.0, 1.0])
    probeMask1 = visual.GratingStim(win=win, mask=raisedCosTexture1, size=(mask_size, mask_size),
                                    colorSpace=this_colourSpace, color=this_bgColour,
                                    tex=None, units='pix', pos=[dist_from_fix + 1, dist_from_fix + 1]
                                    )
    probeMask2 = visual.GratingStim(win=win, mask=raisedCosTexture1, size=(mask_size, mask_size),
                                    colorSpace=this_colourSpace, color=this_bgColour,
                                    units='pix', tex=None, pos=[-dist_from_fix - 1, dist_from_fix + 1])
    probeMask3 = visual.GratingStim(win=win, mask=raisedCosTexture1, size=(mask_size, mask_size),
                                    colorSpace=this_colourSpace, color=this_bgColour,
                                    units='pix', tex=None, pos=[-dist_from_fix - 1, -dist_from_fix - 1])
    probeMask4 = visual.GratingStim(win=win, mask=raisedCosTexture1, size=(mask_size, mask_size),
                                    colorSpace=this_colourSpace, color=this_bgColour,
                                    units='pix', tex=None, pos=[dist_from_fix + 1, -dist_from_fix - 1])
    # probe_mask_list = [probeMask1, probeMask2, probeMask3, probeMask4]
    probes_mask = visual.BufferImageStim(win, stim=[probeMask1, probeMask2, probeMask3, probeMask4])

elif mask_type == '2_spokes':
    # since the middle of the screen is 0, 0; the corners are defined by half the width or height of the screen.
    half_hi_pix = int(heightPix / 2)

    # the thickness of the cross will change the vertices of the cross.
    cross_thickness_pix = 150
    horiz_offset_pix = int(cross_thickness_pix / 2)
    scr_ratio = widthPix / heightPix
    vert_offset_pix = int(horiz_offset_pix / scr_ratio)
    print(f'vert_offset_pix = {vert_offset_pix}')
    # draw a large cross with vertices which reaches the corners of the window
    '''vertices start at the bottom left corner and go clockwise.  
    the first row of four values in are: 1. the bl corner, 2. small distance up left side of screen, 
    3. in toward the middle of the screen, 4. small distance down the left side of the screen from tl corner.  
    '''

    vertices = np.array([[-half_hi_pix - vert_offset_pix, -half_hi_pix], [-vert_offset_pix, 0],
                         [-half_hi_pix - vert_offset_pix, half_hi_pix],
                         [-half_hi_pix + vert_offset_pix, half_hi_pix], [0, vert_offset_pix],
                         [half_hi_pix - vert_offset_pix, half_hi_pix],
                         [half_hi_pix + vert_offset_pix, half_hi_pix], [vert_offset_pix, 0],
                         [half_hi_pix + vert_offset_pix, -half_hi_pix],
                         [half_hi_pix - vert_offset_pix, -half_hi_pix], [0, -vert_offset_pix],
                         [-half_hi_pix + vert_offset_pix, -half_hi_pix]
                         ])

    probes_mask = visual.ShapeStim(win, vertices=vertices, fillColor=this_bgColour, lineColor=this_bgColour)
# use this variable for the probe masks (either circles or spokes)
# probes_mask = visual.BufferImageStim(win, stim=probe_mask_list)


# BACKGROUND
# flow_dots
if bg_speed_cond == 'Normal':
    flow_speed = 0.2
elif bg_speed_cond == 'Half-speed':
    flow_speed = 0.1
else:
    raise ValueError(f'background speed should be selected from drop down menu: Normal or Half-speed')
nDots = 10000
dot_array_width = 10000  # original script used 5000
minDist = 0.5  # depth values
maxDist = 5  # depth values

# pale green
flow_dots_colour = [this_bgColour[0]-adj_dots_col, this_bgColour[1], this_bgColour[2]-adj_dots_col]
if monitor_name == 'OLED':
    # darker green for low contrast against black background
    flow_dots_colour = [this_bgColour[0], this_bgColour[1] + adj_dots_col / 2, this_bgColour[2]]

flow_dots = visual.ElementArrayStim(win, elementTex=None, elementMask='circle',
                                    units='pix', nElements=nDots, sizes=10,
                                    colorSpace=this_colourSpace, colors=flow_dots_colour)
print(f"flow_dot colours: {[this_bgColour[0]-adj_dots_col, this_bgColour[1], this_bgColour[2]-adj_dots_col]}")


# full screen mask to blend off edges and fade to black
# Create a raisedCosine mask array and assign it to a Grating stimulus (grey outside, transparent inside)
# this was useful http://www.cogsci.nl/blog/tutorials/211-a-bit-about-patches-textures-and-masks-in-psychopy
# raisedCosTexture2 = visual.filters.makeMask(1080, shape='raisedCosine', fringeWidth=0.6, radius=[1.0, 1.0])
raisedCosTexture2 = visual.filters.makeMask(heightPix, shape='raisedCosine', fringeWidth=0.6, radius=[1.0, 1.0])
invRaisedCosTexture = -raisedCosTexture2  # inverts mask to blur edges instead of center
slab_width = 420
if monitor_name == 'OLED':
    slab_width = 20

blankslab = np.ones((heightPix, slab_width))  # create blank slabs to put to left and right of image
mmask = np.append(blankslab, invRaisedCosTexture, axis=1)  # append blank slab to left
mmask = np.append(mmask, blankslab, axis=1)  # and right
# changed dotsmask color from grey, fades to black round edges which makes screen edges less visible
dotsMask = visual.GratingStim(win, mask=mmask, tex=None, contrast=1.0,
                              size=(widthPix, heightPix), units='pix', color='black')


# MOUSE - hide cursor
myMouse = event.Mouse(visible=False)

# # KEYBOARD
resp = event.BuilderKeyResponse()

# INSTRUCTION
instructions = visual.TextStim(win=win, name='instructions', font='Arial', height=20,
                               color='white', colorSpace=this_colourSpace,
                               text="\n\n\n\n\n\nFocus on the fixation circle at the centre of the screen.\n\n"
                                    "A small white target will briefly appear on screen,\n"
                                    "press the key related to the location of the probe:\n\n"
                                    "[4]/[Q] top-left\t\t\t[5]/[W] top-right\n\n\n\n"
                                    "[1]/[A] bottom-left\t\t\t[2]/[S] bottom-right.\n\n\n"
                                    "Some targets will be easier to see than others,\n"
                                    "Some will be so dim that you won't see them, so just guess!\n\n"
                                    "You don't need to think for long, respond quickly, "
                                    "but try to push press the correct key!\n\n"
                                    "Don't let your eyes wander, keep focussed on the circle in the middle throughout.")


# BREAKS
max_trials = total_n_trials + max_droped_fr_trials  # expected trials plus repeats
max_without_break = 120  # limit on number of trials without a break
n_breaks = max_trials // max_without_break  # number of breaks
if n_breaks > 0:
    take_break = int(max_trials / (n_breaks + 1))
else:
    take_break = max_without_break
break_dur = 30
print(f"\ntake a {break_dur} second break every {take_break} trials ({n_breaks} breaks in total).")
break_text = f"Break\nTurn on the light and take at least {break_dur} seconds break.\n" \
             "Keep focussed on the fixation circle in the middle of the screen.\n" \
             "Remember, if you don't see the target, just guess!"
breaks = visual.TextStim(win=win, name='breaks', text=break_text, font='Arial',
                         pos=[0, 0], height=20, ori=0, color='white',
                         colorSpace=this_colourSpace)

end_of_exp_text = "You have completed this experiment.\nThank you for your time.\n\n"
end_of_exp = visual.TextStim(win=win, name='end_of_exp',
                             text=end_of_exp_text, color='white',
                             font='Arial', height=20, colorSpace=this_colourSpace)

too_many_dropped_fr = visual.TextStim(win=win, name='too_many_dropped_fr',
                                      text="The experiment had quit as the computer is dropping frames.\n"
                                           "Sorry for the inconvenience.\n"
                                           "Please contact the experimenter.\n\n"
                                           "Press any key to return to the desktop.",
                                      font='Arial', height=20, colorSpace=this_colourSpace,)

while not event.getKeys():
    fixation.setRadius(3)
    fixation.draw()
    instructions.draw()
    win.flip()


# empty variable to store recorded frame durations
exp_n_fr_recorded_list = [0]
exp_n_dropped_fr = 0
dropped_fr_trial_counter = 0
dropped_fr_trial_x_locs = []
fr_int_per_trial = []
recorded_fr_counter = 0
fr_counter_per_trial = []
cond_list = []


# STAIRCASE
expInfo['stair_list'] = list(range(n_stairs))
expInfo['n_trials_per_stair'] = n_trials_per_stair

stairStart = maxLum
if monitor_name == 'OLED':
    stairStart = maxLum * 0.3

miniVal = bgLum
maxiVal = maxLum

print('\nexpInfo (dict)')
for k, v in expInfo.items():
    print(f"{k}: {v}")

stairs = []
for stair_idx in expInfo['stair_list']:

    thisInfo = copy.copy(expInfo)
    thisInfo['stair_idx'] = stair_idx

    thisStair = Staircase(name=stair_names_list[stair_idx],
                          type='simple',
                          value=stairStart,
                          C=stairStart * 0.6,  # initial step size, as prop of reference stim
                          minRevs=3,
                          minTrials=n_trials_per_stair,
                          minVal=miniVal,
                          maxVal=maxiVal,
                          targetThresh=0.75,
                          extraInfo=thisInfo)
    stairs.append(thisStair)

# the number of the trial for the output file
trial_number = 0
# the actual number of trials including repeated trials (trial_number stays the same for these)
actual_trials_inc_rpt = 0


# EXPERIMENT
print('\n*** exp loop*** \n\n')
for step in range(n_trials_per_stair):
    np.random.shuffle(stairs)
    for thisStair in stairs:

        # repeat the trial if [r] has been pressed or frames were dropped
        repeat = True
        while repeat:

            # Trial, stair and step
            trial_number += 1
            actual_trials_inc_rpt += 1
            stair_idx = thisStair.extraInfo['stair_idx']
            print(f"\n({actual_trials_inc_rpt}) trial_number: {trial_number}, "
                  f"stair_idx: {stair_idx}, thisStair: {thisStair}, step: {step}")

            # conditions (ISI, congruence)
            ISI = ISI_vals_list[stair_idx]
            congruent = cong_vals_list[stair_idx]
            cong_name = cong_names_list[stair_idx]
            print(f"ISI: {ISI}, congruent: {congruent} ({cong_name})")

            # conditions (sep, sep_deg, neg_sep)
            sep = sep_vals_list[stair_idx]
            # separation expressed as degrees.
            if -1 < sep < 99:
                sep_deg = sep * pixel_mm_deg_dict['diag_deg']
            else:
                sep_deg = 0

            # negative separation for comparing conditions (e.g., cong sep = 5, incong sep = -5.
            if cong_name == 'incong':
                neg_sep = 0 - sep
                if sep == 0:
                    neg_sep = -.1
            else:
                neg_sep = sep
            print(f"sep: {sep}, sep_deg: {sep_deg}, neg_sep: {neg_sep}")

            # use congruence to determine the flow direction and target jump direction
            flow_dir = np.random.choice([1, -1])
            target_jump = congruent * flow_dir

            # # direction in which the probe jumps : CW or CCW (tangent) or expand vs contract (radial)
            if orientation == 'tangent':
                jump_dir = 'clockwise'
                if target_jump == -1:
                    jump_dir = 'anticlockwise'
            else:  # if radial
                jump_dir = 'cont'
                if target_jump == -1:
                    jump_dir = 'exp'
            print(f"flow_dir: {flow_dir}, jump dir: {target_jump} {jump_dir} ({cong_name})")

            # vary fixation polarity to reduce risk of screen burn.
            if monitor_name == 'OLED':
                if trial_number % 2 == 0:
                    fixation.lineColor = 'grey'
                    fixation.fillColor = 'black'
                else:
                    fixation.lineColor = 'black'
                    fixation.fillColor = 'grey'

            # Luminance (staircase varies probeLum)
            probeLum = thisStair.next()
            probeColor255 = int(probeLum * LumColor255Factor)  # rgb255 are ints.
            probeColor1 = probeLum / maxLum

            this_probeColor = probeColor255
            if this_colourSpace == 'rgb1':
                this_probeColor = probeColor1
            probe1.setFillColor([this_probeColor, this_probeColor, this_probeColor])
            probe2.setFillColor([this_probeColor, this_probeColor, this_probeColor])
            print(f"probeLum: {probeLum}, this_probeColor: {this_probeColor}, "
                  f"probeColor255: {probeColor255}, probeColor1: {probeColor1}")


            # PROBE LOCATION
            # # corners go CCW(!) 45=top-right, 135=top-left, 225=bottom-left, 315=bottom-right
            corner = random.choice([45, 135, 225, 315])
            corner_name = 'top_right'
            if corner == 135:
                corner_name = 'top_left'
            elif corner == 225:
                corner_name = 'bottom_left'
            elif corner == 315:
                corner_name = 'bottom_right'
            print(f"corner: {corner} {corner_name}")


            # flow_dots
            x = np.random.rand(nDots) * dot_array_width - dot_array_width / 2
            y = np.random.rand(nDots) * dot_array_width - dot_array_width / 2
            z = np.random.rand(nDots) * (maxDist - minDist) + minDist
            # z was called z_flow but is actually z position like x and y
            x_flow = x / z
            y_flow = y / z


            # shift probes by separation
            '''Both probes should be equally spaced around the meridian point.
            E.g., if sep = 4, probe 1 will be shifted 2 pixels in one direction and 
            probe 2 will be shifted 2 pixels in opposite direction. 
            Where separation is an odd number (e.g., 5), they will be shifted by 2 and 3 pixels; allocated randomly.
            To check probe locations, uncomment loc_marker'''
            if sep == 99:
                p1_shift = p2_shift = 0
            elif sep % 2 == 0:  # even number
                p1_shift = p2_shift = sep // 2
            else:  # odd number
                extra_shifted_pixel = [0, 1]
                np.random.shuffle(extra_shifted_pixel)
                p1_shift = sep // 2 + extra_shifted_pixel[0]
                p2_shift = (sep // 2) + extra_shifted_pixel[1]


            # set position and orientation of probes
            '''NEW - set orientations to p1=zero and p2=180 (not zero), 
            then add the same orientation change to both'''
            probe1_ori = 0
            probe2_ori = 180
            if probe_n_pixels == 7:
                probe1_ori = 180
                probe2_ori = 0
            if corner == 45:
                '''in top-right corner, both x and y increase (right and up)'''
                loc_x = dist_from_fix * 1
                loc_y = dist_from_fix * 1
                '''orientation' here refers to the relationship between probes, 
                whereas probe1_ori refers to rotational angle of probe stimulus'''
                if orientation == 'tangent':
                    if target_jump == 1:  # CW
                        probe1_ori += 180
                        probe2_ori += 180
                        probe1_pos = [loc_x - p1_shift, loc_y + p1_shift]
                        probe2_pos = [loc_x + p2_shift - probe_size, loc_y - p2_shift]
                    elif target_jump == -1:  # ACW
                        probe1_ori += 0
                        probe2_ori += 0
                        probe1_pos = [loc_x + p1_shift, loc_y - p1_shift]
                        probe2_pos = [loc_x - p2_shift + probe_size, loc_y + p2_shift]
                elif orientation == 'radial':
                    if target_jump == 1:  # inward
                        probe1_ori += 270
                        probe2_ori += 270
                        # probe2 is left and down from probe1
                        probe1_pos = [loc_x + p1_shift, loc_y + p1_shift]
                        probe2_pos = [loc_x - p2_shift + probe_size, loc_y - p2_shift]
                    elif target_jump == -1:  # outward
                        probe1_ori += 90
                        probe2_ori += 90
                        # probe2 is right and up from probe1
                        probe1_pos = [loc_x - p1_shift, loc_y - p1_shift]
                        probe2_pos = [loc_x + p2_shift - probe_size, loc_y + p2_shift]
            elif corner == 135:
                loc_x = dist_from_fix * -1
                loc_y = dist_from_fix * 1
                if orientation == 'tangent':
                    if target_jump == 1:  # ACW
                        probe1_ori += 90
                        probe2_ori += 90
                        probe1_pos = [loc_x - p1_shift, loc_y - p1_shift]
                        probe2_pos = [loc_x + p2_shift - probe_size, loc_y + p2_shift]
                    elif target_jump == -1:  # CW
                        probe1_ori += 270
                        probe2_ori += 270
                        probe1_pos = [loc_x + p1_shift, loc_y + p1_shift]
                        probe2_pos = [loc_x - p2_shift + probe_size, loc_y - p2_shift]
                elif orientation == 'radial':
                    if target_jump == 1:  # inward
                        probe1_ori += 180
                        probe2_ori += 180
                        # probe2 is right and down from probe1
                        probe1_pos = [loc_x - p1_shift, loc_y + p1_shift]
                        probe2_pos = [loc_x + p2_shift - probe_size, loc_y - p2_shift]
                    elif target_jump == -1:  # outward
                        probe1_ori += 0
                        probe2_ori += 0
                        # probe2 is left and up from probe1
                        probe1_pos = [loc_x + p1_shift, loc_y - p1_shift]
                        probe2_pos = [loc_x - p2_shift + probe_size, loc_y + p2_shift]
            elif corner == 225:
                loc_x = dist_from_fix * -1
                loc_y = dist_from_fix * -1
                if orientation == 'tangent':
                    if target_jump == 1:  # CW
                        probe1_ori += 0
                        probe2_ori += 0
                        probe1_pos = [loc_x + p1_shift, loc_y - p1_shift]
                        probe2_pos = [loc_x - p2_shift + probe_size, loc_y + p2_shift]
                    elif target_jump == -1:  # ACW
                        probe1_ori += 180
                        probe2_ori += 180
                        probe1_pos = [loc_x - p1_shift, loc_y + p1_shift]
                        probe2_pos = [loc_x + p2_shift - probe_size, loc_y - p2_shift]
                elif orientation == 'radial':
                    if target_jump == 1:  # inward
                        probe1_ori += 90
                        probe2_ori += 90
                        # probe2 is right and up from probe1
                        probe1_pos = [loc_x - p1_shift, loc_y - p1_shift]
                        probe2_pos = [loc_x + p2_shift - probe_size, loc_y + p2_shift]
                    elif target_jump == -1:  # outward
                        probe1_ori += 270
                        probe2_ori += 270
                        # probe2 is left and down from probe1
                        probe1_pos = [loc_x + p1_shift, loc_y + p1_shift]
                        probe2_pos = [loc_x - p2_shift + probe_size, loc_y - p2_shift]
            else:
                corner = 315
                loc_x = dist_from_fix * 1
                loc_y = dist_from_fix * -1
                if orientation == 'tangent':
                    if target_jump == 1:  # ACW
                        probe1_ori += 270
                        probe2_ori += 270
                        probe1_pos = [loc_x + p1_shift, loc_y + p1_shift]
                        probe2_pos = [loc_x - p2_shift + probe_size, loc_y - p2_shift]
                    elif target_jump == -1:  # CW
                        probe1_ori += 90
                        probe2_ori += 90
                        probe1_pos = [loc_x - p1_shift, loc_y - p1_shift]
                        probe2_pos = [loc_x + p2_shift - probe_size, loc_y + p2_shift]
                elif orientation == 'radial':
                    if target_jump == 1:  # inward
                        probe1_ori += 0
                        probe2_ori += 0
                        # probe2 is left and up from probe1
                        probe1_pos = [loc_x + p1_shift, loc_y - p1_shift]
                        probe2_pos = [loc_x - p2_shift + probe_size, loc_y + p2_shift]
                    elif target_jump == -1:  # outward
                        probe1_ori += 180
                        probe2_ori += 180
                        # probe2 is right and down from probe1
                        probe1_pos = [loc_x - p1_shift, loc_y + p1_shift]
                        probe2_pos = [loc_x + p2_shift - probe_size, loc_y - p2_shift]

            # loc_marker.setPos([loc_x, loc_y])
            probe1.setPos(probe1_pos)
            probe1.setOri(probe1_ori)
            probe2.setPos(probe2_pos)
            probe2.setOri(probe2_ori)
            print(f"loc_marker: {[loc_x, loc_y]}, probe1_pos: {probe1_pos}, "
                  f"probe2_pos: {probe2_pos}. dff: {dist_from_fix}")


            # VARIABLE FIXATION TIME
            '''to reduce anticipatory effects that might arise from fixation always being same length.
            if False, vary_fix == .5 seconds, so t_fixation is 1 second.
            if Ture, vary_fix is between 0 and 1 second, so t_fixation is between .5 and 1.5 seconds.'''
            vary_fix = int(fps / 2)
            if vary_fixation:
                vary_fix = np.random.randint(0, fps)

            # timing in frames for ISI and probe2
            # If probes are presented concurrently, set ISI and probe2 to last for 0 frames.
            isi_dur_fr = ISI
            p2_fr = probe_duration
            if ISI < 0:
                isi_dur_fr = p2_fr = 0

            # cumulative timing in frames for each part of a trial
            t_fixation = int(fps / 2) + vary_fix - prelim_bg_flow_fr
            if t_fixation < 0:
                t_fixation = int(fps / 2)
            t_bg_motion = t_fixation + prelim_bg_flow_fr
            t_probe_1 = t_bg_motion + probe_duration
            t_ISI = t_probe_1 + isi_dur_fr
            t_probe_2 = t_ISI + p2_fr
            t_response = t_probe_2 + 10000 * fps  # ~40 seconds to respond
            print(f"t_fixation: {t_fixation}, t_probe_1: {t_probe_1}, "
                  f"t_ISI: {t_ISI}, t_probe_2: {t_probe_2}, t_response: {t_response}\n")


            # continue_routine refers to flipping the screen to show next frame

            # take a break every ? trials
            # if (trial_number % take_break == 1) & (trial_number > 1):
            if (actual_trials_inc_rpt % take_break == 1) & (actual_trials_inc_rpt > 1):
                print("\nTaking a break.\n")

                breaks.text = break_text + f"\n{trial_number - 1}/{total_n_trials} trials completed."
                breaks.draw()
                win.flip()
                event.clearEvents(eventType='keyboard')
                core.wait(secs=break_dur)
                event.clearEvents(eventType='keyboard')
                breaks.text = break_text + "\n\nPress any key to continue."
                breaks.draw()
                win.flip()
                while not event.getKeys():
                    # continue the breaks routine until a key is pressed
                    continueRoutine = True
            else:
                # else continue the trial routine.
                continueRoutine = True


            # initialise frame number
            frameN = -1
            while continueRoutine:
                frameN = frameN + 1

                # blank screen for n frames if on OLED to stop it adapting to bgColor
                if monitor_name == 'OLED':
                    if frameN == 1:
                        win.color = 'black'

                    if frameN == 5:
                        win.color = this_bgColour

                # recording frame durations - from t_fixation (1 frame before p1), until 1 frame after p2.
                if frameN == t_fixation:

                    # todo: test this on windows, Linux and mac to see if it matters
                    # prioritise psychopy
                    # core.rush(True)

                    # start recording frame intervals
                    if record_fr_durs:
                        win.recordFrameIntervals = True
                        # print(f"{frameN}: win.recordFrameIntervals : {win.recordFrameIntervals}")

                    # reset timer to start with probe1 presentation.
                    resp.clock.reset()

                    # clear any previous key presses
                    event.clearEvents(eventType='keyboard')
                    theseKeys = []

                if frameN == t_probe_2 + 1:
                    # relax psychopy prioritization
                    # core.rush(False)

                    if record_fr_durs:
                        win.recordFrameIntervals = False
                        # print(f"{frameN}: win.recordFrameIntervals : {win.recordFrameIntervals}")


                '''Experiment timings'''
                # FIXATION - up to the end of fixation period
                if t_fixation >= frameN > 0:
                    if background == 'flow_rad':
                        # draw flow_dots but with no motion
                        flow_dots.xys = np.array([x_flow, y_flow]).transpose()
                        flow_dots.draw()
                        # probeMask1.draw()
                        # probeMask2.draw()
                        # probeMask3.draw()
                        # probeMask4.draw()
                        probes_mask.draw()
                        dotsMask.draw()

                    fixation.setRadius(3)
                    fixation.draw()

                # Background motion prior to probe1 - after fixation, but before probe 1
                elif t_bg_motion >= frameN > t_fixation:
                    if background == 'flow_rad':
                        # draw dots with motion
                        z = z + flow_speed * flow_dir
                        z = wrap_depth_vals(z, minDist, maxDist)
                        x_flow = x / z
                        y_flow = y / z

                        flow_dots.xys = np.array([x_flow, y_flow]).transpose()
                        flow_dots.draw()

                        # probeMask1.draw()
                        # probeMask2.draw()
                        # probeMask3.draw()
                        # probeMask4.draw()
                        probes_mask.draw()
                        dotsMask.draw()

                    fixation.setRadius(3)
                    fixation.draw()

                # PROBE 1 - after background motion, before end of probe1 interval
                elif t_probe_1 >= frameN > t_bg_motion:
                    if background == 'flow_rad':
                        # draw dots with motion
                        z = z + flow_speed * flow_dir
                        z = wrap_depth_vals(z, minDist, maxDist)
                        x_flow = x / z
                        y_flow = y / z

                        flow_dots.xys = np.array([x_flow, y_flow]).transpose()
                        flow_dots.draw()

                        # probeMask1.draw()
                        # probeMask2.draw()
                        # probeMask3.draw()
                        # probeMask4.draw()
                        probes_mask.draw()
                        dotsMask.draw()

                    fixation.setRadius(3)
                    fixation.draw()

                    probe1.draw()
                    if ISI == -1:  # SIMULTANEOUS CONDITION (concurrent)
                        if sep <= 18:  # don't draw 2nd probe in 1probe cond (sep==99)
                            probe2.draw()
                    fixation.setRadius(3)
                    fixation.draw()

                # ISI
                elif t_ISI >= frameN > t_probe_1:
                    if background == 'flow_rad':
                        # draw dots with motion
                        z = z + flow_speed * flow_dir
                        z = wrap_depth_vals(z, minDist, maxDist)
                        x_flow = x / z
                        y_flow = y / z

                        flow_dots.xys = np.array([x_flow, y_flow]).transpose()
                        flow_dots.draw()

                        # probeMask1.draw()
                        # probeMask2.draw()
                        # probeMask3.draw()
                        # probeMask4.draw()
                        probes_mask.draw()
                        dotsMask.draw()

                    fixation.setRadius(3)
                    fixation.draw()

                # PROBE 2 - after ISI but before end of probe2 interval
                elif t_probe_2 >= frameN > t_ISI:
                    if background == 'flow_rad':
                        # draw dots with motion
                        z = z + flow_speed * flow_dir
                        z = wrap_depth_vals(z, minDist, maxDist)
                        x_flow = x / z
                        y_flow = y / z

                        flow_dots.xys = np.array([x_flow, y_flow]).transpose()
                        flow_dots.draw()

                        # probeMask1.draw()
                        # probeMask2.draw()
                        # probeMask3.draw()
                        # probeMask4.draw()
                        probes_mask.draw()
                        dotsMask.draw()

                    fixation.setRadius(3)
                    fixation.draw()

                    if ISI >= 0:
                        if sep <= 18:  # don't draw 2nd probe in 1probe cond (sep==99)
                            probe2.draw()

                # ANSWER - after probe 2 interval
                elif frameN > t_probe_2:
                    if background == 'flow_rad':
                        # draw flow_dots but with no motion
                        flow_dots.draw()
                        # probeMask1.draw()
                        # probeMask2.draw()
                        # probeMask3.draw()
                        # probeMask4.draw()
                        probes_mask.draw()
                        dotsMask.draw()

                    fixation.setRadius(2)
                    fixation.draw()

                    # ANSWER
                    theseKeys = event.getKeys(keyList=['num_5', 'num_4', 'num_1',
                                                       'num_2', 'w', 'q', 'a', 's'])

                    if len(theseKeys) > 0:  # at least one key was pressed
                        theseKeys = theseKeys[-1]  # just the last key pressed
                        resp.keys = theseKeys
                        resp.rt = resp.clock.getTime()

                        # default assume response incorrect unless meets criteria below
                        resp.corr = 0

                        if corner == 45:
                            if (resp.keys == 'w') or (resp.keys == 'num_5'):
                                resp.corr = 1
                        elif corner == 135:
                            if (resp.keys == 'q') or (resp.keys == 'num_4'):
                                resp.corr = 1
                        elif corner == 225:
                            if (resp.keys == 'a') or (resp.keys == 'num_1'):
                                resp.corr = 1
                        elif corner == 315:
                            if (resp.keys == 's') or (resp.keys == 'num_2'):
                                resp.corr = 1


                        '''Get frame intervals for this trial, add to experiment and empty cache'''
                        if record_fr_durs:
                            # get trial frameIntervals details
                            trial_fr_intervals = win.frameIntervals
                            n_fr_recorded = len(trial_fr_intervals)
                            # print(f"n_fr_recorded: {n_fr_recorded}")

                            # add to empty lists etc.
                            fr_int_per_trial.append(trial_fr_intervals)
                            fr_counter_per_trial.append(list(range(recorded_fr_counter,
                                                                   recorded_fr_counter + len(trial_fr_intervals))))
                            recorded_fr_counter += len(trial_fr_intervals)
                            exp_n_fr_recorded_list.append(exp_n_fr_recorded_list[-1] + n_fr_recorded)
                            cond_list.append(thisStair.name)

                            # get timings for each segment (probe1, ISI, probe2).
                            fr_diff_ms = [(expected_fr_sec - i) * 1000 for i in trial_fr_intervals]
                            # print(f"sum(fr_diff_ms): {sum(fr_diff_ms)}")

                            p1_durs = fr_diff_ms[:2]
                            p1_diff = sum(p1_durs)
                            if ISI > 0:
                                isi_durs = fr_diff_ms[2:-2]
                            else:
                                isi_durs = []
                            isi_diff = sum(isi_durs)

                            if ISI > -1:
                                p2_durs = fr_diff_ms[-2:]
                            else:
                                p2_durs = []
                            p2_diff = sum(p2_durs)

                            # print(f"\np1_durs: {p1_durs}, p1_diff: {p1_diff}\n"
                            #       f"isi_durs: {isi_durs}, isi_diff: {isi_diff}\n"
                            #       f"p2_durs: {p2_durs}, p2_diff: {p2_diff}\n")

                            # check for dropped frames (or frames that are too short)
                            # if timings are bad, repeat trial
                            if max(trial_fr_intervals) > max_fr_dur_sec or min(trial_fr_intervals) < min_fr_dur_sec:
                                if max(trial_fr_intervals) > max_fr_dur_sec:
                                    # print(f"\n\toh no! Frame too long! {max(trial_fr_intervals)} > {max_fr_dur_sec}")
                                    logging.WARN(f"\n\toh no! Frame too long! {max(trial_fr_intervals)} > {max_fr_dur_sec}")
                                elif min(trial_fr_intervals) < min_fr_dur_sec:
                                    # print(f"\n\toh no! Frame too short! {min(trial_fr_intervals)} < {min_fr_dur_sec}")
                                    logging.WARN(f"\n\toh no! Frame too short! {min(trial_fr_intervals)} < {min_fr_dur_sec}")
                                repeat = True
                                dropped_fr_trial_counter += 1
                                trial_number -= 1
                                thisStair.trialCount = thisStair.trialCount - 1  # so Kesten doesn't count this trial
                                win.frameIntervals = []
                                continueRoutine = False
                                trial_x_locs = [exp_n_fr_recorded_list[-2], exp_n_fr_recorded_list[-1]]
                                dropped_fr_trial_x_locs.append(trial_x_locs)
                                continue
                            # else:
                            #     print('Timing good')

                            # empty frameIntervals cache
                            win.frameIntervals = []

                        # these belong to the end of the answers section
                        repeat = False
                        continueRoutine = False


                # regardless of frameN, check for quit
                if event.getKeys(keyList=["escape"]):
                    thisExp.close()
                    core.quit()

                # If too many trials have had dropped frames, quit experiment
                if dropped_fr_trial_counter > max_droped_fr_trials:
                    while not event.getKeys():
                        # display end of experiment screen
                        too_many_dropped_fr.draw()
                        win.flip()
                    else:
                        # close and quit once a key is pressed
                        # thisExp.abort()  # or data files will save again on exit
                        thisExp.close()
                        win.close()
                        core.quit()

                # refresh the screen
                if continueRoutine:
                    win.flip()


        # add to thisExp for output csv
        thisExp.addData('trial_number', trial_number)
        thisExp.addData('trial_n_inc_rpt', actual_trials_inc_rpt)
        thisExp.addData('stair', stair_idx)
        thisExp.addData('stair_name', thisStair)
        thisExp.addData('step', step)
        thisExp.addData('separation', sep)
        thisExp.addData('sep_deg', sep_deg)
        thisExp.addData('neg_sep', neg_sep)
        thisExp.addData('ISI', ISI)
        thisExp.addData('ISI_selected_ms', ISI_selected_ms)
        thisExp.addData('ISI_actual_ms', ISI_actual_ms)
        thisExp.addData('isi_dur_fr', isi_dur_fr)
        thisExp.addData('congruent', congruent)
        thisExp.addData('flow_dir', flow_dir)
        thisExp.addData('probe_jump', target_jump)
        thisExp.addData('corner', corner)
        thisExp.addData('corner_name', corner_name)
        thisExp.addData('probeLum', probeLum)
        thisExp.addData('probeColor1', probeColor1)
        thisExp.addData('probeColor255', probeColor255)
        thisExp.addData('trial_response', resp.corr)
        thisExp.addData('resp.rt', resp.rt)
        thisExp.addData('probeColor1', probeColor1)
        thisExp.addData('probeColor255', probeColor255)
        thisExp.addData('probe_ecc', probe_ecc)
        thisExp.addData('BGspeed', flow_speed)
        thisExp.addData('orientation', orientation)
        thisExp.addData('vary_fixation', vary_fixation)
        thisExp.addData('t_fixation', t_fixation)
        thisExp.addData('p1_diff', p1_diff)
        thisExp.addData('isi_diff', isi_diff)
        thisExp.addData('p2_diff', p2_diff)
        thisExp.addData('monitor_name', monitor_name)
        thisExp.addData('this_colourSpace', this_colourSpace)
        thisExp.addData('this_bgColour', this_bgColour)
        thisExp.addData('selected_fps', fps)
        thisExp.addData('actual_fps', actualFrameRate)
        thisExp.addData('frame_tolerance_prop', frame_tolerance_prop)
        thisExp.addData('expName', expName)
        thisExp.addData('psychopy_version', psychopy_version)
        thisExp.addData('date', expInfo['date'])
        thisExp.addData('time', expInfo['time'])

        # tell psychopy to move to next trial
        thisExp.nextEntry()

        # update staircase based on whether response was correct or incorrect
        thisStair.newValue(resp.corr)


print("\nend of experiment loop, saving data\n")
# now exp is completed, save as '_output' rather than '_incomplete'
thisExp.dataFileName = os.path.join(save_dir, f'{participant_name}_{run_number}_output')
thisExp.close()


# plot frame intervals
if record_fr_durs:

    # flatten list of lists (fr_int_per_trial) to get len, min and max
    all_fr_intervals = [val for sublist in fr_int_per_trial for val in sublist]
    total_recorded_fr = len(all_fr_intervals)

    print(f"{dropped_fr_trial_counter}/{total_n_trials} trials with bad timing "
          f"(expected: {round(expected_fr_ms, 2)}ms, "
          f"frame_tolerance_ms: +/- {round(frame_tolerance_ms, 2)})")

    '''set colours for lines on plot.'''
    # get set of colours
    my_colours = fig_colours(n_stairs, alternative_colours=False)
    # associate colours with conditions
    colour_dict = {k: v for (k, v) in zip(stair_names_list, my_colours)}
    # make list of colours based on order of conditions
    cond_colour_list = [colour_dict[i] for i in cond_list]

    # plot frame intervals across the experiment with discontinuous line, coloured for each cond
    for trial_x_vals, trial_fr_durs, colour in zip(fr_counter_per_trial, fr_int_per_trial, cond_colour_list):
        plt.plot(trial_x_vals, trial_fr_durs, color=colour)

    # add legend with colours per condition
    legend_handles_list = []
    for cond in stair_names_list:
        leg_handle = mlines.Line2D([], [], color=colour_dict[cond], label=cond,
                                   marker='.', linewidth=.5, markersize=4)
        legend_handles_list.append(leg_handle)

    plt.legend(handles=legend_handles_list, fontsize=6, title='conditions', framealpha=.5)

    # add vertical lines to signify trials, shifted back so trials fall between lines
    fr_v_lines = [i - .5 for i in exp_n_fr_recorded_list]
    for trial_line in fr_v_lines:
        plt.axvline(x=trial_line, color='silver', linestyle='dashed', zorder=0)

    # add horizontal lines: green = expected frame duration, red = frame error tolerance
    plt.axhline(y=expected_fr_sec, color='grey', linestyle='dotted', alpha=.5)
    plt.axhline(y=max_fr_dur_sec, color='red', linestyle='dotted', alpha=.5)
    plt.axhline(y=min_fr_dur_sec, color='red', linestyle='dotted', alpha=.5)

    # shade trials that were repeated: red = bad timing, orange = user repeat
    for loc_pair in dropped_fr_trial_x_locs:
        print(loc_pair)
        x0, x1 = loc_pair[0] - .5, loc_pair[1] - .5
        plt.axvspan(x0, x1, color='red', alpha=0.15, zorder=0, linewidth=None)

    plt.title(f"{monitor_name}, {fps}Hz, {expInfo['date']}\n{dropped_fr_trial_counter}/{total_n_trials} trials."
              f"dropped fr (expected: {round(expected_fr_ms, 2)}ms, "
              f"frame_tolerance_ms: +/- {round(frame_tolerance_ms, 2)})")
    fig_name = f'{participant_name}_{run_number}_frames.png'
    print(f"fig_name: {fig_name}")
    plt.savefig(os.path.join(save_dir, fig_name))
    plt.close()


while not event.getKeys():
    # display end of experiment screen
    end_of_exp.draw()
    win.flip()
    core.wait(secs=5)
    end_of_exp.text = end_of_exp_text + "\n\nPress any key to continue."
    end_of_exp.draw()
    win.flip()
else:
    # logging.flush()  # write messages out to all targets
    thisExp.abort()  # or data files will save again on exit

    # close and quit once a key is pressed
    win.close()
    core.quit()
