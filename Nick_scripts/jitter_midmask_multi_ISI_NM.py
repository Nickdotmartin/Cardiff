from __future__ import division

import copy
import os
from datetime import datetime
from math import tan, sqrt

import numpy as np
from psychopy import __version__ as psychopy_version
from psychopy import gui, visual, core, data, event, monitors

from PsychoPy_tools import check_correct_monitor, get_pixel_mm_deg_values
from kestenSTmaxVal import Staircase

"""
This script takes: 
the probes from EXPERIMENT3_background_motion_SKR, and adds jitter.  
Each separation appears ONCE per run, rather than twice as in Exp1.
This script can include multiple ISIs.
Jitter points randomly change position every 30ms
Jitter colours are same as rad_flow (rbg255[76.5, 114.75, 76.5]), but could be higher contrast?.
Jitter can be random or radial (radial includes both inward and outward motion).
A random interval can be added to fixation time to reduce anticipatory saccades, maybe?
"""

# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# todo: change to asus_cal for testing
# Monitor config from monitor centre
monitor_name = 'HP_24uh'  # 'NickMac' 'asus_cal' 'Asus_VG24' 'HP_24uh' 'ASUS_2_13_240Hz' 'Iiyama_2_18' 'Nick_work_laptop'

# Store info about the experiment session
expName = 'jitter_multi_ISI_NM'  # from the Builder filename that created this script

expInfo = {'1_Participant': 'Nick_test',
           '2_Run_number': '1',
           '3_Probe_dur_in_frames_at_240hz': [2, 50],
           '4_fps': [240, 144, 60],
           '5_Trials_counter': [True, False],
           '6_Probe_orientation': ['tangent', 'radial'],
           '7_Background': ['jitter_random', 'jitter_radial', 'None'],
           '8_bg_speed_cond': ['Normal', 'Half-speed'],
           '9_vary_fixation': [False, True],
           '10_high_contrast': [False, True]}

# GUI
dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
if not dlg.OK:
    core.quit()  # user pressed escape

expInfo['date'] = datetime.now().strftime("%d/%m/%Y")
expInfo['time'] = datetime.now().strftime("%H:%M:%S")

# GUI SETTINGS
participant_name = expInfo['1_Participant']
run_number = int(expInfo['2_Run_number'])
probe_duration = int(expInfo['3_Probe_dur_in_frames_at_240hz'])
fps = int(expInfo['4_fps'])
trials_counter = expInfo['5_Trials_counter']
orientation = expInfo['6_Probe_orientation']  # 'tangent'
background = expInfo['7_Background']  # 'jitter'
bg_speed_cond = expInfo['8_bg_speed_cond']  # 'Normal'
vary_fixation = expInfo['9_vary_fixation']
high_contrast = expInfo['10_high_contrast']
print(f'high_contrast: {high_contrast }')

n_trials_per_stair = 25
probe_ecc = 4

# # background motion to start 70ms before probe1 (e.g., 17frames at 240Hz).
prelim_bg_flow_ms = 70
prelim_bg_flow_fr = int(prelim_bg_flow_ms * fps / 1000)

# VARIABLES
'''Distances between probes (spatially and temporally)
For 1probe condition, use separation==99.
For concurrent probes, use ISI==-1.
'''
separations = [0, 2, 4, 6]  # select from [0, 1, 2, 3, 6, 18, 99]
print(f'\nseparations: {separations}')
ISI_values = [0, 2, 4, 6]  # select from [-1, 0, 2, 4, 6, 9, 12, 24]
print(f'ISI_values: {ISI_values}')
# repeat separation values for each ISI e.g., [0, 0, 6, 6]
sep_vals_list = list(np.repeat(separations, len(ISI_values)))
print(f'sep_vals_list: {sep_vals_list}')
# ISI_vals_list cycles through ISIs e.g., [-1, 6, -1, 6]
ISI_vals_list = list(np.tile(ISI_values, len(separations)))
print(f'ISI_vals_list: {ISI_vals_list}')
# stair_names_list joins sep_vals_list and ISI_vals_list
# e.g., ['sep0_ISI-1', 'sep0_ISI6', 'sep6_ISI-1', 'sep6_ISI6']
stair_names_list = [f'sep{s}_ISI{c}' for s, c in zip(sep_vals_list, ISI_vals_list)]
print(f'stair_names_list: {stair_names_list}')
n_stairs = len(sep_vals_list)
print(f'n_stairs: {n_stairs}')

# FILENAME
filename = f'{_thisDir}{os.sep}' \
           f'{expName}{os.sep}' \
           f'{participant_name}{os.sep}' \
           f'{participant_name}_{run_number}{os.sep}' \
           f'{participant_name}_output'
# files are labelled as '_incomplete' unless entire script runs.
save_output_name = filename + '_incomplete'

# Experiment Handler
thisExp = data.ExperimentHandler(name=expName, version=psychopy_version,
                                 extraInfo=expInfo, runtimeInfo=None,
                                 savePickle=None, saveWideText=True,
                                 dataFileName=save_output_name)

# COLORS AND LUMINANCE
# Lum to Color255
LumColor255Factor = 2.39538706913372
# Color255 to Color1
Color255Color1Factor = 1/127.5  # Color255 * Color255Color1Factor -1
# Lum to Color1
Color1LumFactor = 2.39538706913372

maxLum = 106  # 255 RGB
minLum = 0.12  # 0 RGB

# get ACTUAL bgcolor details
rgb_bg_color = -0.1
if high_contrast is True:
    rgb_bg_color = 1
print(f'\nrgb_bg_color: {rgb_bg_color}')
bgColor255 = (rgb_bg_color + 1) * 127.5
print(f'bgColor255: {bgColor255}')
bgcolor_to_rgb1 = (rgb_bg_color+1)/2
print(f'bgcolor_to_rgb1: {bgcolor_to_rgb1}')
bgLum = bgcolor_to_rgb1*maxLum
print(f'bgLum: {bgLum}')
bgLumP = bgLum/maxLum
print(f'bgLumP: {bgLumP}')


# MONITOR SPEC
thisMon = monitors.Monitor(monitor_name)
this_width = thisMon.getWidth()
mon_dict = {'mon_name': monitor_name,
            'width': thisMon.getWidth(),
            'size': thisMon.getSizePix(),
            'dist': thisMon.getDistance(),
            'notes': thisMon.getNotes()}
print(f"mon_dict: {mon_dict}")

# double check using full screen in lab
display_number = 1  # 0 indexed, 1 for external display, 0 for internal
if monitor_name in ['ASUS_2_13_240Hz', 'asus_cal', 'Nick_work_laptop', 'NickMac']:
    display_number = 0
use_full_screen = True
if display_number > 0:
    use_full_screen = False
widthPix = int(mon_dict['size'][0])
heightPix = int(mon_dict['size'][1])
monitorwidth = float(mon_dict['width'])  # monitor width in cm
viewdist = float(mon_dict['dist'])  # viewing distance in cm
viewdistPix = widthPix / monitorwidth * viewdist
mon = monitors.Monitor(monitor_name, width=monitorwidth, distance=viewdist)
mon.setSizePix((widthPix, heightPix))
mon.save()


# WINDOW SPEC
win = visual.Window(monitor=mon, size=(widthPix, heightPix),
                    colorSpace='rgb255',
                    color=bgColor255,  # bgcolor from Martin's flow script, not bgColor255
                    winType='pyglet',  # I've added pyglet to make it work on pycharm/mac
                    pos=[1, -1],  # pos gives position of top-left of screen
                    units='pix',
                    screen=display_number,
                    allowGUI=False,
                    fullscr=use_full_screen)


# # check correct monitor details (fps, size) have been accessed.
try:
    check_correct_monitor(monitor_name=monitor_name,
                          actual_size=win.size,
                          actual_fps=win.getActualFrameRate(),
                          verbose=True)
    print('\nsize of a single pixel at 57cm')
    get_pixel_mm_deg_values(monitor_name=monitor_name)
    print('Monitor setting all correct')
except ValueError:
    print("Value error when running check_correct_monitor()")
    # don't save csv, no trials have happened yet
    thisExp.abort()

# ELEMENTS
# fixation bull eye
fixation = visual.Circle(win, radius=2, units='pix', lineColor='white',
                         fillColor='black')

# PROBEs
expInfo['6_Probe_size'] = '5pixels'
probeVert = [(0, 0), (1, 0), (1, 1), (2, 1), (2, -1), (1, -1),
             (1, -2), (-1, -2), (-1, -1), (0, -1)]

probe1 = visual.ShapeStim(win, vertices=probeVert, fillColor=(1.0, -1.0, 1.0),
                          lineWidth=0, opacity=1, size=1, interpolate=False)
probe2 = visual.ShapeStim(win, vertices=probeVert, fillColor=[-1.0, 1.0, -1.0],
                          lineWidth=0, opacity=1, size=1, interpolate=False)


# MASKs BEHIND PROBES
raisedCosTexture1 = visual.filters.makeMask(256, shape='raisedCosine',
                                            fringeWidth=0.3, radius=[1.0, 1.0])
mask_size = 150
probeMask1 = visual.GratingStim(win, mask=raisedCosTexture1, tex=None,
                                size=(mask_size, mask_size), units='pix',
                                colorSpace='rgb255', color=bgColor255)
probeMask2 = visual.GratingStim(win, mask=raisedCosTexture1, tex=None,
                                size=(mask_size, mask_size), units='pix',
                                colorSpace='rgb255', color=bgColor255)
probeMask3 = visual.GratingStim(win, mask=raisedCosTexture1, tex=None,
                                size=(mask_size, mask_size), units='pix',
                                colorSpace='rgb255', color=bgColor255)
probeMask4 = visual.GratingStim(win, mask=raisedCosTexture1, tex=None,
                                size=(mask_size, mask_size), units='pix',
                                colorSpace='rgb255', color=bgColor255)

# dist_from_fix is a constant to get 4dva distance from fixation,
dist_from_fix = round((tan(np.deg2rad(probe_ecc)) * viewdistPix) / sqrt(2))

# probe mask locations
probeMask1.setPos([dist_from_fix + 1, dist_from_fix + 1])
probeMask2.setPos([-dist_from_fix - 1, dist_from_fix + 1])
probeMask3.setPos([-dist_from_fix - 1, -dist_from_fix - 1])
probeMask4.setPos([dist_from_fix + 1, -dist_from_fix - 1])

# BACKGROUND flow_dots
if bg_speed_cond == 'Normal':
    # flow_speed = 500
    flow_speed = .01
elif bg_speed_cond == 'Half-speed':
    # flow_speed = 10
    flow_speed = .005
else:
    raise ValueError(f'background speed should be selected from drop down menu: Normal or Half-speed')
n_moving_dots = 1000  # 10000 - I've reduced the number since the dot_field_size is now much smaller
flow_dots_col = [76.5, 114.75, 76.5]  # greeney-grey in rgb255
if high_contrast is True:
    flow_dots_col = [0, 0, 0]
flow_dots = visual.ElementArrayStim(win, elementTex=None, elementMask='circle',
                                    units='pix', nElements=n_moving_dots, sizes=30,
                                    colorSpace='rgb255', colors=flow_dots_col)
dot_lives = np.random.random(n_moving_dots) * 10  # this will be the current life of each element in frames
dot_life_ms = 30  # 13.333333
# maximum number of frames before dots are redrawn.
dot_life_fr = int(round(dot_life_ms / (1000 / fps), 0))
print(f"dot_life_fr: {dot_life_fr}, {dot_life_ms}ms at {fps} fps.")

# dot_field_size = 5000  # taille is french for 'size', 'cut', 'trim', 'clip' etc - what does it actually do here?
dot_field_size = int(mon_dict['size'][1])  # used to set max pixels from centre that dots appear
print(f"\ndot_field_size: {dot_field_size}")

# x and y locations - remember its equivalent to (rand * dot_field_size) - (dot_field_size / 2)
x = np.random.rand(n_moving_dots) * dot_field_size - dot_field_size / 2
y = np.random.rand(n_moving_dots) * dot_field_size - dot_field_size / 2

# # Dots move randomly in smooth directions, then get new direction when reborn.
# # separate x and y motion is non-radial; same values for both gives radial motion.
# # random.normal, centered at 1 (no motion), with a sd given by flow_speed.
x_motion = np.random.normal(1, flow_speed, n_moving_dots)
y_motion = np.random.normal(1, flow_speed, n_moving_dots)
if background == 'jitter_radial':
    y_motion = x_motion
# # original Simon version (z, rather than x_motion/y_motion), harsh radial flow
# z = np.random.rand(n_moving_dots) * (maxDist + minDist) / 2
# # more subtle mix of in and out flow
# z = np.random.normal(1, .01, n_moving_dots)
# # no flow at all  (values > 1 flow out and < 1 flow in)
# z = np.ones(n_moving_dots)


# full screen mask to blend off edges and fade to black
# Create a raisedCosine mask array and assign it to a Grating stimulus (grey outside, transparent inside)
# this was useful http://www.cogsci.nl/blog/tutorials/211-a-bit-about-patches-textures-and-masks-in-psychopy
raisedCosTexture2 = visual.filters.makeMask(1080, shape='raisedCosine', fringeWidth=0.6, radius=[1.0, 1.0])
invRaisedCosTexture = -raisedCosTexture2  # inverts mask to blur edges instead of center
blankslab = np.ones((1080, 420))  # create blank slabs to put to left and right of image
mmask = np.append(blankslab, invRaisedCosTexture, axis=1)  # append blank slab to left
mmask = np.append(mmask, blankslab, axis=1)  # and right
peripheral_mask = visual.GratingStim(win, mask=mmask, tex=None, contrast=1.0,
                              size=(widthPix, heightPix), units='pix', color='black')
# changed peripheral_mask color from grey
# above fades to black round edges which makes screen edges less visible


# central mask behind probemasks
# raisedCosTexture1 = visual.filters.makeMask(256, shape='raisedCosine',
#                                             fringeWidth=0.01, radius=[1.0, 1.0])
raisedCosTexture1 = visual.filters.makeMask(256, shape='raisedCosine',
                                            fringeWidth=0.3, radius=[1.0, 1.0])
print(f"dist_from_fix: {dist_from_fix}")
static_mask_size = 400  # could use dist_from_fix
static_mask = visual.GratingStim(win, mask=raisedCosTexture1, tex=None,
                                 size=(static_mask_size, static_mask_size), units='pix',
                                 colorSpace='rgb255', color=bgColor255)


# # # get start locations for static dots
# n_static_dots = int(n_moving_dots/5)
# phi = np.random.uniform(0, 2*np.pi, n_static_dots)
# r = np.random.uniform(0, static_mask_size/2, n_static_dots)
# static_x = r * np.cos(phi)
# static_y = r * np.sin(phi)
# static_xy_pos = np.array([static_x, static_y]).transpose()
# static_dots = visual.ElementArrayStim(win, elementTex=None, elementMask='circle',
#                                       units='pix', nElements=n_static_dots, sizes=30,
#                                       colorSpace='rgb255', xys=static_xy_pos,
#                                       colors=flow_dots_col)


# WRAP FUNCTION: no longer needed
# add comment to this function.  Move function to the top?
# function for wrapping flow_dots back into volume
# its is used as WrapPoints(z, minDist, maxDist)
# Any dots with a z (depth) value out of bounds are transformed to be in bounds
# note: does not necessarily appear at edge of screen.  If min = -500, max = 500; then difference is 1000.
# any dot with value of -700 will appear at 300 (-700 + 100).
# any dot with value of 1111 will appear at 111 (1111 - 1000).
# def WrapPoints(ii, imin, imax):
#     lessthanmin = (ii < imin)
#     ii[lessthanmin] = ii[lessthanmin] + (imax-imin)
#     morethanmax = (ii > imax)
#     ii[morethanmax] = ii[morethanmax] - (imax-imin)
#     return ii
# 
# '''
# https://stackoverflow.com/questions/34082748/how-to-create-a-continuous-visual-window-background-wrapping-of-background/34122312#34122312
# 
# BORDER_RIGHT = 1  # for 'norm' units, this is the coordinate of the right border
# 
# # Animation
# for frame in range(360):
#     # Determine location of elements on next frame
#     stim.xys += (0.01, 0)  # move all to the right
#     exceeded = stim.xys[:,0] > BORDER_RIGHT  # index of elements whose x-value exceeds the border
#     stim.xys[exceeded] *= (-1, 1)  # for these indicies, mirror the x-coordinate but keep the y-coordinate
# 
#     # Show it 
#     stim.draw()
#     win.flip()'''

# MOUSE - Hide cursor
myMouse = event.Mouse(visible=False)

# # KEYBOARD
resp = event.BuilderKeyResponse()

# INSTRUCTION
instructions = visual.TextStim(win=win, name='instructions',
                               text="\n\n\nPlease maintain focus on the black cross at the centre of the screen.\n\n"
                                    "A small white probe will briefly flash on screen,\n"
                                    "press the key related to the location of the probe:\n\n"
                                    "[4] top-left\t\t\t[5] top-right\n\n\n\n"
                                    "[1] bottom-left\t\t\t[2] bottom-right.\n\n\n"
                                    "Do not rush, aim to be as accurate as possible,\n"
                                    "but if you did not see the probe, please guess.\n\n"
                                    "If you pressed a wrong key by mistake, you can:\n"
                                    "continue or\n"
                                    "press [r] or [9] to redo the previous trial.\n\n"
                                    "Press any key to start.",
                               font='Arial', height=20,
                               colorSpace='rgb255', color='white')
while not event.getKeys():
    fixation.setRadius(3)
    fixation.draw()
    instructions.draw()
    win.flip()

# Trial counter
trials_counter = visual.TextStim(win=win, name='trials_counter', text="???",
                                 font='Arial', height=20,
                                 # default set to black (e.g., invisible)
                                 color='black',
                                 pos=[-widthPix * .45, -heightPix * .45])
if trials_counter:
    # if trials counter yes, change colour to white.
    trials_counter.color = 'white'

# BREAKS
total_n_trials = int(n_trials_per_stair * n_stairs)
take_break = int(total_n_trials/4)
print(f"\ntake_break every {take_break} trials.")
breaks = visual.TextStim(win=win, name='breaks',
                         text="turn on the light and take at least 30-seconds break.\n\n"
                              "When you are ready to continue, press any key.",
                         font='Arial', height=20, colorSpace='rgb255', color=[1, 1, 1])

end_of_exp = visual.TextStim(win=win, name='end_of_exp',
                             text="You have completed this experiment.\n"
                                  "Thank you for your time.\n\n"
                                  "Press any key to return to the desktop.",
                             font='Arial', height=20)


# STAIRCASE
expInfo['stair_list'] = list(range(n_stairs))
expInfo['n_trials_per_stair'] = n_trials_per_stair
print('\nexpInfo (dict)')
for k, v in expInfo.items():
    print(f"{k}: {v}")

stairStart = maxLum
miniVal = bgLum
maxiVal = maxLum

stairs = []
for stair_idx in expInfo['stair_list']:
    thisInfo = copy.copy(expInfo)
    thisInfo['stair_idx'] = stair_idx
    stair_name = stair_names_list[stair_idx]

    thisStair = Staircase(name=stair_name,
                          type='simple',
                          value=stairStart,
                          C=stairStart * 0.6,  # step_size, typically 60% of reference stimulus
                          minRevs=3,
                          minTrials=n_trials_per_stair,
                          minVal=miniVal,
                          maxVal=maxiVal,
                          targetThresh=0.75,
                          extraInfo=thisInfo)
    stairs.append(thisStair)
    
# EXPERIMENT
trial_number = 0
print('\n*** exp loop*** \n\n')
for step in range(n_trials_per_stair):
    np.random.shuffle(stairs)
    for thisStair in stairs:

        trial_number = trial_number + 1
        stair_idx = thisStair.extraInfo['stair_idx']
        print(f"\ntrial_number: {trial_number}, stair_idx: {stair_idx}, thisStair: {thisStair}, step: {step}")

        sep = sep_vals_list[stair_idx]
        ISI = ISI_vals_list[stair_idx]
        print(f"ISI: {ISI}, sep: {sep}")

        # direction in which the probe jumps : CW or CCW
        target_jump = np.random.choice([1, -1])

        # staircase varies probeLum
        probeLum = thisStair.next()
        probeColor255 = int(probeLum * LumColor255Factor)  # rgb255 are ints.
        probeColor1 = (probeColor255 * Color255Color1Factor) - 1
        probe1.color = [probeColor1, probeColor1, probeColor1]
        probe2.color = [probeColor1, probeColor1, probeColor1]
        print(f'probeLum: {probeLum}, probeColor255: {probeColor255}, probeColor1: {probeColor1}')


        # PROBE LOCATIONS
        # corners go CCW(!) 45=top-right, 135=top-left, 225=bottom-left, 315=bottom-right
        corner = np.random.choice([45, 135, 225, 315])
        print(f'\tcorner: {corner}')

        # set probe ori
        if corner == 45:
            # in top-right corner, both x and y increase (right and up)
            p1_x = dist_from_fix * 1
            p1_y = dist_from_fix * 1
            #  'orientation' here refers to the relationship between probes,
            #  whereas probe1.ori refers to rotational angle of probe stimulus
            if orientation == 'tangent':
                if target_jump == 1:  # CCW
                    probe1.ori = 0
                    probe2.ori = 180
                    # probe2 is left and up from probe1
                    probe2.pos = [p1_x - sep + 1, p1_y + sep]
                elif target_jump == -1:  # CW
                    probe1.ori = 180
                    probe2.ori = 0
                    # probe2 is right and down from probe1
                    probe2.pos = [p1_x + sep - 1, p1_y - sep]
            elif orientation == 'radial':
                if target_jump == 1:  # inward
                    probe1.ori = 270
                    probe2.ori = 90
                    # probe2 is left and down from probe1
                    probe2.pos = [p1_x - sep + 1, p1_y - sep]
                elif target_jump == -1:  # outward
                    probe1.ori = 90
                    probe2.ori = 270
                    # probe2 is right and up from probe1
                    probe2.pos = [p1_x + sep - 1, p1_y + sep]
        elif corner == 135:
            # in top-left corner, x decreases (left) and y increases (up)
            p1_x = dist_from_fix * -1
            p1_y = dist_from_fix * 1
            if orientation == 'tangent':
                if target_jump == 1:  # CCW
                    probe1.ori = 90
                    probe2.ori = 270
                    # probe2 is right and up from probe1
                    probe2.pos = [p1_x + sep - 1, p1_y + sep]
                elif target_jump == -1:  # CW
                    probe1.ori = 270
                    probe2.ori = 90
                    # probe2 is left and down from probe1
                    probe2.pos = [p1_x - sep + 1, p1_y - sep]
            elif orientation == 'radial':
                if target_jump == 1:  # inward
                    probe1.ori = 180
                    probe2.ori = 0
                    # probe2 is right and down from probe1
                    probe2.pos = [p1_x + sep - 1, p1_y - sep]
                elif target_jump == -1:  # outward
                    probe1.ori = 0
                    probe2.ori = 180
                    # probe2 is left and up from probe1
                    probe2.pos = [p1_x - sep + 1, p1_y + sep]
        elif corner == 225:
            # in bottom left corner, both x and y decrease (left and down)
            p1_x = dist_from_fix * -1
            p1_y = dist_from_fix * -1
            if orientation == 'tangent':
                if target_jump == 1:  # CCW
                    probe1.ori = 180
                    probe2.ori = 0
                    probe2.pos = [p1_x + sep - 1, p1_y - sep]
                elif target_jump == -1:  # CW
                    probe1.ori = 0
                    probe2.ori = 180
                    probe2.pos = [p1_x - sep + 1, p1_y + sep]
            elif orientation == 'radial':
                if target_jump == 1:  # inward
                    probe1.ori = 90
                    probe2.ori = 270
                    # probe2 is right and up from probe1
                    probe2.pos = [p1_x + sep - 1, p1_y + sep]
                elif target_jump == -1:  # outward
                    probe1.ori = 270
                    probe2.ori = 90
                    # probe2 is left and down from probe1
                    probe2.pos = [p1_x - sep + 1, p1_y - sep]
        else:
            corner = 315
            # in bottom-right corner, x increases (right) and y decreases (down)
            p1_x = dist_from_fix * 1
            p1_y = dist_from_fix * -1
            if orientation == 'tangent':
                if target_jump == 1:  # CCW
                    probe1.ori = 270
                    probe2.ori = 90
                    probe2.pos = [p1_x - sep + 1, p1_y - sep]
                elif target_jump == -1:  # CW
                    probe1.ori = 90
                    probe2.ori = 270
                    probe2.pos = [p1_x + sep - 1, p1_y + sep]
            elif orientation == 'radial':
                if target_jump == 1:  # inward
                    probe1.ori = 0
                    probe2.ori = 180
                    # probe2 is left and up from probe1
                    probe2.pos = [p1_x - sep + 1, p1_y + sep]
                elif target_jump == -1:  # outward
                    probe1.ori = 180
                    probe2.ori = 0
                    # probe2 is right and down from probe1
                    probe2.pos = [p1_x + sep - 1, p1_y - sep]

        probe1.pos = [p1_x, p1_y]

        # to avoid fixation times always being the same which might increase
        # anticipatory effects,
        # add in a random number of frames (up to 1 second) to fixation time
        vary_fix = 0
        if vary_fixation:
            vary_fix = np.random.randint(0, fps)

        # timing in frames
        # fixation time is now 70ms shorter than rad_flow1, as we can have
        # priliminary bg_motion.
        t_fixation = (fps / 2) - prelim_bg_flow_fr + vary_fix
        t_bg_motion = t_fixation + prelim_bg_flow_fr
        t_probe_1 = t_bg_motion + probe_duration
        t_ISI = t_probe_1 + ISI
        t_probe_2 = t_ISI + probe_duration
        t_response = t_probe_2 + 10000 * fps  # essentially unlimited time to respond


        # repeat the trial if [r] has been pressed
        repeat = True
        while repeat:
            frameN = -1

            # Break after trials 100 and 200, or whatever set in take_break
            if (trial_number % take_break == 1) & (trial_number > 1):
                continueRoutine = False
                breaks.draw()
                win.flip()
                while not event.getKeys():
                    continueRoutine = True
            else:
                continueRoutine = True

            while continueRoutine:
                frameN = frameN + 1

                # # update background dot motion
                # update dot's life each frame
                dot_lives = dot_lives + 1

                # get array of x and y pos by calling .xys
                flow_dot_xs = flow_dots.xys[:, 0]
                flow_dot_ys = flow_dots.xys[:, 1]

                # find the dead elements and reset their life
                deadElements = (dot_lives > dot_life_fr)  # numpy vector, not standard python
                dot_lives[deadElements] = 0

                # New x, y locations for dots that are re-born (random array, same shape as dead elements)
                flow_dot_xs[deadElements] = np.random.random(
                    flow_dot_xs[deadElements].shape) * dot_field_size - dot_field_size / 2.0
                flow_dot_ys[deadElements] = np.random.random(
                    flow_dot_ys[deadElements].shape) * dot_field_size - dot_field_size / 2.0

                # each frame, update dot positions by dividing x_location by x_motion
                x_flow = flow_dot_xs / x_motion
                y_flow = flow_dot_ys / y_motion
                xy_pos = np.array([x_flow, y_flow]).transpose()
                flow_dots.xys = xy_pos


                # FIXATION
                if t_fixation >= frameN > 0:
                    # before fixation has finished
                    trials_counter.text = f"{trial_number}/{total_n_trials}"
                    flow_dots.draw()

                    static_mask.draw()
                    # static_dots.draw()

                    probeMask1.draw()
                    probeMask2.draw()
                    probeMask3.draw()
                    probeMask4.draw()
                    peripheral_mask.draw()
                    fixation.setRadius(3)
                    fixation.draw()
                    trials_counter.draw()

                # Background motion prior to probe1
                if t_bg_motion >= frameN > t_fixation:
                    flow_dots.draw()

                    static_mask.draw()
                    # static_dots.draw()

                    probeMask1.draw()
                    probeMask2.draw()
                    probeMask3.draw()
                    probeMask4.draw()
                    peripheral_mask.draw()
                    fixation.setRadius(3)
                    fixation.draw()
                    trials_counter.draw()

                    # reset timer to start with probe1 presentation.
                    resp.clock.reset()

                # PROBE 1: after background motion, before end of probe1 interval
                if t_probe_1 >= frameN > t_bg_motion:
                    flow_dots.draw()

                    static_mask.draw()
                    # static_dots.draw()

                    probeMask1.draw()
                    probeMask2.draw()
                    probeMask3.draw()
                    probeMask4.draw()
                    peripheral_mask.draw()
                    probe1.draw()
                    if ISI == -1:  # SIMULTANEOUS CONDITION (concurrent)
                        if sep <= 18:  # don't draw 2nd probe in 1probe cond (sep==99)
                            probe2.draw()
                    fixation.setRadius(3)
                    fixation.draw()
                    trials_counter.draw()

                # ISI
                if t_ISI >= frameN > t_probe_1:
                    flow_dots.draw()

                    static_mask.draw()
                    # static_dots.draw()

                    probeMask1.draw()
                    probeMask2.draw()
                    probeMask3.draw()
                    probeMask4.draw()
                    peripheral_mask.draw()
                    fixation.setRadius(3)
                    fixation.draw()
                    trials_counter.draw()

                # PROBE 2: after ISI but before end of probe2 interval
                if t_probe_2 >= frameN > t_ISI:
                    flow_dots.draw()

                    static_mask.draw()
                    # static_dots.draw()

                    probeMask1.draw()
                    probeMask2.draw()
                    probeMask3.draw()
                    probeMask4.draw()
                    peripheral_mask.draw()
                    if ISI >= 0:
                        if sep <= 18:  # don't draw 2nd probe in 1probe cond (sep==99)
                            probe2.draw()
                    fixation.setRadius(3)
                    fixation.draw()
                    trials_counter.draw()

                # ANSWER: after probe 2 interval until response
                if frameN > t_probe_2:
                    flow_dots.draw()

                    static_mask.draw()
                    # static_dots.draw()

                    probeMask1.draw()
                    probeMask2.draw()
                    probeMask3.draw()
                    probeMask4.draw()
                    peripheral_mask.draw()
                    fixation.setRadius(2)
                    fixation.draw()
                    trials_counter.draw()

                    # Response
                    theseKeys = event.getKeys(keyList=['num_5', 'num_4', 'num_1',
                                                       'num_2', 'w', 'q', 'a', 's'])
                    if len(theseKeys) > 0:  # at least one key was pressed
                        resp.keys = theseKeys[-1]  # just the last key pressed
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

                        repeat = False
                        continueRoutine = False

                # regardless of frameN, check for quit
                if event.getKeys(keyList=["escape"]):
                    thisExp.close()
                    core.quit()

                # redo the trial if I think I made a mistake
                if event.getKeys(keyList=["r"]) or event.getKeys(keyList=['num_9']):
                    repeat = True
                    continueRoutine = False
                    continue

                # refresh the screen
                if continueRoutine:
                    win.flip()

        # add to exp dict
        thisExp.addData('trial_number', trial_number)
        thisExp.addData('stair', stair_idx)
        thisExp.addData('stair_name', thisStair)
        thisExp.addData('step', step)
        thisExp.addData('separation', sep)
        thisExp.addData('ISI', ISI)
        thisExp.addData('probe_jump', target_jump)
        thisExp.addData('corner', corner)
        thisExp.addData('probeLum', probeLum)
        thisExp.addData('trial_response', resp.corr)
        thisExp.addData('resp.rt', resp.rt)
        thisExp.addData('probeColor1', probeColor1)
        thisExp.addData('probeColor255', probeColor255)
        thisExp.addData('probe_ecc', probe_ecc)
        thisExp.addData('background', background)
        thisExp.addData('bg_speed_cond', bg_speed_cond)
        thisExp.addData('BGspeed', flow_speed)
        thisExp.addData('orientation', orientation)
        thisExp.addData('bgLumP', bgLumP)
        thisExp.addData('bgLum', bgLum)
        thisExp.addData('bgColor255', bgColor255)
        thisExp.addData('weber_thr', (probeLum-bgLum)/probeLum)
        thisExp.addData('vary_fixation', vary_fixation)
        thisExp.addData('expName', expName)
        thisExp.addData('monitor_name', monitor_name)
        thisExp.addData('selected_fps', fps)

        thisExp.nextEntry()

        thisStair.newValue(resp.corr)  # so that the staircase adjusts itself

print("end of exp loop, saving data")
thisExp.dataFileName = filename
thisExp.close()

while not event.getKeys():
    # display end of experiment screen
    end_of_exp.draw()
    win.flip()
else:
    # close and quit once a key is pressed
    win.close()
    core.quit()
