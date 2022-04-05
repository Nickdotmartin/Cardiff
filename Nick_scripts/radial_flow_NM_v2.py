from __future__ import division

import copy
import os
from datetime import datetime
from math import *

import numpy as np
from psychopy import __version__ as psychopy_version
from psychopy import gui, visual, core, data, event, monitors

from PsychoPy_tools import check_correct_monitor, get_pixel_mm_deg_values
from kestenSTmaxVal import Staircase

"""
This script takes: 
the probes from EXPERIMENT3_background_motion_SKR, and adds the option for tangent or radial jump.
the background radial motion is taken from integration_RiccoBloch_flow_new.
ISI is always >=0 (no simultaneous probes).
"""

# todo: This script uses colorspace=rgb, but it should be rgb255.
#  I'll keep it as is for now, but I need to use these vals below for bglum and deltaLum.
#  flow_bgcolor = [-0.1, -0.1, -0.1]  # dark grey converts to:
#  rgb: -0.1 = rgb1: .45 = rgb255: 114.75 = lum: 47.8.
#  for future ref, to match exp1 it should be flow_bgcolor = [-0.6, -0.6, -0.6]  # dark grey

# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# Monitor config from monitor centre
monitor_name = 'HP_24uh'  # 'NickMac' 'asus_cal' 'Asus_VG24' 'HP_24uh' 'ASUS_2_13_240Hz'
display_number = 1  # 0 indexed, 1 for external display

# Store info about the experiment session
expName = 'rad_flow_2'  # from the Builder filename that created this script

expInfo = {'1_Participant': 'Nick_60hz_test_1',
           '2_Probe_dur_in_frames_at_240hz': [2, 50],
           '3_fps': [240, 144, 60],
           '4_ISI_dur_in_ms': [100, 50, 41.67, 37.5, 33.34, 25, 16.67, 8.33, 0],
           '5_Probe_orientation': ['radial', 'tangent'],
           '6_Probe_size': ['5pixels', '6pixels', '3pixels'],
           '7_Trials_counter': [True, False],
           '8_Background': ['flow_rad', 'None'],
           '9_bg_speed_cond': ['Normal', 'Half-speed']
           }

# GUI
dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
if not dlg.OK:
    core.quit()  # user pressed cancel

expInfo['date'] = datetime.now().strftime("%d/%m/%Y")
expInfo['time'] = datetime.now().strftime("%H:%M:%S")

# GUI SETTINGS
participant_name = expInfo['1_Participant']
probe_duration = int(expInfo['2_Probe_dur_in_frames_at_240hz'])
fps = int(expInfo['3_fps'])
orientation = expInfo['5_Probe_orientation']
Probe_size = expInfo['6_Probe_size']
trials_counter = expInfo['7_Trials_counter']
background = expInfo['8_Background']
bg_speed_cond = expInfo['9_bg_speed_cond']

# ISI timing in ms and frames
'''ISI can be given (roughly) in ms, for any monitor it will try to match that value in frames.
The script uses on frames so ISI will be used in that format.
The actual ms timing is given for record.
This means that the experiment should have similar ms timings on monitors with different fps
milliseconds: [100, 50, 41.66, 37.5, 33.34, 25, 16.67, 8.33, 0]
frames@240hz: [24,  12,  10,    9,    8,     6,  4,    2,    0]
'''
ISI_selected_ms = float(expInfo['4_ISI_dur_in_ms'])
ISI_frames = int(ISI_selected_ms * fps / 1000)
ISI_actual_ms = (1/fps) * ISI_frames * 1000
ISI = ISI_frames
print(f"\nSelected {ISI_selected_ms}ms ISI.\n"
      f"At {fps}Hz this is {ISI_frames} frames which each take {ISI_actual_ms}ms.\n")

# VARIABLES
n_trials_per_stair = 25
probe_ecc = 4

# background motion to start 70ms before probe1 (e.g., 17frames at 240Hz).
prelim_bg_flow_ms = 70
prelim_bg_flow_fr = int(prelim_bg_flow_ms * fps / 1000)

# Distances between probes & flow direction
separation_values = [18, 6, 3, 2, 1, 0]
'''each separation value appears in 2 stairs, e.g.,
stair1 will be sep=18, flow_dir=inwards; stair2 will be sep=18, flow_dir=outwards etc.
e.g., separations = [18, 18, 6, 6, 3, 3, 2, 2, 1, 1, 0, 0]
this study does not include the two single probe conditions (labeled 99 in previous exp)
'''
separations = list(np.repeat(separation_values, 2))
n_stairs = len(separations)

# # main contrast is whether the background and target motion is in same or opposite direction.
# congruence_list: 1=congruent/same, -1=incongruent/different
congruence_list = [1, -1]*len(separation_values)

# FILENAME
filename = f'{_thisDir}{os.sep}' \
           f'{expName}{os.sep}' \
           f'{participant_name}{os.sep}' \
           f'ISI_{ISI}_probeDur{probe_duration}{os.sep}' \
           f'{participant_name}'

# Experiment Handler
thisExp = data.ExperimentHandler(name=expName, version=psychopy_version,
                                 extraInfo=expInfo, runtimeInfo=None,
                                 savePickle=None, saveWideText=True,
                                 dataFileName=filename)

# COLORS AND LUMINANCE
# Lum to Color255
LumColor255Factor = 2.39538706913372
# Color255 to Color1
Color255Color1Factor = 1/127.5  # Color255 * Color255Color1Factor -1
# Lum to Color1
Color1LumFactor = 2.39538706913372

maxLum = 106  # 255 RGB
minLum = 0.12  # 0 RGB
# maxColor255 = 255
# minColor255 = 0
# maxColor1 = 1
# minColor1 = -1
# bgLumP = 20
# bgLum = maxLum * bgLumP / 100
# bgColor255 = bgLum * LumColor255Factor  # I could switch to using this.

#  rgb: -0.1 = rgb1: .45 = rgb255: 114.75 = lum: 47.8
flow_bgcolor = [-0.1, -0.1, -0.1]  # dark grey
# flow_bgcolor = [-0.6, -0.6, -0.6]  # these values would be equivalent to exp1a

if background == 'flow_rad':
    # background colour: use darker grey.  set once here and use elsewhere
    bgcolor = flow_bgcolor
else:
    # bgcolor = bgColor255
    bgcolor = flow_bgcolor

# get ACTUAL bgcolor details
actual_bg_color = bgcolor[0]
print(f'actual_bg_color: {actual_bg_color}')
bgcolor_to_rgb255 = (actual_bg_color + 1) * 127.5
# print(f'bgcolor_to_rgb255: {bgcolor_to_rgb255}')
bgcolor_to_rgb1 = (actual_bg_color+1)/2
print(f'bgcolor_to_rgb1: {bgcolor_to_rgb1}')
bgcolor_to_lum = bgcolor_to_rgb1*maxLum
# print(f'bgcolor_to_lum: {bgcolor_to_lum}')
bglum_as_prop_maxLum = bgcolor_to_lum/maxLum
# print(f'bglum_as_prop_maxLum: {bglum_as_prop_maxLum}')
bgLumP = bglum_as_prop_maxLum
print(f'bgLumP: {bgLumP}')
bgLum = bgcolor_to_lum
print(f'bgLum: {bgLum}')
bgColor255 = bgcolor_to_rgb255
print(f'bgColor255: {bgColor255}')


# MONITOR SPEC
thisMon = monitors.Monitor(monitor_name)
this_width = thisMon.getWidth()
mon_dict = {'mon_name': monitor_name,
            'width': thisMon.getWidth(),
            'size': thisMon.getSizePix(),
            'dist': thisMon.getDistance(),
            'notes': thisMon.getNotes()
            }
print(f"mon_dict: {mon_dict}")

# double check using full screen in lab
if monitor_name == 'ASUS_2_13_240Hz':
    display_number = 0
use_full_screen = True
if display_number > 0:
    use_full_screen = False
widthPix = mon_dict['size'][0]
heightPix = mon_dict['size'][1]
monitorwidth = mon_dict['width']  # monitor width in cm
viewdist = mon_dict['dist']  # viewing distance in cm
viewdistPix = widthPix/monitorwidth*viewdist
mon = monitors.Monitor(monitor_name, width=monitorwidth, distance=viewdist)
mon.setSizePix((widthPix, heightPix))
mon.save()


# WINDOW SPEC
win = visual.Window(monitor=mon, size=(widthPix, heightPix),
                    colorSpace='rgb',
                    color=bgcolor,  # bgcolor from Martin's flow script, not bgColor255
                    winType='pyglet',  # I've added pyglet to make it work on pycharm/mac
                    pos=[1, -1],  # pos gives position of top-left of screen
                    units='pix',
                    screen=display_number,
                    allowGUI=False,
                    fullscr=use_full_screen,
                    )


# # check correct monitor details (fps, size) have been accessed.
try:
    check_correct_monitor(monitor_name=monitor_name,
                          actual_size=win.size,
                          actual_fps=win.getActualFrameRate(),
                          verbose=True)
    print('\nsize of a single pixel at 57cm')
    get_pixel_mm_deg_values(monitor_name=monitor_name, use_diagonal=False)
    print('Monitor setting all correct')
except ValueError:
    print("Value error when running check_correct_monitor()")
    # don't save csv, no trials have happened yet
    thisExp.abort()

# CLOCK
trialClock = core.Clock()


# ELEMENTS
# fixation bull eye
if background == 'flow_rad':
    fixation = visual.Circle(win, radius=2, units='pix', lineColor='black', fillColor='black')
else:
    fixation = visual.Circle(win, radius=2, units='pix', lineColor='white', fillColor='black')


# PROBEs
# probe sizes choice
if expInfo['6_Probe_size'] == '6pixels':  # 6 pixels
    probeVert = [(0, 0), (1, 0), (1, 1), (2, 1),
                 (2, -2), (-1, -2), (-1, -1), (0, -1)]
elif expInfo['6_Probe_size'] == '3pixels':  # 3 pixels
    probeVert = [(0, 0), (1, 0), (1, 1), (2, 1), (2, 0), (1, 0), (1, -1),
                 (0, -1), (0, -2), (-1, -2), (-1, -2), (-1, -1), (0, -1)]
else:  # 5 pixels
    # default setting is expInfo['6_Probe_size'] == '5pixels':
    expInfo['6_Probe_size'] = '5pixels'
    probeVert = [(0, 0), (1, 0), (1, 1), (2, 1), (2, -1), (1, -1),
                 (1, -2), (-1, -2), (-1, -1), (0, -1)]

probe1 = visual.ShapeStim(win, vertices=probeVert, fillColor=(1.0, -1.0, 1.0),
                          lineWidth=0, opacity=1, size=1, interpolate=False)
probe2 = visual.ShapeStim(win, vertices=probeVert, fillColor=[-1.0, 1.0, -1.0],
                          lineWidth=0, opacity=1, size=1, interpolate=False)


# MASK BEHIND PROBES
raisedCosTexture1 = visual.filters.makeMask(256, shape='raisedCosine',
                                            fringeWidth=0.3, radius=[1.0, 1.0])
mask_size = 150
probeMask1 = visual.GratingStim(win, mask=raisedCosTexture1, tex=None,
                                size=(mask_size, mask_size), units='pix', color=bgcolor)
probeMask2 = visual.GratingStim(win, mask=raisedCosTexture1, tex=None,
                                size=(mask_size, mask_size), units='pix', color=bgcolor)
probeMask3 = visual.GratingStim(win, mask=raisedCosTexture1, tex=None,
                                size=(mask_size, mask_size), units='pix', color=bgcolor)
probeMask4 = visual.GratingStim(win, mask=raisedCosTexture1, tex=None,
                                size=(mask_size, mask_size), units='pix', color=bgcolor)


# BACKGROUND
# flow_dots
if bg_speed_cond is 'Normal':
    flow_speed = 0.2
if bg_speed_cond is 'Half-speed':
    flow_speed = 0.1
else:
    raise ValueError(f'background speed should be selected from drop down menu: Normal or Half-speed')
nDots = 10000
flow_dots = visual.ElementArrayStim(win, elementTex=None, elementMask='circle',
                                    units='pix', nElements=nDots, sizes=10,
                                    colors=[flow_bgcolor[0]-0.3,
                                            flow_bgcolor[1],
                                            flow_bgcolor[2]-0.3])

# full screen mask to blend off edges and fade to black
# Create a raisedCosine mask array and assign it to a Grating stimulus (grey outside, transparent inside)
# this was useful http://www.cogsci.nl/blog/tutorials/211-a-bit-about-patches-textures-and-masks-in-psychopy
raisedCosTexture2 = visual.filters.makeMask(1080, shape='raisedCosine', fringeWidth=0.6, radius=[1.0, 1.0])
invRaisedCosTexture = -raisedCosTexture2  # inverts mask to blur edges instead of center
blankslab = np.ones((1080, 420))  # create blank slabs to put to left and right of image
mmask = np.append(blankslab, invRaisedCosTexture, axis=1)  # append blank slab to left
mmask = np.append(mmask, blankslab, axis=1)  # and right
dotsMask = visual.GratingStim(win, mask=mmask, tex=None, contrast=1.0,
                              size=(widthPix, heightPix), units='pix', color='black')
# changed dotsmask color from grey
# above fades to black round edges which makes screen edges less visible

# function for wrapping flow_dots back into volume
# its is used as WrapPoints(z, minDist, maxDist)
# Any dots with a z (depth) value out of bounds are transformed to be in bounds
def WrapPoints(ii, imin, imax):
    lessthanmin = (ii < imin)
    ii[lessthanmin] = ii[lessthanmin] + (imax-imin)
    morethanmax = (ii > imax)
    ii[morethanmax] = ii[morethanmax] - (imax-imin)


taille = 5000  # french for 'size', 'cut', 'trim', 'clip' etc
minDist = 0.5
maxDist = 5


# MOUSE - Hide cursor
myMouse = event.Mouse(visible=False)

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
                               colorSpace='rgb', color=[1, 1, 1],
                               )
while not event.getKeys():
    fixation.setRadius(3)
    fixation.draw()
    instructions.draw()
    win.flip()

# Trial counter
trials_counter = visual.TextStim(win=win, name='trials_counter', text="???",
                                 font='Arial', height=20,
                                 # default set to black (e.g., invisible)
                                 color=[-1.0, -1.0, -1.0],
                                 pos=[-widthPix*.45, -heightPix*.45]
                                 )
if trials_counter:
    # if trials counter yes, change colour to white.
    trials_counter.color = [1, 1, 1]

# BREAKS
total_n_trials = int(n_trials_per_stair * n_stairs)
take_break = int(total_n_trials/3)
print(f"take_break every {take_break} trials.")
breaks = visual.TextStim(win=win, name='breaks',
                         text="turn on the light and take at least 30-seconds break.\n\n"
                              "When you are ready to continue, press any key.",
                         font='Arial', height=20, colorSpace='rgb', color=[1, 1, 1])

end_of_exp = visual.TextStim(win=win, name='end_of_exp',
                             text="You have completed this experiment.\n"
                                  "Thank you for your time.\n\n"
                                  "Press any key to return to the desktop.",
                             font='Arial', height=20)


# STAIRCASE
expInfo['stair_list'] = list(range(n_stairs))
expInfo['n_trials_per_stair'] = n_trials_per_stair

stairStart = maxLum
miniVal = bgLum
maxiVal = maxLum

print('\nexpInfo (dict)')
for k, v in expInfo.items():
    print(f"{k}: {v}")


stairs = []
for stair_idx in expInfo['stair_list']:
    thisInfo = copy.copy(expInfo)
    thisInfo['stair_idx'] = stair_idx

    # stair_name will be pos or neg sep value for congruence (e.g., 18, -18, 6, -6 etc)
    # however, change -0 to -.1 to avoid confusion with 0.
    stair_name = separations[stair_idx] * congruence_list[stair_idx]
    if separations[stair_idx] + congruence_list[stair_idx] == -1:
        stair_name = -.1

    thisStair = Staircase(name=f'{stair_name}',
                          type='simple',
                          value=stairStart,
                          C=stairStart * 0.6,  # step_size, typically 60% of reference stimulus
                          minRevs=3,
                          minTrials=n_trials_per_stair,
                          minVal=miniVal,
                          maxVal=maxiVal,
                          targetThresh=0.75,
                          extraInfo=thisInfo
                          )
    stairs.append(thisStair)


# EXPERIMENT
trial_number = 0
print('\n*** exp loop*** \n\n')

for step in range(n_trials_per_stair):
    np.random.shuffle(stairs)
    for thisStair in stairs:

        print(f"thisStair: {thisStair}, step: {step}")

        trial_number = trial_number + 1
        trialClock.reset()

        stair_idx = thisStair.extraInfo['stair_idx']
        sep = separations[stair_idx]
        # congruence is balanced with separation values
        congruent = congruence_list[stair_idx]
        # flow_dir = flow_directions[stair_idx]
        flow_dir = np.random.choice([1, -1])

        # PROBE
        # target_jump = np.random.choice([1, -1])  # direction in which the probe jumps : CW or CCW
        target_jump = congruent * flow_dir

        # flow_dots
        x = np.random.rand(nDots) * taille - taille / 2
        y = np.random.rand(nDots) * taille - taille / 2
        z = np.random.rand(nDots) * (maxDist - minDist) + minDist
        # z was called z_flow but is actually z position like x and y
        x_flow = x / z
        y_flow = y / z


        # Make variable for whether target_jump and flow dir are the same
        # (e.g., both inward or both outward = 1, else -1)
        trgt_flow_same = flow_dir*target_jump

        # staircase varies probeLum
        probeLum = thisStair.next()
        probeColor255 = probeLum * LumColor255Factor
        probeColor1 = (probeColor255 * Color255Color1Factor) - 1
        probe1.color = [probeColor1, probeColor1, probeColor1]
        probe2.color = [probeColor1, probeColor1, probeColor1]

        # PROBE LOCATIONS
        # corners go CCW(!) 45=top-right, 135=top-left, 225=bottom-left, 315=bottom-right
        corner = np.random.choice([45, 135, 225, 315])

        print(f'\tcorner: {corner}, flow_dir: {flow_dir}, target_jump: {target_jump}')
        # dist_from_fix is a constant giving distance form fixation,
        # dist_from_fix was previously 2 identical variables x_prob & y_prob.
        dist_from_fix = round((tan(np.deg2rad(probe_ecc)) * viewdistPix) / sqrt(2))
        # x_prob = y_prob = round((tan(np.deg2rad(probe_ecc)) * viewdistPix) / sqrt(2))

        print(f'probeLum: {probeLum}, probeColor255: {probeColor255}, probeColor1: {probeColor1}')
        print(f'\tbgLum: {bgLum}, bgColor255: {bgColor255}, bgcolor: {bgcolor}')
        print(f'\t\twin.colorSpace: {win.colorSpace}\n')
        # actual_bg_color = bgcolor[0]
        # print(f'bgcolor_to_rgb255: {bgcolor_to_rgb255}')
        # bgcolor_to_rgb255 = (actual_bg_color+1)*127.5
        # print(f'bgcolor_to_rgb255: {bgcolor_to_rgb255}')
        # check_bgcol_rgb = (bgcolor_to_rgb255 * Color255Color1Factor) - 1
        # print(f'check_bgcol_rgb: {check_bgcol_rgb}')
        # check_bg_back_to_255 = check_bgcol_rgb * LumColor255Factor
        # print(f'check_bg_back_to_255: {check_bg_back_to_255}')





        # probe mask locations
        probeMask1.setPos([dist_from_fix+1, dist_from_fix+1])
        probeMask2.setPos([-dist_from_fix-1, dist_from_fix+1])
        probeMask3.setPos([-dist_from_fix-1, -dist_from_fix-1])
        probeMask4.setPos([dist_from_fix+1, -dist_from_fix-1])

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


        # timing in frames
        # fixation time is now 70ms shorted than previously.
        t_fixation = 1 * (fps - prelim_bg_flow_fr)
        t_bg_motion = t_fixation + prelim_bg_flow_fr
        t_interval_1 = t_bg_motion + probe_duration
        t_ISI = t_interval_1 + ISI
        t_interval_2 = t_ISI + probe_duration
        # essentially unlimited time to respond
        t_response = t_interval_2 + 10000 * fps

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

                # FIXATION
                if t_fixation >= frameN > 0:
                    # before fixation has finished
                    trials_counter.text = f"{trial_number}/{total_n_trials}"

                    if background == 'flow_rad':
                        # draw flow_dots but with no motion
                        flow_dots.xys = np.array([x_flow, y_flow]).transpose()
                        flow_dots.draw()
                        probeMask1.draw()
                        probeMask2.draw()
                        probeMask3.draw()
                        probeMask4.draw()
                        dotsMask.draw()

                    fixation.setRadius(3)
                    fixation.draw()
                    trials_counter.draw()

                # Background motion prior to probe1
                if t_bg_motion >= frameN > t_fixation:
                    # after fixation, before end of background motion
                    if background == 'flow_rad':
                        # radial flow_dots motion
                        z = z + flow_speed * flow_dir
                        WrapPoints(z, minDist, maxDist)
                        x_flow = x / z
                        y_flow = y / z

                        flow_dots.xys = np.array([x_flow, y_flow]).transpose()
                        flow_dots.draw()

                        probeMask1.draw()
                        probeMask2.draw()
                        probeMask3.draw()
                        probeMask4.draw()
                        dotsMask.draw()

                    fixation.setRadius(3)
                    fixation.draw()
                    # probe1.draw()
                    trials_counter.draw()


                # PROBE 1
                if t_interval_1 >= frameN > t_bg_motion:
                    # after background motion, before end of probe1 interval
                    if background == 'flow_rad':
                        # radial flow_dots motion
                        z = z + flow_speed * flow_dir
                        WrapPoints(z, minDist, maxDist)
                        x_flow = x / z
                        y_flow = y / z

                        flow_dots.xys = np.array([x_flow, y_flow]).transpose()
                        flow_dots.draw()

                        probeMask1.draw()
                        probeMask2.draw()
                        probeMask3.draw()
                        probeMask4.draw()
                        dotsMask.draw()

                    fixation.setRadius(3)
                    fixation.draw()
                    probe1.draw()
                    trials_counter.draw()


                # ISI
                if t_ISI >= frameN > t_interval_1:
                    if background == 'flow_rad':
                        # radial flow_dots motion
                        z = z + flow_speed * flow_dir
                        WrapPoints(z, minDist, maxDist)
                        x_flow = x / z
                        y_flow = y / z

                        flow_dots.xys = np.array([x_flow, y_flow]).transpose()
                        flow_dots.draw()

                        probeMask1.draw()
                        probeMask2.draw()
                        probeMask3.draw()
                        probeMask4.draw()
                        dotsMask.draw()

                    fixation.setRadius(3)
                    fixation.draw()
                    trials_counter.draw()

                # PROBE 2
                if t_interval_2 >= frameN > t_ISI:
                    # after ISI but before end of probe2 interval
                    if background == 'flow_rad':
                        # radial flow_dots motion
                        z = z + flow_speed * flow_dir
                        WrapPoints(z, minDist, maxDist)
                        x_flow = x / z
                        y_flow = y / z

                        flow_dots.xys = np.array([x_flow, y_flow]).transpose()
                        flow_dots.draw()

                        probeMask1.draw()
                        probeMask2.draw()
                        probeMask3.draw()
                        probeMask4.draw()
                        dotsMask.draw()

                    fixation.setRadius(3)
                    fixation.draw()
                    probe2.draw()
                    trials_counter.draw()

                # ANSWER
                if frameN > t_interval_2:
                    # after probe 2 interval
                    if background == 'flow_rad':
                        # draw flow_dots but with no motion
                        flow_dots.draw()
                        probeMask1.draw()
                        probeMask2.draw()
                        probeMask3.draw()
                        probeMask4.draw()
                        dotsMask.draw()

                    fixation.setRadius(2)
                    fixation.draw()
                    trials_counter.draw()


                    # ANSWER
                    # todo: check if there is a better way to get key responses and RTs
                    resp = event.BuilderKeyResponse()
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

                # regardless of frameN
                # check for quit
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
        thisExp.addData('congruent', congruent)
        thisExp.addData('flow_dir', flow_dir)
        thisExp.addData('probe_jump', target_jump)
        thisExp.addData('corner', corner)
        thisExp.addData('probeLum', probeLum)
        thisExp.addData('trial_response', resp.corr)
        thisExp.addData('resp.rt', resp.rt)
        thisExp.addData('probeColor1', probeColor1)
        thisExp.addData('probeColor255', probeColor255)
        thisExp.addData('probe_ecc', probe_ecc)
        thisExp.addData('BGspeed', flow_speed)
        thisExp.addData('orientation', orientation)
        thisExp.addData('ISI_actual_ms', ISI_actual_ms)
        thisExp.addData('ISI_frames', ISI_frames)
        thisExp.addData('actual_bg_color', actual_bg_color)
        thisExp.addData('bgcolor_to_rgb1', bgcolor_to_rgb1)
        thisExp.addData('bgLumP', bgLumP)
        thisExp.addData('bgLum', bgLum)
        thisExp.addData('bgColor255', bgColor255)
        thisExp.addData('weber_thr', (probeLum-bgLum)/probeLum)
        thisExp.addData('expName', expName)

        thisExp.nextEntry()

        thisStair.newValue(resp.corr)  # so that the staircase adjusts itself

print("end of exp loop, saving data")
thisExp.close()

while not event.getKeys():
    # display end of experiment screen
    end_of_exp.draw()
    win.flip()
else:
    # close and quit once a key is pressed
    win.close()
    core.quit()
