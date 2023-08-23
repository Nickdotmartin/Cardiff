from __future__ import division
from psychopy import sound, gui, visual, core, data, event, logging, clock, monitors
# from psychopy.visual import ShapeStim, EnvelopeGrating, Circle

import os
import numpy as np
from numpy import (arcsin, arccos, arctan, sin, cos, tan, pi, average, sqrt, std, deg2rad, rad2deg)
from numpy.random import shuffle
import random
import copy
import time
from datetime import datetime
from math import *
from scipy.optimize import fsolve

from kestenSTmaxVal import Staircase

"""
This script is adapted from EXPERIMENT3-backgroundMotion.py, 
(Martin also has a radial version called integration_RiccoBloch_flow_new.py which is in 
r'C: \ Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\Martin_scripts\EXPERIMENTS_INTEGRATION\EXPERIMENTS_SCRIPTS_NEW\old scripts\integration_RiccoBloch_flow_new.py'
or Downloads.  

I have made several changes.
- change import statements (from psychopy.visual import ShapeStim, EnvelopeGrating, Circle) 
    to just (from psychopy import visual), then use visual.circle, visual.ShapeStim, etc. DONE
- changed import statement from import numpy to import umpy as np, and changed all calls to numpy to np. DONE
- added a method for recording frame intervals.  DONE
- reduced the number of trials per staircase (trial_number = 25) to 2 to speed up testing.  DONE
- reduced the number of sep conds to speed up testing to two, and reduced expInfo['startPoints'] to 2.  DONE
- changed screen number to 0 (from 1) to display on the laptop screen.  DONE
- converted the rotation bg_motion to radial, for better comparison with other rad_flow exps.
    - I've moved martin's variables relating to rotation motion to the top of the script,
        and set them to None if 'radial' is selected.  DONE
    - I'me added the wrap_depth_vals function for radial dots depth.  DONE
    - for simplicity I'll keep motion only during ISI and probe2 (rather than add prelim_bg_motion period).  DONE
- added option for radial probe orientation (but kept tangent) NOT DONE YET
"""

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
    depth_arr[lessthanmin] += depth_adj
    # adjust depth_arr values more than max_depth by subtracting depth_adj
    morethanmax = (depth_arr > max_depth)
    depth_arr[morethanmax] -= depth_adj
    return depth_arr


# logging.console.setLevel(logging.DEBUG)
logging.console.setLevel(logging.CRITICAL)
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)
# Store info about the experiment session
# psychopyVersion = 'v2020.2.10'
expName = 'integration-EXP1'  # from the Builder filename that created this script
expInfo = {'1. Participant': 'Nick_test_02082023',
           '2. Probe duration in frames at 240hz': '2',
           '3. fps': ['60', '240'],
           '4. ISI duration in frame': ['4', '2', '4', '6', '9', '12', '24'],
           '5. Probe orientation': ['tangent'],
           '6. Probe size': ['5pixels', '6pixels', '3pixels'],
           '7. Background lum in pourcentage of maxLum': '20',
           '8. Red filter': ['no', 'yes'],
           '9. bg_motion_dir': ['radial', 'rotation'],
           # '9. Background speed in deg.s-1': '270',
           # '9. Background motion during': ['transient', 'transient&probe2'],
           # '9. Background direction': ['both', 'same', 'opposite']
           }

# GUI
dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
if dlg.OK == False:
    core.quit()  # user pressed cancel

expInfo['time'] = datetime.now().strftime("%H:%M:%S")
expInfo['date'] = datetime.now().strftime("%d/%m/%Y")

# GUI SETTINGS
participant_name = expInfo['1. Participant']
trial_number = 2  # this is the number of trials per stair
probe_duration = int((expInfo['2. Probe duration in frames at 240hz']))
probe_ecc = 4  # int((expInfo['6. Probe eccentricity in deg']))
fps = float(expInfo['3. fps'])
orientation = expInfo['5. Probe orientation']
bg_motion_dir = expInfo['9. bg_motion_dir']

'''This is new, to give the option for rotation or radial bg motion'''
# if bg_motion_dir == 'rotation':
# Background speed in deg.s-1
speed_deg_BG = 270  # int((expInfo['9. Background speed in deg.s-1']))
speed = deg2rad(speed_deg_BG) / fps  # 20 deg/sec (at 240, or .078 at 60fps)
rot_bg_dir = 'both'  # ['both', 'same', 'opposite'], expInfo['9. Background direction']
bg_motion_during = 'transient&probe2'  # ['transient', 'transient&probe2'], expInfo['9. Background motion during'] == 'transient&probe2
#todo: probes_ori = 'tangent'




# VARIABLES
# Distances between probes
# separations = [18, 18, 6, 6, 3, 3, 2, 2, 1, 1, 0, 0]  # 99 values for single probe condition
separations = [18, 18, 6, 6]
# ISI durations, -1 correspond to simultaneous probes
ISI = int((expInfo['4. ISI duration in frame']))

'''I've moved these up to ~line 75, if bg_motion_dir == 'radial' '''
# # Background speed in deg.s-1
# speed_deg_BG = int((expInfo['9. Background speed in deg.s-1']))
# speed = deg2rad(speed_deg_BG) / fps  # 20 deg/sec

# FILENAME
filename = (_thisDir +
            os.sep + '%s' % (participant_name) +
            # os.sep + '%s' % (expInfo['9. Background direction']) +  # I've changed this to bg_motion_dir
            os.sep + '%s' % (bg_motion_dir) +
            # os.sep + '%s' % (str(speed_deg_BG)) +  # don't need this
            os.sep + ('ISI_' + expInfo['4. ISI duration in frame'] + '_probeDur' + expInfo[
            '2. Probe duration in frames at 240hz']) +
            os.sep + participant_name)

# Experiment Handler
thisExp = data.ExperimentHandler(name=expName, version='',
                                 extraInfo=expInfo, runtimeInfo=None,
                                 savePickle=None, saveWideText=True,
                                 dataFileName=filename)

# COLORS AND LUMINANCES
# Lum to Color255
LumColor255Factor = 2.39538706913372
# Color255 to Color1
Color255Color1Factor = 1 / 127.5  # Color255 * Color255Color1Factor -1
# Lum to Color1
Color1LumFactor = 2.39538706913372  ###

maxLum = 106  # 255 RGB
minLum = 0.12  # 0 RGB
maxColor255 = 255
minColor255 = 0
maxColor1 = 1
minColor1 = -1
bgLumP = int((expInfo['7. Background lum in pourcentage of maxLum']))
bgLum = maxLum * bgLumP / 100
bgColor255 = bgLum * LumColor255Factor

# MONITOR SPEC
widthPix = 1920
heightPix = 1080
monitorwidth = 59.77  # monitor width in cm
viewdist = 57.3  # viewing distance in cm
viewdistPix = widthPix / monitorwidth * viewdist
monitorname = 'Nick_work_laptop'  # gamma set at 2.1
mon = monitors.Monitor(monitorname, width=monitorwidth, distance=viewdist)
mon.setSizePix((widthPix, heightPix))
# mon.save()

# WINDOW SPEC
win = visual.Window(monitor=mon, size=(widthPix, heightPix), colorSpace='rgb255', color=bgColor255,
                    units='pix', screen=0, allowGUI=False, fullscr=None)

# ELEMENTS
# fixation bull eye
fixation = visual.Circle(win, radius=2, units='pix', lineColor='white', fillColor='black')

# Dots
nDots = 2000
dots = visual.ElementArrayStim(win, elementTex=None,
                               elementMask='gauss', units='pix', nElements=nDots,
                               sizes=30, colors=[-0.25, -0.25, -0.25])
if bg_motion_dir == 'rotation':
    x = np.random.rand(nDots) * widthPix - widthPix / 2
    y = np.random.rand(nDots) * heightPix - heightPix / 2
    # tranform in polar
    r_dots = (x ** 2 + y ** 2) ** 0.5
    alpha = (np.random.rand(nDots)) * 2 * pi


elif bg_motion_dir == 'radial':
    nDots = 2000  # 10000

    # todo: probes_ori = 'radial'
    dots_speed = 0.2
    # if monitor_name == 'OLED':
    #     dots_speed = 0.4
    BGspeed = dots_speed
    # todo: do we need to increase the number of dots for OLED?
    # dot_array_width = 10000  # original script used 5000
    # with dot_array_width = widthPix * 3, this gives a values of 5760 on a 1920 monitor,
    # similar to the original setting of 5000.  It also allows the dots to be scaled to the screen for OLED.
    dot_array_width = widthPix * 3  # this scales it for the monitor and keeps more dots on screen

    # todo: most of the dots are off screen using this current dots_min_depth, as the distribution of x_flow has large tails.
    #  Setting it to 1.0 means that the tails are shorted, as dividing x / z only makes values smaller (or the same), not bigger.
    # dots_min_depth = 0.5  # depth values
    dots_min_depth = 1.0
    dots_max_depth = 5  # depth values


    # initial array values
    x = np.random.rand(nDots) * dot_array_width - dot_array_width / 2
    y = np.random.rand(nDots) * dot_array_width - dot_array_width / 2
    z = np.random.rand(nDots) * (dots_max_depth - dots_min_depth) + dots_min_depth
    # print(f"x: {x}, y: {y}, z: {z}")

    x_flow = x / z
    y_flow = y / z


else:
    print('bg_motion_dir not recognised, please check your input')
    raise ValueError


# mask for the 4 areas
raisedCosTexture = visual.filters.makeMask(256, shape='raisedCosine', fringeWidth=0.3, radius=[1.0, 1.0])
aperture = 110
probeMask1 = visual.GratingStim(win, mask=raisedCosTexture, tex=None, contrast=1.0, size=(aperture * 2, aperture * 2),
                                units='pix', colorSpace='rgb255', color=bgColor255)
probeMask2 = visual.GratingStim(win, mask=raisedCosTexture, tex=None, contrast=1.0, size=(aperture * 2, aperture * 2),
                                units='pix', colorSpace='rgb255', color=bgColor255)
probeMask3 = visual.GratingStim(win, mask=raisedCosTexture, tex=None, contrast=1.0, size=(aperture * 2, aperture * 2),
                                units='pix', colorSpace='rgb255', color=bgColor255)
probeMask4 = visual.GratingStim(win, mask=raisedCosTexture, tex=None, contrast=1.0, size=(aperture * 2, aperture * 2),
                                units='pix', colorSpace='rgb255', color=bgColor255)

probe_xy = 91
probeMask1.setPos([probe_xy, probe_xy])
probeMask2.setPos([-probe_xy, probe_xy])
probeMask3.setPos([-probe_xy, -probe_xy])
probeMask4.setPos([probe_xy, -probe_xy])

# PROBEs
# probe color
if expInfo['8. Red filter'] == 'yes':
    redfilter = -1
else:
    redfilter = 1
# probre sizes choice
if expInfo['6. Probe size'] == '6pixels':
    probeVert = [(0, 0), (1, 0), (1, 1), (2, 1), (2, -2), (-1, -2), (-1, -1), (0, -1)]  # 6 pixels
elif expInfo['6. Probe size'] == '5pixels':
    probeVert = [(0, 0), (1, 0), (1, 1), (2, 1), (2, -1), (1, -1), (1, -2), (-1, -2), (-1, -1), (0, -1)]  # 5 pixels
elif expInfo['6. Probe size'] == '3pixels':
    probeVert = [(0, 0), (1, 0), (1, 1), (2, 1), (2, 0), (1, 0), (1, -1), (0, -1), (0, -2), (-1, -2), (-1, -2),
                 (-1, -1), (0, -1)]  # 3 pixels

probe1 = visual.ShapeStim(win, vertices=probeVert, fillColor=(1.0, -1.0, 1.0), lineWidth=0, opacity=1, size=1,
                   interpolate=False)  #
probe2 = visual.ShapeStim(win, vertices=probeVert, fillColor=[-1.0, 1.0, -1.0], lineWidth=0, opacity=1, size=1,
                   interpolate=False)  #

# MOUSE
myMouse = event.Mouse(visible=False)


# empty variable to store recorded frame durations
exp_n_fr_recorded_list = [0]
exp_n_dropped_fr = 0
dropped_fr_trial_counter = 0
dropped_fr_trial_x_locs = []
fr_int_per_trial = []
recorded_fr_counter = 0
fr_counter_per_trial = []
cond_list = []

# delete unneeded variables
expected_fr_sec = 1/fps
expected_fr_ms = expected_fr_sec * 1000
frame_tolerance_prop = .2
max_fr_dur_sec = expected_fr_sec + (expected_fr_sec * frame_tolerance_prop)
# max_fr_dur_ms = max_fr_dur_sec * 1000
win.refreshThreshold = max_fr_dur_sec
frame_tolerance_sec = max_fr_dur_sec - expected_fr_sec
frame_tolerance_ms = frame_tolerance_sec * 1000
min_fr_dur_sec = expected_fr_sec - (expected_fr_sec * frame_tolerance_prop)
max_dropped_fr_trials = 10

too_many_dropped_fr = visual.TextStim(win=win, name='too_many_dropped_fr',
                                      text="The experiment had quit as the computer is dropping frames.\n"
                                           "Sorry for the inconvenience.\n"
                                           "Please contact the experimenter.\n\n"
                                           "Press any key to return to the desktop.",
                                      font='Arial', height=20,
                                      # colorSpace=this_colourSpace
                                      )

# ------------------------------------------------------------------- INSTRUCTION
# ------------------------------------------------------------------- INSTRUCTION
instructions = visual.TextStim(win=win, name='instructions',
                               text="[q] or [4] top-left\n [w] or [5] top-right\n [a] or [1] bottom-left\n [s] or [2] bottom-right \n\n redo the previous trial \n\n[Space bar] to start",
                               font='Arial', pos=[0, 0], height=20, ori=0, color=[255, 255, 255],
                               colorSpace='rgb255', opacity=1, languageStyle='LTR', depth=0.0);

while not event.getKeys():
    instructions.draw()
    win.flip()
# ------------------------------------------------------------------- STAIRCASE
# ------------------------------------------------------------------- STAIRCASE
total_nTrials = 0
'''the line below is wrong, originally there were 12 staircases, labelled from 1 to 12 (e.g., stops before 13)'''
# expInfo['startPoints'] = list(range(1, 13))  # 14 stairtcases (14 conditions)
expInfo['startPoints'] = list(range(1, len(separations)+1))
expInfo['nTrials'] = trial_number

stairStart = maxLum
miniVal = bgLum
maxiVal = maxLum

stairs = []
for thisStart in expInfo['startPoints']:
    thisInfo = copy.copy(expInfo)
    thisInfo['thisStart'] = thisStart

    thisStair = Staircase(name='trials',
                          type='simple',
                          value=stairStart,
                          C=stairStart * 0.6,  # typically 60% of reference stimulus
                          minRevs=3,
                          minTrials=trial_number,
                          minVal=miniVal,
                          maxVal=maxiVal,
                          targetThresh=0.75,  # changed this from prev versions
                          extraInfo=thisInfo
                          )
    stairs.append(thisStair)

# ------------------------------------------------------------------- EXPERIMENT
# ------------------------------------------------------------------- EXPERIMENT
for trialN in range(expInfo['nTrials']):
    shuffle(stairs)
    for thisStair in stairs:

        # conditions
        sep = separations[thisStair.extraInfo[
                              'thisStart'] - 1]  # separation experiment #################################################
        target_jump = random.choice([1, -1])  # direction in which the probe jumps : CW or CCW
        stairNum = thisStair.extraInfo['thisStart']
        probeLum = thisStair.next()
        probeColor255 = probeLum * LumColor255Factor
        probeColor1 = (probeColor255 * Color255Color1Factor) - 1

        total_nTrials = total_nTrials + 1

        # Black or White
        probe1.color = [probeColor1 * redfilter, probeColor1 * redfilter, probeColor1 * redfilter]
        probe2.color = [probeColor1 * redfilter, probeColor1 * redfilter, probeColor1 * redfilter]

        # PROBE LOCATION
        corner = random.choice([45, 135, 225, 315])
        x_prob = round((tan(deg2rad(probe_ecc)) * viewdistPix) / sqrt(2))
        y_prob = round((tan(deg2rad(probe_ecc)) * viewdistPix) / sqrt(2))

        # reset probe ori
        probe1.ori = 0
        probe2.ori = 0
        if corner == 45:
            p1_x = x_prob * 1
            p1_y = y_prob * 1
            if orientation == 'tangent':
                if target_jump == 1:  # CCW
                    probe1.ori = 0
                    probe2.ori = 180
                    probe2.pos = [p1_x - (sep) + 1, p1_y + (sep)]
                elif target_jump == -1:  # CW
                    probe1.ori = 180
                    probe2.ori = 0
                    probe2.pos = [p1_x + (sep) - 1, p1_y - (sep)]
                elif target_jump == 9:
                    probe1.ori = random.choice([0, 180])
        elif corner == 135:
            p1_x = x_prob * -1
            p1_y = y_prob * 1
            if orientation == 'tangent':
                if target_jump == 1:  # CCW
                    probe1.ori = 90
                    probe2.ori = 270
                    probe2.pos = [p1_x + (sep) - 1, p1_y + (sep)]
                elif target_jump == -1:  # CW
                    probe1.ori = 270
                    probe2.ori = 90
                    probe2.pos = [p1_x - (sep) + 1, p1_y - (sep)]
                elif target_jump == 9:
                    probe1.ori = random.choice([90, 270])
        elif corner == 225:
            p1_x = x_prob * -1
            p1_y = y_prob * -1
            if orientation == 'tangent':
                if target_jump == 1:  # CCW
                    probe1.ori = 180
                    probe2.ori = 0
                    probe2.pos = [p1_x + (sep) - 1, p1_y - (sep)]
                elif target_jump == -1:  # CW
                    probe1.ori = 0
                    probe2.ori = 180
                    probe2.pos = [p1_x - (sep) + 1, p1_y + (sep)]
                elif target_jump == 9:
                    probe1.ori = random.choice([0, 180])
        elif corner == 315:
            p1_x = x_prob * 1
            p1_y = y_prob * -1
            if orientation == 'tangent':
                if target_jump == 1:  # CCW
                    probe1.ori = 270
                    probe2.ori = 90
                    probe2.pos = [p1_x - (sep) + 1, p1_y - (sep)]
                elif target_jump == -1:  # CW
                    probe1.ori = 90
                    probe2.ori = 270
                    probe2.pos = [p1_x + (sep) - 1, p1_y + (sep)]
                elif target_jump == 9:
                    probe1.ori = random.choice([90, 270])

        probe1.pos = [p1_x, p1_y]

        # speed
        if corner == 45:
            target_jump2 = target_jump
            rotSpeed = speed * target_jump2
        elif corner == 135:
            target_jump2 = target_jump * -1
            rotSpeed = speed * target_jump2
        elif corner == 225:
            target_jump2 = target_jump
            rotSpeed = speed * target_jump2
        elif corner == 315:
            target_jump2 = target_jump * -1
            rotSpeed = speed * target_jump2

        if bg_motion_dir == 'rotation':
            # I've added the var 'rot_bg_dir' instead of using expInfo['9. Background direction']
            if rot_bg_dir == 'both':
                if stairNum % 2 == 1:  # impair staircase BG motion opposite to probe direction
                    rotSpeed = rotSpeed * -1
            elif rot_bg_dir == 'opposite':
                rotSpeed = rotSpeed * -1
            elif rot_bg_dir == 'same':
                rotSpeed = rotSpeed

        else:  # if radial
            # 1 is contracting / inward / backwards, -1 is expanding / outward / forwards
            flow_dir = np.random.choice([1, -1])


            # timimg in frames
        # if ISI >= 0:
        t_fixation = 1 * fps
        t_interval_1 = t_fixation + probe_duration
        t_ISI = t_interval_1 + ISI
        t_interval_2 = t_ISI + probe_duration
        t_response = t_interval_2 + 10000 * fps  # I presume this means almost unlimited time to respond?

        repeat = True

        while repeat:
            frameN = -1
            continueRoutine = True
            while continueRoutine:
                frameN = frameN + 1

                ######################################################################## ISI YES
                # FIXATION
                if t_fixation >= frameN > 0:
                    if bg_motion_dir == 'rotation':
                        new_x = r_dots * np.cos(alpha)
                        new_y = r_dots * np.sin(alpha)
                    else:  # if radial
                        new_x = x_flow
                        new_y = y_flow

                    dots.xys = np.array([new_x, new_y]).transpose()
                    dots.draw()
                    probeMask1.draw()
                    probeMask2.draw()
                    probeMask3.draw()
                    probeMask4.draw()

                    fixation.setRadius(3)
                    fixation.draw()

                # start recording frames
                if frameN == t_fixation:
                    win.recordFrameIntervals = True

                # PROBE 1
                if t_interval_1 >= frameN > t_fixation:
                    dots.draw()
                    probeMask1.draw()
                    probeMask2.draw()
                    probeMask3.draw()
                    probeMask4.draw()
                    probe1.draw()
                    # SIMULTANEOUS CONDITION
                    if ISI == -1:
                        if sep <= 18:
                            probe2.draw()

                    fixation.setRadius(3)
                    fixation.draw()

                # ISI
                if t_ISI >= frameN > t_interval_1:
                    if bg_motion_dir == 'rotation':
                        alpha = alpha + rotSpeed
                        new_x = r_dots * np.cos(alpha)
                        new_y = r_dots * np.sin(alpha)
                    else:  # if radial
                        z = z + dots_speed * flow_dir
                        z = wrap_depth_vals(z, dots_min_depth, dots_max_depth)
                        x_flow = x / z
                        y_flow = y / z
                        new_x, new_y = x_flow, y_flow

                    dots.xys = np.array([new_x, new_y]).transpose()
                    dots.draw()
                    probeMask1.draw()
                    probeMask2.draw()
                    probeMask3.draw()
                    probeMask4.draw()
                    fixation.setRadius(3)
                    fixation.draw()

                    # PROBE 2
                if t_interval_2 >= frameN > t_ISI:
                    if bg_motion_dir == 'rotation':
                        # i'm using the variable bg_motion_during instead of expInfo['9. Background motion during']
                        if bg_motion_during == 'transient&probe2':
                            alpha = alpha + rotSpeed
                        else:
                            alpha = alpha
                        new_x = r_dots * np.cos(alpha)
                        new_y = r_dots * np.sin(alpha)
                    else:  # if radial
                        if bg_motion_during == 'transient&probe2':
                            z = z + dots_speed * flow_dir
                            z = wrap_depth_vals(z, dots_min_depth, dots_max_depth)
                            x_flow = x / z
                            y_flow = y / z
                            new_x, new_y = x_flow, y_flow

                    dots.xys = np.array([new_x, new_y]).transpose()
                    dots.draw()
                    probeMask1.draw()
                    probeMask2.draw()
                    probeMask3.draw()
                    probeMask4.draw()
                    if ISI >= 0:
                        if sep <= 18:
                            probe2.draw()
                    fixation.setRadius(3)
                    fixation.draw()

                # stop recording frame intervals
                if frameN == t_interval_2:
                    win.recordFrameIntervals = False

                # ANSWER
                if frameN > t_interval_2:
                    dots.draw()
                    probeMask1.draw()
                    probeMask2.draw()
                    probeMask3.draw()
                    probeMask4.draw()
                    fixation.setRadius(2)
                    fixation.draw()

                    # ANSWER
                    resp = event.BuilderKeyResponse()
                    theseKeys = event.getKeys(keyList=['num_5', 'num_4', 'num_1', 'num_2', 'w', 'q', 'a', 's'])
                    if len(theseKeys) > 0:  # at least one key was pressed
                        resp.keys = theseKeys[-1]  # just the last key pressed
                        resp.rt = resp.clock.getTime()

                        if corner == 45:
                            if (resp.keys == str('w')) or (resp.keys == 'w') or (resp.keys == 'num_5'):
                                resp.corr = 1
                            else:
                                resp.corr = 0
                            repeat = False
                            continueRoutine = False
                        elif corner == 135:
                            if (resp.keys == str('q')) or (resp.keys == 'q') or (resp.keys == 'num_4'):
                                resp.corr = 1
                            else:
                                resp.corr = 0
                            repeat = False
                            continueRoutine = False
                        elif corner == 225:
                            if (resp.keys == str('a')) or (resp.keys == 'a') or (resp.keys == 'num_1'):
                                resp.corr = 1
                            else:
                                resp.corr = 0
                            repeat = False
                            continueRoutine = False
                        elif corner == 315:
                            if (resp.keys == str('s')) or (resp.keys == 's') or (resp.keys == 'num_2'):
                                resp.corr = 1
                            else:
                                resp.corr = 0
                            repeat = False
                            continueRoutine = False

                        # sort frame interval times
                        # get trial frameIntervals details
                        trial_fr_intervals = win.frameIntervals
                        n_fr_recorded = len(trial_fr_intervals)

                        # add to empty lists etc.
                        fr_int_per_trial.append(trial_fr_intervals)
                        fr_counter_per_trial.append(list(range(recorded_fr_counter,
                                                               recorded_fr_counter + len(trial_fr_intervals))))
                        recorded_fr_counter += len(trial_fr_intervals)
                        exp_n_fr_recorded_list.append(exp_n_fr_recorded_list[-1] + n_fr_recorded)
                        # cond_list.append(thisStair.name)
                        print(f"stairNum: {stairNum}")
                        cond_list.append(stairNum)

                        # check for dropped frames (or frames that are too short)
                        # if timings are bad, repeat trial
                        # if max(trial_fr_intervals) > max_fr_dur_sec or min(trial_fr_intervals) < min_fr_dur_sec:
                        # todo: I've changed this on 13072023 to see if it reduces timing issues.
                        timing_bad = False
                        if max(trial_fr_intervals) > max_fr_dur_sec:
                            logging.warning(
                                f"\n\toh no! Frame too long! {round(max(trial_fr_intervals), 2)} > {round(max_fr_dur_sec, 2)}: "
                                f"trial: {trial_number}, {thisStair.name}")
                            timing_bad = True

                        if min(trial_fr_intervals) < min_fr_dur_sec:
                            logging.warning(
                                f"\n\toh no! Frame too short! {min(trial_fr_intervals)} < {min_fr_dur_sec}, "
                                f": trial: {trial_number}, {thisStair.name}")
                            timing_bad = True

                        if timing_bad:  # comment out stuff for repetitions for now.
                            # repeat = True
                            dropped_fr_trial_counter += 1
                            # trial_number -= 1
                            # thisStair.trialCount = thisStair.trialCount - 1  # so Kesten doesn't count this trial
                            # win.frameIntervals = []
                            # continueRoutine = False
                            trial_x_locs = [exp_n_fr_recorded_list[-2], exp_n_fr_recorded_list[-1]]
                            dropped_fr_trial_x_locs.append(trial_x_locs)
                            # continue

                        # empty frameIntervals cache
                        win.frameIntervals = []

                # check for quit
                if event.getKeys(keyList=["escape"]):
                    core.quit()

                # If too many trials have had dropped frames, quit experiment
                if dropped_fr_trial_counter > max_dropped_fr_trials:
                    while not event.getKeys():
                        # display end of experiment screen
                        too_many_dropped_fr.draw()
                        win.flip()
                    else:
                        # close and quit once a key is pressed
                        thisExp.close()
                        win.close()
                        core.quit()

                # # redo the trial if i think i made a mistake
                # if event.getKeys(keyList=["r"]) or event.getKeys(keyList=['num_9']):
                #     repeat = True
                #     continueRoutine = False
                #     continue

                # refresh the screen
                if continueRoutine:
                    win.flip()

        thisExp.addData('stair', stairNum)
        thisExp.addData('probe_jump', target_jump)
        thisExp.addData('probeColor1', probeColor1)
        thisExp.addData('probeColor255', probeColor255)
        thisExp.addData('probeLum', probeLum)
        thisExp.addData('trial_response', resp.corr)
        thisExp.addData('BGspeed', rotSpeed)
        thisExp.addData('corner', corner)
        thisExp.addData('probe_ecc', probe_ecc)
        thisExp.addData('resp.rt', resp.rt)
        thisExp.addData('total_nTrials', total_nTrials)
        thisExp.addData('orientation', orientation)
        thisExp.nextEntry()

        thisStair.newValue(resp.corr)  # so that the staircase adjusts itself


# plot frame intervals
# change var names and add vaiables
total_n_trials = total_nTrials  # this is the trial counter so the last value should be correct
n_stairs = len(stairs)
stair_names_list = expInfo['startPoints']
print(f"stair_names_list: {stair_names_list}")
monitor_name = monitorname
run_number = 1
# save_dir = filename = (_thisDir +
#             os.sep + '%s' % (participant_name) +
#             os.sep + '%s' % (expInfo['9. Background direction']) +
#             os.sep + '%s' % (str(speed_deg_BG)) +
#             os.sep + ('ISI_' + expInfo['4. ISI duration in frame'] + '_probeDur' + expInfo[
#             '2. Probe duration in frames at 240hz']))

save_dir = os.path.split(filename)[0]

# flatten list of lists (fr_int_per_trial) to get len
all_fr_intervals = [val for sublist in fr_int_per_trial for val in sublist]
total_recorded_fr = len(all_fr_intervals)

print(f"{dropped_fr_trial_counter}/{total_n_trials} trials with bad timing "
      f"(expected: {round(expected_fr_ms, 2)}ms, "
      f"frame_tolerance_ms: +/- {round(frame_tolerance_ms, 2)})")

'''set colours for lines on plot.'''
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from exp1a_psignifit_analysis import fig_colours
my_colours = fig_colours(n_stairs, alternative_colours=False)

# associate colours with conditions
colour_dict = {k: v for (k, v) in zip(stair_names_list, my_colours)}
for k, v in colour_dict.items():
    print(k, v)
print(f"cond_list: {cond_list}")
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

print("\nexperiment finished")

