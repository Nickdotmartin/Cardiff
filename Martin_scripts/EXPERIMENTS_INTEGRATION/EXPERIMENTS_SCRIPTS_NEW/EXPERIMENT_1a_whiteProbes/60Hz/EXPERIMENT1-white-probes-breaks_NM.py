from __future__ import division
from psychopy import sound, gui, visual, core, data, event, logging, clock, monitors
# from psychopy.visual import ShapeStim, EnvelopeGrating, Circle
from psychopy.visual.shape import ShapeStim
from psychopy.visual.secondorder import EnvelopeGrating
from psychopy.visual.circle import Circle

import psychopy
import os
import numpy
from numpy import (arcsin, arccos, arctan, sin, cos, tan, pi, average, sqrt, std, deg2rad, rad2deg)
from numpy.random import shuffle
import random
import copy
import time
from datetime import datetime
from math import *
from scipy.optimize import fsolve

from kestenSTmaxVal import Staircase

logging.console.setLevel(logging.CRITICAL)

# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)


# Store info about the experiment session
# psychopyVersion = psychopy.__version__
expName = 'integration-EXP1'  # from the Builder filename that created this script

# todo: change dict keys once I've looked at analysis
expInfo = {'1. Participant': 'test',
           '2. Probe duration in frames at 240hz': '2',
           # '3. fps':['60'],
           '3. fps': 60,
           '4. ISI duration in frame': ['0', '2', '4', '6', '9', '12', '24', '-1'],
           '5. Probe orientation': ['tangent'],
           '6. Probe size': ['5pixels', '6pixels', '3pixels'],
           '7. Background lum in percent of maxLum': '20',
           '8. Red filter': ['no', 'yes']}


## GUI
dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
if not dlg.OK:
    core.quit()  # user pressed cancel

expInfo['time'] = datetime.now().strftime("%H:%M:%S")
expInfo['date'] = datetime.now().strftime("%d/%m/%Y")

# GUI SETTINGS
participant_name = expInfo['1. Participant']
trial_number = 25
probe_duration = int(expInfo['2. Probe duration in frames at 240hz'])
probe_ecc = 4  # int((expInfo['6. Probe eccentricity in deg']))
# fps = float(expInfo['3. fps'])
fps = expInfo['3. fps']
orientation = expInfo['5. Probe orientation']


# VARIABLES
# Distances between probes
# 99 values for single probe condition
separations = [18, 18, 6, 6, 3, 3, 2, 2, 1, 1, 0, 0, 99, 99]

# ISI durations, -1 correspond to simultaneous probes
ISI = int(expInfo['4. ISI duration in frame'])

# FILENAME
# todo: look at filenames - are there any sq brackets?  
#  also remove redundant brackets round participant name
filename = (_thisDir + os.sep + '%s' % (participant_name) +
            os.sep + ('ISI_' + expInfo['4. ISI duration in frame'] + 
            '_probeDur' + expInfo['2. Probe duration in frames at 240hz']) +
            os.sep + participant_name)
# filename = f'{_thisDir}{os.sep}'

# Experiment Handler
thisExp = data.ExperimentHandler(name=expName, version='',
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
maxColor255 = 255
minColor255 = 0
maxColor1 = 1
minColor1 = -1
# todo: check these double parenthesis
bgLumP = int((expInfo['7. Background lum in percent of maxLum']))
bgLum = maxLum * bgLumP / 100
bgColor255 = bgLum * LumColor255Factor

# todo: figure out how to access win size property
# print('get mon size')
# win_size = visual.Window.size()
# print(win_size)
# print(type(win_size))
# this_window = visual.Window
# window_size = this_window.size
# print(window_size)
# screenWidth=visual.Window.size[0]
# screenHeight=visual.Window.size[1]
# print(f'width: {screenWidth}, height: {screenHeight}')


# MONITOR SPEC
widthPix = 1440  # 1280
heightPix = 900  # 800
monitorwidth = 30.41  # 32.512  # monitor width in cm
viewdist = 57.3  # viewing distance in cm
viewdistPix = widthPix/monitorwidth*viewdist
monitorname = 'NickMac'  # 'asus_cal'  # gamma set at 2.1
mon = monitors.Monitor(monitorname, width=monitorwidth, distance=viewdist)
# todo: check these double parenthesis
mon.setSizePix((widthPix, heightPix))
mon.save()

# WINDOW SPEC
win = visual.Window(monitor=mon, size=(widthPix, heightPix),
                    colorSpace='rgb255', color=bgColor255,
                    # I've added winType to make it work on my mac
                    winType='pyglet',

                    # pos gives position of top-left of screen
                    pos=[1, -1],
                    units='pix',
                    screen=1,
                    allowGUI=False,
                    fullscr=None
                    )

# ELEMENTS
# fixation bull eye
# todo: why don't I have Circle, `envelopeGrating or ShapeSTIM?
fixation = visual.Circle(win, radius=2, units='pix',
                         lineColor='white', fillColor='black')

# PROBEs
# probe color
if expInfo['8. Red filter'] == 'yes':
    redfilter = -1
else:
    redfilter = 1
# probre sizes choice
# todo: probvert can be undefined - e.g, iuf statements don't capture all possibilities.
if expInfo['6. Probe size'] == '6pixels':
    probeVert = [(0, 0), (1, 0), (1, 1), (2, 1), (2, -2),
                 (-1, -2), (-1, -1), (0, -1)]  # 6 pixels
elif expInfo['6. Probe size'] == '5pixels':
    probeVert = [(0, 0), (1, 0), (1, 1), (2, 1), (2, -1), (1, -1),
                 (1, -2), (-1, -2), (-1, -1), (0, -1)]  # 5 pixels
elif expInfo['6. Probe size'] == '3pixels':
    probeVert = [(0, 0), (1, 0), (1, 1), (2, 1), (2, 0), (1, 0), (1, -1),
                 (0, -1), (0, -2), (-1, -2), (-1, -2), (-1, -1), (0, -1)]  # 3 pixels

probe1 = visual.ShapeStim(win, vertices=probeVert, fillColor=(1.0, -1.0, 1.0),
                          lineWidth=0, opacity=1, size=1, interpolate=False)
probe2 = visual.ShapeStim(win, vertices=probeVert, fillColor=[-1.0, 1.0, -1.0],
                          lineWidth=0, opacity=1, size=1, interpolate=False)

# MOUSE
myMouse = event.Mouse(visible=False)

# ------------------------------------------------------------------- INSTRUCTION
# ------------------------------------------------------------------- INSTRUCTION
# todo: add key for previous trial.
instructions = visual.TextStim(win=win, name='instructions',
                               text="[q] or [4] top-left\n "
                                    "[w] or [5] top-right\n "
                                    "[a] or [1] bottom-left\n "
                                    "[s] or [2] bottom-right \n\n "
                                    "redo the previous trial \n\n"
                                    "[Space bar] to start",
                               font='Arial', pos=[0, 0], height=20, ori=0, color=[255, 255, 255],
                               colorSpace='rgb255', opacity=1, languageStyle='LTR', depth=0.0)

# ------------------------------------------------------------------- BREAKS
breaks = visual.TextStim(win=win, name='breaks',
                         text="turn on the light and  take at least 30-seconds break.",
                         font='Arial', pos=[0, 0], height=20, ori=0, color=[255, 255, 255],
                         colorSpace='rgb255', opacity=1, languageStyle='LTR', depth=0.0)

while not event.getKeys():
    instructions.draw()
    win.flip()
# ------------------------------------------------------------------- STAIRCASE
# ------------------------------------------------------------------- STAIRCASE
total_nTrials = 0
expInfo['startPoints'] = list(range(1, 15))  # 14 staircases (14 conditions)
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
                          C=stairStart*0.6,  # typically 60% of reference stimulus
                          minRevs=3,
                          minTrials=trial_number,
                          minVal=miniVal,
                          maxVal=maxiVal,
                          targetThresh=0.75,  # changed this from prev versions
                          extraInfo=thisInfo)
    stairs.append(thisStair)

# ------------------------------------------------------------------- EXPERIMENT
# ------------------------------------------------------------------- EXPERIMENT
for trialN in range(expInfo['nTrials']):
    shuffle(stairs)
    for thisStair in stairs:

        # conditions
        # separation experiment #################################################
        sep = separations[thisStair.extraInfo['thisStart']-1]
        # direction in which the probe jumps : CW or CCW
        target_jump = random.choice([1, -1])
        stairNum = thisStair.extraInfo['thisStart']
        probeLum = thisStair.next()
        probeColor255 = probeLum * LumColor255Factor
        probeColor1 = (probeColor255 * Color255Color1Factor) - 1

        total_nTrials = total_nTrials + 1

        # Black or White
        probe1.color = [probeColor1, probeColor1*redfilter, probeColor1*redfilter]
        probe2.color = [probeColor1, probeColor1*redfilter, probeColor1*redfilter]

        # PROBE LOCATION
        corner = random.choice([45, 135, 225, 315])
        # todo: check these double parenthesis
        x_prob = round((tan(deg2rad(probe_ecc))*viewdistPix)/sqrt(2))
        y_prob = round((tan(deg2rad(probe_ecc))*viewdistPix)/sqrt(2))

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
                    # todo: are these brackets needed?
                    probe2.pos = [p1_x - (sep)+1, p1_y + (sep)]
                elif target_jump == -1:  # CW
                    probe1.ori = 180
                    probe2.ori = 0
                    probe2.pos = [p1_x + (sep)-1, p1_y - (sep)]
                elif target_jump == 9:
                    probe1.ori = random.choice([0, 180])
        elif corner == 135:
            p1_x = x_prob * -1
            p1_y = y_prob * 1
            if orientation == 'tangent':
                if target_jump == 1:  # CCW
                    probe1.ori = 90
                    probe2.ori = 270
                    probe2.pos = [p1_x + (sep)-1, p1_y + (sep)]
                elif target_jump == -1:  # CW
                    probe1.ori = 270
                    probe2.ori = 90
                    probe2.pos = [p1_x - (sep)+1, p1_y - (sep)]
                elif target_jump == 9:
                    probe1.ori = random.choice([90, 270])
        elif corner == 225:
            p1_x = x_prob * -1
            p1_y = y_prob * -1
            if orientation == 'tangent':
                if target_jump == 1:  # CCW
                    probe1.ori = 180
                    probe2.ori = 0
                    probe2.pos = [p1_x + (sep)-1, p1_y - (sep)]
                elif target_jump == -1:  # CW
                    probe1.ori = 0
                    probe2.ori = 180
                    probe2.pos = [p1_x - (sep)+1, p1_y + (sep)]
                elif target_jump == 9:
                    probe1.ori = random.choice([0, 180])
        elif corner == 315:
            p1_x = x_prob * 1
            p1_y = y_prob * -1
            if orientation == 'tangent':
                if target_jump == 1:  # CCW
                    probe1.ori = 270
                    probe2.ori = 90
                    probe2.pos = [p1_x - (sep)+1, p1_y - (sep)]
                elif target_jump == -1:  # CW
                    probe1.ori = 90
                    probe2.ori = 270
                    probe2.pos = [p1_x + (sep)-1, p1_y + (sep)]
                elif target_jump == 9:
                    probe1.ori = random.choice([90, 270])

        probe1.pos = [p1_x, p1_y]

        # timing in frames
        # if ISI >= 0:
        t_fixation = 1 * fps
        t_interval_1 = t_fixation + probe_duration
        t_ISI = t_interval_1 + ISI
        t_interval_2 = t_ISI + probe_duration
        # I presume this means almost unlimited time to respond?
        t_response = t_interval_2 + 10000*fps


        # repeat the trial if [r] has been pressed
        repeat = True
        while repeat:
            frameN = -1
            # display Break before trials 120 and 240
            if total_nTrials == 120+1 or total_nTrials == 240+1:
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
                    fixation.setRadius(3)
                    fixation.draw()

                # PROBE 1
                if t_interval_1 >= frameN > t_fixation:
                    probe1.draw()
                    # SIMULTANEOUS CONDITION
                    if ISI == -1:
                        if sep <= 18:
                            probe2.draw()
                    fixation.setRadius(3)
                    fixation.draw()

                # ISI
                if t_ISI >= frameN > t_interval_1:
                    fixation.setRadius(3)
                    fixation.draw()

                # PROBE 2
                if t_interval_2 >= frameN > t_ISI:
                    if ISI >= 0:
                        if sep <= 18:
                            probe2.draw()
                    fixation.setRadius(3)
                    fixation.draw()

                # ANSWER
                if frameN > t_interval_2:
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

                # check for quit
                if event.getKeys(keyList=["escape"]):
                    core.quit()
                # redo the trial if i think i made a mistake
                if event.getKeys(keyList=["r"]) or event.getKeys(keyList=['num_9']):
                    repeat = True
                    continueRoutine = False
                    continue

                # refresh the screen
                if continueRoutine:
                    win.flip()

        thisExp.addData('stair', stairNum)
        thisExp.addData('probe_jump', target_jump)
        thisExp.addData('probeColor1', probeColor1)
        thisExp.addData('probeColor255', probeColor255)
        thisExp.addData('probeLum', probeLum)
        thisExp.addData('trial_response', resp.corr)
        thisExp.addData('corner', corner)
        thisExp.addData('probe_ecc', probe_ecc)
        thisExp.addData('resp.rt', resp.rt)
        thisExp.addData('total_nTrials', total_nTrials)
        thisExp.addData('orientation', orientation)
        thisExp.nextEntry()

        thisStair.newValue(resp.corr)   # so that the staircase adjusts itself
