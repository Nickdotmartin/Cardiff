from __future__ import division
from psychopy import gui, visual, core, data, event, logging, monitors
from psychopy import __version__ as psychopy_version
from numpy.random import shuffle

import os
import numpy as np
import copy
from datetime import datetime
from math import *

from kestenSTmaxVal import Staircase

# sets psychoPy to only log critical messages
# logging.console.setLevel(logging.CRITICAL)
logging.console.setLevel(logging.INFO)


# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# Monitor config from monitor centre
monitor_name = 'HP_24uh'  # 'NickMac' 'asus_cal' 'Asus_VG24' 'HP_24uh'
# gamma set at 2.1  [####### this comment is incorrect, its set above i think ############]
display_number = 1  # 0 indexed, 1 for external display

# Store info about the experiment session
# psychopyVersion = 'v2020.2.10'
# psychopyVersion = psychopy_version
# expName = 'integration-EXP3_bckgrnd_motion_SKR'  # from the Builder filename that created this script
# expInfo = {'1. Participant': 'testnm',
#            '2. Probe duration in frames at 240hz': '2',
#            '3. fps': ['60'],
#            '4. ISI duration in frame': ['9', '2', '3', '4', '5', '6', '9', '12', '24'],
#            '5. Probe orientation': ['tangent'],
#            '6. Probe size': ['5pixels', '6pixels', '3pixels'],
#            '7. Background lum in percent of maxLum': '20',
#            '8. Red filter': ['no', 'yes'],
#            '9. Background speed in deg.s-1': '270',
#            '9. Background motion during': ['transient', 'transient&probe2'],
#            '9. Background direction': ['both', 'same', 'opposite']}
#

Participant = 'testnm'

Background_motion_during = 'transient'
Background_direction = 'both'

# # GUI
# dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
# if not dlg.OK:
#     core.quit()  # user pressed cancel

# expInfo['time'] = datetime.now().strftime("%H:%M:%S")
# expInfo['date'] = datetime.now().strftime("%d/%m/%Y")


# GUI SETTINGS
# participant_name = expInfo['1. Participant']
trial_number = 25
probe_duration = 2  # int(expInfo['2. Probe duration in frames at 240hz'])
probe_ecc = 4  # int((expInfo['6. Probe eccentricity in deg']))
fps = 60  # int(expInfo['3. fps'])
orientation = 'both'  # expInfo['5. Probe orientation']

# VARIABLES
# Distances between probes
# separations = [18, 18, 6, 6, 3, 3, 2, 2, 1, 1, 0, 0]  # 99 values for single probe condition
# ISI durations, -1 correspond to simultaneous probes
ISI = 2  # int(expInfo['4. ISI duration in frame'])

# Background speed in deg.s-1
speed_deg_BG = 270  # int(expInfo['9. Background speed in deg.s-1'])
speed = np.deg2rad(speed_deg_BG) / fps  # 20 deg/sec

# # FILENAME
# filename = f'{_thisDir}{os.sep}' \
#            f'{participant_name}{os.sep}' \
#            f'{expInfo["9. Background direction"]}{os.sep}' \
#            f'{speed_deg_BG}{os.sep}' \
#            f'ISI_{ISI}_probeDur{probe_duration}{os.sep}' \
#            f'{participant_name}'
#
# # Experiment Handler
# thisExp = data.ExperimentHandler(name=expName, version=psychopy_version,
#                                  extraInfo=expInfo, runtimeInfo=None,
#                                  savePickle=None, saveWideText=True,
#                                  dataFileName=filename)

# COLORS AND LUMINANCE
# Lum to Color255
LumColor255Factor = 2.39538706913372
# Color255 to Color1
Color255Color1Factor = 1 / 127.5  # Color255 * Color255Color1Factor -1
# Lum to Color1
Color1LumFactor = 2.39538706913372

maxLum = 106  # 255 RGB
minLum = 0.12  # 0 RGB
maxColor255 = 255
minColor255 = 0
maxColor1 = 1
minColor1 = -1
bgLumP = 20  # int(expInfo['7. Background lum in percent of maxLum'])
bgLum = maxLum * bgLumP / 100
bgColor255 = bgLum * LumColor255Factor

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

widthPix = mon_dict['size'][0]  # 1440  # 1280
heightPix = mon_dict['size'][1]  # 900  # 800
monitorwidth = mon_dict['width']  # 30.41  # 32.512  # monitor width in cm
viewdist = mon_dict['dist']  # 57.3  # viewing distance in cm
# viewdistPix = widthPix / monitorwidth * viewdist
mon = monitors.Monitor(monitor_name, width=monitorwidth, distance=viewdist)
mon.setSizePix((widthPix, heightPix))
mon.save()

# WINDOW SPEC
win = visual.Window(monitor=mon, size=(widthPix, heightPix),
                    colorSpace='rgb255', color=bgColor255,
                    winType='pyglet',  # I've added this to make it work on pycharm/mac
                    pos=[1, -1],  # pos gives position of top-left of screen
                    units='pix',
                    screen=display_number,
                    allowGUI=False,
                    fullscr=None
                    )

# # check correct monitor details (fps, size) have been accessed.
# print(win.monitor.name, win.monitor.getSizePix())
# actualFrameRate = int(win.getActualFrameRate())
# if fps in list(range(actualFrameRate - 2, actualFrameRate + 2)):
#     print("fps matches actual frame rate")
# else:
#     # if values don't match, quit experiment
#     print(f"fps ({fps}) does not match actual frame rate ({actualFrameRate})")
#     core.quit()
#
# actual_size = win.size
# if list(mon_dict['size']) == list(actual_size):
#     print(f"monitor is expected size")
# elif list(mon_dict['size']) == list(actual_size / 2):
#     print(f"actual size is double expected size - Its ok, just a mac retina display bug.")
# else:
#     print(f"Display size does not match expected size from montior centre")
#     # check sizes seems unreliable,
#     # it returns different values for same screen if different mon_names are used!
#     check_sizes = win._checkMatchingSizes(mon_dict['size'], actual_size)
#     print(check_sizes)
#     core.quit()

# ELEMENTS
# # fixation bull eye
# fixation = visual.Circle(win, radius=2, units='pix',
#                          lineColor='white', fillColor='black')

# Dots
nDots = 2000
dots = visual.ElementArrayStim(win, elementTex=None, elementMask='gauss',
                               units='pix', nElements=nDots, sizes=30,
                               colors=[-0.25, -0.25, -0.25], name='dots')

# rather than use heightpix we use widthpix again as the dot field persists and rotates
x = np.random.rand(nDots) * widthPix - widthPix / 2
y = np.random.rand(nDots) * widthPix - widthPix / 2

# transform in polar (** is exponential)
r_dots = np.sqrt(x ** 2 + y ** 2)
# arctan2 returns an Array of angles in radians
alpha = np.arctan2(y, x)

# mask for the 4 areas
raisedCosTexture = visual.filters.makeMask(256, shape='raisedCosine',
                                           fringeWidth=0.3, radius=[1.0, 1.0])
aperture = 110
probeMask1 = visual.GratingStim(win, mask=raisedCosTexture, tex=None, contrast=1.0,
                                size=(aperture * 2, aperture * 2), units='pix',
                                colorSpace='rgb255', color=bgColor255)
probeMask2 = visual.GratingStim(win, mask=raisedCosTexture, tex=None, contrast=1.0,
                                size=(aperture * 2, aperture * 2), units='pix',
                                colorSpace='rgb255', color=bgColor255)
probeMask3 = visual.GratingStim(win, mask=raisedCosTexture, tex=None, contrast=1.0,
                                size=(aperture * 2, aperture * 2), units='pix',
                                colorSpace='rgb255', color=bgColor255)
probeMask4 = visual.GratingStim(win, mask=raisedCosTexture, tex=None, contrast=1.0,
                                size=(aperture * 2, aperture * 2), units='pix',
                                colorSpace='rgb255', color=bgColor255)

probe_xy = 91
probeMask1.setPos([probe_xy, probe_xy])
probeMask2.setPos([-probe_xy, probe_xy])
probeMask3.setPos([-probe_xy, -probe_xy])
probeMask4.setPos([probe_xy, -probe_xy])

# PROBEs
# # probe color
# if expInfo['8. Red filter'] == 'yes':
#     redfilter = -1
# else:
#     redfilter = 1
redfilter = 1


# # probe sizes choice
# if expInfo['6. Probe size'] == '6pixels':  # 6 pixels
#     probeVert = [(0, 0), (1, 0), (1, 1), (2, 1),
#                  (2, -2), (-1, -2), (-1, -1), (0, -1)]
# elif expInfo['6. Probe size'] == '3pixels':  # 3 pixels
#     probeVert = [(0, 0), (1, 0), (1, 1), (2, 1), (2, 0), (1, 0), (1, -1),
#                  (0, -1), (0, -2), (-1, -2), (-1, -2), (-1, -1), (0, -1)]
# else:  # 5 pixels
#     # default setting is expInfo['6. Probe size'] == '5pixels':
#     expInfo['6. Probe size'] = '5pixels'
#     probeVert = [(0, 0), (1, 0), (1, 1), (2, 1), (2, -1), (1, -1),
#                  (1, -2), (-1, -2), (-1, -1), (0, -1)]

# probeVert = [(0, 0), (1, 0), (1, 1), (2, 1), (2, -1), (1, -1),
#              (1, -2), (-1, -2), (-1, -1), (0, -1)]
#
# probe1 = visual.ShapeStim(win, vertices=probeVert, fillColor=(1.0, -1.0, 1.0),
#                           lineWidth=0, opacity=1, size=1, interpolate=False)
# probe2 = visual.ShapeStim(win, vertices=probeVert, fillColor=[-1.0, 1.0, -1.0],
#                           lineWidth=0, opacity=1, size=1, interpolate=False)

# Mouse - Hide cursor
myMouse = event.Mouse(visible=False)

# # INSTRUCTION
# instructions = visual.TextStim(win=win, name='instructions',
#                                text="[q] or [4] top-left\n "
#                                     "[w] or [5] top-right\n "
#                                     "[a] or [1] bottom-left\n "
#                                     "[s] or [2] bottom-right \n\n "
#                                     "[r] or [9] to redo the previous trial \n\n"
#                                     "[Space bar] to start",
#                                font='Arial', pos=[0, 0], height=20, ori=0,
#                                color=[255, 255, 255], colorSpace='rgb255',
#                                opacity=1, languageStyle='LTR', depth=0.0)
#
#
# while not event.getKeys():
#     instructions.draw()
#     win.flip()

# STAIRCASE
total_nTrials = 0
# Martin's original script had range(1, 13) - which corresponds to 12 separation values,
# Exp1a has 14 separation values
# so does this mean that there are only 12 staircase conditions?
# expInfo['startPoints'] = list(range(1, 13))  # 14 staircases (14 conditions)
# expInfo['nTrials'] = trial_number

startPoints = list(range(1, 13))  # 14 staircases (14 conditions)
nTrials = trial_number

stairStart = maxLum
miniVal = bgLum
maxiVal = maxLum

# print('\nexpInfo (dict)')
# for k, v in expInfo.items():
#     print(f"{k}: {v}")

# stairs = []
# # for thisStart in expInfo['startPoints']:
# for thisStart in startPoints:
#
#     # thisInfo = copy.copy(expInfo)
#     # thisInfo['thisStart'] = thisStart
#
#     thisStair = Staircase(name='trials',
#                           type='simple',
#                           value=stairStart,
#                           C=stairStart * 0.6,  # typically 60% of reference stimulus
#                           minRevs=3,
#                           minTrials=trial_number,
#                           minVal=miniVal,
#                           maxVal=maxiVal,
#                           targetThresh=0.75,  # changed this from prev versions
#                           # extraInfo=thisInfo
#                           )
#     stairs.append(thisStair)

# EXPERIMENT
# for trialN in range(expInfo['nTrials']):
for trialN in range(nTrials):

    # np.random.shuffle(stairs)
    # for thisStair in stairs:
    for thisStart in startPoints:

        # conditions
        # separation experiment #################################################
        # sep = separations[thisStair.extraInfo['thisStart'] - 1]
        sep = 18

        target_jump = np.random.choice([1, -1])  # direction in which the probe jumps : CW or CCW
        # stairNum = thisStair.extraInfo['thisStart']
        stairNum = 1
        # probeLum = thisStair.next()
        # probeLum = 106
        # probeColor255 = probeLum * LumColor255Factor
        # probeColor1 = (probeColor255 * Color255Color1Factor) - 1

        total_nTrials = total_nTrials + 1

        # Black or White
        # # I don't understand this, why do you want the opp polarity probe if there is a red filter?
        # probe1.color = [probeColor1 * redfilter, probeColor1 * redfilter, probeColor1 * redfilter]
        # probe2.color = [probeColor1 * redfilter, probeColor1 * redfilter, probeColor1 * redfilter]

        # PROBE LOCATION
        corner = np.random.choice([45, 135, 225, 315])
        # x_prob = round((tan(np.deg2rad(probe_ecc)) * viewdistPix) / sqrt(2))
        # y_prob = round((tan(np.deg2rad(probe_ecc)) * viewdistPix) / sqrt(2))

        # reset probe ori
        # probe1.ori = 0
        # probe2.ori = 0
        # if corner == 45:
        #     p1_x = x_prob * 1
        #     p1_y = y_prob * 1
        #     if orientation == 'tangent':
        #         if target_jump == 1:  # CCW
        #             probe1.ori = 0
        #             probe2.ori = 180
        #             probe2.pos = [p1_x - sep + 1, p1_y + sep]
        #         elif target_jump == -1:  # CW
        #             probe1.ori = 180
        #             probe2.ori = 0
        #             probe2.pos = [p1_x + sep - 1, p1_y - sep]
        #         #  # target jump can only be -1 or 1 (see line 205).
        #         # elif target_jump == 9:
        #         #     probe1.ori = np.random.choice([0, 180])
        # elif corner == 135:
        #     p1_x = x_prob * -1
        #     p1_y = y_prob * 1
        #     if orientation == 'tangent':
        #         if target_jump == 1:  # CCW
        #             probe1.ori = 90
        #             probe2.ori = 270
        #             probe2.pos = [p1_x + sep - 1, p1_y + sep]
        #         elif target_jump == -1:  # CW
        #             probe1.ori = 270
        #             probe2.ori = 90
        #             probe2.pos = [p1_x - sep + 1, p1_y - sep]
        # #                elif target_jump == 9:
        # #                    probe1.ori = np.random.choice([90, 270])
        # elif corner == 225:
        #     p1_x = x_prob * -1
        #     p1_y = y_prob * -1
        #     if orientation == 'tangent':
        #         if target_jump == 1:  # CCW
        #             probe1.ori = 180
        #             probe2.ori = 0
        #             probe2.pos = [p1_x + sep - 1, p1_y - sep]
        #         elif target_jump == -1:  # CW
        #             probe1.ori = 0
        #             probe2.ori = 180
        #             probe2.pos = [p1_x - sep + 1, p1_y + sep]
        # #                elif target_jump == 9:
        # #                    probe1.ori = np.random.choice([0, 180])
        # else:
        #     corner = 315
        #     p1_x = x_prob * 1
        #     p1_y = y_prob * -1
        #     if orientation == 'tangent':
        #         if target_jump == 1:  # CCW
        #             probe1.ori = 270
        #             probe2.ori = 90
        #             probe2.pos = [p1_x - sep + 1, p1_y - sep]
        #         elif target_jump == -1:  # CW
        #             probe1.ori = 90
        #             probe2.ori = 270
        #             probe2.pos = [p1_x + sep - 1, p1_y + sep]
        # #                elif target_jump == 9:
        # #                    probe1.ori = np.random.choice([90, 270])
        #
        # probe1.pos = [p1_x, p1_y]

        # speed
        # this looks like target 2 appears CW for 45 or 255 and CCW for 135 and 315
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

        # if expInfo['9. Background direction'] == 'both':
        if Background_direction == 'both':
            # rotation is CCW for odd stairNum and CW for even
            if stairNum % 2 == 1:  # impair staircase BG motion opposite to probe direction
                rotSpeed = rotSpeed * -1
        # elif expInfo['9. Background direction'] == 'opposite':
        elif Background_direction == 'opposite':
            rotSpeed = rotSpeed * -1
        # elif expInfo['9. Background direction'] == 'same':
        elif Background_direction == 'same':
            rotSpeed = rotSpeed
        else:
            raise ValueError(f"expInfo['9. Background direction'] should be "
                             f"('opposite', 'same', or 'both'), not "
                             # f"{expInfo['9. Background direction']}"
                             f"{Background_direction}")

        # timing in frames
        # if ISI >= 0:
        t_fixation = 1 * fps
        t_interval_1 = t_fixation + probe_duration
        t_ISI = t_interval_1 + ISI
        t_interval_2 = t_ISI + probe_duration
        # I presume this means almost unlimited time to respond?
        t_response = t_interval_2 + 10000 * fps

        # repeat the trial if [r] has been pressed
        repeat = True
        while repeat:
            frameN = -1
            continueRoutine = True
            while continueRoutine:
                frameN = frameN + 1

                # ISI YES
                # FIXATION
                if t_fixation >= frameN > 0:
                    # new position for dots at fixation time
                    new_x = r_dots * np.cos(alpha)
                    new_y = r_dots * np.sin(alpha)
                    dots.xys = np.array([new_x, new_y]).transpose()
                    dots.draw()
                    probeMask1.draw()
                    probeMask2.draw()
                    probeMask3.draw()
                    probeMask4.draw()
                    # fixation.setRadius(3)
                    # fixation.draw()

                # PROBE 1
                if t_interval_1 >= frameN > t_fixation:
                    # dots stay in place for probe1
                    dots.draw()
                    probeMask1.draw()
                    probeMask2.draw()
                    probeMask3.draw()
                    probeMask4.draw()
                    # probe1.draw()

                    # SIMULTANEOUS CONDITION
                    # # if no ISI, draw probe2
                    # if ISI == -1:
                    #     # unless sep == 99, code for single probe only
                    #     if sep <= 18:
                    #         probe2.draw()
                    # fixation.setRadius(3)
                    # fixation.draw()

                # ISI
                if t_ISI >= frameN > t_interval_1:
                    # new dot positions during ISI
                    alpha = alpha + rotSpeed
                    new_x = r_dots * np.cos(alpha)
                    new_y = r_dots * np.sin(alpha)
                    dots.xys = np.array([new_x, new_y]).transpose()
                    dots.draw()
                    probeMask1.draw()
                    probeMask2.draw()
                    probeMask3.draw()
                    probeMask4.draw()
                    # fixation.setRadius(3)
                    # fixation.draw()

                    # PROBE 2
                if t_interval_2 >= frameN > t_ISI:
                    # if expInfo['9. Background motion during'] == 'transient&probe2':
                    if Background_motion_during == 'transient&probe2':
                        # background keeps moving as as probe2 appears
                        alpha = alpha + rotSpeed
                    new_x = r_dots * np.cos(alpha)
                    new_y = r_dots * np.sin(alpha)
                    dots.xys = np.array([new_x, new_y]).transpose()
                    dots.draw()
                    probeMask1.draw()
                    probeMask2.draw()
                    probeMask3.draw()
                    probeMask4.draw()
                    # if ISI >= 0:
                    #     if sep <= 18:
                    #         probe2.draw()
                    # fixation.setRadius(3)
                    # fixation.draw()

                # ANSWER
                if frameN > t_interval_2:
                    # dots remail stationary after probe2 until after keypress
                    dots.draw()
                    probeMask1.draw()
                    probeMask2.draw()
                    probeMask3.draw()
                    probeMask4.draw()
                    # fixation.setRadius(2)
                    # fixation.draw()

                    # ANSWER
                    # resp can be undefined
                    resp = event.BuilderKeyResponse()
                    theseKeys = event.getKeys(keyList=['num_5', 'num_4', 'num_1',
                                                       'num_2', 'w', 'q', 'a', 's'])
                    if len(theseKeys) > 0:  # at least one key was pressed
                        resp.keys = theseKeys[-1]  # just the last key pressed
                        resp.rt = resp.clock.getTime()

                        # # default assume response incorrect unless meets criteria below
                        # resp.corr = 0
                        #
                        # if corner == 45:
                        #     if (resp.keys == 'w') or (resp.keys == 'num_5'):
                        #         resp.corr = 1
                        # elif corner == 135:
                        #     if (resp.keys == 'q') or (resp.keys == 'num_4'):
                        #         resp.corr = 1
                        # elif corner == 225:
                        #     if (resp.keys == 'a') or (resp.keys == 'num_1'):
                        #         resp.corr = 1
                        # elif corner == 315:
                        #     if (resp.keys == 's') or (resp.keys == 'num_2'):
                        #         resp.corr = 1

                        repeat = False
                        continueRoutine = False

                # check for quit
                if event.getKeys(keyList=["escape"]):
                    core.quit()

                # # redo the trial if i think i made a mistake
                # if event.getKeys(keyList=["r"]) or event.getKeys(keyList=['num_9']):
                #     repeat = True
                #     continueRoutine = False
                #     continue

                # refresh the screen
                if continueRoutine:
                    win.flip()

        # thisExp.addData('stair', stairNum)
        # thisExp.addData('probe_jump', target_jump)
        # thisExp.addData('probeColor1', probeColor1)
        # thisExp.addData('probeColor255', probeColor255)
        # thisExp.addData('probeLum', probeLum)
        # thisExp.addData('trial_response', resp.corr)
        # thisExp.addData('BGspeed', rotSpeed)
        # thisExp.addData('corner', corner)
        # thisExp.addData('probe_ecc', probe_ecc)
        # thisExp.addData('resp.rt', resp.rt)
        # thisExp.addData('total_nTrials', total_nTrials)
        # thisExp.addData('orientation', orientation)
        # thisExp.nextEntry()
        #
        # thisStair.newValue(resp.corr)  # so that the staircase adjusts itself
