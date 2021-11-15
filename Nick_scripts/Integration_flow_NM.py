from __future__ import division
from psychopy import gui, visual, core, data, event, logging, monitors
from psychopy import __version__ as psychopy_version
import os
from numpy import deg2rad
import numpy as np
import random
import copy
from datetime import datetime
from math import *

from kestenSTmaxVal import Staircase
from PsychoPy_tools import check_correct_monitor



"""
This script takes: 
the probes from EXPERIMENT3_background_motion_SKR, and adds the option for tangent or radial jump.
the background radial motion is taken from integration_RiccoBloch_flow_new.
ISI is always >=0 (no simultaneous probes).
"""

# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# Monitor config from monitor centre
monitor_name = 'HP_24uh'  # 'NickMac' 'asus_cal' 'Asus_VG24' 'HP_24uh'
display_number = 1  # 0 indexed, 1 for external display


# Store info about the experiment session
expName = 'integration_flow'  # from the Builder filename that created this script

# todo: change dict names for trial counter, probe check and background motion.
# todo: change default probe dur back to 2, default ISI back to list.
# todo: add trials counter in from martin's flow integration script.
expInfo = {'1_Participant': 'testnm',
           '2_Probe_dur_in_frames_at_240hz': 50,  # 2,
           '3_fps': [60, 144, 240],
           '4_ISI_dur_in_ms': [100],  # [0, 8.33, 16.67, 25, 37.5, 50, 100],
           '5_Probe_orientation': ['ray', 'tangent'],
           '6_Probe_size': ['5pixels', '6pixels', '3pixels'],
           '7_Trials_counter': [True, False],
           '8_Background': ['flow_rad', 'None']
           }



# GUI
dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
if not dlg.OK:
    core.quit()  # user pressed cancel

expInfo['date'] = datetime.now().strftime("%d/%m/%Y")
expInfo['time'] = datetime.now().strftime("%H:%M:%S")

# GUI SETTINGS
participant_name = expInfo['1_Participant']
# todo: change trial_NUMBER TO total_n_trials or something
probe_duration = int(expInfo['2_Probe_dur_in_frames_at_240hz'])
fps = int(expInfo['3_fps'])
orientation = expInfo['5_Probe_orientation']
Probe_size = expInfo['6_Probe_size']
trials_counter = expInfo['7_Trials_counter']
background = expInfo['8_Background']

# ISI timing in ms and frames
'''ISI can be given (roughly) in ms, for any monitor it will try to match that value in frames.
The script uses on frames so ISI will be used in that format.
The actual ms timing is given for record.
This means that the experiment should have similar ms timings on monitors with different fps'''
ISI_selected_ms = float(expInfo['4_ISI_dur_in_ms'])
ISI_frames = int(ISI_selected_ms * fps / 1000)
ISI_actual_ms = (1/fps) * ISI_frames * 1000
ISI = ISI_frames

# VARIABLES
trial_number = 25
probe_ecc = 4
# Distances between probes
# this study does not include the two 99 values for single probe condition
separations = [18, 18, 6, 6, 3, 3, 2, 2, 1, 1, 0, 0]
# todo: should flow_dir be fixed like this or get rid of this and later
#  just have flow_dir = np.random.choice([1, -1])
# flow_direction is a list of [-1, 1...] of same length as separations
flow_direction = [-1, 1]*int(len(separations)/2)

# FILENAME
filename = f'{_thisDir}{os.sep}' \
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
maxColor255 = 255
minColor255 = 0
maxColor1 = 1
minColor1 = -1
bgLumP = 20
bgLum = maxLum * bgLumP / 100
flow_bgcolor = [-0.1, -0.1, -0.1]

if background == 'flow_rad':
    # background colour: use darker grey.  set once here and use elsewhere
    bgcolor = flow_bgcolor
else:
    bgColor255 = bgLum * LumColor255Factor
    bgcolor = bgColor255


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
viewdistPix = widthPix/monitorwidth*viewdist
mon = monitors.Monitor(monitor_name, width=monitorwidth, distance=viewdist)
mon.setSizePix((widthPix, heightPix))
mon.save()


# WINDOW SPEC
win = visual.Window(monitor=mon, size=(widthPix, heightPix),
                    colorSpace='rgb',
                    color=bgcolor,  # from martin's flow script
                    winType='pyglet',  # I've added pyglet to make it work on pycharm/mac
                    pos=[1, -1],  # pos gives position of top-left of screen
                    units='pix',
                    screen=display_number,
                    allowGUI=False,
                    )



# check correct monitor details (fps, size) have been accessed.
check_correct_monitor(monitor_name=monitor_name,
                      actual_size=win.size,
                      actual_fps=win.getActualFrameRate(),
                      verbose=True)



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
# flow
flow_speed = 0.2
nDots = 10000
flow = visual.ElementArrayStim(win, elementTex=None, elementMask='circle',
                               units='pix', nElements=nDots, sizes=10,
                               colors=[flow_bgcolor[0]-0.3, flow_bgcolor[1], flow_bgcolor[2]-0.3])


# full screen mask to blend off edges and fade to black
# Create a raisedCosine mask array and assign it to a Grating stimulus (grey outside, transparent inside)
# this was useful http://www.cogsci.nl/blog/tutorials/211-a-bit-about-patches-textures-and-masks-in-psychopy
# todo: make the mask slightly larger so it closer to top and bottom edge
raisedCosTexture2 = visual.filters.makeMask(1080, shape='raisedCosine', fringeWidth=0.6, radius=[1.0, 1.0])
invRaisedCosTexture = -raisedCosTexture2  # inverts mask to blur edges instead of center
blankslab = np.ones((1080, 420))  # create blank slabs to put to left and right of image
mmask = np.append(blankslab, invRaisedCosTexture, axis=1)  # append blank slab to left
mmask = np.append(mmask, blankslab, axis=1)  # and right
dotsMask = visual.GratingStim(win, mask=mmask, tex=None, contrast=1.0, size=(1920, 1080), units='pix', color='black')
# changed dotsmask color from grey
# above fades to black round edges which makes screen edges less visible


# function for wrapping flow dots back into volume
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


# Trial counter
trials_counter = visual.TextStim(win=win, name='trials_counter', text="???",
                                 font='Arial', height=20,
                                 # default set to black (e.g., invisible)
                                 color=[-1.0, -1.0, -1.0],
                                 pos=[-800, -500])
if trials_counter:
    # if trials counter yes, change colour to white.
    trials_counter.color = [1, 1, 1]


# MOUSE - Hide cursor
myMouse = event.Mouse(visible=False)

# INSTRUCTION
instructions = visual.TextStim(win=win, name='instructions',
                               text="[q] or [4] top-left\n "
                                    "[w] or [5] top-right\n "
                                    "[a] or [1] bottom-left\n "
                                    "[s] or [2] bottom-right \n\n "
                                    "[r] or [9] to redo the previous trial \n\n"
                                    "[Space bar] to start",
                               font='Arial', pos=[0, 0], height=20, ori=0,
                               color=[255, 255, 255], colorSpace='rgb255',
                               opacity=1, languageStyle='LTR', depth=0.0)

# BREAKS
# todo: add breaks to the script - see exp 1 line 389 ish
breaks = visual.TextStim(win=win, name='breaks',
                         text="turn on the light and  take at least 30-seconds break.",
                         font='Arial', pos=[0, 0], height=20, ori=0, color=[255, 255, 255],
                         colorSpace='rgb255', opacity=1, languageStyle='LTR', depth=0.0)


while not event.getKeys():
    instructions.draw()
    win.flip()

# STAIRCASE
total_nTrials = 0
# number of startpoints will depend on whether the separations list includes 99 (single probe cond)
expInfo['startPoints'] = list(range(1, len(separations)))

expInfo['nTrials'] = trial_number

stairStart = maxLum
miniVal = bgLum
maxiVal = maxLum

print('\nexpInfo (dict)')
for k, v in expInfo.items():
    print(f"{k}: {v}")

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

# EXPERIMENT
for trialN in range(expInfo['nTrials']):
    np.random.shuffle(stairs)
    for thisStair in stairs:

        total_nTrials = total_nTrials + 1
        trialClock.reset()


        stairNum = thisStair.extraInfo['thisStart']
        # stairNum is 1 indexed, but accessing items from zero-indexed lists, so -1
        sep = separations[stairNum - 1]
        flow_dir = flow_direction[stairNum-1]

        # flow
        x = np.random.rand(nDots) * taille - taille / 2
        y = np.random.rand(nDots) * taille - taille / 2
        z = np.random.rand(nDots) * (maxDist - minDist) + minDist
        # z was called z_flow but is actually z position like x and y
        x_flow = x / z
        y_flow = y / z

        # PROBE
        target_jump = np.random.choice([1, -1])  # direction in which the probe jumps : CW or CCW
        # staircase varies probeLum
        probeLum = thisStair.next()
        probeColor255 = probeLum * LumColor255Factor
        probeColor1 = (probeColor255 * Color255Color1Factor) - 1


        # Black or White
        # # I don't understand this, why do you want the opp polarity probe if there is a red filter?
        probe1.color = [probeColor1, probeColor1, probeColor1]
        probe2.color = [probeColor1, probeColor1, probeColor1]

        # PROBE LOCATIONS
        # corners go CCW(!) 45=top-right, 135=top-left, 225=bottom-left, 315=bottom-right
        corner = np.random.choice([45, 135, 225, 315])
        # x_prob and y_prob are constants, so can be defined outside of loop.
        # These just set the distance of the probes from fixation
        x_prob = round((tan(np.deg2rad(probe_ecc)) * viewdistPix) / sqrt(2))
        y_prob = round((tan(np.deg2rad(probe_ecc)) * viewdistPix) / sqrt(2))

        probeMask1.setPos([x_prob+1, y_prob+1])
        probeMask2.setPos([-x_prob-1, y_prob+1])
        probeMask3.setPos([-x_prob-1, -y_prob-1])
        probeMask4.setPos([x_prob+1, -y_prob-1])


        # reset probe ori
        probe1.ori = 0
        probe2.ori = 0
        if corner == 45:
            # in top-right corner, both x and y increase (right and up)
            p1_x = x_prob * 1
            p1_y = y_prob * 1

            #  'orientation' here refers to the relationship between probes,
            #  whereas probe1.ori refers to rotational angle of inidividual probe stimulus
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
            elif orientation == 'ray':
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

        elif corner == 135:
            # in top-left corner, x decreases (left) and y increases (up)
            p1_x = x_prob * -1
            p1_y = y_prob * 1
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
            elif orientation == 'ray':
                if target_jump == 1:  # CCW
                    probe1.ori = 180
                    probe2.ori = 0
                    probe2.pos = [p1_x + sep - 1, p1_y - sep]
                elif target_jump == -1:  # CW
                    probe1.ori = 0
                    probe2.ori = 180
                    probe2.pos = [p1_x - sep + 1, p1_y + sep]


        elif corner == 225:
            # in bottom left corner, both x and y decrease (left and down)
            p1_x = x_prob * -1
            p1_y = y_prob * -1
            if orientation == 'tangent':
                if target_jump == 1:  # CCW
                    probe1.ori = 180
                    probe2.ori = 0
                    probe2.pos = [p1_x + sep - 1, p1_y - sep]
                elif target_jump == -1:  # CW
                    probe1.ori = 0
                    probe2.ori = 180
                    probe2.pos = [p1_x - sep + 1, p1_y + sep]
            elif orientation == 'ray':
                if target_jump == 1:  # CCW
                    probe1.ori = 270
                    probe2.ori = 90
                    probe2.pos = [p1_x - sep + 1, p1_y - sep]
                elif target_jump == -1:  # CW
                    probe1.ori = 90
                    probe2.ori = 270
                    probe2.pos = [p1_x + sep - 1, p1_y + sep]


        else:
            corner = 315
            # in bottom-right corner, x increases (right) and y decreases (down)
            p1_x = x_prob * 1
            p1_y = y_prob * -1
            if orientation == 'tangent':
                if target_jump == 1:  # CCW
                    probe1.ori = 270
                    probe2.ori = 90
                    probe2.pos = [p1_x - sep + 1, p1_y - sep]
                elif target_jump == -1:  # CW
                    probe1.ori = 90
                    probe2.ori = 270
                    probe2.pos = [p1_x + sep - 1, p1_y + sep]
            elif orientation == 'ray':
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


        probe1.pos = [p1_x, p1_y]


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

                # FIXATION
                if t_fixation >= frameN > 0:
                    # before fixation has finished
                    trials_counter.text = f"{total_nTrials}/120"

                    if background == 'flow_rad':
                        # flow
                        flow.xys = np.array([x_flow, y_flow]).transpose()
                        flow.draw()
                        # probe masks
                        probeMask1.draw()
                        probeMask2.draw()
                        probeMask3.draw()
                        probeMask4.draw()

                        dotsMask.draw()
                    trials_counter.draw()

                    fixation.setRadius(3)
                    fixation.draw()

                # PROBE 1
                if t_interval_1 >= frameN > t_fixation:
                    # after fixation, before end of probe1 interval
                    if background =='flow_rad':
                        # flow
                        flow.xys = np.array([x_flow, y_flow]).transpose()
                        flow.draw()
                        # probe masks
                        probeMask1.draw()
                        probeMask2.draw()
                        probeMask3.draw()
                        probeMask4.draw()

                        dotsMask.draw()
                    trials_counter.draw()

                    probe1.draw()

                    fixation.setRadius(3)
                    fixation.draw()

                # ISI
                if t_ISI >= frameN > t_interval_1:
                    if background == 'flow_rad':
                        # radial flow
                        z = z + flow_speed * flow_dir
                        WrapPoints(z, minDist, maxDist)
                        x_flow = x / z
                        y_flow = y / z

                        flow.xys = np.array([x_flow, y_flow]).transpose()
                        flow.draw()

                        # probe masks
                        probeMask1.draw()
                        probeMask2.draw()
                        probeMask3.draw()
                        probeMask4.draw()

                        dotsMask.draw()
                    trials_counter.draw()

                    fixation.setRadius(3)
                    fixation.draw()

                # PROBE 2
                if t_interval_2 >= frameN > t_ISI:
                    # after ISI but before end of probe2 interval
                    if background == 'flow_rad':

                        # if flow motion during probe2
                            # # radial flow
                            # z = z + flow_speed * flow_dir
                            # WrapPoints(z, minDist, maxDist)
                            # x_flow = x / z
                            # y_flow = y / z
                            #
                            # flow.xys = np.array([x_flow, y_flow]).transpose()
                        flow.draw()

                        # probe masks
                        probeMask1.draw()
                        probeMask2.draw()
                        probeMask3.draw()
                        probeMask4.draw()
                        dotsMask.draw()

                    trials_counter.draw()

                    probe2.draw()
                    fixation.setRadius(3)
                    fixation.draw()

                # ANSWER
                if frameN > t_interval_2:
                    # after probe 2 interval
                    if background == 'flow_rad':
                        # flow
                        flow.draw()
                        # probe masks
                        probeMask1.draw()
                        probeMask2.draw()
                        probeMask3.draw()
                        probeMask4.draw()

                        dotsMask.draw()

                    trials_counter.draw()

                    fixation.setRadius(2)
                    fixation.draw()

                    # ANSWER
                    # resp can be undefined
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
                    core.quit()

                # redo the trial if i think i made a mistake
                if event.getKeys(keyList=["r"]) or event.getKeys(keyList=['num_9']):
                    repeat = True
                    continueRoutine = False
                    continue

                # refresh the screen
                if continueRoutine:
                    win.flip()

        # add to exp dict
        thisExp.addData('stair', stairNum)
        thisExp.addData('probe_jump', target_jump)
        thisExp.addData('probeColor1', probeColor1)
        thisExp.addData('probeColor255', probeColor255)
        thisExp.addData('probeLum', probeLum)
        thisExp.addData('trial_response', resp.corr)
        # todo: add in something for background speed to save here?
        # thisExp.addData('BGspeed', rotSpeed)
        thisExp.addData('corner', corner)
        thisExp.addData('probe_ecc', probe_ecc)
        thisExp.addData('resp.rt', resp.rt)
        thisExp.addData('total_nTrials', total_nTrials)
        thisExp.addData('orientation', orientation)
        thisExp.nextEntry()

        thisStair.newValue(resp.corr)  # so that the staircase adjusts itself
