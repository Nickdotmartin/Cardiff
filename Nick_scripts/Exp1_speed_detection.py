from __future__ import division

import copy
import os
import random
import numpy as np
from datetime import datetime
from math import *

from psychopy import __version__ as psychopy_version
from psychopy import gui, visual, core, data, event, monitors

from PsychoPy_tools import check_correct_monitor, get_pixel_mm_deg_values
from kestenSTmaxVal import Staircase

"""
Is based on exp1 - is motion helps targhtes be detected, then for targets of the same size and duration, 
a moving target should be detected with lower luminance than a stationary one.

This script will test this with probes onscreeen for 4 frames (same as ISI0 cond), but 
with speeds of 0, .25., .5, .75, .1, 2, 3 frames per second.
"""


# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# Monitor config from monitor centre
monitor_name = 'asus_cal'  # 'NickMac' 'asus_cal' 'Asus_VG24' 'HP_24uh' 'ASUS_2_13_240Hz'
display_number = 1  # 0 indexed, 1 for external display

# Store info about the experiment session
expName = 'Exp1_speed_detection'  # from the Builder filename that created this script


expInfo = {'1_Participant_name': 'Nick',
           '2_run_number': 2,
           '3_fps': [240, 60, 144],
           # to compare with exp 1 ISIs use [1, 4, 6, 9]
           # to compare with probes + ISIs use [5, 8, 10, 13],
           '4_Probe duration in frames': [4, 6, 8, 10, 13, 1, 4, 6, 9, 24],  # ['12', '3', '6', '12', '18', '24', '30', '36', '120'],
           # '5_prelim_bg_flow_ms': [0, 70],
           '7_Trials_counter': [True, False],
           # '8_Background': ['None', 'flow_rad', ],
           # '9_bg_speed_cond': ['Normal', 'Half-speed'],
           }

# GUI
dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
if not dlg.OK:
    core.quit()  # user pressed cancel

expInfo['date'] = datetime.now().strftime("%d/%m/%Y")
expInfo['time'] = datetime.now().strftime("%H:%M:%S")

# GUI SETTINGS
participant_name = expInfo['1_Participant_name']
run_number = expInfo['2_run_number']
fps = int(expInfo['3_fps'])
trials_counter = expInfo['7_Trials_counter']
background = None  # expInfo['8_Background']
# bg_speed_cond = expInfo['9_bg_speed_cond']

n_trials_per_stair = 25  # int((expInfo['2. Repetitions']))
probe_duration = int(expInfo['4_Probe duration in frames'])

# VARIABLES
probe_ecc = 4

# # background motion to start 70ms before probe1 (e.g., 17frames at 240Hz).
# prelim_bg_flow_ms = int(expInfo['5_prelim_bg_flow_ms'])
# print(f'prelim_bg_flow_ms ({type(prelim_bg_flow_ms)}): {prelim_bg_flow_ms}')
# prelim_bg_flow_fr = int(prelim_bg_flow_ms * fps / 1000)

# # ISI timing in ms and frames
'''milliseconds: [100, 50, 41.66, 37.5, 33.34, 25, 16.67, 8.33, 0]
   frames@240hz: [24,  12,  10,    9,    8,     6,  4,    2,    0]
'''

n_stairs = 7
probe_speeds = [0, .25, .5, .75, 1.0, 1.5, 2.0]

# probe_directions = [-1, 1]  # -1=outward, 1=inwards
# probe_dir_name = ['out' if i == -1 else 'in' for i in probe_directions]

print(f"\nprobe_speeds: {probe_speeds}\n")

participant_run = f'{participant_name}_{run_number}'


# FILENAME
filename = os.path.join(_thisDir, expName, participant_name, participant_run,
                        f'probeDur{probe_duration}', f'{participant_run}_output')
print(f'saving output to: {filename}')
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
bgLumP = 20
bgLum = maxLum * bgLumP / 100
bgColor255 = bgLum * LumColor255Factor  # I could switch to using this.

#  rgb: -0.1 = rgb1: .45 = rgb255: 114.75 = lum: 47.8
flow_bgcolor = [-0.1, -0.1, -0.1]  # dark grey
# flow_bgcolor = [-0.6, -0.6, -0.6]  # these values would be equivalent to exp1a

if background == 'flow_rad':
    # background colour: use darker grey.  set once here and use elsewhere
    bgcolor = flow_bgcolor
else:
    # bgcolor = bgColor255
    bgcolor = bgColor255


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
if monitor_name in  ['ASUS_2_13_240Hz', 'asus_cal']:
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
                    colorSpace='rgb255',
                    color=bgcolor,  # bgcolor from Martin's flow script, not bgColor255
                    winType='pyglet',  # I've added pyglet to make it work on pycharm/mac
                    pos=[1, -1],  # pos gives position of top-left of screen
                    units='pix',
                    screen=display_number,
                    allowGUI=False,
                    fullscr=use_full_screen,
                    )


# CLOCK
trialClock = core.Clock()


# ELEMENTS
# fixation bull eye
if background == 'flow_rad':
    fixation = visual.Circle(win, radius=2, units='pix', lineColor='black', fillColor='black')
else:
    fixation = visual.Circle(win, radius=2, units='pix', lineColor='white', fillColor='black')


# PROBEs
# # Martin' probe for flow parsing
# raisedCosTexture0 = visual.filters.makeMask(256, shape='raisedCosine',
#                                             fringeWidth=0.8, radius=[1.0, 1.0])
# probe = visual.GratingStim(win, mask=raisedCosTexture0, contrast=0.5,
#                            size=(30, 30), units='pix', color='green',)

# integration stimuli probe
probeVert = [(0, 0), (1, 0), (1, 1), (2, 1), (2, -1), (1, -1),
             (1, -2), (-1, -2), (-1, -1), (0, -1)]

probe = visual.ShapeStim(win, vertices=probeVert,
                         fillColor=(1.0, 1.0, 1.0),
                         lineWidth=0, opacity=1, size=1, interpolate=False)

#
# # MASK BEHIND PROBES
# raisedCosTexture1 = visual.filters.makeMask(256, shape='raisedCosine',
#                                             fringeWidth=0.3, radius=[1.0, 1.0])
# mask_size = 150
# probeMask1 = visual.GratingStim(win, mask=raisedCosTexture1, tex=None,
#                                 size=(mask_size, mask_size), units='pix', color=bgcolor)
# probeMask2 = visual.GratingStim(win, mask=raisedCosTexture1, tex=None,
#                                 size=(mask_size, mask_size), units='pix', color=bgcolor)
# probeMask3 = visual.GratingStim(win, mask=raisedCosTexture1, tex=None,
#                                 size=(mask_size, mask_size), units='pix', color=bgcolor)
# probeMask4 = visual.GratingStim(win, mask=raisedCosTexture1, tex=None,
#                                 size=(mask_size, mask_size), units='pix', color=bgcolor)
#
#
# # BACKGROUND
# # flow_dots
# if bg_speed_cond == 'Normal':
#     flow_speed = 0.2
# elif bg_speed_cond == 'Half-speed':
#     flow_speed = 0.1
# else:
#     print(f"bg_speed_cond: {bg_speed_cond}")
#     raise ValueError(f'background speed should be selected from drop down menu: Normal or Half-speed')
# nDots = 10000
# flow_dots = visual.ElementArrayStim(win, elementTex=None, elementMask='circle',
#                                     units='pix', nElements=nDots, sizes=10,
#                                     colors=[flow_bgcolor[0]-0.3,
#                                             flow_bgcolor[1],
#                                             flow_bgcolor[2]-0.3])
#
# # full screen mask to blend off edges and fade to black
# # Create a raisedCosine mask array and assign it to a Grating stimulus (grey outside, transparent inside)
# # this was useful http://www.cogsci.nl/blog/tutorials/211-a-bit-about-patches-textures-and-masks-in-psychopy
# raisedCosTexture2 = visual.filters.makeMask(1080, shape='raisedCosine', fringeWidth=0.6, radius=[1.0, 1.0])
# invRaisedCosTexture = -raisedCosTexture2  # inverts mask to blur edges instead of center
# blankslab = np.ones((1080, 420))  # create blank slabs to put to left and right of image
# mmask = np.append(blankslab, invRaisedCosTexture, axis=1)  # append blank slab to left
# mmask = np.append(mmask, blankslab, axis=1)  # and right
# dotsMask = visual.GratingStim(win, mask=mmask, tex=None, contrast=1.0,
#                               size=(widthPix, heightPix), units='pix', color='black')
# # changed dotsmask color from grey
# # above fades to black round edges which makes screen edges less visible
#
# # function for wrapping flow_dots back into volume
# # its is used as WrapPoints(z, minDist, maxDist)
# # Any dots with a z (depth) value out of bounds are transformed to be in bounds
# def WrapPoints(ii, imin, imax):
#     lessthanmin = (ii < imin)
#     ii[lessthanmin] = ii[lessthanmin] + (imax-imin)
#     morethanmax = (ii > imax)
#     ii[morethanmax] = ii[morethanmax] - (imax-imin)
#
#
# taille = 5000  # french for 'size', 'cut', 'trim', 'clip' etc
# minDist = 0.5
# maxDist = 5


# MOUSE - Hide cursor
myMouse = event.Mouse(visible=False)

# INSTRUCTIONS
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
                               font='Arial', pos=[0, 0], height=20, wrapWidth=None,
                               ori=0, color='white', colorSpace='rgb255',
                               opacity=1, languageStyle='LTR', depth=0.0)
while not event.getKeys():
    fixation.setRadius(3)
    fixation.draw()
    instructions.draw()
    win.flip()

# Trial counter
trials_counter = visual.TextStim(win=win, name='trials_counter', text="???",
                                 font='Arial', height=20,
                                 # default set to black (e.g., invisible)
                                 color=bgColor255,
                                 pos=[-widthPix*.45, -heightPix*.45]
                                 )
if trials_counter:
    # if trials counter yes, change colour to white.
    trials_counter.color = 'white'

# BREAKS
total_n_trials = int(n_trials_per_stair * n_stairs)
take_break = int(total_n_trials/3)+1
print(f"take_break every {take_break} trials.")
breaks = visual.TextStim(win=win, name='breaks',
                         text="turn on the light and take at least 30-seconds break.\n\n"
                              "When you are ready to continue, press any key.",
                         font='Arial', height=20, colorSpace='rgb255', color='white')

end_of_exp = visual.TextStim(win=win, name='end_of_exp',
                             text="You have completed this experiment.\n"
                                  "Thank you for your time.\n\n"
                                  "Press any key to return to the desktop.",
                             font='Arial', height=20)

while not event.getKeys():
    fixation.setRadius(3)
    fixation.draw()
    instructions.draw()
    trials_counter.text = f"0/{total_n_trials}"
    trials_counter.draw()
    win.flip()

# STAIRCASE
expInfo['stair_list'] = list(range(n_stairs))
expInfo['n_trials_per_stair'] = n_trials_per_stair

# probeSpd = 3  # starting value
trial_number = 0

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

    # stair_name = f'{stair_idx}_fl_{flow_dir_name[stair_idx]}_pr_{probe_dir_name[stair_idx]}'
    stair_name = f'{stair_idx}_speed_{probe_speeds[stair_idx]}'
    thisStair = Staircase(name=f'{stair_name}', type='simple', value=stairStart,
                          C=stairStart * 0.6,  # step_size, typically 60% of reference stimulus
                          minRevs=3, minTrials=n_trials_per_stair, minVal=miniVal,
                          maxVal=maxiVal,
                          targetThresh=0.5, extraInfo=thisInfo)
    stairs.append(thisStair)


# EXPERIMENT
trial_number = 0
print('\n*** exp loop*** \n\n')

for step in range(n_trials_per_stair):
    np.random.shuffle(stairs)
    for thisStair in stairs:

        print(f"\ntrial_number: {trial_number}, step: {step}, thisStair: {thisStair}")

        trial_number = trial_number + 1
        trialClock.reset()

        stair_idx = thisStair.extraInfo['stair_idx']
        # flow_dir = flow_directions[stair_idx]
        # probeDir = random.choice([1, -1])

        # abs_probeSpeed = thisStair.next()
        # probeSpeed = abs_probeSpeed * probeDir
        probeSpeed = probe_speeds[stair_idx]
        print(f"probeSpeed: {probeSpeed}")

        contrastprobe = 0.5
        probeLum = thisStair.next()
        probeColor255 = int(probeLum * LumColor255Factor)
        probeColor1 = (probeColor255 * Color255Color1Factor) - 1
        print(f"probeLum: {probeLum}")

        # Black or White
        probe.color = [probeColor1, probeColor1, probeColor1]
        # probe2.color = [probeColor1, probeColor1*redfilter, probeColor1*redfilter]

        # # flow_dots
        # x = np.random.rand(nDots) * taille - taille / 2
        # y = np.random.rand(nDots) * taille - taille / 2
        # z = np.random.rand(nDots) * (maxDist - minDist) + minDist
        # # z was called z_flow but is actually z position like x and y
        # x_flow = x / z
        # y_flow = y / z


        # PROBE LOCATIONS
        # corners go CCW(!) 45=top-right, 135=top-left, 225=bottom-left, 315=bottom-right
        corner = np.random.choice([45, 135, 225, 315])

        # print(f'\tcorner: {corner}, flow_dir: {flow_dir}, probeSpeed: {probeSpeed}')
        # dist_from_fix is a constant giving distance form fixation,
        # dist_from_fix was previously 2 identical variables x_prob & y_prob.
        dist_from_fix = round((tan(np.deg2rad(probe_ecc)) * viewdistPix) / sqrt(2))
        # x_prob = y_prob = round((tan(np.deg2rad(probe_ecc)) * viewdistPix) / sqrt(2))

        # setting x and y positions depending on the side
        # corners go CCW(!) 45=top-right, 135=top-left, 225=bottom-left, 315=bottom-right
        if corner == 45:
            x_position = dist_from_fix
            y_position = dist_from_fix
        elif corner == 135:
            x_position = -dist_from_fix
            y_position = dist_from_fix
        elif corner == 225:
            x_position = -dist_from_fix
            y_position = -dist_from_fix
        elif corner == 315:
            x_position = dist_from_fix
            y_position = -dist_from_fix

        # probe position reset
        probe_x = 0
        probe_y = 0

        # # probe mask locations
        # probeMask1.setPos([dist_from_fix+1, dist_from_fix+1])
        # probeMask2.setPos([-dist_from_fix-1, dist_from_fix+1])
        # probeMask3.setPos([-dist_from_fix-1, -dist_from_fix-1])
        # probeMask4.setPos([dist_from_fix+1, -dist_from_fix-1])


        # timing in frames
        # fixation time is now 70ms shorted than previously.
        t_fixation = 1 * fps  # 240 frames - 70ms for fixation, e.g., <1 second.
        t_interval_1 = t_fixation + probe_duration  # probes appear during probe_duration (e.g., 240ms, 1 second).
        t_response = t_interval_1 + 10000 * fps

        # count frames
        nb_frames_motion = 0

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

                    # if background == 'flow_rad':
                    #     # draw flow_dots but with no motion
                    #     flow_dots.xys = np.array([x_flow, y_flow]).transpose()
                    #     flow_dots.draw()
                    #     probeMask1.draw()
                    #     probeMask2.draw()
                    #     probeMask3.draw()
                    #     probeMask4.draw()
                    #     dotsMask.draw()

                    fixation.setRadius(3)
                    fixation.draw()
                    trials_counter.draw()

                # Background motion prior to probe1
                # if t_bg_motion >= frameN > t_fixation:
                    # after fixation, before end of background motion
                    # if background == 'flow_rad':
                    #     # radial flow_dots motion
                    #     z = z + flow_speed * flow_dir
                    #     WrapPoints(z, minDist, maxDist)
                    #     x_flow = x / z
                    #     y_flow = y / z
                    #
                    #     flow_dots.xys = np.array([x_flow, y_flow]).transpose()
                    #     flow_dots.draw()
                    #
                    #     probeMask1.draw()
                    #     probeMask2.draw()
                    #     probeMask3.draw()
                    #     probeMask4.draw()
                    #     dotsMask.draw()

                    # fixation.setRadius(3)
                    # fixation.draw()
                    # probe1.draw()
                    # trials_counter.draw()


                # PROBE 1
                if t_interval_1 >= frameN > t_fixation:
                    # after background motion, before end of probe1 interval
                    # if background == 'flow_rad':
                    #     # radial flow_dots motion
                    #     z = z + flow_speed * flow_dir
                    #     WrapPoints(z, minDist, maxDist)
                    #     x_flow = x / z
                    #     y_flow = y / z
                    #
                    #     flow_dots.xys = np.array([x_flow, y_flow]).transpose()
                    #     flow_dots.draw()
                    #
                    #     probeMask1.draw()
                    #     probeMask2.draw()
                    #     probeMask3.draw()
                    #     probeMask4.draw()
                    #     dotsMask.draw()

                    fixation.setRadius(3)
                    fixation.draw()
                    # probe1.draw()
                    # probe.draw()
                    trials_counter.draw()


                    # draw probe if 1st interval
                    if corner == 45:  # top-right
                        probe_y = probe_y - probeSpeed
                        probe_x = probe_x - probeSpeed
                    elif corner == 135:  # top-left
                        probe_y = probe_y - probeSpeed
                        probe_x = probe_x + probeSpeed
                    elif corner == 225:  # botom-left
                        probe_y = probe_y + probeSpeed
                        probe_x = probe_x + probeSpeed
                    elif corner == 315:  # bottom-right
                        probe_y = probe_y + probeSpeed
                        probe_x = probe_x - probeSpeed
                    probe.setPos([x_position + probe_x, y_position + probe_y])
                    probe.draw()

                    nb_frames_motion = nb_frames_motion + 1


                # # ANSWER
                # # if frameN > t_interval_2:
                # if frameN > t_interval_1:
                #     # after probe 2 interval
                #     # if background == 'flow_rad':
                #     #     # draw flow_dots but with no motion
                #     #     flow_dots.draw()
                #     #     probeMask1.draw()
                #     #     probeMask2.draw()
                #     #     probeMask3.draw()
                #     #     probeMask4.draw()
                #     #     dotsMask.draw()
                #
                #     fixation.setRadius(2)
                #     fixation.draw()
                #     trials_counter.draw()

                    # ANSWER
                if frameN > t_interval_1:
                    fixation.setRadius(2)
                    fixation.draw()
                    trials_counter.draw()

                    # ANSWER
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
                        # a response ends the routine
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
        thisExp.addData('probeLum', probeLum)
        thisExp.addData('probeColor1', probeColor1)
        thisExp.addData('probeColor255', probeColor255)
        thisExp.addData('corner', corner)
        thisExp.addData('probe_dur', probe_duration)
        thisExp.addData('separation', 99)
        thisExp.addData('ISI', 0)
        # thisExp.addData('congruent', congruent)
        # thisExp.addData('answer', answer)
        # thisExp.addData('rel_answer', rel_answer)
        thisExp.addData('trial_response', resp.corr)
        thisExp.addData('probeSpeed', probeSpeed)
        # thisExp.addData('abs_probeSpeed', abs_probeSpeed)
        thisExp.addData('resp.rt', resp.rt)
        # thisExp.addData('BGspeed', flow_speed)
        # thisExp.addData('prelim_bg_flow_ms', prelim_bg_flow_ms)
        thisExp.addData('expName', expName)
        thisExp.addData('monitor_name', monitor_name)
        thisExp.addData('selected_fps', fps)

        thisExp.nextEntry()

        # thisStair.newValue(rel_answer)  # so that the staircase adjusts itself
        thisStair.newValue(resp.corr)   # so that the staircase adjusts itself


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
