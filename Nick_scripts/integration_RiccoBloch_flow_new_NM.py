from __future__ import division
from psychopy import gui, visual, core, data, event, logging, monitors
from psychopy import __version__ as psychopy_version
import os
import numpy as np
from numpy.random import shuffle
import random
import copy
from datetime import datetime
from math import *

from kestenSTmaxVal import Staircase

"""
'Ricco’s law holds: threshold intensity multiplied by the area equals a constant. 
This means that over this area, which embraces several hundred rods, 
light falling on the individual rods summates, or accumulates, its effects completely, 
so that 100 quanta falling on a single rod are as effective as one quantum falling simultaneously on 100 rods.' 
from https://www.britannica.com/science/human-eye/Plexiform-layers#ref531581

Script designed with 2 conditions in minds: ricco's area and ricco's separation.
Script currently only completed for ricco's area.
"""

# sets psychoPy to only log critical messages
logging.console.setLevel(logging.CRITICAL)
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)


# Monitor config from monitor centre
monitor_name = 'HP_24uh'  # 'NickMac' 'asus_cal' 'Asus_VG24' 'HP_24uh' 'ASUS_2_13_240Hz'
# gamma set at 2.1  [####### this comment is incorrect, its set above i think ############]
display_number = 1  # 0 indexed, 1 for external display

# Store info about the experiment session
psychopyVersion = psychopy_version  # '3.1.0'
expName = 'RadialDots'  # from the Builder filename that created this script
expInfo = {'1. Experiment': ['Ricco_separation', 'Ricco_area'],
           '2. Participant': 'testnm',
           '3. Probe duration in frames': '2',
           '4. fps': ['60', '144', '240'],
           '5. Trials counter': ['no', 'yes'],
           '6. Probe eccentricity in deg': ['4', '8', '10'],
           '7. ISI duration in ms': ['0', '8.33', '16.67', '25', '37.5', '50', '100', 'noISI'],
           '8. Gamma corrected': ['work in progress'],
           '9. Probe orientation': ['ray', 'tangent'],
           '91. Probe check': ['no', 'yes'],
           '99. Background': ['flow', 'static', 'nothing']}

# GUI
dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
if not dlg.OK:
    core.quit()  # user pressed cancel

expInfo['time'] = datetime.now().strftime("%H:%M:%S")
expInfo['date'] = datetime.now().strftime("%d/%m/%Y")

# GUI SETTINGS
experiment = expInfo['1. Experiment']
participant_name = expInfo['2. Participant']
trial_number = 20
probe_duration = int(expInfo['3. Probe duration in frames'])
probe_ecc = int(expInfo['6. Probe eccentricity in deg'])
fps = int(expInfo['4. fps'])
orientation = expInfo['9. Probe orientation']
probe_check = expInfo['91. Probe check']
background = expInfo['99. Background']

if orientation == 'tangent':
    flow_speed = 7
elif orientation == 'ray':
    flow_speed = 0.2

lineW = 1  # unused
lineL = 3  # I think this is the lineLength of the probe - e.g., 3 pixels long.
lineL2 = 3  # linelength of probe2 - 3 pixels
lineL_test = 1  # unused

# todo: selecting the area of 1 causes problems.
#  probe_s makes a list from 0 to selectied area value (e.g., list(range(0, 1))
#  which returns [0], This crashed when it tried to access probeW = probe_s[1]
# areas = [19, 13, 7, 4, 2, 1, 2, 3, 4, 5, 7]  # original values for area experiment
areas = [20, 14, 8, 5, 3, 2, 3, 4, 5, 6, 8]  # Nick new values, each +1

separations = [18, 18, 18, 18, 6, 6, 6, 6, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0]

# Where appears probe2 relative to probe1
# Tangent: positive --> CCW; negative --> CW
# Ray: positive --> outward; negative --> inward
PosOrNeg = [-1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, 1]
flow_direction = [-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1]
# flow_direction = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
# flow_direction = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

# todo: make this into a dict to look up ISI value from ms keys
if expInfo['7. ISI duration in ms'] == '0':
    ISI = 0
elif expInfo['7. ISI duration in ms'] == '8.33':
    ISI = 2
elif expInfo['7. ISI duration in ms'] == '16.67':
    ISI = 4
elif expInfo['7. ISI duration in ms'] == '25':
    ISI = 6
elif expInfo['7. ISI duration in ms'] == '37.5':
    ISI = 9
elif expInfo['7. ISI duration in ms'] == '50':
    ISI = 12
elif expInfo['7. ISI duration in ms'] == '100':
    ISI = 24
elif expInfo['7. ISI duration in ms'] == 'noISI':
    ISI = -1

# FILENAME
filename = f'{_thisDir}{os.sep}integration_RiccoBloch_new{os.sep}' \
           f'{experiment}_{orientation}_{background}' \
           f'_ISI{expInfo["7. ISI duration in ms"]}' \
           f'_ECC{expInfo["6. Probe eccentricity in deg"]}' \
           f'{os.sep}{participant_name}' \
           f'{os.sep}{participant_name}'


# Experiment Handler
thisExp = data.ExperimentHandler(name=expName, version=psychopy_version,
                                 extraInfo=expInfo, runtimeInfo=None,
                                 savePickle=None, saveWideText=True,
                                 dataFileName=filename)

# background colour: use darker grey.  set once here and use elsewhere
bgcolor = [-0.1, -0.1, -0.1]

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
                    colorSpace='rgb', color=bgcolor,
                    winType='pyglet',  # I've added this to make it work on pycharm/mac
                    pos=[1, -1],  # pos gives position of top-left of screen
                    units='pix', screen=1,
                    allowGUI=False,
                    # fullscr=True
                    )

# check correct monitor details (fps, size) have been accessed.
print(win.monitor.name, win.monitor.getSizePix())
actualFrameRate = int(win.getActualFrameRate())
if fps in list(range(actualFrameRate-2, actualFrameRate+2)):
    print("fps matches actual frame rate")
else:
    # if values don't match, quit experiment
    print(f"fps ({fps}) does not match actual frame rate ({actualFrameRate})")
    core.quit()

actual_size = win.size
if list(mon_dict['size']) == list(actual_size):
    print(f"monitor is expected size")
elif list(mon_dict['size']) == list(actual_size/2):
    print(f"actual size is double expected size - Its ok, just a mac retina display bug.")
else:
    print(f"Display size ({actual_size}) does not match expected size from montior centre ({mon_dict['size']})")
    # check sizes seems unreliable,
    # it returns different values for same screen if different mon_names are used!
    check_sizes = win._checkMatchingSizes(mon_dict['size'], actual_size)
    print(check_sizes)
    core.quit()


# CLOCK
trialClock = core.Clock()

# ELEMENTS
# fixation
fixation = visual.Circle(win, radius=2, units='pix',
                         lineColor='black', fillColor='black')

# PROBES
# todo: place probes 1&2 outside if statement.
#  only change is that probe1 is supposed to be blue, but it shows white anyway
if probe_check == 'yes':
    # probe_test places a red dot at the location of the first probe that remains
    # so that it is clear that the second probe is in a different location
    probe1 = visual.Line(win, lineWidth=1.0, units='pix', lineColor='blue',
                         interpolate=False)
    probe2 = visual.Line(win, lineWidth=1.0, units='pix', lineColor='white',
                         interpolate=False)
    probe_test = visual.Line(win, lineWidth=1.0, units='pix', lineColor='red',
                             interpolate=False)
else:
    probe1 = visual.Line(win, lineWidth=1.0, units='pix', lineColor='white',
                         interpolate=False)
    probe2 = visual.Line(win, lineWidth=1.0, units='pix', lineColor='white',
                         interpolate=False)

probe_eccentricity_rad = np.deg2rad(probe_ecc)
probe_eccentricity_cm = tan(probe_eccentricity_rad)*viewdist
probe_eccentricity_pix = probe_eccentricity_cm*(widthPix/monitorwidth)

# mask behind probes
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
nDots = 10000
flow = visual.ElementArrayStim(win, elementTex=None, elementMask='circle',
                               units='pix', nElements=nDots, sizes=10,
                               colors=[bgcolor[0]-0.3, bgcolor[1], bgcolor[2]-0.3])


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


# Trial counter
trials_counter = visual.TextStim(win=win, name='trials_counter', text="???",
                                 font='Arial', height=20,
                                 # default set to black (e.g., invisible)
                                 color=[-1.0, -1.0, -1.0],
                                 pos=[-800, -500])
if expInfo['5. Trials counter'] == 'yes':
    # if trials counter yes, change colour to white.
    trials_counter.color = [1, 1, 1]



# MOUSE
myMouse = event.Mouse(visible=False) 

# INSTRUCTIONS
instructions = visual.TextStim(win=win, name='instructions',
                               text="[q] or [4] top-left\n"
                                    "[w] or [5] top-right\n"
                                    "[a] or [1] bottom-left\n"
                                    "[s] or [2] bottom-right\n\n"
                                    "[r] or [9] to redo the previous trial\n\n"
                                    "[Space bar] to start",
                               font='Arial', pos=[0, 0], height=20, ori=0,
                               color=[1, 1, 1], colorSpace='rgb', opacity=1,
                               languageStyle='LTR', depth=0.0)

while not event.getKeys():
    instructions.draw()
    win.flip()

# STAIRCASE
# todo: change names for total_n_trials, and trial number.
#  Trial number should start at zero and increase for each trial,
#  total_N_trials should be the limit.
#  This is the opposite of how they are at the mo

total_nTrials = 0
# old comments for starpoints: [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
# 6 window duration conditions
# expInfo['startPoints'] = list(range(1, 23))
expInfo['startPoints'] = list(range(1, len(separations)))
if experiment == 'Ricco_area':
    expInfo['startPoints'] = list(range(1, len(areas)))

expInfo['nTrials'] = trial_number

stairStart = 0.7  # 1.0
stairs = []
for thisStart in expInfo['startPoints']:
    thisInfo = copy.copy(expInfo)
    thisInfo['thisStart'] = thisStart  # we might want to keep track of this

    thisStair = Staircase(name='trials',
                          type='simple',
                          value=stairStart,
                          C=stairStart*0.6,  # typically 60% of reference stimulus
                          minRevs=3,
                          minTrials=trial_number,
                          minVal=0.0,
                          maxVal=1.0,
                          targetThresh=0.75,  # changed this from prev versions
                          extraInfo=thisInfo
                          )
    stairs.append(thisStair)


taille = 5000  # french for 'size', 'cut', 'trim', 'clip' etc
minDist = 0.5
maxDist = 5

# EXPERIMENT
for trialN in range(expInfo['nTrials']):
    shuffle(stairs)
    for thisStair in stairs:

        total_nTrials = total_nTrials + 1
        # staircase varies contrastprobe
        contrastprobe = thisStair.next()
        probe1.contrast = contrastprobe  # this is what a staircase varies
        probe2.contrast = contrastprobe

        stairNum = thisStair.extraInfo['thisStart']

        # print(f"stairNum: {stairNum}\n"
        #       f"contrastprobe: {contrastprobe}")


        trialClock.reset()
        
        # conditions
        # todo: probeWW list index out of range for line 340 on area exp.  fix this
        # stairNum is 1 indexed, but accessing items from zero-indexed lists, so -1
        if experiment == 'Ricco_area':
            probeWW = areas[stairNum-1]  # area experiment
        sep = separations[stairNum-1]  # separation experiment
        # target_jump_dir gives position of probe2 relative to probe 1 as -1 or 1,
        # If Tangent: positive=CCW; negative=CW; Ray: positive=outward; negative=inward
        target_jump_dir = PosOrNeg[stairNum-1]
        flow_dir = flow_direction[stairNum-1]
        # flow_speed = flow_speed *flow_dir
        
        # flow
        x = np.random.rand(nDots) * taille - taille/2
        y = np.random.rand(nDots) * taille - taille/2
        z = np.random.rand(nDots) * (maxDist - minDist) + minDist
        # z was called z_flow but is actually z position like x and y
        x_flow = x/z
        y_flow = y/z

        # PROBE LOCATION
        # corners go CCW(!) 45=top-right, 135=top-left, 225=bottom-left, 315=bottom-right
        corner = random.choice([45, 135, 225, 315])
        # x_prob/y_prob is distance from fixation (e.g., 134), in pixels I think
        # todo: change x_prob & y_prob to a single dist_from_fix variable
        x_prob = round((tan(np.deg2rad(probe_ecc))*viewdistPix)/sqrt(2))
        y_prob = round((tan(np.deg2rad(probe_ecc))*viewdistPix)/sqrt(2))

        probeMask1.setPos([x_prob+1, y_prob+1])
        probeMask2.setPos([-x_prob-1, y_prob+1])
        probeMask3.setPos([-x_prob-1, -y_prob-1])
        probeMask4.setPos([x_prob+1, -y_prob-1])
        
        if experiment == 'Ricco_area':
            # list of all positions between 0 and probeWW
            # todo: put probeww back in for Ricco_area exp
            probe_s = list(range(0, probeWW))
        elif experiment == 'Ricco_separation':
            # list of 2 positions: position probe1 and position probe 2
            probe_s = [0, sep]
            
        if corner == 45:
            # corner_coef variables give the direction the probes jumps [-1, 1]
            corner_coef_x = -1 * target_jump_dir
            corner_coef_y = 1 * target_jump_dir
            # probe line direction [-1, 1]
            dir_x = 1
            dir_y = 1
            # probe2 line start [-1, 0, 1]
            probe2start_x = 0
            probe2start_y = 1
            # not sure what this is: ray could be direction of probe2 line, or of jump.  [I've got [-270, 0, 270]
            ray_x = -2*x_prob-2
            ray_y = 0
            # directional velocity? [-1, 1] only used for tangent/traslational flow pattern to multiply speed
            vx_dir = -1
            vy_dir = +1
        elif corner == 135:
            corner_coef_x = -1 * target_jump_dir
            corner_coef_y = -1 * target_jump_dir
            dir_x = -1
            dir_y = 1
            probe2start_x = -1
            probe2start_y = 0
            ray_x = 0
            ray_y = -2*y_prob - 2
            vx_dir = -1
            vy_dir = -1
        elif corner == 225:
            corner_coef_x = 1 * target_jump_dir
            corner_coef_y = -1 * target_jump_dir
            dir_x = -1
            dir_y = -1
            probe2start_x = 0
            probe2start_y = -1
            ray_x = 2*y_prob + 2
            ray_y = 0
            vx_dir = +1
            vy_dir = -1
        elif corner == 315:
            corner_coef_x = 1 * target_jump_dir
            corner_coef_y = 1 * target_jump_dir
            dir_x = 1
            dir_y = -1
            probe2start_x = 1
            probe2start_y = 0
            ray_x = 0
            ray_y = 2*y_prob + 2
            vx_dir = +1
            vy_dir = +1

        print(f"corner: {corner}\n"
              f"corner_coef_x: {corner_coef_x}\n"
              f"corner_coef_y: {corner_coef_y}\n"
              f"dir_x: {dir_x}\n"
              f"dir_y: {dir_y}\n"
              f"probe2start_x: {probe2start_x}\n"
              f"probe2start_y: {probe2start_y}\n"
              f"ray_x: {ray_x}\n"
              f"ray_y: {ray_y}\n"
              f"vx_dir: {vx_dir}\n"
              f"vy_dir: {vy_dir}\n"
              )


        # timimg in frames
        if ISI >= 0:
            # one count
            t_fixation = 1 * fps
            t_interval_1 = t_fixation + probe_duration
            t_ISI = t_interval_1 + ISI
            t_interval_2 = t_ISI + probe_duration
            # I presume 10000*fps means almost unlimited time to respond?
            t_response = t_interval_2 + 10000*fps
        elif ISI == -1:
            t_fixation = 1 * fps
            t_interval = t_fixation + probe_duration
            t_response = t_interval + 10000*fps


        repeat = True
        while repeat:
            resp = event.BuilderKeyResponse()
            frameN = -1
            continueRoutine = True
            while continueRoutine:
                frameN = frameN + 1

                # ISI YES
                if ISI >= 0:
                    # FIXATION
                    if t_fixation >= frameN > 0:
                        trials_counter.text = f"{total_nTrials}/120"
                        # if background in ['flow', 'static']:
                        if background in ['flow', 'static']:

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

                    # PROBE1
                    if t_interval_1 >= frameN > t_fixation:
                        if background in ['flow', 'static']:
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

                        probeW = probe_s[0]

                        # if we want to center the probe around the meridians
                        # floor((probe_s[-1]+1)/2)
                        center_shift = 0
                        if orientation == 'ray':
                            p1_x = ((x_prob * dir_x + probeW*corner_coef_x) +
                                    center_shift*corner_coef_x*-1) + ray_x
                            p1_y = ((y_prob * dir_y + probeW*corner_coef_y) +
                                    center_shift*corner_coef_y*-1) + ray_y
                            if probe_check == 'yes':
                                probe_test.start = [x_prob * dir_x + ray_x,
                                                    y_prob * dir_y + ray_y]
                                probe_test.end = [(x_prob+1) * dir_x + ray_x,
                                                  (y_prob+1) * dir_y + ray_y]
                        elif orientation == 'tangent':
                            p1_x = (x_prob * dir_x + probeW*corner_coef_x) + \
                                   center_shift*corner_coef_x*-1
                            p1_y = (y_prob * dir_y + probeW*corner_coef_y) + \
                                   center_shift*corner_coef_y*-1
                            if probe_check == 'yes':
                                probe_test.start = [x_prob * dir_x, y_prob * dir_y]
                                probe_test.end = [(x_prob+1) * dir_x, (y_prob+1) * dir_y]
                            
                        
                        # probe 1 for the width
                        probe1.start = [p1_x, p1_y]
                        probe1.end = [p1_x + lineL*dir_x, p1_y + lineL*dir_y]
                        probe1.draw()
                        # probe 2 for the width
                        probe2.start = [p1_x + probe2start_x, p1_y + probe2start_y]
                        probe2.end = [p1_x + probe2start_x + lineL2*dir_x, p1_y + probe2start_y + lineL2*dir_y]
                        probe2.draw()
                        # probe test center
                        if probe_check == 'yes':
                            probe_test.draw()

                    # ISI
                    if t_ISI >= frameN > t_interval_1:
                        if background == 'flow':
                            if orientation == 'tangent':
                                # translational flow
                                x_flow = x_flow + (flow_speed * vx_dir) * flow_dir
                                y_flow = y_flow + (flow_speed * vy_dir) * flow_dir
                                # flow.xys = np.array([x_flow, y_flow]).transpose()
                                # flow.draw()
                            elif orientation == 'ray':
                                # radial flow
                                z = z + flow_speed * flow_dir
                                WrapPoints(z, minDist, maxDist)
                                x_flow = x/z
                                y_flow = y/z
                            flow.xys = np.array([x_flow, y_flow]).transpose()
                            flow.draw()

                        if background == 'static':
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

                    # PROBE2
                    if t_interval_2 >= frameN > t_ISI:
                        if background == 'flow':
                            # flow
                            if orientation == 'tangent':
                                # translational flow
                                x_flow = x_flow + (flow_speed * vx_dir) * flow_dir
                                y_flow = y_flow + (flow_speed * vy_dir) * flow_dir
#                                flow.xys = np.array([x_flow, y_flow]).transpose()
#                                flow.draw()
                            elif orientation == 'ray':
                                # radial flow
                                z = z + flow_speed * flow_dir
                                WrapPoints(z, minDist, maxDist)
                                x_flow = x/z
                                y_flow = y/z

                            flow.xys = np.array([x_flow, y_flow]).transpose()
                            flow.draw()

                        if background == 'static':
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

                        print(f"\nprobeW: {probeW}, probe_s: {probe_s}")
                        probeW = probe_s[1]

                        # floor((probe_s[-1]+1)/2) if we want to center the probe around the meridians
                        center_shift = 0
                        if orientation == 'ray':
                            p1_x = ((x_prob * dir_x + probeW*corner_coef_x) +
                                    center_shift*corner_coef_x*-1) + ray_x
                            p1_y = ((y_prob * dir_y + probeW*corner_coef_y) +
                                    center_shift*corner_coef_y*-1) + ray_y
                            if probe_check == 'yes':
                                probe_test.start = [x_prob * dir_x + ray_x,
                                                    y_prob * dir_y + ray_y]
                                probe_test.end = [(x_prob+1) * dir_x + ray_x,
                                                  (y_prob+1) * dir_y + ray_y]
                        elif orientation == 'tangent':
                            p1_x = (x_prob * dir_x + probeW*corner_coef_x) + \
                                   center_shift*corner_coef_x*-1
                            p1_y = (y_prob * dir_y + probeW*corner_coef_y) + \
                                   center_shift*corner_coef_y*-1
                            if probe_check == 'yes':
                                probe_test.start = [x_prob * dir_x, y_prob * dir_y]
                                probe_test.end = [(x_prob+1) * dir_x, (y_prob+1) * dir_y]

                        probe1.start = [p1_x, p1_y]
                        probe1.end = [p1_x + lineL*dir_x, p1_y + lineL*dir_y]
                        probe1.draw()
                        # probe 2 for the width
                        probe2.start = [p1_x + probe2start_x, p1_y + probe2start_y]
                        probe2.end = [p1_x + probe2start_x + lineL2*dir_x, p1_y + probe2start_y + lineL2*dir_y]
                        probe2.draw()
                        # probe test center
                        if probe_check == 'yes':
                            probe_test.draw()

                    if frameN > t_interval_2:
                        if background in ['flow', 'static']:
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
                        theseKeys = event.getKeys(keyList=['num_5', 'num_4', 'num_1',
                                                           'num_2', 'w', 'q', 'a', 's'])
                        if len(theseKeys) > 0:  # at least one key was pressed
                            resp.keys = theseKeys[-1]  # just the last key pressed
                            resp.rt = resp.clock.getTime()
        
                            if corner == 45:
                                if orientation == 'tangent':
                                    if (resp.keys == 'w') or (resp.keys == 'num_5'):
                                        resp.corr = 1
                                    else:
                                        resp.corr = 0
                                elif orientation == 'ray':
                                    if (resp.keys == 'q') or (resp.keys == 'num_4'):
                                        resp.corr = 1
                                    else:
                                        resp.corr = 0
                                repeat = False
                                continueRoutine = False
                            elif corner == 135:
                                if orientation == 'tangent':
                                    if (resp.keys == 'q') or (resp.keys == 'num_4'):
                                        resp.corr = 1
                                    else:
                                        resp.corr = 0
                                elif orientation == 'ray':
                                    if (resp.keys == 'a') or (resp.keys == 'num_1'):
                                        resp.corr = 1
                                    else:
                                        resp.corr = 0
                                repeat = False
                                continueRoutine = False
                            elif corner == 225:
                                if orientation == 'tangent':
                                    if (resp.keys == 'a') or (resp.keys == 'num_1'):
                                        resp.corr = 1
                                    else:
                                        resp.corr = 0
                                elif orientation == 'ray':
                                    if (resp.keys == 's') or (resp.keys == 'num_2'):
                                        resp.corr = 1
                                    else:
                                        resp.corr = 0
                                repeat = False
                                continueRoutine = False
                            elif corner == 315:
                                if orientation == 'tangent':
                                    if (resp.keys == 's') or (resp.keys == 'num_2'):
                                        resp.corr = 1
                                    else:
                                        resp.corr = 0
                                elif orientation == 'ray':
                                    if (resp.keys == 'w') or (resp.keys == 'num_5'):
                                        resp.corr = 1
                                    else:
                                        resp.corr = 0
                                repeat = False
                                continueRoutine = False

                # NO ISI
                if ISI == -1:
                    if t_fixation >= frameN > 0:
                        trials_counter.text = f"{total_nTrials}/120"

                        if background == 'static':
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
                        
                    if t_interval >= frameN > t_fixation:

                        if background == 'static':
                            # flow
                            flow.xys = np.array([x_flow, y_flow]).transpose()
                            flow.draw()
                            # probe masks
                            probeMask1.draw()
                            probeMask2.draw()
                            probeMask3.draw()
                            probeMask4.draw()


                        # if we want to center the probe around the meridians
                        # floor((probe_s[-1]+1)/2)
                        center_shift = 0

                        probeW = probe_s[0]
                        if orientation == 'ray':
                            p1_x = ((x_prob * dir_x + probeW*corner_coef_x) +
                                    center_shift*corner_coef_x*-1) + ray_x
                            p1_y = ((y_prob * dir_y + probeW*corner_coef_y) +
                                    center_shift*corner_coef_y*-1) + ray_y
                            if probe_check == 'yes':
                                probe_test.start = [x_prob * dir_x + ray_x,
                                                    y_prob * dir_y + ray_y]
                                probe_test.end = [(x_prob+1) * dir_x + ray_x,
                                                  (y_prob+1) * dir_y + ray_y]

                        elif orientation == 'tangent':
                            p1_x = (x_prob * dir_x + probeW*corner_coef_x) + \
                                   center_shift*corner_coef_x*-1
                            p1_y = (y_prob * dir_y + probeW*corner_coef_y) + \
                                   center_shift*corner_coef_y*-1
                            if probe_check == 'yes':
                                probe_test.start = [x_prob * dir_x, y_prob * dir_y]
                                probe_test.end = [(x_prob+1) * dir_x, (y_prob+1) * dir_y]

                        probe1.start = [p1_x, p1_y]
                        probe1.end = [p1_x + lineL*dir_x, p1_y + lineL*dir_y]
                        probe1.draw()
                        # probe 2 for the width
                        probe2.start = [p1_x + probe2start_x, p1_y + probe2start_y]
                        probe2.end = [p1_x + probe2start_x + lineL2*dir_x,
                                      p1_y + probe2start_y + lineL2*dir_y]
                        probe2.draw()
                        # probe test center
                        if probe_check == 'yes':
                            probe_test.draw()
                        
                        probeW = probe_s[1]
                        if orientation == 'ray':
                            p1_x = ((x_prob * dir_x + probeW*corner_coef_x) +
                                    center_shift*corner_coef_x*-1) + ray_x
                            p1_y = ((y_prob * dir_y + probeW*corner_coef_y) +
                                    center_shift*corner_coef_y*-1) + ray_y
                            if probe_check == 'yes':
                                probe_test.start = [x_prob * dir_x + ray_x,
                                                    y_prob * dir_y + ray_y]
                                probe_test.end = [(x_prob+1) * dir_x + ray_x,
                                                  (y_prob+1) * dir_y + ray_y]

                        elif orientation == 'tangent':
                            p1_x = (x_prob * dir_x + probeW*corner_coef_x) + \
                                   center_shift*corner_coef_x*-1
                            p1_y = (y_prob * dir_y + probeW*corner_coef_y) + \
                                   center_shift*corner_coef_y*-1
                            if probe_check == 'yes':
                                probe_test.start = [x_prob * dir_x, y_prob * dir_y]
                                probe_test.end = [(x_prob+1) * dir_x, (y_prob+1) * dir_y]

                        probe1.start = [p1_x, p1_y]
                        probe1.end = [p1_x + lineL*dir_x, p1_y + lineL*dir_y]
                        probe1.draw()
                        # probe 2 for the width
                        probe2.start = [p1_x + probe2start_x, p1_y + probe2start_y]
                        probe2.end = [p1_x + probe2start_x + lineL2*dir_x,
                                      p1_y + probe2start_y + lineL2*dir_y]
                        probe2.draw()
                        # probe test center
                        if probe_check == 'yes':
                            probe_test.draw()
                            
                        dotsMask.draw()
                        trials_counter.draw()

                        fixation.draw()
                            
                    if frameN > t_interval:

                        if background == 'static':
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

                        fixation.setRadius(2)
                        fixation.draw()

                        # ANSWER
                        theseKeys = event.getKeys(keyList=['num_5', 'num_4', 'num_1',
                                                           'num_2', 'w', 'q', 'a', 's'])
                        if len(theseKeys) > 0:  # at least one key was pressed
                            resp.keys = theseKeys[-1]  # just the last key pressed
                            resp.rt = resp.clock.getTime()

                            if corner == 45:
                                if orientation == 'tangent':
                                    if (resp.keys == 'w') or (resp.keys == 'num_5'):
                                        resp.corr = 1
                                    else:
                                        resp.corr = 0
                                elif orientation == 'ray':
                                    if (resp.keys == 'q') or (resp.keys == 'num_4'):
                                        resp.corr = 1
                                    else:
                                        resp.corr = 0
                                repeat = False
                                continueRoutine = False
                            elif corner == 135:
                                if orientation == 'tangent':
                                    if (resp.keys == 'q') or (resp.keys == 'num_4'):
                                        resp.corr = 1
                                    else:
                                        resp.corr = 0
                                elif orientation == 'ray':
                                    if (resp.keys == 'a') or (resp.keys == 'num_1'):
                                        resp.corr = 1
                                    else:
                                        resp.corr = 0
                                repeat = False
                                continueRoutine = False
                            elif corner == 225:
                                if orientation == 'tangent':
                                    if (resp.keys == 'a') or (resp.keys == 'num_1'):
                                        resp.corr = 1
                                    else:
                                        resp.corr = 0
                                elif orientation == 'ray':
                                    if (resp.keys == 's') or (resp.keys == 'num_2'):
                                        resp.corr = 1
                                    else:
                                        resp.corr = 0
                                repeat = False
                                continueRoutine = False
                            elif corner == 315:
                                if orientation == 'tangent':
                                    if (resp.keys == 's') or (resp.keys == 'num_2'):
                                        resp.corr = 1
                                    else:
                                        resp.corr = 0
                                elif orientation == 'ray':
                                    if (resp.keys == 'w') or (resp.keys == 'num_5'):
                                        resp.corr = 1
                                    else:
                                        resp.corr = 0
                                repeat = False
                                continueRoutine = False


                # regardless of timings
                # check for quit (typically the Esc key)
                if event.getKeys(keyList=["escape"]):
                    core.quit()
                # redo the trial if i think i made a mistake
                if event.getKeys(keyList=["r"]):
                    repeat = True
                    continueRoutine = False
                    continue
                # refresh the screen
                if continueRoutine:
                    win.flip()

        # Save to exp dict
        thisExp.addData('stair', stairNum)
        if experiment == 'Ricco_area':
            thisExp.addData('probeW', probeW)
        elif experiment == 'Ricco_separation':
            # positive CW - negative CCW relative to the meridians
            thisExp.addData('separation', sep*target_jump_dir)
        # thisExp.addData('targetPos', target_jump_dir)
        thisExp.addData('flow_dir', flow_dir)
        thisExp.addData('contrast', contrastprobe)
        thisExp.addData('trial_response', resp.corr)
        thisExp.addData('corner', corner)
        thisExp.addData('probe_ecc', probe_ecc)
        thisExp.addData('resp.rt', resp.rt)
        thisExp.addData('total_nTrials', total_nTrials)
        thisExp.addData('orientation', orientation)
        thisExp.nextEntry()

        # so that the staircase adjusts itself
        thisStair.newValue(resp.corr)
