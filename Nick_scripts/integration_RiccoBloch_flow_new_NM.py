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

# sets psychoPy to only log critical messages
logging.console.setLevel(logging.CRITICAL)
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# Monitor config from monitor centre
monitor_name = 'HP_24uh'  # 'NickMac' 'asus_cal' 'Asus_VG24' 'HP_24uh'
# gamma set at 2.1  [####### this comment is incorrect, its set above i think ############]
display_number = 1  # 0 indexed, 1 for external display

# Store info about the experiment session
psychopyVersion = psychopy_version  # '3.1.0'
expName = 'RadialDots'  # from the Builder filename that created this script
expInfo = {'1. Experiment': ['Ricco_separation', 'Ricco_area'],
           '2. Participant': 'testnm',
           '3. Probe duration in frames': '2',
           '4. fps': ['240', '60'],
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

if orientation == 'tangent':
    flow_speed = 7
elif orientation == 'ray':
    flow_speed = 0.2

# todo: not sure what these lines refer to yet...
lineW = 1
lineL = 3
lineL2 = 3  # 2
lineL_test = 1

areas = [19, 13, 7, 4, 2, 1, 2, 3, 4, 5, 7]

separations = [18, 18, 18, 18, 6, 6, 6, 6, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0]

# Where appears probe2 relative to probe1
# Tangent: positive --> CCW; negative --> CW
# Ray: positive --> outward; negative --> inward
PosOrNeg = [-1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, 1]
flow_direction = [-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1]
# flow_direction = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
# flow_direction = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

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
filename = (_thisDir + os.sep + 'integration_RiccoBloch_new' + 
            os.sep + (experiment + '_' + orientation + '_' + expInfo['99. Background'] +
                      '_ISI' + expInfo['7. ISI duration in ms']) +
            '_ECC' + expInfo['6. Probe eccentricity in deg'] +
            os.sep + participant_name +
            os.sep + '%s' % participant_name)

print(f"\nold_filename: {filename}")

# FILENAME
filename = f'{_thisDir}{os.sep}integration_RiccoBloch_new{os.sep}' \
           f'{experiment}_{orientation}_{expInfo["99. Background"]}' \
           f'_ISI{expInfo["7. ISI duration in ms"]}' \
           f'_ECC{expInfo["6. Probe eccentricity in deg"]}' \
           f'{os.sep}{participant_name}' \
           f'{os.sep}{participant_name}'
print(f"new_filename: {filename}")



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
                    units='pix', screen=1, allowGUI=False, fullscr=None)

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
    print(f"Display size does not match expected size from montior centre")
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

# probe
if probe_check == 'yes':
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

# flow
nDots = 10000
flow = visual.ElementArrayStim(win, elementTex=None, elementMask='circle',
                               units='pix', nElements=nDots, sizes=10,
                               colors=[bgcolor[0]-0.3, bgcolor[1], bgcolor[2]-0.3])


# full screen mask to blend off edges and fade to black
# Create a raisedCosine mask array and assign it to a Grating stimulus (grey outside, transparent inside)
# this was useful http://www.cogsci.nl/blog/tutorials/211-a-bit-about-patches-textures-and-masks-in-psychopy

raisedCosTexture2 = visual.filters.makeMask(1080, shape='raisedCosine', fringeWidth=0.6, radius=[1.0, 1.0])
invRaisedCosTexture = -raisedCosTexture2  # inverts mask to blur edges instead of center
blankslab = np.ones((1080, 420))  # create blank slabs to put to left and right of image
mmask = np.append(blankslab, invRaisedCosTexture, axis=1)  # append blank slab to left
mmask = np.append(mmask, blankslab, axis=1)  # and right
dotsMask = visual.GratingStim(win, mask=mmask, tex=None, contrast=1.0, size=(1920, 1080), units='pix', color='black')
# changed dotsmask color from grey
# above fades to black round edges which makes screen edges less visible

# ---------------------------------------------------
# function for wrapping flow dots back into volume
def WrapPoints (ii, imin, imax):
    lessthanmin = (ii < imin)
    ii[lessthanmin] = ii[lessthanmin] + (imax-imin)
    morethanmax = (ii > imax)
    ii[morethanmax] = ii[morethanmax] - (imax-imin)


# Trial counter
# todo: what is this for? the text is '???'
trials_counter = visual.TextStim(win=win, name='instructions', text="???",
                                 font='Arial', pos=[550, 450], height=20, ori=0,
                                 color=[1, 1, 1], colorSpace='rgb', opacity=1,
                                 languageStyle='LTR', depth=0.0)

if expInfo['5. Trials counter'] == 'yes':
    trials_counter.color = [1, 1, 1]
else:
    trials_counter.color = [0, 0, 0]


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
total_nTrials = 0
# old comments for starpoints: [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
# 6 window duration conditions
expInfo['startPoints'] = list(range(1, 23))
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

        trialClock.reset()
        
        # conditions
        # todo: put probeww back in for Ricco_area exp
        # probeWW = areas[thisStair.extraInfo['thisStart']-1] # area experiment
        sep = separations[thisStair.extraInfo['thisStart']-1]  # separation experiment
        target_grows = PosOrNeg[thisStair.extraInfo['thisStart']-1]  # direction in which the probe grows - CW or CCW
        flow_dir = flow_direction[thisStair.extraInfo['thisStart']-1]
        # flow_speed = flow_speed *flow_dir
        
        # flow
        x = np.random.rand(nDots) * taille - taille/2
        y = np.random.rand(nDots) * taille - taille/2
        z = np.random.rand(nDots) * (maxDist - minDist) + minDist
        # z was called z_flow but is actually z position like x and y
        x_flow = x/z
        y_flow = y/z

        # PROBE LOCATION
        corner = random.choice([45, 135, 225, 315])
        x_prob = round((tan(np.deg2rad(probe_ecc))*viewdistPix)/sqrt(2))
        y_prob = round((tan(np.deg2rad(probe_ecc))*viewdistPix)/sqrt(2))
        
        probeMask1.setPos([x_prob+1, y_prob+1])
        probeMask2.setPos([-x_prob-1, y_prob+1])
        probeMask3.setPos([-x_prob-1, -y_prob-1])
        probeMask4.setPos([x_prob+1, -y_prob-1])
        
        if experiment == 'Ricco_area':
            # list of all positions between 0 and probeW
            # todo: put probeww back in for Ricco_area exp
            probe_s = list(range(0, probeWW))
        elif experiment == 'Ricco_separation':
            # list of 2 positions: position probe1 and position probe 2
            probe_s = [0, sep]
            
        if corner == 45:
            corner_coef_x = -1 * target_grows  # x direction in which grows the probe
            corner_coef_y = 1 * target_grows  # y direction in which grows the probe
            dir_x = 1  # x line direction
            dir_y = 1  # y line direction
            probe2start_x = 0  # x where starts the second line of the probe
            probe2start_y = 1  # y where starts the second line of the probe
            ray_x = -2*x_prob - 2
            ray_y = 0
            vx_dir = -1  # CCW
            vy_dir = +1
        elif corner == 135:
            corner_coef_x = -1 * target_grows
            corner_coef_y = -1 * target_grows
            dir_x = -1
            dir_y = 1
            probe2start_x = -1
            probe2start_y = 0
            ray_x = 0
            ray_y = -2*y_prob - 2
            vx_dir = -1  # CCW
            vy_dir = -1
        elif corner == 225:
            corner_coef_x = 1 * target_grows
            corner_coef_y = -1 * target_grows
            dir_x = -1
            dir_y = -1
            probe2start_x = 0
            probe2start_y = -1
            ray_x = 2*y_prob + 2
            ray_y = 0
            vx_dir = +1  # CCW
            vy_dir = -1
        elif corner == 315:
            corner_coef_x = 1 * target_grows
            corner_coef_y = 1 * target_grows
            dir_x = 1
            dir_y = -1
            probe2start_x = 1
            probe2start_y = 0
            ray_x = 0
            ray_y = 2*y_prob + 2
            vx_dir = +1  # CCW
            vy_dir = +1

        # timimg in frames
        if ISI >= 0:
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
                    if t_fixation >= frameN > 0:
                        trials_counter.text = "%i /120" % total_nTrials
                        # trials_counter.text = f"{total_nTrials}/120"

                        trials_counter.draw()
                        if expInfo['99. Background'] == 'flow' or expInfo['99. Background'] == 'static':
                            # flow
                            flow.xys = np.array([x_flow, y_flow]).transpose()
                            flow.draw()
                            # probe masks
                            probeMask1.draw()
                            probeMask2.draw()
                            probeMask3.draw()
                            probeMask4.draw()

                        dotsMask.draw()
                        fixation.setRadius(3)
                        fixation.draw()

                    if t_interval_1 >= frameN > t_fixation:
                        trials_counter.draw()
                        if expInfo['99. Background'] == 'flow' or expInfo['99. Background'] == 'static':
                            # flow
                            flow.xys = np.array([x_flow, y_flow]).transpose()
                            flow.draw()
                            # probe masks
                            probeMask1.draw()
                            probeMask2.draw()
                            probeMask3.draw()
                            probeMask4.draw()

                        dotsMask.draw()
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

                    if t_ISI >= frameN > t_interval_1:
                        trials_counter.draw()
                        if expInfo['99. Background'] == 'flow':
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
                            # todo: these were commented out - is that right?
#                           # probe masks
#                            probeMask1.draw()
#                            probeMask2.draw()
#                            probeMask3.draw()
#                            probeMask4.draw()
                        if expInfo['99. Background'] == 'static':
                            # flow
                            flow.xys = np.array([x_flow, y_flow]).transpose()
                            flow.draw()
                        # probe masks
                        probeMask1.draw()
                        probeMask2.draw()
                        probeMask3.draw()
                        probeMask4.draw()
                            
                        dotsMask.draw()
                        fixation.setRadius(3)
                        fixation.draw()                       
                        
                    if t_interval_2 >= frameN > t_ISI:
                        trials_counter.draw()
                        if expInfo['99. Background'] == 'flow':
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

                        if expInfo['99. Background'] == 'static':
                            # flow
                            flow.xys = np.array([x_flow, y_flow]).transpose()
                            flow.draw()

                        # probe masks
                        probeMask1.draw()
                        probeMask2.draw()
                        probeMask3.draw()
                        probeMask4.draw()
                        dotsMask.draw()
                        fixation.setRadius(3)
                        fixation.draw()                       

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
                        trials_counter.draw()
                        if expInfo['99. Background'] == 'flow' or expInfo['99. Background'] == 'static':
                            # flow
                            flow.draw()
                            # probe masks
                            probeMask1.draw()
                            probeMask2.draw()
                            probeMask3.draw()
                            probeMask4.draw()

                        dotsMask.draw()
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
                        # trials_counter.text = "%i /120" %total_nTrials
                        trials_counter.text = f"{total_nTrials}/120"
                        trials_counter.draw()

                        if expInfo['99. Background'] == 'static':
                            # flow
                            flow.xys = np.array([x_flow, y_flow]).transpose()
                            flow.draw()
                            # probe masks
                            probeMask1.draw()
                            probeMask2.draw()
                            probeMask3.draw()
                            probeMask4.draw()
                        dotsMask.draw()
                        fixation.setRadius(3)
                        fixation.draw()
                        
                    if t_interval >= frameN > t_fixation:
                        trials_counter.draw()

                        if expInfo['99. Background'] == 'static':
                            # flow
                            flow.xys = np.array([x_flow, y_flow]).transpose()
                            flow.draw()
                            # probe masks
                            probeMask1.draw()
                            probeMask2.draw()
                            probeMask3.draw()
                            probeMask4.draw()
#                        dotsMask.draw()
#                        fixation.draw()
# this is done below and its just creating a flicker so remove

                        # if we want to center the probe around the meridians
                        # floor((probe_s[-1]+1)/2)
                        center_shift = 0

# ######  this is done above, its just repeating..
#                        if expInfo['99. Background'] == 'static':
#                            # flow
#                            flow.xys = np.array([x_flow, y_flow]).transpose()
#                            flow.draw()
#                            # probe masks
#                            probeMask1.draw()
#                            probeMask2.draw()
#                            probeMask3.draw()
#                            probeMask4.draw()
#
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
                        fixation.draw()
                            
                    if frameN > t_interval:
                        trials_counter.draw()
                        
                        if expInfo['99. Background'] == 'static':
                            # flow
                            flow.xys = np.array([x_flow, y_flow]).transpose()
                            flow.draw()
                            # probe masks
                            probeMask1.draw()
                            probeMask2.draw()
                            probeMask3.draw()
                            probeMask4.draw()
                        
                        dotsMask.draw()
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

        stairNum = thisStair.extraInfo['thisStart']

        thisExp.addData('stair', stairNum)
        if experiment == 'Ricco_area':
            thisExp.addData('probeW', probeW)
        elif experiment == 'Ricco_separation':
            # positive CW - negative CCW relative to the meridians
            thisExp.addData('separation', sep*target_grows)
        # thisExp.addData('targetPos', target_grows)
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

