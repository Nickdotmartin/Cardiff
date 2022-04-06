from __future__ import absolute_import, division
from psychopy import locale_setup, sound, gui, visual, core, data, event, logging, clock, monitors, prefs
from psychopy.visual import ShapeStim, Polygon, EnvelopeGrating, Circle
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)
import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (arcsin, arccos, arctan, sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle
import os  # handy system and path functions
import sys  # to get file system encoding
import numpy
import random
import copy, time  # from the std python libs

# from kestenST import Staircase
from kestenSTmaxVal import Staircase

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v3.0.7),
    on avril 24, 2019, at 15:45
If you publish work using this script please cite the PsychoPy publications:
    Peirce, JW (2007) PsychoPy - Psychophysics software in Python.
        Journal of Neuroscience Methods, 162(1-2), 8-13.
    Peirce, JW (2009) Generating stimuli for neuroscience using PsychoPy.
        Frontiers in Neuroinformatics, 2:10. doi: 10.3389/neuro.11.010.2008
"""



#logging.console.setLevel(logging.DEBUG)
logging.console.setLevel(logging.CRITICAL)
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)
# Store info about the experiment session
psychopyVersion = '3.1.0'
expName = 'FlowParsing'  # from the Builder filename that created this script
expInfo = {'1. participant': 'Nick_test',
           '2. Repetitions': '20',
           '3. Areas(2 or 4)': '4',
           '4. Stim duration in frames':['18', '3', '6', '12', '18', '24', '30', '36'],
           '5. Surrounding motion': ['jitter', 'static'],
           '6. Dots life in frames': '100'}

# GUI
dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
if dlg.OK == False:
    core.quit()  # user pressed cancel

# GUI SETTINGS
participant = (expInfo['1. participant'])
n_trials_per_stair = int((expInfo['2. Repetitions']))

nAFC = float(expInfo['3. Areas(2 or 4)'])

lifeDots = int(expInfo['6. Dots life in frames'])

stimDur = int(expInfo['4. Stim duration in frames'])

stim_dur_txt = f'stimDur{stimDur}-'

print(f'stim_dur_txt: {stim_dur_txt}')
if expInfo['5. Surrounding motion'] == 'jitter':
    jitter = 1
elif expInfo['5. Surrounding motion'] == 'static':
    jitter = 0

# MONITOR SPEC
widthPix = 1920 # screen width in px 
heightPix = 1080 # screen height in px
monitorwidth = 59.77 # monitor width in cm
viewdist = 57.3 # viewing distance in cm
monitorname = 'HP_24uh'
scrn = 1
mon = monitors.Monitor(monitorname, width=monitorwidth, distance=viewdist)
mon.setSizePix((widthPix, heightPix))
mon.save()

# DISPLAY
win = visual.Window(monitor=mon, size=(widthPix,heightPix), colorSpace='rgb', units=None,
    screen=scrn,
    allowGUI=False,
    fullscr=None)

# INFORMATIONS
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName
expInfo['psychopyVersion'] = psychopyVersion
frameRate = expInfo['frameRate'] = win.getActualFrameRate()
frameRate2 = 240

# PARAMETERS
sizeDots = 4
nDots = 1650
probeSpeed = 0

# ELEMENTS COLORS
fixationColor = 'red'
dotsColor = (191/255, 0/255, 64/255)
probeColor = 'green'
dotsMaskColor = 'black'
aperturesColor = 'grey'

# FILENAME
#filename = _thisDir + os.sep + 'data' + os.sep + participant + os.sep + '%s' % (participant)
filename = _thisDir + os.sep + 'from_Martin_FlowParsing' + os.sep + participant + os.sep + '%s' % (stim_dur_txt+participant)
# Experiment Handler
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=None, runtimeInfo=None,
    savePickle=True, saveWideText=True,
    dataFileName=filename)

endExpNow = False  # flag for 'escape' or other condition => quit the exp

# INSTRUCTIONS
instructions = visual.TextStim(win=win, name='instructions',
    text="Press [i] if you see the probe moving inward, and [o] if you see the probe moving outward \.\n\n[Space bar] to start",
    font='Arial', pos=[0, 0], height=0.1, wrapWidth=None, ori=0, color=[1, 1, 1], colorSpace='rgb', opacity=1, languageStyle='LTR', depth=0.0);

# CLOCK
trialClock = core.Clock()

# ELEMENTS
fixation = Circle(win, radius=3, units='pix', lineColor='red', fillColor='red')

dots = visual.ElementArrayStim(win, elementTex=None, units='pix', 
                               nElements=nDots, sizes=sizeDots*4, colors=(1, 0, 1), 
                               elementMask='circle')

raisedCosTexture0 = visual.filters.makeMask(256, shape='raisedCosine', 
                                            fringeWidth=0.8, radius=[1.0, 1.0])
probe = visual.GratingStim(win, mask=raisedCosTexture0, tex=None, contrast=0.5, 
                           size=(30, 30), units='pix', color='green', pos=[-200, 100])

# full screen mask to blend off edges and fade to black
# Create a raisedCosine mask array and assign it to a Grating stimulus (grey outside, transparent inside)
# this was useful http://www.cogsci.nl/blog/tutorials/211-a-bit-about-patches-textures-and-masks-in-psychopy
raisedCosTexture = visual.filters.makeMask(1080, shape= 'raisedCosine', fringeWidth= 0.6, radius = [1.0, 1.0])
invRaisedCosTexture = -raisedCosTexture # inverts mask to blur edges instead of center
blankslab=np.ones((1080, 420))  # create blank slabs to put to left and right of image
mmask=np.append(blankslab, invRaisedCosTexture, axis=1)  # append blank slab to left
mmask=np.append(mmask, blankslab, axis=1) # and right
dotsMask = visual.GratingStim(win, mask=mmask, tex=None, contrast= 1.0, size=(1920, 1080), units='pix', color=dotsMaskColor)

# mask for the 4 areas
raisedCosTexture = visual.filters.makeMask(256, shape='raisedCosine', fringeWidth=0.3, radius=[1.0, 1.0])
aperture = 130
probeMask1 = visual.GratingStim(win, mask = raisedCosTexture, tex=None, contrast= 1.0, size=(aperture*2, aperture*2), units='pix', color = aperturesColor)
probeMask2 = visual.GratingStim(win, mask = raisedCosTexture, tex=None, contrast= 1.0, size=(aperture*2, aperture*2), units='pix', color = aperturesColor)
probeMask3 = visual.GratingStim(win, mask = raisedCosTexture, tex=None, contrast= 1.0, size=(aperture*2, aperture*2), units='pix', color = aperturesColor)
probeMask4 = visual.GratingStim(win, mask = raisedCosTexture, tex=None, contrast= 1.0, size=(aperture*2, aperture*2), units='pix', color = aperturesColor)

probeMask1.setPos([0, 200])
probeMask2.setPos([0, -200])
probeMask3.setPos([200, 0])
probeMask4.setPos([-200, 0])
# sound
tock1 = sound.Sound('400', secs=0.15, sampleRate=44100, stereo=True)
tock2 = sound.Sound('800', secs=0.15, sampleRate=44100, stereo=True)

# ---------------------------------------------------
# function for wrapping flow dots back into volume
def WrapPoints (ii, imin, imax):
    lessthanmin = (ii < imin)
    ii[lessthanmin] = ii[lessthanmin] + (imax-imin)
    morethanmax = (ii > imax)
    ii[morethanmax] = ii[morethanmax] - (imax-imin)

# mouse
myMouse = event.Mouse(visible=False)  #  will use win by default

# Create some handy timers
routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine 

# ------Prepare to start Routine "instr"-------
continueRoutine = True
# -------Start Routine "instr"-------
while continueRoutine == True:
    
    instructions.setAutoDraw(True)
    theseKeys = event.getKeys()
    
    # check for quit:
    if "escape" in theseKeys:
        endExpNow = True
    if endExpNow:
        core.quit()
    if "space" in theseKeys:
        instructions.setAutoDraw(False)
        continueRoutine = False
    if continueRoutine:
        win.flip()
thisExp.nextEntry()

# ---------------------------------------------------------------Prepare to start Staircase "trials" --------
# ----------------- STAIRCASE

probeSpd = 3  # starting value
trial_number = 0
expInfo['stair_list'] = [1, 2, 0]  # 6 window duration conditions
expInfo['n_trials_per_stair'] = n_trials_per_stair

stairStart = probeSpd
stairs = [] 
for stair_idx in expInfo['stair_list']:
    thisInfo = copy.copy(expInfo)
    thisInfo['stair_idx'] = stair_idx #we might want to keep track of this

    thisStair = Staircase(name='trials',
        type='simple',
        value=stairStart,
        C=1.8,  # typically 60% of reference stimulus
        minRevs=3,
        minTrials=n_trials_per_stair,
        minVal=-10,
        targetThresh=0.5,
        extraInfo=thisInfo
        )
    stairs.append(thisStair)

# run the experiment
nDone = 0
for step in range(n_trials_per_stair):
    shuffle(stairs) #this shuffles 'in place' (ie stairs itself is changed, nothing returned)
    #then loop through our randomised order of staircases for this repeat
    for thisStair in stairs:

        trial_number = trial_number + 1
        probeSpeed = thisStair.next()
        contrastprobe = 0.5
        probe.contrast = contrastprobe  # this is what a staircase varies

        # window duration conditions
        if thisStair.extraInfo['stair_idx'] == 1:
            flowDir = 1
            condition = 'expansion'
            probeDir = -1
            probeSpeed = probeSpeed*probeDir
        elif thisStair.extraInfo['stair_idx'] == 2:
            flowDir = -1
            condition = 'contraction'
            probeDir = 1
            probeSpeed = probeSpeed*probeDir
        elif thisStair.extraInfo['stair_idx'] == 0:
            flowDir = random.choice([1,-1])
            condition = 'jitter'
            probeDir = random.choice([1,-1])
            probeSpeed = probeSpeed*probeDir

        # probe and flow speed
        radSpeed = -0.02

        # 4 or 2 areas
        if nAFC == 4:
            side = random.choice([1,2,3,4]) #right left up down
        else:
            side = random.choice([1,2]) #right left

        # setting x and y positions depending on the side
        if side == 1:
            x_position = 200
            y_position = 0
        elif side == 2:
            x_position = -200
            y_position = 0
        elif side == 3:
            x_position = 0
            y_position = 200
        elif side == 4:
            x_position = 0
            y_position = -200

        # probe position reset
        probe_x = 0
        probe_y = 0
        
        # DOTS
        minDist = 0.5
        maxDist = 5
        taille = 5000
        DotLife = lifeDots
        
        # a first stim_dur_txt of points
        x = numpy.random.rand(nDots) * taille - taille/2 #screenH - screenH/2
        y = numpy.random.rand(nDots) * taille - taille/2 #screenH - screenH/2
        z = numpy.random.rand(nDots) * (maxDist - minDist) + minDist
        L = numpy.mod(numpy.linspace(1, nDots, nDots), DotLife)
        screen_x = x/z;            screen_y = y/z

        # a second stim_dur_txt of points which we will use for the random motion direction flow
        x2 = numpy.random.rand(nDots) * taille - taille/2 #screenH - screenH/2
        y2 = numpy.random.rand(nDots) * taille - taille/2 #screenH - screenH/2
        screen_x2 = x2/z;          screen_y2 = y2/z

        # timimg in frames
        t_fixation = 1 * frameRate2  # 240 frames for fixation, e.g., 1 second.
        t_inducer = t_fixation + stimDur  # bg_motion prior to probe for stimDur, e.g., 18 frames or 75ms
        t_static = t_inducer + 1 * frameRate2  # probes appear during 240 frames, e.g., 1 second.
        t_response = t_static + 100*frameRate2  # 100 seconds to respond.
        
        # sounds triggers
        soundTrig1 = 1
        soundTrig2 = 1
        
        # count frames
        nb_frames_motion = 0

        # ------Prepare to start Routine "trial"-------
        t = 0
        trialClock.reset()  # clock
        frameN = -1
        continueRoutine = True
        routineTimer.add(t_response)

        resp = event.BuilderKeyResponse()
        # keep track of which components have finished
        trialComponents = [fixation, dots]
        for thisComponent in trialComponents:
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED

        # -------Start Routine "trial"-------
        while continueRoutine and routineTimer.getTime() > 0:
            # get current time
            t = trialClock.getTime()
            t_frame = t * frameRate2
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
    
# -------------------------------*jitter* updates------------------------------------------------
# -------------------------------*jitter* updates------------------------------------------------
            if t_fixation >= t_frame > 0:

                if jitter == 1:
                    #save the prev screen positions before calculate new ones
                    prev_screen_x = screen_x;     prev_screen_y = screen_y
                    prev_screen_x2 = screen_x2;   prev_screen_y2 = screen_y2
    
                    # move half of the flow dots forwards and half backwards
                    # the flow dots and the shadow flow dots share the same z coord, they
                    # just differ in x,y so no z2 coord
                    z[0:len(z)//2] = z[0:len(z)//2] + radSpeed
                    z[len(z)//2:len(z)] = z[len(z)//2:len(z)] - radSpeed
    
                    # check all flow dots are still within the volume, if not wrap them
                    WrapPoints (z, minDist, maxDist)
                    WrapPoints (x, -taille/2, taille/2)
                    WrapPoints (y, -taille/2, taille/2)
                    screen_x = x/z;            screen_y = y/z
                    delta_screen_x = screen_x - prev_screen_x
                    delta_screen_y = screen_y - prev_screen_y
    
                    # [ no need to wrap z2 as use z ]
                    WrapPoints (x2, -taille/2, taille/2)
                    WrapPoints (y2, -taille/2, taille/2)
                    screen_x2 = x2/z;          screen_y2 = y2/z  # use z not z2
                    delta_screen_x2 = screen_x2 - prev_screen_x2
                    delta_screen_y2 = screen_y2 - prev_screen_y2
    
                    # calc add delta from (x2,y2,z) for jittered flow vectors
                    # for none jittered just add delta from (x,y,z)
                    jittered_screen_x = prev_screen_x+delta_screen_x2
                    jittered_screen_y = prev_screen_y+delta_screen_y2
    
                    # update x,y positions to account for jittered movement
                    x=jittered_screen_x * z
                    y=jittered_screen_y * z
    
                    # replace dots that reached the end of their life
                    # use a trick, just switch to other side of screen
                    L = L - 1
                    dead=numpy.where(L==0)
                    x[dead] = -x[dead]
                    prev_screen_x = -prev_screen_x;     
                    x2[dead] = -x2[dead]
                    prev_screen_x2 = - prev_screen_x2
                    L[dead] = DotLife
                    
                    # draw dots
                    dots.xys = numpy.array([jittered_screen_x, jittered_screen_y]).transpose()
                    dots.draw()
                
                elif jitter == 0:
                    # draw dots
                    dots.xys = numpy.array([screen_x, screen_y]).transpose()
                    dots.draw()
                    
                # stim_dur_txt position and draw each areas
                if nAFC == 4:
                    probeMask1.draw()
                    probeMask2.draw()
                    probeMask3.draw()
                    probeMask4.draw()
                elif nAFC == 2:
                    probeMask1.draw()
                    probeMask2.draw()

                fixation.draw()
                dotsMask.draw()
                
                soundTrig = 1
                
# -------------------------------*motion_condition1* updates [jitter, exp, cont]------------------------------------------------
# -------------------------------*motion_condition1* updates [jitter, exp, cont]------------------------------------------------
            if t_inducer >= t_frame > t_fixation:
                
                if soundTrig1 == 1:
                    tock1.play()
                    soundTrig1 = 0

                #save the prev screen positions before calculate new ones
                prev_screen_x = screen_x;     prev_screen_y = screen_y
                prev_screen_x2 = screen_x2;   prev_screen_y2 = screen_y2

                if thisStair.extraInfo['stair_idx']==0:
                    # move half of the flow dots forwards and half backwards
                    # the flow dots and the shadow flow dots share the same z coord, they
                    # just differ in x,y so no z2 coord
                    z[0:len(z)//2] = z[0:len(z)//2] + radSpeed
                    z[len(z)//2:len(z)] = z[len(z)//2:len(z)] - radSpeed
                elif thisStair.extraInfo['stair_idx']==1:
                    z = z + radSpeed
                elif thisStair.extraInfo['stair_idx']==2:
                    z = z - radSpeed
    
                # check all flow dots are still within the  volume, if not wrap them
                WrapPoints (z, minDist, maxDist)
                WrapPoints (x, -taille/2, taille/2)
                WrapPoints (y, -taille/2, taille/2)
                screen_x = x/z;            screen_y = y/z
                delta_screen_x = screen_x - prev_screen_x
                delta_screen_y = screen_y - prev_screen_y
    
                # [ no need to wrap z2 as use z ]
                WrapPoints (x2, -taille/2, taille/2)
                WrapPoints (y2, -taille/2, taille/2)
                screen_x2 = x2/z;          screen_y2 = y2/z  # use z not z2
                delta_screen_x2 = screen_x2 - prev_screen_x2
                delta_screen_y2 = screen_y2 - prev_screen_y2

                if thisStair.extraInfo['stair_idx'] == 0 :
                    # calc add delta from (x2,y2,z) for jittered flow vectors
                    jittered_screen_x = prev_screen_x+delta_screen_x2
                    jittered_screen_y = prev_screen_y+delta_screen_y2
                else :
                    # for none jittered just add delta from (x,y,z)
                    jittered_screen_x = prev_screen_x+delta_screen_x
                    jittered_screen_y = prev_screen_y+delta_screen_y

                # update x,y positions to account for jittered movement
                x=jittered_screen_x * z
                y=jittered_screen_y * z            

                # replace dots that reached the end of their life
                # use a trick, just switch to other side of screen
                L = L - 1
                dead=numpy.where(L==0)
                x[dead] = -x[dead]
                prev_screen_x = -prev_screen_x;     
                x2[dead] = -x2[dead]
                prev_screen_x2 = - prev_screen_x2
                L[dead] = DotLife

                # draw dots
                dots.xys = numpy.array([jittered_screen_x, jittered_screen_y]).transpose()
                dots.draw()

                # stim_dur_txt position and draw each areas
                if nAFC == 4:
                    probeMask1.draw()
                    probeMask2.draw()
                    probeMask3.draw()
                    probeMask4.draw()
                elif nAFC == 2:
                    probeMask1.draw()
                    probeMask2.draw()

                # todo: I think I am upto here - figuring out the probe motion.
                # draw probe if 1st interval
                if side == 1:
                    probe_y = probe_y + 0
                    probe_x = probe_x - probeSpeed
                    probe.setPos([x_position + probe_x, y_position + probe_y])
                elif side == 2:
                    probe_y = probe_y + 0
                    probe_x = probe_x + probeSpeed
                    probe.setPos([x_position + probe_x, y_position + probe_y])
                elif side == 3:
                    probe_y = probe_y - probeSpeed
                    probe_x = probe_x
                    probe.setPos([x_position + probe_x, y_position + probe_y])
                elif side == 4:
                    probe_y = probe_y + probeSpeed
                    probe_x = probe_x
                    probe.setPos([x_position + probe_x, y_position + probe_y])
                probe.draw()

                nb_frames_motion = nb_frames_motion + 1
                
                dotsMask.draw()
                fixation.draw()
                
                printTrig1 = 1
                

# -------------------------------*resp* updates------------------------------------------------
# -------------------------------*resp* updates------------------------------------------------

            if t_static >= t_frame > t_inducer:

                if jitter == 1:
                    #save the prev screen positions before calculate new ones
                    prev_screen_x = screen_x;     prev_screen_y = screen_y
                    prev_screen_x2 = screen_x2;   prev_screen_y2 = screen_y2
    
                    # move half of the flow dots forwards and half backwards
                    # the flow dots and the shadow flow dots share the same z coord, they
                    # just differ in x,y so no z2 coord
                    z[0:len(z)//2] = z[0:len(z)//2] + radSpeed
                    z[len(z)//2:len(z)] = z[len(z)//2:len(z)] - radSpeed
    
                    # check all flow dots are still within the  volume, if not wrap them
                    WrapPoints (z, minDist, maxDist)
                    WrapPoints (x, -taille/2, taille/2)
                    WrapPoints (y, -taille/2, taille/2)
                    screen_x = x/z;            screen_y = y/z
                    delta_screen_x = screen_x - prev_screen_x
                    delta_screen_y = screen_y - prev_screen_y
    
                    # [ no need to wrap z2 as use z ]
                    WrapPoints (x2, -taille/2, taille/2)
                    WrapPoints (y2, -taille/2, taille/2)
                    screen_x2 = x2/z;          screen_y2 = y2/z  # use z not z2
                    delta_screen_x2 = screen_x2 - prev_screen_x2
                    delta_screen_y2 = screen_y2 - prev_screen_y2
    
                    # calc add delta from (x2,y2,z) for jittered flow vectors
                    # for none jittered just add delta from (x,y,z)
                    jittered_screen_x = prev_screen_x+delta_screen_x2
                    jittered_screen_y = prev_screen_y+delta_screen_y2
    
                    # update x,y positions to account for jittered movement
                    x=jittered_screen_x * z
                    y=jittered_screen_y * z
    
                    # replace dots that reached the end of their life
                    # use a trick, just switch to other side of screen
                    L = L - 1
                    dead=numpy.where(L==0)
                    x[dead] = -x[dead]
                    prev_screen_x = -prev_screen_x;     
                    x2[dead] = -x2[dead]
                    prev_screen_x2 = - prev_screen_x2
                    L[dead] = DotLife
    
                    # draw dots
                    dots.xys = numpy.array([jittered_screen_x, jittered_screen_y]).transpose()
                    dots.draw()
                    
                elif jitter == 0:
                    # draw dots
                    dots.xys = numpy.array([screen_x, screen_y]).transpose()
                    dots.draw()

                # stim_dur_txt position and draw each areas
                if nAFC == 4:
                    probeMask1.draw()
                    probeMask2.draw()
                    probeMask3.draw()
                    probeMask4.draw()
                elif nAFC == 2:
                    probeMask1.draw()
                    probeMask2.draw()

                fixation.draw()
                dotsMask.draw()
                
# -------------------------------*resp* updates------------------------------------------------
# -------------------------------*resp* updates------------------------------------------------
            if t_response >= t_frame > t_static:

                tock1.stop()
                tock2.stop()

                dots.draw()

                # draw areas
                if nAFC == 4:
                    probeMask1.draw()
                    probeMask2.draw()
                    probeMask3.draw()
                    probeMask4.draw()
                elif nAFC == 2:
                    probeMask1.draw()
                    probeMask2.draw()

                fixation.draw()
                dotsMask.draw()

                # ANSWER
                theseKeys = event.getKeys(keyList=['i', 'o'])
                # check for quit:
                if "escape" in theseKeys:
                    endExpNow = True
                if len(theseKeys) > 0:  # at least one key was pressed
                    resp.keys = theseKeys[-1]  # just the last key pressed
                    resp.rt = resp.clock.getTime()

                    if probeDir == 1:
                        if (resp.keys == str('i')) or (resp.keys == 'i'):
                            resp.corr = 1
                            answer = 'i'
                        else:
                            resp.corr = 0
                            answer = 'o'
                        # a response ends the routine
                        continueRoutine = False
                    elif probeDir == -1:
                        if (resp.keys == str('i')) or (resp.keys == 'i'):
                            resp.corr = 0
                            answer = 'i'
                        else:
                            resp.corr = 1
                            answer = 'o'
                        # a response ends the routine
                        continueRoutine = False

            # check for quit (typically the Esc key)
            if endExpNow or event.getKeys(keyList=["escape"]):
                core.quit()

            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trialComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished

            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()

        # -------Ending Routine "trial"-------
        for thisComponent in trialComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # check responses
        if resp.keys in ['', [], None]:  # No response was made
            resp.keys=None
            # was no response the correct answer?!
            if str('w').lower() == 'none':
               resp.corr = 1;  # correct non-response
            else:
               resp.corr = 0;  # failed to respond (incorrectly)

        stairNum = thisStair.extraInfo['stair_idx']

        congruent = flowDir*probeDir

        thisExp.addData('trial_number', trial_number)
        thisExp.addData('stair', stairNum)
        thisExp.addData('step', step)
        thisExp.addData('condition', condition)
        thisExp.addData('radSpeed', radSpeed)
        thisExp.addData('stimDuration', stimDur)
        thisExp.addData('contrast', contrastprobe)
        thisExp.addData('probeSpeed', probeSpeed)
        thisExp.addData('flowDir', flowDir)
        thisExp.addData('probeDir', probeDir)
        thisExp.addData('answer', answer)
        thisExp.addData('congruent', congruent)
        thisExp.addData('resp.corr', resp.corr)
        thisExp.addData('resp.rt', resp.rt)
        thisExp.nextEntry()

        thisStair.newValue(resp.corr) #so that the staircase adjusts itself