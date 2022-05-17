from __future__ import division
from psychopy import gui, visual, core, data, event, logging, monitors
from psychopy import __version__ as psychopy_version
# import psychopy
import os
from numpy import deg2rad
from numpy.random import shuffle
import random
import copy
from datetime import datetime
from math import *

from PsychoPy_tools import check_correct_monitor
from kestenSTmaxVal import Staircase


def getResponse(question):
    """
    I wrote this function to avoid any incorrect responses being inputted.
    Will accept 1, 2 or 9 in response to any question.  If a key other than
    these is entered, it will keep asking until it gets a valid response.

    :param question:  Any string, which will appear on screen
    :return: a valid response (1, 2 or 9)
    """
    respond_1_2_or_9 = input(question)
    if respond_1_2_or_9 == "1" or respond_1_2_or_9 == "2" or respond_1_2_or_9 == "9":
        return int(respond_1_2_or_9)  # converted to int
    else:
        return getResponse(f'Try-again: {question}')


# sets psychoPy to only log critical messages
# logging.console.setLevel(logging.CRITICAL)

# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# Monitor config from monitor centre
monitor_name = 'Asus_VG24'  # 'NickMac' 'asus_cal' 'Asus_VG24' 'HP_24uh' 'ASUS_2_13_240Hz' 'Iiyama_2_18'
# gamma set at 2.1  [####### this comment is incorrect, its set above i think ############]
display_number = 1  # 0 indexed, 1 for external display

'''
Designed ased on the Monitor Response Times assesment Martin sent me.
ISIs: conc, 0, 2, 4, 12, 24 frames
ms:   conc, 0, 8, 16, 50, 100 ms

Sep: 0, 1, 3, 18

All stim at max luminance, recordings for 45 frames/187.5ms

'''

# Store info about the experiment session
expName = 'monitor_resp_times_stim'  # from the Builder filename that created this script

# todo: change dict keys (ISI str to int?) once I've looked at analysis
expInfo = {'1. Participant': 'mon_resp_time_stim',
           '2. Probe duration in frames at 240hz': 2,
           '3. fps': [60, 144, 240],
           # '4. ISI duration in frame': [0, 2, 4, 6, 9, 12, 24, -1],
           '5. Probe orientation': ['tangent'],
           '6. Probe size': ['5pixels', '6pixels', '3pixels'],
           '7. Background lum in percent of maxLum': 20,
           '8. Red filter': ['no', 'yes']
           }


# GUI
dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
if not dlg.OK:
    core.quit()  # user pressed cancel

expInfo['time'] = datetime.now().strftime("%H:%M:%S")
expInfo['date'] = datetime.now().strftime("%d/%m/%Y")

# GUI SETTINGS
participant_name = expInfo['1. Participant']
trial_number = 25
probe_duration = int(expInfo['2. Probe duration in frames at 240hz'])
probe_ecc = 4
fps = int(expInfo['3. fps'])
orientation = expInfo['5. Probe orientation']
# ISI durations, -1 correspond to simultaneous probes
# todo: this will vary - e.g., test several ISIs
# ISI = int(expInfo['4. ISI duration in frame'])


# VARIABLES
# Distances between probes
# 99 values for single probe condition
# separations = [18, 18, 6, 6, 3, 3, 2, 2, 1, 1, 0, 0, 99, 99]
sep_vals = [0, 1, 3, 18]
ISI_vals = [-1, 0, 2, 4, 12, 24]

ISI_list = [-1, 0, 2, 4, 12, 24, -1, 0, 2, 4, 12, 24, -1, 0, 2, 4, 12, 24, -1, 0, 2, 4, 12, 24]
sep_list = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 18, 18, 18, 18, 18, 18]
stairs = list(range(len(sep_list)))

# for idx, (this_stair, isi, sep) in enumerate(zip(stairs, ISI_list, sep_list)):
#     print(idx, this_stair, isi, sep)
#
cond_dict = {i: {'sep': sep, 'isi': isi} for (i, sep, isi) in zip(stairs, sep_list, ISI_list)}
# print(cond_dict)
#
# for this_cond, cond_vals in cond_dict.items():
#     print(this_cond, cond_vals)
#     print(this_cond['sep'])


# # FILENAME
# filename = f'{_thisDir}{os.sep}' \
#            f'{participant_name}{os.sep}' \
#            f'mon_resp_time_stim'
#
# # Experiment Handler
# thisExp = data.ExperimentHandler(name=expName, version=psychopy_version,
#                                  extraInfo=expInfo, runtimeInfo=None,
#                                  savePickle=None, saveWideText=False,
#                                  dataFileName=filename)
#
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
bgLumP = int(expInfo['7. Background lum in percent of maxLum'])  # 20
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
viewdistPix = widthPix/monitorwidth*viewdist
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
                    allowGUI=True,
                    fullscr=False,
                    )

# print(f"check win.size: {win.size}")
# widthPix = widthPix/2
# heightPix = heightPix/2
# print(f"widthPix: {widthPix}, hight: {heightPix}")
# widthPix, heightPix = win.size
# print(f"check win.size: {win.size}")
#
#
# print(f"type(win.getActualFrameRate()): {type(win.getActualFrameRate())} {win.getActualFrameRate()}")
# print(f"check win.size: {win.size}")
# check_correct_monitor(monitor_name=monitor_name,
#                       actual_size=win.size,
#                       actual_fps=win.getActualFrameRate(),
#                       verbose=True)
#
# # check correct monitor details (fps, size) have been accessed.
# print(win.monitor.name, win.monitor.getSizePix())
# actualFrameRate = int(win.getActualFrameRate())
#
# print(f"actual_size: {type(win.monitor.getSizePix())} {win.monitor.getSizePix()}")
# print(f"actual fps: {type(win.getActualFrameRate())} {win.getActualFrameRate()}")
#
# if fps in list(range(actualFrameRate-2, actualFrameRate+2)):
#     print("fps matches actual frame rate")
# else:
#     # if values don't match, quit experiment
#     print(f"fps ({fps}) does not match actual frame rate ({actualFrameRate})")
#     core.quit()
#
#
# # actual_size = win.size
# actual_size = win.monitor.getSizePix()
# print(f"idiot check. I think I should use win.monitor.getSizePix() {win.monitor.getSizePix()}.\n"
#       f"currently using win.size {win.size}")
#
#
# if list(mon_dict['size']) == list(actual_size):
#     print(f"monitor is expected size")
# elif list(mon_dict['size']) == list(actual_size/2):
#     print(f"actual size is double expected size - Its ok, just a mac retina display bug.")
# else:
#     print(f"Display size does not match expected size from montior centre")
#     # check sizes seems unreliable,
#     # it returns different values for same screen if different mon_names are used!
#     check_sizes = win._checkMatchingSizes(mon_dict['size'], actual_size)
#     print(check_sizes)
#     core.quit()
#
# # check sizes seems unreliable,
# print(f"double checking win._checkMatchingSizes({mon_dict['size']}, {actual_size})")
# # it returns different values for same screen if different mon_names are used!
# check_sizes = win._checkMatchingSizes(mon_dict['size'], actual_size)
# print(check_sizes)


# ELEMENTS
# fixation bull eye
# todo: change fixation to trails counter?
fixation = visual.Circle(win, radius=2, units='pix',
                         lineColor='white', fillColor='black')

# PROBEs
# probe color
if expInfo['8. Red filter'] == 'yes':
    redfilter = -1
else:
    redfilter = 1

# probe sizes choice
if expInfo['6. Probe size'] == '6pixels':  # 6 pixels
    probeVert = [(0, 0), (1, 0), (1, 1), (2, 1),
                 (2, -2), (-1, -2), (-1, -1), (0, -1)]

elif expInfo['6. Probe size'] == '3pixels':  # 3 pixels
    probeVert = [(0, 0), (1, 0), (1, 1), (2, 1), (2, 0), (1, 0), (1, -1),
                 (0, -1), (0, -2), (-1, -2), (-1, -2), (-1, -1), (0, -1)]

else:  # 5 pixels
    # default setting is expInfo['6. Probe size'] == '5pixels':
    expInfo['6. Probe size'] = '5pixels'
    probeVert = [(0, 0), (1, 0), (1, 1), (2, 1), (2, -1), (1, -1),
                 (1, -2), (-1, -2), (-1, -1), (0, -1)]

probe1 = visual.ShapeStim(win, vertices=probeVert, fillColor=(1.0, -1.0, 1.0),
                          lineWidth=0, opacity=1, size=1, interpolate=False)
probe2 = visual.ShapeStim(win, vertices=probeVert, fillColor=[-1.0, 1.0, -1.0],
                          lineWidth=0, opacity=1, size=1, interpolate=False)

# MOUSE - hide cursor
# myMouse = event.Mouse(visible=False)
#
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

# Trial counter
trials_counter = visual.TextStim(win=win, name='trials_counter', text="???",
                                 font='Arial', height=20,
                                 # default set to black (e.g., invisible)
                                 color='black',
                                 pos=[-widthPix*.45, -heightPix*.45])
if trials_counter:
    # if trials counter yes, change colour to white.
    trials_counter.color = 'white'



# BREAKS
breaks = visual.TextStim(win=win, name='breaks',
                         text="turn on the light and  take at least 30-seconds break.",
                         font='Arial', pos=[0, 0], height=20, ori=0, color=[255, 255, 255],
                         colorSpace='rgb255', opacity=1, languageStyle='LTR', depth=0.0)

#
# # STAIRCASE
# total_nTrials = 0
# # todo: change from list(range(1, 15)) to list(range(1, len(separations)))
# expInfo['startPoints'] = list(range(1, 15))  # 14 staircases (14 conditions)
# expInfo['nTrials'] = trial_number
#
# stairStart = maxLum
# miniVal = bgLum
# maxiVal = maxLum
#
# stairs = []
# for thisStart in expInfo['startPoints']:
# 
#     thisInfo = copy.copy(expInfo)
#     thisInfo['thisStart'] = thisStart
# 
#     this_cond = Staircase(name='trials',
#                           type='simple',
#                           value=stairStart,
#                           C=stairStart*0.6,  # typically 60% of reference stimulus
#                           minRevs=3,
#                           minTrials=trial_number,
#                           minVal=miniVal,
#                           maxVal=maxiVal,
#                           targetThresh=0.75,  # changed this from prev versions
#                           extraInfo=thisInfo)
#     stairs.append(this_cond)


# EXPERIMENT
# for trialN in range(expInfo['nTrials']):
#     shuffle(stairs)

n_conds = len(sep_list)
this_cond = 0
responseList = []

# for this_cond, cond_vals in cond_dict.items():
# while len(responseList) < n_conds:
while this_cond < n_conds:

    cond_vals = cond_dict[this_cond]
    sep = cond_vals['sep']
    ISI = cond_vals['isi']
    print(f"this_cond: {this_cond}: sep: {sep}, ISI: {ISI}")


    target_jump = 1  # random.choice([1, -1])
    # stairNum = this_cond.extraInfo['thisStart']
    # staircase varied probeLum
    probeLum = maxLum  # this_cond.next()
    probeColor255 = probeLum * LumColor255Factor
    probeColor1 = (probeColor255 * Color255Color1Factor) - 1

    # print(f"\ntrialN: {trialN} this_cond: {this_cond} stairNum: {stairNum}\nprobeLum: {probeLum}")

    # total_nTrials = total_nTrials + 1

    # Black or White
    probe1.color = [probeColor1, probeColor1*redfilter, probeColor1*redfilter]
    probe2.color = [probeColor1, probeColor1*redfilter, probeColor1*redfilter]

    # print(f"\nbgLum: {bgLum} bgColor255: {bgColor255} win.colorSpace: {win.colorSpace}")


    # PROBE LOCATION
    # corners go CCW(!) 45=top-right, 135=top-left, 225=bottom-left, 315=bottom-right
    corner = 225  # random.choice([45, 135, 225, 315])
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
                probe2.pos = [p1_x - sep+1, p1_y + sep]
            elif target_jump == -1:  # CW
                probe1.ori = 180
                probe2.ori = 0
                probe2.pos = [p1_x + sep-1, p1_y - sep]
            elif target_jump == 9:
                probe1.ori = random.choice([0, 180])
    elif corner == 135:
        p1_x = x_prob * -1
        p1_y = y_prob * 1
        if orientation == 'tangent':
            if target_jump == 1:  # CCW
                probe1.ori = 90
                probe2.ori = 270
                probe2.pos = [p1_x + sep-1, p1_y + sep]
            elif target_jump == -1:  # CW
                probe1.ori = 270
                probe2.ori = 90
                probe2.pos = [p1_x - sep+1, p1_y - sep]
            elif target_jump == 9:
                probe1.ori = random.choice([90, 270])
    elif corner == 225:
        p1_x = x_prob * -1
        p1_y = y_prob * -1
        if orientation == 'tangent':
            if target_jump == 1:  # CCW
                probe1.ori = 180
                probe2.ori = 0
                probe2.pos = [p1_x + sep-1, p1_y - sep]
            elif target_jump == -1:  # CW
                probe1.ori = 0
                probe2.ori = 180
                probe2.pos = [p1_x - sep+1, p1_y + sep]
            elif target_jump == 9:
                probe1.ori = random.choice([0, 180])
    else:
        corner = 315
        p1_x = x_prob * 1
        p1_y = y_prob * -1
        if orientation == 'tangent':
            if target_jump == 1:  # CCW
                probe1.ori = 270
                probe2.ori = 90
                probe2.pos = [p1_x - sep+1, p1_y - sep]
            elif target_jump == -1:  # CW
                probe1.ori = 90
                probe2.ori = 270
                probe2.pos = [p1_x + sep-1, p1_y + sep]
            elif target_jump == 9:
                probe1.ori = random.choice([90, 270])

    probe1.pos = [p1_x, p1_y]

    # timing in frames
    # if ISI >= 0:
    # todo: change t_interval_1 and 2 to t_probe_1 and 2?
    t_fixation = 1 * fps
    t_interval_1 = t_fixation + probe_duration
    t_ISI = t_interval_1 + ISI
    t_interval_2 = t_ISI + probe_duration
    # I presume this means almost unlimited time to respond?
    t_response = t_interval_2 + 10000*fps

    # print(f't_fixation: {t_fixation}')
    # print(f't_interval_1: {t_interval_1}')
    # print(f't_ISI: {t_ISI}')
    # print(f't_interval_2: {t_interval_2}')
    # print(f't_response: {t_response}')


    # repeat the trial if [r] has been pressed
    repeat = True
    while repeat:
        frameN = -1

        # display Break before trials 120 and 240
        # if total_nTrials == 120+1 or total_nTrials == 240+1:
        if this_cond == 120+1 or this_cond == 240+1:
            continueRoutine = False
            breaks.draw()
            win.flip()
            while not event.getKeys():
                continueRoutine = True
        else:
            continueRoutine = True

        continueRoutine = True

        while continueRoutine:
            frameN = frameN + 1

            # FIXATION
            if t_fixation >= frameN > 0:
                fixation.setRadius(3)
                fixation.draw()
                trials_counter.text = f"({this_cond}/{n_conds}). sep: {sep}, ISI: {ISI}"
                trials_counter.draw()

            # PROBE 1
            if t_interval_1 >= frameN > t_fixation:
                probe1.draw()
                # SIMULTANEOUS CONDITION
                if ISI == -1:
                    if sep <= 18:
                        probe2.draw()
                fixation.setRadius(3)
                fixation.draw()
                trials_counter.draw()

            # ISI
            if t_ISI >= frameN > t_interval_1:
                fixation.setRadius(3)
                fixation.draw()
                trials_counter.draw()

            # PROBE 2
            if t_interval_2 >= frameN > t_ISI:
                if ISI >= 0:
                    if sep <= 18:
                        probe2.draw()
                fixation.setRadius(3)
                fixation.draw()
                trials_counter.draw()

            # ANSWER
            if frameN > t_interval_2:
                fixation.setRadius(2)
                fixation.draw()
                trials_counter.draw()

                #
                # # # # # #
                # response = getResponse(f"Did it record? yes [1] or no [9]\n")
                #
                # if response == 9:
                #     responseList.pop()  # removed the last item in the list
                #     print(f" Trial:{this_cond}, sep: {cond_vals['sep']}, isi: {cond_vals['isi']}. re-do trial")
                #     # this_cond -= 1  # go back one trial number
                #     # this_cond = this_cond  # go back one trial number
                #
                # else:
                #     # if not re-do trial, it must be a valid response (1 or 2)
                #     # dataToSave = f"{thisStair} {currentSize1} {response}\n"  # old txt version
                #
                #     dataToSave = response
                #     # I just append data to a list, and write the file at the end.
                #     responseList.append(dataToSave)
                #     this_cond += 1
                # # # # # #

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
                            this_cond += 1
                    elif corner == 135:
                        if (resp.keys == 'q') or (resp.keys == 'num_4'):
                            resp.corr = 1
                            this_cond += 1
                    elif corner == 225:
                        if (resp.keys == 'a') or (resp.keys == 'num_1'):
                            resp.corr = 1
                            this_cond += 1
                    elif corner == 315:
                        if (resp.keys == 's') or (resp.keys == 'num_2'):
                            resp.corr = 1
                            this_cond += 1

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


    # thisExp.addData('stair', stairNum)
    # thisExp.addData('probe_jump', target_jump)
    # thisExp.addData('probeColor1', probeColor1)
    # thisExp.addData('probeColor255', probeColor255)
    # thisExp.addData('probeLum', probeLum)
    # thisExp.addData('trial_response', resp.corr)
    # thisExp.addData('corner', corner)
    # thisExp.addData('probe_ecc', probe_ecc)
    # thisExp.addData('resp.rt', resp.rt)
    # thisExp.addData('total_nTrials', total_nTrials)
    # thisExp.addData('orientation', orientation)
    # thisExp.nextEntry()
    #
    # this_cond.newValue(resp.corr)   # so that the staircase adjusts itself
