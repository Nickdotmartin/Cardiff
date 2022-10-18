from __future__ import division
from psychopy import gui, visual, core, data, event, monitors
from psychopy import __version__ as psychopy_version
import os
import numpy as np
from numpy import deg2rad
from numpy.random import shuffle
import random
import copy
from datetime import datetime
from math import tan, sqrt
from kestenSTmaxVal import Staircase
from collections import defaultdict

# todo: change sep and ISI back
'''
Version of Exp1a, but to be viewed at double the distance.
So viewDist, probe_size and separation are all doubled.
Eccentricity is kept at 4dva.
ISI is tuned so that if run on a different montior it gives similar values.

I've not changed text size
'''

# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# Monitor config from monitor centre
# todo: Use asus_cal
monitor_name = 'Asus_VG24'  # 'NickMac' 'asus_cal' 'Asus_VG24' 'HP_24uh' 'ASUS_2_13_240Hz' 'Iiyama_2_18' 'Nick_work_laptop'


# Store info about the experiment session
expName = 'Exp1_double_dist'

expInfo = {'1. Participant': 'nicktest',
           '1. run_number': '1',
           '2. Probe duration in frames at 240hz': [2, 50, 100],
           '3. fps': [60, 240, 144, 60],
           '4_Trials_counter': [True, False],
           '5_vary_fixation': [False, True]
           }

# GUI
dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
if not dlg.OK:
    core.quit()  # user pressed escape

expInfo['time'] = datetime.now().strftime("%H:%M:%S")
expInfo['date'] = datetime.now().strftime("%d/%m/%Y")

# GUI SETTINGS
participant_name = expInfo['1. Participant']
run_number = int(expInfo['1. run_number'])
n_trials_per_stair = 25
probe_duration = int(expInfo['2. Probe duration in frames at 240hz'])
probe_ecc = 4  # kept at 4dva, view dist, sep and size have doubled
fps = int(expInfo['3. fps'])
orientation = 'tangent'  # expInfo['5. Probe orientation']
trials_counter = eval(expInfo['4_Trials_counter'])
vary_fixation = eval(expInfo['5_vary_fixation'])

# 1 is the original size, 2 is double (for when seated at double distance).
probe_size = 2


# VARIABLES
'''Distances between probes (spatially and temporally)
For 1probe condition, use separation==99.
For concurrent probes, use ISI==-1.
'''
# exp1_sep_vals = [0, 6]  # select from [0, 1, 2, 3, 6, 18, 99]
exp1_sep_vals = [0, 3, 6]  # select from [0, 1, 2, 3, 6, 18, 99]
# exp1_ISI_fr_vals = [-1, 6]  # select from [-1, 0, 2, 4, 6, 9, 12, 24]
exp1_ISI_fr_vals = [-1, 3, 6]  # select from [-1, 0, 2, 3, 4, 6, 9, 12, 24]
# exp1_ISI_fr_vals = [-1, 0, 2, 3, 4, 6, 9, 12, 24]  # select from [-1, 0, 2, 4, 6, 9, 12, 24]

separations = [i*probe_size for i in exp1_sep_vals]
print(f'separations: {separations}\n'
      f'These are equivallent to (exp1_sep_vals: {exp1_sep_vals})\n')

print(f'fps: {fps}')

frame_ms_at_240 = 1000/240
exp1_ISI_ms_vals = [-1 if i == -1 else i*frame_ms_at_240 for i in exp1_ISI_fr_vals]

if fps == 240:
    ISI_values = exp1_ISI_fr_vals
    these_ISI_ms_vals = exp1_ISI_ms_vals
    print(f'ISI_values: {ISI_values}')
    print(f'these_ISI_ms_vals: {these_ISI_ms_vals}')
else:
    # if fps not 240, will find frame rates with similar ms to ones selected
    ISI_values = [-1 if i == -1 else int(round(fps*i/1000, 0)) for i in exp1_ISI_ms_vals]

    # fps < 240 will not have the same temporal resolution, so duplicate  ISIs will be removed
    contains_duplicates = any(ISI_values.count(element) > 1 for element in ISI_values)
    if contains_duplicates:
        print(f"\n***WARNING***\nDuplicates in ISI_values at this fps.\n"
              f"Removing duplicates to have fewer ISI condition the\n")

        # getting  indices of duplicates
        tally = defaultdict(list)
        dup_idx_list = []
        for i, item in enumerate(ISI_values):
            tally[item].append(i)
            for key, locs in tally.items():
                if len(locs) > 1:
                    dup_idx_list.append(locs[1:])
        flat_dup_list = [item for sublist in dup_idx_list for item in sublist]
        flat_dup_list = list(set(flat_dup_list))

        # removing duplicates
        ISI_values = [e for i, e in enumerate(ISI_values) if i not in flat_dup_list]
        exp1_ISI_fr_vals = [e for i, e in enumerate(exp1_ISI_fr_vals) if i not in flat_dup_list]
        exp1_ISI_ms_vals = [e for i, e in enumerate(exp1_ISI_ms_vals) if i not in flat_dup_list]

    this_ms_per_frame = 1000/fps
    these_ISI_ms_vals = [-1 if i == -1 else i*this_ms_per_frame for i in ISI_values]
    print(f'ISI_values: {ISI_values}')
    print(f'these_ISI_ms_vals: {these_ISI_ms_vals}')
    print(f'(These are close to exp1_ISI_fr_vals: {exp1_ISI_fr_vals}')
    print(f'exp1_ISI_ms_vals: {exp1_ISI_ms_vals})\n')


# repeat separation values for each ISI e.g., [0, 0, 6, 6]
sep_vals_list = list(np.repeat(separations, len(ISI_values)))
print(f'sep_vals_list: {sep_vals_list}')
# ISI_vals_list cycles through ISIs e.g., [-1, 6, -1, 6]
ISI_vals_list = list(np.tile(ISI_values, len(separations)))
print(f'ISI_vals_list: {ISI_vals_list}')
ISI_ms_list = list(np.tile(these_ISI_ms_vals, len(separations)))
print(f'ISI_vals_list: {ISI_vals_list}')

# stair_names_list joins sep_vals_list and ISI_vals_list
# e.g., ['sep0_ISI-1', 'sep0_ISI6', 'sep6_ISI-1', 'sep6_ISI6']
stair_names_list = [f'sep{s}_ISI{c}' for s, c in zip(sep_vals_list, ISI_vals_list)]
print(f'stair_names_list: {stair_names_list}')
n_stairs = len(sep_vals_list)
print(f'n_stairs: {n_stairs}')
total_n_trials = int(n_trials_per_stair * n_stairs)
print(f'total_n_trials: {total_n_trials}')

# FILENAME
filename = f'{_thisDir}{os.sep}' \
           f'{expName}{os.sep}' \
           f'{participant_name}{os.sep}' \
           f'{participant_name}_{run_number}{os.sep}' \
           f'{participant_name}_{run_number}_output'
# files are labelled as '_incomplete' unless entire script runs.
save_output_name = filename + '_incomplete'

# Experiment Handler
thisExp = data.ExperimentHandler(name=expName, version=psychopy_version,
                                 extraInfo=expInfo, runtimeInfo=None,
                                 savePickle=None, saveWideText=True,
                                 dataFileName=save_output_name)

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
bgLumP = 20  # int(expInfo['7. Background lum in percent of maxLum'])  # 20
bgLum = maxLum * bgLumP / 100
bgColor255 = bgLum * LumColor255Factor
bgColor1 = (bgColor255 * Color255Color1Factor) - 1


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
display_number = 1  # 0 indexed, 1 for external display, 0 for internal
if monitor_name in ['ASUS_2_13_240Hz', 'asus_cal', 'Nick_work_laptop', 'NickMac']:
    display_number = 0
use_full_screen = True
if display_number > 0:
    use_full_screen = False
widthPix = mon_dict['size'][0]
heightPix = mon_dict['size'][1]
monitorwidth = mon_dict['width']  # monitor width in cm
viewdist = float(mon_dict['dist'])  # viewing distance in cm

mon = monitors.Monitor(monitor_name, width=monitorwidth, distance=viewdist)
mon.setSizePix((widthPix, heightPix))
mon.save()

dbl_view_dist = viewdist * probe_size
viewdistPix = widthPix/monitorwidth*dbl_view_dist
print(f"Original view_dist ({viewdist}) changed to: {dbl_view_dist}. viewdistPix is now: {viewdistPix}")

# WINDOW SPEC
win = visual.Window(monitor=mon, size=(widthPix, heightPix),
                    colorSpace='rgb255', color=bgColor255,
                    winType='pyglet',  # I've added this to make it work on pycharm/mac
                    pos=[1, -1],  # pos gives position of top-left of screen
                    units='pix',
                    screen=display_number,
                    allowGUI=False,
                    fullscr=use_full_screen)


# ELEMENTS
# fixation bull eye - radius changed from 2 to 4
fixation = visual.Circle(win, radius=4, units='pix', lineColor='white', fillColor='black')

# PROBEs - size changed from 1 to 2
expInfo['6. Probe size'] = '5pixels'  # ignore this, all experiments use 5pixel probes now.
probeVert = [(0, 0), (1, 0), (1, 1), (2, 1), (2, -1), (1, -1),
             (1, -2), (-1, -2), (-1, -1), (0, -1)]

probe1 = visual.ShapeStim(win, vertices=probeVert, fillColor=(1.0, -1.0, 1.0),
                          lineWidth=0, opacity=1, size=probe_size, interpolate=False)
probe2 = visual.ShapeStim(win, vertices=probeVert, fillColor=[-1.0, 1.0, -1.0],
                          lineWidth=0, opacity=1, size=probe_size, interpolate=False)

# dist_from_fix is a constant to get 4dva distance from fixation,
dist_from_fix = round((tan(deg2rad(probe_ecc)) * viewdistPix) / sqrt(2))
print(f"probe x/y is now dist_from_fix: {dist_from_fix}")

# MOUSE - hide cursor
myMouse = event.Mouse(visible=False)

# # KEYBOARD
resp = event.BuilderKeyResponse()

# INSTRUCTION
instructions = visual.TextStim(win=win, name='instructions',
                               text="\n\n\n\n\n\nFocus on the small circle at the centre of the screen.\n\n"
                                    "A small white probe will briefly flash on screen,\n"
                                    "press the key related to the location of the probe:\n\n"
                                    "[4]/[Q] top-left\t\t\t[5]/[W] top-right\n\n\n\n"
                                    "[1]/[A] bottom-left\t\t\t[2]/[S] bottom-right.\n\n\n"
                                    "Do not rush, aim to be as accurate as possible,\n"
                                    "but if you did not see the probe, please guess.\n\n"
                                    "Don't let your eyes wander, keep focussed on the circle in the middle throughout.",
                               font='Arial', height=20,
                               color='white')

# Trial counter
trials_counter = visual.TextStim(win=win, name='trials_counter', text="???",
                                 font='Arial', height=20,
                                 # default set to black (e.g., invisible)
                                 color='black',
                                 pos=[-widthPix * .45, -heightPix * .45])
trials_counter.text = f"0/{total_n_trials}"
if trials_counter:
    # if trials counter yes, change colour to white.
    trials_counter.color = 'white'

# BREAKS
take_break = 76
print(f"take_break every {take_break} trials.")
breaks = visual.TextStim(win=win, name='breaks',
                         # text="turn on the light and take at least 30-seconds break.",
                         text="Break\nTurn on the light and take at least 30-seconds break.\n"
                              "Remember, if you don't see the flash, just guess!\n"
                              "Keep focussed on the circle in the middle of the screen.",
                         font='Arial', pos=[0, 0], height=20,
                         ori=0, color=[255, 255, 255],
                         colorSpace='rgb255', opacity=1, languageStyle='LTR', depth=0.0)

end_of_exp = visual.TextStim(win=win, name='end_of_exp',
                             text="You have completed this experiment.\n"
                                  "Thank you for your time.\n\n"
                                  "Press any key to return to the desktop.",
                             font='Arial', height=20,
                             )

while not event.getKeys():
    fixation.setRadius(6)  # fixation radius doubled from 3 to 6.
    fixation.draw()
    instructions.draw()
    trials_counter.draw()
    win.flip()

# STAIRCASE
expInfo['stair_list'] = list(range(n_stairs))
expInfo['n_trials_per_stair'] = n_trials_per_stair
print('\nexpInfo (dict)')
for k, v in expInfo.items():
    print(f"{k}: {v}")

stairStart = maxLum
miniVal = bgLum
maxiVal = maxLum

stairs = []
for stair_idx in expInfo['stair_list']:

    thisInfo = copy.copy(expInfo)
    thisInfo['stair_idx'] = stair_idx

    thisStair = Staircase(name=stair_names_list[stair_idx],
                          type='simple',
                          value=stairStart,
                          C=stairStart * 0.6,  # step_size, typically 60% of reference stimulus
                          minRevs=3,
                          minTrials=n_trials_per_stair,
                          minVal=miniVal,
                          maxVal=maxiVal,
                          targetThresh=0.75,
                          extraInfo=thisInfo)
    stairs.append(thisStair)


# EXPERIMENT
trial_number = 0
print('\n*** exp loop*** \n\n')
for step in range(n_trials_per_stair):
    shuffle(stairs)
    for thisStair in stairs:

        trial_number = trial_number + 1
        trials_counter.text = f"{trial_number}/{total_n_trials}"
        stair_idx = thisStair.extraInfo['stair_idx']
        print(f"\ntrial_number: {trial_number}, stair_idx: {stair_idx}, thisStair: {thisStair}, step: {step}")

        sep = sep_vals_list[stair_idx]
        ISI = ISI_vals_list[stair_idx]
        ISI_ms = ISI_ms_list[stair_idx]
        print(f"sep: {sep}, ISI: {ISI}, ({ISI_ms}ms")

        # staircase varies probeLum
        probeLum = thisStair.next()
        probeColor255 = int(probeLum * LumColor255Factor)  # rgb255 are ints.
        probeColor1 = (probeColor255 * Color255Color1Factor) - 1
        probe1.color = [probeColor1, probeColor1, probeColor1]
        probe2.color = [probeColor1, probeColor1, probeColor1]
        print(f"probeLum: {probeLum}, probeColor255: {probeColor255}, probeColor1: {probeColor1}")

        # PROBE LOCATION
        # # corners go CCW(!) 45=top-right, 135=top-left, 225=bottom-left, 315=bottom-right
        corner = random.choice([45, 135, 225, 315])
        # # direction in which the probe jumps : CW or CCW
        target_jump = random.choice([1, -1])
        print(f"corner: {corner}; target_jump: {target_jump}")

        # corner_name = 'top_right'
        # if corner == 135:
        #     corner_name = 'top_left'
        # elif corner == 225:
        #     corner_name = 'bottom_left'
        # elif corner == 315:
        #     corner_name = 'bottom_right'
        #
        # jump_dir = 'clockwise'
        # if target_jump == -1:
        #     jump_dir = 'anticlockwise'
        #
        # print(f"corner_name: {corner_name}; jump_dir: {jump_dir}")

        # set probe ori
        if corner == 45:
            # in top-right corner, both x and y increase (right and up)
            p1_x = dist_from_fix * 1
            p1_y = dist_from_fix * 1
            #  'orientation' here refers to the relationship between probes,
            #  whereas probe1.ori refers to rotational angle of probe stimulus
            if orientation == 'tangent':
                if target_jump == 1:  # CW
                    probe1.ori = 180
                    probe2.ori = 0
                    probe2.pos = [p1_x + sep - probe_size, p1_y - sep]  # probe2 xpos changed from 1 to 2 pixels.
                elif target_jump == -1:  # ACW
                    probe1.ori = 0
                    probe2.ori = 180
                    probe2.pos = [p1_x - sep + probe_size, p1_y + sep]  # probe2 xpos changed from 1 to 2 pixels.
        elif corner == 135:
            p1_x = dist_from_fix * -1
            p1_y = dist_from_fix * 1
            if orientation == 'tangent':
                if target_jump == 1:  # CW
                    probe1.ori = 90
                    probe2.ori = 270
                    probe2.pos = [p1_x + sep - probe_size, p1_y + sep]  # probe2 xpos changed from 1 to 2 pixels.
                elif target_jump == -1:  # ACW
                    probe1.ori = 270
                    probe2.ori = 90
                    probe2.pos = [p1_x - sep + probe_size, p1_y - sep]  # probe2 xpos changed from 1 to 2 pixels.
        elif corner == 225:
            p1_x = dist_from_fix * -1
            p1_y = dist_from_fix * -1
            if orientation == 'tangent':
                if target_jump == 1:  # CW
                    probe1.ori = 0
                    probe2.ori = 180
                    probe2.pos = [p1_x - sep + probe_size, p1_y + sep]  # probe2 xpos changed from 1 to 2 pixels.
                elif target_jump == -1:  # ACW
                    probe1.ori = 180
                    probe2.ori = 0
                    probe2.pos = [p1_x + sep - probe_size, p1_y - sep]  # probe2 xpos changed from 1 to 2 pixels.
        else:
            corner = 315
            p1_x = dist_from_fix * 1
            p1_y = dist_from_fix * -1
            if orientation == 'tangent':
                if target_jump == 1:  # CW
                    probe1.ori = 270
                    probe2.ori = 90
                    probe2.pos = [p1_x - sep + probe_size, p1_y - sep]  # probe2 xpos changed from 1 to 2 pixels.
                elif target_jump == -1:  # ACW
                    probe1.ori = 90
                    probe2.ori = 270
                    probe2.pos = [p1_x + sep - probe_size, p1_y + sep]  # probe2 xpos changed from 1 to 2 pixels.

        probe1.pos = [p1_x, p1_y]

        print(f"probe1: {probe1.pos}, probe2.pos: {probe2.pos}. dff: {dist_from_fix}")

        # to avoid fixation times always being the same which might increase
        # anticipatory effects,
        # add in a random number of frames (up to 1 second) to fixation time
        vary_fix = 0
        if vary_fixation:
            vary_fix = np.random.randint(0, fps)

        # timing in frames
        # fixation time is now 70ms shorter than rad_flow1, as we can have
        # priliminary bg_motion.
        t_fixation = (fps / 2) + vary_fix
        t_probe_1 = t_fixation + probe_duration
        t_ISI = t_probe_1 + ISI
        t_probe_2 = t_ISI + probe_duration
        t_response = t_probe_2 + 10000 * fps  # essentially unlimited time to respond

        # repeat the trial if [r] has been pressed
        repeat = True
        while repeat:
            frameN = -1

            # take a break every ? trials
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
                    fixation.setRadius(6)  # fixation radius doubled from 3 to 6
                    fixation.draw()
                    trials_counter.draw()

                    # reset timer to start with probe1 presentation.
                    resp.clock.reset()


                # PROBE 1
                if t_probe_1 >= frameN > t_fixation:
                    probe1.draw()

                    if ISI == -1:  # SIMULTANEOUS CONDITION (concurrent)
                        if sep <= 18:  # don't draw 2nd probe in 1probe cond (sep==99)
                            probe2.draw()
                    fixation.setRadius(6)  # fixation radius doubled from 3 to 6
                    fixation.draw()
                    trials_counter.draw()

                # ISI
                if t_ISI >= frameN > t_probe_1:
                    fixation.setRadius(6)  # fixation radius doubled from 3 to 6
                    fixation.draw()
                    trials_counter.draw()

                # PROBE 2
                if t_probe_2 >= frameN > t_ISI:
                    if ISI >= 0:
                        if sep <= 18:  # don't draw 2nd probe in 1probe cond (sep==99)
                            probe2.draw()
                    fixation.setRadius(6)  # fixation radius doubled from 3 to 6
                    fixation.draw()
                    trials_counter.draw()

                # ANSWER
                if frameN > t_probe_2:
                    fixation.setRadius(4)  # fixation radius doubled from 2 to 4
                    fixation.draw()
                    trials_counter.draw()

                    # Response
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

                # regardless of frameN, check for quit
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
        thisExp.addData('ISI', ISI)
        thisExp.addData('ISI_ms', ISI_ms)
        thisExp.addData('probe_jump', target_jump)
        thisExp.addData('probeColor1', probeColor1)
        thisExp.addData('probeColor255', probeColor255)
        thisExp.addData('probeLum', probeLum)
        thisExp.addData('trial_response', resp.corr)
        thisExp.addData('corner', corner)
        thisExp.addData('probe_ecc', probe_ecc)
        thisExp.addData('resp.rt', resp.rt)
        thisExp.addData('probe_size', probe_size)
        thisExp.addData('orientation', orientation)
        thisExp.addData('vary_fixation', vary_fixation)
        thisExp.addData('expName', expName)
        thisExp.addData('monitor_name', monitor_name)
        thisExp.addData('selected_fps', fps)

        thisExp.nextEntry()

        thisStair.newValue(resp.corr)   # so that the staircase adjusts itself


print("end of experiment loop, saving data")
thisExp.dataFileName = filename
thisExp.close()

while not event.getKeys():
    # display end of experiment screen
    end_of_exp.draw()
    win.flip()
else:
    # close and quit once a key is pressed
    win.close()
    core.quit()
