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

# from PsychoPy_tools import check_correct_monitor
from kestenSTmaxVal import Staircase


'''Issues with concurrent timings resolved with use of isi_dur variable.'''

'''
Script to break relationship between spatial and temporal integration, which might reduce apparent motion effect.
For split probes, the pixels that compose the original probes have been shuffled, or split between the 2 probes.  
E.g., the first probe to appear is composed of three pixels from the original probe 1 and two from probe 2.  
That is, participants still have to integrate over the same total space and time, but there is not the same coherent motion from
probe 1 location to probe2 location.

THis script allows both split or original probes to be used.
Probes can be oriented tangentally or radially. 
'''


# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# Monitor config from monitor centre
monitor_name = 'Nick_work_laptop'  # 'NickMac' 'asus_cal' 'Asus_VG24' 'HP_24uh' 'ASUS_2_13_240Hz' 'Iiyama_2_18' 'Nick_work_laptop'

# Store info about the experiment session
expName = 'EXP1b_split_probes'  # from the Builder filename that created this script

expInfo = {'1. Participant': 'nicktest',
           '2. Run_number': '1',
           '3. Probe duration in frames at 240hz': [2, 50, 100],
           '4. fps': [60, 144, 240],
           '5. Probe_orientation': ['radial', 'tangent'],
           '6. Trial_counter': [True, False],
           '7. Vary_fixation': [False, True],
           '8. Exp_type': ['just_split', 'split_v_orig'],
           }

# GUI
dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
if not dlg.OK:
    core.quit()  # user pressed escape

expInfo['time'] = datetime.now().strftime("%H:%M:%S")
expInfo['date'] = datetime.now().strftime("%d/%m/%Y")

# GUI SETTINGS
participant_name = expInfo['1. Participant']
run_number = int(expInfo['2. Run_number'])
n_trials_per_stair = 25
probe_duration = int(expInfo['3. Probe duration in frames at 240hz'])
probe_ecc = 4
fps = int(expInfo['4. fps'])
orientation = expInfo['5. Probe_orientation']
trials_counter = eval(expInfo['6. Trial_counter'])
vary_fixation = eval(expInfo['7. Vary_fixation'])
exp_type = expInfo['8. Exp_type']


# VARIABLES
print("\nVariables")
# todo: add extra conds if split v orig: probes_type_list
if exp_type == 'just_split':
    print('do just split')
    probe_types = ['split']
else:
    print('mix split and orig probes with probes_type_list')
    probe_types = ['split', 'orig']
print(f'probe_types: {probe_types}')

# Distances between probes (spatioally and temporally)
'''Sort separation and ISI types'''
separations = [0, 3, 6]  # select from [0, 1, 2, 3, 6, 18, 99]
# separations = [0]
print(f'separations: {separations}')
# # I also have two ISI types
ISI_values = [-1, 3, 6]  # select from [-1, 0, 2, 4, 6, 9, 12, 24]
# ISI_values = [0]
print(f'ISI_values: {ISI_values}')
# repeat separation values for each ISI e.g., [0, 0, 6, 6]
sep_vals_list = list(np.repeat(separations, len(ISI_values))) * len(probe_types)
print(f'sep_vals_list: {sep_vals_list}')


# ISI_vals_list cycles through ISIs e.g., [-1, 6, -1, 6]
ISI_vals_list = list(np.tile(ISI_values, len(separations) * len(probe_types)))
print(f'ISI_vals_list: {ISI_vals_list}')

# probes_type_list = list(np.repeat(probe_types, len(sep_vals_list))) * len(probe_types)

probes_type_list = list(np.repeat(probe_types, len(sep_vals_list) / len(probe_types)))
# probes_type_list = ['orig', 'orig']
print(f'probes_type_list: {probes_type_list}')
# stair_names_list joins sep_vals_list and ISI_vals_list
# e.g., ['sep0_ISI-1', 'sep0_ISI6', 'sep6_ISI-1', 'sep6_ISI6']
stair_names_list = [f'{p}_sep{s}_ISI{i}' for p, s, i in zip(probes_type_list, sep_vals_list, ISI_vals_list)]
print(f'stair_names_list: {stair_names_list}')
n_stairs = len(sep_vals_list)
print(f'n_stairs: {n_stairs}')

# FILENAME
filename = f'{_thisDir}{os.sep}' \
           f'{expName}{os.sep}' \
           f'{exp_type}{os.sep}' \
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
Color255Color1Factor = 1 / 127.5  # Color255 * Color255Color1Factor -1
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
print(f"\nmon_dict: {mon_dict}")

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
viewdist = mon_dict['dist']  # viewing distance in cm
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
                    allowGUI=False,
                    fullscr=use_full_screen)

print(f"check win.size: {win.size}")
widthPix = widthPix / 2
heightPix = heightPix / 2
print(f"widthPix: {widthPix}, hight: {heightPix}")
widthPix, heightPix = win.size
print(f"check win.size: {win.size}")


# ELEMENTS
# fixation bull eye
fixation = visual.Circle(win, radius=2, units='pix',
                         lineColor='white', fillColor='black')

# PROBEs
probe_size = 1

# split probes
probe_L_vert = [(0, 0), (2, 0), (2, 1), (1, 1), (1, -2), (0, -2)]

probe_dots_vert = [(0, -1), (0, -2), (-1, -2), (-1, -1),
                   (2, -1), (2, 0), (1, 0), (1, -1)]
probe_1a_L = visual.ShapeStim(win, vertices=probe_L_vert, fillColor='red',
                              lineWidth=0, opacity=1, size=probe_size, interpolate=False)
probe_2a_dots = visual.ShapeStim(win, vertices=probe_dots_vert, fillColor='green',
                                 lineWidth=0, opacity=1, size=probe_size, interpolate=False)

probe_2b_L = visual.ShapeStim(win, vertices=probe_L_vert, fillColor='green',
                              lineWidth=0, opacity=1, size=probe_size, interpolate=False)
probe_1b_dots = visual.ShapeStim(win, vertices=probe_dots_vert, fillColor='red',
                                 lineWidth=0, opacity=1, size=probe_size, interpolate=False)

# dist_from_fix is a constant to get 4dva distance from fixation,
dist_from_fix = round((tan(deg2rad(probe_ecc)) * viewdistPix) / sqrt(2))
print(f"dist_from_fix: {dist_from_fix}")

# MOUSE - hide cursor
myMouse = event.Mouse(visible=False)

# # KEYBOARD
resp = event.BuilderKeyResponse()

# INSTRUCTION
instructions = visual.TextStim(win=win, name='instructions',
                               text="\n\n\n\n\n\nFocus on the fixation circle at the centre of the screen.\n\n"
                                    "A small white target will briefly appear on screen,\n"
                                    "press the key related to the location of the probe:\n\n"
                                    "[4]/[Q] top-left\t\t\t[5]/[W] top-right\n\n\n\n"
                                    "[1]/[A] bottom-left\t\t\t[2]/[S] bottom-right.\n\n\n"
                                    "Some targets will be easier to see than others,\n"
                                    "Some will be so dim that you won't see them, so just guess!\n\n"
                                    "You don't need to think for long, respond quickly, but try to push press the correct key!\n\n"
                                    "Don't let your eyes wander, keep focussed on the circle in the middle throughout.",
                               font='Arial', height=20,
                               color='white')

# Trial counter
trials_counter = visual.TextStim(win=win, name='trials_counter', text="???",
                                 font='Arial', height=20,
                                 # default set to black (e.g., invisible)
                                 color='black',
                                 pos=[-widthPix * .45, -heightPix * .45])
if trials_counter:
    # if trials counter yes, change colour to white.
    trials_counter.color = 'white'

# BREAKS
take_break = 76
total_n_trials = int(n_trials_per_stair * n_stairs)
print(f"take_break every {take_break} trials.")
breaks = visual.TextStim(win=win, name='breaks',
                         text="Break\nTurn on the light and take at least 30-seconds break.\n"
                              "Keep focussed on the fixation circle in the middle of the screen.\n"
                              "Remember, if you don't see the target, just guess!",
                         font='Arial', pos=[0, 0], height=20, ori=0, color=[255, 255, 255],
                         colorSpace='rgb255', opacity=1, languageStyle='LTR', depth=0.0)

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
                          C=stairStart * 0.6,  # typically, 60% of reference stimulus
                          minRevs=3,
                          minTrials=n_trials_per_stair,
                          minVal=miniVal,
                          maxVal=maxiVal,
                          targetThresh=0.75,
                          extraInfo=thisInfo)
    stairs.append(thisStair)

trial_number = 0
# EXPERIMENT
for step in range(n_trials_per_stair):
    shuffle(stairs)
    for thisStair in stairs:

        trial_number = trial_number + 1
        trials_counter.text = f"{trial_number}/{total_n_trials}"
        stair_idx = thisStair.extraInfo['stair_idx']
        print(f"\ntrial_number: {trial_number}, stair_idx: {stair_idx}, thisStair: {thisStair}, step: {step}")

        sep = sep_vals_list[stair_idx]
        ISI = ISI_vals_list[stair_idx]
        # print(f"ISI: {ISI}, sep: {sep}")

        probes_type = probes_type_list[stair_idx]
        # print(f"probes_type: {probes_type}")

        if probes_type in ['orig']:
            neg_sep = 0-sep
            if sep == 0:
                neg_sep = -.1
        else:
            neg_sep = sep
        print(f"probes_type: {probes_type}, ISI: {ISI}, sep: {sep} (neg_sep: {neg_sep})")


        # staircase varies probeLum
        probeLum = thisStair.next()
        probeColor255 = int(probeLum * LumColor255Factor)  # rgb255 are ints.
        probeColor1 = (probeColor255 * Color255Color1Factor) - 1
        print(f"probeLum: {probeLum}")

        # Black or White
        probe_1a_L.color = [probeColor1, probeColor1, probeColor1]
        probe_2a_dots.color = [probeColor1, probeColor1, probeColor1]
        probe_2b_L.color = [probeColor1, probeColor1, probeColor1]
        probe_1b_dots.color = [probeColor1, probeColor1, probeColor1]
        # probe_1a_L.color = [probeColor1, 0, 0]  # red
        # probe_2a_dots.color = [0, probeColor1, 0]  # green
        # probe_2b_L.color = [0, 0, probeColor1]  # blue
        # probe_1b_dots.color = [probeColor1, 0, probeColor1]  # purple
        print(f"bgLum: {bgLum} bgColor255: {bgColor255} win.colorSpace: {win.colorSpace}")

        """
        My new probes are called: probe_1a_L, probe_2a_dots, probe_2b_L, probe_1b_dots.
        The two shapes are composed like this:
                (probe_1a_L, probe_2a_dots), (probe_2b_L, probe_1b_dots).

        for split probes:
            For timing I need to use 1&2 (both probe1s appear first, then both probe 2s).
            first:  (probe_1a_L,              ), (          , probe_1b_dots).
            second: (          , probe_2a_dots), (probe_2b_L,              ).
            For orientation and separation I need to treat them as shapes (a & b)
        for orig probes:
            I use a & b for timing, separation and orientation.
            first:  (probe_1a_L, probe_2a_dots), (                         ).
            second: (                         ), (probe_2b_L, probe_1b_dots).

        """

        # PROBE LOCATION
        # # corners go CCW(!) 45=top-right, 135=top-left, 225=bottom-left, 315=bottom-right
        corner = random.choice([45, 135, 225, 315])
        corner_name = 'top_right'
        if corner == 135:
            corner_name = 'top_left'
        elif corner == 225:
            corner_name = 'bottom_left'
        elif corner == 315:
            corner_name = 'bottom_right'

        # # direction in which the probe jumps : CW or CCW
        target_jump = random.choice([1, -1])
        if orientation == 'tangent':
            jump_dir = 'clockwise'
            if target_jump == -1:
                jump_dir = 'anticlockwise'
        else:
            jump_dir = 'inward'
            if target_jump == -1:
                jump_dir = 'outward'
        print(f"corner: {corner} {corner_name}; jump dir: {target_jump} {jump_dir}")

        # reset probe ori
        probe_1a_L.ori = 0
        probe_2a_dots.ori = 0
        probe_2b_L.ori = 0
        probe_1b_dots.ori = 0
        if corner == 45:
            p1_x = dist_from_fix * 1
            p1_y = dist_from_fix * 1
            if orientation == 'tangent':
                if target_jump == 1:  # CCW
                    probe_2b_L.ori = probe_1b_dots.ori = 180
                    probe_2b_L.pos = probe_1b_dots.pos = [p1_x - sep + 1, p1_y + sep]
                elif target_jump == -1:  # CW
                    probe_1a_L.ori = probe_2a_dots.ori = 180
                    probe_2b_L.pos = probe_1b_dots.pos = [p1_x + sep - 1, p1_y - sep]
            elif orientation == 'radial':
                if target_jump == 1:  # inward
                    probe_2b_L.ori = probe_1b_dots.ori = 90
                    probe_1a_L.ori = probe_2a_dots.ori = 270
                    # probe2 is left and down from probe1
                    probe_2b_L.pos = probe_1b_dots.pos = [p1_x - sep + 1, p1_y - sep]
                elif target_jump == -1:  # outward
                    probe_2b_L.ori = probe_1b_dots.ori = 270
                    probe_1a_L.ori = probe_2a_dots.ori = 90
                    # probe2 is right and up from probe1
                    probe_2b_L.pos = probe_1b_dots.pos = [p1_x + sep - 1, p1_y + sep]
        elif corner == 135:
            p1_x = dist_from_fix * -1
            p1_y = dist_from_fix * 1
            if orientation == 'tangent':
                if target_jump == 1:  # CCW
                    probe_1a_L.ori = probe_2a_dots.ori = 90
                    probe_2b_L.ori = probe_1b_dots.ori = 270
                    probe_2b_L.pos = probe_1b_dots.pos = [p1_x + sep - 1, p1_y + sep]
                elif target_jump == -1:  # CW
                    probe_1a_L.ori = probe_2a_dots.ori = 270
                    probe_2b_L.ori = probe_1b_dots.ori = 90
                    probe_2b_L.pos = probe_1b_dots.pos = [p1_x - sep + 1, p1_y - sep]
            elif orientation == 'radial':
                if target_jump == 1:  # inward
                    probe_2b_L.ori = probe_1b_dots.ori = 0
                    probe_1a_L.ori = probe_2a_dots.ori = 180
                    # probe2 is right and down from probe1
                    probe_2b_L.pos = probe_1b_dots.pos = [p1_x + sep - 1, p1_y - sep]
                elif target_jump == -1:  # outward
                    probe_2b_L.ori = probe_1b_dots.ori = 180
                    probe_1a_L.ori = probe_2a_dots.ori = 0
                    # probe2 is left and up from probe1
                    probe_2b_L.pos = probe_1b_dots.pos = [p1_x - sep + 1, p1_y + sep]
        elif corner == 225:
            p1_x = dist_from_fix * -1
            p1_y = dist_from_fix * -1
            if orientation == 'tangent':
                if target_jump == 1:  # CCW
                    probe_1a_L.ori = probe_2a_dots.ori = 180
                    probe_2b_L.pos = probe_1b_dots.pos = [p1_x + sep - 1, p1_y - sep]
                elif target_jump == -1:  # CW
                    probe_2b_L.ori = probe_1b_dots.ori = 180
                    probe_2b_L.pos = probe_1b_dots.pos = [p1_x - sep + 1, p1_y + sep]
            elif orientation == 'radial':
                if target_jump == 1:  # inward
                    probe_2b_L.ori = probe_1b_dots.ori = 270
                    probe_1a_L.ori = probe_2a_dots.ori = 90
                    # probe2 is right and up from probe1
                    probe_2b_L.pos = probe_1b_dots.pos = [p1_x + sep - 1, p1_y + sep]
                elif target_jump == -1:  # outward
                    probe_2b_L.ori = probe_1b_dots.ori = 90
                    probe_1a_L.ori = probe_2a_dots.ori = 270
                    # probe2 is left and down from probe1
                    probe_2b_L.pos = probe_1b_dots.pos = [p1_x - sep + 1, p1_y - sep]
        else:
            corner = 315
            p1_x = dist_from_fix * 1
            p1_y = dist_from_fix * -1
            if orientation == 'tangent':
                if target_jump == 1:  # CCW
                    probe_1a_L.ori = probe_2a_dots.ori = 270
                    probe_2b_L.ori = probe_1b_dots.ori = 90
                    probe_2b_L.pos = probe_1b_dots.pos = [p1_x - sep + 1, p1_y - sep]
                elif target_jump == -1:  # CW
                    probe_1a_L.ori = probe_2a_dots.ori = 90
                    probe_2b_L.ori = probe_1b_dots.ori = 270
                    probe_2b_L.pos = probe_1b_dots.pos = [p1_x + sep - 1, p1_y + sep]
            elif orientation == 'radial':
                if target_jump == 1:  # inward
                    probe_2b_L.ori = probe_1b_dots.ori = 180
                    probe_1a_L.ori = probe_2a_dots.ori = 0
                    # probe2 is left and up from probe1
                    probe_2b_L.pos = probe_1b_dots.pos = [p1_x - sep + 1, p1_y + sep]
                elif target_jump == -1:  # outward
                    probe_2b_L.ori = probe_1b_dots.ori = 0
                    probe_1a_L.ori = probe_2a_dots.ori = 180
                    # probe2 is right and down from probe1
                    probe_2b_L.pos = probe_1b_dots.pos = [p1_x + sep - 1, p1_y - sep]

        probe_1a_L.pos = probe_2a_dots.pos = [p1_x, p1_y]

        # sort timings for each condition
        if probes_type == 'split':
            print('split probe config')
            first_probe_L = probe_1a_L
            first_probe_dots = probe_1b_dots
            second_probe_L = probe_2b_L
            second_probe_dots = probe_2a_dots
        else:
            print('orig probe config')
            first_probe_L = probe_1a_L
            first_probe_dots = probe_2a_dots
            second_probe_L = probe_2b_L
            second_probe_dots = probe_1b_dots


        # timing in frames
        isi_dur = ISI
        if ISI < 0:
            isi_dur = 0
        t_fixation = 1 * fps
        t_probe_1 = t_fixation + probe_duration
        t_ISI = t_probe_1 + isi_dur
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
                    fixation.setRadius(3)
                    fixation.draw()
                    trials_counter.draw()

                    # reset timer to start with probe1 presentation.
                    resp.clock.reset()

                # PROBE 1
                if t_probe_1 >= frameN > t_fixation:
                    first_probe_L.draw()
                    first_probe_dots.draw()

                    # SIMULTANEOUS CONDITION
                    if ISI == -1:
                        if sep <= 18:
                            second_probe_dots.draw()
                            second_probe_L.draw()
                    fixation.setRadius(3)
                    fixation.draw()
                    trials_counter.draw()

                # ISI
                # todo: use elif instead of if
                if t_ISI >= frameN > t_probe_1:
                    fixation.setRadius(3)
                    fixation.draw()
                    trials_counter.draw()

                # PROBE 2
                if t_probe_2 >= frameN > t_ISI:
                    if ISI >= 0:
                        if sep <= 18:
                            second_probe_dots.draw()
                            second_probe_L.draw()
                    fixation.setRadius(3)
                    fixation.draw()
                    trials_counter.draw()

                # ANSWER
                if frameN > t_probe_2:
                    fixation.setRadius(2)
                    fixation.draw()
                    trials_counter.draw()

                    # ANSWER
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

        thisExp.addData('trial_number', trial_number)
        thisExp.addData('stair', stair_idx)
        thisExp.addData('stair_name', thisStair)
        thisExp.addData('step', step)
        thisExp.addData('cond_type', probes_type)
        thisExp.addData('separation', sep)
        thisExp.addData('neg_sep', neg_sep)
        thisExp.addData('ISI', ISI)
        thisExp.addData('isi_dur', isi_dur)
        thisExp.addData('probe_jump', target_jump)
        thisExp.addData('jump_dir', jump_dir)
        thisExp.addData('probeColor1', probeColor1)
        thisExp.addData('probeColor255', probeColor255)
        thisExp.addData('probeLum', probeLum)
        thisExp.addData('trial_response', resp.corr)
        thisExp.addData('corner', corner)
        thisExp.addData('corner_name', corner_name)
        thisExp.addData('probe_ecc', probe_ecc)
        thisExp.addData('resp.rt', resp.rt)
        thisExp.addData('fps', fps)
        thisExp.addData('selected_fps', fps)
        thisExp.addData('orientation', orientation)
        thisExp.addData('vary_fixation', vary_fixation)
        thisExp.addData('monitor_name', monitor_name)
        thisExp.addData('exp_type', exp_type)
        thisExp.addData('expName', expName)

        thisExp.nextEntry()

        thisStair.newValue(resp.corr)  # so that the staircase adjusts itself


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
