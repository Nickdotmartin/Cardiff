from __future__ import division  # do I need this?
from psychopy import gui, visual, core, data, event, monitors
from psychopy import __version__ as psychopy_version
import os
import numpy as np
import random
import copy
from datetime import datetime
from math import tan, sqrt
from kestenSTmaxVal import Staircase
from PsychoPy_tools import get_pixel_mm_deg_values


'''
Based on Exp1_Jan23.py but with function to make new Ricco Stim from Ricco_v5.
'''


def make_ricco_vertices(sep_cond, balanced=False, verbose=False):
    """
    Probe vertices can be constructed from four parts.
        1. the top left edge of probe 1 (which is the same for all conds).
        2. zigzag down top-right side (which is has more vertices as sep_cond increases).
        3. bottom-right of probe 2 (calculated by adjusting values from sep0).
        4. zigzag back up bottom-left side (which is has more vertices as sep_cond increases).

    For 1probe condition (sep=99 or -1) it just loads vertices rather than generating them.

    :param sep_cond: equivalent separation condition from exp 1.
    :param balanced: (default = False).  If False, (0, 0) is at the bottom of the middle part of the 'M' of probe1 for all probes.
                    In other words, they are not evenly spread around (0, 0), probe2 is further away from (0, 0) for higher sep values.
                    This is consistent with Exp1 stimuli, where probe1 is always in the same spo, regardless of sep.
                    If True, stimuli are balanced around (0, 0), as was the case for previous Ricco experiments.
    :param verbose: print sections to screen as they are generated

    :return: vertices to draw probe.
    """

    '''top-left of pr1: Use these vertices for all probes'''
    tl_pr1_1 = [(-2, 1), (-1, 1), (-1, 2), (1, 2)]  # top-left of pr1

    if sep_cond in [-1, 99]:
        '''1probe condition, just load vertices'''
        tr_zz_2 = [(1, 1)]
        br_pr2_3 = [(0, 1), (0, 0), (-1, 0), (-1, -1)]
        bl_zz_4 = [(-2, -1)]

    else:  # if not 1probe (sep not in [-1, 99])

        '''zig-zag down top-right: 
        for tr_zz_2, generate x and y values based on separation, then zip.'''
        # tr_zz_2_x_vals start from 1 (once), and then repeat each number (twice) up to sep_cond+1 (once).
        tr_zz_2_x_vals = list(np.repeat(list(range(1, sep_cond+2)), 2))[1:-1]

        # tr_zz_2_y_vals start from zero (twice) and repeat each number (twice) down to -sep_cond+1.
        tr_zz_2_y_vals = list(np.repeat(list(range(0, -sep_cond, -1)), 2))

        # zip x and y together to make list of tuples
        tr_zz_2 = list(zip(tr_zz_2_x_vals, tr_zz_2_y_vals))

        '''bottom-right of pr2: use the values from sep0 as the default and adjust based on sep_cond'''
        br_pr2_sep0 = [(1, -1), (0, -1), (0, -2), (-2, -2)]
        br_pr2_3 = [(i[0]+sep_cond, i[1]-sep_cond) for i in br_pr2_sep0]

        '''zig-zag back up bottom-left side:
        For bl_zz_4_x_vals, generate x and y values based on separation, then zip.'''
        # bl_zz_4_x_vals has the same structure as tr_zz_2_x_vals:
        #   first value, once, then each number repeats twice apart from last one (once).
        # bl_zz_4_x_vals start positive and decrement until -2.
        bl_zz_4_x_vals = list(np.repeat(list(range(-2+sep_cond, -3, -1)), 2))
        bl_zz_4_x_vals = bl_zz_4_x_vals[1:-1]

        # bl_zz_4_y_vals start from -1-sep_cond (twice) and repeat each number (twice) up to -2.
        # print(f"tr_zz_2_y_vals: {tr_zz_2_y_vals}")
        bl_zz_4_y_vals = list(np.repeat(list(range(-1-sep_cond, -1)), 2))

        # zip x and y together to make list of tuples
        bl_zz_4 = list(zip(bl_zz_4_x_vals, bl_zz_4_y_vals))
        # print(f"bl_zz_4: {bl_zz_4}")

    if verbose:
        print(f"\nsep_cond: {sep_cond}")
        print(f"tl_pr1_1: {tl_pr1_1}")
        print(f"tr_zz_2: {tr_zz_2}")
        print(f"br_pr2_3: {br_pr2_3}")
        print(f"bl_zz_4: {bl_zz_4}")

    new_vertices = tl_pr1_1 + tr_zz_2 + br_pr2_3 + bl_zz_4

    if balanced:
        # balancing is roughly based on half the separation value, but with slight differences for odd and even numbers.
        if verbose:
            print('balancing probe around (0, 0)')
        if sep_cond in [-1, 99]:
            half_sep = 0
        elif (sep_cond % 2) != 0:
            half_sep = int(sep_cond / 2) + 1
        else:
            half_sep = int(sep_cond / 2)

        balanced_vertices = [(tup[0] - (half_sep - 1), tup[1] + half_sep) for tup in new_vertices]

        new_vertices = balanced_vertices

    return new_vertices


# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# Monitor config from monitor centre
monitor_name = 'asus_cal'  # 'asus_cal', 'Nick_work_laptop', 'Asus_VG24', 'HP_24uh', 'NickMac', 'Iiyama_2_18',

# Use balanced probes to match previous Ricco experiments
balanced_probes = True

# Store info about the experiment session
expName = 'Exp3_Ricco_NM_v6'

expInfo = {'1. Participant': 'Nick_test',
           '2. Run_number': '1',
           '3. Probe duration in frames at 240hz': [2, 50, 100, 360],
           '4. fps': [240, 120, 60],
           '5. Probe_orientation': ['tangent', 'radial'],
           '6. Vary_fixation': [True, False],
           '7. Record_frame_durs': [False, True]
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
probe_duration = int(expInfo['3. Probe duration in frames at 240hz'])
fps = int(expInfo['4. fps'])
orientation = expInfo['5. Probe_orientation']
vary_fixation = eval(expInfo['6. Vary_fixation'])
record_fr_durs = eval(expInfo['7. Record_frame_durs'])

# expected frame duration
expected_fr_ms = (1/fps) * 1000

# VARIABLES
n_trials_per_stair = 25
probe_ecc = 4

'''Distances between probes (spatially and temporally)
For 1probe condition, use separation==-1.
For concurrent probes, use ISI==-1.
'''
separation_values = [0, 2, 4, 6, 8, 10, 12, 18]
print(f'separation_values: {separation_values}')
cond_types = ['lines']
print(f'cond_types: {cond_types}')
# repeat separation values three times so one of each e.g., [-1, -1, -1, 0, 0, 0, ...]
sep_vals_list = list(np.repeat(separation_values, len(cond_types)))
print(f'sep_vals_list: {sep_vals_list}')
n_stairs = len(sep_vals_list)
print(f'n_stairs: {n_stairs}')
# cond type list cycles through conds e.g., ['2probe', 'lines', 'circles', '2probe', 'lines', 'circles'...]
cond_type_list = list(np.tile(cond_types, len(separation_values)))
print(f'cond_type_list: {cond_type_list}')
# stair_names_list joins sep_vals_list and cond_type_list
# e.g., ['-1_2probe', '-1_lines', '-1_circles', '0_2probe', '0_lines', '0_circles'...]
stair_names_list = [f'{s}_{c}' for s, c in zip(sep_vals_list, cond_type_list)]
print(f'stair_names_list: {stair_names_list}')
n_stairs = len(sep_vals_list)
print(f'n_stairs: {n_stairs}')

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
Color255Color1Factor = 1 / 127.5  # Color255 * Color255Color1Factor -1
# Lum to Color1
Color1LumFactor = 2.39538706913372

maxLum = 106  # 255 RGB
bgLumProp = .2
bgLum = maxLum * bgLumProp
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
viewdist = mon_dict['dist']  # viewing distance in cm
viewdistPix = widthPix / monitorwidth*viewdist
mon = monitors.Monitor(monitor_name, width=monitorwidth, distance=viewdist)
mon.setSizePix((widthPix, heightPix))

# WINDOW SPEC
win = visual.Window(monitor=mon, size=(widthPix, heightPix),
                    colorSpace='rgb255', color=bgColor255,
                    winType='pyglet',  # I've added this to make it work on pycharm/mac
                    pos=[1, -1],  # pos gives position of top-left of screen
                    units='pix',
                    screen=display_number,
                    allowGUI=False,
                    fullscr=use_full_screen)

# refresh rate
actual_fps = win.getActualFrameRate(nIdentical=240, nMaxFrames=240,
                                    nWarmUpFrames=10, threshold=1)
print(f'actual_fps: {actual_fps}')

# pixel size
pixel_mm_deg_dict = get_pixel_mm_deg_values(monitor_name=monitor_name)
print('pixel_mm_deg_dict.items()')
for k, v in pixel_mm_deg_dict.items():
    print(k, v)

# ELEMENTS
# fixation bull eye
fixation = visual.Circle(win, radius=2, units='pix', lineColor='white', fillColor='black')

# PROBEs
probe_vert_list = []
probe_name_list = []
vert_dict = {}
for sep_cond in separation_values:
    probe_vertices = make_ricco_vertices(sep_cond, balanced=balanced_probes)
    if sep_cond == 99:
        len_pix = 1.5
        n_pix = 5
    else:
        len_pix = 2.5 + sep_cond
        n_pix = sep_cond * 5 + 10

    probe_name = f"sep{sep_cond}"
    vert_dict[probe_name] = {}
    vert_dict[f"sep{sep_cond}"]['n_pix'] = n_pix
    vert_dict[f"sep{sep_cond}"]['len_pix'] = len_pix
    vert_dict[f"sep{sep_cond}"]['diag_mm'] = len_pix * pixel_mm_deg_dict['diag_mm']
    vert_dict[f"sep{sep_cond}"]['diag_deg'] = len_pix * pixel_mm_deg_dict['diag_deg']
    vert_dict[f"sep{sep_cond}"]['vertices'] = probe_vertices

    probe_name_list.append(probe_name)
    probe_vert_list.append(probe_vertices)
for k, v in vert_dict.items():
    print(k, v)

# dist_from_fix is a constant to get 4dva distance from fixation,
dist_from_fix = round((tan(np.deg2rad(probe_ecc)) * viewdistPix) / sqrt(2))

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


# BREAKS
take_break = 76
total_n_trials = int(n_trials_per_stair * n_stairs)
print(f"take_break every {take_break} trials.")
break_text = "Break\nTurn on the light and take at least 30-seconds break.\n" \
             "Keep focussed on the fixation circle in the middle of the screen.\n" \
             "Remember, if you don't see the target, just guess!"
breaks = visual.TextStim(win=win, name='breaks',
                         text=break_text,
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
    win.flip()

# frame error tolerance
frame_err_sec = win.refreshThreshold
frame_err_ms = frame_err_sec * 1000
print(f"frame_err_sec (120%): {frame_err_sec} (or {frame_err_ms}ms)")
fr_recorded_list = []

# STAIRCASE
expInfo['stair_list'] = list(range(n_stairs))
expInfo['n_trials_per_stair'] = n_trials_per_stair

stairStart = maxLum
miniVal = bgLum  # 21.2
maxiVal = maxLum  # 106

print('\nexpInfo (dict)')
for k, v in expInfo.items():
    print(f"{k}: {v}")

stairs = []
for stair_idx in expInfo['stair_list']:

    thisInfo = copy.copy(expInfo)
    thisInfo['stair_idx'] = stair_idx

    thisStair = Staircase(name=stair_names_list[stair_idx],
                          type='simple',
                          value=stairStart,
                          C=stairStart * 0.6,  # initial step size, as prop of reference stim
                          minRevs=3,
                          minTrials=n_trials_per_stair,
                          minVal=miniVal,
                          maxVal=maxiVal,
                          targetThresh=0.75,
                          extraInfo=thisInfo)
    stairs.append(thisStair)

# EXPERIMENT
trial_number = 0
for step in range(n_trials_per_stair):
    np.random.shuffle(stairs)
    for thisStair in stairs:

        # Trial, stair and step
        trial_number = trial_number + 1
        stair_idx = thisStair.extraInfo['stair_idx']
        sep = sep_vals_list[stair_idx]
        cond_type = cond_type_list[stair_idx]

        print(f"thisStair: {thisStair}, step: {step}, trial_number: {trial_number}")
        print(f"stair_idx: {stair_idx}, cond_type: {cond_type}, sep: {sep}")

        # sort out stimuli by cond_type - load correct size stim based on sep
        if cond_type == 'lines':
            # there are 7 probe_verts (vertices), but 21 staircases, so I've repeated them to match up
            probe_vert_list_2 = list(np.repeat(probe_vert_list, len(cond_types)))
            probeVert = probe_vert_list_2[stair_idx]
            line_probe = visual.ShapeStim(win, vertices=probeVert, fillColor=(1.0, 1.0, 1.0),
                                          lineWidth=0, opacity=1, size=1, interpolate=False, )
            probe1 = line_probe
        else:
            raise ValueError(f'Unknown cond type: {cond_type}')

        # Luminance (staircase varies probeLum)
        probeLum = thisStair.next()
        probeColor255 = int(probeLum * LumColor255Factor)  # rgb255 are ints.
        probeColor1 = (probeColor255 * Color255Color1Factor) - 1
        probe1.color = [probeColor1, probeColor1, probeColor1]
        print(f"probeLum: {probeLum}, probeColor255: {probeColor255}, probeColor1: {probeColor1}")

        weber_lum = (probeLum-bgLum)/probeLum
        print(f'\t\tbgLum: {bgLum}, weber_lum: {weber_lum}')


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

        # set probe ori
        if corner == 45:
            # in top-right corner, both x and y increase (right and up)
            p1_x = dist_from_fix * 1
            p1_y = dist_from_fix * 1
            #  'orientation' here refers to the relationship between probes,
            #  whereas probe1.ori refers to rotational angle of probe stimulus
            if orientation == 'tangent':
                if target_jump == 1:  # CCW
                    probe1.ori = 0
                elif target_jump == -1:  # CW
                    probe1.ori = 180
            if orientation == 'radial':
                if target_jump == 1:  # CCW
                    probe1.ori = 90
                elif target_jump == -1:  # CW
                    probe1.ori = 270
        elif corner == 135:
            # in top-left corner, x decreases (left) and y increases (up)
            p1_x = dist_from_fix * -1
            p1_y = dist_from_fix * 1
            if orientation == 'tangent':
                if target_jump == 1:  # CCW
                    probe1.ori = 90
                elif target_jump == -1:  # CW
                    probe1.ori = 270
            if orientation == 'radial':
                if target_jump == 1:  # CCW
                    probe1.ori = 180
                elif target_jump == -1:  # CW
                    probe1.ori = 0
        elif corner == 225:
            # in bottom left corner, both x and y decrease (left and down)
            p1_x = dist_from_fix * -1
            p1_y = dist_from_fix * -1
            if orientation == 'tangent':
                if target_jump == 1:  # CCW
                    probe1.ori = 180
                elif target_jump == -1:  # CW
                    probe1.ori = 0
            if orientation == 'radial':
                if target_jump == 1:  # CCW
                    probe1.ori = 270
                elif target_jump == -1:  # CW
                    probe1.ori = 90
        else:
            corner = 315
            # in bottom-right corner, x increases (right) and y decreases (down)
            p1_x = dist_from_fix * 1
            p1_y = dist_from_fix * -1
            if orientation == 'tangent':
                if target_jump == 1:  # CCW
                    probe1.ori = 270
                elif target_jump == -1:  # CW
                    probe1.ori = 90
            if orientation == 'radial':
                if target_jump == 1:  # CCW
                    probe1.ori = 0
                elif target_jump == -1:  # CW
                    probe1.ori = 180

        probe1.pos = [p1_x, p1_y]

        print(f"probe1: {probe1.pos}, dist_from_fix: {dist_from_fix}")

        # VARIABLE FIXATION TIME
        # to reduce anticipatory effects that might arise from fixation always being same length.
        # if False, vary_fix == .5 seconds, so t_fixation is 1 second.
        # if Ture, vary_fix is between 0 and 1 second, so t_fixation is between .5 and 1.5 seconds.
        vary_fix = int(fps / 2)
        if vary_fixation:
            vary_fix = np.random.randint(0, fps)

        # cumulative timing in frames for each part of a trial
        t_fixation = int(fps / 2) + vary_fix
        t_probe_1 = t_fixation + probe_duration
        t_response = t_probe_1 + 10000 * fps  # essentially unlimited time to respond

        print(f"t_fixation: {t_fixation}\n"
              f"t_probe_1: {t_probe_1}\n"
              f"t_response: {t_response}\n")

        # repeat the trial if [r] has been pressed
        repeat = True
        while repeat:
            frameN = -1

            # take a break every ? trials
            if (trial_number % take_break == 1) & (trial_number > 1):
                continueRoutine = False
                breaks.text = break_text + f"\n{trial_number}/{total_n_trials} trials completed."
                breaks.draw()
                win.flip()
                while not event.getKeys():
                    continueRoutine = True
            else:
                continueRoutine = True

            while continueRoutine:
                frameN = frameN + 1

                # probe1.ori = probe1.ori+1

                # FIXATION
                if t_fixation >= frameN > 0:
                    fixation.setRadius(3)
                    fixation.draw()

                    # reset timer to start with probe1 presentation.
                    resp.clock.reset()

                    # start recording frame intervals
                    if record_fr_durs:
                        win.recordFrameIntervals = True


                # PROBE 1
                elif t_probe_1 >= frameN > t_fixation:
                    fixation.setRadius(3)
                    fixation.draw()
                    probe1.draw()

                # ANSWER
                elif frameN > t_probe_1:
                    
                    if record_fr_durs:
                        win.recordFrameIntervals = False
                        total_recorded_fr = len(win.frameIntervals)
                        fr_recorded_list.append(total_recorded_fr)

                    
                    fixation.setRadius(2)
                    fixation.draw()

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
        thisExp.addData('cond_type', cond_type)
        thisExp.addData('separation', sep)
        thisExp.addData('ISI', 0)
        thisExp.addData('probe_jump', target_jump)
        thisExp.addData('jump_dir', jump_dir)
        thisExp.addData('trial_response', resp.corr)
        thisExp.addData('corner', corner)
        thisExp.addData('corner_name', corner_name)
        thisExp.addData('resp.rt', resp.rt)
        thisExp.addData('probeLum', probeLum)
        thisExp.addData('probeColor1', probeColor1)
        thisExp.addData('probeColor255', probeColor255)
        thisExp.addData('bgLum', bgLum)
        thisExp.addData('bgColor255', bgColor255)
        thisExp.addData('weber_lum', weber_lum)
        thisExp.addData('n_pix', vert_dict[f"sep{sep}"]['n_pix'])
        thisExp.addData('len_pix', vert_dict[f"sep{sep}"]['len_pix'])
        thisExp.addData('diag_mm', vert_dict[f"sep{sep}"]['diag_mm'])
        thisExp.addData('diag_deg', vert_dict[f"sep{sep}"]['diag_deg'])
        thisExp.addData('probe_ecc', probe_ecc)
        thisExp.addData('orientation', orientation)
        thisExp.addData('vary_fixation', vary_fixation)
        thisExp.addData('t_fixation', t_fixation)
        thisExp.addData('expName', expName)
        thisExp.addData('monitor_name', monitor_name)
        thisExp.addData('selected_fps', fps)
        thisExp.addData('actual_fps', actual_fps)
        thisExp.addData('balanced_probes', balanced_probes)
        thisExp.addData('psychopy_version', psychopy_version)
        thisExp.addData('date', expInfo['date'])
        thisExp.addData('time', expInfo['time'])

        thisExp.nextEntry()

        thisStair.newValue(resp.corr)   # so that the staircase adjusts itself

print("end of experiment loop, saving data")
thisExp.dataFileName = filename
thisExp.close()

# plot frame intervals
if record_fr_durs:
    import matplotlib.pyplot as plt
    total_recorded_fr = len(win.frameIntervals)
    total_dropped_fr = win.nDroppedFrames
    print(f"{total_dropped_fr}/{total_recorded_fr} dropped in total (expected: {round(expected_fr_ms, 2)}ms, "
          f"'dropped' if > {round(frame_err_ms, 2)})")
    plt.plot(win.frameIntervals)
    plt.title(f"{monitor_name}, {fps}Hz, {expInfo['date']}\n{total_dropped_fr}/{total_recorded_fr} dropped fr "
              f"(expected: {round(expected_fr_ms, 2)}ms, 'dropped' if > {round(frame_err_ms, 2)})")
    plt.vlines(x=fr_recorded_list, ymin=min(win.frameIntervals), ymax=max(win.frameIntervals), 
               colors='silver', linestyles='dashed')
    plt.axhline(y=frame_err_sec, color='red', linestyle='dashed')
    fig_name = filename = f'{_thisDir}{os.sep}' \
                          f'{expName}{os.sep}' \
                          f'{participant_name}{os.sep}' \
                          f'{participant_name}_{run_number}{os.sep}' \
                          f'{participant_name}_{run_number}_frames.png'
    print(f"fig_name: {fig_name}")
    plt.savefig(fig_name)


while not event.getKeys():
    # display end of experiment screen
    end_of_exp.draw()
    win.flip()
else:
    # logging.flush()  # write messages out to all targets
    thisExp.abort()  # or data files will save again on exit

    # close and quit once a key is pressed
    win.close()
    core.quit()
