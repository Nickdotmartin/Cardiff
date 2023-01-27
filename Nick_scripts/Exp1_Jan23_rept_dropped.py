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
Script to demonstrate Exp1:
ISI of -1 (conc) and 6 frames.
Sep of 0 and 6 pixels.  
'''

# prioritise psychopy
#core.rush(True)

# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# Monitor config from monitor centre
monitor_name = 'Nick_work_laptop'  # 'asus_cal', 'Nick_work_laptop', 'Asus_VG24', 'HP_24uh', 'NickMac', 'Iiyama_2_18',

# Store info about the experiment session
expName = 'Exp1_Jan23_rept_dropped'  # from the Builder filename that created this script
expInfo = {'1. Participant': 'Jan23_rept_dropped',
           '2. Run_number': '1',
           '3. Probe duration in frames at 240hz': [2, 1, 50, 100],
           '4. fps': [60, 240, 120, 60],
           '5. Probe_orientation': ['tangent', 'radial'],
           '6. Vary_fixation': [False, True, False],
           '7. Record_frame_durs': [True, False]
           }


# GUI
dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
if not dlg.OK:
    core.quit()  # user pressed escape

expInfo['date'] = datetime.now().strftime("%d/%m/%Y")
expInfo['time'] = datetime.now().strftime("%H:%M:%S")

# GUI SETTINGS
participant_name = expInfo['1. Participant']
run_number = int(expInfo['2. Run_number'])
probe_duration = int(expInfo['3. Probe duration in frames at 240hz'])
fps = int(expInfo['4. fps'])
orientation = expInfo['5. Probe_orientation']
vary_fixation = eval(expInfo['6. Vary_fixation'])
record_fr_durs = eval(expInfo['7. Record_frame_durs'])

n_trials_per_stair = 2  # 25
probe_ecc = 4

# VARIABLES
'''Distances between probes (spatially and temporally)
For 1probe condition, use separation==99.
For concurrent probes, use ISI==-1.
'''
separations = [5]  # select from [0, 1, 2, 3, 6, 18, 99]
# separations = [0, 1, 2, 3, 6, 18, 99]  # select from [0, 1, 2, 3, 6, 18, 99]
print(f'separations: {separations}')
# ISI_values = [-1]  # select from [-1, 0, 2, 4, 6, 9, 12, 24]
ISI_values = [-1, 0, 2]  # select from [-1, 0, 2, 4, 6, 9, 12, 24]
# ISI_values = [-1, 0, 2, 4, 6, 9, 12, 24]  # select from [-1, 0, 2, 4, 6, 9, 12, 24]
print(f'ISI_values: {ISI_values}')
# repeat separation values for each ISI e.g., [0, 0, 6, 6]
sep_vals_list = list(np.repeat(separations, len(ISI_values)))
print(f'sep_vals_list: {sep_vals_list}')
# ISI_vals_list cycles through ISIs e.g., [-1, 6, -1, 6]
ISI_vals_list = list(np.tile(ISI_values, len(separations)))
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
# filename = f'{_thisDir}{os.sep}' \
#            f'{expName}{os.sep}' \
#            f'{participant_name}{os.sep}' \
#            f'{participant_name}_{run_number}{os.sep}' \
#            f'{participant_name}_{run_number}_output'
# todo: check it is saving correctly, then delete commented out stuff.
save_dir = f'{_thisDir}{os.sep}' \
            f'{expName}{os.sep}' \
            f'{participant_name}{os.sep}' \
            f'{participant_name}_{run_number}{os.sep}'

complete_output_filename = f'{participant_name}_{run_number}_output'
incomplete_output_filename = f'{participant_name}_{run_number}_incomplete'

# files are labelled as '_incomplete' unless entire script runs.
save_output_as = os.path.join(save_dir, incomplete_output_filename)

# Experiment Handler
thisExp = data.ExperimentHandler(name=expName, version=psychopy_version,
                                 extraInfo=expInfo, runtimeInfo=None,
                                 savePickle=None, saveWideText=True,
                                 dataFileName=save_output_as)

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
#todo: check OLED montor name
if monitor_name in ['ASUS_2_13_240Hz', 'asus_cal', 'Nick_work_laptop', 'NickMac', 'Dell_AW3423DW']:
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
# todo: if it is on the OLED, I guess it needs to use rgb, not rgb255
win = visual.Window(monitor=mon, size=(widthPix, heightPix),
                    colorSpace='rgb255', color=bgColor255,
                    winType='pyglet',  # I've added this to make it work on pycharm/mac
                    pos=[1, -1],  # pos gives position of top-left of screen
                    units='pix',
                    screen=display_number,
                    allowGUI=False,
                    fullscr=use_full_screen)


# expected frame duration
expected_fr_sec = 1/fps
expected_fr_ms = expected_fr_sec * 1000
print(f"expected frame duraction: {expected_fr_sec}seconds (or {expected_fr_ms}ms).")

actualFrameRate = int(win.getActualFrameRate())
print(f"actual fps: {type(win.getActualFrameRate())} {win.getActualFrameRate()}")
if abs(fps-actualFrameRate) > 5:
    raise ValueError(f"\nfps ({fps}) does not match actualFrameRate ({actualFrameRate}).")

# todo: get rid of this as it assumes square pixles.
# pixel size
pixel_mm_deg_dict = get_pixel_mm_deg_values(monitor_name=monitor_name)
print('pixel_mm_deg_dict.items()')
for k, v in pixel_mm_deg_dict.items():
    print(k, v)

'''set the max and min frame duration to accept, trials with critial frames beyond these bound will be repeated.'''
# frame error tollerance
# todo: set error tollerance with Simon.
frame_tollerance_prop = .015
max_fr_dur_sec = expected_fr_sec + (expected_fr_sec * frame_tollerance_prop)
min_fr_dur_sec = expected_fr_sec - (expected_fr_sec * frame_tollerance_prop)
win.refreshThreshold = max_fr_dur_sec
# win.refreshThreshold = 1/60 + 0.0003
max_fr_dur_sec = win.refreshThreshold
max_fr_dur_ms = max_fr_dur_sec * 1000
print(f"\nmax_fr_dur_sec ({100 + (100 * frame_tollerance_prop)}%): {max_fr_dur_sec} (or {max_fr_dur_ms}ms)")
print(f"\nmin_fr_dur_sec ({100 - (100 * frame_tollerance_prop)}%): {min_fr_dur_sec} (or {min_fr_dur_sec * 1000}ms)")

# empty variable to store recorded frame durations
exp_fr_intervals = []
exp_n_fr_recorded_list = [0]
exp_n_dropped_fr = 0
dropped_fr_trial_counter = 0
dropped_fr_trial_x_locs = []
user_rpt_trial_x_locs = []

# quit experiment if there are more than 10 trials with dropped frames
# todo: change this back to 10
max_droped_fr_trials = 100


# ELEMENTS
# fixation bull eye
fixation = visual.Circle(win, radius=2, units='pix', lineColor='white', fillColor='black')
# loc_marker = visual.Circle(win, radius=2, units='pix', lineColor='green', fillColor='red')

# PROBEs
probeVert = [(0, 0), (1, 0), (1, 1), (2, 1), (2, -1), (1, -1),
             (1, -2), (-1, -2), (-1, -1), (0, -1)]
probe_size = 1
probe1 = visual.ShapeStim(win, vertices=probeVert, fillColor=(1.0, 1.0, 1.0),
                          lineWidth=0, opacity=1, size=probe_size, interpolate=False)
probe2 = visual.ShapeStim(win, vertices=probeVert, fillColor=[1.0, 1.0, 1.0],
                          lineWidth=0, opacity=1, size=probe_size, interpolate=False)

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
take_break = 5
break_dur = 5
print(f"take_break every {take_break} trials.")
break_text = f"Break\nTurn on the light and take at least {break_dur} seconds break.\n" \
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

too_many_dropped_fr = visual.TextStim(win=win, name='too_many_dropped_fr',
                                      text="The experiment had quit as the computer is dropping frames.\n"
                                           "Sorry for the inconvenience.\n"
                                           "Please contact the experimenter.\n\n"
                                           "Press any key to return to the desktop.",
                                      font='Arial', height=20)

while not event.getKeys():
    fixation.setRadius(3)
    fixation.draw()
    instructions.draw()
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
                          C=stairStart * 0.6,  # initial step size, as prop of reference stim
                          minRevs=3,
                          minTrials=n_trials_per_stair,
                          minVal=miniVal,
                          maxVal=maxiVal,
                          targetThresh=0.75,
                          extraInfo=thisInfo)
    stairs.append(thisStair)

# EXPERIMENT
# the number of the trial for the output file
trial_number = 0

# the actual number of trials including repeated trials (trial_number stays the same for these)
actual_trials_inc_rpt = 0
for step in range(n_trials_per_stair):
    np.random.shuffle(stairs)
    for thisStair in stairs:

        # repeat the trial if [r] has been pressed or frames were dropped
        repeat = True
        while repeat:

            # Trial, stair and step
            trial_number += 1
            actual_trials_inc_rpt += 1
            stair_idx = thisStair.extraInfo['stair_idx']
            print(f"\n({actual_trials_inc_rpt}) trial_number: {trial_number}, stair_idx: {stair_idx}, thisStair: {thisStair}, step: {step}")

            # condition (Separation, ISI)
            sep = sep_vals_list[stair_idx]
            # separation expressed as degrees.
            if -1 < sep < 99:
                sep_deg = sep * pixel_mm_deg_dict['diag_deg']
            else:
                sep_deg = 0
            ISI = ISI_vals_list[stair_idx]
            # print(f"ISI: {ISI}, sep: {sep}")

            # Luminance (staircase varies probeLum)
            probeLum = thisStair.next()
            probeColor255 = int(probeLum * LumColor255Factor)  # rgb255 are ints.
            probeColor1 = (probeColor255 * Color255Color1Factor) - 1
            probe1.color = [probeColor1, probeColor1, probeColor1]
            probe2.color = [probeColor1, probeColor1, probeColor1]
            print(f"probeLum: {probeLum}, probeColor255: {probeColor255}, probeColor1: {probeColor1}")

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

            # # direction in which the probe jumps : CW or CCW (tangent) or expand vs contract (radial)
            target_jump = random.choice([1, -1])
            if orientation == 'tangent':
                jump_dir = 'clockwise'
                if target_jump == -1:
                    jump_dir = 'anticlockwise'
            else:
                jump_dir = 'cont'
                if target_jump == -1:
                    jump_dir = 'exp'
            print(f"corner: {corner} {corner_name}; jump dir: {target_jump} {jump_dir}")

            # shift probes by separation
            '''Both probes should be equally spaced around the meridian point.
            E.g., if sep = 4, probe 1 will be shifted 2 pixels in one direction and 
            probe 2 will be shifted 2 pixels in opposite direction. 
            Where separation is an odd number (e.g., 5), they will be shifted by 2 and 3 pixels; allocated randomly.'''
            if sep == 99:
                p1_shift = p2_shift = 0
            elif sep % 2 == 0:  # even number
                p1_shift = p2_shift = sep // 2
            else:  # odd number
                extra_shifted_pixel = [0, 1]
                np.random.shuffle(extra_shifted_pixel)
                p1_shift = sep // 2 + extra_shifted_pixel[0]
                p2_shift = (sep // 2) + extra_shifted_pixel[1]

            # set position and orientation of probes
            '''NEW - set orientations to p1=zero and p2=180 (not zero), 
            then add the same orientation change to both'''
            probe1_ori = 0
            probe2_ori = 180
            if corner == 45:
                '''in top-right corner, both x and y increase (right and up)'''
                loc_x = dist_from_fix * 1
                loc_y = dist_from_fix * 1
                '''orientation' here refers to the relationship between probes, 
                whereas probe1_ori refers to rotational angle of probe stimulus'''
                if orientation == 'tangent':
                    if target_jump == 1:  # CW
                        probe1_ori += 180
                        probe2_ori += 180
                        probe1_pos = [loc_x - p1_shift, loc_y + p1_shift]
                        probe2_pos = [loc_x + p2_shift - 1, loc_y - p2_shift]
                    elif target_jump == -1:  # ACW
                        probe1_ori += 0
                        probe2_ori += 0
                        probe1_pos = [loc_x + p1_shift, loc_y - p1_shift]
                        probe2_pos = [loc_x - p2_shift + 1, loc_y + p2_shift]
                elif orientation == 'radial':
                    if target_jump == 1:  # inward
                        probe1_ori += 270
                        probe2_ori += 270
                        # probe2 is left and down from probe1
                        probe1_pos = [loc_x + p1_shift, loc_y + p1_shift]
                        probe2_pos = [loc_x - p2_shift + 1, loc_y - p2_shift]
                    elif target_jump == -1:  # outward
                        probe1_ori += 90
                        probe2_ori += 90
                        # probe2 is right and up from probe1
                        probe1_pos = [loc_x - p1_shift, loc_y - p1_shift]
                        probe2_pos = [loc_x + p2_shift - 1, loc_y + p2_shift]
            elif corner == 135:
                loc_x = dist_from_fix * -1
                loc_y = dist_from_fix * 1
                if orientation == 'tangent':
                    if target_jump == 1:  # ACW
                        probe1_ori += 90
                        probe2_ori += 90
                        probe1_pos = [loc_x - p1_shift, loc_y - p1_shift]
                        probe2_pos = [loc_x + p2_shift - 1, loc_y + p2_shift]
                    elif target_jump == -1:  # CW
                        probe1_ori += 270
                        probe2_ori += 270
                        probe1_pos = [loc_x + p1_shift, loc_y + p1_shift]
                        probe2_pos = [loc_x - p2_shift + 1, loc_y - p2_shift]
                elif orientation == 'radial':
                    if target_jump == 1:  # inward
                        probe1_ori += 180
                        probe2_ori += 180
                        # probe2 is right and down from probe1
                        probe1_pos = [loc_x - p1_shift, loc_y + p1_shift]
                        probe2_pos = [loc_x + p2_shift - 1, loc_y - p2_shift]
                    elif target_jump == -1:  # outward
                        probe1_ori += 0
                        probe2_ori += 0
                        # probe2 is left and up from probe1
                        probe1_pos = [loc_x + p1_shift, loc_y - p1_shift]
                        probe2_pos = [loc_x - p2_shift + 1, loc_y + p2_shift]
            elif corner == 225:
                loc_x = dist_from_fix * -1
                loc_y = dist_from_fix * -1
                if orientation == 'tangent':
                    if target_jump == 1:  # CW
                        probe1_ori += 0
                        probe2_ori += 0
                        probe1_pos = [loc_x + p1_shift, loc_y - p1_shift]
                        probe2_pos = [loc_x - p2_shift + 1, loc_y + p2_shift]
                    elif target_jump == -1:  # ACW
                        probe1_ori += 180
                        probe2_ori += 180
                        probe1_pos = [loc_x - p1_shift, loc_y + p1_shift]
                        probe2_pos = [loc_x + p2_shift - 1, loc_y - p2_shift]
                elif orientation == 'radial':
                    if target_jump == 1:  # inward
                        probe1_ori += 90
                        probe2_ori += 90
                        # probe2 is right and up from probe1
                        probe1_pos = [loc_x - p1_shift, loc_y - p1_shift]
                        probe2_pos = [loc_x + p2_shift - 1, loc_y + p2_shift]
                    elif target_jump == -1:  # outward
                        probe1_ori += 270
                        probe2_ori += 270
                        # probe2 is left and down from probe1
                        probe1_pos = [loc_x + p1_shift, loc_y + p1_shift]
                        probe2_pos = [loc_x - p2_shift + 1, loc_y - p2_shift]
            else:
                corner = 315
                loc_x = dist_from_fix * 1
                loc_y = dist_from_fix * -1
                if orientation == 'tangent':
                    if target_jump == 1:  # ACW
                        probe1_ori += 270
                        probe2_ori += 270
                        probe1_pos = [loc_x + p1_shift, loc_y + p1_shift]
                        probe2_pos = [loc_x - p2_shift + 1, loc_y - p2_shift]
                    elif target_jump == -1:  # CW
                        probe1_ori += 90
                        probe2_ori += 90
                        probe1_pos = [loc_x - p1_shift, loc_y - p1_shift]
                        probe2_pos = [loc_x + p2_shift - 1, loc_y + p2_shift]
                elif orientation == 'radial':
                    if target_jump == 1:  # inward
                        probe1_ori += 0
                        probe2_ori += 0
                        # probe2 is left and up from probe1
                        probe1_pos = [loc_x + p1_shift, loc_y - p1_shift]
                        probe2_pos = [loc_x - p2_shift + 1, loc_y + p2_shift]
                    elif target_jump == -1:  # outward
                        probe1_ori += 180
                        probe2_ori += 180
                        # probe2 is right and down from probe1
                        probe1_pos = [loc_x - p1_shift, loc_y + p1_shift]
                        probe2_pos = [loc_x + p2_shift - 1, loc_y - p2_shift]

            # probe1_pos = [loc_x, loc_y]
            # loc_marker.setPos([loc_x, loc_y])

            # print(f"probe1_pos: {probe1_pos}, probe2_pos: {probe2_pos}. dff: {dist_from_fix}")

            probe1.setPos(probe1_pos)
            probe1.setOri(probe1_ori)
            probe2.setPos(probe2_pos)
            probe2.setOri(probe2_ori)


            # VARIABLE FIXATION TIME
            # to reduce anticipatory effects that might arise from fixation always being same length.
            # if False, vary_fix == .5 seconds, so t_fixation is 1 second.
            # if Ture, vary_fix is between 0 and 1 second, so t_fixation is between .5 and 1.5 seconds.
            vary_fix = int(fps / 2)
            if vary_fixation:
                vary_fix = np.random.randint(0, fps)

            # timing in frames for ISI and probe2
            # If probes are presented concurrently, set ISI and probe2 to last for 0 frames.
            isi_dur_fr = ISI
            p2_fr = probe_duration
            if ISI < 0:
                isi_dur_fr = p2_fr = 0

            # cumulative timing in frames for each part of a trial
            t_fixation = int(fps / 2) + vary_fix
            t_probe_1 = t_fixation + probe_duration
            t_ISI = t_probe_1 + isi_dur_fr
            t_probe_2 = t_ISI + p2_fr
            t_response = t_probe_2 + 10000 * fps  # ~40 seconds to respond

            print(f"t_fixation: {t_fixation}, t_probe_1: {t_probe_1}, t_ISI: {t_ISI}, t_probe_2: {t_probe_2}, t_response: {t_response}\n")

            # repeat the trial if [r] has been pressed
            # repeat = True
            # while repeat:
            #     frameN = -1

            # continue_routine refers to flipping the screen to show next frame

            # take a break every ? trials
            # if (trial_number % take_break == 1) & (trial_number > 1):
            if (actual_trials_inc_rpt % take_break == 1) & (actual_trials_inc_rpt > 1):
                print("\nTaking a break.\n")
                # continueRoutine = False
                breaks.text = break_text + f"\n{trial_number-1}/{total_n_trials} trials completed."
                breaks.draw()
                win.flip()
                event.clearEvents(eventType='keyboard')
                core.wait(secs=5)
                event.clearEvents(eventType='keyboard')
                breaks.text = break_text + "\n\nPress any key to continue."
                breaks.draw()
                win.flip()
                # continue_after_break.draw()

                while not event.getKeys():
                    # continue the breaks routine until a key is pressed
                    continueRoutine = True
            else:
                # else continue the trial routine.
                continueRoutine = True

            # initialise frame number
            frameN = -1
            while continueRoutine:
                frameN = frameN + 1


                # recording frame durations - from t_fixation (1 frame before p1), until 1 frame after p2.
                if frameN == t_fixation:
                    # start recording frame intervals
                    if record_fr_durs:
                        win.recordFrameIntervals = True
                        print(f"{frameN}: win.recordFrameIntervals : {win.recordFrameIntervals}")

                if frameN == t_probe_2 + 1:
                    if record_fr_durs:
                        win.recordFrameIntervals = False
                        print(f"{frameN}: win.recordFrameIntervals : {win.recordFrameIntervals}")


                '''Experiment timings'''
                # FIXATION
                if t_fixation >= frameN > 0:
                    fixation.setRadius(3)
                    fixation.draw()

                    # loc_marker.draw()

                    # reset timer to start with probe1 presentation.
                    resp.clock.reset()


                # PROBE 1
                elif t_probe_1 >= frameN > t_fixation:
                    probe1.draw()

                    if ISI == -1:  # SIMULTANEOUS CONDITION (concurrent)
                        if sep <= 18:  # don't draw 2nd probe in 1probe cond (sep==99)
                            probe2.draw()
                    fixation.setRadius(3)
                    fixation.draw()

                    # loc_marker.draw()

                # ISI
                elif t_ISI >= frameN > t_probe_1:
                    fixation.setRadius(3)
                    fixation.draw()

                    # loc_marker.draw()

                # PROBE 2
                elif t_probe_2 >= frameN > t_ISI:
                    if ISI >= 0:
                        if sep <= 18:  # don't draw 2nd probe in 1probe cond (sep==99)
                            probe2.draw()
                    fixation.setRadius(3)
                    fixation.draw()

                    # loc_marker.draw()

                # ANSWER
                elif frameN > t_probe_2:

                    fixation.setRadius(2)
                    fixation.draw()

                    # loc_marker.draw()

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

                        '''Get frame intervals for this trial, add to experiment and empty cache'''
                        if record_fr_durs:
                            # get trial frameIntervals details
                            trial_fr_intervals = win.frameIntervals
                            n_fr_recorded = len(trial_fr_intervals)
                            trial_n_dropped_fr = win.nDroppedFrames
                            print(f"n_fr_recorded: {n_fr_recorded}, trial_n_dropped_fr: {trial_n_dropped_fr}, "
                                  f"trial_fr_intervals: {trial_fr_intervals}")

                            # add to experiment info.
                            exp_fr_intervals += trial_fr_intervals
                            exp_n_fr_recorded_list.append(exp_n_fr_recorded_list[-1] + n_fr_recorded)
                            exp_n_dropped_fr += trial_n_dropped_fr
                            print(f"exp_n_fr_recorded_list: {exp_n_fr_recorded_list}, "
                                  f"exp_n_dropped_fr: {exp_n_dropped_fr}")
                            print(f"exp_fr_intervals ({len(exp_fr_intervals)}): {exp_fr_intervals}")

                            # check for dropped frames (or frames that are too short)
                            if max(trial_fr_intervals) > max_fr_dur_sec or min(trial_fr_intervals) < min_fr_dur_sec:
                                if max(trial_fr_intervals) > max_fr_dur_sec:
                                    print(f"\n\toh no! Frame too long! {max(trial_fr_intervals)} > {max_fr_dur_sec}")
                                elif min(trial_fr_intervals) < min_fr_dur_sec:
                                    print(f"\n\toh no! Frame too short! {min(trial_fr_intervals)} < {min_fr_dur_sec}")
                                repeat = True
                                dropped_fr_trial_counter += 1
                                trial_number -= 1
                                win.frameIntervals = []
                                continueRoutine = False
                                trial_x_locs = [exp_n_fr_recorded_list[-2], exp_n_fr_recorded_list[-1]]
                                dropped_fr_trial_x_locs.append(trial_x_locs)
                                continue

                            # empty frameIntervals cache
                            win.frameIntervals = []


                        # these belong to the end of the answers section
                        repeat = False
                        continueRoutine = False

                # regardless of frameN, check for quit
                if event.getKeys(keyList=["escape"]):
                    thisExp.close()
                    core.quit()

                # If too many trials have had droppped frames, quit experiment
                if dropped_fr_trial_counter > max_droped_fr_trials:
                    while not event.getKeys():
                        # display end of experiment screen
                        too_many_dropped_fr.draw()
                        win.flip()
                    else:
                        # close and quit once a key is pressed
                        # thisExp.abort()  # or data files will save again on exit
                        thisExp.close()
                        win.close()
                        core.quit()

                # redo the trial if I think I made a mistake
                # todo: this repeats current trial, which might be too late if they wanted to repeat the one before...
                if event.getKeys(keyList=["r"]) or event.getKeys(keyList=['num_9']):
                    print("\n\tparticipant pressed repeat.")
                    trial_x_locs = [exp_n_fr_recorded_list[-1], exp_n_fr_recorded_list[-1] + n_fr_recorded]
                    user_rpt_trial_x_locs.append(trial_x_locs)
                    repeat = True
                    trial_number -= 1
                    continueRoutine = False
                    continue


                # refresh the screen
                if continueRoutine:
                    win.flip()

        thisExp.addData('trial_number', trial_number)
        thisExp.addData('trial_n_inc_rpt', actual_trials_inc_rpt)
        thisExp.addData('stair', stair_idx)
        thisExp.addData('stair_name', thisStair)
        thisExp.addData('step', step)
        thisExp.addData('separation', sep)
        thisExp.addData('sep_deg', sep_deg)
        thisExp.addData('ISI', ISI)
        thisExp.addData('isi_dur_fr', isi_dur_fr)
        thisExp.addData('isi_ms', (1000 / fps) * isi_dur_fr)
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
        thisExp.addData('orientation', orientation)
        thisExp.addData('vary_fixation', vary_fixation)
        thisExp.addData('t_fixation', t_fixation)
        thisExp.addData('expName', expName)
        thisExp.addData('monitor_name', monitor_name)
        thisExp.addData('selected_fps', fps)
        thisExp.addData('actual_fps', actualFrameRate)
        thisExp.addData('psychopy_version', psychopy_version)
        thisExp.addData('date', expInfo['date'])
        thisExp.addData('time', expInfo['time'])

        thisExp.nextEntry()

        thisStair.newValue(resp.corr)   # so that the staircase adjusts itself

print("\nend of experiment loop, saving data\n")
# thisExp.dataFileName = filename
thisExp.dataFileName = os.path.join(save_dir, complete_output_filename)
thisExp.close()

# plot frame intervals
if record_fr_durs:
    import matplotlib.pyplot as plt

    total_recorded_fr = len(exp_fr_intervals)
    # exp_n_dropped_fr = win.nDroppedFrames
    print(f"{exp_n_dropped_fr}/{total_recorded_fr} dropped in total (expected: {round(expected_fr_ms, 2)}ms, 'dropped' if > {round(max_fr_dur_ms, 2)})")
    plt.plot(exp_fr_intervals)
    plt.title(f"{monitor_name}, {fps}Hz, {expInfo['date']}\n{exp_n_dropped_fr}/{total_recorded_fr} dropped fr (expected: {round(expected_fr_ms, 2)}ms, 'dropped' if > {round(max_fr_dur_ms, 2)})")

    # add vertical lines to signify trials
    fr_v_lines = [i - .5 for i in exp_n_fr_recorded_list]
    plt.vlines(x=fr_v_lines, ymin=min(exp_fr_intervals), ymax=max(exp_fr_intervals), colors='silver', linestyles='dashed')

    # add green horizontal lines to signify expected frame duration
    plt.axhline(y=expected_fr_sec, color='green', linestyle='dashed')
    
    # add red horizontal lines to signify frame error tollerance
    plt.axhline(y=max_fr_dur_sec, color='red', linestyle='dashed')
    plt.axhline(y=min_fr_dur_sec, color='red', linestyle='dashed')

    # shade trials that were repeated due to dropped frames in red
    for loc_pair in dropped_fr_trial_x_locs:
        print(loc_pair)
        x0, x1 = loc_pair[0] - .5, loc_pair[1] - .5
        plt.axvspan(x0, x1, color='red', alpha=0.15, zorder=0, linewidth=None)

    # shade trials that were repeated due to user pressing repeat in orange
    for loc_pair in user_rpt_trial_x_locs:
        x0, x1 = loc_pair[0] - .5, loc_pair[1] - .5
        plt.axvspan(x0, x1, color='orange', alpha=0.15, zorder=0, linewidth=None)
    fig_name = f'{participant_name}_{run_number}_frames.png'
    print(f"fig_name: {fig_name}")
    plt.savefig(os.path.join(save_dir, fig_name))


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
