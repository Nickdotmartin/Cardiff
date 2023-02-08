from __future__ import division  # do I need this?
from psychopy import gui, visual, core, data, event, monitors
from psychopy import __version__ as psychopy_version
import os
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from datetime import datetime
from math import tan, sqrt
from kestenSTmaxVal import Staircase
from PsychoPy_tools import get_pixel_mm_deg_values
from exp1a_psignifit_analysis import fig_colours

'''
Jan23 version of experiment 1 - radial to compare in vs out.  
Now has code to measure frame duration and repeat any trials with bad frame timings.
Also has an enforced break, with core.wait() to do cpu housekeeping.
Also has core.wait() when closing to hopefully reduce hanging.
'''

# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# Monitor config from monitor centre
monitor_name = 'Nick_work_laptop'  # 'asus_cal', 'Nick_work_laptop', 'Asus_VG24', 'HP_24uh', 'NickMac', 'Iiyama_2_18',
# todo: put probe size and conditions back

# Store info about the experiment session (numbers keep the order)
expName = 'Exp1_Jan23_radial_v4'  # from the Builder filename that created this script
expInfo = {'1. Participant': 'Nick_fr_test',
           '2. Run_number': '1',
           '3. separation (pixels)': [5, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 36],
           '4. Probe duration in frames': [1, 2, 1, 50, 100],
           '5. fps': [120, 240, 120, 60],
           '6. Vary_fixation': [True, False],
           '7. Record_frame_durs': [True, False]
           }


# dialogue box
dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
if not dlg.OK:
    core.quit()  # user pressed escape

# dialogue box settings
participant_name = expInfo['1. Participant']
run_number = int(expInfo['2. Run_number'])
this_sep_value = int(expInfo['3. separation (pixels)'])
probe_duration = int(expInfo['4. Probe duration in frames'])
fps = int(expInfo['5. fps'])
vary_fixation = eval(expInfo['6. Vary_fixation'])
record_fr_durs = eval(expInfo['7. Record_frame_durs'])

# VARIABLES
orientation = 'radial'
n_trials_per_stair = 25
probe_ecc = 4
expInfo['time'] = datetime.now().strftime("%H:%M:%S")
expInfo['date'] = datetime.now().strftime("%d/%m/%Y")

'''Distances between probes (spatially and temporally)
For 1probe condition, use separation==99.
For concurrent probes, use ISI==-1.
'''
separations = [this_sep_value]  # select from [0, 1, 2, 3, 6, 18, 99]
print(f'separations: {separations}')
# ISI_values = [-1]  # select from [-1, 0, 2, 4, 6, 9, 12, 24]
ISI_values = [-1, 0, 2]  #, 4, 6, 9, 12, 24]  # select from [-1, 0, 2, 4, 6, 9, 12, 24]
print(f'ISI_values: {ISI_values}')

probe_dirs = ['cont', 'exp']
print(f'probe_dirs: {probe_dirs}')

# repeat separation values for each ISI e.g., [0, 0, 6, 6]
sep_vals_list = list(np.repeat(separations, len(ISI_values))) * len(probe_dirs)
print(f'sep_vals_list: {sep_vals_list}')
ISI_vals_list = list(np.tile(ISI_values, len(separations) * len(probe_dirs)))
print(f'ISI_vals_list: {ISI_vals_list}')
probes_dirs_list = list(np.repeat(probe_dirs, len(sep_vals_list) / len(probe_dirs)))
print(f'probes_dirs_list: {probes_dirs_list}')

# stair_names_list joins sep_vals_list and ISI_vals_list
# e.g., ['sep0_ISI-1', 'sep0_ISI6', 'sep6_ISI-1', 'sep6_ISI6']
# stair_names_list = [f'sep{s}_ISI{c}' for s, c in zip(sep_vals_list, ISI_vals_list)]
stair_names_list = [f'{p}_sep{s}_ISI{i}' for p, s, i in zip(probes_dirs_list, sep_vals_list, ISI_vals_list)]
print(f'stair_names_list: {stair_names_list}')
n_stairs = len(sep_vals_list)
print(f'n_stairs: {n_stairs}')
total_n_trials = int(n_trials_per_stair * n_stairs)
print(f'total_n_trials: {total_n_trials}')

# FILENAME
save_dir = f'{_thisDir}{os.sep}' \
           f'{expName}{os.sep}' \
           f'{participant_name}{os.sep}' \
           f'{participant_name}_{run_number}{os.sep}' \
           f'sep_{this_sep_value}{os.sep}'
# files are labelled as '_incomplete' unless entire script runs.
incomplete_output_filename = f'{participant_name}_{run_number}_incomplete'
save_output_as = os.path.join(save_dir, incomplete_output_filename)


# Experiment Handler
thisExp = data.ExperimentHandler(name=expName, version=psychopy_version,
                                 extraInfo=expInfo, runtimeInfo=None,
                                 savePickle=None, saveWideText=True,
                                 dataFileName=save_output_as)

# Monitor details: colour, luminance, pixel size and frame rate
print(f"monitor_name: {monitor_name}")
thisMon = monitors.Monitor(monitor_name)

# COLORS AND LUMINANCE
# # Lum to Color255 (maxLum = 253)
LumColor255Factor = 2.39538706913372
maxLum = 106  # 255 RGB
bgLumProp = .2
bgLum = maxLum * bgLumProp
bgColor255 = bgLum * LumColor255Factor
bgColor1 = bgLum / maxLum

# colour space
this_colourSpace = 'rgb255'
this_bgColour = bgColor255
if monitor_name == 'OLED':
    this_colourSpace = 'rgb1'
    this_bgColour = bgColor1
print(f"\nthis_colourSpace: {this_colourSpace}, this_bgColour: {this_bgColour}")

# don't use full screen on external monitor
display_number = 1  # 0 indexed, 1 for external display, 0 for internal
if monitor_name in ['asus_cal', 'Nick_work_laptop', 'NickMac', 'Dell_AW3423DW', 'ASUS_2_13_240Hz']:
    display_number = 0
use_full_screen = True
if display_number > 0:
    use_full_screen = False

widthPix = thisMon.getSizePix()[0]
heightPix = thisMon.getSizePix()[1]
monitorwidth = thisMon.getWidth()  # monitor width in cm
viewdist = thisMon.getDistance()  # viewing distance in cm
viewdistPix = widthPix / monitorwidth*viewdist
mon = monitors.Monitor(monitor_name, width=monitorwidth, distance=viewdist)
mon.setSizePix((widthPix, heightPix))

# WINDOW SPEC
win = visual.Window(monitor=mon, size=(widthPix, heightPix),
                    colorSpace=this_colourSpace, color=this_bgColour,
                    winType='pyglet',  # I've added this to make it work on pycharm/mac
                    pos=[1, -1],  # pos gives position of top-left of screen
                    units='pix',
                    screen=display_number,
                    allowGUI=False,
                    fullscr=use_full_screen)


# pixel size
pixel_mm_deg_dict = get_pixel_mm_deg_values(monitor_name=monitor_name)
print(f"diagonal pixel size: {pixel_mm_deg_dict['diag_mm']} mm, or {pixel_mm_deg_dict['diag_deg']} dva")


# expected frame duration
expected_fr_sec = 1/fps
expected_fr_ms = expected_fr_sec * 1000
print(f"\nexpected frame duraction: {expected_fr_ms} ms (or {round(expected_fr_sec, 5)} seconds).")

actualFrameRate = int(win.getActualFrameRate())
print(f"actual fps: {win.getActualFrameRate()}")
if abs(fps-actualFrameRate) > 5:
    raise ValueError(f"\nfps ({fps}) does not match actualFrameRate ({actualFrameRate}).")

'''set the max and min frame duration to accept, trials with critial frames beyond these bound will be repeated.'''
# frame error tolerance - default is approx 20% but seems to vary between runs(!), so set it manually.
frame_tolerance_prop = .2
max_fr_dur_sec = expected_fr_sec + (expected_fr_sec * frame_tolerance_prop)
# max_fr_dur_sec = win.refreshThreshold
max_fr_dur_ms = max_fr_dur_sec * 1000
win.refreshThreshold = max_fr_dur_sec
frame_tolerance_sec = max_fr_dur_sec - expected_fr_sec
frame_tolerance_ms = frame_tolerance_sec * 1000
frame_tolerance_prop = frame_tolerance_sec / expected_fr_sec
min_fr_dur_sec = expected_fr_sec - (expected_fr_sec * frame_tolerance_prop)
# min_fr_dur_sec = expected_fr_sec - frame_tolerance_sec
print(f"\nframe_tolerance_sec: {frame_tolerance_sec} ({frame_tolerance_prop}% of {expected_fr_sec} sec)")
print(f"max_fr_dur_sec ({100 + (100 * frame_tolerance_prop)}%): {max_fr_dur_sec} (or {max_fr_dur_ms}ms)")
print(f"min_fr_dur_sec ({100 - (100 * frame_tolerance_prop)}%): {min_fr_dur_sec} (or {min_fr_dur_sec * 1000}ms)")

# quit experiment if there are more than 10 trials with dropped frames
max_droped_fr_trials = 10


# ELEMENTS
# fixation bull eye
fixation = visual.Circle(win, radius=2, units='pix',
                         lineColor='white', fillColor='black', colorSpace=this_colourSpace)
# loc_marker = visual.Circle(win, radius=2, units='pix',
#                            lineColor='green', fillColor='red', colorSpace=this_colourSpace,)

# PROBEs
probeVert = [(0, 0), (1, 0), (1, 1), (2, 1), (2, -1), (1, -1),
             (1, -2), (-1, -2), (-1, -1), (0, -1)]
probe_size = 1
probe1 = visual.ShapeStim(win, vertices=probeVert, fillColor='white', colorSpace=this_colourSpace,
                          lineWidth=0, opacity=1, size=probe_size, interpolate=False)
probe2 = visual.ShapeStim(win, vertices=probeVert, fillColor='white', colorSpace=this_colourSpace,
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
                               font='Arial', height=20, color='white', colorSpace=this_colourSpace,)


# BREAKS
take_break = 76
break_dur = 30
print(f"\ntake_break every {take_break} trials.")
break_text = f"Break\nTurn on the light and take at least {break_dur} seconds break.\n" \
             "Keep focussed on the fixation circle in the middle of the screen.\n" \
             "Remember, if you don't see the target, just guess!"
breaks = visual.TextStim(win=win, name='breaks', text=break_text, font='Arial',
                         pos=[0, 0], height=20, ori=0, color='white', colorSpace=this_colourSpace,)

end_of_exp_text = "You have completed this experiment.\nThank you for your time.\n\n"
end_of_exp = visual.TextStim(win=win, name='end_of_exp',
                             text=end_of_exp_text,
                             font='Arial', height=20, colorSpace=this_colourSpace,)

too_many_dropped_fr = visual.TextStim(win=win, name='too_many_dropped_fr',
                                      text="The experiment had quit as the computer is dropping frames.\n"
                                           "Sorry for the inconvenience.\n"
                                           "Please contact the experimenter.\n\n"
                                           "Press any key to return to the desktop.",
                                      font='Arial', height=20, colorSpace=this_colourSpace,)

while not event.getKeys():
    fixation.setRadius(3)
    fixation.draw()
    instructions.draw()
    win.flip()


# empty variable to store recorded frame durations
exp_n_fr_recorded_list = [0]
exp_n_dropped_fr = 0
dropped_fr_trial_counter = 0
dropped_fr_trial_x_locs = []
# user_rpt_trial_x_locs = []
fr_int_per_trial = []
recorded_fr_counter = 0
fr_counter_per_trial = []
cond_list = []


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

# the number of the trial for the output file
trial_number = 0
# the actual number of trials including repeated trials (trial_number stays the same for these)
actual_trials_inc_rpt = 0

# EXPERIMENT
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
            print(f"\n({actual_trials_inc_rpt}) trial_number: {trial_number}, "
                  f"stair_idx: {stair_idx}, thisStair: {thisStair}, step: {step}")

            # condition (Separation, ISI)
            sep = sep_vals_list[stair_idx]
            # separation expressed as degrees.
            if -1 < sep < 99:
                sep_deg = sep * pixel_mm_deg_dict['diag_deg']
            else:
                sep_deg = 0
            ISI = ISI_vals_list[stair_idx]

            # Luminance (staircase varies probeLum)
            probeLum = thisStair.next()
            probeColor255 = int(probeLum * LumColor255Factor)  # rgb255 are ints.
            probeColor1 = probeLum / maxLum

            this_probeColor = probeColor255
            if this_colourSpace == 'rgb1':
                this_probeColor = probeColor1
            probe1.setFillColor([this_probeColor, this_probeColor, this_probeColor])
            probe2.setFillColor([this_probeColor, this_probeColor, this_probeColor])
            print(f"probeLum: {probeLum}, this_probeColor: {this_probeColor}, probeColor255: {probeColor255}, probeColor1: {probeColor1}")

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
            jump_dir = probes_dirs_list[stair_idx]
            target_jump = 1
            if jump_dir == 'exp':
                target_jump = -1

            print(f"corner: {corner} {corner_name}; jump dir: {target_jump} {jump_dir}")

            # make negative separation column for comparing inward (contract) and outward (expand).
            if jump_dir == 'exp':
                neg_sep = 0 - sep
                if sep == 0:
                    neg_sep = -.1
            else:
                neg_sep = sep
            print(f"ISI: {ISI}, sep: {sep} (neg_sep: {neg_sep})")

            # shift probes by separation
            '''Both probes should be equally spaced around the meridian point.
            E.g., if sep = 4, probe 1 will be shifted 2 pixels in one direction and 
            probe 2 will be shifted 2 pixels in opposite direction. 
            Where separation is an odd number (e.g., 5), they will be shifted by 2 and 3 pixels; allocated randomly.
            To check probe locations, uncomment loc_marker'''
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
                    if target_jump == 1:  # contract
                        probe1_ori += 270
                        probe2_ori += 270
                        # probe2 is left and down from probe1
                        probe1_pos = [loc_x + p1_shift, loc_y + p1_shift]
                        probe2_pos = [loc_x - p2_shift + 1, loc_y - p2_shift]
                    elif target_jump == -1:  # expand
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
                    if target_jump == 1:  # contract
                        probe1_ori += 180
                        probe2_ori += 180
                        # probe2 is right and down from probe1
                        probe1_pos = [loc_x - p1_shift, loc_y + p1_shift]
                        probe2_pos = [loc_x + p2_shift - 1, loc_y - p2_shift]
                    elif target_jump == -1:  # expand
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
                    if target_jump == 1:  # contract
                        probe1_ori += 90
                        probe2_ori += 90
                        # probe2 is right and up from probe1
                        probe1_pos = [loc_x - p1_shift, loc_y - p1_shift]
                        probe2_pos = [loc_x + p2_shift - 1, loc_y + p2_shift]
                    elif target_jump == -1:  # expand
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
                    if target_jump == 1:  # contract
                        probe1_ori += 0
                        probe2_ori += 0
                        # probe2 is left and up from probe1
                        probe1_pos = [loc_x + p1_shift, loc_y - p1_shift]
                        probe2_pos = [loc_x - p2_shift + 1, loc_y + p2_shift]
                    elif target_jump == -1:  # expand
                        probe1_ori += 180
                        probe2_ori += 180
                        # probe2 is right and down from probe1
                        probe1_pos = [loc_x - p1_shift, loc_y + p1_shift]
                        probe2_pos = [loc_x + p2_shift - 1, loc_y - p2_shift]

            # loc_marker.setPos([loc_x, loc_y])
            probe1.setPos(probe1_pos)
            probe1.setOri(probe1_ori)
            probe2.setPos(probe2_pos)
            probe2.setOri(probe2_ori)
            print(f"loc_marker: {[loc_x, loc_y]}, probe1_pos: {probe1_pos}, "
                  f"probe2_pos: {probe2_pos}. dff: {dist_from_fix}")


            # VARIABLE FIXATION TIME
            '''to reduce anticipatory effects that might arise from fixation always being same length.
            if False, vary_fix == .5 seconds, so t_fixation is 1 second.
            if Ture, vary_fix is between 0 and 1 second, so t_fixation is between .5 and 1.5 seconds.'''
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
            print(f"t_fixation: {t_fixation}, t_probe_1: {t_probe_1}, "
                  f"t_ISI: {t_ISI}, t_probe_2: {t_probe_2}, t_response: {t_response}\n")

            # I've moved the repat option to the top so repetitions don't appear in same corner
            # repeat the trial if [r] has been pressed
            # repeat = True
            # while repeat:
            #     frameN = -1

            # continue_routine refers to flipping the screen to show next frame

            # take a break every ? trials
            # if (trial_number % take_break == 1) & (trial_number > 1):
            if (actual_trials_inc_rpt % take_break == 1) & (actual_trials_inc_rpt > 1):
                print("\nTaking a break.\n")

                breaks.text = break_text + f"\n{trial_number - 1}/{total_n_trials} trials completed."
                breaks.draw()
                win.flip()
                event.clearEvents(eventType='keyboard')
                core.wait(secs=break_dur)
                event.clearEvents(eventType='keyboard')
                breaks.text = break_text + "\n\nPress any key to continue."
                breaks.draw()
                win.flip()
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

                    # todo: test this on windows, Linux and mac to see if it matters
                    # prioritise psychopy
                    # core.rush(True)

                    # start recording frame intervals
                    if record_fr_durs:
                        win.recordFrameIntervals = True
                        print(f"{frameN}: win.recordFrameIntervals : {win.recordFrameIntervals}")

                    # reset timer to start with probe1 presentation.
                    resp.clock.reset()

                    # clear any previous key presses
                    event.clearEvents(eventType='keyboard')
                    theseKeys = []

                if frameN == t_probe_2 + 1:
                    # relax psychopy prioritization
                    # core.rush(False)

                    if record_fr_durs:
                        win.recordFrameIntervals = False
                        print(f"{frameN}: win.recordFrameIntervals : {win.recordFrameIntervals}")


                '''Experiment timings'''
                # FIXATION
                if t_fixation >= frameN > 0:
                    fixation.setRadius(3)
                    fixation.draw()
                    # loc_marker.draw()


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
                        theseKeys = theseKeys[-1]  # just the last key pressed
                        resp.keys = theseKeys
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
                            # trial_n_dropped_fr = win.nDroppedFrames
                            print(f"n_fr_recorded: {n_fr_recorded}, trial_fr_intervals: {trial_fr_intervals}")

                            # add to empty lists etc.
                            # exp_fr_intervals += trial_fr_intervals
                            fr_int_per_trial.append(trial_fr_intervals)
                            fr_counter_per_trial.append(list(range(recorded_fr_counter, recorded_fr_counter + len(trial_fr_intervals))))
                            recorded_fr_counter += len(trial_fr_intervals)
                            exp_n_fr_recorded_list.append(exp_n_fr_recorded_list[-1] + n_fr_recorded)
                            # exp_n_dropped_fr += trial_n_dropped_fr
                            cond_list.append(thisStair.name)

                            # get timings for each segment (probe1, ISI, probe2).
                            fr_diff_ms = [(expected_fr_sec - i) * 1000 for i in trial_fr_intervals]
                            print(f"fr_diff_ms: {fr_diff_ms}, sum: {sum(fr_diff_ms)}")

                            p1_durs = fr_diff_ms[:2]
                            p1_diff = sum(p1_durs)
                            if ISI > 0:
                                isi_durs = fr_diff_ms[2:-2]
                            else:
                                isi_durs = []
                            isi_diff = sum(isi_durs)

                            if ISI > -1:
                                p2_durs = fr_diff_ms[-2:]
                            else:
                                p2_durs = []
                            p2_diff = sum(p2_durs)

                            print(f"\np1_durs: {p1_durs}, p1_diff: {p1_diff}\n"
                                  f"isi_durs: {isi_durs}, isi_diff: {isi_diff}\n"
                                  f"p2_durs: {p2_durs}, p2_diff: {p2_diff}\n")

                            # check for dropped frames (or frames that are too short)
                            # if timings are bad, repeat trial
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
                            else:
                                print('Timing good')

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

                # # User can repeat previous trial.
                # # Note, if they respond incorrectly on trial n, then press 'r',
                # # it is probably too late, as it will be repeating n+1, not n.
                # if event.getKeys(keyList=["r"]) or event.getKeys(keyList=['num_9']):
                #     print("\n\tparticipant pressed repeat.")
                #     trial_x_locs = [exp_n_fr_recorded_list[-1], exp_n_fr_recorded_list[-1] + n_fr_recorded]
                #     user_rpt_trial_x_locs.append(trial_x_locs)
                #     repeat = True
                #     trial_number -= 1
                #     continueRoutine = False
                #     continue


                # refresh the screen
                if continueRoutine:
                    win.flip()

        # add info to output csv
        thisExp.addData('trial_number', trial_number)
        thisExp.addData('trial_n_inc_rpt', actual_trials_inc_rpt)
        thisExp.addData('stair', stair_idx)
        thisExp.addData('stair_name', thisStair)
        thisExp.addData('step', step)
        thisExp.addData('separation', sep)
        thisExp.addData('neg_sep', neg_sep)
        thisExp.addData('sep_deg', sep_deg)
        thisExp.addData('ISI', ISI)
        thisExp.addData('isi_dur_fr', isi_dur_fr)
        thisExp.addData('isi_ms', (1000 / fps) * isi_dur_fr)
        thisExp.addData('cond_type', target_jump)
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
        thisExp.addData('p1_diff', p1_diff)
        thisExp.addData('isi_diff', isi_diff)
        thisExp.addData('p2_diff', p2_diff)
        thisExp.addData('monitor_name', monitor_name)
        thisExp.addData('this_colourSpace', this_colourSpace)
        thisExp.addData('this_bgColour', this_bgColour)
        thisExp.addData('selected_fps', fps)
        thisExp.addData('actual_fps', actualFrameRate)
        thisExp.addData('frame_tolerance_prop', frame_tolerance_prop)
        thisExp.addData('expName', expName)
        thisExp.addData('psychopy_version', psychopy_version)
        thisExp.addData('date', expInfo['date'])
        thisExp.addData('time', expInfo['time'])

        # tell psychopy to move to next trial
        thisExp.nextEntry()

        # update staircase based on whether response was correct or incorrect
        thisStair.newValue(resp.corr)


print("\nend of experiment loop, saving data\n")
# now exp is completed, save as '_output' rather than '_incomplete'
thisExp.dataFileName = os.path.join(save_dir, f'{participant_name}_{run_number}_output')
thisExp.close()


# plot frame intervals
if record_fr_durs:

    # flatten list of lists (fr_int_per_trial) to get len, min and max
    all_fr_intervals = [val for sublist in fr_int_per_trial for val in sublist]
    total_recorded_fr = len(all_fr_intervals)

    print(f"{dropped_fr_trial_counter}/{total_n_trials} trials with bad timing "
          f"(expected: {round(expected_fr_ms, 2)}ms, "
          f"frame_tolerance_ms: +/- {round(frame_tolerance_ms, 2)})")

    '''set colours for lines on plot.'''
    # get set of colours
    my_colours = fig_colours(n_stairs, alternative_colours=False)
    # associate colours with conditions
    colour_dict = {k: v for (k, v) in zip(stair_names_list, my_colours)}
    # make list of colours based on order of conditions
    cond_colour_list = [colour_dict[i] for i in cond_list]

    # plot frame intervals across the experiment with discontinuous line, coloured for each cond
    for trial_x_vals, trial_fr_durs, colour in zip(fr_counter_per_trial, fr_int_per_trial, cond_colour_list):
        plt.plot(trial_x_vals, trial_fr_durs, color=colour)

    # add legend with colours per condition
    legend_handles_list = []
    for cond in stair_names_list:
        leg_handle = mlines.Line2D([], [], color=colour_dict[cond], label=cond,
                                   marker='.', linewidth=.5, markersize=4)
        legend_handles_list.append(leg_handle)

    plt.legend(handles=legend_handles_list, fontsize=6, title='conditions', framealpha=.5)

    # add vertical lines to signify trials, shifted back so trials fall between lines
    fr_v_lines = [i - .5 for i in exp_n_fr_recorded_list]
    for trial_line in fr_v_lines:
        plt.axvline(x=trial_line, color='silver', linestyle='dashed', zorder=0)

    # add horizontal lines: green = expected frame duration, red = frame error tolerance
    plt.axhline(y=expected_fr_sec, color='grey', linestyle='dotted', alpha=.5)
    plt.axhline(y=max_fr_dur_sec, color='red', linestyle='dotted', alpha=.5)
    plt.axhline(y=min_fr_dur_sec, color='red', linestyle='dotted', alpha=.5)

    # shade trials that were repeated: red = bad timing, orange = user repeat
    for loc_pair in dropped_fr_trial_x_locs:
        print(loc_pair)
        x0, x1 = loc_pair[0] - .5, loc_pair[1] - .5
        plt.axvspan(x0, x1, color='red', alpha=0.15, zorder=0, linewidth=None)

    # for loc_pair in user_rpt_trial_x_locs:
    #     x0, x1 = loc_pair[0] - .5, loc_pair[1] - .5
    #     plt.axvspan(x0, x1, color='orange', alpha=0.15, zorder=0, linewidth=None)

    plt.title(f"{monitor_name}, {fps}Hz, {expInfo['date']}\n{dropped_fr_trial_counter}/{total_n_trials} trials."
              f"dropped fr (expected: {round(expected_fr_ms, 2)}ms, "
              f"frame_tolerance_ms: +/- {round(frame_tolerance_ms, 2)})")
    fig_name = f'{participant_name}_{run_number}_frames.png'
    print(f"fig_name: {fig_name}")
    plt.savefig(os.path.join(save_dir, fig_name))
    # plt.show()
    plt.close()


while not event.getKeys():
    # display end of experiment screen
    end_of_exp.draw()
    win.flip()
    core.wait(secs=5)
    end_of_exp.text = end_of_exp_text + "\n\nPress any key to continue."
    end_of_exp.draw()
    win.flip()
else:
    # logging.flush()  # write messages out to all targets
    thisExp.abort()  # or data files will save again on exit

    # close and quit once a key is pressed
    win.close()
    core.quit()
