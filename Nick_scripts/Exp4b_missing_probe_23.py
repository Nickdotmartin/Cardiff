from __future__ import division
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
This version (4b) only presents one type of coherence: 
e.g., does not always include an incoherent condition to compare with coherent condition.

Missing probe study.
Probes will appear in three corners, participants identify corner without probe.

The script is designed it so that all 4 corners have probe motion planned,
then just don't draw the missing_corner that is selected

We want to manipulate the coherence of the movement of the three probes.
Motion is coherent if
1. all tangental: (2) clockwise or anti-clockwise.  [rotation]
2. all radial: (2) inward/outward       [radial]
3. mixed: same quadrant (4) top-right; top-left; bottom-left; bottom-right  [translation]
This gives 8 possibilities for clear coherent motion

Non-coherent if
Mixed - two radial (one in, one out), two tangent (one CW one ACW).
Only with this pattern can you be sure that when you delete one the other three aren't good.
However, with the pattern, there will always be two probes with same absolute (tr, tl, bl, br),
or relational (in, out, cw, ACW) motion, so I have made it so that the
incohenrent probes include all 4 absolute directions, but with two matching for cw/ACW.

There are no mirror symmetrical patterns (less coherent - but still a pattern)
    - e.g., two move up and left, two move up and right
'''

# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# Monitor config from monitor centre
monitor_name = 'Nick_work_laptop'  # 'asus_cal', 'Nick_work_laptop', 'Asus_VG24', 'HP_24uh', 'NickMac', 'Iiyama_2_18',

# Store info about the experiment session
expName = 'Exp4b_missing_probe_23'
expInfo = {'1. Participant': 'nicktest',
           '2. run_number': '1',
           '3. Probe duration in frames': [2, 50, 100],
           '4. fps': [240, 120, 60],
           '5. vary fixation': [True, False],
           '6. probe coherence': ['rotation', 'radial', 'translation', 'incoherent'],
           '7. exp type': ['stair_per_dir', 'mixed_dir'],
           '8. Record_frame_durs': [True, False]
           }

# dialogue box
dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
if not dlg.OK:
    core.quit()  # user pressed escape

# dialogue box settings
participant_name = expInfo['1. Participant']
run_number = int(expInfo['2. run_number'])
probe_duration = int(expInfo['3. Probe duration in frames'])
fps = int(expInfo['4. fps'])
vary_fixation = eval(expInfo['5. vary fixation'])
probe_coherence = expInfo['6. probe coherence']
exp_type = expInfo['7. exp type']
record_fr_durs = eval(expInfo['8. Record_frame_durs'])
print(f'\nTrial condition details:')
print(f'probe_coherence: {probe_coherence}')
print(f'exp_type: {exp_type}')

n_trials_per_stair = 25
probe_ecc = 4
expInfo['date'] = datetime.now().strftime("%d/%m/%Y")
expInfo['time'] = datetime.now().strftime("%H:%M:%S")

# VARIABLES
'''Distances between probes (spatially and temporally)
For 1probe condition, use separation==99.
For concurrent probes, use ISI==-1.
'''
stair_per_dir = True
if exp_type == 'stair_per_dir':
    print('different staircases for each motion direction')
    if probe_coherence == 'rotation':
        target_jump_vals = ['CW', 'ACW']
    elif probe_coherence == 'radial':
        target_jump_vals = ['exp', 'cont']
    else:
        raise ValueError('Can only have different motion stairs if probe_coherence is radial or rotational.')
else:
    stair_per_dir = False
    print('each stair will have a mixture of motion directions.')
    target_jump_vals = ['mixed']


separations = [6]  # select from [0, 1, 2, 3, 6, 18, 99]
ISI_values = [-1, 3, 6]  # select from [-1, 0, 2, 4, 6, 9, 12, 24]
print(f'separations: {separations}')
print(f'ISI_values: {ISI_values}')

# repeat separation values for each ISI (and target jump) e.g., [0, 0, 0, 0, 6, 6, 6, 6]
sep_vals_list = list(np.repeat(separations, len(ISI_values))) * len(target_jump_vals)
print(f'sep_vals_list: {sep_vals_list}')

# ISI_vals_list cycles through ISIs (and target jump) e.g., [-1, 6, -1, 6, -1, 6, -1, 6]
ISI_vals_list = list(np.tile(ISI_values, len(separations) * len(target_jump_vals)))
print(f'ISI_vals_list: {ISI_vals_list}')

if stair_per_dir:
    # add target jump list here too
    target_jump_list = list(np.repeat(target_jump_vals, len(sep_vals_list) / len(target_jump_vals)))
    print(f'target_jump_list: {target_jump_list}')
    stair_names_list = [f'{t}_sep{s}_ISI{i}' for t, s, i in zip(target_jump_list, sep_vals_list, ISI_vals_list)]
else:
    # e.g., ['sep0_ISI-1', 'sep0_ISI6', 'sep6_ISI-1', 'sep6_ISI6']
    stair_names_list = [f'sep{s}_ISI{c}' for s, c in zip(sep_vals_list, ISI_vals_list)]

print(f'stair_names_list: {stair_names_list}')
n_stairs = len(stair_names_list)
print(f'n_stairs: {n_stairs}')

total_n_trials = int(n_trials_per_stair * n_stairs)
print(f'total_n_trials: {total_n_trials}')


# FILENAME
save_dir = os.path.join(_thisDir, expName, exp_type, probe_coherence,
                        participant_name, f'{participant_name}_{run_number}')

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
#todo: check OLED montor name
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


'''
Dictionary of co-ordinated probe motion.  There are 3 coherent types:
rotational, radial and translational.
There are two rotational (CW and ACW), two radial (in/out) and four translational (4 corners).
There are eight types of incoherent motion.
Note, rotation and radial and in a different order to previous versions with ACW/CW and exp/cont swapped order.
'''

probes_dict = {'type': {'rotation': {'n_examples': 2, 'names': ['ACW', 'CW'],
                                     'examples':
                                         {'ACW':
                                               {'tr': {'orientation': 'tangent', 'jump': -1, 'direction': 'ACW'},
                                                'tl': {'orientation': 'tangent', 'jump': -1, 'direction': 'ACW'},
                                                'bl': {'orientation': 'tangent', 'jump': -1, 'direction': 'ACW'},
                                                'br': {'orientation': 'tangent', 'jump': -1, 'direction': 'ACW'}},
                                          'CW':
                                               {'tr': {'orientation': 'tangent', 'jump': 1, 'direction': 'CW'},
                                                'tl': {'orientation': 'tangent', 'jump': 1, 'direction': 'CW'},
                                                'bl': {'orientation': 'tangent', 'jump': 1, 'direction': 'CW'},
                                                'br': {'orientation': 'tangent', 'jump': 1, 'direction': 'CW'}}}},
                        'radial': {'n_examples': 2, 'names': ['exp', 'cont'],
                                    'examples':
                                        {'exp':
                                             {'tr': {'orientation': 'radial', 'jump': -1, 'direction': 'exp'},
                                              'tl': {'orientation': 'radial', 'jump': -1, 'direction': 'exp'},
                                              'bl': {'orientation': 'radial', 'jump': -1, 'direction': 'exp'},
                                              'br': {'orientation': 'radial', 'jump': -1, 'direction': 'exp'}},
                                         'cont':
                                             {'tr': {'orientation': 'radial', 'jump': 1, 'direction': 'cont'},
                                              'tl': {'orientation': 'radial', 'jump': 1, 'direction': 'cont'},
                                              'bl': {'orientation': 'radial', 'jump': 1, 'direction': 'cont'},
                                              'br': {'orientation': 'radial', 'jump': 1, 'direction': 'cont'}}}},
                        'translation': {'n_examples': 4, 'names': ['to_tr', 'to_tl', 'to_bl', 'to_br'],
                                        'examples':
                                            {'to_tr':
                                                 {'tr': {'orientation': 'radial', 'jump': -1, 'direction': 'exp'},
                                                  'tl': {'orientation': 'tangent', 'jump': 1, 'direction': 'CW'},
                                                  'bl': {'orientation': 'radial', 'jump': 1, 'direction': 'cont'},
                                                  'br': {'orientation': 'tangent', 'jump': -1, 'direction': 'ACW'}},
                                             'to_tl':
                                                 {'tr': {'orientation': 'tangent', 'jump': -1, 'direction': 'ACW'},
                                                  'tl': {'orientation': 'radial', 'jump': -1, 'direction': 'exp'},
                                                  'bl': {'orientation': 'tangent', 'jump': 1, 'direction': 'CW'},
                                                  'br': {'orientation': 'radial', 'jump': 1, 'direction': 'cont'}},
                                             'to_bl':
                                                 {'tr': {'orientation': 'radial', 'jump': 1, 'direction': 'cont'},
                                                  'tl': {'orientation': 'tangent', 'jump': -1, 'direction': 'ACW'},
                                                  'bl': {'orientation': 'radial', 'jump': -1, 'direction': 'exp'},
                                                  'br': {'orientation': 'tangent', 'jump': 1, 'direction': 'CW'}},
                                             'to_br':
                                                 {'tr': {'orientation': 'tangent', 'jump': 1, 'direction': 'CW'},
                                                  'tl': {'orientation': 'radial', 'jump': 1, 'direction': 'cont'},
                                                  'bl': {'orientation': 'tangent', 'jump': -1, 'direction': 'ACW'},
                                                  'br': {'orientation': 'radial', 'jump': -1, 'direction': 'exp'}}}},
                        'incoherent': {'n_examples': 8, 'names': ['inc0', 'inc1', 'inc2', 'inc3',
                                                                  'inc4', 'inc5', 'inc6', 'inc7'],
                                       'examples':
                                           {'inc0':
                                                {'tr': {'orientation': 'radial', 'jump': -1, 'direction': 'exp'},
                                                 'tl': {'orientation': 'tangent', 'jump': -1, 'direction': 'ACW'},
                                                 'bl': {'orientation': 'tangent', 'jump': -1, 'direction': 'ACW'},
                                                 'br': {'orientation': 'radial', 'jump': 1, 'direction': 'cont'}},
                                            'inc1':
                                                {'tr': {'orientation': 'radial', 'jump': 1, 'direction': 'cont'},
                                                 'tl': {'orientation': 'radial', 'jump': -1, 'direction': 'exp'},
                                                 'bl': {'orientation': 'tangent', 'jump': -1, 'direction': 'ACW'},
                                                 'br': {'orientation': 'tangent', 'jump': -1, 'direction': 'ACW'}},
                                            'inc2':
                                                {'tr': {'orientation': 'tangent', 'jump': -1, 'direction': 'ACW'},
                                                 'tl': {'orientation': 'radial', 'jump': 1, 'direction': 'cont'},
                                                 'bl': {'orientation': 'radial', 'jump': -1, 'direction': 'exp'},
                                                 'br': {'orientation': 'tangent', 'jump': -1, 'direction': 'ACW'}},
                                            'inc3':
                                                {'tr': {'orientation': 'tangent', 'jump': -1, 'direction': 'ACW'},
                                                 'tl': {'orientation': 'tangent', 'jump': -1, 'direction': 'ACW'},
                                                 'bl': {'orientation': 'radial', 'jump': 1, 'direction': 'cont'},
                                                 'br': {'orientation': 'radial', 'jump': -1, 'direction': 'exp'}},
                                            'inc4':
                                                {'tr': {'orientation': 'radial', 'jump': 1, 'direction': 'cont'},
                                                 'tl': {'orientation': 'tangent', 'jump': 1, 'direction': 'CW'},
                                                 'bl': {'orientation': 'tangent', 'jump': 1, 'direction': 'CW'},
                                                 'br': {'orientation': 'radial', 'jump': -1, 'direction': 'exp'}},
                                            'inc5':
                                                {'tr': {'orientation': 'radial', 'jump': -1, 'direction': 'exp'},
                                                 'tl': {'orientation': 'radial', 'jump': 1, 'direction': 'cont'},
                                                 'bl': {'orientation': 'tangent', 'jump': 1, 'direction': 'CW'},
                                                 'br': {'orientation': 'tangent', 'jump': 1, 'direction': 'CW'}},
                                            'inc6':
                                                {'tr': {'orientation': 'tangent', 'jump': 1, 'direction': 'CW'},
                                                 'tl': {'orientation': 'radial', 'jump': -1, 'direction': 'exp'},
                                                 'bl': {'orientation': 'radial', 'jump': 1, 'direction': 'cont'},
                                                 'br': {'orientation': 'tangent', 'jump': 1, 'direction': 'CW'}},
                                            'inc7':
                                                {'tr': {'orientation': 'tangent', 'jump': 1, 'direction': 'CW'},
                                                 'tl': {'orientation': 'tangent', 'jump': 1, 'direction': 'CW'},
                                                 'bl': {'orientation': 'radial', 'jump': -1, 'direction': 'exp'},
                                                 'br': {'orientation': 'radial', 'jump': 1, 'direction': 'cont'}}}}
                        }
               }

# print('dict test')
# # print(probes_dict['type'][probe_coherence]['examples']['CW'])
# print_nested_round_floats(probes_dict, dict_title='focussed_dict_print')

# ELEMENTS

# PROBEs
probeVert = [(0, 0), (1, 0), (1, 1), (2, 1), (2, -1), (1, -1),
             (1, -2), (-1, -2), (-1, -1), (0, -1)]
probe_size = 1
# # 45=top-right, 135=top-left, 225=bottom-left, 315=bottom-right
probe1_tr = visual.ShapeStim(win, vertices=probeVert, fillColor=(1.0, -1.0, 1.0), colorSpace=this_colourSpace,
                             lineWidth=0, opacity=1, size=probe_size, interpolate=False)
probe2_tr = visual.ShapeStim(win, vertices=probeVert, fillColor=[-1.0, 1.0, -1.0], colorSpace=this_colourSpace,
                             lineWidth=0, opacity=1, size=probe_size, interpolate=False)
probe1_tl = visual.ShapeStim(win, vertices=probeVert, fillColor=(1.0, -1.0, 1.0), colorSpace=this_colourSpace,
                             lineWidth=0, opacity=1, size=probe_size, interpolate=False)
probe2_tl = visual.ShapeStim(win, vertices=probeVert, fillColor=[-1.0, 1.0, -1.0], colorSpace=this_colourSpace,
                             lineWidth=0, opacity=1, size=probe_size, interpolate=False)
probe1_bl = visual.ShapeStim(win, vertices=probeVert, fillColor=(1.0, -1.0, 1.0), colorSpace=this_colourSpace,
                             lineWidth=0, opacity=1, size=probe_size, interpolate=False)
probe2_bl = visual.ShapeStim(win, vertices=probeVert, fillColor=[-1.0, 1.0, -1.0], colorSpace=this_colourSpace,
                             lineWidth=0, opacity=1, size=probe_size, interpolate=False)
probe1_br = visual.ShapeStim(win, vertices=probeVert, fillColor=(1.0, -1.0, 1.0), colorSpace=this_colourSpace,
                             lineWidth=0, opacity=1, size=probe_size, interpolate=False)
probe2_br = visual.ShapeStim(win, vertices=probeVert, fillColor=[-1.0, 1.0, -1.0], colorSpace=this_colourSpace,
                             lineWidth=0, opacity=1, size=probe_size, interpolate=False)


# fixation bull eye
fixation = visual.Circle(win, radius=2, units='pix',
                         lineColor='white', fillColor='black', colorSpace=this_colourSpace)
# location markers are displayed in each corner between the 2 probes to check they are balanced around these locations.
# loc_marker_tr = visual.Circle(win, radius=2, units='pix', lineColor='green', fillColor='red', colorSpace=this_colourSpace)
# loc_marker_tl = visual.Circle(win, radius=2, units='pix', lineColor='green', fillColor='red', colorSpace=this_colourSpace)
# loc_marker_bl = visual.Circle(win, radius=2, units='pix', lineColor='green', fillColor='red', colorSpace=this_colourSpace)
# loc_marker_br = visual.Circle(win, radius=2, units='pix', lineColor='green', fillColor='red', colorSpace=this_colourSpace)

# dist_from_fix is a constant to get 4dva distance from fixation,
dist_from_fix = round((tan(np.deg2rad(probe_ecc)) * viewdistPix) / sqrt(2))

# MOUSE - hide cursor
myMouse = event.Mouse(visible=False)

# # KEYBOARD
resp = event.BuilderKeyResponse()

# INSTRUCTION
instructions = visual.TextStim(win=win, name='instructions',
                               text="\n\n\n\n\n\nFocus on the small circle at the centre of the screen.\n\n"
                                    "Small white probes will briefly flash in three corners,\n"
                                    "press the key related to the direction which did NOT contain a probe:\n\n"
                                    "[4]/[Q] top-left\t\t\t[5]/[W] top-right\n\n\n\n"
                                    "[1]/[A] bottom-left\t\t\t[2]/[S] bottom-right.\n\n\n"
                                    "Some flashes will seem bright and easy to see\n"
                                    "Some will be dim and harder to spot\n"
                                    "Some will be so dim that you won't see them, so just guess!\n\n"
                                    "You don't need to think for long. "
                                    "Respond quickly, but try to push press the correct key!\n\n"
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
                             text=end_of_exp_text, color='white',
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
            print(f"ISI: {ISI}, sep: {sep}")

            # Luminance (staircase varies probeLum)
            probeLum = thisStair.next()
            probeColor255 = int(probeLum * LumColor255Factor)  # rgb255 are ints.
            probeColor1 = probeLum / maxLum

            this_probeColor = probeColor255
            if this_colourSpace == 'rgb1':
                this_probeColor = probeColor1
            probe1_tr.setFillColor([this_probeColor, this_probeColor, this_probeColor])
            probe2_tr.setFillColor([this_probeColor, this_probeColor, this_probeColor])
            probe1_tl.setFillColor([this_probeColor, this_probeColor, this_probeColor])
            probe2_tl.setFillColor([this_probeColor, this_probeColor, this_probeColor])
            probe1_bl.setFillColor([this_probeColor, this_probeColor, this_probeColor])
            probe2_bl.setFillColor([this_probeColor, this_probeColor, this_probeColor])
            probe1_br.setFillColor([this_probeColor, this_probeColor, this_probeColor])
            probe2_br.setFillColor([this_probeColor, this_probeColor, this_probeColor])
            print(f"probeLum: {probeLum}, this_probeColor: {this_probeColor}, probeColor255: {probeColor255}, probeColor1: {probeColor1}")

            '''PROBE LOCATIONs'''
            # which corner does not have a probe
            missing_corner = random.choice(['tr', 'tl', 'bl', 'br'])
            print(f"missing_corner: {missing_corner}")

            # # # # direction in which the probe jumps : CW or CCW (tangent) or exp vs cont (radial)
            if stair_per_dir:  # separate staircases for CW vs ACW, or for in (cont) vs out (exp).

                # jump_dir and target_jump selected from target_jump_list
                jump_dir = target_jump_list[stair_idx]

                if probe_coherence == 'radial':
                    target_jump = 1  # jump_dir is contract
                    this_example = 1
                    if jump_dir == 'exp':
                        target_jump = -1
                        this_example = 0

                elif probe_coherence == 'rotation':
                    target_jump = 1  # jump_dir is CW
                    this_example = 1
                    if jump_dir == 'ACW':
                        target_jump = -1
                        this_example = 0

                else:
                    raise ValueError(f'stair_per_dir only valid for rotation or radial, not {probe_coherence}')

                print(f"target_jump: {target_jump}, jump_dir: {jump_dir}")

                # add negative sep column for comparing results
                neg_sep = sep
                if target_jump == -1:
                    neg_sep = 0 - sep
                    if sep == 0:
                        neg_sep = -.1
                else:
                    neg_sep = sep
                print(f"neg_sep: {neg_sep}")


            else:  # if mixed probe dirs per stair, randomly select example

                n_examples = int(probes_dict['type'][probe_coherence]['n_examples'])
                this_example = random.choice(list(range(n_examples)))
                print(f'this_example: {this_example} from {list(range(n_examples))}')

            example_name = probes_dict['type'][probe_coherence]['names'][this_example]

            print(f"this_example: {this_example}, example_name: {example_name}")


            tr_ori = probes_dict['type'][probe_coherence]['examples'][example_name]['tr']['orientation']
            tr_jump = probes_dict['type'][probe_coherence]['examples'][example_name]['tr']['jump']
            tr_dir = probes_dict['type'][probe_coherence]['examples'][example_name]['tr']['direction']

            tl_ori = probes_dict['type'][probe_coherence]['examples'][example_name]['tl']['orientation']
            tl_jump = probes_dict['type'][probe_coherence]['examples'][example_name]['tl']['jump']
            tl_dir = probes_dict['type'][probe_coherence]['examples'][example_name]['tl']['direction']

            bl_ori = probes_dict['type'][probe_coherence]['examples'][example_name]['bl']['orientation']
            bl_jump = probes_dict['type'][probe_coherence]['examples'][example_name]['bl']['jump']
            bl_dir = probes_dict['type'][probe_coherence]['examples'][example_name]['bl']['direction']

            br_ori = probes_dict['type'][probe_coherence]['examples'][example_name]['br']['orientation']
            br_jump = probes_dict['type'][probe_coherence]['examples'][example_name]['br']['jump']
            br_dir = probes_dict['type'][probe_coherence]['examples'][example_name]['br']['direction']

            print(f"\n{probe_coherence}, {example_name}\n"
                  f"tr: {tr_dir}, {tr_ori}, {tr_jump}\ntl: {tl_dir}, {tl_ori}, {tl_jump}\n"
                  f"bl: {bl_dir}, {bl_ori}, {bl_jump}\nbr: {br_dir}, {br_ori}, {br_jump}\n")


            # shift probes by separation
            '''Both probes for each corner should be equally spaced around the meridian point.
            E.g., if sep = 4, each probe will be 2 pixels from the mid point.  
            Where separation is an odd number (e.g., 5), the extra pixel (e.g., 2, 2+1) will be allocated 
            to one of the probes randomly.'''
            if sep == 99:
                p1_shift = p2_shift = 0
            elif sep % 2 == 0:  # even number
                p1_shift = p2_shift = sep // 2
            else:  # odd number
                extra_shifted_pixel = [0, 1]
                np.random.shuffle(extra_shifted_pixel)
                p1_shift = sep // 2 + extra_shifted_pixel[0]
                p2_shift = (sep // 2) + extra_shifted_pixel[1]


            '''set probe orientation and locations'''
            # in top-right corner, both x and y increase (right and up)
            tr_loc_x = dist_from_fix * 1
            tr_loc_y = dist_from_fix * 1
            #  'orientation' (e.g., tr_ori) here refers to the relationship between probes,
            #  whereas probe1_tr.ori refers to rotational angle of probe stimulus
            if tr_ori == 'tangent':
                if tr_jump == 1:  # ACW
                    probe1_tr.ori = 180
                    probe2_tr.ori = 0
                    # probe2 is left and up from probe1
                    probe1_tr.pos = [tr_loc_x - p1_shift, tr_loc_y + p2_shift]
                    probe2_tr.pos = [tr_loc_x + p2_shift - 1, tr_loc_y - p2_shift]
                elif tr_jump == -1:  # CW
                    probe1_tr.ori = 0
                    probe2_tr.ori = 180
                    # probe2 is right and down from probe1
                    probe1_tr.pos = [tr_loc_x + p1_shift, tr_loc_y - p2_shift]
                    probe2_tr.pos = [tr_loc_x - p2_shift + 1, tr_loc_y + p2_shift]
            elif tr_ori == 'radial':
                if tr_jump == 1:  # inward
                    probe1_tr.ori = 270
                    probe2_tr.ori = 90
                    # probe2 is left and down from probe1
                    probe1_tr.pos = [tr_loc_x + p1_shift, tr_loc_y + p2_shift]
                    probe2_tr.pos = [tr_loc_x - p2_shift + 1, tr_loc_y - p2_shift]
                elif tr_jump == -1:  # outward
                    probe1_tr.ori = 90
                    probe2_tr.ori = 270
                    # probe2 is right and up from probe1
                    probe1_tr.pos = [tr_loc_x - p1_shift, tr_loc_y - p2_shift]
                    probe2_tr.pos = [tr_loc_x + p2_shift - 1, tr_loc_y + p2_shift]

            # in top-left corner, x decreases (left) and y increases (up)
            tl_loc_x = dist_from_fix * -1
            tl_loc_y = dist_from_fix * 1
            if tl_ori == 'tangent':
                if tl_jump == 1:  # ACW
                    probe1_tl.ori = 90
                    probe2_tl.ori = 270
                    # probe2 is right and up from probe1
                    probe1_tl.pos = [tl_loc_x - p1_shift, tl_loc_y - p2_shift]
                    probe2_tl.pos = [tl_loc_x + p2_shift - 1, tl_loc_y + p2_shift]
                elif tl_jump == -1:  # CW
                    probe1_tl.ori = 270
                    probe2_tl.ori = 90
                    # probe2 is left and down from probe1
                    probe1_tl.pos = [tl_loc_x + p1_shift, tl_loc_y + p2_shift]
                    probe2_tl.pos = [tl_loc_x - p2_shift + 1, tl_loc_y - p2_shift]
            elif tl_ori == 'radial':
                if tl_jump == 1:  # inward
                    probe1_tl.ori = 180
                    probe2_tl.ori = 0
                    # probe2 is right and down from probe1
                    probe1_tl.pos = [tl_loc_x - p1_shift, tl_loc_y + p2_shift]
                    probe2_tl.pos = [tl_loc_x + p2_shift - 1, tl_loc_y - p2_shift]
                elif tl_jump == -1:  # outward
                    probe1_tl.ori = 0
                    probe2_tl.ori = 180
                    # probe2 is left and up from probe1
                    probe1_tl.pos = [tl_loc_x + p1_shift, tl_loc_y - p2_shift]
                    probe2_tl.pos = [tl_loc_x - p2_shift + 1, tl_loc_y + p2_shift]

            # in bottom left corner, both x and y decrease (left and down)
            bl_loc_x = dist_from_fix * -1
            bl_loc_y = dist_from_fix * -1
            if bl_ori == 'tangent':
                if bl_jump == 1:  # ACW
                    probe1_bl.ori = 0
                    probe2_bl.ori = 180
                    probe1_bl.pos = [bl_loc_x + p1_shift, bl_loc_y - p2_shift]
                    probe2_bl.pos = [bl_loc_x - p2_shift + 1, bl_loc_y + p2_shift]
                elif bl_jump == -1:  # CW
                    probe1_bl.ori = 180
                    probe2_bl.ori = 0
                    probe1_bl.pos = [bl_loc_x - p1_shift, bl_loc_y + p2_shift]
                    probe2_bl.pos = [bl_loc_x + p2_shift - 1, bl_loc_y - p2_shift]
            elif bl_ori == 'radial':
                if bl_jump == 1:  # inward
                    probe1_bl.ori = 90
                    probe2_bl.ori = 270
                    # probe2 is right and up from probe1
                    probe1_bl.pos = [bl_loc_x - p1_shift, bl_loc_y - p2_shift]
                    probe2_bl.pos = [bl_loc_x + p2_shift - 1, bl_loc_y + p2_shift]
                elif bl_jump == -1:  # outward
                    probe1_bl.ori = 270
                    probe2_bl.ori = 90
                    # probe2 is left and down from probe1
                    probe1_bl.pos = [bl_loc_x + p1_shift, bl_loc_y - p2_shift]
                    probe2_bl.pos = [bl_loc_x - p2_shift + 1, bl_loc_y - p2_shift]

            # in bottom-right corner, x increases (right) and y decreases (down)
            br_loc_x = dist_from_fix * 1
            br_loc_y = dist_from_fix * -1
            if br_ori == 'tangent':
                if br_jump == 1:  # ACW
                    probe1_br.ori = 270
                    probe2_br.ori = 90
                    probe1_br.pos = [br_loc_x + p1_shift, br_loc_y + p2_shift]
                    probe2_br.pos = [br_loc_x - p2_shift + 1, br_loc_y - p2_shift]
                elif br_jump == -1:  # CW
                    probe1_br.ori = 90
                    probe2_br.ori = 270
                    probe1_br.pos = [br_loc_x - p1_shift, br_loc_y - p2_shift]
                    probe2_br.pos = [br_loc_x + p2_shift - 1, br_loc_y + p2_shift]
            elif br_ori == 'radial':
                if br_jump == 1:  # inward
                    probe1_br.ori = 0
                    probe2_br.ori = 180
                    # probe2 is left and up from probe1
                    probe1_br.pos = [br_loc_x + p1_shift, br_loc_y - p2_shift]
                    probe2_br.pos = [br_loc_x - p2_shift + 1, br_loc_y + p2_shift]
                elif br_jump == -1:  # outward
                    probe1_br.ori = 180
                    probe2_br.ori = 0
                    # probe2 is right and down from probe1
                    probe1_br.pos = [br_loc_x - p1_shift, br_loc_y + p2_shift]
                    probe2_br.pos = [br_loc_x + p2_shift - 1, br_loc_y - p2_shift]

            # loc_marker_tr.setPos([tr_loc_x, tr_loc_y])
            # loc_marker_tl.setPos([tl_loc_x, tl_loc_y])
            # loc_marker_bl.setPos([bl_loc_x, bl_loc_y])
            # loc_marker_br.setPos([br_loc_x, br_loc_y])


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

            # I've moved the repeat option to the top so repetitions don't appear in same corner
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
                    # loc_marker_tr.draw()
                    # loc_marker_tl.draw()
                    # loc_marker_bl.draw()
                    # loc_marker_br.draw()


                # PROBE 1: draw three probes, not one in 'missing_corner'
                elif t_probe_1 >= frameN > t_fixation:
                    if missing_corner != 'tr':
                        probe1_tr.draw()
                    if missing_corner != 'tl':
                        probe1_tl.draw()
                    if missing_corner != 'bl':
                        probe1_bl.draw()
                    if missing_corner != 'br':
                        probe1_br.draw()

                    if ISI == -1:  # SIMULTANEOUS CONDITION (concurrent)
                        if sep <= 18:  # don't draw 2nd probes in 1probe cond (sep==99)
                            if missing_corner != 'tr':
                                probe2_tr.draw()
                            if missing_corner != 'tl':
                                probe2_tl.draw()
                            if missing_corner != 'bl':
                                probe2_bl.draw()
                            if missing_corner != 'br':
                                probe2_br.draw()
                    fixation.setRadius(3)
                    fixation.draw()
                    # loc_marker_tr.draw()
                    # loc_marker_tl.draw()
                    # loc_marker_bl.draw()
                    # loc_marker_br.draw()

                # ISI
                elif t_ISI >= frameN > t_probe_1:
                    fixation.setRadius(3)
                    fixation.draw()
                    # loc_marker_tr.draw()
                    # loc_marker_tl.draw()
                    # loc_marker_bl.draw()
                    # loc_marker_br.draw()

                # PROBE 2
                elif t_probe_2 >= frameN > t_ISI:
                    if ISI >= 0:
                        if sep <= 18:  # don't draw 2nd probes in 1probe cond (sep==99)
                            if missing_corner != 'tr':
                                probe2_tr.draw()
                            if missing_corner != 'tl':
                                probe2_tl.draw()
                            if missing_corner != 'bl':
                                probe2_bl.draw()
                            if missing_corner != 'br':
                                probe2_br.draw()
                    fixation.setRadius(3)
                    fixation.draw()
                    # loc_marker_tr.draw()
                    # loc_marker_tl.draw()
                    # loc_marker_bl.draw()
                    # loc_marker_br.draw()

                # ANSWER
                elif frameN > t_probe_2:
                    fixation.setRadius(2)
                    fixation.draw()
                    # loc_marker_tr.draw()
                    # loc_marker_tl.draw()
                    # loc_marker_bl.draw()
                    # loc_marker_br.draw()

                    # ANSWER
                    theseKeys = event.getKeys(keyList=['num_5', 'num_4', 'num_1',
                                                       'num_2', 'w', 'q', 'a', 's'])

                    if len(theseKeys) > 0:  # at least one key was pressed
                        theseKeys = theseKeys[-1]  # just the last key pressed
                        resp.keys = theseKeys
                        resp.rt = resp.clock.getTime()

                        # default assume response incorrect unless meets criteria below
                        resp.corr = 0

                        if missing_corner == 'tr':
                            if (resp.keys == 'w') or (resp.keys == 'num_5'):
                                resp.corr = 1
                        elif missing_corner == 'tl':
                            if (resp.keys == 'q') or (resp.keys == 'num_4'):
                                resp.corr = 1
                        elif missing_corner == 'bl':
                            if (resp.keys == 'a') or (resp.keys == 'num_1'):
                                resp.corr = 1
                        elif missing_corner == 'br':
                            if (resp.keys == 's') or (resp.keys == 'num_2'):
                                resp.corr = 1


                        '''Get frame intervals for this trial, add to experiment and empty cache'''
                        if record_fr_durs:
                            # get trial frameIntervals details
                            trial_fr_intervals = win.frameIntervals
                            n_fr_recorded = len(trial_fr_intervals)
                            print(f"n_fr_recorded: {n_fr_recorded}, trial_fr_intervals: {trial_fr_intervals}")

                            # add to empty lists etc.
                            fr_int_per_trial.append(trial_fr_intervals)
                            fr_counter_per_trial.append(list(range(recorded_fr_counter,
                                                                   recorded_fr_counter + len(trial_fr_intervals))))
                            recorded_fr_counter += len(trial_fr_intervals)
                            exp_n_fr_recorded_list.append(exp_n_fr_recorded_list[-1] + n_fr_recorded)
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
        thisExp.addData('sep_deg', sep_deg)
        thisExp.addData('ISI', ISI)
        thisExp.addData('isi_dur_fr', isi_dur_fr)
        thisExp.addData('isi_ms', (1000 / fps) * isi_dur_fr)
        thisExp.addData('cond_type', probe_coherence)
        thisExp.addData('example_name', example_name)
        thisExp.addData('probeColor1', probeColor1)
        thisExp.addData('probeColor255', probeColor255)
        thisExp.addData('probeLum', probeLum)
        thisExp.addData('trial_response', resp.corr)
        thisExp.addData('missing_corner', missing_corner)
        if stair_per_dir:
            thisExp.addData('neg_sep', neg_sep)
        thisExp.addData('this_example', this_example)
        thisExp.addData('example_name', example_name)
        thisExp.addData('probe_ecc', probe_ecc)
        thisExp.addData('resp.rt', resp.rt)
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
        thisExp.addData('exp_type', exp_type)
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
