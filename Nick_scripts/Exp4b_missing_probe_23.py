from __future__ import division
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

'''Issues with concurrent timings resolved with use of isi_dur_fr variable.'''

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
monitor_name = 'asus_cal'  # 'asus_cal', 'Nick_work_laptop', 'Asus_VG24', 'HP_24uh', 'NickMac', 'Iiyama_2_18',

# Store info about the experiment session
expName = 'Exp4b_missing_probe_23'

expInfo = {'1. Participant': 'nicktest',
           '2. run_number': '1',
           '3. Probe duration in frames': [2, 50, 100],
           '4. fps': [240, 120, 60],
           '5. vary fixation': [True, False],
           '6. probe coherence': ['rotation', 'radial', 'translation', 'incoherent'],
           '7. exp type': ['stair_per_dir', 'mixed_dir'],
           '8. Record_frame_durs': [False, True]
           }

# GUI
dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
if not dlg.OK:
    core.quit()  # user pressed escape

expInfo['date'] = datetime.now().strftime("%d/%m/%Y")
expInfo['time'] = datetime.now().strftime("%H:%M:%S")

# GUI SETTINGS
participant_name = expInfo['1. Participant']
run_number = int(expInfo['2. run_number'])
probe_duration = int(expInfo['3. Probe duration in frames'])
fps = int(expInfo['4. fps'])
vary_fixation = eval(expInfo['5. vary fixation'])
probe_coherence = expInfo['6. probe coherence']
exp_type = expInfo['7. exp type']
record_fr_durs = eval(expInfo['8. Record_frame_durs'])
print(f'\nprobe_coherence: {probe_coherence}')
print(f'exp_type: {exp_type}')


n_trials_per_stair = 25
probe_ecc = 4

# expected frame duration
expected_fr_ms = (1/fps) * 1000

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
filename = f'{_thisDir}{os.sep}' \
           f'{expName}{os.sep}' \
           f'{exp_type}{os.sep}' \
           f'{probe_coherence}{os.sep}' \
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

actualFrameRate = int(win.getActualFrameRate())
print(f"actual fps: {type(win.getActualFrameRate())} {win.getActualFrameRate()}")
if abs(fps-actualFrameRate) > 5:
    raise ValueError(f"\nfps ({fps}) does not match actualFrameRate ({actualFrameRate}).")

# pixel size
pixel_mm_deg_dict = get_pixel_mm_deg_values(monitor_name=monitor_name)
print('pixel_mm_deg_dict.items()')
for k, v in pixel_mm_deg_dict.items():
    print(k, v)

# frame error tollerance
frame_err_sec = win.refreshThreshold
frame_err_ms = frame_err_sec * 1000
print(f"frame_err_sec (120%): {frame_err_sec} (or {frame_err_ms}ms)")
fr_recorded_list = []

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
probe1_tr = visual.ShapeStim(win, vertices=probeVert, fillColor=(1.0, -1.0, 1.0),
                             lineWidth=0, opacity=1, size=probe_size, interpolate=False)
probe2_tr = visual.ShapeStim(win, vertices=probeVert, fillColor=[-1.0, 1.0, -1.0],
                             lineWidth=0, opacity=1, size=probe_size, interpolate=False)
probe1_tl = visual.ShapeStim(win, vertices=probeVert, fillColor=(1.0, -1.0, 1.0),
                             lineWidth=0, opacity=1, size=probe_size, interpolate=False)
probe2_tl = visual.ShapeStim(win, vertices=probeVert, fillColor=[-1.0, 1.0, -1.0],
                             lineWidth=0, opacity=1, size=probe_size, interpolate=False)
probe1_bl = visual.ShapeStim(win, vertices=probeVert, fillColor=(1.0, -1.0, 1.0),
                             lineWidth=0, opacity=1, size=probe_size, interpolate=False)
probe2_bl = visual.ShapeStim(win, vertices=probeVert, fillColor=[-1.0, 1.0, -1.0],
                             lineWidth=0, opacity=1, size=probe_size, interpolate=False)
probe1_br = visual.ShapeStim(win, vertices=probeVert, fillColor=(1.0, -1.0, 1.0),
                             lineWidth=0, opacity=1, size=probe_size, interpolate=False)
probe2_br = visual.ShapeStim(win, vertices=probeVert, fillColor=[-1.0, 1.0, -1.0],
                             lineWidth=0, opacity=1, size=probe_size, interpolate=False)


# fixation bull eye
fixation = visual.Circle(win, radius=2, units='pix', lineColor='white', fillColor='black')

# location markers are displayed in each corner between the 2 probes to check they are balanced around these locations.
# loc_marker_tr = visual.Circle(win, radius=2, units='pix', lineColor='green', fillColor='red')
# loc_marker_tl = visual.Circle(win, radius=2, units='pix', lineColor='green', fillColor='red')
# loc_marker_bl = visual.Circle(win, radius=2, units='pix', lineColor='green', fillColor='red')
# loc_marker_br = visual.Circle(win, radius=2, units='pix', lineColor='green', fillColor='red')

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
                               font='Arial', height=20,
                               color='white')

# BREAKS
take_break = 100
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
trial_number = 0
for step in range(n_trials_per_stair):
    np.random.shuffle(stairs)
    for thisStair in stairs:

        # Trial, stair and step
        trial_number += 1
        stair_idx = thisStair.extraInfo['stair_idx']
        print(f"\ntrial_number: {trial_number}, stair_idx: {stair_idx}, thisStair: {thisStair}, step: {step}")

        # condition (Separation, ISI)
        sep = sep_vals_list[stair_idx]
        # separation expressed as degrees.
        if -1 < sep < 99:
            sep * pixel_mm_deg_dict['diag_deg']
        else:
            sep_deg = 0
        ISI = ISI_vals_list[stair_idx]
        print(f"ISI: {ISI}, sep: {sep}")

        # Luminance (staircase varies probeLum)
        probeLum = thisStair.next()
        probeColor255 = int(probeLum * LumColor255Factor)  # rgb255 are ints.
        probeColor1 = (probeColor255 * Color255Color1Factor) - 1
        probe1_tr.color = [probeColor1, probeColor1, probeColor1]
        probe2_tr.color = [probeColor1, probeColor1, probeColor1]
        probe1_tl.color = [probeColor1, probeColor1, probeColor1]
        probe2_tl.color = [probeColor1, probeColor1, probeColor1]
        probe1_bl.color = [probeColor1, probeColor1, probeColor1]
        probe2_bl.color = [probeColor1, probeColor1, probeColor1]
        probe1_br.color = [probeColor1, probeColor1, probeColor1]
        probe2_br.color = [probeColor1, probeColor1, probeColor1]
        print(f"probeLum: {probeLum}, probeColor255: {probeColor255}, probeColor1: {probeColor1}")

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

        '''set probe timings'''
        # VARIABLE FIXATION TIME
        # to reduce anticipatory effects that might arise from fixation always being same length.
        # if False, vary_fix == .5 seconds, so t_fixation is 1 second.
        # if Ture, vary_fix is between 0 and 1 second, so t_fixation is between .5 and 1.5 seconds.
        vary_fix = int(fps / 2)
        if vary_fixation:
            vary_fix = np.random.randint(0, fps)

        # timing in frames
        # fixation time is now 70ms shorter than rad_flow1, as we can have
        # priliminary bg_motion.
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

        print(f"t_fixation: {t_fixation}\n"
              f"t_probe_1: {t_probe_1}\n"
              f"t_ISI: {t_ISI}\n"
              f"t_probe_2: {t_probe_2}\n"
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

                # FIXATION
                if t_fixation >= frameN > 0:
                    fixation.setRadius(3)
                    fixation.draw()

                    # loc_marker_tr.draw()
                    # loc_marker_tl.draw()
                    # loc_marker_bl.draw()
                    # loc_marker_br.draw()

                    # reset timer to start with probe1 presentation.
                    resp.clock.reset()

                    # start recording frame intervals
                    if record_fr_durs:
                        win.recordFrameIntervals = True

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
                    if record_fr_durs:
                        win.recordFrameIntervals = False
                        total_recorded_fr = len(win.frameIntervals)
                        fr_recorded_list.append(total_recorded_fr)
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
                        resp.keys = theseKeys[-1]  # just the last key pressed
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
        thisExp.addData('exp_type', exp_type)
        thisExp.addData('expName', expName)
        thisExp.addData('monitor_name', monitor_name)
        thisExp.addData('selected_fps', fps)
        thisExp.addData('actual_fps', actualFrameRate)
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
    print(f"{total_dropped_fr}/{total_recorded_fr} dropped in total (expected: {round(expected_fr_ms, 2)}ms, 'dropped' if > {round(frame_err_ms, 2)})")
    plt.plot(win.frameIntervals)
    plt.title(f"{monitor_name}, {fps}Hz, {expInfo['date']}\n{total_dropped_fr}/{total_recorded_fr} dropped fr (expected: {round(expected_fr_ms, 2)}ms, 'dropped' if > {round(frame_err_ms, 2)})")
    plt.vlines(x=fr_recorded_list, ymin=min(win.frameIntervals), ymax=max(win.frameIntervals), colors='silver', linestyles='dashed')
    plt.axhline(y=frame_err_sec, color='red', linestyle='dashed')
    fig_name = filename = f'{_thisDir}{os.sep}' \
                          f'{expName}{os.sep}' \
                          f'{exp_type}{os.sep}' \
                          f'{probe_coherence}{os.sep}' \
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
