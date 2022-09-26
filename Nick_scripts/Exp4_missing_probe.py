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


'''
Missing probe study.
Probes will appear in three corners, participants identify corner without probe.

I think I will design it so that all 4 corners have probe motion planned, 
then just don't draw the corner that is selected

We want to manipulate the coherence of the movement of the three probes.
Motion is coherent if
1. all tangental: (2) clockwise or anti-clockwise.  [rotation]
2. all radial: (2) inward/outward       [radial]
3. mixed: same quadrant (4) top-right; top-left; bottom-left; bottom-right  [shift]
This gives 8 possibilities for clear coherent motion

Non-coherent if
Mixed - two radial (one in, one out), two tangent (one CW one CCW).
Only with this pattern can you be sure that when you delete one the other three aren't good.  
I think there are 4x3x2=24 possible combinations of this pattern.
 

AVOID less coherent - but still a pattern
4. same general direction: all contain upward motion, but relies on symmetry
    - e.g., two move up and left, two move up and right
  
'''

# sets psychoPy to only log critical messages
# logging.console.setLevel(logging.CRITICAL)

# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# Monitor config from monitor centre
monitor_name = 'Asus_VG24'  # 'NickMac' 'asus_cal' 'Asus_VG24' 'HP_24uh' 'ASUS_2_13_240Hz' 'Iiyama_2_18' 'Nick_work_laptop'
# gamma set at 2.1  [####### this comment is incorrect, its set above i think ############]


# Store info about the experiment session
expName = 'EXP4_missing_probe'  # from the Builder filename that created this script

expInfo = {'1. Participant': 'nicktest',
           '1. run_number': '1',
           '2. Probe duration in frames at 240hz': [2, 50, 100],
           '3. fps': [60, 144, 240],
           '4_Trials_counter': [True, False],
           '5_vary_fixation': [False, True]}

# GUI
dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
if not dlg.OK:
    core.quit()  # user pressed escape

expInfo['date'] = datetime.now().strftime("%d/%m/%Y")
expInfo['time'] = datetime.now().strftime("%H:%M:%S")

# GUI SETTINGS
participant_name = expInfo['1. Participant']
run_number = int(expInfo['1. run_number'])
n_trials_per_stair = 25
probe_duration = int(expInfo['2. Probe duration in frames at 240hz'])
probe_ecc = 4
fps = int(expInfo['3. fps'])
orientation = 'tangent'  # expInfo['5. Probe orientation']
trials_counter = expInfo['4_Trials_counter']
vary_fixation = expInfo['5_vary_fixation']


# VARIABLES
'''Distances between probes (spatially and temporally)
For 1probe condition, use separation==99.
For concurrent probes, use ISI==-1.
'''
# separations = [0, 2, 4, 6]  # select from [0, 1, 2, 3, 6, 18, 99]
separations = [0, 3, 6]  # select from [0, 1, 2, 3, 6, 18, 99]
print(f'separations: {separations}')
# ISI_values = [0, 2, 4, 6]  # select from [-1, 0, 2, 4, 6, 9, 12, 24]
ISI_values = [-1, 3, 6]  # select from [-1, 0, 2, 4, 6, 9, 12, 24]
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

# FILENAME
filename = f'{_thisDir}{os.sep}' \
           f'{expName}{os.sep}' \
           f'{participant_name}{os.sep}' \
           f'{participant_name}_{run_number}{os.sep}' \
           f'{participant_name}_output'
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
                    fullscr=use_full_screen,
                    )

print(f"check win.size: {win.size}")
widthPix = widthPix/2
heightPix = heightPix/2
print(f"widthPix: {widthPix}, hight: {heightPix}")
widthPix, heightPix = win.size
print(f"check win.size: {win.size}")

'''
DIctionary of co-ordinated probe motion.  There are 3 coherent types:
rotational, radial and translational.  
There are two roational (CW and CCW), two rdial (in/out) and four translational (4 corners).
An experiment can select one of these types of coherent motion.
There are eight types of incoherent motion.
An experiment will use one type of coherent, selecting the version randomly, along with random selection of incoherent motion.
'''

probe_types = ['rotation', 'radial', 'translation', 'incoherent']

probes_dict = {'type': {'rotation': {'n_examples': 2, 'names': ['CW', 'CCW'],
                                     'examples':
                                         {'CW':
                                              {'tr': {'orientation': 'tangent', 'jump': 1, 'direction': 'CW'},
                                               'tl': {'orientation': 'tangent', 'jump': 1, 'direction': 'CW'},
                                               'bl': {'orientation': 'tangent', 'jump': 1, 'direction': 'CW'},
                                               'br': {'orientation': 'tangent', 'jump': 1, 'direction': 'CW'}},
                                          'CCW':
                                               {'tr': {'orientation': 'tangent', 'jump': -1, 'direction': 'CCW'},
                                                'tl': {'orientation': 'tangent', 'jump': -1, 'direction': 'CCW'},
                                                'bl': {'orientation': 'tangent', 'jump': -1, 'direction': 'CCW'},
                                                'br': {'orientation': 'tangent', 'jump': -1, 'direction': 'CCW'}}}},
                        'radial': {'n_examples': 2, 'names': ['in', 'out'],
                                    'examples':
                                        {'in':
                                             {'tr': {'orientation': 'radial', 'jump': 1, 'direction': 'in'},
                                              'tl': {'orientation': 'radial', 'jump': 1, 'direction': 'in'},
                                              'bl': {'orientation': 'radial', 'jump': 1, 'direction': 'in'},
                                              'br': {'orientation': 'radial', 'jump': 1, 'direction': 'in'}},
                                         'out':
                                             {'tr': {'orientation': 'radial', 'jump': -1, 'direction': 'out'},
                                              'tl': {'orientation': 'radial', 'jump': -1, 'direction': 'out'},
                                              'bl': {'orientation': 'radial', 'jump': -1, 'direction': 'out'},
                                              'br': {'orientation': 'radial', 'jump': -1, 'direction': 'out'}}}},
                        'translation': {'n_examples': 4, 'names': ['tr', 'tl', 'bl', 'br'],
                                        'examples':
                                            {'tr':
                                                 {'tr': {'orientation': 'radial', 'jump': -1, 'direction': 'out'},
                                                  'tl': {'orientation': 'tangent', 'jump': 1, 'direction': 'CW'},
                                                  'bl': {'orientation': 'radial', 'jump': 1, 'direction': 'in'},
                                                  'br': {'orientation': 'tangent', 'jump': -1, 'direction': 'CCW'}},
                                             'tl':
                                                 {'tr': {'orientation': 'tangent', 'jump': -1, 'direction': 'CCW'},
                                                  'tl': {'orientation': 'radial', 'jump': -1, 'direction': 'out'},
                                                  'bl': {'orientation': 'tangent', 'jump': 1, 'direction': 'CW'},
                                                  'br': {'orientation': 'radial', 'jump': 1, 'direction': 'in'}},
                                             'bl':
                                                 {'tr': {'orientation': 'radial', 'jump': 1, 'direction': 'in'},
                                                  'tl': {'orientation': 'tangent', 'jump': -1, 'direction': 'CCW'},
                                                  'bl': {'orientation': 'radial', 'jump': -1, 'direction': 'out'},
                                                  'br': {'orientation': 'tangent', 'jump': 1, 'direction': 'CW'}},
                                             'br':
                                                 {'tr': {'orientation': 'tangent', 'jump': 1, 'direction': 'CW'},
                                                  'tl': {'orientation': 'radial', 'jump': 1, 'direction': 'in'},
                                                  'bl': {'orientation': 'tangent', 'jump': -1, 'direction': 'CCW'},
                                                  'br': {'orientation': 'radial', 'jump': -1, 'direction': 'out'}}}},
                        'incoherent': {'n_examples': 8, 'names': ['inc1', 'inc2', 'inc3', 'inc4', 'inc5', 'inc6', 'inc7', 'inc8'],
                                       'examples':
                                           {'inc1':
                                                {'tr': {'orientation': 'radial', 'jump': -1, 'direction': 'out'},
                                                 'tl': {'orientation': 'tangent', 'jump': -1, 'direction': 'CCW'},
                                                 'bl': {'orientation': 'tangent', 'jump': 1, 'direction': 'CW'},
                                                 'br': {'orientation': 'radial', 'jump': 1, 'direction': 'in'}},
                                            'inc2':
                                                {'tr': {'orientation': 'radial', 'jump': 1, 'direction': 'in'},
                                                 'tl': {'orientation': 'radial', 'jump': -1, 'direction': 'out'},
                                                 'bl': {'orientation': 'tangent', 'jump': 1, 'direction': 'CW'},
                                                 'br': {'orientation': 'tangent', 'jump': 1, 'direction': 'CW'}},
                                            'inc3':
                                                {'tr': {'orientation': 'tangent', 'jump': -1, 'direction': 'CW'},
                                                 'tl': {'orientation': 'radial', 'jump': 1, 'direction': 'in'},
                                                 'bl': {'orientation': 'radial', 'jump': -1, 'direction': 'out'},
                                                 'br': {'orientation': 'tangent', 'jump': -1, 'direction': 'CCW'}},
                                            'inc4':
                                                {'tr': {'orientation': 'tangent', 'jump': -1, 'direction': 'CCW'},
                                                 'tl': {'orientation': 'tangent', 'jump': -1, 'direction': 'CCW'},
                                                 'bl': {'orientation': 'radial', 'jump': 1, 'direction': 'in'},
                                                 'br': {'orientation': 'radial', 'jump': -1, 'direction': 'out'}},
                                            'inc5':
                                                {'tr': {'orientation': 'radial', 'jump': 1, 'direction': 'in'},
                                                 'tl': {'orientation': 'tangent', 'jump': 1, 'direction': 'CW'},
                                                 'bl': {'orientation': 'tangent', 'jump': -1, 'direction': 'CW'},
                                                 'br': {'orientation': 'radial', 'jump': -1, 'direction': 'out'}},
                                            'inc6':
                                                {'tr': {'orientation': 'radial', 'jump': -1, 'direction': 'out'},
                                                 'tl': {'orientation': 'radial', 'jump': 1, 'direction': 'in'},
                                                 'bl': {'orientation': 'tangent', 'jump': -1, 'direction': 'CCW'},
                                                 'br': {'orientation': 'tangent', 'jump': -1, 'direction': 'CCW'}},
                                            'inc7':
                                                {'tr': {'orientation': 'tangent', 'jump': 1, 'direction': 'CW'},
                                                 'tl': {'orientation': 'radial', 'jump': -1, 'direction': 'out'},
                                                 'bl': {'orientation': 'radial', 'jump': 1, 'direction': 'in'},
                                                 'br': {'orientation': 'tangent', 'jump': 1, 'direction': 'CCW'}},
                                            'inc8':
                                                {'tr': {'orientation': 'tangent', 'jump': 1, 'direction': 'CW'},
                                                 'tl': {'orientation': 'tangent', 'jump': 1, 'direction': 'CW'},
                                                 'bl': {'orientation': 'radial', 'jump': -1, 'direction': 'out'},
                                                 'br': {'orientation': 'radial', 'jump': 1, 'direction': 'in'}}}}
                        }
               }

for probe_type in probe_types:
    print(probe_type)
    # print(probes_dict['type'].keys())
    n_examples = int(probes_dict['type'][probe_type]['n_examples'])
    for ex in range(n_examples):
        example_name = probes_dict['type'][probe_type]['names'][ex]
        tr_ori = probes_dict['type'][probe_type]['examples'][example_name]['tr']['orientation']
        tr_jump = probes_dict['type'][probe_type]['examples'][example_name]['tr']['jump']
        tr_dir = probes_dict['type'][probe_type]['examples'][example_name]['tr']['direction']

        tl_ori = probes_dict['type'][probe_type]['examples'][example_name]['tl']['orientation']
        tl_jump = probes_dict['type'][probe_type]['examples'][example_name]['tl']['jump']
        tl_dir = probes_dict['type'][probe_type]['examples'][example_name]['tl']['direction']

        bl_ori = probes_dict['type'][probe_type]['examples'][example_name]['bl']['orientation']
        bl_jump = probes_dict['type'][probe_type]['examples'][example_name]['bl']['jump']
        bl_dir = probes_dict['type'][probe_type]['examples'][example_name]['bl']['direction']

        br_ori = probes_dict['type'][probe_type]['examples'][example_name]['br']['orientation']
        br_jump = probes_dict['type'][probe_type]['examples'][example_name]['br']['jump']
        br_dir = probes_dict['type'][probe_type]['examples'][example_name]['br']['direction']

        print(f"\n{probe_type}, {example_name}\n"
              f"tr: {tr_dir}, {tr_ori}, {tr_jump}\n"
              f"tl: {tl_dir}, {tl_ori}, {tl_jump}\n"
              f"bl: {bl_dir}, {bl_ori}, {bl_jump}\n"
              f"br: {br_dir}, {br_ori}, {br_jump}\n"
              )

#
# # ELEMENTS
# # fixation bull eye
# fixation = visual.Circle(win, radius=2, units='pix',
#                          lineColor='white', fillColor='black')
#
# # PROBEs
# expInfo['6. Probe size'] = '5pixels'  # ignore this, all experiments use 5pixel probes now.
# probeVert = [(0, 0), (1, 0), (1, 1), (2, 1), (2, -1), (1, -1),
#              (1, -2), (-1, -2), (-1, -1), (0, -1)]
#
# # # 45=top-right, 135=top-left, 225=bottom-left, 315=bottom-right
# probe1_45tr = visual.ShapeStim(win, vertices=probeVert, fillColor=(1.0, -1.0, 1.0),
#                           lineWidth=0, opacity=1, size=1, interpolate=False)
# probe2_45tr = visual.ShapeStim(win, vertices=probeVert, fillColor=[-1.0, 1.0, -1.0],
#                           lineWidth=0, opacity=1, size=1, interpolate=False)
# probe1_135tl = visual.ShapeStim(win, vertices=probeVert, fillColor=(1.0, -1.0, 1.0),
#                           lineWidth=0, opacity=1, size=1, interpolate=False)
# probe2_135tl = visual.ShapeStim(win, vertices=probeVert, fillColor=[-1.0, 1.0, -1.0],
#                           lineWidth=0, opacity=1, size=1, interpolate=False)
# probe1_225bl = visual.ShapeStim(win, vertices=probeVert, fillColor=(1.0, -1.0, 1.0),
#                           lineWidth=0, opacity=1, size=1, interpolate=False)
# probe2_225bl = visual.ShapeStim(win, vertices=probeVert, fillColor=[-1.0, 1.0, -1.0],
#                           lineWidth=0, opacity=1, size=1, interpolate=False)
# probe1_315br = visual.ShapeStim(win, vertices=probeVert, fillColor=(1.0, -1.0, 1.0),
#                           lineWidth=0, opacity=1, size=1, interpolate=False)
# probe2_315br = visual.ShapeStim(win, vertices=probeVert, fillColor=[-1.0, 1.0, -1.0],
#                           lineWidth=0, opacity=1, size=1, interpolate=False)
#
#
# # dist_from_fix is a constant to get 4dva distance from fixation,
# dist_from_fix = round((tan(deg2rad(probe_ecc)) * viewdistPix) / sqrt(2))
#
# # MOUSE - hide cursor
# myMouse = event.Mouse(visible=False)
#
# # # KEYBOARD
# resp = event.BuilderKeyResponse()
#
# # INSTRUCTION
# instructions = visual.TextStim(win=win, name='instructions',
#                                text="\n\n\n\n\n\nFocus on the small circle at the centre of the screen.\n\n"
#                                     "A small white probe will briefly flash on screen,\n"
#                                     "press the key related to the location of the probe:\n\n"
#                                     "[4]/[Q] top-left\t\t\t[5]/[W] top-right\n\n\n\n"
#                                     "[1]/[A] bottom-left\t\t\t[2]/[S] bottom-right.\n\n\n"
#                                     "Some flashes will seem bright and easy to see\n"
#                                     "Some will be dim and harder to spot\n"
#                                     "Some will be so dim that you won't see them, so just guess!\n\n"
#                                     "You don't need to think for long, respond quickly, but try to push press the correct key!\n\n"
#                                     "Don't let your eyes wander, keep focussed on the circle in the middle throughout.",
#                                font='Arial', height=20,
#                                color='white')
#
# # Trial counter
# trials_counter = visual.TextStim(win=win, name='trials_counter', text="???",
#                                  font='Arial', height=20,
#                                  # default set to black (e.g., invisible)
#                                  color='black',
#                                  pos=[-widthPix * .45, -heightPix * .45])
# if trials_counter:
#     # if trials counter yes, change colour to white.
#     trials_counter.color = 'white'
#
# # BREAKS
# take_break = 76
# total_n_trials = int(n_trials_per_stair * n_stairs)
# # take_break = int(total_n_trials/2)+1
# print(f"take_break every {take_break} trials.")
# breaks = visual.TextStim(win=win, name='breaks',
#                          # text="turn on the light and take at least 30-seconds break.",
#                          text="Break\n"
#                               "Remember, if you don't see the flash, just guess!\n"
#                               "Keep focussed on the circle in the middle of the screen.",
#                          font='Arial', pos=[0, 0], height=20, ori=0, color=[255, 255, 255],
#                          colorSpace='rgb255', opacity=1, languageStyle='LTR', depth=0.0)
#
# end_of_exp = visual.TextStim(win=win, name='end_of_exp',
#                              text="You have completed this experiment.\n"
#                                   "Thank you for your time.\n\n"
#                                   "Press any key to return to the desktop.",
#                              font='Arial', height=20)
#
# while not event.getKeys():
#     fixation.setRadius(3)
#     fixation.draw()
#     instructions.draw()
#     win.flip()
#
# # STAIRCASE
# expInfo['stair_list'] = list(range(n_stairs))
# expInfo['n_trials_per_stair'] = n_trials_per_stair
#
# stairStart = maxLum
# miniVal = bgLum
# maxiVal = maxLum
#
# stairs = []
# for stair_idx in expInfo['stair_list']:
#
#     thisInfo = copy.copy(expInfo)
#     thisInfo['stair_idx'] = stair_idx
#     stair_name = stair_names_list[stair_idx]
#
#     thisStair = Staircase(name=stair_name,
#                           type='simple',
#                           value=stairStart,
#                           C=stairStart*0.6,  # typically, 60% of reference stimulus
#                           minRevs=3,
#                           minTrials=n_trials_per_stair,
#                           minVal=miniVal,
#                           maxVal=maxiVal,
#                           targetThresh=0.75,
#                           extraInfo=thisInfo)
#     stairs.append(thisStair)
#
# trial_number = 0
#
# # EXPERIMENT
# for step in range(n_trials_per_stair):
#     shuffle(stairs)
#     for thisStair in stairs:
#
#         trial_number = trial_number + 1
#
#         stair_idx = thisStair.extraInfo['stair_idx']
#         print(f"\ntrial_number: {trial_number}, stair_idx: {stair_idx}, thisStair: {thisStair}, step: {step}")
#
#         sep = sep_vals_list[stair_idx]
#         ISI = ISI_vals_list[stair_idx]
#         print(f"ISI: {ISI}, sep: {sep}")
#
#         # staircase varies probeLum
#         probeLum = thisStair.next()
#         probeColor255 = int(probeLum * LumColor255Factor)  # rgb255 are ints.
#         probeColor1 = (probeColor255 * Color255Color1Factor) - 1
#         # probe1.color = [probeColor1, probeColor1, probeColor1]
#         # probe2.color = [probeColor1, probeColor1, probeColor1]
#         print(f"probeLum: {probeLum}, probeColor255: {probeColor255}, probeColor1: {probeColor1}")
#
#         # PROBE LOCATION
#         # corners go CCW(!) 45=top-right, 135=top-left, 225=bottom-left, 315=bottom-right
#         corner = random.choice([45, 135, 225, 315])
#
#         # todo: add separate target jump variables for each corner.
#         # direction in which the probe jumps : -1 == CCW/outward; 1 == CW/inward
#         # target_jump = random.choice([1, -1])
#         top_right_jump = 1
#         top_left_jump = 1
#         bottom_left_jump = 1
#         bottom_right_jump = 1
#
#         ori_45tr = 'tangent'
#         ori_135tl = 'tangent'
#         ori_225bl = 'tangent'
#         ori_312br = 'tangent'
#
#         # set probe ori
#         # if corner == 45:
#         # in top-right corner, both x and y increase (right and up)
#         # todo: update p1_x and p1_y for each corner
#         p1_x = dist_from_fix * 1
#         p1_y = dist_from_fix * 1
#         #  'orientation' here refers to the relationship between probes,
#         #  whereas probe1.ori refers to rotational angle of probe stimulus
#         if orientation == 'tangent':
#             if target_jump == 1:  # CCW
#                 probe1_45tr.ori = 180
#                 probe2_45tr.ori = 0
#                 # probe2 is left and up from probe1
#                 probe2_45tr.pos = [p1_x + sep - 1, p1_y - sep]
#             elif target_jump == -1:  # CW
#                 probe1_45tr.ori = 0
#                 probe2_45tr.ori = 180
#                 # probe2 is right and down from probe1
#                 probe2_45tr.pos = [p1_x - sep + 1, p1_y + sep]
#         elif orientation == 'radial':
#             if target_jump == 1:  # inward
#                 probe1_45tr.ori = 270
#                 probe2_45tr.ori = 90
#                 # probe2 is left and down from probe1
#                 probe2_45tr.pos = [p1_x - sep + 1, p1_y - sep]
#             elif target_jump == -1:  # outward
#                 probe1_45tr.ori = 90
#                 probe2_45tr.ori = 270
#                 # probe2 is right and up from probe1
#                 probe2_45tr.pos = [p1_x + sep - 1, p1_y + sep]
#         # elif corner == 135:
#         # in top-left corner, x decreases (left) and y increases (up)
#         p1_x = dist_from_fix * -1
#         p1_y = dist_from_fix * 1
#         if orientation == 'tangent':
#             if target_jump == 1:  # CCW
#                 probe1_135tl.ori = 90
#                 probe2_135tl.ori = 270
#                 # probe2 is right and up from probe1
#                 probe2_135tl.pos = [p1_x + sep - 1, p1_y + sep]
#             elif target_jump == -1:  # CW
#                 probe1_135tl.ori = 270
#                 probe2_135tl.ori = 90
#                 # probe2 is left and down from probe1
#                 probe2_135tl.pos = [p1_x - sep + 1, p1_y - sep]
#         elif orientation == 'radial':
#             if target_jump == 1:  # inward
#                 probe1_135tl.ori = 180
#                 probe2_135tl.ori = 0
#                 # probe2 is right and down from probe1
#                 probe2_135tl.pos = [p1_x + sep - 1, p1_y - sep]
#             elif target_jump == -1:  # outward
#                 probe1_135tl.ori = 0
#                 probe2_135tl.ori = 180
#                 # probe2 is left and up from probe1
#                 probe2_135tl.pos = [p1_x - sep + 1, p1_y + sep]
#         # elif corner == 225:
#         # in bottom left corner, both x and y decrease (left and down)
#         p1_x = dist_from_fix * -1
#         p1_y = dist_from_fix * -1
#         if orientation == 'tangent':
#             if target_jump == 1:  # CCW
#                 probe1_225bl.ori = 0
#                 probe2_225bl.ori = 180
#                 probe2_225bl.pos = [p1_x - sep + 1, p1_y + sep]
#             elif target_jump == -1:  # CW
#                 probe1_225bl.ori = 180
#                 probe2_225bl.ori = 0
#                 probe2_225bl.pos = [p1_x + sep - 1, p1_y - sep]
#         elif orientation == 'radial':
#             if target_jump == 1:  # inward
#                 probe1_225bl.ori = 90
#                 probe2_225bl.ori = 270
#                 # probe2 is right and up from probe1
#                 probe2_225bl.pos = [p1_x + sep - 1, p1_y + sep]
#             elif target_jump == -1:  # outward
#                 probe1_225bl.ori = 270
#                 probe2_225bl.ori = 90
#                 # probe2 is left and down from probe1
#                 probe2_225bl.pos = [p1_x - sep + 1, p1_y - sep]
#         # else:
#         corner = 315
#         # in bottom-right corner, x increases (right) and y decreases (down)
#         p1_x = dist_from_fix * 1
#         p1_y = dist_from_fix * -1
#         if orientation == 'tangent':
#             if target_jump == 1:  # CCW
#                 probe1_315br.ori = 270
#                 probe2_315br.ori = 90
#                 probe2_315br.pos = [p1_x - sep + 1, p1_y - sep]
#             elif target_jump == -1:  # CW
#                 probe1_315br.ori = 90
#                 probe2_315br.ori = 270
#                 probe2_315br.pos = [p1_x + sep - 1, p1_y + sep]
#         elif orientation == 'radial':
#             if target_jump == 1:  # inward
#                 probe1_315br.ori = 0
#                 probe2_315br.ori = 180
#                 # probe2 is left and up from probe1
#                 probe2_315br.pos = [p1_x - sep + 1, p1_y + sep]
#             elif target_jump == -1:  # outward
#                 probe1_315br.ori = 180
#                 probe2_315br.ori = 0
#                 # probe2 is right and down from probe1
#                 probe2_315br.pos = [p1_x + sep - 1, p1_y - sep]
#
#         # todo: update these positions for each probe1
#         probe1_45tr.pos = [p1_x, p1_y]
#
#         # to avoid fixation times always being the same which might increase
#         # anticipatory effects,
#         # add in a random number of frames (up to 1 second) to fixation time
#         vary_fix = 0
#         if vary_fixation:
#             vary_fix = np.random.randint(0, fps)
#
#         # timing in frames
#         # fixation time is now 70ms shorter than rad_flow1, as we can have
#         # priliminary bg_motion.
#         t_fixation = (fps / 2) + vary_fix
#         t_probe_1 = t_fixation + probe_duration
#         t_ISI = t_probe_1 + ISI
#         t_probe_2 = t_ISI + probe_duration
#         t_response = t_probe_2 + 10000 * fps  # essentially unlimited time to respond
#
#         # repeat the trial if [r] has been pressed
#         repeat = True
#         while repeat:
#             frameN = -1
#
#             # take a break every ? trials
#             if (trial_number % take_break == 1) & (trial_number > 1):
#                 continueRoutine = False
#                 breaks.draw()
#                 win.flip()
#                 while not event.getKeys():
#                     continueRoutine = True
#             else:
#                 continueRoutine = True
#
#             while continueRoutine:
#                 frameN = frameN + 1
#
#                 # FIXATION
#                 if t_fixation >= frameN > 0:
#                     fixation.setRadius(3)
#                     fixation.draw()
#                     trials_counter.text = f"{trial_number}/{total_n_trials}"
#                     trials_counter.draw()
#
#                     # reset timer to start with probe1 presentation.
#                     resp.clock.reset()
#
#
#                 # PROBE 1
#                 if t_probe_1 >= frameN > t_fixation:
#                     # todo: add all probe1s
#                     probe1.draw()
#
#                     if ISI == -1:  # SIMULTANEOUS CONDITION (concurrent)
#                         if sep <= 18:  # don't draw 2nd probe in 1probe cond (sep==99)
#                             # todo: add all probe2s
#                             probe2.draw()
#                     fixation.setRadius(3)
#                     fixation.draw()
#                     trials_counter.draw()
#
#                 # ISI
#                 if t_ISI >= frameN > t_probe_1:
#                     fixation.setRadius(3)
#                     fixation.draw()
#                     trials_counter.draw()
#
#                 # PROBE 2
#                 if t_probe_2 >= frameN > t_ISI:
#                     if ISI >= 0:
#                         if sep <= 18:  # don't draw 2nd probe in 1probe cond (sep==99)
#                             # todo: add all probe2s
#                             probe2.draw()
#                     fixation.setRadius(3)
#                     fixation.draw()
#                     trials_counter.draw()
#
#                 # ANSWER
#                 if frameN > t_probe_2:
#                     fixation.setRadius(2)
#                     fixation.draw()
#                     trials_counter.draw()
#
#                     # ANSWER
#                     theseKeys = event.getKeys(keyList=['num_5', 'num_4', 'num_1',
#                                                        'num_2', 'w', 'q', 'a', 's'])
#
#                     if len(theseKeys) > 0:  # at least one key was pressed
#                         resp.keys = theseKeys[-1]  # just the last key pressed
#                         resp.rt = resp.clock.getTime()
#
#                         # default assume response incorrect unless meets criteria below
#                         resp.corr = 0
#
#                         if corner == 45:
#                             if (resp.keys == 'w') or (resp.keys == 'num_5'):
#                                 resp.corr = 1
#                         elif corner == 135:
#                             if (resp.keys == 'q') or (resp.keys == 'num_4'):
#                                 resp.corr = 1
#                         elif corner == 225:
#                             if (resp.keys == 'a') or (resp.keys == 'num_1'):
#                                 resp.corr = 1
#                         elif corner == 315:
#                             if (resp.keys == 's') or (resp.keys == 'num_2'):
#                                 resp.corr = 1
#
#                         repeat = False
#                         continueRoutine = False
#
#                 # regardless of frameN, check for quit
#                 if event.getKeys(keyList=["escape"]):
#                     thisExp.close()
#                     core.quit()
#
#                 # redo the trial if I think I made a mistake
#                 if event.getKeys(keyList=["r"]) or event.getKeys(keyList=['num_9']):
#                     repeat = True
#                     continueRoutine = False
#                     continue
#
#                 # refresh the screen
#                 if continueRoutine:
#                     win.flip()
#
#         thisExp.addData('trial_number', trial_number)
#         thisExp.addData('stair', stair_idx)
#         thisExp.addData('stair_name', stair_name)
#         thisExp.addData('step', step)
#         thisExp.addData('separation', sep)
#         thisExp.addData('ISI', ISI)
#         thisExp.addData('probe_jump', target_jump)
#         thisExp.addData('probeColor1', probeColor1)
#         thisExp.addData('probeColor255', probeColor255)
#         thisExp.addData('probeLum', probeLum)
#         thisExp.addData('trial_response', resp.corr)
#         thisExp.addData('corner', corner)
#         thisExp.addData('probe_ecc', probe_ecc)
#         thisExp.addData('resp.rt', resp.rt)
#         thisExp.addData('orientation', orientation)
#         thisExp.addData('vary_fixation', vary_fixation)
#         thisExp.addData('expName', expName)
#         thisExp.addData('monitor_name', monitor_name)
#         thisExp.addData('selected_fps', fps)
#         thisExp.nextEntry()
#
#         thisStair.newValue(resp.corr)   # so that the staircase adjusts itself
#
#
# print("end of experiment loop, saving data")
# thisExp.dataFileName = filename
# thisExp.close()
#
# while not event.getKeys():
#     # display end of experiment screen
#     end_of_exp.draw()
#     win.flip()
# else:
#     # close and quit once a key is pressed
#     win.close()
#     core.quit()
