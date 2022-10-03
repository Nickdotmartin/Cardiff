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

The script is designed it so that all 4 corners have probe motion planned, 
then just don't draw the missing_corner that is selected

We want to manipulate the coherence of the movement of the three probes.
Motion is coherent if
1. all tangental: (2) clockwise or anti-clockwise.  [rotation]
2. all radial: (2) inward/outward       [radial]
3. mixed: same quadrant (4) top-right; top-left; bottom-left; bottom-right  [translation]
This gives 8 possibilities for clear coherent motion

Non-coherent if
Mixed - two radial (one in, one out), two tangent (one CW one CCW).
Only with this pattern can you be sure that when you delete one the other three aren't good.  
However, with the pattern, there will always be two probes with same absolute (tr, tl, bl, br), 
or relational (in, out, cw, ccw) motion, so I have made it so that the 
incohenrent probes include all 4 absolute directions, but with two matching for cw/ccw.
 
There are no mirror symmetrical patterns (less coherent - but still a pattern)
    - e.g., two move up and left, two move up and right  
'''

# sets psychoPy to only log critical messages
# logging.console.setLevel(logging.CRITICAL)

# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# Monitor config from monitor centre
monitor_name = 'HP_24uh'  # 'NickMac' 'asus_cal' 'Asus_VG24' 'HP_24uh' 'ASUS_2_13_240Hz' 'Iiyama_2_18' 'Nick_work_laptop'

# Store info about the experiment session
expName = 'EXP4_missing_probe'  # from the Builder filename that created this script

expInfo = {'1. Participant': 'nicktest',
           '1. run_number': '1',
           '2. Probe duration in frames at 240hz': [2, 50, 100],
           '3. fps': [60, 144, 240],
           '4_Trials_counter': [True, False],
           '5_vary_fixation': [False, True],
           '6_probe_coherence': ['rotation', 'radial', 'translation']}

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
trials_counter = eval(expInfo['4_Trials_counter'])
vary_fixation = eval(expInfo['5_vary_fixation'])
probe_coherence = expInfo['6_probe_coherence']


# VARIABLES
'''Distances between probes (spatially and temporally)
For 1probe condition, use separation==99.
For concurrent probes, use ISI==-1.
'''
separations = [0, 3, 6]  # select from [0, 1, 2, 3, 6, 18, 99]
print(f'separations: {separations}')
ISI_values = [-1, 3, 6]  # select from [-1, 0, 2, 4, 6, 9, 12, 24]
print(f'ISI_values: {ISI_values}')
# repeat separation values for each ISI e.g., [0, 0, 6, 6]
sep_vals_list = list(np.repeat(separations, len(ISI_values))) + list(np.repeat(separations, len(ISI_values)))
print(f'sep_vals_list: {sep_vals_list}')
# ISI_vals_list cycles through ISIs e.g., [-1, 6, -1, 6]
ISI_vals_list = list(np.tile(ISI_values, len(separations)*2))
print(f'ISI_vals_list: {ISI_vals_list}')
coherence_list = list(np.repeat(['inc', probe_coherence[:3]], int(len(sep_vals_list)/2)))
print(f'coherence_list: {coherence_list}')

# stair_names_list joins sep_vals_list and ISI_vals_list
# e.g., ['sep0_ISI-1', 'sep0_ISI6', 'sep6_ISI-1', 'sep6_ISI6']
stair_names_list = [f'{c}_sep{s}_ISI{i}' for c, s, i in zip(coherence_list, sep_vals_list, ISI_vals_list)]
print(f'stair_names_list: {stair_names_list}')
n_stairs = len(sep_vals_list)
print(f'n_stairs: {n_stairs}')
total_n_trials = int(n_trials_per_stair * n_stairs)
print(f'total_n_trials: {total_n_trials}')


# FILENAME
filename = f'{_thisDir}{os.sep}' \
           f'{expName}{os.sep}' \
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
Dictionary of co-ordinated probe motion.  There are 3 coherent types:
rotational, radial and translational.
There are two rotational (CW and CCW), two radial (in/out) and four translational (4 corners).
An experiment can select one of these types of coherent motion.
There are eight types of incoherent motion.
An experiment will use one type of coherent motion, selecting the example randomly,
interleaved with random selections of incoherent motion.
'''

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
                        'translation': {'n_examples': 4, 'names': ['to_tr', 'to_tl', 'to_bl', 'to_br'],
                                        'examples':
                                            {'to_tr':
                                                 {'tr': {'orientation': 'radial', 'jump': -1, 'direction': 'out'},
                                                  'tl': {'orientation': 'tangent', 'jump': 1, 'direction': 'CW'},
                                                  'bl': {'orientation': 'radial', 'jump': 1, 'direction': 'in'},
                                                  'br': {'orientation': 'tangent', 'jump': -1, 'direction': 'CCW'}},
                                             'to_tl':
                                                 {'tr': {'orientation': 'tangent', 'jump': -1, 'direction': 'CCW'},
                                                  'tl': {'orientation': 'radial', 'jump': -1, 'direction': 'out'},
                                                  'bl': {'orientation': 'tangent', 'jump': 1, 'direction': 'CW'},
                                                  'br': {'orientation': 'radial', 'jump': 1, 'direction': 'in'}},
                                             'to_bl':
                                                 {'tr': {'orientation': 'radial', 'jump': 1, 'direction': 'in'},
                                                  'tl': {'orientation': 'tangent', 'jump': -1, 'direction': 'CCW'},
                                                  'bl': {'orientation': 'radial', 'jump': -1, 'direction': 'out'},
                                                  'br': {'orientation': 'tangent', 'jump': 1, 'direction': 'CW'}},
                                             'to_br':
                                                 {'tr': {'orientation': 'tangent', 'jump': 1, 'direction': 'CW'},
                                                  'tl': {'orientation': 'radial', 'jump': 1, 'direction': 'in'},
                                                  'bl': {'orientation': 'tangent', 'jump': -1, 'direction': 'CCW'},
                                                  'br': {'orientation': 'radial', 'jump': -1, 'direction': 'out'}}}},
                        'incoherent': {'n_examples': 8, 'names': ['inc0', 'inc1', 'inc2', 'inc3', 'inc4', 'inc5', 'inc6', 'inc7'],
                                       'examples':
                                           {'inc0':
                                                {'tr': {'orientation': 'radial', 'jump': -1, 'direction': 'out'},
                                                 'tl': {'orientation': 'tangent', 'jump': -1, 'direction': 'CCW'},
                                                 'bl': {'orientation': 'tangent', 'jump': -1, 'direction': 'CCW'},
                                                 'br': {'orientation': 'radial', 'jump': 1, 'direction': 'in'}},
                                            'inc1':
                                                {'tr': {'orientation': 'radial', 'jump': 1, 'direction': 'in'},
                                                 'tl': {'orientation': 'radial', 'jump': -1, 'direction': 'out'},
                                                 'bl': {'orientation': 'tangent', 'jump': -1, 'direction': 'CCW'},
                                                 'br': {'orientation': 'tangent', 'jump': -1, 'direction': 'CCW'}},
                                            'inc2':
                                                {'tr': {'orientation': 'tangent', 'jump': -1, 'direction': 'CCW'},
                                                 'tl': {'orientation': 'radial', 'jump': 1, 'direction': 'in'},
                                                 'bl': {'orientation': 'radial', 'jump': -1, 'direction': 'out'},
                                                 'br': {'orientation': 'tangent', 'jump': -1, 'direction': 'CCW'}},
                                            'inc3':
                                                {'tr': {'orientation': 'tangent', 'jump': -1, 'direction': 'CCW'},
                                                 'tl': {'orientation': 'tangent', 'jump': -1, 'direction': 'CCW'},
                                                 'bl': {'orientation': 'radial', 'jump': 1, 'direction': 'in'},
                                                 'br': {'orientation': 'radial', 'jump': -1, 'direction': 'out'}},
                                            'inc4':
                                                {'tr': {'orientation': 'radial', 'jump': 1, 'direction': 'in'},
                                                 'tl': {'orientation': 'tangent', 'jump': 1, 'direction': 'CW'},
                                                 'bl': {'orientation': 'tangent', 'jump': 1, 'direction': 'CW'},
                                                 'br': {'orientation': 'radial', 'jump': -1, 'direction': 'out'}},
                                            'inc5':
                                                {'tr': {'orientation': 'radial', 'jump': -1, 'direction': 'out'},
                                                 'tl': {'orientation': 'radial', 'jump': 1, 'direction': 'in'},
                                                 'bl': {'orientation': 'tangent', 'jump': 1, 'direction': 'CW'},
                                                 'br': {'orientation': 'tangent', 'jump': 1, 'direction': 'CW'}},
                                            'inc6':
                                                {'tr': {'orientation': 'tangent', 'jump': 1, 'direction': 'CW'},
                                                 'tl': {'orientation': 'radial', 'jump': -1, 'direction': 'out'},
                                                 'bl': {'orientation': 'radial', 'jump': 1, 'direction': 'in'},
                                                 'br': {'orientation': 'tangent', 'jump': 1, 'direction': 'CW'}},
                                            'inc7':
                                                {'tr': {'orientation': 'tangent', 'jump': 1, 'direction': 'CW'},
                                                 'tl': {'orientation': 'tangent', 'jump': 1, 'direction': 'CW'},
                                                 'bl': {'orientation': 'radial', 'jump': -1, 'direction': 'out'},
                                                 'br': {'orientation': 'radial', 'jump': 1, 'direction': 'in'}}}}
                        }
               }



# ELEMENTS


# PROBEs
expInfo['6. Probe size'] = '5pixels'  # ignore this, all experiments use 5pixel probes now.
probeVert = [(0, 0), (1, 0), (1, 1), (2, 1), (2, -1), (1, -1),
             (1, -2), (-1, -2), (-1, -1), (0, -1)]

# # 45=top-right, 135=top-left, 225=bottom-left, 315=bottom-right
probe1_tr = visual.ShapeStim(win, vertices=probeVert, fillColor=(1.0, -1.0, 1.0),
                          lineWidth=0, opacity=1, size=1, interpolate=False)
probe2_tr = visual.ShapeStim(win, vertices=probeVert, fillColor=[-1.0, 1.0, -1.0],
                          lineWidth=0, opacity=1, size=1, interpolate=False)
probe1_tl = visual.ShapeStim(win, vertices=probeVert, fillColor=(1.0, -1.0, 1.0),
                          lineWidth=0, opacity=1, size=1, interpolate=False)
probe2_tl = visual.ShapeStim(win, vertices=probeVert, fillColor=[-1.0, 1.0, -1.0],
                          lineWidth=0, opacity=1, size=1, interpolate=False)
probe1_bl = visual.ShapeStim(win, vertices=probeVert, fillColor=(1.0, -1.0, 1.0),
                          lineWidth=0, opacity=1, size=1, interpolate=False)
probe2_bl = visual.ShapeStim(win, vertices=probeVert, fillColor=[-1.0, 1.0, -1.0],
                          lineWidth=0, opacity=1, size=1, interpolate=False)
probe1_br = visual.ShapeStim(win, vertices=probeVert, fillColor=(1.0, -1.0, 1.0),
                          lineWidth=0, opacity=1, size=1, interpolate=False)
probe2_br = visual.ShapeStim(win, vertices=probeVert, fillColor=[-1.0, 1.0, -1.0],
                          lineWidth=0, opacity=1, size=1, interpolate=False)


# fixation bull eye
fixation = visual.Circle(win, radius=2, units='pix',
                         lineColor='white', fillColor='black')

# dist_from_fix is a constant to get 4dva distance from fixation,
dist_from_fix = round((tan(deg2rad(probe_ecc)) * viewdistPix) / sqrt(2))

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
# take_break = int(total_n_trials/2)+1
print(f"take_break every {take_break} trials.")
breaks = visual.TextStim(win=win, name='breaks',
                         # text="turn on the light and take at least 30-seconds break.",
                         text="Break\nTurn on the light and take at least 30-seconds break.\n"
                              "Remember, if you don't see the flash, just guess!\n"
                              "Keep focussed on the circle in the middle of the screen.",
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
                          C=stairStart*0.6,  # typically, 60% of reference stimulus
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

        stair_idx = thisStair.extraInfo['stair_idx']
        print(f"\ntrial_number: {trial_number}, stair_idx: {stair_idx}, thisStair: {thisStair}, step: {step}")

        coherence = coherence_list[stair_idx]
        sep = sep_vals_list[stair_idx]
        ISI = ISI_vals_list[stair_idx]

        if coherence in ['incoherent', 'inc']:
            neg_sep = 0-sep
            if sep == 0:
                neg_sep = -.1
        else:
            neg_sep = sep
        print(f"coherence: {coherence}, ISI: {ISI}, sep: {sep} (neg_sep: {neg_sep})")

        # staircase varies probeLum
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

        # PROBE LOCATION
        missing_corner = random.choice(['tr', 'tl', 'bl', 'br'])

        if coherence == 'rot':
            probes_type = 'rotation'
        elif coherence == 'rad':
            probes_type = 'radial'
        elif coherence == 'tra':
            probes_type = 'translation'
        elif coherence == 'inc':
            probes_type = 'incoherent'
        else:
            raise ValueError('coherence should be rot, rad, tra or inc')
        print(f"probes_type: {probes_type}")

        n_examples = int(probes_dict['type'][probes_type]['n_examples'])
        this_example = random.choice(list(range(n_examples)))
        print(f'this_example: {this_example} from {list(range(n_examples))}')

        example_name = probes_dict['type'][probes_type]['names'][this_example]
        tr_ori = probes_dict['type'][probes_type]['examples'][example_name]['tr']['orientation']
        tr_jump = probes_dict['type'][probes_type]['examples'][example_name]['tr']['jump']
        tr_dir = probes_dict['type'][probes_type]['examples'][example_name]['tr']['direction']

        tl_ori = probes_dict['type'][probes_type]['examples'][example_name]['tl']['orientation']
        tl_jump = probes_dict['type'][probes_type]['examples'][example_name]['tl']['jump']
        tl_dir = probes_dict['type'][probes_type]['examples'][example_name]['tl']['direction']

        bl_ori = probes_dict['type'][probes_type]['examples'][example_name]['bl']['orientation']
        bl_jump = probes_dict['type'][probes_type]['examples'][example_name]['bl']['jump']
        bl_dir = probes_dict['type'][probes_type]['examples'][example_name]['bl']['direction']

        br_ori = probes_dict['type'][probes_type]['examples'][example_name]['br']['orientation']
        br_jump = probes_dict['type'][probes_type]['examples'][example_name]['br']['jump']
        br_dir = probes_dict['type'][probes_type]['examples'][example_name]['br']['direction']

        print(f"\n{probes_type}, {example_name}\n"
              f"tr: {tr_dir}, {tr_ori}, {tr_jump}\n"
              f"tl: {tl_dir}, {tl_ori}, {tl_jump}\n"
              f"bl: {bl_dir}, {bl_ori}, {bl_jump}\n"
              f"br: {br_dir}, {br_ori}, {br_jump}\n"
              )


        # set probe ori
        # in top-right corner, both x and y increase (right and up)
        tr_p1_x = dist_from_fix * 1
        tr_p1_y = dist_from_fix * 1
        probe1_tr.pos = [tr_p1_x, tr_p1_y]
        #  'orientation' (e.g., tr_ori) here refers to the relationship between probes,
        #  whereas probe1_tr.ori refers to rotational angle of probe stimulus
        if tr_ori == 'tangent':
            if tr_jump == 1:  # CCW
                probe1_tr.ori = 180
                probe2_tr.ori = 0
                # probe2 is left and up from probe1
                probe2_tr.pos = [tr_p1_x + sep - 1, tr_p1_y - sep]
            elif tr_jump == -1:  # CW
                probe1_tr.ori = 0
                probe2_tr.ori = 180
                # probe2 is right and down from probe1
                probe2_tr.pos = [tr_p1_x - sep + 1, tr_p1_y + sep]
        elif tr_ori == 'radial':
            if tr_jump == 1:  # inward
                probe1_tr.ori = 270
                probe2_tr.ori = 90
                # probe2 is left and down from probe1
                probe2_tr.pos = [tr_p1_x - sep + 1, tr_p1_y - sep]
            elif tr_jump == -1:  # outward
                probe1_tr.ori = 90
                probe2_tr.ori = 270
                # probe2 is right and up from probe1
                probe2_tr.pos = [tr_p1_x + sep - 1, tr_p1_y + sep]

        # in top-left corner, x decreases (left) and y increases (up)
        tl_p1_x = dist_from_fix * -1
        tl_p1_y = dist_from_fix * 1
        probe1_tl.pos = [tl_p1_x, tl_p1_y]
        if tl_ori == 'tangent':
            if tl_jump == 1:  # CCW
                probe1_tl.ori = 90
                probe2_tl.ori = 270
                # probe2 is right and up from probe1
                probe2_tl.pos = [tl_p1_x + sep - 1, tl_p1_y + sep]
            elif tl_jump == -1:  # CW
                probe1_tl.ori = 270
                probe2_tl.ori = 90
                # probe2 is left and down from probe1
                probe2_tl.pos = [tl_p1_x - sep + 1, tl_p1_y - sep]
        elif tl_ori == 'radial':
            if tl_jump == 1:  # inward
                probe1_tl.ori = 180
                probe2_tl.ori = 0
                # probe2 is right and down from probe1
                probe2_tl.pos = [tl_p1_x + sep - 1, tl_p1_y - sep]
            elif tl_jump == -1:  # outward
                probe1_tl.ori = 0
                probe2_tl.ori = 180
                # probe2 is left and up from probe1
                probe2_tl.pos = [tl_p1_x - sep + 1, tl_p1_y + sep]

        # in bottom left corner, both x and y decrease (left and down)
        bl_p1_x = dist_from_fix * -1
        bl_p1_y = dist_from_fix * -1
        probe1_bl.pos = [bl_p1_x, bl_p1_y]
        if bl_ori == 'tangent':
            if bl_jump == 1:  # CCW
                probe1_bl.ori = 0
                probe2_bl.ori = 180
                probe2_bl.pos = [bl_p1_x - sep + 1, bl_p1_y + sep]
            elif bl_jump == -1:  # CW
                probe1_bl.ori = 180
                probe2_bl.ori = 0
                probe2_bl.pos = [bl_p1_x + sep - 1, bl_p1_y - sep]
        elif bl_ori == 'radial':
            if bl_jump == 1:  # inward
                probe1_bl.ori = 90
                probe2_bl.ori = 270
                # probe2 is right and up from probe1
                probe2_bl.pos = [bl_p1_x + sep - 1, bl_p1_y + sep]
            elif bl_jump == -1:  # outward
                probe1_bl.ori = 270
                probe2_bl.ori = 90
                # probe2 is left and down from probe1
                probe2_bl.pos = [bl_p1_x - sep + 1, bl_p1_y - sep]

        # in bottom-right corner, x increases (right) and y decreases (down)
        br_p1_x = dist_from_fix * 1
        br_p1_y = dist_from_fix * -1
        probe1_br.pos = [br_p1_x, br_p1_y]
        if br_ori == 'tangent':
            if br_jump == 1:  # CCW
                probe1_br.ori = 270
                probe2_br.ori = 90
                probe2_br.pos = [br_p1_x - sep + 1, br_p1_y - sep]
            elif br_jump == -1:  # CW
                probe1_br.ori = 90
                probe2_br.ori = 270
                probe2_br.pos = [br_p1_x + sep - 1, br_p1_y + sep]
        elif br_ori == 'radial':
            if br_jump == 1:  # inward
                probe1_br.ori = 0
                probe2_br.ori = 180
                # probe2 is left and up from probe1
                probe2_br.pos = [br_p1_x - sep + 1, br_p1_y + sep]
            elif br_jump == -1:  # outward
                probe1_br.ori = 180
                probe2_br.ori = 0
                # probe2 is right and down from probe1
                probe2_br.pos = [br_p1_x + sep - 1, br_p1_y - sep]


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
                    fixation.setRadius(3)
                    fixation.draw()
                    trials_counter.text = f"{trial_number}/{total_n_trials}"
                    trials_counter.draw()

                    # reset timer to start with probe1 presentation.
                    resp.clock.reset()


                # PROBE 1: draw three probes, not one in 'missing_corner'
                if t_probe_1 >= frameN > t_fixation:
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
                    trials_counter.draw()

                # ISI
                if t_ISI >= frameN > t_probe_1:
                    fixation.setRadius(3)
                    fixation.draw()
                    trials_counter.draw()

                # PROBE 2
                if t_probe_2 >= frameN > t_ISI:
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
        thisExp.addData('ISI', ISI)
        thisExp.addData('cond_type', probes_type)
        thisExp.addData('neg_sep', neg_sep)
        thisExp.addData('example_name ', example_name)
        thisExp.addData('probeColor1', probeColor1)
        thisExp.addData('probeColor255', probeColor255)
        thisExp.addData('probeLum', probeLum)
        thisExp.addData('trial_response', resp.corr)
        thisExp.addData('missing_corner', missing_corner)
        thisExp.addData('probe_ecc', probe_ecc)
        thisExp.addData('resp.rt', resp.rt)
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
