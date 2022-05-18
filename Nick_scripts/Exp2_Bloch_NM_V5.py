from __future__ import division

import copy
import os
from datetime import datetime
from math import *

import numpy as np
from psychopy import __version__ as psychopy_version
from psychopy import gui, visual, core, data, event, monitors

from PsychoPy_tools import check_correct_monitor, get_pixel_mm_deg_values
from kestenSTmaxVal import Staircase

'''
This script is a follow on from exp1a (but uses radial_flow_NM_v2 as its basis):
Exp2_Bloch_NM - no spatial separation - just temporal - test Bloch.
two probe conditions:
1. 2probe - same stimuli as exp1a but probe 2 in exact same place as probe 1.  
    Probes presented for 2frames, with ISI as in exp1a.
2. 1probe - same as 1probe stimuli from exp1.
    no isi, but probe duration varies to fit with 2probe stimuli. 
'''

# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# Monitor config from monitor centre
monitor_name = 'Asus_VG24'  # 'NickMac' 'asus_cal' 'Asus_VG24' 'HP_24uh' 'ASUS_2_13_240Hz' 'Iiyama_2_18'

# Store info about the experiment session
expName = 'Exp2_Bloch_NM_v5'  # from the Builder filename that created this script

expInfo = {'1_Participant': 'Nick_test',
           '2_Probe_dur_in_frames_at_240hz': [2, 50],
           '3_fps': [240, 144, 60],
           '4_Trials_counter': [True, False]}

# GUI
dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
if not dlg.OK:
    core.quit()  # user pressed cancel

expInfo['date'] = datetime.now().strftime("%d/%m/%Y")
expInfo['time'] = datetime.now().strftime("%H:%M:%S")

# GUI SETTINGS
participant_name = expInfo['1_Participant']
probe_duration = int(expInfo['2_Probe_dur_in_frames_at_240hz'])
fps = int(expInfo['3_fps'])
orientation = 'tangent'  # expInfo['5_Probe_orientation']
probe_size = '5pixels'  # expInfo['6_probe_size']
trials_counter = expInfo['4_Trials_counter']
background = None  # expInfo['8_Background']

# ISI timing in ms and frames
'''ISI can be given (roughly) in ms, for any monitor it will try to match that value in frames.
The script uses on frames so ISI will be used in that format.
The actual ms timing is given for record.
This means that the experiment should have similar ms timings on monitors with different fps
milliseconds: [100, 50, 41.66, 37.5, 33.34, 25, 16.67, 8.33, 0]
frames@240hz: [24,  12,  10,    9,    8,     6,  4,    2,    0]
'''

# -2 is the 1pr condition
# ISI_ms_list = [-2, 0, 8.33, 16.67, 25, 37.5, 50, 100, 200]
ISI_ms_list = [-2, 0, 8.3333334,
               16.6666667,
               25, 37.5, 50, 100]  # , 200]
print(f'ISI_ms_list: {ISI_ms_list}')

isi_fr_list = [-2 if i == -2 else int(i * fps / 1000) for i in ISI_ms_list]
print(f'isi_fr_list: {isi_fr_list}')

# VARIABLES
n_trials_per_stair = 25
probe_ecc = 4
# for future versions probe 2 might be rotated 90
rotate_probe2 = 0

# background motion to start 70ms before probe1 (e.g., 17frames at 240Hz).
prelim_bg_flow_ms = 70
prelim_bg_flow_fr = int(prelim_bg_flow_ms * fps / 1000)

# # I now have three versions of each condition - 2probe, line and area.
cond_types = ['1probe']
print(f'cond_types: {cond_types}')
isi_vals_list = list(np.repeat(ISI_ms_list, len(cond_types)))
print(f'isi_vals_list: {isi_vals_list}')
isi_fr_vals_list = list(np.repeat(isi_fr_list, len(cond_types)))
print(f'isi_fr_vals_list: {isi_fr_vals_list}')

# full stimulus duration, not just isi
dur_fr_list = [i + 4 for i in isi_fr_vals_list]
one_frame_ms = 1000 / fps
dur_ms_list = [i * one_frame_ms for i in dur_fr_list]
print(f'dur_fr_list: {dur_fr_list}')
print(f'dur_ms_list: {dur_ms_list}')

n_stairs = len(isi_vals_list)

# cond type list cycles through conds e.g., ['2probe', 'lines', 'circles', '2probe', 'lines', 'circles'...]
cond_type_list = list(np.tile(cond_types, len(ISI_ms_list)))
print(f'cond_type_list: {cond_type_list}')
# stair_names_list joins sep_vals_list and cond_type_list
# e.g., ['-2_1probe', '-2_2probe', '0_1probe', '0_2probe', '0_1probe', '0_2probe', ...]
stair_names_list = [f'{s}_{c}' for s, c in zip(isi_fr_vals_list, cond_type_list)]
print(f'stair_names_list: {stair_names_list}')

"""
main contrast is whether the background and target motion is in same or opposite directions
congruence_list: 1=congruent/same, -1=incongruent/different
congruence_list = [1, -1]*len(separation_values)"""
flow_dir_list = [1, -1] * len(ISI_ms_list)

# FILENAME
filename = f'{_thisDir}{os.sep}' \
           f'{expName}{os.sep}' \
           f'{participant_name}{os.sep}' \
           f'{participant_name}_output'

# Experiment Handler
thisExp = data.ExperimentHandler(name=expName, version=psychopy_version,
                                 extraInfo=expInfo, runtimeInfo=None,
                                 savePickle=None, saveWideText=True,
                                 dataFileName=filename)

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
bgLumP = 20
bgLum = maxLum * bgLumP / 100  # bgLum is 20% of max lum == 21.2
# NEW using bgColor255 now, not just bgLum.
bgColor255 = bgLum * LumColor255Factor

"""
To relate rgb255 to rad_flow experiments using rgb...
flow_bgcolor = [-0.1, -0.1, -0.1]  # darkgrey
bgcolor = 114.75  # equivalent to rad_flow if used with colorSpace='rgb255'
bgcolor = flow_bgcolor  # equivalent to rad_flow if used with colorSpace='rgb'
"""

print(f"bgLum: {bgLum}, bgColor255: {bgColor255}")

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
display_number = 1  # 0 indexed, 1 for external display
if monitor_name in ['ASUS_2_13_240Hz', 'asus_cal']:
    display_number = 0
use_full_screen = True
if display_number > 0:
    use_full_screen = False
widthPix = mon_dict['size'][0]
heightPix = mon_dict['size'][1]
monitorwidth = mon_dict['width']  # monitor width in cm
viewdist = mon_dict['dist']  # viewing distance in cm
viewdistPix = widthPix / monitorwidth * viewdist
mon = monitors.Monitor(monitor_name, width=monitorwidth, distance=viewdist)
mon.setSizePix((widthPix, heightPix))
mon.save()

# WINDOW SPEC
win = visual.Window(monitor=mon, size=(widthPix, heightPix),
                    colorSpace='rgb255',
                    color=bgColor255,
                    winType='pyglet',  # I've added pyglet to make it work on pycharm/mac
                    pos=[1, -1],  # pos gives position of top-left of screen
                    units='pix',
                    screen=display_number,
                    allowGUI=False,
                    fullscr=use_full_screen)

# # check correct monitor details (fps, size) have been accessed.
actual_fps = win.getActualFrameRate(nIdentical=240, nMaxFrames=240,
                                    nWarmUpFrames=10, threshold=1)
print(f'actual_fps: {actual_fps}')

try:
    check_correct_monitor(monitor_name=monitor_name,
                          actual_size=win.size,
                          actual_fps=actual_fps,
                          verbose=True)
    print('\nsize of a single pixel at 57cm')
    get_pixel_mm_deg_values(monitor_name=monitor_name)
    print('Monitor setting all correct')
except ValueError:
    print("Value error when running check_correct_monitor()")
    # don't save csv, no trials have happened yet
    thisExp.abort()

# CLOCK
trialClock = core.Clock()

# ELEMENTS
# fixation bull eye
fixation = visual.Circle(win, radius=2, units='pix', lineColor='white', fillColor='black')

# PROBEs
# probe sizes choice
#     # default setting is probe_size == '5pixels':
probeVert = [(0, 0), (1, 0), (1, 1), (2, 1), (2, -1), (1, -1),
             (1, -2), (-1, -2), (-1, -1), (0, -1)]

probe1 = visual.ShapeStim(win, vertices=probeVert, fillColor=(1.0, -1.0, 1.0),
                          lineWidth=0, opacity=1, size=1, interpolate=False)
probe2 = visual.ShapeStim(win, vertices=probeVert, fillColor=[-1.0, 1.0, -1.0],
                          lineWidth=0, opacity=1, size=1, interpolate=False)

# MASK BEHIND PROBES
raisedCosTexture1 = visual.filters.makeMask(256, shape='raisedCosine',
                                            fringeWidth=0.3, radius=[1.0, 1.0])
mask_size = 150
probeMask1 = visual.GratingStim(win, mask=raisedCosTexture1, tex=None,
                                size=(mask_size, mask_size), units='pix', color=bgColor255)
probeMask2 = visual.GratingStim(win, mask=raisedCosTexture1, tex=None,
                                size=(mask_size, mask_size), units='pix', color=bgColor255)
probeMask3 = visual.GratingStim(win, mask=raisedCosTexture1, tex=None,
                                size=(mask_size, mask_size), units='pix', color=bgColor255)
probeMask4 = visual.GratingStim(win, mask=raisedCosTexture1, tex=None,
                                size=(mask_size, mask_size), units='pix', color=bgColor255)


"""full screen mask to blend off edges and fade to black
Create a raisedCosine mask array and assign it to a Grating stimulus (grey outside, transparent inside)
this was useful http://www.cogsci.nl/blog/tutorials/211-a-bit-about-patches-textures-and-masks-in-psychopy"""
raisedCosTexture2 = visual.filters.makeMask(1080, shape='raisedCosine', fringeWidth=0.6, radius=[1.0, 1.0])
invRaisedCosTexture = -raisedCosTexture2  # inverts mask to blur edges instead of center
blankslab = np.ones((1080, 420))  # create blank slabs to put to left and right of image
mmask = np.append(blankslab, invRaisedCosTexture, axis=1)  # append blank slab to left
mmask = np.append(mmask, blankslab, axis=1)  # and right
dotsMask = visual.GratingStim(win, mask=mmask, tex=None, contrast=1.0,
                              size=(widthPix, heightPix), units='pix', color='black')

# MOUSE - Hide cursor
myMouse = event.Mouse(visible=False)

# INSTRUCTION
instructions = visual.TextStim(win=win, name='instructions',
                               text="\n\n\nPlease maintain focus on the black cross at the centre of the screen.\n\n"
                                    "A small white probe will briefly flash on screen,\n"
                                    "press the key related to the location of the probe:\n\n"
                                    "[4] top-left\t\t\t[5] top-right\n\n\n\n"
                                    "[1] bottom-left\t\t\t[2] bottom-right.\n\n\n"
                                    "Do not rush, aim to be as accurate as possible,\n"
                                    "but if you did not see the probe, please guess.\n\n"
                                    "If you pressed a wrong key by mistake, you can:\n"
                                    "continue or\n"
                                    "press [r] or [9] to redo the previous trial.\n\n"
                                    "Press any key to start.",
                               font='Arial', height=20,
                               color='white')
while not event.getKeys():
    fixation.setRadius(3)
    fixation.draw()
    instructions.draw()
    win.flip()

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
total_n_trials = int(n_trials_per_stair * n_stairs)
take_break = int(total_n_trials / 2) + 1
print(f"take_break every {take_break} trials.")
breaks = visual.TextStim(win=win, name='breaks',
                         text="turn on the light and take at least 30-seconds break.\n\n"
                              "When you are ready to continue, press any key.",
                         font='Arial', height=20,
                         color='white')

end_of_exp = visual.TextStim(win=win, name='end_of_exp',
                             text="You have completed this experiment.\n"
                                  "Thank you for your time.\n\n"
                                  "Press any key to return to the desktop.",
                             font='Arial', height=20)

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

    """
    stair_name will be pos or neg sep value for congruence (e.g., 18, -18, 6, -6 etc)
    however, change -0 to -.1 to avoid confusion with 0.
    stair_name = separations[stair_idx] * congruence_list[stair_idx]
    if separations[stair_idx] + congruence_list[stair_idx] == -1:
        stair_name = -.1
    """
    stair_name = stair_names_list[stair_idx]

    thisStair = Staircase(name=f'{stair_name}',
                          type='simple',
                          value=stairStart,
                          C=stairStart * 0.6,  # step_size, typically 60% of reference stimulus
                          minRevs=3,
                          minTrials=n_trials_per_stair,
                          minVal=miniVal,
                          maxVal=maxiVal,
                          targetThresh=0.75,
                          extraInfo=thisInfo
                          )
    stairs.append(thisStair)

# EXPERIMENT
trial_number = 0
print('\n*** exp loop*** \n\n')

for step in range(n_trials_per_stair):
    np.random.shuffle(stairs)
    for thisStair in stairs:

        trial_number = trial_number + 1
        trialClock.reset()

        stair_idx = thisStair.extraInfo['stair_idx']
        sep = 0
        cond_type = cond_type_list[stair_idx]

        print(f"thisStair: {thisStair}, stair_idx: {stair_idx}, step: {step}, trial_number: {trial_number}")
        print(f"cond_type: {cond_type}, sep: {sep}")

        # ISI_cond is the condition, ISI_time is for timing
        ISI_cond = isi_vals_list[stair_idx]
        ISI_fr = isi_fr_vals_list[stair_idx]
        dur_fr = dur_fr_list[stair_idx]
        dur_ms = dur_ms_list[stair_idx]
        if ISI_cond == -2:
            ISI_time = 0
            ISI_actual_ms = 0
            ISI_fr = 0
        else:
            ISI_time = ISI_cond
            ISI_actual_ms = (1000 / fps) * ISI_fr

        # # congruence is balanced with separation values
        # congruent = congruence_list[stair_idx]
        flow_dir = flow_dir_list[stair_idx]

        # PROBE
        # target_jump = np.random.choice([1, -1])  # direction in which the probe jumps : CW or CCW
        # don't need target_jump, both probes in same position
        # target_jump = congruent * flow_dir
        target_jump = flow_dir

        # staircase varies probeLum
        probeLum = thisStair.next()
        probeColor255 = probeLum * LumColor255Factor
        probeColor1 = (probeColor255 * Color255Color1Factor) - 1
        probe1.color = [probeColor1, probeColor1, probeColor1]
        probe2.color = [probeColor1, probeColor1, probeColor1]

        # PROBE LOCATIONS
        # corners go CCW(!) 45=top-right, 135=top-left, 225=bottom-left, 315=bottom-right
        corner = np.random.choice([45, 135, 225, 315])

        print(f' ISI_cond: {ISI_cond}, ISI_fr: {ISI_fr}, ISI_time: {ISI_time}')
        print(f'\tcorner: {corner}, flow_dir: {flow_dir}, target_jump: {target_jump}')

        weber_lum = (probeLum - bgLum) / probeLum

        print(f'\t\tprobeLum: {probeLum}, bgLum: {bgLum}, weber_lum: {weber_lum}')
        print(f'\t\t\tprobeColor255: {probeColor255}, probeColor1: {probeColor1}')
        print(f'\t\t\t\twin.colorSpace: {win.colorSpace}, bgColor255: {bgColor255}\n')

        dist_from_fix = round((tan(np.deg2rad(probe_ecc)) * viewdistPix) / sqrt(2))

        # probe mask locations
        probeMask1.setPos([dist_from_fix + 1, dist_from_fix + 1])
        probeMask2.setPos([-dist_from_fix - 1, dist_from_fix + 1])
        probeMask3.setPos([-dist_from_fix - 1, -dist_from_fix - 1])
        probeMask4.setPos([dist_from_fix + 1, -dist_from_fix - 1])

        # set probe ori
        if corner == 45:
            # in top-right corner, both x and y increase (right and up)
            p1_x = dist_from_fix * 1
            p1_y = dist_from_fix * 1
            #  'orientation' here refers to the relationship between probes,
            #  whereas probe1.ori refers to rotational angle of probe stimulus
            # Note: rotate_probe2 = 0
            if orientation == 'tangent':
                if target_jump == 1:  # CCW
                    probe1.ori = 0
                    probe2.ori = probe1.ori
                    probe2.pos = [p1_x, p1_y]
                elif target_jump == -1:  # CW
                    probe1.ori = 180
                    probe2.ori = probe1.ori
                    probe2.pos = [p1_x, p1_y]

        elif corner == 135:
            # in top-left corner, x decreases (left) and y increases (up)
            p1_x = dist_from_fix * -1
            p1_y = dist_from_fix * 1
            if orientation == 'tangent':
                if target_jump == 1:  # CCW
                    probe1.ori = 90
                    probe2.ori = probe1.ori
                    probe2.pos = [p1_x, p1_y]
                elif target_jump == -1:  # CW
                    probe1.ori = 270
                    probe2.ori = probe1.ori
                    probe2.pos = [p1_x, p1_y]

        elif corner == 225:
            # in bottom left corner, both x and y decrease (left and down)
            p1_x = dist_from_fix * -1
            p1_y = dist_from_fix * -1
            if orientation == 'tangent':
                if target_jump == 1:  # CCW
                    probe1.ori = 180
                    probe2.ori = probe1.ori
                    probe2.pos = [p1_x, p1_y]
                elif target_jump == -1:  # CW
                    probe1.ori = 0
                    probe2.ori = probe1.ori
                    probe2.pos = [p1_x, p1_y]

        else:
            corner = 315
            # in bottom-right corner, x increases (right) and y decreases (down)
            p1_x = dist_from_fix * 1
            p1_y = dist_from_fix * -1
            if orientation == 'tangent':
                if target_jump == 1:  # CCW
                    probe1.ori = 270
                    probe2.ori = probe1.ori
                    probe2.pos = [p1_x, p1_y]
                elif target_jump == -1:  # CW
                    probe1.ori = 90
                    probe2.ori = probe1.ori
                    probe2.pos = [p1_x, p1_y]

        probe1.pos = [p1_x, p1_y]

        # timing in frames
        # fixation time is now 70ms shorted than previously.
        t_fixation = 1 * (fps - prelim_bg_flow_fr)
        t_bg_motion = t_fixation + prelim_bg_flow_fr
        t_interval_1 = t_bg_motion + probe_duration
        t_ISI = t_interval_1 + ISI_fr
        t_interval_2 = t_ISI + probe_duration
        # essentially unlimited time to respond
        t_response = t_interval_2 + 10000 * fps

        bg_mot_dur = t_bg_motion - t_fixation
        int_1_dur = t_interval_1 - t_bg_motion
        isi_dur = t_ISI - t_interval_1
        int_2_dur = t_interval_2 - t_ISI
        resp_dur = t_response - t_interval_2

        print(f't_fixation: {t_fixation}fr; {t_fixation*(1/fps)*1000}ms\n'
              f't_bg_motion: {t_bg_motion}fr; {t_bg_motion*(1/fps)*1000}ms\n'
              f'\tbg_mot_dur: {bg_mot_dur}fr; {bg_mot_dur*(1/fps)*1000}ms\n'
              f't_interval_1: {t_interval_1}fr; {t_interval_1*(1/fps)*1000}ms\n'
              f'\tint_1_dur: {int_1_dur}fr; {int_1_dur*(1/fps)*1000}ms\n'
              f't_ISI: {t_ISI}fr; {t_ISI*(1/fps)*1000}ms\n'
              f'\tisi_dur: {isi_dur}fr; {isi_dur*(1/fps)*1000}ms\n'
              f't_interval_2: {t_interval_2}fr; {t_interval_2*(1/fps)*1000}ms\n'
              f'\tint_2_dur: {int_2_dur}fr; {int_2_dur*(1/fps)*1000}ms\n'
              f't_response: {t_response}fr; {t_response*(1/fps)*1000}ms\n'
              f'\tresp_dur: {resp_dur}fr; {resp_dur*(1/fps)*1000}ms\n')

        # repeat the trial if [r] has been pressed
        repeat = True
        while repeat:
            frameN = -1

            # Break after trials 75 and 150, or whatever set in take_break
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
                    # before fixation has finished
                    trials_counter.text = f"{trial_number}/{total_n_trials}"

                    fixation.setRadius(3)
                    fixation.draw()
                    trials_counter.draw()

                # Background motion prior to probe1
                if t_bg_motion >= frameN > t_fixation:
                    # after fixation, before end of background motion
                    fixation.setRadius(3)
                    fixation.draw()
                    trials_counter.draw()

                # PROBE 1
                if t_interval_1 >= frameN > t_bg_motion:
                    # after background motion, before end of probe1 interval
                    fixation.setRadius(3)
                    fixation.draw()
                    probe1.draw()
                    trials_counter.draw()

                # ISI
                if t_ISI >= frameN > t_interval_1:
                    fixation.setRadius(3)
                    fixation.draw()
                    if ISI_cond != -2:
                        # print('not isi-2')
                        if cond_type == '1probe':
                            # print('drawing 1probe throughout isi')
                            probe1.draw()
                    trials_counter.draw()

                # PROBE 2
                if t_interval_2 >= frameN > t_ISI:
                    # after ISI but before end of probe2 interval
                    fixation.setRadius(3)
                    fixation.draw()

                    # don't show 2nd probe in 1pr condition
                    if ISI_cond != -2:
                        # print('drawing probe2')
                        probe2.draw()
                    trials_counter.draw()

                # ANSWER
                if frameN > t_interval_2:
                    # after probe 2 interval
                    fixation.setRadius(2)
                    fixation.draw()
                    trials_counter.draw()

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

                # regardless of frameN
                # check for quit
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
        thisExp.addData('step', step)
        thisExp.addData('cond_type', cond_type)
        thisExp.addData('stair_name', thisStair)
        thisExp.addData('separation', sep)
        thisExp.addData('ISI', ISI_cond)
        thisExp.addData('ISI_time', ISI_time)
        thisExp.addData('probeLum', probeLum)
        thisExp.addData('trial_response', resp.corr)
        thisExp.addData('resp.rt', resp.rt)
        thisExp.addData('probe_jump', target_jump)
        thisExp.addData('corner', corner)
        thisExp.addData('probeColor1', probeColor1)
        thisExp.addData('probeColor255', probeColor255)
        thisExp.addData('probe_ecc', probe_ecc)
        thisExp.addData('orientation', orientation)
        thisExp.addData('ISI_frames', ISI_fr)
        thisExp.addData('ISI_actual_ms', ISI_actual_ms)
        thisExp.addData('dur_fr', dur_fr)
        thisExp.addData('dur_ms', dur_ms)
        thisExp.addData('rotate_probe2', rotate_probe2)
        thisExp.addData('bgLum', bgLum)
        thisExp.addData('bgColor255', bgColor255)
        thisExp.addData('weber_lum', weber_lum)
        thisExp.addData('expName', expName)
        thisExp.addData('monitor_name', monitor_name)
        thisExp.addData('selected_fps', fps)
        thisExp.addData('actual_fps', actual_fps)

        thisExp.nextEntry()

        thisStair.newValue(resp.corr)  # so that the staircase adjusts itself

print("end of exp loop, saving data")
thisExp.close()

while not event.getKeys():
    # display end of experiment screen
    end_of_exp.draw()
    win.flip()
else:
    # close and quit once a key is pressed
    win.close()
    core.quit()
