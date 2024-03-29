from __future__ import division  # do I need this?
from psychopy import gui, visual, core, data, event, monitors
from psychopy import __version__ as psychopy_version
from psychopy.hardware import keyboard
import os
import numpy as np
# from numpy import deg2rad  # just import numpy and use np.deg2rad
from numpy.random import shuffle  # just import numpy and use np.random.shuffle
import random
import copy
from datetime import datetime
from math import tan, sqrt
from kestenSTmaxVal import Staircase


'''
Script to demonstrate Exp1:
ISI of -1 (conc) and 6 frames.
Sep of 0 and 6 pixels.  
'''

# prioritise psychopy
#core.rush(True)


# sets psychoPy to only log critical messages
# logging.console.setLevel(logging.CRITICAL)

# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# Monitor config from monitor centre
monitor_name = 'HP_24uh'  # 'NickMac' 'asus_cal' 'Asus_VG24' 'HP_24uh' 'ASUS_2_13_240Hz' 'Iiyama_2_18' 'Nick_work_laptop'


# Store info about the experiment session
expName = 'Exp1_Jan23_radial'  # from the Builder filename that created this script
expInfo = {'1. Participant': 'rad_fr_test',
           '2. Run_number': '1',
           # '3. ISI duration in frames': [0, 2, 4, 6, 9, 12, 24, -1],
           '3. separation (pixels)': [5, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 36],
           '4. Probe duration in frames at 240hz': [2, 50, 100],
           '5. fps': [240, 60],
           '7. Vary_fixation': [True, False],
           '8. Record_frame_durs': [False, True]
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
this_sep_value = int(expInfo['3. separation (pixels)'])
probe_duration = int(expInfo['4. Probe duration in frames at 240hz'])
fps = int(expInfo['5. fps'])
vary_fixation = eval(expInfo['7. Vary_fixation'])
record_fr_durs = eval(expInfo['8. Record_frame_durs'])

# expected frame duration
expected_fr_ms = (1/fps) * 1000

# VARIABLES
orientation = 'radial'
n_trials_per_stair = 2
probe_ecc = 4

'''Distances between probes (spatially and temporally)
For 1probe condition, use separation==99.
For concurrent probes, use ISI==-1.
'''
# separations = [0, 1, 2, 3, 6, 18, 99]  # select from [0, 1, 2, 3, 6, 18, 99]
separations = [this_sep_value]  # select from [0, 1, 2, 3, 6, 18, 99]
print(f'separations: {separations}')
# ISI_values = [-1, 6]  # select from [-1, 0, 2, 4, 6, 9, 12, 24]
ISI_values = [-1, 0, 2, 4, 6, 9, 12, 24]  # select from [-1, 0, 2, 4, 6, 9, 12, 24]
print(f'ISI_values: {ISI_values}')

probe_dirs = ['inward', 'outward']
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

# FILENAME
filename = f'{_thisDir}{os.sep}' \
           f'{expName}{os.sep}' \
           f'{participant_name}{os.sep}' \
           f'{participant_name}_{run_number}{os.sep}' \
           f'sep_{this_sep_value}{os.sep}' \
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
viewdistPix = widthPix/monitorwidth*viewdist
mon = monitors.Monitor(monitor_name, width=monitorwidth, distance=viewdist)
mon.setSizePix((widthPix, heightPix))
# mon.save()

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
# fixation bull eye
fixation = visual.Circle(win, radius=2, units='pix', lineColor='white', fillColor='black')

# PROBEs
expInfo['6. Probe size'] = '5pixels'  # ignore this, all experiments use 5pixel probes now.
probeVert = [(0, 0), (1, 0), (1, 1), (2, 1), (2, -1), (1, -1),
             (1, -2), (-1, -2), (-1, -1), (0, -1)]

probe1 = visual.ShapeStim(win, vertices=probeVert, fillColor=(1.0, -1.0, 1.0),
                          lineWidth=0, opacity=1, size=1, interpolate=False)
probe2 = visual.ShapeStim(win, vertices=probeVert, fillColor=[-1.0, 1.0, -1.0],
                          lineWidth=0, opacity=1, size=1, interpolate=False)

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
                         # text="turn on the light and take at least 30-seconds break.",
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

# freame error tollerance
frame_err_sec = win.refreshThreshold
frame_err_ms = frame_err_sec * 1000
print(f"frame_err_sec (120%): {frame_err_sec} (or {frame_err_ms}ms)")
fr_recorded_list = []

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
    shuffle(stairs)
    for thisStair in stairs:

        # Trial, stair and step
        trial_number = trial_number + 1
        stair_idx = thisStair.extraInfo['stair_idx']
        print(f"\ntrial_number: {trial_number}, stair_idx: {stair_idx}, thisStair: {thisStair}, step: {step}")

        # condition (Separation, ISI)
        sep = sep_vals_list[stair_idx]
        ISI = ISI_vals_list[stair_idx]
        print(f"ISI: {ISI}, sep: {sep}")

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

        # # direction in which the probe jumps : CW or CCW
        jump_dir = probes_dirs_list[stair_idx]
        target_jump = 1
        if jump_dir == 'outward':
            target_jump = -1

        # target_jump = random.choice([1, -1])
        # if orientation == 'tangent':
        #     jump_dir = 'clockwise'
        #     if target_jump == -1:
        #         jump_dir = 'anticlockwise'
        # else:
        #     jump_dir = 'inward'
        #     if target_jump == -1:
        #         jump_dir = 'outward'
        print(f"corner: {corner} {corner_name}; jump dir: {target_jump} {jump_dir}")


        # todo: update such that probe2_ori resets to 180, then add sem value to both.
        # reset probe ori
        probe1.ori = 0
        probe2.ori = 0
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
                    probe2.pos = [p1_x + sep - 1, p1_y - sep]
                elif target_jump == -1:  # ACW
                    probe1.ori = 0
                    probe2.ori = 180
                    probe2.pos = [p1_x - sep + 1, p1_y + sep]
            elif orientation == 'radial':
                if target_jump == 1:  # inward
                    probe1.ori = 270
                    probe2.ori = 90
                    # probe2 is left and down from probe1
                    probe2.pos = [p1_x - sep + 1, p1_y - sep]
                elif target_jump == -1:  # outward
                    probe1.ori = 90
                    probe2.ori = 270
                    # probe2 is right and up from probe1
                    probe2.pos = [p1_x + sep - 1, p1_y + sep]
        elif corner == 135:
            p1_x = dist_from_fix * -1
            p1_y = dist_from_fix * 1
            if orientation == 'tangent':
                if target_jump == 1:  # CCW
                    probe1.ori = 90
                    probe2.ori = 270
                    probe2.pos = [p1_x + sep - 1, p1_y + sep]
                elif target_jump == -1:  # CW
                    probe1.ori = 270
                    probe2.ori = 90
                    probe2.pos = [p1_x - sep + 1, p1_y - sep]
            elif orientation == 'radial':
                if target_jump == 1:  # inward
                    probe1.ori = 180
                    probe2.ori = 0
                    # probe2 is right and down from probe1
                    probe2.pos = [p1_x + sep - 1, p1_y - sep]
                elif target_jump == -1:  # outward
                    probe1.ori = 0
                    probe2.ori = 180
                    # probe2 is left and up from probe1
                    probe2.pos = [p1_x - sep + 1, p1_y + sep]
        elif corner == 225:
            p1_x = dist_from_fix * -1
            p1_y = dist_from_fix * -1
            if orientation == 'tangent':
                if target_jump == 1:  # CW
                    probe1.ori = 0
                    probe2.ori = 180
                    probe2.pos = [p1_x - sep + 1, p1_y + sep]
                elif target_jump == -1:  # ACW
                    probe1.ori = 180
                    probe2.ori = 0
                    probe2.pos = [p1_x + sep - 1, p1_y - sep]
            elif orientation == 'radial':
                if target_jump == 1:  # inward
                    probe1.ori = 90
                    probe2.ori = 270
                    # probe2 is right and up from probe1
                    probe2.pos = [p1_x + sep - 1, p1_y + sep]
                elif target_jump == -1:  # outward
                    probe1.ori = 270
                    probe2.ori = 90
                    # probe2 is left and down from probe1
                    probe2.pos = [p1_x - sep + 1, p1_y - sep]
        else:
            corner = 315
            p1_x = dist_from_fix * 1
            p1_y = dist_from_fix * -1
            if orientation == 'tangent':
                if target_jump == 1:  # CCW
                    probe1.ori = 270
                    probe2.ori = 90
                    probe2.pos = [p1_x - sep + 1, p1_y - sep]
                elif target_jump == -1:  # CW
                    probe1.ori = 90
                    probe2.ori = 270
                    probe2.pos = [p1_x + sep - 1, p1_y + sep]
            elif orientation == 'radial':
                if target_jump == 1:  # inward
                    probe1.ori = 0
                    probe2.ori = 180
                    # probe2 is left and up from probe1
                    probe2.pos = [p1_x - sep + 1, p1_y + sep]
                elif target_jump == -1:  # outward
                    probe1.ori = 180
                    probe2.ori = 0
                    # probe2 is right and down from probe1
                    probe2.pos = [p1_x + sep - 1, p1_y - sep]

        probe1.pos = [p1_x, p1_y]

        print(f"probe1: {probe1.pos}, probe2.pos: {probe2.pos}. dff: {dist_from_fix}")

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

                    # reset timer to start with probe1 presentation.
                    resp.clock.reset()

                    # start recording frame intervals
                    if record_fr_durs:
                        win.recordFrameIntervals = True

                # PROBE 1
                elif t_probe_1 >= frameN > t_fixation:
                    probe1.draw()

                    if ISI == -1:  # SIMULTANEOUS CONDITION (concurrent)
                        if sep <= 18:  # don't draw 2nd probe in 1probe cond (sep==99)
                            probe2.draw()
                    fixation.setRadius(3)
                    fixation.draw()

                # ISI
                elif t_ISI >= frameN > t_probe_1:
                    fixation.setRadius(3)
                    fixation.draw()

                # PROBE 2
                elif t_probe_2 >= frameN > t_ISI:
                    if ISI >= 0:
                        if sep <= 18:  # don't draw 2nd probe in 1probe cond (sep==99)
                            probe2.draw()
                    fixation.setRadius(3)
                    fixation.draw()

                # ANSWER
                elif frameN > t_probe_2:

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
        thisExp.addData('separation', sep)
        thisExp.addData('ISI', ISI)
        thisExp.addData('isi_dur_fr', isi_dur_fr)
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
                          f'{participant_name}{os.sep}' \
                          f'{participant_name}_{run_number}{os.sep}' \
                          f'sep_{this_sep_value}{os.sep}' \
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
