import psychopy
# psychopyVersion = '2021.2.3'  # '2020.2.10'=MartinExp1Version', 2021.2.3'=nickMac version, latest nickmac version=2022.2.4
# psychopy.useVersion(psychopyVersion)
from psychopy import __version__ as psychopyVersion  # uses the computer's downloaded version

print(f"psychopy.version: {psychopyVersion}")

from psychopy import gui, visual, core, data, event, monitors  # logging, info
from psychopy.hardware import keyboard
import os
import numpy as np
from numpy.random import shuffle
import random
import copy
from datetime import datetime
from math import tan, sqrt
from kestenSTmaxVal import Staircase



'''
Updated Exp1 script.  

make sure probes and trials counter are ontop of blanded mask.
Definately hangs after if trials counter and debugger are both true and console logging is left to deafult.  

Fixed timing issues for concurrent probes: I've updated isi_dur_fr and pr2_fr.
I've changed the responses to use the keyboard as suggested for coder experiments, rather than imported builder code.    
Hopefully reduced memory drain issues too: removed thisExp.close, put logginf controls back in.
If memory issues stl not fixed, I could try to:
1. set auto_log=False in the experiment handler: doesn't help
2. get rid of trials_counter (or use TextBox2 rather than text_stim?): doesn't help
3. set theseKeys once earlier before frame loop.: can't do this
4. check for take a break once earlier before per-frame loop.
5. manually write to csv each trial myself: commenting out all addData calls didn't help either!
#  rather than using addData.
#  see https://discourse.psychopy.org/t/crashes-after-30-mins-wavs-have-clicks-cant-use-midi-files/4311/10
6. try turning off pyglet or using glfw
'''

# todo: Exp1_fr_test doesn't use this
clock = core.Clock()  # a clock to check times from

# todo: Exp1_fr_test doesn't use this
# print(f"psychopy.version: {psychopy.__version__}")


# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# Monitor config from monitor centre
monitor_name = 'NickMac'  # 'NickMac' 'asus_cal' 'Asus_VG24' 'HP_24uh' 'ASUS_2_13_240Hz' 'Iiyama_2_18' 'Nick_work_laptop'


# Store info about the experiment session
expName = 'Exp1_Dec22'  # from the Builder filename that created this script

# todo: add monitor to drop down menu?
expInfo = {'1. Participant': 'nick_fr_test_20230103',
           '2. Run_number': '1',
           '3. Probe duration in frames at 240hz': [2, 50, 100],
           '4. fps': [60, 144, 240],
           '5. Probe_orientation': ['radial', 'tangent'],
           '6. Trial_counter': [False, True],
           '7. Vary_fixation': [False, True],

           # todo: Exp1_fr_test doesn't use this
           '8. Blend_off_edges': [False, True],
           '9. testing/de-bugging': [False, True],
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
n_trials_per_stair = 3
probe_duration = int(expInfo['3. Probe duration in frames at 240hz'])
probe_ecc = 4
fps = int(expInfo['4. fps'])
orientation = expInfo['5. Probe_orientation']
use_trials_counter = eval(expInfo['6. Trial_counter'])
vary_fixation = eval(expInfo['7. Vary_fixation'])
blend_off_edges = eval(expInfo['8. Blend_off_edges'])
verbose = eval(expInfo['9. testing/de-bugging'])

# expected frame duration
expected_fr_ms = (1/fps) * 1000

# todo: Exp1_fr_test doesn't use this
# # LOGGING AND PRINTING TO SCREEN
# # # todo: try with critical logging only
# # sets psychoPy to only log critical messages
# if verbose:
#     logging.console.setLevel(logging.DEBUG)
# else:
#     logging.console.setLevel(logging.CRITICAL)


# VARIABLES
'''Distances between probes (spatially and temporally)
For 1probe condition, use separation==99.
For concurrent probes, use ISI==-1.
'''
separations = [0, 6]  # select from [0, 1, 2, 3, 6, 18, 99]
# separations = [18]  # select from [0, 1, 2, 3, 6, 18, 99]
print(f'\nseparations: {separations}')

# todo: add in code to find equivallent isi_dur_frames for different fps from double_dist?
# ISI_values = [-1, 0, 1, 2]  # select from [-1, 0, 2, 4, 6, 9, 12, 24]
ISI_values = [-1, 6]  # , 0, 1, 2]  # select from [-1, 0, 2, 4, 6, 9, 12, 24]
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
           f'{participant_name}_{run_number}_output'
# files are labelled as '_incomplete' unless entire script runs.
save_output_name = filename + '_incomplete'
print(f'filename: {filename}')


# Experiment Handler
thisExp = data.ExperimentHandler(name=expName,
                                 version=psychopyVersion,  # does not set anything, just saved as string for record-keeping.
                                 extraInfo=expInfo,
                                 savePickle=True,  # todo: Exp1_fr_test doesn't use this
                                 saveWideText=True,
                                 dataFileName=save_output_name,
                                 )

# COLORS AND LUMINANCE
# Lum to Color255
LumColor255Factor = 2.39538706913372
# Color255 to Color1
Color255Color1Factor = 1 / 127.5  # Color255 * Color255Color1Factor -1
# Lum to Color1
Color1LumFactor = 2.39538706913372

maxLum = 106  # 255 RGB
# minLum = 0.12  # 0 RGB
# maxColor255 = 255
# minColor255 = 0
# maxColor1 = 1
# minColor1 = -1
bgLumProp = .2
bgLum = maxLum * bgLumProp
bgColor255 = bgLum * LumColor255Factor
bgColor1 = (bgColor255 * Color255Color1Factor) - 1

# COLOUR SPACE
# todo: Exp1_fr_test doesn't use this
# todo: set to run with RGB, but still convert to rgb255 at output.
colour_space = 'rgb255'
background_col = bgColor255


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
viewdistPix = widthPix / monitorwidth * viewdist
mon = monitors.Monitor(monitor_name, width=monitorwidth, distance=viewdist)
mon.setSizePix((widthPix, heightPix))



# WINDOW SPEC
win = visual.Window(monitor=mon, size=(widthPix, heightPix),
                    colorSpace=colour_space, color=bgColor255,
                    # winType='pyglet',  # I've added this to make it work on pycharm/mac
                    winType='glfw',  # todo: Exp1_fr_test doesn't use this
                    pos=[1, -1],  # pos gives position of top-left of screen
                    units='pix',
                    screen=display_number,
                    allowGUI=False,
                    fullscr=use_full_screen)
print(f"win.colorSpace: {win.colorSpace}")
print(f"win.winType: {win.winType}")

# todo: check this - is it just for apple retina screen
if monitor_name == 'NickMac':
    widthPix = widthPix / 2
    heightPix = heightPix / 2
    widthPix, heightPix = win.size
# if verbose:
print(f"check win.size: {win.size}")
print(f"widthPix: {widthPix}, hight: {heightPix}")


# todo: Exp1_fr_test doesn't use this
# '''set an ideal frame time and margin of error for dropped frames'''
# fr_sec = 1/fps
# fr_error = 1.5
# max_fr_sec = fr_error * fr_sec
# print(f"expected frames duration is {fr_sec}.  An acceptable error is {fr_error} times this.\n"
#       f"If there are frames longer than {max_fr_sec}, the trial will be re-done.")





# ELEMENTS
# fixation bull eye
fixation = visual.Circle(win, radius=2, units='pix', lineColor='white', fillColor='black', name="fixation")

# PROBEs
probeVert = [(0, 0), (1, 0), (1, 1), (2, 1), (2, -1), (1, -1),
             (1, -2), (-1, -2), (-1, -1), (0, -1)]

probe1 = visual.ShapeStim(win, vertices=probeVert, fillColor=[0, 0, 0], name='probe1',
                          lineWidth=0, opacity=1, size=1, interpolate=False)
probe2 = visual.ShapeStim(win, vertices=probeVert, fillColor=[0, 0, 0], name='probe2',
                          lineWidth=0, opacity=1, size=1, interpolate=False)
# todo: Exp1_fr_test doesn't use this
probe1.colorSpace = 'rgb255'
probe2.colorSpace = 'rgb255'
print(f"probe1.colorSpace: {probe1.colorSpace}, probe2.colorSpace: {probe2.colorSpace}")

# probe1 = visual.ShapeStim(win, vertices=probeVert, fillColor=[1.0, -1.0, 1.0],
#                           lineWidth=0, opacity=1, size=1, interpolate=False)
# probe2 = visual.ShapeStim(win, vertices=probeVert, fillColor=[-1.0, 1.0, -1.0],
#                           lineWidth=0, opacity=1, size=1, interpolate=False)

# dist_from_fix is a constant to get 4dva distance from fixation,
dist_from_fix = round((tan(np.deg2rad(probe_ecc)) * viewdistPix) / sqrt(2))



# # HARDWARE
# MOUSE - hide cursor
myMouse = event.Mouse(visible=False)

# # KEYBOARD
# todo: changed this from builder to keyboard
# todo: Exp1_fr_test doesn't use this (it uses event.BuilderKeyResponse())
kb = keyboard.Keyboard()



# TEXT TO DISPLAY (changed from textStim to TextBox2)
# todo: get rid of tabs
# INSTRUCTIONS
insturction_text = "\n\n\n\n\n\nFocus on the fixation circle at the centre of the screen.\n\n" \
                   "A small white target will briefly appear on screen,\n" \
                   "press the key related to the location of the probe:\n\n" \
                   "[4]/[Q] top-left            [5]/[W] top-right\n\n\n\n" \
                   "[1]/[A] bottom-left            [2]/[S] bottom-right.\n\n\n" \
                   "Some targets will be easier to see than others,\n" \
                   "Some will be so dim that you won't see them, so just guess!\n\n" \
                   "You don't need to think for long, respond quickly, " \
                   "but try to push press the correct key!\n\n" \
                   "Don't let your eyes wander, keep focussed on the circle in the middle throughout."
# todo: Exp1_fr_test doesn't use this - it uses TextStim
instructions = visual.TextBox2(win=win, name='instructions', text=insturction_text,
                               # font='Arial',
                               font='Open Sans',
                               letterHeight=20, color='white',
                               alignment='center',  anchor='center',
                               size=[None, None])

# Trial counter
# todo: put trials counter back to .45 of widthPix and heightPix pos
# todo: Exp1_fr_test doesn't use this - it uses TextStim
trials_counter = visual.TextBox2(win=win, name='trials_counter', text="???",
                                 # font='Arial',
                                 font='Open Sans',
                                 letterHeight=20,
                                 # default set to background colour (e.g., invisible)
                                 color=bgColor255,
                                 pos=[-widthPix * .45, -heightPix * .45])
                                 # pos=[-widthPix * .20, -heightPix * .20])
if use_trials_counter:
    # if trials counter yes, change colour to white.
    trials_counter.color = 'white'

# BREAKS
take_break = 76
total_n_trials = int(n_trials_per_stair * n_stairs)
if verbose:
    print(f"take_break every {take_break} trials.")
breaks_text = "Break\nTurn on the light and take at least 30-seconds break.\n" \
              "Keep focussed on the fixation circle in the middle of the screen.\n" \
              "Remember, if you don't see the target, just guess!"
# todo: Exp1_fr_test doesn't use this - it uses TextStim
breaks = visual.TextBox2(win=win, name='breaks', text=breaks_text,
                         # font='Arial',
                         font='Open Sans',
                         color='white',
                         alignment='center', anchor='center',
                         # pos=[0, 0],
                         # letterHeight=20, ori=0, color=[255, 255, 255],
                         # colorSpace='rgb255', opacity=1, languageStyle='LTR', depth=0.0
                         )

# END OF EXPERIMENT MESSAGE
end_of_exp_text = "You have completed this experiment.\n" \
                  "Thank you for your time.\n\n" \
                  "Press any key to return to the desktop."
# todo: Exp1_fr_test doesn't use this - it uses TextStim
end_of_exp = visual.TextBox2(win=win, name='end_of_exp', text=end_of_exp_text,
                             # font='Arial',
                             font='Open Sans',
                             alignment='center', anchor='center',
                             letterHeight=20)

# # todo: add record intervals in dlg
# frame_err_sec = win.refreshThreshold
# print(f"frame_err_sec (120%): {frame_err_sec}")
#
# # create empty variables to use later
# fr_recorded_list = []


# todo: Exp1_fr_test doesn't use this
prev_total_recorded_fr = 0
prev_total_dropped_fr = 0

# SCREEN BEFORE EXPERIMENT
while not kb.getKeys():
    fixation.setRadius(3)
    fixation.draw()
    instructions.draw()
    trials_counter.text = f"0/{total_n_trials}"
    trials_counter.draw()
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
                          C=stairStart * 0.6,  # initial step size, as prop of referene stim
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
        trials_counter.text = f"{trial_number}/{total_n_trials}"
        stair_idx = thisStair.extraInfo['stair_idx']
        print(f"\ntrial_number: {trial_number}, stair_idx: {stair_idx}, thisStair: {thisStair}, step: {step}")

        # condition (Seprataion, ISI)
        sep = sep_vals_list[stair_idx]
        ISI = ISI_vals_list[stair_idx]
        if verbose:
            print(f"ISI: {ISI}, sep: {sep}")

        # Luminance (staircase varies probeLum)
        probeLum = thisStair.next()
        probeColor255 = int(probeLum * LumColor255Factor)  # rgb255 are ints.
        probeColor1 = (probeColor255 * Color255Color1Factor) - 1
        # todo: Exp1_fr_test doesn't use this - it uses probe1.color
        probe1.setColor([probeColor255, probeColor255, probeColor255], colour_space)
        probe2.setColor([probeColor255, probeColor255, probeColor255], colour_space)
        if verbose:
            print(f"probeLum: {probeLum}, probeColor255: {probeColor255}, probeColor1: {probeColor1}")
            print(f"probe1.colorSpace: {probe1.colorSpace}, probe2.colorSpace: {probe2.colorSpace}")

        # PROBE LOCATION
        # # corners go ACW(!) 45=top-right, 135=top-left, 225=bottom-left, 315=bottom-right
        # todo: change to use tuple or names tuple?
        corner = random.choice([45, 135, 225, 315])
        corner_name = 'top_right'
        if corner == 135:
            corner_name = 'top_left'
        elif corner == 225:
            corner_name = 'bottom_left'
        elif corner == 315:
            corner_name = 'bottom_right'

        # # direction in which the probe jumps : CW or ACW
        # todo change to use dict or tuples?
        target_jump = random.choice([1, -1])
        if orientation == 'tangent':
            jump_dir = 'clockwise'
            if target_jump == -1:
                jump_dir = 'anticlockwise'
        else:
            jump_dir = 'inward'
            if target_jump == -1:
                jump_dir = 'outward'
        # if verbose:
        print(f"corner: {corner} {corner_name}; jump dir: {target_jump} {jump_dir}")


        # NEW - set orientations to p1=zero and p2=180 (not zero), than add same orientation change to both
        # reset probe ori
        # todo: Exp1_fr_test doesn't use this - it has (0, 0) here and then different values throughout.
        probe1_ori = 0
        probe2_ori = 180
        if corner == 45:
            # in top-right corner, both x and y increase (right and up)
            p1_x = dist_from_fix * 1
            p1_y = dist_from_fix * 1
            #  'orientation' here refers to the relationship between probes,
            #  whereas probe1_ori refers to rotational angle of probe stimulus
            if orientation == 'tangent':
                if target_jump == 1:  # CW
                    probe1_ori += 180
                    probe2_ori += 180
                    probe2_pos = [p1_x + sep - 1, p1_y - sep]
                elif target_jump == -1:  # ACW
                    probe1_ori += 0
                    probe2_ori += 0
                    probe2_pos = [p1_x - sep + 1, p1_y + sep]
            elif orientation == 'radial':
                if target_jump == 1:  # inward
                    probe1_ori += 270
                    probe2_ori += 270
                    # probe2 is left and down from probe1
                    probe2_pos = [p1_x - sep + 1, p1_y - sep]
                elif target_jump == -1:  # outward
                    probe1_ori += 90
                    probe2_ori += 90
                    # probe2 is right and up from probe1
                    probe2_pos = [p1_x + sep - 1, p1_y + sep]
        elif corner == 135:
            p1_x = dist_from_fix * -1
            p1_y = dist_from_fix * 1
            if orientation == 'tangent':
                if target_jump == 1:  # ACW
                    probe1_ori += 90
                    probe2_ori += 90
                    probe2_pos = [p1_x + sep - 1, p1_y + sep]
                elif target_jump == -1:  # CW
                    probe1_ori += 270
                    probe2_ori += 270
                    probe2_pos = [p1_x - sep + 1, p1_y - sep]
            elif orientation == 'radial':
                if target_jump == 1:  # inward
                    probe1_ori += 180
                    probe2_ori += 180
                    # probe2 is right and down from probe1
                    probe2_pos = [p1_x + sep - 1, p1_y - sep]
                elif target_jump == -1:  # outward
                    probe1_ori += 0
                    probe2_ori += 0
                    # probe2 is left and up from probe1
                    probe2_pos = [p1_x - sep + 1, p1_y + sep]
        elif corner == 225:
            p1_x = dist_from_fix * -1
            p1_y = dist_from_fix * -1
            if orientation == 'tangent':
                if target_jump == 1:  # CW
                    probe1_ori += 0
                    probe2_ori += 0
                    probe2_pos = [p1_x - sep + 1, p1_y + sep]
                elif target_jump == -1:  # ACW
                    probe1_ori += 180
                    probe2_ori += 180
                    probe2_pos = [p1_x + sep - 1, p1_y - sep]
            elif orientation == 'radial':
                if target_jump == 1:  # inward
                    probe1_ori += 90
                    probe2_ori += 90
                    # probe2 is right and up from probe1
                    probe2_pos = [p1_x + sep - 1, p1_y + sep]
                elif target_jump == -1:  # outward
                    probe1_ori += 270
                    probe2_ori += 270
                    # probe2 is left and down from probe1
                    probe2_pos = [p1_x - sep + 1, p1_y - sep]
        else:
            corner = 315
            p1_x = dist_from_fix * 1
            p1_y = dist_from_fix * -1
            if orientation == 'tangent':
                if target_jump == 1:  # ACW
                    probe1_ori += 270
                    probe2_ori += 270
                    probe2_pos = [p1_x - sep + 1, p1_y - sep]
                elif target_jump == -1:  # CW
                    probe1_ori += 90
                    probe2_ori += 90
                    probe2_pos = [p1_x + sep - 1, p1_y + sep]
            elif orientation == 'radial':
                if target_jump == 1:  # inward
                    probe1_ori += 0
                    probe2_ori += 0
                    # probe2 is left and up from probe1
                    probe2_pos = [p1_x - sep + 1, p1_y + sep]
                elif target_jump == -1:  # outward
                    probe1_ori += 180
                    probe2_ori += 180
                    # probe2 is right and down from probe1
                    probe2_pos = [p1_x + sep - 1, p1_y - sep]

        # todo: Exp1_fr_test doesn't use this - it adjust .ori and .pos as it goes.
        probe1.ori = probe1_ori
        probe2.ori = probe2_ori
        probe1_pos = [p1_x, p1_y]
        probe1.pos = probe1_pos
        probe2.pos = probe2_pos

        if verbose:
            print(f"probe1: {probe1_pos}, probe2_pos: {probe2_pos}. dff: {dist_from_fix}")


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
        p2_fr = probe_duration # todo: Exp1_fr_test doesn't use this
        if ISI < 0:
            isi_dur_fr = p2_fr = 0

        # cumulative timing in frames for each part of a trial
        t_fixation = int(fps / 2) + vary_fix
        t_probe_1 = t_fixation + probe_duration
        t_ISI = t_probe_1 + isi_dur_fr
        t_probe_2 = t_ISI + p2_fr
        t_response = t_probe_2 + 10000 * fps  # ~40 seconds to respond

        if verbose:
            print(f"t_fixation: {t_fixation}\n"
                  f"t_probe_1: {t_probe_1}\n"
                  f"t_ISI: {t_ISI}\n"
                  f"t_probe_2: {t_probe_2}\n"
                  f"t_response: {t_response}\n")

        # repeat the trial if [r] has been pressed
        # todo: keep the per-frame stuff to a minimum to reduce the load.

        # take a break every ? trials
        if (trial_number % take_break == 1) & (trial_number > 1):
            continueRoutine = False
            breaks.text = breaks_text + f"\nyou have completed {trial_number}/{total_n_trials} trials."
            breaks.draw()

            # adding this to flush out any logged messages during the breaks.
            # todo: Exp1_fr_test doesn't use this
            # logging.flush()  # write messages out to all targets

            win.flip()

            while not kb.getKeys():
                continueRoutine = True
        else:
            continueRoutine = True


        # todo: add in core.rush(True) for these time critical parts of script.
        # todo: add in global timer and get time from last frame f fixation until first frame of response.
        #  I can later use this to see if there were dropped frames or other time issues across probe1, isi, probe2.


        # loop per frame
        repeat = True
        while repeat:
            frameN = -1

            continueRoutine = True
            while continueRoutine:
                frameN = frameN + 1

                # todo: reset clock once.
                if frameN == t_fixation:
                    # radius is set twice, one here, and once at response time.
                    fixation.setRadius(3)
                    # reset timer to start with probe1 presentation (at last fixation frame).
                    kb.clock.reset()
                    if verbose:
                        print(f"{frameN}: frameN == t_fixation: reset timer")

                    # record time of last fixation frame.
                    # last_fix_fr_time = win.callOnFlip(clock.getTime())

                    win.recordFrameIntervals = True


                # todo: Changed ifs to elifs
                # FIXATION
                if t_fixation >= frameN > 0:
                    # fixation.setRadius(3)
                    # blend_edge_mask.draw()
                    fixation.draw()
                    trials_counter.draw()

                    # if verbose:
                    #     print(f"{frameN}: t_fixation >= frameN > 0: fixation")


                # PROBE 1
                elif t_probe_1 >= frameN > t_fixation:
                    if verbose:
                        print(f"{frameN}: t_probe_1 >= frameN > t_fixation: probe 1")

                    # win.recordFrameIntervals = True
                    # fixation.setRadius(3)
                    # blend_edge_mask.draw()
                    fixation.draw()
                    trials_counter.draw()
                    probe1.draw()
                    if ISI == -1:  # SIMULTANEOUS CONDITION (concurrent)
                        if sep <= 18:  # don't draw 2nd probe in 1probe cond (sep==99)
                            probe2.draw()
                            if verbose:
                                print(f"\t{frameN}: probe2.draw(): conc probes")



                # ISI (only occurs if ISI > 0)
                elif t_ISI >= frameN > t_probe_1:
                    # fixation.setRadius(3)
                    # blend_edge_mask.draw()
                    fixation.draw()
                    trials_counter.draw()
                    if verbose:
                        print(f"{frameN}: t_ISI >= frameN > t_probe_1: ISI")

                # PROBE 2 (Only occurs if ISI > -1, e.g., not concurrent probes)
                elif t_probe_2 >= frameN > t_ISI:
                    if verbose:
                        print(f"{frameN}: t_probe_2 >= frameN > t_ISI: probe 2")

                    # todo: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    # todo: shouldn't this be if isi_dur_fr!!!
                    if ISI >= 0:
                        if sep <= 18:  # don't draw 2nd probe in 1probe cond (sep==99)
                            probe2.draw()
                            if verbose:
                                print(f"\t{frameN}: probe2.draw()")
                    # fixation.setRadius(3)
                    # blend_edge_mask.draw()
                    fixation.draw()
                    trials_counter.draw()

                # elif frameN == t_probe_2:
                #     # record time of last probe frame.
                #     last_probe_fr_time = win.callOnFlip(clock.getTime())


                # Response time
                elif frameN > t_probe_2:
                    # print(f"{frameN}: frameN > t_probe_2: response")

                    win.recordFrameIntervals = False
                    total_recorded_fr = len(win.frameIntervals)
                    fr_recorded_list.append(total_recorded_fr)
                    # win.saveFrameIntervals(f"/Users/nickmartin/Library/CloudStorage/OneDrive-CardiffUniversity/PycharmProjects/Cardiff/memory_and_timings/frameIntervals_20221121/run4/FrameIntervals_{trial_number}_ISI{ISI}")

                    # blend_edge_mask.draw()
                    fixation.setRadius(2)
                    fixation.draw()
                    trials_counter.draw()

                    # ANSWER keys
                    # todo: Exp1_fr_test doesn't use this
                    theseKeys = kb.getKeys(keyList=['num_5', 'num_4', 'num_1',
                                                    'num_2', 'w', 'q', 'a', 's'])
                    if len(theseKeys) > 0:  # at least one key was pressed
                        last_key = theseKeys[-1]
                        resp_key = last_key.name
                        resp_rt = last_key.rt
                        if verbose:
                            print(f"theseKeys: {list([i for i in theseKeys])}")
                            print(f"resp_key: {resp_key}")
                            print(f"resp_rt: {resp_rt}")
                            print(f"key.duration: {last_key.duration}")


                        # default assume response incorrect unless meets criteria below
                        resp_corr = 0

                        if corner == 45:
                            if (resp_key == 'w') or (resp_key == 'num_5'):
                                resp_corr = 1
                        elif corner == 135:
                            if (resp_key == 'q') or (resp_key == 'num_4'):
                                resp_corr = 1
                        elif corner == 225:
                            if (resp_key == 'a') or (resp_key == 'num_1'):
                                resp_corr = 1
                        elif corner == 315:
                            if (resp_key == 's') or (resp_key == 'num_2'):
                                resp_corr = 1

                        repeat = False
                        continueRoutine = False

                # regardless of frameN, check for quit
                if kb.getKeys(keyList=["escape"]):
                    thisExp.close()
                    core.quit()

                # redo the trial if I think I made a mistake
                if kb.getKeys(keyList=["r"]) or kb.getKeys(keyList=['num_9']):
                    repeat = True
                    continueRoutine = False
                    continue

                # # todo: sort out how to allow participants to respond, then go back and repeat the frame.
                # #  At the moment it can get stuck in a loop!)
                # if len(win.frameIntervals) > 0:
                #     if max(win.frameIntervals) > max_fr_sec:
                #         print('Repeating this trial as dropped frame.')
                #         print(win.frameIntervals)
                #         win.frameIntervals.clear()
                #         repeat = True
                #         continueRoutine = False
                #         continue


                # gets rid of double presses
                kb.getKeys(clear=True)

                # refresh the screen
                if continueRoutine:
                    win.flip()

        # get duration from last fixation frame to last probe frame
        # probes_fr_dur = last_probe_fr_time - last_fix_fr_time

        # print(win.frameIntervals)
        #
        # # keep this, so it only checks last set of frames.
        # win.frameIntervals.clear()
        # total_recorded_fr = len(win.frameIntervals)
        # total_dropped_fr = win.nDroppedFrames

        # todo: Exp1_fr_test doesn't use this
        # print(f"{total_dropped_fr}/{total_recorded_fr} dropped frames in total")
        # this_trial_recorded_fr = total_recorded_fr - prev_total_recorded_fr
        # this_trial_dropped_fr = total_dropped_fr - prev_total_dropped_fr
        # print(f"{this_trial_dropped_fr}/{this_trial_recorded_fr} dropped frames on this trial")
        # prev_total_recorded_fr = total_recorded_fr
        # prev_total_dropped_fr = total_dropped_fr
        # fr_recorded_list.append(total_recorded_fr)

        # TrialHandler adds info to CSV (but stored in memory until end?)
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
        thisExp.addData('trial_response', resp_corr)
        thisExp.addData('corner', corner)
        thisExp.addData('corner_name', corner_name)
        thisExp.addData('probe_ecc', probe_ecc)
        # thisExp.addData('resp.rt', resp.rt)
        thisExp.addData('resp_key', resp_key)
        thisExp.addData('resp_rt', resp_rt)
        thisExp.addData('orientation', orientation)
        thisExp.addData('vary_fixation', vary_fixation)
        thisExp.addData('t_fixation', t_fixation)
        # thisExp.addData('last_fix_fr_time', last_fix_fr_time)
        # thisExp.addData('last_probe_fr_time', last_probe_fr_time)
        # thisExp.addData('probes_fr_dur', probes_fr_dur)
        thisExp.addData('monitor_name', monitor_name)
        thisExp.addData('selected_fps', fps)
        thisExp.addData('expName', expName)
        thisExp.addData('psychopyVersion', psychopyVersion)
        thisExp.addData('date', expInfo['date'])
        thisExp.addData('time', expInfo['time'])



        # indicates that this trial has finished
        thisExp.nextEntry()

        # updates staircase
        thisStair.newValue(resp_corr)   # so that the staircase adjusts itself


print("end of experiment loop, saving data")



thisExp.dataFileName = filename

# print(f"thisExp: {thisExp.getAllEntries()}")

thisExp.close()


n_dropped_fr = win.nDroppedFrames
print(f"n_dropped_fr: {n_dropped_fr}")

import matplotlib.pyplot as plt
# plt.plot(win.frameIntervals)
# plt.show()
total_recorded_fr = len(win.frameIntervals)
total_dropped_fr = win.nDroppedFrames
print(f"{total_dropped_fr}/{total_recorded_fr} dropped in total (expected: {round(expected_fr_ms, 2)}ms, 'dropped' if > {round(frame_err_ms, 2)})")
plt.plot(win.frameIntervals)
plt.title(f"{monitor_name}, {fps}Hz, {expInfo['date']}\n{total_dropped_fr}/{total_recorded_fr} dropped fr (expected: {round(expected_fr_ms, 2)}ms, 'dropped' if > {round(frame_err_ms, 2)})")
plt.vlines(x=fr_recorded_list, ymin=min(win.frameIntervals), ymax=max(win.frameIntervals), colors='silver', linestyles='dashed')
plt.axhline(y=frame_err_sec, color='red', linestyle='dashed')
# plt.savefig(f"{expInfo['participant']}{os.sep}{expInfo['participant']}_{expInfo['run_number']}{os.sep}{expInfo['participant']}_{expInfo['run_number']}_frames.png")
fig_name = filename = f'{_thisDir}{os.sep}' \
                                         f'{expName}{os.sep}' \
                                         f'{participant_name}{os.sep}' \
                                         f'{participant_name}_{run_number}{os.sep}' \
                                         f'{participant_name}_{run_number}_frames.png'
print(f"fig_name: {fig_name}")
plt.savefig(fig_name)

#
# win.saveFrameIntervals(fileName=None, clear=True)
# the stuff below certainly seems to be what's recommended (close window then core quit)

while not kb.getKeys():
    # display end of experiment screen
    end_of_exp.draw()
    win.flip()
else:
    # logging.flush()  # write messages out to all targets
    thisExp.abort()  # or data files will save again on exit

    # close and quit once a key is pressed
    win.close()
    core.quit()
