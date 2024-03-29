from __future__ import division
from psychopy import gui, visual, core, data, event, logging, clock, monitors
# from psychopy.visual import ShapeStim, EnvelopeGrating, Circle

from os import path, chdir
import numpy as np
# from numpy import deg2rad, array, cos, sin
# from numpy.random import shuffle, choice
# import random
import copy
import time
from datetime import datetime
from math import *
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from kestenSTmaxVal import Staircase

"""
This script is adapted from EXPERIMENT3-backgroundMotion.py, 
(Martin also has a radial version called integration_RiccoBloch_flow_new.py which is in 
r'C:\ Users \sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\Martin_scripts\EXPERIMENTS_INTEGRATION\EXPERIMENTS_SCRIPTS_NEW\old scripts\integration_RiccoBloch_flow_new.py'
or Downloads.  

to make rad_flow_martin.py
I have made several changes.
- change import statements (from psychopy.visual import ShapeStim, EnvelopeGrating, Circle) 
    to just (from psychopy import visual), then use visual.circle, visual.ShapeStim, etc. DONE
- changed import statement from import numpy to import umpy as np, and changed all calls to numpy to np. DONE
- added a method for recording frame intervals.  DONE
- reduced the number of trials per staircase (n_trials_per_stair = 25) to 2 to speed up testing.  DONE
- reduced the number of sep conds to speed up testing to two, and reduced expInfo['stair_list'] to 2.  DONE
- changed screen number to 0 (from 1) to display on the laptop screen.  DONE
- converted the rotation bg_motion to radial, for better comparison with other rad_flow exps.
    - I've moved martin's variables relating to rotation motion to the top of the script,
        and set them to None if 'radial' is selected.  DONE
    - I'me added the wrap_depth_vals function for radial flow_dots depth.  DONE
    - for simplicity I'll keep motion only during ISI and probe2 (rather than add prelim_bg_motion period).  DONE
- added option for radial probe orientation (but kept tangent) NOT DONE YET

for rad_flow_Martin_1_names_no_rot.py
- changed naming convention to match my scripts. (e.g., startPoints to stair_list, thisStart to stair_idx, trialN to step etc) DONE
- removed all rotational motion options - hard coded in radial flow. Done
- However, I have made an option for no bg motion, and space for rings. DONE
- I have also removed red_filter and probe_size(pixels) from script as I no longer need those. DONE

for this script: rad_flow_Martin_2_basic_funcs.py
- changed dot motion from probe1 to end of probe2 (not added prelim yet).  DONE
- changed how probe locations and orientations are generated to use function (if use_pos_ori_func==True, else use original method).  DONE
- changed how dot positions are generated to use function (if use_flow_function==True, else uses original method). DONE
- changed how trial timings are plotted to use function. DONE

"""


# function to get the pixel location and orientation of the probes
def get_probe_pos_dict(separation, target_jump, corner, dist_from_fix,
                       probe_size=1, probes_ori='radial', verbose=False):
    """
    This function gets the pixel positions and orientation of the two probes, given the parameters.

    (loc_x loc_y) is the pixel positions along the meridian line (given by 'corner'),
    at the correct eccentricity (e.g., distance from fixation, given by 'dist_from_fix').
    The centre of the screen is 0, 0, so whether these values are positive or negative
    will depend on the corner the probes are due to appear in.

    The probes should be equally spaced around (loc_x_loc_y) by separation.  e.g., if separation = 4, probe 1 will be
    shifted 2 pixels away from (loc_x_loc_y) in one direction and probe 2 will be
    shifted 2 pixels away from (loc_x_loc_y) in the other direction.
    However, if separation is an odd number, an addition pixel will be added to either probe 1 or probe 2.
    The first step is to calculate this shift for each probe.

    The second step is to get the (loc_x, loc_y) values, which the shift is applied to.
    The default probes_ori is 'radial' meaning both probes appear ON the meridian line.
    A value of 'tangent' means the probes appear either side of the meridian line.
    The probes are rotated according to the corner but should always be facing each other
    (e.g., orientation differs by 180 degrees).

    Finally. probe 2 is always ofset from probe 1 (by probe_size),
    e.g., the 'm' and 'w' shaped probes don't mirror each other, but fit together like a zipper.

    :param separation: spatial separation between probes in pixels.
    :param target_jump: Whether probe 2 is inward (contracting) or outward (expanding) from probe 1.
    :param corner: Which corner the probes are in. 0 = top left, 45 = top right, 90 = bottom right, 135 = bottom left.
                   This will determine whether (loc_x, loc_y) are positive or negative.
    :param dist_from_fix: Distance in pixels from the centre of the screen along the meridian line.
    :param probe_size: Gives the ofset of probe 2, even if the probes have been scaled (to test stimuli).
    :param probes_ori: The relationship between the probes.
                       Default is 'radial', where both probes appear ON the meridian line.
                       'tangent' means the probes appear either side of the meridian.
    :param verbose: Whether to print details to the console.

    :return: A dictionary with the pixel positions and orientation of the two probes,
             along with the (loc_x, loc_y) values for showing the loc_marker guide (during script testing).
    """

    # # First calculate the shift of the probes from the meridian line.
    if separation == 99:  # there is only one probe, which is ON the merrian line
        p1_shift = p2_shift = 0
    elif separation % 2 == 0:  # even number
        p1_shift = p2_shift = (separation * probe_size) // 2
    else:  # odd number: shift by half sep, then either add 1 or 0 extra pixel to the shift.
        extra_shifted_pixel = [0, 1]
        np.random.shuffle(extra_shifted_pixel)
        p1_shift = (separation * probe_size) // 2 + extra_shifted_pixel[0]
        p2_shift = (separation * probe_size) // 2 + extra_shifted_pixel[1]
    if verbose:
        print(f"p1_shift: {p1_shift}, p2_shift: {p2_shift}")

    # # Second, get position and orientation of probes
    probe1_ori = 0
    probe2_ori = 180
    if corner == 45:  # top right
        '''in top-right corner, both x and y increase (right and up)'''
        loc_x = dist_from_fix * 1
        loc_y = dist_from_fix * 1
        '''probes_ori' here refers to the relationship between probes (radial or tangent), 
        whereas probe1_ori refers to rotational angle of probe stimulus'''
        if probes_ori == 'tangent':
            if target_jump == 1:  # CW
                probe1_ori += 180
                probe2_ori += 180
                # probe 2 is right and down from probe 1
                probe1_pos = [loc_x - p1_shift, loc_y + p1_shift]
                probe2_pos = [loc_x + p2_shift - probe_size, loc_y - p2_shift]
            elif target_jump == -1:  # ACW
                probe1_ori += 0
                probe2_ori += 0
                # probe 2 is left and up from probe 1
                probe1_pos = [loc_x + p1_shift, loc_y - p1_shift]
                probe2_pos = [loc_x - p2_shift + probe_size, loc_y + p2_shift]
        elif probes_ori == 'radial':
            if target_jump == 1:  # inward
                probe1_ori += 270
                probe2_ori += 270
                # probe2 is left and down from probe1
                probe1_pos = [loc_x + p1_shift, loc_y + p1_shift]
                probe2_pos = [loc_x - p2_shift + probe_size, loc_y - p2_shift]
            elif target_jump == -1:  # outward
                probe1_ori += 90
                probe2_ori += 90
                # probe2 is right and up from probe1
                probe1_pos = [loc_x - p1_shift, loc_y - p1_shift]
                probe2_pos = [loc_x + p2_shift - probe_size, loc_y + p2_shift]
    elif corner == 135:  # top-left
        loc_x = dist_from_fix * -1
        loc_y = dist_from_fix * 1
        if probes_ori == 'tangent':
            if target_jump == 1:  # ACW
                probe1_ori += 90
                probe2_ori += 90
                # probe2 is left and down from probe1
                probe1_pos = [loc_x - p1_shift, loc_y - p1_shift]
                probe2_pos = [loc_x + p2_shift - probe_size, loc_y + p2_shift]
            elif target_jump == -1:  # CW
                probe1_ori += 270
                probe2_ori += 270
                # probe2 is right and up from probe1
                probe1_pos = [loc_x + p1_shift, loc_y + p1_shift]
                probe2_pos = [loc_x - p2_shift + probe_size, loc_y - p2_shift]
        elif probes_ori == 'radial':
            if target_jump == 1:  # inward
                probe1_ori += 180
                probe2_ori += 180
                # probe2 is right and down from probe1
                probe1_pos = [loc_x - p1_shift, loc_y + p1_shift]
                probe2_pos = [loc_x + p2_shift - probe_size, loc_y - p2_shift]
            elif target_jump == -1:  # outward
                probe1_ori += 0
                probe2_ori += 0
                # probe2 is left and up from probe1
                probe1_pos = [loc_x + p1_shift, loc_y - p1_shift]
                probe2_pos = [loc_x - p2_shift + probe_size, loc_y + p2_shift]
    elif corner == 225:  # bottom-left
        loc_x = dist_from_fix * -1
        loc_y = dist_from_fix * -1
        if probes_ori == 'tangent':
            if target_jump == 1:  # CW
                probe1_ori += 0
                probe2_ori += 0
                # probe2 is left and up from probe1
                probe1_pos = [loc_x + p1_shift, loc_y - p1_shift]
                probe2_pos = [loc_x - p2_shift + probe_size, loc_y + p2_shift]
            elif target_jump == -1:  # ACW
                probe1_ori += 180
                probe2_ori += 180
                # probe2 is right and down from probe1
                probe1_pos = [loc_x - p1_shift, loc_y + p1_shift]
                probe2_pos = [loc_x + p2_shift - probe_size, loc_y - p2_shift]
        elif probes_ori == 'radial':
            if target_jump == 1:  # inward
                probe1_ori += 90
                probe2_ori += 90
                # probe2 is right and up from probe1
                probe1_pos = [loc_x - p1_shift, loc_y - p1_shift]
                probe2_pos = [loc_x + p2_shift - probe_size, loc_y + p2_shift]
            elif target_jump == -1:  # outward
                probe1_ori += 270
                probe2_ori += 270
                # probe2 is left and down from probe1
                probe1_pos = [loc_x + p1_shift, loc_y + p1_shift]
                probe2_pos = [loc_x - p2_shift + probe_size, loc_y - p2_shift]
    else:
        corner = 315  # bottom-right
        loc_x = dist_from_fix * 1
        loc_y = dist_from_fix * -1
        if probes_ori == 'tangent':
            if target_jump == 1:  # ACW
                probe1_ori += 270
                probe2_ori += 270
                # probe2 is right and up from probe1
                probe1_pos = [loc_x + p1_shift, loc_y + p1_shift]
                probe2_pos = [loc_x - p2_shift + probe_size, loc_y - p2_shift]
            elif target_jump == -1:  # CW
                probe1_ori += 90
                probe2_ori += 90
                # probe2 is left and down from probe1
                probe1_pos = [loc_x - p1_shift, loc_y - p1_shift]
                probe2_pos = [loc_x + p2_shift - probe_size, loc_y + p2_shift]
        elif probes_ori == 'radial':
            if target_jump == 1:  # inward
                probe1_ori += 0
                probe2_ori += 0
                # probe2 is left and up from probe1
                probe1_pos = [loc_x + p1_shift, loc_y - p1_shift]
                probe2_pos = [loc_x - p2_shift + probe_size, loc_y + p2_shift]
            elif target_jump == -1:  # outward
                probe1_ori += 180
                probe2_ori += 180
                # probe2 is right and down from probe1
                probe1_pos = [loc_x - p1_shift, loc_y + p1_shift]
                probe2_pos = [loc_x + p2_shift - probe_size, loc_y - p2_shift]

    probe_pos_dict = {'probe1_pos': probe1_pos, 'probe2_pos': probe2_pos,
                      'probe1_ori': probe1_ori, 'probe2_ori': probe2_ori,
                      'corner': corner, 'loc_x': loc_x, 'loc_y': loc_y}

    return probe_pos_dict


def new_dots_depth_and_pos(x_array, y_array, depth_array, dots_speed, flow_dir, min_depth, max_depth):
    """
    This is a function to update flow_dots depth array and get new pixel co-ordinatesusing the original x_array and y_array.

    1a. Update depth_array by adding dots_speed * flow_dir to the current z values.
    1b. adjust any values below dots_min_depth or above dots_max_depth.

    2a. Get new x_pos and y_pos co-ordinates values by dividing x_array and y_array by the new depth_array.
    2b. put the new x_pos and y_pos co-ordinates into an array and transposes it.

    :param x_array: Original x_array positions for the dots (shape = (nDots, 1))
    :param y_array: Original y_array positions for the dots (shape = (nDots, 1))
    :param depth_array: array of depth values for the dots (shape = (nDots, 1))
    :param dots_speed: speed of the dots (float, smaller = slower, larger = faster)
    :param flow_dir: either 1 (contracting/inward/backwards) or -1 (expanding/outward/forwards)
    :param dots_min_depth: default is .5, values below this are adjusted to dots_max_depth
    :param dots_max_depth: default is 5, values above this are adjusted to dots_min_depth
    :return: new dots_pos_array
    """

    # # 1. Update z (depth values) # #
    # Add dots_speed * flow_dir to the current z values.
    updated_dept_arr = depth_array + dots_speed * flow_dir

    # adjust any depth values below min_depth or above max_depth by depth_adj
    depth_adj = max_depth - min_depth
    # adjust updated_dept_arr values less than min_depth by adding depth_adj
    lessthanmin = (updated_dept_arr < min_depth)
    updated_dept_arr[lessthanmin] += depth_adj
    # adjust updated_dept_arr values more than max_depth by subtracting depth_adj
    morethanmax = (updated_dept_arr > max_depth)
    updated_dept_arr[morethanmax] -= depth_adj
    # print(f"updated_dept_arr (clipped):\n{updated_dept_arr}\n")

    # # 2. Get new pixel co-ordinates for dots using original x_array and y_array and updated_dept_arr # #
    x_pos = x_array / updated_dept_arr
    y_pos = y_array / updated_dept_arr

    # puts the new co-ordinates into an array and transposes it, ready to use.
    dots_pos_array = np.array([x_pos, y_pos]).T

    return updated_dept_arr, dots_pos_array


def plt_fr_ints(time_p_trial_nested_list, n_trials_w_dropped_fr,
                expected_fr_dur_ms, allowed_err_ms,
                all_cond_name_list, fr_nums_p_trial, dropped_trial_x_locs,
                mon_name, date, frame_rate, participant, run_num,
                save_path, incomplete=False):
    """
    This function takes in the frame intervals per trial and plots them.  Rather than a single line plot,
    each trial has its own (discontinuous) line (since recording stops between trials), in a distinct colour.
    The colours might make any systematic frame drops easier to spot.
    Trials containing dropped frames are highlighted to make them easy to spot.
    The expected frame rate and bounds of an error are also shown.

    :param time_p_trial_nested_list: a list of lists, where each sublist contains the frame intervals for each trial.
    :param n_trials_w_dropped_fr: int.  How many of the recorded dropped frames included dropped frames.
    :param expected_fr_dur_ms: the expected duration of each frame in ms.
    :param allowed_err_ms: The tolerance for variation in the frame duration in ms.
    :param all_cond_name_list: a list of condition names for each trial (used to colour plots).
    :param fr_nums_p_trial: a nested list of frame numbers for each trial, to use as x_axis.
                Using a nexted list allows me to plot each condition separately.
    :param dropped_trial_x_locs:
    :param mon_name: name of monitor from psychopy monitor centre
    :param date: date of experiment
    :param frame_rate: Frames per second of monitor/experiment.
    :param participant: name of participant
    :param run_num: run number
    :param incomplete: default=False.  Flag as True if the experiment quits early.
    :param save_path: path to save plots to
    """

    total_recorded_trials = len(time_p_trial_nested_list)

    # get unique conditions for selecting colours and plotting legend
    unique_conds = sorted(list(set(all_cond_name_list)))

    '''select colours for lines on plot (up to 20)'''
    # select colour for each condition from tab20, using order shown colours_in_order
    # this is because tab20 has 10 pairs of colours, with similarity between [0, 1], [2, 3], etc.
    colours_in_order = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    if len(unique_conds) > 20:  # just use dark blue for all conds if more than 20
        selected_colours = [0] * len(unique_conds)
    else:
        selected_colours = colours_in_order[:len(unique_conds)]
    my_colours = iter([plt.cm.tab20(i) for i in selected_colours])
    colour_dict = {k: v for (k, v) in zip(unique_conds, my_colours)}

    '''plot frame intervals, one trial at a time'''
    for trial_x_vals, trial_fr_durs, this_cond in zip(fr_nums_p_trial, time_p_trial_nested_list, all_cond_name_list):
        plt.plot(trial_x_vals, trial_fr_durs, color=colour_dict[this_cond])

    # add legend with colours per condition
    if len(all_cond_name_list) < 20:
        legend_handles_list = []
        for cond in unique_conds:
            leg_handle = mlines.Line2D([], [], color=colour_dict[cond], label=cond,
                                       marker='.', linewidth=.5, markersize=4)
            legend_handles_list.append(leg_handle)
        plt.legend(handles=legend_handles_list, fontsize=6, title='conditions', framealpha=.5)

    # add vertical lines between trials, ofset by -.5
    trial_v_lines = [fr_nums_p_trial[i][0] - .5 for i in range(len(fr_nums_p_trial))]
    for trial_line in trial_v_lines:
        plt.axvline(x=trial_line, color='silver', linestyle='dashed', zorder=0)

    # add horizontal lines: green = expected frame duration, red = frame error tolerance
    plt.axhline(y=expected_fr_dur_ms / 1000, color='green', linestyle='dotted', alpha=.5)
    plt.axhline(y=(expected_fr_dur_ms - allowed_err_ms) / 1000, color='red', linestyle='dotted', alpha=.5)
    plt.axhline(y=(expected_fr_dur_ms + allowed_err_ms) / 1000, color='red', linestyle='dotted', alpha=.5)

    # shade trials red that had bad timing
    for loc_pair in dropped_trial_x_locs:
        x0, x1 = loc_pair[0] - .5, loc_pair[1] - .5
        plt.axvspan(x0, x1, color='red', alpha=0.15, zorder=0, linewidth=None)

    plt.title(f"{mon_name}, {frame_rate}Hz, {date}\n{n_trials_w_dropped_fr}/{total_recorded_trials} trials."
              f"dropped fr (expected: {round(expected_fr_dur_ms, 2)}ms, "
              f"allowed_err_ms: +/- {round(allowed_err_ms, 2)})")
    fig_name = f'{participant}_{run_num}_frames.png'
    if incomplete:
        fig_name = f'{participant}_{run_num}_frames_incomplete.png'
    plt.savefig(path.join(save_path, fig_name))
    plt.close()



def wrap_depth_vals(depth_arr, min_depth, max_depth):
    """
    function to take an array (depth_arr) and adjust any values below min_depth
    or above max_depth with +/- (max_depth-min_depth)
    :param depth_arr: np.random.rand(nDots) array giving depth values for radial_flow flow_dots.
    :param min_depth: value to set as minimum depth.
    :param max_depth: value to set as maximum depth.
    :return: updated depth array.
    """
    depth_adj = max_depth - min_depth
    # adjust depth_arr values less than min_depth by adding depth_adj
    lessthanmin = (depth_arr < min_depth)
    depth_arr[lessthanmin] += depth_adj
    # adjust depth_arr values more than max_depth by subtracting depth_adj
    morethanmax = (depth_arr > max_depth)
    depth_arr[morethanmax] -= depth_adj
    return depth_arr


# logging.console.setLevel(logging.DEBUG)
logging.console.setLevel(logging.CRITICAL)
# Ensure that relative paths start from the same directory as this script
_thisDir = path.dirname(path.abspath(__file__))
chdir(_thisDir)
# Store info about the experiment session
# psychopyVersion = 'v2020.2.10'
# expName = 'integration-EXP1'  # from the Builder filename that created this script
expName = path.basename(__file__)[:-3]   # from the Builder filename that created this script

# todo: update ISI to ms not frames.
# todo: add verbose.
expInfo = {'1. Participant': 'Nick_test_07082023',
           '2. Probe duration in frames at 240hz': '2',
           '3. fps': ['60', '240'],
           '4. ISI duration in frame': ['4', '2', '4', '6', '9', '12', '24'],
           '5. Probe orientation': ['tangent', 'radial'],
           '9. bg_motion_dir': ['radial', None],  # 'rotation has been removed
           '9. Background': ['flow_dots', None],  # 'flow_rings', None],


           }

# GUI
dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
if dlg.OK == False:
    core.quit()  # user pressed cancel

expInfo['time'] = datetime.now().strftime("%H:%M:%S")
expInfo['date'] = datetime.now().strftime("%d/%m/%Y")

# GUI SETTINGS
participant_name = expInfo['1. Participant']
run_number = 1  # int(expInfo['2. Run_number'])  # todo: add to dlg
probe_duration = int((expInfo['2. Probe duration in frames at 240hz']))
fps = float(expInfo['3. fps'])
orientation = expInfo['5. Probe orientation']
record_fr_durs = True  # eval(expInfo['8. Record_frame_durs'])  # todo: add to dlg
verbose = True  # todo: add to dlg
monitor_name = 'Nick_work_laptop'  # gamma set at 2.1  # todo: add monitor_name = expInfo['11. monitor_name'] to dlg

# todo: collapse background (flow_dots, flow_rings) or None and bg_motion_dir (rotation, radial, None) into one variable.
background = expInfo['9. Background']
bg_motion_dir = expInfo['9. bg_motion_dir']  # radial or None (rotation has been removed


use_flow_function = True  # e.g.  If True, it uses the new_dots_depth function.
use_pos_ori_function = True  # e.g.  if True, it uses the get_probe_pos_dict function.



# VARIABLES
# todo: add ISI variables.
n_trials_per_stair = 2  # this is the number of trials per stair
probe_ecc = 4  # int((expInfo['6. Probe eccentricity in deg']))1


# Distances between probes
# todo: change separation to only be input once.
# separations = [18, 18, 6, 6, 3, 3, 2, 2, 1, 1, 0, 0]  # 99 values for single probe condition
separations = [18, 18, 6, 6]
# ISI durations, -1 correspond to simultaneous probes
ISI = int((expInfo['4. ISI duration in frame']))

# todo: add conruent/incongruent and stair_names_list, n_stairs and total_n_trials
total_n_trials = 999


# todo: add rings/dot dir to save dir
# save each participant's files into separate dir for each ISI
isi_dir = f'ISI_{ISI}'
save_dir = path.join(_thisDir, expName, participant_name,
                     # background, f'bg{prelim_bg_flow_ms}',
                        f'{participant_name}_{run_number}',
                     isi_dir)
print(f"save_dir: {save_dir}")

# files are labelled as '_incomplete' unless entire script runs.
incomplete_output_filename = f'{participant_name}_{run_number}_incomplete'
save_output_as = path.join(save_dir, incomplete_output_filename)

# Experiment Handler
thisExp = data.ExperimentHandler(name=expName, version='',
                                 extraInfo=expInfo, runtimeInfo=None,
                                 savePickle=None, saveWideText=True,
                                 dataFileName=save_output_as)

# COLORS AND LUMINANCES
# Lum to Color255
LumColor255Factor = 2.39538706913372
# Color255 to Color1
Color255Color1Factor = 1 / 127.5  # Color255 * Color255Color1Factor -1
# Lum to Color1
Color1LumFactor = 2.39538706913372  ###

maxLum = 106  # 255 RGB
minLum = 0.12  # 0 RGB
maxColor255 = 255
minColor255 = 0
maxColor1 = 1
minColor1 = -1
bgLumProp = .2  # .2  # todo: use .45 to match radial_flow_NM_v2.py, or .2 to match exp1
# bgLumProp = .45  # .2  # todo: use .45 to match radial_flow_NM_v2.py, or .2 to match exp1
bgLum = maxLum * bgLumProp
bgColor255 = bgLum * LumColor255Factor
this_bgColour = bgColor255
this_colourSpace = 'rgb255'

# MONITOR SPEC
widthPix = 1920  # todo: change to int(thisMon.getSizePix()[0])
heightPix = 1080  # todo: change to int(thisMon.getSizePix()[1])
mon_width_cm = 59.77  # monitor width in cm  # todo: change this to thisMon.getWidth()
view_dist_cm = 57.3  # viewing distance in cm  # todo: change this to thisMon.getDistance()
view_dist_pix = widthPix / mon_width_cm*view_dist_cm  # used for calculating visual angle (e.g., probe locations at 4dva)
mon = monitors.Monitor(monitor_name, width=mon_width_cm, distance=view_dist_pix)
mon.setSizePix((widthPix, heightPix))
# mon.save()

# WINDOW SPEC
win = visual.Window(monitor=mon, size=(widthPix, heightPix), colorSpace='rgb255', color=bgColor255,
                    units='pix', screen=0, allowGUI=False, fullscr=None)

# ELEMENTS
# fixation bull eye
fixation = visual.Circle(win, radius=2, units='pix', lineColor='white', fillColor='black')


dist_from_fix = round((tan(np.deg2rad(probe_ecc)) * view_dist_pix) / sqrt(2))



# flow_dots
if bg_motion_dir == 'radial':
    nDots = 2000  # 10000

    # todo: probes_ori = 'radial'
    dots_speed = 0.2
    # if monitor_name == 'OLED':
    #     dots_speed = 0.4
    BGspeed = dots_speed
    # todo: do we need to increase the number of flow_dots for OLED?
    # dot_array_width = 10000  # original script used 5000
    # with dot_array_width = widthPix * 3, this gives a values of 5760 on a 1920 monitor,
    # similar to the original setting of 5000.  It also allows the flow_dots to be scaled to the screen for OLED.
    dot_array_width = widthPix * 3  # this scales it for the monitor and keeps more flow_dots on screen

    # todo: most of the flow_dots are off screen using this current dots_min_depth, as the distribution of x_flow has large tails.
    #  Setting it to 1.0 means that the tails are shorted, as dividing x / z only makes values smaller (or the same), not bigger.
    # dots_min_depth = 0.5  # depth values
    dots_min_depth = 1.0
    dots_max_depth = 5  # depth values


    # initial array values
    x = np.random.rand(nDots) * dot_array_width - dot_array_width / 2
    y = np.random.rand(nDots) * dot_array_width - dot_array_width / 2
    z = np.random.rand(nDots) * (dots_max_depth - dots_min_depth) + dots_min_depth
    # print(f"x: {x}, y: {y}, z: {z}")

    x_flow = x / z
    y_flow = y / z

    dots_xys_array = np.array([x_flow, y_flow]).T

    flow_dots = visual.ElementArrayStim(win, elementTex=None,
                                        elementMask='gauss', units='pix', nElements=nDots,
                                        sizes=30, colors=[-0.25, -0.25, -0.25])


else:  # if None (or later if flow_rings)
    print(f"bg_motion_dir: {bg_motion_dir}")
    nDots = 0  # no dots
    dots_speed = None
    BGspeed = dots_speed
    dot_array_width = None  # this scales it for the monitor and keeps more flow_dots on screen
    dots_min_depth = None
    dots_max_depth = None  # depth values





# mask for the 4 areas
mask_size = 150

raisedCosTexture1 = visual.filters.makeMask(256, shape='raisedCosine', fringeWidth=0.3, radius=[1.0, 1.0])
probeMask1 = visual.GratingStim(win=win, mask=raisedCosTexture1, size=(mask_size, mask_size),
                                colorSpace=this_colourSpace, color=this_bgColour,
                                tex=None, units='pix', pos=[dist_from_fix + 1, dist_from_fix + 1])
probeMask2 = visual.GratingStim(win=win, mask=raisedCosTexture1, size=(mask_size, mask_size),
                                colorSpace=this_colourSpace, color=this_bgColour,
                                units='pix', tex=None, pos=[-dist_from_fix - 1, dist_from_fix + 1])
probeMask3 = visual.GratingStim(win=win, mask=raisedCosTexture1, size=(mask_size, mask_size),
                                colorSpace=this_colourSpace, color=this_bgColour,
                                units='pix', tex=None, pos=[-dist_from_fix - 1, -dist_from_fix - 1])
probeMask4 = visual.GratingStim(win=win, mask=raisedCosTexture1, size=(mask_size, mask_size),
                                colorSpace=this_colourSpace, color=this_bgColour,
                                units='pix', tex=None,
                                pos=[dist_from_fix + 1, -dist_from_fix - 1]
                                )

# PROBEs
probeVert = [(0, 0), (1, 0), (1, 1), (2, 1), (2, -1), (1, -1), (1, -2), (-1, -2), (-1, -1), (0, -1)]  # 5 pixels
probe1 = visual.ShapeStim(win, vertices=probeVert, fillColor=(1.0, -1.0, 1.0), lineWidth=0, opacity=1, size=1,
                   interpolate=False)  #
probe2 = visual.ShapeStim(win, vertices=probeVert, fillColor=[-1.0, 1.0, -1.0], lineWidth=0, opacity=1, size=1,
                   interpolate=False)  #

# MOUSE
myMouse = event.Mouse(visible=False)


# empty variable to store recorded frame durations
exp_n_fr_recorded_list = [0]
exp_n_dropped_fr = 0
dropped_fr_trial_counter = 0
dropped_fr_trial_x_locs = []
fr_int_per_trial = []
recorded_fr_counter = 0
fr_counter_per_trial = []
cond_list = []

# delete unneeded variables
expected_fr_sec = 1/fps
expected_fr_ms = expected_fr_sec * 1000
frame_tolerance_prop = .2
max_fr_dur_sec = expected_fr_sec + (expected_fr_sec * frame_tolerance_prop)
# max_fr_dur_ms = max_fr_dur_sec * 1000
win.refreshThreshold = max_fr_dur_sec
frame_tolerance_sec = max_fr_dur_sec - expected_fr_sec
frame_tolerance_ms = frame_tolerance_sec * 1000
min_fr_dur_sec = expected_fr_sec - (expected_fr_sec * frame_tolerance_prop)
max_dropped_fr_trials = 10

too_many_dropped_fr = visual.TextStim(win=win, name='too_many_dropped_fr',
                                      text="The experiment had quit as the computer is dropping frames.\n"
                                           "Sorry for the inconvenience.\n"
                                           "Please contact the experimenter.\n\n"
                                           "Press any key to return to the desktop.",
                                      font='Arial', height=20,
                                      # colorSpace=this_colourSpace
                                      )

# ------------------------------------------------------------------- INSTRUCTION
# ------------------------------------------------------------------- INSTRUCTION
instructions = visual.TextStim(win=win, name='instructions',
                               text="[q] or [4] top-left\n [w] or [5] top-right\n [a] or [1] bottom-left\n [s] or [2] bottom-right \n\n redo the previous trial \n\n[Space bar] to start",
                               font='Arial', pos=[0, 0], height=20, ori=0, color=[255, 255, 255],
                               colorSpace='rgb255', opacity=1, languageStyle='LTR', depth=0.0);

while not event.getKeys():
    instructions.draw()
    win.flip()
# ------------------------------------------------------------------- STAIRCASE
# ------------------------------------------------------------------- STAIRCASE
trial_number = 0
'''the line below is wrong, originally there were 12 staircases, labelled from 1 to 12 (e.g., stops before 13)'''
# expInfo['stair_list'] = list(range(1, 13))  # 14 stairtcases (14 conditions)
expInfo['stair_list'] = list(range(1, len(separations)+1))
expInfo['n_trials_per_stair'] = n_trials_per_stair

stairStart = maxLum
miniVal = bgLum
maxiVal = maxLum

stairs = []
for stair_idx in expInfo['stair_list']:
    thisInfo = copy.copy(expInfo)
    thisInfo['stair_idx'] = stair_idx

    thisStair = Staircase(name='trials',  # todo: change to stair_names_list[stair_idx],
                          type='simple',
                          value=stairStart,
                          C=stairStart * 0.6,  # initial step size, as prop of reference stim
                          minRevs=3,
                          minTrials=n_trials_per_stair,
                          minVal=miniVal,
                          maxVal=maxiVal,
                          targetThresh=0.75,  # changed this from prev versions
                          extraInfo=thisInfo
                          )
    stairs.append(thisStair)

# ------------------------------------------------------------------- EXPERIMENT
# ------------------------------------------------------------------- EXPERIMENT
for step in range(expInfo['n_trials_per_stair']):
    np.random.shuffle(stairs)
    for thisStair in stairs:
        
        # todo: add repeat = True, while repeat: here incase trial is dropped.

        # conditions
        trial_number += 1
        #todo: add in actual_trials_inc_rpt += 1
        stair_idx = thisStair.extraInfo['stair_idx']

        
        # todo: get values from sep_vals_list[stair_idx] etc
        sep = separations[thisStair.extraInfo[
                              'stair_idx'] - 1]  # separation experiment #################################################
        target_jump = np.random.choice([1, -1])  # direction in which the probe jumps : CW or CCW
        probeLum = thisStair.next()
        probeColor255 = probeLum * LumColor255Factor
        probeColor1 = (probeColor255 * Color255Color1Factor) - 1


        # Black or White
        probe1.color = [probeColor255, probeColor255, probeColor255]
        probe2.color = [probeColor255, probeColor255, probeColor255]


        # PROBE LOCATION
        corner = np.random.choice([45, 135, 225, 315])
        
        if not use_pos_ori_function:
            # # original version
    
            x_prob = round((tan(np.deg2rad(probe_ecc)) * view_dist_pix) / sqrt(2))
            y_prob = round((tan(np.deg2rad(probe_ecc)) * view_dist_pix) / sqrt(2))
    
            # reset probe ori
            probe1.ori = 0
            probe2.ori = 0
            if corner == 45:
                p1_x = x_prob * 1
                p1_y = y_prob * 1
                if orientation == 'tangent':
                    if target_jump == 1:  # CCW
                        probe1.ori = 0
                        probe2.ori = 180
                        probe2.pos = [p1_x - (sep) + 1, p1_y + (sep)]
                    elif target_jump == -1:  # CW
                        probe1.ori = 180
                        probe2.ori = 0
                        probe2.pos = [p1_x + (sep) - 1, p1_y - (sep)]
                    elif target_jump == 9:
                        probe1.ori = np.random.choice([0, 180])
            elif corner == 135:
                p1_x = x_prob * -1
                p1_y = y_prob * 1
                if orientation == 'tangent':
                    if target_jump == 1:  # CCW
                        probe1.ori = 90
                        probe2.ori = 270
                        probe2.pos = [p1_x + (sep) - 1, p1_y + (sep)]
                    elif target_jump == -1:  # CW
                        probe1.ori = 270
                        probe2.ori = 90
                        probe2.pos = [p1_x - (sep) + 1, p1_y - (sep)]
                    elif target_jump == 9:
                        probe1.ori = np.random.choice([90, 270])
            elif corner == 225:
                p1_x = x_prob * -1
                p1_y = y_prob * -1
                if orientation == 'tangent':
                    if target_jump == 1:  # CCW
                        probe1.ori = 180
                        probe2.ori = 0
                        probe2.pos = [p1_x + (sep) - 1, p1_y - (sep)]
                    elif target_jump == -1:  # CW
                        probe1.ori = 0
                        probe2.ori = 180
                        probe2.pos = [p1_x - (sep) + 1, p1_y + (sep)]
                    elif target_jump == 9:
                        probe1.ori = np.random.choice([0, 180])
            elif corner == 315:
                p1_x = x_prob * 1
                p1_y = y_prob * -1
                if orientation == 'tangent':
                    if target_jump == 1:  # CCW
                        probe1.ori = 270
                        probe2.ori = 90
                        probe2.pos = [p1_x - (sep) + 1, p1_y - (sep)]
                    elif target_jump == -1:  # CW
                        probe1.ori = 90
                        probe2.ori = 270
                        probe2.pos = [p1_x + (sep) - 1, p1_y + (sep)]
                    elif target_jump == 9:
                        probe1.ori = np.random.choice([90, 270])
    
            probe1.pos = [p1_x, p1_y]
            
        else:  # use_pos_ori_function
            # PROBE POSITION (including shift around dist_from_fix)
            probe_pos_dict = get_probe_pos_dict(sep, target_jump, corner,
                                                dist_from_fix, verbose=verbose)

            # loc_marker.setPos([loc_x, loc_y])
            probe1.setPos(probe_pos_dict['probe1_pos'])
            probe1.setOri(probe_pos_dict['probe1_ori'])
            probe2.setPos(probe_pos_dict['probe2_pos'])
            probe2.setOri(probe_pos_dict['probe2_ori'])
            if verbose:
                print(f"loc_marker: {[probe_pos_dict['loc_x'], probe_pos_dict['loc_y']]}, "
                      f"probe1_pos: {probe_pos_dict['probe1_pos']}, "
                      f"probe2_pos: {probe_pos_dict['probe2_pos']}. dff: {dist_from_fix}")



        if bg_motion_dir == 'radial':
            # 1 is contracting / inward / backwards, -1 is expanding / outward / forwards
            flow_dir = np.random.choice([1, -1])
        else:  # if None
            flow_dir = None

        # timimg in frames
        # if ISI >= 0:
        end_fix_fr = 1 * fps
        end_p1_fr = end_fix_fr + probe_duration
        end_ISI_fr = end_p1_fr + ISI
        end_p2_fr = end_ISI_fr + probe_duration
        end_response_fr = end_p2_fr + 10000 * fps  # I presume this means almost unlimited time to respond?

        repeat = True

        while repeat:
            frameN = -1
            continueRoutine = True
            while continueRoutine:
                frameN = frameN + 1

                ######################################################################## ISI YES
                # FIXATION
                if end_fix_fr >= frameN > 0:
                    if bg_motion_dir == 'radial':
                        if not use_flow_function:
                            new_x = x_flow
                            new_y = y_flow
                            flow_dots.xys = np.array([new_x, new_y]).transpose()
                        else:  # e.g., using new function
                            flow_dots.xys = dots_xys_array
                        flow_dots.draw()

                    probeMask1.draw()
                    probeMask2.draw()
                    probeMask3.draw()
                    probeMask4.draw()

                    fixation.setRadius(3)
                    fixation.draw()

                # start recording frames
                if frameN == end_fix_fr:
                    win.recordFrameIntervals = True

                # PROBE 1
                if end_p1_fr >= frameN > end_fix_fr:
                    if bg_motion_dir == 'radial':
                        if not use_flow_function:
                            z = z + dots_speed * flow_dir
                            z = wrap_depth_vals(z, dots_min_depth, dots_max_depth)
                            x_flow = x / z
                            y_flow = y / z
                            new_x, new_y = x_flow, y_flow
                            flow_dots.xys = np.array([new_x, new_y]).transpose()
                        else:  # e.g., using new function
                            # get new depth_vals array (z) and dots_xys_array (x, y)
                            z, dots_xys_array = new_dots_depth_and_pos(x, y, z, dots_speed, flow_dir,
                                                                       dots_min_depth, dots_max_depth)
                            flow_dots.xys = dots_xys_array

                        flow_dots.draw()

                    probeMask1.draw()
                    probeMask2.draw()
                    probeMask3.draw()
                    probeMask4.draw()
                    probe1.draw()
                    # SIMULTANEOUS CONDITION
                    if ISI == -1:
                        if sep <= 18:
                            probe2.draw()

                    fixation.setRadius(3)
                    fixation.draw()

                # ISI
                if end_ISI_fr >= frameN > end_p1_fr:
                    if bg_motion_dir == 'radial':
                        if not use_flow_function:
                            z = z + dots_speed * flow_dir
                            z = wrap_depth_vals(z, dots_min_depth, dots_max_depth)
                            x_flow = x / z
                            y_flow = y / z
                            new_x, new_y = x_flow, y_flow
                            flow_dots.xys = np.array([new_x, new_y]).transpose()
                        else:  # e.g., using new function
                            # get new depth_vals array (z) and dots_xys_array (x, y)
                            z, dots_xys_array = new_dots_depth_and_pos(x, y, z, dots_speed, flow_dir,
                                                                       dots_min_depth, dots_max_depth)
                            flow_dots.xys = dots_xys_array

                        flow_dots.draw()

                    probeMask1.draw()
                    probeMask2.draw()
                    probeMask3.draw()
                    probeMask4.draw()
                    fixation.setRadius(3)
                    fixation.draw()

                    # PROBE 2
                if end_p2_fr >= frameN > end_ISI_fr:
                    if bg_motion_dir == 'radial':
                        if not use_flow_function:
                            z = z + dots_speed * flow_dir
                            z = wrap_depth_vals(z, dots_min_depth, dots_max_depth)
                            x_flow = x / z
                            y_flow = y / z
                            new_x, new_y = x_flow, y_flow
                            flow_dots.xys = np.array([new_x, new_y]).transpose()
                        else:  # e.g., using new function
                            # get new depth_vals array (z) and dots_xys_array (x, y)
                            z, dots_xys_array = new_dots_depth_and_pos(x, y, z, dots_speed, flow_dir,
                                                                       dots_min_depth, dots_max_depth)
                            flow_dots.xys = dots_xys_array

                        flow_dots.draw()

                    probeMask1.draw()
                    probeMask2.draw()
                    probeMask3.draw()
                    probeMask4.draw()
                    if ISI >= 0:
                        if sep <= 18:
                            probe2.draw()
                    fixation.setRadius(3)
                    fixation.draw()

                # stop recording frame intervals
                if frameN == end_p2_fr:
                    win.recordFrameIntervals = False

                # ANSWER
                if frameN > end_p2_fr:
                    if bg_motion_dir == 'radial':
                        flow_dots.draw()
                    probeMask1.draw()
                    probeMask2.draw()
                    probeMask3.draw()
                    probeMask4.draw()
                    fixation.setRadius(2)
                    fixation.draw()

                    # ANSWER
                    resp = event.BuilderKeyResponse()
                    theseKeys = event.getKeys(keyList=['num_5', 'num_4', 'num_1', 'num_2', 'w', 'q', 'a', 's'])
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

                        # sort frame interval times
                        # get trial frameIntervals details
                        trial_fr_intervals = win.frameIntervals
                        n_fr_recorded = len(trial_fr_intervals)

                        # add to empty lists etc.
                        fr_int_per_trial.append(trial_fr_intervals)
                        fr_counter_per_trial.append(list(range(recorded_fr_counter,
                                                               recorded_fr_counter + len(trial_fr_intervals))))
                        recorded_fr_counter += len(trial_fr_intervals)
                        exp_n_fr_recorded_list.append(exp_n_fr_recorded_list[-1] + n_fr_recorded)
                        # cond_list.append(thisStair.name)
                        print(f"stair_idx: {stair_idx}")
                        cond_list.append(stair_idx)

                        # check for dropped frames (or frames that are too short)
                        # if timings are bad, repeat trial
                        # if max(trial_fr_intervals) > max_fr_dur_sec or min(trial_fr_intervals) < min_fr_dur_sec:
                        # todo: I've changed this on 13072023 to see if it reduces timing issues.
                        timing_bad = False
                        if max(trial_fr_intervals) > max_fr_dur_sec:
                            logging.warning(
                                f"\n\toh no! Frame too long! {round(max(trial_fr_intervals), 2)} > {round(max_fr_dur_sec, 2)}: "
                                f"trial: {n_trials_per_stair}, {thisStair.name}")
                            timing_bad = True

                        if min(trial_fr_intervals) < min_fr_dur_sec:
                            logging.warning(
                                f"\n\toh no! Frame too short! {min(trial_fr_intervals)} < {min_fr_dur_sec}, "
                                f": trial: {n_trials_per_stair}, {thisStair.name}")
                            timing_bad = True

                        if timing_bad:  # comment out stuff for repetitions for now.
                            # repeat = True
                            dropped_fr_trial_counter += 1
                            # n_trials_per_stair -= 1
                            # thisStair.trialCount = thisStair.trialCount - 1  # so Kesten doesn't count this trial
                            win.frameIntervals = []
                            # continueRoutine = False
                            trial_x_locs = [exp_n_fr_recorded_list[-2], exp_n_fr_recorded_list[-1]]
                            dropped_fr_trial_x_locs.append(trial_x_locs)
                            # continue

                        # empty frameIntervals cache
                        win.frameIntervals = []

                        # these belong to the end of the answers section
                        repeat = False
                        continueRoutine = False

                # check for quit
                if event.getKeys(keyList=["escape"]):
                    core.quit()

                # If too many trials have had dropped frames, quit experiment
                if dropped_fr_trial_counter > max_dropped_fr_trials:
                    while not event.getKeys():
                        # display end of experiment screen
                        too_many_dropped_fr.draw()
                        win.flip()
                    else:
                        # close and quit once a key is pressed
                        thisExp.close()
                        win.close()
                        core.quit()


                # refresh the screen
                if continueRoutine:
                    win.flip()

        thisExp.addData('trial_number', trial_number)
        thisExp.addData('stair', stair_idx)
        thisExp.addData('probe_jump', target_jump)

        thisExp.addData('trial_response', resp.corr)
        # thisExp.addData('BGspeed', rotSpeed)
        thisExp.addData('corner', corner)
        thisExp.addData('probe_ecc', probe_ecc)
        thisExp.addData('resp.rt', resp.rt)
        # thisExp.addData('total_nTrials', total_nTrials)
        thisExp.addData('orientation', orientation)

        thisExp.addData('stair_name', thisStair)
        thisExp.addData('step', step)
        thisExp.addData('separation', sep)
        # thisExp.addData('sep_deg', sep_deg)
        # thisExp.addData('neg_sep', neg_sep)
        thisExp.addData('ISI', ISI)
        # thisExp.addData('ISI_selected_ms', ISI_selected_ms)
        # thisExp.addData('ISI_actual_ms', ISI_actual_ms)
        # thisExp.addData('isi_dur_fr', isi_dur_fr)
        # thisExp.addData('congruent', congruent)
        thisExp.addData('flow_dir', flow_dir)
        thisExp.addData('probe_jump', target_jump)
        thisExp.addData('corner', corner)
        # thisExp.addData('corner_name', corner_name)
        thisExp.addData('probeLum', probeLum)
        thisExp.addData('probeColor1', probeColor1)
        thisExp.addData('probeColor255', probeColor255)
        thisExp.addData('trial_response', resp.corr)
        thisExp.addData('resp.rt', resp.rt)
        thisExp.addData('probeColor1', probeColor1)
        thisExp.addData('probeColor255', probeColor255)
        thisExp.addData('probe_ecc', probe_ecc)
        thisExp.addData('BGspeed', BGspeed)
        thisExp.addData('orientation', orientation)
        thisExp.addData('bg_motion_dir', bg_motion_dir)
        # thisExp.addData('vary_fixation', vary_fixation)
        thisExp.addData('end_fix_fr', end_fix_fr)
        # thisExp.addData('p1_diff', p1_diff)
        # thisExp.addData('isi_diff', isi_diff)
        # thisExp.addData('p2_diff', p2_diff)
        thisExp.addData('monitor_name', monitor_name)
        thisExp.addData('this_colourSpace', this_colourSpace)
        thisExp.addData('this_bgColour', this_bgColour)
        thisExp.addData('selected_fps', fps)
        # thisExp.addData('actual_fps', actualFrameRate)
        thisExp.addData('frame_tolerance_prop', frame_tolerance_prop)
        thisExp.addData('expName', expName)
        # thisExp.addData('psychopy_version', psychopy_version)
        thisExp.addData('date', expInfo['date'])
        thisExp.addData('time', expInfo['time'])

        thisExp.nextEntry()

        thisStair.newValue(resp.corr)  # so that the staircase adjusts itself


print("\nend of experiment loop, saving data\n")
# now exp is completed, save as '_output' rather than '_incomplete'
thisExp.dataFileName = path.join(save_dir, f'{participant_name}_{run_number}_output')
thisExp.close()


# plot frame intervals
if record_fr_durs:

    print(f"{dropped_fr_trial_counter}/{total_n_trials} trials with bad timing "
          f"(expected: {round(expected_fr_ms, 2)}ms, "
          f"frame_tolerance_ms: +/- {round(frame_tolerance_ms, 2)})")

    plt_fr_ints(time_p_trial_nested_list=fr_int_per_trial, n_trials_w_dropped_fr=dropped_fr_trial_counter,
                expected_fr_dur_ms=expected_fr_ms, allowed_err_ms=frame_tolerance_ms,
                all_cond_name_list=cond_list, fr_nums_p_trial=fr_counter_per_trial,
                dropped_trial_x_locs=dropped_fr_trial_x_locs,
                mon_name=monitor_name, date=expInfo['date'], frame_rate=fps,
                participant=participant_name, run_num=run_number,
                save_path=save_dir, incomplete=False)



print("\nexperiment finished")

