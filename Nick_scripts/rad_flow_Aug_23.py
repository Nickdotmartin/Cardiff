from __future__ import division
import psychopy
'''
The lab machine uses 2021.2.3, but this doesn't work on my laptop.
ImportError: cannot import name '_vmTesting' from 'psychopy.tests' (unknown location)
 However, I can test on an older version (e.g., 2021.2.2) which does work.
psychopy.useVersion('2021.2.3')'''
psychopy.useVersion('2021.2.2')  # works

from psychopy import gui, visual, core, data, event, monitors

import logging
from os import path, chdir
from copy import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from datetime import datetime
from math import tan, sqrt  # todo: change these to numpy (from numpy import tan, sqrt)
from PsychoPy_tools import get_pixel_mm_deg_values
from kestenSTmaxVal import Staircase

from psychopy import __version__ as psychopy_version
print(f"PsychoPy_version: {psychopy_version}")

'''
Updated version of radial flow experiment.
Has variable fixation time, records frame rate, 
Can have radial or tangential probes.
Can vary the preliminary motion duration.
Has same colour space and background colour as exp1.
Updated wrap_depth_vals (WrapPoints) function.  
'''

# function to get the pixel location and orientation of the probes
def get_probe_pos_dict(separation, target_jump, corner, dist_from_fix,
                       probe_size=1, probes_ori='radial', verbose=False):
    """
    This function gets the pixel positions of the two probes, given the parameters.

    The default probes_ori is 'radial' meaning both probes appear ON the meridian line.
    A value of 'tangent' means the probes appear either side of the meridian line.

    The mid-point between the two probes is give by (loc_x_loc_y).  The probes should be equally
    spaced around (loc_x_loc_y) by separation.  e.g., if separation = 4, probe 1 will be
    shifted 2 pixels away from (loc_x_loc_y) in one direction and probe 2 will be
    shifted 2 pixels away from (loc_x_loc_y) in the other direction.
    However, if separation is an odd number, an addition pixel will be added to either probe 1 or probe 2.
    The first step is to calculate this shift for each probe.

    (loc_x loc_y) is the pixel positions along the meridian line (given by 'corner'),
    at the correct eccentricity (e.g., distance from fixation, given by 'dist_from_fix').
    The centre of the screen is 0, 0, so whether these values are positive or negative
    will depend on the corner the probes are due to appear in.
    The second step is to get the (loc_x, loc_y) values, which the shift is applied to.

    The probes are rotated according to the corner but should always be facing each other
    (e.g., orientation differs by 180 degrees).

    Finally. probe 2 is always ofset by probe_size from probe 1,
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

    # Get position and orientation of probes
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


def get_next_radii(current_radii, rings_speed, min_radius, max_radius, expanding=True):
    """Generates radii for the next frame of the animation.
    Radii near the edge of the screen change by larger values than those in the center.

    Args:
    current_radii: The array of current radii.
    rings_speed: The speed at which the rings expand.
    min_radius: The minimum radius of the rings.
    max_radius: The maximum radius of the rings.
    expanding: Whether the rings are expanding or contracting.

    Returns: an array of updated radii.
    """

    # add (or sub) element*speed to each element, so radii near edge change by larger values than those in center.
    if expanding:
        next_radii = current_radii + np.multiply(current_radii, rings_speed)
    else:
        next_radii = current_radii - np.multiply(current_radii, rings_speed)

    # adjust depth_arr values less than min_depth by adding depth_adj
    lessthanmin = (next_radii < min_radius)
    next_radii[lessthanmin] = max_radius
    # adjust depth_arr values more than max_depth by subtracting depth_adj
    morethanmax = (next_radii > max_radius)
    next_radii[morethanmax] = min_radius

    # print(f"current_radii: {current_radii}, next_radii: {next_radii}\n")

    return next_radii



def plt_fr_ints(fr_int_per_trial, dropped_fr_trial_counter,
                expected_fr_ms, frame_tolerance_ms,
                cond_list, fr_counter_per_trial, dropped_fr_trial_x_locs,
                monitor_name, date, fps, participant_name, run_number,
                save_dir, incomplete=False):

    """
    This function takes in the frame intervals per trial and plots them.  Rather than a single line plot,
    each trial has its own (discontinuous) line (since recording stops between trials), in a distinct colour.
    The colours might make any systematic frame drops easier to spot.
    Trials containing dropped frames are highlighted to make them easy to spot.
    The expected frame rate and bounds of an error are also shown.

    :param fr_int_per_trial: a list of lists, where each sublist contains the frame intervals for each trial.
    :param dropped_fr_trial_counter: int.  How many of the recorded dropped frames included dropped frames.
    :param expected_fr_ms: the expected duration of each frame in ms.
    :param frame_tolerance_ms: The tolerance for variation in the frame duration in ms.
    :param cond_list: a list of condition names for each trial (used to colour plots).
    :param fr_counter_per_trial: a nested list of frame numbers for each trial, to use as x_axis.
                Using a nexted list allows me to plot each condition separately.
    :param dropped_fr_trial_x_locs:
    :param monitor_name: name of monitor from psychopy monitor centre
    :param date: date of experiment
    :param fps: Frames per second of monitor/experiment.
    :param participant_name: name of participant
    :param run_number: run number
    :param save_dir: path to save plots to
    """

    total_recorded_trials = len(fr_int_per_trial)

    # get unique conditions for selecting colours and plotting legend
    unique_conds = sorted(list(set(cond_list)))

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
    for trial_x_vals, trial_fr_durs, this_cond in zip(fr_counter_per_trial, fr_int_per_trial, cond_list):
        plt.plot(trial_x_vals, trial_fr_durs, color=colour_dict[this_cond])

    # add legend with colours per condition
    if len(cond_list) < 20:
        legend_handles_list = []
        for cond in unique_conds:
            leg_handle = mlines.Line2D([], [], color=colour_dict[cond], label=cond,
                                       marker='.', linewidth=.5, markersize=4)
            legend_handles_list.append(leg_handle)
        plt.legend(handles=legend_handles_list, fontsize=6, title='conditions', framealpha=.5)


    # add vertical lines between trials, ofset by -.5
    trial_v_lines = [fr_counter_per_trial[i][0]-.5 for i in range(len(fr_counter_per_trial))]
    for trial_line in trial_v_lines:
        plt.axvline(x=trial_line, color='silver', linestyle='dashed', zorder=0)

    # add horizontal lines: green = expected frame duration, red = frame error tolerance
    plt.axhline(y=expected_fr_ms/1000, color='green', linestyle='dotted', alpha=.5)
    plt.axhline(y=(expected_fr_ms-frame_tolerance_ms)/1000, color='red', linestyle='dotted', alpha=.5)
    plt.axhline(y=(expected_fr_ms+frame_tolerance_ms)/1000, color='red', linestyle='dotted', alpha=.5)

    # shade trials red that had bad timing
    for loc_pair in dropped_fr_trial_x_locs:
        x0, x1 = loc_pair[0] - .5, loc_pair[1] - .5
        plt.axvspan(x0, x1, color='red', alpha=0.15, zorder=0, linewidth=None)

    plt.title(f"{monitor_name}, {fps}Hz, {date}\n{dropped_fr_trial_counter}/{total_recorded_trials} trials."
              f"dropped fr (expected: {round(expected_fr_ms, 2)}ms, "
              f"frame_tolerance_ms: +/- {round(frame_tolerance_ms, 2)})")
    fig_name = f'{participant_name}_{run_number}_frames.png'
    if incomplete:
        fig_name = f'{participant_name}_{run_number}_frames_incomplete.png'
    plt.savefig(path.join(save_dir, fig_name))
    plt.close()



# Ensure that relative paths start from the same directory as this script
_thisDir = path.dirname(path.abspath(__file__))
chdir(_thisDir)

# todo: uses ASUS_2_13_240Hz for replicating old results, but then use asus_cal for testing.

# Store info about the experiment session (numbers keep the order)
# todo: new - automatically get exp name
expName = path.basename(__file__)[:-3]   # from the Builder filename that created this script

expInfo = {'1. Participant': 'Nick_test_04082023',
           '2. Run_number': '1',
           '3. Probe duration in frames': [2, 1, 50, 100],
           '4. fps': [60, 240, 120, 60],
           '5. ISI_dur_in_ms': [25, 16.67, 100, 50, 41.67, 37.5, 33.34, 25, 16.67, 8.33, 0, -1],
           '6. Probe_orientation': ['radial', 'tangent'],
           '7. Vary_fixation': [True, False],
           '8. Record_frame_durs': [True, False],
           '9. Background': ['flow_dots', 'flow_rings', None],
           '10. prelim_bg_flow_ms': [20, 350, 200, 70],
           '11. monitor_name': ['Nick_work_laptop', 'OLED', 'asus_cal', 'Samsung',
                                'Asus_VG24', 'HP_24uh', 'NickMac', 'Iiyama_2_18', 'ASUS_2_13_240Hz'],
           '12. mask_type': ['4_circles', '2_spokes'],
           '13. verbose': [True, False]
           }

# dialogue box
dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
if not dlg.OK:
    core.quit()  # user pressed escape

# dialogue box settings
participant_name = expInfo['1. Participant']
run_number = int(expInfo['2. Run_number'])
probe_duration = int(expInfo['3. Probe duration in frames'])
fps = int(expInfo['4. fps'])
ISI_selected_ms = float(expInfo['5. ISI_dur_in_ms'])
orientation = expInfo['6. Probe_orientation']
vary_fixation = eval(expInfo['7. Vary_fixation'])
record_fr_durs = eval(expInfo['8. Record_frame_durs'])
background = expInfo['9. Background']
prelim_bg_flow_ms = int(expInfo['10. prelim_bg_flow_ms'])
monitor_name = expInfo['11. monitor_name']
mask_type = expInfo['12. mask_type']
verbose = eval(expInfo['13. verbose'])


# misc settings
n_trials_per_stair = 2  # 25
probe_ecc = 4
expInfo['date'] = datetime.now().strftime("%d/%m/%Y")
expInfo['time'] = datetime.now().strftime("%H:%M:%S")


# ISI timing in ms and frames
'''ISI can be given (roughly) in ms, for any monitor it will try to match that value in frames.
The script uses on frames so ISI will be used in that format.
The actual ms timing is given for record.
This means that the experiment should have similar ms timings on monitors with different fps
milliseconds: [100, 50, 41.66, 37.5, 33.34, 25, 16.67, 8.33, 0]
frames@240hz: [24,  12,  10,    9,    8,     6,  4,    2,    0]
'''
if ISI_selected_ms == -1:
    ISI_frames = -1
    ISI_actual_ms = -1
else:
    ISI_frames = int(ISI_selected_ms * fps / 1000)
    ISI_actual_ms = (1/fps) * ISI_frames * 1000
ISI = ISI_frames
if verbose:
    print(f"\nSelected {ISI_selected_ms}ms ISI.\n"
          f"At {fps}Hz this is {ISI_frames} frames which each take {round(1000/fps, 2)} ms.\n")
ISI_list = [ISI_frames]
if verbose:
    print(f'ISI_list: {ISI_list}')


# Separation values in pixels
# separations = [0, 1, 2, 3, 6, 18]
separations = [3]
separations.sort(reverse=True)
if verbose:
    print(f'separations: {separations}')


# # main contrast is whether the background and target motion is in same or opposite direction.
# congruence_vals: 1=congruent/same, -1=incongruent/different
# todo: DO i need to sort congruence to make sure that the staircases are in the same order?
congruence_vals = [1, -1]
congruence_names = ['cong', 'incong']
if verbose:
    print(f'congruence_vals: {congruence_vals}')
    print(f'congruence_names: {congruence_names}')

# lists of values for each condition (all list are same length = n_stairs)
'''each separation value appears in 2 stairs, e.g.,
stair1 will be sep=18, flow_dir=inwards; stair2 will be sep=18, flow_dir=outwards etc.
e.g., sep_vals_list = [18, 18, 6, 6, 3, 3, 2, 2, 1, 1, 0, 0]
this study does not include the two single probe conditions (labeled 99 in previous exp)
'''
ISI_vals_list = list(np.repeat(ISI_list, len(separations))) * len(congruence_vals)
sep_vals_list = list(np.tile(separations, len(ISI_list) * len(congruence_vals)))
cong_vals_list = list(np.repeat(congruence_vals, len(sep_vals_list) / len(congruence_vals)))
cong_names_list = list(np.repeat(congruence_names, len(sep_vals_list) / len(congruence_vals)))
if verbose:
    print(f'ISI_vals_list: {ISI_vals_list}')
    print(f'sep_vals_list: {sep_vals_list}')
    print(f'cong_vals_list: {cong_vals_list}')
    print(f'cong_names_list: {cong_names_list}')


# stair_names_list joins cong_names_list, sep_vals_list and ISI_vals_list
# e.g., ['cong_sep18_ISI6', 'cong_sep6_ISI6', 'incong_sep18_ISI6', 'incong_sep6_ISI6', ]
stair_names_list = [f'{p}_sep{s}_ISI{i}' for p, s, i in zip(cong_names_list, sep_vals_list, ISI_vals_list)]
n_stairs = len(sep_vals_list)
total_n_trials = int(n_trials_per_stair * n_stairs)
if verbose:
    print(f'stair_names_list: {stair_names_list}')
    print(f'n_stairs: {n_stairs}')
    print(f'total_n_trials: {total_n_trials}')


# Experiment handling and saving
# todo: add rings/dot dir to save dir
# save each participant's files into separate dir for each ISI
isi_dir = f'ISI_{ISI}'
save_dir = path.join(_thisDir, expName, participant_name, background, f'bg{prelim_bg_flow_ms}',
                        f'{participant_name}_{run_number}', isi_dir)


# files are labelled as '_incomplete' unless entire script runs.
incomplete_output_filename = f'{participant_name}_{run_number}_incomplete'
save_output_as = path.join(save_dir, incomplete_output_filename)


# Experiment Handler
thisExp = data.ExperimentHandler(name=expName, version=psychopy_version,
                                 extraInfo=expInfo, runtimeInfo=None,
                                 savePickle=True, saveWideText=True,
                                 dataFileName=save_output_as)



# MONITOR details: colour, luminance, pixel size and frame rate
if verbose:
    print(f"\nmonitor_name: {monitor_name}")
thisMon = monitors.Monitor(monitor_name)

'''
rad_flow_NM_v2 used a lighter screen that exp1.  (bg as 45% not 20%)
flow_bgcolor = [-0.1, -0.1, -0.1]  # dark grey converts to:
rgb: -0.1 = rgb1: .45 = rgb255: 114.75 = lum: 47.8.
for future ref, to match exp1 it should be flow_bgcolor = [-0.6, -0.6, -0.6]  # dark grey
'''
# # Lum to Color255 (maxLum = 253)
LumColor255Factor = 2.39538706913372
maxLum = 106  # 255 RGB  # todo: the actual maxLum is ~150 (as measured with spyder on 12062023), not 106
bgLumProp = .45  # .2  # todo: use .45 to match radial_flow_NM_v2.py, or .2 to match exp1
if monitor_name == 'OLED':
    bgLumProp = .0
bgLum = maxLum * bgLumProp
bgColor255 = int(bgLum * LumColor255Factor)
bgColor_rgb1 = bgLum / maxLum
bg_color_rgb = (bgColor_rgb1 * 2) - 1
if verbose:
    print(f'bgLum: {bgLum}, bgColor255: {bgColor255}, bgColor_rgb1: {bgColor_rgb1}, bg_color_rgb: {bg_color_rgb}')

# colour space
this_colourSpace = 'rgb255'  # 'rgb255', 'rgb1'
this_bgColour = [bgColor255, bgColor255, bgColor255]
adj_dots_col = int(255 * .15)
if monitor_name == 'OLED':
    this_colourSpace = 'rgb1'  # values between 0 and 1
    this_bgColour = [bgColor_rgb1, bgColor_rgb1, bgColor_rgb1]
    adj_dots_col = .15
if verbose:
    print(f"this_colourSpace: {this_colourSpace}, this_bgColour: {this_bgColour}")
    print(f"adj_dots_col colours: {adj_dots_col}")

# don't use full screen on external monitor
display_number = 1  # 0 indexed, 1 for external display, 0 for internal
if monitor_name in ['asus_cal', 'Nick_work_laptop', 'NickMac', 'OLED', 'ASUS_2_13_240Hz']:
    display_number = 0
use_full_screen = True
if display_number > 0:
    use_full_screen = False
if verbose:
    print(f"monitor_name: {monitor_name}, display_number: {display_number}, use_full_screen: {use_full_screen}")

widthPix = int(thisMon.getSizePix()[0])
heightPix = int(thisMon.getSizePix()[1])
mon_width_cm = thisMon.getWidth()  # monitor width in cm
view_dist_cm = thisMon.getDistance()  # viewing distance in cm
viewdistPix = widthPix / mon_width_cm*view_dist_cm  # used for calculating visual angle (e.g., probe locations at 4dva)
mon = monitors.Monitor(monitor_name, width=mon_width_cm, distance=view_dist_cm)
mon.setSizePix((widthPix, heightPix))
if verbose:
    print(f"widthPix: {widthPix}, heightPix: {heightPix}, mon_width_cm: {mon_width_cm}, "
          f"view_dist_cm: {view_dist_cm}, viewdistPix: {viewdistPix}")

# WINDOW
# note change of winType 23/05/2023 from pyglet to glfw, might still need pyglet on pycharm/mac though.
win = visual.Window(monitor=mon, size=(widthPix, heightPix),
                    colorSpace=this_colourSpace, color=this_bgColour,
                    winType='pyglet',
                    pos=[1, -1],  # pos gives position of top-left of screen
                    units='pix',
                    screen=display_number,
                    allowGUI=False,
                    fullscr=use_full_screen)
if verbose:
    print(f'winType: {win.winType}')


# pixel size
pixel_mm_deg_dict = get_pixel_mm_deg_values(monitor_name=monitor_name)
if verbose:
    print(f"diagonal pixel size: {pixel_mm_deg_dict['diag_mm']} mm, or {pixel_mm_deg_dict['diag_deg']} dva")


# frame duration (expected and actual)
expected_fr_sec = 1/fps
expected_fr_ms = expected_fr_sec * 1000
if verbose:
    print(f"\nexpected frame duration: {expected_fr_ms} ms (or {round(expected_fr_sec, 5)} seconds).")
actualFrameRate = int(win.getActualFrameRate())
if verbose:
    print(f"actual fps: {win.getActualFrameRate()}")
if abs(fps-actualFrameRate) > 5:
    raise ValueError(f"\nfps ({fps}) does not match actualFrameRate ({actualFrameRate}).")


'''set the max and min frame duration to accept, trials with critical frames beyond these bound will be repeated.'''
# frame error tolerance - default is approx 20% but seems to vary between runs(!), so set it manually.
frame_tolerance_prop = .2
max_fr_dur_sec = expected_fr_sec + (expected_fr_sec * frame_tolerance_prop)
max_fr_dur_ms = max_fr_dur_sec * 1000
win.refreshThreshold = max_fr_dur_sec
frame_tolerance_sec = max_fr_dur_sec - expected_fr_sec
frame_tolerance_ms = frame_tolerance_sec * 1000
frame_tolerance_prop = frame_tolerance_sec / expected_fr_sec
min_fr_dur_sec = expected_fr_sec - (expected_fr_sec * frame_tolerance_prop)
if verbose:
    print(f"\nframe_tolerance_sec: {frame_tolerance_sec} ({frame_tolerance_prop}% of {expected_fr_sec} sec)")
    print(f"max_fr_dur_sec ({100 + (100 * frame_tolerance_prop)}%): {max_fr_dur_sec} (or {max_fr_dur_ms}ms)")
    print(f"min_fr_dur_sec ({100 - (100 * frame_tolerance_prop)}%): {min_fr_dur_sec} (or {min_fr_dur_sec * 1000}ms)")

# quit experiment if there are more than 10 trials with dropped frames
max_dropped_fr_trials = 10



# ELEMENTS
# fixation bull eye
if background is None:
    fixation = visual.Circle(win, radius=2, units='pix',
                             lineColor='white', fillColor='black', colorSpace=this_colourSpace)
else:
    fixation = visual.Circle(win, radius=2, units='pix',
                             lineColor='black', fillColor='grey', colorSpace=this_colourSpace)

# PROBEs
# default is to use 5 pixel probes,but can use 7 on OLED if needed
probe_n_pixels = 5  # 7

probeVert = [(0, 0), (1, 0), (1, 1), (2, 1), (2, -1), (1, -1),
             (1, -2), (-1, -2), (-1, -1), (0, -1)]

probe_size = 1
probe1 = visual.ShapeStim(win, vertices=probeVert, fillColor='red', colorSpace=this_colourSpace,
                          lineWidth=0, opacity=1, size=probe_size, interpolate=False)
probe2 = visual.ShapeStim(win, vertices=probeVert, fillColor='green', colorSpace=this_colourSpace,
                          lineWidth=0, opacity=1, size=probe_size, interpolate=False)


# dist_from_fix is a constant to get 4dva distance from fixation for probes and probe_masks
dist_from_fix = round((tan(np.deg2rad(probe_ecc)) * viewdistPix) / sqrt(2))


# MASK BEHIND PROBES (infront of flow dots to keep probes and motion separate)
'''This is either circles in the four probe locations or a diagonal cross shape'''
mask_size = 150

if mask_type == '4_circles':
    raisedCosTexture1 = visual.filters.makeMask(256, shape='raisedCosine',
                                                fringeWidth=0.3, radius=[1.0, 1.0])
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
                                    units='pix', tex=None, pos=[dist_from_fix + 1, -dist_from_fix - 1])
    probe_mask_list = [probeMask1, probeMask2, probeMask3, probeMask4]

elif mask_type == '2_spokes':
    # draw a large diagonal cross (X) with vertices which reaches the top and bottom of the window

    # since the middle of the screen is 0, 0; the corners are defined by half the width or height of the screen.
    half_hi_pix = int(heightPix / 2)

    # the corners of the cross are offset (by around 42 pixels on my laptop);
    # which is half the mask_size / the screen aspect ratio (pixl shape)
    offset_pix = int((mask_size / 2) / (widthPix / heightPix))
    if verbose:
        print(f'offset_pix = {offset_pix}')

    '''vertices start at the bottom left corner and go clockwise, with three values for each side.  
    The first three values are for the left of the X, the next three for the top
    1. the bl corner of the cross, which is at the bottom of the window, with an offset (e.g., not in the corner of the window).
    2. horizontally centred, but offset to the left of the centre.
    3. the tl corner of the cross, which is at the top of the window, with an offset (e.g., not in the corner of the window).
    4. offset to the right of 3.
    5. vertically centred, but offset above the centre.
    6. the tr corner of the cross, which is at the top of the window, with an offset (e.g., not in the corner of the window).
    '''
    # # original vertices as plain cross X
    # vertices = np.array([[-half_hi_pix - offset_pix, -half_hi_pix], [-offset_pix, 0], [-half_hi_pix - offset_pix, half_hi_pix],
    #                      [-half_hi_pix + offset_pix, half_hi_pix], [0, offset_pix], [half_hi_pix - offset_pix, half_hi_pix],
    #                      [half_hi_pix + offset_pix, half_hi_pix], [offset_pix, 0], [half_hi_pix + offset_pix, -half_hi_pix],
    #                      [half_hi_pix - offset_pix, -half_hi_pix], [0, -offset_pix], [-half_hi_pix + offset_pix, -half_hi_pix]
    #                      ])

    # updated vertices with wedge shape
    vertices = np.array([[-half_hi_pix - offset_pix*2, -half_hi_pix], [-offset_pix/2, 0], [-half_hi_pix - offset_pix*2, half_hi_pix],
                         [-half_hi_pix + offset_pix*2, half_hi_pix], [0, offset_pix/2], [half_hi_pix - offset_pix*2, half_hi_pix],
                         [half_hi_pix + offset_pix*2, half_hi_pix], [offset_pix/2, 0], [half_hi_pix + offset_pix*2, -half_hi_pix],
                         [half_hi_pix - offset_pix*2, -half_hi_pix], [0, -offset_pix/2], [-half_hi_pix + offset_pix*2, -half_hi_pix]
                         ])

    spokes_mask = visual.ShapeStim(win, vertices=vertices, colorSpace=this_colourSpace,
                                   fillColor=this_bgColour, lineColor=this_bgColour, lineWidth=0)

    probe_mask_list = [spokes_mask]


# BACKGROUND (flow_dots or flow_rings or None)

# settings for dots or rings
# timing for background motion converted to frames (e.g., 70ms is 17frames at 240Hz).
prelim_bg_flow_fr = int(prelim_bg_flow_ms * fps / 1000)
actual_prelim_bg_flow_ms = prelim_bg_flow_fr * 1000 / fps
if verbose:
    print(f'\nprelim_bg_flow_ms: {prelim_bg_flow_ms}')
    print(f'prelim_bg_flow_fr: {prelim_bg_flow_fr}')
    print(f'actual_prelim_bg_flow_ms: {actual_prelim_bg_flow_ms}')

# pale green
flow_colour = [this_bgColour[0]-adj_dots_col, this_bgColour[1], this_bgColour[2]-adj_dots_col]
if monitor_name == 'OLED':
    # darker green for low contrast against black background
    flow_colour = [this_bgColour[0], this_bgColour[1] + adj_dots_col / 2, this_bgColour[2]]
if verbose:
    print(f"flow_dot colours: {flow_colour}")

# # settings for dots OR rings
if background == 'flow_dots':
    # flow_dots
    # the flow speed on the OLED appears half the speed of the 240Hz monitor because the monitor is 120Hz.
    # doubling it on the OLED should rectify this
    dots_speed = 0.2  # 0.2  low values are slow and higher values are faster
    if monitor_name == 'OLED':
        dots_speed = 0.4
    BGspeed = dots_speed
    # todo: do we need to increase the number of dots for OLED?
    nDots = 10000
    # dot_array_width = 10000  # original script used 5000
    # with dot_array_width = widthPix * 3, this gives a values of 5760 on a 1920 monitor,
    # similar to the original setting of 5000.  It also allows the dots to be scaled to the screen for OLED.
    dot_array_width = widthPix * 3  # this scales it for the monitor and keeps more dots on screen

    # todo: most of the dots are off screen using this current dots_min_depth, as the distribution of x_flow has large tails.
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
    if verbose:
        print(f"flow_dots (initial): {np.shape(dots_xys_array)}"
              f"\n{dots_xys_array}")


    flow_dots = visual.ElementArrayStim(win, elementTex=None, elementMask='circle',
                                        units='pix', nElements=nDots, sizes=10,
                                        colorSpace=this_colourSpace, colors=flow_colour)

elif background == 'flow_rings':
    # values for rings
    '''Vary number of rings depending on mon_width_cm(cm, e.g., (int(30.9/10) + 5 for my laptop = 8.
    OLED would be (int(79.7/10) + 5 for my laptop = 13.'''
    # n_rings = 5
    n_rings = (int(mon_width_cm / 10) + 5) * 2
    print(f"mon_width_cm: {mon_width_cm}")
    print(f"mon_width_cm / 100: {mon_width_cm / 10}")
    # todo: scale ring_line_width and rings_speed with monitor size and or view_dist.
    # motion speed looks right at .1 at 60Hz, so using 6 / fps
    # rings_speed = 0.1
    rings_speed = 6 / fps
    BGspeed = rings_speed
    ring_line_width = 25
    # max_radius = calculate_maximum_radius_including_corners((widthPix, heightPix))
    max_radius = heightPix / 2  # with edge_mask on, there is no need to expand to full screen size
    min_radius = int(ring_line_width/2) + 2  # fixation has radius of 2, so at ring_line_width/2, the ring will touch fixation.

    # ring_radii_array are exponentially spaced values between 0 and max_radius (e.g., more dots near centre)
    ring_radii_array = np.geomspace(start=min_radius, stop=max_radius, num=n_rings)
    print(f"ring_radii_array:\n{ring_radii_array}")


    # instead of lots of flow dots, I want to have a few expanding rings, whose radii are defined by the x_flow variable (below)
    ring_list = []

    for i in range(n_rings):

        # alternate between flow_colour and this_bgColour
        if i % 2 == 0:
            this_ring_colour = flow_colour
        else:
            this_ring_colour = this_bgColour

        ring_list.append(visual.Circle(win, radius=2, units='pix',
                                       lineColor=this_ring_colour, lineWidth=ring_line_width,
                                       fillColor=None, colorSpace=this_colourSpace))




# full screen mask to blend off edges and fade to black
# Create a raisedCosine mask array and assign it to a Grating stimulus (grey outside, transparent inside)
# this was useful http://www.cogsci.nl/blog/tutorials/211-a-bit-about-patches-textures-and-masks-in-psychopy
raisedCosTexture2 = visual.filters.makeMask(heightPix, shape='raisedCosine', fringeWidth=0.6, radius=[1.0, 1.0])
invRaisedCosTexture = -raisedCosTexture2  # inverts mask to blur edges instead of center
slab_width = 420
if monitor_name == 'OLED':
    slab_width = 20

blankslab = np.ones((heightPix, slab_width))  # create blank slabs to put to left and right of image
mmask = np.append(blankslab, invRaisedCosTexture, axis=1)  # append blank slab to left
mmask = np.append(mmask, blankslab, axis=1)  # and right
# changed dotsmask color from grey, fades to black round edges which makes screen edges less visible
edge_mask = visual.GratingStim(win, mask=mmask, tex=None, contrast=1.0,
                              size=(widthPix, heightPix), units='pix', color='black')


# MOUSE - hide cursor
myMouse = event.Mouse(visible=False)

# # KEYBOARD
resp = event.BuilderKeyResponse()

# INSTRUCTION
instructions = visual.TextStim(win=win, name='instructions', font='Arial', height=20,
                               color='white', colorSpace=this_colourSpace,
                               text="\n\n\n\n\n\nFocus on the fixation circle at the centre of the screen.\n\n"
                                    "A small white target will briefly appear on screen,\n"
                                    "press the key related to the location of the probe:\n\n"
                                    "[4]/[Q] top-left\t\t\t[5]/[W] top-right\n\n\n\n"
                                    "[1]/[A] bottom-left\t\t\t[2]/[S] bottom-right.\n\n\n"
                                    "Some targets will be easier to see than others,\n"
                                    "Some will be so dim that you won't see them, so just guess!\n\n"
                                    "You don't need to think for long, respond quickly, "
                                    "but try to push press the correct key!\n\n"
                                    "Don't let your eyes wander, keep focussed on the circle in the middle throughout.")


# BREAKS
max_trials = total_n_trials + max_dropped_fr_trials  # expected trials plus repeats
max_without_break = 120  # limit on number of trials without a break
n_breaks = max_trials // max_without_break  # number of breaks
if n_breaks > 0:
    take_break = int(max_trials / (n_breaks + 1))
else:
    take_break = max_without_break
break_dur = 30
if verbose:
    print(f"\ntake a {break_dur} second break every {take_break} trials ({n_breaks} breaks in total).")
break_text = f"Break\nTurn on the light and take at least {break_dur} seconds break.\n" \
             "Keep focussed on the fixation circle in the middle of the screen.\n" \
             "Remember, if you don't see the target, just guess!"
breaks = visual.TextStim(win=win, name='breaks', text=break_text, font='Arial',
                         pos=[0, 0], height=20, ori=0, color='white',
                         colorSpace=this_colourSpace)

end_of_exp_text = "You have completed this experiment.\nThank you for your time.\n\n"
end_of_exp = visual.TextStim(win=win, name='end_of_exp',
                             text=end_of_exp_text, color='white',
                             font='Arial', height=20, colorSpace=this_colourSpace)

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
if monitor_name == 'OLED':
    stairStart = maxLum * 0.3

miniVal = bgLum
maxiVal = maxLum

if verbose:
    print('\nexpInfo (dict)')
    for k, v in expInfo.items():
        print(f"{k}: {v}")

stairs = []
for stair_idx in expInfo['stair_list']:

    thisInfo = copy(expInfo)
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

# counters
# the number of the trial for the output file
trial_number = 0
# the actual number of trials including repeated trials (trial_number stays the same for these)
actual_trials_inc_rpt = 0

# todo: turn on high priority here.


# EXPERIMENT
if verbose:
    print('\n*** exp loop*** \n')
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
            if verbose:
                print(f"\n({actual_trials_inc_rpt}) trial_number: {trial_number}, "
                      f"stair_idx: {stair_idx}, thisStair: {thisStair}, step: {step}")

            # conditions (ISI, congruence)
            ISI = ISI_vals_list[stair_idx]
            congruent = cong_vals_list[stair_idx]
            cong_name = cong_names_list[stair_idx]
            if verbose:
                print(f"ISI: {ISI}, congruent: {congruent} ({cong_name})")

            # conditions (sep, sep_deg, neg_sep)
            sep = sep_vals_list[stair_idx]
            # separation expressed as degrees.
            if -1 < sep < 99:
                sep_deg = sep * pixel_mm_deg_dict['diag_deg']
            else:
                sep_deg = 0

            # negative separation for comparing conditions (e.g., cong sep = 5, incong sep = -5.
            if cong_name == 'incong':
                neg_sep = 0 - sep
                if sep == 0:
                    neg_sep = -.1
            else:
                neg_sep = sep
            if verbose:
                print(f"sep: {sep}, sep_deg: {sep_deg}, neg_sep: {neg_sep}")

            # use congruence to determine the flow direction and target jump direction
            # 1 is contracting/inward/backwards, -1 is expanding/outward/forwards
            flow_dir = np.random.choice([1, -1])
            target_jump = congruent * flow_dir

            exp_rings = False
            if flow_dir == -1:
                exp_rings = True

            # # direction in which the probe jumps : CW or CCW (tangent) or expand vs contract (radial)
            if orientation == 'tangent':
                jump_dir = 'clockwise'
                if target_jump == -1:
                    jump_dir = 'anticlockwise'
            else:  # if radial
                jump_dir = 'cont'
                if target_jump == -1:
                    jump_dir = 'exp'
            if verbose:
                print(f"flow_dir: {flow_dir}, jump dir: {target_jump} {jump_dir} ({cong_name})")

            # vary fixation polarity to reduce risk of screen burn.
            if monitor_name == 'OLED':
                if trial_number % 2 == 0:
                    fixation.lineColor = 'grey'
                    fixation.fillColor = 'black'
                else:
                    fixation.lineColor = 'black'
                    fixation.fillColor = 'grey'

            # Luminance (staircase varies probeLum)
            probeLum = thisStair.next()
            probeColor255 = int(probeLum * LumColor255Factor)  # rgb255 are ints.
            probeColor1 = probeLum / maxLum

            this_probeColor = probeColor255
            if this_colourSpace == 'rgb1':
                this_probeColor = probeColor1
            probe1.setFillColor([this_probeColor, this_probeColor, this_probeColor])
            probe2.setFillColor([this_probeColor, this_probeColor, this_probeColor])
            if verbose:
                print(f"probeLum: {probeLum}, this_probeColor: {this_probeColor}, "
                      f"probeColor255: {probeColor255}, probeColor1: {probeColor1}")


            # PROBE LOCATION
            # # corners go CCW(!) 45=top-right, 135=top-left, 225=bottom-left, 315=bottom-right
            corner = np.random.choice([45, 135, 225, 315])
            corner_name_dict = {45: 'top_right', 135: 'top_left', 225: 'bottom_left', 315: 'bottom_right'}
            corner_name = corner_name_dict[corner]
            if verbose:
                print(f"corner: {corner} {corner_name}")


            # flow_dots
            # todo: should I get new x and y co-ordinates each trial?  If so, uncomment next 2 lines
            # x = np.random.rand(nDots) * dot_array_width - dot_array_width / 2
            # y = np.random.rand(nDots) * dot_array_width - dot_array_width / 2
            '''
            x and y
            np.random.rand(nDots) = array of floats (between 0 and 1).
            dot_array_width = 10000
            np.random.rand(nDots) * dot_array_width = array of floats (between 0 and 10000 e.g., dot_array_width). 
            np.random.rand(nDots) * dot_array_width - dot_array_width / 2 = array of floats (between -dot_array_width/2 and dot_array_width/2).
            e.g., between -5000 and 5000
            it's a fairly uniform distribution
            
            z
            dots_max_depth, dots_min_depth = 5, .5
            np.random.rand(nDots) * (dots_max_depth - dots_min_depth) + dots_min_depth = array of floats (between dots_min_depth and dots_max_depth).
            e.g., floats are multiplied by 4.5 (giving values in the range 0 to 4.5, then .5 is added, giving values in the range .5 to 5).
            this is a fairly uniform distribution.
            Note, z values are updated later (in per frame section) with z = z + dots_speed * flow_dir
            
            x_flow = x / z
            this is an array of floats (between -10000 and 10000) divided by an array of floats (between .5 and 5).
            max x_flow value of (+/-) 5000 if 10000 if divided by .5, and 1000 if divided by 5.          
            So there is a cluster between -1000 and 1000.
            That is, the middle is fairly uniform (between -1000 and 1000) including around 45% of points, 
            but there are tails outside of this including around 55% of points (between -5000 and -1000, and 1000 and 5000).
            
            Better settings would be to set the dots_min_depth to 1 (not .5) so that when the x or y arrays are divided by z, 
            their values only get smaller (or stay the same), not bigger (when dividing by a number less than 1, values get bigger).
            This reduces the tails of the distribution, while keeping a fairly uniform distribution in the middle.
            That way the proportion of dots on screen can increase from ~45% to ~84%. 
            
            # later, (in per frame section), zs are updated with z = z + dots_speed * flow_dir
            dots_speed is currently set to .2.  so zs are updated by adding either .2 or -.2.
            on the first update, xs are divided by new zs which are in range .7 to 5.2.  
            max x values of 5000 is 7142 if divided by .7, and 961 if divided by 5.2.
            
            the next updated, xs are divided by new zs which are in range .9 to 5.4.
            max x values of 5000 is 5555 if divided by .9, and 925 if divided by 5.4.
            
            '''

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


            # VARIABLE FIXATION TIME
            '''to reduce anticipatory effects that might arise from fixation always being same length.
            if False, vary_fix == .5 seconds, so end_fix_fr is 1 second.
            if Ture, vary_fix is between 0 and 1 second, so end_fix_fr is between .5 and 1.5 seconds.'''
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
            end_fix_fr = int(fps / 2) + vary_fix - prelim_bg_flow_fr
            if end_fix_fr < 0:
                end_fix_fr = int(fps / 2)
            end_bg_motion_fr = end_fix_fr + prelim_bg_flow_fr
            end_p1_fr = end_bg_motion_fr + probe_duration
            end_ISI_fr = end_p1_fr + isi_dur_fr
            end_p2_fr = end_ISI_fr + p2_fr
            end_response_fr = end_p2_fr + 10000 * fps  # ~40 seconds to respond
            if verbose:
                print(f"end_fix_fr: {end_fix_fr}, end_p1_fr: {end_p1_fr}, "
                      f"end_ISI_fr: {end_ISI_fr}, end_p2_fr: {end_p2_fr}, end_response_fr: {end_response_fr}\n")


            # continue_routine refers to flipping the screen to show next frame

            # take a break every ? trials
            if (actual_trials_inc_rpt % take_break == 1) & (actual_trials_inc_rpt > 1):
                print("\nTaking a break.\n")

                breaks.text = break_text + f"\n{trial_number - 1}/{total_n_trials} trials completed."
                breaks.draw()
                win.flip()
                event.clearEvents(eventType='keyboard')
                core.wait(secs=break_dur)
                event.clearEvents(eventType='keyboard')
                # todo: turn off high priority here during enforced break?
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

                # blank screen for n frames if on OLED to stop it adapting to bgColor
                if monitor_name == 'OLED':
                    if frameN == 1:
                        win.color = 'black'

                    if frameN == 5:
                        win.color = this_bgColour

                # recording frame durations - from end_fix_fr (1 frame before p1), until 1 frame after p2.
                if frameN == end_fix_fr:

                    # start recording frame intervals
                    if record_fr_durs:
                        win.recordFrameIntervals = True

                    # reset timer to start with probe1 presentation.
                    resp.clock.reset()

                    # clear any previous key presses
                    event.clearEvents(eventType='keyboard')
                    theseKeys = []

                if frameN == end_p2_fr + 1:

                    if record_fr_durs:
                        win.recordFrameIntervals = False


                '''Experiment timings'''
                # FIXATION - up to the end of fixation period
                if end_fix_fr >= frameN > 0:
                    if background != None:
                        # draw dots/rings but with no motion
                        if background == 'flow_dots':
                            flow_dots.xys = dots_xys_array
                            flow_dots.draw()
                        elif background == 'flow_rings':
                            for idx, ring in enumerate(ring_list):
                                # ring.radius = ring_flow[idx]
                                ring.radius = ring_radii_array[idx]
                                ring.draw()

                        # probes are drawn on top of probe mask.  dots/ring are behind probe_mask and edge_mask
                        for probe_mask in probe_mask_list:
                            probe_mask.draw()
                        edge_mask.draw()

                    fixation.setRadius(3)
                    fixation.draw()

                # Background motion prior to probe1 - after fixation, but before probe 1
                elif end_bg_motion_fr >= frameN > end_fix_fr:
                    if background != None:
                        # draw dots/rings with motion
                        if background == 'flow_dots':
                            # get new depth_vals array (z) and dots_xys_array (x, y)
                            z, dots_xys_array = new_dots_depth_and_pos(x, y, z, dots_speed, flow_dir,
                                                                       dots_min_depth, dots_max_depth)
                            flow_dots.xys = dots_xys_array
                            flow_dots.draw()
                        elif background == 'flow_rings':
                            ring_radii_array = get_next_radii(ring_radii_array, rings_speed,
                                                              min_radius, max_radius, expanding=exp_rings)
                            for idx, ring in enumerate(ring_list):
                                ring.radius = ring_radii_array[idx]
                                ring.draw()

                        # probes are drawn on top of probe mask.  dots/ring are behind probe_mask and edge_mask
                        for probe_mask in probe_mask_list:
                            probe_mask.draw()
                        edge_mask.draw()


                    fixation.setRadius(3)
                    fixation.draw()

                # PROBE 1 - after background motion, before end of probe1 interval
                elif end_p1_fr >= frameN > end_bg_motion_fr:
                    if background != None:
                        # draw dots/rings with motion
                        if background == 'flow_dots':
                            # get new depth_vals array (z) and dots_xys_array (x, y)
                            z, dots_xys_array = new_dots_depth_and_pos(x, y, z, dots_speed, flow_dir,
                                                                       dots_min_depth, dots_max_depth)
                            flow_dots.xys = dots_xys_array
                            flow_dots.draw()
                        elif background == 'flow_rings':
                            ring_radii_array = get_next_radii(ring_radii_array, rings_speed,
                                                              min_radius, max_radius, expanding=exp_rings)
                            for idx, ring in enumerate(ring_list):
                                ring.radius = ring_radii_array[idx]
                                ring.draw()

                        # probes are drawn on top of probe mask.  dots/ring are behind probe_mask and edge_mask
                        for probe_mask in probe_mask_list:
                            probe_mask.draw()
                        edge_mask.draw()
                    fixation.setRadius(3)
                    fixation.draw()

                    probe1.draw()
                    if ISI == -1:  # SIMULTANEOUS CONDITION (concurrent)
                        if sep <= 18:  # don't draw 2nd probe in 1probe cond (sep==99)
                            probe2.draw()

                # ISI
                elif end_ISI_fr >= frameN > end_p1_fr:
                    if background != None:
                        if background == 'flow_dots':
                            # get new depth_vals array (z) and dots_xys_array (x, y)
                            z, dots_xys_array = new_dots_depth_and_pos(x, y, z, dots_speed, flow_dir,
                                                                       dots_min_depth, dots_max_depth)
                            flow_dots.xys = dots_xys_array
                            flow_dots.draw()
                        elif background == 'flow_rings':
                            ring_radii_array = get_next_radii(ring_radii_array, rings_speed,
                                                              min_radius, max_radius, expanding=exp_rings)
                            for idx, ring in enumerate(ring_list):
                                ring.radius = ring_radii_array[idx]
                                ring.draw()

                        # probes are drawn on top of probe mask.  dots/ring are behind probe_mask and edge_mask
                        for probe_mask in probe_mask_list:
                            probe_mask.draw()
                        edge_mask.draw()
                    fixation.setRadius(3)
                    fixation.draw()

                # PROBE 2 - after ISI but before end of probe2 interval
                elif end_p2_fr >= frameN > end_ISI_fr:
                    if background != None:
                        if background == 'flow_dots':
                            # get new depth_vals array (z) and dots_xys_array (x, y)
                            z, dots_xys_array = new_dots_depth_and_pos(x, y, z, dots_speed, flow_dir,
                                                                       dots_min_depth, dots_max_depth)
                            flow_dots.xys = dots_xys_array
                            flow_dots.draw()
                        elif background == 'flow_rings':
                            ring_radii_array = get_next_radii(ring_radii_array, rings_speed,
                                                              min_radius, max_radius, expanding=exp_rings)
                            for idx, ring in enumerate(ring_list):
                                ring.radius = ring_radii_array[idx]
                                ring.draw()

                        # probes are drawn on top of probe mask.  dots/ring are behind probe_mask and edge_mask
                        for probe_mask in probe_mask_list:
                            probe_mask.draw()
                        edge_mask.draw()
                    fixation.setRadius(3)
                    fixation.draw()

                    if ISI >= 0:
                        if sep <= 18:  # don't draw 2nd probe in 1probe cond (sep==99)
                            probe2.draw()

                # ANSWER - after probe 2 interval
                elif frameN > end_p2_fr:
                    if background != None:
                        # draw dots/rings but with no motion
                        if background == 'flow_dots':
                            flow_dots.draw()
                        elif background == 'flow_rings':
                            for idx, ring in enumerate(ring_list):
                                ring.radius = ring_radii_array[idx]
                                ring.draw()

                        # probes are drawn on top of probe mask.  dots/ring are behind probe_mask and edge_mask
                        for probe_mask in probe_mask_list:
                            probe_mask.draw()
                        edge_mask.draw()
                    fixation.setRadius(2)
                    fixation.draw()


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

                            # add to empty lists etc.
                            fr_int_per_trial.append(trial_fr_intervals)
                            fr_counter_per_trial.append(list(range(recorded_fr_counter,
                                                                   recorded_fr_counter + len(trial_fr_intervals))))
                            recorded_fr_counter += len(trial_fr_intervals)
                            exp_n_fr_recorded_list.append(exp_n_fr_recorded_list[-1] + n_fr_recorded)
                            cond_list.append(thisStair.name)

                            # get timings for each segment (probe1, ISI, probe2).
                            fr_diff_ms = [(expected_fr_sec - i) * 1000 for i in trial_fr_intervals]

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

                            # check for dropped frames (or frames that are too short)
                            # if timings are bad, repeat trial
                            # if max(trial_fr_intervals) > max_fr_dur_sec or min(trial_fr_intervals) < min_fr_dur_sec:
                            # todo: I've changed this on 13072023 to see if it reduces timing issues.
                            timing_bad = False
                            if max(trial_fr_intervals) > max_fr_dur_sec:
                                logging.warning(f"\n\toh no! Frame too long! {round(max(trial_fr_intervals), 2)} > {round(max_fr_dur_sec, 2)}: "
                                                f"trial: {trial_number}, {thisStair.name}")
                                timing_bad = True

                            if min(trial_fr_intervals) < min_fr_dur_sec:
                                logging.warning(f"\n\toh no! Frame too short! {min(trial_fr_intervals)} < {min_fr_dur_sec}, "
                                                f": trial: {trial_number}, {thisStair.name}")
                                timing_bad = True

                            if timing_bad:
                                repeat = True
                                dropped_fr_trial_counter += 1
                                trial_number -= 1
                                thisStair.trialCount = thisStair.trialCount - 1  # so Kesten doesn't count this trial
                                win.frameIntervals = []
                                continueRoutine = False
                                trial_x_locs = [exp_n_fr_recorded_list[-2], exp_n_fr_recorded_list[-1]]
                                dropped_fr_trial_x_locs.append(trial_x_locs)
                                continue
                                # todo: still plot trial timings if it quits early.  

                            # empty frameIntervals cache
                            win.frameIntervals = []

                        # these belong to the end of the answers section
                        repeat = False
                        continueRoutine = False


                # regardless of frameN, check for quit
                if event.getKeys(keyList=["escape"]):
                    thisExp.close()
                    core.quit()

                # If too many trials have had dropped frames, quit experiment
                if dropped_fr_trial_counter > max_dropped_fr_trials:
                    while not event.getKeys():
                        # display end of experiment screen
                        too_many_dropped_fr.draw()
                        win.flip()
                    else:
                        print(f"{dropped_fr_trial_counter}/{len(fr_int_per_trial)} trials so far with bad timing "
                              f"(expected: {round(expected_fr_ms, 2)}ms, "
                              f"frame_tolerance_ms: +/- {round(frame_tolerance_ms, 2)})")

                        plt_fr_ints(fr_int_per_trial=fr_int_per_trial,
                                    dropped_fr_trial_counter=dropped_fr_trial_counter,
                                    expected_fr_ms=expected_fr_ms, frame_tolerance_ms=frame_tolerance_ms,
                                    cond_list=cond_list, fr_counter_per_trial=fr_counter_per_trial,
                                    dropped_fr_trial_x_locs=dropped_fr_trial_x_locs,
                                    monitor_name=monitor_name, date=expInfo['date'], fps=fps,
                                    participant_name=participant_name, run_number=run_number,
                                    save_dir=save_dir, incomplete=True)

                        # close and quit once a key is pressed
                        thisExp.close()
                        win.close()
                        core.quit()

                # refresh the screen
                if continueRoutine:
                    win.flip()


        # add to thisExp for output csv
        thisExp.addData('trial_number', trial_number)
        thisExp.addData('trial_n_inc_rpt', actual_trials_inc_rpt)
        thisExp.addData('stair', stair_idx)
        thisExp.addData('stair_name', thisStair)
        thisExp.addData('step', step)
        thisExp.addData('separation', sep)
        thisExp.addData('sep_deg', sep_deg)
        thisExp.addData('neg_sep', neg_sep)
        thisExp.addData('ISI', ISI)
        thisExp.addData('ISI_selected_ms', ISI_selected_ms)
        thisExp.addData('ISI_actual_ms', ISI_actual_ms)
        thisExp.addData('isi_dur_fr', isi_dur_fr)
        thisExp.addData('congruent', congruent)
        thisExp.addData('flow_dir', flow_dir)
        thisExp.addData('probe_jump', target_jump)
        thisExp.addData('corner', corner)
        thisExp.addData('corner_name', corner_name)
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
        thisExp.addData('vary_fixation', vary_fixation)
        thisExp.addData('end_fix_fr', end_fix_fr)
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
thisExp.dataFileName = path.join(save_dir, f'{participant_name}_{run_number}_output')
thisExp.close()


# plot frame intervals
if record_fr_durs:

    print(f"{dropped_fr_trial_counter}/{total_n_trials} trials with bad timing "
          f"(expected: {round(expected_fr_ms, 2)}ms, "
          f"frame_tolerance_ms: +/- {round(frame_tolerance_ms, 2)})")

    plt_fr_ints(fr_int_per_trial=fr_int_per_trial, dropped_fr_trial_counter=dropped_fr_trial_counter,
                expected_fr_ms=expected_fr_ms, frame_tolerance_ms=frame_tolerance_ms,
                cond_list=cond_list, fr_counter_per_trial=fr_counter_per_trial,
                dropped_fr_trial_x_locs=dropped_fr_trial_x_locs,
                monitor_name=monitor_name, date=expInfo['date'], fps=fps,
                participant_name=participant_name, run_number=run_number, save_dir=save_dir)



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
