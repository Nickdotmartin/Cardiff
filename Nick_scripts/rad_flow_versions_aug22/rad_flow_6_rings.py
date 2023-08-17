from __future__ import division
from psychopy import gui, visual, core, data, event, monitors, logging

from psychopy import __version__ as psychopy_version
from os import path, chdir
import numpy as np
import copy
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from kestenSTmaxVal import Staircase

print(f"PsychoPy_version: {psychopy_version}")


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

for rad_flow_Martin_2_basic_funcs.py
- changed dot motion from probe1 to end of probe2 (not added prelim yet).  DONE
- changed how probe locations and orientations are generated to use function (if use_pos_ori_func==True, else use original method).  DONE
- changed how dot positions are generated to use function (if use_flow_function==True, else uses original method). DONE
- changed how trial timings are plotted to use function. DONE

for rad_flow_Martin_3_full_options.py
- remove original method for probe locations and orientations, dots_pos, and trial timings.  Remove use_flow_function and use_flow_function.  DONE
- add options to dlg for ISI_ms, verbose, probe_dur, run_number, record_frames, prelim flow, monitor name  DONE
- add vary_fix and prelim_bg_flow options.  DONE
- update how ISI is calculated (ms to fr)  DONE
- add congruent and incongrent, get sep_list, ISI_list, cong_list to make stair_names_list  DONE

for rad_flow_Martin_4_full_options.py
- update flow dots (fuzzy old ones to crisp new ones) DONE
- add axis labels to frame to frame interval plots; expected and bounds to legend; thisStair.name not stair_idx for legend names  DONE
- add dots mask to flow dots  DONE
- update instructions  DONE
- add in breaks (every n trials) and end of exp text DONE
- change monitor details to use int(temp_mon.getSizePix()[0]) or temp_mon.get_width etc  Done
- get tangent probes working  DONE
- added screen number selector DONE
- get rid of BGspeed variable, just using dots_speed  DONE
- changed how trials are repeated (from start of trial, not per-frame bit) and got rid of user repeats  DONE
- add in edge masks  DONE
- add in prelim bg motion period  DONE
- updated speed to be scaled by frame rate to appear the same across monitors  DONE
- change error to be 1ms regardless of fps?  DONE
- added colorSpace=this_colourSpace to all stimuli (probes weren't changing)  DONE
- changes ALL experiments to use RGB1 not RGB255  DONE

rad_flow_Martin_5_contRoutine.py
- change continueRoutine so after keypress it sorts correctAns and timings in segment before the next trial - DONE
- changed verbose to debug, which if True, selects less trials and prints more info to console.  DONE

rad_flow_6_rings.py
- add in rings (as element array stim?)  - DONE
- set variables for rings (min/max depth, n_rings, etc)  - DONE
- confirmed flow_speed has same appearance across monitors  - DONE
- added setting for more realistic dots, with deeper cone and changing sizes - DONE
- add in spokes from vertices
- set it for 'asus_cal', not uncalibrated monitor.

"""

"""
To use this script you will need the width (cm), screen dims (pixels, width heght) and view dist for you
monitor into psychopy monitor centre.  Then select you monior.
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


def new_dots_z_and_pos(x_array, y_array, z_array, dots_speed, flow_dir, min_z, max_z):
    """
    This is a function to update flow_dots depth array and get new pixel co-ordinates
    using the original x_array and y_array.

    1a. Update z_array by adding dots_speed * flow_dir to the current z values.
    1b. adjust any values below dots_min_z or above dots_max_z.

    2a. Get new x_pos and y_pos co-ordinates values by dividing x_array and y_array by the new z_array.
    2b. put the new x_pos and y_pos co-ordinates into an array and transposes it.

    :param x_array: Original x_array positions for the dots (shape = (n_dots, 1))
    :param y_array: Original y_array positions for the dots (shape = (n_dots, 1))
    :param z_array: array of depth values for the dots (shape = (n_dots, 1))
    :param dots_speed: speed of the dots (float, smaller = slower, larger = faster)
    :param flow_dir: either 1 (contracting/inward/backwards) or -1 (expanding/outward/forwards)
    :param dots_min_z: default is .5, values below this are adjusted to dots_max_z
    :param dots_max_z: default is 5, values above this are adjusted to dots_min_z
    :return: new dots_pos_array
    """

    # # 1. Update z (depth values) # #
    # Add dots_speed * flow_dir to the current z values.
    updated_z_array = z_array + dots_speed * flow_dir

    # adjust any depth values below min_z or above max_z by z_adjust
    z_adjust = max_z - min_z
    # adjust updated_z_array values less than min_z by adding z_adjust
    less_than_min = (updated_z_array < min_z)
    updated_z_array[less_than_min] += z_adjust
    # adjust updated_z_array values more than max_z by subtracting z_adjust
    more_than_max = (updated_z_array > max_z)
    updated_z_array[more_than_max] -= z_adjust
    # print(f"updated_z_array (clipped):\n{updated_z_array}\n")

    # # 2. Get new pixel co-ordinates for dots using original x_array and y_array and updated_z_array # #
    x_pos = x_array / updated_z_array
    y_pos = y_array / updated_z_array

    # puts the new co-ordinates into an array and transposes it, ready to use.
    dots_pos_array = np.array([x_pos, y_pos]).T

    return updated_z_array, dots_pos_array


def roll_rings_z_and_colours(z_array, ring_colours, min_z, max_z, flow_dir, flow_speed, initial_x_vals):
    """
    This rings will spawn a new ring if the old one either grows too big for the screen (expanding),
    or shrinks too small (if contracting).

    This function updates the z_array (depth) values for the rings, and adjusts any values below min_z or
    above max_z by z_adjust.  Any values that are adjusted are then rolled to the end or beginning of the array,
    depending on whether they are below min_z or above max_z.
    The same values are then also rolled in the ring_colours array.

    :param z_array: Numpy array of z_array values for the rings (shape = (n_rings, 1))
    :param ring_colours: List of RGB1 colours for the rings (shape = (n_rings, 3))
    :param min_z: minimum depth value for the rings (how close they can get to the screen)
    :param max_z: maximum depth value for the rings (how far away they can get from the screen)
    :param flow_dir: either 1 (contracting/inward/backwards) or -1 (expanding/outward/forwards)
    :param flow_speed: speed of the rings (float, smaller = slower, larger = faster)
    :param initial_x_vals: nupmy array of ring sizes, (all the same size, e.g., 1080, shape = (n_rings, 1))

    :return: z_array (updated), ring_radii_array, ring_colours (rolled if any z_array values are rolled)
    """

    # update depth values
    z_array = z_array + flow_speed * flow_dir

    # z_adjust is the size of change to make to out-of-bounds rings
    z_adjust = max_z - min_z

    # adjust any z_array values below min_z or above max_z by z_adjust
    if flow_dir == -1:  # expanding, getting closer, might be below min_z
        # find which rings are less than min and add z_adjust to those rings
        less_than_min = (z_array < min_z)
        z_array[less_than_min] += z_adjust

        # shift arrays by this amount (e.g., if 3 rings are less than min, shift by 3)
        # (note negative shift to move them backwards)
        shift_num = -sum(less_than_min)

    elif flow_dir == 1:  # contracting, getting further away, might be above max_z
        # find which rings are more_than_max and subtract z_adjust to those rings
        more_than_max = (z_array > max_z)
        z_array[more_than_max] -= z_adjust

        # shift arrays by this amount (e.g., if 3 rings are more_than_max, shift by 3)
        shift_num = sum(more_than_max)

    # roll the depth and colours arrays so that adjusted rings move to other end of array
    z_array = np.roll(z_array, shift=shift_num, axis=0)
    ring_colours = np.roll(ring_colours, shift=shift_num, axis=0)

    # get new ring_radii_array
    ring_radii_array = initial_x_vals / z_array

    # print(f"\nz_array:\n{z_array}\nring_radii_array:\n{ring_radii_array}\nshift_num:\n{shift_num}\n")

    return z_array, ring_radii_array, ring_colours



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

    '''decorate plot'''
    # add legend with colours per condition
    legend_handles_list = []
    if len(unique_conds) < 20:
        for cond in unique_conds:
            leg_handle = mlines.Line2D([], [], color=colour_dict[cond], label=cond,
                                       marker='.', linewidth=.5, markersize=4)
            legend_handles_list.append(leg_handle)

    # add light-grey vertical lines between trials, ofset by -.5
    trial_v_lines = [fr_nums_p_trial[i][0] - .5 for i in range(len(fr_nums_p_trial))]
    for trial_line in trial_v_lines:
        plt.axvline(x=trial_line, color='gainsboro', linestyle='dashed', zorder=0)

    # add horizontal lines: green = expected frame duration, red = frame error tolerance
    plt.axhline(y=expected_fr_dur_ms / 1000, color='green', linestyle='dotted', alpha=.5)
    plt.axhline(y=(expected_fr_dur_ms - allowed_err_ms) / 1000, color='red', linestyle='dotted', alpha=.5)
    plt.axhline(y=(expected_fr_dur_ms + allowed_err_ms) / 1000, color='red', linestyle='dotted', alpha=.5)
    legend_handles_list.append(mlines.Line2D([], [], color='green', label='expected fr duration',
                               linestyle='dotted', linewidth=.5, markersize=0))
    legend_handles_list.append(mlines.Line2D([], [], color='red', label='bad timing boundary',
                               linestyle='dotted', linewidth=.5, markersize=0))

    # plot legend
    plt.legend(handles=legend_handles_list, fontsize=6, title='conditions', framealpha=.5)

    # shade trials red that had bad timing
    for loc_pair in dropped_trial_x_locs:
        x0, x1 = loc_pair[0] - .5, loc_pair[1] - .5
        plt.axvspan(x0, x1, color='red', alpha=0.15, zorder=0, linewidth=None)

    # axis labels and title
    plt.xlabel('frame number')
    plt.ylabel('frame duration (sec)')
    plt.title(f"{mon_name}, {frame_rate}Hz, {date}\n{n_trials_w_dropped_fr}/{total_recorded_trials} trials."
              f"dropped fr (expected: {round(expected_fr_dur_ms, 2)}ms, "
              f"allowed_err_ms: +/- {round(allowed_err_ms, 2)})")

    # save fig
    fig_name = f'{participant}_{run_num}_frames.png'
    if incomplete:
        fig_name = f'{participant}_{run_num}_frames_incomplete.png'
    plt.savefig(path.join(save_path, fig_name))
    plt.close()



# get filename and path for this experiment
_thisDir = path.dirname(path.abspath(__file__))
chdir(_thisDir)
expName = path.basename(__file__)[:-3]


# dialogue box/drop-down option when exp starts (1st item is default val)
expInfo = {'1. Participant': 'Nick_test_17082023',   # 'Nick_orig_dots_17082023',
           '2. Run_number': '1',
           '3. Probe duration in frames': [2, 1, 50, 100],
           '4. fps': [240, 120, 60],
           '5. ISI_dur_in_ms': [25, 16.67, 100, 50, 41.67, 37.5, 33.34, 25, 16.67, 8.33, 0, -1],
           '6. Probe_orientation': ['radial', 'tangent'],
           '7. Record_frame_durs': [True, False],
           '8. Background': ['flow_dots', 'flow_rings', 'no_bg'],
           '9. prelim_bg_flow_ms': [70, 200, 350, 2000],
           '10. monitor_name': ['ASUS_2_13_240Hz', 'asus_cal', 'OLED', 'Nick_work_laptop',
                                'Samsung', 'Asus_VG24', 'HP_24uh', 'NickMac', 'Iiyama_2_18'],
           '12. mask_type': ['4_circles', '2_spokes'],
           '13. debug': [False, True]
           }

# run drop-down menu, OK continues, cancel quits
dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
if dlg.OK == False:
    core.quit()  # user pressed cancel


# Dialogue box settings
participant_name = expInfo['1. Participant']
run_number = int(expInfo['2. Run_number'])
probe_duration = int(expInfo['3. Probe duration in frames'])
fps = int(expInfo['4. fps'])
ISI_selected_ms = float(expInfo['5. ISI_dur_in_ms'])
orientation = expInfo['6. Probe_orientation']
record_fr_durs = eval(expInfo['7. Record_frame_durs'])
background = expInfo['8. Background']
prelim_bg_flow_ms = int(expInfo['9. prelim_bg_flow_ms'])
monitor_name = expInfo['10. monitor_name']
mask_type = expInfo['12. mask_type']
debug = eval(expInfo['13. debug'])

# print settings from dlg
print("dlg dict")
for k, v in expInfo.items():
    print(f'{k}: {v}')


# Misc settings
n_trials_per_stair = 25  # this is the number of trials per stair
if debug:
    n_trials_per_stair = 2
probe_ecc = 4  # int((expInfo['6. Probe eccentricity in deg']))1
vary_fixation = True  # vary fixation period between .5 and 1.5 seconds.
expInfo['time'] = datetime.now().strftime("%H:%M:%S")
expInfo['date'] = datetime.now().strftime("%d/%m/%Y")

'''Probe conditions (staircases): Separation, ISI, Congruence'''
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
ISI_list = [ISI_frames]
if debug:
    print(f"\nSelected {ISI_selected_ms}ms ISI.\n"
          f"At {fps}Hz this is {ISI_frames} frames which each take {round(1000/fps, 2)} ms.\n"
          f"ISI_list (frames): {ISI_list}")


# Separation values in pixels.  select from [18, 6, 3, 2, 1, 0] or 99 for 1probe
#separations = [18, 6, 3, 2, 1, 0]
separations = [6, 3, 1]
if debug:
    separations = [18, 1]

# # main contrast is whether the background and target motion is in same or opposite direction.
# congruence_vals: 1=congruent/same, -1=incongruent/different
congruence_vals = [1, -1]
congruence_names = ['cong', 'incong']
if background == 'no_bg':
    congruence_vals = [1]
    congruence_names = ['No_bg']
if debug:
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
if debug:
    print(f'ISI_vals_list: {ISI_vals_list}')
    print(f'sep_vals_list: {sep_vals_list}')
    print(f'cong_vals_list: {cong_vals_list}')
    print(f'cong_names_list: {cong_names_list}')


# stair_names_list joins cong_names_list, sep_vals_list and ISI_vals_list
# e.g., ['cong_sep18_ISI6', 'cong_sep6_ISI6', 'incong_sep18_ISI6', 'incong_sep6_ISI6', ]
stair_names_list = [f'{p}_sep{s}_ISI{i}' for p, s, i in zip(cong_names_list, sep_vals_list, ISI_vals_list)]
n_stairs = len(sep_vals_list)
total_n_trials = int(n_trials_per_stair * n_stairs)
print(f'\nstair_names_list: {stair_names_list}')
print(f'n_stairs: {n_stairs}')
print(f'total_n_trials: {total_n_trials}')


'''Experiment handling and saving'''
# save each participant's files into separate dir for each ISI
save_dir = path.join(_thisDir, expName, participant_name,
                     background, f'bg{prelim_bg_flow_ms}',
                     f'{participant_name}_{run_number}',
                     f'ISI_{ISI_frames}')
print(f"\nexperiment save_dir: {save_dir}")

# files are labelled as '_incomplete' unless entire script runs.
incomplete_output_filename = f'{participant_name}_{run_number}_incomplete'
save_output_as = path.join(save_dir, incomplete_output_filename)

# Experiment Handler
thisExp = data.ExperimentHandler(name=expName, version=psychopy_version,
                                 extraInfo=expInfo, runtimeInfo=None,
                                 savePickle=None, saveWideText=True,
                                 dataFileName=save_output_as)


'''MONITOR/screen/window details: colour, luminance, pixel size and frame rate'''
# COLORS AND LUMINANCES
# Lum to Color255
LumColor255Factor = 2.39538706913372
# Color255 to Color1
Color255Color1Factor = 1 / 127.5  # Color255 * Color255Color1Factor -1
# Lum to Color1
Color1LumFactor = 2.39538706913372  ###

maxLum = 106  # 255 RGB
# minLum = 0.12  # 0 RGB  # todo: this is currently unused
bgLumProp = .45  # .2  # todo: use .45 to match radial_flow_NM_v2.py, or .2 to match exp1
bgLum = maxLum * bgLumProp

# colour space
this_colourSpace = 'rgb1'  # values between 0 and 1
bgColor_rgb1 = bgLum / maxLum
this_bgColour = [bgColor_rgb1, bgColor_rgb1, bgColor_rgb1]

# Flow colours
adj_flow_colour = .15
# Give dots a pale green colour, which is adj_flow_colour different to the background
flow_colour = [this_bgColour[0] - adj_flow_colour, this_bgColour[1], this_bgColour[2] - adj_flow_colour]
if monitor_name == 'OLED':  # darker green for low contrast against black background
    flow_colour = [this_bgColour[0], this_bgColour[1] + adj_flow_colour / 2, this_bgColour[2]]


# MONITOR SPEC
if debug:
    print(f"\nmonitor_name: {monitor_name}")
mon = monitors.Monitor(monitor_name)

widthPix = int(mon.getSizePix()[0])
heightPix = int(mon.getSizePix()[1])
mon_width_cm = mon.getWidth()  # monitor width in cm
view_dist_cm = mon.getDistance()  # viewing distance in cm
view_dist_pix = widthPix / mon_width_cm*view_dist_cm  # used for calculating visual angle (e.g., probe locations at 4dva)

# set monitor values
mon.setSizePix((widthPix, heightPix))
mon.setDistance(view_dist_cm)
mon.setWidth(mon_width_cm)

# screen number
display_number = 1  # 0 indexed, 1 for external display, 0 for internal
if monitor_name in ['asus_cal', 'Nick_work_laptop', 'NickMac', 'OLED', 'ASUS_2_13_240Hz']:
    display_number = 0

# WINDOW SPEC
win = visual.Window(monitor=mon, size=(widthPix, heightPix), colorSpace=this_colourSpace, color=this_bgColour,
                    units='pix', screen=display_number, allowGUI=False, fullscr=True)


'''ELEMENTS'''
# MOUSE
myMouse = event.Mouse(visible=False)

# # KEYBOARD
resp = event.BuilderKeyResponse()


# fixation bull eye
fixation = visual.Circle(win, radius=2, units='pix', lineColor='white', fillColor='black', colorSpace=this_colourSpace)
# todo: add change fixation colours on OLED (if if trial_num % 2 == 1: fixation.fillColor = 'white', lineColor = 'black')


# PROBEs
probe_size = 1  # can make them larger for testing new configurations etc
probeVert = [(0, 0), (1, 0), (1, 1), (2, 1), (2, -1), (1, -1), (1, -2), (-1, -2), (-1, -1), (0, -1)]  # 5 pixels
probe1 = visual.ShapeStim(win, vertices=probeVert, lineWidth=0, opacity=1, size=probe_size, interpolate=False,
                          colorSpace=this_colourSpace)
probe2 = visual.ShapeStim(win, vertices=probeVert, lineWidth=0, opacity=1, size=probe_size, interpolate=False,
                          colorSpace=this_colourSpace)

# probes and probe_masks are at dist_from_fix pixels from middle of the screen
dist_from_fix = int((np.tan(np.deg2rad(probe_ecc)) * view_dist_pix) / np.sqrt(2))


# MASK BEHIND PROBES (infront of flow dots to keep probes and motion separate)
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
                                units='pix', tex=None, pos=[dist_from_fix + 1, -dist_from_fix - 1])

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




# flow speed should be scaled by fps, so dots have a greater change per frame on slower monitors.
# e.g., .2 at 240Hz, .4 at 120Hz and .8 at 60Hz.
# todo: this appears too fast to me, but it is the same as the original script.
flow_speed = 48 / fps



# timing for background motion converted to frames (e.g., 70ms is 17frames at 240Hz).
prelim_bg_flow_fr = int(prelim_bg_flow_ms * fps / 1000)
actual_prelim_bg_flow_ms = prelim_bg_flow_fr * 1000 / fps

# flow_dots - e.g., moving background to simulate self motion
if background == 'flow_dots':

    # If False, use orginal settings, if True, increase dots depth and scale their size with depth
    deep_with_sizes = False

    dots_speed = flow_speed

    # Changing dots_min_z from .5 to one means that the proportion of dots onscreen increases from ~42% to ~82%.
    # Therefore, I can half n_dots with little change in the number of dots onscreen, saving processing resources.
    # Note: 'onscreen' was defined as half widthPix (960).  Once the edge mask is added, the square of the visible screen is 1080x1080,
    # minus the blurred edges, so 960 seems reasonable.
    dots_min_z = 1.0  # original script used .5, which increased the tails meaning more dots were offscreen.
    dots_max_z = 5.5  # depth values  # todo: changed to 5.5 to match original script depth range?
    if deep_with_sizes:
        # increase cone depth
        dots_max_depth = 101

    # todo: do we need to increase n_dots for OLED?
    n_dots = 5000

    # dot_array_spread is the spread of x and ys BEFORE they are divided by their depth value to get actual positions.
    # with dot_array_spread = widthPix * 3, this gives a values of 5760 on a 1920 monitor,
    # similar to the original setting of 5000.  It also allows the flow_dots to be scaled to the screen for OLED.
    dot_array_spread = widthPix * 3  # this scales it for the monitor and keeps more flow_dots on screen

    # initial array values.  x and y are scaled by z_array, so x and y values can be larger than the screen.
    # x and y are the position of the dots when they are at depth = 1.  These values can be larger than the monitor.
    # at depths > 1, x and y are divided by z_array, so they are appear closer to the middle of the screen
    x_array = np.random.rand(n_dots) * dot_array_spread - dot_array_spread / 2
    y_array = np.random.rand(n_dots) * dot_array_spread - dot_array_spread / 2
    if deep_with_sizes:
        # narrower spread of dots
        x = np.random.uniform(-widthPix, widthPix, n_dots)
        y = np.random.uniform(-widthPix, widthPix, n_dots)
    z_array = np.random.rand(n_dots) * (dots_max_z - dots_min_z) + dots_min_z
    # print(f"x_array: {x_array}, y_array: {y_array}, z_array: {z_array}")

    # x_flow and y_flow are the actual x_array and y_array positions of the dots, after being divided by their depth value.
    x_flow = x_array / z_array
    y_flow = y_array / z_array

    # array of x_array, y_array positions of dots to pass to ElementArrayStim
    dots_xys_array = np.array([x_flow, y_flow]).T

    dot_sizes = 10
    if deep_with_sizes:
        dot_sizes = 50

    # itialise flow_dots
    flow_dots = visual.ElementArrayStim(win, elementTex=None, elementMask='circle',  # orig used 'gauss'
                                        units='pix', nElements=n_dots, sizes=dot_sizes,
                                        colorSpace=this_colourSpace, colors=flow_colour)
    if deep_with_sizes:
        flow_dots.sizes = dot_sizes / z_array

elif background == 'flow_rings':
    # # # RINGS
    ring_speed = flow_speed / 4  # todo: this is a quarter the speed of the dots .02  # 48 / fps  # 0.2 at 240Hz
    n_rings = 100  # scale this to screen size?
    rings_min_z = .1  # A value < 1 of .1 means that the closest ring's radius is 10x the size of the screen.
    # print(f"ring_speed: {ring_speed}")
    # print(f"n_rings: {n_rings}")
    # print(f"rings_min_z: {rings_min_z}")

    # set the limits on ring size
    max_radius = heightPix  # Biggest ring is height of screen
    min_radius = 10  # smallest ring is 10 pixels

    # If I want the smallest radius to be 10 pixels, then the max depth of 108 (1080/108=10)
    rings_max_z = max_radius / min_radius
    # print(f"rings_max_z: {rings_max_z}")

    # adjust ring depth values by rings_z_adjust
    rings_z_adjust = rings_max_z - rings_min_z

    '''
    Dots_array_width was used to give the dots unique x/y positions in 'space'.
    For rings, they are all at the same x/y position (0, 0), so I don't need dot_array_wdith for them.
    '''
    ring_size_list = [1080] * n_rings
    # print(f"ring_size_list: {ring_size_list}")

    # depth values are evenly spaces and in ascending order, so smaller rings are drawn on top of larger ones.
    # stop=stop=rings_max_z-(rings_z_adjust/n_rings) gives space the new ring to appear
    ring_z_array = np.linspace(start=rings_min_z, stop=rings_max_z - (rings_z_adjust / n_rings), num=n_rings)

    # the actual radii list is in descending order, so smaller rings are drawn on top of larger ones.
    ring_radii_array = ring_size_list / ring_z_array

    # RING COLOURS (alernating this_bgColour and flow_colour
    ring_colours = [this_bgColour, flow_colour] * int(n_rings / 2)

    # # use ElementArrayStim to draw the rings
    flow_rings = visual.ElementArrayStim(win, elementTex=None, elementMask='circle', interpolate=True,
                                         units='pix', nElements=n_rings, sizes=ring_radii_array,
                                         colors=ring_colours, colorSpace=this_colourSpace)
elif background == 'no_bg':

    # if No moving background, use these values (see below for if there is a moving background)
    n_dots = 0  # no dots
    dots_speed = None
    dot_array_spread = None  # this scales it for the monitor and keeps more flow_dots on screen
    dots_min_z = None
    dots_max_z = None  # depth values

    # settings for dots or rings
    # timing for background motion converted to frames (e.g., 70ms is 17frames at 240Hz).
    prelim_bg_flow_fr = 0
    actual_prelim_bg_flow_ms = prelim_bg_flow_fr * 1000 / fps


if debug:
    print(f'\nprelim_bg_flow_ms: {prelim_bg_flow_ms}')
    print(f'prelim_bg_flow_fr: {prelim_bg_flow_fr}')
    print(f'actual_prelim_bg_flow_ms: {actual_prelim_bg_flow_ms}')
    print(f'flow_speed: {flow_speed}')
    if background == 'flow_dots':
        print(f'dots_speed: {dots_speed}')
        print(f'n_dots: {n_dots}')
        print(f'dot_array_spread: {dot_array_spread}')
        print(f'dots_min_z: {dots_min_z}')
        print(f'dots_max_z: {dots_max_z}')
    elif background == 'flow_rings':
        print(f"ring_speed: {ring_speed}")
        print(f"n_rings: {n_rings}")
        print(f"rings_min_z: {rings_min_z}")
        print(f"rings_max_z: {rings_max_z}")



'''Timing: expected frame duration and tolerance
with frame_tolerance_prop = .24, frame_tolerance_ms == 1ms at 240Hz, 2ms at 120Hz, 4ms at 60Hz
For a constant frame_tolerance_ms of 1ms, regardless of fps, use frame_tolerance_prop = 1/expected_fr_sec
Psychopy records frames in seconds, but I prefer to think in ms. So wo variables are labelled with _sec or _ms.
'''
expected_fr_sec = 1/fps
expected_fr_ms = expected_fr_sec * 1000
frame_tolerance_prop = 1/expected_fr_ms  # frame_tolerance_ms == 1ms, regardless of fps..
max_fr_dur_sec = expected_fr_sec + (expected_fr_sec * frame_tolerance_prop)
min_fr_dur_sec = expected_fr_sec - (expected_fr_sec * frame_tolerance_prop)
frame_tolerance_ms = (max_fr_dur_sec - expected_fr_sec) * 1000
max_dropped_fr_trials = 10  # number of trials with dropped frames to allow before experiment is aborted
if debug:
    print(f"\nexpected_fr_ms: {expected_fr_ms}")
    print(f"frame_tolerance_prop: {frame_tolerance_prop}")
    print(f"frame_tolerance_ms: {frame_tolerance_ms}")
    print(f"max_dropped_fr_trials: {max_dropped_fr_trials}")


# empty variable to store recorded frame durations
fr_int_per_trial = []  # nested list of frame durations for each trial (y values)
recorded_fr_counter = 0  # how many frames have been recorded
fr_counter_per_trial = []  # nested list of recorded_fr_counter values for plotting frame intervals (x values)
cond_list = []  # stores stair name for each trial, to colour code plot lines and legend
dropped_fr_trial_counter = 0  # counter for how many trials have dropped frames
dropped_fr_trial_x_locs = []  # nested list of [1st fr of dropped fr trial, 1st fr of next trial] for trials with dropped frames


'''Messages to display on screen'''
instructions = visual.TextStim(win=win, name='instructions', font='Arial', height=20,
                               color='white', colorSpace=this_colourSpace,
                               wrapWidth=widthPix / 2,
                               text="\n\n\nFocus on the fixation circle at the centre of the screen.\n\n"
                                    "A small white target will briefly appear on screen,\n"
                                    "press the key related to the location of the probe:\n\n"
                                    "[4]/[Q] top-left\t\t\t[5]/[W] top-right\n\n\n\n"
                                    "[1]/[A] bottom-left\t\t\t[2]/[S] bottom-right.\n\n\n"
                                    "Some targets will be easy to see, others will be hard to spot.\n"
                                    "If you aren't sure whether you saw the target, just guess!\n\n"
                                    "You don't need to think for long, respond quickly, "
                                    "but try to push press the correct key!\n\n"
                                    "Don't let your eyes wander, keep focussed on the circle in the middle throughout.")


too_many_dropped_fr = visual.TextStim(win=win, name='too_many_dropped_fr',
                                      text="The experiment had quit as the computer is dropping frames.\n"
                                           "Sorry for the inconvenience.\n"
                                           "Please contact the experimenter.\n\n"
                                           "Press any key to return to the desktop.",
                                      font='Arial', height=20, colorSpace=this_colourSpace)

# BREAKS
max_trials = total_n_trials + max_dropped_fr_trials  # expected trials plus repeats
max_without_break = 120  # limit on number of trials without a break
n_breaks = max_trials // max_without_break  # number of breaks
if n_breaks > 0:
    take_break = int(max_trials / (n_breaks + 1))
else:
    take_break = max_without_break
break_dur = 30
if debug:
    print(f"\ntake a {break_dur} second break every {take_break} trials ({n_breaks} breaks in total).")
break_text = f"Break\nTurn on the light and take at least {break_dur} seconds break.\n" \
             "Keep focussed on the fixation circle in the middle of the screen.\n" \
             "Remember, if you don't see the target, just guess!"
breaks = visual.TextStim(win=win, name='breaks', text=break_text, font='Arial',
                         pos=[0, 0], height=20, ori=0, color='white',
                         colorSpace=this_colourSpace)

end_of_exp_text = "You have completed this experiment.\nThank you for your time."
end_of_exp = visual.TextStim(win=win, name='end_of_exp',
                             text=end_of_exp_text, color='white',
                             font='Arial', height=20, colorSpace=this_colourSpace)

# show instructions screen
while not event.getKeys():
    fixation.draw()
    instructions.draw()
    win.flip()


'''Construct staircases'''
# start luminance value
stairStart = maxLum
if monitor_name == 'OLED':  # dimmer on OLED
    stairStart = maxLum * 0.3

stairs = []
for stair_idx in range(n_stairs):
    thisInfo = copy.copy(expInfo)
    thisInfo['stair_idx'] = stair_idx

    thisStair = Staircase(name=stair_names_list[stair_idx],
                          type='simple',
                          value=stairStart,
                          C=stairStart * 0.6,  # initial step size, as prop of maxLum
                          minRevs=3,
                          minTrials=n_trials_per_stair,
                          minVal=bgLum,
                          maxVal=maxLum,
                          targetThresh=0.75,
                          extraInfo=thisInfo)
    stairs.append(thisStair)


'''Run experiment'''
# counters
trial_num_inc_repeats = 0  # number of trials including repeated trials
trial_number = 0  # the number of the trial for the output file

# # todo: turn on high priority here. (and turn off garbage collection)
# import gc
# gc.disable()
# core.rush(True)
# if monitor_name == 'OLED':
#     core.rush(True, realtime=True)


for step in range(n_trials_per_stair):
    np.random.shuffle(stairs)
    for thisStair in stairs:

        # # Assume the trial needs to be repeated until I've confirmed that no frames were dropped
        repeat = True
        while repeat:

            # Trial, stair and step
            trial_number += 1
            trial_num_inc_repeats += 1
            stair_idx = thisStair.extraInfo['stair_idx']
            if debug:
                print(f"\n({trial_num_inc_repeats}) trial_number: {trial_number}, "
                      f"stair_idx: {stair_idx}, thisStair: {thisStair}, step: {step}")

            # conditions (ISI, congruence)
            ISI = ISI_vals_list[stair_idx]
            congruent = cong_vals_list[stair_idx]
            cong_name = cong_names_list[stair_idx]
            if debug:
                print(f"ISI: {ISI}, congruent: {congruent} ({cong_name})")

            # conditions (sep, neg_sep)
            sep = sep_vals_list[stair_idx]

            # negative separation for comparing conditions (e.g., cong sep = 5, incong sep = -5.
            if cong_name == 'incong':
                neg_sep = 0 - sep
                if sep == 0:
                    neg_sep = -.1
            else:
                neg_sep = sep
            if debug:
                print(f"sep: {sep}, neg_sep: {neg_sep}")

            # Luminance (staircase varies probeLum)
            probeLum = thisStair.next()
            # probeColor255 = int(probeLum * LumColor255Factor)  # rgb255 are ints.
            probeColor1 = probeLum / maxLum

            # this_probeColor = probeColor255
            # if this_colourSpace == 'rgb1':
            #     this_probeColor = probeColor1
            this_probeColor = probeColor1
            probe1.fillColor = [this_probeColor, this_probeColor, this_probeColor]
            probe2.fillColor = [this_probeColor, this_probeColor, this_probeColor]
            if debug:
                print(f"probeLum: {probeLum}, this_probeColor: {this_probeColor}, "
                      # f"probeColor255: {probeColor255}, "
                      f"probeColor1: {probeColor1}")


            # PROBE LOCATION
            corner = np.random.choice([45, 135, 225, 315])

            # use congruence to determine the flow direction and target jump direction
            # 1 is contracting/inward/backwards, -1 is expanding/outward/forwards
            flow_dir = np.random.choice([1, -1])
            target_jump = congruent * flow_dir


            # PROBE POSITION (including shift around dist_from_fix)
            probe_pos_dict = get_probe_pos_dict(sep, target_jump, corner, dist_from_fix,
                                                probes_ori=orientation, verbose=debug)

            # loc_marker.setPos([loc_x, loc_y])
            probe1.setPos(probe_pos_dict['probe1_pos'])
            probe1.setOri(probe_pos_dict['probe1_ori'])
            probe2.setPos(probe_pos_dict['probe2_pos'])
            probe2.setOri(probe_pos_dict['probe2_ori'])
            if debug:
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
            if debug:
                print(f"end_fix_fr: {end_fix_fr}, end_bg_motion_fr: {end_bg_motion_fr}, "
                      f"end_p1_fr: {end_p1_fr}, end_ISI_fr: {end_ISI_fr}, end_p2_fr: {end_p2_fr}\n")


            # take a break every ? trials
            if (trial_num_inc_repeats % take_break == 1) & (trial_num_inc_repeats > 1):
                if debug:
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
                # else continue the trial routine (per frame section).
                continueRoutine = True


            '''TRIAL ROUTINE (per frame)'''
            frameN = -1
            # continueRoutine = True
            # # continueRoutine here runs the per-frame section of the trial
            while continueRoutine:
                frameN = frameN + 1

                # move record frames analysis to outside continue routine

                '''Turn recording on and off from just before probe1 til just after probe2. '''
                if frameN == end_bg_motion_fr:
                    if record_fr_durs:  # start recording frames just before probe1 presentation
                        win.recordFrameIntervals = True

                    # clear any previous key presses
                    event.clearEvents(eventType='keyboard')
                    theseKeys = []

                    # reset timer to start with probe1 presentation.
                    resp.clock.reset()


                # stop recording frame intervals
                elif frameN == end_p2_fr + 1:
                    if record_fr_durs:
                        win.recordFrameIntervals = False


                # FIXATION
                if end_fix_fr >= frameN > 0:
                    if background == 'flow_dots':
                        flow_dots.xys = dots_xys_array
                        flow_dots.draw()
                    elif background == 'flow_rings':
                        flow_rings.draw()

                    probeMask1.draw()
                    probeMask2.draw()
                    probeMask3.draw()
                    probeMask4.draw()
                    edge_mask.draw()

                    fixation.setRadius(3)
                    fixation.draw()


                # Background motion prior to probe1 - after fixation, but before probe 1
                elif end_bg_motion_fr >= frameN > end_fix_fr:
                    if background == 'flow_dots':
                        # get new depth_vals array (z_array) and dots_xys_array (x_array, y_array)
                        z_array, dots_xys_array = new_dots_z_and_pos(x_array, y_array, z_array, dots_speed, flow_dir,
                                                                   dots_min_z, dots_max_z)
                        flow_dots.xys = dots_xys_array
                        if deep_with_sizes:
                            flow_dots.sizes = dot_sizes / z_array
                        flow_dots.draw()
                    elif background == 'flow_rings':
                        ring_z_array, ring_radii_array, ring_colours = roll_rings_z_and_colours(ring_z_array,
                                                                                                ring_colours,
                                                                                                rings_min_z,
                                                                                                rings_max_z,
                                                                                                flow_dir, ring_speed,
                                                                                                ring_size_list)
                        flow_rings.sizes = ring_radii_array
                        flow_rings.colors = ring_colours
                        flow_rings.draw()

                    probeMask1.draw()
                    probeMask2.draw()
                    probeMask3.draw()
                    probeMask4.draw()
                    edge_mask.draw()
                    fixation.setRadius(3)
                    fixation.draw()

                # PROBE 1
                elif end_p1_fr >= frameN > end_bg_motion_fr:
                    if background == 'flow_dots':
                        # get new depth_vals array (z_array) and dots_xys_array (x_array, y_array)
                        z_array, dots_xys_array = new_dots_z_and_pos(x_array, y_array, z_array, dots_speed, flow_dir,
                                                                   dots_min_z, dots_max_z)
                        flow_dots.xys = dots_xys_array
                        if deep_with_sizes:
                            flow_dots.sizes = dot_sizes / z_array
                        flow_dots.draw()
                    elif background == 'flow_rings':
                        ring_z_array, ring_radii_array, ring_colours = roll_rings_z_and_colours(ring_z_array,
                                                                                                ring_colours,
                                                                                                rings_min_z,
                                                                                                rings_max_z,
                                                                                                flow_dir, ring_speed,
                                                                                                ring_size_list)
                        flow_rings.sizes = ring_radii_array
                        flow_rings.colors = ring_colours
                        flow_rings.draw()

                    probeMask1.draw()
                    probeMask2.draw()
                    probeMask3.draw()
                    probeMask4.draw()
                    edge_mask.draw()
                    fixation.setRadius(3)
                    fixation.draw()

                    probe1.draw()
                    # SIMULTANEOUS CONDITION
                    if ISI == -1:
                        if sep <= 18:
                            probe2.draw()



                # ISI
                elif end_ISI_fr >= frameN > end_p1_fr:
                    if background == 'flow_dots':
                        # get new depth_vals array (z_array) and dots_xys_array (x_array, y_array)
                        z_array, dots_xys_array = new_dots_z_and_pos(x_array, y_array, z_array, dots_speed, flow_dir,
                                                                   dots_min_z, dots_max_z)
                        flow_dots.xys = dots_xys_array
                        if deep_with_sizes:
                            flow_dots.sizes = dot_sizes / z_array
                        flow_dots.draw()
                    elif background == 'flow_rings':
                        ring_z_array, ring_radii_array, ring_colours = roll_rings_z_and_colours(ring_z_array,
                                                                                                ring_colours,
                                                                                                rings_min_z,
                                                                                                rings_max_z,
                                                                                                flow_dir, ring_speed,
                                                                                                ring_size_list)
                        flow_rings.sizes = ring_radii_array
                        flow_rings.colors = ring_colours
                        flow_rings.draw()

                    probeMask1.draw()
                    probeMask2.draw()
                    probeMask3.draw()
                    probeMask4.draw()
                    edge_mask.draw()
                    fixation.setRadius(3)
                    fixation.draw()

                # PROBE 2
                elif end_p2_fr >= frameN > end_ISI_fr:
                    if background == 'flow_dots':
                        # get new depth_vals array (z_array) and dots_xys_array (x_array, y_array)
                        z_array, dots_xys_array = new_dots_z_and_pos(x_array, y_array, z_array, dots_speed, flow_dir,
                                                                   dots_min_z, dots_max_z)
                        flow_dots.xys = dots_xys_array
                        if deep_with_sizes:
                            flow_dots.sizes = dot_sizes / z_array
                        flow_dots.draw()
                    elif background == 'flow_rings':
                        ring_z_array, ring_radii_array, ring_colours = roll_rings_z_and_colours(ring_z_array,
                                                                                                ring_colours,
                                                                                                rings_min_z,
                                                                                                rings_max_z,
                                                                                                flow_dir, ring_speed,
                                                                                                ring_size_list)
                        flow_rings.sizes = ring_radii_array
                        flow_rings.colors = ring_colours
                        flow_rings.draw()

                    probeMask1.draw()
                    probeMask2.draw()
                    probeMask3.draw()
                    probeMask4.draw()
                    edge_mask.draw()
                    fixation.setRadius(3)
                    fixation.draw()
                    if ISI >= 0:  # if not concurrent condition (ISI=-1)
                        if sep != 99:  # If not 1probe condition (sep = 99)
                            probe2.draw()


                # ANSWER
                elif frameN > end_p2_fr:
                    if background == 'flow_dots':
                        flow_dots.draw()
                    elif background == 'flow_rings':
                        flow_rings.draw()

                    probeMask1.draw()
                    probeMask2.draw()
                    probeMask3.draw()
                    probeMask4.draw()
                    edge_mask.draw()
                    fixation.setRadius(2)
                    fixation.draw()

                    # RESPONSE HANDLING
                    theseKeys = event.getKeys(keyList=['num_5', 'num_4', 'num_1', 'num_2', 'w', 'q', 'a', 's'])
                    if len(theseKeys) > 0:  # at least one key was pressed
                        resp.keys = theseKeys[-1]  # just the last key pressed
                        resp.rt = resp.clock.getTime()

                        # a response ends the per-frame_section
                        continueRoutine = False


                # check for quit
                if event.getKeys(keyList=["escape"]):
                    core.quit()

                # refresh the screen
                if continueRoutine:
                    win.flip()


            '''End of per-frame section in continueRoutine = False"'''

            # CHECK RESPONSES
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



            '''sort frame interval times to use for plots later'''
            if record_fr_durs:
                # actual frame interval times (in seconds) for this trial
                trial_fr_intervals = win.frameIntervals
                fr_int_per_trial.append(trial_fr_intervals)

                # add list of contiguous frame numbers for this trial
                fr_counter_per_trial.append(list(range(recorded_fr_counter,
                                                       recorded_fr_counter + len(trial_fr_intervals))))
                recorded_fr_counter += len(trial_fr_intervals)

                # add condition name for this staircase
                cond_list.append(thisStair.name)

                # empty frameIntervals cache
                win.frameIntervals = []

                # check for dropped frames (or frames that are too short)
                if max(trial_fr_intervals) > max_fr_dur_sec or min(trial_fr_intervals) < min_fr_dur_sec:

                    # Timing is bad, this trial will be repeated (with new corner and target_jump)
                    if debug:
                        print(f"\n\toh no! A frame had bad timing! trial: {trial_number}, {thisStair.name}"
                              f"{round(max(trial_fr_intervals), 2)} > {round(max_fr_dur_sec, 2)} or "
                              f"{round(min(trial_fr_intervals), 2)} < {round(min_fr_dur_sec, 2)}")

                    print(f"Timing bad, trial {trial_number} repeated\nrepeat: {repeat}, continueRoutine: {continueRoutine}")

                    # decrement trial and stair so that the correct values are used for the next trial
                    trial_number -= 1
                    thisStair.trialCount = thisStair.trialCount - 1  # so Kesten doesn't count this trial

                    # get first and last frame numbers for this trial
                    trial_x_locs = [fr_counter_per_trial[-1][0], fr_counter_per_trial[-1][-1] + 1]  # 1st fr of this trial to 1st of next trial
                    dropped_fr_trial_x_locs.append(trial_x_locs)
                    dropped_fr_trial_counter += 1
                    continue
                else:
                    repeat = False  # breaks out of while repeat=True loop to progress to new trial
                    # print(f"Timing good, trial {trial_number} not repeated\nrepeat: {repeat}, continueRoutine: {continueRoutine}")


            # # # trial completed # # #

            # If too many trials have had dropped frames, quit experiment
            if dropped_fr_trial_counter > max_dropped_fr_trials:
                while not event.getKeys():
                    # display too_many_dropped_fr message until screen is pressed
                    too_many_dropped_fr.draw()
                    win.flip()
                else:
                    # print text to screen with dropped frames info and make plt_fr_ints()
                    print(f"{dropped_fr_trial_counter}/{trial_num_inc_repeats} trials so far with bad timing "
                          f"(expected: {round(expected_fr_ms, 2)}ms, "
                          f"frame_tolerance_ms: +/- {round(frame_tolerance_ms, 2)})")
                    plt_fr_ints(time_p_trial_nested_list=fr_int_per_trial,
                                n_trials_w_dropped_fr=dropped_fr_trial_counter,
                                expected_fr_dur_ms=expected_fr_ms, allowed_err_ms=frame_tolerance_ms,
                                all_cond_name_list=cond_list, fr_nums_p_trial=fr_counter_per_trial,
                                dropped_trial_x_locs=dropped_fr_trial_x_locs,
                                mon_name=monitor_name, date=expInfo['date'], frame_rate=fps,
                                participant=participant_name, run_num=run_number,
                                save_path=save_dir, incomplete=True)


                    # close and quit once a key is pressed
                    thisExp.close()
                    win.close()
                    core.quit()

        # add trial info to csv
        thisExp.addData('trial_number', trial_number)
        thisExp.addData('trial_n_inc_rpt', trial_num_inc_repeats)
        thisExp.addData('stair', stair_idx)
        thisExp.addData('stair_name', thisStair)
        thisExp.addData('step', step)
        thisExp.addData('separation', sep)
        # thisExp.addData('sep_deg', sep_deg)
        thisExp.addData('neg_sep', neg_sep)
        thisExp.addData('ISI', ISI)
        thisExp.addData('ISI_selected_ms', ISI_selected_ms)
        thisExp.addData('ISI_actual_ms', ISI_actual_ms)
        thisExp.addData('isi_dur_fr', isi_dur_fr)
        thisExp.addData('congruent', congruent)
        thisExp.addData('flow_dir', flow_dir)
        thisExp.addData('probe_jump', target_jump)
        thisExp.addData('corner', corner)
        # thisExp.addData('corner_name', corner_name)
        thisExp.addData('probeLum', probeLum)
        thisExp.addData('probeColor1', probeColor1)
        # thisExp.addData('probeColor255', probeColor255)
        thisExp.addData('trial_response', resp.corr)
        thisExp.addData('resp.rt', resp.rt)
        thisExp.addData('probe_ecc', probe_ecc)
        thisExp.addData('flow_speed', flow_speed)
        thisExp.addData('orientation', orientation)
        thisExp.addData('background', background)
        thisExp.addData('background', background)
        thisExp.addData('vary_fix', vary_fix)
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

    print(f"{dropped_fr_trial_counter}/{trial_num_inc_repeats} trials with bad timing "
          f"(expected: {round(expected_fr_ms, 2)}ms, "
          f"frame_tolerance_ms: +/- {round(frame_tolerance_ms, 2)})")

    plt_fr_ints(time_p_trial_nested_list=fr_int_per_trial, n_trials_w_dropped_fr=dropped_fr_trial_counter,
                expected_fr_dur_ms=expected_fr_ms, allowed_err_ms=frame_tolerance_ms,
                all_cond_name_list=cond_list, fr_nums_p_trial=fr_counter_per_trial,
                dropped_trial_x_locs=dropped_fr_trial_x_locs,
                mon_name=monitor_name, date=expInfo['date'], frame_rate=fps,
                participant=participant_name, run_num=run_number,
                save_path=save_dir, incomplete=False)


# # todo: turn off high priority mode and turn garbage collection back on
# gc.enable()
# core.rush(False)


# display end of experiment screen with dropped_fr_trial_counter, then allow continue after 5 seconds (to allow for processes to finish)
end_of_exp_text2 = end_of_exp_text + f"\n\n{dropped_fr_trial_counter}/{trial_num_inc_repeats} trials with bad timing."
end_of_exp.text = end_of_exp_text2
end_of_exp_text3 = end_of_exp_text2 + "\n\nPress any key to continue."
while not event.getKeys():
    end_of_exp.draw()
    win.flip()
    core.wait(secs=5)
    end_of_exp.text = end_of_exp_text3
    end_of_exp.draw()
    win.flip()
else:
    logging.flush()  # write messages out to all targets
    thisExp.abort()  # or data files will save again on exit

    # close and quit once a key is pressed
    win.close()
    core.quit()
