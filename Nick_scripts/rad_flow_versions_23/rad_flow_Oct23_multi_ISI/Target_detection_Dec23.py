from __future__ import division
from psychopy import gui, visual, core, data, event, monitors, logging
from psychopy import __version__ as psychopy_version
from datetime import datetime
from os import path, chdir
from copy import copy
from kestenSTmaxVal import Staircase
from PsychoPy_tools import get_pixel_mm_deg_values
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import gc
# for numpy attributes access by per-frame functions, acceess them with name instead of np.name.
import numpy as np
from numpy import array, random, where, sum, linspace, pi, rad2deg, arctan, arctan2, cos, sin, hypot
print(f"PsychoPy_version: {psychopy_version}")


'''
Selectively eliminate attribute access – Every use of the dot (.) operator to access attributes comes with a cost. 
One can often avoid attribute lookups by using the 'from module import name' form of import statement,
and accessing the name directly (e.g.., name() instead of module.name()).
However, it must be emphasized that these changes only make sense in frequently executed code, such as loops. 
So, this optimization really only makes sense in carefully selected places.
https://www.geeksforgeeks.org/python-making-program-run-faster/

Similarly, putting things inside functions (rather than in main code) can make them run faster, 
because they are only compiled once, rather than each time the code is run.
It has something to do with local variables being faster to access than global variables.

'''


"""
VERSION HISTORY
This script is adapted from Martin's EXPERIMENT3-backgroundMotion.py, 
(Martin also has a radial version called integration_RiccoBloch_flow_new.py which is in 
r'C:\ Users \sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\Martin_scripts\EXPERIMENTS_INTEGRATION\EXPERIMENTS_SCRIPTS_NEW\old scripts\integration_RiccoBloch_flow_new.py'
or Downloads.  

to make rad_flow_martin.py
- my cleaned up version of Martin's script, import statments, record frame ints. DONE

for rad_flow_Martin_1_names_no_rot.py
- rename variables, removed rotating motion, added no motion option.  

for rad_flow_Martin_2_basic_funcs.py
- changed dot motion from probe1 to end of probe2 (not added prelim yet), added fnction for probe locations.

for rad_flow_Martin_3_full_options.py
- updated dlg, added vary_fix and prelim_bg_flow options, added congruent and incongrent,

for rad_flow_Martin_4_full_options.py


rad_flow_Martin_5_contRoutine.py
- change continueRoutine so after keypress it sorts correctAns and timings in segment before the next trial - DONE
- changed verbose to debug, which if True, selects less trials and prints more info to console.  DONE

rad_flow_6_rings.py
- add in rings (as element array stim?)  - DONE

rad_flow_7_spokes.py
- add in setting for OLED (bgLumProp, startLum etc) - DONE
- add in spokes as option to compare with 4CircleMasks - Done

rad_flow_8_prelim_interleaved.py
- add in interleaved prelim period (with bg motion)  - DONE
- changed prelim_dir to 'interleaved' - DONE

rad_flow_new_spokes.py - OCT 23
- added in new functions for dots with spokes from flow_parse exp scripts DONE
- changed imports to minimise attribute access DONE
- removed variables from dlg - now all hard coded values DONE
     background - always flow_dots
     orientation - always radial
     record frame ints - always True
     prelim - has multiple prelims in staircase conds
- alternate fixation colours each trial DONE
- removed 'interleaved' dir from file structure and added monitor DONE

rad_flow_multi_ISI_v1.py - 20th Oct 23
- switch exp logic around, select one separation value and run a range of interleaved ISIs DONE
- put no bg option back in.  DONE

rad_flow_v2_motion_window.py - 23rd Oct 23
- instead of a fixed prelim motion variable, there is a background motion window (bg_motion), and the probes are presented in the middle.
    e.g., if the window is 10 frames and the ISI and probes are each 2 frames, then the probes are presented in frames 3 and 7. DONE

rad_flow_v3_motion_window.py - 1st Nov 23
- changed staircase, smaller starting val, higher c_multiplier (step size).  - DONE
- changed bg_lum to be 5%, not black  - DONE
- same settings for all monitors as OLED (apart from probe verts and core.rush) - DONE

Target_detection_Dec23.py (version to use for experiment.)
- motion_duration (window) is set to 200ms - Done
- maxLum has been updated - Done
"""

"""
To use this script you will need the width (cm), screen dims (pixels, width height) and view dist for you
monitor into psychopy monitor centre.  Then select your monior.
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
        random.shuffle(extra_shifted_pixel)
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
        whereas probe1_ori refers to rotational angle of probe stimulus.
        CW and ACW are clockwise and anticlockwise, respectively.'''
        if probes_ori == 'tangent':
            if target_jump == 1:  # CW; probe2 is right and down from probe1
                probe1_ori += 180
                probe2_ori += 180
                probe1_pos = [loc_x - p1_shift, loc_y + p1_shift]
                probe2_pos = [loc_x + p2_shift - probe_size, loc_y - p2_shift]
            elif target_jump == -1:  # ACW; probe2 is left and up from probe1
                probe1_ori += 0
                probe2_ori += 0
                probe1_pos = [loc_x + p1_shift, loc_y - p1_shift]
                probe2_pos = [loc_x - p2_shift + probe_size, loc_y + p2_shift]
        elif probes_ori == 'radial':
            if target_jump == 1:  # inward; probe2 is left and down from probe1
                probe1_ori += 270
                probe2_ori += 270
                probe1_pos = [loc_x + p1_shift, loc_y + p1_shift]
                probe2_pos = [loc_x - p2_shift + probe_size, loc_y - p2_shift]
            elif target_jump == -1:  # outward; probe2 is right and up from probe1
                probe1_ori += 90
                probe2_ori += 90
                probe1_pos = [loc_x - p1_shift, loc_y - p1_shift]
                probe2_pos = [loc_x + p2_shift - probe_size, loc_y + p2_shift]
    elif corner == 135:  # top-left
        loc_x = dist_from_fix * -1
        loc_y = dist_from_fix * 1
        if probes_ori == 'tangent':
            if target_jump == 1:  # ACW; probe2 is left and down from probe1
                probe1_ori += 90
                probe2_ori += 90
                probe1_pos = [loc_x - p1_shift, loc_y - p1_shift]
                probe2_pos = [loc_x + p2_shift - probe_size, loc_y + p2_shift]
            elif target_jump == -1:  # CW; probe2 is right and up from probe1
                probe1_ori += 270
                probe2_ori += 270
                probe1_pos = [loc_x + p1_shift, loc_y + p1_shift]
                probe2_pos = [loc_x - p2_shift + probe_size, loc_y - p2_shift]
        elif probes_ori == 'radial':
            if target_jump == 1:  # inward; probe2 is right and down from probe1
                probe1_ori += 180
                probe2_ori += 180
                probe1_pos = [loc_x - p1_shift, loc_y + p1_shift]
                probe2_pos = [loc_x + p2_shift - probe_size, loc_y - p2_shift]
            elif target_jump == -1:  # outward; probe2 is left and up from probe1
                probe1_ori += 0
                probe2_ori += 0
                probe1_pos = [loc_x + p1_shift, loc_y - p1_shift]
                probe2_pos = [loc_x - p2_shift + probe_size, loc_y + p2_shift]
    elif corner == 225:  # bottom-left
        loc_x = dist_from_fix * -1
        loc_y = dist_from_fix * -1
        if probes_ori == 'tangent':
            if target_jump == 1:  # CW; probe2 is left and up from probe1
                probe1_ori += 0
                probe2_ori += 0
                probe1_pos = [loc_x + p1_shift, loc_y - p1_shift]
                probe2_pos = [loc_x - p2_shift + probe_size, loc_y + p2_shift]
            elif target_jump == -1:  # ACW; probe2 is right and down from probe1
                probe1_ori += 180
                probe2_ori += 180
                probe1_pos = [loc_x - p1_shift, loc_y + p1_shift]
                probe2_pos = [loc_x + p2_shift - probe_size, loc_y - p2_shift]
        elif probes_ori == 'radial':
            if target_jump == 1:  # inward; probe2 is right and up from probe1
                probe1_ori += 90
                probe2_ori += 90
                probe1_pos = [loc_x - p1_shift, loc_y - p1_shift]
                probe2_pos = [loc_x + p2_shift - probe_size, loc_y + p2_shift]
            elif target_jump == -1:  # outward; probe2 is left and down from probe1
                probe1_ori += 270
                probe2_ori += 270
                probe1_pos = [loc_x + p1_shift, loc_y + p1_shift]
                probe2_pos = [loc_x - p2_shift + probe_size, loc_y - p2_shift]
    else:
        corner = 315  # bottom-right
        loc_x = dist_from_fix * 1
        loc_y = dist_from_fix * -1
        if probes_ori == 'tangent':
            if target_jump == 1:  # ACW; probe2 is right and up from probe1
                probe1_ori += 270
                probe2_ori += 270
                probe1_pos = [loc_x + p1_shift, loc_y + p1_shift]
                probe2_pos = [loc_x - p2_shift + probe_size, loc_y - p2_shift]
            elif target_jump == -1:  # CW; probe2 is left and down from probe1
                probe1_ori += 90
                probe2_ori += 90
                probe1_pos = [loc_x - p1_shift, loc_y - p1_shift]
                probe2_pos = [loc_x + p2_shift - probe_size, loc_y + p2_shift]
        elif probes_ori == 'radial':
            if target_jump == 1:  # inward; probe2 is left and up from probe1
                probe1_ori += 0
                probe2_ori += 0
                probe1_pos = [loc_x + p1_shift, loc_y - p1_shift]
                probe2_pos = [loc_x - p2_shift + probe_size, loc_y + p2_shift]
            elif target_jump == -1:  # outward; probe2 is right and down from probe1
                probe1_ori += 180
                probe2_ori += 180
                probe1_pos = [loc_x - p1_shift, loc_y + p1_shift]
                probe2_pos = [loc_x + p2_shift - probe_size, loc_y - p2_shift]

    probe_pos_dict = {'probe1_pos': probe1_pos, 'probe2_pos': probe2_pos,
                      'probe1_ori': probe1_ori, 'probe2_ori': probe2_ori,
                      'corner': corner, 'loc_x': loc_x, 'loc_y': loc_y}

    return probe_pos_dict


def find_angle(adjacent, opposite):
    """Finds the angle in a right triangle given the lengths of the adjacent and opposite sides.
    e.g., for getting the visual angle of a square at a given distance,
    the adjacent side is the distance from the screen,
    and the opposite side is the size of the square onscreen.

    :param adjacent: A numpy array of the lengths (in cm) of the adjacent sides (e.g., distance z_array).
    :param opposite: The (scalar) length (in cm) of the side opposite the angle you want to find.
    :return: A numpy array of the angles in degrees.
    """
    return rad2deg(arctan(opposite / adjacent))



def check_z_start_bounds(z_array, closest_z, furthest_z, max_dot_life_fr, dot_life_array, flow_dir):
    """
    check all z values.  If they are out of bounds (too close when expanding or too far when contracting), then
    set their dot life to max, so they are redrawn with new x, y and z values.

    :param z_array: array of current dot distances
    :param closest_z: near boundary for z values (relevant when expanding)
    :param furthest_z: far boundary for z values (relevant when contracting)
    :param max_dot_life_fr: maximum lifetime of a dot in frames.
    :param dot_life_array: array of dot lifetimes (ints) between 0 and dot_max_fr.
    :param flow_dir: either 1 (contracting/inward/backwards) or -1 (expanding/outward/forwards):
    :return: updated dot_life_array
    """

    # if expanding, check if any z values are too close or far, and if so, set their dot life to max
    if flow_dir == -1:  # expanding
        dot_life_array = where(z_array > furthest_z, max_dot_life_fr, dot_life_array)
    elif flow_dir == 1:  # contracting
        dot_life_array = where(z_array < closest_z, max_dot_life_fr, dot_life_array)

    return dot_life_array


def update_dotlife(dotlife_array, dot_max_fr,
                   x_array, y_array, z_array,
                   x_bounds, y_bounds, z_start_bounds):
    """
    Function to update the lifetime of the dots.  Dots that have reached their maximum lifetime
    have their life reset to zero and are redrawn with new x, y and z values.

    1. increment all dots by 1
    2. make a mask of any to be replaced (life >= max_life)
    3. replace these with new x, y and z values
    4. reset life of replaced dots to 0

    :param dotlife_array: np.array of dot lifetimes (ints) between 0 and dot_max_fr.
    :param dot_max_fr: maximum lifetime of a dot in frames.
    :param x_array: np.array of x positions of dots (in meters).
    :param y_array: np.array of y positions of dots (in meters).
    :param z_array: np.array of z positions of dots (in meters).

    :param x_bounds: value passed for distribution of x_values, from -x_bounds to x_bounds.  Half the width of the array.
    :param y_bounds: value passed for distribution of y_values, from -y_bounds to y_bounds.  Half the height of the array.
    :param z_start_bounds: tuple, values passed for distribution of z_values, from z_start_bounds[0] to z_start_bounds[1].
    :return: updated dotlife_array, x_array, y_array, z_array
    """

    # increment all dots by 1
    dotlife_array += 1

    # make a mask of any to be replaced (life >= max_life)
    replace_mask = (dotlife_array >= dot_max_fr)

    # replace these with new x and y values (from same distribution as originals)
    x_array[replace_mask] = random.uniform(low=-x_bounds, high=x_bounds, size=sum(replace_mask))
    y_array[replace_mask] = random.uniform(low=-y_bounds, high=y_bounds, size=sum(replace_mask))
    z_array[replace_mask] = random.uniform(low=z_start_bounds[0], high=z_start_bounds[1], size=sum(replace_mask))

    # reset life of replaced dots to 0
    dotlife_array[replace_mask] = 0

    return dotlife_array, x_array, y_array, z_array



def make_xy_spokes(x_array, y_array):
    """
    Function to take dots evenly spaced across screen, and make it so that they appear in
    4 'spokes' (top, bottom, left and right).  That is, wedge shaped regions, with the point of the
    wedge at the centre of the screen, and the wide end at the edge of the screen.
    There are four blank regions with no dots between each spoke, extending to the four corners of the screen.
    Probes are presented in the four corners, so using make_xy_spokes means that the probes are never presented
    on top of dots.

    1. get constants to use:
        rad_eighth_slice is the wedge width in radians (e.g., 45 degrees)
        rad_octants is list of 8 equally spaced values between -pi and pi, ofset by rad_sixteenth_slice (e.g., -22.5 degrees)


    rad_octants (like quadrants, but eight of them, e.g., 45 degrees)
        ofset them by adding rad_eighth_slice / 2 to them  (e.g., equivillent to 22.5 degrees).
        I've hard coded these, so they don't need to be calculated each frame.
    2. convert cartesian (x, y) co-ordinates to polar co-ordinates (e.g., distance and angle (radians) from centre).
    3. rotate values between pairs of rad_octants by rad_sixteenth_slice (e.g., -45 degrees).
    4. add 2*pi to any values less than -pi, to make them positive, but similarly rotated (360 degrees is 2*pi radians).
    5. convert back to cartesian co-ordinates.

    :param x_array: numpy array of x values with shape (n_dots, 1), 0 as middle of screen.
    :param y_array: numpy array of y values with shape (n_dots, 1), 0 as middle of screen.
    :return: new x_array and y_array
    """


    # # # CONSTANT VALUES TO USE # # #
    # # spokes/wedges width is: degrees = 360 / 8 = 45; radians = 2*pi / 8 = pi / 4 = 0.7853981633974483
    rad_eighth_slice = 0.7853981633974483

    # # rad_octants is list of 8 equally spaced values between -pi and pi, ofset by rad_sixteenth_slice (e.g., -22.5 degrees)
    # # in degrees this would be [22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5]
    # rad_octants = [i + rad_eighth_slice / 2 for i in linspace(-pi, pi, 8, endpoint=False)]
    rad_octants = [-2.748893571891069, -1.9634954084936207, -1.1780972450961724, -0.39269908169872414,
                   0.39269908169872414, 1.1780972450961724, 1.9634954084936207, 2.748893571891069]


    # # # RUN FUNCTION USING CONSTANTS # # #
    # Convert Cartesian coordinates to polar coordinates.
    # r is distance, theta is angle in radians (from -pi to pi)
    r_array, theta_array = hypot(x_array, y_array), arctan2(y_array, x_array)

    # # make a mask for values between pairs of rad_octants in theta_array
    mask = ((theta_array >= rad_octants[0]) & (theta_array < rad_octants[1])) | \
                ((theta_array >= rad_octants[2]) & (theta_array < rad_octants[3])) | \
                    ((theta_array >= rad_octants[4]) & (theta_array < rad_octants[5])) | \
                        ((theta_array >= rad_octants[6]) & (theta_array < rad_octants[7]))

    # rotate values specified by mask by rad_eighth_slice (e.g., -45 degrees)
    theta_array[mask] -= rad_eighth_slice

    # if any values are less than -pi, add 2*pi to make them positive, but similarly rotated (360 degrees is 2*pi radians)
    theta_array = where(theta_array < -pi, theta_array + 2*pi, theta_array)

    # convert r and theta arrays back to x and y arrays (e.g., radians to cartesian)
    return r_array * cos(theta_array), r_array * sin(theta_array)  # x_array, y_array



def scaled_dots_pos_array(x_array, y_array, z_array, frame_size_cm, reference_angle):
    """
    This is a function to get new pixel x, y co-ordinates for the flow dots using the x, y and z arrays.
    Use this after updating z_array and dot_life_array.

    :param x_array: Original x_array positions for the dots (shape = (n_dots, 1))
    :param y_array: Original y_array positions for the dots (shape = (n_dots, 1))
    :param z_array: array of distance values for the dots (shape = (n_dots, 1))
    :param frame_size_cm: onscreen size in cm of frame containing dots.
    :param reference_angle: angle in degrees of the reference distance (e.g., screen size angle at 57.3cm)
    :return: new dots_pos_array
    """

    # 1. convert frame size at z distances to angles and
    # 2. scale these by dividing by reference angle (e.g., screen size at view dist)
    scale_factor_array = find_angle(adjacent=z_array, opposite=frame_size_cm) / reference_angle

    # 3. scale x and y values by multiplying by scaled distances and
    # 4. put scaled x and y values into an array and transpose it.
    return array([x_array * scale_factor_array, y_array * scale_factor_array]).T



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


################################################################################

#######################
# # # MAIN SCRIPT # # #
#######################

# get filename and path for this experiment
_thisDir = path.dirname(path.abspath(__file__))
chdir(_thisDir)
expName = path.basename(__file__)[:-3]


# # # DIALOGUE BOX # # #
# dialogue box/drop-down option when exp starts (1st item is default val)
expInfo = {'01. Participant ID': '',
           '02. Separation': [4, 6],   # [4, 2, 4, 6, 8, 0, 10],
           '03. monitor_name': ['OLED', 'Nick_work_laptop', 'asus_cal',
                                'Asus_VG24', 'HP_24uh', 'NickMac'],
           '04. debug': [False, True]
           }

# run drop-down menu, OK continues, cancel quits
dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
if not dlg.OK:
    core.quit()  # user pressed cancel

# stop experiment if participant name is missing
if expInfo['01. Participant ID'] == '':
    raise ValueError('Participant name is missing.')
    core.quit()


# Dialogue box settings
participant_ID = expInfo['01. Participant ID']
separation = int(expInfo['02. Separation'])
monitor_name = expInfo['03. monitor_name']
debug = eval(expInfo['04. debug'])


# check participant folder to find out which runs have already been completed
check_run_path = path.join(_thisDir, expName, monitor_name, participant_ID)
if not path.exists(check_run_path):
    run_number = 1
else:
    for i in list(range(12)):
        run_number = i + 1
        check_run_path = path.join(_thisDir, expName, monitor_name, participant_ID,
                                   f'{participant_ID}_{run_number}', f'sep_{separation}')
        if not path.isfile(path.join(check_run_path, f'{participant_ID}_{run_number}_output.csv')):
            break
        elif run_number == 12:
            raise ValueError(f"Participant {participant_ID} has already completed all {run_number} runs of {expName}")

# print settings from dlg
print("\ndlg dict")
for k, v in expInfo.items():
    print(f'{k}: {v}')


# # # MISC SETTINGS # # #
n_trials_per_stair = 25  # this is the number of trials per stair
if debug:
    n_trials_per_stair = 2
probe_ecc = 4  # probe eccentricity in dva
vary_fixation = True  # vary fixation time between trials to reduce anticipatory effects
record_fr_durs = True  # always record frame durs
expInfo['date'] = datetime.now().strftime("%d/%m/%Y")
expInfo['time'] = datetime.now().strftime("%H:%M:%S")

# these setting for final experiment version
orientation = 'radial'  # 'radial' or 'tangent'
background = 'flow_dots'  # 'flow_dots' or 'no_bg' for no background
selected_bg_motion_ms = 200  # bg_motion duration, stim presented in the middle. 0 for no radial motion

# get fps and probe duration from moniotor name
probe_duration = 1  # Probe duration in frames
if monitor_name == 'OLED':
    fps = 120  # int(expInfo['04. fps'])
elif monitor_name == 'asus_cal':
    probe_duration = 2
    fps = 240
else:
    fps = 60


# # # EXPERIMENT HANDLING AND SAVING # # #
# save each participant's files into separate dir for each ISI
save_dir = path.join(_thisDir, expName, monitor_name,  # added monitor name to analysis structure
                     participant_ID,
                     f'{participant_ID}_{run_number}',
                     f'sep_{separation}')
print(f"\nexperiment save_dir: {save_dir}")

# files are labelled as '_incomplete' unless entire script runs.
p_name_run = f"{participant_ID}_{run_number}"
if debug:
    p_name_run = f"{participant_ID}_{run_number}_debug"
incomplete_output_filename = f'{p_name_run}_incomplete'
save_output_as = path.join(save_dir, incomplete_output_filename)

# Experiment Handler
thisExp = data.ExperimentHandler(name=expName, version=psychopy_version,
                                 extraInfo=expInfo, runtimeInfo=None,
                                 savePickle=None, saveWideText=True,
                                 dataFileName=save_output_as)


# # # CONDITIONS AND STAIRCASES # # #
'''ISI can be given (roughly) in ms, for any monitor it will try to match that value in frames.
The script uses on frames so ISI will be used in that format.
The actual ms timing is given for record.
This means that the experiment should have similar ms timings on monitors with different fps
milliseconds: [100, 50, 41.66, 37.5, 33.34, 25, 16.67, 8.33, 0]
frames@240hz: [24,  12,  10,    9,    8,     6,  4,    2,    0]
frames@120hz: [12,   6,   5,     ,    4,     3,  2,    1,    0]
frames@60hz:  [ 6,   3,    ,     ,    2,      ,  1,     ,    0]
'''
# selected_isi_ms_vals = [100, 50, 41.66, 33.34, 25, 16.67, 8.33, 0, -1]  # dropped 37.5 / ISI 9 as it won't run on OLED
selected_isi_ms_vals = [100, 50, 41.66, 33.34, 25, 16.67, 8.33, 0, -1]  # dropped 37.5 / ISI 9 as it won't run on OLED
isi_fr_vals = [-1 if i == -1 else int(i * fps / 1000) for i in selected_isi_ms_vals]
isi_ms_vals = [-1 if i == -1 else round((1 / fps) * i * 1000, 2) for i in isi_fr_vals]

# # keep isi_name in to use for checking conditions against frame rate
isi_name_vals = ['conc' if i == -1 else f'{int(i)}ms' for i in selected_isi_ms_vals]
isi_selected_zip = list(zip(isi_ms_vals, isi_fr_vals, isi_name_vals))

# remove conditions that aren't possible at this frame rate
isi_zip = []
for this_zip in isi_selected_zip:
    act_ms_int = int(this_zip[0])

    if act_ms_int == -1:  # concurrent
        isi_zip.append(this_zip)
        name_int = this_zip[2]
    else:  # not concurrent
        name_int = int(float(this_zip[2][:-2]))
        if name_int == act_ms_int:
            isi_zip.append(this_zip)
        else:
            print(f"dropping ISI: {this_zip}, as it is not possible on a monitor that is {round(1000/fps, 2)}ms per frame.")

if debug:
    print(f"isi_zip: {isi_zip}")

# # Conditions/staricases: ISI, Congruence (cong, incong)
# # main contrast is whether the background and target motion is in same or opposite direction.
# congruence_vals: -1=incongruent/different; 1=congruent/same,
congruence_vals = [-1, 1]
congruence_names = ['incong', 'cong']
if background == 'no_bg':
    congruence_vals = [1]
    congruence_names = ['no_bg']
cong_zip = list(zip(congruence_names, congruence_vals))

if debug:
    print(f'congruence_vals: {congruence_vals}')
    print(f'congruence_names: {congruence_names}')
    print(f"cong_zip: {cong_zip}")


# get all possible combinations of these three lists
combined_conds = [(i, cz) for i in isi_zip for cz in cong_zip]
print(f"combined_conds: {combined_conds}")

# split the combined_conds into separate lists
isi_ms_conds_list = [i[0][0] for i in combined_conds]
isi_fr_conds_list = [i[0][1] for i in combined_conds]
cong_name_conds_list = [i[1][0] for i in combined_conds]
cong_val_conds_list = [i[1][1] for i in combined_conds]

if debug:
    print(f'isi_ms_conds_list: {isi_ms_conds_list}')
    print(f'isi_fr_conds_list: {isi_fr_conds_list}')
    print(f'cong_val_conds_list: {cong_val_conds_list}')
    print(f'cong_name_conds_list: {cong_name_conds_list}')


stair_names_list = [f"ISI_{iz[2]}_{cz[0]}_{cz[1]}" for iz, cz in combined_conds]
if background == 'no_bg':
    stair_names_list = [f"ISI_{iz[2]}_{cz[0]}" for iz, cz in combined_conds]


n_stairs = len(combined_conds)
total_n_trials = int(n_trials_per_stair * n_stairs)
print(f'\nstair_names_list: {stair_names_list}')
print(f'n_stairs: {n_stairs}, total_n_trials: {total_n_trials}')


# # # MONITOR SETTINGS # # #
# # COLOURS AND LUMINANCE
# updated maxLum values from spyder measurements
maxLum = 120  # default value
if monitor_name == 'OLED':
    maxLum = 25  # from spyder_15112023.xlsx
elif monitor_name == ('asus_cal'):
    maxLum = 151  # from Spyder_23022023.xlsx
else:
    maxLum = 120  # default value for all other monitors

# now using same bg lum settings for all monitors
bgLumProp = 0   # use .2 to match exp1 or .45 to match radial_flow_NM_v2.py
bgLum = 0.0 / 1000  # hard coded to zero rather than maxLum * bgLumProp

# colour space
this_colourSpace = 'rgb1'  # values between 0 and 1
bgColor_rgb1 = bgLum / maxLum
this_bgColour = [bgColor_rgb1, bgColor_rgb1, bgColor_rgb1]
print(f"this_colourSpace: {this_colourSpace}, bgLumProp: {bgLumProp}; bgLum: {bgLum}; bgColor_rgb1: {bgColor_rgb1}")

# Flow colours
# Give dots a pale green colour, which is adj_flow_colour different to the background
adj_flow_colour = .2
flow_colour = [this_bgColour[0], this_bgColour[1] + adj_flow_colour, this_bgColour[2]]


# # # MONITOR DETAILS # # #
if debug:
    print(f"\nmonitor_name: {monitor_name}")
mon = monitors.Monitor(monitor_name)

widthPix = int(mon.getSizePix()[0])
heightPix = int(mon.getSizePix()[1])
mon_width_cm = mon.getWidth()  # monitor width in cm
view_dist_cm = mon.getDistance()  # viewing distance in cm
view_dist_pix = widthPix / mon_width_cm * view_dist_cm  # used for calculating visual angle (e.g., probe locations at 4dva)
mon_height_cm = mon_width_cm / (widthPix/heightPix)  # used for calculating visual angle of dots

# screen number
display_number = 1  # 0 indexed, 1 for external display, 0 for internal
if monitor_name in ['asus_cal', 'Nick_work_laptop', 'NickMac', 'OLED', 'ASUS_2_13_240Hz']:
    display_number = 0

# WINDOW SPEC
win = visual.Window(monitor=mon, size=(widthPix, heightPix),
                    winType='pyglet', bpc=[10, 10, 10],  # pyglet should ow work for 10-bit
                    colorSpace=this_colourSpace, color=this_bgColour,
                    units='pix', screen=display_number, allowGUI=False, fullscr=True, useFBO=False)

# pixel size
pixel_mm_deg_dict = get_pixel_mm_deg_values(monitor_name=monitor_name)
if debug:
    print(f"diagonal pixel size: {pixel_mm_deg_dict['diag_mm']} mm, or {pixel_mm_deg_dict['diag_deg']} dva")


# # # PSYCHOPY COMPONENTS # # #
# MOUSE
# todo: check forum for other ideas if mouse is still there
win.mouseVisible = False
# myMouse = event.Mouse(visible=False)

# # KEYBOARD
resp = event.BuilderKeyResponse()

# fixation bull eye
fixation = visual.Circle(win, radius=2, units='pix', lineColor='white', fillColor='black', colorSpace=this_colourSpace)

# add a small blurred mask behind fixation so dots are separated from fxation and less dirstracting
fix_mask_size = 75
# Create a raisedCosine mask array and assign it to a Grating stimulus (grey outside, transparent inside)
raisedCosTexture1 = visual.filters.makeMask(256, shape='raisedCosine',
                                            # fringeWidth=0.3,
                                            fringeWidth=0.8,  # proportion of mask that is blured (0 to 1)
                                            radius=[1.0, 1.0])
fix_mask = visual.GratingStim(win=win, mask=raisedCosTexture1, size=(fix_mask_size, fix_mask_size),
                                colorSpace=this_colourSpace,
                                color=this_bgColour,
                                tex=None, units='pix')

# PROBEs
probe_size = 1  # can make them larger for testing new configurations etc
probeVert = [(0, 0), (1, 0), (1, 1), (2, 1), (2, -1), (1, -1),
             (1, -2), (-1, -2), (-1, -1), (0, -1)]  # 5 pixels

if monitor_name == 'OLED':  # smaller, 3-pixel probes for OLED
    probeVert = [(0, 0), (1, 0), (1, 1), (2, 1),
                 (2, 0), (1, 0), (1, -1), (0, -1),
                 (0, -2), (-1, -2), (-1, -1), (0, -1)]

probe1 = visual.ShapeStim(win, vertices=probeVert, lineWidth=0, opacity=1, size=probe_size, interpolate=False,
                          colorSpace=this_colourSpace)
probe2 = visual.ShapeStim(win, vertices=probeVert, lineWidth=0, opacity=1, size=probe_size, interpolate=False,
                          colorSpace=this_colourSpace)

# probes and probe_masks are at dist_from_fix pixels from middle of the screen
dist_from_fix = int((np.tan(np.deg2rad(probe_ecc)) * view_dist_pix) / np.sqrt(2))


# full screen mask to blend off edges and fade to black
# Create a raisedCosine mask array and assign it to a Grating stimulus (grey outside, transparent inside)
# this was useful http://www.cogsci.nl/blog/tutorials/211-a-bit-about-patches-textures-and-masks-in-psychopy
raisedCosTexture2 = visual.filters.makeMask(heightPix, shape='raisedCosine', fringeWidth=0.6, radius=[1.0, 1.0])
invRaisedCosTexture = -raisedCosTexture2  # inverts mask to blur edges instead of center
slab_width = 20

blankslab = np.ones((heightPix, slab_width))  # create blank slabs to put to left and right of image
mmask = np.append(blankslab, invRaisedCosTexture, axis=1)  # append blank slab to left
mmask = np.append(mmask, blankslab, axis=1)  # and right
# changed edge_mask color from grey, fades to black round edges which makes screen edges less visible
edge_mask = visual.GratingStim(win, mask=mmask, tex=None, contrast=1.0,
                               size=(widthPix, heightPix), units='pix',
                               color='black', colorSpace=this_colourSpace)


'''FLOW DOT SETTINGS'''
if background == 'flow_dots':
    # # # # flow dots settings
    # fustrum dimensions (3d shape containing dots).  Plane distances take into accouunt view_dist,
    # so if the viewer is 50ms from screen, and the plane is at 100cm, the plane is 50cm 'behind' the screen.
    near_plane_cm = 107  # later use 107 to match studies (.3?)
    far_plane_cm = 207  # later use 207 to match studies (.3?)

    # frame dimensions (2d shape containing dots on screen, in real-world cm (measure with ruler)).
    # If dots are at a distance greater then view_dist, then they won't fill the frame, or if at a distance less than view_dist, they will extend beyond the frame.
    frame_size_cm = mon_width_cm  # size of square in cm
    '''To give the illusion of distance, all x and y co-ordinates are scaled by the distance of the dot.
    This scaling is done relative to the reference angle
    (e.g., the angle of the screen/frame containing stimuli when it is at z=view_dist, typically 57.3cm).
    The reference angle has a scale factor of 1, and all other distances are scaled relative to this.
    x and y values are scaled by multiplying them by the scale factor.
    '''
    ref_angle = find_angle(adjacent=view_dist_cm, opposite=frame_size_cm)
    print(f"ref_angle: {ref_angle}")


    # bg_motion speed in cm/s
    flow_speed_cm_p_sec = 150  # 1.2m/sec matches previous flow parsing study (Evans et al. 2020)
    flow_speed_cm_p_fr = flow_speed_cm_p_sec / fps  # 1.66 cm per frame = 1m per second


    # initialise dots - for 1 per sq cm, divide by 2 because make_xy_spokes doubles the density
    dots_per_sq_cm = 1 / 2
    n_dots = int(dots_per_sq_cm * mon_width_cm * mon_height_cm)
    if debug:
        print(f"n_dots: {n_dots}")


    flow_dots = visual.ElementArrayStim(win, elementTex=None, elementMask='circle',
                                        units='cm', nElements=n_dots, sizes=.2,
                                        colorSpace=this_colourSpace,
                                        colors=flow_colour)

    # initialize x and y positions of dots to fit in window (frame_size_cm) at distance 0
    x_array = np.random.uniform(-frame_size_cm/2, frame_size_cm/2, n_dots)  # x values in cm
    y_array = np.random.uniform(-frame_size_cm/2, frame_size_cm/2, n_dots)  # y values in cm

    # initialize z values (distance/distance from viewer) in cm
    z_array = np.random.uniform(low=near_plane_cm, high=far_plane_cm, size=n_dots)    # distances in cm

    # convert x and y into spokes
    x_array, y_array = make_xy_spokes(x_array, y_array)

    # get starting distances and scale xys
    dots_pos_array = scaled_dots_pos_array(x_array, y_array, z_array, frame_size_cm, ref_angle)
    flow_dots.xys = dots_pos_array

    # dot lifetime ms
    dot_life_max_ms = 666  # Simon says use longer dot life than on original exp which used 166.67
    dot_life_max_fr = int(dot_life_max_ms / 1000 * fps)
    print(f"dot_life_max_fr: {dot_life_max_fr}")

    # initialize lifetime of each dot (in frames)
    dot_lifetime_array = np.random.randint(0, dot_life_max_fr, n_dots)

    # when dots are redrawn with a new z value, they should be at least this far away the boundary
    # otherwise they might have to be re-drawn after a couple of frames, which could lead to flickering.
    # this is the max z_distance in meters they can travel in n frames
    max_z_cm_in_life = flow_speed_cm_p_fr * dot_life_max_fr
    print(f"max_z_cm_in_life: {max_z_cm_in_life}")

    if max_z_cm_in_life > (far_plane_cm - near_plane_cm):
        print(f"max_z_cm_in_life ({max_z_cm_in_life}) is greater than the distance between the near and far planes ({far_plane_cm - near_plane_cm}).")
        max_possible_dot_life_fr = (far_plane_cm - near_plane_cm) / flow_speed_cm_p_fr
        max_possible_dot_life_ms = max_possible_dot_life_fr / fps * 1000
        print(f"max_possible_dot_life_ms: {max_possible_dot_life_ms}")
        raise ValueError(f"dot_life_max_ms ({dot_life_max_ms}) is set too high, dots will travel the full distance in "
                         f"max_possible_dot_life_ms ({max_possible_dot_life_ms}), please select a lower value.  ")

else:  # if background  == 'no_bg'
    flow_dots = None
    flow_speed_cm_p_sec = None
    flow_speed_cm_p_fr = None
    n_dots = None
    dot_life_max_ms = None



# # # TIMINGS - expected frame duration and tolerance # # #
expected_fr_sec = 1 / fps
expected_fr_ms = expected_fr_sec * 1000
frame_tolerance_prop = 1 / expected_fr_ms  # frame_tolerance_ms == 1ms, regardless of fps.
max_fr_dur_sec = expected_fr_sec + (expected_fr_sec * frame_tolerance_prop)
min_fr_dur_sec = expected_fr_sec - (expected_fr_sec * frame_tolerance_prop)
frame_tolerance_ms = (max_fr_dur_sec - expected_fr_sec) * 1000
max_dropped_fr_trials = 10  # number of trials with dropped frames to allow before experiment is aborted
if debug:
    print(f"\nexpected_fr_ms: {expected_fr_ms}")
    print(f"frame_tolerance_prop: {frame_tolerance_prop}")
    print(f"frame_tolerance_ms: {frame_tolerance_ms}")
    print(f"max_dropped_fr_trials: {max_dropped_fr_trials}")


# # # ACCURACY # # #
'''If accuracy is bad after first n trials, suggest updating starting distance'''
resp_corr_list = []  # accuracy feedback during breaks
check_start_acc_after = n_stairs  # check accuracy after 10 trials.
initial_acc_thresh = .7  # initial accuracy threshold from first n trials to continue experiment


# empty variable to store recorded frame durations
fr_int_per_trial = []  # nested list of frame durations for each trial (y values)
recorded_fr_counter = 0  # how many frames have been recorded
fr_counter_per_trial = []  # nested list of recorded_fr_counter values for plotting frame intervals (x values)
cond_list = []  # stores stair name for each trial, to colour code plot lines and legend
dropped_fr_trial_counter = 0  # counter for how many trials have dropped frames
dropped_fr_trial_x_locs = []  # nested list of [1st fr of dropped fr trial, 1st fr of next trial] for trials with dropped frames


# # # BREAKS  - every n trials # # #
max_trials = total_n_trials + max_dropped_fr_trials  # expected trials plus repeats
max_without_break = 120  # limit on number of trials without a break
n_breaks = max_trials // max_without_break  # number of breaks
if n_breaks > 0:
    take_break = int(max_trials / (n_breaks + 1))
else:
    take_break = max_without_break
break_dur = 30
if debug:
    break_dur = 5
    print(f"\ntake a {break_dur} second break every {take_break} trials ({n_breaks} breaks in total).")


# # # ON-SCREEN MESSAGES # # #
instructions = visual.TextStim(win=win, name='instructions', font='Arial', height=20,
                               color='white', colorSpace=this_colourSpace,
                               wrapWidth=widthPix / 2,
                               text="\n\nFocus on the fixation circle at the centre of the screen.\n\n"
                                    "A small white target will briefly appear on screen,\n"
                                    "press the key related to the location of the probe:\n\n"
                                    "[4]/[Q] top-left\t\t\t[5]/[W] top-right\n\n\n\n"
                                    "[1]/[A] bottom-left\t\t\t[2]/[S] bottom-right.\n\n\n"
                                    "Some targets will be easy to see, others will be hard to spot.\n"
                                    "If you aren't sure, just guess!\n\n"
                                    "Please move the mouse offscreen, then press any key to start")


too_many_dropped_fr = visual.TextStim(win=win, name='too_many_dropped_fr',
                                      text="The experiment had quit as the computer is dropping frames.\n"
                                           "Sorry for the inconvenience.\n"
                                           "Please contact the experimenter.\n\n"
                                           "Press any key to return to the desktop.",
                                      font='Arial', height=20, colorSpace=this_colourSpace)

break_text = f"Turn on the light and take at least {break_dur} seconds break.\n" \
             "Keep focussed on the fixation circle in the middle of the screen.\n" \
             "Remember, if you don't see the target, just guess!"
breaks = visual.TextStim(win=win, name='breaks', text=break_text, font='Arial',
                         pos=[0, 0], height=20, ori=0, color='white',
                         colorSpace=this_colourSpace)

acc_warning_text = "The experiment had quit as your score is low.\n\n" \
                   "The first few trials should be relatively easy, this doesn't seem to the be the case.\n\n" \
                   f"Please try again.  If you see this message repeatedly, contact the experimentor.\n\n"

acc_warning = visual.TextStim(win=win, name='acc_warning', text=acc_warning_text, font='Arial',
                         pos=[0, 0], height=20, ori=0, color='white',
                         colorSpace=this_colourSpace)

end_of_exp_text = f"You have completed {run_number}/12 runs of this experiment.\nThank you for your time."
end_of_exp = visual.TextStim(win=win, name='end_of_exp',
                             text=end_of_exp_text, color='white',
                             font='Arial', height=20, colorSpace=this_colourSpace)


# # # PRIORITY to increase speed # # #
# # turn on high priority here. (and turn off garbage collection)
gc.disable()
core.rush(True)
if monitor_name in ['Nick_work_laptop', 'OLED']:  # e.g., windows machine
    core.rush(True, realtime=True)



# # # CONSTRUCT STAIRCASES # # #
start_lum_prop = .6  # proportion of maxLum to start at, e.g., .6 = 60% of maxLum
if debug:
    start_lum_prop = 1.0
stairStart = maxLum * start_lum_prop

# proportion of stairStart to use to get intial step size, but this is NOT the actual step size
c_multiplier = 1.2  # previously was 0.6 until 01/11/2023

k_type = 'simple'  # step size changes after each reversal only
if c_multiplier > 1.1:
    k_type = 'accel'  # step size changes after first two trials, then after each reversal


stairs = []
for stair_idx in range(n_stairs):
    thisInfo = copy(expInfo)
    thisInfo['stair_idx'] = stair_idx
    thisInfo['ISI_ms'] = isi_ms_conds_list[stair_idx]
    thisInfo['ISI_fr'] = isi_fr_conds_list[stair_idx]
    thisInfo['sep'] = separation  # sep_conds_list[stair_idx]
    thisInfo['cong_val'] = cong_val_conds_list[stair_idx]
    thisInfo['cong_name'] = cong_name_conds_list[stair_idx]


    thisStair = Staircase(name=stair_names_list[stair_idx],
                          type=k_type,
                          value=stairStart,
                          C=stairStart * c_multiplier,  # used to calculate initial step size, as prop of stairStart
                          minRevs=3,
                          minTrials=n_trials_per_stair,
                          minVal=bgLum,
                          maxVal=maxLum,
                          targetThresh=0.75,
                          extraInfo=thisInfo)
    stairs.append(thisStair)


# # # SHOW INSTRUCTIONS # # #
while not event.getKeys():
    fixation.draw()
    instructions.draw()
    win.flip()


# # # INITIALIZE COUNTERS # # #
trial_num_inc_repeats = 0  # number of trials including repeated trials
trial_number = 0  # the number of the trial for the output file


# # # RUN EXPERIMENT # # #
for step in range(n_trials_per_stair):
    np.random.shuffle(stairs)  # shuffle order for each step (e.g., shuffle, run all stairs, shuffle again etc)
    for thisStair in stairs:

        # # # PER-TRIAL VARIABLES # # #

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

            # conditions (ISI, congruence, sep, prelim)
            # sep = thisStair.extraInfo['sep']  # separation is already set at top of script
            sep = separation
            isi_ms = thisStair.extraInfo['ISI_ms']
            isi_cond_fr = thisStair.extraInfo['ISI_fr']  # shows number of frames or -1 for concurrent
            congruent = thisStair.extraInfo['cong_val']
            cong_name = thisStair.extraInfo['cong_name']
            if debug:
                print(f"isi_ms: {isi_ms}, isi_cond_fr: {isi_cond_fr}, congruent: {congruent}, cong_name: {cong_name}")


            # # # SEP COND variables # # #
            # separation expressed as degrees.
            if -1 < sep < 99:
                sep_deg = sep * pixel_mm_deg_dict['diag_deg']
            else:
                sep_deg = None

            # negative separation for comparing conditions (e.g., cong sep = 5, incong sep = -5.
            if cong_name == 'incong':
                neg_sep = 0 - sep
                neg_sep_deg = 0 - sep_deg
                if sep == 0:
                    neg_sep = -.1
                    neg_sep_deg = -pixel_mm_deg_dict['diag_deg'] / 10
            else:
                neg_sep = sep
                neg_sep_deg = sep_deg
            if debug:
                print(f"sep: {sep}, neg_sep: {neg_sep}; sep_deg: {sep_deg}, neg_sep_deg: {neg_sep_deg}")



            # # # GET BACKGROUND ATTRIBUTES # # #

            # use congruence to determine the flow direction and target jump direction
            # 1 is contracting/inward/backwards, -1 is expanding/outward/forwards
            flow_dir = np.random.choice([1, -1])
            flow_name = 'cont'
            if flow_dir == -1:
                flow_name = 'exp'

            target_jump = congruent * flow_dir
            if debug:
                print(f"flow_dir: {flow_dir}, flow_name: {flow_name}, target_jump: {target_jump}")

            if background == 'flow_dots':
                # boundaries for z position (distance from screen) during radial flow
                if flow_dir == -1:  # expanding
                    z_start_bounds = [near_plane_cm + max_z_cm_in_life, far_plane_cm]
                else:  # contracting, flow_dir == 1
                    z_start_bounds = [near_plane_cm, far_plane_cm - max_z_cm_in_life]
                if debug:
                    print(f"z_start_bounds: {z_start_bounds}")

            # vary fixation polarity to reduce risk of screen burn.
            if trial_num_inc_repeats % 2 == 0:
                fixation.lineColor = 'grey'
                fixation.fillColor = 'black'
            else:
                fixation.lineColor = 'black'
                fixation.fillColor = 'grey'

            # reset fixation radius - reduces in size at response segment of each trial
            fixation.setRadius(3)


            # # # GET PROBE ATTRIBUTES # # #
            # Luminance (staircase varies probeLum)
            probeLum = thisStair.next()
            probeColor1 = probeLum / maxLum
            this_probeColor = probeColor1
            probe1.fillColor = [this_probeColor, this_probeColor, this_probeColor]
            probe2.fillColor = [this_probeColor, this_probeColor, this_probeColor]
            if debug:
                print(f"probeLum: {probeLum}, this_probeColor: {this_probeColor}, "
                      f"probeColor1: {probeColor1}")

            # PROBE LOCATION - 45=top-right, 135=top-left, 225=bottom-left, 315=bottom-right
            corner = np.random.choice([45, 135, 225, 315])

            # PROBE POSITION (including shift around dist_from_fix)
            probe_pos_dict = get_probe_pos_dict(sep, target_jump, corner, dist_from_fix,
                                                probe_size=probe_size,
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


            # # # GET TIMINGS in frames # # #
            # # # MOTION WINDOW # # #
            '''Rather than just having preliminary motion (before probe1), we now have a background motion (bg_motion) window of a fixed duration.
            The stimuli (probe1, ISI, probe2) occur in the middle of this window.
            e.g., if the window is 10 frames, and the ISI is 4 frames, then the total stimulus duration is 8 frames,
            so there will be one frame of bg_motion before probe1 and one frame of bg_motion after probe2.
            if the window was 100 frames then there would be 46 frames of bg_motion before probe1 and 46 frames after probe2.'''

            # get number of frames for bg_motion duration
            bg_motion_fr = int(selected_bg_motion_ms * fps / 1000)
            bg_motion_ms = bg_motion_fr * 1000 / fps
            if debug:
                print(f'\nselected_bg_motion_ms: {selected_bg_motion_ms}')
                print(f'bg_motion_fr: {bg_motion_fr}')
                print(f'bg_motion_ms: {bg_motion_ms}')

            # Get the number of frames for probes and ISI
            # If probes are presented concurrently, set isi_dur_fr and p2_dur_fr to last for 0 frames.
            isi_dur_fr = isi_cond_fr
            p1_dur_fr = p2_dur_fr = probe_duration
            if isi_cond_fr < 0:
                isi_dur_fr = p2_dur_fr = 0

            # get number of frames for the total stimulus duration
            stim_dur_fr = p1_dur_fr + isi_dur_fr + p2_dur_fr
            if isi_dur_fr == -1:
                stim_dur_fr = p1_dur_fr

            # get duration of preliminary bg_motion (before probe1) and post-probe2 bg_motion
            # if these number are not equal, prelim should be 1 frame longer than post
            pre_and_post_fr = int(bg_motion_fr - stim_dur_fr)  # remaining frames not including stim_dur_fr
            post_dur_fr = int(pre_and_post_fr / 2)  # number of frames after stim_dur_fr
            if pre_and_post_fr % 2 == 0:  # if there is an odd number of frames, make prelim one frame longer than post
                prelim_dur_fr = post_dur_fr
            else:
                prelim_dur_fr = post_dur_fr + 1


            ''''''

            # variable fixation time
            '''to reduce anticipatory effects that might arise from fixation always being same length.
            if False, vary_fix == .5 seconds, so end_fix_fr is 1 second.
            if Ture, vary_fix is between 0 and 1 second, so end_fix_fr is between .5 and 1.5 seconds.'''
            vary_fix = int(fps / 2)
            if vary_fixation:
                vary_fix = np.random.randint(0, fps)



            # cumulative timing in frames for each segment of a trial
            end_fix_fr = int(fps / 2) + vary_fix - prelim_dur_fr
            if end_fix_fr < 0:
                end_fix_fr = int(fps / 2)
            end_prelim_fr = end_fix_fr + prelim_dur_fr
            end_p1_fr = end_prelim_fr + probe_duration
            end_ISI_fr = end_p1_fr + isi_dur_fr
            end_p2_fr = end_ISI_fr + p2_dur_fr
            end_post_fr = end_p2_fr + post_dur_fr
            if debug:
                print(f"end_fix_fr: {end_fix_fr}, end_prelim_fr: {end_prelim_fr}, end_p1_fr: {end_p1_fr}, \n"
                      f"end_ISI_fr: {end_ISI_fr}, end_p2_fr: {end_p2_fr}, end_post_fr: {end_post_fr}\n")


            # # # SHOW BREAKS SCREEN EVERY N TRIALS # # #
            if (trial_num_inc_repeats % take_break == 1) & (trial_num_inc_repeats > 1):
                if debug:
                    print("\nTaking a break.\n")

                prop_correct = np.mean(resp_corr_list)
                breaks.text = break_text + (f"\n{trial_number - 1}/{total_n_trials} trials completed.\n"
                                            f"{prop_correct * 100:.2f}% correct.\n\n")
                breaks.draw()
                win.flip()
                event.clearEvents(eventType='keyboard')
                # # turn off high priority mode and turn garbage collection back on
                gc.enable()
                core.rush(False)
                core.wait(secs=break_dur)  # enforced 30-second break
                # # turn on high priority here. (and turn off garbage collection)
                gc.disable()
                core.rush(True)
                if monitor_name in ['Nick_work_laptop', 'OLED']:  # e.g., windows machine
                    core.rush(True, realtime=True)
                event.clearEvents(eventType='keyboard')
                breaks.text = break_text + "\n\nPress any key to continue."
                breaks.draw()
                win.flip()
                while not event.getKeys():
                    # continue the breaks routine until a key is pressed
                    continueRoutine = True
            else:
                # else continue the trial routine (per frame section).
                continueRoutine = True


            # # # PER_FRAME SEGMENTS # # #
            frameN = -1
            # # continueRoutine here runs the per-frame section of the trial
            while continueRoutine:
                frameN = frameN + 1

                # # # RECORD FRAME DURATIONS # # #
                # Turn recording on and off from just before probe1 til just after probe2.
                if frameN == end_prelim_fr:
                    if record_fr_durs:  # start recording frames just before probe1 presentation
                        win.recordFrameIntervals = True

                    # clear any previous key presses
                    event.clearEvents(eventType='keyboard')
                    theseKeys = []

                    # reset timer to start with probe1 presentation.
                    resp.clock.reset()


                # stop recording frame intervals AFTER PROBE 2
                elif frameN == end_p2_fr + 1:
                    if record_fr_durs:
                        win.recordFrameIntervals = False


                # # # FIXATION until end of fixation interval # # #
                if end_fix_fr >= frameN > 0:
                    if background == 'flow_dots':

                        '''just have incoherent bg_motion from re-spawning dots, z bounds as full z range'''
                        # 1. don't update z values
                        # 2. check if any z values are out of bounds (too close when expanding or too far when contracting),
                        # if so, set their dot life to max, so they are given new x, y and z values by update_dotlife() below.
                        dot_lifetime_array = check_z_start_bounds(z_array, near_plane_cm, far_plane_cm, dot_life_max_fr,
                                                                  dot_lifetime_array, flow_dir)

                        # 3. update dot lifetime, give new x, y, z coords to dots whose lifetime is max.
                        dotlife_array, x_array, y_array, z_array = update_dotlife(dotlife_array=dot_lifetime_array,
                                                                                  dot_max_fr=dot_life_max_fr,
                                                                                  x_array=x_array, y_array=y_array,
                                                                                  z_array=z_array,
                                                                                  x_bounds=frame_size_cm / 2,
                                                                                  y_bounds=frame_size_cm / 2,
                                                                                  z_start_bounds=[near_plane_cm,
                                                                                                  far_plane_cm])

                        # 4. put new x and y values into spokes
                        x_array, y_array = make_xy_spokes(x_array, y_array)

                        # 5. scale x and y positions by distance
                        dots_pos_array = scaled_dots_pos_array(x_array, y_array, z_array, frame_size_cm, ref_angle)
                        flow_dots.xys = dots_pos_array
                        flow_dots.draw()

                    edge_mask.draw()
                    fix_mask.draw()
                    fixation.draw()


                # # # PRELIM BACKGROUND MOTION prior to probe1 - after fixation, but before probe 1 # # #
                elif end_prelim_fr >= frameN > end_fix_fr:
                    if background == 'flow_dots':

                        # 1. Update z (distance values): Add dots_speed * flow_dir to the current z values.
                        z_array = z_array + flow_speed_cm_p_fr * flow_dir

                        # 2. check if any z values are out of bounds (too close when expanding or too far when contracting),
                        # if so, set their dot life to max, so they are given new x, y and z values by update_dotlife() below.
                        dot_lifetime_array = check_z_start_bounds(z_array, near_plane_cm, far_plane_cm, dot_life_max_fr,
                                                                  dot_lifetime_array, flow_dir)

                        # 3. update dot lifetime, give new x, y, z coords to dots whose lifetime is max.
                        dotlife_array, x_array, y_array, z_array = update_dotlife(dotlife_array=dot_lifetime_array,
                                                                                  dot_max_fr=dot_life_max_fr,
                                                                                  x_array=x_array, y_array=y_array,
                                                                                  z_array=z_array,
                                                                                  x_bounds=frame_size_cm / 2,
                                                                                  y_bounds=frame_size_cm / 2,
                                                                                  z_start_bounds=z_start_bounds)
                        # 4. put new x and y values into spokes
                        x_array, y_array = make_xy_spokes(x_array, y_array)

                        # 5. scale x and y positions by distance
                        dots_pos_array = scaled_dots_pos_array(x_array, y_array, z_array, frame_size_cm, ref_angle)
                        flow_dots.xys = dots_pos_array
                        flow_dots.draw()

                    edge_mask.draw()

                    fix_mask.draw()
                    fixation.draw()

                # # # PROBE 1 - after prelim bg bg_motion, before ISI # # #
                elif end_p1_fr >= frameN > end_prelim_fr:
                    if background == 'flow_dots':

                        # 1. Update z (distance values): Add dots_speed * flow_dir to the current z values.
                        z_array = z_array + flow_speed_cm_p_fr * flow_dir

                        # 2. check if any z values are out of bounds (too close when expanding or too far when contracting),
                        # if so, set their dot life to max, so they are given new x, y and z values by update_dotlife() below.
                        dot_lifetime_array = check_z_start_bounds(z_array, near_plane_cm, far_plane_cm, dot_life_max_fr,
                                                                  dot_lifetime_array, flow_dir)

                        # 3. update dot lifetime, give new x, y, z coords to dots whose lifetime is max.
                        dotlife_array, x_array, y_array, z_array = update_dotlife(dotlife_array=dot_lifetime_array,
                                                                                  dot_max_fr=dot_life_max_fr,
                                                                                  x_array=x_array, y_array=y_array,
                                                                                  z_array=z_array,
                                                                                  x_bounds=frame_size_cm / 2,
                                                                                  y_bounds=frame_size_cm / 2,
                                                                                  z_start_bounds=z_start_bounds)
                        # 4. put new x and y values into spokes
                        x_array, y_array = make_xy_spokes(x_array, y_array)

                        # 5. scale x and y positions by distance
                        dots_pos_array = scaled_dots_pos_array(x_array, y_array, z_array, frame_size_cm, ref_angle)
                        flow_dots.xys = dots_pos_array
                        flow_dots.draw()

                    edge_mask.draw()
                    fix_mask.draw()

                    fixation.draw()

                    probe1.draw()
                    # SIMULTANEOUS CONDITION
                    if isi_cond_fr == -1:
                        if sep <= 18:
                            probe2.draw()



                # # # ISI - after probe 1, before probe 2 (or nothing if isi_cond_fr < 1) # # #
                elif end_ISI_fr >= frameN > end_p1_fr:
                    if background == 'flow_dots':

                        # 1. Update z (distance values): Add dots_speed * flow_dir to the current z values.
                        z_array = z_array + flow_speed_cm_p_fr * flow_dir

                        # 2. check if any z values are out of bounds (too close when expanding or too far when contracting),
                        # if so, set their dot life to max, so they are given new x, y and z values by update_dotlife() below.
                        dot_lifetime_array = check_z_start_bounds(z_array, near_plane_cm, far_plane_cm, dot_life_max_fr,
                                                                  dot_lifetime_array, flow_dir)

                        # 3. update dot lifetime, give new x, y, z coords to dots whose lifetime is max.
                        dotlife_array, x_array, y_array, z_array = update_dotlife(dotlife_array=dot_lifetime_array,
                                                                                  dot_max_fr=dot_life_max_fr,
                                                                                  x_array=x_array, y_array=y_array,
                                                                                  z_array=z_array,
                                                                                  x_bounds=frame_size_cm / 2,
                                                                                  y_bounds=frame_size_cm / 2,
                                                                                  z_start_bounds=z_start_bounds)
                        # 4. put new x and y values into spokes
                        x_array, y_array = make_xy_spokes(x_array, y_array)

                        # 5. scale x and y positions by distance
                        dots_pos_array = scaled_dots_pos_array(x_array, y_array, z_array, frame_size_cm, ref_angle)
                        flow_dots.xys = dots_pos_array
                        flow_dots.draw()

                    edge_mask.draw()
                    fix_mask.draw()
                    fixation.draw()

                # # # PROBE 2 - after ISI before response segment (unless isi_cond_fr < 1) # # #
                elif end_p2_fr >= frameN > end_ISI_fr:
                    if background == 'flow_dots':

                        # 1. Update z (distance values): Add dots_speed * flow_dir to the current z values.
                        z_array = z_array + flow_speed_cm_p_fr * flow_dir

                        # 2. check if any z values are out of bounds (too close when expanding or too far when contracting),
                        # if so, set their dot life to max, so they are given new x, y and z values by update_dotlife() below.
                        dot_lifetime_array = check_z_start_bounds(z_array, near_plane_cm, far_plane_cm, dot_life_max_fr,
                                                                  dot_lifetime_array, flow_dir)

                        # 3. update dot lifetime, give new x, y, z coords to dots whose lifetime is max.
                        dotlife_array, x_array, y_array, z_array = update_dotlife(dotlife_array=dot_lifetime_array,
                                                                                  dot_max_fr=dot_life_max_fr,
                                                                                  x_array=x_array, y_array=y_array,
                                                                                  z_array=z_array,
                                                                                  x_bounds=frame_size_cm / 2,
                                                                                  y_bounds=frame_size_cm / 2,
                                                                                  z_start_bounds=z_start_bounds)

                        # 4. put new x and y values into spokes
                        x_array, y_array = make_xy_spokes(x_array, y_array)

                        # 5. scale x and y positions by distance
                        dots_pos_array = scaled_dots_pos_array(x_array, y_array, z_array, frame_size_cm, ref_angle)
                        flow_dots.xys = dots_pos_array
                        flow_dots.draw()

                    edge_mask.draw()
                    fix_mask.draw()
                    fixation.draw()
                    if isi_cond_fr >= 0:  # if not concurrent condition (ISI=-1)
                        if sep != 99:  # If not 1probe condition (sep = 99)
                            probe2.draw()


                # # # POST STIMULUS MOTION - after probe 2 (unless isi_cond_fr < 1) # # #
                elif end_post_fr >= frameN > end_p2_fr:
                    if background == 'flow_dots':
                        # 1. Update z (distance values): Add dots_speed * flow_dir to the current z values.
                        z_array = z_array + flow_speed_cm_p_fr * flow_dir

                        # 2. check if any z values are out of bounds (too close when expanding or too far when contracting),
                        # if so, set their dot life to max, so they are given new x, y and z values by update_dotlife() below.
                        dot_lifetime_array = check_z_start_bounds(z_array, near_plane_cm, far_plane_cm,
                                                                  dot_life_max_fr,
                                                                  dot_lifetime_array, flow_dir)

                        # 3. update dot lifetime, give new x, y, z coords to dots whose lifetime is max.
                        dotlife_array, x_array, y_array, z_array = update_dotlife(
                            dotlife_array=dot_lifetime_array,
                            dot_max_fr=dot_life_max_fr,
                            x_array=x_array, y_array=y_array,
                            z_array=z_array,
                            x_bounds=frame_size_cm / 2,
                            y_bounds=frame_size_cm / 2,
                            z_start_bounds=z_start_bounds)

                        # 4. put new x and y values into spokes
                        x_array, y_array = make_xy_spokes(x_array, y_array)

                        # 5. scale x and y positions by distance
                        dots_pos_array = scaled_dots_pos_array(x_array, y_array, z_array, frame_size_cm,
                                                               ref_angle)
                        flow_dots.xys = dots_pos_array
                        flow_dots.draw()

                    edge_mask.draw()

                    fix_mask.draw()
                    fixation.draw()


                # # # ANSWER - after post_stim-bg_motion, before end of trial # # #
                elif frameN > end_post_fr:
                    if background == 'flow_dots':

                        '''just have incoherent bg_motion from re-spawning dots, z bounds as full z range'''
                        # 1. don't update z values
                        # 2. check if any z values are out of bounds (too close when expanding or too far when contracting),
                        # if so, set their dot life to max, so they are given new x, y and z values by update_dotlife() below.
                        dot_lifetime_array = check_z_start_bounds(z_array, near_plane_cm, far_plane_cm, dot_life_max_fr,
                                                                  dot_lifetime_array, flow_dir)

                        # 3. update dot lifetime, give new x, y, z coords to dots whose lifetime is max.
                        dotlife_array, x_array, y_array, z_array = update_dotlife(dotlife_array=dot_lifetime_array,
                                                                                  dot_max_fr=dot_life_max_fr,
                                                                                  x_array=x_array, y_array=y_array,
                                                                                  z_array=z_array,
                                                                                  x_bounds=frame_size_cm / 2,
                                                                                  y_bounds=frame_size_cm / 2,
                                                                                  z_start_bounds=[near_plane_cm, far_plane_cm])

                        # 4. put new x and y values into spokes
                        x_array, y_array = make_xy_spokes(x_array, y_array)

                        # 5. scale x and y positions by distance
                        dots_pos_array = scaled_dots_pos_array(x_array, y_array, z_array, frame_size_cm, ref_angle)
                        flow_dots.xys = dots_pos_array
                        flow_dots.draw()

                    edge_mask.draw()
                    fix_mask.draw()
                    fixation.setRadius(2)
                    fixation.draw()

                    # RESPONSE HANDLING
                    theseKeys = event.getKeys(keyList=['num_5', 'num_4', 'num_1', 'num_2', 'w', 'q', 'a', 's'])
                    if len(theseKeys) > 0:  # at least one key was pressed
                        resp.keys = theseKeys[-1]  # just the last key pressed
                        resp.rt = resp.clock.getTime()

                        # a response ends the per-frame_section
                        continueRoutine = False


                # # # REGARDLESS OF FRAME NUMBER # # #
                # check for quit
                if event.getKeys(keyList=["escape"]):
                    thisExp.close()
                    core.quit()

                # refresh the screen
                if continueRoutine:
                    win.flip()


            # # # END OF PER-FRAME SECTION if continueRoutine = False # # #
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
            resp_corr_list.append(resp.corr)


            # # # SORT FRAME INTERVALS TO USE FOR PLOTS LATER # # #
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
                        print(f"\n\toh no! A frame had bad timing! trial: {trial_number}, {thisStair.name} "
                              f"{round(max(trial_fr_intervals), 3)} > {round(max_fr_dur_sec, 2)} or "
                              f"{round(min(trial_fr_intervals), 3)} < {round(min_fr_dur_sec, 2)}")

                    print(f"Timing bad, repeating trial {trial_number}. "
                          f"repeated: {dropped_fr_trial_counter} / {max_dropped_fr_trials}")

                    # decrement trial and stair so that the correct values are used for the next trial
                    trial_number -= 1
                    thisStair.trialCount = thisStair.trialCount - 1  # so Kesten doesn't count this trial

                    # remove last response from resp_corr_list
                    resp_corr_list.pop()

                    # get first and last frame numbers for this trial
                    trial_x_locs = [fr_counter_per_trial[-1][0],
                                    fr_counter_per_trial[-1][-1] + 1]  # 1st fr of this trial to 1st of next trial
                    dropped_fr_trial_x_locs.append(trial_x_locs)
                    dropped_fr_trial_counter += 1
                    continue
                else:
                    repeat = False  # breaks out of while repeat=True loop to progress to new trial


            # # # TRIAL COMPLETED # # #
            # If too many trials have had dropped frames, quit experiment
            if dropped_fr_trial_counter > max_dropped_fr_trials:
                event.clearEvents(eventType='keyboard')
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
                                participant=participant_ID, run_num=run_number,
                                save_path=save_dir, incomplete=True)


                    # close and quit once a key is pressed
                    thisExp.close()
                    win.close()
                    core.quit()


        # # # ADD TRIAL INFO TO OUTPUT CSV # # #
        thisExp.addData('trial_number', trial_number)
        thisExp.addData('trial_n_inc_rpt', trial_num_inc_repeats)
        thisExp.addData('stair', stair_idx)
        thisExp.addData('stair_name', thisStair)
        thisExp.addData('step', step)
        thisExp.addData('separation', sep)
        thisExp.addData('sep_deg', sep_deg)
        thisExp.addData('neg_sep', neg_sep)
        thisExp.addData('neg_sep_deg', neg_sep_deg)
        thisExp.addData('isi_ms', isi_ms)
        thisExp.addData('isi_cond_fr', isi_cond_fr)
        thisExp.addData('isi_dur_fr', isi_dur_fr)
        thisExp.addData('selected_bg_motion_ms', selected_bg_motion_ms)
        thisExp.addData('bg_motion_fr', bg_motion_fr)
        thisExp.addData('bg_motion_ms', bg_motion_ms)
        thisExp.addData('prelim_dur_fr', prelim_dur_fr)
        thisExp.addData('post_dur_fr', post_dur_fr)
        thisExp.addData('congruent', congruent)
        thisExp.addData('flow_dir', flow_dir)
        thisExp.addData('flow_name', flow_name)
        thisExp.addData('probe_jump', target_jump)
        thisExp.addData('corner', corner)
        thisExp.addData('probeLum', probeLum)
        thisExp.addData('probeColor1', probeColor1)
        thisExp.addData('resp_keys', resp.keys)
        thisExp.addData('resp_corr', resp.corr)
        thisExp.addData('resp_rt', resp.rt)
        thisExp.addData('flow_speed_cm_p_sec', flow_speed_cm_p_sec)
        thisExp.addData('flow_speed_cm_p_fr', flow_speed_cm_p_fr)
        thisExp.addData('n_dots', n_dots)
        thisExp.addData('dot_life_max_ms', dot_life_max_ms)
        thisExp.addData('probe_duration', probe_duration)
        thisExp.addData('probe_ecc', probe_ecc)
        thisExp.addData('orientation', orientation)
        thisExp.addData('background', background)
        thisExp.addData('vary_fix', vary_fix)
        thisExp.addData('end_fix_fr', end_fix_fr)
        thisExp.addData('monitor_name', monitor_name)
        thisExp.addData('this_colourSpace', this_colourSpace)
        thisExp.addData('bgColor_rgb1', bgColor_rgb1)
        thisExp.addData('selected_fps', fps)
        thisExp.addData('frame_tolerance_prop', frame_tolerance_prop)
        thisExp.addData('expName', expName)
        thisExp.addData('psychopy_version', psychopy_version)
        thisExp.addData('participant_ID', participant_ID)
        thisExp.addData('run_number', run_number)
        thisExp.addData('date', expInfo['date'])
        thisExp.addData('time', expInfo['time'])

        # tell psychopy to move to next trial
        thisExp.nextEntry()

        # update staircase based on whether response was correct or incorrect
        thisStair.newValue(resp.corr)


# # # END OF EXPERIMENT # # #
# now exp is completed, save as '_output' rather than '_incomplete'
thisExp.dataFileName = path.join(save_dir, f'{p_name_run}_output')
thisExp.close()
print(f"\nend of experiment loop, saving data to:\n{thisExp.dataFileName}\n")


# # # PLOT FRAME INTERVALS # # #
if record_fr_durs:
    print(f"{dropped_fr_trial_counter}/{trial_num_inc_repeats} trials with bad timing "
          f"(expected: {round(expected_fr_ms, 2)}ms, "
          f"frame_tolerance_ms: +/- {round(frame_tolerance_ms, 2)})")

    plt_fr_ints(time_p_trial_nested_list=fr_int_per_trial, n_trials_w_dropped_fr=dropped_fr_trial_counter,
                expected_fr_dur_ms=expected_fr_ms, allowed_err_ms=frame_tolerance_ms,
                all_cond_name_list=cond_list, fr_nums_p_trial=fr_counter_per_trial,
                dropped_trial_x_locs=dropped_fr_trial_x_locs,
                mon_name=monitor_name, date=expInfo['date'], frame_rate=fps,
                participant=participant_ID, run_num=run_number,
                save_path=save_dir, incomplete=False)


# # # CLOSING PSYCHOPY # # #
# # turn off high priority mode and turn garbage collection back on
gc.enable()
core.rush(False)


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
