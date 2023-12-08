from __future__ import division
from psychopy import gui, visual, core, data, event, monitors, logging
from psychopy import __version__ as psychopy_version
from psychopy.tools.monitorunittools import cm2pix, pix2cm, deg2pix, pix2deg, convertToPix
from os import path, chdir
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from numpy import array, random, where, sum, linspace, pi, rad2deg, arctan, arctan2, cos, sin, hypot
import os
import copy
import seaborn as sns
import matplotlib.pyplot as plt

'''
This script is to work out how to match the optic flow settings from Simons flow parsing studies.
Dots appear to move from 207 to 107 cm away at 100cm per second (with a lifetime of 160ms).
This is based on a distance of 57.3cm from the screen.  
I've not added a short dot lifetime yet, does it need it?
'''



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
    if flow_dir == -1:  # expanding - z values reduce
        # dot_life_array = np.where(z_array > furthest_z, max_dot_life_fr, dot_life_array)
        dot_life_array = np.where(z_array < closest_z, max_dot_life_fr, dot_life_array)

    elif flow_dir == 1:  # contracting - z values increase
        # dot_life_array = np.where(z_array < closest_z, max_dot_life_fr, dot_life_array)
        dot_life_array = np.where(z_array > furthest_z, max_dot_life_fr, dot_life_array)

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
    :param x_array: np.array of x positions of dots (in cm).
    :param y_array: np.array of y positions of dots (in cm).
    :param z_array: np.array of z positions of dots (in cm).

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
    x_array[replace_mask] = np.random.uniform(low=-x_bounds, high=x_bounds, size=np.sum(replace_mask))
    y_array[replace_mask] = np.random.uniform(low=-y_bounds, high=y_bounds, size=np.sum(replace_mask))
    z_array[replace_mask] = np.random.uniform(low=z_start_bounds[0], high=z_start_bounds[1], size=np.sum(replace_mask))

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



# initialize window
# monitor_name = 'HP_24uh'  # Nick_work_laptop, HP_24uh
# fps = 60
monitor_name = 'OLED'  # Nick_work_laptop, HP_24uh
fps = 120  # 60

mon = monitors.Monitor(monitor_name)

# screen size in pixels and cm
widthPix = int(mon.getSizePix()[0])
heightPix = int(mon.getSizePix()[1])
mon_width_cm = mon.getWidth()  # monitor width in cm
view_dist_cm = mon.getDistance()  # viewing distance in cm
view_dist_pix = widthPix / mon_width_cm * view_dist_cm  # used for calculating visual angle (e.g., probe locations at 4dva)
mon_height_cm = mon_width_cm / (widthPix/heightPix)
print(f"(widthPix, heightPix): ({widthPix}, {heightPix})")
print(f"mon_width_cm: {mon_width_cm}")
print(f"view_dist_cm: {view_dist_cm}")


# colour space
maxLum = 106  # 255 RGB
bgLumProp = 0   # use .2 to match exp1 or .45 to match radial_flow_NM_v2.py
bgLum = 0.0 / 1000  # hard coded to zero rather than maxLum * bgLumProp

# colour space
this_colourSpace = 'rgb1'  # values between 0 and 1
bgColor_rgb1 = bgLum / maxLum
this_bgColour = [bgColor_rgb1, bgColor_rgb1, bgColor_rgb1]

# Give dots a pale green colour, which is adj_flow_colour different to the background
adj_flow_colour = .2
flow_colour = [this_bgColour[0], this_bgColour[1] + adj_flow_colour, this_bgColour[2]]

# screen number
display_number = 1  # 0 indexed, 1 for external display, 0 for internal
if monitor_name in ['asus_cal', 'Nick_work_laptop', 'NickMac', 'OLED', 'ASUS_2_13_240Hz']:
    display_number = 0

# # WINDOW SPEC
# win = visual.Window(monitor=mon, size=(widthPix, heightPix), colorSpace=this_colourSpace, color=this_bgColour,
#                     units='pix', screen=display_number, allowGUI=False, fullscr=True, useFBO=False)


view_dist_cm = 43.7
probe_ecc = 4  # probe eccentricity in dva
dva_3 = 3  # 3 dva in cm
dva_5 = 5  # 5 dva in cm

# get dva per pixel
dva_per_pix = pix2deg(1, mon)
print(f"dva_per_pix: {dva_per_pix}")

dist_from_fix = int((np.tan(np.deg2rad(probe_ecc)) * view_dist_pix) / np.sqrt(2))
print(f"dist_from_fix: {dist_from_fix}")
dff_2 = int((np.tan(np.deg2rad(dva_3)) * view_dist_pix) / np.sqrt(2))
print(f"dff_2: {dff_2}")
dff_6 = int((np.tan(np.deg2rad(dva_5)) * view_dist_pix) / np.sqrt(2))


'''FLOW DOT SETTINGS'''
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
# n_dots = 10
print(f"n_dots: {n_dots}")

# flow_dots = visual.ElementArrayStim(win, elementTex=None, elementMask='circle',
#                                     units='cm', nElements=n_dots, sizes=.2,
#                                     colorSpace=this_colourSpace,
#                                     colors=flow_colour)

# initialize x and y positions of dots to fit in window (frame_size_cm) at distance 0
x_array = np.random.uniform(-frame_size_cm/2, frame_size_cm/2, n_dots)  # x values in cm
y_array = np.random.uniform(-frame_size_cm/2, frame_size_cm/2, n_dots)  # y values in cm

# initialize z values (distance/distance from viewer) in cm
z_array = np.random.uniform(low=near_plane_cm, high=far_plane_cm, size=n_dots)    # distances in cm

# convert x and y into spokes
x_array, y_array = make_xy_spokes(x_array, y_array)

# get starting distances and scale xys
dots_pos_array = scaled_dots_pos_array(x_array, y_array, z_array, frame_size_cm, ref_angle)
# flow_dots.xys = dots_pos_array

# dot lifetime ms
dot_life_max_ms = 666  # Simon says use longer dot life than on original exp which used 166.67
dot_life_max_fr = int(dot_life_max_ms / 1000 * fps)
print(f"dot_life_max_fr: {dot_life_max_fr}")

# initialize lifetime of each dot (in frames)
dot_lifetime_array = np.random.randint(0, dot_life_max_fr, n_dots)

# when dots are redrawn with a new z value, they should be at least this far away the boundary
# otherwise they might have to be re-drawn after a couple of frames, which could lead to flickering.
# this is the max z_distance in cm they can travel in n frames
max_z_cm_in_life = flow_speed_cm_p_fr * dot_life_max_fr
print(f"max_z_cm_in_life: {max_z_cm_in_life}")

if max_z_cm_in_life > (far_plane_cm - near_plane_cm):
    print(f"max_z_cm_in_life ({max_z_cm_in_life}) is greater than the distance between the near and far planes ({far_plane_cm - near_plane_cm}).")
    max_possible_dot_life_fr = (far_plane_cm - near_plane_cm) / flow_speed_cm_p_fr
    max_possible_dot_life_ms = max_possible_dot_life_fr / fps * 1000
    print(f"max_possible_dot_life_ms: {max_possible_dot_life_ms}")
    raise ValueError(f"dot_life_max_ms ({dot_life_max_ms}) is set too high, dots will travel the full distance in "
                     f"max_possible_dot_life_ms ({max_possible_dot_life_ms}), please select a lower value.  ")


contracting = True
if contracting:
    flow_dir = 1
else:  # expanding
    flow_dir = -1

if flow_dir == -1:  # expanding - z values reduce
    z_start_bounds = [near_plane_cm + max_z_cm_in_life, far_plane_cm]
else:  # contracting, flow_dir == 1, z values increase
    z_start_bounds = [near_plane_cm, far_plane_cm - max_z_cm_in_life]
print(f"z_start_bounds: {z_start_bounds}")

motion_type = 'flow_motion'  # press 1
motion_ms = 200
motion_fr = int(motion_ms / 1000 * fps)
print(f"motion_ms: {motion_ms}, motion_fr: {motion_fr}")

motion_data = []

dot_idx = list(range(n_dots))

frame_num = 0
# PRESENT STIMULI
# while not event.getKeys(keyList=["escape"]):
while frame_num < motion_fr:
# while frame_num < motion_fr * 2:
#     if frame_num > motion_fr*2:
#         flow_dir = -1
#
#     if flow_dir == -1:  # expanding - z values reduce
#         z_start_bounds = [near_plane_cm + max_z_cm_in_life, far_plane_cm]
#     else:  # contracting, flow_dir == 1, z values increase
#         z_start_bounds = [near_plane_cm, far_plane_cm - max_z_cm_in_life]

    frame_num += 1


# present radial flow until another option is selected (1 or 2), can go back to radial flow by pressing 0.
    if event.getKeys(keyList=["0"]):
        motion_type = 'flow_motion'
    elif event.getKeys(keyList=["1"]):
        motion_type = 'rand_z_dir'
    elif event.getKeys(keyList=["2"]):
        motion_type = 'new_xy_when_born'

    if motion_type == 'flow_motion':  # pressed 0
        # 1. Update z (distance values): Add dots_speed * flow_dir to the current z values.
        z_array = z_array + flow_speed_cm_p_fr * flow_dir


    elif motion_type == 'rand_z_dir':  # pressed 1
        # 1. Update z (distance values): Add dots_speed * flow_dir to the current z values.
        # create random_z_dir array, which is either 1 or -1, to add to z_array
        random_z_dir = np.random.choice([-1, 1], size=n_dots)
        random_speed_array = flow_speed_cm_p_fr * random_z_dir
        z_array = z_array + random_speed_array


    elif motion_type == 'new_xy_when_born':  # pressed 2
        # dots only change as a result of getting new x & ys at the end of their life, no continuous motion in x, y or z directions.
        z_array = z_array



    # # # all methods use the code below # # #

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
    # flow_dots.xys = dots_pos_array
    # flow_dots.draw()
    #
    #
    #
    # flow_dots.draw()
    # win.flip()




    # store data for the x, y dotlife. of the dots on each frame in motion_data
    for dot_idx, (x_cm, y_cm, z_cm, dot_life) in enumerate(zip(dots_pos_array.T[0], dots_pos_array.T[1], z_array, dot_lifetime_array)):
        motion_data.append([frame_num, dot_idx, x_cm, y_cm, z_cm, dot_life])

# win.close()

'''from the motion data, get the mean probe speed per frame by using the x and y data.
I want this for:
    1. all dots
    2. dots that are between 3 and 5 dva from the centre of the screen.
I want the speeds in:
    a. pixels per frame
    b. degrees per second.
Return the mean and standard deviation of the speeds.
Plot histograms of the speeds.
'''
save_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\project_stuff\dot_speeds"

dot_df = pd.DataFrame(motion_data, columns=['frame_num', 'dot_idx', 'x_cm', 'y_cm', 'z_cm', 'dot_life'])
print(dot_df.head())

# plot histograms of x_cm, y_cm and z_cm values
sns.histplot(data=dot_df, x='x_cm')
plt.show()
sns.histplot(data=dot_df, x='y_cm')
plt.show()
sns.histplot(data=dot_df, x='z_cm')
plt.title(f"n_frames: {frame_num}, contracting: {contracting}, z_start_bounds: {z_start_bounds}")
plt.show()

dot_df['x_pix'] = cm2pix(dot_df['x_cm'], monitor=mon)
sns.histplot(data=dot_df, x='x_pix')
# plt.title("x_pix")
plt.show()
dot_df['y_pix'] = cm2pix(dot_df['y_cm'], monitor=mon)
sns.histplot(data=dot_df, x='y_pix')
# plt.title("y_pix")
plt.show()



# from the x and y positions, get the distance from the centre of the screen in pixels
# dist_from_fix_cm = np.sqrt(dot_df['x_cm']**2 + dot_df['y_cm']**2)
# dff_pix = dist_from_fix_cm / mon_width_cm * widthPix
dff_pix = np.sqrt(dot_df['x_pix']**2 + dot_df['y_pix']**2)
# add this to the dataframe
dot_df['dff_pix'] = dff_pix

# add dva4_region column to dataframe for dots that are between 3 and 5 dva from the centre of the screen.
dot_df['dva4_region'] = np.where((dot_df['dff_pix'] >= dff_2) & (dot_df['dff_pix'] <= dff_6), 1, 0)

# get the distance travelled by each dot in pixels using the x and y positions and dot_idx.
# compare position on frame_num, but if NaN if dot_life is 0.
# sort by dot_idx and frame_num
dot_df = dot_df.sort_values(by=['dot_idx', 'frame_num'])
dot_df['x_pix_prev'] = dot_df['x_pix'].shift(1)
dot_df['y_pix_prev'] = dot_df['y_pix'].shift(1)

# set x_pix_prev and y_pix_prev to NaN if dot_life is 0 or if frame_num is 1.
dot_df['x_pix_prev'] = np.where((dot_df['dot_life'] == 0) | (dot_df['frame_num'] == 1), np.nan, dot_df['x_pix_prev'])
dot_df['y_pix_prev'] = np.where((dot_df['dot_life'] == 0) | (dot_df['frame_num'] == 1), np.nan, dot_df['y_pix_prev'])

# get distance travelled by each dot in pixels
dot_df['spd_pix_fr'] = np.sqrt((dot_df['x_pix'] - dot_df['x_pix_prev'])**2 + (dot_df['y_pix'] - dot_df['y_pix_prev'])**2)

# 1a. pix per frame for all dots
# get mean and std dev of spd_pix_fr for all dots
mean_spd_pix_p_fr = dot_df['spd_pix_fr'].mean()
std_spd_pix_p_fr = dot_df['spd_pix_fr'].std()

# make a scatter plot of spd_pix_fr against dff_pix with a colour bar for z
fig, ax = plt.subplots()
sns.scatterplot(data=dot_df, x='dff_pix', y='spd_pix_fr', hue='z_cm',
                alpha=.1, palette='rocket', legend=None, ax=ax)

# add colour bar
norm = plt.Normalize(dot_df['z_cm'].min(), dot_df['z_cm'].max())
sm = plt.cm.ScalarMappable(cmap="rocket", norm=norm)
sm.set_array([])
ax.figure.colorbar(sm, ax=ax, label='z (cm)')

# decorate plot
plt.title(f'spd_pix_fr - all dots M={round(mean_spd_pix_p_fr, 2)}, SD={round(std_spd_pix_p_fr, 2)}')
plt.xlabel('dff_pix')
plt.ylabel('spd_pix_fr')
plt.savefig(os.path.join(save_path, 'spd_pix_fr_all_dots.png'))
plt.show()



# 1b. dva per second for all dots
# get spd_dva_fr by multiplying spd_pix_fr by dva_per_pix
# dot_df['spd_dva_fr'] = dot_df['spd_pix_fr'] * dva_per_pix
dot_df['spd_dva_fr'] = pix2deg(dot_df['spd_pix_fr'], mon)
# get spd_dva_sec by multiplying spd_dva_fr by fps
dot_df['spd_dva_sec'] = dot_df['spd_dva_fr'] * fps

# get dff_dva by multiplying dff_pix by dva_per_pix
dot_df['dff_dva'] = dot_df['dff_pix'] * dva_per_pix


# get mean and std dev of spd_dva_sec for all dots
mean_spd_dva_p_sec = dot_df['spd_dva_sec'].mean()
std_spd_dva_p_sec = dot_df['spd_dva_sec'].std()

# plot scatter plot of spd_dva_sec against dff_dva
# plt.scatter(dot_df['dff_dva'], dot_df['spd_dva_sec'], alpha=.2, s=.1)
fig, ax = plt.subplots()
sns.scatterplot(data=dot_df, x='dff_dva', y='spd_dva_sec', hue='z_cm',
                alpha=.1, palette='rocket', legend=None, ax=ax)

# add colour bar
norm = plt.Normalize(dot_df['z_cm'].min(), dot_df['z_cm'].max())
sm = plt.cm.ScalarMappable(cmap="rocket", norm=norm)
sm.set_array([])
ax.figure.colorbar(sm, ax=ax, label='z (cm)')

# decorate plot
plt.title(f'spd_dva_sec - all dots M={round(mean_spd_dva_p_sec, 2)}, SD={round(std_spd_dva_p_sec, 2)}')
plt.xlabel('dff_dva')
plt.ylabel('spd_dva_sec')
plt.savefig(os.path.join(save_path, 'spd_dva_sec_all_dots.png'))
plt.show()

### part 2, dots in dva4_region
dot_dva4_df = dot_df[dot_df['dva4_region'] == 1]

# 2a. pix per frame for dots in dva4_region
# get mean and std dev of spd_pix_fr for dots in dva4_region
mean_spd_pix_p_fr_dva4 = dot_dva4_df['spd_pix_fr'].mean()
std_spd_pix_p_fr_dva4 = dot_dva4_df['spd_pix_fr'].std()

# plot scatter plot of spd_pix_fr against dff_pix for dots in dva4_region
# plt.scatter(dot_dva4_df['dff_pix'], dot_dva4_df['spd_pix_fr'], alpha=.1, s=5)
fig, ax = plt.subplots()
sns.scatterplot(data=dot_dva4_df, x='dff_pix', y='spd_pix_fr',
                hue='z_cm', palette='rocket',
                alpha=.4, legend=None, ax=ax)

# add colour bar
norm = plt.Normalize(dot_dva4_df['z_cm'].min(), dot_dva4_df['z_cm'].max())
sm = plt.cm.ScalarMappable(cmap="rocket", norm=norm)
sm.set_array([])
ax.figure.colorbar(sm, ax=ax, label='z (cm)')

# decorate plot
plt.title(f'spd_pix_fr - dva4_region M={round(mean_spd_pix_p_fr_dva4, 2)}, SD={round(std_spd_pix_p_fr_dva4, 2)}')
plt.xlabel('dff_pix')
plt.ylabel('spd_pix_fr')
plt.savefig(os.path.join(save_path, 'spd_pix_fr_dva4.png'))
plt.show()

# 2b. dva per second for dots in dva4_region
# get mean and std dev of spd_dva_sec for dots in dva4_region
mean_spd_dva_p_sec_dva4 = dot_dva4_df['spd_dva_sec'].mean()
std_spd_dva_p_sec_dva4 = dot_dva4_df['spd_dva_sec'].std()

# # plot scatter plot of spd_dva_sec against dff_dva for dots in dva4_region
# plt.scatter(dot_dva4_df['dff_dva'], dot_dva4_df['spd_dva_sec'], alpha=.1, s=5)
fig, ax = plt.subplots()
sns.scatterplot(data=dot_dva4_df, x='dff_dva', y='spd_dva_sec',
                hue='z_cm', palette='rocket',
                alpha=.4, legend=None, ax=ax)

# add colour bar
norm = plt.Normalize(dot_dva4_df['z_cm'].min(), dot_dva4_df['z_cm'].max())
sm = plt.cm.ScalarMappable(cmap="rocket", norm=norm)
sm.set_array([])
ax.figure.colorbar(sm, ax=ax, label='z (cm)')

# decorate plot
plt.title(f'spd_dva_sec - dva4_region M={round(mean_spd_dva_p_sec_dva4, 2)}, SD={round(std_spd_dva_p_sec_dva4, 2)}')
plt.xlabel('dff_dva')
plt.ylabel('spd_dva_sec')
plt.savefig(os.path.join(save_path, 'spd_dva_sec_dva4.png'))
plt.show()



dot_df.to_csv(os.path.join(save_path, 'dot_speeds.csv'), index=False)

print("Finished successfully")



