from __future__ import division
from psychopy import gui, visual, core, data, event, monitors, logging
from psychopy import __version__ as psychopy_version
from psychopy.tools.monitorunittools import cm2pix, pix2cm
from os import path, chdir

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

import copy


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

    :param adjacent: A numpy array of the lengths of the adjacent sides (e.g., distance z_array).
    :param opposite: The (scalar) length of the side opposite the angle you want to find.
    :return: A numpy array of the angles in degrees.
    """
    return np.rad2deg(np.arctan(opposite / adjacent))



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
        dot_life_array = np.where(z_array > furthest_z, max_dot_life_fr, dot_life_array)
    elif flow_dir == 1:  # contracting
        dot_life_array = np.where(z_array < closest_z, max_dot_life_fr, dot_life_array)

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
    x_array[replace_mask] = np.random.uniform(low=-x_bounds, high=x_bounds, size=np.sum(replace_mask))
    y_array[replace_mask] = np.random.uniform(low=-y_bounds, high=y_bounds, size=np.sum(replace_mask))
    z_array[replace_mask] = np.random.uniform(low=z_start_bounds[0], high=z_start_bounds[1], size=np.sum(replace_mask))

    # reset life of replaced dots to 0
    dotlife_array[replace_mask] = 0

    return dotlife_array, x_array, y_array, z_array


def scaled_dots_pos_array(x_array, y_array, z_array, frame_size_cm, reference_angle):
    """
    This is a function to get new pixel x, y co-ordinates for the flow dots using the x, y and z arrays.
    Use this after updating z_array and dot_life_array.

    1. Convert distances (cm) to angles (degrees) using find_angle().
    2. scale distances by dividing by reference angle (e.g., screen angle when z=view_dist).
    3. scale x and y values by multiplying by scaled distances.
    4. put the new x_pos and y_pos co-ordinates into an array and transposes it.

    :param x_array: Original x_array positions for the dots (shape = (n_dots, 1))
    :param y_array: Original y_array positions for the dots (shape = (n_dots, 1))
    :param z_array: array of distance values for the dots (shape = (n_dots, 1))
    :param frame_size_cm: onscreen size in cm of frame containing dots.
    :param reference_angle: angle in degrees of the reference distance (57.3cm)
    :return: new dots_pos_array
    """

    # # 1. convert distances to angles
    # z_array_deg = find_angle(adjacent=z_array, opposite=frame_size_cm)
    #
    # # 2. scale distances by dividing by reference angle
    # scale_factor_array = z_array_deg / reference_angle
    #
    # # 3. scale x and y values by multiplying by scaled distances
    # scaled_x = x_array * scale_factor_array
    # scaled_y = y_array * scale_factor_array
    #
    # # 4. scale x and y values by multiplying by scaled distances
    # dots_pos_array = np.array([scaled_x, scaled_y]).T

    # 1. convert distances to angles and scale distances by dividing by reference angle
    scale_factor_array = find_angle(adjacent=z_array, opposite=frame_size_cm) / reference_angle

    # 2. scale x and y values by multiplying by scaled distances
    return np.array([x_array * scale_factor_array, y_array * scale_factor_array]).T



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
monitor_name = 'HP_24uh'  # Nick_work_laptop, HP_24uh
fps = 60

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
bgLumProp = .45  # .2  # todo: use .45 to match radial_flow_NM_v2.py, or .2 to match exp1
bgLum = maxLum * bgLumProp

# colour space
this_colourSpace = 'rgb1'  # values between 0 and 1
bgColor_rgb1 = bgLum / maxLum
this_bgColour = [bgColor_rgb1, bgColor_rgb1, bgColor_rgb1]

adj_flow_colour = .15
# Give dots a pale green colour, which is adj_flow_colour different to the background
flow_colour = [this_bgColour[0] - adj_flow_colour, this_bgColour[1], this_bgColour[2] - adj_flow_colour]

# screen number
display_number = 1  # 0 indexed, 1 for external display, 0 for internal
if monitor_name in ['asus_cal', 'Nick_work_laptop', 'NickMac', 'OLED', 'ASUS_2_13_240Hz']:
    display_number = 0

# WINDOW SPEC
# WINDOW SPEC
win = visual.Window(monitor=mon, size=(widthPix, heightPix), colorSpace=this_colourSpace, color=this_bgColour,
                    units='pix', screen=display_number, allowGUI=False, fullscr=True, useFBO=False)


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


# motion speed in cm/s
flow_speed_cm_p_sec = 120  # 1m/sec matches previous flow parsing study (Evans et al. 2020)
flow_speed_cm_p_fr = flow_speed_cm_p_sec / fps  # 1.66 cm per frame = 1m per second


# initialise dots - 1 per sq cm
dots_per_sq_cm = 1
n_dots = int(dots_per_sq_cm * mon_width_cm * mon_height_cm)
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

# get starting distances and scale xys
dots_pos_array = scaled_dots_pos_array(x_array, y_array, z_array, frame_size_cm, ref_angle)

flow_dots.xys = dots_pos_array

# dot lifetime ms
dot_life_max_ms = 166.67
dot_life_max_fr = int(dot_life_max_ms / 1000 * fps)
print(f"dot_life_max_fr: {dot_life_max_fr}")

# initialize lifetime of each dot (in frames)
dot_lifetime_array = np.random.randint(0, dot_life_max_fr, n_dots)

# when dots are redrawn with a new z value, they should be at least this far away the boundary
# otherwise they might have to be re-drawn after a couple of frames, which could lead to flickering.
# this is the max z_distance in meters they can travel in n frames
max_dist_in_life = flow_speed_cm_p_fr * dot_life_max_fr
print(f"max_dist_in_life: {max_dist_in_life}")

contracting = False
if contracting:
    flow_dir = 1
else:  # expanding
    flow_dir = -1

# boundaries for z position (distance from screen)
if flow_dir == -1:  # expanding
    z_start_bounds = [near_plane_cm + max_dist_in_life, far_plane_cm]
else:  # contracting, flow_dir == 1
    z_start_bounds = [near_plane_cm, far_plane_cm - max_dist_in_life]
print(f"z_start_bounds: {z_start_bounds}")

motion_type = 'flow_motion'  # press 1

# PRESENT STIMULI
while not event.getKeys(keyList=["escape"]):


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

    # 4. scale x and y positions by distance
    dots_pos_array = scaled_dots_pos_array(x_array, y_array, z_array, frame_size_cm, ref_angle)
    flow_dots.xys = dots_pos_array



    flow_dots.draw()
    win.flip()

win.close()

print("Finished successfully")



