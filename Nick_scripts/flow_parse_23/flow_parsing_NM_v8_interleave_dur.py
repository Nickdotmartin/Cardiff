from __future__ import division
from psychopy import gui, visual, core, data, event, monitors, logging
from psychopy import __version__ as psychopy_version
from psychopy.tools.monitorunittools import cm2pix, pix2cm
from datetime import datetime
from os import path, chdir
from kestenSTmaxVal import Staircase
from copy import copy
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import gc
import numpy as np
# for numpy attributes access by per-frame functions, acceess them with name instead of np.name.
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
This script takes: 
the background from Evans et al., 2020, and moving probes from flow_parsing_NM
to test that we do get flowParsing effects.

This script uses pure psychopy code (no openGL), it drops frames on windows but runs fine on linux.

Evans et al (Exp 2) used:
Stimuli and tasks in both experiments were
Samsung LCD monitor at 120 Hz,  1680 × 1050 pixel (47.3 × 29.6 cm).
Iparticipants were stationary, seated in a dark room, heads stabilized on chinrest, eyes approximately 57 cm from the screen. 
Forward observer-movement was simulated by presenting onscreen patterns of radial motion consistent with such a movement.

Our methods were similar to several previous studies using a relative tilt task to probe optic flow parsing (e.g., Warren & Rushton, 2008; Warren & Rushton, 2009a; Foulkes et al., 2013a). 
The stimulus comprised an expanding limited lifetime optic flow field, generated as a virtual cloud of 300 red dots on a black background. 
Random uniform onscreen two-dimensional dot were generated before assigning the dots random depth locations spanning the full screen size 
(47.3° × 29.6°) between 107 cm and 207 cm (57 cm viewing distance + a uniform sample in the range [50, 150] cm) from the observer. 
The dot speed was consistent with simulated forward observer motion of 120 cm/s (i.e., a slow walking speed). 
Dots had a limited lifetime of 20 frames (166.67 ms), after which they were regenerated at a new random location to maintain dot density. 

In addition to the flow field, a probe dot was presented either −4° of expansion of the flow field. 
On each trial the probe (left) or +4° (right) of a central fixation dot at the focus moved at a speed of 0.8cm/s 
and at an angle of 75°, 90°, or 105° (measured clockwise relative to 0°, which was defined as rightward horizontal movement). 
These angles were selected randomly on each trial so physical trajectories could not be predicted.
Each 2.5-second trial consisted of 0.5-second presentation of the fixation cross alone, 
followed by 2-second presentation of the fixation cross together with the flow and probe stimuli.


- version 6 updates (10/10/2023:
    - changes polarity so that +ive values are inward for Flow and probes, (probes used to be opposite): DONE
        - check responses, probe start pos, probe position and probe start speeds: DONE
    - set start distance to be no biggest than sep 18pixels  
    
- version 7 12/10/2023
    - incoherent dot motion when not in prelim or probe presentation part of exp
        - no change in z, just new x and y positions when dots respawn.  DONE
    - accuracy feedback during break  DONE
    - added function to put dots into spoke formation  DONE
    
version 8 17/10/2023
    - Flip polarity of fixation on OLED
    - add small mask behine fixation so dots don't distract too much
    - add variable fixation time
    
version 8 interleaved probe durations
    - add multiple probe durations to interleave
"""


def find_angle(adjacent, opposite):
    """Finds the angle in a right triangle given the lengths of the adjacent and opposite sides.
    e.g., for getting the visual angle of a square at a given distance,
    the adjacent side is the distance from the screen,
    and the opposite side is the size of the square onscreen.

    :param adjacent: A numpy array of the lengths of the adjacent sides (e.g., distance z_array).
    :param opposite: The (scalar) length of the side opposite the angle you want to find.
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


# # #todo: Nick, try prelim durations of 210 and 490, to give 0, 70, 210, 350, 490

# # # DIALOGUE BOX # # #
expInfo = {'1_participant_name': 'Nicktest',
           '2_run_number': 1,
           '3_monitor_name': ['Nick_work_laptop', 'OLED', 'asus_cal', 'ASUS_2_13_240Hz',
                              'Samsung', 'Asus_VG24', 'HP_24uh', 'NickMac', 'Iiyama_2_18'],
           '4_fps': [60, 240, 120, 60],
           # '5_probe_dur_ms': [41.67, 116.67, 66.67, 54.17, 50, 41.67, 33.34, 25,  500],
           '6_probe_start_dist_pix': [6, 2, 4, 6, 8, 10],
           '7_debug': [False, True]
           }

# run drop-down menu, OK continues, cancel quits
dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
if not dlg.OK:
    core.quit()  # user pressed cancel

# Settings from dialogue box
participant_name = str(expInfo['1_participant_name'])
run_number = int(expInfo['2_run_number'])
monitor_name = str(expInfo['3_monitor_name'])
fps = int(expInfo['4_fps'])
# selected_probe_dur_ms = float(expInfo['5_probe_dur_ms'])  # interleave probe durs on this script
selected_probe_dur_ms = [41.67, 116.67]
probe_start_dist_pix = int(expInfo['6_probe_start_dist_pix'])  #
debug = eval(expInfo['7_debug'])

# print settings from dlg
print("\ndlg dict")
for k, v in expInfo.items():
    print(f'{k}: {v}')


# # # MISC SETTINGS # # #
n_trials_per_stair = 25  # this is the number of trials per stair
if debug:
    n_trials_per_stair = 2
probe_ecc = 4  # probe eccentricity in dva
record_fr_durs = True  # eval(expInfo['7_record_frame_durs'])  # always record frame durs
vary_fixation = True  # vary fixation time between trials to reduce anticipatory effects
expInfo['date'] = datetime.now().strftime("%d/%m/%Y")
expInfo['time'] = datetime.now().strftime("%H:%M:%S")


# # # CONVERT TIMINGS TO USE IN SAVE PATH # # #
# # probe_dur_ms and equivalent ISI_fr cond on 240Hz (total frames is ISI_fr plus 4 for probes)
'''   
dur_ms:       [8.34, 16.67, 25, 33.34, 41.67, 50, 54.17, 58.38, 66.67, 116.67]
ISI cond:     [conc,     0,  2,     4,     6,  8,     9,    10,    12,     24] 
frames@240Hz: [   2,     4,  6,     8,    10, 12,    13,    14,    16,     28]
frames@120hz: [   1,     2,  3,     4,     5,  6,      ,     7,     8,     14]
frames@60hz:  [    ,     1,   ,     2,      ,  3,      ,      ,     4,      7]
'''
# probe_dur_fr = int(selected_probe_dur_ms * fps / 1000)
# probe_dur_ms = (1 / fps) * probe_dur_fr * 1000
# print(f"\nprobe duration: {probe_dur_ms}ms, or {probe_dur_fr} frames")
# if probe_dur_fr == 0:
#     raise ValueError(f"probe_dur_fr is 0 because selected_probe_dur_ms ({selected_probe_dur_ms}) is less than a frame on this monitor ({1000/fps})ms")


# # # EXPERIMENT HANDLING AND SAVING # # #
# save each participant's files into separate dir for each ISI
save_dir = path.join(_thisDir, expName, monitor_name,
                     participant_name,
                     f'{participant_name}_{run_number}',  # don't use p_name_run here, as it's not a separate folder
                     # f'probeDur{int(probe_dur_ms)}')
                     f'probeDur_41_116')
print(f"\nexperiment save_dir: {save_dir}")

# files are labelled as '_incomplete' unless entire script runs.
p_name_run = f"{participant_name}_{run_number}"
if debug:
    p_name_run = f"{participant_name}_{run_number}_debug"
incomplete_output_filename = f'{p_name_run}_incomplete'
save_output_as = path.join(save_dir, incomplete_output_filename)

# Experiment Handler
thisExp = data.ExperimentHandler(name=expName, version=psychopy_version,
                                 extraInfo=expInfo, runtimeInfo=None,
                                 savePickle=None, saveWideText=True,
                                 dataFileName=save_output_as)


# # # CONDITIONS AND STAIRCASES # # #
# probe_dur_selected_ms_vals = [int(i) for i in selected_probe_dur_ms]
probe_dur_selected_ms_vals = selected_probe_dur_ms
# probe_dur_fr = int(selected_probe_dur_ms * fps / 1000)
probe_dur_fr_vals = [int(i * fps / 1000) for i in probe_dur_selected_ms_vals]
# probe_dur_ms = (1 / fps) * probe_dur_fr * 1000
probe_dur_ms_vals = [round((1 / fps) * i * 1000, 2) for i in probe_dur_fr_vals]

probe_dur_all_vals = zip(probe_dur_selected_ms_vals, probe_dur_fr_vals, probe_dur_ms_vals)

# check if any probe_dur_fr_vals are < 1
if any([i < 1 for i in probe_dur_fr_vals]):
    raise ValueError(f"probe_dur_fr is 0 because probe_dur_selected_ms_vals ({probe_dur_selected_ms_vals}) "
                     f"contains values less than a frame on this monitor ({1000/fps})ms\n"
                     f"probe_dur_fr_vals: {probe_dur_fr_vals}")

print(f"probe_dur_selected_ms_vals: {probe_dur_selected_ms_vals}")
print(f"probe_dur_fr_vals: {probe_dur_fr_vals}")
print(f"probe_dur_ms_vals: {probe_dur_ms_vals}")
print(f"probe_dur_all_vals: {list(probe_dur_all_vals)}")

# print(f"\nprobe duration: {probe_dur_ms}ms, or {probe_dur_fr} frames")
# if probe_dur_fr == 0:
#     raise ValueError(f"probe_dur_fr is 0 because selected_probe_dur_ms ({selected_probe_dur_ms}) is less than a frame on this monitor ({1000/fps})ms")

# # Conditions/staricases: flow_dir (exp, contract) x prelim motion (0, 70, 350)
# 1 = inward/contracting, -1 = outward/expanding
flow_dir_vals = [1, -1]

# 'prelim' (preliminary motion) is how long (ms) the background motion starts before the probe appears
prelim_vals = [0, 70, 350]
if debug:
    prelim_vals = [70]

# get all possible combinations of these three lists
# combined_conds = [(f, p) for f in flow_dir_vals for p in prelim_vals]
combined_conds = [(f, p, d) for f in flow_dir_vals for p in prelim_vals for d in probe_dur_all_vals]


print(f"\ncombined_conds ({len(combined_conds)}: {combined_conds}")
stair_idx_list = list(range(len(combined_conds)))

# lists of values for each condition (all list are same length = n_stairs)
'''each flow_dir value appears in 3 stairs, e.g.,
flow_dir (expand/contract) x prelim (0, 70, 350)'''
# split the combined_conds into separate lists
flow_dir_list = [i[0] for i in combined_conds]
prelim_conds_list = [i[1] for i in combined_conds]

probe_dur_selected_ms_list = [i[2][0] for i in combined_conds]
probe_dur_fr_list = [i[2][1] for i in combined_conds]
probe_dur_ms_list = [i[2][2] for i in combined_conds]


# make flow name list to avoid confusion with 1s and -1 from flow_dir_list
flow_name_list = ['exp' if i == -1 else 'cont' for i in flow_dir_list]

# stair_names_list joins sep_conds_list, cong_name_conds_list and prelim_conds_list
# e.g., ['sep_6_cong_1_prelim_70', 'sep_6_cong_1_prelim_350', 'sep_6_cong_-1_prelim_70'...]
stair_names_list = [f"{i}_flow_{f}_{n}_prelim_{p}_probeDur_{int(d)}" for i, f, n, p, d in
                    zip(stair_idx_list, flow_dir_list, flow_name_list, prelim_conds_list, probe_dur_ms_list)]

if debug:
    print(f'flow_dir_list: {flow_dir_list}')
    print(f"flow_name_list: {flow_name_list}")
    print(f'prelim_conds_list: {prelim_conds_list}')
    print(f'probe_dur_fr_list: {probe_dur_fr_list}')
    print(f'probe_dur_ms_list: {probe_dur_ms_list}')
    print(f'probe_dur_selected_ms_list: {probe_dur_selected_ms_list}')


n_stairs = len(stair_idx_list)
total_n_trials = int(n_trials_per_stair * n_stairs)
print(f'\nstair_names_list: {stair_names_list}')
print(f'n_stairs: {n_stairs}, total_n_trials: {total_n_trials}')


# # # MONITOR SETTINGS # # #
# # COLORS AND LUMINANCES
maxLum = 106  # minLum = 0.12
bgLumProp = .2  # use .2 to match exp1 or .45 to match radial_flow_NM_v2.py
if monitor_name == 'OLED':
    bgLumProp = .0
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
    # flow_colour = [this_bgColour[0], this_bgColour[1] + adj_flow_colour / 2, this_bgColour[2]]
    flow_colour = [this_bgColour[0], this_bgColour[1] + adj_flow_colour / 3, this_bgColour[2]]  # even dimmer 12/10/2023


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
win = visual.Window(monitor=mon, size=(widthPix, heightPix), colorSpace=this_colourSpace, color=this_bgColour,
                    units='pix', screen=display_number, allowGUI=False, fullscr=True, useFBO=False)

# todo: check forum for other ideas if mouse is still there
win.mouseVisible = False

# # # PSYCHOPY COMPONENTS # # #
# MOUSE
myMouse = event.Mouse(visible=False)

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
                                # color='red',  # for testing
                                tex=None, units='pix',
                                # pos=[0, 0]
                                )

# PROBEs
probe_size = 1  # can make them larger for testing new configurations etc
probeVert = [(0, 0), (1, 0), (1, 1), (2, 1), (2, -1), (1, -1), (1, -2), (-1, -2), (-1, -1), (0, -1)]  # 5 pixels

if monitor_name == 'OLED':  # smaller, 3-pixel probes for OLED
    probeVert = [(0, 0), (1, 0), (1, 1), (2, 1),
                 (2, 0), (1, 0), (1, -1), (0, -1),
                 (0, -2), (-1, -2), (-1, -1), (0, -1)]

probe = visual.ShapeStim(win, vertices=probeVert, lineWidth=0, opacity=1, size=probe_size, interpolate=False,
                         fillColor=(1.0, 1.0, 1.0), colorSpace=this_colourSpace)

# probes and probe_masks are at dist_from_fix pixels from middle of the screen
dist_from_fix = int((np.tan(np.deg2rad(probe_ecc)) * view_dist_pix) / np.sqrt(2))



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
# changed edge_mask color from grey, fades to black round edges which makes screen edges less visible
edge_mask = visual.GratingStim(win, mask=mmask, tex=None, contrast=1.0,
                               size=(widthPix, heightPix), units='pix', color='black')


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
dot_life_max_ms = 666  # Simon says longer dot life than on original exp which used 166.67
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
check_start_acc_after = 10  # check accuracy after 10 trials.
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
    print(f"\ntake a {break_dur} second break every {take_break} trials ({n_breaks} breaks in total).")


# # # ON-SCREEN MESSAGES # # #
instructions = visual.TextStim(win=win, name='instructions', font='Arial', height=20,
                               color='white', colorSpace=this_colourSpace,
                               wrapWidth=widthPix / 2,
                               text="\n\n\nFocus on the fixation circle at the centre of the screen.\n\n"
                                    "A small white target will briefly appear on screen.\n\n"
                                    "Press 'I' or '1' if you see the probe moving inward (towards centre of screen),\n"
                                    "Press 'O' or '0' if you see the probe moving outward (towards edge of screen).\n\n"
                                    "If you aren't sure, just guess!\n\n"
                                    "Press any key to start")


too_many_dropped_fr = visual.TextStim(win=win, name='too_many_dropped_fr',
                                      text="The experiment had quit as the computer is dropping frames.\n"
                                           "Sorry for the inconvenience.\n"
                                           "Please contact the experimenter.\n\n"
                                           "Press any key to return to the desktop.",
                                      font='Arial', height=20, colorSpace=this_colourSpace)

break_text = f"Break\nTurn on the light and take at least {break_dur} seconds break.\n" \
             "Keep focussed on the fixation circle in the middle of the screen.\n" \
             "Remember, if you don't see the target, just guess!"
breaks = visual.TextStim(win=win, name='breaks', text=break_text, font='Arial',
                         pos=[0, 0], height=20, ori=0, color='white',
                         colorSpace=this_colourSpace)

acc_warning_text = "The experiment had quit as your score is low.\n\n" \
                   "The first few trials should be relatively easy, this doesn't seem to the be the case.\n\n" \
                   "Try selecting a different 6_probe_start_dist_pix value from the drop down menu:\n" \
                   f"If you were finding that the probe looked like a static point," \
                   f"try selecting a higher starting distance (> {probe_start_dist_pix});\n" \
                   f"if it looked like a line or streak, try decreasing the starting distance (> {probe_start_dist_pix}).\n\n"

acc_warning = visual.TextStim(win=win, name='acc_warning', text=acc_warning_text, font='Arial',
                         pos=[0, 0], height=20, ori=0, color='white',
                         colorSpace=this_colourSpace)


end_of_exp_text = "You have completed this experiment.\nThank you for your time."
end_of_exp = visual.TextStim(win=win, name='end_of_exp',
                             text=end_of_exp_text, color='white',
                             font='Arial', height=20, colorSpace=this_colourSpace)


# # # PRIORITY to increase speed # # #
# # turn on high priority here. (and turn off garbage collection)
gc.disable()
core.rush(True)
if monitor_name == 'OLED':
    core.rush(True, realtime=True)


# # # PROBE STARTING SPEED # # #
# """
# In Evans et al, 2020, the probes moved at 0.8cm/s, but it was present for 2000ms (e.g., moved 1.6cm)
# This seems like a good starting value for our staircases.
# However, I will scale it with probe_dur_ms, so that the probe moves 1.6cm in probe_dur_ms.
# Otherwise the shortest durations are too hard.
# I need to convert it from cm/s to pixels/frame,
# which depends on the monitor's refresh rate.
# I'm going to try starting with 18 pixels in dur.  Not sure what that is in cm.
# I'm using this because we used 18 as our max value in previous study.
# 18 pixels starts too fast, so now trying 12
# """
# start_dist_pix_in_dur = 0  # 8  # 12  # starting dist in pixels in probe_dur_ms
# start_dist_pix_per_fr = start_dist_pix_in_dur / probe_dur_fr  # starting dist in pixels per frame
# start_dist_pix_per_second = start_dist_pix_per_fr * fps  # starting dist in pixels per second
# start_dist_cm_per_second = pix2cm(pixels=start_dist_pix_per_second, monitor=mon)  # starting dist in cm per second
# print(f"\nstart_dist_pix_in_dur: {start_dist_pix_in_dur}pix\n"
#       f"start_dist_pix_per_fr: {start_dist_pix_per_fr:.2f}pix/fr\n"
#       f"start_dist_pix_per_second: {start_dist_pix_per_second:.2f}pix/s\n"
#       f"start_dist_cm_per_second: {start_dist_cm_per_second:.2f}cm/s")
#
# start_cm_per_s = start_dist_cm_per_second
#
#
# start_pix_per_s = cm2pix(cm=start_cm_per_s, monitor=mon)  # convert to pixels per second
# start_pix_per_fr = start_pix_per_s / fps  # convert to pixels per frame
# if start_pix_per_fr < 1:
#     start_pix_per_fr = 1

'''
New method - set probe_start_dist_pix from dlg, then convert to cm/s and pix/fr'''
start_dist_pix_in_dur = probe_start_dist_pix  # 8  # 12  # starting dist in pixels in probe_dur_ms
# start_pix_per_fr = start_dist_pix_in_dur / probe_dur_fr  # starting dist in pixels per frame
start_pix_per_fr = start_dist_pix_in_dur / min(probe_dur_fr_list)  # starting dist in pixels per frame
start_pix_per_s = start_pix_per_fr * fps  # starting dist in pixels per second
start_cm_per_s = pix2cm(pixels=start_pix_per_s, monitor=mon)  # starting dist in cm per second
if debug:
    print(f"\nstart_cm_per_s: {start_cm_per_s:.2f}cm/s\nstart_pix_per_s: {start_pix_per_s:.2f}pix/s, "
          f"start_pix_per_fr: {start_pix_per_fr:.2f}pix/fr")


# # # CONSTRUCT STAIRCASES # # #
stairStart = start_pix_per_fr
miniVal = -10
maxiVal = 10

stairs = []
for stair_idx in stair_idx_list:
    thisInfo = copy(expInfo)
    thisInfo['stair_idx'] = stair_idx
    thisInfo['stair_name'] = stair_names_list[stair_idx]
    thisInfo['flow_dir'] = flow_dir_list[stair_idx]
    thisInfo['flow_name'] = flow_name_list[stair_idx]
    thisInfo['prelim_ms'] = prelim_conds_list[stair_idx]
    thisInfo['probe_dur_ms'] = probe_dur_ms_list[stair_idx]
    thisInfo['probe_dur_fr'] = probe_dur_fr_list[stair_idx]
    thisInfo['probe_dur_selected_ms'] = probe_dur_selected_ms_list[stair_idx]

    thisStair = Staircase(name=stair_names_list[stair_idx],
                          type='simple',  # step size changes after each reversal only
                          value=stairStart * -flow_dir_list[stair_idx],  # each stair starts with opposite probe dir as bg motion
                          C=stairStart * 0.6,  # initial step size, as prop of maxLum
                          minRevs=3,
                          minTrials=n_trials_per_stair,
                          minVal=miniVal,
                          maxVal=maxiVal,
                          targetThresh=0.5,
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
            # if debug:
            print(f"\n({trial_num_inc_repeats}) trial_number: {trial_number}, "
                  f"stair_idx: {stair_idx}, thisStair: {thisStair}, step: {step}")


            # conditions (flow_dir, prelim)
            flow_dir = thisStair.extraInfo['flow_dir']
            flow_name = thisStair.extraInfo['flow_name']
            prelim_ms = thisStair.extraInfo['prelim_ms']

            # probe duration
            # selected_probe_dur_ms = thisStair.extraInfo['probe_dur_ms']
            # probe_dur_fr = int(selected_probe_dur_ms * fps / 1000)
            # probe_dur_ms = (1 / fps) * probe_dur_fr * 1000
            # # print(f"\nprobe duration: {probe_dur_ms}ms, or {probe_dur_fr} frames")
            # if probe_dur_fr == 0:
            #     raise ValueError(
            #         f"probe_dur_fr is 0 because selected_probe_dur_ms ({selected_probe_dur_ms}) is less than a frame on this monitor ({1000 / fps})ms")

            probe_dur_selected_ms = thisStair.extraInfo['probe_dur_selected_ms']
            probe_dur_fr = thisStair.extraInfo['probe_dur_fr']
            probe_dur_ms = thisStair.extraInfo['probe_dur_ms']

            if debug:
                print(f"flow_dir: {flow_dir}, flow_name: {flow_name}, prelim_ms: {prelim_ms} "
                      f"probe_dur_ms: {probe_dur_ms}, probe_dur_fr: {probe_dur_fr}")


            # boundaries for z position (distance from screen) during radial flow
            if flow_dir == -1:  # expanding
                z_start_bounds = [near_plane_cm + max_z_cm_in_life, far_plane_cm]
            else:  # contracting, flow_dir == 1
                z_start_bounds = [near_plane_cm, far_plane_cm - max_z_cm_in_life]
            if debug:
                print(f"z_start_bounds: {z_start_bounds}")


            # # # GET PROBE ATTRIBUTES # # #
            probe_pix_p_fr = thisStair.next()  # in pixels per frame, for thie script
            probe_cm_p_sec = pix2cm(pixels=probe_pix_p_fr * fps, monitor=mon)  # for analysis file

            probe_dir = 'out'
            if probe_pix_p_fr > 0:
                probe_dir = 'in'
            # if debug:
            print(f"probe_dir: {probe_dir}, probe_pix_p_fr: {probe_pix_p_fr}, "
                  f"probe_cm_p_sec: {probe_cm_p_sec}")

            # timing for background motion converted to frames (e.g., 70ms is 17frames at 240Hz).
            prelim_fr = int(prelim_ms * fps / 1000)
            actual_prelim_ms = prelim_fr * 1000 / fps
            if debug:
                print(f'prelim_ms: {prelim_ms}, prelim_fr: {prelim_fr}, actual_prelim_ms: {actual_prelim_ms}')

            # PROBE LOCATIONS
            # corners go CCW(!) 45=top-right, 135=top-left, 225=bottom-left, 315=bottom-right
            corner = np.random.choice([45, 135, 225, 315])
            if debug:
                print(f'\tcorner: {corner}')


            # setting probe x and y starting positions

            # I want the starting position of the probe to take into account the direction it will travel.
            # e.g., if it's moving inwards, it should start further out and vice versa.
            # total distance travelled by the probe is probe_pix_p_fr * probe_dur_fr
            probe_start_offset = probe_pix_p_fr * probe_dur_fr / 2
            if corner == 45:
                probe_start_x = dist_from_fix + probe_start_offset
                probe_start_y = dist_from_fix + probe_start_offset
            elif corner == 135:
                probe_start_x = -dist_from_fix - probe_start_offset
                probe_start_y = dist_from_fix + probe_start_offset
            elif corner == 225:
                probe_start_x = -dist_from_fix - probe_start_offset
                probe_start_y = -dist_from_fix - probe_start_offset
            elif corner == 315:
                probe_start_x = dist_from_fix + probe_start_offset
                probe_start_y = -dist_from_fix - probe_start_offset

            # reset distrance travelled this trial
            probe_moved_x = 0
            probe_moved_y = 0

            # vary fixation polarity to reduce risk of screen burn.
            # if monitor_name == 'OLED':
            if trial_num_inc_repeats % 2 == 0:
                fixation.lineColor = 'grey'
                fixation.fillColor = 'black'
            else:
                fixation.lineColor = 'black'
                fixation.fillColor = 'grey'


            # VARIABLE FIXATION TIME
            '''to reduce anticipatory effects that might arise from fixation always being same length.
            if False, vary_fix == .5 seconds, so end_fix_fr is 1 second.
            if Ture, vary_fix is between 0 and 1 second, so end_fix_fr is between .5 and 1.5 seconds.'''
            vary_fix = int(fps / 2)
            if vary_fixation:
                vary_fix = np.random.randint(0, fps)



            # timing in frames
            # fixation time is now 70ms shorted than previously.
            # end_fix_fr = 1 * (fps - prelim_fr)  # 240 frames - 70ms for fixation, e.g., <1 second.
            end_fix_fr = int(fps / 2) + vary_fix - prelim_fr
            if end_fix_fr < 0:  # if prelim_fr is longer than 500ms, then end_fix_fr might be negative
                end_fix_fr = int(fps / 2)

            end_bg_motion_fr = end_fix_fr + prelim_fr  # bg_motion prior to probe for 70ms
            end_probe_fr = end_bg_motion_fr + probe_dur_fr  # probes appear during probe_duration (e.g., 240ms, 1 second).


            # reset fixation radius
            fixation.setRadius(3)



            # show accuracy warning after first n trials
            if trial_num_inc_repeats == check_start_acc_after:
                prop_correct = np.mean(resp_corr_list)
                acc_warning_text = acc_warning_text + (f"{prop_correct * 100:.2f}% correct.\n\n"
                                                       f"Press any key to return to the desktop.")
                acc_warning.text = acc_warning_text

                if prop_correct < initial_acc_thresh:
                    acc_warning.draw()
                    win.flip()
                    event.waitKeys()
                    core.quit()



            # take a break every ? trials
            if (trial_num_inc_repeats % take_break == 1) & (trial_num_inc_repeats > 1):
                if debug:
                    print("\nTaking a break.\n")

                prop_correct = np.mean(resp_corr_list)
                breaks.text = break_text + (f"\n{trial_number - 1}/{total_n_trials} trials completed.\n"
                                            f"{prop_correct * 100:.2f}% correct.\n\n")
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


            # # # PER_FRAME VARIABLES # # #
            frameN = -1
            # # continueRoutine here runs the per-frame section of the trial
            while continueRoutine:
                frameN = frameN + 1

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
                elif frameN == end_probe_fr + 1:
                    if record_fr_durs:
                        win.recordFrameIntervals = False


                # FIXATION until end of fixation interval
                if end_fix_fr >= frameN > 0:

                    '''just have incoherent motion from re-spawning dots, z bounds as full z range'''
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

                    # probeMask1.draw()
                    # probeMask2.draw()
                    # probeMask3.draw()
                    # probeMask4.draw()
                    edge_mask.draw()

                    fix_mask.draw()
                    fixation.draw()

                # Background motion after fixation, but before probe
                elif end_bg_motion_fr >= frameN > end_fix_fr:

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

                    # probeMask1.draw()
                    # probeMask2.draw()
                    # probeMask3.draw()
                    # probeMask4.draw()
                    edge_mask.draw()

                    fix_mask.draw()
                    fixation.draw()

                    # reset timer to start with probe1 presentation.
                    resp.clock.reset()

                # PROBE interval (with background motion), after preliminary background motion, before response
                elif end_probe_fr >= frameN > end_bg_motion_fr:

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

                    # probeMask1.draw()
                    # probeMask2.draw()
                    # probeMask3.draw()
                    # probeMask4.draw()
                    edge_mask.draw()

                    fix_mask.draw()
                    fixation.draw()


                    # draw probe if 1st interval
                    if corner == 45:  # top-right
                        probe_moved_y = probe_moved_y - probe_pix_p_fr
                        probe_moved_x = probe_moved_x - probe_pix_p_fr
                    elif corner == 135:  # top-left
                        probe_moved_y = probe_moved_y - probe_pix_p_fr
                        probe_moved_x = probe_moved_x + probe_pix_p_fr
                    elif corner == 225:  # bottom-left
                        probe_moved_y = probe_moved_y + probe_pix_p_fr
                        probe_moved_x = probe_moved_x + probe_pix_p_fr
                    elif corner == 315:  # bottom-right
                        probe_moved_y = probe_moved_y + probe_pix_p_fr
                        probe_moved_x = probe_moved_x - probe_pix_p_fr
                    probe.setPos([probe_start_x + probe_moved_x, probe_start_y + probe_moved_y])
                    probe.draw()



                # ANSWER - after probe interval, before next trial
                elif frameN > end_probe_fr:

                    '''just have incoherent motion from re-spawning dots, z bounds as full z range'''
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

                    # probeMask1.draw()
                    # probeMask2.draw()
                    # probeMask3.draw()
                    # probeMask4.draw()
                    edge_mask.draw()

                    fix_mask.draw()
                    fixation.setRadius(2)
                    fixation.draw()

                    # RESPONSE HANDLING
                    # resp = event.BuilderKeyResponse()
                    theseKeys = event.getKeys(keyList=['i', 'o', 'num_1', 'num_0'])
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
            # Kesten updates using response (in/out) not resp.corr correct/incorrect.
            # Kesten will try to find the speed that get 50% in and out responses.
            # But storing resp.corrs in the output file anyway.
            if (resp.keys == str('i')) or (resp.keys == 'num_1'):
                response = 1
                if probe_dir == 'in':
                    resp.corr = 1
                else:
                    resp.corr = 0
            elif (resp.keys == str('o')) or (resp.keys == 'num_0'):
                response = 0
                if probe_dir == 'out':
                    resp.corr = 1
                else:
                    resp.corr = 0
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


        # # # ADD TRIAL INFO TO OUTPUT CSV # # #
        thisExp.addData('trial_number', trial_number)
        thisExp.addData('trial_n_inc_rpt', trial_num_inc_repeats)
        thisExp.addData('stair', stair_idx)
        thisExp.addData('stair_name', thisStair)
        thisExp.addData('step', step)
        thisExp.addData('flow_dir', flow_dir)
        thisExp.addData('flow_name', flow_name)
        thisExp.addData('prelim_ms', prelim_ms)
        thisExp.addData('prelim_fr', prelim_fr)
        thisExp.addData('actual_prelim_ms', actual_prelim_ms)
        thisExp.addData('probe_dur_selected_ms', probe_dur_selected_ms)
        thisExp.addData('probe_dur_ms', probe_dur_ms)
        thisExp.addData('probe_dur_fr', probe_dur_fr)
        thisExp.addData('probe_dir', probe_dir)
        thisExp.addData('probe_pix_p_fr', probe_pix_p_fr)
        thisExp.addData('probe_cm_p_sec', probe_cm_p_sec)
        thisExp.addData('response', response)
        thisExp.addData('resp_corr', resp.corr)
        thisExp.addData('resp_rt', resp.rt)
        thisExp.addData('corner', corner)
        thisExp.addData('flow_speed_cm_p_sec', flow_speed_cm_p_sec)
        thisExp.addData('flow_speed_cm_p_fr', flow_speed_cm_p_fr)
        thisExp.addData('n_dots', n_dots)
        thisExp.addData('dot_life_max_ms', dot_life_max_ms)
        thisExp.addData('expName', expName)
        thisExp.addData('psychopy_version', psychopy_version)
        thisExp.addData('date', expInfo['date'])
        thisExp.addData('time', expInfo['time'])

        # tell psychopy to move to next trial
        thisExp.nextEntry()
        # update staircase based on whether response was inward or outward
        thisStair.newValue(response)  # so that the staircase adjusts itself


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
                participant=participant_name, run_num=run_number,
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
