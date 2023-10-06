from __future__ import division
from psychopy import gui, visual, core, data, event, monitors, logging
from psychopy import __version__ as psychopy_version
from psychopy.tools.monitorunittools import cm2pix, pix2cm
from datetime import datetime
from os import path, chdir
from kestenSTmaxVal import Staircase

import pyglet.gl as gl
import psychopy.tools.viewtools as vt
import psychopy.tools.mathtools as mt

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import copy
import gc


print(f"PsychoPy_version: {psychopy_version}")


"""
This script takes: 
1. background settings from Evans et al., 2020, 
2. flow dots with openGL
3. moving probes from flow_parsing_NM
to test that we do get flowParsing effects.

1. Evans, L., Champion, R. A., Rushton, S. K., Montaldi, D., & Warren, P. A. (2020). 
Detection of scene-relative object movement and optic flow parsing across the adult lifespan. 
Journal of Vision, 20(9), 1–18. https://doi.org/10.1167/JOV.20.9.12

2. openGL dots adapted demo from: https://discourse.psychopy.org/t/draw-dots-in-3d-space/8918.  
    Kathia, Sep '19: "Awesome! thanks! I tried it out and it worked."
"""

# # # FUNCTIONS # # #


def angle_from_dist_and_height(height, distance):
    """
    Gives the visual angle of an object from its distance and height.
    Used for calculating the size of the dot array at a given distance.

    :param height: Actual size of screen or size on screen in meters.
    :param distance: from screen in meters (e.g., .573m) or from screen plus z_values (e.g., .573m + 1m)
    :return: angle in degrees
    """
    return np.rad2deg(np.arctan(height / distance))


def height_from_dist_and_deg(distance, visual_angle):
    """
    Gives the height of an object from its distance and visual angle.
    Used for calculating the size of the dot array at a given distance.

    :param distance: from screen in meters (e.g., .573m) or from screen plus z_values (e.g., .573m + 1m)
    :param visual_angle:
    :return: height in meters
    """
    return distance * np.tan(np.deg2rad(visual_angle))


def draw_flow_dots(x_array, y_array, z_array, flow_colour_rgb1):
    """
    Function to draw flow dots in openGL.  All values are in meters.

    :param x_array: numpy array of x positions of dots
    :param y_array: numpy array of y positions of dots
    :param z_array: numpy array of z positions of dots (distance)
    """
    # join x, y, z into single 2d array (n, 3)
    dots_pos_array = np.array([x_array, y_array, z_array]).T

    # get number of dots
    n_points, _ = np.shape(dots_pos_array)

    # --- render loop ---
    # Apply the current view and projection matrices specified by ‘viewMatrix’ and ‘projectionMatrix’ using ‘immediate mode’ OpenGL.
    # Subsequent drawing operations will be affected until ‘flip()’ is called.
    # All transformations in GL_PROJECTION and GL_MODELVIEW matrix stacks will be cleared (set to identity) prior to applying.
    win.applyEyeTransform()

    # gl.glPushMatrix()

    # dot settings
    # gl.glColor3f(1.0, 1.0, 1.0)
    gl.glColor3f(flow_colour_rgb1[0], flow_colour_rgb1[1], flow_colour_rgb1[2])
    gl.glPointSize(5.0)

    # draw the dots
    gl.glBegin(gl.GL_POINTS)
    for i in range(n_points):
        gl.glVertex3f(*dots_pos_array[i, :])

    gl.glEnd()
    # gl.glPopMatrix()


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
        dot_life_array = np.where(z_array > closest_z, max_dot_life_fr, dot_life_array)
    elif flow_dir == 1:  # contracting
        dot_life_array = np.where(z_array < furthest_z, max_dot_life_fr, dot_life_array)

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


#######################
# # # MAIN SCRIPT # # #
#######################

# get filename and path for this experiment
_thisDir = path.dirname(path.abspath(__file__))
chdir(_thisDir)
expName = path.basename(__file__)[:-3]


# # # DIALOGUE BOX # # #
expInfo = {'1_participant_name': 'Nicktest_03102023',
           '2_run_number': 1,
           '3_monitor_name': ['Nick_work_laptop', 'OLED', 'asus_cal', 'ASUS_2_13_240Hz',
                              'Samsung', 'Asus_VG24', 'HP_24uh', 'NickMac', 'Iiyama_2_18'],
           '4_fps': [60, 240, 120, 60],
           '5_probe_dur_ms': [5000, 41.67, 8.33, 16.67, 25, 33.33, 41.67, 50, 58.38, 66.67, 500],
           '6_debug': [True, False, True]
           }

# run drop-down menu, OK continues, cancel quits
dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
if not dlg.OK:
    core.quit()  # user pressed cancel

# Settings from dialogue box
participant_name = expInfo['1_participant_name']
run_number = expInfo['2_run_number']
monitor_name = expInfo['3_monitor_name']
fps = int(expInfo['4_fps'])
probe_dur_ms = float(expInfo['5_probe_dur_ms'])
debug = eval(expInfo['6_debug'])

# print settings from dlg
print("\ndlg dict")
for k, v in expInfo.items():
    print(f'{k}: {v}')


# # # MISC SETTINGS # # #
n_trials_per_stair = 25  # this is the number of trials per stair
if debug:
    n_trials_per_stair = 2
probe_ecc = 4  # probe eccentricity in dva
expInfo['date'] = datetime.now().strftime("%d/%m/%Y")
expInfo['time'] = datetime.now().strftime("%H:%M:%S")


# trials_counter = False  # eval(expInfo['7_Trials_counter'])
# background = 'flow_rad'  # expInfo['8_Background'] # fix this to always be flow_rad
# bg_speed_cond = 'Normal'  # expInfo['9_bg_speed_cond']  # fix this to be 1m/s
# mask_type = expInfo['6_mask_type']

# todo: always record frame durs so get rid of this.
record_fr_durs = True  # eval(expInfo['7_record_frame_durs'])


# # # CONVERT TIMINGS TO USE IN SAVE PATH # # #
# Convert probe_dur_ms to probe_dur_fr
# # probe_dur_ms and equivalent ISI_fr cond on 240Hz (total frames is ISI_fr plus 4 for probes)
'''   
dur_ms:       [8.34, 16.67, 25, 33.34, 41.67, 50, 58.38, 66.67]
frames@240Hz: [   2,     4,  6,     8,    10, 12,    14,    16]
ISI cond:     [conc,     0,  2,     4,     6,  8,    10,    12] 
'''
probe_dur_fr = int(probe_dur_ms * fps / 1000)
probe_dur_actual_ms = (1 / fps) * probe_dur_fr * 1000
print(f"\nprobe duration: {probe_dur_actual_ms}ms, or {probe_dur_fr} frames")
if probe_dur_fr == 0:
    raise ValueError(f"probe_dur_fr is 0 because probe_dur_ms ({probe_dur_ms}) is less than "
                     f"one frame on this monitor ({1000/fps})ms")



# # # EXPERIMENT HANDLING AND SAVING # # #
# save each participant's files into separate dir for each ISI
save_dir = path.join(_thisDir, expName, monitor_name,
                     participant_name,
                     f'{participant_name}_{run_number}',
                     f'probeDur{int(probe_dur_ms)}')
print(f"\nexperiment save_dir: {save_dir}")

# files are labelled as '_incomplete' unless entire script runs.
incomplete_output_filename = f'{participant_name}_{run_number}_incomplete'
save_output_as = path.join(save_dir, incomplete_output_filename)

# Experiment Handler
thisExp = data.ExperimentHandler(name=expName, version=psychopy_version,
                                 extraInfo=expInfo, runtimeInfo=None,
                                 savePickle=None, saveWideText=True,
                                 dataFileName=save_output_as)


# # # CONDITIONS AND STAIRCASES # # #
# # Conditions/staircases: flow_dir (exp, contract) x prelim motion (0, 70, 350)
# 1 = inward/contracting, -1 = outward/expanding
flow_dir_vals = [1, -1]

# 'prelim' (preliminary motion) is how long (ms) the background motion starts before the probe appears
prelim_vals = [0]  # [0, 70, 350]

# get all possible combinations of these three lists
combined_conds = [(f, p) for f in flow_dir_vals for p in prelim_vals]

# lists of values for each condition (all list are same length = n_stairs)
'''each flow_dir value appears in 3 stairs, e.g.,
flow_dir (expand/contract) x prelim (0, 70, 350)'''
# split the combined_conds into separate lists
flow_dir_list = [i[0] for i in combined_conds]
prelim_conds_list = [i[1] for i in combined_conds]

# make flow name list to avoid confusion with 1s and -1 from flow_dir_list
flow_name_list = ['exp' if i == -1 else 'cont' for i in flow_dir_list]

# stair_names_list joins sep_conds_list, cong_name_conds_list and prelim_conds_list
# e.g., ['sep_6_cong_1_prelim_70', 'sep_6_cong_1_prelim_350', 'sep_6_cong_-1_prelim_70'...]
stair_names_list = [f"flow_{f}_{n}_prelim_{p}" for f, n, p in zip(flow_dir_list, flow_name_list, prelim_conds_list)]

if debug:
    print(f'flow_dir_list: {flow_dir_list}')
    print(f"flow_name_list: {flow_name_list}")
    print(f'prelim_conds_list: {prelim_conds_list}')


n_stairs = len(flow_dir_list)
total_n_trials = int(n_trials_per_stair * n_stairs)
print(f'\nstair_names_list: {stair_names_list}')
print(f'n_stairs: {n_stairs}, total_n_trials: {total_n_trials}')


# # # MONITOR SETTINGS # # #
'''MONITOR/screen/window details: colour, luminance, pixel size and frame rate'''
maxLum = 106  # 255 RGB
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
# Give dots a pale green colour, which is adj_flow_colour difference from the background
flow_colour = [this_bgColour[0] - adj_flow_colour, this_bgColour[1], this_bgColour[2] - adj_flow_colour]
if monitor_name == 'OLED':  # darker green for low contrast against black background
    flow_colour = [this_bgColour[0], this_bgColour[1] + adj_flow_colour / 2, this_bgColour[2]]

print(f"\nthis_bgColour: {this_bgColour}")
print(f"flow_colour: {flow_colour}")


# # # MONITOR DETAILS # # #
if debug:
    print(f"\nmonitor_name: {monitor_name}")
mon = monitors.Monitor(monitor_name)

widthPix = int(mon.getSizePix()[0])
heightPix = int(mon.getSizePix()[1])
mon_width_cm = mon.getWidth()  # monitor width in cm
view_dist_cm = mon.getDistance()  # viewing distance in cm
view_dist_pix = widthPix / mon_width_cm * view_dist_cm  # used for calculating dist_from_fix (probes at 4 degrees)


# values in meters for openGl
view_dist_m = view_dist_cm / 100  # original code had .50 commented as # 50cm
mon_width_m = mon_width_cm / 100  # # original code had 0.53 commented as # 53cm
scrAspect = widthPix / heightPix  # widthPix / heightPix
mon_height_m = mon_width_m / scrAspect
print(f"view_dist_m: {view_dist_m:.2f}m")
print(f"scrAspect: {scrAspect:.2f}, screen size: {mon_width_m:.2f}m x {mon_height_m:.2f}m")


# screen number
display_number = 1  # 0 indexed, 1 for external display, 0 for internal
if monitor_name in ['asus_cal', 'Nick_work_laptop', 'NickMac', 'OLED', 'ASUS_2_13_240Hz']:
    display_number = 0

# WINDOW SPEC
win = visual.Window(monitor=mon, size=(widthPix, heightPix), colorSpace=this_colourSpace, color=this_bgColour,
                    units='pix',
                    # units='cm',

                    screen=display_number, allowGUI=False, fullscr=True,
                    blendMode='avg',

                    # winType='pyglet',  # might need this to get the openGL dots to work
                    )



# # # PSYCHOPY COMPONENTS # # #
# MOUSE
myMouse = event.Mouse(visible=False)

# # KEYBOARD
resp = event.BuilderKeyResponse()

# fixation bull eye
fixation = visual.Circle(win, radius=2, units='pix', lineColor='white', fillColor='black', colorSpace=this_colourSpace)


# PROBEs
probe_size = 10  # can make them larger for testing new configurations etc
probeVert = [(0, 0), (1, 0), (1, 1), (2, 1), (2, -1), (1, -1), (1, -2), (-1, -2), (-1, -1), (0, -1)]  # 5 pixels

if monitor_name == 'OLED':  # smaller, 3-pixel probes for OLED
    probeVert = [(0, 0), (1, 0), (1, 1), (2, 1),
                 (2, 0), (1, 0), (1, -1), (0, -1),
                 (0, -2), (-1, -2), (-1, -1), (0, -1)]


probe = visual.ShapeStim(win, vertices=probeVert, lineWidth=0, opacity=1, size=probe_size, interpolate=False,
                         # fillColor=(1.0, 1.0, 1.0),
                         fillColor=(1.0, 0.0, 0.0),
                         colorSpace=this_colourSpace)

# probes and probe_masks are at dist_from_fix pixels from middle of the screen (converted from degrees)
dist_from_fix = int((np.tan(np.deg2rad(probe_ecc)) * view_dist_pix) / np.sqrt(2))


# MASK BEHIND PROBES (in front of flow dots to keep probes and motion separate)
mask_size = 150
probe_mask_colour = 'blue'  # this_bgColour
# Create a raisedCosine mask array and assign it to a Grating stimulus (grey outside, transparent inside)
raisedCosTexture1 = visual.filters.makeMask(256, shape='raisedCosine', fringeWidth=0.3, radius=[1.0, 1.0])
probeMask1 = visual.GratingStim(win=win, mask=raisedCosTexture1, size=(mask_size, mask_size),
                                colorSpace=this_colourSpace, color=probe_mask_colour,
                                tex=None, units='pix', pos=[dist_from_fix + 1, dist_from_fix + 1],
                                opacity=1
                                )
probeMask2 = visual.GratingStim(win=win, mask=raisedCosTexture1, size=(mask_size, mask_size),
                                colorSpace=this_colourSpace, color=probe_mask_colour,
                                units='pix', tex=None, pos=[-dist_from_fix - 1, dist_from_fix + 1],
                                opacity=.5)
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
# changed edge_mask color from grey, fades to black round edges which makes screen edges less visible
edge_mask = visual.GratingStim(win, mask=mmask, tex=None, contrast=1.0,
                               size=(widthPix, heightPix), units='pix', color='black')



# # # OPENGL SETTINGS for 3d projection # # #

# Set up the 3D projection matrix
# Frustum - a cone (or pyramid with the top cut off) from viewer, to screen, to onscreen items at 'distance'.
frustum = vt.computeFrustum(mon_width_m, scrAspect, view_dist_m, nearClip=0.01, farClip=10000.0)
P = vt.perspectiveProjectionMatrix(*frustum)

# Transformation for points (model/view matrix) - subtract view dist to place things in space
MV = mt.translationMatrix((0.0, 0.0, -view_dist_m))  # X, Y, Z

win.projectionMatrix = P
win.viewMatrix = MV

# set up for a different background colour in openGL
gl.glClearColor(bgColor_rgb1, bgColor_rgb1, bgColor_rgb1, 0)  # final value is alpha (transparency)
gl.glClear(gl.GL_COLOR_BUFFER_BIT)





# # # FLOW DOT SETTINGS (for openGL) # # #
n_dots = 300  # number of dots


# dot distance settings - openGL uses negative values for z axis, so closest is smaller -ve, furthest is larger -ve
closest_z = -.5  # .5m away, from Evals et al., 2020
furthest_z = -1.5  # 1.5m away, from Evals et al., 2020
closest_dist_m = closest_z - view_dist_m  # combines view_dist and z value
furthest_dist_m = furthest_z - view_dist_m  # combines view_dist and z value
if debug:
    print(f"closes_dist_m: {closest_dist_m}m = closest_z: {closest_z}m - view_dist_m: {view_dist_m}m")
    print(f"furthest_dist_m: {furthest_dist_m}m = furthest_z: {furthest_z}m - view_dist_m: {view_dist_m}m")

# initialise z array with random distances (meters) based on values above.
z_array = np.random.uniform(low=closest_z, high=furthest_z, size=n_dots)


# dot height and width settings
''' For the dots to fill the screen, but be a distance 'behind' the screen, 
they should be be drawn in a space 'bigger' than the screen (which then appears smaller at distance).
I first get the visual angle of the screen, using its size and the viewing distance.  
(or I can scale it so the dots fit on the screen with small border, e.g., 90% of the screen size).
I can then work out how big an object would be if it had the same visual angle, but was at a further distance.'''
scale = .9  # use 1.0 for full screen

# get the angle of the screen width and height
dot_frame_angles = vt.visualAngle(size=[mon_width_m * scale, mon_height_m * scale], distance=view_dist_m, degrees=True)
if debug:
    print(f"dot_frame_angles (w, h): {dot_frame_angles} degrees")

# use visual angle and distance (to screen, and then beyond) to get dot array size in meters
dot_fr_width_m = height_from_dist_and_deg(distance=closest_dist_m, visual_angle=dot_frame_angles[0])
dot_fr_height_m = height_from_dist_and_deg(distance=closest_dist_m, visual_angle=dot_frame_angles[1])
if debug:
    print(f"dot frame (meters): ({dot_fr_width_m:.2f}, {dot_fr_height_m:.2f})")

# initialise dot arrays with random position (meters) based on size calculated above.
x_array = np.random.uniform(low=-dot_fr_width_m/2, high=dot_fr_width_m/2, size=n_dots)
y_array = np.random.uniform(low=-dot_fr_height_m/2, high=dot_fr_height_m/2, size=n_dots)


# flow speed settings
flow_speed_m_p_s = 1.2  # 1m/sec matches previous flow parsing study (Evans et al. 2020)
flow_speed_m_p_fr = flow_speed_m_p_s / fps  # 1.66 cm per frame = 1m per second
if debug:
    print(f"flow_speed_m_p_fr: {flow_speed_m_p_fr:.3f}m per frame")


# dot lifetime settings - dots disappear after a fixed time and are redrawn with new x, y and z values.
dot_life_max_ms = 167  # maximum lifetime in ms, 167 to match Evans et al., 2020.
dot_life_max_fr = int(dot_life_max_ms / 1000 * fps)  # max life in frames
if debug:
    print(f"dot_life_max_fr: {dot_life_max_fr}")

# initialize lifetime of each dot (in frames)
dot_lifetime_array = np.random.randint(0, dot_life_max_fr, n_dots)

# when dots are redrawn with a new z value, they should be at least this far away the boundary
# otherwise they might have to be re-drawn after a couple of frames, which could lead to flickering.
max_dist_in_life = flow_speed_m_p_fr * dot_life_max_fr


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


# # # CONSTRUCT STAIRCASES # # #
"""
In Evans et al, 2020, the probes moved at 0.8cm/s.  
This seems like a good starting value for our staircases.
I need to convert it from cm/s to pixels/frame, which depends on the monitor's refresh rate.
"""
start_cm_per_s = 0.8  # starting value in cm per second
start_pix_per_s = cm2pix(cm=start_cm_per_s, monitor=mon)  # convert to pixels per second
start_pix_per_fr = start_pix_per_s / fps  # convert to pixels per frame
if start_pix_per_fr < 1:
    start_pix_per_fr = 1
if debug:
    print(f"\nstart_cm_per_s: {start_cm_per_s:.2f}cm/s, start_pix_per_s: {start_pix_per_s:.2f}pix/s, "
          f"start_pix_per_fr: {start_pix_per_fr}pix/fr")




stairStart = start_pix_per_fr  # 3  # starting value, for motion in pixels per frame
miniVal = -10  # Do I need to adjust these for slower monitors?
maxiVal = 10


stairs = []
for stair_idx in range(n_stairs):
    thisInfo = copy.copy(expInfo)
    thisInfo['stair_idx'] = stair_idx
    thisInfo['stair_name'] = stair_names_list[stair_idx]
    thisInfo['flow_dir'] = flow_dir_list[stair_idx]
    thisInfo['flow_name'] = flow_name_list[stair_idx]
    thisInfo['prelim_bg_flow_ms'] = prelim_conds_list[stair_idx]

    thisStair = Staircase(name=stair_names_list[stair_idx],
                          type='simple',
                          # value=stairStart,
                          # todo: put variable start dir back in
                          value=stairStart * -flow_dir_list[stair_idx],  # each stair starts with motion opposite to bg flow
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
    # fixation.draw()

    probeMask1.draw()
    probeMask2.draw()
    probeMask3.draw()
    probeMask4.draw()
    edge_mask.draw()

    fixation.draw()


    instructions.draw()
    win.flip()


# # # INITIALIZE COUNTERS # # #
trial_num_inc_repeats = 0  # number of trials including repeated trials
trial_number = 0  # the number of the trial for the output file (excluding repeats)



# # # RUN EXPERIMENT # # #
for step in range(n_trials_per_stair):
    np.random.shuffle(stairs)  # shuffle order each time after they've all been run.
    for thisStair in stairs:

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
            prelim_bg_flow_ms = thisStair.extraInfo['prelim_bg_flow_ms']
            if debug:
                print(f"flow_dir: {flow_dir}, flow_name: {flow_name}, prelim_bg_flow_ms: {prelim_bg_flow_ms}")


            # boundaries for z position (distance from screen)
            if flow_dir == -1:  # expanding
                z_start_bounds = [closest_z - max_dist_in_life, furthest_z]
            else:  # contracting, flow_dir == 1
                z_start_bounds = [closest_z, furthest_z + max_dist_in_life]


            probeSpeed = thisStair.next()
            # todo: put prbeSpeed back in
            # probeSpeed = 0.001
            if debug:
                print(f"probeSpeed: {probeSpeed}")


            # timing for background motion converted to frames (e.g., 70ms is 17frames at 240Hz).
            prelim_bg_flow_fr = int(prelim_bg_flow_ms * fps / 1000)
            actual_prelim_bg_flow_ms = prelim_bg_flow_fr * 1000 / fps
            if debug:
                print(f'\nprelim_bg_flow_ms: {prelim_bg_flow_ms}')
                print(f'prelim_bg_flow_fr: {prelim_bg_flow_fr}')
                print(f'actual_prelim_bg_flow_ms: {actual_prelim_bg_flow_ms}')

            # PROBE LOCATIONS
            # corners go CCW(!) 45=top-right, 135=top-left, 225=bottom-left, 315=bottom-right
            # corner = np.random.choice([45, 135, 225, 315])
            # todo: put corner back in
            corner = 45
            if debug:
                print(f'corner: {corner}, flow_dir: {flow_dir}, probeSpeed: {probeSpeed}')

            # setting x and y positions depending on the side
            # corners go CCW(!) 45=top-right, 135=top-left, 225=bottom-left, 315=bottom-right
            if corner == 45:
                x_position = dist_from_fix
                y_position = dist_from_fix
            elif corner == 135:
                x_position = -dist_from_fix
                y_position = dist_from_fix
            elif corner == 225:
                x_position = -dist_from_fix
                y_position = -dist_from_fix
            elif corner == 315:
                x_position = dist_from_fix
                y_position = -dist_from_fix

            # probe position reset
            probe_x = 0
            probe_y = 0


            # timing in frames
            # fixation time is now 70ms shorted than previously.
            end_fix_fr = 1 * (fps - prelim_bg_flow_fr)  # 240 frames - 70ms for fixation, e.g., <1 second.
            end_bg_motion_fr = end_fix_fr + prelim_bg_flow_fr  # bg_motion prior to probe for 70ms
            end_probe_fr = end_bg_motion_fr + probe_dur_fr  # probes appear during probe_duration (e.g., 240ms, 1 second).

            # reset fixation radius
            fixation.setRadius(3)

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
                elif frameN == end_probe_fr + 1:
                    if record_fr_durs:
                        win.recordFrameIntervals = False

                # FIXATION until end of fixation interval
                if end_fix_fr >= frameN > 0:

                    probeMask1.draw()
                    probeMask2.draw()
                    probeMask3.draw()
                    probeMask4.draw()
                    edge_mask.draw()

                    fixation.draw()

                    # draw flow_dots but with no motion
                    draw_flow_dots(x_array=x_array, y_array=y_array, z_array=z_array,
                                   flow_colour_rgb1=flow_colour)

                # preliminary background motion between fixation and probe intervals
                elif end_bg_motion_fr >= frameN > end_fix_fr:  
                    


                    probeMask1.draw()
                    probeMask2.draw()
                    probeMask3.draw()
                    probeMask4.draw()
                    edge_mask.draw()

                    # fixation.setRadius(3)
                    fixation.draw()

                    # update distance array: subtract as OpenGL distance is negative (psychopy was +ive).
                    z_array -= flow_speed_m_p_fr * flow_dir  # distance to move the dots per frame towards/away from viewer

                    # check if any z values are out of bounds (too close when expanding or too far when contracting),
                    # if so, set their dot life to max, so they are given new x, y and z values by update_dotlife() below.
                    dot_lifetime_array = check_z_start_bounds(z_array, closest_z, furthest_z, dot_life_max_fr,
                                                              dot_lifetime_array, flow_dir)

                    # update dot lifetime, give new x, y, z coords to dots whose lifetime is max.
                    dotlife_array, x_array, y_array, z_array = update_dotlife(dotlife_array=dot_lifetime_array,
                                                                              dot_max_fr=dot_life_max_fr,
                                                                              x_array=x_array, y_array=y_array,
                                                                              z_array=z_array,
                                                                              x_bounds=dot_fr_width_m / 2,
                                                                              y_bounds=dot_fr_height_m / 2,
                                                                              z_start_bounds=z_start_bounds)
                    # draw stimuli
                    draw_flow_dots(x_array=x_array, y_array=y_array, z_array=z_array,
                                   flow_colour_rgb1=flow_colour)



                    # reset timer to start with probe1 presentation.
                    resp.clock.reset()

                # PROBE interval (with background motion), after preliminary background motion, before response
                elif end_probe_fr >= frameN > end_bg_motion_fr:

                    probeMask1.draw()
                    probeMask2.draw()
                    probeMask3.draw()
                    probeMask4.draw()
                    edge_mask.draw()

                    # fixation.setRadius(3)
                    fixation.draw()

                    # draw moving probe
                    if corner == 45:  # top-right
                        probe_y = probe_y - probeSpeed
                        probe_x = probe_x - probeSpeed
                    elif corner == 135:  # top-left
                        probe_y = probe_y - probeSpeed
                        probe_x = probe_x + probeSpeed
                    elif corner == 225:  # bottom-left
                        probe_y = probe_y + probeSpeed
                        probe_x = probe_x + probeSpeed
                    elif corner == 315:  # bottom-right
                        probe_y = probe_y + probeSpeed
                        probe_x = probe_x - probeSpeed
                    probe.setPos([x_position + probe_x, y_position + probe_y])
                    probe.draw()

                    # update distance array: subtract as OpenGL distance is negative (psychopy was +ive).
                    z_array -= flow_speed_m_p_fr * flow_dir  # distance to move the dots per frame towards/away from viewer

                    # check if any z values are out of bounds (too close when expanding or too far when contracting),
                    # if so, set their dot life to max, so they are given new x, y and z values by update_dotlife() below.
                    dot_lifetime_array = check_z_start_bounds(z_array, closest_z, furthest_z, dot_life_max_fr,
                                                              dot_lifetime_array, flow_dir)

                    # update dot lifetime, give new x, y, z coords to dots whose lifetime is max.
                    dotlife_array, x_array, y_array, z_array = update_dotlife(dotlife_array=dot_lifetime_array,
                                                                              dot_max_fr=dot_life_max_fr,
                                                                              x_array=x_array, y_array=y_array,
                                                                              z_array=z_array,
                                                                              x_bounds=dot_fr_width_m / 2,
                                                                              y_bounds=dot_fr_height_m / 2,
                                                                              z_start_bounds=z_start_bounds)
                    # draw stimuli
                    draw_flow_dots(x_array=x_array, y_array=y_array, z_array=z_array,
                                   flow_colour_rgb1=flow_colour)



                    # # idiot check
                    # '''I want to know if the probe is moving towards or away from the centre of the screen. (0, 0)'''
                    # # if debug:
                    # print(f"\n{frameN}. probe_x: {probe_x}, probe_y: {probe_y}")


                # ANSWER - after probe interval, before next trial
                elif frameN > end_probe_fr:



                    probeMask1.draw()
                    probeMask2.draw()
                    probeMask3.draw()
                    probeMask4.draw()
                    edge_mask.draw()

                    fixation.setRadius(2)
                    fixation.draw()

                    # draw flow_dots but with no motion
                    draw_flow_dots(x_array=x_array, y_array=y_array, z_array=z_array,
                                   flow_colour_rgb1=flow_colour)


                    # ANSWER
                    resp = event.BuilderKeyResponse()
                    theseKeys = event.getKeys(keyList=['i', 'o', 'num_1', 'num_0'])
                    if len(theseKeys) > 0:  # at least one key was pressed
                        resp.keys = theseKeys[-1]  # just the last key pressed
                        resp.rt = resp.clock.getTime()

                        # a response ends the routine
                        continueRoutine = False

                        


                # regardless of frameN
                # check for quit
                if event.getKeys(keyList=["escape"]):
                    thisExp.close()
                    core.quit()

                # refresh the screen
                if continueRoutine:
                    win.flip()


            '''End of per-frame section in continueRoutine = False"'''


            # CHECK RESPONSES
            # default assume response incorrect unless meets criteria below
            resp.corr = 0


            # todo: new response logic.  The staircase algorithm will aim for a response score of .5.
            #  So if 'i' = 1 and 'o' = 0, then it will try to find the speed where they balance.
            # regardless of background or probe direction, just use responses.
            if (resp.keys == str('i')) or (resp.keys == 'num_1'):
                response = 1
            elif (resp.keys == str('o')) or (resp.keys == 'num_0'):
                response = 0
            else:
                response = None
            # resp.corr = response

            actual_probe_dir = 'out'
            if probeSpeed > 0:
                actual_probe_dir = 'in'
            print(f"\nprobeSpeed: {probeSpeed}, actual_probe_dir: {actual_probe_dir}")
            
            
            # for output file, to allow comparison between monitors 
            probeSpeed_cm_per_s = pix2cm(pixels=probeSpeed * fps, monitor=mon)

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
                        print(f"\n\toh no! A frame had bad timing! trial: {trial_number}, {thisStair.name} "
                              f"{round(max(trial_fr_intervals), 3)} > {round(max_fr_dur_sec, 2)} or "
                              f"{round(min(trial_fr_intervals), 3)} < {round(min_fr_dur_sec, 2)}")

                    print(f"Timing bad, repeating trial {trial_number}.")

                    # decrement trial and stair so that the correct values are used for the next trial
                    trial_number -= 1
                    thisStair.trialCount = thisStair.trialCount - 1  # so Kesten doesn't count this trial

                    # get first and last frame numbers for this trial
                    trial_x_locs = [fr_counter_per_trial[-1][0],
                                    fr_counter_per_trial[-1][-1] + 1]  # 1st fr of this trial to 1st of next trial
                    dropped_fr_trial_x_locs.append(trial_x_locs)
                    dropped_fr_trial_counter += 1
                    continue
                else:
                    repeat = False  # breaks out of while repeat=True loop to progress to new trial


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
        thisExp.addData('stair', stair_idx)
        thisExp.addData('stair_name', thisStair)

        thisExp.addData('flow_dir', flow_dir)
        thisExp.addData('flow_name', flow_name)
        thisExp.addData('prelim_bg_flow_ms', prelim_bg_flow_ms)
        thisExp.addData('prelim_bg_flow_fr', prelim_bg_flow_fr)
        thisExp.addData('step', step)
        thisExp.addData('probeSpeed', probeSpeed)
        thisExp.addData('actual_probe_dir', actual_probe_dir)
        thisExp.addData('probeSpeed_cm_per_s', probeSpeed_cm_per_s)
        
        thisExp.addData('response', response)
        # thisExp.addData('rel_answer', rel_answer)
        thisExp.addData('trial_response', resp.corr)
        thisExp.addData('resp.rt', resp.rt)

        thisExp.addData('corner', corner)
        thisExp.addData('probe_dur_ms', probe_dur_ms)
        thisExp.addData('probe_dur_actual_ms', probe_dur_actual_ms)
        thisExp.addData('probe_dur_fr', probe_dur_fr)


        thisExp.addData('flow_speed_m_p_s', flow_speed_m_p_s)
        thisExp.addData('flow_speed_m_p_fr', flow_speed_m_p_fr)
        thisExp.addData('n_dots', n_dots)
        thisExp.addData('dot_life_max_ms', dot_life_max_ms)
        # thisExp.addData('probeLum', probeLum)

        thisExp.addData('expName', expName)

        # tell psychopy to move to next trial
        thisExp.nextEntry()
        # update staircase based on whether response was correct or incorrect
        thisStair.newValue(response)  # so that the staircase adjusts itself

# print("end of exp loop, saving data")
# thisExp.close()


# now exp is completed, save as '_output' rather than '_incomplete'
thisExp.dataFileName = path.join(save_dir, f'{participant_name}_{run_number}_output')
thisExp.close()
print(f"\nend of experiment loop, saving data to:\n{thisExp.dataFileName}\n")



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


