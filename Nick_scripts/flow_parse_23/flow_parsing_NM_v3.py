from __future__ import division
from psychopy import gui, visual, core, data, event, monitors, logging
from psychopy import __version__ as psychopy_version
from psychopy.tools.monitorunittools import cm2pix, pix2cm
from datetime import datetime
from os import path, chdir

import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from kestenSTmaxVal import Staircase

print(f"PsychoPy_version: {psychopy_version}")


"""
This script takes: 
the background from Evans et al., 2020, and moving probes from flow_parsing_NM
to test that we do get flowParsing effects."""


def find_angle(adjacent, opposite):
    """Finds the angle in a right triangle given the lengths of the adjacent and opposite sides.
    e.g., for getting the visual angle of a square at a given distance, the adjacent side is the distance from the screen,
    and the opposite side is the size of the square onscreen.

    :param adjacent: A numpy array of the lengths of the adjacent sides (e.g., distance z_array).
    :param opposite: The (scalar) length of the side opposite the angle you want to find.
    :return: A numpy array of the angles in degrees.
    """
    radians = np.arctan(opposite / adjacent)  # radians
    degrees = radians * 180 / np.pi  # degrees
    return degrees


def new_dots_z_and_pos(x_array, y_array, z_array, dots_speed, flow_dir, min_z, max_z,
                       frame_size_cm, reference_angle):
    """
    This is a function to update flow_dots distance array and get new pixel co-ordinates
    using the original x_array and y_array.

    1. Update z_array by adding dots_speed * flow_dir to the current z values.
    2. adjust any values below dots_min_z or above dots_max_z.
    3. Convert distances (cm) to angles (degrees) using find_angle().
    4. scale distances by dividing by reference angle (e.g., screen angle when z=view_dist).
    5. scale x and y values by multiplying by scaled distances.
    6. put the new x_pos and y_pos co-ordinates into an array and transposes it.

    :param x_array: Original x_array positions for the dots (shape = (n_dots, 1))
    :param y_array: Original y_array positions for the dots (shape = (n_dots, 1))
    :param z_array: array of distance values for the dots (shape = (n_dots, 1))
    :param dots_speed: speed of the dots (float, smaller = slower, larger = faster)
    :param flow_dir: either 1 (contracting/inward/backwards) or -1 (expanding/outward/forwards)
    :param min_z: default is .5, values below this are adjusted to dots_max_z
    :param max_z: default is 5, values above this are adjusted to dots_min_z
    :param frame_size_cm: onscreen size in cm of frame containing dots.
    :param reference_angle: angle in degrees of the reference distance (57.3cm)
    :return: updated_z_array, new dots_pos_array
    """

    # # 1. Update z (distance values): Add dots_speed * flow_dir to the current z values.
    updated_z_array = z_array + dots_speed * flow_dir

    # todo: make sure new distances and dot life work together so there is no flickering.
    # 2. adjust any distance values below min_z or above max_z by z_adjust
    z_adjust = max_z - min_z
    # adjust updated_z_array values less than min_z by adding z_adjust
    less_than_min = (updated_z_array < min_z)
    updated_z_array[less_than_min] += z_adjust
    # adjust updated_z_array values more than max_z by subtracting z_adjust
    more_than_max = (updated_z_array > max_z)
    updated_z_array[more_than_max] -= z_adjust

    # 3. convert distances to angles
    z_array_deg = find_angle(adjacent=updated_z_array, opposite=frame_size_cm)

    # 4. scale distances by dividing by reference angle
    scale_factor_array = z_array_deg / reference_angle

    # 5. scale x and y values by multiplying by scaled distances
    scaled_x = x_array * scale_factor_array
    scaled_y = y_array * scale_factor_array

    # 6. scale x and y values by multiplying by scaled distances
    dots_pos_array = np.array([scaled_x, scaled_y]).T

    return updated_z_array, dots_pos_array


def update_dotlife(dotlife_array, dot_max_fr, x_array, y_array, dot_boundary):
    """
    This is a function to update the lifetime of the dots.

    1. increment all dots by 1
    2. make a mask of any to be replaced (life > max_life)
    3. replace these with new x and y values
    4. reset life of replaced dots to 0

    :param dotlife_array: np.array of dot lifetimes (ints) between 0 and dot_max_fr.
    :param dot_max_fr: maximum lifetime of a dot in frames.
    :param x_array: np.array of x positions of dots.
    :param y_array: np.array of y positions of dots.
    :param dot_boundary: width of the frame in cm for drawing new x and y values from.
    :return: updated dotlife_array, x_array, y_array
    """

    # increment all dots by 1
    dotlife_array += 1

    # make a mask of any to be replaced (life > max_life)
    replace_mask = (dotlife_array > dot_max_fr)

    # replace these with new x and y values (from same distribution as originals)
    x_array[replace_mask] = np.random.uniform(-dot_boundary / 2, dot_boundary / 2, np.sum(replace_mask))
    y_array[replace_mask] = np.random.uniform(-dot_boundary / 2, dot_boundary / 2, np.sum(replace_mask))

    # reset life of replaced dots to 0
    dotlife_array[replace_mask] = 0

    return dotlife_array, x_array, y_array



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


expInfo = {'1_participant_name': 'Nicktest_28092023',
           '2_run_number': 1,
           '3_monitor_name': ['Nick_work_laptop', 'OLED', 'asus_cal', 'ASUS_2_13_240Hz',
                              'Samsung', 'Asus_VG24', 'HP_24uh', 'NickMac', 'Iiyama_2_18'],
           '4_fps': [60, 240, 120, 60],
           # todo: changed probe_dur to be in ms, to allow comparison between monitors.
           '5_probe_dur_ms': [41.67, 8.33, 16.67, 25, 33.33, 41.67, 50, 58.38, 66.67, 500],
           # '6_mask_type': ['4_circles', '2_spokes'],
           '7_record_frame_durs': [True, False],
           '8_debug': [False, True]
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
# mask_type = expInfo['6_mask_type']
record_fr_durs = eval(expInfo['7_record_frame_durs'])
debug = eval(expInfo['8_debug'])

# print settings from dlg
print("\ndlg dict")
for k, v in expInfo.items():
    print(f'{k}: {v}')

# Misc settings
n_trials_per_stair = 25  # this is the number of trials per stair
if debug:
    n_trials_per_stair = 2
probe_ecc = 4  # probe eccentricity in dva
expInfo['date'] = datetime.now().strftime("%d/%m/%Y")
expInfo['time'] = datetime.now().strftime("%H:%M:%S")


# trials_counter = False  # eval(expInfo['7_Trials_counter'])
# background = 'flow_rad'  # expInfo['8_Background'] # fix this to always be flow_rad
# bg_speed_cond = 'Normal'  # expInfo['9_bg_speed_cond']  # fix this to be 1m/s


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
    raise ValueError(f"probe_dur_fr is 0 because probe_dur_ms ({probe_dur_ms}) is less than a frame on this monitor ({1000/fps})ms")


# # Conditions/staricases: flow_dir (exp, contract) x prelim motion (0, 70, 350)
# 1 = inward/contracting, -1 = outward/expanding
flow_dir_vals = [1, -1]

# 'prelim' (preliminary motion) is how long (ms) the background motion starts before the probe appears
prelim_vals = [0]  # , 70, 350]

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




'''Experiment handling and saving'''
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

# # COLORS AND LUMINANCE



'''MONITOR/screen/window details: colour, luminance, pixel size and frame rate'''
# # COLORS AND LUMINANCES
maxLum = 106  # 255 RGB
# minLum = 0.12  # 0 RGB  # todo: this is currently unused
bgLumProp = .2  # .2  # todo: use .45 to match radial_flow_NM_v2.py, or .2 to match exp1
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
    flow_colour = [this_bgColour[0], this_bgColour[1] + adj_flow_colour / 2, this_bgColour[2]]



# MONITOR SPEC
if debug:
    print(f"\nmonitor_name: {monitor_name}")
mon = monitors.Monitor(monitor_name)

widthPix = int(mon.getSizePix()[0])
heightPix = int(mon.getSizePix()[1])
mon_width_cm = mon.getWidth()  # monitor width in cm
view_dist_cm = mon.getDistance()  # viewing distance in cm
view_dist_pix = widthPix / mon_width_cm * view_dist_cm  # used for calculating visual angle (e.g., probe locations at 4dva)

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


# PROBEs

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
flow_speed_cm_p_sec = 100  # 1m/sec matches previous flow parsing study (Evans et al. 2020)
flow_speed_cm_p_fr = flow_speed_cm_p_sec / fps  # 1.66 cm per frame = 1m per second


# initialise dots
n_dots = 300  # use 300 to match flow parsing studies, or 10000 to match our rad flow studies
flow_dots = visual.ElementArrayStim(win, elementTex=None, elementMask='circle',
                                    units='cm', nElements=n_dots, sizes=.2,
                                    colorSpace=this_colourSpace,
                                    colors=flow_colour,
                                    )
# dot lifetime ms
dot_life_max_ms = 166.67
dot_life_max_fr = int(dot_life_max_ms / 1000 * fps)
print(f"dot_life_max_fr: {dot_life_max_fr}")


# initialize x and y positions of dots to fit in window (frame_size_cm) at distance 0
x_array = np.random.uniform(-frame_size_cm/2, frame_size_cm/2, n_dots)  # x values in cm
y_array = np.random.uniform(-frame_size_cm/2, frame_size_cm/2, n_dots)  # y values in cm

# initialize z values (distance/distance from viewer) in cm
z_array_cm = np.random.uniform(near_plane_cm, far_plane_cm, n_dots)    # distances in cm

# initialize lifetime of each dot (in frames)
dot_lifetime_array = np.random.randint(0, dot_life_max_fr, n_dots)

# # MASK BEHIND PROBES
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


# get starting distances and scale xys
# update z and scale xys
z_array_cm, scaled_xys = new_dots_z_and_pos(x_array=x_array, y_array=y_array, z_array=z_array_cm,
                                            dots_speed=flow_speed_cm_p_fr, flow_dir=1,
                                            min_z=near_plane_cm, max_z=far_plane_cm,
                                            frame_size_cm=frame_size_cm,
                                            reference_angle=ref_angle)
flow_dots.xys = scaled_xys




# full screen mask to blend off edges and fade to black
# Create a raisedCosine mask array and assign it to a Grating stimulus (grey outside, transparent inside)
# this was useful http://www.cogsci.nl/blog/tutorials/211-a-bit-about-patches-textures-and-masks-in-psychopy
raisedCosTexture2 = visual.filters.makeMask(heightPix, shape='raisedCosine', fringeWidth=0.6, radius=[1.0, 1.0])
invRaisedCosTexture = -raisedCosTexture2  # inverts mask to blur edges instead of center
slab_width = 420
# todo: try without this on OLED to reduce flicker?
if monitor_name == 'OLED':
    slab_width = 20

blankslab = np.ones((heightPix, slab_width))  # create blank slabs to put to left and right of image
mmask = np.append(blankslab, invRaisedCosTexture, axis=1)  # append blank slab to left
mmask = np.append(mmask, blankslab, axis=1)  # and right
# changed edge_mask color from grey, fades to black round edges which makes screen edges less visible
edge_mask = visual.GratingStim(win, mask=mmask, tex=None, contrast=1.0,
                               size=(widthPix, heightPix), units='pix', color='black')



'''Timing: expected frame duration and tolerance
with frame_tolerance_prop = .24, frame_tolerance_ms == 1ms at 240Hz, 2ms at 120Hz, 4ms at 60Hz
For a constant frame_tolerance_ms of 1ms, regardless of fps, use frame_tolerance_prop = 1/expected_fr_sec
Psychopy records frames in seconds, but I prefer to think in ms. So wo variables are labelled with _sec or _ms.
'''
expected_fr_sec = 1 / fps
expected_fr_ms = expected_fr_sec * 1000
frame_tolerance_prop = 1 / expected_fr_ms  # frame_tolerance_ms == 1ms, regardless of fps..
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
                                    "A small white target will briefly appear on screen.\n\n"
                                    "Press [i] if you see the probe moving inward (towards centre of screen),\n"
                                    "Press [o] if you see the probe moving outward (towards edge of screen).\n\n"
                                    "If you aren't sure, just guess!\n\n"
                                    "Press [Space bar] to start")


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


# # turn on high priority here. (and turn off garbage collection)
import gc
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
start_pix_per_fr = int(start_pix_per_s / fps)  # convert to pixels per frame
if start_pix_per_fr < 1:
    start_pix_per_fr = 1
if debug:
    print(f"\nstart_cm_per_s: {start_cm_per_s:.2f}cm/s, start_pix_per_s: {start_pix_per_s:.2f}pix/s, "
          f"start_pix_per_fr: {start_pix_per_fr}pix/fr")


stairStart = start_pix_per_fr
miniVal = -10
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
                          value=stairStart * -flow_dir_list[stair_idx],  # start with motion opposite to bg
                          C=stairStart * 0.6,  # initial step size, as prop of maxLum
                          minRevs=3,
                          minTrials=n_trials_per_stair,
                          minVal=miniVal,
                          maxVal=maxiVal,
                          targetThresh=0.5,
                          extraInfo=thisInfo)
    stairs.append(thisStair)


# EXPERIMENT
# trial_number = 0
# print('\n*** exp loop*** \n\n')

'''Run experiment'''
# counters
trial_num_inc_repeats = 0  # number of trials including repeated trials
trial_number = 0  # the number of the trial for the output file


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
            if debug:
                print(f"\n({trial_num_inc_repeats}) trial_number: {trial_number}, "
                      f"stair_idx: {stair_idx}, thisStair: {thisStair}, step: {step}")


            # conditions (flow_dir, prelim)
            flow_dir = thisStair.extraInfo['flow_dir']
            flow_name = thisStair.extraInfo['flow_name']
            prelim_bg_flow_ms = thisStair.extraInfo['prelim_bg_flow_ms']
            if debug:
                print(f"flow_dir: {flow_dir}, flow_name: {flow_name}, prelim_bg_flow_ms: {prelim_bg_flow_ms}")


            # probe_direction starts in opposite direction to flow_dir
            # todo: change to probe_pix_per_fr
            probeSpeed = thisStair.next()
            if debug:
                print(f"probeSpeed: {probeSpeed}, probeSpeed: {probeSpeed}")





            # timing for background motion converted to frames (e.g., 70ms is 17frames at 240Hz).
            prelim_bg_flow_fr = int(prelim_bg_flow_ms * fps / 1000)
            actual_prelim_bg_flow_ms = prelim_bg_flow_fr * 1000 / fps
            if debug:
                print(f'\nprelim_bg_flow_ms: {prelim_bg_flow_ms}')
                print(f'prelim_bg_flow_fr: {prelim_bg_flow_fr}')
                print(f'actual_prelim_bg_flow_ms: {actual_prelim_bg_flow_ms}')

            # PROBE LOCATIONS
            # corners go CCW(!) 45=top-right, 135=top-left, 225=bottom-left, 315=bottom-right
            corner = np.random.choice([45, 135, 225, 315])

            print(f'\tcorner: {corner}, flow_dir: {flow_dir}, probeSpeed: {probeSpeed}')
            # dist_from_fix is a constant giving distance form fixation,
            # dist_from_fix was previously 2 identical variables x_prob & y_prob.
            # dist_from_fix = round((tan(np.deg2rad(probe_ecc)) * viewdistPix) / sqrt(2))
            # x_prob = y_prob = round((tan(np.deg2rad(probe_ecc)) * viewdistPix) / sqrt(2))

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

                # FIXATION
                if end_fix_fr >= frameN > 0:
                    # before fixation has finished

                    # draw flow_dots but with no motion
                    flow_dots.draw()
                    probeMask1.draw()
                    probeMask2.draw()
                    probeMask3.draw()
                    probeMask4.draw()
                    edge_mask.draw()

                    fixation.draw()

                # Background motion prior to probe1
                elif end_bg_motion_fr >= frameN > end_fix_fr:
                    # after fixation, before end of background motion
                    # update dot lifetime + 1 and get  new x and y positions for dots that are re-born.
                    dotlife_array, x_array, y_array = update_dotlife(dotlife_array=dot_lifetime_array,
                                                                     dot_max_fr=dot_life_max_fr,
                                                                     x_array=x_array, y_array=y_array,
                                                                     dot_boundary=frame_size_cm)

                    # update z and scale xys
                    z_array_cm, scaled_xys = new_dots_z_and_pos(x_array=x_array, y_array=y_array, z_array=z_array_cm,
                                                                dots_speed=flow_speed_cm_p_fr, flow_dir=flow_dir,
                                                                min_z=near_plane_cm, max_z=far_plane_cm,
                                                                frame_size_cm=frame_size_cm,
                                                                reference_angle=ref_angle)
                    flow_dots.xys = scaled_xys

                    flow_dots.draw()

                    probeMask1.draw()
                    probeMask2.draw()
                    probeMask3.draw()
                    probeMask4.draw()
                    edge_mask.draw()

                    fixation.draw()

                    # reset timer to start with probe1 presentation.
                    resp.clock.reset()

                # PROBE 1
                elif end_probe_fr >= frameN > end_bg_motion_fr:
                    # after background motion, before end of probe1 interval

                    # update dot lifetime + 1 and get  new x and y positions for dots that are re-born.
                    dotlife_array, x_array, y_array = update_dotlife(dotlife_array=dot_lifetime_array,
                                                                     dot_max_fr=dot_life_max_fr,
                                                                     x_array=x_array, y_array=y_array,
                                                                     dot_boundary=frame_size_cm)

                    # update z and scale xys
                    z_array_cm, scaled_xys = new_dots_z_and_pos(x_array=x_array, y_array=y_array, z_array=z_array_cm,
                                                                dots_speed=flow_speed_cm_p_fr, flow_dir=flow_dir,
                                                                min_z=near_plane_cm, max_z=far_plane_cm,
                                                                frame_size_cm=frame_size_cm,
                                                                reference_angle=ref_angle)
                    flow_dots.xys = scaled_xys

                    flow_dots.draw()

                    probeMask1.draw()
                    probeMask2.draw()
                    probeMask3.draw()
                    probeMask4.draw()
                    edge_mask.draw()

                    fixation.draw()


                    # draw probe if 1st interval
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



                # ANSWER
                # if frameN > t_interval_2:
                elif frameN > end_probe_fr:
                    # after probe 2 interval
                    # draw flow_dots but with no motion
                    flow_dots.draw()
                    probeMask1.draw()
                    probeMask2.draw()
                    probeMask3.draw()
                    probeMask4.draw()
                    edge_mask.draw()

                    fixation.setRadius(2)
                    fixation.draw()

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


        thisExp.addData('flow_speed_cm_p_sec', flow_speed_cm_p_sec)
        thisExp.addData('flow_speed_cm_p_fr', flow_speed_cm_p_fr)
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
