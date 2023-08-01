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
from math import tan, sqrt
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


def wrap_depth_vals(depth_arr, min_depth, max_depth):
    """
    function to take an array (depth_arr) and adjust any values below min_depth
    or above max_depth with +/- (max_depth-min_depth)
    :param depth_arr: np.random.rand(nDots) array giving depth values for radial_flow dots.
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


def plot_trial_timing_errors(stim_err_ms, cond_list, fr_err_ms, save_fig_path,
                             monitor_name=None, fps=None,
                             frame_tolerance_prop=None, n_frames_per_trial=None):
    """
    Take in frame error timing data and plot it with a line plot and coloured scatter plot.
    :param stim_err_ms: array of timing errors for each trial.
    :param cond_list: A list of condition names. Must be the same length as stim_err_ms.
    :param fr_err_ms: The allowed error in frame timing.
    :param save_fig_path: Path to save fig to
    :param monitor_name: Name of monitor (optional)
    :param fps: Frame rate (optional)
    :param frame_tolerance_prop: Proportion of expected frame duration that makes fr_err_ms (optional)
    :return: figure
    """

    # check that stim_err_ms and cond_list are the same length (so all dots have a colour)
    if len(stim_err_ms) != len(cond_list):
        raise ValueError(f"stim_err_ms ({len(stim_err_ms)}) and cond_list ({len(cond_list)}) are not the same length")

    # get unique conditions for selecting colours
    unique_conds = list(set(cond_list))

    # select colour for each condition from tab20, using order shown colours_in_order
    colours_in_order = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    selected_colours = colours_in_order[:len(unique_conds)]
    my_colours = iter([plt.cm.tab20(i) for i in selected_colours])
    colour_dict = {k: v for (k, v) in zip(unique_conds, my_colours)}

    # get trial numbers for x-axis
    trial_nums = list(range(1, len(stim_err_ms) + 1))

    # plot stim_err_ms_trials as a line plot
    plt.plot(trial_nums, stim_err_ms, c='lightgrey')

    # plot stim_err_ms_trials as a scatter plot, coloured for each condition
    plt.scatter(trial_nums, stim_err_ms, c=list(map(colour_dict.get, cond_list)))

    # get legend with colours per condition
    legend_handles_list = []
    for this_cond in unique_conds:
        leg_handle = mlines.Line2D([], [], color=colour_dict[this_cond], label=this_cond,
                                   marker='o', linewidth=0, markersize=6)
        legend_handles_list.append(leg_handle)

    # horizontal line showing threshold for timing error
    plt.axhline(y=fr_err_ms, color='orange', linestyle='--')
    legend_handles_list.append(mlines.Line2D([], [], color='orange', label='error threshold',
                                             linestyle='dashed'))

    # add line for double expected length if values exceed this.
    if max(stim_err_ms) > expected_fr_ms:
        plt.axhline(y=expected_fr_ms, color='r', linestyle='-.')
        legend_handles_list.append(mlines.Line2D([], [], color='r', label='frame duration',
                                                 linestyle='dashed'))

    # plot legend
    plt.legend(handles=legend_handles_list, fontsize=6, title='conditions', framealpha=.5)

    # decorate plot
    plt.xticks(trial_nums)
    plt.ylabel('Trial error (ms)')
    plt.xlabel('Trial number')

    # count the number of values in stim_err_ms that are greater than fr_err_ms
    n_bad = sum([1 for x in stim_err_ms if x > fr_err_ms])

    # make title text for fig
    title_text = f'{n_bad}/{trial_nums[-1]} trials with bad timing'
    if monitor_name is not None and fps is not None:
        title_text = title_text + f'{monitor_name} {fps}Hz. '
    if frame_tolerance_prop is not None and fr_err_ms is not None:
        title_text = title_text + f'\n(orange dashed line is error threshold of {frame_tolerance_prop * 100}%={round(fr_err_ms, 2)}ms))'
    if n_frames_per_trial is not None:
        title_text = title_text + f'\n1 trial = {n_frames_per_trial} frames'
    plt.title(title_text)

    plt.tight_layout()

    plt.savefig(save_fig_path)

    # plt.close()

    return plt.gcf()


# Ensure that relative paths start from the same directory as this script
_thisDir = path.dirname(path.abspath(__file__))
chdir(_thisDir)

# todo: uses ASUS_2_13_240Hz for replicating old results, but then use asus_cal for testing.

# Store info about the experiment session (numbers keep the order)
expName = 'rad_flow_23_rings'   # from the Builder filename that created this script
expInfo = {'1. Participant': 'Nick_test_31072023',
           '2. Run_number': '1',
           '3. Probe duration in frames': [2, 1, 50, 100],
           '4. fps': [60, 240, 120, 60],
           '5. ISI_dur_in_ms': [25, 16.67, 100, 50, 41.67, 37.5, 33.34, 25, 16.67, 8.33, 0, -1],
           '6. Probe_orientation': ['radial', 'tangent'],
           '7. Vary_fixation': [True, False],
           '8. Record_frame_durs': [True, False],
           '9. Background': ['flow_rings', 'flow_dots', None],
           # '10. bg_speed_cond': ['Normal', 'Half-speed'],
           '11. prelim_bg_flow_ms': [200, 350, 200, 70],
           '12. monitor_name': ['Nick_work_laptop', 'OLED', 'asus_cal', 'Samsung',
                                'Asus_VG24', 'HP_24uh', 'NickMac', 'Iiyama_2_18', 'ASUS_2_13_240Hz'],
           '13. mask_type': ['4_circles', '2_spokes'],
           '14. verbose': [False, True]
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
# bg_speed_cond = expInfo['10. bg_speed_cond']
prelim_bg_flow_ms = int(expInfo['11. prelim_bg_flow_ms'])
monitor_name = expInfo['12. monitor_name']
mask_type = expInfo['13. mask_type']
verbose = eval(expInfo['14. verbose'])


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
# todo: change this so prelim_bg_flow_ms are different folders within the participant directory.
# FILENAME: join participant_name with prelim_bg_flow_ms to keep different prelim values separate
# participant_name = participant_name + f'_bg{prelim_bg_flow_ms}'

# save each participant's files into separate dir for each ISI
isi_dir = f'ISI_{ISI}'
# save_dir = path.join(_thisDir, expName, participant_name,
#                         f'{participant_name}_{run_number}', isi_dir)
save_dir = path.join(_thisDir, expName, participant_name, f'_bg{prelim_bg_flow_ms}',
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
    dots_speed = 0.2
    if monitor_name == 'OLED':
        dots_speed = 0.4
    BGspeed = dots_speed
    # todo: do we need to increase the number of dots for OLED?
    nDots = 10000
    dot_array_width = 10000  # original script used 5000
    dots_min_depth = 0.5  # depth values
    dots_max_depth = 5  # depth values

    # initial array values
    x = np.random.rand(nDots) * dot_array_width - dot_array_width / 2
    y = np.random.rand(nDots) * dot_array_width - dot_array_width / 2
    z = np.random.rand(nDots) * (dots_max_depth - dots_min_depth) + dots_min_depth
    # print(f"x: {x}, y: {y}, z: {z}")

    x_flow = x / z
    y_flow = y / z

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

    thisInfo = copy.copy(expInfo)
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
    print('\n*** exp loop*** \n\n')
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
            # todo: use multiple assignments here.
            # ISI = ISI_vals_list[stair_idx]
            # congruent = cong_vals_list[stair_idx]
            # cong_name = cong_names_list[stair_idx]
            ISI, sep = ISI_vals_list[stair_idx], sep_vals_list[stair_idx]
            congruent, cong_name = cong_vals_list[stair_idx], cong_names_list[stair_idx]
            if verbose:
                print(f"ISI: {ISI}, congruent: {congruent} ({cong_name})")

            # conditions (sep, sep_deg, neg_sep)
            # sep = sep_vals_list[stair_idx]
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
            '''
            x and y
            np.random.rand(nDots) = array of floats (between 0 and 1).
            dot_array_width = 10000
            np.random.rand(nDots) * dot_array_width = array of floats (between 0 and 10000 e.g., dot_array_width). 
            np.random.rand(nDots) * dot_array_width - dot_array_width / 2 = array of floats (between -dot_array_width/2 and dot_array_width/2).
            e.g., between -5000 and 5000
            
            z
            dots_max_depth, dots_min_depth = 5, .5
            np.random.rand(nDots) * (dots_max_depth - dots_min_depth) + dots_min_depth = array of floats (between dots_min_depth and dots_max_depth).
            e.g., floats are multiplied by 4.5 (giving values in the range 0 to 4.5, then .5 is added, giving values in the range .5 to 5).
            
            x_flow = x / z
            this is an (fairly) normally distributed array of floats (between -10000 and 10000) divided by an array of floats (between .5 and 5).
            max x value of 5000 is 10000 if divided by .5, and 1000 if divided by 5.
            So there is a cluster between -1000 and 1000.
            
            # later, (in per frame section), zs are updated with z = z + dots_speed * flow_dir
            dots_speed is currently set to .2.  so zs are updated by adding either .2 or -.2.
            on the first update, xs are divided by new zs which are in range .7 to 5.2.  
            max x values of 5000 is 7142 if divided by .7, and 961 if divided by 5.2.
            
            the next updated, xs are divided by new zs which are in range .9 to 5.4.
            max x values of 5000 is 5555 if divided by .9, and 925 if divided by 5.4.
            
            '''
            # if background == 'flow_dots':
                # x = np.random.rand(nDots) * dot_array_width - dot_array_width / 2
                # y = np.random.rand(nDots) * dot_array_width - dot_array_width / 2
                # z = np.random.rand(nDots) * (dots_max_depth - dots_min_depth) + dots_min_depth
                # # print(f"x: {x}, y: {y}, z: {z}")
                # print(f"dot_array_width: {dot_array_width}, dots_max_depth: {dots_max_depth}, dots_min_depth: {dots_min_depth}")
                # print(f"(dots_max_depth - dots_min_depth) + dots_min_depth: {(dots_max_depth - dots_min_depth) + dots_min_depth}")
                # z was called z_flow but is actually z position like x and y
                # x_flow = x / z
                # y_flow = y / z
                # print(f"x_flow: {x_flow}, y_flow: {y_flow}")


            # shift probes by separation
            # todo: turn shifted pixels into a function?
            '''Both probes should be equally spaced around the meridian point.
            The original script had probe 1 on meridian and probe 2 shifted by separation.
            So now they both should be shifted by half the separation away from meridian in opposite directions.
            E.g., if sep = 4, probe 1 will be shifted 2 pixels away from meridian in one direction 
            probe 2 will be shifted 2 pixels away from the meridian in the other direction. 
            Where separation is an odd number, both probes are shift by half sep, 
            then the extra pixel is added onto either probe 1 r probe 2.  
            E.g., if sep = 5, either probe1 shifts by 2 and probe 2 by 3, or vice versa. 
            To check probe locations, uncomment loc_marker'''
            if sep == 99:
                p1_shift = p2_shift = 0
            elif sep % 2 == 0:  # even number
                p1_shift = p2_shift = (sep*probe_size) // 2
            else:  # odd number: shift by half sep, then either add 1 or 0 extra pixel to the shift.
                extra_shifted_pixel = [0, 1]
                np.random.shuffle(extra_shifted_pixel)
                p1_shift = (sep*probe_size) // 2 + extra_shifted_pixel[0]
                p2_shift = (sep*probe_size) // 2 + extra_shifted_pixel[1]
            if verbose:
                print(f"p1_shift: {p1_shift}, p2_shift: {p2_shift}")


            # set position and orientation of probes
            # todo: turn probe positions and orientation into a function?
            '''NEW - set orientations to p1=zero and p2=180 (not zero), 
            then add the same orientation change to both'''
            probe1_ori = 0
            probe2_ori = 180
            if probe_n_pixels == 7:
                probe1_ori = 180
                probe2_ori = 0
            if corner == 45:  # top right
                '''in top-right corner, both x and y increase (right and up)'''
                loc_x = dist_from_fix * 1
                loc_y = dist_from_fix * 1
                '''orientation' here refers to the relationship between probes, 
                whereas probe1_ori refers to rotational angle of probe stimulus'''
                if orientation == 'tangent':
                    if target_jump == 1:  # CW
                        probe1_ori += 180
                        probe2_ori += 180
                        probe1_pos = [loc_x - p1_shift, loc_y + p1_shift]
                        probe2_pos = [loc_x + p2_shift - probe_size, loc_y - p2_shift]
                    elif target_jump == -1:  # ACW
                        probe1_ori += 0
                        probe2_ori += 0
                        probe1_pos = [loc_x + p1_shift, loc_y - p1_shift]
                        probe2_pos = [loc_x - p2_shift + probe_size, loc_y + p2_shift]
                elif orientation == 'radial':
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
                if orientation == 'tangent':
                    if target_jump == 1:  # ACW
                        probe1_ori += 90
                        probe2_ori += 90
                        probe1_pos = [loc_x - p1_shift, loc_y - p1_shift]
                        probe2_pos = [loc_x + p2_shift - probe_size, loc_y + p2_shift]
                    elif target_jump == -1:  # CW
                        probe1_ori += 270
                        probe2_ori += 270
                        probe1_pos = [loc_x + p1_shift, loc_y + p1_shift]
                        probe2_pos = [loc_x - p2_shift + probe_size, loc_y - p2_shift]
                elif orientation == 'radial':
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
                if orientation == 'tangent':
                    if target_jump == 1:  # CW
                        probe1_ori += 0
                        probe2_ori += 0
                        probe1_pos = [loc_x + p1_shift, loc_y - p1_shift]
                        probe2_pos = [loc_x - p2_shift + probe_size, loc_y + p2_shift]
                    elif target_jump == -1:  # ACW
                        probe1_ori += 180
                        probe2_ori += 180
                        probe1_pos = [loc_x - p1_shift, loc_y + p1_shift]
                        probe2_pos = [loc_x + p2_shift - probe_size, loc_y - p2_shift]
                elif orientation == 'radial':
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
                if orientation == 'tangent':
                    if target_jump == 1:  # ACW
                        probe1_ori += 270
                        probe2_ori += 270
                        probe1_pos = [loc_x + p1_shift, loc_y + p1_shift]
                        probe2_pos = [loc_x - p2_shift + probe_size, loc_y - p2_shift]
                    elif target_jump == -1:  # CW
                        probe1_ori += 90
                        probe2_ori += 90
                        probe1_pos = [loc_x - p1_shift, loc_y - p1_shift]
                        probe2_pos = [loc_x + p2_shift - probe_size, loc_y + p2_shift]
                elif orientation == 'radial':
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

            # loc_marker.setPos([loc_x, loc_y])
            probe1.setPos(probe1_pos)
            probe1.setOri(probe1_ori)
            probe2.setPos(probe2_pos)
            probe2.setOri(probe2_ori)
            if verbose:
                print(f"loc_marker: {[loc_x, loc_y]}, probe1_pos: {probe1_pos}, "
                      f"probe2_pos: {probe2_pos}. dff: {dist_from_fix}")


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
            # if (trial_number % take_break == 1) & (trial_number > 1):
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
                            flow_dots.xys = np.array([x_flow, y_flow]).transpose()
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
                            z = z + dots_speed * flow_dir
                            z = wrap_depth_vals(z, dots_min_depth, dots_max_depth)
                            x_flow = x / z
                            y_flow = y / z
                            flow_dots.xys = np.array([x_flow, y_flow]).transpose()
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
                            z = z + dots_speed * flow_dir
                            z = wrap_depth_vals(z, dots_min_depth, dots_max_depth)
                            x_flow = x / z
                            y_flow = y / z
                            flow_dots.xys = np.array([x_flow, y_flow]).transpose()
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
                            # draw dots with motion
                            z = z + dots_speed * flow_dir
                            z = wrap_depth_vals(z, dots_min_depth, dots_max_depth)
                            x_flow = x / z
                            y_flow = y / z
                            flow_dots.xys = np.array([x_flow, y_flow]).transpose()
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
                            # draw dots with motion
                            z = z + dots_speed * flow_dir
                            z = wrap_depth_vals(z, dots_min_depth, dots_max_depth)
                            x_flow = x / z
                            y_flow = y / z
                            flow_dots.xys = np.array([x_flow, y_flow]).transpose()
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

# todo: convert plot_frame_intervals into a function.
# plot frame intervals
# if record_fr_durs:
#
#     # flatten list of lists (fr_int_per_trial) to get len
#     all_fr_intervals = [val for sublist in fr_int_per_trial for val in sublist]
#     total_recorded_fr = len(all_fr_intervals)
#
#     print(f"{dropped_fr_trial_counter}/{total_n_trials} trials with bad timing "
#           f"(expected: {round(expected_fr_ms, 2)}ms, "
#           f"frame_tolerance_ms: +/- {round(frame_tolerance_ms, 2)})")
#
#     '''set colours for lines on plot.'''
#
#     from exp1a_psignifit_analysis import fig_colours
#     my_colours = fig_colours(n_stairs, alternative_colours=False)
#
#     # associate colours with conditions
#     colour_dict = {k: v for (k, v) in zip(stair_names_list, my_colours)}
#     # make list of colours based on order of conditions
#     cond_colour_list = [colour_dict[i] for i in cond_list]
#
#     # todo: delete line below
#     print(f"fr_counter_per_trial: {fr_counter_per_trial}")
#
#
#     # plot frame intervals across the experiment with discontinuous line, coloured for each cond
#     for trial_x_vals, trial_fr_durs, colour in zip(fr_counter_per_trial, fr_int_per_trial, cond_colour_list):
#         plt.plot(trial_x_vals, trial_fr_durs, color=colour)
#
#     # add legend with colours per condition
#     legend_handles_list = []
#     for cond in stair_names_list:
#         leg_handle = mlines.Line2D([], [], color=colour_dict[cond], label=cond,
#                                    marker='.', linewidth=.5, markersize=4)
#         legend_handles_list.append(leg_handle)
#
#     plt.legend(handles=legend_handles_list, fontsize=6, title='conditions', framealpha=.5)
#
#     # add vertical lines to signify trials, shifted back so trials fall between lines
#     fr_v_lines = [i - .5 for i in exp_n_fr_recorded_list]
#     for trial_line in fr_v_lines:
#         plt.axvline(x=trial_line, color='silver', linestyle='dashed', zorder=0)
#
#     # add horizontal lines: green = expected frame duration, red = frame error tolerance
#     plt.axhline(y=expected_fr_sec, color='grey', linestyle='dotted', alpha=.5)
#     plt.axhline(y=max_fr_dur_sec, color='red', linestyle='dotted', alpha=.5)
#     plt.axhline(y=min_fr_dur_sec, color='red', linestyle='dotted', alpha=.5)
#
#     # shade trials that were repeated: red = bad timing, orange = user repeat
#     for loc_pair in dropped_fr_trial_x_locs:
#         print(loc_pair)
#         x0, x1 = loc_pair[0] - .5, loc_pair[1] - .5
#         plt.axvspan(x0, x1, color='red', alpha=0.15, zorder=0, linewidth=None)
#
#     plt.title(f"{monitor_name}, {fps}Hz, {expInfo['date']}\n{dropped_fr_trial_counter}/{total_n_trials} trials."
#               f"dropped fr (expected: {round(expected_fr_ms, 2)}ms, "
#               f"frame_tolerance_ms: +/- {round(frame_tolerance_ms, 2)})")
#     fig_name = f'{participant_name}_{run_number}_frames.png'
#     print(f"fig_name: {fig_name}")
#     plt.savefig(path.join(save_dir, fig_name))
#     plt.close()


# todo: this function is for plotting error per trial (p2_stop-p1-start) not frame intervals
# if record_fr_durs:
#     # if all values in fr_per_trial are the same, then n_frames_per_trial equals the first value, else raise an error
#     if all(x == fr_per_trial[0] for x in fr_per_trial):
#         n_frames_per_trial = fr_per_trial[0]
#     else:
#         raise ValueError("not all values in fr_per_trial are the same")
#     print(f"n_frames_per_trial: {n_frames_per_trial}")
#
#     timing_fig = plot_trial_timing_errors(stim_err_ms=stim_err_ms, cond_list=cond_list, fr_err_ms=fr_err_ms,
#                                           monitor_name=monitor_name, fps=fps, frame_tolerance_prop=frame_tolerance_prop,
#                                           n_frames_per_trial=n_frames_per_trial, save_fig_path=save_fig_path)

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
