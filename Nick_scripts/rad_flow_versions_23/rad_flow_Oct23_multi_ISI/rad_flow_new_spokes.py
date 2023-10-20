from __future__ import division
from psychopy import gui, visual, core, data, event, monitors, logging
from psychopy import __version__ as psychopy_version
from psychopy.tools.monitorunittools import cm2pix, pix2cm
from datetime import datetime
from os import path, chdir
from copy import copy
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
# for numpy attributes access by per-frame functions, acceess them with name instead of np.name.
from numpy import array, random, where, sum, linspace, pi, rad2deg, arctan, arctan2, cos, sin, hypot

from kestenSTmaxVal import Staircase
from PsychoPy_tools import get_pixel_mm_deg_values


# import copy
import gc
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
- add congruent and incongrent, get sep_list, ISI_vals, cong_list to make stair_names_list  DONE

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

rad_flow_7_spokes.py
- add in setting for OLED (bgLumProp, startLum etc) - DONE
- add in spokes as option to compare with 4CircleMasks - Done
    Its a bit fudged at the moment, hard coded as 22.5 deg spokes.
- add smaller probes on OLED?  - DONE
- set it for 'asus_cal', not uncalibrated monitor.  - DONE

rad_flow_8_prelim_interleaved.py
- add in interleaved prelim period (with bg motion)  - DONE
- changed prelim_dir to 'interleaved' - DONE

rad_flow_new_spokes.py
- added in new function for dots with spokes from flow_parse exp scripts DONE
- changed imports to minimise attribute access DONE
- removed variables from dlg - now all hard coded values DONE
     background - always flow_dots
     orientation - always radial
     record frame ints - always True
     prelim - has multiple prelims in staircase conds
- alternate fixation colours each trial DONE
- removed 'interleaved' dir from file structure and added monitor DONE

"""

"""
To use this script you will need the width (cm), screen dims (pixels, width heght) and view dist for you
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
#
#
# def new_dots_z_and_pos(x_array, y_array, z_array, dots_speed, flow_dir, min_z, max_z):
#     """
#     This is a function to update flow_dots depth array and get new pixel co-ordinates
#     using the original x_array and y_array.
#
#     1a. Update z_array by adding dots_speed * flow_dir to the current z values.
#     1b. adjust any values below dots_min_z or above dots_max_z.
#
#     2a. Get new x_pos and y_pos co-ordinates values by dividing x_array and y_array by the new z_array.
#     2b. put the new x_pos and y_pos co-ordinates into an array and transposes it.
#
#     :param x_array: Original x_array positions for the dots (shape = (n_dots, 1))
#     :param y_array: Original y_array positions for the dots (shape = (n_dots, 1))
#     :param z_array: array of depth values for the dots (shape = (n_dots, 1))
#     :param dots_speed: speed of the dots (float, smaller = slower, larger = faster)
#     :param flow_dir: either 1 (contracting/inward/backwards) or -1 (expanding/outward/forwards)
#     :param dots_min_z: default is .5, values below this are adjusted to dots_max_z
#     :param dots_max_z: default is 5, values above this are adjusted to dots_min_z
#     :return: new dots_pos_array
#     """
#
#     # # 1. Update z (depth values) # #
#     # Add dots_speed * flow_dir to the current z values.
#     updated_z_array = z_array + dots_speed * flow_dir
#
#     # adjust any depth values below min_z or above max_z by z_adjust
#     z_adjust = max_z - min_z
#     # adjust updated_z_array values less than min_z by adding z_adjust
#     less_than_min = (updated_z_array < min_z)
#     updated_z_array[less_than_min] += z_adjust
#     # adjust updated_z_array values more than max_z by subtracting z_adjust
#     more_than_max = (updated_z_array > max_z)
#     updated_z_array[more_than_max] -= z_adjust
#     # print(f"updated_z_array (clipped):\n{updated_z_array}\n")
#
#     # # 2. Get new pixel co-ordinates for dots using original x_array and y_array and updated_z_array # #
#     x_pos = x_array / updated_z_array
#     y_pos = y_array / updated_z_array
#
#     # puts the new co-ordinates into an array and transposes it, ready to use.
#     dots_pos_array = np.array([x_pos, y_pos]).T
#
#     return updated_z_array, dots_pos_array
#
#
# def roll_rings_z_and_colours(z_array, ring_colours, min_z, max_z, flow_dir, flow_speed, initial_x_vals):
#     """
#     This rings will spawn a new ring if the old one either grows too big for the screen (expanding),
#     or shrinks too small (if contracting).
#
#     This function updates the z_array (depth) values for the rings, and adjusts any values below min_z or
#     above max_z by z_adjust.  Any values that are adjusted are then rolled to the end or beginning of the array,
#     depending on whether they are below min_z or above max_z.
#     The same values are then also rolled in the ring_colours array.
#
#     :param z_array: Numpy array of z_array values for the rings (shape = (n_rings, 1))
#     :param ring_colours: List of RGB1 colours for the rings (shape = (n_rings, 3))
#     :param min_z: minimum depth value for the rings (how close they can get to the screen)
#     :param max_z: maximum depth value for the rings (how far away they can get from the screen)
#     :param flow_dir: either 1 (contracting/inward/backwards) or -1 (expanding/outward/forwards)
#     :param flow_speed: speed of the rings (float, smaller = slower, larger = faster)
#     :param initial_x_vals: nupmy array of ring sizes, (all the same size, e.g., 1080, shape = (n_rings, 1))
#
#     :return: z_array (updated), ring_radii_array, ring_colours (rolled if any z_array values are rolled)
#     """
#
#     # update depth values
#     z_array = z_array + flow_speed * flow_dir
#
#     # z_adjust is the size of change to make to out-of-bounds rings
#     z_adjust = max_z - min_z
#
#     # adjust any z_array values below min_z or above max_z by z_adjust
#     if flow_dir == -1:  # expanding, getting closer, might be below min_z
#         # find which rings are less than min and add z_adjust to those rings
#         less_than_min = (z_array < min_z)
#         z_array[less_than_min] += z_adjust
#
#         # shift arrays by this amount (e.g., if 3 rings are less than min, shift by 3)
#         # (note negative shift to move them backwards)
#         shift_num = -sum(less_than_min)
#
#     elif flow_dir == 1:  # contracting, getting further away, might be above max_z
#         # find which rings are more_than_max and subtract z_adjust to those rings
#         more_than_max = (z_array > max_z)
#         z_array[more_than_max] -= z_adjust
#
#         # shift arrays by this amount (e.g., if 3 rings are more_than_max, shift by 3)
#         shift_num = sum(more_than_max)
#
#     # roll the depth and colours arrays so that adjusted rings move to other end of array
#     z_array = np.roll(z_array, shift=shift_num, axis=0)
#     ring_colours = np.roll(ring_colours, shift=shift_num, axis=0)
#
#     # get new ring_radii_array
#     ring_radii_array = initial_x_vals / z_array
#
#     # print(f"\nz_array:\n{z_array}\nring_radii_array:\n{ring_radii_array}\nshift_num:\n{shift_num}\n")
#
#     return z_array, ring_radii_array, ring_colours



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


# # # DIALOGUE BOX # # #

# dialogue box/drop-down option when exp starts (1st item is default val)
expInfo = {'01. Participant': 'scriptTest_18102023',
           '02. Run_number': '1',
           '03. Probe duration in frames': [2, 1, 50, 100],
           '04. fps': [60, 240, 120, 60],
           '05. ISI_dur_in_ms': [33.34, 100, 50, 41.67, 37.5, 33.34, 25, 16.67, 8.33, 0, -1],
           # '06. Probe_orientation': ['radial', 'tangent'],
           # '07. Record_frame_durs': [True, False],
           # '08. Background': ['flow_dots', 'no_bg'],  # no 'flow_rings', as it's not implemented here
           # '09. prelim_ms': [70, 200, 350, 2000],
           '10. monitor_name': ['Nick_work_laptop', 'OLED', 'asus_cal', 'ASUS_2_13_240Hz',
                                'Samsung', 'Asus_VG24', 'HP_24uh', 'NickMac', 'Iiyama_2_18'],
           # '11. mask_type': ['4_circles', '2_spokes'],
           '12. debug': [False, True]
           }

# run drop-down menu, OK continues, cancel quits
dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
if not dlg.OK:
    core.quit()  # user pressed cancel

# Dialogue box settings
participant_name = expInfo['01. Participant']
run_number = int(expInfo['02. Run_number'])
probe_duration = int(expInfo['03. Probe duration in frames'])
fps = int(expInfo['04. fps'])
ISI_selected_ms = float(expInfo['05. ISI_dur_in_ms'])
# background = expInfo['08. Background']  # Always have flow dots on this script
# prelim_ms = int(expInfo['09. prelim_ms'])  # multiple prelims conds hard coded below
monitor_name = expInfo['10. monitor_name']
# mask_type = expInfo['11. mask_type']  # always have spokes in this script
debug = eval(expInfo['12. debug'])

# print settings from dlg
print("\ndlg dict")
for k, v in expInfo.items():
    print(f'{k}: {v}')


# # # MISC SETTINGS # # #
n_trials_per_stair = 25  # this is the number of trials per stair
if debug:
    n_trials_per_stair = 2
probe_ecc = 4  # int((expInfo['6. Probe eccentricity in deg']))1
vary_fixation = True  # vary fixation period between .5 and 1.5 seconds.
record_fr_durs = True  # eval(expInfo['07. Record_frame_durs'])
orientation = 'radial'  # expInfo['06. Probe_orientation']  # could add tangent back in
expInfo['date'] = datetime.now().strftime("%d/%m/%Y")
expInfo['time'] = datetime.now().strftime("%H:%M:%S")


# # # EXPERIMENT HANDLING AND SAVING # # #
# todo: currently 450 trials (3 (sep) x (3 (prelim) x 2 (cong).  If any increase I'll need to change file structure with dirs for prelim or sep
# save each participant's files into separate dir for each ISI
save_dir = path.join(_thisDir, expName, monitor_name,  # added monitor name to analysis structure
                     participant_name,
                     # background, 'interleaved',  # always use dots background and interleave prelims
                     f'{participant_name}_{run_number}',  # don't use p_name_run here, as it's not a separate folder
                     f'ISI_{int(ISI_selected_ms)}')  # I've changed this to int(ms) not frames, for easier comparision of monitors
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
# ISI timing in ms and frames - only one per run, so not included in staircases
'''ISI can be given (roughly) in ms, for any monitor it will try to match that value in frames.
The script uses on frames so ISI will be used in that format.
The actual ms timing is given for record.
This means that the experiment should have similar ms timings on monitors with different fps
milliseconds: [100, 50, 41.66, 37.5, 33.34, 25, 16.67, 8.33, 0]
frames@240hz: [24,  12,  10,    9,    8,     6,  4,    2,    0]
frames@120hz: [12,   6,   5,     ,    4,     3,  2,    1,    0]
frames@60hz:  [ 6,   3,    ,     ,    2,      ,  1,     ,    0]
'''
if ISI_selected_ms == -1:  # concurrent/simultaneous probes
    ISI_cond_fr = -1
    ISI_actual_ms = -1
else:
    ISI_cond_fr = int(ISI_selected_ms * fps / 1000)
    ISI_actual_ms = (1 / fps) * ISI_cond_fr * 1000
if debug:
    print(f"\nSelected {ISI_selected_ms}ms ISI.\n"
          f"At {fps}Hz this is {ISI_cond_fr} frames, which each take {round(1000 / fps, 2)} ms.\n"
          f"ISI_cond_fr: {ISI_cond_fr}")

# # Conditions/staricases: Separation, Congruence (cong, incong) x prelim motion (0, 70, 350)

# Separation values in pixels.  select from [18, 6, 3, 2, 1, 0] or 99 for 1probe
# sep_vals = [18, 6, 3, 2, 1, 0]
sep_vals = [6, 3, 1]
if debug:
    sep_vals = [0, 1]

# # main contrast is whether the background and target motion is in same or opposite direction.
# congruence_vals: 1=congruent/same, -1=incongruent/different
congruence_vals = [1, -1]
congruence_names = ['cong', 'incong']
cong_zip = list(zip(congruence_names, congruence_vals))
# if background == 'no_bg':
#     congruence_vals = [1]
#     congruence_names = ['No_bg']
# if debug:
#     print(f'congruence_vals: {congruence_vals}')
#     print(f'congruence_names: {congruence_names}')
#     print(f"cong_zip: {cong_zip}")


# 'prelim' (preliminary motion) is how long (ms) the background motion starts before the probe appears
prelim_vals = [0, 70, 350]
if debug:
    prelim_vals = [500]

# lists of values for each condition (all list are same length = n_stairs)
'''each separation value appears in 4 stairs, e.g.,
flow_dir (expand/contract) x prelim (70, 350)
 e.g., if sep_vals = [6, 3, 1], sep_conds_list = [6, 6, 6, 6, 3, 3, 3, 3, 1, 1, 1, 1]
'''
#
# sep_conds_list = list(np.repeat(sep_vals, len(congruence_vals) * len(prelim_vals)))
# cong_val_conds_list = list(np.tile(np.repeat(congruence_vals, len(prelim_vals)), len(sep_vals)))
# cong_name_conds_list = list(np.tile(np.repeat(congruence_names, len(prelim_vals)), len(sep_vals)))
# prelim_conds_list = list(np.tile(prelim_vals, len(sep_vals) * len(congruence_vals)))
# ISI_conds_list = list(np.repeat(ISI_vals, len(sep_vals) * len(congruence_vals) * len(prelim_vals)))
#
# # stair_names_list joins sep_conds_list, cong_name_conds_list and prelim_conds_list
# # e.g., ['sep_6_cong_1_prelim_70', 'sep_6_cong_1_prelim_350', 'sep_6_cong_-1_prelim_70'...]
# stair_names_list = [f"sep_{s}_cong_{c}_prelim_{p}" for s, c, p in zip(sep_conds_list, cong_val_conds_list, prelim_conds_list)]
# print("\n4")
# print(f'ISI_conds_list: {ISI_conds_list}')
# print(f'sep_conds_list: {sep_conds_list}')
# print(f'cong_val_conds_list: {cong_val_conds_list}')
# print(f'cong_name_conds_list: {cong_name_conds_list}')
# print(f'prelim_conds_list: {prelim_conds_list}')
# print(f'\nstair_names_list: {stair_names_list}')



'''New method'''
# get all possible combinations of these three lists
combined_conds = [(s, cz, p) for s in sep_vals for cz in cong_zip for p in prelim_vals]
# print(f"combined_conds: {combined_conds}")

# split the combined_conds into separate lists
sep_conds_list = [i[0] for i in combined_conds]
cong_name_conds_list = [i[1][0] for i in combined_conds]
cong_val_conds_list = [i[1][1] for i in combined_conds]
prelim_conds_list = [i[2] for i in combined_conds]

# # get all possible combinations of these three lists
# combined_conds = [(s, c, p) for s in sep_vals for c in congruence_vals for p in prelim_vals]
# print(f"combined_conds: {combined_conds}")
#
# # split the combined_conds into separate lists
# sep_conds_list = [i[0] for i in combined_conds]
# cong_val_conds_list = [i[1] for i in combined_conds]
# cong_name_conds_list = [congruence_names[0] if i[1] == 1 else congruence_names[1] for i in combined_conds]
# prelim_conds_list = [i[2] for i in combined_conds]
#
# # # ISI conds list is a bit redundant while there is only one ISI per run, but I'll keep it for now.
# # ISI_conds_list = [ISI_vals[0] for i in combined_conds]
#
# # stair_names_list joins sep_conds_list, cong_zip and prelim_conds_list
# # e.g., ['sep_6_cong_1_prelim_70', 'sep_6_cong_1_prelim_350', 'sep_6_cong_-1_prelim_70'...]
# stair_names_list = [f"sep_{s}_cong_{c}_prelim_{p}" for s, c, p in combined_conds]
stair_names_list = [f"sep_{s}_{cz[0]}_{cz[1]}_prelim_{p}" for s, cz, p in combined_conds]


# print("\n5. zip then unpack")
# print(f'ISI_conds_list: {ISI_conds_list}')
# print(f'sep_conds_list: {sep_conds_list}')
# print(f'cong_val_conds_list: {cong_val_conds_list}')
# print(f'cong_name_conds_list: {cong_name_conds_list}')
# print(f'prelim_conds_list: {prelim_conds_list}')
# print(f'\nstair_names_list: {stair_names_list}')



if debug:
    # print(f'ISI_conds_list: {ISI_conds_list}')
    print(f'sep_conds_list: {sep_conds_list}')
    print(f'cong_val_conds_list: {cong_val_conds_list}')
    print(f'cong_name_conds_list: {cong_name_conds_list}')
    print(f'prelim_conds_list: {prelim_conds_list}')



n_stairs = len(sep_conds_list)
total_n_trials = int(n_trials_per_stair * n_stairs)
print(f'\nstair_names_list: {stair_names_list}')
print(f'n_stairs: {n_stairs}')
print(f'total_n_trials: {total_n_trials}')




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

# COLORS AND LUMINANCES
# # Lum to Color255
# LumColor255Factor = 2.39538706913372
# # Color255 to Color1
# Color255Color1Factor = 1 / 127.5  # Color255 * Color255Color1Factor -1
# # Lum to Color1
# Color1LumFactor = 2.39538706913372  ###
#
# maxLum = 106  # 255 RGB
# # minLum = 0.12  # 0 RGB  # this is currently unused
# bgLumProp = .2  # .2  # use .45 to match radial_flow_NM_v2.py, or .2 to match exp1
# if monitor_name == 'OLED':
#     bgLumProp = .0
# bgLum = maxLum * bgLumProp
#
# # colour space
# this_colourSpace = 'rgb1'  # values between 0 and 1
# bgColor_rgb1 = bgLum / maxLum
# this_bgColour = [bgColor_rgb1, bgColor_rgb1, bgColor_rgb1]
#
# # Flow colours
# adj_flow_colour = .15
# # Give dots a pale green colour, which is adj_flow_colour different to the background
# flow_colour = [this_bgColour[0] - adj_flow_colour, this_bgColour[1], this_bgColour[2] - adj_flow_colour]
# if monitor_name == 'OLED':  # darker green for low contrast against black background
#     flow_colour = [this_bgColour[0], this_bgColour[1] + adj_flow_colour / 2, this_bgColour[2]]
# print(f"\nthis_bgColour: {this_bgColour}")
# print(f"flow_colour: {flow_colour}")

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

# pixel size
pixel_mm_deg_dict = get_pixel_mm_deg_values(monitor_name=monitor_name)
if debug:
    print(f"diagonal pixel size: {pixel_mm_deg_dict['diag_mm']} mm, or {pixel_mm_deg_dict['diag_deg']} dva")


# # # PSYCHOPY COMPONENTS # # #
# MOUSE
myMouse = event.Mouse(visible=False)

# # KEYBOARD
resp = event.BuilderKeyResponse()


# fixation bull eye
fixation = visual.Circle(win, radius=2, units='pix', lineColor='white', fillColor='black', colorSpace=this_colourSpace)

# add a small blurred mask behind fixation so dots are separated from fxation and less dirstracting
fix_mask_size = 50
# Create a raisedCosine mask array and assign it to a Grating stimulus (grey outside, transparent inside)
raisedCosTexture1 = visual.filters.makeMask(256, shape='raisedCosine',
                                            # fringeWidth=0.3,
                                            fringeWidth=0.9,  # proportion of mask that is blured (0 to 1)
                                            radius=[1.0, 1.0])
fix_mask = visual.GratingStim(win=win, mask=raisedCosTexture1, size=(fix_mask_size, fix_mask_size),
                                colorSpace=this_colourSpace,
                                color=this_bgColour,
                                # color='red', # for testing
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

probe1 = visual.ShapeStim(win, vertices=probeVert, lineWidth=0, opacity=1, size=probe_size, interpolate=False,
                          colorSpace=this_colourSpace)
probe2 = visual.ShapeStim(win, vertices=probeVert, lineWidth=0, opacity=1, size=probe_size, interpolate=False,
                          colorSpace=this_colourSpace)

# probes and probe_masks are at dist_from_fix pixels from middle of the screen
dist_from_fix = int((np.tan(np.deg2rad(probe_ecc)) * view_dist_pix) / np.sqrt(2))


# # flow speed should be scaled by fps, so dots have a greater change per frame on slower monitors.
# # e.g., .2 at 240Hz, .4 at 120Hz and .8 at 60Hz.
# # this appears too fast to me, but it is the same as the original script.
# flow_speed = 48 / fps
#
#
# # flow_dots - e.g., moving background to simulate self motion
# if background == 'flow_dots':
#
#     # If False, use orginal settings, if True, increase dots depth and scale their size with depth
#     deep_with_sizes = True  # False, True
#
#     dots_speed = flow_speed
#     flow_speed = dots_speed
#
#     # Changing dots_min_z from .5 to one means that the proportion of dots onscreen increases from ~42% to ~82%.
#     # Therefore, I can half n_dots with little change in the number of dots onscreen, saving processing resources.
#     # Note: 'onscreen' was defined as half widthPix (960).  Once the edge mask is added, the square of the visible screen is 1080x1080,
#     # minus the blurred edges, so 960 seems reasonable.
#     dots_min_z = 1.0  # original script used .5, which increased the tails meaning more dots were offscreen.
#     dots_max_z = 5.5  # depth values  # changed to 5.5 to match original script depth range?
#     if deep_with_sizes:
#         # increase cone depth
#         dots_max_z = 101
#
#     # do we need to increase n_dots for OLED?
#     n_dots = 5000
#
#     # dot_array_spread is the spread of x and ys BEFORE they are divided by their depth value to get actual positions.
#     # with dot_array_spread = widthPix * 3, this gives a values of 5760 on a 1920 monitor,
#     # similar to the original setting of 5000.  It also allows the flow_dots to be scaled to the screen for OLED.
#     dot_array_spread = widthPix * 3  # this scales it for the monitor and keeps more flow_dots on screen
#
#     # initial array values.  x and y are scaled by z_array, so x and y values can be larger than the screen.
#     # x and y are the position of the dots when they are at depth = 1.  These values can be larger than the monitor.
#     # at depths > 1, x and y are divided by z_array, so they are appear closer to the middle of the screen
#     # x_array = np.random.rand(n_dots) * dot_array_spread - dot_array_spread / 2
#     # y_array = np.random.rand(n_dots) * dot_array_spread - dot_array_spread / 2
#
#     # these array are more clear when initialized as
#     x_array = np.random.uniform(-dot_array_spread / 2, dot_array_spread / 2, n_dots)
#     y_array = np.random.uniform(-dot_array_spread / 2, dot_array_spread / 2, n_dots)
#
#     if deep_with_sizes:
#         # narrower spread of dots
#         x_array = np.random.uniform(-widthPix, widthPix, n_dots)
#         y_array = np.random.uniform(-widthPix, widthPix, n_dots)
#
#     # z_array = np.random.rand(n_dots) * (dots_max_z - dots_min_z) + dots_min_z
#     # This is clearer when written as
#     z_array = np.random.uniform(dots_min_z, dots_max_z, n_dots)
#
#
#     # print(f"x_array: {x_array}, y_array: {y_array}, z_array: {z_array}")
#
#     # x_flow and y_flow are the actual x_array and y_array positions of the dots, after being divided by their depth value.
#     x_flow = x_array / z_array
#     y_flow = y_array / z_array
#
#     # array of x_array, y_array positions of dots to pass to ElementArrayStim
#     dots_xys_array = np.array([x_flow, y_flow]).T
#
#     dot_sizes = 10
#     if deep_with_sizes:
#         dot_sizes = 50
#
#     # itialise flow_dots
#     flow_dots = visual.ElementArrayStim(win, elementTex=None, elementMask='circle',  # orig used 'gauss'
#                                         units='pix', nElements=n_dots, sizes=dot_sizes,
#                                         colorSpace=this_colourSpace, colors=flow_colour)
#     if deep_with_sizes:
#         flow_dots.sizes = dot_sizes / z_array
#
# elif background == 'flow_rings':
#     # # # RINGS
#     ring_speed = flow_speed / 4  # this is a quarter the speed of the dots .02  # 48 / fps  # 0.2 at 240Hz
#     flow_speed = ring_speed
#
#
#     n_rings = 100  # scale this to screen size?
#     rings_min_z = .1  # A value < 1 of .1 means that the closest ring's radius is 10x the size of the screen.
#     # print(f"ring_speed: {ring_speed}")
#     # print(f"n_rings: {n_rings}")
#     # print(f"rings_min_z: {rings_min_z}")
#
#     # set the limits on ring size
#     max_radius = heightPix  # Biggest ring is height of screen
#     # on OLED, allow bigger max radius to reduce flicker?  widthPix?
#     min_radius = 10  # smallest ring is 10 pixels
#
#     # If I want the smallest radius to be 10 pixels, then the max depth of 108 (1080/108=10)
#     rings_max_z = max_radius / min_radius
#     # print(f"rings_max_z: {rings_max_z}")
#
#     # adjust ring depth values by rings_z_adjust
#     rings_z_adjust = rings_max_z - rings_min_z
#
#     '''
#     Dots_array_width was used to give the dots unique x/y positions in 'space'.
#     For rings, they are all at the same x/y position (0, 0), so I don't need dot_array_wdith for them.
#     '''
#     ring_size_list = [1080] * n_rings
#     # print(f"ring_size_list: {ring_size_list}")
#
#     # depth values are evenly spaces and in ascending order, so smaller rings are drawn on top of larger ones.
#     # stop=stop=rings_max_z-(rings_z_adjust/n_rings) gives space the new ring to appear
#     ring_z_array = np.linspace(start=rings_min_z, stop=rings_max_z - (rings_z_adjust / n_rings), num=n_rings)
#
#     # the actual radii list is in descending order, so smaller rings are drawn on top of larger ones.
#     ring_radii_array = ring_size_list / ring_z_array
#
#     # RING COLOURS (alernating this_bgColour and flow_colour
#     ring_colours = [this_bgColour, flow_colour] * int(n_rings / 2)
#
#     # # use ElementArrayStim to draw the rings
#     flow_rings = visual.ElementArrayStim(win, elementTex=None, elementMask='circle', interpolate=True,
#                                          units='pix', nElements=n_rings, sizes=ring_radii_array,
#                                          colors=ring_colours, colorSpace=this_colourSpace)
# elif background == 'no_bg':
#
#     # if No moving background, use these values (see below for if there is a moving background)
#     n_dots = 0  # no dots
#     dots_speed = None
#     dot_array_spread = None  # this scales it for the monitor and keeps more flow_dots on screen
#     dots_min_z = None
#     dots_max_z = None  # depth values
#
#     # settings for dots or rings
#     # timing for background motion converted to frames (e.g., 70ms is 17frames at 240Hz).
#     prelim_fr = 0
#     actual_prelim_ms = prelim_fr * 1000 / fps
#
# if debug:
#     print(f'flow_speed: {flow_speed}')
#     if background == 'flow_dots':
#         print(f'dots_speed: {dots_speed}')
#         print(f'n_dots: {n_dots}')
#         print(f'dot_array_spread: {dot_array_spread}')
#         print(f'dots_min_z: {dots_min_z}')
#         print(f'dots_max_z: {dots_max_z}')
#     elif background == 'flow_rings':
#         print(f"ring_speed: {ring_speed}")
#         print(f"n_rings: {n_rings}")
#         print(f"rings_min_z: {rings_min_z}")
#         print(f"rings_max_z: {rings_max_z}")
#
#
# # MASK BEHIND PROBES (infront of flow dots to keep probes and motion separate)
# mask_size = 150
#
# if mask_type == '4_circles':
#     # four circlular masks, at dist_from_fix pixels from middle of the screen, on in each corner.
#
#     raisedCosTexture1 = visual.filters.makeMask(256, shape='raisedCosine', fringeWidth=0.3, radius=[1.0, 1.0])
#     probeMask1 = visual.GratingStim(win=win, mask=raisedCosTexture1, size=(mask_size, mask_size),
#                                     colorSpace=this_colourSpace, color=this_bgColour,
#                                     tex=None, units='pix', pos=[dist_from_fix + 1, dist_from_fix + 1])
#     probeMask2 = visual.GratingStim(win=win, mask=raisedCosTexture1, size=(mask_size, mask_size),
#                                     colorSpace=this_colourSpace, color=this_bgColour,
#                                     units='pix', tex=None, pos=[-dist_from_fix - 1, dist_from_fix + 1])
#     probeMask3 = visual.GratingStim(win=win, mask=raisedCosTexture1, size=(mask_size, mask_size),
#                                     colorSpace=this_colourSpace, color=this_bgColour,
#                                     units='pix', tex=None, pos=[-dist_from_fix - 1, -dist_from_fix - 1])
#     probeMask4 = visual.GratingStim(win=win, mask=raisedCosTexture1, size=(mask_size, mask_size),
#                                     colorSpace=this_colourSpace, color=this_bgColour,
#                                     units='pix', tex=None, pos=[dist_from_fix + 1, -dist_from_fix - 1])
#
# else:
#     # 2 spokes - wedges extending from the centre to the edges so that there is no motion directly behind probe locations.
#
#     if background == 'flow_dots':
#         # Remove any dots that would pass behind the probe locations
#         print(f"\nRemoving dots that would pass behind the probe locations.")
#         '''Simon's polar co-ords version'''
#         # remove 4 spokes so clear for probe to move
#         orient = 45  # 0 gives horizontal/vertical, 45 gives diagonals
#         # spoke_width_angle = np.arcsin((mask_size / 2) / dist_from_fix) * 180 / np.pi
#
#         # seems to wide, so making it smaller
#         spoke_width_angle = 22.5  # spoke_width_angle * .8
#         print(f"dist_from_fix: {dist_from_fix}")
#         print(f"mask_size: {mask_size}")
#         print(f"spoke_width_angle: {spoke_width_angle}")
#
#         # convert to radial
#         dot_rad_dist = np.sqrt(x_array * x_array + y_array * y_array)  # polar distance
#         dot_direction = np.arctan2(x_array, y_array)  # polar direction
#         dot_direction = np.degrees(dot_direction)  # convert to degrees so can use modulus
#
#         # collapse the four quadrants on to one (mod 90deg) to make more efficient
#         dot_direction_quad = dot_direction % 90
#
#         # identify dots that are in the lower part of spokes
#         dot_direction_low = (dot_direction_quad < spoke_width_angle)
#         # first rotate them by to get them all out of the lower part of the spoke
#         dot_direction_offsets = spoke_width_angle
#         # now add some noise to evenly distribute them
#         dot_direction_offsets += np.random.random_sample(n_dots) * (90 - 3 * spoke_width_angle)
#         # add offset
#         dot_direction[dot_direction_low] += dot_direction_offsets[dot_direction_low]
#
#         # now identify dots that are in upper part of spoke (repeat of upper)
#         dot_direction_high = (dot_direction_quad > (90 - spoke_width_angle))
#         # rotate them by to get them all out of the lower part of the spoke
#         dot_direction_offsets = spoke_width_angle
#         # now add some noise to evenly distribute them
#         dot_direction_offsets += np.random.random_sample(n_dots) * (90 - 3 * spoke_width_angle)
#         # subtract offsets
#         dot_direction[dot_direction_high] -= dot_direction_offsets[dot_direction_high]
#
#         # now convert back to cartesian
#         dot_direction -= orient  # first rotate for specified orientation
#         dot_direction = np.radians(dot_direction)  # convert back to degrees
#         x_array = dot_rad_dist * np.sin(dot_direction)
#         y_array = dot_rad_dist * np.cos(dot_direction)
#
#         # x_flow and y_flow are the actual x_array and y_array positions of the dots, after being divided by their depth value.
#         x_flow = x_array / z_array
#         y_flow = y_array / z_array
#
#         # array of x_array, y_array positions of dots to pass to ElementArrayStim
#         dots_xys_array = np.array([x_flow, y_flow]).T
#
#
#     else:
#         # if background == 'flow_rings':
#         # add a mask infront of the rings to keep probes and motion separate
#         print(f"\nAdding a mask infront of the rings to keep probes and motion separate.")
#
#         # draw a large diagonal cross (X) with vertices which reaches the top and bottom of the window
#
#         # since the middle of the screen is 0, 0; the corners are defined by half the width or height of the screen.
#         half_hi_pix = int(heightPix / 2)
#
#         # the corners of the cross are offset (by around 42 pixels on my laptop);
#         # which is half the mask_size / the screen aspect ratio (pixel shape)
#         offset_pix = int((mask_size / 2) / (widthPix / heightPix))
#         if debug:
#             print(f'offset_pix = {offset_pix}')
#
#         '''vertices start at the bottom left corner and go clockwise, with three values for each side.
#         The first three values are for the left of the X, the next three for the top
#         1. the bl corner of the cross, which is at the bottom of the window, with an offset (e.g., not in the corner of the window).
#         2. horizontally centred, but offset to the left of the centre.
#         3. the tl corner of the cross, which is at the top of the window, with an offset (e.g., not in the corner of the window).
#         4. offset to the right of 3.
#         5. vertically centred, but offset above the centre.
#         6. the tr corner of the cross, which is at the top of the window, with an offset (e.g., not in the corner of the window).
#         '''
#         # # original vertices as plain cross X
#         # vertices = np.array([[-half_hi_pix - offset_pix, -half_hi_pix], [-offset_pix, 0], [-half_hi_pix - offset_pix, half_hi_pix],
#         #                      [-half_hi_pix + offset_pix, half_hi_pix], [0, offset_pix], [half_hi_pix - offset_pix, half_hi_pix],
#         #                      [half_hi_pix + offset_pix, half_hi_pix], [offset_pix, 0], [half_hi_pix + offset_pix, -half_hi_pix],
#         #                      [half_hi_pix - offset_pix, -half_hi_pix], [0, -offset_pix], [-half_hi_pix + offset_pix, -half_hi_pix]
#         #                      ])
#
#         # updated vertices with wedge shape
#         vertices = np.array([[-half_hi_pix - offset_pix * 2, -half_hi_pix], [-offset_pix / 2, 0],
#                              [-half_hi_pix - offset_pix * 2, half_hi_pix],
#                              [-half_hi_pix + offset_pix * 2, half_hi_pix], [0, offset_pix / 2],
#                              [half_hi_pix - offset_pix * 2, half_hi_pix],
#                              [half_hi_pix + offset_pix * 2, half_hi_pix], [offset_pix / 2, 0],
#                              [half_hi_pix + offset_pix * 2, -half_hi_pix],
#                              [half_hi_pix - offset_pix * 2, -half_hi_pix], [0, -offset_pix / 2],
#                              [-half_hi_pix + offset_pix * 2, -half_hi_pix]
#                              ])
#
#         probeMask1 = visual.ShapeStim(win, vertices=vertices, colorSpace=this_colourSpace,
#                                        fillColor=this_bgColour, lineColor=this_bgColour, lineWidth=0)
#
#


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

# '''Timing: expected frame duration and tolerance
# with frame_tolerance_prop = .24, frame_tolerance_ms == 1ms at 240Hz, 2ms at 120Hz, 4ms at 60Hz
# For a constant frame_tolerance_ms of 1ms, regardless of fps, use frame_tolerance_prop = 1/expected_fr_sec
# Psychopy records frames in seconds, but I prefer to think in ms. So wo variables are labelled with _sec or _ms.
# '''
# expected_fr_sec = 1 / fps
# expected_fr_ms = expected_fr_sec * 1000
# frame_tolerance_prop = 1 / expected_fr_ms  # frame_tolerance_ms == 1ms, regardless of fps..
# max_fr_dur_sec = expected_fr_sec + (expected_fr_sec * frame_tolerance_prop)
# min_fr_dur_sec = expected_fr_sec - (expected_fr_sec * frame_tolerance_prop)
# frame_tolerance_ms = (max_fr_dur_sec - expected_fr_sec) * 1000
# max_dropped_fr_trials = 10  # number of trials with dropped frames to allow before experiment is aborted
# if debug:
#     print(f"\nexpected_fr_ms: {expected_fr_ms}")
#     print(f"frame_tolerance_prop: {frame_tolerance_prop}")
#     print(f"frame_tolerance_ms: {frame_tolerance_ms}")
#     print(f"max_dropped_fr_trials: {max_dropped_fr_trials}")


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
                               text="\n\nFocus on the fixation circle at the centre of the screen.\n\n"
                                    "A small white target will briefly appear on screen,\n"
                                    "press the key related to the location of the probe:\n\n"
                                    "[4]/[Q] top-left\t\t\t[5]/[W] top-right\n\n\n\n"
                                    "[1]/[A] bottom-left\t\t\t[2]/[S] bottom-right.\n\n\n"
                                    "Some targets will be easy to see, others will be hard to spot.\n"
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
stairStart = maxLum  # start luminance value
if monitor_name == 'OLED':  # dimmer on OLED
    stairStart = maxLum * 0.3

stairs = []
for stair_idx in range(n_stairs):
    thisInfo = copy(expInfo)
    thisInfo['stair_idx'] = stair_idx
    # thisInfo['ISI_cond_fr'] = ISI_conds_list[stair_idx]
    thisInfo['ISI_cond_fr'] = ISI_cond_fr
    thisInfo['sep'] = sep_conds_list[stair_idx]
    thisInfo['cong_val'] = cong_val_conds_list[stair_idx]
    thisInfo['cong_name'] = cong_name_conds_list[stair_idx]
    thisInfo['prelim_ms'] = prelim_conds_list[stair_idx]


    thisStair = Staircase(name=stair_names_list[stair_idx],
                          type='simple',  # step size changes after each reversal only
                          value=stairStart,
                          C=stairStart * 0.6,  # initial step size, as prop of maxLum
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
            # if debug:
            print(f"\n({trial_num_inc_repeats}) trial_number: {trial_number}, "
                  f"stair_idx: {stair_idx}, thisStair: {thisStair}, step: {step}")

            # conditions (ISI, congruence, sep, prelim)
            ISI_cond_fr = thisStair.extraInfo['ISI_cond_fr']
            sep = thisStair.extraInfo['sep']
            congruent = thisStair.extraInfo['cong_val']
            cong_name = thisStair.extraInfo['cong_name']
            prelim_ms = thisStair.extraInfo['prelim_ms']
            if debug:
                print(f"ISI_cond_fr: {ISI_cond_fr}, sep: {sep}, congruent: {congruent}, cong_name: {cong_name}, prelim_ms: {prelim_ms}")


            # # # SEP COND variables # # #
            # negative separation for comparing conditions (e.g., cong sep = 5, incong sep = -5.
            if cong_name == 'incong':
                neg_sep = 0 - sep
                if sep == 0:
                    neg_sep = -.1
            else:
                neg_sep = sep
            if debug:
                print(f"sep: {sep}, neg_sep: {neg_sep}")

            # separation expressed as degrees.
            # todo: add neg_sep_deg?
            if -1 < sep < 99:
                sep_deg = sep * pixel_mm_deg_dict['diag_deg']
            else:
                sep_deg = 0


            # # # GET BACKGROUND ATTRIBUTES # # #

            # use congruence to determine the flow direction and target jump direction
            # 1 is contracting/inward/backwards, -1 is expanding/outward/forwards
            flow_dir = np.random.choice([1, -1])
            flow_name = 'cont'
            if flow_dir == -1:
                flow_name = 'exp'

            target_jump = congruent * flow_dir

            # boundaries for z position (distance from screen) during radial flow
            if flow_dir == -1:  # expanding
                z_start_bounds = [near_plane_cm + max_z_cm_in_life, far_plane_cm]
            else:  # contracting, flow_dir == 1
                z_start_bounds = [near_plane_cm, far_plane_cm - max_z_cm_in_life]
            if debug:
                print(f"z_start_bounds: {z_start_bounds}")

            # vary fixation polarity to reduce risk of screen burn.
            # if monitor_name == 'OLED':  # same for all moniotrs for consistency
            if trial_num_inc_repeats % 2 == 0:
                fixation.lineColor = 'grey'
                fixation.fillColor = 'black'
            else:
                fixation.lineColor = 'black'
                fixation.fillColor = 'grey'

            # reset fixation radius - reduces in size after probe 2
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

            # PROBE LOCATION
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
            # timing for background motion converted to frames (e.g., 70ms is 17frames at 240Hz).
            prelim_fr = int(prelim_ms * fps / 1000)
            actual_prelim_ms = prelim_fr * 1000 / fps
            if debug:
                print(f'\nprelim_ms: {prelim_ms}')
                print(f'prelim_fr: {prelim_fr}')
                print(f'actual_prelim_ms: {actual_prelim_ms}')

            # variable fixation time
            '''to reduce anticipatory effects that might arise from fixation always being same length.
            if False, vary_fix == .5 seconds, so end_fix_fr is 1 second.
            if Ture, vary_fix is between 0 and 1 second, so end_fix_fr is between .5 and 1.5 seconds.'''
            vary_fix = int(fps / 2)
            if vary_fixation:
                vary_fix = np.random.randint(0, fps)

            # timing in frames for ISI and probe2
            # If probes are presented concurrently, set isi_dur_fr and p2_segment_fr to last for 0 frames.
            isi_dur_fr = ISI_cond_fr
            p2_segment_fr = probe_duration
            if ISI_cond_fr < 0:
                isi_dur_fr = p2_segment_fr = 0

            # cumulative timing in frames for each segment of a trial
            end_fix_fr = int(fps / 2) + vary_fix - prelim_fr
            if end_fix_fr < 0:
                end_fix_fr = int(fps / 2)
            end_bg_motion_fr = end_fix_fr + prelim_fr
            end_p1_fr = end_bg_motion_fr + probe_duration
            end_ISI_fr = end_p1_fr + isi_dur_fr
            end_p2_fr = end_ISI_fr + p2_segment_fr
            if debug:
                print(f"end_fix_fr: {end_fix_fr}, end_bg_motion_fr: {end_bg_motion_fr}, "
                      f"end_p1_fr: {end_p1_fr}, end_ISI_fr: {end_ISI_fr}, end_p2_fr: {end_p2_fr}\n")


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
                # todo: turn off high priority here during enforced break, turn back on after wait?
                core.wait(secs=break_dur)
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
                if frameN == end_bg_motion_fr:
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
                    # if background == 'flow_dots':
                    #     flow_dots.xys = dots_xys_array
                    #     flow_dots.draw()
                    # elif background == 'flow_rings':
                    #     flow_rings.draw()
                    #
                    # if mask_type == '4_circles':
                    #     probeMask1.draw()
                    #     probeMask2.draw()
                    #     probeMask3.draw()
                    #     probeMask4.draw()
                    # elif (mask_type == '2_spokes') & (background == 'flow_rings'):
                    #     probeMask1.draw()

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
                elif end_bg_motion_fr >= frameN > end_fix_fr:
                    # if background == 'flow_dots':
                    #     # get new depth_vals array (z_array) and dots_xys_array (x_array, y_array)
                    #     z_array, dots_xys_array = new_dots_z_and_pos(x_array, y_array, z_array, dots_speed, flow_dir,
                    #                                                  dots_min_z, dots_max_z)
                    #     flow_dots.xys = dots_xys_array
                    #     if deep_with_sizes:
                    #         flow_dots.sizes = dot_sizes / z_array
                    #     flow_dots.draw()
                    # elif background == 'flow_rings':
                    #     ring_z_array, ring_radii_array, ring_colours = roll_rings_z_and_colours(ring_z_array,
                    #                                                                             ring_colours,
                    #                                                                             rings_min_z,
                    #                                                                             rings_max_z,
                    #                                                                             flow_dir, ring_speed,
                    #                                                                             ring_size_list)
                    #     flow_rings.sizes = ring_radii_array
                    #     flow_rings.colors = ring_colours
                    #     flow_rings.draw()
                    #
                    # if mask_type == '4_circles':
                    #     probeMask1.draw()
                    #     probeMask2.draw()
                    #     probeMask3.draw()
                    #     probeMask4.draw()
                    # elif (mask_type == '2_spokes') & (background == 'flow_rings'):
                    #     probeMask1.draw()

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

                # # # PROBE 1 - after prelim bg motion, before ISI # # #
                elif end_p1_fr >= frameN > end_bg_motion_fr:
                    # if background == 'flow_dots':
                    #     # get new depth_vals array (z_array) and dots_xys_array (x_array, y_array)
                    #     z_array, dots_xys_array = new_dots_z_and_pos(x_array, y_array, z_array, dots_speed, flow_dir,
                    #                                                  dots_min_z, dots_max_z)
                    #     flow_dots.xys = dots_xys_array
                    #     if deep_with_sizes:
                    #         flow_dots.sizes = dot_sizes / z_array
                    #     flow_dots.draw()
                    # elif background == 'flow_rings':
                    #     ring_z_array, ring_radii_array, ring_colours = roll_rings_z_and_colours(ring_z_array,
                    #                                                                             ring_colours,
                    #                                                                             rings_min_z,
                    #                                                                             rings_max_z,
                    #                                                                             flow_dir, ring_speed,
                    #                                                                             ring_size_list)
                    #     flow_rings.sizes = ring_radii_array
                    #     flow_rings.colors = ring_colours
                    #     flow_rings.draw()
                    #
                    # if mask_type == '4_circles':
                    #     probeMask1.draw()
                    #     probeMask2.draw()
                    #     probeMask3.draw()
                    #     probeMask4.draw()
                    # elif (mask_type == '2_spokes') & (background == 'flow_rings'):
                    #     probeMask1.draw()

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
                    if ISI_cond_fr == -1:
                        if sep <= 18:
                            probe2.draw()



                # # # ISI - after probe 1, before probe 2 (or nothing if ISI_cond_fr < 1) # # #
                elif end_ISI_fr >= frameN > end_p1_fr:
                    # if background == 'flow_dots':
                    #     # get new depth_vals array (z_array) and dots_xys_array (x_array, y_array)
                    #     z_array, dots_xys_array = new_dots_z_and_pos(x_array, y_array, z_array, dots_speed, flow_dir,
                    #                                                  dots_min_z, dots_max_z)
                    #     flow_dots.xys = dots_xys_array
                    #     if deep_with_sizes:
                    #         flow_dots.sizes = dot_sizes / z_array
                    #     flow_dots.draw()
                    # elif background == 'flow_rings':
                    #     ring_z_array, ring_radii_array, ring_colours = roll_rings_z_and_colours(ring_z_array,
                    #                                                                             ring_colours,
                    #                                                                             rings_min_z,
                    #                                                                             rings_max_z,
                    #                                                                             flow_dir, ring_speed,
                    #                                                                             ring_size_list)
                    #     flow_rings.sizes = ring_radii_array
                    #     flow_rings.colors = ring_colours
                    #     flow_rings.draw()
                    #
                    # if mask_type == '4_circles':
                    #     probeMask1.draw()
                    #     probeMask2.draw()
                    #     probeMask3.draw()
                    #     probeMask4.draw()
                    # elif (mask_type == '2_spokes') & (background == 'flow_rings'):
                    #     probeMask1.draw()

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

                # # # PROBE 2 - after ISI before response segment (unless ISI_cond_fr < 1) # # #
                elif end_p2_fr >= frameN > end_ISI_fr:
                    # if background == 'flow_dots':
                    #     # get new depth_vals array (z_array) and dots_xys_array (x_array, y_array)
                    #     z_array, dots_xys_array = new_dots_z_and_pos(x_array, y_array, z_array, dots_speed, flow_dir,
                    #                                                  dots_min_z, dots_max_z)
                    #     flow_dots.xys = dots_xys_array
                    #     if deep_with_sizes:
                    #         flow_dots.sizes = dot_sizes / z_array
                    #     flow_dots.draw()
                    # elif background == 'flow_rings':
                    #     ring_z_array, ring_radii_array, ring_colours = roll_rings_z_and_colours(ring_z_array,
                    #                                                                             ring_colours,
                    #                                                                             rings_min_z,
                    #                                                                             rings_max_z,
                    #                                                                             flow_dir, ring_speed,
                    #                                                                             ring_size_list)
                    #     flow_rings.sizes = ring_radii_array
                    #     flow_rings.colors = ring_colours
                    #     flow_rings.draw()
                    #
                    # if mask_type == '4_circles':
                    #     probeMask1.draw()
                    #     probeMask2.draw()
                    #     probeMask3.draw()
                    #     probeMask4.draw()
                    # elif (mask_type == '2_spokes') & (background == 'flow_rings'):
                    #     probeMask1.draw()

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
                    if ISI_cond_fr >= 0:  # if not concurrent condition (ISI=-1)
                        if sep != 99:  # If not 1probe condition (sep = 99)
                            probe2.draw()


                # # # ANSWER - after probe 2, before end of trial # # #
                elif frameN > end_p2_fr:
                    # if background == 'flow_dots':
                    #     flow_dots.draw()
                    # elif background == 'flow_rings':
                    #     flow_rings.draw()
                    #
                    # if mask_type == '4_circles':
                    #     probeMask1.draw()
                    #     probeMask2.draw()
                    #     probeMask3.draw()
                    #     probeMask4.draw()
                    # elif (mask_type == '2_spokes') & (background == 'flow_rings'):
                    #     probeMask1.draw()

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
                print(f"checking: dropped_fr_trial_counter {dropped_fr_trial_counter} > "
                      f"1max_dropped_fr_trials: {max_dropped_fr_trials} = {dropped_fr_trial_counter > max_dropped_fr_trials}")
                event.clearEvents(eventType='keyboard')
                while not event.getKeys():
                    # display too_many_dropped_fr message until screen is pressed
                    too_many_dropped_fr.draw()
                    win.flip()
                    core.wait(secs=5)

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
        thisExp.addData('separation', sep)
        thisExp.addData('sep_deg', sep_deg)
        thisExp.addData('neg_sep', neg_sep)
        thisExp.addData('ISI_selected_ms', ISI_selected_ms)
        thisExp.addData('ISI_actual_ms', ISI_actual_ms)
        thisExp.addData('ISI_cond_fr', ISI_cond_fr)
        thisExp.addData('isi_dur_fr', isi_dur_fr)
        thisExp.addData('prelim_ms', prelim_ms)
        thisExp.addData('prelim_fr', prelim_fr)
        thisExp.addData('actual_prelim_ms', actual_prelim_ms)
        thisExp.addData('congruent', congruent)
        thisExp.addData('flow_dir', flow_dir)
        thisExp.addData('flow_name', flow_name)
        thisExp.addData('probe_jump', target_jump)
        thisExp.addData('corner', corner)
        # thisExp.addData('corner_name', corner_name)
        thisExp.addData('probeLum', probeLum)
        thisExp.addData('probeColor1', probeColor1)
        # thisExp.addData('probeColor255', probeColor255)
        # thisExp.addData('trial_response', resp.corr)
        thisExp.addData('resp_keys', resp.keys)
        thisExp.addData('resp_corr', resp.corr)
        thisExp.addData('resp_rt', resp.rt)
        thisExp.addData('flow_speed_cm_p_sec', flow_speed_cm_p_sec)
        thisExp.addData('flow_speed_cm_p_fr', flow_speed_cm_p_fr)
        thisExp.addData('n_dots', n_dots)
        thisExp.addData('dot_life_max_ms', dot_life_max_ms)
        thisExp.addData('probe_ecc', probe_ecc)
        thisExp.addData('orientation', orientation)
        # thisExp.addData('background', background)
        # thisExp.addData('flow_speed', flow_speed)
        thisExp.addData('vary_fix', vary_fix)
        thisExp.addData('end_fix_fr', end_fix_fr)
        # thisExp.addData('p1_diff', p1_diff)
        # thisExp.addData('isi_diff', isi_diff)
        # thisExp.addData('p2_diff', p2_diff)
        # thisExp.addData('mask_type', mask_type)
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
