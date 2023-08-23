import numpy as np
from psychopy import visual, core, event

'''
This script is for measuring the mean speed of flow_dots and flow_rings.
'''


def new_dots_depth_and_pos(x_array, y_array, depth_array, dots_speed, flow_dir, min_depth, max_depth):
    """
    This is a function to update flow_dots depth array and get new pixel co-ordinatesusing the original x_array and y_array.

    1a. Update depth_array by adding dots_speed * flow_dir to the current z values.
    1b. adjust any values below dots_min_depth or above dots_max_depth.

    2a. Get new x_pos and y_pos co-ordinates values by dividing x_array and y_array by the new depth_array.
    2b. put the new x_pos and y_pos co-ordinates into an array and transposes it.

    :param x_array: Original x_array positions for the dots (shape = (n_dots, 1))
    :param y_array: Original y_array positions for the dots (shape = (n_dots, 1))
    :param depth_array: array of depth values for the dots (shape = (n_dots, 1))
    :param dots_speed: speed of the dots (float, smaller = slower, larger = faster)
    :param flow_dir: either 1 (contracting/inward/backwards) or -1 (expanding/outward/forwards)
    :param dots_min_depth: default is .5, values below this are adjusted to dots_max_depth
    :param dots_max_depth: default is 5, values above this are adjusted to dots_min_depth
    :return: new dots_pos_array
    """

    # todo: change depth to z in names.

    # # 1. Update z (depth values) # #
    # Add dots_speed * flow_dir to the current z values.
    updated_depth_arr = depth_array + dots_speed * flow_dir

    # adjust any depth values below min_depth or above max_depth by depth_adj
    depth_adj = max_depth - min_depth
    # adjust updated_depth_arr values less than min_depth by adding depth_adj
    less_than_min = (updated_depth_arr < min_depth)
    updated_depth_arr[less_than_min] += depth_adj
    # adjust updated_depth_arr values more than max_depth by subtracting depth_adj
    more_than_max = (updated_depth_arr > max_depth)
    updated_depth_arr[more_than_max] -= depth_adj
    # print(f"updated_depth_arr (clipped):\n{updated_depth_arr}\n")

    # # 2. Get new pixel co-ordinates for dots using original x_array and y_array and updated_depth_arr # #
    x_pos = x_array / updated_depth_arr
    y_pos = y_array / updated_depth_arr

    # puts the new co-ordinates into an array and transposes it, ready to use.
    dots_pos_array = np.array([x_pos, y_pos]).T

    return updated_depth_arr, dots_pos_array


def roll_rings_z_and_colours(z, ring_colours, min_z, max_z, flow_dir, flow_speed, initial_x_vals):
    """
    This rings will spawn a new ring if the old one either grows too big for the screen (expanding),
    or shrinks too small (if contracting).

    This function updates the z (depth) values for the rings, and adjusts any values below min_z or
    above max_z by depth_adj.  Any values that are adjusted are then rolled to the end or beginning of the array,
    depending on whether they are below min_z or above max_z.
    The same values are then also rolled in the ring_colours array.

    :param z: Numpy array of z values for the rings (shape = (n_rings, 1))
    :param ring_colours: List of RGB1 colours for the rings (shape = (n_rings, 3))
    :param min_z: minimum depth value for the rings (how close they can get to the screen)
    :param max_z: maximum depth value for the rings (how far away they can get from the screen)
    :param flow_dir: either 1 (contracting/inward/backwards) or -1 (expanding/outward/forwards)
    :param flow_speed: speed of the rings (float, smaller = slower, larger = faster)
    :param initial_x_vals: nupmy array of ring sizes, (all the same size, e.g., 1080, shape = (n_rings, 1))

    :return: z (updated), ring_radii_array, ring_colours (rolled if any z values are rolled)
    """

    # update depth values
    z = z + flow_speed * flow_dir

    # depth_adj is the size of change to make to out-of-bounds rings
    # todo: change to z_adj
    depth_adj = max_z - min_z

    # adjust any z values below min_depth or above max_depth by depth_adj
    if flow_dir == -1:  # expanding, getting closer, might be below min_z
        # find which rings are less than min and add depth_adj to those rings
        less_than_min = (z < min_z)
        z[less_than_min] += depth_adj

        # shift arrays by this amount (e.g., if 3 rings are less than min, shift by 3)
        # (note negative shift to move them backwards)
        shift_num = -sum(less_than_min)

    elif flow_dir == 1:  # contracting, getting further away, might be above max_z
        # find which rings are more_than_max and subtract depth_adj to those rings
        more_than_max = (z > max_z)
        z[more_than_max] -= depth_adj

        # shift arrays by this amount (e.g., if 3 rings are more_than_max, shift by 3)
        shift_num = sum(more_than_max)

    # roll the depth and colours arrays so that adjusted rings move to other end of array
    z = np.roll(z, shift=shift_num, axis=0)
    ring_colours = np.roll(ring_colours, shift=shift_num, axis=0)

    # get new ring_radii_array
    ring_radii_array = initial_x_vals / z

    return z, ring_radii_array, ring_colours, shift_num


# screen details
widthPix = 1920
heightPix = 1080
monitor_name = 'Nick_work_laptop'
fps = 60

# flow_dir = np.random.choice([-1, 1])
flow_dir = -1  # -1 = expanding, 1 = contracting
print(f"flow_dir: {flow_dir}")

# colours and colour space
LumColor255Factor = 2.39538706913372
maxLum = 106  # 255 RGB
bgLumProp = .2  # .2  # todo: use .45 to match radial_flow_NM_v2.py, or .2 to match exp1
bgLum = maxLum * bgLumProp
bgColor255 = bgLum * LumColor255Factor

# colour space
this_colourSpace = 'rgb1'  # values between 0 and 1
bgColor_rgb1 = bgLum / maxLum
this_bgColour = [bgColor_rgb1, bgColor_rgb1, bgColor_rgb1]

# Give dots a pale green colour, which is adj_dots_col different to the background
adj_dots_col = .15
flow_colour = [this_bgColour[0] - adj_dots_col, this_bgColour[1], this_bgColour[2] - adj_dots_col]

# window
win = visual.Window([widthPix, heightPix], monitor="Nick_work_laptop", units="pix", fullscr=True,
                    color=this_bgColour,
                    colorSpace=this_colourSpace)

'''
dots - 
1. start with min_max_depth = .5, 5
2. change to min_max_depth = 1, 5
3. change to min_max_depth = 1, 5.5
 - depth impacts as further way things appear to move slower.
 - so 1 to 5.5. is not the same as .5 to 5.

rings -
1. start with min_max_depth = 1, 5 (or 5.5. if 1, 5.5. matches .5, 5 for dots)
2. change to min_max_depth = 1, 108 (e.g., looks best for rings)

# later
check that changing fps changes the speed of the dots/rings as expected.
e.g., 120fps should be half as fast as 60fps, 240 should be a quarter as fast as 60fps.  

# if I match all variables (n_items, min_depth, max_depth, depth_adj, flow_speed) for dots and rings,
# is the mean speed the same for both??
No, rings speed is faster than dots at 1-5.5., but slower at 5.5-10.
I think the difference is due to the ring sizes.  When the max depth is 5.5., the smallest ring is 1080/5.5=196 pixels.
Since objects at the edge move faster than objects near the centre, the rings have a higher mean speed when they 
are stopped from being small.
'''

min_depth = 5.5
max_depth = 10
n_items = 1000
flow_speed = .2  # 48 / fps  # 0.2 at 240Hz
record_n_frames = 100
# background = 'flow_rings'  # 'flow_dots', 'flow_rings'
background_list = ['flow_dots', 'flow_rings']

dots_horiz_diff = 0
dots_diag_diff = 0
rings_horiz_diff = 0

for background in background_list:

    if background == 'flow_dots':
        print("flow_dots")

        prelim_bg_flow_ms = 0

        # timing for background motion converted to frames (e.g., 70ms is 17frames at 240Hz).
        prelim_bg_flow_fr = int(prelim_bg_flow_ms * fps / 1000)
        actual_prelim_bg_flow_ms = prelim_bg_flow_fr * 1000 / fps

        # rate of change for dots (dots_speed is added to their depth value), which is then divided by their previous x/ y pos.
        # dots speed should be scaled by fps, so dots have a greater change per frame on slower monitors.
        # e.g., .2 at 240Hz, .4 at 120Hz and .8 at 60Hz.
        # todo: this appears too fast to me, but it is the same as the original script.
        # dots_speed = 48 / fps
        dots_speed = flow_speed

        # dot_array_spread is the spread of x and ys BEFORE they are divided by their depth value to get actual positions.

        # todo: most of the flow_dots are off screen using this current dots_min_depth, as the distribution of x_flow has large tails.
        #  Setting it to 1.0 means that the tails are shorter, as dividing x / z only makes values smaller (or the same), not bigger.
        dots_min_depth = 1.0  # original script used .5, which increased the tails meaning more dots were offscreen.
        dots_max_depth = 5.5  # depth values  # todo: change 5.5 to match original script depth range?
        # dots_min_depth = min_depth
        # dots_max_depth = max_depth

        # Changing dots_min_depth from .5 to one means that the proportion of dots onscreen increases from ~42% to ~82%.
        # Therefore, I can half n_dots with little change in the number of dots onscreen, saving processing resources.
        # Note: 'onscreen' was defined as half widthPix (960).  Once the edge mask is added, the square of the visible screen is 1080x1080,
        # minus the blurred edges, so 960 seems reasonable.
        # todo: do we need to increase n_dots for OLED?
        n_dots = 5000
        # n_dots = n_items

        # with dot_array_spread = widthPix * 3, this gives a values of 5760 on a 1920 monitor,
        # similar to the original setting of 5000.  It also allows the flow_dots to be scaled to the screen for OLED.
        dot_array_spread = widthPix * 3  # this scales it for the monitor and keeps more flow_dots on screen

        # initial array values.  x and y are scaled by z, so x and y values can be larger than the screen.
        # x and y are the position of the dots when they are at depth = 1.  These values can be larger than the monitor.
        # at depths > 1, x and y are divided by z, so they are appear closer to the middle of the screen
        x = np.random.rand(n_dots) * dot_array_spread - dot_array_spread / 2
        y = np.random.rand(n_dots) * dot_array_spread - dot_array_spread / 2
        z = np.random.rand(n_dots) * (dots_max_depth - dots_min_depth) + dots_min_depth
        # print(f"x: {x}, y: {y}, z: {z}")

        # x_flow and y_flow are the actual x and y positions of the dots, after being divided by their depth value.
        x_flow = x / z
        y_flow = y / z

        # array of x, y positions of dots to pass to ElementArrayStim
        dots_xys_array = np.array([x_flow, y_flow]).T

        # Give dots a pale green colour, which is adj_dots_col different to the background
        adj_dots_col = .15

        flow_colour = [this_bgColour[0] - adj_dots_col, this_bgColour[1], this_bgColour[2] - adj_dots_col]
        if monitor_name == 'OLED':
            # darker green for low contrast against black background
            flow_colour = [this_bgColour[0], this_bgColour[1] + adj_dots_col / 2, this_bgColour[2]]

        flow_dots = visual.ElementArrayStim(win, elementTex=None, elementMask='circle',  # orig used 'gauss'
                                            units='pix', nElements=n_dots, sizes=10,
                                            colorSpace=this_colourSpace, colors=flow_colour)

    elif background == 'flow_rings':
        print("flow_rings")

        # # # RINGS
        # flow_speed = 48 / fps  # 0.2 at 240Hz
        # n_rings = 20  # n_rings = (int(mon_width_cm / 10) + 5) * 2
        # rings_min_depth = 1  # 0.5  # depth values

        # ring_speed = flow_speed
        # n_rings = n_items
        # rings_min_depth = min_depth
        ring_speed = flow_speed * 8.9  # todo: change this back?
        n_rings = 100
        rings_min_depth = .01  # less than one, will mean rings can have radii upto 1080/0.01=108000 pixels

        max_radius = heightPix
        min_radius = 10

        # the smallest ring radii is 1080/5=216.  If I want the smallest radius to be 5 pixels, then the max depth of 216 (1080/216=5)
        rings_max_depth = max_radius / min_radius
        # rings_max_depth = max_depth
        # print(f"rings_max_depth: {rings_max_depth}")

        # adjust ring depth values by depth_adj
        depth_adj = rings_max_depth - rings_min_depth

        # RING COLOURS (alernating this_bgColour and flow_colour)
        ring_colours = [this_bgColour, flow_colour] * int(n_rings / 2)

        '''
        Dots_array_width was used to give the dots unique x/y positions in 'space'.
        For rings, they are all at the same x/y position (0, 0), so I don't need dot_array_wdith for them.
        '''
        ring_size_list = [1080] * n_rings
        # print(f"ring_size_list: {ring_size_list}")

        '''
        original z array for dots uses:
        z = np.random.rand(n_rings) * (rings_max_depth - rings_min_depth) + rings_min_depth
        0 to 1, times 4 (5-1) giving values in the range 0 to 4, plus 1, giving values in the range 1 to 5. (min to max)
        But these are in a random order.
        I want values to be evently spaced and in order (smallest depth first), so that when ring_size_list is divided by z,
        the largest ring is drawn first (at the back) and the smallest ring is drawn last (at the front).

        I could set z for n+1 rings, then when a ring is rolled from the end to the beginning, it will be in the correct position.
        But instead I will set it for n_rings on a slightly smaller range (rings_max_depth-(depth_adj/n_rings)), 
        so that when a ring is rolled from the end to the beginning, it will be in the correct position.
        '''

        # stop=stop=rings_max_depth-(depth_adj/n_rings) gives space the new ring to appear
        ring_z_array = np.linspace(start=rings_min_depth, stop=rings_max_depth - (depth_adj / n_rings), num=n_rings)

        # print(f"ring_z_array: {ring_z_array}")

        ring_radii_array = ring_size_list / ring_z_array
        # print(f"ring_radii_array: {ring_radii_array}")

        # # use ElementArrayStim to draw the rings
        flow_rings = visual.ElementArrayStim(win, elementTex=None,
                                             elementMask='circle',
                                             interpolate=True,
                                             units='pix', nElements=n_rings,
                                             sizes=ring_radii_array,
                                             colors=ring_colours,
                                             colorSpace=this_colourSpace
                                             )

    #
    # for frame in range(120):
    #
    #     if background == 'flow_dots':
    #
    #         # get new depth_vals array (z) and dots_xys_array (x, y)
    #         z, dots_xys_array = new_dots_depth_and_pos(x, y, z, dots_speed, flow_dir,
    #                                                    dots_min_depth, dots_max_depth)
    #         flow_dots.xys = dots_xys_array
    #         flow_dots.draw()
    #
    #     elif background == 'flow_rings':
    #         # update depth values and ring colours and get new ring_radii_array
    #         ring_z_array, ring_radii_array, ring_colours = roll_rings_z_and_colours(ring_z_array, ring_colours,
    #                                                                                 rings_min_depth, rings_max_depth,
    #                                                                                 flow_dir, ring_speed, ring_size_list)
    #         flow_rings.sizes = ring_radii_array
    #         flow_rings.colors = ring_colours
    #
    #         flow_rings.draw()
    #
    #     win.flip()
    #
    win.close()

    '''
    In this script I want to measure the mean speed of the flow_dots and flow_rings.
    I will do this by measuring the difference between dots_xys_array or ring_radii_array between two frames.
    I will measure several frames and take the mean of these differences.

    I will get the value for all dots (or rings), and also the mean only including dots (or rings) that are onscreen.

    I want to investigate the impact of changing min_dot_depth and max_dot_depth (or rings_min_depth and rings_max_depth).

    I will also try to find values that work so that flow_dots and flow_rings have the same mean speed.

    '''

    # empty list to store the values
    recorded_fr_list = []
    shift_num_list = []
    for frame in range(record_n_frames):
        # print(f"\nframe: {frame}")

        if background == 'flow_dots':

            # get new depth_vals array (z) and dots_xys_array (x, y)
            z, dots_xys_array = new_dots_depth_and_pos(x, y, z, dots_speed, flow_dir,
                                                       dots_min_depth, dots_max_depth)
            recorded_fr_list.append(dots_xys_array)


        elif background == 'flow_rings':
            # update depth values and ring colours and get new ring_radii_array
            ring_z_array, ring_radii_array, ring_colours, shift_num = roll_rings_z_and_colours(ring_z_array,
                                                                                               ring_colours,
                                                                                               rings_min_depth,
                                                                                               rings_max_depth,
                                                                                               flow_dir, ring_speed,
                                                                                               ring_size_list)
            recorded_fr_list.append(ring_radii_array)
            # print(f"ring_radii_array: {ring_radii_array}")

            # print(f"shift_num: {shift_num}")
            shift_num_list.append(shift_num)

        # win.flip()

    print(f"recorded_fr_list: {np.shape(recorded_fr_list)}")
    print(f"shift_num_list: {np.shape(shift_num_list)}\n{shift_num_list}")
    # # I want to remove flow_dots values that are off-screen at this point.
    # # I am considering a square that is heightPix x heightPix, centred on the screen.
    # # SO anything outside -heightPix/2 to heightPix/2 is offscreen.
    # # I want to do this for pair of frames.  Get the onsceen indices to make a mask for frames 1 and 2,
    # # then add the masks together to get a mask for both frames.
    import math

    if background == 'flow_dots':
        onscrn_x_y_diff = []
        # loop through pairs of frames
        for i in range(len(recorded_fr_list) - 1):
            # get the indices of dots that are onscreen for frame 1
            onscreen_dots_1 = (recorded_fr_list[i][:, 0] > -heightPix / 2) & (
                        recorded_fr_list[i][:, 0] < heightPix / 2) & \
                              (recorded_fr_list[i][:, 1] > -heightPix / 2) & (recorded_fr_list[i][:, 1] < heightPix / 2)
            # get the indices of dots that are onscreen for frame 2
            onscreen_dots_2 = (recorded_fr_list[i + 1][:, 0] > -heightPix / 2) & (
                        recorded_fr_list[i + 1][:, 0] < heightPix / 2) & \
                              (recorded_fr_list[i + 1][:, 1] > -heightPix / 2) & (
                                          recorded_fr_list[i + 1][:, 1] < heightPix / 2)
            # get the indices of dots that are onscreen for both frames
            onscreen_dots = onscreen_dots_1 & onscreen_dots_2
            # print(f"onscreen_dots: {onscreen_dots}")

            # get the difference pairs of frames
            paired_fr_diff = recorded_fr_list[i + 1][onscreen_dots] - recorded_fr_list[i][onscreen_dots]
            # print(f"paired_fr_diff: {paired_fr_diff}")

            onscrn_x_y_diff.append(paired_fr_diff)

        # convert to 2d numpy array
        onscrn_x_y_diff = np.concatenate(onscrn_x_y_diff, axis=0)
        # print(f"onscrn_x_y_diff: {np.shape(onscrn_x_y_diff)}\n{onscrn_x_y_diff}")

        # just take x difference values
        onscrn_x_diff = onscrn_x_y_diff[:, 0]
        # print(f"onscrn_x_diff: {np.shape(onscrn_x_diff)}\n{onscrn_x_diff}")

        # get an array of diagonal distances using pythogoras on the x and y values
        onscrn_diag_diff = np.sqrt(onscrn_x_y_diff[:, 0] ** 2 + onscrn_x_y_diff[:, 1] ** 2)
        # print(f"onscrn_diag_diff: {np.shape(onscrn_diag_diff)}\n{onscrn_diag_diff}")

    elif background == 'flow_rings':
        onscrn_rad_diff = []

        # loop through pairs of frames
        for i in range(len(recorded_fr_list) - 1):
            # first, roll the rings in frame 1 by shift num, so that they match frame 2
            rolled_fr1_radii = np.roll(recorded_fr_list[i], shift=shift_num_list[i], axis=0)

            # print(f"\n{i}. shift_num_list[i]: {shift_num_list[i]}\nrolled_fr1_radii: {rolled_fr1_radii}\nrecorded_fr_list[i + 1]: {recorded_fr_list[i + 1]}")

            # get the indices of rings that are onscreen for frame 1
            # onscreen here uses full screen height for radius rather than -heightpix/2 to heightpix/2 pixel co-ords
            onscreen_rings_1 = (rolled_fr1_radii > min_radius) & (rolled_fr1_radii < max_radius)

            # get the indices of rings that are onscreen for frame 2
            onscreen_rings_2 = (recorded_fr_list[i + 1] > min_radius) & (recorded_fr_list[i + 1] < max_radius)

            # get the indices of rings that are onscreen for both frames
            onscreen_rings = onscreen_rings_1 & onscreen_rings_2
            # print(f"onscreen_rings: {onscreen_rings}")

            # get the difference pairs of frames
            paired_fr_diff = recorded_fr_list[i + 1][onscreen_rings] - rolled_fr1_radii[onscreen_rings]
            # print(f"paired_fr_diff: {paired_fr_diff}")

            onscrn_rad_diff.append(paired_fr_diff)

        # flatten and convert to numpy array
        onscrn_rad_diff = np.array([item for sublist in onscrn_rad_diff for item in sublist])
        # print(f"onscrn_rad_diff: {np.shape(onscrn_rad_diff)}\n{onscrn_rad_diff}")

        # onscrn_diag_diff = []
        # # # loop through onscrn_rad_diff and get the diagonal distance between values (sqrt of a squared plus a squared)
        # for i in range(len(onscrn_rad_diff)):
        #     # get the diagonal distance between each pair of x and y values
        #     diag_dist = np.sqrt(onscrn_rad_diff[i] ** 2 + onscrn_rad_diff[i] ** 2)
        #     onscrn_diag_diff.append(diag_dist)
        # onscrn_diag_diff = np.array(onscrn_diag_diff)
        # print(f"onscrn_diag_diff: {np.shape(onscrn_diag_diff)}\n{onscrn_diag_diff}")

    # get the mean difference
    if background == 'flow_dots':
        # just horizontal differences
        diff_list = np.abs(onscrn_x_diff)

        dots_horiz_diff += np.mean(diff_list)

        # diagonal x/y differences
        diag_diff_list = np.abs(onscrn_diag_diff)

        dots_diag_diff += np.mean(onscrn_diag_diff)


    elif background == 'flow_rings':
        diff_list = np.abs(onscrn_rad_diff)
        rings_horiz_diff += np.mean(diff_list)

        diag_diff = None

print(f"dots_diag_diff: {dots_diag_diff}")
print(f"dots_horiz_diff: {dots_horiz_diff}")
print(f"rings_horiz_diff: {rings_horiz_diff}")

#
# # plot histogram of the mean_diff and onscreen_mean_diff, two plots side by side
# import matplotlib.pyplot as plt
#
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
#
# # if flow_dots, flatten arrays for mean_diff and onscreen_mean_diff
#
# ax1.hist(diff_list, bins=100)
# ax1.title.set_text(f'change between frames (mean: {round(mean_diff, 2)})')
#
# ax2.hist(onscrn_diag_diff, bins=100)
# ax2.title.set_text(f'diagonal change between frames (mean: {round(diag_diff, 2)})')
#
# plt.suptitle(f"{n_items} {background}. depth: {min_depth} to {max_depth}. flow_dir: {flow_dir}. flow_speed: {flow_speed}.")
# plt.show()


# # add values to a csv, if it doesn't exist, create it.
import os
import csv

flow_mean_speed_csv = 'flow_mean_speed.csv'
csv_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\project_stuff"
save_csv_path = os.path.join(csv_path, flow_mean_speed_csv)

# csv headers
csv_headers = ['fps', 'n_frames', 'flow_dir',
               'dots_speed', 'n_dots', 'dot_min_depth', 'dot_max_depth', 'dot_depth_adj',
               'ring_speed', 'n_rings', 'ring_min_depth', 'ring_max_depth', 'ring_depth_adj',
               'dots_diag_diff', 'dots_horiz_diff', 'rings_horiz_diff']

if not os.path.exists(save_csv_path):
    with open(save_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)

if background == 'flow_dots':
    flow_speed = dots_speed
    n_items = n_dots
    min_depth = dots_min_depth
    max_depth = dots_max_depth
    depth_adj = max_depth - min_depth
elif background == 'flow_rings':
    flow_speed = ring_speed
    n_items = n_rings
    min_depth = rings_min_depth
    max_depth = rings_max_depth
    depth_adj = max_depth - min_depth

# add values to csv
with open(save_csv_path, 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([fps, record_n_frames, flow_dir,
                     dots_speed, n_dots, dots_min_depth, dots_max_depth, dots_max_depth - dots_min_depth,
                     ring_speed, n_rings, rings_min_depth, rings_max_depth, rings_max_depth - rings_min_depth,
                     dots_diag_diff, dots_horiz_diff, rings_horiz_diff])









