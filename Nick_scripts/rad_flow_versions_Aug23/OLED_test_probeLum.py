import numpy as np
from psychopy import visual, core, event




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

    # print(f"\nz:\n{z}\nring_radii_array:\n{ring_radii_array}\nshift_num:\n{shift_num}\n")

    return z, ring_radii_array, ring_colours


# stop numpy from using scientific-notation
np.set_printoptions(suppress=True)

# screen details
monitor_name = 'Nick_work_laptop'  # 'OLED' or 'Nick_work_laptop'
if monitor_name == 'OLED':
    widthPix = 1920
    heightPix = 1080
    fps = 120
else:
    widthPix = 1920
    heightPix = 1080
    fps = 60

# flow direction
# flow_dir = np.random.choice([-1, 1])
flow_dir = 1  # -1 = expanding, 1 = contracting
print(f"flow_dir: {flow_dir}")

# colours and colour space
LumColor255Factor = 2.39538706913372
maxLum = 106  # 255 RGB
bgLumProp = .2  # .2  # todo: use .45 to match radial_flow_NM_v2.py, or .2 to match exp1
if monitor_name == 'OLED':
    bgLumProp = .0
bgLum = maxLum * bgLumProp
bgColor255 = bgLum * LumColor255Factor

# colour space
this_colourSpace = 'rgb1'  # values between 0 and 1
bgColor_rgb1 = bgLum / maxLum
this_bgColour = [bgColor_rgb1, bgColor_rgb1, bgColor_rgb1]

# Give dots a pale green colour, which is adj_dots_col different to the background
adj_flow_colour = .15
# Give dots a pale green colour, which is adj_flow_colour different to the background
flow_colour = [this_bgColour[0] - adj_flow_colour, this_bgColour[1], this_bgColour[2] - adj_flow_colour]
if monitor_name == 'OLED':  # darker green for low contrast against black background
    flow_colour = [this_bgColour[0], this_bgColour[1] + adj_flow_colour / 2, this_bgColour[2]]

# window
win = visual.Window([widthPix, heightPix],
                    # monitor="Nick_work_laptop",
                    units="pix", fullscr=True,
                    color=this_bgColour,
                    colorSpace=this_colourSpace)

# full screen mask to blend off edges and fade to black
# Create a raisedCosine mask array and assign it to a Grating stimulus (grey outside, transparent inside)
# this was useful http://www.cogsci.nl/blog/tutorials/211-a-bit-about-patches-textures-and-masks-in-psychopy
raisedCosTexture2 = visual.filters.makeMask(heightPix, shape='raisedCosine', fringeWidth=0.6, radius=[1.0, 1.0])
invRaisedCosTexture = -raisedCosTexture2  # inverts mask to blur edges instead of center
slab_width = 420

blankslab = np.ones((heightPix, slab_width))  # create blank slabs to put to left and right of image
mmask = np.append(blankslab, invRaisedCosTexture, axis=1)  # append blank slab to left
mmask = np.append(mmask, blankslab, axis=1)  # and right
# changed dotsmask color from grey, fades to black round edges which makes screen edges less visible
edge_mask = visual.GratingStim(win, mask=mmask, tex=None, contrast=1.0,
                               size=(widthPix, heightPix), units='pix', color='black')
fixation = visual.Circle(win, radius=2, units='pix', lineColor='white', fillColor='black', colorSpace=this_colourSpace)

mask_size = 150
dist_from_fix = 185
raisedCosTexture1 = visual.filters.makeMask(256, shape='raisedCosine', fringeWidth=0.3, radius=[1.0, 1.0])
probeMask1 = visual.GratingStim(win=win, mask=raisedCosTexture1, size=(mask_size, mask_size),
                                colorSpace=this_colourSpace, color=this_bgColour,
                                tex=None, units='pix', pos=[dist_from_fix + 1, dist_from_fix + 1])

'''
This script shows the flow_dots and flow_rings at the same time, 
useful for eye-balling their relative speeds.

Code will run until the escape key is pressed.
You can use 'up' and 'down' keys to increase or decrease the speed of the rings.

Things to condsider:
    We are pretty happy with dots (min_dept 1, max_depth 5.5, dots_speed .2, n_dots 5000)

    Rings are up for grabs, but they need a different max_depth and min_depth to dots, 
    so that the smallest ring is about the same size as the fixation, and largest is biggest than screen.

'''


# MOTION SPEED
# At the moment both rings and dots use this speed, but they can be set independently below.
# When the script is running the up and down keys can adjust the ring speed.
flow_speed = .08  # we use .2 on the 240Hz




print("\nflow_dots")

dots_speed = flow_speed
n_dots = 5000

# Changing dots_min_depth from .5 to one means that the proportion of dots onscreen increases from ~42% to ~82%.
# Therefore, I can half n_dots with little change in the number of dots onscreen, saving processing resources.
dots_min_depth = 1.0
dots_max_depth = 5.5


# with dot_array_spread = widthPix * 3, is 5760 on a 1920 monitor, close to orig value of 5000, but allows for scaling on OLED.
dot_array_spread = widthPix * 3

# initial array values.  x and y are scaled by z, so x and y values can be larger than the screen.
x = np.random.rand(n_dots) * dot_array_spread - dot_array_spread / 2
y = np.random.rand(n_dots) * dot_array_spread - dot_array_spread / 2

# depth values for the dots
z = np.random.rand(n_dots) * (dots_max_depth - dots_min_depth) + dots_min_depth

# x_flow and y_flow are the actual x and y positions of the dots, after being divided by their depth value.
x_flow = x / z
y_flow = y / z

# array of x, y positions of dots to pass to ElementArrayStim
dots_xys_array = np.array([x_flow, y_flow]).T

# Give dots a pale green colour, which is adj_dots_col different to the background
flow_dots = visual.ElementArrayStim(win, elementTex=None, elementMask='circle',  # orig used 'gauss'
                                    units='pix', nElements=n_dots, sizes=10,
                                    colorSpace=this_colourSpace, colors=flow_colour)
print(f"dots_speed: {dots_speed}")
print(f"n_dots: {n_dots}")
print(f"dots_min_depth: {dots_min_depth}")
print(f"dots_max_depth: {dots_max_depth}")


print("\nflow_rings")

# # # RINGS
# flow_speed * 8.9 gave a comparable measured speed, but it looks wrong.  Use 'up' and 'down' keys to adjust.
ring_speed = flow_speed
n_rings = 100
rings_min_depth = .1  # A value < 1 of .1 means that the closest ring's radius is 10x the size of the screen.

# set the limits on ring size
max_radius = heightPix  # Biggest ring is height of screen
min_radius = 10  # smallest ring is 10 pixels

# If I want the smallest radius to be 10 pixels, then the max depth of 108 (1080/108=10)
rings_max_depth = max_radius / min_radius

# adjust ring depth values by ring_depth_adj
ring_depth_adj = rings_max_depth - rings_min_depth

# RING COLOURS (alernating this_bgColour and flow_ring_colour (red to make them easy to see)
ring_colours = [this_bgColour, flow_colour] * int(n_rings / 2)

ring_size_list = [1080] * n_rings

# depth values are evenly spaces and in ascending order, so smaller rings are drawn on top of larger ones.
# stop=stop=rings_max_depth-(ring_depth_adj/n_rings) gives space the new ring to appear
ring_z_array = np.linspace(start=rings_min_depth, stop=rings_max_depth - (ring_depth_adj / n_rings), num=n_rings)

# the actual radii list is in descending order, so smaller rings are drawn on top of larger ones.
ring_radii_array = ring_size_list / ring_z_array

# # use ElementArrayStim to draw the rings
flow_rings = visual.ElementArrayStim(win, elementTex=None, elementMask='circle', interpolate=True,
                                     units='pix', nElements=n_rings, sizes=ring_radii_array,
                                     colors=ring_colours, colorSpace=this_colourSpace)
print(f"ring_speed: {ring_speed}")
print(f"n_rings: {n_rings}")
print(f"rings_min_depth: {rings_min_depth}")
print(f"rings_max_depth: {rings_max_depth}")

# PROBE
probe_size = 1  # can make them larger for testing new configurations etc
# probeVert = [(0, 0), (1, 0), (1, 1), (2, 1), (2, -1), (1, -1), (1, -2), (-1, -2), (-1, -1), (0, -1)]  # 5 pixels

# smaller probes for OLED
# # 3 pixels
# probeVert = [(0, 0), (1, 0), (1, 1), (2, 1), (2, -1),
#              (0, -1)]  # 3 pixels
# probeVert = [(0, 0), (1, 0), (1, 1), (2, 1),
#              (2, 0), (1, 0), (1, -1), (0, -1),
#              (0, -2), (-1, -2), (-1, -1), (0, -1)]  # 3 pixels

probeVert = [(0, 0), (2, 0), (2, 1), (1, 1), (1, -1), (-1, -1), (-1, -2), (0, -2)]  # 5 pixels


# # 1 pixel
# probeVert = [(0, 0), (1, 0), (1, 1), (0, 1)]  # 1 pixel


probe1 = visual.ShapeStim(win, vertices=probeVert, lineWidth=0, opacity=1, size=probe_size, interpolate=False,
                          colorSpace=this_colourSpace)

probe1.pos = (dist_from_fix, dist_from_fix)
this_probeColor = .5
probe1.fillColor = [this_probeColor, this_probeColor, this_probeColor]

# PRESENT STIMULI

while not event.getKeys(keyList=["escape"]):

    # CONTROLS
    # # if I press 'up', increase ring_speed by .1, if I press 'down', decrease ring_speed by .1
    # if event.getKeys(keyList=["up"]):
    #     ring_speed += .01
    #     print(f"ring_speed: {ring_speed}")
    # elif event.getKeys(keyList=["down"]):
    #     ring_speed -= .01
    #     print(f"ring_speed: {ring_speed}")

    # CONTROLS
    # If I press 'up', increase this_probeColor by .1, if I press 'down', decrease this_probeColor by .1
    # if I press 'right', increase this_probeColor by .01, if I press 'left', decrease this_probeColor by .01
    if event.getKeys(keyList=["up"]):
        this_probeColor += .1
        print(f"this_probeColor: {this_probeColor}")
    elif event.getKeys(keyList=["down"]):
        this_probeColor -= .1
        print(f"this_probeColor: {this_probeColor}")
    elif event.getKeys(keyList=["right"]):
        this_probeColor += .01
        print(f"this_probeColor: {this_probeColor}")
    elif event.getKeys(keyList=["left"]):
        this_probeColor -= .01
        print(f"this_probeColor: {this_probeColor}")

    # DOTS: get new depth_vals array (z) and dots_xys_array (x, y)
    z, dots_xys_array = new_dots_depth_and_pos(x, y, z, dots_speed, flow_dir,
                                               dots_min_depth, dots_max_depth)
    flow_dots.xys = dots_xys_array

    # RINGS: update depth values and ring colours and get new ring_radii_array
    ring_z_array, ring_radii_array, ring_colours = roll_rings_z_and_colours(ring_z_array, ring_colours,
                                                                            rings_min_depth, rings_max_depth,
                                                                            flow_dir, ring_speed, ring_size_list)
    flow_rings.sizes = ring_radii_array
    flow_rings.colors = ring_colours

    # update probe colour
    probe1.fillColor = [this_probeColor, this_probeColor, this_probeColor]

    # draw components
    flow_rings.draw()
    flow_dots.draw()
    edge_mask.draw()
    fixation.draw()
    probeMask1.draw()
    probe1.draw()

    win.flip()

win.close()











