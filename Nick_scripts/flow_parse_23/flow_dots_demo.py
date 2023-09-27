import numpy as np
from psychopy import visual, core, event, monitors


'''
This script is to work out how to match the optic flow settings from Simons flow parsing studies.
Dots appear to move from 207 to 107 cm away at 100cm per second (with a lifetime of 160ms).
This is based on a distance of 57.3cm from the screen.  
I've not added a short dot lifetime yet, does it need it?
'''



def find_angle(adjacent, opposite):
    """Finds the angle in a right triangle given the lengths of the adjacent and opposite sides.

    Args:
    adjacent: A numpy array of the lengths of the adjacent sides.
    opposite: The length of the side opposite the angle you want to find.

    Returns:
    A numpy array of the angles in degrees.
    """

    radians = np.arctan(opposite / adjacent)  # radians
    degrees = radians * 180 / np.pi  # degrees
    return degrees


def new_dots_z_and_pos(x_array, y_array, z_array, dots_speed, flow_dir, min_z, max_z,
                       reference_angle,
                       ):
    """
    This is a function to update flow_dots depth array and get new pixel co-ordinates
    using the original x_array and y_array.

    1. Update z_array by adding dots_speed * flow_dir to the current z values.
    2. adjust any values below dots_min_z or above dots_max_z.
    3. Convert depths (cm) to angles (degrees) using find_angle().
    4. scale depths by dividing by reference angle.
    5. scale x and y values by multiplying by scaled depths.
    6. put the new x_pos and y_pos co-ordinates into an array and transposes it.

    :param x_array: Original x_array positions for the dots (shape = (n_dots, 1))
    :param y_array: Original y_array positions for the dots (shape = (n_dots, 1))
    :param z_array: array of depth values for the dots (shape = (n_dots, 1))
    :param dots_speed: speed of the dots (float, smaller = slower, larger = faster)
    :param flow_dir: either 1 (contracting/inward/backwards) or -1 (expanding/outward/forwards)
    :param dots_min_z: default is .5, values below this are adjusted to dots_max_z
    :param dots_max_z: default is 5, values above this are adjusted to dots_min_z
    :param reference_angle: angle in degrees of the reference distance (57.3cm)
    :return: updated_z_array, new dots_pos_array
    """

    # # 1. Update z (depth values) # #
    # Add dots_speed * flow_dir to the current z values.
    updated_z_array = z_array + dots_speed * flow_dir

    # 2. adjust any depth values below min_z or above max_z by z_adjust
    z_adjust = max_z - min_z
    # adjust updated_z_array values less than min_z by adding z_adjust
    less_than_min = (updated_z_array < min_z)
    # print(f"less_than_min:\n{less_than_min}")
    updated_z_array[less_than_min] += z_adjust
    # adjust updated_z_array values more than max_z by subtracting z_adjust
    more_than_max = (updated_z_array > max_z)
    updated_z_array[more_than_max] -= z_adjust
    # print(f"updated_z_array (clipped):\n{updated_z_array}\n")

    # 3. convert depths to angles
    z_array_deg = find_angle(adjacent=updated_z_array, opposite=frame_size_cm)
    # print(f"z_array_deg:\n{z_array_deg}")

    # 4. scale depths by dividing by reference angle
    scale_factor_array = z_array_deg / reference_angle
    # print(f"scale_factor_array:\n{scale_factor_array}")

    # 5. scale x and y values by multiplying by scaled depths
    scaled_x = x_array * scale_factor_array
    scaled_y = y_array * scale_factor_array

    # 6. scale x and y values by multiplying by scaled depths
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
    
    # replace these with new x and y values
    x_array[replace_mask] = np.random.uniform(-dot_boundary/2, dot_boundary/2, np.sum(replace_mask))
    y_array[replace_mask] = np.random.uniform(-dot_boundary/2, dot_boundary/2, np.sum(replace_mask))
    
    # reset life of replaced dots to 0
    dotlife_array[replace_mask] = 0
    
    return dotlife_array, x_array, y_array



# initialize window
monitor_name = 'Nick_work_laptop'  # Nick_work_laptop, HP_24uh
mon = monitors.Monitor(monitor_name)

# screen size in pixels and cm
widthPix = int(mon.getSizePix()[0])  # 1920
heightPix = int(mon.getSizePix()[1])  # 1080
mon_width_cm = mon.getWidth()  # monitor width in cm, 34.5 for ASUS, 30.94 for Nick_work_laptop
view_dist_cm = mon.getDistance()  # viewing distance in cm, 57.3
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

# WINDOW SPEC
win = visual.Window(monitor=mon, size=(widthPix, heightPix), colorSpace=this_colourSpace, color=this_bgColour,
                    units='pix', allowGUI=False, fullscr=True)




# fustrum dimensions
near_plane_cm = 107  # later use 107 to match studies (.3?)
far_plane_cm = 207  # later use 207 to match studies (.3?)

# square dimensions
frame_size_cm = mon_width_cm / 2  # size of square in cm

# motion speed in cm/s
fps = 60
flow_speed_cm_p_sec = 100
flow_speed_cm_p_fr = flow_speed_cm_p_sec / fps  # 1.66 cm per frame = 1m per second


# initialise dots
n_dots = 300  # later try 300 to match flow parsing studies, or 10000 to match our rad flow studies
flow_dots = visual.ElementArrayStim(win, elementTex=None, elementMask='circle',
                                    units='cm', nElements=n_dots, sizes=.2,
                                    colorSpace=this_colourSpace,
                                    colors=flow_colour,
                                    )
# dot lifetime ms
dot_life_max_ms = 160
dot_life_max_fr = dot_life_max_ms / 1000 * fps


'''Part 1, from screen size, to 'actual size' at distance'''
# This square is the frame, all dots appear within it.  Dots are supposed to appear to move from 207 to 107cm away, 
# so dots are 50cm to 150cm 'behind' this square frame.
# check monitor scale is 100% on windows (default/recommended is 150%) so cms are correct
square = visual.Rect(win, width=frame_size_cm, height=frame_size_cm, units='cm', fillColor=None, lineColor='white', opacity=.5)


# When viewing from 57.3cm, dva = cm_on_screen.
# I'm going to scale x and y values by depth so that they are consistent with the depths I want.
# I'll do this based on the difference in angle (and therefore onscreen size) between
# my reference angle (57.3) and dot depths (between 107 and 207cm).
# Note if I convert dot locations to pixels, I need to dived by 2 as (0, 0) in  centre of screen,
# but if I stick with everything in cm, I don't need to.
ref_angle = find_angle(adjacent=view_dist_cm, opposite=frame_size_cm)
print(f"ref_angle: {ref_angle}")


# initialize dots to fit in window
x_array = np.random.uniform(-frame_size_cm/2, frame_size_cm/2, n_dots)  # x values in cm
y_array = np.random.uniform(-frame_size_cm/2, frame_size_cm/2, n_dots)  # y values in cm

# initialize depths in cm
z_array_cm = np.random.uniform(near_plane_cm, far_plane_cm, n_dots)    # depths in cm

# initialize lifetime
dot_lifetime_array = np.random.randint(0, dot_life_max_fr, n_dots)

contracting = False
if contracting:
    flow_dir = 1
else:  # expanding
    flow_dir = -1

# PRESENT STIMULI
while not event.getKeys(keyList=["escape"]):

    # update dot lifetime
    # dot_lifetime_array += 1
    dotlife_array, x_array, y_array = update_dotlife(dotlife_array=dot_lifetime_array, dot_max_fr=dot_life_max_fr, 
                                                     x_array=x_array, y_array=y_array, dot_boundary=frame_size_cm)

    # update z and xys
    z_array_cm, scaled_xys = new_dots_z_and_pos(x_array=x_array, y_array=y_array, z_array=z_array_cm,
                                                dots_speed=flow_speed_cm_p_fr, flow_dir=flow_dir,
                                                min_z=near_plane_cm, max_z=far_plane_cm,
                                                reference_angle=ref_angle)
    flow_dots.xys = scaled_xys

    # draw
    flow_dots.draw()
    square.draw()

    win.flip()

win.close()

print("Finished successfully")



