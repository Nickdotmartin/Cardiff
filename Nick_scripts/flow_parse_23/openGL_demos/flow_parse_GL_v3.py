import pyglet.gl as gl
import psychopy.tools.viewtools as vt
import psychopy.tools.mathtools as mt
import numpy as np
import psychopy
from psychopy import visual, event, monitors
from psychopy.tools.monitorunittools import cm2pix, pix2cm


'''Demo adapted from  demo from: https://discourse.psychopy.org/t/draw-dots-in-3d-space/8918.  
    Kathia, Sep '19: "Awesome! thanks! I tried it out and it worked."'''

def angle_from_dist_and_height(height, distance):
    """
    Gives the visual angle of an object from its distance and height.
    :param height: Actual size of screen or size on screen in meters.
    :param distance: from screen in meters (e.g., .573m) or from screen plus z_values (e.g., .573m + 1m)
    :return: angle in degrees
    """
    return np.rad2deg(np.arctan(height / distance))


def height_from_dist_and_deg(distance, visual_angle):
    """
    Gives the height of an object from its distance and visual angle.
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

    # dot settings
    # gl.glColor3f(1.0, 1.0, 1.0)
    gl.glColor3f(flow_colour_rgb1[0], flow_colour_rgb1[1], flow_colour_rgb1[2])
    gl.glPointSize(5.0)

    # draw the dots
    gl.glBegin(gl.GL_POINTS)
    for i in range(n_points):
        gl.glVertex3f(*dots_pos_array[i, :])
    gl.glEnd()


def draw_filled_circle_gl(x, y, z, radius, n_points=100, fill_colour=[0.0, 0.0, 1.0]):
    """
    Function to draw a filled circle in openGL.  All values are in meters.

    :param x: x position of the centre of the circle.
    :param y: y position of the centre of the circle.
    :param z: z position of the centre of the circle (distance).
    :param radius: radius of the circle (in meters).
    :param n_points: number of points to use to draw the circle.
    :param fill_colour: colour of the circle (rgb1).
    """

    # --- render loop ---
    # Apply the current view and projection matrices specified by ‘viewMatrix’ and ‘projectionMatrix’ using ‘immediate mode’ OpenGL.
    # Subsequent drawing operations will be affected until ‘flip()’ is called.
    # All transformations in GL_PROJECTION and GL_MODELVIEW matrix stacks will be cleared (set to identity) prior to applying.
    win.applyEyeTransform()

    # dot settings
    # gl.glColor3f(1.0, 1.0, 1.0)
    gl.glColor3f(fill_colour[0], fill_colour[1], fill_colour[2])
    gl.glPointSize(5.0)

    # draw the dots
    gl.glBegin(gl.GL_POLYGON)
    for i in range(n_points):
        gl.glVertex3f(x + radius * np.cos(2 * np.pi * i / n_points), y + radius * np.sin(2 * np.pi * i / n_points), z)
    gl.glEnd()



def draw_2d_probe_gl(x, y, fill_colour=[1.0, 1.0, 1.0], n_pix=5):
    """
    Function to draw a 2d probe in openGL.  All values are in meters.
    :param x: x position of probe
    :param y: y position of probe
    :param fill_colour: colour of the probe (rgb1).
    :param n_pix: default is a 5 pixel probe, but can do 3pix probe for OLED
    """


    if n_pix == 5:  # default, 5-pixel probes
        probeVert = [(0, 0), (1, 0), (1, 1), (2, 1), (2, -1), (1, -1), (1, -2), (-1, -2), (-1, -1), (0, -1)]  # 5 pixels
    elif n_pix == 3:  # smaller, 3-pixel probes for OLED
        probeVert = [(0, 0), (1, 0), (1, 1), (2, 1),
                     (2, 0), (1, 0), (1, -1), (0, -1),
                     (0, -2), (-1, -2), (-1, -1), (0, -1)]
    else:
        print(f"n_pix must be 3 or 5, not {n_pix}")
        raise ValueError

    # # Set your projection matrix
    # gl.glMatrixMode(gl.GL_PROJECTION)
    # gl.glOrtho(-1920/2, 1920/2, -1080/2, 1080/2, -1, 1);
    # # Restore the default matrix mode
    # gl.glMatrixMode(gl.GL_MODELVIEW)

    gl.glDisable(gl.GL_TEXTURE_2D)

    # --- render loop ---
    gl.glBegin(gl.GL_LINE_LOOP)
    for i in range(len(probeVert)):
        gl.glVertex3f(x + probeVert[i][0], y + probeVert[i][1], 0.0)
    gl.glEnd()







def check_z_start_bounds(z_array, closest_z, furthest_z, dot_life_max_fr, dot_lifetime_array, flow_dir):
    """
    check all z values.  If they are out of bounds (too close when expanding or too far when contracting), then
    set their dot life to max so they are redrawn with new x, y and z values.

    :param z_array: array of current dot distances
    :param closest_z: near boundary for z values (relevant when expanding)
    :param furthest_z: far boundary for z values (relavant when contracting)
    :param dot_life_max_fr: maximum lifetime of a dot in frames.
    :param dot_lifetime_array: array of dot lifetimes (ints) between 0 and dot_max_fr.
    :param flow_dir: either 1 (contracting/inward/backwards) or -1 (expanding/outward/forwards):
    :return: updated dot_lifetime_array
    """

    # if expanding, check if any z values are too close, and if so, set their dot life to max
    if flow_dir == -1:  # expanding
        dot_lifetime_array = np.where(z_array > closest_z, dot_life_max_fr, dot_lifetime_array)
    elif flow_dir == 1:  # contracting
        dot_lifetime_array = np.where(z_array < furthest_z, dot_life_max_fr, dot_lifetime_array)

    return dot_lifetime_array


def update_dotlife(dotlife_array, dot_max_fr,
                   x_array, y_array, z_array,
                   x_bounds, y_bounds, z_start_bounds
                   ):
    """
    This is a function to update the lifetime of the dots.

    1. increment all dots by 1
    2. make a mask of any to be replaced (life > max_life)
    3. replace these with new x and y values
    4. reset life of replaced dots to 0

    :param dotlife_array: np.array of dot lifetimes (ints) between 0 and dot_max_fr.
    :param dot_max_fr: maximum lifetime of a dot in frames.
    :param x_array: np.array of x positions of dots (in meters).
    :param y_array: np.array of y positions of dots (in meters).
    :param y_array: np.array of z positions of dots (in meters).

    :param x_bounds: value passed for distribution of x_values, from -x_bounds to x_bounds.  Half the width of the array.
    :param y_bounds: value passed for distribution of y_values, from -y_bounds to y_bounds.  Half the height of the array.
    :param z_start_bounds: tuple, values passed for distribution of z_values, from z_start_bounds[0] to z_start_bounds[1].
    :return: updated dotlife_array, x_array, y_array, z_array
    """

    # increment all dots by 1
    dotlife_array += 1

    # make a mask of any to be replaced (life > max_life)
    replace_mask = (dotlife_array > dot_max_fr)

    # replace these with new x and y values (from same distribution as originals)
    x_array[replace_mask] = np.random.uniform(low=-x_bounds, high=x_bounds, size=np.sum(replace_mask))
    y_array[replace_mask] = np.random.uniform(low=-y_bounds, high=y_bounds, size=np.sum(replace_mask))
    z_array[replace_mask] = np.random.uniform(low=z_start_bounds[0], high=z_start_bounds[1], size=np.sum(replace_mask))

    # reset life of replaced dots to 0
    dotlife_array[replace_mask] = 0

    return dotlife_array, x_array, y_array, z_array







# initialize window
monitor_name = 'Nick_work_laptop'  # Nick_work_laptop, HP_24uh
mon = monitors.Monitor(monitor_name)

# screen size in pixels and cm
widthPix = int(mon.getSizePix()[0])  # 1920
heightPix = int(mon.getSizePix()[1])  # 1080
mon_width_cm = mon.getWidth()  # monitor width in cm, 34.5 for ASUS, 30.94 for Nick_work_laptop
view_dist_cm = mon.getDistance()  # viewing distance in cm, 57.3
scrAspect = widthPix / heightPix  # widthPix / heightPix
print(f"scrAspect: {scrAspect}")

# openGl uses meters, not centimeters, so convert my values here.
view_dist_m = view_dist_cm / 100  # original code had .50 commented as # 50cm
mon_width_m = mon_width_cm / 100  # # original code had 0.53 commented as # 53cm
mon_height_m = mon_width_m / scrAspect
print(f"view_dist_m: {view_dist_m:.2f}m")
print(f"mon_width_m: {mon_width_m:.2f}m")
print(f"mon_height_m: {mon_height_m:.2f}m")


# screen number
display_number = 1  # 0 indexed, 1 for external display, 0 for internal
if monitor_name in ['asus_cal', 'Nick_work_laptop', 'NickMac', 'OLED', 'ASUS_2_13_240Hz']:
    display_number = 0


# Create a window
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

win = psychopy.visual.Window(monitor=mon,
                             # size=(800, 800),  # check if aspect changes between sq and rect screen
                             size=(widthPix, heightPix),
                             color=this_bgColour, colorSpace=this_colourSpace,
                             units='pix',  # openGL is not impacted by psychopy units, so I'll stick with pix
                             screen=display_number,
                             allowGUI=True, fullscr=True)

fps = 60

# flow speed settings
flow_speed_m_p_s = 1.0  # 1m/sec matches previous flow parsing study (Evans et al. 2020)
flow_speed_m_p_fr = flow_speed_m_p_s / fps  # 1.66 cm per frame = 1m per second
print(f"flow_speed_m_p_fr: {flow_speed_m_p_fr:.3f}m per frame")


# OPENGL SETTINGS for 3d projection
# Frustum
frustum = vt.computeFrustum(mon_width_m, scrAspect, view_dist_m, nearClip=0.01, farClip=10000.0)
P = vt.perspectiveProjectionMatrix(*frustum)

# Transformation for points (model/view matrix) - subtract view dist to place things in space
MV = mt.translationMatrix((0.0, 0.0, -view_dist_m))  # X, Y, Z

win.projectionMatrix = P
win.viewMatrix = MV

# set up for a different background colour in openGL
gl.glClearColor(bgColor_rgb1, bgColor_rgb1, bgColor_rgb1, 0)  # final value is alpha (transparency)
gl.glClear(gl.GL_COLOR_BUFFER_BIT)


'''dots settings'''
# create array of random points
n_dots = 300
# pos = np.zeros((n_dots, 3), dtype=np.float32)


# dot distance settings
closest_z = -.5  # .5m away
furthest_z = -1.5  # 1.5m away
closest_dist_m = closest_z - view_dist_m
furthest_dist_m = furthest_z - view_dist_m
print(f"closes_dist_m: {closest_dist_m}m = closest_z: {closest_z}m - view_dist_m: {view_dist_m}m")
print(f"furthest_dist_m: {furthest_dist_m}m = furthest_z: {furthest_z}m - view_dist_m: {view_dist_m}m")

# initialise z array with random distances (meters) based on values above.
# z_array = np.random.uniform(closest_z, furthest_z, (n_dots,))
z_array = np.random.uniform(low=closest_z, high=furthest_z, size=n_dots)


# dot height and width settings
'''
For the dots to fill the screen, but be a distance 'behind' the screen, they should be be drawn in a space 'bigger' than the screen (then appears smaller at distance).
To calculate the size of the frame containing the dots (x and y), I need the visual angle of the screen, and the distance of the frame from the screen.
I can then calculate the size of the object with that visual angle at a greater distance.
If I want to check they are fitting on the screen, add a border (e.g., scale to 90%).
'''
scale = .9  # use 1.0 for full screen

# get the angle of the screen width and height
dot_frame_angles = vt.visualAngle(size=[mon_width_m * scale, mon_height_m * scale], distance=view_dist_m, degrees=True)
print(f"dot_frame_angles (w, h): {dot_frame_angles} degrees")

# use visual angle and distance (to screen, and then beyond) to get dot array size in meters
dot_fr_width_m = height_from_dist_and_deg(distance=closest_dist_m, visual_angle=dot_frame_angles[0])
dot_fr_height_m = height_from_dist_and_deg(distance=closest_dist_m, visual_angle=dot_frame_angles[1])
print(f"dot_fr_width_m: {dot_fr_width_m}m")
print(f"dot_fr_height_m: {dot_fr_height_m}m")

# initialise dot arrays with random positiion (meters) based on size calculated above.
x_array = np.random.uniform(low=-dot_fr_width_m/2, high=dot_fr_width_m/2, size=n_dots)
y_array = np.random.uniform(low=-dot_fr_height_m/2, high=dot_fr_height_m/2, size=n_dots)


# dot lifetime settings
dot_life_max_ms = 167  # maximum lifetime in ms, 167 to match Evans et al., 2020.
dot_life_max_fr = int(dot_life_max_ms / 1000 * fps)  # max life in frames
print(f"dot_life_max_fr: {dot_life_max_fr}")


# initialize lifetime of each dot (in frames)
dot_lifetime_array = np.random.randint(0, dot_life_max_fr, n_dots)

# I'm going to use dot_life to deal with dots whose z is out of bounds to close or far.
# Instead of just giving them a new z_value, I'll max out their dotlife, which will give them a new x, y and z.
'''
dots move at 1 meter per second, the max distance they travel is 1 meter, which would take 1 second (fps).  
Their lifetime is 166.67ms, or 10 frames at 60fps.
There is no point giving dots a new z value too close to the boundary, where they would have to be redrawn after a couple of frames.
So the z_start_bounds should take this into account - subtract the max distance the dots can travel in their lifetime.
'''
max_dist_in_life = flow_speed_m_p_fr * dot_life_max_fr



# flow direction
# either 1 (contracting/inward/backwards) or -1 (expanding/outward/forwards):
# 1 is contracting/inward/backwards, -1 is expanding/outward/forwards

expanding = False
if expanding:
    flow_dir = -1
    z_start_bounds = [closest_z - max_dist_in_life, furthest_z]
    flow_dir_name = 'expanding'
else:  # contracting
    flow_dir = 1
    z_start_bounds = [closest_z, furthest_z + max_dist_in_life]
    flow_dir_name = 'contracting'
print(f"flow_dir: {flow_dir}, z_start_bounds: {z_start_bounds}, flow_dir_name: {flow_dir_name}")


# add additional psychopy stimuli
fixation = visual.Circle(win, radius=2, units='pix', lineColor='white', fillColor='black', colorSpace=this_colourSpace,
                            lineWidth=1, edges=32, interpolate=True, autoLog=False)

# Create a raisedCosine mask array and assign it to a Grating stimulus (grey outside, transparent inside)
mask_size = 150
probe_ecc = 4  # degrees from fixation
this_bgColour = 'green'
view_dist_pix = widthPix / mon_width_cm * view_dist_cm  # used for calculating dist_from_fix (probes at 4 degrees)

dist_from_fix = int((np.tan(np.deg2rad(probe_ecc)) * view_dist_pix) / np.sqrt(2))
print(f"dist_from_fix: {dist_from_fix} pixels")

raisedCosTexture1 = visual.filters.makeMask(256, shape='raisedCosine', fringeWidth=0.3, radius=[1.0, 1.0])
probeMask1 = visual.GratingStim(win=win, mask=raisedCosTexture1, size=(mask_size, mask_size),
                                colorSpace=this_colourSpace, color=this_bgColour, opacity=1,
                                tex=None, units='pix', pos=[dist_from_fix + 1, dist_from_fix + 1])
probeMask2 = visual.GratingStim(win=win, mask=raisedCosTexture1, size=(mask_size, mask_size),
                                colorSpace=this_colourSpace, color=this_bgColour, opacity=2,
                                units='pix', tex=None, pos=[-dist_from_fix - 1, dist_from_fix + 1])
probeMask3 = visual.GratingStim(win=win, mask=raisedCosTexture1, size=(mask_size, mask_size),
                                colorSpace=this_colourSpace, color=this_bgColour,
                                units='pix', tex=None, pos=[-dist_from_fix - 1, -dist_from_fix - 1])
probeMask4 = visual.GratingStim(win=win, mask=raisedCosTexture1, size=(mask_size, mask_size),
                                colorSpace=this_colourSpace, color=this_bgColour,
                                units='pix', tex=None, pos=[dist_from_fix + 1, -dist_from_fix - 1])

# opengl probemask with draw circle
# calculate radius in meters from mask_size diameter in pixels
circle_cl_radius_m = pix2cm(mask_size / 2.7, mon) / 100  # radius in cm
print(f"circle_cl_radius_m: {circle_cl_radius_m}m")

# calculate dist_from_fix in meters from dist_from_fix in pixels
dist_from_fix_m = pix2cm(dist_from_fix + 1, mon) / 100  # radius in cm
print(f"dist_from_fix (pixels): {dist_from_fix + 1}, meters: {dist_from_fix_m}")



# PROBEs
probe_size = 10  # can make them larger for testing new configurations etc
probeVert = [(0, 0), (1, 0), (1, 1), (2, 1), (2, -1), (1, -1), (1, -2), (-1, -2), (-1, -1), (0, -1)]  # 5 pixels


probe = visual.ShapeStim(win, vertices=probeVert, lineWidth=0, opacity=1, size=probe_size, interpolate=False,
                         # fillColor=(1.0, 1.0, 1.0),
                         fillColor=(1.0, 0.0, 0.0),
                         colorSpace=this_colourSpace)
probe.pos = (dist_from_fix, dist_from_fix)





while 1:

    # win.applyEyeTransform(clearDepth=True)

    probeMask1.draw()
    probeMask2.draw()
    probeMask3.draw()
    probeMask4.draw()
    fixation.draw()
    probe.draw()
    # win.applyEyeTransform(clearDepth=True)


    # update distance array: subtract as OpenGL distance is negative (psychopy was +ive).
    z_array -= flow_speed_m_p_fr * flow_dir  # distance to move the dots per frame towards/away from viewer

    # check if any z values are out of bounds (too close when expanding or too far when contracting),
    # if so, set their dot life to max, so they are given new x, y and z values by update_dotlife() below.
    dot_lifetime_array = check_z_start_bounds(z_array, closest_z, furthest_z, dot_life_max_fr, dot_lifetime_array, flow_dir)

    # update dot lifetime, give new x, y, z coords to dots whose lifetime is max.
    dotlife_array, x_array, y_array, z_array = update_dotlife(dotlife_array=dot_lifetime_array, dot_max_fr=dot_life_max_fr,
                                                              x_array=x_array, y_array=y_array, z_array=z_array,
                                                              x_bounds=dot_fr_width_m/2, y_bounds=dot_fr_height_m/2,
                                                              z_start_bounds=z_start_bounds)
    # draw stimuli
    # draw_flow_dots(x_array=x_array, y_array=y_array, z_array=z_array,
    #                                flow_colour_rgb1=flow_colour)


    # draw filled circle
    # draw_filled_circle_gl(x=dist_from_fix_m, y=dist_from_fix_m, z=0.0,
    #                       radius=circle_cl_radius_m, n_points=100, fill_colour=[0.0, 0.0, 1.0])
    # draw_filled_circle_gl(x=dist_from_fix_m, y=-dist_from_fix_m, z=0.0,
    #                       radius=circle_cl_radius_m, n_points=100, fill_colour=[0.0, 0.0, 1.0])
    # draw_filled_circle_gl(x=-dist_from_fix_m, y=-dist_from_fix_m, z=0.0,
    #                       radius=circle_cl_radius_m, n_points=100, fill_colour=[0.0, 0.0, 1.0])
    # draw_filled_circle_gl(x=-dist_from_fix_m, y=dist_from_fix_m, z=0.0,
    #                       radius=circle_cl_radius_m, n_points=100, fill_colour=[0.0, 0.0, 1.0])

    # draw 2d probe
    gl.glEnable(gl.GL_TEXTURE_2D)
    draw_2d_probe_gl(x=0, y=0, fill_colour=[1.0, 1.0, 1.0], n_pix=5)

    win.flip()

    # check events to break out of the loop!
    if len(event.getKeys()) > 0:
        break
    event.clearEvents()

win.close()



