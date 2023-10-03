import pyglet.gl as GL
import psychopy.tools.viewtools as vt
import psychopy.tools.mathtools as mt
import numpy as np
import psychopy
from psychopy import visual, event, monitors


'''Demo adapted from  demo from: https://discourse.psychopy.org/t/draw-dots-in-3d-space/8918.  
    Kathia, Sep '19: "Awesome! thanks! I tried it out and it worked."'''

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
win = psychopy.visual.Window(monitor=mon,
                             # size=(800, 800),  # check if aspect changes between sq and rect screen
                             size=(widthPix, heightPix),
                             color='Black', colorSpace='rgb1',
                             units='pix',  # openGL is not impacted by psychopy units, so I'll stick with pix
                             screen=display_number,
                             allowGUI=True, fullscr=True)

fps = 60
flow_speed_m_p_s = 1.0  # 1m/sec matches previous flow parsing study (Evans et al. 2020)
flow_speed_m_p_fr = flow_speed_m_p_s / fps  # 1.66 cm per frame = 1m per second
print(f"flow_speed_m_p_fr: {flow_speed_m_p_fr:.3f}m per frame")

'''
Section below is from Kathia, Sep '19, 
Awesome! thanks! I tried it out and it worked.'''
# Frustum
frustum = vt.computeFrustum(mon_width_m, scrAspect, view_dist_m, nearClip=0.01, farClip=10000.0)
P = vt.perspectiveProjectionMatrix(*frustum)

# Transformation for points (model/view matrix) - subtract view dist to place things in space
MV = mt.translationMatrix((0.0, 0.0, -view_dist_m))  # X, Y, Z



'''If the parameters are correct, then the frame containing the dots should be the same size as the screen.
However, the co-ordinates in cm space won't be literally the same as for the screen, as they are 'behind' the screen, (will be larger cm).
The red frame is 90% of the screen height = 0.1740375m * .9 = 0.15663375m
If the red frame is 15cm high, then it's visual angle at 57cm will be ~ 15 degrees.
So my dots, in cm should be whatever has the same visual angle at 107cm away?'''
red_frame_angle = vt.visualAngle(size=mon_height_m*.9, distance=view_dist_m, degrees=True)
print(f"red_frame_angle: {red_frame_angle} degrees")

def height_from_dist_and_deg(distance, visual_angle):
    return distance * np.tan(np.deg2rad(visual_angle))


check_red_frame_height = height_from_dist_and_deg(distance=view_dist_m, visual_angle=red_frame_angle)
print(f"check_red_frame_height: {check_red_frame_height}m")

dot_fr_height_m = height_from_dist_and_deg(distance=1.07, visual_angle=red_frame_angle)
print(f"dot_fr_height_m: {dot_fr_height_m}m")

win.projectionMatrix = P
win.viewMatrix = MV


'''draw a square of fixed size and use this to test depth'''
def sq_from_list(x_list, y_list, z_list):
    # testing depth with this triangle to see when it disappears/appears.
    GL.glColor3f(1.0, .1, .1)

    GL.glBegin(GL.GL_QUADS)

    for i in range(len(x_list)):
        GL.glVertex3f(x_list[i], y_list[i], z_list[i])
    GL.glEnd()

# this red square uses normalized units, so it will occcupy 90% of the screen height
x_sq_list = [-.95, .95, .95, -.95]
y_sq_list = [.95, .95, -.95, -.95]
z_sq_list = [1.0, 1.0, 1.0, 1.0]


# create array of random points
nPoints = 900
pos = np.zeros((nPoints, 3), dtype=np.float32)

# random X, Y
xy_width = dot_fr_height_m  # .33  # full width of the frame containing dots in m.
xy_bounds = xy_width / 2  # half width, so that the frame is centered on 0,0
print(f"xy_bounds: {xy_bounds}, xy_width: {xy_width}")
pos[:, :2] = np.random.uniform(low=-xy_bounds, high=xy_bounds, size=(nPoints, 2))


# random Z, use negative values between -nearClip and -farClip (e.g., -0.01 and -10000.0)
closest_z = -.5  # .5m away
furthest_z = -1.5  # 1.5m away
pos[:, 2] = np.random.uniform(closest_z, furthest_z, (nPoints,))


# flow direction
expanding = False
if expanding:
    flow_dir = 1
else:  # contracting
    flow_dir = -1


while 1:
    # --- render loop ---
    win.applyEyeTransform()
    # draw 3D stuff here

    GL.glColor3f(1.0, 1.0, 1.0)
    GL.glPointSize(5.0)
    GL.glBegin(GL.GL_POINTS)
    for i in range(nPoints):  # go over our array and draw the points using the coordinates
        # color can be varied with distance if you like
        GL.glVertex3f(*pos[i, :])  # position
    GL.glEnd()

    win.flip()



    sq_from_list(x_list=x_sq_list, y_list=y_sq_list, z_list=z_sq_list)



    # transform points -Z direction, in OpenGL -Z coordinates are forward
    pos[:, 2] += flow_speed_m_p_fr * flow_dir  # distance to move the dots per frame towards the viewer
    # todo: add multiplier for expand/contract


    # if a point its behind us, return to initial -Z position
    # todo: instead of giving it a new z, just max out its dotlife
    pos[:, 2] = np.where(pos[:, 2] > closest_z, furthest_z, pos[:, 2])
    # todo: update rule for contracting dots

    # todo: add dotlife which gives new x, y and z, max life is 166.67 ms, or 10 frames at 60fps.  respawn a z range -10 frames * flow_speed_m_p_fr

    # check events to break out of the loop!
    if len(event.getKeys()) > 0:
        break
    event.clearEvents()

win.close()



