import pyglet.gl as GL
import psychopy.tools.viewtools as vt
import psychopy.tools.mathtools as mt
import numpy as np
import psychopy
from psychopy import visual, event, monitors

'''
This script is adapted from one of Kathia's codes on https://discourse.psychopy.org/t/draw-dots-in-3d-space/8918/11
with comment "Awesome! thanks! I tried it out and it worked..."

Great article for understanding & visualising fristrum: http://relativity.net.au/gaming/java/Frustum.html

'''

# initialize window
monitor_name = 'Nick_work_laptop'  # Nick_work_laptop, HP_24uh
mon = monitors.Monitor(monitor_name)

# screen size in pixels and cm
widthPix = int(mon.getSizePix()[0])  # 1920
heightPix = int(mon.getSizePix()[1])  # 1080
mon_width_cm = mon.getWidth()  # monitor width in cm, 34.5 for ASUS, 30.94 for Nick_work_laptop
view_dist_cm = mon.getDistance()  # viewing distance in cm, 57.3

# PsychoPy computeFrustum requires distances in m (not cm) so adjust here.
view_dist_m = view_dist_cm / 100  # original code had .50 commented as # 50cm
mon_width_m = mon_width_cm / 100  # # original code had 0.53 commented as # 53cm
scrAspect = widthPix / heightPix  # widthPix / heightPix
mon_height_m = mon_width_m / scrAspect
print(f"view_dist_m: {view_dist_m}")
print(f"mon_width_m: {mon_width_m}")
print(f"scrAspect: {scrAspect}")
print(f"mon_height_m: {mon_height_m}")

# screen number
display_number = 1  # 0 indexed, 1 for external display, 0 for internal
if monitor_name in ['asus_cal', 'Nick_work_laptop', 'NickMac', 'OLED', 'ASUS_2_13_240Hz']:
    display_number = 0


# Create a window
win = psychopy.visual.Window(
    monitor=mon,
    # size=(800, 800),  # check if aspect changes between sq and rect screen
    size=(widthPix, heightPix),
    color='Black',
    # colorSpace='rgb1',
    colorSpace='rgb',
    # units='deg',
    units='pix',
    # units='cm',
    screen=display_number,
    allowGUI=True,
    fullscr=False)


# screen params
''' Settings from exp2 in Detection of scene-relative object movement and optic flow parsing across the adult lifespan. Evans, et al., 2020
"Random uniform onscreen two-dimensional dot were generated before assigning the dots random depth locations spanning the full screen size (47.3° × 29.6°)
 between 107 cm and 207 cm (57 cm viewing distance + a uniform sample in the range [50, 150] cm) from the observer."
'''

# todo: frustum plane values are positive realworld distances, the near plane is closer (small +ive) to the viewer than the far plane (large +ive)
# todo: the z axis is negative in OpenGL, so moving behind the camera goes from the near plane (small -ive) to the far plane (larger -ive).
# todo: behind the scenes, the z values between the clipping planes is actually a non-linear transform from the near plane (-1) to the far plane (+1), but I can ignore this).

# fustrum dimensions - should be positive values as these are distances in real-world values.
near_plane_m = .5  # 0.5  # matches viewdist later use 1.07 to match studies (.3?)
far_plane_m = 150  # 1.5  # 2.0  # later use 2.07 to match studies (.3?)
print(f"viewer to screen: {view_dist_m}, screen to near plane: {near_plane_m}, near plane to far plane: {far_plane_m - near_plane_m}")
print(f"viewer to screen: {view_dist_m}, viewer to near plane: {view_dist_m + near_plane_m}, viewer to far plane: {view_dist_m + far_plane_m}")


# motion speed in cm/s
fps = 60
flow_speed_cm_p_sec = 100  # 1m/sec matches previous flow parsing study (Evans et al. 2020)
flow_speed_cm_p_fr = flow_speed_cm_p_sec / fps  # 1.66 cm per frame = 1m per second
flow_speed_m_p_fr = flow_speed_cm_p_fr / 100  # convert to m per frame
print(f"flow_speed_m_p_fr: {flow_speed_m_p_fr:.2f}m/s")

# Frustum
'''
computeFrustum info from https://docs.psychopy.org/api/tools/viewtools.html#psychopy.tools.viewtools.computeFrustum
scrWidth, scrDist, nearClip, farClip are all positive floats given in meters.  
nearClip (float) – Distance to the near clipping plane in meters from the viewer. Should be at least less than view_dist_m.
farClip (float) – Distance to the far clipping plane from the viewer in meters. Must be >nearClip.
'''

# frustum = vt.computeFrustum(scrWidth=mon_width_m, scrAspect=scrAspect, scrDist=view_dist_m, nearClip=.01, farClip=1000)
# frustum = vt.computeFrustum(scrWidth=mon_width_m, scrAspect=scrAspect, scrDist=view_dist_m, nearClip=near_plane_m, farClip=far_plane_m)
frustum = vt.computeFrustum(scrWidth=mon_width_m, scrAspect=scrAspect, scrDist=view_dist_m, nearClip=0.01, farClip=1000)
# since I'm not changing the viewpoint, it might be better to use
P = vt.perspectiveProjectionMatrix(*frustum)  # apply the frustum to the projection matrix
win.projectionMatrix = P  # set the projection matrix

# todo: I think all z values after the projection matrix are inverted (negative) because the z axis is inverted in OpenGL, that is, moving away from the viewer is in negative values.


# Transformation for points (model/view matrix)
'''Without moving back in z (negative value), the points will be drawn at the near clipping plane, even with depth pushed back.'''
MV = mt.translationMatrix((0.0, 0.0, -view_dist_m))  # X, Y, Z
# MV = mt.translationMatrix((0.0, 0.0, 0.0))  # X, Y, Z
win.viewMatrix = MV

trfm_near_plane_m = -near_plane_m
trfm_far_plane_m = -far_plane_m
print(f"near_plane_m: {near_plane_m}, trfm_near_plane_m: {trfm_near_plane_m}")
print(f"far_plane_m: {far_plane_m}, trfm_far_plane_m: {trfm_far_plane_m}")


#
# # check depth with this trianglei
# def tri_from_list(x_list, y_list, z_list):
#     # testing depth with this triangle to see when it disappears/appears.
#     GL.glColor3f(1.0, .5, .5)
#
#     GL.glBegin(GL.GL_TRIANGLES)
#
#     for i in range(len(x_list)):
#         GL.glVertex3f(x_list[i], y_list[i], z_list[i])
#     GL.glEnd()
#
# # x_list = [0.0, -0.2, 0.2]
# # y_list = [0.2, -0.2, -0.2]
# x_list = np.random.uniform(-mon_height_m, mon_height_m, 3)
# y_list = np.random.uniform(-mon_height_m, mon_height_m, 3)
# # z_list = [1., 1., 1.]
# # all depth values are given as negative values, as the z axis is inverted in OpenGL.
# # z_list = np.random.uniform(-near_plane_m, -far_plane_m, 3)  #
# z_list = np.random.uniform(trfm_near_plane_m, trfm_far_plane_m, 3)  #

change_by = 0.1  # how much to change x, y, z by when pressing arrow keys, 'i', 'o', or 'z'. 'x'.


# create array of random points
nPoints = 900
pos = np.zeros((nPoints, 3), dtype=np.float32)

# random X, Y
pos[:, :2] = np.random.uniform(-mon_height_m, mon_height_m, (nPoints, 2))
# random z
# all depth values are given as negative values, as the z axis is inverted in OpenGL.

# pos[:, 2] = np.random.uniform(-near_plane_m, -far_plane_m, (nPoints,))  # negative z
pos[:, 2] = np.random.uniform(trfm_near_plane_m, trfm_far_plane_m, (nPoints,))  # negative z



# flow direction
expanding = True
if expanding:  # we are moving dots towards camera, from large -ive to smaller -ive, so flow_dir is +ive (e.g., -10 + 7 = -3)
    flow_dir = 1
else:  # contracting, # we are moving dots away from camera, from smaller -ive to larger -ive, so flow_dir is -ive (e.g., -1 - 5 = -6)
    flow_dir = -1

# while 1:
while not event.getKeys(keyList=["escape"]):

    # # CONTROLS - move triangle around in space
    # # if I press 'up', increase ring_speed by .1, if I press 'down', decrease ring_speed by .1
    # if event.getKeys(keyList=["up"]):
    #     y_list = [y + change_by for y in y_list]
    #     print(f"x: {x_list[0]}, y: {y_list[0]}, z: {z_list[0]}, change_by: {change_by}")
    # elif event.getKeys(keyList=["down"]):
    #     y_list = [y - change_by for y in y_list]
    #     print(f"x: {x_list[0]}, y: {y_list[0]}, z: {z_list[0]}, change_by: {change_by}")
    # elif event.getKeys(keyList=["left"]):
    #     x_list = [x - change_by for x in x_list]
    #     print(f"x: {x_list[0]}, y: {y_list[0]}, z: {z_list[0]}, change_by: {change_by}")
    # elif event.getKeys(keyList=["right"]):
    #     x_list = [x + change_by for x in x_list]
    #     print(f"x: {x_list[0]}, y: {y_list[0]}, z: {z_list[0]}, change_by: {change_by}")
    # elif event.getKeys(keyList=["i"]):
    #     z_list = [z + change_by for z in z_list]
    #     print(f"x: {x_list[0]}, y: {y_list[0]}, z: {z_list[0]}, change_by: {change_by}")
    # elif event.getKeys(keyList=["o"]):
    #     z_list = [z - change_by for z in z_list]
    #     print(f"x: {x_list[0]}, y: {y_list[0]}, z: {z_list[0]}, change_by: {change_by}")
    # elif event.getKeys(keyList=["x"]):
    #     change_by = change_by * 10
    #     print(f"x: {x_list[0]}, y: {y_list[0]}, z: {z_list[0]}, change_by: {change_by}")
    # elif event.getKeys(keyList=["z"]):
    #     change_by = change_by / 10
    #     print(f"x: {x_list[0]}, y: {y_list[0]}, z: {z_list[0]}, change_by: {change_by}")
    #
    #
    # # --- render loop ---
    # win.applyEyeTransform()
    #
    # # testing depth with this triangle to see when it disappears/appears.
    # GL.glColor3f(1.0, .5, .5)
    #
    # GL.glBegin(GL.GL_TRIANGLES)
    # # for i in range(len(x_list)):
    # for i in range(3):
    #     GL.glVertex3f(x_list[i], y_list[i], z_list[i])
    #     # GL.glVertex3f(*pos[i, :])  # position
    # GL.glEnd()



    # draw 3D stuff here
    GL.glColor3f(1.0, 1.0, 1.0)
    GL.glPointSize(5.0);

    GL.glBegin(GL.GL_POINTS)
    for i in range(nPoints):  # go over our array and draw the points using the coordinates
        # color can be varied with distance if you like
        GL.glVertex3f(*pos[i, :])  # position

    GL.glEnd()

    win.flip()

    # transform points -Z direction, in OpenGL -Z coordinates are forward
    # pos[:, 2] += 5.0  # distance to move the dots per frame towards the viewer
    pos[:, 2] += flow_speed_m_p_fr * flow_dir  # distance to move the dots per frame towards the viewer


    # reset dots that have gone past the far clipping plane
    # # todo: check this
    if expanding:
        pos[:, 2] = np.where(pos[:, 2] > -0.1, -1000.0, pos[:, 2])
        # pos[:, 2] = np.where(pos[:, 2] > -near_plane_m, -far_plane_m, pos[:, 2])  # negative values
        # if a point is too close, return to initial -Z position.  e.g., if trfm_near_plane_m is at -1, and a point is at -0.5, then point > trfm_near_plane_m == True.
        # pos[:, 2] = np.where(pos[:, 2] > trfm_near_plane_m, trfm_far_plane_m, pos[:, 2])  # negative values

    else:
        pos[:, 2] = np.where(pos[:, 2] < -1000, -1, pos[:, 2])  # negative values
        # if a point is too far, return to initial -Z position.  e.g., if trfm_far_plane_m is at -100, and a point is at -150, then point < trfm_far_plane_m == True.
        # pos[:, 2] = np.where(pos[:, 2] < trfm_far_plane_m, trfm_near_plane_m, pos[:, 2])  # negative values



    # # CONTROLS - move triangle around in space
    # # if I press 'up', increase ring_speed by .1, if I press 'down', decrease ring_speed by .1
    # if event.getKeys(keyList=["up"]):
    #     y_list = [y + change_by for y in y_list]
    #     print(f"x: {x_list[0]}, y: {y_list[0]}, z: {z_list[0]}, change_by: {change_by}")
    # elif event.getKeys(keyList=["down"]):
    #     y_list = [y - change_by for y in y_list]
    #     print(f"x: {x_list[0]}, y: {y_list[0]}, z: {z_list[0]}, change_by: {change_by}")
    # elif event.getKeys(keyList=["left"]):
    #     x_list = [x - change_by for x in x_list]
    #     print(f"x: {x_list[0]}, y: {y_list[0]}, z: {z_list[0]}, change_by: {change_by}")
    # elif event.getKeys(keyList=["right"]):
    #     x_list = [x + change_by for x in x_list]
    #     print(f"x: {x_list[0]}, y: {y_list[0]}, z: {z_list[0]}, change_by: {change_by}")
    # elif event.getKeys(keyList=["i"]):
    #     z_list = [z - change_by for z in z_list]
    #     print(f"x: {x_list[0]}, y: {y_list[0]}, z: {z_list[0]}, change_by: {change_by}")
    # elif event.getKeys(keyList=["o"]):
    #     z_list = [z + change_by for z in z_list]
    #     print(f"x: {x_list[0]}, y: {y_list[0]}, z: {z_list[0]}, change_by: {change_by}")
    # elif event.getKeys(keyList=["x"]):
    #     change_by = change_by * 10
    #     print(f"x: {x_list[0]}, y: {y_list[0]}, z: {z_list[0]}, change_by: {change_by}")
    # elif event.getKeys(keyList=["z"]):
    #     change_by = change_by / 10
    #     print(f"x: {x_list[0]}, y: {y_list[0]}, z: {z_list[0]}, change_by: {change_by}")

    # # draw test triangle
    # tri_from_list(x_list, y_list, z_list)




    # check events to break out of the loop!

    # if len(event.getKeys()) > 0:
    #     break
    # event.clearEvents()

win.close()
