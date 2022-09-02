from __future__ import division
import numpy as np
from psychopy import __version__ as psychopy_version
from psychopy import visual, core, event


# # make window to display stim
win = visual.Window(fullscr=True, units='pix', monitor='testMonitor')

# # set hypothetical frame rate to test max_lives variable
fps = 240

# # choose between random motion and radial motion (both in and outward)
background = 'jitter_random'  # 'jitter_random', 'jitter_radial'

# # set same colours as radial flow exp
flow_dots_col = [76.5, 114.75, 76.5]

# # dot parameters
nDots = 100
dot_field_size = 800  # previously called taille: french for 'size'
flow_speed = .01

# # set number of frames that dots persist for
dot_lives = np.random.random(nDots) * 10  # this will be the current life of each element
max_lives = int(fps/30)  # original jitter example has pixels flipping at ~30fps
print(f"max_lives: {max_lives}")

# # get start locations for dots
# flow_dots - remember its equivalent to (rand * dot_field_size) - (dot_field_size / 2)
x = np.random.rand(nDots) * dot_field_size - dot_field_size / 2
y = np.random.rand(nDots) * dot_field_size - dot_field_size / 2

# # Dots move randomly in smooth directions, then get new direction when reborn.
# # separate x and y motion is non-radial; same values for both gives radial motion.
# # random.normal, centered at 1 (no motion), with an sd given by flow_speed.
x_motion = np.random.normal(1, flow_speed, nDots)
y_motion = np.random.normal(1, flow_speed, nDots)
if background == 'jitter_radial':
    y_motion = x_motion

flow_dots = visual.ElementArrayStim(win, elementTex=None, elementMask='circle',
                                    units='pix', nElements=nDots, sizes=30,
                                    colorSpace='rgb255',
                                    colors=flow_dots_col)

# # display runs until a key is pressed
while not event.getKeys():

    # update dot's life each frame
    dot_lives = dot_lives + 1

    # newXYs = flow_dots.xys
    flow_dot_xs = flow_dots.xys[:, 0]
    flow_dot_ys = flow_dots.xys[:, 1]

    # find the dead elemnts and reset their life
    deadElements = (dot_lives > max_lives)  # numpy vector, not standard python
    dot_lives[deadElements] = 0

    # New x, y locations for dots that are re-born (random array same shape as dead elements)
    # newXYs[deadElements, :] = np.random.random(newXYs[deadElements, :].shape) * dot_field_size - dot_field_size/2.0
    flow_dot_xs[deadElements] = np.random.random(flow_dot_xs[deadElements].shape) * dot_field_size - dot_field_size/2.0
    flow_dot_ys[deadElements] = np.random.random(flow_dot_ys[deadElements].shape) * dot_field_size - dot_field_size/2.0

    # each frame update dot positions by dividing x_location by x_motion
    x_flow = flow_dot_xs / x_motion
    y_flow = flow_dot_ys / y_motion
    xy_pos = np.array([x_flow, y_flow]).transpose()
    flow_dots.xys = xy_pos

    # # the above is called just once in the experiment loop.
    # flow_dots.draw() is all that needs to be called for each part of exp (fixation, probe1, isi etc)
    flow_dots.draw()

    win.flip()

win.close()
core.quit()
