from __future__ import division
import numpy as np
from psychopy import visual, core, event, monitors


'''
This illusion is based on the paper by Murikami & Cavanagh (98)
Murakami, I., & Cavanagh, P. (1998). 
A jitter after-effect reveals motion-based stabilization of vision. Nature, 395(October).
http://illusion-forum.ilab.ntt.co.jp/visual-jitter/index.html
Stare at the central fixation cross for 15 seconds, while a jitter pattern is presented on the outer ring.
Then, then the jitter stops, the central circle should appear to move or jitter with eye movements.
'''

"""parameters"""
# if True, uses B&W, else uses colours from radial flow experiment.
use_high_contrast = False

# How many frames dots are active for, original jitter example has pixels flip every 13.3ms
# dot_life_fr = 4
fps = 60
dot_life_ms = 13.333333
dot_life_fr = int(round(dot_life_ms/(1000/fps), 0))
print(f"dot_life_fr: {dot_life_fr}, {dot_life_ms}ms at {fps} fps.")

# # choose between random motion and radial motion (both in and outward)
background = 'jitter_random'  # 'jitter_random', 'jitter_radial'

# # dot parameters
n_moving_dots = 1000  # 350
flow_speed = .01  # actually gives the variance in speeds

# adaptation time in seconds
adaptation_time = 5
illusion_time = 5


"""back end"""
# get dimensions of monitor to fit mask to
monitor_name = 'Nick_work_laptop'  # 'NickMac' 'asus_cal' 'Asus_VG24' 'HP_24uh' 'ASUS_2_13_240Hz' 'Iiyama_2_18' 'Nick_work_laptop'
thisMon = monitors.Monitor(monitor_name)
this_width = thisMon.getWidth()
mon_dict = {'mon_name': monitor_name,
            'width': thisMon.getWidth(),
            'size': thisMon.getSizePix(),
            'dist': thisMon.getDistance(),
            'notes': thisMon.getNotes()}
print(f"mon_dict: {mon_dict}")

# double check using full screen in lab
display_number = 1  # 0 indexed, 1 for external display, 0 for internal
if monitor_name in ['ASUS_2_13_240Hz', 'asus_cal', 'Nick_work_laptop', 'NickMac']:
    display_number = 0
use_full_screen = True
if display_number > 0:
    use_full_screen = False
widthPix = int(mon_dict['size'][0])
heightPix = int(mon_dict['size'][1])
monitorwidth = float(mon_dict['width'])  # monitor width in cm
viewdist = float(mon_dict['dist'])  # viewing distance in cm
mon = monitors.Monitor(monitor_name, width=monitorwidth, distance=viewdist)
mon.setSizePix((widthPix, heightPix))
mon.save()

# # colour scheme to use
if use_high_contrast:
    flow_dots_col = [0, 0, 0]
    bgColor255 = 255
else:
    # # set same colours as radial flow exp
    flow_dots_col = [76.5, 114.75, 76.5]
    bgColor255 = 114.75


# WINDOW SPEC
win = visual.Window(monitor=mon, size=(widthPix, heightPix),
                    colorSpace='rgb255',
                    color=bgColor255,  # bgcolor from Martin's flow script, not bgColor255
                    winType='pyglet',  # I've added pyglet to make it work on pycharm/mac
                    pos=[1, -1],  # pos gives position of top-left of screen
                    units='pix',
                    screen=display_number,
                    allowGUI=False,
                    fullscr=use_full_screen)


# full screen mask to blend off edges and fade to black
raisedCosTexture2 = visual.filters.makeMask(heightPix, shape='raisedCosine', fringeWidth=0.1, radius=[1.0, 1.0])
invRaisedCosTexture = -raisedCosTexture2  # inverts mask to blur edges instead of center
blankslab = np.ones((heightPix, int((widthPix-heightPix)/2)))  # create blank slabs to put to left and right of image
mmask = np.append(blankslab, invRaisedCosTexture, axis=1)  # append blank slab to left
mmask = np.append(mmask, blankslab, axis=1)  # and right
peripheral_mask = visual.GratingStim(win, mask=mmask, tex=None, contrast=1.0,
                                     size=(widthPix, heightPix), units='pix', color='black')

# outer dot parameters
dot_field_size = heightPix  # previously called taille: french for 'size'

# # set number of frames that dots persist for
dot_lives = np.random.random(n_moving_dots) * 10  # this will be the current life of each element

# # get start locations for dots
# flow_dots - remember its equivalent to (rand * dot_field_size) - (dot_field_size / 2)
x = np.random.rand(n_moving_dots) * dot_field_size - dot_field_size / 2
y = np.random.rand(n_moving_dots) * dot_field_size - dot_field_size / 2

# # Dots move randomly in smooth directions, then get new direction when reborn.
# # separate x and y motion is non-radial; same values for both gives radial motion.
# # random.normal, centered at 1 (no motion), with an sd given by flow_speed.
x_motion = np.random.normal(1, flow_speed, n_moving_dots)
y_motion = np.random.normal(1, flow_speed, n_moving_dots)
if background == 'jitter_radial':
    y_motion = x_motion

flow_dots = visual.ElementArrayStim(win, elementTex=None, elementMask='circle',
                                    units='pix', nElements=n_moving_dots, sizes=30,
                                    colorSpace='rgb255',
                                    colors=flow_dots_col)


# central mask behind probemasks
raisedCosTexture1 = visual.filters.makeMask(256, shape='raisedCosine',
                                            fringeWidth=0.01, radius=[1.0, 1.0])
static_mask_size = 400
static_mask = visual.GratingStim(win, mask=raisedCosTexture1, tex=None,
                                 size=(static_mask_size, static_mask_size), units='pix',
                                 colorSpace='rgb255', color=bgColor255)


# # get start locations for static dots
n_static_dots = int(n_moving_dots/5)
phi = np.random.uniform(0, 2*np.pi, n_static_dots)
r = np.random.uniform(0, static_mask_size/2, n_static_dots)
static_x = r * np.cos(phi)
static_y = r * np.sin(phi)
static_xy_pos = np.array([static_x, static_y]).transpose()
static_dots = visual.ElementArrayStim(win, elementTex=None, elementMask='circle',
                                      units='pix', nElements=n_static_dots, sizes=30,
                                      colorSpace='rgb255', xys=static_xy_pos,
                                      colors=flow_dots_col)

# fixation bull eye
fixation = visual.Circle(win, radius=3, units='pix', lineColor='white',
                         fillColor='black')

# set a countdown timer for 30 seconds
countDown = core.CountdownTimer()
countDown.add(adaptation_time)

# Keyboard
resp = event.BuilderKeyResponse()
# theseKeys = event.getKeys(keyList=['space'])
myMouse = event.Mouse(visible=False)  # MOUSE - hide cursor


# # display runs for 15 seconds
while countDown.getTime() > 0:

    # update dot's life each frame
    dot_lives = dot_lives + 1

    # newXYs = flow_dots.xys
    flow_dot_xs = flow_dots.xys[:, 0]
    flow_dot_ys = flow_dots.xys[:, 1]

    # find the dead elements and reset their life
    deadElements = (dot_lives > dot_life_fr)  # np vector, not standard python
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
    # flow_dots.draw() is repeatedly called for each part of exp (fixation, probe1, isi etc)
    flow_dots.draw()

    peripheral_mask.draw()
    static_mask.draw()
    static_dots.draw()
    fixation.draw()

    win.flip()

    if event.getKeys(['escape']):
        core.quit()

# reset response clock
resp.clock.reset()

# present static display for illusion_time seconds
countDown.add(illusion_time)
while countDown.getTime() > 0:
    resp.clock.reset()

    flow_dots.draw()

    peripheral_mask.draw()
    static_mask.draw()
    static_dots.draw()
    fixation.draw()

    win.flip()

    theseKeys = event.getKeys(keyList=['space'])
    if len(theseKeys) > 0:  # at least one key was pressed
        resp.keys = theseKeys[-1]  # just the last key pressed
        resp.rt = resp.clock.getTime()

    if event.getKeys(['escape']):
        core.quit()

# # end message
end_of_exp = visual.TextStim(win=win, name='end_of_exp',
                             text="You have completed this experiment.\n"
                                  f"The illusion persisted for {resp.rt}."
                                  "Thank you for your time.\n\n"
                                  "Press any key to return to the desktop.",
                             font='Arial', height=20)

print(f"response time: {resp.rt}")

while not event.getKeys():
    # display end of experiment screen
    end_of_exp.draw()
    win.flip()
else:
    # close and quit once a key is pressed
    win.close()
    core.quit()

# win.close()
# core.quit()
