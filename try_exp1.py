# import os
import random

from psychopy import visual, event, core

"""
original moving stim code from
https://discourse.psychopy.org/t/how-to-move-a-stimulus-along-a-vector-and-not-only-left-or-right/18059

1. Show fixation
2. moving targets appear for 100ms
    - single or multiple items
    - moving towards or away from fixation
        - direct in or out, or any 360 degree direction, with binary label? 
3. targets disappear
    - if multiple targets probe highlights end position of target of interest
4. participant reports perception of movement - `inward' or `outwards'

"""

# # equipment/set-up details
# mon = monitors.Monitor('HP 24uh')  # fetch the most recent calib for this monitor
# mon.setDistance(75)  # further away than normal?
# win = visual.Window(size=[1920, 1080], monitor=mon)
win = visual.Window(units='pix', color='black')  # Set the window

# # exp details
n_trials = 1
stim_dur = 100  # in ms
n_stil = 1
# stim_size =

# # fixation cross - remain on screen of dissapear?
# fixationClock = core.Clock()
fixPoint = visual.TextStim(win=win, name='fixPoint',
                           text='+',
                           font='Courier',
                           pos=(0, 0), height=50, wrapWidth=None, ori=0,
                           color='white', colorSpace='rgb', opacity=1,
                           # languageStyle='LTR',
                           # depth=0.0
                           )




# initial probe position
# x = -300
# y = 0
x = random.randint(-300, 300)
y = random.randint(-300, 300)

# create just once, no need to specify a position yet:
circle = visual.Circle(win, radius=10, fillColor='yellow')

# # make probe only appear for 100ms

# # set speed of probe.
speed_var = random.randint(6, 500)
print(f"Speed: {speed_var}")
x_change = abs(x)/speed_var
y_change = abs(y)/speed_var

slow_it_down = 100

for timestep in range(slow_it_down):
    # circle.pos = [x, y] # directly update both x *and* y
    fixPoint.draw()
    win.flip()  # make the drawn things visible

# # 6 timesteps at 60hz corresponds to 100ms
for timestep in range(slow_it_down):
# while True:  # runs infinately if true
    print(f"ts{timestep}: ({x}, {y})")
    if x > 0:
        x -= x_change  # make circle constantly move left
    elif x < 0:
        x += x_change  # make circle constantly move right

    if y > 0:
        y -= y_change  # make circle constantly move down
    elif y < 0:
        y += y_change  # make circle constantly move up

    circle.pos = [x, y]  # directly update both x *and* y
    circle.draw()
    fixPoint.draw()
    win.flip()  # make the drawn things visible

win.close()
core.quit()

