import os
import random

from psychopy import visual, event, core

"""
original moving stim code from
https://discourse.psychopy.org/t/how-to-move-a-stimulus-along-a-vector-and-not-only-left-or-right/18059
"""

# # equipment/set-up details
# mon = monitors.Monitor('HP 24uh')  # fetch the most recent calib for this monitor
# mon.setDistance(75)  # further away than normal?
# win = visual.Window(size=[1920, 1080], monitor=mon)
win = visual.Window(units='pix', color='black') # Set the window

# # exp details
n_trials = 1
stim_dur = 100 # in ms
n_stil = 1
# stim_size =

# # fixation cross - remain on screen of dissapear?
# fixation = visual.




# initial probe position
x = -300
# x = random.randint(-100, 100)
y = 0

# create just once, no need to specify a position yet:
circle = visual.Circle(win, radius = 10, fillColor= 'yellow')

# # make probe only appear for 100ms

# # set speed of probe.

while True: # draw moving stimulus
    print(f"x: {x}")
    if x > 0:
        x -= 1 # make circle constantly move left(here i want to use a Vector)
    elif x < 0:
        x += 1 # make circle constantly move left(here i want to use a Vector)

    circle.pos = [x, y] # directly update both x *and* y
    circle.draw()
    win.flip() # make the drawn things visible

win.close()
core.quit()