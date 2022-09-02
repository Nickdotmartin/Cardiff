from __future__ import division

import copy
import os
from datetime import datetime
from math import *

import numpy as np
from psychopy import __version__ as psychopy_version
from psychopy import gui, visual, core, data, event, monitors, colors

from PsychoPy_tools import check_correct_monitor, get_pixel_mm_deg_values
from kestenSTmaxVal import Staircase

from psychopy import visual, core, event
from numpy.random import random, shuffle



win = visual.Window(fullscr=True, units='pix', monitor='testMonitor')

nDots = 100
flow_dots_col = [76.5, 114.75, 76.5]


taille = 800  # french for 'size', 'cut', 'trim', 'clip' etc - what does it actually do here?
minDist = 0.5
maxDist = 5

lives = random(nDots) * 10  # this will be the current life of each element
max_life = 8

# flow_dots - remember its equivalent to (rand * taille) - (taille / 2)
x = np.random.rand(nDots) * taille - taille / 2
y = np.random.rand(nDots) * taille - taille / 2

# # original Simon version - harsh radial flow
# z = np.random.rand(nDots) * (maxDist + minDist) / 2
# # more subtle mix of in and out flow
# z = np.random.normal(1, .01, nDots)
# # no flow at all  (values > 1 flow out and < 1 flow in)
z = np.ones(nDots)

flow_dots = visual.ElementArrayStim(win, elementTex=None, elementMask='circle',
                                    units='pix', nElements=nDots, sizes=30,
                                    colorSpace='rgb255',
                                    colors=flow_dots_col)


while not event.getKeys():

    # newXYs = flow_dots.xys
    flow_dot_xs = flow_dots.xys[:, 0]
    flow_dot_ys = flow_dots.xys[:, 1]


    # find the dead elemnts and reset their life
    deadElements = (lives > max_life)  # numpy vector, not standard python
    lives[deadElements] = 0

    # for the dead elements update the xy and ori
    # random array same shape as dead elements
    # newXYs[deadElements, :] = random(newXYs[deadElements, :].shape) * taille - taille/2.0
    flow_dot_xs[deadElements] = random(flow_dot_xs[deadElements].shape) * taille - taille/2.0
    flow_dot_ys[deadElements] = random(flow_dot_ys[deadElements].shape) * taille - taille/2.0

    # # If I'm not using z, I can just access or create xys like this.
    # xys = random([nDots, 2]) * taille - taille / 2.0  # numpy vector
    # flow_dots.xys = np.array([x_flow, y_flow]).transpose()
    # # If I am using z, I create the xys like this
    x_flow = flow_dot_xs / z
    y_flow = flow_dot_ys / z
    xy_pos = np.array([x_flow, y_flow]).transpose()
    flow_dots.xys = xy_pos
    flow_dots.draw()

    lives = lives + 1


    win.flip()

win.close()
core.quit()