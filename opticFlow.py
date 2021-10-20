from __future__ import division  # so that 1/3=0.333 instead of 1/3=0
from psychopy import visual, core, data, event, logging, sound, gui, monitors
from psychopy.constants import *  # things like STARTED, FINISHED
import numpy  # whole numpy lib is available, prepend 'np.'
from numpy import sin, cos, tan, log, log10, pi, average, sqrt, std, deg2rad, rad2deg, linspace, asarray
from numpy.random import random, randint, normal, shuffle
import os  # handy system and path functions
from random import choice, randrange, shuffle, uniform
from psychopy.tools.coordinatetools import pol2cart, cart2pol
import time
from psychopy.tools.filetools import fromFile, toFile


"""
original code adapted from 
https://discourse.psychopy.org/t/speed-difference-in-different-types-of-optic-flow-stimuli/1878
"""


# win = visual.Window([1680, 1050], units='deg',
#                     monitor='Umram', color='black', fullscr=False)

mon = monitors.Monitor('HP 24uh')  # fetch the most recent calib for this monitor
mon.setDistance(75)  # further away than normal?
win = visual.Window(size=[1920, 1080], monitor=mon)


# experiment details
refRate = 60  # 1 second
nTrials = 15
stimDur = refRate  # stimulus duration = 2 seconds

# important parameters
dotsN = 40
fieldSize = 3  # 3x3 square dot field
elemSize = 0.25

speed = 7 / 60  # 7 degree/seconds

dotsX = numpy.random.uniform(low=-fieldSize, high=fieldSize,
                             size=(dotsN,))  # array of random float numbers between fieldSize range
dotsY = numpy.random.uniform(low=-fieldSize, high=fieldSize, size=(dotsN,))

dotsTheta = numpy.random.rand(dotsN) * 360  # array with shape (500,)
dotsRadius = numpy.random.rand(dotsN) * fieldSize

randDotsX = numpy.random.uniform(low=-fieldSize, high=fieldSize, size=(dotsN,))
randDotsY = numpy.random.uniform(low=-fieldSize, high=fieldSize, size=(dotsN,))

# initializing experiment stimuli
transDots = visual.ElementArrayStim(win, nElements=dotsN, sizes=elemSize, elementTex=None,
                                    colors=(1.0, 1.0, 1.0), xys=random([dotsN, 2]) * fieldSize,
                                    colorSpace='rgb', elementMask='circle', fieldSize=fieldSize)

rotDots = visual.ElementArrayStim(win, nElements=dotsN, sizes=elemSize, elementTex=None,
                                  colors=(1.0, 1.0, 1.0), xys=random([dotsN, 2]),
                                  colorSpace='rgb', elementMask='circle', texRes=128, fieldSize=fieldSize, )

fixation = visual.GratingStim(win, size=0.2, pos=[0, 0], sf=0, color='red')
# fixation = visual.g

condList = (['rotClock', 'rotCounterClock'], ['radialIn', 'radialOut'],
            ['transUp', 'transDown'], ['transRight', 'transLeft'])

for trials in range(nTrials):  # number of blocks
    cond = choice(condList)
    print(f"{cond}")
    # 2

    for condType in cond:
        # assign parameters according to the condition
        if condType == 'transUp' or condType == 'transRight' or condType == 'rotCounterClock' or condType == 'radialOut':
            moveSign = 1  # positive for up and right movement
        elif condType == 'transDown' or condType == 'transLeft' or condType == 'rotClock' or condType == 'radialIn':
            moveSign = -1  # negative for down and left movement

        # Frame Loop
        for frameN in range(stimDur):
            dieScoreArray = numpy.random.rand(dotsN)  # generating array of float numbers
            deathDots = (dieScoreArray < 0.01)  # assigning True(death warrant) for numbers below 0.01
            # handling each motion type individually
            if condType == 'transUp':
                dotsY = (dotsY + speed * moveSign)
                transMove = True
                transDots.setXYs(numpy.array([dotsX, dotsY]).transpose())
                outfieldDots = (dotsY >= fieldSize)
                dotsY[outfieldDots] = numpy.random.rand(
                    sum(outfieldDots)) * fieldSize - fieldSize  # out of field dots are replotted
                dotsY[deathDots] = numpy.random.rand(sum(deathDots)) * (
                            fieldSize + fieldSize) - fieldSize  # death dots are replotted
            elif condType == 'transDown':
                dotsY = (dotsY + speed * moveSign)
                transMove = True
                transDots.setXYs(numpy.array([dotsX, dotsY]).transpose())
                outfieldDots = (dotsY <= -fieldSize)
                dotsY[outfieldDots] = numpy.random.rand(sum(outfieldDots)) * fieldSize
                dotsY[deathDots] = numpy.random.rand(sum(deathDots)) * (fieldSize + fieldSize) - fieldSize
            elif condType == 'transRight':
                dotsX = (dotsX + speed * moveSign)
                transMove = True
                transDots.setXYs(numpy.array([dotsX, dotsY]).transpose())
                outfieldDots = (dotsX >= fieldSize)
                dotsX[outfieldDots] = numpy.random.rand(sum(outfieldDots)) * fieldSize - fieldSize
                dotsX[deathDots] = numpy.random.rand(sum(deathDots)) * (fieldSize + fieldSize) - fieldSize
            elif condType == 'transLeft':
                dotsX = (dotsX + speed * moveSign)
                transMove = True
                transDots.setXYs(numpy.array([dotsX, dotsY]).transpose())
                outfieldDots = (dotsX <= -fieldSize)
                dotsX[outfieldDots] = numpy.random.rand(sum(outfieldDots)) * fieldSize
                dotsX[deathDots] = numpy.random.rand(sum(deathDots)) * (fieldSize + fieldSize) - fieldSize
            elif condType == 'radialIn':
                dotsRadius = (dotsRadius + speed * moveSign)
                transMove = False
                outFieldRadius = (dotsRadius <= 0.03)
                dotsRadius[outFieldRadius] = numpy.random.rand(sum(outFieldRadius)) * fieldSize
                dotsRadius[deathDots] = numpy.random.rand(sum(deathDots)) * fieldSize
                thetaX, radiusY = pol2cart(dotsTheta, dotsRadius)
                rotDots.setXYs(numpy.array([thetaX, radiusY]).transpose())
            elif condType == 'radialOut':
                dotsRadius = (dotsRadius + speed * moveSign)
                transMove = False
                outFieldRadius = (dotsRadius >= fieldSize)
                dotsRadius[outFieldRadius] = numpy.random.rand(sum(outFieldRadius))
                dotsRadius[deathDots] = numpy.random.rand(sum(deathDots))
                thetaX, radiusY = pol2cart(dotsTheta, dotsRadius)
                rotDots.setXYs(numpy.array([thetaX, radiusY]).transpose())
            elif condType == 'rotClock' or condType == 'rotCounterClock':
                dotsTheta += speed * moveSign  # clockwise or counterclocwise
                transMove = False
                dotsTheta[deathDots] = numpy.random.rand(sum(deathDots)) * 360
                thetaX, radiusY = pol2cart(dotsTheta, dotsRadius)
                rotDots.setXYs(numpy.array([thetaX, radiusY]).transpose())

            # drawing stimuli
            if transMove == True:
                transDots.draw()
            elif transMove == False:
                rotDots.draw()
            fixation.draw()
            win.flip()

win.close()
core.quit()

