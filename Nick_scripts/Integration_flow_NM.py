from __future__ import division
from psychopy import gui, visual, core, data, event, logging, monitors
from psychopy import __version__ as psychopy_version
import os
from numpy import deg2rad
from numpy.random import shuffle
import random
import copy
from datetime import datetime
from math import *

from kestenSTmaxVal import Staircase



"""
This script takes: 
the probes from EXPERIMENT1-white-probes-breaks, and adds the option for tangent or radial jump.
the background radial motion from integration_RiccoBloch_flow_new.
ISI is always >=0 (no simultaneous probes).
"""

# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# Monitor config from monitor centre
monitor_name = 'HP_24uh'  # 'NickMac' 'asus_cal' 'Asus_VG24' 'HP_24uh'
display_number = 1  # 0 indexed, 1 for external display


# Store info about the experiment session
expName = 'integration_flow'  # from the Builder filename that created this script

expInfo = {'Participant': 'testnm',
           'Probe_dur_in_frames_at_240hz': 2,
           'fps': [60, 144, 240],
           'ISI_dur_in_ms': [0, 8.33, 16.67, 25, 37.5, 50, 100],
           'Probe_orientation': ['tangent', 'ray'],
           'Probe_size': ['5pixels', '6pixels', '3pixels']
           }



# GUI
dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
if not dlg.OK:
    core.quit()  # user pressed cancel

expInfo['date'] = datetime.now().strftime("%d/%m/%Y")
expInfo['time'] = datetime.now().strftime("%H:%M:%S")

# GUI SETTINGS
participant_name = expInfo['Participant']
# todo: change trial_NUMBER TO total_n_trials or something
trial_number = 25
probe_duration = int(expInfo['Probe_dur_in_frames_at_240hz'])
probe_ecc = 4
fps = int(expInfo['fps'])
orientation = expInfo['Probe_orientation']
Probe_size = expInfo['Probe_size']

'''ISI can be given (roughly) in ms, for any monitor it will try to match that value in frames.
The script uses on frames so ISI will be used in that format.
The actual ms timing is given for record.
This means that the experiment should have similar ms timings on monitors with different fps'''
ISI_selected_ms = expInfo['ISI_dur_in_ms']
ISI_frames = int(ISI_selected_ms * fps / 1000)
ISI_actual_ms = (1/fps) * ISI_frames * 1000
ISI = ISI_frames

# VARIABLES
# Distances between probes
# 99 values for single probe condition
separations = [18, 18, 6, 6, 3, 3, 2, 2, 1, 1, 0, 0, 99, 99]



# FILENAME
filename = f'{_thisDir}{os.sep}' \
           f'{participant_name}{os.sep}' \
           f'ISI_{ISI}_probeDur{probe_duration}{os.sep}' \
           f'{participant_name}'

# Experiment Handler
thisExp = data.ExperimentHandler(name=expName, version=psychopy_version,
                                 extraInfo=expInfo, runtimeInfo=None,
                                 savePickle=None, saveWideText=True,
                                 dataFileName=filename)

# COLORS AND LUMINANCE
# Lum to Color255
LumColor255Factor = 2.39538706913372
# Color255 to Color1
Color255Color1Factor = 1/127.5  # Color255 * Color255Color1Factor -1
# Lum to Color1
Color1LumFactor = 2.39538706913372

maxLum = 106  # 255 RGB
minLum = 0.12  # 0 RGB
maxColor255 = 255
minColor255 = 0
maxColor1 = 1
minColor1 = -1
bgLumP = 20  # int(expInfo['7. Background lum in percent of maxLum'])
bgLum = maxLum * bgLumP / 100
bgColor255 = bgLum * LumColor255Factor


# MONITOR SPEC
thisMon = monitors.Monitor(monitor_name)
this_width = thisMon.getWidth()
mon_dict = {'mon_name': monitor_name,
            'width': thisMon.getWidth(),
            'size': thisMon.getSizePix(),
            'dist': thisMon.getDistance(),
            'notes': thisMon.getNotes()
            }
print(f"mon_dict: {mon_dict}")

widthPix = mon_dict['size'][0]  # 1440  # 1280
heightPix = mon_dict['size'][1]  # 900  # 800
monitorwidth = mon_dict['width']  # 30.41  # 32.512  # monitor width in cm
viewdist = mon_dict['dist']  # 57.3  # viewing distance in cm
viewdistPix = widthPix/monitorwidth*viewdist
mon = monitors.Monitor(monitor_name, width=monitorwidth, distance=viewdist)
mon.setSizePix((widthPix, heightPix))
mon.save()


# WINDOW SPEC
win = visual.Window(monitor=mon, size=(widthPix, heightPix),
                    colorSpace='rgb255', color=bgColor255,
                    winType='pyglet',  # I've added this to make it work on pycharm/mac
                    pos=[1, -1],  # pos gives position of top-left of screen
                    units='pix',
                    screen=display_number,
                    allowGUI=False,
                    fullscr=None
                    )

# check correct monitor details (fps, size) have been accessed.
print(win.monitor.name, win.monitor.getSizePix())
actualFrameRate = int(win.getActualFrameRate())
if fps in list(range(actualFrameRate-2, actualFrameRate+2)):
    print("fps matches actual frame rate")
else:
    # if values don't match, quit experiment
    print(f"fps ({fps}) does not match actual frame rate ({actualFrameRate})")
    core.quit()


actual_size = win.size
if list(mon_dict['size']) == list(actual_size):
    print(f"monitor is expected size")
elif list(mon_dict['size']) == list(actual_size/2):
    print(f"actual size is double expected size - Its ok, just a mac retina display bug.")
else:
    print(f"Display size does not match expected size from montior centre")
    # check sizes seems unreliable,
    # it returns different values for same screen if different mon_names are used!
    check_sizes = win._checkMatchingSizes(mon_dict['size'], actual_size)
    print(check_sizes)
    core.quit()




