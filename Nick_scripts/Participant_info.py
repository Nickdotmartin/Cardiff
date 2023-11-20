from psychopy import gui, visual, core, data, event, monitors, logging
from psychopy import __version__ as psychopy_version
from psychopy.tools.monitorunittools import cm2pix, pix2cm, pix2deg
from datetime import datetime
from os import path, chdir

'''This script will use PsychoPy to collect participant ID, age, sex and gender information. from the dlg and saves it to a csv file.'''

#######################
# # # MAIN SCRIPT # # #
#######################

# get filename and path for this experiment
_thisDir = path.dirname(path.abspath(__file__))
chdir(_thisDir)
expName = path.basename(__file__)[:-3]


# # # DIALOGUE BOX # # #

# dialogue box/drop-down option when exp starts (1st item is default val)
expInfo = {'01. Participant ID': '',
           '02. age': '',
           '03. What was your registered sex at birth?': ['Female', 'Male', 'Other', 'Prefer not to say'],
           }

# run drop-down menu, OK continues, cancel quits
dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
if not dlg.OK:
    core.quit()  # user pressed cancel

# add date and time to the filename
participant_ID = expInfo['01. Participant ID']
age = expInfo['02. age']
reg_sex = expInfo['03. What was your registered sex at birth?']


expInfo['date'] = datetime.now().strftime("%d/%m/%Y")
expInfo['time'] = datetime.now().strftime("%H:%M:%S")

monitor_name = 'OLED'

# save each participant's files into separate dir for each ISI
save_dir = path.join(_thisDir, expName, monitor_name,  # added monitor name to analysis structure
                     participant_ID)

save_output_as = path.join(save_dir, 'participant_info')


# Experiment Handler
thisExp = data.ExperimentHandler(name=expName, version=psychopy_version,
                                 runtimeInfo=None,
                                 savePickle=None, saveWideText=True,
                                 dataFileName=save_output_as)

thisExp.addData('participant_ID', participant_ID)
thisExp.addData('age', age)
thisExp.addData('reg_sex', reg_sex)
thisExp.addData('date', expInfo['date'])
thisExp.addData('time', expInfo['time'])


thisExp.close()
logging.flush()  # write messages out to all targets
thisExp.abort()  # or data files will save again on exit

print(f"\n\nThankyou, please continue with the other experiments.\n\n")

# close and quit once a key is pressed
core.quit()
