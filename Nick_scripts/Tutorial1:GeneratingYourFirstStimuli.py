from psychopy import visual, core, event

"""
from https://www.psychopy.org/coder/tutorial1.html
"""

# create a window
win = visual.Window([800, 600], monitor="NickMac", units="deg")

# create some stimuli
grating = visual.GratingStim(win=win, mask='circle', size=3, pos=[-4, 0], sf=3)
fixation = visual.GratingStim(win=win, size=0.2, pos=[0, 0], sf=0, rgb=-1)



# draw the stimuli and update the window
while True:  # this creates a never-ending loop
    grating.setPhase(0.05, '+')  # advance phase by 0.05 of a cycle
    grating.draw()
    fixation.draw()

    win.flip()

    if len(event.getKeys()) > 0:
        break
    event.clearEvents()

# cleanup
win.close()
core.quit()


