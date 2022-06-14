import numpy as np
import psychopy.event
import psychopy.visual



'''
[09/06/2022 08:05] Simon Rushton
If you have a few mins you could write a quick psychopy program for me,
using pyglet, that draws 1024 stripes next to each other.
For a first band of stripes the luminance should be -1+2*(i/1023) where i is  0->1023 and
next to that we want a stripes that are -1+2*((i%4)/1023).
That should give us two grey-scale gradients to compare, one that should be 10bit and
one that should be 8bit.  That provides a very easy way of checking we are really getting 10bit -
the 10bit stripe should be a lot smoother than the 8 bit stripe.

adapted from script for Ponzo illusion at 
https://www.djmannion.net/psych_programming/vision/draw_shapes/draw_shapes.html

'''


# get 1024 (10-bit) values between -1 and 1.
list_10_bit = [-1+2*(i/1023) for i in list(range(1024))]

# get 256 (8-bit) values between -1 and 1, each repeated*4 to give 1024 items
# list_8_bit = [-1+2*(i/256) for i in list(range(256))]
# list_8_bit = np.repeat(list_8_bit, 4)
list_8_bit = [-1+2*((i//4)*4)/1023 for i in list(range(1024))]


# print lists of equivalent values from 10-bit and 8-bit lists
# for idx, (a_10, b_8) in enumerate(zip(list_10_bit, list_8_bit)):
#     print(f"{idx}: 10-bit: {a_10}, 8-bit: {b_8}")


# initialize pyglet window
win = psychopy.visual.Window(size=[1024, 1024], units="pix", fullscr=False,
                             color=[1, 1, 1], winType='pyglet', screen=1)

# define line stimuli
line = psychopy.visual.Line(win=win, units="pix")

# horizontal positions for lines [-512 to 512]
x_pos = [i-512 for i in list(range(1024))]

# draw 10-bit lines at top of screen
for idx, bar_offset in enumerate(x_pos):

    line.start = [bar_offset, 500]
    line.end = [bar_offset, 0]

    line.lineColor = [list_10_bit[idx], list_10_bit[idx], list_10_bit[idx]]

    line.draw()

# draw 8-bit lines at bottom of screen
for idx, bar_offset in enumerate(x_pos):

    line.start = [bar_offset, 0]
    line.end = [bar_offset, -500]

    line.lineColor = [list_8_bit[idx], list_8_bit[idx], list_8_bit[idx]]

    line.draw()


win.flip()

win.getMovieFrame()
psychopy.event.waitKeys()

win.close()
