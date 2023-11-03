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

this_colourSpace = 'rgb1'  # 'rgb' 'rgb1'

if this_colourSpace == 'rgb1':
    # get 1024 (10-bit) values between 0 and 1.
    list_10_bit = [i/1023 for i in list(range(1024))]


    # get 256 (8-bit) values between 0 and 1, each repeated*4 to give 1024 items
    list_8_bit = [i/255 for i in list(range(256))]
    list_8_bit = np.repeat(list_8_bit, 4)

    bg_colour = [0.5, 0.5, 0.5]

elif this_colourSpace == 'rgb':
    # get 1024 (10-bit) values between -1 and 1.
    list_10_bit = [-1 + 2 * (i / 1023) for i in list(range(1024))]

    # get 256 (8-bit) values between -1 and 1, each repeated*4 to give 1024 items
    list_8_bit = [-1 + 2 * ((i // 4) * 4) / 1023 for i in list(range(1024))]

    bg_colour = [0, 0, 0]

print(f"list_10_bit: {len(list_10_bit)}\n{list_10_bit[:5]} to {list_10_bit[-5:]}")
print(f"list_8_bit: {len(list_8_bit)}\n{list_8_bit[:5]} to {list_8_bit[-5:]}")
# # print lists of equivalent values from 10-bit and 8-bit lists
# for idx, (a_10, b_8) in enumerate(zip(list_10_bit, list_8_bit)):
#     print(f"{idx}: 10-bit: {a_10}, 8-bit: {b_8}")


# # If you want to display on external monitor, set this variable to True
external_monitor=True
use_screen = 0
if external_monitor:
    use_screen = 1

# initialize pyglet window
win = psychopy.visual.Window(
                             units="pix", fullscr=True,
                             colorSpace=this_colourSpace,
#                             color=bg_colour,
                             color=[0, 0, 0],
                             bpc=[10, 10, 10],
                             screen=use_screen)
print(f"win.colorSpace: {win.colorSpace}")



# define line stimuli
line = psychopy.visual.Line(win=win, units="pix", colorSpace=this_colourSpace)

# horizontal positions for lines [-512 to 512]
x_pos_list = [i-512 for i in list(range(1024))]
# print(f"x_pos_list: {x_pos_list}")
#
# # draw a two squares next to each other, each size 525 x 525
# have less than 4 values between them,
dark_sq_col = list_10_bit[1]
light_sq_col = list_10_bit[2]
dark_sq = psychopy.visual.Rect(win=win, units="pix", width=525, height=525, pos=[-525/2, 300],
                                fillColor=list_10_bit[0], colorSpace=this_colourSpace)
light_sq = psychopy.visual.Rect(win=win, units="pix", width=525, height=525, pos=[525/2, 300],
                                fillColor=list_10_bit[1], colorSpace=this_colourSpace)
print(f"black_sq col: {dark_sq_col}, white_sq_col: {light_sq_col}")


# present stimuli until escape key is pressed
while not psychopy.event.getKeys(keyList=["escape"]):

    dark_sq.draw()
    light_sq.draw()


   # draw 10-bit lines at top of screen
   for idx, x_pos in enumerate(x_pos_list):

       line.start = [x_pos, 500]
       line.end = [x_pos, 0]

       line.lineColor = [list_10_bit[idx], list_10_bit[idx], list_10_bit[idx]]

       line.draw()

   # draw 8-bit lines at bottom of screen
   for idx, x_pos in enumerate(x_pos_list):

       line.start = [x_pos, 0]
       line.end = [x_pos, -500]

       line.lineColor = [list_8_bit[idx], list_8_bit[idx], list_8_bit[idx]]

       line.draw()


    win.flip()


win.close()
