from psychopy import visual, core  # import some libraries from PsychoPy
from psychopy import visual, event, core
from psychopy.tools.monitorunittools import posToPix
from PsychoPy_tools import get_pixel_mm_deg_values

#create a window
win = visual.Window(monitor="HP_24uh", units="pix",
                    colorSpace='rgb',
                    color=[-0.1, -0.1, -0.1],  # bgcolor from Martin's flow script, not bgColor255
                    winType='pyglet',  # I've added pyglet to make it work on pycharm/mac
                    screen=1,
                    )

# # probe 2 uses the vertices given in Martin's code
# probe2 = [(0, 0), (1, 0), (1, 1), (2, 1), (2, -1), (1, -1),
#           (1, -2), (-1, -2), (-1, -1), (0, -1)]
#
#
# '''these stim are aligned to appear as if probe1 is always in same place'''
# oneP = [(-2, -1), (-2, 1), (-1, 1), (-1, 2), (1, 2),
#         (1, 1), (0, 1), (0, 0), (-1, 0), (-1, -1)]
#
# sep0 = [(-2, -1), (-2, 1), (-1, 1), (-1, 2), (1, 2),
#         (1, -1), (0, -1), (0, -2), (-2, -2)]
# sep1 = [(-2, -1), (-2, 1), (-1, 1), (-1, 2), (1, 2),
#         (1, 0), (2, 0), (2, -2), (1, -2), (1, -3), (-1, -3), (-1, -2),
#         (-2, -2)]
# sep2 = [(-2, -1), (-2, 1), (-1, 1), (-1, 2), (1, 2),
#         (1, 0), (2, 0),
#         (2, -1), (3, -1),
#         (3, -3), (2, -3), (2, -4), (0, -4), (0, -3),
#         (-1, -3), (-1, -2), (-2, -2)]
# sep3 = [(-2, -1), (-2, 1), (-1, 1), (-1, 2), (1, 2),
#         (1, 0), (2, 0),
#         (2, -1), (3, -1),
#
#         (3, -2), (4, -2), (4, -4),
#         (3, -4), (3, -5), (1, -5), (1, -4),
#
#         (0, -4), (0, -3),
#         (-1, -3), (-1, -2), (-2, -2)]
#
# sep6 = [(-2, -1), (-2, 1), (-1, 1), (-1, 2), (1, 2),
#         (1, 0), (2, 0),
#         (2, -1), (3, -1),
#         (3, -2), (4, -2),
#
#         (4, -3), (5, -3), (5, -4), (6, -4), (6, -5), (7, -5),
#         (7, -7), (6, -7), (6, -8), (4, -8),
#         (4, -7), (3, -7), (3, -6), (2, -6), (2, -5),
#
#         (1, -5), (1, -4),
#         (0, -4), (0, -3),
#         (-1, -3), (-1, -2), (-2, -2)]
#
# sep18 = [(-2, -1), (-2, 1), (-1, 1), (-1, 2), (1, 2),
#          (1, 0), (2, 0),
#          (2, -1), (3, -1),
#          (3, -2), (4, -2),
#          (4, -3), (5, -3),
#          (5, -4), (6, -4),
#          (6, -5), (7, -5),
#          (7, -6), (8, -6),
#          (8, -7), (9, -7),
#          (9, -8), (10, -8),
#          (10, -9), (11, -9),
#          (11, -10), (12, -10),
#          (12, -11), (13, -11),
#          (13, -12), (14, -12),
#          (14, -13), (15, -13),
#          (15, -14), (16, -14),
#          (16, -15), (17, -15),
#          (17, -16), (18, -16),
#          (18, -17), (19, -17),
#
#          (19, -19), (18, -19), (18, -20), (16, -20),
#          (16, -19), (15, -19),
#
#          (15, -18), (14, -18),
#          (14, -17), (13, -17),
#          (13, -16), (12, -16),
#          (12, -15), (11, -15),
#          (11, -14), (10, -14),
#          (10, -13), (9, -13),
#          (9, -12), (8, -12),
#          (8, -11), (7, -11),
#          (7, -10), (6, -10),
#          (6, -9), (5, -9),
#          (5, -8), (4, -8),
#          (4, -7), (3, -7),
#          (3, -6), (2, -6),
#          (2, -5), (1, -5),
#          (1, -4), (0, -4),
#          (0, -3), (-1, -3),
#          (-1, -2), (-2, -2)]
#
# line_probe_list = [oneP, sep0, sep1, sep2, sep3, sep6, sep18]
# text_list = ['oneP', 'sep0', 'sep1', 'sep2', 'sep3', 'sep6', 'sep18']

# # re-center the stimuli
'''these stim are designed to be centered'''
probeVert = [(0, 0), (1, 0), (1, 1), (2, 1), (2, -1), (1, -1),
                 (1, -2), (-1, -2), (-1, -1), (0, -1)]
centre_sq = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

oneP_vert = [(-1, -2), (-1, 0), (0, 0), (0, 1), (2, 1),
             (2, 0), (1, 0), (1, -1), (0, -1), (0, -2)]
sep0_vert = [(-2, -1), (-2, 1), (-1, 1), (-1, 2), (1, 2),
             (1, -1), (0, -1), (0, -2), (-2, -2)]
sep1_vert = [(-2, 0), (-2, 2), (-1, 2), (-1, 3), (1, 3), (1, 1), (2, 1),
             (2, -1), (1, -1), (1, -2), (-1, -2), (-1, -1), (-2, -1)]
sep2_vert = [(-2, 0), (-2, 2), (-1, 2), (-1, 3), (1, 3), (1, 1), (2, 1), (2, 0),
             (3, 0), (3, -2), (2, -2), (2, -3), (0, -3), (0, -2), (-1, -2),
             (-1, -1), (-2, -1)]
sep3_vert = [(-3, 1), (-3, 3), (-2, 3), (-2, 4), (0, 4), (0, 2), (1, 2), (1, 1),
             (2, 1), (2, 0), (3, 0), (3, -2), (2, -2), (2, -3), (0, -3), (0, -2),
             (-1, -2), (-1, -1), (-2, -1), (-2, 0), (-3, 0)]

sep6_vert = [(-4, 2), (-4, 4), (-3, 4), (-3, 5), (-1, 5), (-1, 3), (0, 3),
             (0, 2), (1, 2), (1, 1), (2, 1), (2, 0), (3, 0), (3, -1), (4, -1),
             (4, -2), (5, -2), (5, -4), (4, -4), (4, -5), (2, -5), (2, -4),
             (1, -4), (1, -3), (0, -3), (0, -2), (-1, -2), (-1, -1), (-2, -1),
             (-2, 0), (-3, 0), (-3, 1), (-4, 1)]

sep18_vert = [(-10, 8), (-10, 10), (-9, 10), (-9, 11), (-7, 11), (-7, 9),
              (-6, 9), (-6, 8), (-5, 8), (-5, 7), (-4, 7), (-4, 6), (-3, 6),
              (-3, 5), (-2, 5), (-2, 4), (-1, 4), (-1, 3), (0, 3), (0, 2),
              (1, 2), (1, 1), (2, 1), (2, 0), (3, 0), (3, -1), (4, -1),
              (4, -2), (5, -2), (5, -3), (6, -3), (6, -4), (7, -4), (7, -5),
              (8, -5), (8, -6), (9, -6), (9, -7), (10, -7), (10, -8), (11, -8),
              (11, -10), (10, -10), (10, -11), (8, -11), (8, -10), (7, -10),
              (7, -9), (6, -9), (6, -8), (5, -8), (5, -7), (4, -7), (4, -6),
              (3, -6), (3, -5), (2, -5), (2, -4), (1, -4), (1, -3), (0, -3),
              (0, -2), (-1, -2), (-1, -1), (-2, -1), (-2, 0), (-3, 0), (-3, 1),
              (-4, 1), (-4, 2), (-5, 2), (-5, 3), (-6, 3), (-6, 4), (-7, 4),
              (-7, 5), (-8, 5), (-8, 6), (-9, 6), (-9, 7), (-10, 7)]

# starts clockwise round probe1 from bottom left
sep36_vert = [(-20, 18), (-20, 20), (-19, 20), (-19, 21), (-17, 21),
              # then down and right diagonally towards mid point
              (-17, 19), (-16, 19),
              (-16, 18), (-15, 18),
              (-15, 17), (-14, 17),
              (-14, 16), (-13, 16),
              (-13, 15), (-12, 15),
              (-12, 14), (-11, 14),
              (-11, 13), (-10, 13),
              (-10, 12), (-9, 12),
              (-9, 11), (-8, 11),
              (-8, 10), (-7, 10),
              (-7, 9), (-6, 9),
              (-6, 8), (-5, 8),
              (-5, 7), (-4, 7),
              (-4, 6), (-3, 6),
              (-3, 5), (-2, 5),
              (-2, 4), (-1, 4),
              (-1, 3), (0, 3),
              (0, 2), (1, 2),
              # (1, 1) is up and right one from bottom right of white spot
              (1, 1), (2, 1),
              (2, 0), (3, 0),
              (3, -1), (4, -1),
              (4, -2), (5, -2),
              (5, -3), (6, -3),
              (6, -4), (7, -4),
              (7, -5), (8, -5),
              (8, -6), (9, -6),
              (9, -7), (10, -7),
              (10, -8), (11, -8),
              (11, -9), (12, -9),
              (12, -10), (13, -10),
              (13, -11), (14, -11),
              (14, -12), (15, -12),
              (15, -13), (16, -13),
              (16, -14), (17, -14 ),
              (17, -15), (18, -15),
              (18, -16), (19, -16),

              # round probe2, (16, -19) is bottom left of probe2
              (19, -18), (18, -18),
              (18, -19), (16, -19),

              # then back up and left along diagonal
              (16, -18), (15, -18),
              (15, -17), (14, -17),
              (14, -16), (13, -16),
              (13, -15), (12, -15),
              (12, -14), (11, -14),
              (11, -13), (10, -13),
              (10, -12), (9, -12),
              (9, -11), (8, -11),
              (8, -10), (7, -10),
              (7, -9), (6, -9),
              (6, -8), (5, -8),
              (5, -7), (4, -7),
              (4, -6), (3, -6),
              (3, -5), (2, -5),
              (2, -4), (1, -4),
              (1, -3), (0, -3),
              (0, -2), (-1, -2),

              # (-1, 1) is down and left one from bottom right of white spot
              (-1, -1), (-2, -1),
              (-2, 0), (-3, 0),
              (-3, 1), (-4, 1),
              (-4, 2), (-5, 2),
              (-5, 3), (-6, 3),
              (-6, 4), (-7, 4),
              (-7, 5), (-8, 5),
              (-8, 6), (-9, 6),
              (-9, 7), (-10, 7),
              (-10, 8), (-11, 8),
              (-11, 9), (-12, 9),
              (-12, 10), (-13, 10),
              (-13, 11), (-14, 11),
              (-14, 12), (-15, 12),
              (-15, 13), (-16, 13),
              (-16, 14), (-17, 14),
              (-17, 15), (-18, 15),
              (-18, 16), (-19, 16),
              (-19, 17), (-20, 17),
              ]  # bottom left of probe1 is (-20, 18),


'''probeVert is Martin's stimulus, oneP is my version.  They are are 180º to each other.  I guess just use mine for both conds but check the orientation.'''
probe_vert_list = [probeVert, oneP_vert, sep0_vert, sep1_vert, sep2_vert, sep3_vert, sep6_vert, sep18_vert, sep36_vert]
text_list = ['probeVert', 'oneP', 'sep0', 'sep1', 'sep2', 'sep3', 'sep6', 'sep18', 'sep36']
circle_probe_rad_list = [2.15, 2.15, 2.5, 2.8, 3.4, 4.1, 6.1, 14.6, 27.3]
# probe_vert_list = [sep36_vert]
# text_list = ['sep36']
# circle_probe_rad_list = [27.3]
message = visual.TextStim(win, text='hello', pos=(0, 150))
message.autoDraw = True  # Automatically draw every frame

probe_pos = [[-9, 9], [12, 1], [0, -12], [-5, 3.5],
             [4, -9], [-5, 5], [-5, 5], [0.5, -.5]]
# probe_pos = [[0.5, -.5]]
size = 10

# scaled_probe_pos = [[i/25*size for i in x] for x in probe_pos]
scaled_probe_pos = [[i*size for i in x] for x in probe_pos]

print(f'scaled_probe_pos\n{scaled_probe_pos}')


for idx, this_stim in enumerate(probe_vert_list):

    line_probe = visual.ShapeStim(win, vertices=this_stim, fillColor=(1.0, 1.0, 1.0),
                                 lineWidth=0, opacity=1, size=size, interpolate=False,)
    circle_probe = visual.Circle(win, radius=circle_probe_rad_list[idx], units='pix', size=size,
                           lineColor='black', fillColor='black', lineWidth=0, interpolate=False)
    mid_point = visual.ShapeStim(win, vertices=centre_sq, fillColor=(1.0, 1.0, 1.0),
                                 lineWidth=0, opacity=1, size=size, interpolate=False,)

    message.text = text_list[idx]  # Change properties of existing stim

    # this_pos = scaled_probe_pos[idx]
    # print(f'\n{text_list[idx]}\tpos: {this_pos}')


    while not event.getKeys():

        line_probe.setOri(5, '-')  # rotate
        # line_probe.pos = this_pos

        circle_probe.draw()
        line_probe.draw()
        win.flip()

    posPix = posToPix(line_probe)
    print(f'posPix: {posPix}')

    print(f'\nget mm and deg for {circle_probe_rad_list[idx]}')
    get_pixel_mm_deg_values(monitor_name='HP_24uh', n_pixels=1)
    get_pixel_mm_deg_values(monitor_name='NickMac', n_pixels=1)
    get_pixel_mm_deg_values(monitor_name='ASUS_2_13_240Hz', n_pixels=1)

    # get_pixel_mm_deg_values(monitor_name='HP_24uh', n_pixels=circle_probe_rad_list[idx])

    win.update()

    #pause, so you get a chance to see it!
    core.wait(2.0)