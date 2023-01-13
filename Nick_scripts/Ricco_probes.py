from psychopy import visual, core  # import some libraries from PsychoPy
from psychopy import visual, event, core
from psychopy.tools.monitorunittools import posToPix
from PsychoPy_tools import get_pixel_mm_deg_values
import numpy as np


def make_ricco_vertices(sep_cond, balanced=False, verbose=False):
    """
    Probe vertices can be constructed from four parts.
        1. the top left edge of probe 1 (which is the same for all conds).
        2. zig-zag down top-right side (which is has more vertices as sep_cond increases).
        3. bottom-right of probe 2 (calculated by adjusting values from sep0).
        4. zig-zag back up bottom-left side (which is has more vertices as sep_cond increases).

    For 1probe condition (sep=99 or -1) it just loads vertices rather than generating them.

    :param sep_cond: equivalent separation condition from exp 1.
    :param balanced: (default = False).  If False, (0, 0) is at the bottom of the middle part of the 'M' of probe1 for all probes.
                    In other words, they are not evenly spread around (0, 0), probe2 is further away from (0, 0) for higher sep values.
                    This is consistent with Exp1 stimuli, where probe1 is always in the same spo, regardless of sep.
                    If True, stimuli are balanced around (0, 0), as was the case for previous Ricco experiments.
    :param verbose: print sections to screen as they are generated

    :return: verticies to draw probe.
    """
    # print(f"\nsep_cond: {sep_cond}")

    '''top-left of pr1: Use these vertices for all probes'''
    tl_pr1_1 = [(-2, 1), (-1, 1), (-1, 2), (1, 2)]  # top-left of pr1

    if sep_cond in [-1, 99]:
        '''1probe condition, just load vertices'''
        tr_zz_2 = [(1, 1)]
        br_pr2_3 = [(0, 1), (0, 0), (-1, 0), (-1, -1)]
        bl_zz_4 = [(-2, -1)]

    else:  # if not 1probe (sep not in [-1, 99])

        '''zig-zag down top-right: 
        for tr_zz_2, generate x and y values based on separation, then zip.'''
        # tr_zz_2_x_vals start from 1 (once), and then repeat each number (twice) up to sep_cond+1 (once).
        tr_zz_2_x_vals = list(np.repeat(list(range(1, sep_cond+2)), 2))[1:-1]

        # tr_zz_2_y_vals start from zero (twice) and repeat each number (twice) down to -sep_cond+1.
        tr_zz_2_y_vals = list(np.repeat(list(range(0, -sep_cond, -1)), 2))

        # zip x and y together to make list of tuples
        tr_zz_2 = list(zip(tr_zz_2_x_vals, tr_zz_2_y_vals))

        '''bottom-right of pr2: use the values from sep0 as the default and adjust based on sep_cond'''
        br_pr2_sep0 = [(1, -1), (0, -1), (0, -2), (-2, -2)]
        br_pr2_3 = [(i[0]+sep_cond, i[1]-sep_cond) for i in br_pr2_sep0]

        '''zig-zag back up bottom-left side:
        For bl_zz_4_x_vals, generate x and y values based on separation, then zip.'''
        # bl_zz_4_x_vals have same structure as tr_zz_2_x_vals:
        #   first value, once, then each number repeats twice apart from last one (once).
        # bl_zz_4_x_vals start positive and decrement until -2.
        bl_zz_4_x_vals = list(np.repeat(list(range(-2+sep_cond, -3, -1)), 2))
        bl_zz_4_x_vals = bl_zz_4_x_vals[1:-1]

        # bl_zz_4_y_vals start from -1-sep_cond (twice) and repeat each number (twice) up to -2.
        # print(f"tr_zz_2_y_vals: {tr_zz_2_y_vals}")
        bl_zz_4_y_vals = list(np.repeat(list(range(-1-sep_cond, -1)), 2))

        # zip x and y together to make list of tuples
        bl_zz_4 = list(zip(bl_zz_4_x_vals, bl_zz_4_y_vals))
        # print(f"bl_zz_4: {bl_zz_4}")

    if verbose:
        print(f"\nsep_cond: {sep_cond}")
        print(f"tl_pr1_1: {tl_pr1_1}")
        print(f"tr_zz_2: {tr_zz_2}")
        print(f"br_pr2_3: {br_pr2_3}")
        print(f"bl_zz_4: {bl_zz_4}")

    new_verticies = tl_pr1_1 + tr_zz_2 + br_pr2_3 + bl_zz_4

    if balanced:
        if verbose:
            print('balancing probe around (0, 0)')
        # balancing is roughly based on half the separation value, but with slight differences for odd and even numbers.
        if sep_cond in [-1, 99]:
            half_sep = 0
        elif (sep_cond % 2) != 0:
            half_sep = int(sep_cond / 2) + 1
        else:
            half_sep = int(sep_cond / 2)

        balanced_vertices = [(tup[0] - (half_sep - 1), tup[1] + half_sep) for tup in new_verticies]

        new_verticies = balanced_vertices

    return new_verticies


#create a window
win = visual.Window(monitor="Nick_work_laptop", units="pix",
                    colorSpace='rgb',
                    color=[-0.1, -0.1, -0.1],  # bgcolor from Martin's flow script, not bgColor255
                    winType='pyglet',  # I've added pyglet to make it work on pycharm/mac
                    screen=0,
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
probe_vert_list = [oneP_vert, sep0_vert, sep1_vert, sep2_vert, sep3_vert, sep6_vert, sep18_vert, sep36_vert]
text_list = ['oneP', 'sep0', 'sep1', 'sep2', 'sep3', 'sep6', 'sep18', 'sep36']
circle_probe_rad_list = [2.15, 2.5, 2.8, 3.4, 4.1, 6.1, 14.6, 27.3]


# from operator import itemgetter
separations = [-1, 0, 1, 2, 3, 6, 18, 36]
vert_dict = {}
for sep_idx, sep_cond in enumerate(separations):
    print(f"\n{sep_idx}. sep_cond: {sep_cond}")
    # print(sep_cond, make_ricco_vertices(sep_cond))
    new_verts = make_ricco_vertices(sep_cond, balanced=True)
    orig_verts = probe_vert_list[sep_idx]

    vert_dict[f"sep{sep_cond}"] = new_verts
    vert_dict[f"sep{sep_cond}"] = {}
    vert_dict[f"sep{sep_cond}"]['sep_cond'] = sep_cond
    vert_dict[f"sep{sep_cond}"]['vertices'] = new_verts
    vert_dict[f"sep{sep_cond}"]['orig_verts'] = orig_verts

for k, v in vert_dict.items():
    print(k, v)

# probe_vert_list = [sep36_vert]
# text_list = ['sep36']
# circle_probe_rad_list = [27.3]
message = visual.TextStim(win, text='hello', pos=(0, 150))
message.autoDraw = True  # Automatically draw every frame

# probe_pos = [[-9, 9], [12, 1], [0, -12], [-5, 3.5],
#              [4, -9], [-5, 5], [-5, 5], [0.5, -.5]]
# probe_pos = [[0.5, -.5]]
# probe_pos = np.tile((0, 0), len(probe_vert_list))
# probe_pos = (0, 0) * len(probe_vert_list)
probe_pos = [[0, 0]] * len(probe_vert_list)
print(f'probe_pos\n{probe_pos}')

size = 10

# scaled_probe_pos = [[i/25*size for i in x] for x in probe_pos]
scaled_probe_pos = [[i*size for i in x] for x in probe_pos]

print(f'scaled_probe_pos\n{scaled_probe_pos}')


# for idx, sep_name in enumerate(probe_vert_list):
for idx, sep_name in enumerate(vert_dict.keys()):


    sep_cond = vert_dict[sep_name]['sep_cond']
    print(idx, sep_name, sep_cond)

    new_verts = vert_dict[sep_name]['vertices']
    orig_verts = vert_dict[sep_name]['orig_verts']



    line_probe_orig = visual.ShapeStim(win, vertices=orig_verts, fillColor=(1.0, 1.0, 1.0), pos=[-100, 0],
                                 lineWidth=0, opacity=1, size=size, interpolate=False,)
    circle_probe_orig = visual.Circle(win, radius=circle_probe_rad_list[idx], units='pix', size=size, pos=[-100, 0],
                           lineColor='black', fillColor='black', lineWidth=0, interpolate=False)
    mid_point_orig = visual.ShapeStim(win, vertices=centre_sq, fillColor=(1.0, 1.0, 1.0), pos=[-100, 0],
                                 lineWidth=0, opacity=1, size=size, interpolate=False,)

    line_probe_new = visual.ShapeStim(win, vertices=new_verts, fillColor=(1.0, 1.0, 1.0), pos=[100, 0],
                                 lineWidth=0, opacity=1, size=size, interpolate=False,)
    circle_probe_new = visual.Circle(win, radius=circle_probe_rad_list[idx], units='pix', size=size, pos=[100, 0],
                           lineColor='black', fillColor='black', lineWidth=0, interpolate=False)
    mid_point_new = visual.ShapeStim(win, vertices=centre_sq, fillColor=(1.0, 1.0, 1.0), pos=[100, 0],
                                 lineWidth=0, opacity=1, size=size, interpolate=False,)

    # line_probe = visual.ShapeStim(win, vertices=sep_name, fillColor=(1.0, 1.0, 1.0),
    #                              lineWidth=0, opacity=1, size=size, interpolate=False,)
    # circle_probe = visual.Circle(win, radius=circle_probe_rad_list[idx], units='pix', size=size,
    #                        lineColor='black', fillColor='black', lineWidth=0, interpolate=False)
    # mid_point = visual.ShapeStim(win, vertices=centre_sq, fillColor=(1.0, 1.0, 1.0),
    #                              lineWidth=0, opacity=1, size=size, interpolate=False,)

    message.text = text_list[idx]  # Change properties of existing stim

    # this_pos = scaled_probe_pos[idx]
    # print(f'\n{text_list[idx]}\tpos: {this_pos}')


    while not event.getKeys():

        circle_probe_orig.draw()
        circle_probe_new.draw()

        # line_probe_orig.setOri(5, '-')  # rotate
        # line_probe_new.setOri(5, '-')  # rotate


        line_probe_orig.draw()
        line_probe_new.draw()
        win.flip()


    # posPix = posToPix(line_probe)
    # print(f'posPix: {posPix}')

    print(f'\nget mm and deg for {circle_probe_rad_list[idx]}')
    get_pixel_mm_deg_values(monitor_name='Nick_work_laptop', n_pixels=1)


    win.update()

    #pause, so you get a chance to see it!
    core.wait(2.5)

