import random

import pandas as pd
import numpy as np

# use these for the lines conditions
oneP_vert = [(-2, 1), (-1, 1), (-1, 2), (1, 2),
             (1, 1), (0, 1), (0, 0), (-1, 0), (-1, -1), (-2, -1)]
sep0_vert = [(-2, 1), (-1, 1), (-1, 2), (1, 2),  # top-left of pr1
             (1, -1), (0, -1), (0, -2), (-2, -2)]  # bottom-right of pr2
sep1_vert = [(-2, 1), (-1, 1), (-1, 2), (1, 2),  # top-left of pr1
             (1, 0), (2, 0), # zig-zag down top-right
             (2, -2), (1, -2), (1, -3), (-1, -3), # bottom-right of pr2
             (-1, -2), (-2, -2)]  # zig-zag back up bottom-left
sep2_vert = [(-2, 1), (-1, 1), (-1, 2), (1, 2),  # top-left of pr1
             (1, 0), (2, 0), (2, -1), (3, -1),  # zig-zag down top-right
             (3, -3), (2, -3), (2, -4), (0, -4),  # bottom-right of pr2
             (0, -3), (-1, -3), (-1, -2), (-2, -2)]  # zig-zag back up bottom-left
sep3_vert = [(-2, 1), (-1, 1), (-1, 2), (1, 2),  # top-left of pr1
             (1, 0), (2, 0), (2, -1), (3, -1), (3, -2), (4, -2), # zig-zag down top-right
             (4, -4), (3, -4), (3, -5), (1, -5),  # bottom-right of pr2
             (1, -4), (0, -4), (0, -3), (-1, -3), (-1, -2), (-2, -2)]  # zig-zag back up bottom-left
sep4_vert = [(-2, 1), (-1, 1), (-1, 2), (1, 2),  # top-left of pr1
             (1, 0), (2, 0), (2, -1), (3, -1), (3, -2), (4, -2), (4, -1), (5, -3), # zig-zag down top-right
             (5, -5), (4, -5), (4, -6), (2, -6),  # bottom-right of pr2
             (2, -5), (1, -5), (1, -4), (0, -4), (0, -3), (-1, -3), (-1, -2), (-2, -2)]  # zig-zag back up bottom-left
# probe_vert_list = [sep0_vert, sep1_vert, sep2_vert, sep3_vert]
def make_ricco_vertices(sep_cond, verbose=False):
    """
    Vertices are centered at (0, 0), the bottom of the middle part of the 'M' of probe1 for all probes.
    In other words, they are not evenly spread around (0, 0), probe2 is further away for higher sep values.
    Probe vertices can be constructed from four parts.
        1. the top left edge of probe 1 (which is the same for all conds).
        2. zig-zag down top-right side (which is has more vertices as sep_cond increases).
        3. bottom-right of probe 2 (calculated by adjusting values from sep0).
        4. zig-zag back up bottom-left side (which is has more vertices as sep_cond increases).

    For 1probe condition (sep=99 or -1) it just loads vertices rather than generating them.

    :param sep_cond: equivalent separation condition from exp 1.
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

    return new_verticies


separations = [0, 1, 2, 3, 4, 99]
vert_dict = {}
for sep_cond in separations:
    # print(sep_cond, make_ricco_vertices(sep_cond))
    new_verts = make_ricco_vertices(sep_cond)
    vert_dict[f"sep{sep_cond}_vert"] = new_verts
    # d["string{0}".format(x)] = "Hello"
    # print(sep_cond, new_verts)
for k, v in vert_dict.items():
    print(k, v)

