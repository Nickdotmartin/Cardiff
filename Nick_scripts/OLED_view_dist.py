import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from psychology_stats import linear_reg_OLS, p_val_sig_stars
from python_tools import print_nested_round_floats
from numpy import array, random, where, sum, linspace, pi, rad2deg, arctan, arctan2, cos, sin, hypot
from PsychoPy_tools import get_pixel_mm_deg_values


def find_angle(adjacent, opposite):
    """Finds the angle in a right triangle given the lengths of the adjacent and opposite sides.
    e.g., for getting the visual angle of a square at a given distance,
    the adjacent side is the distance from the screen,
    and the opposite side is the size of the square onscreen.

    :param adjacent: A numpy array of the lengths of the adjacent sides (e.g., distance z_array).
    :param opposite: The (scalar) length of the side opposite the angle you want to find.
    :return: A numpy array of the angles in degrees.
    """
    return rad2deg(arctan(opposite / adjacent))

'''
Pixels on the OLED are not the same size as on the 240Hz monitor.
For them to 'appear' to be the same, we need to change the viewer's distance from the screen.

To do this I can use the find_angle function (above)

'''

asus_cal_pix_dict = get_pixel_mm_deg_values(monitor_name='asus_cal', n_pixels=1)
print(f"asus_cal_pix_dict: {asus_cal_pix_dict}")
'''asus pixels are not sqaure, taller than wide, so I am going to take their mean'''
asus_pix_mm = np.mean([asus_cal_pix_dict['width_mm'], asus_cal_pix_dict['height_mm']])
print(f"asus_pix_mm: {asus_pix_mm}")

OLED_pix_dict = get_pixel_mm_deg_values(monitor_name='OLED', n_pixels=1)
print(f"OLED_pix_dict: {OLED_pix_dict}")
oled_pix_mm = np.mean([OLED_pix_dict['width_mm'], OLED_pix_dict['height_mm']])
print(f"oled_pix_mm: {oled_pix_mm}")



