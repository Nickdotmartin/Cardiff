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




'''
Pixels on the OLED are not the same size as on the 240Hz monitor.
For them to 'appear' to be the same, we need to change the viewer's distance from the screen.
Asus pixels aren't as square as the OLED ones, but I'm going to try to match their diagonal distance.

To do this I can use the find_angle function (below) to get the degree of visual angle for the diagonal of the asus, 
and the oled, the ratio of these two, which I can use to scale the distance.

THE NEW DISTANCE IS 43.74cm

'''


def find_angle(adjacent, opposite):
    """Finds the angle in a right triangle given the lengths of the adjacent and opposite sides.
    e.g., for getting the visual angle of a square at a given distance,
    the adjacent side is the distance from the screen,
    and the opposite side is the size of the square onscreen.

    :param adjacent: A numpy array of the lengths (in cm) of the adjacent sides (e.g., distance z_array).
    :param opposite: The (scalar) length (in cm) of the side opposite the angle you want to find.
    :return: A numpy array of the angles in degrees.
    """
    return rad2deg(arctan(opposite / adjacent))


asus_cal_pix_dict = get_pixel_mm_deg_values(monitor_name='asus_cal', n_pixels=1)
print(f"asus_cal_pix_dict: {asus_cal_pix_dict}")
asus_diag_mm = asus_cal_pix_dict['diag_mm']
print(f"asus_diag_mm: {asus_diag_mm}")

OLED_pix_dict = get_pixel_mm_deg_values(monitor_name='OLED', n_pixels=1)
print(f"OLED_pix_dict: {OLED_pix_dict}")
oled_diag_mm = OLED_pix_dict['diag_mm']
print(f"oled_diag_mm: {oled_diag_mm}")

# get the degree of visual angle for asus_diag_mm at 57.3cm
asus_pix_deg = find_angle(adjacent=57.3, opposite=asus_diag_mm/10)
print(f"asus_pix_deg: {asus_pix_deg}")

# get the degree of visual angle for oled_diag_mm at 57.3cm
oled_pix_deg = find_angle(adjacent=57.3, opposite=oled_diag_mm/10)
print(f"oled_pix_deg: {oled_pix_deg}")

# get the scaled difference for the distances
# scale_factor = asus_pix_deg / oled_pix_deg
scale_factor = oled_pix_deg / asus_pix_deg
print(f"scale_factor: {scale_factor}")

# get the new distance for the OLED where the pixels will appear the same size as on the asus
oled_dist = 57.3 * scale_factor
print(f"oled_dist: {oled_dist}")

# get the new degree of visual angle for the OLED at the new distance
oled_pix_deg = find_angle(adjacent=oled_dist, opposite=oled_diag_mm/10)
print(f"oled_pix_deg: {oled_pix_deg}")


