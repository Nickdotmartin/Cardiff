import random
from operator import itemgetter
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

# stair_names_list = ['sep5_ISI-1', 'sep5_ISI0', 'sep5_ISI2']
# colour_dict = {'sep5_ISI-1': (0.00392156862745098, 0.45098039215686275, 0.6980392156862745),
#                'sep5_ISI0': (0.8705882352941177, 0.5607843137254902, 0.0196078431372549),
#                'sep5_ISI2': (0.00784313725490196, 0.6196078431372549, 0.45098039215686275)}
#
# cond_list = ['sep5_ISI0', 'sep5_ISI-1', 'sep5_ISI0', 'sep5_ISI-1', 'sep5_ISI2','sep5_ISI0', 'sep5_ISI2','sep5_ISI-1', 'sep5_ISI2']
#
# cond_colour_list = [colour_dict[i] for i in cond_list]
# print(cond_colour_list)

stair_name = 'exp_sep5_ISI2'
print(stair_name[:4])
print(stair_name[4:])

