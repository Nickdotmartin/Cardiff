import random
from operator import itemgetter

import pandas as pd
import numpy as np

rptd_trial_x_locs = [[0, 1], [4, 6], [10, 19]]
for loc_pair in rptd_trial_x_locs:
    x0, x1 = loc_pair[0], loc_pair[1]
    print(x0, x1)