import random
from operator import itemgetter
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

probeVert = [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (0, 2), (0, 1),
             (-1, 1), (-1, 0), (-2, 0), (-2, -2), (-1, -2), (-1, -1), (0, -1)]

new_vert = []

for i in probeVert:
    this_vert = (i[0] + 1, i[1]-1)
    new_vert.append(this_vert)

print(new_vert)