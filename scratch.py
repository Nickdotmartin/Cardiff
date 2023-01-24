import random
from operator import itemgetter

import pandas as pd
import numpy as np


probe_duration = 2
ISI_list = [2, 0, -1]
# for ISI in ISI_list:

ISI =-1

# timing in frames for ISI and probe2
# If probes are presented concurrently, set ISI and probe2 to last for 0 frames.
isi_dur_fr = ISI
p2_fr = probe_duration
if ISI < 0:
    isi_dur_fr = p2_fr = 0

# cumulative timing in frames for each part of a trial
t_fixation = 30  # int(fps / 2) + vary_fix
t_probe_1 = t_fixation + probe_duration
t_ISI = t_probe_1 + isi_dur_fr
t_probe_2 = t_ISI + p2_fr
# t_response = t_probe_2 + 10000 * fps  # ~40 seconds to respond
t_response = t_probe_2 + 10  # ~40 seconds to respond

frame_list = list(range(25, 45))
print(frame_list)

# for frameN in frame_list:
#     if 0 < frameN <= t_fixation:
#         print(f"{frameN}: 'fix'")
#     elif t_fixation < frameN <= t_probe_1:
#         print(f"{frameN}: 'p1'")
#     elif t_probe_1 < frameN <= t_ISI:
#         print(f"{frameN}: 'ISI'")
#     elif t_ISI < frameN <= t_probe_2:
#         print(f"{frameN}: 'p2'")
#     elif t_probe_2 < frameN <= t_response:
#         print(f"{frameN}: 'resp'")
#     else:
#         print(f"{frameN}: \t'dunno'")

for frameN in frame_list:
    if t_fixation >= frameN > 0:
        print(f"{frameN}: 'fix'")
    elif t_probe_1 >= frameN > t_fixation:
        print(f"{frameN}: 'p1'")
    elif t_ISI >= frameN > t_probe_1:
        print(f"{frameN}: 'ISI'")
    elif t_probe_2 >= frameN > t_ISI:
        print(f"{frameN}: 'p2'")
    elif t_response >= frameN > t_probe_2:
        print(f"{frameN}: 'resp'")
    # else:
    #     print(f"{frameN}: \t'dunno'")