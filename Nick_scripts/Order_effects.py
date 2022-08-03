



'''
To understand why there is a difference in rad_flow_2 data at sep2 when analysed
using the last luminance value (e.g., step25), yet there is no difference when
using all 25 datapoints with psignifit; this script will look for order effects.
That is, do participants show adaptation to repeated radial flow motion.
This method was suggest by Simon (From Teams > Simon, 24/03/2022).

For each ISI, sep=2, plot all the congruent and incongruent staircases onto one plot,
including a mean staircase.
If there is an order effect then the staircases will diverge over time.

1. loop through all conditions.
2. load data for staircases.
3. add data to master list for calculating mean

4. # two options here
a) add data to plot in loop, one line at a time.
b) create plot from master list with multiple lines simultaneously

5. add mean lines


I could base process on b3_plot_staircase.  But I need to adapt it.
Don't show final thr hline
Don't have multiple panels
(although I could do all stairs and mean on different plots if its too busy)
'''

