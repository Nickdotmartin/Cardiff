import os
import numpy as np
from scipy.stats import linregress
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from rad_flow_psignifit_analysis import fig_colours
from PsychoPy_tools import get_pixel_mm_deg_values




def ricco_bloch_fit(log_x_array, log_y_array, est_intercept1=None, est_breakpoint=None,
                    point_labels=None, x_axis_label=None, y_axis_label=None,
                    exp_version=None, p_name=None, save_path=None, save_name=None):
    '''
    Function to estimate the size of Ricco's area or Bloch's duration using iterative two-phase least squares.
    The slope of the first line is fixed at -1.
    Estimates of the intercept of the first line, the slope of the second line and breakpoint are given as staring values,
    which the regression improves on.

    Results are plotted showing the -1 slope, 2nd slope and the estimated size (for Ricco/Bloch).

    :param log_x_array: a numpy array of log10 values of the x-axis (e.g., stimulus size: diag_min or probe_dur_ms)
    :param log_y_array: a numpy array of log10 values of the y-axis (e.g., response size: delta I)
    :param est_intercept1: estimate of the intercept of the 1st slope.  If None, will automatically produce an estimate.
    :param est_breakpoint: estimate of the breakpoint.  If None, will automatically start from data mid-point.
    :param point_labels: Labels for the datapoints, relating to stimulus size (e.g., sep or ISI vals from exp 1)
    :param x_axis_label: Optional label for x-axis.
    :param y_axis_label: Optional label for y-axis.

    :return: fig, slope2, breakpoint, r2,
    '''

    print("\n***Running ricco_bloch_fit ***")

    # first line is fixed at slope -1
    slope1 = -1

    #  get estimate for the intercept of slope 1 (y - x)
    if not est_intercept1:
        est_intercept1 = log_y_array[0] - log_x_array[0]
    intercept1 = est_intercept1

    # get estimate of breakpoint, if None, third to last point
    if not est_breakpoint:
        est_breakpoint = log_x_array[int(len(log_x_array) - 3)]

    # second line is fitted
    x2 = log_x_array[log_x_array > est_breakpoint]
    y2 = log_y_array[log_x_array > est_breakpoint]
    slope2, intercept2, r_value, p_value, std_err = linregress(x2, y2)

    # breakpoint is fitted with max 1000 iterations, slope1 is fixed at -1
    print("\nfitting breakpoint")
    breakpoint = est_breakpoint
    breakpoint_old = 0
    i = 0
    while abs(breakpoint - breakpoint_old) > 0.0001 and i < 1000:

        # check breakpoint has two datapoint either side to allow 2 lines to be fitted
        if breakpoint < log_x_array[1]:
            print(f"breakpoint < log_x_array[1]: {breakpoint} < {log_x_array[1]}.  "
                  f"NEW breakpoint: {log_x_array[1] + .001}")
            breakpoint = log_x_array[1] + .001
        elif breakpoint > log_x_array[-2]:
            print(f"breakpoint > log_x_array[-2]: {breakpoint} > {log_x_array[-2]}.  "
                  f"NEW breakpoint: {log_x_array[-2] - .001}")
            breakpoint = log_x_array[-2] - .001

        breakpoint_old = breakpoint
        x1 = log_x_array[log_x_array < breakpoint]
        y1 = log_y_array[log_x_array < breakpoint]
        x2 = log_x_array[log_x_array > breakpoint]
        y2 = log_y_array[log_x_array > breakpoint]
        print(f"{i}. breakpoint_old: {breakpoint_old}")
        print(f"x1 (len: {len(x1)}): {x1}, y1 (len: {len(y1)}): {y1}, "
              f"x2 (len: {len(x2)}): {x2}, y2 (len: {len(y2)}): {y2}")

        # calculate the y intercept of the first line
        intercept1 = np.mean([y1[i] - slope1 * x1[i] for i in range(len(x1))])
        print(f"intercept1: {intercept1}")

        # calculate the slope and intercept of the second line
        slope2, intercept2, r_value, p_value, std_err = linregress(x2, y2)

        # calculate breakpoint of two lines
        breakpoint = (intercept2 - intercept1) / (slope1 - slope2)

        i += 1


    # r2
    x1 = log_x_array[log_x_array < breakpoint]
    x2 = log_x_array[log_x_array > breakpoint]
    y1_fit = slope1 * x1 + intercept1
    y2_fit = slope2 * x2 + intercept2
    y_fit = np.concatenate((y1_fit, y2_fit))
    r2 = r2_score(log_y_array, y_fit)

    # extend lines to meet at breakpoint
    x1 = np.concatenate((x1, np.array([breakpoint])))
    x2 = np.concatenate((x2, np.array([breakpoint])))
    y1_fit = np.concatenate((y1_fit, np.array([slope1 * breakpoint + intercept1])))
    y2_fit = np.concatenate((y2_fit, np.array([slope2 * breakpoint + intercept2])))


    # plot the datapoints
    fig, ax = plt.subplots(figsize=(6, 6))
    if point_labels is None:
        plt.scatter(x=log_x_array, y=log_y_array, marker='o', s=100, label='data')
    else: # different colour and label for each point
        my_colours = fig_colours(len(log_x_array))
        for i in range(len(log_x_array)):
            plt.plot(log_x_array[i], log_y_array[i], marker="o", markersize=10,
                     color=my_colours[i], label=point_labels[i])

    # make the axes have same length
    plt.axis('square')

    # plot the fit
    plt.plot(x1, y1_fit, 'lightgrey', linestyle='-.', label="-1 slope")
    plt.plot(x2, y2_fit, 'silver', linestyle='--', label=f"{round(slope2, 2)} slope")

    # add vertical line for breakpoint from breakpoint co-odinates to mix y-axis value
    ymin, ymax = ax.get_ylim()
    plt.plot([breakpoint, breakpoint], [y1_fit[-1], ymin], 'k--', scaley=False, label=f'est_size: {round(breakpoint, 2)}')

    # add title and axis labels to plot
    if not exp_version:
        fig_title = f"estimated breakpoint at: {round(breakpoint, 2)}. slope2: {round(slope2, 2)}. r2: {round(r2, 2)}"
    else:
        fig_title = f"{exp_version} breakpoint at: {round(breakpoint, 2)}. slope2: {round(slope2, 2)}. r2: {round(r2, 2)}"
    if p_name:
        fig_title = f"{p_name}: {fig_title}"

    plt.title(fig_title)

    if not x_axis_label:
        x_axis_label = 'log(stim strength)'
        if 'Ricco' in exp_version:
            x_axis_label = 'log(stim length)'
        elif 'Bloch' in exp_version:
            x_axis_label = 'log(stim duration ms)'
    plt.xlabel(x_axis_label)

    if not y_axis_label:
        y_axis_label = 'log(∆ threshold)'
    plt.ylabel(y_axis_label)
    plt.legend()

    if save_path is not None:
        if save_name is not None:
            plt.savefig(os.path.join(save_path, save_name))



    pixel_mm_deg_dict = get_pixel_mm_deg_values(monitor_name='asus_cal')
    print('pixel_mm_deg_dict.items()')
    for k, v in pixel_mm_deg_dict.items():
        print(k, ': ', v)

    # # get breakpoint converted back to original scale and translate to exp units
    print(f"\nconvert breakpoint value: ({breakpoint})")
    exp_breakpoint = np.exp(breakpoint)
    print(f"exp_breakpoint: {exp_breakpoint}")

    breakpoint_dict = {'breakpoint': breakpoint,
                       'slope1': slope1,
                       'intercept1': intercept1,
                       'slope2': slope2,
                       'intercept2': intercept2,
                       'r2': r2,
                       }


    # get original units for breakpoint
    if 'min' in x_axis_label:
        # exponent of breakpoint converts back to size in minutes
        breakpoint_min = exp_breakpoint
        breakpoint_dict['min'] = breakpoint_min

        breakpoint_deg = breakpoint_min / 60  # size in degrees
        breakpoint_dict['deg'] = breakpoint_deg

        breakpoint_pix = breakpoint_deg / pixel_mm_deg_dict['diag_deg']  # length in diag pixels
        breakpoint_dict['pix'] = breakpoint_pix

        breakpoint_sep = breakpoint_pix - 2.5  # exp1 separation values
        breakpoint_dict['sep'] = breakpoint_sep

    elif 'dur' in x_axis_label:
        # exponent of breakpoint converts back to duration in ms
        breakpoint_ms = exp_breakpoint
        breakpoint_dict['ms'] = breakpoint_ms

        one_frame = 1000 / 240
        breakpoint_fr = breakpoint_ms / one_frame  # duration in frames
        breakpoint_dict['fr'] = breakpoint_fr

        breakpoint_isi = breakpoint_fr - 4  # duration in ISI values
        breakpoint_dict['isi'] = breakpoint_isi

    else:
        print("x axis units not recognised.")
        breakpoint_dict['np.exp(breakpoint)'] = exp_breakpoint

    print('\n\nbreakpoint_dict.items()')
    for k, v in breakpoint_dict.items():
        print(k, v)

    print("\n***Finished ricco_bloch_fit ***")

    return fig, breakpoint_dict

##################
#
# tony_x = np.exp(np.array([-2.06168, -1.61702, -1.18638, -0.759550, -0.101964, 0.667556]))
# tony_y = np.exp(np.array([2.1, 1.6, 1.19, 0.99, 0.83, 0.63]))
# est_intercept1 = -.3
# est_slope2 = .25
# est_breakpoint = -1
#
# # nick data
# diag_min = np.array([3.863123978, 6.438539964, 9.013955949, 11.58937193, 14.16478792, 16.74020391,
#                      21.89103588, 27.04186785, 32.19269982, 37.34353179, 52.7960277], dtype='float')
#
# delta_I = np.array([42.47651194, 25.59670762, 16.51389355, 14.40893667, 12.56193105, 9.749886427,
#                     8.517070092, 7.636765425, 7.365920293, 6.248884876, 5.190982016], dtype='float')
#
# # tony data
# diag_deg = np.array([0.193156199, 0.279003398, 0.321926998, 0.364850598, 0.407774198, 0.450697797, 0.536544997, 0.622392196], dtype='float')
# diag_min = diag_deg * 60
# delta_I = np.array([14.34573709, 10.54470039, 9.706980014, 9.010134084, 9.112683076, 8.647930959, 8.120115479, 7.707480824], dtype='float')
# sep_vals_list = [2, 4, 5, 6, 7, 8, 10, 12]
#
#
# orig_x = diag_min
# print(f"orig_x: {orig_x}")
# orig_y = delta_I
# print(f"orig_y: {orig_y}")
#
# log_x = np.log(orig_x)
# print(f"log_x: {log_x}")
# log_y = np.log(orig_y)
# print(f"log_y: {log_y}")
#
# # intercept of a line with slope -1 from a single datapoint is given as y - x
# est_intercept1 = log_y[0] - log_x[0]
# print(f"est_intercept1: {est_intercept1}")
#
# est_slope2 = .25
# est_breakpoint = log_x[int(len(log_x) / 2)]
# print(f"est_breakpoint: {est_breakpoint}")
#
#
# ricco_bloch_fit(log_x_array=log_x, log_y_array=log_y,
#                 # est_intercept1=est_intercept1,
#                 # est_breakpoint=est_breakpoint,
#                 point_labels=sep_vals_list
#                 )
# plt.show()



