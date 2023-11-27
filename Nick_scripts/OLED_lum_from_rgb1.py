import numpy as np
import matplotlib.pyplot as plt

'''
This script is used to get accurate luminance values from the OLED monitor.
It is based on a set of rgb1 and luminance measurements taken by Nick.

These measurements show that the monitor was not linear, and we haven't been able to linearise it.

The lumiance values it gave assumed it was linear, so were incorrect.

I will find the curve that best fits the measured lum values and use that to find the luminance values.

Measurements were taken on 17/11/2023 by Nick with spyderX pro running DisplayCal on MacBook

'''

'''Don't use this version of the function, use the one copied into rad_flow_psignifit_analysis.py'''
def get_OLED_luminance(rgb1):
    """
    This function takes a list of rgb1 values and returns a list of luminance values.
    New luminance values are calculated using a polynomial fit to the measured values.
    Measurements were taken on 17/11/2023 by Nick with spyderX pro running DisplayCal on MacBook.
    :param rgb1: value to be converted to luminance
    :return: corresponding luminance value
    """

    '''data to use for fitting'''
    # measurements were taken at 18 evenly spaced rgb1 points between 0 and 1
    rbg1_values = [0, 0.058823529, 0.117647059, 0.176470588, 0.235294118, 0.294117647, 0.352941176,
                   0.411764706, 0.470588235, 0.529411765, 0.588235294, 0.647058824, 0.705882353,
                   0.764705882, 0.823529412, 0.882352941, 0.941176471, 1]

    # measured luminance values for each of the rgb1 values
    measured_lum = [0.01, 0.17,	0.48, 0.91, 1.55, 2.45, 3.58,
                    4.91, 6.49, 8.4, 10.37, 12.77, 13.03,
                    16.3, 19.61, 23.26, 24.78, 24.8]

    # because of the kink in the data (after rgb1=.64) I'm just going to take the first 12 values
    measured_lum = measured_lum[:12]
    rbg1_values = rbg1_values[:12]


    '''fitting the curve'''
    # calculate polynomial to fit the leasured values.
    z = np.polyfit(rbg1_values, measured_lum, 3)
    f = np.poly1d(z)


    '''getting correct luminance values'''
    if type(rgb1) == list:
        rgb1 = np.array(rgb1)
    new_lum = f(rgb1)

    return new_lum


#
# '''Run this code to plot the incorrect and correct values.'''
# just_low_vals = True
# '''data to use for fitting'''
# # measurements were taken at 18 evenly spaced rgb1 points between 0 and 1
# rbg1_values = [0, 0.058823529, 0.117647059, 0.176470588, 0.235294118, 0.294117647, 0.352941176,
#                0.411764706, 0.470588235, 0.529411765, 0.588235294, 0.647058824, 0.705882353,
#                0.764705882, 0.823529412, 0.882352941, 0.941176471, 1]
#
# # measured luminance values for each of the rgb1 values
# measured_lum = [0.01, 0.17, 0.48, 0.91, 1.55, 2.45, 3.58, 4.91, 6.49, 8.4, 10.37, 12.77, 13.03, 16.3, 19.61, 23.26,
#                 24.78, 24.8]
#
# # because of the kink in the data (after rgb1=.64) I'm just going to take the first 12 values
# if just_low_vals:
#     measured_lum = measured_lum[:12]
#     rbg1_values = rbg1_values[:12]
#
# '''fitting the curve'''
# # calculate polynomial to fit the leasured values.
# z = np.polyfit(rbg1_values, measured_lum, 3)
# f = np.poly1d(z)
# '''making plots'''
# x_vals = rbg1_values
# y_vals = measured_lum
#
#
#
# # plot the x_vals and y_vals with blue dots
# plt.plot(x_vals, y_vals, 'o', label='measured lum values', c='blue')
#
# # infer lots of new x's and y's for plotting the fitted curve with a green line
# x_interpolate = np.linspace(x_vals[0], x_vals[-1], 50)
# y_interpolate = f(x_interpolate)
# plt.plot(x_interpolate, y_interpolate, label='adjusted values', c='green')
#
# # for comparison, these are values from an output csv showing the (incorrect) luminance values it gave.
# incorrect_lum_rgb1s = [0, .07, .175, .28, .385, .49, .7, 1]
# incorrect_lum = [0, 1.75, 4.375, 7, 9.625, 12.25, 17.5, 25]
# if just_low_vals:
#     # remove datapoints after rgb1=.64
#     incorrect_lum_rgb1s = incorrect_lum_rgb1s[:-2]
#     incorrect_lum = incorrect_lum[:-2]
# old_xs = incorrect_lum_rgb1s
# old_ys = incorrect_lum
# # for comparisson, plot the old values in yellow
# plt.scatter(old_xs, old_ys, c='y', label='old values')
#
# # # calculate polynomial to fit the old values and plot it with orange line
# z_old = np.polyfit(old_xs, old_ys, 1)
# f_old = np.poly1d(z_old)
# y_interp_old = f_old(x_interpolate)
# plt.plot(x_interpolate, y_interp_old, c='orange', label='old values fitted')
#
# # decorate the plot
# plt.suptitle('Measured luminance values for OLED screen (spyder 17/11/2023)')
# if just_low_vals:
#     plt.title('only using the first 12 values')
# else:
#     plt.title('all 18 values')
# plt.xlabel('RGB1 values')
# plt.ylabel('Luminance (cd/m$^2$)')
# plt.legend(['measured lum values', 'fitted values', 'incorrect lum', 'old values fitted'])
#
# plt.show()

