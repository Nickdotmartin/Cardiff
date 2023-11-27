import numpy as np
import matplotlib.pyplot as plt
'''
This script is used to linearise the RGB values of the OLED screen.
It is based on a set of rgb1 and luminance measurements taken by Nick.

From these it will find the equation of the line of best fit.

I will find the straight line from the fist to last points that represents linear values.

I will then use this to linearise the RGB values of the OLED screen.

Measurements were taken on 17/11/2023 by Nick with spyderX pro running DisplayCal on MacBook

'''


measured_lum = [0.01, 0.17,	0.48, 0.91, 1.55, 2.45, 3.58, 4.91, 6.49, 8.4, 10.37, 12.77, 13.03, 16.3, 19.61, 23.26, 24.78, 24.8]
rbg1_values = [0, 0.058823529, 0.117647059, 0.176470588, 0.235294118, 0.294117647, 0.352941176,
               0.411764706, 0.470588235, 0.529411765, 0.588235294, 0.647058824, 0.705882353,
               0.764705882, 0.823529412, 0.882352941, 0.941176471, 1]

# because of the kink in the data I'm just going to take the first 12 values
y_vals = measured_lum[:12]
x_vals = rbg1_values[:12]

# plot the measured values
plt.plot(x_vals, y_vals, 'o', label='measured values')
plt.show()

# what is the function that best fits these values?
# I'm going to start with a polynomial function
# I'm going to use numpy to find the coefficients of the polynomial
# I'm going to use the coefficients to plot the polynomial function
# I'm going to use the function to find the linearised values of the RGB values

x = x_vals
y = y_vals

# calculate polynomial
z = np.polyfit(x, y, 3)
f = np.poly1d(z)

# calculate new x's and y's
x_new = np.linspace(x[0], x[-1], 50)
y_new = f(x_new)

plt.plot(x,y,'o', x_new, y_new)
# plt.xlim([x[0]-1, x[-1] + 1 ])
plt.show()

# I want to be able to pass in a value of x and get out a value of y
# I can do this using the function f
# I can also do this using the coefficients of the polynomial
# I can also do this using the equation of the line
# I can also do this using the equation of the line of best fit
# old_x = 0.5
# new_y = f(old_x)
# print(f"new_y: {new_y}")

old_xs = [.07, .175, .28, .385, .49, .7, 1]
old_ys = [1.75, 4.375, 7, 9.625, 12.25, 17.5, 25]

new_ys = [f(i) for i in old_xs]
print(f"new_ys: {new_ys}")

new_y_vals = [f(i) for i in x_vals]
print(f"new_y_vals: {new_y_vals}")
plt.plot(x_vals, new_y_vals, 'o', label='adjusted values')
plt.show()

# # find the inverse of the function to go from y to x
# z_inv = np.polyfit(y, x, 3)
# f_inv = np.poly1d(z_inv)
#
#
# check_ys = [0, 2, 4, 6, 8, 10, 12]
# check_xs = [f_inv(i) for i in check_ys]
# print(f"check_xs: {check_xs}")
# plt.plot(check_ys, check_xs, 'o', label='inverse function')
# plt.show()