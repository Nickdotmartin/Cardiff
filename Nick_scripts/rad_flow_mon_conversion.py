import numpy as np


'''
These functions are to be used to take the rgb255 values from radial flow experiments that were run on the uncalibrated monitor (asus_2_13_240Hz), 
and use the luminance profiles measured with the spyder to find the corresponding rgb255 values for the calibrated monitor (asus_cal).
'''


def return_target_or_above_below(array_of_values, target_value):
    """
    Returns the target value if it is in the array, otherwise,
    returns the array values above and below the target (to use for interpolation).

    array_of_values: An array of values to search.
    target_value: The target value.
        """

    if type(array_of_values) is list:
        array_of_values = np.array(array_of_values)

    '''
    ValueError: zero-size array to reduction operation minimum which has no identity
    This error occurs when the array is empty, 
    e.g., the target value is less than the minimum value, or more than the maximum value in the array.
    '''

    # Check if the target value is in the array.
    if target_value in array_of_values:
        return target_value

    else:

        # if target not in array, return the value below and above.
        item_below = np.amax(array_of_values[array_of_values < target_value])
        item_above = np.amin(array_of_values[array_of_values > target_value])

        # Return a tuple of the array values above and below the target.
        return item_below, item_above




def rad_flow_mon_conversion(uncalibrated_rgb255_val, verbose=False):
    """
    this is a function that uses a nested-dictionary containing two sets of luminance values
    (one for each of two monitor settings) at each of 18 evenly spaced rgb255 values.
    The function will take an rgb255 value from the uncalibrated monitor and find the corresponding luminance value.
    It will then use that luminance value to find the corresponding rgb255 value for the calibrated monitor (asus_cal).

    :param uncalibrated_rgb255_val: The rgb255 value from the uncalibrated monitor.

    :return: The equivalent rgb255 value for the calibrated monitor (asus_cal).
    """

    if verbose: print(f'\nuncalibrated_rgb255_val: {uncalibrated_rgb255_val}')


    spyder_values_dicts = {'uncalibrated': {0: 0.13, 15: 0.46, 30: 1.52, 45: 3.59, 60: 6.69, 75: 10.91, 90: 16.18,
                                            105: 22.66, 120: 30.29, 135: 39.22, 150: 49.27, 165: 60.44, 180: 72.73,
                                            195: 86.43, 210: 101.26, 225: 117.09, 240: 134.09, 255: 151.05},
                           'asus_cal': {0: 0.14, 15: 8.05, 30: 16.66, 45: 25.58, 60: 34.6, 75: 43.44, 90: 52.51,
                                        105: 61.69, 120: 70.94, 135: 79.81, 150: 89.37, 165: 98.38, 180: 107.48,
                                        195: 116.36, 210: 125.33, 225: 134.43, 240: 142.94, 255: 150.63}
                           }

    array_of_rgb255_vals = np.array(list(spyder_values_dicts['uncalibrated'].keys()))
    if verbose:
        print(f'array_of_rgb255_vals: {array_of_rgb255_vals}')

    '''
    Part one, get the uncalibrated luminance from the uncalibrated RGB255 value
    '''

    # check if the target value is in the array, otherwise, return the array values above and below the target.
    vals_for_uncali_interp = return_target_or_above_below(array_of_rgb255_vals, uncalibrated_rgb255_val)

    if isinstance(vals_for_uncali_interp, int):
        # if the uncalibrated_rgb255_val is in the array, use that for conversion
        uncalibrated_lum_val = spyder_values_dicts['uncalibrated'][vals_for_uncali_interp]
    else:
        # interpolate between the array values below and above the uncalibrated_rgb255_val to find the 
        # luminance value for the uncalibrated monitor that corresponds to the uncalibrated rgb255 value

        if verbose:
            print(f"use {uncalibrated_rgb255_val} to interpolate between {vals_for_uncali_interp} to get corresponding "
              f"interpolation between {[spyder_values_dicts['uncalibrated'][vals_for_uncali_interp[0]], spyder_values_dicts['uncalibrated'][vals_for_uncali_interp[1]]]}")
        uncalibrated_lum_val = np.interp(uncalibrated_rgb255_val, vals_for_uncali_interp, [spyder_values_dicts['uncalibrated'][vals_for_uncali_interp[0]], spyder_values_dicts['uncalibrated'][vals_for_uncali_interp[1]]])
    if verbose:
        print(f'uncalibrated_lum_val: {uncalibrated_lum_val}')

    '''
    Part 2: get the calibrated rgb255 value from the uncalibrated luminance value'''
    
    # now switch, such that the array is the calibrated luminances and we want to find the corresponding rgb255 value
    array_of_cal_lum_vals = np.array(list(spyder_values_dicts['asus_cal'].values()))
    if verbose:
        print(f'\narray_of_cal_lum_vals: {array_of_cal_lum_vals}')

    vals_for_asus_cal_interp = return_target_or_above_below(array_of_cal_lum_vals, uncalibrated_lum_val)

    if isinstance(vals_for_asus_cal_interp, int):
        # if the uncalibrated_rgb255_val is in the array, use that for conversion
        calibrated_rgb255_val = list(spyder_values_dicts['asus_cal'].keys())[list(spyder_values_dicts['asus_cal'].values()).index(vals_for_asus_cal_interp)][0]
    else:
        # interpolate between the array values below and above the uncalibrated_rgb255_val to find the
        # luminance value for the uncalibrated monitor that corresponds to the uncalibrated rgb255 value

        asus_cal_key_below = list(spyder_values_dicts['asus_cal'].keys())[list(spyder_values_dicts['asus_cal'].values()).index(vals_for_asus_cal_interp[0])]
        asus_cal_key_above = list(spyder_values_dicts['asus_cal'].keys())[list(spyder_values_dicts['asus_cal'].values()).index(vals_for_asus_cal_interp[1])]

        if verbose:
            print(f"use {uncalibrated_lum_val} to interpolate between {vals_for_asus_cal_interp} to get corresponding "
              f"interpolation between {asus_cal_key_below, asus_cal_key_above}")

        calibrated_rgb255_val = int(np.interp(uncalibrated_lum_val, vals_for_asus_cal_interp, [asus_cal_key_below, asus_cal_key_above]))
        if verbose:
            print(f'calibrated_rgb255_val: {calibrated_rgb255_val}\n')

    return calibrated_rgb255_val

# for i in [130, 145, 209, 250]:

for i in [1, 14, 15, 17, 46, 62.367, 100]:
# for i in [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255]:

    print(f'{i}: {rad_flow_mon_conversion(i, verbose=True)}')

