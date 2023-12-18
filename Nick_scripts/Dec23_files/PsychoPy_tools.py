from psychopy import monitors
import numpy as np
import math

'''
Page has tools to use for psychoPy experiments.

'''

monitor_pixel_size_dict = {
    'asus_cal': {'model_name': 'Asus_VG279VM',
                 'dimensions_mm': [567.6, 336.15],
                 'dimensions_pix': [1920, 1080]},
    'ASUS_2_13_240Hz': {'model_name': 'Asus_VG279VM',
                 'dimensions_mm': [567.6, 336.15],
                 'dimensions_pix': [1920, 1080]},
    'Samsung': {'model_name': 'Samsung_F24G35TFWU',
                 'dimensions_mm': [527.04, 296.46],
                 'dimensions_pix': [1920, 1080]},
    'OLED': {'model_name': 'Dell_AW3423DW',
                 'dimensions_mm': [797.22, 333.72],
                 'dimensions_pix': [3440, 1440]},
    'Nick_work_laptop': {'model_name': 'Dell_latitude_5400',
                 'dimensions_mm': [309.4, 173.95],
                 'dimensions_pix': [1920, 1080]},
    'Asus_VG24': {'model_name': 'Asus_VG248QE',
                 'dimensions_mm': [531.36, 298.89],
                 'dimensions_pix': [1920, 1080]},
    'HP_24uh': {'model_name': 'HP_24uh',
                'dimensions_mm': [531.36, 298.89],
                'dimensions_pix': [1920, 1080]},
    'NickMac': {'model_name': 'Macbook_air_(retina)_13"_2018',
                'dimensions_mm': [304.1, 191.9],
                'dimensions_pix': [2560, 1600]},
}


def mm_to_degrees(pixel_size_mm, distance_mm):

    # convert size of pixel to visual angle in degrees
    dva = 2 * math.degrees(math.atan(pixel_size_mm / (2 * distance_mm)))

    return dva

def get_pixel_mm_deg_values(monitor_name='asus_cal', n_pixels=1):
    """
    use psychopy's monitor tools to convert pixels to mm or degrees at a certain viewing distance.
    Use monitor_pixel_size_dict for calculating diagonal pixel size as monitor centre only has width.
    My old version gave square pixels as the output.
    :param monitor_name: default='Asus_VG24', monitor in lab 2.13.
            str - should match saved file in psychopy's monitor centre.
    :param n_pixels: default = 1.
    """

    this_monitor = monitors.Monitor(monitor_name)
    print(f'monitor_name: {monitor_name}, (n_pixels: {n_pixels})')

    # check for details in monitor_pixel_size_dict
    dimensions_mm = monitor_pixel_size_dict[monitor_name]['dimensions_mm']
    dimensions_pix = monitor_pixel_size_dict[monitor_name]['dimensions_pix']

    # get pixel sizes in mm
    pix_width_mm = dimensions_mm[0] / dimensions_pix[0]
    pix_height_mm = dimensions_mm[1] / dimensions_pix[1]
    pix_diag_mm = np.sqrt(pix_width_mm ** 2 + pix_height_mm ** 2)


    # convert pixel chosen pixel size to dva
    viewdist_cm = this_monitor.getDistance()  # view dist is stored in cm
    if viewdist_cm is not None:
        viewdist_mm = this_monitor.getDistance() * 10  # view dist is stored in cm
    else:    # if viewdist_mm is None:
        viewdist_mm = 437.5  # default value for OLED monitor
    pix_width_deg = mm_to_degrees(pixel_size_mm=pix_width_mm, distance_mm=viewdist_mm)
    pix_height_deg = mm_to_degrees(pixel_size_mm=pix_height_mm, distance_mm=viewdist_mm)
    pix_diag_deg = mm_to_degrees(pixel_size_mm=pix_diag_mm, distance_mm=viewdist_mm)

    pixel_mm_deg_dict = {'this_monitor': monitor_name,
                         'n_pixels': n_pixels,
                         'width_mm': pix_width_mm,
                         'height_mm': pix_height_mm,
                         'diag_mm': pix_diag_mm,
                         'width_deg': pix_width_deg,
                         'heigth_deg': pix_height_deg,
                         'diag_deg': pix_diag_deg}

    return pixel_mm_deg_dict

