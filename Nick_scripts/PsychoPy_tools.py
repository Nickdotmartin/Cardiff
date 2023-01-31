from psychopy import monitors
from psychopy.tools import monitorunittools
import numpy as np
import math

'''
Page has tools to use for psychoPy experiments.

'''

def check_correct_monitor(monitor_name, actual_size, actual_fps, verbose=False):
    """
    Function to compare the expected frame refresh rate (fps) and
    monitor size with the actual refresh rate and monitor size.

    The actual size calculation may be double the true size for apple retina
    displays, so the comparison check for values that twice the expected size.
    The comparison of frame rate allows an error of +/- 5 fps.

    :param monitor_name: (str) name given to monitor in psychopy monitor centre.
    This function wil use monitor_name to load the monitor configuration from
    the monitor centre and get the expected size and fps.
    :param actual_size: (list) From win.size, gives the size in pixels [width, height].
    :param actual_fps: (np.float) Ude win.getActualFrameRate() to get the actual 
    frame rate as a np.float.
    :param verbose: (bool) If True, will print info to screen

    :return: Nothing - but raise error and quit if they don't match.
    """

    if verbose:
        print(f"\n*** running check_correct_monitor() ***\n"
              f"monitor_name={monitor_name}\n"
              f"actual_size={actual_size}\n"
              f"actual_fps={actual_fps}\n"
              f"verbose={verbose}")

    this_monitor = monitors.Monitor(monitor_name)
    monitor_dict = {'mon_name': monitor_name,
                    'size': this_monitor.getSizePix(),
                    'notes': this_monitor.getNotes()}
    if verbose:
        print(f"monitor_dict: {monitor_dict}")

    # MONITOR SIZE CHECK
    if list(monitor_dict['size']) == list(actual_size):
        print(f"monitor is expected size")
    elif list(monitor_dict['size']) == list(actual_size / 2):
        print(f"actual_size ({actual_size}) is double expected size ({monitor_dict['size']}).\n"
              f"This is to be expected for a mac retina display.")
    else:
        raise ValueError(f"actual_size ({actual_size}) does not match expected "
                         f"size ({monitor_dict['size']}).")

    # FRAME RATE CHECK
    if actual_fps is None:
        print('failed to calculate actual frame rate')
    else:
        actual_frame_rate = int(actual_fps)
        print(f'actual_frame_rate: {actual_frame_rate}')

        # search for refresh rate in notes
        if monitor_dict['notes'] is None:
            print('No monitor dict notes containing fps/Hz info.')
            # raise ValueError
        else:  # if monitor_dict['notes'] is not None:
            notes = monitor_dict['notes'].lower()

            # check for 'hz' in notes, and that it only occurs once
            if "hz" in notes:
                if notes.count('hz') == 1:
                    find_hz = notes.find("hz")

                    # if hz are last two characters, use -2 for the 2 letters 'hz'
                    if find_hz == -1:
                        find_hz = len(notes) - 2

                    slice = notes[:find_hz]  # slice text upto 'hz'

                    # remove any space between int and 'hz'
                    if slice[-1] == " ":
                        slice = slice[:-1]

                    # to find how many ints there are, look for white space
                    whitespaces = []
                    for index, character in enumerate(slice):
                        if character == " ":
                            whitespaces.append(index)

                    # if there are no whitespaces, just use this slice
                    if len(whitespaces) == 0:
                        convert_this = slice
                    else:
                        # slice from space before 'hz' to just leave numbers
                        convert_this = slice[whitespaces[-1]:]

                    converted = int(convert_this)
                    expected_fps = converted

                if verbose:
                    print(f"expected_fps: {expected_fps}")

                # check fps
                margin = 5
                if expected_fps in list(range(actual_frame_rate - margin, actual_frame_rate + margin)):
                    print("expected_fps matches actual frame rate")
                else:
                    raise ValueError(f"expected_fps ({expected_fps}) does not match actual "
                                     f"frame rate ({actual_frame_rate})")

            else:
                print('No fps/Hz info found in monitor dict notes.')

    if verbose:
        print("check_correct_monitor() complete")


# def get_pixel_mm_deg_values(monitor_name='ASUS_2_13_240Hz', n_pixels=1):
#     """
#     use psychopy's monitor tools to convert pixels to mm or degrees at a certain viewing distance.
#     Choose horizontal or diagonal pixels.
#     :param monitor_name: default='Asus_VG24', monitor in lab 2.13.
#             str - should match saved file in psychopy's monitor centre.
#     :param n_pixels: default = 1.
#     """
#
#     this_monitor = monitors.Monitor(monitor_name)
#     # print(f'this_monitor: {monitor_name}')
#     # print(f'n_pixels: {n_pixels}')
#
#     # This gets horizontal pixel size.
#     pix2cm = monitorunittools.pix2cm(pixels=n_pixels, monitor=this_monitor)
#     # print(f'\nHorizontal pixel size:\npix2cm: {pix2cm}')
#     width_mm = pix2cm * 10
#     # print(f'width_mm = pix2cm*10: {width_mm}')
#
#     pix2deg = monitorunittools.pix2deg(pixels=n_pixels, monitor=this_monitor)
#     # print(f'pix2deg: {pix2deg}')
#
#     # if pix measurements are horizontal then diag pix will be
#     # todo: this assumes that pixels are square, which they are not.
#     print('\nConverted to Diagonal pixel sizes')
#     diag_mm = width_mm * np.sqrt(2)
#     # print(f'diag_mm: {diag_mm}')
#
#     # get nnumber of widths that fit into diagonal.
#     len_of_diag_to_w = 1 * np.sqrt(2)
#     # print(f'len_of_diag_to_w: {len_of_diag_to_w}')
#     diag_deg = monitorunittools.pix2deg(pixels=n_pixels * len_of_diag_to_w,
#                                         monitor=this_monitor)
#     # print(f'diag_deg: {diag_deg}')
#
#     pixel_mm_deg_dict = {'this_monitor': monitor_name,
#                          'n_pixels': n_pixels,
#                          'horiz_mm': width_mm,
#                          'horiz_deg': pix2deg,
#                          'diag_mm': diag_mm,
#                          'diag_deg': diag_deg}
#
#     return pixel_mm_deg_dict


# todo: make disctionary with diagonal pixel sizes in mm/cm
# todo: write function for converting diaginal pixels distance into degrees.

monitor_pixel_size_dict = {
    'asus_cal': {'model_name': 'Asus_VG279VM',
                 'dimensions_mm': [567.6, 336.15],
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

    'Iiyama_2_18': {'model_name': '',  # todo: get model name
                'dimensions_mm': [],  # todo: get screen dimensions
                'dimensions_pix': [1920, 1080]},

}



def mm_to_degrees(pixel_size_mm, distance_mm):

    # convert size of pixel to visual angle in degrees
    dva = 2 * math.degrees(math.atan(pixel_size_mm / (2 * distance_mm)))

    return dva

def get_pixel_mm_deg_values(monitor_name='ASUS_2_13_240Hz', n_pixels=1):
    """
    use psychopy's monitor tools to convert pixels to mm or degrees at a certain viewing distance.
    Use monitor_pixel_size_dict for calculating diagonal pixel size as monitor centre only has width.
    My old version gave square pixels as the output.
    :param monitor_name: default='Asus_VG24', monitor in lab 2.13.
            str - should match saved file in psychopy's monitor centre.
    :param n_pixels: default = 1.
    """
    # print("\n***running get_pixel_mm_deg_values()***")

    this_monitor = monitors.Monitor(monitor_name)
    # print(f'monitor_name: {monitor_name}, (n_pixels: {n_pixels})')

    # check for details in monitor_pixel_size_dict
    dimensions_mm = monitor_pixel_size_dict[monitor_name]['dimensions_mm']
    dimensions_pix = monitor_pixel_size_dict[monitor_name]['dimensions_pix']

    # get pixel sizes in mm
    pix_width_mm = dimensions_mm[0] / dimensions_pix[0]
    pix_height_mm = dimensions_mm[1] / dimensions_pix[1]
    pix_diag_mm = np.sqrt(pix_width_mm ** 2 + pix_height_mm ** 2)
    # print(f"pix_width_mm: {pix_width_mm}\n"
    #       f"pix_width_mm ** 2: {pix_width_mm ** 2}\n"
    #       f"pix_height_mm: {pix_height_mm}\n"
    #       f"pix_height_mm ** 2: {pix_height_mm ** 2}\n"
    #       f"pix_width_mm ** 2 + pix_height_mm ** 2: {pix_width_mm ** 2 + pix_height_mm ** 2}\n"
    #       f"pix_diag_mm = np.sqrt(pix_width_mm ** 2 + pix_height_mm ** 2): {np.sqrt(pix_width_mm ** 2 + pix_height_mm ** 2)}"
    #       )


    # convert pixel chosen pixel size to dva
    viewdist_mm = this_monitor.getDistance() * 10  # view dist is stored in cm
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

    # print("***finished get_pixel_mm_deg_values()***\n")

    return pixel_mm_deg_dict