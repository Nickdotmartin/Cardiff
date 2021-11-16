from psychopy import monitors, core

'''
Page has tools to use for psychoPy experiments.

'''

def check_correct_monitor(monitor_name, actual_size, actual_fps, verbose=False):
    """
    Function to compare the expected frame refresh rate (fps) and
    monitor size with the actual refresh rate and monitor size.

    The actual size calculation may be double the true size for apple retina
    displays, so the comparison check for values that twice the expected size.
    The comparison of frame rate allows an error of +/- 2 fps.

    :param monitor_name: (str) name given to monitor in psychopy monitor centre.
    This function wil use monitor_name to load the monitor configuration from
    the monitor centre and get the expect size and fps.
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
    actual_frame_rate = int(actual_fps)
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
    # search for refresh rate in notes
    if monitor_dict['notes'] is None:
        print('No fps info found in monitor dict notes.')
        raise ValueError
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
    if expected_fps in list(range(actual_frame_rate-2, actual_frame_rate+2)):
        print("expected_fps matches actual frame rate")
    else:
        raise ValueError(f"expected_fps ({expected_fps}) does not match actual "
                         f"frame rate ({actual_frame_rate})")

    if verbose:
        print("check_correct_monitor() complete")

