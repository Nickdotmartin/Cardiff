import psychopy
from psychopy import monitors, info, core, visual
# from psychopy import gui, visual, data, event, logging, monitors

'''

Page has a set of tools to use for psychoPy experiments

'''


# def select_monitor(mon_name='NickMac'):
#     """
#     Function to adjust the refresh rate and size for the relevant monitor.
#     Input the monitor name
#
#
#     :return: a dict with relevant values (size, refresh etc)
#     """
#
#     mon_dict = {'mon_name': 'NickMac',
#                 'width': 30.41,
#                 'size': [1440, 900],
#                 'dist': 60.0,
#                 'notes': 'MacBook Air 13inch, 60Hz'}
#
#     if mon_name == 'Asus_VG24':
#         mon_dict = {'mon_name': 'Asus_VG24',
#                     'width': 53.13,
#                     'size': [1920.0, 1080.0],
#                     'dist': 75.0,
#                     'notes': 'Asus monitor in room 2.18, 144 Hz refresh'}
#
#     elif mon_name == 'asus_cal':
#         mon_dict = {'mon_name': 'asus_cal',
#                     'width': 32.512,
#                     'size': [1280, 800],
#                     'dist': 57.3,
#                     'notes': None}
#
#     elif mon_name == 'testMontior':
#         mon_dict = {'mon_name': 'testMonitor',
#                     'width': 30,
#                     'size': [1024, 768],
#                     'dist': 57,
#                     'notes': 'default (not very useful) monitor'}
#
#
#
#     # # add in stuff for monitor in 2.13d
#     # elif mon_name == 'fancy 2.13d':
#     #     mon_dict['refresh'] = 200
#
#     return mon_dict
#
#
# print(select_monitor('Asus_VG24'))
#
# runtime_info = info.RunTimeInfo(refreshTest=False)
# print(runtime_info)
#
#
#
# names = monitors.getAllMonitors()
# for thisName in names:
#     thisMon = monitors.Monitor(thisName)
#     this_width = thisMon.getWidth()
#     mon_dict = {'mon_name': thisName,
#                 'width': thisMon.getWidth(),
#                 'size': thisMon.getSizePix(),
#                 'dist': thisMon.getDistance(),
#                 'notes': thisMon.getNotes()
#                 }
#
#     print(thisName)
#     # # search for refresh rate in notes
#     if mon_dict['notes'] is not None:
#         notes = mon_dict['notes'].lower()
#
#         # check for 'hz', and that it only occurs once
#         if "hz" in notes:
#             if notes.count('hz') == 1:
#                 find_hz = notes.find("hz")
#
#                 # if hz are last two characters, don't use -1
#                 if find_hz == -1:
#                     # use -2 because there are two characters in 'hz'
#                     find_hz = len(notes) - 2
#
#                 # strip away text from 'hz' onwards
#                 slice = notes[:find_hz]
#
#                 # remove space between int and 'hz' if present
#                 if slice[-1] == " ":
#                     slice = slice[:-1]
#
#                 # to find how many ints there are, look for white space
#                 whitespaces = []
#                 for index, character in enumerate(slice):
#                     if character == " ":
#                         whitespaces.append(index)
#
#                 # if there are no whitespaces, just use this slice
#                 if len(whitespaces) == 0:
#                     convert_this = slice
#                 else:
#                     # slice from space before 'hz' to just leave numbers
#                     last_space = whitespaces[-1]
#                     convert_this = slice[last_space:]
#                 converted = int(convert_this)
#
#                 mon_dict['refresh'] = converted
#
#         #     else:
#         #         print('There are multiple refeences to Hz in notes.  '
#         #               'This script only works if there is a single reference.')
#         # else:
#         #     print("refresh rate info ('Hz') not found")
#
#                 print(mon_dict)

# # MONITOR SPEC
#
# monitor_name = 'HP_24uh'  # 'NickMac' 'asus_cal' 'Asus_VG24' 'HP_24uh'
# display_number = 1  # 0 indexed, 1 for external display
#
# thisMon = monitors.Monitor(monitor_name)
# this_width = thisMon.getWidth()
# mon_dict = {'mon_name': monitor_name,
#             'width': thisMon.getWidth(),
#             'size': thisMon.getSizePix(),
#             'dist': thisMon.getDistance(),
#             'notes': thisMon.getNotes()
#             }
# print(f"mon_dict: {mon_dict}")
#
# widthPix = mon_dict['size'][0]  # 1440  # 1280
# heightPix = mon_dict['size'][1]  # 900  # 800
# monitorwidth = mon_dict['width']  # 30.41  # 32.512  # monitor width in cm
# viewdist = mon_dict['dist']  # 57.3  # viewing distance in cm
# viewdistPix = widthPix/monitorwidth*viewdist
# mon = monitors.Monitor(monitor_name, width=monitorwidth, distance=viewdist)
# mon.setSizePix((widthPix, heightPix))
# mon.save()
#
#
# # WINDOW SPEC
# win = visual.Window(monitor=mon, size=(widthPix, heightPix),
#                     # colorSpace='rgb255', color=bgColor255,
#                     winType='pyglet',  # I've added this to make it work on pycharm/mac
#                     pos=[1, -1],  # pos gives position of top-left of screen
#                     units='pix',
#                     screen=display_number,
#                     allowGUI=False,
#                     fullscr=None
#                     )
#
# this_mon = monitors.Monitor()
# print(this_mon)

def check_correct_monitor(monitor_name, actual_size, actual_fps, verbose=False):
    """
    Function to compare the expected frame refresh rate (fps) and
    monitor size with the actual refresh rate and monitor size.

    Note: The actual size calculation may be double the true size for apple retina
    displays, so the comparisson check for values that twice the expected size.

    The comparisson of frame rate allows an error of +/- 2 fps.

    :param monitor_name (str): name given to monitor in psychopy monitor centre.
    This function wil use monitor_name to load the monitor configuration from
    the monitor centre and get the expect size and fps.
    :param actual_size (list): gives the size in pixels as a list [width, height]
    :param actual_fps (np.float): gives the actual frame rate as a np.float.
    :param verbose (bool): If True, will print info to screen

    :return: Nothing - but quit or raise error if they don't match
    """

    if verbose:
        print(f"\n*** running check_correct_monitor() ***\n"
              f"monitor_name={monitor_name}\n"
              f"actual_size={actual_size}\n"
              f"actual_fps={actual_fps}\n"
              f"verbose={verbose}")

    thisMon = monitors.Monitor(monitor_name)

    actualFrameRate = int(actual_fps)

    mon_dict = {'mon_name': monitor_name,
                'size': thisMon.getSizePix(),
                'notes': thisMon.getNotes()
                }
    if verbose:
        print(f"mon_dict: {mon_dict}")

    # check size
    if monitor_name == 'NickMac':
        print("I've not sorted screen size for air retina display")
    else:
        if list(mon_dict['size']) == list(actual_size):
            print(f"monitor is expected size")
        elif list(mon_dict['size']) == list(actual_size / 2):
            print(f"actual size is double expected size - Its ok, just a mac retina display bug.")
        else:
            print(f"Display size does not match expected size from montior centre")
            core.quit()

    # search for refresh rate in notes
    if mon_dict['notes'] is None:
        print('No fps info found in monitor dict notes.')
        raise ValueError
    else:
        # if mon_dict['notes'] is not None:
        notes = mon_dict['notes'].lower()

        # check for 'hz', and that it only occurs once
        if "hz" in notes:
            if notes.count('hz') == 1:
                find_hz = notes.find("hz")

                # if hz are last two characters, don't use -1
                if find_hz == -1:
                    # use -2 because there are two characters in 'hz'
                    find_hz = len(notes) - 2

                # strip away text from 'hz' onwards
                slice = notes[:find_hz]

                # remove space between int and 'hz' if present
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
                    last_space = whitespaces[-1]
                    convert_this = slice[last_space:]
                converted = int(convert_this)

                mon_dict['refresh'] = converted
                expected_fps = converted
        if verbose:
            print(f"expected_fps: {mon_dict['refresh']}")

    # check fps
    if expected_fps in list(range(actualFrameRate-2, actualFrameRate+2)):
        print("expected_fps matches actual frame rate")
    else:
        # if values don't match, quit experiment
        print(f"expected_fps ({expected_fps}) does not match actual frame rate ({actualFrameRate})")
        core.quit()

    if verbose:
        print("check_correct_monitor() complete")


# # # # #
# monitor_name = 'HP_24uh'  # 'NickMac' 'asus_cal' 'Asus_VG24' 'HP_24uh'
# display_number = 1  # 0 indexed, 1 for external display
# check_correct_monitor(monitor_name='HP_24uh',
#                       actual_size, actual_fps, verbose=False)