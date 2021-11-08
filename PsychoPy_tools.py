import psychopy
from psychopy import monitors, info
'''

Page has a set of tools to use for psychoPy experiments

'''

print(psychopy.__version__)

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


# print(select_monitor('Asus_VG24'))

# runtime_info = info.RunTimeInfo(refreshTest=False)
# print(runtime_info)

this_mon = monitors.Monitor()
print(this_mon)

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



