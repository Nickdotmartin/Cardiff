import os
import pathlib


def which_path(path):
    '''
    Function to let me know which machine a file is on.

    :param path: path to a file
    '''
    mac_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff'
    mac_oneD_path = '/Users/nickmartin/Library/CloudStorage/OneDrive-CardiffUniversity/PycharmProjects/Cardiff'
    wind_path = os.path.normpath(r'C:\Users\sapnm4\PycharmProjects\Cardiff')
    wind_oneD_path = os.path.normpath(r'C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff')

    if path[:len(mac_path)] == mac_path:
        print('this is a mac_path')
    elif path[:len(mac_oneD_path)] == mac_oneD_path:
        print('this is a mac_oneD_path')
    elif path[:len(wind_path)] == wind_path:
        print('this is a wind_path')
    elif path[:len(wind_oneD_path)] == wind_oneD_path:
        print('this is a wind_oneD_path')
    else:
        print('Unknown')


def switch_path(orig_path, change_to):
    '''
    Function to switch home directories as I move between my mac and work laptop.

    :param orig_path: Original path
    :param change_to: What I want the new prefix to be (e.g., 'mac', 'mac_oneDrive', 'windows', 'windows_oneDrive')

    :return: New path with updated prefix.
    '''

    # root paths to use
    mac_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff'
    mac_oneD_path = '/Users/nickmartin/Library/CloudStorage/OneDrive-CardiffUniversity/PycharmProjects/Cardiff'
    wind_path = os.path.normpath(r'C:\Users\sapnm4\PycharmProjects\Cardiff')
    wind_oneD_path = os.path.normpath(r'C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff')

    # Get old prefix to change
    if orig_path[:len(mac_path)] == mac_path:
        old_prefix = mac_path
    elif orig_path[:len(mac_oneD_path)] == mac_oneD_path:
        old_prefix = mac_oneD_path
    elif orig_path[:len(wind_path)] == wind_path:
        old_prefix = wind_path
    elif orig_path[:len(wind_oneD_path)] == wind_oneD_path:
        old_prefix = wind_oneD_path
    else:
        raise TypeError(f'orig_path not recognised: {orig_path}')
    print(f"old_prefix: {old_prefix}")

    # keep suffix, and set as OLD filepath type
    characters_to_snip = len(old_prefix) + 1
    suffix_to_keep = orig_path[characters_to_snip:]
    if old_prefix in [mac_path, mac_oneD_path]:
        suffix_to_keep = pathlib.PurePosixPath(suffix_to_keep)
    else:
        suffix_to_keep = pathlib.PureWindowsPath(suffix_to_keep)
    print(f"suffix_to_keep: {suffix_to_keep}")


    # orig_path to change_to
    if change_to.lower() in ['mac', 'mac_path']:
        new_prefix = pathlib.PurePosixPath(mac_path)
        print(f"new_prefix ({change_to.lower()}): {new_prefix}")
        suffix_to_keep = pathlib.Path(suffix_to_keep).as_posix()
        print(f"suffix_to_keep: {suffix_to_keep}")
        # suffix_to_keep = pathlib.PurePosixPath(suffix_to_keep)
        # print(f"suffix_to_keep: {suffix_to_keep}")
        join_paths = f'{new_prefix}/{suffix_to_keep}'
        new_path = pathlib.PurePosixPath(join_paths)

    elif change_to.lower() in ['mac_oned', 'mac_oned_path', 'mac_onedrive', 'mac_one_drive', 'mac_one_d']:
        new_prefix = pathlib.PurePosixPath(mac_oneD_path)
        print(f"new_prefix ({change_to.lower()}): {new_prefix}")
        suffix_to_keep = pathlib.Path(suffix_to_keep).as_posix()
        join_paths = f'{new_prefix}/{suffix_to_keep}'
        new_path = pathlib.PurePosixPath(join_paths)

    elif change_to.lower() in ['wind_path', 'wind', 'windows', 'windows_path', 'win']:
        new_prefix = wind_path
        print(f"new_prefix ({change_to.lower()}): {new_prefix}")
        join_paths = f'{new_prefix}\{suffix_to_keep}'
        new_path = pathlib.PureWindowsPath(join_paths)

    elif change_to.lower() in ['wind_oned', 'wind_oned_path', 'wind_onedrive', 'wind_one_drive', 'wind_one_d',
                               'win_oned', 'win_oned_path', 'win_onedrive', 'win_one_drive', 'win_one_d']:
        new_prefix = wind_oneD_path
        print(f"new_prefix ({change_to.lower()}): {new_prefix}")
        join_paths = f'{new_prefix}\{suffix_to_keep}'
        new_path = pathlib.PureWindowsPath(join_paths)
    else:
        raise TypeError(f'change_to not recognised: {change_to}')

    return new_path