import sys
import os

def running_on_laptop(verbose=True):
    """
    Check if I am on my laptop (not my work machine), might need to change paths
    :param verbose:
    :return:
    """
    # if verbose:
    # print("checking for laptop")
    if sys.executable[:18] == '/Users/nickmartin/':
        if verbose:
            print("Script is running on Nick's laptop")
    else:
        if verbose:
            print("Script is not running on Nick's laptop")
    return sys.executable[:18] == '/Users/nickmartin/'


def switch_home_dirs(change_from, change_to):
    # todo: add 'change_to' variable with option of mac_path, mac_oneDrive_path, windows_oneDrive_path
    """
    Try this module anytime I am having a problem on my laptop with uni paths.

    :param change_from:
    :return: new path: to try out
    """

    mac_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff'

    mac_oneDrive_path = '/Users/nickmartin/Library/CloudStorage/OneDrive-CardiffUniversity/PycharmProjects/Cardiff'
    
    windows_oneDrive_path = ''

    # iCloud_path = '/Users/nickmartin/Library/Mobile Documents/com~apple~CloudDocs/Documents/PhD/python_v2/'
    # GPU_path = '/home/nm13850/Documents/PhD/python_v2/'

    if mac_path == change_from[:len(mac_path)]:
        # print('old path is laptop hd')
        snip_end = change_from[len(mac_path):]
        new_path = os.path.join(change_to, snip_end)

    if mac_oneDrive_path == change_from[:len(mac_oneDrive_path)]:
        # print('old path is icloud')
        snip_end = change_from[len(mac_oneDrive_path):]
        new_path = os.path.join(change_to, snip_end)

    elif windows_oneDrive_path == change_from[:len(windows_oneDrive_path)]:
        # print('old path is gpu')
        snip_end = change_from[len(windows_oneDrive_path):]
        new_path = os.path.join(change_to, snip_end)

    else:
        print(f"path not found in laptop or GPU paths\n{change_from}")
        raise ValueError

    print(f'changed from {change_from} to:\n{new_path}')

    return new_path
