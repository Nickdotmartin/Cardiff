

'''
Page has a set of tools to use for psychoPy experiments

'''

def select_monitor(mon_name='NickMac'):
    '''
    Function to adjust the refresh rate and size for the relevant monitor.
    Input the monitor name


    :return: a dict with relevant values (size, refresh etc)
    '''

    mon_dict = {'mon_name': mon_name,
                'refresh': 60,
                'dims': [2560, 1600],
                'width_cm': 30.41}
    if mon_name == 'Asus_VG24':
        mon_dict['refresh'] = 144
        mon_dict['dims'] = [1920, 1080]
        mon_dict['width_cm'] = 53.13

    # # add in stuff for monitor in 2.13d
    # elif mon_name == 'fancy 2.13d':
    #     mon_dict['refresh'] = 200

    return mon_dict

print(select_monitor('Asus_VG24'))

