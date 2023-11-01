import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import itertools
from psignifit_tools import psignifit_thr_df_Oct23
from rad_flow_psignifit_analysis import make_long_df, get_n_rows_n_cols
from kestenSTmaxVal import Staircase


data_list = []

target_value = 7

maxLum = 106
bgLum = maxLum * 0.05
print(f"maxLum = {maxLum}, bgLum = {bgLum}")

monitor_name = 'OLED'
n_stairs = 1
n_trials_per_stair = 25
stairStart = maxLum  # start luminance value
OLED_start_prop = 0.15  # proportion of maxLum to start at for OLED

save_plots = True
if save_plots:
    save_dir = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\project_stuff\Kesten_plots"
    # # cond_dir = f"type_{k_type}_stairStart_{stairStart}_c_multiplier_{c_multiplier}"
    # save_dir = root_dir  # os.path.join(root_dir, cond_dir)

k_type = 'simple'  # accel or simple
target_thr = .75
c_multiplier = .6

for target_value in [5.5, 6, 7, 8]:
    for OLED_start_prop in [.2]:  # [0.1, 0.15, 0.2, .25, .3, .35, .4]:
        for c_multiplier in [.6, 1.2]:  # [.4, .6, .8, 1, 1.2, 1.4, 1.6, 1.8]:
            for k_type in ['simple', 'accel']:
                print(f"\n\nRunning: target_value = {target_value}, OLED_start_prop = {OLED_start_prop}, c_multiplier = {c_multiplier}, k_type = {k_type}\n")


                if monitor_name == 'OLED':  # dimmer on OLED
                    stairStart = maxLum * OLED_start_prop




                # set up stairs
                stairs = []
                for stair_idx in range(n_stairs):
                    thisStair = Staircase(name='test',  # stair_names_list[stair_idx],
                                          # type='simple',  # step size changes after each reversal only
                                          type=k_type,  # step size changes after each trial for first 2, then reversal only
                                          value=stairStart,
                                          C=stairStart * c_multiplier,  # initial step size, as prop of maxLum
                                          minRevs=3,
                                          minTrials=n_trials_per_stair,
                                          minVal=bgLum,
                                          maxVal=maxLum,
                                          targetThresh=target_thr,
                                          # extraInfo=thisInfo
                                          )
                    stairs.append(thisStair)

                # run stairs demo
                # print(f"C = stairStart * c_multiplier = {stairStart} * {c_multiplier} = {stairStart * c_multiplier}\n")
                probe_lum_list = []
                resp_list = []
                formula_list = []
                formula_prop_list = []

                for step in range(n_trials_per_stair):
                    np.random.shuffle(stairs)  # shuffle order for each step (e.g., shuffle, run all stairs, shuffle again etc)
                    for thisStair in stairs:
                        probeLum = thisStair.next()
                        probe_lum_list.append(probeLum)
                        # print(f'\n{step}. probeLum = ', probeLum)

                        # enter either 0 or 1 for correct or incorrect response
                        # resp_corr = input('Enter 0 or 1 for correct or incorrect response: ')
                        # resp_corr = int(resp_corr)
                        # if resp_corr not in [0, 1]:
                        #     raise ValueError('resp_corr must be 0 or 1')
                        if probeLum > target_value:
                            resp_corr = 1
                        else:
                            resp_corr = 0
                        resp_list.append(resp_corr)

                        n_reversals = thisStair.countReversals()
                        # print(f"n_reversals = {n_reversals}, Response = {resp_corr}, ")


                        '''formula = (self.C*(self.resp - self.targetThresh) / (1 + self.countReversals()))
                        new values  = old values - formula'''

                        # print(f"formula = self.C*(self.resp - self.targetThresh) / (1 + self.countReversals())\n"
                        #       f"formula = {thisStair.C}*({resp_corr} - .75) / (1 + {thisStair.countReversals()})\n"
                        #       f"formula = {thisStair.C}*({resp_corr - .75}) / {1 + thisStair.countReversals()}\n"
                        #       f"formula = {thisStair.C*(resp_corr - .75)} / {1 + thisStair.countReversals()}\n"
                        #       f"formula = {(thisStair.C*(resp_corr - .75) / (1 + thisStair.countReversals()))}"
                        #       )
                        formula = (thisStair.C*(resp_corr - .75) / (1 + thisStair.countReversals()))
                        formula_list.append(formula)
                        formula_prop = formula / thisStair.value
                        formula_prop_list.append(formula_prop)
                        # print(f"formula_prop = formula / thisStair.value\n"
                        #       f"formula_prop = {formula} / {thisStair.value}\n"
                        #       f"formula_prop = {formula_prop}\n")
                        #
                        # print(f"newValue = oldValue - formula\n"
                        #       f"newValue = {thisStair.value} - "
                        #       f"{(thisStair.C*(resp_corr - .75) / (1 + thisStair.countReversals()))}\n"
                        #       f"newValue = {thisStair.value - (thisStair.C*(resp_corr - .75) / (1 + thisStair.countReversals()))}")

                        # update staircase based on whether response was correct or incorrect
                        thisStair.newValue(resp_corr)

                n_reversals = thisStair.countReversals()
                n_incorrect = len(resp_list) - sum(resp_list)
                last_3_mean = round(np.mean(probe_lum_list[-3:]), 2)
                target_last_3_diff = round(abs(target_value - last_3_mean), 2)
                print(f"\nn_reversals = {n_reversals}, n_incorrect = {n_incorrect}")

                # set up subplots
                fix, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), sharex=True, sharey=False)

                # plot probeLum_list
                # plt.plot(probe_lum_list)
                axes[0, 0].plot(probe_lum_list)
                # add points for each value
                for x, y in enumerate(probe_lum_list):
                    if resp_list[x] == 1:
                        axes[0, 0].scatter(x, y, color='green')
                    else:
                        axes[0, 0].scatter(x, y, color='red')
                # add horizontal line for bgLum
                axes[0, 0].axhline(y=bgLum, color='black', linestyle='--')
                # add green horizontal line for target value
                axes[0, 0].axhline(y=target_value, color='green', linestyle='--')
                # x axis label is step number, y axis label is probeLum
                # axes[0, 0].set_xlabel('trial number (for this staircase)')
                axes[0, 0].set_ylabel('Probe luminance')
                axes[0, 0].set_title(f'Kesten Values (probeLum)')

                # add text showing number of reversals
                axes[0, 0].text(x=0.30, y=0.8, s=f'{n_reversals} reversals\n'
                                                 f'last_3_mean = {last_3_mean}\n'
                                                 f'target_last_3_diff = {target_last_3_diff}',
                        # color='tab:blue',
                        # needs transform to appear with rest of plot.
                        transform=axes[0, 0].transAxes, fontsize=16)

                # plt.show()
                # print('probe_lum_list = ', probe_lum_list)

                # plot formula_list
                axes[0, 1].plot(formula_list)
                # add points for each value
                for x, y in enumerate(formula_list):
                    if resp_list[x] == 1:
                        axes[0, 1].scatter(x, y, color='green')
                    else:
                        axes[0, 1].scatter(x, y, color='red')
                axes[0, 1].axhline(y=0, color='black', linestyle='--')
                # axes[0, 1].set_xlabel('trial number (for this staircase)')
                axes[0, 1].set_ylabel('Probe luminance change (formula)')
                axes[0, 1].set_title(f'Formula (change size in exp units, e.g., luminance)')

                # show each value as a proportion of the previous value
                New_value_prop_list = []
                for idx, value in enumerate(probe_lum_list):
                    if idx == 0:
                        New_value_prop_list.append(0)
                    else:
                        New_value_prop_list.append(value / probe_lum_list[idx - 1])
                # print('New_value_prop_list = ', New_value_prop_list)
                # plot New_value_prop_list
                axes[1, 0].plot(New_value_prop_list)
                # add points for each value
                for x, y in enumerate(New_value_prop_list):
                    if resp_list[x] == 1:
                        axes[1, 0].scatter(x, y, color='green')
                    else:
                        axes[1, 0].scatter(x, y, color='red')
                axes[1, 0].axhline(y=0, color='black', linestyle='--')


                axes[1, 0].set_xlabel('trial number (for this staircase)')
                axes[1, 0].set_ylabel('New value as proportion of previous value')
                axes[1, 0].set_title(f'New Value as proportion of previous value')
                # plt.show()


                # plot formula_prop_list
                axes[1, 1].plot(formula_prop_list)
                # add points for each value
                for x, y in enumerate(formula_prop_list):
                    if resp_list[x] == 1:
                        axes[1, 1].scatter(x, y, color='green')
                    else:
                        axes[1, 1].scatter(x, y, color='red')
                axes[1, 1].axhline(y=0, color='black', linestyle='--')
                axes[1, 1].set_xlabel('trial number (for this staircase)')
                axes[1, 1].set_ylabel('Formula (as proportion of previous value)')
                axes[1, 1].set_title(f'Formula (change size) as proportion of previous value')


                plt.suptitle(f'\ntype: {k_type}, maxLum: {maxLum}, start_prop: {OLED_start_prop}, stairStart: {round(maxLum*OLED_start_prop, 2)}, c_multiplier: {c_multiplier}\n'
                             f'C = c_multiplier * stairStart = {c_multiplier} * {round(stairStart, 2)} = {c_multiplier * stairStart}\n'
                             f'step size = formula = C*(response - targetThresh) / (1 + n_reversals')

                if save_plots:
                    # save_name = f"Kesten_plots_{k_type}_stairStart_{round(stairStart, 2)}_c_multiplier_{c_multiplier}.png"
                    save_name = (f"Kesten_plots_startProp_{OLED_start_prop}_c_multiplier_{c_multiplier}_trgt_{target_value}_"
                                 f"{k_type}.png")
                    plt.savefig(os.path.join(save_dir, save_name))

                plt.show()


                data_list.append([target_value, k_type, maxLum, OLED_start_prop, round(stairStart, 2), c_multiplier, n_reversals, last_3_mean, target_last_3_diff])


# # save data_list as csv
# headers = ['target_value', 'k_type', 'maxLum', 'OLED_start_prop', 'stairStart', 'c_multiplier', 'n_reversals', 'last_3_mean', 'target_diff']
# data_df = pd.DataFrame(data_list, columns=headers)
# data_df.to_csv(os.path.join(save_dir, 'Kesten_investigation.csv'), index=False)






# # plot probeLum_list
# plt.plot(probe_lum_list)
# # add points for each value
# for x, y in enumerate(probe_lum_list):
#     if resp_list[x] == 1:
#         plt.scatter(x, y, color='green')
#     else:
#         plt.scatter(x, y, color='red')
# # add horizontal line for bgLum
# plt.axhline(y=bgLum, color='black', linestyle='--')
# # x axis label is step number, y axis label is probeLum
# plt.xlabel('trial number (for this staircase)')
# plt.ylabel('Probe luminance')
# plt.title(f'Kesten Values (probeLum)\n{k_type}, C = c_multiplier * stairStart = {c_multiplier} * {round(stairStart, 2)} = {c_multiplier * stairStart}')
# plt.show()
# print('probe_lum_list = ', probe_lum_list)
#
# # plot formula_list
# plt.plot(formula_list)
# # add points for each value
# for x, y in enumerate(formula_list):
#     if resp_list[x] == 1:
#         plt.scatter(x, y, color='green')
#     else:
#         plt.scatter(x, y, color='red')
# plt.axhline(y=0, color='black', linestyle='--')
# plt.xlabel('trial number (for this staircase)')
# plt.ylabel('Probe luminance change (formula)')
# plt.title(f'Formula (change size in exp units, e.g., luminance)\n{k_type}, C = c_multiplier * stairStart = {c_multiplier} * {round(stairStart, 2)} = {c_multiplier * stairStart}')
# plt.show()
#
# # make difference list, where each value is the difference between the current and previous value
# value_diff_list = []
# for idx, value in enumerate(probe_lum_list):
#     if idx == 0:
#         value_diff_list.append(0)
#     else:
#         value_diff_list.append(value - probe_lum_list[idx - 1])
# print('value_diff_list = ', value_diff_list)
#
#
# # # plot each diff as a proportion of the previous value
# # step_size_prop_list = []
# # for idx, value in enumerate(value_diff_list):
# #     # if idx == 0:
# #     #     step_size_prop_list.append(0)
# #     # else:
# #     #     step_size_prop_list.append(value / probe_lum_list[idx - 1])
# #     step_size_prop_list.append(value / probe_lum_list[idx])
# # print('step_size_prop_list = ', step_size_prop_list)
# # # plot step_size_prop_list, coloured by response
# # plt.plot(step_size_prop_list)
# # for x, y in enumerate(step_size_prop_list):
# #     if resp_list[x] == 1:
# #         plt.scatter(x, y, color='green')
# #     else:
# #         plt.scatter(x, y, color='red')
# # plt.axhline(y=0, color='black', linestyle='--')
# # plt.xlabel('trial number (for this staircase)')
# # plt.ylabel('Difference as proportion of previous value')
# # plt.title(f'Step size as proportion of previous value\n{k_type}, C = c_multiplier * stairStart = {c_multiplier} * {round(stairStart, 2)} = {c_multiplier * stairStart}')
# # plt.show()
#
#
#
# # plot formula_prop_list
# plt.plot(formula_prop_list)
# # add points for each value
# for x, y in enumerate(formula_prop_list):
#     if resp_list[x] == 1:
#         plt.scatter(x, y, color='green')
#     else:
#         plt.scatter(x, y, color='red')
# plt.axhline(y=0, color='black', linestyle='--')
# plt.xlabel('trial number (for this staircase)')
# plt.ylabel('Formula (as proportion of previous value)')
# plt.title(f'Formula (change size) as proportion of previous value\n{k_type}, C = c_multiplier * stairStart = {c_multiplier} * {round(stairStart, 2)} = {c_multiplier * stairStart}')
# plt.show()
#
# # show each value as a proportion of the previous value
# New_value_prop_list = []
# for idx, value in enumerate(probe_lum_list):
#     if idx == 0:
#         New_value_prop_list.append(0)
#     else:
#         New_value_prop_list.append(value / probe_lum_list[idx - 1])
# print('New_value_prop_list = ', New_value_prop_list)
# # plot New_value_prop_list
# plt.plot(New_value_prop_list)
# # add points for each value
# for x, y in enumerate(New_value_prop_list):
#     if resp_list[x] == 1:
#         plt.scatter(x, y, color='green')
#     else:
#         plt.scatter(x, y, color='red')
# plt.axhline(y=0, color='black', linestyle='--')
# plt.xlabel('trial number (for this staircase)')
# plt.ylabel('New value as proportion of previous value')
# plt.title(f'New Value as proportion of previous value\n{k_type}, C = c_multiplier * stairStart = {c_multiplier} * {round(stairStart, 2)} = {c_multiplier * stairStart}')
# plt.show()

