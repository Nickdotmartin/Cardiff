import matplotlib.pyplot as plt
import numpy as np
import psignifit as ps
import pandas as pd
import seaborn as sns

ave_thr_df_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/exp1a_data/MASTER_exp_ave_thr.csv'
ave_thr_df = pd.read_csv(ave_thr_df_path)
ave_thr_df["separation"].replace({20: -1}, inplace=True)
ave_thr_df = ave_thr_df.reindex([6, 0, 1, 2, 3, 4, 5])
print(f'ave_thr_df:\n{ave_thr_df}\n')

area_dur_path = '/Users/nickmartin/Documents/PycharmProjects/Cardiff/exp1a_notes_info_etc/probe_sizes.xlsx'

# constants
bgLum = 21.2
onePixel_width_mm = .311
oneFrame_dur_ms = 4.16666666666666

print('Duration Plots')
bloch_df = ave_thr_df.iloc[0][1:]
bloch_df = pd.DataFrame(bloch_df)
bloch_df.reset_index(inplace=True)
bloch_df.columns = ['cond', 'thr']
# bloch_df = bloch_df.reindex([1, 2, 3, 4, 5, 6, 7, 0])
# bloch_df.reset_index(drop=True, inplace=True)
print(f'bloch_df:\n{bloch_df}')

dur_df = pd.read_excel(area_dur_path, engine='openpyxl', sheet_name='duration',
                       usecols=['cond', 'probes', 'active_fr', 'duration_fr', 'prop_active'],
                       nrows=8)
print(f'dur_df:\n{dur_df}\n')

x_dur_frames = dur_df['duration_fr'].to_list() # [1:]
x_dur_ms = [i*oneFrame_dur_ms for i in x_dur_frames]
# print(f'x_dur_ms: {x_dur_ms}\n')
thr_var = bloch_df['thr'].to_list()
delta_thr = [(i-bgLum)/bgLum for i in thr_var]
print(f'thr_var: {thr_var}\n')
print(f'delta_thr: {delta_thr}\n')

'''log dur x log delta lum'''
x_var = x_dur_ms
print(f'x_var: {x_var}\n')
y_var = delta_thr
print(f'y_var: {y_var}\n')

fig, ax = plt.subplots()
sns.lineplot(x=x_var, y=y_var, ax=ax, marker='o')
ax.set_xticks(x_var)
ax.set_xticklabels(x_var)
ax.set(xscale="log", yscale="log")
ax.set_xlabel('duration of stimulus in ms (probe1 + ISI + probe2)')
ax.set_ylabel('∆lum/BGlum')
plt.title('Stimulus duration x Delta Lumination')
plt.tight_layout()
plt.savefig('/Users/nickmartin/Documents/PycharmProjects/Cardiff/exp1a_data/log_dur_x_log_delta_lum.png')
plt.show()

'''log dur x WEIGHTED log delta lum'''
x_var = x_dur_ms
print(f'x_var: {x_var}\n')

thr_multiplier = dur_df['prop_active'].to_list()
print(f'thr_multiplier: {thr_multiplier}\n')
y_var = [a*b for a, b in zip(delta_thr, thr_multiplier)] # [1:]
print(f'y_var: {y_var}\n')

fig, ax = plt.subplots()
sns.lineplot(x=x_var, y=y_var, ax=ax, marker='o')
ax.set_xticks(x_var)
ax.set_xticklabels(x_var)
ax.set(xscale="log", yscale="log")
ax.set_xlabel('duration of stimulus in ms (probe1 + ISI + probe2)')
ax.set_ylabel('∆lum/BGlum x proportion of stimulus that is illuminated')
plt.title('Stimulus duration x WEIGHTED Delta Lumination')
plt.tight_layout()
plt.savefig('/Users/nickmartin/Documents/PycharmProjects/Cardiff/exp1a_data/log_dur_x_log_weighted_delta_lum.png')
plt.show()

print('Area Plot')
ricco_df = ave_thr_df[['separation', 'Concurrent']]
ricco_df.columns = ['separation', 'thr']
print(f'ricco_df:\n{ricco_df}\n')

area_df = pd.read_excel(area_dur_path, engine='openpyxl', sheet_name='Area',
                        usecols=['index', 'spatial_cond', 'active_pixels', 'area_type', 'area', 'prop_active'],
                        index_col='index')
print(f'area_df:\n{area_df}\n')

this_area_df = area_df[area_df['area_type'] == 'circle_calc']
print(f'this_area_df:\n{this_area_df}\n')

x_area_pix = this_area_df['area'].to_list() # [1:]
x_area_mm = [i*onePixel_width_mm for i in x_area_pix]
print(f'x_area_mm: {x_area_mm}\n')

thr_var = ricco_df['thr'].to_list()
delta_thr = [(i-bgLum)/bgLum for i in thr_var]
print(f'thr_var: {thr_var}\n')
print(f'delta_thr: {delta_thr}\n')

'''log area x  log delta lum'''
x_var = x_area_mm
print(f'x_var: {x_var}\n')

y_var = delta_thr
print(f'y_var: {y_var}\n')

fig, ax = plt.subplots()
sns.lineplot(x=x_var, y=y_var, ax=ax, marker='o')
ax.set_xticks(x_var)
ax.set_xticklabels(x_var)
ax.set(xscale="log", yscale="log")
ax.set_xlabel('Circlular area of stimulus in mm$\mathregular{^2}$')
ax.set_ylabel('∆lum/BGlum')
plt.title('Stimulus area x Delta Lumination')
plt.tight_layout()
plt.savefig('/Users/nickmartin/Documents/PycharmProjects/Cardiff/exp1a_data/log_area_x_log_delta_lum.png')
plt.show()

'''log area x WEIGHTED log delta lum'''
x_var = x_area_mm
print(f'x_var: {x_var}\n')

thr_multiplier = this_area_df['prop_active'].to_list()
print(f'thr_multiplier: {thr_multiplier}\n')
y_var = [a*b for a,b in zip(delta_thr, thr_multiplier)] # [1:]
print(f'y_var: {y_var}\n')

fig, ax = plt.subplots()
sns.lineplot(x=x_var, y=y_var, ax=ax, marker='o')
ax.set_xticks(x_var)
ax.set_xticklabels(x_var)
ax.set(xscale="log", yscale="log")
ax.set_xlabel('Circlular area of stimulus in mm$\mathregular{^2}$')
ax.set_ylabel('∆lum/BGlum x proportion of stimulus that is illuminated')
plt.title('Stimulus area x WEIGHTED Delta Lumination')
plt.tight_layout()
plt.savefig('/Users/nickmartin/Documents/PycharmProjects/Cardiff/exp1a_data/log_area_x_log_weighted_delta_lum.png')
plt.show()


'''combined plot'''
all_thr_df = pd.read_csv(ave_thr_df_path)  # , index_col='separation')
# print(f'all_thr_df:\n{all_thr_df}\n')
all_thr_df["separation"].replace({20: -1}, inplace=True)
all_thr_df = ave_thr_df.reindex([6, 0, 1, 2, 3, 4, 5])
all_thr_df.insert(0, 'area', x_area_mm)
all_thr_df.set_index('area', drop=True, inplace=True)
all_thr_df.drop('separation', axis=1, inplace=True)
print(f'all_thr_df:\n{all_thr_df}\n')

all_delta_df = (all_thr_df-bgLum)/bgLum
print(f'all_delta_df:\n{all_delta_df}\n')

area_x_dur_df = pd.read_excel(area_dur_path, engine='openpyxl', sheet_name='area_x_frames',
                       usecols=['separation', 'Concurrent', 'ISI0', 'ISI2', 'ISI4', 'ISI6', 'ISI9', 'ISI12', 'ISI24'],
                       nrows=8, index_col='separation')
print(f'area_x_dur_df:\n{area_x_dur_df}\n')

all_weights_df = pd.read_excel(area_dur_path, engine='openpyxl', sheet_name='all_weights',
                       usecols=['separation', 'Concurrent', 'ISI0', 'ISI2', 'ISI4', 'ISI6', 'ISI9', 'ISI12', 'ISI24'],
                       nrows=8, index_col='separation')
all_weights_df.insert(0, 'area', x_area_mm)
all_weights_df.set_index('area', drop=True, inplace=True)
print(f'all_weights_df:\n{all_weights_df}\n')

weighted_thr_df = all_delta_df.mul(all_weights_df)
print(f'weighted_thr_df:\n{weighted_thr_df}\n')

fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(data=all_delta_df, markers=True, dashes=False, ax=ax)
ax.set(xscale="log", yscale="log")
ax.set_xlabel('Circlular area of stimulus in mm$\mathregular{^2}$')
ax.set_ylabel('∆lum/BGlum')
plt.title('Stimulus area x Delta Lumination')
plt.savefig('/Users/nickmartin/Documents/PycharmProjects/Cardiff/exp1a_data/log_area_x_log_delta_lum_ALL.png')
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(data=weighted_thr_df, markers=True, dashes=False, ax=ax)
ax.set(xscale="log", yscale="log")
ax.set_xlabel('Circlular area of stimulus in mm$\mathregular{^2}$')
ax.set_ylabel('∆lum/BGlum x proportion of stimulus that is illuminated')
plt.title('Stimulus area x WEIGHTED Delta Lumination')
plt.savefig('/Users/nickmartin/Documents/PycharmProjects/Cardiff/exp1a_data/log_area_x_log_weighted_delta_lum_ALL.png')
plt.show()


'''scatter plot'''
area_x_dur_np = area_x_dur_df.to_numpy().flatten()
print(f'area_x_dur_np:\n{area_x_dur_np}\n')

weighted_thr_np = weighted_thr_df.to_numpy().flatten()
print(f'weighted_thr_np:\n{weighted_thr_np}\n')

fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=area_x_dur_np, y=weighted_thr_np)
ax.set(xscale="log", yscale="log")
ax.set_xlabel('Stimulus area (pixels) x duration (frames)')
ax.set_ylabel('∆lum/BGlum x proportion of stimulus that is illuminated')
plt.title('Stimulus (area x duration) x WEIGHTED Delta Lumination')
plt.savefig('/Users/nickmartin/Documents/PycharmProjects/Cardiff/exp1a_data/scatter_area_x_dur_weighted.png')
plt.show()
