# -*- coding: utf-8 -*-
'''
Run some experiments using the stationsim_gcs.stationsim_gcs_model.

The script

@author: patricia-ternes
'''

import sys
sys.path.append('../../stationsim_gcs')
from stationsim_gcs.stationsim_gcs_model import Model
from time import strftime
id = strftime('%y%m%d_%H%M%S')

model_params = {'pop_total': 100, 'station': 'Grand_Central'}

model = Model(**model_params) 
for _ in range(model.step_limit):
	model.step()

fig = model.get_trails()
fig.savefig(f'{id}_trails.png')

analytics = model.get_analytics()
print(analytics)
print(analytics, file=open(f'{id}_stats.txt','w'))


fig = model.get_histogram()
fig.savefig(f'{id}_histogram.png')

fig = model.get_wiggle_map()  # slow
fig.savefig(f'{id}_wiggle_map.png')

fig = model.get_location_map(do_kdeplot=False)
fig.savefig(f'{id}_location_map_fast.png')

fig = model.get_location_map()  # slow
fig.savefig(f'{id}_location_map.png')

ani = model.get_ani(show_separation=True)  # slow
ani.save(f'{id}_ani.mp4')