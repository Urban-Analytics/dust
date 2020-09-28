# -*- coding: utf-8 -*-
'''
@author: patricia-ternes
@author: nickmalleson
'''

# Need to append the main project directory (ABM_DA) and stationsim folders to the path, otherwise either
# this script will fail, or the code in the stationsim directory will fail.
import sys
#from particle_filter_gcs_temper import ParticleFilter
from particle_filter_gcs_orig import ParticleFilter
from stationsim_density_model import Model
#from stationsim_gcs_model import Model
#from stationsim_gcs_model_orig import Model
import os
import glob
import time
import warnings
import multiprocessing
import datetime
import pandas as pd
import matplotlib.pyplot as plt

t0=time.time()

# Organizing the results folder.
directory = 'results_exp4/'
if not(os.path.exists(directory)):
    os.mkdir(directory)
folders = glob.glob(directory+'*')
N_folders = len(folders)
while True:
    results_dir = directory + str(N_folders)
    if not(os.path.exists(results_dir)):
        os.mkdir(results_dir)
        break
    else:
        N_folders +=1


model_params = {'pop_total': 274, 'batch_iterations': 3100, 'step_limit': 3100, 'birth_rate': 25./15, 'do_history': False, 'do_print': False, 'station': 'Grand_Central'}
filter_params = {'agents_to_visualise': 100, 'number_of_runs': 1, 'multi_step': False, 'particle_std': 1.0, 'model_std': 1.0, 'do_save': True, 'plot_save': False,
                 'do_ani': True, 'show_ani': False, 'do_external_data': True, 'resample_window': 100,
                 'number_of_particles': 5000,
                 'do_resample': True, # True for experiments with D.A.
                 'external_info': ['gcs_final_real_data/', False, False]}  # [Real data dir, Use external velocit?, Use external gate_out?]

# Name of the log file:
outfile = os.path.join(results_dir,
        "pf_{}_particles_{}_agents_{}_noise.dat".format(
        str(int(filter_params['number_of_particles'])),
        str(int(model_params['pop_total'])),
        str(filter_params['particle_std'])
    ))
# Open the log file:
f = open(outfile, 'w')

# Save the headers of the log file:
print('# ', datetime.datetime.now(), file=f)
print('# ', file=f)
print('# ', 'PF params:', filter_params, file=f)
print('# ', file=f)
print('# ', 'Model params:', model_params, file=f)
print('# ', file=f)
print('# ', 'Min_Mean_errors', 'Max_Mean_errors', 'Average_mean_errors',
      'Min_Absolute_errors', 'Max_Absolute_errors', 'Average_Absolute_errors',
      'Min_variances', 'Max_variances', 'Average_variances',
      'Before_resample?', file=f)

print("PF params: " + str(filter_params))
print("Model params: " + str(model_params))
print("Saving files to: {}".format(outfile))

import numpy as np

start_time = time.time()  # Time how long the whole run take

# Create the Particle Filter object
pf = ParticleFilter(Model, model_params, filter_params)

# Run the particle filter
result = pf.step()
pf.pool.close()


variances = [ pf.variances[j]   for j in range(len(pf.variances))   if not pf.before_resample[j] ]
errors = [ pf.mean_errors[j] for j in range(len(pf.mean_errors)) if not pf.before_resample[j] ]
windows = list(range(1, len([x for x in pf.before_resample if x==True]) +1 ) )
data=pd.DataFrame(list(zip(windows, errors, variances)), columns=["Window", "Error", "Variance"])
data.to_csv('exp4_5000.csv')

# Save the animation
for i in range (len(pf.animation)):
    save_name = "_ani-{}agents-{}particles-window{}.png".format(
        model_params['pop_total'], filter_params['number_of_particles'], i)
    save_name = os.path.join(results_dir, save_name)
    pf.animation[i].savefig(save_name, bbox_inches="tight")

with open(outfile, 'a') as f:
    if result == None: # If no results then don't write anything
        warnings.warn("Result from the particle filter is 'none' for some reason. This sometimes happens when there is only 1 agent in the model.")
    else:
        # Two sets of errors are created, those before resampling, and those after. Results is a list with two tuples.
        # First tuple has eerrors before resampling, second has errors afterwards.
        # Save the log file:
        for before in [0, 1]:
            print(*result[before], before,  file=f)
            #print(*result[before], before, sep = ", ", file=f) # if .csv file was used!


print("Run: {}, particles: {}, agents: {}, took: {}(s), result: {}".format(1, filter_params['number_of_particles'], model_params['pop_total'], round(time.time() - start_time), result), flush=True)

print("Finished single run")

pf.estimate_model.get_distace_plot('gcs_final_real_data/frame_', 1500, 3100)

# Save the log file:
f = open(outfile, 'a')
print('# ', file=f)
print('# ', 'Distance_Mean', 'Distance_Mean_errors',file=f)
graphY1 = np.asarray(pf.estimate_model.graphY1)
graphERR1 = np.asarray(pf.estimate_model.graphERR1)
print(graphY1.mean()/14., graphERR1.mean()/14., file=f)
f.close()

save_file = open(results_dir+'/distance_1.dat', 'w')
print('# x', 'y', 'dy', file=save_file)
for i in range(len(pf.estimate_model.graphX1)):
    print(pf.estimate_model.graphX1[i], pf.estimate_model.graphY1[i], pf.estimate_model.graphERR1[i], file=save_file)
save_file.close()
save_file = open(results_dir+'/distance_2.dat', 'w')
print('# x', 'y', 'dy', file=save_file)
for i in range(len(pf.estimate_model.graphX2)):
    print(pf.estimate_model.graphX2[i], pf.estimate_model.graphY2[i], pf.estimate_model.graphERR2[i], file=save_file)
save_file.close()

