
import os
import re
import math
import numpy as np
import matplotlib.pyplot as plt
#plt.ioff() # Turn off interactive mode
import pandas as pd
import sys
import warnings
from scipy.interpolate import griddata # For interpolating across irregularly spaced grid
import pickle # For saving computationally-expensive operations

# The following is to import the Particle Filter code
# (mostly we just read results that were created previously, but sometimes it's  useful to
# visualise additional experiments).
import sys
sys.path.append('../../stationsim')
sys.path.append('../..')
from stationsim.particle_filter import ParticleFilter
from stationsim.stationsim_model import Model
import time
import multiprocessing




# These are the basic parameter settings required. 
# We will chance the number of particles and agents to see what the experiments are like

model_params = {
    'width': 200,
    'height': 100,
    'pop_total': 10, # IMPORTANT: number of agents
    'speed_min': .1,
    'separation': 2,
    'batch_iterations': 4000,  # Only relevant in batch() mode
    'do_history': False,
    'do_print': False,
}
# Model(model_params).batch() # Runs the model as normal (one run)

filter_params = {
    'number_of_particles': 10, #IMPORTANT: number of particles
    'number_of_runs': 1,  # Number of times to run each particle filter configuration
    'resample_window': 100,
    'multi_step': True,  # Whether to predict() repeatedly until the sampling window is reached
    'particle_std': 1.0, # Noise added to particles
    'model_std': 1.0, # Observation noise
    'agents_to_visualise': 10,
    'do_save': True,
    'plot_save': False,
    'do_ani': True, # Do the animation (generatea plot at each data assimilation window)
    'show_ani': False, # Don't actually show the animation. They can be extracted later from self.animation
}




N = 50
NP = 5000

# Try to load the complete particle filter results from a pickled file. If none are available then re-run the PF
pf = None
pickle_file = f"./pickles/{N}agents_{NP}particles.pickle"

try:
    with open(pickle_file, 'rb') as f:
        pf = pickle.load(f)
        print(f"Loaded previous PF run from file {pickle_file}")
except FileNotFoundError as e:
    print("Could not load previously-run PF from file. Re-running code (go and make a cup of tea)")
    pf = None # (not sure this is necessary)
    
if pf == None:
    
    model_params['pop_total'] = N
    filter_params['number_of_particles'] = NP
    filter_params['agents_to_visualise'] = N
    print("Constructing PF")
    pf = ParticleFilter(Model, model_params, filter_params, numcores = int(multiprocessing.cpu_count()))
    print("\tconstructed object. Starting run")
    result = pf.step() # Run the particle filter

    # For some reason the pool doesn't always kill it's child processes (notebook problem?)
    pf.pool.close()
    
    pf.pool = None # This is necessary to allow pickling
    
    # Save the particle filter object as a pickle file
    with open(pickle_file, 'wb') as f:
        pickle.dump(pf, f)


print('Finished')
