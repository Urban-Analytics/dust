# Make some visuslisations of the particle filter under different conditions.

#%% Initialisation 
import time

from sys import path
#path.append('../..') # Add the main stationsim code directory
path.append('../../stationsim') # Add the main stationsim code directory
from particle_filter import ParticleFilter
from stationsim import Model 

# These are the parameters used in the experiments
model_params = {
    'width': 200,
    'height': 100,
    'speed_min': .1,
    'separation': 2,
    'batch_iterations': 4000,
    'do_history': False,
    'do_print': False,
}
# Model(model_params).batch() # Runs the model as normal (one run)

filter_params = {
    'number_of_runs': 1,  # Number of times to run each particle filter configuration
    'multi_step': True,  # Whether to predict() repeatedly until the sampling window is reached
    'particle_std': 2.0,  # was 2 or 10
    'model_std': 1.0,  # was 2 or 10
    'do_save': False,
    'plot_save': True,
    'do_ani': True,

}

#%%
# Show what a PF looks like with few agents
N = 1000
model_params['pop_total'] = N
filter_params['agents_to_visualise'] = N
filter_params['number_of_particles'] = 50
filter_params['resample_window'] = 50

start_time = time.time()  # Time how long the whole run take
pf = ParticleFilter(Model, model_params, filter_params)
result = pf.step()








#%%
