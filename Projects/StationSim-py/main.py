"""
main.py
@author: ksuchak1990
date_created: 19/04/10
Main python script for running model experiments.
"""

# Imports
import numpy as np
from numpy.random import normal
from EnsembleKalmanFilter import EnsembleKalmanFilter
from Model import Model

np.random.seed(42)

# Functions
def make_truth_data(model_params):
    """
    Run StationSim to generate synthetic truth data.
    Returns a list of the states of each agent;
    Each list entry is the state of the agents at a timestep.
    """
    # Run model with provided params
    model = Model(model_params)
    model.batch()

    # Extract agent tracks
    return model.state_history

def make_observation_data(truth, noise_mean=0, noise_std=1):
    """
    Add noise to truth data to generate noisy observations.
    Noise is drawn from a Normal distribution.
    We assume that sensors are unbiased, and so by default mean=0.
    """
    observations = list()
    for timestep in truth:
        ob = timestep + normal(noise_mean, noise_std, timestep.shape)
        observations.append(ob)
    return observations

def run_enkf(model_params, filter_params):
    """
    Run Ensemble Kalman Filter on model using the generated noisy data.
    """
    # Initialise filter with StationSim and params
    enkf = EnsembleKalmanFilter(Model, filter_params, model_params)

    for i in range(filter_params['max_iterations']):
        if i % 25 == 0:
            print('step {0}'.format(i))
        enkf.step()
    return enkf

def run_all(pop_size=20, its=300, assimilation_period=50, ensemble_size=10):
    """
    Overall function to run everything.
    """
    # Set up params
    model_params = {'width': 200,
                    'height': 100,
                    'pop_total': pop_size,
                    'entrances': 3,
                    'entrance_space': 2,
                    'entrance_speed': 4,
                    'exits': 2,
                    'exit_space': 1,
                    'speed_min': .1,
                    'speed_desire_mean': 1,
                    'speed_desire_std': 1,
                    'separation': 4,
                    'wiggle': 1,
                    'batch_iterations': its,
                    'do_save': True,
                    'do_plot': False,
                    'do_ani': False}

    OBS_NOISE_STD = 1
    vec_length = 2 * model_params['pop_total']

    filter_params = {'max_iterations': model_params['batch_iterations'],
                     'assimilation_period': assimilation_period,
                     'ensemble_size': ensemble_size,
                     'state_vector_length': vec_length,
                     'data_vector_length': vec_length,
                     'H': np.identity(vec_length),
                     'R_vector': OBS_NOISE_STD * np.ones(vec_length),
                     'keep_results': True,
                     'vis': True}

    # Run enkf and process results
    enkf = run_enkf(model_params, filter_params)
    enkf.process_results()

def run_combos():
    """
    Run the ensemble kalman filter for different combinations of:
        - assimilation period
        - ensemble size
    """
    # Parameter combos
    assimilation_periods = [2, 5, 10, 20, 50, 100]
    ensemble_sizes = [2, 5, 10, 20]

    for a in assimilation_periods:
        for e in ensemble_sizes:
            combo = (20, 300, a, e)
            run_all(*combo)

run_all(20, 300, 50, 10)
# run_combos()
