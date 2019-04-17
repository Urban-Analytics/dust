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

def run_enkf(observations, model_params, filter_params):
    """
    Run Ensemble Kalman Filter on model using the generated noisy data.
    Plot rmse per agent over time.
    """
    # Initialise filter with StationSim and params
    filter = EnsembleKalmanFilter(Model, filter_params, model_params)

    # Step filter
#    for i in range(filter_params['max_iterations']):
    for i in range(filter_params['max_iterations']):
        print('step {0}'.format(i))
        if i % filter_params['assimilation_period'] == 0:
            print('step with update')
            filter.step(observations[i])
        else:
            filter.step()
    return filter.results

def plot_results(results):
    pass

def make_errors(results):
    pass

def process_results(results, truth):
    """
    Function to process results, comparing against truth.
    Calculate x-error and y-error for each agent at each timestep.
    Average over all agents.
    Plot how average error varies over timestep. 
    """

def run_all():
    """
    Overall function to run everything.
    """
    # Set up params
    model_params = {'width': 200, 'height': 100, 'pop_total': 100,
                    'entrances': 3, 'entrance_space': 2, 'entrance_speed': 4,
                    'exits': 2, 'exit_space': 1, 'speed_min': .1,
                    'speed_desire_mean': 1, 'speed_desire_std': 1, 'separation': 4,
                    'wiggle': 1, 'batch_iterations': 900, 'do_save': True,
                    'do_plot': False, 'do_ani': False}

    OBS_NOISE_MEAN = 0
    OBS_NOISE_STD = 1

    vec_length = 2 * model_params['pop_total']

    filter_params = {'max_iterations': model_params['batch_iterations'],
                     'assimilation_period': 5,
                     'ensemble_size': 10,
                     'state_vector_length': vec_length,
                     'data_vector_length': vec_length,
                     'H': np.identity(vec_length),
                     'R_vector': OBS_NOISE_STD * np.ones(vec_length)}

    truth_data = make_truth_data(model_params)
    observation_data = make_observation_data(truth_data, OBS_NOISE_MEAN,
                                             OBS_NOISE_STD)
    results = run_enkf(observation_data, model_params, filter_params)

run_all()
