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

def run_enkf(observations, model_params):
    """
    Run Ensemble Kalman Filter on model using the generated noisy data.
    Plot rmse per agent over time.
    """
    results = None
    plot_results(results)

def plot_results(results):
    pass

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

    truth_data = make_truth_data(model_params)
    observation_data = make_observation_data(truth_data, OBS_NOISE_MEAN,
                                             OBS_NOISE_STD)

run_all()
