"""
main.py
@author: ksuchak1990
date_created: 19/04/10
Main python script for running model experiments.
"""

# Imports
import numpy as np
from numpy.random import normal
import matplotlib.pyplot as plt
from EnsembleKalmanFilter import EnsembleKalmanFilter
from Model import Model

np.random.seed(42)

# Functions
# def make_truth_data(model_params):
    # """
    # Run StationSim to generate synthetic truth data.
    # Returns a list of the states of each agent;
    # Each list entry is the state of the agents at a timestep.
    # """
    # # Run model with provided params
    # model = Model(model_params)
    # model.batch()

    # # Extract agent tracks
    # return model.state_history

# def make_observation_data(truth, noise_mean=0, noise_std=1):
    # """
    # Add noise to truth data to generate noisy observations.
    # Noise is drawn from a Normal distribution.
    # We assume that sensors are unbiased, and so by default mean=0.
    # """
    # observations = list()
    # for timestep in truth:
        # ob = timestep + normal(noise_mean, noise_std, timestep.shape)
        # observations.append(ob)
    # return observations

def run_enkf(model_params, filter_params):
    """
    Run Ensemble Kalman Filter on model using the generated noisy data.
    """
    # Initialise filter with StationSim and params
    enkf = EnsembleKalmanFilter(Model, filter_params, model_params)
    am = enkf.assimilation_period

    # # Step filter
    # for i in range(filter_params['max_iterations']):
        # if i % 50 == 0:
            # print('step {0}'.format(i))
        # if i % filter_params['assimilation_period'] == 0:
            # print('step with update')
            # filter.step(observations[i])
        # else:
            # filter.step()
    # return filter.results
    for i in range(filter_params['max_iterations']):
        if i % 25 == 0:
            print('step {0}'.format(i))
        enkf.step()
    return enkf
    # return enkf.results, enkf.base_model.state_history

# def plot_results(x_errors, y_errors):
    # """
    # Function to plot the evolution of errors in the filter.
    # """
    # plt.figure()
    # plt.scatter(range(len(x_errors)), x_errors, label='$\mu_x$', s=1)
    # plt.scatter(range(len(y_errors)), y_errors, label='$\mu_y$', s=1)
    # plt.xlabel('Time')
    # plt.ylabel('Mean absolute error')
    # plt.legend()
    # plt.show()

# def plot_tracks(results, truth):
    # """
    # Function to plot the tracks of the agents.
    # Plots the filter tracks and the true tracks.
    # """
    # # Transform data

# def separate_coords(arr):
    # """
    # Function to split a flat array into xs and ys.
    # Assumes that xs and ys alternate.
    # """
    # return arr[::2], arr[1::2]

# def make_errors(results, truth):
    # """
    # Function to calculate x errors and y errors.
    # """
    # x_results, y_results = separate_coords(results)
    # x_truth, y_truth = separate_coords(truth)
    # x_error = np.abs(x_results - x_truth)
    # y_error = np.abs(y_results - y_truth)
    # return x_error, y_error

# def process_results(results, truth):
    # """
    # Function to process results, comparing against truth.
    # Calculate x-error and y-error for each agent at each timestep.
    # Average over all agents.
    # Plot how average error varies over timestep.
    # """
    # x_errors = list()
    # y_errors = list()
    # x_mean_errors = list()
    # y_mean_errors = list()

    # for i, result in enumerate(results):
        # x_error, y_error = make_errors(result, truth[i])
        # x_errors.append(x_error)
        # y_errors.append(y_error)
        # x_mean_errors.append(np.mean(x_errors))
        # y_mean_errors.append(np.mean(y_errors))

    # plot_results(x_mean_errors, y_mean_errors)

def run_all():
    """
    Overall function to run everything.
    """
    # Set up params
    model_params = {'width': 200, 'height': 100, 'pop_total': 20,
                    'entrances': 3, 'entrance_space': 2, 'entrance_speed': 4,
                    'exits': 2, 'exit_space': 1, 'speed_min': .1,
                    'speed_desire_mean': 1, 'speed_desire_std': 1, 'separation': 4,
                    'wiggle': 1, 'batch_iterations': 300, 'do_save': True,
                    'do_plot': False, 'do_ani': False}

    OBS_NOISE_MEAN = 0
    OBS_NOISE_STD = 1

    vec_length = 2 * model_params['pop_total']

    # truth_data = make_truth_data(model_params)
    # observation_data = make_observation_data(truth_data, OBS_NOISE_MEAN,
                                             # OBS_NOISE_STD)

    filter_params = {'max_iterations': model_params['batch_iterations'],
                     'assimilation_period': 50,
                     'ensemble_size': 10,
                     'state_vector_length': vec_length,
                     'data_vector_length': vec_length,
                     'H': np.identity(vec_length),
                     'R_vector': OBS_NOISE_STD * np.ones(vec_length)}

    # results = run_enkf(observation_data, model_params, filter_params)
    # process_results(results, truth_data)
    enkf = run_enkf(model_params, filter_params)
    enkf.process_results()
    # results, truth_data = run_enkf(model_params, filter_params)
    # process_results(results, truth_data)

run_all()
