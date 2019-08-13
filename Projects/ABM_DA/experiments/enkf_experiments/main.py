"""
main.py
@author: ksuchak1990
Python script for running experiments with the enkf.
"""

# Imports
import json
import sys
import numpy as np
from numpy.random import normal
sys.path.append('../../stationsim/')

from stationsim_model import Model
from ensemble_kalman_filter import EnsembleKalmanFilter

# np.random.seed(42)

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
                    'gates_in': 3,
                    'gates_space': 2,
                    'gates_speed': 4,
                    'gates_out': 2,
                    'speed_min': .1,
                    'speed_mean': 1,
                    'speed_std': 1,
                    'speed_steps': 3,
                    'separation': 4,
                    'max_wiggle': 1,
                    'step_limit': its,
                    'do_history': True,
                    'do_print': False}

    OBS_NOISE_STD = 1
    vec_length = 2 * model_params['pop_total']

    filter_params = {'max_iterations': its,
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

def run_repeat(N=10, write_json=False):
    """
    Repeatedly run an enkf realisation of stationsim.

    Run a realisation of stationsim with the enkf repeatedly.
    Produces RMSE values for forecast, analysis and observations at each
    assimilation step.

    Parameters
    ----------
    N : int
        The number of times we want to run the ABM-DA.
    """
    model_params = {'width': 200,
                    'height': 100,
                    'pop_total': 20,
                    'gates_in': 3,
                    'gates_space': 2,
                    'gates_speed': 4,
                    'gates_out': 2,
                    'speed_min': .1,
                    'speed_mean': 1,
                    'speed_std': 1,
                    'speed_steps': 3,
                    'separation': 4,
                    'max_wiggle': 1,
                    'step_limit': 300,
                    'do_history': True,
                    'do_print': False}

    OBS_NOISE_STD = 1
    vec_length = 2 * model_params['pop_total']

    filter_params = {'max_iterations': model_params['step_limit'],
                     'assimilation_period': 50,
                     'ensemble_size': 10,
                     'state_vector_length': vec_length,
                     'data_vector_length': vec_length,
                     'H': np.identity(vec_length),
                     'R_vector': OBS_NOISE_STD * np.ones(vec_length),
                     'keep_results': True,
                     'vis': False}

    errors = list()
    for i in range(N):
        print('Running iteration {0}'.format(i+1))
        enkf = run_enkf(model_params, filter_params)
        errors.append(enkf.rmse)

    if write_json:
        with open('data.json', 'w', encoding='utf-8') as f:
            json.dump(errors, f, ensure_ascii=False, indent=4)
    return errors

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

def process_repeat_results(results):
    forecasts = list()
    analyses = list()
    observations = list()

    for res in results:
        # Sort results by time
        res = sorted(res, key=lambda k: k['time'])
        forecast = list()
        analysis = list()
        observation = list()
        for r in res:
            forecast.append(r['forecast'])
            analysis.append(r['analysis'])
            observation.append(r['obs'])
        forecasts.append(forecast)
        analyses.append(analysis)
        observations.append(observation)

    return np.array(forecasts), np.array(analyses), np.array(observations)

# run_all(20, 300, 50, 10)
# run_combos()
# run_repeat()

with open('data.json') as json_file:
    data = json.load(json_file)

forecasts, analyses, observations = process_repeat_results(data)
print(forecasts[0,:])
