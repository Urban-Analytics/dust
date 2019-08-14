"""
main.py
@author: ksuchak1990
Python script for running experiments with the enkf.
"""

# Imports
import json
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    """
    process_repeat_results

    Takes the results of running the enkf repeatedly and restructures it into
    separate data structures for forecasts, analyses and observations.

    Parameters
    ----------
    results : list(list(dict()))
        Each list entry is a list of dictionaries which stores the time-series
        of the forecast, analysis and observations for that realisation.
        Each dictionary contains entries for:
            - time
            - forecast
            - analysis
            - observation
    """
    forecasts = list()
    analyses = list()
    observations = list()
    first = True
    times = list()

    for res in results:
        # Sort results by time
        res = sorted(res, key=lambda k: k['time'])
        forecast = list()
        analysis = list()
        observation = list()
        for r in res:
            if first:
                times.append(r['time'])
            forecast.append(r['forecast'])
            analysis.append(r['analysis'])
            observation.append(r['obs'])
        first = False
        forecasts.append(forecast)
        analyses.append(analysis)
        observations.append(observation)

    forecasts = make_dataframe(forecasts, times)
    analyses = make_dataframe(analyses, times)
    observations = make_dataframe(observations, times)
    return forecasts, analyses, observations

def make_dataframe(dataset, times):
    """
    make_dataframe

    Make a dataframe from a dataset.
    This requires that the data undergo the following transformations:
        - Convert to numpy array
        - Transpose array
        - Convert array to pandas dataframe
        - Calculate row-mean in new column
        - Add time to data
        - Set time as index

    Parameters
    ----------
    dataset : list(list())
        List of lists containing data.
        Each inner list contains a single time-series.
        The outer list contains a collection of inner lists, each pertaining to
        a realisation of the model.
    times : list-like
        List of times at which data is provided.
    """
    d = pd.DataFrame(np.array(dataset).T)
    d['mean'] = d.mean(axis=1)
    d['sd'] = d.std(axis=1)
    d['time'] = times
    return d.set_index('time')

def plot_results(dataset):
    """
    plot_results

    Plot results for a single dataset (i.e. either forecast, analysis or
    observations). Produces a line graph containing individual lines for each
    realisation (low alpha and dashed), and a line for the mean of the
    realisations (full alpha and bold).

    Parameters
    ----------
    dataset : pandas dataframe
        pandas dataframe of data containing multiple realisations and mean of
        all realisations indexed on time.
    """
    colnames = list(dataset)
    plt.figure()
    for col in colnames:
        if col == 'mean':
            plt.plot(dataset[col], 'b-', linewidth=5, label='mean')
        elif col != 'sd':
            plt.plot(dataset[col], 'b--', alpha=0.25, label='_nolegend_')
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('RMSE')
    plt.show()

def plot_all_results(forecast, analysis, observation):
    """
    plot_all_results

    Plot forecast, analysis and observations in one plot.
    Contains three subplots, each one pertaining to one of the datasets.
    Subplots share x-axis and y-axis.

    Parameters
    ----------
    forecast : pandas dataframe
        pandas dataframe of forecast data.
    analysis : pandas dataframe
        pandas dataframe of analysis data.
    observation : pandas dataframe
        pandas dataframe of observation data.
    """
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True)

    colnames = list(forecast)
    for col in colnames:
        if col != 'mean':
            ax1.plot(forecast[col], 'b--', alpha=0.25, label='_nolegend_')
        else:
            ax1.plot(forecast[col], 'b-', linewidth=2, label='forecast mean')
    ax1.legend(loc='upper left')
    ax1.set_ylabel('RMSE')

    colnames = list(analysis)
    for col in colnames:
        if col != 'mean':
            ax2.plot(analysis[col], 'g--', alpha=0.25, label='_nolegend_')
        else:
            ax2.plot(analysis[col], 'g-', linewidth=2, label='analysis mean')
    ax2.legend(loc='upper left')
    ax2.set_ylabel('RMSE')

    colnames = list(observation)
    for col in colnames:
        if col != 'mean':
            ax3.plot(observation[col], 'k--', alpha=0.25, label='_nolegend_')
        else:
            ax3.plot(observation[col], 'k-', linewidth=2, label='observation mean')
    ax3.legend(loc='upper left')
    ax3.set_xlabel('time')
    ax3.set_ylabel('RMSE')

    plt.show()

# run_all(20, 300, 50, 10)
# run_combos()
# run_repeat()

def testing():
    with open('data.json') as json_file:
        data = json.load(json_file)
    forecasts, analyses, observations = process_repeat_results(data)
    plot_all_results(forecasts, analyses, observations)

testing()
