"""
processing.py
author: ksuchak1990
A collection of functions for processing data from the enkf.
"""

# Imports
import json
import numpy as np
from numpy.random import normal
from os import listdir
import pandas as pd
from vis_utils import make_all_heatmaps


# Functions
def process_batch(read_time=False, write_time=True):
    """
    Process the output data from a batch of runs.

    Stage 1:
    Consider each file in the results/repeats/ directory.
    For each file:
    1) Derive parameter values from the filename.
    2) Read the results.
    3) Find the means for the forecast, analytis and observations.
    4) Average each over time.
    5) Add to existing results.
    6) If writing then output combined results to json.

    Stage 2:
    1) If reading then read in exsiting results, else follow stage 1.
    2) Convert output data to dataframe.
    3) Produce a collection of heatmaps to summarise results.

    Parameters
    ----------
    read_time : boolean
        Boolean to choose whether to read in existing json of results.
    write_time : boolean
        Boolean to choose whether to write out processed results to json.
    """
    if read_time:
        with open('results/map_data.json') as f:
            output = json.load(f)
    else:
        # Set up link to directory
        results_path = './results/repeats/'
        results_list = listdir(results_path)
        output = list()

        for r in results_list:
            # Derive parameters from filename
            components = r.split('__')

            ap = int(components[0].split('_')[-1])
            es = int(components[1])
            pop_size = int(components[2])
            pre_sigma = components[3].split('.')[0]
            sigma = float(pre_sigma.replace('_', '.'))

            # Read in set of results:
            p = './results/repeats/{0}'.format(r)
            with open(p) as f:
                d = json.load(f)

            # Reduce to means for forecast, analysis and obs
            forecasts, analyses, observations = process_repeat_results(d)

            # Take mean over time
            forecast = forecasts['mean'].mean()
            analysis = analyses['mean'].mean()
            observation = observations['mean'].mean()

            # Add to output list
            row = {'assimilation_period': ap,
                   'ensemble_size': es,
                   'population_size': pop_size,
                   'std': sigma,
                   'forecast': forecast,
                   'analysis': analysis,
                   'obsevation': observation}

            output.append(row)

        if write_time:
            with open('results/map_data.json', 'w', encoding='utf-8') as f:
                json.dump(output, f, ensure_ascii=False, indent=4)

    data = pd.DataFrame(output)
    make_all_heatmaps(data)


def extract_array(df, var1, var2):
    """
    Function to extract data array pertaining to the variables that we are
    interested in.

    Extract an array of the mean errors with two parameters varying; other
    parameters are kept fixed.
    First define the default values for each of the four possible parameters
    (assimilation period, ensemble size, population size and observation noise
    standard deviation).
    Get the sorted values that each of the chosen parameters take.
    Create an array of the data that fits the above conditions, and convert
    into an array with column indices taking the values of the first parameter
    and the row indices taking the value of the second parameter.

    Parameters
    ----------
    df : pandas dataframe
        A pandas dataframe containing all of the mean errors for each of the
        parameter combinations.
    var1 : string
        Name of the first variable that we want to consider variation with
        respect to.
    var2 : string
        Name of the second variable that we want to consider variation with
        respect to.
    """
    # Define variables to fix and filter
    fixed_values = {'assimilation_period': 20,
                    'ensemble_size': 20,
                    'population_size': 15,
                    'std': 1.5}

    var1_vals = sorted(df[var1].unique())
    var2_vals = sorted(df[var2].unique())
    fix_vars = [x for x in fixed_values.keys() if x not in [var1, var2]]
    print(var1, var1_vals)
    print(var2, var2_vals)

    # Filtering down to specific fixed values
    cond1 = df[fix_vars[0]] == fixed_values[fix_vars[0]]
    cond2 = df[fix_vars[1]] == fixed_values[fix_vars[1]]
    tdf = df[cond1 & cond2]

    # Reformat to array
    a = np.zeros(shape=(len(var1_vals), len(var2_vals)))
    for i, u in enumerate(var1_vals):
        for j, v in enumerate(var2_vals):
            var1_cond = tdf[var1] == u
            var2_cond = tdf[var2] == v
            d = tdf[var1_cond & var2_cond]
            a[i][j] = d['analysis'].values[0]

    output = pd.DataFrame(a, index=var1_vals, columns=var2_vals)
    # output = pd.DataFrame(a.T, index=var2_vals, columns=var1_vals)
    # return a.T, var2_vals, var1_vals
    return output.T


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
    m = d.mean(axis=1)
    s = d.std(axis=1)
    up = d.max(axis=1)
    down = d.min(axis=1)
    d['mean'] = m
    d['up_diff'] = up - m
    d['down_diff'] = m - down
    d['sd'] = s
    d['time'] = times
    return d.set_index('time')

