"""
vis.py
author: ksuchak1990
A collection of functions for visualising results from the enkf.
"""

# Imports
import matplotlib.pyplot as plt
import numpy as np
from processing_utils import extract_array

# Functions
def make_all_heatmaps(data):
    """
    Make a collection of error heatmaps.

    Use plot_heatmap() to produce heatmaps showing how the mean error varies
    with respect to assimilation period and population size, ensemble size and
    population size, and obsevation standard deviation and population size.

    Parameters
    ----------
    data : pandas dataframe
        A pandas dataframe containing mean errors and values for each of the
        input parameters.
    """
    # plot_heatmap(data, 'assimilation_period', 'ensemble_size')
    plot_heatmap(data, 'assimilation_period', 'population_size')
    # plot_heatmap(data, 'assimilation_period', 'std')
    plot_heatmap(data, 'ensemble_size', 'population_size')
    # plot_heatmap(data, 'ensemble_size', 'std')
    plot_heatmap(data, 'std', 'population_size')


def plot_heatmap(data, var1, var2):
    """
    Plotting a heat map of variation of errors with respect to two variables.

    Extract the appropriate data array from the data.
    Produce a matplotlib contour plot of the variation of the mean error with
    respect to var1 and var2.
    Save as a pdf figure with name based on the two variables.

    Parameters
    ----------
    data : pandas dataframe
        A pandas dataframe in which each row pertains to the error resulting
        from an input set of parameters. Consequently, each row contains the
        mean error, as well as the relevant parameter values.
    var1 : string
        The first variable against which we would like to measure the variation
        of the mean error.
    var2 : string
        The second variable against which we would like to measure the
        variation of the mean error.
    """
    label_dict = {'assimilation_period': 'Assimilation Period',
                  'ensemble_size': 'Ensemble Size',
                  'population_size': 'Population Size',
                  'std': 'Observation Error Standard Deviation'}
    # d, rows, cols = extract_array(data, var1, var2)
    # print(d)
    # print(rows)
    # print(cols)
    # plt.contourf(rows, cols, d)
    d = extract_array(data, var1, var2)
    print(d)
    plt.contourf(d, levels=10, cmap='PuBu')
    plt.yticks(np.arange(0, len(d.index), 1), d.index)
    plt.xticks(np.arange(0, len(d.columns), 1), d.columns)
    # plt.xticks(np.arange(0, len(cols), 1), cols)
    # plt.yticks(np.arange(0, len(rows), 1), rows)
    plt.xlabel(label_dict[var1])
    plt.ylabel(label_dict[var2])
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('./results/figures/{0}_{1}.pdf'.format(var1, var2))
    plt.show()


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
    no_plot = ['sd', 'up_diff', 'down_diff']
    colnames = list(dataset)
    plt.figure()
    for col in colnames:
        if col == 'mean':
            plt.plot(dataset[col], 'b-', linewidth=5, label='mean')
        elif col not in no_plot:
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
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True,
                                      figsize=(8, 12))
    # f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True)

    no_plot = ['sd', 'up_diff', 'down_diff']

    colnames = list(forecast)
    for col in colnames:
        if col == 'mean':
            ax1.plot(forecast[col], 'b-', linewidth=2, label='forecast mean')
        elif col not in no_plot:
            ax1.plot(forecast[col], 'b--', alpha=0.25, label='_nolegend_')
    ax1.legend(loc='upper left')
    ax1.set_ylabel('RMSE')

    colnames = list(analysis)
    for col in colnames:
        if col == 'mean':
            ax2.plot(analysis[col], 'g-', linewidth=2, label='analysis mean')
        elif col not in no_plot:
            ax2.plot(analysis[col], 'g--', alpha=0.25, label='_nolegend_')
    ax2.legend(loc='upper left')
    ax2.set_ylabel('RMSE')

    colnames = list(observation)
    for col in colnames:
        if col == 'mean':
            ax3.plot(observation[col], 'k-', linewidth=2, label='observation mean')
        elif col not in no_plot:
            ax3.plot(observation[col], 'k--', alpha=0.25, label='_nolegend_')
    ax3.legend(loc='upper left')
    ax3.set_xlabel('time')
    ax3.set_ylabel('RMSE')

    plt.savefig('results/figures/all_results.pdf')

    plt.show()


def plot_with_errors(forecast, analysis, observation):
    """
    Plot results with errors.

    Plot forecast, analysis and observations in one plot.
    Contains three subplots, each one pertaining to one of the datasets.
    Subplots share x-axis and y-axis.
    Each subplot contains a mean line, and a shaded area pertaining to the
    range of the data at each point in model time.

    Parameters
    ----------
    forecast : pandas dataframe
        pandas dataframe of forecast data.
    analysis : pandas dataframe
        pandas dataframe of analysis data.
    observation : pandas dataframe
        pandas dataframe of observation data.
    """
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True,
                                      figsize=(8, 12))

    ax1.plot(forecast['mean'], 'b-', label='forecast mean')
    ax1.fill_between(forecast.index,
                     forecast['mean'] - forecast['down_diff'],
                     forecast['mean'] + forecast['up_diff'],
                     alpha=0.25)
    ax1.legend(loc='upper left')
    ax1.set_ylabel('RMSE')

    ax2.plot(analysis['mean'], 'g-', label='analysis mean')
    ax2.fill_between(analysis.index,
                     analysis['mean'] - analysis['down_diff'],
                     analysis['mean'] + analysis['up_diff'],
                     alpha=0.25)
    ax2.legend(loc='upper left')
    ax2.set_ylabel('RMSE')

    ax3.plot(observation['mean'], 'k-', label='observation mean')
    ax3.fill_between(observation.index,
                     observation['mean'] - observation['down_diff'],
                     observation['mean'] + observation['up_diff'],
                     alpha=0.25)
    ax3.legend(loc='upper left')
    ax3.set_ylabel('RMSE')

    plt.savefig('results/figures/all_with_errors.pdf')

    plt.show()

def plot_model(enkf, title_str, obs=[]):
    """
    Plot base_model, observations and ensemble mean for each agent.
    """
    # Get coords
    base_state = enkf.base_model.get_state(sensor='location')
    base_x, base_y = enkf.separate_coords(base_state)
    mean_x, mean_y = enkf.separate_coords(enkf.state_mean)
    if len(obs) > 0:
        obs_x, obs_y = enkf.separate_coords(obs)
    if enkf.run_vanilla:
        vanilla_x, vanilla_y = enkf.separate_coords(enkf.vanilla_state_mean)

    # Plot agents
    plot_width = 8
    aspect_ratio = enkf.base_model.height / enkf.base_model.width
    plot_height = round(aspect_ratio * plot_width)
    plt.figure(figsize=(plot_width, plot_height))
    plt.xlim(0, enkf.base_model.width)
    plt.ylim(0, enkf.base_model.height)
    # plt.title(title_str)
    plt.scatter(base_x, base_y, label='Ground truth')
    plt.scatter(mean_x, mean_y, marker='x', label='Ensemble mean')
    if len(obs) > 0:
        plt.scatter(obs_x, obs_y, marker='*', label='Observation')
    if enkf.run_vanilla:
        plt.scatter(vanilla_x, vanilla_y, alpha=0.5, label='mean w/o da',
                    color='black')

    # Plot ensemble members
    # for model in self.models:
        # xs, ys = self.separate_coords(model.state)
        # plt.scatter(xs, ys, s=1, color='red')

    # Finish fig
    plt.legend()
    plt.xlabel('x-position')
    plt.ylabel('y-position')
    plt.savefig('./results/{0}.eps'.format(title_str))
    plt.show()

def plot_model2(enkf, title_str, obs=[]):
    """
    Plot base_model and ensemble members for a single agent.
    """
    # List of all plotted coords
    x_coords = list()
    y_coords = list()

    # Get coords
    base_state = enkf.base_model.get_state(sensor='location')
    base_x, base_y = enkf.separate_coords(base_state)
    mean_x, mean_y = enkf.separate_coords(enkf.state_mean)
    if len(obs) > 0:
        obs_x, obs_y = enkf.separate_coords(obs)
    if enkf.run_vanilla:
        vanilla_x, vanilla_y = enkf.separate_coords(enkf.vanilla_state_mean)

    # Plot agents
    plot_width = 8
    aspect_ratio = enkf.base_model.height / enkf.base_model.width
    plot_height = round(aspect_ratio * plot_width)
    plt.figure(figsize=(plot_width, plot_height))
    plt.xlim(0, enkf.base_model.width)
    plt.ylim(0, enkf.base_model.height)
    # plt.title(title_str)
    plt.scatter(base_x[enkf.agent_number],
                base_y[enkf.agent_number], label='Ground truth')
    plt.scatter(mean_x[enkf.agent_number],
                mean_y[enkf.agent_number], marker='x', label='Ensemble mean')
    if len(obs) > 0:
        plt.scatter(obs_x[enkf.agent_number],
                    obs_y[enkf.agent_number], marker='*', label='Observation')
    if enkf.run_vanilla:
        plt.scatter(vanilla_x, vanilla_y, alpha=0.5, label='mean w/o da',
                    color='black')

    x_coords.extend([base_x[enkf.agent_number], mean_x[enkf.agent_number],
                     obs_x[enkf.agent_number]])
    y_coords.extend([base_y[enkf.agent_number], mean_y[enkf.agent_number],
                     obs_y[enkf.agent_number]])

    # Plot ensemble members
    for i, model in enumerate(enkf.models):
        state_vector = model.get_state(sensor='location')
        xs, ys = enkf.separate_coords(state_vector)
        if i == 0:
            plt.scatter(xs[enkf.agent_number], ys[enkf.agent_number],
                        s=1, color='red', label='Ensemble members')
        else:
            plt.scatter(xs[enkf.agent_number], ys[enkf.agent_number],
                        s=1, color='red')
        x_coords.append(xs[enkf.agent_number])
        y_coords.append(ys[enkf.agent_number])

    # Finish fig
    plt.legend()
    plt.xlabel('x-position')
    plt.ylabel('y-position')
    plt.xlim(x_coords[0]-5, x_coords[0]+5)
    plt.ylim(y_coords[0]-5, y_coords[0]+5)
    # plt.xlim(min(x_coords)-1, max(x_coords)+1)
    # plt.ylim(min(y_coords)-1, max(y_coords)+1)
    plt.savefig('./results/{0}_single.eps'.format(title_str))
    plt.show()

def plot_model_results(distance_errors, x_errors, y_errors):
    """
    Method to plot the evolution of errors in the filter.
    """
    plt.figure()
    plt.scatter(range(len(distance_errors)), distance_errors,
                label='$\mu$', s=1)
    plt.scatter(range(len(x_errors)), x_errors, label='$\mu_x$', s=1)
    plt.scatter(range(len(y_errors)), y_errors, label='$\mu_y$', s=1)
    plt.xlabel('Time')
    plt.ylabel('MAE')
    plt.legend()
    plt.savefig('./results/errors.eps')
    plt.show()

def plot_model_results2(errors_da, errors_vanilla):
    """
    Method to plot the evolution of errors for abm with da vs w/o da.
    """
    plt.figure()
    plt.scatter(range(len(errors_da)), errors_da, label='with')
    plt.scatter(range(len(errors_vanilla)), errors_vanilla, label='w/o')
    plt.xlabel('Time')
    plt.ylabel('MAE')
    plt.legend()
    plt.show()

