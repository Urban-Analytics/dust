# Imports
from math import sqrt
import numpy as np
import pytest
import sys
sys.path.append('../stationsim/')

from ensemble_kalman_filter import EnsembleKalmanFilter
from ensemble_kalman_filter import EnsembleKalmanFilterType
from ensemble_kalman_filter import ActiveAgentNormaliser
from stationsim_gcs_model import Model


# Functions
def make_observation_operator(population_size, mode):
    if mode == EnsembleKalmanFilterType.STATE:
        return np.identity(2 * population_size)
    elif mode == EnsembleKalmanFilterType.DUAL_EXIT:
        return make_exit_observation_operator(population_size)
    else:
        raise ValueError(f'Unexpected filter mode: {mode}')


def make_exit_observation_operator(population_size):
    a = np.identity(2 * population_size)
    b = np.zeros(shape=(2 * population_size, population_size))
    return np.hstack((a, b))


def make_state_vector_length(population_size, mode):
    if mode == EnsembleKalmanFilterType.STATE:
        return 2 * population_size
    elif mode == EnsembleKalmanFilterType.DUAL_EXIT:
        return 3 * population_size
    else:
        raise ValueError(f'Unexpected filter mode: {mode}')


def set_up_enkf(error_normalisation=None):
    np.random.seed(666)
    pop_size = 5
    mode = EnsembleKalmanFilterType.STATE
    data_mode = EnsembleKalmanFilterType.STATE
    its = 10
    assimilation_period = 5
    ensemble_size = 5

    model_params = {'pop_total': pop_size,
                    'station': 'Grand_Central',
                    'do_print': False}

    OBS_NOISE_STD = 1
    observation_operator = make_observation_operator(pop_size, mode)
    state_vec_length = make_state_vector_length(pop_size, mode)
    data_mode = EnsembleKalmanFilterType.STATE
    data_vec_length = make_state_vector_length(pop_size, data_mode)

    filter_params = {'max_iterations': its,
                     'assimilation_period': assimilation_period,
                     'ensemble_size': ensemble_size,
                     'population_size': pop_size,
                     'state_vector_length': state_vec_length,
                     'data_vector_length': data_vec_length,
                     'mode': mode,
                     'error_normalisation': error_normalisation,
                     'H': observation_operator,
                     'R_vector': OBS_NOISE_STD * np.ones(data_vec_length),
                     'keep_results': True,
                     'run_vanilla': False,
                     'vis': False}

    enkf = EnsembleKalmanFilter(Model, filter_params, model_params)
    return enkf


def get_round_destination_data():
    round_destination_data = [(-1.2, 10, 9),
                              (-0.8, 10, 9),
                              (0.1, 10, 0),
                              (9.8, 10, 0),
                              (10.2, 10, 0),
                              (11.1, 10, 1)]
    return round_destination_data


def get_pair_coords_data():
    pair_coords_data = [([1, 2, 3, 4, 5],
                         [10, 9, 8, 7, 6],
                         [1, 10, 2, 9, 3, 8, 4, 7, 5, 6])]
    return pair_coords_data


def get_pair_coords_error_data():
    pair_coords_error_data = [([1, 2, 3, 4],
                               [10, 9, 8, 7, 6],
                               pytest.raises(ValueError)),
                              ([1, 2, 3, 4, 5],
                               [10, 9, 8],
                               pytest.raises(ValueError))]
    return pair_coords_error_data


def get_separate_coords_data():
    separate_coords_data = [([1, 2, 3, 4, 5, 6, 7, 8],
                             [1, 3, 5, 7],
                             [2, 4, 6, 8])]
    return separate_coords_data


def get_separate_coords_error_data():
    separate_coords_error_data = [([1, 2, 3, 4, 5],
                                   pytest.raises(ValueError))]
    return separate_coords_error_data


def get_random_destination_data():
    return [(10, 10, 0)]


def get_n_active_agents_base_data():
    d = [(i, i) for i in range(1, 5)]
    return d


def get_n_active_agents_mean_data():
    d = [([2, 3, 4, 3, 2], 3),
         ([1, 1, 1, 1, 1], 1),
         ([4, 3, 2, 1, 0], 2)]
    return d


def get_n_active_agents_max_data():
    d = [([2, 3, 4, 3, 2], 4),
         ([1, 1, 1, 1, 1], 1),
         ([4, 3, 2, 1, 0], 4)]
    return d


def get_n_active_agents_min_data():
    d = [([2, 3, 4, 3, 2], 2),
         ([1, 1, 1, 1, 1], 1),
         ([4, 3, 2, 1, 0], 0)]
    return d


def get_population_mean_base_data():
    diffs = ([0, 1, 1, 1, 3],
             [0, 4, 0, 2, 3],
             [1, 0, 0, 1, 0])

    active_pops = (3, 2, 2)

    expected = (2, 4.5, 1)

    d = list()
    for i in range(len(diffs)):
        x = (diffs[i],
             active_pops[i],
             expected[i])
        d.append(x)

    return d


def get_population_mean_mean_data():
    diffs = ([0, 3, 1, 4, 2],
             [0, 0, 3, 3, 0],
             [1, 1, 1, 1, 1])

    ensemble_actives = ([4, 4, 4, 4, 4],
                        [1, 2, 3, 2, 3],
                        [0, 0, 1, 0, 0])

    expected = (2.5, 3, 0)

    d = list()

    for i in range(len(diffs)):
        x = (diffs[i],
             ensemble_actives[i],
             expected[i])
        d.append(x)

    return d


def get_population_mean_max_data():
    diffs = ([0, 3, 1, 4, 2],
             [0, 0, 3, 3, 0],
             [1, 1, 1, 1, 1])

    ensemble_actives = ([4, 4, 4, 4, 4],
                        [1, 2, 3, 2, 3],
                        [0, 0, 1, 0, 0])

    expected = (2.5, 2, 5)

    d = list()

    for i in range(len(diffs)):
        x = (diffs[i],
             ensemble_actives[i],
             expected[i])
        d.append(x)

    return d


def get_population_mean_min_data():
    diffs = ([0, 3, 1, 4, 2],
             [0, 0, 3, 3, 0],
             [1, 1, 1, 1, 1])

    ensemble_actives = ([4, 4, 4, 4, 4],
                        [1, 2, 3, 2, 3],
                        [0, 0, 1, 0, 0])

    expected = (2.5, 6, 0)

    d = list()

    for i in range(len(diffs)):
        x = (diffs[i],
             ensemble_actives[i],
             expected[i])
        d.append(x)

    return d


def get_mean_data():
    results = ([2, 5, 3, 6, 4],
               [3, 6, 9, 12, 15],
               [0, 0, 0, 0, 0])

    truths = ([2, 2, 2, 2, 2],
              [3, 6, 6, 9, 15],
              [1, 1, 1, 1, 1])

    expected = (2, 1.2, 1)

    d = list()

    for i in range(len(results)):
        x = (results[i],
             truths[i],
             expected[i])
        d.append(x)

    return d


def get_error_normalisation_type_data():
    error_normalisations = (None,
                            ActiveAgentNormaliser.BASE,
                            ActiveAgentNormaliser.MEAN_EN)

    results = [2, 5, 3, 6, 4]

    truths = [2, 2, 2, 2, 2]

    active_pop = 4

    ensemble_active = [1, 2, 2, 2, 3]

    expected = (2, 2.5, 5)

    d = list()

    for i in range(len(error_normalisations)):
        x = (error_normalisations[i], results, truths,
             active_pop, ensemble_active, expected[i])
        d.append(x)

    return d


def get_distance_error_default_data():
    x_errors = ([0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
                [1, 2, 3, 4, 5])

    y_errors = ([1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0],
                [2, 4, 6, 8, 10])

    expected = (1, 1, 3 * sqrt(5))

    output = list()

    for i in range(len(x_errors)):
        x = (x_errors[i], y_errors[i], expected[i])
        output.append(x)

    return output


def get_distance_error_base_data():
    x_errors = ([0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
                [1, 2, 3, 4, 5])

    y_errors = ([1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0],
                [2, 4, 6, 8, 10])

    n_active = (1, 2, 3)

    expected = (5, 2.5, 5 * sqrt(5))

    output = list()

    for i in range(len(x_errors)):
        x = (x_errors[i], y_errors[i],
             n_active[i], expected[i])
        output.append(x)

    return output


def get_calculate_rmse_default_data():
    x_truths = ([22, 53, 45],
                [123, 321, 222])

    y_truths = ([18, 46, 80],
                [11, 32, 23])

    x_results = ([26, 50, 40],
                 [120, 336, 210])

    y_results = ([15, 50, 80],
                 [15, 40, 18])

    expected = (5, 35/3)

    output = list()

    for i in range(len(x_truths)):
        x = (x_truths[i], y_truths[i],
             x_results[i], y_results[i],
             expected[i])
        output.append(x)

    return output


def get_make_obs_error_data():
    truth = [1, 2, 6, 6, 12, 15]

    results = [1, 1, 6, 6, 15, 19]

    active_pop = (3, 4, 5)

    ensemble_active = ([2, 3, 4, 5, 3],
                       [1, 2, 1, 2, 2],
                       [1, 1, 2, 1, 1])

    normaliser = (None,
                  ActiveAgentNormaliser.BASE,
                  ActiveAgentNormaliser.MEAN_EN)

    expected = (2, 2, 2)

    output = list()

    for i in range(len(expected)):
        x = (truth, results,
             active_pop[i], ensemble_active[i],
             normaliser[i], expected[i])
        output.append(x)

    return output


def get_make_gain_matrix_data():
    state_ensemble = np.array([[5, 6, 4],
                               [10, 11, 9]])

    data_covariance = np.identity(2)

    H = np.identity(2)

    expected = 1/3 * np.ones((2, 2))
    return [(state_ensemble, data_covariance, H, expected)]
