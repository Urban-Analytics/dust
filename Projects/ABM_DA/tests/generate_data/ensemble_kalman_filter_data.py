# Imports
from math import sqrt
import numpy as np
import pytest
import sys
sys.path.append('../stationsim/')

from generate_data.data_utils import wrap_up
from ensemble_kalman_filter import EnsembleKalmanFilter
from ensemble_kalman_filter import EnsembleKalmanFilterType
from ensemble_kalman_filter import ActiveAgentNormaliser
from stationsim_gcs_model import Model


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

    output = wrap_up([diffs, active_pops, expected])
    return output


def get_population_mean_mean_data():
    diffs = ([0, 3, 1, 4, 2],
             [0, 0, 3, 3, 0],
             [1, 1, 1, 1, 1])

    ensemble_actives = ([4, 4, 4, 4, 4],
                        [1, 2, 3, 2, 3],
                        [0, 0, 1, 0, 0])

    expected = (2.5, 3, 0)

    output = wrap_up([diffs, ensemble_actives, expected])
    return output


def get_population_mean_max_data():
    diffs = ([0, 3, 1, 4, 2],
             [0, 0, 3, 3, 0],
             [1, 1, 1, 1, 1])

    ensemble_actives = ([4, 4, 4, 4, 4],
                        [1, 2, 3, 2, 3],
                        [0, 0, 1, 0, 0])

    expected = (2.5, 2, 5)

    output = wrap_up([diffs, ensemble_actives, expected])
    return output


def get_population_mean_min_data():
    diffs = ([0, 3, 1, 4, 2],
             [0, 0, 3, 3, 0],
             [1, 1, 1, 1, 1])

    ensemble_actives = ([4, 4, 4, 4, 4],
                        [1, 2, 3, 2, 3],
                        [0, 0, 1, 0, 0])

    expected = (2.5, 6, 0)

    output = wrap_up([diffs, ensemble_actives, expected])
    return output


def get_mean_data():
    results = ([2, 5, 3, 6, 4],
               [3, 6, 9, 12, 15],
               [0, 0, 0, 0, 0])

    truths = ([2, 2, 2, 2, 2],
              [3, 6, 6, 9, 15],
              [1, 1, 1, 1, 1])

    expected = (2, 1.2, 1)

    output = wrap_up([results, truths, expected])
    return output


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

    output = wrap_up([x_errors, y_errors, expected])
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

    output = wrap_up([x_errors, y_errors, n_active, expected])
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

    output = wrap_up([x_truths, y_truths, x_results, y_results, expected])
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
    x = np.array([[5, 6, 4],
                  [10, 11, 9]])
    y = np.array([[11, 10, 10, 9, 10],
                  [5, 6, 4, 4, 6],
                  [21, 15, 18, 18, 18],
                  [0, 0, 0, 0, 0]])
    state_ensembles = (x, y)

    data_covariances = (np.identity(2), np.identity(4))

    Hs = (np.identity(2), np.identity(4))

    ex_x = 1/3 * np.ones((2, 2))
    ex_y = np.array([[0.2494382, 0.13932584, 0.12134831, 0.0],
                     [0.13932584, 0.44719101, -0.09438202, 0.0],
                     [0.12134831, -0.09438202, 0.78876404, 0.0],
                     [0.0, 0.0, 0.0, 0.0]])
    expected = (ex_x, ex_y)

    output = wrap_up([state_ensembles, data_covariances, Hs, expected])
    return output


def get_separate_coords_exits_data():
    state_vectors = [np.array([10, 20, 3]),
                     np.array([10, 12, 15,
                               20, 35, 25,
                               3, 4, 5]),
                     np.array([10, 12, 15, 9, 5,
                               20, 35, 25, 4, 13,
                               3, 4, 5, 7, 2])]

    pop_sizes = (1, 3, 5)

    expected = [(np.array([10]), np.array([20]), np.array([3])),
                (np.array([10, 12, 15]),
                 np.array([20, 35, 25]),
                 np.array([3, 4, 5])),
                (np.array([10, 12, 15, 9, 5]),
                 np.array([20, 35, 25, 4, 13]),
                 np.array([3, 4, 5, 7, 2]))]

    output = wrap_up([state_vectors, pop_sizes, expected])
    return output


def get_update_status_data():
    m_statuses = [[1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [1, 0, 1, 0, 1],
                  [1, 0, 1, 0, 1]]

    filter_status = (True, False,
                     True, False,
                     True, False)

    expected = (True, True,
                False, False,
                True, True)

    output = wrap_up([m_statuses, filter_status, expected])
    return output
