# Imports
import numpy as np
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
