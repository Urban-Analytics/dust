# Imports
from generate_data.ensemble_kalman_filter_data import *
import numpy as np
import pytest
import sys
sys.path.append('../stationsim/')

from ensemble_kalman_filter import EnsembleKalmanFilter, ActiveAgentNormaliser

# Test data
round_destination_data = get_round_destination_data()

pair_coords_data = get_pair_coords_data()

pair_coords_error_data = get_pair_coords_error_data()

separate_coords_data = get_separate_coords_data()

separate_coords_error_data = get_separate_coords_error_data()

random_destination_data = get_random_destination_data()

n_active_agents_base_data = get_n_active_agents_base_data()

n_active_agents_mean_data = get_n_active_agents_mean_data()

n_active_agents_max_data = get_n_active_agents_max_data()

n_active_agents_min_data = get_n_active_agents_min_data()

population_mean_base_data = get_population_mean_base_data()

population_mean_mean_data = get_population_mean_mean_data()

population_mean_min_data = get_population_mean_min_data()

population_mean_max_data = get_population_mean_max_data()

mean_data = get_mean_data()

# Tests
@pytest.mark.parametrize('dest, n_dest, expected', round_destination_data)
def test_round_destination(dest, n_dest, expected):
    assert EnsembleKalmanFilter.round_destination(dest, n_dest) == expected


@pytest.mark.parametrize('arr1, arr2, expected', pair_coords_data)
def test_pair_coords(arr1, arr2, expected):
    assert EnsembleKalmanFilter.pair_coords(arr1, arr2) == expected


@pytest.mark.parametrize('arr1, arr2, expected', pair_coords_error_data)
def test_pair_coords_error(arr1, arr2, expected):
    with expected:
        assert EnsembleKalmanFilter.pair_coords(arr1, arr2)


@pytest.mark.parametrize('arr, expected1, expected2', separate_coords_data)
def test_separate_coords(arr, expected1, expected2):
    assert EnsembleKalmanFilter.separate_coords(arr) == (expected1, expected2)


@pytest.mark.parametrize('arr, expected', separate_coords_error_data)
def test_separate_coords_error(arr, expected):
    with expected:
        assert EnsembleKalmanFilter.separate_coords(arr)


@pytest.mark.parametrize('gates_in, gates_out, gate_in',
                         random_destination_data)
def test_make_random_destination(gates_in, gates_out, gate_in):
    enkf = set_up_enkf()
    gate_out = enkf.make_random_destination(gates_in, gates_out, gate_in)
    assert gate_out != gate_in
    assert 0 <= gate_out < enkf.n_exits


def test_error_normalisation_default_init():
    # Set up enkf
    enkf = set_up_enkf()
    assert enkf.error_normalisation is None


@pytest.mark.parametrize('n_active, expected', n_active_agents_base_data)
def test_get_n_active_agents_base(n_active, expected):
    enkf = set_up_enkf()
    enkf.error_normalisation = ActiveAgentNormaliser.BASE
    enkf.base_model.pop_active = n_active
    assert enkf.get_n_active_agents() == expected


@pytest.mark.parametrize('ensemble_active, expected',
                         n_active_agents_mean_data)
def test_get_n_active_agents_mean_en(ensemble_active, expected):
    enkf = set_up_enkf()
    enkf.error_normalisation = ActiveAgentNormaliser.MEAN_EN
    for i, model in enumerate(enkf.models):
        model.pop_active = ensemble_active[i]
    assert enkf.get_n_active_agents() == expected


@pytest.mark.parametrize('ensemble_active, expected',
                         n_active_agents_max_data)
def test_get_n_active_agents_max_en(ensemble_active, expected):
    enkf = set_up_enkf()
    enkf.error_normalisation = ActiveAgentNormaliser.MAX_EN
    for i, model in enumerate(enkf.models):
        model.pop_active = ensemble_active[i]
    assert enkf.get_n_active_agents() == expected


@pytest.mark.parametrize('ensemble_active, expected',
                         n_active_agents_min_data)
def test_get_n_active_agents_min_en(ensemble_active, expected):
    enkf = set_up_enkf()
    enkf.error_normalisation = ActiveAgentNormaliser.MIN_EN
    for i, model in enumerate(enkf.models):
        model.pop_active = ensemble_active[i]
    assert enkf.get_n_active_agents() == expected


@pytest.mark.parametrize('results, truth, n_active, expected',
                         population_mean_base_data)
def test_get_population_mean_base(results, truth, n_active, expected):
    enkf = set_up_enkf()
    enkf.error_normalisation = ActiveAgentNormaliser.BASE
    enkf.base_model.pop_active = n_active
    results = np.array(results)
    truth = np.array(truth)

    assert enkf.get_population_mean(results, truth) == expected


@pytest.mark.parametrize('results, truth, ensemble_active, expected',
                         population_mean_mean_data)
def test_get_population_mean_mean_en(results, truth,
                                     ensemble_active, expected):
    # Set up enkf
    enkf = set_up_enkf()
    enkf.error_normalisation = ActiveAgentNormaliser.MEAN_EN

    # Define number of active agents for each ensemble member
    for i, model in enumerate(enkf.models):
        model.pop_active = ensemble_active[i]

    # Define results and truth values
    results = np.array(results)
    truth = np.array(truth)

    assert enkf.get_population_mean(results, truth) == expected


@pytest.mark.parametrize('results, truth, ensemble_active, expected',
                         population_mean_min_data)
def test_get_population_mean_min_en(results, truth,
                                    ensemble_active, expected):
    # Set up enkf
    enkf = set_up_enkf()
    enkf.error_normalisation = ActiveAgentNormaliser.MIN_EN

    # Define number of active agents for each ensemble member
    for i, model in enumerate(enkf.models):
        model.pop_active = ensemble_active[i]

    # Define results and truth values
    results = np.array(results)
    truth = np.array(truth)

    assert enkf.get_population_mean(results, truth) == expected


@pytest.mark.parametrize('results, truth, ensemble_active, expected',
                         population_mean_max_data)
def test_get_population_mean_max_en(results, truth,
                                    ensemble_active, expected):
    # Set up enkf
    enkf = set_up_enkf()
    enkf.error_normalisation = ActiveAgentNormaliser.MAX_EN

    # Define number of active agents for each ensemble member
    for i, model in enumerate(enkf.models):
        model.pop_active = ensemble_active[i]

    # Define results and truth values
    results = np.array(results)
    truth = np.array(truth)

    assert enkf.get_population_mean(results, truth) == expected


@pytest.mark.parametrize('results, truth, expected', mean_data)
def test_get_mean(results, truth, expected):
    enkf = set_up_enkf()

    results = np.array(results)
    truth = np.array(truth)

    assert enkf.get_mean(results, truth) == expected
