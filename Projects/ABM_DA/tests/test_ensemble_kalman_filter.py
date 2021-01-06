# Imports
from generate_data.ensemble_kalman_filter_data import *
import numpy as np
import pytest
import sys
sys.path.append('../stationsim/')

from ensemble_kalman_filter import EnsembleKalmanFilter


# Test data
round_destination_data = get_round_destination_data()

pair_coords_data = get_pair_coords_data()

pair_coords_error_data = get_pair_coords_error_data()

separate_coords_data = get_separate_coords_data()

separate_coords_error_data = get_separate_coords_error_data()

make_random_destination_data = get_random_destination_data()


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
                         make_random_destination_data)
def test_make_random_destination(gates_in, gates_out, gate_in):
    enkf = set_up_enkf()
    gate_out = enkf.make_random_destination(gates_in, gates_out, gate_in)
    assert gate_out != gate_in
    assert 0 <= gate_out < enkf.n_exits
