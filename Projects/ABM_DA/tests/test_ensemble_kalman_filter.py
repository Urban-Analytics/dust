# Imports
from generate_data.ensemble_kalman_filter_data import *
from utils import *

import numpy as np
import pytest
import sys
sys.path.append('../stationsim/')

from ensemble_kalman_filter import EnsembleKalmanFilter
# from ensemble_kalman_filter import ActiveAgentNormaliser
from ensemble_kalman_filter import AgentIncluder
from ensemble_kalman_filter import GateEstimator
from ensemble_kalman_filter import Inflation
from ensemble_kalman_filter import ExitRandomisation

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

# error_normalisation_type_data = get_error_normalisation_type_data()

distance_error_default_data = get_distance_error_default_data()

distance_error_base_data = get_distance_error_base_data()

calculate_rmse_default_data = get_calculate_rmse_default_data()

# make_obs_error_data = get_make_obs_error_data()

make_gain_matrix_data = get_make_gain_matrix_data()

separate_coords_exits_data = get_separate_coords_exits_data()

update_status_data = get_update_status_data()

make_noise_data = get_make_noise_data()

update_state_mean_data = get_update_state_mean_data()

np_cov_data = get_np_cov_data()

destination_vector_data = get_destination_vector_data()

origin_vector_data = get_origin_vector_data()

agent_statuses_data = get_agent_statuses_data()

filter_vector_data = get_filter_vector_data()

state_vector_statuses_data = get_state_vector_statuses_data()

forecast_error_data = get_forecast_error_data()

gate_estimator_allocation_data = get_gate_estimator_allocation_data()

get_angle_data = get_get_angle_data()

edge_angle_data = get_edge_angle_data()

reverse_bisect_data = get_reverse_bisect_data()

edge_loc_data = get_edge_loc_data()

angle_destination_out_data = get_angle_destination_out_data()

angle_destination_out_gate_data = get_angle_destination_out_gate_data()

angle_destination_in_data = get_angle_destination_in_data()

construct_state_from_angles_gates_data = get_construct_state_from_angles_gates_data()

construct_state_from_angles_locs_data = get_construct_state_from_angles_locs_data()

# angle_destination_in_data = get_angle_destination_in_data()

round_target_angle_data = get_round_target_angle_data()

convert_vector_angle_to_gate_data = get_convert_vector_angle_to_gate_data()

process_state_vector_data = get_process_state_vector_data()

mod_angles_data = get_mod_angles_data()

multi_gain_data = get_multi_gain_data()

exit_randomisation_adjacent_data = get_exit_randomisation_adjacent_data()

standardisation_data = get_standardisation_data()

unstandardisation_data = get_unstandardisation_data()

alternating_to_sequential_data = get_alternating_to_sequential_data()

update_data = get_update_data()

reformat_obs_data = get_reformat_obs_data()

standardise_ensemble_data = get_standardise_ensemble_data()

unstandardise_ensemble_data = get_unstandardise_ensemble_data()


# Tests
@pytest.mark.parametrize('dest, n_dest, expected', round_destination_data)
def test_round_destination(dest, n_dest, expected):
    """
    Test EnsembleKalmanFilter.round_destination()

    Test that the round_destination method rounds the destination exit number
    to the nearest integer value modulo the number of gates.

    Parameters
    ----------
    dest : float
        Floating point value of the estimated destination number
    n_dest : int
        The integer number of possible destination gates
    expected : int
        Expected result of round_destination
    """
    assert EnsembleKalmanFilter.round_destination(dest, n_dest) == expected


@pytest.mark.parametrize('arr1, arr2, expected', pair_coords_data)
def test_pair_coords(arr1, arr2, expected):
    """
    Test EnsembleKalmanFilter.pair_coords()

    Test that the pair_coords method returns an appropriate list when provided
    with two list-like input arrays, i.e. given
        arr1 = [x0, x1, ..., xN]
        arr2 = [y0, y1, ..., yN]

    the method returns a list of
        [x0, y0, x1, y1, ..., xN, yN]

    Parameters
    ----------
    arr1 : list-like
        First list of elements
    arr2 : list-like
        Second list of elements
    expected : list
        Expected resulting list of pair_coords() as outlined above
    """
    assert EnsembleKalmanFilter.pair_coords(arr1, arr2) == expected


@pytest.mark.parametrize('arr1, arr2, expected', pair_coords_error_data)
def test_pair_coords_error(arr1, arr2, expected):
    """
    Test that EnsembleKalmanFilter.pair_coords() throws an appropriate error

    Test that, when provided with arrays of incompatible length, pair_coords()
    throws an appropriate error.

    Parameters
    ----------
    arr1 : list-like
        First list of elements
    arr2 : list-like
        Second list of elements
    expected : Exception
        Expected ValueError exception
    """
    with expected:
        assert EnsembleKalmanFilter.pair_coords(arr1, arr2)


@pytest.mark.parametrize('arr, expected1, expected2', separate_coords_data)
def test_separate_coords(arr, expected1, expected2):
    """
    Test that EnsembleKalmanFilter.separate_coords() splits coordinate array
    into two arrays

    Test that separate_coords() splits an array like
        [x0, y0, x1, y1, ..., xN, yN]

    into two arrays like
        [x0, x1, ..., xN]
        [y0, y1, ..., yN]

    Parameters
    ----------
    arr : list-like
        Input array of alternating x-y's
    expected1 : list-like
        Array of x-coordinates
    expected2 : list-like
        Array of y-coordinates
    """
    assert EnsembleKalmanFilter.separate_coords(arr) == (expected1, expected2)


@pytest.mark.parametrize('arr, expected', separate_coords_error_data)
def test_separate_coords_error(arr, expected):
    """
    Test that EnsembleKalmanFilter.separate_coords() throws an appropriate
    error

    Test that, when provided with an array of odd length, separate_coords()
    throws an appropriate error

    Parameters
    ----------
    arr : list-like
        Input list
    expected : Exception
        Expected ValueError exception
    """
    with expected:
        assert EnsembleKalmanFilter.separate_coords(arr)


@pytest.mark.parametrize('gates_in, gates_out, gate_in, excluded_gates',
                         random_destination_data)
def test_make_random_destination(gates_in, gates_out, gate_in, excluded_gates):
    enkf = set_up_enkf()
    gate_out = enkf.make_random_destination(gates_in, gates_out, gate_in)
    assert gate_out != gate_in
    assert 0 <= gate_out < enkf.n_exits
    assert gate_out not in excluded_gates


def test_error_normalisation_default_init():
    # Set up enkf
    enkf = set_up_enkf()
    assert enkf.error_normalisation is None


# @pytest.mark.parametrize('n_active, expected', n_active_agents_base_data)
# def test_get_n_active_agents_base(n_active, expected):
#     enkf = set_up_enkf()
#     enkf.error_normalisation = ActiveAgentNormaliser.BASE
#     enkf.base_model.pop_active = n_active
#     assert enkf.get_n_active_agents() == expected


# @pytest.mark.parametrize('ensemble_active, expected',
#                          n_active_agents_mean_data)
# def test_get_n_active_agents_mean_en(ensemble_active, expected):
#     enkf = set_up_enkf()
#     enkf.error_normalisation = ActiveAgentNormaliser.MEAN_EN
#     for i, model in enumerate(enkf.models):
#         model.pop_active = ensemble_active[i]
#     assert enkf.get_n_active_agents() == expected


# @pytest.mark.parametrize('ensemble_active, expected',
#                          n_active_agents_max_data)
# def test_get_n_active_agents_max_en(ensemble_active, expected):
#     enkf = set_up_enkf()
#     enkf.error_normalisation = ActiveAgentNormaliser.MAX_EN
#     for i, model in enumerate(enkf.models):
#         model.pop_active = ensemble_active[i]
#     assert enkf.get_n_active_agents() == expected


# @pytest.mark.parametrize('ensemble_active, expected',
#                          n_active_agents_min_data)
# def test_get_n_active_agents_min_en(ensemble_active, expected):
#     enkf = set_up_enkf()
#     enkf.error_normalisation = ActiveAgentNormaliser.MIN_EN
#     for i, model in enumerate(enkf.models):
#         model.pop_active = ensemble_active[i]
#     assert enkf.get_n_active_agents() == expected


# @pytest.mark.parametrize('diffs, n_active, expected',
#                          population_mean_base_data)
# def test_get_population_mean_base(diffs, n_active, expected):
#     enkf = set_up_enkf()
#     enkf.error_normalisation = ActiveAgentNormaliser.BASE
#     enkf.base_model.pop_active = n_active
#     diffs = np.array(diffs)

#     assert enkf.get_population_mean(diffs) == expected


# @pytest.mark.parametrize('diffs, ensemble_active, expected',
#                          population_mean_mean_data)
# def test_get_population_mean_mean_en(diffs, ensemble_active, expected):
#     # Set up enkf
#     enkf = set_up_enkf()
#     enkf.error_normalisation = ActiveAgentNormaliser.MEAN_EN

#     # Define number of active agents for each ensemble member
#     for i, model in enumerate(enkf.models):
#         model.pop_active = ensemble_active[i]

#     # Define results and truth values
#     diffs = np.array(diffs)

#     assert enkf.get_population_mean(diffs) == expected


# @pytest.mark.parametrize('diffs, ensemble_active, expected',
#                          population_mean_min_data)
# def test_get_population_mean_min_en(diffs, ensemble_active, expected):
#     # Set up enkf
#     enkf = set_up_enkf()
#     enkf.error_normalisation = ActiveAgentNormaliser.MIN_EN

#     # Define number of active agents for each ensemble member
#     for i, model in enumerate(enkf.models):
#         model.pop_active = ensemble_active[i]

#     # Define results and truth values
#     diffs = np.array(diffs)

#     assert enkf.get_population_mean(diffs) == expected


# @pytest.mark.parametrize('diffs, ensemble_active, expected',
#                          population_mean_max_data)
# def test_get_population_mean_max_en(diffs, ensemble_active, expected):
#     # Set up enkf
#     enkf = set_up_enkf()
#     enkf.error_normalisation = ActiveAgentNormaliser.MAX_EN

#     # Define number of active agents for each ensemble member
#     for i, model in enumerate(enkf.models):
#         model.pop_active = ensemble_active[i]

#     # Define results and truth values
#     diffs = np.array(diffs)

#     assert enkf.get_population_mean(diffs) == expected


@pytest.mark.parametrize('results, truth, expected', mean_data)
def test_get_mean(results, truth, expected):
    enkf = set_up_enkf()

    results = np.array(results)
    truth = np.array(truth)

    assert enkf.get_mean_error(results, truth) == expected


# x = '''error_normalisation, results, truth, active_pop, ensemble_active,
#        expected'''


# @pytest.mark.parametrize(x, error_normalisation_type_data)
# def test_error_normalisation_type(error_normalisation, results, truth,
#                                   active_pop, ensemble_active, expected):
#     enkf = set_up_enkf(error_normalisation=error_normalisation)

#     results = np.array(results)
#     truth = np.array(truth)

#     for i, model in enumerate(enkf.models):
#         model.pop_active = ensemble_active[i]

#     enkf.base_model.pop_active = active_pop

#     assert enkf.get_mean_error(results, truth) == expected


@pytest.mark.parametrize('x_error, y_error, expected',
                         distance_error_default_data)
def test_make_distance_error_default(x_error, y_error, expected):
    enkf = set_up_enkf()

    x_error = np.array(x_error)
    y_error = np.array(y_error)

    assert pytest.approx(expected) == enkf.make_distance_error(x_error,
                                                               y_error)


# @pytest.mark.parametrize('x_error, y_error, n_active, expected',
#                          distance_error_base_data)
# def test_make_distance_error_base(x_error, y_error, n_active, expected):
#     enkf = set_up_enkf(error_normalisation=ActiveAgentNormaliser.BASE)
#     enkf.base_model.pop_active = n_active

#     assert pytest.approx(expected) == enkf.make_distance_error(x_error,
#                                                                y_error)


@pytest.mark.parametrize('x_truth, y_truth, x_result, y_result, expected',
                         calculate_rmse_default_data)
def test_calculate_rmse_default(x_truth, y_truth,
                                x_result, y_result, expected):
    # Setup
    enkf = set_up_enkf()

    # Convert all inputs to arrays
    x_truth = np.array(x_truth)
    x_result = np.array(x_result)
    y_truth = np.array(y_truth)
    y_result = np.array(y_result)

    results = enkf.calculate_rmse(x_truth, y_truth, x_result, y_result)
    assert results[0] == expected


# x = 'truth, result, active_pop, ensemble_active, normaliser, expected'

# @pytest.mark.parametrize(x, make_obs_error_data)
# def test_make_obs_error(truth, result,
#                         active_pop, ensemble_active,
#                         normaliser, expected):
#     """
#     Test that enkf.make_obs_error() correctly calculates observation errors,
#     ignoring changes to how many agents are active in the base model and the
#     ensemble-member models.
#     """
#     # Setup
#     enkf = set_up_enkf(error_normalisation=normaliser)

#     # Set active agents
#     enkf.base_model.pop_active = active_pop
#     for i, model in enumerate(enkf.models):
#         model.pop_active = ensemble_active[i]

#     # Convert to arrays where necessary
#     truth = np.array(truth)
#     result = np.array(result)

#     # Assertion
#     assert enkf.make_obs_error(truth, result) == expected


@pytest.mark.parametrize('state_ensemble, data_covariance, H, expected',
                         make_gain_matrix_data)
def test_make_gain_matrix(state_ensemble, data_covariance, H, expected):
    enkf = set_up_enkf()
    H_transpose = H.T
    result = enkf.make_gain_matrix(state_ensemble,
                                   data_covariance,
                                   H, H_transpose)
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize('state_vector, pop_size, expected',
                         separate_coords_exits_data)
def test_separate_coords_exits(state_vector, pop_size, expected):
    enkf = set_up_enkf()
    enkf.population_size = pop_size
    np.testing.assert_array_equal

    result = enkf.separate_coords_exits(pop_size, state_vector)

    for i in range(len(result)):
        np.testing.assert_array_equal(result[i], expected[i])


@pytest.mark.parametrize('m_statuses, filter_status, expected',
                         update_status_data)
def test_update_status(m_statuses, filter_status, expected):
    enkf = set_up_enkf()

    enkf.active = filter_status

    for i, s in enumerate(m_statuses):
        enkf.models[i].status = s

    enkf.update_status()

    assert enkf.active == expected


@pytest.mark.parametrize('array, expected', np_cov_data)
def test_np_cov(array, expected):
    result = np.cov(array)
    assert result.shape == expected.shape
    np.testing.assert_array_almost_equal(result, expected)


@pytest.mark.parametrize('shape, R_vector', make_noise_data)
def test_make_noise(shape, R_vector):
    enkf = set_up_enkf()

    if isinstance(shape, int):
        assert len(R_vector) == shape
    elif isinstance(shape, tuple):
        assert (len(R_vector) == shape[0]) or (len(R_vector) == shape[1])

    all_samples = list()

    for i in range(1000):
        # Pass seed for test reproducibility
        noise = enkf.make_noise(shape, R_vector, seed=i)
        all_samples.append(noise)

    all_samples = np.array(all_samples)

    for i in range(shape):
        m = np.mean(all_samples[:, i])
        s = np.std(all_samples[:, i])
        assert m == pytest.approx(0, abs=0.1)
        assert s == pytest.approx(R_vector[i], 0.2)


@pytest.mark.parametrize('state_ensemble, expected', update_state_mean_data)
def test_update_state_mean(state_ensemble, expected):
    enkf = set_up_enkf()

    result = enkf.update_state_mean(state_ensemble)

    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize('agent_destinations, expected',
                         destination_vector_data)
def test_make_base_destination_vector(agent_destinations, expected):
    enkf = set_up_enkf()

    for i, agent in enumerate(enkf.base_model.agents):
        agent.loc_desire = np.array(agent_destinations[i])

    result = enkf.make_base_destinations_vector()
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize('origins, statuses, expected', origin_vector_data)
def test_make_base_origin_vector(origins, statuses, expected):
    enkf = set_up_enkf()

    for i, agent in enumerate(enkf.base_model.agents):
        agent.status = statuses[i]
        agent.loc_start = origins[i]

    result = enkf.make_base_origins_vector()
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize('base_statuses, en_statuses, inclusion, expected',
                         agent_statuses_data)
def test_agent_statuses_data(base_statuses, en_statuses, inclusion, expected):
    enkf = set_up_enkf(pop_size=3, ensemble_size=3, agent_inclusion=inclusion)

    # Check we have as many agents as we do statuses
    assert len(enkf.base_model.agents) == len(base_statuses)
    # Check we have as many ensemble members as we do status vectors
    assert enkf.ensemble_size == len(en_statuses)
    # Check we have as many agents in each model as we do statuses
    for i in range(len(en_statuses)):
        assert len(en_statuses[i]) == len(enkf.models[i].agents)


@pytest.mark.parametrize('base_statuses, en_statuses, inclusion, expected',
                         agent_statuses_data)
def test_get_agent_statuses(base_statuses, en_statuses, inclusion, expected):
    enkf = set_up_enkf(pop_size=3, ensemble_size=3, agent_inclusion=inclusion)

    # Set statuses of agents in base model
    for i, status in enumerate(base_statuses):
        enkf.base_model.agents[i].status = status

    # Set statuses of agents in ensemble member models
    for i, statuses in enumerate(en_statuses):
        for j, status in enumerate(statuses):
            enkf.models[i].agents[j].status = status

    result = enkf.get_agent_statuses()

    assert result == expected


@pytest.mark.parametrize('vector, statuses, expected', filter_vector_data)
def test_filter_vector(vector, statuses, expected):
    enkf = set_up_enkf()

    result = enkf.filter_vector(vector, statuses)
    np.testing.assert_array_equal(result, expected)


x = 'base_statuses, en_statuses, inclusion, vector_mode, expected'


@pytest.mark.parametrize(x, state_vector_statuses_data)
def test_get_state_vector_statuses(base_statuses, en_statuses, inclusion,
                                   vector_mode, expected):
    enkf = set_up_enkf(pop_size=3, ensemble_size=3, agent_inclusion=inclusion)

    # Set statuses of agents in base model
    for i, status in enumerate(base_statuses):
        enkf.base_model.agents[i].status = status

    # Set statuses of agents in ensemble member models
    for i, statuses in enumerate(en_statuses):
        for j, status in enumerate(statuses):
            enkf.models[i].agents[j].status = status

    result = enkf.get_state_vector_statuses(vector_mode)

    assert result == expected


x = 'inclusion, base_statuses, ensemble_statuses, truth, state_mean, expected'


@pytest.mark.parametrize(x, forecast_error_data)
def test_get_forecast_error(inclusion, base_statuses, ensemble_statuses,
                            truth, state_mean, expected):
    # For now, we assume that we are only dealing with state estimation
    # This means that we know that the following will be in place
    # enkf.mode = EnsembleKalmanFilterType.STATE
    # enkf.error_func = enkf.make_errors
    enkf = set_up_enkf(pop_size=3, ensemble_size=3, agent_inclusion=inclusion)

    # Provide initial values for enkf
    # Filter state_mean
    enkf.state_mean = state_mean

    # Base model agent statuses
    enkf.set_base_statuses(base_statuses)
    # for i, agent in enumerate(enkf.base_model.agents):
    #     agent.status = base_statuses[i]

    # Ensemble models agent statuses
    for i, model in enumerate(enkf.models):
        for j, agent in enumerate(model.agents):
            agent.status = ensemble_statuses[i][j]

    result = enkf.get_forecast_error(truth)

    assert result == expected


@pytest.mark.parametrize('estimator_type, expected',
                         gate_estimator_allocation_data)
def test_gate_estimator_allocation(estimator_type, expected):
    if (estimator_type == GateEstimator.ROUNDING or estimator_type ==
            GateEstimator.ANGLE):
        ft = EnsembleKalmanFilterType.DUAL_EXIT
    else:
        ft = EnsembleKalmanFilterType.STATE
    enkf = set_up_enkf(gate_estimator=estimator_type, filter_type=ft)
    # enkf = set_up_enkf(gate_estimator=estimator_type)
    assert enkf.gate_estimator == expected


@pytest.mark.parametrize('vector_tail, vector_head, expected',
                         get_angle_data)
def test_get_angle(vector_tail, vector_head, expected):
    enkf = set_up_enkf()
    assert enkf.get_angle(vector_tail, vector_head) == expected


@pytest.mark.parametrize('expected', edge_angle_data)
def test_edge_angle_setup(expected):
    enkf = set_up_enkf(gate_estimator=GateEstimator.ANGLE)

    edge_angles = list()
    for _, gate_angles in enkf.gate_angles.items():
        edge_angles.extend(gate_angles)

    unique_edge_angles = list(set(edge_angles))
    unique_edge_angles.sort(reverse=True)

    assert unique_edge_angles == expected


@pytest.mark.parametrize('expected', edge_angle_data)
def test_unique_edge_angles(expected):
    enkf = set_up_enkf(gate_estimator=GateEstimator.ANGLE)

    unique_edge_angles = list(enkf.unique_gate_angles)
    unique_edge_angles.sort(reverse=True)

    assert unique_edge_angles == expected


@pytest.mark.parametrize('element, iterable, expected', reverse_bisect_data)
def test_reverse_bisect(element, iterable, expected):
    enkf = set_up_enkf()

    result = enkf.bisect_left_reverse(element, iterable)

    assert result == expected


@pytest.mark.parametrize('expected', edge_loc_data)
def test_unique_edge_locs(expected):
    enkf = set_up_enkf(gate_estimator=GateEstimator.ANGLE)

    unique_edge_locs = list(enkf.unique_gate_edges)
    assert unique_edge_locs == expected


@pytest.mark.parametrize('angle, expected', angle_destination_out_data)
def test_angle_destination_out(angle, expected):
    enkf = set_up_enkf(gate_estimator=GateEstimator.ANGLE)

    result = enkf.get_destination_angle(angle)

    assert result == expected


@pytest.mark.parametrize('angle, expected', angle_destination_out_gate_data)
def test_angle_destination_in_gate(angle, expected):
    enkf = set_up_enkf(gate_estimator=GateEstimator.ANGLE)

    _, gate = enkf.get_destination_angle(angle, gate_out=True)

    assert gate == expected


@pytest.mark.parametrize('angle, expected', angle_destination_in_data)
def test_angle_destination_in(angle, expected):
    enkf = set_up_enkf(gate_estimator=GateEstimator.ANGLE)

    # Use same offset from wall as in Agent.set_agent_location()
    offset = enkf.base_model.agents[0].size * 1.05

    result = enkf.get_destination_angle(angle)

    if isinstance(expected[0], tuple):
        assert pytest.approx(result[1], offset) == expected[1]
        assert expected[0][0] <= result[0] <= expected[0][1]
    elif isinstance(expected[1], tuple):
        assert pytest.approx(result[0], offset) == expected[0]
        assert expected[1][0] <= result[1] <= expected[1][1]
    else:
        raise ValueError(f'Unexpected test value provided: {expected}')


@pytest.mark.parametrize('angle, insertion_idx, expected',
                         round_target_angle_data)
def test_round_target_angle(angle, insertion_idx, expected):
    enkf = set_up_enkf(gate_estimator=GateEstimator.ANGLE)

    result = enkf.round_target_angle(angle, insertion_idx)

    assert result == expected


@pytest.mark.parametrize('angles, expected',
                         construct_state_from_angles_locs_data)
def test_construct_state_from_angles_locs(angles, expected):
    enkf = set_up_enkf(pop_size=3, gate_estimator=GateEstimator.ANGLE)

    _, locations = enkf.construct_state_from_angles(angles)

    assert list(locations) == expected


@pytest.mark.parametrize('angles, expected',
                         construct_state_from_angles_gates_data)
def test_construct_state_from_angles_gates(angles, expected):
    enkf = set_up_enkf(pop_size=3, gate_estimator=GateEstimator.ANGLE)

    gates, _ = enkf.construct_state_from_angles(angles)

    assert list(gates) == expected


@pytest.mark.parametrize('state_vector, expected',
                         convert_vector_angle_to_gate_data)
def test_convert_vector_angle_to_gate(state_vector, expected):
    enkf = set_up_enkf(pop_size=3, gate_estimator=GateEstimator.ANGLE)

    gate_state_vector = enkf.convert_vector_angle_to_gate(state_vector)

    np.testing.assert_array_equal(gate_state_vector, expected)


@pytest.mark.parametrize('state, filter_mode, gate_estimator, expected',
                         process_state_vector_data)
def test_process_state_vector(state, filter_mode, gate_estimator, expected):
    enkf = set_up_enkf(pop_size=3, gate_estimator=gate_estimator,
                       filter_type=filter_mode)

    state_vector = enkf.process_state_vector(state)

    np.testing.assert_array_equal(state_vector, expected)


@pytest.mark.parametrize('angles, expected', mod_angles_data)
def test_mod_angles(angles, expected):
    enkf = set_up_enkf()

    modded_angles = enkf.mod_angles(angles)

    np.testing.assert_array_almost_equal(modded_angles, expected)


@pytest.mark.parametrize('state, data_cov, H, inf_rate, expected',
                         multi_gain_data)
def test_multi_gain(state, data_cov, H, inf_rate, expected):
    enkf = set_up_enkf()
    enkf.inflation = Inflation.MULTIPLICATIVE
    enkf.inflation_rate = inf_rate

    gain_matrix = enkf.make_gain_matrix(state, data_cov, H, H.T)

    np.testing.assert_array_almost_equal(gain_matrix, expected)


def test_exit_randomisation_by_agent():
    enkf = set_up_enkf(exit_randomisation=ExitRandomisation.BY_AGENT)

    for i in range(len(enkf.base_model.agents)):
        agent_gate_out = enkf.models[0].agents[i].gate_out

        for _, model in enumerate(enkf.models):
            assert model.agents[i].gate_out == agent_gate_out


@pytest.mark.parametrize('n_adjacent', exit_randomisation_adjacent_data)
def test_exit_randomisation_adjacent(n_adjacent):
    enkf = set_up_enkf(exit_randomisation=ExitRandomisation.ADJACENT,
                       n_adjacent=n_adjacent)

    adjacent_range = list(range(-n_adjacent, n_adjacent+1))
    n_gates = enkf.base_model.gates_out

    for i, agent in enumerate(enkf.base_model.agents):
        base_gate_out = agent.gate_out

        gate_range = [(base_gate_out + x) % n_gates for x in adjacent_range]

        for model in enkf.models:
            model_gate_out = model.agents[i].gate_out
            assert model_gate_out in gate_range


@pytest.mark.parametrize('n_adjacent', exit_randomisation_adjacent_data)
def test_exit_randomisation_adjacent_range(n_adjacent):
    n_runs = 100
    pop_size = 3
    ensemble_size = 25
    min_results = [[] for _ in range(pop_size)]
    max_results = [[] for _ in range(pop_size)]
    adjacent_range = list(range(-n_adjacent, n_adjacent+1))


    for _ in range(n_runs):
        enkf = set_up_enkf(exit_randomisation=ExitRandomisation.ADJACENT,
                           n_adjacent=n_adjacent, pop_size=pop_size,
                           ensemble_size=ensemble_size)
        # base_gates = [agent.gate_out for agent in enkf.base_model.agents]
        n_gates = enkf.base_model.gates_out

        for i in range(pop_size):
            base_gate_out = enkf.base_model.agents[i].gate_out
            gate_range = [(base_gate_out + x) % n_gates for x in adjacent_range]
            max_gate = max(gate_range)
            min_gate = min(gate_range)

            ensemble_gates_out = list()

            for model in enkf.models:
                ensemble_gates_out.append(model.agents[i].gate_out)

            min_correct = [gate == min_gate for gate in ensemble_gates_out]
            max_correct = [gate == max_gate for gate in ensemble_gates_out]

            min_results[i].append(any(min_correct))
            max_results[i].append(any(max_correct))

    for i in range(pop_size):
        min_proportion = sum(min_results[i]) / len(min_results[i])
        max_proportion = sum(max_results[i]) / len(max_results[i])

        assert min_proportion > 0.5
        assert max_proportion > 0.5


@pytest.mark.parametrize('state_vector, top, bottom, expected',
                         standardisation_data)
def test_standardisation(state_vector, top, bottom, expected):
    enkf = set_up_enkf()

    # Set state
    result = enkf.standardise(state_vector, top, bottom)

    # Test
    np.testing.assert_equal(result, expected)


@pytest.mark.parametrize('state_vector, top, bottom, expected',
                         unstandardisation_data)
def test_unstandardisation(state_vector, top, bottom, expected):
    enkf = set_up_enkf()

    result = enkf.unstandardise(state_vector, top, bottom)

    np.testing.assert_equal(result, expected)


@pytest.mark.parametrize('state_vector, expected',
                         alternating_to_sequential_data)
def test_alternating_to_sequential(state_vector, expected):
    enkf = set_up_enkf()

    result = enkf.convert_alternating_to_sequential(state_vector)

    np.testing.assert_equal(result, expected)


@pytest.mark.parametrize('ft, state_ensemble, data_cov, data, H, expected',
                         update_data)
def test_update(ft, state_ensemble, data_cov, data, H, expected):
    # Derive params
    pop_size = len(data_cov)
    ensemble_size = state_ensemble.shape[1]

    # Set up enkf
    enkf = set_up_enkf(pop_size=pop_size, ensemble_size=ensemble_size,
                       filter_type=ft)

    # Assign attributes
    enkf.state_ensemble = state_ensemble
    enkf.data_covariance = data_cov
    enkf.H = H
    enkf.H_transpose = H.T
    enkf.data_vector_length = len(data_cov.diagonal())

    # Run update
    enkf.update(data)

    # Assert updated ensemble is expected value
    np.testing.assert_allclose(enkf.state_ensemble, expected)


@pytest.mark.parametrize('dvector_length, ensemble_size, data, expected',
                         reformat_obs_data)
def test_reformat_obs(dvector_length, ensemble_size, data, expected):
    # Set up filter
    enkf = set_up_enkf(ensemble_size=ensemble_size)

    # Assign attributes
    enkf.data_vector_length = dvector_length

    # Reformat data
    result = enkf.reformat_obs(data)

    np.testing.assert_equal(result, expected)


@pytest.mark.parametrize('state, pop_size, en_size, n_var, gate_est, expected',
                         standardise_ensemble_data)
def test_standardise_ensemble(state, pop_size, en_size, n_var, gate_est,
                              expected):
    enkf = set_up_enkf(ensemble_size=en_size, pop_size=pop_size,
                       gate_estimator=gate_est)

    result = enkf.standardise_ensemble(state, n_var)

    np.testing.assert_equal(result, expected)


@pytest.mark.parametrize('state, pop_size, en_size, n_var, gate_est, expected',
                         unstandardise_ensemble_data)
def test_unstandardise_ensemble(state, pop_size, en_size, n_var, gate_est,
                                expected):
    enkf = set_up_enkf(ensemble_size=en_size, pop_size=pop_size,
                       gate_estimator=gate_est)

    result = enkf.unstandardise_ensemble(state, n_var)

    np.testing.assert_equal(result, expected)
