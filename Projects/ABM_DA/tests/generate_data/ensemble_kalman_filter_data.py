# Imports
from cmath import phase
from math import pi, sqrt
import numpy as np
import pytest
import sys
sys.path.append('../stationsim/')

from generate_data.data_utils import wrap_up, get_angle
from ensemble_kalman_filter import EnsembleKalmanFilter
from ensemble_kalman_filter import EnsembleKalmanFilterType
# from ensemble_kalman_filter import ActiveAgentNormaliser
from ensemble_kalman_filter import AgentIncluder
from stationsim_gcs_model import Model
from ensemble_kalman_filter import GateEstimator


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
    gate_ins = [0, 1, 3, 7]
    excluded_gates = [{0},
                      {1, 2},
                      {3, 4, 5, 6},
                      {7, 8, 9, 10}]

    gates_in = [10 for _ in range(len(gate_ins))]
    gates_out = [10 for _ in range(len(gate_ins))]

    output = wrap_up((gates_in, gates_out, gate_ins, excluded_gates))
    return output


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


# def get_error_normalisation_type_data():
#     error_normalisations = (None,
#                             ActiveAgentNormaliser.BASE,
#                             ActiveAgentNormaliser.MEAN_EN)

#     results = [2, 5, 3, 6, 4]

#     truths = [2, 2, 2, 2, 2]

#     active_pop = 4

#     ensemble_active = [1, 2, 2, 2, 3]

#     expected = (2, 2.5, 5)

#     d = list()

#     for i in range(len(error_normalisations)):
#         x = (error_normalisations[i], results, truths,
#              active_pop, ensemble_active, expected[i])
#         d.append(x)

#     return d


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


# def get_make_obs_error_data():
#     truth = [1, 2, 6, 6, 12, 15]

#     results = [1, 1, 6, 6, 15, 19]

#     active_pop = (3, 4, 5)

#     ensemble_active = ([2, 3, 4, 5, 3],
#                        [1, 2, 1, 2, 2],
#                        [1, 1, 2, 1, 1])

#     normaliser = (None,
#                   ActiveAgentNormaliser.BASE,
#                   ActiveAgentNormaliser.MEAN_EN)

#     expected = (2, 2, 2)

#     output = list()

#     for i in range(len(expected)):
#         x = (truth, results,
#              active_pop[i], ensemble_active[i],
#              normaliser[i], expected[i])
#         output.append(x)

#     return output


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
    m_statuses = [(1, 1, 1, 1, 1),
                  (1, 1, 1, 1, 1),
                  (0, 0, 0, 0, 0),
                  (0, 0, 0, 0, 0),
                  (1, 0, 1, 0, 1),
                  (1, 0, 1, 0, 1)]

    filter_status = (True, False,
                     True, False,
                     True, False)

    expected = (True, True,
                False, False,
                True, True)

    output = wrap_up([m_statuses, filter_status, expected])
    return output


def get_make_noise_data():
    shapes = (10, 20)

    R_vectors = (np.ones(shapes[0]),
                 np.full(shapes[1], 1.5))

    output = wrap_up([shapes, R_vectors])
    return output


def get_update_state_mean_data():
    state_ensembles = (np.array([[11, 10, 9, 9, 11],
                                 [5, 5, 5, 4, 6],
                                 [21, 15, 18, 18, 18],
                                 [0, 0, 0, 0, 0]]),
                       np.array([[1, 1, 1, 1, 1],
                                 [2, 2, 2, 2, 2],
                                 [1, 2, 3, 4, 5]]))

    expected = (np.array([10, 5, 18, 0]),
                np.array([1, 2, 3]))

    outputs = wrap_up([state_ensembles, expected])

    return outputs


def get_np_cov_data():
    arrays = [np.array([[0, 2], [1, 1], [2, 0]]).T]

    expected = [np.array([[1., -1.], [-1.,  1.]])]

    outputs = wrap_up([arrays, expected])
    return outputs


def get_destination_vector_data():
    destinations = ([[0, 100], [100, 0], [50, 50], [300, 100], [125, 225]],
                    [[325, 725], [125, 0], [5, 100], [300, 3], [17, 23]])

    expected = (np.array([0, 100, 100, 0, 50, 50, 300, 100, 125, 225]),
                np.array([325, 725, 125, 0, 5, 100, 300, 3, 17, 23]))

    outputs = wrap_up([destinations, expected])
    return outputs


def get_origin_vector_data():
    origins = ([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
               [[0, 100], [100, 0], [300, 125], [125, 300], [0, 0]],
               [[100, 125], [0, 0], [125, 225], [0, 0], [750, 125]])

    statuses = ([0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
                [1, 0, 1, 0, 1])

    expected = (np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                np.array([0, 100, 100, 0, 300, 125, 125, 300, 0, 0]),
                np.array([100, 125, 0, 0, 125, 225, 0, 0, 750, 125]))

    outputs = wrap_up([origins, statuses, expected])
    return outputs


def get_agent_statuses_data():
    base_statuses = [[0, 1, 2],
                     [0, 1, 2],
                     [0, 1, 2],
                     [0, 1, 2],
                     [0, 1, 2]]

    en_statuses = [[[0, 0, 0],
                    [1, 1, 1],
                    [2, 2, 2]],
                   [[0, 0, 0],
                    [1, 1, 1],
                    [2, 2, 2]],
                   # Uniform across ensemble
                   [[0, 1, 2],
                    [0, 1, 2],
                    [0, 1, 2]],
                   # Mixed, no majority
                   [[0, 0, 0],
                    [1, 1, 1],
                    [2, 2, 2]],
                   # Mixed, majority
                   [[0, 1, 1],
                    [1, 1, 2],
                    [0, 2, 2]]]

    inclusion = [None, AgentIncluder.BASE,
                 AgentIncluder.MODE_EN, AgentIncluder.MODE_EN,
                 AgentIncluder.MODE_EN]

    expected = [[False, True, False],
                [False, True, False],
                [False, True, False],
                [False, False, False],
                [False, True, False]]

    output = wrap_up([base_statuses, en_statuses, inclusion, expected])
    return output


def get_filter_vector_data():
    vector = [np.arange(5) for _ in range(3)]

    statuses = [[False, False, False, False, False],
                [True, True, True, True, True],
                [True, False, True, False, False]]

    expected = [np.array([]),
                np.array([0, 1, 2, 3, 4]),
                np.array([0, 2])]

    output = wrap_up([vector, statuses, expected])
    return output


def get_state_vector_statuses_data():
    base_statuses = [[0, 1, 2],
                     [0, 1, 2],
                     [0, 1, 2],
                     [0, 1, 2]]

    en_statuses = [[[0, 1, 2],
                    [0, 1, 2],
                    [0, 1, 2]],
                   [[0, 1, 2],
                    [0, 1, 2],
                    [0, 1, 2]],
                   [[1, 0, 2],
                    [1, 0, 2],
                    [1, 0, 2]],
                   [[1, 0, 2],
                    [1, 0, 2],
                    [1, 0, 2]]]

    inclusion = [AgentIncluder.BASE, AgentIncluder.BASE,
                 AgentIncluder.MODE_EN, AgentIncluder.MODE_EN]

    vector_mode = [EnsembleKalmanFilterType.STATE,
                   EnsembleKalmanFilterType.DUAL_EXIT,
                   EnsembleKalmanFilterType.STATE,
                   EnsembleKalmanFilterType.DUAL_EXIT]

    expected = [[False, False, True, True, False, False],
                [False, True, False,
                 False, True, False,
                 False, True, False],
                [True, True, False, False, False, False],
                [True, False, False,
                 True, False, False,
                 True, False, False]]

    output = wrap_up((base_statuses, en_statuses, inclusion,
                      vector_mode, expected))
    return output


def get_forecast_error_data():
    # Case 1: None inclusion
    # Case 2: Base inclusion, ensemble var 1
    # Case 3: Base inclusion, ensemble var 2
    # Case 4: Mode inclusion, base var 1
    # Case 5: Mode inclusion, base var 2
    inclusion = [None,
                 AgentIncluder.BASE, AgentIncluder.BASE,
                 AgentIncluder.MODE_EN, AgentIncluder.MODE_EN]

    base_statuses = [[1, 1, 1],
                     [1, 0, 1],
                     [1, 0, 1],
                     [1, 1, 1],
                     [0, 0, 0]]

    ensemble_statuses = [[[1, 1, 1], [1, 0, 1], [1, 0, 0]],
                         [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                         [[1, 1, 1], [1, 0, 1], [1, 0, 0]],
                         [[1, 1, 1], [1, 0, 1], [1, 0, 1]],
                         [[1, 1, 1], [1, 0, 1], [1, 0, 1]]]

    t = np.array([1, 1, 5, 10, 22, 76])
    truth = [t for _ in range(5)]

    state_mean = [np.array([4, 5, 9, 7, 25, 80]),
                  np.array([6, 13, 9, 7, 25, 80]),
                  np.array([6, 13, 9, 7, 25, 80]),
                  np.array([6, 13, 9, 7, 25, 80]),
                  np.array([6, 13, 9, 7, 25, 80])]

    expected = [5, 9, 9, 9, 9]

    output = wrap_up((inclusion, base_statuses, ensemble_statuses,
                      truth, state_mean, expected))
    return output


def get_gate_estimator_allocation_data():
    output = [(None, GateEstimator.NO_ESTIMATE),
              (GateEstimator.ROUNDING, GateEstimator.ROUNDING),
              (GateEstimator.ANGLE, GateEstimator.ANGLE)]
    return output


def get_get_angle_data():
    heads = [(1, 0),
             (1, 1),
             (0, 1),
             (-1, 1),
             (-1, 0),
             (-1, -1),
             (0, -1),
             (1, -1)]

    tails = [(0, 0) for _ in range(len(heads))]

    expected = [0, pi/4, pi/2, 3*pi/4, pi,
                -3*pi/4, -pi/2, -pi/4]

    output = wrap_up([tails, heads, expected])
    return output


def get_edge_angle_data():
    edge_locs = np.array([[0, 400],
                          [0, 700], [250, 700], [455, 700], [700, 700],
                          [740, 700], [740, 610], [740, 550], [740, 400],
                          [740, 340], [740, 190], [740, 125], [740, 5],
                          [740, 0], [555, 0], [370, 0], [185, 0], [0, 0],
                          [0, 150]])

    n_angles = len(edge_locs)

    centre_locs = np.array([[370, 350] for _ in range(n_angles)])

    vectors = edge_locs - centre_locs
    angles = list()
    for i in range(n_angles):
        angles.append(phase(complex(vectors[i][0], vectors[i][1])))

    return [angles]


def get_reverse_bisect_data():
    elements = [12, 10, 7, 4, 2, 0]

    iterable = [10, 8, 5, 3, 1]
    iterables = [iterable for _ in range(len(elements))]

    expected = [0, 0, 2, 3, 4, 5]

    output = wrap_up([elements, iterables, expected])
    return output


def get_edge_loc_data():
    edge_locs = [(0, 400),
                 (0, 700), (250, 700), (455, 700), (700, 700),
                 (740, 700), (740, 610), (740, 550), (740, 400),
                 (740, 340), (740, 190), (740, 125), (740, 5),
                 (740, 0), (555, 0), (370, 0), (185, 0), (0, 0),
                 (0, 150)]
    return [edge_locs]


def get_angle_destination_out_gate_data():
    angles = [2.9, 2.4, pi/2]

    expected = [0, 1, 2]

    output = wrap_up((angles, expected))
    return output


def get_angle_destination_out_data():
    angles = [2.9, 2.4, pi/2]

    expected = [(0, 400), (0, 700), (455, 700)]

    output = wrap_up([angles, expected])
    return output


def get_angle_destination_in_data():
    angles = [pi, 3*pi/4, -pi/4, -pi/2]

    xs = [0, (0, 250), (555, 740), (370, 555)]
    ys = [(150, 400), 700, 0, 0]

    expected = list(zip(xs, ys))

    output = wrap_up([angles, expected])
    return output


def get_construct_state_from_angles_gates_data():
    angles = [[2.9, 2.4, pi/2]]

    expected = [[0, 1, 2]]

    output = wrap_up((angles, expected))
    return output


def get_construct_state_from_angles_locs_data():
    angles = [[2.9, 2.4, pi/2]]

    expected = [[0, 0, 455, 400, 700, 700]]

    output = wrap_up((angles, expected))
    return output


def get_round_target_angle_data():
    angles = [2.9, 2.4]

    insertion_idxs = [1, 1]

    rounded_idxs = [0, 1]

    output = wrap_up((angles, insertion_idxs, rounded_idxs))
    return output


def get_convert_vector_angle_to_gate_data():
    state_vectors = [[25, 155, 740,
                      0, 0, 400,
                      2.9, 2.4, pi/2]]
    expected = [[25, 155, 740,
                 0, 0, 400,
                 0, 1, 2]]

    state_vectors = [np.array(x) for x in state_vectors]
    expected = [np.array(x) for x in expected]

    output = wrap_up((state_vectors, expected))
    return output


def get_process_state_vector_data():
    states = [[10, 15, 100, 221, 52, 700],
              [10, 100, 52, 15, 221, 700, 1, 2, 3],
              [10, 100, 52,
               15, 221, 700,
               1, 2, 3,
               125, 577.5, 740,
               700, 700, 655]]

    filter_modes = [EnsembleKalmanFilterType.STATE,
                    EnsembleKalmanFilterType.DUAL_EXIT,
                    EnsembleKalmanFilterType.DUAL_EXIT]

    gate_estimators = [None, GateEstimator.ROUNDING, GateEstimator.ANGLE]

    model_centre = (370, 350)
    dest0 = (states[2][9], states[2][12])
    dest1 = (states[2][10], states[2][13])
    dest2 = (states[2][11], states[2][14])
    theta0 = get_angle(model_centre, dest0)
    theta1 = get_angle(model_centre, dest1)
    theta2 = get_angle(model_centre, dest2)
    expected = [[10, 15, 100, 221, 52, 700],
                [10, 100, 52, 15, 221, 700, 1, 2, 3],
                [10, 100, 52, 15, 221, 700, theta0, theta1, theta2]]

    states = [np.array(x) for x in states]
    expected = [np.array(x) for x in expected]

    output = wrap_up((states, filter_modes, gate_estimators, expected))
    return output


def get_mod_angles_data():
    angles = [np.array([pi/4, pi/2, 3*pi/4]),
              np.array([-pi/4, -pi/2, -3*pi/4]),
              np.array([5*pi/4, 3*pi/2, 7*pi/4]),
              np.array([-5*pi/4, -3*pi/2, -7*pi/4])]

    expected = [np.array([pi/4, pi/2, 3*pi/4]),
                np.array([-pi/4, -pi/2, -3*pi/4]),
                np.array([-3*pi/4, -pi/2, -pi/4]),
                np.array([3*pi/4, pi/2, pi/4])]

    output = wrap_up((angles, expected))
    return output


def get_multi_gain_data():
    inf_rates = [1.0, 1.1]

    state = np.array([[10, 20, 30],
                      [40, 60, 80],
                      [3, 4, 5]])
    data_cov = np.array([[1, 0],
                         [0, 1]])
    H = np.array([[1, 0, 0],
                  [0, 1, 0]])

    states = [state for _ in range(len(inf_rates))]
    data_covs = [data_cov for _ in range(len(inf_rates))]
    Hs = [H for _ in range(len(inf_rates))]

    expected = [np.array([[0.1996008, 0.3992016],
                          [0.3992016, 0.79840319],
                          [0.01996008, 0.03992016]]),
                np.array([[0.19963702, 0.39927405],
                          [0.39927405, 0.79854809],
                          [0.0199637, 0.0399274]])]

    output = wrap_up((states, data_covs, Hs, inf_rates, expected))
    return output


def get_exit_randomisation_adjacent_data():
    n_adjacents = [1, 2, 3]
    return n_adjacents


def get_standardisation_data():
    vectors = [np.array([0, 185, 370, 555, 740]),
               np.array([0, 175, 350, 525, 700]),
               np.array([0, 2, 5, 8, 10]),
               np.array([-pi, -pi/2, 0, pi/2, pi])]

    tops = [740, 700, 10, pi]
    bottoms = [0, 0, 0, -pi]

    expected = [np.array([-1, -0.5, 0, 0.5, 1]),
                np.array([-1, -0.5, 0, 0.5, 1]),
                np.array([-1, -0.6, 0, 0.6, 1]),
                np.array([-1, -0.5, 0, 0.5, 1])]

    output = wrap_up((vectors, tops, bottoms, expected))
    return output


def get_unstandardisation_data():
    vectors = [np.array([-1, -0.5, 0, 0.5, 1]),
               np.array([-1, -0.5, 0, 0.5, 1]),
               np.array([-1, -0.6, 0, 0.6, 1]),
               np.array([-1, -0.5, 0, 0.5, 1])]

    tops = [740, 700, 10, pi]
    bottoms = [0, 0, 0, -pi]

    expected = [np.array([0, 185, 370, 555, 740]),
                np.array([0, 175, 350, 525, 700]),
                np.array([0, 2, 5, 8, 10]),
                np.array([-pi, -pi/2, 0, pi/2, pi])]

    output = wrap_up((vectors, tops, bottoms, expected))
    return output


def get_alternating_to_sequential_data():
    vector = [[1, 2, 3, 4, 5, 6],
              ['x1', 'y1', 'x2', 'y2'],
              np.array([1, 2, 3, 4, 5, 6])]

    expected = [[1, 3, 5, 2, 4, 6],
                ['x1', 'x2', 'y1', 'y2'],
                np.array([1, 3, 5, 2, 4, 6])]

    output = wrap_up((vector, expected))
    return output


def get_update_data():
    fts = [EnsembleKalmanFilterType.STATE,
           EnsembleKalmanFilterType.DUAL_EXIT]

    state_ensembles = [np.array([[5, 6, 4],
                                 [25, 26, 24],
                                 [10, 11, 9],
                                 [20, 21, 19]]),
                       np.array([[5, 6, 4],
                                 [25, 26, 24],
                                 [10, 11, 9],
                                 [20, 21, 19],
                                 [1, 2, 1],
                                 [2, 3, 3]])]

    data_covs = [np.eye(4), np.eye(4)]

    data = [np.array([[7, 8, 9],
                      [21, 22, 23],
                      [10, 11, 9],
                      [17, 18, 19]]),
            np.array([[7, 8, 9],
                      [21, 22, 23],
                      [10, 11, 9],
                      [17, 18, 19]])]

    Hs = [np.eye(4),
          np.array([[1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0]])]

    expected = [np.array([[4, 5, 4.8],
                          [24, 25, 24.8],
                          [9, 10, 9.8],
                          [19, 20, 19.8]]),
                np.array([[4, 5, 4.8],
                          [24, 25, 24.8],
                          [9, 10, 9.8],
                          [19, 20, 19.8],
                          [0.5, 1.5, 1.4],
                          [2, 3, 3]])]

    output = wrap_up((fts, state_ensembles, data_covs, data, Hs, expected))
    return output


def get_reformat_obs_data():
    vector_lengths = [4, 6]
    ensemble_sizes = [2, 3]

    data = [np.array([[1, 2],
                      [3, 4],
                      [5, 6],
                      [7, 8]]),
            np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9],
                      [10, 11, 12],
                      [13, 14, 15],
                      [16, 17, 18]])]

    expected = [np.array([[1, 2],
                          [5, 6],
                          [3, 4],
                          [7, 8]]),
                np.array([[1, 2, 3],
                          [7, 8, 9],
                          [13, 14, 15],
                          [4, 5, 6],
                          [10, 11, 12],
                          [16, 17, 18]])]

    output = wrap_up((vector_lengths, ensemble_sizes, data, expected))
    return output


def get_standardise_ensemble_data():
    states = [np.array([[0, 740],
                        [185, 555],
                        [0, 700],
                        [175, 525]]),
              np.array([[0, 740],
                        [185, 555],
                        [0, 700],
                        [175, 525],
                        [0, pi/2],
                        [pi, -pi/2]]),
              np.array([[0, 740],
                        [185, 555],
                        [0, 700],
                        [175, 525],
                        [0, 10],
                        [5, 8]])]

    pop_sizes = [2, 2, 2]
    en_sizes = [2, 2, 2]
    n_vars = [2, 3, 3]
    gate_estimators = [None, GateEstimator.ANGLE, GateEstimator.ROUNDING]

    expected = [np.array([[-1, 1],
                          [-0.5, 0.5],
                          [-1, 1],
                          [-0.5, 0.5]]),
                np.array([[-1, 1],
                          [-0.5, 0.5],
                          [-1, 1],
                          [-0.5, 0.5],
                          [0, 0.5],
                          [1, -0.5]]),
                np.array([[-1, 1],
                          [-0.5, 0.5],
                          [-1, 1],
                          [-0.5, 0.5],
                          [-1, 1],
                          [0, 0.6]])]

    output = wrap_up((states, pop_sizes, en_sizes, n_vars, gate_estimators,
                      expected))
    return output


def get_unstandardise_ensemble_data():
    states = [np.array([[-1, 1],
                        [-0.5, 0.5],
                        [-1, 1],
                        [-0.5, 0.5]]),
              np.array([[-1, 1],
                        [-0.5, 0.5],
                        [-1, 1],
                        [-0.5, 0.5],
                        [0, 0.5],
                        [1, -0.5]]),
              np.array([[-1, 1],
                        [-0.5, 0.5],
                        [-1, 1],
                        [-0.5, 0.5],
                        [-1, 1],
                        [0, 0.6]])]

    pop_sizes = [2, 2, 2]
    en_sizes = [2, 2, 2]
    n_vars = [2, 3, 3]
    gate_estimators = [None, GateEstimator.ANGLE, GateEstimator.ROUNDING]

    expected = [np.array([[0, 740],
                          [185, 555],
                          [0, 700],
                          [175, 525]]),
                np.array([[0, 740],
                          [185, 555],
                          [0, 700],
                          [175, 525],
                          [0, pi/2],
                          [pi, -pi/2]]),
                np.array([[0, 740],
                          [185, 555],
                          [0, 700],
                          [175, 525],
                          [0, 10],
                          [5, 8]])]

    output = wrap_up((states, pop_sizes, en_sizes, n_vars, gate_estimators,
                      expected))
    return output
