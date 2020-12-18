# Imports
from math import sqrt
import numpy as np
import sys
sys.path.append('../stationsim')

from stationsim_gcs_model import Model


# Functions
def set_up_model(population_size=3, step_limit=3600):
    model_params = {'pop_total': population_size,
                    'station': 'Grand_Central',
                    'step_limit': step_limit}
    model = Model(**model_params)
    return model


def get_location_data():
    agent_locations1 = [(1, 2), (3, 4), (5, 6)]
    agent_locations2 = [(20, 10), (45, 80), (99, 32)]

    expected1 = np.array([1, 2, 3, 4, 5, 6])
    expected2 = np.array([20, 10, 45, 80, 99, 32])

    location_data = [(agent_locations1, expected1),
                     (agent_locations2, expected2)]

    return location_data


def get_location2D_data():
    agent_locations1 = [(1, 2), (3, 4), (5, 6)]
    agent_locations2 = [(20, 10), (45, 80), (99, 32)]

    expected1 = [(1, 2), (3, 4), (5, 6)]
    expected2 = [(20, 10), (45, 80), (99, 32)]

    location2D_data = [(agent_locations1, expected1),
                       (agent_locations2, expected2)]

    return location2D_data


def get_loc_exit_data():
    agent_locations1 = [(1, 2), (3, 4), (5, 6)]
    agent_locations2 = [(20, 10), (45, 80), (99, 32)]

    exits1 = [0, 3, 5]
    exits2 = [2, 8, 4]

    expected1 = [1, 3, 5, 2, 4, 6, 0, 3, 5]
    expected2 = [20, 45, 99, 10, 80, 32, 2, 8, 4]

    loc_exit_data = [(agent_locations1, exits1, expected1),
                     (agent_locations2, exits2, expected2)]

    return loc_exit_data


def __get_gate_locations():
    gate_numbers = list(range(11))
    gate_locations = [[0, 275], [125, 700], [577.5, 700],
                      [740, 655], [740, 475], [740, 265], [740, 65],
                      [647.5, 0], [462.5, 0], [277.5, 0], [92.5, 0]]

    return gate_numbers, gate_locations


def get_gate_location_data():
    gate_numbers, gate_locations = __get_gate_locations()

    gate_location_data = [(gate_numbers[i], gate_locations[i]) for i in
                          range(len(gate_numbers))]
    return gate_location_data


def get_distance_data():
    location1s = [[3, 17], [322, 154], [128, 307],
                  [123, 321], [172, 13], [111, 60]]
    location2s = [[6, 21], [310, 149], [113, 315],
                  [133, 316], [320, 116], [111, 60]]
    distances = [5, 13, 17, 5 * sqrt(5), sqrt(32513), 0]

    assert len(location1s) == len(location2s)
    assert len(location1s) == len(distances)

    distance_data = list()

    for i in range(len(distances)):
        t = (location1s[i], location2s[i], distances[i])
        distance_data.append(t)

    return distance_data


def __generate_side_locations(indices, gate_locations):
    side_locations = [gate_locations[i] for i in indices]
    return side_locations


def __generate_upper_lower_bounds(side_locations, side):
    sides = {'left': {'upper': [1.05, 10], 'lower': [1.05, -10]},
             'right': {'upper': [-1.05, 10], 'lower': [-1.05, -10]},
             'top': {'upper': [10, -1.05], 'lower': [-10, -1.05]},
             'bottom': {'upper': [10, 1.05], 'lower': [-10, 1.05]}}

    upper_bounds, lower_bounds = list(), list()
    for sl in side_locations:
        upper_bound = [sl[0] + sides[side]['upper'][0],
                       sl[1] + sides[side]['upper'][1]]
        lower_bound = [sl[0] + sides[side]['lower'][0],
                       sl[1] + sides[side]['lower'][1]]
        upper_bounds.append(upper_bound)
        lower_bounds.append(lower_bound)

    return upper_bounds, lower_bounds


def __get_side_bounds(gate_numbers, gate_locations, side_name):
    side_locations = __generate_side_locations(gate_numbers, gate_locations)
    side_upper, side_lower = __generate_upper_lower_bounds(side_locations,
                                                           side_name)
    return side_upper, side_lower


def get_agent_gate_location_data():
    gate_numbers, gate_locations = __get_gate_locations()

    # Top and bottom
    i_top = [0, 1]
    i_bottom = [7, 8]
    top_upper, top_lower = __get_side_bounds(i_top, gate_locations, 'top')
    bottom_upper, bottom_lower = __get_side_bounds(i_bottom, gate_locations,
                                                   'bottom')

    # Left and right
    i_left = [9]
    i_right = [2, 3, 4, 5, 6]
    left_upper, left_lower = __get_side_bounds(i_left, gate_locations, 'left')
    right_upper, right_lower = __get_side_bounds(i_right, gate_locations,
                                                 'right')

    output = {'top': {'upper': top_upper, 'lower': top_lower},
              'bottom': {'upper': bottom_upper, 'lower': bottom_lower},
              'left': {'upper': left_upper, 'lower': left_lower},
              'right': {'upper': right_upper, 'lower': right_lower}}

    gate_indices = {'top': i_top, 'bottom': i_bottom,
                    'left': i_left, 'right': i_right}
    return output, gate_indices
