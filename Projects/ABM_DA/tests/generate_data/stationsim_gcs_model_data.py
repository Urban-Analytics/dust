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


def __get_gate_widths():
    gate_widths = [250, 250, 245,
                   90, 150, 150, 120,
                   185, 185, 185, 185]

    return gate_widths


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


def __get_left_right_gate_location_data(gate_locations, gate_widths):
    gate_data = list()
    left_gate_ids = [0]
    right_gate_ids = [3, 4, 5, 6]
    gate_ids = [{'side': 'left', 'ids': left_gate_ids},
                {'side': 'right', 'ids': right_gate_ids}]
    x_left = 7.35
    x_right = 732.65

    for side_dict in gate_ids:
        side = side_dict['side']
        ids = side_dict['ids']
        for gate_number in ids:
            d = dict()
            d['side'] = side
            d['gate_number'] = gate_number
            x = x_left if side == 'left' else x_right
            location = gate_locations[gate_number]
            width = gate_widths[gate_number]
            y_upper = location[1] + (width / 2)
            y_lower = location[1] - (width / 2)
            d['upper'] = (x, y_upper)
            d['lower'] = (x, y_lower)

            gate_data.append(d)

    return gate_data


def __get_top_bottom_gate_location_data(gate_locations, gate_widths):
    gate_data = list()
    top_gate_ids = [1, 2]
    bottom_gate_ids = [7, 8, 9, 10]
    gate_ids = [{'side': 'top', 'ids': top_gate_ids},
                {'side': 'bottom', 'ids': bottom_gate_ids}]
    y_top = 692.65
    y_bottom = 7.35

    for side_dict in gate_ids:
        side = side_dict['side']
        ids = side_dict['ids']
        for gate_number in ids:
            d = dict()
            d['side'] = side
            d['gate_number'] = gate_number
            y = y_top if side == 'top' else y_bottom
            location = gate_locations[gate_number]
            width = gate_widths[gate_number]
            x_upper = location[0] + (width / 2)
            x_lower = location[0] - (width / 2)
            d['upper'] = (x_upper, y)
            d['lower'] = (x_lower, y)

            gate_data.append(d)

    return gate_data


def get_agent_gate_location_data():
    gate_numbers, gate_locations = __get_gate_locations()
    gate_widths = __get_gate_widths()

    output = list()
    left_right_gates = __get_left_right_gate_location_data(gate_locations,
                                                           gate_widths)
    top_bottom_gates = __get_top_bottom_gate_location_data(gate_locations,
                                                           gate_widths)
    output.extend(left_right_gates)
    output.extend(top_bottom_gates)

    return output
