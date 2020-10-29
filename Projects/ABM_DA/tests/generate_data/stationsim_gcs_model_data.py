# Imports
from math import sqrt
import numpy as np
import sys
sys.path.append('../stationsim')

from stationsim_gcs_model import Model


# Functions
def set_up_model(population_size=3):
    model_params = {'pop_total': population_size,
                    'station': 'Grand_Central'}
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
    gate_numbers = list(range(10))
    gate_locations = [[20, 400], [170, 400], [200, 340], [200, 275],
                      [200, 200], [200, 125], [200, 60], [170, 0],
                      [20, 0], [0, 200]]
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
