# Imports
import numpy as np
import sys
sys.path.append('../stationsim')

from stationsim_gcs_model import Model


# Functions
def set_up_model():
    model_params = {'pop_total': 3,
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
