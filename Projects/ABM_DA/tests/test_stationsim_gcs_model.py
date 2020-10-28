# Imports
from generate_data.stationsim_gcs_model_data import *
import numpy as np
import pytest


# Data
get_state_location_data = get_location_data()

get_state_location2D_data = get_location2D_data()

get_state_loc_exit_data = get_loc_exit_data()


# Tests
@pytest.mark.parametrize('agent_locations, expected', get_state_location_data)
def test_get_state_location(agent_locations, expected):
    model = set_up_model()
    for i, agent in enumerate(model.agents):
        agent.location = agent_locations[i]
    assert np.array_equal(model.get_state('location'), expected)


@pytest.mark.parametrize('agent_locations, expected',
                         get_state_location2D_data)
def test_get_state_location2D(agent_locations, expected):
    model = set_up_model()
    for i, agent in enumerate(model.agents):
        agent.location = agent_locations[i]
    assert model.get_state('location2D') == expected


@pytest.mark.parametrize('agent_locations, exits, expected',
                         get_state_loc_exit_data)
def test_get_state_loc_exit(agent_locations, exits, expected):
    model = set_up_model()
    for i, agent in enumerate(model.agents):
        agent.location = agent_locations[i]
        agent.gate_out = exits[i]
    assert model.get_state('loc_exit') == expected
