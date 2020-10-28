# Imports
from generate_data.stationsim_gcs_model_data import *
import numpy as np
import pytest


# Data
get_state_location_data = get_location_data()

get_state_location2D_data = get_location2D_data()

get_state_loc_exit_data = get_loc_exit_data()

gate_location_data = get_gate_location_data()


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


# Model tests
def test_station_setup():
    model = set_up_model()
    assert model.width == 200
    assert model.height == 400


def test_gate_setup_number():
    model = set_up_model()
    # Check for correct number of gates
    assert model.gates_in == 10
    assert model.gates_out == 10


@pytest.mark.parametrize('gate_number, gate_location', gate_location_data)
def test_gate_setup_location(gate_number, gate_location):
    model = set_up_model()
    # Correct x
    assert model.gates_locations[gate_number][0] == gate_location[0]
    # Correct y
    assert model.gates_locations[gate_number][1] == gate_location[1]


def test_clock_setup():
    pass


def test_gate_out_allocation():
    pass


# Agent tests
def test_speed_allocation():
    pass


def test_distance_calculation():
    pass


def test_direction_calculation():
    pass


def test_normal_direction_calculation():
    pass


def test_agent_activation():
    pass


def test_agent_deactivation():
    pass


def test_set_agent_location():
    pass


def test_set_wiggle():
    pass
