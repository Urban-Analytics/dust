# Imports
from generate_data.stationsim_gcs_model_data import *
from math import floor
import numpy as np
import pytest
import sys
sys.path.append('../stationsim')
from stationsim_gcs_model import Agent


# Data
get_state_location_data = get_location_data()

get_state_location2D_data = get_location2D_data()

get_state_loc_exit_data = get_loc_exit_data()

gate_location_data = get_gate_location_data()

distance_data = get_distance_data()


# Helper functions
def __is_valid_location(agent_location, upper, lower, side):
    if side == 'left' or side == 'right':
        return __is_valid_location_left_right(agent_location, upper, lower)
    elif side == 'top' or side == 'bottom':
        return __is_valid_location_top_bottom(agent_location, upper, lower)
    else:
        raise ValueError(f'Invalid side provided: {side}')


def __is_valid_location_left_right(agent_location, upper, lower):
    x = agent_location[0] == upper[0] and agent_location[0] == lower[0]
    y = lower[1] <= agent_location[1] <= upper[1]
    return x and y


def __is_valid_location_top_bottom(agent_location, upper, lower):
    x = lower[0] <= agent_location[0] <= upper[0]
    y = agent_location[1] == upper[1] and agent_location[1] == lower[1]
    return x and y


# Tests
# Model tests
def test_station_setup():
    model = set_up_model()
    a = model.width == 740
    b = model.height == 700
    assert a and b


def test_gate_setup_number():
    model = set_up_model()
    # Check for correct number of gates
    a = model.gates_in == 11
    b = model.gates_out == 11
    assert a and b


@pytest.mark.parametrize('gate_number, gate_location', gate_location_data)
def test_gate_setup_location(gate_number, gate_location):
    model = set_up_model()
    # Correct x
    a = model.gates_locations[gate_number][0] == gate_location[0]
    # Correct y
    b = model.gates_locations[gate_number][1] == gate_location[1]
    assert a and b


def test_clock_setup():
    model = set_up_model()
    # Radius of 10
    size = model.clock.size == 56
    # Stationary
    speed = model.clock.speed == 0
    # Location
    location = model.clock.location == [370, 275]
    # isAgent
    cl = isinstance(model.clock, Agent)
    assert all([size, speed, location, cl])


def test_gate_out_allocation():
    diff_gates = list()
    num_models = 100
    population_size = 5

    for _ in range(num_models):
        model = set_up_model(population_size)
        # Check that each agent's entrance is different from its exit
        model_diff_gates = [agent.gate_in != agent.gate_out for agent in
                            model.agents]
        diff_gates.append(all(model_diff_gates))

    # Check that this is true for each model run
    assert all(diff_gates)


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


def test_model_deactivation_time():
    # When should model finish?
    # A) When timer runs out
    # B) When all agents have finished

    # When should the model not finish?
    # If there are still active agents and time remaining

    model = set_up_model(step_limit=20, population_size=1)
    # Check model status upon init - should be 1
    assert model.status == 1

    while model.step_id < model.step_limit:
        model.step()

    model.step()
    assert model.status == 0


def test_model_deactivation_agents():
    model = set_up_model(population_size=1)
    assert model.status == 1
    agent = model.agents[0]

    while (agent.status == 0 or agent.status == 1):
        model.step()

    model.step()
    assert model.status == 0


def test_model_speed_defaults():
    model = set_up_model(population_size=1)

    # Model speed params
    assert model.speed_min == 0.2
    assert model.speed_mean == 0.839236
    assert model.speed_std == 0.349087
    assert model.speed_steps == 3


# Agent tests
def test_speed_allocation():
    model = set_up_model(population_size=1)
    agent = model.agents[0]

    # Agent speed params
    speed_max = max(agent.speeds)
    assert speed_max > model.speed_min
    assert min(agent.speeds) >= model.speed_min
    assert agent.speed in agent.speeds

    # Ensure that 68% of the agent max speeds lie within 1 sd of mean
    n_trials = 500
    count = 0
    for _ in range(n_trials):
        model = set_up_model()
        agent = model.agents[0]
        speed_max = max(agent.speeds)
        if speed_max < 2:
            count += 1
    proportion = count / n_trials
    assert proportion > 0.68


@pytest.mark.parametrize('location1, location2, distance', distance_data)
def test_distance_calculation(location1, location2, distance):
    assert Agent.distance(location1, location2) == distance


def test_direction_calculation():
    pass


def test_normal_direction_calculation():
    pass


def test_agent_activation():
    model = set_up_model(population_size=1)
    agent = model.agents[0]
    agent_activation_time = agent.time_activate

    # Agent should be inactive at start of model
    assert agent.status == 0

    # Run model until model time exceeds activation time
    while model.total_time < agent_activation_time:
        assert agent.status == 0
        model.step()

    # Agent activated on next step
    model.step()

    assert agent.status == 1


def test_agent_deactivation():
    # Given that there is only 1 agent in the system, the time taken per step
    # should be 1
    # We should therefore find the case where the distance between the agent
    # and its target destination is 1 * speed
    # Until this point, the agent should have a status of either 0 or 1
    # If we step the agent again then it should reach its destination and be
    # deactivated
    model = set_up_model(population_size=1)
    agent = model.agents[0]

    assert agent.status == 0

    distance = agent.distance(agent.location, agent.loc_desire)

    assert distance != 0

    while distance > agent.speed + model.gates_space:
        assert (agent.status == 0 or agent.status == 1)
        model.step()
        distance = agent.distance(agent.location, agent.loc_desire)

    model.step()
    assert agent.status == 2


def test_set_agent_location():
    model = set_up_model()
    agent = model.agents[0]

    location_data, gate_numbers_by_side = get_agent_gate_location_data()
    results = list()

    for side, gate_numbers in gate_numbers_by_side.items():
        for i, gate_number in enumerate(gate_numbers):
            # Test multiple times because set_agent_location has
            # some randomness
            for _ in range(50):
                test_location = agent.set_agent_location(gate_number)
                gate_upper = location_data[side]['upper'][i]
                gate_lower = location_data[side]['lower'][i]
                test_result = __is_valid_location(test_location, gate_upper,
                                                  gate_lower, side)
                results.append(test_result)

    assert all(results)


def test_set_wiggle():
    pass
