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
    """
    Check if the provided agent location is valid.

    Given an agent location, check that it is a valid location in the model at
    which to be introduced via a gate. This function acts as a wrapper for two
    further functions - one for checking the validity of an agent location along
    the left and right edges of the environment and the other for checking the
    validity of an agent location along the top and bottom edges of the
    environment.

    Parameters
    ----------
    agent_location : array_like
        An array containing two elements, the (x, y) location proposed for the
        agent in proximity to a gate.
    upper : array_like
        An array containing two elements, the (x, y) location acting as one of
        the limits for the agent location.
    lower : array_like
        An array containing two elements, the (x, y) location acting as the
        second of the limits for the agent location.
    side : str
        A string indicating on which side the gate lies. Restricted to one of
        ['top', 'bottom', 'left', 'right'].

    Raises
    ------
    ValueError:
        If a string other than those listed above is passed as a value of
        'side'.
    """
    if side == 'left' or side == 'right':
        return __is_valid_location_left_right(agent_location, upper, lower)
    elif side == 'top' or side == 'bottom':
        return __is_valid_location_top_bottom(agent_location, upper, lower)
    else:
        raise ValueError(f'Invalid side provided: {side}')


def __is_valid_location_left_right(agent_location, upper, lower):
    """
    Check if the provided agent location is a valid position along the left or
    right edges of the environment.

    Check that the agent x-location is the same as those provided for the limit
    locations, and that the agent y-location lies between the y-locations
    provided by the limit locations.

    Parameters
    ----------
    agent_location : array_like
        An array containing two elements, the (x, y) location proposed for the
        agent in proximity to a gate on either the left or right edges of the
        environment.
    upper : array_like
        An array containing two elements, the (x, y) location acting as one of
        the limits for the agent location. Holds the larger of the two y-values.
    lower : array_like
        An array containing two elements, the (x, y) location acting as one of
        the limits for the agent location. Holds the smaller of the two
        y-values.
    """
    x_diff_upper = agent_location[0] - upper[0]
    x_diff_lower = agent_location[0] - lower[0]
    x = x_diff_upper == pytest.approx(0) and x_diff_lower == pytest.approx(0)
    y = lower[1] <= agent_location[1] <= upper[1]
    return x and y


def __is_valid_location_top_bottom(agent_location, upper, lower):
    """
    Check if the provided agent location is a valid position along the top or
    bottom edges of the environment.

    Check that the agent y-location is the same as those provided for the limit
    locations, and that the agent x-location lies between the x-locations
    provided by the limit locations.

    Parameters
    ----------
    agent_location : array_like
        An array containing two elements, the (x, y) location proposed for the
        agent in proximity to a gate one either the top or bottom edges of the
        environment.
    upper : array_like
        An array containing two elements, the (x, y) location acting as one of
        the limits for the agent location. Holds the larger of the two x-values.
    lower : array_like
        An array containing two elements, the (x, y) location acting as one of
        the limits for the agents location. Holds the smaller of the two
        x-values.
    """
    x = lower[0] <= agent_location[0] <= upper[0]
    y_diff_upper = agent_location[1] - upper[1]
    y_diff_lower = agent_location[1] - lower[1]
    y = y_diff_upper == pytest.approx(0) and y_diff_lower == pytest.approx(0)
    return x and y


# Tests
# Model tests
def test_station_setup():
    """
    Test station setup

    Test that the station is set up correctly by checking the dimensions of the
    station produced.
    """
    model = set_up_model()
    a = model.width == 740
    b = model.height == 700
    assert a and b


def test_gate_setup_number():
    """
    Test number of gates

    Test that the correct number of gates are produced as part of the station
    setup process.
    """
    model = set_up_model()
    # Check for correct number of gates
    a = model.gates_in == 11
    b = model.gates_out == 11
    assert a and b


@pytest.mark.parametrize('gate_number, gate_location', gate_location_data)
def test_gate_setup_location(gate_number, gate_location):
    """
    Test gate locations.

    Test that gates are allocated to the correct locations in the environment as
    part of the setup process.

    Parameters
    ----------
    gate_number : int
        The index of the gate, takes integer values 0-10 inclusive.
    gate_location : array_like
        An array containing two elements, the (x, y) location of the gate
        indexed by the gate number.
    """
    model = set_up_model()
    # Correct x
    a = model.gates_locations[gate_number][0] == gate_location[0]
    # Correct y
    b = model.gates_locations[gate_number][1] == gate_location[1]
    assert a and b


def test_clock_setup():
    """
    Test clock setup

    Test that, upon setup, the clock in the environment has the following
    characteristics:
    - Has the correct size,
    - Has the correct speed, i.e. it is stationary,
    - Has the correct location,
    - Is of type Agent.
    """
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
    """
    Test allocation of exit gates.

    Test that agents are allocated an exit gate which is different from their
    entrance gate. This is checked with a number of instances of the model as
    the gate allocation process is a random process.
    """
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
    """
    Test Model.get_state('location').

    Test that Model.get_state('location') provides a state vector in the correct
    form with the correct values.

    Parameters
    ----------
    agent_locations : list_like
        List of tuples, each of which is an (x, y) location to prescribe for an
        agent in the model.
    expected : array_like 
        A state vector representing the locations of the agents.
    """
    model = set_up_model()
    for i, agent in enumerate(model.agents):
        agent.location = agent_locations[i]
    assert np.array_equal(model.get_state('location'), expected)


@pytest.mark.parametrize('agent_locations, expected',
                         get_state_location2D_data)
def test_get_state_location2D(agent_locations, expected):
    """
    Test Model.get_state('location2D').

    Test that Model.get_state('location2D') provides a state vector in the
    correct form with the correct values.

    Parameters
    ----------
    agent_locations : list_like
        List of tuples, each of which is an (x, y) location to prescribe for an
        agent in the model.
    expected : array_like 
        A state vector representing the locations of the agents.
    """
    model = set_up_model()
    for i, agent in enumerate(model.agents):
        agent.location = agent_locations[i]
    assert model.get_state('location2D') == expected


@pytest.mark.parametrize('agent_locations, exits, expected',
                         get_state_loc_exit_data)
def test_get_state_loc_exit(agent_locations, exits, expected):
    """
    Test Model.get_state('loc_exit').

    Test that Model.get_state('loc_exit') provides a state vector in the correct
    form with the correct values.

    Parameters
    ----------
    agent_locations : list_like
        List of tuples, each of which is an (x, y) location to prescribe for an
        agent in the model.
    exits : list_like 
        List of integers, each of which indexes a gate.
    expected : array_like 
        A state vector representing the locations of the agents.
    """
    model = set_up_model()
    for i, agent in enumerate(model.agents):
        agent.location = agent_locations[i]
        agent.gate_out = exits[i]
    assert model.get_state('loc_exit') == expected


def test_model_deactivation_time():
    """
    Test model deactivation with respect to time.

    Test that the model deactivates under the correct conditions, i.e.
    - When the timer runs out (reaches the step limit)
    """
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
    """
    Test model deactivation with respect to agents.

    Test that the model deactivates under the correct conditions, i.e.
    - When all agents have finished the movements.
    """
    model = set_up_model(population_size=1)
    assert model.status == 1
    agent = model.agents[0]

    while (agent.status == 0 or agent.status == 1):
        model.step()

    model.step()
    assert model.status == 0


def test_model_speed_defaults():
    """
    Test model speed defaults

    Test that the default values prescribing the distribution of agents speeds
    is correct.
    """
    model = set_up_model(population_size=1)

    # Model speed params
    assert model.speed_min == 0.2
    assert model.speed_mean == 0.839236
    assert model.speed_std == 0.349087
    assert model.speed_steps == 3


# Agent tests
def test_speed_allocation():
    """
    Test agent speeds allocation

    Test that the speeds allocated to agents are within the feasible range
    provided, and that they distributed sensibly.
    """
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
    """
    Test distance calculation

    Test that the calculation of distances between two locations is correct.

    Parameters
    ----------
    location1 : array_like
        An array with two elements - an (x, y) location.
    location2 : array_like
        An array with two elements - an (x, y) location.
    distance : float
        The Euclidean distance between the two locations.
    """
    assert Agent.distance(location1, location2) == distance


def test_direction_calculation():
    pass


def test_normal_direction_calculation():
    pass


def test_agent_activation():
    """
    Test agent activation.

    Test that agents are activated at the correct time.
    """
    model = set_up_model(population_size=1)
    agent = model.agents[0]
    agent_activation_time = agent.steps_activate

    # Agent should be inactive at start of model
    assert agent.status == 0

    # Run model until model time exceeds activation time
    while model.total_time < agent_activation_time:
        assert agent.status == 0
        model.step()

    # Agent activated on next step
    # May have to wait at gate
    model.step()
    model.step()

    assert agent.status == 1


def test_agent_deactivation():
    """
    Test agent deactivation.

    Test that agents are deactivated at the correct time, i.e. that they
    deactivate after having reached their destination.
    """
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
    """
    Test Agent.set_agent_location().

    Test that the process of allocating locations to agents near to entrance
    gates is correct.
    """
    model = set_up_model()
    agent = model.agents[0]

    results = list()
    location_data = get_agent_gate_location_data()

    for gate in location_data:
        test_location = agent.set_agent_location(gate['gate_number'])
        test_result = __is_valid_location(test_location, gate['upper'],
                                          gate['lower'], gate['side'])
        results.append(test_result)
        print(test_location, gate['upper'], gate['lower'], '\n')

    assert all(results)


def test_set_wiggle():
    pass
