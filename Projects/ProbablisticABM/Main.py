from Simulation.SimpleAgent import SimpleAgent
from Simulation.Environment import Environment
from Simulation.Schedule import Schedule
from Simulation.ExitGate import ExitGate

import tensorflow as tf
import tensorflow_probability as tfp

from random import randint
from random import getrandbits

n_steps = 480 / 10


# Define how to step the state of the model. Should be able to change to distributions later on.
def custom_stepper(agent):
    if not agent.attributes['isActive'] and not agent.attributes['completed']:
        agent.attributes['isActive'] = bool(getrandbits(1))

    if agent.attributes['isActive']:
        agent.attributes['s'] = randint(0, 4)
        agent.attributes['x'] = agent.attributes['x'] + agent.attributes['s']

        if agent.attributes['x'] > 20:
            agent.attributes['isActive'] = False
            agent.attributes['completed'] = True
            print('Agent {} exited at gate {}'.format(agent.attributes['id'], agent.attributes['d']))

    return agent


def run():
    # Build new environment for the agents to live in.
    environment = Environment(480, 480)

    # Set the number of agents we want in the environment.
    environment.n_agents = 100000

    # Deterministic destination and position for a single agent in the system( d )
    # Populate the environment with our custom agent class.
    environment.populate(SimpleAgent(x=0,
                                     y=environment.height * 0.5,
                                     s=0,
                                     d=0,
                                     isActive=0,
                                     completed=0))

    # Build a destination gate for agents to move to.
    # Gate one is in the center for now. Will shift a third the way up and add second gate as soon as agent is finished.
    gate_1 = ExitGate(width=1, height=48)
    gate_1.x = 480
    gate_1.y = 480 * 0.5

    schedule = Schedule()
    schedule.n_steps = n_steps

    environment.agents = schedule.run(agents=environment.agents, step=custom_stepper)

    print(environment.agents)


if __name__ == "__main__":
    run()
