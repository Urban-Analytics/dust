from Simulation.SimpleAgent import SimpleAgent
from Simulation.Environment import Environment
from Simulation.Schedule import Schedule
from Simulation.ExitGate import ExitGate

n_steps = 480 / 10


# Define how to step the state of the model. Should be able to change to distributions later on.
def custom_stepper(agent):
    agent.attributes['x'] = agent.attributes['x'] + agent.attributes['s']
    return agent


def run():
    # Build new environment for the agents to live in.
    environment = Environment(480, 480)

    # Set the number of agents we want in the environment.
    environment.n_agents = 5

    # Deterministic destination and position for a single agent in the system( d )
    # Populate the environment with our custom agent.
    environment.populate(SimpleAgent(x=0,
                                     y=environment.height * 0.5,
                                     s=1,
                                     d=0))

    # Build a destination gate for agents to move to.
    gate_1 = ExitGate(width=1, height=48)

    gate_1.x = 480
    # Gate one is in the center for now. Will shift a third the way up and add second gate as soon as agent is finished.
    gate_1.y = 480 * 0.5

    schedule = Schedule()
    schedule.n_steps = n_steps

    schedule.step(environment=environment, step=custom_stepper)


if __name__ == "__main__":
    run()
