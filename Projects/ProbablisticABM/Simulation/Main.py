from Simulation.SimpleAgent import SimpleAgent
from Simulation.Environment import Environment
from Simulation.Schedule import Schedule
from Simulation.ExitGate import ExitGate

import random
import time

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

'''
TensorFlow's eager execution is an imperative programming environment that evaluates operations immediately, 
without building graphs: operations return concrete values instead of constructing a computational graph to run 
later. This makes it easy to get started with TensorFlow and debug models 
'''
try:
    tf.enable_eager_execution()
except ValueError:
    pass

# Build new environment for the agents to live in.
environment = Environment(480, 480)

# Set the number of agents we want in the environment.
environment.n_agents = 10

# Deterministic destination and position for a single agent in the system( d )
# Populate the environment with our custom agent class.
environment.populate(SimpleAgent(x=0,
                                 y=environment.height * 0.5,
                                 s=0,
                                 d=0,
                                 isActive=0,
                                 completed=0))

# Scheduler that handles stepping the model and tracking its history.
schedule_1 = Schedule(n_steps=480/10)


# Define how to step the state of the agents.
def custom_stepper(agent):
    # Randomly activate agent.
    if not agent.attributes['isActive'] and not agent.attributes['completed']:
        agent.attributes['isActive'] = random.getrandbits(1)
        agent.attributes['s'] = tfd.Normal(loc=1, scale=1.).sample()
        # agent.attributes['d'] = tfd.Categorical(probs=[0.5, 0.5])

    if agent.attributes['isActive'] and not agent.attributes['completed']:
        agent.attributes['x'] = tf.math.add(agent.attributes['x'], agent.attributes['s'])

        if tf.cond(pred=tf.greater(agent.attributes['x'], tf.constant(20.)),
                   true_fn=lambda: True,
                   false_fn=lambda: False):
            agent.deactivate()
    return agent


def run():
    start = time.time()
    schedule_1.run(agents=environment.agents, step=custom_stepper)
    end = time.time()
    print('Run finished in {} seconds.'.format(end - start))


if __name__ == "__main__":
    run()
