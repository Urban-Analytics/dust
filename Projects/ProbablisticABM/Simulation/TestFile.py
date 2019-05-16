from Simulation.SimpleAgent import SimpleAgent
from multiprocessing import cpu_count
import os

# Define how to step the state of the model. Should be able to change to distributions later on.
def customStepper(agent):
    agent.attributes['x'] = agent.attributes['x'] + agent.attributes['s']
    return agent

if __name__ == '__main__':
    print(cpu_count())
    print(os.cpu_count())
