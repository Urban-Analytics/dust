from Simulation.SimpleAgent import SimpleAgent

# Define how to step the state of the model. Should be able to change to distributions later on.
def customStepper(agent):
    agent.attributes['x'] = agent.attributes['x'] + agent.attributes['s']
    return agent

if __name__ == '__main__':
    customStepper(SimpleAgent(x=0, y=0, s=1, d=0))