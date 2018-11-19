"""SimpleModel (Python implementation)

This is a python implementation of the 'SimpleModel' which can be used to test 
data assimilation implementations. It's basically a loop that draws a random 
number and increments a counter.

The main function that it provides is called 'step'. This is the state
transition function.
"""

import random
from typing import List

class SimpleModel:
    
    def __init__(self, threshold:float):
        """
        Initialise the model.
        
        Parameters:
            - threshold: the threshold at which point the counter should
            be incremented or decremented
        """
        self.threshold = threshold        
        #self.counter = 0 # Keeps track of the current state
        
        
    def step(self, state:List[int]) -> List[int]:
        """
        The transition function. Takes the previous model state and
        returns the new state after one iteration.
        The state is implemented as a list, but the list only has one
        entry (the model state is just a single integer)
        """
        if random.gauss(0,1) > self.threshold:
            return [state[0] + 1]
        else:
            return [state[0] - 1]



# Test the model for 1000 iterations
if __name__=="__main__":
    m = SimpleModel(0.2)
    state = [0]
    history = []
    
    for i in range(0,1000):
        history.append(state[0])
        state = m.step(state)
        
    print("Finsihed. History is {}".format(history))