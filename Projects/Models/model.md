# MesaFree
The simplest agent based modelling module for python.  Designed by Andrew West, for questions email `gyawe`.

#### Sample Model
Here agents are created with a random location.

At each step they move up and to the right one space.

#### Agent
The `Agent` class takes in few parameters, `model` is required and `unique_id` is heavily suggested, `location` is our variable of interest.

`step()` is an empty function/method which I believe is where any data assimilation would take place. These agents just `move()`.

Edit:
`# parameters`
`step()`
`move()` and more

#### Model
The `model` class is the step function you would call.  In `model.step() ~= agent[for all].step()`, a shuffle can be applied in case the order is significant.

The rest creates all the agents at the start.  In general the system design is pretty specific.

Edit:
`# parameters`
`step()`
`initialise_agents()`

#### Plotting
This plotting uses generic `matplotlib` tools.  But a warning before you run edit it:
Using pause every iteration will keep showing you plots, so make sure the number of iterations isn't too long.
Here 100 * .05 = 5s minimum (processing time per step may take longer than .05s)

#### Comments
Is using `Agent` > `self.model = model` a bad idea for ram purposes?

Can you think of a none `for loop` way of plotting location?

The only random numbers used are in:
line 45 > `shuffle` - shuffling order (only if requested).
line 52 > `random` - initialising location.

I call the animation created *pause-animations* as they aren't saved and run as the program does.
