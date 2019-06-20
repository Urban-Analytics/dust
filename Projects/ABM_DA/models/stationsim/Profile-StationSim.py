"""
Profile StationSim

Simply run stationsim N (10) times. This can be called from the profiling tools in Intellij IDEA to
profile the funtion times.
"""

from StationSim_KM import Model

model_params = {
    'width': 200,
    'height': 100,
    'pop_total': 700,
    'entrances': 3,
    'entrance_space': 2,
    'entrance_speed': .1,
    'exits': 2,
    'exit_space': 1,
    'speed_min': .1,
    'speed_desire_mean': 1,
    'speed_desire_std': 1,
    'separation': 2,
    'batch_iterations': 900,
    'do_save': False,
    'do_ani': False,
}

# Run the model 10 times
for i in range(10):
    print("Run {} ...".format(i))
    Model(model_params).batch()

print("Finished batch model run")
