# Exercise File
'''
Todo:
filter
    measure the pf efficiency
    model uncertainty
    liturature review for pf
sspmm
    multiprocessing
    save data
    initialised wiggle
'''
from data_assimilation.ParticleFilter import ParticleFilter
import numpy as np


if 0:  # use the basic Model

    from models.basic import Model

    model = Model(100)
    pf = ParticleFilter(model, number_of_particles=10, particle_std=.0, resample_window=1, do_copies=False)

    for _ in range(40):
        model.step()
        true_state = model.agents2state()
        measured_state = true_state + np.random.normal(0, 0., true_state.shape)
        pf.step(measured_state)

        pf.ani(model, 2)


if 1:  # Use sspmm

    from models.sspmm import Model

    model_params = {
        'width': 200,
        'height': 100,
        'pop_total': 300,
        'entrances': 2,
        'entrance_space': 4,
        'entrance_speed': 1,
        'exits': 2,
        'exit_space': 1,
        'speed_min': .1,
        'speed_desire_mean': 2,
        'speed_desire_std': 1,
        'separation': 4,
        'batch_iterations': 200,
        'do_save': True,
        'do_ani': True,
    }
    model = Model(model_params)
    if True:  # Run the model
        model.batch()
    else:  # Run the particle filter
        model.do_save = False
        filter_params = {
            'number_of_particles': 10,
            'particle_std': 0,
            'resample_window': 1,
            'do_copies': True,
            'do_save': False
        }
        pf = ParticleFilter(model, **filter_params)
        for _ in range(model_params['batch_iterations']):
            model.step()
            true_state = model.agents2state()
            measured_state = true_state + np.random.normal(0, 0., true_state.shape)
            pf.step(measured_state, true_state)
            pf.ani(model, 2)
        pf.plot_save()
