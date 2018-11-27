# Exercise File
'''
Todo:
filter
    measure the pf efficiency
    liturature review for pf
    multiprocessing
sspmm
    save data
'''
from data_assimilation.ParticleFilter import ParticleFilter
import numpy as np
import matplotlib.pyplot as plt


if 0:  # basic

    from models.basic import Model

    model = Model(100)

    filter_params = {
        'number_of_particles': 10,
        'particle_std': 0,
        'resample_window': 1,
        'do_copies': False,
        'do_save': False
    }
    pf = ParticleFilter(model, **filter_params)

    for _ in range(400):
        model.step()
        true_state = model.agents2state()
        measured_state = true_state + np.random.normal(0, 0., true_state.shape)
        pf.step(measured_state)
        pf.ani(model, 2)
    plt.plot_save()


if 0:  # sspmm

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

    if not True:  # Run the model
        model.batch()
    else:  # Run the particle filter
        model.do_save = False
        filter_params = {
            'number_of_particles': 10,
            'particle_std': 0,
            'resample_window': 1,
            'do_copies': False,
            'do_save': False
        }
        pf = ParticleFilter(model, **filter_params)
        for _ in range(model_params['batch_iterations']):
            model.step()
            true_state = model.agents2state()
            measured_state = true_state + np.random.normal(0, 0., true_state.shape)
            pf.step(measured_state, true_state)
            pf.ani(model, 2)
            plt.show()
        pf.plot_save()


if 1:  # BusSim

    from models.BusSim_truth import pickle_Model

    pickle_Model()

    from models.BusSim_model import unpickle_Model

    model, GroundTruth = unpickle_Model()

    filter_params = {
        'number_of_particles': 100,
        'particle_std': 0,
        'resample_window': 401,
        'do_copies': True,
        'do_save': True
        }
    pf = ParticleFilter(model, **filter_params)

    for niter in range(400):
        model.step()
        if 0:  # Use sim over truth
            true_state = model.agents2state()
            measured_state = true_state + np.random.normal(0, 0., true_state.shape)
        else:
            true_state = GroundTruth[niter]
            measured_state = true_state + np.random.normal(0, 0., true_state.shape)
        pf.step(measured_state, true_state)

    plt.figure(3)
    plt.clf()
    for i in range(len(pf.models)-1):  # something wrong with the last model
        plt.plot(np.array([bus.trajectory for bus in pf.models[i].buses]).T, alpha=.3)
    plt.plot(GroundTruth[:niter, 1::4], '-k')
    #plt.pause(1/30)
    pf.plot_save()
