# Exercise File
'''
Todo:
filter
    measure the pf efficiency
    liturature review for pf
    multiprocessing
sspmm
    save data
BusSim
    last model error
'''
from data_assimilation.ParticleFilter import ParticleFilter
import numpy as np
import matplotlib.pyplot as plt
from jupyterthemes.jtplot import style as jtstyle
# jtstyle('gruvbox')
import time
from copy import deepcopy


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


if 1:  # sspmm

    from models.sspmm import easy_model
    model = easy_model()

    if True:  # Run the model
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
        pf.plot_save()


if 0:  # BusSim

    from models.BusSim_truth import pickle_Model
    pickle_Model()
    from models.BusSim_model import unpickle_Model
    model0, GroundTruth = unpickle_Model()
    t = np.arange(0, 10*len(GroundTruth), 10)

    for p in (20,):
        for std in (0, .01):
            for w in (1, 401):
                print(p, std, w)
                name = '../images/BusSim/' + 'p{}std{}w{}'.format(p, std, w)
                formatting = '.pdf'

                filter_params = {
                    'number_of_particles': p,
                    'particle_std': std,
                    'resample_window': w,
                    'do_copies': True,
                    'do_save': True
                    }
                model = deepcopy(model0)
                pf = ParticleFilter(model, **filter_params)

                for niter in range(len(GroundTruth)):
                    model.step()
                    if 0:  # Use Model
                        true_state = model.agents2state()
                        measured_state = true_state + np.random.normal(0, 0., true_state.shape)
                    else:  # Use Truth
                        true_state = GroundTruth[niter]
                        measured_state = true_state + np.random.normal(0, 0., true_state.shape)
                    pf.step(measured_state, true_state)

                    do_ani = False
                    if niter == len(GroundTruth)-1 or do_ani:
                        plt.figure(3, figsize=(16 / 2, 9 / 2))
                        plt.clf()
                        for particle in range(len(pf.models)-1):
                            x = np.array([bus.trajectory for bus in pf.models[particle].buses]).T
                            x[x <= 0] = np.nan
                            plt.plot(t, x, linewidth=.5)
                        x = GroundTruth[:niter+1, 1::4]
                        x[x <= 0] = np.nan
                        plt.plot(t, x, 'k')
                        plt.xlabel('Time (s)')
                        plt.ylabel('Distance (m)')
                        if do_ani:
                            plt.pause(1/30)


                do_save = False
                if do_save:
                    plt.savefig(name + ' Trails' + formatting)
                    time.sleep(.1)

                    pf.plot_save(do_save, name, formatting, t)
                    time.sleep(.1)
                else:
                    pf.plot_save(do_save, name, formatting, t)
                    time.sleep(.1)

                    plt.show()

