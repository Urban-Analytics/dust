# Particle Filter
'''
TODO:
    Multiprocessing
    time -> step_id
    comments


A particle filter designed to work with agent base models.
    model requires: agents2state(), state2agents(), agents[:].active
    model requests: boundaries
    do_copies requires: model.params
    step requires: measured_state
    save requires: model.agents[:]
    save requests: true_state (measurement_state use otherwise)
'''
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


class ParticleFilter:
    '''
    A particle filter to model the dynamics of the state of the model as it develops in time.

    Parameters:
        'number_of_particles': The number of particles used to simulate the model
        'number_of_iterations': The number of iterations to run the model/particle filter
        'resample_window': The number of iterations between resampling particles
        'agents_to_visualise': The number of agents to plot particles for
        'particle_std': The standard deviation of the noise added to particle states
        'model_std': The standard deviation of the noise added to model observations
        'do_save': Boolean to determine if data should be saved and stats printed
        'do_ani': Boolean to determine if particle filter data should be animated and displayed
    '''

    def __init__(self, model, number_of_particles=10, particle_std=0, resample_window=10, do_copies=True, do_save=False):
        '''
        Firstly, set all attributes using filter parameters. Set time and initialise base model using model parameters. Initialise particle models using a deepcopy of base model. Determine particle filter dimensions, set state of particles to the state of the base model, and then initialise all remaining arrays.
        '''
        self.time = 0
        # Dimensions
        self.number_of_particles = number_of_particles
        self.dimensions = len(model.agents2state())
        # Models
        self.models = list([deepcopy(model) for _ in range(self.number_of_particles)])
        if not do_copies:  # Only relevent if there is randomness in the initialisation of the model.
            [model.__init__(*model.params) for model in self.models]
        for unique_id in range(len(self.models)):
            self.models[unique_id].unique_id = unique_id
        # Filter
        self.states = np.empty((self.number_of_particles, self.dimensions))
        for particle in range(self.number_of_particles):
            self.states[particle] = self.models[particle].agents2state()
        self.weights = np.empty(self.number_of_particles).fill(1 / self.number_of_particles)  # Initial Equal Weights
        # Params
        self.particle_std = particle_std
        self.resample_window = resample_window
        # Save
        self.do_save = do_save
        if self.do_save:
            self.datum_per_agent = int(self.dimensions / len(model.buses))
            self.active = []
            self.mean = []
            self.var = []
        return

    def step(self, measured_state, true_state=None):
        '''
        Loop through process. Predict the base model and particles forward. If the resample window has been reached, reweight particles based on distance to base model and resample particles choosing particles with higher weights. Then save and animate the data. When done, plot save figures.
        '''
        self.predict()
        self.reweight(measured_state)
        self.resample()
        if self.do_save:
            if true_state is None:
                self.save(measured_state)
            else:
                self.save(true_state)
        return

    def reweight(self, measured_state):
        '''
        Add noise to the base model state to get a measured state. Calculate the distance between the particle states and the measured base model state and then calculate the new particle weights as 1/distance. Add a small term to avoid dividing by 0. Normalise the weights.
        '''
        states = self.states[:, :len(measured_state)]  # For shorter measurements to state vectors
        distance = np.linalg.norm(states - measured_state, axis=1)
        self.weights = 1 / np.fmax(distance, 1e-99)  # to avoid fp_err
        self.weights /= np.sum(self.weights)
        return

    def resample(self):
        '''
        Systematic Resampling
        Calculate a random partition of (0,1) and then take the cumulative sum of the particle weights. Carry out a systematic resample of particles. Set the new particle states and weights and then update agent locations in particle models using multiprocessing methods.
        '''
        if not self.time % self.resample_window:
            offset_partition = (np.arange(self.number_of_particles) + np.random.uniform()) / self.number_of_particles
            cumsum = np.cumsum(self.weights)
            i, j = 0, 0
            indexes = np.zeros(self.number_of_particles, 'i')
            while i < self.number_of_particles and j < self.number_of_particles:
                if offset_partition[i] < cumsum[j]:  # reject
                    indexes[i] = j
                    i += 1
                else:
                    j += 1
            self.states = self.states[indexes]
            self.weights = self.weights[indexes]
        if self.particle_std:
            self.states += np.random.normal(0, self.particle_std, self.states.shape)
        return

    def predict(self):
        '''
        Increment time. Step the base model. Set self as a constant in step_particles and then use a multiprocessing method to step particle models, set the particle states as the agent locations with some added noise, and reassign the locations of the particle agents using the new particle states. We extract the models and states from the stepped particles variable.

        This is the only interaction between the model and the particle filter.
        '''
        for particle in range(self.number_of_particles):
            self.models[particle].state2agents(self.states[particle])
            self.models[particle].step()
            self.states[particle] = self.models[particle].agents2state()
        self.time += 1
        return

    def save(self, true_state):
        '''
        Calculate number of active agents, mean, and variance of particles and calculate mean error between the mean and the true base model state. Plot active agents, mean error and mean variance.
        '''
        # For shorter measurements than states
        states = self.states[:, :len(true_state)]
        # Ignore inactive agents with a mask
        activity_mask = np.zeros(states.shape, dtype='b')
        active = 0
        for particle in range(self.number_of_particles):
            for unique_id in range(len(self.models[particle].buses)):
                if self.models[particle].buses[unique_id].status in (1, 2):
                    activity_mask[particle, unique_id * self.datum_per_agent:unique_id * self.datum_per_agent + self.datum_per_agent] = True
                    active += 1
        states = np.ma.masked_where(activity_mask, states)
        # Calculate
        means = np.average(states, weights=self.weights, axis=0)
        variances = np.average((states - means)**2, weights=self.weights, axis=0)
        # Save
        self.active.append(active / self.number_of_particles)
        self.var.append(np.average(variances))
        self.mean.append(np.linalg.norm(means - true_state, axis=0))
        return

    def save_plot(self, do_save=False, name='', formatting='.jpg', t=None):
        '''
        Call this function to plot the statistics
        '''
        if self.do_save:
            if t is None:
                t = np.arange(self.time)

            plt.figure(1, figsize=(16 / 2, 9 / 2))
            plt.plot(self.active)
            plt.ylabel('Active agents')
            if do_save:
                plt.savefig(name + ' Active' + formatting)

            _, ax1 = plt.subplots(1, figsize=(16 / 2, 9 / 2))
            x = self.mean
            l1, = ax1.plot(x, '-m')
            ax1.set_ylabel('Particle Mean Error')

            ax2 = plt.twinx()
            x = self.var
            l2, = ax2.plot(x, '--b')
            ax2.set_ylabel('Particle Variance Error')

            plt.xlabel('Time (s)')
            plt.legend([l1, l2], ['Mean Error', 'Mean Variance'], loc=0)
            if do_save:
                plt.savefig(name + ' Analysis' + formatting)

            print('Max mean error = ', max(self.mean))
            print('Average mean error = ', np.average(self.mean))
            print('Max mean variance = ', max(self.var))
            print('Average mean variance = ', np.average(self.var))
        else:
            print('Warning:  do_save=False')
        return

    def ani(self, model, pf_agents=None):
        '''
        Animate

        Plot the base model state and some of the
        particles. Only do this if there is at least 1 active
        agent in the base model. We adjust the markersizes of
        each particle to represent the weight of that particle.
        We then plot some of the agent locations in the particles
        and draw lines between the particle agent location and
        the agent location in the base model.
        '''
        plt.figure(1)
        plt.clf()

        # pf
        if np.std(self.weights):
            markersizes = self.weights
            markersizes *= 4 / np.std(markersizes)    # revar
            markersizes += 4 - np.mean(markersizes)   # remean
            markersizes = np.clip(markersizes, 1, 8)  # clip
        else:
            markersizes = 8 * np.ones(self.weights.shape)

        for m in self.models:
            for a in m.buses[:pf_agents]:
                if a.active == 1:
                    loc0 = model.buses[a.unique_id].location
                    loc = a.location
                    locs = np.array([loc0, loc]).T
                    plt.plot(*locs, '-k', alpha=.1, linewidth=.3)
                    plt.plot(*loc, '+r', alpha=.3, markersize=markersizes[m.unique_id])
        # model
        state = model.agents2state(do_ravel=False).T
        plt.plot(state[0], state[1], '.k')
        try:
            plt.axis(np.ravel(model.boundaries, 'F'))
        except:
            pass
        plt.pause(1 / 30)
        return
