# Particle Filter
'''

This function has originally been written by Kevin Minors, an intern at the University of Leeds

It has been adapted to work with the BusSim models

A particle filter designed to work with agent base models.
model requires: agents2state(), state2agents()
model requests: boundaries, agents[:].active
step requires: measured_state
save requests: true_state
'''
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

class ParticleFilter:
    '''
    A particle filter to model the dynamics of the
    state of the model as it develops in time.

    Parameters:

    'number_of_particles': The number of particles used to simulate the model
    'number_of_iterations': The number of iterations to run the model/particle filter
    'resample_window': The number of iterations between resampling particles
    'agents_to_visualise': The number of agents to plot particles for
    'particle_std': The standard deviation of the noise added to particle states
    'model_std': The standard deviation of the noise added to model observations
    'do_save': Boolean to determine if data should be saved and stats printed
    'do_ani': Boolean to determine if particle filter data should be animated
        and displayed
    '''
    def __init__(self, model, number_of_particles, arr_std=0,dep_std=0, traffic_std=0, resample_window=10, do_copies=True, do_save=False):
        '''
        Initialise Particle Filter

        Firstly, set all attributes using filter parameters. Set time and
        initialise base model using model parameters. Initialise particle
        models using a deepcopy of base model. Determine particle filter
        dimensions, set state of particles to the state of the base model,
        and then initialise all remaining arrays.
        '''
        self.time = 0
        # Dimensions
        self.number_of_particles = number_of_particles
        self.dimensions = len(model.agents2state().T)
        # Models
        self.models = list([deepcopy(model) for _ in range(self.number_of_particles)])
        for unique_id in range(len(self.models)):
            self.models[unique_id].unique_id = unique_id
        if not do_copies:
            # Only relevent if there is randomness in the initialisation of the model.
            for model in self.models:
                model.__init__(*model.params)
        # Filter
        self.states = np.empty((self.number_of_particles, self.dimensions))
        for particle in range(self.number_of_particles):
            self.states[particle] = self.models[particle].agents2state()
        self.weights = np.ones(self.number_of_particles)
        # Params
        self.arr_std = arr_std
        self.dep_std = dep_std
        self.traffic_std=traffic_std
        self.resample_window = resample_window
        # Save
        self.do_save = do_save
        if self.do_save:
            self.active = []
            self.means = []
            self.mean_errors = []
            self.variances = []
        return

    def step(self, measured_state, true_state=None):
        '''
        Step Particle Filter

        Loop through process. Predict the base model and particles
        forward. If the resample window has been reached,
        reweight particles based on distance to base model and resample
        particles choosing particles with higher weights. Then save
        and animate the data. When done, plot save figures.
        '''
        self.predict(measured_state)
        self.reweight(measured_state)
        self.resample(measured_state)
        if self.do_save:
            self.save(true_state)
        return


    def reweight(self, measured_state):
        '''def 
        Reweight

        Add noise to the base model state to get a measured state. Calculate
        the distance between the particle states and the measured base model
        state and then calculate the new particle weights as 1/distance.
        Add a small term to avoid dividing by 0. Normalise the weights.
        '''
        
        states = self.states[:, :len(measured_state)]  # For shorter measurements to state vectors
        
        #print(states)
        distance = np.linalg.norm(states - measured_state, axis=1)  #Frobenius norm
        #print(distance)
        self.weights = 1 / np.fmax(distance, 1e-99)  # to avoid fp_err
        #self.weights = np.exp(-np.fmax(distance, 1e-99))  # to avoid fp_err
        self.weights /= np.sum(self.weights)
        #print(self.weights)
        return 

    def resample(self,measured_state):  # systematic sampling
        '''
        Resample

        Calculate a random partition of (0,1) and then
        take the cumulative sum of the particle weights.
        Carry out a systematic resample of particles.
        Set the new particle states and weights and then
        update agent locations in particle models.
        '''
        if not self.time % self.resample_window:
            offset = (np.arange(self.number_of_particles) + np.random.uniform()) / self.number_of_particles
            cumsum = np.cumsum(self.weights)
            i, j = 0, 0
            indexes = np.zeros(self.number_of_particles, 'i')
            while i < self.number_of_particles and j < self.number_of_particles:
                if offset[i] < cumsum[j]:  # reject
                    indexes[i] = j
                    i += 1
                else:
                    j += 1
            self.states = self.states[indexes]
            #print(self.states)
            self.weights = self.weights[indexes]
            
            self.states[:,len(measured_state)+1:len(measured_state)+self.models[0].FleetSize-1] += np.random.normal(0, self.arr_std, self.states[:,len(measured_state)+1:len(measured_state)+self.models[0].FleetSize-1].shape)
            self.states[:,len(measured_state)+self.models[0].FleetSize+2:-3] += np.random.normal(0, self.dep_std, self.states[:,len(measured_state)+self.models[0].FleetSize+2:-3].shape)
            self.states[:,-1] += np.random.normal(0, self.traffic_std, self.states[:,-1].shape)

            #apply the measured_state
            #for s in range(len(self.states)):
            #    self.states[s,:len(measured_state)]=measured_state
        
        
        return

    def predict(self,measured_state):
        '''
        Predict

        Increment time. Step the base model. For each particle,
        step the particle model and then set the particle states
        as the agent locations with some added noise. Reassign the
        locations of the particle agents using the new particle
        states.

        This is the main interaction between the
        model and the particle filter.
        '''
       
        for particle in range(self.number_of_particles):
            self.models[particle].state2agents(self.states[particle])
            self.models[particle].step()
            self.states[particle] = self.models[particle].agents2state()
        self.time += 1
        return

    def predict_one(self, particle):  # not working
        map(self.predict_one, np.arange(self.number_of_particles))
        self.models[particle].state2agents(self.states[particle])
        self.models[particle].step()
        self.states[particle] = self.models[particle].agents2state()
        return

    def save(self, true_state):
        '''
        Save and Plot Save

        Calculate number of active agents, mean, and variance
        of particles and calculate mean error between the mean
        and the true base model state. Plot active agents,mean
        error and mean variance.
        '''
        states = self.states[:, :len(true_state)]  # For shorter measurements to state vectors

        try:
            activity = sum([agent.active for agent in self.model.agents])
            print('act')
        except AttributeError:
            activity = None
        self.active.append(activity)

        mean = np.average(states, weights=self.weights, axis=0)
        self.means.append(mean)

        variance = np.average((states - mean)**2, weights=self.weights, axis=0)
        self.variances.append(np.average(variance))

        if true_state is None:
            self.mean_errors.append(None)
        else:
            self.mean_errors.append(np.linalg.norm(mean - true_state, axis=0))
        return

    def plot_save(self):

        plt.figure(1)
        plt.plot(self.active)
        plt.ylabel('Active agents')

        plt.figure(2)
        plt.plot(self.mean_errors, '-k')
        plt.ylabel('Particle Mean Error')

        ax2 = plt.twinx()
        ax2.plot(self.variances)
        ax2.set_ylabel('Particle Variance Error')

        plt.show()

        print('Max mean error = ',max(self.mean_errors))
        print('Average mean error = ',np.average(self.mean_errors))
        print('Max mean variance = ',max(self.variances))
        print('Average mean variance = ',np.average(self.variances))
        return

    def ani(self, model, pf_agents=2):
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
            markersizes = 8*np.ones(self.weights.shape)

        for unique_id in range(pf_agents):
            for particle in range(self.number_of_particles):

                loc0 = model.agents[unique_id].location
                loc = self.models[particle].agents[unique_id].location
                locs = np.array([loc0, loc]).T

                plt.plot(*locs, '-k', alpha=.1, linewidth=.3)
                plt.plot(*loc, '+r', alpha=.3, markersize=markersizes[particle])

        # model
        state = model.agents2state(do_ravel=False).T
        plt.plot(state[0], state[1], '.k')
        try:
            plt.axis(np.ravel(model.boundaries, 'F'))
        except:
            pass
        plt.pause(1/30)
        return
