'''
Particle Filter - external data version
    modified by: patricia-ternes
    created: 30/04/2020
'''
from filter import Filter
from stationsim_gcs_model import Model


import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import multiprocessing
import warnings
import itertools
import time


class ParticleFilter(Filter): 
    '''
    A particle filter to model the dynamics of the
    state of the model as it develops in time.
    
    TODO: refactor to properly inherit from Filter.
    '''

    def __init__(self, ModelClass:Model, model_params:dict, filter_params:dict, numcores:int = None):
        '''
        Initialise Particle Filter
            
        PARAMETERS
         - number_of_particles:     The number of particles used to simulate the model
         - number_of_runs:          The number of times to run this particle filter (e.g. experiment)
         - resample_window:         The number of iterations between resampling particles
         - multi_step:              Whether to do all model iterations in between DA windows in one go
         - particle_std:            The standard deviation of the noise added to particle states
         - model_std:               The standard deviation of the error added to observations
         - agents_to_visualise:     The number of agents to plot particles for
         - model_std:               The standard deviation of the noise added to model observations
         - do_resample:             Whether or not to resample (default true, this is mainly for benchmarking)
         - do_save:                 Boolean to determine if data should be saved and stats printed
         - do_ani:                  Boolean to determine if particle filter data should be animated
                                    and displayed
         - show_ani:                If false then don't actually show the animation. The individual
                                    can be retrieved later from self.animation
         - external_data:           Boolean to determine whether base data should be created internally or loaded from external files.
        
        DESCRIPTION
        Firstly, set all attributes using filter parameters. Set time and
        initialise base model using model parameters. Initialise particle
        models using a deepcopy of base model. Determine particle filter 
        dimensions, initialise all remaining arrays, and set initial
        particle states to the base model state using multiprocessing. 
        '''
        for key, value in filter_params.items():
            setattr(self, key, value)
        self.time = 0
        self.number_of_iterations = model_params['batch_iterations']
        self.base_model = ModelClass(**model_params) # (Model does not need a unique id)        
        if self.external_data:
            self.set_initial_conditions()
        self.models = list([deepcopy(self.base_model) for _ in range(self.number_of_particles)])
        self.dimensions = len(self.base_model.get_state(sensor='location'))
        self.states = np.zeros((self.number_of_particles, self.dimensions))
        self.weights = np.ones(self.number_of_particles)
        self.indexes = np.zeros(self.number_of_particles, 'i')
        self.window_counter = 0 # Just for printing the progress of the PF
        # Pool object needed for multiprocessing
        if numcores == None:
            numcores = multiprocessing.cpu_count()

        # Assume that we do want to do resampling
        try:
            self.do_resample # Don't assume that the do_resample parameter has been set in the first place
        except AttributeError:
            self.do_resample = True
        if not self.do_resample:
            print("**Warning**: Not resampling. This should only be used for benchmarking")

        ## We get problems when there are more processes than particles (larger particle variance for some reason)
        #if numcores > self.number_of_particles:
        #    numcores = self.number_of_particles
        self.pool = multiprocessing.Pool(processes=numcores)
        if self.do_save or self.p_save:
            self.active_agents = []
            self.mean_states = [] # Mean state of all partciles, weighted by distance from observations
            self.mean_errors = [] # Mean distance between weighted mean state and the true state
            self.variances = []
            self.absolute_errors = [] # Unweighted distance between mean state and the true state
            self.unique_particles = []
            self.before_resample = [] # Records whether the errors were before or after resampling

        self.animation = [] # Keep a record of each plot created if animating so the individual ones can be viewed later

        #print("Creating initial states ... ")
        base_model_state = self.base_model.get_state(sensor='location')
        self.states = np.array([ self.initial_state(i, base_model_state) for i in range(self.number_of_particles )])
        #print("\t ... finished")
        print("Running filter with {} particles and {} runs (on {} cores) with {} agents.".format(
            filter_params['number_of_particles'], filter_params['number_of_runs'], numcores, model_params["pop_total"]),
            flush=True)

    def initial_state(self, particle_number, base_model_state):
        """
        Set the state of the particles to the state of the
        base model.
        """
        self.states[particle_number, :] = base_model_state
        return self.states[particle_number]

    def set_initial_conditions(self):
        '''
         to use external file to determine some agents parameters values;
        '''
        file_name = 'base_model/activation.dat'
        agentID, time_activation, gate_in, gate_out, speed, loc_desireX, loc_desireY = np.loadtxt(file_name,unpack=True)
        i = 0
        for agent in self.base_model.agents:
            agent.unique_id = int(agentID[i])
            agent.gate_in = int(gate_in[i])
            agent.gate_out = int(gate_out[i])
            agent.loc_desire = [loc_desireX[i], loc_desireY[i]]
            agent.steps_activate = time_activation[i]
            agent.speed = speed[i]
            i += 1

    @classmethod
    def assign_agents(cls, particle_num: int, state: np.array, model: Model):
        """
        Assign the state of the particles to the
        locations of the agents.
        :param particle_num
        :param state: The state of the particle to be assigned
        :param model: The model to assign the state to
        :type model: Return the model after having the agents assigned according to the state.
        """
        model.set_state(state, sensor='location')
        return model

    @classmethod
    def step_particle(cls, particle_num: int, model: Model, num_iter: int, particle_std: float, particle_shape: tuple):
        """
        Step a particle, assign the locations of the
        agents to the particle state with some noise, and
        then use the new particle state to set the location
        of the agents.

        :param particle_num: The particle number to step
        :param model: A pointer to the model object associated with the particle that needs to be stepped
        :param num_iter: The number of iterations to step
        :param particle_std: the particle noise standard deviation
        :param particle_shape: the shape of the particle array
        """
        # Force the model to re-seed its random number generator (otherwise each child process
        # has the same generator https://stackoverflow.com/questions/14504866/python-multiprocessing-numpy-random
        model.set_random_seed()
        for i in range(num_iter):
            model.step()

        noise = np.random.normal(0, particle_std ** 2, size=particle_shape)
        state = model.get_state(sensor='location') + noise
        model.set_state(state, sensor='location')
        return model, state

    def step(self):
        '''
        Step Particle Filter

        DESCRIPTION
        Loop through process. Predict the base model and particles
        forward. If the resample window has been reached,
        reweight particles based on distance to base model and resample
        particles choosing particles with higher weights. Then save
        and animate the data. When done, plot save figures.

        Note: if the multi_step is True then predict() is called once, but
        steps the model forward until the next window. This is quicker but means that
        animations and saves will only report once per window, rather than
        every iteration

        :return: Information about the run as a list with two tuples. The first
        tuple has information about the state of the PF *before* reweighting,
        the second has the state after reweighting.
        Each tuple has the following:
           min(self.mean_errors) - the error of the particle with the smallest error
           max(self.mean_errors) - the error of the particle with the largest error
           np.average(self.mean_errors) - average of all particle errors
           min(self.variances) - min particle variance
           max(self.variances) - max particle variance
           np.average(self.variances) - mean particle variance
        '''
        print("Starting particle filter step()")

        try:

            window_start_time = time.time()  # time how long each window takes
            while self.time < self.number_of_iterations:

                # Whether to run predict repeatedly, or just once
                numiter = 1
                if self.multi_step:
                    self.time += self.resample_window
                    numiter = self.resample_window
                else:
                    self.time += 1

                # See if some particles still have active agents
                if any([agent.status != 2 for agent in self.base_model.agents]):
                    # print(self.time/self.number_of_iterations)

                    self.predict(numiter=numiter)

                    if self.time % self.resample_window == 0:
                        self.window_counter += 1

                        # Store the model states before and after resampling
                        if self.do_save or self.p_save:
                            self.save(before=True)


                        if self.do_resample: # Can turn off resampling for benchmarking
                            self.reweight()
                            self.resample()

                        # Store the model states before and after resampling
                        if self.do_save or self.p_save:
                            self.save(before=False)

                        # Animate this window
                        if self.do_ani:
                            self.ani()

                        print("\tFinished window {}, step {} (took {}s)".format(
                            self.window_counter, self.time, round(float(time.time() - window_start_time), 2)))
                        window_start_time = time.time()

                    elif self.multi_step:
                        assert (
                            False), "Should not get here, if multi_step is true then the condition above should always run"

                else:
                    pass # Don't print the message below any more
                    #print("\tNo more active agents. Finishing particle step")

            if self.plot_save:
                self.p_save()

            # Return the errors and variences before and after sampling (if we're saving information)
            # Useful for debugging in console:
            # for i, a in enumerate(zip([x[1] for x in zip(self.before_resample, self.mean_errors) if x[0] == True],
            #                          [x[1] for x in zip(self.before_resample, self.mean_errors) if x[0] == False])):
            #    print("{} - before: {}, after: {}".format(i, a[0], a[1]))
            if self.do_save:
                if self.mean_errors == []:
                    warnings.warn("For some reason the mean_errors array is empty. Cannot store errors for this run.")
                    return

                # Return two tuples, one with the about the error before reweighting, one after

                # Work out which array indices point to results before and after reweighting
                before_indices = [i for i, x in enumerate(self.before_resample) if x]
                after_indices = [i for i, x in enumerate(self.before_resample) if not x]

                result = []

                for before in [before_indices, after_indices]:
                    result.append([
                        min(np.array(self.mean_errors)[before]),
                        max(np.array(self.mean_errors)[before]),
                        np.average(np.array(self.mean_errors)[before]),
                        min(np.array(self.absolute_errors)[before]),
                        max(np.array(self.absolute_errors)[before]),
                        np.average(np.array(self.absolute_errors)[before]),
                        min(np.array(self.variances)[before]),
                        max(np.array(self.variances)[before]),
                        np.average(np.array(self.variances)[before])
                    ])
                return result

            # If not saving then just return null
            return

        finally: # Whatever happens, make sure the multiprocessing pool is colsed
            self.pool.close()

    def predict(self, numiter=1):
        '''
        Predict

        DESCRIPTION
        Increment time. Step the base model. Use a multiprocessing method to step
        particle models, set the particle states as the agent
        locations with some added noise, and reassign the
        locations of the particle agents using the new particle
        states. We extract the models and states from the stepped
        particles variable.

        :param numiter: The number of iterations to step (usually either 1, or the  resample window
        '''
        if not self.external_data:
            for i in range(numiter):
                self.base_model.step()

        stepped_particles = self.pool.starmap(ParticleFilter.step_particle, list(zip( \
            range(self.number_of_particles),  # Particle numbers (in integer)
            [m for m in self.models],  # Associated Models (a Model object)
            [numiter] * self.number_of_particles,  # Number of iterations to step each particle (an integer)
            [self.particle_std] * self.number_of_particles,  # Particle std (for adding noise) (a float)
            [s.shape for s in self.states],  # Shape (for adding noise) (a tuple)
        )))

        self.models = [stepped_particles[i][0] for i in range(len(stepped_particles))]
        self.states = np.array([stepped_particles[i][1] for i in range(len(stepped_particles))])

        return

    def reweight(self):
        '''
        Reweight

        DESCRIPTION
        Add noise to the base model state to get a measured state, or
        use external data to get a measured state. Calculate
        the distance between the particle states and the measured base model
        state and then calculate the new particle weights as 1/distance.
        Add a small term to avoid dividing by 0. Normalise the weights.
        '''
        if self.external_data:
            file_name = 'base_model/frame_' + str(self.time)+ '.dat'

            try:
                agentID, x, y = np.loadtxt(file_name,unpack=True)
                j = 0
                for agent in self.base_model.agents:
                    if (agent.unique_id in agentID):
                        
                        agent.status = 1
                        agent.location = [x[j], y[j]]
                        j += 1
                    elif (agent.status == 1):
                        agent.status = 2
            except TypeError:
                '''
                This error occurs when only one agent is active. In
                this case, the data is read as a float instead of an
                array.
                '''
                for agent in self.base_model.agents:
                    if (agent.unique_id == agentID):
                        agent.status = 1
                        agent.location = [x, y]
                    elif (agent.status == 1):
                        agent.status = 2
            except ValueError:
                '''
                 This error occurs when there is no active agent in
                 the frame.
                 - Deactivate all active agents.
                '''
                for agent in self.base_model.agents:
                    if (agent.status == 1):
                        agent.status = 2

            except OSError:
                '''
                This error occurs when there is no external file to
                read. It should only occur at the end of the simulation.
                - Deactivate all agent.
                '''
                for agent in self.base_model.agents:
                    agent.status = 2
            
            measured_state = self.base_model.get_state(sensor='location')
        else:        
            measured_state = (self.base_model.get_state(sensor='location')
                              + np.random.normal(0, self.model_std ** 2, size=self.states.shape))

        distance = np.linalg.norm(self.states - measured_state, axis=1)
        self.weights = 1 / (distance + 1e-9) ** 2
        self.weights /= np.sum(self.weights)

        return

    def resample(self):
        '''
        Resample

        DESCRIPTION
        Calculate a random partition of (0,1) and then
        take the cumulative sum of the particle weights.
        Carry out a systematic resample of particles.
        Set the new particle states and weights and then
        update agent locations in particle models using
        multiprocessing methods.
        '''
        offset_partition = ((np.arange(self.number_of_particles)
                             + np.random.uniform()) / self.number_of_particles)
        cumsum = np.cumsum(self.weights)
        i, j = 0, 0
        while i < self.number_of_particles:
            if offset_partition[i] < cumsum[j]:
                self.indexes[i] = j
                i += 1
            else:
                j += 1

        self.states[:] = self.states[self.indexes]
        self.weights[:] = self.weights[self.indexes]

        # self.unique_particles.append(len(np.unique(self.states,axis=0)))

        # Could use pool.starmap here, but it's quicker to do it in a single process
        self.models = list(itertools.starmap(ParticleFilter.assign_agents, list(zip(
            range(self.number_of_particles),  # Particle numbers (in integer)
            [s for s in self.states],  # States
            [m for m in self.models]  # Associated Models (a Model object)
        ))))
        return

    def save(self, before: bool):
        '''
        Save

        DESCRIPTION
        Calculate number of active agents, mean, and variance
        of particles and calculate mean error between the mean
        and the true base model state.

        :param before: whether this is being called before or after resampling as this will have a big impact on
        what the errors mean (if they're after resampling then they should be low, before and they'll be high)
        '''
        self.active_agents.append(sum([agent.status == 1 for agent in self.base_model.agents]))

        active_states = [agent.status == 1 for agent in self.base_model.agents for _ in range(2)]

        if any(active_states):
            # Mean and variance state of all particles, weighted by their distance to the observation
            mean = np.average(self.states[:, active_states], weights=self.weights, axis=0)
            unweighted_mean = np.average(self.states[:, active_states], axis=0)
            variance = np.average((self.states[:, active_states] - mean) ** 2, weights=self.weights, axis=0)

            self.mean_states.append(mean)
            self.variances.append(np.average(variance))
            self.before_resample.append(before)  # Whether this save reflects the errors before or after resampling

            truth_state = self.base_model.agents2state()
            self.mean_errors.append(np.linalg.norm(mean - truth_state[active_states], axis=0))
            self.absolute_errors.append(np.linalg.norm(unweighted_mean - truth_state[active_states], axis=0))

            # min(mean_errors) is returning empty. CHeck small values for agents/particles

        return

    def p_save(self):
        '''
        Plot Save

        DESCRIPTION
        Plot active agents, mean error and mean variance.
        '''
        plt.figure(2)
        plt.plot(self.active_agents)
        plt.ylabel('Active agents')
        plt.show()

        plt.figure(3)
        plt.plot(self.mean_errors)
        plt.ylabel('Mean Error')
        plt.show()

        plt.figure(4)
        plt.plot(self.variances)
        plt.ylabel('Mean Variance')
        plt.show()

        plt.figure(5)
        plt.plot(self.unique_particles)
        plt.ylabel('Unique Particles')
        plt.show()

        print('Max mean error = ', max(self.mean_errors))
        print('Average mean error = ', np.average(self.mean_errors))
        print('Max mean variance = ', max(self.variances[2:]))
        print('Average mean variance = ', np.average(self.variances[2:]))

    def ani(self):
        '''
        Animate

        DESCRIPTION
        Plot the base model state and some of the
        particles. Only do this if there is at least 1 active
        agent in the base model. We adjust the markersizes of
        each particle to represent the weight of that particle.
        We then plot some of the agent locations in the particles
        and draw lines between the particle agent location and
        the agent location in the base model.
        '''
        if any([agent.status == 1 for agent in self.base_model.agents]):

            if not self.show_ani:
                # Turn interactive plotting off
                plt.ioff()

            fig = plt.figure(len(self.animation)+1) # Make sure figures aren't overridden
            plt.clf()

            markersizes = self.weights
            if np.std(markersizes) != 0:
                markersizes *= 4 / np.std(markersizes)  # revar
            markersizes += 8 - np.mean(markersizes)  # remean

            particle = -1
            for model in self.models:
                particle += 1
                markersize = np.clip(markersizes[particle], .5, 8)
                for agent in model.agents[:self.agents_to_visualise]:
                    if agent.status == 1:
                        unique_id = agent.unique_id
                        if self.base_model.agents[unique_id].status == 1:
                            locs = np.array([self.base_model.agents[unique_id].location, agent.location]).T
                            plt.plot(*locs, '-k', alpha=.5, linewidth=.5)
                            plt.plot(*agent.location, 'or', alpha=.3, markersize=markersize)

            for agent in self.base_model.agents:
                if agent.status == 1:
                    plt.plot(*agent.location, 'sk', markersize=4)

            plt.axis(np.ravel(self.base_model.boundaries, 'F'))
            plt.title(f"{self.models[0].pop_total} agents, {self.number_of_particles} particles, {self.time} iterations", fontsize=13)
            plt.xlabel("X position")
            plt.ylabel("Y position")
            if self.show_ani:
                plt.pause(1.0 / 4) # If we're showing animations then show and pause briefly

            self.animation.append(fig) # Store this plot to browse later


if __name__ == '__main__':
    warnings.warn("The particle_filter.py code should not be run directly. Create a separate script and use that "
                  "to run experimets (e.g. see ABM_DA/experiments/pf_experiments/run_pf.py")
    print("Nothing to do")
