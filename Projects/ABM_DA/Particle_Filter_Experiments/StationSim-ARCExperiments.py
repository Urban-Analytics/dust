import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from copy import deepcopy
import multiprocessing
import time
import warnings
import sys
import itertools

def error(text='Self created error.'):
    from sys import exit
    print()
    exit(text)
    return


class Agent:

    def __init__(self, model, unique_id):
        # Required
        self.unique_id = unique_id
        self.active = 0  # 0 Not Started, 1 Active, 2 Finished
        model.pop_active += 1
        # Location
        self.location = model.loc_entrances[np.random.randint(model.entrances)]
        self.location[1] += model.entrance_space * (np.random.uniform() - .5)
        self.loc_desire = model.loc_exits[np.random.randint(model.exits)]
        # Parameters
        self.time_activate = np.random.exponential(model.entrance_speed)
        self.speed_desire = max(np.random.normal(model.speed_desire_mean, model.speed_desire_std), 2*model.speed_min)
        self.speeds = np.arange(self.speed_desire, model.speed_min, -model.speed_step)
        if model.do_save:
            self.history_loc = []
        return

    def step(self, model):
        if self.active == 0:
            self.activate(model)
        elif self.active == 1:
            self.move(model)
            self.exit_query(model)
            self.save(model)
        return

    def activate(self, model):
        if not self.active and model.time_id > self.time_activate:
            self.active = 1
            self.time_start = model.time_id
            self.time_expected = np.linalg.norm(self.location - self.loc_desire) / self.speed_desire
        return

    def move(self, model):
        for speed in self.speeds:
            # Direct
            new_location = self.lerp(self.loc_desire, self.location, speed)
            if not self.collision(model, new_location):
                break
            elif speed == self.speeds[-1]:
                # Wiggle
                new_location = self.location + np.random.randint(-1, 1+1, 2)
        # Rebound
        within_bounds = all(model.boundaries[0] <= new_location) and all(new_location <= model.boundaries[1])
        if not within_bounds:
            new_location = np.clip(new_location, model.boundaries[0], model.boundaries[1])
        # Move
        self.location = new_location
        return

    def collision(self, model, new_location):
        within_bounds = all(model.boundaries[0] <= new_location) and all(new_location <= model.boundaries[1])
        if not within_bounds:
            collide = True
        elif self.neighbourhood(model, new_location):
            collide = True
        else:
            collide = False
        return collide

    def neighbourhood(self, model, new_location, do_kd_tree=True):
        neighbours = False
        neighbouring_agents = model.tree.query_ball_point(new_location, model.separation)
        for neighbouring_agent in neighbouring_agents:
            agent = model.agents[neighbouring_agent]
            if agent.active == 1 and new_location[0] <= agent.location[0]:
                neighbours = True
                break
        return neighbours

    def lerp(self, loc1, loc2, speed):
        distance = np.linalg.norm(loc1 - loc2)
        loc = loc2 + speed * (loc1 - loc2) / distance
        return loc

    def exit_query(self, model):
        if np.linalg.norm(self.location - self.loc_desire) < model.exit_space:
            self.active = 2
            model.pop_active -= 1
            model.pop_finished += 1
            if model.do_save:
                time_delta = model.time_id - self.time_start
                model.time_taken.append(time_delta)
                time_delta -= self.time_expected
                model.time_delayed.append(time_delta)
        return

    def save(self, model):
        if model.do_save:
            self.history_loc.append(self.location)
        return


class Model:

    def __init__(self, params):
        self.params = (params,)
        [setattr(self, key, value) for key, value in params.items()]
        self.speed_step = (self.speed_desire_mean - self.speed_min) / 3  # Average number of speeds to check
        # Batch Details
        self.time_id = 0
        self.step_id = 0
        if self.do_save:
            self.time_taken = []
            self.time_delayed = []
        # Model Parameters
        self.boundaries = np.array([[0, 0], [self.width, self.height]])
        self.pop_active = 0
        self.pop_finished = 0
        # Initialise
        self.initialise_gates()
        self.agents = list([Agent(self, unique_id) for unique_id in range(self.pop_total)])
        return

    def step(self):
        if self.pop_finished < self.pop_total:
            self.kdtree_build()
            [agent.step(self) for agent in self.agents]
        self.time_id += 1
        self.step_id += 1
        return

    def initialise_gates(self):
        # Entrances
        self.loc_entrances = np.zeros((self.entrances, 2))
        self.loc_entrances[:, 0] = 0
        if self.entrances == 1:
            self.loc_entrances[0, 1] = self.height / 2
        else:
            self.loc_entrances[:, 1] = np.linspace(self.height / 4, 3 * self.height / 4, self.entrances)
        # Exits
        self.loc_exits = np.zeros((self.exits, 2))
        self.loc_exits[:, 0] = self.width
        if self.exits == 1:
            self.loc_exits[0, 1] = self.height / 2
        else:
            self.loc_exits[:, 1] = np.linspace(self.height / 4, 3 * self.height / 4, self.exits)
        return

    def kdtree_build(self):
        state = self.agents2state(do_ravel=False)
        self.tree = cKDTree(state)
        return

    def agents2state(self, do_ravel=True):
        state = [agent.location for agent in self.agents]
        if do_ravel:
            state = np.ravel(state)
        else:
            state = np.array(state)
        return state

    def state2agents(self, state):
        for i in range(len(self.agents)):
            self.agents[i].location = state[2 * i:2 * i + 2]
        return

    def batch(self):
        for i in range(self.batch_iterations):
            self.step()
            if self.do_ani:
                self.ani()
            if self.pop_finished == self.pop_total:
                print('Everyone made it!')
                break
        if self.do_save:
            self.save_stats()
            self.save_plot()
        return

    def ani(self):
        plt.figure(1)
        plt.clf()
        for agent in self.agents:
            if agent.active == 1:
                plt.plot(*agent.location, '.k')#, markersize=4)
        plt.axis(np.ravel(self.boundaries, 'F'))
        plt.xlabel('Corridor Width')
        plt.ylabel('Corridor Height')
        plt.pause(1 / 30)
        return

    def save_ani(self):



        return

    def save_plot(self):
        # Trails
        plt.figure()
        for agent in self.agents:
            if agent.active == 0:
                colour = 'r'
            elif agent.active == 1:
                colour = 'b'
            else:
                colour = 'm'
            locs = np.array(agent.history_loc).T
            plt.plot(locs[0], locs[1], color=colour, linewidth=.5)
        plt.axis(np.ravel(self.boundaries, 'F'))
        plt.xlabel('Corridor Width')
        plt.ylabel('Corridor Height')
        plt.legend(['Agent trails'])
        # Time Taken, Delay Amount
        plt.figure()
        plt.hist(self.time_taken, alpha=.5, label='Time taken')
        plt.hist(self.time_delayed, alpha=.5, label='Time delay')
        plt.xlabel('Time')
        plt.ylabel('Number of Agents')
        plt.legend()

        plt.show()
        return

    def save_stats(self):
        print()
        print('Stats:')
        print('Finish Time: ' + str(self.time_id))
        print('Active / Finished / Total agents: ' + str(self.pop_active) + '/' + str(self.pop_finished) + '/' + str(self.pop_total))
        print('Average time taken: ' + str(np.mean(self.time_taken)) + 's')
        return

class ParticleFilter: 
    '''
    A particle filter to model the dynamics of the
    state of the model as it develops in time.
    '''

    def __init__(self, Model, model_params, filter_params):
        '''
        Initialise Particle Filter
            
        PARAMETERS
         - number_of_particles:     The number of particles used to simulate the model
         - number_of_iterations:    The number of iterations to run the model/particle filter
         - resample_window:         The number of iterations between resampling particles
         - agents_to_visualise:     The number of agents to plot particles for
         - particle_std:            The standard deviation of the noise added to particle states
         - model_std:               The standard deviation of the noise added to model observations
         - do_save:                 Boolean to determine if data should be saved and stats printed
         - do_ani:                  Boolean to determine if particle filter data should be animated
                                    and displayed
        
        DESCRIPTION
        Firstly, set all attributes using filter parameters. Set time and
        initialise base model using model parameters. Initialise particle
        models using a deepcopy of base model. Determine particle filter 
        dimensions, initialise all remaining arrays, and set initial
        particle states to the base model state using multiprocessing. 
        '''    
        print(1)
        for key, value in filter_params.items():
            setattr(self, key, value)
        self.time = 0
        self.number_of_iterations = model_params['batch_iterations']
        self.base_model = Model(model_params)
        self.models = list([deepcopy(self.base_model) for _ in range(self.number_of_particles)])  
        self.dimensions = len(self.base_model.agents2state())
        self.states = np.zeros((self.number_of_particles, self.dimensions))
        self.weights = np.ones(self.number_of_particles)
        self.indexes = np.zeros(self.number_of_particles, 'i')
        self.window_counter = 0 # Just for printing the progress of the PF
        if self.do_save:
            self.active_agents = []
            self.mean_states = [] # Mean state of all partciles, weighted by distance from observations
            self.mean_errors = [] # Mean distance between weighted mean state and the true state
            self.variances = []
            self.absolute_errors = [] # Unweighted distance between mean state and the true state
            self.unique_particles = []
            self.before_resample = [] # Records whether the errors were before or after resampling

        print("Creating initial states ... ")
        base_model_state = self.base_model.agents2state()
        self.states = np.array([ self.initial_state2(i, base_model_state) for i in range(self.number_of_particles )])
        print("\t ... finished")
        #pool.starmap(ParticleFilter.initial_state,list(zip(range(self.number_of_particles),[self]*self.number_of_particles))))

    def initial_state2(self, particle_number, base_model_state):
        """
        Set the state of the particles to the state of the
        base model.
        """
        self.states[particle_number, :] = base_model_state
        return self.states[particle_number]

    # Multiprocessing methods
    @classmethod
    def initial_state(cls, particle,self):
        """
        Set the state of the particles to the state of the
        base model.
        """
        warnings.warn(
            "initial_state has been replaced with initial_state2 (non multiprocess) and should no longer be used",
            DeprecationWarning
        )
        self.states[particle,:] = self.base_model.agents2state()
        return self.states[particle]

    @classmethod
    def assign_agents(cls, particle_num:int, state:np.array, model:Model):
        """
        Assign the state of the particles to the
        locations of the agents.
        :param particle_num
        :param state: The state of the particle to be assigned
        :param model: The model to assign the state to
        :type model: Return the model after having the agents assigned according to the state.
        """
        model.state2agents(state)
        return model

    @classmethod
    def step_particles(cls, particle_num, self):
        """
        Step each particle model, assign the locations of the
        agents to the particle state with some noise, and
        then use the new particle state to set the location
        of the agents.

        :param particle_num: The particle number to step
        :param self: A pointer to the calling ParticleFilter object.
        """
        warnings.warn(
            "step_particles has been replaced with step_particle and should no longer be used",
            DeprecationWarning
        )
        self.models[particle_num].step()
        self.states[particle_num] = (self.models[particle_num].agents2state()
                                     + np.random.normal(0, self.particle_std ** 2,
                                                        size=self.states[particle_num].shape))
        self.models[particle_num].state2agents(self.states[particle_num])
        return self.models[particle_num], self.states[particle_num]

    @classmethod
    def step_particle(cls, particle_num:int, model:Model, num_iter:int, particle_std:float, particle_shape:tuple):
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
        #self.models[particle_num].step()
        #warnings.warn(
        #    "ParticleFilter.step_particle has been replaced with a global step_particle method. This one should no longer be used",
        #    DeprecationWarning
        #)
        for i in range(num_iter):
            model.step()

        state = (model.agents2state() +
                 np.random.normal(0, particle_std ** 2, size=particle_shape))
        model.state2agents(state)
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

        window_start_time = time.time() # time how long each window takes
        while self.time < self.number_of_iterations:

            # Whether to run predict repeatedly, or just once
            numiter = 1
            if self.multi_step:
                self.time += self.resample_window
                numiter = self.resample_window
            else:
                self.time += 1

            # See if some particles still have active agents
            if any([agent.active != 2 for agent in self.base_model.agents]):
                #print(self.time/self.number_of_iterations)

                self.predict(numiter=numiter)


                if self.time % self.resample_window == 0:
                    self.window_counter += 1

                    # Store the model states before and after resampling
                    if self.do_save:
                        self.save(before=True)

                    self.reweight()

                    self.resample()

                    # Store the model states before and after resampling
                    if self.do_save:
                        self.save(before=False)


                    print("\tFinished window {}, step {} (took {}s)".format(
                        self.window_counter, self.time, round(float(time.time() - window_start_time),2)))
                    window_start_time = time.time()

                elif self.multi_step:
                    assert (False), "Should not get here, if multi_step is true then the condition above should always run"


            else:
                print("\tNo more active agents. Finishing particle step")

            if self.do_ani:
                    self.ani()


        if self.plot_save:
            self.p_save()

        # Return the errors and variences before and after sampling
        # XXXX HERE
        # Useful for debugging in console:
        #for i, a in enumerate(zip([x[1] for x in zip(self.before_resample, self.mean_errors) if x[0] == True],
        #                          [x[1] for x in zip(self.before_resample, self.mean_errors) if x[0] == False])):
        #    print("{} - before: {}, after: {}".format(i, a[0], a[1]))
        
        if not self.mean_errors == []:
            
            # Return two tuples, one with the about the error before reweighting, one after

            # Work out which array indices point to results before and after reweighting
            before_indices = [i for i, x in enumerate(self.before_resample) if x ]
            after_indices  = [i for i, x in enumerate(self.before_resample) if not x ]

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
        else:
            return 
       
    
    def predict(self, numiter=1):
        '''
        Predict
        
        DESCRIPTION
        Increment time. Step the base model. Set self as a constant
        in step_particles and then use a multiprocessing method to step 
        particle models, set the particle states as the agent 
        locations with some added noise, and reassign the
        locations of the particle agents using the new particle
        states. We extract the models and states from the stepped
        particles variable.

        :param numiter: The number of iterations to step (usually either 1, or the  resample window
        '''
        for i in range(numiter):
            self.base_model.step()

        stepped_particles = pool.starmap(ParticleFilter.step_particle, list(zip( \
            range(self.number_of_particles), # Particle numbers (in integer)
            [ m for m in self.models],  # Associated Models (a Model object)
            [numiter]*self.number_of_particles, # Number of iterations to step each particle (an integer)
            [self.particle_std]*self.number_of_particles, # Particle std (for adding noise) (a float)
            [ s.shape for s in self.states], #Shape (for adding noise) (a tuple)
        )))

        self.models = [stepped_particles[i][0] for i in range(len(stepped_particles))]
        self.states = np.array([stepped_particles[i][1] for i in range(len(stepped_particles))])
        
        return
    
    def reweight(self):
        '''
        Reweight
        
        DESCRIPTION
        Add noise to the base model state to get a measured state. Calculate 
        the distance between the particle states and the measured base model 
        state and then calculate the new particle weights as 1/distance. 
        Add a small term to avoid dividing by 0. Normalise the weights.
        '''
        measured_state = (self.base_model.agents2state() 
                          + np.random.normal(0, self.model_std**2, size=self.states.shape))
        distance = np.linalg.norm(self.states - measured_state, axis=1)
        self.weights = 1 / (distance + 1e-9)**2
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
        
        #self.unique_particles.append(len(np.unique(self.states,axis=0)))

        # Could use pool.starmap here, but it's quicker to do it in a single process
        self.models = list(itertools.starmap(ParticleFilter.assign_agents,list(zip(
            range(self.number_of_particles),  # Particle numbers (in integer)
            [s for s in self.states],  # States
            [m for m in self.models]   # Associated Models (a Model object)
        ))))
        return


    def save(self, before:bool):
        '''
        Save
        
        DESCRIPTION
        Calculate number of active agents, mean, and variance 
        of particles and calculate mean error between the mean 
        and the true base model state.

        :param before: whether this is being called before or after resampling as this will have a big impact on
        what the errors mean (if they're after resampling then they should be low, before and they'll be high)
        '''
        self.active_agents.append(sum([agent.active == 1 for agent in self.base_model.agents]))
        
        active_states = [agent.active == 1 for agent in self.base_model.agents for _ in range(2)]
        
        if any(active_states):
            # Mean and variance state of all particles, weighted by their distance to the observation
            mean = np.average(self.states[:,active_states], weights=self.weights, axis=0)
            unweighted_mean = np.average(self.states[:,active_states], axis=0)
            variance = np.average((self.states[:,active_states] - mean)**2, weights=self.weights, axis=0)
            
            self.mean_states.append(mean)
            self.variances.append(np.average(variance))
            self.before_resample.append(before) # Whether this save reflects the errors before or after resampling

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
        
        print('Max mean error = ',max(self.mean_errors))
        print('Average mean error = ',np.average(self.mean_errors))
        print('Max mean variance = ',max(self.variances[2:]))
        print('Average mean variance = ',np.average(self.variances[2:]))

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
        if any([agent.active == 1 for agent in self.base_model.agents]):
    
            plt.figure(1)
            plt.clf()
            
            markersizes = self.weights
            if np.std(markersizes) != 0:
                markersizes *= 4 / np.std(markersizes)   # revar
            markersizes += 8 - np.mean(markersizes)  # remean

            particle = -1
            for model in self.models:
                particle += 1
                markersize = np.clip(markersizes[particle], .5, 8)
                for agent in model.agents[:self.agents_to_visualise]:
                    if agent.active == 1:
                        unique_id = agent.unique_id
                        if self.base_model.agents[unique_id].active == 1:     
                            locs = np.array([self.base_model.agents[unique_id].location, agent.location]).T
                            plt.plot(*locs, '-k', alpha=.5, linewidth=.5)
                            plt.plot(*agent.location, 'or', alpha=.3, markersize=markersize)
            
            for agent in self.base_model.agents:                
                if agent.active == 1:
                    plt.plot(*agent.location, 'sk',markersize = 4)
            
            plt.axis(np.ravel(self.base_model.boundaries, 'F'))
            plt.pause(1 / 4)


def single_run_particle_numbers():

    filter_params = {
        'number_of_particles': param_list[int(sys.argv[1])-1][0], # particles read from ARC task array variable
        'number_of_runs': 10, # Number of times to run each particle filter configuration
        'resample_window': 100,
        'multi_step' : True, # Whether to predict() repeatedly until the sampling window is reached
        'particle_std': 2.0, # was 2 or 10
        'model_std': 1.0, # was 2 or 10
        'agents_to_visualise': 10,
        'do_save': True,
        'plot_save': False,
        'do_ani': False,
        
    }
    
    # Open a file to write the results to
    outfile = "results/pf_particles_{}_agents_{}_noise_{}-{}.csv".format(
        str(int(filter_params['number_of_particles'])),
        str(int(model_params['pop_total'])),
        str(filter_params['particle_std']),
        str(int(time.time()))
    )
    with open(outfile, 'w') as f:
        # Write the parameters first
        f.write("PF params: "+str(filter_params)+"\n")
        f.write("Model params: "+str(model_params)+"\n")
        # Now write the csv headers
        f.write("Min_Mean_errors,Max_Mean_errors,Average_mean_errors,Min_Absolute_errors,Max_Absolute_errors,Average_Absolute_errors,Min_variances,Max_variances,Average_variances,Before_resample?\n")
   
    print("Running filter with {} particles and {} runs (on {} cores) with {} agents. Saving files to: {}".format(
        filter_params['number_of_particles'], filter_params['number_of_runs'] , numcores, model_params["pop_total"], outfile), flush=True)
    print("PF params: "+str(filter_params)+"\n")
    print("Model params: "+str(model_params)+"\n")    
    
    for i in range(filter_params['number_of_runs']):
    
        # Run the particle filter
        
        start_time = time.time()#Time how long the whole run take
        pf = ParticleFilter(Model, model_params, filter_params)
        result = pf.step()
    
        # Write the results of this run
        with open(outfile, 'a') as f:
            # Two sets of errors are created, those before resampling, and those after. Results is a list with two tuples.
            # First tuple has eerrors before resampling, second has errors afterwards.
            for before in [0,1]:
                f.write(str(result[before])[1:-1].replace(" ","")+","+str(before)+"\n") # (slice to get rid of the brackets aruond the tuple)

        print("Run: {}, particles: {}, agents: {}, took: {}(s), result: {}".format(
            i, filter_params['number_of_particles'], model_params['pop_total'], round(time.time()-start_time), result), flush=True) 
    
    print("Finished single run")


if __name__ == '__main__':
    __spec__ = None

    # Pool object needed for multiprocessing
    numcores = multiprocessing.cpu_count()
    #numcores = 5
    pool = multiprocessing.Pool(processes=numcores)
    
    # Lists of particle and agent values to run (old experiments by Kevin)
    #num_par = [1]+list(range(10,1010,10))
    #num_age = [1]+list(range(10,310,10))

    # New ones by nick: (requires 1540 experiments)
    #num_par = list(range(1,49,1))  + list(range(50,501,50)) + list(range(600,2001,100)) + list(range(2500,4001,500))
    #num_age = list(range(1,21,1))
    # New ones by nick: (requires 133 experiments)
    num_par = list([1] + list(range(10,50,10))  + list(range(100,501,100)) + list(range(1000,2001,500)) + list(range(3000,10001,1500)) + [10000] )
    num_age = list(range(1,21,3))

    # List of all particle-agent combinations. ARC task
    # array variable loops through this list
    param_list = [(x,y) for x in num_par for y in num_age]
    
    # Use below to update param_list if some runs abort
    # If used, need to update ARC task array variable
    # 
    # aborted = [2294, 2325, 2356, 2387, 2386, 2417, 2418, 2448, 2449, 2479, 2480, 2478, 2509, 2510, 2511, 2540, 2541, 2542]
    # param_list = [param_list[x-1] for x in aborted]

    model_params = {
        'width': 200,
        'height': 100,
        'pop_total': param_list[int(sys.argv[1])-1][1], # agents read from ARC task array variable
        'entrances': 3,
        'entrance_space': 2,
        'entrance_speed': .1,
        'exits': 2,
        'exit_space': 1,
        'speed_min': .1,
        'speed_desire_mean': 1,
        'speed_desire_std': 1,
        'separation': 2,
        'batch_iterations': 4000, # Only relevant in batch() mode
        'do_save': True, # Saves output data (only relevant in batch() mode)
        'do_ani': False, # Animates the model (only relevant in batch() mode)
        }
    #Model(model_params).batch() # Runs the model as normal (one run)

    # Run the particle filter
    single_run_particle_numbers()
