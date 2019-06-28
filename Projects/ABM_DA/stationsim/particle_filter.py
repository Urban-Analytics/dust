from filter import Filter
from stationsim import Model

import os
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from copy import deepcopy
import multiprocessing
import time
import warnings
import sys
import itertools


class ParticleFilter(Filter): 
    '''
    A particle filter to model the dynamics of the
    state of the model as it develops in time.
    
    TODO: refactor to properly inherit from Filter.
    '''

    def __init__(self, ModelConstructor, model_params, filter_params):
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
        for key, value in filter_params.items():
            setattr(self, key, value)
        self.time = 0
        self.number_of_iterations = model_params['batch_iterations']
        self.base_model = ModelConstructor(**model_params) # (Model does not need a unique id)
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
            if any([agent.status != 2 for agent in self.base_model.agents]):
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
        f
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
        self.active_agents.append(sum([agent.status == 1 for agent in self.base_model.agents]))
        
        active_states = [agent.status == 1 for agent in self.base_model.agents for _ in range(2)]
        
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


def single_run_particle_numbers(num_particles, model_params):

    filter_params = {
        'number_of_particles': num_particles, # particles read from ARC task array variable
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
    # The results directory should be in the parent directory of this script
    results_dir = os.path.join( sys.path[0], "..", "results") # sys.path[0] gets the location of this file
    if not os.path.exists(results_dir):
        raise FileNotFoundError("Results directory ('{}') not found. Are you ".format(results_dir))
    print("Writing results to: {}".format(results_dir))  
    
    outfile = (results_dir + "/pf_particles_{}_agents_{}_noise_{}-{}.csv").format(
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

    _model_params = {
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

    # Run the particle filter, getting the experiment number from the task array
    single_run_particle_numbers(num_particles=param_list[int(sys.argv[1])-1][0], model_params=_model_params)
