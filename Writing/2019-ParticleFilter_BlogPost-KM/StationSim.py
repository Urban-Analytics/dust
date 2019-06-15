# -*- coding: utf-8 -*-

"""

Created on Tue Nov 20 15:25:27 2018



@author: medkmin

"""



        # sspmm.py

'''

StationSim (aka Mike's model) converted into python.



Todos:

multiprocessing

profile functions

'''



#%% INIT 

import numpy as np

from scipy.spatial import cKDTree

import matplotlib.pyplot as plt

from copy import deepcopy

from multiprocessing import Pool

import multiprocessing



def error(text='Self created error.'):

    from sys import exit

    print()

    exit(text)

    return





#%% MODEL

    

class Agent:



    def __init__(self, model, unique_id):

        """

        Initialise a new agent.

        

        PARAMETERS

          - model: a pointer to the station sim model that is creating this agent

        

        DESCRIPTION

        Creates a new agent and gives it a randomly chosen entrance, exit, and 

        desired speed. All agents start with active state 0 ('not started').

        Their initial location (** HOW IS LOCATION REPRESENTED?? **) is set

        to the location of the entrance that they are assigned to.

        """

        # Required

        self.unique_id = unique_id

        self.active = 0  # 0 Not Started, 1 Active, 2 Finished

        model.pop_active += 1

        # Location

        self.location = model.loc_entrances[np.random.randint(model.entrances)]

        self.location[1] += model.entrance_space * (np.random.uniform() - .5) # XXXX WHAT DOES THIS DO?

        self.loc_desire = model.loc_exits[np.random.randint(model.exits)]

        # Parameters

        self.time_activate = np.random.exponential(model.entrance_speed) # XXXX WHAT DOES THIS DO?

        self.speed_desire = max(np.random.normal(model.speed_desire_mean, model.speed_desire_std), 2*model.speed_min)

        self.speeds = np.arange(self.speed_desire, model.speed_min, -model.speed_step)# XXXX WHAT DOES THIS DO?

        if model.do_save:

            self.history_loc = []

        return



    def step(self, model):

        """

        Iterate the agent. If they are inactive then it checks to see if they

        should become active. If they are active then then move (see 

        self.move()) and, possibly, leave the model (see exit_query())).

        """

        if self.active == 0:

            self.activate(model)

        elif self.active == 1:

            self.move(model)

            self.exit_query(model)

            self.save(model)

        return



    def activate(self, model):

        """

        Test whether an agent should become active. This happens when the model

        time is greater than the agent's activate time.

        """

        if not self.active and model.time_id > self.time_activate:

            self.active = 1

            self.time_start = model.time_id

            self.time_expected = np.linalg.norm(self.location - self.loc_desire) / self.speed_desire

        return



    def move(self, model):

        """

        Move the agent towards their destination. If the way is clear then the

        agent moves the maximum distance they can given their maximum possible

        speed (self.speed_desire). If not, then they iteratively test smaller

        and smaller distances until they find one that they can travel to

        without causing a colision with another agent.

        """

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

        """

        Detects whether a move to the new_location will cause a colision 

        (either with the model boundary or another agent).

        """

        within_bounds = all(model.boundaries[0] <= new_location) and all(new_location <= model.boundaries[1])

        if not within_bounds:

            collide = True

        elif self.neighbourhood(model, new_location):

            collide = True

        else:

            collide = False

        return collide



    def neighbourhood(self, model, new_location, do_kd_tree=True):

        """

       

        PARMETERS

         - model:        the model that this agent is part of

         - new_location: the proposed new location that the agent will move to


         - do_kd_tree    whther to use a spatial index (kd_tree) (default true)

        """

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

        """

        Determine whether the agent should leave the model and, if so, 

        remove them. Otherwise do nothing.

        """

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

        """

        Create a new model, reading parameters from a dictionary. 

        

        """

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

        if self.pop_finished < self.pop_total and self.step:

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

        """

        Run the model.

        """

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

    

    

    @classmethod

    def run_defaultmodel(cls):

        """

        Run a model with some common parameters. Mostly used for testing.

        """

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

            'do_save': True,

        'do_ani': False,

        }

        # Run the model

        Model(model_params).batch()



#%% PARTICLE FILTER



# Multiprocessing Methods

def initial_state(particle,self):

    """

    Set the state of the particles to the state of the

    base model.

    """

    self.states[particle,:] = self.base_model.agents2state()

    return self.states[particle]

    

def assign_agents(particle,self):

    """

    Assign the state of the particles to the 

    locations of the agents.

    """

    self.models[particle].state2agents(self.states[particle])

    return self.models[particle]



def step_particles(particle,self):

    """

    Step each particle model, assign the locations of the 

    agents to the particle state with some noise, and 

    then use the new particle state to set the location 

    of the agents.

    """

    self.models[particle].step()

    self.states[particle] = (self.models[particle].agents2state()

                           + np.random.normal(0, self.particle_std**2, 

                                                 size=self.states[particle].shape))

    self.models[particle].state2agents(self.states[particle])

    return self.models[particle], self.states[particle]



class ParticleFilter: 

    '''

    A particle filter to model the dynamics of the

    state of the model as it develops in time.

    '''

    

    # check mean error calculation

    # Update to only measure a certain percentage of the base model

    # Rewrite using only pool.map without starmap



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

        if self.do_save:

            self.active_agents = []

            self.means = []

            self.mean_errors = []

            self.variances = []

            self.unique_particles = []

        

        self.states = np.array(pool.starmap(initial_state,list(zip(range(self.number_of_particles),[self]*self.number_of_particles))))

    

    def step(self):

        '''

        Step Particle Filter

        

        DESCRIPTION

        Loop through process. Predict the base model and particles

        forward. If the resample window has been reached, 

        reweight particles based on distance to base model and resample 

        particles choosing particles with higher weights. Then save

        and animate the data. When done, plot save figures.

        '''     

        while (self.time < self.number_of_iterations) & any([agent.active != 2 for agent in self.base_model.agents]):

            self.time += 1

            

            #if any([agent.active != 2 for agent in self.base_model.agents]):

#                print(self.time/self.number_of_iterations)

          #  print(self.number_of_iterations)

            self.predict()

            

            if self.time % self.resample_window == 0:

                self.reweight()

                self.resample()



            if self.do_save:

                self.save()

            if self.do_ani:

                self.ani()

              

        if self.plot_save:

            self.p_save()

            

        

        return max(self.mean_errors), np.average(self.mean_errors), max(self.variances), np.average(self.variances)

    

    def predict(self):

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

        '''    

        self.base_model.step()



        stepped_particles = pool.starmap(step_particles,list(zip(range(self.number_of_particles),[self]*self.number_of_particles)))

            

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

        self.weights = 1 / (distance + 1e-99)**2

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

        

        self.unique_particles.append(len(np.unique(self.states,axis=0)))



        self.models = pool.starmap(assign_agents,list(zip(range(self.number_of_particles),[self]*self.number_of_particles)))



        return



    def save(self):

        '''

        Save

        

        DESCRIPTION

        Calculate number of active agents, mean, and variance 

        of particles and calculate mean error between the mean 

        and the true base model state. Plot active agents,mean 

        error and mean variance. 

        '''    

        self.active_agents.append(sum([agent.active == 1 for agent in self.base_model.agents]))

        

        active_states = [agent.active == 1 for agent in self.base_model.agents for _ in range(2)]

        

        if any(active_states):

            mean = np.average(self.states[:,active_states], weights=self.weights, axis=0)

            variance = np.average((self.states[:,active_states] - mean)**2, weights=self.weights, axis=0)

            

            self.means.append(mean)

            self.variances.append(np.average(variance))



            truth_state = self.base_model.agents2state()

            self.mean_errors.append(np.linalg.norm(mean - truth_state[active_states], axis=0))

        

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

                            plt.plot(*locs, '-k', alpha=.1, linewidth=.3)

                            plt.plot(*agent.location, 'or', alpha=.3, markersize=markersize)

            

            for agent in self.base_model.agents:                

                if agent.active == 1:

                    plt.plot(*agent.location, 'sk',markersize = 4)

            

            plt.axis(np.ravel(self.base_model.boundaries, 'F'))

            plt.pause(1 / 4)





def single_run_particle_numbers():

    particle_num = 20

    runs = 100



    for i in range(runs):



        # Run the particle filter

        filter_params = {

            'number_of_particles': particle_num,

            'resample_window': 100,

            'agents_to_visualise': 2,

            'particle_std': 1,

            'model_std': 1,

            'do_save': True,

            'plot_save':False,

            'do_ani': False,

        }

        pf = ParticleFilter(Model, model_params, filter_params)

        print(i, particle_num, pf.step())



def f(i):

    print(i)

    return i



import cProfile



if __name__ == '__main__':

    __spec__ = None



# Pool object needed for multiprocessing

    pool = Pool(processes=multiprocessing.cpu_count())

    # XXXX the pool object isn't actually used - what is it's purpose?

    # It's used in lines 420, 474, 524



    model_params = {

        'width': 200,

        'height': 100,

        'pop_total':700,

        'entrances': 3,

        'entrance_space': 2,

        'entrance_speed': .1,

        'exits': 2,

        'exit_space': 1,

        'speed_min': .1,

        'speed_desire_mean': 1,

        'speed_desire_std': 1,

        'separation': 2,

        'batch_iterations': 4000,

        'do_save': True,

        'do_ani': True,

        }

    if True:

        # Run the model

#        p = cProfile.Profile()

#        p.run('Model(model_params).batch()')

#        p.print_stats()

        Model(model_params).batch()

    else:

        single_run_particle_numbers()        



        #prof = cProfile.Profile()

        #prof.run('single_run_particle_numbers()')

        #prof.print_stats()





#        p = multiprocessing.Pool()

#        mylist = range(1000)

##        p.map(f, mylist)

#        prof = cProfile.Profile()

#        prof.run('p.map(f, mylist)')

#        prof.print_stats()

        

        

