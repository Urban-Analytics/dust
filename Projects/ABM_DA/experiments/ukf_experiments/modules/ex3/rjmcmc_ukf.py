#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 16:25:08 2020

@author: medrclaa

This file contains code for implementing the Reversible Jump Markov Chain Monte
Carlo (RJMCMC) on the Unscented Kalman  Filter (UKF). The main aim is to assume 
the exit gate of each agent is unknown, I.E. we dont know where anyone is going, 
and need to estimate the exit gates of each agent. The problem results in an
enormous number of potential models (n_gates-1)**n_agents (e.g. around 282 million
potential models for 10 agents and 8 gates). It is obviously infeasible to test 
all possible models for a solution in a reasonable amount of time. The aim is to
apply RJMCMC, an algorithm designed to scan a list of potential models, to this very
large list of models M to try and narrow down to a small sublist of models in real time.

The rough outline of the ukf algorithm is as follows:

1. initiate some model M_0 and its associated parameters theta_0
for each time step k = 1, 2, ...
2. propose some new model N_k-1 with parameters phi_k-1
3. progress both models iforwards in time to calculate new model parameters 
theta_k phi_k
4. calculate some probability of accepting the new model N based on a posterior
distribution of each set of parameters occuring given some observations z_k
5. given this probability reject or accept the new model accordingly setting the
kept model to the new model M_k.
repeat 2-5 as necessary until the true model ends

This poses a number of significant problems:
    
- How to draw new candidate models. drawing uniformly over all models
is impossible so we need to find a probability distribution of models to draw from.
This may be as simple as finding which direction each agent is heading, isolating
the 1-3 gates in that direction and only choosing from the set of models where each
agent can only use those gates. This will massively reduce the number of models 
and can be done dynamically for agents with more elaborate paths but may still 
provide an infeasible number of models (e.g. 2 gates 10 agents = 1024 models). 

- for a further reduction in models we could also draw models stepwise. 
That is we keep the old model but only allow some small number of agents (say 5) to 
draw new exit gates. This would even further reduce the number of
 potential models (e.g. 2**5 = 32) but is vulnerable to local minima. As such, 
 this is often paired with techniques such as Hamiltonian Monte Carlo or Simlated
 Annealing to increase the chance of finding the true global maximum.

- convergence: If we pretty clearly find the correct exit gate for an agent can we 
   stop it being randomised until further evidence to suggest otherwise.

- calculation of acceptance probability. This may be very nice to do with a Kalman 
Filter since you're essentially calculating these posteriors anyway. Essentially
a ratio of two high dimensional Gaussian densities for our observations given 
posteriors for models M and N respectively.

"""

########
#imports
########

import os
import sys
import logging
import datetime
import random
import numpy as np
import multiprocessing
from scipy.stats import multivariate_normal
from scipy.spatial.distance import mahalanobis
from copy import copy, deepcopy
import pickle

# imports from stationsim folder
# double appends so works on arc
sys.path.append("../../..")
sys.path.append("../../../..")
from stationsim.ukf2 import *

# imports from modules
sys.path.append("..")
sys.path.append("../..")
# double appends so works on arc
from modules.sensors import generate_Camera_Rect
import modules.ex3.stationsim_densities as sd

class rjmcmc_ukf():
    
    def __init__(self, model_params, ukf_params, base_model, get_gates, set_gates):
        """init rjmcmc_ukf class
        
        - load in model and ukf params.
        - make every model/ukf params part of self for cleaner code
        - load in multiprocessing pool and base_model
        - build exit gate polygons and assign them integers
        - build base_model boundary
        
        Parameters
        ----------
        model_params, ukf_params : dict
            stationsim parameters `model_params` and ukf parameters 
            `ukf_params`.
        base_model : cls
            `base_model` ABM of choice.
        get_gates, set_gates : func
            each ABM needs different functions in order to assign `get_gates`
            and overwrite `set_gates` each agents exit gates.

        Returns
        -------
        None.

        """
        # parameters
        self.ukf_params = ukf_params
        self.model_params = model_params
        #load parameters into self for readability 
        for key in model_params.keys():
            setattr(self, key, model_params[key])
        for key in ukf_params.keys():
            setattr(self, key, ukf_params[key])
            
        #load init arguments for model and gates functions into self
        self.base_model = base_model
        self.get_gates = get_gates
        self.set_gates = set_gates
        
        #self.base_models = [self.base_model] * ((4 * self.pop_total) + 1)
        #self.fx_kwargs = [self.fx_kwargs] * ((4 * self.pop_total) + 1)

        if self.jump_rate % self.sample_rate != 0:
            print("Warning! jump rate is not a multiple of the sample rate.")
            print("Jump decisions will be made on unassimilated position predictions.")
            print("This can result in poor estimation of exit gates.")
            
        # Build two dictionaries to translate an intiger exit gate into a
        # 2d gate centroid coordinate and back.
        
        # build ABM boundary polygon in shapely.
        self.boundary = generate_Camera_Rect(np.array([0, 0]), 
                                             np.array([0, self.height]),
                                             np.array([self.width, self.height]), 
                                             np.array([self.width, 0]))
        # build polygons indicating where the exit gates are.
        # these will be semicircles with buffer radius.
        buffer = self.base_model.gates_space
        self.exit_polys = sd.exit_points_to_polys(self.exit_gates, self.boundary, buffer)
        
        #calculate true exit gates and make a number of placeholder lists for data

        self.true_gate = self.get_gates(base_model, self.get_gates_dict)
        
        #various placeholder lists
        self.obs = []  # actual sensor observations
        self.obs_key = []
        self.ukf_histories = []  # ukf predictions
        self.forecasts = []  # pure stationsim predictions
        self.truths = []  # noiseless observations
        self.ps = [] #covariances
        self.status_key = []
        
        #self.model_choices = []
        #self.alphas = []
                
        self.true_gates = self.true_gate
        self.estimated_gates = []

        self.time1 = datetime.datetime.now()
        
        # initial agent postions to initiate ukf and calcualte agent probabilities
        self.start_position = self.base_model.get_state("location")
        
    def agent_probabilities(self, start_position, current_position, base_model,
                            theta, get_gate_dict):
        """calculate probabilities an agent is heading to each exit gate.

        - calculate heading using line between start and current position
        - build a cone of vision about this heading at the agents location
        - calculate the length of each gate within the agents vision
        - normalise these lengths so they sum to 1 to give an empirical distribution
        
        Parameters
        --------
        
        start_position, current_position : array_like
            `start_position` starting location and `current_position` based on
            current observation. 
            
        theta : float
            how wide is half the agents field of vision in radians `theta`. 
            E.g. if we choose theta = pi/4 the agent will have a pi/2 cone of 
            vision seeing pi/4 either side of its heading.
            0 < theta <= pi
        
        get_gate_dict : dict
            `get_gate_dict` dictionary for converting gate locations into 
            ID intigers.
            
        Returns
        -------
        gate_probabilities : array_like
            `gate_probabilites` (n_agent x n_gates) transition probability matrix. 
            Element i in row j gives the chance the agent i is heading to gate j.
        """
        # generate angle agent is heading drawing a line through its start and
        # current observations
        angles = sd.start_gate_heading(self.start_position, current_position)
        # generate an arc of vision at the current position centered on this heading
        # some 2theta radians wide.
        polys = sd.vision_polygon(current_position, angles, theta, 
                                       self.boundary)
        # find the intersection of this vision arc with the stationsim boundary
        cut_bounds = sd.cut_boundaries(polys, self.boundary)
        # find the proportion of each gate in the agents vision and normalise
        # them to 1 for empirical probabilities.
        gate_probabilities = sd.exit_gate_probabilities(self.exit_polys, cut_bounds)
        
        # loop over each agent and assign them a new gate.
        for i in range(gate_probabilities.shape[0]):
            #if np.sum(gate_probabilities[i, :]) == 0:
            # dont redraw agents with 0 visible gates or if theyre not active
            if np.sum(gate_probabilities[i, :]) == 0 or base_model.agents[i].status != 1:
                # get the current gate for the agent and assign it probability 1
                # basically keep the agent doing what it was doing before
                # this is an easy way to avoid errors at the start of the model run
                current_gate = self.current_gates[i]
             
                gate_probabilities[i, :] = 0     
                gate_probabilities[i, current_gate] = 1
                
        return gate_probabilities
            
        
    def draw_new_gates(self, gate_probabilities, current_gates, n_replace):
        """draw some new exit gates for a candidate model
        

        Parameters
        ----------
        gate_probabilities : array_like
            `gate_probabilites` (n_agent x n_gates) transition probability matrix. 
            Element i in row j gives the chance the agent i is heading to gate j.
        `current_gates` : list
            `current_gates` what are the current exit gates we are 
            moving from
        n_replace : int
            `n_replace` how many agents gates to reshuffle

        Returns
        -------
        new_gates : array_like
            `new_gates` some (n_agents x 1) array of intigers indicating 
            new exit gates the candidate model will use.
        """
        
        #sample without replacement which gates to change
        new_gates = copy(current_gates)
        n_agents, n_gates = gate_probabilities.shape
        if n_agents < n_replace:
            n_replace = n_agents
        sample = random.sample(range(n_agents), n_replace)
        
        for item in sample:
            weights = gate_probabilities[item, :]
            choice = random.choices(range(n_gates),
                                     weights = weights, k = 1)
            new_gates[item] = choice[0]
            
        return new_gates
    
    ########
    #acceptance probability
    ########
    
    def transition_probability(self, old_gates, new_gates, gates_distribution):
        """ how likely is a given set of gates to be drawn from the gate probability distribution
        
        E.g. if we have three agents and 2 gates assume this probability distrubiton
        [0.25, 0.75]
        [0.5 , 0.5 ]
        [0.66, 0.33]
        I.E agent 1 is 25% likely to go to gate 1 and 75% to gate 2.
        
        If the new gates are (2, 2, 1) giving agent 1 to gate 2 and so on,
        the transition probability is given as 
        p = 0.75 * 0.5 * 0.66 
        
        Parameters
        ----------
        gates_distribution : array_like
            array of (n_agents x n_gates) shape. element i,j is the probability 
            of the ith agent moving to the jth gate.
        new_gates : array_like
            `new_gates` some (n_agents x 1) array of intigers indicating 
            new exit gates the candidate model will use.
        Returns
        -------
        probability : float
            probability of moving from 

        """
        prob = 1
        
        for i, gate in enumerate(new_gates):
            
            new_gate_prob =  gates_distribution[i, new_gates[i]]
            old_gate_prob = gates_distribution[i, old_gates[i]]
            
            if old_gate_prob == 0:
                prob = np.inf
                break
            else:
                new_over_old_gate_ratio = new_gate_prob/old_gate_prob 
                prob *= new_over_old_gate_ratio
        return prob
    
    def ratio_of_mv_normals(self, ukf_1, ukf_2, obs):
        """take the ratio of two mv normal densities too small for python
        
        often we see densities <1e-16 which python can struggle to handle. 
        Its easier to do this by hand especially as many of the terms in
        the ratio cancel out reducing the chance of numeric errors.
        
        Parameters
        --------
        ukf_1, ukf_2 : cls
            original `ukf1` ukf model and candidate `ukf2` ukf model
        Returns
        -------
        
        ratio : float
            `ratio` of two mv normals. prob > 0
        """
        x1 = ukf_1.x
        p1 = ukf_1.p
        x2 = ukf_2.x
        p2 = ukf_2.p
        obs1 = np.matmul(ukf_1.k, obs)
        obs2 = np.matmul(ukf_2.k, obs)
        
        ratio = 1.
        ratio *= (np.linalg.det(p1)/np.linalg.det(p2))**(-1/2)
        distance = -0.5*(mahalanobis(obs2, x2, p2)**2 - mahalanobis(obs1, x1, p1)**2)
        if np.exp(distance) == np.inf:
            ratio *= 0.
        else:
            ratio *= np.exp(distance)
        return ratio
            
    def acceptance_probability(self, ukf_1, ukf_2, obs):
        """ given observations and two potential models, calculate their relative strength
        
        Parameters
        ------
        ukf_1, ukf_2 : cls
            original `ukf1` ukf model and candidate `ukf2` ukf model
        obs : array_like
            `obs` some observations to determine likelihood of each model
        
        Returns
        -------
        prob : float
            chance of accepting new model over current model. This is a float
            between 0 and 1. If it is say 0.8 it gives us an 80% chance of 
            changing models when we draw a number between 0 and 1 from a 
            uniform distribution.
        """
        #placeholder 
        prob = 1
        #posterior Gaussian densities of observations.
        #I.E how likely are the observations to be from the given posterior. 
        #larger density implies more likely
        prob *= self.ratio_of_mv_normals(ukf_1, ukf_2, obs)
        
        
        #transition probabilities from old to new model and vice versa
        new_gates = self.get_gates(ukf_2.base_model, self.get_gates_dict)
        #old_new_transition = self.transition_probability(self.gate_probabilities,
        #                                                 new_gates)
        old_gates = self.get_gates(ukf_1.base_model, self.get_gates_dict)
        #new_old_transition = self.transition_probability(self.gate_probabilities,
        #                                                 old_gates)
        
        prob *= self.transition_probability(old_gates, new_gates, 
                                            self.gate_probabilities)
        #prob *= new_old_transition
        #prob /= old_new_transition

        #if old_new_transition != 0:
        #    print("Warning. 0 chance of moving from current to new gates.")
        #    prob = np.nan
        #else:
        #    prob /= old_new_transition

        #if prob == np.nan:
        #    print("Warning. nan probability. This is usually just the initiation step.")
        prob = np.min([prob, 1.])
        #Jacobian 1 for now. could change if we vary stationsim dimensions.
        #prob *= Jacobian
        return prob
    
    def choose_model(self, alpha):
        """given acceptance probability alpha do we choose the old or new model

        Parameters
        ----------
        alpha : float
            probability of accepting the new model `alpha`. Must be 
            0<=alpha<=1 by definition.

        Returns
        -------
        choice : bool
            Final `choice` bool. 
            False if we KEEP the old model 
            True if we CHANGE to the new model

        """
        if alpha  == 1.:
            choice = True
        elif alpha == 0 or np.nan:
            choice = False
        else:
            choice = np.random.binomial(1, alpha)
            if choice ==0:
                choice = False
            else:
                choice = True
        return choice
    
    
    ########
    #main step function
    ########
    
    def rj_assign(self):
        """assign new exit gates to candidate model
        
        - given current exit gates process stationsim one step through ukf
        - at the same time draw new exit gates and process that one step through
            ukf too
        - calculate acceptance probablility
        - accept or reject new model accordingly
        - update as necessary
        
        Parameters
        --------
        ukf_1, ukf_2: cls
         `ukf_1` current ukf and
         `ukf_2` candidate ukf
         
        Returns
        -------
        None
    
        """
        #calculate exit gate pdf and draw new gates for candidate model
        #copy original model and give it new gates      

        self.ukf_2 =  pickle.loads(pickle.dumps(self.ukf_1, -1))
        #self.ukf_2 = deepcopy(self.ukf_1) #generally slower
        #for i, model in enumerate(self.ukf_2.base_models):
        #    self.ukf_2.base_models[i] = self.set_gates(model, 
        #                                          self.new_gates,
        #                                          self.set_gates_dict)
        self.ukf_2.base_model = self.set_gates(self.ukf_2.base_model, 
                                                 self.new_gates,
                                                  self.set_gates_dict)
        
    def rj_choose(self, step):
        """choose which of the two ukfs to keep
        

        Parameters
        ---------
        step : int
            current `step` of model run
        
        Returns
        ------
        """
        obs = self.base_model.get_state("location")
        #calculate alpha
        alpha = self.acceptance_probability(self.ukf_1, self.ukf_2, obs)
        #print(alpha)
        #self.alphas.append(alpha)
        #choose new model
        choice = self.choose_model(alpha)
        if choice:
            # accept new model. swap the models over
            # ukf_2 becomes ukf_1. ukf_1 will be discarded and assigned new gates.
            self.current_gates = self.new_gates
            model_choice = 1
            #model swap. done this way to avoiding using deepcopy (its slow as hell).
            self.ukf_1 = self.ukf_2
            self.ukf_2 = None
        else:
            #if not choice keep current model ukf_1. assign ukf_2 new gates
            model_choice = 0
            self.ukf_2 = None
            
        self.estimated_gates.append(self.current_gates)
        #self.model_choices.append(model_choice)
            
    def main(self):
        """main function for rjmcmc_ukf
        
        - set up base_models with all 0 gates
        - set up current and candidate ukfs
        - run both until a jump
        - keep the better perforkming set of gates
        - repeat until the model run ends.
        
        Returns
        -------
        None.

        """
        
        logging.info("rjmcmc_ukf start")
        no_gates_model = deepcopy(self.base_model)
        no_gates_model = self.set_gates(no_gates_model, 
                                          [0]*(self.pop_total),
                                          self.set_gates_dict)
        #no_gates_models = []
        #for i in range(((4*self.pop_total) + 1)):
        #    no_gates_models.append(deepcopy(no_gates_model))
        
        self.current_gates = [0]*(self.pop_total)
        self.estimated_gates.append(self.current_gates)
        
        self.ukf_1 = ukf_ss(self.model_params, self.ukf_params, no_gates_model)
        self.ukf_2 = deepcopy(self.ukf_1)
        
        for step in range(self.step_limit):    
            #one step burn in so we have kalman gains for ukf
            
            self.base_model.step()
            state = noisy_State(self.base_model, self.noise)
            self.ukf_1.step(step, state)
            self.ukf_2.step(step, state)
            
            if step % self.jump_rate == 0 and step >= self.sample_rate:  
                
                self.obs = self.base_model.get_state(sensor="location").astype(float)
                if self.noise != 0:
                    noise_array = np.ones(self.pop_total*2)
                    noise_array[np.repeat([agent.status != 1 for
                                           agent in self.base_model.agents], 2)] = 0
                    noise_array *= np.random.normal(0, self.noise, self.pop_total*2)
                    self.obs += noise_array
            
                self.gate_probabilities = self.agent_probabilities(self.start_position,
                                                                   self.obs,
                                                                   self.base_model,
                                                                   self.vision_angle, 
                                                                   self.get_gates_dict)
                self.current_gates = self.get_gates(self.ukf_1.base_model,
                                                            self.get_gates_dict)
                self.new_gates = self.draw_new_gates(self.gate_probabilities, 
                                                     self.current_gates, self.n_jumps)
                
                print(self.true_gates)
                print(self.current_gates)                
                self.rj_choose(step)
                self.rj_assign()
                #self.true_gates.append(self.get_gates(self.ukf_1.base_model,
                #                                  self.get_gates_dict))
                
            if step % self.sample_rate == 0 and step > 0:
                #self.forecasts.append(self.ukf_1.forecast)
                self.ukf_histories.append(self.ukf_1.prediction)  # append histories
                #self.obs.append(self.ukf_1.obs)
                self.obs_key.append(self.ukf_1.obs_key)
                #self.ps.append(self.ukf_1.p)
                
            self.truths.append(self.base_model.get_state(sensor="location"))
            self.status_key.append([agent.status for agent in self.base_model.agents])

            if step%100 == 0 :
                logging.info(f"Iterations: {step}")
                
            finished = self.base_model.pop_finished == self.pop_total
            if finished:  # break condition
                logging.info("ukf broken early. all agents finished")
                break
    
        self.time2 = datetime.datetime.now()  # timer
        if not finished:
            logging.info(f"ukf timed out. max iterations {self.step_limit} of stationsim reached")
        time = self.time2-self.time1
        time = f"Time Elapsed: {time}"
        print(time)
        logging.info(time)
        
        #self.pool.close()
        #self.pool.join()
        #self.ukf_1.pool = None
        #self.pool = None

        
if __name__ == "__main__":
    pass
