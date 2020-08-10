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

#imports from stationsim folder
sys.path.append("../../../../stationsim")
from ukf2 import ukf_ss

#imports from modules
sys.path.append("..")
from sensors import generate_Camera_Rect
import stationsim_densities as sd


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
            DESCRIPTION.
        pool : multiprocessing.pool
            `pool` for multiprocessing used in map/starmap.

        Returns
        -------
        None.

        """
        self.ukf_params = ukf_params
        self.model_params = model_params
        for key in model_params.keys():
            setattr(self, key, model_params[key])
        for key in ukf_params.keys():
            setattr(self, key, ukf_params[key])
            
        self.base_model = base_model
        self.get_gates = get_gates
        self.set_gates = set_gates
        
        #self.base_models = [self.base_model] * ((4 * self.pop_total) + 1)
        #self.fx_kwargs = [self.fx_kwargs] * ((4 * self.pop_total) + 1)

        
        "Build exit gate and boundary polygons."
        "Assign each exit gate an integer."
        self.get_gates_dict, self.set_gates_dict = self.gates_dict(self, self.base_model)

        self.boundary = generate_Camera_Rect(np.array([0, 0]), 
                                             np.array([0, self.height]),
                                             np.array([self.width, self.height]), 
                                             np.array([self.width, 0]))
        buffer = self.base_model.gates_space
        self.exit_polys = sd.exit_points_to_polys(self.exit_gates, self.boundary, buffer)

        self.true_gate = self.get_gates(base_model, self.get_gates_dict)
        self.true_gates = []
        self.estimated_gates = []
        self.model_choices = []
        self.alphas = []
        self.time1 = datetime.datetime.now()
        
        self.start_position = self.base_model.get_state("location")
        

        
        
    def agent_probabilities(self, base_model, theta, get_gate_dict,  plot = False):
        """possibly only sample from more contentiuous agents for speed
        would fit into draw_new_gates rather than sample agents uniformly

        Parameters
        --------
        
        Returns
        -------
        gate_probabilities : array_like
            

        """

        position = base_model.get_state(sensor = "location")
        
        angles = sd.start_gate_heading(self.start_position, position)
        polys = sd.vision_polygon(position, angles, theta, 
                                       self.boundary)
        cut_bounds = sd.cut_boundaries(polys, self.boundary)
        gate_probabilities = sd.exit_gate_probabilities(self.exit_polys, cut_bounds)
        for i in range(gate_probabilities.shape[0]):
            #if np.sum(gate_probabilities[i, :]) == 0:
            if np.sum(gate_probabilities[i, :] == 0) or base_model.agents[i].status != 1:
                try:
                    current_gate = base_model.agents[i].gate_out
                except:
                    current_gate = self.get_gates_dict[str(base_model.agents[i].loc_desire)]

                gate_probabilities[i, :] = 0     
                gate_probabilities[i, current_gate] = 1

        if plot:
            sd.plot_vision(cut_bounds, self.exit_polys, polys, self.boundary)
        return gate_probabilities
            
        
    def draw_new_gates(self, gate_probabilities, current_gates, n_replace):
        """draw some new exit gates for a candidate model
        

        Parameters
        ----------
        agate_probabilities : array_like
            `gate_probabilities` given an agent which gate to choose
        `current_gates` : list
            `current_gates` what are the current exit gates we are 
            moving from
        n_replace : int
            `n_replace` how many agents gates to reshuffle

        Returns
        -------
        None.

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
    
    def transition_probability(self, gates_distribution, new_gates):
        """ probability of moving to one combination of gates from another.
        
        if proposal gates are m = m_1,..., m_i we times together the probabilities
        gates_distribution[i, m_i] for i in n_agents.

        Parameters
        ----------
        gates_distribution : array_like
            array of (n_agents x n_gates) shape. element i,j is the probability 
            of the ith agent moving to the jth gate.
        new_gates: array_like
            `new_gates` proposed exit gates.
        Returns
        -------
        probability : float
            probability of moving from 

        """
        prob = 1
        for i, gate in enumerate(new_gates):
            prob *= gates_distribution[i, gate]
        return prob
    
    def ratio_of_mv_normals(self, ukf1, ukf2, obs):
        """take the ratio of two mv normal densities too small for python
        
        often we see densities <1e-16 which python cant handle
        as we have a ratio of these densities we can cancel out elements
        of each density making it calculable in python.
        Returns
        -------
        None.

        """
        x1 = ukf1.x
        p1 = ukf1.p
        x2 = ukf2.x
        p2 = ukf2.p
        obs1 = np.matmul(ukf1.k, obs)
        obs2 = np.matmul(ukf2.k, obs)
        
        prob = 1.
        prob *= (np.linalg.det(p1)/np.linalg.det(p2))**(-1/2)
        distance = -0.5*(mahalanobis(obs2, x2, p2)**2 - mahalanobis(obs1, x1, p1)**2)
        if np.exp(distance) == np.inf:
            prob *= 0.
        else:
            prob *= np.exp(distance)
        return prob
            
    def acceptance_probability(self, ukf1, ukf2, obs):
        """ given observations and two potential models, calculate their relative strength
        
        
        Parameters
        ------
        ukf1, ukf2 : cls
            `model1` and `model2` are the current and candidate models 
            with different exit gates.
        obs : some observations to determine likelihood of each model
        
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
        prob *= self.ratio_of_mv_normals(ukf1, ukf2, obs)
        
        
        #transition probabilities from old to new model and vice versa
        new_gates = self.get_gates(ukf2.base_model, self.get_gates_dict)
        old_new_transition = self.transition_probability(self.gate_probabilities,
                                                         new_gates)
        old_gates = self.get_gates(ukf1.base_model, self.get_gates_dict)
        new_old_transition = self.transition_probability(self.gate_probabilities,
                                                         old_gates)
        prob *= new_old_transition
        prob /= old_new_transition

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
            `Final `choice` bool. 
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
    
    ######
    #general functions for interacting with stationsim
    ######
    
    @staticmethod
    def gates_dict(self, base_model):
        """assign each exit gate location an integer for a given model
        
        Parameters
        ----------
        base_model : cls
            some model `base_model` e.g. stationsim

        Returns
        -------
        gates_dict : dict
            `gates_dict` dictionary with intiger keys for each gate and a 
            2x1 numpy array value for each key giving the location.

        """
        
        gates_locations = self.exit_gates
        get_gates_dict = {}
        set_gates_dict = {}
        for i in range(gates_locations.shape[0]):
            gate_location = gates_locations[i,:]
            key = i
            get_gates_dict[str(gate_location)] = key
            set_gates_dict[key] = gate_location
            
        return get_gates_dict, set_gates_dict
    
    ########
    #main step function
    ########
    
    def rj_clone(self, ukf_1):
        """build two ukfs to run in tandem
        
        - given current exit gates process stationsim one step through ukf
        - at the same time draw new exit gates and process that one step through
            ukf too
        - calculate acceptance probablility
        - accept or reject new model accordingly
        - update as necessary
        
        Parameters
        --------
        ukf_1 : cls
         `ukf_1` current ukf class to clone and assign new gates
        
        Returns
        -------
        ukf_2 : cls
            `ukf_2` cloned ukf with newly drawn gates
    
        """
        
                #if self.batch:
        #    if self.base_model.step_id == len(self.batch_truths):
        #        break
        #        print("ran out of truths. maybe batch model ended too early.")
        base_model = ukf_1.base_model
        #calculate exit gate pdf and draw new gates for candidate model
        self.gate_probabilities = self.agent_probabilities(base_model, self.vision_angle, self.get_gates_dict)
        #copy original model and give it new gates
        ukf_2 = deepcopy(ukf_1)
        self.current_gates = self.get_gates(ukf_1.base_models[0], self.get_gates_dict)
        
        self.new_gates = self.draw_new_gates(self.gate_probabilities, 
                                             self.current_gates, self.n_jumps)
        
        for i, model in enumerate(ukf_2.base_models):
            ukf_2.base_models[i] = self.set_gates(model, 
                                                  self.new_gates,
                                                  self.set_gates_dict)
        #step both models forwards to get posterior distributions for mu/sigma
        return ukf_2
                
    def rj_choose(self, ukf_1, ukf_2, step):
        """choose which of the two ukfs to keep
        

        Returns
        -------
        new_ukf : TYPE
            DESCRIPTION.

        """
        obs = ukf_1.base_model.get_state("location")
        #calculate alpha
        alpha = self.acceptance_probability(ukf_1, ukf_2, obs)
        self.alphas.append(alpha)
        #choose new model
        choice = self.choose_model(alpha)
        if choice:
            self.current_gates = self.new_gates
            model_choice = 1
            new_ukf =  ukf_2

        else:
            model_choice = 0
            new_ukf =  ukf_1
            
        self.estimated_gates.append(self.current_gates)
        self.model_choices.append(model_choice)
        
        return new_ukf
    
    def main(self):
        """main function for rjmcmc_ukf
        

        Returns
        -------
        None.

        """
        
        logging.info("rjmcmc_ukf start")
        no_gates_model = deepcopy(self.base_model)
        no_gates_model = self.set_gates(no_gates_model, 
                                          [0]*(self.pop_total),
                                          self.set_gates_dict)
        no_gates_models = []
        for i in range(((4*self.pop_total) + 1)):
            no_gates_models.append(deepcopy(no_gates_model))
        self.estimated_gates.append([0]*(self.pop_total))
        
        self.ukf_1 = ukf_ss(self.model_params, self.ukf_params, self.base_model,
                            no_gates_models)
        self.ukf_2 = self.rj_clone(self.ukf_1)
        
        for step in range(self.step_limit):    
            #one step burn in so we have kalman gains for ukf
            self.ukf_1.step(step)
            self.ukf_2.step(step)
            if step % self.jump_rate == 0 and step >= self.sample_rate:
                self.ukf_1 = self.rj_choose(self.ukf_1, self.ukf_2, step)
                self.ukf_2 = self.rj_clone(self.ukf_1)

            self.true_gates.append(self.get_gates(self.ukf_1.base_model,
                                                  self.get_gates_dict))
            if step%100 == 0 :
                logging.info(f"Iterations: {step}")
                
            finished = self.ukf_1.base_model.pop_finished == self.pop_total
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
