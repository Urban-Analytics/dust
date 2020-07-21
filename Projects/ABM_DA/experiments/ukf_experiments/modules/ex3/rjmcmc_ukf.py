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
#exit gate probability distribution functions
########

import os
import sys
sys.path.append("../../../../stationsim")
from ukf2 import ukf, ukf_ss
import random
import numpy as np
import multiprocessing
sys.path.append("..")
from sensors import generate_Camera_Rect
import stationsim_densities as sd

class rjmcmc_ukf(ukf_ss):
    
    def __init__(self, model_params, ukf_params, base_model, pool):
    
        for key in model_params.keys():
            setattr(self, key, model_params[key])
        for key in ukf_params.keys():
            setattr(self, key, ukf_params[key])
        self.base_model = base_model
        self.get_gates_dict, self.set_gates_dict = self.gates_dict(self.base_model)
        
        self.boundary = generate_Camera_Rect(np.array([0, 0]), 
                                             np.array([0, self.height]),
                                             np.array([self.width, self.height]), 
                                             np.array([self.width, 0]))
        buffer = base_model.gates_space
        exit_gates =  base_model.gates_locations[-2:]
        self.exit_polys = sd.exit_points_to_polys(exit_gates, self.boundary, buffer)
    

    def agent_probabilities(self, base_model, theta, plot = False):
        """possibly only sample from more contentiuous agents for speed
        would fit into draw_new_gates rather than sample agents uniformly

        Returns
        -------
        gate_probabilities : array_like
            

        """
        start_position = np.hstack([agent.loc_start for agent in base_model.agents])
        position = base_model.get_state(sensor = "location")
        
        angles = sd.start_gate_heading(start_position, position)
        polys = sd.vision_polygon(position, angles, theta, 
                                       self.boundary)
        cut_bounds = sd.cut_boundaries(polys, self.boundary)
        gate_probabilities = sd.exit_gate_probabilities(self.exit_polys, cut_bounds)
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
        n_agents, n_gates = gate_probabilities.shape
        if n_agents < n_replace:
            n_replace = n_agents
        sample = random.sample(range(n_agents), n_replace)
        
        for item in sample:
            weights = gate_probabilities[item, :]
            choice = random.choices(range(n_gates),
                                     weights = weights, k = 1)
            current_gates[item] = choice[0]
            
        return current_gates
    
    ########
    #acceptance probability
    ########
    
    def transition_probability(self, gates_distribution, old_gates, new_gates):
        """ probability of moving to one combination of gates from another.
        
        if proposal gates are m = m_1,..., m_i we times together the probabilities
        gates_distribution[i, m_i] for i in n_agents.

        Parameters
        ----------
        gates_distribution : array_like
            array of (n_agents x n_gates) shape. element i,j is the probability 
            of the ith agent moving to the jth gate.
        old_gates, new_gates: array_like
            `old_gates` current agent exit gates and `new_gates` proposed exit
            gates.
        Returns
        -------
        probability : float
            probability of moving from 

        """
    
    def acceptance_probability(model1, model2, obs):
        """ given observations and two potential models, calculate their relative strength
        
        
        Parameters
        ------
        model1, model2 : cls
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
    
        prob = 1
        #times in probabilities
        return prob
    
    
    ######
    #general functions for interacting with stationsim
    ######
    
    @staticmethod
    def gates_dict(base_model):
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
        gates_locations = base_model.gates_locations[-2:]
        get_gates_dict = {}
        set_gates_dict = {}
        for i in range(gates_locations.shape[0]):
            gate_location = gates_locations[i,:]
            key = i
            get_gates_dict[str(gate_location)] = key
            set_gates_dict[key] = gate_location
        return get_gates_dict, set_gates_dict
    
    @staticmethod
    def get_gates(self, base_model, get_gates_dict):
        """get model exit gate combination from stationsim model
        
        Parameters
        ------
        stationsim : cls
            some `stationsim` class we wish to extract the gates of
        Returns
        -------
        gates : int
            Intinger incidating which of the exit `gates` an agent is heading 
            to.
        """
        #gate centroids from model
        gates = [agent.loc_desire for agent in base_model.agents]
        #convert centroid into intigers using gates_dict
        for i, desire in enumerate(gates):
            gates[i] = get_gates_dict[str(desire)]
        return gates
    
    @staticmethod
    def set_gates(self, base_model, new_gates, set_gates_dict):
        """ assign stationsim a certain combination of gates
        
        Parameters
        ------
        
        gates : list
            slist of agents exit `gates`
            
        Returns 
        ------
        
        stationsim : cls
            some `stationsim` class we wish to extract the gates of
        """

        for i, gate in enumerate(new_gates):
            new_gate = set_gates_dict[gate]
            base_model.agents[i].loc_desire = new_gate
        
        return base_model
    
    ########
    #main step function
    ########
    
    def step():
        """main step function for rjmcmc
        
        - given current exit gates process stationsim one step through ukf
        - at the same time draw new exit gates and process that one step through
            ukf too
        - calculate acceptance probablility
        - accept or reject new model accordingly
        - update as necessary
        Returns
        -------
        None.
    
        """
        #calculate exit gate pdf and draw new gates

        #copy old model
        
        #step both models
        
        #calculate alpha
        
        #choose new model
        
if __name__ == "__main__":
#    rjmcmc_ukf = rjmcmc_ukf(base_model)
    pass
