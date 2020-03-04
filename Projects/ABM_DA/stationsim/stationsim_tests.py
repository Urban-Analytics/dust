#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 11:01:10 2020

@author: medrclaa

Skeleton file for stationsim ABM tests. 
This file is currently just the types of testing needed 
and potential ideas for said tests.
This is just a placeholder of ideas for now. Feel free to delete.

This isnt particulaly straightforward as unit testing has a vaguer meaning
in ABM literature.

Specifically, unit tests for ABMs is the testing of individual units (agents) 
and their associated deterministic mechanisms (functions). 
AKA white box testing/ micro-level testing.

Similarly, integration tests in which we test the interaction between multiple 
functions also have vaguer definitions in ABM literature.

We have two types of integration test known as meso/macro level testings
(grey/black box).
 
Meso testing typically involves testing groups of agents with the main aim 
of testing the interactions between agents.

These are often difficult to properly test due to the stochastic nature 
of ABMs and this is often done through MCMC techniques or heavy seeding.

Black box testing usually tests the entire environment, such as a full stationsimrun 
and are also highly stochastic. (See RK_Validation for an attempt at this.)
"""

class Test_Micros():

    
    """
    Testing the individual usually deterministic mechanisms as with standard unit
    testing. These may include:
    • Testing building blocks of agents like behaviors, knowledge base and
    so forth and their integration inside agents. 
    • Testing building blocks of simulated environments like non-agent entities, 
    services and so forth and their integration inside simulated environments.
    • Testing the outputs of agents during their lifetime. An
    output can be a log entry, a message to another agent or
    to the simulated environment.
    • Testing if an agent achieves something (reaching a state
    or adapting something) in a considerable amount of time (or before and/or 
    after the occurrence of some specific events) with different
    initial conditions.
    • Testing the interactions between basic elements, commu- nication 
    protocols and semantics.
    • Testing the quality properties of agents, such as their workload 
    (number of behaviors scheduled at a specific time).

    """
    pass

class Test_Mesos():
    
    
    """
    Meso testing typically involves testing groups of agents with the main aim 
    of testing the interactions between agents.
    
    • There is some simple testing of communication protocols of the elements 
    from their perspective in micro-level. However, communication protocols 
    are more adequately tested during the meso-level testing when each element 
    is finally connected to a full and working implementation of those 
    communication protocols.
    • Testing the organization of the agents (how they are situated in a 
    simulation environment or who is interacting with who) during their lifetime. 
    In this sense, the well- known K-means algorithm can be 
    used in order to discover and assess interacting groupings of model elements.
    • Testing whether a group of basic elements exhibits the same long-term 
    behavior (which could be emergent or not) with different initial conditions.
    • Testing whether a group of basic elements is capable of producing some 
    known output data for a given set of input data.
    • Testing the timing requirements of the meso-level behaviours of a
    group of basic elements.
    • Testing the workload for the system as a whole (number of agents, 
    number of behaviors scheduled, number of interactions etc.).
    """
    
    pass

     
class Test_Macros(seed, model, model_params, ukf_params):
    
    """
    testing the ABM/UKF as an entire entity (environment).
    
    The idea is to test "black box" elements made so by the stochasticity of agents.
    The standard idea of a certain input gives a specific output applies but is often not 
    straightforwards due to said randomness. As such, we implement a number of monte carlo
    techniques asserting satistically over multiple ABM runs.
    We do NOT test deterministic elements such as individual agent mechanisms
    
    
    • Testing whether the overall system is capable of produc- ing some known 
    output data for a given set of legal input data.3
    2This algorithm arranges data points into clusters and it locates a 
    centroid in each cluster This centroid is the point at which the distance 
    from the rest of the points of the clusters is on average minimum.
    3Law (Law, 2007) defines a simulation as a numerical technique that 
    takes input data and creates output data based upon a model of a system.
    • Testing whether the overall system is capable of remain- ing available 
    for a given set of illegal input data.
    • Testing whether the overall system is capable of produc- ing some 
    known output within given time constraints.
    • Testing whether the overall system exhibits the same
    long-term behavior (which could be emergent or not) with
    different initial conditions.4
    • Testing the workload for the system as a whole (number
    of agents, number of behaviors scheduled, number of
    interactions etc.).
    • Testing the significance of the simulated data with respect
    to reference data. This can be done by various data comparison techniques 
    such as cross-correlation analysis, coherence analysis, goodness of fit tests etc.
    • The communication protocols are tested in micro- and meso-level testing 
    levels from individual and group per- spectives. However, having a correct 
    execution of pro- tocols does not imply the overall system is behaving 
    correctly. Hence, an agent can execute protocols and still insist on 
    collaborating with wrong agents. To detect such situations, some 
    post-mortem analysis might be required as suggested by 
    (Serrano et al., 2009). To be able to conduct such an analysis, 
    large amount of data should be collected (and sorted) and
    intelligent data analysis techniques must be performed.
    • Stress testing of the overall system with a load that causes it to 
    allocate its resources in maximum amounts. The objective of this test is 
    to try to break the system by finding the circumstances under which it 
    will crash (Burnstein, 2003).
    • Testing the robustness to parameter alterations of the overall system,
    in order to fully trust the results of the simulation runs.
    """
        

        