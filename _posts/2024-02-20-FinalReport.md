---
layout: post
title: "Final Project Report"
tagline:
category: announce
tags: [abm, data assimilation]
---

# Final Project Report

The dust project has come to an end! It officially finished at the end of December 2023 and I am in the process of compiling the final reports. 

If you would like to learn more about what the project did, the best places to look are on the <a href="{{site.baseurl}}/publications.html">publications</a> and <a href="{{site.baseurl}}/presentations.html">presentations</a> pages. 

Here is a high-level summary of what the project was about:

## Context and Motivation

When developing computer models of human systems, there are many situations where it is important to model the individual entities that drive the system, rather than using aggregate methods that combine individuals into homogeneous groups. In such circumstances, the technique of ‘agent-based modelling’ has been shown to be useful. In an agent-based model, individual, autonomous ‘agents’ are created. These agents can be given rules or objectives that determine how they will behave in a given situation. They can also interact with each other and with their environment. In this manner it is possible to simulate many of the interesting outcomes that emerge as social systems evolve which might be missed using aggregate methods that do not account for individuals, or their interactions, specifically.

There is a drawback with agent-based models though; they cannot incorporate real-time data into their simulations. This means that it is not possible to use agent-based modelling to simulate systems in real time. Agent-based modelling is an ideal tool for modelling systems such as traffic congestion, crowd dynamics, economic behaviour, disease spread, etc., but it is limited in its ability to make short term, real-world predictions or forecasts because it is not able to react to the most up to date data.

## Project Aims 

The aim of this project was to develop new methods that will allow real-time data to be fed in to agent-based models to bring them in to line with reality. It leveraged existing ‘data assimilation’ methods, that have been established in fields such as meteorology, and tested how applicable the methods are when adapted for agent-based modelling. The main difficulty that the project encountered was that, due to the complexity of an agent-based model, very large numbers of individual models were often required to be run simultaneously. This requires extremely large amounts of computing power. Therefore the algorithms that were the most useful were typically those that could conduct data assimilation while running the smallest number of models.

## Results 

The results of the project are largely methodological; it has adapted methods and has shown them to be capable of feeding real-time data into agent-based models at runtime. This has important implications for society because it means that new types of models, such as social ‘digital twins’, could be possible if sufficient computing power and data were available. Immediate future work for the project will be to start to implement the methods in real situations. 

## Main Challenges 

The project largely focussed on methodological development, with some preliminary empirical applications in later stages. It adapted a number of methods that are commonly used in other fields and has tested how well they are able to allow us to conduct data assimilation for use agent-based models. Specifically, we looked at techniques called Particle Filters, Unscented Kalman Filters and Ensemble Kalman Filters. We also developed an entirely new methods based on quantum field theory and created a way to build agent-based models that allows Markov Chain Monte-Carlo sampling to take place (this is a very efficient way to incorporate observational data). 

There were a number of challenges that the project faced. 

Firstly, agent-based models are extremely computationally expensive so can take a long time to run. Most data assimilation methods also require a large number of models to be run simultaneously, which significantly exacerbates the problem. Some of the methods, such as the Kalman Filter variations, required smaller numbers of models to be run simultaneously so these delivered the most useful results. In addition, we also looked at emulator/surrogate methods that should allow the full agent-based model to run more quickly, albeit at a lower resolution (rather than having individual agents in the model, we create aggregate approximations that behave as if they were made of individuals). We have tested Random Forests and Gaussian Process Emulators; both of which hold promise. 

Another challenge is that data assimilation methods are typically designed for models with continuous variables (e.g. air pressure, wind speed, etc.) but agent-based models often have discrete variables (e.g. an agent’s age, or their destination). Therefore we have adapted the standard approaches to allow the data assimilation methods to include categorical methods.

In the later stages of work the project has also begun to apply the new methods to a variety of different types of model to explore their use in different systems. These include crowding in busy public places, the spread of disease, and international policy diffusion. 

## Future work 

Future work will continue to develop these kinds of applications. In particular, the results of the project could make an important contribution in the area of ‘digital twins’ by allowing real-time agent-based models to become part of larger ‘twin’-like simulations.



