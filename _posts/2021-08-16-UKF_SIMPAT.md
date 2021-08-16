---
layout: post
title: "New paper: Real-Time Crowd Modelling"
tagline:
category: announce
tags: [abm, uncertainty, data assimilation, conference]
---

<!-- # Quantifying the uncertainty in agent-based models (aka what the \*$&^% is going on in my model and why?) -->

<figure style="float:right; width:30%; padding:10px;" >
<a href="https://www.sciencedirect.com/journal/simulation-modelling-practice-and-theory"><img src="https://ars.els-cdn.com/content/image/1-s2.0-S1569190X21X00056-cov200h.gif" alt="Journal cover picture" /></a>
</figure>

We have just published a new paper in [_Simulation Modelling Practice and Theory_](www.elsevier.com/locate/simpat). The paper presents a new method to allow a crowd simulation to be optimised with real data _while the model is running_. The full details are:


Clay, Robert, J. A. Ward, P. Ternes, Le-Minh Kieu, N. Malleson (2021) Real-time agent-based crowd simulation with the Reversible Jump Unscented Kalman Filter. _Simulation Modelling Practice and Theory_ 113 (102386) DOI: [10.1016/j.simpat.2021.102386](https://doi.org/10.1016/j.simpat.2021.102386)

<blockquote>
Commonly-used data assimilation methods are being adapted for use with agent-based models with the aim of allowing optimisation in response to new data in real-time. However, existing methods face difficulties working with categorical parameters, which are common in agent-based models. This paper presents a new method, the RJUKF, that combines the Unscented Kalman Filter (UKF) data assimilation algorithm with elements of the Reversible Jump (RJ) Markov chain Monte Carlo method. The proposed method is able to conduct data assimilation on both continuous and categorical parameters simultaneously. Compared to similar techniques for mixed state estimation, the RJUKF has the advantage of being efficient enough for online (i.e. real-time) application. The new method is demonstrated on the simulation of a crowd of people traversing a train station and is able to estimate both their current position (a continuous, Gaussian variable) and their chosen destination (a categorical parameter). This method makes a valuable contribution towards the use of agent-based models as tools for the management of crowds in busy places such as public transport hubs, shopping centres, or high streets.
</blockquote>

