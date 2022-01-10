---
layout: post
title: "Emulating Stochastic Models"
tagline:
category: announce
tags: [abm, uncertainty, data assimilation, emulation]
---

# Exploration of Gaussian processes for emulation of stochastic models

<figure style="float:right; width:60%; padding:10px;" >
<a href="https://github.com/Urban-Analytics/stochastic-gp/"><img src="{{site.baseurl}}/figures/stochastic_emulation_example.png" alt="Example graph showing uncertainty in bus predictions" /></a>
</figure>

Working with Turing Research Software Engineer [Louise Bowler](https://github.com/LouiseABowler), we have recently finished a short project that experimented with the use Gaussian Processes (GPs) to emulate agent-based models. It was developed as part of the project [Uncertainty in agent-based models for smart city forecasts](https://www.turing.ac.uk/research/research-projects/uncertainty-agent-based-models-smart-city-forecasts), funded by the [Alan Turing Institute](https://www.turing.ac.uk/).

The results are available in full from the [GitHub Repo](https://github.com/Urban-Analytics/stochastic-gp). The emulators worked well at emulating the behaviour of deterministic agent-based models. But when a degree of randomness was introduced into the model, as is often the case in agent-based modelling, they struggled to cope with the additional uncertainty and did not emulate the behaviour of the underlying models particularly well. The project has identified a number of interesting questions that need further research.



