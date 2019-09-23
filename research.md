---
layout: page
title: Research and Publications
tagline: 
---

<figure style="width:45%; float:right; padding-left: 1em;">
  <img src="./figures/shutterstock_788457058-small.jpg" alt="Picture of london skyline with lines that look like flows of data" />
</figure>


The aim of DUST is to create new methods for dynamically assimilating data into agent-based models which will allow us to more reliably simulate the daily ebb abd flow of urban systems. This is a big challenge, so there are a number of different related research questions that DUST, and other projects, are working on.

## Publications

**Malleson, N.**, Kevin Minors, Le-Minh Kieu, Jonathan A. Ward, Andrew A. West, Alison Heppenstall (2019) Simulating Crowds in Real Time with Agent-Based Modelling and a Particle Filter. _Preprint_: [arXiv:1909.09397 [cs.MA]](https://arxiv.org/abs/1909.09397).
 	
Kieu, Le-Minh, **N. Malleson**, and A. Heppenstall (2019) Dealing with Uncertainty in Agent-Based Models for Short-Term Predictions’. _Preprint_ [arXiv:1908.08288 [cs.MA]](https://arxiv.org/abs/1908.08288).


## Projects


### Uncertainty in agent-based models for smart city forecasts

<figure style="width:20%;float:left;" >
<img src="{{site.url}}{{site.baseurl}}/figures/LOGO_TURING.png" alt="The ALan Turing Institute logo" />
</figure>

Full details on the [Alan Turing Institute Website](https://www.turing.ac.uk/research/research-projects/uncertainty-agent-based-models-smart-city-forecasts)

Individual-level modelling approaches, such as agent-based modelling (ABM), are ideally suited to modelling the behaviour and evolution of social systems. However, there is inevitably a high degree of uncertainty in projections of social systems, so one of the key challenges facing the discipline is the quantification of uncertainty within the outputs of these models. The aim of this project is to develop methods that can be used to better understand uncertainty in individual-level models. In particular, it will explore and extend the state-of-the-art in two related areas: ensemble modelling and associated emulators for use in individual-level models.

### Probabilistic Programming and Data Assimilation for Next Generation City Simulation

This Leeds Institute for Data Alaytics internship project, which is being funded by [Improbable](https://improbable.io/), is experimenting with the use of a Bayesian inference and probabilistic programming languages, such as [Keanu](https://github.com/improbable-research/keanu) to better capture the uncertainty in simualtions of human systems.

Relevant conference presentation: [Understanding Input Data Requirements and Quantifying Uncertainty for Successfully Modelling 'Smart' Cities]({{site.baseurl}}/p/2018-07-15-abmus-da.html). Presentation to the 3rd International Workshop on Agent-Based Modelling of Urban Systems ([ABMUS](http://modelling-urban-systems.com/abmus2018)), part of the International Conference on Autonomous Agents and Multiagent Systems ([AAMAS 2018](http://celweb.vuse.vanderbilt.edu/aamas18/home/)). 10-15 July, Stockholm. [Full abstract (pdf)]({{site.baseurl}}/p/2018-07-15-abmus-da-abstract.pdf). 




### Latest Blog Posts

<ul class="posts">
  {% for post in site.posts %}
    <li><span>{{ post.date | date_to_string }}</span> &raquo; <a href="{{ site.baseurl }}{{ post.url }}">{{ post.title }}</a></li>
  {% endfor %}
</ul>

