---
layout: post
title: "New Preprint: Dealing with uncertainty in agent-based models for short-term predictions"
tagline: ""
category: announce
tags: [abm, data assimilation, particle filter]
---

<!--
<img style="float:right; width:50%" src="{{site.url}}{{site.baseurl}}/figures/bus-bunching.png" alt="Graph showing movement of buses away from the depot and the phenomena of 'bus bunching'"/>-->

<video controls autoplay style="float:right; width:60%;">
                        <source src="{{site.url}}{{site.baseurl}}/videos/bussim-pf-video.mp4" type="video/mp4"/>
                        <!-- if video doesn't work then show a picture-->
                         <img style="float:right; width:60%" src="{{site.url}}{{site.baseurl}}/figures/bus-bunching.png" alt="Graph showing movement of buses away from the depot and the phenomena of 'bus bunching'"/>
                    </video>

My colleague [Minh Kieu](https://environment.leeds.ac.uk/geography/staff/2520/dr-minh-kieu) has just uploaded a [new paper to the archive](https://arxiv.org/abs/1908.08288) about using a particle filter to incorporate real-time information into a model of a bus route. It is currently under peer review in a journal. 

Kieu, Le-Minh, Nicolas Malleson, and Alison Heppenstall. 2019. Dealing with Uncertainty in Agent-Based Models for Short-Term Predictions. _ArXiv:1908.08288 [Cs]_. [http://arxiv.org/abs/1908.08288](http://arxiv.org/abs/1908.08288).


The abstract is:


<q>_Agent-based models (ABM) are gaining traction as one of the most powerful modelling tools within the social sciences. They are particularly suited to simulating complex systems. Despite many methodological advances within ABM, one of the major drawbacks is their inability to incorporate real-time data to make accurate short-term predictions. This paper presents an approach that allows ABMs to be dynamically optimised. Through a combination of parameter calibration and data assimilation (DA), the accuracy of model-based predictions using ABM in real time is increased. We use the exemplar of a bus route system to explore these methods. The bus route ABMs developed in this research are examples of ABMs that can be dynamically optimised by a combination of parameter calibration and DA. The proposed model and framework can also be used in an passenger information system, or in an Intelligent Transport Systems to provide forecasts of bus locations and arrival times._</q>




The paper was also recently presented at the 4th International Workshop on Agent-Based Modelling of Urban Systems ([ABMUS](http://modelling-urban-systems.com/abmus2019)), part of the International Conference on Autonomous Agents and Multiagent Systems ([AAMAS 2019](http://aamas2019.encs.concordia.ca/)). 13-17 May, Montreal. The [slides]({{site.baseurl}}/p/2019-05-14-ABMUS-BusSim-MK.html) from that talk are below:

 <iframe src="{{site.url}}{{site.baseurl}}/p/2019-05-14-ABMUS-BusSim-MK.html" style="width:640px; height:420px"></iframe> 