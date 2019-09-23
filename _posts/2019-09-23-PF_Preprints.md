---
layout: post
title: "New Preprint: Real-time Crowd Simulation with a Particle Filter"
tagline: ""
category: announce
tags: [abm, data assimilation, particle filter]
---

                    
<iframe src="{{site.url}}{{site.baseurl}}/p/2019-09-20-Crowd_Simulation-CASA.html" style="width:320px; height:210px; float:right"></iframe> 

I have just uploaded a new preprint to the archive [new paper to the archive ](http://arxiv.org/abs/1908.08288) about using a particle filters to assimilate data into a real-time model of pedestrian crowding:
                    
**Malleson, N.**, Kevin Minors, Le-Minh Kieu, Jonathan A. Ward, Andrew A. West, Alison Heppenstall (2019) Simulating Crowds in Real Time with Agent-Based Modelling and a Particle Filter. _Preprint_: [arXiv:1909.09397 [cs.MA]](https://arxiv.org/abs/1909.09397).

The abstract is:


<p>Agent-based modelling is a valuable approach for systems whose behaviour is driven by the interactions between distinct entities. They have shown particular promise as a means of modelling crowds of people in streets, public transport terminals, stadiums, etc. However, the methodology faces a fundamental difficulty: there are no established mechanisms for dynamically incorporating real-time data into models. This limits simulations that are inherently dynamic, such as pedestrian movements, to scenario testing of, for example, the potential impacts of new architectural configurations on movements. This paper begins to address this fundamental gap by demonstrating how a particle filter could be used to incorporate real data into an agent-based model of pedestrian movements at run time. The experiments show that it is indeed possible to use a particle filter to perform online (real time) model optimisation. However, as the number of agents increases, the number of individual particles (and hence the computational complexity) required increases exponentially. By laying the groundwork for the real-time simulation of crowd movements, this paper has implications for the management of complex environments (both nationally and internationally) such as transportation hubs, hospitals, shopping centres, etc. </p>


The paper was also recently presented at the [MacArthur Workshop on Urban Modelling & Complexity Science](https://www.eventbrite.co.uk/e/the-macarthur-workshop-on-urban-modelling-complexity-science-tickets-69983794413), [Centre for Advanced Spatial Analysis](http://www.casa.ucl.ac.uk/), [UCL](http://www.casa.ucl.ac.uk/), 19-20 September 2019. The slides are available above ([direct link]({{site.url}}{{site.baseurl}}/p/2019-09-20-Crowd_Simulation-CASA.html)).

