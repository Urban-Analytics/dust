---
layout: page
title:  Data Assimilation for Agent-Based Modelling
tagline: 
---

<img style="float:right; padding:10px ;width:200px; height:auto;" src="./figures/erc_logo.jpg" alt="ERC logo" />



This is the website for the _dust_ research project, a new initiative at the [University of Leeds](http://www.leeds.ac.uk/) that has been funded with €1.5M from the [European Research Council](https://erc.europa.eu/). It started in Janurary 2018. 

 - For the latest news, see the [blog]({{site.baseurl}}/blog.html).
 - See the following pages for some relevant [presentations]({{site.baseurl}}/presentations.html) and [publications]({{site.baseurl}}/publications.html).
 - If you have any questions, or would like to discuss the project, then please contact the Principal Investigator: [Nick Malleson](http://www.nickmalleson.co.uk/).
 - For the model code and other programming work see the [GitHub page](https://github.com/Urban-Analytics/dust)

## DUST Overview
 
<figure>
	<img style="float:right; width:50%" src="./figures/shutterstock_788458099-small.jpg" alt="image of a city" />
</figure>

Civil emergencies such as flooding, terrorist attacks, fire, etc., can have devastating impacts on people, infrastructure, and economies. Knowing how to best respond to an emergency can be extremely difficult because building a clear picture of the emerging situation is challenging with the limited data and modelling capabilities that are available. Agent-based modelling (ABM) is a field that excels in its ability to simulate human systems and has therefore become a popular tool for simulating disasters and for modelling strategies that are aimed at mitigating developing problems. However, the field suffers from a serious drawback: models are not able to incorporate up-to-date data (e.g. social media, mobile telephone use, public transport records, etc.). Instead they are initialised with historical data and therefore their forecasts diverge rapidly from reality.

To address this major shortcoming, this new research project will develop dynamic data assimilation methods for use in ABMs. These techniques have already revolutionised weather forecasts and could offer the same advantages for ABMs of social systems. There are serious methodological barriers that must be overcome, but this research has the potential to produce a step change in the ability of models to create accurate short-term forecasts of social systems.

The project will evidence the efficacy of the new methods by developing a cutting-edge simulation of a city – entitled the Dynamic Urban Simulation Technique (DUST) – that can be dynamically optimised with streaming ‘big’ data. The model will ultimately be used in three areas of important policy impact: (1) as a tool for understanding and managing cities; (2) as a planning tool for exploring and preparing for potential emergency situations; and (3) as a real-time management tool, drawing on current data as they emerge to create the most reliable picture of the current situation.

## Latest Posts

<ul class="posts">
  {% for post in site.posts %}
    <li><span>{{ post.date | date_to_string }}</span> &raquo; <a href="{{ site.baseurl }}{{ post.url }}">{{ post.title }}</a></li>
  {% endfor %}
</ul>

