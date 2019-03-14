---
layout: page
title: Research 
tagline: 
---

<p style="border:3px; border-style:solid; border-color:#AAAAAA; padding: 1em;">
The aim of DUST is to create new methods for dynamically assimilating data into agent-based models which will allow us to more reliably simulate the daily ebb abd flow of urban systems. This is a big challenge, so there are a number of different related research questions that DUST, and other effors, are working on.</p>


<figure class="right">
  <img src="./figures/shutterstock_788457058-small.jpg" alt="Picture of london skyline with lines that look like flows of data" />
</figure>

## 




https://www.turing.ac.uk/research/research-projects/uncertainty-agent-based-models-smart-city-forecasts
## Latest Blog Posts

<ul class="posts">
  {% for post in site.posts %}
    <li><span>{{ post.date | date_to_string }}</span> &raquo; <a href="{{ site.baseurl }}{{ post.url }}">{{ post.title }}</a></li>
  {% endfor %}
</ul>

