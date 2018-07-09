---
layout: page
title : Blog
header : Blog
group: navigation
---

<ul class="posts">
  {% for post in site.posts %}
    <li><span>{{ post.date | date_to_string }}</span> &raquo; <a href="{{ base.url }}{{ post.url }}">{{ post.title }}</a></li>
  {% endfor %}
</ul>

