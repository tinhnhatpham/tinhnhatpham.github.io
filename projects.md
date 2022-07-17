---
layout: default
title: Projects
permalink: /my_projects/
---
<!--
{% for project in site.projects %}
  <h2>
    <a href="{{ project.url }}">
      {{ project.class }} - {{ project.family }}
    </a>
  </h2>
{% endfor %}
-->

<!-- create categories array-->
{% assign categories_array = "" | split:"|" %}

<!--Add each unique 'my_collection' category to the array-->
{% for post in site.projects %}
    {% for category in post.categories %}
        {% assign categories_array = categories_array | push: category | uniq %}
    {% endfor %}
{% endfor %}

<!--Output the categories-->
{% for category in categories_array %}
   <h1> {{category}} </h1>
    {% assign projects_by_cat = site.projects | where: "categories", category %}
    {% for project in projects_by_cat %}
   <h2>
    <a href="{{ project.url }}">
        {{ project.title }}
    </a>
   </h2>
    {% endfor %}
{% endfor %}