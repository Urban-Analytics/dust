# dust website

This branch ([gh-pages](https://github.com/Urban-Analytics/dust/tree/gh-pages)) contains the code for the DUST project website: https://urban-analytics.github.io/dust/


## Slides tutorial (how to make/edit slides)

### Introduction

The website uses [Jekyll](https://jekyllrb.com/) to convert markdown and html into a full website. It also uses [reveal](https://revealjs.com/) to make the actual slides. You can install jekyll locally, but if you push the [gh-pages](https://github.com/Urban-Analytics/dust/tree/gh-pages) branch to github it will automatically build the website and make it available at  https://urban-analytics.github.io/dust/

### Website structure (jekyll)

There are markdown documents in this folder. Each of these is rendered as an html page when the website is built by jeykll. E.g. the [publications.md](./publications.md) file is turned into the [publications.html](https://urban-analytics.github.io/dust/publications.html) page.

Blog entries are placed in the [_posts/](./_posts) folder.

The presentations are all normal `html` files, placed in the [p](./p) directory (see below).

### Writing new presentations using 'reveal'

Presentations go in the [p](./p) directory and a link to them should also be added in the [presentations.md](./presentations.md) file (otherwise there's no way to find them). There are loads of [figures](./figures/) and [videos](./videos/) in their respective folder that might be useful. (If adding new figures/videos please put them in those folders).

The slides use a library called [reveal.js](https://revealjs.com/) which makes them look nice and handles moving between slides etc. The easiest way to see how this works is just to look at an existing presentation (e.g. [this one](./p/2021-09-14-ABM_DWP_Sparkle.html). There is some library loading at the top of the html file which looks complicated, but most of that doesn't need to be editted. 

A basic slide goes between `<section>` tags and looks like this:

```html
<section id="Intro">
    <h2>Overview</h2>
	<p>Introduction to ABM</p>
    <p>ABM Example: Simulating daily mobility</p>
    <p>Introduction to Microsimulation</p>
    <p>Microsimulation examples</p>
    <p class="l2">Simulating implications for tax policies</p>
    <p class="l2">Future ageing</p>
    <p>Discussion</p>
</section>
```

There is a slide header and bullet points. The `class="l2"` class makes the bullet points indended to level 2. There is also a `class="l3"` class for further indentation.

Figures can be included like:

```html
<img class="right" data-src="../figures/regression.png" alt="Diagram of regression" />
```

Note that the `class="right"` moves the figure to the right (`left` is also available) and `data-src` is used rather than `src` because it allows reveal to pre-load the images and make transitions between slides quicker (`src` also works and I've not noticed a huge difference to be honest).

You can also have background videos or images. E.g. the below uses a timelapse video that is played behind the slide. Note the use of the `whitebackground2` and `fragment`. `whitebackground2` makes an opaque white box around the text (otherwise it can be hard to read with the video/image behind) and `fragment` is a reveal feature that hides an element until you step through it (so on the slide you can see the video unobstructed until you progress through the slides to make the text appear).

```html
<section data-background-video="../videos/trafalgar_timelapse-1280×720.mp4"
               data-background-video-loop="loop"  id="Intro">
    <div class="whitebackground2 fragment">
    <h2>Overview</h2>
	    <p>Introduction to ABM</p>
        <p>ABM Example: Simulating daily mobility</p>
        <p>Introduction to Microsimulation</p>
        <p>Microsimulation examples</p>
        <p class="l2">Simulating implications for tax policies</p>
        <p class="l2">Future ageing</p>
        <p>Discussion</p>
    </div>
</section>
```





