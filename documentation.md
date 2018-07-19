---
layout: page
title: Project Documentation 
tagline: An overview of the aims of DUST
---

_The following is the original application that was submitted to the Eurpoean Research Council_

<p style="border:3px; border-style:solid; border-color:#AAAAAA; padding: 1em;"> This research will create a new method for dynamically assimilating
data into agent-based models. This will create a step-change in our
ability to reliably simulate urban systems and to forecast of the
impacts of civil emergencies (and their management plans) on human
populations. </p>

# Abstract 

Civil emergencies such as flooding, terrorist attacks, fire, etc., can
have devastating impacts on people, infrastructure, and economies.
Knowing how to best respond to an emergency can be extremely difficult
because building a clear picture of the emerging situation is
challenging with the limited data and modelling capabilities that are
available. Agent-based modelling (ABM) is a field that excels in its
ability to simulate human systems and has therefore become a popular
tool for simulating disasters and for modelling strategies that are
aimed at mitigating developing problems. However, the field suffers from
a serious drawback: models are not able to incorporate up-to-date data
(e.g. social media, mobile telephone use, public transport records,
etc.). Instead they are initialised with historical data and therefore
their forecasts diverge rapidly from reality.

To address this major shortcoming, this research will develop _dynamic
data assimilation_ methods for use in ABMs. These
techniques have already revolutionised weather forecasts and could offer
the same advantages for ABMs of social systems. There are serious
methodological barriers that must be overcome, but this research has the
potential to produce a step change in the ability of models to create
accurate short-term forecasts of social systems. The project is largely
methodological, and will evidence the efficacy of the new methods by
developing a cutting-edge simulation of a city -- entitled the Dynamic
Urban Simulation Technique (DUST) -- that can be dynamically optimised
with streaming 'big' data. The model will ultimately be used in three
areas of important policy impact: (1) as a tool for understanding and
managing cities; (2) as a planning tool for exploring and preparing for
potential emergency situations; and (3) as a real-time management tool,
drawing on current data as they emerge to create the most reliable
picture of the current situation.



# What is the problem and why is it important?

Civil emergencies such as flooding, terrorist attacks, fire, train/air
crashes, earthquakes, severe short-term air quality deterioration, etc.,
can have devastating impacts on people, infrastructure, and
economies \[[8](#Xcabinet_office_national_2015)\]. Knowing how to best
respond to a developing emergency can be extremely difficult. This is
because building a clear picture of the emerging situation can be
challenging with the limited data available and also because, although
models for internal evacuations (e.g. buildings, aeroplanes, etc.) are
relatively well developed, models for evacuating urban regions are not.
Those that could be used are often based on aggregate mathematical
equations \[see, e.g. [23](#Xkachroo_pedestrian_2009)\] and struggle to
account for behavioural heterogeneity in the population or the highly
complex nature of the physical environment. A disaggregate simulation
method that is able to incorporate the limited up-to-date data that
emerge during a crisis _and_ simulate the underlying
system to a high degree of detail would be an extremely important
development.

Agent-based modelling (ABM) is a field that excels in its ability to
simulate human systems \[[6](#Xbonabeau_agent_2002)\] and has
occasionally been used for simulating civil
emergencies \[[13](#Xcrooks_gis_2013), [45](#Xmustapha_modeling_2013), [48](#Xren_agent-based_2009), [50](#Xschoenharl_design_2011)\].
Rather than attempting to derive aggregate mathematical equations to
describe the behaviour of discrete individual entities (e.g. people),
ABMs encapsulate system-wide characteristics by simulating the behaviour
of individual 'agents' directly. This has been shown to be much more
effective at modelling complex systems than traditional aggregate
approaches \[[2](#Xbatty_building_2012)\]. However, the field suffers
from a serious drawback: models are not able to incorporate up-to-date
data that describe the state of the system (e.g. from social media,
mobile telephone use, public transport records, etc.). Instead they are
initialised solely with with historical data (censuses, surveys, etc)
and thus diverge rapidly from reality. This limits their ability to
create accurate short-term forecasts of the impacts of civil emergences
or other unusual events. To address this major shortcoming, this
research will develop _dynamic data assimilation_ methods for ABM. These techniques have already
revolutionised the accuracy of numerical weather
predictions \[[24](#Xkalnay_atmospheric_2003)\] and could offer the same
advantages for models of urban systems. There are serious methodological
barriers that must be overcome, but this research has the potential to
produce a step change in the ability of models to create accurate
short-term forecasts of social systems. The research will evidence the
efficacy of the approach through the development of a cutting-edge
simulation of urban dynamics that will be used to improve emergency
plans and responses in a case study area. []{#Q1-1-4}

# What are the main gaps and challenges?

Due to its aofrementioned ability to model complex social systems, ABM
is becoming increasingly important for modelling human systems and is an
ideal tool for simulating 'normal' urban activities as well as modelling
civil
emergencies \[e.g. [13](#Xcrooks_gis_2013), [45](#Xmustapha_modeling_2013), [48](#Xren_agent-based_2009), [50](#Xschoenharl_design_2011)\].
Although the proposed research will focus on modelling as a tool for
emergency management, the development of dynamic data assimilation
methods for ABM will have much wider applicability. It will have
applications in forecasting phenomena such as traffic congestion and
real-time crowd behaviour, and more generally as a means of utilising
the instantaneous, streaming characteristics of many modern datasets to
produce short-term, high-quality, local analyses and forecasts that can
inform agile and responsive policy-making. This vision of policy making
is one of the greatest potential advantages of the 'smart cities'
movement, but has yet to be properly realised as most initiatives are
purely reactive
\[[7](#Xbond_massdot_2015), [19](#Xgeertman_planning_2015), [26](#Xkitchin_real-time_2013), [56](#Xyamamoto_development_2015)\].
Therefore the methodological innovation proposed here has the potential
to significantly advance the field and its applicability extends well
beyond civil emergency management.

ABMs are commonly used either as _in silico_ thought
experiments or as detailed models of the real world. The latter are
commonly termed _predictive_ models and are becoming
increasingly important as a means of understanding and forecasting
change in social systems \[[16](#Xfarmer_economy_2009)\]. This is
particularly pertinent in the era of 'big data' and 'smart cities', in
which short-term urban management practices are increasingly being
driven by the vast new streams of data that are being created by
citizens and sensors (for example see
\[[19](#Xgeertman_planning_2015)\]). Sources such as mobile phone call
data records (CDRs) \[[14](#Xdiao_inferring_2016)\], public transport
smart cards \[[3](#Xbatty_smart_2013)\], vehicle traffic
counters \[[7](#Xbond_massdot_2015)\], social media
contributions \[[33](#Xmalleson_impact_2015)\], etc. hold a wealth of
information about the dynamics of cities. 'Smart' cities attempt to make
use of these data to monitor and manage city flows
\[[10](#Xching_smart_2015), [26](#Xkitchin_real-time_2013)\]. At
present, however, the majority of planning systems currently in use are
purely reactive; they are able to respond to present circumstances but
do not attempt to forecast the impacts of short-term policy
changes \[[7](#Xbond_massdot_2015), [9](#Xcampagna_role_2015), [26](#Xkitchin_real-time_2013), [56](#Xyamamoto_development_2015)\].
ABM, on the other hand, holds the potential to encapsulate these diverse
data streams and use the observations of citizens for simulations of
near real-time emergency management.

The field of ABM has a substantial drawback that limits its potential as
the planning tool of the future: namely it is unable to incorporate
emerging observational data streams to reduce uncertainty. Typically,
historical data are used to estimate suitable model parameters and
models are subsequently iterated forward in time, independently of any
new data that might arise. As the systems under study are
_complex_, model predictions diverge rapidly from
reality. Therefore a mechanism to reduce uncertainty in model
predictions in response to new information about the world must be
developed if this important field is to advance.

Fortunately, methods do exist to tackle this problem. _Dynamic data
assimilation_ (DDA) is a technique that has been widely
used in fields such as meteorology, hydrology, oceanography, etc., and
is one of the main reasons that weather forecasts have improved so
substantially in recent decades \[[24](#Xkalnay_atmospheric_2003)\]. In
effect, DDA refers to a suite of mathematical approaches that make it
possible to incorporate up-to-date observational data (from weather
stations, satellite images, etc.) into models. This makes it possible to
more accurately represent the _current_ state of the
system, and therefore reduce the uncertainty in future predictions.
However, DDA methods are intrinsic to their underlying models --
typically systems of partial differential equations -- and cannot easily
be disassociated from them for use in ABM. Figure [1](#x1-30011)
represents the overall vision for a dynamically optimised city model.
This, in turn, will spawn a new generation of social forecasting models
that will be integral to the planning and evaluation of contemporary
('smart') cities.

<figure>
<img src="{{site.baseurl}}/figures/dda-smaller.png" alt="Diagram of data assimilation" />
<figcaption>An illustration of dynamic data assimilation: the
state of a hypothetical agent-based city simulation (vertical axis) that
can be optimised in response to new data (e.g. from social media) as
they are created.</figcaption>
</figure>

What are the aims and objectives of the project?

The overarching aim of the research is to **develop data assimilation
methods for use in ABM that will underpin the next
generation of urban models**. Specifically, the research
will generate a new agent-based model of urban dynamics, optimised in
real time using data assimilation methods, that can ultimately be used
as a tool both for planning responses to emergency situations and for
_real-time_ emergency response management. The research
objectives are to:

 1. Adapt data assimilation methods that have been successfully used in
    other fields for use in agent-based modelling (activity 1).

 2. Develop a suite of companion methods, including _ensembles_ and _model emulators_, that will
    help to make agent-based modelling more amenable to dynamic data
    assimilation (activity 2).

 3.  Review the available data sources that offer insight into real-time
    urban dynamics and develop data analytics to extract information
    that is useful for the simulation (activity 3).

 4. Develop a comprehensive agent-based model -- the Dynamic Urban
    Simulation Tool (DUST) -- that will be capable of simulating the
    most common activities of all individuals in an urban area, that can
    be optimised dynamically in response to real-time data streams
    (activity 3).

 5.  Implement an emergency planning and response tool for to demonstrate
    the efficacy of the new dynamic agent-based model (activity 3).

 6.  Develop a suite of machine learning methods that estimate future
    population flows to complement and validate the new agent-based
    simulation (activity 4).

What are the novel and ground-breaking aspects of the project?

The most ground-breaking and novel aspects of this project are:

-   The development of methods for dynamically assimilating data into
    ABMs.
-   A proof-of-principle model (DUST) capable of dynamically
    assimilating real-world streaming data that can be used as an
    emergency planning tool and for more general urban management.

What do I plan to do? 

This project will establish a team who will develop cutting-edge urban
simulations that are capable of, for the first time, assimilating
real-time data to reduce error in forecasts. The team will be inherently
interdisciplinary, drawing on expertise from mathematics, geography,
computer science, and the environmental sciences. The work involves four
complementary activities. Activities 1 and 2 are the most important and
focus on the core methodological development by adapting data
assimilation and companion methods for use in ABM. Activities 3 and 4
are empirical, and will draw on the methodological innovation in
activities 1 and 2 to create new models of urban dynamics.

**Activity 1 -- Dynamic Data Assimilation for Agent-Based
Models**

The quality of weather predictions has
improved significantly in recent decades, to the extent that 7-day
forecasts are now more accurate than 5-day forecasts were in the
1990s \[[4](#Xbauer_quiet_2015)\]. Part of this innovation can be
attributed to improvements in data assimilation
techniques \[[24](#Xkalnay_atmospheric_2003)\]. The need for data
assimilation was born out of data scarcity. Numerical weather prediction
models typically have two orders of magnitude more degrees of freedom
than they do observation data, so it is necessary to add additional
information into models during initialisation (termed _background_ or _first guess_ information). To
solve this problem, models began to be initialised with a combination of
real observations (from satellites, weather stations, etc.) as well as
predictions from other forecasts. This allows models to produce
estimates that are consistent in space and time, using up-to-date
observational data. This has the effect of both improving forecast
accuracy and transporting information from geographical regions that are
data rich to those that are data
poor \[[24](#Xkalnay_atmospheric_2003)\]. Both of these benefits are
extremely relevant for a model of an urban system.

There are a variety of specific methods that perform dynamic data
assimilation, including the Successive Corrections Method, Optimal
Interpolation, 3D-Var, 4D-Var, and (Ensemble) Kalman Filtering. This
activity will begin by testing all of the relevant methods, with respect
to their potential application in urban models, and then iteratively
adapting them for ABM. This, in itself, will be an extremely challenging
endeavour. Numerical weather prediction models are typically based on
aggregate differential equations, with functions linearised
mathematically, so the aforementioned data assimilation methods make
assumptions about the linearity of their underlying models. Agent-based
models simulate the interactions between discrete entities whose
behaviours are heterogeneous and models are therefore inherently
non-linear. Although early work in this area by the PI points to
Ensemble Kalman Filters (a DDA method that samples from a number of
independent model runs to efficiently estimate uncertainty) as a means
of overcoming problems with non-linearity \[[53](#Xward_dynamic_2016)\],
this is far from comprehensive and it is not clear how the method can be
adapted for anything beyond the simplest of agent-based models. To adapt
these techniques, that have not been designed to work in social systems,
the activity will begin by designing methods for the simplest forms of
agent based models, before moving on to models that encapsulate more
complicated features (agent heterogeneity, feedback mechanisms,
interactions, etc.).

This activity is the most challenging aspect of the proposal, but it
also has the potential of the greatest reward. It proposes exploratory
methodological work that has the potential to make a very significant
contribution to the field. It will begin a move towards bringing
agent-based models in to line with best practice from more established
fields, and uncover new, long-term research challenges that need to be
addressed. For this reason, Activity 1 will be the most heavily
resourced, with support by a long-term, full-time PDRA with a background
in applied mathematics and/or environmental modelling, a full-time PhD
student, and the majority of the Principal Investigator's time.

**Activity 2 -- Companion Methods**

This Activity will
draw on companion methods in fields such as meteorology to support the
core agent-based modelling and data assimilation work. Like Activity 1,
this work is novel, but, as the methods to be adapted are not embedded
in their underlying mathematical models to the same extent, it does not
need to be as heavily resourced. The two methods, in particular, that
the activity will focus on initially are _ensemble modelling_ and _emulators_.

Ensemble modelling is a technique that is used commonly in meteorology
to produce more accurate forecasts and as a means of quantifying
forecast uncertainty. It was born out of a recognition that the growth
in uncertainties that arise from inaccuracies in initial model starting
conditions needs to be encapsulated
\[[30](#Xlorenz_deterministic_1963)\] and is one of innovations that has
lead to the largest improvement in the predictive ability of
forecasts \[[4](#Xbauer_quiet_2015)\]. Rather than running a single
model, an _ensemble_ of models are each initialised
with small variations in their starting conditions. By comparing the
results of different individual model instances it is possible to
quantify the impact of the different starting conditions.

There are a number of methodological difficulties regarding ensemble
modelling that Activity 2 will address. Firstly, because the behavioural
theories on which ABMs are based contain a degree of uncertainty (e.g.
we are rarely _certain that an individual will take a
particular action), agent-based models are not deterministic so
identical models will naturally diverge to some extent. Unless this
divergence is understood and quantified, the computational difficulty in
running ensembles of models will increase greatly, as numerous model
runs are required for each instance in an ensemble. Furthermore, models
might diverge to the point of being incomparable, whereby it would be
impossible to create a single representative summary for an ensemble
instance. This is a difficulty that ABM in general has yet to overcome,
and one that this project will contribute to. Emulators (discussed
below) will be useful here. The second difficulty is that the accuracy
of the social behavioural theories that ultimately drive agent-based
models are much less certain than the laws that drive physical systems.
Hence uncertainty is introduced not just through the input data, but
also through inaccuracies in the internal model
dynamics \[[49](#Xschindler_about_2013)\]. Generally, although ensemble
modelling is desired in the field of agent-based
modelling \[[20](#Xgreeven_emergence_2016)\], there is a lack of
research into how it should be best used.

The second technique that the activity will adapt is that of model
_emulators_. These are simplified representations of a
more complex and computationally expensive model that are easier to
compute \[[29](#Xlee_complexities_2015)\]. Emulators are used in a range
of
fields \[[5](#Xbennett_systems_2005), [18](#Xfrolov_fast_2009), [51](#Xstruebing_computer-aided_2013)\],
but are in their infancy in agent-based
modelling \[[47](#Xoremland_optimization_2014)\]. Agent-based models are
typically extremely computationally expensive, so developing a means of
reliably emulating models is essential. One of the most difficult
aspects of the work will be designing emulators that are able to account
for the non-linearity that agent-based models exhibit. The Activity will
begin by reviewing existing implementations that use
regression \[[21](#Xhappe_agent-based_2006)\], before exploring more
advanced methods that have already been used successfully to create
emulators for models of non-linear
systems \[[44](#Xmarrel_global_2011)\], although not for agent-based
models.

**Activity 3 -- Modelling Urban Dynamics: the Dynamic Urban Simulation
Tool (DUST)**

Activity 3 will enact the methodological
work in the preceding activities by implementing the Dynamic Urban
Simulation Tool (DUST). This agent-based modelling tool will create
hypothetical proxy agents to represent _all_ individuals in an urban area and simulate the most
'common' behaviours that ultimately drive urban dynamics (e.g.
commuting, shopping, education, etc.) as determined from prior empirical
research \[[17](#Xfisher_multinational_2015), [27](#Xlader_time_2006)\].
The ultimate outcome will be a dynamically-optimised city model that can
be used to for three areas of important policy impact: (1) as a tool for
understanding and managing cities in its own right; (2) as a planning
tool for exploring potential emergency scenarios and preparing for them;
and (3) as a _real-time_ management tool that runs
_during_ an emergency, drawing on current data as they
emerge to create the most reliable picture of the current situation and
subsequently produce highly accurate short-term forecasts. DUST will not
attempt to replace existing tools that focus on efficient information
sharing and coordination of the emergency
services \[[8](#Xcabinet_office_national_2015), [46](#Xordnance_survey_resiliencedirect:_2016), [54](#Xwest_yorkshire_resilience_forum_west_2013)\].
Rather, the model will allows policy makers to better understand how the
wider population will react in the event of disruption to
infrastructure, and how policies can be designed to limit this
disruption. Although this activity is challenging, it is achievable with
the resources requested. The PI has considerable expertise in the
technologies required to build such a model, such as ABM and synthetic
population
generation \[[22](#Xheppenstall_space_2016), [31](#Xmalleson_agent-based_2010), [32](#Xmalleson_using_2012), [34](#Xmalleson_towards_2011), [35](#Xmalleson_analysis_2012), [36](#Xmalleson_generating_2013), [37](#Xmalleson_prototype_2008), [38](#Xmalleson_agent-based_2009), [39](#Xmalleson_agent-based_2014), [40](#Xmalleson_crime_2010), [41](#Xmalleson_using_2013), [42](#Xmalleson_implementing_2012), [43](#Xmalleson_optimising_2014)\],
and the challenges are largely technical rather than methodological.
Although large agent-based models have been
attempted \[[15](#Xepstein_modelling_2009)\], a model that
simultaneously captures individual behavioural complexity, a realistic
environment, and dynamic optimisation from real data will be a new and
timely addition to the field.

**Activity 4 -- Deep Learning Models of Urban Dynamics**

Validating agent-based models (i.e. quantifying the extent to which they
are able to accurately represent the system under study) is one of the
key ongoing challenges for the
discipline \[[11](#Xcrooks_key_2008), [12](#Xcrooks_introduction_2012)\].
Although data assimilation will reduce uncertainty in the model outcomes
by "using all the available information" \[[52](#Xtalagrand_use_1991)\],
it will be important to apply additional validation methods.
"Docking" \[[1](#Xaxtell_aligning_1996), [55](#Xwilensky_making_2007)\]
is commonly used, whereby a second model is used ascertain whether
similar results can be replicated. This project proposes a novel
approach: drawing on the ongoing innovations in the field of machine
learning to adapt a recurrent neural network (RNN) to model urban flows
at an aggregate level. Unlike traditional neural networks, that are
stateless, RNNs maintain information about a history of past
inputs \[[28](#Xlecun_deep_2015)\]. They use this historical state
vector to predict future states. For example, RNNs have been used to
generate new text in the style of the author that they were initialised
with, e.g. Shakespeare \[[25](#Xkarpathy_unreasonable_2015)\]. Therefore
data about population flows or densities (from cameras that count the
number of passers-by, aggregate mobile phone counts, etc.) will be used
to initialise an RNN that can subsequently be used to make short-term
predictions about future urban dynamics. This approach is novel and
interesting in its own right -- neural networks have not been applied to
the problem of predicting short-term population flows -- and will
provide a valuable means of validating the research results.
[]{#Q1-1-13}

# What are the main risks with the project?

This is a high risk, high reward project. It proposes novel, ambitious,
and innovative methodological developments that, as with all
methodological work, carry the risk of not satisfying the original aims.
However, the potential rewards far outweigh the risks; it is the use of
incomplete and unproven methods that makes this project state of the
art. If successful, ability of ABMs to incorporate diverse data and make
more accurate predictions will be of direct relevance to the community
of urban planners, policy makers, and agent-based modellers. The
empirical outcomes of the work have the potential to revolutionise urban
planning in the context of smart cities by providing a means of
accurately simulating urban dynamics at higher levels of accuracy than
has been possible before.

The risks will be mitigated through iterative, achievable work packages
and by resourcing the activities with the highest risk the most heavily.
The risks are also mediated by the quality of the research team. The PI
has a wealth of experience in building highly advanced models,
particularly agent-based
models \[[22](#Xheppenstall_space_2016), [39](#Xmalleson_agent-based_2014), [41](#Xmalleson_using_2013)\],
and is well versed in problems of model execution
cost \[[42](#Xmalleson_implementing_2012)\], the development of
innovative methods for model
optimisation \[[43](#Xmalleson_optimising_2014)\], and with early
explorations of dynamic data assimilation methods for agent-based
modelling \[[53](#Xward_dynamic_2016)\]. The post-doctoral researchers
will be employed on long-term contracts which will encourage the
strongest applicants. Furthermore, through attendance at conferences and
workshops, the project will draw on the growing community of agent-based
modellers to engage with and support the work. Most importantly, if
Activity 1 does proceed as planned, the development of ensemble models
and emulators (activity 2) and the parts of the DUST model that do not
assimilate data (activity 3) are all of significant value their own
right.

# References

[1]    Axtell, R., Axelrod, R., Epstein, J. M., and Cohen, M. D. Aligning simulation models: A case study and results. Computational & Mathematical Organization Theory 1, 2 (Feb. 1996), 123–141.

[2]    Batty, M. Building a science of cities. Cities 29, 1 (2012), S9–S16.

[3]    Batty, M., Manley, E., Milton, R., and Reades, J. Smart London. In Imagining the Future City: London 2062, S. Bell and J. Paskins, Eds., 1 ed. Ubiquity Press, 2013, pp. 31–40.

[4]    Bauer, P., Thorpe, A., and Brunet, G. The quiet revolution of numerical weather prediction. Nature 525, 7567 (Sept. 2015), 47–55.

[5]    Bennett, E. M., Cumming, G. S., and Peterson, G. D. A Systems Model Approach to Determining Resilience Surrogates for Case Studies. Ecosystems 8, 8 (Dec. 2005), 945–957.

[6]    Bonabeau, E. Agent based modeling: Methods and techniques for simulating human systems. Proceedings of the National Academy of Sciences 99, 90003 (2002), 7280–7287.

[7]    Bond, R., and Kanaan, A. MassDOT Real Time Traffic Management System. In Planning Support Systems and Smart Cities, S. Geertman, J. Ferreira, R. Goodspeed, and J. Stillwell, Eds. Springer International Publishing, Cham, 2015, pp. 471–488.

[8]    Cabinet Office. National Risk Register of Civil Emergencies, 2015.

[9]    Campagna, M., Floris, R., Massa, P., Girsheva, A., and Ivanov, K. The Role of Social Media Geographic Information (SMGI) in Spatial Planning. In Planning Support Systems and Smart Cities, S. Geertman, J. Ferreira, J. Stillwell, and R. Goodspeed, Eds., Lecture Notes in Geoinformation and Cartography. Springer International Publishing, 2015.

[10]    Ching, T.-Y., and Ferreira, J. Smart Cities: Concepts, Perceptions and Lessons for Planners. In Planning Support Systems and Smart Cities, S. Geertman, J. Ferreira, R. Goodspeed, and J. Stillwell, Eds. Springer International Publishing, Cham, 2015, pp. 145–168.

[11]    Crooks, A., Castle, C., and Batty, M. Key challenges in agent-based modelling for geo-spatial simulation. Computers, Environment and Urban Systems 32, 6 (Nov. 2008), 417–430.

[12]    Crooks, A. T., and Heppenstall, A. J. Introduction to Agent-Based Modelling. In Agent-Based Models for Geographical Systems. Springer, 2012, pp. 85–105.

[13]    Crooks, A. T., and Wise, S. GIS and agent-based models for humanitarian assistance. Computers, Environment and Urban Systems 41 (Sept. 2013), 100–111.

[14]    Diao, M., Zhu, Y., Ferreira, J., and Ratti, C. Inferring individual daily activities from mobile phone traces: A Boston example. Environment and Planning B: Planning and Design 43, 5 (Sept. 2016), 920–940.

[15]    Epstein, J. M. Modelling to contain pandemics. Nature 460, 7256 (2009), 687–687.

[16]    Farmer, J. D., and Foley, D. The economy needs agent-based modelling. Nature 460, 7256 (Aug. 2009), 685–686.

[17]    Fisher, K., and Gershuny, J. Multinational Time Use Study. 2015.

[18]    Frolov, S., Baptista, A. M., Leen, T. K., Lu, Z., and van der Merwe, R. Fast data assimilation using a nonlinear Kalman filter and a model surrogate: An application to the Columbia River estuary. Dynamics of Atmospheres and Oceans 48, 1-3 (Oct. 2009), 16–45.

[19]    Geertman, S., Ferreira, J., Goodspeed, R., and Stillwell, J., Eds. Planning Support Systems and Smart Cities. Lecture Notes in Geoinformation and Cartography. Springer International Publishing, Cham, 2015.

[20]    Greeven, S., Kraan, O., Chappin, E. J. L., and Kwakkel, J. H. The Emergence of Climate Change Mitigation Action by Society: An Agent-Based Scenario Discovery Study. Journal of Artificial Societies and Social Simulation 19, 3 (2016), 9.

[21]    Happe, A., Kellermann, K., and Balmann, A. Agent-based analysis of agricultural policies: an illustration of the agricultural policy simulator AgriPoliS, its adaptation, and behavior. Ecology and Society 11, 1 (2006), 49.

[22]    Heppenstall, A., Malleson, N., and Crooks, A. Space, the Final Frontier: How Good are Agent-Based Models at Simulating Individuals and Space in Cities? Systems 4, 1 (Jan. 2016), 9.

[23]    Kachroo, P. Pedestrian dynamics: mathematical theory and evacuation control. CRC Press, Boca Raton, 2009.

[24]    Kalnay, E. Atmospheric Modeling, Data Assimilation and Predictability. Cambridge University Press, 2003.

[25]    Karpathy, A. The Unreasonable Effectiveness of Recurrent Neural Networks, 2015.

[26]    Kitchin, R. The Real-Time City? Big Data and Smart Urbanism. SSRN Electronic Journal (2013).

[27]    Lader, D., Short, S., and Gershuny, J. The Time Use Survey, 2005: How we spend our time. Tech. rep., Office for National Statistics, London, UK, 2006.

[28]    LeCun, Y., Bengio, Y., and Hinton, G. Deep learning. Nature 521, 7553 (May 2015), 436–444.

[29]    Lee, J.-S., Filatova, T., Ligmann-Zielinska, A., Hassani-Mahmooei, B., Stonedahl, F., Lorscheid, I., Voinov, A., Polhill, G., Sun, Z., and Parker, D. C. The Complexities of Agent-Based Modeling Output Analysis. Journal of Artificial Societies and Social Simulation 18, 4 (2015).

[30]    Lorenz, E. N. Deterministic Nonperiodic Flow. Journal of the Atmospheric Sciences 20, 2 (Mar. 1963), 130–141.

[31]    Malleson, N. Agent-Based Modelling of Burglary. PhD thesis, School of Geography, University of Leeds, UK, 2010.

[32]    Malleson, N. Using Agent-Based models to Simulate Crime. In Agent-Based Models of Geographical Systems. Springer, 2012, pp. 411–434.

[33]    Malleson, N., and Andresen, M. A. The impact of using social media data in crime rate calculations: shifting hot spots and changing spatial patterns. Cartography and Geographic Information Science 42, 2 (2015), 112–121.

[34]    Malleson, N., and Birkin, M. Towards victim-oriented crime modelling in a social science e-infrastructure. Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences 369, 1949 (2011), 3353–3371.

[35]    Malleson, N., and Birkin, M. Analysis of crime patterns through the integration of an agent-based model and a population microsimulation. Computers, Environment and Urban Systems 36, 6 (2012), 551–561.

[36]    Malleson, N., and Birkin, M. Generating Individual Behavioural Routines from Massive Social Data for the Simulation of Urban Dynamics. In Proceedings of the European Conference on Complex Systems 2012, T. Gilbert, M. Kirkilionis, and G. Nicolis, Eds., Springer Proceedings in Complexity. Springer International Publishing, 2013, pp. 849–855.

[37]    Malleson, N., and Brantingham, P. Prototype burglary simulations for crime reduction and forecasting. Crime Patterns and Analysis 2 (2008), 47–66.

[38]    Malleson, N., and Evans, A. An agent-based model of burglary. Environment and Planning B: Planning and Design 36, 6 (2009), 1103–1123.

[39]    Malleson, N., and Evans, A. Agent-Based Models to Predict Crime at Places. In Encyclopedia of Criminology and Criminal Justice, G. Bruinsma and D. Weisburd, Eds. Springer, New York, NY, 2014, pp. 41–48.

[40]    Malleson, N., Heppenstall, A., and See, L. Crime reduction through simulation: An agent-based model of burglary. Computers, Environment and Urban Systems 34, 3 (2010), 236–250.

[41]    Malleson, N., Heppenstall, A., See, L., and Evans, A. Using an Agent-Based Crime Simulation to Predict the Effects of Urban Regeneration on Individual Household Burglary Risk. Environment and Planning B: Planning and Design 40, 3 (2013), 405–426.

[42]    Malleson, N., See, L., Evans, A., and Heppenstall, A. Implementing comprehensive offender behaviour in a realistic agent-based model of burglary. Simulation: Transactions of the Society for Modeling and Simulation International 88, 1 (2012), 50–71.

[43]    Malleson, N., See, L., Evans, A., and Heppenstall, A. Optimising an Agent-Based Model to Explore the Behaviour of Simulated Burglars. In Theories and Simulations of Complex Social Systems, V. Dabbaghian and V. K. Mago, Eds., no. 52 in Intelligent Systems Reference Library. Springer Berlin Heidelberg, 2014, pp. 179–204.

[44]    Marrel, A., Iooss, B., Jullien, M., Laurent, B., and Volkova, E. Global sensitivity analysis for models with spatially dependent outputs. Environmetrics 22, 3 (May 2011), 383–397.

[45]    Mustapha, K., Mcheick, H., and Mellouli, S. Modeling and Simulation Agent-based of Natural Disaster Complex Systems. Procedia Computer Science 21 (2013), 148–155.

[46]    Ordnance Survey. ResilienceDirect: A common information platform for central and local resilience. Cabinet Office, London, 2016.

[47]    Oremland, M., and Laubenbacher, R. Optimization of Agent-Based Models: Scaling Methods and Heuristic Algorithms. Journal of Artificial Societies and Social Simulation 17, 2 (2014).

[48]    Ren, C., Yang, C., and Jin, S. Agent-Based Modeling and Simulation on Emergency Evacuation. In Complex Sciences, O. Akan, P. Bellavista, J. Cao, F. Dressler, D. Ferrari, M. Gerla, H. Kobayashi, S. Palazzo, S. Sahni, X. S. Shen, M. Stan, J. Xiaohua, A. Zomaya, G. Coulson, and J. Zhou, Eds., vol. 5. Springer Berlin Heidelberg, Berlin, Heidelberg, 2009, pp. 1451–1461.

[49]    Schindler, J. About the Uncertainties in Model Design and Their Effects: An Illustration with a Land-Use Model. Journal of Artificial Societies and Social Simulation 16, 4 (2013).

[50]    Schoenharl, T., and Madey, G. Design and Implementation of An Agent-Based Simulation for Emergency Response and Crisis Management. Journal of Algorithms & Computational Technology 5, 4 (2011), 601–622.

[51]    Struebing, H., Ganase, Z., Karamertzanis, P. G., Siougkrou, E., Haycock, P., Piccione, P. M., Armstrong, A., Galindo, A., and Adjiman, C. S. Computer-aided molecular design of solvents for accelerated reaction kinetics. Nature Chemistry 5, 11 (Sept. 2013), 952–957.

[52]    Talagrand, O. The Use of Adjoint Equations in Numerical Modelling of the Atmospheric Circulation. In Automatic Differentiation of Algorithms: Theory, Implementation, and Application, A. Griewank and G. F. Corliss, Eds. SIAM, Philadelphia, PA, 1991, pp. 169–180.

[53]    Ward, J. A., Evans, A. J., and Malleson, N. S. Dynamic calibration of agent-based models using data assimilation. Open Science 3, 4 (2016).

[54]    West Yorkshire Resilience Forum. West Yorkshire Public Community Risk Register, 2013.

[55]    Wilensky, U., and Rand, W. Making Models Match: Replicating an Agent-Based Model. Journal of Artificial Societies and Social Simulation 10, 4 (2007), 2.

[56]    Yamamoto, S. Development and Operation of Social Media GIS for Disaster Risk Management in Japan. In Planning Support Systems and Smart Cities, S. Geertman, J. Ferreira, J. Stillwell, and R. Goodspeed, Eds., Lecture Notes in Geoinformation and Cartography. Springer International Publishing, 2015.

