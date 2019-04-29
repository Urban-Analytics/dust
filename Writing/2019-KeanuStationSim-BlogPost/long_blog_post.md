# Using Keanu for Data Assimilation on an Agent-Based Model

## The Problem - ABM Single Historical Calibration

Social phenomena are naturally complex and difficult to model, involving stochasticity from any number of sources. For an accurate representation of the whole system, it is often not sufficient to group individuals; instead models have to be created where each individual is simulated separately and the complex network of decision and interaction is fully represented. The collective name for these kind of models is [Microscale Models](https://en.wikipedia.org/wiki/Microscale_and_macroscale_models), the most common and best studied of which are Agent-Based Models (ABMs). 

Unlike many other modelling techniques, ABMs are not defined by a set of equations that describe a system of interest. Instead ABMs are composed of any number of autonomous agents, each given a simple set of rules to follow that will determine it's behaviour. Building the model with autonomous agents in this way allows us to highlight and study the behaviour of the individual, and means no information is lost by grouping agents together. This also means the model can represent social interaction - one of the most important and influential features of social phenomena. Modelling the dynamic interaction between agents can often lead to a phenomena called **emergence**. This is where individuals with simple rules interact in such a way as to form complex new patterns or processes. Key to the idea of emergence is that what is produced cannot be predicted, worded famously by Aristotle as "_the sum is greater than the parts_". ABMs have been extensively studied in social simulation, prompting the creation of a field all of its own - [Agent-based Social Simulation](https://en.wikipedia.org/wiki/Agent-based_social_simulation). Some examples of effective use of an ABM in social science include in [crime](https://journals.sagepub.com/doi/abs/10.1068/b38057) [1], [human migration](https://link.springer.com/article/10.1007/s10680-015-9362-0) [2], and [consumer behaviour](https://www.sciencedirect.com/science/article/pii/S2405896318312059) [3].

Despite ABM being used extensively in social simulation, it is still limited in that models are typically only calibrated (trained) once using historical data. With enough calibration data, ABMs can accurately represent a system of interest. But as these models are never deterministic (due to the stochasticity that is almost always present in social interaction models) simulations of phenomena in _real time_ will begin to diverge from the system almost immediately. With this one-shot calibration, simulations can be produced that look and act like the phenomena in question, but do not represent its true current state. This may not sound like a big problem, but there are some uses for ABMs that could benefit massively from a truly accurate simulation.

To illustrate this, let's take an example. In a recent (2015) review, Erick Mas et al. [4] investigated a number of agent-based tsunami evacuation models based in the far east and South America. Using data from coastal towns to calibrate ABMs, they found that they were able to identify bottlenecks, uncover limitations in shelter capacity or placement, provide casualty estimates, and evaluate the use of vehicles in evacuations. This information could not be collected through traditional research methods such as evacuation drills or questionnaire surveys, and can heavily influence how these scenarios are managed. However, the review also highlighted two key issues of ABM for this purpose. Firstly, human behaviour is not easy to model (but not impossible!). When making a decision, we tend to quickly compare the potential positive and negative outcomes, and choose the path that looks the most positive. But this is often a flawed process based on incomplete data, and taking into account the stress and panic in these scenarios makes it more difficult still. On top of this, these complex processes must then be captured in mathematical equations in order to add them to the model, which is not trivial. Another problem is in the Verification & Validation of the model. This is already a complicated process in ABM, as the models are based on simulations rather than equations that can be solved analytically. 

Mas and his team suggested that disaster "Big Data" could address these limitations, and point to a dataset collected during the 2011 Great East Japan Tsunami (GEJT). Mobile phones, car navigation systems, and social media provided data on things like user location and speed. This data can be used to train better models of human movement and decision under stress. However, the GEJT data is currently the only large dataset of this kind, so any potential use of it would be limited, and any conclusions would have to be taken with a pinch of salt as they might not be wholly representative. Mas theorised that producing 'virtual big data' for these situations could be very useful, but could be a challenge in itself to produce virtual data that looks and acts like real-world data. 

Another suggestion made in this paper was, in my opinion, the most interesting. Mas talked about using this big data not just to train the model initially, but also to update the model *whilst it is running*. This would give the model updated information, and the ability to modify it's state to better represent the real-world system it is meant to represent. Instead of having a simulation that looks and acts like a real-world system, we would have a model that accurately represents the current state of the system in question. 

So according to Mas' recommendation we need to find a way to incorporate up-to-date information - or in other words to __assimilate data__ - in real time into the model. I wonder where we're going with this...

## The Solution - Data Assimilation

Data Assimilation (DA) is a name given to a group of techniques developed primarily for use in numerical weather prediction, with the first of these published in 1922 by the mathematician Lewis Fry Richardson. DA is attributed as one of the major reasons for the improvement of weather prediction, and has been thoroughly studied and developed for as long as we have been predicting the weather [6]. In it's simplest form, DA is the combination of theory (usually in the form of a predictive numerical model) with observations of the system being modelled. It allows for a more accurate estimation of the state of the system, even when observations are noisy or otherwise unreliable. 

Data assimilation is a Bayesian process, and so relies on [Bayes' theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem). Bayes theorem describes the probability of an event given prior knowledge of the conditions that might relate to that event. Here is the equation:

<img src="BayesRule.png" alt="Bayes Theorem" width="700" height="300"/>

To the uninitiated Bayes theorem (and data assimilation in general) can be quite a scary concept, but it is a theory we apply in our daily life almost without thought. Every morning when we wake up we have to decide what clothes to wear that day, which heavily depends on the weather. If we have no information about the conditions outside then we don't know what clothes are appropriate, and we would basically be taking a blind guess at predicting the weather. To improve this then, we might think about what the weather was like yesterday, as the two can often be linked. Here, we are using some **prior** knowledge about the weather to inform our prediction and hopefully make it more accurate. But still the weather on consecutive days can be very different, so even after involving our prior there is a high probability that our prediction is not accurate. To improve this we need some evidence of the current state of the system, which we can get by looking out of the window. By **observing** the current weather, we are once again updating our belief in the state of the system, and hopefully making our prediction more accurate. This is essentially Bayes theorem in a nutshell; when attempting to make an accurate prediction, we take our **prior** knowledge of a system, apply some **observations**, and after combining these two things produce a (hopefully) more accurate **posterior** estimate of the state of the system. 

Data assimilation is just a special case of Bayes theorem. In data assimilation, our prior comes from a predictive model of the system in question. In weather prediction, a large scale complex model is run for a timestep (lets assume 1 hour) to form **the prediction**. Observations on the weather would then be applied, leading to a more accurate posterior prediction of the weather, after combining a prediction with some observations.

A full explanation of DA is far beyond the scope of this post (and author), so if you wish to learn more on this topic I would reccomend a book by Eugenia Kalnay [6], the full text of which can be found online [here](http://www.meteoclub.gr/proo1/Atmospheric_modeling_data_assimilation_and_predictability.pdf "Atmospheric Modelling, Data Assimilation and Predictability"). This text is well known in the field and offers an accessible introduction to DA, beginning with simpler techniques such as Nudging and the Successive Corrections Method (SCM), before moving onto the relatively complex variational approaches.

## Aims of this work

The aims of this work are two-fold, where both aims are very closely related:

1. To develop a new Data Assimilation algorithm using a Probabilistic Programming framework and
2. To successfully apply this DA algorithm on a simple pedestrian crowd model.

As stated earlier, Data Assimilation has been extensively studied in the past and many different varieties can be found in the literature. However, our proposed method using Probabilistic Programming has not been attempted before, and would therefore be breaking new ground. In addition, although there are a handful of examples of successful data assimilation on an agent-based model, these models are often in very specific forms that can be mathematically coupled to more aggregate models. The simple pedestrian crowd model we use for testing is called **StationSim**, which will be described in detail later in this post. 

## The Framework - Probabilistic Programming

Probabilistic Programming Languages (PPLs) have become something of a hot topic recently, with effective research coming from both the programming languages and machine learning communities. PPLs are based around a simple-sounding idea; to combine powerful features of programming languages, such as code abstraction and reuse, with statistical modelling. Before PPLs, creating statistical models was left exclusively to the experts. This is because defining these models and calculating them _by hand_ is by no means trivial, and most often requires considerable domain knowledge and experience.

To quote from Gordon et al. (2014)[5]:
>"probabilistic programs are usual functional or imperative programs with two added constructs: (1) the ability to draw values at random from distributions, and (2) the ability to _condition_ values of variables in a program via observations."

These are the 2 key constructs of a PPL, but they also rely on the language being able to efficiently represent distributions as native objects.

The PPL we have been using is named [Keanu](https://improbable-research.github.io/keanu/), and is in development by Improbable Worlds Ltd based in London. Keanu is currently in a pre-alpha stage, but is already a powerful PPL. It is based on directed acyclic graphical models, which in the realm of statistical modeling are also known as **Bayesian Networks**. These networks represent the conditional dependency relationships between variables in the model, on which some fairly efficient algorithms for inference have been developed. Below is an example of a simple Bayesian Network.  

<img src="directed-acyclic-graph.png" alt="SimpleModel workflow" width="500" height="300"/>

## Test Case - SimpleModel

In order to learn more about Keanu and it's algorithms, we created a test model. The purpose of this model was to be as simple as possible (hence the name SimpleModel), containing only the bare minimum features needed to carry out state estimation and data assimilation. Of these features, the most important is obviously data! In this case it is not possible to collect data from a real world system, so we use what is called a 'Digital Twin' model. This means running the model once to collect data, and then running the model a second time to carry out the assimilation and state estimation. The model workflow is outlined in **Figure 1**.

<figure>
  <img src="SimpleModel_blogPost.png" alt="SimpleModel workflow" width="440" height="600"/>
    <figcaption>
**Figure 1** - A flowchart to describe the workflow of SimpleModel. Inside the loop the model first checks to see if it has reached it's maximum iteration. If no, it samples from a new Gaussian distribution with a mean of 0 and standard deviation of 1. If the sample is above a previously determined threshold, we increment the state. If below the threshold, we decrease the state by 1. After the maximum number of iterations, the state history is written to file along with the threshold value.
    </figcaption>

Using Keanu to define the framework, the steps for data assimilation are:

- Define model parameters
- Produce the pseudotruth data
- Start the assimilation window loop
  - Initialise state
  - Step the model (200 iterations)
  - Observe the truth data
  - Create the Bayesian Network
  - Sample from the Posterior
  - Get information out of the samples
  - Reassign Posterior as Prior

### Define Model Parameters

At the start of a run, the model parameters are defined. The maximum number of iterations (`NUM_ITER`) is set at 2000, and the threshold value is determined by randomly sampling from a uniform distribution between -1 and 1.

```java
/* Model Parameters */
private static final UniformVertex threshold = new UniformVertex(-1.0, 1.0);
private static final int NUM_ITER = 2000; // Total No. of iterations

/* Hyperparameters */
private static final int WINDOW_SIZE = 200; // Number of iterations per update window
private static final int NUM_WINDOWS = NUM_ITER / WINDOW_SIZE; // Number of update windows
private static final double SIGMA_NOISE = 1.0; // Noise added to observations
private static final int NUM_SAMPLES = 2000; // Number of samples to MCMC - MH: 100000, HMC: 2000, NUTS: 2000
private static final int DROP_SAMPLES = 1; // Burn in period
private static final int DOWN_SAMPLE = 5; // Only keep every x sample

// Initialise random number generator used throughout
private static final KeanuRandom RAND_GENERATOR = new KeanuRandom();
```

A number of additional hyperparameters are set, including `WINDOW_SIZE` and `SIGMA_NOISE`. Throughout this post, some insignificant features of the code (e.g. writing results to file, assertions) are intentionally removed for clarity, but the full code is available on github [here](https://github.com/Urban-Analytics/dust/tree/master/Improbable/NativeModel-keanu) if you wish to delve deeper or have a go yourself.

### Produce Pseudotruth Data

Below is the code for the first twin of our model; producing the 'truth' data. 

```
public static void main(String[] args) {
    
    // Initialise threshold for model run
    Double truthThreshold = threshold.sample(KeanuRandom.getDefaultRandom()).getValue(0);

    /*
     ************ CREATE THE TRUTH DATA ************
     */

    // Create list to hold truth data for each iteration
    List<DoubleVertex> truthData = new ArrayList<>();
    
    // Create initial state Vertex as Gaussian distribution with mu=0 and sigma=1
    DoubleVertex truthState = new GaussianVertex(0, 1.0);

    // Generate truth data
    for (int i=0; i < NUM_ITER; i++) {
        truthState = 
                RAND_GENERATOR.nextGaussian() > truthThreshold ? truthState.plus(1) : truthState.minus(1);
        // add state to history
        truthData.add(truthState);
    }
```

Each iteration up to the maximum, we sample from a new Gaussian (default mu=0, sigma=1.0). If the sample is larger than the threshold value, we increase the value of `truthState` by 1; if the sample is below the threshold, we decrease `truthState` by 1. Increasing and decreasing a distribution is not as simple as adding or subtracting, so Keanu provides the `.plus()` and `.minus()` methods for all vertices. 

### Start the Assimilation Window Loop

Now our parameters are set and we have collected our pseudotruth data, we can start the first assimilation window. The steps contained in the loop are outlined in pseudocode for clarity; each will be discussed briefly and code shown when necessary. 

```
DoubleVertex priorMu = new GaussianVertex(0.0, SIGMA_NOISE); // Prior distribution has mean of 0 to start

List<DoubleVertex> totalStateHistory = new ArrayList<>(); // Store history from each iteration

for (int window=0; window < NUM_WINDOWS; window++) {
    
    Initialise state
    Step the State
    Observe Truth Data
    Create the Bayesian Network
    (Optimise?)
    Sample from the Posterior
    Get Information from the Samples
    Reassign Posterior as Prior

}
```

#### Initialise State

We initialise the state as a Gaussian distribution. Gaussians are computationally efficient distributions to produce, as they are defined mathematically by just 2 values - mu and sigma. Note here that the state is reinitialised at the start of each window, using `priorMu` as the mean value. 

#### Step the Model

Stepping the model is done via a simple for loop, using the same logic we used to generate the pseudotruth data. The only difference here is that we only run the model for a window at a time (in our case 200 steps). The state is saved at each iteration for visualising the results later on. 

#### Observe Truth Data

 

#### Create the Bayesian Network
#### Sample from the Posterior
#### Get Information out of the Samples
#### Reassign Posterior as Prior

Now that we have our pseudotruth data we can move onto the second twin of the model. This is where it gets interesting, as this is where Keanu's algorithms come into play. Whilst the two models are basically the same, there are some slight changes in the second to accommodate Keanu. Firstly, we run this model in 200 iteration windows known as **assimilation windows**. This is so we can define a prior distribution at the start of the window, which we can then transform with the model and our observations to produce a posterior distribution. The key part of our data assimilation algorithm as opposed to just state estimation, is that we then **reintroduce the posterior distribution as prior** at the start of the next assimilation window. We also changed the state variable to be a DoubleVertex, one of Keanu's main data structures. The DoubleVertex object represents the pdf of the model state, allowing us to use this.

## The Model - StationSim

Having developed the DA workflow on our test model, we began to modify it for our more complex agent-based model, **StationSim**. StationSim was built using the MASON simulation library (again in Java). MASON is a powerful multi-agent simulation toolkit, which is designed to be the foundation for large custom-purpose Java simulations. The full model code for StationSim is available on github [here](https://github.com/Urban-Analytics/dust/tree/master/Improbable/stationsim-keanu/src/main/java/StationSim).

StationSim is meant to be a heavily simplified representation of pedestrian movement in a train station, where passengers alight from one of multiple train doors and move through the station to an exit. Our agents are added to the model through green entrances on the left and are removed via red exits on the right. 

<img src="stationSimImage.png" alt="StationSim" width="650" height="590"/>
**Figure 2**

StationSim was created specifically as a means to test data assimilation on an ABM. It is much simpler than a real world system, yet still contains enough complexity to produce **emergent behaviour**. This emergent behaviour is the crowding pattern that forms when faster agents are stuck behind slower ones, which varies between each run due to three main sources of stochasticity. Firstly when agents are activated, we assign a random value within a range as it's desired speed. Agents are then assigned one of the two exits based on a pre-determined set of probabilities - the probability of choosing each exit differs based on the entrance they emerge from. Finally, when an agent is blocked from reaching the exit by slower moving agents, it will make a decision to move either up or down and around the slow agent(s). To make this decision, it samples from a Gaussian distribution, comparing to a threshold set at 0.5. These sources of stochasticity ensure no two runs of this model are the same, and different patterns of crowding in the model emerge with each successive run. 

## Data Assimilation on StationSim

The increased complexity of StationSim threw up a lot of challenges when adapting our previous algorithm to the new model. First and most fundamental, was the difference in the model state. Where SimpleModel's state was fully represented as a scalar integer, StationSim's is a vector with three vertices for each active agent; x position, y position and desired speed. The minimum number of agents required to see emergent crowding behaviour is 700, so the state vector must have a minimum length of 2100 vertices. 

## Results - (lets see what we have)

## Impact

The potential for this work could impact many distinct fields of study. 

## References

[1] - [Malleson, Heppenstall, See and Evans (2013)](https://journals.sagepub.com/doi/abs/10.1068/b38057)
[2] - [Klabunde & Willekens (2016)](https://link.springer.com/article/10.1007/s10680-015-9362-0)
[3] - [Huiru et al. (2018)](https://www.sciencedirect.com/science/article/pii/S2405896318312059)
[4] - [Mas et al. (2015)](https://link.springer.com/article/10.1007/s00024-015-1105-y)
[5] - [Gordon et al. (2014)](http://hci-kdd.org/wordpress/wp-content/uploads/2016/10/GORDON-HENZINGER-NORI-RAJAMANI-2014-Probabilistic-Programming.pdf)
[6] - Kalnay, E. (2003). _Atmospheric Modeling, Data Assimilation and Predictability_. Cambridge University Press.
