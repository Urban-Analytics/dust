[Keanu documentation](https://improbable-research.github.io/keanu/)

[Keanu github](https://github.com/improbable-research/keanu)

[Keanu javadoc](https://www.javadoc.io/doc/io.improbable/keanu/0.0.24)

[DUST github](https://github.com/Urban-Analytics/dust)

[UnaryOpLambda github issue](https://github.com/improbable-research/keanu/issues/471)

# DataAssimilation.java

## Parameters

`truthModel`		- instance of Station class created to produce pseudotruth data (first 'twin')  
`tempModel`			- instance of Station class for testing Data Assimilation  

`NUM_ITER`			- Total number of iterations  
`WINDOW_SIZE` 		- Number of iterations per update window  
`NUM_WINDOWS`		- Number of update windows  

`NUM_SAMPLES`		- Number of samples to MCMC  
`DROP_SAMPLES` 		- Burn-in period (drop first x amount of samples)  
`DOWN_SAMPLE` 		- Only keep every x samples  

`SIGMA_NOISE` 		- standard deviation/noise to add to observations/gaussians

`numVerticesPP` 	- integer value to specify the number of vertices per person (PP) in the stateVector  
					(- i.e. Per agent: x_pos, y_pos, speed. Therefore 3 vertices per person, numVerticesPP = 3.)

`KeanuRandom rand`	- Keanu built in random number generator. Used just like the built in Math rng

`tempVertexList`    - List of `VertexLabels` to help in selecting specific vertices from BayesNet

`truthHistory` 		- List of truth vectors (double[]) collected at each iteration of the truthModel run
`tempHistory`       - List of tempModel vectors (DoubleTensor[]) collected at each iteration of tempModel run

## runDataAssimilation()

This is the main function of this class, and contains the data assimilation workflow:

- Runs truthModel for NUM_ITER and stores observations at each iteration (**previously stored only every 200 iter - Explain why below**)
- Start the tempModel and create the initial state vector
- Start Main loop
  - Initialise Black Box model
  - Observe some truth data
  - Create the BayesNet and Probabilistic Model
  - Optimise???
  - Sample from the Posterior
  - Get information on Posterior distribution from samples
  - Reintroduce posterior as prior for next window

Everything in this file will be explained in the order it happens in this loop. Where logic is abstracted from the loop into methods, the method will be introduced in this order despite possibly being organised in a different order in the script.

### Create Truth Data

Here we run the first twin of the model. The code for collecting truth data is contained in the method `getTruthData()`, but also calls `getTruthVector()` which will be shown below.

```java

    private static void getTruthData() {
        // start truthModel
        truthModel.start(rand);
        System.out.println("truthModel.start() has executed successfully");

        // Step truth model whilst step number is less than NUM_ITER
        while (truthModel.schedule.getSteps() < NUM_ITER + 1) { // +1 to account for an error on final run, trying to access iteration 2000 of truthHistory
            double[] truthVector = getTruthVector(); // get truthVector for observations
            truthHistory.add(truthVector); // Save for later
            truthModel.schedule.step(truthModel);
        }
        System.out.println("\tHave stepped truth model for: " + truthModel.schedule.getSteps() + " steps.");

        // Check correct number of truthVectors extracted
        assert (truthHistory.size() == NUM_ITER + 1) : "truthHistory collected at wrong number of iterations (" +
                truthHistory.size() + "), should have been: " + NUM_ITER;

        // Finish model
        truthModel.finish();
        System.out.println("Executed truthModel.finish()");
    }
```

This method calls `.start()` on the model, which is a required method from MASON that starts a lot of behind the scenes things like the scheduler. It then steps the truthModel for `NUM_ITER`, building a vector at each iteration and storing it for analysis later. After some assertions, it calls `.finish()` allowing MASON to fully stop the model and shut down it's processes. 

At the beginning of each step it calls `getTruthVector()`, to produce the vector of agent positions and speed. This vector is then stored for observations later.
```java

    private static double[] getTruthVector() {
        // Get all objects from model area (all people) and initialise a list
        List<Person> truthPersonList = new ArrayList<>(truthModel.area.getAllObjects());

        assert (!truthPersonList.isEmpty()) : "truthPersonList is empty, there is nobody in the truthModel";
        assert (truthPersonList.size() == truthModel.getNumPeople()) : "truthPersonList (" + truthPersonList.size() +
                ") is the wrong size, should be: " +
                truthModel.getNumPeople();

        // Create new double[] to hold truthVector (Each person has 3 vertices (or doubles in this case))
        double[] truthVector = new double[truthModel.getNumPeople() * numVerticesPP];

        for (Person person : truthPersonList) {
            // Get persons ID to create an index for truth vector
            int pID = person.getID();
            int index = pID * numVerticesPP;

            // Each person relates to 2 variables in truthVector (previously 3 with desiredSpeed)
            truthVector[index] = person.getLocation().x;
            truthVector[index + 1] = person.getLocation().y;
            truthVector[index + 2] = person.getCurrentSpeed();
        }
        return truthVector;
    }
```
The method loops through all the people it finds in the model, first calculating their ID and then an index from the ID. The index is calculated by multiplying the ID with the number of vertices per person; for example, the second vertex for agent 31 would be in position ((31 * 3) + 1) = 94. The x,y position and current speed of that agent are added into the 3 corresponding vector positions, and this process is repeated until the loop has accessed all agents. 

Previously the vector was only stored every 200 iterations; now it is stored at each iteration for visualising changes in the state.

### Create the State Vector

Before starting the main loop we start tempModel, and create the stateVector from this model to be updated throughout the loop. 

```java

		// Start tempModel and create the initial state
        tempModel.start(rand);
        System.out.println("tempModel.start() has executed successfully");

        CombineDoubles stateVector = createStateVector(tempModel); // initial state vector
```

`createStateVector()` needs to be called to produce the initial state vector, as well as applying the unique label to each vertex. Each agent has 3 seperate vertices; x position, y position and speed.

- collects all the people into a `Bag` object (one of MASON's collection objects)
- Loops through all people in the model using their unique ID
- Creates a unique label for each vertex based on the corresponding agent ID and vertex number (explained later), and adds the label to a list
- Creates a new `GaussianVertex` for each element with mu and sigma of 0.0.
- Sets the label for the correct vertex
- Finally returns a new `CombineDoubles` object constructed with the vertices

The code:
```java

    private static CombineDoubles createStateVector(Station model) {
        // Create new collection to hold vertices for CombineDoubles
        List<DoubleVertex> stateVertices = new ArrayList<>();
        // Get all objects from model area (all people) and initialise a list
        List<Person> tempPersonList = new ArrayList<>(model.area.getAllObjects());

        for (Person person : tempPersonList) {
            // Get person ID for labelling
            int pID = person.getID();

            // Produce VertexLabels for each vertex in stateVector and add to static list
            // Labels are named based on person ID, i.e. "Vertex 0(pID) 0(vertNumber), Vertex 0 1, Vertex 0 2, Vertex 1 0, Vertex 1 1" etc.
            VertexLabel lab0 = new VertexLabel("Vertex " + pID + " " + 0);
            VertexLabel lab1 = new VertexLabel("Vertex " + pID + " " + 1);
            VertexLabel lab2 = new VertexLabel("Vertex " + pID + " " + 2);

            // Add labels to list for reference
            tempVertexList.add(lab0);
            tempVertexList.add(lab1);
            tempVertexList.add(lab2);

            // Now build the initial state vector
            // Init new GaussianVertices as DoubleVertex is abstract; mean = 0.0, sigma = 0.0
            DoubleVertex xLoc = new GaussianVertex(0.0, 0.0);
            DoubleVertex yLoc = new GaussianVertex(0.0, 0.0);
            DoubleVertex currentSpeed = new GaussianVertex(0.0, 0.0);

            // set unique labels for vertices
            xLoc.setLabel(lab0); yLoc.setLabel(lab1); currentSpeed.setLabel(lab2);

            // Add labelled vertices to collection
            stateVertices.add(xLoc); stateVertices.add(yLoc); stateVertices.add(currentSpeed);
        }
        return new CombineDoubles(stateVertices);
    }
```
**Labels**: Each label is created with a unique string. "Vertex " followed by the persons ID, with the vertex number at the end. For example, the 2nd vertex of person 45 would be labelled "Vertex 45 1".

The method returns a `CombineDoubles` object containing all the vertices, this class is explained below `DataAssimilation.java`.

### Start the Main Loop
The loop contains the steps required to run the model in windows, observing ‘truth’ data inbetween windows and sampling from the posterior distribution. In pseudocode:

```java

				// Start data assimilation window
				for (int i = 1; i < NUM_WINDOWS + 1; i++) {
    				* Initialise the Black Box model
                    * Observe the truth data
                    * Create the BayesNet and Probabilistic Model
                    * Optimise
                    * Sample from the Posterior
                    * Get information on posterior distribution from samples
                    * Reintroduce posterior as prior for next window
				}
```
#### Initialise the Black Box Model

The black box model is a wrapper provided by Keanu's `UnaryOpLambda` class. The `predict()` method is where the model is stepped for a number of iterations, collecting the state vector at each iteration. The output of this is used with the `UnaryOpLambda` in the method below called `step()`. The `UnaryOpLambda` object wraps around the process of updating the state, in a way that allows us to generate the `BayesianNetwork` and use Keanus algorithms further down the line.

```java

    // This is how it is called inside the main loop...
    Vertex<DoubleTensor[]> box = predict(stateVector)

    // ... and the corresponding method
    private static Vertex<DoubleTensor[]> predict(CombineDoubles initialState) {
        System.out.println("\tPREDICTING...");

        // Check model contains agents
        assert (tempModel.area.getAllObjects().size() > 0) : "No agents in tempModel before prediction";

        CombineDoubles stateVector = updateStateVector(initialState); // stateVector before prediction

        // Run the model to make the prediction
        for (int i = 0; i < WINDOW_SIZE; i++) {
            tempHistory.add(stateVector.calculate()); // take observations of tempModel (num active people at each step)
            tempModel.schedule.step(tempModel); // Step all the people
            stateVector = updateStateVector(stateVector); // update the stateVector
        }

        // Check that correct number of people in model before updating state vector
        assert (tempModel.area.getAllObjects().size() == tempModel.getNumPeople()) : 
                String.format("Wrong number of people in the model before building state vector. Expected %d but got %d",
                tempModel.getNumPeople(), tempModel.area.getAllObjects().size());

        // Return OpLambda wrapper
        UnaryOpLambda<DoubleTensor[], DoubleTensor[]> box = new UnaryOpLambda<>(
                new long[0],
                stateVector,
                DataAssimilation::step
                //(currentState) -> step(currentState) // IntelliJ suggested changing this line for 'DataAssimilation::step' above^
        );

        System.out.println("\tPREDICTION FINISHED.");
        return box;
    }
```
We supply the current state vector `initialState` as a method argument, as the purpose of this method is to transform the state of the predictive model (tempModel) to produce a prediction; hence `predict()`. This method repeatedly calls another, which I will show below.

The `UnaryOpLambda` is a complicated object, and its full function is not obvious from its code or documentation. As gordoncaleb (Improbable head(?) of engineering) explained on the [github issue](https://github.com/improbable-research/keanu/issues/471), the purpose of the UnaryOpLambda class is to allow us to apply a single operation (predict) on a range of values, returning an object of type `Vertex<DoubleTensor[]>`. I believe this is the major problem with the model; the output of this (box) is used to create the `BayesianNetwork` a little later on. Therefore if predict isn't correct and the `UnaryOpLambda` is not coded right, then the `BayesianNetwork` will also not be right. In his explanation he also said:
> If you're feeding the array of values into a single point of integration with MASON then you should probably go for the first [UnaryOpLambda] approach. Meaning, the UnaryOpLambda would call out to MASON and return its result.

Does this mean that the UnaryOpLambda should call out to MASONs `model.schedule.step()` as the operation? Or should MASONs step function be added to DataAssimilation::step (below)? If either of these are correct, then why in his reply to the issue does he say that before using the OpLambda wrapper, we should "// take 'state' and run the model forward x iterations"?

Here is the step function that is called inside the UnaryOpLambda wrapper. This does not behave how we might expect (or at least how I expected it to). I have exprimented with some counters and print statements in this method, some inside the loop and some after the loop and before the return statement (WINDOW_SIZE has always been set to 200 for clarity). What I noticed is that during `predict()`, `step()` is called thousands of times when I thought it should only be called once per predict. Also, when I added a counter inside the loop which should only be called 200 times, the counter was in the tens of thousands after only the second window.

```java

    private static DoubleTensor[] step(DoubleTensor[] initialState) {
        DoubleTensor[] steppedState = new DoubleTensor[initialState.length];

        for (int i=0; i < WINDOW_SIZE; i++) {
            steppedState = initialState; // actually update the state here
        }
        return steppedState;
    }
```

It might be useful to try the other approach that Caleb suggested in the github issue. He suggested having predict be an array of operations on an array of values. This means no complicated UnaryOpLambda and would allow us to use `DoubleVertex`'s throughout instead of `Vertex<DoubleTensor[]>` (which are horrible), but would most likely mean a large change to the current model. Instead of using the same `Person.class` in both twin models, we would need to replace Person in the tempModel with a new `AssimilationPerson.class`. This class would be very similar to Person.class, with these differences:

- x,y position (and current/desired speed?) to be replaced with DoubleVertex (or GaussianV)
- Must extend Agent.class, so when constructing AssimilationPerson could call super with new Double2D taken from the values of the vertices
  - e.g. constructor... super(size, new Double2D(x.scalar(), y.scalar()), name)
  - This allows MASON to visualise it properly whilst keeping the vertices
- Then use Keanu's algorithms where possible (as in x_position.plus(x_speed), y_pos.plus(y_speed)) to transform the state. This would be quite a large change to how the model works but could still keep the same functionality

##### Update the State Vector

This method collects information on the agents in the model at the current iteration and updates the state vector with this information. It is only called inside `predict()`, after each step of the `tempModel`. 

```java

    private static CombineDoubles updateStateVector(CombineDoubles stateVector) {
        List<Person> personList = new ArrayList<>(tempModel.area.getAllObjects());

        assert (!personList.isEmpty());
        assert (personList.size() == tempModel.getNumPeople()) : "personList size is wrong: " + personList.size();

        for (Person person : personList) {
            // get ID
            int pID = person.getID();
            int pIndex = pID * numVerticesPP;

            // Access DoubleVertex's directly from CombineDoubles class and update in place
            // This is much simpler than previous attempts and doesn't require creation of new stateVector w/ every iter
            stateVector.vertices[pIndex].setValue(person.getLocation().getX());
            stateVector.vertices[pIndex + 1].setValue(person.getLocation().getY());
            stateVector.vertices[pIndex + 2].setValue(person.getCurrentSpeed());
        }
        assert (stateVector.getLength() == tempModel.getNumPeople() * numVerticesPP);

        return stateVector;
    }

```

Each person in the model is collected into a list, which is then looped over. Similar to when initially creating the state vector, the persons ID is collected and an index created to access the 3 vertices corresponding to each person. The values of each vertex is then set to the x,y and current speed values, and after completion of the loop the updated state vector is returned. 

#### Observe some Truth Data

This is one of the two key steps for carrying out data assimilation in Keanu, alongside `predict()`. It is where the value of the distribution for each vertex in the vector is set according to the corresponding pseudotruth data. This is essentially the step where data (truth observations) is assimilated with the predictive model (BayesianNetwork, prior), to produce a posterior state distribution. When we call `observe()` on a vertex that is part of the `BayesianNetwork`, the value of the vertex is set to the value being observed (element of truth data) and the vertex is no longer a probabilistic distribution. Keanu can then 'cascade' that value through the whole network, calculating the most likely value or **posterior estimate** of each vertex. In basic terms, this is how Keanu combines the prior distribution with observations and calculates a posterior.  

```java

    private static CombineDoubles keanuObserve(Vertex<DoubleTensor[]> box, int windowNum, CombineDoubles stateVector) {
        System.out.println("\tObserving truth data. Adding noise with standard dev: " + SIGMA_NOISE);

        int historyIter = ((windowNum + 1) * WINDOW_SIZE); // find the correct iteration

        // Get history from correct iteration
        double[] history = truthHistory.get(historyIter);
        System.out.println("\tCollected history from iteration: " + (historyIter));

        // GET L1 Norm to check if BayesNet has updated
        System.out.println("\tL1 Norm before observe: " + getL1Norm(stateVector.calculate()));

        assert (history.length == tempModel.getNumPeople() * numVerticesPP) : String.format("History for iteration %d is incorrect length: %d",
                windowNum, history.length);

        // Output with a bit of noise. Lower sigma makes it more constrained
        GaussianVertex[] noisyOutput = new GaussianVertex[history.length];
        for (int i=0; i < history.length; i++) {
            if (history[i] == 0.0) {
                continue; // If vertex == 0.0, agent must be inactive so don't need to observe
            }
            // Add a bit of noise to each value in 'truth history'
            noisyOutput[i] = new GaussianVertex(history[i], SIGMA_NOISE);
            // Access each element of the box through the loop
            Vertex<DoubleTensor> element = CombineDoubles.getAtElement(i, box);
            // Observe the new noisyOutput value for element
            element.observe(noisyOutput[i].getValue());
            // Use the new observed element to set the value of the stateVector
            stateVector.vertices[i].setValue(element.getValue());
        }
        System.out.println("\tObservations complete.");

        updatePeople(stateVector);

        // Get L1 Norm again and compare to L1 Norm before obs
        System.out.println("\tL1 Norm after observe: " + getL1Norm(stateVector.calculate()));

        return stateVector;
    }
```

Before starting the loop, the method collects truth data from the correct iteration (which is calculated from the current assimilation window number and `WINDOW_SIZE`). Also, before and after applying the observations it calculates and prints the L1 norm. This was added just to assess how the observations were changing the state vector (or if they were changing at all). In its current form, there is always a fairly significant change in the L1 norm after observations, which leads me to believe that this method is not the main issue for the algorithm. 

Having said that, there are some differences in how we observe in stationSim compared to how we did it in NativeModel or the SimpleModels. In SimpleWrapperB (and C?) noise is added to the output of the model, not the truth data. i.e. here `noisyOutput[k]` is a `GaussianVertex` created with a mean of `history[k]` which is truth data. Should we instead create a Gaussian with an element of `box` (therefore adding noise to the output of box) before observing? Proper example below:

```java

        for (int i=0; i < history.length; i++) {
            if (history[i] == 0.0) {
                continue;
            }
            // Access each element of the box through in turn
            Vertex<DoubleTensor> element = CombineDoubles.getAtElement(i, box);

            // Add a bit of noise to each value in 'truth history'
            GaussianVertex noisyOutput = new GaussianVertex(element.getValue().scalar(), SIGMA_NOISE);
            
            noisyOutput.observe(history[i]);
            // Use the new observed element to set the value of the stateVector
            stateVector.vertices[i].setValue(noisyOutput.getValue());
        }
```

I have tested this, and the model runs with no issue **apart from the optimiser** (which is explained below).

Something to note, 

Finally, if we look back at NativeModel, the state observes the value of state for each iteration in the window (i.e. 1-200, 200-400 instead of just 200, 400 etc). This might be less realistic in terms of the model but would definitively say whether the observations were functioning correctly, could then rule out this as a problem point for the model. 

#### Create the BayesNet and Probabilistic Model

The `BayesianNetwork` is probably the most important aspect of this framework. The network represents the conditional relationship between all random (and unknown) variables in the model, and is the basis for all inference, sampling, and generally anything useful we want to do on our model with Keanu. In data assimilation terms, the `BayesianNetwork` represents the predictive model that provides our prior distribution. When we call `observe()` on vertices that are part of this network we set their value, allowing Keanu to cascade this value through the network and calculate the most likely values of each, determined by their conditional relationship. 

The BayesNet constructors require a collection of an object type that extends `Vertex`. We do this now using the output of the `UnaryOpLambda` explained earlier, but we achieve similar results when 

```java

            System.out.println("\tCreating BayesNet");
            // First create BayesNet, then create Probabilistic Model from BayesNet
            BayesianNetwork net = new BayesianNetwork(box.getConnectedGraph());
            // Create probabilistic model from BayesNet
            KeanuProbabilisticModel model = new KeanuProbabilisticModel(net);
```

#### Optimise

A good explanation of this process can be found [here](https://improbable-research.github.io/keanu/docs/inference-map/) in the keanu documentation. The `Optimizer` allows us to carry out inference with a method called Maximum A-Posteriori. As explained in the docs, MAP inference calculates the most probable values or **posterior estimates** of components in a model (vertices in a network) given certain condition or **observations**. This is Bayesian inference, and so is one of the ways we can carry out data assimilation BUT, it relies on the `BayesianNetwork` which has to be correct in order for it to function properly. The `Optimizer` class is an interface for a 2 different types of optimization, the Gradient Optimizer and Non-Gradient Optimizer. The gradient optimizer works through gradient descent (I believe), whilst the non-gradient optimizer uses the [BOBYQA algorithm](http://www.damtp.cam.ac.uk/user/na/NA_papers/NA2009_06.pdf) (Bound Optimization BY Quatratic Approximation). The interface can be accessed through a utility class, `Keanu`. This class holds 2 utility classes, one for optimization and one for sampling. 

```java

            // MAP optimizer allows us to calculate the most probable values of model components given certain conditions or 'observations'
            Optimizer optimizer = Keanu.Optimizer.of(net);
            optimizer.maxAPosteriori();
```

The optimiser (hate spelling it with a 'z') provides some evidence that the BayesNet is not quite right. When `.maxAPosteriori()` is called, Keanu carries out a fitness evaluation to check if the BayesNet is suitable for being optimised, and to decide which method (gradient/non-gradient) to use. The fitness evaluation calculates a Log Probability value (which I assume relates to the probability of the BayesNet, this isn't mentioned). The problem is that the LogProb evaluates to -Infinity from start to finish,which means the probability equals 0. To help explain what I think this means, here is Bayes theorem:

<img src="bayes-rule.png" alt="Bayes Theorem" width="420" height="300"/>

- P(A|B) = Probability of _A_ given observations _B_
- P(B|A) = Likelihood of the observations _B_ given that _A_ is true
- P(A)   = Prior probability that A is true
- P(B)   = Prior probability that the evidence itself is true

Here is the log version:

log(P(A|B)) = log(P(B|A)) + log(P(A)) ~~- log(P(B))~~

log(P(B)) can safely be ignored, as it is constant and not important at this stage (according to Dan Tang). So if the log probability of the model given the observations is -infinity, then the log likelihood and/or log prior must also be -infinity. Which in turn means one or both of the likelihood or prior probability is 0. I think it is most likely that the likelihood of the observations given the model is 0, which I believe is evidence that the BayesNet is not correct. 

HOWEVER (sorry), there have also been changes to the model that have caused the optimiser to break completely, not running at all. One example is with the modified observations shown above. In these cases the following error is produced:

`Exception in thread "main" java.lang.IllegalArgumentException: Cannot start optimizer on zero probability network`

This clearly means that the BayesNet is zero probability right from the start. Does this mean that it is not an incorrect BayesNet causing the LogProb to register as -infinity?

#### Sample From the Posterior

Creating the sampler is one of the more simple processes in Keanu, so I won't spend too long on it. As I understand it, sampling is where the model changes the value of individual elements iteratively, assessing whether the change in this value has made the observations more likely given the model. It repeats this process for a specified number of iterations (`NUM_SAMPLES`), keeping the change if the model better fits the observations. 

```java

            // Use Metropolis Hastings algo for sampling
            PosteriorSamplingAlgorithm samplingAlgorithm = Keanu.Sampling.MetropolisHastings.withDefaultConfig();

            // Create sampler from KeanuProbabilisticModel
            NetworkSamples sampler = samplingAlgorithm.getPosteriorSamples(
                    model,
                    model.getLatentVariables(),
                    NUM_SAMPLES
            );

            // Down sample and drop sample
            sampler = sampler.drop(DROP_SAMPLES).downSample(DOWN_SAMPLE);
```

The sampling algorithm is created by accessing the `Keanu` utility class. Metropolis Hastings is the least accurate method of sampling (other 2 are Hamiltonian Monte Carlo and NUTS (No-U-Turn-Sampler)), but is our preferred method because it is much less computationally expensive than the others. Sampling requires a lot of memory, especially as the number of assimilation windows increases. The sampler is created as shown, using the probabilistic model we created from the BayesNet. The final part here is dropping samples and downsampling. We use the parameters defined at the top of the class. Down sampling means that we only keep every x number of samples, and dropping samples relates to the burn in period. We drop the first x number of samples. 

#### Get the Information out of the Samples

Extracting information from the samples has proved difficult; I made this with the help of Caleb from Improbable. The stateVector is passed in as an argument so we can update the value of the vertices after sampling over them.

```java
    // Method call in main loop
    getSampleInfo(stateVector, net, sampler);

    // Method
    private static void getSampleInfo(CombineDoubles stateVector, BayesianNetwork net, NetworkSamples sampler) {
        // Get samples from each individual vertex
        for (int j=0; j < stateVector.getLength(); j++) {

            Vertex vert = net.getVertexByLabel(vertexLabelList.get(j));

            Samples<DoubleTensor> samples = sampler.get(vert);

            KDEVertex x = GaussianKDE.approximate(samples);

            //stateVector.vertices[j].setValue(x.getValue());
            stateVector.vertices[j].setAndCascade(x.getValue());
        }
        updatePeople(stateVector); // update people to reflect the observations
    }
```

Each vertex is accessed in turn from the BayesNet, using the `vertexLabelList` we created earlier. We then sample from the vertex using the `sampler`, and the resulting DoubleTensor samples can be used directly to approximate a `GaussianKDE` object. The `GaussianKDE` is not a simple Gaussian, and can capture much more information about the posterior distribution than one. The final step is to set the value of the corresponding vertex in the state vector. This can be done with either a `.setValue()` or `.setAndCascade()`. The only difference between these 2 methods (that I can see) is that when using `.setAndCascade()`, Keanu cascades the value to calculate posterior estimates for the remaining vertices in the network. Also, `.setAndCascade()` takes longer.

`updatePeople(stateVector)` is the final step to sampling (and the data assimilation window), and is where the posterior estimates for x,y position and speed are reintroduced as prior to the agents. The method follows:

```java

    private static void updatePeople(CombineDoubles stateVector) {
        assert (stateVector.getValue().length == tempModel.getNumPeople() * numVerticesPP) : "State Vector is incorrect length: " + stateVector.getValue().length;

        // Find all the people and add to list
        List<Person> initialPeopleList = new ArrayList<>(tempModel.area.getAllObjects());

        // Loop through peopleList created before model was stepped
        for (Person person : initialPeopleList) {
            // get person ID to find correct vertices in stateVector
            int pID = person.getID();
            int pIndex = pID * numVerticesPP;

            // TODO: Is this right? Do we want the DoubleTensor from CombineDoubles or do we want the DoubleVertex?
            // Extract DoubleTensor for each vertex using CombineDoubles.getAtElement()
            Vertex<DoubleTensor> xLocation = CombineDoubles.getAtElement(pIndex, stateVector);
            Vertex<DoubleTensor> yLocation = CombineDoubles.getAtElement(pIndex + 1, stateVector);
            Vertex<DoubleTensor> currentSpeed = CombineDoubles.getAtElement(pIndex + 2, stateVector);

            // Build Double2D
            Double2D loc = new Double2D(xLocation.getValue().scalar(), yLocation.getValue().scalar()); // Build Double2D for location
            // Get speed as double val
            double speedAsDouble = currentSpeed.getValue().scalar();

            // Set the location and current speed of the agent from the stateVector
            tempModel.area.setObjectLocation(person, loc);
            person.setCurrentSpeed(speedAsDouble);
        }
    }

```

This is similar to `updateStateVector` in shown earlier, just the other way around. Once again all the people in the model are collected into a list and looped over. The ID is collected and index calculated for each, then each vertex is accessed in turn with `CombineDoubles.getAtElement()`, passing the index and stateVector as arguments (this method will be explained in CombineDoubles_docs.md). A `Double2D` (MASON location object) is created from the x and y vertices, and the mode of the speed distribution taken as a double precision value. We then set the location and speed of each corresponding agent with these values. 

### End Loop

Two final things happen after the assimilation loop has finished. 

```java

        tempModel.finish();
        getMetrics();

```

`tempModel.finish()` is called to shut down tempModel processes, and `getMetrics()` is called to calculate vector norms and error for each. 

#### takeMetrics()

`...L1`				- arrays to store the L1 norm of truth and tempVector's at each iteration. [Vector Norms](https://machinelearningmastery.com/vector-norms-machine-learning/)

`...L2`				- arrays to story L2 norm of both vectors at each iteration (see link above)

`totalError`		- totalError between truth and temp vectors at each iteration

This method is responsible for calculating vector norms at each iteration, calculating the total error with each iter, then creating files and writing the data. I'm not going to go into detail with this one as the code is mostly self exaplanatory. The only confusing parts may be with the vector norms, so I included a link above to a page that explains them. Also, `totalError` refers to the sum of absolute errors between corresponding vertices in the truth and temp vectors. The code for that one might make it more clear:

```java

    private static double getError(double[] truthVector, DoubleTensor[] calculatedVec) {
        double[] sumVecValues = new double[calculatedVec.length];
        for (int j = 0; j < calculatedVec.length; j++) {
            sumVecValues[j] = calculatedVec[j].sum(); // call .sum() on each vertex to get a double precision value, store in array
        }

        double totalSumError = 0d;
        for (int k = 0; k < truthVector.length; k++) {
            totalSumError += abs(sumVecValues[k] - truthVector[k]); // take the absolute error between each vertex and add to cumulative total
        }
        return totalSumError;
    }

```

The number of people in each model at each iteration used to also be collected, but this did not show enough difference between the two models (or seem to react to changes in the code) so I removed it. It should be easy to add back in. 
