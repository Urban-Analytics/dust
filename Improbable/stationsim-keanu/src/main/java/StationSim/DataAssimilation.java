package StationSim;

import io.improbable.keanu.Keanu;
import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.PosteriorSamplingAlgorithm;
import io.improbable.keanu.algorithms.Samples;
import io.improbable.keanu.algorithms.variational.GaussianKDE;
import io.improbable.keanu.algorithms.variational.optimizer.Optimizer;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.network.KeanuProbabilisticModel;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.KDEVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.operators.unary.UnaryOpLambda;

import sim.util.Double2D;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

import static java.lang.Math.abs;
import static java.lang.Math.pow;
import static java.lang.Math.sqrt;

public class DataAssimilation {

    // Create truth and temporary model
    private static Station truthModel = new Station(System.currentTimeMillis()); // Station model used to produce truth data
    private static Station tempModel = new Station(System.currentTimeMillis() + 1); // Station model used for state estimation

    private static int NUM_ITER = 2000; // Total number of iterations
    private static int WINDOW_SIZE = 200; // Number of iterations per update window
    private static int NUM_WINDOWS = NUM_ITER / WINDOW_SIZE; // Number of update windows

    private static int NUM_SAMPLES = 2000; // Number of samples to MCMC
    private static int DROP_SAMPLES = 1; // Burn-in period
    private static int DOWN_SAMPLE = 5; // Only keep every x sample

    private static double SIGMA_NOISE = 1.0;

    private static int numVerticesPP = 3; // Number of vectors per person (x_pos, y_pos, speed)

    // Initialise random var
    // random generator for start() method
    private static KeanuRandom rand = new KeanuRandom();

    private static List<VertexLabel> vertexLabelList = new ArrayList<>(); // List of VertexLabels to help in selecting specific vertices from BayesNet

    /* Observations */
    private static List<double[]> truthHistory = new ArrayList<>(); // List of double arrays to store the truthVector at every iteration
    private static List<DoubleTensor[]> tempHistory = new ArrayList<>(); // List of DoubleTensor arrays to store the tempModel state vector at each iteration


    /**
     * Main function of model. Contains the data assimilation workflow. Namely:
     *
     *  * Runs truthModel for X iter, and collects state information at end of each window
     *  * Start Main loop
     *      * Initialise the Black Box model
     *      * Observe the truth data
     *      * Create the BayesNet and Probabilistic Model
     *     (* Optimise??)
     *      * Sample from the Posterior
     *      * Get information on posterior distribution from samples
     *      * Reintroduce posterior as prior for next window
     *
     */
    private static void runDataAssimilation() throws IOException {

        // ************ CREATE THE TRUTH DATA ************

        getTruthData(); // Run truth model, collect truth vector at each iter and store in truthHistory

        // ************ START THE MAIN LOOP ************

        System.out.println("Starting DA loop");

        // Start tempModel and create the initial state
        tempModel.start(rand);
        System.out.println("tempModel.start() has executed successfully");

        CombineDoubles stateVector = createStateVector(tempModel); // initial state vector

        // Start data assimilation window
        for (int i = 0; i < NUM_WINDOWS; i++) {
            System.out.println("Entered Data Assimilation window " + i);

            // ************ INITIALISE THE BLACK BOX MODEL ************

            Vertex<DoubleTensor[]> box = predict(stateVector);

            // If this is the last window, we don't need to observe or sample anything as they won't show up in the graphs
            // We only need to run predict to show the tempModel to the end, to compare the model after the last observation
            // with truthModel
            if (i == NUM_WINDOWS - 1) {
                continue; // skip the last run and finish the model/take observations
            }

            // ************ OBSERVE SOME TRUTH DATA ************

            stateVector = keanuObserve(box, i, stateVector);

            // ************ CREATE THE BAYES NET ************

            System.out.println("\tCreating BayesNet");
            // First create BayesNet, then create Probabilistic Model from BayesNet
            BayesianNetwork net = new BayesianNetwork(box.getConnectedGraph());
            // Create probabilistic model from BayesNet
            KeanuProbabilisticModel model = new KeanuProbabilisticModel(net);

            // ************ OPTIMISE ************

            // MAP optimizer allows us to calculate the most probable values of model components given certain conditions
            // or 'observations'
            Optimizer optimizer = Keanu.Optimizer.of(net);
            optimizer.maxAPosteriori();

            // ************ SAMPLE FROM THE POSTERIOR************

            System.out.println("\tSAMPLING");

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

            assert(sampler != null) : "Sampler is null, this is not good";

            // ************ GET THE INFORMATION OUT OF THE SAMPLES ************

            System.out.println("\tEXTRACTING INFO FROM SAMPLES...");
            getSampleInfo(stateVector, net, sampler);
        }
        tempModel.finish();
        getMetrics();
    }


    // ***************************** TRUTH DATA *****************************


    /**
     * This method runs the truth model, collecting a truth vector at each iteration. The truth vector consists
     * of 3 double precision values for each agent; an x and y position and desired speed.
     */
    private static void getTruthData() {
        // start truthModel
        truthModel.start(rand);
        System.out.println("truthModel.start() has executed successfully");

        // Step truth model whilst step number is less than NUM_ITER
        while (truthModel.schedule.getSteps() < NUM_ITER) { // +1 to account for an error on final run, trying to access iteration 2000 of history
            double[] truthVector = getTruthVector(); // get truthVector for observations
            truthHistory.add(truthVector); // Save for later
            truthModel.schedule.step(truthModel); // Step
        }
        System.out.println("\tHave stepped truth model for: " + truthModel.schedule.getSteps() + " steps.");

        // Check correct number of truthVectors extracted
        assert (truthHistory.size() == NUM_ITER) : "truthHistory collected at wrong number of iterations (" +
                truthHistory.size() + "), should have been: " + NUM_ITER;

        // Finish model
        truthModel.finish();
        System.out.println("Executed truthModel.finish()");
    }


    /**
     * Method to produce the truthVector, called inside getTruthData(). truthVector index is based on person ID,
     * so positions in the truthVector remain fixed throughout the model. This is important when the truth vector data
     * is observed by tempModel.
     *
     * @return  a vector of x,y coordinates and speed of all agents in truthModel
     */
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


    // ***************************** STATE VECTOR *****************************


    /**
     * Method to use when first creating the state vector for data assimilation model. It creates a stateVector of
     * the x position, y position (and previously desired speed) for each agent (so length of stateVector is 2 times the number of
     * agents in the model). The method then creates VertexLabels for each vertex, based on an agents ID
     * (i.e. for agent 15 vertex 2, the label would be: "Person 151" - Person + ID + vertNum(0-1)).
     * Each label is added to the corresponding vertex, and when all vertices are produced and held in a collection,
     * a CombineDoubles object is created from this collection and returned as the original stateVector
     *
     * @param model     The Station instance for which to create the state vector
     * @return          New CombineDoubles object from the stateVertices
     */
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
            vertexLabelList.add(lab0);
            vertexLabelList.add(lab1);
            vertexLabelList.add(lab2);

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


    // ***************************** PREDICT *****************************


    /**
     * This is where the temporary Model is stepped for WINDOW_SIZE iterations, collecting the stateVector for each
     * iteration. Observations are taken before each step, and the stateVector is updated after each with the new agent
     * positions. A UnaryOpLambda model wrapper object is used to update the state using the output of the predictive
     * model, that allows creation of the Bayesian Network from its output.
     *
     * @param initialState  The stateVector of tempModel, to update with a new prediction
     * @return              The output of the UnaryOpLambda, used to build the BayesianNetwork
     */
    private static Vertex<DoubleTensor[]> predict(CombineDoubles initialState) {
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
        assert (tempModel.area.getAllObjects().size() == tempModel.getNumPeople()) : String.format(
                "Wrong number of people in the model before building state vector. Expected %d but got %d",
                tempModel.getNumPeople(), tempModel.area.getAllObjects().size());

        // Return OpLambda wrapper, calling step() function below multiple times
        UnaryOpLambda<DoubleTensor[], DoubleTensor[]> box = new UnaryOpLambda<>(
                new long[0],
                stateVector,
                DataAssimilation::step
        );
        return box;
    }


    /**
     * This method is provided as the operation that the UnaryOpLambda needs to transform the input vertex.
     * For each element of the stateVector, the UnaryOpLambda loops through and assigns the value of that element
     * to a different state object. This process of mapping the vector after each step using the UnaryOpLambda is
     * crucial as the resulting object will be used to build the Bayesian Network.
     *
     * @param initialState  The state vector of the initial state to be 'stepped'
     * @return              The initial state at every iteration through WINDOW_SIZE? This one is confusing.
     */
    private static DoubleTensor[] step(DoubleTensor[] initialState) {
        DoubleTensor[] steppedState = new DoubleTensor[initialState.length];

        for (int i=0; i < WINDOW_SIZE; i++) {
            steppedState = initialState; // actually update the state here
        }
        return steppedState;
    }


    // ***************************** OBSERVE *****************************


    /**
     * Observe is a key function here, and is where vectors in the Bayesian Network are observed using data from
     * the digital twin. First the truth history from the correct iteration is retrieved, and the L1 norm is
     * checked to assess both before and after the vector is observed. Noise is added to each truth observation in a
     * loop, which then collects the correct element of the stateVector and observes the truth observation on the
     * distribution. The observed element is then used to set the value of the corresponding vertex inside
     * the CombineDoubles object.
     *
     * @param box           The output of UnaryOpLambda, the black box model
     * @param windowNum     Current window number
     * @param stateVector   vector of x,y positions and speed for each agent
     * @return              An observed CombineDoubles state vector, for use when building the BayesianNetwork
     */
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


    // ***************************** UPDATE *****************************


    /**
     * This method loops through all the agents in the model via their ID number, and updates the corresponding
     * vertices in the state vector with new x,y positions. Previously speed was also included.
     *
     * @param stateVector   Current vector of agent coordinates to be updated
     * @return              Updated vector with new x,y positions
     */
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


    /**
     * Method to update the positions of the people from the stateVector. A Double2D object is created from the x,y
     * positions of each agent in the vector. This Double2D is then used to update the position of the agent to fit
     * with the stateVector.
     *
     * @param stateVector   Vector of x,y positions
     */
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


    // ***************************** GET SAMPLE INFO *****************************


    private static void getSampleInfo(CombineDoubles stateVector, BayesianNetwork net, NetworkSamples sampler) {
        // Get samples from each individual vertex
        for (int j=0; j < stateVector.getLength(); j++) {

            Vertex vert = net.getVertexByLabel(vertexLabelList.get(j));

            Samples<DoubleTensor> samples = sampler.get(vert);

            //DoubleVertex x = GaussianKDE.approximate(samples);
            KDEVertex x = GaussianKDE.approximate(samples);

            //stateVector.vertices[j].setValue(x.getValue());
            stateVector.vertices[j].setAndCascade(x.getValue());
        }
        updatePeople(stateVector); // update people to reflect the observations
    }


    // ***************************** TAKE OBSERVATIONS *****************************


    /**
     * This method collects all model observations into one place, first producing them from the vectors created for
     * each iteration. Observations taken are L1 norm, L2 norm and total error between vectors.
     *
     * @throws IOException      IntelliJ recommended throwing this exception due to opening files for writing
     */
    private static void getMetrics() throws IOException {
        assert(truthHistory.size() == tempHistory.size()) : "truthHistory and tempHistory are not the same size, " +
                "truthHistory: " + truthHistory.size() + ", tempHistory: " + tempHistory.size();

        double[] truthL1 = new double[NUM_ITER];
        double[] tempL1 = new double[NUM_ITER];

        double[] truthL2 = new double[NUM_ITER];
        double[] tempL2 = new double[NUM_ITER];

        double[] totalError = new double[NUM_ITER];

        for (int i = 0; i < NUM_ITER; i++) {
            double[] truthVec = truthHistory.get(i);
            DoubleTensor[] tempVec = tempHistory.get(i);

            truthL1[i] = getL1Norm(truthVec);
            truthL2[i] = getL2Norm(truthVec);

            tempL1[i] = getL1Norm(tempVec);
            tempL2[i] = getL2Norm(tempVec);

            totalError[i] = getError(truthVec, tempVec);
        }
        System.out.println("\tWriting out observations...");
        System.out.println("L1Obs truth length: " + truthL1.length);
        System.out.println("L1Obs temp length: " + tempL1.length);
        System.out.println("L2Obs truth length: " + truthL2.length);
        System.out.println("L2Obs temp length: " + tempL2.length);
        System.out.println("Error length: " + totalError.length);

        /* INIT FILES */
        String theTime = String.valueOf(System.currentTimeMillis()); // So files have unique names
        String dirName = "simulation_outputs/"; // Place to store results

        Writer L1ObsWriter = new BufferedWriter(new OutputStreamWriter(
                    new FileOutputStream(dirName + "L1Obs_" + theTime + ".csv"), "utf-8"));
        Writer L2ObsWriter = new BufferedWriter(new OutputStreamWriter(
                    new FileOutputStream(dirName + "L2Obs_" + theTime + ".csv"), "utf-8"));
        Writer errorWriter = new BufferedWriter(new OutputStreamWriter(
                    new FileOutputStream(dirName + "ErrorObs_" + theTime + ".csv"), "utf-8"));


        // try catch for the writing
        try {
            L1ObsWriter.write("Iteration, truth, temp\n");
            for (int j = 0; j < truthL1.length; j++) {
                L1ObsWriter.write(String.format("%d,%f,%f\n", j, truthL1[j], tempL1[j]));
            }
            L1ObsWriter.close();

            L2ObsWriter.write("Iteration, truth, temp\n");
            for (int k = 0; k < truthL2.length; k++) {
                L2ObsWriter.write(String.format("%d,%f,%f\n", k, truthL2[k], tempL2[k]));
            }
            L2ObsWriter.close();

            errorWriter.write("Iteration,error\n");
            for (int l = 0; l < totalError.length; l++) {
                errorWriter.write(String.format("%d,%f\n", l, totalError[l]));
            }
            errorWriter.close();
        } catch (IOException ex) {
            System.err.println("Error writing observations to file: "+ ex.getMessage());
            ex.printStackTrace();
        }
        System.out.println("\tObservations written to file");
    }


    /**
     * L1 Norm is the sum of absolute values of the vector. This is used as an observation to assess data assimilation.
     * Each absolute value of the vector is added to a rolling total, which is then returned.
     *
     * @param vector
     * @return
     */
    private static double getL1Norm(double[] vector) {
        // Calculate sum of the vector for this iteration (L1 Norm)
        double sumTotal = 0d;
        for (double aVector : vector) {
            sumTotal += abs(aVector);
        }
        return sumTotal;
    }

    /**
     * Overloaded from method above to allow DoubleTensor[] vector.
     *
     * @param calculatedVec     calculated form of CombineDoubles stateVector (see CombineDoubles.calculate() for info)
     * @return                  calls getL1Norm() with a double[] produced from calculatedVec
     */
    private static double getL1Norm(DoubleTensor[] calculatedVec) {
        double[] vector = new double[calculatedVec.length];
        for (int i = 0; i < calculatedVec.length; i++) {
            vector[i] = calculatedVec[i].scalar();
        }
        return getL1Norm(vector);
    }

    /**
     * L2 Norm is the square root of the sum of squared vector values. Used as an observation.
     * Each value in the vector is squared and added to a rolling total. This square root of this total
     * is returned.
     *
     * @param truthVector   The vector from the 'truth' model.
     * @return              The square root of the sum of squared values.
     */
    private static double getL2Norm(double[] truthVector) {
        double squaredTotal = 0d;
        for (double aTruthVector : truthVector) {
            squaredTotal += pow(aTruthVector, 2);
        }
        return sqrt(squaredTotal);
    }

    /**
     * Overloaded from method above to allow DoubleTensor[] vector.
     *
     * @param calculatedVec     calculated form of CombineDoubles stateVector
     * @return                  calls getL2Norm() with a double[] produced from calculatedVec
     */
    private static double getL2Norm(DoubleTensor[] calculatedVec) {
        double[] vector = new double[calculatedVec.length];
        for (int i = 0; i < calculatedVec.length; i++) {
            vector[i] = calculatedVec[i].scalar();
        }
        return getL2Norm(vector);
    }

    /**
     * Get the absolute error (difference) between each vertex at each element.
     *
     * @return
     */
    private static double getError(double[] truthVector, DoubleTensor[] calculatedVec) {
        double[] sumVecValues = new double[calculatedVec.length];
        for (int j = 0; j < calculatedVec.length; j++) {
            sumVecValues[j] = calculatedVec[j].sum(); // call .sum() on each vertex to get a double precision value
        }

        double totalSumError = 0d;
        for (int k = 0; k < truthVector.length; k++) {
            totalSumError += abs(sumVecValues[k] - truthVector[k]); // take the absolute error between each vertex and add to cumulative total
        }
        return totalSumError;
    }


    // ***************************** MAIN *****************************


    public static void main(String[] args) throws IOException {
        runDataAssimilation();
        System.out.println("Data Assimilation finished! Excellent!");
    }
}
