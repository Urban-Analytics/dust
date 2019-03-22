package StationSim;

import io.improbable.keanu.Keanu;
import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.PosteriorSamplingAlgorithm;
import io.improbable.keanu.algorithms.Samples;
import io.improbable.keanu.algorithms.variational.GaussianKDE;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.network.KeanuProbabilisticModel;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.KDEVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.operators.unary.UnaryOpLambda;
import sim.util.Bag;
import sim.util.Double2D;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import static java.lang.Math.abs;
import static java.lang.Math.pow;
import static java.lang.Math.sqrt;

public class DataAssimilation {

    // Create truth and temporary model
    private static Station truthModel = new Station(System.currentTimeMillis()); // Station model used to produce truth data
    private static Station tempModel = new Station(System.currentTimeMillis() + 1); // Station model used for state estimation

    private static int NUM_ITER = 1000; // Total number of iterations
    private static int WINDOW_SIZE = 200; // Number of iterations per update window
    private static int NUM_WINDOWS = NUM_ITER / WINDOW_SIZE; // Number of update windows

    private static int NUM_SAMPLES = 2000; // Number of samples to MCMC
    private static int DROP_SAMPLES = 1; // Burn-in period
    private static int DOWN_SAMPLE = 5; // Only keep every x sample

    private static double SIGMA_NOISE = 2.0; //

    // Initialise random var
    // random generator for start() method
    private static KeanuRandom rand = new KeanuRandom();

    // List of agent exits to use when rebuilding from stateVector
    //static DoubleVertex[][] stateVector;
    //private static Exit[] agentExits;

    /* Observations */
    private static double[] tempNumPeople = new double[NUM_ITER];
    private static double[] truthNumPeople = new double[NUM_ITER];

    private static double[] truthL1 = new double[NUM_ITER];
    private static double[] tempL1 = new double[NUM_ITER];

    private static double[] truthL2 = new double[NUM_ITER];
    private static double[] tempL2 = new double[NUM_ITER];

    private static double[] totalError = new double[NUM_ITER];


    private static List<double[]> truthHistory = new ArrayList<>();

    private static List<VertexLabel> tempVertexList = new ArrayList<>();

    //private static CombineDoubles stateVector;


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
     *      * Get information out of the samples
     *
     */
    private static void runDataAssimilation() throws IOException {

        /*
         ************ CREATE THE TRUTH DATA ************
         */

        getTruthData();

        /*
         ************ START THE MAIN LOOP ************
         */

        System.out.println("Starting DA loop");

        // Start tempModel and create the initial state
        tempModel.start(rand);
        System.out.println("tempModel.start() has executed successfully");
        // initial state vector
        CombineDoubles stateVector = createStateVector(tempModel);

        // Start data assimilation window
        for (int i = 0; i < NUM_WINDOWS; i++) {

            System.out.println("Entered Data Assimilation window " + i);

            /*
             ************ INITIALISE THE BLACK BOX MODEL ************
             */

            Vertex<DoubleTensor[]> box = predict(stateVector);

            //CombineDoubles box = predict(stateVector);

            /*
             ************ OBSERVE SOME TRUTH DATA ************
             */

            DoubleTensor[] calculatedVec = stateVector.calculate();

            /* NEW IDEA: Do we need to observe using box? Box runs the model and produces output but
             * doesn't necessarily mean that box has to be observed, wouldn't stateVector need to be observed?
              * But then what would box do? What does predict() do?*/

            stateVector = keanuObserve(box, i, stateVector);

            /*
             ************ CREATE THE BAYES NET ************
             */

            System.out.println("\tCreating BayesNet");
            // First create BayesNet, then create Probabilistic Model from BayesNet
            BayesianNetwork net = new BayesianNetwork(stateVector.getConnectedGraph());
            // Create probabilistic model from BayesNet
            KeanuProbabilisticModel model = new KeanuProbabilisticModel(net);

            /*
             ************ OPTIMISE ************
             */

            /*
            if (i > 0) {
                Optimizer optimizer = Keanu.Optimizer.of(net);
                optimizer.maxAPosteriori();
            }
            */

            /*
             ************ SAMPLE FROM THE POSTERIOR************
             */

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

            /*
             ************ GET THE INFORMATION OUT OF THE SAMPLES ************
             */

            System.out.println("\tEXTRACTING INFO FROM SAMPLES...");

            /*
            GaussianKDE class!!!
            More accurate approximation for posterior distribution

            Can instantiate a GaussianKDE directly from samples of a DoubleVertex
            HOWEVER if posterior distribution is simple (i.e. almost same as simple Gaussian) then much more simple and
            efficient to produce another Gaussian from mu and sigma (these can be calculated from a separate BayesNet?)


            Label samples in updateStateVector() and reference labels when sampling
            Need to loop through each vertex and sample individually
            Can then use the sampled distributions to reintroduce as Prior for next window

            Caleb talked about swapping the prior for the black box model each time instead of recreating. Not sure
            how this would look/work but he said it was possible and would be good for efficiency (also point below).

            Try to swap prior for black box model instead of creating at each iteration (efficiency improvement)
             */

            List<KDEVertex> KDE_List = new ArrayList<>();

            // Get samples from each individual vertex
            for (int j=0; j < stateVector.getLength(); j++) {

                Vertex vert = net.getVertexByLabel(tempVertexList.get(j));

                Samples<DoubleTensor> samples = sampler.get(vert);

                //DoubleVertex x = GaussianKDE.approximate(samples);

                KDEVertex x = GaussianKDE.approximate(samples);

                KDE_List.add(x);
            }

            for (int k=0; k < stateVector.getLength(); k++) {
                // Think .setAndCascade() is important here? Used in Lorenz example to update priors
                stateVector.vertices[k].setAndCascade(KDE_List.get(k).getValue());
            }

            updatePeople(stateVector);

        }
        tempModel.finish();

        writeObservations();
    }


    // ***************************** TRUTH DATA *****************************

    /**
     * This is the main function for running the pseudo-truth model and collecting truth data to observe.
     *
     * What this function does:
     *  - Calls start to set up the truth model
     *  - Runs the truth model for NUM_ITER steps
     *      - Builds a truth vector from the model at every iteration
     *      - Adds the total of this vector to a double array to write to file later
     *      - Every 200 iterations (beginning at 0), saves the full truth vector to a list to observe in keanuObserve()
     *  - Finally, calls truthModel.finish() to stop the model
     *
     */
    private static void getTruthData() {

        // start truthModel
        truthModel.start(rand);
        System.out.println("truthModel.start() has executed successfully");

        // Step truth model whilst step number is less than NUM_ITER
        while (truthModel.schedule.getSteps() < NUM_ITER) {
            // get truthVector for observations
            double[] truthVector = getTruthVector();
            // Take observations of people in model at every step (including before the first step)
            takeTruthObservations(truthVector);
            // Step
            truthModel.schedule.step(truthModel);
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
     * Function to produce the truthVector. truthVector index is based on person ID, so positions in the truthVector
     *  remain fixed throughout the model.
     */
    private static double[] getTruthVector() {
        //System.out.println("\tBuilding Truth Vector...");

        // Get all objects from model area (all people) and initialise a list for sorting
        Bag people = truthModel.area.getAllObjects();
        List<Person> truthPersonList = new ArrayList<>(people);

        assert (!truthPersonList.isEmpty()) : "truthPersonList is empty, there is nobody in the truthModel";
        assert (truthPersonList.size() == truthModel.getNumPeople()) : "truthPersonList (" + truthPersonList.size() +
                ") is the wrong size, should be: " +
                truthModel.getNumPeople();

        // Create new double[] to hold truthVector (Each person has 3 vertices (or doubles in this case))
        double[] truthVector = new double[truthModel.getNumPeople() * 3];

        for (Person person : truthPersonList) {

            // Get persons ID to create an index for truth vector
            int pID = person.getID();
            int index = pID * 3;

            // Each person relates to 3 variables in truthVector
            truthVector[index] = person.getLocation().x;
            truthVector[index + 1] = person.getLocation().y;
            // TODO: Try this with desired speed to see the difference (When finished and running experiments)
            truthVector[index + 2] = person.getCurrentSpeed();
        }
        // NEED ASSERTION HERE?

        //System.out.println("\tTruth Vector Built at iteration: " + truthModel.schedule.getSteps());

        return truthVector;
    }


    // ***************************** STATE VECTOR *****************************


    /**
     * Method to use when first creating the state vector for data assimilation model
     *
     * @param model The Station instance for which to create the state vector
     */
    private static CombineDoubles createStateVector(Station model) {
        // TODO: With agent locations now being updated (not created and destroyed), can we do this without speed? With only location vars? (i.e. do Keanu's algo's need speed?)

        // Create new collection to hold vertices for CombineDoubles
        List<DoubleVertex> stateVertices = new ArrayList<>();

        // Get all objects from model area (all people) and initialise a list for sorting
        Bag people = model.area.getAllObjects();
        List<Person> tempPersonList = new ArrayList<>(people);

        // loop through numPeople
        //for (int pNum=0; pNum < model.getNumPeople(); pNum++) {
        for (Person person : tempPersonList) {
            // second int variable to access each element of triplet vertices (3 vertices per agent)
            //int index = pNum * 3;

            // Get person ID for labelling
            int pID = person.getID();

            // Produce VertexLabels for each vertex in stateVector and add to static list
            // Labels are named based on person ID, i.e. "Person 0(pID) 0(vertNumber), Person 01, Person 02, Person 10" etc.
            VertexLabel lab0 = new VertexLabel("Person " + pID + 0);
            VertexLabel lab1 = new VertexLabel("Person " + pID + 1);
            VertexLabel lab2 = new VertexLabel("Person " + pID + 2);

            // Add labels to list for reference
            tempVertexList.add(lab0);
            tempVertexList.add(lab1);
            tempVertexList.add(lab2);

            // Now build the initial state vector
            // Init new GaussianVertices as DoubleVertex is abstract
            // mean = 0.0, sigma = 0.0
            DoubleVertex xLoc = new GaussianVertex(0.0, 0.0);
            DoubleVertex yLoc = new GaussianVertex(0.0, 0.0);
            DoubleVertex desSpeed = new GaussianVertex(0.0, 0.0);

            // set unique labels for vertices
            xLoc.setLabel(lab0); yLoc.setLabel(lab1); desSpeed.setLabel(lab2);

            // Add labelled vertices to collection
            stateVertices.add(xLoc); stateVertices.add(yLoc); stateVertices.add(desSpeed);
        }
        // Instantiate stateVector with vertices
        //stateVector = new CombineDoubles(stateVertices);

        return new CombineDoubles(stateVertices);
    }


    // ***************************** PREDICT *****************************


    /**
     * THIS NEEDS TO BE REWRITTEN MORE CLEANLY : (DONE?)
     *
     * @param initialState
     * @return
     */
    private static Vertex<DoubleTensor[]> predict(CombineDoubles initialState) {
        System.out.println("\tPREDICTING...");

        // Check model contains agents
        assert (tempModel.area.getAllObjects().size() > 0) : "No agents in tempModel before prediction";

        // Update position and speed of agents from initialState
        updatePeople(initialState); // TODO: Is this step necessary? Do we even need to updatePeople before stepping? Any change from end state from previous window?

        // Run the model to make the prediction
        for (int i = 0; i < WINDOW_SIZE; i++) {
            initialState = updateStateVector(initialState);
            // take observations of tempModel (num active people at each step)
            takeTempObservations(initialState);
            // Step all the people window_size times
            tempModel.schedule.step(tempModel);
        }

        // Check that correct number of people in model before updating state vector
        assert (tempModel.area.getAllObjects().size() == tempModel.getNumPeople()) : String.format(
                "Wrong number of people in the model before building state vector. Expected %d but got %d",
                tempModel.getNumPeople(), tempModel.area.getAllObjects().size());

        CombineDoubles stateVector = updateStateVector(initialState);

        // Return OpLambda wrapper
        UnaryOpLambda<DoubleTensor[], DoubleTensor[]> box = new UnaryOpLambda<>(
                new long[0],
                stateVector,
                (currentState) -> step(currentState)
                //DataAssimilation::step // IntelliJ suggested replacing lambda with method reference here
        );

        System.out.println("\tPREDICTION FINISHED.");
        return box;
    }


    private static DoubleTensor[] step(DoubleTensor[] initialState) {

        DoubleTensor[] steppedState = new DoubleTensor[initialState.length];

        for (int i=0; i < WINDOW_SIZE; i++) {
            steppedState = initialState; // actually update the state here
        }
        return steppedState;
    }


    // ***************************** OBSERVE *****************************


    private static CombineDoubles keanuObserve(Vertex<DoubleTensor[]> box, int windowNum, CombineDoubles stateVector) {
        System.out.println("\tObserving truth data. Adding noise with standard dev: " + SIGMA_NOISE);

        double[] history = truthHistory.get(windowNum * WINDOW_SIZE); // Get history from correct iteration
        System.out.println("Collected history from iteration: " + (windowNum * WINDOW_SIZE));

        // GET L1 Norm to check if BayesNet has updated

        assert (history.length == tempModel.getNumPeople() * 3) : String.format("History for iteration %d is incorrect length: %d",
                windowNum, history.length);

        // Output with a bit of noise. Lower sigma makes it more constrained
        GaussianVertex[] noisyOutput = new GaussianVertex[history.length];
        for (int i=0; i < history.length; i++) {
            noisyOutput[i] = new GaussianVertex(history[i], SIGMA_NOISE);

            CombineDoubles.getAtElement(i, box).observe(noisyOutput[i].getValue());
        }
        System.out.println("Observations complete.");

        // Get L1 Norm again and compare to L1 Norm before obs

        return box;
    }


    // ***************************** UPDATE *****************************


    private static CombineDoubles updateStateVector(CombineDoubles stateVector) {
        //System.out.println("\tUPDATING STATE VECTOR...");

        List<Person> personList = new ArrayList<>(tempModel.area.getAllObjects());

        assert (!personList.isEmpty());
        assert (personList.size() == tempModel.getNumPeople()) : personList.size();

        for (Person person : personList) {

            // get ID
            int pID = person.getID();
            int pIndex = pID * 3;

            //getAtElement -> get Vertex<DoubleTensor>  (Can only .setValue() on these with DoubleTensor
            //.getValue() -> get DoubleTensor           (Can .setValue() with double)
            //.setValue() -> set DoubleTensor value     (Don't know if this DoubleTensor is a reference or a new object)
            /*
            CombineDoubles.getAtElement(pIndex, stateVector).getValue().setValue(person.getLocation().getX());
            CombineDoubles.getAtElement(pID + 1, stateVector).getValue().setValue(person.getLocation().getY());
            CombineDoubles.getAtElement(pID + 2, stateVector).getValue().setValue(person.getCurrentSpeed());
            */

            // Access DoubleVertex's directly from CombineDoubles class and update in place
            // This is much simpler than previous attempts and doesn't require creation of new stateVector w/ every iter
            /*
            stateVector.vertices[pIndex].setAndCascade(person.getLocation().getX());
            stateVector.vertices[pIndex + 1].setAndCascade(person.getLocation().getY());
            stateVector.vertices[pIndex + 2].setAndCascade(person.getCurrentSpeed());
            */
            stateVector.vertices[pIndex].setValue(person.getLocation().getX());
            stateVector.vertices[pIndex + 1].setValue(person.getLocation().getY());
            stateVector.vertices[pIndex + 2].setValue(person.getCurrentSpeed());

        }
        assert (stateVector.getLength() == tempModel.getNumPeople() * 3);
        //System.out.println("\tFINISHED UPDATING STATE VECTOR.");

        return stateVector;
    }


    private static void updatePeople(Vertex<DoubleTensor[]> stateVector) {
        //System.out.println("\tUPDATING PERSON LIST...");

        assert (stateVector.getValue().length == tempModel.getNumPeople() * 3) : "State Vector is incorrect length: " + stateVector.getValue().length;

        //System.out.println("\tStateVector length: " + stateVector.getValue().length);

        // Find all the people and add to list
        Bag people = new Bag(tempModel.area.getAllObjects());
        List<Person> initialPeopleList = new ArrayList<>(people);

        // Loop through peopleList created before model was stepped
        for (Person person : initialPeopleList) {

            //System.out.println(person.toString());

            // get person ID to find correct vertices in stateVector
            int pID = person.getID();
            int pIndex = pID * 3;

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
            person.setCurrentSpeed(speedAsDouble); // TODO: CAREFUL HERE, THIS COULD BE WRONG
        }
    }


    // ***************************** TAKE OBSERVATIONS *****************************


    private static void takeTruthObservations(double[] truthVector) {
        // take numPeople in model at each step for observations
        double numPeopleInModel = truthModel.activeNum;
        int stepNum = (int) truthModel.schedule.getSteps();

        truthNumPeople[stepNum] = numPeopleInModel; // record numPeople in model
        //double[] truthVector = getTruthVector(); // get truthVector of current model state

        truthHistory.add(truthVector); // Add Double[] to truthHistory to be accessed later

        // **** get L1 Norm **** (sum of absolute values of the vector)
        double L1Norm = getL1Norm(truthVector);
        truthL1[stepNum] = L1Norm; // save L1 Norm for later

        // **** get L2 Norm **** (square root of the sum of squared vector values)
        double L2Norm = getL2Norm(truthVector);
        truthL2[stepNum] = L2Norm;
    }


    private static void takeTempObservations(CombineDoubles stateVector) {

        // take numPeople in model at each step for observations
        double numPeopleInModel = tempModel.activeNum;
        int stepNum = (int) tempModel.schedule.getSteps();

        tempNumPeople[stepNum] = numPeopleInModel; // record numPeople

        // get DoubleTensor values of each vertex
        DoubleTensor[] calculatedVec = stateVector.calculate();

        // **** get L1 norm ****
        double L1Norm = getL1Norm(calculatedVec);
        tempL1[stepNum] = L1Norm;

        // **** get L2 norm ****
        double L2Norm = getL2Norm(calculatedVec);
        tempL2[stepNum] = L2Norm;

        // get truthVector from the correct timestep
        double[] truthVector = truthHistory.get(stepNum);
        double error = getTotalError(truthVector, calculatedVec);
        totalError[stepNum] = error;
    }

    /**
     * L1 Norm is the sum of absolute values of the vector. This is used as an observation to assess data assimilation.
     * Each absolute value of the vector is added to a rolling total, which is then returned.
     *
     * @param truthVector
     * @return
     */
    private static double getL1Norm(double[] truthVector) {
        // Calculate sum of the truthVector for this iteration (L1 Norm)
        double sumTotal = 0d;
        for (int i = 0; i < truthVector.length; i++) {
            sumTotal += abs(truthVector[i]);
        }
        return sumTotal;
    }

    private static double getL1Norm(DoubleTensor[] calculatedVec) {
        // total counter
        double sumTotal = 0d;
        for (DoubleTensor aTensor : calculatedVec) {
            sumTotal += abs(aTensor.sum()); // take sum of each DoubleTensor and add to total
        }
        return sumTotal;
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
        for (int i = 0; i < truthVector.length; i++) {
            squaredTotal += pow(truthVector[i], 2);
        }
        return sqrt(squaredTotal);
    }

    private static double getL2Norm(DoubleTensor[] calculatedVec) {
        double squaredTotal = 0d;
        for (DoubleTensor aTensor : calculatedVec) {
            squaredTotal += pow(aTensor.sum(), 2);
        }
        return sqrt(squaredTotal);
    }

    /**
     * Get the absolute error (difference) between each vertex at each element
     * @return
     */
    private static double getTotalError(double[] truthVector, DoubleTensor[] calculatedVec) {
        double[] sumVecValues = new double[calculatedVec.length];

        for (int i = 0; i < calculatedVec.length; i++) {
            sumVecValues[i] = calculatedVec[i].sum();
        }

        double totalSumError = 0d;

        for (int j = 0; j < truthVector.length; j++) {
            totalSumError += abs(truthVector[j] - sumVecValues[j]);
        }

        //double averageError = totalSumError / totalSumError.length;

        return totalSumError;
    }


    private static void writeObservations() throws IOException {

        System.out.println("\tWriting out observations...");

        System.out.println("truthNumPeople length: " + truthNumPeople.length);
        System.out.println("tempNumPeople length: " + tempNumPeople.length);

        System.out.println("L1Obs truth length: " + truthL1.length);
        System.out.println("L1Obs temp length: " + tempL1.length);

        System.out.println("L2Obs truth length: " + truthL2.length);
        System.out.println("L2Obs temp length: " + tempL2.length);

        System.out.println("Error length: " + totalError.length);

        System.out.println("truthHistory length: " + truthHistory);

        /* INIT FILES */
        String theTime = String.valueOf(System.currentTimeMillis()); // So files have unique names
        // Place to store results
        String dirName = "simulation_outputs/";

        Writer numPeopleObsWriter = new BufferedWriter(new OutputStreamWriter(
                new FileOutputStream(dirName + "PeopleObs_" + theTime + ".csv"), "utf-8"));
        Writer L1ObsWriter = new BufferedWriter(new OutputStreamWriter(
                new FileOutputStream(dirName + "L1Obs_" + theTime + ".csv"), "utf-8"));
        Writer L2ObsWriter = new BufferedWriter(new OutputStreamWriter(
                new FileOutputStream(dirName + "L2Obs_" + theTime + ".csv"), "utf-8"));
        Writer errorWriter = new BufferedWriter(new OutputStreamWriter(
                new FileOutputStream(dirName + "ErrorObs_" + theTime + ".csv"), "utf-8"));

        // try catch for the writing
        try {
            numPeopleObsWriter.write("Iteration,truth,temp,\n");
            for (int i = 0; i < truthNumPeople.length; i++) {
                numPeopleObsWriter.write(String.format("%d,%f,%f,\n", i, truthNumPeople[i], tempNumPeople[i]));
            }
            numPeopleObsWriter.close();

            L1ObsWriter.write("Iteration, truth, temp,\n");
            for (int j = 0; j < truthL1.length; j++) {
                L1ObsWriter.write(String.format("%d,%f,%f,\n", j, truthL1[j], tempL1[j]));
            }
            L1ObsWriter.close();

            L2ObsWriter.write("Iteration, truth, temp,\n");
            for (int k = 0; k < truthL2.length; k++) {
                L2ObsWriter.write(String.format("%d,%f,%f,\n", k, truthL2[k], tempL2[k]));
            }
            L2ObsWriter.close();

            errorWriter.write("Iteration,error,\n");
            for (int l = 0; l < totalError.length; l++) {
                errorWriter.write(String.format("%d,%f,\n", l, totalError[l]));
            }
            errorWriter.close();

        } catch (IOException ex) {
            System.err.println("Error writing observations to file: "+ ex.getMessage());
            ex.printStackTrace();
        }
        System.out.println("\tObservations written to file");
    }


    // ***************************** MAIN *****************************


    public static void main(String[] args) throws IOException {
        runDataAssimilation();
        System.out.println("Data Assimilation finished! Excellent!");
    }
}
