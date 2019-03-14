package StationSim;

import io.improbable.keanu.Keanu;
import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.PosteriorSamplingAlgorithm;
import io.improbable.keanu.algorithms.Samples;
import io.improbable.keanu.algorithms.particlefiltering.ParticleFilter;
import io.improbable.keanu.algorithms.particlefiltering.ParticleFilterBuilder;
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
import sim.util.Bag;
import sim.util.Double2D;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class DataAssimilation {

    // Create truth and temporary model
    private static Station truthModel = new Station(System.currentTimeMillis()); // Station model used to produce truth data
    private static Station tempModel = new Station(System.currentTimeMillis() + 1); // Station model used for state estimation

    private static int NUM_ITER = 5000; // Total number of iterations
    private static int WINDOW_SIZE = 200; // Number of iterations per update window
    private static int NUM_WINDOWS = NUM_ITER / WINDOW_SIZE; // Number of update windows

    private static int NUM_SAMPLES = 2000; // Number of samples to MCMC
    private static int DROP_SAMPLES = 1; // Burn-in period
    private static int DOWN_SAMPLE = 5; // Only keep every x sample

    private static double SIGMA_NOISE = 1.0; //

    // Initialise random var
    // random generator for start() method
    private static KeanuRandom rand = new KeanuRandom();

    // List of agent exits to use when rebuilding from stateVector
    //static DoubleVertex[][] stateVector;
    private static Exit[] agentExits;

    private static double[] tempResults = new double[NUM_ITER];
    private static double[] truthResults = new double[NUM_ITER];

    private static double[] truthVecTotals = new double[NUM_ITER];
    private static double[] tempVecTotals = new double[NUM_ITER];

    private static List<Double[]> truthHistory = new ArrayList<>();

    private static List<VertexLabel> tempVertexList = new ArrayList<>();

    private static CombineDoubles stateVector;

    /* Admin parameters */
    private static String dirName = "simulation_outputs/"; // Place to store results

    private static Writer numPeopleObsWriter;
    private static Writer vecTotalsObsWriter;


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
        createStateVector(tempModel);



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

            keanuObserve(box, i);

            /*
             ************ CREATE THE BAYES NET ************
             */

            System.out.println("\tCreating BayesNet");
            // First create BayesNet, then create Probabilistic Model from BayesNet
            BayesianNetwork net = new BayesianNetwork(box.getConnectedGraph());
            // Create probabilistic model from BayesNet
            KeanuProbabilisticModel model = new KeanuProbabilisticModel(net);


            // Attempting to build a particle filter using in built Keanu algorithms
            ParticleFilterBuilder partFilterBuilder = ParticleFilter.ofGraph(box.getConnectedGraph());

            ParticleFilter partFilter = new ParticleFilter(box.getConnectedGraph(),
                                                            100,
                                                            10,
                                                            0.5,
                                                            rand);

            /*
             ************ OPTIMISE ************
             */

            if (i > 0) {
                Optimizer optimizer = Keanu.Optimizer.of(net);
                optimizer.maxAPosteriori();
            }

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

            //List<DoubleVertex> KDE_List = new ArrayList<>();

            /*
            for (int j=0; j < stateVector.getLength(); j++) {
                // Get vertex from bayesNet by label (loop counter used to access elements of tempVertexList)
                Vertex vert = net.getVertexByLabel(tempVertexList.get(j));

                Samples<DoubleTensor> samples = sampler.get(vert);

                KDEVertex newKDE = GaussianKDE.approximate(samples);

                DoubleVertex x = newKDE;

                x.setLabel(tempVertexList.get(j));

                KDE_List.add(x);
            }

            stateVector = new CombineDoubles(KDE_List);
            */

            for (int j=0; j < stateVector.getLength(); j++) {

                // Get DoubleVertex from stateVector to update after sampling
                DoubleVertex vert = stateVector.vertices[j];

                // Get samples from vertex
                Samples<DoubleTensor> samples = sampler.get(vert);

                //samples.getMode();

                // Create new GaussianKDE from DoubleTensor samples
                KDEVertex newKDE = GaussianKDE.approximate(samples);

                if (j % 200 == 0.0) {
                    System.out.println("Vertex " + j + " value: " + vert.getValue(0));
                    System.out.println("KDEVert " + j + " value: " + newKDE.getValue() + "\n");
                }


                // Assign value of vertex in stateVector from newKDE
                stateVector.vertices[j].setValue(newKDE.getValue());
            }
        }
        tempModel.finish();

        writeObservations();
    }


    private static void writeObservations() throws IOException {

        System.out.println("Writing out observations...");

        // init files
        String theTime = String.valueOf(System.currentTimeMillis()); // So files have unique names

        numPeopleObsWriter = new BufferedWriter(new OutputStreamWriter(
                new FileOutputStream(dirName + "PeopleObs_" + theTime + ".csv"), "utf-8"));

        vecTotalsObsWriter = new BufferedWriter(new OutputStreamWriter(
                new FileOutputStream(dirName + "VectorObs_" + theTime + ".csv"), "utf-8"));

        // try catch for the
        try {
            numPeopleObsWriter.write("Iteration,truthObservation,tempObservation,\n");

            for (int i = 0; i < truthResults.length; i++) {
                numPeopleObsWriter.write(String.format("%d,%f,%f,\n", i, truthResults[i], tempResults[i]));
            }
        } catch (IOException ex) {
            System.err.println("Error writing numPeople to file: "+ ex.getMessage());
            ex.printStackTrace();
        }

        try {
            vecTotalsObsWriter.write("Iteration, truthVecTotal, tempVecTotal,\n");

            for (int i = 0; i < truthVecTotals.length; i++) {
                vecTotalsObsWriter.write(String.format("%d,%f,%f,\n", i, truthVecTotals[i], tempVecTotals[i]));
            }
        } catch (IOException ex) {
            System.err.println("Error writing vector totals to file: " + ex.getMessage());
            ex.printStackTrace();
        }

        System.out.println("Observations written to file");
    }


    private static void keanuObserve(Vertex<DoubleTensor[]> stateVec, int windowNum) {
        System.out.println("\tObserving truth data. Adding noise with standard dev: " + SIGMA_NOISE);

        // Get DoubleTensor[] output from stateVector
        DoubleTensor[] output = stateVec.getValue();
        assert (output.length == tempModel.getNumPeople() * 3) : "Output is incorrect length before observations";

        // Output with a bit of noise. Lower sigma makes it more constrained
        GaussianVertex[] noisyOutput = new GaussianVertex[output.length];
        for (int k=0; k < output.length; k++) {
            noisyOutput[k] = new GaussianVertex(output[k].scalar(), SIGMA_NOISE); // Take kth output and add Gaussian noise
        }

        // Observe the output (Could do this at same time as adding noise?)
        Double[] history = truthHistory.get(windowNum); // Get corresponding history from list for each update window
        assert (history.length == tempModel.getNumPeople() * 3) : String.format("History for iteration %d is incorrect length: %d",
                windowNum, history.length);

        // Loop through and observe
        for (int j=0; j < noisyOutput.length; j++) {
            noisyOutput[j].observe(history[j]);
        }
        System.out.println("\tObservations complete");
    }


    /**
     * THIS NEEDS TO BE REWRITTEN MORE CLEANLY : (DONE?)
     *
     * @param initialState
     * @return
     */
    private static Vertex<DoubleTensor[]> predict(Vertex<DoubleTensor[]> initialState) {
        System.out.println("\tPREDICTING...");

        // Check model contains agents
        assert (tempModel.area.getAllObjects().size() > 0) : "No agents in tempModel before prediction";

        // Find all the people and add to list
        Bag people = new Bag(tempModel.area.getAllObjects());
        List<Person> initialPeopleList = new ArrayList<>(people);

        // Update position and speed of agents from initialState
        updatePeople(initialState, initialPeopleList); // TODO: Is this step necessary? Do we even need to updatePeople before stepping? Any change from end state from previous window?

        // Run the model to make the prediction
        for (int i = 0; i < WINDOW_SIZE; i++) {
            // take observations of tempModel (num active people at each step)
            takeObservations(tempModel);
            // Get sum of values from each vertex in stateVector, to use as additional observation
            getStateVecTotal();
            // Step all the people window_size times
            tempModel.schedule.step(tempModel);
        }

        // Check that correct number of people in model before updating state vector
        assert (tempModel.area.getAllObjects().size() == tempModel.getNumPeople()) : String.format(
                "Wrong number of people in the model before building state vector. Expected %d but got %d",
                tempModel.getNumPeople(), tempModel.area.getAllObjects().size());

        // Get new stateVector after prediction
        List<Person> newPersonList = new ArrayList<>(tempModel.area.getAllObjects());
        updateStateVector(newPersonList);

        // Return OpLambda wrapper
        UnaryOpLambda<DoubleTensor[], DoubleTensor[]> box = new UnaryOpLambda<>(
                new long[0],
                stateVector,
                //(currentState) -> step(currentState)
                DataAssimilation::step // IntelliJ suggested replacing lambda with method reference here
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


    private static void getStateVecTotal() {

        // take stateVector of new model state
        List<Person> newPersonList = new ArrayList<>(tempModel.area.getAllObjects());
        updateStateVector(newPersonList);

        // get DoubleTensor values of each vertex
        DoubleTensor[] calculatedVec = stateVector.calculate();

        // total counter
        Double total = 0d;

        for (int i = 0; i < calculatedVec.length; i++) {
            // take scalar value of each DoubleTensor and add to total
            total += calculatedVec[i].scalar();
        }
        // Log total in results
        tempVecTotals[(int) tempModel.schedule.getSteps()] = total;
    }


    private static void updateStateVector(List<Person> personList) {
        //System.out.println("\tUPDATING STATE VECTOR...");
        assert (!personList.isEmpty());
        assert (personList.size() == tempModel.getNumPeople()) : personList.size();

        for (Person person : personList) {

            // get ID
            int pID = person.getID();
            int pIndex = pID * 3;

            // Access DoubleVertex's directly from CombineDoubles class and update in place
            // This is much simpler than previous attempts and doesn't require creation of new stateVector w/ every iter
            stateVector.vertices[pIndex].setValue(person.getLocation().x);
            stateVector.vertices[pIndex + 1].setValue(person.getLocation().y);
            stateVector.vertices[pIndex + 2].setValue(person.getCurrentSpeed());
        }
        assert (stateVector.getLength() == tempModel.getNumPeople() * 3);
        //System.out.println("\tFINISHED UPDATING STATE VECTOR.");
    }


    private static void updatePeople(Vertex<DoubleTensor[]> stateVector, List<Person> initialPeopleList) {
        System.out.println("\tREBUILDING PERSON LIST...");

        assert (stateVector.getValue().length == tempModel.getNumPeople() * 3) : "State Vector is incorrect length: " + stateVector.getValue().length;

        System.out.println("\tStateVector length: " + stateVector.getValue().length);

        int counter = 0;

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


    /**
     * Method to use when first creating the state vector for data assimilation model
     *
     * @param model The Station instance for which to create the state vector
     */
    private static void createStateVector(Station model) {
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
            // Init new GaussianVertices with mean 0.0 as DoubleVertex is abstract
            DoubleVertex xLoc = new GaussianVertex(0.0, SIGMA_NOISE);
            DoubleVertex yLoc = new GaussianVertex(0.0, SIGMA_NOISE);
            DoubleVertex desSpeed = new GaussianVertex(0.0, SIGMA_NOISE);

            // set unique labels for vertices
            xLoc.setLabel(lab0); yLoc.setLabel(lab1); desSpeed.setLabel(lab2);

            // Add labelled vertices to collection
            stateVertices.add(xLoc); stateVertices.add(yLoc); stateVertices.add(desSpeed);
        }
        // Instantiate stateVector with vertices
        stateVector = new CombineDoubles(stateVertices);
    }


    private static void takeObservations(Station model) {

        // take numPeople in model at each step for observations
        double numPeopleInModel = model.activeNum;

        if (model == truthModel) {
            truthResults[(int) model.schedule.getSteps()] = numPeopleInModel;
        } else if (model == tempModel) {
            tempResults[(int) model.schedule.getSteps()] = numPeopleInModel;
        }
    }


    /**
     * Function to run truthModel and extract state information at specific iterations
     */
    private static void getTruthData() {

        // start truthModel
        truthModel.start(rand);
        System.out.println("truthModel.start() has executed successfully");

        // Step truth model whilst step number is less than NUM_ITER
        while (truthModel.schedule.getSteps() < NUM_ITER) {

            // get truthVector at this iteration
            Double[] truthVector = getTruthVector(truthModel);
            double total = getTruthVecTotal(truthVector);
            truthVecTotals[(int)truthModel.schedule.getSteps()] = total;
            // Take observations of people in model at every step (including before the first step)
            takeObservations(truthModel);

            // Stop stepping and do something every 200 steps (every WINDOW_SIZE steps)
            if (truthModel.schedule.getSteps() % WINDOW_SIZE == 0.0) {
                // Add Double[] to truthHistory to be accessed later
                truthHistory.add(truthVector);
            }

            // Step
            truthModel.schedule.step(truthModel);
        }
        System.out.println("\tHave stepped truth model for: " + truthModel.schedule.getSteps() + " steps.");

        // Check correct number of truthVectors extracted
        assert (truthHistory.size() == NUM_WINDOWS) : "truthHistory collected at wrong number of iterations (" +
                                                        truthHistory.size() + "), should have been: " + NUM_WINDOWS;

        // As the model has finished everyone should be inactive
        //assert truthModel.inactivePeople.size() == truthModel.getNumPeople() : "All (" + truthModel.getNumPeople() +
          //                                                                      ") people should be inactive at end of " +
            //                                                                    "model, there are only: " + truthModel.inactivePeople.size();
        // Finish model
        truthModel.finish();
        System.out.println("Executed truthModel.finish()");
    }


    /**
     * Function to produce the truthVector. truthVector index is based on person ID, so positions in the truthVector
     *  remain fixed throughout the model.
     *
     * @param model Station model from which to save truth vector (should always be truthModel)
     */
    private static Double[] getTruthVector(Station model) {
        //System.out.println("\tBuilding Truth Vector...");

        // Get all objects from model area (all people) and initialise a list for sorting
        Bag people = model.area.getAllObjects();
        List<Person> truthPersonList = new ArrayList<>(people);

        assert (!truthPersonList.isEmpty()) : "truthPersonList is empty, there is nobody in the truthModel";
        assert (truthPersonList.size() == model.getNumPeople()) : "truthPersonList (" + truthPersonList.size() +
                                                                        ") is the wrong size, should be: " +
                                                                        model.getNumPeople();

        // Create new Double[] to hold truthVector (Each person has 3 vertices (or doubles in this case))
        Double[] truthVector = new Double[model.getNumPeople() * 3];

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


    private static double getTruthVecTotal(Double[] truthVector) {
        double total = 0;

        for (int i = 0; i < truthVector.length; i++) {
            total += truthVector[i];
        }
        return total;
    }


    public static void main(String[] args) throws IOException {
        runDataAssimilation();
        System.out.println("Data Assimilation finished! Excellent!");
    }















/*
    private static List<Person> updatePeople(Vertex<DoubleTensor[]> stateVector, List<Person> initialPeopleList) {
        System.out.println("REBUILDING PERSON LIST...");

        assert (stateVector.getValue().length == tempModel.getNumPeople() * 3) : "State Vector is incorrect length: " + stateVector.getValue().length;

        // list to hold the people
        List<Person> newPersonList = new ArrayList<>();

        // Loop through peopleList created before model was stepped
        for (Person person : initialPeopleList) {
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

            // Add person to newPersonList
            newPersonList.add(person);
        }
        assert (newPersonList.size() == tempModel.getNumPeople()) : "personList contains " + newPersonList.size() + " agents, this is wrong!";

        System.out.println("PERSON LIST REBUILT.");
        return newPersonList;
    }
*/
















}
