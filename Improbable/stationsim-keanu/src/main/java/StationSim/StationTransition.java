/* Created by Luke Archer on 16/12/2018.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

package StationSim;

import io.improbable.keanu.Keanu;
import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.PosteriorSamplingAlgorithm;
import io.improbable.keanu.algorithms.Samples;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.network.KeanuProbabilisticModel;
import io.improbable.keanu.network.NetworkState;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.operators.unary.UnaryOpLambda;
import sim.util.Bag;
import sim.util.Double2D;

import java.io.Writer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;


public class StationTransition {

    // Create truth model (and temp model)
    private static Station truthModel = new Station(System.currentTimeMillis()); // Station model used to produce truth data
    private static Station tempModel = new Station(System.currentTimeMillis() + 1); // Station model used for state estimation

    private static int NUM_ITER = 5000; // Total number of iterations
    private static int WINDOW_SIZE = 200; // Number of iterations per update window
    private static int NUM_WINDOWS = NUM_ITER / WINDOW_SIZE; // Number of update windows

    private static int NUM_SAMPLES = 2000; // Number of samples to MCMC
    private static int DROP_SAMPLES = 1; // Burn-in period
    private static int DOWN_SAMPLE = 5; // Only keep every x sample

    private static double SIGMA_NOISE = 1.0; //

    // Writing to file
    private static Writer stateWriter; // To write stateVectorHistory

    // Initialise random var
    // random generator for start() method
    private static KeanuRandom rand = new KeanuRandom();

    // List of agent exits to use when rebuilding from stateVector
    //static DoubleVertex[][] stateVector;
    private static Exit[] agentExits;

    private static double[] results;

    private static List<Vertex<DoubleTensor[]>> truthHistory = new ArrayList<>();

    private static List<VertexLabel> vertexLabelList = new ArrayList<>();


    private static void runDataAssimilation() {


        /*
         ************ CREATE THE TRUTH DATA ************
         */

        // Run the truth model
        truthModel.start(rand);
        System.out.println("truthModel.start() has executed successfully");

        int counter = 0;
        results = new double[(NUM_WINDOWS)];

        // Rewrote how truthModel is run as it allows easier extraction of truth data at intervals
        while (truthModel.schedule.getSteps() < NUM_ITER) {
            //if ((truthModel.schedule.getSteps() != 0) && (truthModel.schedule.getSteps() % WINDOW_SIZE == 0.0)) {;
            if (truthModel.schedule.getSteps() % WINDOW_SIZE == 0.0) {
                // Get numPeople output
                Bag truthPeople = truthModel.area.getAllObjects();
                results[counter] = truthPeople.size();

                // CHeck the number of people expected and the number of people in the model is correct.
                assert truthPeople.size() == truthModel.getNumPeople() :
                        String.format( "truthPeople: '%d', truthModel: %d",truthPeople.size(),truthModel.getNumPeople());

                // Build stateVector to observe (unless iter == 0)
                List<Person> truthList = new ArrayList<>(truthPeople);
                Vertex<DoubleTensor[]> truthVector = buildStateVector(truthList);

                // Add stateVector to history
                truthHistory.add(truthVector);
                System.out.println("\tSaving truthVector at step: " + truthModel.schedule.getSteps());
                // Increment counter
                counter++;
            }
            // Step while condition is not true
            truthModel.schedule.step(truthModel);
        }
        System.out.println("\tHave stepped truth model for: " + counter);
        // Finish model
        truthModel.finish();
        System.out.println("\tExecuted truthModel.finish()");

        // Ensure truthModel has successfully recorded history
        assert (!truthHistory.isEmpty()) : "truthHistory is empty, this is not right.";

        // As the model has finished everyone should be inactive
        assert truthModel.inactivePeople.size() == truthModel.getNumPeople() : String.format(
                "Everyone should be inactive but there are only '%d' inactive people: %s",
                    truthModel.inactivePeople.size(), truthModel.inactivePeople);

        /*
         ************ START THE MAIN LOOP ************
         */
        System.out.println("Starting DA loop");

        // Produce VertexLabels for each vertex in stateVector and add to static list
        // THIS IS NEW
        for (int p=0; p < tempModel.getNumPeople(); p++) {
            // second int variable to access each element of triplet vertices (3 vertices per agent)
            int pp = p * 3;

            // Create labels for each vertex
            VertexLabel lab0 = new VertexLabel("Person " + pp + 0);
            VertexLabel lab1 = new VertexLabel("Person " + pp + 1);
            VertexLabel lab2 = new VertexLabel("Person " + pp + 2);

            // Add labels to list for reference
            vertexLabelList.add(lab0);
            vertexLabelList.add(lab1);
            vertexLabelList.add(lab2);
        }


        // Create the current state (initially 0). Use the temporary model because so far it is in its initial state
        tempModel.start(rand);
        System.out.println("tempModel.start() has executed successfully");
        Vertex<DoubleTensor[]> currentStateEstimate = buildStateVector(tempModel);

        // Start data assimilation window
        for (int i = 0; i < NUM_WINDOWS; i++) {

            System.out.println("Entered Data Assimilation window " + i);

            /*
             ************ INITIALISE THE BLACK BOX MODEL ************
             */

            Vertex<DoubleTensor[]> box = predict(WINDOW_SIZE, currentStateEstimate);

            /*
             ************ OBSERVE SOME TRUTH DATA ************
             */

            // Observe the truth data plus some noise?
            System.out.println("Observing truth data. Adding noise with standard dev: " + SIGMA_NOISE);
            // Get output from the box model (200 iterations)
            DoubleTensor[] output = box.getValue();
            //System.out.println("Output length: " + output.length);

            // output with a bit of noise. Lower sigma makes it more constrained.;
            GaussianVertex[] noisyOutput = new GaussianVertex[output.length];
            for (int k=0; k < output.length; k++) {
                noisyOutput[k] = new GaussianVertex(output[k].scalar(), SIGMA_NOISE);
            }

            // Observe the output (Could do at same time as adding noise to each element in the state vector?)
            Vertex<DoubleTensor[]> history = truthHistory.get(i);
            DoubleTensor[] hist = history.getValue();
            //System.out.println("Hist length: " + hist.length);

            for (int f=0; f < noisyOutput.length; f++) {
                noisyOutput[f].observe(hist[f]); // Observe each element in the truth state vector
            }
            System.out.println("Observations complete");

            /*
             ************ CREATE THE BAYES NET ************
             */

            System.out.println("Creating BayesNet");
            // First create BayesNet, then create Probabilistic Model from BayesNet
            BayesianNetwork net = new BayesianNetwork(box.getConnectedGraph());
            // Create probabilistic model from BayesNet
            KeanuProbabilisticModel model = new KeanuProbabilisticModel(net);

            /*
             ************ OPTIMISE ************
             */

            //Optimizer optimizer = Keanu.Optimizer.of(net);
            //optimizer.maxAPosteriori();

            /*
             ************ SAMPLE FROM THE POSTERIOR************
             */

            System.out.println("Sampling");

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

            /*
            GaussianKDE class!!!
            More accurate approximation for posterior distribution

            Can instantiate a GaussianKDE directly from samples of a DoubleVertex
            HOWEVER if posterior distribution is simple (i.e. almost same as simple Gaussian) then much more simple and
            efficient to produce another Gaussian from mu and sigma (these can be calculated from a separate BayesNet?)


            Label samples in buildStateVector() and reference labels when sampling
            Need to loop through each vertex and sample individually
            Can then use the sampled distributions to reintroduce as Prior for next window

            Caleb talked about swapping the prior for the black box model each time instead of recreating. Not sure
            how this would look/work but he said it was possible and would be good for efficiency (also point below).

            Try to swap prior for black box model instead of creating at each iteration (efficiency improvement)
             */

            // for each DoubleTensor in Vertex<DoubleTensor[]> box
            for (DoubleTensor dubTense : box.getValue()) {
                // first sample from each DoubleTensor (What does each DoubleTensor represent? Need to be clear on this)
                // using net.getSamplesByLabel()
                // would I need to create array of VertexLabels as class variable? Then could loop through all these
                // in relatively efficient manner, list would be immutable.

                // Instantiate GaussianKDE object from sample
                // then collect into Vertex<> object?
                // public static KDEVertex approximate(Samples<DoubleTensor> vertexSamples) (constructor)

                // Reintroduce this Gaussian (or Vertex<>) as currentStateEstimate
                // (might need to cast these as DoubleVertex? i.e. DoubleVertex x = new GaussianKDE(vertSamples);)
            }

            // This is important! This is how we get individual vertices from their label
            // Vertex's are labelled when building the stateVector
            Vertex pop = net.getVertexByLabel(vertexLabelList.get(0));
            // This is a Vertex? can I call getValue() and get the DoubleTensor?
            // Calling .getValue() returns a Java object. Can I cast this to something else?






            //currentStateEstimate = meanStateVector;
        }
        tempModel.finish();
        //writeModelHistory(truthModel, "truthModel" + System.currentTimeMillis());
        //writeModelHistory(tempModel, "tempModel" + System.currentTimeMillis());
    }


    private static Vertex<DoubleTensor[]> buildStateVector(List<Person> personList) {
        System.out.println("\tBUILDING STATE VECTOR...");
        assert (!personList.isEmpty());
        assert (personList.size() == truthModel.getNumPeople()) : personList.size();

        // Create new collection to hold vertices for CombineDoubles
        List<DoubleVertex> stateVertices = new ArrayList<>();
        // Create array to hold exit list
        agentExits = new Exit[personList.size()];

        int counter = 0;
        for (Person person : personList) {

            // Init new GaussianVertices with mean 0.0 as DoubleVertex is abstract
            DoubleVertex xLoc = new GaussianVertex(0.0, SIGMA_NOISE);
            DoubleVertex yLoc = new GaussianVertex(0.0, SIGMA_NOISE);
            DoubleVertex desSpeed = new GaussianVertex(0.0, SIGMA_NOISE);

            // Set value of Vertices to location and speed values of person agents
            xLoc.setValue(person.getLocation().x);
            yLoc.setValue(person.getLocation().y);
            desSpeed.setValue(person.getDesiredSpeed());

            // Set labels of vertices dependant on counter
            // '... + 1', '... + 2' etc. is to ensure unique labels when building the BayesNet
            // Uniqueness is enforced on instantiation of the BayesNet
            xLoc.setLabel(new VertexLabel("Person " + counter + 1));
            yLoc.setLabel(new VertexLabel("Person " + counter + 2));
            desSpeed.setLabel(new VertexLabel("Person " + counter + 3));

            // Add labelled vertices to collection
            stateVertices.add(xLoc);
            stateVertices.add(yLoc);
            stateVertices.add(desSpeed);

            // Add agent exit to list for rebuilding later
            agentExits[counter] = person.getExit();

            counter++;
        }
        // Ensure exit array is same length as stateVertices / 3 (as 3 vertices per agent)
        assert (agentExits.length == stateVertices.size() / 3);
        assert (stateVertices.size() == tempModel.getNumPeople() * 3) : "State Vector is incorrect size: " + stateVertices.size();
        System.out.println("\tSTATE VECTOR BUILT");
        return new CombineDoubles(stateVertices);
    }


    private static Vertex<DoubleTensor[]> buildStateVector(Bag people) {
        List<Person> personList = new ArrayList<>(people);
        return buildStateVector(personList);
    }


    private static Vertex<DoubleTensor[]> buildStateVector(Station model) {
        return buildStateVector(model.area.getAllObjects());
    }


    private static List<Person> rebuildPersonList(Vertex<DoubleTensor[]> stateVector) {

        System.out.println("REBUILDING PERSON LIST...");

        assert(stateVector.getValue().length == tempModel.getNumPeople() * 3) : "State Vector is incorrect length: " + stateVector.getValue().length;

        // list to hold the people
        List<Person> personList = new ArrayList<>();

        // initialise iterator so call to .next() gets new agent with every iteration
        Iterator<Person> inactivePeeps = tempModel.inactivePeople.iterator();

        // Find halfway point so equal number of agents assigned to entrance 1 and 2 (doesn't affect model run)
        int halfwayPoint = tempModel.getNumPeople() / 2;
        int entranceNum = 0;

        int inactiveCounter = 0;

        for (int i = 0; i < tempModel.getNumPeople(); i++) {
            // j is equal to ith lot of 3s (this is because each agent is represented by three vertices in stateVector)
            int j = i * 3;

            // Extract vertex's from stateVector
            Vertex<DoubleTensor> xLoc = CombineDoubles.getAtElement(j, stateVector);
            Vertex<DoubleTensor> yLoc = CombineDoubles.getAtElement(j+1, stateVector);
            Vertex<DoubleTensor> desSpeed = CombineDoubles.getAtElement(j+2, stateVector);

            // TODO: Make range from -SIGMA_NOISE to SIGMA_NOISE?? (So noisy observations don't break this check)
            // if x value is 0.0 agent is inactive
            if (yLoc.getValue().scalar() == 0.0) {
                Person inactive = inactivePeeps.next();
                personList.add(inactive);
                inactiveCounter++;
                //tempModel.inactivePeople.remove(inactive);
                continue; // inactive agent added, no need to go through rest
            }

            // Build Double2D
            Double2D loc = new Double2D(xLoc.getValue().scalar(), yLoc.getValue().scalar()); // Build Double2D for location
            // Get speed as double val
            double speedAsDouble = desSpeed.getValue().scalar();

            // Get exit from agentExits[]
            Exit exit = agentExits[i];

            // Change entranceNum if we have created more than half the people
            if (i > halfwayPoint) {
                entranceNum = 1;
            }

            // Get entrances from tempModel, select entrance based on entranceNum
            List<Entrance> entrances = tempModel.getEntrances();
            Entrance entrance = entrances.get(entranceNum);

            // TODO: Make this name equal the right name (id?) for agents so we can follow one agent through model more easily
            String name = "Person: " + i;

            // Create person using location and speed from stateVector
            Person person = new Person(tempModel.getPersonSize(), loc, name, tempModel, exit, entrance, speedAsDouble);

            // add person to personList
            personList.add(person);
        }
        assert(personList.size() == tempModel.getNumPeople()) : "personList contains " + personList.size() + " agents, this is wrong!";

        System.out.println("PERSON LIST BUILT SUCCESSFULLY");
        return personList;
    }


    private static boolean withinRange(double[] range, double value) {
        assert(range.length == 2) : "Range has too many values in array";
        Arrays.sort(range);
        if (value > range[0] && value < range[1]) {
            return true;
        }
        return false;
    }


    private static Vertex<DoubleTensor[]> predict(int stepCount, Vertex<DoubleTensor[]> initialState) {
        System.out.println("PREDICTING...");
        // Find all the people
        Bag people = new Bag(tempModel.area.getAllObjects());

        assert(tempModel.area.getAllObjects().size() != 0);
        List<Person> initialPeopleList = new ArrayList<>(people);

        for (Person person : initialPeopleList) {
            tempModel.area.remove(person);
        }

        assert(tempModel.area.getAllObjects().size() == 0);

        // Convert from the initial state ( a list of numbers) to a list of agents
        List<Person> personList = new ArrayList<>(rebuildPersonList(initialState)); // FIXED NOW (this will break it (needs to convert initialState -> PersonList))

        assert(personList.size() == 700) : "personList is not 700 agents exactly";

        // Add people from the personList
        for (Object o : personList) {
            Person p = (Person) o;
            tempModel.area.setObjectLocation(p, p.getLocation());
        }

        assert (tempModel.area.getAllObjects().size() > 0);

        for (int i = 0; i < WINDOW_SIZE; i++) {
            // Step all the people window_size times
            tempModel.schedule.step(tempModel);
        }

        assert(tempModel.area.getAllObjects().size() == truthModel.getNumPeople()) : String.format(
                "Wrong number of people in the model before building state vector. Expected %d but got %d",
                truthModel.getNumPeople(), tempModel.area.getAllObjects().size() );

        // Get new stateVector
        Vertex<DoubleTensor[]> state = buildStateVector(tempModel.area.getAllObjects()); // build stateVector directly from Bag people? YES

        UnaryOpLambda<DoubleTensor[], DoubleTensor[]> box = new UnaryOpLambda<>(
                new long[0],
                state,
                (currentState) -> step(stepCount, currentState)
        );

        System.out.println("PREDICTION FINISHED.");
        return box;
    }


    private static DoubleTensor[] step(int stepCount, DoubleTensor[] initialState) {

        DoubleTensor[] steppedState = new DoubleTensor[initialState.length];

        for (int i=0; i < stepCount; i++) {
            steppedState = initialState; // actually update the state here
        }
        return steppedState;
    }


    private static List<Double[]> getDoubleSamples(List<DoubleTensor[]> tensorSamples) {

        List<Double[]> doubleSamples = new ArrayList<>();

        for (DoubleTensor[] sample : tensorSamples) {
            Double[] sampleResults = new Double[sample.length];
            for (int k = 0; k < sample.length; k++) {
                sampleResults[k] = sample[k].getValue(0);
            }
            doubleSamples.add(sampleResults);
        }
        return doubleSamples;
    }


    private static Double[] getStateVectorMean(List<Double[]> doubleSamples) {

        // var to hold number of samples
        int sampleSize = doubleSamples.get(0).length;

        // New list to hold means of each element in stateVector
        Double[] sampleMeans = new Double[sampleSize];

        for (int i=0; i < sampleSize; i++) {
            double indexTotal = 0;
            for (Double[] vectorSample : doubleSamples) {
                indexTotal += vectorSample[i];
            }
            sampleMeans[i] = indexTotal / sampleSize;
        }
        return sampleMeans;
    }


    public static void main(String[] args) {

        StationTransition.runDataAssimilation();
        System.out.println("Happy days, runDataAssimilation has executed successfully!");

        System.exit(0);
    }


    private static void writeModelHistory(Station model, String fileName) {
        model.analysis.writeDataFrame(model.analysis.stateDataFrame, fileName + ".csv");
    }
}