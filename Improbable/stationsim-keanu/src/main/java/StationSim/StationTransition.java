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
import io.improbable.keanu.algorithms.*;
import io.improbable.keanu.network.KeanuProbabilisticModel;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.operators.unary.UnaryOpLambda;

import sim.util.Bag;
import sim.util.Double2D;

import java.io.Writer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


public class StationTransition {

    // Create truth model (and temp model)
    private static Station truthModel = new Station(System.currentTimeMillis()); // Station model used to produce truth data
    private static Station tempModel = new Station(System.currentTimeMillis() + 1); // Station model used for state estimation

    private static int NUM_ITER = 2000; // Total number of iterations
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


    /**
     * Still have error in predict() somewhere.
     * When predict is called, the model has 700 agents in it (tempModel.area.getAllObjects())
     *
     * Predict pseudocode:
     * Predict first removes 700 agents (successfully)
     * Then builds a list of 700 agents from the state vector (successfully)
     * It then runs (steps) the model (containing 700 agents) forward by 200 iterations (PROBLEM HERE)
     * Returns new UnaryOpLambda
     *
     *      PROBLEM: predict causes the number of agents in the model to reduce. Usually only one person in the model
     *      by 2nd or 3rd window. This is caught by an AssertionError in predict but assertions have to be turned on
     *      in IntelliJ with -ea flag (https://stackoverflow.com/questions/18168257/where-to-add-compiler-options-like-ea-in-intellij-idea)
     *
     *      Original problem was that the Set of inactiveAgents was not referenced when building personList,
     *      so when agents were removed from inactiveAgents they were not being removed from the model. This has now
     *      been fixed (to a point) where personList checks if agent is inactive and adds inactive agent if true.
     *      Explanation of how is in rebuildPersonList.
     *
     *      I'm in presentation training from 2-4:30pm today so won't be contactable then
     */


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
                        String.format( "truthPeople: '%d', truthModel: %d",truthPeople.size(),truthModel.getNumPeople() );

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
        assert (!truthHistory.isEmpty());

        // As the model has finished everyone should be inactive
        assert truthModel.inactivePeople.size() == 0 : String.format(
                "There should be no inactive people, not $d",truthModel.inactivePeople);

        /*
         ************ START THE MAIN LOOP ************
         */
        System.out.println("Starting DA loop");


        // Create the current state (initially 0). Use the temporary model because so far it is in its initial state
        tempModel.start(rand);
        //tempModel.schedule.step(tempModel);
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
            //System.out.println("Observing truth data. Adding noise with standard dev: " + SIGMA_NOISE);
            /*
            System.out.println("Observing at iterations: ");
            for (Integer j = 0; j < WINDOW_SIZE + WINDOW_SIZE; j++) {
                System.out.print(j + ",");

                // Need to get the output from the box model
                //DoubleTensor[] output = ((CombineDoubles) box).calculate();
                DoubleTensor[] output = box.getValue();

                // output with a bit of noise. Lower sigma makes it more constrained.
                GaussianVertex noisyOutput[] = new GaussianVertex[state.getValue().length];
                for (int k = 0; k < state.getValue().length; k++) {
                    noisyOutput[k] = new GaussianVertex(output[k].scalar(), SIGMA_NOISE);
                }

                // Observe the output (could do at same time as adding noise to each element in the state vector?)
                Vertex<DoubleTensor[]> history = truthHistory.get(i); // This is the truth state at the end of window i
                DoubleTensor[] hist = history.getValue();
                for (int k = 0; k < state.getValue().length; k++) {
                    noisyOutput[k].observe(hist[k]); // Observe each element in the truth state vector
                }
            }
            System.out.println();
            System.out.println("Observations complete");
            */


            // Observe the truth data plus some noise?
            System.out.println("Observing truth data. Adding noise with standard dev: " + SIGMA_NOISE);
            // Get output from the box model (200 iterations)
            DoubleTensor[] output = box.getValue();
            System.out.println("Output length: " + output.length);

            // output with a bit of noise. Lower sigma makes it more constrained.;
            GaussianVertex[] noisyOutput = new GaussianVertex[output.length];
            for (int k=0; k < output.length; k++) {
                noisyOutput[k] = new GaussianVertex(output[k].scalar(), SIGMA_NOISE);
            }

            // Observe the output (Could do at same time as adding noise to each element in the state vector?)
            Vertex<DoubleTensor[]> history = truthHistory.get(i);
            DoubleTensor[] hist = history.getValue();
            System.out.println("Hist length: " + hist.length);

            for (int f=0; f < noisyOutput.length; f++) {
                noisyOutput[f].observe(hist[f]); // Observe each element in the truth state vector
            }
            System.out.println("Observations complete");

            /*
             ************ CREATE THE BAYES NET ************
             */

            System.out.println("Creating BayesNet");
            KeanuProbabilisticModel model = new KeanuProbabilisticModel(box.getConnectedGraph());
            //BayesianNetwork net = new BayesianNetwork(box.getConnectedGraph());

            /*
             ************ OPTIMISE ************
             */

            //Optimizer optimizer = Keanu.Optimizer.of(net);
            //optimizer.maxAPosteriori();

            /*
             ************ SAMPLE FROM THE POSTERIOR************
             */

            System.out.println("Sampling");

            PosteriorSamplingAlgorithm samplingAlgorithm = Keanu.Sampling.MetropolisHastings.withDefaultConfig();

            samplingAlgorithm.getPosteriorSamples(
                    model,
                    model.getLatentOrObservedVertices(),
                    NUM_SAMPLES
            );

            /*
            NetworkSamples sampler = Keanu.Sampling.MetropolisHastings.withDefaultConfig().getPosteriorSamples(
                    model,
                    model.getLatentOrObservedVertices(),
                    NUM_SAMPLES
            );
            */

            // Down sample and drop sample
            //sampler = sampler.drop(DROP_SAMPLES).downSample(DOWN_SAMPLE);

            /*
             ************ GET THE INFORMATION OUT OF THE SAMPLES ************
             */

            //List<DoubleTensor> stateSamples = sampler.getDoubleTensorSamples(box).asList();
            //Samples stateSamples = sampler.get(box);

            //List samp = stateSamples.asList();

            //System.out.println("Sample size: " + samp.size());
            //System.out.println(samp);

            // *** Model state ***
            /*List<DoubleTensor[]> stateSamplesTensor = sampler.get(box).asList();

            List<Double[]> stateSamplesDouble = getDoubleSamples(stateSamplesTensor);

            Double[] meanStateVector = getStateVectorMean(stateSamplesDouble);
            */
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

        List<Person> personList = new ArrayList<>();
        // Find halfway point so equal number of agents assigned entrance 1 and 2 (doesn't affect model run)
        int halfwayPoint = tempModel.getNumPeople() / 2; // Divide by 2 to get half number of agents
        int entranceNum = 0;

        for (int i = 0; i < tempModel.getNumPeople(); i++) {
            // j is equal to ith lot of 3s
            int j = i * 3;

            // Extract vertex's from stateVector
            Vertex<DoubleTensor> xLoc = CombineDoubles.getAtElement(j, stateVector);
            Vertex<DoubleTensor> yLoc = CombineDoubles.getAtElement(j+1, stateVector);
            Vertex<DoubleTensor> desSpeed = CombineDoubles.getAtElement(j+2, stateVector);

            /**
             * This is where we check to see if agent (from stateVector) is inactive.
             * Inactive agents have desiredSpeed of 0 (or between -1:1,-1:1 if noise has been added in observations)
             * If vertex is for inactive agent, add inactiveAgent from set and continue
             */
            // TODO: Make range from -SIGMA_NOISE to SIGMA_NOISE?? (So noisy observations don't break this check)
            // if x value is 0.0 agent is inactive
            if (withinRange(new double[]{-1,1}, desSpeed.getValue().scalar())) {
                Person inactive = tempModel.inactivePeople.iterator().next();
                personList.add(inactive);
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
        assert(personList.size() == 700) : "personList contains " + personList.size() + " agents, this is wrong!";
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
        System.out.println("\tBag people size: " + people.size());

        assert(tempModel.area.getAllObjects().size() != 0);
        List<Person> personList1 = new ArrayList<>(people);

        for (Person person : personList1) {
            tempModel.area.remove(person);
        }

        System.out.println("\tThere are " + tempModel.area.getAllObjects().size() + " people in the model.");
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

        System.out.println("\tThere are " + tempModel.area.getAllObjects().size() + " people in the model before stepping.");

        /**
         * Error is in .step() function below. As model is stepped, new agents should be added and inactive ones
         * removed from the model.
         * HOWEVER, each step causes quite considerable number of agents to be lost from the model.
         *
         */

        // Propagate the model
        System.out.println("\tStepping...");
        System.out.println("\tPROBLEM HERE");
        System.out.println("Number of inactive agents: " + tempModel.inactivePeople.size());
        for (int i = 0; i < WINDOW_SIZE; i++) {
            // Step all the people window_size times
            tempModel.schedule.step(tempModel);
            //System.out.println("Number of inactive agents: " + tempModel.inactivePeople.size());
        }
        System.out.println("Number of inactive agents: " + tempModel.inactivePeople.size());
        System.out.println("\tThere are " + tempModel.area.getAllObjects().size() + " people in the model after stepping.");

        /**
         * BELOW IS THE ASSERTION ERROR DUE TO WRONG NUMBER OF AGENTS, STACK TRACE WRONGLY POINTS TO FOR LOOP ABOVE
         */
        assert(tempModel.area.getAllObjects().size() == truthModel.getNumPeople()) : String.format(
                "Wrong number of people in the model before building state vector. Expected %d but got %d",
                truthModel.getNumPeople(), tempModel.area.getAllObjects().size() );
        // Get new stateVector
        //List<Person> personList2 = new ArrayList<>(tempModel.area.getAllObjects());
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


    private static void update(Tensor<DoubleVertex> stateVector) {
        /*
        During the update, the estimate of the state is updated using the observations from truthModel

        This is the Data Assimilation step similar to the workflow of the SimpleModel's.
        Here we need to:
            Initialise the Black Box model
            Run for XX iterations
            Observe truth data
            Create the BayesNet
            Sample from the posterior
            Get information from the samples
        */
        //System.out.println("UPDATING");

        // INITIALISE THE BLACK BOX MODEL

        //UnaryOpLambda<GenericTensor, GenericTensor> box = new UnaryOpLambda<>(stateVector, stateHistory???);
        //UnaryOpLambda<GenericTensor, ???> box = new UnaryOpLambda<>(stateVector, tempModel.schedule.step(tempModel));

        //UnaryOpLambda<GenericTensor, Integer[]> box = new UnaryOpLambda<>(stateVector, tempModel.area.getAllObjects())

        // RUN FOR XX ITERATIONS
        // Are both of these done in predict() or runDataAssim...?

        // OBSERVE TRUTH DATA
        // Loop through and apply observations


    }


    private static void writeModelHistory(Station model, String fileName) {
        model.analysis.writeDataFrame(model.analysis.stateDataFrame, fileName + ".csv");
    }
}