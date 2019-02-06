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

import io.improbable.keanu.algorithms.NetworkSample;
import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.ProbabilisticModel;
import io.improbable.keanu.algorithms.mcmc.MetropolisHastings;
import io.improbable.keanu.network.BayesianNetwork;
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

import javax.swing.*;
import java.io.Writer;
import java.util.ArrayList;
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


    private static void runDataAssimilation() {


        /*
         ************ CREATE THE TRUTH DATA ************
         */

        // Run the truth model
        truthModel.start(rand);
        System.out.println("truthModel.start() has executed successfully");

        int counter = 0;
        results = new double[(NUM_ITER / WINDOW_SIZE)];

        // Rewrote how truthModel is run as it allows easier extraction of truth data at intervals
        while (truthModel.schedule.getSteps() < NUM_ITER) {
            if (!(truthModel.schedule.getSteps() == 0) && (truthModel.schedule.getSteps() % WINDOW_SIZE == 0.0)) {
                // Get numPeople output
                Bag truthPeople = truthModel.area.getAllObjects();
                results[counter] = truthPeople.size();

                // Build stateVector to observe (unless iter == 0)
                List<Person> truthList = new ArrayList<>(truthPeople);
                Vertex<DoubleTensor[]> truthVector = buildStateVector(truthList);

                // Add stateVector to history
                truthHistory.add(truthVector);

                // Increment counter
                counter++;
            }
            // Step while condition is not true
            truthModel.schedule.step(truthModel);
        }
        // Finish model
        truthModel.finish();
        System.out.println("Executed truthModel.finish()");

        // Ensure truthModel has successfully recorded history
        assert (!truthHistory.isEmpty());

        /*
         ************ START THE MAIN LOOP ************
         */
        System.out.println("Starting DA loop");


        // Create the current state (initially 0). Use the temporary model because so far it is in its initial state
        tempModel.start(rand);
        Vertex<DoubleTensor[]> currentStateEstimate = buildStateVector(tempModel); // TODO: Shouldn't this be tempModel?

        // Start data assimilation window
        for (int i = 0; i < NUM_WINDOWS; i++) {

            System.out.println("Entered Data Assimilation window " + i);

            /*
             ************ INITIALISE THE BLACK BOX MODEL ************
             */

            Vertex<DoubleTensor[]> state = currentStateEstimate;
            Vertex<DoubleTensor[]> box = predict(WINDOW_SIZE, state);


            /*
             ************ OBSERVE SOME TRUTH DATA ************
             */


            // Observe the truth data plus some noise?
            System.out.println("Observing truth data. Adding noise with standard dev: " + SIGMA_NOISE);
            /*
            System.out.println("Observing at iterations: ");
            for (Integer j = i * WINDOW_SIZE; j < i * WINDOW_SIZE + WINDOW_SIZE; j++) {
                System.out.print(j + ",");

                // Need to get the output from the box model
                DoubleTensor[] output = ((CombineDoubles) box).calculate();

                // output with a bit of noise. Lower sigma makes it more constrained.
                GaussianVertex noisyOutput[] = new GaussianVertex[state.getValue().length];
                for (int k = 0; k < state.getValue().length; k++) {
                    noisyOutput[k] = new GaussianVertex(output[k].getValue(), SIGMA_NOISE);
                }

                // Observe the output (could do at same time as adding noise to each element in the state vector?)
                Vertex<DoubleTensor[]> history = truthHistory.get(i); // This is the truth state at the end of window i
                for (int k = 0; k < state.getValue().length; k++) {
                    noisyOutput[k].observe(history.getValue()[k].getValue(0)); // Observe each element in the truth state vector
                }
            }
            System.out.println();
            */

            // Get last value of output from box
            Vertex<DoubleTensor> out = CombineDoubles.getAtElement(WINDOW_SIZE-1, box);
            // Output with a bit of noise
            GaussianVertex noiseOut = new GaussianVertex(out.getValue().scalar(), SIGMA_NOISE);
            // Observe the output
            noiseOut.observe(truthHistory.get(i).getValue()[WINDOW_SIZE-1]);


            /*
             ************ CREATE THE BAYES NET ************
             */

            BayesianNetwork net = new BayesianNetwork(box.getConnectedGraph());

            /*
             ************ SAMPLE FROM THE POSTERIOR************
             */

            List<Vertex> parameters = new ArrayList<>();
            parameters.add(box);

            NetworkSamples sampler = MetropolisHastings.builder().build().getPosteriorSamples(
                    (ProbabilisticModel) net,
                    parameters,
                    NUM_SAMPLES);

            sampler = sampler.drop(DROP_SAMPLES).downSample(DOWN_SAMPLE);

            /*
             ************ GET THE INFORMATION OUT OF THE SAMPLES ************
             */

            // for 1000 iterations

        }
        tempModel.finish();
        //writeModelHistory(truthModel, "truthModel" + System.currentTimeMillis());
        //writeModelHistory(tempModel, "tempModel" + System.currentTimeMillis());
    }


    private static Vertex<DoubleTensor[]> buildStateVector(List<Person> personList) {

        assert (!personList.isEmpty());

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

            // Ensure exit array is same length as stateVertices / 3 (as 3 vertices per agent)
            assert (agentExits.length == stateVertices.size() / 3);

            counter++;
        }
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
        int halfwayPoint = (stateVector.getValue().length / 3) / 2; // Divide by 3 and then by 2 to divide into triplets (3 Vertex's per agent), then half number of agents
        //int halfwayPoint = (stateVector.size() / 3) / 2;
        int entranceNum = 0;

        //System.out.println(stateVector.getValue().length);
        //System.out.println(stateVector.getValue().length / 3);
        //System.out.println(halfwayPoint);

        for (int i = 0; i < stateVector.getValue().length / 3; i++) {
            // j is equal to ith lot of 3s
            int j = i * 3;

            // Extract vertex's from stateVector
            Vertex<DoubleTensor> xLoc = CombineDoubles.getAtElement(j, stateVector);
            Vertex<DoubleTensor> yLoc = CombineDoubles.getAtElement(j+1, stateVector);
            // Build Double2D
            Double2D loc = new Double2D(xLoc.getValue().scalar(), yLoc.getValue().scalar()); // Build Double2D for location

            Vertex<DoubleTensor> desSpeed = CombineDoubles.getAtElement(j+2, stateVector);
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

            String name = "Person: " + i;

            // Create person using location and speed from stateVector
            Person person = new Person(tempModel.getPersonSize(), loc, name, tempModel, exit, entrance, speedAsDouble);

            // add person to personList
            personList.add(person);
        }
        return personList;
    }


    private static Vertex<DoubleTensor[]> predict(int stepCount, Vertex<DoubleTensor[]> initialState) {

        // Find all the people
        Bag people = tempModel.area.getAllObjects();

        // Remove all old people
        for (Object o : people) {
            tempModel.area.remove(o);
        }

        // Convert from the initial state ( a list of numbers) to a list of agents
        List<Person> personList = new ArrayList<>(rebuildPersonList(initialState)); // FIXED NOW (this will break it (needs to convert initialState -> PersonList))

        // Add people from the personList
        for (Object o : personList) {
            Person p = (Person) o;
            tempModel.area.setObjectLocation(p, p.getLocation());
        }

        // Propagate the model
        for (int i = 0; i < WINDOW_SIZE; i++) {
            // Step all the people window_size times
            tempModel.schedule.step(tempModel);
        }

        // Get new stateVector
        List<Person> personList2 = new ArrayList<>(tempModel.area.getAllObjects());
        Vertex<DoubleTensor[]> state = buildStateVector(personList2); // build stateVector directly from Bag people?

        UnaryOpLambda<DoubleTensor[], DoubleTensor[]> box = new UnaryOpLambda<>(
                new long[0],
                state,
                (currentState) -> step(stepCount, currentState)
        );

        return box;
    }


    private static DoubleTensor[] step(int stepCount, DoubleTensor[] initialState) {

        DoubleTensor[] steppedState = new DoubleTensor[initialState.length];

        for (int i=0; i < stepCount; i++) {
            steppedState = initialState; // actually update the state here
        }
        return steppedState;
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