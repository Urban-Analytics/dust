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

import com.sun.j3d.utils.scenegraph.io.state.com.sun.j3d.utils.geometry.ConeState;
import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.mcmc.MetropolisHastings;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.generic.GenericTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.CastDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.operators.unary.UnaryOpLambda;
import org.yaml.snakeyaml.scanner.Constant;
import sim.util.Bag;
import sim.util.Double2D;

import java.io.Writer;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.List;


public class StationTransition {

    // Create truth model (and temp model)
    static Station truthModel = new Station(System.currentTimeMillis());
    static Station tempModel = new Station(System.currentTimeMillis() + 1);

    private static int NUM_ITER = 2000; // 2000
    private static int WINDOW_SIZE = 200; // 200
    private static int NUM_WINDOWS = NUM_ITER / WINDOW_SIZE;

    private static int NUM_SAMPLES = 2000;
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

    private static List<ConstantDoubleVertex[]> truthHistory = new ArrayList<>();


    private static void runDataAssimilation() {


        /*
         ************ CREATE THE TRUTH DATA ************
         */

        /*do
            if (!truthModel.schedule.step(truthModel)) break;
        while (truthModel.schedule.getSteps() < 2000);
        truthModel.finish();
        */

        // Run the truth model
        truthModel.start(rand);
        System.out.println("truthModel.start() has executed successfully");

        int counter = 0;
        results = new double[(NUM_ITER / WINDOW_SIZE)];

        // Rewrote how truthModel is run as it allows easier extraction of truth data at intervals
        while(truthModel.schedule.getSteps() < NUM_ITER) {
            if (!(truthModel.schedule.getSteps() == 0) && (truthModel.schedule.getSteps() % WINDOW_SIZE == 0.0)) {
                // Get numPeople output
                Bag truthPeople = truthModel.area.getAllObjects();
                results[counter] = truthPeople.size();

                // Build stateVector to observe (unless iter == 0)

                List<Person> truthList = new ArrayList<>(truthPeople);
                ConstantDoubleVertex[] truthVector = buildStateVector(truthList);
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

        assert(!truthHistory.isEmpty());

        /*
         ************ START THE MAIN LOOP ************
         */
        System.out.println("Starting DA loop");

        // Create the current state (initially 0). Use the temporary model because so far it is in its initial state
        tempModel.start(rand);
        //tempModel.schedule.step(tempModel);
        ConstantDoubleVertex[] currentStateEstimate = buildStateVector(truthModel);

        // Start data assimilation window
            // predict
            // update
            // for 1000 iterations

        //tempModel.start(rand);
        //tempModel.schedule.step(tempModel);

        //Bag people = tempModel.area.getAllObjects();
        //List<Person> personList = new ArrayList<Person>(people);
        //assert(personList.size() != 0);



        // Start data assimilation window
        for (int i = 0; i < NUM_WINDOWS; i++) {

            System.out.println("Entered Data Assimilation window " + i);

            // Step the model
            //for (int j=0; j<WINDOW_SIZE; j++) {
            //    tempModel.schedule.step(tempModel);
            //}

            /*
             ************ INITIALISE THE BLACK BOX MODEL ************
             */

            ConstantDoubleVertex[] state = currentStateEstimate;

            UnaryOpLambda<ConstantDoubleVertex[], ConstantDoubleVertex[]> box =
                    new UnaryOpLambda<ConstantDoubleVertex[], ConstantDoubleVertex[]>(new long[]{state.length}, state, StationTransition::predict);


            //Vertex<ConstantDoubleVertex> state =

            // Build new stateVector
            //personList = new ArrayList<Person>(tempModel.area.getAllObjects());

            //ConstantDoubleVertex[] stateVector = buildStateVector(personList);


            /*
             ************ OBSERVE SOME TRUTH DATA ************
             */

            // Observe the truth data plus some noise?
            System.out.println("Observing truth data. Adding noise with standard dev: " + SIGMA_NOISE);
            System.out.println("Observing at iterations: ");
            for (Integer j = i * WINDOW_SIZE; j < i * WINDOW_SIZE + WINDOW_SIZE ; j++) {
                System.out.print(j+",");

                // Need to get the output from the box model
                ConstantDoubleVertex[] output = box.getValue();

                // output with a bit of noise. Lower sigma makes it more constrained.
                GaussianVertex noisyOutput[] = new GaussianVertex[state.length];
                for (int k=0; k<state.length; k++) {
                    noisyOutput[k] = new GaussianVertex(output[k], SIGMA_NOISE);
                }

                // Observe the output (could do at same time as adding noise to each element in the state vector?)
                ConstantDoubleVertex[] history = truthHistory.get(i); // This is the truth state at the end of window i
                for (int k=0; k<state.length; k++) {
                    noisyOutput[k].observe(history[k].getValue(0)); // Observe each element in the truth state vector
                }
            }
            System.out.println();


            /*
             ************ CREATE THE BAYES NET ************
             */

            BayesianNetwork net = new BayesianNetwork(box.getConnectedGraph());

            /*
             ************ SAMPLE FROM THE POSTERIOR************
             */

            /*
             ************ GET THE INFORMATION OUT OF THE SAMPLES ************
             */

            // for 1000 iterations

        }
        tempModel.finish();
        //writeModelHistory(truthModel, "truthModel" + System.currentTimeMillis());
        //writeModelHistory(tempModel, "tempModel" + System.currentTimeMillis());
    }

    private static ConstantDoubleVertex[] buildStateVector(Station model) {

        List<Person> people = new ArrayList<>(model.area.getAllObjects());

        return buildStateVector(people);
    }


    private static ConstantDoubleVertex[] buildStateVector(List<Person> personList) {

        assert(!personList.isEmpty());

        ConstantDoubleVertex[] stateVector = new ConstantDoubleVertex[personList.size() * 3];
        agentExits = new Exit[personList.size()];

        int counter = 0;
        for (Person person : personList) {
            int counter2 = counter * 3;

            stateVector[counter2] = new ConstantDoubleVertex(person.getLocation().x);
            stateVector[counter2 + 1] =  new ConstantDoubleVertex(person.getLocation().y);
            stateVector[counter2 + 2] =  new ConstantDoubleVertex(person.getDesiredSpeed());

            agentExits[counter] = person.getExit();

            assert(agentExits.length == stateVector.length / 3);

            counter++;
        }

        return stateVector;
    }


    private static List<Person> rebuildPersonList(ConstantDoubleVertex[] stateVector) {

        List<Person> personList = new ArrayList<>();
        // Find halfway point so equal number of agents assigned entrance 1 and 2 (doesn't affect model run)
        int halfwayPoint = (stateVector.length / 3) / 2;
        int entranceNum = 0;

        System.out.println(stateVector.length);
        System.out.println(stateVector.length / 3);
        System.out.println(halfwayPoint);

        for (int i=0; i < stateVector.length / 3; i++) {
            // j is equal to ith lot of 3s
            int j = i * 3;
            // Extract vertex's from stateVector
            double xLoc = stateVector[j].getValue(0);
            double yLoc = stateVector[j+1].getValue(0);
            Double2D loc = new Double2D(xLoc, yLoc); // Build Double2D for location

            double desSpeed = stateVector[j+2].getValue(0);

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
            Person person = new Person(tempModel.getPersonSize(), loc, name, tempModel, exit, entrance, desSpeed);

            // add person to personList
            personList.add(person);
        }
        return personList;
    }


    private static ConstantDoubleVertex[] predict(ConstantDoubleVertex[] initialState) {

        // Find all the people
        Bag people = tempModel.area.getAllObjects();

        // Remove all old people
        for (Object o : people) {
            tempModel.area.remove(o);
        }

        // Convert from the initial state ( a list of numbers) to a list of agents
        List<Person> personList = new ArrayList<>(rebuildPersonList(initialState)); // NOW FIXED (this will break it (needs to convert initialState -> PersonList))

        // Add people from the personList
        for (Object o : personList) {
            Person p = (Person) o;
            tempModel.area.setObjectLocation(p, p.getLocation());
        }

        // Propagate the model
        for (int i=0; i < WINDOW_SIZE; i++) {
            // Step all the people window_size times
            tempModel.schedule.step(tempModel);
        }

        // Return the new state, after those iterations

        return buildStateVector(tempModel);

        /*
        // Create personList and populate
        List<Person> newPersonList = new ArrayList<>();
        personList.addAll(tempModel.area.getAllObjects());

        // Check personList is not empty, then that it is same size as Bag people
        assert(newPersonList.size() > 0);
        assert(people.size() == personList.size());

        return newPersonList;
        */
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


    public static void main(String[] args) {

        StationTransition.runDataAssimilation();
        System.out.println("Happy days, runDataAssimilation has executed successfully!");

        System.exit(0);
    }

    private static Integer[] runModel() {
        // Create output array
        Integer[] output = new Integer[WINDOW_SIZE];
        // Step the model
        for (int i=9; i<WINDOW_SIZE; i++) {
            tempModel.schedule.step(tempModel);
            output[i] = tempModel.area.getAllObjects().size();
        }
        return output;
    }

    private static double[] buildStateVector2(List<Person> personList) {

        double[] stateVector = new double[personList.size() * 3];
        agentExits = new Exit[personList.size()];

        int counter = 0;
        for (Person person : personList) {
            int counter2 = counter * 3;

            stateVector[counter2] = person.getLocation().x;
            stateVector[counter2 + 1] = person.getLocation().y;
            stateVector[counter2 + 2] = person.getDesiredSpeed();

            agentExits[counter] = person.getExit();

            assert(agentExits.length == stateVector.length / 3);

            counter++;
        }

        return stateVector;
    }

    private static ConstantDoubleVertex[] buildStateVector3(List<Person> personList) {

        assert(!personList.isEmpty());

        ConstantDoubleVertex[] stateVector = new ConstantDoubleVertex[personList.size() * 3];
        agentExits = new Exit[personList.size()];

        int counter = 0;
        for (Person person : personList) {
            int counter2 = counter * 3;

            stateVector[counter2] = new ConstantDoubleVertex(person.getLocation().x);
            stateVector[counter2 + 1] =  new ConstantDoubleVertex(person.getLocation().y);
            stateVector[counter2 + 2] =  new ConstantDoubleVertex(person.getDesiredSpeed());

            agentExits[counter] = person.getExit();

            assert(agentExits.length == stateVector.length / 3);

            counter++;
        }

        return stateVector;
    }

    private static DoubleTensor buildStateVector4(List<Person> personList) {

        assert(!personList.isEmpty());

        DoubleTensor stateVector = DoubleTensor.create(0.0, new long[]{personList.size()});

        //ConstantDoubleVertex[] stateVector = new ConstantDoubleVertex[personList.size() * 3];
        agentExits = new Exit[personList.size()];

        int counter = 0;
        for (Person person : personList) {
            int counter2 = counter * 3;

            stateVector.setValue(person.getLocation().x, counter2);
            stateVector.setValue(person.getLocation().y, counter2 + 1);
            stateVector.setValue(person.getDesiredSpeed());

            agentExits[counter] = person.getExit();

            assert(agentExits.length == (stateVector.getLength() / 3));

            counter++;
        }

        return stateVector;
    }

}

