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

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import sim.util.Bag;
import sim.util.Double2D;

import java.io.Writer;
import java.util.ArrayList;
import java.util.List;


public class StationTransition {

    // Create truth model (and temp model)
    static Station truthModel = new Station(System.currentTimeMillis());
    static Station tempModel = new Station(System.currentTimeMillis() + 1);

    private static int NUM_ITER = 2000; // 2000
    private static int WINDOW_SIZE = 200; // 200
    private static int NUM_WINDOWS = NUM_ITER / WINDOW_SIZE;

    // Writing to file
    private static Writer stateWriter; // To write stateVectorHistory

    // Initialise random var
    // random generator for start() method
    private static KeanuRandom rand = new KeanuRandom();

    // List of agent exits to use when rebuilding from stateVector
    //static DoubleVertex[][] stateVector;
    private static Exit[] agentExits;

    private static double[] results;


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

        // Rewrote propagation of this model as it allows easier extraction of truth data at intervals
        while(truthModel.schedule.getSteps() < NUM_ITER) {
            if (truthModel.schedule.getSteps() % WINDOW_SIZE == 0.0) {
                results[counter] = truthModel.area.getAllObjects().size();
                counter++;
            }
            truthModel.schedule.step(truthModel);
        }
        truthModel.finish();

        System.out.println("Executed truthModel.finish()");


        /*
         ************ START THE MAIN LOOP ************
         */
        System.out.println("Starting DA loop");

        // Start data assimilation window
            // predict
            // update
            // for 1000 iterations

        tempModel.start(rand);
        tempModel.schedule.step(tempModel);
        Bag people = tempModel.area.getAllObjects();

        //Bag people = tempModel.area.getAllObjects();
        List<Person> personList = new ArrayList<Person>(people);
        assert(personList.size() != 0);

        // Start data assimilation window
        for (int i = 0; i < NUM_WINDOWS; i++) {

            System.out.println("Entered Data Assimilation window " + i);

            // Step the model
            for (int j=0; j<WINDOW_SIZE; j++) {
                tempModel.schedule.step(tempModel);
            }

            // Build new stateVector
            //personList = new ArrayList<Person>(tempModel.area.getAllObjects());


            /*
             ************ INITIALISE THE BLACK BOX MODEL ************
             */

            //UnaryOpLambda<DoubleTensor, Integer[]> box = new UnaryOpLambda<DoubleTensor, Integer[]>(stateVector, StationTransition::runModel);


            /*
             ************ OBSERVE SOME TRUTH DATA ************
             */


            /*
             ************ CREATE THE BAYES NET ************
             */


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


    /*
    private static double[] buildStateVector(List<Person> personList) {

        double[] stateVector = new double[personList.size() * 3];


    } */


    private static List<Person> predict(List<Person> personList) {
        /*
        During prediction, the prior state is propagated forward in time (i.e. stepped) by window_size iterations
         */

        //System.out.println("PREDICTING");

        // Find all the people
        Bag people = tempModel.area.getAllObjects();

        // Remove all old people
        for (Object o : people) {
            tempModel.area.remove(o);
        }

        // Add people from the personList
        for (Object o : personList) {
            Person p = (Person) o;
            tempModel.area.setObjectLocation(p, p.getLocation()); // DOES THIS ADD A NEW AGENT AND MOVE THEM AT THE SAME TIME??
        }

        // Propagate the model
        for (int i=0; i < WINDOW_SIZE; i++) {
            // Step all the people window_size times
            tempModel.schedule.step(tempModel);
        }

        // Create personList and populate
        List<Person> newPersonList = new ArrayList<>();
        personList.addAll(tempModel.area.getAllObjects());

        // Check personList is not empty, then that it is same size as Bag people
        assert(newPersonList.size() > 0);
        assert(people.size() == personList.size());

        return newPersonList;
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
        model.analysis.writeDataFrame(model.analysis.stateDataFrame, fileName + ".txt");
    }


    public static void main(String[] args) {

        StationTransition.runDataAssimilation();
        System.out.println("Happy days, runDataAssimilation has executed successfully!");

        System.exit(0);
    }
}
