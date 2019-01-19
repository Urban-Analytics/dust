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
import io.improbable.keanu.tensor.generic.GenericTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.operators.unary.UnaryOpLambda;
import sim.util.Bag;
import sim.util.Double2D;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;


public class StationTransition {

    // Create truth model (and temp model)
    static Station truthModel = new Station(System.currentTimeMillis());
    static Station tempModel = new Station(System.currentTimeMillis() + 1);

    private static int NUM_ITER = 2000; // 2000
    private static int WINDOW_SIZE = 200; // 200
    private static int NUM_WINDOWS = NUM_ITER / WINDOW_SIZE;

    // List of agent exits to use when rebuilding from stateVector
    static DoubleVertex[][] stateVector;
    static Exit[] agentExits;


    private static void runDataAssimilation() {

        // Run the truth model
        truthModel.start();
        System.out.println("truthModel.start() has executed successfully");

        // The following to replace doLoop() due to premature exit of simulation
        // Run model to generate hypothetical truth data
        do
            if (!truthModel.schedule.step(truthModel)) break;
        while (truthModel.schedule.getSteps() < 2000);
        truthModel.finish();

        // Should I run for only 1 step to get the agents into the simulation? Then go from there?

        System.out.println("Executed truthModel.finish()");

        System.out.println("Starting DA loop");

        // Start data assimilation window
            // predict
            // update
            // for 1000 iterations

        tempModel.start();
        tempModel.schedule.step(tempModel);
        Bag people = tempModel.area.getAllObjects();

        //Bag people = tempModel.area.getAllObjects();
        List<Person> personList = new ArrayList<>(people);

        assert(personList.size() != 0);

        // Start data assimilation window
        // Should be i<NUM_WINDOWS
        for (int i = 0; i < 3; i++) {

            System.out.println("Entered DA window " + i);

            // ****************** Predict ******************
            //personList = predict(personList, WINDOW_SIZE);
            //System.out.println("Predict finished");

            // ****************** Initialise Black Box Model ******************

            //double[][] stateVector = buildPrimitiveStateVector(personList);
            //GenericTensor stateVector = buildTensorStateVector(personList);
            //UnaryOpLambda<GenericTensor, List<Person>> box = new UnaryOpLambda<>(stateVector, predict(personList, WINDOW_SIZE));

            // ****************** Update ******************


            // Step model
            for (int j=0; j<WINDOW_SIZE; j++) {
                tempModel.schedule.step(tempModel);
            }

            // Build stateVector
            stateVector = buildProbableStateVector(personList);

            // Observe truth data









            update(stateVector);

            System.out.println("Update finished");

            // Rebuild personList from stateVector
            personList = rebuildCurrentState(stateVector);

            System.out.println("Current State rebuilt");

            // for 1000 iterations

            /**
             * To Do for new idea:
             *      Replace x and y position doubles with DoubleVertex's                        DONE
             *      Modify how position is treated in agent.class                               DONE
             *      Modify step method in Person.class
             *      Modify lerp() method in Person.class
             *      How does the slowing distance work? And do I need to alter how this works?
             *      Make new probableSpeed Vertex for Person                                    Done
             *      Alter all methods in agent movement
             *      Figure out (and write down somewhere) how MASON represents agents,
             *          i.e. what information it needs (just x,y?)
             *
             *      COMMENT OUT ALL NON-PROBABILISTIC CONSTRUCTORS: Find out where errors arise and fix
             *          (should be able to uncomment after with no ill effects)
             */

        }
    }


    private static List<Person> predict(List<Person> currentState, int window_size) {
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

        // Add people from the currentState
        for (Object o : currentState) {
            Person p = (Person) o;
            tempModel.area.setObjectLocation(p, p.getLocation()); // DOES THIS ADD A NEW AGENT AND MOVE THEM AT THE SAME TIME??
        }

        // Propagate the model
        for (int i=0; i < window_size; i++) {
            // Step all the people window_size times
            tempModel.schedule.step(tempModel);
        }

        // Create personList and populate
        List<Person> personList = new ArrayList<>();
        personList.addAll(tempModel.area.getAllObjects());

        // Check personList is not empty, then that it is same size as Bag people
        assert(personList.size() > 0);
        assert(people.size() == personList.size());

        return personList;
    }


    // Swap back to double[][] when not using GenericTensor
    private static void update(DoubleVertex[][] stateVector) {
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
        for (int i=0; i < WINDOW_SIZE; i++) {
            // NativeModel style observations (without ...OpLambda) or SimpleWrapperC style?
        }

    }


    private static DoubleVertex[][] buildProbableStateVector(List<Person> personList) {

        DoubleVertex[][] stateVector = new DoubleVertex[personList.size()][3];

        // Build state vector [[x, y, probableSpeed],...]
        int counter = 0;
        for (Person person : personList) {
            stateVector[counter][0] = person.xLoc;
            stateVector[counter][1] = person.yLoc;
            stateVector[counter][2] = person.probableSpeed;

            // Add agent exits to list to use when rebuilding
            agentExits[counter] = person.getExit();


            counter++;
        }
        assert(personList.size() == stateVector.length);

        return stateVector;
    }


    private static GenericTensor buildTensorStateVector(List<Person> currentState) {

        ConstantDoubleVertex[] data = new ConstantDoubleVertex[currentState.size() * 4];

        //GenericTensor stateVector = GenericTensor.createFilled(0, new long[] {currentState.size(), 4});
        GenericTensor<ConstantDoubleVertex> stateVector = GenericTensor.create(data, new long[] {currentState.size(), 4});

        // Build state vector [[x, y, exit, desiredSpeed],...]
        int counter = 0;
        for (Person person : currentState) {
            Double2D loc = person.getLocation();
            ConstantDoubleVertex xLoc = new ConstantDoubleVertex(loc.x);
            ConstantDoubleVertex yLoc = new ConstantDoubleVertex(loc.y);
            ConstantDoubleVertex desiredSpeed = new ConstantDoubleVertex(person.getDesiredSpeed());

            // Get exit and exitNumber as double
            Exit exit = person.getExit();
            ConstantDoubleVertex exitNum = new ConstantDoubleVertex(exit.exitNumber);

            stateVector.setValue(xLoc, counter, 0)
                    .setValue(yLoc, counter, 1)
                    .setValue(exitNum, counter, 2)
                    .setValue(desiredSpeed, counter, 3);


            counter++;
        }
        assert(currentState.size() == stateVector.getLength());
        return stateVector;
    }


    private static List<Person> rebuildCurrentState(double[][] stateVector) {

        List<Person> personList = new ArrayList<>();

        int halfwayPoint = stateVector.length / 2;
        int entrance = 0;

        for (int i=0; i < stateVector.length; i++) {
            double xLoc = stateVector[i][0];
            double yLoc = stateVector[i][1];
            int exitNum = (int) stateVector[i][2];
            double desiredSpeed = stateVector[i][3];

            Exit exit = truthModel.exits.get(exitNum-1);
            Double2D location = new Double2D(xLoc, yLoc);

            String name = "Person " + (i+1);

            if (i > halfwayPoint) { entrance = 1; }

            ArrayList<Entrance> entrances = truthModel.getEntrances();

            Person person = new Person(tempModel.getPersonSize(), location, name, tempModel, exit,
                    entrances.get(entrance), desiredSpeed);

            personList.add(person);
        }
        return personList;
    }


    private static List<Person> rebuildCurrentState (GenericTensor stateVector) {

        List<Person> personList = new ArrayList<>();

        int halfwayPoint = (int) stateVector.getLength() / 2;
        int entrance = 0;

        for (int i=0; i < (stateVector.getLength() / 4); i++) {
            ConstantDoubleVertex xLoc = (ConstantDoubleVertex) stateVector.getValue(i, 0);
            ConstantDoubleVertex yLoc = (ConstantDoubleVertex) stateVector.getValue(i, 1);
            ConstantDoubleVertex exitNum = (ConstantDoubleVertex) stateVector.getValue(i, 2);
            ConstantDoubleVertex desiredSpeed = (ConstantDoubleVertex) stateVector.getValue(i, 3);

            double xLocD = xLoc.getValue(0);
            double yLocD = yLoc.getValue(0);
            int exitNumD = (int) exitNum.getValue(0);
            double desiredSpeedD = desiredSpeed.getValue(0);

            Exit exit = truthModel.exits.get(exitNumD-1);
            Double2D location = new Double2D(xLocD, yLocD);

            String name = "Person " + (i+1);

            if (i > halfwayPoint) { entrance = 1; }

            ArrayList<Entrance> entrances = truthModel.getEntrances();

            Person person = new Person(tempModel.getPersonSize(), location, name, tempModel, exit,
                    entrances.get(entrance), desiredSpeedD);

            personList.add(person);

            /*
            if (i % 10 == 0) {
                System.out.println("Built person number " + i);
            }
            */
        }
        return personList;
    }


    public static void main(String[] args) {

        StationTransition.runDataAssimilation();
        System.out.println("Happy days, runDataAssimilation has executed successfully!");

        System.exit(0);
    }
}
