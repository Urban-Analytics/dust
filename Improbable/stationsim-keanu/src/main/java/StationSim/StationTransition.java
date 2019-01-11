package StationSim;

import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.generic.GenericTensor;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.operators.unary.UnaryOpLambda;
import org.yaml.snakeyaml.scanner.Constant;
import sim.util.Bag;
import sim.util.Double2D;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;


public class StationTransition {

    // Create truth model (and temp model)
    static Station truthModel = new Station(System.currentTimeMillis());
    static Station tempModel = new Station(System.currentTimeMillis() + 1);

    private static int NUM_ITER = 2000; // 2000
    private static int WINDOW_SIZE = 200; // 200
    private static int NUM_WINDOWS = NUM_ITER / WINDOW_SIZE;


    public static void runDataAssimilation() {

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

        //tempModel.start();
        //tempModel.schedule.step(tempModel);
        //Bag people = tempModel.area.getAllObjects();

        Bag people = truthModel.area.getAllObjects();
        List<Person> currentState = new ArrayList<>(people);

        assert(currentState.size() != 0);

        // Start data assimilation window
        for (int i = 0; i < 3; i++) {

            System.out.println("Entered DA window " + i);

            // ****************** Predict ******************

            currentState = predict(currentState, WINDOW_SIZE);

            System.out.println("Predict finished");

            // ****************** Update ******************

            // Create state vector

            //double[][] stateVector = buildPrimitiveStateVector(currentState);
            GenericTensor stateVector = buildTensorStateVector(currentState);

            System.out.println("State Vector built.");

            update(stateVector);

            System.out.println("Update finished");

            // Rebuild currentState from stateVector
            currentState = rebuildCurrentState(stateVector);

            System.out.println("Current State rebuilt");

            // for 1000 iterations

            /*
            Try building box inside runDataAssimilation() with output from predict() (first method)
            Might need to buildStateVector() inside predict()
             */
        }
    }


    public static List<Person> predict(List<Person> currentState, int window_size) {
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

        for (int i=0; i < window_size; i++) {
            // Step all the people window_size times
            tempModel.schedule.step(truthModel);
        }

        // Create personList and populate
        List<Person> personList = new ArrayList<>();
        personList.addAll(tempModel.area.getAllObjects());

        assert(personList.size() > 0);
        assert(people.size() == personList.size());

        return personList;
    }


    // Swap back to double[][] when not using GenericTensor
    public static void update(GenericTensor stateVector) {
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
        //UnaryOpLambda<GenericTensor, Integer[]> box = new UnaryOpLambda<>(stateVector, stateHistory???);
        //UnaryOpLambda<GenericTensor, Integer[]> box = new UnaryOpLambda<>(stateVector, tempModel.schedule.step(tempModel));
        // RUN FOR XX ITERATIONS
        // Are both of these done in predict() or runDataAssim...?

        // OBSERVE TRUTH DATA


    }


    public static double[][] buildPrimitiveStateVector(List<Person> currentState) {

        double[][] stateVector = new double[currentState.size()][4];

        // Build state vector [[x, y, exit, desiredSpeed],...]
        int counter = 0;
        for (Person person : currentState) {
            Double2D loc = person.getLocation();
            double xLoc = loc.x;
            double yLoc = loc.y;
            double desiredSpeed = person.getDesiredSpeed();

            // Get exit and exitNumber as double
            Exit exit = person.getExit();
            double exitNum = exit.exitNumber;

            stateVector[counter][0] = xLoc;
            stateVector[counter][1] = yLoc;
            stateVector[counter][2] = exitNum;
            stateVector[counter][3] = desiredSpeed;

            counter++;
        }
        assert(currentState.size() == stateVector.length);

        return stateVector;
    }


    public static GenericTensor buildTensorStateVector(List<Person> currentState) {

        GenericTensor stateVector = GenericTensor.createFilled(0, new long[] {currentState.size(), 4});

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

            stateVector.setValue(xLoc, counter, 0);
            stateVector.setValue(yLoc, counter, 1);
            stateVector.setValue(exitNum, counter, 2);
            stateVector.setValue(desiredSpeed, counter, 3);

            counter++;
        }
        assert(currentState.size() == stateVector.getLength());
        return stateVector;
    }


    public static List<Person> rebuildCurrentState(double[][] stateVector) {

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


    public static List<Person> rebuildCurrentState (GenericTensor stateVector) {

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
