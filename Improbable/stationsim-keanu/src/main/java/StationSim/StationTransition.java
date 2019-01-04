package StationSim;

import com.google.common.base.CharMatcher;
import sim.util.Bag;
import sim.util.Double2D;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;


public class StationTransition {

    private static int NUM_ITER = 1000;

    // Create truth model
    static Station truthModel = new Station(System.currentTimeMillis());

    // Create state vector
    static double[][] stateVector;

    public static void runDataAssimilation() {

        // Run the truth model
        truthModel.start();
        System.out.println("truthModel.start() has executed successfully");

        // The following to replace doLoop() due to premature exit of simulation
        do
            if (!truthModel.schedule.step(truthModel)) break;
        while (truthModel.schedule.getSteps() < 1000);
        truthModel.finish();

        // Should I run for only 1 step to get the agents into the simulation? Then go from there?

        System.out.println("Executed truthModel.finish()");

        // XXXX RUN UNTIL FINISHED (doLoop??)
        //truthModel.doLoop(Station.class, args);
        //System.out.println("StationTransition doLoop finished");
        //truthModel.finish();

        System.out.println("Starting DA window");

        // Start data assimilation window
        // predict
        // update
        // for 1000 iterations

        Bag people = truthModel.area.getAllObjects();
        List<Person> currentState = new ArrayList<>(people);

        // Start data assimilation window
        for (int iter = 0; iter < NUM_ITER; iter++) {

            System.out.println("Entered DA iteration " + iter);

            // ****************** Predict ******************

            stateVector = predict(currentState);

            // ****************** Update ******************

            currentState = update(currentState);

            // for 1000 iterations

        }
    }


    public static double[][] predict(List<Person> currentState) {
        System.out.println("PREDICTING");

        //List<double[]> stateVector = new ArrayList<>();
        double[][] stateVector = new double[currentState.size()][4];

        // Build state vector [[x, y, exit, currentSpeed],...]
        int counter = 0;
        for (Person person : currentState) {
            Double2D loc = person.getLocation();
            double xLoc = loc.x;
            double yLoc = loc.y;
            double currentSpeed = person.getCurrentSpeed();

            // Get assigned exit and extract exit number
            //Exit exit = person.exit;
            //String exitName = exit.name;
            //System.out.println(exitName);
            //String exitNum = CharMatcher.digit().retainFrom(exitName);
            // Extract integer of exit number from exitName and retain as double
            //double exitNum = Integer.parseInt(CharMatcher.digit().retainFrom(exitName));

            // Get exit and exitNumber as double
            Exit exit = person.getExit();
            double exitNum = exit.exitNumber;

            // Produce tempArray and add it to stateVector List
            //double[] tempArray = new double[] {xLoc, yLoc, currentSpeed, exitNum};
            //stateVector.add(tempArray);

            stateVector[counter][0] = xLoc;
            stateVector[counter][1] = yLoc;
            stateVector[counter][2] = exitNum;
            stateVector[counter][3] = currentSpeed;

            counter++;
        }
        assert(currentState.size() == stateVector.length);

        return stateVector;
    }


    public static List<Person> update(List<Person> currentState) {
        System.out.println("UPDATING");

        // Find all the people
        Bag people = truthModel.area.getAllObjects();

        // Remove all old people
        for (Object o : people) {
            truthModel.area.remove(o);
        }

        // Add people from the currentState
        for (Object o : currentState) {
            Person p = (Person) o;
            truthModel.area.setObjectLocation(p, p.getLocation()); // DOES THIS ADD A NEW AGENT AND MOVE THEM AT THE SAME TIME??
        }

        // Step all the people
        truthModel.schedule.step(truthModel);

        /*
        // Return them
        List<Person> personList = ArrayList<>();
        for (Object o : station.area.getAllObjects()) {
            personList.add((Person) o);
        }
        */

        List<Person> personList = new ArrayList<>();
        truthModel.area.getAllObjects().stream().collect(Collectors.toCollection(() -> personList));
        //List<Person> personList = Collections.synchronizedList(new ArrayList<>(Arrays.asList(truthModel.area.getAllObjects())));

        assert(people.size() == personList.size());

        return personList;
    }


    public static void main(String[] args) {

        StationTransition.runDataAssimilation();
        System.out.println("Happy days, runDataAssimilation has run properly!");

        System.out.println(stateVector);

        System.exit(0);
    }
}
