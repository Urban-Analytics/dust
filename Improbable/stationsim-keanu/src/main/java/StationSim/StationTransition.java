package StationSim;

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

    public static void runDataAssimilation(String[] args) {

        System.out.println("Does any of this run?");
        // Run the truth model
        truthModel.start();

        System.out.println("truthModel.start() has executed successfully");

        // XXXX RUN UNTIL FINISHED (doLoop??)
        truthModel.doLoop(Station.class, args);

        System.out.println("StationTransition doLoop finished");

        //truthModel.finish();

        // Start data assimilation window
            // update
            // predict
            // for 1000 iterations

        System.out.println("Starting DA window");
        // Start data assimilation window
        for (int iter = 0; iter < NUM_ITER; iter++) {

            System.out.println("Entered DA iteration " + iter);
            // ****************** Update ******************
            Bag people = truthModel.area.getAllObjects();

            List<Person> currentState = new ArrayList<>(people);

            currentState = update(currentState);

            // predict
            predict(currentState);

            // for 1000 iterations

        }
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
        //List<Person> personList = (List<Person>) truthModel.area.getAllObjects().stream().collect(Collectors.toCollection(() -> personList));

        return personList;
    }


    public static void predict(List<Person> currentState) {
        System.out.println("PREDICTING");

        List<Double> stateVector = new ArrayList<>();

        // Build state vector [[x, y, exit, currentSpeed],...]
        for (Person person : currentState) {
            Double2D loc = person.getLocation();
            Double xLoc = loc.x;
            Double yLoc = loc.y;
            Double currentSpeed = person.getCurrentSpeed();

            Exit exit = person.exit;
            String exitName = exit.name;
            System.out.println(exitName);
        }


    }

    public static void main(String[] args) {

        StationTransition.runDataAssimilation(args);
        System.out.println("Happy days, runDataAssimilation has run properly!");

        System.exit(0);
    }
}
