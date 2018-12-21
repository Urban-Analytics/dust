package StationSim;

import sim.util.Bag;

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

        // Run the truth model
        truthModel.start();

        // XXXX RUN UNTIL FINISHED (doLoop??)
        truthModel.doLoop(Station.class, args);


        // Start data assimilation window
            // update
            // predict
            // for 1000 iterations

        // Start data assimilation window
        for (int iter = 0; iter < NUM_ITER; iter++) {

            // ****************** Update ******************
            Bag people = truthModel.area.getAllObjects();

            List<Person> currentState = new ArrayList<>();
            currentState.addAll(people);

            update(currentState);

            // predict

            // for 1000 iterations

        }
    }


    public static List<Person> update(List<Person> currentState) {

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

    public static void main(String[] args) {

        StationTransition.runDataAssimilation(args);

    }
}
