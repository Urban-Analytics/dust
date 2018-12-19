package StationSim;

import sim.util.Bag;

import java.util.ArrayList;
import java.util.List;


public class StationTransition {

    public static List<Person> update(List<Person> currentState) {

        // Initialise a new model. (In practice, is there a way to avoid calling these every time?)
        Station s = new Station(System.currentTimeMillis());
        s.start();

        // Find all the people
        Bag people = s.area.getAllObjects();

        // Remove all old people
        for (Object o : people) {
            s.area.remove(o);
        }

        // Add people from the currentState
        for (Object o : currentState) {
            Person p = (Person) o;
            s.area.setObjectLocation(p, p.getLocation()); // DOES THIS ADD A NEW AGENT AND MOVE THEM AT THE SAME TIME??
        }

        // Step all the people
        s.schedule.step(s);

        // Return them
        List<Person> l = new ArrayList<>();
        for (Object o : s.area.getAllObjects()) {
            l.add((Person)o);
        }
        // (A challenge: use stream() to do the for loop above)
        return l;


    }

    public static void main(String[] args) {
        XXXX

                // for 1000 iterations
    }

}
