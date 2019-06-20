package StationSim;

import sim.engine.Sequence;
import sim.engine.SimState;
import sim.engine.Steppable;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;

/** Contains people agents for putting on the schedule.
 *  People are ordered in the sequence by distance to exit.
 *  People closest to the exit are added first. The sequence is
 *  updated each step.
 */
public class Sequencer implements Steppable {

    Sequence peopleSequence;

    public Sequencer(Station station) {
        peopleSequence = new Sequence(new ArrayList());
        station.schedule.scheduleRepeating(peopleSequence, 2, 1.0);
    }

    /** Replace sequence with new ordering. People closet to the exit are added first.
     * @param state Current sim state
     */
    @Override
    public void step(SimState state) {
        Station station = (Station) state;

        ArrayList<Person> people = new ArrayList<Person>(station.area.getAllObjects()); // Check for best collection to use
        Collections.sort(people , new Comparator<Person>() {
            @Override
            public int compare(Person a, Person b) {
                return Double.compare(b.distanceToExit(), a.distanceToExit());
            }
        });
        peopleSequence.replaceSteppables(people);
    }
}
