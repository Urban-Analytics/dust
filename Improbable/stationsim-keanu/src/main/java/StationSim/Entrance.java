/* Created by Michael Adcock on 17/04/2018.
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

import sim.engine.SimState;
import sim.util.Bag;
import sim.util.Double2D;

import java.util.HashSet;
import java.util.Set;

/**
 * An entrance that spawns n people per time step based on the size of the Entrance.
 * A target exit is assigned to a person to move toward at spawn time.
 */
public class Entrance extends Agent {

    private int personSize;
    public int numPeople;
    //private double buffer = 0.2;
    private int entranceInterval; // How often are Person objects produced

    // Exit for people agents to aim for
    public Exit exit;
    public double[] exitProbs;
    public int totalAdded;



    public Entrance(int size, Double2D location, String name, int numPeople, double[] exitProbs, Station state) {
        super(size, location, name);
        Station station = state;
        this.entranceInterval = station.getEntranceInterval();
        this.personSize = station.getPersonSize();
        this.size *= personSize;
        this.numPeople = numPeople;
        this.exitProbs = exitProbs;
        this.totalAdded = 0;

    }

    public Exit getExit() {
        return exit;
    }

    /**
     * Generates a set number of person agents per step and assigns them an exit to aim for.
     * The number of people that can be generated is finite.
     * @param state The current state of the simulation
     */
    @Override
    public void step(SimState state) {
        super.step(state);

        // Print the agents for testing
        /*
        System.out.print("Ticks:"+ state.schedule.getSteps() + " -\n\t" +
                "Num inactive: "+ inactivePeople.size()  + " -\n\t" +
                "Total num agents:: "+ ((Station)state).area.getAllObjects().size()  + " -\n\t"
                //Entrance.inactivePeople.toString() + "\n\t"
        );
        for (Object o : ((Station)state).area.getAllObjects()) {
            System.out.print(((Person) o).toString() + " ");
        }
        System.out.println();
        */

        // First check if this is a Person generating step
        if (station.schedule.getSteps() % entranceInterval == 0) {
            // Set number of people to be generated
            int toEnter = numPeople;
            if (toEnter > size) {
                toEnter = size;
            }
            int addedCount = 0;
            // Generate people agents to pass through entrance and set as stoppables.
            for (int i = 0; i < toEnter; i++) {
                double x = location.getX();
                double y = location.getY();
                Double2D spawnLocation = new Double2D(x + 1,
                        (y + i) - ((size / 2.0) - personSize / 2.0)); // need to add a buffer

                // Assign exit from exit probs
                double randDouble = station.random.nextDoubleNonZero();
                station.numRandoms++;
                double cumulativeProb = 0.0;
                for (int j = 0; j < exitProbs.length; j++) {
                    if(randDouble < exitProbs[j] + cumulativeProb) {
                        exit = station.exits.get(j);
                    } else {
                        cumulativeProb += exitProbs[j];
                    }
                }
                // Use new person constructor to assign exit and not exitProbability
                //Person person = new Person(personSize, spawnLocation, "Person: " + (station.addedCount + 1), station, exitProbs, this);
                Person person = new Person(personSize, spawnLocation, "Person: " + (station.addedCount + 1), station, exit, this);
                if (!person.collision(spawnLocation)) {
                    // add the person to the model
                    station.area.setObjectLocation(person, spawnLocation);
                    addedCount++;
                    station.addedCount++;

                    // remove an inactive agent from the model and the bank of inactive agents
                    Station s = ((Station)state);
                    Person inactive = s.inactivePeople.iterator().next();


                    /**
                     * Tried this initially, realised that the problem was with the link between Station.inactivePeople
                     * and building personList in StationTransition2.rebuildPersonList()
                     */
                    // Check if any inactive agents left before looping through objects (to save time)
                    /*if (s.inactivePeople.size() > 0) {
                        // Loop through and remove an inactive agent so total stays at 700
                        for (Object inTheModel : s.area.getAllObjects()) {
                            if (!((Person) inTheModel).isActive()) { // Cast inTheModel to Person and check if active
                                s.area.remove(inTheModel);
                                break;
                            }
                        }
                    }*/

                    //s.area.remove(inactive); // Null, inactive never == object in model after personList is built in predict

                    /**
                     * With personList now linked to Set of inactiveAgents (and being built from it) it is only
                     * necessary to remove the inactive people from the Set
                     */
                    s.inactivePeople.remove(inactive); // agent removed from inactive agents set in station

                    //System.out.println("Number of objects in s: " + s.area.getAllObjects().size());

                }
            }
            // Number of people left for further steps
            totalAdded += addedCount;
            numPeople -= addedCount;
        }
    }
}
