/* Created by Micahel Adcock on 17/04/2018.
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
import sim.util.Double2D;

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

    public Entrance(int size, Double2D location, String name, int numPeople, double[] exitProbs, SimState state) {
        super(size, location, name);
        Station station = (Station) state;
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
                Person person = new Person(personSize, spawnLocation, "Person: " + (station.addedCount + 1), station, exitProbs, this);
                if (!person.collision(spawnLocation)) {
                    station.area.setObjectLocation(person, spawnLocation);
                    addedCount++;
                    station.addedCount++;
                }
            }
            // Number of people left for further steps
            totalAdded += addedCount;
            numPeople -= addedCount;
        }
    }
}
