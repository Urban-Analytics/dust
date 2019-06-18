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
import sim.util.Bag;
import sim.util.Double2D;


/** Serves as a target for persons to move towards to. Removes a person from the simulation when
 *  they reach the exit. People are removed every n steps dictated by exitInterval. The maximum
 *  number of people removed is dictated by the exit size.
 */
public class Exit extends Agent {


    private int exitInterval;
    public int totalRemoved;

    // Added explicitly assigned exitNumber to help DataAssimilation
    public int exitNumber;


    public Exit(int size, Double2D location, String name, int exitInterval) {
        super(size, location, name);
        this.exitInterval = exitInterval;
        this.totalRemoved = 0;
    }

    public Exit(int size, Double2D location, String name, int exitInterval, int exitNumber) {
        super(size, location, name);
        this.exitInterval = exitInterval;
        this.totalRemoved = 0;
        this.exitNumber = exitNumber;
    }


    /** Removes person agents from the simulation that are touching the exit. The number
     * of people that can be removed per step is bounded by the exit size.
     * @param state
     */
    @Override
    public void step(SimState state) {
        super.step(state);
        Double2D personLocation;
        Bag people;

        if (station.schedule.getSteps() % exitInterval == 0) {
            people = station.area.getAllObjects();
            for (int i = 0; i < people.size(); i++) {
                Person p = (Person) people.get(i);
                personLocation = p.getLocation();
                // If a person is in front of exit then remove
                if (personLocation.getX() >= location.getX() - (station.wallWidth * 2.0) - (p.getRadius() * 2)  &&
                        personLocation.getY() < (location.getY() + (size / 2.0)) && // check this (size / 2.0)
                                personLocation.getY() > (location.getY() - (size / 2.0)) &&
                                    p.isActive()) {
                    assert (p.isActive()) : "Person to be removed is inactive! This is very wrong";
                    station.finishedPeople.add(p); // Put into bag of finished people. This is used in the analysis
                    //station.inactivePeople.add(p); // Make them inactive
                    station.activeNum--; // reduce activeNum for housekeeping
                    p.makeInactive(station, "Previously Active");
                    //station.area.remove(p); //remove from sim
                    totalRemoved++;
                }
            }
        }
    }
}
