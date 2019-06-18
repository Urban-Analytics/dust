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
import sim.engine.Steppable;
import sim.util.Double2D;

/** Base class for agents used in StationSim
 */
public abstract class Agent implements Steppable {

    private static final long serialVersionUID = 1;

    // location
    Double2D location;

    protected int size;
    protected Station station;
    String name;


    Agent(int size, Double2D location, String name) {
        this.size = size;
        this.location = location;
        this.name = name;
    }


    Double2D getLocation() {
        return location;
    }

    @Override
    public void step(SimState state) {
        station = (Station) state;
    }

    @Override
    public String toString() {
        return name;
    }


}
