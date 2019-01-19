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

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import sim.engine.SimState;
import sim.engine.Steppable;
import sim.util.Double2D;

/** Base class for agents used in StationSim
 */
public abstract class Agent implements Steppable {

    private static final long serialVersionUID = 1;

    // Old location
    protected Double2D location;

    // New location
    protected DoubleVertex xLoc;
    protected DoubleVertex yLoc;

    protected int size;
    protected Station station;
    protected String name;

    /*
    public Agent(int size, Double2D location, String name) {
        this.size = size;
        this.location = location;
        this.name = name;
    }
    */


    public Agent(int size, DoubleVertex xLoc, DoubleVertex yLoc, String name) {
        this.size = size;
        this.xLoc = xLoc;
        this.yLoc = yLoc;
        this.name = name;
    }


    public Double2D getLocation() {
        return location;
    }

    /*
    Moved this method to Person to be overridden, just in case replacing this method in agent caused any trouble

    public Double2D getLocation() {
        double xVal = xLoc.getValue(0);
        double yVal = yLoc.getValue(0);
        Double2D position = new Double2D(xVal, yVal);
        return position;
    }
    */

    @Override
    public void step(SimState state) {
        station = (Station) state;
    }

    @Override
    public String toString() {
        return name;
    }


}
