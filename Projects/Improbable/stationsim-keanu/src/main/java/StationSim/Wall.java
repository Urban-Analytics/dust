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

import sim.util.Double2D;

/**
 * Simple class defining a wall in the simulation. Collision is dealt with in Person class
 */
public class Wall {

    private static final long serialVersionUID = 1;

    Double2D location;
    double width;
    double height;

    public Wall(Double2D location, double width, double height) {
        this.location = location;
        this.width = width;
        this.height = height;
    }

    public Double2D getLocation() {
        return location;
    }

    public double getWidth() {
        return width;
    }

    public double getHeight() {
        return height;
    }


}


