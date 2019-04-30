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
import org.jetbrains.annotations.NotNull;
import sim.engine.SimState;
import sim.util.Bag;
import sim.util.Double2D;

/**
 * An agent that moves toward a set target while interacting with other Person agents along the way.
 */
public class Person extends Agent implements Comparable<Person> {
    private static final long serialVersionUID = 1;

    private Station station;

    public Entrance entrance;
    public Exit exit;
    private double desiredSpeed;
    private double minSpeed = 0.05;
    private double speedMultiplier = 1.0;
    private double radius;
    private double currentSpeed;
    private boolean active = true; // Whether or not agents take part in the simulation. One constructor makes this false.
    private int id; // agents will have incrementally increasing IDs

    public double[][] exitProbs = {{0.2, 0.8},
                                {0.3, 0.7},
                                {0.9, 0.1}};


    /**
     * Used for creating inactive agents. These are used so that all agents can be created initially, but when
     * we actually need an agent this inactive agent will be deleted and a new one will be created one of the other
     * available constructors. Agents created using this constructor have active=false so do nothing when their step()
     * method is called.
     */
    Person(int size, Double2D location, String name, Exit exit, int id) {
        super(size, location, name);
        this.active = false;
        this.desiredSpeed = 0;
        radius = size / 2.0;
        this.exit = exit;
        // Give inactive agents unique ID
        this.id = id; // This is the only constructor including IDs. ID's come from ID_Counter in Station.class
    }

    void makeInactive(Station station, String name) {
        this.active = false;
        this.name = name;
        this.desiredSpeed = 0;
        station.area.setObjectLocation(this, new Double2D(0d,0d));
    }

    void makeActive(Double2D location, Station station, Exit exit, Entrance entrance) {
        this.active = true;
        this.name = "Person";
        this.location = location;
        this.station = station;
        this.exit = exit;
        this.entrance = entrance;
        desiredSpeed = station.random.nextDouble() + minSpeed;
        station.numRandoms++;
    }


    Person(int size, Double2D location, String name, Station station, double[] exitProbs, Entrance entrance) {
        super(size, location, name);
        this.station = station;
        this.entrance = entrance;
        radius = size / 2.0;
        desiredSpeed = station.random.nextDouble() + minSpeed;
        desiredSpeed = nextExponential(1.0) + minSpeed;
        // System.out.println(desiredSpeed + ",");
        station.numRandoms++;
        currentSpeed = 0.0;
        //this.id = StationSim.Person.ID_Counter++; // This isn't necessary here (But might be useful later)

        double randDouble = station.random.nextDouble();
        station.numRandoms++;
        double cumulativeProb = 0.0;
        for (int i = 0; i < exitProbs.length; i++) {
            if(randDouble < exitProbs[i] + cumulativeProb) {
                exit = station.exits.get(i);
            } else {
                cumulativeProb += exitProbs[i];
            }
        }
    }


    // New Person constructor to accept Exit object instead of exitProbs. Easier to build state vector with.
    Person(int size, Double2D location, String name, Station station, Exit exit, Entrance entrance) {
        super(size, location, name);
        this.station = station;
        this.entrance = entrance;
        this.exit = exit;
        radius = size / 2.0;
        desiredSpeed = station.random.nextDouble() + minSpeed;
        station.numRandoms++;
        currentSpeed = 0.0;
    }


    // Constructor with desiredSpeed included
    Person(int size, Double2D location, String name, Station station, Exit exit, Entrance entrance, double desiredSpeed) {
        this(size, location, name, station, exit, entrance); // Use the other Person constructor, saves on code repetition
        this.desiredSpeed = desiredSpeed;
        radius = size / 2.0;
    }


    /** Moves the Person closer to their exit and interacts with other Person agents if necessary.
     * @param state Current sim state
     */
    @Override
    public void step(SimState state) {
        if (!this.active) { // If the person has not been activated yet then don't do anything
            return;
        }
        assert this.active ;
        station = (Station) state;

        Double2D newLocation;
        double speed;

        // find how fast agent can move
        speed = desiredSpeed;
        int slowingDistance = 15;
        for (int i = 0; i < slowingDistance; i ++) {
            newLocation = lerp(getLocation(), station.doorways.getObjectLocation(exit), speed, distanceToExit());
            if (collision(newLocation)) {
                speed *= ((slowingDistance - i) / slowingDistance) * 0.1; // Sort out magic number
                break;
            }
        }

        // find agent's new location
        newLocation = lerp(getLocation(), station.doorways.getObjectLocation(exit), speed, distanceToExit());

        // adjust so no wall collision
        double wallBarrier = station.getAreaWidth() - (station.getWallWidth() * 2) - getRadius();
        if (location.getX() > wallBarrier) {
            newLocation =  new Double2D(wallBarrier, location.getY());
        }

        // if new location collides with other agent, attempt to move around
        //int i = 0;
        //int attempts = 3;
        //while (collision(newLocation) &&  i < attempts) {
        //    newLocation = new Double2D(getLocation().getX(),
        //            getLocation().getY() + (station.random.nextDouble() - 0.5) * desiredSpeed);
        //    station.numRandoms++;
        //    i++;
        //}

        // if new location collides with other agent, attempt to move around
        // Think of a more elegant way to code this
        if(collision(newLocation)) {
            double randNum = station.random.nextDouble();
            if (randNum > 0.5) {
                newLocation = new Double2D(getLocation().getX(),
                    getLocation().getY() + randNum);
                if(collision(newLocation)) {
                    newLocation = new Double2D(getLocation().getX(),
                        (getLocation().getY() - randNum));
                }
            }
            else {
                newLocation = new Double2D(getLocation().getX(),
                    getLocation().getY() - randNum);
                if(collision(newLocation)) {
                    newLocation = new Double2D(getLocation().getX(),
                        (getLocation().getY() + randNum));
                }
            }
        }
        // Update location (or not)
        if(!collision(newLocation)) {
            currentSpeed = getDistance(getLocation(), newLocation);
            location = newLocation;
        }
        station.area.setObjectLocation(this, location);
    }

    /** Linear interpolation at constant rate. Calculates coordinate for agent to move
     * at constant rate towards target.
     * @param p1 Point one is typically a location of this agent
     * @param p2 Point two is typically a location of the chosen exit
     * @param speed The current speed of the person
     * @param distance The euclidean distance to chosen exit
     * @return Updated location
     */
    private Double2D lerp(Double2D p1, Double2D p2, double speed, double distance) {
        double x = p1.getX() + (speed / distance) * (p2.getX() - p1.getX());
        double y = p1.getY() + (speed / distance) * (p2.getY() - p1.getY());
        return new Double2D(x, y);
    }

    /**
     * Check for collision with any other people at a given location
     * @param location Test location for this object
     * @return Whether this object will intersect with any other people at the given location
     */
    boolean collision(Double2D location) {
        StationSim.Person p;
        Bag people = station.area.getNeighborsWithinDistance(location, radius * 5);
        if (people != null) {
            for (int i = 0; i < people.size(); i++) {
                p = (StationSim.Person) people.get(i);
                if (p != this && intersect(location, p)) { //check if evaluation is lazy
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * Check if two people(this one at a given location aon another at its current location) intersect.
     * Typically used to check for collision before location assignment to this object.
     * @param location The location of this peron to test
     * @param other Another person with this current location in the simulation
     * @return Whether the two people will intersect
     */
    private boolean intersect(Double2D location, StationSim.Person other) {
        double distX = location.getX() - other.getLocation().getX();
        double distY = location.getY() - other.getLocation().getY();

        double distSq = distX * distX + distY * distY;
        double radii = getRadius() + other.getRadius();

        return distSq <= radii * radii;
    }

    /** Calculate Euclidean distance to target exit.
     * @return The distance to the chosen exit of this person
     */
    public double distanceToExit() {
        if (!active) {// Inactive agents don't have an exit yet, so just give them an arbitrary large distance
            return Double.MAX_VALUE;
        }
        return getDistance(this.getLocation(), exit.getLocation());
    }

    /**
     * Calculate Euclidean distance between two points
     * @param a point a
     * @param b point b
     * @return
     */
    private double getDistance(Double2D a, Double2D b) {
        return Math.hypot(a.getX() - b.getX(), a.getY() - b.getY());
    }

    private double nextExponential(double lambda) {
        return  Math.log(1 - station.random.nextDouble()) / (-lambda);
    }



    double getCurrentSpeed() {
        return currentSpeed;
    }

    void setCurrentSpeed(double currentSpeed) { this.currentSpeed = currentSpeed; }

    Station getStation() { return station; }

    double getDesiredSpeed() { return desiredSpeed; }

    int getID() { return id; }

    public Exit getExit() {
        return exit;
    }

    double getRadius() {
        return radius;
    }

    boolean isActive() { return active; }

    @Override
    public String toString() {
        return "["+ this.id+"]"+this.name;
    }

    @Override
    public boolean equals(Object o) {
        // If the object is compared with itself then return true
        if (o == this) {
            return true;
        }

        /* Check if o is an instance of Person or not
          "null instanceof [type]" also returns false */
        if (!(o instanceof StationSim.Person)) {
            return false;
        }

        // typecast o to Complex so that we can compare data members
        StationSim.Person p = (StationSim.Person) o;
        return this.id == p.id;
    }

    @Override
    public int compareTo(@NotNull Person anotherPerson) {
        return this.getID() - anotherPerson.getID();
    }
}
