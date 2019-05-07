package StationSim;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.jetbrains.annotations.NotNull;
import sim.engine.SimState;
import sim.util.Bag;
import sim.util.Double2D;

public class AssimilationPerson extends Agent implements Comparable<AssimilationPerson> {

    private static final long serialVersionUID = 1;

    private Station station;

    public Entrance entrance;
    public Exit exit;

    private DoubleVertex xPosition;
    private DoubleVertex yPosition;
    private DoubleVertex desiredSpeed; // DoubleVertex here? Or keep desiredSpeed as double and currentSpeed as DoubleVert? Is that even possible with MASON running the show?

    private double minSpeed = 0.05;
    private double speedMultiplier = 1.0;
    private double radius;
    private double currentSpeed;
    private boolean active = true; // Whether or not agents take part in the simulation. One constructor makes this false.
    private int id; // agents will have incrementally increasing IDs


    /**
     * Used for creating inactive agents. These are used so that all agents can be created initially, but when
     * we actually need an agent this inactive agent will be deleted and a new one will be created one of the other
     * available constructors. Agents created using this constructor have active=false so do nothing when their step()
     * method is called.
     */
    AssimilationPerson(int size, DoubleVertex xPos, DoubleVertex yPos, String name, Exit exit, int id) {
        // Create Double2D object from xPos and yPos and pass as location
        super(size, new Double2D(xPos.getValue().sum(), yPos.getValue().sum()), name);

        this.active = false;
        this.desiredSpeed = new GaussianVertex(0.0, 1.0); // speed set as gaussian now?
        this.xPosition = xPos;
        this.yPosition = yPos;
        radius = size / 2.0;
        this.exit = exit;

        // Give inactive agents unique ID
        this.id = id; // This is the only constructor including IDs. ID's come from ID_Counter in Station.class
    }

    void makeInactive(Station station, String name) {
        this.active = false;
        this.name = name;
        this.desiredSpeed = new GaussianVertex(0.0, 1.0);
        station.area.setObjectLocation(this, new Double2D(0d,0d));
    }

    void makeActive(Double2D location, Station station, Exit exit, Entrance entrance) {
        this.active = true;
        this.name = "Person";
        this.location = location;
        this.xPosition = new GaussianVertex(location.x, 1.0);
        this.yPosition = new GaussianVertex(location.y, 1.0);
        this.station = station;
        this.exit = exit;
        this.entrance = entrance;
        desiredSpeed = new GaussianVertex(station.random.nextDouble() + minSpeed, 1.0);
        station.numRandoms++;
    }


    /** Moves the Person closer to their exit and interacts with other Person agents if necessary.
     * @param state Current sim state
     */
    @Override
    public void step(SimState state) {
        if (!this.active) { // If the person has not been activated yet then don't do anything
            return;
        }

        XXXX // Copy over step() logic from Person.class, modify for Keanus algorithms
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

        XXXX // should be easy to collect DoubleVertices of each agent instead of Double2D
        // Then recreate the logic below using DoubleVertices instead of doubles
        // Can't see why (at the minute) it wouldn't just be a straight conversion from
        // x + s/d  -->  x.plus(speed.divideBy(distance) (All Keanu vertex(?) methods)

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
        StationSim.AssimilationPerson p;
        // Need to leave this line as is, need to use MASONs getNeighbors...() using Double2D location
        Bag people = station.area.getNeighborsWithinDistance(location, radius * 5);
        if (people != null) {
            for (int i = 0; i < people.size(); i++) {
                p = (StationSim.AssimilationPerson) people.get(i);
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

        XXXX // again just converting this to take DoubleVertex locations and StationSim.AssimilationPerson
        //  Then just changing + and - into .plus() and .minus() for distributions

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

        XXXX // uses trigonometry to calculate distance between 2 points
        // Should be easy to convert to Keanu algo's inside the brackets
        // Math.hypot might be more difficult, probably requires double value inputs
        // Possible to recreate the hypotenuse stuff using Keanu? Or replace with dot product

        return Math.hypot(a.getX() - b.getX(), a.getY() - b.getY());
    }

    private double nextExponential(double lambda) {

        XXXX // dont really know what this does, never changed it or came across anywhere that used
        // it so I imagine leaving it alone will be ok

        return  Math.log(1 - station.random.nextDouble()) / (-lambda);
    }


    // This method has to be included when class implements Comparable<>
    // This is to allow agents to be compared using their unique ID number, and helps to
    // put agents in order before adding them into the model
    @Override
    public int compareTo(@NotNull AssimilationPerson anotherPerson) {
        return this.getID() - anotherPerson.getID();
    }
}

