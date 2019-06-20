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

import io.improbable.keanu.KeanuRandom;
import sim.engine.SimState;
import sim.field.continuous.Continuous2D;
import sim.util.Bag;
import sim.util.Double2D;
import sim.util.Interval;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Simulates a very simple train station with a given number of entrances and exits which are set through
 * GUI or command line arguments. People move from the entrance to a given exit, interacting on the way.
 */
public class Station extends SimState {

    public int numRandoms = 0;

    private static final long serialVersionUID = 1;
    public static int modelCount = 0 ; // Record the total number of models created.

    // Representations of simulation space
    private final double areaWidth = 200.0;
    private final double areaHeight = 100.0;
    int wallWidth = 2;
    double wallHeight;
    public Continuous2D area = new Continuous2D(1.0, areaWidth, areaHeight);
    public Continuous2D doorways = new Continuous2D(1.0, areaWidth, areaHeight);
    public Continuous2D walls = new Continuous2D(1.0, areaWidth, areaHeight);

    // A bank of people who are inactive. Everyone is created at the beginning, but they are inactive until
    // the entrance activates them
    List<Person> inactivePeople = null;

    // Default values for parameters
    private final int numPeople = 700;
    private final int numEntrances = 3;
    private final int numExits = 2;
    public double[][] exitProbs = {{0.2, 0.8},
                                    {0.3, 0.7},
                                    {0.9, 0.1}};
    private int exitInterval = 30;
    private int entranceInterval = 2;
    private int entranceSize = 10;
    private int exitSize = 10;
    private int personSize = 1; // sort bug here
    public int addedCount;
    private boolean writeResults = false;

    private int ID_Counter = 0; // To assign unique IDs for each agent

    //public RandomGenerator random;
    public KeanuRandom random;
    //public RandomDoubleFactory random;

    //Bag of people who have finished the simulation
    public Bag finishedPeople = new Bag();

    int activeNum = 0;

    // Store  exit agents for passing to constructors
    public ArrayList<Exit> exits;
    public ArrayList<Entrance> entrances;

    //Analysis agent
    public Analysis analysis;

    // Run with default settings only
    private boolean hideParameters = false;


    public Station(long seed) {
        super(seed);
    }


    /**
     * Start a simulation. This is the version that is called directly from the Wrapper and uses the
     * random factory supplied as a parameter.
     */
    public void start(KeanuRandom rand) {
        System.out.println("Model "+ StationSim.Station.modelCount++ +" starting");
        super.start();
        random = rand;
        area.clear();
        doorways.clear();
        walls.clear();
        addedCount = 0;

        createWalls();
        createExits();
        createEntrances();
        createInactivePeople();

        // This changes when number of exits of size of exits do
        wallHeight = (areaHeight / (numExits + 1)) - (exitSize * 2.0  * personSize / 2.0);

        // The sequencer will contain all the person objects on the schedule and order them in distance to exit.
        Sequencer sequencer = new Sequencer(this);
        schedule.scheduleRepeating(sequencer, 1, 1.0);

        // Analysis and outputs from the model are contained in this agent
        analysis = new Analysis(this);
        schedule.scheduleRepeating(analysis, 3, 1.0);
    }


    /**
     * Start the simulation. This is the version that is called from the GUI or Station.main() methods. It needs to
     * create its own random factory
     */
    @Override
    public void start() {
        System.out.println("WARNING! This should not be called unless using the GUI.");
        super.start();
        area.clear();
        doorways.clear();
        walls.clear();
        addedCount = 0;
        //random = new VertexBackedRandomGenerator(Wrapper.numRandomDoubles, 0, 0);
        random = new KeanuRandom();
        numRandoms = 0;

        createWalls();
        createExits();
        createEntrances();
        createInactivePeople();

        // This changes when number of exits of size of exits do
        wallHeight = (areaHeight / (numExits + 1)) - (exitSize * 2.0  * personSize / 2.0);

        // The sequencer will contain all the person objects on the schedule and order them in distance to exit.
        Sequencer sequencer = new Sequencer(this);
        schedule.scheduleRepeating(sequencer, 1, 1.0);

        // Analysis and outputs from the model are contained in this agent
        analysis = new Analysis(this);
        schedule.scheduleRepeating(analysis, 3, 1.0);
    }


    /** Call appropriate analysis methods and cleans up
     */
    @Override
    public void finish() {
        super.finish();
        System.out.println("Number of random draws: " + numRandoms);
    }


    /** Setup walls on the right side of the simulation. These the position and size of these are
     * change based on the number of exits and size of the exits.
     */
    public void createWalls() {
        Double2D pos;
        Wall wall;

        for (int i = 0; i < numExits + 1; i++) {
            pos = new Double2D(areaWidth - (wallWidth), areaHeight / (numExits * 2 + 2) * (i * 2 + 1));
            wall = new Wall(pos, wallWidth / 2, wallHeight /2);
            walls.setObjectLocation(wall, wall.getLocation());

        }

        // back corners - These cover gaps in the top and bottom corners
        pos = new Double2D(areaWidth - (wallWidth), exitSize / 1.5);
        wall = new Wall(pos, wallWidth, exitSize + 1);
        walls.setObjectLocation(wall, wall.getLocation());

        pos = new Double2D(areaWidth - (wallWidth), areaHeight - (exitSize / 1.5));
        wall = new Wall(pos, wallWidth, exitSize + 1); //exitSize  * personSize / 2
        walls.setObjectLocation(wall, wall.getLocation());
    }


    /** Creates entrance agents based on the number of entrances and size of entrances set in GUI.
     * The size of the entrance represents how many people can pass through it per step (i.e. the number
     * of person agents that are generated by an entrance per step). People are evenly assigned to the
     * entrances using integer division (remainders discarded). Entrances are evenly spaced on left side
     * of the simulation.
     */
    private void createEntrances() {
        double interval, y;
        //System.out.println("Creating Entrances");
        entrances = new ArrayList<>();
        // Calculate even spacing of entrances
        interval = areaHeight / (numEntrances + 1);
        y = 0;

        // Create entrances
        for (int i = 0; i < numEntrances; i++) {
            y += interval;
            Double2D location = new Double2D(0, y);
            Entrance entrance =  new Entrance(entranceSize, location, "Entrance: " + (i + 1),
                    numPeople / numEntrances, exitProbs[i], this);
            doorways.setObjectLocation(entrance, location);
            schedule.scheduleRepeating(entrance, 2, 1.0);
            entrances.add(entrance);
        }

    }


    /** Creates exit agents based on the number of exits and size of exits set in GUI.
     * The size of the entrance represents how many people can pass through it per step
     * (i.e. The number of person agents that can be removed from the simulation by an
     * exit per step). Exits are evenly spaced on right side of simulation.
     */
    private void createExits() {
        double intervalSpacer, y;
        Double2D position;

        exits = new ArrayList<>();
        // Calculate even spacing for exits
        intervalSpacer = areaHeight / (numExits + 1);
        y = 0;

        // Create exits
        for (int i = 0; i < numExits; i++) {
            y += intervalSpacer;
            position = new Double2D(areaWidth, y);
            //Exit exit =  new Exit(exitSize, position, "Exit: " + (i + 1), exitInterval);
            Exit exit =  new Exit(exitSize, position, "Exit: " + (i + 1), exitInterval, (i+1));
            doorways.setObjectLocation(exit, position);
            schedule.scheduleRepeating(exit, 0, 1.0);
            exits.add(exit);
        }
    }


    /** Creates inactive agents at the start of the simulation run. This is to ensure the
     * state vector does not change length throughout the data assimilation windows in DataAssimilation.java
     */
    private void createInactivePeople() {
        // Create all of the people (unless another entrance has done this already

        System.out.print("Populating inactive set: ");
        this.inactivePeople = new ArrayList<>();
        for (int i=0; i<(this.getNumPeople()); i++) {
            Double2D spawnLocation = new Double2D(0.0, 0.0);
            Exit exit = this.getExits().get(0); // Assign all inactive agents to first exit (Will change when activated)
            int id = ID_Counter++;
            Person person = new Person(personSize, spawnLocation, "Inactive Person", exit, id);
            this.area.setObjectLocation(person, spawnLocation);
            this.inactivePeople.add(person);
        }
        System.out.println("... created "+this.inactivePeople.size()+" inactive agents. " +
                "There are "+this.getNumPeople() + " agents in the model.");
    }


    // Lots of getters and setters use hideParameters variable to hide parameters from GUI
    public final int getNumPeople() {
        return numPeople;
    }

    public String desNumPeople() {
        return "Total number of people to enter simulation split evenly between all entrances";
    }

    /*public void setNumPeople(int people) {
        numPeople = people;
    } */

    public Object domNumPeople() {
        return new Interval(1, 10000);
    }

    public boolean hideNumPeople() {
        return hideParameters;
    }

    public int getNumEntrances() {
        return numEntrances;
    }

    /*public void setNumEntrances(int entrances) {
        numEntrances = entrances;
    }*/

    public boolean hideNumEntrances() {
        return hideParameters;
    }

    public Object domNumEntrances() {
        return new Interval(2, 10);
    }

    public int getNumExits() {
        return numExits;
    }

    /*public void setNumExits(int exits) {
        numExits = exits;
    }*/

    public boolean hideNumExits() {
        return hideParameters;
    }

    public Object domNumExits() {
        return new Interval(2, 10);
    }

    public int getEntranceSize() {
        return entranceSize;
    }

    public void setEntranceSize(int size) {
        entranceSize = size;
    }

    public boolean hideEntranceSize() {
        return hideParameters;
    }

    public Object domEntranceSize() {
        return new Interval(1, 10);
    }

    public String desEntranceSize() {
        return "Number of people who can come through each entrance per step";
    }

    public int getExitSize() {
        return exitSize;
    }

    public void setExitSize(int size) {
        exitSize = size;
    }

    public String desExitSize() {
        return "Number of people who can pass through each exit per step";
    }

    public boolean hideExitSize() {
        return hideParameters;
    }

    public Object domExitSize() {
        return new Interval(1, 10);
    }

    public double[][] getExitProbs() {
        return exitProbs;
    }

    //public void setExitProb(double exitProb) {
    //    this.exitProb = exitProb;
    //}

    public boolean hideExitProb() {
        return hideParameters;
    }

    public Object domExitProb() {
        return new Interval(0.01, 1.0);
    }

    public String desExitProb() {
        return "Probability that person will choose exit assigned to them";
    }

    public int getPersonSize() {
        return personSize;
    }

    public void setPersonSize(int size) {
        personSize = size;
    }

    // Set to always true at the moment as there is a bug when peron size is changed
    public boolean hidePersonSize() {
        return true;
    }

    public ArrayList<Entrance> getEntrances() {
        return entrances;
    }

    public boolean hideEntrances() {
        return true;
    }

    public double getAreaWidth() {
        return areaWidth;
    }

    public boolean hideAreaWidth() {
        return true;
    }

    public int getWallWidth() {
        return wallWidth;
    }

    public boolean hideWallWidth() {
        return true;
    }

    public ArrayList<Exit> getExits() {
        //Bag exitsCopy = new Bag();
        //exitsCopy.addAll(exits);
        //return exitsCopy;
        return exits;
    }

    public boolean hideExits() {
        return true;
    }

    public int getExitInterval() {
        return exitInterval;
    }

    public void setExitInterval(int exitInterval) {
        this.exitInterval = exitInterval;
    }

    public boolean hideExitInterval() {
        return hideParameters;
    }

    public String desExitInterval() {
        return "How often the exits can remove people (in num of steps)";
    }

    public Object domExitInterval() {
        return new Interval(1, 100);
    }

    public int getEntranceInterval() {
        return entranceInterval;
    }

    public void setEntranceInterval(int entranceInterval) {
        this.entranceInterval = entranceInterval;
    }

    public Object domEntranceInterval() {
        return new Interval(1, 100);
    }

    public boolean hideEntranceInterval() {
        return hideParameters;
    }

    public String desEntranceInterval() {
        return "How often people enter simulation (in steps)";
    }

    public boolean getWriteResults() {
        return writeResults;
    }

    public void setWriteResults(boolean write) {
        writeResults = write;
    }

    /** Run simulation without GUI. Default values are automatically used ##### Fix this ########
     * @param args
     */
    public static void main(String[] args) {
        doLoop(StationSim.Station.class, args);
        System.exit(0);
    }
}

// seed -460236115

// TO DO:
//
// Fix person number bug
// Allow arguments for simulation values to be passed into Station DataAssimKotlin.main()
//
// Start Bayesian methods for hackers