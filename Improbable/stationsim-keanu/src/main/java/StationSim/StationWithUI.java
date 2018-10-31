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

import org.jfree.data.xy.XYSeries;
import sim.display.Controller;
import sim.display.Display2D;
import sim.display.GUIState;
import sim.engine.SimState;
import sim.engine.Steppable;
import sim.portrayal.Inspector;
import sim.portrayal.continuous.ContinuousPortrayal2D;
import sim.portrayal.simple.ShapePortrayal2D;
import sim.util.Bag;
import sim.util.media.chart.TimeSeriesChartGenerator;

import javax.swing.*;
import java.awt.*;

/**
 * Visualization class for Station simulation. Various simulation parameters can be set using the GUI.
 *
 */
public class StationWithUI extends GUIState {

    private Display2D display;
    private JFrame displayFrame;

    ContinuousPortrayal2D areaPortrayal = new ContinuousPortrayal2D();
    ContinuousPortrayal2D doorwaysPortrayal = new ContinuousPortrayal2D();
    ContinuousPortrayal2D wallsPortrayal = new ContinuousPortrayal2D();
    TimeSeriesChartGenerator speedChart;
    JFrame speedChartFrame;
    TimeSeriesChartGenerator numPeopleChart;
    JFrame numPeopleChartFrame;

    public StationWithUI() {
        super(new Station(System.currentTimeMillis()));
    }

    public StationWithUI(SimState state) {
        super(state);
    }

    public static String getName() {
        return "Station";
    }

    public static Object getInfo() {
        return "<H2>Station</H2><p>A simple simulation of people coming into and exiting a train station";
    }

    @Override
    public Object getSimulationInspectedObject() {
        return state;
    }

    @Override
    public Inspector getInspector() {
        Inspector inspector = super.getInspector();
        inspector.setVolatile(true);
        return inspector ;
    }

    @Override
    public void start() {
        super.start();
        setupPortrayals();
        XYSeries mean_series, max_series, min_series, numPeopleSeries;

        speedChart.removeAllSeries();
        mean_series = new XYSeries(
                "mean speed",
                false);
        max_series= new XYSeries(
                "max speed",
                false);
        min_series= new XYSeries(
                "min speed",
                false);
        speedChart.addSeries(mean_series, null);
        speedChart.addSeries(max_series, null);
        speedChart.addSeries(min_series, null);
        scheduleRepeatingImmediatelyAfter(new Steppable()
        {
            public void step(SimState state)
            {
                Station station = (Station) state;
                double x = station.schedule.getSteps();
                //double y = station.area.getAllObjects().size();

                Bag people = station.area.getAllObjects();
                double speed;
                double minSpeed = Double.POSITIVE_INFINITY;
                double maxSpeed = 0.0;
                double speedSum = 0.0;
                long numPeople = people.size();
                for (int i = 0; i < numPeople; i++) {
                    speed = ((Person) people.get(i)).getCurrentSpeed();
                    speedSum += speed;
                    if (speed > maxSpeed) {
                        maxSpeed = speed;
                    }
                    if (speed < minSpeed) {
                        minSpeed = speed;
                    }
                }

                // now add the data
                if (x >= state.schedule.EPOCH && x < state.schedule.AFTER_SIMULATION)
                {
                    max_series.add(x, maxSpeed, false);
                    min_series.add(x, minSpeed, false);
                    mean_series.add(x, speedSum / numPeople, false);


                    speedChart.updateChartWithin(state.schedule.getSteps(), 1000);  // update within one second (1000 milliseconds))
                }
            }
        });

        numPeopleChart.removeAllSeries();
        numPeopleSeries = new XYSeries(
                "mean speed",
                false);

        numPeopleChart.addSeries(numPeopleSeries, null);
        scheduleRepeatingImmediatelyAfter(new Steppable()
        {
            public void step(SimState state)
            {
                Station station = (Station) state;
                double x = station.schedule.getSteps();
                double y = station.area.getAllObjects().size();

                // now add the data
                if (x >= state.schedule.EPOCH && x < state.schedule.AFTER_SIMULATION)
                {
                    numPeopleSeries.add(x, y, false);  // don't immediately redraw on adding data
                    numPeopleChart.updateChartWithin(state.schedule.getSteps(), 1000);  // update within one second (1000 milliseconds))
                }
            }
        });
    }

    private void setupPortrayals() {
        Station station = (Station) state;

        doorwaysPortrayal.setField(station.doorways);
        double entranceCorner = station.getEntranceSize() / 2.0;
        double exitCorner = station.getExitSize() / 2.0;

        doorwaysPortrayal.setPortrayalForClass(Entrance.class,
                new ShapePortrayal2D(new double[] {-0.5, -0.5, 0.5, 0.5},
                        new double[] {-entranceCorner, entranceCorner, entranceCorner, -entranceCorner}, Color.green));

        doorwaysPortrayal.setPortrayalForClass(Exit.class,
                new ShapePortrayal2D(new double[] {-station.wallWidth * 2.0, -station.wallWidth * 2.0, 0.5, 0.5},
                        new double[] {-exitCorner, exitCorner, exitCorner, -exitCorner}, Color.red));

        areaPortrayal.setField(station.area);
        areaPortrayal.setPortrayalForClass(Person.class, new ShapePortrayal2D(
                ShapePortrayal2D.X_POINTS_OCTAGON, ShapePortrayal2D.Y_POINTS_OCTAGON, Color.blue));

        wallsPortrayal.setField(station.walls);
        wallsPortrayal.setPortrayalForAll(
                new ShapePortrayal2D(
                        new double[] {-station.wallWidth, -station.wallWidth, station.wallWidth , station.wallWidth},
                new double[] {-station.wallHeight / 2, station.wallHeight / 2, station.wallHeight / 2, -station.wallHeight /2},
                        Color.black));

        display.reset();
        display.setBackdrop(Color.white);
        display.repaint();
    }

    @Override
    public void init(Controller controller) {
        super.init(controller);
        display = new Display2D(1000, 500, this);
        displayFrame = display.createFrame();
        displayFrame.setTitle("Station Display");
        controller.registerFrame(displayFrame);
        displayFrame.setVisible(true);

        // Attach portrayals here
        display.attach(areaPortrayal, "area");
        display.attach(doorwaysPortrayal, "doorways");
        display.attach(wallsPortrayal, "walls");

        //Chart - Speed of people
        speedChart = new TimeSeriesChartGenerator();
        speedChart.setTitle("Speed of agents in simulation");
        speedChart.setXAxisLabel("Step");
        speedChart.setYAxisLabel("Speed (distance per step)");
        speedChartFrame = new JFrame();
        speedChartFrame.setSize(760,760);
        speedChartFrame.add(speedChart);
        // perhaps you might move the chart to where you like.
        speedChartFrame.setVisible(true);
        speedChartFrame.pack();
        controller.registerFrame(speedChartFrame);

        //Chart - number of people
        numPeopleChart = new TimeSeriesChartGenerator();
        numPeopleChart.setTitle("Number of people in Simulation");
        numPeopleChart.setXAxisLabel("Step");
        numPeopleChart.setYAxisLabel("People");
        numPeopleChartFrame = new JFrame();
        numPeopleChartFrame.setSize(760,760);
        numPeopleChartFrame.add(numPeopleChart);
        // perhaps you might move the chart to where you like.
        numPeopleChartFrame.setVisible(true);
        numPeopleChartFrame.pack();
        controller.registerFrame(numPeopleChartFrame);
    }

    @Override
    public void finish()
    {
        super.finish();

        speedChart.update(state.schedule.getSteps(), true);
        speedChart.repaint();
        speedChart.stopMovie();

        numPeopleChart.update(state.schedule.getSteps(), true);
        numPeopleChart.repaint();
        numPeopleChart.stopMovie();
    }


    @Override
    public void quit() {
        super.quit();

        if (displayFrame != null) {
            displayFrame.dispose();
        }
        displayFrame = null;

        speedChart.update(state.schedule.getSteps(), true);
        speedChart.repaint();
        speedChart.stopMovie();
        if (speedChartFrame != null)	speedChartFrame.dispose();
        speedChartFrame = null;

        numPeopleChart.update(state.schedule.getSteps(), true);
        numPeopleChart.repaint();
        numPeopleChart.stopMovie();
        if (numPeopleChartFrame != null)	numPeopleChartFrame.dispose();
        numPeopleChartFrame = null;
    }


    public static void main(String [] args) {
        new StationWithUI().createController();
    }
}
