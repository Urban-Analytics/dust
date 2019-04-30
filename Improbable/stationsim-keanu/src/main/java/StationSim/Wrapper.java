package StationSim;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

/**
 * Created by nick on 22/06/2018.
 */
public class Wrapper{

    private static int numTimeSteps = 1000;
    private static int numSamples = 500;
    private static int dropSamples = 200;
    private static int downSample = 3;
    private static double sigmaNoise = 0.1; // The amount of noise to be added to the truth

    // Options are used to set what is *observed* and what is *output*. See createOptions() for details.
    private static HashMap<Integer, Integer> optionsMap = createOptions();
    private static int option = 1;
    private static int numOutputs = optionsMap.get(option);

    private static boolean justCreateGraphs = false; // Create graphs and then exit, no sampling

    private static String dirName = "results/plot/three/"; // Place to store results

    private static HashMap<Integer, Integer> createOptions() {
        // Map options to the number of outputs given from model - key = the option number to use : value = length of array given as output per step
        HashMap optionsMap = new HashMap<Integer, Integer>();
        optionsMap.put(1, 2); //Observe total num people in simulation : Output same
        optionsMap.put(2, 6); // Observe total people that have passed through each entrance/exit : Output cumulative number of people in a given grid square
        optionsMap.put(3, 6); // Observe total people that have passed through each entrance/exit : Output number of people in a given grid square per step
        return optionsMap;
    }



    public static Integer[] run(KeanuRandom rand) {
        Station stationSim = new Station(System.currentTimeMillis());
        System.out.println("Model "+ Station.modelCount++ +" starting");
        stationSim.start(rand);
        System.out.println("Num of outputs: " + numOutputs);

        Integer[] stepOutput;
        Integer[] results = new Integer[numTimeSteps * numOutputs]; //why was a "+ 1" in here?
        for (int i = 0; i < results.length; i++) {
            results[i] = 0;
        }
        int i = 0;
        do {
            // Run a step of each simulation
            if (!stationSim.schedule.step(stationSim)) {
                break;
            }
            switch(option) {
                case 1:
                    stepOutput = new Integer[] {stationSim.area.getAllObjects().size(), stationSim.area.getAllObjects().size()}; // added twice as fix - later on the first is observed and the second used as an output - not doing this leads to no observation
                    break;
                case 2:
                    stepOutput = stationSim.analysis.getOption2();
                    break;
                case 3:
                    stepOutput = stationSim.analysis.getOption3();
                    break;
                default:
                    stepOutput = new Integer[] {0};
                    System.out.println("\nIncorrect option number");
                    System.exit(-1);
            }

            //stepOutput = new Integer[] {stationSim.area.getAllObjects().size()};
            for (int j = 0; j < numOutputs; j++) {
                results[i * numOutputs + j] = stepOutput[j];
            }
            i++;
        } while (stationSim.area.getAllObjects().size() > 0 && i < numTimeSteps);
        stationSim.finish();

        return results;
    }

    //public static List<Integer[]> keanu(Integer[] truth, int obInterval, long timestamp, boolean createGraph) {
    public static void keanu(Integer[] truth, int obInterval, long timestamp, boolean createGraph) {

        // (Useful string for writing results)
        int totalNumPeople = new Station(System.currentTimeMillis()).getNumPeople();
        //String params = "obInterval" + obInterval + "_numSamples" + numSamples + "_numTimeSteps" + numTimeSteps + "_numRandomDoubles" + numRandomDoubles + "_totalNumPeople" + totalNumPeople + "_dropSamples" + dropSamples + "_downSample" + "_sigmaNoise" + sigmaNoise + "_downsample" + downSample + "_timeStamp" + timestamp;
        String params = "obInterval" + obInterval + "_numSamples" + numSamples + "_numTimeSteps" + numTimeSteps + "_totalNumPeople" + totalNumPeople + "_dropSamples" + dropSamples + "_downSample" + "_sigmaNoise" + sigmaNoise + "_downsample" + downSample + "_timeStamp" + timestamp;

        System.out.println("Initialising random number stream");
        //VertexBackedRandomFactory random = new VertexBackedRandomFactory(numInputs,, 0, 0);
        //RandomFactoryVertex random = new RandomFactoryVertex (numRandomDoubles, 0, 0);
        KeanuRandom random = new KeanuRandom();


        System.out.println("Initialising state and state history");
        // Initialise state as probabilistic variable
        DoubleVertex state = new GaussianVertex(0.0, 1.0);
        // Init state history
        List<DoubleVertex> stateHistory = new ArrayList<>();

        // Run model and collect state information
        Integer[] results = run(random);

        // Loop through results and assign value of each to state
        // Then collect state in stateHistory
        for (int i=0; i<results.length; i++) {
            //DoubleVertex state = new GaussianVertex(results[i], sigmaNoise);
            state.setValue(results[i]);
            stateHistory.add(state);
        }

        if (obInterval > 0) {
            System.out.println("Observing truth data. Adding noise with standard dev: " + sigmaNoise);
            for (int i=0; i<numTimeSteps; i++) {
                if ((i+1) % numOutputs != 0) {

                    DoubleVertex currentHist = stateHistory.get(i);

                    GaussianVertex observedVertex = new GaussianVertex(currentHist, sigmaNoise);

                    observedVertex.observe(truth[i]);
                }
            }
        } else {
            System.out.println("Not observing truth data");
        }

        /*
        // This is the 'black box' vertex that runs the model. It's input is the random numbers and
        // output is a list of Integer(tensor)s (the number of agents in the model at each iteration).
        System.out.println("Initialising black box model");
        //UnaryOpLambda<VertexBackedRandomGenerator, Integer[]> box = new UnaryOpLambda<>( random, Wrapper::run);
        UnaryOpLambda<KeanuRandom, Integer[]> box = new UnaryOpLambda<>(random, Wrapper::run);


        // Observe the truth data plus some noise?
        if (obInterval > 0) {
            System.out.println("Observing truth data. Adding noise with standard dev: " + sigmaNoise);
            for (Integer i = 0; i < numTimeSteps; i++) {
                //if((i % obInterval == 0) {
                if((i + 1) % numOutputs != 0) {
                    // output is the ith element of the model output (from box)
                    IntegerArrayIndexingVertex output = new IntegerArrayIndexingVertex(box, i);
                    // output with a bit of noise. Lower sigma makes it more constrained.
                    GaussianVertex noisyOutput = new GaussianVertex(new CastDoubleVertex(output), sigmaNoise);
                    // Observe the output
                    noisyOutput.observe(truth[i].doubleValue()); //.toDouble().scalar());
                }
            }
        }
        else {
            System.out.println("Not observing truth data");
        }
        */

        System.out.println("Creating BayesNet");
        //BayesianNetwork testNet = new BayesianNetwork(box.getConnectedGraph());
        //BayesianNetwork testNet = new BayesianNetwork(box.getConnectedGraph());

        // Create a graph and write it out
        // Write out samples
        /*
        try {
            System.out.println("Writing out graph");
            Writer graphWriter = new BufferedWriter(new OutputStreamWriter(
                new FileOutputStream(dirName + "Graph_" + params + ".dot"),
                "utf-8"));
            graphWriter.write(GraphvizKt.toGraphvizString(testNet, new HashMap<>()) );
            graphWriter.close();
        } catch (IOException ex) {
            System.out.println("Error writing graph to file");
        }
        */

        /*
        // If just creating a graph then don't do anything further
        if (createGraph) {
            System.out.println("\n\n\n" + GraphvizKt.toGraphvizString(testNet, new HashMap<>()) + "\n\n\n");
            System.out.println("Have created graph. Not sampling");
            return new ArrayList<Integer[]>();
        }
        */

        // Workaround for too many evaluations during sample startup
        //random.setAndCascade(random.getValue());

        // Sample: feed each randomNumber in and runModel the model
        System.out.println("Sampling");
        //NetworkSamples sampler = MetropolisHastings.getPosteriorSamples(testNet, Arrays.asList(box), numSamples);

        // Interrogate the samples

        // Get the number of people per iteration (an array of IntegerTensors) for each sample
        //List<Integer[]> samples = sampler.drop(dropSamples).downSample(downSample).get(box).asList();

        //writeResults(samples, truth, params);

        //return samples;
    }

    public static void writeResults(List<Integer[]> samples, Integer[] truth, String params) {
        Writer writer = null;

        // Write out samples
        try {
            writer = new BufferedWriter(new OutputStreamWriter(
                new FileOutputStream(dirName + "Samples_" + params + ".csv"),
                "utf-8"));
            System.out.println("writing: " + dirName + "Samples_" + params + ".csv");
            for (int i = 0; i < samples.size(); i++) {
                Integer[] peoplePerIter = samples.get(i);
                for (int j = 0; j <  peoplePerIter.length ; j++) {
                    if ((j + 1) % numOutputs == 0) {
                        writer.write(peoplePerIter[j] + "");
                        if (j < peoplePerIter.length - numOutputs) {
                            writer.write(",");
                        }
                    }
                }
                writer.write(System.lineSeparator());
            }
        } catch (IOException ex) {
            System.out.println("Error writing to file");
        } finally {
            try {
                writer.close();
            } catch (Exception ex) {
                System.out.println("Error closing file");
            }
        }

        // Write out Truth
        try {
            writer = new BufferedWriter(new OutputStreamWriter(
                new FileOutputStream(dirName + "Truth_" + params + ".csv"),
                "utf-8"));
            for (int i = 0; i < truth.length ; i++) {
                if ((i + 1) % numOutputs == 0) {
                    writer.write(truth[i] + "");
                    if (i < truth.length - numOutputs) {
                        writer.write(",");
                    }
                }
            }
            writer.write(System.lineSeparator());

        } catch (IOException ex) {
            System.out.println("Error writing to file");
        } finally {
            try {
                writer.close();
            } catch (Exception ex) {
                System.out.println("Error closing file");
            }
        }
    }

    public static void main(String[] args) {

        long timestamp = System.currentTimeMillis();

        System.out.println("Starting. Number of iterations: " + numTimeSteps);

        // Make truth data
        System.out.println("Making truth data");
        //VertexBackedRandomGenerator truthRandom = new VertexBackedRandomGenerator(numRandomDoubles, 0, 0);
        KeanuRandom truthRandom = new KeanuRandom();
        Integer[] truth = Wrapper.run(truthRandom);
        System.out.println("Truth data length: " + truth.length);

        //Run keanu
        ArrayList<Integer> obIntervals = new ArrayList<>(Arrays.asList(0,1));
        obIntervals.parallelStream().forEach(i -> keanu(truth, i, timestamp, justCreateGraphs));

    }
}
