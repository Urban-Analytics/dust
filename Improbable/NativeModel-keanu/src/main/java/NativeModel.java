import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.mcmc.MetropolisHastings;
import io.improbable.keanu.algorithms.variational.optimizer.Optimizer;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.stream.Collectors;

public class NativeModel {

    /* Model Parameters */
    private static final UniformVertex threshold = new UniformVertex(-1.0, 1.0);
    private static final int NUM_ITER = 2000; // Total No. of iterations

    /* Hyperparameters */
    private static final int WINDOW_SIZE = 200; // Number of iterations per update window
    private static final int NUM_WINDOWS = NUM_ITER / WINDOW_SIZE; // Number of update windows
    private static final double SIGMA_NOISE = 1.0; // Noise added to observations
    private static final int NUM_SAMPLES = 2000; // Number of samples to MCMC
    private static final int DROP_SAMPLES = 1; // Burn in period
    private static final int DOWN_SAMPLE = 5; // Only keep every x sample

    // Initialise random number generator used throughout
    //public static final int NUM_RANDOM_DOUBLES = 10000;
    //private static final VertexBackedRandomGenerator RAND_GENERATOR =
    //        new VertexBackedRandomGenerator(NUM_RANDOM_DOUBLES, 0, 0);
    private static final KeanuRandom RAND_GENERATOR = new KeanuRandom();

    /* Admin parameters */
    private static String dirName = "results/"; // Place to store results

    /* Bookkeeping parameters */
    private static boolean firstRun = true; // Horrible hack to do with writing files. Set to false after first window
    private static Writer thresholdWriter; // Writer for the threshold estimates
    private static Writer stateWriter; // Writer for the state estimates (i.e. results)


    /**
     *   Run the probabilistic model. This is the main function.
     **/
    public static void main(String[] args) throws IOException {
        // Still need to throw exception?

        System.out.println("Starting.\n" +
                "\tNumber of iterations: " + NUM_ITER +"\n"+
                "\tWindow size: " + WINDOW_SIZE +"\n"+
                "\tNumber of windows: " + NUM_WINDOWS);

        // Initialise stuff
        Double truthThreshold = threshold.sample(KeanuRandom.getDefaultRandom()).getValue(0);

        /*
         ************ CREATE THE TRUTH DATA ************
         */

        System.out.println("Making truth data");

        // Generate truth data

        List<DoubleVertex> truthData = new ArrayList<>();
        // TODO: Replace this sigma with SIGMA_NOISE?
        DoubleVertex currentTruthState = new GaussianVertex(0, 1.0);
        for (int i=0; i < NUM_ITER; i++) {
            currentTruthState =
                    RAND_GENERATOR.nextGaussian() > truthThreshold ? currentTruthState.plus(1) : currentTruthState.minus(1);
            // add state to history
            truthData.add(currentTruthState);
        }

        System.out.println("SimpleModel configured with truth threshold: "+ truthThreshold);
        System.out.println("Truth data length: " + truthData.size());
        System.out.println("Truth data: ");
        for (DoubleVertex aTruthData : truthData) {
            System.out.print(aTruthData.getValue(0) + ",  ");
        }
        System.out.println("\nTruth threshold is: "+ truthThreshold);

        initFiles(); // Get files ready to start writing results

        /*
         ************ START THE MAIN LOOP ************
         */
        int iter = 0; // Record the total number of iterations we have been through
        double currentStateEstimate = 0.0; // Save our estimate of the state at the end of the window. Initially 0
        double currentThresholdEstimate = -1.0; //  Interesting to see what the threshold estimate is (not used in assimilation)
        double priorMu = 0.0;


        // TODO: Replace this loop with a while loop (Need to calculate error w/ each iteration and stop loop when error is low enough)
        for (int window=0; window < NUM_WINDOWS; window++) {

            System.out.println(String.format("Entering update window: %s (iterations %s -> %s)", window, iter, iter+WINDOW_SIZE));
            System.out.println(String.format("\tCurrent state (at iter %s) estimate / actual: %s, %s: ",
                    iter,
                    currentStateEstimate,
                    truthData.get(iter).getValue(0)));
            System.out.println(String.format("\tCurrent threshold estimate (for info): %.2f", currentThresholdEstimate));

            //Increment the counter with how many iterations the model has been run for.
            // In first window increment by -1 otherwise we run off end of truth array on last iteration
            iter += window==0 ? WINDOW_SIZE-1 : WINDOW_SIZE;

            /*
             ************ INITIALISE STATE ************
             */

            DoubleVertex state = new GaussianVertex(priorMu, 1.0);

            List<DoubleVertex> history = new ArrayList<>();

            /*
             ************ STEP ************
             */

            for (int i=0; i < WINDOW_SIZE; i++) {

                // Temporarily imagine that we know the threshold (later try to estimate this)
                // TODO: Try to estimate threshold here and not pass explicitly
                state = RAND_GENERATOR.nextGaussian() > truthThreshold ? state.plus(1) : state.minus(1);
                // add state to history
                history.add(state);
            }

            state.setAndCascade(0); // Sets state value to 0 (known at start of model)

            /*
             ************ OBSERVE SOME TRUTH DATA ************
             */

            // Loop through and apply some observations
            for (int i=0; i < WINDOW_SIZE; i++) {

                // This is model iteration number:
                int t = window==0 ?
                        window * (WINDOW_SIZE - 1) + i : // on first iteration reduce the window size by 1
                        window * (WINDOW_SIZE ) + i;

                DoubleVertex currentHist = history.get(i);

                GaussianVertex observedVertex = new GaussianVertex(currentHist, SIGMA_NOISE);

                observedVertex.observe(truthData.get(t).getValue(0));
            }

            /*
             ************ CREATE THE BAYES NET ************
             */

            // Create the BayesNet
            System.out.println("\tCreating BayesNet");
            BayesianNetwork net = new BayesianNetwork(state.getConnectedGraph());

            // write Bayes net
            writeBayesNetToFile(net);

            /*
             ************ OPTIMISE ************
             */

            System.out.println("\t\tOptimising with Max A Posteriori");
            System.out.println("\t\tPrevious state value: " + state.getValue(0));

            System.out.println("\t\tRunning Max A Posteriori...");

            Optimizer netOptimiser = Optimizer.of(net);
            netOptimiser.maxAPosteriori();

            System.out.println("\t\tNew state value: " + state.getValue(0));

            /*
             ************ SAMPLE FROM THE POSTERIOR ************
             */

            // Sample from the posterior
            System.out.println("\tSampling");

            // Collect all the parameters that we want to sample
            List<Vertex> parameters = new ArrayList<>();
            parameters.add(threshold);
            parameters.add(state);

            NetworkSamples sampler = MetropolisHastings.withDefaultConfig().getPosteriorSamples(
                    net,
                    parameters,
                    NUM_SAMPLES);

            /*
            NetworkSamples sampler = MetropolisHastings.withDefaultConfig().getPosteriorSamples(
                    net,
                    net.getLatentVertices(),
                    NUM_SAMPLES);
            */

            // Downsample etc
            sampler = sampler.drop(DROP_SAMPLES).downSample(DOWN_SAMPLE);

            /*
             ************ GET THE INFORMATION OUT OF THE SAMPLES ************
             */

            // TODO: PROBLEM AREA. Possibly convert all List<DoubleTensor> to List<Double>?

            // ****** The threshold ******
            // Get the threshold estimates as a list (Convert from Tensors to Doubles)
            List<Double> thresholdSamples = sampler.get(threshold).asList().
                    stream().map( (d) -> d.getValue(0)).collect(Collectors.toList());

            // Find the mean threshold estimate (for info, not used)
            currentThresholdEstimate = thresholdSamples.stream().reduce(0d, (a,b) -> a+b) / thresholdSamples.size();

            // Write the threshold distribution
            //NativeModel.writeThresholds(thresholdSamples, truthThreshold, window, iter);

            // TODO: Fix this so that the contents of the List and DoubleTensors can be accessed easily later on
            // ****** The model states (out of the box) ******
            List<DoubleTensor> stateSamples = sampler.get(state).asList();
            //System.out.println(stateSamples);


            // Previous attempts:
            //List<DoubleVertex> stateSamples = sampler.get(history).asList();
            //List<DoubleVertex> stateSamples = sampler.get(history)
            //System.out.println(stateSamplesDouble.toString());
            //List<DoubleTensor> stateSamples = sampler.getDoubleTensorSamples(state).asList();

            List<Double> stateSamplesDouble = sampler.get(state).asList().
                    stream().map( (d) -> d.getValue(0)).collect(Collectors.toList());

            // Get mean of stateSamples to compare to truthValue and posterior
            double stateSum = stateSamplesDouble.stream().mapToDouble(Double::doubleValue).sum();
            double stateSamplesMean = stateSum / stateSamplesDouble.size();

            assert stateSamples.size() == thresholdSamples.size();

            //NativeModel.writeResults(stateSamples, truthData);

            /*
             ************ ASSIGN VALUES FOR NEXT PRIOR AND PRINT RESULTS ************
             */

            // Get posterior distribution, extract and assign Mu
            DoubleTensor posterior = state.getValue();
            priorMu = posterior.scalar();

            currentStateEstimate = state.getValue(0);


            // Print out interesting values
            System.out.println("\tstateSamplesMean: " + stateSamplesMean);
            //System.out.println("\tPosterior: " + posterior.getValue(0));
            System.out.println("\tposterior == priorMu == currentStateEstimate == " + priorMu);
            System.out.println("\tCurrent truth state: " + truthData.get(iter).getValue(0));
            //System.out.println("\tcurrentStateEstimate: " + currentStateEstimate);


            firstRun = false; // To say the first window has finished (horrible hack to do with writing files)

        } // Update window

        //NativeModel.closeResultsFiles();

    } // main()

    /*
     **************** ADMIN STUFF ****************
     */


    private static void writeThresholds(List<Double> randomNumberSamples, Double truthThreshold, int window, int iteration) {

        // Write out random numbers used  and the actual results
        try {
            if (firstRun) { // if true then the files have just been initialised. Write the truth data to
                thresholdWriter.write("-1,-1 ," + truthThreshold + ",\n"); // Truth value (one off)
            }

            // Now the samples
            for (double sample : randomNumberSamples) {
                thresholdWriter.write(String.format("%s, %s, %s, ", window, iteration, sample));
                thresholdWriter.write("\n");
            }

        } catch (IOException ex) {
            System.out.println("Error writing thresholds to file: " + ex.getMessage());
        }
    }


    private static void writeBayesNetToFile(BayesianNetwork net) {
        try {
            Writer graphWriter = new BufferedWriter(new OutputStreamWriter(
                    new FileOutputStream(dirName + "Graph_" + System.currentTimeMillis() + ".dot"), "utf-8"));
            //graphWriter.write(GraphvizKt.toGraphvizString(net, new HashMap<>()));
            graphWriter.close();
        } catch (IOException ex) {
            System.err.println("Error writing graph to file: " + ex.getMessage());
            ex.printStackTrace();
        }
    }


    private static void writeResults() {

    }


    private static void initFiles() throws IOException {
        String theTime = String.valueOf(System.currentTimeMillis()); // So files have unique names

        thresholdWriter = new BufferedWriter(new OutputStreamWriter(
                new FileOutputStream(dirName + "Params_" + theTime + ".csv"), "utf-8"));

        thresholdWriter.write("Window, Iteration, Threshold,\n");

        thresholdWriter = new BufferedWriter(new OutputStreamWriter(
                new FileOutputStream(dirName + "Results_" + theTime + ".csv"), "utf-8"));
    }


    private static void closeResultsFiles() throws Exception {
        System.out.println("Closing output files.");

        /*
        // Write the cached samples
        for (Double sample : samplesHistory) {
            for (int i=0; i < sample.size(); i++) {
                stateWriter.write(sample.getValue(i) +  ",");
            }
            stateWriter.write("\n");
        }
        */

        thresholdWriter.close();
        stateWriter.close();
    }
}
