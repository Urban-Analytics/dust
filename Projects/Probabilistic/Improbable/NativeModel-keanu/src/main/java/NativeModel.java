import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.VertexSamples;
import io.improbable.keanu.algorithms.mcmc.Hamiltonian;
import io.improbable.keanu.algorithms.mcmc.MetropolisHastings;
import io.improbable.keanu.algorithms.mcmc.NUTS;
import io.improbable.keanu.algorithms.variational.optimizer.Optimizer;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertexSamples;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;

import java.io.*;
import java.util.*;
import java.util.stream.Collectors;

public class NativeModel {

    /* Model Parameters */
    private static final UniformVertex threshold = new UniformVertex(-1.0, 1.0);
    private static final int NUM_ITER = 2000; // Total No. of iterations

    /* Hyperparameters */
    private static final int WINDOW_SIZE = 200; // Number of iterations per update window
    private static final int NUM_WINDOWS = NUM_ITER / WINDOW_SIZE; // Number of update windows
    private static final double SIGMA_NOISE = 1.0; // Noise added to observations
    private static final int NUM_SAMPLES = 2000; // Number of samples to MCMC - MH: 100000, HMC: 2000, NUTS: 2000
    private static final int DROP_SAMPLES = 1; // Burn in period
    private static final int DOWN_SAMPLE = 5; // Only keep every x sample

    // Initialise random number generator used throughout
    private static final KeanuRandom RAND_GENERATOR = new KeanuRandom();

    /* Admin parameters */
    private static String dirName = "results/"; // Place to store results

    /* Bookkeeping parameters */
    private static Writer stateWriter; // Writer for the state estimates (i.e. results)


    /**
     *   Run the probabilistic model. This is the main function.
     **/
    public static void main(String[] args) throws Exception {
        // Still need to throw exception?

        System.out.println("Starting.\n" +
                "\tNumber of iterations: " + NUM_ITER +"\n"+
                "\tWindow size: " + WINDOW_SIZE +"\n"+
                "\tNumber of windows: " + NUM_WINDOWS);

        // Initialise threshold for model run
        Double truthThreshold = threshold.sample(KeanuRandom.getDefaultRandom()).getValue(0);

        /*
         ************ CREATE THE TRUTH DATA ************
         */

        System.out.println("Making truth data");



        // Create list to hold truth data for each iteration
        List<DoubleVertex> truthData = new ArrayList<>();

        // Create initial state Vertex as Gaussian distribution with mu=0 and sigma=1
        DoubleVertex truthState = new GaussianVertex(0, 1.0);

        // Generate truth data
        for (int i=0; i < NUM_ITER; i++) {
            truthState =
                    RAND_GENERATOR.nextGaussian() > truthThreshold ? truthState.plus(1) : truthState.minus(1);

            // add state to history
            truthData.add(truthState);
        }

        System.out.println("SimpleModel configured with truth threshold: "+ truthThreshold);
        System.out.println("Truth data length: " + truthData.size());
        System.out.println("Truth data: ");
        for (DoubleVertex aTruthData : truthData) {
            System.out.print(aTruthData.getValue(0) + ",  ");
        }
        System.out.println("\nTruth threshold is: "+ truthThreshold);

        initFiles(truthData); // Open necessary files and write truthThreshold data


        /*
         ************ START THE MAIN LOOP ************
         */
        int iter = 0; // Record the total number of iterations we have been through
        double currentStateEstimate = 0.0; // Save our estimate of the state at the end of the window. Initially 0
        DoubleVertex priorMu = new GaussianVertex(0.0, SIGMA_NOISE);
        List<DoubleVertex> totalStateHistory = new ArrayList<>(); // Store history from each iteration
        List<DoubleTensor> stateSamplesHistory = new ArrayList<>(); // Store state samples


        for (int window=0; window < NUM_WINDOWS; window++) {

            System.out.println(String.format("Entering update window: %s (iterations %s -> %s)", window, iter, iter+WINDOW_SIZE));
            System.out.println(String.format("\tCurrent state (at iter %s) estimate / actual: %s, %s: ",
                    iter,
                    currentStateEstimate,
                    truthData.get(iter).getValue(0)));

            //Increment the counter with how many iterations the model has been run for.
            // In first window increment by -1 otherwise we run off end of truth array on last iteration
            iter += window==0 ? WINDOW_SIZE-1 : WINDOW_SIZE;

            /*
             ************ INITIALISE STATE ************
             */

            // state is probabilistic variable
            DoubleVertex state = new GaussianVertex(priorMu, SIGMA_NOISE);

            List<DoubleVertex> history = new ArrayList<>();

//            // Don't know if this should be here or above STEP?
//            if (window == 0) {
//                state.setAndCascade(priorMu.getValue(0));
//            }

            /*
             ************ STEP ************
             */

            for (int i=0; i < WINDOW_SIZE; i++) {

                // Temporarily imagine that we know the threshold (later try to estimate this)
                state = RAND_GENERATOR.nextGaussian() > truthThreshold ? state.plus(1) : state.minus(1);
                // add state to history
                history.add(state);
            }

            // Collect history from each iteration
            //totalStateHistory.add(history);
            totalStateHistory.addAll(history);

            /*
             ************ OBSERVE TRUTH DATA ************
             */

            System.out.println("State value before obs: " + state.getValue(0));

            // Loop through and apply some observations
            for (int i=0; i < WINDOW_SIZE; i++) {

                // This is model iteration number:
                int t = window==0 ?
                        window * (WINDOW_SIZE - 1) + i : // on first iteration reduce the window size by 1
                        window * (WINDOW_SIZE) + i;

                DoubleVertex currentHist = history.get(i);

                GaussianVertex observedVertex = new GaussianVertex(currentHist, SIGMA_NOISE);

                observedVertex.observe(truthData.get(t).getValue());

                state.setValue(observedVertex.getValue());
            }

            System.out.println("State value after obs: " + state.getValue(0));

            /*

            // This is model iteration number:
            int t = window==0 ?
                    window * (WINDOW_SIZE - 1) : // on first iteration reduce the window size by 1
                    window * (WINDOW_SIZE);

            GaussianVertex noisyObservation = new GaussianVertex(truthData.get(t), SIGMA_NOISE);

            state.observe(noisyObservation.getValue());
            */

            /*
             ************ CREATE THE BAYES NET ************
             */

            // Create the BayesNet
            System.out.println("\tCreating BayesNet");
            BayesianNetwork net = new BayesianNetwork(state.getConnectedGraph());

            // write Bayesnet
            //writeBayesNetToFile(net);

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
            parameters.add(state);

            ///*
            NetworkSamples sampler = MetropolisHastings.withDefaultConfig().getPosteriorSamples(
                    net,
                    parameters,
                    NUM_SAMPLES);
            //*/

            /*
            NetworkSamples sampler = NUTS.withDefaultConfig().getPosteriorSamples(
                    net,
                    parameters,
                    NUM_SAMPLES);
            */

            /*
            NetworkSamples sampler = Hamiltonian.withDefaultConfig().getPosteriorSamples(
                    net,
                    net.getLatentVertices(),
                    NUM_SAMPLES);
            */

            // Downsample etc
            sampler = sampler.drop(DROP_SAMPLES).downSample(DOWN_SAMPLE);

            /*
             ************ GET THE INFORMATION OUT OF THE SAMPLES ************
             */

            // Getting stateSamples
            List<DoubleTensor> stateSamples = sampler.get(state).asList();
            stateSamplesHistory.addAll(stateSamples);

            // Convert List<DoubleTensor> into DoubleVertexSamples
            // Keep for later when devs have produced DoubleVertexSamples.asTensor()
            // Should allow for vertex of shape (a x b) with n samples to become tensor of shape (a x b x n)
            DoubleVertexSamples stateVertexSamples = new DoubleVertexSamples(stateSamples);

            List<Double> stateSamplesDouble = stateSamples.stream()
                    .map( (d) -> d.getValue(0)).collect(Collectors.toList());

            // Get mean of stateSamples to compare to truthValue and posterior
            double stateSum = stateSamplesDouble.stream().mapToDouble(Double::doubleValue).sum();
            double stateSamplesMean = stateSum / stateSamplesDouble.size();

            /*
             ************ ASSIGN VALUES FOR NEXT PRIOR AND PRINT RESULTS ************
             */

            // Get posterior distribution, extract and assign Mu
            //DoubleTensor posterior = state.getValue();
            //priorMu = posterior.scalar();

            // Assign state posterior distribution to priorMu
            priorMu = state;

            // Find current state estimate
            currentStateEstimate = state.getValue(0);


            // Print out interesting values
            System.out.println("\tstateSamplesMean: " + stateSamplesMean);
            System.out.println("\tcurrentStateEstimate == " + currentStateEstimate);
            System.out.println("\tCurrent truth state: " + truthData.get(iter).getValue(0));
            System.out.println("\tTruth threshold is: "+ truthThreshold);

        } // Update window

        /*
        System.out.println("totalStateHistory information:");
        System.out.println("totalStateHistory.size() == " + totalStateHistory.size());

        System.out.println("truthData information:");
        System.out.println("truthData.size() == " + truthData.size());

        System.out.println("stateSamplesHistory information:");
        System.out.println("stateSamplesHistory.size() == " + stateSamplesHistory.size());

        Set<DoubleTensor> stateSamplesHistorySet = new HashSet<>(stateSamplesHistory);
        System.out.println("stateSamplesHistorySet information:");
        System.out.println("stateSamplesHistorySet.size() == " + stateSamplesHistorySet.size());
        */

        NativeModel.writeResults(totalStateHistory, truthData);

    } // main()

    /*
     **************** ADMIN STUFF ****************
     */

    private static void initFiles(List<DoubleVertex> truthData) throws IOException {
        // So files have unique names
        String theTime = String.valueOf(System.currentTimeMillis());

        // Open Results file to hold state data
        stateWriter = new BufferedWriter(new OutputStreamWriter(
                new FileOutputStream(dirName + "Results_" + theTime + ".csv"), "utf-8"));

        // Write truthData first
        for (DoubleVertex truth : truthData) {
            stateWriter.write(truth.getValue(0) + ",");
        }
        stateWriter.write("\n");
    }


    private static void writeResults(List<DoubleVertex> totalHistory, List<DoubleVertex> truthData) {

        assert totalHistory.size() == truthData.size();

        try {
            // Write information from history
            for (DoubleVertex element : totalHistory) {
                stateWriter.write(element.getValue(0) + ",");
            }
            stateWriter.write("\n");

            stateWriter.close();

        } catch (IOException ex) {
            System.err.println("Error writing states to file: " + ex.getMessage());
            ex.printStackTrace();
        }
    }
}
