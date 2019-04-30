package SimpleModel;


import org.apache.commons.math3.random.RandomGenerator;

import java.util.ArrayList;

public class SimpleWrapper {

    /* Model parameters */
    private static final double threshold = 0.0;
    public static final int NUM_RAND_DOUBLES = 50;
    private static final int NUM_ITER = 10000;

    /* Hyperparameters */
    private static final double SIGMA_NOISE = 0.1;
    private static final int NUM_SAMPLES = 10000;
    private static final int DROP_SAMPLES = 1;
    //private static final int DROP_SAMPLES = NUM_SAMPLES/4;
    private static final int DOWN_SAMPLE = 5;

    private static final int numObservations = 5; // Number of points to observe (temporary - will be replaced with proper tests)

    private static ArrayList<SimpleModel> models = new ArrayList<>(); // Keep all the models for analysis later

    //private static final long SEED = 1l;
    //private static final RandomGenerator random = new MersenneTwister(SEED);



    /* Admin parameters */
    private static String dirName = "results/simple/"; // Place to store results


    /** Run the SimpleModel and return the count at each iteration **/

    public static Integer[] runModel(RandomGenerator rand) {
        SimpleModel s = new SimpleModel(SimpleWrapper.threshold, rand);

        for (int i=0; i<NUM_ITER; i++) {
            s.step();
        }
        SimpleWrapper.models.add(s);
        return s.getHistory();
    }


    /**
     * Run the probabilistic model
     **/
    public static void run() {
/*        System.out.println("Starting. Number of iterations: " + NUM_ITER);

        /*
                ************ CREATE THE TRUTH DATA ************
         */
/*
        System.out.println("Making truth data");
        System.out.println("Initialising random number stream for truth data");
        //VertexBackedRandomGenerator truthRandom = new VertexBackedRandomGenerator(NUM_RAND_DOUBLES, 0, 0);

        // Store the random numbers used (for comparison later)
        List<Double> truthRandomNumbers = new ArrayList<>(NUM_RAND_DOUBLES);
        //for (int i=0; i<NUM_RAND_DOUBLES; i++) truthRandomNumbers.add(truthRandom.nextGaussian());

        // Run the model
        //Integer[] truth = SimpleWrapper.runModel(truthRandom);

        System.out.println("Truth data length: " + truth.length);
        System.out.println("Truth data: "+Arrays.asList(truth).toString() + "\n\n");
        System.out.println("Truth random numbers:");
        System.out.println(truthRandomNumbers.toString());
        (new ArrayList<String>(NUM_RAND_DOUBLES)).forEach(i -> System.out.print(truthRandom.nextGaussian()+", "));
        System.out.println();



        /*
         ************ INITIALISE THE BLACK BOX MODEL ************
         */
/*
        System.out.println("Initialising new random number stream");
        RandomFactoryVertex random = new RandomFactoryVertex(NUM_RAND_DOUBLES, 0, 0);

        // This is the 'black box' vertex that runs the model. It's input is the random numbers and
        // output is a list of Integer(tensor)s (the number of agents in the model at each iteration).
        System.out.println("Initialising black box model");
        UnaryOpLambda<VertexBackedRandomGenerator, Integer[]> box =
            new UnaryOpLambda<>( random, SimpleWrapper::runModel);



        /*
         ************ OBSERVE SOME TRUTH DATA ************
         */

/*
        // Observe the truth data plus some noise?
        System.out.println("Observing truth data. Adding noise with standard dev: " + SIGMA_NOISE);
        System.out.print("Observing at iterations: ");
        for (Integer i = 0; i < NUM_ITER; i+=NUM_ITER/numObservations) {
            System.out.print(i+",");
            // output is the ith element of the model output (from box)
            IntegerArrayIndexingVertex output = new IntegerArrayIndexingVertex(box, i);
            // output with a bit of noise. Lower sigma makes it more constrained.
            GaussianVertex noisyOutput = new GaussianVertex(new CastDoubleVertex(output), SIGMA_NOISE);
            // Observe the output
            noisyOutput.observe(truth[i].doubleValue()); //.toDouble().scalar());
        }
        System.out.println();


        /*
         ************ CREATE THE BAYES NET ************
         */
/*
        // Create the BayesNet
        System.out.println("Creating BayesNet");
        BayesianNetwork net = new BayesianNetwork(box.getConnectedGraph());
        SimpleWrapper.writeBaysNetToFile(net);

        // Workaround for too many evaluations during sample startup
        //random.setAndCascade(random.getValue());


        /*
         ************ SAMPLE FROM THE POSTERIOR************
         */
/*
        // Sample from the posterior
        System.out.println("Sampling");
/*
        // These objectcs represent the random numbers used in the probabilistic model
        List<GaussianVertex> randNumbers = new ArrayList(random.getValue().randDoubleSource);

        // Collect all the parameters that we want to sample (the random numbers and the box model)
        List<Vertex> parameters = new ArrayList<>(NUM_RAND_DOUBLES+1); // Big enough to hold the random numbers and the box
        parameters.add(box);
        parameters.addAll(randNumbers);

        // Sample from the box and the random numbers
        NetworkSamples sampler = MetropolisHastings.getPosteriorSamples(
            net,                // The bayes net with latent variables (the random numbers?)
            parameters,         // The vertices to include in the returned samples
            NUM_SAMPLES);       // The number of samples

        // Sample using a stream.
        /*
        NetworkSamples sampler = MetropolisHastings.generatePosteriorSamples(
            net,                // The bayes net with latent variables (the random numbers?)
            parameters          // The vertices to include in the returned samples
        )
            .dropCount(DROP_SAMPLES)
            .downSampleInterval(DOWN_SAMPLE)
            .stream()
            .limit(NUM_SAMPLES)
            .map(networkState -> {
                for()
                    networkState.get(x)
            })
            .average().getAsDouble();
            */
/*
        System.out.println("Finished running MCMC.");
/*
        // Downsample etc
        sampler = sampler.drop(DROP_SAMPLES).downSample(DOWN_SAMPLE);


        /*
         ************ GET THE INFORMATION OUT OF THE SAMPLES ************
         */
/*
        // Get the random numbers. A 2D list. First dimension holds the random number, second dimensions holds its samples.
        List<List<Double>> randomNumberSamples = new ArrayList<List<Double>>(NUM_SAMPLES);
        // Add each random number parameter to the list
        for (int i=0; i<NUM_RAND_DOUBLES; i++) {
            List<DoubleTensor> randSamples = sampler.get(randNumbers.get(i)).asList();
            // Convert from Tensors to Doubles
            List<Double> randSamplesDouble =
                randSamples.stream().map( (d) -> d.getValue(0)).collect(Collectors.toList());
            randomNumberSamples.add(randSamplesDouble);
        }

        String theTime = String.valueOf(System.currentTimeMillis()); // So files have unique names
        SimpleWrapper.writeRandomNumbers(randomNumberSamples, truthRandomNumbers, theTime);

        // Get the number of people per iteration for each sample
        List<Integer[]> peopleSamples = sampler.get(box).asList();
        System.out.println("Have saved " + peopleSamples.size() + " samples and ran " + models.size() + " models");
        SimpleWrapper.writeResults(peopleSamples , truth, theTime);

    }





    /*
    **************** ADMIN STUFF ****************
     */
/*
    private static void writeRandomNumbers(List<List<Double>> randomNumberSamples, List<Double> truthRandomNumbers, String name) {

        // Write out random numbers used and the actual results
        Writer w1;
        try {
            w1 = new BufferedWriter(new OutputStreamWriter(
                new FileOutputStream(dirName + "Rands_" + name + ".csv"), "utf-8"));

            // Do the truth data first
            for (double val : truthRandomNumbers) w1.write(val + ",");
            w1.write("\n");

            // Now the samples.
            for (int sample = 0; sample<randomNumberSamples.get(0).size(); sample++) {
                for (int d = 0; d<NUM_RAND_DOUBLES; d++) {
                    w1.write(randomNumberSamples.get(d).get(sample)+", ");
                }
                w1.write("\n");
            }
            w1.close();
        } catch (IOException ex) {
            System.out.println("Error writing to file");
        }
    }


    private static void writeBaysNetToFile(BayesianNetwork net) {
        try {
            System.out.println("Writing out graph");
            Writer graphWriter = new BufferedWriter(new OutputStreamWriter(
                new FileOutputStream(dirName + "Graph_" + System.currentTimeMillis() + ".dot"),
                "utf-8"));
            graphWriter.write(GraphvizKt.toGraphvizString(net, new HashMap<>()) );
            graphWriter.close();
        } catch (IOException ex) {
            System.out.println("Error writing graph to file");
        }
    }

/*
    private static void writeResults(List<Integer[]> samples, Integer[] truth, String name) {

        // Write out the model results (people per iteration)
        Writer w1;
        try {
            w1 = new BufferedWriter(new OutputStreamWriter(
                new FileOutputStream(dirName + "Results_" + name + ".csv"), "utf-8"));

            // Do the truth data first
            for (int val : truth) {
                w1.write(val + ","); // Values
            }
            w1.write("\n");

            // Now the samples.
            for (Integer[] sample : samples) {
                for (int val : sample) {
                    w1.write(val + ",");
                }
                w1.write("\n");
            }
            w1.close();
        } catch (IOException ex) {
            System.out.println("Error writing to file");
        }
    }

    public static void main (String[] args) {

        SimpleWrapper.run();
/*  */  }
}