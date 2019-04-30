import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;

import java.util.ArrayList;
import java.util.List;

public class DataGenerator {


    private final static int NUM_ITER = 2000;
    private final static KeanuRandom RAND_GENERATOR = new KeanuRandom();

    private static final double truthThreshold = 0.73;

    private static List<List<DoubleVertex>> fullHistory = new ArrayList<>();

    public static void main (String[] args) {

        System.out.println("Let us begin");

        for (int i = 0; i < 10; i++) {
            getSingleHistory();
        }
    }


    public static void getSingleHistory() {
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

        fullHistory.add(truthData);
    }


    public static void writeToFile() {

    }
}
