package StationSim;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;

import java.util.ArrayList;
import java.util.List;

public class CombineDoublesTest {

    public static List<DoubleVertex> testBuild() {

        List<DoubleVertex> newList = new ArrayList<>();

        for (int i=0; i<100; i++) {
            DoubleVertex banana = new GaussianVertex(i, 1.0);
            newList.add(banana);
        }
        return newList;
    }

    public static void main(String[] args) {

        List<DoubleVertex> testList = testBuild();

        CombineDoubles combined = new CombineDoubles(testList);

        for (DoubleTensor comb : combined.calculate()) {
            System.out.println(comb.getValue(0));
        }
    }
}
