package StationSim;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.generic.nonprobabilistic.operators.unary.UnaryOpLambda;

public class CombineDoubles extends Vertex<DoubleTensor[]> implements NonProbabilistic<DoubleTensor[]> {

    private DoubleVertex[] vertices;


    /**
     * Should this method accept 3 DoubleVertex's? To combine xPos, yPos and desiredSpeed?
     * Or is this method to be called repeatedly adding 1 vertex each iteration?
     * i.e. Vert1 + Vert2 = Vert12;
     *      Vert12 + Vert3 = Vert123;
     *      Vert123 + Vert4 = Vert1234 etc.
     *
     * @param x0
     * @param x1
     */
    public CombineDoubles(DoubleVertex x0, DoubleVertex x1) {
        super(new long[0]);
        setParents(x0, x1);
        vertices = new DoubleVertex[]{x0, x1};
    }

    @Override
    public DoubleTensor[] sample(KeanuRandom random) {
        //ignore
        return null;
    }

    @Override
    public DoubleTensor[] calculate() {
        DoubleTensor[] values = new DoubleTensor[vertices.length];

        for (int i=0; i < values.length; i++) {
            values[i] = vertices[i].getValue();
        }

        return values;
    }
}
