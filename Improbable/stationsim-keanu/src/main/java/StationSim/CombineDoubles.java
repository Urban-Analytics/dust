package StationSim;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.operators.unary.UnaryOpLambda;

import java.util.List;

public class CombineDoubles extends Vertex<DoubleTensor[]> implements NonProbabilistic<DoubleTensor[]> {

    // Removed private tag from this to access and update more easily
    DoubleVertex[] vertices;

    /**
     * @param x0
     * @param x1
     */
    CombineDoubles(DoubleVertex x0, DoubleVertex x1) {
        super(new long[0]);
        setParents(x0, x1);
        vertices = new DoubleVertex[]{x0, x1};
    }

    /**
     *
     * @param doublesToAdd  A collection of DoubleVertex's to be combined
     */
    CombineDoubles(List<DoubleVertex> doublesToAdd) {
        super(new long[0]);
        setParents(doublesToAdd);
        vertices = new DoubleVertex[doublesToAdd.size()];

        for (int i=0; i < doublesToAdd.size(); i++) {
            vertices[i] = doublesToAdd.get(i);
        }
    }


    /**
     *
     * @param element   integer index of element required
     * @param input     Vertex<DoubleTensor[]> of input (StateVector)
     * @return
     */
    static Vertex<DoubleTensor> getAtElement(int element, Vertex<DoubleTensor[]> input) {

        return new UnaryOpLambda<>(
                new long[0],
                input,
                currentState -> currentState[element]
        );
    }

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


    int getLength() { return vertices.length; }
}
