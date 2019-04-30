package StationSim;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.graphtraversal.VertexValuePropagation;
import io.improbable.keanu.algorithms.mcmc.MetropolisHastings;
import io.improbable.keanu.network.BayesianNetwork;
//import io.improbable.keanu.research.vertices.DoubleArrayIndexingVertex;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.operators.unary.UnaryOpLambda;

import java.util.Arrays;

public class MHTest {
    public static int numOutputs = 500;
    public static int numEvaluations = 0;


    public static Double [] run(DoubleTensor in) {
        System.out.println("Evaluation "+MHTest.numEvaluations++ +" starting");

        Double [] numPeople = new Double[numOutputs];
        for (int i = 0; i < numOutputs; i++) {
            numPeople[i] = 0.0;
        }
        return numPeople;
    }

/*
    public static void main(String[] args) {
        DoubleVertex input = new GaussianVertex(0.0,1.0);
        UnaryOpLambda<DoubleTensor,Double []> box = new UnaryOpLambda<>( input, MHTest::run);

        for (Integer i = 0; i< numOutputs; i++) {
            DoubleArrayIndexingVertex output = new DoubleArrayIndexingVertex(box, i);
            new GaussianVertex(output, 1.0).observe(0.0);
        }
        input.setValue(0.0);
        //input2.setValue(21123);
        VertexValuePropagation.cascadeUpdate(input);
        BayesianNetwork testNet = new BayesianNetwork(box.getConnectedGraph());
        NetworkSamples samples = MetropolisHastings.getPosteriorSamples( testNet, Arrays.asList(box), 40 );

    } */



}
