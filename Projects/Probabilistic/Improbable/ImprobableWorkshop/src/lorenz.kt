import io.improbable.keanu.algorithms.mcmc.MetropolisHastings
import io.improbable.keanu.algorithms.variational.GradientOptimizer
import io.improbable.keanu.vertices.dbl.DoubleVertex
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex
import java.util.Arrays.asList

import io.improbable.keanu.kotlin.*
import io.improbable.keanu.network.BayesNet

/**
 * Created by nick on 23/05/2018.
 * Example shows how to assimilate data into a chaotic system (the Lorenz equations)
 */

fun main (args: Array<String>) {

    // Initial conditions for the lorenz equations
    val x = GaussianVertex(20.0, 1.0)
    val y = GaussianVertex(19.0, 1.0)
    val z = GaussianVertex(50.0, 1.0)
    // The state consists of values for x, y, and z
    var state : MutableList<DoubleVertex> = asList(x,y,z)

    // Five noisy observations of the x coordinate
    val observations = asList(19.670, 19.316, 18.179, 16.088, 15.144)

    for (step in 0..4) { // Iterate Lorenz. We have one observation for each iteration
        println(""+step+" - x: "+state[0].value+"y: "+state[1].value+"z: "+state[2].value)

        // Step the lorenz model through one iteration
        state = lorenzTimestep(state[0], state[1], state[2])

        // Create a Gaussian to represent our observations, this allows to represent noice
        val observation = GaussianVertex(state[0], 1.0)
        observation.observe(observations[step])


    }

    // Make a model from these observations and priors
    var model = BayesNet(state[0].connectedGraph)
    val optimiser = GradientOptimizer(model)
    optimiser.maxAPosteriori(1000)

    // Final estimates for the states:
    println("x: ${state[0].value} y: ${state[1].value} z: ${state[2].value}")

    // Do some sampling
    val samples = MetropolisHastings.getPosteriorSamples(
            model, listOf(state[0],state[1],state[2]), 10000)
            .drop(1000).downSample(5)
    println("Modes: of T: ${samples.get(state[0]).mode}, ${samples.get(state[1]).mode}, ${samples.get(state[2]).mode}")

}

fun lorenzTimestep(x:DoubleVertex, y:DoubleVertex, z:DoubleVertex) : MutableList<DoubleVertex> {

    val sigma = 10.0
    val beta = 8.0/3.0
    val rho = 28.0
    val dt = 0.01

    val dxdt = sigma*(y-x)
    val dydt = x*(rho - z) - y
    val dzdt = x*y - beta*z

    return asList(x+dxdt*dt, y+dydt*dt, z+dzdt*dt)

}