import io.improbable.keanu.algorithms.mcmc.MetropolisHastings
import io.improbable.keanu.algorithms.variational.GradientOptimizer
import io.improbable.keanu.network.BayesNet
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex

/**
 * Created by nick on 23/05/2018.
 */


fun main(args:Array<String> ) {
    // Thermometer example. Have two thermometers, thermol1 and 2,
    // and a real temperature (T) that we don't know.

    // T is the real temperature, expressed here as a normal distribution
    // (our prior belief about what the temperature is)
    // This is a *hidden* variable.
    var T = GaussianVertex(20.0,5.0)

    // Two thermometers, and a Gaussian representing our prior belief:
    // They will measure the real temperature plus 1.0 degrees of noise.
    var thermol1 = GaussianVertex(T,1.0)
    var thermol2 = GaussianVertex(T,1.0)

    // Observe some temperatures
    thermol1.observe(20.6)
    thermol2.observe(19.6)

    // Useful: partial derivative of thermol1 with respect to T
    // (i.e. how much thermol1 varies with T)

    // Useful: can add, subtract, etc
    var Tsum = (thermol1 + thermol2)
    println("TSum sample: "+Tsum.sample())
    println("Tminus sample: "+(thermol1 - thermol2).sample())
    // And do deriviatives: e.g. partial derivative of Tsum with
    // respect to thermol1
    println("Partial deriv:"+Tsum.dualNumber.partialDerivatives.withRespectTo(thermol1))

    // What is the most probable value of T, given these observations
    // and the priors (the 'maxiumum a posteriori probability')

    // Create a model (Bayes net)
    var model = BayesNet(T.connectedGraph)

    // Optimise (?) (a way of making the model more linear?)
    var optimiser = GradientOptimizer(model)

    // Find the max a posteriori. This will set the whole model
    // to the state that is the most probable - all variables
    // will be set to their most probable state
    optimiser.maxAPosteriori(1000)

    println("Estimate for T: "+ T.value)

    // Now sample from the posterior distribution to learn about the
    // posterior (in most cases these are very complex so sampling is
    // needed for inference)
    // Give it the model, the parameters that we're sampling from
    // (just T in this case) and the number of samples
    val samples = MetropolisHastings.getPosteriorSamples(model, listOf(T), 10000)

    // Drop a few (burn in)
    samples.drop(1000)

    // Down sample (take every xth sample, because many samples will be close to each other
    samples.downSample(5)
    //println("Samples from posterior of T:"+samples.get(T).asList())
    println("Mode of T: "+samples.get(T).mode)











}