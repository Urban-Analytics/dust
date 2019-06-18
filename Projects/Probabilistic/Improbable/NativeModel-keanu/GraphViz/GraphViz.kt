package GraphViz;

import io.improbable.keanu.algorithms.graphtraversal.DiscoverGraph
import io.improbable.keanu.network.BayesianNetwork
import io.improbable.keanu.vertices.Vertex

fun BayesianNetwork.toGraphvizString(vertexNames : HashMap<Vertex<Any>,String> = hashMapOf()) : String {
    var dot = "digraph bayesNet {\n"
    if(latentAndObservedVertices.size != 0) {
        val allVertices = DiscoverGraph.getEntireGraph(latentAndObservedVertices[0])
        allVertices.forEachIndexed({ i, v ->
            if(!vertexNames.containsKey(v)) vertexNames[v] = "  ${v.javaClass.simpleName}${i}"
        })

        dot += "  {\n"
        observedVertices.forEach({v ->
            dot += "    ${vertexNames[v]} [style=filled fillcolor=yellow]\n"
        })
        latentVertices.forEach({v ->
            dot += "    ${vertexNames[v]} [style=filled fillcolor=red]\n"
        })
        dot += "  }\n"

        allVertices.forEach({parent ->
            parent.children.forEach({child ->
                dot += "${vertexNames[parent]} -> ${vertexNames[child]}\n"
            })
        })
    }
    return dot+"}\n"
}