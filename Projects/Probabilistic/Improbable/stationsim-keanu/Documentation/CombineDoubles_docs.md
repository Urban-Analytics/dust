# CombineDoubles.java

This is a class that Improbable wrote for us (modified since then, original [here](https://github.com/improbable-research/keanu/issues/471)). It is an object that acts as a way of collecting multiple vertices together into a single `Vertex<? extends Tensor[]>` object. This singular `Vertex<>` object (in our case `Vertex<DoubleTensor[]>`) can then run Keanus algorithms over the whole collection of vertices (the state) at once. 

## Field

The only field is the array of `DoubleVertex`'s that make up the CombineDoubles object. This field used to be private (when Improbable first gave it to us) meaning the vertices were hidden away and the only way to access the info was from using the `calculate()` method. I changed this for one main reason (although it made a lot of things more simple overall). The reason was that when calling `getAtElement()` or `calculate()`, I don't think the value that is returned is a part of the CombineDoubles object. To elaborate, I think a new object is created when calling `getAtElement()` for example, and held in a separate space in the memory. Therefore, when this new object is observed for example, only the value of the object would change and not its corresponding `DoubleVertex/DoubleTensor` in here. That meant this method was useless, as it was not saving any modifications made on it by other algorithms. 

`DoubleVertex[] vertices;`

## Constructor

**This could be the problem with the whole model**

This is a method Improbable made for us that we changed slightly. (can see the original suggestion on the [Github issue](<https://github.com/improbable-research/keanu/issues/471>)). The original created an object that combined only 2 vertices, so we modified it a bit to accept a full state vector. 

```java

    CombineDoubles(List<DoubleVertex> doublesToAdd) {
        super(new long[0]);
        setParents(doublesToAdd);
        vertices = new DoubleVertex[doublesToAdd.size()];

        for (int i=0; i < doublesToAdd.size(); i++) {
            vertices[i] = doublesToAdd.get(i);
        }
    }
```

`setParents(doublesToAdd)` could be the problem with the BayesNet. The `BayesianNetwork` is supposed to be a representation of the model, so I think setting parent and child nodes correctly is important. I'm not clear on how they are supposed to relate to each other so not sure what is best to do at this point. For example, each time we build this do we need to set the previous collection as the parents? Then the `BayesianNetwork` will be related to all previous nets? Reason we know (or strongly suspect) that the `BayesianNetwork` is not correct is that when we try to optimise after building the network, Keanu does a fitness evaluation on the model. It gives a LogProb of negative infinity, meaning that the probability of the model being in that state is zero. More detailed explanation in DataAssimilation_docs.md. 

## getAtElement()

```java

    static Vertex<DoubleTensor> getAtElement(int element, Vertex<DoubleTensor[]> input) {

        return new UnaryOpLambda<>(
                new long[0],
                input,
                currentState -> currentState[element]
        );
    }

```

This method allows us to extract a single `DoubleTensor` object from the `DoubleTensor[]`, wrapped once again inside a Vertex. It might be important to mention that `DoubleVertex` is an extension of `Vertex<DoubleTensor>`, although you cannot cast between the two and I haven't figured out how or if its important.

## sample()

This method is unused. In a previous version of Keanu, a class that extended Vertex<DoubleTensor[]> has to override the sample() method, but this seems to have changed in the newer versions. I left it anyway as it wasn't causing any trouble and I didn't know if it was important. 

## calculate()

```java

    @Override
    public DoubleTensor[] calculate() {
        DoubleTensor[] values = new DoubleTensor[vertices.length];

        for (int i=0; i < values.length; i++) {
            values[i] = vertices[i].getValue();
        }
        return values;
    }

```

This is important for collecting the metrics at the end. It evaluates all the vertices in the object to `DoubleTensor`s, and returns them in an array. 

## getLength()

Simple method to get the number of vertices in the object. 