from random import rand
from time.time import now
import math

import basalt.nn as nn
from basalt import Tensor, TensorShape
from basalt import dtype
from basalt import Graph, Symbol, OP
from basalt.utils.tensorutils import fill
from basalt.autograd.attributes import Attribute, AttributeVector


fn create_simple_nn(batch_size: Int, n_inputs: Int, n_outputs: Int) -> Graph:
    var g = Graph()

    var x = g.input(TensorShape(batch_size, n_inputs))
    var y_true = g.input(TensorShape(batch_size, n_outputs))

    var x1 = nn.CKAN(g, x, 1, 8, 8)
    var x2 = nn.CKAN(g, x1, 8, 1, 8)
    var x3 = g.op(
        OP.RESHAPE,
        x2,
        attributes=AttributeVector(
            Attribute("shape", TensorShape(batch_size, n_outputs))
        ),
    )
    g.out(x3)

    var loss = nn.MSELoss(g, x3, y_true)
    g.loss(loss)

    g.compile()

    return g ^


fn main():
    alias batch_size = 32
    alias n_inputs = 1
    alias n_outputs = 1
    alias learning_rate = 0.01

    alias epochs = 20000

    alias graph = create_simple_nn(batch_size, n_inputs, n_outputs)

    # try: graph.render("operator")
    # except: print("Could not render graph")

    var model = nn.Model[graph]()
    var optimizer = nn.optim.Adam[graph](Reference(model.parameters), lr=learning_rate)

    var x_data = Tensor[dtype](batch_size, n_inputs)
    var y_data = Tensor[dtype](batch_size, n_outputs)

    print("Training started")
    var start = now()
    for i in range(epochs):
        rand[dtype](x_data.data(), x_data.num_elements())

        for j in range(batch_size):
            x_data[j] = x_data[j] * 2 - 1
            y_data[j] = math.sin(x_data[j])

        var out = model.forward(x_data, y_data)

        if (i + 1) % 1000 == 0:
            print("[", i + 1, "/", epochs, "] \tLoss: ", out[0])

        optimizer.zero_grad()
        model.backward()
        optimizer.step()

    print("Training finished: ", (now() - start) / 1e9, "seconds")
