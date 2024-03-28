from random import rand
from time.time import now
import math

import basalt.nn as nn
from basalt import Tensor, TensorShape
from basalt import dtype
from basalt import Graph, Symbol, OP
from basalt.utils.tensorutils import fill


fn create_simple_nn(batch_size: Int, n_inputs: Int, n_outputs: Int) -> Graph:
    var g = Graph()

    var x = g.input(TensorShape(batch_size, n_inputs))
    var y_true = g.input(TensorShape(batch_size, n_outputs))

    var x1 = nn.Linear(g, x, n_outputs=32)
    var x2 = nn.ReLU(g, x1)
    var x3 = nn.Linear(g, x2, n_outputs=32)
    var x4 = nn.ReLU(g, x3)
    var y_pred = nn.Linear(g, x4, n_outputs=n_outputs)
    g.out(y_pred)

    var loss = nn.MSELoss(g, y_pred, y_true)
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

    # try:
    #     graph.render("operator")
    # except e:
    #     print("Failed to render graph")
    #     print(e)

    var model = nn.Model[graph]()
    var optimizer = nn.optim.Adam[graph](lr=learning_rate)
    optimizer.allocate_rms_and_momentum(model.parameters)

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

        optimizer.zero_grad(model.parameters)
        model.backward()
        optimizer.step(model.parameters)

    print("Training finished: ", (now() - start) / 1e9, "seconds")
