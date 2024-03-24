from random import rand
from time.time import now

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

    alias graph = create_simple_nn(batch_size, n_inputs, n_outputs)

    var model = nn.Model[graph]()
    var optimizer = nn.optim.Adam[graph](lr=learning_rate)
    optimizer.allocate_rms_and_momentum(model.parameters)

    var x = Tensor[dtype](batch_size, n_inputs)
    rand[dtype](x.data(), x.num_elements())
    var y = Tensor[dtype](batch_size, n_outputs)

    for i in range(batch_size):
        y[i] = math.sin(x[i])

    print("Training started")
    var start = now()
    
    alias epochs = 10000
    for i in range(epochs):
        var out = model.forward(x, y)

        if i % 1000 == 0:
            print("[", i + 1, "/", epochs,"] \tLoss: ", out[0])

        optimizer.zero_grad(model.parameters)
        model.backward()
        optimizer.step(model.parameters)

    print("Training finished: ", (now() - start)/1e9, "seconds")