from random import rand
from time.time import now
from tensor import TensorShape

import basalt.nn as nn
from basalt import dtype
from basalt import Graph, Symbol, OP
from basalt.utils.tensorutils import fill


fn mse(inout g: Graph, y_true: Symbol, y_pred: Symbol) -> Symbol:

    var diff = g.op(OP.SUB, y_true, y_pred)
    var loss = g.op(OP.MUL, diff, diff)
    var mean_loss = g.op(OP.MEAN, loss, None)

    return mean_loss ^


fn create_linear_graph(batch_size: Int, n_inputs: Int, n_outputs: Int) -> Graph:
    var g = Graph()

    var x = g.input(TensorShape(batch_size, n_inputs))
    var y_true = g.input(TensorShape(batch_size, n_outputs))

    var W1 = g.param(TensorShape(n_inputs, batch_size))
    var W2 = g.param(TensorShape(batch_size, batch_size))
    var W3 = g.param(TensorShape(batch_size, n_outputs))
    
    var b1 = g.param(TensorShape(batch_size))
    var b2 = g.param(TensorShape(batch_size))
    var b3 = g.param(TensorShape(n_outputs))

    var res = g.op(OP.DOT, x, W1)
    res = g.op(OP.ADD, res, b1)
    res = g.op(OP.RELU, res)

    res = g.op(OP.DOT, res, W2)
    res = g.op(OP.ADD, res, b2)
    res = g.op(OP.RELU, res)

    res = g.op(OP.DOT, res, W3)
    res = g.op(OP.ADD, res, b3)

    var y_pred = res

    var loss = mse(g, y_true, y_pred)
    g.loss(loss)

    g.compile()

    return g ^


fn main():
    alias batch_size = 32
    alias n_inputs = 1
    alias n_outputs = 1
    alias learning_rate = 0.01

    alias graph = create_linear_graph(batch_size, n_inputs, n_outputs)

    var model = nn.Model[graph]()
    var optimizer = nn.optim.Adam[graph](lr=learning_rate)
    optimizer.allocate_rms_and_momentum(model.parameters)

    var x = rand[dtype](batch_size, n_inputs)
    var y = rand[dtype](batch_size, n_outputs)

    for i in range(batch_size):
        y[i] = math.sin[dtype, 1](15.0 * i)

    print("Training started")
    var start = now()
    
    alias epochs = 200000
    for i in range(epochs):
        var out = model.forward(x, y)

        if i % 1000 == 0:
            print("[", i + 1, "/", epochs,"] \tLoss: ", out[0])

        optimizer.zero_grad(model.parameters)
        model.backward()
        optimizer.step(model.parameters)

    print("Training finished: ", (now() - start)/1e9, "seconds")