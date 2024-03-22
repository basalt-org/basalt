from random import rand
from time.time import now

import basalt.nn as nn
from basalt import dtype
from basalt import Tensor, TensorShape
from basalt import Graph, Symbol, OP
from basalt.utils.tensorutils import fill

from basalt.autograd.attributes import Attribute, AttributeVector


fn mse(inout g: Graph, y_true: Symbol, y_pred: Symbol) -> Symbol:

    var diff = g.op(OP.SUB, y_true, y_pred)
    var loss = g.op(OP.MUL, diff, diff)
    var mean_loss = g.op(OP.MEAN, loss, None)

    return mean_loss


fn create_linear_graph(batch_size: Int, n_inputs: Int, n_outputs: Int) -> Graph:
    var g = Graph()

    var x = g.input(TensorShape(batch_size, n_inputs))
    var y_true = g.input(TensorShape(batch_size, n_outputs))

    var W = g.param(TensorShape(n_inputs, n_outputs))
    var b = g.param(TensorShape(n_outputs))

    var res = g.op(OP.DOT, x, W)

    var y_pred = g.op(OP.ADD, res, b)
    g.out(y_pred)

    var loss = mse(g, y_true, y_pred)
    g.loss(loss)

    g.compile()

    return g ^


fn main():
    alias batch_size = 64
    alias n_inputs = 22
    alias n_outputs = 1
    alias learning_rate = 1e-4

    alias graph = create_linear_graph(batch_size, n_inputs, n_outputs)

    # try:
    #     graph.render("operator")  # also try: "operator"
    # except e:
    #     print("Error rendering graph")
    #     print(e)

    # var model = nn.Model[graph]()
    # var optimizer = nn.optim.Adam[graph](lr=learning_rate)
    # optimizer.allocate_rms_and_momentum(model.parameters)

    # # Dummy data
    # var x = rand[dtype](batch_size, n_inputs)
    # var y = rand[dtype](batch_size, n_outputs)

    # print("Training started")
    # var start = now()
    
    # alias epochs = 1000
    # for i in range(epochs):
    #     var out = model.forward(x, y)
    #     print("[", i + 1, "/", epochs,"] \tLoss: ", out[0])

    #     # Backward pass
    #     optimizer.zero_grad(model.parameters)
    #     model.backward()
    #     optimizer.step(model.parameters)

    # print("Training finished: ", (now() - start)/1e9, "seconds")