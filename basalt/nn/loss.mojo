import basalt.nn as nn
from basalt import Tensor, TensorShape
from basalt import Graph, Symbol, OP
from basalt.utils.tensorutils import fill


struct Loss[g: Graph]:
    var module: nn.Module[g]

    fn __init__(inout self):
        self.module = nn.Module[g]()

    fn __call__(inout self, input: Tensor[dtype], target: Tensor[dtype]) -> Tensor[dtype]:
        return self.module.forward(input, target)[0]



fn MSELoss[
    graph: Graph,
    g: Graph = mse_loss(
        graph.outputs[0],
        graph.symbol_count
    )
]() -> Loss[g]:
    return Loss[g]()


fn mse_loss(input: Symbol, count_start: UInt32) -> Graph:
    # 1/N * sum( (input - targets)^2 )
    var g = Graph(count_start)
    g.input(input)
    var target = g.input(input.shape)

    var diff = g.op(OP.SUB, input, target)
    var loss = g.op(OP.POW, diff, 2)
    var mean_loss = g.op(OP.MEAN, loss)

    g.out(mean_loss)
    return g ^



fn CrossEntropyLoss[
    graph: Graph,
    g: Graph = cross_entropy_loss(
        graph.outputs[0],
        graph.symbol_count
    )
]() -> Loss[g]:
    return Loss[g]()


fn cross_entropy_loss(input: Symbol, count_start: UInt32) -> Graph:
    # -1/N * sum( targets * log_softmax(outputs) )
    var g = Graph()
    g.input(input)
    var target = g.input(input.shape)

    var log_softmax = nn.LogSoftmax(g, input, axis=1)

    # CrossEntropy (reduction Mean)
    var targets_log_softmax = g.op(OP.MUL, target, log_softmax)
    var ret = g.op(OP.SUM, targets_log_softmax)
    var negDivN = g.op(OP.MUL, ret, -1.0 / input.shape[0])

    g.out(negDivN)
    return g ^
