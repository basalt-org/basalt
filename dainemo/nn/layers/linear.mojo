from tensor import TensorShape

from dainemo import Graph, Symbol, OP


fn Linear(inout g: Graph,
    inputs: Symbol,
    n_outputs: Int,
) -> Symbol:
    """
    A fully connected layer.
    """

    var weights = g.param(TensorShape(inputs.static_shape[1], n_outputs), init="kaiming_normal")
    var b = g.param(TensorShape(n_outputs))
    var res = g.op(OP.DOT, inputs, weights)

    return g.op(OP.ADD, res, b)
