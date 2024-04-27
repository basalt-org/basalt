from basalt import Tensor, TensorShape
from basalt import Graph, Symbol, OP
from basalt.utils import q_sqrt
from basalt.autograd.params import Param


fn Linear(
    inout g: Graph,
    inputs: Symbol,
    n_outputs: Int,
) -> Symbol:
    """
    A fully connected layer.
    """

    var fan_in: Scalar[dtype] = inputs.shape[1]
    var bound = q_sqrt(fan_in)
    var weights = g.param(
        TensorShape(inputs.shape[1], n_outputs),
        init=Param("random_uniform", -bound, bound)
        # init=Param("random_uniform", 1) # NOTE: mode: fan_out required as weight are defined transposed
    )
    var b = g.param(TensorShape(n_outputs), init=Param("random_uniform", -bound, bound))

    var res = g.op(OP.DOT, inputs, weights)
    return g.op(OP.ADD, res, b)
