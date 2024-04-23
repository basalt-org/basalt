from math import nan
from memory.unsafe import bitcast

from basalt import Tensor, TensorShape
from basalt import Graph, Symbol, OP
from basalt.autograd.params import Param


@always_inline("nodebug")
fn q_sqrt(value: Float32) -> Float32:
    var y = bitcast[DType.float32](0x5F3759DF - (bitcast[DType.uint32](value) >> 1))
    return y * (1.5 - 0.5 * value * y * y)


fn Linear(
    inout g: Graph,
    inputs: Symbol,
    n_outputs: Int,
) -> Symbol:
    """
    A fully connected layer.
    """

    var fan_in: Float64 = inputs.shape[1]
    var bound = q_sqrt(fan_in)
    var weights = g.param(
        TensorShape(inputs.shape[1], n_outputs),
        init=Param("random_uniform", -bound, bound)
        # init=Param("random_uniform", 1) # NOTE: mode: fan_out required as weight are defined transposed
    )
    var b = g.param(TensorShape(n_outputs), init=Param("random_uniform", -bound, bound))

    var res = g.op(OP.DOT, inputs, weights)
    return g.op(OP.ADD, res, b)
