from math import nan

from basalt.autograd.params import Param


# BUG: Mojo 24.1.0 does not support the comp time `sqrt` function


fn sqrt[Type: DType](value: SIMD[Type, 1]) -> SIMD[Type, 1]:
    """
    Returns the square root of the input simd vector.
    """
    if value == 0:
        return 0
    elif value < 0:
        return nan[Type]()
    var start = value if value > 1 else 1 / value
    var a: SIMD[Type, 1] = start
    var b: SIMD[Type, 1] = (a + 1) / 2
    while b < a:
        a = b
        b = (a + start / a) / 2
    return a if value > 1 else 1 / a


fn Linear(
    inout g: Graph,
    inputs: Symbol,
    n_outputs: Int,
) -> Symbol:
    """
    A fully connected layer.
    """

    var fan_in: SIMD[dtype, 1] = inputs.shape[1]
    var bound = 1 / sqrt(fan_in)
    var weights = g.param(
        TensorShape(inputs.shape[1], n_outputs),
        init=Param("random_uniform", -bound, bound)
        # init=Param("random_uniform", 1) # NOTE: mode: fan_out required as weight are defined transposed
    )
    var b = g.param(TensorShape(n_outputs), init=Param("random_uniform", -bound, bound))

    var res = g.op(OP.DOT, inputs, weights)
    return g.op(OP.ADD, res, b)
