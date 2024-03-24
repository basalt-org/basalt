# from math import sqrt
from basalt import Tensor, TensorShape

from basalt import Graph, Symbol, OP
from basalt.autograd.params import Param
from basalt.autograd.attributes import AttributeVector, Attribute


# BUG: Mojo 24.1.0 does not support the comp time `sqrt` function
@always_inline
fn sqrt[type: DType](value: SIMD[type, 1]) -> SIMD[type, 1]:
    """Returns the square root of the input simd vector."""
    if value == 0: return 0
    elif value < 0: return nan[type]()
    var start = value if value > 1 else 1/value
    var a: SIMD[type,1] = start
    var b: SIMD[type,1] = (a + 1) / 2
    while b < a:
        a = b
        b = (a + start/a) / 2
    return a if value > 1 else 1/a


fn Conv2d( inout g: Graph,
    inputs: Symbol,
    out_channels: Int,
    kernel_size: StaticIntTuple[2],
    padding: StaticIntTuple[2] = 0,
    stride: StaticIntTuple[2] = 1,
    dilation: StaticIntTuple[2] = 1,
) -> Symbol:
    """
    A 2D Convolution Layer.

    Parameters
        inputs.shape     [batch, in_channels, iX, iY]
        kernel.shape     [out_channels, in_channels, kX, kY] (or weights)
        bias.shape       [out_channels].
        output.shape     [batch, out_channels, oX, oY].
    """

    var in_channels: Int = inputs.shape[1]
    var fan_in: SIMD[dtype, 1] = in_channels * kernel_size[0] * kernel_size[1]
    var bound = 1/sqrt(fan_in)
    var weights = g.param(
        TensorShape(out_channels, in_channels, kernel_size[0], kernel_size[1]),
        init=Param("random_uniform", -bound, bound)
        # init=Param("kaiming_uniform", 0)
    )
    var bias = g.param(
        TensorShape(out_channels),
        init=Param("random_uniform", -bound, bound)
    )

    return g.op(OP.CONV2D, inputs, weights, bias, attributes=AttributeVector(
        Attribute("padding", padding),
        Attribute("stride", stride),
        Attribute("dilation", dilation)
    ))

