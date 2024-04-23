from math import nan
from basalt import Tensor, TensorShape

from basalt import Graph, Symbol, OP
from basalt.autograd.params import Param
from basalt.autograd.attributes import AttributeVector, Attribute


@always_inline("nodebug")
fn q_sqrt(value: Float32) -> Float32:
    var y = bitcast[DType.float32](0x5F3759DF - (bitcast[DType.uint32](value) >> 1))
    return y * (1.5 - 0.5 * value * y * y)


fn Conv2d(
    inout g: Graph,
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
    var bound = q_sqrt(fan_in)
    var weights = g.param(
        TensorShape(out_channels, in_channels, kernel_size[0], kernel_size[1]),
        init=Param("random_uniform", -bound, bound)
        # init=Param("kaiming_uniform", 0)
    )
    var bias = g.param(
        TensorShape(out_channels), init=Param("random_uniform", -bound, bound)
    )

    return g.op(
        OP.CONV2D,
        inputs,
        weights,
        bias,
        attributes=AttributeVector(
            Attribute("padding", padding),
            Attribute("stride", stride),
            Attribute("dilation", dilation),
        ),
    )
