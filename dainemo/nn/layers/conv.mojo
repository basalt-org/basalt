from tensor import TensorShape

from dainemo import Graph, Symbol, OP
from dainemo.autograd.attributes import AttributeVector, Attribute


def Conv2d( inout g: Graph,
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

    var in_channels: Int = inputs.static_shape[1]
    var weights = g.param(TensorShape(out_channels, in_channels, kernel_size[0], kernel_size[1]), init="kaiming_normal")
    var bias = g.param(TensorShape(out_channels))

    return g.op(OP.CONV2D, inputs, weights, bias, attributes=AttributeVector(
        Attribute("padding", padding),
        Attribute("stride", stride),
        Attribute("dilation", dilation)
    ))


# # <------------CONV3D------------>
# # TODO