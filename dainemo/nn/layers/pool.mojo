from tensor import TensorShape

from dainemo import Graph, Symbol, OP
from dainemo.autograd.attributes import AttributeVector, Attribute


# <------------MAXPOOL2D------------>
fn MaxPool2d(inout g: Graph,
    inputs: Symbol,
    kernel_size: StaticIntTuple[2],
    padding: StaticIntTuple[2] = 0,
    stride: StaticIntTuple[2] = 1,
    dilation: StaticIntTuple[2] = 1,
) -> Symbol:
    """
    A 2D Max Pooling Layer.

    Kernel is unaware of the in_channels and out_channels of the input tensor.
    kernel.size     (kX, kY)
    """
    # TODO: assert padding <= kernel_size / 2 (at compile time)

    return g.op(OP.MAXPOOL2D, inputs, attributes=AttributeVector(
        Attribute("kernel_size", kernel_size),
        Attribute("padding", padding),
        Attribute("stride", stride),
        Attribute("dilation", dilation)
    ))


# # <------------MAXPOOL3D------------>
# # TODO