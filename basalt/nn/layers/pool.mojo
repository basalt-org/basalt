from basalt import Tensor, TensorShape
from collections.optional import Optional
from utils.index import IndexList

from basalt import Graph, Symbol, OP
from basalt.autograd.attributes import AttributeVector, Attribute


fn set_static_stride(
    kernel_size: IndexList[2], stride: Optional[Int] = None
) -> IndexList[2]:
    if stride:
        return IndexList[2](stride.value(), stride.value())
    else:
        return kernel_size


fn MaxPool2d(
    inout g: Graph,
    inputs: Symbol,
    kernel_size: IndexList[2],
    stride: Optional[Int] = None,
    padding: IndexList[2] = 0,
    dilation: IndexList[2] = 1,
) -> Symbol:
    """
    A 2D Max Pooling Layer.

    Kernel is unaware of the in_channels and out_channels of the input tensor.
    kernel.size     (kX, kY)
    """

    # TODO: assert padding <= kernel_size / 2 (at compile time)

    var stride_temp = set_static_stride(kernel_size, stride)

    return MaxPool2d(g, inputs, kernel_size, stride_temp, padding, dilation)


fn MaxPool2d(
    inout g: Graph,
    inputs: Symbol,
    kernel_size: IndexList[2],
    stride: IndexList[2],  # stride should be 1 or more
    padding: IndexList[2] = 0,
    dilation: IndexList[2] = 1,
) -> Symbol:
    """
    A 2D Max Pooling Layer.

    Kernel is unaware of the in_channels and out_channels of the input tensor.
    kernel.size     (kX, kY)
    """
    # TODO: assert padding <= kernel_size / 2 (at compile time)

    return g.op(
        OP.MAXPOOL2D,
        inputs,
        attributes=AttributeVector(
            Attribute("kernel_size", kernel_size),
            Attribute("padding", padding),
            Attribute("stride", stride),
            Attribute("dilation", dilation),
        ),
    )


# # TODO
