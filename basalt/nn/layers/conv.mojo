from basalt import Graph, Symbol, OP
from basalt import Tensor, TensorShape
from basalt.utils import q_sqrt
from basalt.autograd.params import Param
from basalt.autograd.attributes import AttributeVector, Attribute

from utils.index import IndexList


fn Conv2d(
    inout g: Graph,
    inputs: Symbol,
    out_channels: Int,
    kernel_size: IndexList[2],
    padding: IndexList[2] = 0,
    stride: IndexList[2] = 1,
    dilation: IndexList[2] = 1,
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
    var fan_in: Scalar[dtype] = in_channels * kernel_size[0] * kernel_size[1]
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
