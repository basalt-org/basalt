from tensor import TensorShape
from random import rand

from dainemo import GRAPH
from dainemo.nn.layers import Layer
from dainemo.autograd.node import Node
from dainemo.autograd.ops.pool import MAXPOOL2D


# <------------MAXPOOL2D------------>

struct MaxPool2d[
    in_channels: Int,
    kernel_size: StaticIntTuple[2],
    padding: StaticIntTuple[2] = 0,
    stride: StaticIntTuple[2] = 1,
    dilation: StaticIntTuple[2] = 1
](Layer):
    """
    A 2D Max Pooling Layer.

    Since out_channels == in_channels
    kernel.shape     [in_channels, in_channels, X, Y]
    """
    alias kernel_shape = TensorShape(in_channels, in_channels, kernel_size[0], kernel_size[1])

    fn __init__(inout self):
        # padding should be at most half of the kernel size
        # TODO: assert padding <= kernel_size / 2 (at compile time)
        pass

    fn forward(self, inputs: Node[dtype]) -> Node[dtype]:
        """
        Forward pass of the MaxPool2d layer.
        """
        return MAXPOOL2D.forward[self.kernel_shape, padding, stride, dilation](inputs)

    fn __call__(self, inputs: Node[dtype]) -> Node[dtype]:
        return self.forward(inputs)


# <------------MAXPOOL3D------------>
# TODO