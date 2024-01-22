from tensor import Tensor
from random import rand

from dainemo import GRAPH
from dainemo.nn.layers import Layer
from dainemo.autograd.node import Node
from dainemo.autograd.ops.conv import CONV2D


# <------------CONV2D------------>
struct Conv2d[
    padding: StaticIntTuple[2] = 0,
    stride: StaticIntTuple[2] = 1,
    dilation: StaticIntTuple[2] = 1,
](Layer):
    """
    A 2D Convolution Layer.

    Parameters
        inputs.shape     [batch, in_channels, X, Y]
        kernel.shape     [out_channels, in_channels, X, Y] (or weights)
        bias.shape       [out_channels].
        output.shape     [batch, out_channels, X, Y].
    """

    var weights: Node[dtype]
    var bias: Node[dtype]

    fn __init__(inout self, in_channels: Int, out_channels: Int, kernel_size: Int):
        self.weights = Node[dtype](
            rand[dtype](out_channels, in_channels, kernel_size, kernel_size),
            requires_grad=True,
            param=True,
        )
        self.bias = Node[dtype](
            Tensor[dtype](out_channels), requires_grad=True, param=True
        )
        GRAPH.add_node(self.weights)
        GRAPH.add_node(self.bias)

    fn __init__(
        inout self, in_channels: Int, out_channels: Int, kernel_size: Tuple[Int, Int]
    ):
        self.weights = Node[dtype](
            rand[dtype](
                out_channels,
                in_channels,
                kernel_size.get[0, Int](),
                kernel_size.get[1, Int](),
            ),
            requires_grad=True,
            param=True,
        )
        self.bias = Node[dtype](
            Tensor[dtype](out_channels), requires_grad=True, param=True
        )
        GRAPH.add_node(self.weights)
        GRAPH.add_node(self.bias)

    fn forward(self, inputs: Node[dtype]) -> Node[dtype]:
        """
        Forward pass of the convolution layer.
        """

        # COPY self.weight & self.bias directly from GRAPH
        # Workaround because model parameters are created and change in copies.
        # TODO: Redo when lifetimes are there. [INVESTIGATE HOW TO AVOID THIS]
        let weights = GRAPH.graph[GRAPH.get_node_idx(self.weights.uuid)]
        let bias = GRAPH.graph[GRAPH.get_node_idx(self.bias.uuid)]

        return CONV2D.forward[padding, stride, dilation](inputs, weights, bias)

    fn __call__(self, inputs: Node[dtype]) -> Node[dtype]:
        return self.forward(inputs)


# <------------CONV3D------------>
# TODO