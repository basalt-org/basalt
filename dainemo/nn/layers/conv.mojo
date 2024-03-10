from tensor import TensorShape
# from random import rand
# from math import sqrt

from dainemo import Graph, Symbol, OP
from dainemo.autograd.attributes import AttributeVector, Attribute
# from dainemo.utils.tensorutils import rand_uniform



# def Conv2d(
#     inout g: Graph,
#     inputs: Symbol,
#     in_channels: Int,
#     out_channels: Int,
#     kernel_size: StaticIntTuple[2],
#     padding: StaticIntTuple[2] = 0,
#     stride: StaticIntTuple[2] = 1,
#     dilation: StaticIntTuple[2] = 1,
# ) -> Symbol:
#     """
#     A 2D Convolution Layer.

#     Parameters
#         inputs.shape     [batch, in_channels, X, Y]
#         kernel.shape     [out_channels, in_channels, X, Y] (or weights)
#         bias.shape       [out_channels].
#         output.shape     [batch, out_channels, X, Y].
#     """

#     # TODO proper initialization of weights and bias parameters
#     var weights = g.param(TensorShape(out_channels, in_channels, kernel_size[0], kernel_size[1]))
#     var bias = g.param(TensorShape(out_channels))

#     return g.op(OP.CONV2D, inputs, weights, bias, attributes=AttributeVector(
#         Attribute("padding", padding),
#         Attribute("stride", stride),
#         Attribute("dilation", dilation)
#     ))

    
#             rand_uniform[dtype, nelts](
#                 TensorShape(out_channels, in_channels, kernel_size.get[0, Int](), kernel_size.get[1, Int]()),
#                 -1/sqrt(k), 1/sqrt(k)
#             ),
#             trainable=True,
#             param=True,
#         )
#         self.bias = Node[dtype](
#             rand_uniform[dtype, nelts](TensorShape(out_channels), -1/sqrt(k), 1/sqrt(k)),
#             trainable=True, 
#             param=True
#         )



# # <------------CONV2D------------>
# struct Conv2d[
#     padding: StaticIntTuple[2] = 0,
#     stride: StaticIntTuple[2] = 1,
#     dilation: StaticIntTuple[2] = 1,
# ](Layer):
#     """
#     A 2D Convolution Layer.

#     Parameters
#         inputs.shape     [batch, in_channels, X, Y]
#         kernel.shape     [out_channels, in_channels, X, Y] (or weights)
#         bias.shape       [out_channels].
#         output.shape     [batch, out_channels, X, Y].
#     """

#     var weights: Node[dtype]
#     var bias: Node[dtype]

#     fn __init__(inout self, in_channels: Int, out_channels: Int, kernel_size: Int):
#         self.__init__(in_channels, out_channels, (kernel_size, kernel_size))

#     fn __init__(
#         inout self, in_channels: Int, out_channels: Int, kernel_size: Tuple[Int, Int]
#     ):
#         var k: SIMD[dtype, 1] = in_channels * kernel_size.get[0, Int]() * kernel_size.get[1, Int]()
#         self.weights = Node[dtype](
#             rand_uniform[dtype, nelts](
#                 TensorShape(out_channels, in_channels, kernel_size.get[0, Int](), kernel_size.get[1, Int]()),
#                 -1/sqrt(k), 1/sqrt(k)
#             ),
#             trainable=True,
#             param=True,
#         )
#         self.bias = Node[dtype](
#             rand_uniform[dtype, nelts](TensorShape(out_channels), -1/sqrt(k), 1/sqrt(k)),
#             trainable=True, 
#             param=True
#         )
#         GRAPH.add_node(self.weights)
#         GRAPH.add_node(self.bias)

#     fn forward(self, inputs: Node[dtype]) -> Node[dtype]:
#         """
#         Forward pass of the convolution layer.
#         """

#         # COPY self.weight & self.bias directly from GRAPH
#         # Workaround because model parameters are created and change in copies.
#         # TODO: Redo when lifetimes are there. [INVESTIGATE HOW TO AVOID THIS]
#         var weights = GRAPH.graph[GRAPH.get_node_idx(self.weights.uuid)]
#         var bias = GRAPH.graph[GRAPH.get_node_idx(self.bias.uuid)]

#         return CONV2D.forward[padding, stride, dilation](inputs, weights, bias)

#     fn __call__(self, inputs: Node[dtype]) -> Node[dtype]:
#         return self.forward(inputs)


# # <------------CONV3D------------>
# # TODO