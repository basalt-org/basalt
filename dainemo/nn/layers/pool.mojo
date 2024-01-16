from random import rand

from dainemo import GRAPH
from dainemo.autograd.node import Node
from dainemo.autograd.ops.pool import MAXPOOL2D


# <------------MAXPOOL2D------------>

# struct MaxPool2d[
#     padding: Int,
#     stride: Int,
#     kernel_size: Int,
# ]:
#     """
#     A 2D Max Pooling Layer.
#     """

#     var weight: Node[dtype]

#     fn __init__(inout self, in_channels: Int):
#         MaxPool2d[padding, stride, (kernel_size, kernel_size)](in_channels)


# struct MaxPool2d[
#     padding: Int,
#     stride: Int,
#     kernel_size: Tuple[Int, Int],
# ]:
#     """
#     A 2D Max Pooling Layer.
#     """

#     fn __init__(inout self, in_channels: Int):
#         # padding should be at most half of the kernel size
#         # TODO: assert padding <= kernel_size / 2
#         # out_channels == in_channels --> kernel (in_channels, in_channels, kernel_size, kernel_size)
#         self.weight = Node[dtype](
#             rand[dtype](in_channels, in_channels, kernel_size, kernel_size)
#         )


# <------------MAXPOOL3D------------>