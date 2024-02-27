# from tensor import TensorShape
# from random import rand
# from collections.optional import Optional

# from dainemo import GRAPH
# from dainemo.nn.layers import Layer
# from dainemo.autograd.node import Node
# from dainemo.autograd.ops.pool import MAXPOOL2D


# # <------------MAXPOOL2D------------>
# fn set_static_stride(kernel_size: StaticIntTuple[2], stride: Optional[Int] = None) -> StaticIntTuple[2]:
#     if stride:
#         return StaticIntTuple[2](stride.value(), stride.value())
#     else:
#         return kernel_size


# struct MaxPool2d[
#     kernel_size: StaticIntTuple[2],
#     stride: Optional[Int] = None,
#     padding: StaticIntTuple[2] = 0,
#     dilation: StaticIntTuple[2] = 1
# ](Layer):
#     """
#     A 2D Max Pooling Layer.

#     Kernel is unaware of the in_channels and out_channels of the input tensor.
#     kernel.shape     [_, _, X, Y]
#     """
#     alias input_stride: StaticIntTuple[2] = set_static_stride(kernel_size, stride)

#     fn __init__(inout self):
#         # padding should be at most half of the kernel size
#         # TODO: assert padding <= kernel_size / 2 (at compile time)
#         pass

#     fn forward(self, inputs: Node[dtype]) -> Node[dtype]:
#         """
#         Forward pass of the MaxPool2d layer.
#         """

#         return MAXPOOL2D.forward[kernel_size, self.input_stride, padding, dilation](inputs)


#     fn __call__(self, inputs: Node[dtype]) -> Node[dtype]:
#         return self.forward(inputs)


# # <------------MAXPOOL3D------------>
# # TODO