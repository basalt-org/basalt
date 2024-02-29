# from python.python import Python
# from tensor import Tensor, TensorShape
# from testing import assert_equal
# from utils.index import Index
# from random import rand

# from dainemo import GRAPH
# from dainemo.autograd.ops.conv import CONV2D
# from dainemo.autograd.ops.conv import get_result_shape
# from dainemo.utils.tensorutils import fill
# from test_tensorutils import assert_tensors_equal


# alias dtype = DType.float32
# alias nelts: Int = simdwidthof[dtype]()


# fn test_get_result_shape() raises:
#     # padding=2, stride=1, dilation=1
#     # input shape: (4, 28, 28)  kernel shape: (1, 16)
#     # result:  (32, 17)
#     var inputs = Tensor[dtype](4, 28, 28)
#     var kernel = Tensor[dtype](1, 16)

#     var res = get_result_shape[2, 1, 1](inputs.shape(), kernel.shape())
#     assert_equal(res[0], 32)
#     assert_equal(res[1], 17)

#     # padding=0, stride=1, dilation=1
#     # input shape: (4, 32, 17)  kernel shape: (2, 2)
#     # result:  (31, 16)
#     inputs = Tensor[dtype](4, 32, 17)
#     kernel = Tensor[dtype](2, 2)

#     res = get_result_shape[0, 1, 1](inputs.shape(), kernel.shape())
#     assert_equal(res[0], 31)
#     assert_equal(res[1], 16)

#     # padding=(3, 1), stride=1, dilation=2
#     # input shape: (4, 32, 17)  kernel shape: (2, 2)
#     # result:  (36, 17)
#     inputs = Tensor[dtype](4, 32, 17)
#     kernel = Tensor[dtype](2, 2)

#     res = get_result_shape[StaticIntTuple[2](3, 1), 1, 2](
#         inputs.shape(), kernel.shape()
#     )
#     assert_equal(res[0], 36)
#     assert_equal(res[1], 17)

#     # padding=(3, 2), stride=(2, 1), dilation=(2, 3)
#     # input shape: (4, 32, 17)  kernel shape: (2, 2)
#     # result:  (18, 18)
#     inputs = Tensor[dtype](4, 32, 17)
#     kernel = Tensor[dtype](3, 2)

#     res = get_result_shape[
#         StaticIntTuple[2](3, 2), StaticIntTuple[2](2, 1), StaticIntTuple[2](2, 3)
#     ](inputs.shape(), kernel.shape())
#     assert_equal(res[0], 17)
#     assert_equal(res[1], 18)


# def to_numpy(tensor: Tensor) -> PythonObject:
#     var np = Python.import_module("numpy")
#     np.set_printoptions(4)

#     rank = tensor.rank()
#     var pyarray: PythonObject = np.array([0])
#     if rank == 1:
#         pyarray = np.empty((tensor.dim(0)))
#     elif rank == 2:
#         pyarray = np.empty((tensor.dim(0), tensor.dim(1)))
#     elif rank == 3:
#         pyarray = np.empty((tensor.dim(0), tensor.dim(1), tensor.dim(2)))
#     elif rank == 4:
#         pyarray = np.empty((tensor.dim(0), tensor.dim(1), tensor.dim(2), tensor.dim(3)))
#     else:
#         print("Error: rank not supported: ", rank)

#     for i in range(tensor.num_elements()):
#         pyarray.itemset((i), tensor[i])

#     return pyarray


# fn to_tensor(np_array: PythonObject) raises -> Tensor[dtype]:
#     var shape = DynamicVector[Int]()
#     for i in range(np_array.ndim):
#         shape.push_back(np_array.shape[i].to_float64().to_int())

#     var tensor = Tensor[dtype](TensorShape(shape))

#     for i in range(tensor.num_elements()):
#         tensor[i] = np_array.ravel()[i].to_float64().cast[dtype]()

#     return tensor


# @value
# struct torch_conv2d_output:
#     var expected: Tensor[dtype]
#     var expected_inputs_grad: Tensor[dtype]
#     var expected_kernel_grad: Tensor[dtype]
#     var expected_bias_grad: Tensor[dtype]


# fn torch_conv2d(
#     inputs: Tensor,
#     kernel: Tensor,
#     bias: Tensor,
#     padding: StaticIntTuple[2],
#     stride: StaticIntTuple[2],
#     dilation: StaticIntTuple[2],
#     upper_grad: Tensor,
# ) -> torch_conv2d_output:
#     var out: torch_conv2d_output

#     try:
#         var torch = Python.import_module("torch")
#         var F = Python.import_module("torch.nn.functional")
#         var np = Python.import_module("numpy")

#         var inputs = torch.from_numpy(to_numpy(inputs)).requires_grad_(True)
#         var weights = torch.from_numpy(to_numpy(kernel)).requires_grad_(True)
#         var bias = torch.from_numpy(to_numpy(bias)).requires_grad_(True)

#         var expected = F.conv2d(
#             inputs,
#             weights,
#             bias,
#             (stride[0], stride[1]),
#             (padding[0], padding[1]),
#             (dilation[0], dilation[1]),
#         )

#         # uppergrad & backwards
#         var upper_grad = torch.from_numpy(to_numpy(upper_grad))
#         _ = expected.backward(upper_grad)

#         # expected output
#         out = torch_conv2d_output(
#             to_tensor(expected.detach().numpy()),
#             to_tensor(inputs.grad.numpy()),
#             to_tensor(weights.grad.numpy()),
#             to_tensor(bias.grad.numpy()),
#         )
#         return out

#     except:
#         print("Error importing torch")
#         var d = Tensor[dtype](1)
#         var out: torch_conv2d_output = torch_conv2d_output(d, d, d, d)
#         return out


# fn test_forward_1() raises:
#     # padding=2, stride=1, dilation=1
#     # input shape: (4, 1, 28, 28)  kernel shape: (1, 1, 1, 16)
#     # result_shape:  (4, 1, 32, 17)
#     alias padding = 2
#     alias stride = 1
#     alias dilation = 1
#     var inputs = Tensor[dtype](4, 1, 28, 28)
#     var kernel = Tensor[dtype](1, 1, 1, 16)
#     fill[dtype, nelts](inputs, 1.0)
#     fill[dtype, nelts](kernel, 1.0)
#     var bias = Tensor[dtype](1)

#     var res = CONV2D.forward[padding, stride, dilation](inputs, kernel, bias)
#     var torch_out = torch_conv2d(
#         inputs,
#         kernel,
#         bias=bias,
#         padding=padding,
#         stride=stride,
#         dilation=dilation,
#         upper_grad=res.grad,
#     )
#     assert_tensors_equal(res.tensor, torch_out.expected)
#     GRAPH.reset_all()


# fn test_forward_2() raises:
#     # padding=0, stride=1, dilation=1
#     # input shape: (4, 1, 32, 17)  kernel shape: (1, 1, 2, 2)
#     # result_shape:  (4, 1, 31, 16)
#     alias padding = 0
#     alias stride = 1
#     alias dilation = 1
#     var inputs = Tensor[dtype](4, 1, 32, 17)
#     var kernel = Tensor[dtype](1, 1, 2, 2)
#     fill[dtype, nelts](inputs, 1.0)
#     fill[dtype, nelts](kernel, 1.0)
#     var bias = Tensor[dtype](1)
#     fill[dtype, nelts](bias, 66.99)

#     var res = CONV2D.forward[padding, stride, dilation](inputs, kernel, bias)
#     var torch_out = torch_conv2d(
#         inputs,
#         kernel,
#         bias=bias,
#         padding=padding,
#         stride=stride,
#         dilation=dilation,
#         upper_grad=res.grad,
#     )
#     assert_tensors_equal(res.tensor, torch_out.expected)
#     GRAPH.reset_all()


# fn test_forward_3() raises:
#     # padding=(3, 1), stride=(2, 3), dilation=(2, 3)
#     # input shape: (4, 3, 32, 17)  kernel shape: (2, 3, 2, 2)
#     # result_shape:  (4, 2, 18, 6)
#     alias padding = StaticIntTuple[2](3, 1)
#     alias stride = StaticIntTuple[2](2, 3)
#     alias dilation = StaticIntTuple[2](2, 3)
#     var inputs = Tensor[dtype](4, 3, 32, 17)
#     var kernel = Tensor[dtype](2, 3, 2, 2)
#     fill[dtype, nelts](inputs, 3.0)
#     fill[dtype, nelts](kernel, 2.0)
#     var bias = Tensor[dtype](2)
#     fill[dtype, nelts](bias, 3)

#     var res = CONV2D.forward[padding, stride, dilation](inputs, kernel, bias)
#     var torch_out = torch_conv2d(
#         inputs,
#         kernel,
#         bias=bias,
#         padding=padding,
#         stride=stride,
#         dilation=dilation,
#         upper_grad=res.grad,
#     )
#     assert_tensors_equal(res.tensor, torch_out.expected)
#     GRAPH.reset_all()


# fn test_backward_1() raises:
#     # padding=2, stride=1, dilation=1
#     alias padding = 2
#     alias stride = 1
#     alias dilation = 1
#     alias batch = 4
#     alias in_channels = 2
#     alias out_channels = 3
#     var inputs = Tensor[dtype](batch, in_channels, 28, 28)
#     var kernel = Tensor[dtype](out_channels, in_channels, 1, 16)
#     fill[dtype, nelts](inputs, 1.0)
#     fill[dtype, nelts](kernel, 1.0)
#     var bias: Tensor[dtype] = rand[dtype](out_channels)

#     var res = CONV2D.forward[padding, stride, dilation](inputs, kernel, bias)

#     var gn = GRAPH.graph[GRAPH.get_node_idx(res.uuid)]
#     assert_equal(gn.parents.size, 3)
#     var upper_grad: Tensor[dtype] = rand[dtype](res.tensor.shape())

#     var ug1 = gn.backward_fn(upper_grad, gn.parents, 0)  # inputs.grad
#     var ug2 = gn.backward_fn(upper_grad, gn.parents, 1)  # kernel.grad
#     var ug3 = gn.backward_fn(upper_grad, gn.parents, 2)  # bias.grad

#     var torch_out = torch_conv2d(
#         inputs,
#         kernel,
#         bias=bias,
#         padding=padding,
#         stride=stride,
#         dilation=dilation,
#         upper_grad=upper_grad,
#     )
#     assert_tensors_equal(res.tensor, torch_out.expected)
#     assert_tensors_equal(ug1, torch_out.expected_inputs_grad, "almost")
#     assert_tensors_equal(ug2, torch_out.expected_kernel_grad, "almost")
#     assert_tensors_equal(ug3, torch_out.expected_bias_grad, "almost")
#     GRAPH.reset_all()


# fn test_backward_2() raises:
#     # padding=(2, 4), stride=(3, 1), dilation=2
#     alias padding = StaticIntTuple[2](2, 4)
#     alias stride = StaticIntTuple[2](3, 1)
#     alias dilation = 2
#     alias batch = 4
#     alias in_channels = 2
#     alias out_channels = 3
#     var inputs = Tensor[dtype](batch, in_channels, 28, 28)
#     var kernel = Tensor[dtype](out_channels, in_channels, 4, 8)
#     fill[dtype, nelts](inputs, 3.0)
#     fill[dtype, nelts](kernel, 1.0)
#     var bias: Tensor[dtype] = rand[dtype](out_channels)

#     var res = CONV2D.forward[padding, stride, dilation](inputs, kernel, bias)

#     var gn = GRAPH.graph[GRAPH.get_node_idx(res.uuid)]
#     assert_equal(gn.parents.size, 3)
#     var upper_grad: Tensor[dtype] = rand[dtype](res.tensor.shape())

#     var ug1 = gn.backward_fn(upper_grad, gn.parents, 0)  # inputs.grad
#     var ug2 = gn.backward_fn(upper_grad, gn.parents, 1)  # kernel.grad
#     var ug3 = gn.backward_fn(upper_grad, gn.parents, 2)  # bias.grad

#     var torch_out = torch_conv2d(
#         inputs,
#         kernel,
#         bias=bias,
#         padding=padding,
#         stride=stride,
#         dilation=dilation,
#         upper_grad=upper_grad,
#     )
#     assert_tensors_equal(res.tensor, torch_out.expected)
#     assert_tensors_equal(ug1, torch_out.expected_inputs_grad, "almost")
#     assert_tensors_equal(ug2, torch_out.expected_kernel_grad, "almost")
#     assert_tensors_equal(ug3, torch_out.expected_bias_grad, "almost")
#     GRAPH.reset_all()


# fn test_backward_3() raises:
#     # padding=(2, 4), stride=2, dilation=(3, 2)
#     alias padding = StaticIntTuple[2](3, 2)
#     alias stride = 2
#     alias dilation = StaticIntTuple[2](3, 2)
#     alias batch = 4
#     alias in_channels = 2
#     alias out_channels = 3
#     var inputs = Tensor[dtype](batch, in_channels, 28, 28)
#     var kernel = Tensor[dtype](out_channels, in_channels, 5, 6)
#     fill[dtype, nelts](inputs, 3.0)
#     fill[dtype, nelts](kernel, 4.0)
#     var bias: Tensor[dtype] = rand[dtype](out_channels)

#     var res = CONV2D.forward[padding, stride, dilation](inputs, kernel, bias)

#     var gn = GRAPH.graph[GRAPH.get_node_idx(res.uuid)]
#     assert_equal(gn.parents.size, 3)
#     var upper_grad: Tensor[dtype] = rand[dtype](res.tensor.shape())

#     var ug1 = gn.backward_fn(upper_grad, gn.parents, 0)  # inputs.grad
#     var ug2 = gn.backward_fn(upper_grad, gn.parents, 1)  # kernel.grad
#     var ug3 = gn.backward_fn(upper_grad, gn.parents, 2)  # bias.grad

#     var torch_out = torch_conv2d(
#         inputs,
#         kernel,
#         bias=bias,
#         padding=padding,
#         stride=stride,
#         dilation=dilation,
#         upper_grad=upper_grad,
#     )
#     assert_tensors_equal(res.tensor, torch_out.expected)
#     assert_tensors_equal(ug1, torch_out.expected_inputs_grad, "almost")
#     assert_tensors_equal(ug2, torch_out.expected_kernel_grad, "almost")
#     assert_tensors_equal(ug3, torch_out.expected_bias_grad, "almost")
#     GRAPH.reset_all()


# fn main():
#     try:
#         test_get_result_shape()
#         test_forward_1()
#         test_forward_2()
#         test_forward_3()
#         test_backward_1()
#         test_backward_2()
#         test_backward_3()
#     except e:
#         print("[Error] Error in Conv2D")
#         print(e)
