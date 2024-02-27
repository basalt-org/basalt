# from tensor import Tensor
# from testing import assert_equal

# from dainemo import GRAPH
# import dainemo.nn as nn
# from dainemo.utils.tensorutils import fill
# from test_tensorutils import assert_tensors_equal

# alias dtype = DType.float32
# alias nelts: Int = simdwidthof[dtype]()


# # <------------SOFTMAX------------>
# fn test_SOFTMAX() raises:
#     var x = Tensor[dtype](2, 3, 2)
#     fill[dtype, nelts](x, 4)

#     let f0 = nn.Softmax[0]()
#     var res = f0(x)
#     var expected = Tensor[dtype](2, 3, 2)
#     fill[dtype, nelts](expected, 0.5)
#     assert_tensors_equal(res.tensor, expected)
#     assert_equal(GRAPH.graph.size, 6) # inputs, max_values, exp_values, sum_values, diff_max_values, result_div
#     GRAPH.reset_all()

#     let f1 = nn.Softmax[1]()
#     res = f1(x)
#     expected = Tensor[dtype](2, 3, 2)
#     fill[dtype, nelts](expected, 1.0 / 3.0)
#     assert_tensors_equal(res.tensor, expected, "almost")
#     assert_equal(GRAPH.graph.size, 6)
#     GRAPH.reset_all()

#     let f2 = nn.Softmax[2]()
#     res = f2(x)
#     expected = Tensor[dtype](2, 3, 2)
#     fill[dtype, nelts](expected, 0.5)
#     assert_tensors_equal(res.tensor, expected)
#     assert_equal(GRAPH.graph.size, 6)
#     GRAPH.reset_all()


# # <------------LOGSOFTMAX------------>
# fn test_LOGSOFTMAX() raises:
#     var x = Tensor[dtype](2, 3, 2)
#     fill[dtype, nelts](x, 4)

#     let f0 = nn.LogSoftmax[0]()
#     var res = f0(x)
#     var expected = Tensor[dtype](2, 3, 2)
#     fill[dtype, nelts](expected, -0.69314718)
#     assert_tensors_equal(res.tensor, expected)
#     assert_equal(GRAPH.graph.size, 7) # inputs, max_values, exp_values, sum_values, diff_max_values, log_values, result_sub
#     GRAPH.reset_all()

#     let f1 = nn.LogSoftmax[1]()
#     res = f1(x)
#     expected = Tensor[dtype](2, 3, 2)
#     fill[dtype, nelts](expected, -1.09861231)
#     assert_tensors_equal(res.tensor, expected, "almost")
#     assert_equal(GRAPH.graph.size, 7)
#     GRAPH.reset_all()

#     let f2 = nn.LogSoftmax[2]()
#     res = f2(x)
#     expected = Tensor[dtype](2, 3, 2)
#     fill[dtype, nelts](expected, -0.69314718)
#     assert_tensors_equal(res.tensor, expected)
#     assert_equal(GRAPH.graph.size, 7)
#     GRAPH.reset_all()


# # <------------RELU------------>
# fn test_RELU() raises:
#     var t1: Tensor[dtype] = Tensor[dtype](2, 3)
#     for i in range(3):
#         t1[i] = 3
#     for i in range(3, 6):
#         t1[i] = -3

#     let f = nn.ReLU()
#     let res = f(t1)

#     var expected = Tensor[dtype](2, 3)
#     for i in range(3):
#         expected[i] = 3
#     for i in range(3, 6):
#         expected[i] = 0
#     assert_tensors_equal(res.tensor, expected)
#     assert_equal(GRAPH.graph.size, 2)
#     GRAPH.reset_all()


# # <------------SIGMOID------------>
# fn test_SIGMOID() raises:
#     let t1: Tensor[dtype] = Tensor[dtype](2, 3)  # filled with zeroes

#     var upper_grad: Tensor[dtype] = Tensor[dtype](2, 3)
#     fill[dtype, nelts](upper_grad, 5.0)

#     let f = nn.Sigmoid()
#     let res = f(t1)

#     let gn = GRAPH.graph[GRAPH.get_node_idx(res.uuid)]
#     assert_equal(gn.parents.size, 1)

#     let ug1 = gn.backward_fn(upper_grad, gn.parents, 0)

#     var expected_ug1 = Tensor[dtype](2, 3)
#     fill[dtype, nelts](
#         expected_ug1, 5.0 * 0.25
#     )  # 0.25 = d(sigmoid(0))/dx = sigmoid(0) * (1 - sigmoid(0))
#     assert_tensors_equal(ug1, expected_ug1)
#     GRAPH.reset_all()


# # <------------TANH------------>
# fn test_TANH() raises:
#     let t1: Tensor[dtype] = Tensor[dtype](2, 3)

#     let f = nn.Tanh()
#     let res = f(t1)

#     var expected = Tensor[dtype](2, 3)
#     fill[dtype, nelts](expected, 0.0)
#     assert_tensors_equal(res.tensor, expected)
#     assert_equal(GRAPH.graph.size, 2)
#     GRAPH.reset_all()


# fn main():
#     try:
#         test_SOFTMAX()
#         test_LOGSOFTMAX()
#         test_RELU()
#         test_SIGMOID()
#         test_TANH()
#     except e:
#         print("[ERROR] Error in activations")
#         print(e)
