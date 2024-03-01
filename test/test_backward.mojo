# from random import rand
from tensor import Tensor, TensorShape
# from math import equal, log, exp
from testing import assert_true, assert_equal
from test_tensorutils import assert_tensors_equal


# from dainemo.autograd.node import Node
from dainemo.utils.tensorutils import fill
from dainemo.autograd.ops.basics import ADD, SUB

alias dtype = DType.float32
alias nelts: Int = simdwidthof[dtype]()


# <------------ADD------------>
fn test_ADD() raises:
    alias t1_shape = TensorShape(2, 3)
    alias t2_shape = TensorShape(2, 3)
    alias ug_shape = TensorShape(2, 3)
    var t1 = Tensor[dtype](t1_shape)
    var t2 = Tensor[dtype](t2_shape)

    fill[dtype, nelts](t1, 1.0)
    fill[dtype, nelts](t2, 2.0)

    var ug = Tensor[dtype](ug_shape)
    fill[dtype, nelts](ug, 1.0)

    var expected_grad = Tensor[dtype](t1_shape)
    fill[dtype, nelts](expected_grad, 1.0)
   

    var grad = ADD.backward[0, ug_shape, t1_shape, t2_shape](ug, t1, t2)
    assert_tensors_equal(grad, expected_grad)
    grad = ADD.backward[1, ug_shape, t1_shape, t2_shape](ug, t1, t2)
    assert_tensors_equal(grad, expected_grad)

# <------------SUB------------>
fn test_SUB() raises:
    alias t1_shape = TensorShape(2, 3)
    alias t2_shape = TensorShape(2, 3)
    alias ug_shape = TensorShape(2, 3)
    var t1 = Tensor[dtype](t1_shape)
    var t2 = Tensor[dtype](t2_shape)

    fill[dtype, nelts](t1, 2.0)
    fill[dtype, nelts](t2, 1.0)

    var ug = Tensor[dtype](ug_shape)
    fill[dtype, nelts](ug, 1.0)

    var expected_grad = Tensor[dtype](t1_shape)
    fill[dtype, nelts](expected_grad, 1.0)

    var grad = SUB.backward[0, ug_shape, t1_shape, t2_shape](ug, t1, t2)
    assert_tensors_equal(grad, expected_grad)

    fill[dtype, nelts](expected_grad, -1.0)
    grad = SUB.backward[1, ug_shape, t1_shape, t2_shape](ug, t1, t2)
    assert_tensors_equal(grad, expected_grad)


# # <------------MUL------------>
# fn test_MUL() raises:
#     var t1: Tensor[dtype] = Tensor[dtype](2, 3)
#     var t2: Tensor[dtype] = Tensor[dtype](2, 3)
#     var upper_grad: Tensor[dtype] = Tensor[dtype](2, 3)
#     fill[dtype, nelts](t1, 1.0)
#     fill[dtype, nelts](t2, 2.0)
#     fill[dtype, nelts](upper_grad, 1.0)

#     var res = MUL.forward(t1, t2)

#     var gn = GRAPH.graph[GRAPH.get_node_idx(res.uuid)]
#     assert_equal(gn.parents.size, 2)

#     var ug1 = gn.backward_fn(upper_grad, gn.parents, 0)
#     var ug2 = gn.backward_fn(upper_grad, gn.parents, 1)

#     var expected_ug1 = Tensor[dtype](2, 3)
#     fill[dtype, nelts](expected_ug1, 2.0)
#     var expected_ug2 = Tensor[dtype](2, 3)
#     fill[dtype, nelts](expected_ug2, 1.0)

#     assert_tensors_equal(ug1, expected_ug1)
#     assert_tensors_equal(ug2, expected_ug2)
#     GRAPH.reset_all()


# # <------------DIV------------>
# fn test_DIV() raises:
#     var t1: Tensor[dtype] = Tensor[dtype](2, 3)
#     var t2: Tensor[dtype] = Tensor[dtype](2, 3)
#     var upper_grad: Tensor[dtype] = Tensor[dtype](2, 3)
#     fill[dtype, nelts](t1, 1.0)
#     fill[dtype, nelts](t2, 2.0)
#     fill[dtype, nelts](upper_grad, 1.0)

#     var res = DIV.forward(t1, t2)

#     var gn = GRAPH.graph[GRAPH.get_node_idx(res.uuid)]
#     assert_equal(gn.parents.size, 2)

#     var ug1 = gn.backward_fn(upper_grad, gn.parents, 0)
#     var ug2 = gn.backward_fn(upper_grad, gn.parents, 1)

#     var expected_ug1 = Tensor[dtype](2, 3)
#     fill[dtype, nelts](expected_ug1, 1.0 / 2.0)
#     var expected_ug2 = Tensor[dtype](2, 3)
#     fill[dtype, nelts](expected_ug2, -1.0 / (2.0**2))

#     assert_tensors_equal(ug1, expected_ug1)
#     assert_tensors_equal(ug2, expected_ug2)
#     GRAPH.reset_all()


# # <------------DOT------------>
# fn test_DOT() raises:
#     var t1: Tensor[dtype] = Tensor[dtype](2, 3)
#     var t2: Tensor[dtype] = Tensor[dtype](2, 3)
#     var upper_grad: Tensor[dtype] = Tensor[dtype](2, 3)
#     fill[dtype, nelts](t1, 1.0)
#     fill[dtype, nelts](t2, 2.0)
#     fill[dtype, nelts](upper_grad, 1.0)

#     var res = DOT.forward(t1, t2)

#     var gn = GRAPH.graph[GRAPH.get_node_idx(res.uuid)]
#     assert_equal(gn.parents.size, 2)

#     var ug1 = gn.backward_fn(upper_grad, gn.parents, 0)
#     var ug2 = gn.backward_fn(upper_grad, gn.parents, 1)

#     var expected_ug1 = Tensor[dtype](2, 2)
#     fill[dtype, nelts](expected_ug1, 6.0)
#     var expected_ug2 = Tensor[dtype](3, 3)
#     fill[dtype, nelts](expected_ug2, 2.0)

#     assert_tensors_equal(ug1, expected_ug1)
#     assert_tensors_equal(ug2, expected_ug2)
#     GRAPH.reset_all()


# # <------------EXP------------>
# fn test_EXP() raises:
#     var t1: Tensor[dtype] = Tensor[dtype](2, 3)
#     var upper_grad: Tensor[dtype] = Tensor[dtype](2, 3)
#     fill[dtype, nelts](t1, 2.0)
#     fill[dtype, nelts](upper_grad, 5.0)

#     var res = EXP.forward(t1)

#     var gn = GRAPH.graph[GRAPH.get_node_idx(res.uuid)]
#     assert_equal(gn.parents.size, 1)

#     var ug1 = gn.backward_fn(upper_grad, gn.parents, 0)

#     var expected_ug1 = Tensor[dtype](2, 3)
#     fill[dtype, nelts](expected_ug1, 5.0 * exp[dtype, 1](2.0))
#     assert_tensors_equal(ug1, expected_ug1)
#     GRAPH.reset_all()


# # <------------LOG------------>
# fn test_LOG() raises:
#     var t1: Tensor[dtype] = Tensor[dtype](2, 3)
#     var upper_grad: Tensor[dtype] = Tensor[dtype](2, 3)
#     fill[dtype, nelts](t1, 2.0)
#     fill[dtype, nelts](upper_grad, 5.0)

#     var res = LOG.forward(t1)

#     var gn = GRAPH.graph[GRAPH.get_node_idx(res.uuid)]
#     assert_equal(gn.parents.size, 1)

#     var ug1 = gn.backward_fn(upper_grad, gn.parents, 0)

#     var expected_ug1 = Tensor[dtype](2, 3)
#     fill[dtype, nelts](expected_ug1, 5.0 / 2.0)
#     assert_tensors_equal(ug1, expected_ug1)
#     GRAPH.reset_all()


# # <------------POW------------>
# fn test_POW() raises:
#     var t2: Tensor[dtype] = Tensor[dtype](2, 3)
#     var upper_grad: Tensor[dtype] = Tensor[dtype](2, 3)
#     fill[dtype, nelts](t2, 2.0)
#     fill[dtype, nelts](upper_grad, 1.0)

#     var res = POW.forward(t2, 2)

#     var gn = GRAPH.graph[GRAPH.get_node_idx(res.uuid)]
#     assert_equal(gn.parents.size, 2)

#     var ug1 = gn.backward_fn(upper_grad, gn.parents, 0)
#     var ug2 = gn.backward_fn(upper_grad, gn.parents, 1)

#     var expected_ug1 = Tensor[dtype](2, 3)
#     fill[dtype, nelts](expected_ug1, 4.0)
#     var expected_ug2 = Tensor[dtype](2, 3)
#     fill[dtype, nelts](expected_ug2, (2**2) * log[dtype, 1](2))

#     assert_tensors_equal(ug1, expected_ug1)
#     assert_tensors_equal(ug2, expected_ug2)
#     GRAPH.reset_all()


# # <------------SUM------------>
# fn test_SUM() raises:
#     # SUM ALL ELEMENTS
#     var t1: Tensor[dtype] = Tensor[dtype](2, 3)
#     fill[dtype, nelts](t1, 1.0)

#     var res = SUM.forward(t1)

#     # uppergrad has always to same shape as res
#     var upper_grad: Tensor[dtype] = Tensor[dtype](res.tensor.shape())
#     fill[dtype, nelts](upper_grad, 9.0)
#     var gn = GRAPH.graph[GRAPH.get_node_idx(res.uuid)]
#     assert_equal(gn.parents.size, 1)  # one parent

#     var ug1 = gn.backward_fn(upper_grad, gn.parents, 0)

#     var expected_ug1 = Tensor[dtype](2, 3)
#     fill[dtype, nelts](expected_ug1, 9.0)
#     assert_tensors_equal(ug1, expected_ug1)
#     GRAPH.reset_all()


# fn test_SUM_0() raises:
#     # SUM ALONG AXIS 0
#     var t1: Tensor[dtype] = Tensor[dtype](2, 3)
#     fill[dtype, nelts](t1, 1.0)

#     var res = SUM.forward[axis=0](t1)

#     # uppergrad has always to same shape as res
#     var upper_grad: Tensor[dtype] = Tensor[dtype](res.tensor.shape())
#     upper_grad[0] = 0.0
#     upper_grad[1] = 1.0
#     upper_grad[2] = 2.0
#     var gn = GRAPH.graph[GRAPH.get_node_idx(res.uuid)]
#     assert_equal(gn.parents.size, 1)  # one parent

#     var ug1 = gn.backward_fn(upper_grad, gn.parents, 0)

#     var expected_ug1 = Tensor[dtype](2, 3)
#     for i in range(expected_ug1.num_elements()):
#         expected_ug1[i] = i % 3
#     assert_tensors_equal(ug1, expected_ug1)
#     GRAPH.reset_all()


# fn test_SUM_1() raises:
#     # SUM ALONG AXIS 1
#     var t1: Tensor[dtype] = Tensor[dtype](2, 3)
#     fill[dtype, nelts](t1, 1.0)

#     var res = SUM.forward[axis=1](t1)

#     # uppergrad has always to same shape as res
#     var upper_grad: Tensor[dtype] = Tensor[dtype](res.tensor.shape())
#     upper_grad[0] = 0.0
#     upper_grad[1] = 1.0
#     var gn = GRAPH.graph[GRAPH.get_node_idx(res.uuid)]
#     assert_equal(gn.parents.size, 1)  # one parent

#     var ug1 = gn.backward_fn(upper_grad, gn.parents, 0)

#     var expected_ug1 = Tensor[dtype](2, 3)
#     for i in range(expected_ug1.num_elements()):
#         expected_ug1[i] = 0 if i < 3 else 1
#     assert_tensors_equal(ug1, expected_ug1)
#     GRAPH.reset_all()


# # <------------MAX------------>
# fn test_MAX() raises:
#     # SUM ALL ELEMENTS
#     var t1: Tensor[dtype] = Tensor[dtype](2, 3)
#     fill[dtype, nelts](t1, 1.0)
#     t1[0] = 2.0
#     t1[1] = 2.0

#     var res = MAX.forward(t1)

#     # uppergrad has always to same shape as res
#     var upper_grad: Tensor[dtype] = Tensor[dtype](res.tensor.shape())
#     fill[dtype, nelts](upper_grad, 9.0)
#     var gn = GRAPH.graph[GRAPH.get_node_idx(res.uuid)]
#     assert_equal(gn.parents.size, 1)  # one parent

#     var ug1 = gn.backward_fn(upper_grad, gn.parents, 0)

#     var expected_ug1 = Tensor[dtype](2, 3)
#     expected_ug1[0] = 4.5
#     expected_ug1[1] = 4.5
#     assert_tensors_equal(ug1, expected_ug1)
#     GRAPH.reset_all()


# fn test_MAX_0() raises:
#     # MAX ALONG AXIS 0
#     var t1: Tensor[dtype] = Tensor[dtype](2, 3, 2)
#     for i in range(t1.num_elements()):
#         t1[i] = i + 1
#     t1[0] = 7.0

#     var res = MAX.forward[axis=0](t1)

#     # uppergrad has always to same shape as res
#     var upper_grad: Tensor[dtype] = Tensor[dtype](res.tensor.shape())
#     var gn = GRAPH.graph[GRAPH.get_node_idx(res.uuid)]
#     fill[dtype, nelts](upper_grad, 2.0)
#     assert_equal(gn.parents.size, 1)  # one parent

#     var ug1 = gn.backward_fn(upper_grad, gn.parents, 0)

#     var expected_ug1 = Tensor[dtype](2, 3, 2)
#     expected_ug1[0] = 1.0
#     expected_ug1[6] = 1.0
#     expected_ug1[7] = 2.0
#     expected_ug1[8] = 2.0
#     expected_ug1[9] = 2.0
#     expected_ug1[10] = 2.0
#     expected_ug1[11] = 2.0
#     assert_tensors_equal(ug1, expected_ug1)
#     GRAPH.reset_all()


# fn test_MAX_1() raises:
#     # MAX ALONG AXIS 1
#     var t1: Tensor[dtype] = Tensor[dtype](2, 3, 2)
#     for i in range(t1.num_elements()):
#         t1[i] = i + 1
#     t1[0] = 5.0

#     var res = MAX.forward[axis=1](t1)

#     # uppergrad has always to same shape as res
#     var upper_grad: Tensor[dtype] = Tensor[dtype](res.tensor.shape())
#     var gn = GRAPH.graph[GRAPH.get_node_idx(res.uuid)]
#     fill[dtype, nelts](upper_grad, 2.0)
#     assert_equal(gn.parents.size, 1)  # one parent

#     var ug1 = gn.backward_fn(upper_grad, gn.parents, 0)

#     var expected_ug1 = Tensor[dtype](2, 3, 2)
#     expected_ug1[0] = 1.0
#     expected_ug1[4] = 1.0
#     expected_ug1[5] = 2.0
#     expected_ug1[10] = 2.0
#     expected_ug1[11] = 2.0
#     assert_tensors_equal(ug1, expected_ug1)
#     GRAPH.reset_all()


# fn test_MAX_2() raises:
#     # MAX ALONG AXIS 2
#     var t1: Tensor[dtype] = Tensor[dtype](2, 3, 2)
#     for i in range(t1.num_elements()):
#         t1[i] = i + 1
#     t1[0] = 2.0

#     var res = MAX.forward[axis=2](t1)

#     # uppergrad has always to same shape as res
#     var upper_grad: Tensor[dtype] = Tensor[dtype](res.tensor.shape())
#     var gn = GRAPH.graph[GRAPH.get_node_idx(res.uuid)]
#     fill[dtype, nelts](upper_grad, 2.0)
#     assert_equal(gn.parents.size, 1)  # one parent

#     var ug1 = gn.backward_fn(upper_grad, gn.parents, 0)

#     var expected_ug1 = Tensor[dtype](2, 3, 2)
#     expected_ug1[0] = 1.0
#     expected_ug1[1] = 1.0
#     expected_ug1[3] = 2.0
#     expected_ug1[5] = 2.0
#     expected_ug1[7] = 2.0
#     expected_ug1[9] = 2.0
#     expected_ug1[11] = 2.0
#     assert_tensors_equal(ug1, expected_ug1)
#     GRAPH.reset_all()


# # <------------TRANSPOSE------------>
# fn test_TRANSPOSE() raises:
#     var t1 = Tensor[dtype](2, 3)

#     var res = TRANSPOSE.forward(t1)

#     # uppergrad has always to same shape as res
#     var upper_grad: Tensor[dtype] = Tensor[dtype](res.tensor.shape())
#     assert_equal(upper_grad.dim(0), 3)
#     assert_equal(upper_grad.dim(1), 2)
#     for i in range(3):
#         upper_grad[2*i] = i+1
#         upper_grad[2*i+1] = i+4
#     var gn = GRAPH.graph[GRAPH.get_node_idx(res.uuid)]
#     assert_equal(gn.parents.size, 1)  # one parent

#     var ug1 = gn.backward_fn(upper_grad, gn.parents, 0)

#     var expected_ug1 = Tensor[dtype](t1.shape())
#     for i in range(6):
#         expected_ug1[i] = i+1
#     assert_tensors_equal(ug1, expected_ug1)
#     GRAPH.reset_all()
    

# # <------------FLATTEN------------>
# fn test_FLATTEN() raises:
#     var t1 = Tensor[dtype](2, 3)

#     var res = FLATTEN.forward(t1)

#     # uppergrad has always to same shape as res
#     var upper_grad: Tensor[dtype] = Tensor[dtype](res.tensor.shape())
#     fill[dtype, nelts](upper_grad, 1.0)
#     assert_equal(upper_grad.dim(0), 6)
#     var gn = GRAPH.graph[GRAPH.get_node_idx(res.uuid)]
#     assert_equal(gn.parents.size, 1)  # one parent

#     var ug1 = gn.backward_fn(upper_grad, gn.parents, 0)

#     var expected_ug1 = Tensor[dtype](t1.shape())
#     fill[dtype, nelts](expected_ug1, 1.0)
#     assert_tensors_equal(ug1, expected_ug1)
#     GRAPH.reset_all()


# # <------------RESHAPE------------>
# fn test_RESHAPE() raises:
#     var t1 = Tensor[dtype](2, 2, 5)
#     var new_shape = TensorShape(2, 10)

#     var res = RESHAPE.forward(t1, new_shape)

#     # uppergrad has always to same shape as res
#     var upper_grad: Tensor[dtype] = Tensor[dtype](res.tensor.shape())
#     fill[dtype, nelts](upper_grad, 1.0)
#     assert_equal(upper_grad.dim(0), 2)
#     assert_equal(upper_grad.dim(1), 10)
#     var gn = GRAPH.graph[GRAPH.get_node_idx(res.uuid)]
#     assert_equal(gn.parents.size, 1)  # one parent

#     var ug1 = gn.backward_fn(upper_grad, gn.parents, 0)

#     var expected_ug1 = Tensor[dtype](t1.shape())
#     fill[dtype, nelts](expected_ug1, 1.0)
#     assert_tensors_equal(ug1, expected_ug1)
#     GRAPH.reset_all()


fn main():
    try:
        test_ADD()
        test_SUB()
#         test_MUL()
#         test_DIV()
#         test_DOT()
#         test_EXP()
#         test_LOG()
#         test_POW()
#         test_SUM()
#         test_SUM_0()
#         test_SUM_1()
#         test_MAX()
#         test_MAX_0()
#         test_MAX_1()
#         test_MAX_2()
#         test_TRANSPOSE()
#         test_FLATTEN()
#         test_RESHAPE()
    except e:
        print(e)
        print("[ERROR] Error in backward pass.")
