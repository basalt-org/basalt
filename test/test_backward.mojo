# from random import rand
from tensor import Tensor, TensorShape
from math import log, exp
from testing import assert_true, assert_equal
from test_tensorutils import assert_tensors_equal


# from dainemo.autograd.node import Node
from dainemo.utils.tensorutils import fill, tsum
from dainemo.autograd.ops.basics import ADD, SUB, MUL, DIV, DOT, EXP, LOG, POW, MEAN, FLATTEN, SUM
from dainemo.autograd.node import Attribute, AttributeVector

alias dtype = DType.float32
alias nelts: Int = simdwidthof[dtype]()


# <------------ADD------------>
fn test_ADD() raises:
    alias t1_shape = TensorShape(2, 3)
    alias t2_shape = TensorShape(2, 3)
    alias ug_shape = TensorShape(2, 3)
    var t1 = Tensor[dtype](t1_shape)
    var t2 = Tensor[dtype](t2_shape)
    var ug = Tensor[dtype](ug_shape)
    fill[dtype, nelts](t1, 1.0)
    fill[dtype, nelts](t2, 2.0)
    fill[dtype, nelts](ug, 1.0)

    var expected_grad = Tensor[dtype](ug_shape)
    fill[dtype, nelts](expected_grad, 1.0)

    var grad1 = ADD.backward[0, ug_shape, t1_shape, t2_shape](ug, t1, t2)
    assert_tensors_equal(grad1, expected_grad)
    var grad2 = ADD.backward[1, ug_shape, t1_shape, t2_shape](ug, t1, t2)
    assert_tensors_equal(grad2, expected_grad)


# <------------SUB------------>
fn test_SUB() raises:
    alias t1_shape = TensorShape(2, 3)
    alias t2_shape = TensorShape(2, 3)
    alias ug_shape = TensorShape(2, 3)
    var t1 = Tensor[dtype](t1_shape)
    var t2 = Tensor[dtype](t2_shape)
    var ug = Tensor[dtype](ug_shape)
    fill[dtype, nelts](t1, 2.0)
    fill[dtype, nelts](t2, 1.0)
    fill[dtype, nelts](ug, 1.0)

    var grad1 = SUB.backward[0, ug_shape, t1_shape, t2_shape](ug, t1, t2)
    var expected_grad1 = Tensor[dtype](t1_shape)
    fill[dtype, nelts](expected_grad1, 1.0)
    assert_tensors_equal(grad1, expected_grad1)

    var grad2 = SUB.backward[1, ug_shape, t1_shape, t2_shape](ug, t1, t2)
    var expected_grad2 = Tensor[dtype](t2_shape)
    fill[dtype, nelts](expected_grad2, -1.0)
    assert_tensors_equal(grad2, expected_grad2)


# <------------MUL------------>
fn test_MUL() raises:
    alias t1_shape = TensorShape(2, 3)
    alias t2_shape = TensorShape(2, 3)
    alias ug_shape = TensorShape(2, 3)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    var t2: Tensor[dtype] = Tensor[dtype](t2_shape)
    var ug: Tensor[dtype] = Tensor[dtype](ug_shape)
    fill[dtype, nelts](t1, 1.0)
    fill[dtype, nelts](t2, 2.0)
    fill[dtype, nelts](ug, 1.0)

    var grad1 = MUL.backward[0, ug_shape, t1_shape, t2_shape](ug, t1, t2)
    var expected_grad1 = Tensor[dtype](t1_shape)
    fill[dtype, nelts](expected_grad1, 2.0)
    assert_tensors_equal(grad1, expected_grad1)

    var grad2 = MUL.backward[1, ug_shape, t1_shape, t2_shape](ug, t1, t2)
    var expected_grad2 = Tensor[dtype](t2_shape)
    fill[dtype, nelts](expected_grad2, 1.0)
    assert_tensors_equal(grad2, expected_grad2)


# <------------DIV------------>
fn test_DIV() raises:
    alias t1_shape = TensorShape(2, 3)
    alias t2_shape = TensorShape(2, 3)
    alias ug_shape = TensorShape(2, 3)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    var t2: Tensor[dtype] = Tensor[dtype](t2_shape)
    var ug: Tensor[dtype] = Tensor[dtype](ug_shape)
    fill[dtype, nelts](t1, 1.0)
    fill[dtype, nelts](t2, 2.0)
    fill[dtype, nelts](ug, 1.0)

    var grad1 = DIV.backward[0, ug_shape, t1_shape, t2_shape](ug, t1, t2)
    var expected_grad1 = Tensor[dtype](t1_shape)
    fill[dtype, nelts](expected_grad1, 1.0 / 2.0)
    assert_tensors_equal(grad1, expected_grad1)

    var grad2 = DIV.backward[1, ug_shape, t1_shape, t2_shape](ug, t1, t2)
    var expected_grad2 = Tensor[dtype](t2_shape)
    fill[dtype, nelts](expected_grad2, -1.0 / (2.0**2))
    assert_tensors_equal(grad2, expected_grad2)


# <------------DOT------------>
fn test_DOT() raises:
    alias t1_shape = TensorShape(2, 3)
    alias t2_shape = TensorShape(3, 2)
    alias ug_shape = TensorShape(2, 2)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    var t2: Tensor[dtype] = Tensor[dtype](t2_shape)
    var ug: Tensor[dtype] = Tensor[dtype](ug_shape)
    fill[dtype, nelts](t1, 1.0)
    fill[dtype, nelts](t2, 2.0)
    fill[dtype, nelts](ug, 1.0)

    var grad1 = DOT.backward[0, ug_shape, t1_shape, t2_shape](ug, t1, t2)
    var expected_grad1 = Tensor[dtype](t1_shape)
    fill[dtype, nelts](expected_grad1, 4.0)
    assert_tensors_equal(grad1, expected_grad1)

    var grad2 = DOT.backward[1, ug_shape, t1_shape, t2_shape](ug, t1, t2)
    var expected_grad2 = Tensor[dtype](t2_shape)
    fill[dtype, nelts](expected_grad2, 2.0)
    assert_tensors_equal(grad2, expected_grad2)


# <------------EXP------------>
fn test_EXP() raises:
    alias t1_shape = TensorShape(2, 3)
    alias ug_shape = TensorShape(2, 3)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    var ug: Tensor[dtype] = Tensor[dtype](ug_shape)
    fill[dtype, nelts](t1, 2.0)
    fill[dtype, nelts](ug, 5.0)

    var grad1 = EXP.backward[ug_shape, t1_shape](ug, t1)
    var expected_grad1 = Tensor[dtype](t1_shape)
    fill[dtype, nelts](expected_grad1, 5.0 * exp[dtype, 1](2.0))
    assert_tensors_equal(grad1, expected_grad1)


# <------------LOG------------>
fn test_LOG() raises:
    alias t1_shape = TensorShape(2, 3)
    alias ug_shape = TensorShape(2, 3)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    var ug: Tensor[dtype] = Tensor[dtype](ug_shape)
    fill[dtype, nelts](t1, 2.0)
    fill[dtype, nelts](ug, 5.0)

    var grad1 = LOG.backward[ug_shape, t1_shape](ug, t1)
    var expected_grad1 = Tensor[dtype](t1_shape)
    fill[dtype, nelts](expected_grad1, 5.0 / 2.0)
    assert_tensors_equal(grad1, expected_grad1)


# <------------POW------------>
fn test_POW() raises:
    alias t1_shape = TensorShape(2, 3)
    alias t2_shape = TensorShape(1)
    alias ug_shape = TensorShape(2, 3)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    var t2: Tensor[dtype] = Tensor[dtype](t2_shape)
    var ug: Tensor[dtype] = Tensor[dtype](ug_shape)
    fill[dtype, nelts](t1, 2.0)
    t2[0] = 2
    fill[dtype, nelts](ug, 1.0)

    var grad1 = POW.backward[0, ug_shape, t1_shape, t2_shape](ug, t1, t2)
    var expected_grad1 = Tensor[dtype](t1_shape)
    fill[dtype, nelts](expected_grad1, 4.0)
    assert_tensors_equal(grad1, expected_grad1)

    var grad2 = POW.backward[1, ug_shape, t1_shape, t2_shape](ug, t1, t2)
    var temp = Tensor[dtype](2, 3)
    fill[dtype, nelts](temp, (2**2) * log[dtype, 1](2))
    assert_equal(grad2[0], tsum(temp))
    assert_equal(grad2.shape(), 1)


# <------------SUM------------>
fn test_SUM() raises:
    alias t1_shape = TensorShape(2, 3)
    alias ug_shape = TensorShape(2, 3)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    var ug: Tensor[dtype] = Tensor[dtype](ug_shape)
    fill[dtype, nelts](t1, 1.0)
    fill[dtype, nelts](ug, 9.0)

    var grad1 = SUM.backward[ug_shape, t1_shape](ug, t1)
    var expected_grad1 = Tensor[dtype](t1_shape)
    fill[dtype, nelts](expected_grad1, 9.0)
    assert_tensors_equal(grad1, expected_grad1)


fn test_SUM_0() raises:
    alias t1_shape = TensorShape(2, 3)
    alias ug_shape = TensorShape(1, 3)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    var ug: Tensor[dtype] = Tensor[dtype](ug_shape)
    fill[dtype, nelts](t1, 1.0)
    ug[0] = 0.0
    ug[1] = 1.0
    ug[2] = 2.0

    alias attributes = AttributeVector(Attribute("axis", 0))
    var grad1 = SUM.backward[ug_shape, t1_shape, attributes](ug, t1)
    var expected_grad1 = Tensor[dtype](t1_shape)
    for i in range(expected_grad1.num_elements()):
        expected_grad1[i] = i % 3

    assert_tensors_equal(grad1, expected_grad1)

fn test_SUM_1() raises:
    alias t1_shape = TensorShape(2, 3)
    alias ug_shape = TensorShape(2, 1)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    var ug: Tensor[dtype] = Tensor[dtype](ug_shape)
    fill[dtype, nelts](t1, 1.0)
    ug[0] = 0.0
    ug[1] = 1.0

    alias attributes = AttributeVector(Attribute("axis", 1))
    var grad1 = SUM.backward[ug_shape, t1_shape, attributes](ug, t1)
    var expected_grad1 = Tensor[dtype](t1_shape)
    for i in range(expected_grad1.num_elements()):
        expected_grad1[i] = 0 if i < 3 else 1

    assert_tensors_equal(grad1, expected_grad1)


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

#     var grad1 = gn.backward_fn(upper_grad, gn.parents, 0)

#     var expected_grad1 = Tensor[dtype](2, 3)
#     expected_grad1[0] = 4.5
#     expected_grad1[1] = 4.5
#     assert_tensors_equal(grad1, expected_grad1)
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

#     var grad1 = gn.backward_fn(upper_grad, gn.parents, 0)

#     var expected_grad1 = Tensor[dtype](2, 3, 2)
#     expected_grad1[0] = 1.0
#     expected_grad1[6] = 1.0
#     expected_grad1[7] = 2.0
#     expected_grad1[8] = 2.0
#     expected_grad1[9] = 2.0
#     expected_grad1[10] = 2.0
#     expected_grad1[11] = 2.0
#     assert_tensors_equal(grad1, expected_grad1)
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

#     var grad1 = gn.backward_fn(upper_grad, gn.parents, 0)

#     var expected_grad1 = Tensor[dtype](2, 3, 2)
#     expected_grad1[0] = 1.0
#     expected_grad1[4] = 1.0
#     expected_grad1[5] = 2.0
#     expected_grad1[10] = 2.0
#     expected_grad1[11] = 2.0
#     assert_tensors_equal(grad1, expected_grad1)
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

#     var grad1 = gn.backward_fn(upper_grad, gn.parents, 0)

#     var expected_grad1 = Tensor[dtype](2, 3, 2)
#     expected_grad1[0] = 1.0
#     expected_grad1[1] = 1.0
#     expected_grad1[3] = 2.0
#     expected_grad1[5] = 2.0
#     expected_grad1[7] = 2.0
#     expected_grad1[9] = 2.0
#     expected_grad1[11] = 2.0
#     assert_tensors_equal(grad1, expected_grad1)
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

#     var grad1 = gn.backward_fn(upper_grad, gn.parents, 0)

#     var expected_grad1 = Tensor[dtype](t1.shape())
#     for i in range(6):
#         expected_grad1[i] = i+1
#     assert_tensors_equal(grad1, expected_grad1)
#     GRAPH.reset_all()
    

# <------------FLATTEN------------>
fn test_FLATTEN() raises:
    alias t1_shape = TensorShape(2, 3)
    alias ug_shape = TensorShape(t1_shape.num_elements())
    var t1 = Tensor[dtype](t1_shape)
    var ug: Tensor[dtype] = Tensor[dtype](ug_shape)
    fill[dtype, nelts](ug, 1.0)
    assert_equal(ug.dim(0), 6)

    var grad1 = FLATTEN.backward[ug_shape, t1_shape](ug, t1)

    var expected_grad1 = Tensor[dtype](t1_shape)
    fill[dtype, nelts](expected_grad1, 1.0)

    assert_tensors_equal(grad1, expected_grad1)


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

#     var grad1 = gn.backward_fn(upper_grad, gn.parents, 0)

#     var expected_grad1 = Tensor[dtype](t1.shape())
#     fill[dtype, nelts](expected_grad1, 1.0)
#     assert_tensors_equal(grad1, expected_grad1)
#     GRAPH.reset_all()


fn main():
    try:
        test_ADD()
        test_SUB()
        test_MUL()
        test_DIV()
        test_DOT()
        test_EXP()
        test_LOG()
        test_POW()
        test_SUM()
        test_SUM_0()
        test_SUM_1()
#         test_MAX()
#         test_MAX_0()
#         test_MAX_1()
#         test_MAX_2()
#         test_TRANSPOSE()
        test_FLATTEN()
#         test_RESHAPE()
    except e:
        print(e)
        print("[ERROR] Error in backward pass.")
