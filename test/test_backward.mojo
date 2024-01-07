from random import rand
from tensor import Tensor, TensorShape
from math import equal, log, exp
from testing import assert_true, assert_equal
from test_tensorutils import assert_tensors_equal

from dainemo import GRAPH
from dainemo.autograd.node import Node
from dainemo.utils.tensorutils import fill
from dainemo.autograd.ops.basics import ADD, SUB, MUL, DIV, DOT, EXP, LOG, POW, SUM, TRANSPOSE, FLATTEN, RESHAPE

alias dtype = DType.float32
alias nelts: Int = simdwidthof[dtype]()


# <------------ADD------------>
fn test_ADD() raises:
    var t1: Tensor[dtype] = Tensor[dtype](2, 3)
    var t2: Tensor[dtype] = Tensor[dtype](2, 3)
    var upper_grad: Tensor[dtype] = Tensor[dtype](2, 3)
    fill[dtype, nelts](t1, 1.0)
    fill[dtype, nelts](t2, 2.0)
    fill[dtype, nelts](upper_grad, 1.0)

    let res = ADD.forward(t1, t2)

    let gn = GRAPH.graph[GRAPH.get_node_idx(res.uuid)]
    assert_equal(gn.parents.size, 2)

    let ug1 = gn.backward_fn(upper_grad, gn.parents, 0)
    let ug2 = gn.backward_fn(upper_grad, gn.parents, 1)

    assert_tensors_equal(ug1, upper_grad)
    assert_tensors_equal(ug1, upper_grad)
    GRAPH.reset_all()


# <------------SUB------------>
fn test_SUB() raises:
    var t1: Tensor[dtype] = Tensor[dtype](2, 3)
    var t2: Tensor[dtype] = Tensor[dtype](2, 3)
    var upper_grad: Tensor[dtype] = Tensor[dtype](2, 3)
    fill[dtype, nelts](t1, 1.0)
    fill[dtype, nelts](t2, 2.0)
    fill[dtype, nelts](upper_grad, 1.0)

    let res = SUB.forward(t1, t2)

    let gn = GRAPH.graph[GRAPH.get_node_idx(res.uuid)]
    assert_equal(gn.parents.size, 2)

    let ug1 = gn.backward_fn(upper_grad, gn.parents, 0)
    let ug2 = gn.backward_fn(upper_grad, gn.parents, 1)

    var expected_ug2 = Tensor[dtype](2, 3)
    fill[dtype, nelts](expected_ug2, -1.0)

    assert_tensors_equal(ug1, upper_grad)
    assert_tensors_equal(ug2, expected_ug2)
    GRAPH.reset_all()


# <------------MUL------------>
fn test_MUL() raises:
    var t1: Tensor[dtype] = Tensor[dtype](2, 3)
    var t2: Tensor[dtype] = Tensor[dtype](2, 3)
    var upper_grad: Tensor[dtype] = Tensor[dtype](2, 3)
    fill[dtype, nelts](t1, 1.0)
    fill[dtype, nelts](t2, 2.0)
    fill[dtype, nelts](upper_grad, 1.0)

    let res = MUL.forward(t1, t2)

    let gn = GRAPH.graph[GRAPH.get_node_idx(res.uuid)]
    assert_equal(gn.parents.size, 2)

    let ug1 = gn.backward_fn(upper_grad, gn.parents, 0)
    let ug2 = gn.backward_fn(upper_grad, gn.parents, 1)

    var expected_ug1 = Tensor[dtype](2, 3)
    fill[dtype, nelts](expected_ug1, 2.0)
    var expected_ug2 = Tensor[dtype](2, 3)
    fill[dtype, nelts](expected_ug2, 1.0)

    assert_tensors_equal(ug1, expected_ug1)
    assert_tensors_equal(ug2, expected_ug2)
    GRAPH.reset_all()


# <------------DIV------------>
fn test_DIV() raises:
    var t1: Tensor[dtype] = Tensor[dtype](2, 3)
    var t2: Tensor[dtype] = Tensor[dtype](2, 3)
    var upper_grad: Tensor[dtype] = Tensor[dtype](2, 3)
    fill[dtype, nelts](t1, 1.0)
    fill[dtype, nelts](t2, 2.0)
    fill[dtype, nelts](upper_grad, 1.0)

    let res = DIV.forward(t1, t2)

    let gn = GRAPH.graph[GRAPH.get_node_idx(res.uuid)]
    assert_equal(gn.parents.size, 2)

    let ug1 = gn.backward_fn(upper_grad, gn.parents, 0)
    let ug2 = gn.backward_fn(upper_grad, gn.parents, 1)

    var expected_ug1 = Tensor[dtype](2, 3)
    fill[dtype, nelts](expected_ug1, 1.0 / 2.0)
    var expected_ug2 = Tensor[dtype](2, 3)
    fill[dtype, nelts](expected_ug2, -1.0 / (2.0**2))

    assert_tensors_equal(ug1, expected_ug1)
    assert_tensors_equal(ug2, expected_ug2)
    GRAPH.reset_all()


# <------------DOT------------>
fn test_DOT() raises:
    var t1: Tensor[dtype] = Tensor[dtype](2, 3)
    var t2: Tensor[dtype] = Tensor[dtype](2, 3)
    var upper_grad: Tensor[dtype] = Tensor[dtype](2, 3)
    fill[dtype, nelts](t1, 1.0)
    fill[dtype, nelts](t2, 2.0)
    fill[dtype, nelts](upper_grad, 1.0)

    let res = DOT.forward(t1, t2)

    let gn = GRAPH.graph[GRAPH.get_node_idx(res.uuid)]
    assert_equal(gn.parents.size, 2)

    let ug1 = gn.backward_fn(upper_grad, gn.parents, 0)
    let ug2 = gn.backward_fn(upper_grad, gn.parents, 1)

    var expected_ug1 = Tensor[dtype](2, 2)
    fill[dtype, nelts](expected_ug1, 6.0)
    var expected_ug2 = Tensor[dtype](3, 3)
    fill[dtype, nelts](expected_ug2, 2.0)

    assert_tensors_equal(ug1, expected_ug1)
    assert_tensors_equal(ug2, expected_ug2)
    GRAPH.reset_all()


# <------------EXP------------>
fn test_EXP() raises:
    var t1: Tensor[dtype] = Tensor[dtype](2, 3)
    var upper_grad: Tensor[dtype] = Tensor[dtype](2, 3)
    fill[dtype, nelts](t1, 2.0)
    fill[dtype, nelts](upper_grad, 5.0)

    let res = EXP.forward(t1)

    let gn = GRAPH.graph[GRAPH.get_node_idx(res.uuid)]
    assert_equal(gn.parents.size, 1)

    let ug1 = gn.backward_fn(upper_grad, gn.parents, 0)

    var expected_ug1 = Tensor[dtype](2, 3)
    fill[dtype, nelts](expected_ug1, 5.0 * exp[dtype, 1](2.0))
    assert_tensors_equal(ug1, expected_ug1)
    GRAPH.reset_all()


# <------------LOG------------>
fn test_LOG() raises:
    var t1: Tensor[dtype] = Tensor[dtype](2, 3)
    var upper_grad: Tensor[dtype] = Tensor[dtype](2, 3)
    fill[dtype, nelts](t1, 2.0)
    fill[dtype, nelts](upper_grad, 5.0)

    let res = LOG.forward(t1)

    let gn = GRAPH.graph[GRAPH.get_node_idx(res.uuid)]
    assert_equal(gn.parents.size, 1)

    let ug1 = gn.backward_fn(upper_grad, gn.parents, 0)

    var expected_ug1 = Tensor[dtype](2, 3)
    fill[dtype, nelts](expected_ug1, 5.0 / 2.0)
    assert_tensors_equal(ug1, expected_ug1)
    GRAPH.reset_all()


# <------------POW------------>
fn test_POW() raises:
    var t2: Tensor[dtype] = Tensor[dtype](2, 3)
    var upper_grad: Tensor[dtype] = Tensor[dtype](2, 3)
    fill[dtype, nelts](t2, 2.0)
    fill[dtype, nelts](upper_grad, 1.0)

    let res = POW.forward(t2, 2)

    let gn = GRAPH.graph[GRAPH.get_node_idx(res.uuid)]
    assert_equal(gn.parents.size, 2)

    let ug1 = gn.backward_fn(upper_grad, gn.parents, 0)
    let ug2 = gn.backward_fn(upper_grad, gn.parents, 1)

    var expected_ug1 = Tensor[dtype](2, 3)
    fill[dtype, nelts](expected_ug1, 4.0)
    var expected_ug2 = Tensor[dtype](2, 3)
    fill[dtype, nelts](expected_ug2, (2**2) * log[dtype, 1](2))

    assert_tensors_equal(ug1, expected_ug1)
    assert_tensors_equal(ug2, expected_ug2)
    GRAPH.reset_all()


# <------------SUM------------>
fn test_SUM() raises:
    # SUM ALL ELEMENTS
    var t1: Tensor[dtype] = Tensor[dtype](2, 3)
    fill[dtype, nelts](t1, 1.0)

    let res = SUM.forward(t1)

    # uppergrad has always to same shape as res
    var upper_grad: Tensor[dtype] = Tensor[dtype](res.tensor.shape())
    fill[dtype, nelts](upper_grad, 9.0)
    let gn = GRAPH.graph[GRAPH.get_node_idx(res.uuid)]
    assert_equal(gn.parents.size, 1)  # one parent

    let ug1 = gn.backward_fn(upper_grad, gn.parents, 0)

    var expected_ug1 = Tensor[dtype](2, 3)
    fill[dtype, nelts](expected_ug1, 9.0)
    assert_tensors_equal(ug1, expected_ug1)
    GRAPH.reset_all()


fn test_SUM_0() raises:
    # SUM ALONG AXIS 0
    var t1: Tensor[dtype] = Tensor[dtype](2, 3)
    fill[dtype, nelts](t1, 1.0)

    let res = SUM.forward[axis=0](t1)

    # uppergrad has always to same shape as res
    var upper_grad: Tensor[dtype] = Tensor[dtype](res.tensor.shape())
    upper_grad[0] = 0.0
    upper_grad[1] = 1.0
    upper_grad[2] = 2.0
    let gn = GRAPH.graph[GRAPH.get_node_idx(res.uuid)]
    assert_equal(gn.parents.size, 1)  # one parent

    let ug1 = gn.backward_fn(upper_grad, gn.parents, 0)

    var expected_ug1 = Tensor[dtype](2, 3)
    for i in range(expected_ug1.num_elements()):
        expected_ug1[i] = i % 3
    assert_tensors_equal(ug1, expected_ug1)
    GRAPH.reset_all()


fn test_SUM_1() raises:
    # SUM ALONG AXIS 1
    var t1: Tensor[dtype] = Tensor[dtype](2, 3)
    fill[dtype, nelts](t1, 1.0)

    let res = SUM.forward[axis=1](t1)

    # uppergrad has always to same shape as res
    var upper_grad: Tensor[dtype] = Tensor[dtype](res.tensor.shape())
    upper_grad[0] = 0.0
    upper_grad[1] = 1.0
    let gn = GRAPH.graph[GRAPH.get_node_idx(res.uuid)]
    assert_equal(gn.parents.size, 1)  # one parent

    let ug1 = gn.backward_fn(upper_grad, gn.parents, 0)

    var expected_ug1 = Tensor[dtype](2, 3)
    for i in range(expected_ug1.num_elements()):
        expected_ug1[i] = 0 if i < 3 else 1
    assert_tensors_equal(ug1, expected_ug1)
    GRAPH.reset_all()


# <------------TRANSPOSE------------>
fn test_TRANSPOSE() raises:
    let t1 = Tensor[dtype](2, 3)

    let res = TRANSPOSE.forward(t1)

    # uppergrad has always to same shape as res
    var upper_grad: Tensor[dtype] = Tensor[dtype](res.tensor.shape())
    assert_equal(upper_grad.dim(0), 3)
    assert_equal(upper_grad.dim(1), 2)
    for i in range(3):
        upper_grad[2*i] = i+1
        upper_grad[2*i+1] = i+4
    let gn = GRAPH.graph[GRAPH.get_node_idx(res.uuid)]
    assert_equal(gn.parents.size, 1)  # one parent

    let ug1 = gn.backward_fn(upper_grad, gn.parents, 0)

    var expected_ug1 = Tensor[dtype](t1.shape())
    for i in range(6):
        expected_ug1[i] = i+1
    assert_tensors_equal(ug1, expected_ug1)
    GRAPH.reset_all()
    

# <------------FLATTEN------------>
fn test_FLATTEN() raises:
    let t1 = Tensor[dtype](2, 3)

    let res = FLATTEN.forward(t1)

    # uppergrad has always to same shape as res
    var upper_grad: Tensor[dtype] = Tensor[dtype](res.tensor.shape())
    fill[dtype, nelts](upper_grad, 1.0)
    assert_equal(upper_grad.dim(0), 6)
    let gn = GRAPH.graph[GRAPH.get_node_idx(res.uuid)]
    assert_equal(gn.parents.size, 1)  # one parent

    let ug1 = gn.backward_fn(upper_grad, gn.parents, 0)

    var expected_ug1 = Tensor[dtype](t1.shape())
    fill[dtype, nelts](expected_ug1, 1.0)
    assert_tensors_equal(ug1, expected_ug1)
    GRAPH.reset_all()


# <------------RESHAPE------------>
fn test_RESHAPE() raises:
    let t1 = Tensor[dtype](2, 2, 5)
    let new_shape = TensorShape(2, 10)

    let res = RESHAPE.forward(t1, new_shape)

    # uppergrad has always to same shape as res
    var upper_grad: Tensor[dtype] = Tensor[dtype](res.tensor.shape())
    fill[dtype, nelts](upper_grad, 1.0)
    assert_equal(upper_grad.dim(0), 2)
    assert_equal(upper_grad.dim(1), 10)
    let gn = GRAPH.graph[GRAPH.get_node_idx(res.uuid)]
    assert_equal(gn.parents.size, 1)  # one parent

    let ug1 = gn.backward_fn(upper_grad, gn.parents, 0)

    var expected_ug1 = Tensor[dtype](t1.shape())
    fill[dtype, nelts](expected_ug1, 1.0)
    assert_tensors_equal(ug1, expected_ug1)
    GRAPH.reset_all()


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
        test_TRANSPOSE()
        test_FLATTEN()
        test_RESHAPE()
    except:
        print("[ERROR] Error in backward pass.")
