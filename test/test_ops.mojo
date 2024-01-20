from random import rand
from tensor import Tensor, TensorShape
from testing import assert_equal
from test_tensorutils import assert_tensors_equal
from math import exp, log

from dainemo import GRAPH
from dainemo.autograd.ops.basics import (
    ADD,
    SUB,
    MUL,
    DIV,
    DOT,
    EXP,
    LOG,
    POW,
    SUM,
    MAX,
    TRANSPOSE,
    FLATTEN,
    RESHAPE,
)
from dainemo.utils.tensorutils import fill

alias dtype = DType.float32
alias nelts: Int = simdwidthof[dtype]()


# <------------ADD------------>
fn test_ADD() raises:
    var t1: Tensor[dtype] = Tensor[dtype](2, 3)
    var t2: Tensor[dtype] = Tensor[dtype](2, 3)
    fill[dtype, nelts](t1, 1.0)
    fill[dtype, nelts](t2, 1.0)

    let res = ADD.forward(t1, t2)

    var expected = Tensor[dtype](2, 3)
    fill[dtype, nelts](expected, 2.0)
    assert_tensors_equal(res.tensor, expected)
    assert_equal(GRAPH.graph.size, 3)
    GRAPH.reset_all()


# <------------SUB------------>
fn test_SUB() raises:
    var t1: Tensor[dtype] = Tensor[dtype](2, 3)
    var t2: Tensor[dtype] = Tensor[dtype](2, 3)
    fill[dtype, nelts](t1, 1.0)
    fill[dtype, nelts](t2, 1.0)

    let res = SUB.forward(t1, t2)

    let expected = Tensor[dtype](2, 3)
    assert_tensors_equal(res.tensor, expected)
    assert_equal(GRAPH.graph.size, 3)
    GRAPH.reset_all()


# <------------MUL------------>
fn test_MUL() raises:
    var t1: Tensor[dtype] = Tensor[dtype](2, 3)
    var t2: Tensor[dtype] = Tensor[dtype](2, 3)
    fill[dtype, nelts](t1, 1.0)
    fill[dtype, nelts](t2, 1.0)

    var res = MUL.forward(t1, t2)

    var expected = Tensor[dtype](2, 3)
    fill[dtype, nelts](expected, 1.0)
    assert_tensors_equal(res.tensor, expected)
    assert_equal(GRAPH.graph.size, 3)
    GRAPH.reset_all()

    res = MUL.forward(t1, 5)
    fill[dtype, nelts](expected, 5)
    assert_tensors_equal(res.tensor, expected)
    assert_equal(GRAPH.graph.size, 3)
    GRAPH.reset_all()


# <------------DIV------------>
fn test_DIV() raises:
    var t1: Tensor[dtype] = Tensor[dtype](2, 3)
    var t2: Tensor[dtype] = Tensor[dtype](2, 3)
    fill[dtype, nelts](t1, 1.0)
    fill[dtype, nelts](t2, 3.0)

    var res = DIV.forward(t1, t2)

    var expected = Tensor[dtype](2, 3)
    fill[dtype, nelts](expected, 1.0 / 3.0)
    assert_tensors_equal(res.tensor, expected)
    assert_equal(GRAPH.graph.size, 3)
    GRAPH.reset_all()

    res = DIV.forward(t1, 5)
    fill[dtype, nelts](expected, 1.0 / 5.0)
    assert_tensors_equal(res.tensor, expected)
    assert_equal(GRAPH.graph.size, 3)
    GRAPH.reset_all()


# <------------DOT------------>
fn test_DOT() raises:
    var t1: Tensor[dtype] = Tensor[dtype](2, 3)
    var t2: Tensor[dtype] = Tensor[dtype](3, 2)
    fill[dtype, nelts](t1, 1.0)
    fill[dtype, nelts](t2, 1.0)

    let res = DOT.forward(t1, t2)

    var expected = Tensor[dtype](2, 2)
    fill[dtype, nelts](expected, 3.0)
    assert_tensors_equal(res.tensor, expected)
    assert_equal(GRAPH.graph.size, 3)
    GRAPH.reset_all()


# <------------EXP------------>
fn test_EXP() raises:
    var t1: Tensor[dtype] = Tensor[dtype](2, 3)
    fill[dtype, nelts](t1, 2.0)

    let res = EXP.forward(t1)

    var expected = Tensor[dtype](2, 3)
    fill[dtype, nelts](expected, exp[dtype, 1](2.0))
    assert_tensors_equal(res.tensor, expected)
    assert_equal(GRAPH.graph.size, 2)
    GRAPH.reset_all()


# <------------LOG------------>
fn test_LOG() raises:
    var t1: Tensor[dtype] = Tensor[dtype](2, 3)
    fill[dtype, nelts](t1, 2.0)

    let res = LOG.forward(t1)

    var expected = Tensor[dtype](2, 3)
    fill[dtype, nelts](expected, log[dtype, 1](2.0))
    assert_tensors_equal(res.tensor, expected)
    assert_equal(GRAPH.graph.size, 2)
    GRAPH.reset_all()


# <------------POW------------>
fn test_POW() raises:
    var t1: Tensor[dtype] = Tensor[dtype](2, 3)
    fill[dtype, nelts](t1, 2.0)

    let res = POW.forward(t1, 2)

    var expected = Tensor[dtype](2, 3)
    fill[dtype, nelts](expected, 4.0)
    assert_tensors_equal(res.tensor, expected)
    assert_equal(GRAPH.graph.size, 3)
    GRAPH.reset_all()


# <------------SUM------------>
fn test_SUM() raises:
    var t1: Tensor[dtype] = Tensor[dtype](2, 3)
    fill[dtype, nelts](t1, 1.0)

    let res_scalar = SUM.forward(t1)

    var expected = Tensor[dtype](1)
    fill[dtype, nelts](expected, 6.0)
    assert_tensors_equal(res_scalar.tensor, expected)
    assert_equal(GRAPH.graph.size, 2)
    GRAPH.reset_all()

    let res_0 = SUM.forward[axis=0](t1)

    expected = Tensor[dtype](1, 3)
    fill[dtype, nelts](expected, 2.0)
    assert_tensors_equal(res_0.tensor, expected)
    assert_equal(GRAPH.graph.size, 2)
    GRAPH.reset_all()

    let res_1 = SUM.forward[axis=1](t1)

    expected = Tensor[dtype](2, 1)
    fill[dtype, nelts](expected, 3.0)
    assert_tensors_equal(res_1.tensor, expected)
    assert_equal(GRAPH.graph.size, 2)
    GRAPH.reset_all()


# <------------MAX------------>
fn test_MAX() raises:
    var t = Tensor[dtype](2, 3, 2)
    for i in range(12):
        t[i] = i + 1

    let tensor_max = MAX.forward(t)
    var expected = Tensor[dtype](1)
    fill[dtype, nelts](expected, 12)
    assert_tensors_equal(tensor_max.tensor, expected)
    assert_equal(GRAPH.graph.size, 2)
    GRAPH.reset_all()

    @parameter
    fn fill_tensor[size: Int](inout tensor: Tensor[dtype], values: StaticIntTuple[size]):
        for i in range(tensor.num_elements()):
            tensor[i] = values[i]

    let tensor_max_axis_0 = MAX.forward[axis=0](t)
    var expected_max_axis_0_temp = StaticIntTuple[6](7, 8, 9, 10, 11, 12)
    expected = Tensor[dtype](1, 3, 2)
    fill_tensor(expected, expected_max_axis_0_temp)
    assert_tensors_equal(tensor_max_axis_0.tensor, expected)
    assert_equal(GRAPH.graph.size, 2)
    GRAPH.reset_all()

    let tensor_max_axis_1 = MAX.forward[axis=1](t)
    var expected_max_axis_1_temp = StaticIntTuple[4](5, 6, 11, 12)
    expected = Tensor[dtype](2, 1, 2)
    fill_tensor(expected, expected_max_axis_1_temp)
    assert_tensors_equal(tensor_max_axis_1.tensor, expected)
    GRAPH.reset_all()


# <------------TRANSPOSE------------>
fn test_TRANSPOSE() raises:
    var A = Tensor[dtype](2, 3)
    var B = Tensor[dtype](3, 2)
    for i in range(6):
        A[i] = i + 1
    for i in range(3):
        B[2 * i] = i + 1
        B[2 * i + 1] = i + 4

    let res = TRANSPOSE.forward(A)

    assert_tensors_equal(res.tensor, B)
    assert_equal(GRAPH.graph.size, 2)
    GRAPH.reset_all()


# <------------FLATTEN------------>
fn test_FLATTEN() raises:
    var A = Tensor[dtype](2, 3)
    var B = Tensor[dtype](6)
    for i in range(6):
        A[i] = i + 1
        B[i] = i + 1

    let res = FLATTEN.forward(A)

    assert_tensors_equal(res.tensor, B)
    assert_equal(GRAPH.graph.size, 2)
    GRAPH.reset_all()


# <------------RESHAPE------------>
fn test_RESHAPE() raises:
    var A = Tensor[dtype](2, 2, 5)
    let new_shape = TensorShape(2, 10)

    var B = Tensor[dtype](new_shape)
    for i in range(20):
        A[i] = i + 1
        B[i] = i + 1

    let res = RESHAPE.forward(A, new_shape)

    assert_tensors_equal(res.tensor, B)
    assert_equal(GRAPH.graph.size, 2)
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
        test_MAX()
        test_TRANSPOSE()
        test_FLATTEN()
        test_RESHAPE()
    except e:
        print("[ERROR] Error in ops")
        print(e)
        return
