from random import rand
from tensor import Tensor
from testing import assert_equal
from test_tensorutils import assert_tensors_equal

from dainemo import GRAPH
from dainemo.autograd.ops.basics import ADD, SUB, DOT, SUM, MUL, POW, DIV
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
    GRAPH.reset()


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
    GRAPH.reset()


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
    GRAPH.reset()

    res = MUL.forward(t1, 5)
    fill[dtype, nelts](expected, 5)
    assert_tensors_equal(res.tensor, expected)
    assert_equal(GRAPH.graph.size, 3)
    GRAPH.reset()


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
    GRAPH.reset()

    res = DIV.forward(t1, 5)
    fill[dtype, nelts](expected, 1.0 / 5.0)
    assert_tensors_equal(res.tensor, expected)
    assert_equal(GRAPH.graph.size, 3)
    GRAPH.reset()


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
    GRAPH.reset()


# <------------POW------------>
fn test_POW() raises:
    var t1: Tensor[dtype] = Tensor[dtype](2, 3)
    fill[dtype, nelts](t1, 2.0)

    let res = POW.forward(t1, 2)

    var expected = Tensor[dtype](2, 3)
    fill[dtype, nelts](expected, 4.0)
    assert_tensors_equal(res.tensor, expected)
    assert_equal(GRAPH.graph.size, 3)
    GRAPH.reset()


# <------------SUM------------>
fn test_SUM() raises:
    var t1: Tensor[dtype] = Tensor[dtype](2, 3)
    fill[dtype, nelts](t1, 1.0)

    let res_scalar = SUM.forward(t1)

    var expected = Tensor[dtype](1)
    fill[dtype, nelts](expected, 6.0)
    assert_tensors_equal(res_scalar.tensor, expected)
    assert_equal(GRAPH.graph.size, 2)
    GRAPH.reset()

    let res_0 = SUM.forward[axis=0](t1)

    expected = Tensor[dtype](1, 3)
    fill[dtype, nelts](expected, 2.0)
    assert_tensors_equal(res_0.tensor, expected)
    assert_equal(GRAPH.graph.size, 2)
    GRAPH.reset()

    let res_1 = SUM.forward[axis=1](t1)

    expected = Tensor[dtype](2, 1)
    fill[dtype, nelts](expected, 3.0)
    assert_tensors_equal(res_1.tensor, expected)
    assert_equal(GRAPH.graph.size, 2)
    GRAPH.reset()


fn main():
    try:
        test_ADD()
        test_SUB()
        test_MUL()
        test_DIV()
        test_DOT()
        test_POW()
        test_SUM()
    except:
        print("[ERROR] Error in ops")
        return
