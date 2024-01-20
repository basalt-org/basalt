from tensor import Tensor
from testing import assert_equal

from dainemo import GRAPH
import dainemo.nn as nn
from dainemo.utils.tensorutils import fill
from test_tensorutils import assert_tensors_equal

alias dtype = DType.float32
alias nelts: Int = simdwidthof[dtype]()


fn test_SOFTMAX() raises:
    var x = Tensor[dtype](2, 3, 2)
    fill[dtype, nelts](x, 4)

    var res = nn.Softmax.forward[0](x)
    var expected = Tensor[dtype](2, 3, 2)
    fill[dtype, nelts](expected, 0.5)
    assert_tensors_equal(res.tensor, expected)
    assert_equal(GRAPH.graph.size, 6) # inputs, max_values, exp_values, sum_values, diff_max_values, result_div
    GRAPH.reset_all()

    res = nn.Softmax.forward[1](x)
    expected = Tensor[dtype](2, 3, 2)
    fill[dtype, nelts](expected, 1.0 / 3.0)
    assert_tensors_equal(res.tensor, expected, "almost")
    assert_equal(GRAPH.graph.size, 6)
    GRAPH.reset_all()

    res = nn.Softmax.forward[2](x)
    expected = Tensor[dtype](2, 3, 2)
    fill[dtype, nelts](expected, 0.5)
    assert_tensors_equal(res.tensor, expected)
    assert_equal(GRAPH.graph.size, 6)
    GRAPH.reset_all()


fn test_LOGSOFTMAX() raises:
    var x = Tensor[dtype](2, 3, 2)
    fill[dtype, nelts](x, 4)

    var res = nn.LogSoftmax.forward[0](x)
    var expected = Tensor[dtype](2, 3, 2)
    fill[dtype, nelts](expected, -0.69314718)
    assert_tensors_equal(res.tensor, expected)
    assert_equal(GRAPH.graph.size, 7)
    GRAPH.reset_all()

    res = nn.LogSoftmax.forward[1](x)
    expected = Tensor[dtype](2, 3, 2)
    fill[dtype, nelts](expected, -1.09861231)
    assert_tensors_equal(res.tensor, expected, "almost")
    assert_equal(GRAPH.graph.size, 7)
    GRAPH.reset_all()

    res = nn.LogSoftmax.forward[2](x)
    expected = Tensor[dtype](2, 3, 2)
    fill[dtype, nelts](expected, -0.69314718)
    assert_tensors_equal(res.tensor, expected)
    assert_equal(GRAPH.graph.size, 7)
    GRAPH.reset_all()


fn main():
    try:
        test_SOFTMAX()
        test_LOGSOFTMAX()
    except e:
        print("[ERROR] Error in activations")
        print(e)
