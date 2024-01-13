from random import rand
from tensor import Tensor, TensorShape
from testing import assert_equal
from test_tensorutils import assert_tensors_equal

from dainemo import GRAPH
from dainemo.autograd.ops.mlops import SIGMOID
from dainemo.utils.tensorutils import fill

alias dtype = DType.float32
alias nelts: Int = simdwidthof[dtype]()


fn test_SIGMOID() raises:
    let t1: Tensor[dtype] = Tensor[dtype](2, 3)  # filled with zeroes

    let res = SIGMOID.forward(t1)

    var expected = Tensor[dtype](2, 3)
    fill[dtype, nelts](expected, 0.5)
    assert_tensors_equal(res.tensor, expected)
    assert_equal(GRAPH.graph.size, 2)
    GRAPH.reset_all()


fn test_backward_SIGMOID() raises:
    let t1: Tensor[dtype] = Tensor[dtype](2, 3)  # filled with zeroes

    var upper_grad: Tensor[dtype] = Tensor[dtype](2, 3)
    fill[dtype, nelts](upper_grad, 5.0)

    let res = SIGMOID.forward(t1)

    let gn = GRAPH.graph[GRAPH.get_node_idx(res.uuid)]
    assert_equal(gn.parents.size, 1)

    let ug1 = gn.backward_fn(upper_grad, gn.parents, 0)

    var expected_ug1 = Tensor[dtype](2, 3)
    fill[dtype, nelts](expected_ug1, 5.0 * 0.25)  # 0.25 = sigmoid(0) * (1 - sigmoid(0))
    assert_tensors_equal(ug1, expected_ug1)
    GRAPH.reset_all()


fn main():
    try:
        test_SIGMOID()
    except e:
        print("[ERROR] Error in forward mlops")
        print(e)
        return

    try:
        test_backward_SIGMOID()
    except e:
        print("[ERROR] Error in backward mlops")
        print(e)
        return
