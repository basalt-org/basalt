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


fn main():
    try:
        test_SIGMOID()
    except:
        print("[ERROR] Error in mlops")
        return
