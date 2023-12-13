from tensor import Tensor
from testing import assert_equal

from dainemo import GRAPH
import dainemo.nn as nn
from dainemo.utils.tensorutils import fill

alias dtype = DType.float32 
alias nelts: Int = simdwidthof[dtype]()


fn test_MSE_perfect() raises:
    var loss_func = nn.MSELoss()

    var output = Tensor[dtype](2, 10)       # batch of 2, 10 classes
    var labels = Tensor[dtype](2, 10)       
    fill[dtype, nelts](output, 1)
    fill[dtype, nelts](labels, 1)

    let loss = loss_func(output, labels)

    assert_equal(loss.tensor.dim(0), 2)     # batch size
    assert_equal(loss.tensor.dim(1), 1)     # 1 loss per batch
    assert_equal(loss.tensor[0], 0)         # loss is 0
    assert_equal(loss.tensor[1], 0)         # loss is 0

    assert_equal(GRAPH.graph.size, 8)        # outputs, labels, diff, [2], pow, sum, div2n, loss
    GRAPH.reset()


fn test_MSE_imperfect() raises:
    var loss_func = nn.MSELoss()

    var output = Tensor[dtype](1, 10)       # batch of 1, 3 classes
    var labels = Tensor[dtype](1, 10)       
    fill[dtype, nelts](output, 1)
    for i in range(10):
        labels[i] = i

    let loss = loss_func(output, labels)

    assert_equal(loss.tensor.dim(0), 1)     # batch size
    assert_equal(loss.tensor.dim(1), 1)     # 1 loss per batch
    
    var expected_loss: SIMD[dtype, 1] = 0.0
    for i in range(10):
        expected_loss += (output[i] - labels[i])**2
    expected_loss = expected_loss / (2*10)
    assert_equal(loss.tensor[0], expected_loss)

    assert_equal(GRAPH.graph.size, 8)
    GRAPH.reset()


fn main():
    try:
        test_MSE_perfect()
        test_MSE_imperfect()
    except:
        print("[ERROR] Error in loss")
    