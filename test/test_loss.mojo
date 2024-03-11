from tensor import Tensor, TensorShape
from testing import assert_equal, assert_almost_equal
# from math import log
# from math.limit import max_finite

import dainemo.nn as nn
from dainemo import Graph, Symbol, OP
from dainemo.utils.tensorutils import fill

alias dtype = DType.float32 
alias nelts: Int = simdwidthof[dtype]()


fn test_MSE_perfect() raises:
    alias y_pred_shape = TensorShape(2, 10)
    alias y_true_shape = TensorShape(2, 10)

    fn create_graph() -> Graph:
        var g = Graph()

        var y_pred = g.input(y_pred_shape)
        var y_true = g.input(y_true_shape)

        var loss = nn.MSELoss(g, y_pred, y_true)

        g.out(loss)

        return g ^

    alias graph = create_graph()
    assert_equal(len(graph.nodes), 3)

    var y_pred = Tensor[dtype](y_pred_shape)       # batch of 2, 10 classes
    var y_true = Tensor[dtype](y_true_shape)

    fill[dtype, nelts](y_pred, 1)
    fill[dtype, nelts](y_true, 1)

    var model = nn.Model[graph](inference_only=True)
    
    var loss = model.inference(y_pred, y_true)[0]

    assert_equal(loss.dim(0), 1)     # MSE summed over all elements
    assert_equal(loss[0], 0)         # loss is 0


fn test_MSE_imperfect() raises:
    alias y_pred_shape = TensorShape(1, 10)
    alias y_true_shape = TensorShape(1, 10)

    fn create_graph() -> Graph:
        var g = Graph()

        var y_pred = g.input(y_pred_shape)
        var y_true = g.input(y_true_shape)

        var loss = nn.MSELoss(g, y_pred, y_true)

        g.out(loss)

        return g ^

    alias graph = create_graph()
    assert_equal(len(graph.nodes), 3)

    var y_pred = Tensor[dtype](y_pred_shape)       # batch of 1, 10 classes
    var y_true = Tensor[dtype](y_true_shape)

    fill[dtype, nelts](y_pred, 1)

    for i in range(10):
        y_true[i] = i

    var model = nn.Model[graph](inference_only=True)

    var loss = model.inference(y_pred, y_true)[0]

    var expected_loss: SIMD[dtype, 1] = 0.0

    for i in range(10):
        expected_loss += (y_pred[i] - y_true[i])**2

    expected_loss = expected_loss / y_true_shape[1]

    assert_almost_equal(loss[0], expected_loss)


fn main():
    try:
        test_MSE_perfect()
        test_MSE_imperfect()
    except e:
        print("[ERROR] Error in loss")
        print(e)
    