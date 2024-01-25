from tensor import Tensor
from testing import assert_equal, assert_almost_equal
from math import log
from math.limit import max_finite

from dainemo import GRAPH
import dainemo.nn as nn
from dainemo.autograd.node import Node
from dainemo.utils.tensorutils import fill

alias dtype = DType.float32 
alias nelts: Int = simdwidthof[dtype]()


fn test_MSE_perfect() raises:
    var loss_func = nn.MSELoss()

    var output = Tensor[dtype](2, 10)       # batch of 2, 10 classes
    var labels = Tensor[dtype](2, 10)       
    fill[dtype, nelts](output, 1)
    fill[dtype, nelts](labels, 1)
    let outputs = Node[dtype](output)

    let loss = loss_func(outputs, labels)

    assert_equal(loss.tensor.dim(0), 1)     # MSE summed over all elements
    assert_equal(loss.tensor[0], 0)         # loss is 0

    assert_equal(GRAPH.graph.size, 8)        # outputs, sum, div2n, squared_difference, difference, [2], outputs, targets
    GRAPH.reset_all()


fn test_MSE_imperfect() raises:
    var loss_func = nn.MSELoss()

    var output = Tensor[dtype](1, 10)       # batch of 1, 10 classes
    var labels = Tensor[dtype](1, 10)       
    fill[dtype, nelts](output, 1)
    let outputs = Node[dtype](output)
    for i in range(10):
        labels[i] = i

    let loss = loss_func(outputs, labels)
    
    var expected_loss: SIMD[dtype, 1] = 0.0
    for i in range(10):
        expected_loss += (output[i] - labels[i])**2
    expected_loss = expected_loss / (2*10)
    assert_equal(loss.tensor[0], expected_loss)

    assert_equal(GRAPH.graph.size, 8)
    GRAPH.reset_all()


fn calc_CE_loss(output: Tensor[dtype], labels: Tensor[dtype]) raises -> SIMD[dtype, 1]:
    assert_equal(output.num_elements(), labels.num_elements())
    var expected_loss: SIMD[dtype, 1] = 0.0
    alias epsilon = 1e-9
    for i in range(output.num_elements()):
        expected_loss += labels[i] * log[dtype](output[i]+epsilon)
    expected_loss = -expected_loss / output.dim(0)
    return expected_loss


fn test_CE_perfect() raises:
    var loss_func = nn.CrossEntropyLoss()

    var output = Tensor[dtype](2, 10)       # batch of 2, 10 classes
    var labels = Tensor[dtype](2, 10)       
    fill[dtype, nelts](output, 2)
    fill[dtype, nelts](labels, 2)
    var outputs = Node[dtype](output)


    let loss = loss_func(outputs, labels)

    let expected_loss = calc_CE_loss(output, labels)
    assert_equal(loss.tensor.dim(0), 1)     # CE summed over all elements
    assert_almost_equal(loss.tensor[0], expected_loss, relative_tolerance=1e-5)

    assert_equal(GRAPH.graph.size, 9)        # outputs, epsilon, add, entropy, negdivN, targets_logout, targets, logout, outputs
    GRAPH.reset_all()


fn test_CE_imperfect() raises:
    var loss_func = nn.CrossEntropyLoss()

    var output = Tensor[dtype](1, 10)       # batch of 1, 10 classes
    var labels = Tensor[dtype](1, 10)       
    for i in range(10):
        labels[i] = i
        output[i] = 10 - i
    var outputs = Node[dtype](output)

    let loss = loss_func(outputs, labels)

    let expected_loss = calc_CE_loss(output, labels)
    assert_almost_equal(loss.tensor[0], expected_loss, relative_tolerance=1e-5)
    assert_equal(GRAPH.graph.size, 9)
    GRAPH.reset_all()


fn calc_BCE_loss(output: Tensor[dtype], labels: Tensor[dtype]) raises -> SIMD[dtype, 1]:
    assert_equal(output.num_elements(), labels.num_elements())
    var expected_loss: SIMD[dtype, 1] = 0.0
    alias epsilon = 1e-9
    for i in range(output.num_elements()):
        expected_loss += labels[i] * log[dtype](output[i]+epsilon) + (1-labels[i]) * log[dtype](1-output[i]+epsilon)
    expected_loss = -expected_loss / output.num_elements()
    return expected_loss


fn test_BCE_perfect() raises:
    var loss_func = nn.BCELoss()

    var output = Tensor[dtype](2, 10)       # batch of 2, 10 classes
    var labels = Tensor[dtype](2, 10)       
    fill[dtype, nelts](output, 0.9999999)
    fill[dtype, nelts](labels, 1)
    var outputs = Node[dtype](output)

    let loss = loss_func(outputs, labels)

    let expected_loss = calc_BCE_loss(output, labels)
    assert_equal(loss.tensor.dim(0), 1)     # CE summed over all elements
    assert_almost_equal(loss.tensor[0], expected_loss, relative_tolerance=1e-5)

    assert_equal(GRAPH.graph.size, 13)        # outputs, entropy, negdivN, sum, targets_logout, targets_logout_1min, targets_1min, logout_1min, [1], targets, log, outputs, logout
    GRAPH.reset_all()


fn test_BCE_imperfect() raises:
    var loss_func = nn.BCELoss()

    var output = Tensor[dtype](2, 10)       # batch of 2, 10 classes
    var labels = Tensor[dtype](2, 10)       
    fill[dtype, nelts](output, 0.55)
    fill[dtype, nelts](labels, 1)
    var outputs = Node[dtype](output)

    let loss = loss_func(outputs, labels)

    let expected_loss = calc_BCE_loss(output, labels)
    assert_almost_equal(loss.tensor[0], expected_loss, relative_tolerance=1e-5)
    assert_equal(GRAPH.graph.size, 13)
    GRAPH.reset_all()



fn main():
    try:
        test_MSE_perfect()
        test_MSE_imperfect()
        test_CE_perfect()
        test_CE_imperfect()
        test_BCE_perfect()
        test_BCE_imperfect()
    except e:
        print("[ERROR] Error in loss")
        print(e)
    