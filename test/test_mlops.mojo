from random import rand
from tensor import TensorShape
from testing import assert_equal
from test_tensorutils import assert_tensors_equal


import dainemo.nn as nn
from dainemo import Graph, Symbol, OP
from dainemo.autograd.ops.mlops import SIGMOID, RELU, TANH
from dainemo.utils.tensorutils import fill


alias dtype = DType.float32
alias nelts: Int = simdwidthof[dtype]()



# ------ Test Unary Ops ------
fn test_unary_op[
    op: OP, t1_shape: TensorShape
](t1: Tensor[dtype], expected: Tensor[dtype]) raises:
    fn create_graph() -> Graph:
        var g = Graph()
        var t1 = g.input(t1_shape)

        var res = g.op(op, t1)
        g.out(res)

        return g ^

    alias graph = create_graph()
    assert_equal(len(graph.nodes), 1)

    var model = nn.Model[graph]()
    var res = model.forward(t1)

    assert_tensors_equal(res, expected)


# <------------SIGMOID------------>
fn test_SIGMOID() raises:
    alias t1_shape = TensorShape(2, 3)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)  # filled with zeroes

    var expected = Tensor[dtype](2, 3)
    fill[dtype, nelts](expected, 0.5)
    
    test_unary_op[OP.SIGMOID, t1_shape](t1, expected)


fn test_backward_SIGMOID() raises:
    alias t1_shape = TensorShape(2, 3)
    alias ug_shape = TensorShape(2, 3)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)  # filled with zeroes
    var ug: Tensor[dtype] = Tensor[dtype](ug_shape)
    fill[dtype, nelts](ug, 5.0)

    var expected_grad = Tensor[dtype](2, 3)
    fill[dtype, nelts](
        expected_grad, 5.0 * 0.25
    )  # 0.25 = d(sigmoid(0))/dx = sigmoid(0) * (1 - sigmoid(0))
    
    var grad = SIGMOID.backward[ug_shape, t1_shape](ug, t1)
    assert_tensors_equal(grad, expected_grad)


# <------------RELU------------>
fn test_RELU() raises:
    alias t1_shape = TensorShape(2, 3)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    # TODO: When tensors can do slices, this could be changed to two fill functions.
    for i in range(3):
        t1[i] = 3
    for i in range(3, 6):
        t1[i] = -3

    var expected = Tensor[dtype](2, 3)
    for i in range(3):
        expected[i] = 3
    for i in range(3, 6):
        expected[i] = 0
    
    test_unary_op[OP.RELU, t1_shape](t1, expected)


fn test_backward_RELU() raises:
    alias t1_shape = TensorShape(2, 3)
    alias ug_shape = TensorShape(2, 3)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    var ug: Tensor[dtype] = Tensor[dtype](ug_shape)
    for i in range(3):
        t1[i] = 3
    for i in range(3, 6):
        t1[i] = -3
    fill[dtype, nelts](ug, 5.0)

    var expected_grad = Tensor[dtype](2, 3)
    for i in range(3):
        expected_grad[i] = 1 * 5.0  # 1 = d(relu(3))/dx
    for i in range(3, 6):
        expected_grad[i] = 0 * 5.0  # 0 = d(relu(-3))/dx

    var grad = RELU.backward[ug_shape, t1_shape](ug, t1)
    assert_tensors_equal(grad, expected_grad)


# <------------TANH------------>
fn test_TANH() raises:
    alias t1_shape = TensorShape(2, 3)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)  # filled with zeroes

    var expected = Tensor[dtype](2, 3)
    fill[dtype, nelts](expected, 0.0)

    test_unary_op[OP.TANH, t1_shape](t1, expected)


fn test_backward_TANH() raises:
    alias t1_shape = TensorShape(2, 3)
    alias ug_shape = TensorShape(2, 3)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)  # filled with zeroes
    var ug: Tensor[dtype] = Tensor[dtype](ug_shape)
    fill[dtype, nelts](ug, 5.0)

    var expected_grad = Tensor[dtype](2, 3)
    fill[dtype, nelts](expected_grad, 5.0 * 1.0)  # 1.0 = d(tanh(0))/dx = 1 - tanh(0)^2
    
    var grad = TANH.backward[ug_shape, t1_shape](ug, t1)
    assert_tensors_equal(grad, expected_grad)


fn main():
    try:
        test_SIGMOID()
        test_RELU()
        test_TANH()
    except e:
        print("[ERROR] Error in forward mlops")
        print(e)
        return

    try:
        test_backward_SIGMOID()
        test_backward_RELU()
        test_backward_TANH()
    except e:
        print("[ERROR] Error in backward mlops")
        print(e)
        return
