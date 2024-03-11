from tensor import Tensor, TensorShape
from testing import assert_equal, assert_almost_equal

from test_tensorutils import assert_tensors_equal

import dainemo.nn as nn
from dainemo import Graph, Symbol, OP
from dainemo.utils.tensorutils import fill


alias dtype = DType.float32
alias nelts: Int = simdwidthof[dtype]()


# <------------SOFTMAX------------>
fn test_SOFTMAX() raises:
    alias x_shape = TensorShape(2, 3, 2)

    fn create_graph(axis: Int) -> Graph:
        var g = Graph()

        var x = g.input(x_shape)

        var softmax = nn.Softmax(g, x, axis)

        g.out(softmax)

        return g ^

    var x = Tensor[dtype](x_shape)

    fill[dtype, nelts](x, 4)

    # Test axis 0
    alias graph = create_graph(0)

    var model = nn.Model[graph](inference_only=True)
    var res = model.inference(x)[0]

    var expected = Tensor[dtype](x_shape)

    fill[dtype, nelts](expected, 0.5)

    assert_tensors_equal(res, expected, "almost")

    assert_equal(len(graph.nodes), 5) # max_values, exp_values, sum_values, diff_max_values, result_div

    # Test axis 1
    alias graph_2 = create_graph(1)

    var model_2 = nn.Model[graph_2](inference_only=True)
    res = model_2.inference(x)[0]

    expected = Tensor[dtype](x_shape)

    fill[dtype, nelts](expected, 1.0 / 3.0)

    assert_tensors_equal(res, expected, "almost")

    # Test axis 2
    alias graph_3 = create_graph(2)

    var model_3 = nn.Model[graph_3](inference_only=True)
    res = model_3.inference(x)[0]

    expected = Tensor[dtype](x_shape)

    fill[dtype, nelts](expected, 0.5)

    assert_tensors_equal(res, expected, "almost")


# <------------LOGSOFTMAX------------>
fn test_LOGSOFTMAX() raises:
    alias x_shape = TensorShape(2, 3, 2)

    fn create_graph(axis: Int) -> Graph:
        var g = Graph()

        var x = g.input(x_shape)

        var logsoftmax = nn.LogSoftmax(g, x, axis)

        g.out(logsoftmax)

        return g ^
    
    var x = Tensor[dtype](x_shape)

    fill[dtype, nelts](x, 4)

    # Test axis 0
    alias graph = create_graph(0)

    var model = nn.Model[graph](inference_only=True)
    var res = model.inference(x)[0]

    var expected = Tensor[dtype](x_shape)

    fill[dtype, nelts](expected, -0.69314718)

    assert_tensors_equal(res, expected, "almost")

    assert_equal(len(graph.nodes), 6) # max_values, exp_values, sum_values, diff_max_values, log_values, result_sub

    # Test axis 1
    alias graph_2 = create_graph(1)

    var model_2 = nn.Model[graph_2](inference_only=True)
    res = model_2.inference(x)[0]

    expected = Tensor[dtype](x_shape)

    fill[dtype, nelts](expected, -1.09861231)

    assert_tensors_equal(res, expected, "almost")

    # Test axis 2
    alias graph_3 = create_graph(2)

    var model_3 = nn.Model[graph_3](inference_only=True)
    res = model_3.inference(x)[0]

    expected = Tensor[dtype](x_shape)

    fill[dtype, nelts](expected, -0.69314718)

    assert_tensors_equal(res, expected, "almost")


# <------------RELU------------>
fn test_RELU() raises:
    alias x_shape = TensorShape(2, 3)

    fn create_graph() -> Graph:
        var g = Graph()

        var x = g.input(x_shape)

        var relu = nn.ReLU(g, x)

        g.out(relu)

        return g ^
    
    var x = Tensor[dtype](x_shape)
    for i in range(3):
        x[i] = 3
    for i in range(3, 6):
        x[i] = -3

    alias graph = create_graph()

    var model = nn.Model[graph](inference_only=True)
    var res = model.inference(x)[0]

    var expected = Tensor[dtype](x_shape)
    for i in range(3):
        expected[i] = 3
    for i in range(3, 6):
        expected[i] = 0

    assert_tensors_equal(res, expected, "almost")
    assert_equal(len(graph.nodes), 1)


# <------------SIGMOID------------>
fn test_SIGMOID() raises:
    alias x_shape = TensorShape(2, 3)

    fn create_graph() -> Graph:
        var g = Graph()

        var x = g.input(x_shape)

        var sigmoid = nn.Sigmoid(g, x)

        g.out(sigmoid)

        return g ^

    var x = Tensor[dtype](x_shape)
    fill[dtype, nelts](x, 0)

    alias graph = create_graph()

    var model = nn.Model[graph](inference_only=True)
    var res = model.inference(x)[0]

    var expected = Tensor[dtype](x_shape)
    fill[dtype, nelts](expected, 0.5)

    assert_tensors_equal(res, expected, "almost")
    assert_equal(len(graph.nodes), 1)


# <------------TANH------------>
fn test_TANH() raises:
    alias x_shape = TensorShape(2, 3)

    fn create_graph() -> Graph:
        var g = Graph()

        var x = g.input(x_shape)

        var tanh = nn.Tanh(g, x)

        g.out(tanh)

        return g ^

    var x = Tensor[dtype](x_shape)
    fill[dtype, nelts](x, 0)

    alias graph = create_graph()

    var model = nn.Model[graph](inference_only=True)
    var res = model.inference(x)[0]

    var expected = Tensor[dtype](x_shape)
    fill[dtype, nelts](expected, 0.0)

    assert_tensors_equal(res, expected, "almost")
    assert_equal(len(graph.nodes), 1)


fn main():
    try:
        test_SOFTMAX()
        test_LOGSOFTMAX()
        test_RELU()
        test_SIGMOID()
        test_TANH()
    except e:
        print("[ERROR] Error in activations")
        print(e)
