from testing import assert_equal

from basalt import dtype
from basalt.nn import (
    Tensor,
    TensorShape,
    Model,
    Softmax,
    LogSoftmax,
    ReLU,
    LeakyReLU,
    Sigmoid,
    Tanh,
    Hardtanh,
    Threshold,
)
from basalt.autograd import Graph, Symbol
from basalt.utils.tensorutils import fill

from tests import assert_tensors_equal


alias Activation = fn (inout g: Graph, input: Symbol) -> Symbol
alias AxisActivation = fn (inout g: Graph, input: Symbol, axis: Int) -> Symbol
alias LeakyReLUActivation = fn (
    inout g: Graph, input: Symbol, negative_slope: Scalar[dtype]
) -> Symbol
alias ThresholdActivation = fn (
    inout g: Graph,
    input: Symbol,
    threshold: Scalar[dtype],
    value: Scalar[dtype],
) -> Symbol


fn create_graph[
    shape: TensorShape,
    func: AxisActivation,
    axis: Int,
]() -> Graph:
    var g = Graph()
    var x = g.input(shape)
    var activation = func(g, x, axis)
    g.out(activation)
    return g^


fn create_graph[
    shape: TensorShape,
    func: LeakyReLUActivation,
    negative_slope: Scalar[dtype],
]() -> Graph:
    var g = Graph()
    var x = g.input(shape)
    var activation = func(g, x, negative_slope)
    g.out(activation)
    return g^


fn create_graph[shape: TensorShape, func: Activation]() -> Graph:
    var g = Graph()
    var x = g.input(shape)
    var activation = func(g, x)
    g.out(activation)
    return g^


fn create_graph[
    shape: TensorShape,
    func: ThresholdActivation,
    threshold: Scalar[dtype],
    value: Scalar[dtype],
]() -> Graph:
    var g = Graph()
    var x = g.input(shape)
    var activation = func(g, x, threshold, value)
    g.out(activation)
    return g^


fn test_graph[
    shape: TensorShape,
    func: ThresholdActivation,
    nodes: Int,
    threshold: Scalar[dtype],
    value: Scalar[dtype],
](input: Tensor[dtype], expected: Tensor[dtype]) raises:
    alias graph = create_graph[shape, func, threshold, value]()

    var model = Model[graph](inference_only=True)
    var res = model.inference(input)[0]

    assert_tensors_equal["almost"](res, expected)
    assert_equal(len(graph.nodes), nodes)


fn test_graph[
    shape: TensorShape,
    func: AxisActivation,
    nodes: Int,
    axis: Int,
](input: Tensor[dtype], expected: Tensor[dtype]) raises:
    alias graph = create_graph[shape, func, axis]()

    var model = Model[graph](inference_only=True)
    var res = model.inference(input)[0]

    assert_tensors_equal["almost"](res, expected)
    assert_equal(len(graph.nodes), nodes)


fn test_graph[
    shape: TensorShape,
    func: LeakyReLUActivation,
    nodes: Int,
    negative_slope: Scalar[dtype],
](input: Tensor[dtype], expected: Tensor[dtype]) raises:
    alias graph = create_graph[shape, func, negative_slope]()

    var model = Model[graph](inference_only=True)
    var res = model.inference(input)[0]

    assert_tensors_equal["almost"](res, expected)
    assert_equal(len(graph.nodes), nodes)


# TODO: All these overloads feel redundant. Find a way to condense them
fn test_graph[
    shape: TensorShape,
    func: Activation,
    nodes: Int,
](input: Tensor[dtype], expected: Tensor[dtype]) raises:
    alias graph = create_graph[shape, func]()

    var model = Model[graph](inference_only=True)
    var res = model.inference(input)[0]

    assert_tensors_equal["almost", "Tensor equality failed"](res, expected)
    assert_equal(len(graph.nodes), nodes, "Node count failed")


fn test_THRESHOLD() raises:
    alias shape = TensorShape(2, 3)
    alias nodes = 1

    alias THRESHOLD = 3
    alias VALUE = 2

    var input = Tensor[dtype](shape)

    for i in range(6):
        input[i] = i

    var expected = Tensor[dtype](shape)

    for i in range(6):
        expected[i] = i if i > THRESHOLD else VALUE

    test_graph[shape, Threshold, nodes, THRESHOLD, VALUE](input, expected)


fn test_SOFTMAX() raises:
    alias shape = TensorShape(2, 3, 2)
    alias nodes = 5

    var input = Tensor[dtype](shape)
    fill(input, 4)

    var expected = Tensor[dtype](shape)

    fill(expected, 0.5)
    test_graph[shape, Softmax, nodes, 0](input, expected)

    fill(expected, 1.0 / 3.0)
    test_graph[shape, Softmax, nodes, 1](input, expected)

    fill(expected, 0.5)
    test_graph[shape, Softmax, nodes, 2](input, expected)


fn test_LOGSOFTMAX() raises:
    alias shape = TensorShape(2, 3, 2)
    alias nodes = 6

    var input = Tensor[dtype](shape)
    fill(input, 4)

    var expected = Tensor[dtype](shape)

    fill(expected, -0.69314718)
    test_graph[shape, LogSoftmax, nodes, 0](input, expected)

    fill(expected, -1.09861231)
    test_graph[shape, LogSoftmax, nodes, 1](input, expected)

    fill(expected, -0.69314718)
    test_graph[shape, LogSoftmax, nodes, 2](input, expected)


fn test_RELU() raises:
    alias shape = TensorShape(2, 3)
    alias nodes = 1

    var input = Tensor[dtype](shape)

    for i in range(6):
        input[i] = 3 if i < 3 else -3

    var expected = Tensor[dtype](shape)

    for i in range(6):
        expected[i] = 3 if i < 3 else 0

    test_graph[shape, ReLU, nodes](input, expected)


fn test_LEAKYRELU() raises:
    alias negative_slope = 0.1

    alias shape = TensorShape(2, 3)
    alias nodes = 1

    var input = Tensor[dtype](shape)

    for i in range(6):
        input[i] = i - 3

    var expected = Tensor[dtype](shape)

    for i in range(6):
        expected[i] = i - 3 if i - 3 > 0 else negative_slope * (i - 3)

    test_graph[shape, LeakyReLU, nodes, negative_slope](input, expected)


fn test_SIGMOID() raises:
    alias shape = TensorShape(2, 3)
    alias nodes = 1

    var input = Tensor[dtype](shape)
    fill(input, 0)

    var expected = Tensor[dtype](shape)

    fill(expected, 0.5)
    test_graph[shape, Sigmoid, nodes](input, expected)


fn test_TANH() raises:
    alias shape = TensorShape(2, 3)
    alias nodes = 1

    var input = Tensor[dtype](shape)
    fill(input, 0)

    var expected = Tensor[dtype](shape)

    fill(expected, 0.0)
    test_graph[shape, Tanh, nodes](input, expected)


fn test_HARDTANH() raises:
    alias shape = TensorShape(3, 3)
    alias nodes = 1

    alias MIN_VAL = -2
    alias MAX_VAL = 2

    var input = Tensor[dtype](shape)

    for i in range(9):
        input[i] = i - 4

    var expected = Tensor[dtype](shape)

    for j in range(0, 9):
        var i = j - 4
        if i < MIN_VAL:
            expected[j] = MIN_VAL

        elif i > MAX_VAL:
            expected[j] = MAX_VAL

        else:
            expected[j] = i

    test_graph[shape, Hardtanh, nodes, MIN_VAL, MAX_VAL](input, expected)


fn main():
    try:
        test_THRESHOLD()
        test_SOFTMAX()
        test_LOGSOFTMAX()
        test_RELU()
        test_SIGMOID()
        test_TANH()
        test_HARDTANH()
    except e:
        print("[ERROR] Error in activations")
        print(e)
