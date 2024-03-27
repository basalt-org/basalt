from random import rand
from testing import assert_equal
from test_tensorutils import assert_tensors_equal
from python.python import Python

import basalt.nn as nn
from basalt import Tensor, TensorShape
from basalt import Graph, Symbol, OP
from basalt.autograd.attributes import Attribute, AttributeVector
from basalt.utils.tensorutils import fill
from basalt.autograd.ops.ops import backward_op

from test_utils_extras import to_numpy, to_tensor

alias dtype = DType.float32
alias nelts: Int = simdwidthof[dtype]()


# ------ Test Unary Ops ------
@value
struct torch_output_unary_op:
    var expected: Tensor[dtype]
    var grad_1: Tensor[dtype]


fn torch_unary_op(op: OP, input_1: Tensor, upper_grad: Tensor) -> torch_output_unary_op:
    try:
        var torch = Python.import_module("torch")
        var np = Python.import_module("numpy")

        var input_1 = torch.from_numpy(to_numpy(input_1)).requires_grad_(True)

        var expected: PythonObject

        if op == OP.SIGMOID:
            expected = torch.sigmoid(input_1)
        elif op == OP.RELU:
            expected = torch.relu(input_1)
        elif op == OP.TANH:
            expected = torch.tanh(input_1)
        else:
            print("Error: op not supported (returning the value input_1): ", op)
            expected = input_1

        # uppergrad & backwards
        var upper_grad = torch.from_numpy(to_numpy(upper_grad))
        _ = expected.backward(upper_grad)

        return torch_output_unary_op(
            to_tensor(expected.detach().numpy()),
            to_tensor(input_1.grad.numpy()),
        )

    except:
        print("Error importing torch")
        var d = Tensor[dtype](1)
        return torch_output_unary_op(d, d)


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

    var model = nn.Model[graph](inference_only=True)
    var res = model.inference(t1)[0]

    assert_tensors_equal(res, expected, "almost")


fn test_unary_op_backward[
    op: OP, t1_shape: TensorShape, ug_shape: TensorShape
](t1: Tensor[dtype], ug: Tensor[dtype], grad_1_expected: Tensor[dtype],) raises:
    var grad_1 = Tensor[dtype](t1_shape)
    backward_op[0, op, ug_shape, t1_shape, AttributeVector()](ug, t1, grad_1)
    assert_tensors_equal(grad_1, grad_1_expected, "almost")


fn test_SIGMOID() raises:
    alias t1_shape = TensorShape(37, 63, 107)
    alias ug_shape = TensorShape(37, 63, 107)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    rand(t1.data(), t1.num_elements())

    var ug = Tensor[dtype](ug_shape)
    rand(ug.data(), ug.num_elements())

    var expected_and_grad = torch_unary_op(OP.SIGMOID, t1, ug)


fn test_RELU() raises:
    alias t1_shape = TensorShape(37, 63, 107)
    alias ug_shape = TensorShape(37, 63, 107)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    rand(t1.data(), t1.num_elements())

    var ug = Tensor[dtype](ug_shape)
    rand(ug.data(), ug.num_elements())

    var expected_and_grad = torch_unary_op(OP.RELU, t1, ug)


fn test_TANH() raises:
    alias t1_shape = TensorShape(37, 63, 107)
    alias ug_shape = TensorShape(37, 63, 107)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    rand(t1.data(), t1.num_elements())

    var ug = Tensor[dtype](ug_shape)
    rand(ug.data(), ug.num_elements())

    var expected_and_grad = torch_unary_op(OP.TANH, t1, ug)


fn main():
    print("Running mlops (compare with torch) tests")
    try:
        test_SIGMOID()
        test_RELU()
        test_TANH()
    except e:
        print("[ERROR] Error in mlops (compare with torch)")
        print(e)
        return

    print("Finished mlops (compare with torch) tests")