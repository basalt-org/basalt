from random import rand
from testing import assert_equal
from math.limit import min_finite, max_finite
from test_tensorutils import assert_tensors_equal
from collections.optional import OptionalReg
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


fn torch_unary_op(
    op: OP,
    input_1: Tensor,
    upper_grad: Tensor,
    attrs: OptionalReg[AttributeVector] = None,
    attrs_tuple: OptionalReg[PythonObject] = None,
) -> torch_output_unary_op:
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
        elif op == OP.CLIP:
            var min_attr = attrs.value()["min"]
            var max_attr = attrs.value()["max"]
            var min_val = min_attr.value().to_scalar[
                dtype
            ]() if min_attr else min_finite[dtype]()
            var max_val = max_attr.value().to_scalar[
                dtype
            ]() if max_attr else max_finite[dtype]()
            expected = torch.clamp(input_1, min_val, max_val)
        elif op == OP.SQUEEZE:
            if attrs:
                var attrs = attrs.value()
                var dim = attrs["dim"]

                if dim:
                    expected = torch.squeeze(input_1, dim=dim.value().to_int())
                elif attrs_tuple:
                    expected = torch.squeeze(input_1, dim=attrs_tuple.value())
                else:
                    expected = torch.squeeze(input_1)
            else:
                expected = torch.squeeze(input_1)
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
    op: OP, t1_shape: TensorShape, attrs: OptionalReg[AttributeVector] = None
](t1: Tensor[dtype], expected: Tensor[dtype]) raises:
    fn create_graph() -> Graph:
        var g = Graph()
        var t1 = g.input(t1_shape)

        var res: Symbol
        if attrs:
            res = g.op(op, t1, attributes=attrs.value())
        else:
            res = g.op(op, t1)
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


fn test_unary_op_backward[
    op: OP, t1_shape: TensorShape, ug_shape: TensorShape, attrs: AttributeVector
](t1: Tensor[dtype], ug: Tensor[dtype], grad_1_expected: Tensor[dtype],) raises:
    var grad_1 = Tensor[dtype](t1_shape)
    backward_op[0, op, ug_shape, t1_shape, attrs](ug, t1, grad_1)
    assert_tensors_equal(grad_1, grad_1_expected, "almost")


fn test_SIGMOID() raises:
    alias t1_shape = TensorShape(37, 63, 107)
    alias ug_shape = TensorShape(37, 63, 107)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    rand(t1.data(), t1.num_elements())

    var ug = Tensor[dtype](ug_shape)
    rand(ug.data(), ug.num_elements())

    var expected_and_grad = torch_unary_op(OP.SIGMOID, t1, ug)
    test_unary_op[OP.SIGMOID, t1_shape](t1, expected_and_grad.expected)
    test_unary_op_backward[OP.SIGMOID, t1_shape, ug_shape](
        t1, ug, expected_and_grad.grad_1
    )


fn test_RELU() raises:
    alias t1_shape = TensorShape(37, 63, 107)
    alias ug_shape = TensorShape(37, 63, 107)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    rand(t1.data(), t1.num_elements())

    var ug = Tensor[dtype](ug_shape)
    rand(ug.data(), ug.num_elements())

    var expected_and_grad = torch_unary_op(OP.RELU, t1, ug)
    test_unary_op[OP.RELU, t1_shape](t1, expected_and_grad.expected)
    test_unary_op_backward[OP.RELU, t1_shape, ug_shape](
        t1, ug, expected_and_grad.grad_1
    )


fn test_TANH() raises:
    alias t1_shape = TensorShape(37, 63, 107)
    alias ug_shape = TensorShape(37, 63, 107)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    rand(t1.data(), t1.num_elements())

    var ug = Tensor[dtype](ug_shape)
    rand(ug.data(), ug.num_elements())

    var expected_and_grad = torch_unary_op(OP.TANH, t1, ug)
    test_unary_op[OP.TANH, t1_shape](t1, expected_and_grad.expected)
    test_unary_op_backward[OP.TANH, t1_shape, ug_shape](
        t1, ug, expected_and_grad.grad_1
    )


fn test_CLIP() raises:
    alias t1_shape = TensorShape(37, 63, 107)
    alias ug_shape = TensorShape(37, 63, 107)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    rand(t1.data(), t1.num_elements())

    var ug = Tensor[dtype](ug_shape)
    rand(ug.data(), ug.num_elements())

    # No clipping
    var expected_and_grad = torch_unary_op(OP.CLIP, t1, ug)
    test_unary_op[OP.CLIP, t1_shape](t1, expected_and_grad.expected)
    test_unary_op_backward[OP.CLIP, t1_shape, ug_shape](
        t1, ug, expected_and_grad.grad_1
    )

    # Clip with min
    alias min_attr = Attribute("min", 0.3333)
    expected_and_grad = torch_unary_op(OP.CLIP, t1, ug, AttributeVector(min_attr))
    test_unary_op[OP.CLIP, t1_shape, AttributeVector(min_attr)](
        t1, expected_and_grad.expected
    )
    test_unary_op_backward[OP.CLIP, t1_shape, ug_shape, AttributeVector(min_attr)](
        t1, ug, expected_and_grad.grad_1
    )

    # Clip with max
    alias max_attr = Attribute("max", 0.6666)
    expected_and_grad = torch_unary_op(OP.CLIP, t1, ug, AttributeVector(max_attr))
    test_unary_op[OP.CLIP, t1_shape, AttributeVector(max_attr)](
        t1, expected_and_grad.expected
    )
    test_unary_op_backward[OP.CLIP, t1_shape, ug_shape, AttributeVector(max_attr)](
        t1, ug, expected_and_grad.grad_1
    )

    # Clip with min and max
    expected_and_grad = torch_unary_op(
        OP.CLIP, t1, ug, AttributeVector(min_attr, max_attr)
    )
    test_unary_op[OP.CLIP, t1_shape, AttributeVector(min_attr, max_attr)](
        t1, expected_and_grad.expected
    )
    test_unary_op_backward[
        OP.CLIP, t1_shape, ug_shape, AttributeVector(min_attr, max_attr)
    ](t1, ug, expected_and_grad.grad_1)


fn test_SQUEEZE() raises:
    alias t1_shape = TensorShape(20, 1, 28, 1)
    alias ug_shape = TensorShape(20, 28)
    var t1 = Tensor[dtype](t1_shape)
    rand(t1.data(), t1.num_elements())

    var ug = Tensor[dtype](ug_shape)
    rand(ug.data(), ug.num_elements())

    var expected_and_grad = torch_unary_op(OP.SQUEEZE, t1, ug)
    test_unary_op[OP.SQUEEZE, t1_shape](t1, expected_and_grad.expected)
    test_unary_op_backward[OP.SQUEEZE, t1_shape, ug_shape](
        t1, ug, expected_and_grad.grad_1
    )

    # Squeeze with one dim
    alias ug_shape_1 = TensorShape(20, 1, 28)
    ug = Tensor[dtype](ug_shape_1)
    rand(ug.data(), ug.num_elements())

    alias dim = Attribute("dim", 3)

    expected_and_grad = torch_unary_op(OP.SQUEEZE, t1, ug, AttributeVector(dim))
    test_unary_op[OP.SQUEEZE, t1_shape, AttributeVector(dim)](
        t1, expected_and_grad.expected
    )
    test_unary_op_backward[OP.SQUEEZE, t1_shape, ug_shape_1, AttributeVector(dim)](
        t1, ug, expected_and_grad.grad_1
    )

    alias ug_shape_2 = TensorShape(20, 28, 1)
    ug = Tensor[dtype](ug_shape_2)
    rand(ug.data(), ug.num_elements())

    alias dim_2 = Attribute("dim", 1)

    expected_and_grad = torch_unary_op(OP.SQUEEZE, t1, ug, AttributeVector(dim_2))
    test_unary_op[OP.SQUEEZE, t1_shape, AttributeVector(dim_2)](
        t1, expected_and_grad.expected
    )
    test_unary_op_backward[OP.SQUEEZE, t1_shape, ug_shape_2, AttributeVector(dim_2)](
        t1, ug, expected_and_grad.grad_1
    )

    # Squeeze with multiple dims
    ug = Tensor[dtype](ug_shape)
    rand(ug.data(), ug.num_elements())

    alias dims_shape = TensorShape(1, 3)
    alias dims_tuple = (dims_shape[0], dims_shape[1])

    alias dims = Attribute("dims", dims_shape)

    expected_and_grad = torch_unary_op(OP.SQUEEZE, t1, ug, attrs_tuple=OptionalReg[PythonObject](dims_tuple))
    test_unary_op[OP.SQUEEZE, t1_shape, AttributeVector(dims)](
        t1, expected_and_grad.expected
    )
    test_unary_op_backward[OP.SQUEEZE, t1_shape, ug_shape, AttributeVector(dims)](
        t1, ug, expected_and_grad.grad_1
    )


fn main():
    print("Running mlops (compare with torch) tests")
    try:
        test_SIGMOID()
        test_RELU()
        test_TANH()
        test_CLIP()
        test_SQUEEZE()
    except e:
        print("[ERROR] Error in mlops (compare with torch)")
        print(e)
        return

    print("Finished mlops (compare with torch) tests")
