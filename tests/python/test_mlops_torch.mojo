from random import rand
from math.limit import min_finite, max_finite
from collections.optional import OptionalReg, Optional
from python.python import Python
from python.object import PythonObject

from basalt import dtype, nelts
from basalt.autograd import OP
from basalt.autograd.attributes import Attribute, AttributeVector
from basalt.nn import Tensor, TensorShape

from tests import (
    assert_tensors_equal,
    to_numpy,
    to_tensor,
    test_unary_op,
    test_binary_op,
    test_unary_op_backward,
    test_binary_op_backward,
)


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
    attrs_tuple: Optional[PythonObject] = None,
) -> torch_output_unary_op:
    try:
        var torch = Python.import_module("torch")
        var np = Python.import_module("numpy")
        var py = Python.import_module("builtins")

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
                var dim = attrs["dims"]

                if dim:
                    expected = torch.squeeze(input_1, dim=dim.value().to_shape()[0])
                else:
                    expected = torch.squeeze(input_1)
            elif attrs_tuple:
                expected = torch.squeeze(input_1, dim=attrs_tuple.value()[])
            else:
                expected = torch.squeeze(input_1)
        elif op == OP.UNSQUEEZE:
            if attrs:
                var attrs = attrs.value()
                var dim = attrs["dims"]

                if dim:
                    expected = torch.unsqueeze(input_1, dim=dim.value().to_shape()[0])
                else:
                    expected = torch.unsqueeze(input_1, 0)
            elif attrs_tuple:
                expected = torch.reshape(input_1, attrs_tuple.value()[])
            else:
                expected = torch.unsqueeze(input_1, 0)
        elif op == OP.SLICE:
            var attrs = attrs_tuple.value()[]
            var indices = torch.tensor(py.range(attrs[0], attrs[1], attrs[2]))
            var dim = attrs[3]
            expected = torch.index_select(input_1, dim, indices)
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

    alias dim = Attribute("dims", TensorShape(3))

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

    alias dim_2 = Attribute("dims", TensorShape(1))

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

    expected_and_grad = torch_unary_op(
        OP.SQUEEZE, t1, ug, attrs_tuple=PythonObject(dims_tuple)
    )
    test_unary_op[OP.SQUEEZE, t1_shape, AttributeVector(dims)](
        t1, expected_and_grad.expected
    )
    test_unary_op_backward[OP.SQUEEZE, t1_shape, ug_shape, AttributeVector(dims)](
        t1, ug, expected_and_grad.grad_1
    )


fn test_UNSQUEEZE() raises:
    alias t1_shape = TensorShape(20, 28)
    alias ug_shape = TensorShape(20, 1, 28)
    var t1 = Tensor[dtype](t1_shape)
    rand(t1.data(), t1.num_elements())

    var ug = Tensor[dtype](ug_shape)
    rand(ug.data(), ug.num_elements())

    alias dim = Attribute("dims", TensorShape(1))

    var expected_and_grad = torch_unary_op(OP.UNSQUEEZE, t1, ug, AttributeVector(dim))
    test_unary_op[OP.UNSQUEEZE, t1_shape, AttributeVector(dim)](
        t1, expected_and_grad.expected
    )
    test_unary_op_backward[OP.UNSQUEEZE, t1_shape, ug_shape, AttributeVector(dim)](
        t1, ug, expected_and_grad.grad_1
    )

    # Unsqueeze with multiple dims
    alias ug_shape_2 = TensorShape(20, 1, 28, 1)
    ug = Tensor[dtype](ug_shape_2)

    alias dims_shape = TensorShape(1, 3)
    alias dims_tuple = (20, 1, 28, 1)

    alias dims = Attribute("dims", dims_shape)

    expected_and_grad = torch_unary_op(
        OP.UNSQUEEZE, t1, ug, attrs_tuple=PythonObject(dims_tuple)
    )
    test_unary_op[OP.UNSQUEEZE, t1_shape, AttributeVector(dims)](
        t1, expected_and_grad.expected
    )
    test_unary_op_backward[OP.UNSQUEEZE, t1_shape, ug_shape_2, AttributeVector(dims)](
        t1, ug, expected_and_grad.grad_1
    )


fn test_SLICE() raises:
    alias t1_shape = TensorShape(37, 63, 107)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    rand(t1.data(), t1.num_elements())

    # dim = 0
    alias slice_0 = Slice(5, 24, 3)
    alias attrs_0 = AttributeVector(
        Attribute("slice", StaticIntTuple[3](slice_0.start, slice_0.end, slice_0.step)),
        Attribute("dim", 0)
    )

    alias ug_shape = TensorShape(7, 63, 107)
    var ug = Tensor[dtype](ug_shape)
    rand(ug.data(), ug.num_elements())

    var attrs_tuple_0 = PythonObject((slice_0.start, slice_0.end, slice_0.step, 0))
    var expected_and_grad = torch_unary_op(OP.SLICE, t1, ug, attrs_tuple=attrs_tuple_0)
    test_unary_op[OP.SLICE, t1_shape, attrs_0](t1, expected_and_grad.expected)
    test_unary_op_backward[OP.SLICE, t1_shape, ug_shape, attrs_0](t1, ug, expected_and_grad.grad_1)

    # dim = 1
    alias slice_1 = Slice(10, 50, 5)
    alias attrs_1 = AttributeVector(
        Attribute("slice", StaticIntTuple[3](slice_1.start, slice_1.end, slice_1.step)),
        Attribute("dim", 1)
    )

    alias ug_shape_1 = TensorShape(37, 8, 107)
    ug = Tensor[dtype](ug_shape_1)
    rand(ug.data(), ug.num_elements())

    var attrs_tuple_1 = PythonObject((slice_1.start, slice_1.end, slice_1.step, 1))
    expected_and_grad = torch_unary_op(OP.SLICE, t1, ug, attrs_tuple=attrs_tuple_1)
    test_unary_op[OP.SLICE, t1_shape, attrs_1](t1, expected_and_grad.expected)
    test_unary_op_backward[OP.SLICE, t1_shape, ug_shape_1, attrs_1](t1, ug, expected_and_grad.grad_1)

    # dim = 2
    alias slice_2 = Slice(99, 33, -7)
    alias attrs_2 = AttributeVector(
        Attribute("slice", StaticIntTuple[3](slice_2.start, slice_2.end, slice_2.step)),
        Attribute("dim", 2)
    )

    alias ug_shape_2 = TensorShape(37, 63, 10)
    ug = Tensor[dtype](ug_shape_2)
    rand(ug.data(), ug.num_elements())

    var attrs_tuple_2 = PythonObject((slice_2.start, slice_2.end, slice_2.step, 2))
    expected_and_grad = torch_unary_op(OP.SLICE, t1, ug, attrs_tuple=attrs_tuple_2)
    test_unary_op[OP.SLICE, t1_shape, attrs_2](t1, expected_and_grad.expected)
    test_unary_op_backward[OP.SLICE, t1_shape, ug_shape_2, attrs_2](t1, ug, expected_and_grad.grad_1)


fn main():
    print("Running mlops (compare with torch) tests")
    try:
        test_SIGMOID()
        test_RELU()
        test_TANH()
        test_CLIP()
        test_SQUEEZE()
        test_UNSQUEEZE()
        test_SLICE()
    except e:
        print("[ERROR] Error in mlops (compare with torch)")
        print(e)
        return

    print("Finished mlops (compare with torch) tests")
