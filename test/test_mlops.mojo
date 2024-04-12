from random import rand
from testing import assert_equal
from test_tensorutils import assert_tensors_equal
from collections.optional import OptionalReg


import basalt.nn as nn
from basalt import Tensor, TensorShape
from basalt import Graph, Symbol, OP
from basalt.autograd.ops.mlops import SIGMOID, RELU, TANH, CLIP
from basalt.utils.tensorutils import fill
from basalt.autograd.attributes import AttributeVector, Attribute


alias dtype = DType.float32
alias nelts: Int = simdwidthof[dtype]()


# ------ Test Unary Ops ------
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
    assert_tensors_equal(res, expected)


fn test_SIGMOID() raises:
    alias t1_shape = TensorShape(2, 3)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)  # filled with zeroes

    var expected = Tensor[dtype](2, 3)
    fill(expected, 0.5)

    test_unary_op[OP.SIGMOID, t1_shape](t1, expected)


fn test_backward_SIGMOID() raises:
    alias t1_shape = TensorShape(2, 3)
    alias ug_shape = TensorShape(2, 3)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)  # filled with zeroes
    var ug: Tensor[dtype] = Tensor[dtype](ug_shape)
    fill(ug, 5.0)

    var expected_grad = Tensor[dtype](2, 3)
    fill(
        expected_grad, 5.0 * 0.25
    )  # 0.25 = d(sigmoid(0))/dx = sigmoid(0) * (1 - sigmoid(0))

    var grad = SIGMOID.backward[ug_shape, t1_shape](ug, t1)
    assert_tensors_equal(grad, expected_grad)


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
    fill(ug, 5.0)

    var expected_grad = Tensor[dtype](2, 3)
    for i in range(3):
        expected_grad[i] = 1 * 5.0  # 1 = d(relu(3))/dx
    for i in range(3, 6):
        expected_grad[i] = 0 * 5.0  # 0 = d(relu(-3))/dx

    var grad = RELU.backward[ug_shape, t1_shape](ug, t1)
    assert_tensors_equal(grad, expected_grad)


fn test_TANH() raises:
    alias t1_shape = TensorShape(2, 3)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)  # filled with zeroes

    var expected = Tensor[dtype](2, 3)
    fill(expected, 0.0)

    test_unary_op[OP.TANH, t1_shape](t1, expected)


fn test_backward_TANH() raises:
    alias t1_shape = TensorShape(2, 3)
    alias ug_shape = TensorShape(2, 3)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)  # filled with zeroes
    var ug: Tensor[dtype] = Tensor[dtype](ug_shape)
    fill(ug, 5.0)

    var expected_grad = Tensor[dtype](2, 3)
    fill(expected_grad, 5.0 * 1.0)  # 1.0 = d(tanh(0))/dx = 1 - tanh(0)^2

    var grad = TANH.backward[ug_shape, t1_shape](ug, t1)
    assert_tensors_equal(grad, expected_grad)


fn test_CLIP() raises:
    alias t1_shape = TensorShape(2, 3)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    for i in range(6):
        t1[i] = i - 3

    # Clip without min and max
    var expected_no = t1
    test_unary_op[OP.CLIP, t1_shape](t1, expected_no)

    # Clip with min
    alias min_attr = Attribute("min", -1.1)
    var expected_min = Tensor[dtype](2, 3)
    for i in range(6):
        var val = Scalar[dtype](i - 3)
        expected_min[i] = val if (val > -1.1) else -1.1
    test_unary_op[OP.CLIP, t1_shape, AttributeVector(min_attr)](t1, expected_min)

    # Clip with max
    alias max_attr = Attribute("max", 1.1)
    var expected_max = Tensor[dtype](2, 3)
    for i in range(6):
        var val = Scalar[dtype](i - 3)
        expected_max[i] = val if (val < 1.1) else 1.1
    test_unary_op[OP.CLIP, t1_shape, AttributeVector(max_attr)](t1, expected_max)

    # Clip with min and max
    var expected = Tensor[dtype](2, 3)
    for i in range(6):
        var val = Scalar[dtype](i - 3)
        if val < -1.1:
            expected[i] = -1.1
        elif val > 1.1:
            expected[i] = 1.1
        else:
            expected[i] = val
    test_unary_op[OP.CLIP, t1_shape, AttributeVector(min_attr, max_attr)](t1, expected)


fn test_backward_CLIP() raises:
    alias t1_shape = TensorShape(2, 3)
    alias ug_shape = TensorShape(2, 3)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    for i in range(6):
        t1[i] = i - 3
    var ug: Tensor[dtype] = Tensor[dtype](ug_shape)
    fill(ug, 5.0)

    # Clip without min and max
    var expected_no = ug
    var grad_no = CLIP.backward[ug_shape, t1_shape](ug, t1)
    assert_tensors_equal(grad_no, expected_no)

    # Clip with min
    alias min_attr = Attribute("min", -1.1)
    var expected_min = Tensor[dtype](2, 3)
    for i in range(6):
        var val = Scalar[dtype](i - 3)
        expected_min[i] = 5.0 if (val > -1.1) else 0.0
    var grad_min = CLIP.backward[ug_shape, t1_shape, AttributeVector(min_attr)](ug, t1)
    assert_tensors_equal(grad_min, expected_min)

    # Clip with max
    alias max_attr = Attribute("max", 1.1)
    var expected_max = Tensor[dtype](2, 3)
    for i in range(6):
        var val = Scalar[dtype](i - 3)
        expected_max[i] = 5.0 if (val < 1.1) else 0.0
    var grad_max = CLIP.backward[ug_shape, t1_shape, AttributeVector(max_attr)](ug, t1)
    assert_tensors_equal(grad_max, expected_max)

    # Clip with min and max
    var expected = Tensor[dtype](2, 3)
    for i in range(6):
        var val = Scalar[dtype](i - 3)
        if val < -1.1 or val > 1.1:
            expected[i] = 0.0
        else:
            expected[i] = 5.0
    var grad = CLIP.backward[ug_shape, t1_shape, AttributeVector(min_attr, max_attr)](
        ug, t1
    )
    assert_tensors_equal(grad, expected)


fn main():
    try:
        test_SIGMOID()
        test_RELU()
        test_TANH()
        test_CLIP()
    except e:
        print("[ERROR] Error in forward mlops")
        print(e)
        return

    try:
        test_backward_SIGMOID()
        test_backward_RELU()
        test_backward_TANH()
        test_backward_CLIP()
    except e:
        print("[ERROR] Error in backward mlops")
        print(e)
        return
