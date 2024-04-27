from math import log, exp
from testing import assert_true, assert_equal
from test_tensorutils import assert_tensors_equal
from test_utils_extras import (
    test_unary_op_backward,
    test_binary_op_backward,
    test_ternary_op_backward,
)

from basalt import Tensor, TensorShape, OP
from basalt.utils.tensorutils import fill, tsum
from basalt.autograd.attributes import Attribute, AttributeVector

alias dtype = DType.float32
alias nelts: Int = simdwidthof[dtype]()


fn test_ADD() raises:
    alias t1_shape = TensorShape(2, 3)
    alias t2_shape = TensorShape(2, 3)
    alias ug_shape = TensorShape(2, 3)
    var t1 = Tensor[dtype](t1_shape)
    var t2 = Tensor[dtype](t2_shape)
    var ug = Tensor[dtype](ug_shape)
    fill(t1, 1.0)
    fill(t2, 2.0)
    fill(ug, 1.0)

    var expected_grad = Tensor[dtype](ug_shape)
    fill(expected_grad, 1.0)
    test_binary_op_backward[OP.ADD, t1_shape, t2_shape, ug_shape](
        t1, t2, ug, expected_grad, expected_grad
    )


fn test_SUB() raises:
    alias t1_shape = TensorShape(2, 3)
    alias t2_shape = TensorShape(2, 3)
    alias ug_shape = TensorShape(2, 3)
    var t1 = Tensor[dtype](t1_shape)
    var t2 = Tensor[dtype](t2_shape)
    var ug = Tensor[dtype](ug_shape)
    fill(t1, 2.0)
    fill(t2, 1.0)
    fill(ug, 1.0)

    var expected_grad1 = Tensor[dtype](t1_shape)
    var expected_grad2 = Tensor[dtype](t2_shape)
    fill(expected_grad1, 1.0)
    fill(expected_grad2, -1.0)
    test_binary_op_backward[OP.SUB, t1_shape, t2_shape, ug_shape](
        t1, t2, ug, expected_grad1, expected_grad2
    )


fn test_MUL() raises:
    alias t1_shape = TensorShape(2, 3)
    alias t2_shape = TensorShape(2, 3)
    alias ug_shape = TensorShape(2, 3)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    var t2: Tensor[dtype] = Tensor[dtype](t2_shape)
    var ug: Tensor[dtype] = Tensor[dtype](ug_shape)
    fill(t1, 1.0)
    fill(t2, 2.0)
    fill(ug, 1.0)

    var expected_grad1 = Tensor[dtype](t1_shape)
    var expected_grad2 = Tensor[dtype](t2_shape)
    fill(expected_grad1, 2.0)
    fill(expected_grad2, 1.0)
    test_binary_op_backward[OP.MUL, t1_shape, t2_shape, ug_shape](
        t1, t2, ug, expected_grad1, expected_grad2
    )


fn test_DIV() raises:
    alias t1_shape = TensorShape(2, 3)
    alias t2_shape = TensorShape(2, 3)
    alias ug_shape = TensorShape(2, 3)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    var t2: Tensor[dtype] = Tensor[dtype](t2_shape)
    var ug: Tensor[dtype] = Tensor[dtype](ug_shape)
    fill(t1, 1.0)
    fill(t2, 2.0)
    fill(ug, 1.0)

    var expected_grad1 = Tensor[dtype](t1_shape)
    var expected_grad2 = Tensor[dtype](t2_shape)
    fill(expected_grad1, 1.0 / 2.0)
    fill[dtype](expected_grad2, -1.0 / (2.0**2))
    test_binary_op_backward[OP.DIV, t1_shape, t2_shape, ug_shape](
        t1, t2, ug, expected_grad1, expected_grad2
    )


fn test_DOT() raises:
    alias t1_shape = TensorShape(2, 3)
    alias t2_shape = TensorShape(3, 2)
    alias ug_shape = TensorShape(2, 2)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    var t2: Tensor[dtype] = Tensor[dtype](t2_shape)
    var ug: Tensor[dtype] = Tensor[dtype](ug_shape)
    fill(t1, 1.0)
    fill(t2, 2.0)
    fill(ug, 1.0)

    var expected_grad1 = Tensor[dtype](t1_shape)
    var expected_grad2 = Tensor[dtype](t2_shape)
    fill(expected_grad1, 4.0)
    fill(expected_grad2, 2.0)
    test_binary_op_backward[OP.DOT, t1_shape, t2_shape, ug_shape](
        t1, t2, ug, expected_grad1, expected_grad2
    )


fn test_EXP() raises:
    alias t1_shape = TensorShape(2, 3)
    alias ug_shape = TensorShape(2, 3)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    var ug: Tensor[dtype] = Tensor[dtype](ug_shape)
    fill(t1, 2.0)
    fill(ug, 5.0)

    var expected_grad1 = Tensor[dtype](t1_shape)
    fill(expected_grad1, 5.0 * exp[dtype, 1](2.0))
    test_unary_op_backward[OP.EXP, t1_shape, ug_shape](t1, ug, expected_grad1)


fn test_LOG() raises:
    alias t1_shape = TensorShape(2, 3)
    alias ug_shape = TensorShape(2, 3)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    var ug: Tensor[dtype] = Tensor[dtype](ug_shape)
    fill(t1, 2.0)
    fill(ug, 5.0)

    var expected_grad1 = Tensor[dtype](t1_shape)
    fill(expected_grad1, 5.0 / 2.0)
    test_unary_op_backward[OP.LOG, t1_shape, ug_shape](t1, ug, expected_grad1)


fn test_POW() raises:
    alias t1_shape = TensorShape(2, 3)
    alias t2_shape = TensorShape(1)
    alias ug_shape = TensorShape(2, 3)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    var t2: Tensor[dtype] = Tensor[dtype](t2_shape)
    var ug: Tensor[dtype] = Tensor[dtype](ug_shape)
    fill(t1, 2.0)
    t2[0] = 2
    fill(ug, 1.0)

    var expected_grad1 = Tensor[dtype](t1_shape)
    var expected_grad2 = Tensor[dtype](t2_shape)
    fill(expected_grad1, 4.0)
    var temp = Tensor[dtype](2, 3)
    fill(temp, (2**2) * log[dtype, 1](2))
    expected_grad2[0] = tsum(temp)

    test_binary_op_backward[OP.POW, t1_shape, t2_shape, ug_shape](
        t1, t2, ug, expected_grad1, expected_grad2
    )


fn test_SUM() raises:
    alias t1_shape = TensorShape(2, 3)
    alias ug_shape = TensorShape(1)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    var ug: Tensor[dtype] = Tensor[dtype](ug_shape)
    fill(t1, 1.0)
    fill(ug, 9.0)

    var expected_grad1 = Tensor[dtype](t1_shape)
    fill(expected_grad1, 9.0)
    test_unary_op_backward[OP.SUM, t1_shape, ug_shape](t1, ug, expected_grad1)


fn test_SUM_0() raises:
    alias t1_shape = TensorShape(2, 3)
    alias ug_shape = TensorShape(1, 3)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    var ug: Tensor[dtype] = Tensor[dtype](ug_shape)
    fill(t1, 1.0)
    ug[0] = 0.0
    ug[1] = 1.0
    ug[2] = 2.0

    alias attributes = AttributeVector(Attribute("axis", 0))
    var expected_grad1 = Tensor[dtype](t1_shape)
    for i in range(expected_grad1.num_elements()):
        expected_grad1[i] = i % 3

    test_unary_op_backward[OP.SUM, t1_shape, ug_shape, attributes](
        t1, ug, expected_grad1
    )


fn test_SUM_1() raises:
    alias t1_shape = TensorShape(2, 3)
    alias ug_shape = TensorShape(2, 1)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    var ug: Tensor[dtype] = Tensor[dtype](ug_shape)
    fill(t1, 1.0)
    ug[0] = 0.0
    ug[1] = 1.0

    alias attributes = AttributeVector(Attribute("axis", 1))
    var expected_grad1 = Tensor[dtype](t1_shape)
    for i in range(expected_grad1.num_elements()):
        expected_grad1[i] = 0 if i < 3 else 1

    test_unary_op_backward[OP.SUM, t1_shape, ug_shape, attributes](
        t1, ug, expected_grad1
    )


fn test_MAX() raises:
    alias t1_shape = TensorShape(2, 3)
    alias ug_shape = TensorShape(1)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    var ug: Tensor[dtype] = Tensor[dtype](ug_shape)
    fill(t1, 1.0)
    t1[0] = 2.0
    t1[1] = 2.0
    fill(ug, 9.0)

    var expected_grad = Tensor[dtype](t1_shape)
    expected_grad[0] = 4.5
    expected_grad[1] = 4.5
    test_unary_op_backward[OP.MAX, t1_shape, ug_shape](t1, ug, expected_grad)


fn test_MAX_0() raises:
    alias t1_shape = TensorShape(2, 3, 2)
    alias ug_shape = TensorShape(1, 3, 2)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    var ug: Tensor[dtype] = Tensor[dtype](ug_shape)
    for i in range(t1.num_elements()):
        t1[i] = i + 1
    t1[0] = 7.0

    fill(ug, 2.0)

    alias attributes = AttributeVector(Attribute("axis", 0))
    var expected_grad = Tensor[dtype](t1_shape)
    expected_grad[0] = 1.0
    expected_grad[6] = 1.0
    expected_grad[7] = 2.0
    expected_grad[8] = 2.0
    expected_grad[9] = 2.0
    expected_grad[10] = 2.0
    expected_grad[11] = 2.0

    test_unary_op_backward[OP.MAX, t1_shape, ug_shape, attributes](
        t1, ug, expected_grad
    )


fn test_MAX_1() raises:
    alias t1_shape = TensorShape(2, 3, 2)
    alias ug_shape = TensorShape(2, 1, 2)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    var ug: Tensor[dtype] = Tensor[dtype](ug_shape)
    for i in range(t1.num_elements()):
        t1[i] = i + 1
    t1[0] = 5.0
    fill(ug, 2.0)

    alias attributes = AttributeVector(Attribute("axis", 1))
    var expected_grad = Tensor[dtype](t1_shape)
    expected_grad[0] = 1.0
    expected_grad[4] = 1.0
    expected_grad[5] = 2.0
    expected_grad[10] = 2.0
    expected_grad[11] = 2.0

    test_unary_op_backward[OP.MAX, t1_shape, ug_shape, attributes](
        t1, ug, expected_grad
    )


fn test_MAX_2() raises:
    alias t1_shape = TensorShape(2, 3, 2)
    alias ug_shape = TensorShape(2, 3, 1)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    var ug: Tensor[dtype] = Tensor[dtype](ug_shape)
    for i in range(t1.num_elements()):
        t1[i] = i + 1
    t1[0] = 2.0
    fill(ug, 2.0)

    alias attributes = AttributeVector(Attribute("axis", 2))
    var expected_grad = Tensor[dtype](t1_shape)
    expected_grad[0] = 1.0
    expected_grad[1] = 1.0
    expected_grad[3] = 2.0
    expected_grad[5] = 2.0
    expected_grad[7] = 2.0
    expected_grad[9] = 2.0
    expected_grad[11] = 2.0

    test_unary_op_backward[OP.MAX, t1_shape, ug_shape, attributes](
        t1, ug, expected_grad
    )


fn test_MEAN() raises:
    alias t1_shape = TensorShape(2, 3)
    alias ug_shape = TensorShape(1)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    var ug: Tensor[dtype] = Tensor[dtype](ug_shape)
    fill(t1, 1.0)
    fill(ug, 9.0)

    var expected_grad = Tensor[dtype](t1_shape)
    fill(expected_grad, 9.0 / 6.0)
    test_unary_op_backward[OP.MEAN, t1_shape, ug_shape](t1, ug, expected_grad)


fn test_MEAN_0() raises:
    alias t1_shape = TensorShape(2, 3)
    alias ug_shape = TensorShape(1, 3)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    var ug: Tensor[dtype] = Tensor[dtype](ug_shape)
    fill(t1, 1.0)
    fill(ug, 3.0)

    alias attributes = AttributeVector(Attribute("axis", 0))
    var expected_grad = Tensor[dtype](t1_shape)
    for i in range(expected_grad.num_elements()):
        expected_grad[i] = 1.0 / t1_shape[0] * 3.0

    test_unary_op_backward[OP.MEAN, t1_shape, ug_shape, attributes](
        t1, ug, expected_grad
    )


fn test_MEAN_1() raises:
    alias t1_shape = TensorShape(2, 3)
    alias ug_shape = TensorShape(2, 1)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    var ug: Tensor[dtype] = Tensor[dtype](ug_shape)
    fill(t1, 1.0)
    fill(ug, 3.0)

    alias attributes = AttributeVector(Attribute("axis", 1))
    var expected_grad = Tensor[dtype](t1_shape)
    for i in range(expected_grad.num_elements()):
        expected_grad[i] = 1.0 / t1_shape[1] * 3.0

    test_unary_op_backward[OP.MEAN, t1_shape, ug_shape, attributes](
        t1, ug, expected_grad
    )


fn test_TRANSPOSE() raises:
    alias t1_shape = TensorShape(2, 3, 4)
    alias ug_shape = TensorShape(4, 3, 2)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    var ug: Tensor[dtype] = Tensor[dtype](ug_shape)

    fn arange(inout t: Tensor[dtype]):
        var n = t.num_elements()
        for i in range(n):
            t[i] = i + 1

    arange(t1)
    arange(ug)

    # No attributes is reversion the order
    var expected_grad = Tensor[dtype](t1_shape)
    var t1_strides = t1_shape.strides()
    for i in range(ug_shape[0]):
        for j in range(ug_shape[1]):
            for k in range(ug_shape[2]):
                expected_grad[k * t1_strides[0] + j * t1_strides[1] + i] = ug[
                    i * ug_shape[1] * ug_shape[2] + j * ug_shape[2] + k
                ]

    test_unary_op_backward[OP.TRANSPOSE, t1_shape, ug_shape](t1, ug, expected_grad)

    # Test Transpose 1, 2, 0
    alias ug_shape_2 = TensorShape(3, 4, 2)
    ug = Tensor[dtype](ug_shape_2)
    arange(ug)

    alias attributes_2 = AttributeVector(Attribute("axes", TensorShape(1, 2, 0)))
    expected_grad = Tensor[dtype](t1_shape)
    for i in range(ug_shape_2[0]):
        for j in range(ug_shape_2[1]):
            for k in range(ug_shape_2[2]):
                expected_grad[k * t1_strides[0] + i * t1_strides[1] + j] = ug[
                    i * ug_shape_2[1] * ug_shape_2[2] + j * ug_shape_2[2] + k
                ]

    test_unary_op_backward[OP.TRANSPOSE, t1_shape, ug_shape_2, attributes_2](
        t1, ug, expected_grad
    )


fn test_FLATTEN() raises:
    alias t1_shape = TensorShape(2, 3)
    alias ug_shape = TensorShape(t1_shape.num_elements())
    var t1 = Tensor[dtype](t1_shape)
    var ug: Tensor[dtype] = Tensor[dtype](ug_shape)
    fill(ug, 1.0)
    assert_equal(ug.dim(0), 6)

    var expected_grad1 = Tensor[dtype](t1_shape)
    fill(expected_grad1, 1.0)
    test_unary_op_backward[OP.FLATTEN, t1_shape, ug_shape](t1, ug, expected_grad1)


fn test_RESHAPE() raises:
    alias t1_shape = TensorShape(2, 2, 5)
    alias ug_shape = TensorShape(2, 10)

    var t1 = Tensor[dtype](t1_shape)
    var ug: Tensor[dtype] = Tensor[dtype](ug_shape)
    var expected_grad = Tensor[dtype](t1_shape)
    for i in range(20):
        ug[i] = i + 1
        expected_grad[i] = i + 1

    test_unary_op_backward[OP.RESHAPE, t1_shape, ug_shape](t1, ug, expected_grad)


fn main():
    try:
        test_ADD()
        test_SUB()
        test_MUL()
        test_DIV()
        test_DOT()
        test_EXP()
        test_LOG()
        test_POW()
        test_SUM()
        test_SUM_0()
        test_SUM_1()
        test_MAX()
        test_MAX_0()
        test_MAX_1()
        test_MAX_2()
        test_MEAN()
        test_MEAN_0()
        test_MEAN_1()
        test_TRANSPOSE()
        test_FLATTEN()
        test_RESHAPE()
    except e:
        print(e)
        print("[ERROR] Error in backward pass.")
