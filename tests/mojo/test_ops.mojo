from math import exp, log
from utils.index import IndexList

from basalt import dtype, nelts
from basalt.autograd import OP
from basalt.autograd.attributes import Attribute, AttributeVector
from basalt.utils.tensorutils import fill
from basalt.nn import Tensor, TensorShape

from tests import test_unary_op, test_binary_op, test_ternary_op


fn test_ADD() raises:
    alias t1_shape = TensorShape(2, 3)
    alias t2_shape = TensorShape(2, 3)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    var t2: Tensor[dtype] = Tensor[dtype](t2_shape)
    fill(t1, 1.0)
    fill(t2, 1.0)

    var expected = Tensor[dtype](2, 3)
    fill(expected, 2.0)

    test_binary_op[OP.ADD, t1_shape, t2_shape](t1, t2, expected)


fn test_SUB() raises:
    alias t1_shape = TensorShape(2, 3)
    alias t2_shape = TensorShape(2, 3)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    var t2: Tensor[dtype] = Tensor[dtype](t2_shape)
    fill(t1, 2.0)
    fill(t2, 1.0)

    var expected = Tensor[dtype](2, 3)
    fill(expected, 1.0)

    test_binary_op[OP.SUB, t1_shape, t2_shape](t1, t2, expected)


fn test_MUL() raises:
    alias t1_shape = TensorShape(2, 3)
    alias t2_shape = TensorShape(2, 3)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    var t2: Tensor[dtype] = Tensor[dtype](t2_shape)
    fill(t1, 2.0)
    fill(t2, 3.0)

    var expected = Tensor[dtype](2, 3)
    fill(expected, 6.0)

    test_binary_op[OP.MUL, t1_shape, t2_shape](t1, t2, expected)


fn test_DIV() raises:
    alias t1_shape = TensorShape(2, 3)
    alias t2_shape = TensorShape(2, 3)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    var t2: Tensor[dtype] = Tensor[dtype](t2_shape)
    fill(t1, 6.0)
    fill(t2, 2.0)

    var expected = Tensor[dtype](2, 3)
    fill(expected, 3.0)

    test_binary_op[OP.DIV, t1_shape, t2_shape](t1, t2, expected)


fn test_DOT() raises:
    alias t1_shape = TensorShape(2, 3)
    alias t2_shape = TensorShape(3, 2)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    var t2: Tensor[dtype] = Tensor[dtype](t2_shape)
    fill(t1, 1.0)
    fill(t2, 2.0)

    var expected = Tensor[dtype](2, 2)
    fill(expected, 6.0)

    test_binary_op[OP.DOT, t1_shape, t2_shape](t1, t2, expected)


fn test_EXP() raises:
    alias t1_shape = TensorShape(2, 3)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    fill(t1, 2.0)

    var expected = Tensor[dtype](2, 3)
    fill(expected, exp(SIMD[dtype, 1](2.0)))

    test_unary_op[OP.EXP, t1_shape](t1, expected)


fn test_LOG() raises:
    alias t1_shape = TensorShape(2, 3)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    fill(t1, 2.0)

    var expected = Tensor[dtype](2, 3)
    fill(expected, log(SIMD[dtype, 1](2.0)))

    test_unary_op[OP.LOG, t1_shape](t1, expected)


fn test_POW() raises:
    alias t1_shape = TensorShape(2, 3)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    fill(t1, 2.0)

    alias t2_shape = TensorShape(1)
    var t2: Tensor[dtype] = Tensor[dtype](t2_shape)
    t2[0] = 2.0

    var expected = Tensor[dtype](2, 3)
    fill(expected, 4.0)

    test_binary_op[OP.POW, t1_shape, t2_shape](t1, t2, expected)


fn test_SUM() raises:
    alias t1_shape = TensorShape(2, 3, 4)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    fill(t1, 1.0)

    # No axis specified
    var expected = Tensor[dtype](1)
    fill(expected, 24.0)
    test_unary_op[OP.SUM, t1_shape](t1, expected)

    # Test axis 1
    alias attrs = AttributeVector(Attribute("axis", 1))
    expected = Tensor[dtype](2, 1, 4)
    fill(expected, 3.0)
    test_unary_op[OP.SUM, t1_shape, attrs](t1, expected)


fn test_MAX() raises:
    alias t1_shape = TensorShape(2, 3, 2)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    for i in range(t1_shape.num_elements()):
        t1[i] = i + 1

    # No axis specified
    var expected = Tensor[dtype](1)
    fill(expected, t1_shape.num_elements())
    test_unary_op[OP.MAX, t1_shape](t1, expected)

    @parameter
    fn fill_tensor[
        size: Int
    ](inout tensor: Tensor[dtype], values: IndexList[size]):
        for i in range(tensor.num_elements()):
            tensor[i] = values[i]

    # Test axis 0
    alias attrs = AttributeVector(Attribute("axis", 0))
    var expected_max_axis_0_temp = IndexList[6](7, 8, 9, 10, 11, 12)
    expected = Tensor[dtype](1, 3, 2)
    fill_tensor(expected, expected_max_axis_0_temp)
    test_unary_op[OP.MAX, t1_shape, attrs](t1, expected)

    # Test axis 1
    alias attrs_1 = AttributeVector(Attribute("axis", 1))
    var expected_max_axis_1_temp = IndexList[4](5, 6, 11, 12)
    expected = Tensor[dtype](2, 1, 2)
    fill_tensor(expected, expected_max_axis_1_temp)
    test_unary_op[OP.MAX, t1_shape, attrs_1](t1, expected)

    # Test axis 2
    alias attrs_2 = AttributeVector(Attribute("axis", 2))
    var expected_max_axis_2_temp = IndexList[6](2, 4, 6, 8, 10, 12)
    expected = Tensor[dtype](2, 3, 1)
    fill_tensor(expected, expected_max_axis_2_temp)
    test_unary_op[OP.MAX, t1_shape, attrs_2](t1, expected)


fn test_MEAN() raises:
    alias t1_shape = TensorShape(2, 3)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    fill(t1, 5.0)

    # No axis specified
    var expected = Tensor[dtype](1)
    fill(expected, 5.0)
    test_unary_op[OP.MEAN, t1_shape](t1, expected)

    # Test axis 0
    alias attrs = AttributeVector(Attribute("axis", 0))
    expected = Tensor[dtype](1, 3)
    fill(expected, 5.0)
    test_unary_op[OP.MEAN, t1_shape, attrs](t1, expected)

    # Test axis 1
    alias attrs_1 = AttributeVector(Attribute("axis", 1))
    expected = Tensor[dtype](2, 1)
    fill(expected, 5.0)
    test_unary_op[OP.MEAN, t1_shape, attrs_1](t1, expected)


fn test_TRANSPOSE() raises:
    alias t1_shape = TensorShape(2, 3, 4)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    for i in range(t1_shape.num_elements()):
        t1[i] = i + 1

    # Test tranpose (no attributes = reversing the axis by default)
    var expected = Tensor[dtype](4, 3, 2)
    var expected_strides = expected.strides()
    for i in range(t1_shape[0]):
        for j in range(t1_shape[1]):
            for k in range(t1_shape[2]):
                expected[k * expected_strides[0] + j * expected_strides[1] + i] = t1[
                    i * t1_shape[1] * t1_shape[2] + j * t1_shape[2] + k
                ]

    test_unary_op[OP.TRANSPOSE, t1_shape](t1, expected)

    # Test tranpose 1, 2, 0
    alias attrs = AttributeVector(Attribute("axes", TensorShape(1, 2, 0)))
    var expected_axis_1 = Tensor[dtype](3, 4, 2)
    var expected_axis_1_strides = expected_axis_1.strides()
    for i in range(t1_shape[0]):
        for j in range(t1_shape[1]):
            for k in range(t1_shape[2]):
                expected_axis_1[
                    j * expected_axis_1_strides[0] + k * expected_axis_1_strides[1] + i
                ] = t1[i * t1_shape[1] * t1_shape[2] + j * t1_shape[2] + k]

    test_unary_op[OP.TRANSPOSE, t1_shape, attrs](t1, expected_axis_1)


fn test_FLATTEN() raises:
    alias t1_shape = TensorShape(2, 3, 4)
    var t1 = Tensor[dtype](t1_shape)
    fill(t1, 1.0)

    var expected = Tensor[dtype](24)
    fill(expected, 1.0)

    test_unary_op[OP.FLATTEN, t1_shape](t1, expected)


fn test_RESHAPE() raises:
    alias t_shape = TensorShape(2, 2, 5)
    alias new_shape = TensorShape(2, 10)

    var t = Tensor[dtype](t_shape)
    var expected = Tensor[dtype](new_shape)
    for i in range(20):
        t[i] = i + 1
        expected[i] = i + 1

    alias attrs = AttributeVector(Attribute("shape", new_shape))
    test_unary_op[OP.RESHAPE, t_shape, attrs](t, expected)


fn test_FMA() raises:
    alias t1_shape = TensorShape(2, 3)
    alias t2_shape = TensorShape(2, 3)
    alias t3_shape = TensorShape(2, 3)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    var t2: Tensor[dtype] = Tensor[dtype](t2_shape)
    var t3: Tensor[dtype] = Tensor[dtype](t3_shape)
    fill(t1, 1.0)
    fill(t2, 2.0)
    fill(t3, 3.0)

    var expected = Tensor[dtype](2, 3)
    fill(expected, 1.0 * 2.0 + 3.0)

    test_ternary_op[OP.FMA, t1_shape, t2_shape, t3_shape](t1, t2, t3, expected)


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
        test_MAX()
        test_MEAN()
        test_TRANSPOSE()
        test_FLATTEN()
        test_RESHAPE()
        test_FMA()
    except e:
        print("[ERROR] Error in ops")
        print(e)
