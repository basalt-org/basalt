from math import log, exp
from testing import assert_true, assert_equal
from test_tensorutils import assert_tensors_equal


from basalt import Tensor, TensorShape
from basalt.utils.tensorutils import fill, tsum
from basalt.autograd.ops.basics import ADD, SUB, MUL, DIV, DOT, EXP, LOG, POW, MEAN, FLATTEN, SUM, MAX, RESHAPE, TRANSPOSE
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

    var grad1 = ADD.backward[0, ug_shape, t1_shape, t2_shape](ug, t1, t2)
    assert_tensors_equal(grad1, expected_grad)
    var grad2 = ADD.backward[1, ug_shape, t1_shape, t2_shape](ug, t1, t2)
    assert_tensors_equal(grad2, expected_grad)


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

    var grad1 = SUB.backward[0, ug_shape, t1_shape, t2_shape](ug, t1, t2)
    var expected_grad1 = Tensor[dtype](t1_shape)
    fill(expected_grad1, 1.0)
    assert_tensors_equal(grad1, expected_grad1)

    var grad2 = SUB.backward[1, ug_shape, t1_shape, t2_shape](ug, t1, t2)
    var expected_grad2 = Tensor[dtype](t2_shape)
    fill(expected_grad2, -1.0)
    assert_tensors_equal(grad2, expected_grad2)


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

    var grad1 = MUL.backward[0, ug_shape, t1_shape, t2_shape](ug, t1, t2)
    var expected_grad1 = Tensor[dtype](t1_shape)
    fill(expected_grad1, 2.0)
    assert_tensors_equal(grad1, expected_grad1)

    var grad2 = MUL.backward[1, ug_shape, t1_shape, t2_shape](ug, t1, t2)
    var expected_grad2 = Tensor[dtype](t2_shape)
    fill(expected_grad2, 1.0)
    assert_tensors_equal(grad2, expected_grad2)


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

    var grad1 = DIV.backward[0, ug_shape, t1_shape, t2_shape](ug, t1, t2)
    var expected_grad1 = Tensor[dtype](t1_shape)
    fill(expected_grad1, 1.0 / 2.0)
    assert_tensors_equal(grad1, expected_grad1)

    var grad2 = DIV.backward[1, ug_shape, t1_shape, t2_shape](ug, t1, t2)
    var expected_grad2 = Tensor[dtype](t2_shape)
    fill(expected_grad2, -1.0 / (2.0**2))
    assert_tensors_equal(grad2, expected_grad2)


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

    var grad1 = DOT.backward[0, ug_shape, t1_shape, t2_shape](ug, t1, t2)
    var expected_grad1 = Tensor[dtype](t1_shape)
    fill(expected_grad1, 4.0)
    assert_tensors_equal(grad1, expected_grad1)

    var grad2 = DOT.backward[1, ug_shape, t1_shape, t2_shape](ug, t1, t2)
    var expected_grad2 = Tensor[dtype](t2_shape)
    fill(expected_grad2, 2.0)
    assert_tensors_equal(grad2, expected_grad2)


fn test_EXP() raises:
    alias t1_shape = TensorShape(2, 3)
    alias ug_shape = TensorShape(2, 3)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    var ug: Tensor[dtype] = Tensor[dtype](ug_shape)
    fill(t1, 2.0)
    fill(ug, 5.0)

    var grad1 = EXP.backward[ug_shape, t1_shape](ug, t1)
    var expected_grad1 = Tensor[dtype](t1_shape)
    fill(expected_grad1, 5.0 * exp[dtype, 1](2.0))
    assert_tensors_equal(grad1, expected_grad1)


fn test_LOG() raises:
    alias t1_shape = TensorShape(2, 3)
    alias ug_shape = TensorShape(2, 3)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    var ug: Tensor[dtype] = Tensor[dtype](ug_shape)
    fill(t1, 2.0)
    fill(ug, 5.0)

    var grad1 = LOG.backward[ug_shape, t1_shape](ug, t1)
    var expected_grad1 = Tensor[dtype](t1_shape)
    fill(expected_grad1, 5.0 / 2.0)
    assert_tensors_equal(grad1, expected_grad1)


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

    var grad1 = POW.backward[0, ug_shape, t1_shape, t2_shape](ug, t1, t2)
    var expected_grad1 = Tensor[dtype](t1_shape)
    fill(expected_grad1, 4.0)
    assert_tensors_equal(grad1, expected_grad1)

    var grad2 = POW.backward[1, ug_shape, t1_shape, t2_shape](ug, t1, t2)
    var temp = Tensor[dtype](2, 3)
    fill(temp, (2**2) * log[dtype, 1](2))
    assert_equal(grad2[0], tsum(temp))
    assert_equal(grad2.shape(), 1)


fn test_SUM() raises:
    alias t1_shape = TensorShape(2, 3)
    alias ug_shape = TensorShape(1)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    var ug: Tensor[dtype] = Tensor[dtype](ug_shape)
    fill(t1, 1.0)
    fill(ug, 9.0)

    var grad1 = SUM.backward[ug_shape, t1_shape](ug, t1)
    var expected_grad1 = Tensor[dtype](t1_shape)
    fill(expected_grad1, 9.0)
    assert_tensors_equal(grad1, expected_grad1)


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
    var grad1 = SUM.backward[ug_shape, t1_shape, attributes](ug, t1)
    var expected_grad1 = Tensor[dtype](t1_shape)
    for i in range(expected_grad1.num_elements()):
        expected_grad1[i] = i % 3

    assert_tensors_equal(grad1, expected_grad1)

fn test_SUM_1() raises:
    alias t1_shape = TensorShape(2, 3)
    alias ug_shape = TensorShape(2, 1)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    var ug: Tensor[dtype] = Tensor[dtype](ug_shape)
    fill(t1, 1.0)
    ug[0] = 0.0
    ug[1] = 1.0

    alias attributes = AttributeVector(Attribute("axis", 1))
    var grad1 = SUM.backward[ug_shape, t1_shape, attributes](ug, t1)
    var expected_grad1 = Tensor[dtype](t1_shape)
    for i in range(expected_grad1.num_elements()):
        expected_grad1[i] = 0 if i < 3 else 1

    assert_tensors_equal(grad1, expected_grad1)


fn test_MAX() raises:
    alias t1_shape = TensorShape(2, 3)
    alias ug_shape = TensorShape(1)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    var ug: Tensor[dtype] = Tensor[dtype](ug_shape)
    fill(t1, 1.0)
    t1[0] = 2.0
    t1[1] = 2.0
    fill(ug, 9.0)

    var grad = MAX.backward[ug_shape, t1_shape](ug, t1)

    var expected_grad = Tensor[dtype](t1_shape)
    expected_grad[0] = 4.5
    expected_grad[1] = 4.5
    assert_tensors_equal(grad, expected_grad)


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
    var grad = MAX.backward[ug_shape, t1_shape, attributes](ug, t1)
    var expected_grad = Tensor[dtype](t1_shape)
    expected_grad[0] = 1.0
    expected_grad[6] = 1.0
    expected_grad[7] = 2.0
    expected_grad[8] = 2.0
    expected_grad[9] = 2.0
    expected_grad[10] = 2.0
    expected_grad[11] = 2.0

    assert_tensors_equal(grad, expected_grad)


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
    var grad = MAX.backward[ug_shape, t1_shape, attributes](ug, t1)
    var expected_grad = Tensor[dtype](t1_shape)
    expected_grad[0] = 1.0
    expected_grad[4] = 1.0
    expected_grad[5] = 2.0
    expected_grad[10] = 2.0
    expected_grad[11] = 2.0

    assert_tensors_equal(grad, expected_grad)


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
    var grad = MAX.backward[ug_shape, t1_shape, attributes](ug, t1)
    var expected_grad = Tensor[dtype](t1_shape)
    expected_grad[0] = 1.0
    expected_grad[1] = 1.0
    expected_grad[3] = 2.0
    expected_grad[5] = 2.0
    expected_grad[7] = 2.0
    expected_grad[9] = 2.0
    expected_grad[11] = 2.0

    assert_tensors_equal(grad, expected_grad)


fn test_MEAN() raises:
    alias t1_shape = TensorShape(2, 3)
    alias ug_shape = TensorShape(1)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    var ug: Tensor[dtype] = Tensor[dtype](ug_shape)
    fill(t1, 1.0)
    fill(ug, 9.0)

    var grad = MEAN.backward[ug_shape, t1_shape](ug, t1)
    var expected_grad = Tensor[dtype](t1_shape)
    fill(expected_grad, 9.0 / 6.0)

    assert_tensors_equal(grad, expected_grad)


fn test_MEAN_0() raises:
    alias t1_shape = TensorShape(2, 3)
    alias ug_shape = TensorShape(1, 3)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    var ug: Tensor[dtype] = Tensor[dtype](ug_shape)
    fill(t1, 1.0)
    fill(ug, 3.0)

    alias attributes = AttributeVector(Attribute("axis", 0))
    var grad = MEAN.backward[ug_shape, t1_shape, attributes](ug, t1)
    var expected_grad = Tensor[dtype](t1_shape)
    for i in range(expected_grad.num_elements()):
        expected_grad[i] = 1.0 / t1_shape[0] * 3.0

    assert_tensors_equal(grad, expected_grad)


fn test_MEAN_1() raises:
    alias t1_shape = TensorShape(2, 3)
    alias ug_shape = TensorShape(2, 1)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    var ug: Tensor[dtype] = Tensor[dtype](ug_shape)
    fill(t1, 1.0)
    fill(ug, 3.0)

    alias attributes = AttributeVector(Attribute("axis", 1))
    var grad = MEAN.backward[ug_shape, t1_shape, attributes](ug, t1)
    var expected_grad = Tensor[dtype](t1_shape)
    for i in range(expected_grad.num_elements()):
        expected_grad[i] = 1.0 / t1_shape[1] * 3.0

    assert_tensors_equal(grad, expected_grad)


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

    alias attributes = AttributeVector(Attribute("axes", TensorShape(2, 1, 0)))
    var grad = TRANSPOSE.backward[ug_shape, t1_shape, attributes](ug, t1)
    var expected_grad = Tensor[dtype](t1_shape)
    var t1_strides = t1_shape.strides()

    for i in range(ug_shape[0]):
        for j in range(ug_shape[1]):
            for k in range(ug_shape[2]):
                expected_grad[k * t1_strides[0] + j * t1_strides[1] + i] = ug[i * ug_shape[1] * ug_shape[2] + j * ug_shape[2] + k]

    assert_tensors_equal(grad, expected_grad)

    # Test Transpose 1, 2, 0

    alias ug_shape_2 = TensorShape(3, 4, 2)
    ug = Tensor[dtype](ug_shape_2)
    arange(ug)

    alias attributes_2 = AttributeVector(Attribute("axes", TensorShape(1, 2, 0)))
    grad = TRANSPOSE.backward[ug_shape, t1_shape, attributes_2](ug, t1)
    expected_grad = Tensor[dtype](t1_shape)

    for i in range(ug_shape_2[0]):
        for j in range(ug_shape_2[1]):
            for k in range(ug_shape_2[2]):
                expected_grad[k * t1_strides[0] + i * t1_strides[1] + j] = ug[i * ug_shape_2[1] * ug_shape_2[2] + j * ug_shape_2[2] + k]

    assert_tensors_equal(grad, expected_grad)


fn test_FLATTEN() raises:
    alias t1_shape = TensorShape(2, 3)
    alias ug_shape = TensorShape(t1_shape.num_elements())
    var t1 = Tensor[dtype](t1_shape)
    var ug: Tensor[dtype] = Tensor[dtype](ug_shape)
    fill(ug, 1.0)
    assert_equal(ug.dim(0), 6)

    var grad1 = FLATTEN.backward[ug_shape, t1_shape](ug, t1)

    var expected_grad1 = Tensor[dtype](t1_shape)
    fill(expected_grad1, 1.0)

    assert_tensors_equal(grad1, expected_grad1)


fn test_RESHAPE() raises:
    alias t1_shape = TensorShape(2, 2, 5)
    alias ug_shape = TensorShape(2, 10)

    var t1 = Tensor[dtype](t1_shape)
    var ug: Tensor[dtype] = Tensor[dtype](ug_shape)
    var expected_grad = Tensor[dtype](t1_shape)
    for i in range(20):
        ug[i] = i + 1
        expected_grad[i] = i + 1

    var grad = RESHAPE.backward[ug_shape, t1_shape](ug, t1)

    assert_tensors_equal(grad, expected_grad)


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
