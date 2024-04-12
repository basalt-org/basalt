from math.limit import inf, neginf, max_finite, min_finite

from basalt import TensorShape
from basalt.autograd.attributes import Attribute

from testing import assert_equal, assert_true


fn test_attribute_key() raises:
    alias a = Attribute(name="test", value=-1)

    assert_true(str(a.name) == "test")


fn test_attribute_int() raises:
    alias value: Int = 1

    alias a = Attribute(name="test", value=value)

    assert_true(a.to_int() == 1)


fn test_attribute_string() raises:
    alias a = Attribute(name="test", value="hello")

    assert_true(a.to_string() == "hello")


fn test_attribute_tensor_shape() raises:
    alias value: TensorShape = TensorShape(1, 2, 3)

    alias a = Attribute(name="test", value=value)

    assert_true(a.to_shape() == value)


fn test_attribute_static_int_tuple() raises:
    alias value: StaticIntTuple[7] = StaticIntTuple[7](1, 2, 3, 4, 5, 6, 7)

    alias a = Attribute(name="test", value=value)

    assert_true(a.to_static[7]() == value)


fn test_attribute_scalar() raises:
    alias value_a: Float32 = 1.23456
    alias a = Attribute(name="test", value=value_a)
    alias a_val = a.to_scalar[DType.float32]()
    assert_true(a_val == value_a, "Float32 scalar attribute failed")

    alias a2 = Attribute(name="test", value=65151)
    alias a2_val = a2.to_scalar[DType.float32]()
    assert_true(a2_val == 65151, "Float32 scalar attribute failed")

    alias value_b: Float64 = -1.23456
    alias b = Attribute(name="test", value=value_b)
    alias b_val = b.to_scalar[DType.float64]()
    assert_true(b_val == value_b, "Float64 scalar attribute failed")

    alias b2 = Attribute(name="test", value=123456)
    alias b2_val = b2.to_scalar[DType.float64]()
    assert_true(b2_val == 123456, "Float64 scalar attribute failed")

    alias value_c: Int32 = 666
    alias c = Attribute(name="test", value=value_c)
    alias c_val = c.to_scalar[DType.int32]()
    assert_true(c_val == value_c, "Int32 scalar attribute failed")

    alias c2 = Attribute(name="test", value=-666)
    alias c2_val = c2.to_scalar[DType.int32]()
    assert_true(c2_val == -666, "Int32 scalar attribute failed")

    alias value_d: Float32 = 0.3333
    alias d = Attribute(name="test", value=value_d)
    alias d_val = d.to_scalar[DType.float32]()
    assert_true(d_val == value_d, "SMALL scalar attribute failed")


fn test_attribute_big_scalar() raises:
    alias value_a: Float32 = 1e20
    alias a = Attribute(name="test", value=value_a)
    assert_true(a.to_scalar[DType.float32]() == value_a, "BIG scalar attribute failed")

    alias value_b: Float32 = max_finite[DType.float32]()
    alias b = Attribute(name="test", value=value_a)
    assert_true(
        b.to_scalar[DType.float32]() == value_b, "MAX FINITE scalar attribute failed"
    )

    alias value_c: Float32 = inf[DType.float32]()
    alias c = Attribute(name="test", value=value_a)
    assert_true(c.to_scalar[DType.float32]() == value_c, "INF scalar attribute failed")


fn main():
    try:
        test_attribute_key()
        test_attribute_int()
        test_attribute_string()
        test_attribute_tensor_shape()
        test_attribute_static_int_tuple()
        test_attribute_scalar()
        # test_attribute_big_scalar()  # BUG'S
    except e:
        print(e)
