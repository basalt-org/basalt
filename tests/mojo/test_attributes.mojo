from testing import assert_equal, assert_true
from utils.index import IndexList

from basalt.nn import TensorShape
from basalt.autograd.attributes import Attribute


fn test_attribute_key() raises:
    alias a = Attribute(name="test", value=-1)

    assert_true(str(a.name) == "test")


fn test_attribute_int() raises:
    alias value: Int = 1
    alias a = Attribute(name="test", value=value)

    assert_true(a.to_int() == 1)


fn test_attribute_string() raises:
    alias value: String = "hello"
    alias a = Attribute(name="test", value=value)

    assert_true(a.to_string() == value)


fn test_attribute_tensor_shape() raises:
    alias value: TensorShape = TensorShape(1, 2, 3)
    alias a = Attribute(name="test", value=value)

    assert_true(a.to_shape() == value)


fn test_attribute_static_int_tuple() raises:
    alias value: IndexList[7] = IndexList[7](1, 2, 3, 4, 5, 6, 7)
    alias a = Attribute(name="test", value=value)

    assert_true(a.to_static[7]() == value)


fn test_attribute_scalar() raises:
    fn test_float32() raises:
        alias value_a: Float32 = 1.23456
        alias a1 = Attribute(name="test", value=value_a)
        assert_true(
            a1.to_scalar[DType.float32]() == value_a,
            "Float32 scalar attribute failed",
        )

        alias value_b: Float32 = 65151
        alias a2 = Attribute(name="test", value=value_b)
        assert_true(
            a2.to_scalar[DType.float32]() == value_b,
            "Float32 scalar attribute failed",
        )

    fn test_float_literal() raises:
        alias value_c: FloatLiteral = -1.1
        alias a3 = Attribute(name="test", value=value_c)
        assert_true(
            a3.to_scalar[DType.float32]() == value_c,
            "FloatLiteral scalar attribute failed",
        )

    fn test_float64() raises:
        alias value_a: Float64 = -1.23456
        alias a1 = Attribute(name="test", value=value_a)
        assert_true(
            a1.to_scalar[DType.float64]() == value_a,
            "Float64 scalar attribute failed",
        )

        alias value_b: Float64 = 123456
        alias a2 = Attribute(name="test", value=value_b)
        assert_true(
            a2.to_scalar[DType.float64]() == value_b,
            "Float64 scalar attribute failed",
        )

    fn test_int32() raises:
        alias value_a: Int32 = 666
        alias a1 = Attribute(name="test", value=value_a)
        assert_true(
            a1.to_scalar[DType.int32]() == value_a,
            "Int32 scalar attribute failed",
        )

        alias value_b: Int32 = -666
        alias a2 = Attribute(name="test", value=value_b)
        assert_true(
            a2.to_scalar[DType.int32]() == value_b,
            "Int32 scalar attribute failed",
        )

    fn test_attribute_small_scalar() raises:
        alias value_a: Float32 = 1e-18
        alias a = Attribute(name="test", value=value_a)
        assert_true(
            a.to_scalar[DType.float32]() == value_a,
            "SMALL scalar attribute failed",
        )

    fn test_attribute_big_scalar() raises:
        alias value_a: Float32 = 1e40
        alias a = Attribute(name="test", value=value_a)
        assert_true(
            a.to_scalar[DType.float32]() == value_a,
            "BIG scalar attribute failed",
        )

    test_float32()
    test_float_literal()
    test_float64()
    test_int32()
    test_attribute_small_scalar()
    test_attribute_big_scalar()


fn main():
    try:
        test_attribute_key()
        test_attribute_int()
        test_attribute_string()
        test_attribute_tensor_shape()
        test_attribute_static_int_tuple()
        test_attribute_scalar()
    except e:
        print("[ERROR] Error in attributes")
        print(e)
