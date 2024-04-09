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
    var value_a: Scalar[DType.float32] = Scalar[DType.float32](1.23456)
    var a = Attribute(name="test", value=value_a)
    assert_true(a.to_scalar[DType.float32]() == value_a, "Float32 scalar attribute failed")
    
    var value_b: Scalar[DType.float64] = Scalar[DType.float64](1.23456)
    var b = Attribute(name="test", value=value_b)
    assert_true(b.to_scalar[DType.float64]() == value_b, "Float64 scalar attribute failed")

    var value_c: Scalar[DType.int32] = Scalar[DType.int32](666)
    var c = Attribute(name="test", value=value_c)
    assert_true(c.to_scalar[DType.int32]() == value_c, "Int32 scalar attribute failed")


fn main():
    try:
        test_attribute_key()
        test_attribute_int()
        test_attribute_string()
        test_attribute_tensor_shape()
        test_attribute_static_int_tuple()
        test_attribute_scalar()
    except e:
        print(e)