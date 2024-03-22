from testing import assert_equal

from basalt import dtype
from basalt import Tensor, TensorShape, Symbol
from basalt.utils.uuid import UUID, UUIDGenerator
from basalt.utils.collection import Collection
from basalt.utils.tensorutils import fill
from test_tensorutils import assert_tensors_equal


fn test_append_tensors(inout uuid: UUIDGenerator) raises:
    alias t1_shape = TensorShape(1, 10)
    alias t2_shape = TensorShape(2, 20)
    var s1 = Symbol(uuid.next(), dtype, t1_shape, True)
    var s2 = Symbol(uuid.next(), dtype, t2_shape, True)

    var c = Collection(capacity=2)
    assert_equal(c.capacity, 2)
    assert_equal(c.size, 0)
    assert_equal(len(c.symbol_map), 0)

    c.append(Tensor[dtype](s1.shape), s1)
    assert_equal(c.size, 1)
    assert_equal(len(c.symbol_map), 1)

    c.append(Tensor[dtype](s2.shape), s2)
    assert_equal(c.size, 2)
    assert_equal(len(c.symbol_map), 2)


fn test_get_tensor_reference(inout uuid: UUIDGenerator) raises:
    alias t1_shape = TensorShape(1, 10)
    alias t2_shape = TensorShape(2, 20)
    var s1 = Symbol(uuid.next(), dtype, t1_shape, True)
    var s2 = Symbol(uuid.next(), dtype, t2_shape, True)

    var t1 = Tensor[dtype](s1.shape)
    var t2 = Tensor[dtype](s2.shape)
    fill(t1, 1)
    fill(t2, 2)

    var c = Collection(capacity=2)
    c.append(t1^, s1)
    c.append(t2^, s2)

    var t1_expected = Tensor[dtype](s1.shape)
    var t2_expected = Tensor[dtype](s2.shape)
    fill(t1_expected, 1)
    fill(t2_expected, 2)

    assert_tensors_equal(c[s1], t1_expected)
    assert_tensors_equal(c[s2], t2_expected)


fn test_resize_collection(inout uuid: UUIDGenerator) raises:
    alias t1_shape = TensorShape(1, 10)
    alias t2_shape = TensorShape(2, 20)
    alias t3_shape = TensorShape(3, 30)
    var s1 = Symbol(uuid.next(), dtype, t1_shape, True)
    var s2 = Symbol(uuid.next(), dtype, t2_shape, True)
    var s3 = Symbol(uuid.next(), dtype, t3_shape, True)

    var t1 = Tensor[dtype](s1.shape)
    var t2 = Tensor[dtype](s2.shape)
    var t3 = Tensor[dtype](s3.shape)
    fill(t1, 1)
    fill(t2, 2)
    fill(t3, 3)

    var c = Collection(capacity=1)
    assert_equal(c.size, 0)
    assert_equal(c.capacity, 1)

    c.append(t1^, s1)
    assert_equal(c.size, 1)
    assert_equal(c.capacity, 1)

    c.append(t2^, s2)
    assert_equal(c.size, 2) 
    assert_equal(c.capacity, 2) # current capacity * 2

    c.append(t3^, s3)
    assert_equal(c.size, 3)
    assert_equal(c.capacity, 4) # current capacity * 2

    var t1_expected = Tensor[dtype](s1.shape)
    var t2_expected = Tensor[dtype](s2.shape)
    var t3_expected = Tensor[dtype](s3.shape)
    fill(t1_expected, 1)
    fill(t2_expected, 2)
    fill(t3_expected, 3)

    assert_tensors_equal(c[s1], t1_expected)
    assert_tensors_equal(c[s2], t2_expected)
    assert_tensors_equal(c[s3], t3_expected)


fn test_set_zero(inout uuid: UUIDGenerator) raises:
    alias t1_shape = TensorShape(1, 10)
    alias t2_shape = TensorShape(2, 20)
    var s1 = Symbol(uuid.next(), dtype, t1_shape, True)
    var s2 = Symbol(uuid.next(), dtype, t2_shape, True)
    var t1 = Tensor[dtype](s1.shape)
    var t2 = Tensor[dtype](s2.shape)
    fill(t1, 1)
    fill(t2, 2)

    var c = Collection(capacity=2)
    c.append(t1^, s1)
    c.append(t2^, s2)

    var t1_expected = Tensor[dtype](s1.shape)
    var t2_expected = Tensor[dtype](s2.shape)
    fill(t1_expected, 1)
    fill(t2_expected, 2)
    assert_tensors_equal(c[s1], t1_expected)
    assert_tensors_equal(c[s2], t2_expected)
    
    c.set_zero()

    assert_tensors_equal(c[s1], Tensor[dtype](t1_shape))
    assert_tensors_equal(c[s2], Tensor[dtype](t2_shape))



fn main() raises:
    var uuid = UUIDGenerator(42)

    try:
        test_append_tensors(uuid)
        test_get_tensor_reference(uuid)
        test_resize_collection(uuid)
        test_set_zero(uuid)
    except e:
        print(e)
        raise e