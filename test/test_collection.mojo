from testing import assert_equal

from basalt import dtype
from basalt import Tensor, TensorShape, Symbol
from basalt.utils.collection import Collection
from basalt.utils.tensorutils import fill
from test_tensorutils import assert_tensors_equal


fn test_append_tensors() raises:
    alias t1_shape = TensorShape(1, 10)
    alias t2_shape = TensorShape(2, 20)
    var s1 = Symbol(0, dtype, t1_shape, True)
    var s2 = Symbol(1, dtype, t2_shape, True)

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


fn test_get_tensor_reference() raises:
    alias t1_shape = TensorShape(1, 10)
    alias t2_shape = TensorShape(2, 20)
    var s1 = Symbol(0, dtype, t1_shape, True)
    var s2 = Symbol(1, dtype, t2_shape, True)

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


fn test_resize_collection() raises:
    alias t1_shape = TensorShape(1, 10)
    alias t2_shape = TensorShape(2, 20)
    alias t3_shape = TensorShape(3, 30)
    var s1 = Symbol(0, dtype, t1_shape, True)
    var s2 = Symbol(1, dtype, t2_shape, True)
    var s3 = Symbol(2, dtype, t3_shape, True)

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


fn test_set_zero() raises:
    alias t1_shape = TensorShape(1, 10)
    alias t2_shape = TensorShape(2, 20)
    var s1 = Symbol(0, dtype, t1_shape, True)
    var s2 = Symbol(1, dtype, t2_shape, True)
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


fn test_operate_on_reference() raises:
    alias res_shape = TensorShape(1, 10)
    alias t1_shape = TensorShape(1, 10)
    var sr = Symbol(0, dtype, t1_shape, True)
    var s1 = Symbol(1, dtype, t1_shape, True)
    var res = Tensor[dtype](res_shape)
    var t1 = Tensor[dtype](s1.shape)
        
    var c = Collection(capacity=2)
    c.append(res^, sr)
    c.append(t1^, s1)

    fn some_operation[res_shape: TensorShape, t_shape: TensorShape](inout res: Tensor[dtype], t1: Tensor[dtype]):
        for i in range(res.num_elements()):
            res[i] = t1[i]

    for i in range(1, 10):
        some_operation[res_shape, t1_shape](c[sr], c[s1])        
        fill(c[s1], i)

        # Expected
        var res_expected = Tensor[dtype](res_shape)
        var t1_expected = Tensor[dtype](t1_shape)
        fill(res_expected, i-1)
        fill(t1_expected, i)

        assert_tensors_equal(c[sr], res_expected)
        assert_tensors_equal(c[s1], t1_expected)


fn main() raises:
    try:
        test_append_tensors()
        test_get_tensor_reference()
        test_resize_collection()
        test_set_zero()
        test_operate_on_reference()
    except e:
        print(e)
        raise e