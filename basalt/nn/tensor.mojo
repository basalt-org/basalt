from math import min

from tensor import Tensor as _Tensor
from tensor import TensorShape as _TensorShape
from collections.vector import InlinedFixedVector


alias max_rank = 8


@register_passable("trivial")
struct TensorShape(Stringable):
    var _rank: Int
    var _shape: StaticIntTuple[max_rank]

    @always_inline("nodebug")
    fn __init__(inout self, *shape: Int):
        self._rank = len(shape)
        self._shape = StaticIntTuple[max_rank]()
        for i in range(min(self._rank, max_rank)):
            self._shape[i] = shape[i]

    @always_inline("nodebug")
    fn __init__(inout self, shape: DynamicVector[Int]):
        self._rank = len(shape)
        self._shape = StaticIntTuple[max_rank]()
        for i in range(min(self._rank, max_rank)):
            self._shape[i] = shape[i]

    @always_inline("nodebug")
    fn __getitem__(self, index: Int) -> Int:
        return self._shape[index]

    @always_inline("nodebug")
    fn rank(self) -> Int:
        return self._rank

    @always_inline("nodebug")
    fn num_elements(self) -> Int:
        var result = 1
        for i in range(self._rank):
            result *= self._shape[i]
        return result

    @always_inline("nodebug")
    fn strides(self) -> InlinedFixedVector[Int]:
        var result = InlinedFixedVector[Int](self.rank())
        result[self.rank() - 1] = 1
        for i in range(self.rank() - 2, -1, -1):
            result[i] = result[i + 1] * self._shape[i + 1]
        return result

    @always_inline("nodebug")
    fn _std_shape(self) -> _TensorShape:
        var s = DynamicVector[Int](capacity=self.rank())
        for i in range(self.rank()):
            s.push_back(self[i])
        return _TensorShape(s)
    
    @always_inline("nodebug")
    fn __str__(self) -> String:
        return str(self._std_shape())


@register_passable("trivial")
struct Tensor[dtype: DType, shape: TensorShape](Stringable):
    var _data: DTypePointer[dtype]

    @always_inline("nodebug")
    fn __init__(inout self):
        self._data = DTypePointer[dtype].alloc(shape.num_elements())

    # @always_inline("nodebug")
    # fn __init__(inout self, data: DTypePointer[dtype]):
    #     self._data = data

    @always_inline("nodebug")
    fn __getitem__(self, index: Int) -> SIMD[dtype, 1]:
        return self._data[index]

    @always_inline("nodebug")
    fn __setitem__(self, index: Int, value: SIMD[dtype, 1]):
        self._data[index] = value

    @always_inline("nodebug")
    fn data(self) -> DTypePointer[dtype]:
        return self._data

    @always_inline("nodebug")
    fn simd_load[simd_width: Int](self, index: Int) -> SIMD[dtype, simd_width]:
        return self._data.simd_load[simd_width](index)

    @always_inline("nodebug")
    fn simd_store[simd_width: Int](self, index: Int, value: SIMD[dtype, simd_width]):
        self._data.simd_store[simd_width](index, value)

    @always_inline("nodebug")
    fn strides(self) -> InlinedFixedVector[Int]:
        return shape.strides()

    @always_inline("nodebug")
    fn rank(self) -> Int:
        return shape.rank()

    @always_inline("nodebug")
    fn num_elements(self) -> Int:
        return shape.num_elements()

    @always_inline("nodebug")
    fn __str__(self) -> String:
        return str(_Tensor[dtype](self._data, shape._std_shape()))
    