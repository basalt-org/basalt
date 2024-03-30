from math import min
from testing import assert_true

from tensor import Tensor as _Tensor
from tensor import TensorShape as _TensorShape


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
    fn __init__(inout self, shapes: VariadicList[Int]):
        self._rank = len(shapes)
        self._shape = StaticIntTuple[max_rank]()
        for i in range(min(self._rank, max_rank)):
            self._shape[i] = shapes[i]

    @always_inline("nodebug")
    fn __init__(inout self, shape: List[Int]):
        self._rank = len(shape)
        self._shape = StaticIntTuple[max_rank]()
        for i in range(min(self._rank, max_rank)):
            self._shape[i] = shape[i]

    @always_inline("nodebug")
    fn __init__[num: Int](inout self, shape: StaticIntTuple[num]):
        self._rank = num
        self._shape = StaticIntTuple[max_rank]()
        for i in range(min(self._rank, max_rank)):
            self._shape[i] = shape[i]

    @always_inline("nodebug")
    fn __init__(inout self, rank: Int, shape: StaticIntTuple[max_rank]):
        self._rank = rank
        self._shape = shape

    @always_inline("nodebug")
    fn __getitem__(self, index: Int) -> Int:
        return self._shape[index if index >= 0 else self._rank + index]

    @always_inline("nodebug")
    fn __setitem__(inout self, index: Int, value: Int):
        self._shape[index if index >= 0 else self._rank + index] = value

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
    fn strides(self) -> StaticIntTuple[max_rank]:
        var result = StaticIntTuple[max_rank](0)
        result[self._rank - 1] = 1
        for i in range(self._rank - 2, -1, -1):
            result[i] = result[i + 1] * self._shape[i + 1]
        return result

    @always_inline("nodebug")
    fn _std_shape(self) -> _TensorShape:
        var s = List[Int](capacity=self.rank())
        for i in range(self.rank()):
            s.append(self[i])
        return _TensorShape(s)

    @always_inline("nodebug")
    fn __str__(self) -> String:
        return str(self._std_shape())

    @always_inline("nodebug")
    fn __eq__(self, other: TensorShape) -> Bool:
        if self.rank() != other.rank():
            return False
        for i in range(self.rank()):
            if self[i] != other[i]:
                return False
        return True

    @always_inline("nodebug")
    fn __ne__(self, other: TensorShape) -> Bool:
        return not self.__eq__(other)


struct Tensor[dtype: DType](Stringable, Movable, CollectionElement):
    var _data: DTypePointer[dtype]
    var _shape: TensorShape

    @always_inline("nodebug")
    fn __init__(inout self, *dims: Int):
        self._shape = TensorShape(dims)
        self._data = DTypePointer[dtype].alloc(self._shape.num_elements())
        memset_zero(self._data, self._shape.num_elements())

    @always_inline("nodebug")
    fn __init__(inout self, owned shape: TensorShape):
        self._data = DTypePointer[dtype].alloc(shape.num_elements())
        memset_zero(self._data, shape.num_elements())
        self._shape = shape

    @always_inline("nodebug")
    fn __init__(inout self, owned data: DTypePointer[dtype], owned shape: TensorShape):
        self._data = data
        self._shape = shape

    @always_inline("nodebug")
    fn __moveinit__(inout self, owned other: Tensor[dtype]):
        self._data = other._data
        self._shape = other._shape

    @always_inline("nodebug")
    fn __copyinit__(inout self, other: Tensor[dtype]):
        # print("[WARNING] Copying tensor")
        self._data = DTypePointer[dtype].alloc(other._shape.num_elements())
        memcpy(self._data, other._data, other.num_elements())
        self._shape = other._shape

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
    fn shape(self) -> TensorShape:
        return self._shape

    @always_inline("nodebug")
    fn load[simd_width: Int](self, index: Int) -> SIMD[dtype, simd_width]:
        return self._data.load[width=simd_width](index)

    @always_inline("nodebug")
    fn store[simd_width: Int](self, index: Int, value: SIMD[dtype, simd_width]):
        self._data.store[width=simd_width](index, value)

    @always_inline("nodebug")
    fn strides(self) -> StaticIntTuple[max_rank]:
        return self._shape.strides()

    @always_inline("nodebug")
    fn rank(self) -> Int:
        return self._shape.rank()

    @always_inline("nodebug")
    fn num_elements(self) -> Int:
        return self._shape.num_elements()

    @always_inline("nodebug")
    fn dim(self, index: Int) -> Int:
        return self._shape[index]

    @always_inline("nodebug")
    fn zero(self):
        memset_zero(self._data, self.num_elements())

    @always_inline("nodebug")
    fn ireshape(inout self, new_shape: TensorShape) raises:
        # NOTE Consider not raising on error
        assert_true(self.num_elements() == new_shape.num_elements())
        self._shape = new_shape

    @always_inline("nodebug")
    fn __str__(self) -> String:
        var new_data = DTypePointer[dtype].alloc(self.num_elements())
        var std_shape = self._shape._std_shape()
        memcpy(new_data, self._data, self.num_elements())
        return str(_Tensor[dtype](ptr=new_data, shape=std_shape))

    @always_inline("nodebug")
    fn __del__(owned self):
        self._data.free()
