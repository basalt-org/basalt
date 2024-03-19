from tensor import Tensor as _Tensor
from tensor import TensorShape as _TensorShape
from collections.vector import InlinedFixedVector


@register_passable("trivial")
struct TensorShape(Stringable):
    var _shape: VariadicList[Int]

    @always_inline("nodebug")
    fn __init__(inout self, *shape: Int):
        self._shape = shape

    @always_inline("nodebug")
    fn __getitem__(self, index: Int) -> Int:
        return self._shape[index]

    @always_inline("nodebug")
    fn rank(self) -> Int:
        return len(self._shape)

    @always_inline("nodebug")
    fn num_elements(self) -> Int:
        var result = 1
        for i in self._shape:
            result *= i
        return result

    @always_inline("nodebug")
    fn strides(self) -> InlinedFixedVector[Int]:
        var result = InlinedFixedVector[Int](self.rank())
        result[self.rank() - 1] = 1
        for i in range(self.rank() - 2, -1, -1):
            result[i] = result[i + 1] * self._shape[i + 1]
        return result

    @always_inline("nodebug")
    fn __str__(self) -> String:
        return str(_TensorShape(self._shape))


@register_passable("trivial")
struct Tensor[dtype: DType, shape: TensorShape](Stringable):
    var _data: DTypePointer[dtype]

    @always_inline("nodebug")
    fn __init__(inout self):
        self._data = DTypePointer[dtype].alloc(shape.num_elements())

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
        return str(_Tensor[dtype](self._data, _TensorShape(shape._shape)))