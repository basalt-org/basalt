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
    fn __init__(inout self, shape: DynamicVector[Int]):
        self._shape = unpack(shape)

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


# NOTE: Mojo 24.1.0
# There seems to be no way of having a decent init method in TensorShape that accepts a DynamicVector and unpack its content to create a variadic list. 
# Alternatives are changing variadic to a different type. However, that would require to parameterize TensorShape[rank] if the type needs to be register_passable.
# Since `Tensor[dtype: DType, shape: TensorShape[rank]]`  It would then only be possible to create a Tensor with a fixed rank. e.g. TensorShape[max_rank] if Tensor needs to be general.
# We chose for manual unpacking, over a predifiend constant max_rank, that occupies compiled memory for all tensors.

@always_inline("nodebug")
fn unpack(s: DynamicVector[Int]) -> VariadicList[Int]:
    var rank = len(s)

    if rank == 0:
        return VariadicList[Int]()
    elif rank == 1:
        return VariadicList[Int](s[0])
    elif rank == 2:
        return VariadicList[Int](s[0], s[1])
    elif rank == 3:
        return VariadicList[Int](s[0], s[1], s[2])
    elif rank == 4:
        return VariadicList[Int](s[0], s[1], s[2], s[3])
    elif rank == 5:
        return VariadicList[Int](s[0], s[1], s[2], s[3], s[4])
    elif rank == 6:
        return VariadicList[Int](s[0], s[1], s[2], s[3], s[4], s[5])
    elif rank == 7:
        return VariadicList[Int](s[0], s[1], s[2], s[3], s[4], s[5], s[6])
    elif rank == 8:
        return VariadicList[Int](s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7])
    elif rank == 9:
        return VariadicList[Int](s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8])
    elif rank == 10:
        return VariadicList[Int](s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], s[9])
    else:
        print("[ERROR] Unpacking to TensorShape with rank > 10 is not supported.")
        return VariadicList[Int]()
    