from testing import assert_true
from algorithm import vectorize
from utils.index import IndexList
from memory import memset_zero, memcpy, UnsafePointer


alias MAX_RANK = 8


@register_passable("trivial")
struct TensorShape(Stringable):
    var _rank: Int
    var _shape: IndexList[MAX_RANK]

    fn __init__(inout self, *shape: Int):
        self._rank = len(shape)
        self._shape = IndexList[MAX_RANK]()
        for i in range(min(self._rank, MAX_RANK)):
            self._shape[i] = shape[i]

    fn __init__(inout self, shapes: VariadicList[Int]):
        self._rank = len(shapes)
        self._shape = IndexList[MAX_RANK]()
        for i in range(min(self._rank, MAX_RANK)):
            self._shape[i] = shapes[i]

    fn __init__(inout self, shape: List[Int]):
        self._rank = len(shape)
        self._shape = IndexList[MAX_RANK]()
        for i in range(min(self._rank, MAX_RANK)):
            self._shape[i] = shape[i]

    fn __init__[num: Int](inout self, shape: IndexList[num]):
        self._rank = num
        self._shape = IndexList[MAX_RANK]()
        for i in range(min(self._rank, MAX_RANK)):
            self._shape[i] = shape[i]

    fn __init__(inout self, rank: Int, shape: IndexList[MAX_RANK]):
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

    fn num_elements(self) -> Int:
        var result = 1
        for i in range(self._rank):
            result *= self._shape[i]
        return result

    fn strides(self) -> IndexList[MAX_RANK]:
        var result = IndexList[MAX_RANK](0)
        result[self._rank - 1] = 1
        for i in range(self._rank - 2, -1, -1):
            result[i] = result[i + 1] * self._shape[i + 1]
        return result

    fn __str__(self) -> String:
        var s: String = "("
        for i in range(self._rank):
            s += str(self._shape[i])
            if i < self._rank - 1:
                s += ", "
        return s + ")"

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

    fn __contains__(self, value: Int) -> Bool:
        for i in range(self.rank()):
            if self[i] == value:
                return True
        return False

    fn to_list(self) -> List[Int]:
        var result = List[Int]()
        for i in range(self.rank()):
            result.append(self[i])
        return result


struct Tensor[dtype: DType](Stringable, Movable, CollectionElement):
    var _data: UnsafePointer[Scalar[dtype]]
    var _shape: TensorShape

    fn __init__(inout self, *dims: Int):
        self._shape = TensorShape(dims)
        self._data = UnsafePointer[Scalar[dtype]].alloc(self._shape.num_elements())
        memset_zero(self._data, self._shape.num_elements())

    fn __init__(inout self, owned shape: TensorShape):
        self._data = UnsafePointer[Scalar[dtype]].alloc(shape.num_elements())
        memset_zero(self._data, shape.num_elements())
        self._shape = shape

    fn __init__(
        inout self, owned data: UnsafePointer[Scalar[dtype]], owned shape: TensorShape
    ):
        # NOTE: Remember to use _ = your_tensor that you passed, so there is no weird behavior in this function
        self._data = UnsafePointer[Scalar[dtype]].alloc(shape.num_elements())
        self._shape = shape
        
        memcpy(self._data, data, self._shape.num_elements())
        _ = data

    fn __moveinit__(inout self, owned other: Tensor[dtype]):
        self._data = other._data
        self._shape = other._shape

    fn __copyinit__(inout self, other: Tensor[dtype]):
        # print("[WARNING] Copying tensor")
        self._data = UnsafePointer[Scalar[dtype]].alloc(other._shape.num_elements())
        memcpy(self._data, other._data, other.num_elements())
        self._shape = other._shape

    @always_inline("nodebug")
    fn __getitem__(self, index: Int) -> Scalar[dtype]:
        return self._data[index]

    @always_inline("nodebug")
    fn __setitem__(self, index: Int, value: Scalar[dtype]):
        self._data[index] = value

    @always_inline("nodebug")
    fn data(self) -> UnsafePointer[Scalar[dtype]]:
        return self._data

    @always_inline("nodebug")
    fn shape(self) -> TensorShape:
        return self._shape

    @always_inline("nodebug")
    fn load[simd_width: Int](self, index: Int) -> SIMD[dtype, simd_width]:
        return self._data.load[width=simd_width](index)

    @always_inline("nodebug")
    fn store[simd_width: Int](self, index: Int, value: SIMD[dtype, simd_width]):
        self._data.store(index, value)

    @always_inline("nodebug")
    fn strides(self) -> IndexList[MAX_RANK]:
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

    fn __str__(self) -> String:
        # temp fix
        var s: String = "["
        for i in range(self.num_elements()):
            s += str(self[i])
            if i < self.num_elements() - 1:
                s += ", "
        return s + "]"


    @always_inline("nodebug")
    fn __del__(owned self):
        self._data.free()
