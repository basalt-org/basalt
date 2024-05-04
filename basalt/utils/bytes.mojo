from math import nan
from math.limit import inf

alias ScalarBytes = DType.uint64.sizeof()


@register_passable("trivial")
struct Bytes[capacity: Int](Stringable, CollectionElement, EqualityComparable):
    """
    Static sequence of bytes.
    """

    var data: StaticTuple[UInt8, capacity]

    @always_inline("nodebug")
    fn __init__(inout self):
        var data = StaticTuple[UInt8, capacity]()

        @unroll
        for i in range(capacity):
            data[i] = 0

        self.data = data

    @always_inline("nodebug")
    fn __init__(inout self, s: String):
        var data = StaticTuple[UInt8, capacity]()
        var length = len(s)

        @unroll
        for i in range(capacity):
            data[i] = ord(s[i]) if i < length else 0

        self.data = data

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        return capacity

    @always_inline("nodebug")
    fn __setitem__(inout self, index: Int, value: UInt8):
        self.data[index] = value

    @always_inline("nodebug")
    fn __getitem__(self, index: Int) -> UInt8:
        return self.data[index]

    @always_inline("nodebug")
    fn __eq__(self, other: Self) -> Bool:
        @unroll
        for i in range(capacity):
            if self[i] != other[i]:
                return False
        return True

    @always_inline("nodebug")
    fn __ne__(self, other: Self) -> Bool:
        @unroll
        for i in range(capacity):
            if self[i] != other[i]:
                return True
        return False

    @always_inline("nodebug")
    fn __str__(self) -> String:
        var result: String = ""

        @unroll
        for i in range(capacity):
            var val = self[i]
            if val != 0:
                result += chr(int(val))

        return result


fn scalar_to_bytes[
    Type: DType, Size: Int = ScalarBytes
](value: Scalar[Type]) -> Bytes[Size]:
    constrained[Size >= ScalarBytes, "Size must be at least ${ScalarBytes}"]()

    var int_bytes = bitcast[DType.uint64](value.cast[expand_type[Type]()]())
    var data = Bytes[Size]()

    @parameter
    fn copy_bytes[Index: Int]():
        data[Index] = (int_bytes >> (Index * 8) & 0xFF).cast[DType.uint8]()

    unroll[copy_bytes, ScalarBytes]()

    return data


fn bytes_to_scalar[Type: DType](bytes: Bytes) -> Scalar[Type]:
    constrained[
        bytes.capacity >= ScalarBytes, "Size must be at least ${ScalarBytes}"
    ]()

    var int_bytes: UInt64 = 0

    @parameter
    fn copy_bytes[Index: Int]():
        int_bytes |= bytes[Index].cast[DType.uint64]() << (Index * 8)

    unroll[copy_bytes, ScalarBytes]()

    return bitcast[expand_type[Type]()](int_bytes).cast[Type]()


fn expand_type[Type: DType]() -> DType:
    @parameter
    if Type.is_floating_point():
        return DType.float64
    elif Type.is_signed():
        return DType.int64
    return DType.uint64
