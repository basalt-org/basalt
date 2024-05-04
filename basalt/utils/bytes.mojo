from math import nan
from math.limit import inf


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
    Type: DType, Size: Int = Type.sizeof()
](value: Scalar[Type]) -> Bytes[Size]:
    var int_bytes = bitcast[DType.uint64](value.cast[expand_type[Type]()]())
    var bytes = Bytes[Size]()

    @unroll
    for i in range(DType.uint64.sizeof()):
        bytes[i] = (int_bytes >> (i * 8) & 0xFF).cast[DType.uint8]()

    return bytes


fn bytes_to_scalar[Type: DType, Size: Int = Type.sizeof()](bytes: Bytes) -> Scalar[Type]:
    var int_bytes: Scalar[DType.uint64] = 0
    
    @unroll
    for i in range(DType.uint64.sizeof()):
        int_bytes |= (bytes[i].cast[DType.uint64]() << (i * 8))

    return bitcast[expand_type[Type]()](int_bytes).cast[Type]()


fn expand_type[Type: DType]() -> DType:
    @parameter
    if Type.is_floating_point():
        return DType.float64
    elif Type.is_signed():
        return DType.int64
    return DType.uint64
