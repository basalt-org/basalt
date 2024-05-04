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
    alias UInt_N = get_uint_type[Type]()

    var int_bytes = bitcast[UInt_N](value)
    var bytes = Bytes[Size]()

    @unroll
    for i in range(Type.sizeof()):
        bytes[i] = (int_bytes >> (i * 8)).cast[DType.uint8]()

    return bytes


fn bytes_to_scalar[Type: DType, Size: Int = Type.sizeof()](bytes: Bytes) -> Scalar[Type]:
    alias UInt_N = get_uint_type[Type]()
    
    var int_bytes: Scalar[UInt_N] = 0
    
    @unroll
    for i in range(Size):
        int_bytes |= (bytes[i].cast[UInt_N]() << (i * 8))

    return bitcast[Type](int_bytes)


fn get_uint_type[Type: DType, Size: Int = Type.sizeof()]() -> DType:
    @parameter
    if Size == 8:
        return DType.uint64
    elif Size == 4:
        return DType.uint32
    elif Size == 2:
        return DType.uint16
    return DType.uint8
