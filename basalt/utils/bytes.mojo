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
                result += chr(val.to_int())

        return result


@always_inline("nodebug")
fn f64_to_bytes[
    size: Int = DType.float64.sizeof()
](value: Scalar[DType.float64]) -> Bytes[size]:
    """
    Convert a f64 number to a sequence of bytes in IEEE 754 format.
    """
    alias exponent_bits = 11
    alias mantissa_bits = 52
    alias exponent_bias = 1023

    if value == 0:
        return Bytes[size]()

    var sign: Int64
    var abs: Float64

    if value > 0:
        sign = 0
        abs = value
    else:
        sign = 1
        abs = -value

    var exponent: Int64 = exponent_bias

    while abs >= 2.0:
        abs /= 2.0
        exponent += 1
    while abs < 1.0:
        abs *= 2.0
        exponent -= 1

    var mantissa = (abs - 1.0) * (1 << mantissa_bits)
    var binary_rep: Int64 = (sign << (exponent_bits + mantissa_bits)) | (
        exponent << mantissa_bits
    ) | mantissa

    var result = Bytes[size]()

    @parameter
    fn fill_bytes[Index: Int]():
        alias Offest: Int64 = Index * 8
        result[Index] = (binary_rep >> Offest & 0xFF).cast[DType.uint8]()

    unroll[fill_bytes, size]()

    return result


fn bytes_to_f64[
    size: Int = DType.float64.sizeof()
](bytes: Bytes[size]) -> Scalar[DType.float64]:
    """
    Convert a sequence of bytes in IEEE 754 format to a floating point number.
    """

    alias exponent_bits = 11
    alias mantissa_bits = 52
    alias exponent_bias = 1023

    var binary_rep: Int64 = 0

    @parameter
    fn to_bin[Index: Int]():
        alias Offest: Int64 = Index * 8
        binary_rep |= bytes[Index].cast[DType.int64]() << Offest

    unroll[to_bin, size]()

    var sign = (-1) ** ((binary_rep >> (exponent_bits + mantissa_bits)) & 1).to_int()
    var exponent: Int = (
        (binary_rep >> mantissa_bits) & ((1 << exponent_bits) - 1)
    ).to_int() - exponent_bias
    var mantissa: Float64 = (binary_rep & ((1 << mantissa_bits) - 1)).cast[
        DType.float64
    ]() / (1 << mantissa_bits) + Float64(exponent != -exponent_bias)

    if exponent == exponent_bias + 1:
        return inf[DType.float64]() if mantissa == 0 else nan[DType.float64]()
    elif exponent == -exponent_bias and mantissa == 0:
        return 0.0
    elif exponent < 0:
        return sign * 1.0 / Float64(2**-exponent) * mantissa
    else:
        return sign * Float64(2**exponent) * mantissa
