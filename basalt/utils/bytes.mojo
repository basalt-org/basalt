from math import min, nan
from math.limit import inf


@value
@register_passable("trivial")
struct Bytes[capacity: Int](Stringable, CollectionElement):
    """
    Static sequence of bytes.
    """

    var _vector: StaticTuple[UInt8, capacity]

    fn __init__(inout self):
        self._vector = StaticTuple[UInt8, capacity](0)

    fn __init__(inout self, s: String):
        var _vector = StaticTuple[UInt8, capacity](0)
        for i in range(min(len(s), capacity)):
            _vector[i] = ord(s[i])
        self._vector = _vector

    fn __len__(self) -> Int:
        return len(self._vector)

    fn __setitem__(inout self, index: Int, value: UInt8):
        self._vector[index] = value

    fn __getitem__(self, index: Int) -> UInt8:
        return self._vector[index]

    fn __str__(self) -> String:
        var result: String = ""
        for i in range(self.__len__()):
            if self[i].to_int() != 0:
                result += chr(self[i].to_int())
        return result

    fn __eq__(self, other: Self) -> Bool:
        for i in range(self.__len__()):
            if self[i] != other[i]:
                return False
        return True

    fn hex(self) -> String:
        var result: String = ""
        alias hex_table: String = "0123456789abcdef"
        for i in range(self.__len__()):
            result += (
                hex_table[((self[i] >> 4) & 0xF).to_int()]
                + hex_table[(self[i] & 0xF).to_int()]
            )
        return result


@always_inline("nodebug")
fn f64_to_bytes[size: Int = DType.float64.sizeof()](value: Scalar[DType.float64]) -> Bytes[size]:
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
    var binary_rep: Int64 = (sign << (exponent_bits + mantissa_bits)) | (exponent << mantissa_bits) | mantissa

    var result = Bytes[size]()

    @parameter
    fn fill_bytes[Index: Int]():
        alias Offest: Int64 = Index * 8
        result[Index] = (binary_rep >> Offest & 0xFF).cast[DType.uint8]()

    unroll[fill_bytes, size]()

    return result


fn bytes_to_f64[size: Int = DType.float64.sizeof()](bytes: Bytes[size]) -> Scalar[DType.float64]:
    """
    Convert a sequence of bytes in IEEE 754 format to a floating point number.
    """

    alias exponent_bits = 11
    alias mantissa_bits = 52
    alias exponent_bias = 1023

    var binary_rep = 0

    @parameter
    fn to_bin[Index: Int]():
        alias Offest = Index * 8
        binary_rep |= bytes[Index].to_int() << Offest

    unroll[to_bin, size]()

    var exponent = (
        (binary_rep >> mantissa_bits) & ((1 << exponent_bits) - 1)
    ) - exponent_bias
    var mantissa = (binary_rep & ((1 << mantissa_bits) - 1)) / (
        1 << mantissa_bits
    ) + (exponent != -exponent_bias)


    if exponent == exponent_bias + 1:
        return inf[DType.float64]() if mantissa == 0 else nan[DType.float64]()
    elif exponent == -exponent_bias and mantissa == 0:
        return 0.0
    
    var sign = (-1) ** ((binary_rep >> (exponent_bits + mantissa_bits)) & 1)
    return sign * (2**exponent) * mantissa