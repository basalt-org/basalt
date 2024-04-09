from math import min


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
fn float_to_bytes[
    dtype: DType,
    size: Int = dtype.sizeof()
](value: Scalar[dtype]) -> Bytes[size]:
    """
    Convert a floating point number to a sequence of bytes in IEEE 754 format.
    Supported byte sizes are 2 (f16), 4 (f32) and 8 (f64) bytes.
    """
    constrained[
        dtype == DType.float16 or dtype == DType.float32 or dtype == DType.float64,
        "must be eiter float16, float32 or float64"
    ]()
    
    var exponent_bits: Int
    var mantissa_bits: Int
    
    @parameter
    if dtype == DType.float16:
        exponent_bits = 5
        mantissa_bits = 10
    elif dtype == DType.float32:
        exponent_bits = 8
        mantissa_bits = 23
    else: # DType.float64
        exponent_bits = 11
        mantissa_bits = 52

    var exponent_bias = (1 << (exponent_bits - 1)) - 1

    # Extract sign, exponent and mantissa
    var sign = 0 if value >= 0 else 1
    var abs = value if value >= 0 else -value

    var mantissa: SIMD[dtype, 1] = 0
    var exponent = exponent_bias
    if value == 0.0:
        exponent = 0
        mantissa = 0
    else:
        while abs >= 2.0:
            abs /= 2.0
            exponent += 1
        while abs < 1.0:
            abs *= 2.0
            exponent -= 1

        mantissa = (abs - 1.0) * (1 << mantissa_bits)

    # Binary representation to bytes
    var binary_rep = (sign << (exponent_bits + mantissa_bits)) | (exponent << mantissa_bits) | int(mantissa)
    
    var result = Bytes[size]()
    for i in range(size):
        result[i] = (binary_rep >> (8 * i)) & 0xFF

    return result


fn bytes_to_float[
    dtype: DType,
    size: Int = dtype.sizeof()
](bytes: Bytes[size]) -> Scalar[dtype]:
    """
    Convert a sequence of bytes in IEEE 754 format to a floating point number.
    Supported byte sizes are 2 (f16), 4 (f32) and 8 (f64) bytes.
    """
    constrained[
        dtype == DType.float16 or dtype == DType.float32 or dtype == DType.float64,
        "must be eiter float16, float32 or float64"
    ]()
    
    var exponent_bits: Int
    var mantissa_bits: Int
    var exponent_bias: Int
    
    @parameter
    if dtype == DType.float16:
        exponent_bits = 5
        mantissa_bits = 10
        exponent_bias = 15
    elif dtype == DType.float32:
        exponent_bits = 8
        mantissa_bits = 23
        exponent_bias = 127
    else: # DType.float64
        exponent_bits = 11
        mantissa_bits = 52
        exponent_bias = 1023


    var binary_rep = 0
    for i in range(size):
        binary_rep |= bytes[i].to_int() << (8 * i)

    # Extract sign, exponent, and mantissa
    var sign = (-1)**((binary_rep >> (exponent_bits + mantissa_bits)) & 0x1)
    var exponent = ((binary_rep >> mantissa_bits) & ((1 << exponent_bits) - 1)) - exponent_bias
    var mantissa: Scalar[dtype] = binary_rep & ((1 << mantissa_bits) - 1)

    # Normalize the mantissa
    mantissa = 1 + mantissa / (1 << mantissa_bits) if exponent != -exponent_bias else mantissa / (1 << mantissa_bits)

    if exponent == exponent_bias + 1:
        print("inf" if mantissa == 0 else "nan")
        # return float('inf') if mantissa == 0 else float('nan')
        # TODO
        return -1
    elif exponent == -exponent_bias and mantissa == 0:
        return 0.0 * sign
    else:
        return Scalar[dtype](sign) * (Scalar[dtype](2) ** exponent) * mantissa