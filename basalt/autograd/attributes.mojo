from collections import Optional, OptionalReg

from basalt.nn.tensor import Tensor, TensorShape, MAX_RANK
from basalt.utils.bytes import Bytes, scalar_to_bytes, bytes_to_scalar


alias MAX_ATTRS = 10
alias MAX_NAME_CHARS = 16
alias MAX_DATA_BYTES = 32


@register_passable("trivial")
struct AttributeType(Stringable):
    alias BOOL = AttributeType(0, "BOOL")
    alias INT = AttributeType(1, "INT")
    alias FLOAT = AttributeType(2, "FLOAT")
    alias STRING = AttributeType(3, "STRING")
    alias INTS = AttributeType(4, "INTS")
    alias FLOATS = AttributeType(5, "FLOATS")

    var id: UInt8
    var name: Bytes[MAX_NAME_CHARS]

    fn __init__(inout self, id: UInt8, name: String):
        self.id = id
        self.name = Bytes[MAX_NAME_CHARS](name)

    fn __init__(inout self, type: DType):
        if type.is_floating_point():
            self = AttributeType.FLOAT
        elif type.is_bool():
            self = AttributeType.BOOL
        else:
            self = AttributeType.INT

    fn __eq__(self, other: Self) -> Bool:
        return self.id == other.id

    fn __str__(self) -> String:
        return str(self.name)


@register_passable("trivial")
struct AttributeVector(Sized, Stringable, CollectionElement):
    var attributes: StaticTuple[Attribute, MAX_ATTRS]
    var size: Int

    @always_inline("nodebug")
    fn __init__(inout self, *attributes: Attribute):
        self.attributes = StaticTuple[Attribute, MAX_ATTRS]()
        self.size = len(attributes)
        for i in range(self.size):
            self.attributes[i] = attributes[i]

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        return self.size

    @always_inline("nodebug")
    fn __getitem__(self, index: Int) -> Attribute:
        return self.attributes[index]

    @always_inline("nodebug")
    fn __getitem__(self, index: StringLiteral) -> OptionalReg[Attribute]:
        for i in range(self.size):
            if self.attributes[i].name == Bytes[MAX_NAME_CHARS](index):
                return self.attributes[i]
        return None

    @always_inline("nodebug")
    fn append(inout self, attribute: Attribute):
        self.attributes[self.size] = attribute
        self.size += 1

    @always_inline("nodebug")
    fn __str__(self) -> String:
        var s: String = "["
        for i in range(self.size):
            s += str(self.attributes[i])
            if i < self.size - 1:
                s += ", "
        return s + "]"


@register_passable("trivial")
struct Attribute(Stringable, CollectionElement):
    var data_shape: StaticIntTuple[MAX_RANK]
    var name: Bytes[MAX_NAME_CHARS]
    var data: Bytes[MAX_DATA_BYTES]
    var type: AttributeType
    var size: Int

    @always_inline("nodebug")
    fn __init__(inout self, name: String, value: String):
        self.data_shape = StaticIntTuple[MAX_RANK]()
        self.name = Bytes[MAX_NAME_CHARS](name)
        self.data = Bytes[MAX_DATA_BYTES](value)
        self.type = AttributeType.STRING
        self.size = len(value)

    @always_inline("nodebug")
    fn __init__(inout self, name: String, value: TensorShape):
        self.data_shape = StaticIntTuple[MAX_RANK]()
        self.name = Bytes[MAX_NAME_CHARS](name)
        self.data = Bytes[MAX_DATA_BYTES]()
        self.type = AttributeType.INTS
        self.size = value.rank()

        for i in range(self.size):
            self.data_shape[i] = value._shape[i]

    @always_inline("nodebug")
    fn __init__[N: Int](inout self, name: String, value: StaticIntTuple[N]):
        constrained[N < MAX_RANK, "Attribute rank must be less than MAX_RANK."]()

        self.data_shape = StaticIntTuple[MAX_RANK]()
        self.name = Bytes[MAX_NAME_CHARS](name)
        self.data = Bytes[MAX_DATA_BYTES]()
        self.type = AttributeType.INTS
        self.size = N

        for i in range(self.size):
            self.data_shape[i] = value[i]

    @always_inline("nodebug")
    fn __init__[dtype: DType](inout self, name: String, value: Scalar[dtype]):
        constrained[dtype.is_numeric(), "Attribute value must be numeric."]()

        self.data_shape = StaticIntTuple[MAX_RANK]()
        self.name = Bytes[MAX_NAME_CHARS](name)
        self.data = scalar_to_bytes[dtype, MAX_DATA_BYTES](value)
        self.type = AttributeType(dtype)
        self.size = 1

    @always_inline("nodebug")
    fn __init__(inout self, name: String, value: Int):
        self.__init__(name, Int64(value))
        self.data_shape[0] = 1

    @always_inline("nodebug")
    fn __init__(inout self, name: String, value: FloatLiteral):
        self.__init__(name, Float64(value))
        self.data_shape[0] = 1

    @always_inline("nodebug")
    fn __str__(self) -> String:
        return "Attribute(" + str(self.name) + ", " + "..." + ")"

    @always_inline("nodebug")
    fn to_string(self) -> String:
        return str(self.data)

    @always_inline("nodebug")
    fn to_shape(self) -> TensorShape:
        return TensorShape(rank=self.size, shape=self.data_shape)

    @always_inline("nodebug")
    fn to_static[N: Int](self) -> StaticIntTuple[N]:
        constrained[N < MAX_RANK, "Attribute rank must be less than MAX_RANK."]()

        var result = StaticIntTuple[N]()

        for i in range(N):
            result[i] = int(self.data_shape[i])

        return result

    @always_inline("nodebug")
    fn to_scalar[dtype: DType](self) -> Scalar[dtype]:
        constrained[dtype.is_numeric(), "Attribute value must be numeric."]()

        return bytes_to_scalar[dtype](self.data)

    @always_inline("nodebug")
    fn to_int(self) -> Int:
        return int(self.to_scalar[DType.int64]())

    fn json(self) -> String:
        var result = '{"name": "' + str(self.name) + '", '

        var type: String = ""
        var value: String = ""

        if self.type == AttributeType.STRING:
            type = "STRING"
            value = '"' + self.to_string() + '"'
        elif self.type == AttributeType.INTS:
            type = "INTS"

            var value_temp = self.to_shape()
            value = "["
            for i in range(value_temp.rank()):
                value += str(value_temp._shape[i])
                if i < value_temp.rank() - 1:
                    value += ", "
            value += "]"
        elif self.type == AttributeType.FLOAT:
            type = "FLOAT"
            value = str(self.to_scalar[DType.float64]())
        elif self.type == AttributeType.INT:
            type = "INT"
            value = str(self.to_int())
        else:
            type = "UNKNOWN"
            value = "UNKNOWN"

        result += '"type": "' + type + '", ' + '"value": ' + value

        return result + "}"
