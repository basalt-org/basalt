from collections import Optional

from basalt import Tensor, TensorShape
from basalt.nn.tensor import MAX_RANK
from basalt.utils.bytes import Bytes


alias MAX_ATTRS = 10
alias MAX_NAME_CHARS = 16
alias MAX_DATA_BYTES = 32


@register_passable("trivial")
struct Attribute(Stringable, CollectionElement):
    var name: Bytes[MAX_NAME_CHARS]
    var data: Bytes[MAX_DATA_BYTES]
    var data_shape: StaticIntTuple[MAX_RANK]

    @always_inline("nodebug")
    fn __init__(inout self, name: String, value: Int):
        self.name = Bytes[MAX_NAME_CHARS](name)
        self.data = Bytes[MAX_DATA_BYTES]()
        self.data_shape = StaticIntTuple[MAX_RANK]()
        self.data[0] = value
        self.data_shape[0] = 1

    @always_inline("nodebug")
    fn __init__(inout self, name: String, value: String):
        self.name = Bytes[MAX_NAME_CHARS](name)
        self.data = Bytes[MAX_DATA_BYTES](value)
        self.data_shape = StaticIntTuple[MAX_RANK]()
        self.data_shape[0] = len(value)

    @always_inline("nodebug")
    fn __init__(inout self, name: String, value: TensorShape):
        self.name = Bytes[MAX_NAME_CHARS](name)
        self.data = Bytes[MAX_DATA_BYTES]()
        self.data_shape = StaticIntTuple[MAX_RANK]()
        self.data[0] = value.rank()
        for i in range(value.rank()):
            self.data_shape[i] = value._shape[i]

    @always_inline("nodebug")
    fn __init__[N: Int](inout self, name: String, value: StaticIntTuple[N]):
        self.name = Bytes[MAX_NAME_CHARS](name)
        self.data = Bytes[MAX_DATA_BYTES]()
        self.data_shape = StaticIntTuple[MAX_RANK]()
        for i in range(N):
            self.data_shape[i] = value[i]

    @always_inline("nodebug")
    fn __init__(inout self, name: String, value: Scalar):
        alias Type = value.element_type

        self.name = Bytes[MAX_NAME_CHARS](name)
        self.data = to_bytes(value)
        self.data_shape = StaticIntTuple[MAX_RANK]()

    @always_inline("nodebug")
    fn __str__(self) -> String:
        return "Attribute(" + str(self.name) + ", " + "..." + ")"

    @always_inline("nodebug")
    fn to_int(self) -> Int:
        return self.data[0].to_int()

    @always_inline("nodebug")
    fn to_string(self) -> String:
        return str(self.data)

    @always_inline("nodebug")
    fn to_shape(self) -> TensorShape:
        return TensorShape(rank=self.data[0].to_int(), shape=self.data_shape)

    @always_inline("nodebug")
    fn to_static[N: Int](self) -> StaticIntTuple[N]:
        var result = StaticIntTuple[N]()
        for i in range(N):
            result[i] = self.data_shape[i]
        return result

    @always_inline("nodebug")
    fn to_scalar[Type: DType](self) -> Scalar[Type]:
        return from_bytes[Type](self.data)


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
    fn __getitem__(self, index: StringLiteral) -> Optional[Attribute]:
        for i in range(self.size):
            if self.attributes[i].name == Bytes[MAX_NAME_CHARS](index):
                return self.attributes[i]
        return None

    @always_inline("nodebug")
    fn __str__(self) -> String:
        var s: String = "["
        for i in range(self.size):
            s += str(self.attributes[i])
            if i < self.size - 1:
                s += ", "
        return s + "]"


@always_inline("nodebug")
fn to_bytes[Type: DType](value: Scalar[Type]) -> Bytes[MAX_DATA_BYTES]:
    alias TypeSize = Type.sizeof()
    alias Bits = 1 << (TypeSize + 3)

    var result = Bytes[MAX_DATA_BYTES]()

    @parameter
    @always_inline("nodebug")
    fn set_bytes[i: Int]():
        alias Shift = i * 8
        result[i] = (value >> Shift).cast[DType.uint8]()

    unroll[set_bytes, Bits]()

    return result


@always_inline("nodebug")
fn from_bytes[Type: DType](value: Bytes[MAX_DATA_BYTES]) -> Scalar[Type]:
    alias TypeSize = Type.sizeof()
    alias Bits = 1 << (TypeSize + 3)

    var result: Scalar[Type] = 0

    @parameter
    @always_inline("nodebug")
    fn get_bytes[i: Int]():
        alias Shift = i * 8
        result |= (value[i].cast[Type]()) << Shift
    
    unroll[get_bytes, Bits]()

    return result