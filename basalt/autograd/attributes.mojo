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

    fn __init__(inout self, name: String, value: Int):
        self.name = Bytes[MAX_NAME_CHARS](name)
        self.data = Bytes[MAX_DATA_BYTES]()
        self.data_shape = StaticIntTuple[MAX_RANK]()
        self.data[0] = value
        self.data_shape[0] = 1

    fn __init__(inout self, name: String, value: String):
        self.name = Bytes[MAX_NAME_CHARS](name)
        self.data = Bytes[MAX_DATA_BYTES](value)
        self.data_shape = StaticIntTuple[MAX_RANK]()
        self.data_shape[0] = len(value)

    fn __init__(inout self, name: String, value: TensorShape):
        self.name = Bytes[MAX_NAME_CHARS](name)
        self.data = Bytes[MAX_DATA_BYTES]()
        self.data_shape = StaticIntTuple[MAX_RANK]()
        self.data[0] = value.rank()
        for i in range(value.rank()):
            self.data_shape[i] = value._shape[i]

    fn __init__[N: Int](inout self, name: String, value: StaticIntTuple[N]):
        self.name = Bytes[MAX_NAME_CHARS](name)
        self.data = Bytes[MAX_DATA_BYTES]()
        self.data_shape = StaticIntTuple[MAX_RANK]()
        for i in range(N):
            self.data_shape[i] = value[i]

    fn __init__(inout self, name: String, value: Scalar):
        alias Type = value.element_type

        self.name = Bytes[MAX_NAME_CHARS](name)
        self.data = to_bytes(value)
        self.data_shape = StaticIntTuple[MAX_RANK]()

    fn __str__(self) -> String:
        return "Attribute(" + str(self.name) + ", " + "..." + ")"

    fn to_int(self) -> Int:
        return self.data[0].to_int()

    fn to_string(self) -> String:
        return str(self.data)

    fn to_shape(self) -> TensorShape:
        return TensorShape(rank=self.data[0].to_int(), shape=self.data_shape)

    fn to_static[N: Int](self) -> StaticIntTuple[N]:
        var result = StaticIntTuple[N]()
        for i in range(N):
            result[i] = self.data_shape[i]
        return result

    fn to_scalar[Type: DType](self) -> Scalar[Type]:
        return from_bytes[Type](self.data)


@register_passable("trivial")
struct AttributeVector(Sized, Stringable, CollectionElement):
    var _attrs: StaticTuple[Attribute, MAX_ATTRS]
    var _size: Int

    fn __init__(inout self, *attributes: Attribute):
        self._attrs = StaticTuple[Attribute, MAX_ATTRS]()
        self._size = len(attributes)
        for i in range(self._size):
            self._attrs[i] = attributes[i]

    fn __len__(self) -> Int:
        return self._size

    fn __getitem__(self, index: Int) -> Attribute:
        return self._attrs[index]

    fn __getitem__(self, index: StringLiteral) -> Optional[Attribute]:
        for i in range(self._size):
            if self._attrs[i].name == Bytes[MAX_NAME_CHARS](index):
                return self._attrs[i]
        return None

    fn __str__(self) -> String:
        var s: String = "["
        for i in range(self._size):
            s += str(self._attrs[i])
            if i < self._size - 1:
                s += ", "
        return s + "]"


fn to_bytes[Type: DType](value: Scalar[Type]) -> Bytes[MAX_DATA_BYTES]:
    alias TypeSize = Type.sizeof()
    alias Bits = 1 << (TypeSize + 3)
    var result = Bytes[MAX_DATA_BYTES]()

    @unroll
    for i in range(Bits):
        result[i] = (value >> (i * 8)).cast[DType.uint8]()

    return result


fn from_bytes[Type: DType](value: Bytes[MAX_DATA_BYTES]) -> Scalar[Type]:
    alias TypeSize = Type.sizeof()
    alias Bits = 1 << (TypeSize + 3)
    var result: Scalar[Type] = 0

    @unroll
    for i in range(Bits):
        result |= (value[i].cast[Type]()) << (i * 8)

    return result