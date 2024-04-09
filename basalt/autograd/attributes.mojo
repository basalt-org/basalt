from collections import Optional
from utils import Variant
from math import min

from basalt import Tensor, TensorShape
from basalt.nn.tensor import MAX_RANK
from basalt.utils.bytes import Bytes


alias MAX_ATTRS = 10
alias MAX_NAME_CHARS = 16
alias MAX_DATA_BYTES = 32


@register_passable("trivial")
struct Attribute(Stringable, CollectionElement):
    var name: Bytes[MAX_NAME_CHARS]  # maximum number of chars in the string
    var data: Bytes[MAX_DATA_BYTES]  # maximum number of bytes in the value
    var data_shape: StaticIntTuple[
        MAX_RANK
    ]  # maximum number of dimensions in the value

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

# BELOW CODE COULD BE DRASTICALLY IMPROVED

fn to_bytes[Type: DType](value: Scalar[Type]) -> Bytes[MAX_DATA_BYTES]:
    var result = Bytes[MAX_DATA_BYTES]()

    @parameter
    if Type == DType.bool:
        result[0] = value.cast[DType.uint8]()
    elif Type == DType.uint8 or Type == DType.int8:

        @unroll
        for i in range(8):
            result[i] = value.cast[DType.uint8]()[i]
    elif Type == DType.uint16 or Type == DType.int16 or Type == DType.float16:

        @unroll
        for i in range(8):
            @unroll
            for j in range(2):
                result[i * 2 + j] = value.cast[DType.uint8]()[i * 2 + j]
    elif Type == DType.uint32 or Type == DType.int32 or Type == DType.float32:

        @unroll
        for i in range(8):
            @unroll
            for j in range(4):
                result[i * 4 + j] = value.cast[DType.uint8]()[i * 4 + j]
    elif Type == DType.uint64 or Type == DType.int64 or Type == DType.float64:

        @unroll
        for i in range(8):
            @unroll
            for j in range(8):
                result[i * 8 + j] = value.cast[DType.uint8]()[i * 8 + j]
    else:
        constrained[False, "Invalid DType"]()
    return result


fn from_bytes[Type: DType](value: Bytes[MAX_DATA_BYTES]) -> Scalar[Type]:
    if Type == DType.bool:
        return Scalar[Type](value[0].cast[Type]())
    elif Type == DType.uint8 or Type == DType.int8:
        var result: Scalar[Type] = 0

        @unroll
        for i in range(8):
            result |= value[i].cast[Type]() << i
        return Scalar[Type](result)
    elif Type == DType.uint16 or Type == DType.int16 or Type == DType.float16:
        var result: Scalar[Type] = 0

        @unroll
        for i in range(8):
            result |= value[i * 2].cast[Type]() << (i * 2)
            result |= value[i * 2 + 1].cast[Type]() << (i * 2 + 1)
        return Scalar[Type](result)
    elif Type == DType.uint32 or Type == DType.int32 or Type == DType.float32:
        var result: Scalar[Type] = 0

        @unroll
        for i in range(8):
            result |= value[i * 4].cast[Type]() << (i * 4)
            result |= value[i * 4 + 1].cast[Type]() << (i * 4 + 1)
            result |= value[i * 4 + 2].cast[Type]() << (i * 4 + 2)
            result |= value[i * 4 + 3].cast[Type]() << (i * 4 + 3)
        return Scalar[Type](result)
    elif Type == DType.uint64 or Type == DType.int64 or Type == DType.float64:
        var result: Scalar[Type] = 0

        @unroll
        for i in range(8):
            result |= value[i * 8].cast[Type]() << (i * 8)
            result |= value[i * 8 + 1].cast[Type]() << (i * 8 + 1)
            result |= value[i * 8 + 2].cast[Type]() << (i * 8 + 2)
            result |= value[i * 8 + 3].cast[Type]() << (i * 8 + 3)

            result |= value[i * 8 + 4].cast[Type]() << (i * 8 + 4)
            result |= value[i * 8 + 5].cast[Type]() << (i * 8 + 5)
            result |= value[i * 8 + 6].cast[Type]() << (i * 8 + 6)
            result |= value[i * 8 + 7].cast[Type]() << (i * 8 + 7)

        return Scalar[Type](result)
        
    else:
        constrained[False, "Invalid DType"]()

    return Scalar[Type](0)