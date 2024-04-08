from collections.optional import Optional
from math import min

from basalt import Tensor, TensorShape
from basalt.nn.tensor import max_rank
from basalt.utils.bytes import bytes


alias max_attrs = 10
alias max_attr_char_size = 16
alias max_attr_value_size = 32


@register_passable("trivial")
struct AttributeVector(Sized, Stringable, CollectionElement):
    var _attrs: StaticTuple[Attribute, max_attrs]
    var _size: Int

    fn __init__(inout self, *attributes: Attribute):
        self._attrs = StaticTuple[Attribute, max_attrs]()
        self._size = len(attributes)
        for i in range(self._size):
            self._attrs[i] = attributes[i]

    fn __len__(self) -> Int:
        return self._size

    fn __getitem__(self, index: Int) -> Attribute:
        return self._attrs[index]

    fn __getitem__(self, index: StringLiteral) -> Optional[AttributeValue]:
        for i in range(self._size):
            if self._attrs[i].name == bytes[max_attr_char_size](index):
                return self._attrs[i].value
        return None

    fn __str__(self) -> String:
        var s: String = "["
        for i in range(self._size):
            s += str(self._attrs[i])
            if i < self._size - 1:
                s += ", "
        return s + "]"


@register_passable("trivial")
struct Attribute(Stringable, CollectionElement):
    var name: bytes[max_attr_char_size] # maximum number of chars in the string
    var value: AttributeValue # Variant doesn't seem to be register passable

    fn __init__(inout self, name: String, value: Int):
        self.name = bytes[max_attr_char_size](name)
        self.value = AttributeValue(value)

    fn __init__(inout self, name: String, value: String):
        self.name = bytes[max_attr_char_size](name)
        self.value = AttributeValue(value)

    fn __init__(inout self, name: String, value: TensorShape):
        self.name = bytes[max_attr_char_size](name)
        self.value = AttributeValue(value)

    fn __init__[N: Int](inout self, name: String, value: StaticIntTuple[N]):
        self.name = bytes[max_attr_char_size](name)
        self.value = AttributeValue(value)

    fn __str__(self) -> String:
        return "Attribute(" + str(self.name) + ", " + "..." + ")"


@register_passable("trivial")
struct AttributeValue(CollectionElement):
    """
    Storing attributes as a bytes buffer with given layout to support
    Variant attribute values, while still register passable.
    """

    var _buffer: bytes[max_attr_value_size]
    var _shape: StaticIntTuple[max_rank]

    # AttributeValue: Int
    fn __init__(inout self, value: Int):
        self._buffer = bytes[max_attr_value_size]()
        self._shape = StaticIntTuple[max_rank]()
        self._shape[0]  = value

    fn to_int(self) -> Int:
        return self._shape[0]

    # AttributeValue: String
    fn __init__(inout self, s: String):
        self._buffer = bytes[max_attr_value_size](s)
        self._shape = StaticIntTuple[max_rank]()

    fn to_string(self) -> String:
        return str(self._buffer)

    # AttributeValue: TensorShape
    fn __init__(inout self, shape: TensorShape):
        self._buffer = bytes[max_attr_value_size]()
        self._buffer[0] = shape._rank
        self._shape = shape._shape

    fn to_shape(self) -> TensorShape:
        return TensorShape(rank=self._buffer[0].to_int(), shape=self._shape)

    # AttributeValue: StaticIntTuple (of size N)
    fn __init__[N: Int](inout self, value: StaticIntTuple[N]):
        self._buffer = bytes[max_attr_value_size]()
        self._shape = StaticIntTuple[max_rank]()
        for i in range(N):
            self._shape[i] = value[i]

    fn to_static[N: Int](self) -> StaticIntTuple[N]:
        var result = StaticIntTuple[N]()
        for i in range(N):
            result[i] = self._shape[i]
        return result

    # # AttributeValue: Scalar
    # fn __init__[dtype: DType](inout self, value: SIMD[dtype, 1]):
    #     alias num_bytes = sizeof[dtype]()
