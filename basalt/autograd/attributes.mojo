from collections.optional import Optional
from math import min

from basalt.utils.bytes import Bytes


alias MAX_ATTRIBUTES = 10
alias MAX_CHAR_SIZE = 16
alias MAX_VALUE_SIZE = 32


@register_passable("trivial")
struct AttributeVector(Sized, Stringable, CollectionElement):
    var _attrs: StaticTuple[Attribute, MAX_ATTRIBUTES]
    var _size: Int

    fn __init__(inout self, *attributes: Attribute):
        self._attrs = StaticTuple[Attribute, MAX_ATTRIBUTES]()
        self._size = len(attributes)
        for i in range(self._size):
            self._attrs[i] = attributes[i]

    fn __len__(self) -> Int:
        return self._size

    fn __getitem__(self, index: Int) -> Attribute:
        return self._attrs[index]

    fn __getitem__(self, index: StringLiteral) -> Optional[AttributeValue]:
        for i in range(self._size):
            if self._attrs[i].name == Bytes[MAX_CHAR_SIZE](index):
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
    var name: Bytes[
        MAX_CHAR_SIZE
    ]  # defines the maximum number of characters in the string
    var value: AttributeValue  # Variant doesn't seem to be register passable

    fn __init__(inout self, name: String, value: Int):
        self.name = Bytes[MAX_CHAR_SIZE](name)
        self.value = AttributeValue(value)

    fn __init__(inout self, name: String, value: String):
        self.name = Bytes[MAX_CHAR_SIZE](name)
        self.value = AttributeValue(value)

    fn __init__(inout self, name: String, value: TensorShape):
        self.name = Bytes[MAX_CHAR_SIZE](name)
        self.value = AttributeValue(value)

    fn __init__[Length: Int](inout self, name: String, value: StaticIntTuple[Length]):
        self.name = Bytes[MAX_CHAR_SIZE](name)
        self.value = AttributeValue(value)

    fn __str__(self) -> String:
        return "Attribute(" + str(self.name) + ", " + "..." + ")"


@register_passable("trivial")
struct AttributeValue(CollectionElement):
    """
    Workaround to support Variant attribute values, while still register passable.
    """

    var _buffer: StaticIntTuple[MAX_VALUE_SIZE]
    var _shape: TensorShape

    # AttributeValue: Int

    fn __init__(inout self, value: Int):
        self._buffer = StaticIntTuple[MAX_VALUE_SIZE]()
        self._shape = 1
        self._buffer[0] = value

    fn to_int(self) -> Int:
        return self._buffer[0]

    # AttributeValue: String

    fn __init__(inout self, s: String):
        self._buffer = StaticIntTuple[MAX_VALUE_SIZE]()
        self._shape = len(s)
        for i in range(min(len(s), MAX_VALUE_SIZE)):
            self._buffer[i] = ord(s[i])

    fn to_string(self) -> String:
        var result: String = ""
        for i in range(self._shape[0]):
            result += chr(self._buffer[i])
        return result

    # AttributeValue: TensorShape

    fn __init__(inout self, shape: TensorShape):
        self._buffer = StaticIntTuple[MAX_VALUE_SIZE]()
        self._shape = shape

    fn to_shape(self) -> TensorShape:
        return self._shape

    # AttributeValue: StaticIntTuple (of size N)

    fn __init__[Length: Int](inout self, value: StaticIntTuple[Length]):
        self._buffer = StaticIntTuple[MAX_VALUE_SIZE]()
        self._shape = Length
        for i in range(Length):
            self._buffer[i] = value[i]

    fn to_static[Length: Int](self) -> StaticIntTuple[Length]:
        var result = StaticIntTuple[Length]()
        for i in range(Length):
            result[i] = self._buffer[i]
        return result
