from collections.optional import Optional
from math import min
from tensor import TensorShape

from dainemo.utils.uuid import bytes


alias max_attrs = 10
alias max_attr_char_size = 16
alias max_attr_value_size = 16


@value
@register_passable
struct AttributeVector(Sized, Stringable, CollectionElement):
    var _attrs: StaticTuple[max_attrs, Attribute]
    var _size: Int

    fn __init__(inout self, *attributes: Attribute):
        self._attrs = StaticTuple[max_attrs, Attribute]()
        self._size = min(max_attrs, len(attributes))
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
            if i < self._size - 1: s += ", "
        return s + "]"


@value
@register_passable
struct Attribute(Stringable):
    var name: bytes[max_attr_char_size] # defines the maximum number of characters in the string
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

    fn __str__(self) -> String:
        return "Attribute(" + str(self.name) + ", " + "..." + ")"


@value
@register_passable
struct AttributeValue(CollectionElement):
    """
    Workaround to support Variant attribute values, while still register passable.
    """
    var _buffer: StaticIntTuple[max_attr_value_size]
    var _size: Int

    # AttributeValue: Int
    fn __init__(inout self, value: Int):
        self._buffer = StaticIntTuple[max_attr_value_size]()
        self._size = 1
        self._buffer[0] = value

    fn to_int(self) -> Int:
        return self._buffer[0]

    # AttributeValue: String
    fn __init__(inout self, s: String):
        self._buffer = StaticIntTuple[max_attr_value_size]()
        self._size = len(s)
        for i in range(min(len(s), max_attr_value_size)):
            self._buffer[i] = ord(s[i])

    fn to_string(self) -> String:
        var result: String = ""
        for i in range(self._size):
            result += chr(self._buffer[i])
        return result

    # AttributeValue: TensorShape
    fn __init__(inout self, shape: TensorShape):
        self._buffer = StaticIntTuple[max_attr_value_size]()
        self._size = shape.rank()
        for i in range(shape.rank()):
            self._buffer[i] = shape[i]

    fn to_shape(self) -> TensorShape:
        var tmp = DynamicVector[Int]()
        for i in range(self._size):
            tmp.push_back(self._buffer[i])
        return TensorShape(tmp)