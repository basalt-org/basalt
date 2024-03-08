from collections.optional import Optional
from utils.variant import Variant

from dainemo.autograd import Symbol
from dainemo.autograd.ops import OP
from dainemo.utils.uuid import bytes
from math import min


alias max_attrs = 10
alias max_attr_chars = 32


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

    fn __getitem__(self, index: StringLiteral) -> Optional[Int]:
        for i in range(self._size):
            if self._attrs[i].name == bytes[max_attr_chars](index):
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
    var name: bytes[max_attr_chars] # defines the maximum number of characters in the string
    var value: Int # Variant doesn't seem to be register passable

    fn __init__(inout self, name: String, value: Int):
        self.name = bytes[max_attr_chars](name)
        self.value = value

    fn __str__(self) -> String:
        return "Attribute(" + str(self.name) + ", " + str(self.value) + ")"


@value
struct Node(CollectionElement, Stringable):
    var operator: OP
    var output: Symbol
    var input_1: Symbol
    var input_2: Optional[Symbol]
    var attributes: AttributeVector

    fn __init__(
        inout self,
        operator: OP,
        output: Symbol,
        input_1: Symbol,
        input_2: Optional[Symbol],
    ):
        self.operator = operator
        self.output = output
        self.input_1 = input_1
        self.input_2 = input_2
        self.attributes = AttributeVector()

    fn __init__(
        inout self,
        operator: OP,
        output: Symbol,
        input_1: Symbol,
        input_2: Optional[Symbol],
        attributes: AttributeVector,
    ):
        self.operator = operator
        self.output = output
        self.input_1 = input_1
        self.input_2 = input_2
        self.attributes = attributes

    fn __str__(self) -> String:
        return self.json()

    fn json(self) -> String:
        var s: String = '{"operator": "' + str(self.operator.name) + '", "inputs": ['
        s += self.input_1.json()
        if self.input_2:
            s += ", " + self.input_2.value().json()
        s += '], "outputs": ['
        s += self.output.json() + "]}"
        return s
