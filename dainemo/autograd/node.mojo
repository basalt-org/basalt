from collections.optional import Optional
from utils.variant import Variant

from dainemo.autograd import Symbol
from dainemo.autograd.ops import OP


@value
struct AttributeVector:
    var attributes: DynamicVector[Attribute]

    fn __init__(inout self, owned *attributes: Attribute):
        self.attributes = DynamicVector[Attribute]()

        if len(attributes) > 0:
            for a in attributes:
                self.attributes.push_back(a[])

    fn __len__(self) -> Int:
        return len(self.attributes)

    fn __getitem__(self, index: Int) -> Attribute:
        return self.attributes[index]

    fn __getitem__(self, index: StringLiteral) -> Optional[Attribute]:
        for a in self.attributes:
            if a[].name == index:
                return a[]
        return None


@value
struct Attribute(CollectionElement):
    alias T = Variant[
        Int
    ]  # Variant because in the value we should be able to have other stuff like booleans, floats, lists, etc.
    var name: String
    var value: Self.T

    fn __init__(inout self, name: String, value: Int):
        self.name = name
        self.value = Self.T(value)


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
        owned attributes: AttributeVector,
    ):
        # owned is needed for attributes (because it isn't a register_passable value) to be able to deal with lifetimes at compile time by doing copies or by transfering ownership
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
