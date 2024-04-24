from collections.optional import Optional
from utils.variant import Variant

from basalt.autograd import Symbol
from basalt.autograd.ops import OP

from .attributes import AttributeVector


@value
struct Node(CollectionElement, Stringable):
    var operator: OP
    var inputs: List[Symbol]
    var output: Symbol
    var attributes: AttributeVector

    fn __init__(
        inout self,
        operator: OP,
        inputs: List[Symbol],
        output: Symbol,
        attributes: AttributeVector = AttributeVector(),
    ):
        self.operator = operator
        self.inputs = inputs
        self.output = output
        self.attributes = attributes

    fn __str__(self) -> String:
        return self.json()

    fn json(self) -> String:
        var s: String = '{"operator": "' + str(self.operator.name) + '", "inputs": ['
        for i in range(len(self.inputs)):
            s += self.inputs[i].json()
            if i < len(self.inputs) - 1:
                s += ", "
        s += '], "outputs": ['
        s += self.output.json() + "]}"
        return s
